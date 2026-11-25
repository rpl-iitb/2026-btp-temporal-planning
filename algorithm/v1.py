"""T-iFLAX V1: iFLAX expansion + 4 recovery branches + improvement phase, two-head scorer, bilevel training.

What it does: iFLAX's skeleton made temporal. A GNN scores objects; score-guided expansion grows an object set S
(T1 = 0.6 T, growing per-attempt caps) and POPF plans on the restricted task Pi|S (every plan VAL-checked on the full
task). Parallel recovery when Pi|S fails: Repair / Restart / Rollback / Temporal (adds resources that lower the
relaxed lower bound). Two-head GNN (object score + temporal score), trained bilevel. Leftover budget goes to an
improvement phase (deadline tightening, OPTIC anytime, capacity candidates, re-timing).
Earlier result (pre-final Overcooked-AI benchmark, 60 stress kitchens, 30 s): 9 / 60 solved.
Also the engine of V2-V5 (they override the hooks of ObjectSelectionPlanner).

    python -m algorithm.v1 train --seed 0
"""
import argparse
import concurrent.futures
import os
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .pddlkit import TemporalPlan, validate_plan
from . import results_dir, split_dirs
from .gnn import LabelConfig, Scorer, object_labels, temporal_labels
from .grounding import INF, GroundTask, Task, comp_t
from .planning import Attempt, Budget, Workspace, _full_files, attempt, ensure_hash_seed
from .semantics import Simulator, analyze, bind_plan, resource_predicates, retime

@dataclass
class MethodConfig:
    name: str = "TiFLAX_V1"
    q_max: float = 0.81                  # FLAX: q_max = 0.81, gamma = 0.9
    gamma: float = 0.9
    expansion_frac: float = 0.6          # iFLAX T1 / T
    proofs: bool = True                  # TRPG infeasibility proofs before planner calls
    recovery: tuple = ("repair", "restart", "rollback", "temporal")
    recovery_mode: str = "parallel"      # parallel | sequential
    temporal_topk: int = 2               # resource objects added per Temporal-recovery round
    temporal_rounds: int = 3
    refine: bool = True                  # improvement phase after the first valid plan (leftover budget)
    refine_rounds: int = 4
    refine_min_cap: float = 3.0
    refine_width: int = 3                # capacity candidates per round (one per resource type, each with COMP_T)
    tighten_frac: float = 0.05           # deadline tightening: every goal deadline -> min(d, C - max(1, frac*C))
    refine_optic: bool = True            # also run OPTIC with its cost cap at the current makespan (anytime)
    refine_parallel: int = 4             # concurrent planner runs in one improvement round
    retime: bool = True                  # earliest-start re-timing of every accepted plan
    train_select: str = "makespan"       # train mode label: shortest makespan | "states" (iFLAX: fewest states)
    # Per-attempt cap during expansion. None = iFLAX: an attempt may use all of T1 - t and a timeout
    # ends expansion. A float c = T-iFLAX: attempt k gets c * T * growth^k; a timeout is "unknown"
    # (not evidence that S is insufficient), so expansion continues with more objects and a larger cap.
    attempt_cap_frac: Optional[float] = 0.05
    attempt_cap_growth: float = 1.5
    planner: str = "popf"

    def __post_init__(self):
        self.recovery = tuple(self.recovery)


@dataclass
class RunResult:
    method: str
    instance: str
    budget_s: float
    solved: bool
    makespan: Optional[float] = None
    first_makespan: Optional[float] = None
    first_valid_s: Optional[float] = None
    total_s: float = 0.0
    n_objects: Optional[int] = None
    n_total: int = 0
    branch: Optional[str] = None
    attempts: List[dict] = field(default_factory=list)
    ledger: Dict[str, float] = field(default_factory=dict)
    planner_calls: int = 0
    states: int = 0
    proofs: int = 0
    invalid_on_full: int = 0
    timeouts: int = 0
    notes: List[str] = field(default_factory=list)
    plan_text: Optional[str] = None
    objects: Optional[List[str]] = None

    def to_dict(self):
        return asdict(self)


class _Run:
    """State of one solve() call (shared by recovery threads)."""

    def __init__(self, planner, task, budget_s, mode):
        self.p = planner
        self.cfg = planner.cfg
        self.task = task
        self.budget = Budget(budget_s)
        self.ws = Workspace()
        self.mode = mode
        self.attempts: List[Attempt] = []
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.gt = None
        self.reward = {}
        self.scores = {}
        self.notes = []
        self._rough = None
        self._rough_lock = threading.Lock()

    # ---- helpers ------------------------------------------------------------------------
    def closure(self, S):
        with self.budget.charge("closure"):
            return comp_t(self.gt, S)

    exp_closure = closure

    def plan(self, S, stage, cap=None, cancel=None, gt=None, planner=None, **kw):
        gt = gt if gt is not None else self.gt
        a = attempt(self.task, gt if self.cfg.proofs else None, S, self.budget, stage, self.ws, cap=cap,
                    cancel=cancel if cancel is not None else self.cancel, check_proof=self.cfg.proofs,
                    planner=planner or self.cfg.planner, **kw)
        with self.lock:
            self.attempts.append(a)
        return a

    def rough_objects(self):
        """Objects of a rough plan of the full task (Repair / Restart guidance)."""
        with self._rough_lock:
            if self._rough is not None:
                return self._rough or None          # False = computed and failed (cached)
            with self.budget.charge("rough_plan"):
                self._rough = self.gt.relaxed_plan_objects(self.gt.full) or False
            return self._rough or None

    def by_reward(self, objs):
        return sorted(objs, key=lambda o: (-self.reward.get(o, 0.0), o))

    # ---- score-guided iterative expansion (iFLAX Alg. 1 lines 1-11) -----------------------
    def iterative_expansion(self, base, end_elapsed, stage, cancel=None):
        """-> (valid attempt or None, S_N, S_{N-1}, reason)"""
        allobj = set(self.task.objects)
        q = self.cfg.q_max * self.p.scale
        S = self.exp_closure(base)
        prev = set(S)
        tried = set()
        k = 0
        while True:
            if self.budget.elapsed() >= end_elapsed or (cancel is not None and cancel.is_set()):
                return None, S, prev, "time"
            cand = self.exp_closure(S | {o for o in allobj if self.reward.get(o, 0.0) >= q})
            while frozenset(cand) in tried and len(cand) < len(allobj):
                q *= self.cfg.gamma
                cand = self.exp_closure(S | {o for o in allobj if self.reward.get(o, 0.0) >= q})
            if frozenset(cand) in tried:
                return None, S, prev, "exhausted"
            tried.add(frozenset(cand))
            prev, S = S, cand
            cap = max(0.0, end_elapsed - self.budget.elapsed())
            if self.cfg.attempt_cap_frac:
                cap = min(cap, max(1.0, self.cfg.attempt_cap_frac * self.budget.total) * self.cfg.attempt_cap_growth ** k)
            k += 1
            a = self.plan(S, stage, cap=cap, cancel=cancel)
            if a.outcome == "valid":
                return a, S, prev, "valid"
            if a.outcome in ("cancelled", "budget_exhausted"):
                return None, S, prev, a.outcome
            if a.outcome == "timeout" and (not self.cfg.attempt_cap_frac or len(S) >= len(allobj)):
                return None, S, prev, "stuck"
            if len(S) >= len(allobj):
                return None, S, prev, "exhausted"
            q *= self.cfg.gamma

    # ---- recovery branches ----------------------------------------------------------------
    def repair(self, S_N, S_prev, S_0, cancel):
        rough = self.rough_objects()
        if rough is None:
            return None
        return self._valid(self.plan(self.closure(S_N | rough), "repair", cancel=cancel))

    def restart(self, S_N, S_prev, S_0, cancel):
        rough = self.rough_objects()
        if rough is None:
            return None
        base = self.closure(S_0 | rough)
        a, _, _, _ = self.iterative_expansion(base, self.budget.total, "restart", cancel=cancel)
        return a

    def rollback(self, S_N, S_prev, S_0, cancel):
        S = set(S_prev)
        for o in self.by_reward(set(self.task.objects) - S):
            if cancel.is_set() or self.budget.expired():
                return None
            if o in S:
                continue
            S = self.exp_closure(S | {o})
            a = self.plan(S, "rollback", cancel=cancel)
            if a.outcome == "valid":
                return a
            if a.outcome in ("timeout", "cancelled", "budget_exhausted"):
                return None
        return None

    def temporal(self, S_N, S_prev, S_0, cancel):
        """Temporal recovery: add the resource objects (cooks, pots, ...) that
        most lower the TRPG lower bound / clear a deadline proof, ranked by
        temporal score; replan; repeat while time remains."""
        S = set(S_N)
        for _ in range(self.cfg.temporal_rounds):
            if cancel.is_set() or self.budget.expired():
                return None
            add = self.p.capacity_candidates(self, S)[: self.cfg.temporal_topk]
            if not add:
                return None
            S = self.closure(S | set(add))
            a = self.plan(S, "temporal", cancel=cancel)
            if a.outcome == "valid":
                return a
            if a.outcome in ("cancelled", "budget_exhausted"):
                return None
        return None

    def _valid(self, a):
        return a if a is not None and a.outcome == "valid" else None


class ObjectSelectionPlanner:
    def __init__(self, cfg: MethodConfig, scorer=None):
        self.cfg = cfg
        self.scorer = scorer
        self.scale = scorer.reward_scale if scorer is not None else 1.0
        self.last_run = None

    # ---- entry point ------------------------------------------------------------------------
    def solve(self, task, budget_s, mode="deploy", gt=None, graph=None) -> RunResult:
        cfg = self.cfg
        R = _Run(self, task, budget_s, mode)
        self.last_run = R
        try:
            with R.budget.charge("ground"):
                R.gt = gt if gt is not None else GroundTask(task)
                comp_t(R.gt, set())          # builds the closure index once (thread safety)
                bind_plan(R.gt, _EmptyPlan())
            with R.budget.charge("graph"):
                if graph is None and hasattr(self.scorer, "graph"):
                    graph = self.scorer.graph(task, R.gt)
            with R.budget.charge("gnn"):
                R.scores = self.scorer.scores(task, R.gt, graph=graph)
                R.reward = self.rewards(R)
            best, branch = self.first_plan(R)
            if best is not None and cfg.refine and mode == "deploy":
                best = self._refine(R, best)
            return self._result(R, best, branch)
        finally:
            R.cancel.set()
            R.ws.close()

    # ---- hooks (overridden by later versions) ----------------------------------------------------
    def rewards(self, R):
        """reward(o) thresholded by the expansion (V1: object_score + temporal_score)."""
        return {o: self.scorer.reward(p) for o, p in R.scores.items()}

    def first_plan(self, R):
        """-> (first valid attempt or None, branch). V1: iFLAX expansion + recovery."""
        return self._iflax(R)

    def temporal_score(self, R):
        """Rank of a capacity candidate after its LB gain (V1: the temporal head)."""
        return lambda o: R.scores.get(o, [0.0, 0.0])[-1]

    def refine_start(self, R, best):
        """Called once before the improvement rounds; the value is passed to refine_end."""
        return None

    def refine_end(self, R, best, state):
        return best

    def front_sets(self, R, S):
        """Object sets tried first in an improvement round, also run with plain POPF (V3.1)."""
        return []

    def extra_sets(self, R, S):
        """Object sets added after the capacity candidates of an improvement round (V2.1)."""
        return []

    def tighten_fraction(self, improved_last):
        return self.cfg.tighten_frac

    # ---- iFLAX skeleton (Algorithm 1) -------------------------------------------------------------------------
    def _iflax(self, R):
        cfg, task = self.cfg, R.task
        S_0 = R.closure(set(task.goal_objects))
        T1 = R.budget.elapsed() + (R.budget.total - R.budget.elapsed()) * cfg.expansion_frac
        a, S_N, S_prev, why = R.iterative_expansion(S_0, T1, "expansion")
        if a is not None:
            return a, "expansion"
        R.notes.append("expansion ended: " + why)
        if not cfg.recovery or R.budget.expired():
            return None, None
        branches = {b: getattr(R, b) for b in cfg.recovery}
        if R.mode == "train" or cfg.recovery_mode == "parallel":
            return self._parallel(R, branches, S_N, S_prev, S_0)
        for name, fn in branches.items():
            res = fn(S_N, S_prev, S_0, R.cancel)
            if res is not None:
                return res, name
        return None, None

    def _parallel(self, R, branches, S_N, S_prev, S_0):
        found = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(branches)) as pool:
            futs = {pool.submit(fn, S_N, S_prev, S_0, R.cancel): name for name, fn in branches.items()}
            for fut in concurrent.futures.as_completed(futs):
                name = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:      # a crashing branch must not take the others down
                    R.notes.append("{} crashed: {!r}".format(name, e))
                    continue
                if res is not None:
                    found[name] = res
                    if R.mode == "deploy":
                        R.cancel.set()
        if not found:
            return None, None
        if R.mode == "deploy":
            name = min(found, key=lambda n: found[n].t_end)
        elif self.cfg.train_select == "makespan":
            name = min(found, key=lambda n: (found[n].makespan, found[n].states or INF))
        else:
            name = min(found, key=lambda n: (found[n].states if found[n].states is not None else INF))
        return found[name], name

    # ---- Temporal recovery helpers / refinement ------------------------------------------------------
    def capacity_candidates(self, R, S):
        """Resource objects outside S, best first: by how much adding them (with
        COMP_T) lowers LB(S) / clears a proof, then by temporal score."""
        gt, task = R.gt, R.task
        rtypes = set(resource_predicates(task.spec).values())
        rel = getattr(gt, "_rel_by_obj", {})
        cands = [o for o in task.objects if o not in S and o in rel and any(task.is_a(o, t) for t in rtypes)]
        if not cands:
            return []
        lb0 = gt.trpg(S)
        lb0 = lb0.lb if lb0.solvable_relaxed else INF
        tscore = self.temporal_score(R)
        per_type = {}
        for o in sorted(cands, key=lambda o: (-tscore(o), o)):
            per_type.setdefault(task.type_of(o), []).append(o)
        cands = [o for os_ in per_type.values() for o in os_[:8]]      # cap per type, not overall
        ranked = []
        for o in cands:
            r = gt.trpg(comp_t(gt, S | {o}))
            lb = r.lb if r.solvable_relaxed else INF
            gain = (lb0 - lb) if lb0 < INF else (INF if lb < INF else 0.0)
            ranked.append((-(gain if gain > 1e-9 else 0.0), -tscore(o), o))
        ranked.sort()
        return [o for _, _, o in ranked]

    def _refine(self, R, best):
        """Improvement phase with the leftover budget (the first valid plan is already recorded).

        Every round, for the current best plan (makespan C, objects S):
          * candidate object sets: S, and S + each of the best `refine_width` capacity
            candidates (resource types on the critical path first, then LB gain / temporal
            score), closed with COMP_T;
          * on each set, in parallel: POPF on a copy whose goal deadlines are tightened to
            C - delta (POPF returns its first plan, so the bound is what forces it to use
            the extra capacity; the TRPG proof can reject a tightened set without a call),
            and OPTIC with its cost cap at C (metric = total-time, so cost = makespan);
          * every result is VAL-checked on the original task, re-timed, and kept only if
            shorter. A round without improvement ends the phase.
        """
        cfg, task, gt = self.cfg, R.task, R.gt
        R.cancel = threading.Event()          # the recovery winner set the old one
        best = self._retime(R, best)
        state = self.refine_start(R, best)
        tried = set()
        improved_last = True
        for rnd in range(cfg.refine_rounds):
            rem = R.budget.remaining()
            if rem < cfg.refine_min_cap:
                break
            C = best.makespan
            S = frozenset(best.objects)
            sets = [S] if not any(k[0] == S and k[2] == C for k in tried) else []
            front = self.front_sets(R, set(S))
            for B in front:
                if B not in sets and not any(k[0] == B and k[2] == C for k in tried):
                    sets.insert(0, B)
            for o in self._refine_candidates(R, best):
                if len(sets) >= cfg.refine_width + 1:
                    break
                S2 = frozenset(R.closure(set(S) | {o}))
                if S2 not in sets and not any(k[0] == S2 and k[2] == C for k in tried):
                    sets.append(S2)
            for S2 in self.extra_sets(R, S):
                if S2 not in sets and not any(k[0] == S2 and k[2] == C for k in tried):
                    sets.append(S2)
            delta = max(1.0, self.tighten_fraction(improved_last) * C)
            tight = tightened_task(task, C - delta)
            gt_t = gt.with_task(tight) if tight is not None else None
            jobs = []
            bulk = set(front)
            for S2 in sets:
                if S2 in bulk and (S2, "plain", C) not in tried:
                    # plain POPF on a front set (V3.1: what B1 does on the full task, on a smaller set)
                    jobs.append((S2, "plain", dict()))
                if tight is not None and (S2, "tighten", C) not in tried:
                    jobs.append((S2, "tighten", dict(plan_task=tight, gt=gt_t)))
                if cfg.refine_optic and (S2, "optic", C) not in tried:
                    jobs.append((S2, "optic", dict(planner="optic-clp", planner_args=("-n{:.3f}".format(C - 0.001),),
                                                   anytime=True)))
            if not jobs:
                break
            for S2, kind, _ in jobs:
                tried.add((S2, kind, C))
            cap = max(cfg.refine_min_cap, rem / max(1, cfg.refine_rounds - rnd))
            with concurrent.futures.ThreadPoolExecutor(min(cfg.refine_parallel, len(jobs))) as pool:
                outs = list(pool.map(lambda j: R.plan(j[0], "improve_" + j[1], cap=cap, **j[2]), jobs))
            better = [a for a in outs if a.outcome == "valid" and a.makespan < C - 1e-6]
            improved_last = bool(better)
            if better:            # no improvement: the next round tries the remaining candidates
                best = self._retime(R, min(better, key=lambda a: a.makespan))
        return self.refine_end(R, best, state)

    def _refine_candidates(self, R, best):
        """Resource objects outside the plan's set, round-robin over resource types so every
        kind of capacity gets tried (the TRPG bound ignores contention, so a second machine
        often shows no LB gain). Type order: types waiting on the critical path, then types
        with fewer objects (machines and agents before floor tiles); within a type, by LB
        gain then temporal score (capacity_candidates)."""
        task, gt = R.task, R.gt
        with R.budget.charge("refine_analysis"):
            steps, durs, unknown = bind_plan(gt, best.plan)
            crit = {}
            if not unknown:
                an = analyze(task, Simulator(task).run(steps, durs))
                crit = an.critical_resource_types
            ranked = self.capacity_candidates(R, set(best.objects))
        by_type = {}
        for o in ranked:
            by_type.setdefault(task.type_of(o), []).append(o)
        count = {t: len(task.of_type(t)) for t in by_type}
        order = sorted(by_type, key=lambda t: (-crit.get(t, 0.0), count[t], t))
        out, i = [], 0
        while any(i < len(by_type[t]) for t in order):
            out += [by_type[t][i] for t in order if i < len(by_type[t])]
            i += 1
        return out

    def _retime(self, R, a):
        if not self.cfg.retime or a is None or a.plan is None:
            return a
        with R.budget.charge("retime"):
            rt = retime(R.task, R.gt, a.plan)
            if rt is None or rt.makespan >= a.makespan - 1e-6:
                return a
            dom, prob = _full_files(R.task, R.ws)
            if not validate_plan(dom, prob, rt).valid:
                R.notes.append("retimed plan rejected by VAL")
                return a
        b = Attempt("retime", a.n_objects, "valid", makespan=rt.makespan, plan=rt, t_end=R.budget.elapsed(),
                    objects=a.objects)
        with R.lock:
            R.attempts.append(b)
        return b

    # ---- result ---------------------------------------------------------------------------------------
    def _result(self, R, best, branch):
        task, B = R.task, R.budget
        valid = [a for a in R.attempts if a.outcome == "valid"]
        first = min(valid, key=lambda a: a.t_end) if valid else None
        res = RunResult(
            method=self.cfg.name, instance=task.name, budget_s=B.total, solved=best is not None,
            makespan=best.makespan if best else None, first_makespan=first.makespan if first else None,
            first_valid_s=first.t_end if first else None, total_s=B.elapsed(),
            n_objects=len(best.objects) if best and best.objects else None, n_total=len(task.objects),
            branch=branch if best is not None else None, attempts=[a.row() for a in R.attempts],
            ledger=B.report()["ledger_s"], planner_calls=sum(1 for a in R.attempts if a.planner_s > 0),
            states=sum(a.states or 0 for a in R.attempts),
            proofs=sum(1 for a in R.attempts if a.outcome == "proved_infeasible"),
            invalid_on_full=sum(1 for a in R.attempts if a.outcome == "invalid_on_full"),
            timeouts=sum(1 for a in R.attempts if a.outcome == "timeout"), notes=R.notes,
            plan_text=best.plan.to_text() if best else None,
            objects=sorted(best.objects) if best and best.objects else None)
        return res


def tightened_task(task, bound):
    """Copy of task whose goal deadlines are min(deadline, bound); None if the domain
    has no deadline encoding or the bound is not positive."""
    import dataclasses
    enc = task.spec.deadline_encoding
    if not enc or bound <= 0:
        return None
    facts = set(map(tuple, task.problem.facts))
    goals = []
    for g in task.problem.goals:
        window = (enc["window_predicate"],) + tuple(g.fact[1:])
        if g.fact[0] == enc["goal_predicate"] and window in facts:
            d = bound if g.deadline is None else min(g.deadline, bound)
            goals.append(dataclasses.replace(g, deadline=d))
        else:
            goals.append(g)
    q = dataclasses.replace(task.problem, goals=goals)
    return Task(task.spec, q, _domain_file=task._domain_file)


class _EmptyPlan:
    actions = []


# ==================== building a configured planner (evaluate.py) ==================================
def model_prefix(name, seed):
    """results/<version folder>/<name>_seed<k>: v1_seed0, v2.1_seed0, ..."""
    return os.path.join(results_dir(name.split(".")[0]), "{}_seed{}".format(name, seed))


def build(key, spec, seed=0, **over):
    """V1: the trained two-head network (results/v1/v1_seed<k>)."""
    assert key == "V1", key
    return ObjectSelectionPlanner(MethodConfig(**over), Scorer.load(model_prefix("v1", seed), spec))


# ==================== bilevel training ==============================================================
class Example:
    def __init__(self, d, task, need_gt):
        self.dir, self.task = d, task
        self.gt = GroundTask(task) if need_gt else None
        self.graph = None


def load_split(name):
    return [(d, Task.load(d)) for d in split_dirs(name)]


def bce_masked(logits, y, m, pos_weight=None):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    return (loss * m).sum() / m.sum().clamp(min=1.0)


class BilevelTrainer:
    """V1's training; V2 / V2.1 override the scorer, labels, loss and the final step."""
    name = "v1"

    def add_args(self, ap):
        pass

    def config(self, args):
        return MethodConfig()

    def planner(self, cfg, scorer):
        return ObjectSelectionPlanner(cfg, scorer)

    def new_scorer(self, spec, args):
        return Scorer(spec, heads=2, temporal_features=True, hidden=args.hidden, steps=args.steps, seed=args.seed)

    def labels(self, scorer, ex, plan, S, scores, label_cfg, ws):
        objs = ex.graph.objects
        y_obj = object_labels(ex.task, objs, plan)
        m_obj = np.ones_like(y_obj)
        if scorer.heads == 1:
            return y_obj, m_obj, {}
        y_t, m_t, info = temporal_labels(ex.task, ex.gt, objs, plan, S, scores, label_cfg, ws)
        return np.concatenate([y_obj, y_t], 1), np.concatenate([m_obj, m_t], 1), info

    def loss(self, scorer, z, y, m, args, pos_weight):
        loss = bce_masked(z[:, :1], y[:, :1], m[:, :1], pos_weight)
        if scorer.heads > 1:
            loss = loss + args.lam_t * bce_masked(z[:, 1:2], y[:, 1:2], m[:, 1:2])
        return loss

    def finish(self, scorer, calib, log):
        """After training: fit one temperature per head on the validation labels."""
        temps = scorer.calibrate([(g, y, m) for g, y, m, _ in calib])
        log("temperatures:", [round(t, 3) for t in temps])

    # ---- the loop ----------------------------------------------------------------------------------
    def step(self, scorer, opt, ex, y, m, args, pos_weight):
        scorer.model.train()
        z = scorer.logits(ex.graph)
        loss = self.loss(scorer, z, torch.from_numpy(y), torch.from_numpy(m), args, pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item())

    def plan_for_labels(self, planner, ex, budget, label_cfg, scorer):
        """Lower level of the bilevel loop: plan with current scores (train mode),
        then turn the plan into pseudo-labels."""
        res = planner.solve(ex.task, budget, mode="train", gt=ex.gt, graph=ex.graph)
        if not res.solved:
            return None
        plan = TemporalPlan.parse(res.plan_text)
        scores = planner.last_run.scores
        ws = Workspace()
        try:
            y, m, info = self.labels(scorer, ex, plan, set(res.objects), scores, label_cfg, ws)
        finally:
            ws.close()
        return y, m, info, res

    def run(self, argv=None):
        ap = argparse.ArgumentParser(description="bilevel training of T-iFLAX " + self.name.upper())
        ap.add_argument("--train", default="train")
        ap.add_argument("--val", default="val")
        ap.add_argument("--epochs", type=int, default=6)
        ap.add_argument("--budget", type=float, default=15.0, help="planning budget per task in the bilevel loop")
        ap.add_argument("--parallel", type=int, default=3)
        ap.add_argument("--lr", type=float, default=1e-3)
        ap.add_argument("--hidden", type=int, default=16)
        ap.add_argument("--steps", type=int, default=3)
        ap.add_argument("--lam_t", type=float, default=1.0)
        ap.add_argument("--pos_weight", type=float, default=1.0)
        ap.add_argument("--cf_k", type=int, default=4)
        ap.add_argument("--cf_cap", type=float, default=8.0)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--max_tasks", type=int, default=0)
        ap.add_argument("--out", default=None)
        self.add_args(ap)
        args = ap.parse_args(argv)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        out = args.out or model_prefix(self.name, args.seed)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        logf = open(out + ".log", "w")

        def log(*a):
            s = " ".join(str(x) for x in a)
            print(s, flush=True)
            logf.write(s + "\n")
            logf.flush()

        cfg = self.config(args)
        t0 = time.time()
        train = load_split(args.train)
        val = load_split(args.val)
        if args.max_tasks:
            train = train[: args.max_tasks]
        scorer = self.new_scorer(train[0][1].spec, args)
        exs = [Example(d, t, True) for d, t in train]
        vexs = [Example(d, t, True) for d, t in val]
        for ex in exs + vexs:
            ex.graph = scorer.graph(ex.task, ex.gt)
        log("[{}] {} train / {} val tasks, graphs built in {:.1f}s".format(self.name, len(exs), len(vexs),
                                                                           time.time() - t0))
        opt = torch.optim.Adam(scorer.model.parameters(), lr=args.lr)
        pw = torch.tensor([args.pos_weight])
        label_cfg = LabelConfig(counterfactual_k=args.cf_k, counterfactual_cap=args.cf_cap)

        def safe(planner, ex):
            try:
                return self.plan_for_labels(planner, ex, args.budget, label_cfg, scorer)
            except Exception as e:     # one bad task must not stop training; it is reported
                log("    ! {} failed: {!r}".format(ex.task.name, e))
                return None

        for epoch in range(args.epochs):
            te = time.time()
            order = list(range(len(exs)))
            random.shuffle(order)
            solved, losses, osr, infos = 0, [], [], []
            for b in range(0, len(order), args.parallel):
                batch = [exs[i] for i in order[b: b + args.parallel]]
                planners = [self.planner(cfg, scorer) for _ in batch]
                with concurrent.futures.ThreadPoolExecutor(len(batch)) as pool:
                    outs = list(pool.map(lambda pe: safe(*pe), zip(planners, batch)))
                for ex, o in zip(batch, outs):
                    if o is None:
                        continue
                    y, m, info, res = o
                    solved += 1
                    osr.append(res.n_objects / res.n_total)
                    losses.append(self.step(scorer, opt, ex, y, m, args, pw))
                    if info:
                        infos.append(info)
            rec = {"epoch": epoch, "solved": solved, "total": len(exs),
                   "loss": float(np.mean(losses)) if losses else None,
                   "osr": float(np.mean(osr)) if osr else None, "seconds": round(time.time() - te, 1)}
            if infos:
                rec["mean_parallelism"] = float(np.mean([i["parallelism"] for i in infos]))
                rec["mean_makespan"] = float(np.mean([i["makespan"] for i in infos]))
            scorer.history.append(rec)
            log("  epoch {:2d}: solved {}/{}  loss {}  kept {}  ({:.0f}s){}".format(
                epoch, solved, len(exs), "{:.4f}".format(rec["loss"]) if losses else "-",
                "{:.2f}".format(rec["osr"]) if osr else "-", rec["seconds"],
                "  ms {:.1f}".format(rec["mean_makespan"]) if infos else ""))
        # validation labels, produced by the trained planner
        calib = []
        planners = [self.planner(cfg, scorer) for _ in vexs]
        with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
            outs = list(pool.map(lambda pe: safe(*pe), zip(planners, vexs)))
        for ex, o in zip(vexs, outs):
            if o is not None:
                calib.append((ex.graph, o[0], o[1], ex.task))
        log("validation labels: {}/{}".format(len(calib), len(vexs)))
        self.finish(scorer, calib, log)
        scorer.save(out, extra={"method": self.name, "args": vars(args), "train_split": args.train,
                                "val_split": args.val, "train_seconds": round(time.time() - t0, 1)})
        log("saved", out, "in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    import sys
    ensure_hash_seed()
    if sys.argv[1:2] != ["train"]:
        raise SystemExit("usage: python -m algorithm.v1 train [--seed 0 --epochs 6 --budget 15 ...]")
    BilevelTrainer().run(sys.argv[2:])
