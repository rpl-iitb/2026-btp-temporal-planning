"""T-iFLAX V5 (set scorer on POPF outcomes) and V5a (3-head importance / time GNN).

What it does: a GNN on Pi|S predicts P(POPF solves S) and the makespan, trained on POPF trials of candidate sets; it
ranks V4.2's candidates and POPF runs on one at a time. V5a: a 3-head GNN (importance, T_min, T_max) with temporal
filtration and a GNN seed inside V4.2.
Earlier result (pre-final Overcooked-AI benchmark, seeds-only protocol): V5 39 / 60 (relaxed-bound order 42).

    python -m algorithm.v5 labels|train|seedtest|v5a-labels|v5a-train|v5a-seedtest [--split ...]
"""
import argparse
import concurrent.futures
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass

from .pddlkit import TemporalPlan
from . import PDDL, results_dir, split_dirs
from .gnn import Scorer, temporal_lb, to_tensors
from .grounding import GroundTask, Task, comp_t
from .planning import Budget, Workspace, attempt, ensure_hash_seed
from .v3 import agent_seeds
from .v4 import V4Config, V4Planner, seed_rounds

OUT = results_dir("v5")
TIME_W = 0.2                               # weight of the time terms (their Huber values are ~10x the BCE)
CAPS = (10.0, 7.0, 5.0, 4.0, 4.0)          # sequential POPF caps by rank within a 30 s budget (quick test)


def candidates(gt):
    """Every candidate set V4.2's generator can make: [(lb, label, S)], TRPG-proof-free, distinct."""
    singles = agent_seeds(gt, 1000, kind="mix")
    _, second = seed_rounds(gt, 4)
    out, seen = [], set()
    for lb, c, S in singles + [x for x in second if "+" in x[1]]:
        if S not in seen and gt.proof(S) is None:
            seen.add(S); out.append((lb, c, S))
    return out


def v42_order(cands):
    """V4.2's own ranking: TRPG lower bound, then size."""
    return sorted(cands, key=lambda x: (x[0], len(x[2]), x[1]))


# ---------------------------------------------------------------- labels from POPF
def completion(plan, serial=False):
    c, t = {}, 0.0
    for a in plan.actions:
        end = (t + a.duration) if serial else a.end
        t = end if serial else t
        for o in a.args:
            c[o] = max(c.get(o, 0.0), end)
    return c


def label_instance(d, cap=10.0):
    task = Task.load(d)
    gt = GroundTask(task)
    sets = {"%s_%d" % (c, i): S for i, (_, c, S) in enumerate(candidates(gt))}
    cert = os.path.join(d, "certificate.plan")
    if os.path.exists(cert):
        used = TemporalPlan.parse(open(cert).read()).objects_used() & set(task.objects)
        sets["cert"] = frozenset(comp_t(gt, used))
    sets["full"] = frozenset(task.objects)

    def one(item):
        ws = Workspace()
        try:
            return item[0], item[1], attempt(task, gt, item[1], Budget(cap), "label", ws, check_proof=False)
        finally:
            ws.close()
    with concurrent.futures.ThreadPoolExecutor(min(12, len(sets))) as ex:
        res = list(ex.map(one, sorted(sets.items(), key=lambda x: x[0])))
    rows = []
    for name, S, a in res:
        r = {"set": name, "objects": sorted(S), "solve": 1.0 if a.outcome == "valid" else 0.0, "outcome": a.outcome}
        if a.outcome == "valid":
            r["makespan"] = a.makespan
            r["tmin"], r["tmax"] = completion(a.plan), completion(a.plan, serial=True)
        rows.append(r)
    return {"instance": os.path.basename(d), "sets": rows}


def make_labels(split, cap, parallel):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        rows = list(ex.map(lambda d: label_instance(d, cap), split_dirs(split)))
    out = os.path.join(OUT, "sets_%s.json" % split)
    json.dump(rows, open(out, "w"))
    n = sum(len(r["sets"]) for r in rows)
    pos = sum(s["solve"] for r in rows for s in r["sets"])
    print("%s: %d kitchens, %d candidate sets, %d solved by POPF (%.0f %%), %.0f s -> %s"
          % (split, len(rows), n, pos, 100 * pos / max(1, n), time.time() - t0, out))


# ---------------------------------------------------------------- the set scorer
class SetScorer(Scorer):
    """GNN on the graph of Pi|S; heads: 0 imp, 1 solve (mean + max pooled), 2 T_min, 3 gap (times / LB(Pi|S))."""

    def __init__(self, spec, hidden=16, steps=3, seed=0):
        super().__init__(spec, heads=4, temporal_features=True, hidden=hidden, steps=steps, seed=seed)
        self.combine = "v5"

    def outputs(self, g):
        z = self.model(*to_tensors(g))
        tmin = nn.functional.softplus(z[:, 2])
        return z[:, 0], z[:, 1].mean() + z[:, 1].max(), tmin, tmin + nn.functional.softplus(z[:, 3])

    def score_set(self, task, S):
        """-> (score, P(solve), {order: (T_min, T_max, deadline)}), times in seconds."""
        sub = task.restrict(S)
        g = self.graph(sub, GroundTask(sub))
        self.model.eval()
        with torch.no_grad():
            _, zs, tmin, tmax = self.outputs(g)
        p = float(torch.sigmoid(zs))
        idx = {o: i for i, o in enumerate(g.objects)}
        on_time, orders = 1.0, {}
        for goal in task.problem.goals:
            if goal.deadline is None:
                continue
            o = goal.fact[1]
            a, b = float(tmin[idx[o]]) * g.lb, float(tmax[idx[o]]) * g.lb
            orders[o] = (a, b, goal.deadline)
            if b - a < 1e-6:
                on_time *= 1.0 if a <= goal.deadline else 0.0
            else:
                on_time *= min(1.0, max(0.0, (goal.deadline - a) / (b - a)))
        return p * on_time, p, orders

    def rank(self, task, cands):
        """Best first: P(solve) (its labels already include the deadlines), then predicted makespan
        (latest predicted order completion), then TRPG bound. The per-order on-time product is not used:
        orders of one recipe are interchangeable, so a predicted time does not belong to a fixed deadline."""
        scored = [(self.score_set(task, S), lb, c, S) for lb, c, S in cands]
        scored.sort(key=lambda x: (-round(x[0][1], 2), max((v[0] for v in x[0][2].values()), default=0.0), x[1], len(x[3]), x[2]))
        return [(lb, c, S, sc) for sc, lb, c, S in scored]

    def ranked_seeds(self, task, gt):
        """Planner hook: all candidate sets, best GNN score first -> [(lb, label, S)]."""
        return [(lb, c, S) for lb, c, S, _ in self.rank(task, candidates(gt))]

    def scores(self, task, gt=None, graph=None):
        """Object scores are not used by V5's seeds; uniform, so the fallback and improvement behave as V4.2's."""
        return {o: [0.5, 0.5] for o in task.objects}

    def reward(self, p):
        return sum(p)

    @property
    def reward_scale(self):
        return 2.0

    @classmethod
    def load(cls, prefix, spec):
        meta = json.load(open(prefix + ".json"))
        mc = meta["model"]
        s = cls(spec, hidden=mc["hidden"], steps=mc["steps"])
        if s.vocab.signature() != meta["vocab"]:
            raise ValueError("vocabulary mismatch for " + prefix)
        s.model.load_state_dict(torch.load(prefix + ".pt", weights_only=True))
        s.history = meta.get("history", [])
        return s


# ---------------------------------------------------------------- training
def examples(split, scorer):
    out = []
    for r in json.load(open(os.path.join(OUT, "sets_%s.json" % split))):
        task = Task.load(os.path.join(PDDL, split, r["instance"]))
        for s in r["sets"]:
            sub = task.restrict(s["objects"])
            g = scorer.graph(sub, GroundTask(sub))
            n = len(g.objects)
            y = np.zeros((n, 3), dtype=np.float32)
            m = np.zeros((n, 3), dtype=np.float32)
            if s["solve"]:
                for i, o in enumerate(g.objects):
                    y[i, 0], m[i, 0] = (1.0 if o in s["tmin"] else 0.0), 1.0
                    if o in s["tmin"]:
                        y[i, 1], y[i, 2] = s["tmin"][o] / g.lb, s["tmax"][o] / g.lb
                        m[i, 1] = m[i, 2] = 1.0
            out.append((r["instance"], s["set"], g, torch.tensor(s["solve"]), torch.from_numpy(y), torch.from_numpy(m)))
    return out


def loss_of(sc, g, ys, y, m):
    zi, zs, tmin, tmax = sc.outputs(g)
    l = nn.functional.binary_cross_entropy_with_logits(zs, ys)
    if ys > 0:
        hub = lambda a, b, w: (nn.functional.huber_loss(a, b, reduction="none") * w).sum() / w.sum().clamp(min=1.0)  # noqa: E731
        l = l + nn.functional.binary_cross_entropy_with_logits(zi, y[:, 0]) + TIME_W * (hub(tmin, y[:, 1], m[:, 1]) + hub(tmax, y[:, 2], m[:, 2]))
    return l, zs, tmin, tmax


def evaluate_split(sc, exs):
    sc.model.eval()
    loss, acc, et, nt = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for _, _, g, ys, y, m in exs:
            l, zs, tmin, tmax = loss_of(sc, g, ys, y, m)
            loss += float(l); acc += float((zs > 0).float() == ys)
            k = m[:, 1] > 0
            et += float((tmin[k] - y[k, 1]).abs().sum() + (tmax[k] - y[k, 2]).abs().sum()) / 2; nt += int(k.sum())
    return {"loss": loss / len(exs), "solve_acc": acc / len(exs), "time_mae": et / max(1, nt)}


def train(epochs, lr, hidden, steps, seed):
    random.seed(seed); torch.manual_seed(seed)
    sc = SetScorer(Task.load(split_dirs("train")[0]).spec, hidden=hidden, steps=steps, seed=seed)
    tr, va = examples("train", sc), examples("val", sc)
    base = max(np.mean([float(e[3]) for e in va]), 1 - np.mean([float(e[3]) for e in va]))
    print("train sets %d, val sets %d; val majority-class solve accuracy %.3f" % (len(tr), len(va), base))
    opt = torch.optim.Adam(sc.model.parameters(), lr=lr)
    best, state, t0 = None, None, time.time()
    for ep in range(epochs):
        sc.model.train()
        idx = list(range(len(tr))); random.shuffle(idx)
        for i in idx:
            l = loss_of(sc, *tr[i][2:])[0]
            opt.zero_grad(); l.backward(); opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            v = evaluate_split(sc, va)
            sc.history.append(dict(epoch=ep + 1, val=v))
            print("epoch %3d  val loss %.3f  solve acc %.3f  time MAE %.3f LB" % (ep + 1, v["loss"], v["solve_acc"], v["time_mae"]))
            if best is None or v["loss"] < best:
                best, state = v["loss"], {k: t.clone() for k, t in sc.model.state_dict().items()}
    sc.model.load_state_dict(state)
    prefix = os.path.join(OUT, "v5_seed%d" % seed)
    sc.save(prefix, extra={"method": "V5", "kind": "set scorer", "train_seconds": round(time.time() - t0, 1),
                           "n_train_sets": len(tr), "n_val_sets": len(va), "best_val_loss": best})
    print("saved", prefix, "in %.0f s" % (time.time() - t0))


# ---------------------------------------------------------------- quick test: one POPF at a time
def sequential(task, gt, ranked, budget_s):
    """POPF on the ranked sets one after another, caps CAPS, within budget_s; first valid plan wins."""
    t0, tried = time.monotonic(), 0
    for k, x in enumerate(ranked):
        left = budget_s - (time.monotonic() - t0)
        if left < 0.5:
            break
        cap = min(CAPS[k] if k < len(CAPS) else CAPS[-1], left)
        ws = Workspace()
        try:
            a = attempt(task, gt, x[2], Budget(cap), "seed", ws, check_proof=False)
        finally:
            ws.close()
        tried += 1
        if a.outcome == "valid":
            return {"solved": True, "rank": k, "tried": tried, "t": round(time.monotonic() - t0, 2),
                    "makespan": a.makespan, "n": len(x[2])}
    return {"solved": False, "tried": tried, "t": round(time.monotonic() - t0, 2)}


def seedtest(seed, budget_s, parallel, split):
    dirs = split_dirs(split)
    sc = SetScorer.load(os.path.join(OUT, "v5_seed%d" % seed), Task.load(dirs[0]).spec)

    def run(d):
        task = Task.load(d)
        gt = GroundTask(task)
        cands = candidates(gt)
        t0 = time.monotonic()
        ranked = sc.rank(task, cands)
        gnn_s = time.monotonic() - t0
        row = {"instance": os.path.basename(d), "n_cands": len(cands), "gnn_s": round(gnn_s, 2),
               "top": [(c, len(S), round(s[0], 3)) for _, c, S, s in ranked[:5]]}
        row["GNN"] = sequential(task, gt, ranked, budget_s - gnn_s)
        row["LB"] = prev[row["instance"]]["LB"] if row["instance"] in prev else sequential(task, gt, v42_order(cands), budget_s)
        print(row["instance"], "GNN", row["GNN"], "| LB", row["LB"], flush=True)
        return row
    out = os.path.join(OUT, "sequential_%s.jsonl" % split)
    old = out.replace(".jsonl", "_onTimeProduct.jsonl")   # first version's run: its TRPG-bound arm is reused
    prev = {json.loads(l)["instance"]: json.loads(l) for l in open(old)} if os.path.exists(old) else {}
    done = {json.loads(l)["instance"] for l in open(out)} if os.path.exists(out) else set()   # resumable

    def run_and_save(d):
        row = run(d)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        list(ex.map(run_and_save, [d for d in dirs if os.path.basename(d) not in done]))
    rows = [json.loads(l) for l in open(out)]
    for k in ("GNN", "LB"):
        ok = [r[k] for r in rows if r[k]["solved"]]
        print("%-3s order, one POPF at a time, %g s: solved %d / %d; mean POPF calls %.2f; mean time to plan %.1f s"
              % (k, budget_s, len(ok), len(rows), np.mean([r[k]["tried"] for r in rows]), np.mean([o["t"] for o in ok]) if ok else 0))
    print("only GNN:", [r["instance"] for r in rows if r["GNN"]["solved"] and not r["LB"]["solved"]])
    print("only LB: ", [r["instance"] for r in rows if r["LB"]["solved"] and not r["GNN"]["solved"]])
    print("%.0f s -> %s" % (time.time() - t0, out))


# ---------------------------------------------------------------- V5 inside the planner (V4.2 pipeline)
@dataclass
class V5Config(V4Config):
    name: str = "TiFLAX_V5"
    # one POPF at a time in the GNN's order; the k-th seed gets seed_caps[k] * T
    seed_caps: tuple = (1 / 3, 7 / 30, 1 / 6, 2 / 15, 2 / 15)
    seed_frac: float = 1.0
    recovery_mode: str = "sequential"
    refine_parallel: int = 1


class V5Planner(V4Planner):
    def seed_rounds(self, R):
        return [self.scorer.ranked_seeds(R.task, R.gt)]       # all candidate sets, ranked by the GNN

    def _seed_phase(self, R):
        cfg = self.cfg
        with R.budget.charge("seeds"):
            seeds = [x for r in self.seed_rounds(R) for x in r]
        for k, seed in enumerate(seeds):
            left = R.budget.remaining()
            if left < 0.5:
                break
            c = cfg.seed_caps[min(k, len(cfg.seed_caps) - 1)] * R.budget.total
            a = R.plan(seed[2], "seed", cap=min(c, left))
            if a.outcome == "valid":
                return a
        R.notes.append("no seed solved")
        return None


def build(key, spec, seed=0, **over):
    assert key == "V5", key
    return V5Planner(V5Config(**over), SetScorer.load(os.path.join(OUT, "v5_seed%d" % seed), spec))


# ==================================================================================================
# V5a (first attempt)
# V5a (first V5 attempt): a 3-head GNN (object importance, min / max completion time) + V4.2's seeds.
# Labels (POPF only, every plan VAL-valid on the full task). For each training kitchen POPF runs on a pool
# of object sets: the full task, the certificate's objects and V4.2's seeds. For a valid plan pi and an
# object o it uses, the completion time c_o(pi) = end of the last action of pi that has o as an argument.
#   importance(o) = 1 if o is in the shortest plan of the pool (or a goal object), else 0
#   T_min(o)      = min over pool plans using o of c_o(pi)      (in units of LB(Pi))
#   T_max(o)      = max over pool plans using o of c_o(pi)
# Objects no plan uses get no time label (mask 0). These are empirical bounds over the plans POPF found,
# not proven bounds over all valid plans.
#
# Network: FLAX's encode-process-decode GNN with 3 outputs per object: z_imp, z_min, z_gap;
# p_imp = sigmoid(z_imp), T_min = softplus(z_min), T_max = T_min + softplus(z_gap), so T_min <= T_max.
# Loss = BCE(p_imp) + Huber(T_min) + Huber(T_max) (time terms masked).
#
# Selection (V5 = V2's mixing of two signals + V4.2's seeds). D = latest goal deadline / LB.
#   on-time probability a(o) = P(T <= D) for T uniform on [T_min(o), T_max(o)]
#   score r(o) = p_imp(o) * a(o)
#   temporal filtration: drop o if T_min(o) > 1.1 D (it cannot finish in time); goal objects are kept
#   seeds: V4.2's work-optimal single-agent and two-agent seeds built on the filtered objects, with the
#   earliest-time seed replaced by a GNN seed COMP_T(goals + {o : r(o) >= 0.5}); same count as V4.2 (6).
#   The improvement phase ranks capacity candidates by r(o) (the last score head).
# ==================================================================================================
FILTER_SLACK = 1.1


def latest_deadline(task):
    ds = [g.deadline for g in task.problem.goals if g.deadline is not None]
    return max(ds) if ds else None


def v42_seeds(gt, k=4, universe=None, rank_key=None):
    """V4.2's single round: V4's mix (3 work-optimal + 1 earliest-time) + 2 two-agent seeds."""
    first, second = seed_rounds(gt, k, rank_key=rank_key, universe=universe)
    return first + [x for x in second if "+" in x[1]][:2]


# ---------------------------------------------------------------- labels from POPF plans
def completion_times(plan):
    c = {}
    for a in plan.actions:
        for o in a.args:
            c[o] = max(c.get(o, 0.0), a.end)
    return c


def v5a_label_instance(d, cap=10.0):
    task = Task.load(d)
    gt = GroundTask(task)
    lb = temporal_lb(gt)
    pool = {"full": frozenset(task.objects)}
    cert = os.path.join(d, "certificate.plan")
    if os.path.exists(cert):
        used = TemporalPlan.parse(open(cert).read()).objects_used() & set(task.objects)
        pool["cert"] = frozenset(comp_t(gt, used))
    for i, (_, c, S) in enumerate(v42_seeds(gt)):
        pool.setdefault("seed%d_%s" % (i, c), frozenset(S))

    def one(item):
        ws = Workspace()
        try:
            return item[0], attempt(task, gt, item[1], Budget(cap), item[0], ws, check_proof=True)
        finally:
            ws.close()
    with concurrent.futures.ThreadPoolExecutor(len(pool)) as ex:
        res = list(ex.map(one, sorted(pool.items())))
    plans = [(k, a.plan) for k, a in res if a.outcome == "valid"]
    row = {"instance": os.path.basename(d), "lb": lb, "n_objects": len(task.objects),
           "plans": {k: a.makespan for k, a in res if a.outcome == "valid"},
           "outcomes": {k: a.outcome for k, a in res}}
    if not plans:
        row["labels"] = None
        return row
    best = min(plans, key=lambda x: x[1].makespan)[1]
    used_best = best.objects_used() | set(task.goal_objects)
    cs = [completion_times(p) for _, p in plans]
    lab = {}
    for o in sorted(task.objects):
        ts = [c[o] / lb for c in cs if o in c]
        lab[o] = {"imp": 1.0 if o in used_best else 0.0,
                  "tmin": min(ts) if ts else None, "tmax": max(ts) if ts else None}
    row["labels"] = lab
    return row


def v5a_make_labels(split, cap, parallel):
    dirs = split_dirs(split)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        rows = list(ex.map(lambda d: v5a_label_instance(d, cap), dirs))
    out = os.path.join(OUT, "v5a_labels_%s.json" % split)
    json.dump(rows, open(out, "w"), indent=0)
    ok = [r for r in rows if r["labels"]]
    print("%s: %d / %d kitchens with a valid POPF plan, %.0f s -> %s" % (split, len(ok), len(rows), time.time() - t0, out))
    for r in rows:
        print("  %-14s %s" % (r["instance"], {k: round(v, 1) for k, v in r["plans"].items()}))


# ---------------------------------------------------------------- the 3-head scorer
class TimeScorer(Scorer):
    """heads: 0 = importance logit, 1 = T_min pre-softplus, 2 = gap pre-softplus (times in LB units)."""

    def __init__(self, spec, hidden=16, steps=3, seed=0):
        super().__init__(spec, heads=3, temporal_features=True, hidden=hidden, steps=steps, seed=seed)
        self.combine = "v5"

    def predict(self, task, gt=None, graph=None):
        g = graph or self.graph(task, gt)
        self.model.eval()
        with torch.no_grad():
            z = self.logits(g)
        p = torch.sigmoid(z[:, 0]).numpy()
        tmin = nn.functional.softplus(z[:, 1]).numpy()
        tmax = tmin + nn.functional.softplus(z[:, 2]).numpy()
        return g, {o: (float(p[i]), float(tmin[i]), float(tmax[i])) for i, o in enumerate(g.objects)}

    @staticmethod
    def on_time(tmin, tmax, D):
        if D is None:
            return 1.0
        if tmax - tmin < 1e-6:
            return 1.0 if tmin <= D else 0.0
        return float(min(1.0, max(0.0, (D - tmin) / (tmax - tmin))))

    def scores(self, task, gt=None, graph=None):
        """-> {o: [p_imp, r]}; r = p_imp * on-time probability. The planner reads head 0 to rank agents and the
        last head to rank capacity candidates."""
        g, pr = self.predict(task, gt, graph)
        D = latest_deadline(task)
        D = None if D is None else D / g.lb
        return {o: [p, p * self.on_time(a, b, D)] for o, (p, a, b) in pr.items()}

    def reward(self, p):
        return p[-1]

    @property
    def reward_scale(self):
        return 1.0

    def seed_sets(self, gt, k=4):
        """V5 seeds (see module doc)."""
        task = gt.task
        g, pr = self.predict(task, gt)
        D = latest_deadline(task)
        D = None if D is None else D / g.lb
        goal = set(task.goal_objects)
        U = {o for o, (p, a, b) in pr.items() if D is None or a <= FILTER_SLACK * D} | goal
        r = {o: p * self.on_time(a, b, D) for o, (p, a, b) in pr.items()}
        rank = lambda c: r.get(c, 0.0)  # noqa: E731
        _, second = seed_rounds(gt, k, rank_key=rank, universe=U)
        out = agent_seeds(gt, 3, rank_key=rank, kind="work", universe=U)
        S = frozenset(comp_t(gt, goal | {o for o, v in r.items() if v >= 0.5}))
        if gt.proof(S) is None:
            out.append((gt.trpg(S).lb, "gnn", S))
        out += [x for x in second if "+" in x[1]][:2]
        seen, uniq = set(), []
        for x in out:
            if x[2] not in seen:
                seen.add(x[2]); uniq.append(x)
        for x in v42_seeds(gt):              # filtration removed too much: fill up with V4.2's own seeds
            if len(uniq) >= 6:
                break
            if x[2] not in seen:
                seen.add(x[2]); uniq.append(x)
        return uniq[:6]

    @classmethod
    def load(cls, prefix, spec):
        meta = json.load(open(prefix + ".json"))
        mc = meta["model"]
        s = cls(spec, hidden=mc["hidden"], steps=mc["steps"])
        if s.vocab.signature() != meta["vocab"]:
            raise ValueError("vocabulary mismatch for " + prefix)
        s.model.load_state_dict(torch.load(prefix + ".pt", weights_only=True))
        s.history = meta.get("history", [])
        return s


# ---------------------------------------------------------------- training
def v5a_examples(split, scorer):
    rows = json.load(open(os.path.join(OUT, "v5a_labels_%s.json" % split)))
    out = []
    for r in rows:
        if not r["labels"]:
            continue
        d = os.path.join(PDDL, split, r["instance"])
        task = Task.load(d)
        g = scorer.graph(task, GroundTask(task))
        lab = r["labels"]
        y = np.zeros((len(g.objects), 3), dtype=np.float32)
        m = np.zeros((len(g.objects), 3), dtype=np.float32)
        for i, o in enumerate(g.objects):
            L = lab[o]
            y[i, 0], m[i, 0] = L["imp"], 1.0
            if L["tmin"] is not None:
                y[i, 1], y[i, 2] = L["tmin"], L["tmax"]
                m[i, 1] = m[i, 2] = 1.0
        out.append((r["instance"], g, torch.from_numpy(y), torch.from_numpy(m)))
    return out


def v5a_loss_of(scorer, g, y, m):
    z = scorer.model(*to_tensors(g))
    tmin = nn.functional.softplus(z[:, 1])
    tmax = tmin + nn.functional.softplus(z[:, 2])
    l_imp = nn.functional.binary_cross_entropy_with_logits(z[:, 0], y[:, 0])
    hub = lambda a, b, w: (nn.functional.huber_loss(a, b, reduction="none") * w).sum() / w.sum().clamp(min=1.0)  # noqa: E731
    l_t = hub(tmin, y[:, 1], m[:, 1]) + hub(tmax, y[:, 2], m[:, 2])
    return l_imp + l_t, z, tmin, tmax


def v5a_evaluate_split(scorer, exs):
    scorer.model.eval()
    acc, n, err_min, err_max, nt, loss = 0.0, 0, 0.0, 0.0, 0, 0.0
    with torch.no_grad():
        for _, g, y, m in exs:
            l, z, tmin, tmax = v5a_loss_of(scorer, g, y, m)
            loss += float(l)
            acc += float(((z[:, 0] > 0).float() == y[:, 0]).sum()); n += len(y)
            k = m[:, 1] > 0
            err_min += float((tmin[k] - y[k, 1]).abs().sum()); err_max += float((tmax[k] - y[k, 2]).abs().sum())
            nt += int(k.sum())
    return {"loss": loss / max(1, len(exs)), "imp_acc": acc / max(1, n),
            "tmin_mae": err_min / max(1, nt), "tmax_mae": err_max / max(1, nt)}


def v5a_train(epochs, lr, hidden, steps, seed):
    random.seed(seed); torch.manual_seed(seed)
    task0 = Task.load(split_dirs("train")[0])
    sc = TimeScorer(task0.spec, hidden=hidden, steps=steps, seed=seed)
    tr, va = v5a_examples("train", sc), v5a_examples("val", sc)
    opt = torch.optim.Adam(sc.model.parameters(), lr=lr)
    best, best_state, t0 = None, None, time.time()
    for ep in range(epochs):
        sc.model.v5a_train()
        order = list(range(len(tr))); random.shuffle(order)
        for i in order:
            _, g, y, m = tr[i]
            l = v5a_loss_of(sc, g, y, m)[0]
            opt.zero_grad(); l.backward(); opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            v = v5a_evaluate_split(sc, va)
            sc.history.append(dict(epoch=ep + 1, train=v5a_evaluate_split(sc, tr), val=v))
            print("epoch %3d  val loss %.3f  imp acc %.3f  T_min MAE %.3f LB  T_max MAE %.3f LB"
                  % (ep + 1, v["loss"], v["imp_acc"], v["tmin_mae"], v["tmax_mae"]))
            if best is None or v["loss"] < best:
                best, best_state = v["loss"], {k: t.clone() for k, t in sc.model.state_dict().items()}
    sc.model.load_state_dict(best_state)
    prefix = os.path.join(OUT, "v5a_seed%d" % seed)
    sc.save(prefix, extra={"method": "V5a", "train_seconds": round(time.time() - t0, 1),
                           "n_train": len(tr), "n_val": len(va), "best_val_loss": best})
    print("saved", prefix, "in %.0f s" % (time.time() - t0))


# ---------------------------------------------------------------- quick test: seeds only
def v5a_seedtest(seed, cap, parallel, split):
    model = os.path.join(OUT, "v5a_seed%d" % seed)
    dirs = split_dirs(split)
    spec = Task.load(dirs[0]).spec
    sc = TimeScorer.load(model, spec)

    def run(d):
        task = Task.load(d)
        gt = GroundTask(task)
        sets = {"V4.2": [x[2] for x in v42_seeds(gt)], "V5": [x[2] for x in sc.seed_sets(gt)]}
        uniq = sorted({S for v in sets.values() for S in v}, key=len)

        def one(S):
            ws = Workspace()
            try:
                return S, attempt(task, gt, S, Budget(cap), "seed", ws, check_proof=False)
            finally:
                ws.close()
        with concurrent.futures.ThreadPoolExecutor(len(uniq) or 1) as ex:
            res = dict(ex.map(one, uniq))
        row = {"instance": os.path.basename(d), "n": len(task.objects)}
        for k, v in sets.items():
            ok = [res[S] for S in v if res[S].outcome == "valid"]
            row[k] = {"solved": bool(ok), "n_seeds": len(v), "sizes": [len(S) for S in v],
                      "first_s": min((a.planner_s for a in ok), default=None),
                      "makespan": min((a.makespan for a in ok), default=None)}
        return row
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        rows = list(ex.map(run, dirs))
    out = os.path.join(OUT, "v5a_seedtest_%s.json" % split)
    json.dump(rows, open(out, "w"), indent=1)
    for k in ("V4.2", "V5"):
        print("%-5s seeds solved %d / %d" % (k, sum(r[k]["solved"] for r in rows), len(rows)))
    both = [r for r in rows if r["V4.2"]["solved"] and r["V5"]["solved"]]
    if both:
        print("both solve %d; mean best seed makespan V4.2 %.1f vs V5 %.1f; V5 shorter on %d, longer on %d"
              % (len(both), np.mean([r["V4.2"]["makespan"] for r in both]), np.mean([r["V5"]["makespan"] for r in both]),
                 sum(r["V5"]["makespan"] < r["V4.2"]["makespan"] - 0.5 for r in both),
                 sum(r["V5"]["makespan"] > r["V4.2"]["makespan"] + 0.5 for r in both)))
    print("only V4.2:", [r["instance"] for r in rows if r["V4.2"]["solved"] and not r["V5"]["solved"]])
    print("only V5:  ", [r["instance"] for r in rows if r["V5"]["solved"] and not r["V4.2"]["solved"]])
    print("%.0f s -> %s" % (time.time() - t0, out))


def main():
    ensure_hash_seed()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["labels", "train", "seedtest", "v5a-labels", "v5a-train", "v5a-seedtest"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--cap", type=float, default=None, help="labels: POPF cap per set (10 s); seedtest: total "
                    "budget (30 s); v5a-seedtest: POPF cap per seed (15 s)")
    ap.add_argument("--parallel", type=int, default=None, help="default 1 (V5), 2 (V5a)")
    ap.add_argument("--epochs", type=int, default=None, help="default 60 (V5), 200 (V5a)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    v5a = a.cmd.startswith("v5a")
    par = a.parallel or (2 if v5a else 1)
    if a.cmd == "labels":
        make_labels(a.split, a.cap or 10.0, par)
    elif a.cmd == "train":
        train(a.epochs or 60, a.lr, a.hidden, a.steps, a.seed)
    elif a.cmd == "seedtest":
        seedtest(a.seed, a.cap or 30.0, par, a.split)
    elif a.cmd == "v5a-labels":
        v5a_make_labels(a.split, a.cap or 10.0, par)
    elif a.cmd == "v5a-train":
        v5a_train(a.epochs or 200, a.lr, a.hidden, a.steps, a.seed)
    else:
        v5a_seedtest(a.seed, a.cap or 15.0, par, a.split)


if __name__ == "__main__":
    main()
