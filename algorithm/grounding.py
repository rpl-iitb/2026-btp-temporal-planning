"""Task (Pi, Pi|S, PDDL files), grounding, temporal relaxed planning graph (TRPG: LB(S), proofs,
relaxed plans) and the COMP_T closure. See algorithm/README.md.
"""
import hashlib
import heapq
import itertools
import math
import os
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from pddl_generator import HLProblem, deadline_literals, load_domain_spec, write_domain, write_problem
from pddl_generator.domain_spec import Atom


# ==================== Task =======================================================================
Fact = Tuple[str, ...]


@dataclass(frozen=True)
class TIL:
    time: float
    fact: Fact
    value: bool          # True = becomes true, False = becomes false


def check_restriction_sound(spec):
    """Raise if Pi|S could admit plans that are invalid on Pi."""
    bad = []
    for act in spec.actions.values():
        for pt, atoms in act.conditions.items():
            bad += ["{} {} (not {})".format(act.name, pt, a) for a in atoms if a.negated]
    if bad:
        raise ValueError("negative conditions make Pi|S unsound: " + "; ".join(bad))


class Task:
    def __init__(self, spec, problem: HLProblem, _domain_file=None):
        check_restriction_sound(spec)
        self.spec = spec
        self.problem = problem
        self.name = problem.instance_id
        self.objects: Dict[str, str] = {o.name: o.type for o in problem.objects}
        self.constants: Dict[str, str] = dict(spec.constants)
        self.init = frozenset(tuple(f) for f in problem.facts)
        self.fluents = dict(problem.fluents)
        tils = [TIL(float(t.time), tuple(t.fact), bool(t.value)) for t in problem.timed_literals]
        tils += [TIL(float(t), tuple(w), False) for t, w in deadline_literals(spec, problem)]
        self.tils: List[TIL] = sorted(tils, key=lambda x: (x.time, x.fact, x.value))
        self.goals: List[Fact] = [tuple(g.fact) for g in problem.goals]
        self.deadlines: Dict[Fact, float] = {tuple(g.fact): float(g.deadline)
                                             for g in problem.goals if g.deadline is not None}
        self.goal_objects = frozenset(a for g in self.goals for a in g[1:] if a in self.objects)
        self._domain_file = _domain_file
        self._files = None

    # ---- construction --------------------------------------------------------------------
    @classmethod
    def load(cls, instance_dir, spec=None, domain_yaml=None):
        """instance_dir holds hl_problem.json (as written by pddl_generator.cli)."""
        prob = HLProblem.load(os.path.join(instance_dir, "hl_problem.json"))
        if spec is None:
            from pddl_generator import get_benchmark
            spec = load_domain_spec(domain_yaml) if domain_yaml else get_benchmark(prob.benchmark).domain_spec()
        return cls(spec, prob)

    def restrict(self, keep) -> "Task":
        """Pi|S. Goal objects are always kept (as in FLAX/iFLAX)."""
        keep = set(keep) | set(self.goal_objects)
        sub = self.problem.restrict(keep, instance_id=self.name)
        return Task(self.spec, sub, _domain_file=self._domain_file)

    # ---- typing ------------------------------------------------------------------------------
    def type_of(self, name) -> Optional[str]:
        return self.objects.get(name) or self.constants.get(name)

    def is_a(self, name, ty) -> bool:
        t = self.type_of(name)
        return t is not None and self.spec.is_subtype(t, ty)

    def of_type(self, ty):
        return sorted(o for o, t in self.objects.items() if self.spec.is_subtype(t, ty))

    # ---- files -------------------------------------------------------------------------------
    def domain_file(self, workdir=None) -> str:
        if self._domain_file is None or not os.path.exists(self._domain_file):
            text = write_domain(self.spec)
            h = hashlib.sha256(text.encode()).hexdigest()[:16]
            d = workdir or os.path.join(tempfile.gettempdir(), "tiflax_domains")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "domain_{}.pddl".format(h))
            if not os.path.exists(path):
                tmp = path + ".{}.{}.tmp".format(os.getpid(), threading.get_ident())   # unique per thread
                with open(tmp, "w") as f:
                    f.write(text)
                os.replace(tmp, path)
            self._domain_file = path
        return self._domain_file

    def write(self, workdir, tag="problem"):
        """-> (domain_file, problem_file) for this (possibly restricted) task."""
        os.makedirs(workdir, exist_ok=True)
        pf = os.path.join(workdir, "{}.pddl".format(tag))
        with open(pf, "w") as f:
            f.write(write_problem(self.spec, self.problem))
        return self.domain_file(), pf

    def size(self) -> dict:
        return {"objects": len(self.objects), "init_facts": len(self.init), "fluents": len(self.fluents),
                "goals": len(self.goals), "tils": len(self.tils)}


# ==================== Grounding and the temporal relaxed planning graph ==========================
EPS = 0.001
INF = math.inf


@dataclass(frozen=True)
class GroundAction:
    name: str
    args: Tuple[str, ...]
    duration: float
    cond_start: FrozenSet[Fact]
    cond_over: FrozenSet[Fact]
    cond_end: FrozenSet[Fact]
    add_start: FrozenSet[Fact]
    del_start: FrozenSet[Fact]
    add_end: FrozenSet[Fact]
    del_end: FrozenSet[Fact]
    objects: FrozenSet[str]          # object (non-constant) arguments

    @property
    def key(self):
        return (self.name,) + self.args

    def __str__(self):
        return "({})".format(" ".join(self.key))


def _subst(atom: Atom, b: dict) -> Fact:
    return (atom.pred,) + tuple(b.get(x, x) for x in atom.args)


def ground_actions(task) -> List[GroundAction]:
    """All type-correct ground actions whose static conditions hold in the
    initial state and whose duration is defined. (Dynamic conditions are
    filtered later by relaxed reachability.)"""
    spec = task.spec
    static = task.problem.static_predicates(spec)
    facts_by_pred = defaultdict(list)
    for f in sorted(task.init):
        facts_by_pred[f[0]].append(f)
    init = task.init
    names = dict(task.objects)
    names.update(task.constants)
    typed = defaultdict(list)
    for n in sorted(names):
        for anc in spec.ancestors(names[n]):
            typed[anc].append(n)

    out = []
    for act in spec.actions.values():
        ptype = {p.name: p.type for p in act.params}
        conds = [(pt, a) for pt in ("at_start", "over_all", "at_end") for a in act.conditions[pt]]
        st_atoms = [a for _, a in conds if a.pred in static]
        # bind via static atoms first, most selective first
        st_atoms.sort(key=lambda a: len(facts_by_pred[a.pred]))

        def match(atom, b):
            for f in facts_by_pred[atom.pred]:
                nb = dict(b)
                ok = True
                for x, v in zip(atom.args, f[1:]):
                    if x.startswith("?"):
                        if x in nb:
                            if nb[x] != v:
                                ok = False
                                break
                        else:
                            ty = names.get(v)
                            if ty is None or not spec.is_subtype(ty, ptype[x]):
                                ok = False
                                break
                            nb[x] = v
                    elif x != v:
                        ok = False
                        break
                if ok:
                    yield nb

        def bindings(i, b):
            if i == len(st_atoms):
                free = [p.name for p in act.params if p.name not in b]
                for combo in itertools.product(*[typed[ptype[x]] for x in free]):
                    nb = dict(b)
                    nb.update(zip(free, combo))
                    yield nb
                return
            for nb in match(st_atoms[i], b):
                yield from bindings(i + 1, nb)

        seen = set()
        for b in bindings(0, {}):
            args = tuple(b[p.name] for p in act.params)
            if args in seen:
                continue
            seen.add(args)
            if any(_subst(a, b) not in init for a in st_atoms):
                continue
            if isinstance(act.duration, Atom):
                d = task.fluents.get(_subst(act.duration, b))
                if d is None:
                    continue
            else:
                d = act.duration
            fs = lambda atoms: frozenset(_subst(a, b) for a in atoms if a.pred not in static)  # noqa: E731
            pos = lambda atoms: frozenset(_subst(a, b) for a in atoms if not a.negated)  # noqa: E731
            neg = lambda atoms: frozenset(_subst(a, b) for a in atoms if a.negated)  # noqa: E731
            out.append(GroundAction(
                act.name, args, float(d),
                fs(act.conditions["at_start"]), fs(act.conditions["over_all"]), fs(act.conditions["at_end"]),
                pos(act.effects["at_start"]), neg(act.effects["at_start"]),
                pos(act.effects["at_end"]), neg(act.effects["at_end"]),
                frozenset(a for a in args if a in task.objects)))
    return out


@dataclass
class TRPGResult:
    fact_time: Dict[Fact, float]
    act_start: Dict[int, float]                     # action index -> earliest start
    achiever: Dict[Fact, Optional[Tuple[int, str]]]  # fact -> (action index, "start"|"end") or None (init / TIL)
    goal_time: Dict[Fact, float]
    keep: Optional[FrozenSet[str]]

    @property
    def lb(self) -> float:
        return max(self.goal_time.values(), default=0.0)

    @property
    def unreachable_goals(self):
        return [g for g, t in self.goal_time.items() if t == INF]

    @property
    def solvable_relaxed(self) -> bool:
        return not self.unreachable_goals


class GroundTask:
    """The full task, grounded once; TRPG / relaxed plans / proofs for any S."""

    def __init__(self, task):
        self.task = task
        raw = ground_actions(task)
        # relaxed reachability on the full task prunes dead ground actions
        self.actions: List[GroundAction] = raw
        self._index()
        r = self.trpg(None, windows=False)
        self.actions = [a for i, a in enumerate(raw) if r.act_start.get(i, INF) < INF]
        self._index()
        self.full = self.trpg(None)
        self.relevant = self._goal_relevant(self.full)

    def _index(self):
        self.consumers = defaultdict(list)
        self.pre = []
        self.by_object = defaultdict(list)
        for i, a in enumerate(self.actions):
            pre = a.cond_start | (a.cond_over - a.add_start) | (a.cond_end - a.add_start)
            self.pre.append(pre)
            for f in pre:
                self.consumers[f].append(i)
            for o in a.objects:
                self.by_object[o].append(i)
        self._windows()

    def _windows(self):
        # windows: initially true, deleted by a TIL, never re-added
        added = {f for a in self.actions for f in a.add_start | a.add_end}
        til_add = {t.fact for t in self.task.tils if t.value}
        self.window = {}
        for t in self.task.tils:
            if not t.value and t.fact in self.task.init and t.fact not in added and t.fact not in til_add:
                self.window[t.fact] = min(self.window.get(t.fact, INF), t.time)

    def with_task(self, task2):
        """Same ground actions, different timed literals (e.g. tightened deadlines).
        task2 must have the same objects, initial facts and fluents as self.task."""
        import copy
        g = copy.copy(self)
        g.task = task2
        g._windows()
        g.full = g.trpg(None)
        return g

    # ---- TRPG ---------------------------------------------------------------------------
    def trpg(self, keep=None, windows=True) -> TRPGResult:
        task = self.task
        keep = None if keep is None else frozenset(keep) | task.goal_objects
        objs = task.objects

        def in_s(f):
            return keep is None or all(x in keep or x not in objs for x in f[1:])

        time = {}
        heap = []

        def relax(f, t, ach):
            if t < time.get(f, INF):
                time[f] = t
                achiever[f] = ach
                heapq.heappush(heap, (t, f))

        achiever = {}
        for f in sorted(task.init):
            if in_s(f):
                relax(f, 0.0, None)
        for t in task.tils:
            if t.value and in_s(t.fact):
                relax(t.fact, t.time, None)

        active = [keep is None or a.objects <= keep for a in self.actions]
        count = [0] * len(self.actions)
        start = {}
        popped = set()

        def evaluate(i):
            a = self.actions[i]
            s = 0.0
            for f in a.cond_start | (a.cond_over - a.add_start):
                s = max(s, time[f])
            for f in a.cond_end - a.add_start:
                s = max(s, time[f] - a.duration)
            if windows:
                for f in a.cond_start:
                    if f in self.window and s >= self.window[f] - EPS:
                        return
                for f in (a.cond_over | a.cond_end) - a.add_start:
                    if f in self.window and s + a.duration > self.window[f] - EPS:
                        return
            if s < start.get(i, INF):
                start[i] = s
                for f in a.add_start:
                    relax(f, s, (i, "start"))
                for f in a.add_end:
                    relax(f, s + a.duration, (i, "end"))

        for i, a in enumerate(self.actions):
            if active[i] and not self.pre[i]:
                evaluate(i)
        while heap:
            t, f = heapq.heappop(heap)
            if t > time.get(f, INF):
                continue
            first = f not in popped
            popped.add(f)
            for i in self.consumers.get(f, ()):
                if not active[i]:
                    continue
                if first:
                    count[i] += 1
                if count[i] == len(self.pre[i]):
                    evaluate(i)
        goal_time = {g: time.get(g, INF) for g in task.goals}
        return TRPGResult(time, start, achiever, goal_time, keep)

    # ---- relaxed plans, relevance -------------------------------------------------------
    def relaxed_plan(self, r: TRPGResult) -> List[int]:
        """Backchain from the goals through earliest achievers."""
        plan, seen = [], set()
        stack = [g for g in self.task.goals if r.fact_time.get(g, INF) < INF]
        while stack:
            f = stack.pop()
            if f in seen:
                continue
            seen.add(f)
            ach = r.achiever.get(f)
            if ach is None:
                continue
            i = ach[0]
            if i not in plan:
                plan.append(i)
                stack.extend(self.pre[i])
        return sorted(plan, key=lambda i: r.act_start[i])

    def guided_relaxed_plan(self, r: TRPGResult, score, tol) -> List[int]:
        """Relaxed plan whose achievers are chosen among the near-earliest ones (achieves f by
        time[f] + tol) by preferring objects already in the plan (a compact set of stations an
        agent can cycle between), then objects with a high score(o) (e.g. the learned object score).
        tol = 0 gives the earliest-achiever plan up to ties."""
        if not hasattr(self, "_producers"):
            prod = defaultdict(list)
            for i, a in enumerate(self.actions):
                for f in a.add_start:
                    prod[f].append((i, "start"))
                for f in a.add_end - a.add_start:
                    prod[f].append((i, "end"))
            self._producers = prod
        plan, seen, chosen = [], set(), set(self.task.goal_objects)
        stack = [g for g in self.task.goals if r.fact_time.get(g, INF) < INF]
        while stack:
            f = stack.pop()
            if f in seen:
                continue
            seen.add(f)
            if r.achiever.get(f) is None:          # initial fact / timed literal
                continue
            limit = r.fact_time[f] + tol
            best, best_key = None, None
            for i, when in self._producers.get(f, ()):
                s0 = r.act_start.get(i)
                if s0 is None:
                    continue
                t = s0 + (self.actions[i].duration if when == "end" else 0.0)
                if t > limit + 1e-9:
                    continue
                objs = self.actions[i].objects
                key = (sum(1.0 if o in chosen else score(o) for o in objs) - len(objs), -t, -i)
                if best_key is None or key > best_key:
                    best, best_key = i, key
            if best is None:
                best = r.achiever[f][0]
            if best not in plan:
                plan.append(best)
                chosen |= self.actions[best].objects
                stack.extend(self.pre[best])
        return sorted(plan, key=lambda i: r.act_start[i])

    def cost_relaxed_plan(self, keep=None, windows_from=None, obj_cost=None) -> List[int]:
        """Work-optimal relaxed plan: h_add-style best supporters with action duration as cost
        (cost(f) = min over achievers a of dur(a) + sum of cost of a's conditions), then backchain.
        Unlike the earliest-time relaxed plan, it prefers achievers whose conditions are cheap to reach
        from each other -- for an agent that repeats a cycle per goal, a compact set of stations.
        Actions that violate a deadline window in the TRPG of `windows_from` are excluded.
        obj_cost: optional {object: extra cost}, added once per action for each object it uses (V6: a
        learned bias that makes unimportant objects expensive)."""
        task = self.task
        keep = None if keep is None else frozenset(keep) | task.goal_objects
        extra = [sum(obj_cost.get(o, 0.0) for o in a.objects) if obj_cost else 0.0 for a in self.actions]
        objs = task.objects
        in_s = lambda f: keep is None or all(x in keep or x not in objs for x in f[1:])  # noqa: E731
        usable = windows_from.act_start if windows_from is not None else None
        cost, sup, heap = {}, {}, []
        for f in sorted(task.init):
            if in_s(f):
                cost[f] = 0.0; heapq.heappush(heap, (0.0, f))
        for t in task.tils:
            if t.value and in_s(t.fact) and t.fact not in cost:
                cost[t.fact] = 0.0; heapq.heappush(heap, (0.0, t.fact))
        active = [(keep is None or a.objects <= keep) and (usable is None or i in usable)
                  for i, a in enumerate(self.actions)]
        count = [0] * len(self.actions)
        done = set()

        def fire(i):
            a = self.actions[i]
            c = a.duration + extra[i] + sum(cost[f] for f in self.pre[i])
            for f in a.add_start | a.add_end:
                if c < cost.get(f, INF):
                    cost[f] = c; sup[f] = i; heapq.heappush(heap, (c, f))
        for i in range(len(self.actions)):
            if active[i] and not self.pre[i]:
                fire(i)
        while heap:
            c, f = heapq.heappop(heap)
            if f in done or c > cost.get(f, INF):
                continue
            done.add(f)
            for i in self.consumers.get(f, ()):
                if active[i]:
                    count[i] += 1
                    if count[i] == len(self.pre[i]):
                        fire(i)
        plan, seen, stack = [], set(), [g for g in task.goals if g in cost]
        if len(stack) < len(task.goals):
            return []
        while stack:
            f = stack.pop()
            if f in seen or f not in sup:
                continue
            seen.add(f)
            i = sup[f]
            if i not in plan:
                plan.append(i); stack.extend(self.pre[i])
        return plan

    def relaxed_plan_objects(self, r: TRPGResult) -> set:
        return {o for i in self.relaxed_plan(r) for o in self.actions[i].objects}

    def _goal_relevant(self, r: TRPGResult) -> set:
        """Indices of reachable actions that can contribute to a goal
        (backward relevance over the reachable part)."""
        producers = defaultdict(list)
        for i, a in enumerate(self.actions):
            if i in r.act_start:
                for f in a.add_start | a.add_end:
                    producers[f].append(i)
        rel_f, rel_a = set(self.task.goals), set()
        stack = list(rel_f)
        while stack:
            f = stack.pop()
            for i in producers.get(f, ()):
                if i not in rel_a:
                    rel_a.add(i)
                    for g in self.pre[i]:
                        if g not in rel_f:
                            rel_f.add(g)
                            stack.append(g)
        return rel_a

    # ---- per-object facts used by closure and features -----------------------------------
    def earliest_use(self, r: TRPGResult = None) -> Dict[str, float]:
        """Earliest TRPG start of a goal-relevant action mentioning each object."""
        r = r or self.full
        out = {}
        for i in self.relevant:
            s = r.act_start.get(i, INF)
            for o in self.actions[i].objects:
                if s < out.get(o, INF):
                    out[o] = s
        return out

    def proof(self, keep) -> Optional[dict]:
        """None, or a dict explaining why Pi|keep provably has no plan."""
        r = self.trpg(keep)
        if r.solvable_relaxed:
            return None
        no_win = self.trpg(keep, windows=False)
        temporal = no_win.solvable_relaxed
        return {"kind": "deadline" if temporal else "unreachable",
                "goals": [" ".join(g) for g in r.unreachable_goals],
                "lb_without_deadlines": no_win.lb if temporal else INF}


# ==================== COMP_T closure =============================================================
def _enable(gt, i):
    """Objects of action i and of the full-task relaxed plan for its conditions."""
    r = gt.full
    objs = set(gt.actions[i].objects)
    stack, seen = list(gt.pre[i]), set()
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        ach = r.achiever.get(f)
        if ach is None:
            continue
        j = ach[0]
        objs |= gt.actions[j].objects
        stack.extend(gt.pre[j])
    return objs


def _goal_support(gt, goals):
    r = gt.full
    objs, stack, seen = set(), list(goals), set()
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        ach = r.achiever.get(f)
        if ach is None:
            continue
        j = ach[0]
        objs |= gt.actions[j].objects
        stack.extend(gt.pre[j])
    return objs


def comp_t(gt, S, max_rounds=20, info=None):
    """Action-grounded temporal closure. Returns a new set."""
    task = gt.task
    S = set(S) | set(task.goal_objects)
    if not hasattr(gt, "_rel_by_obj"):
        rel = {}
        for i in gt.relevant:
            for o in gt.actions[i].objects:
                rel.setdefault(o, []).append(i)
        for o in rel:
            rel[o].sort(key=lambda i: (gt.full.act_start.get(i, INF) + gt.actions[i].duration, len(gt.actions[i].objects)))
        gt._rel_by_obj = rel
    added_usable, added_goal = set(), set()
    for _ in range(max_rounds):
        r = gt.trpg(S)
        new = set()
        for o in sorted(S):
            acts = gt._rel_by_obj.get(o)
            if not acts:
                continue          # object no goal-relevant action uses (e.g. a constant-like count): keep as is
            if any(i in r.act_start for i in acts):
                continue
            new |= _enable(gt, acts[0]) - S
        if not new and not r.solvable_relaxed:
            new = _goal_support(gt, r.unreachable_goals) - S
            added_goal |= new
        else:
            added_usable |= new
        if not new:
            break
        S |= new
    if info is not None:
        info["comp_added_usable"] = len(added_usable)
        info["comp_added_goal"] = len(added_goal)
    return S


def agent_type(task):
    """The resource type used by the most action schemas (Overcooked: cook). Domain-level."""
    from collections import Counter
    from .semantics import resource_types
    rts = resource_types(task.spec)
    cnt = Counter()
    for act in task.spec.actions.values():
        for t in {p.type for p in act.params}:
            for rt in rts:
                if task.spec.is_subtype(t, rt):
                    cnt[rt] += 1
    return cnt.most_common(1)[0][0] if cnt else None
