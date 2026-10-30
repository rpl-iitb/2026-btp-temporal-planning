"""PDDL 2.1 durative-action semantics and plan simulator (agrees with VAL), schedule analysis
(precedence, CPM slack, critical path, resources) and earliest-start re-timing of valid plans.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from pddl_generator.planners import TemporalPlan, TimedAction

from .grounding import EPS, GroundAction


# ==================== Durative-action semantics and plan simulator ===============================
Fact = Tuple[str, ...]


class _Unbounded:
    """Open-ended validity. Ordered after every number; not a timestamp."""
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self):
        return "UNBOUNDED"

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return other is self

    def __gt__(self, other):
        return other is not self

    def __ge__(self, other):
        return True

    def __eq__(self, other):
        return other is self

    def __hash__(self):
        return hash("UNBOUNDED")

    def _no(self, *a):
        raise TypeError("UNBOUNDED is not a timestamp")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __float__ = __int__ = _no


UNBOUNDED = _Unbounded()


@dataclass
class Interval:
    fact: Fact
    added: float
    deleted: object = UNBOUNDED          # float or UNBOUNDED
    added_by: Optional[Tuple[int, str]] = None   # (plan step, "start"|"end"), None = init, ("til", k)
    deleted_by: Optional[Tuple[int, str]] = None

    @property
    def open_ended(self):
        return self.deleted is UNBOUNDED


@dataclass
class Running:
    step: int
    action: GroundAction
    start: float
    end: float


@dataclass
class CausalLink:
    producer: Optional[Tuple[int, str]]   # (step, "start"|"end"); None = initial state; ("til", k)
    consumer: Tuple[int, str]             # (step, "start"|"over"|"end")
    fact: Fact


@dataclass
class Violation:
    time: float
    step: Optional[int]
    kind: str
    detail: str


@dataclass
class TemporalState:
    """State between happenings: time, true facts with their validity intervals,
    running actions (with invariants and pending ends), pending timed literals."""
    time: float
    facts: Dict[Fact, Interval]
    running: List[Running] = field(default_factory=list)

    def holds(self, f) -> bool:
        return f in self.facts

    def copy(self):
        return TemporalState(self.time, dict(self.facts), list(self.running))


@dataclass
class _Happening:
    time: float
    kind: str               # "start" | "end" | "til"
    step: int               # plan step, or TIL index
    action: Optional[GroundAction] = None
    til: object = None


@dataclass
class SimResult:
    valid: bool
    violations: List[Violation]
    makespan: float
    final_state: Optional[TemporalState]
    intervals: List[Interval]                   # complete history of fact validity
    links: List[CausalLink]
    goal_achieved_at: Dict[Fact, float]
    steps: List[Tuple[float, GroundAction]]


def _interferes(h1: _Happening, h2: _Happening) -> Optional[str]:
    def parts(h):
        a = h.action
        if h.kind == "til":
            t = h.til
            return (set(), {t.fact} if t.value else set(), set() if t.value else {t.fact})
        if h.kind == "start":
            return (set(a.cond_start), set(a.add_start), set(a.del_start))
        return (set(a.cond_end), set(a.add_end), set(a.del_end))
    pre1, add1, del1 = parts(h1)
    pre2, add2, del2 = parts(h2)
    bad = (del1 & (pre2 | add2)) | (del2 & (pre1 | add1))
    return " ".join(sorted(map(str, bad))) if bad else None


class Simulator:
    """Executes a schedule [(start, GroundAction)] from the task's initial state."""

    def __init__(self, task, eps=EPS):
        self.task = task
        self.eps = eps

    def initial_state(self) -> TemporalState:
        return TemporalState(0.0, {f: Interval(f, 0.0) for f in self.task.init})

    def run(self, steps: List[Tuple[float, GroundAction]], durations: List[float] = None) -> SimResult:
        task, eps = self.task, self.eps
        viol: List[Violation] = []
        hs: List[_Happening] = []
        for k, (s, a) in enumerate(steps):
            d = a.duration
            if durations is not None and abs(durations[k] - d) > eps:
                viol.append(Violation(s, k, "duration", "{} stated {} but domain gives {}".format(a, durations[k], d)))
            hs.append(_Happening(s, "start", k, a))
            hs.append(_Happening(s + d, "end", k, a))
        for k, t in enumerate(task.tils):
            hs.append(_Happening(t.time, "til", k, til=t))
        order = {"end": 0, "til": 1, "start": 2}
        hs.sort(key=lambda h: (h.time, order[h.kind], h.step))

        st = self.initial_state()
        history: List[Interval] = []
        links: List[CausalLink] = []
        goal_at: Dict[Fact, float] = {}

        # group happenings that are closer than eps
        groups, cur = [], []
        for h in hs:
            if cur and h.time - cur[0].time > eps / 2:
                groups.append(cur)
                cur = []
            cur.append(h)
        if cur:
            groups.append(cur)

        def check(f, consumer, t, kind):
            iv = st.facts.get(f)
            if iv is None:
                viol.append(Violation(t, consumer[0], kind, "{} false".format(" ".join(f))))
            else:
                links.append(CausalLink(iv.added_by, consumer, f))

        for grp in groups:
            t = grp[0].time
            # interference inside one happening
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    if grp[i].kind == "til" and grp[j].kind == "til":
                        continue
                    msg = _interferes(grp[i], grp[j])
                    if msg:
                        viol.append(Violation(t, grp[j].step if grp[j].kind != "til" else None, "mutex",
                                              "simultaneous happenings interfere on {}".format(msg)))
            # conditions in the state before the happening
            for h in grp:
                if h.kind == "start":
                    for f in h.action.cond_start:
                        check(f, (h.step, "start"), t, "at-start")
                elif h.kind == "end":
                    for f in h.action.cond_end:
                        check(f, (h.step, "end"), t, "at-end")
            # apply: deletes, then adds
            adds = []
            for h in grp:
                if h.kind == "til":
                    tag = ("til", h.step)
                    (adds.append((h.til.fact, tag)) if h.til.value else self._delete(st, h.til.fact, t, history, tag))
                else:
                    a = h.action
                    dels, ads = (a.del_start, a.add_start) if h.kind == "start" else (a.del_end, a.add_end)
                    for f in dels:
                        self._delete(st, f, t, history, (h.step, h.kind))
                    adds += [(f, (h.step, h.kind)) for f in ads]
            for f, tag in adds:
                if f not in st.facts:
                    st.facts[f] = Interval(f, t, UNBOUNDED, tag)
            # running set
            st.running = [r for r in st.running if not any(h.kind == "end" and h.step == r.step for h in grp)]
            for h in grp:
                if h.kind == "start":
                    st.running.append(Running(h.step, h.action, t, t + h.action.duration))
                    for f in h.action.cond_over:
                        if f in st.facts:
                            links.append(CausalLink(st.facts[f].added_by, (h.step, "over"), f))
            st.time = t
            # invariants of running actions must hold after this happening
            for r in st.running:
                for f in r.action.cond_over:
                    if f not in st.facts:
                        viol.append(Violation(t, r.step, "invariant",
                                              "{} needs {} over all, false at {:.3f}".format(r.action, " ".join(f), t)))
            for g in task.goals:
                if g in st.facts and g not in goal_at:
                    goal_at[g] = t
        for g in task.goals:
            if g not in st.facts:
                viol.append(Violation(st.time, None, "goal", "{} not achieved".format(" ".join(g))))
        history += list(st.facts.values())
        makespan = max((s + a.duration for s, a in steps), default=0.0)
        return SimResult(not viol, viol, makespan, st, history, links, goal_at, steps)

    @staticmethod
    def _delete(st, f, t, history, by):
        iv = st.facts.pop(f, None)
        if iv is not None:
            history.append(Interval(iv.fact, iv.added, t, iv.added_by, by))


def bind_plan(gt, plan) -> Tuple[List[Tuple[float, GroundAction]], List[float], List[str]]:
    """TemporalPlan (parsed planner output) -> ground actions of GroundTask gt.
    Returns (steps, stated durations, unknown action strings)."""
    if not hasattr(gt, "_by_key"):
        gt._by_key = {a.key: a for a in gt.actions}
    steps, durs, unknown = [], [], []
    for ta in plan.actions:
        a = gt._by_key.get((ta.name,) + tuple(ta.args))
        if a is None:
            unknown.append("({})".format(" ".join((ta.name,) + tuple(ta.args))))
            continue
        steps.append((ta.start, a))
        durs.append(ta.duration)
    return steps, durs, unknown


def concurrent_ok(a: GroundAction, sa: float, b: GroundAction, sb: float, eps=EPS) -> Optional[str]:
    """Can a (at sa) and b (at sb) overlap without violating each other's
    invariants or interfering at coinciding happenings? None if yes, else why."""
    ea, eb = sa + a.duration, sb + b.duration
    if ea <= sb + eps or eb <= sa + eps:
        return None
    probs = []
    # an effect of one inside the other's open interval must not delete its invariant
    for (x, sx, ex), (y, sy, ey) in (((a, sa, ea), (b, sb, eb)), ((b, sb, eb), (a, sa, ea))):
        if sy < sx < ey and x.del_start & y.cond_over:
            probs.append("{} start deletes invariant of {}".format(x, y))
        if sy < ex < ey and x.del_end & y.cond_over:
            probs.append("{} end deletes invariant of {}".format(x, y))
    return "; ".join(probs) or None


# ==================== Plan analysis ==============================================================
def resource_predicates(spec) -> Dict[str, str]:
    """Unary predicates deleted by some action effect and added by some action
    effect: {predicate: argument type}. Domain-level, no per-domain rules."""
    dels, adds = set(), set()
    for act in spec.actions.values():
        for pt, atoms in act.effects.items():
            for a in atoms:
                (dels if a.negated else adds).add(a.pred)
    return {p: sig[0].type for p, sig in spec.predicates.items()
            if len(sig) == 1 and p in dels and p in adds}


def resource_types(spec) -> Set[str]:
    return set(resource_predicates(spec).values())


@dataclass
class ScheduleAnalysis:
    makespan: float
    total_action_time: float
    parallelism: float
    slack: List[float]                    # per step
    critical_steps: List[int]
    crit: Dict[str, float]                # object -> 1 - min slack / C_max over its actions
    on_critical_path: Set[str]
    utilization: Dict[str, float]         # resource object -> busy time / C_max
    waits: Dict[str, float]               # resource object -> total time actions waited for it
    critical_waits: Dict[str, float]      # resource object -> waits of critical actions
    critical_resource_types: Dict[str, float]  # type -> critical wait time

    def summary(self):
        return {"makespan": self.makespan, "parallelism": round(self.parallelism, 3),
                "critical_actions": len(self.critical_steps),
                "critical_resource_types": {k: round(v, 3) for k, v in self.critical_resource_types.items()}}


def analyze(task, sim, tol=10 * EPS) -> ScheduleAnalysis:
    steps = sim.steps
    n = len(steps)
    C = sim.makespan
    S = [s for s, _ in steps]
    D = [a.duration for _, a in steps]
    off = lambda k, p: D[k] if p == "end" else 0.0  # noqa: E731

    # edges (k1, p1) -> (k2, p2): S_k2 + off2 >= S_k1 + off1 + eps
    edges = []
    for l in sim.links:
        if l.producer is None or l.producer[0] == "til":
            continue
        edges.append((l.producer[0], l.producer[1], l.consumer[0], l.consumer[1]))
    deleter = {(iv.fact, iv.added_by): iv.deleted_by for iv in sim.intervals if iv.deleted_by is not None}
    for l in sim.links:
        if l.consumer[1] in ("over", "end"):
            by = deleter.get((l.fact, l.producer))
            if by is not None and by[0] != "til" and by[0] != l.consumer[0]:
                edges.append((l.consumer[0], "end", by[0], by[1]))

    # deadline windows: facts deleted by a timed literal
    window = {}
    for t in task.tils:
        if not t.value:
            window[t.fact] = min(window.get(t.fact, float("inf")), t.time)
    LS = [C - D[k] for k in range(n)]
    for k, (_, a) in enumerate(steps):
        for f in a.cond_start:
            if f in window:
                LS[k] = min(LS[k], window[f] - EPS)
        for f in a.cond_over | a.cond_end:
            if f in window:
                LS[k] = min(LS[k], window[f] - EPS - D[k])
    for _ in range(n + 1):
        changed = False
        for k1, p1, k2, p2 in edges:
            bound = LS[k2] + off(k2, p2) - off(k1, p1) - EPS
            if bound < LS[k1] - 1e-9:
                LS[k1] = bound
                changed = True
        if not changed:
            break
    slack = [max(0.0, LS[k] - S[k]) for k in range(n)]
    critical = [k for k in range(n) if slack[k] <= tol]

    crit, on_cp = {}, set()
    for k, (_, a) in enumerate(steps):
        c = 1.0 - min(1.0, slack[k] / C) if C > 0 else 1.0
        for o in a.objects:
            crit[o] = max(crit.get(o, 0.0), c)
            if k in critical:
                on_cp.add(o)

    # resources
    rp = resource_predicates(task.spec)
    present = defaultdict(float)
    for iv in sim.intervals:
        if iv.fact[0] in rp:
            end = C if iv.open_ended else min(iv.deleted, C)
            present[iv.fact[1]] += max(0.0, end - min(iv.added, C))
    res_objs = {o for o in task.objects if task.type_of(o) in set(rp.values())
                or any(task.is_a(o, ty) for ty in rp.values())}
    util = {}
    for o in res_objs:
        tokens = sum(1 for p, ty in rp.items() if task.is_a(o, ty))
        util[o] = max(0.0, 1.0 - present[o] / (C * max(1, tokens))) if C > 0 else 0.0

    # resource waits: binding producer of each start is a resource token released by another action
    by_consumer = defaultdict(list)
    for l in sim.links:
        if l.consumer[1] in ("start", "over"):
            by_consumer[l.consumer[0]].append(l)
    waits, cwaits = defaultdict(float), defaultdict(float)
    for k in range(n):
        ls = by_consumer.get(k, [])
        times = []
        for l in ls:
            if l.producer is None:
                times.append((0.0, l))
            elif l.producer[0] == "til":
                times.append((task.tils[l.producer[1]].time, l))
            else:
                times.append((S[l.producer[0]] + off(l.producer[0], l.producer[1]), l))
        if not times:
            continue
        t_bind, l_bind = max(times, key=lambda x: x[0])
        if l_bind.fact[0] in rp and l_bind.producer not in (None,) and l_bind.producer[0] != "til":
            others = [t for t, l in times if l.fact[0] not in rp]
            ready = max(others, default=0.0)
            w = max(0.0, t_bind - ready)
            if w > EPS * 2:
                o = l_bind.fact[1]
                waits[o] += w
                if k in critical:
                    cwaits[o] += w
    ctypes = defaultdict(float)
    for o, w in cwaits.items():
        ctypes[task.type_of(o)] += w
    tat = sum(D)
    return ScheduleAnalysis(C, tat, tat / C if C > 0 else 0.0, slack, critical, crit, on_cp,
                            util, dict(waits), dict(cwaits), dict(ctypes))


# ==================== Re-timing ==================================================================
def retime(task, gt, plan, eps=EPS):
    steps, durs, unknown = bind_plan(gt, plan)
    if unknown or not steps:
        return None
    sim = Simulator(task).run(steps, durs)
    if not sim.valid:
        return None
    n = len(steps)
    D = [a.duration for _, a in steps]

    def o(k, p):
        return D[k] if p == "end" else 0.0

    lower = [0.0] * n
    edges = []          # (k1, p1, k2, p2): S_k2 + o2 >= S_k1 + o1 + eps
    for l in sim.links:
        k2, p2 = l.consumer
        if l.producer is None:
            continue
        if l.producer[0] == "til":
            t = task.tils[l.producer[1]].time
            lower[k2] = max(lower[k2], t + eps - o(k2, p2))
            continue
        edges.append((l.producer[0], l.producer[1], k2, p2))

    # happenings touching each fact
    touch = defaultdict(list)        # fact -> [(time, k, point, role)]
    for k, (s, a) in enumerate(steps):
        for pt, pre, add, dele in (("start", a.cond_start, a.add_start, a.del_start),
                                   ("end", a.cond_end, a.add_end, a.del_end)):
            t = s + (a.duration if pt == "end" else 0.0)
            for f in pre:
                touch[f].append((t, k, pt, "pre"))
            for f in add:
                touch[f].append((t, k, pt, "add"))
            for f in dele:
                touch[f].append((t, k, pt, "del"))
    for f, hs in touch.items():
        hs.sort()
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                t1, k1, p1, r1 = hs[i]
                t2, k2, p2, r2 = hs[j]
                if k1 == k2:
                    continue
                if r1 == "del" or r2 == "del":
                    edges.append((k1, p1, k2, p2))
    # invariants
    deleters = defaultdict(list)
    for f, hs in touch.items():
        for t, k, p, r in hs:
            if r == "del":
                deleters[f].append((t, k, p))
    for k, (s, a) in enumerate(steps):
        e = s + a.duration
        for f in a.cond_over:
            for t, k2, p2 in deleters.get(f, ()):
                if k2 == k:
                    continue
                if t <= s + eps / 2:
                    edges.append((k2, p2, k, "start"))
                elif t >= e - eps / 2:
                    edges.append((k, "end", k2, p2))
    # timed literals that delete / add facts actions touch: keep the side
    for tl in task.tils:
        for t, k, p, r in touch.get(tl.fact, ()):
            if t > tl.time:
                lower[k] = max(lower[k], tl.time + eps - o(k, p))

    S = list(lower)
    for _ in range(n + 2):
        changed = False
        for k1, p1, k2, p2 in edges:
            need = S[k1] + o(k1, p1) + eps - o(k2, p2)
            if need > S[k2] + 1e-9:
                S[k2] = need
                changed = True
        if not changed:
            break
    else:
        return None                      # cyclic constraints: should not happen for a valid plan
    acts = [TimedAction(round(S[k], 3), a.name, a.args, a.duration) for k, (_, a) in enumerate(steps)]
    return TemporalPlan(sorted(acts, key=lambda x: (x.start, x.name, x.args)))
