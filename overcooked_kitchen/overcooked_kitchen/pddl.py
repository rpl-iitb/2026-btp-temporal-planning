"""PDDL bridge: the catalogue domain, export of a live simulator state, planners (POPF / OPTIC), VAL, plan execution.

export(sim) encodes the simulator's current state as a problem of domain.yaml, time shifted so now = 0:
  * held items -> holding-* (no hand-empty); plates by place and state; a plate with a diner -> timed literals
    returning it dirty to its drop-off;
  * a cooking recipe pot -> on-stove + timed soup-ready; a ready pot -> on-stove + soup-ready; the stove timer's
    latest end (burn - 1 after ready) removes on-stove (a burn deadline); a soup that can no longer be dished up in
    time -> the pot turns dirty when it burns; a cooking non-recipe pot turns dirty when it spoils; burnt / spoiled /
    dirty pots -> pot-dirty;
  * partial chop / wash / scrub progress -> not started (full strokes: pessimistic, the domain is unchanged);
  * a running task -> its cook not idle, its target tile occupied (on the grid clock its current tile too, until
    its end); what its effect deletes is false now, what it adds (with at, idle) holds from its nominal end;
  * deadlines shifted to now; an order past or unable to meet its deadline gets none (annotations
    "dropped_deadlines"); hard_deadlines=False drops all deadlines.
PlanPolicy runs a plan in Sim: every move / turn of a cook is a core move task, every interact the matching legal
task, in the plan's order per cook; tiles are handed over and stations used across cooks in the plan's order (see
PlanPolicy).
"""
"""This file is UN-VERIFIED"""
import copy
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from overcooked_kitchen.layout import DIR_BY_NAME, DIR_NAME, INGREDIENTS, PDDL_TYPE, PROVIDES, pose_name, tile_name
from overcooked_kitchen.pddlkit import (Goal, HLProblem, Obj, TemporalPlan, TimedLiteral, load_domain_spec,
                                        planners, write_domain, write_problem)
from overcooked_kitchen.scenarios import recipe_name
from overcooked_kitchen.tasks import WAIT, stations_of
from overcooked_kitchen.world import cook_time

DOMAIN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "domain.yaml")
BENCHMARK = "overcooked_kitchen"
NO_BURN = 100000
EPS = 1e-6
_SPEC = []


def spec():
    if not _SPEC:
        _SPEC.append(load_domain_spec(DOMAIN_FILE))
    return _SPEC[0]


def domain_text() -> str:
    return write_domain(spec())


def problem_text(problem) -> str:
    return write_problem(spec(), problem)


# ---- export ---------------------------------------------------------------------------------
def _hand(c, held):
    if held is None:
        return ("hand-empty", c)
    if held[0] == "raw":
        return ("holding-raw", c, held[1])
    if held[0] == "chopped":
        return ("holding-chopped", c, held[1])
    return ("holding-plate", c, held[1])


def _pot(w, name, pot, recipes, t):
    """(facts, timed) of a pot; timed = {(absolute time, fact, value)}."""
    if pot.state in ("dirty", "burnt", "spoiled"):
        return {("pot-dirty", name)}, set()
    if pot.state == "idle":
        n = pot.comp()
        return {("pot-idle", name), ("pot-has", name) + tuple("n{}".format(k) for k in n)}, set()
    r = recipe_name(pot.comp())
    if not pot.recipe or r not in recipes:
        return set(), {(pot.ready_at, ("pot-dirty", name), True)}
    burn = w.params.get("burn")
    last = None if burn is None else pot.ready_at + burn - 1
    if last is not None and last < max(pot.ready_at, t) + 1:
        return set(), {(pot.ready_at + burn, ("pot-dirty", name), True)}
    facts, timed = {("on-stove", name, r)}, set()
    if pot.state == "ready":
        facts.add(("soup-ready", name, r))
    else:
        timed.add((pot.ready_at, ("soup-ready", name, r), True))
    if last is not None:
        timed.add((last, ("on-stove", name, r), False))
    return facts, timed


def _state(w, cooks, st_name, recipes, t):
    """Dynamic facts of a world (surfaces, pots, sinks, plates, hands, orders) and its timed literals."""
    facts, timed = set(), set()
    for cell, name in st_name.items():
        k = w.kind[cell]
        if k in ("counter", "board"):
            item = w.surface[cell]
            if item is None:
                facts.add(("clear", name))
            elif item[0] == "raw":
                facts.add(("on-raw", name, item[1]))
            elif item[0] == "chopped":
                facts.add(("on-chopped", name, item[1]))
            else:
                facts.add(("plate-at", item[1], name))
        elif k == "sink":
            if w.sink[cell] is None:
                facts.add(("clear", name))
            else:
                facts.add(("plate-at", w.sink[cell], name))
        elif k in ("rack", "dropoff"):
            facts |= {("plate-at", pid, name) for pid in w.stack[cell]}
        elif k == "pot":
            f, tl = _pot(w, name, w.pots[cell], recipes, t)
            facts |= f
            timed |= tl
    for c, pl in zip(cooks, w.players):
        facts.add(_hand(c, pl.held))
    for pid, pl in w.plates.items():
        if pl.state == "clean":
            facts.add(("plate-clean", pid))
        elif pl.state == "dirty":
            facts.add(("plate-dirty", pid))
        elif pl.state == "soup" and recipe_name(pl.soup) in recipes:
            facts.add(("plate-soup", pid, recipe_name(pl.soup)))
    for pid, (ret, drop) in w.diners.items():
        timed |= {(ret, ("plate-at", pid, st_name[drop]), True), (ret, ("plate-dirty", pid), True)}
    for o in w.orders:
        facts |= {("served", o["id"])} if o["id"] in w.served else {("pending", o["id"]), ("open", o["id"])}
    return facts, timed


def _nominal_end(sim, cook, r):
    """Absolute nominal end of a running task: start + nominal duration (grid: remaining walk + strokes)."""
    t = sim.t
    if sim.clock.stepped:
        if r.end is not None:
            return max(r.end, t + 1)
        p = sim.world.players[sim.cooks.index(cook)]
        walk = sim.world.lay.bfs_poses((p.pos[0], p.pos[1], tuple(p.facing))).get(r.task.pose, 0)
        return t + max(1, walk + r.task.strokes - r.done)
    return max(r.start + r.task.duration, t + 1)


def _projected(sim, cook, r, end):
    """Copy of the world with the running task's pending effect applied at end - 1 (nominal)."""
    w = sim.world
    memo = {id(w.lay): w.lay}
    cache = w.__dict__.get("_stations_cache")
    if cache is not None:
        memo[id(cache)] = cache
    wc = copy.deepcopy(w, memo)
    task, i = r.task, sim.cooks.index(cook)
    if not r.applied and task.event != "move" and wc.interact_event(i, task.pose) == task.event:
        wc.apply(i, task.event, task.cell, end - 1, max(1, task.strokes - r.done))
    return wc


def _earliest(w, comp, t):
    """Lower bound on the time from now to serve a dish of comp (travel ignored)."""
    comp = tuple(comp)
    if any(pl.state == "soup" and tuple(pl.soup) == comp for pl in w.plates.values()):
        return 1
    ready = [pot.ready_at for pot in w.pots.values() if pot.state in ("cooking", "ready") and pot.recipe
             and pot.comp() == comp]
    if ready:
        return max(min(ready) - t, 0) + 2
    return cook_time(w.params, comp) + 2


def export(sim, hard_deadlines=True) -> HLProblem:
    w, t = sim.world, sim.t
    P = w.params
    mx = P["max_ingredients"]
    counts = ["n{}".format(k) for k in range(mx + 1)]
    stations = stations_of(w, sim.counters)
    st_name = {s.cell: s.name for s in stations}
    objects, static, fluents = [], set(), {}

    recipes = {}
    for o in w.orders:
        name = recipe_name(o["comp"])
        if name in recipes:
            continue
        if not 1 <= sum(o["comp"]) <= mx:
            raise ValueError("dish {} has {} ingredients (1..{} allowed)".format(o["dish"], sum(o["comp"]), mx))
        ct = cook_time(P, o["comp"])
        recipes[name] = {"dish": o["dish"], "comp": list(o["comp"]), "cook_time": ct, "burn": P.get("burn")}
        objects.append(Obj(name, "recipe", dict(recipes[name])))
        static.add(("recipe-of", name) + tuple(counts[n] for n in o["comp"]))
        fluents[("cook-steps", name)] = ct - 1
        fluents[("burn-window", name)] = ct + (P["burn"] if P.get("burn") is not None else NO_BURN) - 1
    for k, c in enumerate(counts):
        objects.append(Obj(c, "count", {"value": k}))
    static.add(("is-zero", counts[0]))
    for k in range(mx):
        static.add(("next", counts[k], counts[k + 1]))
    for a in range(mx + 1):
        for b in range(mx + 1 - a):
            for c in range(mx + 1 - a - b):
                static.add(("fits", counts[a], counts[b], counts[c]))
    for i in INGREDIENTS:
        fluents[("chop-steps", i)] = P["chop"][i]

    poses = {}
    drops = [s for s in stations if s.kind == "dropoff"]
    for s in stations:
        objects.append(Obj(s.name, PDDL_TYPE[s.kind], {"kind": s.kind, "cell": list(s.cell)}))
        if s.kind in PROVIDES:
            static.add(("provides", s.name, PROVIDES[s.kind]))
        for p in w.lay.access_poses(s.cell):
            poses.setdefault(p, pose_name(p))
            static.add(("faces", poses[p], s.name))
        if s.kind == "pot":
            fluents[("scrub-steps", s.name)] = P["scrub_pot"]
        elif s.kind == "sink":
            fluents[("wash-steps", s.name)] = P["wash_plate"]
        elif s.kind == "serving":
            fluents[("eat-steps", s.name)] = P["eat"]
            if drops:
                d = min(drops, key=lambda d: (abs(d.cell[0] - s.cell[0]) + abs(d.cell[1] - s.cell[1]), d.cell))
                static.add(("returns-to", s.name, d.name))
    station_poses = set(poses)

    # dynamic state now, and each running task's effect projected to its nominal end
    facts, timed = _state(w, sim.cooks, st_name, recipes, t)
    deleted, running, served_soon = set(), {}, set()
    for c in sim.cooks:
        r = sim.running.get(c)
        if r is None:
            continue
        end = _nominal_end(sim, c, r)
        after, timed_after = _state(_projected(sim, c, r, end), sim.cooks, st_name, recipes, t)
        deleted |= facts - after
        timed |= timed_after | {(end, f, True) for f in after - facts}
        served_soon |= {f[1] for f in after - facts if f[0] == "served"}
        running[c] = (r, end)
    init = (facts - deleted) | static

    # cooks and tiles
    occupied, cook_attrs = set(), {}
    for k, c in enumerate(sim.cooks):
        pl = w.players[k]
        here = (pl.pos[0], pl.pos[1], tuple(pl.facing))
        if c not in running:
            poses.setdefault(here, pose_name(here))
            init |= {("at", c, poses[here]), ("idle", c)}
            occupied.add(here[:2])
            cook_attrs[c] = {"player_index": k, "pose": poses[here]}
            continue
        r, end = running[c]
        target = r.task.pose
        poses.setdefault(target, pose_name(target))
        timed |= {(end, ("at", c, poses[target]), True), (end, ("idle", c), True)}
        occupied.add(target[:2])
        if sim.clock.stepped and here[:2] != target[:2] and any(p[:2] == here[:2] for p in poses):
            occupied.add(here[:2])
            timed.add((end, ("free", tile_name(here[:2])), True))
        cook_attrs[c] = {"player_index": k, "pose": poses[target], "running": r.task.action,
                         "station": r.task.station, "until": round(end - t, 3)}
    for c in sim.cooks:
        objects.append(Obj(c, "cook", cook_attrs[c]))

    tiles = sorted({(x, y) for x, y, _ in poses})
    for tl in tiles:
        objects.append(Obj(tile_name(tl), "tile", {"cell": list(tl)}))
        if tl not in occupied:
            init.add(("free", tile_name(tl)))
    for p, n in sorted(poses.items(), key=lambda kv: kv[1]):
        objects.append(Obj(n, "pose", {"tile": tile_name(p[:2]), "cell": [p[0], p[1]], "facing": DIR_NAME[p[2]]}))
        init.add(("pose-tile", n, tile_name(p[:2])))
    for p, n in poses.items():
        dist = w.lay.bfs_poses(p)
        for q in station_poses:
            if q == p or q not in dist:
                continue
            init.add(("connected", n, poses[q]) if p[:2] != q[:2] else ("same-tile", n, poses[q]))
            fluents[("travel", n, poses[q])] = dist[q]

    for pid, pl in sorted(w.plates.items()):
        objects.append(Obj(pid, "plate", {"state": pl.state}))

    goals, dropped = [], []
    for o in w.orders:
        r = recipe_name(o["comp"])
        objects.append(Obj(o["id"], "order", {"recipe": r, "dish": o["dish"], "deadline": o["deadline"]}))
        init.add(("order-recipe", o["id"], r))
        if o["id"] in w.served or o["id"] in served_soon:
            continue
        deadline = None
        if hard_deadlines and o["deadline"] is not None:
            rel = round(o["deadline"] - t, 3)
            if rel >= _earliest(w, o["comp"], t):
                deadline = rel
            else:
                dropped.append(o["id"])
        goals.append(Goal(("served", o["id"]), deadline))

    tls = [TimedLiteral(round(at - t, 3), f, v) for at, f, v in sorted(timed, key=lambda x: (x[0], x[1], x[2]))]
    return HLProblem(
        instance_id="{}_t{}".format(sim.scenario["scenario_id"], str(t).replace(".", "_")), benchmark=BENCHMARK,
        domain_name=spec().name, objects=objects, facts=sorted(init), fluents=fluents, goals=goals,
        timed_literals=tls,
        annotations={"grid": list(w.lay.rows), "sim_params": P, "cooks": list(sim.cooks), "recipes": recipes,
                     "counter_policy": sim.counters, "t_now": t, "clock": sim.clock.name,
                     "dropped_deadlines": dropped})


def write(sim, out_dir, hard_deadlines=True) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    d, p = os.path.join(out_dir, "domain.pddl"), os.path.join(out_dir, "problem.pddl")
    with open(d, "w") as f:
        f.write(domain_text())
    with open(p, "w") as f:
        f.write(problem_text(export(sim, hard_deadlines)))
    return d, p


# ---- planners -------------------------------------------------------------------------------
def run_planner(problem, planner="popf", timeout=60.0) -> planners.PlannerResult:
    """Run a planner on an export (written to a temporary directory); raises FileNotFoundError without a binary."""
    if not problem.goals:
        return planners.PlannerResult("solved", TemporalPlan([]), 0.0, None, 0, planner, [])
    with tempfile.TemporaryDirectory() as tmp:
        d, p = os.path.join(tmp, "domain.pddl"), os.path.join(tmp, "problem.pddl")
        with open(d, "w") as f:
            f.write(domain_text())
        with open(p, "w") as f:
            f.write(problem_text(problem))
        return planners.run_temporal_planner(d, p, timeout=timeout, planner=planner)


def solve(sim, planner="popf", timeout=60.0, problem=None) -> Optional[TemporalPlan]:
    """Plan from the sim's current state (or the given export); None if no binary, timeout or no plan."""
    if not planners.available(planner):
        return None
    res = run_planner(export(sim) if problem is None else problem, planner, timeout)
    return res.plan if res.status == "solved" else None


def validate(domain_path, problem_path, plan) -> bool:
    return planners.validate_plan(domain_path, problem_path, plan).valid


# ---- plan execution -------------------------------------------------------------------------
@dataclass
class Step:
    cook: str
    action: str
    kind: str                   # "move" (move / turn: a core move task) or "interact" (a legal task)
    station: Optional[str]
    event: Optional[str]
    pose: tuple                 # move: target pose; interact: planned pose (x, y, direction)
    src: Optional[tuple]        # move: tile left (None for a turn)
    start: float                # planned start, absolute
    end: float                  # planned end, absolute
    key: tuple = ()             # (planned start, index in the plan): a total order of the plan
    preds: list = field(default_factory=list, repr=False)
    task: object = None
    sim_start: Optional[float] = None
    sim_end: Optional[float] = None
    failed: bool = False

    @property
    def tile(self):
        return self.pose[:2]


def _next_event(sim) -> Optional[float]:
    """Next decision point known now: a running task's end (nominal when unknown) or a world timer."""
    times = [r.end if r.end is not None else _nominal_end(sim, c, r) for c, r in sim.running.items()]
    times.append(sim.world.next_timer())
    times = [x for x in times if x is not None]
    return min(times) if times else None


def _pose_of(name, objs) -> tuple:
    """(x, y, direction) of a pose object, or parsed from its name p<x>_<y><n|s|e|w>."""
    if name in objs and "cell" in objs[name].attrs:
        at = objs[name].attrs
        return at["cell"][0], at["cell"][1], DIR_BY_NAME[at["facing"]]
    x, rest = name[1:].split("_")
    return int(x), int(rest[:-1]), DIR_BY_NAME[rest[-1]]


class PlanPolicy:
    """Policy for Sim.run that follows a temporal plan made at problem.annotations["t_now"].

    Every plan action of a cook becomes a step, run in the plan's order: a move or turn is a core move task to its
    target pose (Sim.move_task), an interact is the matching legal task. Cross-cook order: a step entering a tile
    waits until every earlier move of another cook leaving that tile has started (the tile is handed over as in the
    plan); an interact waits until every earlier interact of another cook at the same station has ended. Mode
    "timed": a step is also due only at its planned start plus the current offset (actual - planned start of the
    latest started step); it goes early when no decision point comes before that time. include_idle: Sim.run offers
    decisions to cooks whose only legal choice is WAIT (they may have a move to make)."""

    include_idle = True

    def __init__(self, plan: TemporalPlan, problem: HLProblem, mode="asap"):
        if mode not in ("asap", "timed"):
            raise ValueError("unknown mode '{}'".format(mode))
        self.plan, self.problem, self.mode = plan, problem, mode
        t0 = problem.annotations.get("t_now", 0)
        objs = {o.name: o for o in problem.objects}
        sp = spec()
        self.queues = {}
        self.steps: List[Step] = []
        order = sorted(range(len(plan.actions)), key=lambda k: (plan.actions[k].start, k))
        for k in order:
            a = plan.actions[k]
            act = sp.actions[a.name]
            b = act.sim or {}
            if b.get("kind") not in ("navigate", "interact"):
                continue
            arg = dict(zip(act.param_names, a.args))
            cook = arg[b["actor"]]
            if b["kind"] == "navigate":
                pose, src = _pose_of(arg[b["to"]], objs), None
                if "?from" in arg:
                    here = _pose_of(arg["?from"], objs)[:2]
                    src = here if here != pose[:2] else None
                s = Step(cook, a.name, "move", None, None, pose, src, t0 + a.start, t0 + a.end)
            else:
                s = Step(cook, a.name, "interact", arg[b["station"]], b["event"], _pose_of(arg[b["pose"]], objs),
                         None, t0 + a.start, t0 + a.end)
            s.key = (a.start, k)
            self.queues.setdefault(cook, []).append(s)
            self.steps.append(s)
        for s in self.steps:
            if s.kind == "move" and s.src is not None:
                s.preds = [u for u in self.steps if u.cook != s.cook and u.kind == "move" and u.src == s.tile
                           and u.key < s.key]
            elif s.kind == "interact":
                s.preds = [u for u in self.steps if u.cook != s.cook and u.kind == "interact"
                           and u.station == s.station and u.key < s.key]
        self.active = {}
        self.offset = 0.0
        self._seen = 0

    def sync(self, sim):
        """Record the ends of the started steps from the sim's log."""
        for e in sim.log[self._seen:]:
            s = self.active.get(e.get("cook"))
            if s is not None and e["type"] in ("end", "task_failed") and e["action"] == s.task.action:
                s.sim_end, s.failed = e["t"], e["type"] == "task_failed"
                del self.active[s.cook]
        self._seen = len(sim.log)

    @property
    def done(self) -> bool:
        return not self.active and not any(self.queues.values())

    def _due(self, sim, planned) -> bool:
        if self.mode == "asap":
            return True
        gate = planned + self.offset
        if sim.t >= gate - EPS:
            return True
        nxt = _next_event(sim)
        return nxt is None or nxt > gate + EPS

    @staticmethod
    def _released(s) -> bool:
        if s.kind == "move":
            return all(u.sim_start is not None for u in s.preds)
        return all(u.sim_end is not None for u in s.preds)

    def __call__(self, sim, cook, legal):
        self.sync(sim)
        q = self.queues.get(cook)
        while q and self._released(q[0]):
            s = q[0]
            if s.kind == "move":
                p = sim.world.players[sim.cooks.index(cook)]
                if (p.pos[0], p.pos[1], tuple(p.facing)) == s.pose:
                    q.pop(0)
                    s.sim_start = s.sim_end = sim.t
                    continue
                task = sim.move_task(cook, pose=s.pose) if self._due(sim, s.start) else None
            else:
                task = next((x for x in legal if x != WAIT and x.station == s.station and x.event == s.event
                             and x.action == s.action), None)
                task = task if task is not None and self._due(sim, s.start) else None
            if task is None:
                return WAIT
            q.pop(0)
            s.task, s.sim_start = task, sim.t
            self.active[cook] = s
            self.offset = sim.t - s.start
            return task
        return WAIT


class PlannerPolicy:
    """POPF baseline: plan at the first decision (one retry at the first decision after time has advanced, if that
    fails) and follow the plan.
    With replan=True, re-solve from export(sim) when a task ends more than tol steps later than planned or fails, or
    once when the plan runs out with orders pending; at most max_replans re-solves. A failed re-solve keeps the old
    plan and backs off: no new solve until another task ends late. solver(problem, planner, timeout) ->
    PlannerResult defaults to run_planner."""

    include_idle = True

    def __init__(self, planner="popf", timeout=300.0, mode="asap", replan=False, tol=3, max_replans=10,
                 hard_deadlines=True, solver=None):
        self.planner, self.timeout, self.mode = planner, timeout, mode
        self.replan, self.tol, self.max_replans, self.hard_deadlines = replan, tol, max_replans, hard_deadlines
        self.solver = solver
        self.policy: Optional[PlanPolicy] = None
        self.solves = []
        self.attempts = 0
        self._attempt_t = None
        self.replans = 0
        self._checked = set()
        self._backoff = False
        self._exhausted = False

    def _solve(self, sim) -> bool:
        problem = export(sim, self.hard_deadlines)
        plan, status, runtime = None, "no_binary", 0.0
        if self.solver is not None or planners.available(self.planner):
            res = (self.solver or run_planner)(problem, self.planner, self.timeout)
            plan = res.plan if res.status == "solved" else None
            status, runtime = res.status, round(res.runtime_s, 3)
        self.solves.append({"t": sim.t, "status": status, "runtime_s": runtime})
        if plan is None:
            return False
        self.policy = PlanPolicy(plan, problem, self.mode)
        self._checked, self._backoff, self._exhausted = set(), False, False
        return True

    def _off_plan(self, sim) -> bool:
        pol = self.policy
        pol.sync(sim)
        late = False
        for k, s in enumerate(pol.steps):
            if s.sim_end is None or k in self._checked:
                continue
            self._checked.add(k)
            late = late or s.failed or s.sim_end - s.end > self.tol
        if late:
            return True
        if self._backoff or self._exhausted or not pol.done or sim.success:
            return False
        self._exhausted = True
        return True

    def __call__(self, sim, cook, legal):
        if self.policy is None:
            if self.attempts < 2 and (self._attempt_t is None or sim.t > self._attempt_t):
                self.attempts += 1
                self._attempt_t = sim.t
                self._solve(sim)
        elif self.replan and self.replans < self.max_replans and self._off_plan(sim):
            self.replans += 1
            self._backoff = not self._solve(sim)
        if self.policy is None:
            return WAIT
        return self.policy(sim, cook, legal)
