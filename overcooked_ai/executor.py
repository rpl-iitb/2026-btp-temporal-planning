"""Execute a temporal plan in Overcooked-AI 1.1.0 (modes timed / asap) and measure timing deviation."""

"""This file is UN-VERIFIED"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from .layout import Layout, DIRS

DIR_BY_NAME = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0)}


@dataclass
class Step:
    action: object                   # pddl_generator.TimedAction
    kind: str                        # navigate | interact
    cook: str
    pose: tuple                      # (x, y, dir) to be at
    station: Optional[str]
    release: int                     # earliest sim step (timed mode)
    sim_start: Optional[int] = None
    sim_end: Optional[int] = None
    preds: list = field(default_factory=list, repr=False)


@dataclass
class ExecutionReport:
    success: bool                    # every goal order delivered
    deadlines_met: bool
    sim_makespan: Optional[int]      # sim step at which the last dispatched action finished
    plan_makespan: float
    served_at: dict
    deadlines: dict
    steps_run: int
    sim_reward: int
    max_abs_deviation: Optional[int]  # |sim end - planned end| over dispatched actions
    mean_deviation: Optional[float]
    blocked_steps: int               # cook wanted to move but had to wait for another cook
    readiness_waits: int             # cook at its pose waiting for a sim precondition
    order_waits: int                 # step held back by a plan precedence on a shared tile / station
    detour_steps: int                # steps taken off the shortest path to get around another cook
    sidesteps: int
    stalls_injected: int
    unfinished_steps: int
    trace: List[dict] = field(default_factory=list)

    def summary(self) -> dict:
        d = dict(self.__dict__)
        d.pop("trace")
        return d


class PlanExecutor:

    def __init__(self, problem, plan, spec, mode="timed", max_steps=None, stall_prob=0.0, seed=0):
        if mode not in ("timed", "asap"):
            raise ValueError(mode)
        ann = problem.annotations
        self.problem, self.plan, self.spec, self.mode = problem, plan, spec, mode
        self.mdp = OvercookedGridworld.from_grid(list(ann["grid"]), base_layout_params=dict(ann["sim_params"]))
        self.state = self.mdp.get_standard_start_state()
        self.lay = Layout(ann["grid"])
        objs = {o.name: o for o in problem.objects}
        self.pose = {n: (o.attrs["cell"][0], o.attrs["cell"][1], DIR_BY_NAME[o.attrs["facing"]])
                     for n, o in objs.items() if o.type == "pose"}
        self.cell = {n: tuple(o.attrs["cell"]) for n, o in objs.items() if "kind" in o.attrs}
        self.count_value = {n: o.attrs["value"] for n, o in objs.items() if o.type == "count"}
        self.cooks = sorted((o.attrs["player_index"], n) for n, o in objs.items() if o.type == "cook")
        self.cook_names = [n for _, n in self.cooks]
        t0 = ann.get("t_now", 0)
        self.deadlines = {g.fact[1]: (None if g.deadline is None else g.deadline) for g in problem.goals
                          if g.fact[0] == "served"}
        self.t0 = t0
        self.rng = random.Random(seed)
        self.stall_prob = stall_prob
        self.max_steps = max_steps or int(math.ceil(plan.makespan)) * 3 + 60
        self.queues = {c: [] for c in self.cook_names}
        self.steps: List[Step] = []
        for a in plan.actions:
            act = spec.actions[a.name]
            b = act.sim or {}
            if b.get("kind") not in ("navigate", "interact"):
                continue
            arg = dict(zip(act.param_names, a.args))
            cook = arg[b["actor"]]
            pose = self.pose[arg[b["to"] if b["kind"] == "navigate" else b["pose"]]]
            st = Step(a, b["kind"], cook, pose, arg.get(b.get("station")), int(round(a.start)))
            self.queues[cook].append(st)
            self.steps.append(st)
        for q in self.queues.values():
            q.sort(key=lambda s: s.action.start)
        # Plan precedences between cooks on shared resources: a step waits until
        # every earlier-planned step of another cook on the same tile or the same
        # station has finished. Acyclic, because it follows planned start times.
        for s in self.steps:
            s.preds = [u for u in self.steps if u.cook != s.cook and u.action.start < s.action.start - 1e-6
                       and (u.pose[:2] == s.pose[:2] or (s.station is not None and u.station == s.station))]
        self.blocked = {c: 0 for c in self.cook_names}
        self.stats = {"blocked": 0, "ready_wait": 0, "order_wait": 0, "detour": 0, "sidestep": 0, "stall": 0}

    # ---- simulator views ----------------------------------------------------------
    def _player(self, cook):
        return self.state.players[self.cook_names.index(cook)]

    def _cur_pose(self, cook):
        p = self._player(cook)
        return (p.position[0], p.position[1], tuple(p.orientation))

    def _held(self, cook):
        o = self._player(cook).held_object
        return None if o is None else o.name

    def _obj_at(self, station):
        c = self.cell[station]
        return self.state.get_object(c) if self.state.has_object(c) else None

    def _ready(self, s: Step) -> bool:
        name, held = s.action.name, self._held(s.cook)
        arg = dict(zip(self.spec.actions[name].param_names, s.action.args))
        o = self._obj_at(s.station) if s.station else None
        if name == "take":
            return held is None
        if name in ("add-onion", "add-tomato"):
            ing = "onion" if name == "add-onion" else "tomato"
            return held == ing and (o is None or (o.is_idle and not o.is_full))
        if name == "start-cooking":
            want = (self.count_value[arg["?no"]], self.count_value[arg["?nt"]])
            return held is None and o is not None and o.is_idle and \
                (o.ingredients.count("onion"), o.ingredients.count("tomato")) == want
        if name == "pick-soup":
            return held == "dish" and o is not None and o.is_ready
        if name == "deliver":
            return held == "soup"
        if name == "put-down":
            return held == arg["?i"] and o is None
        if name == "put-down-soup":
            return held == "soup" and o is None
        if name == "pick-up":
            return held is None and o is not None and o.name == arg["?i"]
        if name == "pick-up-soup":
            return held is None and o is not None and o.name == "soup"
        raise KeyError("no readiness rule for {}".format(name))

    def _effect_done(self, s: Step, held_before, obj_before) -> bool:
        name, held = s.action.name, self._held(s.cook)
        o = self._obj_at(s.station) if s.station else None
        if name == "take":
            return held is not None and held_before is None
        if name in ("add-onion", "add-tomato", "put-down", "put-down-soup", "deliver"):
            return held is None and held_before is not None
        if name == "start-cooking":
            return o is not None and not o.is_idle
        if name == "pick-soup":
            return held == "soup"
        if name in ("pick-up", "pick-up-soup"):
            return held is not None and held_before is None
        return False

    # ---- control ------------------------------------------------------------------------
    def _head(self, cook, t):
        q = self.queues[cook]
        while q:
            s = q[0]
            if self.mode == "timed" and t < s.release:
                return None
            if any(u.sim_end is None for u in s.preds):
                self.stats["order_wait"] += 1
                return None
            if s.sim_start is None:
                s.sim_start = t
            if s.kind == "navigate" and self._cur_pose(cook) == s.pose:
                s.sim_end = t          # already there (e.g. after a detour)
                q.pop(0)
                continue
            return s
        return None

    def _toward(self, cook, pose, others):
        cur = self._cur_pose(cook)
        free_path = self.lay.next_step(cur, pose)
        d = self.lay.next_step(cur, pose, blocked=frozenset(others)) or free_path
        if d is None:
            return Action.STAY
        if free_path is not None and self.lay.bfs_poses(self._after(cur, d)).get(pose, 10 ** 9) + 1 > \
                self.lay.bfs_poses(cur)[pose]:
            self.stats["detour"] += 1      # another cook sits on every shortest path
        return d

    def _after(self, pose, d):
        n = (pose[0] + d[0], pose[1] + d[1])
        return (n[0], n[1], d) if self.lay.is_floor(n) else (pose[0], pose[1], d)

    def _desires(self, t):
        acts, steps = {}, {}
        pos = {c: tuple(self._player(c).position) for c in self.cook_names}
        for c in self.cook_names:
            others = {p for k, p in pos.items() if k != c}
            s = self._head(c, t)
            steps[c] = s
            if s is None:
                acts[c] = Action.STAY
            elif self._cur_pose(c) != s.pose:
                acts[c] = self._toward(c, s.pose, others)
            elif s.kind == "interact" and self._ready(s):
                acts[c] = Action.INTERACT
            else:
                acts[c] = Action.STAY
                if s.kind == "interact":
                    self.stats["ready_wait"] += 1
        return acts, steps, pos

    def _next_pos(self, c, a, pos):
        if a in DIRS:
            n = (pos[c][0] + a[0], pos[c][1] + a[1])
            return n if self.lay.is_floor(n) else pos[c]
        return pos[c]

    def _resolve(self, acts, steps, pos):
        """Make the joint action collision-free (simulator semantics)."""
        prio = {c: (steps[c].action.start if steps[c] else math.inf, i) for i, c in enumerate(self.cook_names)}
        for _ in range(len(self.cook_names) ** 2 + 1):
            new = {c: self._next_pos(c, acts[c], pos) for c in self.cook_names}
            clash = None
            for i, a in enumerate(self.cook_names):
                for b in self.cook_names[i + 1:]:
                    if new[a] == new[b] or (new[a] == pos[b] and new[b] == pos[a]):
                        clash = (a, b)
                        break
                if clash:
                    break
            if clash is None:
                return acts
            a, b = clash
            moving = [c for c in clash if new[c] != pos[c]]
            loser = moving[0] if len(moving) == 1 else max(clash, key=lambda c: prio[c])
            acts[loser] = Action.STAY
            self.blocked[loser] += 1
            self.stats["blocked"] += 1
        return {c: Action.STAY for c in self.cook_names}

    def _unblock(self, acts, steps, pos):
        """A cook in someone's way that is not moving steps aside; a cook blocked
        for several steps yields too (symmetry broken by cook index)."""
        targets = {steps[c].pose[:2] for c in self.cook_names if steps[c] is not None}
        for i, c in enumerate(self.cook_names):
            idle_in_way = acts[c] == Action.STAY and pos[c] in targets and \
                (steps[c] is None or steps[c].pose[:2] != pos[c])
            stuck = self.blocked[c] >= 3 + i
            if not (idle_in_way or stuck):
                continue
            occupied = set(pos.values())
            free = [d for d in DIRS if self.lay.is_floor((pos[c][0] + d[0], pos[c][1] + d[1]))
                    and (pos[c][0] + d[0], pos[c][1] + d[1]) not in occupied
                    and (pos[c][0] + d[0], pos[c][1] + d[1]) not in targets]
            if free:
                acts[c] = self.rng.choice(free)
                self.blocked[c] = 0
                self.stats["sidestep"] += 1
        return acts

    def run(self) -> ExecutionReport:
        served, reward, t = {}, 0, 0
        for t in range(self.max_steps):
            if all(not q for q in self.queues.values()):
                break
            acts, steps, pos = self._desires(t)
            acts = self._unblock(acts, steps, pos)
            acts = self._resolve(acts, steps, pos)
            if self.stall_prob > 0:
                for c in self.cook_names:
                    if self.rng.random() < self.stall_prob:
                        acts[c] = Action.STAY
                        self.stats["stall"] += 1
            before = {c: (self._held(c), self._obj_at(steps[c].station) if steps[c] and steps[c].station else None)
                      for c in self.cook_names}
            joint = tuple(acts[c] for c in self.cook_names)
            self.state, info = self.mdp.get_state_transition(self.state, joint)
            reward += sum(info["sparse_reward_by_agent"])
            for i, c in enumerate(self.cook_names):
                s = steps[c]
                if s is None:
                    continue
                if acts[c] in DIRS and self._next_pos(c, acts[c], pos) != pos[c]:
                    self.blocked[c] = 0
                done = False
                if s.kind == "navigate":
                    done = self._cur_pose(c) == s.pose
                elif acts[c] == Action.INTERACT:
                    done = self._effect_done(s, *before[c])
                if done:
                    s.sim_end = t + 1
                    self.queues[c].pop(0)
                    if s.action.name == "deliver":
                        o = dict(zip(self.spec.actions["deliver"].param_names, s.action.args))["?o"]
                        served[o] = t + 1
        else:
            t = self.max_steps
        ends = [s.sim_end for s in self.steps if s.sim_end is not None]
        devs = [s.sim_end - int(round(s.action.end)) for s in self.steps if s.sim_end is not None]
        goals = set(self.deadlines)
        success = goals <= set(served)
        on_time = success and all(self.deadlines[o] is None or served[o] <= self.deadlines[o] for o in goals)
        return ExecutionReport(
            success=success, deadlines_met=on_time, sim_makespan=max(ends) if ends else None,
            plan_makespan=self.plan.makespan, served_at=served, deadlines=self.deadlines,
            steps_run=t, sim_reward=reward,
            max_abs_deviation=max((abs(d) for d in devs), default=None),
            mean_deviation=(sum(devs) / len(devs)) if devs else None,
            blocked_steps=self.stats["blocked"], readiness_waits=self.stats["ready_wait"],
            order_waits=self.stats["order_wait"],
            detour_steps=self.stats["detour"], sidesteps=self.stats["sidestep"], stalls_injected=self.stats["stall"],
            unfinished_steps=sum(len(q) for q in self.queues.values()),
            trace=[{"cook": s.cook, "action": s.action.name, "args": list(s.action.args),
                    "planned_start": s.action.start, "planned_end": s.action.end,
                    "sim_start": s.sim_start, "sim_end": s.sim_end} for s in self.steps])
