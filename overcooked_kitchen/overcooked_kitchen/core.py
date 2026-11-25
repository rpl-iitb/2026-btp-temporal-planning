"""Core engine: task-level decision loop over the kitchen world.

Cooks start compound tasks (catalogue actions); a clock says when each ends. A task's interact effect is its own
event at end - 1, interleaved with the world's timers: applied if still possible, else the task fails softly.
Effect points are not decision points (decisions see an effect from the task's end on). Pots, burning and diners
run by themselves through the world's timers. Decision loop: next_decision() names a cook, act() starts its choice.
On the grid clock time moves one step at a time (tick); idle cooks in the way yield, driven cooks take grid actions.
A policy may also start a core move task (move_task) to stand at a pose ahead of a task that is not legal yet.
pending() offers decisions to free cooks with a choice besides WAIT; pending(include_idle=True) (used by run() for
policies with include_idle = True, e.g. plan execution) also offers cooks whose only legal choice is WAIT, so that
they can start a move task, and offers a cook that chose WAIT again after another task started at the same time.
"""
"""This file is UN-VERIFIED"""
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from overcooked_kitchen import grid
from overcooked_kitchen.clocks import GridClock, Noise, SampledClock, round_t
from overcooked_kitchen.scenarios import example, load_scenario, make_world
from overcooked_kitchen.tasks import WAIT, Task, legal_tasks, stations_of

DEFAULT_HORIZON = 10000
MODES = ("auto", "task", "drive")


@dataclass
class RunningTask:
    cook: str
    task: Task
    start: float
    end: Optional[float]
    effect: Optional[float] = None
    applied: bool = False
    found: Optional[str] = None
    done: int = 0


def resolve_scenario(scenario) -> dict:
    """A scenario dict from a dict (copied), a .yaml path or an example name."""
    if isinstance(scenario, dict):
        return load_scenario(scenario)
    if os.path.isfile(str(scenario)):
        return load_scenario(scenario)
    return load_scenario(example(str(scenario)))


def _noise(noise, cv_scale):
    if noise is None:
        return None
    if isinstance(noise, str):
        if noise != "default":
            raise ValueError("unknown noise '{}'".format(noise))
        return Noise.default_profile(cv_scale)
    return noise.scaled(cv_scale)


class Sim:
    def __init__(self, scenario, clock="sampled", seed=0, noise=None, cv_scale=1.0, horizon=None, record=False,
                 stuck_steps=50, step_rate=None):
        self.scenario = resolve_scenario(scenario)
        self.world = make_world(self.scenario)
        self.counters = self.scenario.get("counters", "none") or "none"
        self.rng = np.random.default_rng(seed)
        self.noise = _noise(noise, cv_scale)
        if clock not in ("sampled", "grid"):
            raise ValueError("unknown clock '{}'".format(clock))
        self.clock = (GridClock if clock == "grid" else SampledClock)(self.noise, self.rng)
        self.horizon = DEFAULT_HORIZON if horizon is None else horizon
        self.record, self.stuck_steps, self.step_rate = record, stuck_steps, step_rate
        self.t = 0
        self.cooks: List[str] = ["c{}".format(k + 1) for k in range(len(self.world.players))]
        self._idx = {c: k for k, c in enumerate(self.cooks)}
        self.mode = {c: "auto" for c in self.cooks}
        self._stuck = 0
        self.running: Dict[str, RunningTask] = {}
        self.log: List[dict] = []
        self.frames: List[dict] = []
        self.failure: Optional[str] = None
        self.decided = set()
        self._waited: Dict[str, int] = {}
        self._starts = 0
        self.decisions = 0
        self.busy = {c: 0 for c in self.cooks}
        self.blocked = {c: 0 for c in self.cooks}
        self._legal = {}
        self._seen = self._world_marks()
        self._closed = False
        self._wall0 = time.perf_counter()
        self._wall = None

    # ---- decisions ----------------------------------------------------------------------
    def _taken_tiles(self, cook) -> frozenset:
        """Tiles of the other cooks: running tasks' targets, idle cooks' positions."""
        out = set()
        for c in self.cooks:
            if c == cook:
                continue
            if c in self.running:
                out.add(self.running[c].task.pose[:2])
            else:
                out.add(tuple(self.world.players[self._idx[c]].pos))
        return frozenset(out)

    def legal(self, cook) -> List:
        if cook in self.running or self.mode[cook] == "drive":
            return [WAIT]
        if cook not in self._legal:
            locked = frozenset(r.task.cell for c, r in self.running.items() if c != cook and r.task.cell is not None)
            self._legal[cook] = [WAIT] + legal_tasks(self.world, self._idx[cook], locked, self._taken_tiles(cook),
                                                     self.counters)
        return list(self._legal[cook])

    def move_task(self, cook, station=None, pose=None) -> Optional[Task]:
        """A core move task for a free cook: walk to `pose`, or to the nearest free reachable access pose of
        `station` (by name). None if the cook is busy or driven, already there, or no such pose is reachable and free.
        Moves are not offered by legal(); act() accepts the task returned here. A move locks no station."""
        if cook in self.running or self.mode[cook] == "drive" or (station is None) == (pose is None):
            return None
        w = self.world
        i = self._idx[cook]
        p = w.players[i]
        here = (p.pos[0], p.pos[1], tuple(p.facing))
        if pose is not None:
            cands = [(pose[0], pose[1], tuple(pose[2]))]
        else:
            cells = [st.cell for st in stations_of(w, self.counters) if st.name == station]
            cands = w.lay.access_poses(cells[0]) if cells else []
        dist = w.lay.bfs_poses(here)
        taken = self._taken_tiles(cook)
        cands = [q for q in cands if q in dist and q[:2] not in taken]
        if not cands or here in cands:
            return None
        best = min(cands, key=lambda q: (dist[q], q))
        return Task(i, None, None, best, "move", "move", 0, dist[best])

    def act(self, cook, choice):
        if cook not in self._idx:
            raise ValueError("unknown cook {!r}; cooks: {}".format(cook, self.cooks))
        if cook in self.running:
            raise ValueError("{} is busy until t={}".format(cook, self.running[cook].end))
        if isinstance(choice, Task) and choice.event == "move":
            if choice != self.move_task(cook, pose=choice.pose):
                raise ValueError("illegal move {} for {} at t={}".format(choice, cook, self.t))
        elif choice != WAIT and choice not in self.legal(cook):
            raise ValueError("illegal task {} for {} at t={}".format(choice, cook, self.t))
        self.decisions += 1
        self.decided.add(cook)
        if choice == WAIT:
            self._waited[cook] = self._starts
            return
        self._start(cook, choice)

    def _start(self, cook, task, kind="start", **extra):
        end = self.clock.start(self, self._idx[cook], task)
        self._starts += 1
        self.running[cook] = RunningTask(cook, task, self.t, end, self.clock.effect_time(end))
        self._legal.clear()
        entry = {"type": kind, "t": self.t, "cook": cook, "action": task.action, "station": task.station,
                 "cell": task.cell, "event": task.event, "end": end}
        entry.update(extra)
        self.log.append(entry)

    def pending(self, include_idle=False) -> List[str]:
        """Free cooks (not running, not driven) to decide for now: not yet decided at this decision point and with a
        choice besides WAIT. include_idle: also cooks with only WAIT, and a cook that chose WAIT is offered again once
        another task has started since."""
        out = []
        for c in self.cooks:
            if c in self.running or self.mode[c] == "drive":
                continue
            if not include_idle:
                if c not in self.decided and len(self.legal(c)) > 1:
                    out.append(c)
            elif c not in self.decided or self._waited.get(c, self._starts) < self._starts:
                out.append(c)
        return out

    @property
    def success(self) -> bool:
        return not self.world.pending_orders()

    def next_decision(self, include_idle=False) -> Optional[str]:
        while not self.success and self.failure is None:
            p = self.pending(include_idle)
            if p:
                return p[0]
            if self.clock.stepped and (self.running or self.clock.queued):
                self.tick()
                continue
            times = [x for x in (self.clock.next_time(self), self.world.next_timer()) if x is not None]
            eff = self.clock.next_effect(self)
            if eff is None and not times and "drive" in self.mode.values():
                break
            if eff is None and not times:
                busy = any(len(self.legal(c)) > 1 for c in self.cooks if c not in self.running)
                self.failure = "stalled" if busy else "deadlock"
                break
            if eff is not None and (not times or eff < min(times)):
                if eff > self.horizon:
                    self.failure = "horizon"
                    break
                self._effects(eff)
                if self.success:
                    self._advance(max(self.world.served.values()))
                continue
            t = min(times)
            if self.clock.stepped:
                t, self._stuck = math.ceil(t), 0
                self.clock.near.clear()
                self.clock.head_on.clear()
            if t > self.horizon:
                self.failure = "horizon"
                break
            self._advance(t)
        self._close()
        return None

    def run(self, policy, max_decisions=10 ** 6, include_idle=None) -> dict:
        """Decision loop; include_idle defaults to the policy's include_idle attribute (False if absent)."""
        if include_idle is None:
            include_idle = getattr(policy, "include_idle", False)
        for _ in range(max_decisions):
            c = self.next_decision(include_idle)
            if c is None:
                break
            self.act(c, policy(self, c, self.legal(c)))
        return self.metrics()

    # ---- time -----------------------------------------------------------------------------
    def _advance(self, t):
        """Decision point t (a task end or a world timer); effects before t are already applied."""
        self._frame()
        self.world.advance(t)
        for c in self.clock.finished(self, t):
            self._end(c, t)
        self.t = t
        self._sync_log()
        self._clear_decided()
        self._legal.clear()

    def _effects(self, e):
        """Effect point e: world timers up to e fire first, then the due effects in cook order (no decisions)."""
        self.world.advance(max(self.world.t, e))
        self._sync_log()
        for c in self.clock.effects_due(self, e):
            self._effect(c)
        self._sync_log()
        self._legal.clear()
        self.decided.update(c for c in self.cooks if c not in self.running)

    def _effect(self, cook):
        r = self.running[cook]
        task, i = r.task, self._idx[cook]
        p = self.world.players[i]
        if task.event == "move":
            r.applied, r.found = True, "move"
            return
        r.applied, r.found = True, self.world.interact_event(i, task.pose)
        if r.found != task.event:
            return
        ta = r.effect
        pid = p.held[1] if task.event == "deliver" else None
        self.world.apply(i, task.event, task.cell, ta, task.strokes)
        self._timers_after(task.event, task.cell, ta, pid)

    def _timers_after(self, event, cell, ta, pid=None):
        """Cook and eat times drawn by the clock after a start / deliver at ta (nominal keeps the world's)."""
        if event == "start":
            pot = self.world.pots[cell]
            pot.ready_at = round_t(ta + self.clock.sample("cook", pot.ready_at - ta))
        elif pid in self.world.diners:
            ret, drop = self.world.diners[pid]
            self.world.diners[pid] = (round_t(ta + 1 + self.clock.sample("eat", ret - ta - 1)), drop)

    def _end(self, cook, t):
        r = self.running.pop(cook)
        task = r.task
        if not self.clock.stepped:
            p = self.world.players[self._idx[cook]]
            p.pos, p.facing = (task.pose[0], task.pose[1]), task.pose[2]
        self.busy[cook] += t - r.start
        entry = {"type": "end", "t": t, "cook": cook, "action": task.action, "station": task.station,
                 "cell": task.cell, "event": task.event}
        if r.found != task.event:
            entry.update(type="task_failed", found=r.found)
        self.log.append(entry)

    # ---- grid clock: steps, modes, driving -------------------------------------------------
    def tick(self) -> None:
        """One grid step: running tasks walk / stroke, driven cooks take their queued action; no decisions."""
        if not self.clock.stepped:
            raise ValueError("tick() needs clock='grid'")
        if self.success or self.failure is not None:
            return
        if self.t + 1 > self.horizon:
            self.failure = "horizon"
            return
        self._frame()
        had, before = bool(self.running), self._progress()
        driven = self.clock.step(self)
        self.t += 1
        ended = self.clock.finished(self, self.t)
        for c in ended:
            self._end(c, self.t)
        for c, (ev, cell) in driven.items():
            self.log.append({"type": "drive", "t": self.t - 1, "cook": c, "event": ev, "cell": cell})
        for c, other in self.clock.asides:
            self.log.append({"type": "step_aside", "t": self.t - 1, "cook": c, "for": other})
        seen = self._seen
        self._sync_log()
        for cook, task, other in self.clock.yields(self):
            self._start(cook, task, "yield", pose=task.pose, **{"for": other})
        driving = "drive" in self.mode.values()
        self._stuck = self._stuck + 1 if had and not driving and self._progress() == before else 0
        if self.stuck_steps is not None and self._stuck >= self.stuck_steps:
            self.failure = "deadlock"
        if ended or driven or seen != self._seen:
            self._clear_decided()
        self._legal.clear()

    def _clear_decided(self):
        self.decided.clear()
        self._waited.clear()

    def set_mode(self, cook, mode):
        """auto / task: the cook is offered decisions (pending); drive: grid actions via drive() (grid clock)."""
        if cook not in self._idx:
            raise ValueError("unknown cook {!r}; cooks: {}".format(cook, self.cooks))
        if mode not in MODES:
            raise ValueError("unknown mode '{}'".format(mode))
        if mode == "drive" and not self.clock.stepped:
            raise ValueError("drive mode needs clock='grid'")
        if mode == "drive" and cook in self.running:
            r = self.running.pop(cook)
            self.busy[cook] += self.t - r.start
            self.log.append({"type": "abort", "t": self.t, "cook": cook, "action": r.task.action,
                             "station": r.task.station, "cell": r.task.cell, "event": r.task.event})
        if self.clock.stepped:
            self.clock.queued.pop(cook, None)
        self.mode[cook] = mode
        self.decided.discard(cook)
        self._waited.pop(cook, None)
        self._legal.clear()

    def drive(self, cook, action):
        """Queue one grid action (north, south, east, west, stay, interact) for the cook's next step."""
        if not self.clock.stepped:
            raise ValueError("drive() needs clock='grid'")
        if self.mode[cook] != "drive":
            raise ValueError("{} is not in drive mode".format(cook))
        if action not in grid.ACTIONS:
            raise ValueError("unknown grid action {!r}".format(action))
        self.clock.queued[cook] = action

    def _progress(self):
        w = self.world
        return ([(tuple(p.pos), p.held) for p in w.players], dict(w.progress),
                [(p.state, len(p.contents), p.scrub) for p in w.pots.values()])

    # ---- logs and frames ----------------------------------------------------------------
    def _world_marks(self):
        w = self.world
        return (len(w.burnt), len(w.spoiled), dict(w.diners), {c: p.state for c, p in w.pots.items()})

    def _sync_log(self):
        w = self.world
        nb, ns, diners, states = self._seen
        for when, cell, comp in w.burnt[nb:]:
            self.log.append({"type": "burnt", "t": when, "cell": cell, "comp": comp})
        for when, cell, comp in w.spoiled[ns:]:
            self.log.append({"type": "spoiled", "t": when, "cell": cell, "comp": comp})
        for c, pot in w.pots.items():
            if states[c] == "cooking" and pot.state in ("ready", "burnt"):
                self.log.append({"type": "soup_ready", "t": pot.ready_at, "cell": c, "comp": pot.comp()})
        for pid in sorted(set(diners) - set(w.diners)):
            self.log.append({"type": "plate_returned", "t": diners[pid][0], "plate": pid})
        self._seen = self._world_marks()

    def _frame(self):
        if not self.record:
            return
        run = {c: {"action": r.task.action, "station": r.task.station, "cell": r.task.cell, "pose": r.task.pose,
                   "start": r.start, "end": r.end} for c, r in self.running.items()}
        self.frames.append({"t": self.t, "world": self.world.snapshot(), "running": run})

    def _close(self):
        if self._closed:
            return
        self._closed = True
        self._frame()
        self._wall = time.perf_counter() - self._wall0

    # ---- metrics --------------------------------------------------------------------------
    def metrics(self) -> dict:
        w = self.world
        T = self.t
        n = len(w.orders)
        on_time = sum(1 for o in w.orders if o["id"] in w.served
                      and (o.get("deadline") is None or w.served[o["id"]] <= o["deadline"]))
        busy = {c: self.busy[c] + (T - self.running[c].start if c in self.running else 0) for c in self.cooks}
        wall = self._wall if self._wall is not None else time.perf_counter() - self._wall0
        return {
            "success": self.success,
            "failure": self.failure,
            "makespan": max(w.served.values(), default=0) if self.success else None,
            "served": len(w.served),
            "late": len(w.late),
            "total_lateness": round_t(sum(w.late.values())),
            "on_time": on_time / n if n else 1.0,
            "burnt": len(w.burnt),
            "spoiled": len(w.spoiled),
            "decisions": self.decisions,
            "cook_utilisation": {c: round(busy[c] / T, 4) if T > 0 else 0.0 for c in self.cooks},
            "idle": {c: round_t(T - busy[c] - self.blocked[c]) for c in self.cooks},
            "blocked": dict(self.blocked),
            "wall_s": round(wall, 4),
        }
