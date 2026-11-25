"""Time sources for the core: lognormal duration noise, the sampled clock and the grid clock.

A clock decides when a started task ends and when its interact effect happens (end - 1).
The sampled clock draws durations; the grid clock walks and strokes each task on the step simulator.
"""
"""This file is UN-VERIFIED"""
import collections
import math
from typing import Dict, List, Optional

import numpy as np

from overcooked_kitchen import grid
from overcooked_kitchen.layout import DIRS
from overcooked_kitchen.tasks import Task

CATEGORIES = ("move", "chop", "wash", "scrub", "interact", "cook", "eat")
PROFILE = {"move": 0.25, "chop": 0.2, "wash": 0.2, "scrub": 0.2, "interact": 0.1, "cook": 0.1, "eat": 0.3}


def round_t(x):
    return round(x, 3) if isinstance(x, float) else x


class Noise:
    """Lognormal multiplicative noise with mean 1 per category: d = nominal * X, E[X] = 1, CV[X] = cv."""

    def __init__(self, cv: Dict[str, float] = None, default: float = 0.0, clip=(0.4, 3.0)):
        self.cv, self.default, self.clip = dict(cv or {}), default, clip

    @classmethod
    def default_profile(cls, scale=1.0):
        return cls({k: v * scale for k, v in PROFILE.items()})

    def scaled(self, k):
        return Noise({a: c * k for a, c in self.cv.items()}, self.default * k, self.clip)

    def sample(self, category, nominal, rng):
        cv = self.cv.get(category, self.default)
        if cv <= 0 or nominal <= 0:
            return nominal
        s2 = math.log(1 + cv * cv)
        x = rng.lognormal(-s2 / 2, math.sqrt(s2))
        return round(nominal * float(np.clip(x, *self.clip)), 3)


def category(event):
    return event if event in ("chop", "wash", "scrub") else "interact"


class SampledClock:
    """Each task's end is drawn when it starts: move(travel) + category(event)(strokes), at least one step."""

    name = "sampled"
    stepped = False

    def __init__(self, noise: Optional[Noise] = None, rng=None):
        self.noise, self.rng = noise, rng

    def sample(self, cat, nominal):
        if self.noise is None:
            return nominal
        return self.noise.sample(cat, nominal, self.rng)

    def start(self, sim, i, task):
        d = self.sample("move", task.travel) + self.sample(category(task.event), task.strokes)
        return round_t(sim.t + max(1, d))

    def effect_time(self, end):
        return round_t(end - 1)

    def next_time(self, sim) -> Optional[float]:
        ends = [r.end for r in sim.running.values()]
        return min(ends) if ends else None

    def next_effect(self, sim) -> Optional[float]:
        effects = [r.effect for r in sim.running.values() if not r.applied]
        return min(effects) if effects else None

    def effects_due(self, sim, t) -> List[str]:
        return [c for c in sim.cooks if c in sim.running and not sim.running[c].applied and sim.running[c].effect <= t]

    def finished(self, sim, t) -> List[str]:
        return [c for c in sim.cooks if c in sim.running and sim.running[c].end <= t]


def _pose(p):
    return (p.pos[0], p.pos[1], tuple(p.facing))


def arrived(lay, pose, target) -> bool:
    """A move reaches target on its tile, facing it unless it faces floor (no turning towards floor in place)."""
    if pose[:2] != target[:2]:
        return False
    return pose[2] == target[2] or lay.is_floor((target[0] + target[2][0], target[1] + target[2][1]))


def path_tiles(lay, src, dst) -> List[tuple]:
    """Tiles entered on a shortest path src -> dst that ignores the other cooks."""
    tiles, cur = [], src
    while True:
        d = lay.next_step(cur, dst)
        if d is None:
            return tiles
        n = (cur[0] + d[0], cur[1] + d[1])
        if lay.is_floor(n):
            cur = (n[0], n[1], d)
            tiles.append(n)
        else:
            cur = (cur[0], cur[1], d)


class GridClock(SampledClock):
    """Each running task walks to its pose (replanned around the other cooks every step), then sends INTERACT
    while its event holds until its strokes are done: effect at that step t, end at t + 1. A move task ends at the
    step it arrives (turning to face its station if needed). Time is integer steps.
    Noise (when given) only draws cook and eat times."""

    name = "grid"
    stepped = True

    def __init__(self, noise: Optional[Noise] = None, rng=None):
        super().__init__(noise, rng)
        self.queued: Dict[str, tuple] = {}
        self.near: Dict[str, int] = {}
        self._wants: Dict[str, List[tuple]] = {}
        self.give_way: Dict[str, str] = {}
        self.head_on: Dict[tuple, int] = {}
        self.asides: List[tuple] = []
        self.last_actions: List[tuple] = []

    def start(self, sim, i, task):
        return None

    def effect_time(self, end):
        return None

    def next_time(self, sim) -> Optional[float]:
        return None

    def next_effect(self, sim) -> Optional[float]:
        return None

    def effects_due(self, sim, t) -> List[str]:
        return []

    def finished(self, sim, t) -> List[str]:
        return [c for c in sim.cooks if c in sim.running and sim.running[c].end is not None and sim.running[c].end <= t]

    def step(self, sim) -> Dict[str, tuple]:
        """One grid step t -> t + 1; returns {cook: (event, cell)} for the driven cooks' interacts.

        Paths avoid the other cooks' tiles and the tiles already claimed this step by earlier cooks (cook order),
        so two walkers never livelock on the same free tile. Two walkers blocking each other for 2 steps: one
        steps aside (part of its walk) and gives way until the other has passed."""
        w, lay, t = sim.world, sim.world.lay, sim.t
        n = len(sim.cooks)
        acts, expected = [grid.STAY] * n, [None] * n
        tiles = [tuple(p.pos) for p in w.players]
        moving, stroking, stuck = {}, {}, []
        self._wants, self.asides = {}, []
        for c in sim.cooks:
            i = sim._idx[c]
            if sim.mode[c] == "drive":
                acts[i] = self.queued.pop(c, grid.STAY)
                continue
            r = sim.running.get(c)
            if r is None or r.applied:
                self.give_way.pop(c, None)
                continue
            pose = _pose(w.players[i])
            if r.task.event == "move" and arrived(lay, pose, r.task.pose):
                self._done(r, t, "move")
                continue
            if pose != r.task.pose:
                others = frozenset(tiles[j] for j in range(n) if j != i) | frozenset(moving.values())
                others |= self._passing(sim, c)
                d = lay.next_step(pose, r.task.pose, others)
                if d is None:
                    d = lay.next_step(pose, r.task.pose)
                    nxt = (pose[0] + d[0], pose[1] + d[1]) if d is not None else None
                    if d is None or nxt in others:
                        stuck.append(c)
                        self._wants[c] = path_tiles(lay, pose, r.task.pose)
                        continue
                acts[i] = d
                nxt = (pose[0] + d[0], pose[1] + d[1])
                if lay.is_floor(nxt):
                    moving[c] = nxt
                continue
            found = w.interact_event(i)
            if found != r.task.event:
                self._done(r, t, found)
                continue
            acts[i], expected[i] = grid.INTERACT, r.task.event
            stroking[c] = r
        for c, d, tile, other in self._step_aside(sim, tiles, moving):
            acts[sim._idx[c]], moving[c] = d, tile
            stuck.remove(c)
            self._wants.pop(c)
            self.give_way[c] = other
            self.asides.append((c, other))
        for c in stuck:
            self._blocked(sim, c)
        held = [p.held for p in w.players]
        self.last_actions = list(acts)
        events = grid.step(w, acts, expected)
        driven = {}
        for c in sim.cooks:
            i = sim._idx[c]
            ev = events[i]
            if ev is not None:
                cell = w.facing_cell(i)
                sim._timers_after(ev, cell, t, held[i][1] if ev == "deliver" else None)
                if c not in stroking:
                    driven[c] = (ev, cell)
            if c in stroking:
                r = stroking[c]
                if ev is None:
                    self._done(r, t, w.interact_event(i))
                    continue
                r.done += 1
                if r.done >= r.task.strokes:
                    self._done(r, t, ev)
            elif c in moving and tuple(w.players[i].pos) != moving[c]:
                self._blocked(sim, c)
            elif c in sim.running and not sim.running[c].applied and sim.running[c].task.event == "move" \
                    and arrived(lay, _pose(w.players[i]), sim.running[c].task.pose):
                self._done(sim.running[c], t, "move")
        return driven

    def _passing(self, sim, cook) -> frozenset:
        """Tiles a cook giving way keeps clear: the other walker's tile and its remaining path."""
        other = self.give_way.get(cook)
        r = sim.running.get(other) if other is not None else None
        if r is None or r.applied or _pose(sim.world.players[sim._idx[other]]) == r.task.pose:
            self.give_way.pop(cook, None)
            return frozenset()
        pose = _pose(sim.world.players[sim._idx[other]])
        return frozenset(path_tiles(sim.world.lay, pose, r.task.pose)) | {pose[:2]}

    def _step_aside(self, sim, tiles, moving) -> List[tuple]:
        """(cook, direction, tile, other) for each pair of walkers blocking each other for 2 consecutive steps:
        the higher-index cook steps to a free adjacent floor tile off the other's path, else the lower one."""
        occ = {tile: c for c, tile in zip(sim.cooks, tiles)}
        pairs = {}
        for a, path in self._wants.items():
            b = occ.get(path[0]) if path else None
            if b is not None and b in self._wants and self._wants[b] and self._wants[b][0] == tiles[sim._idx[a]]:
                pairs[tuple(sorted((a, b), key=sim.cooks.index))] = True
        self.head_on = {p: self.head_on.get(p, 0) + 1 for p in pairs}
        out, taken = [], set(moving.values())
        for (lo, hi), k in self.head_on.items():
            if k < 2:
                continue
            for c, other in ((hi, lo), (lo, hi)):
                x, y = tiles[sim._idx[c]]
                free = [(d, (x + d[0], y + d[1])) for d in DIRS]
                free = [(d, nb) for d, nb in free if sim.world.lay.is_floor(nb) and nb not in occ
                        and nb not in taken and nb not in self._wants[other]]
                if free:
                    out.append((c, free[0][0], free[0][1], other))
                    taken.add(free[0][1])
                    break
        for c, _, _, _ in out:
            self.head_on = {p: k for p, k in self.head_on.items() if c not in p}
        return out


    def yields(self, sim) -> List[tuple]:
        """(cook, move task, for) for every idle cook on the next tile of a blocked cook's path 2 steps running."""
        occ = {tuple(p.pos): c for c, p in zip(sim.cooks, sim.world.players)}
        hits = {}
        for c, path in self._wants.items():
            j = occ.get(path[0]) if path else None
            if j is not None and j not in sim.running and sim.mode[j] != "drive" and j not in hits:
                hits[j] = (c, path)
        self.near = {j: self.near.get(j, 0) + 1 for j in hits}
        out = []
        for j, (c, path) in hits.items():
            if self.near[j] < 2:
                continue
            task = self._refuge(sim, j, set(path))
            if task is not None:
                self.near.pop(j)
                out.append((j, task, c))
        return out

    def _refuge(self, sim, cook, avoid) -> Optional[Task]:
        """Move task to the nearest free floor tile off `avoid` and off every running task's target tile."""
        w, lay = sim.world, sim.world.lay
        i = sim._idx[cook]
        src = tuple(w.players[i].pos)
        occupied = {tuple(p.pos) for p in w.players}
        reserved = {r.task.pose[:2] for r in sim.running.values()}
        seen, q = {src}, collections.deque([src])
        while q:
            x, y = q.popleft()
            for d in DIRS:
                nb = (x + d[0], y + d[1])
                if nb in seen or not lay.is_floor(nb) or nb in occupied:
                    continue
                seen.add(nb)
                if nb not in avoid and nb not in reserved:
                    dist = lay.bfs_poses(_pose(w.players[i]))
                    pose = min((p for p in dist if p[:2] == nb), key=lambda p: (dist[p], DIRS.index(p[2])))
                    return Task(i, None, None, pose, "move", "move", 0, dist[pose])
                q.append(nb)
        return None

    @staticmethod
    def _done(r, t, found):
        r.applied, r.found, r.effect, r.end = True, found, t, t + 1

    @staticmethod
    def _blocked(sim, cook):
        """A step a running cook wanted to move but could not: blocked, not busy."""
        sim.blocked[cook] += 1
        sim.busy[cook] -= 1
