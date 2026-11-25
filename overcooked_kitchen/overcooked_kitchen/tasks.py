"""Compound tasks: one catalogue action by one cook at a station, with travel and strokes.

legal_tasks lists, for a cook, every station interaction the catalogue allows in the current state.
"""
"""This file is UN-VERIFIED"""
import os
from dataclasses import dataclass
from typing import FrozenSet, List, Optional

import yaml

from overcooked_kitchen.layout import Cell, Pose, PROVIDES

WAIT = ("wait",)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "domain.yaml")) as _f:
    _ACTIONS = {a["name"]: a for a in yaml.safe_load(_f)["actions"]}

CATALOGUE_EVENTS: FrozenSet[str] = frozenset(
    a["sim"]["event"] for a in _ACTIONS.values() if a.get("sim", {}).get("kind") == "interact")

_BY_EVENT = {}
for _name, _a in _ACTIONS.items():
    if _a.get("sim", {}).get("kind") == "interact":
        _BY_EVENT.setdefault(_a["sim"]["event"], []).append(_name)

_TAKE_PLATE = {"rack": "take-clean-plate", "dropoff": "take-dirty-plate", "sink": "take-from-sink",
               "counter": "take-plate"}


@dataclass(frozen=True)
class Task:
    cook: int
    station: str
    cell: Cell
    pose: Pose
    event: str
    action: str
    strokes: int
    travel: int

    @property
    def duration(self) -> int:
        return self.travel + self.strokes


def catalogue_action(world, i, event, cell) -> Optional[str]:
    """Catalogue action name for an interact event by cook i at station cell, else None."""
    if event not in CATALOGUE_EVENTS:
        return None
    kind = world.kind.get(cell)
    if event == "take_raw":
        return "take-raw" if kind in PROVIDES else "take-raw-back"
    if event == "take_plate":
        return _TAKE_PLATE.get(kind)
    if event == "add":
        held = world.players[i].held
        return "add-" + held[1] if held else None
    names = _BY_EVENT[event]
    return names[0] if len(names) == 1 else None


def stations_of(world, counter_policy="none"):
    """Stations of the layout: policy counters plus counters now holding an item.

    Names are fixed per cell (counters numbered once over all reachable counters), so they
    do not change when an item appears on a counter.
    """
    cache = world.__dict__.setdefault("_stations_cache", {})
    if ("names",) not in cache:
        cache[("names",)] = world.lay.stations("all")
    key = counter_policy if isinstance(counter_policy, str) else tuple(map(tuple, counter_policy))
    if key not in cache:
        keep = {c.cell for c in world.lay.stations(counter_policy)}
        cache[key] = keep
    out = []
    for st in cache[("names",)]:
        if st.kind != "counter":
            out.append(st)
        elif st.cell in cache[key] or world.surface.get(st.cell) is not None:
            out.append(st)
    return out


def _strokes(world, event, cell) -> int:
    p = world.params
    if event == "chop":
        return p["chop"][world.surface[cell][1]] - world.progress[cell]
    if event == "wash":
        return p["wash_plate"] - world.progress[cell]
    if event == "scrub":
        return p["scrub_pot"] - world.pots[cell].scrub
    return 1


def legal_tasks(world, i, locked_cells=frozenset(), blocked_tiles=frozenset(), counter_policy="none") -> List[Task]:
    pl = world.players[i]
    dist = world.lay.bfs_poses((pl.pos[0], pl.pos[1], pl.facing))
    out = []
    for st in stations_of(world, counter_policy):
        if st.cell in locked_cells:
            continue
        best = None
        for pose in world.lay.access_poses(st.cell):
            if pose[:2] in blocked_tiles or pose not in dist:
                continue
            if best is None or dist[pose] < dist[best]:
                best = pose
        if best is None:
            continue
        ev = world.interact_event(i, best)
        action = catalogue_action(world, i, ev, st.cell) if ev else None
        if action is None:
            continue
        if ev == "start" and not world.is_recipe(world.pots[st.cell].comp()):
            continue
        out.append(Task(i, st.name, st.cell, best, ev, action, _strokes(world, ev, st.cell), dist[best]))
    return out
