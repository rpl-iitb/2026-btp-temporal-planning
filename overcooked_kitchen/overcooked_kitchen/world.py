"""Kitchen world state and forgiving rules: stations, plates, pots, timers, orders.

Cooks act through apply(i, event, cell, t); time moves only through advance(t).
"""
"""This file is UN-VERIFIED"""
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from overcooked_kitchen.layout import (Layout, Pose, NORTH, INGREDIENTS, STATION_CHARS,
                                       PROVIDES)

DEFAULTS = {
    "chop": {"onion": 4, "tomato": 3, "carrot": 5},
    "cook_base": 2,
    "cook": {"onion": 5, "tomato": 3, "carrot": 6},
    "burn": 20,
    "scrub_pot": 4,
    "wash_plate": 3,
    "eat": 12,
    "max_ingredients": 4,
}


def sim_params(overrides=None):
    p = copy.deepcopy(DEFAULTS)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    return p


def cook_time(params, comp):
    """comp: (onions, tomatoes, carrots)."""
    return params["cook_base"] + sum(n * params["cook"][i] for i, n in zip(INGREDIENTS, comp))


@dataclass
class Plate:
    pid: str
    state: str = "clean"
    soup: Optional[Tuple[int, int, int]] = None


@dataclass
class Pot:
    contents: List[str] = field(default_factory=list)
    state: str = "idle"
    ready_at: Optional[int] = None
    cook_start: Optional[int] = None
    scrub: int = 0
    recipe: bool = True

    def comp(self):
        return tuple(self.contents.count(i) for i in INGREDIENTS)


@dataclass
class Player:
    pos: Tuple[int, int]
    facing: Tuple[int, int]
    held: Optional[tuple] = None


class World:
    """Full state plus event rules. Station state is kept per cell."""

    def __init__(self, rows, params, plates_clean=1, plates_dirty=0, dirty_pots=0, orders=()):
        self.lay = Layout(rows)
        self.params = params
        self.t = 0
        self.kind = {}
        for y, r in enumerate(self.lay.terrain):
            for x, ch in enumerate(r):
                if ch in STATION_CHARS:
                    self.kind[(x, y)] = STATION_CHARS[ch]
        self.players = [Player(self.lay.starts[k], NORTH) for k in sorted(self.lay.starts)]
        self.surface: Dict[tuple, Optional[tuple]] = {}
        self.progress: Dict[tuple, int] = {}
        self.pots: Dict[tuple, Pot] = {}
        self.stack: Dict[tuple, List[str]] = {}
        self.sink: Dict[tuple, Optional[str]] = {}
        self.plates: Dict[str, Plate] = {}
        for c, k in sorted(self.kind.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if k in ("counter", "board"):
                self.surface[c] = None
            elif k == "pot":
                self.pots[c] = Pot()
            elif k in ("rack", "dropoff"):
                self.stack[c] = []
            elif k == "sink":
                self.sink[c] = None
            if k in ("board", "sink"):
                self.progress[c] = 0
        racks = sorted(c for c, k in self.kind.items() if k == "rack")
        drops = sorted(c for c, k in self.kind.items() if k == "dropoff")
        n = 0
        for _ in range(plates_clean):
            n += 1
            self.plates["plate{}".format(n)] = Plate("plate{}".format(n))
            self.stack[racks[0]].append("plate{}".format(n))
        for _ in range(plates_dirty):
            n += 1
            self.plates["plate{}".format(n)] = Plate("plate{}".format(n), "dirty")
            self.stack[drops[0]].append("plate{}".format(n))
        for c in sorted(self.pots, key=lambda c: (c[1], c[0]))[:dirty_pots]:
            self.pots[c].state = "dirty"
        self.orders = [dict(o) for o in orders]
        self.served: Dict[str, int] = {}
        self.late: Dict[str, int] = {}
        self.diners: Dict[str, Tuple[int, tuple]] = {}
        self.events: List[tuple] = []
        self.burnt: List[tuple] = []
        self.spoiled: List[tuple] = []

    # ---- queries ------------------------------------------------------------------------
    def facing_cell(self, i):
        p = self.players[i]
        return (p.pos[0] + p.facing[0], p.pos[1] + p.facing[1])

    def pending_orders(self):
        return [o for o in self.orders if o["id"] not in self.served]

    def is_recipe(self, comp):
        return any(tuple(o["comp"]) == tuple(comp) for o in self.orders)

    def interact_event(self, i, pose: Optional[Pose] = None) -> Optional[str]:
        """What INTERACT by cook i would do, standing at pose when given."""
        if pose is None:
            c = self.facing_cell(i)
        else:
            c = (pose[0] + pose[2][0], pose[1] + pose[2][1])
        k, h = self.kind.get(c), self.players[i].held
        if k is None:
            return None
        if k in PROVIDES:
            return "take_raw" if h is None else None
        if k in ("counter", "board"):
            item = self.surface[c]
            if h is None:
                if item is None:
                    return None
                if item[0] == "raw":
                    return "chop" if k == "board" else "take_raw"
                return "take_chopped" if item[0] == "chopped" else "take_plate"
            if item is not None:
                return None
            if h[0] == "plate":
                return "put_plate" if k == "counter" else None
            return "put_" + h[0]
        if k == "pot":
            pot = self.pots[c]
            if h is None:
                if pot.state in ("dirty", "burnt", "spoiled"):
                    return "scrub"
                if pot.state == "idle" and pot.contents:
                    return "start"
                return None
            if h[0] == "chopped":
                ok = pot.state == "idle" and len(pot.contents) < self.params["max_ingredients"]
                return "add" if ok else None
            if h[0] == "plate":
                pl = self.plates[h[1]]
                return "dish_up" if pl.state == "clean" and pot.state == "ready" else None
            return None
        if k == "serving":
            if h is not None and h[0] == "plate" and self.plates[h[1]].state == "soup":
                comp = self.plates[h[1]].soup
                if any(tuple(o["comp"]) == tuple(comp) for o in self.pending_orders()):
                    return "deliver"
            return None
        if k == "rack":
            if h is None:
                return "take_plate" if self.stack[c] else None
            if h[0] == "plate" and self.plates[h[1]].state == "clean":
                return "rack_plate"
            return None
        if k == "dropoff":
            return "take_plate" if h is None and self.stack[c] else None
        if k == "sink":
            pid = self.sink[c]
            if h is None:
                if pid is None:
                    return None
                return "wash" if self.plates[pid].state == "dirty" else "take_plate"
            if pid is None and h[0] == "plate" and self.plates[h[1]].state == "dirty":
                return "put_in_sink"
            return None
        return None

    # ---- transitions --------------------------------------------------------------------
    def apply(self, i, ev, c, t, strokes=1):
        """Apply event ev by cook i on station cell c at time t."""
        p = self.players[i]
        k = self.kind[c]
        self.events.append((t, i, ev, c))
        if ev == "take_raw":
            if k in PROVIDES:
                p.held = ("raw", PROVIDES[k])
            else:
                p.held, self.surface[c] = self.surface[c], None
        elif ev in ("put_raw", "put_chopped", "put_plate"):
            self.surface[c], p.held = p.held, None
        elif ev == "chop":
            self.progress[c] += strokes
            ing = self.surface[c][1]
            if self.progress[c] >= self.params["chop"][ing]:
                self.surface[c], self.progress[c] = ("chopped", ing), 0
        elif ev == "take_chopped":
            p.held, self.surface[c] = self.surface[c], None
        elif ev == "add":
            self.pots[c].contents.append(p.held[1])
            p.held = None
        elif ev == "start":
            pot = self.pots[c]
            pot.state, pot.cook_start = "cooking", t
            pot.recipe = self.is_recipe(pot.comp())
            pot.ready_at = t + cook_time(self.params, pot.comp())
        elif ev == "dish_up":
            pot, pl = self.pots[c], self.plates[p.held[1]]
            pl.state, pl.soup = "soup", pot.comp()
            pot.contents, pot.state, pot.ready_at, pot.cook_start = [], "dirty", None, None
        elif ev == "scrub":
            pot = self.pots[c]
            pot.scrub += strokes
            if pot.scrub >= self.params["scrub_pot"]:
                pot.state, pot.scrub, pot.contents = "idle", 0, []
                pot.ready_at, pot.cook_start, pot.recipe = None, None, True
        elif ev == "deliver":
            pl = self.plates[p.held[1]]
            cands = [o for o in self.pending_orders() if tuple(o["comp"]) == tuple(pl.soup)]
            o = min(cands, key=lambda o: (o.get("deadline") is None, o.get("deadline") or 0, o["id"]))
            self.served[o["id"]] = t + 1
            if o.get("deadline") is not None and t + 1 > o["deadline"]:
                self.late[o["id"]] = t + 1 - o["deadline"]
            drop = min((d for d, kk in self.kind.items() if kk == "dropoff"),
                       key=lambda d: (abs(d[0] - c[0]) + abs(d[1] - c[1]), d), default=None)
            pl.state, pl.soup = "eating", None
            if drop is not None:
                self.diners[pl.pid] = (t + 1 + self.params["eat"], drop)
            p.held = None
        elif ev == "take_plate":
            if k in ("rack", "dropoff"):
                p.held = ("plate", self.stack[c].pop())
            elif k == "sink":
                p.held, self.sink[c] = ("plate", self.sink[c]), None
            else:
                p.held, self.surface[c] = self.surface[c], None
        elif ev == "rack_plate":
            self.stack[c].append(p.held[1])
            p.held = None
        elif ev == "put_in_sink":
            self.sink[c], p.held = p.held[1], None
        elif ev == "wash":
            self.progress[c] += strokes
            if self.progress[c] >= self.params["wash_plate"]:
                self.plates[self.sink[c]].state, self.progress[c] = "clean", 0

    def _timers(self):
        burn = self.params.get("burn")
        out = []
        for c, pot in self.pots.items():
            if pot.state == "cooking":
                out.append((pot.ready_at, "cooked", c))
            elif pot.state == "ready" and burn is not None:
                out.append((pot.ready_at + burn, "burn", c))
        for pid, (ret, cell) in self.diners.items():
            out.append((ret, "return", pid))
        return out

    def next_timer(self):
        times = [x[0] for x in self._timers()]
        return min(times) if times else None

    def advance(self, t):
        """Fire every timer with time <= t, in time order, then set self.t = t."""
        while True:
            due = sorted((x for x in self._timers() if x[0] <= t), key=lambda x: x[0])
            if not due:
                break
            when, what, key = due[0]
            if what == "cooked":
                pot = self.pots[key]
                if pot.recipe:
                    pot.state = "ready"
                else:
                    pot.state = "spoiled"
                    self.spoiled.append((when, key, pot.comp()))
            elif what == "burn":
                self.pots[key].state = "burnt"
                self.burnt.append((when, key, self.pots[key].comp()))
            else:
                cell = self.diners.pop(key)[1]
                self.plates[key].state = "dirty"
                self.stack[cell].append(key)
        self.t = t

    # ---- snapshots ----------------------------------------------------------------------
    def snapshot(self):
        return {
            "t": self.t,
            "players": [{"pos": p.pos, "facing": p.facing, "held": self._held_view(p.held)} for p in self.players],
            "surface": {c: self._held_view(v) for c, v in self.surface.items()},
            "progress": dict(self.progress),
            "pots": {c: {"contents": list(p.contents), "state": p.state, "ready_at": p.ready_at,
                         "cook_start": p.cook_start, "scrub": p.scrub, "recipe": p.recipe} for c, p in self.pots.items()},
            "stack": {c: [self.plates[x].state for x in v] for c, v in self.stack.items()},
            "sink": {c: (None if v is None else self.plates[v].state) for c, v in self.sink.items()},
            "served": dict(self.served),
            "late": dict(self.late),
            "burnt": len(self.burnt),
            "spoiled": len(self.spoiled),
            "diners": {pid: r for pid, (r, _) in self.diners.items()},
            "diner_cells": {pid: c for pid, (_, c) in self.diners.items()},
        }

    def _held_view(self, h):
        if h is None:
            return None
        if h[0] == "plate":
            pl = self.plates[h[1]]
            return ("plate", pl.state, pl.soup)
        return h
