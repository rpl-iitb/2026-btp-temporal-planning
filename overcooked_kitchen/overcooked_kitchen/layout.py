"""Static grid analysis for the kitchen: poses, BFS travel times, stations.

Floor is ' ' (player digits are replaced by ' '); every other cell blocks.
A motion into a non-floor cell only turns the cook; every motion takes one step.
INTERACT acts on the faced cell. Poses are (x, y, facing) triples.
"""
"""This file is UN-VERIFIED"""
import collections
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

NORTH, SOUTH, EAST, WEST = (0, -1), (0, 1), (1, 0), (-1, 0)
DIRS = (NORTH, SOUTH, EAST, WEST)
DIR_NAME = {NORTH: "n", SOUTH: "s", EAST: "e", WEST: "w"}

DIR_BY_NAME = {v: k for k, v in DIR_NAME.items()}

INGREDIENTS = ("onion", "tomato", "carrot")

STATION_CHARS = {"X": "counter", "B": "board", "P": "pot", "O": "onion_crate", "T": "tomato_crate",
                 "C": "carrot_crate", "D": "rack", "R": "dropoff", "W": "sink", "S": "serving"}
NAME_PREFIX = {"counter": "ctr", "board": "board", "pot": "pot", "onion_crate": "ocrate",
               "tomato_crate": "tcrate", "carrot_crate": "ccrate", "rack": "rack", "dropoff": "drop",
               "sink": "sink", "serving": "serve"}
PDDL_TYPE = {"counter": "counter", "board": "board", "pot": "pot", "onion_crate": "crate",
             "tomato_crate": "crate", "carrot_crate": "crate", "rack": "rack", "dropoff": "dropoff",
             "sink": "sink", "serving": "serving"}
PROVIDES = {"onion_crate": "onion", "tomato_crate": "tomato", "carrot_crate": "carrot"}

Cell = Tuple[int, int]
Pose = Tuple[int, int, Tuple[int, int]]   # (x, y, direction)

BFS_LAYOUTS = 8                           # terrains whose BFS tables are kept (shared by layouts of equal terrain)
_BFS = collections.OrderedDict()


@dataclass
class Station:
    name: str
    kind: str
    cell: Cell


class Layout:
    """Grid rows (strings, Overcooked format, player digits allowed)."""

    def __init__(self, rows: List[str]):
        if len({len(r) for r in rows}) != 1:
            raise ValueError("ragged grid")
        self.rows = list(rows)
        self.H, self.W = len(rows), len(rows[0])
        self.starts: Dict[int, Cell] = {}
        for y, r in enumerate(rows):
            for x, ch in enumerate(r):
                if ch.isdigit():
                    self.starts[int(ch)] = (x, y)
        self.terrain = [[" " if ch.isdigit() else ch for ch in r] for r in rows]
        key = tuple("".join(r) for r in self.terrain)
        if key not in _BFS:
            _BFS[key] = {}
            if len(_BFS) > BFS_LAYOUTS:
                _BFS.popitem(last=False)
        _BFS.move_to_end(key)
        self._bfs: Dict[Pose, Dict[Pose, int]] = _BFS[key]
        self._access: Dict[Cell, List[Pose]] = {}

    # ---- cells ---------------------------------------------------------------------
    def char(self, c: Cell) -> str:
        x, y = c
        if 0 <= x < self.W and 0 <= y < self.H:
            return self.terrain[y][x]
        return "#"

    def is_floor(self, c: Cell) -> bool:
        return self.char(c) == " "

    def floor_tiles(self) -> List[Cell]:
        return [(x, y) for y in range(self.H) for x in range(self.W) if self.is_floor((x, y))]

    def regions(self) -> Dict[Cell, int]:
        """Connected component id of every floor tile."""
        comp, k = {}, 0
        for t in self.floor_tiles():
            if t in comp:
                continue
            q = collections.deque([t])
            comp[t] = k
            while q:
                x, y = q.popleft()
                for dx, dy in DIRS:
                    n = (x + dx, y + dy)
                    if self.is_floor(n) and n not in comp:
                        comp[n] = k
                        q.append(n)
            k += 1
        return comp

    def access_poses(self, cell: Cell) -> List[Pose]:
        """Every (floor tile, facing) from which INTERACT reaches `cell` (a fresh list)."""
        cell = tuple(cell)
        if cell not in self._access:
            out = []
            for d in DIRS:
                t = (cell[0] - d[0], cell[1] - d[1])
                if self.is_floor(t):
                    out.append((t[0], t[1], d))
            self._access[cell] = out
        return list(self._access[cell])

    # ---- stations --------------------------------------------------------------------
    def stations(self, counter_policy="none", extra_counters=()) -> List[Station]:
        found, counts = [], collections.Counter()
        extra = {tuple(c) for c in extra_counters}
        explicit = {tuple(c) for c in counter_policy} if isinstance(counter_policy, (list, tuple)) else set()
        for y in range(self.H):
            for x in range(self.W):
                kind = STATION_CHARS.get(self.terrain[y][x])
                if kind is None:
                    continue
                if kind == "counter":
                    keep = (x, y) in extra or (x, y) in explicit or (
                        counter_policy == "all" and self.access_poses((x, y)))
                    if not keep:
                        continue
                counts[kind] += 1
                found.append(Station("{}{}".format(NAME_PREFIX[kind], counts[kind]), kind, (x, y)))
        return found

    # ---- travel times ----------------------------------------------------------------
    def bfs_poses(self, src: Pose) -> Dict[Pose, int]:
        """Exact single-agent steps from pose `src` to every reachable pose (cached per source; do not modify)."""
        src = (src[0], src[1], tuple(src[2]))
        if src in self._bfs:
            return self._bfs[src]
        dist = {src: 0}
        q = collections.deque([src])
        while q:
            x, y, o = q.popleft()
            for d in DIRS:
                n = (x + d[0], y + d[1])
                nxt = (n[0], n[1], d) if self.is_floor(n) else (x, y, d)
                if nxt not in dist:
                    dist[nxt] = dist[(x, y, o)] + 1
                    q.append(nxt)
        self._bfs[src] = dist
        return dist

    def next_step(self, src: Pose, dst: Pose, blocked=frozenset()) -> Optional[Tuple[int, int]]:
        """First motion direction of a shortest path src -> dst that never
        enters a `blocked` tile; None if already there or unreachable."""
        if src == dst:
            return None
        prev = {src: None}
        q = collections.deque([src])
        while q:
            cur = q.popleft()
            if cur == dst:
                break
            x, y, _ = cur
            for d in DIRS:
                n = (x + d[0], y + d[1])
                if self.is_floor(n):
                    if n in blocked:
                        continue
                    nxt = (n[0], n[1], d)
                else:
                    nxt = (x, y, d)
                if nxt not in prev:
                    prev[nxt] = (cur, d)
                    q.append(nxt)
        if dst not in prev:
            return None
        cur = dst
        while prev[cur][0] != src:
            cur = prev[cur][0]
        return prev[cur][1]


def pose_name(p: Pose) -> str:
    return "p{}_{}{}".format(p[0], p[1], DIR_NAME[p[2]])


def tile_name(c: Cell) -> str:
    return "t{}_{}".format(c[0], c[1])
