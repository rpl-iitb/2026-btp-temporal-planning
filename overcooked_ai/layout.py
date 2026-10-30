"""Grid analysis: stations, interaction poses, exact travel times."""

"""This file is UN-VERIFIED"""


import collections
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Direction tuples exactly as overcooked_ai_py.mdp.actions.Direction
NORTH, SOUTH, EAST, WEST = (0, -1), (0, 1), (1, 0), (-1, 0)
DIRS = (NORTH, SOUTH, EAST, WEST)
DIR_NAME = {NORTH: "n", SOUTH: "s", EAST: "e", WEST: "w"}

STATION_CHARS = {"P": "pot", "O": "onion_dispenser", "T": "tomato_dispenser",
                 "D": "dish_dispenser", "S": "serving", "X": "counter"}
NAME_PREFIX = {"pot": "pot", "onion_dispenser": "odisp", "tomato_dispenser": "tdisp",
               "dish_dispenser": "ddisp", "serving": "serve", "counter": "ctr"}
PDDL_TYPE = {"pot": "pot", "onion_dispenser": "dispenser", "tomato_dispenser": "dispenser",
             "dish_dispenser": "dispenser", "serving": "serving", "counter": "counter"}
PROVIDES = {"onion_dispenser": "onion", "tomato_dispenser": "tomato", "dish_dispenser": "dish"}

Cell = Tuple[int, int]
Pose = Tuple[int, int, Tuple[int, int]]   # (x, y, direction)


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
        """Every (floor tile, facing) from which INTERACT reaches `cell`."""
        out = []
        for d in DIRS:
            t = (cell[0] - d[0], cell[1] - d[1])
            if self.is_floor(t):
                out.append((t[0], t[1], d))
        return out

    # ---- stations --------------------------------------------------------------------
    def stations(self, counter_policy: str = "shared", extra_counters=()) -> List[Station]:
        """Pots, dispensers and serving windows always; counters per policy:
        none | shared (touching >= 2 floor regions: hand-off counters) | all
        (touching any floor). `extra_counters` are always included (e.g. counters
        holding an object in the initial state)."""
        comp = self.regions()
        found, counts = [], collections.Counter()
        for y in range(self.H):
            for x in range(self.W):
                ch = self.terrain[y][x]
                if ch not in STATION_CHARS:
                    continue
                kind = STATION_CHARS[ch]
                poses = self.access_poses((x, y))
                if kind == "counter":
                    regs = {comp[(px, py)] for px, py, _ in poses}
                    keep = (x, y) in set(map(tuple, extra_counters)) or (
                        counter_policy == "all" and regs) or (counter_policy == "shared" and len(regs) >= 2)
                    if not keep:
                        continue
                counts[kind] += 1
                found.append(Station("{}{}".format(NAME_PREFIX[kind], counts[kind]), kind, (x, y)))
        return found

    # ---- travel times ----------------------------------------------------------------
    def bfs_poses(self, src: Pose) -> Dict[Pose, int]:
        """Exact single-agent steps from pose `src` to every reachable pose."""
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
