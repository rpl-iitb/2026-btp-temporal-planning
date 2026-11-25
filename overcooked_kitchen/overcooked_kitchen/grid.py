"""Grid stepping with Overcooked motion rules: interacts first, then simultaneous moves."""
"""This file is UN-VERIFIED"""
from typing import List, Optional, Sequence

from overcooked_kitchen.layout import NORTH, SOUTH, EAST, WEST, DIRS

STAY = "stay"
INTERACT = "interact"
ACTIONS = (NORTH, SOUTH, EAST, WEST, STAY, INTERACT)
ACTION_NAMES = ("north", "south", "east", "west", "stay", "interact")


def step(world, actions: Sequence, expected: Optional[Sequence] = None) -> List[Optional[str]]:
    """expected[i] (optional): the only event cook i's INTERACT may apply, judged at apply time; None = any."""
    t = world.t
    events = [None] * len(actions)
    for i, a in enumerate(actions):
        if a == INTERACT:
            ev = world.interact_event(i)
            if expected is not None and expected[i] is not None and ev != expected[i]:
                continue
            if ev is not None:
                world.apply(i, ev, world.facing_cell(i), t)
                events[i] = ev
    _move(world, actions)
    world.advance(t + 1)
    return events


def _move(world, actions):
    want = []
    for p, a in zip(world.players, actions):
        if a in DIRS:
            p.facing = a
            n = (p.pos[0] + a[0], p.pos[1] + a[1])
            want.append(n if world.lay.is_floor(n) else p.pos)
        else:
            want.append(p.pos)
    cur = [p.pos for p in world.players]
    ok = [True] * len(want)
    for i in range(len(want)):
        for j in range(len(want)):
            if i != j and (want[i] == want[j] or (want[i] == cur[j] and want[j] == cur[i])):
                ok[i] = False
    for i in range(len(want)):
        if ok[i] and any(j != i and want[i] == cur[j] and (not ok[j] or want[j] == cur[j])
                         for j in range(len(want))):
            ok[i] = False
    for i, p in enumerate(world.players):
        if ok[i]:
            p.pos = want[i]
