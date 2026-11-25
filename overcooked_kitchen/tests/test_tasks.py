"""Tests for compound tasks and legality."""
"""This file is UN-VERIFIED"""
import os

import yaml

from overcooked_kitchen import tasks
from overcooked_kitchen.layout import NORTH, WEST
from overcooked_kitchen.scenarios import load_scenario, make_world, example
from overcooked_kitchen.tasks import CATALOGUE_EVENTS, catalogue_action, legal_tasks, stations_of
from overcooked_kitchen.world import World, sim_params

GRID = ["XTXPX",
        "B 1 S",
        "XXDRW"]


def _world(orders=None, grid=GRID, **sim):
    orders = orders or [{"id": "o1", "comp": (0, 1, 0), "deadline": None}]
    return World(grid, sim_params(sim), plates_clean=1 if "D" in "".join(grid) else 0, orders=orders)


def test_k01_first_tasks():
    w = make_world(load_scenario(example("k01_one_soup")))
    ts = legal_tasks(w, 0)
    assert {t.event for t in ts} <= {"take_raw", "take_plate"}
    raw = [t for t in ts if t.event == "take_raw"]
    assert {t.cell for t in raw} == {c for c, k in w.kind.items() if k.endswith("_crate")}
    assert all(t.action == "take-raw" and t.strokes == 1 for t in raw)
    assert [t.action for t in ts if t.event == "take_plate"] == ["take-clean-plate"]
    assert all(t.duration == t.travel + 1 for t in ts)


def test_events_match_domain_catalogue():
    path = os.path.join(os.path.dirname(tasks.__file__), "domain.yaml")
    with open(path) as f:
        acts = yaml.safe_load(f)["actions"]
    assert len(acts) == 26
    inter = [a for a in acts if a["sim"]["kind"] == "interact"]
    assert CATALOGUE_EVENTS == {a["sim"]["event"] for a in inter}
    names = {a["name"] for a in acts}
    w = _world()
    w.players[0].held = ("chopped", "onion")
    kinds = {"take_raw": ["onion_crate", "counter"], "take_plate": ["rack", "dropoff", "sink", "counter"]}
    cell_of = {"onion_crate": (1, 0), "rack": (2, 2), "dropoff": (3, 2), "sink": (4, 2)}
    w.kind[(0, 0)] = "counter"
    cell_of["counter"] = (0, 0)
    for ev in CATALOGUE_EVENTS:
        for k in kinds.get(ev, ["pot"]):
            cell = cell_of.get(k, (3, 0))
            assert catalogue_action(w, 0, ev, cell) in names
    assert catalogue_action(w, 0, "put_nothing", (3, 0)) is None


def test_non_recipe_start_not_offered():
    w = _world()
    w.pots[(3, 0)].contents = ["onion"]
    assert w.interact_event(0, (3, 1, NORTH)) == "start"
    assert not [t for t in legal_tasks(w, 0) if t.event == "start"]
    w.pots[(3, 0)].contents = ["tomato"]
    assert [t for t in legal_tasks(w, 0) if t.event == "start"]


def test_locked_station_not_offered():
    w = _world()
    assert any(t.cell == (1, 0) for t in legal_tasks(w, 0))
    assert not any(t.cell == (1, 0) for t in legal_tasks(w, 0, locked_cells={(1, 0)}))


def test_blocked_pose_uses_other_access_pose():
    grid = ["XXOXX",
            "X   X",
            "X 1 X",
            "XXXXX"]
    w = _world(grid=grid)
    w.players[0].pos, w.players[0].facing = (2, 2), NORTH
    t0 = [t for t in legal_tasks(w, 0) if t.cell == (2, 0)][0]
    assert t0.pose == (2, 1, NORTH) and t0.travel == 1
    assert [t for t in legal_tasks(w, 0, blocked_tiles={(2, 1)}) if t.cell == (2, 0)] == []


def test_blocked_nearer_pose_falls_back_to_farther():
    grid = ["XXXXXXX",
            "XO    X",
            "X  1  X",
            "XXXXXXX"]
    w = _world(grid=grid)
    w.players[0].pos, w.players[0].facing = (3, 2), WEST
    t0 = [t for t in legal_tasks(w, 0) if t.cell == (1, 1)][0]
    assert t0.pose == (2, 1, WEST) and t0.travel == 2
    t1 = [t for t in legal_tasks(w, 0, blocked_tiles={(2, 1)}) if t.cell == (1, 1)][0]
    assert t1.pose == (1, 2, NORTH) and t1.travel == 3
    assert t1.pose[:2] not in {(2, 1)}


def test_unreachable_station_not_offered():
    grid = ["XXXXXXX",
            "XT 1 XX",
            "XXXXXPX",
            "XXXXXXX"]
    w = _world(grid=grid)
    w.players[0].held = ("chopped", "tomato")
    ts = legal_tasks(w, 0)
    assert not any(t.cell == (5, 2) for t in ts)
    w.players[0].held = None
    assert any(t.cell == (1, 1) for t in legal_tasks(w, 0))


def test_remaining_strokes_after_partial_chop():
    w = _world()
    w.surface[(0, 1)] = ("raw", "tomato")
    w.progress[(0, 1)] = 2
    w.players[0].pos, w.players[0].facing = (1, 1), WEST
    ch = [t for t in legal_tasks(w, 0) if t.event == "chop"]
    assert len(ch) == 1 and ch[0].strokes == w.params["chop"]["tomato"] - 2
    assert ch[0].action == "chop"


def test_stations_policy_counters():
    w = _world(grid=["XXXXX", "X 1 X", "XXXXX"])
    assert [s.kind for s in stations_of(w)] == []
    assert len(stations_of(w, "all")) == 8


def test_counter_item_added_after_first_call_appears():
    w = _world(grid=["XXXXX", "X 1 X", "XXXXX"])
    assert stations_of(w) == []
    assert legal_tasks(w, 0) == []
    w.surface[(2, 0)] = ("raw", "onion")
    st = stations_of(w)
    assert [s.cell for s in st] == [(2, 0)]
    name = st[0].name
    ts = legal_tasks(w, 0)
    assert [(t.cell, t.action) for t in ts] == [((2, 0), "take-raw-back")]
    w.surface[(2, 0)] = None
    w.surface[(1, 0)] = ("plate", "plate1")
    assert [s.cell for s in stations_of(w)] == [(1, 0)]
    w.surface[(2, 0)] = ("raw", "onion")
    assert {s.cell: s.name for s in stations_of(w)}[(2, 0)] == name
