"""Tests for grid stepping."""
"""This file is UN-VERIFIED"""
from overcooked_kitchen import grid
from overcooked_kitchen.layout import NORTH, SOUTH, EAST, WEST
from overcooked_kitchen.world import World, sim_params

GRID = ["XTXPX",
        "X1 2S",
        "XXDRW"]
TOMATO = (0, 1, 0)


def _world(**sim):
    orders = [{"id": "o1", "comp": TOMATO, "deadline": None}]
    return World(GRID, sim_params(sim), plates_clean=1, orders=orders)


def _set(w, i, pos, facing=NORTH, held=None):
    p = w.players[i]
    p.pos, p.facing, p.held = pos, facing, held


def test_same_tile_conflict_both_stay():
    w = _world()
    _set(w, 0, (1, 1))
    _set(w, 1, (3, 1))
    grid.step(w, [EAST, WEST])
    assert w.players[0].pos == (1, 1) and w.players[1].pos == (3, 1)
    assert w.players[0].facing == EAST and w.players[1].facing == WEST
    assert w.t == 1


def test_swap_conflict_both_stay():
    w = _world()
    _set(w, 0, (1, 1))
    _set(w, 1, (2, 1))
    grid.step(w, [EAST, WEST])
    assert w.players[0].pos == (1, 1) and w.players[1].pos == (2, 1)


def test_move_into_counter_only_turns():
    w = _world()
    _set(w, 0, (1, 1), EAST)
    grid.step(w, [NORTH, grid.STAY])
    assert w.players[0].pos == (1, 1) and w.players[0].facing == NORTH


def test_interact_returns_event():
    w = _world()
    _set(w, 0, (1, 1), NORTH)
    _set(w, 1, (3, 1), SOUTH)
    assert grid.step(w, [grid.INTERACT, grid.STAY]) == ["take_raw", None]
    assert w.players[0].held == ("raw", "tomato")


def test_diner_returns_plate():
    w = _world()
    eat = w.params["eat"]
    _set(w, 0, (3, 1), EAST, ("plate", "plate1"))
    _set(w, 1, (2, 1))
    w.stack[(2, 2)].remove("plate1")
    w.plates["plate1"].state, w.plates["plate1"].soup = "soup", TOMATO
    t = w.t
    assert grid.step(w, [grid.INTERACT, grid.STAY])[0] == "deliver"
    assert w.served["o1"] == t + 1
    while w.t < t + eat:
        grid.step(w, [grid.STAY, grid.STAY])
    assert w.plates["plate1"].state != "dirty"
    assert "plate1" not in w.stack[(3, 2)]
    grid.step(w, [grid.STAY, grid.STAY])
    assert w.t == t + 1 + eat
    assert w.plates["plate1"].state == "dirty"
    assert w.stack[(3, 2)] == ["plate1"]


def test_spoiled_pot_scrubs_to_idle():
    w = _world()
    pot = (3, 0)
    w.pots[pot].state = "spoiled"
    w.pots[pot].contents = ["onion", "tomato"]
    w.pots[pot].recipe = False
    _set(w, 0, (3, 1), NORTH)
    _set(w, 1, (2, 1))
    for _ in range(w.params["scrub_pot"]):
        assert grid.step(w, [grid.INTERACT, grid.STAY])[0] == "scrub"
    assert w.pots[pot].state == "idle"
    assert w.pots[pot].contents == []
    assert w.pots[pot].recipe is True


def test_expected_event_guards_interact():
    w = _world()
    _set(w, 0, (1, 1), NORTH)
    _set(w, 1, (3, 1), SOUTH)
    assert grid.step(w, [grid.INTERACT, grid.STAY], expected=["chop", None]) == [None, None]
    assert w.players[0].held is None and w.t == 1
    assert grid.step(w, [grid.INTERACT, grid.STAY], expected=["take_raw", None]) == ["take_raw", None]
    assert w.players[0].held == ("raw", "tomato")
