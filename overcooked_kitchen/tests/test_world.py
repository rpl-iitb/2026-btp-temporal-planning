"""Tests for the kitchen world rules."""
"""This file is UN-VERIFIED"""
from overcooked_kitchen.layout import NORTH, WEST
from overcooked_kitchen.world import World, sim_params, cook_time

GRID = ["XTXPX",
        "B 1 S",
        "XXDRW"]
TOMATO = (0, 1, 0)


def _world(orders=None, **sim):
    orders = orders or [{"id": "o1", "comp": TOMATO, "deadline": None}]
    return World(GRID, sim_params(sim), plates_clean=1, orders=orders)


def _cook(w, pos=(3, 1), facing=NORTH, held=None):
    p = w.players[0]
    p.pos, p.facing, p.held = pos, facing, held


def test_chop_then_add():
    w = _world()
    board, crate, pot = (0, 1), (1, 0), (3, 0)
    _cook(w, (1, 1), NORTH)
    assert w.interact_event(0) == "take_raw"
    w.apply(0, "take_raw", crate, 0)
    assert w.players[0].held == ("raw", "tomato")
    _cook(w, (1, 1), WEST, w.players[0].held)
    assert w.interact_event(0) == "put_raw"
    w.apply(0, "put_raw", board, 1)
    assert w.surface[board] == ("raw", "tomato") and w.players[0].held is None
    n = w.params["chop"]["tomato"]
    for k in range(n - 1):
        assert w.interact_event(0) == "chop"
        w.apply(0, "chop", board, 2 + k)
    assert w.surface[board] == ("raw", "tomato")
    w.apply(0, "chop", board, 9)
    assert w.surface[board] == ("chopped", "tomato")
    assert w.interact_event(0) == "take_chopped"
    w.apply(0, "take_chopped", board, 10)
    assert w.players[0].held == ("chopped", "tomato")
    _cook(w, (3, 1), NORTH, w.players[0].held)
    assert w.interact_event(0) == "add"
    w.apply(0, "add", pot, 11)
    assert w.pots[pot].contents == ["tomato"] and w.players[0].held is None


def test_pot_timing_and_dirty_cycle():
    w = _world()
    pot = w.pots[(3, 0)]
    _cook(w, held=("chopped", "tomato"))
    w.apply(0, "add", (3, 0), 0)
    assert pot.contents == ["tomato"]
    assert w.interact_event(0) == "start"
    t0 = 1
    w.apply(0, "start", (3, 0), t0)
    ct = cook_time(w.params, TOMATO)
    assert w.next_timer() == t0 + ct
    w.advance(t0 + ct - 1)
    assert pot.state == "cooking"
    w.advance(t0 + ct)
    assert pot.state == "ready" and w.t == t0 + ct
    w.stack[(2, 2)].remove("plate1")
    _cook(w, held=("plate", "plate1"))
    assert w.interact_event(0) == "dish_up"
    w.apply(0, "dish_up", (3, 0), w.t)
    assert pot.state == "dirty" and w.plates["plate1"].state == "soup"
    assert w.interact_event(0) is None
    w.players[0].held = None
    assert w.interact_event(0) == "scrub"
    for k in range(w.params["scrub_pot"]):
        w.apply(0, "scrub", (3, 0), w.t + k)
    assert pot.state == "idle" and not pot.contents


def test_soup_burns_and_burnt_pot_needs_scrubbing():
    w = _world(burn=5)
    pot = w.pots[(3, 0)]
    _cook(w, held=("chopped", "tomato"))
    w.apply(0, "add", (3, 0), 0)
    w.apply(0, "start", (3, 0), 1)
    ready = 1 + cook_time(w.params, TOMATO)
    w.advance(ready)
    assert pot.state == "ready"
    w.advance(ready + 4)
    assert pot.state == "ready"
    w.advance(ready + 5)
    assert pot.state == "burnt" and len(w.burnt) == 1
    _cook(w, held=("plate", "plate1"))
    assert w.interact_event(0) is None
    w.players[0].held = None
    assert w.interact_event(0) == "scrub"
    w.apply(0, "scrub", (3, 0), w.t, strokes=w.params["scrub_pot"])
    assert pot.state == "idle" and not pot.contents


def test_late_delivery_counts_lateness():
    w = _world(orders=[{"id": "o1", "comp": TOMATO, "deadline": 10},
                       {"id": "o2", "comp": TOMATO, "deadline": 1000}])
    w.plates["plate1"].state, w.plates["plate1"].soup = "soup", TOMATO
    _cook(w, (3, 1), (1, 0), ("plate", "plate1"))
    assert w.interact_event(0) == "deliver"
    w.apply(0, "deliver", (4, 1), 50)
    assert w.served == {"o1": 51}
    assert w.late == {"o1": 41}
    assert [o["id"] for o in w.pending_orders()] == ["o2"]


def test_non_recipe_pot_spoils():
    w = _world()
    pot = w.pots[(3, 0)]
    _cook(w, held=("chopped", "onion"))
    w.apply(0, "add", (3, 0), 0)
    w.apply(0, "start", (3, 0), 1)
    assert pot.recipe is False
    ready = pot.ready_at
    w.advance(ready)
    assert pot.state == "spoiled" and len(w.spoiled) == 1
    w.players[0].held = None
    assert w.interact_event(0) == "scrub"
    _cook(w, held=("plate", "plate1"))
    assert w.interact_event(0) is None
    w.advance(ready + 100)
    assert pot.state == "spoiled"


def test_partial_chop_keeps_progress():
    w = _world()
    board = (0, 1)
    w.surface[board] = ("raw", "onion")
    _cook(w, (1, 1), WEST)
    w.apply(0, "chop", board, 0, strokes=2)
    assert w.surface[board] == ("raw", "onion")
    _cook(w, (3, 1), NORTH)
    _cook(w, (1, 1), WEST)
    w.apply(0, "chop", board, 10, strokes=2)
    assert w.surface[board] == ("chopped", "onion")


def test_interact_event_at_hypothetical_pose():
    w = _world()
    _cook(w, (1, 1), NORTH, ("chopped", "onion"))
    assert w.interact_event(0) is None
    assert w.interact_event(0, (3, 1, NORTH)) == "add"
