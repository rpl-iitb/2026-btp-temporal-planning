"""Tests for the grid clock: nominal agreement with the sampled clock, yielding, deadlock, driving."""
"""This file is UN-VERIFIED"""
import pytest

from helpers import serial_policy
from overcooked_kitchen import grid
from overcooked_kitchen.core import Sim
from overcooked_kitchen.layout import EAST, NORTH, WEST
from overcooked_kitchen.tasks import WAIT

REGION = ["XOXPXX",
          "B    S",
          "X    X",
          "XDXWRX"]
STARTS = [(1, 1), (4, 2), (2, 2), (3, 1)]


def _ends(sim):
    return [(e["cook"], e["event"], e["station"], e["t"]) for e in sim.log if e["type"] in ("end", "task_failed")]


def _agree(make):
    a, b = make("sampled"), make("grid")
    ma, mb = a.run(serial_policy), b.run(serial_policy)
    assert ma["success"] and mb["success"], (ma, mb)
    ea, eb = _ends(a), _ends(b)
    assert [e[:3] for e in ea] == [e[:3] for e in eb]
    assert all(abs(x[3] - y[3]) <= 1 for x, y in zip(ea, eb))
    assert abs(ma["makespan"] - mb["makespan"]) <= 1
    assert mb["blocked"] == {c: 0 for c in b.cooks}
    assert not [e for e in b.log if e["type"] == "yield"]


@pytest.mark.parametrize("name", ["k01_one_soup", "k02_two_dishes", "k03_service_cleanup", "k04_burn_risk"])
def test_nominal_agreement_one_cook(name):
    _agree(lambda clock: Sim(name, clock=clock))


def _regions(n):
    rows = ["".join(REGION[y] for _ in range(n)) for y in range(len(REGION))]
    for k in range(n):
        x, y = STARTS[k]
        x += 6 * k
        rows[y] = rows[y][:x] + str(k + 1) + rows[y][x + 1:]
    return {"scenario_id": "apart{}".format(n), "grid": rows, "plates": {"clean": n},
            "orders": [{"recipe": {"onion": 1}, "count": n}]}


def _spread_plates(sim):
    w = sim.world
    racks = sorted((c for c, k in w.kind.items() if k == "rack"), key=lambda c: c[0])
    plates = w.stack[racks[0]]
    for r in racks[1:]:
        w.stack[r].append(plates.pop())
    return sim


@pytest.mark.parametrize("n", [2, 4])
def test_nominal_agreement_cooks_apart(n):
    _agree(lambda clock: _spread_plates(Sim(_regions(n), clock=clock)))


CORRIDOR = {"scenario_id": "corridor", "plates": {"clean": 0}, "orders": [{"recipe": {"tomato": 1}}],
            "grid": ["XXXXXXX",
                     "X1  2 T",
                     "XXXXX X",
                     "XXXXXXX"]}


def test_blocked_corridor_lengthens_task():
    sim = Sim(CORRIDOR, clock="grid")
    first = {}

    def policy(s, cook, legal):
        if cook == "c1" and not first:
            first["task"] = next(t for t in legal if t != WAIT and t.event == "take_raw")
            return first["task"]
        return WAIT

    m = sim.run(policy)
    task = first["task"]
    start = next(e for e in sim.log if e["type"] == "start" and e["cook"] == "c1")
    end = next(e for e in sim.log if e["type"] == "end" and e["cook"] == "c1")
    assert end["t"] - start["t"] > task.duration
    yields = [e for e in sim.log if e["type"] == "yield"]
    assert yields and yields[0]["cook"] == "c2" and yields[0]["for"] == "c1"
    assert sim.world.players[0].held == ("raw", "tomato")
    assert m["blocked"]["c1"] > 0 and m["blocked"]["c2"] == 0
    assert sim.world.players[1].pos == (5, 2)


def test_drive_mode_moves_and_interacts():
    sim = Sim("k01_one_soup", clock="grid")
    sim.set_mode("c1", "drive")
    assert sim.pending() == []
    assert sim.world.players[0].pos == (4, 2)
    for a in (WEST, WEST, WEST, NORTH, grid.INTERACT):
        sim.drive("c1", a)
        sim.tick()
        assert "c1" not in sim.pending()
    assert sim.t == 5
    assert sim.world.players[0].pos == (1, 1) and sim.world.players[0].facing == NORTH
    assert sim.world.players[0].held == ("raw", "onion")
    sim.tick()
    assert sim.t == 6 and sim.world.players[0].held == ("raw", "onion")
    assert sim.legal("c1") == [WAIT]
    with pytest.raises(ValueError):
        Sim("k01_one_soup").drive("c1", WEST)
    with pytest.raises(ValueError):
        sim.drive("c1", "jump")
    sim.set_mode("c1", "task")
    assert sim.pending() == ["c1"]


DEAD_END = {"scenario_id": "dead_end", "plates": {"clean": 0}, "orders": [{"recipe": {"tomato": 1}}],
            "grid": ["XXXXXX",
                     "O 12 T",
                     "XXXXXX"]}


def test_grid_deadlock_after_stuck_steps():
    sim = Sim(DEAD_END, clock="grid", stuck_steps=10)
    want = {"c1": "tomato_crate", "c2": "onion_crate"}

    def policy(s, cook, legal):
        return next((t for t in legal if t != WAIT and s.world.kind[t.cell] == want[cook]), WAIT)

    m = sim.run(policy)
    assert m["failure"] == "deadlock" and not m["success"]
    assert sim.t == 10
    assert not [e for e in sim.log if e["type"] in ("yield", "end")]
    assert m["blocked"] == {"c1": 10, "c2": 10}


def test_grid_task_fails_softly_like_sampled():
    sc = {"scenario_id": "k01_burn1", "grid": Sim("k01_one_soup").scenario["grid"], "sim": {"burn": 1},
          "plates": {"clean": 1}, "orders": [{"recipe": {"tomato": 3}}]}
    logs = []
    for clock in ("sampled", "grid"):
        sim = Sim(sc, clock=clock, horizon=300)
        sim.run(serial_policy)
        logs.append([e for e in sim.log if e["type"] in ("end", "task_failed", "burnt")])
    failed = [e for e in logs[1] if e["type"] == "task_failed"]
    assert failed and failed[0]["event"] == "dish_up" and failed[0]["found"] != "dish_up"
    assert logs[0] == logs[1]


POCKET = {"scenario_id": "pocket", "plates": {"clean": 0}, "orders": [{"recipe": {"tomato": 1}}],
          "grid": ["XXXXXXXXX",
                   "O 1   2 T",
                   "XXXX XXXX",
                   "XXXXXXXXX"]}


def test_head_on_walkers_step_aside():
    sim = Sim(POCKET, clock="grid", stuck_steps=10)
    want = {"c1": "tomato_crate", "c2": "onion_crate"}
    started = set()

    def policy(s, cook, legal):
        if cook in started:
            return WAIT
        started.add(cook)
        return next(t for t in legal if t != WAIT and s.world.kind[t.cell] == want[cook])

    sim.run(policy)
    assert not sim.running and sim.t == max(e["t"] for e in sim.log)
    ends = [e for e in sim.log if e["type"] == "end"]
    assert sorted(e["cook"] for e in ends) == ["c1", "c2"]
    assert sim.world.players[0].held == ("raw", "tomato") and sim.world.players[1].held == ("raw", "onion")
    asides = [e for e in sim.log if e["type"] == "step_aside"]
    assert asides and asides[0]["cook"] == "c1" and asides[0]["for"] == "c2"
    assert not [e for e in sim.log if e["type"] == "yield"]


def test_driven_cook_changes_station_task_fails_without_effect():
    sc = {"scenario_id": "lock", "plates": {"clean": 0}, "orders": [{"recipe": {"onion": 1}}],
          "grid": ["XXOXX", "X1B2X", "XXXXX"]}
    sim = Sim(sc, clock="grid")
    w = sim.world
    board = (2, 1)
    w.surface[board] = ("raw", "onion")
    w.players[0].facing, w.players[1].facing = EAST, WEST
    sim.set_mode("c1", "drive")
    sim.drive("c1", grid.INTERACT)
    sim.tick()
    chop = next(t for t in sim.legal("c2") if t != WAIT and t.event == "chop")
    sim.act("c2", chop)
    for a in (grid.INTERACT, grid.INTERACT, grid.STAY):
        sim.drive("c1", a)
        sim.tick()
    assert w.players[1].held is None
    assert w.surface[board] == ("chopped", "onion")
    failed = [e for e in sim.log if e["type"] == "task_failed"]
    assert len(failed) == 1 and failed[0]["cook"] == "c2"
    assert failed[0]["event"] == "chop" and failed[0]["found"] == "take_chopped"
    assert "c2" not in sim.running


def test_move_task_on_grid_ends_at_its_pose():
    sim = Sim("k01_one_soup", clock="grid")
    c = sim.next_decision()
    mv = sim.move_task(c, station="pot1")
    sim.act(c, mv)
    assert sim.next_decision() == c and sim.t == mv.travel
    p = sim.world.players[0]
    assert (p.pos[0], p.pos[1], p.facing) == mv.pose
    assert sim.log[-1]["type"] == "end" and sim.log[-1]["action"] == "move"


def _blocked_pair(**kw):
    sim = Sim(DEAD_END, clock="grid", **kw)
    task = next(t for t in sim.legal("c1") if t != WAIT and sim.world.kind[t.cell] == "tomato_crate")
    sim.act("c1", task)
    return sim


def test_stuck_steps_none_disables_deadlock():
    sim = _blocked_pair(stuck_steps=None)
    sim.act("c2", WAIT)
    for _ in range(60):
        sim.tick()
    assert sim.failure is None


def test_no_deadlock_while_a_cook_is_driven():
    sim = _blocked_pair(stuck_steps=10)
    sim.set_mode("c2", "drive")
    for _ in range(30):
        sim.tick()
    assert sim.failure is None
    sim.set_mode("c2", "auto")
    for _ in range(12):
        sim.tick()
    assert sim.failure == "deadlock"
