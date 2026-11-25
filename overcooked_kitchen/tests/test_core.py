"""Tests for the core engine and the sampled clock."""
"""This file is UN-VERIFIED"""
import dataclasses

import numpy as np
import pytest

from helpers import serial_policy
from overcooked_kitchen.clocks import Noise, SampledClock
from overcooked_kitchen.core import Sim
from overcooked_kitchen.scenarios import example, load_scenario
from overcooked_kitchen.tasks import WAIT

METRIC_KEYS = {"success", "failure", "makespan", "served", "late", "total_lateness", "on_time", "burnt",
               "spoiled", "decisions", "cook_utilisation", "idle", "blocked", "wall_s"}


def _k01(**over):
    sc = load_scenario(example("k01_one_soup"))
    sc.update(over)
    return sc


def test_k01_scripted_nominal_makespan():
    sim = Sim("k01_one_soup", noise=None)
    m = sim.run(serial_policy)
    assert set(m) == METRIC_KEYS
    assert m["success"] and m["failure"] is None
    assert m["burnt"] == 0
    assert m["makespan"] <= 1.5 * 69
    assert m["served"] == 1 and m["on_time"] == 1.0
    assert m["blocked"] == {"c1": 0}
    assert 0 < m["cook_utilisation"]["c1"] <= 1


@pytest.mark.parametrize("name", ["k01_one_soup", "k02_two_dishes", "k03_service_cleanup", "k04_burn_risk"])
def test_serial_policy_serves_single_cook_examples(name):
    m = Sim(name).run(serial_policy)
    assert m["success"], m
    assert m["burnt"] == 0 and m["spoiled"] == 0


def test_nominal_durations_ignore_cv_scale():
    a = Sim("k03_service_cleanup", noise=None, cv_scale=5.0).run(serial_policy)
    b = Sim("k03_service_cleanup").run(serial_policy)
    assert a["makespan"] == b["makespan"]


def test_station_lock_same_round():
    sim = Sim("m01_three_cooks")
    assert sim.cooks == ["c1", "c2", "c3"]
    crate = next(t for t in sim.legal("c1") if t != WAIT and t.event == "take_raw")
    assert any(t.cell == crate.cell for t in sim.legal("c2") if t != WAIT)
    sim.act("c1", crate)
    assert "c1" in sim.running
    assert not [t for t in sim.legal("c2") if t != WAIT and t.cell == crate.cell]
    tile = crate.pose[:2]
    assert not [t for t in sim.legal("c2") if t != WAIT and t.pose[:2] == tile]


def test_occupied_tile_blocked():
    sim = Sim("m01_three_cooks")
    pos2 = sim.world.players[1].pos
    assert not [t for t in sim.legal("c1") if t != WAIT and t.pose[:2] == pos2]


def test_act_rejects_illegal_task():
    sim = Sim("m01_three_cooks")
    crate = next(t for t in sim.legal("c1") if t != WAIT and t.event == "take_raw")
    sim.act("c1", crate)
    with pytest.raises(ValueError):
        sim.act("c2", crate)
    with pytest.raises(ValueError):
        sim.act("c1", WAIT)


def test_pending_once_per_event_time():
    sim = Sim("m01_three_cooks")
    assert sim.pending() == ["c1", "c2", "c3"]
    sim.act("c1", WAIT)
    assert sim.pending() == ["c2", "c3"]
    assert sim.next_decision() == "c2"
    t0 = sim.t
    sim.act("c2", next(t for t in sim.legal("c2") if t != WAIT))
    sim.act("c3", WAIT)
    assert sim.next_decision() == "c1"
    assert sim.t > t0


def test_task_fails_softly_when_soup_burns_en_route():
    sim = Sim(_k01(sim={"burn": 1}))
    while True:
        c = sim.next_decision()
        failed = next((e for e in sim.log if e["type"] == "task_failed"), None)
        if failed is not None:
            break
        assert c is not None, sim.metrics()
        sim.act(c, serial_policy(sim, c, sim.legal(c)))
    assert c == "c1"
    assert failed["event"] == "dish_up" and failed["found"] != "dish_up"
    assert "c1" not in sim.running
    assert sim.world.players[0].held[0] == "plate"
    assert sim.world.plates[sim.world.players[0].held[1]].state == "clean"
    m = sim.metrics()
    assert m["burnt"] == 1 and m["failure"] is None
    sim.act("c1", serial_policy(sim, "c1", sim.legal("c1")))
    assert "c1" in sim.running


def test_noise_reproducible():
    runs = [Sim("k03_service_cleanup", seed=s, noise="default", cv_scale=1).run(serial_policy) for s in (0, 0, 1)]
    for m in runs:
        m.pop("wall_s")
    assert runs[0] == runs[1]
    assert runs[0]["success"] and runs[2]["success"]
    assert runs[0]["makespan"] != runs[2]["makespan"]


def test_noise_profile_and_sampling():
    n = Noise.default_profile(2.0)
    assert n.cv == {"move": 0.5, "chop": 0.4, "wash": 0.4, "scrub": 0.4, "interact": 0.2, "cook": 0.2, "eat": 0.6}
    rng = np.random.default_rng(0)
    xs = [Noise.default_profile().sample("move", 10, rng) for _ in range(4000)]
    assert abs(np.mean(xs) - 10) < 0.3
    assert min(xs) >= 4.0 and max(xs) <= 30.0
    assert Noise().sample("chop", 4, rng) == 4
    assert Noise.default_profile().sample("chop", 0, rng) == 0


def test_sampled_clock_nominal_duration():
    sim = Sim("k01_one_soup")
    assert isinstance(sim.clock, SampledClock)
    t = next(t for t in sim.legal("c1") if t != WAIT)
    sim.act("c1", t)
    assert sim.running["c1"].end == t.duration


def test_noisy_cook_time_overwrites_timer():
    sim = Sim("k01_one_soup", seed=3, noise=Noise({"cook": 0.5}))
    while True:
        c = sim.next_decision()
        assert c is not None
        sim.act(c, serial_policy(sim, c, sim.legal(c)))
        start = next((e for e in sim.log if e["type"] == "end" and e["event"] == "start"), None)
        if start:
            break
    pot = next(p for p in sim.world.pots.values() if p.state == "cooking")
    assert pot.ready_at != start["t"] - 1 + 2 + 3 * 3


def test_deadlock_detected():
    sc = {"scenario_id": "no_carrot", "grid": ["XPXSX", "X 1 X", "XXXXX"], "plates": {"clean": 0},
          "orders": [{"recipe": {"carrot": 1}}]}
    sim = Sim(sc)
    m = sim.run(serial_policy)
    assert m["failure"] == "deadlock" and not m["success"] and m["makespan"] is None


def test_stalled_detected():
    m = Sim("k01_one_soup").run(lambda sim, cook, legal: WAIT)
    assert m["failure"] == "stalled"


def test_horizon():
    m = Sim("k01_one_soup", horizon=20).run(serial_policy)
    assert m["failure"] == "horizon" and not m["success"]


def test_record_frames():
    sim = Sim("k01_one_soup", record=True)
    sim.run(serial_policy)
    assert sim.frames and sim.frames[0]["t"] == 0
    assert [f["t"] for f in sim.frames] == sorted({f["t"] for f in sim.frames})
    assert sim.frames[-1]["t"] == sim.t
    assert "world" in sim.frames[0] and "running" in sim.frames[0]


def test_effect_before_burn_inside_task_succeeds():
    sim = Sim("k01_one_soup")
    w = sim.world
    rack = next(c for c, k in w.kind.items() if k == "rack")
    w.players[0].held = ("plate", w.stack[rack].pop())
    cell = min(w.pots, key=lambda c: (c[1], c[0]))
    pot = w.pots[cell]
    pot.contents, pot.state = ["tomato"] * 3, "ready"
    task = next(t for t in sim.legal("c1") if t != WAIT and t.event == "dish_up" and t.cell == cell)
    pot.ready_at = task.duration - 0.5 - w.params["burn"]
    assert task.duration - 1 < w.next_timer() < task.duration
    sim.act("c1", task)
    assert sim.next_decision() == "c1"
    assert sim.t == task.duration
    end = sim.log[-1]
    assert end["type"] == "end" and end["event"] == "dish_up"
    assert sim.metrics()["burnt"] == 0
    assert w.plates[w.players[0].held[1]].state == "soup"


def test_noisy_effects_judged_at_effect_time():
    for seed in range(10):
        sc = _k01(sim={"burn": 3})
        sim = Sim(sc, seed=seed, noise="default", cv_scale=1.5, horizon=400)
        sim.run(serial_policy)
        burns = [e for e in sim.log if e["type"] == "burnt"]
        for e in sim.log:
            if e["type"] == "task_failed" and e["event"] == "dish_up":
                assert not [b for b in burns if b["cell"] == e["cell"] and e["t"] - 1 < b["t"] < e["t"]], (seed, e)


def test_move_task_walks_ahead_and_is_not_offered():
    sim = Sim("k01_one_soup")
    c = sim.next_decision()
    mv = sim.move_task(c, station="tcrate1")
    assert mv is not None and (mv.event, mv.action, mv.station, mv.cell, mv.strokes) == ("move", "move", None, None, 0)
    assert mv not in sim.legal(c)
    take = next(t for t in sim.legal(c) if t != WAIT and t.station == "tcrate1")
    assert mv.pose == take.pose and mv.travel == take.travel
    sim.act(c, mv)
    assert sim.move_task(c, station="pot1") is None                     # busy
    assert sim.next_decision() == c and sim.t == mv.travel
    p = sim.world.players[0]
    assert (p.pos[0], p.pos[1], p.facing) == mv.pose and p.held is None
    assert sim.log[-1]["type"] == "end" and sim.log[-1]["action"] == "move"
    assert sim.move_task(c, station="tcrate1") is None                  # already there
    take = next(t for t in sim.legal(c) if t != WAIT and t.station == "tcrate1")
    assert take.travel == 0


def test_act_validates_move_tasks():
    sim = Sim("m01_three_cooks")
    c = sim.next_decision()
    mv = sim.move_task(c, station="serve1")
    assert mv is not None
    with pytest.raises(ValueError):
        sim.act(c, dataclasses.replace(mv, travel=mv.travel - 1))
    other = sim.world.players[1]
    taken = next(q for q in sim.world.lay.bfs_poses((other.pos[0], other.pos[1], other.facing))
                 if q[:2] == tuple(other.pos))
    assert sim.move_task(c, pose=taken) is None
    with pytest.raises(ValueError):
        sim.act(c, dataclasses.replace(mv, pose=taken))
    assert sim.move_task(c, station="nowhere") is None
    sim.act(c, mv)
    assert "serve1" not in {r.task.station for r in sim.running.values()}


def _corridor():
    """c1 holds a raw onion with no board or counter (only WAIT is legal); c2 can take a tomato."""
    sc = {"scenario_id": "corridor", "grid": ["XXXXXXX", "O1   2T", "XXXSDXX"],
          "orders": [{"dish": "onion_soup", "recipe": {"onion": 2}}]}
    sim = Sim(sc)
    sim.world.players[0].held = ("raw", "onion")
    return sim


def test_pending_include_idle_offers_wait_only_cooks():
    sim = _corridor()
    assert sim.legal("c1") == [WAIT]
    assert sim.pending() == ["c2"]
    assert sim.pending(include_idle=True) == ["c1", "c2"]
    sim.act("c1", WAIT)
    assert sim.pending(include_idle=True) == ["c2"]
    sim.act("c2", next(t for t in sim.legal("c2") if t != WAIT))
    assert sim.pending() == []
    assert sim.pending(include_idle=True) == ["c1"]            # offered again: c2 started a task since c1 waited
    sim.act("c1", sim.move_task("c1", pose=(3, 1, (1, 0))))
    assert sim.pending(include_idle=True) == []


def test_run_uses_policy_include_idle():
    asked = []

    class Idle:
        include_idle = True

        def __call__(self, sim, cook, legal):
            asked.append((sim.t, cook, len(legal)))
            return WAIT

    def plain(sim, cook, legal):
        asked.append((sim.t, cook, len(legal)))
        return WAIT

    _corridor().run(plain)
    assert [x[:2] for x in asked] == [(0, "c2")]
    del asked[:]
    _corridor().run(Idle())
    assert [x[:2] for x in asked] == [(0, "c1"), (0, "c2")] and asked[0][2] == 1


def test_unknown_cook_rejected():
    sim = Sim("k01_one_soup")
    with pytest.raises(ValueError):
        sim.set_mode("c9", "task")
    with pytest.raises(ValueError):
        sim.act("c9", WAIT)
