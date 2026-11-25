"""Tests for the baseline policies: the rule-based dispatcher and uniform random."""
"""This file is UN-VERIFIED"""
import pytest

from overcooked_kitchen.baselines import Dispatcher, make
from overcooked_kitchen.core import Sim
from overcooked_kitchen.scenarios import examples
from overcooked_kitchen.tasks import WAIT

EXAMPLES = ["k01_one_soup", "k02_two_dishes", "k03_service_cleanup", "k04_burn_risk", "m01_three_cooks",
            "m02_four_cooks"]


@pytest.mark.parametrize("name", EXAMPLES)
def test_dispatcher_serves_examples(name):
    m = Sim(name, noise=None).run(make("dispatcher"))
    assert m["success"], m
    assert m["burnt"] == 0 and m["spoiled"] == 0, m


@pytest.mark.parametrize("name", ["m01_three_cooks", "m02_four_cooks"])
def test_dispatcher_under_noise(name):
    ok = [Sim(name, seed=s, noise="default", cv_scale=1).run(make("dispatcher"))["success"] for s in range(5)]
    assert sum(ok) >= 4, ok


def test_dispatcher_grid_clock():
    m = Sim("m01_three_cooks", clock="grid").run(make("dispatcher"))
    assert m["success"], m


@pytest.mark.parametrize("name", EXAMPLES)
def test_random_never_crashes(name):
    for seed in range(3):
        m = Sim(name, seed=seed, horizon=3000).run(make("random", seed=seed))
        assert m["failure"] in (None, "horizon", "deadlock", "stalled")
        assert m["decisions"] > 0


def test_random_is_seeded_and_picks_legal():
    a = Sim("m01_three_cooks", horizon=500).run(make("random", seed=3))
    b = Sim("m01_three_cooks", horizon=500).run(make("random", seed=3))
    a.pop("wall_s")
    b.pop("wall_s")
    assert a == b
    sim = Sim("m01_three_cooks")
    legal = sim.legal("c1")
    pol = make("random", seed=0)
    assert all(pol(sim, "c1", legal) in legal for _ in range(20))


def test_random_waits_only_when_nothing_else_is_legal():
    """WAIT never while a task is legal; with a horizon some seed serves an order (k01 random runs always reach a
    dead end with both boards and the hand full of raw veg, so m01 is used)."""
    pol = make("random", seed=1)
    for name in ("k01_one_soup", "m01_three_cooks"):
        sim = Sim(name, horizon=3000)
        while True:
            c = sim.next_decision()
            if c is None:
                break
            legal = sim.legal(c)
            choice = pol(sim, c, legal)
            assert (choice == WAIT) == (legal == [WAIT])
            sim.act(c, choice)
    assert pol(sim, "c1", [WAIT]) == WAIT
    served = [Sim("m01_three_cooks", seed=s, horizon=3000).run(make("random", seed=s))["served"] for s in range(20)]
    assert max(served) >= 1, served


def test_dispatcher_is_pure_and_returns_legal():
    sim = Sim("m01_three_cooks")
    d = Dispatcher()
    for _ in range(60):
        c = sim.next_decision()
        if c is None:
            break
        legal = sim.legal(c)
        choice = d(sim, c, legal)
        assert choice == WAIT or choice in legal
        assert Dispatcher()(sim, c, legal) == choice
        sim.act(c, choice)


def test_dispatcher_claims_a_soup_once():
    """Three idle cooks and one soup ready soon: one fetches a plate (a claim seen through sim.running), the
    others keep cooking."""
    sim = Sim("m01_three_cooks")
    pot = sim.world.pots[(9, 0)]
    pot.contents, pot.state, pot.cook_start, pot.ready_at = ["onion", "onion"], "cooking", -10, 3
    d = make("dispatcher")
    choices = []
    for _ in sim.cooks:
        c = sim.next_decision()
        choices.append(d(sim, c, sim.legal(c)))
        sim.act(c, choices[-1])
    assert sim.t == 0
    assert [t.action for t in choices if t != WAIT].count("take-clean-plate") == 1
    assert all(t != WAIT for t in choices)


def test_dispatcher_initial_fetches_follow_needs():
    """At t=0 the cooks fetch distinct needed vegetables, not the same one twice beyond its need."""
    sim = Sim("m01_three_cooks")
    d = make("dispatcher")
    takes = []
    for _ in sim.cooks:
        c = sim.next_decision()
        choice = d(sim, c, sim.legal(c))
        sim.act(c, choice)
        takes.append(choice)
    assert all(t != WAIT and t.event == "take_raw" for t in takes)
    assert len({t.cell for t in takes}) == len(takes)


def test_make_unknown():
    with pytest.raises(ValueError):
        make("nope")
    assert set(EXAMPLES) <= set(examples())
