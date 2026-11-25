"""Tests for the PDDL bridge: domain, export from live (and messy) states, POPF plans executed in the simulator."""
"""This file is UN-VERIFIED"""
import os
import subprocess

import pytest
import yaml

from overcooked_kitchen import pddl, suite
from overcooked_kitchen.baselines import make
from overcooked_kitchen.core import Sim
from overcooked_kitchen.pddlkit import TemporalPlan, TimedAction, planners
from overcooked_kitchen.scenarios import example
from overcooked_kitchen.tasks import WAIT

SUITE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "v1")
# PDDL travel ignores the other cooks on the way (only target tiles are reserved); on the grid clock cooks detour
# around each other, so delays can add up. Here 6 steps by t = 110 on the critical chain (c1 36-44 walks around c2 at
# (1, 3), c2 91-110 ...) make c1 reach pot2 at 124, after its soup burnt at 122 (dish-up planned at 118).
GRID_DETOURS = {"v3_val_m3_003"}
needs_planner = pytest.mark.skipif(not (planners.available("popf") and planners.available("validate")),
                                   reason="POPF / VAL binaries not found")
needs_parser = pytest.mark.skipif(not planners.available("parser"), reason="VAL Parser not found")
K02_TIMEOUT = 300.0     # POPF needs ~90 s on k02 from t = 0 here (v2 README: 43 s); later states solve in seconds


def _parses(d, p=None):
    r = planners.parse_check(d, p)
    assert r.ok, r.output[-2000:]


def _holding_soup_and_cooking(w):
    soup = any(p.held and p.held[0] == "plate" and w.plates[p.held[1]].state == "soup" for p in w.players)
    return soup and any(pot.state == "cooking" for pot in w.pots.values())


@needs_planner
@needs_parser
def test_domain_has_26_actions_and_parses(tmp_path):
    assert len(pddl.spec().actions) == 26
    path = tmp_path / "domain.pddl"
    path.write_text(pddl.domain_text())
    _parses(str(path))


@needs_planner
@pytest.mark.parametrize("name", ["k01_one_soup", pytest.param("k02_two_dishes", marks=pytest.mark.slow)])
def test_popf_val_core_and_grid(name, tmp_path):
    sim = Sim(name)
    d, p = pddl.write(sim, str(tmp_path))
    problem = pddl.export(sim)
    plan = pddl.solve(sim, timeout=K02_TIMEOUT)
    assert plan is not None
    assert pddl.validate(d, p, plan)
    for clock in ("sampled", "grid"):
        m = Sim(name, clock=clock).run(pddl.PlanPolicy(plan, problem))
        assert m["success"], (clock, m)
        assert m["burnt"] == 0 and m["late"] == 0 and m["on_time"] == 1.0, (clock, m)


@needs_planner
@needs_parser
def test_export_mid_run_holding_soup(tmp_path):
    """k02, one cook: the dispatcher (and POPF's plan) serve one dish at a time, so a plated soup in hand never
    meets a cooking pot. Run the dispatcher to the first plated soup in hand, then complete the second pot
    (one onion conjured) and light it through the world's own start event."""
    sim = Sim("k02_two_dishes")
    pol = make("dispatcher")
    w = sim.world
    while True:
        c = sim.next_decision()
        assert c is not None
        if any(p.held and p.held[0] == "plate" and w.plates[p.held[1]].state == "soup" for p in w.players):
            break
        sim.act(c, pol(sim, c, sim.legal(c)))
    cell = next(c for c, pot in sorted(w.pots.items()) if pot.state == "idle" and pot.contents)
    w.pots[cell].contents = ["carrot", "carrot", "onion"]
    w.apply(0, "start", cell, sim.t)
    assert _holding_soup_and_cooking(w) and not sim.running
    d, p = pddl.write(sim, str(tmp_path))
    _parses(d, p)
    problem = pddl.export(sim)
    facts = set(problem.facts)
    assert any(f[0] == "holding-plate" for f in facts) and any(f[0] == "plate-soup" for f in facts)
    assert not any(f[0] == "hand-empty" for f in facts)
    assert any(f[0] == "on-stove" for f in facts)
    assert any(tl.fact[0] == "soup-ready" and tl.value for tl in problem.timed_literals)
    assert any(tl.fact[0] == "on-stove" and not tl.value for tl in problem.timed_literals)
    assert problem.annotations["t_now"] == sim.t
    plan = pddl.solve(sim, timeout=60)
    assert plan is not None
    assert pddl.validate(d, p, plan)
    m = sim.run(pddl.PlanPolicy(plan, problem))
    assert m["success"] and m["burnt"] == 0 and m["late"] == 0, m


@needs_planner
@needs_parser
@pytest.mark.parametrize("clock", ["sampled", "grid"])
def test_export_mid_run_with_running_tasks(clock, tmp_path):
    """m01 (3 cooks) under the dispatcher: a cook holds a plated soup, a pot cooks, other cooks are mid-task."""
    sim = Sim("m01_three_cooks", clock=clock)
    pol = make("dispatcher")
    while True:
        c = sim.next_decision()
        assert c is not None
        if _holding_soup_and_cooking(sim.world) and sim.running:
            break
        sim.act(c, pol(sim, c, sim.legal(c)))
    d, p = pddl.write(sim, str(tmp_path))
    _parses(d, p)
    problem = pddl.export(sim)
    facts = set(problem.facts)
    tils = {(tl.fact, tl.value) for tl in problem.timed_literals}
    for cook, r in sim.running.items():
        assert ("idle", cook) not in facts
        assert not any(f[0] == "at" and f[1] == cook for f in facts)
        assert (("idle", cook), True) in tils
        assert any(f[0] == "at" and f[1] == cook and v for f, v in tils)
    for cook in sim.cooks:
        if cook not in sim.running:
            assert ("idle", cook) in facts


@needs_planner
@needs_parser
def test_export_messy_states(tmp_path):
    with open(example("k02_two_dishes")) as f:
        sc = yaml.safe_load(f)
    sc["orders"][0]["deadline"] = 2
    sim = Sim(sc)
    pol = make("dispatcher")
    while sim.t <= 2:
        c = sim.next_decision()
        sim.act(c, pol(sim, c, sim.legal(c)))
    c = sim.next_decision()
    assert c is not None and not sim.running
    w = sim.world
    burnt, spoiled = sorted(w.pots)
    w.pots[burnt].contents, w.pots[burnt].state = ["tomato", "tomato"], "burnt"
    w.pots[spoiled].contents, w.pots[spoiled].state, w.pots[spoiled].recipe = ["onion"], "spoiled", False
    board = next(b for b, k in sorted(w.kind.items()) if k == "board" and w.surface[b] is None)
    w.surface[board], w.progress[board] = ("raw", "onion"), 2
    d, p = pddl.write(sim, str(tmp_path))
    _parses(d, p)
    problem = pddl.export(sim)
    names = {tuple(o.attrs["cell"]): o.name for o in problem.objects if o.type in ("pot", "board")}
    facts = set(problem.facts)
    assert ("pot-dirty", names[burnt]) in facts and ("pot-dirty", names[spoiled]) in facts
    assert ("on-raw", names[board], "onion") in facts
    assert ("chop-steps", "onion") in problem.fluents and problem.fluents[("chop-steps", "onion")] == 4
    goals = {g.fact[1]: g.deadline for g in problem.goals}
    assert goals["o1"] is None and "o1" in problem.annotations["dropped_deadlines"]
    assert goals["o2"] == 160 - sim.t
    loose = pddl.export(sim, hard_deadlines=False)
    assert all(g.deadline is None for g in loose.goals)


@needs_planner
def test_popf_baseline_k01():
    pol = make("popf")
    m = Sim("k01_one_soup").run(pol)
    assert m["success"] and m["burnt"] == 0, m
    assert len(pol.solves) == 1 and pol.solves[0]["status"] == "solved"


@needs_planner
@pytest.mark.slow
def test_popf_replan_baseline_k02_noise():
    pol = make("popf-replan", timeout=K02_TIMEOUT)
    m = Sim("k02_two_dishes", seed=0, noise="default", cv_scale=1).run(pol)
    assert m["success"], (m, pol.solves)
    assert 2 <= len(pol.solves) <= 11, pol.solves          # it did re-plan, within the bound


@needs_planner
@pytest.mark.parametrize("burn", [4, 6])
def test_tight_burn_plans_run_without_burning(burn, tmp_path):
    """The plan dishes up close to the burn deadline: the cook must walk to the pot before the soup is ready."""
    with open(example("k01_one_soup")) as f:
        sc = yaml.safe_load(f)
    sc["sim"] = {"burn": burn}
    sim = Sim(sc)
    d, p = pddl.write(sim, str(tmp_path))
    problem = pddl.export(sim)
    plan = pddl.solve(sim, timeout=120)
    assert plan is not None and pddl.validate(d, p, plan)
    for clock in ("sampled", "grid"):
        for mode in ("asap", "timed"):
            run = Sim(sc, clock=clock)
            m = run.run(pddl.PlanPolicy(plan, problem, mode))
            assert m["success"] and m["burnt"] == 0, (clock, mode, m)
            assert any(e["type"] == "start" and e["action"] == "move" for e in run.log), (clock, mode)


def test_planner_process_group_killed_on_interrupt(monkeypatch):
    procs = []
    real_wait = subprocess.Popen.wait

    def wait(self, timeout=None):
        procs.append(self)
        if timeout is not None:
            raise KeyboardInterrupt
        return real_wait(self)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    with pytest.raises(KeyboardInterrupt):
        planners._run(["sleep", "30"], timeout=10)
    monkeypatch.undo()
    assert procs and procs[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(procs[0].pid, 0)


class _Stub:
    """Planner stub: returns the given plans in turn, then fails; counts calls."""

    def __init__(self, *plans):
        self.plans, self.calls = list(plans), 0

    def __call__(self, problem, planner, timeout):
        self.calls += 1
        plan = self.plans.pop(0) if self.plans else None
        return planners.PlannerResult("solved" if plan else "unsolvable", plan, 0.0, None, int(bool(plan)), planner, [])


def test_planner_policy_retries_initial_solve_once_after_time_advances():
    stub = _Stub()
    pol = make("popf", solver=stub)
    sim = Sim("k01_one_soup")
    for _ in range(3):
        assert pol(sim, "c1", sim.legal("c1")) == WAIT
    assert stub.calls == 1                 # the same state is not solved twice
    sim.t = 5
    for _ in range(3):
        assert pol(sim, "c1", sim.legal("c1")) == WAIT
    assert stub.calls == 2 and [x["status"] for x in pol.solves] == ["unsolvable", "unsolvable"]
    assert [x["t"] for x in pol.solves] == [0, 5]
    assert make("popf").timeout == 300.0


def test_planner_policy_backs_off_after_failed_resolve():
    late = TemporalPlan([TimedAction(0.0, "take-raw", ("c1", "p2_1n", "tcrate1", "tomato"), 1.0)])
    stub = _Stub(late)
    pol = make("popf-replan", solver=stub, tol=1)
    sim = Sim("k01_one_soup")
    m = sim.run(pol)                       # take-raw ends at 4, planned 1: re-solve fails, the old plan is done
    assert m["failure"] == "stalled" and stub.calls == 2
    for _ in range(3):
        assert pol(sim, "c1", sim.legal("c1")) == WAIT
    assert stub.calls == 2 and pol.replans == 1


def _certificates(*splits):
    rows = [r for split in splits for r in suite.read_index(SUITE) if r["split"] == split and r["cert_kind"] == "val"]
    return [pytest.param(r, clock, id="{}-{}".format(r["name"], clock),
                         marks=[pytest.mark.xfail(strict=True, reason="grid detours")]
                         if clock == "grid" and r["name"] in GRID_DETOURS else [])
            for r in rows for clock in ("sampled", "grid")]


def _replay(row, clock, mode="asap"):
    sc = yaml.safe_load(open(os.path.join(SUITE, row["path"])))
    plan = TemporalPlan.parse(open(os.path.join(SUITE, row["cert_plan"])).read())
    sim = Sim(sc, clock=clock)
    pol = pddl.PlanPolicy(plan, pddl.export(sim), mode)
    return sim, pol, sim.run(pol)


@pytest.mark.parametrize("row,clock", _certificates("v2_val", "v3_val"))
def test_shipped_certificates_replay_nominal(row, clock):
    """Every v2_val / v3_val certificate runs in Sim with nominal durations: all served, no burn, no lateness."""
    sim, pol, m = _replay(row, clock)
    assert m["success"] and m["burnt"] == 0 and m["late"] == 0, (row["name"], clock, m)
    if clock == "sampled":
        assert m["makespan"] <= row["cert_makespan"] + 1e-6, (m["makespan"], row["cert_makespan"])


@pytest.mark.parametrize("clock", ["sampled", "grid"])
def test_plan_moves_hand_over_tiles(clock):
    """v3_val_m3_000: c2 holds a soup for serve1 whose pose (4, 1) c3 occupies; c3 must scrub pot1 from (2, 1),
    where c2 stands. The plan resolves it with a move of c2 to a tile with no station (2, 5) at 77.031."""
    row = next(r for r in suite.read_index(SUITE) if r["name"] == "v3_val_m3_000")
    sim, pol, m = _replay(row, clock)
    assert m["success"] and m["burnt"] == 0 and m["late"] == 0, m
    moves = [s for s in pol.steps if s.cook == "c2" and s.kind == "move" and s.pose == (2, 5, (0, 1))]
    assert moves and all(s.sim_start is not None for s in moves)
    c3 = next(s for s in pol.steps if s.cook == "c3" and s.kind == "move" and s.pose[:2] == (2, 1)
              and s.start > 77)
    assert any(u is moves[0] for u in c3.preds) and c3.sim_start >= moves[0].sim_start
    assert any(e["type"] == "start" and e["cook"] == "c2" and e["action"] == "move" for e in sim.log)


def test_plan_steps_and_predecessors():
    plan = TemporalPlan([TimedAction(0.0, "move", ("c1", "p2_1n", "p3_1n", "t2_1", "t3_1"), 1.0),
                         TimedAction(0.5, "move", ("c2", "p4_1n", "p2_1n", "t4_1", "t2_1"), 2.0),
                         TimedAction(1.0, "turn", ("c1", "p3_1n", "p3_1s"), 1.0)])
    problem = pddl.export(Sim("k01_one_soup"))
    pol = pddl.PlanPolicy(plan, problem)
    m1, m2, turn = pol.steps
    assert (m1.kind, m1.src, m1.pose) == ("move", (2, 1), (3, 1, (0, -1)))
    assert m2.preds == [m1] and m1.preds == [] and turn.src is None and turn.preds == []
