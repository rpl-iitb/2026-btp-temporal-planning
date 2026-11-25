"""The hard suite's PDDL (lane traffic, hard deadlines) in algorithm/menv.py with the robot dispatcher: it finishes
under random durations, its nominal trajectory is a VAL-valid plan, and a learned policy sees the traffic features."""
import pytest

from factory_sim import pddl as FP
from factory_sim.env import build_sim
from factory_sim.scenarios import HARD_FAMILIES, make_instance, set_dues
from algorithm import v10
from algorithm.menv import MAEnv
from algorithm.pddlkit import TemporalPlan, validate_plan
from factory_test.closed_eval import plan_text
from factory_test.factory_dispatcher import NF, RULES, Dispatcher, View, features
from factory_test.factory_v13 import NOISE


def task(family, size, deadlines=False, index=0):
    inst = make_instance(family, size, "train", index)
    sim = build_sim(inst, seed=0, time_source="nominal")
    dom, prob = FP.domain(sim.F), FP.export(sim, hard_deadlines=deadlines, name=inst["name"])
    return inst, dom, prob, v10.Task(v10.Domain(dom), v10.Problem(prob))


def rollout(T, noise, seed, horizon):
    env = MAEnv(T, "robot", noise=noise, seed=seed, horizon=horizon)
    D = Dispatcher(env)
    return env.rollout(lambda e, S, a, lg: D(e, S, a, lg), seed=seed), env


@pytest.mark.parametrize("family", HARD_FAMILIES)
def test_dispatcher_finishes_under_noise(family):
    for size in ("S", "M"):
        inst, _, _, T = task(family, size, index=1)
        ref = inst["ground_truth"].get("best_known") or 3 * inst["ground_truth"]["lower_bound"]
        for nk in (1, 3):
            r, _ = rollout(T, NOISE.scaled(nk), seed=nk, horizon=4 * ref)
            assert r["success"], (inst["name"], nk, r["reason"])


@pytest.mark.parametrize("family", ["hard_base", "hard_flexible", "hard_blocking"])
def test_nominal_dispatcher_plan_is_val_valid(family, tmp_path):
    """With due dates from its own makespan, the dispatcher's nominal trajectory is a valid plan for the PDDL with
    hard deadlines and lane traffic (junction reservations as written in the domain)."""
    inst, dom, prob, T = task(family, "S")
    r, _ = rollout(T, None, 0, 10 ** 6)
    assert r["success"], r["reason"]
    set_dues(inst, r["makespan"])
    sim = build_sim(inst, seed=0, time_source="nominal")
    prob = FP.export(sim, hard_deadlines=True, name=inst["name"])
    T = v10.Task(v10.Domain(dom), v10.Problem(prob))
    r, _ = rollout(T, None, 0, 10 ** 6)
    assert r["success"], r["reason"]
    (tmp_path / "domain.pddl").write_text(dom)
    (tmp_path / "problem.pddl").write_text(prob)
    v = validate_plan(str(tmp_path / "domain.pddl"), str(tmp_path / "problem.pddl"), TemporalPlan.parse(plan_text(r["steps"])))
    assert v.valid, getattr(v, "log", v)


def test_options_and_features():
    _, _, _, T = task("hard_base", "M")
    env = MAEnv(T, "robot", seed=0)
    D, seen = Dispatcher(env), set()

    def ctl(e, S, a, lg):
        V = View(e, S)
        opts = D.options(V, a, lg)
        X = features(D, V, a, opts)
        assert X.shape == (len(opts), NF)
        seen.update(o["rule"] for o in opts)
        return D.decide(V, a, lg, opts)
    r = env.rollout(ctl, seed=0)
    assert r["success"]
    assert {"go_pickup", "go_deliver", "wait", "load-out", "unload-sink"} <= seen and seen <= set(RULES)
