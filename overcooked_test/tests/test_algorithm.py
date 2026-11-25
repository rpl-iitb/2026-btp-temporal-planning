"""Tests on overcooked_kitchen instances (data/, make_data.py): semantics vs VAL, Pi|S soundness, re-timing, every version end to end."""
import glob
import os
import random

import pytest

from algorithm import PDDL
from overcooked_test.evaluate import METHODS, build
from algorithm.gnn import Scorer
from algorithm.grounding import GroundTask, Task, comp_t
from algorithm.planning import Budget, Workspace, attempt
from algorithm.semantics import UNBOUNDED, Simulator, bind_plan, retime
from algorithm.v1 import MethodConfig, ObjectSelectionPlanner
from algorithm.pddlkit import TemporalPlan, TimedAction, validate_plan


# ==================== Semantics agree with VAL ===================================================
CERTS = sorted(glob.glob(os.path.join(PDDL, "v2_train", "v2_train_s_*", "certificate.plan")))[:5] + \
    sorted(glob.glob(os.path.join(PDDL, "v2_test", "v2_test_m_*", "certificate.plan")))[:3] + \
    sorted(glob.glob(os.path.join(PDDL, "v3_test", "v3_test_s3_*", "certificate.plan")))[:2]


def _sim(task, gt, plan):
    steps, durs, unknown = bind_plan(gt, plan)
    assert not unknown, unknown
    return Simulator(task).run(steps, durs)


@pytest.mark.parametrize("cert", CERTS, ids=lambda p: p.split(os.sep)[-2])
def test_certificate_valid(cert):
    d = os.path.dirname(cert)
    task = Task.load(d)
    gt = GroundTask(task)
    with open(cert) as f:
        plan = TemporalPlan.parse(f.read())
    r = _sim(task, gt, plan)
    assert r.valid, [v.detail for v in r.violations][:5]
    assert abs(r.makespan - plan.makespan) < 1e-6
    for g in task.goals:
        assert g in r.goal_achieved_at
        if g in task.deadlines:
            assert r.goal_achieved_at[g] <= task.deadlines[g] + 1e-6


@pytest.mark.parametrize("cert", CERTS[:6], ids=lambda p: p.split(os.sep)[-2])
def test_mutants_agree_with_val(cert):
    d = os.path.dirname(cert)
    task = Task.load(d)
    gt = GroundTask(task)
    dom, prob = task.domain_file(), os.path.join(d, "problem.pddl")
    with open(cert) as f:
        plan = TemporalPlan.parse(f.read())
    rng = random.Random(0)
    agree = 0
    for k in range(6):
        acts = list(plan.actions)
        i = rng.randrange(len(acts))
        if k % 2 == 0:
            a = acts[i]
            acts[i] = TimedAction(max(0.0, a.start - rng.choice([1.0, 3.0, 10.0])), a.name, a.args, a.duration)
        else:
            del acts[i]
        mutant = TemporalPlan(sorted(acts, key=lambda a: (a.start, a.name, a.args)))
        v = validate_plan(dom, prob, mutant)
        r = _sim(task, gt, mutant)
        assert r.valid == v.valid, (k, v.errors[:3], [x.detail for x in r.violations][:3])
        agree += 1
    assert agree == 6


def test_unbounded_is_not_a_time():
    assert UNBOUNDED > 1e18 and not (UNBOUNDED < 5)
    with pytest.raises(TypeError):
        UNBOUNDED + 1.0


def test_intervals_open_ended():
    d = os.path.dirname(CERTS[0])
    task = Task.load(d)
    gt = GroundTask(task)
    with open(CERTS[0]) as f:
        plan = TemporalPlan.parse(f.read())
    r = _sim(task, gt, plan)
    open_ended = [iv for iv in r.intervals if iv.open_ended]
    closed = [iv for iv in r.intervals if not iv.open_ended]
    assert open_ended and closed
    assert all(iv.deleted >= iv.added for iv in closed)
    for g in task.goals:
        assert any(iv.fact == g and iv.open_ended for iv in open_ended)


# ==================== Pi|S is never misleading ===================================================
INST = sorted(glob.glob(os.path.join(PDDL, "v2_train", "v2_train_s_*")))[:4] + \
    sorted(glob.glob(os.path.join(PDDL, "v2_test", "v2_test_m_*")))[:3] + \
    sorted(glob.glob(os.path.join(PDDL, "v3_test", "v3_test_s3_*")))[:2]
INST = [d for d in INST if os.path.isdir(d)]


@pytest.mark.parametrize("d", INST, ids=os.path.basename)
def test_proof_never_fires_on_certificate_objects(d):
    task = Task.load(d)
    gt = GroundTask(task)
    with open(os.path.join(d, "certificate.plan")) as f:
        cert = TemporalPlan.parse(f.read())
    S = cert.objects_used() & set(task.objects)
    assert gt.proof(S) is None
    assert gt.proof(set(task.objects)) is None
    assert gt.trpg(S).lb <= cert.makespan + 1e-6       # LB is admissible on a known plan


@pytest.mark.parametrize("d", INST, ids=os.path.basename)
def test_comp_t_relaxed_solvable(d):
    task = Task.load(d)
    gt = GroundTask(task)
    rng = random.Random(1)
    objs = sorted(task.objects)
    for _ in range(5):
        S = comp_t(gt, rng.sample(objs, rng.randint(0, len(objs) // 3)))
        assert gt.proof(S) is None


@pytest.mark.parametrize("d", INST[:5], ids=os.path.basename)
def test_reduced_plans_valid_on_full(d):
    task = Task.load(d)
    gt = GroundTask(task)
    rng = random.Random(2)
    objs = sorted(task.objects)
    ws = Workspace()
    try:
        outcomes = []
        for k in range(3):
            S = comp_t(gt, rng.sample(objs, rng.randint(0, len(objs) // 2)))
            a = attempt(task, gt, S, Budget(20), "test", ws, cap=10)
            outcomes.append(a.outcome)
            assert a.outcome != "invalid_on_full", a.detail
            if a.outcome == "valid":
                assert a.plan.objects_used() <= S | set(task.constants)
        assert set(outcomes) <= {"valid", "planner_gave_up", "timeout"}
    finally:
        ws.close()


# ==================== Re-timing ==================================================================
CERTS = sorted(glob.glob(os.path.join(PDDL, "v2_test", "*", "certificate.plan")))[::6]


@pytest.mark.parametrize("cert", CERTS, ids=lambda p: p.split(os.sep)[-2])
def test_retime_valid(cert, tmp_path):
    d = os.path.dirname(cert)
    task = Task.load(d)
    with open(cert) as f:
        plan = TemporalPlan.parse(f.read())
    rt = retime(task, GroundTask(task), plan)
    assert rt is not None
    assert rt.makespan <= plan.makespan + 1e-6
    dom, prob = task.write(str(tmp_path), "full")
    assert validate_plan(dom, prob, rt).valid


# ==================== End to end =================================================================
GRACE = 2.0


def make(task):
    return ObjectSelectionPlanner(MethodConfig(), Scorer(task.spec, heads=2))


def test_small_instance_valid(tmp_path):
    task = Task.load(os.path.join(PDDL, "v2_train", "v2_train_s_001"))
    r = make(task).solve(task, 20)
    assert r.solved
    assert r.total_s <= 20 + GRACE
    assert r.invalid_on_full == 0
    assert r.makespan <= r.first_makespan + 1e-6           # improvement phase never makes it worse
    dom, prob = task.write(str(tmp_path), "full")
    assert validate_plan(dom, prob, TemporalPlan.parse(r.plan_text)).valid


def test_tiny_budget_is_respected():
    task = Task.load(os.path.join(PDDL, "v2_test", "v2_test_xl_002"))
    r = make(task).solve(task, 4)
    assert r.total_s <= 4 + GRACE
    if not r.solved:
        outcomes = {a["outcome"] for a in r.attempts}
        assert outcomes <= {"timeout", "budget_exhausted", "cancelled", "planner_gave_up", "proved_infeasible"}
        assert not any(a["outcome"] == "proved_infeasible" and a["n_objects"] == r.n_total for a in r.attempts)


@pytest.mark.parametrize("method", [m for m in METHODS if m.startswith("V")])
def test_every_version_solves_a_small_instance(method, tmp_path):
    """Each version (trained models from results/) on a small kitchen: VAL-valid plan within the budget."""
    task = Task.load(os.path.join(PDDL, "v2_train", "v2_train_s_001"))
    try:
        planner = build(method, task.spec)
    except FileNotFoundError as e:
        pytest.skip("model not trained: {}".format(e))
    r = planner.solve(task, 20)
    assert r.solved and r.total_s <= 20 + GRACE and r.invalid_on_full == 0
    dom, prob = task.write(str(tmp_path), "full")
    assert validate_plan(dom, prob, TemporalPlan.parse(r.plan_text)).valid
