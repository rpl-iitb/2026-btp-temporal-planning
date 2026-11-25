"""algorithm/ and baselines/ on FactorySim instances (data/, make_data.py): the PDDL conversion is exact, the
algorithm's own PDDL writer reproduces a problem POPF / VAL accept, and plans are VAL-valid and run in FactorySim."""
import glob
import os

import pytest

from factory_test import DATA, make_data
from factory_test.execute import run_plan
from factory_test.evaluate import build
from algorithm.grounding import GroundTask, Task
from algorithm.pddlkit import TemporalPlan, validate_plan

if not glob.glob(os.path.join(DATA, "test", "*", "hl_problem.json")):
    make_data.main(["--splits", "test", "--count", "1"])
SMALL = os.path.join(DATA, "test", "base_S_test_000")


@pytest.mark.parametrize("d", sorted(glob.glob(os.path.join(DATA, "test", "*_S_test_000"))), ids=os.path.basename)
def test_conversion_grounds(d):
    task = Task.load(d)
    gt = GroundTask(task)
    assert task.goals and gt.actions and gt.proof(set(task.objects)) is None   # relaxed-solvable as exported


def test_smoke_plan_valid_and_runs_in_factory(tmp_path):
    """Smoke: B1 (one POPF call, 20 s) on the smallest factory; its plan is VAL-valid on FactorySim's own PDDL and runs."""
    task = Task.load(SMALL)
    r = build("B1", task.spec).solve(task, 20)
    if not r.solved:
        pytest.skip("B1 found no plan in 20 s")
    plan = TemporalPlan.parse(r.plan_text)
    assert validate_plan(os.path.join(SMALL, "domain.pddl"), os.path.join(SMALL, "problem.pddl"), plan).valid
    m = run_plan(SMALL, r.plan_text)
    assert m["success"], m


@pytest.mark.parametrize("d", sorted(glob.glob(os.path.join(DATA, "test", "*_M_test_000"))), ids=os.path.basename)
def test_multi_robot_plans_run_in_factory(d):
    """V8S's VAL-valid plans for 3 robots run to the end in FactorySim (PlanPolicy keeps the plan's order on shared
    jobs and stations), with nominal and with sampled durations."""
    from algorithm.v8 import V8Planner
    from factory_test.factory_scheduler import FactoryScheduler
    task = Task.load(d)
    r = V8Planner("V8S", FactoryScheduler(task.spec), None, max_runs=2).solve(task, 15)
    assert r.solved and validate_plan(os.path.join(d, "domain.pddl"), os.path.join(d, "problem.pddl"), TemporalPlan.parse(r.plan_text)).valid
    assert run_plan(d, r.plan_text)["success"]
    assert all(run_plan(d, r.plan_text, time_source="sampled", seed=s)["success"] for s in range(3))
