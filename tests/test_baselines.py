"""B1-B3 through the adapter: FLAX rules give a sensible closure, plans are VAL-valid, budgets hold."""
import os

import pytest

from algorithm import PDDL
from algorithm.grounding import Task
from baselines import adapter
from baselines.adapter import RULES
from baselines.b1 import FullPlanner
from baselines.b2 import FlaxBaseline
from baselines.b3 import IFlaxBaseline
from pddl_generator import TemporalPlan, validate_plan


class Uniform:
    def score_object(self, obj, state):
        return 0.5


def test_complementary_closure():
    import json
    adapter.install_rule_wrappers()
    import planning.my_planner as mp
    task = Task.load(os.path.join(PDDL, "examples", "s04_asym_c2_o2_dl"))
    state = adapter.task_to_state(task)
    with open(RULES["complementary"]) as f:
        rules = json.load(f)
    goal_objs = {o for l in state.goal.literals for o in l.variables}
    objs, _ = mp.apply_complementary_rules(state, set(goal_objs), rules)
    names = {adapter._name(o) for o in objs}
    assert set(task.of_type("count")) <= names            # zero count + successors
    assert not [o for o in names if task.is_a(o, "pot")]  # goals alone pull in no station


def test_relaxed_task_drops_time():
    import json
    adapter.install_rule_wrappers()
    import planning.my_planner as mp
    task = Task.load(os.path.join(PDDL, "examples", "s02_cramped_c2_o2_dl"))
    state = adapter.task_to_state(task)
    with open(RULES["relaxation"]) as f:
        rules = json.load(f)
    _, rs = mp.apply_relaxation_rules(state, rules, adapter.domain_for(task.spec))
    rt = adapter.state_to_task(rs)
    assert not rt.deadlines and not rt.tils
    assert all(("free", t) in rt.init for t in task.of_type("tile") if any(f[0] == "pose-tile" and f[2] == t for f in task.init))


@pytest.mark.parametrize("cls", ["B1", "B2", "B3"])
def test_solve_valid(cls, tmp_path):
    task = Task.load(os.path.join(PDDL, "examples", "s04_asym_c2_o2_dl"))
    p = {"B1": FullPlanner(), "B2": FlaxBaseline(Uniform()), "B3": IFlaxBaseline(Uniform())}[cls]
    r = p.solve(task, 20)
    assert r.solved and r.total_s <= 22 and r.invalid_on_full == 0
    dom, prob = task.write(str(tmp_path), "full")
    assert validate_plan(dom, prob, TemporalPlan.parse(r.plan_text)).valid
