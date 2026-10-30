"""Benchmark 1: Overcooked-AI 1.1.0 behind the generic Benchmark interface."""
import os

from pddl_generator import Benchmark, load_domain_spec
from . import builder
from .executor import PlanExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAIN_FILE = os.path.join(HERE, "domain.yaml")
SCENARIO_DIR = os.path.join(HERE, "scenarios")
PDDL_DIR = os.path.join(HERE, "generated_pddl")


class OvercookedV1(Benchmark):
    name = "overcooked_ai"
    scenario_dir, pddl_dir = SCENARIO_DIR, PDDL_DIR

    def __init__(self):
        self._spec = None

    def domain_spec(self):
        if self._spec is None:
            self._spec = load_domain_spec(DOMAIN_FILE)
        return self._spec

    def load_scenario(self, path):
        return builder.load_scenario(path)

    def build_problem(self, scenario):
        return builder.build_problem(scenario).validate(self.domain_spec())

    def execute(self, problem, plan, mode="timed", **kwargs):
        return PlanExecutor(problem, plan, self.domain_spec(), mode=mode, **kwargs).run()
