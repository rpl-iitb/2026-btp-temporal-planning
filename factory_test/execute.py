"""Run a plan from algorithm/ or baselines/ in FactorySim (the plan's releases, pickups and deliveries in order;
factory_sim.pddl.PlanPolicy), nominal or sampled durations. -> FactorySim metrics (success, makespan, ...)."""
import json
import os

from factory_sim import pddl as FP
from factory_sim.env import build_sim


def run_plan(instance_dir, plan_text, time_source="nominal", seed=0):
    inst = json.load(open(os.path.join(instance_dir, "instance.json")))
    sim0 = build_sim(inst, seed=seed, time_source="nominal")
    policy = FP.PlanPolicy(sim0, plan_text)
    return build_sim(inst, seed=seed, time_source=time_source).run(policy)
