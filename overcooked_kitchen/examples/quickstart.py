"""Python API tour: the simulator with a baseline, the Gymnasium env, the PDDL export. Runs headless in seconds."""
"""This file is UN-VERIFIED"""
import os
import tempfile

import gymnasium as gym
import numpy as np

import overcooked_kitchen  # noqa: F401  (registers the Gymnasium ids)
from overcooked_kitchen import pddl
from overcooked_kitchen.baselines import make
from overcooked_kitchen.core import Sim
from overcooked_kitchen.tasks import WAIT

# 1. simulator: the rule dispatcher on three cooks, sampled clock with duration noise
sim = Sim("m01_three_cooks", clock="sampled", seed=0, noise="default")
metrics = sim.run(make("dispatcher"))
print("dispatcher on m01: success={success} makespan={makespan} burnt={burnt} on_time={on_time:.0%}".format(**metrics))


# 2. your own policy: policy(sim, cook, legal) -> one of legal (see examples/my_policy.py)
def serve_first(sim, cook, legal):
    for task in legal:
        if task != WAIT and task.event in ("deliver", "dish_up"):
            return task
    return dispatcher(sim, cook, legal)


dispatcher = make("dispatcher")
print("custom policy on k01: makespan", Sim("k01_one_soup", noise=None).run(serve_first)["makespan"])

# 3. Gymnasium: masked random actions, one decision per step (info["agent"] says which cook decides)
env = gym.make("OvercookedKitchen-v0", instance="k01_one_soup")
obs, info = env.reset(seed=0)
rng = np.random.default_rng(0)
steps = 0
while True:
    legal = np.flatnonzero(info["action_mask"])
    nonwait = legal[legal > 0]
    action = int(rng.choice(nonwait if len(nonwait) else legal))
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
    if terminated or truncated:
        break
print("gym episode: {} decisions, served all orders: {}".format(steps, terminated))

# 4. PDDL: the current state as a temporal planning problem
with tempfile.TemporaryDirectory() as tmp:
    domain, problem = pddl.write(Sim("k01_one_soup"), tmp)
    print("pddl:", os.path.basename(domain), os.path.basename(problem), os.path.getsize(problem), "bytes")
