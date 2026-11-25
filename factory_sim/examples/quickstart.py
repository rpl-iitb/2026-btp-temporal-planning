"""Three ways to use FactorySim: the core with a policy, the Gymnasium env, and the PettingZoo env."""
import numpy as np

import factory_sim
from factory_sim import Factory, Sim, baselines


def my_policy(sim, agent, legal):
    """A policy sees the simulator and the legal actions of the agent that must decide now."""
    picks = [a for a in legal if a[0] in ("release", "deliver", "pickup")]
    return picks[0] if picks else ("wait",)


sim = Sim(Factory("lab"), seed=0)
print("core, custom policy:", sim.run(my_policy)["makespan"])
print("core, EDD rule:", Sim(Factory("lab"), seed=0).run(baselines.make("edd"))["makespan"])

import gymnasium as gym  # noqa: E402

env = gym.make("FactorySim-v0", instance="lab")
obs, info = env.reset(seed=0)
rng, done = np.random.default_rng(0), False
while not done:
    obs, reward, term, trunc, info = env.step(rng.choice(np.flatnonzero(info["action_mask"])))
    done = term or trunc
print("gymnasium, random masked actions:", info["metrics"]["makespan"])

aec = factory_sim.aec_env(instance="lab")
aec.reset(seed=0)
for agent in aec.agent_iter():
    obs, reward, term, trunc, info = aec.last()
    aec.step(None if term or trunc else int(np.flatnonzero(info["action_mask"])[0]))
print("pettingzoo agents:", aec.possible_agents)
