"""RL reference: masked PPO (sb3-contrib) on a FactorySim family; evaluates on the suite's test split.

    pip install -e ".[rl]"
    python examples/train_ppo.py --family base --steps 200000
"""
import argparse

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from factory_sim import scenarios
from factory_sim.env import FactoryEnv


class Rotating(FactoryEnv):
    """A new training instance of the family at every reset."""

    def __init__(self, family, size):
        self.family, self.size, self.k = family, size, 0
        super().__init__(instance=scenarios.make_instance(family, size, "train", 0))

    def reset(self, seed=None, options=None):
        self.instance = scenarios.make_instance(self.family, self.size, "train", self.k % 50)
        self.k += 1
        return super().reset(seed=seed, options=options)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="base")
    ap.add_argument("--size", default="S")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--suite", default="benchmarks/v1.0")
    a = ap.parse_args()
    env = ActionMasker(Rotating(a.family, a.size), lambda e: e.unwrapped.action_masks())
    model = MaskablePPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=256, gamma=0.999)
    model.learn(a.steps)
    model.save("ppo_{}_{}".format(a.family, a.size))
    res = []
    for inst in scenarios.iter_suite(a.suite, "test", [a.family]):
        e = FactoryEnv(instance=inst)
        obs, info = e.reset(seed=0)
        done = False
        while not done:
            act, _ = model.predict(obs, action_masks=info["action_mask"].astype(bool), deterministic=True)
            obs, r, term, trunc, info = e.step(int(act))
            done = term or trunc
        m = info["metrics"]
        res.append(m["makespan"] / inst["ground_truth"]["lower_bound"] if m["makespan"] else np.nan)
    print("test {}: success {:.0%}, makespan / LB {:.3f}".format(a.family, np.mean(np.isfinite(res)), np.nanmean(res)))


if __name__ == "__main__":
    main()
