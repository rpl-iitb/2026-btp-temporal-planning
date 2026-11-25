"""V9's factory side: the robot dispatcher's decisions as an RL environment (algorithm/v9.py's protocol). At every
robot decision of a nominal menv rollout the policy picks one of the dispatcher's options (factory_dispatcher.py:
load / unload here, go pick up, go deliver, go to a buffer, wait) from their features; the dispatcher's own choice is
the rule (behaviour cloning). Every finished rollout is a plan; V9 keeps the shortest VAL-valid one.

    python -m factory_test.factory_v9 train [--epochs 20 ...]     # -> results/models/TIFLAX9_factorysim_seed0.{pt,json}
"""
import argparse
import os

import factory_test  # noqa: F401  (paths)
from factory_test import RESULTS
from factory_test.factory_dispatcher import NF, Dispatcher, View, expert_index, features
from factory_test.factory_v13 import load_split
from algorithm import v9
from algorithm.menv import MAEnv, plan_text
from algorithm.pddlkit import TemporalPlan

MODELS = os.path.join(RESULTS, "models")


class Env:
    n_features = NF

    def __init__(self, T, score=None):
        self.T, self.score, self._expert = T, score or {}, None

    def objects(self):
        return set(self.T.prob.objects)

    def rule_choice(self, cands):
        return self._expert

    def rollout(self, choose, record=False):
        env = MAEnv(self.T, "robot", noise=None, seed=0)
        D = Dispatcher(env, "nearest", self.score)
        traj = []

        def ctl(env_, S, r, legal):
            V = View(env_, S)
            opts = D.options(V, r, legal)
            self._expert = expert_index(D, V, r, legal, opts)
            F = features(D, V, r, opts)
            j = choose(S, opts, F)
            if record:
                traj.append((F, j))
            return opts[j]["k"]
        res = env.rollout(ctl)
        return (TemporalPlan.parse(plan_text(res["steps"])) if res["success"] else None), traj


class FactoryDomain:
    """The planning side of V9's protocol: one environment per task (the dispatcher fixes no object choice)."""

    def envs(self, task, score):
        from factory_test.factory_scheduler import FactoryScheduler
        return [Env(FactoryScheduler(task.spec)._task(task.problem), score)]

    def search_envs(self, task, score, deadline_fn):
        return []


def episodes(split):
    return [(name, Env(T), ref, None) for name, T, ref, meta in load_split(split)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    for k, v in (("epochs", 20), ("bc_epochs", 3), ("rollouts", 8), ("hidden", 64), ("seed", 0)):
        t.add_argument("--" + k, type=int, default=v)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--entropy", type=float, default=0.01)
    a = ap.parse_args(argv)
    v9.train(episodes("train"), episodes("val"), os.path.join(MODELS, "TIFLAX9_factorysim_seed{}".format(a.seed)), epochs=a.epochs,
             bc_epochs=a.bc_epochs, rollouts=a.rollouts, lr=a.lr, entropy=a.entropy, hidden=a.hidden, seed=a.seed, meta={"bench": "factorysim"})


if __name__ == "__main__":
    main()
