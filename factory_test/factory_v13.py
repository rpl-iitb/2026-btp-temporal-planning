"""V12 / V13's factory side (the domain module of algorithm/v13.py): the robot dispatcher (factory_dispatcher.py) as
option generator and DAgger expert, its option features, duration noise and the factory data (data/, make_data.py).

    python -m algorithm.v13 --domain factory_test.factory_v13 train
    python -m algorithm.v13 --domain factory_test.factory_v13 eval --split test --seeds 5
"""
import json
import os

import factory_test  # noqa: F401  (paths)
from factory_test import DATA
from factory_test.factory_dispatcher import NF, Dispatcher, View, expert_index, features  # noqa: F401  (domain protocol)
from algorithm import v10
from algorithm.menv import MAEnv, Noise

MAEnv.max_rounds = 10 ** 6      # the large lane-traffic plants need more decision rounds than the kitchens

AGENT = "robot"
SPLITS = ("train", "val")
NOISE = Noise({"move": 0.2, "move-j": 0.2, "load-machine": 0.15, "load-out": 0.15, "load-buffer": 0.15, "unload": 0.15,
               "unload-buffer": 0.15, "unload-sink": 0.15, "process": 0.1, "convey": 0.05}, default=0.1)


def task(split, name):
    d = os.path.join(DATA, split, name)
    return v10.Task(v10.Domain(open(os.path.join(d, "domain.pddl")).read()), v10.Problem(open(os.path.join(d, "problem.pddl")).read()))


class LazyTask:
    """A v10 Task grounded on first use (the large plants take minutes to ground; listing a split needs none)."""

    def __init__(self, split, name):
        self._key, self._T = (split, name), None

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        if self._T is None:
            self._T = task(*self._key)
        return getattr(self._T, attr)


def load_split(split):
    """-> [(name, v10 Task (grounded when first used), reference makespan, meta)]; the reference is FactorySim's best
    known makespan."""
    meta = json.load(open(os.path.join(DATA, split, "dataset.json")))
    out = []
    for inst in meta["instances"]:
        ref = inst.get("best_known") or 2.0 * (inst.get("lower_bound") or 1000.0)
        out.append((inst["instance_id"], LazyTask(split, inst["instance_id"]), float(ref), inst))
    return out
