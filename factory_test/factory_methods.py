"""Builders for the versions that need factory parts or factory models (evaluate.py's METHODS): V8 / V8S with the
robot dispatcher as scheduler (factory_scheduler.py; V8 also with a GNN scorer trained by teacher.py), V9 with the
dispatcher's decisions as RL environment (factory_v9.py), V11 with its model trained on the factory instances."""
import os

from factory_test import RESULTS

MODELS = os.path.join(RESULTS, "models")
TAG = "_factorysim"


def build(method, spec, seed=0, **over):
    from algorithm.gnn import Scorer
    from .factory_scheduler import FactoryScheduler
    if method in ("V8", "V8S"):
        from algorithm.v8 import V8Planner
        scorer = None if method == "V8S" else Scorer.load(os.path.join(MODELS, "TEACHER{}_seed{}".format(TAG, seed)), spec)
        return V8Planner("TiFLAX_" + method, FactoryScheduler(spec), scorer, **over)
    if method == "V9":
        from algorithm.v9 import V9Planner, load_policy
        from .factory_v9 import FactoryDomain
        return V9Planner(FactoryDomain(), Scorer.load(os.path.join(MODELS, "TEACHER{}_seed{}".format(TAG, seed)), spec),
                         load_policy(os.path.join(MODELS, "TIFLAX9{}_seed{}".format(TAG, seed))), **over)
    if method == "V11":
        from algorithm import v11
        return v11.build(method, spec, seed, model=os.path.join(MODELS, "V11_factorysim"), **over)
    raise KeyError(method)
