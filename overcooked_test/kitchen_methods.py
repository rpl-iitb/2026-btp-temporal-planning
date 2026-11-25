"""Builders for the versions that need kitchen parts or kitchen models (evaluate.py's METHODS): V8 / V8S / V8.2 with the
kitchen scheduler (and the teacher-trained GNN: teacher.py), V9 with the kitchen macro-step environment and its RL policy
(kitchen_v9.py), V11 with its model trained on the kitchens (python -m algorithm.v11 train --bench overcooked_kitchen)."""
import os

from overcooked_test import RESULTS

MODELS = os.path.join(RESULTS, "models")
TAG = "_overcooked_kitchen"   # models trained on this harness's data


def build(method, spec, seed=0, **over):
    from algorithm.gnn import Scorer
    from .kitchen_scheduler import KitchenScheduler
    if method in ("V8", "V8S", "V8.2"):
        from algorithm.v8 import V8Planner
        if method == "V8S":
            return V8Planner("TiFLAX_V8_symbolic", KitchenScheduler(), None, **over)
        net, name = ("TEACHER2", "TiFLAX_V8.2") if method == "V8.2" else ("TEACHER", "TiFLAX_V8")
        return V8Planner(name, KitchenScheduler(), Scorer.load(os.path.join(MODELS, "{}{}_seed{}".format(net, TAG, seed)), spec), **over)
    if method == "V9":
        from algorithm.v9 import V9Planner, load_policy
        from .kitchen_v9 import KitchenDomain
        return V9Planner(KitchenDomain(), Scorer.load(os.path.join(MODELS, "TEACHER{}_seed{}".format(TAG, seed)), spec),
                         load_policy(os.path.join(MODELS, "TIFLAX9{}_seed{}".format(TAG, seed))), **over)
    if method == "V11":
        from algorithm import v11
        return v11.build(method, spec, seed, model=os.path.join(MODELS, "V11_overcooked_kitchen"), **over)
    raise KeyError(method)
