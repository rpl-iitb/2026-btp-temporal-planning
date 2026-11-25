"""A tiny custom policy: serve at once when possible, otherwise let the rule dispatcher decide.

    overcooked-kitchen bench --suite benchmarks/v1 --split v2_val --policy examples/my_policy.py:policy
"""
"""This file is UN-VERIFIED"""
from overcooked_kitchen.baselines import make
from overcooked_kitchen.tasks import WAIT

dispatcher = make("dispatcher")


def policy(sim, cook, legal):
    """legal: the cook's legal tasks (WAIT is always one of them); return one of them."""
    for t in legal:
        if t != WAIT and t.event in ("deliver", "dish_up"):
            return t
    return dispatcher(sim, cook, legal)
