# ADAPTER (replaces flax/planning/__init__.py): same names; the STRIPS validator is replaced by
# the temporal one (baselines/adapter.py). Fast Downward wrappers are not used.
from planning.planner import Planner, PlanningTimeout, PlanningFailure


def validate_strips_plan(domain_file, problem_file, plan):
    """The plan was already checked by temporal VAL on the full task when baselines.adapter.TemporalPOPF
    produced it; this returns that verdict."""
    return bool(getattr(plan, "valid", False))
