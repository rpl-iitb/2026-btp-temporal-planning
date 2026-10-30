# ADAPTER (replaces the pddlgym package): environments are task sets registered by
# baselines/adapter.py; only structs.py is pddlgym's own code.


def make(env_id):
    from baselines.adapter import make_env
    return make_env(env_id)
