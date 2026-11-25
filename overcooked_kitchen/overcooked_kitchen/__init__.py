"""Multi-cook temporal kitchen benchmark."""
"""This file is UN-VERIFIED"""

__version__ = "0.1.0"

try:
    import gymnasium as _gym

    for _id, _entry in (("OvercookedKitchen-v0", "KitchenEnv"), ("OvercookedKitchenGrid-v0", "KitchenGridEnv")):
        if _id not in _gym.registry:
            _gym.register(id=_id, entry_point="overcooked_kitchen.env:" + _entry)
except ImportError:
    pass
