"""FactorySim: a smart-factory benchmark for multi-robot temporal planning, scheduling and reinforcement learning."""
from .config import Factory, builtin
from .core import DISPATCHER, Sim
from . import baselines, scenarios

__version__ = "2.0.0"
__all__ = ["Factory", "Sim", "baselines", "scenarios", "DISPATCHER", "builtin"]

try:
    from .env import FactoryEnv, aec_env, build_sim
    from gymnasium.envs.registration import register
    register(id="FactorySim-v0", entry_point="factory_sim.env:FactoryEnv")
    __all__ += ["FactoryEnv", "aec_env", "build_sim"]
except ImportError:
    pass
