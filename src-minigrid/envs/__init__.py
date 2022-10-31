from functools import partial
from .multiagentenv import MultiAgentEnv
from .doorkey import DoorKeyEnv5x5
from .multiroom import MultiRoomEnvN2S4
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["dk"] = partial(env_fn, env=DoorKeyEnv5x5)
REGISTRY["mr"] = partial(env_fn, env=MultiRoomEnvN2S4)
