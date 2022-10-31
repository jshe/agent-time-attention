from functools import partial
from .multiagentenv import MultiAgentEnv
from .coin1d_env import Coin1DEnv

import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["c1d"] = partial(env_fn, env=Coin1DEnv)

