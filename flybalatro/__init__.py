"""Balatro environment and binary feature encoding for the fly-connectome readout.

The game half of the project: a thin wrapper over the ``pylatro`` Rust engine
(:mod:`flybalatro.env`) and a fixed-length binary state encoding
(:mod:`flybalatro.features`) intended to drive olfactory receptor neurons in
the MaleCNS connectome simulation. The brain half owns :mod:`flybalatro.brain`,
:mod:`flybalatro.connectome` and :mod:`flybalatro.encode`.
"""

from __future__ import annotations

from .env import (
    ACTION_NAMES,
    N_ACTIONS,
    NOOP_ACTION_INDICES,
    BalatroEnv,
    IllegalActionError,
    RewardConfig,
)
from .features import FEATURE_NAMES, N_FEATURES

# `encode` is deliberately NOT re-exported here. The brain half of the project
# owns the submodule `flybalatro.encode` (bits -> ORN currents), and importing
# that submodule rebinds the `flybalatro.encode` attribute, which would shadow
# a re-exported function. Import the state encoder explicitly:
#     from flybalatro.features import encode

__all__ = [
    "BalatroEnv",
    "IllegalActionError",
    "RewardConfig",
    "ACTION_NAMES",
    "N_ACTIONS",
    "NOOP_ACTION_INDICES",
    "FEATURE_NAMES",
    "N_FEATURES",
]

__version__ = "0.1.0"
