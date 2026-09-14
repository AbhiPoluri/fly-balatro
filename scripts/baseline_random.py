"""Uniform-random-legal-action baseline.

Run:
    source .venv/bin/activate
    python scripts/baseline_random.py [n_episodes]

Writes outputs/baseline_random.json.
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

# scripts/ is a package (it has __init__.py), so make the sibling helper
# importable whether this file is run as a script or as -m scripts.<name>.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
from numpy.typing import NDArray

from _baseline_common import report, run_episodes  # noqa: E402
from flybalatro.env import BalatroEnv


class RandomPolicy:
    """Samples uniformly from the legal indices in the mask."""

    def __init__(self, seed: int = 0) -> None:
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def __call__(self, env: BalatroEnv, mask: NDArray[np.int8]) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise RuntimeError("no legal actions but episode is not done")
        return int(legal[self._rng.integers(legal.size)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_episodes", nargs="?", type=int, default=2000)
    parser.add_argument("--ante-end", type=int, default=1)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    metrics = run_episodes(
        RandomPolicy(seed=args.policy_seed),
        n_episodes=args.n_episodes,
        ante_end=args.ante_end,
        seed0=args.seed0,
        max_steps=args.max_steps,
    )
    report("baseline_random", metrics)


if __name__ == "__main__":
    main()
