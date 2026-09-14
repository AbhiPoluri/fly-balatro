"""Shared rollout loop, metrics and JSON reporting for the baseline scripts."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flybalatro.env import BalatroEnv  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs"

# A policy sees the env (for state) and the legal mask, and returns an index.
Policy = Callable[[BalatroEnv, NDArray[np.int8]], int]

JsonValue = Union[int, float, str, bool, None, List[float], Dict[str, float]]


def run_episodes(
    policy: Policy,
    n_episodes: int,
    ante_end: int = 1,
    seed0: int = 0,
    max_steps: int = 500,
) -> Dict[str, JsonValue]:
    """Run ``n_episodes`` of ``policy`` and return an aggregate metrics dict.

    Episode ``i`` uses engine seed ``seed0 + i``, so both baselines see the
    same deck shuffles and are directly comparable.
    """
    env = BalatroEnv(ante_end=ante_end, max_steps=max_steps)

    chips: List[int] = []
    best_scores: List[int] = []
    lengths: List[int] = []
    rewards: List[float] = []
    wins = 0
    truncations = 0
    rounds_reached: List[int] = []

    total_steps = 0
    elapsed = 0.0

    for episode in range(n_episodes):
        t0 = time.perf_counter()
        _features, mask = env.reset(seed=seed0 + episode)
        best_score = 0
        total_reward = 0.0
        truncated = False
        last_round = 0
        while not env.done:
            action = policy(env, mask)
            _features, mask, reward, _done, info = env.step(action)
            total_reward += reward
            score = int(info["score"])
            if score > best_score:
                best_score = score
            last_round = int(info["round"])
            truncated = bool(info["truncated"])
        elapsed += time.perf_counter() - t0

        total_steps += env.steps
        chips.append(env.chips_total)
        best_scores.append(best_score)
        lengths.append(env.steps)
        rewards.append(total_reward)
        rounds_reached.append(last_round)
        if env.is_win:
            wins += 1
        if truncated:
            truncations += 1

    return {
        "n_episodes": n_episodes,
        "ante_end": ante_end,
        "seed0": seed0,
        "max_steps": max_steps,
        "chips_mean": float(statistics.fmean(chips)),
        "chips_median": float(statistics.median(chips)),
        "chips_stdev": float(statistics.pstdev(chips)),
        "chips_max": int(max(chips)),
        "best_blind_score_mean": float(statistics.fmean(best_scores)),
        "best_blind_score_median": float(statistics.median(best_scores)),
        "best_blind_score_max": int(max(best_scores)),
        "win_rate": wins / n_episodes,
        "wins": wins,
        "episode_len_mean": float(statistics.fmean(lengths)),
        "episode_len_median": float(statistics.median(lengths)),
        "episode_len_max": int(max(lengths)),
        "rounds_reached_mean": float(statistics.fmean(rounds_reached)),
        "reward_mean": float(statistics.fmean(rewards)),
        "truncated_episodes": truncations,
        "total_steps": total_steps,
        "wall_seconds": elapsed,
        "steps_per_second": total_steps / elapsed if elapsed > 0 else 0.0,
        "episodes_per_second": n_episodes / elapsed if elapsed > 0 else 0.0,
    }


def report(name: str, metrics: Dict[str, JsonValue], out_path: Optional[Path] = None) -> Path:
    """Print metrics and write them to ``outputs/<name>.json``."""
    path = out_path if out_path is not None else OUTPUT_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, JsonValue] = {"policy": name}
    payload.update(metrics)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"== {name} ==")
    key_order: Sequence[str] = (
        "n_episodes",
        "ante_end",
        "chips_mean",
        "chips_median",
        "chips_max",
        "best_blind_score_mean",
        "best_blind_score_median",
        "win_rate",
        "episode_len_mean",
        "episode_len_median",
        "rounds_reached_mean",
        "reward_mean",
        "truncated_episodes",
        "total_steps",
        "wall_seconds",
        "steps_per_second",
    )
    for key in key_order:
        value = payload[key]
        if isinstance(value, float):
            print(f"  {key:24s} {value:,.3f}")
        else:
            print(f"  {key:24s} {value}")
    print(f"  wrote {path}")
    return path
