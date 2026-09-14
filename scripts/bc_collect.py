"""Stage 1 of behavior cloning: collect (state, legal mask, expert action) triples.

Rollouts come from a 50/50 mix of two behaviour policies over disjoint fixed
seed blocks:

    heuristic-driven   seeds ``--seed0-heur + i``, actions from the teacher
    random-driven      seeds ``--seed0-rand + i``, actions uniform over the
                       legal mask (with the 48 no-op reorder/sort actions and
                       nothing else removed)

The *label* is always what the teacher would do in that state, so the random
block supplies off-policy coverage (DAgger-flavoured, one round) rather than a
second expert. Storing the label at collection time is mandatory: the teacher
reads ``env.engine.state``, which the feature vector does not reconstruct.

``--teacher`` picks the expert (``v1`` = the pair/flush heuristic, ``v2`` = the
hand-aware one in ``scripts/baseline_heuristic_v2.py``) and ``--feature-version``
picks the encoding (1 = 283 bits, 2 = the same bits behind a 32-bit hand-type
relay block). Both default to the v1 behaviour so an old run reproduces.

Everything is stored bit-packed. ``episode`` lets downstream stages split
train/test by episode instead of by state, which matters because consecutive
states inside one blind are near-duplicates.

Run:
    source .venv/bin/activate
    python scripts/bc_collect.py --target-states 150000
    python scripts/bc_collect.py --feature-version 2 --teacher v2 \
        --out outputs/bc2/states.npz

Writes outputs/bc/states.npz (skipped if it already exists; --force to redo).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from flybalatro.env import N_ACTIONS, BalatroEnv  # noqa: E402
from flybalatro.features_v2 import encoder_for_version, n_features_for_version  # noqa: E402

OUT = ROOT / "outputs" / "bc" / "states.npz"

BEHAVIOUR_HEURISTIC = 0
BEHAVIOUR_RANDOM = 1


def make_teacher(name: str):
    """``v1`` = pair/flush heuristic, ``v2`` = hand-aware teacher."""
    if name == "v1":
        from baseline_heuristic import HeuristicPolicy  # noqa: PLC0415

        return HeuristicPolicy()
    if name == "v2":
        from baseline_heuristic_v2 import HandAwarePolicy  # noqa: PLC0415

        return HandAwarePolicy()
    raise ValueError(f"unknown teacher {name!r}; expected v1 or v2")


def collect(target_states: int, seed0_heur: int, seed0_rand: int,
            max_steps: int, ante_end: int, rng_seed: int, verbose: bool = True,
            feature_version: int = 1, teacher: str = "v1"):
    n_features = n_features_for_version(feature_version)
    env = BalatroEnv(ante_end=ante_end, max_steps=max_steps, mask_noop_actions=True,
                     encoder=encoder_for_version(feature_version), n_features=n_features)
    expert = make_teacher(teacher)
    rng = np.random.default_rng(rng_seed)

    feats: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    episodes: list[int] = []
    behaviours: list[int] = []
    steps_in_ep: list[int] = []
    stages: list[int] = []

    ep_records: list[dict] = []
    episode_id = 0
    t0 = time.perf_counter()
    # Alternate blocks so that a truncated run still holds both behaviours.
    n_heur = n_rand = 0
    while len(feats) < target_states:
        use_heur = (episode_id % 2) == 0
        if use_heur:
            seed = seed0_heur + n_heur
            n_heur += 1
        else:
            seed = seed0_rand + n_rand
            n_rand += 1
        behaviour = BEHAVIOUR_HEURISTIC if use_heur else BEHAVIOUR_RANDOM

        features, mask = env.reset(seed=seed)
        step_i = 0
        while not env.done:
            stage = int(env.engine.state.stage.int())
            label = expert(env, mask)
            if mask[label] != 1:  # HeuristicPolicy guarantees legality; be loud.
                raise AssertionError(f"expert produced illegal action {label}")
            feats.append(features.astype(np.uint8))
            masks.append(mask.astype(np.uint8))
            labels.append(int(label))
            episodes.append(episode_id)
            behaviours.append(behaviour)
            steps_in_ep.append(step_i)
            stages.append(stage)

            if use_heur:
                action = label
            else:
                legal = np.flatnonzero(mask)
                action = int(legal[rng.integers(len(legal))])
            features, mask, _r, _done, info = env.step(action)
            step_i += 1
        ep_records.append(dict(episode=episode_id, seed=int(seed), behaviour=behaviour,
                               steps=int(env.steps), win=bool(env.is_win),
                               chips=int(env.chips_total)))
        episode_id += 1
        if verbose and episode_id % 500 == 0:
            el = time.perf_counter() - t0
            print(f"  {episode_id} episodes, {len(feats)} states, {el:.1f}s "
                  f"({len(feats)/el:,.0f} states/s)", flush=True)

    F = np.packbits(np.asarray(feats, np.uint8), axis=1)
    M = np.packbits(np.asarray(masks, np.uint8), axis=1)
    data = dict(
        features_packed=F,
        mask_packed=M,
        label=np.asarray(labels, np.int16),
        episode=np.asarray(episodes, np.int32),
        behaviour=np.asarray(behaviours, np.int8),
        step_in_episode=np.asarray(steps_in_ep, np.int16),
        stage=np.asarray(stages, np.int8),
        n_features=np.int32(n_features),
        n_actions=np.int32(N_ACTIONS),
        feature_version=np.int32(feature_version),
    )
    meta = dict(
        n_states=len(labels), n_episodes=episode_id,
        n_heuristic_episodes=n_heur, n_random_episodes=n_rand,
        seed0_heur=seed0_heur, seed0_rand=seed0_rand,
        ante_end=ante_end, max_steps=max_steps, mask_noop_actions=True,
        feature_version=int(feature_version), teacher=teacher,
        wall_seconds=round(time.perf_counter() - t0, 1),
        episode_records_head=ep_records[:5],
    )
    return data, meta


def unpack_features(z) -> np.ndarray:
    return np.unpackbits(z["features_packed"], axis=1)[:, : int(z["n_features"])]


def unpack_masks(z) -> np.ndarray:
    return np.unpackbits(z["mask_packed"], axis=1)[:, : int(z["n_actions"])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-states", type=int, default=150_000)
    ap.add_argument("--seed0-heur", type=int, default=0)
    ap.add_argument("--seed0-rand", type=int, default=500_000)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--ante-end", type=int, default=1)
    ap.add_argument("--rng-seed", type=int, default=17)
    ap.add_argument("--feature-version", type=int, default=1, choices=(1, 2))
    ap.add_argument("--teacher", default="v1", choices=("v1", "v2"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.force:
        z = np.load(out, allow_pickle=False)
        print(f"{out} exists ({len(z['label']):,} states); skipping. --force to redo.")
        return

    data, meta = collect(a.target_states, a.seed0_heur, a.seed0_rand,
                         a.max_steps, a.ante_end, a.rng_seed,
                         feature_version=a.feature_version, teacher=a.teacher)

    F = np.unpackbits(data["features_packed"], axis=1)[:, : int(data["n_features"])]
    uniq = len(np.unique(data["features_packed"], axis=0))
    lab = data["label"]
    meta["unique_feature_vectors"] = int(uniq)
    meta["duplicate_fraction"] = round(1.0 - uniq / len(lab), 4)
    meta["mean_bits_active"] = round(float(F.sum(1).mean()), 2)
    meta["label_entropy_bits"] = round(
        float(-sum(p * np.log2(p) for p in np.bincount(lab, minlength=N_ACTIONS) / len(lab) if p > 0)), 3)
    top = np.argsort(np.bincount(lab, minlength=N_ACTIONS))[::-1][:8]
    meta["top_labels"] = [[int(i), int(np.sum(lab == i))] for i in top]
    meta["majority_label_rate"] = round(float(np.bincount(lab, minlength=N_ACTIONS).max() / len(lab)), 4)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    (out.parent / "collect_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "episode_records_head"}, indent=2))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
