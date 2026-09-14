"""Check a saved KC -> MBON weight file against the run that produced it.

``--save-weights`` writes the learned synapses; this reads them back onto a
freshly built fly and re-runs the *post*-training evaluation from scratch. If the
file carries what it claims, the numbers have to come out the same, because
nothing in that evaluation is random except one seeded generator:
``scripts/plast_run.py``'s ``_evaluate`` uses ``default_rng(50_000 + seed)``, the
frozen calibration, the same 60 games from ``EVAL_SEED0`` and no dopamine. So
this is an equality check, not a "within noise" check, and it fails loudly if the
edge alignment, the calibration or the exploration floor is off by anything.

Usage::

    python -m scripts.plast_weights --config v2
    python -m scripts.plast_weights --config v1 --eval-games 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from flybalatro.viewer.plastic import PLASTIC_CONFIGS
from scripts import plast_common as K


def evaluate(setup, decider, eval_games: int, seed: int) -> dict:
    """``plast_run.Runner._evaluate`` with the weights already in place."""
    env = K.make_env()
    rng = np.random.default_rng(50_000 + seed)
    fly = K.FlyPolicy(setup, decider, rng)
    games = [K.play_game(env, K.EVAL_SEED0 + i, fly) for i in range(eval_games)]
    records = [h for g in games for h in g.hands]
    return dict(
        summary=K.summarise_games(games),
        p_play_by_bucket=K.p_play_table(records, "bucket_name"),
        weight_stats=setup.plast.weight_stats(),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=sorted(PLASTIC_CONFIGS), default="v2")
    ap.add_argument("--weights", default=None,
                    help="default: the config's reference weights file")
    ap.add_argument("--run-json", default=None,
                    help="default: the config's published run_<reference>.json")
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0,
                    help="the run's seed; it selects the evaluation RNG")
    ap.add_argument("--explore-floor", type=float, default=None,
                    help="default: whatever the weights file recorded")
    a = ap.parse_args(argv)

    spec = PLASTIC_CONFIGS[a.config]
    weights = Path(a.weights) if a.weights else spec.weights_path
    run_json = Path(a.run_json) if a.run_json else (
        spec.out_dir / f"run_{spec.reference_run}.json")
    if not weights.exists():
        raise SystemExit(f"{weights} does not exist")

    t0 = time.time()
    cfg = json.loads(spec.config_path.read_text())
    setup = K.build_setup(wiring="real", valence_scheme="nt", cfg=cfg)
    loaded = setup.plast.load_weights(weights)
    meta = loaded["meta"]
    floor = (a.explore_floor if a.explore_floor is not None
             else float(meta.get("explore_floor", 0.1)))
    decider = K.make_decider(setup, explore_floor=floor)
    decider.bias = float(meta["bias"])
    decider.temperature = float(meta["temperature"])

    print(f"config      {spec.config_path.relative_to(K.ROOT)}")
    print(f"weights     {weights.relative_to(K.ROOT)}  "
          f"({loaded['n_edges']:,} edges, mean {loaded['mean_ratio']:.6f} of "
          f"original, {loaded['frac_at_floor']:.4%} at the floor)")
    print(f"calibration bias {decider.bias:+.6f}  temperature "
          f"{decider.temperature:.6f}  explore floor {floor}")
    print(f"decider     {len(decider.approach)} approach / {len(decider.avoid)} "
          f"avoid MBONs")

    got = evaluate(setup, decider, a.eval_games, a.seed)
    s = got["summary"]
    print(f"\nreloaded fly, {a.eval_games} games from seed {K.EVAL_SEED0}:")
    print(f"  clear {s['clear_rate']:.4f}  chips {s['chips_mean']:.3f}  "
          f"P(play|legal) {s['p_play_when_legal']:.6f}  hands {s['n_hands']}")
    for b, row in sorted(got["p_play_by_bucket"].items()):
        print(f"    {b:8s} n={row['n']:4d}  P(play) {row['p_play']:.4f}")

    if not run_json.exists():
        print(f"\n(no {run_json} to compare against)")
        return 0

    published = json.loads(run_json.read_text())["post"]
    p = published["summary"]
    print(f"\npublished post-training ({run_json.relative_to(K.ROOT)}):")
    print(f"  clear {p['clear_rate']:.4f}  chips {p['chips_mean']:.3f}  "
          f"P(play|legal) {p['p_play_when_legal']:.6f}  hands {p['n_hands']}")

    checks = [
        ("P(play|legal)", s["p_play_when_legal"], p["p_play_when_legal"]),
        ("clear rate", s["clear_rate"], p["clear_rate"]),
        ("mean chips", s["chips_mean"], p["chips_mean"]),
        ("hands", float(s["n_hands"]), float(p["n_hands"])),
    ]
    bad = [name for name, a_, b_ in checks if abs(a_ - b_) > 1e-9]
    width = max(len(n) for n, _, _ in checks)
    print()
    for name, mine, theirs in checks:
        flag = "OK  " if abs(mine - theirs) <= 1e-9 else "DIFF"
        print(f"  {flag} {name:{width}s} {mine:.6f} vs {theirs:.6f}")
    print(f"\n{time.time() - t0:.0f}s")
    if bad:
        print("MISMATCH in: " + ", ".join(bad))
        return 1
    print("the reloaded weights reproduce the published post-training run exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
