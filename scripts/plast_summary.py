"""Collect `outputs/plast/run_*.json` into one summary table.

Aggregates the per-run logs by condition (seeds pooled), lines the fly up against
the policy baselines on the same 60 evaluation games, and writes
`outputs/plast/summary.json`. Prints the tables that go into `REPORT.md`.

Run: ``python -m scripts.plast_summary``
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts import plast_common as K  # noqa: E402

OUT = K.OUT_DIR
_SEED_RE = re.compile(r"_s(\d+)$")

BUCKETS = ("lt0.25", "lt0.5", "lt1.0", "ge1.0")
HAND_TYPES = ("High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
              "Flush", "Full House", "Four of a Kind", "Straight Flush")


def condition_of(name: str) -> str:
    return _SEED_RE.sub("", name)


def _pool_p_play(tables: Sequence[Dict[str, dict]], keys: Sequence[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for k in keys:
        n = sum(t.get(k, {}).get("n", 0) for t in tables)
        plays = sum(t.get(k, {}).get("plays", 0) for t in tables)
        if n:
            out[k] = dict(n=n, plays=plays, p_play=round(plays / n, 4))
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT / "summary.json"))
    a = ap.parse_args(argv)

    runs: Dict[str, dict] = {}
    for p in sorted(OUT.glob("run_*.json")):
        d = json.loads(p.read_text())
        runs[d["spec"]["name"]] = d
    if not runs:
        raise SystemExit("no run_*.json in outputs/plast/")
    baselines = {}
    bl = OUT / "baselines.json"
    if bl.exists():
        baselines = json.loads(bl.read_text())
    probe = {}
    pp = OUT / "probe.json"
    if pp.exists():
        probe = json.loads(pp.read_text())

    by_cond: Dict[str, List[dict]] = {}
    for name, d in runs.items():
        by_cond.setdefault(condition_of(name), []).append(d)

    conditions: Dict[str, dict] = {}
    for cond, ds in sorted(by_cond.items()):
        pre = [d["pre"]["summary"] for d in ds]
        post = [d["post"]["summary"] for d in ds]
        tr = [d["train"]["summary"] for d in ds]
        row = dict(
            n_seeds=len(ds),
            seeds=[d["spec"]["seed"] for d in ds],
            spec={k: ds[0]["spec"][k] for k in
                  ("wiring", "eta", "tuning", "valence", "plasticity", "omission")},
            note=ds[0]["spec"]["note"],
            eval_games=ds[0]["eval_games"], train_hands=ds[0]["train_hands"],
            pre=dict(
                clear_rate=round(float(np.mean([s["clear_rate"] for s in pre])), 4),
                clear_rate_per_seed=[s["clear_rate"] for s in pre],
                chips_mean=round(float(np.mean([s["chips_mean"] for s in pre])), 1),
                p_play=round(float(np.mean([s["p_play"] for s in pre])), 4),
                p_play_when_legal=round(float(np.mean([s["p_play_when_legal"]
                                                      for s in pre])), 4),
                n_hands=int(np.mean([s["n_hands"] for s in pre])),
            ),
            post=dict(
                clear_rate=round(float(np.mean([s["clear_rate"] for s in post])), 4),
                clear_rate_per_seed=[s["clear_rate"] for s in post],
                chips_mean=round(float(np.mean([s["chips_mean"] for s in post])), 1),
                chips_sd_across_seeds=round(float(np.std([s["chips_mean"]
                                                         for s in post])), 1),
                p_play=round(float(np.mean([s["p_play"] for s in post])), 4),
                p_play_when_legal=round(float(np.mean([s["p_play_when_legal"]
                                                      for s in post])), 4),
                n_hands=int(np.mean([s["n_hands"] for s in post])),
            ),
            train=dict(
                reward_rate=round(float(np.mean([s["reward_rate"] for s in tr])), 4),
                punish_rate=round(float(np.mean([s["punish_rate"] for s in tr])), 4),
                p_play=round(float(np.mean([s["p_play"] for s in tr])), 4),
                clear_rate=round(float(np.mean([s["clear_rate"] for s in tr])), 4),
                n_games=int(np.mean([s["n_games"] for s in tr])),
                dopamine=_merge_counts([d["train"]["dopamine_counts"] for d in ds]),
                first_third_p_play=round(float(np.mean(
                    [d["train"]["first_third"]["p_play"] for d in ds])), 4),
                last_third_p_play=round(float(np.mean(
                    [d["train"]["last_third"]["p_play"] for d in ds])), 4),
                first_third_reward=round(float(np.mean(
                    [d["train"]["first_third"]["reward_rate"] for d in ds])), 4),
                last_third_reward=round(float(np.mean(
                    [d["train"]["last_third"]["reward_rate"] for d in ds])), 4),
            ),
            weights_after=dict(
                mean_ratio=round(float(np.mean(
                    [d["weight_stats_after_training"]["mean_ratio"] for d in ds])), 4),
                frac_at_floor=round(float(np.mean(
                    [d["weight_stats_after_training"]["frac_at_floor"] for d in ds])), 4),
                min_ratio=round(float(np.min(
                    [d["weight_stats_after_training"]["min_ratio"] for d in ds])), 4),
                any_negative=bool(any(
                    d["weight_stats_after_training"]["any_negative"] for d in ds)),
                any_above_original=bool(any(
                    d["weight_stats_after_training"]["any_above_original"] for d in ds)),
            ),
            p_play_by_bucket=dict(
                pre=_pool_p_play([d["pre"]["p_play_by_bucket"] for d in ds], BUCKETS),
                post=_pool_p_play([d["post"]["p_play_by_bucket"] for d in ds], BUCKETS),
            ),
            p_play_by_hand_type=dict(
                pre=_pool_p_play([d["pre"]["p_play_by_hand_type"] for d in ds],
                                 HAND_TYPES),
                post=_pool_p_play([d["post"]["p_play_by_hand_type"] for d in ds],
                                  HAND_TYPES),
            ),
            learning_curve_reward=_mean_curve(
                [d["train"]["reward_rate_rolling"] for d in ds]),
            learning_curve_p_play=_mean_curve(
                [d["train"]["p_play_rolling"] for d in ds]),
            learning_curve_play_drive=_mean_curve(
                [d["train"]["play_drive_rolling"] for d in ds]),
        )
        # is the learned policy the right shape? correlation of P(play) with bucket
        row["bucket_gradient"] = dict(
            pre=_gradient(row["p_play_by_bucket"]["pre"]),
            post=_gradient(row["p_play_by_bucket"]["post"]),
        )
        conditions[cond] = row

    payload = dict(
        conditions=conditions,
        baselines={k: v["summary"] for k, v in baselines.items()
                   if isinstance(v, dict) and "summary" in v},
        baseline_p_play_by_bucket={k: v["p_play_by_bucket"] for k, v in baselines.items()
                                   if isinstance(v, dict) and "p_play_by_bucket" in v},
        baseline_bucket_gradient={k: _gradient(v["p_play_by_bucket"])
                                  for k, v in baselines.items()
                                  if isinstance(v, dict) and "p_play_by_bucket" in v},
        sanity=_sanity_extract(probe),
        eval_seed0=K.EVAL_SEED0, train_seed0=K.TRAIN_SEED0, calib_seed0=K.CALIB_SEED0,
        runs=sorted(runs),
    )
    K.write_json(Path(a.out), payload)
    _print_tables(payload)
    print("wrote", a.out)


def _merge_counts(dicts: Sequence[Dict[str, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + int(v)
    return out


def _mean_curve(curves: Sequence[Sequence[float]], n: int = 60) -> List[float]:
    if not curves:
        return []
    m = min(len(c) for c in curves)
    if m == 0:
        return []
    arr = np.array([c[:m] for c in curves], float).mean(axis=0)
    idx = np.linspace(0, m - 1, min(n, m)).astype(int)
    return [round(float(arr[i]), 4) for i in idx]


def _gradient(table: Dict[str, dict]) -> Optional[float]:
    """P(play) at the two highest buckets minus the two lowest; + = right shape."""
    hi = [table[b] for b in ("lt1.0", "ge1.0") if b in table]
    lo = [table[b] for b in ("lt0.25", "lt0.5") if b in table]
    if not hi or not lo:
        return None
    h = sum(t["plays"] for t in hi) / sum(t["n"] for t in hi)
    l = sum(t["plays"] for t in lo) / sum(t["n"] for t in lo)
    return round(h - l, 4)


def _sanity_extract(probe: dict) -> dict:
    if not probe:
        return {}
    out: dict = {}
    for wiring in ("real", "shuffled"):
        sec = probe.get(wiring)
        if not sec:
            continue
        out[wiring] = dict(
            sanity_a={k: sec["sanity_a"][k] for k in
                      ("raw_min", "raw_max", "raw_range", "raw_sd", "quantum_hz",
                       "range_in_quanta", "kc_active_per_state", "kc_always_on",
                       "kc_input_dependent", "mbon_hz_mean", "approach_hz_mean",
                       "avoid_hz_mean") if k in sec["sanity_a"]},
            drive_by_bucket=sec["sanity_a"].get("by_bucket"),
            naive_policy=sec.get("naive_policy"),
            conditioning=sec.get("sanity_b_panel"),
        )
    if "real" in probe:
        out["bounds"] = probe["real"].get("sanity_c")
        out["window_scan"] = probe["real"].get("window_scan")
        out["valence"] = probe["real"].get("valence")
    return out


def _print_tables(payload: dict) -> None:
    print("\n== eval (60 games, seeds 100000+) ==")
    hdr = f"{'condition':34s} {'clear pre':>9s} {'clear post':>10s} {'chips pre':>9s} {'chips post':>10s} {'P(play) pre':>11s} {'post':>6s} {'grad pre':>8s} {'post':>6s}"
    print(hdr)
    for cond, r in payload["conditions"].items():
        print(f"{cond:34s} {r['pre']['clear_rate']:>9.3f} {r['post']['clear_rate']:>10.3f} "
              f"{r['pre']['chips_mean']:>9.0f} {r['post']['chips_mean']:>10.0f} "
              f"{r['pre']['p_play_when_legal']:>11.3f} {r['post']['p_play_when_legal']:>6.3f} "
              f"{str(r['bucket_gradient']['pre']):>8s} {str(r['bucket_gradient']['post']):>6s}")
    print("\n-- policy baselines, same games --")
    grads = payload["baseline_bucket_gradient"]
    for name, s in payload["baselines"].items():
        print(f"{name:34s} {'':>9s} {s['clear_rate']:>10.3f} {'':>9s} "
              f"{s['chips_mean']:>10.0f} {'':>11s} {s['p_play_when_legal']:>6.3f} "
              f"{'':>8s} {str(grads.get(name)):>6s}")
    print("\n== P(play) by score-vs-needed bucket ==")
    for cond, r in payload["conditions"].items():
        pre = r["p_play_by_bucket"]["pre"]
        post = r["p_play_by_bucket"]["post"]
        cells = "  ".join(
            f"{b}: {pre.get(b, {}).get('p_play', float('nan')):.2f}->"
            f"{post.get(b, {}).get('p_play', float('nan')):.2f}" for b in BUCKETS)
        print(f"{cond:34s} {cells}")


if __name__ == "__main__":
    main()
