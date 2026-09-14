"""Paired comparisons between the real calyx and each rewired one, plus the tables.

Nothing here simulates anything; it reads what the other ``calyx_*`` stages
wrote and produces the numbers the report quotes.

Why paired. Every condition plays the *same* 400 episode seeds and is scored on
the *same* held-out behaviour-cloning states, so the difference between two
conditions is a within-unit difference and an unpaired z-test on two rates
throws away most of the power (and, worse, understates the precision on a
comparison where the two rows share their episode-to-episode variance almost
entirely).

* clear rate: exact McNemar on the 400 shared episode seeds -- the discordant
  pairs (real cleared / rewired did not, and the reverse) are the whole evidence;
* mean chips and imitation top-1: paired bootstrap, resampling **episodes**
  (never states: consecutive states inside a blind are near-duplicates);
* across the three rewiring seeds, the spread is reported as a range, not as a
  confidence interval -- three draws do not support one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent
for _p in (str(ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import calyx_common as K  # noqa: E402
from scripts import calyx_feats as F  # noqa: E402

OUT_DIR = K.OUT_DIR
BOOT = 10_000


def mcnemar_exact(a_win: np.ndarray, b_win: np.ndarray) -> dict:
    from scipy.stats import binomtest

    b = int((a_win & ~b_win).sum())
    c = int((~a_win & b_win).sum())
    n = b + c
    p = float(binomtest(b, n, 0.5).pvalue) if n else 1.0
    return dict(a_only=b, b_only=c, both=int((a_win & b_win).sum()),
                neither=int((~a_win & ~b_win).sum()),
                diff=round(float(a_win.mean() - b_win.mean()), 4),
                mcnemar_p=round(p, 5), n_discordant=n)


def paired_bootstrap(x: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
                     n: int = BOOT, seed: int = 0) -> dict:
    """CI of ``mean(x) - mean(y)`` resampling the shared unit (episode) with repl."""
    rng = np.random.default_rng(seed)
    if groups is None:
        idx_of = [np.array([i]) for i in range(len(x))]
    else:
        uniq = np.unique(groups)
        order = np.argsort(groups, kind="stable")
        gs = groups[order]
        starts = np.searchsorted(gs, uniq, "left")
        ends = np.searchsorted(gs, uniq, "right")
        idx_of = [order[s:e] for s, e in zip(starts, ends)]
    ng = len(idx_of)
    d = np.empty(n)
    for k in range(n):
        pick = rng.integers(0, ng, ng)
        sel = np.concatenate([idx_of[i] for i in pick])
        d[k] = x[sel].mean() - y[sel].mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(diff=round(float(x.mean() - y.mean()), 5),
                ci95=[round(float(lo), 5), round(float(hi), 5)],
                p_two_sided=round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 5),
                n_units=int(ng))


def load_episodes(variant: str, cond: str, readout: str,
                  out_dir: Path = OUT_DIR) -> Optional[dict]:
    p = F.cond_dir(variant, cond, out_dir) / f"episodes_calyx_{cond}_{readout}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def load_correct(variant: str, cond: str, readout: str,
                 out_dir: Path = OUT_DIR) -> Optional[dict]:
    p = F.cond_dir(variant, cond, out_dir) / f"test_correct_{readout}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def compare(variant: str, conds: Sequence[str], readouts: Sequence[str],
            out_dir: Path = OUT_DIR) -> dict:
    out: Dict[str, dict] = {}
    for ro in readouts:
        base_ep = load_episodes(variant, "real", ro, out_dir)
        base_co = load_correct(variant, "real", ro, out_dir)
        for cond in conds:
            if cond == "real":
                continue
            rec: Dict[str, dict] = {}
            ep = load_episodes(variant, cond, ro, out_dir)
            if base_ep is not None and ep is not None:
                assert np.array_equal(base_ep["seed"], ep["seed"]), "episode seeds differ"
                rec["clear_rate"] = dict(
                    real=round(float(base_ep["win"].mean()), 4),
                    rewired=round(float(ep["win"].mean()), 4),
                    **mcnemar_exact(base_ep["win"], ep["win"]))
                rec["chips"] = dict(
                    real=round(float(base_ep["chips"].mean()), 1),
                    rewired=round(float(ep["chips"].mean()), 1),
                    **paired_bootstrap(base_ep["chips"].astype(float),
                                       ep["chips"].astype(float)))
                pb_r = base_ep["plays_best"].sum() / max(1, base_ep["plays"].sum())
                pb_w = ep["plays_best"].sum() / max(1, ep["plays"].sum())
                rec["best_subset_play_fraction"] = dict(
                    real=round(float(pb_r), 4), rewired=round(float(pb_w), 4),
                    diff=round(float(pb_r - pb_w), 4))
            co = load_correct(variant, cond, ro, out_dir)
            if base_co is not None and co is not None:
                assert np.array_equal(base_co["state"], co["state"]), "test sets differ"
                a = base_co["correct"].astype(float)
                b = co["correct"].astype(float)
                rec["imitation_top1"] = dict(
                    real=round(float(a.mean()), 4), rewired=round(float(b.mean()), 4),
                    real_right_rewired_wrong=int(((a > 0) & (b == 0)).sum()),
                    rewired_right_real_wrong=int(((b > 0) & (a == 0)).sum()),
                    n_states=int(len(a)),
                    **paired_bootstrap(a, b, groups=base_co["episode"]))
            if rec:
                out[f"{ro}/{cond}"] = rec
    return out


def seed_spread(cmp: dict, readouts: Sequence[str]) -> dict:
    """Across-rewiring-seed range of each paired difference."""
    out: Dict[str, dict] = {}
    for ro in readouts:
        rows = [v for k, v in cmp.items() if k.startswith(f"{ro}/")]
        if not rows:
            continue
        entry: Dict[str, dict] = {}
        for metric, field in (("imitation_top1", "diff"), ("clear_rate", "diff"),
                              ("chips", "diff")):
            vals = [r[metric][field] for r in rows if metric in r]
            if vals:
                entry[metric] = dict(
                    n_seeds=len(vals),
                    mean_diff_real_minus_rewired=round(float(np.mean(vals)), 5),
                    min=round(float(np.min(vals)), 5),
                    max=round(float(np.max(vals)), 5))
        out[ro] = entry
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="raw", choices=F.VARIANTS)
    ap.add_argument("--conditions", default="real,rw1,rw2,rw3")
    ap.add_argument("--readouts", default="kc,dn")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    ros = [r.strip() for r in a.readouts.split(",") if r.strip()]
    cmp = compare(a.variant, conds, ros, Path(a.out_dir))
    payload = dict(variant=a.variant, conditions=conds, readouts=ros,
                   paired=cmp, across_seeds=seed_spread(cmp, ros))
    p = Path(a.out_dir) / f"paired_{a.variant}.json"
    p.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(json.dumps(payload["across_seeds"], indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
