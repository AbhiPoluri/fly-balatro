"""Probe phase: can a linear readout recover the labels from the saved counts?

Reads ``outputs/mb2/counts_<label>.npz`` written by ``scripts/mb2_grid.py`` and
writes ``outputs/mb2/probes.json`` (its own file, so it can run while the grid is
still simulating later settings). Separate from the simulation so a probe change
costs no resimulation, and resumable per (setting, population, window, label).

Probe = ``scripts.brain_probe.probe``: 5-fold stratified CV logistic regression
on ``log1p`` spike counts, zero-variance columns dropped, best of a C grid, and
the identical pipeline on permuted labels as the honest chance level. The C grid
is shortened to 3 values (``mb2_sample.C_GRID_SHORT``) because the full 6-value
grid would cost more than the whole simulation budget at 13 settings x 4
populations x 2 windows x 3 labels x 2.

Labels:
  hand_type    9-way, relay bits 0-8. The gate label.
  sel_is_best  binary, relay bit 27. 11% positive, so accuracy is reported next
               to the 0.89 majority floor and a balanced accuracy.
  intensity    control: driven-ORN count above the median. The previous round's
               result was that the mushroom body is a total-drive meter and a
               chance-level identity detector; this column is how we check
               whether that is still true.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.brain_probe import probe
from scripts.mb2_sample import C_GRID_SHORT

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "mb2"
GRID = OUT_DIR / "grid.json"
PROBES = OUT_DIR / "probes.json"

#: ``KCbin`` is the KC matrix binarised to active/inactive. The graded readout can
#: succeed on spike *counts* while the active *set* is nearly the same every time
#: (the always-on / Jaccard problem), and a KC->MBON plasticity rule that only
#: cares which KCs fired would then have nothing to work with. This column
#: separates the two.
POPS: tuple[str, ...] = ("KC", "KCbin", "MBON", "ALPN", "DN")
WINDOWS: tuple[str, ...] = ("full", "late")
LABELS: tuple[str, ...] = ("hand_type", "sel_is_best", "intensity")


def probe_grouped(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  c_grid=C_GRID_SHORT, seed: int = 0) -> float:
    """Best-of-grid accuracy with no relay pattern shared between train and test.

    The brain is a deterministic function of the 32 relay bits, and 2,000 sampled
    states contain only 1,169 distinct relay patterns, so plain k-fold CV can put
    a bit-identical count vector in both the training and the test fold. Grouping
    the folds by relay pattern removes that, and is the number to quote.
    """
    keep = X.std(0) > 0
    if keep.sum() == 0:
        return None
    Xk = X[:, keep]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    best = -1.0
    for c in c_grid:
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=c, max_iter=5000, solver="lbfgs"))
        s = cross_val_score(pipe, Xk, y, cv=cv, groups=groups, scoring="accuracy",
                            n_jobs=-1)
        best = max(best, float(s.mean()))
    return round(best, 4)


def balanced_acc(X: np.ndarray, y: np.ndarray, c: float, seed: int = 0) -> float:
    """Balanced accuracy at one C -- the binary label is 11% positive."""
    keep = X.std(0) > 0
    if keep.sum() == 0:
        return float("nan")
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=c, max_iter=5000, solver="lbfgs"))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    s = cross_val_score(pipe, X[:, keep], y, cv=cv, scoring="balanced_accuracy",
                        n_jobs=-1)
    return round(float(s.mean()), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated setting labels")
    ap.add_argument("--pops", default=",".join(POPS))
    ap.add_argument("--labels", default=",".join(LABELS))
    ap.add_argument("--perm-seed", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    z = np.load(OUT_DIR / "sample.npz", allow_pickle=True)
    ys = dict(hand_type=z["y_hand"], sel_is_best=z["y_best"],
              intensity=z["y_intensity"])
    rng = np.random.default_rng(a.perm_seed)
    perm = {k: rng.permutation(v) for k, v in ys.items()}
    _, groups = np.unique(z["relay"], axis=0, return_inverse=True)

    pops = [p for p in a.pops.split(",") if p]
    labels = [l for l in a.labels.split(",") if l]
    todo = sorted(l for l in json.loads(GRID.read_text())["settings"]
                  if not a.only or l in set(a.only.split(",")))

    for lbl in todo:
        cpath = OUT_DIR / f"counts_{lbl}.npz"
        if not cpath.exists():
            print(f"skip {lbl}: no counts file")
            continue
        cz = np.load(cpath)
        out = json.loads(PROBES.read_text()) if PROBES.exists() else {}
        block = out.setdefault(lbl, {})
        for pop in pops:
            src = "KC" if pop == "KCbin" else pop
            full = cz[src].astype(np.int32)
            late = full - cz[src + "_early"].astype(np.int32)
            if pop == "KCbin":
                full, late = (full > 0).astype(np.int32), (late > 0).astype(np.int32)
            for win, M in (("full", full), ("late", late)):
                X = np.log1p(M.astype(np.float64))
                for lab in labels:
                    key = f"{pop}.{win}.{lab}"
                    if key in block and not a.force:
                        continue
                    t0 = time.perf_counter()
                    r = probe(X, ys[lab], c_grid=C_GRID_SHORT)
                    rp = probe(X, perm[lab], c_grid=C_GRID_SHORT)
                    rec = dict(acc=r["best_acc"], permuted=rp["best_acc"],
                               margin=(round(r["best_acc"] - rp["best_acc"], 4)
                                       if r["best_acc"] is not None
                                       and rp["best_acc"] is not None else None),
                               best_C=r["best_C"],
                               n_used_features=r["n_used_features"],
                               n_total_features=r["n_total_features"])
                    rec["grouped_acc"] = probe_grouped(X, ys[lab], groups)
                    rec["grouped_permuted"] = probe_grouped(X, perm[lab], groups)
                    if lab == "sel_is_best" and r["best_acc"] is not None:
                        rec["balanced_acc"] = balanced_acc(X, ys[lab], r["best_C"])
                        rec["balanced_permuted"] = balanced_acc(
                            X, perm[lab], r["best_C"])
                    rec["seconds"] = round(time.perf_counter() - t0, 1)
                    block[key] = rec
                    print(f"  {lbl:24s} {key:22s} acc {rec['acc']} "
                          f"grp {rec['grouped_acc']} "
                          f"perm {rec['permuted']} margin {rec['margin']} "
                          f"cols {rec['n_used_features']} "
                          f"({rec['seconds']}s)", flush=True)
                    out[lbl] = block
                    PROBES.write_text(json.dumps(out, indent=2))
    print("wrote", PROBES)


if __name__ == "__main__":
    main()
