"""Does the APL correction disturb the Kenyon-cell code `outputs/mb2/REPORT.md` measured?

`outputs/plast/tuned_config.json` adds two knobs to the mb2 tuning
(``apl_kc_only``, ``apl_mbon_scale``) so the MBON output stage is not silent.
Both only touch APL's output, and only ``apl_kc_only`` can reach the Kenyon cells
at all, but it does change the KC code slightly (APL's 6% non-KC output is no
longer doubled, and some of that returns to the calyx indirectly), so the claim
"the Kenyon-cell code is essentially unchanged" has to be measured, not asserted.

Reports, for both tunings, on the same 200 calibration hands: KC active fraction,
always-on fraction, input-dependent cell count, and 9-way hand-type decoding
accuracy from the KC spike counts, plain 5-fold and grouped by relay pattern,
against a permuted-label baseline. Same probe as `scripts/mb2_probe.py`.

Run: ``python -m scripts.plast_kccheck``  (~1 min)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts import plast_common as K  # noqa: E402


def _probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> dict:
    keep = X.std(axis=0) > 0
    Xs = X[:, keep] if keep.any() else X[:, :1]
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0, n_jobs=1))
    plain = float(cross_val_score(pipe, Xs, y,
                                  cv=StratifiedKFold(5, shuffle=True,
                                                     random_state=seed)).mean())
    rng = np.random.default_rng(seed)
    yp = rng.permutation(y)
    perm = float(cross_val_score(pipe, Xs, yp,
                                 cv=StratifiedKFold(5, shuffle=True,
                                                    random_state=seed)).mean())
    n_groups = len(np.unique(groups))
    grouped = None
    if n_groups >= 5:
        try:
            grouped = float(cross_val_score(
                pipe, Xs, y, groups=groups,
                cv=StratifiedGroupKFold(5, shuffle=True, random_state=seed)).mean())
        except ValueError:
            grouped = None
    return dict(n=int(len(y)), n_features=int(Xs.shape[1]), accuracy=round(plain, 4),
                grouped=None if grouped is None else round(grouped, 4),
                permuted=round(perm, 4), n_groups=int(n_groups))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--out", default=str(K.OUT_DIR / "kc_check.json"))
    a = ap.parse_args(argv)

    from scripts.plast_probe import collect_calibration_hands, hand_bits

    graph = K.C.load(5)
    recs = collect_calibration_hands(a.n_calib, K.CALIB_SEED0, rng_seed=7)
    bits = hand_bits(recs)
    y = np.array([r["best_type"] for r in recs])
    groups = np.array([int(r["odor"]) for r in recs])

    payload: dict = dict(n_hands=len(recs), n_patterns=int(len(np.unique(groups))),
                         label="9-way best hand type")
    for variant in ("spec", "plast"):
        cfg = K.load_config(variant)
        setup = K.build_setup("real", graph=graph, cfg=cfg)
        X = np.zeros((len(bits), len(setup.kc)), np.float32)
        M = np.zeros((len(bits), len(setup.valence.mbon)), np.float32)
        for i, b in enumerate(bits):
            c = setup.run_window(b)
            X[i] = c[setup.kc]
            M[i] = c[setup.valence.mbon]
        act = X > 0
        rate = act.mean(axis=0)
        row = dict(
            tuning=cfg["tuning"],
            kc_active_frac=round(float(act.mean()), 4),
            kc_active_per_state=round(float(act.sum(axis=1).mean()), 1),
            kc_always_on_frac=round(float((rate > 0.5).mean()), 4),
            kc_always_on=int((rate > 0.5).sum()),
            kc_input_dependent=int(((rate > 0) & (rate < 1)).sum()),
            mbon_hz=round(float(M.mean()) * 1000.0 / setup.window_ms, 3),
            mbon_varying=int((((M > 0).mean(axis=0) > 0)
                              & ((M > 0).mean(axis=0) < 1)).sum()),
            kc_hand_type=_probe(np.log1p(X), y, groups),
            kcbin_hand_type=_probe(act.astype(np.float32), y, groups),
            mbon_hand_type=_probe(np.log1p(M), y, groups),
        )
        payload[variant] = row
        print(f"== {variant} ==")
        print(f"   KC {row['kc_active_per_state']}/state ({row['kc_active_frac']:.4f}), "
              f"always-on {row['kc_always_on']} ({row['kc_always_on_frac']:.4f}), "
              f"input-dependent {row['kc_input_dependent']}")
        print(f"   KC hand type {row['kc_hand_type']['accuracy']:.3f} "
              f"(grouped {row['kc_hand_type']['grouped']}, "
              f"permuted {row['kc_hand_type']['permuted']:.3f})  "
              f"KCbin {row['kcbin_hand_type']['accuracy']:.3f}")
        print(f"   MBON {row['mbon_hz']:.2f} Hz, {row['mbon_varying']} varying, "
              f"hand type {row['mbon_hand_type']['accuracy']:.3f}")
        del setup
    K.write_json(Path(a.out), payload)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
