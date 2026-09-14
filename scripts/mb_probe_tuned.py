"""The identity probe again, with the mushroom body calibrated.

Same gate as ``scripts/brain_probe.py`` and the same helpers (``make_dataset``,
``probe``, ``readout_stats``, ``glomerular_ceiling``): 600 random 180-bit inputs
at 20% active, 10 ORNs per bit, 30 mV tonic, one window from reset, log1p spike
counts -> StandardScaler -> L2 logistic regression, 5-fold CV, best of a C grid,
with a permuted-label run through the identical pipeline as the honest chance
level.

What is new:
  * every readout is measured at several ``flybalatro.tuning`` settings, with the
    untuned model re-run in the same process as the controlled baseline;
  * the KC readout is also probed on **late-window counts only** (spikes from
    ``--late-from-ms`` onwards), because the first volley reaches the Kenyon cells
    before APL feedback can arrive and is dense whatever the tuning;
  * the intensity control ("number of active input bits above the median") is run
    on KC and MBON, so a gain in accuracy can be attributed to identity rather
    than to the readout counting how much drive went in.

The shuffled-wiring control is not repeated here; it is in
``outputs/brain_probe.json`` and is unaffected by these knobs.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from flybalatro import connectome as C
from flybalatro.brain import Brain
from flybalatro.tuning import Tuning
from scripts.brain_probe import (
    C_GRID,
    glomerular_ceiling,
    make_dataset,
    probe,
    readout_stats,
)
from scripts.mb_sparsity import OUT_DIR, build_inputs, populations, selector_provenance

POPS = ("ALPN", "KC", "MBON", "DN")


def simulate(brain: Brain, fm, F: np.ndarray, pops: dict, window_ms: float,
             drive_mv: float, late_from_ms: int, label: str = "") -> tuple[dict, float]:
    """Full-window counts per population, plus KC counts from `late_from_ms` on."""
    mats = {k: np.zeros((len(F), len(pops[k])), np.int32) for k in POPS}
    kc_late = np.zeros((len(F), len(pops["KC"])), np.int32)
    kc = pops["KC"]
    t0 = time.perf_counter()
    brain.warmup()
    early_ms = min(late_from_ms, int(round(window_ms)))
    for i, f in enumerate(F):
        d = fm.drive(f, drive_mv=drive_mv)
        brain.reset()
        if early_ms > 0:
            brain.step(d, float(early_ms), reset_counts=False)
            early = brain.counts[kc].copy()
        else:
            early = np.zeros(len(kc), np.int32)
        if window_ms - early_ms > 0:
            brain.step(d, float(window_ms - early_ms), reset_counts=False)
        for k in POPS:
            mats[k][i] = brain.counts[pops[k]]
        kc_late[i] = mats["KC"][i] - early
        if label and (i + 1) % 200 == 0:
            print(f"    [{label}] {i+1}/{len(F)}  {time.perf_counter()-t0:.0f}s", flush=True)
    mats["KC_late"] = kc_late
    return mats, time.perf_counter() - t0


def run_probes(mats: dict, ys: dict, seed: int, rng: np.random.Generator) -> dict:
    """Every (label, readout) pair, plus a permuted-label run per readout."""
    names = list(POPS) + ["KC_late"]
    out: dict = {}
    for lab, y in ys.items():
        block = {}
        for name in names:
            X = np.log1p(mats[name].astype(np.float64))
            block[name] = probe(X, y, seed)
        yp = rng.permutation(y)
        for name in names:
            X = np.log1p(mats[name].astype(np.float64))
            block[f"permuted:{name}"] = probe(X, yp, seed)
        out[lab] = dict(
            majority_class_rate=round(float(max(y.mean(), 1 - y.mean())), 4),
            readouts=block,
        )
    return out


def print_probes(tag: str, res: dict) -> None:
    for lab, blk in res.items():
        print(f"  [{tag}] label {lab}  (majority {blk['majority_class_rate']})")
        for name, r in blk["readouts"].items():
            if name.startswith("permuted:"):
                continue
            p = blk["readouts"][f"permuted:{name}"]["best_acc"]
            print(f"    {name:<10} acc {r['best_acc']}   permuted {p}   "
                  f"({r['n_used_features']}/{r['n_total_features']} cols)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--n-features", type=int, default=180)
    ap.add_argument("--neurons-per-feature", type=int, default=10)
    ap.add_argument("--active-frac", type=float, default=0.20)
    ap.add_argument("--window", type=float, default=50.0)
    ap.add_argument("--drive-mv", type=float, default=30.0)
    ap.add_argument("--v-jitter", type=float, default=6.9)
    ap.add_argument("--late-from-ms", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--brain-seed", type=int, default=1)
    ap.add_argument("--settings", default='[{}]',
                    help='JSON list of tuning dicts; {} is the untuned baseline')
    ap.add_argument("--tag", default="probe_tuned")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-counts", default=None)
    a = ap.parse_args()

    tunings = [Tuning.from_dict(d) for d in json.loads(a.settings)]
    g = C.load(a.threshold)
    pops = populations(g)
    fm, F, ys, spec = build_inputs(g, a.n_samples, a.n_features,
                                   a.neurons_per_feature, a.active_frac,
                                   a.drive_mv, a.seed)
    # Intensity control: can the readout just count how much drive went in?
    bits = F.sum(1)
    ys_all = dict(ys)
    ys_all["intensity"] = (bits > np.median(bits)).astype(np.int8)
    M_ceil, chans = glomerular_ceiling(fm, g, F)

    print("populations: " + "  ".join(f"{k}={len(v)}" for k, v in pops.items()))
    print(f"window {a.window} ms  drive {a.drive_mv} mV  late-from {a.late_from_ms} ms")
    print(f"{len(tunings)} settings: " + " | ".join(t.label() for t in tunings),
          flush=True)

    rng = np.random.default_rng(a.seed + 999)
    reference = {}
    for lab, y in ys_all.items():
        reference[lab] = dict(
            raw_bits=probe(F.astype(np.float64), y, a.seed),
            glomerular_ceiling=probe(np.log1p(M_ceil), y, a.seed),
            permuted_raw_bits=probe(F.astype(np.float64), rng.permutation(y), a.seed),
            majority_class_rate=round(float(max(y.mean(), 1 - y.mean())), 4),
        )
        print(f"  reference label {lab}: raw_bits {reference[lab]['raw_bits']['best_acc']}"
              f"  ceiling {reference[lab]['glomerular_ceiling']['best_acc']}"
              f"  permuted_raw {reference[lab]['permuted_raw_bits']['best_acc']}", flush=True)

    runs: dict = {}
    counts_dump: dict = {}
    for t in tunings:
        print(f"\nsimulating {t.label()} ...", flush=True)
        brain = Brain(g, seed=a.brain_seed, v_jitter=a.v_jitter, tuning=t)
        mats, secs = simulate(brain, fm, F, pops, a.window, a.drive_mv,
                              a.late_from_ms, t.label())
        print(f"  {secs:.0f}s", flush=True)
        res = run_probes(mats, ys_all, a.seed, np.random.default_rng(a.seed + 999))
        print_probes(t.label(), res)
        runs[t.label()] = dict(
            tuning=t.to_dict(), tuning_info=brain.tuning_info,
            sim_seconds=round(secs, 1),
            firing={k: readout_stats(v) for k, v in mats.items()},
            results=res,
        )
        if a.save_counts:
            for k, v in mats.items():
                counts_dump[f"{t.label()}__{k}"] = v

    out = dict(
        tag=a.tag,
        settings=dict(
            threshold=a.threshold, n_samples=a.n_samples, n_features=a.n_features,
            neurons_per_feature=a.neurons_per_feature, active_frac=a.active_frac,
            window_ms=a.window, drive_mv=a.drive_mv, v_jitter_mv=a.v_jitter,
            late_from_ms=a.late_from_ms, dataset_seed=a.seed,
            brain_seed=a.brain_seed, C_grid=C_GRID,
            cv="StratifiedKFold(5, shuffle=True, random_state=seed)",
            readout_transform="log1p(counts) -> StandardScaler -> LogisticRegression(L2)",
            feature_map=fm.describe(), n_glomerular_channels=len(chans),
            machine=f"{platform.machine()} python {platform.python_version()}",
            note="tuning values are calibration knobs, not connectome data",
        ),
        dataset=dict(spec, intensity_control=dict(
            kind="number of active input bits > median",
            median=float(np.median(bits)),
            positive_rate=round(float(ys_all["intensity"].mean()), 4))),
        selectors=selector_provenance(g),
        populations={k: len(v) for k, v in pops.items()},
        reference=reference,
        runs=runs,
    )
    p = Path(a.out) if a.out else OUT_DIR / f"{a.tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    if a.save_counts:
        sc = Path(a.save_counts)
        sc.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sc, F=F, **{f"y_{k}": v for k, v in ys_all.items()},
                            **counts_dump)
        print("wrote counts", sc)
    print("wrote", p)


if __name__ == "__main__":
    main()
