"""THE GATE: does information injected at the ORNs reach a linearly-readable code
in the descending neurons?

N random sparse binary feature vectors, two synthetic labels, and a 5-fold CV
logistic-regression linear probe under three conditions: real wiring, shuffled
wiring, and the raw feature bits with no brain.

Readouts are layered along the olfactory pathway so a failure can be localised:

    glomerular_ceiling  no simulation at all: active driven ORNs per (ORN type,
                        side) channel. The most any downstream population could
                        know if the antennal lobe encoded ORN counts perfectly.
    ORN                 the driven sensory layer itself
    ALPN                antennal-lobe projection neurons, one hop downstream
    KC                  mushroom-body Kenyon cells
    MBON / CX / DN      candidate RL readouts

A permuted-label probe gives the honest chance level, since taking the best of a
C grid inflates accuracy above the majority-class rate.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flybalatro import connectome as C
from flybalatro.brain import Brain
from flybalatro.encode import FeatureMap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "brain_probe.json"
C_GRID = [0.003, 0.03, 0.3, 3.0, 30.0, 300.0]


def make_dataset(n_samples, n_features, active_frac, seed):
    rng = np.random.default_rng(seed)
    F = (rng.random((n_samples, n_features)) < active_frac).astype(np.int8)

    # Label A: "count of bits in group X >= k". k is picked to balance the classes;
    # a 12-bit group keeps the label dependent on few bits so the raw-bit reference
    # probe is strong rather than sample-starved.
    grpA = rng.choice(n_features, size=12, replace=False)
    cntA = F[:, grpA].sum(1)
    kA = min(range(1, 13), key=lambda k: abs((cntA >= k).mean() - 0.5))
    yA = (cntA >= kA).astype(np.int8)

    # Label B: random Gaussian-weighted linear threshold on 10 bits, split at the
    # median. Continuous weights make the split exactly balanced (an integer-valued
    # score piles up on ties and skews the classes badly).
    grpB = rng.choice(n_features, size=10, replace=False)
    wB = rng.normal(size=10)
    sB = F[:, grpB] @ wB
    tB = float(np.median(sB))
    yB = (sB > tB).astype(np.int8)

    spec = dict(
        n_samples=int(n_samples), n_features=int(n_features),
        active_frac=float(active_frac), seed=int(seed),
        label_A=dict(kind="count of bits in a random 12-bit group >= k",
                     group=[int(x) for x in grpA], k=int(kA),
                     positive_rate=round(float(yA.mean()), 4)),
        label_B=dict(kind="random Gaussian-weighted linear threshold on 10 bits",
                     group=[int(x) for x in grpB], weights=[float(x) for x in wB],
                     threshold=tB, positive_rate=round(float(yB.mean()), 4)),
        mean_bits_active=round(float(F.sum(1).mean()), 2),
    )
    return F, {"A": yA, "B": yB}, spec


def glomerular_ceiling(fm, graph, F):
    """Active driven ORNs per (ORN type, side) channel. No simulation."""
    lab = np.array(
        [f"{graph.type[i]}|{graph.side[i]}" for i in fm.groups.ravel()], dtype=object
    )
    chans, inv = np.unique(lab, return_inverse=True)
    P = np.zeros((fm.n_features, len(chans)), np.float32)
    inv = inv.reshape(fm.groups.shape)
    for f in range(fm.n_features):
        np.add.at(P[f], inv[f], 1.0)
    return F.astype(np.float32) @ P, [str(c) for c in chans]


def simulate(brain, fm, F, window_ms, pops, drive_mv, persistent=False, label=""):
    mats = {k: np.zeros((len(F), len(v)), np.int32) for k, v in pops.items()}
    t0 = time.perf_counter()
    brain.warmup()
    if persistent:
        brain.reset()
    for i, f in enumerate(F):
        d = fm.drive(f, drive_mv=drive_mv)
        if not persistent:
            brain.reset()
        counts, _ = brain.step(d, window_ms)
        for k, idx in pops.items():
            mats[k][i] = counts[idx]
        if label and (i + 1) % 200 == 0:
            print(f"    [{label}] {i+1}/{len(F)}  {time.perf_counter()-t0:.0f}s", flush=True)
    return mats, time.perf_counter() - t0


def probe(X, y, seed=0, c_grid=None):
    """5-fold CV accuracy of a linear probe. Zero-variance columns dropped.

    ``c_grid`` defaults to the module-level ``C_GRID``; a caller running hundreds
    of probes can pass a shorter grid.
    """
    X = np.asarray(X, np.float64)
    keep = X.std(0) > 0
    if keep.sum() == 0:
        return dict(best_acc=None, best_C=None, n_used_features=0, per_C={},
                    n_total_features=int(X.shape[1]), note="no varying features")
    Xk = X[:, keep]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    per_c = {}
    for c in (C_GRID if c_grid is None else c_grid):
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=5000, solver="lbfgs"),
        )
        s = cross_val_score(pipe, Xk, y, cv=cv, scoring="accuracy", n_jobs=-1)
        per_c[str(c)] = dict(mean=round(float(s.mean()), 4), std=round(float(s.std()), 4))
    best = max(per_c, key=lambda k: per_c[k]["mean"])
    return dict(
        best_acc=per_c[best]["mean"], best_std=per_c[best]["std"], best_C=float(best),
        n_used_features=int(keep.sum()), n_total_features=int(X.shape[1]), per_C=per_c,
    )


def readout_stats(M):
    fired = M > 0
    return dict(
        population=int(M.shape[1]),
        ever_firing=int(fired.any(0).sum()),
        fraction_ever_firing=round(float(fired.any(0).mean()), 4),
        nonzero_variance=int((M.std(0) > 0).sum()),
        fraction_nonzero_variance=round(float((M.std(0) > 0).mean()), 4),
        mean_firing_per_sample=round(float(fired.sum(1).mean()), 1),
        mean_total_spikes_per_sample=round(float(M.sum(1).mean()), 1),
    )


POPS = ["ORN", "ALPN", "KC", "MBON", "CX", "DN"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--n-features", type=int, default=180)
    ap.add_argument("--neurons-per-feature", type=int, default=10)
    ap.add_argument("--active-frac", type=float, default=0.20)
    ap.add_argument("--window", type=float, default=50.0)
    ap.add_argument("--drive-mv", type=float, default=30.0)
    ap.add_argument("--v-jitter", type=float, default=6.9)
    ap.add_argument("--grouping", default="random", choices=["random", "by_type"])
    ap.add_argument("--persistent", action="store_true",
                    help="never reset the brain between samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--save-counts", default=None,
                    help="npz path to dump the raw count matrices for re-probing")
    a = ap.parse_args()

    g = C.load(a.threshold)
    pops = {"ORN": g.orn_indices(), "ALPN": g.alpn_indices(), "KC": g.kc_indices(),
            "MBON": g.mbon_indices(), "CX": g.cx_indices(), "DN": g.dn_indices()}
    fm = FeatureMap.for_graph(g, a.n_features, a.neurons_per_feature, seed=a.seed,
                              grouping=a.grouping, drive_mv=a.drive_mv)
    F, ys, spec = make_dataset(a.n_samples, a.n_features, a.active_frac, a.seed + 7)
    M_ceil, chans = glomerular_ceiling(fm, g, F)
    print(f"populations: " + "  ".join(f"{k}={len(v)}" for k, v in pops.items()))
    print(f"glomerular channels: {len(chans)}")
    print("feature map:", fm.describe(), flush=True)

    real = Brain(g, seed=1, v_jitter=a.v_jitter)
    shuf = real.shuffled(seed=1234)

    print(f"\nsimulating real wiring (window {a.window} ms, drive {a.drive_mv} mV, "
          f"v_jitter {a.v_jitter} mV, persistent={a.persistent}) ...", flush=True)
    M_real, t_real = simulate(real, fm, F, a.window, pops, a.drive_mv, a.persistent, "real")
    print(f"  real: {t_real:.0f}s", flush=True)
    print("simulating shuffled wiring ...", flush=True)
    M_shuf, t_shuf = simulate(shuf, fm, F, a.window, pops, a.drive_mv, a.persistent, "shuf")
    print(f"  shuffled: {t_shuf:.0f}s", flush=True)

    combos = {"MBON+CX": ("MBON", "CX"), "DN+MBON+CX": ("DN", "MBON", "CX")}

    results = {}
    rng = np.random.default_rng(a.seed + 999)
    for lab, y in ys.items():
        chance = round(float(max(y.mean(), 1 - y.mean())), 4)
        conds = {}
        conds["raw_bits"] = probe(F.astype(np.float64), y, a.seed)
        conds["glomerular_ceiling"] = probe(np.log1p(M_ceil), y, a.seed)
        print(f"\nlabel {lab}  (majority-class rate {chance})")
        print(f"  {'raw_bits':<26} acc {conds['raw_bits']['best_acc']}")
        print(f"  {'glomerular_ceiling':<26} acc {conds['glomerular_ceiling']['best_acc']}"
              f"  ({len(chans)} channels)", flush=True)
        for cond, M in (("real", M_real), ("shuffled", M_shuf)):
            for rname in POPS:
                X = np.log1p(M[rname].astype(np.float64))
                r = probe(X, y, a.seed)
                conds[f"{cond}:{rname}"] = r
                print(f"  {cond+':'+rname:<26} acc {r['best_acc']} "
                      f"({r['n_used_features']}/{r['n_total_features']} cols)", flush=True)
            for cname, parts in combos.items():
                X = np.log1p(np.hstack([M[p] for p in parts]).astype(np.float64))
                r = probe(X, y, a.seed)
                conds[f"{cond}:{cname}"] = r
                print(f"  {cond+':'+cname:<26} acc {r['best_acc']} "
                      f"({r['n_used_features']}/{r['n_total_features']} cols)", flush=True)
        # Honest chance level: same pipeline, same best-of-C-grid, shuffled labels.
        yp = rng.permutation(y)
        conds["permuted_label:real:DN"] = probe(
            np.log1p(M_real["DN"].astype(np.float64)), yp, a.seed)
        conds["permuted_label:raw_bits"] = probe(F.astype(np.float64), yp, a.seed)
        print(f"  {'PERMUTED real:DN':<26} acc {conds['permuted_label:real:DN']['best_acc']}"
              f"   {'PERMUTED raw':<10} acc {conds['permuted_label:raw_bits']['best_acc']}")
        results[lab] = {"majority_class_rate": chance, "conditions": conds}

    out = dict(
        tag=a.tag,
        settings=dict(
            threshold=a.threshold, window_ms=a.window, drive_mv=a.drive_mv,
            v_jitter_mv=a.v_jitter, persistent=a.persistent, grouping=a.grouping,
            seed=a.seed, dt_ms=real.dt, shuffle_seed=shuf.shuffle_seed,
            C_grid=C_GRID, cv="StratifiedKFold(5, shuffle=True, random_state=seed)",
            readout_transform="log1p(spike_counts) -> StandardScaler -> LogisticRegression(L2)",
            feature_map=fm.describe(), n_glomerular_channels=len(chans),
            machine=f"{platform.machine()} python {platform.python_version()}",
            sim_seconds=dict(real=round(t_real, 1), shuffled=round(t_shuf, 1)),
        ),
        dataset=spec,
        graph=dict(g.meta, neurons=g.n, edges_csr=g.m),
        populations={k: len(v) for k, v in pops.items()},
        glomerular_channels=chans,
        firing=dict(
            real={k: readout_stats(v) for k, v in M_real.items()},
            shuffled={k: readout_stats(v) for k, v in M_shuf.items()},
        ),
        results=results,
    )
    if a.save_counts:
        sc = Path(a.save_counts)
        sc.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sc, F=F, yA=ys["A"], yB=ys["B"], ceil=M_ceil,
                            **{f"real_{k}": v for k, v in M_real.items()},
                            **{f"shuf_{k}": v for k, v in M_shuf.items()})
        print("wrote counts", sc)

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("\nfiring (real):")
    for k in POPS:
        s = out["firing"]["real"][k]
        print(f"  {k:<5} {s['ever_firing']:>5}/{s['population']:<5} ever fire, "
              f"{s['nonzero_variance']:>5} varying, "
              f"{s['mean_total_spikes_per_sample']:>8} spikes/sample")
    print("wrote", p)


if __name__ == "__main__":
    main()
