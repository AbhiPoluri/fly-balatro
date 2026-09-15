"""Rescue check: drive the same binary features into the VISUAL pathway instead of
the ORNs, and ask again whether input identity is linearly decodable from the
descending neurons.

Everything except the drive pool is held identical to
`scripts/brain_probe.py --tag primary_w50_d30`: the same dataset
(`make_dataset(600, 180, 0.20, seed+7)`, so `raw_bits` accuracy must reproduce
exactly), the same 180 features x 10 disjoint neurons, 20% active, 50 ms window,
30 mV tonic drive, stateless reset per sample, the same brain seed (1), the same
shuffled-wiring control (seed 1234), the same log1p -> StandardScaler ->
LogisticRegression(L2) probe over the same C grid with StratifiedKFold(5,
shuffle=True, random_state=seed), and the same permuted-label baseline.

Three drive pools are run, only the first of which is the pool asked for:

  photoreceptor      superclass=="ol_sensory" & class=="visual"  (6,091)
                     THE SPEC'D POOL. Every photoreceptor in MaleCNS is
                     histaminergic, and `connectome.SIGN` maps histamine to -1,
                     so the whole pool is inhibitory. This LIF has no
                     spontaneous activity, so an inhibitory input layer has
                     nothing to modulate: the rest of the brain emits exactly
                     zero spikes at any window/drive (see
                     `propagation_diagnostics`). Every downstream readout is
                     therefore degenerate, for a reason that has nothing to do
                     with DN coding.
  lamina_L2L3        superclass=="ol_intrinsic" & type in {"L2","L3"}  (3,551)
                     DEVIATION FROM SPEC. The cholinergic (sign +1) first-order
                     targets of R1-R6: the first excitatory stage of the same
                     pathway. Activity spreads through the medulla but dies
                     before the lobula projection neurons at this drive density.
  optic_excitatory   superclass=="ol_intrinsic" & sign>0  (57,734)
                     DEVIATION FROM SPEC. Broad excitatory optic-lobe drive, not
                     a sensory-layer injection: the driven cells sit at mixed
                     depths, so some have short paths to the central brain. This
                     is the only visual pool that makes DNs fire at all at the
                     spec'd 180x10 / 20% density, i.e. the only one under which
                     the identity question is even answerable.

Readouts, all spike counts over the window:

  INPUT   the 1,800 driven neurons themselves (sanity: should be near raw_bits)
  LC      superclass=="visual_projection" & type startswith "LC"  (4,164), the
          intermediate lobula-columnar stage
  DN      superclass=="descending_neuron"  (1,314), the readout under test

Also run: the "total active feature bits > median" intensity control on real:DN,
and, only if real:DN identity accuracy beats its permuted-label baseline by
>= 0.06, one rerun at a 100 ms window.
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
from scripts.brain_probe import C_GRID, glomerular_ceiling, make_dataset, readout_stats, simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "brain_probe_visual.json"

# Identical to scripts.brain_probe.probe except n_jobs=1: another job owns the
# other cores, and joblib would fan out to every one of them. Deterministic CV,
# so the numbers are unaffected.
def probe(X, y, seed=0):
    """5-fold CV accuracy of a linear probe. Zero-variance columns dropped."""
    X = np.asarray(X, np.float64)
    keep = X.std(0) > 0
    if keep.sum() == 0:
        return dict(best_acc=None, best_C=None, n_used_features=0, per_C={},
                    n_total_features=int(X.shape[1]), note="no varying features")
    Xk = X[:, keep]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    per_c = {}
    for c in C_GRID:
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=5000, solver="lbfgs"),
        )
        s = cross_val_score(pipe, Xk, y, cv=cv, scoring="accuracy", n_jobs=1)
        per_c[str(c)] = dict(mean=round(float(s.mean()), 4), std=round(float(s.std()), 4))
    best = max(per_c, key=lambda k: per_c[k]["mean"])
    return dict(
        best_acc=per_c[best]["mean"], best_std=per_c[best]["std"], best_C=float(best),
        n_used_features=int(keep.sum()), n_total_features=int(X.shape[1]), per_C=per_c,
    )


def pool_specs(g):
    """name -> (selector string, neuron indices). Exact annotation strings only."""
    sc = g.superclass.astype(str)
    cl = g.cls.astype(str)
    ty = g.type.astype(str)
    sel = lambda m: np.flatnonzero(m).astype(np.int32)
    return {
        "photoreceptor": (
            'superclass=="ol_sensory" & class=="visual"  (identical to '
            'superclass=="ol_sensory" & type startswith "R"; the 7 HBeyelet cells '
            'in the 6,098-cell ol_sensory superclass are the only ones excluded)',
            sel((sc == "ol_sensory") & (cl == "visual")),
            False,
        ),
        "lamina_L2L3": (
            'superclass=="ol_intrinsic" & type in {"L2","L3"}',
            sel((sc == "ol_intrinsic") & np.isin(ty, ["L2", "L3"])),
            True,
        ),
        "optic_excitatory": (
            'superclass=="ol_intrinsic" & sign>0  (sign is +1 for '
            "acetylcholine/dopamine/serotonin/octopamine per connectome.SIGN)",
            sel((sc == "ol_intrinsic") & (g.sign > 0)),
            True,
        ),
    }


def readout_specs(g):
    sc = g.superclass.astype(str)
    ty = g.type.astype(str)
    return {
        "LC": ('superclass=="visual_projection" & type startswith "LC"',
               np.flatnonzero((sc == "visual_projection") & np.char.startswith(ty, "LC")).astype(np.int32)),
        "DN": ('superclass=="descending_neuron"',
               g.dn_indices()),
    }


def diagnostic_groups(g):
    sc = g.superclass.astype(str)
    d = {k: np.flatnonzero(sc == k).astype(np.int32) for k in
         ("ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal",
          "cb_sensory", "cb_intrinsic", "descending_neuron")}
    d["CX"] = g.cx_indices()
    d["MBON"] = g.mbon_indices()
    return d


def propagation_diagnostics(brain, fm, f_one, groups):
    """Where does the drive die? One sample, a few window/drive settings, plus an
    all-features-on saturation case. Cheap, and it rules out 'the window was too
    short' / 'the drive was too weak' as the reason a readout is silent."""
    out = []
    allon = np.ones(fm.n_features, np.int8)
    cases = [(50.0, 30.0, f_one, "spec sample 0"),
             (100.0, 30.0, f_one, "spec sample 0"),
             (50.0, 60.0, f_one, "spec sample 0"),
             (200.0, 120.0, f_one, "spec sample 0"),
             (50.0, 30.0, allon, "all 180 features on"),
             (200.0, 120.0, allon, "all 180 features on")]
    for w, d, f, what in cases:
        brain.reset()
        counts, _ = brain.step(fm.drive(f, d), w)
        out.append(dict(
            window_ms=w, drive_mv=d, input=what,
            n_driven=int(f.sum()) * fm.neurons_per_feature,
            total_spikes=int(counts.sum()),
            firing={k: int((counts[v] > 0).sum()) for k, v in groups.items()},
        ))
    brain.reset()
    return out


def run_pool(name, selector, pool, deviation, g, F, ys, a, groups, readouts, window):
    fm = FeatureMap(a.n_features, pool, a.neurons_per_feature, seed=a.seed, n_total=g.n,
                    grouping="random", types=g.type, fallback_indices=None,
                    drive_mv=a.drive_mv)
    driven = fm.all_indices()
    pops = {"INPUT": driven, **{k: v for k, (_, v) in readouts.items()}}
    M_ceil, chans = glomerular_ceiling(fm, g, F)

    print(f"\n{'='*72}\npool {name}: {len(pool)} cells, selector: {selector}")
    print(f"  feature map: {fm.describe()}")
    print(f"  readouts: " + "  ".join(f"{k}={len(v)}" for k, v in pops.items()))
    print(f"  (type|side) channels over the driven cells: {len(chans)}", flush=True)

    real = Brain(g, seed=1, v_jitter=a.v_jitter)
    shuf = real.shuffled(seed=1234)
    real.warmup()
    diag = propagation_diagnostics(real, fm, F[0], groups)
    print("  propagation diagnostics (real wiring):")
    for d in diag:
        print(f"    w={d['window_ms']:>5.0f} drive={d['drive_mv']:>5.0f} "
              f"driven={d['n_driven']:>5} spikes={d['total_spikes']:>7} "
              + " ".join(f"{k}:{v}" for k, v in d["firing"].items()), flush=True)

    print(f"  simulating real wiring (window {window} ms, drive {a.drive_mv} mV) ...",
          flush=True)
    M_real, t_real = simulate(real, fm, F, window, pops, a.drive_mv, False, f"{name}/real")
    print(f"    real: {t_real:.0f}s", flush=True)
    M_shuf, t_shuf = simulate(shuf, fm, F, window, pops, a.drive_mv, False, f"{name}/shuf")
    print(f"    shuffled: {t_shuf:.0f}s", flush=True)

    results = {}
    rng = np.random.default_rng(a.seed + 999)
    for lab, y in ys.items():
        chance = round(float(max(y.mean(), 1 - y.mean())), 4)
        conds = {}
        conds["raw_bits"] = probe(F.astype(np.float64), y, a.seed)
        conds["channel_ceiling"] = probe(np.log1p(M_ceil), y, a.seed)
        print(f"\n  label {lab}  (majority-class rate {chance})")
        print(f"    {'raw_bits':<24} acc {conds['raw_bits']['best_acc']}")
        print(f"    {'channel_ceiling':<24} acc {conds['channel_ceiling']['best_acc']}"
              f"  ({len(chans)} channels)", flush=True)
        for cond, M in (("real", M_real), ("shuffled", M_shuf)):
            for rname in pops:
                r = probe(np.log1p(M[rname].astype(np.float64)), y, a.seed)
                conds[f"{cond}:{rname}"] = r
                print(f"    {cond+':'+rname:<24} acc {r['best_acc']} "
                      f"({r['n_used_features']}/{r['n_total_features']} cols)", flush=True)
        yp = rng.permutation(y)
        conds["permuted_label:real:DN"] = probe(np.log1p(M_real["DN"].astype(np.float64)),
                                               yp, a.seed)
        conds["permuted_label:raw_bits"] = probe(F.astype(np.float64), yp, a.seed)
        print(f"    {'PERMUTED real:DN':<24} acc {conds['permuted_label:real:DN']['best_acc']}"
              f"   PERMUTED raw acc {conds['permuted_label:raw_bits']['best_acc']}", flush=True)
        results[lab] = {"majority_class_rate": chance, "conditions": conds}

    # Intensity control: is the DN code just "how much total input is on"?
    tot = F.sum(1)
    y_int = (tot > np.median(tot)).astype(np.int8)
    rng2 = np.random.default_rng(a.seed + 4321)
    intensity = dict(
        label="total active feature bits > median",
        positive_rate=round(float(y_int.mean()), 4),
        pearson_r_label_A_vs_total_bits=round(float(np.corrcoef(ys["A"], tot)[0, 1]), 4),
        pearson_r_label_B_vs_total_bits=round(float(np.corrcoef(ys["B"], tot)[0, 1]), 4),
        accuracy={},
    )
    for rname in pops:
        r = probe(np.log1p(M_real[rname].astype(np.float64)), y_int, a.seed)
        intensity["accuracy"][f"real:{rname}"] = r["best_acc"]
    r = probe(np.log1p(M_real["DN"].astype(np.float64)), rng2.permutation(y_int), a.seed)
    intensity["accuracy"]["PERMUTED real:DN"] = r["best_acc"]
    print("\n  intensity control (total active bits > median):",
          json.dumps(intensity["accuracy"]), flush=True)

    block = dict(
        pool=name,
        selector=selector,
        deviation_from_spec=deviation,
        pool_size=int(len(pool)),
        window_ms=window,
        feature_map=fm.describe(),
        readout_selectors={"INPUT": "the 1,800 driven neurons of this pool",
                           **{k: s for k, (s, _) in readouts.items()}},
        populations={k: int(len(v)) for k, v in pops.items()},
        n_channels=len(chans),
        sim_seconds=dict(real=round(t_real, 1), shuffled=round(t_shuf, 1)),
        propagation_diagnostics=diag,
        firing=dict(real={k: readout_stats(v) for k, v in M_real.items()},
                    shuffled={k: readout_stats(v) for k, v in M_shuf.items()}),
        results=results,
        intensity_control=intensity,
    )
    return block, M_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--n-features", type=int, default=180)
    ap.add_argument("--neurons-per-feature", type=int, default=10)
    ap.add_argument("--active-frac", type=float, default=0.20)
    ap.add_argument("--window", type=float, default=50.0)
    ap.add_argument("--window-rescue", type=float, default=100.0)
    ap.add_argument("--drive-mv", type=float, default=30.0)
    ap.add_argument("--v-jitter", type=float, default=6.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pools", default="photoreceptor,lamina_L2L3,optic_excitatory")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    g = C.load(a.threshold)
    specs = pool_specs(g)
    readouts = readout_specs(g)
    groups = diagnostic_groups(g)
    F, ys, spec = make_dataset(a.n_samples, a.n_features, a.active_frac, a.seed + 7)
    print(f"graph: {g.n} neurons, {g.m} edges (threshold {a.threshold})")
    print("pools available: " + "  ".join(f"{k}={len(v[1])}" for k, v in specs.items()))
    print("readouts: " + "  ".join(f"{k}={len(v[1])}" for k, v in readouts.items()))
    print(f"dataset: {a.n_samples} samples, mean bits active {spec['mean_bits_active']}",
          flush=True)

    runs = {}
    t0 = time.perf_counter()
    for name in a.pools.split(","):
        selector, pool, deviation = specs[name]
        block, _ = run_pool(name, selector, pool, deviation, g, F, ys, a, groups,
                            readouts, a.window)
        runs[f"{name}_w{int(a.window)}"] = block

        # Conditional 100 ms rerun: only if the DNs beat chance here.
        lift = []
        for lab in ys:
            c = block["results"][lab]["conditions"]
            acc, perm = c["real:DN"]["best_acc"], c["permuted_label:real:DN"]["best_acc"]
            if acc is not None and perm is not None:
                lift.append(acc - perm)
        if lift and max(lift) >= 0.06:
            print(f"\n  real:DN beats permuted by {max(lift):.3f} for pool {name}; "
                  f"rerunning at {a.window_rescue} ms", flush=True)
            block2, _ = run_pool(name, selector, pool, deviation, g, F, ys, a, groups,
                                 readouts, a.window_rescue)
            block2["note"] = (f"triggered rerun: real:DN beat its permuted-label baseline "
                              f"by {max(lift):.3f} at {a.window} ms")
            runs[f"{name}_w{int(a.window_rescue)}"] = block2
        else:
            best = f"{max(lift):.3f}" if lift else "n/a (degenerate readout)"
            print(f"\n  no 100 ms rerun for {name}: best real:DN lift over permuted = {best}"
                  f" (< 0.06)", flush=True)

    table = {}
    for key, b in runs.items():
        for lab, r in b["results"].items():
            c = r["conditions"]
            table[f"{key}|label_{lab}"] = {
                "majority_class_rate": r["majority_class_rate"],
                "permuted_real_DN": c["permuted_label:real:DN"]["best_acc"],
                "permuted_raw_bits": c["permuted_label:raw_bits"]["best_acc"],
                "raw_bits": c["raw_bits"]["best_acc"],
                "channel_ceiling": c["channel_ceiling"]["best_acc"],
                "real:INPUT": c["real:INPUT"]["best_acc"],
                "shuffled:INPUT": c["shuffled:INPUT"]["best_acc"],
                "real:LC": c["real:LC"]["best_acc"],
                "shuffled:LC": c["shuffled:LC"]["best_acc"],
                "real:DN": c["real:DN"]["best_acc"],
                "shuffled:DN": c["shuffled:DN"]["best_acc"],
                "fraction_DN_ever_firing": b["firing"]["real"]["DN"]["fraction_ever_firing"],
            }

    out = dict(
        note=(
            "Visual rescue check for scripts/brain_probe.py. The spec'd pool "
            '(superclass=="ol_sensory") is entirely histaminergic, and '
            "connectome.SIGN maps histamine to -1, so all 6,091 photoreceptors are "
            "inhibitory. This LIF has no spontaneous activity, so an inhibitory input "
            "layer has nothing to modulate and the rest of the brain emits exactly zero "
            "spikes at every window and drive tried (see propagation_diagnostics). That "
            "is a property of the model's sign convention plus its silent baseline, not "
            "evidence about DN coding, so two excitatory visual pools are run alongside "
            "it and flagged deviation_from_spec=true."
        ),
        se_note=(
            "At n=600 the CV standard error is ~0.020 and taking the best of a 6-value C "
            "grid inflates accuracy by a further ~0.03-0.05; the permuted-label probe in "
            "each run measures that inflation directly. Treat a readout as carrying "
            "information only if it beats its permuted-label baseline by >= 0.06."
        ),
        conditions_explained={
            "raw_bits": "logistic probe straight on the binary feature vector, no brain",
            "channel_ceiling": "active driven cells per (type|side) channel of the driven "
                               "pool; no simulation. Analogue of brain_probe's "
                               "glomerular_ceiling.",
            "real:INPUT": "spike counts of the 1,800 driven visual neurons (sanity)",
            "real:LC": "lobula-columnar visual projection neurons, intermediate stage",
            "real:DN": "descending neurons, the readout under test",
            "shuffled:*": "edge targets globally permuted, in-degree preserved. Not a "
                          "null: permuting post hands the DNs near-direct input, i.e. a "
                          "sparse random projection with no recurrence.",
            "permuted_label:*": "same pipeline and same best-of-C-grid on shuffled "
                                "labels; the honest chance level.",
        },
        reproduction_check=(
            "Dataset and CV are identical to outputs/probe_primary_w50_d30.json, so "
            "raw_bits must be 0.9 (label A) and 0.8533 (label B). probe() here is "
            "brain_probe.probe with n_jobs=1 instead of -1 (one process, as required); "
            "the CV is deterministic so results are unaffected."
        ),
        settings=dict(
            threshold=a.threshold, window_ms=a.window, window_rescue_ms=a.window_rescue,
            drive_mv=a.drive_mv, v_jitter_mv=a.v_jitter, persistent=False,
            grouping="random", seed=a.seed, dt_ms=0.1, shuffle_seed=1234, C_grid=C_GRID,
            cv="StratifiedKFold(5, shuffle=True, random_state=seed)",
            readout_transform="log1p(spike_counts) -> StandardScaler -> LogisticRegression(L2)",
            n_features=a.n_features, neurons_per_feature=a.neurons_per_feature,
            active_frac=a.active_frac, n_samples=a.n_samples,
            machine=f"{platform.machine()} python {platform.python_version()}",
            total_seconds=round(time.perf_counter() - t0, 1),
        ),
        dataset=spec,
        graph=dict(g.meta, neurons=g.n, edges_csr=g.m),
        pool_sizes={k: int(len(v[1])) for k, v in specs.items()},
        pool_selectors={k: v[0] for k, v in specs.items()},
        summary_table=table,
        runs=runs,
    )
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 72)
    for k, v in table.items():
        print(f"{k}: " + "  ".join(f"{kk}={vv}" for kk, vv in v.items()))
    print("wrote", p)


if __name__ == "__main__":
    main()
