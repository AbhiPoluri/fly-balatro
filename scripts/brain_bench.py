"""Wall-clock and firing statistics for the full brain-only graph vs a shuffled control.

Windows of 20 / 50 / 100 ms under ~30% of features active. The numba kernel is
compiled by an untimed warmup call first, so no JIT time enters the numbers.
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
from flybalatro.encode import FeatureMap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "brain_bench.json"


def stats_for(brain: Brain, fm: FeatureMap, pops: dict, windows, n_random: int,
              active_frac: float, seed: int, label: str) -> dict:
    rng = np.random.default_rng(seed)
    brain.warmup()
    res = {"windows": {}, "distinct_over_random_inputs": {}}

    for w in windows:
        per_call, spikes, dn_firing, n_active = [], [], [], []
        for rep in range(5):
            f = rng.random(fm.n_features) < active_frac
            d = fm.drive(f)
            brain.reset()
            counts, el = brain.step(d, w)
            per_call.append(el)
            spikes.append(int(counts.sum()))
            dn_firing.append(int((counts[pops["DN"]] > 0).sum()))
            n_active.append(brain.n_active)
        res["windows"][str(w)] = dict(
            wall_ms_mean=round(1000 * float(np.mean(per_call)), 2),
            wall_ms_min=round(1000 * float(np.min(per_call)), 2),
            wall_ms_max=round(1000 * float(np.max(per_call)), 2),
            realtime_factor=round(float(np.mean(per_call)) * 1000.0 / w, 3),
            total_spikes_mean=int(np.mean(spikes)),
            dns_with_ge1_spike_mean=round(float(np.mean(dn_firing)), 1),
            dns_with_ge1_spike_range=[int(min(dn_firing)), int(max(dn_firing))],
            kernel_active_neurons_mean=int(np.mean(n_active)),
        )
        print(f"  [{label}] {w:>5.0f} ms  {1000*np.mean(per_call):7.1f} ms wall "
              f"({np.mean(per_call)*1000/w:5.2f}x realtime)  "
              f"spikes {int(np.mean(spikes)):>9,}  DN>=1 {np.mean(dn_firing):6.1f}/"
              f"{len(pops['DN'])}  active {int(np.mean(n_active)):>7,}")

    # How much of each readout population is ever recruited across many inputs.
    w = float(windows[-1])
    ever = {k: np.zeros(len(v), bool) for k, v in pops.items()}
    per_trial = {k: [] for k in pops}
    t0 = time.perf_counter()
    for _ in range(n_random):
        f = rng.random(fm.n_features) < active_frac
        d = fm.drive(f)
        brain.reset()
        counts, _ = brain.step(d, w)
        for k, idx in pops.items():
            fired = counts[idx] > 0
            ever[k] |= fired
            per_trial[k].append(int(fired.sum()))
    res["distinct_over_random_inputs"] = dict(
        n_inputs=n_random,
        window_ms=w,
        seconds=round(time.perf_counter() - t0, 1),
        **{
            k: dict(
                population=len(pops[k]),
                distinct_ever_firing=int(ever[k].sum()),
                fraction_ever_firing=round(float(ever[k].mean()), 4),
                per_input_firing_mean=round(float(np.mean(per_trial[k])), 1),
            )
            for k in pops
        },
    )
    for k in pops:
        e = res["distinct_over_random_inputs"][k]
        print(f"  [{label}] {k:<5} {e['distinct_ever_firing']:>5}/{e['population']:<5} "
              f"distinct over {n_random} inputs  (mean {e['per_input_firing_mean']}/input)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--n-features", type=int, default=180)
    ap.add_argument("--neurons-per-feature", type=int, default=10)
    ap.add_argument("--active-frac", type=float, default=0.30)
    ap.add_argument("--windows", type=float, nargs="+", default=[20.0, 50.0, 100.0])
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--drive-mv", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    g = C.load(a.threshold, verbose=True)
    pops = {"DN": g.dn_indices(), "MBON": g.mbon_indices(), "CX": g.cx_indices(),
            "ORN": g.orn_indices()}
    fm = FeatureMap.for_graph(g, a.n_features, a.neurons_per_feature, seed=a.seed,
                              drive_mv=a.drive_mv)
    print("\nfeature map:", fm.describe())

    print("\n--- real wiring ---")
    real = Brain(g, seed=1)
    r_real = stats_for(real, fm, pops, a.windows, a.n_random, a.active_frac,
                       a.seed + 100, "real")
    print("\n--- shuffled wiring (targets permuted, in-degree preserved) ---")
    shuf = real.shuffled(seed=1234)
    r_shuf = stats_for(shuf, fm, pops, a.windows, a.n_random, a.active_frac,
                       a.seed + 100, "shuf")

    out = dict(
        settings=dict(
            threshold=a.threshold, windows=a.windows, n_random=a.n_random,
            active_frac=a.active_frac, drive_mv=a.drive_mv, seed=a.seed,
            feature_map=fm.describe(), dt_ms=real.dt, v_jitter_mv=real.v_jitter,
            shuffle_seed=shuf.shuffle_seed,
            machine=f"{platform.machine()} {platform.processor()} python {platform.python_version()}",
        ),
        graph=dict(g.meta, neurons=g.n, edges_csr=g.m),
        populations={k: len(v) for k, v in pops.items()},
        real=r_real,
        shuffled=r_shuf,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
