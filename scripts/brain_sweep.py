"""Config sweep to find a regime where the real wiring carries more input
information to the DNs than the shuffled control.

Small (300-sample) runs over drive strength, window length, ORNs per feature, and
Poisson vs tonic input. Feeds the settings chosen for scripts/brain_probe.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybalatro import connectome as C
from flybalatro.brain import Brain
from flybalatro.encode import FeatureMap
from scripts.brain_probe import make_dataset, probe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "brain_sweep.json"

CONFIGS = [
    dict(name="w20_d30_n10", window=20.0, drive=30.0, npf=10),
    dict(name="w50_d30_n10", window=50.0, drive=30.0, npf=10),
    dict(name="w50_d12_n10", window=50.0, drive=12.0, npf=10),
    dict(name="w50_d9_n10", window=50.0, drive=9.0, npf=10),
    dict(name="w100_d12_n10", window=100.0, drive=12.0, npf=10),
    dict(name="w50_d30_n4", window=50.0, drive=30.0, npf=4),
    dict(name="w50_pois50_n10", window=50.0, drive=0.0, npf=10, poisson_hz=50.0),
    dict(name="w50_pois150_n10", window=50.0, drive=0.0, npf=10, poisson_hz=150.0),
]


def run_one(g, pops, F, ys, cfg, n_features, seed):
    fm = FeatureMap.for_graph(g, n_features, cfg["npf"], seed=seed,
                              drive_mv=max(cfg["drive"], 1.0))
    real = Brain(g, seed=1, kick_mv=0.275 * 250 if cfg.get("poisson_hz") else 0.0)
    shuf = real.shuffled(seed=1234)
    out = {"config": cfg, "conditions": {}}
    for cond, brain in (("real", real), ("shuffled", shuf)):
        brain.warmup()
        mats = {k: np.zeros((len(F), len(v)), np.int32) for k, v in pops.items()}
        t0 = time.perf_counter()
        for i, f in enumerate(F):
            brain.reset()
            if cfg.get("poisson_hz"):
                brain.kick_prob.fill(0.0)
                on = np.flatnonzero(f.astype(bool))
                if len(on):
                    brain.kick_prob[fm.groups[on].ravel()] = (
                        cfg["poisson_hz"] * brain.dt / 1000.0
                    )
                d = np.zeros(g.n, np.float32)
            else:
                d = fm.drive(f, drive_mv=cfg["drive"])
            counts, _ = brain.step(d, cfg["window"])
            for k, idx in pops.items():
                mats[k][i] = counts[idx]
        sim_s = time.perf_counter() - t0
        entry = {"sim_seconds": round(sim_s, 1)}
        for rname, M in (("DN", mats["DN"]), ("MBON", mats["MBON"]), ("CX", mats["CX"])):
            entry[f"{rname}_frac_varying"] = round(float((M.std(0) > 0).mean()), 4)
            entry[f"{rname}_spikes_per_sample"] = round(float(M.sum(1).mean()), 1)
            for lab, y in ys.items():
                r = probe(np.log1p(M.astype(np.float64)), y, seed)
                entry[f"{rname}_acc_{lab}"] = r["best_acc"]
        out["conditions"][cond] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--n-features", type=int, default=180)
    ap.add_argument("--active-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    g = C.load(5)
    pops = {"DN": g.dn_indices(), "MBON": g.mbon_indices(), "CX": g.cx_indices()}
    F, ys, spec = make_dataset(a.n_samples, a.n_features, a.active_frac, a.seed + 7)
    chance = {k: round(float(max(v.mean(), 1 - v.mean())), 4) for k, v in ys.items()}
    raw = {k: probe(F.astype(np.float64), v, a.seed)["best_acc"] for k, v in ys.items()}
    print(f"chance {chance}   raw-bits {raw}", flush=True)

    results = []
    for cfg in CONFIGS:
        if a.only and cfg["name"] not in a.only:
            continue
        print(f"\n=== {cfg['name']} ===", flush=True)
        r = run_one(g, pops, F, ys, cfg, a.n_features, a.seed)
        results.append(r)
        for cond, e in r["conditions"].items():
            print(f"  {cond:<9} DN A={e['DN_acc_A']} B={e['DN_acc_B']} "
                  f"(var {e['DN_frac_varying']}, {e['DN_spikes_per_sample']} sp) | "
                  f"MBON A={e['MBON_acc_A']} B={e['MBON_acc_B']} | "
                  f"CX A={e['CX_acc_A']} B={e['CX_acc_B']} | {e['sim_seconds']}s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        dict(dataset=spec, chance=chance, raw_bits=raw,
             n_samples=a.n_samples, results=results), indent=2))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
