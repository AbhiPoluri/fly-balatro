"""Why do the mushroom-body knobs plateau? The decomposition behind REPORT.md.

Three measurements, all on the untuned model unless a tuning is passed:

  1. Where a Kenyon cell's input comes from: total signed conductance into the
     KC population, broken down by presynaptic class. (ALPN is the biggest
     source, but KC -> KC recurrence is second.)

  2. Between-KC versus within-KC variance of the drive each KC receives.
     `between` is the SD across KCs of each KC's mean total delivered
     conductance (input-independent: it is set by which and how many synapses
     that KC has). `within` is the mean over KCs of the SD across inputs
     (input-dependent: the part that could carry odour identity). A large
     between/within ratio means threshold crossing is decided by wiring, not by
     the stimulus, and no knob that rescales a pathway or shifts a threshold
     uniformly can change which cells win, only how many.

  3. How input-specific the antennal-lobe code is in the first place: the CV of
     the *total* ALPN spike count across inputs, and the per-cell CV.

  4. When APL's first spike arrives relative to the first KC spike, which bounds
     what APL feedback can do to the opening volley.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybalatro import connectome as C
from flybalatro.brain import Brain
from flybalatro.tuning import Tuning
from scripts.mb_sparsity import OUT_DIR, build_inputs, populations


def kc_input_sources(graph) -> dict:
    """Total signed / absolute conductance into the KC population, by source class."""
    kc = graph.kc_indices()
    kc_mask = np.zeros(graph.n, bool)
    kc_mask[kc] = True
    pre_of_edge = np.repeat(np.arange(graph.n, dtype=np.int32), np.diff(graph.ptr))
    sel = kc_mask[graph.post]
    pre = pre_of_edge[sel]
    w = graph.weight[sel]
    cls = graph.cls.astype(str)
    typ = graph.type.astype(str)
    name = np.where(cls[pre] != "", cls[pre],
                    np.where(typ[pre] != "", "type:" + typ[pre], "unlabeled"))
    out = {}
    for nm in np.unique(name):
        m = name == nm
        out[str(nm)] = dict(
            edges=int(m.sum()),
            signed_mv_total=round(float(w[m].sum()), 1),
            signed_mv_per_kc=round(float(w[m].sum() / len(kc)), 3),
        )
    return dict(n_kc=int(len(kc)),
                by_source=dict(sorted(out.items(),
                                      key=lambda t: -abs(t[1]["signed_mv_total"]))))


def delivered_matrix(brain: Brain, fm, F, kc, kc_mask, kc_pos) -> np.ndarray:
    """(n_inputs, n_kc) total synaptic conductance delivered to each KC per input.

    Uses the brain's own (possibly tuned) weights and the spike counts it
    produced, so it measures the drive the kernel saw.
    """
    D = np.zeros((len(F), len(kc)), np.float64)
    brain.warmup()
    for i, f in enumerate(F):
        brain.reset()
        counts, _ = brain.step(fm.drive(f, drive_mv=fm.drive_mv), 50.0)
        for j in np.flatnonzero(counts > 0):
            a, b = int(brain.ptr[j]), int(brain.ptr[j + 1])
            p = brain.post[a:b]
            m = kc_mask[p]
            if m.any():
                np.add.at(D[i], kc_pos[p[m]], brain.weight[a:b][m] * counts[j])
    return D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--n-inputs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--brain-seed", type=int, default=1)
    ap.add_argument("--v-jitter", type=float, default=6.9)
    ap.add_argument("--settings", default='[{}, {"apl_scale": 2.0}]')
    ap.add_argument("--out", default=str(OUT_DIR / "diagnostics.json"))
    a = ap.parse_args()

    g = C.load(a.threshold)
    pops = populations(g)
    kc, alpn, apl = pops["KC"], pops["ALPN"], pops["APL"]
    kc_mask = np.zeros(g.n, bool)
    kc_mask[kc] = True
    kc_pos = -np.ones(g.n, np.int64)
    kc_pos[kc] = np.arange(len(kc))
    fm, F, _ys, spec = build_inputs(g, a.n_inputs, 180, 10, 0.20, 30.0, a.seed)

    out: dict = dict(
        n_inputs=a.n_inputs, window_ms=50.0, drive_mv=30.0,
        brain_seed=a.brain_seed, v_jitter_mv=a.v_jitter,
        dataset=dict(mean_bits_active=spec["mean_bits_active"],
                     n_features=spec["n_features"]),
        kc_input_sources=kc_input_sources(g),
        runs={},
    )

    # ALPN timing / selectivity and APL first-spike, on the untuned model
    b = Brain(g, seed=a.brain_seed, v_jitter=a.v_jitter)
    b.warmup()
    AL = np.zeros((len(F), len(alpn)), np.float64)
    apl_first, kc_first = [], []
    for i, f in enumerate(F):
        d = fm.drive(f, 30.0)
        b.reset()
        fa = fk = None
        for ms in range(50):
            b.step(d, 1.0, reset_counts=False)
            if fa is None and b.counts[apl].sum() > 0:
                fa = ms + 1
            if fk is None and b.counts[kc].sum() > 0:
                fk = ms + 1
        apl_first.append(fa)
        kc_first.append(fk)
        AL[i] = b.counts[alpn]
    per_input_total = AL.sum(1)
    out["alpn_code"] = dict(
        mean_count_per_cell=round(float(AL.mean()), 3),
        cv_of_total_across_inputs=round(
            float(per_input_total.std() / per_input_total.mean()), 4),
        mean_per_cell_cv_across_inputs=round(
            float(np.mean(AL.std(0) / np.maximum(AL.mean(0), 1e-9))), 4),
        note=("a near-zero CV of the total means every input drives the antennal "
              "lobe about equally hard: 36 of 180 bits active puts 360 ORNs into "
              "all ~54 glomeruli, so no ALPN is silent for any input"),
    )
    out["timing"] = dict(
        first_kc_spike_ms=dict(mean=round(float(np.mean(kc_first)), 2),
                               min=int(min(kc_first)), max=int(max(kc_first))),
        first_apl_spike_ms=dict(mean=round(float(np.mean(apl_first)), 2),
                                min=int(min(apl_first)), max=int(max(apl_first))),
        synaptic_delay_ms=1.8,
        note=("APL is driven by the Kenyon cells, so its inhibition cannot reach "
              "them before first_apl_spike + 1.8 ms delay. Everything the KCs do "
              "before that is APL-free whatever apl_scale is."),
    )

    for d_ in json.loads(a.settings):
        t = Tuning.from_dict(d_)
        t0 = time.perf_counter()
        brain = Brain(g, seed=a.brain_seed, v_jitter=a.v_jitter, tuning=t)
        D = delivered_matrix(brain, fm, F, kc, kc_mask, kc_pos)
        KC = np.zeros((len(F), len(kc)), np.int32)
        brain.warmup()
        for i, f in enumerate(F):
            brain.reset()
            counts, _ = brain.step(fm.drive(f, 30.0), 50.0)
            KC[i] = counts[kc]
        freq = (KC > 0).mean(0)
        between = float(D.mean(0).std())
        within = float(D.std(0).mean())
        out["runs"][t.label()] = dict(
            tuning=t.to_dict(),
            delivered_conductance_mean_mv=round(float(D.mean()), 2),
            between_kc_sd_mv=round(between, 2),
            within_kc_sd_across_inputs_mv=round(within, 2),
            between_over_within=round(between / within, 2),
            corr_firing_frequency_with_mean_input=round(
                float(np.corrcoef(freq, D.mean(0))[0, 1]), 3),
            kc_active_fraction=round(float((KC > 0).mean()), 4),
            always_on_fraction=round(float((freq > 0.5).mean()), 4),
            seconds=round(time.perf_counter() - t0, 1),
        )
        r = out["runs"][t.label()]
        print(f"{t.label():<14} delivered mean {r['delivered_conductance_mean_mv']:>8.1f} mV  "
              f"between {r['between_kc_sd_mv']:>7.2f}  within {r['within_kc_sd_across_inputs_mv']:>6.2f}  "
              f"ratio {r['between_over_within']:>5.2f}  "
              f"corr(freq, mean input) {r['corr_firing_frequency_with_mean_input']:.3f}",
              flush=True)

    print("\nKC input by source class (signed mV per KC):")
    for nm, v in list(out["kc_input_sources"]["by_source"].items())[:8]:
        print(f"  {nm:<24} {v['signed_mv_per_kc']:>8.2f}  ({v['edges']:,} edges)")
    print(f"\nfirst KC spike {out['timing']['first_kc_spike_ms']['mean']} ms, "
          f"first APL spike {out['timing']['first_apl_spike_ms']['mean']} ms")
    print(f"ALPN total-count CV across inputs "
          f"{out['alpn_code']['cv_of_total_across_inputs']}")

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
