"""Is the Kenyon-cell code sparse? Measurement only: no plasticity, no learning.

For N random sparse binary inputs (the same generator ``scripts/brain_probe.py``
uses: 180 bits, 20% active, 10 ORNs per bit, 30 mV tonic, one 50 ms window from
reset) this records, per input, the set of Kenyon cells with at least one spike,
and reports:

    (i)   mean fraction of KCs active
    (ii)  mean pairwise Jaccard overlap of the active sets across inputs
          (reference: for independent Bernoulli-p sets, E[J] ~= p/(2-p))
    (iii) fraction of KCs that fire for more than half of all inputs ("always-on")
    (iv)  KC spikes per millisecond, averaged over inputs; the first ~10 ms are
          structurally APL-free, because APL is driven *by* the KCs and its
          feedback cannot arrive before the first KC volley has already happened
    (v)   MBON and ALPN mean rates
    (vi)  the MN9 sugar check: spikes in the proboscis-extension motor neuron when
          the labellar gustatory receptor neurons are driven, versus no drive and
          versus a random olfactory input (specificity)

In vivo a KC responds to ~5-10% of odours. The untuned model is far denser, which
is the thing the knobs in ``flybalatro.tuning`` exist to fix. This script is also
the metric provider for ``scripts/mb_sweep.py``.
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
from flybalatro.tuning import Tuning, apl_indices
from scripts.brain_probe import make_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "mb"

# --- exact annotation strings -------------------------------------------------
#: Sugar / labellar gustatory receptor neurons. MaleCNS v1.0 carries no "sugar",
#: "Gr64" or "GRN" string for them (the only "GRN" types are leg taste-peg
#: neurons, claw_tpGRN / dorsal_tpGRN). The reference check in
#: vendor/fly-craftax/scripts/mn9_check.py selects them by type prefix "LB"
#: (labellar bristle), all of class "gustatory", superclass "cb_sensory".
GRN_TYPE_PREFIX = "LB"
GRN_CLASS = "gustatory"
#: Proboscis-extension motor neuron (Shiu et al. 2024): type "MN9", class is the
#: empty string, superclass "cb_motor", one per hemisphere.
MN9_TYPE = "MN9"


def populations(graph) -> dict:
    t = graph.type.astype(str)
    return dict(
        KC=graph.kc_indices(),
        MBON=graph.mbon_indices(),
        ALPN=graph.alpn_indices(),
        ORN=graph.orn_indices(),
        DN=graph.dn_indices(),
        APL=apl_indices(graph),
        GRN=np.flatnonzero(np.char.startswith(t, GRN_TYPE_PREFIX)).astype(np.int32),
        MN9=np.flatnonzero(t == MN9_TYPE).astype(np.int32),
    )


def selector_provenance(graph) -> dict:
    t = graph.type.astype(str)
    cl = graph.cls.astype(str)
    sc = graph.superclass.astype(str)
    pops = populations(graph)
    out = {}
    for name, sel in [
        ("KC", 'class == "Kenyon_Cell"'),
        ("MBON", 'class == "MBON"'),
        ("ALPN", 'class == "ALPN"'),
        ("ORN", 'superclass == "cb_sensory" and class == "olfactory"'),
        ("DN", 'superclass == "descending_neuron"'),
        ("APL", 'type == "APL"'),
        ("GRN", 'type.startswith("LB")'),
        ("MN9", 'type == "MN9"'),
    ]:
        idx = pops[name]
        out[name] = dict(
            selector=sel, count=int(len(idx)),
            types=sorted({t[i] for i in idx})[:20],
            classes=sorted({cl[i] for i in idx}),
            superclasses=sorted({sc[i] for i in idx}),
            sides=sorted({str(graph.side[i]) for i in idx}),
            signs=sorted({float(graph.sign[i]) for i in idx}),
            body_ids=([int(graph.body_id[i]) for i in idx] if len(idx) <= 4 else None),
        )
    return out


# --- per-input simulation -----------------------------------------------------
def run_input(brain: Brain, drive: np.ndarray, window_ms: float,
              late_from_ms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One window from reset, sliced at 1 ms so a per-ms profile falls out.

    Returns (full-window counts, counts accumulated in [0, late_from_ms),
    per-ms total-brain spike counts). Slicing costs ~15% over one call.
    """
    n_ms = int(round(window_ms))
    if abs(n_ms - window_ms) > 1e-9:
        raise ValueError("window must be a whole number of ms for the per-ms profile")
    brain.reset()
    per_ms = np.zeros(n_ms, np.int64)
    early = None
    prev = 0
    for ms in range(n_ms):
        brain.step(drive, 1.0, reset_counts=False)
        tot = int(brain.counts.sum())
        per_ms[ms] = tot - prev
        prev = tot
        if ms + 1 == late_from_ms:
            early = brain.counts.copy()
    if early is None:
        early = np.zeros_like(brain.counts)
    return brain.counts.copy(), early, per_ms


def collect(brain: Brain, fm: FeatureMap, F: np.ndarray, pops: dict,
            window_ms: float, drive_mv: float, late_from_ms: int = 15,
            label: str = "") -> dict:
    """Simulate every input; keep count matrices for the populations we report."""
    keep = ["KC", "MBON", "ALPN", "ORN", "DN", "APL"]
    mats = {k: np.zeros((len(F), len(pops[k])), np.int32) for k in keep}
    kc_late = np.zeros((len(F), len(pops["KC"])), np.int32)
    kc_per_ms = np.zeros((len(F), int(round(window_ms))), np.int64)
    brain_per_ms = np.zeros((len(F), int(round(window_ms))), np.int64)
    t0 = time.perf_counter()
    brain.warmup()
    kc = pops["KC"]
    for i, f in enumerate(F):
        d = fm.drive(f, drive_mv=drive_mv)
        # Per-ms KC counts need the KC slice each ms, so run the slices here.
        brain.reset()
        prev_all = 0
        prev_kc = 0
        early = None
        for ms in range(int(round(window_ms))):
            brain.step(d, 1.0, reset_counts=False)
            tot = int(brain.counts.sum())
            kct = int(brain.counts[kc].sum())
            brain_per_ms[i, ms] = tot - prev_all
            kc_per_ms[i, ms] = kct - prev_kc
            prev_all, prev_kc = tot, kct
            if ms + 1 == late_from_ms:
                early = brain.counts[kc].copy()
        for k in keep:
            mats[k][i] = brain.counts[pops[k]]
        kc_late[i] = mats["KC"][i] - (early if early is not None else 0)
        if label and (i + 1) % 50 == 0:
            print(f"    [{label}] {i+1}/{len(F)}  {time.perf_counter()-t0:.0f}s", flush=True)
    return dict(mats=mats, kc_late=kc_late, kc_per_ms=kc_per_ms,
                brain_per_ms=brain_per_ms, seconds=time.perf_counter() - t0)


# --- metrics ------------------------------------------------------------------
def mean_pairwise_jaccard(active: np.ndarray) -> float:
    """Mean Jaccard over all input pairs. `active` is bool (n_inputs, n_cells)."""
    A = active.astype(np.int32)
    n = len(A)
    if n < 2:
        return float("nan")
    inter = A @ A.T
    size = A.sum(1)
    union = size[:, None] + size[None, :] - inter
    iu = np.triu_indices(n, 1)
    num = inter[iu].astype(np.float64)
    den = union[iu].astype(np.float64)
    j = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return float(j.mean())


def kc_metrics(counts: np.ndarray, window_ms: float) -> dict:
    """Sparsity metrics from a (n_inputs, n_kc) spike-count matrix."""
    active = counts > 0
    frac = active.mean(1)
    p = float(frac.mean())
    per_cell = active.mean(0)
    return dict(
        mean_active_fraction=round(p, 4),
        std_active_fraction=round(float(frac.std()), 4),
        min_active_fraction=round(float(frac.min()), 4),
        max_active_fraction=round(float(frac.max()), 4),
        mean_active_cells=round(float(active.sum(1).mean()), 1),
        mean_pairwise_jaccard=round(mean_pairwise_jaccard(active), 4),
        jaccard_bernoulli_reference=round(p / (2 - p), 4) if p < 2 else None,
        always_on_fraction=round(float((per_cell > 0.5).mean()), 4),
        always_on_cells=int((per_cell > 0.5).sum()),
        never_on_fraction=round(float((per_cell == 0).mean()), 4),
        mean_rate_hz=round(float(counts.mean() * 1000.0 / window_ms), 2),
        mean_rate_hz_active_only=(
            round(float(counts[active].mean() * 1000.0 / window_ms), 2)
            if active.any() else 0.0),
        nonzero_variance_cells=int((counts.std(0) > 0).sum()),
        population=int(counts.shape[1]),
        n_inputs=int(counts.shape[0]),
    )


def pop_rate(counts: np.ndarray, window_ms: float) -> dict:
    active = counts > 0
    return dict(
        population=int(counts.shape[1]),
        mean_rate_hz=round(float(counts.mean() * 1000.0 / window_ms), 2),
        mean_fraction_active=round(float(active.mean()), 4),
        mean_spikes_per_input=round(float(counts.sum(1).mean()), 1),
        nonzero_variance_cells=int((counts.std(0) > 0).sum()),
    )


def profile_summary(per_ms: np.ndarray, window_ms: float) -> dict:
    """Mean per-ms KC count plus the early/late split that motivates a t>=15 readout."""
    mean = per_ms.mean(0)
    n = len(mean)
    first = np.flatnonzero(mean > 0)
    early = mean[:15].sum()
    late = mean[15:].sum()
    return dict(
        per_ms_mean=[round(float(x), 2) for x in mean],
        first_spike_ms=int(first[0] + 1) if len(first) else None,
        peak_ms=int(mean.argmax() + 1),
        peak_per_ms=round(float(mean.max()), 2),
        total=round(float(mean.sum()), 1),
        spikes_first_15ms=round(float(early), 1),
        spikes_after_15ms=round(float(late), 1),
        fraction_in_first_15ms=round(float(early / mean.sum()), 4) if mean.sum() else None,
        mean_per_ms_first_10=round(float(mean[:10].mean()), 2),
        mean_per_ms_last_10=round(float(mean[n - 10:].mean()), 2),
    )


# --- MN9 sugar check ----------------------------------------------------------
def mn9_check(brain: Brain, pops: dict, n_total: int, drive_mv: float = 30.0,
              window_ms: float = 200.0, control_drive: np.ndarray | None = None,
              matched_seed: int = 0) -> dict:
    """Shiu et al. 2024 sanity check: sugar GRNs in -> MN9 (proboscis extension) out.

    Conditions:
      sugar_grn_drive     all labellar GRNs at `drive_mv`
      no_drive            nothing driven (must be exactly 0 spikes everywhere)
      matched_orn_control the same *number* of randomly chosen ORNs at the same
                          drive, the control that matters, because "MN9 fires
                          for sugar" is worthless if MN9 fires for any input of
                          comparable size
      olfactory_control   one of the real 180-bit probe inputs (1,800 ORNs), the
                          drive the rest of this project uses
    """
    grn, mn9, orn = pops["GRN"], pops["MN9"], pops["ORN"]
    rng = np.random.default_rng(matched_seed)
    matched = rng.choice(orn, size=min(len(grn), len(orn)), replace=False)
    out: dict = dict(
        window_ms=window_ms, drive_mv=drive_mv,
        n_grn=int(len(grn)), n_mn9=int(len(mn9)),
        mn9_body_ids=[int(brain.graph.body_id[i]) for i in mn9],
        matched_control=f"{len(matched)} random ORNs at {drive_mv} mV, seed {matched_seed}",
    )
    for name, idx in (("sugar_grn_drive", grn), ("no_drive", None),
                      ("matched_orn_control", matched)):
        d = np.zeros(n_total, np.float32)
        if idx is not None:
            d[idx] = drive_mv
        brain.reset()
        counts, _ = brain.step(d, window_ms)
        out[name] = _mn9_row(counts, mn9, grn, window_ms)
    if control_drive is not None:
        brain.reset()
        counts, _ = brain.step(control_drive, window_ms)
        out["olfactory_control"] = _mn9_row(counts, mn9, grn, window_ms)
    s = out["sugar_grn_drive"]["mn9_spikes_total"]
    z = out["no_drive"]["mn9_spikes_total"]
    mc = out["matched_orn_control"]["mn9_spikes_total"]
    c = out.get("olfactory_control", {}).get("mn9_spikes_total")
    out["verdict"] = dict(
        mn9_responds_to_sugar=bool(s > 0),
        silent_without_drive=bool(z == 0),
        sugar_selective=bool(s > max(1, 2 * mc)),
        summary=f"sugar {s} spikes / no-drive {z} / matched-ORN {mc} / 1800-ORN {c}",
    )
    return out


def _mn9_row(counts: np.ndarray, mn9: np.ndarray, grn: np.ndarray,
             window_ms: float) -> dict:
    return dict(
        mn9_spikes=[int(x) for x in counts[mn9]],
        mn9_spikes_total=int(counts[mn9].sum()),
        mn9_rate_hz=round(float(counts[mn9].mean() * 1000.0 / window_ms), 2),
        grn_rate_hz=round(float(counts[grn].mean() * 1000.0 / window_ms), 2),
        brain_spikes=int(counts.sum()),
        neurons_firing=int((counts > 0).sum()),
    )


# --- one full measurement -----------------------------------------------------
def measure(graph, tuning: Tuning, F: np.ndarray, fm: FeatureMap, pops: dict,
            window_ms: float = 50.0, drive_mv: float = 30.0, seed: int = 1,
            v_jitter: float = 6.9, late_from_ms: int = 15, mn9_window: float = 200.0,
            label: str = "", keep_matrices: bool = False) -> dict:
    """Build a Brain at this tuning, run every input, return the metric block."""
    brain = Brain(graph, seed=seed, v_jitter=v_jitter, tuning=tuning)
    got = collect(brain, fm, F, pops, window_ms, drive_mv, late_from_ms, label)
    mats = got["mats"]
    res = dict(
        tuning=tuning.to_dict(),
        tuning_label=tuning.label(),
        tuning_info=brain.tuning_info,
        window_ms=window_ms,
        drive_mv=drive_mv,
        n_inputs=int(len(F)),
        sim_seconds=round(got["seconds"], 1),
        KC=kc_metrics(mats["KC"], window_ms),
        KC_late_from_ms=late_from_ms,
        KC_late=kc_metrics(got["kc_late"], window_ms - late_from_ms),
        MBON=pop_rate(mats["MBON"], window_ms),
        ALPN=pop_rate(mats["ALPN"], window_ms),
        ORN=pop_rate(mats["ORN"], window_ms),
        DN=pop_rate(mats["DN"], window_ms),
        APL=pop_rate(mats["APL"], window_ms),
        kc_profile=profile_summary(got["kc_per_ms"], window_ms),
        brain_profile=profile_summary(got["brain_per_ms"], window_ms),
        brain_total_spikes_per_input=round(float(got["brain_per_ms"].sum(1).mean()), 1),
    )
    res["mn9"] = mn9_check(brain, pops, graph.n, drive_mv=drive_mv,
                           window_ms=mn9_window,
                           control_drive=fm.drive(F[0], drive_mv=drive_mv).copy())
    if keep_matrices:
        res["_matrices"] = dict(KC=mats["KC"], KC_late=got["kc_late"],
                                kc_per_ms=got["kc_per_ms"])
    return res


def build_inputs(graph, n_inputs: int, n_features: int, neurons_per_feature: int,
                 active_frac: float, drive_mv: float, seed: int):
    fm = FeatureMap.for_graph(graph, n_features, neurons_per_feature, seed=seed,
                              grouping="random", drive_mv=drive_mv)
    F, ys, spec = make_dataset(n_inputs, n_features, active_frac, seed + 7)
    return fm, F, ys, spec


def print_block(name: str, r: dict) -> None:
    k = r["KC"]
    p = r["kc_profile"]
    print(f"  {name}")
    print(f"    KC active fraction {k['mean_active_fraction']:.4f} "
          f"({k['mean_active_cells']:.0f}/{k['population']})  "
          f"Jaccard {k['mean_pairwise_jaccard']:.4f} "
          f"(Bernoulli ref {k['jaccard_bernoulli_reference']})  "
          f"always-on {k['always_on_fraction']:.4f} ({k['always_on_cells']})")
    print(f"    KC rate {k['mean_rate_hz']:.2f} Hz   "
          f"first KC spike {p['first_spike_ms']} ms   peak {p['peak_ms']} ms   "
          f"{100*(p['fraction_in_first_15ms'] or 0):.0f}% of KC spikes in first 15 ms")
    print(f"    MBON {r['MBON']['mean_rate_hz']:.2f} Hz   "
          f"ALPN {r['ALPN']['mean_rate_hz']:.2f} Hz   "
          f"APL {r['APL']['mean_rate_hz']:.2f} Hz   "
          f"brain {r['brain_total_spikes_per_input']:.0f} spikes/input")
    print(f"    MN9: {r['mn9']['verdict']['summary']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--n-inputs", type=int, default=200)
    ap.add_argument("--n-features", type=int, default=180)
    ap.add_argument("--neurons-per-feature", type=int, default=10)
    ap.add_argument("--active-frac", type=float, default=0.20)
    ap.add_argument("--window", type=float, default=50.0)
    ap.add_argument("--drive-mv", type=float, default=30.0)
    ap.add_argument("--v-jitter", type=float, default=6.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--brain-seed", type=int, default=1)
    ap.add_argument("--late-from-ms", type=int, default=15)
    ap.add_argument("--mn9-window", type=float, default=200.0)
    ap.add_argument("--apl-scale", type=float, default=1.0)
    ap.add_argument("--alpn-kc-scale", type=float, default=1.0)
    ap.add_argument("--kc-vth-offset", type=float, default=0.0)
    ap.add_argument("--kc-bias", type=float, default=0.0)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    g = C.load(a.threshold)
    pops = populations(g)
    prov = selector_provenance(g)
    fm, F, _ys, spec = build_inputs(g, a.n_inputs, a.n_features, a.neurons_per_feature,
                                    a.active_frac, a.drive_mv, a.seed)
    tuning = Tuning(apl_scale=a.apl_scale, alpn_kc_scale=a.alpn_kc_scale,
                    kc_vth_offset_mv=a.kc_vth_offset, kc_bias_mv=a.kc_bias)
    print("populations: " + "  ".join(f"{k}={len(v)}" for k, v in pops.items()))
    print("feature map:", fm.describe())
    print(f"tuning: {tuning.to_dict()}  (calibration knobs, not connectome data)",
          flush=True)

    r = measure(g, tuning, F, fm, pops, window_ms=a.window, drive_mv=a.drive_mv,
                seed=a.brain_seed, v_jitter=a.v_jitter, late_from_ms=a.late_from_ms,
                mn9_window=a.mn9_window, label=a.tag)
    print()
    print_block(a.tag, r)

    out = dict(
        tag=a.tag,
        settings=dict(threshold=a.threshold, n_inputs=a.n_inputs,
                      n_features=a.n_features,
                      neurons_per_feature=a.neurons_per_feature,
                      active_frac=a.active_frac, window_ms=a.window,
                      drive_mv=a.drive_mv, v_jitter_mv=a.v_jitter,
                      dataset_seed=a.seed, brain_seed=a.brain_seed,
                      late_from_ms=a.late_from_ms, mn9_window_ms=a.mn9_window,
                      feature_map=fm.describe(),
                      machine=f"{platform.machine()} python {platform.python_version()}"),
        dataset=spec,
        graph=dict(neurons=g.n, edges_csr=g.m, w_syn=g.meta["w_syn"],
                   threshold=g.meta["threshold"]),
        selectors=prov,
        result=r,
    )
    p = Path(a.out) if a.out else OUT_DIR / f"{a.tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
