"""Sweep the mushroom-body calibration knobs and record the sparsity metrics.

Every setting is a ``flybalatro.tuning.Tuning``; every metric comes from
``scripts.mb_sparsity.measure`` on the same N inputs from the same generator, so
rows are directly comparable. Results are merged into ``outputs/mb/sweep.json``
keyed by tuning label, so stages can be run one at a time.

Target for a usable mushroom body: KC active fraction 5-10%, always-on fraction
near 0, pairwise Jaccard close to the Bernoulli reference p/(2-p), and the MN9
gustatory check unchanged (none of these knobs is on the GRN -> MN9 path, so a
change there would mean a bug).

Stages:
    apl     APL out-synapse scale in {1, 2, 5, 10, 20, 50}
    vth     KC threshold offset in {0, 2, 4, 6, 8} mV
    pnkc    ALPN -> KC scale in {1, 0.5, 0.25} (and >1 via --custom)
    kckc    KC -> KC recurrent scale in {1, 0.5, 0.25, 0}
    grid    2-D grid: apl_scale x kc_vth_offset_mv
    grid_kckc  kc_kc_scale x apl_scale
    norm    per-KC ALPN input normalisation, alone and combined
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from flybalatro import connectome as C
from flybalatro.tuning import Tuning
from scripts.mb_sparsity import (
    OUT_DIR,
    build_inputs,
    measure,
    populations,
    print_block,
    selector_provenance,
)

STAGES: dict[str, list[Tuning]] = {
    "apl": [Tuning(apl_scale=s) for s in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)],
    "vth": [Tuning(kc_vth_offset_mv=v) for v in (0.0, 2.0, 4.0, 6.0, 8.0)],
    "pnkc": [Tuning(alpn_kc_scale=s) for s in (1.0, 0.5, 0.25)],
    # Filled in after the 1-D stages; see --custom for ad-hoc settings.
    "kckc": [Tuning(kc_kc_scale=s) for s in (1.0, 0.5, 0.25, 0.0)],
    # 2-D grid over the two knobs that moved the metrics most (see REPORT.md).
    "grid": [Tuning(apl_scale=a, kc_vth_offset_mv=v)
             for a in (1.0, 2.0, 3.0, 5.0) for v in (0.0, 2.0, 4.0, 6.0)],
    "grid_kckc": [Tuning(kc_kc_scale=k, apl_scale=a)
                  for k in (0.0, 0.25) for a in (1.0, 2.0, 3.0, 5.0)],
    # Per-KC ALPN input normalisation: the only non-uniform knob.
    "norm": [Tuning(kc_input_norm=1.0), Tuning(kc_input_norm=0.5),
             Tuning(kc_input_norm=1.0, kc_kc_scale=0.0),
             Tuning(kc_input_norm=1.0, kc_kc_scale=0.0, apl_scale=2.0),
             Tuning(kc_input_norm=1.0, kc_kc_scale=0.0, apl_scale=3.0),
             Tuning(kc_input_norm=1.0, kc_kc_scale=0.0, apl_scale=5.0),
             Tuning(kc_input_norm=1.0, apl_scale=2.0),
             Tuning(kc_input_norm=1.0, apl_scale=3.0)],
}

ROW_KEYS = ("apl_scale", "alpn_kc_scale", "kc_vth_offset_mv", "kc_bias_mv",
            "kc_kc_scale", "kc_input_norm")


def row(r: dict) -> dict:
    """One flat table row per setting, for the report."""
    k, p, lt = r["KC"], r["kc_profile"], r["KC_late"]
    return dict(
        label=r["tuning_label"],
        **{key: r["tuning"][key] for key in ROW_KEYS},
        kc_active_fraction=k["mean_active_fraction"],
        kc_active_cells=k["mean_active_cells"],
        kc_jaccard=k["mean_pairwise_jaccard"],
        kc_jaccard_bernoulli_ref=k["jaccard_bernoulli_reference"],
        kc_always_on_fraction=k["always_on_fraction"],
        kc_always_on_cells=k["always_on_cells"],
        kc_never_on_fraction=k["never_on_fraction"],
        kc_rate_hz=k["mean_rate_hz"],
        kc_varying_cells=k["nonzero_variance_cells"],
        kc_active_fraction_late=lt["mean_active_fraction"],
        kc_jaccard_late=lt["mean_pairwise_jaccard"],
        kc_first_spike_ms=p["first_spike_ms"],
        kc_peak_ms=p["peak_ms"],
        kc_fraction_first_15ms=p["fraction_in_first_15ms"],
        mbon_rate_hz=r["MBON"]["mean_rate_hz"],
        alpn_rate_hz=r["ALPN"]["mean_rate_hz"],
        apl_rate_hz=r["APL"]["mean_rate_hz"],
        dn_rate_hz=r["DN"]["mean_rate_hz"],
        brain_spikes_per_input=r["brain_total_spikes_per_input"],
        mn9_sugar_spikes=r["mn9"]["sugar_grn_drive"]["mn9_spikes_total"],
        mn9_no_drive_spikes=r["mn9"]["no_drive"]["mn9_spikes_total"],
        mn9_matched_orn_spikes=r["mn9"]["matched_orn_control"]["mn9_spikes_total"],
        sim_seconds=r["sim_seconds"],
    )


def print_table(rows: list[dict]) -> None:
    head = (f"{'setting':<30} {'KCfrac':>7} {'always':>7} {'Jacc':>7} {'ref':>6} "
            f"{'KCHz':>6} {'MBONHz':>7} {'ALPNHz':>7} {'MN9':>4}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['label']:<30} {r['kc_active_fraction']:>7.4f} "
              f"{r['kc_always_on_fraction']:>7.4f} {r['kc_jaccard']:>7.4f} "
              f"{r['kc_jaccard_bernoulli_ref']:>6.3f} {r['kc_rate_hz']:>6.2f} "
              f"{r['mbon_rate_hz']:>7.2f} {r['alpn_rate_hz']:>7.2f} "
              f"{r['mn9_sugar_spikes']:>4}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="apl",
                    help="comma-separated: " + ",".join(STAGES))
    ap.add_argument("--custom", default=None,
                    help='JSON list of tuning dicts, e.g. \'[{"apl_scale":3}]\'')
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
    ap.add_argument("--out", default=str(OUT_DIR / "sweep.json"))
    a = ap.parse_args()

    settings: list[Tuning] = []
    if a.custom:
        settings += [Tuning.from_dict(d) for d in json.loads(a.custom)]
    for st in [s for s in a.stages.split(",") if s]:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st!r}; have {sorted(STAGES)}")
        settings += STAGES[st]
    # De-duplicate by label, preserving order.
    seen: dict[str, Tuning] = {}
    for t in settings:
        seen.setdefault(t.label(), t)
    settings = list(seen.values())

    g = C.load(a.threshold)
    pops = populations(g)
    fm, F, _ys, spec = build_inputs(g, a.n_inputs, a.n_features,
                                    a.neurons_per_feature, a.active_frac,
                                    a.drive_mv, a.seed)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(out_path.read_text()) if out_path.exists() else {}
    doc.setdefault("settings", dict(
        threshold=a.threshold, n_inputs=a.n_inputs, n_features=a.n_features,
        neurons_per_feature=a.neurons_per_feature, active_frac=a.active_frac,
        window_ms=a.window, drive_mv=a.drive_mv, v_jitter_mv=a.v_jitter,
        dataset_seed=a.seed, brain_seed=a.brain_seed,
        late_from_ms=a.late_from_ms, mn9_window_ms=a.mn9_window,
        feature_map=fm.describe(),
        machine=f"{platform.machine()} python {platform.python_version()}",
        note="all knobs are calibration parameters, not connectome data",
    ))
    doc.setdefault("dataset", spec)
    doc.setdefault("selectors", selector_provenance(g))
    doc.setdefault("runs", {})
    doc.setdefault("stages_run", [])
    doc["stages_run"] = sorted(set(doc["stages_run"]) | set(a.stages.split(",")) - {""})

    print(f"{len(settings)} settings x {a.n_inputs} inputs", flush=True)
    t0 = time.perf_counter()
    for i, t in enumerate(settings, 1):
        if t.label() in doc["runs"]:
            print(f"[{i}/{len(settings)}] {t.label()} -- cached, skipping", flush=True)
            continue
        print(f"[{i}/{len(settings)}] {t.to_dict()}", flush=True)
        r = measure(g, t, F, fm, pops, window_ms=a.window, drive_mv=a.drive_mv,
                    seed=a.brain_seed, v_jitter=a.v_jitter,
                    late_from_ms=a.late_from_ms, mn9_window=a.mn9_window,
                    label=t.label())
        print_block(t.label(), r)
        doc["runs"][t.label()] = r
        doc["table"] = [row(v) for v in doc["runs"].values()]
        out_path.write_text(json.dumps(doc, indent=2))
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    print_table(doc["table"])
    print("wrote", out_path)


if __name__ == "__main__":
    main()
