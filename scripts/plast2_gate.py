"""Gate for the homeostatic Kenyon-cell code, on 500 held-out relay patterns.

Five criteria, all pre-registered in the task brief, measured on the HELD-OUT
half of the ``outputs/bc2`` relay patterns (the calibration half is what
``scripts/kc_homeo.py`` fitted the offsets on, and never appears here):

  (i)   always-on fraction, frac(r > 0.5) < 0.02 and frac(r > 0.3) < 0.05
  (ii)  KC active fraction in 3-10%
  (iii) 9-way hand-type decoding from the KC spike counts >= 0.85, logistic,
        5-fold CV grouped by relay pattern
  (iv)  the MBON approach/avoid drive still varies across odours, and the
        reward-side MBONs (MBON01-07) are not silent
  (v)   conditioning specificity, exactly the ``outputs/plast/probe`` protocol
        (8-odour panel from the 300000+ calibration games, 20 pulses, eta 0.05):
        the paired odour's delta must exceed the mean unpaired delta by >= 2x,
        i.e. >= 50% of the effect is odour-specific, in both directions.

The baseline column is the *same* measurement on ``outputs/plast/tuned_config.json``,
so every number in the report is a before/after on one odour set.

Run::

    python -m scripts.plast2_gate                     # baseline + homeostasis
    python -m scripts.plast2_gate --fallback          # + the KC -> MBON normalisation
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from flybalatro.tuning import Tuning  # noqa: E402
from scripts import kc_homeo as H  # noqa: E402
from scripts import plast_common as K  # noqa: E402
from scripts.plast_kccheck import _probe  # noqa: E402
from scripts.plast_probe import (  # noqa: E402
    collect_calibration_hands,
    conditioning_panel,
    hand_bits,
    pick_odor_panel,
)

OUT_DIR = H.OUT_DIR
GATE_JSON = OUT_DIR / "gate.json"

#: Pass thresholds, from the brief.
MAX_FRAC_R_GT_05 = 0.02
MAX_FRAC_R_GT_03 = 0.05
ACTIVE_FRAC_RANGE = (0.03, 0.10)
MIN_DECODE = 0.85
MIN_SPECIFICITY_SHARE = 0.5     # paired delta >= 2x mean unpaired delta
PANEL_ETA = 0.05
PANEL_TRIALS = 20
PANEL_HANDS = 200

#: The reward arm's MBONs: glutamatergic, PAM-innervated, and the ones that were
#: silent for every input before ``apl_mbon_scale`` was introduced.
REWARD_MBON_TYPES = tuple(f"MBON{i:02d}" for i in range(1, 8))


# --------------------------------------------------------------------------- #
def measure(setup: K.Setup, patterns: NDArray[np.float32],
            labels: NDArray[np.int_], dec) -> dict:
    """One full pass: KC rates, decoding, MBON drive range."""
    n_kc = len(setup.kc)
    mbon = setup.valence.mbon
    types = setup.graph.type.astype(str)[mbon]
    reward_rows = np.flatnonzero(np.isin(types, REWARD_MBON_TYPES))
    scale = 1000.0 / setup.window_ms

    X = np.zeros((len(patterns), n_kc), np.float32)
    M = np.zeros((len(patterns), len(mbon)), np.float32)
    raw = np.zeros(len(patterns))
    appr = np.zeros(len(patterns))
    avo = np.zeros(len(patterns))
    for i, b in enumerate(patterns):
        c = setup.run_window(b)
        X[i] = c[setup.kc]
        M[i] = c[mbon]
        raw[i] = dec.raw_drive(c)
        appr[i] = float(c[dec.approach].mean()) * scale
        avo[i] = float(c[dec.avoid].mean()) * scale

    act = X > 0
    r = act.mean(axis=0)
    mb_rate = M.mean(axis=0) * scale
    reward_rate = mb_rate[reward_rows]
    quantum = (1.0 / len(dec.approach) + 1.0 / len(dec.avoid)) * scale

    return dict(
        kc=dict(
            **H.rate_stats(r, act.sum(axis=1).astype(np.int32), n_kc),
            kc_spikes_per_state=round(float(X.sum(axis=1).mean()), 1),
            rate_percentiles={str(p): round(float(np.percentile(r, p)), 4)
                              for p in (50, 75, 90, 95, 99, 100)},
        ),
        decode=dict(
            kc_counts=_probe(np.log1p(X), labels, np.arange(len(patterns))),
            kc_binary=_probe(act.astype(np.float32), labels,
                             np.arange(len(patterns))),
        ),
        mbon=dict(
            n_mbon=int(len(mbon)),
            mean_hz=round(float(M.mean()) * scale, 3),
            n_varying=int((((M > 0).mean(axis=0) > 0)
                           & ((M > 0).mean(axis=0) < 1)).sum()),
            n_silent=int((mb_rate == 0.0).sum()),
            drive_min=round(float(raw.min()), 4),
            drive_max=round(float(raw.max()), 4),
            drive_range=round(float(raw.max() - raw.min()), 4),
            drive_sd=round(float(raw.std()), 4),
            drive_distinct=int(len(np.unique(np.round(raw, 6)))),
            quantum_hz=round(quantum, 4),
            range_in_quanta=round(float((raw.max() - raw.min()) / quantum), 2),
            approach_hz_mean=round(float(appr.mean()), 3),
            avoid_hz_mean=round(float(avo.mean()), 3),
            reward_side_types=list(REWARD_MBON_TYPES),
            reward_side_n=int(len(reward_rows)),
            reward_side_hz_mean=round(float(reward_rate.mean()), 4)
            if len(reward_rows) else 0.0,
            reward_side_hz_max=round(float(reward_rate.max()), 4)
            if len(reward_rows) else 0.0,
            reward_side_n_silent=int((reward_rate == 0.0).sum()),
        ),
        raw_drive=[round(float(x), 4) for x in raw],
    )


def specificity(setup: K.Setup, dec, panel_bits: NDArray[np.float32],
                panel_idx: Sequence[int], recs: Sequence[dict]) -> dict:
    out: Dict[str, dict] = {}
    for kind in ("punish", "reward"):
        out[kind] = conditioning_panel(setup, dec, panel_bits, panel_idx, recs,
                                       kind, PANEL_ETA, PANEL_TRIALS)
    return out


def verdict(m: dict, spec: dict) -> dict:
    kc = m["kc"]
    mb = m["mbon"]
    c1 = bool(kc["frac_r_gt_0_5"] < MAX_FRAC_R_GT_05
              and kc["frac_r_gt_0_3"] < MAX_FRAC_R_GT_03)
    c2 = bool(ACTIVE_FRAC_RANGE[0] <= kc["active_frac"] <= ACTIVE_FRAC_RANGE[1])
    acc = m["decode"]["kc_counts"]["grouped"]
    acc = acc if acc is not None else m["decode"]["kc_counts"]["accuracy"]
    c3 = bool(acc >= MIN_DECODE)
    c4 = bool(mb["drive_range"] > mb["quantum_hz"]
              and mb["reward_side_n_silent"] < mb["reward_side_n"])
    shares = {k: spec[k]["specificity_share"] for k in ("reward", "punish")}
    dirs = {k: spec[k]["direction_ok"] for k in ("reward", "punish")}
    c5 = bool(all(v >= MIN_SPECIFICITY_SHARE for v in shares.values())
              and all(dirs.values()))
    return dict(
        i_always_on=dict(pass_=c1, frac_r_gt_0_5=kc["frac_r_gt_0_5"],
                         frac_r_gt_0_3=kc["frac_r_gt_0_3"],
                         limits=[MAX_FRAC_R_GT_05, MAX_FRAC_R_GT_03]),
        ii_active_frac=dict(pass_=c2, active_frac=kc["active_frac"],
                            range=list(ACTIVE_FRAC_RANGE)),
        iii_decode=dict(pass_=c3, grouped_accuracy=acc,
                        plain_accuracy=m["decode"]["kc_counts"]["accuracy"],
                        permuted=m["decode"]["kc_counts"]["permuted"],
                        binary_grouped=m["decode"]["kc_binary"]["grouped"],
                        limit=MIN_DECODE),
        iv_mbon=dict(pass_=c4, drive_range_hz=mb["drive_range"],
                     drive_min=mb["drive_min"], drive_max=mb["drive_max"],
                     quantum_hz=mb["quantum_hz"],
                     reward_side_hz_mean=mb["reward_side_hz_mean"],
                     reward_side_n_silent=mb["reward_side_n_silent"],
                     reward_side_n=mb["reward_side_n"]),
        v_specificity=dict(pass_=c5, specificity_share=shares,
                           direction_ok=dirs,
                           delta_self={k: spec[k]["mean_delta_self"]
                                       for k in shares},
                           delta_other={k: spec[k]["mean_delta_other"]
                                        for k in shares},
                           limit=MIN_SPECIFICITY_SHARE),
        all_pass=bool(c1 and c2 and c3 and c4 and c5),
    )


# --------------------------------------------------------------------------- #
def fallback_out_scale(rate_calib: NDArray[np.float64],
                       target: float = H.TARGET) -> NDArray[np.float32]:
    """Static KC -> MBON normalisation: ``m_i = target / max(r_i, target)``.

    The documented second mechanism. A Kenyon cell whose residual response rate
    is still 2x the population target contributes half as much to the output
    stage, so the credit a dopamine pulse can put on a broadly-tuned cell is
    scaled by how broadly tuned it is. Measured on the CALIBRATION patterns (the
    same set the thresholds were fitted on), applied once at ``Brain``
    construction, and never changed while the network runs -- a calibration of
    the output gain, not plasticity.
    """
    r = np.asarray(rate_calib, np.float64)
    return np.minimum(1.0, target / np.maximum(r, target)).astype(np.float32)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fallback", action="store_true",
                    help="also measure the kc_mbon_out_scale normalisation")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--write-config", action="store_true",
                    help="fold the fallback normalisation into "
                         "outputs/plast2/tuned_config.json (needs --fallback)")
    ap.add_argument("--out", default=str(GATE_JSON))
    a = ap.parse_args(argv)

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = K.C.load(5)
    sets = H.relay_patterns()
    gate = sets["gate"]
    labels = gate[:, :9].argmax(axis=1).astype(int)

    print(f"conditioning panel: {PANEL_HANDS} hands from seeds {K.CALIB_SEED0}+")
    recs = collect_calibration_hands(PANEL_HANDS, K.CALIB_SEED0, rng_seed=7)
    panel_bits = hand_bits(recs)
    panel_idx = pick_odor_panel(recs, 8)

    homeo = json.loads(H.HOMEO_JSON.read_text())
    rate_calib = np.asarray(homeo["final_calib_rate"], np.float64)
    offsets = np.asarray(homeo["offsets"], np.float32)

    conditions: List[tuple] = []
    if not a.skip_baseline:
        conditions.append(("baseline_plast", None, None))
    conditions.append(("kc_homeostasis", offsets, None))
    if a.fallback:
        conditions.append(("kc_homeostasis_plus_mbon_norm", offsets,
                           fallback_out_scale(rate_calib)))

    payload: dict = dict(
        patterns=sets["meta"],
        gate_set="the 500 HELD-OUT patterns; the calibration 500 never appear here",
        panel=dict(n_hands=PANEL_HANDS, seed0=K.CALIB_SEED0, rng_seed=7,
                   eta=PANEL_ETA, trials=PANEL_TRIALS,
                   odors=[dict(index=int(i), bucket=recs[i]["bucket_name"],
                               hand_type=recs[i]["best_type_name"])
                          for i in panel_idx]),
        criteria=dict(i_frac_r_gt_0_5_max=MAX_FRAC_R_GT_05,
                      i_frac_r_gt_0_3_max=MAX_FRAC_R_GT_03,
                      ii_active_frac_range=list(ACTIVE_FRAC_RANGE),
                      iii_min_grouped_accuracy=MIN_DECODE,
                      iv="drive range > one quantum and MBON01-07 not all silent",
                      v_min_specificity_share=MIN_SPECIFICITY_SHARE),
        conditions={},
    )

    base_cfg = H.base_config()
    for name, off, out_scale in conditions:
        print(f"\n== {name} ==", flush=True)
        setup = H.build(base_cfg, graph, off, out_scale)
        dec = K.make_decider(setup)
        m = measure(setup, gate, labels, dec)
        kc, mb = m["kc"], m["mbon"]
        print(f"   KC active {kc['active_frac']:.4f} "
              f"({kc['active_per_state']:.0f}/state), r>0.5 {kc['n_r_gt_0_5']} "
              f"({kc['frac_r_gt_0_5']:.4f}), r>0.3 {kc['n_r_gt_0_3']} "
              f"({kc['frac_r_gt_0_3']:.4f}), responsive {kc['n_responsive']}",
              flush=True)
        print(f"   decode grouped {m['decode']['kc_counts']['grouped']} "
              f"(plain {m['decode']['kc_counts']['accuracy']}, "
              f"permuted {m['decode']['kc_counts']['permuted']})", flush=True)
        print(f"   MBON {mb['mean_hz']:.2f} Hz, drive {mb['drive_min']:+.2f}.."
              f"{mb['drive_max']:+.2f} Hz ({mb['range_in_quanta']} quanta), "
              f"reward-side {mb['reward_side_hz_mean']:.2f} Hz, "
              f"{mb['reward_side_n_silent']}/{mb['reward_side_n']} silent",
              flush=True)

        # calibrate the decider on these same odours only so the panel's
        # play_drive is on a sensible scale; deltas are bias-independent.
        dec.calibrate(np.asarray(m["raw_drive"]), explore=0.20)
        spec = specificity(setup, dec, panel_bits, panel_idx, recs)
        for kind in ("punish", "reward"):
            s = spec[kind]
            print(f"   {kind:7s} dself {s['mean_delta_self']:+.3f} "
                  f"dother {s['mean_delta_other']:+.3f} "
                  f"specificity {s['mean_specificity']:+.3f} "
                  f"({s['specificity_share']:.0%})  "
                  f"dir_ok={s['direction_ok']} "
                  f"{s['n_odors_right_direction']}/{s['n_odors']}", flush=True)

        v = verdict(m, spec)
        row = dict(m)
        row.pop("raw_drive", None)
        payload["conditions"][name] = dict(
            measure=row, specificity=spec, verdict=v,
            tuning_info_kc_vth=setup.brain.tuning_info.get("kc_vth_offsets", {}),
            tuning_info_kc_mbon=setup.brain.tuning_info.get("kc_mbon_out_scale", {}),
            decider=dec.to_dict(),
        )
        print("   verdict: " + "  ".join(
            f"({k.split('_')[0]}){'PASS' if v[k]['pass_'] else 'FAIL'}"
            for k in ("i_always_on", "ii_active_frac", "iii_decode", "iv_mbon",
                      "v_specificity")) + f"  ALL={'PASS' if v['all_pass'] else 'FAIL'}",
              flush=True)
        del setup

    payload["wall_seconds"] = round(time.time() - t0, 1)
    K.write_json(Path(a.out), payload)
    print(f"\nwrote {a.out} in {payload['wall_seconds']:.0f}s")

    if a.write_config:
        # Fold the fallback normalisation into the shipped fly. Only ever called
        # explicitly, and only after its own gate column is on the record.
        name = "kc_homeostasis_plus_mbon_norm"
        if name not in payload["conditions"]:
            raise SystemExit("--write-config needs --fallback")
        sc = fallback_out_scale(rate_calib)
        cfg2 = H.config_with(base_cfg, offsets, sc)
        cfg2["derived_from"] = "outputs/plast/tuned_config.json"
        cfg2["label"] = Tuning.from_dict(cfg2["tuning"]).label()
        shipped = json.loads(H.CONFIG_JSON.read_text())
        cfg2["kc_homeostasis"] = shipped.get("kc_homeostasis", {})
        cfg2["kc_mbon_normalisation"] = dict(
            script="scripts/plast2_gate.py", rule="m_i = target / max(r_i, target)",
            target=H.TARGET,
            measured_on=("the 500 CALIBRATION relay patterns (the same set the "
                         "thresholds were fitted on)"),
            n_scaled=int((sc < 1.0).sum()), scale_min=round(float(sc.min()), 4),
            scale_mean=round(float(sc.mean()), 4),
            note=("static calibration of the output gain, applied once at Brain "
                  "construction so KcMbonPlasticity sees it as w0; nothing about "
                  "it changes while the network runs"),
            gate_verdict=payload["conditions"][name]["verdict"],
        )
        K.write_json(H.CONFIG_JSON, cfg2)
        print(f"wrote {H.CONFIG_JSON} with kc_mbon_out_scale "
              f"({int((sc < 1.0).sum())} KCs scaled, min {float(sc.min()):.4f})")


if __name__ == "__main__":
    main()
