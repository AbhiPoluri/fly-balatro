"""Per-Kenyon-cell homeostatic thresholds for the **separated** bucket encoding.

The rule, the target, the deadband, the annealed gain schedule, the per-cell
lower clamp and the stop criterion are all ``scripts/kc_homeo.py`` exactly as it
shipped for v2 -- this script only swaps the odour map (four disjoint
ORN-balanced bucket ensembles instead of one glomerulus per relay bit) and the
output paths, so ``outputs/plast2/`` is never written.

    offset_i += k * log((r_i + eps) / (target + eps))        target = 0.07

``r_i`` is the fraction of odours for which Kenyon cell ``i`` emits at least one
spike. The update sees **only that** -- no labels, no reward, no MBONs, no
readout -- which is why recalibrating for a new encoding is not a way of fitting
the task.

One honest difference from v2, forced by the encoding and declared in
``PREREGISTRATION.md`` section 3: the separated encoding drives receptors from 13
of the 32 relay bits (9 hand types + 4 buckets), so the odour universe is 36
patterns and the 500/500 calibration/held-out split of distinct 32-bit patterns
collapses -- both halves project onto the same handful of odours. The calibration
set is therefore the distinct *effective* odours of the bc2 calibration half, and
there is no held-out odour set to report the gate on. Stated, not hidden.

Run::

    python -m scripts.plast3_homeo --iterations 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from flybalatro.brain import V_TH
from flybalatro.tuning import Tuning
from scripts import kc_homeo as H
from scripts import plast3_common as C
from scripts import plast_common as K

HOMEO_JSON = C.OUT_DIR / "kc_homeo_sep.json"
CONFIG_JSON = C.OUT_DIR / "tuned_config_sep.json"


def effective_odours(patterns: NDArray[np.float32]) -> NDArray[np.float32]:
    """Distinct odours the separated encoding can tell apart, in a stable order."""
    proj = np.asarray([C.effective_bits(C.ENC_SEPARATED, b) for b in patterns],
                      np.float32)
    uniq = np.unique(proj, axis=0)
    return uniq.astype(np.float32)


def build(cfg: dict, graph, offsets: Optional[NDArray[np.float32]] = None) -> K.Setup:
    return C.build_setup3(C.ENC_SEPARATED, H.config_with(cfg, offsets), graph)


def calibrate(n_iter: int, out: Path = HOMEO_JSON,
              config_out: Path = CONFIG_JSON) -> dict:
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    graph = K.C.load(5)
    cfg = H.base_config()
    sets = H.relay_patterns()
    calib = effective_odours(sets["calib"])
    gate = effective_odours(sets["gate"])
    shared = int(sum(1 for g in gate if any((g == c).all() for c in calib)))

    setup0 = build(cfg, graph)
    n_kc = len(setup0.kc)
    v0_kc = np.asarray(setup0.brain._v0[setup0.kc], np.float64)
    floor = np.maximum(H.OFFSET_LO, v0_kc - V_TH + H.V0_MARGIN_MV).astype(np.float64)
    del setup0

    history: List[dict] = []
    offsets = np.zeros(n_kc, np.float64)
    prev_sign = np.zeros(n_kc, np.int8)

    payload: dict = dict(
        encoding=C.ENC_SEPARATED,
        assignment=[list(x) for x in C.assignment_for(C.ENC_SEPARATED, graph)],
        target=H.TARGET, eps=H.EPS, deadband=list(H.DEADBAND),
        k_schedule=list(H.K_SCHEDULE), offset_clamp=[H.OFFSET_LO, H.OFFSET_HI],
        max_step_down_mv=H.MAX_STEP_DOWN_MV, max_step_up_mv=H.MAX_STEP_UP_MV,
        v0_margin_mv=H.V0_MARGIN_MV,
        stop_frac_r_gt_0_3=H.STOP_FRAC_R30,
        n_kc=int(n_kc), base_tuning=cfg["tuning"],
        rule=("offset_i += k * log((r_i + eps) / (target + eps)); r_i measured on "
              "the CALIBRATION odours only; no labels, no reward, no readout"),
        odours=dict(
            source=sets["meta"]["source"],
            n_calib_patterns_32bit=int(len(sets["calib"])),
            n_calib_effective=int(len(calib)),
            n_gate_effective=int(len(gate)),
            n_gate_effective_shared_with_calib=shared,
            note=("the separated encoding drives receptors from 13 of the 32 relay "
                  "bits, so the 500/500 distinct-pattern split projects onto the "
                  "same small odour universe; there is no held-out odour set under "
                  "this encoding, and the calibration rule sees only each cell's "
                  "own firing rate either way"),
            quantisation=("r_i is a multiple of 1/%d, so target 0.07 is %.1f odours "
                          "and the deadband [0.04, 0.12] is 2-4 odours"
                          % (len(calib), 0.07 * len(calib))),
        ),
    )

    for it in range(n_iter):
        t0 = time.time()
        setup = build(cfg, graph, offsets.astype(np.float32))
        resp = H.kc_response(setup, calib)
        r = resp["rate"]
        st = H.rate_stats(r, resp["active_per_state"], n_kc)
        gain = H.K_SCHEDULE[min(it, len(H.K_SCHEDULE) - 1)]
        step = gain * np.log((r + H.EPS) / (H.TARGET + H.EPS))
        step = np.clip(step, -H.MAX_STEP_DOWN_MV, H.MAX_STEP_UP_MV)
        in_band = (r >= H.DEADBAND[0]) & (r <= H.DEADBAND[1])
        step[in_band] = 0.0
        sign = np.sign(step).astype(np.int8)
        moved = sign != 0
        flipped = moved & (prev_sign != 0) & (sign != prev_sign)
        new = np.clip(offsets + step, floor, H.OFFSET_HI)

        row = dict(iteration=int(it), gain=float(gain), **st,
                   n_moved=int(moved.sum()),
                   frac_sign_flipped=round(float(flipped.sum() / max(1, moved.sum())), 4),
                   n_at_floor=int((new <= floor + 1e-6).sum()),
                   n_at_ceiling=int((new >= H.OFFSET_HI - 1e-6).sum()),
                   offset_mean=round(float(offsets.mean()), 4),
                   offset_sd=round(float(offsets.std()), 4),
                   offset_min=round(float(offsets.min()), 4),
                   offset_max=round(float(offsets.max()), 4),
                   seconds=round(time.time() - t0, 1))
        history.append(row)
        print(f"[{it:2d}] k={gain:<4g} active {st['active_frac']:.4f} "
              f"({st['active_per_state']:.0f}/odour)  r>0.5 {st['n_r_gt_0_5']:4d}  "
              f"r>0.3 {st['n_r_gt_0_3']:4d}  in-band {st['n_in_band']:4d}  "
              f"silent {st['n_silent']:4d}  flips {row['frac_sign_flipped']:.2f}  "
              f"{row['seconds']:.0f}s", flush=True)
        del setup

        converged = bool(st["frac_r_gt_0_3"] < H.STOP_FRAC_R30 and it >= 3)
        row["converged"] = converged
        row["update_applied"] = not converged
        payload.update(iterations=history,
                       offsets=[round(float(x), 4) for x in offsets],
                       wall_seconds=round(time.time() - t_start, 1))
        C.write_json(out, payload)
        if converged:
            print(f"converged at iteration {it}: frac(r>0.3) = "
                  f"{st['frac_r_gt_0_3']:.4f} (last update not applied)")
            break
        offsets = new
        prev_sign = sign

    setup = build(cfg, graph, offsets.astype(np.float32))
    resp = H.kc_response(setup, calib)
    final = H.rate_stats(resp["rate"], resp["active_per_state"], n_kc)
    respg = H.kc_response(setup, gate)
    final_gate = H.rate_stats(respg["rate"], respg["active_per_state"], n_kc)
    final.update(offset_mean=round(float(offsets.mean()), 4),
                 offset_sd=round(float(offsets.std()), 4),
                 offset_min=round(float(offsets.min()), 4),
                 offset_max=round(float(offsets.max()), 4),
                 n_at_floor=int((offsets <= floor + 1e-6).sum()),
                 n_at_ceiling=int((offsets >= H.OFFSET_HI - 1e-6).sum()))
    payload["final_calib"] = final
    payload["final_gate"] = final_gate
    payload["offsets"] = [round(float(x), 4) for x in offsets]
    payload["wall_seconds"] = round(time.time() - t_start, 1)
    C.write_json(out, payload)

    cfg2 = H.config_with(cfg, offsets.astype(np.float32))
    cfg2["derived_from"] = "outputs/plast/tuned_config.json"
    cfg2["label"] = Tuning.from_dict(cfg2["tuning"]).label() + "_sepenc"
    cfg2["encoding"] = dict(
        kind="glomerular_ensemble", n_features=32,
        assignment=[dict(bit=i, glomeruli=list(g))
                    for i, g in enumerate(C.assignment_for(C.ENC_SEPARATED, graph))],
        note="separated bucket ensembles; see outputs/plast3/PREREGISTRATION.md",
    )
    cfg2["kc_homeostasis"] = dict(
        script="scripts/plast3_homeo.py", detail=C.rel_path(out),
        target=H.TARGET, iterations=len(history),
        calibration_odours=int(len(calib)),
        summary={k: final[k] for k in
                 ("active_frac", "active_per_state", "frac_r_gt_0_5",
                  "frac_r_gt_0_3", "n_silent", "n_responsive", "offset_mean",
                  "offset_sd", "offset_min", "offset_max")},
        note=("per-KC v_th offsets calibrated only from each KC's own response "
              "rate on the separated encoding's odours; no labels, no task signal"),
    )
    C.write_json(config_out, cfg2)
    print(f"\nwrote {out} and {config_out}")
    print(f"final: active {final['active_frac']:.4f} "
          f"({final['active_per_state']:.0f}/odour), r>0.5 {final['n_r_gt_0_5']}, "
          f"r>0.3 {final['n_r_gt_0_3']}, silent {final['n_silent']}, "
          f"offsets {final['offset_min']:+.1f}..{final['offset_max']:+.1f} mV")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--out", default=str(HOMEO_JSON))
    ap.add_argument("--config-out", default=str(CONFIG_JSON))
    a = ap.parse_args(argv)
    calibrate(a.iterations, Path(a.out), Path(a.config_out))


if __name__ == "__main__":
    main()
