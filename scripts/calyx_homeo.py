"""Per-KC homeostatic thresholds, recalibrated independently for every wiring.

The fairness requirement of the calyx control: *the same label-free calibration
applied to every condition*. ``scripts/kc_homeo.py`` calibrates per-Kenyon-cell
``v_th`` offsets from each cell's own response rate on real relay patterns --

    r_i        = fraction of CALIBRATION odours for which KC i emits >= 1 spike
    offset_i  += k * log((r_i + eps) / (target + eps))          target = 0.07

-- with no reference to labels, reward or readout. This script runs exactly that
loop, with exactly those constants and exactly that odour split (imported from
``scripts.kc_homeo``), once per wiring condition, on that condition's own graph.

Two deliberate differences from ``scripts/kc_homeo.py``, both recorded in the
output JSON:

* the base tuning is ``outputs/mb2/tuned_config.json`` (``apl_scale = 2``, tonic
  30 mV), the v3 / ``outputs/bc3`` operating point this control is meant to
  speak to, rather than ``outputs/plast/tuned_config.json`` (which adds
  ``apl_kc_only`` and ``apl_mbon_scale`` for the plasticity experiment);
* the brain is built directly (``Brain`` + ``GlomerularMap``) instead of through
  ``plast_common.build_setup``, because the plastic KC -> MBON synapse table is
  not used here and costs memory.

Run::

    python -m scripts.calyx_homeo --conditions real,rw1
    python -m scripts.calyx_homeo --conditions rw2,rw3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent
for _p in (str(ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flybalatro.brain import V_TH, Brain  # noqa: E402
from flybalatro.encode import GlomerularMap  # noqa: E402
from flybalatro.tuning import Tuning  # noqa: E402
from scripts import calyx_common as K  # noqa: E402
from scripts import kc_homeo as H  # noqa: E402

OUT_DIR = K.OUT_DIR
MB2_CONFIG = ROOT / "outputs" / "mb2" / "tuned_config.json"
N_ITER = 12


class Fly:
    """The minimum ``kc_homeo.kc_response`` needs: ``run_window`` and ``kc``."""

    def __init__(self, graph, cfg: dict, offsets=None):
        self.graph = graph
        gloms = [b["glomerulus"] for b in cfg["encoding"]["assignment"]]
        self.gm = GlomerularMap(graph, gloms,
                                drive_mv=float(cfg["drive"]["tonic_mv"]))
        t = dict(cfg["tuning"])
        if offsets is not None:
            t["kc_vth_offsets"] = [float(x) for x in offsets]
        self.brain = Brain(graph, seed=int(cfg["brain_seed"]),
                           v_jitter=float(cfg["v_jitter_mv"]),
                           tuning=Tuning.from_dict(t))
        self.kc = graph.kc_indices()
        self.window_ms = float(cfg["window_ms"])

    def run_window(self, bits):
        self.brain.reset()
        counts, _ = self.brain.step(self.gm.drive(np.asarray(bits, np.float32)[:32]),
                                    self.window_ms)
        return counts


def base_config() -> dict:
    return json.loads(MB2_CONFIG.read_text())


def homeo_path(cond: str, out_dir: Path = OUT_DIR) -> Path:
    return out_dir / f"homeo_{cond}.json"


def calibrate(cond: str, n_iter: int = N_ITER, out_dir: Path = OUT_DIR,
              resume: bool = True) -> dict:
    out = homeo_path(cond, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    graph = K.graph_for(cond)
    cfg = base_config()
    sets = H.relay_patterns()
    calib = sets["calib"]

    fly0 = Fly(graph, cfg)
    n_kc = len(fly0.kc)
    v0_kc = np.asarray(fly0.brain._v0[fly0.kc], np.float64)
    floor = np.maximum(H.OFFSET_LO, v0_kc - V_TH + H.V0_MARGIN_MV)
    del fly0

    offsets = np.zeros(n_kc, np.float64)
    prev_sign = np.zeros(n_kc, np.int8)
    history: List[dict] = []
    it0 = 0
    if resume and out.exists():
        prev = json.loads(out.read_text())
        if int(prev.get("n_kc", -1)) == n_kc and not prev.get("complete"):
            offsets = np.asarray(prev["offsets"], np.float64)
            history = list(prev.get("iterations", []))
            prev_sign = np.asarray(prev.get("last_update_sign", prev_sign), np.int8)
            it0 = len(history)
            print(f"[{cond}] resumed at iteration {it0}", flush=True)
        elif prev.get("complete"):
            print(f"[{cond}] already complete", flush=True)
            return prev

    payload: dict = dict(
        condition=cond, complete=False,
        procedure="scripts/kc_homeo.py, constants and odour split imported verbatim",
        base_config=str(MB2_CONFIG.relative_to(ROOT)),
        base_tuning=cfg["tuning"],
        deviation_from_kc_homeo=(
            "base tuning is outputs/mb2/tuned_config.json (the v3 / bc3 operating "
            "point) rather than outputs/plast/tuned_config.json"),
        target=H.TARGET, eps=H.EPS, deadband=list(H.DEADBAND),
        k_schedule=list(H.K_SCHEDULE), offset_clamp=[H.OFFSET_LO, H.OFFSET_HI],
        max_step_down_mv=H.MAX_STEP_DOWN_MV, max_step_up_mv=H.MAX_STEP_UP_MV,
        v0_margin_mv=H.V0_MARGIN_MV, stop_frac_r_gt_0_3=H.STOP_FRAC_R30,
        n_iter_budget=int(n_iter), n_kc=int(n_kc), patterns=sets["meta"],
        rule=("offset_i += k * log((r_i + eps) / (target + eps)); r_i measured on "
              "the CALIBRATION patterns only; no labels, no reward, no readout"),
        graph_provenance=graph.meta.get("calyx_rewiring", "real MaleCNS v1.0"),
    )

    converged = False
    for it in range(it0, n_iter):
        t0 = time.time()
        fly = Fly(graph, cfg, offsets.astype(np.float32))
        resp = H.kc_response(fly, calib)
        r = resp["rate"]
        st = H.rate_stats(r, resp["active_per_state"], n_kc)
        gain = H.K_SCHEDULE[min(it, len(H.K_SCHEDULE) - 1)]
        step = np.clip(gain * np.log((r + H.EPS) / (H.TARGET + H.EPS)),
                       -H.MAX_STEP_DOWN_MV, H.MAX_STEP_UP_MV)
        step[(r >= H.DEADBAND[0]) & (r <= H.DEADBAND[1])] = 0.0
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
                   seconds=round(time.time() - t0, 1))
        history.append(row)
        print(f"[{cond}][{it:2d}] k={gain:<4g} active {st['active_frac']:.4f} "
              f"({st['active_per_state']:.0f}/state) r>0.5 {st['n_r_gt_0_5']:4d} "
              f"r>0.3 {st['n_r_gt_0_3']:4d} in-band {st['n_in_band']:4d} "
              f"silent {st['n_silent']:4d} {row['seconds']:.0f}s", flush=True)
        del fly
        converged = bool(st["frac_r_gt_0_3"] < H.STOP_FRAC_R30 and it >= 3)
        row["converged"] = converged
        row["update_applied"] = not converged
        if converged:
            print(f"[{cond}] converged at iteration {it} "
                  f"(frac r>0.3 = {st['frac_r_gt_0_3']:.4f}); last update not applied",
                  flush=True)
            break
        offsets = new
        prev_sign = sign
        payload.update(iterations=history,
                       offsets=[round(float(x), 4) for x in offsets],
                       last_update_sign=[int(x) for x in prev_sign])
        _write(out, payload)

    fly = Fly(graph, cfg, offsets.astype(np.float32))
    resp = H.kc_response(fly, calib)
    final = H.rate_stats(resp["rate"], resp["active_per_state"], n_kc)
    payload.update(
        complete=True, converged=bool(converged), iterations=history,
        n_iterations_run=len(history),
        offsets=[round(float(x), 4) for x in offsets],
        last_update_sign=[int(x) for x in prev_sign],
        final_calib=final,
        final_calib_rate=[round(float(x), 4) for x in resp["rate"]],
        offset_summary=dict(mean=round(float(offsets.mean()), 4),
                            sd=round(float(offsets.std()), 4),
                            min=round(float(offsets.min()), 4),
                            max=round(float(offsets.max()), 4),
                            n_at_floor=int((offsets <= floor + 1e-6).sum()),
                            n_at_ceiling=int((offsets >= H.OFFSET_HI - 1e-6).sum())),
        wall_seconds=round(time.time() - t_start, 1),
    )
    _write(out, payload)
    cfg2 = H.config_with(cfg, offsets.astype(np.float32))
    cfg2["derived_from"] = str(MB2_CONFIG.relative_to(ROOT))
    cfg2["calyx_condition"] = cond
    cfg2["label"] = Tuning.from_dict(cfg2["tuning"]).label()
    _write(out_dir / f"tuned_{cond}.json", cfg2)
    print(f"[{cond}] wrote {out.name} and tuned_{cond}.json  "
          f"active {final['active_frac']:.4f} r>0.3 {final['frac_r_gt_0_3']:.4f} "
          f"silent {final['n_silent']} in {payload['wall_seconds']/60:.1f} min",
          flush=True)
    return payload


def _write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float) + "\n")
    tmp.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conditions", default="real,rw1,rw2,rw3")
    ap.add_argument("--iterations", type=int, default=N_ITER)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    for cond in [c.strip() for c in a.conditions.split(",") if c.strip()]:
        calibrate(cond, a.iterations, Path(a.out_dir))


if __name__ == "__main__":
    main()
