"""Per-Kenyon-cell homeostatic thresholds, calibrated from each cell's own firing.

Why
---

``outputs/plast/REPORT.md`` found that the fly's own dopamine-gated depression
rule *does* learn from chips (P(play) 0.53 -> 0.95 with a frozen control that does
not move) but learns one number for every odour, and located the cause in the
Kenyon-cell code: 188 of the ~211 KCs active per odour come from the pool that
responds to **more than half of all odours**, and those cells collect 70-75% of
every weight change. Credit assignment at KC -> MBON is a coincidence detector, so
a presynaptic population that is active for everything makes it odour-independent
by construction.

None of the existing knobs can fix that. ``apl_scale``, ``alpn_kc_scale``,
``kc_vth_offset_mv`` and ``kc_kc_scale`` are population-wide rescalings: they
change *how many* Kenyon cells cross threshold, never *which*, because the
between-KC spread in total input (SD 112.6 mV, set by in-degree) is 11.8x the
within-KC spread across odours. The ranking is fixed by the wiring and a uniform
shift slides the cut along it.

The mechanism
-------------

Real Kenyon cells are not uniform-threshold cells. KC excitability is under
homeostatic control, and the measured population consequence is the textbook
sparse code: an individual KC responds to roughly 5-10% of odours (Turner et al.
2008; Honegger et al. 2011; Lin et al. 2014). A synapse-count model inherits
topology and a sign; it cannot inherit each cell's set point any more than it can
inherit per-synapse efficacy. So this script calibrates it the way the animal
arrives at it: iteratively, from each cell's own measured response rate:

    r_i         = fraction of CALIBRATION odours for which KC i emits >= 1 spike
    offset_i   += k * log((r_i + eps) / (target + eps))          target = 0.07

and nothing else. The update sees only "how often did I fire". It never sees the
labels, the hand type, the score bucket, the reward, the MBONs, or the readout, so
it cannot smuggle the task into the encoding. Cells inside a deadband
(r in [0.04, 0.12]) are left alone; ``k`` is annealed so cells with a steep
0 -> 1 transition settle instead of ping-ponging.

Bounds
------

The brief's clamp is [-10, +30] mV. The lower end needs one biophysical
correction, applied per cell and reported: this kernel has ``v_rest = -52`` and a
fixed per-neuron ``v0 = -52 + U(0, 6.9)`` mV, so a threshold pushed below a cell's
own ``v0`` makes that cell fire on the first tick of *every* window regardless of
the odour. That is the exact pathology homeostasis exists to remove, so the floor
is ``max(-10, v0_i - V_TH + 0.25)``: "a spike threshold has to sit above the
cell's own resting potential". With 3,683 of 4,064 Kenyon cells silent at the
starting point, without this floor the loop drives almost the whole
population to -10 mV and manufactures ~3,600 always-on cells.

The odour set
-------------

Real relay patterns, from ``outputs/bc2/states.npz`` (the behaviour-cloning state
dump), first 32 columns = the ``features_v2`` relay block. Filtered to rows that
are inside a blind with a hand (block non-zero) **and** have the ``sel_none`` bit
on, so each pattern is a pattern the fly meets at a decision point, and
restricted to episodes >= 1000. Deduplicated ("responds to 5-10% of odours" is a
statement about the odour set, not about how often the game happens to deal each
one), then split stratified by hand type into 500 CALIBRATION and 500 HELD-OUT
patterns with a fixed seed. The bc2 episodes come from seeds 0-1359 (heuristic)
and 500000-501358 (random), which are disjoint from the plasticity calibration
(300000+), training (200000+) and evaluation (100000-100059) seeds by
construction, not just by the episode cut.

Run::

    python -m scripts.kc_homeo                 # 10 iterations, ~15 min
    python -m scripts.kc_homeo --resume        # continue where it stopped
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from flybalatro.brain import V_TH  # noqa: E402
from flybalatro.tuning import Tuning  # noqa: E402
from scripts import plast_common as K  # noqa: E402

ROOT = K.ROOT
OUT_DIR = ROOT / "outputs" / "plast2"
STATES = ROOT / "outputs" / "bc2" / "states.npz"
HOMEO_JSON = OUT_DIR / "kc_homeo.json"
CONFIG_JSON = OUT_DIR / "tuned_config.json"

#: Target per-KC odour response rate. The middle of the 5-10% the mushroom-body
#: literature reports for single Kenyon cells.
TARGET: float = 0.07
#: Softener in the log so r = 0 gives a finite step.
EPS: float = 0.02
#: Cells already inside this band are not touched (stops needless oscillation).
DEADBAND: Tuple[float, float] = (0.04, 0.12)
#: Gain per iteration, annealed.
K_SCHEDULE: Tuple[float, ...] = (4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0,
                                 0.75, 0.5)
#: The brief's clamp.
OFFSET_LO: float = -10.0
OFFSET_HI: float = 30.0
#: Largest move any single cell makes in one iteration, asymmetric on purpose.
#: *Lowering* a threshold recruits a cell, and because Kenyon cells inhibit each
#: other through APL every recruitment is also a change to every other cell's
#: input: move the whole population down several mV at once and the loop becomes
#: a sequence of unrelated network states instead of a relaxation. *Raising* one
#: is locally stabilising (it only removes that cell's own contribution), and it
#: is the direction that has the most ground to cover, since a cell answering to
#: every odour has to climb tens of mV.
MAX_STEP_DOWN_MV: float = 2.5
MAX_STEP_UP_MV: float = 4.0
#: Margin kept between a KC's threshold and its own fixed initial voltage.
V0_MARGIN_MV: float = 0.25
#: Convergence: stop when fewer than this fraction of KCs respond to > 30% of odours.
STOP_FRAC_R30: float = 0.02

#: Relay-block bit indices (see flybalatro.features_v2).
BIT_SEL_NONE = 26
N_RELAY = 32
#: Only episodes at or above this index are used.
EPISODE_MIN = 1000
#: Split sizes and seed.
N_CALIB = 500
N_GATE = 500
SPLIT_SEED = 20260913


# --------------------------------------------------------------------------- #
# the odour sets
# --------------------------------------------------------------------------- #
def relay_patterns(episode_min: int = EPISODE_MIN, n_calib: int = N_CALIB,
                   n_gate: int = N_GATE, seed: int = SPLIT_SEED,
                   states: Path = STATES) -> dict:
    """Distinct decision-point relay patterns from ``outputs/bc2``, split 500/500.

    Returns ``{calib, gate, meta}`` where ``calib`` / ``gate`` are
    ``float32[n, 32]`` and never share a pattern.
    """
    d = np.load(states, allow_pickle=True)
    bits = np.unpackbits(d["features_packed"], axis=1)[:, :N_RELAY]
    episode = np.asarray(d["episode"])
    keep = (bits.sum(axis=1) > 0) & (bits[:, BIT_SEL_NONE] == 1) & (episode >= episode_min)
    uniq = np.unique(bits[keep], axis=0)
    hand_type = uniq[:, :9].argmax(axis=1)

    # Stratified by hand type so both halves carry every class (Straight Flush
    # has only 15 distinct patterns in the whole dump).
    rng = np.random.default_rng(seed)
    calib_rows: List[int] = []
    gate_rows: List[int] = []
    for t in range(9):
        rows = np.flatnonzero(hand_type == t)
        rows = rows[rng.permutation(len(rows))]
        share = len(rows) / len(uniq)
        n_c = int(round(n_calib * share))
        n_g = int(round(n_gate * share))
        calib_rows.extend(rows[:n_c].tolist())
        gate_rows.extend(rows[n_c:n_c + n_g].tolist())
    # Top up / trim to the exact sizes from whatever is left over.
    used = set(calib_rows) | set(gate_rows)
    spare = [i for i in rng.permutation(len(uniq)).tolist() if i not in used]
    while len(calib_rows) < n_calib and spare:
        calib_rows.append(spare.pop())
    while len(gate_rows) < n_gate and spare:
        gate_rows.append(spare.pop())
    calib_rows = sorted(calib_rows[:n_calib])
    gate_rows = sorted(gate_rows[:n_gate])
    assert not (set(calib_rows) & set(gate_rows)), "calib / gate sets overlap"

    calib = uniq[calib_rows].astype(np.float32)
    gate = uniq[gate_rows].astype(np.float32)
    meta = dict(
        source=str(states.relative_to(ROOT)),
        columns="features_packed[:, :32] = the features_v2 relay block",
        filter=(f"relay block non-zero AND sel_none bit ({BIT_SEL_NONE}) on "
                f"AND episode >= {episode_min}"),
        rows_matching=int(keep.sum()),
        distinct_patterns=int(len(uniq)),
        deduplicated=True,
        dedup_note=("'responds to 5-10% of odours' is a statement about the odour "
                    "set, so each distinct pattern counts once rather than once "
                    "per time the game dealt it"),
        split_seed=int(seed),
        stratified_by="best hand type (relay bits 0-8)",
        n_calib=int(len(calib)), n_gate=int(len(gate)),
        n_spare=int(len(uniq) - len(calib) - len(gate)),
        calib_hand_type_counts=np.bincount(calib[:, :9].argmax(axis=1).astype(int),
                                          minlength=9).tolist(),
        gate_hand_type_counts=np.bincount(gate[:, :9].argmax(axis=1).astype(int),
                                          minlength=9).tolist(),
        calib_bucket_counts=np.bincount(calib[:, 28:32].argmax(axis=1).astype(int),
                                        minlength=4).tolist(),
        gate_bucket_counts=np.bincount(gate[:, 28:32].argmax(axis=1).astype(int),
                                       minlength=4).tolist(),
        seed_disjointness=("bc2 episodes come from seeds 0-1359 (heuristic teacher) "
                           "and 500000-501358 (random), disjoint from the "
                           "plasticity calib (300000+), train (200000+) and eval "
                           "(100000-100059) seeds by construction"),
    )
    return dict(calib=calib, gate=gate, meta=meta)


# --------------------------------------------------------------------------- #
# one measurement pass
# --------------------------------------------------------------------------- #
def base_config() -> dict:
    """The plast operating point (apl_scale 2, apl_kc_only, apl_mbon_scale 0.1)."""
    return json.loads(K.TUNED_CONFIG.read_text())


def config_with(cfg: dict, offsets: Optional[NDArray[np.float32]] = None,
                out_scale: Optional[NDArray[np.float32]] = None) -> dict:
    """A copy of ``cfg`` carrying the per-KC vectors in its ``tuning`` block."""
    out = json.loads(json.dumps(cfg))
    t = dict(out["tuning"])
    if offsets is not None:
        t["kc_vth_offsets"] = [round(float(x), 4) for x in offsets]
    if out_scale is not None:
        t["kc_mbon_out_scale"] = [round(float(x), 6) for x in out_scale]
    out["tuning"] = t
    return out


def build(cfg: dict, graph, offsets: Optional[NDArray[np.float32]] = None,
          out_scale: Optional[NDArray[np.float32]] = None) -> K.Setup:
    """A fresh Setup whose Brain was built through ``Tuning.apply`` with ``offsets``.

    Rebuilt rather than poked in place, so the code path is the same one the
    saved ``tuned_config.json`` will exercise in every later run.
    """
    return K.build_setup("real", graph=graph,
                         cfg=config_with(cfg, offsets, out_scale))


def kc_response(setup: K.Setup, patterns: NDArray[np.float32],
                collect: bool = False) -> dict:
    """Per-KC response rate over ``patterns``; optionally keep the spike matrix."""
    n = len(patterns)
    n_kc = len(setup.kc)
    hits = np.zeros(n_kc, np.int32)
    spikes = np.zeros(n_kc, np.int64)
    mat = np.zeros((n, n_kc), np.int16) if collect else None
    active_per = np.zeros(n, np.int32)
    for i, b in enumerate(patterns):
        c = setup.run_window(b)[setup.kc]
        act = c > 0
        hits += act
        spikes += c.astype(np.int64)
        active_per[i] = int(act.sum())
        if mat is not None:
            mat[i] = c
    r = hits / max(1, n)
    return dict(rate=r, hits=hits, spikes=spikes, matrix=mat,
                active_per_state=active_per)


def rate_stats(r: NDArray[np.float64], active_per: NDArray[np.int32],
               n_kc: int) -> dict:
    return dict(
        n_kc=int(n_kc),
        mean_rate=round(float(r.mean()), 5),
        active_frac=round(float(active_per.mean() / n_kc), 5),
        active_per_state=round(float(active_per.mean()), 1),
        active_per_state_sd=round(float(active_per.std()), 1),
        frac_r_gt_0_5=round(float((r > 0.5).mean()), 5),
        n_r_gt_0_5=int((r > 0.5).sum()),
        frac_r_gt_0_3=round(float((r > 0.3).mean()), 5),
        n_r_gt_0_3=int((r > 0.3).sum()),
        n_r_eq_1=int((r >= 1.0).sum()),
        n_silent=int((r == 0.0).sum()),
        n_responsive=int((r > 0.0).sum()),
        n_in_band=int(((r >= DEADBAND[0]) & (r <= DEADBAND[1])).sum()),
        median_rate_of_responsive=round(float(np.median(r[r > 0])), 5)
        if (r > 0).any() else 0.0,
    )


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def calibrate(n_iter: int, resume: bool, out: Path = HOMEO_JSON,
              gain_override: Optional[float] = None,
              max_step_down: float = MAX_STEP_DOWN_MV,
              max_step_up: float = MAX_STEP_UP_MV) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    graph = K.C.load(5)
    cfg = base_config()
    sets = relay_patterns()
    calib = sets["calib"]

    # A brain with no offsets, only to read the fixed per-neuron v0 the kernel
    # will use (deterministic in brain_seed / v_jitter) and the KC count.
    setup0 = build(cfg, graph)
    n_kc = len(setup0.kc)
    v0_kc = np.asarray(setup0.brain._v0[setup0.kc], np.float64)
    floor = np.maximum(OFFSET_LO, v0_kc - V_TH + V0_MARGIN_MV).astype(np.float64)
    del setup0

    history: List[dict] = []
    offsets = np.zeros(n_kc, np.float64)
    prev_sign = np.zeros(n_kc, np.int8)
    it0 = 0
    if resume and out.exists():
        prev = json.loads(out.read_text())
        if int(prev.get("n_kc", -1)) == n_kc:
            offsets = np.asarray(prev["offsets"], np.float64)
            history = list(prev.get("iterations", []))
            prev_sign = np.asarray(prev.get("last_update_sign", prev_sign), np.int8)
            it0 = len(history)
            print(f"resumed at iteration {it0}")

    payload: dict = dict(
        target=TARGET, eps=EPS, deadband=list(DEADBAND),
        k_schedule=list(K_SCHEDULE), offset_clamp=[OFFSET_LO, OFFSET_HI],
        max_step_down_mv=max_step_down, max_step_up_mv=max_step_up,
        gain_override=gain_override,
        v0_margin_mv=V0_MARGIN_MV,
        v0_floor_note=("per-cell lower clamp max(-10, v0_i - V_TH + 0.25): a "
                       "threshold below the cell's own fixed v0 makes it spike on "
                       "tick 1 for every odour"),
        floor_mv=dict(min=round(float(floor.min()), 4),
                      max=round(float(floor.max()), 4),
                      n_above_spec_lo=int((floor > OFFSET_LO + 1e-9).sum())),
        stop_frac_r_gt_0_3=STOP_FRAC_R30,
        n_kc=int(n_kc), patterns=sets["meta"], base_tuning=cfg["tuning"],
        rule=("offset_i += k * log((r_i + eps) / (target + eps)); r_i measured on "
              "the CALIBRATION patterns only; no labels, no reward, no readout"),
    )

    for it in range(it0, n_iter):
        t0 = time.time()
        setup = build(cfg, graph, offsets.astype(np.float32))
        resp = kc_response(setup, calib)
        r = resp["rate"]
        st = rate_stats(r, resp["active_per_state"], n_kc)

        gain = (float(gain_override) if gain_override is not None
                else K_SCHEDULE[min(it, len(K_SCHEDULE) - 1)])
        step = gain * np.log((r + EPS) / (TARGET + EPS))
        step = np.clip(step, -max_step_down, max_step_up)
        in_band = (r >= DEADBAND[0]) & (r <= DEADBAND[1])
        step[in_band] = 0.0
        sign = np.sign(step).astype(np.int8)
        moved = sign != 0
        flipped = moved & (prev_sign != 0) & (sign != prev_sign)
        new = np.clip(offsets + step, floor, OFFSET_HI)

        row = dict(
            iteration=int(it), gain=float(gain),
            **st,
            n_moved=int(moved.sum()),
            frac_sign_flipped=round(float(flipped.sum() / max(1, moved.sum())), 4),
            n_at_floor=int((new <= floor + 1e-6).sum()),
            n_at_floor_and_silent=int(((new <= floor + 1e-6) & (r == 0.0)).sum()),
            n_at_ceiling=int((new >= OFFSET_HI - 1e-6).sum()),
            n_at_ceiling_and_r_gt_0_5=int(((new >= OFFSET_HI - 1e-6) & (r > 0.5)).sum()),
            max_step_down_mv=float(max_step_down),
            max_step_up_mv=float(max_step_up),
            offset_mean=round(float(offsets.mean()), 4),
            offset_sd=round(float(offsets.std()), 4),
            offset_min=round(float(offsets.min()), 4),
            offset_max=round(float(offsets.max()), 4),
            seconds=round(time.time() - t0, 1),
        )
        history.append(row)
        print(f"[{it:2d}] k={gain:<4g} active {st['active_frac']:.4f} "
              f"({st['active_per_state']:.0f}/state)  r>0.5 {st['n_r_gt_0_5']:4d} "
              f"({st['frac_r_gt_0_5']:.4f})  r>0.3 {st['n_r_gt_0_3']:4d} "
              f"({st['frac_r_gt_0_3']:.4f})  in-band {st['n_in_band']:4d}  "
              f"silent {st['n_silent']:4d}  flips {row['frac_sign_flipped']:.2f}  "
              f"{row['seconds']:.0f}s", flush=True)

        del setup
        converged = bool(st["frac_r_gt_0_3"] < STOP_FRAC_R30 and it >= 3)
        row["converged"] = converged
        row["update_applied"] = not converged
        if converged:
            # Ship the offsets whose statistics are *on the record*, not the ones
            # the last update would have produced. The update after a converged
            # measurement has never been measured, and at this gain a 1-2 mV move
            # is enough to push a cell across threshold, so applying it would mean
            # shipping an operating point nobody looked at.
            print(f"converged at iteration {it}: frac(r>0.3) = "
                  f"{st['frac_r_gt_0_3']:.4f} < {STOP_FRAC_R30} "
                  "(last update not applied)")
            payload.update(iterations=history,
                           offsets=[round(float(x), 4) for x in offsets],
                           last_update_sign=[int(x) for x in prev_sign],
                           wall_seconds=round(time.time() - t_start, 1))
            K.write_json(out, payload)
            break

        offsets = new
        prev_sign = sign
        payload.update(iterations=history,
                       offsets=[round(float(x), 4) for x in offsets],
                       last_update_sign=[int(x) for x in prev_sign],
                       wall_seconds=round(time.time() - t_start, 1))
        K.write_json(out, payload)

    # One final measurement at the offsets we are going to ship.
    setup = build(cfg, graph, offsets.astype(np.float32))
    resp = kc_response(setup, calib)
    final = rate_stats(resp["rate"], resp["active_per_state"], n_kc)
    final["offset_mean"] = round(float(offsets.mean()), 4)
    final["offset_sd"] = round(float(offsets.std()), 4)
    final["offset_min"] = round(float(offsets.min()), 4)
    final["offset_max"] = round(float(offsets.max()), 4)
    final["n_at_floor"] = int((offsets <= floor + 1e-6).sum())
    final["n_at_ceiling"] = int((offsets >= OFFSET_HI - 1e-6).sum())
    payload["final_calib"] = final
    payload["final_calib_rate"] = [round(float(x), 4) for x in resp["rate"]]
    payload["offsets"] = [round(float(x), 4) for x in offsets]
    payload["tuning_info"] = setup.brain.tuning_info.get("kc_vth_offsets", {})
    payload["wall_seconds"] = round(time.time() - t_start, 1)
    K.write_json(out, payload)

    # The shipped fly.
    cfg2 = config_with(cfg, offsets.astype(np.float32))
    cfg2["derived_from"] = "outputs/plast/tuned_config.json"
    cfg2["label"] = Tuning.from_dict(cfg2["tuning"]).label()
    cfg2["kc_homeostasis"] = dict(
        script="scripts/kc_homeo.py", detail=str(out.relative_to(ROOT)),
        target=TARGET, iterations=len(history),
        calibration_patterns=int(len(calib)),
        patterns=sets["meta"],
        summary={k: final[k] for k in
                 ("active_frac", "active_per_state", "frac_r_gt_0_5",
                  "frac_r_gt_0_3", "n_silent", "n_responsive", "offset_mean",
                  "offset_sd", "offset_min", "offset_max")},
        note=("per-KC v_th offsets calibrated only from each KC's own response "
              "rate on real relay patterns; no labels, no task signal"),
    )
    K.write_json(CONFIG_JSON, cfg2)
    print(f"\nwrote {out} and {CONFIG_JSON}")
    print(f"final: active {final['active_frac']:.4f} "
          f"({final['active_per_state']:.0f}/state), r>0.5 {final['n_r_gt_0_5']} "
          f"({final['frac_r_gt_0_5']:.4f}), r>0.3 {final['n_r_gt_0_3']} "
          f"({final['frac_r_gt_0_3']:.4f}), silent {final['n_silent']}, "
          f"offsets {final['offset_min']:+.1f}..{final['offset_max']:+.1f} mV")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gain", type=float, default=None,
                    help="override the annealed schedule for the iterations run "
                         "by this invocation (recorded per iteration)")
    ap.add_argument("--max-step-down", type=float, default=MAX_STEP_DOWN_MV)
    ap.add_argument("--max-step-up", type=float, default=MAX_STEP_UP_MV)
    ap.add_argument("--out", default=str(HOMEO_JSON))
    a = ap.parse_args(argv)
    calibrate(a.iterations, a.resume, Path(a.out), a.gain, a.max_step_down,
              a.max_step_up)


if __name__ == "__main__":
    main()
