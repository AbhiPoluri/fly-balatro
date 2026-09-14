"""Dopamine pulse calibration: make one punishment pulse worth one reward pulse.

This is **model calibration, not reward shaping.** What the game gives the fly is
unchanged: a reward pulse after a play that paid its way, a punishment pulse after
a play that lost the blind, nothing after a discard, at whatever frequency ante 1
happens to produce (``outputs/plast/REPORT.md``: 692 reward pulses against 230
punishments per run). What is being calibrated is the *per-pulse gain of the two
arms of our own rule*, which in the previous round differed by 4x for a reason
that has nothing to do with dopamine:

* the reward arm depresses KC -> (avoid, PAM) synapses. There are 24 avoid MBONs, so
  one lost spike there moves ``play_drive`` by ``1/24 * 20`` = 0.83 Hz.
* the punishment arm depresses KC -> (approach, PPL1) synapses. There are 66
  approach MBONs, so one lost spike moves it by ``1/66 * 20`` = 0.30 Hz.

The asymmetry is a bookkeeping artefact of the pool sizes the neurotransmitter
rule produces, multiplied by whatever the two arms' edge counts happen to be. A
real PPL1 pulse is not 4x weaker than a real PAM pulse, and nothing in the
connectome says it should be. So ``eta_punish`` is scaled until a single
punishment pulse moves the paired odour's drive by the same amount a single reward
pulse does, measured on the same 8-odour conditioning panel the specificity gate
uses. ``eta_reward`` keeps the protocol's values (0.02, 0.05) and the result is
reported as a ratio applied to both.

Measurement. One pulse moves the drive by less than the 1.1 Hz quantisation
quantum of the MBON pool means, so the single-pulse number is mostly zeros. The
trajectory is therefore run out to 20 pulses per odour and ``|delta| / N`` is
reported at N = 1, 5 and 20; the match is made on N = 5 unless it is degenerate,
and all three are in the output.

Run: ``python -m scripts.plast2_eta``   (~5 min)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from scripts import kc_homeo as H  # noqa: E402
from scripts import plast_common as K  # noqa: E402
from scripts.plast_probe import (  # noqa: E402
    collect_calibration_hands,
    hand_bits,
    pick_odor_panel,
)

OUT_DIR = H.OUT_DIR
ETA_JSON = OUT_DIR / "eta_calib.json"

N_PULSES = 20
MEASURE_AT = (1, 5, 20)
MATCH_AT = 5
ETA_REWARDS = (0.02, 0.05)
PUNISH_GRID = (0.05, 0.07, 0.1, 0.14, 0.2, 0.28, 0.4, 0.6, 0.9)
ETA_MAX = 0.9


def trajectory(setup: K.Setup, dec, bits: NDArray[np.float32], kind: str,
               eta: float, n_pulses: int = N_PULSES) -> List[float]:
    """``play_drive`` after 0, 1, ... n_pulses pulses of ``kind`` on one odour."""
    setup.plast.reset_weights()
    out: List[float] = []
    for t in range(n_pulses + 1):
        counts = setup.run_window(bits)
        out.append(dec.play_drive(counts))
        if t < n_pulses:
            setup.plast.deliver(kind, counts[setup.kc], eta)
    setup.plast.reset_weights()
    return out


def per_pulse(setup: K.Setup, dec, panel_bits: NDArray[np.float32],
              panel_idx: Sequence[int], kind: str, eta: float,
              n_pulses: int = N_PULSES, measure_at: Sequence[int] = MEASURE_AT) -> dict:
    """Mean |delta drive| per pulse on the paired odour, over the whole panel."""
    trajs = [trajectory(setup, dec, panel_bits[i], kind, eta, n_pulses)
             for i in panel_idx]
    out: dict = dict(kind=kind, eta=float(eta), n_odors=len(trajs),
                     n_pulses=int(n_pulses),
                     trajectories=[[round(x, 4) for x in t] for t in trajs])
    for n in [m for m in measure_at if m <= n_pulses]:
        d = np.array([t[n] - t[0] for t in trajs])
        out[f"n{n}"] = dict(
            mean_delta=round(float(d.mean()), 5),
            mean_abs_delta=round(float(np.abs(d).mean()), 5),
            abs_per_pulse=round(float(np.abs(d).mean() / n), 6),
            n_nonzero=int((d != 0).sum()),
            sd=round(float(d.std()), 5),
        )
    return out


def cumulative_max(values: Sequence[float]) -> List[float]:
    """Monotone envelope of a noisy eta -> effect curve (eta must be ascending).

    The measured per-pulse |delta| is quantised to ~1.1 Hz steps, so a 9-point
    grid is not monotone even though the underlying relation is. The running
    maximum is the cheapest monotone summary that does not invent values the
    measurement never produced.
    """
    out: List[float] = []
    best = -np.inf
    for v in values:
        best = max(best, float(v))
        out.append(best)
    return out


def interpolate_eta(grid: Sequence[float], values: Sequence[float],
                    target: float) -> Tuple[float, str]:
    """log-log interpolate ``eta`` so the measured per-pulse effect hits ``target``."""
    g = np.asarray(grid, float)
    v = np.asarray(values, float)
    ok = v > 0
    if not ok.any():
        return float("nan"), "degenerate: every punishment eta gave zero effect"
    g, v = g[ok], v[ok]
    order = np.argsort(g)
    g, v = g[order], np.asarray(cumulative_max(v[order]), float)
    keep = np.concatenate([[True], np.diff(v) > 0])
    g, v = g[keep], v[keep]
    if len(v) == 1 or target <= v[0]:
        return float(g[0]), f"target {target:.5g} at or below the grid minimum"
    if target >= v[-1]:
        return float(g[-1]), f"target {target:.5g} at or above the grid maximum"
    eta = float(np.exp(np.interp(np.log(target), np.log(v), np.log(g))))
    return eta, "log-log interpolation on the monotone envelope of the grid"


def choose(payload: dict, match_at: int, n_odors: int) -> dict:
    """One ratio, applied to every protocol ``eta_reward``.

    The grid brackets the reward target only for the larger ``eta_reward``; at
    0.02 the target falls below what the smallest punishment eta on the grid
    produces, and the clamped answer there is a grid artefact, not a measurement.
    Depression is multiplicative in ``1 - eta * elig``, so the per-pulse effect is
    linear in eta to first order and the honest summary is a single ratio measured
    where the grid does bracket the target, applied to both.
    """
    match = payload["match"]
    rows = {}
    for r in payload["reward"]:
        key = f"n{match_at}_eta_reward{r['eta']:g}"
        m = match[key]
        rows[f"{r['eta']:g}"] = dict(
            eta_reward=r["eta"],
            eta_punish_raw=round(float(m["eta_punish"]), 4),
            ratio_raw=m["ratio"], how=m["how"],
            bracketed=bool("interpolation" in str(m["how"])),
            target_abs_per_pulse=m["target_abs_per_pulse"],
        )
    bracketed = [v for v in rows.values() if v["bracketed"]]
    ref = max(bracketed, key=lambda v: v["eta_reward"]) if bracketed else \
        max(rows.values(), key=lambda v: v["eta_reward"])
    ratio = float(ref["ratio_raw"])
    for v in rows.values():
        v["ratio"] = round(ratio, 4)
        v["eta_punish"] = round(min(ratio * v["eta_reward"], ETA_MAX), 4)
        v["capped"] = bool(ratio * v["eta_reward"] > ETA_MAX)
        v["matched_at_n_pulses"] = int(match_at)
        v["n_odors"] = int(n_odors)
        v["ratio_reference_eta_reward"] = ref["eta_reward"]
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--n-odors", type=int, default=8,
                    help="conditioning-panel size; 8 is the gate's panel, and a "
                         "wider one averages down the quantisation noise")
    ap.add_argument("--n-pulses", type=int, default=N_PULSES)
    ap.add_argument("--match-at", type=int, default=MATCH_AT)
    ap.add_argument("--recompute", action="store_true",
                    help="recompute `chosen` from an existing --out file, no "
                         "simulation")
    ap.add_argument("--out", default=str(ETA_JSON))
    a = ap.parse_args(argv)

    if a.recompute:
        payload = json.loads(Path(a.out).read_text())
        payload["chosen"] = choose(payload, int(a.match_at),
                                   int(payload.get("n_odors", len(payload["panel"]))))
        K.write_json(Path(a.out), payload)
        print(json.dumps(payload["chosen"], indent=2))
        return

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = K.C.load(5)
    cfg = json.loads(H.CONFIG_JSON.read_text())
    setup = K.build_setup("real", graph=graph, cfg=cfg)
    dec = K.make_decider(setup, explore_floor=0.1)

    recs = collect_calibration_hands(a.n_calib, K.CALIB_SEED0, rng_seed=7)
    panel_bits = hand_bits(recs)
    panel_idx = pick_odor_panel(recs, a.n_odors)
    # The decider's bias/temperature, frozen the same way the runs freeze them.
    raw = [dec.raw_drive(setup.run_window(b)) for b in panel_bits]
    cal = dec.calibrate(raw, explore=0.20)

    n_a, n_v = len(dec.approach), len(dec.avoid)
    scale = 1000.0 / setup.window_ms
    payload: dict = dict(
        note=("model calibration of the per-pulse gain of our own rule's two arms; "
              "reward/punishment FREQUENCY is whatever the game gives"),
        pools=dict(n_approach=n_a, n_avoid=n_v,
                   reward_arm_edges=setup.plast.targets["reward"].n,
                   punish_arm_edges=setup.plast.targets["punish"].n,
                   reward_quantum_hz=round(scale / n_v, 4),
                   punish_quantum_hz=round(scale / n_a, 4)),
        calibration=cal, decider=dec.to_dict(),
        panel=[dict(index=int(i), bucket=recs[i]["bucket_name"],
                    hand_type=recs[i]["best_type_name"]) for i in panel_idx],
        n_pulses=int(a.n_pulses), n_odors=len(panel_idx),
        measure_at=[n for n in MEASURE_AT if n <= a.n_pulses],
        match_at=int(a.match_at),
        eta_rewards=list(ETA_REWARDS), punish_grid=list(PUNISH_GRID),
        reward=[], punish=[],
    )

    for eta in ETA_REWARDS:
        r = per_pulse(setup, dec, panel_bits, panel_idx, "reward", eta,
                      a.n_pulses, payload["measure_at"])
        payload["reward"].append(r)
        print(f"reward eta {eta:g}: per-pulse |d| "
              + "  ".join(f"N{n}={r[f'n{n}']['abs_per_pulse']:.4f}"
                          for n in payload["measure_at"]), flush=True)
    for eta in PUNISH_GRID:
        r = per_pulse(setup, dec, panel_bits, panel_idx, "punish", eta,
                      a.n_pulses, payload["measure_at"])
        payload["punish"].append(r)
        print(f"punish eta {eta:g}: per-pulse |d| "
              + "  ".join(f"N{n}={r[f'n{n}']['abs_per_pulse']:.4f}"
                          for n in payload["measure_at"]), flush=True)

    grid = [r["eta"] for r in payload["punish"]]
    match: Dict[str, dict] = {}
    for n in payload["measure_at"]:
        vals = [r[f"n{n}"]["abs_per_pulse"] for r in payload["punish"]]
        for r in payload["reward"]:
            target = r[f"n{n}"]["abs_per_pulse"]
            eta_p, how = interpolate_eta(grid, vals, target)
            eta_p_c = min(eta_p, ETA_MAX) if np.isfinite(eta_p) else eta_p
            match[f"n{n}_eta_reward{r['eta']:g}"] = dict(
                n_pulses=n, eta_reward=r["eta"], target_abs_per_pulse=target,
                punish_grid_values=vals, eta_punish=eta_p_c,
                capped=bool(np.isfinite(eta_p) and eta_p > ETA_MAX),
                ratio=round(eta_p_c / r["eta"], 4) if np.isfinite(eta_p_c) else None,
                how=how)
    payload["match"] = match

    payload["chosen"] = choose(payload, int(a.match_at), len(panel_idx))
    payload["wall_seconds"] = round(time.time() - t0, 1)
    K.write_json(Path(a.out), payload)
    print("\nchosen:", json.dumps(chosen, indent=2))
    print(f"wrote {a.out} in {payload['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
