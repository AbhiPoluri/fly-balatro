"""Reward/punishment per-pulse gain for the **separated** encoding fly.

Model calibration, not reward shaping, and the same procedure v2 used
(``scripts/plast2_eta.py``): the *frequency* of dopamine is whatever ante 1
gives, and only the per-pulse gain of the two arms of our own rule is matched,
because the neurotransmitter valence rule gives 66 approach MBONs against 24
avoid ones and one lost spike therefore moves ``play_drive`` by 0.30 Hz on one
arm and 0.83 Hz on the other. The separated encoding changes which Kenyon cells
are eligible, so the match has to be measured again on that fly.

Reuses ``plast2_eta``'s measurement (``per_pulse``), its monotone-envelope
interpolation and its ``choose``; only the setup and the output path differ.

Run::

    python -m scripts.plast3_eta          # ~6 min
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from scripts import plast2_eta as E
from scripts import plast3_common as C
from scripts import plast_common as K
from scripts.plast_probe import collect_calibration_hands, hand_bits, pick_odor_panel

ETA_JSON = C.OUT_DIR / "eta_calib_sep.json"
CONFIG = C.OUT_DIR / "tuned_config_sep.json"


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--n-odors", type=int, default=23)
    ap.add_argument("--n-pulses", type=int, default=8)
    ap.add_argument("--match-at", type=int, default=E.MATCH_AT)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--out", default=str(ETA_JSON))
    a = ap.parse_args(argv)

    t0 = time.time()
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = K.C.load(5)
    cfg = json.loads(Path(a.config).read_text())
    setup = C.build_setup3(C.ENC_SEPARATED, cfg, graph)
    dec = K.make_decider(setup, explore_floor=C.EXPLORE_FLOOR3)

    recs = collect_calibration_hands(a.n_calib, K.CALIB_SEED0, rng_seed=7)
    panel_bits = hand_bits(recs)
    panel_idx = pick_odor_panel(recs, a.n_odors)
    raw = [dec.raw_drive(setup.run_window(b)) for b in panel_bits]
    cal = dec.calibrate(raw, explore=0.20)

    scale = 1000.0 / setup.window_ms
    measure_at = [n for n in E.MEASURE_AT if n <= a.n_pulses]
    payload: dict = dict(
        encoding=C.ENC_SEPARATED, config=C.rel_path(Path(a.config)),
        note=("per-pulse gain match for the separated-encoding fly; reward and "
              "punishment FREQUENCY untouched"),
        pools=dict(n_approach=len(dec.approach), n_avoid=len(dec.avoid),
                   reward_arm_edges=setup.plast.targets["reward"].n,
                   punish_arm_edges=setup.plast.targets["punish"].n,
                   reward_quantum_hz=round(scale / len(dec.avoid), 4),
                   punish_quantum_hz=round(scale / len(dec.approach), 4)),
        calibration=cal, decider=dec.to_dict(),
        panel=[dict(index=int(i), bucket=recs[i]["bucket_name"],
                    hand_type=recs[i]["best_type_name"]) for i in panel_idx],
        panel_note=("the separated encoding sees only hand type x bucket, so this "
                    "panel is close to its entire odour universe"),
        n_pulses=int(a.n_pulses), n_odors=len(panel_idx),
        measure_at=measure_at, match_at=int(a.match_at),
        eta_rewards=list(E.ETA_REWARDS), punish_grid=list(E.PUNISH_GRID),
        reward=[], punish=[],
    )

    for eta in E.ETA_REWARDS:
        r = E.per_pulse(setup, dec, panel_bits, panel_idx, "reward", eta,
                        a.n_pulses, measure_at)
        payload["reward"].append(r)
        print(f"reward eta {eta:g}: " + "  ".join(
            f"N{n}={r[f'n{n}']['abs_per_pulse']:.4f}" for n in measure_at), flush=True)
    for eta in E.PUNISH_GRID:
        r = E.per_pulse(setup, dec, panel_bits, panel_idx, "punish", eta,
                        a.n_pulses, measure_at)
        payload["punish"].append(r)
        print(f"punish eta {eta:g}: " + "  ".join(
            f"N{n}={r[f'n{n}']['abs_per_pulse']:.4f}" for n in measure_at), flush=True)

    grid = [r["eta"] for r in payload["punish"]]
    match: Dict[str, dict] = {}
    for n in measure_at:
        vals = [r[f"n{n}"]["abs_per_pulse"] for r in payload["punish"]]
        for r in payload["reward"]:
            target = r[f"n{n}"]["abs_per_pulse"]
            eta_p, how = E.interpolate_eta(grid, vals, target)
            eta_p_c = min(eta_p, E.ETA_MAX) if np.isfinite(eta_p) else eta_p
            match[f"n{n}_eta_reward{r['eta']:g}"] = dict(
                n_pulses=n, eta_reward=r["eta"], target_abs_per_pulse=target,
                punish_grid_values=vals, eta_punish=eta_p_c,
                capped=bool(np.isfinite(eta_p) and eta_p > E.ETA_MAX),
                ratio=round(eta_p_c / r["eta"], 4) if np.isfinite(eta_p_c) else None,
                how=how)
    payload["match"] = match
    payload["chosen"] = E.choose(payload, int(a.match_at), len(panel_idx))
    payload["wall_seconds"] = round(time.time() - t0, 1)
    C.write_json(Path(a.out), payload)
    print("\nchosen:", json.dumps(payload["chosen"], indent=2))
    print(f"wrote {a.out} in {payload['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
