"""MN9 sugar check, tonic vs Poisson, with several noise seeds.

``scripts/mb2_grid.py`` already runs one MN9 check per setting. Under Poisson
drive that is a single noise realisation of a two-neuron readout, which is not
enough to say anything: this script repeats it over several seeds and reports the
spread. Tonic drive is deterministic and needs one run.

Conditions per the reference check (``vendor/fly-craftax/scripts/mn9_check.py``):
the 165 labellar ``LB*`` gustatory receptor neurons, no drive at all, and 165
size-matched random ORNs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flybalatro import connectome as C
from scripts.mb2_grid import build_brain, mn9_check
from scripts.mb_sparsity import populations

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "mb2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="apl1,apl2,apl5",
                    help="apl scales to test, comma separated")
    ap.add_argument("--vth", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--window", type=float, default=200.0)
    ap.add_argument("--drive-mv", type=float, default=30.0)
    ap.add_argument("--threshold", type=int, default=5)
    a = ap.parse_args()

    g = C.load(a.threshold)
    pops = populations(g)
    out: dict = dict(
        window_ms=a.window, drive_mv=a.drive_mv, n_seeds=a.seeds,
        n_grn=int(len(pops["GRN"])), n_mn9=int(len(pops["MN9"])),
        mn9_body_ids=[int(g.body_id[i]) for i in pops["MN9"]],
        note=("tonic drive is deterministic (one run); Poisson is repeated over "
              "noise seeds because MN9 is a two-neuron readout"),
        settings={},
    )
    for tag in a.settings.split(","):
        apl = float(tag.replace("apl", ""))
        for mode in ("tonic", "poisson"):
            cfg = dict(label=f"{tag}_{mode}", apl_scale=apl, drive=mode,
                       kc_vth_offset_mv=a.vth, kc_input_norm=0.0)
            brain = build_brain(g, cfg, seed=1, v_jitter=6.9)
            brain.warmup()
            runs = []
            for k in range(1 if mode == "tonic" else a.seeds):
                r = mn9_check(brain, pops, g.n, mode, drive_mv=a.drive_mv,
                              window_ms=a.window, noise_seed=991 + 37 * k)
                runs.append({c: r[c]["mn9_spikes_total"]
                             for c in ("sugar_grn_drive", "no_drive",
                                       "matched_orn_control")}
                            | dict(grn_rate_hz=r["sugar_grn_drive"]["grn_rate_hz"],
                                   brain_spikes=r["sugar_grn_drive"]["brain_spikes"]))
            key = f"{tag}_{mode}"
            sug = [x["sugar_grn_drive"] for x in runs]
            mat = [x["matched_orn_control"] for x in runs]
            zer = [x["no_drive"] for x in runs]
            out["settings"][key] = dict(
                mode=mode, apl_scale=apl, runs=runs,
                sugar=dict(mean=round(float(np.mean(sug)), 2), min=min(sug),
                           max=max(sug)),
                matched_orn=dict(mean=round(float(np.mean(mat)), 2), min=min(mat),
                                 max=max(mat)),
                no_drive=dict(mean=round(float(np.mean(zer)), 2), max=max(zer)),
                grn_rate_hz=round(float(np.mean([x["grn_rate_hz"] for x in runs])), 2),
                sugar_selective=bool(np.mean(sug) > 2 * max(1.0, np.mean(mat))),
            )
            s = out["settings"][key]
            print(f"{key:16s} sugar {s['sugar']} matched {s['matched_orn']} "
                  f"no-drive {s['no_drive']} GRN {s['grn_rate_hz']} Hz  "
                  f"selective={s['sugar_selective']}", flush=True)
    (OUT_DIR / "mn9.json").write_text(json.dumps(out, indent=2))
    print("wrote", OUT_DIR / "mn9.json")


if __name__ == "__main__":
    main()
