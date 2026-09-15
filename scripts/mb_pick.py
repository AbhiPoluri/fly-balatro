"""Pick a tuning from outputs/mb/sweep.json against explicit gates, write the config.

The choice is a script, not a judgement call in prose, so it can be re-derived.
Gates, in the order they are applied:

  sparsity   KC active fraction inside [--lo, --hi]              (target 5-10%)
  sanity     ALPN mean rate within --alpn-tol of the untuned
             model's, because these knobs are supposed to calibrate the
             mushroom body, and a drifting antennal-lobe rate means
             a knob is reaching outside it (APL sends ~6% of its
             output to non-KC targets, so a large apl_scale does)
  readable   MBON mean rate >= --mbon-hz and at least --mbon-var
             MBON cells with nonzero variance across inputs; an
             MBON population that barely spikes has nothing for a
             downstream readout (or a plasticity rule) to use
  mn9        the no-drive control must still be exactly 0 spikes and MN9
             must still respond to sugar drive at all. Strict equality with
             the untuned count is *not* required: driving 165 GRNs puts the
             whole brain into an ~80k-spike storm and MN9 is a few hops
             downstream of the mushroom body, so a KC-only knob moves it by
             a spike or two. (The untuned model already fails Shiu et al.'s
             check on specificity, see REPORT.md, so this gate only
             checks that tuning did not make it worse.)

Survivors are then ranked by the separability of the KC code: first the share of
*active* cells that are always-on (an absolute always-on fraction would just
favour the sparsest row inside the band), then how close the mean pairwise
Jaccard is to the p/(2-p) value independent sets of the same size would give.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flybalatro.tuning import Tuning
from scripts.mb_sparsity import OUT_DIR

BASELINE = "default"


def gate(r: dict, base: dict, lo: float, hi: float, alpn_tol: float,
         mbon_hz: float, mbon_var: int) -> list[str]:
    fails = []
    if not lo <= r["kc_active_fraction"] <= hi:
        fails.append(f"kc_active_fraction {r['kc_active_fraction']} outside [{lo},{hi}]")
    drift = abs(r["alpn_rate_hz"] - base["alpn_rate_hz"]) / base["alpn_rate_hz"]
    if drift > alpn_tol:
        fails.append(f"ALPN rate drift {drift:.3f} > {alpn_tol}")
    if r["mbon_rate_hz"] < mbon_hz:
        fails.append(f"MBON rate {r['mbon_rate_hz']} < {mbon_hz} Hz")
    if r.get("mbon_varying_cells", mbon_var) < mbon_var:
        fails.append(f"MBON varying cells {r.get('mbon_varying_cells')} < {mbon_var}")
    if r["mn9_no_drive_spikes"] != 0:
        fails.append(f"MN9 fires with no drive ({r['mn9_no_drive_spikes']})")
    if r["mn9_sugar_spikes"] <= 0:
        fails.append("MN9 no longer responds to sugar drive at all")
    return fails


def score(r: dict) -> tuple:
    """Lower is better: always-on share of the active set, then Jaccard / reference."""
    always_share = (r["kc_always_on_fraction"] / r["kc_active_fraction"]
                    if r["kc_active_fraction"] else 1e9)
    ratio = (r["kc_jaccard"] / r["kc_jaccard_bernoulli_ref"]
             if r["kc_jaccard_bernoulli_ref"] else 1e9)
    return (round(always_share, 3), ratio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(OUT_DIR / "sweep.json"))
    ap.add_argument("--out", default=str(OUT_DIR / "tuned_config.json"))
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.10)
    # 10%: APL sends ~6% of its output weight to non-KC targets (ALPN calyx
    # terminals, MBONs, DANs), so scaling APL moves the ALPN rate a little even
    # when it is doing exactly what it is supposed to do inside the calyx.
    ap.add_argument("--alpn-tol", type=float, default=0.10)
    ap.add_argument("--mbon-hz", type=float, default=5.0)
    ap.add_argument("--mbon-var", type=int, default=40)
    a = ap.parse_args()

    doc = json.loads(Path(a.sweep).read_text())
    runs = doc["runs"]
    rows = {r["label"]: r for r in doc["table"]}
    for lab, r in rows.items():
        r["mbon_varying_cells"] = runs[lab]["MBON"]["nonzero_variance_cells"]
    base = rows[BASELINE]

    ok, rejected = [], {}
    for lab, r in rows.items():
        if lab == BASELINE:
            continue
        f = gate(r, base, a.lo, a.hi, a.alpn_tol, a.mbon_hz, a.mbon_var)
        (ok.append(r) if not f else rejected.setdefault(lab, f))
    ok.sort(key=score)

    print(f"{len(ok)} of {len(rows)-1} settings pass the gates")
    head = (f"{'setting':<34}{'KCfrac':>8}{'always':>8}{'Jacc':>8}{'ref':>7}"
            f"{'J/ref':>7}{'a/f':>6}{'MBONHz':>8}{'MBONvar':>8}{'ALPNHz':>8}")
    print(head)
    print("-" * len(head))
    for r in ok[:15]:
        print(f"{r['label']:<34}{r['kc_active_fraction']:>8.4f}"
              f"{r['kc_always_on_fraction']:>8.4f}{r['kc_jaccard']:>8.4f}"
              f"{r['kc_jaccard_bernoulli_ref']:>7.4f}"
              f"{r['kc_jaccard']/max(r['kc_jaccard_bernoulli_ref'],1e-9):>7.1f}"
              f"{r['kc_always_on_fraction']/max(r['kc_active_fraction'],1e-9):>6.2f}"
              f"{r['mbon_rate_hz']:>8.2f}{r['mbon_varying_cells']:>8}"
              f"{r['alpn_rate_hz']:>8.2f}")
    if not ok:
        print("\nno setting passes; nearest misses:")
        for lab, f in sorted(rejected.items(), key=lambda t: len(t[1]))[:10]:
            print(f"  {lab:<34} {'; '.join(f)}")
        raise SystemExit(1)

    best = ok[0]
    t = Tuning.from_dict({k: best[k] for k in
                          ("apl_scale", "alpn_kc_scale", "kc_vth_offset_mv",
                           "kc_bias_mv", "kc_kc_scale", "kc_input_norm")})
    out = dict(
        tuning=t.to_dict(),
        label=t.label(),
        note=("Calibration knobs for flybalatro.tuning.Tuning, NOT connectome data. "
              "Load with Tuning.from_dict(json.load(f)['tuning']) and pass to "
              "Brain(graph, tuning=...). Chosen by scripts/mb_pick.py from "
              "outputs/mb/sweep.json."),
        gates=dict(kc_active_fraction=[a.lo, a.hi], alpn_rate_tolerance=a.alpn_tol,
                   mbon_min_rate_hz=a.mbon_hz, mbon_min_varying_cells=a.mbon_var,
                   mn9_no_drive_must_be_zero=True,
                   mn9_sugar_response_must_be_nonzero=True),
        brain=dict(seed=doc["settings"]["brain_seed"],
                   v_jitter_mv=doc["settings"]["v_jitter_mv"],
                   window_ms=doc["settings"]["window_ms"],
                   drive_mv=doc["settings"]["drive_mv"],
                   threshold=doc["settings"]["threshold"]),
        metrics=best,
        baseline_metrics=base,
        runners_up=[r["label"] for r in ok[1:6]],
        rejected={k: v for k, v in sorted(rejected.items())},
    )
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nbest: {t.label()}  {t.to_dict()}")
    print("wrote", p)


if __name__ == "__main__":
    main()
