"""Gate, ranking and chosen setting for the glomerular-encoding test.

The four gate criteria are fixed by the task and are applied mechanically:

    1. KC active-set Jaccard, between hand types / within hand type   < 0.5
    2. KC always-on fraction (fires for > 50% of states)              < 0.05
    3. KC 9-way hand-type probe accuracy  >= permuted-label + 0.15
    4. MBON mean rate < 30 Hz  AND  >= 20 MBONs varying across states

Ranking, also fixed before the grid finished: number of criteria passed, then
the KC hand-type margin over permuted labels (grouped CV, the leakage-free
number), then the lower KC Jaccard ratio.

Emits the markdown table for the report, the per-setting verdicts, and
``outputs/mb2/tuned_config.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "mb2"

JACCARD_RATIO_MAX = 0.5
ALWAYS_ON_MAX = 0.05
PROBE_MARGIN_MIN = 0.15
MBON_RATE_MAX = 30.0
MBON_VARYING_MIN = 20


def input_dependent_kcs(lbl: str) -> dict:
    """KCs whose firing actually depends on the input.

    A KC->MBON plasticity rule modifies the synapses of the KCs that fired. If a
    KC fires for every state it contributes the same update every time, and if it
    never fires it contributes nothing: only cells with a per-cell active rate
    strictly between 0 and 1 give such a rule any leverage. Mean Jaccard does not
    separate those cases, which is why this column exists beside it.
    """
    p = OUT_DIR / f"counts_{lbl}.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    out = {}
    for name in ("full", "late"):
        M = z["KC"].astype(np.int32)
        if name == "late":
            M = M - z["KC_early"].astype(np.int32)
        rate = (M > 0).mean(0)
        var = (rate > 0) & (rate < 1)
        out[name] = dict(
            input_dependent_cells=int(var.sum()),
            input_dependent_fraction=round(float(var.mean()), 4),
            always_cells=int((rate == 1).sum()),
            never_cells=int((rate == 0).sum()),
            mean_active_cells=round(float((M > 0).sum(1).mean()), 1),
            mean_input_dependent_active_per_state=round(
                float((M[:, var] > 0).sum(1).mean()), 1) if var.any() else 0.0,
        )
    return out


def row_for(lbl: str, s: dict, pr: dict) -> dict:
    sim = s["sim"]
    kc, kcl = sim["KC"], sim["KC_late"]
    j = sim["KC_jaccard_by_type"]
    p = pr.get(f"KC.full.hand_type", {})
    pb = pr.get(f"KC.full.sel_is_best", {})
    pi = pr.get(f"KC.full.intensity", {})
    pl = pr.get(f"KC.late.hand_type", {})
    pbin = pr.get(f"KCbin.full.hand_type", {})
    m = sim["MBON"]
    idep = input_dependent_kcs(lbl)
    checks = dict(
        jaccard_ratio=(j["ratio"] is not None and j["ratio"] < JACCARD_RATIO_MAX),
        always_on=kc["always_on_fraction"] < ALWAYS_ON_MAX,
        kc_probe=(p.get("margin") is not None and p["margin"] >= PROBE_MARGIN_MIN),
        mbon=(m["mean_rate_hz"] < MBON_RATE_MAX
              and m["nonzero_variance_cells"] >= MBON_VARYING_MIN),
    )
    return dict(
        label=lbl, config=s["config"], checks=checks,
        n_passed=int(sum(checks.values())), passed=all(checks.values()),
        kc_active=kc["mean_active_fraction"],
        kc_always_on=kc["always_on_fraction"],
        kc_always_over_active=kc["always_on_over_active"],
        kc_cells=kc["mean_active_cells"],
        kc_varying=kc["nonzero_variance_cells"],
        j_within=j["within"], j_between=j["between"], j_ratio=j["ratio"],
        kc_late_active=kcl["mean_active_fraction"],
        mbon_hz=m["mean_rate_hz"], mbon_varying=m["nonzero_variance_cells"],
        alpn_hz=sim["ALPN"]["mean_rate_hz"],
        alpn_cv=sim["ALPN"]["total_count_cv"],
        alpn_never=sim["ALPN"]["cells_never_firing"],
        orn_hz=sim["ORN_mean_rate_hz"], apl_hz=sim["APL_mean_rate_hz"],
        brain_spikes=sim["brain_spikes_per_state"],
        kc_hand=p.get("acc"), kc_hand_grp=p.get("grouped_acc"),
        kc_hand_perm=p.get("permuted"), kc_hand_margin=p.get("margin"),
        kc_late_hand=pl.get("acc"), kc_late_hand_grp=pl.get("grouped_acc"),
        kcbin_hand=pbin.get("acc"), kcbin_hand_grp=pbin.get("grouped_acc"),
        kc_best=pb.get("acc"), kc_best_bal=pb.get("balanced_acc"),
        kc_intensity=pi.get("acc"),
        kc_late_cols=pl.get("n_used_features"),
        kc_input_dependent=idep.get("full", {}).get("input_dependent_cells"),
        kc_input_dependent_frac=idep.get("full", {}).get("input_dependent_fraction"),
        kc_always_exactly=idep.get("full", {}).get("always_cells"),
        kc_idep_active_per_state=idep.get("full", {}).get(
            "mean_input_dependent_active_per_state"),
        kc_late_input_dependent=idep.get("late", {}).get("input_dependent_cells"),
        mn9=sim["mn9"]["verdict"]["summary"],
    )


def rank_key(r: dict):
    return (-r["n_passed"],
            -(r["kc_hand_grp"] or 0) + (r["kc_hand_perm"] or 0),
            r["j_ratio"] if r["j_ratio"] is not None else 9.9)


def table(rows: List[dict]) -> str:
    head = ("| setting | KC active | always-on | a/a | KC input-dep | J within | "
            "J between | J b/w | MBON Hz | MBON vary | ALPN Hz | ALPN CV | "
            "KC hand (grp) | perm | KCbin hand | KC intensity | gate |")
    sep = "|" + "---|" * 17
    out = [head, sep]
    for r in rows:
        marks = "".join("1" if r["checks"][k] else "0"
                        for k in ("jaccard_ratio", "always_on", "kc_probe", "mbon"))
        out.append(
            f"| `{r['label']}` | {r['kc_active']:.4f} | {r['kc_always_on']:.4f} | "
            f"{r['kc_always_over_active']} | {r['kc_input_dependent']} "
            f"({r['kc_input_dependent_frac']}) | "
            f"{r['j_within']} | {r['j_between']} | "
            f"**{r['j_ratio']}** | {r['mbon_hz']:.2f} | {r['mbon_varying']} | "
            f"{r['alpn_hz']:.2f} | {r['alpn_cv']} | "
            f"{r['kc_hand']} ({r['kc_hand_grp']}) | {r['kc_hand_perm']} | "
            f"{r['kcbin_hand']} ({r['kcbin_hand_grp']}) | {r['kc_intensity']} | "
            f"{'PASS' if r['passed'] else 'fail'} {marks} |")
    return "\n".join(out)


def probe_table(order: List[str], probes: dict, label: str,
                pops=("KC", "KCbin", "MBON", "ALPN", "DN")) -> str:
    """One label, rows = settings, cells = full / late accuracy (permuted)."""
    out = [f"| setting | " + " | ".join(pops) + " |", "|" + "---|" * (len(pops) + 1)]
    for lbl in order:
        pr = probes.get(lbl, {})
        cells = []
        for pop in pops:
            f = pr.get(f"{pop}.full.{label}", {})
            l = pr.get(f"{pop}.late.{label}", {})
            if not f:
                cells.append("-")
                continue
            cells.append(f"{f['acc']} / {l.get('acc','-')} ({f['permuted']})")
        out.append(f"| `{lbl}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument("--probe-tables", action="store_true")
    a = ap.parse_args()
    grid = json.loads((OUT_DIR / "grid.json").read_text())
    probes = json.loads((OUT_DIR / "probes.json").read_text())
    rows = [row_for(l, s, probes.get(l, {}))
            for l, s in grid["settings"].items() if "sim" in s]
    rows.sort(key=rank_key)
    # The kc_input_norm retry is reported separately and is not eligible to be
    # the chosen setting: the task frames it as a retry *under* the best setting
    # from the 12-cell grid.
    eligible = [r for r in rows if "norm" not in r["label"]]
    print(table(rows))
    print()
    for r in rows:
        c = r["checks"]
        print(f"{r['label']:26s} {'PASS' if r['passed'] else 'FAIL'}  "
              f"J {r['j_ratio']}{'<' if c['jaccard_ratio'] else '>='}0.5  "
              f"always-on {r['kc_always_on']}{'<' if c['always_on'] else '>='}0.05  "
              f"KC margin {r['kc_hand_margin']}{'>=' if c['kc_probe'] else '<'}0.15  "
              f"MBON {r['mbon_hz']:.1f}Hz/{r['mbon_varying']}"
              f"{' ok' if c['mbon'] else ' bad'}")
    if a.probe_tables:
        order = [r["label"] for r in rows]
        for lab in ("hand_type", "sel_is_best", "intensity"):
            print(f"\n#### {lab} -- accuracy full / t>=15 ms (permuted)\n")
            print(probe_table(order, probes, lab))
        print()
    winners = [r for r in rows if r["passed"]]
    print(f"\n{len(winners)} of {len(rows)} settings pass all four criteria")
    best = eligible[0]
    print("chosen:", best["label"], best["config"])
    if a.write_config:
        grid_meta = grid["meta"]
        cfg = dict(
            tuning=grid["settings"][best["label"]]["tuning"],
            drive=dict(
                mode=best["config"]["drive"],
                tonic_mv=grid_meta["drive_mv"],
                poisson_hz=grid_meta["poisson_hz"],
                poisson_kick_mv=grid_meta["poisson_kick_mv"],
            ),
            encoding=dict(
                kind="glomerular",
                n_features=32,
                source="flybalatro.features_v2 relay block (bits 0-31)",
                dropped="the 283 v1 state bits are not encoded",
                map_class="flybalatro.encode.GlomerularMap",
                assignment=grid_meta["assignment"],
            ),
            window_ms=grid_meta["window_ms"],
            brain_seed=grid_meta["brain_seed"],
            v_jitter_mv=grid_meta["v_jitter_mv"],
            label=best["label"],
            gate=dict(checks=best["checks"], passed=best["passed"],
                      n_passed=best["n_passed"],
                      criteria=dict(jaccard_ratio_max=JACCARD_RATIO_MAX,
                                    always_on_max=ALWAYS_ON_MAX,
                                    probe_margin_min=PROBE_MARGIN_MIN,
                                    mbon_rate_max=MBON_RATE_MAX,
                                    mbon_varying_min=MBON_VARYING_MIN)),
        )
        (OUT_DIR / "tuned_config.json").write_text(json.dumps(cfg, indent=2))
        print("wrote", OUT_DIR / "tuned_config.json")


if __name__ == "__main__":
    main()
