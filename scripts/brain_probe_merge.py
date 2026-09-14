"""Merge every scripts/brain_probe.py run into one outputs/brain_probe.json.

Also re-runs the "intensity" control from the saved count matrices: is the DN code
just total input magnitude (how many bits are on) rather than which bits?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUT = OUTPUTS / "brain_probe.json"
PRIMARY_TAG = "primary_w50_d30"

READOUTS = ["glomerular_ceiling", "ORN", "ALPN", "KC", "MBON", "CX", "DN",
            "MBON+CX", "DN+MBON+CX"]

SE_NOTE = (
    "At n=600 the CV standard error is ~0.020 and taking the best of a 6-value C "
    "grid inflates accuracy by a further ~0.03-0.05; the permuted-label probe in "
    "each run measures that inflation directly. Treat a readout as carrying "
    "information only if it beats its permuted-label baseline by >= 0.06."
)


def intensity_control(counts_npz: Path, seed: int = 0) -> dict:
    """Does the readout encode 'how many bits are on' rather than 'which bits'?"""
    from scripts.brain_probe import probe

    z = np.load(counts_npz)
    F = z["F"]
    tot = F.sum(1)
    y = (tot > np.median(tot)).astype(np.int8)
    res = {
        "label": "total active feature bits > median",
        "positive_rate": round(float(y.mean()), 4),
        "pearson_r_label_A_vs_total_bits": round(
            float(np.corrcoef(z["yA"].astype(float), tot.astype(float))[0, 1]), 3),
        "pearson_r_label_B_vs_total_bits": round(
            float(np.corrcoef(z["yB"].astype(float), tot.astype(float))[0, 1]), 3),
        "accuracy": {},
    }
    res["accuracy"]["glomerular_ceiling"] = probe(np.log1p(z["ceil"]), y, seed)["best_acc"]
    for pop in ["ORN", "ALPN", "KC", "MBON", "CX", "DN"]:
        res["accuracy"][f"real:{pop}"] = probe(
            np.log1p(z[f"real_{pop}"].astype(np.float64)), y, seed)["best_acc"]
    rng = np.random.default_rng(seed + 3)
    res["accuracy"]["PERMUTED real:DN"] = probe(
        np.log1p(z["real_DN"].astype(np.float64)), rng.permutation(y), seed)["best_acc"]
    return res


def main():
    runs = {}
    for p in sorted(OUTPUTS.glob("probe_*.json")):
        d = json.loads(p.read_text())
        runs[d["tag"]] = dict(d, source_file=str(p))
    if not runs:
        raise SystemExit("no probe_*.json runs found")

    table = {}
    for tag, d in runs.items():
        for lab, r in d["results"].items():
            conds = r["conditions"]
            row = {
                "majority_class_rate": r["majority_class_rate"],
                "permuted_real_DN": conds.get("permuted_label:real:DN", {}).get("best_acc"),
                "permuted_raw_bits": conds.get("permuted_label:raw_bits", {}).get("best_acc"),
                "raw_bits": conds.get("raw_bits", {}).get("best_acc"),
            }
            for k in READOUTS:
                if k == "glomerular_ceiling":
                    row[k] = conds.get(k, {}).get("best_acc")
                else:
                    row[f"real:{k}"] = conds.get(f"real:{k}", {}).get("best_acc")
                    row[f"shuffled:{k}"] = conds.get(f"shuffled:{k}", {}).get("best_acc")
            table[f"{tag}|label_{lab}"] = row

    out = dict(
        se_note=SE_NOTE,
        primary_tag=PRIMARY_TAG,
        conditions_explained={
            "raw_bits": "logistic probe straight on the binary feature vector, no brain",
            "glomerular_ceiling": "active driven ORNs per (ORN type, side) channel; no "
                                  "simulation. Upper bound for any downstream readout "
                                  "under perfect linear PN encoding.",
            "real:*": "spike counts from the MaleCNS wiring",
            "shuffled:*": "spike counts with edge targets globally permuted (in-degree "
                          "preserved). Not a null: permuting post gives DNs direct "
                          "one-hop ORN input, i.e. a sparse random projection with no "
                          "recurrence, which is close to ideal for a linear probe.",
            "permuted_label:*": "same pipeline and same best-of-C-grid on shuffled "
                                "labels; this is the honest chance level.",
        },
        summary_table=table,
        intensity_control=None,
        runs=runs,
    )
    npz = OUTPUTS / "counts_w50_d30.npz"
    if npz.exists():
        out["intensity_control"] = intensity_control(npz)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"merged {len(runs)} runs -> {OUT}")
    for tag in runs:
        print("  ", tag)


if __name__ == "__main__":
    main()
