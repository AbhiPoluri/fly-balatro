"""Markdown tables for outputs/calyx/REPORT.md, straight out of the stage JSONs.

Pure formatting: every number here was produced by ``calyx_common`` (rewiring),
``calyx_homeo`` (calibration), ``calyx_probe`` (activity / code / decoding),
``calyx_train`` (imitation) or ``calyx_stats`` (paired tests). Written to
``outputs/calyx/tables_<variant>.md`` so the report quotes a generated table
rather than a hand-copied one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent
for _p in (str(ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_DIR = ROOT / "outputs" / "calyx"


def _f(x, nd=4, dash="-"):
    if x is None:
        return dash
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def activity_table(probe: dict, conds: Sequence[str]) -> str:
    rows = ["| condition | ALPN Hz | KC Hz | MBON Hz | DN Hz | KC active frac | "
            "KC active/state | KC always-on (r>0.5) | KC r>0.3 | KC silent | "
            "non-constant ALPN+KC+DN |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        if c not in probe:
            continue
        a = probe[c]["activity"]
        rows.append(
            f"| {c} | {_f(a['alpn']['mean_rate_hz'],2)} | {_f(a['kc']['mean_rate_hz'],2)} "
            f"| {_f(a['mbon']['mean_rate_hz'],2)} | {_f(a['dn']['mean_rate_hz'],2)} "
            f"| {_f(a['kc']['active_frac_per_state'],4)} | {_f(a['kc']['active_per_state'],1)} "
            f"| {_f(a['kc']['frac_r_gt_0_5'],4)} ({a['kc']['n_r_gt_0_5']}) "
            f"| {_f(a['kc']['frac_r_gt_0_3'],4)} ({a['kc']['n_r_gt_0_3']}) "
            f"| {a['kc']['n_silent']} | {a['non_constant_channels_alpn_kc_dn']} |")
    return "\n".join(rows)


def population_table(probe: dict, conds: Sequence[str]) -> str:
    rows = ["| condition | population | mean Hz | active/state | non-constant | ever active |",
            "|---|---|---|---|---|---|"]
    for c in conds:
        if c not in probe:
            continue
        for p in ("alpn", "kc", "mbon", "dn"):
            a = probe[c]["activity"][p]
            rows.append(f"| {c} | {p.upper()} ({a['n']}) | {_f(a['mean_rate_hz'],2)} "
                        f"| {_f(a['active_per_state'],1)} | {a['n_non_constant']} "
                        f"| {a['n_ever_active']} |")
    return "\n".join(rows)


def jaccard_table(probe: dict, conds: Sequence[str]) -> str:
    rows = ["| condition | J within hand type | J between hand types | ratio | "
            "mean KC active-set size |", "|---|---|---|---|---|"]
    for c in conds:
        if c not in probe:
            continue
        j = probe[c]["kc_code"]["jaccard"]
        rows.append(f"| {c} | {_f(j['within'])} | {_f(j['between'])} | {_f(j['ratio'])} "
                    f"| {_f(j['mean_set_size'],1)} |")
    return "\n".join(rows)


def decode_table(probe: dict, conds: Sequence[str],
                 pops=("kc", "kcbin", "dn", "alpn", "mbon")) -> str:
    head = "| condition | " + " | ".join(p.upper() for p in pops) + " |"
    rows = [head, "|---" * (len(pops) + 1) + "|"]
    for c in conds:
        if c not in probe:
            continue
        d = probe[c]["decode_hand_type"]
        cells = []
        for p in pops:
            if p not in d:
                cells.append("-")
                continue
            cells.append(f"{_f(d[p]['acc'],3)} / {_f(d[p]['acc_grouped'],3)} "
                         f"({_f(d[p]['permuted'],3)})")
        rows.append(f"| {c} | " + " | ".join(cells) + " |")
    b = probe.get("_baselines", {})
    rows.append("")
    rows.append("Baselines on the same 500 held-out patterns: majority "
                f"{_f(b.get('majority'),4)}, permuted labels on the 32 relay bits "
                f"{_f(b.get('permuted_labels_on_bits'),4)}, the 32 relay bits "
                f"{_f(b.get('relay_bits_32'),4)}, bits-on count only "
                f"{_f(b.get('bits_on_count_only'),4)}, **driven-ORN count only "
                f"{_f(b.get('driven_orn_count_only'),4)}** "
                f"(`outputs/mb2/REPORT.md` measured 0.267 for that confound on its "
                "own 2,000-state sample).")
    return "\n".join(rows)


def imitation_table(variant: str, conds: Sequence[str], readouts: Sequence[str],
                    out_dir: Path) -> str:
    rows = ["| condition | readout | held-out imitation top-1 | effective dims |",
            "|---|---|---|---|"]
    for c in conds:
        p = out_dir / variant / c / "train_metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        for ro in readouts:
            k = f"{ro}/linear"
            if k in m:
                rows.append(f"| {c} | {ro} | {_f(m[k]['test']['top1'],4)} "
                            f"| {m[k].get('effective_dims','-')} / {m[k]['raw_dims']} |")
    return "\n".join(rows)


def live_table(variant: str, conds: Sequence[str], readouts: Sequence[str],
               out_dir: Path) -> str:
    rows = ["| condition | readout | clear ante 1 | mean chips | plays that were "
            "the best subset | plays |", "|---|---|---|---|---|---|"]
    for c in conds:
        p = out_dir / variant / c / "eval.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        for ro in readouts:
            k = f"calyx_{c}_{ro}"
            if k not in m:
                continue
            s = m[k]
            ev = s.get("hand_evidence") or {}
            rows.append(f"| {c} | {ro} | {s['clear_rate']*100:.1f}% "
                        f"| {s['chips_mean']:.0f} "
                        f"| {_f(ev.get('plays_that_were_the_best_subset'),3)} "
                        f"| {ev.get('plays','-')} |")
    return "\n".join(rows)


def paired_table(paired: dict, readouts: Sequence[str]) -> str:
    rows = ["| readout | rewiring | imitation real - rw (95% CI) | clear real - rw "
            "| McNemar p (discordant) | chips real - rw (95% CI) |",
            "|---|---|---|---|---|---|"]
    for ro in readouts:
        for key, r in paired.get("paired", {}).items():
            if not key.startswith(f"{ro}/"):
                continue
            cond = key.split("/", 1)[1]
            im = r.get("imitation_top1", {})
            cl = r.get("clear_rate", {})
            ch = r.get("chips", {})
            im_s = (f"{_f(im.get('diff'),4)} [{_f(im.get('ci95',[None,None])[0],4)}, "
                    f"{_f(im.get('ci95',[None,None])[1],4)}]" if im else "-")
            cl_s = (f"{cl['diff']*100:+.1f} pp ({cl['real']*100:.1f}% vs "
                    f"{cl['rewired']*100:.1f}%)" if cl else "-")
            mc_s = (f"{_f(cl.get('mcnemar_p'),4)} ({cl.get('a_only')}/"
                    f"{cl.get('b_only')})" if cl else "-")
            ch_s = (f"{_f(ch.get('diff'),1)} [{_f(ch.get('ci95',[None,None])[0],1)}, "
                    f"{_f(ch.get('ci95',[None,None])[1],1)}]" if ch else "-")
            rows.append(f"| {ro} | {cond} | {im_s} | {cl_s} | {mc_s} | {ch_s} |")
    return "\n".join(rows)


def homeo_table(conds: Sequence[str], out_dir: Path) -> str:
    rows = ["| condition | iterations | converged | KC active frac | active/state | "
            "r>0.5 | r>0.3 | silent | offset mean / SD / min / max (mV) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        p = out_dir / f"homeo_{c}.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())
        f_ = h.get("final_calib", {})
        o = h.get("offset_summary", {})
        rows.append(
            f"| {c} | {h.get('n_iterations_run','-')} | {h.get('converged')} "
            f"| {_f(f_.get('active_frac'),4)} | {_f(f_.get('active_per_state'),1)} "
            f"| {_f(f_.get('frac_r_gt_0_5'),4)} | {_f(f_.get('frac_r_gt_0_3'),4)} "
            f"| {f_.get('n_silent','-')} | {_f(o.get('mean'),2)} / {_f(o.get('sd'),2)} "
            f"/ {_f(o.get('min'),1)} / {_f(o.get('max'),1)} |")
    return "\n".join(rows)


def rewiring_table(out_dir: Path) -> str:
    p = out_dir / "rewiring.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    rows = ["| rewiring | swaps accepted | edges with a new target | novel (PN, KC) "
            "pairs | per-KC inputs relocated (mean +- SD) |", "|---|---|---|---|---|"]
    for k, v in d["rewirings"].items():
        rows.append(f"| {k} | {v['swaps_accepted']:,} "
                    f"| {v['frac_edges_with_new_target']*100:.1f}% "
                    f"| {v['frac_edges_novel_pair']*100:.1f}% "
                    f"| {v['per_kc_frac_inputs_relocated_mean']*100:.1f}% +- "
                    f"{v['per_kc_frac_inputs_relocated_sd']*100:.1f}% |")
    return "\n".join(rows)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="raw")
    ap.add_argument("--conditions", default="real,rw1,rw2,rw3")
    ap.add_argument("--readouts", default="kc,dn")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    ros = [r.strip() for r in a.readouts.split(",") if r.strip()]
    probe_p = out_dir / f"probe_{a.variant}.json"
    probe = json.loads(probe_p.read_text()) if probe_p.exists() else {}
    paired_p = out_dir / f"paired_{a.variant}.json"
    paired = json.loads(paired_p.read_text()) if paired_p.exists() else {}
    parts = [
        f"## Rewiring depth\n\n{rewiring_table(out_dir)}",
        f"## Per-KC homeostasis\n\n{homeo_table(conds, out_dir)}",
        f"## Activity matching ({a.variant})\n\n{activity_table(probe, conds)}",
        f"## Populations ({a.variant})\n\n{population_table(probe, conds)}",
        f"## KC active-set Jaccard ({a.variant})\n\n{jaccard_table(probe, conds)}",
        ("## 9-way hand-type decoding, acc / grouped-CV (permuted) "
         f"({a.variant})\n\n{decode_table(probe, conds)}"),
        f"## Imitation ({a.variant})\n\n{imitation_table(a.variant, conds, ros, out_dir)}",
        f"## Live play, 400 episodes ({a.variant})\n\n{live_table(a.variant, conds, ros, out_dir)}",
        f"## Paired comparisons ({a.variant})\n\n{paired_table(paired, ros)}",
    ]
    txt = "\n\n".join(parts) + "\n"
    p = out_dir / f"tables_{a.variant}.md"
    p.write_text(txt)
    print(txt)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
