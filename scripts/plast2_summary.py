"""Aggregate ``outputs/plast2/run_*.json`` into the tables the report quotes.

Produces ``outputs/plast2/summary.json`` and prints the same content as markdown:

* the evaluation table (real learned per eta, frozen, and every reference policy);
* P(play | hand type x score-vs-needed bucket) before vs after, plus the
  across-cell variance for naive / learned / teacher, the number that says
  whether what the fly learned is odour-specific or one global valence;
* the learning curves, decimated;
* the dopamine bookkeeping.

Run: ``python -m scripts.plast2_summary``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts import kc_homeo as H  # noqa: E402
from scripts import plast_common as K  # noqa: E402
from scripts.plast2_run import joint_p_play_table  # noqa: E402

OUT = H.OUT_DIR
BUCKETS = ("lt0.25", "lt0.5", "lt1.0", "ge1.0")


def load_runs() -> Dict[str, dict]:
    return {p.stem[len("run_"):]: json.loads(p.read_text())
            for p in sorted(OUT.glob("run_*.json"))}


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return round(float(np.mean(xs)), 4) if xs else None


def group_rows(runs: Dict[str, dict]) -> Dict[str, dict]:
    """Average the per-seed runs of each condition."""
    groups: Dict[str, List[dict]] = {}
    for name, r in runs.items():
        eta = r["spec"].get("eta_reward", r["spec"]["eta"])
        key = ("frozen" if not r["spec"]["plasticity"]
               else f"real_eta{eta:g}")
        groups.setdefault(key, []).append(r)
    out: Dict[str, dict] = {}
    for key, rs in sorted(groups.items()):
        pre = [r["pre"]["summary"] for r in rs]
        post = [r["post"]["summary"] for r in rs]
        out[key] = dict(
            n_seeds=len(rs), seeds=[r["spec"]["seed"] for r in rs],
            eta_reward=rs[0]["spec"].get("eta_reward", rs[0]["spec"]["eta"]),
            eta_punish=rs[0]["spec"].get("eta_punish", 0.0),
            clear_before=_mean([s["clear_rate"] for s in pre]),
            clear_after=_mean([s["clear_rate"] for s in post]),
            chips_before=_mean([s["chips_mean"] for s in pre]),
            chips_after=_mean([s["chips_mean"] for s in post]),
            p_play_before=_mean([s["p_play_when_legal"] for s in pre]),
            p_play_after=_mean([s["p_play_when_legal"] for s in post]),
            joint_var_before=_mean([r["pre"]["joint_p_play"]["variance"] for r in rs]),
            joint_var_after=_mean([r["post"]["joint_p_play"]["variance"] for r in rs]),
            joint_spread_before=_mean([r["pre"]["joint_p_play"]["spread"] for r in rs]),
            joint_spread_after=_mean([r["post"]["joint_p_play"]["spread"] for r in rs]),
            train_reward_rate=_mean([r["train"]["summary"]["reward_rate"] for r in rs]),
            train_reward_first_third=_mean([r["train"]["first_third"]["reward_rate"]
                                            for r in rs]),
            train_reward_last_third=_mean([r["train"]["last_third"]["reward_rate"]
                                           for r in rs]),
            train_p_play_first_third=_mean([r["train"]["first_third"]["p_play"]
                                            for r in rs]),
            train_p_play_last_third=_mean([r["train"]["last_third"]["p_play"]
                                           for r in rs]),
            dopamine=_merge_counts([r["train"]["dopamine_counts"] for r in rs]),
            weight_mean_ratio=_mean([r["weight_stats_after_training"]["mean_ratio"]
                                     for r in rs]),
            weight_frac_at_floor=_mean([r["weight_stats_after_training"]["frac_at_floor"]
                                        for r in rs]),
            weight_frac_unchanged=_mean([
                r["weight_stats_after_training"]["frac_unchanged"] for r in rs]),
            bucket_before=_bucket_means(rs, "pre"),
            bucket_after=_bucket_means(rs, "post"),
            runs=[r["spec"]["name"] for r in rs],
        )
    return out


def _merge_counts(dicts: Sequence[dict]) -> dict:
    out: Dict[str, float] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    return {k: round(v / max(1, len(dicts)), 1) for k, v in sorted(out.items())}


def _bucket_means(rs: Sequence[dict], tag: str) -> dict:
    out: Dict[str, Optional[float]] = {}
    for b in BUCKETS:
        out[b] = _mean([r[tag]["p_play_by_bucket"].get(b, {}).get("p_play")
                        for r in rs])
    return out


def joint_grid(cells: dict, order: Sequence[str]) -> List[List[str]]:
    """A markdown-ready hand-type x bucket grid of ``p_play`` (n)."""
    rows: List[List[str]] = []
    for ht in order:
        row = [ht]
        for b in BUCKETS:
            c = cells.get(f"{ht}|{b}")
            row.append(f"{c['p_play']:.2f} ({c['n']})" if c else "-")
        rows.append(row)
    return rows


def merged_joint(runs: Dict[str, dict], names: Sequence[str], tag: str) -> dict:
    recs = [h for n in names for h in runs[n][tag]["records"]]
    return joint_p_play_table(recs)


def teacher_alignment(cells: dict, teacher: dict, min_n: int = 5) -> dict:
    """How close one policy's per-cell P(play) is to the teacher's, cell by cell.

    The across-cell *variance* the brief asks for says how much a policy varies,
    not whether it varies in the right direction: a fly that plays everything
    except one hand type has low variance and is useless, and one that varies
    arbitrarily has high variance and is also useless. This adds the direction.
    Cells are weighted by how many decisions the fly faced in them.
    """
    keys = [k for k in cells
            if k in teacher and cells[k]["n"] >= min_n and teacher[k]["n"] >= min_n]
    if len(keys) < 3:
        return dict(n_cells=len(keys))
    x = np.array([cells[k]["p_play"] for k in keys], float)
    y = np.array([teacher[k]["p_play"] for k in keys], float)
    w = np.array([cells[k]["n"] for k in keys], float)
    corr = (None if x.std() == 0.0 or y.std() == 0.0
            else round(float(np.corrcoef(x, y)[0, 1]), 4))
    return dict(
        n_cells=len(keys),
        correlation=corr,
        weighted_mean_abs_error=round(float((w * np.abs(x - y)).sum() / w.sum()), 4),
        weighted_agreement=round(float((w * (1.0 - np.abs(x - y))).sum() / w.sum()), 4),
    )


def bucket_gradient(by_bucket: dict) -> Optional[float]:
    """P(play | ge1.0) - P(play | lt0.25): the one ordered axis the odour carries."""
    hi, lo = by_bucket.get("ge1.0"), by_bucket.get("lt0.25")
    if hi is None or lo is None:
        return None
    hi = hi["p_play"] if isinstance(hi, dict) else hi
    lo = lo["p_play"] if isinstance(lo, dict) else lo
    if hi is None or lo is None:
        return None
    return round(float(hi) - float(lo), 4)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT / "summary.json"))
    a = ap.parse_args(argv)

    runs = load_runs()
    if not runs:
        raise SystemExit(f"no run_*.json in {OUT}")
    groups = group_rows(runs)
    baselines = json.loads((OUT / "baselines.json").read_text())
    homeo = json.loads((H.HOMEO_JSON).read_text())
    gate = json.loads((OUT / "gate.json").read_text())
    eta = json.loads((OUT / "eta_calib.json").read_text())

    # Pooled joint tables, so every cell has enough decisions to mean something.
    learned_names = [n for n, r in runs.items()
                     if r["spec"]["plasticity"]
                     and r["spec"].get("eta_reward", r["spec"]["eta"]) == 0.05]
    frozen_names = [n for n, r in runs.items() if not r["spec"]["plasticity"]]
    pooled = dict(
        naive=merged_joint(runs, learned_names, "pre"),
        learned=merged_joint(runs, learned_names, "post"),
        frozen=merged_joint(runs, frozen_names, "post"),
        teacher=baselines["teacher_dig"]["joint_p_play"],
        best_bucket=baselines["bucket_ge_lt1"]["joint_p_play"],
        always_play=baselines["always_play"]["joint_p_play"],
    )

    order = [ht for ht in ("High Card", "Pair", "Two Pair", "Three of a Kind",
                           "Straight", "Flush", "Full House", "Four of a Kind",
                           "Straight Flush")
             if any(k.startswith(ht + "|") for k in pooled["learned"]["cells"])]

    tcells = pooled["teacher"]["cells"]
    alignment = {k: teacher_alignment(v["cells"], tcells) for k, v in pooled.items()}
    gradients = dict(
        **{f"{k}_before": bucket_gradient(g["bucket_before"])
           for k, g in groups.items()},
        **{f"{k}_after": bucket_gradient(g["bucket_after"]) for k, g in groups.items()},
        **{k: bucket_gradient(v["p_play_by_bucket"]) for k, v in baselines.items()
           if isinstance(v, dict) and "p_play_by_bucket" in v},
    )

    payload = dict(
        runs=sorted(runs), groups=groups,
        teacher_alignment=alignment, bucket_gradient=gradients,
        baselines={k: v["summary"] for k, v in baselines.items()
                   if isinstance(v, dict) and "summary" in v},
        baseline_joint={k: {kk: vv for kk, vv in v["joint_p_play"].items()
                            if kk != "cells"}
                        for k, v in baselines.items()
                        if isinstance(v, dict) and "joint_p_play" in v},
        pooled_joint={k: dict(v, grid=joint_grid(v["cells"], order))
                      for k, v in pooled.items()},
        joint_variance={k: v["variance"] for k, v in pooled.items()},
        joint_spread={k: v["spread"] for k, v in pooled.items()},
        homeostasis=dict(
            iterations=homeo["iterations"], final=homeo["final_calib"],
            patterns=homeo["patterns"], target=homeo["target"],
            clamp=homeo["offset_clamp"], v0_margin=homeo["v0_margin_mv"]),
        gate={k: v["verdict"] for k, v in gate["conditions"].items()},
        gate_kc={k: v["measure"]["kc"] for k, v in gate["conditions"].items()},
        gate_mbon={k: v["measure"]["mbon"] for k, v in gate["conditions"].items()},
        eta_calibration=eta["chosen"],
        learning_curves={
            n: dict(reward_rate=runs[n]["train"]["reward_rate_rolling"][::20],
                    p_play=runs[n]["train"]["p_play_rolling"][::20],
                    play_drive=runs[n]["train"]["play_drive_rolling"][::20])
            for n in sorted(runs)},
    )
    K.write_json(Path(a.out), payload)

    # ---- printed tables --------------------------------------------------
    print("\n## Evaluation, 60 ante-1 games on seeds 100000-100059\n")
    print("| row | clear before | clear after | chips after | P(play|legal) after |")
    print("|---|---|---|---|---|")
    for key, g in groups.items():
        label = (f"real, eta_r {g['eta_reward']:g} / eta_p {g['eta_punish']:g} "
                 f"({g['n_seeds']} seeds)" if key != "frozen"
                 else f"frozen fly, no dopamine ({g['n_seeds']} seeds)")
        print(f"| {label} | {g['clear_before']:.3f} | {g['clear_after']:.3f} | "
              f"{g['chips_after']:.0f} | {g['p_play_after']:.3f} |")
    for name, lab in (("always_play", "always play"),
                      ("always_discard", "always discard"),
                      ("bucket_ge_lt1", "best odour-only policy (play iff >= lt1.0)"),
                      ("teacher_dig", "v2 teacher dig rule (sees plays_left)")):
        s = baselines[name]["summary"]
        pp = s["p_play_when_legal"]
        print(f"| {lab} | - | {s['clear_rate']:.3f} | {s['chips_mean']:.0f} | "
              f"{pp:.3f} |")

    print("\n## Variance of P(play) across hand-type x bucket cells\n")
    print("| condition | cells (n>=5) | variance | sd | spread | mean | "
          "corr w/ teacher | wtd |err| |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in pooled.items():
        al = alignment[k]
        print(f"| {k} | {v['n_cells_ge_min_n']} | {v['variance']} | {v['sd']} | "
              f"{v['spread']} | {v['mean']} | {al.get('correlation')} | "
              f"{al.get('weighted_mean_abs_error')} |")

    print("\n## P(play) by score-vs-needed bucket, and the gradient\n")
    print("| row | lt0.25 | lt0.5 | lt1.0 | ge1.0 | ge1.0 - lt0.25 |")
    print("|---|---|---|---|---|---|")
    for k, g in groups.items():
        for tag in ("before", "after"):
            b = g[f"bucket_{tag}"]
            print(f"| {k} {tag} | " + " | ".join(
                f"{b[x]:.3f}" if b.get(x) is not None else "-" for x in BUCKETS)
                + f" | {bucket_gradient(b):+.3f} |")
    for name in ("teacher_dig", "bucket_ge_lt1"):
        b = baselines[name]["p_play_by_bucket"]
        print(f"| {name} | " + " | ".join(
            f"{b[x]['p_play']:.3f}" if x in b else "-" for x in BUCKETS)
            + f" | {bucket_gradient(b):+.3f} |")

    for k in ("naive", "learned", "teacher"):
        print(f"\n### P(play | hand type x bucket) -- {k}\n")
        print("| hand type | " + " | ".join(BUCKETS) + " |")
        print("|---" * (len(BUCKETS) + 1) + "|")
        for row in joint_grid(pooled[k]["cells"], order):
            print("| " + " | ".join(row) + " |")

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
