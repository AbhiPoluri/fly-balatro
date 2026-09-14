"""Tables and the preregistered decision for plasticity v4.

Reads every ``outputs/plast4/run_*.json`` plus ``baselines4.json`` and applies
the rule fixed in ``outputs/plast4/PREREGISTRATION.md`` section 6:

* primary outcome: ante-1 clear rate, **greedy**, on the 400 paired seeds;
* two paired deltas per condition -- against the shared frozen control and
  against always-discard -- each with a two-way cluster bootstrap over
  evaluation seeds **and** training runs;
* exact McNemar per training run, never pooled across runs (they share seeds);
* Holm correction across the three novel families ``{B, C80, D}`` on the
  bootstrap p-value of the always-discard delta. ``C50`` and ``C100`` are
  sensitivity analyses and cannot promote anything;
* success requires **both** deltas positive with a 95% CI excluding zero, and a
  Holm-corrected p < 0.05 on the always-discard delta.

Writes ``summary.json`` and ``summary.md``. Nothing is recomputed by hand.

Run::

    python -m scripts.plast4_summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts import plast3_common as C3
from scripts import plast4_common as C4
from scripts import plast_common as K

OUT = C4.OUT_DIR
BUCKETS = ("lt0.25", "lt0.5", "lt1.0", "ge1.0")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _cleared(payload: dict, mode: str) -> np.ndarray:
    return np.asarray([1.0 if c else 0.0 for c in payload["eval"][mode]["cleared"]],
                      np.float64)


def condition_rows(out_dir: Path) -> dict:
    base = _load(out_dir / "baselines4.json")
    frozen = _load(out_dir / "run_frozen.json")
    discard = np.asarray([1.0 if c else 0.0
                          for c in base["always_discard"]["cleared"]], np.float64)

    rows: Dict[str, dict] = {}
    for cond in C4.conditions():
        runs = []
        for s in C3.TRAIN3_SEEDS:
            p = out_dir / f"run_{cond.name}_s{s}.json"
            if not p.exists():
                break
            runs.append(_load(p))
        if len(runs) != len(C3.TRAIN3_SEEDS):
            print(f"  (skipping {cond.name}: {len(runs)}/{len(C3.TRAIN3_SEEDS)} runs)")
            continue
        row: dict = dict(name=cond.name, label=cond.label,
                         reward_rule=cond.reward_rule, gamma=cond.gamma,
                         n_runs=len(runs), modes={})
        for mode in C3.EVAL_MODES:
            A = [_cleared(r, mode) for r in runs]
            f = _cleared(frozen, mode)
            vs_frozen = C3.cluster_bootstrap(A, f).to_dict()
            vs_discard = C3.cluster_bootstrap(A, discard).to_dict()
            row["modes"][mode] = dict(
                clear_rate=round(float(np.mean([a.mean() for a in A])), 4),
                per_run_clear=[round(float(a.mean()), 4) for a in A],
                frozen_clear=round(float(f.mean()), 4),
                discard_clear=round(float(discard.mean()), 4),
                vs_frozen=vs_frozen,
                vs_discard=vs_discard,
                mcnemar_vs_frozen=[C4.mcnemar_exact(a, f) for a in A],
                mcnemar_vs_discard=[C4.mcnemar_exact(a, discard) for a in A],
                p_play_when_legal=round(float(np.mean(
                    [r["eval"][mode]["summary"]["p_play_when_legal"] for r in runs])), 4),
                chips_mean=round(float(np.mean(
                    [r["eval"][mode]["summary"]["chips_mean"] for r in runs])), 1),
                bucket_gradient=round(float(np.mean(
                    [r["eval"][mode]["bucket_gradient"] for r in runs])), 4),
                p_play_by_bucket=_mean_bucket(runs, mode),
                p_play_by_bucket_per_run=[
                    {b: round(r["eval"][mode]["p_play_by_bucket"].get(b, {})
                              .get("p_play", float("nan")), 4) for b in BUCKETS}
                    for r in runs],
            )
        row["ledger"] = _ledger(runs)
        row["trace_specificity"] = _trace_specificity(runs)
        for b in BUCKETS:
            d = row["ledger"][b]
            d["net_lower_bound"] = (d["reward"] - d["punish_immediate"]
                                    - d["n_terminal_pulses"])
        row["train"] = dict(
            n_terminal_pulses=[r["train"]["n_terminal_pulses"] for r in runs],
            trace_weight_delivered=[r["train"]["trace_weight_delivered"] for r in runs],
            dopamine_counts=[r["train"]["dopamine_counts"] for r in runs],
            reward_rule_agreement=[r["train"]["reward_rule_agreement"] for r in runs],
            p_play_when_legal=[round(r["train"]["summary"]["p_play_when_legal"], 4)
                               for r in runs],
            weight_mean_ratio=[round(r["weight_stats"]["mean_ratio"], 5) for r in runs],
        )
        rows[cond.name] = row
    return dict(conditions=rows,
                frozen=dict(greedy=round(float(_cleared(frozen, C3.GREEDY).mean()), 4),
                            sampled=round(float(_cleared(frozen, C3.SAMPLED).mean()), 4)),
                baselines={k: round(v["summary"]["clear_rate"], 4)
                           for k, v in base.items() if isinstance(v, dict)})


def _mean_bucket(runs: Sequence[dict], mode: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for b in BUCKETS:
        vals = [r["eval"][mode]["p_play_by_bucket"].get(b, {}).get("p_play")
                for r in runs]
        vals = [v for v in vals if v is not None]
        out[b] = round(float(np.mean(vals)), 4) if vals else float("nan")
    return out


def _trace_specificity(runs: Sequence[dict]) -> dict:
    """How bucket-selective the terminal pulse is.

    One lost blind fires **one** pulse whose eligibility is the union of that
    blind's recent plays, so the summed ``gamma^k`` weight is an upper bound on
    what a bucket receives and the number of pulses that *contain* the bucket is
    a lower bound. The fraction of pulses containing each bucket is the direct
    measure of whether terminal credit can separate one bucket from another.
    """
    total = int(sum(r["train"]["n_terminal_pulses"] for r in runs))
    if total == 0:
        return dict(n_pulses=0, pulse_share={}, weight_share={})
    pulses = {b: 0 for b in BUCKETS}
    weight = {b: 0.0 for b in BUCKETS}
    for r in runs:
        for b in BUCKETS:
            src = r["train"]["ledger_by_bucket"].get(b, {})
            pulses[b] += int(src.get("n_terminal_pulses", 0))
            weight[b] += float(src.get("punish_trace_weight", 0.0))
    wtot = sum(weight.values()) or 1.0
    return dict(n_pulses=total,
                pulse_share={b: round(pulses[b] / total, 4) for b in BUCKETS},
                weight_share={b: round(weight[b] / wtot, 4) for b in BUCKETS})


def _ledger(runs: Sequence[dict]) -> Dict[str, dict]:
    """Pooled dopamine ledger per score bucket, plus the learned P(play)."""
    out: Dict[str, dict] = {}
    for b in BUCKETS:
        row = dict(plays=0, reward=0, punish_immediate=0,
                   punish_trace_weight=0.0, n_terminal_pulses=0,
                   rewarded_but_lost=0)
        for r in runs:
            src = r["train"]["ledger_by_bucket"].get(b)
            if not src:
                continue
            for k in row:
                row[k] += src.get(k, 0)
        row["punish_trace_weight"] = round(float(row["punish_trace_weight"]), 1)
        row["net"] = round(row["reward"] - row["punish_immediate"]
                           - row["punish_trace_weight"], 1)
        row["p_play_greedy"] = _mean_bucket(runs, C3.GREEDY)[b]
        row["p_play_sampled"] = _mean_bucket(runs, C3.SAMPLED)[b]
        out[b] = row
    return out


def decide(payload: dict) -> dict:
    """The preregistered decision rule, applied to the greedy primary mode."""
    conds = payload["conditions"]
    fam = [f for f in C4.NOVEL_FAMILIES if f in conds]
    praw = [conds[f]["modes"][C3.GREEDY]["vs_discard"]["p_two_sided"] for f in fam]
    padj = C4.holm(praw) if praw else []
    holm_map = {f: dict(p_raw=praw[i], p_holm=round(padj[i], 4))
                for i, f in enumerate(fam)}
    verdicts: Dict[str, dict] = {}
    for name, row in conds.items():
        g = row["modes"][C3.GREEDY]
        c1 = bool(g["vs_frozen"]["ci_lo"] > 0.0)
        c2_ci = bool(g["vs_discard"]["ci_lo"] > 0.0)
        h = holm_map.get(name)
        c2_p = bool(h is not None and h["p_holm"] < 0.05)
        novel = name in C4.NOVEL_FAMILIES
        if not novel:
            verdict = ("reference" if name == "A" else "sensitivity only")
            success = None
        else:
            success = bool(c1 and c2_ci and c2_p)
            verdict = ("SUCCESS" if success else
                       "learned, still not competitive" if c1 else "no effect")
        verdicts[name] = dict(beats_frozen=c1, beats_discard_ci=c2_ci,
                              holm=h, success=success, verdict=verdict)
    any_success = any(v["success"] for v in verdicts.values() if v["success"])
    return dict(novel_families=fam, holm=holm_map, verdicts=verdicts,
                any_condition_beat_always_discard=bool(any_success))


def _ci(d: dict) -> str:
    return f"{d['delta']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"


def to_markdown(payload: dict, decision: dict) -> str:
    L: List[str] = []
    conds = payload["conditions"]
    L.append("### Conditions x evaluation mode, 400 paired seeds (400000-400399)\n")
    L.append("| condition | reward | gamma | mode | clear | per-run | vs frozen "
             "[95% CI] | vs always-discard [95% CI] | P(play\\|legal) | grad |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, row in conds.items():
        for mode in C3.EVAL_MODES:
            m = row["modes"][mode]
            per = " / ".join(f"{x:.3f}" for x in m["per_run_clear"])
            L.append(
                f"| {name} | {row['reward_rule']} | "
                f"{row['gamma'] if row['gamma'] is not None else '-'} | {mode} | "
                f"**{m['clear_rate']:.3f}** | {per} | {_ci(m['vs_frozen'])} | "
                f"{_ci(m['vs_discard'])} | {m['p_play_when_legal']:.3f} | "
                f"{m['bucket_gradient']:+.3f} |")
    L.append("")
    L.append(f"Shared frozen control: greedy {payload['frozen']['greedy']:.4f}, "
             f"sampled {payload['frozen']['sampled']:.4f}.\n")

    L.append("### Baselines on the same 400 games\n")
    L.append("| baseline | clear |")
    L.append("|---|---|")
    for k, v in payload["baselines"].items():
        L.append(f"| {k} | {v:.4f} |")
    L.append("")

    L.append("### Decision (greedy, preregistered)\n")
    L.append("| condition | beats frozen | beats always-discard (CI) | "
             "bootstrap p | Holm p | verdict |")
    L.append("|---|---|---|---|---|---|")
    for name, v in decision["verdicts"].items():
        h = v["holm"]
        L.append(f"| {name} | {'yes' if v['beats_frozen'] else 'no'} | "
                 f"{'yes' if v['beats_discard_ci'] else 'no'} | "
                 f"{h['p_raw']:.4f} | {h['p_holm']:.4f} | {v['verdict']} |"
                 if h else
                 f"| {name} | {'yes' if v['beats_frozen'] else 'no'} | "
                 f"{'yes' if v['beats_discard_ci'] else 'no'} | - | - | "
                 f"{v['verdict']} |")
    L.append("")
    L.append(f"**Anything beat always-discard: "
             f"{decision['any_condition_beat_always_discard']}**\n")

    L.append("### Dopamine ledger per score bucket (pooled over the 3 training runs)\n")
    for name, row in conds.items():
        L.append(f"**{name}** -- {row['label']}\n")
        L.append("| bucket | plays | reward | punish (immediate) | punish "
                 "(trace weight) | terminal pulses | rewarded-but-lost | net | "
                 "P(play) greedy | P(play) sampled |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for b in BUCKETS:
            d = row["ledger"][b]
            L.append(f"| `{b}` | {d['plays']} | {d['reward']} | "
                     f"{d['punish_immediate']} | {d['punish_trace_weight']:.1f} | "
                     f"{d['n_terminal_pulses']} | {d['rewarded_but_lost']} | "
                     f"{d['net']:+.1f} | {d['p_play_greedy']:.2f} | "
                     f"{d['p_play_sampled']:.2f} |")
        L.append("")

    L.append("### How bucket-selective the terminal pulse is\n")
    L.append("One lost blind fires one pulse over the union of that blind's recent "
             "plays, so `pulse share` is the fraction of terminal pulses that "
             "contain **any** play from that bucket.\n")
    L.append("| condition | terminal pulses | " + " | ".join(f"`{b}`" for b in BUCKETS)
             + " |")
    L.append("|---|---|" + "---|" * len(BUCKETS))
    for name, row in conds.items():
        ts = row["trace_specificity"]
        if not ts["n_pulses"]:
            continue
        L.append(f"| {name} | {ts['n_pulses']} | "
                 + " | ".join(f"{ts['pulse_share'][b]:.2f}" for b in BUCKETS) + " |")
    L.append("")

    L.append("### Exact McNemar per training run (greedy, 400 paired games)\n")
    L.append("| condition | run | vs frozen b10/b01, p | vs always-discard b10/b01, p |")
    L.append("|---|---|---|---|")
    for name, row in conds.items():
        m = row["modes"][C3.GREEDY]
        for i in range(row["n_runs"]):
            a, b = m["mcnemar_vs_frozen"][i], m["mcnemar_vs_discard"][i]
            L.append(f"| {name} | s{i} | {a['b10']}/{a['b01']}, p={a['p_exact']:.2e} | "
                     f"{b['b10']}/{b['b01']}, p={b['p_exact']:.2e} |")
    L.append("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)

    payload = condition_rows(out_dir)
    decision = decide(payload)
    payload["decision"] = decision
    C4.write_json(out_dir / "summary.json", payload)
    md = to_markdown(payload, decision)
    (out_dir / "summary.md").write_text(md + "\n")
    print(md)
    print(f"\nwrote {out_dir/'summary.json'} and {out_dir/'summary.md'}")


if __name__ == "__main__":
    main()
