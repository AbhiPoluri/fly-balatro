"""The 2x2 table, with the paired / clustered uncertainty the preregistration fixed.

Reads ``outputs/plast3/run_*.json`` and ``baselines3.json``, applies the two
comparisons named in ``PREREGISTRATION.md`` section 5 -- against the cell's own
frozen control and against always-discard, both on the same 400 evaluation
seeds -- and writes ``outputs/plast3/summary.json`` plus a markdown table.

Run::

    python -m scripts.plast3_summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts import plast3_common as C
from scripts import plast_common as K

OUT = C.OUT_DIR


def load_runs(out_dir: Path) -> Dict[str, dict]:
    return {p.stem[len("run_"):]: json.loads(p.read_text())
            for p in sorted(out_dir.glob("run_*.json"))}


def cell_rows(runs: Dict[str, dict]) -> Dict[str, List[str]]:
    """Cell name -> its training-run names, in seed order."""
    cells: Dict[str, List[str]] = {}
    for name, p in runs.items():
        c = p["cell"]
        if not c["plasticity"]:
            continue
        cells.setdefault(f"{c['punishment']}_{c['encoding']}", []).append(name)
    return {k: sorted(v) for k, v in cells.items()}


def vec(payload: dict, mode: str) -> np.ndarray:
    return np.asarray([1.0 if x else 0.0 for x in payload["eval"][mode]["cleared"]],
                      np.float64)


def mean_of(payload: dict, mode: str, key: str) -> float:
    return float(payload["eval"][mode]["summary"][key])


def summarise(out_dir: Path, n_boot: int = 10_000) -> dict:
    runs = load_runs(out_dir)
    base = json.loads((out_dir / "baselines3.json").read_text())
    cells = cell_rows(runs)
    ad = np.asarray([1.0 if x else 0.0
                     for x in base["always_discard"]["cleared"]], np.float64)

    payload: dict = dict(
        eval_seeds=dict(seed0=C.EVAL3_SEED0, n=len(ad)),
        primary_mode=C.GREEDY,
        decision_rule=("a cell succeeds only if, under greedy evaluation, its "
                       "paired delta beats its own frozen control AND beats "
                       "always-discard, both with a 95% two-way clustered CI "
                       "excluding 0 (PREREGISTRATION.md section 5)"),
        baselines={k: dict(clear_rate=v["summary"]["clear_rate"],
                           chips_mean=round(v["summary"]["chips_mean"], 1))
                   for k, v in base.items() if isinstance(v, dict)},
        frozen={}, cells={},
    )
    for enc in C.ENCODINGS:
        f = runs.get(f"frozen_{enc}")
        if f is None:
            continue
        payload["frozen"][enc] = {
            m: dict(clear_rate=mean_of(f, m, "clear_rate"),
                    chips_mean=round(mean_of(f, m, "chips_mean"), 1),
                    p_play_when_legal=round(mean_of(f, m, "p_play_when_legal"), 4),
                    bucket_gradient=f["eval"][m]["bucket_gradient"])
            for m in C.EVAL_MODES}

    for cell, names in sorted(cells.items()):
        enc = runs[names[0]]["cell"]["encoding"]
        frozen = runs.get(f"frozen_{enc}")
        row: dict = dict(runs=names, encoding=enc,
                         punishment=runs[names[0]]["cell"]["punishment"],
                         eta_reward=runs[names[0]]["cell"]["eta_reward"],
                         eta_punish=runs[names[0]]["cell"]["eta_punish"],
                         dopamine_counts=[runs[n]["train"]["dopamine_counts"]
                                          for n in names],
                         train_reward_rate=[round(
                             runs[n]["train"]["summary"]["reward_rate"], 4)
                             for n in names])
        for m in C.EVAL_MODES:
            vs = [vec(runs[n], m) for n in names]
            r_frozen = C.cluster_bootstrap(vs, vec(frozen, m), n_boot=n_boot)
            r_base = C.cluster_bootstrap(vs, ad, n_boot=n_boot)
            row[m] = dict(
                clear_rate=round(float(np.mean([v.mean() for v in vs])), 4),
                clear_rate_per_run=[round(float(v.mean()), 4) for v in vs],
                chips_mean=round(float(np.mean(
                    [mean_of(runs[n], m, "chips_mean") for n in names])), 1),
                p_play_when_legal=round(float(np.mean(
                    [mean_of(runs[n], m, "p_play_when_legal") for n in names])), 4),
                bucket_gradient=round(float(np.mean(
                    [runs[n]["eval"][m]["bucket_gradient"] for n in names])), 4),
                joint_variance=round(float(np.mean(
                    [runs[n]["eval"][m]["joint_p_play"]["variance"]
                     for n in names])), 6),
                vs_frozen=r_frozen.to_dict(),
                vs_always_discard=r_base.to_dict(),
                passes_prereg=bool(r_frozen.lo > 0 and r_base.lo > 0),
            )
        payload["cells"][cell] = row
    payload["verdict"] = dict(
        any_cell_passes=bool(any(v[C.GREEDY]["passes_prereg"]
                                 for v in payload["cells"].values())),
        cells_beating_always_discard_greedy=[
            k for k, v in payload["cells"].items()
            if v[C.GREEDY]["vs_always_discard"]["ci_lo"] > 0],
        cells_beating_frozen_greedy=[
            k for k, v in payload["cells"].items()
            if v[C.GREEDY]["vs_frozen"]["ci_lo"] > 0],
    )
    return payload


def markdown(payload: dict) -> str:
    lines: List[str] = []
    n = payload["eval_seeds"]["n"]
    lines.append(f"### 2x2, {n} paired evaluation games (seeds "
                 f"{payload['eval_seeds']['seed0']}-"
                 f"{payload['eval_seeds']['seed0'] + n - 1})\n")
    lines.append("| cell | mode | clear | frozen | delta vs frozen [95% CI] | "
                 "delta vs always-discard [95% CI] | P(play\\|legal) | grad |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cell, v in sorted(payload["cells"].items()):
        enc = v["encoding"]
        for m in C.EVAL_MODES:
            d = v[m]
            fz = payload["frozen"][enc][m]["clear_rate"]
            f = d["vs_frozen"]
            b = d["vs_always_discard"]
            lines.append(
                f"| {cell} | {m} | **{d['clear_rate']:.3f}** | {fz:.3f} | "
                f"{f['delta']:+.3f} [{f['ci_lo']:+.3f}, {f['ci_hi']:+.3f}] | "
                f"{b['delta']:+.3f} [{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}] | "
                f"{d['p_play_when_legal']:.3f} | {d['bucket_gradient']:+.3f} |")
    lines.append("")
    lines.append("| baseline | clear |")
    lines.append("|---|---|")
    for k, v in payload["baselines"].items():
        lines.append(f"| {k} | {v['clear_rate']:.3f} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    payload = summarise(out_dir, a.n_boot)
    C.write_json(out_dir / "summary.json", payload)
    md = markdown(payload)
    (out_dir / "summary.md").write_text(md + "\n")
    print(md)
    print("\nverdict:", json.dumps(payload["verdict"], indent=2))


if __name__ == "__main__":
    main()
