"""How far does a conditioned valence spread, as a function of odour similarity?

Gate criterion (v) measures specificity against a panel of eight *deliberately
different* odours (one per hand-type x score-bucket pair), and the homeostatic code
passes it: 71% of the reward effect and 104% of the punishment effect stay on the
paired odour. That is the right test for "is the credit odour-specific at all", and
the wrong test for "can this fly separate the decisions the game asks it
to separate", because the decisions that matter are between *near* odours.

The relay block puts the score-vs-needed bucket in 1 of its 32 bits. Two Pair that
can clear the blind and Two Pair that cannot therefore differ in **two** bits (one
bucket bit off, one on) out of the ~7 that are on, and they are the pair the fly
has to tell apart to stop wasting plays. This script measures the leak directly:
condition one odour, then measure the change in every other held-out odour, binned
by Hamming distance from the conditioned one.

Run: ``python -m scripts.plast2_neighbour``   (~4 min)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from scripts import kc_homeo as H  # noqa: E402
from scripts import plast_common as K  # noqa: E402

OUT_JSON = H.OUT_DIR / "neighbour.json"
N_ANCHORS = 8
N_PROBES = 12
N_PULSES = 20
ETA = 0.05
#: Hamming-distance bins over the 32 relay bits.
BINS = ((2, 2), (4, 4), (6, 8), (10, 32))


def pick_anchors_and_probes(patterns: NDArray[np.float32], n_anchors: int,
                            n_probes: int, seed: int = 11) -> List[dict]:
    """For each anchor, a probe set spread over Hamming distance, incl. its nearest."""
    rng = np.random.default_rng(seed)
    P = patterns.astype(np.int8)
    out: List[dict] = []
    # prefer anchors that have a distance-2 neighbour in the set
    dist = (P[:, None, :] != P[None, :, :]).sum(axis=2)
    np.fill_diagonal(dist, 999)
    has_near = np.flatnonzero((dist == 2).any(axis=1))
    order = has_near[rng.permutation(len(has_near))][:n_anchors]
    for i in order:
        d = dist[i]
        probes: List[int] = []
        for lo, hi in BINS:
            cand = np.flatnonzero((d >= lo) & (d <= hi))
            if len(cand) == 0:
                continue
            take = cand[rng.permutation(len(cand))][: max(1, n_probes // len(BINS))]
            probes.extend(int(x) for x in take)
        out.append(dict(anchor=int(i), probes=probes,
                        distances=[int(d[j]) for j in probes]))
    return out


def condition(setup: K.Setup, dec, patterns: NDArray[np.float32], anchor: int,
              probes: Sequence[int], kind: str, eta: float, n_pulses: int) -> dict:
    setup.plast.reset_weights()
    idx = [anchor] + list(probes)
    before = {j: dec.play_drive(setup.run_window(patterns[j])) for j in idx}
    for _ in range(n_pulses):
        counts = setup.run_window(patterns[anchor])
        setup.plast.deliver(kind, counts[setup.kc], eta)
    after = {j: dec.play_drive(setup.run_window(patterns[j])) for j in idx}
    setup.plast.reset_weights()
    d_self = after[anchor] - before[anchor]
    return dict(anchor=int(anchor), kind=kind, delta_self=round(d_self, 4),
                deltas={int(j): round(after[j] - before[j], 4) for j in probes})


def summarise_bins(rows: Sequence[dict]) -> Dict[str, dict]:
    """Relative spread per Hamming bin, two ways.

    ``mean_relative_spread`` is the mean of per-probe ratios and is what the eye
    wants, but a small ``delta_self`` makes individual ratios explode.
    ``spread_ratio_of_means`` divides the mean |delta| over the probes in the bin by
    the mean |delta| on the anchors, which has no such tail.
    """
    out: Dict[str, dict] = {}
    for kind in ("reward", "punish"):
        sel = [r for r in rows if r["kind"] == kind]
        self_abs = float(np.mean([abs(r["delta_self"]) for r in sel])) if sel else 0.0
        for lo, hi in BINS:
            rel: List[float] = []
            probe_abs: List[float] = []
            for r in sel:
                for j, d in r["distances"].items():
                    if lo <= d <= hi:
                        probe_abs.append(abs(r["deltas"][j]))
                        if r["delta_self"] != 0.0:
                            rel.append(r["deltas"][j] / r["delta_self"])
            out[f"{kind}_hamming_{lo}_{hi}"] = dict(
                n=len(probe_abs),
                mean_relative_spread=round(float(np.mean(rel)), 4) if rel else None,
                sd=round(float(np.std(rel)), 4) if rel else None,
                median_relative_spread=round(float(np.median(rel)), 4) if rel else None,
                spread_ratio_of_means=round(float(np.mean(probe_abs) / self_abs), 4)
                if probe_abs and self_abs else None,
                mean_abs_delta_probe=round(float(np.mean(probe_abs)), 4)
                if probe_abs else None,
                mean_abs_delta_self=round(self_abs, 4),
            )
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", type=int, default=N_ANCHORS)
    ap.add_argument("--probes", type=int, default=N_PROBES)
    ap.add_argument("--pulses", type=int, default=N_PULSES)
    ap.add_argument("--eta", type=float, default=ETA)
    ap.add_argument("--recompute", action="store_true",
                    help="re-summarise an existing --out file, no simulation")
    ap.add_argument("--out", default=str(OUT_JSON))
    a = ap.parse_args(argv)

    if a.recompute:
        payload = json.loads(Path(a.out).read_text())
        for label, c in payload["conditions"].items():
            for r in c["per_anchor"]:
                r["deltas"] = {int(k): v for k, v in r["deltas"].items()}
                r["distances"] = {int(k): v for k, v in r["distances"].items()}
            c["by_bin"] = summarise_bins(c["per_anchor"])
            print(f"== {label}")
            for k, v in c["by_bin"].items():
                print(f"   {k:28s} n={v['n']:4d}  mean ratio "
                      f"{v['mean_relative_spread']}  median "
                      f"{v['median_relative_spread']}  ratio-of-means "
                      f"{v['spread_ratio_of_means']}")
        K.write_json(Path(a.out), payload)
        return

    t0 = time.time()
    graph = K.C.load(5)
    sets = H.relay_patterns()
    gate = sets["gate"]
    plan = pick_anchors_and_probes(gate, a.anchors, a.probes)

    payload: dict = dict(
        n_anchors=len(plan), n_pulses=a.pulses, eta=a.eta,
        bins=[list(b) for b in BINS],
        patterns="the 500 HELD-OUT relay patterns from outputs/bc2",
        note=("relative spread = delta(probe) / delta(anchor); 1.0 means the "
              "conditioning transferred completely, 0.0 means it did not transfer"),
        conditions={},
    )

    for label, off, sc in (("baseline_plast", None, None),
                           ("kc_homeostasis", np.asarray(
                               json.loads(H.HOMEO_JSON.read_text())["offsets"],
                               np.float32), None)):
        setup = H.build(H.base_config(), graph, off, sc)
        dec = K.make_decider(setup)
        raw = [dec.raw_drive(setup.run_window(b)) for b in gate[:120]]
        dec.calibrate(np.asarray(raw), explore=0.20)
        rows: List[dict] = []
        for kind in ("reward", "punish"):
            for p in plan:
                r = condition(setup, dec, gate, p["anchor"], p["probes"], kind,
                              a.eta, a.pulses)
                r["distances"] = {int(j): int(d)
                                  for j, d in zip(p["probes"], p["distances"])}
                rows.append(r)
        by_bin = summarise_bins(rows)
        payload["conditions"][label] = dict(
            per_anchor=rows, by_bin=by_bin,
            mean_delta_self={k: round(float(np.mean(
                [r["delta_self"] for r in rows if r["kind"] == k])), 4)
                for k in ("reward", "punish")})
        print(f"== {label}")
        for k, v in by_bin.items():
            print(f"   {k:28s} n={v['n']:4d}  mean ratio "
                  f"{v['mean_relative_spread']}  median {v['median_relative_spread']}"
                  f"  ratio-of-means {v['spread_ratio_of_means']}")
        del setup

    payload["wall_seconds"] = round(time.time() - t0, 1)
    K.write_json(Path(a.out), payload)
    print(f"wrote {a.out} in {payload['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
