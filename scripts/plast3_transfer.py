"""Bucket-only conditioning transfer: the test the v2 "Hamming distance 2" claim needed.

v2 concluded that the encoding, not the plasticity, was the remaining bottleneck:
the score-vs-needed bucket is one bit of 32, so two hands that differ only in
whether they can clear the blind sit at Hamming distance 2, where
``outputs/plast2/neighbour.json`` measured 0.80-0.84 conditioning transfer. That
probe binned **arbitrary** held-out relay patterns by Hamming distance, so its
distance-2 bin mixes bucket changes with best-slot-mask changes and it does not
establish the claim.

This one holds a real decision-point pattern fixed and changes **only** the score
bucket, one bucket bit off, one on, every other bit identical, which is
exactly the discrimination the task asks the fly to make. A hand-type-only
contrast (bucket held, hand type changed) is measured on the same anchors, so
"everything moves together" and "this axis moves together" can be told apart.

Protocol as ``scripts/plast2_neighbour.py``: condition the anchor with 20 pulses
at eta 0.05, measure ``delta play_drive`` on the anchor and on each variant,
report ``relative spread = delta(probe) / delta(anchor)``. Both dopamine arms,
both encodings.

Run::

    python -m scripts.plast3_transfer        # ~6 min
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from scripts import kc_homeo as H
from scripts import plast3_common as C
from scripts import plast_common as K
from scripts.plast_probe import collect_calibration_hands, hand_bits

OUT_JSON = C.OUT_DIR / "transfer.json"
N_ANCHORS = 16
N_PULSES = 20
ETA = 0.05
HAND_TYPE_BITS = C.HAND_TYPE_BITS
BUCKET_BITS = C.BUCKET_BITS


def set_one_hot(bits: NDArray[np.float32], block: Sequence[int],
                on: int) -> NDArray[np.float32]:
    out = np.asarray(bits, np.float32).copy()
    for b in block:
        out[b] = 0.0
    out[on] = 1.0
    return out


def variants(anchor: NDArray[np.float32]) -> Dict[str, List[dict]]:
    """The bucket-only and hand-type-only neighbours of one real pattern."""
    cur_bucket = int([b for b in BUCKET_BITS if anchor[b] > 0][0])
    cur_type = int([b for b in HAND_TYPE_BITS if anchor[b] > 0][0])
    out: Dict[str, List[dict]] = {"bucket_only": [], "hand_type_only": []}
    for b in BUCKET_BITS:
        if b == cur_bucket:
            continue
        v = set_one_hot(anchor, BUCKET_BITS, b)
        out["bucket_only"].append(dict(
            bits=v, changed_to=int(b),
            hamming=int((v != anchor).sum()),
            bucket_step=abs(BUCKET_BITS.index(b) - BUCKET_BITS.index(cur_bucket))))
    for b in HAND_TYPE_BITS:
        if b == cur_type:
            continue
        v = set_one_hot(anchor, HAND_TYPE_BITS, b)
        out["hand_type_only"].append(dict(
            bits=v, changed_to=int(b), hamming=int((v != anchor).sum()),
            bucket_step=None))
    return out


def condition(setup: K.Setup, dec, anchor: NDArray[np.float32],
              probes: Sequence[NDArray[np.float32]], kind: str, eta: float,
              n_pulses: int) -> dict:
    setup.plast.reset_weights()
    before_a = dec.play_drive(setup.run_window(anchor))
    before = [dec.play_drive(setup.run_window(p)) for p in probes]
    for _ in range(n_pulses):
        counts = setup.run_window(anchor)
        setup.plast.deliver(kind, counts[setup.kc], eta)
    after_a = dec.play_drive(setup.run_window(anchor))
    after = [dec.play_drive(setup.run_window(p)) for p in probes]
    setup.plast.reset_weights()
    return dict(delta_self=round(after_a - before_a, 4),
                deltas=[round(x - y, 4) for x, y in zip(after, before)])


def summarise(rows: Sequence[dict]) -> Dict[str, dict]:
    """Relative spread per (arm, contrast), two ways, as plast2_neighbour reports."""
    out: Dict[str, dict] = {}
    for kind in ("reward", "punish"):
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        self_abs = float(np.mean([abs(r["delta_self"]) for r in sel]))
        for contrast in ("bucket_only", "hand_type_only"):
            rel: List[float] = []
            probe_abs: List[float] = []
            for r in sel:
                for d in r[contrast]:
                    probe_abs.append(abs(d))
                    if r["delta_self"] != 0.0:
                        rel.append(d / r["delta_self"])
            out[f"{kind}_{contrast}"] = dict(
                n=len(probe_abs), n_anchors=len(sel),
                mean_relative_spread=round(float(np.mean(rel)), 4) if rel else None,
                median_relative_spread=round(float(np.median(rel)), 4) if rel else None,
                sd=round(float(np.std(rel)), 4) if rel else None,
                spread_ratio_of_means=round(float(np.mean(probe_abs) / self_abs), 4)
                if probe_abs and self_abs else None,
                mean_abs_delta_probe=round(float(np.mean(probe_abs)), 4)
                if probe_abs else None,
                mean_abs_delta_self=round(self_abs, 4),
            )
    return out


def pick_anchors(patterns: NDArray[np.float32], n: int,
                 seed: int = 3) -> List[int]:
    """Anchors spread over hand type x bucket, from the held-out bc2 patterns."""
    rng = np.random.default_rng(seed)
    key = [(int(np.argmax(p[list(HAND_TYPE_BITS)])),
            int(np.argmax(p[list(BUCKET_BITS)]))) for p in patterns]
    by: Dict[tuple, List[int]] = {}
    for i, k in enumerate(key):
        by.setdefault(k, []).append(i)
    keys = sorted(by)
    order = [keys[i] for i in rng.permutation(len(keys))]
    out: List[int] = []
    while len(out) < n and order:
        for k in list(order):
            cand = by[k]
            out.append(int(cand[rng.integers(0, len(cand))]))
            if len(out) >= n:
                break
    return out[:n]


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", type=int, default=N_ANCHORS)
    ap.add_argument("--pulses", type=int, default=N_PULSES)
    ap.add_argument("--eta", type=float, default=ETA)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--out", default=str(OUT_JSON))
    a = ap.parse_args(argv)

    t0 = time.time()
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = K.C.load(5)
    gate = H.relay_patterns()["gate"]
    idx = pick_anchors(gate, a.anchors)
    plan = [dict(index=int(i), bits=gate[i], **variants(gate[i])) for i in idx]

    recs = collect_calibration_hands(a.n_calib, K.CALIB_SEED0, rng_seed=7)
    calib_bits = hand_bits(recs)

    payload: dict = dict(
        n_anchors=len(plan), n_pulses=a.pulses, eta=a.eta,
        anchors=[dict(index=int(p["index"]),
                      hand_type=int(np.argmax(p["bits"][list(HAND_TYPE_BITS)])),
                      bucket=int(np.argmax(p["bits"][list(BUCKET_BITS)])),
                      n_bucket_variants=len(p["bucket_only"]),
                      n_hand_type_variants=len(p["hand_type_only"]))
                 for p in plan],
        note=("relative spread = delta(probe) / delta(anchor) on play_drive; 1.0 "
              "means the conditioning transferred completely. bucket_only holds "
              "the whole relay pattern fixed and changes only the score bucket; "
              "hand_type_only holds the bucket and changes the hand type."),
        conditions={},
    )

    configs = {
        C.ENC_CURRENT: K.ROOT / "outputs" / "plast2" / "tuned_config.json",
        C.ENC_SEPARATED: C.OUT_DIR / "tuned_config_sep.json",
    }
    for enc, cfg_path in configs.items():
        if not cfg_path.exists():
            print(f"skip {enc}: {cfg_path} missing")
            continue
        cfg = json.loads(cfg_path.read_text())
        setup = C.build_setup3(enc, cfg, graph)
        dec = K.make_decider(setup, explore_floor=C.EXPLORE_FLOOR3)
        raw = [dec.raw_drive(setup.run_window(b)) for b in calib_bits]
        cal = dec.calibrate(raw, explore=0.20)
        rows: List[dict] = []
        for kind in ("reward", "punish"):
            for p in plan:
                probes = [v["bits"] for v in p["bucket_only"]] + \
                         [v["bits"] for v in p["hand_type_only"]]
                r = condition(setup, dec, p["bits"], probes, kind, a.eta, a.pulses)
                nb = len(p["bucket_only"])
                rows.append(dict(anchor=int(p["index"]), kind=kind,
                                 delta_self=r["delta_self"],
                                 bucket_only=r["deltas"][:nb],
                                 hand_type_only=r["deltas"][nb:],
                                 bucket_hamming=[v["hamming"] for v in p["bucket_only"]],
                                 hand_type_hamming=[v["hamming"]
                                                    for v in p["hand_type_only"]]))
        by = summarise(rows)
        payload["conditions"][enc] = dict(
            config=C.rel_path(cfg_path), calibration=cal,
            per_anchor=rows, summary=by,
            mean_delta_self={k: round(float(np.mean(
                [r["delta_self"] for r in rows if r["kind"] == k])), 4)
                for k in ("reward", "punish")})
        print(f"== {enc}")
        for k, v in by.items():
            print(f"   {k:26s} n={v['n']:4d}  median {v['median_relative_spread']}"
                  f"  mean {v['mean_relative_spread']}  ratio-of-means "
                  f"{v['spread_ratio_of_means']}", flush=True)
        del setup

    payload["wall_seconds"] = round(time.time() - t0, 1)
    C.write_json(Path(a.out), payload)
    print(f"wrote {a.out} in {payload['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
