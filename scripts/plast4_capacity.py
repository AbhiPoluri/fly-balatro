"""Post-hoc probe: can this readout separate `lt0.5` from `lt1.0` at all?

**Added after the v4 result, and labelled as post-hoc everywhere it is quoted.**
It is not a preregistered condition, it plays no game and it uses no labels or
outcomes: it takes a trained fly, delivers punishment pulses to a bucket's own
odours, and reads `play_drive` off the MBONs. Same category as
`scripts/plast3_transfer.py`.

Why it exists. v4 concluded that the fly stops at `play iff bucket >= lt0.5`
(0.448) rather than `play iff bucket >= lt1.0` (0.530) because terminal credit
cannot target one bucket -- 85% of terminal pulses contain an `lt0.5` play and
65% contain an `lt1.0` play. That conclusion presumes the alternative: that the
approach-minus-avoidance drive *could* put `lt0.5` below zero and `lt1.0` above
it if a perfectly selective signal existed. The fly never demonstrated that. It
demonstrated `lt0.25 -> 0` with `lt1.0` at 1, which is a separation between
**non-adjacent** buckets.

So this asks the question directly: drive the `lt0.5` odours down with a
perfectly selective punishment -- the best any reinforcement could do -- and
watch what happens to `lt1.0`.

Two arms:

* ``anchor`` -- pulses against **one** `lt0.5` odour's Kenyon-cell eligibility,
  which measures transfer;
* ``bucket`` -- pulses cycling over **every** `lt0.5` odour, which is the
  perfectly bucket-selective signal and therefore the ceiling on what any reward
  or credit rule could achieve at this stage.

The answer is whichever of these happens first as pulses accumulate: the mean
`lt0.5` drive crosses zero (the representation can express 0.530, so the v4
verdict of a credit-assignment limit stands), or the `lt1.0` drive crosses with
it (a capacity limit at KC -> MBON, and the verdict flips).

Run::

    python -m scripts.plast4_capacity
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from flybalatro import plasticity as P
from scripts import plast3_common as C3
from scripts import plast4_common as C4
from scripts import plast_common as K
from scripts.plast_probe import collect_calibration_hands, hand_bits

BUCKETS = ("lt0.25", "lt0.5", "lt1.0", "ge1.0")
CONFIG = K.ROOT / "outputs" / "plast3" / "tuned_config_sep.json"
ETA_FILE = K.ROOT / "outputs" / "plast3" / "eta_calib_sep.json"


def odour_table(n_hands: int = 600, seed0: int = K.CALIB_SEED0
                ) -> "OrderedDict[int, dict]":
    """Distinct separated-encoding odours seen at real decision points, by bucket.

    Drawn from the calibration seed range with the coin-flip policy, so the set is
    not biased by any learned policy, and never from the 400 evaluation seeds.
    """
    recs = collect_calibration_hands(n_hands, seed0, rng_seed=7)
    bits = hand_bits(recs)
    out: "OrderedDict[int, dict]" = OrderedDict()
    for r, b in zip(recs, bits):
        eff = C3.effective_bits(C3.ENC_SEPARATED, b)
        key = int(K._odor_key(eff))
        row = out.setdefault(key, dict(bits=b.copy(), bucket=str(r["bucket_name"]),
                                       n=0))
        row["n"] += 1
    return out


def drives(setup: K.Setup, dec, table) -> Dict[int, float]:
    """`play_drive` for every odour, at the fly's current weights."""
    return {k: dec.play_drive(setup.run_window(v["bits"])) for k, v in table.items()}


def by_bucket(vals: Dict[int, float], table) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for k, d in vals.items():
        b = table[k]["bucket"]
        out.setdefault(b, dict(vals=[], keys=[]))
        out[b]["vals"].append(float(d))
        out[b]["keys"].append(k)
    return {b: dict(n=len(v["vals"]), mean=float(np.mean(v["vals"])),
                    min=float(np.min(v["vals"])), max=float(np.max(v["vals"])),
                    frac_play=float(np.mean([x >= 0.0 for x in v["vals"]])))
            for b, v in out.items()}


def target_weight_stats(setup: K.Setup, table, targets: Sequence[int]) -> dict:
    """State of the synapses a punishment pulse on ``targets`` can actually reach.

    The punish arm depresses KC -> **approach** MBON edges, gated by the Kenyon
    cells this odour drives. If those edges are already at ``WEIGHT_FLOOR``, no
    further punishment can move the drive whatever its strength -- which is a
    different failure from not knowing which bucket to punish.
    """
    plast = setup.plast
    ts = plast.targets["punish"]
    active = np.zeros(plast.n_kc, bool)
    for key in targets:
        counts = setup.run_window(table[key]["bits"])
        active |= plast.eligibility(counts[setup.kc]) > 0.0
    m = active[ts.kc_row]
    if not m.any():
        return dict(n_edges=0)
    w = setup.brain.weight[ts.edge[m]]
    ratio = w / ts.w0[m]
    return dict(n_edges=int(m.sum()),
                n_kc_active=int(active.sum()),
                mean_ratio=round(float(ratio.mean()), 5),
                frac_at_floor=round(float((ratio <= plast.floor_frac + 1e-6).mean()), 4),
                frac_unchanged=round(float((ratio >= 1.0 - 1e-6).mean()), 4))


def run_arm(setup: K.Setup, dec, table, weights: Optional[Path],
            targets: Sequence[int], eta: float, n_pulses: int, arm: str) -> dict:
    """Deliver ``n_pulses`` punishment pulses to ``targets`` and track every bucket."""
    setup.plast.reset_weights()
    if weights is not None:
        setup.plast.load_weights(weights)
    before = target_weight_stats(setup, table, targets)
    trace: List[dict] = []
    trace.append(dict(pulse=0, buckets=by_bucket(drives(setup, dec, table), table)))
    for i in range(1, n_pulses + 1):
        key = targets[(i - 1) % len(targets)]
        counts = setup.run_window(table[key]["bits"])
        setup.plast.deliver("punish", counts[setup.kc], eta)
        trace.append(dict(pulse=i,
                          buckets=by_bucket(drives(setup, dec, table), table)))
    after = target_weight_stats(setup, table, targets)
    return dict(arm=arm, n_targets=len(targets), eta=eta, trace=trace,
                reachable_synapses_before=before, reachable_synapses_after=after)


def crossings(arm: dict) -> dict:
    """First pulse at which each bucket's mean drive goes negative, if ever."""
    out: Dict[str, Optional[int]] = {}
    for b in BUCKETS:
        out[b] = None
        for step in arm["trace"]:
            row = step["buckets"].get(b)
            if row is not None and row["mean"] < 0.0:
                out[b] = int(step["pulse"])
                break
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(C4.OUT_DIR / "weights_C80_s0.npz"))
    ap.add_argument("--pulses", type=int, default=120)
    ap.add_argument("--out", default=str(C4.OUT_DIR / "capacity.json"))
    a = ap.parse_args(argv)

    cfg = json.loads(CONFIG.read_text())
    graph = K.C.load(5)
    setup = C3.build_setup3(C3.ENC_SEPARATED, cfg, graph)
    dec = K.make_decider(setup, explore_floor=C3.EXPLORE_FLOOR3)
    meta = P.read_weights(a.weights)["meta"]
    dec.bias = float(meta["bias"])
    dec.temperature = float(meta["temperature"])
    eta = float(json.loads(ETA_FILE.read_text())["chosen"]["0.05"]["eta_punish"])

    table = odour_table()
    lt05 = [k for k, v in table.items() if v["bucket"] == "lt0.5"]
    if not lt05:
        raise SystemExit("no lt0.5 odours found")
    anchor = [max(lt05, key=lambda k: table[k]["n"])]

    print(f"weights {C4.rel_path(Path(a.weights))}  bias {dec.bias:.4f}  "
          f"temperature {dec.temperature:.4f}  eta_punish {eta:g}")
    print(f"{len(table)} distinct separated odours; "
          + ", ".join(f"{b}: {sum(1 for v in table.values() if v['bucket']==b)}"
                      for b in BUCKETS))

    arms = {}
    for name, targets, w in (("anchor", anchor, Path(a.weights)),
                             ("bucket", lt05, Path(a.weights)),
                             ("bucket_from_naive", lt05, None)):
        arm = run_arm(setup, dec, table, w, targets, eta, a.pulses, name)
        arm["crossings"] = crossings(arm)
        arms[name] = arm
        print(f"\n== arm {name}: {len(targets)} target odour(s), "
              f"{a.pulses} pulses at eta {eta:g}")
        print(f"   {'pulse':>5s} " + " ".join(f"{b:>18s}" for b in BUCKETS))
        for step in arm["trace"]:
            if step["pulse"] % 20 and step["pulse"] != a.pulses:
                continue
            cells = " ".join(
                f"{step['buckets'][b]['mean']:+8.3f}/{step['buckets'][b]['frac_play']:.2f}"
                .rjust(18) if b in step["buckets"] else " " * 18 for b in BUCKETS)
            print(f"   {step['pulse']:5d} {cells}")
        print(f"   first pulse where the mean drive goes negative: "
              f"{arm['crossings']}")
        print(f"   reachable KC -> approach synapses before: "
              f"{arm['reachable_synapses_before']}")
        print(f"   reachable KC -> approach synapses after:  "
              f"{arm['reachable_synapses_after']}")

    setup.plast.reset_weights()
    payload = dict(
        weights=C4.rel_path(Path(a.weights)), eta_punish=eta,
        bias=dec.bias, temperature=dec.temperature,
        n_odours=len(table),
        odours_by_bucket={b: sum(1 for v in table.values() if v["bucket"] == b)
                          for b in BUCKETS},
        anchor_odour=int(anchor[0]), n_lt05_odours=len(lt05),
        arms={k: dict(arm=v["arm"], n_targets=v["n_targets"], eta=v["eta"],
                      crossings=v["crossings"],
                      reachable_synapses_before=v["reachable_synapses_before"],
                      reachable_synapses_after=v["reachable_synapses_after"],
                      trace=[s for s in v["trace"]
                             if s["pulse"] % 5 == 0 or s["pulse"] <= 10])
              for k, v in arms.items()},
        note=("post-hoc, added after the v4 result; label-free, no game played, "
              "no condition promoted by it"),
    )
    C4.write_json(Path(a.out), payload)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
