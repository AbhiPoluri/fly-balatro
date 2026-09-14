"""Pre-registered sanity checks and the one-off decision calibration.

Nothing in the main protocol may run until this passes:

(a) **The decision depends on the odour.** With plasticity off, ``play_drive``
    must vary across hands. Reported by hand type *and* by score-vs-needed
    bucket (the decision-relevant axis), against the quantisation floor of the
    MBON pool means.
(b) **Classic aversive conditioning works.** Pair one fixed odour with 20
    punishment pulses and ``play_drive`` for that odour must drop while a
    different odour's drops less. The reward-paired version must move the other
    way.
(c) **Weight bounds hold.** No KC -> MBON weight ever goes negative, above its
    original value, or below ``WEIGHT_FLOOR`` x original.

It also freezes the two free numbers of the decision rule -- the bias and the
softmax temperature -- **per wiring**, on 200 decision states drawn from
calibration seeds that neither training nor evaluation ever sees, before any
dopamine is delivered.

Run: ``python -m scripts.plast_probe``  (~3 min)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts import plast_common as K  # noqa: E402
from flybalatro import plasticity as P  # noqa: E402

OUT = K.OUT_DIR


# --------------------------------------------------------------------------- #
def collect_calibration_hands(n_hands: int, seed0: int, rng_seed: int) -> List[dict]:
    """Decision states from coin-flip play/dig games, so the set is not policy-biased."""
    rng = np.random.default_rng(rng_seed)

    def coin(ctx: K.HandContext):
        if not ctx.discard_ok:
            return K.PLAY, {}
        return (K.PLAY if rng.random() < 0.5 else K.DISCARD), {}

    env = K.make_env()
    out: List[dict] = []
    seed = seed0
    while len(out) < n_hands:
        g = K.play_game(env, seed, coin)
        for h in g.hands:
            out.append(h)
            if len(out) >= n_hands:
                break
        seed += 1
    return out


def hand_bits(records: Sequence[dict]) -> np.ndarray:
    """Rebuild the 32-bit odour from the packed key in each record."""
    bits = np.zeros((len(records), 32), np.float32)
    for i, r in enumerate(records):
        k = int(r["odor"])
        for b in range(32):
            if k >> b & 1:
                bits[i, b] = 1.0
    return bits


def sweep_drives(setup: K.Setup, dec: P.Decider, bits: np.ndarray) -> dict:
    """Raw drive, pool rates and KC/MBON diagnostics for every odour."""
    kc = setup.kc
    mbon = setup.valence.mbon
    scale = 1000.0 / setup.window_ms
    raw = np.zeros(len(bits))
    appr = np.zeros(len(bits))
    avoid = np.zeros(len(bits))
    kc_act = np.zeros((len(bits), len(kc)), bool)
    mbon_hz = np.zeros(len(bits))
    for i, b in enumerate(bits):
        counts = setup.run_window(b)
        raw[i] = dec.raw_drive(counts)
        appr[i] = float(counts[dec.approach].mean()) * scale
        avoid[i] = float(counts[dec.avoid].mean()) * scale
        mbon_hz[i] = float(counts[mbon].mean()) * scale
        kc_act[i] = counts[kc] > 0
    return dict(raw=raw, approach_hz=appr, avoid_hz=avoid, kc_act=kc_act,
                mbon_hz=mbon_hz)


def group_stats(values: np.ndarray, labels: Sequence[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for lab in sorted(set(labels)):
        m = np.array([l == lab for l in labels])
        v = values[m]
        out[lab] = dict(n=int(m.sum()), mean=round(float(v.mean()), 4),
                        sd=round(float(v.std()), 4), min=round(float(v.min()), 4),
                        max=round(float(v.max()), 4))
    return out


# --------------------------------------------------------------------------- #
def conditioning_test(setup: K.Setup, dec: P.Decider, bits_a: np.ndarray,
                      bits_b: np.ndarray, kind: str, eta: float, n_trials: int,
                      always_on: np.ndarray) -> dict:
    """Pair odour A with ``n_trials`` dopamine pulses; measure A and B before/after."""
    setup.plast.reset_weights()
    before_a = dec.play_drive(setup.run_window(bits_a))
    before_b = dec.play_drive(setup.run_window(bits_b))
    traj: List[float] = []
    dw_total = 0.0
    dw_on_always = 0.0
    ts = setup.plast.targets[kind]
    on_always = always_on[ts.kc_row]
    for _ in range(n_trials):
        counts = setup.run_window(bits_a)
        kcc = counts[setup.kc]
        w_before = setup.brain.weight[ts.edge].copy()
        setup.plast.deliver(kind, kcc, eta)
        dw = np.abs(setup.brain.weight[ts.edge] - w_before, dtype=np.float64)
        dw_total += float(dw.sum())
        dw_on_always += float(dw[on_always].sum())
        traj.append(dec.play_drive(setup.run_window(bits_a)))
    after_a = dec.play_drive(setup.run_window(bits_a))
    after_b = dec.play_drive(setup.run_window(bits_b))
    stats = setup.plast.weight_stats()

    # share of the eligible edges whose presynaptic KC is always-on
    counts = setup.run_window(bits_a)
    elig = setup.plast.eligibility(counts[setup.kc])
    f = elig[ts.kc_row] > 0
    frac_always = float(always_on[ts.kc_row][f].mean()) if f.any() else 0.0

    setup.plast.reset_weights()
    return dict(
        kind=kind, eta=eta, n_trials=n_trials,
        drive_a_before=round(before_a, 4), drive_a_after=round(after_a, 4),
        drive_b_before=round(before_b, 4), drive_b_after=round(after_b, 4),
        delta_a=round(after_a - before_a, 4), delta_b=round(after_b - before_b, 4),
        trajectory=[round(x, 4) for x in traj],
        weight_stats=stats, sum_abs_dw=round(dw_total, 3),
        frac_dw_on_always_on_kc=round(dw_on_always / dw_total, 4) if dw_total else 0.0,
        frac_eligible_edges_always_on_kc=round(frac_always, 4),
    )


def conditioning_panel(setup: K.Setup, dec: P.Decider, bits: np.ndarray,
                       idx: Sequence[int], recs: Sequence[dict], kind: str,
                       eta: float, n_trials: int) -> dict:
    """Condition each odour in turn and measure it *and* the others.

    The classic result has two halves: the paired odour's valence moves, and a
    different odour's does not. Reporting only one pair cannot separate learning
    from a global shift, so every odour in ``idx`` is conditioned in its own
    fresh run and the mean change in the other seven is the generalisation term.
    """
    per: List[dict] = []
    for i in idx:
        setup.plast.reset_weights()
        base = {j: dec.play_drive(setup.run_window(bits[j])) for j in idx}
        for _ in range(n_trials):
            counts = setup.run_window(bits[i])
            setup.plast.deliver(kind, counts[setup.kc], eta)
        after = {j: dec.play_drive(setup.run_window(bits[j])) for j in idx}
        d_self = after[i] - base[i]
        d_other = float(np.mean([after[j] - base[j] for j in idx if j != i]))
        per.append(dict(
            index=int(i), bucket=recs[i]["bucket_name"],
            hand_type=recs[i]["best_type_name"],
            drive_before=round(base[i], 4), drive_after=round(after[i], 4),
            delta_self=round(d_self, 4), delta_other=round(d_other, 4),
            specificity=round(d_self - d_other, 4),
        ))
    setup.plast.reset_weights()
    ds = np.array([p["delta_self"] for p in per])
    do = np.array([p["delta_other"] for p in per])
    expect = 1.0 if kind == "reward" else -1.0
    return dict(
        kind=kind, eta=eta, n_trials=n_trials, n_odors=len(idx), per_odor=per,
        mean_delta_self=round(float(ds.mean()), 4),
        sd_delta_self=round(float(ds.std()), 4),
        mean_delta_other=round(float(do.mean()), 4),
        mean_specificity=round(float((ds - do).mean()), 4),
        specificity_share=round(float((ds - do).mean() / ds.mean()), 4)
        if ds.mean() != 0 else 0.0,
        direction_ok=bool(np.sign(ds.mean()) == expect),
        n_odors_right_direction=int((np.sign(ds) == expect).sum()),
        n_odors_more_than_others=int((np.abs(ds) > np.abs(do)).sum()),
    )


def pick_odor_panel(recs: Sequence[dict], n: int = 8) -> List[int]:
    """One representative hand per distinct (bucket, hand type) pair, in order."""
    seen = set()
    idx: List[int] = []
    for i, r in enumerate(recs):
        key = (r["bucket_name"], r["best_type_name"])
        if key in seen:
            continue
        seen.add(key)
        idx.append(i)
        if len(idx) >= n:
            break
    return idx


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--window-ms", type=float, default=None)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--explore", type=float, default=0.20)
    ap.add_argument("--explore-floor", type=float, default=0.0,
                    help="epsilon floor on the softmax: "
                         "p = eps/2 + (1-eps)*sigmoid(drive/T)")
    ap.add_argument("--valence", default="nt")
    ap.add_argument("--window-scan", default=None,
                    help="comma-separated window lengths for the "
                         "conditioning panel, e.g. 50,150,300")
    ap.add_argument("--out", default=str(OUT / "probe.json"))
    a = ap.parse_args(argv)

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    graph = K.C.load(5)
    cfg = K.load_config()
    payload: dict = dict(
        config=cfg, args=vars(a),
        note="sanity checks and frozen decision calibration; no protocol run here",
    )

    print("collecting calibration hands ...")
    recs = collect_calibration_hands(a.n_calib, K.CALIB_SEED0, rng_seed=7)
    bits = hand_bits(recs)
    types = [r["best_type_name"] for r in recs]
    buckets = [r["bucket_name"] for r in recs]
    payload["calibration_set"] = dict(
        n=len(recs), seeds=f"{K.CALIB_SEED0}+",
        distinct_odors=len(set(int(r["odor"]) for r in recs)),
        by_hand_type={k: types.count(k) for k in sorted(set(types))},
        by_bucket={k: buckets.count(k) for k in sorted(set(buckets))},
        discard_legal=int(sum(r["discard_ok"] for r in recs)),
    )
    print(f"  {len(recs)} hands, {payload['calibration_set']['distinct_odors']} odours")

    calib_out: Dict[str, dict] = {}
    for wiring in ("real", "shuffled"):
        print(f"== {wiring} wiring ==")
        setup = K.build_setup(wiring=wiring, valence_scheme=a.valence,
                              window_ms=a.window_ms, graph=graph, cfg=cfg)
        dec = K.make_decider(setup, explore_floor=a.explore_floor)
        sec = dict(valence=setup.valence.summary(),
                   plasticity=setup.plast.describe(),
                   window_ms=setup.window_ms)
        if wiring == "real":
            sec["valence_table"] = setup.valence.table()

        sw = sweep_drives(setup, dec, bits)
        raw = sw["raw"]
        n_a, n_v = len(dec.approach), len(dec.avoid)
        scale = 1000.0 / setup.window_ms
        quantum = (1.0 / n_a + 1.0 / n_v) * scale
        kc_rate = sw["kc_act"].mean(axis=0)
        always_on = kc_rate > 0.5
        sec["sanity_a"] = dict(
            n=len(raw), raw_mean=round(float(raw.mean()), 4),
            raw_sd=round(float(raw.std()), 4),
            raw_min=round(float(raw.min()), 4), raw_max=round(float(raw.max()), 4),
            raw_range=round(float(raw.max() - raw.min()), 4),
            distinct_values=int(len(np.unique(np.round(raw, 6)))),
            quantum_hz=round(quantum, 4),
            range_in_quanta=round(float((raw.max() - raw.min()) / quantum), 2),
            by_hand_type=group_stats(raw, types),
            by_bucket=group_stats(raw, buckets),
            approach_hz_mean=round(float(sw["approach_hz"].mean()), 3),
            avoid_hz_mean=round(float(sw["avoid_hz"].mean()), 3),
            mbon_hz_mean=round(float(sw["mbon_hz"].mean()), 3),
            kc_active_frac_mean=round(float(sw["kc_act"].mean()), 4),
            kc_active_per_state=round(float(sw["kc_act"].sum(axis=1).mean()), 1),
            kc_always_on=int(always_on.sum()),
            kc_input_dependent=int(((kc_rate > 0) & (kc_rate < 1)).sum()),
            kc_never=int((kc_rate == 0).sum()),
        )
        print(f"  raw drive {raw.min():.3f}..{raw.max():.3f} Hz "
              f"({sec['sanity_a']['range_in_quanta']} quanta), "
              f"KC active {sec['sanity_a']['kc_active_per_state']}, "
              f"MBON {sec['sanity_a']['mbon_hz_mean']:.2f} Hz")

        cal = dec.calibrate(raw, explore=a.explore)
        sec["calibration"] = cal
        sec["decider"] = dec.to_dict()
        calib_out[wiring] = dict(bias=dec.bias, temperature=dec.temperature,
                                 calibration=cal, decider=dec.to_dict())
        p_naive = np.array([1.0 / (1.0 + np.exp(-(x + dec.bias) / dec.temperature))
                            for x in raw])
        sec["naive_policy"] = dict(
            p_play_mean=round(float(p_naive.mean()), 4),
            **{"frac_outside_0.1_0.9":round(float(((p_naive < 0.1) | (p_naive > 0.9)).mean()), 4)},
            frac_below_0_1=round(float((p_naive < 0.1).mean()), 4),
            frac_above_0_9=round(float((p_naive > 0.9).mean()), 4),
            p_play_by_bucket={k: round(float(p_naive[np.array(buckets) == k].mean()), 4)
                              for k in sorted(set(buckets))},
            p_play_by_hand_type={k: round(float(p_naive[np.array(types) == k].mean()), 4)
                                 for k in sorted(set(types))},
            explore_floor=dec.explore_floor,
        )
        print(f"  bias {dec.bias:+.4f}  T {dec.temperature:.4f}  "
              f"explore {cal['explore_achieved']:.3f}  "
              f"naive P(play) {p_naive.mean():.3f}, "
              f"{sec['naive_policy']['frac_outside_0.1_0.9']:.0%} of odours near-deterministic")
        print("    naive P(play) by bucket:", sec["naive_policy"]["p_play_by_bucket"])

        # the honest conditioning test, for BOTH wirings: condition each of eight
        # odours in its own fresh run and measure it and the other seven.
        panel = pick_odor_panel(recs, 8)
        sec["conditioning_panel_odors"] = [
            dict(index=int(i), bucket=recs[i]["bucket_name"],
                 hand_type=recs[i]["best_type_name"]) for i in panel]
        sec["sanity_b_panel"] = {}
        for kind in ("punish", "reward"):
            r = conditioning_panel(setup, dec, bits, panel, recs, kind,
                                   a.eta, a.trials)
            sec["sanity_b_panel"][kind] = r
            print(f"    panel {kind}: dself {r['mean_delta_self']:+.3f}"
                  f"+-{r['sd_delta_self']:.3f}  dother {r['mean_delta_other']:+.3f}"
                  f"  specificity {r['mean_specificity']:+.3f}"
                  f" ({r['specificity_share']:.0%} of the effect)"
                  f"  direction_ok={r['direction_ok']}"
                  f"  {r['n_odors_right_direction']}/{r['n_odors']} right way")

        if wiring == "real":
            # pick two odours: the most common low-bucket and high-bucket hands
            def pick(bucket_name: str) -> Optional[int]:
                cand = [i for i, r in enumerate(recs) if r["bucket_name"] == bucket_name]
                if not cand:
                    return None
                keys = [int(recs[i]["odor"]) for i in cand]
                best = max(set(keys), key=keys.count)
                return cand[keys.index(best)]

            i_a = pick("lt0.25")
            i_b = pick("ge1.0")
            if i_a is None or i_b is None:
                order = np.argsort([r["bucket"] for r in recs])
                i_a, i_b = int(order[0]), int(order[-1])
            sec["conditioning_odors"] = dict(
                a=dict(index=int(i_a), hand_type=recs[i_a]["best_type_name"],
                       bucket=recs[i_a]["bucket_name"], best_score=recs[i_a]["best_score"],
                       needed=recs[i_a]["needed"], bits_on=recs[i_a]["bits_on"]),
                b=dict(index=int(i_b), hand_type=recs[i_b]["best_type_name"],
                       bucket=recs[i_b]["bucket_name"], best_score=recs[i_b]["best_score"],
                       needed=recs[i_b]["needed"], bits_on=recs[i_b]["bits_on"]),
            )
            print(f"  conditioning: A={recs[i_a]['best_type_name']}/"
                  f"{recs[i_a]['bucket_name']}  B={recs[i_b]['best_type_name']}/"
                  f"{recs[i_b]['bucket_name']}")
            sec["sanity_b"] = {}
            for kind in ("punish", "reward"):
                r = conditioning_test(setup, dec, bits[i_a], bits[i_b], kind,
                                      a.eta, a.trials, always_on)
                r["quantum_hz"] = round(quantum, 4)
                r["delta_a_in_quanta"] = round(r["delta_a"] / quantum, 2)
                r["delta_b_in_quanta"] = round(r["delta_b"] / quantum, 2)
                r["specific"] = bool(abs(r["delta_a"]) > abs(r["delta_b"]))
                sec["sanity_b"][kind] = r
                print(f"    {kind}: dA {r['delta_a']:+.4f} ({r['delta_a_in_quanta']:+.1f}q) "
                      f"dB {r['delta_b']:+.4f} ({r['delta_b_in_quanta']:+.1f}q) "
                      f"specific={r['specific']}")
            if a.window_scan:
                sec["window_scan"] = {}
                for wms in [float(x) for x in a.window_scan.split(",")]:
                    s2 = K.build_setup(wiring="real", valence_scheme=a.valence,
                                       window_ms=wms, graph=graph, cfg=cfg)
                    d2 = K.make_decider(s2)
                    raw2 = np.array([d2.raw_drive(s2.run_window(b)) for b in bits[:80]])
                    d2.calibrate(raw2, explore=a.explore)
                    row = dict(range_hz=round(float(raw2.max() - raw2.min()), 4),
                               sd_hz=round(float(raw2.std()), 4))
                    for kind in ("punish", "reward"):
                        rr = conditioning_panel(s2, d2, bits, panel, recs, kind,
                                                a.eta, a.trials)
                        row[kind] = {k: rr[k] for k in
                                     ("mean_delta_self", "mean_delta_other",
                                      "mean_specificity", "specificity_share",
                                      "direction_ok", "n_odors_right_direction")}
                    sec["window_scan"][f"{wms:g}"] = row
                    print(f"    window {wms:g} ms: range {row['range_hz']:.2f} Hz, "
                          f"punish dself {row['punish']['mean_delta_self']:+.2f} "
                          f"(spec {row['punish']['mean_specificity']:+.2f}), "
                          f"reward dself {row['reward']['mean_delta_self']:+.2f} "
                          f"(spec {row['reward']['mean_specificity']:+.2f})")
                    del s2
            # (c) bounds, after a deliberately saturating run
            setup.plast.reset_weights()
            for _ in range(200):
                counts = setup.run_window(bits[i_a])
                setup.plast.deliver("punish", counts[setup.kc], 0.5)
                setup.plast.deliver("reward", counts[setup.kc], 0.5)
            sec["sanity_c"] = setup.plast.weight_stats()
            sec["sanity_c"]["floor_frac"] = P.WEIGHT_FLOOR
            sec["sanity_c"]["ok"] = bool(
                not sec["sanity_c"]["any_negative"]
                and not sec["sanity_c"]["any_above_original"]
                and sec["sanity_c"]["min_ratio"] >= P.WEIGHT_FLOOR - 1e-6
            )
            print(f"  bounds ok={sec['sanity_c']['ok']} "
                  f"min_ratio={sec['sanity_c']['min_ratio']:.4f} "
                  f"at_floor={sec['sanity_c']['frac_at_floor']:.3f}")
            setup.plast.reset_weights()

        payload[wiring] = sec
        del setup

    payload["wall_seconds"] = round(time.time() - t0, 1)
    K.write_json(Path(a.out), payload)
    K.write_json(OUT / "calib.json", dict(
        args=vars(a), per_wiring=calib_out,
        calibration_hands=[dict(odor=int(r["odor"]), best_type_name=r["best_type_name"],
                                bucket_name=r["bucket_name"], best_score=r["best_score"],
                                needed=r["needed"], plays_before=r["plays_before"],
                                discards_before=r["discards_before"])
                           for r in recs],
    ))
    print(f"wrote {a.out} and {OUT / 'calib.json'} in {payload['wall_seconds']}s")


if __name__ == "__main__":
    main()
