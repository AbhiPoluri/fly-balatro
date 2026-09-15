"""Task 1: re-evaluate the v2 fly greedily as well as by sampling.

`scripts/plast_run.py` evaluates with the **training** decider
(``_evaluate`` builds ``K.FlyPolicy(setup, dec, rng)`` and ``FlyPolicy.__call__``
draws ``rng.random() < p``), and the v2 runs carry ``explore_floor = 0.1``. So
every "greedy (no dopamine, frozen weights) at evaluation" number in
``outputs/plast2/REPORT.md`` and in the RESULTS.md v2 section is a **floored
softmax sample**, not the greedy policy. With the learned low bucket sitting at
P(play) = 0.545, 4.5 points above the threshold, the two can differ a lot.

This script evaluates the v2 learned weights and the v2 frozen control **both**
ways on the standard evaluation seeds (100000-100059, the ones the v2 table
used), and writes ``outputs/plast3/eval_mode.json``.

Greedy is ``PLAY iff p >= 0.5``. Because ``p = e/2 + (1 - e) * sigmoid(d / T)``
is monotone in ``d`` and equals 0.5 exactly at ``d = 0`` for any floor ``e < 1``,
that is identical to ``play_drive >= 0`` and does not depend on the exploration
floor at all.

Run::

    python -m scripts.plast3_evalmode
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts import plast3_common as C
from scripts import plast_common as K
from scripts.plast_probe import collect_calibration_hands, hand_bits

OUT_JSON = C.OUT_DIR / "eval_mode.json"
PLAST2 = K.ROOT / "outputs" / "plast2"
RETRAIN = C.OUT_DIR / "v2retrain"
#: The v2 protocol's own evaluation games.
EVAL_SEED0 = K.EVAL_SEED0
EVAL_GAMES = 60


def find_weights() -> List[dict]:
    """Every saved v2 weight file, published first, retrained second."""
    out: List[dict] = []
    for p in sorted(PLAST2.glob("weights_*.npz")):
        out.append(dict(path=p, source="outputs/plast2 (published)"))
    for p in sorted(RETRAIN.glob("weights_*.npz")):
        out.append(dict(path=p, source="outputs/plast3/v2retrain (re-trained)"))
    return out


def weights_meta(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return json.loads(str(d["meta"]))


def compare_npz(a: Path, b: Path) -> dict:
    """Bit-for-bit comparison of two weight files."""
    da, db = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    same_edges = bool(np.array_equal(da["edge"], db["edge"]))
    wa, wb = da["weight"], db["weight"]
    return dict(
        a=C.rel_path(a), b=C.rel_path(b),
        same_edge_index=same_edges,
        identical=bool(same_edges and np.array_equal(wa, wb)),
        max_abs_diff=float(np.abs(wa - wb).max()) if same_edges else None,
        n_differing=int((wa != wb).sum()) if same_edges else None,
    )


def evaluate(setup: K.Setup, dec, mode: str, seeds: Sequence[int],
             rng_seed: int, env) -> dict:
    rng = np.random.default_rng(rng_seed) if mode == C.SAMPLED else None
    pol = C.EvalFlyPolicy(setup, dec, mode, rng=rng, cache=True)
    games = C.evaluate_policy(pol, seeds, env=env)
    recs = [h for g in games for h in g.hands]
    s = K.summarise_games(games)
    bucket = K.p_play_table(recs, "bucket_name")
    grad = None
    if "ge1.0" in bucket and "lt0.25" in bucket:
        grad = round(bucket["ge1.0"]["p_play"] - bucket["lt0.25"]["p_play"], 4)
    return dict(
        mode=mode, rng_seed=int(rng_seed) if mode == C.SAMPLED else None,
        summary=s, p_play_by_bucket=bucket, bucket_gradient=grad,
        p_play_by_hand_type=K.p_play_table(recs, "best_type_name"),
        cache=pol.cache_stats(),
        cleared=[bool(g.cleared) for g in games],
        seeds=[int(x) for x in seeds],
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-games", type=int, default=EVAL_GAMES)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--out", default=str(OUT_JSON))
    a = ap.parse_args(argv)

    t0 = time.time()
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = K.C.load(5)
    cfg = json.loads((PLAST2 / "tuned_config.json").read_text())
    setup = C.build_setup3(C.ENC_CURRENT, cfg, graph)
    env = K.make_env()
    seeds = [EVAL_SEED0 + i for i in range(a.eval_games)]

    # The decider, calibrated exactly the way the runs calibrate it.
    dec = K.make_decider(setup, explore_floor=C.EXPLORE_FLOOR3)
    recs = collect_calibration_hands(a.n_calib, K.CALIB_SEED0, rng_seed=7)
    bits = hand_bits(recs)
    setup.plast.reset_weights()
    raw = [dec.raw_drive(setup.run_window(b)) for b in bits]
    cal = dec.calibrate(raw, explore=0.20)
    print(f"calibration: bias {dec.bias:.6f} T {dec.temperature:.6f}", flush=True)

    payload: dict = dict(
        note=("plast_run/plast2_run evaluate with the training decider, which "
              "samples rng.random() < p with explore_floor 0.1. Greedy here is "
              "PLAY iff p >= 0.5, equivalently play_drive >= 0."),
        eval_seeds=dict(seed0=EVAL_SEED0, n=len(seeds)),
        calibration=cal, decider=dec.to_dict(), config_label=cfg.get("label"),
        conditions={}, weight_files=[], weight_comparisons=[],
    )

    # --- the frozen control ------------------------------------------------
    setup.plast.reset_weights()
    frozen: Dict[str, dict] = {}
    frozen[C.GREEDY] = evaluate(setup, dec, C.GREEDY, seeds, 0, env)
    for s in (0, 1, 2):
        frozen[f"{C.SAMPLED}_s{s}"] = evaluate(setup, dec, C.SAMPLED, seeds,
                                               C.EVAL3_RNG_SEED + s, env)
    payload["conditions"]["frozen"] = dict(
        note="no dopamine ever; the v2 'before' fly", runs=frozen,
        sampled_mean_clear=round(float(np.mean(
            [frozen[f"{C.SAMPLED}_s{s}"]["summary"]["clear_rate"]
             for s in (0, 1, 2)])), 4),
        greedy_clear=frozen[C.GREEDY]["summary"]["clear_rate"],
    )
    print(f"frozen: greedy {frozen[C.GREEDY]['summary']['clear_rate']:.3f} "
          f"sampled {payload['conditions']['frozen']['sampled_mean_clear']:.3f}",
          flush=True)

    # --- every learned weight file ----------------------------------------
    files = find_weights()
    by_run: Dict[str, List[dict]] = {}
    for f in files:
        m = weights_meta(f["path"])
        f["run"] = m["run"]
        f["meta"] = m
        payload["weight_files"].append(
            dict(path=str(f["path"].relative_to(K.ROOT)), source=f["source"],
                 run=m["run"], eta_reward=m["eta_reward"],
                 eta_punish=m["eta_punish"], train_hands=m["train_hands"]))
        by_run.setdefault(m["run"], []).append(f)

    for run, fs in sorted(by_run.items()):
        if len(fs) > 1:
            cmp = compare_npz(fs[0]["path"], fs[1]["path"])
            cmp["run"] = run
            payload["weight_comparisons"].append(cmp)
            print(f"compare {run}: identical={cmp['identical']} "
                  f"n_differing={cmp['n_differing']}", flush=True)
        f = fs[0]
        m = f["meta"]
        if m.get("eta_reward", 0.0) <= 0.0:
            continue  # a frozen run's weights are the unlearned ones
        info = setup.plast.load_weights(f["path"])
        assert abs(float(m["bias"]) - dec.bias) < 1e-9, (
            f"{run}: stored bias {m['bias']} != recomputed {dec.bias}")
        assert abs(float(m["temperature"]) - dec.temperature) < 1e-9
        runs: Dict[str, dict] = {C.GREEDY: evaluate(setup, dec, C.GREEDY, seeds,
                                                    0, env)}
        runs[C.SAMPLED] = evaluate(setup, dec, C.SAMPLED, seeds,
                                   C.EVAL3_RNG_SEED + int(m["seed"]), env)
        payload["conditions"][run] = dict(
            weights=str(f["path"].relative_to(K.ROOT)), source=f["source"],
            meta=m, weight_load=info, runs=runs)
        print(f"{run}: greedy {runs[C.GREEDY]['summary']['clear_rate']:.3f} "
              f"P(play|legal) {runs[C.GREEDY]['summary']['p_play_when_legal']:.3f} "
              f"| sampled {runs[C.SAMPLED]['summary']['clear_rate']:.3f} "
              f"P(play|legal) "
              f"{runs[C.SAMPLED]['summary']['p_play_when_legal']:.3f}", flush=True)
        setup.plast.reset_weights()

    # --- baselines on the same games --------------------------------------
    base: Dict[str, dict] = {}
    for name, pol in (("always_play", K.always_play_policy),
                      ("always_discard", K.make_always_discard_policy()),
                      ("teacher_dig", K.make_teacher_dig_policy(1.0)),
                      ("bucket_ge_lt1", K.make_bucket_policy(["ge1.0", "lt1.0"]))):
        games = C.evaluate_policy(pol, seeds, env=env)
        base[name] = dict(summary=K.summarise_games(games),
                          cleared=[bool(g.cleared) for g in games])
        print(f"  {name:16s} clear {base[name]['summary']['clear_rate']:.3f}",
              flush=True)
    payload["baselines"] = base

    # --- the paired, clustered comparison the v2 z = 1.75 should have been --
    payload["paired"] = paired_tables(payload)
    payload["wall_seconds"] = round(time.time() - t0, 1)
    C.write_json(Path(a.out), payload)
    print(f"wrote {a.out} in {payload['wall_seconds']:.0f}s")


def paired_tables(payload: dict) -> dict:
    """Learned vs frozen on the same 60 games, clustered by seed and by run."""
    out: dict = {}
    cond = payload["conditions"]
    for mode in C.EVAL_MODES:
        for eta in (0.02, 0.05):
            names = sorted(k for k in cond if k.startswith(f"real_eta{eta:g}_s"))
            if not names:
                continue
            runs = [np.asarray(cond[n]["runs"][mode]["cleared"], np.float64)
                    for n in names]
            # Pair each learned run against the frozen control that used the
            # SAME evaluation RNG, so the sampled comparison is paired on the
            # draw as well as on the game seed. Greedy consults no RNG, so its
            # control is one deterministic vector.
            if mode == C.GREEDY:
                controls = [np.asarray(cond["frozen"]["runs"][C.GREEDY]["cleared"],
                                       np.float64)] * len(names)
            else:
                controls = [np.asarray(
                    cond["frozen"]["runs"][
                        f"{C.SAMPLED}_s{int(cond[n]['meta']['seed'])}"]["cleared"],
                    np.float64) for n in names]
            diffs = [a - b for a, b in zip(runs, controls)]
            control = np.zeros_like(diffs[0])
            r = C.cluster_bootstrap(diffs, control)
            d = r.to_dict()
            d["runs"] = names
            d["learned_clear"] = round(float(np.mean([x.mean() for x in runs])), 4)
            d["control_clear"] = round(
                float(np.mean([x.mean() for x in controls])), 4)
            out[f"{mode}_eta{eta:g}_vs_frozen"] = d
            ad = np.asarray(payload["baselines"]["always_discard"]["cleared"],
                            np.float64)
            rb = C.cluster_bootstrap(runs, ad)
            db = rb.to_dict()
            db["learned_clear"] = d["learned_clear"]
            db["baseline_clear"] = round(float(ad.mean()), 4)
            out[f"{mode}_eta{eta:g}_vs_always_discard"] = db
    return out


if __name__ == "__main__":
    main()
