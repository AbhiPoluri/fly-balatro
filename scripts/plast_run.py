"""The experiment: does dopamine-gated depression at KC -> MBON learn to play Balatro?

One run = one (wiring, valence scheme, tuning, eta, seed) cell of the protocol:

1. **Calibrate** the decision bias and softmax temperature on 200 hands from
   calibration seeds (:data:`plast_common.CALIB_SEED0`), with plasticity off.
   Frozen for the rest of the run. Cached per (tuning, wiring, valence, window).
2. **Pre-evaluate** on ``--eval-games`` ante-1 games from
   :data:`plast_common.EVAL_SEED0`, plasticity off. This is the "before" fly.
3. **Train** for ``--train-hands`` decisions on games from
   :data:`plast_common.TRAIN_SEED0` + 1000 x seed, delivering dopamine from the
   game's own chips. Nothing else about the fly changes.
4. **Post-evaluate** on the *same* eval games with plasticity frozen and no
   dopamine, so before/after differ only in the KC -> MBON weights.

Baselines (always-play, the v2 teacher's dig rule, always-discard) run the same
harness over the same eval games and are written once.

Run one cell::

    python -m scripts.plast_run --wiring real --eta 0.05 --seed 0

Run the whole protocol, resumable (skips any run whose json already exists)::

    python -m scripts.plast_run --all
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from flybalatro import plasticity as P  # noqa: E402
from scripts import plast_common as K  # noqa: E402

OUT = K.OUT_DIR

#: Pre-registered: the controls run at the middle of the eta sweep, chosen
#: before any run finished so "best eta" cannot be picked for them post hoc.
CONTROL_ETA = 0.05
ETAS = (0.02, 0.05, 0.1)
SEEDS = (0, 1, 2)


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunSpec:
    name: str
    wiring: str = "real"
    eta: float = CONTROL_ETA
    seed: int = 0
    tuning: str = "plast"
    valence: str = "nt"
    plasticity: bool = True
    omission: float = 0.0   # PPL1 pulse after an unrewarded PLAY, as a fraction of eta
    note: str = ""

    def setup_key(self, window_ms: Optional[float]) -> Tuple:
        return (self.tuning, self.wiring, self.valence, window_ms)


def protocol() -> List[RunSpec]:
    runs: List[RunSpec] = []
    for eta in ETAS:
        for s in SEEDS:
            runs.append(RunSpec(f"real_eta{eta:g}_s{s}", "real", eta, s,
                                note="real wiring, plasticity on"))
    for s in SEEDS:
        runs.append(RunSpec(f"shuffled_eta{CONTROL_ETA:g}_s{s}", "shuffled",
                            CONTROL_ETA, s,
                            note="degree-preserving wiring shuffle, same rule"))
    for s in SEEDS:
        runs.append(RunSpec(f"frozen_s{s}", "real", 0.0, s, plasticity=False,
                            note="frozen fly: same decision rule, no dopamine"))
    runs.append(RunSpec(f"specTuning_eta{CONTROL_ETA:g}_s0", "real", CONTROL_ETA, 0,
                        tuning="spec",
                        note="mb2 tuning exactly as chosen; reward arm lands on "
                             "MBONs that never spike"))
    runs.append(RunSpec(f"briefValence_eta{CONTROL_ETA:g}_s0", "real", CONTROL_ETA, 0,
                        valence="brief",
                        note="valence from the 9 compartments named in the brief "
                             "instead of the Aso neurotransmitter rule"))
    runs.append(RunSpec(f"omission_eta{CONTROL_ETA:g}_s0", "real", CONTROL_ETA, 0,
                        omission=1.0,
                        note="flagged variant: an unrewarded PLAY also gets a "
                             "full-strength PPL1 pulse (reward omission). Predicted "
                             "before running to be insufficient: the measured gain "
                             "is +9.2 Hz per reward pulse against -2.3 Hz per "
                             "punishment pulse, so equal counts still drift to play"))
    return runs


# --------------------------------------------------------------------------- #
class Runner:
    """Holds the expensive objects (graph, brains) across runs in one process."""

    def __init__(self, window_ms: Optional[float], n_calib: int,
                 explore: float, explore_floor: float = 0.0) -> None:
        self.graph = K.C.load(5)
        self.window_ms = window_ms
        self.n_calib = n_calib
        self.explore = explore
        self.explore_floor = float(explore_floor)
        self._setups: Dict[Tuple, K.Setup] = {}
        self._calib: Dict[Tuple, dict] = {}
        self._calib_bits: Optional[np.ndarray] = None
        self._calib_recs: Optional[List[dict]] = None

    # -- shared pieces -----------------------------------------------------
    def calib_hands(self) -> Tuple[np.ndarray, List[dict]]:
        if self._calib_bits is None:
            from scripts.plast_probe import collect_calibration_hands, hand_bits
            recs = collect_calibration_hands(self.n_calib, K.CALIB_SEED0, rng_seed=7)
            self._calib_recs = recs
            self._calib_bits = hand_bits(recs)
        return self._calib_bits, self._calib_recs  # type: ignore[return-value]

    def setup(self, spec: RunSpec) -> K.Setup:
        key = spec.setup_key(self.window_ms)
        if key not in self._setups:
            cfg = K.load_config(spec.tuning)
            self._setups[key] = K.build_setup(
                wiring=spec.wiring, valence_scheme=spec.valence,
                window_ms=self.window_ms, graph=self.graph, cfg=cfg)
        return self._setups[key]

    def decider(self, spec: RunSpec) -> Tuple[P.Decider, dict]:
        setup = self.setup(spec)
        dec = K.make_decider(setup, explore_floor=self.explore_floor)
        key = spec.setup_key(self.window_ms)
        if key not in self._calib:
            bits, _ = self.calib_hands()
            setup.plast.reset_weights()
            raw = [dec.raw_drive(setup.run_window(b)) for b in bits]
            self._calib[key] = dec.calibrate(raw, explore=self.explore)
            self._calib[key]["key"] = list(map(str, key))
            self._calib[key]["bias"] = dec.bias
            self._calib[key]["temperature"] = dec.temperature
        cal = self._calib[key]
        dec.bias = float(cal["bias"])
        dec.temperature = float(cal["temperature"])
        return dec, cal

    # -- one run -----------------------------------------------------------
    def run(self, spec: RunSpec, train_hands: int, eval_games: int,
            save_weights: Optional[Path] = None) -> dict:
        t0 = time.time()
        setup = self.setup(spec)
        dec, cal = self.decider(spec)
        setup.plast.reset_weights()
        env = K.make_env()

        payload: dict = dict(
            spec=dict(name=spec.name, wiring=spec.wiring, eta=spec.eta,
                      seed=spec.seed, tuning=spec.tuning, valence=spec.valence,
                      plasticity=spec.plasticity, omission=spec.omission,
                      note=spec.note),
            config=setup.cfg, window_ms=setup.window_ms,
            calibration=cal, decider=dec.to_dict(),
            valence_summary=setup.valence.summary(),
            plasticity_pools=setup.plast.describe(),
            train_hands=train_hands, eval_games=eval_games,
            seeds=dict(calib=K.CALIB_SEED0, train=K.TRAIN_SEED0 + 1000 * spec.seed,
                       eval=K.EVAL_SEED0),
        )

        # --- 2. before -----------------------------------------------------
        pre = self._evaluate(setup, dec, env, eval_games, spec, tag="pre")
        payload["pre"] = pre

        # --- 3. train ------------------------------------------------------
        setup.plast.reset_weights()
        train = self._train(setup, dec, env, spec, train_hands)
        payload["train"] = train

        # The learned synapses, written *before* the post-evaluation so a crash
        # in the eval does not throw away five minutes of training. The
        # calibration goes in with them: these weights are only meaningful to a
        # Decider with this bias and temperature.
        if save_weights is not None:
            meta = dict(
                run=spec.name, wiring=spec.wiring, tuning=spec.tuning,
                valence=spec.valence, seed=spec.seed,
                eta_reward=float(spec.eta),
                eta_punish=float(getattr(spec, "eta_punish", 0.0)) or float(spec.eta),
                train_hands=int(train["n_hands"]),
                bias=float(dec.bias), temperature=float(dec.temperature),
                explore_floor=float(dec.explore_floor),
                play_bias_p=float(dec.play_bias_p),
                window_ms=float(setup.window_ms),
                config=str(setup.cfg.get("label", spec.tuning)),
                dopamine_counts=train.get("dopamine_counts", {}),
            )
            path = setup.plast.save_weights(save_weights, meta=meta)
            payload["saved_weights"] = str(path)
            print(f"   wrote {path}", flush=True)

        # --- 4. after ------------------------------------------------------
        post = self._evaluate(setup, dec, env, eval_games, spec, tag="post")
        payload["post"] = post
        payload["weight_stats_after_training"] = setup.plast.weight_stats()
        payload["wall_seconds"] = round(time.time() - t0, 1)
        setup.plast.reset_weights()
        return payload

    def _train(self, setup: K.Setup, dec: P.Decider, env, spec: RunSpec,
               n_hands: int) -> dict:
        rng = np.random.default_rng(10_000 + spec.seed)
        fly = K.FlyPolicy(setup, dec, rng)
        eta = float(spec.eta) if spec.plasticity else 0.0

        def on_outcome(rec: dict, kcc) -> dict:
            if kcc is None or eta <= 0.0:
                return dict(dopamine="none", mean_abs_dw=0.0)
            if rec["reward"]:
                info = setup.plast.deliver("reward", kcc, eta)
                return dict(dopamine="reward", mean_abs_dw=round(info["mean_abs_dw"], 6),
                            n_changed=info["n_changed"])
            if rec["punish"]:
                info = setup.plast.deliver("punish", kcc, eta)
                return dict(dopamine="punish", mean_abs_dw=round(info["mean_abs_dw"], 6),
                            n_changed=info["n_changed"])
            if spec.omission > 0.0 and rec["action"] == K.PLAY:
                info = setup.plast.deliver("punish", kcc, eta * spec.omission)
                return dict(dopamine="omission",
                            mean_abs_dw=round(info["mean_abs_dw"], 6),
                            n_changed=info["n_changed"])
            return dict(dopamine="none", mean_abs_dw=0.0)

        records: List[dict] = []
        games: List[K.GameResult] = []
        seed = K.TRAIN_SEED0 + 1000 * spec.seed
        while len(records) < n_hands:
            g = K.play_game(env, seed, fly, on_outcome=on_outcome, fly=fly,
                            max_hands=n_hands - len(records))
            games.append(g)
            records.extend(g.hands)
            seed += 1
        rewards = [r["reward"] for r in records]
        return dict(
            n_hands=len(records), n_games=len(games), last_seed=seed - 1,
            games=summarise_train_games(games),
            summary=K.summarise_games(games),
            reward_rate_rolling=[round(x, 4) for x in K.rolling(rewards, 50)],
            p_play_rolling=[round(x, 4)
                            for x in K.rolling([r["action"] == K.PLAY for r in records], 50)],
            play_drive_rolling=[round(x, 4)
                                for x in K.rolling([r.get("play_drive", 0.0)
                                                    for r in records], 50)],
            dopamine_counts=_counts(records, "dopamine"),
            first_third=_phase(records[: len(records) // 3]),
            last_third=_phase(records[-(len(records) // 3):]),
            records=records,
            weight_stats=setup.plast.weight_stats(),
        )

    def _evaluate(self, setup: K.Setup, dec: P.Decider, env, n_games: int,
                  spec: RunSpec, tag: str) -> dict:
        """Frozen plasticity, no dopamine, the same eval seeds for every row."""
        rng = np.random.default_rng(50_000 + spec.seed)
        fly = K.FlyPolicy(setup, dec, rng)
        games = [K.play_game(env, K.EVAL_SEED0 + i, fly) for i in range(n_games)]
        records = [h for g in games for h in g.hands]
        return dict(
            tag=tag, summary=K.summarise_games(games),
            p_play_by_hand_type=K.p_play_table(records, "best_type_name"),
            p_play_by_bucket=K.p_play_table(records, "bucket_name"),
            play_drive_mean=round(float(np.mean([r["play_drive"] for r in records])), 4)
            if records else None,
            weight_stats=setup.plast.weight_stats(),
            records=records,
        )


def _counts(records: Sequence[dict], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in records:
        out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
    return out


def _phase(records: Sequence[dict]) -> dict:
    if not records:
        return {}
    return dict(
        n=len(records),
        p_play=float(np.mean([r["action"] == K.PLAY for r in records])),
        reward_rate=float(np.mean([r["reward"] for r in records])),
        punish_rate=float(np.mean([r["punish"] for r in records])),
        play_drive_mean=round(float(np.mean([r.get("play_drive", 0.0)
                                             for r in records])), 4),
        p_play_by_bucket=K.p_play_table(records, "bucket_name"),
        p_play_by_hand_type=K.p_play_table(records, "best_type_name"),
    )


def summarise_train_games(games: Sequence[K.GameResult]) -> List[dict]:
    return [dict(seed=g.seed, cleared=g.cleared, chips=g.chips, rounds=g.rounds,
                 hands=len(g.hands)) for g in games]


# --------------------------------------------------------------------------- #
def run_baselines(runner: Runner, eval_games: int) -> dict:
    """Always-play, the teacher's dig rule, always-discard, on the eval games."""
    env = K.make_env()
    out: dict = dict(eval_games=eval_games, eval_seed0=K.EVAL_SEED0)
    policies = dict(
        always_play=K.always_play_policy,
        teacher_dig=K.make_teacher_dig_policy(1.0),
        always_discard=K.make_always_discard_policy(),
        # every odour-only policy: the bucket is the only ordered axis the odour
        # carries, so these four bracket the ceiling for a learned valence map.
        bucket_ge1=K.make_bucket_policy(["ge1.0"]),
        bucket_ge_lt1=K.make_bucket_policy(["ge1.0", "lt1.0"]),
        bucket_ge_lt1_lt05=K.make_bucket_policy(["ge1.0", "lt1.0", "lt0.5"]),
    )
    for name, pol in policies.items():
        games = [K.play_game(env, K.EVAL_SEED0 + i, pol) for i in range(eval_games)]
        recs = [h for g in games for h in g.hands]
        out[name] = dict(summary=K.summarise_games(games),
                         p_play_by_hand_type=K.p_play_table(recs, "best_type_name"),
                         p_play_by_bucket=K.p_play_table(recs, "bucket_name"))
        print(f"  {name:15s} clear {out[name]['summary']['clear_rate']:.3f} "
              f"chips {out[name]['summary']['chips_mean']:.0f} "
              f"hands {out[name]['summary']['n_hands']}")
    return out


def weights_path(arg: Optional[str], spec, specs: Sequence) -> Optional[Path]:
    """``--save-weights`` resolved for one run.

    One selected run writes exactly the path given, so the file name in the
    report is the file name on disk. Several runs would otherwise all write the
    same file, so the run name is inserted before the suffix.
    """
    if not arg:
        return None
    p = Path(arg)
    if len(specs) <= 1:
        return p
    return p.with_name(f"{p.stem}_{spec.name}{p.suffix or '.npz'}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run the whole protocol")
    ap.add_argument("--only", default=None, help="comma-separated run names")
    ap.add_argument("--wiring", default="real")
    ap.add_argument("--eta", type=float, default=CONTROL_ETA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tuning", default="plast")
    ap.add_argument("--valence", default="nt")
    ap.add_argument("--no-plasticity", action="store_true")
    ap.add_argument("--omission", type=float, default=0.0)
    ap.add_argument("--train-hands", type=int, default=600)
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--explore", type=float, default=0.20)
    ap.add_argument("--explore-floor", type=float, default=0.0)
    ap.add_argument("--window-ms", type=float, default=None)
    ap.add_argument("--baselines", action="store_true", help="also write baselines.json")
    ap.add_argument("--force", action="store_true", help="rerun even if output exists")
    ap.add_argument("--out-dir", default=None,
                    help="write run_<name>.json here instead of outputs/plast "
                         "(so a rerun made only to save weights cannot overwrite "
                         "the run on the record)")
    ap.add_argument("--save-weights", default=None,
                    help="write the post-training KC -> MBON weights to this .npz "
                         "(with the run's calibration in its metadata). With more "
                         "than one run selected, the run name is appended.")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir) if a.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(a.window_ms, a.n_calib, a.explore, a.explore_floor)

    if a.all or a.baselines:
        bl_path = out_dir / "baselines.json"
        if a.force or not bl_path.exists():
            print("baselines:")
            K.write_json(bl_path, run_baselines(runner, a.eval_games))

    if a.all:
        specs = protocol()
    elif a.only:
        wanted = set(a.only.split(","))
        specs = [s for s in protocol() if s.name in wanted]
        if not specs:
            raise SystemExit(f"no protocol run matches {sorted(wanted)}")
    else:
        name = (f"{a.wiring}_eta{a.eta:g}_s{a.seed}" if not a.no_plasticity
                else f"frozen_s{a.seed}")
        specs = [RunSpec(name, a.wiring, a.eta, a.seed, a.tuning, a.valence,
                         not a.no_plasticity, a.omission, note="ad hoc")]

    for spec in specs:
        path = out_dir / f"run_{spec.name}.json"
        if path.exists() and not a.force:
            print(f"skip {spec.name} (exists)")
            continue
        print(f"== {spec.name} ({spec.note}) ==", flush=True)
        payload = runner.run(spec, a.train_hands, a.eval_games,
                             save_weights=weights_path(a.save_weights, spec, specs))
        K.write_json(path, payload)
        pre, post = payload["pre"]["summary"], payload["post"]["summary"]
        print(f"   clear {pre['clear_rate']:.3f} -> {post['clear_rate']:.3f} | "
              f"chips {pre['chips_mean']:.0f} -> {post['chips_mean']:.0f} | "
              f"P(play|legal) {pre['p_play_when_legal']:.3f} -> "
              f"{post['p_play_when_legal']:.3f} | "
              f"train reward {payload['train']['summary']['reward_rate']:.3f} | "
              f"{payload['wall_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
