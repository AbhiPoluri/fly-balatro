"""The preregistered 2x2: punishment rule x bucket encoding.

See ``outputs/plast3/PREREGISTRATION.md``, written before any run here started.

| factor | levels |
|---|---|
| A punishment | ``current`` (terminal losing play only) / ``omission`` (every unrewarded PLAY) |
| B encoding | ``current`` (mb2, one glomerulus per relay bit) / ``separated`` (four disjoint ORN-balanced bucket ensembles) |

3 training seeds per cell, 1,200 decisions each, ``eta_reward = 0.05`` with the
per-encoding calibrated ``eta_punish``, plus one frozen control per encoding.
Every condition is evaluated on the **same 400 fresh game seeds**
(400000-400399), greedy and sampled.

Nothing about the reward changes and no teacher is consulted: both punishment
rules are the negative branch of the same inequality
``chips_gained >= needed / plays_left`` that already defines reward, computed by
``plast_common.resolve_outcome`` from the game's own chips.

Run (resumable; skips any run whose json exists)::

    python -m scripts.plast3_run --all
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from flybalatro import plasticity as P
from scripts import plast2_run as R2
from scripts import plast3_common as C
from scripts import plast_common as K
from scripts import plast_run as R
from scripts.plast_probe import collect_calibration_hands, hand_bits

OUT = C.OUT_DIR
CONFIGS = {
    C.ENC_CURRENT: K.ROOT / "outputs" / "plast2" / "tuned_config.json",
    C.ENC_SEPARATED: OUT / "tuned_config_sep.json",
}
ETA_FILES = {
    C.ENC_CURRENT: K.ROOT / "outputs" / "plast2" / "eta_calib.json",
    C.ENC_SEPARATED: OUT / "eta_calib_sep.json",
}


@dataclass(frozen=True)
class Cell:
    name: str
    punishment: str
    encoding: str
    seed: int
    eta_reward: float = C.ETA_REWARD3
    eta_punish: float = 0.0
    plasticity: bool = True


def eta_punish_for(encoding: str, eta_reward: float) -> Tuple[float, dict]:
    d = json.loads(ETA_FILES[encoding].read_text())
    row = d["chosen"][f"{eta_reward:g}"]
    return float(row["eta_punish"]), row


def protocol() -> List[Cell]:
    cells: List[Cell] = []
    for enc in C.ENCODINGS:
        cells.append(Cell(f"frozen_{enc}", C.PUNISH_CURRENT, enc, 0,
                          eta_reward=0.0, plasticity=False))
    for punishment in C.PUNISHMENTS:
        for enc in C.ENCODINGS:
            ep, _ = eta_punish_for(enc, C.ETA_REWARD3)
            for s in C.TRAIN3_SEEDS:
                cells.append(Cell(f"{punishment}_{enc}_s{s}", punishment, enc, s,
                                  eta_punish=ep))
    return cells


class Runner3:
    """One process, both flies, every cell. Expensive objects built once."""

    def __init__(self, n_calib: int = 200, explore: float = 0.20,
                 eval_games: int = C.EVAL3_GAMES) -> None:
        self.graph = K.C.load(5)
        self.n_calib = n_calib
        self.explore = explore
        self.eval_games = int(eval_games)
        self.env = K.make_env()
        self._setups: Dict[str, K.Setup] = {}
        self._calib: Dict[str, dict] = {}
        self._calib_bits: Optional[np.ndarray] = None
        self.seeds = C.eval_seeds(self.eval_games)

    def calib_bits(self) -> np.ndarray:
        if self._calib_bits is None:
            recs = collect_calibration_hands(self.n_calib, K.CALIB_SEED0, rng_seed=7)
            self._calib_bits = hand_bits(recs)
        return self._calib_bits

    def setup(self, encoding: str) -> K.Setup:
        if encoding not in self._setups:
            cfg = json.loads(CONFIGS[encoding].read_text())
            self._setups[encoding] = C.build_setup3(encoding, cfg, self.graph)
        return self._setups[encoding]

    def decider(self, encoding: str) -> Tuple[P.Decider, dict]:
        setup = self.setup(encoding)
        dec = K.make_decider(setup, explore_floor=C.EXPLORE_FLOOR3)
        if encoding not in self._calib:
            setup.plast.reset_weights()
            raw = [dec.raw_drive(setup.run_window(b)) for b in self.calib_bits()]
            cal = dec.calibrate(raw, explore=self.explore)
            cal["bias"], cal["temperature"] = dec.bias, dec.temperature
            self._calib[encoding] = cal
        cal = self._calib[encoding]
        dec.bias = float(cal["bias"])
        dec.temperature = float(cal["temperature"])
        return dec, cal

    # -- one cell ----------------------------------------------------------
    def run(self, cell: Cell, train_hands: int,
            save_weights: Optional[Path] = None) -> dict:
        t0 = time.time()
        setup = self.setup(cell.encoding)
        dec, cal = self.decider(cell.encoding)
        setup.plast.reset_weights()

        payload: dict = dict(
            cell=dict(name=cell.name, punishment=cell.punishment,
                      encoding=cell.encoding, seed=cell.seed,
                      eta_reward=cell.eta_reward, eta_punish=cell.eta_punish,
                      plasticity=cell.plasticity),
            config=str(CONFIGS[cell.encoding].relative_to(K.ROOT)),
            config_label=setup.cfg.get("label"),
            window_ms=setup.window_ms, calibration=cal, decider=dec.to_dict(),
            valence_summary=setup.valence.summary(),
            plasticity_pools=setup.plast.describe(),
            odour_map=setup.gm.describe(),
            train_hands=int(train_hands) if cell.plasticity else 0,
            eval_seeds=dict(seed0=C.EVAL3_SEED0, n=self.eval_games),
            seeds=dict(calib=K.CALIB_SEED0,
                       train=K.TRAIN_SEED0 + 1000 * cell.seed),
        )

        if cell.plasticity:
            payload["train"] = self._train(setup, dec, cell, train_hands)
            if save_weights is not None:
                meta = dict(run=cell.name, encoding=cell.encoding,
                            punishment=cell.punishment, seed=cell.seed,
                            eta_reward=cell.eta_reward, eta_punish=cell.eta_punish,
                            train_hands=int(payload["train"]["n_hands"]),
                            bias=float(dec.bias), temperature=float(dec.temperature),
                            explore_floor=float(dec.explore_floor),
                            window_ms=float(setup.window_ms),
                            dopamine_counts=payload["train"]["dopamine_counts"])
                payload["saved_weights"] = str(
                    setup.plast.save_weights(save_weights, meta=meta))

        payload["eval"] = {m: self._evaluate(setup, dec, cell, m)
                           for m in C.EVAL_MODES}
        payload["weight_stats"] = setup.plast.weight_stats()
        payload["wall_seconds"] = round(time.time() - t0, 1)
        setup.plast.reset_weights()
        return payload

    def _train(self, setup: K.Setup, dec: P.Decider, cell: Cell,
               n_hands: int) -> dict:
        rng = np.random.default_rng(10_000 + cell.seed)
        fly = K.FlyPolicy(setup, dec, rng)
        eta_r, eta_p = float(cell.eta_reward), float(cell.eta_punish)

        def on_outcome(rec: dict, kcc) -> dict:
            kind = C.dopamine_for(rec, cell.punishment)
            if kcc is None or kind is None:
                return dict(dopamine="none", mean_abs_dw=0.0)
            eta = eta_r if kind == "reward" else eta_p
            if eta <= 0.0:
                return dict(dopamine="none", mean_abs_dw=0.0)
            info = setup.plast.deliver(kind, kcc, eta)
            return dict(dopamine=kind, eta=eta,
                        mean_abs_dw=round(info["mean_abs_dw"], 6),
                        n_changed=info["n_changed"])

        records: List[dict] = []
        games: List[K.GameResult] = []
        seed = K.TRAIN_SEED0 + 1000 * cell.seed
        while len(records) < n_hands:
            g = K.play_game(self.env, seed, fly, on_outcome=on_outcome, fly=fly,
                            max_hands=n_hands - len(records))
            games.append(g)
            records.extend(g.hands)
            seed += 1
        third = len(records) // 3
        return dict(
            n_hands=len(records), n_games=len(games), last_seed=seed - 1,
            eta_reward=eta_r, eta_punish=eta_p,
            summary=K.summarise_games(games),
            reward_rate_rolling=[round(x, 4) for x in
                                 K.rolling([r["reward"] for r in records], 50)],
            p_play_rolling=[round(x, 4) for x in
                            K.rolling([r["action"] == K.PLAY for r in records], 50)],
            dopamine_counts=R._counts(records, "dopamine"),
            dopamine_by_bucket=_dopamine_by_bucket(records),
            first_third=R._phase(records[:third]),
            last_third=R._phase(records[-third:]),
            weight_stats=setup.plast.weight_stats(),
            records=records,
        )

    def _evaluate(self, setup: K.Setup, dec: P.Decider, cell: Cell,
                  mode: str) -> dict:
        rng = (np.random.default_rng(C.EVAL3_RNG_SEED) if mode == C.SAMPLED
               else None)
        pol = C.EvalFlyPolicy(setup, dec, mode, rng=rng, cache=True,
                              encoding=cell.encoding)
        games = C.evaluate_policy(pol, self.seeds, env=self.env)
        recs = [h for g in games for h in g.hands]
        bucket = K.p_play_table(recs, "bucket_name")
        grad = (round(bucket["ge1.0"]["p_play"] - bucket["lt0.25"]["p_play"], 4)
                if "ge1.0" in bucket and "lt0.25" in bucket else None)
        return dict(
            mode=mode, summary=K.summarise_games(games),
            p_play_by_bucket=bucket, bucket_gradient=grad,
            p_play_by_hand_type=K.p_play_table(recs, "best_type_name"),
            joint_p_play=R2.joint_p_play_table(recs),
            cache=pol.cache_stats(),
            cleared=[bool(g.cleared) for g in games],
            chips=[int(g.chips) for g in games],
        )


def _dopamine_by_bucket(records: Sequence[dict]) -> Dict[str, dict]:
    """The v2 report's table 6.1: how many plays in each bucket earned what."""
    out: Dict[str, dict] = {}
    for r in records:
        if r["action"] != K.PLAY:
            continue
        d = out.setdefault(str(r["bucket_name"]),
                           dict(plays=0, reward=0, punish=0, none=0))
        d["plays"] += 1
        k = str(r.get("dopamine", "none"))
        d[k if k in ("reward", "punish") else "none"] += 1
    return out


def run_baselines(runner: Runner3) -> dict:
    """Reference policies on the same 400 evaluation games."""
    out: dict = dict(eval_seed0=C.EVAL3_SEED0, eval_games=runner.eval_games)
    policies = dict(
        always_play=K.always_play_policy,
        always_discard=K.make_always_discard_policy(),
        teacher_dig=K.make_teacher_dig_policy(1.0),
        bucket_ge1=K.make_bucket_policy(["ge1.0"]),
        bucket_ge_lt1=K.make_bucket_policy(["ge1.0", "lt1.0"]),
        bucket_ge_lt1_lt05=K.make_bucket_policy(["ge1.0", "lt1.0", "lt0.5"]),
    )
    for name, pol in policies.items():
        games = C.evaluate_policy(pol, runner.seeds, env=runner.env)
        recs = [h for g in games for h in g.hands]
        out[name] = dict(summary=K.summarise_games(games),
                         p_play_by_bucket=K.p_play_table(recs, "bucket_name"),
                         joint_p_play=R2.joint_p_play_table(recs),
                         cleared=[bool(g.cleared) for g in games])
        print(f"  {name:20s} clear {out[name]['summary']['clear_rate']:.3f} "
              f"chips {out[name]['summary']['chips_mean']:.0f}", flush=True)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated cell names")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--train-hands", type=int, default=C.TRAIN3_HANDS)
    ap.add_argument("--eval-games", type=int, default=C.EVAL3_GAMES)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner3(a.n_calib, eval_games=a.eval_games)

    if a.baselines or a.all:
        p = out_dir / "baselines3.json"
        if a.force or not p.exists():
            print("baselines (400 games):", flush=True)
            C.write_json(p, run_baselines(runner))
        else:
            print("skip baselines3.json (exists)")

    if not (a.all or a.only):
        return
    cells = protocol()
    if a.only:
        wanted = set(a.only.split(","))
        cells = [c for c in cells if c.name in wanted]
        if not cells:
            raise SystemExit(f"no cell matches {sorted(wanted)}")

    for cell in cells:
        path = out_dir / f"run_{cell.name}.json"
        if path.exists() and not a.force:
            print(f"skip {cell.name} (exists)")
            continue
        print(f"== {cell.name} (punish={cell.punishment} enc={cell.encoding} "
              f"eta_r={cell.eta_reward:g} eta_p={cell.eta_punish:g}) ==", flush=True)
        wpath = (out_dir / f"weights_{cell.name}.npz") if cell.plasticity else None
        payload = runner.run(cell, a.train_hands, save_weights=wpath)
        C.write_json(path, payload)
        g, s = payload["eval"][C.GREEDY], payload["eval"][C.SAMPLED]
        print(f"   greedy clear {g['summary']['clear_rate']:.3f} "
              f"P(play|legal) {g['summary']['p_play_when_legal']:.3f} "
              f"grad {g['bucket_gradient']} | sampled clear "
              f"{s['summary']['clear_rate']:.3f} "
              f"P(play|legal) {s['summary']['p_play_when_legal']:.3f} "
              f"grad {s['bucket_gradient']} | {payload['wall_seconds']:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
