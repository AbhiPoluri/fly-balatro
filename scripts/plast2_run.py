"""Plasticity v2: the same protocol as ``scripts/plast_run.py`` at the new operating point.

Three things differ from ``outputs/plast``, and nothing else does:

1. the fly is ``outputs/plast2/tuned_config.json`` -- the ``outputs/plast``
   operating point plus the per-Kenyon-cell homeostatic thresholds from
   ``scripts/kc_homeo.py``;
2. the punishment pulse carries its own ``eta`` (``scripts/plast2_eta.py``), so one
   punishment pulse is worth one reward pulse in ``|delta drive|``. The *frequency*
   of the two is whatever ante 1 gives;
3. 1,200 training hands instead of 600, because a code that assigns credit
   per-odour learns each odour separately and therefore more slowly.

Same decision rule, same bias/temperature calibration on 200 hands from seeds
300000+ frozen before any dopamine, same 10% exploration floor during training,
greedy (no dopamine, frozen weights) at evaluation on seeds 100000-100059.

Run one cell::

    python -m scripts.plast2_run --only real_eta0.05_s0

The whole protocol (resumable; skips any run whose json exists)::

    python -m scripts.plast2_run --all
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
from scripts import kc_homeo as H  # noqa: E402
from scripts import plast_common as K  # noqa: E402
from scripts import plast_run as R  # noqa: E402

OUT = H.OUT_DIR
CONFIG = H.CONFIG_JSON
ETA_JSON = H.OUT_DIR / "eta_calib.json"

ETAS = (0.02, 0.05)
SEEDS = (0, 1, 2)
TRAIN_HANDS = 1200
EVAL_GAMES = 60
EXPLORE_FLOOR = 0.1


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Spec2(R.RunSpec):
    """``RunSpec`` plus the separately calibrated punishment learning rate."""

    eta_punish: float = 0.0

    def setup_key(self, window_ms: Optional[float]) -> Tuple:
        return ("plast2", self.wiring, self.valence, window_ms)


def eta_punish_for(eta_reward: float) -> Tuple[float, dict]:
    """The calibrated punishment eta for this reward eta, from ``eta_calib.json``."""
    d = json.loads(ETA_JSON.read_text())
    key = f"{eta_reward:g}"
    if key not in d["chosen"]:
        raise SystemExit(f"eta_calib.json has no entry for eta_reward={key}")
    row = d["chosen"][key]
    return float(row["eta_punish"]), row


def protocol() -> List[Spec2]:
    runs: List[Spec2] = []
    for eta in ETAS:
        ep, _ = eta_punish_for(eta)
        for s in SEEDS:
            runs.append(Spec2(f"real_eta{eta:g}_s{s}", "real", eta, s,
                              tuning="plast2", eta_punish=ep,
                              note=("real wiring, KC homeostasis, calibrated "
                                    "punishment pulse")))
    for s in SEEDS:
        runs.append(Spec2(f"frozen_s{s}", "real", 0.0, s, tuning="plast2",
                          plasticity=False, eta_punish=0.0,
                          note="frozen fly: same decision rule, no dopamine"))
    return runs


# --------------------------------------------------------------------------- #
def joint_p_play_table(records: Sequence[dict], min_n: int = 5) -> dict:
    """P(play | hand type x score-vs-needed bucket), decisions where discard was legal.

    The marginals in ``outputs/plast`` were both flat, which is exactly what a
    single learned number for every odour looks like. The joint is what
    distinguishes "learned a valence per odour" from "learned a valence".
    """
    cells: Dict[str, dict] = {}
    for r in records:
        if not r.get("discard_ok"):
            continue
        key = f"{r['best_type_name']}|{r['bucket_name']}"
        d = cells.setdefault(key, dict(hand_type=r["best_type_name"],
                                       bucket=r["bucket_name"], n=0, plays=0))
        d["n"] += 1
        d["plays"] += int(r["action"] == K.PLAY)
    for d in cells.values():
        d["p_play"] = round(d["plays"] / d["n"], 4) if d["n"] else float("nan")
    big = [d["p_play"] for d in cells.values() if d["n"] >= min_n]
    return dict(
        min_n=min_n, cells=cells, n_cells=len(cells), n_cells_ge_min_n=len(big),
        variance=round(float(np.var(big)), 6) if big else None,
        sd=round(float(np.std(big)), 4) if big else None,
        spread=round(float(max(big) - min(big)), 4) if big else None,
        mean=round(float(np.mean(big)), 4) if big else None,
    )


# --------------------------------------------------------------------------- #
class Runner2(R.Runner):
    """``plast_run.Runner`` with the plast2 config and two learning rates."""

    def setup(self, spec) -> K.Setup:
        key = spec.setup_key(self.window_ms)
        if key not in self._setups:
            cfg = json.loads(CONFIG.read_text())
            self._setups[key] = K.build_setup(
                wiring=spec.wiring, valence_scheme=spec.valence,
                window_ms=self.window_ms, graph=self.graph, cfg=cfg)
        return self._setups[key]

    def _train(self, setup: K.Setup, dec: P.Decider, env, spec, n_hands: int) -> dict:
        """Identical to the parent except for the two etas and the joint table."""
        rng = np.random.default_rng(10_000 + spec.seed)
        fly = K.FlyPolicy(setup, dec, rng)
        on = bool(spec.plasticity)
        eta_r = float(spec.eta) if on else 0.0
        eta_p = float(spec.eta_punish) if on else 0.0

        def on_outcome(rec: dict, kcc) -> dict:
            if kcc is None or not on:
                return dict(dopamine="none", mean_abs_dw=0.0)
            if rec["reward"] and eta_r > 0.0:
                info = setup.plast.deliver("reward", kcc, eta_r)
                return dict(dopamine="reward", eta=eta_r,
                            mean_abs_dw=round(info["mean_abs_dw"], 6),
                            n_changed=info["n_changed"])
            if rec["punish"] and eta_p > 0.0:
                info = setup.plast.deliver("punish", kcc, eta_p)
                return dict(dopamine="punish", eta=eta_p,
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
        third = len(records) // 3
        return dict(
            n_hands=len(records), n_games=len(games), last_seed=seed - 1,
            eta_reward=eta_r, eta_punish=eta_p,
            games=R.summarise_train_games(games),
            summary=K.summarise_games(games),
            reward_rate_rolling=[round(x, 4) for x in K.rolling(rewards, 50)],
            p_play_rolling=[round(x, 4) for x in
                            K.rolling([r["action"] == K.PLAY for r in records], 50)],
            play_drive_rolling=[round(x, 4) for x in
                                K.rolling([r.get("play_drive", 0.0)
                                           for r in records], 50)],
            dopamine_counts=R._counts(records, "dopamine"),
            first_third=R._phase(records[:third]),
            last_third=R._phase(records[-third:]),
            records=records,
            weight_stats=setup.plast.weight_stats(),
        )

    def run(self, spec, train_hands: int, eval_games: int,
            save_weights: Optional[Path] = None) -> dict:
        payload = super().run(spec, train_hands, eval_games,
                              save_weights=save_weights)
        payload["spec"]["eta_reward"] = float(spec.eta)
        payload["spec"]["eta_punish"] = float(spec.eta_punish)
        payload["spec"]["tuning"] = "plast2"
        # The per-KC vectors are 4,064 numbers each; keep a summary and point at
        # the file rather than repeating them in every run json.
        # A private copy: the parent stores ``setup.cfg`` by reference, and the
        # setup is cached across the runs in this process.
        payload["config"] = json.loads(json.dumps(payload["config"]))
        t = payload["config"].get("tuning", {})
        for f in ("kc_vth_offsets", "kc_mbon_out_scale"):
            v = t.get(f)
            if isinstance(v, list):
                a = np.asarray(v, float)
                t[f] = dict(n=len(a), mean=round(float(a.mean()), 4),
                            sd=round(float(a.std()), 4),
                            min=round(float(a.min()), 4),
                            max=round(float(a.max()), 4),
                            source=str(CONFIG.relative_to(K.ROOT)))
        for tag in ("pre", "post"):
            payload[tag]["joint_p_play"] = joint_p_play_table(payload[tag]["records"])
        return payload


# --------------------------------------------------------------------------- #
def run_baselines(runner: Runner2, eval_games: int) -> dict:
    """Reference policies on the same eval games, with their per-hand records kept.

    The records are needed for the joint table and the across-cell variance the
    learned fly is compared against.
    """
    env = K.make_env()
    out: dict = dict(eval_games=eval_games, eval_seed0=K.EVAL_SEED0)
    policies = dict(
        always_play=K.always_play_policy,
        always_discard=K.make_always_discard_policy(),
        teacher_dig=K.make_teacher_dig_policy(1.0),
        bucket_ge1=K.make_bucket_policy(["ge1.0"]),
        bucket_ge_lt1=K.make_bucket_policy(["ge1.0", "lt1.0"]),
        bucket_ge_lt1_lt05=K.make_bucket_policy(["ge1.0", "lt1.0", "lt0.5"]),
    )
    for name, pol in policies.items():
        games = [K.play_game(env, K.EVAL_SEED0 + i, pol) for i in range(eval_games)]
        recs = [h for g in games for h in g.hands]
        out[name] = dict(summary=K.summarise_games(games),
                         p_play_by_hand_type=K.p_play_table(recs, "best_type_name"),
                         p_play_by_bucket=K.p_play_table(recs, "bucket_name"),
                         joint_p_play=joint_p_play_table(recs),
                         records=recs)
        s = out[name]["summary"]
        j = out[name]["joint_p_play"]
        print(f"  {name:20s} clear {s['clear_rate']:.3f} chips {s['chips_mean']:.0f} "
              f"joint var {j['variance']}", flush=True)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated run names")
    ap.add_argument("--train-hands", type=int, default=TRAIN_HANDS)
    ap.add_argument("--eval-games", type=int, default=EVAL_GAMES)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--explore", type=float, default=0.20)
    ap.add_argument("--explore-floor", type=float, default=EXPLORE_FLOOR)
    ap.add_argument("--window-ms", type=float, default=None)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="write run_<name>.json here instead of outputs/plast2")
    ap.add_argument("--save-weights", default=None,
                    help="write the post-training KC -> MBON weights to this .npz")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir) if a.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner2(a.window_ms, a.n_calib, a.explore, a.explore_floor)

    if a.baselines:
        p = out_dir / "baselines.json"
        if a.force or not p.exists():
            print("baselines:")
            K.write_json(p, run_baselines(runner, a.eval_games))
        else:
            print("skip baselines (exists)")

    specs = protocol()
    if a.only:
        wanted = set(a.only.split(","))
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            raise SystemExit(f"no protocol run matches {sorted(wanted)}")
    elif not a.all:
        if not a.baselines:
            raise SystemExit("pass --all, --only NAME, or --baselines")
        specs = []

    for spec in specs:
        path = out_dir / f"run_{spec.name}.json"
        if path.exists() and not a.force:
            print(f"skip {spec.name} (exists)")
            continue
        print(f"== {spec.name} (eta_r={spec.eta:g} eta_p={spec.eta_punish:g}) ==",
              flush=True)
        t0 = time.time()
        payload = runner.run(spec, a.train_hands, a.eval_games,
                             save_weights=R.weights_path(a.save_weights, spec, specs))
        K.write_json(path, payload)
        pre, post = payload["pre"]["summary"], payload["post"]["summary"]
        jp, jq = payload["pre"]["joint_p_play"], payload["post"]["joint_p_play"]
        print(f"   clear {pre['clear_rate']:.3f} -> {post['clear_rate']:.3f} | "
              f"chips {pre['chips_mean']:.0f} -> {post['chips_mean']:.0f} | "
              f"P(play|legal) {pre['p_play_when_legal']:.3f} -> "
              f"{post['p_play_when_legal']:.3f} | joint var {jp['variance']} -> "
              f"{jq['variance']} | train reward "
              f"{payload['train']['summary']['reward_rate']:.3f} | "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
