"""The preregistered reward-function comparison: A, B, C x 3 gammas, D.

See ``outputs/plast4/PREREGISTRATION.md``, written before any run here started.

Every condition is v3's winning cell (``separated`` encoding, ``omission``
immediate punishment, ``eta_reward = 0.05`` / ``eta_punish = 0.0797``, 1,200
training decisions, 10% exploration floor) with **only** the reward
specification changed:

| condition | reward | eligibility trace |
|---|---|---|
| ``A``   | share (v3, unchanged) | none |
| ``B``   | pace (cumulative schedule) | none |
| ``C50`` / ``C80`` / ``C100`` | share | gamma = 0.5 / 0.8 / 1.0 |
| ``D``   | pace | gamma = 0.8 (fixed in advance) |

3 training seeds per condition, evaluated on the **same 400 game seeds as v3**
(400000-400399), greedy and sampled. One frozen control, shared: the reward only
affects learning, so the naive fly is the same fly in every condition.

Run (resumable; skips any run whose json exists)::

    python -m scripts.plast4_run --gate          # frozen + A x3, then verify vs v3
    python -m scripts.plast4_run --all
    python -m scripts.plast4_run --only C80_s1
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
from scripts import plast3_common as C3
from scripts import plast3_run as R3
from scripts import plast4_common as C4
from scripts import plast_common as K
from scripts import plast_run as R
from scripts.plast_probe import collect_calibration_hands, hand_bits

OUT = C4.OUT_DIR
ENCODING = C3.ENC_SEPARATED
CONFIG = K.ROOT / "outputs" / "plast3" / "tuned_config_sep.json"
ETA_FILE = K.ROOT / "outputs" / "plast3" / "eta_calib_sep.json"
FROZEN = "frozen"


@dataclass(frozen=True)
class Cell:
    name: str
    condition: Optional[C4.Condition]
    seed: int
    eta_reward: float = C3.ETA_REWARD3
    eta_punish: float = 0.0
    plasticity: bool = True


def eta_punish_for(eta_reward: float) -> Tuple[float, dict]:
    row = json.loads(ETA_FILE.read_text())["chosen"][f"{eta_reward:g}"]
    return float(row["eta_punish"]), row


def protocol() -> List[Cell]:
    cells = [Cell(FROZEN, None, 0, eta_reward=0.0, plasticity=False)]
    ep, _ = eta_punish_for(C3.ETA_REWARD3)
    for cond in C4.conditions():
        for s in C3.TRAIN3_SEEDS:
            cells.append(Cell(f"{cond.name}_s{s}", cond, s, eta_punish=ep))
    return cells


class Runner4:
    """One process, one fly, every condition. Expensive objects built once."""

    def __init__(self, n_calib: int = 200, explore: float = 0.20,
                 eval_games: int = C3.EVAL3_GAMES) -> None:
        self.graph = K.C.load(5)
        self.n_calib = n_calib
        self.explore = explore
        self.eval_games = int(eval_games)
        self.env = K.make_env()
        self._setup: Optional[K.Setup] = None
        self._calib: Optional[dict] = None
        self._calib_bits: Optional[np.ndarray] = None
        self.seeds = C3.eval_seeds(self.eval_games)

    def calib_bits(self) -> np.ndarray:
        if self._calib_bits is None:
            recs = collect_calibration_hands(self.n_calib, K.CALIB_SEED0, rng_seed=7)
            self._calib_bits = hand_bits(recs)
        return self._calib_bits

    def setup(self) -> K.Setup:
        if self._setup is None:
            cfg = json.loads(CONFIG.read_text())
            self._setup = C3.build_setup3(ENCODING, cfg, self.graph)
        return self._setup

    def decider(self) -> Tuple[P.Decider, dict]:
        setup = self.setup()
        dec = K.make_decider(setup, explore_floor=C3.EXPLORE_FLOOR3)
        if self._calib is None:
            setup.plast.reset_weights()
            raw = [dec.raw_drive(setup.run_window(b)) for b in self.calib_bits()]
            cal = dec.calibrate(raw, explore=self.explore)
            cal["bias"], cal["temperature"] = dec.bias, dec.temperature
            self._calib = cal
        dec.bias = float(self._calib["bias"])
        dec.temperature = float(self._calib["temperature"])
        return dec, self._calib

    # -- one cell ----------------------------------------------------------
    def run(self, cell: Cell, train_hands: int,
            save_weights: Optional[Path] = None) -> dict:
        t0 = time.time()
        setup = self.setup()
        dec, cal = self.decider()
        setup.plast.reset_weights()
        cond = cell.condition
        reward_rule = cond.reward_rule if cond is not None else C4.REWARD_SHARE

        payload: dict = dict(
            cell=dict(name=cell.name, seed=cell.seed, encoding=ENCODING,
                      punishment=C3.PUNISH_OMISSION,
                      reward_rule=reward_rule,
                      gamma=(cond.gamma if cond is not None else None),
                      condition=(cond.name if cond is not None else FROZEN),
                      label=(cond.label if cond is not None else "frozen control"),
                      eta_reward=cell.eta_reward, eta_punish=cell.eta_punish,
                      plasticity=cell.plasticity),
            config=str(CONFIG.relative_to(K.ROOT)),
            config_label=setup.cfg.get("label"),
            window_ms=setup.window_ms, calibration=cal, decider=dec.to_dict(),
            valence_summary=setup.valence.summary(),
            plasticity_pools=setup.plast.describe(),
            odour_map=setup.gm.describe(),
            train_hands=int(train_hands) if cell.plasticity else 0,
            eval_seeds=dict(seed0=C3.EVAL3_SEED0, n=self.eval_games),
            seeds=dict(calib=K.CALIB_SEED0,
                       train=K.TRAIN_SEED0 + 1000 * cell.seed),
        )

        if cell.plasticity:
            payload["train"] = self._train(setup, dec, cell, train_hands)
            if save_weights is not None:
                meta = dict(run=cell.name, encoding=ENCODING,
                            punishment=C3.PUNISH_OMISSION,
                            reward_rule=reward_rule,
                            gamma=(cond.gamma if cond is not None else None),
                            seed=cell.seed, eta_reward=cell.eta_reward,
                            eta_punish=cell.eta_punish,
                            train_hands=int(payload["train"]["n_hands"]),
                            bias=float(dec.bias), temperature=float(dec.temperature),
                            explore_floor=float(dec.explore_floor),
                            window_ms=float(setup.window_ms),
                            dopamine_counts=payload["train"]["dopamine_counts"])
                payload["saved_weights"] = str(
                    setup.plast.save_weights(save_weights, meta=meta))

        payload["eval"] = {m: self._evaluate(setup, dec, m, reward_rule)
                           for m in C3.EVAL_MODES}
        payload["weight_stats"] = setup.plast.weight_stats()
        payload["wall_seconds"] = round(time.time() - t0, 1)
        setup.plast.reset_weights()
        return payload

    # -- training ----------------------------------------------------------
    def _train(self, setup: K.Setup, dec: P.Decider, cell: Cell,
               n_hands: int) -> dict:
        cond = cell.condition
        assert cond is not None
        rng = np.random.default_rng(10_000 + cell.seed)
        fly = K.FlyPolicy(setup, dec, rng)
        eta_r, eta_p = float(cell.eta_reward), float(cell.eta_punish)
        tracer = (C4.TraceRunner(setup.plast, float(cond.gamma), eta_p)
                  if cond.has_trace else None)
        ledger: Dict[str, dict] = {}

        def bucket_row(name: str) -> dict:
            return ledger.setdefault(str(name), dict(
                plays=0, reward=0, punish_immediate=0, punish_trace_weight=0.0,
                n_terminal_pulses=0, rewarded_but_lost=0))

        def on_outcome(rec: dict, kcc) -> dict:
            extra: dict = {}
            kind = C4.dopamine_for4(rec)
            if kind is None or kcc is None:
                extra.update(dopamine="none", mean_abs_dw=0.0)
            else:
                row = bucket_row(rec["bucket_name"])
                row["plays"] += 1
                if kind == "reward":
                    row["reward"] += 1
                else:
                    row["punish_immediate"] += 1
                if rec["reward"] and rec["lost"]:
                    row["rewarded_but_lost"] += 1
                eta = eta_r if kind == "reward" else eta_p
                if eta <= 0.0:
                    extra.update(dopamine="none", mean_abs_dw=0.0)
                else:
                    info = setup.plast.deliver(kind, kcc, eta)
                    extra.update(dopamine=kind, eta=eta,
                                 mean_abs_dw=round(info["mean_abs_dw"], 6),
                                 n_changed=info["n_changed"])

            if tracer is not None:
                extra.update(tracer.on_record(rec, kcc))
            return extra

        records: List[dict] = []
        games: List[K.GameResult] = []
        seed = K.TRAIN_SEED0 + 1000 * cell.seed
        while len(records) < n_hands:
            g = C4.play_game4(self.env, seed, fly,
                              reward_rule=cond.reward_rule,
                              on_outcome=on_outcome, fly=fly,
                              max_hands=n_hands - len(records))
            games.append(g)
            records.extend(g.hands)
            seed += 1
        third = len(records) // 3
        if tracer is not None:
            for b, w in tracer.weight_by_bucket.items():
                bucket_row(b)["punish_trace_weight"] = round(float(w), 3)
            for b, n in tracer.pulses_by_bucket.items():
                bucket_row(b)["n_terminal_pulses"] = int(n)
        return dict(
            n_hands=len(records), n_games=len(games), last_seed=seed - 1,
            eta_reward=eta_r, eta_punish=eta_p,
            trace=(tracer.describe() if tracer is not None else None),
            n_terminal_pulses=(tracer.n_pulses if tracer is not None else 0),
            trace_weight_delivered=(round(tracer.weight_delivered, 2)
                                    if tracer is not None else 0.0),
            summary=K.summarise_games(games),
            reward_rate_rolling=[round(x, 4) for x in
                                 K.rolling([r["reward"] for r in records], 50)],
            p_play_rolling=[round(x, 4) for x in
                            K.rolling([r["action"] == K.PLAY for r in records], 50)],
            dopamine_counts=R._counts(records, "dopamine"),
            dopamine_by_bucket=R3._dopamine_by_bucket(records),
            ledger_by_bucket=ledger,
            reward_rule_agreement=_reward_agreement(records),
            first_third=R._phase(records[:third]),
            last_third=R._phase(records[-third:]),
            weight_stats=setup.plast.weight_stats(),
            records=records,
        )

    # -- evaluation --------------------------------------------------------
    def _evaluate(self, setup: K.Setup, dec: P.Decider, mode: str,
                  reward_rule: str) -> dict:
        rng = (np.random.default_rng(C3.EVAL3_RNG_SEED) if mode == C3.SAMPLED
               else None)
        pol = C3.EvalFlyPolicy(setup, dec, mode, rng=rng, cache=True,
                               encoding=ENCODING)
        games = C4.evaluate_policy4(pol, self.seeds, env=self.env,
                                    reward_rule=reward_rule)
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


def _reward_agreement(records: Sequence[dict]) -> dict:
    """How often the two reward rules disagree on this run's own plays."""
    n = s_only = p_only = both = neither = 0
    for r in records:
        if r["action"] != K.PLAY:
            continue
        n += 1
        s, p = bool(r["reward_share"]), bool(r["reward_pace"])
        both += int(s and p)
        s_only += int(s and not p)
        p_only += int(p and not s)
        neither += int(not s and not p)
    return dict(n_plays=n, both=both, share_only=s_only, pace_only=p_only,
                neither=neither,
                flip_fraction=round((s_only + p_only) / n, 5) if n else 0.0)


def run_baselines(runner: Runner4) -> dict:
    """Reference policies on the same 400 evaluation games."""
    out: dict = dict(eval_seed0=C3.EVAL3_SEED0, eval_games=runner.eval_games)
    policies = dict(
        always_play=K.always_play_policy,
        always_discard=K.make_always_discard_policy(),
        teacher_dig=K.make_teacher_dig_policy(1.0),
        bucket_ge1=K.make_bucket_policy(["ge1.0"]),
        bucket_ge_lt1=K.make_bucket_policy(["ge1.0", "lt1.0"]),
        bucket_ge_lt1_lt05=K.make_bucket_policy(["ge1.0", "lt1.0", "lt0.5"]),
    )
    for name, pol in policies.items():
        games = C4.evaluate_policy4(pol, runner.seeds, env=runner.env)
        recs = [h for g in games for h in g.hands]
        out[name] = dict(summary=K.summarise_games(games),
                         p_play_by_bucket=K.p_play_table(recs, "bucket_name"),
                         cleared=[bool(g.cleared) for g in games])
        print(f"  {name:20s} clear {out[name]['summary']['clear_rate']:.3f} "
              f"chips {out[name]['summary']['chips_mean']:.0f}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# the provenance gate
# --------------------------------------------------------------------------- #
GATE_PAIRS = (
    ("run_frozen.json", "run_frozen_separated.json", "weights_frozen.npz", None),
    ("run_A_s0.json", "run_omission_separated_s0.json",
     "weights_A_s0.npz", "weights_omission_separated_s0.npz"),
    ("run_A_s1.json", "run_omission_separated_s1.json",
     "weights_A_s1.npz", "weights_omission_separated_s1.npz"),
    ("run_A_s2.json", "run_omission_separated_s2.json",
     "weights_A_s2.npz", "weights_omission_separated_s2.npz"),
)


def check_gate(out_dir: Path) -> dict:
    """Require the v4 harness to reproduce v3's frozen control and A runs exactly."""
    v3 = K.ROOT / "outputs" / "plast3"
    rows: List[dict] = []
    ok_all = True
    for new_name, old_name, new_w, old_w in GATE_PAIRS:
        new_p, old_p = out_dir / new_name, v3 / old_name
        row: dict = dict(new=C4.rel_path(new_p), old=C4.rel_path(old_p))
        if not new_p.exists():
            row.update(status="missing"); rows.append(row); ok_all = False; continue
        a, b = json.loads(new_p.read_text()), json.loads(old_p.read_text())
        checks: Dict[str, bool] = {}
        for mode in C3.EVAL_MODES:
            checks[f"{mode}_cleared"] = a["eval"][mode]["cleared"] == b["eval"][mode]["cleared"]
            checks[f"{mode}_chips"] = a["eval"][mode]["chips"] == b["eval"][mode]["chips"]
            checks[f"{mode}_clear_rate"] = (
                a["eval"][mode]["summary"]["clear_rate"]
                == b["eval"][mode]["summary"]["clear_rate"])
        if old_w is not None:
            wa = P.read_weights(out_dir / new_w)["weight"]
            wb = P.read_weights(v3 / old_w)["weight"]
            checks["weights_bit_identical"] = bool(np.array_equal(wa, wb))
            checks["n_synapses_differing"] = int((wa != wb).sum())
        row["checks"] = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                         for k, v in checks.items()}
        bad = [k for k, v in checks.items()
               if (v is False) or (k == "n_synapses_differing" and v != 0)]
        row["status"] = "ok" if not bad else "FAIL"
        row["failed"] = bad
        ok_all &= not bad
        rows.append(row)
    return dict(passed=bool(ok_all), rows=rows)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gate", action="store_true",
                    help="run the frozen control and A x3, then verify against v3")
    ap.add_argument("--check-gate", action="store_true",
                    help="verify an already-run gate without running anything")
    ap.add_argument("--only", default=None, help="comma-separated cell names")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--train-hands", type=int, default=C3.TRAIN3_HANDS)
    ap.add_argument("--eval-games", type=int, default=C3.EVAL3_GAMES)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.check_gate:
        res = check_gate(out_dir)
        C4.write_json(out_dir / "gate.json", res)
        print(json.dumps(res, indent=2))
        raise SystemExit(0 if res["passed"] else 1)

    cells = protocol()
    if a.gate:
        wanted = {FROZEN, "A_s0", "A_s1", "A_s2"}
        cells = [c for c in cells if c.name in wanted]
    elif a.only:
        wanted = set(a.only.split(","))
        cells = [c for c in cells if c.name in wanted]
        if not cells:
            raise SystemExit(f"no cell matches {sorted(wanted)}")
    elif not a.all:
        cells = []

    runner = Runner4(a.n_calib, eval_games=a.eval_games) if (cells or a.baselines or a.all) else None

    if a.baselines or a.all:
        p = out_dir / "baselines4.json"
        if a.force or not p.exists():
            print("baselines (400 games):", flush=True)
            C4.write_json(p, run_baselines(runner))
        else:
            print("skip baselines4.json (exists)")

    for cell in cells:
        path = out_dir / f"run_{cell.name}.json"
        if path.exists() and not a.force:
            print(f"skip {cell.name} (exists)")
            continue
        cond = cell.condition
        print(f"== {cell.name} (reward={cond.reward_rule if cond else '-'} "
              f"gamma={cond.gamma if cond else '-'} "
              f"eta_r={cell.eta_reward:g} eta_p={cell.eta_punish:g}) ==", flush=True)
        wpath = out_dir / f"weights_{cell.name}.npz"
        payload = runner.run(cell, a.train_hands,
                             save_weights=wpath if cell.plasticity else None)
        C4.write_json(path, payload)
        g, s = payload["eval"][C3.GREEDY], payload["eval"][C3.SAMPLED]
        tr = payload.get("train", {}).get("n_terminal_pulses", 0)
        print(f"   greedy clear {g['summary']['clear_rate']:.3f} "
              f"P(play|legal) {g['summary']['p_play_when_legal']:.3f} "
              f"grad {g['bucket_gradient']} | sampled clear "
              f"{s['summary']['clear_rate']:.3f} | terminal pulses {tr} "
              f"| {payload['wall_seconds']:.0f}s", flush=True)

    if a.gate:
        res = check_gate(out_dir)
        C4.write_json(out_dir / "gate.json", res)
        print("\n== provenance gate ==")
        for row in res["rows"]:
            print(f"  {row['status']:4s} {row['new']} vs {row['old']}"
                  + (f"  failed: {row.get('failed')}" if row.get("failed") else ""))
        print(f"  gate passed: {res['passed']}")
        raise SystemExit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
