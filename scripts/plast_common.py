"""Shared harness for the plasticity experiment: one fly, one binary decision per hand.

Task framing (stated here, in the report, and in the run logs)
--------------------------------------------------------------

The harness -- ordinary Python, not biological in any sense -- does everything
except the choice:

* :mod:`flybalatro.hands` enumerates all 218 subsets of the dealt cards,
  classifies and scores them exactly as ``balatro-rs`` does, and reports the
  best one;
* :mod:`flybalatro.features_v2` turns that answer into the 32-bit relay block,
  and :class:`flybalatro.encode.GlomerularMap` turns those 32 bits into tonic
  current on 32 whole ORN glomeruli -- the "odour" of this hand;
* once the fly has chosen, the harness presses the ``select_card[i]`` keys and
  then ``play`` or ``discard``.

The fly's decision is **binary**: play the best subset now, or discard the junk
and dig (only legal while discards remain). So what it can learn is a *valence*:
which hand-odours mean approach and which mean avoid. That is a mushroom-body
computation; enumerating poker subsets is not.

Nothing here is a teacher. The only signal that reaches the fly is chips:
a dopamine pulse after a play that paid its way, and a punishment pulse after a
play that lost the blind. Discards produce no dopamine at all (delayed credit
assignment across a re-deal is beyond a fly).
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent
for _p in (str(ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flybalatro import connectome as C  # noqa: E402
from flybalatro import hands  # noqa: E402
from flybalatro import plasticity as P  # noqa: E402
from flybalatro.brain import Brain  # noqa: E402
from flybalatro.encode import GlomerularMap  # noqa: E402
from flybalatro.env import (  # noqa: E402
    CASH_OUT_INDEX,
    DISCARD_INDEX,
    NEXT_ROUND_INDEX,
    PLAY_INDEX,
    SELECT_BLIND_INDEX,
    BalatroEnv,
)
from flybalatro.features_v2 import (  # noqa: E402
    N_HAND_BLOCK,
    SCORE_BUCKET_LABELS,
    _score_bucket,
    hand_block,
)
from flybalatro.tuning import Tuning  # noqa: E402

OUT_DIR = ROOT / "outputs" / "plast"
#: Frozen spec of the fly. ``plast`` is the mb2 config plus the two APL knobs
#: documented in its ``deviation`` field; ``spec`` is mb2 exactly as chosen.
TUNED_CONFIG = OUT_DIR / "tuned_config.json"
MB2_CONFIG = ROOT / "outputs" / "mb2" / "tuned_config.json"

STAGE_PRE_BLIND = 0
STAGE_BLINDS = (1, 2, 3)
STAGE_POST_BLIND = 4
STAGE_SHOP = 5

#: Seed ranges, disjoint by construction.
CALIB_SEED0 = 300_000
TRAIN_SEED0 = 200_000
EVAL_SEED0 = 100_000

PLAY = "play"
DISCARD = "discard"


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def load_config(variant: str = "plast") -> dict:
    """``"plast"`` = the corrected tuning used here; ``"spec"`` = mb2 unchanged."""
    if variant == "plast":
        return json.loads(TUNED_CONFIG.read_text())
    if variant == "spec":
        return json.loads(MB2_CONFIG.read_text())
    raise ValueError(f"unknown config variant {variant!r}")


@dataclass
class Setup:
    """Everything frozen: graph, encoding, brain, valence map, plastic synapses."""

    graph: object
    cfg: dict
    gm: GlomerularMap
    brain: Brain
    valence: P.MbonValence
    plast: P.KcMbonPlasticity
    kc: NDArray[np.int32]
    window_ms: float
    wiring: str

    def odor_drive(self, bits: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.gm.drive(bits[:N_HAND_BLOCK])

    def run_window(self, bits: NDArray[np.float32]) -> NDArray[np.int32]:
        self.brain.reset()
        counts, _ = self.brain.step(self.odor_drive(bits), self.window_ms)
        return counts


def build_setup(wiring: str = "real", valence_scheme: str = "nt",
                window_ms: Optional[float] = None, shuffle_seed: int = 0,
                graph=None, cfg: Optional[dict] = None) -> Setup:
    """Build the frozen fly. ``wiring`` is ``"real"`` or ``"shuffled"``."""
    cfg = cfg or load_config()
    graph = graph if graph is not None else C.load(5)
    window = float(cfg["window_ms"] if window_ms is None else window_ms)
    gloms = [b["glomerulus"] for b in cfg["encoding"]["assignment"]]
    gm = GlomerularMap(graph, gloms, drive_mv=float(cfg["drive"]["tonic_mv"]))
    brain = Brain(graph, seed=int(cfg["brain_seed"]),
                  v_jitter=float(cfg["v_jitter_mv"]),
                  tuning=Tuning.from_dict(cfg["tuning"]))
    if wiring == "shuffled":
        brain = brain.shuffled(shuffle_seed)
    elif wiring != "real":
        raise ValueError(f"wiring must be 'real' or 'shuffled', got {wiring!r}")
    brain.warmup()
    valence = P.mbon_valence_table(graph, brain.post, brain.weight,
                                   scheme=valence_scheme)
    plast = P.KcMbonPlasticity(brain, graph, valence)
    return Setup(graph=graph, cfg=cfg, gm=gm, brain=brain, valence=valence,
                 plast=plast, kc=graph.kc_indices(), window_ms=window,
                 wiring=wiring)


def make_decider(setup: Setup, explore_floor: float = 0.0) -> P.Decider:
    return P.Decider(approach=setup.valence.approach, avoid=setup.valence.avoid,
                     window_ms=setup.window_ms, explore_floor=explore_floor)


# --------------------------------------------------------------------------- #
# hand-level game driving
# --------------------------------------------------------------------------- #
def dig_slots(available: Sequence[object], best_slots: Sequence[int]) -> Tuple[int, ...]:
    """The worst cards outside the best subset -- what "dig" throws away.

    Same rule as the v2 teacher's ``dig_target`` with nothing selected: lowest
    rank first, at most :data:`flybalatro.hands.MAX_SELECTED` cards.
    """
    keep = set(int(s) for s in best_slots)
    outside = [i for i in range(len(available)) if i not in keep]
    if not outside:
        return ()
    outside.sort(key=lambda i: (hands.RANK_CHIPS[available[i].rank_index], i))
    return tuple(sorted(outside[: hands.MAX_SELECTED]))


@dataclass
class HandContext:
    """What the harness knows about one decision point, before the choice."""

    bits: NDArray[np.float32]
    best_type: int
    best_type_name: str
    best_score: int
    best_slots: Tuple[int, ...]
    dig: Tuple[int, ...]
    needed: int
    plays: int
    discards: int
    bucket: int
    bucket_name: str
    round: int
    discard_ok: bool


def hand_context(state: object) -> Optional[HandContext]:
    """Analyse a blind state with nothing selected; ``None`` if there is no hand."""
    bits, analysis, needed = hand_block(state)
    if analysis is None or not analysis.has_cards:
        return None
    available = list(state.available)[: hands.HAND_SLOTS]
    dig = dig_slots(available, analysis.best_slots)
    bucket = _score_bucket(analysis.best_score, needed)
    discards = int(state.discards)
    return HandContext(
        bits=bits, best_type=analysis.best_type,
        best_type_name=analysis.best_type_name, best_score=int(analysis.best_score),
        best_slots=tuple(analysis.best_slots), dig=dig, needed=needed,
        plays=int(state.plays), discards=discards, bucket=bucket,
        bucket_name=SCORE_BUCKET_LABELS[bucket], round=int(state.round),
        discard_ok=bool(discards > 0 and dig),
    )


#: A policy maps a hand context to ``(action, extra log fields)``.
Policy = Callable[[HandContext], Tuple[str, dict]]


def always_play_policy(ctx: HandContext) -> Tuple[str, dict]:
    return PLAY, {}


def make_teacher_dig_policy(play_share: float = 1.0) -> Policy:
    """The v2 teacher's dig rule as a binary decision, nothing else borrowed."""

    def policy(ctx: HandContext) -> Tuple[str, dict]:
        if ctx.needed <= 0:
            return PLAY, {}
        fair = ctx.best_score * max(1, ctx.plays) >= ctx.needed * play_share
        if ctx.discard_ok and not fair:
            return DISCARD, {}
        return PLAY, {}

    return policy


def make_bucket_policy(play_buckets: Sequence[str]) -> Policy:
    """Best possible odour-only policy: play iff the score bucket is in the set.

    The fly's odour carries the hand type, the best-subset mask and the
    score-vs-needed bucket, but **not** how many plays or discards are left. This
    is therefore the ceiling for any map from this odour to a binary action, and
    the number the fly has to be measured against -- not the v2 teacher, which
    reads ``plays_left`` directly.
    """
    keep = set(str(b) for b in play_buckets)

    def policy(ctx: HandContext) -> Tuple[str, dict]:
        if not ctx.discard_ok:
            return PLAY, {}
        return (PLAY if ctx.bucket_name in keep else DISCARD), {}

    return policy


def make_always_discard_policy() -> Policy:
    def policy(ctx: HandContext) -> Tuple[str, dict]:
        return (DISCARD if ctx.discard_ok else PLAY), {}

    return policy


class FlyPolicy:
    """One 50 ms window from reset per decision, then a softmax on the MBONs."""

    def __init__(self, setup: Setup, decider: P.Decider, rng: np.random.Generator,
                 record_counts: bool = False) -> None:
        self.setup = setup
        self.decider = decider
        self.rng = rng
        self.record_counts = record_counts
        self.last_kc_counts: Optional[NDArray[np.int32]] = None

    def __call__(self, ctx: HandContext) -> Tuple[str, dict]:
        counts = self.setup.run_window(ctx.bits)
        kcc = counts[self.setup.kc]
        self.last_kc_counts = kcc.copy()
        raw = self.decider.raw_drive(counts)
        drive = raw + self.decider.bias
        p = self.decider.p_play(counts)
        if not ctx.discard_ok:
            action = PLAY
            explored = False
        else:
            action = PLAY if float(self.rng.random()) < p else DISCARD
            explored = (action == PLAY) != (p >= 0.5)
        scale = 1000.0 / self.setup.window_ms
        return action, dict(
            raw_drive=round(raw, 4),
            play_drive=round(drive, 4),
            p_play=round(p, 4),
            explored=bool(explored),
            kc_active=int((kcc > 0).sum()),
            kc_spikes=int(kcc.sum()),
            approach_hz=round(float(counts[self.decider.approach].mean()) * scale, 3),
            avoid_hz=round(float(counts[self.decider.avoid].mean()) * scale, 3),
        )


def resolve_outcome(ctx: HandContext, action: str, info: dict, done: bool) -> dict:
    """What the game said about a resolved hand, and whether dopamine is due.

    The *only* definition of reward and punishment in the project, so the live
    viewer and the batch runs cannot drift apart:

    * ``cleared`` -- the blind is over, or this play alone met what was needed;
    * ``lost``    -- the run ended on this play and it was not a win or a
      truncation;
    * ``reward``  -- a PLAY that cleared or paid its fair share
      (``needed / plays_left``), which is the v2 teacher's inequality;
    * ``punish``  -- a PLAY that lost the blind.

    A DISCARD never produces either: crediting a re-deal across a discard is
    delayed credit assignment and beyond a fly.
    """
    chips = int(info["chips_gained"])
    stage_after = int(info["stage"])
    cleared = (stage_after in (STAGE_POST_BLIND, STAGE_SHOP)
               or chips >= ctx.needed > 0)
    truncated = bool(info["truncated"])
    lost = bool(done and not bool(info["is_win"]) and not truncated and not cleared)
    share = ctx.needed / max(1, ctx.plays)
    reward = int(action == PLAY and (cleared or chips >= share))
    punish = int(action == PLAY and lost and not reward)
    return dict(chips_gained=chips, cleared=bool(cleared), lost=bool(lost),
                truncated=truncated, share=float(share),
                reward=reward, punish=punish)


@dataclass
class GameResult:
    seed: int
    hands: List[dict]
    cleared: bool
    chips: int
    rounds: int
    truncated: bool


def play_game(env: BalatroEnv, seed: int, policy: Policy,
              on_outcome: Optional[Callable[[dict, Optional[NDArray[np.int32]]], dict]] = None,
              fly: Optional[FlyPolicy] = None,
              max_hands: int = 60) -> GameResult:
    """One ante-1 run. The policy decides play-or-dig; the harness does the rest.

    ``on_outcome(record, kc_counts)`` is called after each hand resolves and may
    deliver dopamine; whatever it returns is merged into the log record.
    """
    env.reset(seed=seed)
    records: List[dict] = []
    while not env.done and len(records) < max_hands:
        state = env.engine.state
        stage = int(state.stage.int())
        mask = env.action_mask()
        if stage == STAGE_PRE_BLIND:
            env.step(SELECT_BLIND_INDEX if mask[SELECT_BLIND_INDEX] else int(np.flatnonzero(mask)[0]))
            continue
        if stage == STAGE_POST_BLIND:
            env.step(CASH_OUT_INDEX if mask[CASH_OUT_INDEX] else int(np.flatnonzero(mask)[0]))
            continue
        if stage == STAGE_SHOP:
            env.step(NEXT_ROUND_INDEX if mask[NEXT_ROUND_INDEX] else int(np.flatnonzero(mask)[0]))
            continue
        if stage not in STAGE_BLINDS:
            env.step(int(np.flatnonzero(mask)[0]))
            continue
        ctx = hand_context(state)
        if ctx is None:
            env.step(int(np.flatnonzero(mask)[0]))
            continue

        action, extra = policy(ctx)
        if action == DISCARD and not ctx.discard_ok:
            action = PLAY
        target = ctx.best_slots if action == PLAY else ctx.dig
        act_index = PLAY_INDEX if action == PLAY else DISCARD_INDEX

        # Press the keys: select the target slots, then play or discard.
        for slot in target:
            mask = env.action_mask()
            if slot < len(mask) and mask[slot] == 1:
                env.step(int(slot))
        mask = env.action_mask()
        if mask[act_index] != 1:
            other = DISCARD_INDEX if act_index == PLAY_INDEX else PLAY_INDEX
            if mask[other] == 1:
                act_index, action = other, (PLAY if other == PLAY_INDEX else DISCARD)
            else:
                env.step(int(np.flatnonzero(mask)[0]))
                continue
        _f, _m, _r, done, info = env.step(act_index)

        out = resolve_outcome(ctx, action, info, done)
        chips = out["chips_gained"]
        cleared, lost = out["cleared"], out["lost"]
        reward, punish = out["reward"], out["punish"]

        record = dict(
            seed=int(seed), round=ctx.round, hand_in_game=len(records),
            best_type=int(ctx.best_type), best_type_name=ctx.best_type_name,
            best_score=int(ctx.best_score), bucket=int(ctx.bucket),
            bucket_name=ctx.bucket_name, needed=int(ctx.needed),
            plays_before=int(ctx.plays), discards_before=int(ctx.discards),
            discard_ok=bool(ctx.discard_ok), bits_on=int(ctx.bits.sum()),
            odor=int(_odor_key(ctx.bits)),
            action=action, chips_gained=chips, cleared=bool(cleared),
            lost=bool(lost), reward=reward, punish=punish,
        )
        record.update(extra)
        if on_outcome is not None:
            kcc = fly.last_kc_counts if fly is not None else None
            record.update(on_outcome(record, kcc))
        records.append(record)

    return GameResult(
        seed=int(seed), hands=records, cleared=bool(env.is_win),
        chips=int(env.chips_total), rounds=int(env.engine.state.round),
        truncated=bool(env.steps >= 500),
    )


def _odor_key(bits: NDArray[np.float32]) -> int:
    """The 32 relay bits packed into one integer, so odours can be grouped."""
    on = np.flatnonzero(np.asarray(bits[:N_HAND_BLOCK]).astype(bool))
    k = 0
    for i in on:
        k |= 1 << int(i)
    return k


def make_env() -> BalatroEnv:
    """Ante-1 env with a no-op encoder: the fly's input comes from ``hand_block``."""
    return BalatroEnv(ante_end=1, mask_noop_actions=True,
                      encoder=lambda s: np.zeros(1, np.float32), n_features=1)


# --------------------------------------------------------------------------- #
# aggregation helpers
# --------------------------------------------------------------------------- #
def p_play_table(records: Sequence[dict], key: str) -> Dict[str, dict]:
    """P(play) grouped by ``record[key]``, decisions where discard was legal."""
    out: Dict[str, dict] = {}
    for r in records:
        if not r.get("discard_ok"):
            continue
        k = str(r[key])
        d = out.setdefault(k, dict(n=0, plays=0))
        d["n"] += 1
        d["plays"] += int(r["action"] == PLAY)
    for d in out.values():
        d["p_play"] = d["plays"] / d["n"] if d["n"] else float("nan")
    return out


def summarise_games(games: Sequence[GameResult]) -> dict:
    n = len(games)
    if n == 0:
        return dict(n_games=0)
    hands = [h for g in games for h in g.hands]
    plays = [h for h in hands if h["action"] == PLAY]
    return dict(
        n_games=n,
        n_hands=len(hands),
        clear_rate=sum(g.cleared for g in games) / n,
        chips_mean=float(np.mean([g.chips for g in games])),
        chips_sd=float(np.std([g.chips for g in games])),
        rounds_mean=float(np.mean([g.rounds for g in games])),
        hands_per_game=len(hands) / n,
        p_play=float(np.mean([h["action"] == PLAY for h in hands])) if hands else 0.0,
        p_play_when_legal=float(np.mean([h["action"] == PLAY for h in hands
                                         if h["discard_ok"]]))
        if any(h["discard_ok"] for h in hands) else float("nan"),
        reward_rate=float(np.mean([h["reward"] for h in plays])) if plays else 0.0,
        punish_rate=float(np.mean([h["punish"] for h in plays])) if plays else 0.0,
        truncated=sum(g.truncated for g in games),
    )


def rolling(values: Sequence[float], window: int = 50) -> List[float]:
    v = np.asarray(values, np.float64)
    if len(v) == 0:
        return []
    c = np.cumsum(np.insert(v, 0, 0.0))
    out: List[float] = []
    for i in range(len(v)):
        a = max(0, i - window + 1)
        out.append(float((c[i + 1] - c[a]) / (i + 1 - a)))
    return out


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_default) + "\n")
    tmp.replace(path)
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")
