"""Shared pieces for plasticity v4: reward functions, and an eligibility trace.

See ``outputs/plast4/PREREGISTRATION.md``, written before any v4 game was played.

v3 ended with a measured diagnosis rather than a result: the fly converged on the
exact optimum of the reward it was given (``chips_gained >= needed / plays_left``,
"did this play earn its share") and that optimum ties always-discard. This
module holds the two things v4 varies, and nothing else:

1. **The reward predicate.** ``share`` is v3's, unchanged. ``pace`` asks instead
   whether the *cumulative* round score after the play is at or above the linear
   schedule that clears the blind in the plays it started with, so a shortfall is
   not forgiven by re-baselining. Both read only ``required_score``, ``score``,
   ``plays`` and ``chips_gained``: the game's own payouts and costs. Neither
   consults :mod:`flybalatro.hands` about what the *right* play would have been;
   that constraint is the one this whole project runs under.

2. **The eligibility trace.** The v3 audit in the preregistration shows that
   ``reward AND lost`` is impossible by construction (on the last play of a
   blind the fair share *is* the whole remaining requirement, so a rewarded last
   play has cleared) and confirms it empirically (0 of 2,374 training plays).
   So "paid its share but still lost the blind" cannot be expressed by any
   immediate same-hand term at all: it lives in the *earlier* plays of a blind
   that was later lost, and only a trace can reach them.
   :class:`EligibilityTrace` keeps a decaying per-Kenyon-cell trace over the
   plays of the current blind and hands it to
   :meth:`flybalatro.plasticity.KcMbonPlasticity.deliver_eligibility` when the
   blind is lost.

Everything else (the encoding, the immediate ``omission`` punishment, the
calibrated etas, the decider, the 400 paired evaluation seeds) is v3's and is
imported, not re-specified. ``plast_common.py`` and ``plast3_common.py`` are
imported unchanged.

:func:`play_game4` is a reimplementation of :func:`plast_common.play_game`, not a
wrapper: the pace reward needs ``state.required_score`` and the blind's starting
play count, which the v3 record does not carry, and ``plast_common.py`` is
additive-only this round. Because a reimplementation is a risk, the
preregistration gates the whole round on it reproducing v3's frozen control and
its three ``omission_separated`` runs bit for bit, and
``tests/test_plast4.py::test_play_game4_matches_play_game`` asserts record-level
equality on a fixed policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from flybalatro import plasticity as P
from flybalatro.env import (
    CASH_OUT_INDEX,
    DISCARD_INDEX,
    NEXT_ROUND_INDEX,
    PLAY_INDEX,
    SELECT_BLIND_INDEX,
    BalatroEnv,
)
from scripts import plast3_common as C3
from scripts import plast_common as K

OUT_DIR = K.ROOT / "outputs" / "plast4"

# --------------------------------------------------------------------------- #
# reward predicates (factor: reward specification)
# --------------------------------------------------------------------------- #
#: v3's reward: a PLAY that cleared, or whose chips met ``needed / plays_left``.
REWARD_SHARE: str = "share"
#: The cumulative-schedule reward: see :func:`pace_ok`.
REWARD_PACE: str = "pace"
REWARDS: Tuple[str, ...] = (REWARD_SHARE, REWARD_PACE)


def share_ok(needed: int, plays: int, chips: int) -> bool:
    """v3's inequality, spelled out: did this play earn its share of what is left.

    ``plays`` is the number of plays remaining **including this one** (the game
    reports 4, 3, 2, 1 over a blind), so on the last play the rule demands
    ``chips >= needed``, i.e. it demands clearing. There is no off-by-one; see
    section 2 of the preregistration.
    """
    return float(chips) >= float(needed) / float(max(1, int(plays)))


def pace_ok(required: int, score_before: int, plays: int, p0: int,
            chips: int) -> bool:
    """Is the cumulative round score after this play at or above the schedule.

    The schedule is linear in plays used: a blind that started with ``p0`` plays
    and needs ``required`` chips is on pace after ``k`` plays when the round
    score is at least ``required * k / p0``. Written without division so it is
    exact in integers::

        (score_before + chips) * p0 >= required * (p0 - plays + 1)

    Unlike :func:`share_ok` this does **not** re-baseline: a shortfall stays a
    shortfall for the rest of the blind, and an overshoot stays banked. The two
    agree exactly when the blind is on schedule and on the first play of every
    blind.
    """
    used_after = int(p0) - int(plays) + 1
    return (int(score_before) + int(chips)) * int(p0) >= int(required) * used_after


def resolve_outcome4(required: int, score_before: int, needed: int, plays: int,
                     p0: int, action: str, info: dict, done: bool,
                     reward_rule: str) -> dict:
    """:func:`plast_common.resolve_outcome` plus the pace branch and its inputs.

    ``cleared``, ``lost`` and the ``share`` reward are computed by exactly the
    same expressions as v3 so that ``reward_rule = "share"`` reproduces it; the
    pace reward is the only addition. Both reward branches are always computed
    and both are recorded, so a run's log shows what the *other* rule would have
    said about every hand.
    """
    if reward_rule not in REWARDS:
        raise ValueError(f"reward_rule must be one of {REWARDS}, got {reward_rule!r}")
    chips = int(info["chips_gained"])
    stage_after = int(info["stage"])
    cleared = (stage_after in (K.STAGE_POST_BLIND, K.STAGE_SHOP)
               or chips >= needed > 0)
    truncated = bool(info["truncated"])
    lost = bool(done and not bool(info["is_win"]) and not truncated and not cleared)
    is_play = action == K.PLAY
    r_share = int(is_play and (cleared or share_ok(needed, plays, chips)))
    r_pace = int(is_play and (cleared or pace_ok(required, score_before, plays,
                                                 p0, chips)))
    reward = r_share if reward_rule == REWARD_SHARE else r_pace
    punish = int(is_play and lost and not reward)
    return dict(chips_gained=chips, cleared=bool(cleared), lost=bool(lost),
                truncated=truncated, share=float(needed) / float(max(1, plays)),
                reward=int(reward), punish=int(punish),
                reward_share=int(r_share), reward_pace=int(r_pace))


# --------------------------------------------------------------------------- #
# the eligibility trace (factor: credit assignment)
# --------------------------------------------------------------------------- #
#: ``gamma`` values swept. 0.5 is a half-life of about one hand, 1.0 is a flat
#: trace over the blind; see section 3 of the preregistration for the mapping to
#: the fly literature's seconds-to-tens-of-seconds window.
TRACE_GAMMAS: Tuple[float, ...] = (0.5, 0.8, 1.0)
#: ``gamma`` for condition D, fixed in advance and never chosen from C's results.
TRACE_GAMMA_D: float = 0.8


class EligibilityTrace:
    """Decaying per-Kenyon-cell eligibility over the plays of one blind.

    ``update`` is called once per PLAY with that decision's eligibility vector
    (the existing presynaptic factor of the learning rule, spike counts saturating
    at ``KC_REF``); ``reset`` is called at every blind boundary. The trace is
    clipped to ``[0, 1]`` so it stays inside the range a single decision's
    eligibility occupies and the depression step keeps every invariant it has.

    Three choices are fixed by the preregistration and are not swept:

    * only PLAY decisions enter the trace, because a DISCARD earns no dopamine anywhere
      in this project, and tracing discards would punish the odours the fly
      discarded on and push it toward the degenerate always-discard policy;
    * the trace resets at every blind boundary, so losing the Boss Blind cannot
      punish the plays that cleared the Small Blind;
    * the losing play is hit twice when it was unrewarded, once by the immediate
      rule and once here. It both failed its share and lost the blind.
    """

    def __init__(self, n_kc: int, gamma: float) -> None:
        if not (0.0 <= float(gamma) <= 1.0):
            raise ValueError(f"gamma must lie in [0, 1], got {gamma!r}")
        self.n_kc = int(n_kc)
        self.gamma = float(gamma)
        self.trace = np.zeros(self.n_kc, np.float32)
        #: ``(weight, bucket_name)`` of every play still represented, newest last,
        #: so the report can attribute a terminal pulse's weight to buckets.
        self.contributions: List[Tuple[float, str]] = []
        self.n_updates = 0

    def reset(self) -> None:
        self.trace.fill(0.0)
        self.contributions.clear()

    def update(self, elig: NDArray[np.float32], bucket_name: str = "") -> None:
        e = np.asarray(elig, np.float32)
        if e.shape != (self.n_kc,):
            raise ValueError(f"expected {self.n_kc} eligibilities, got {e.shape}")
        self.trace *= np.float32(self.gamma)
        self.trace += e
        np.clip(self.trace, 0.0, 1.0, out=self.trace)
        self.contributions = [(w * self.gamma, b) for w, b in self.contributions]
        self.contributions.append((1.0, str(bucket_name)))
        self.n_updates += 1

    def weight_by_bucket(self) -> Dict[str, float]:
        """Recency weight currently carried by each bucket's plays.

        Attribution only: the delivered depression is the clipped vector, so
        these weights sum to at least what is delivered. Reported as
        ``punish_trace_weight`` and labelled as an upper bound.
        """
        out: Dict[str, float] = {}
        for w, b in self.contributions:
            out[b] = out.get(b, 0.0) + float(w)
        return out

    def sum_trace(self) -> float:
        return float(self.trace.sum())

    def describe(self) -> dict:
        return dict(gamma=self.gamma, n_kc=self.n_kc, n_updates=self.n_updates,
                    play_only=True, reset_per_blind=True, clip=[0.0, 1.0])


class TraceRunner:
    """Per-record bookkeeping for the trace: reset at a blind, update, fire on loss.

    Separated from the training loop so the three rules the preregistration fixes
    (reset at every blind boundary, PLAY decisions only, one extra punishment
    pulse **iff** the record carries ``lost``) are testable without a brain and
    without a game. ``plast`` need only provide ``n_kc``, ``eligibility`` and
    ``deliver_eligibility``.
    """

    def __init__(self, plast, gamma: float, eta_punish: float) -> None:
        self.plast = plast
        self.eta_punish = float(eta_punish)
        self.trace = EligibilityTrace(int(plast.n_kc), gamma)
        self.blind: Optional[tuple] = None
        self.n_pulses = 0
        self.weight_delivered = 0.0
        self.weight_by_bucket: Dict[str, float] = {}
        self.pulses_by_bucket: Dict[str, int] = {}

    def on_record(self, record: dict, kc_counts) -> dict:
        key = (int(record["seed"]), int(record["round"]))
        if key != self.blind:
            self.trace.reset()
            self.blind = key
        if kc_counts is None:
            return {}
        if record["action"] == K.PLAY:
            self.trace.update(self.plast.eligibility(kc_counts),
                              str(record.get("bucket_name", "")))
        if not record["lost"]:
            return {}
        if self.eta_punish <= 0.0 or not self.trace.contributions:
            return {}
        info = self.plast.deliver_eligibility("punish", self.trace.trace,
                                              self.eta_punish)
        by_bucket = self.trace.weight_by_bucket()
        for b, w in by_bucket.items():
            self.weight_by_bucket[b] = self.weight_by_bucket.get(b, 0.0) + float(w)
            self.pulses_by_bucket[b] = self.pulses_by_bucket.get(b, 0) + 1
        self.n_pulses += 1
        self.weight_delivered += self.trace.sum_trace()
        out = dict(trace_pulse=True,
                   trace_sum=round(self.trace.sum_trace(), 3),
                   trace_by_bucket={b: round(w, 4) for b, w in by_bucket.items()},
                   trace_mean_abs_dw=round(float(info["mean_abs_dw"]), 6),
                   trace_n_changed=int(info["n_changed"]))
        self.trace.reset()
        return out

    def describe(self) -> dict:
        d = self.trace.describe()
        d.update(eta_punish=self.eta_punish, n_pulses=self.n_pulses,
                 weight_delivered=round(self.weight_delivered, 2))
        return d


# --------------------------------------------------------------------------- #
# conditions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    """One reward specification. Everything else is v3's ``omission/separated``."""

    name: str
    reward_rule: str
    gamma: Optional[float]   # None = no eligibility trace
    label: str

    @property
    def has_trace(self) -> bool:
        return self.gamma is not None


def conditions() -> List[Condition]:
    """The preregistered set: A, B, C x 3 gammas, D. No additions after the fact."""
    out = [
        Condition("A", REWARD_SHARE, None,
                  "baseline: v3 omission/separated, share reward"),
        Condition("B", REWARD_PACE, None,
                  "pace reward: cumulative schedule, no re-baselining"),
    ]
    for g in TRACE_GAMMAS:
        out.append(Condition(f"C{int(round(g * 100)):02d}", REWARD_SHARE, g,
                             f"share reward + eligibility trace, gamma={g:g}"))
    out.append(Condition("D", REWARD_PACE, TRACE_GAMMA_D,
                         f"pace reward + eligibility trace, gamma={TRACE_GAMMA_D:g}"))
    return out


#: The three *novel* families the decision rule is Holm-corrected across. The C
#: sweep is one family, represented by its prespecified central gamma; the other
#: two gammas are sensitivity analyses and cannot promote anything.
NOVEL_FAMILIES: Tuple[str, ...] = ("B", f"C{int(round(TRACE_GAMMA_D * 100)):02d}", "D")


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = [float(x) for x in pvalues]
    n = len(p)
    order = sorted(range(n), key=lambda i: p[i])
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mcnemar_exact(a: Sequence[float], b: Sequence[float]) -> dict:
    """Exact two-sided McNemar on paired 0/1 outcomes ``a`` against ``b``.

    ``b01`` is "b cleared, a did not" and ``b10`` the reverse; the exact test is
    the two-sided binomial on ``b10`` out of ``b01 + b10`` at p = 0.5.
    """
    from math import comb

    x = np.asarray(a, np.int64)
    y = np.asarray(b, np.int64)
    if x.shape != y.shape:
        raise ValueError(f"paired vectors must align: {x.shape} vs {y.shape}")
    b10 = int(((x == 1) & (y == 0)).sum())
    b01 = int(((x == 0) & (y == 1)).sum())
    n = b10 + b01
    if n == 0:
        return dict(b10=0, b01=0, n_discordant=0, p_exact=1.0)
    k = min(b10, b01)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return dict(b10=b10, b01=b01, n_discordant=n,
                p_exact=float(min(1.0, 2.0 * tail)))


# --------------------------------------------------------------------------- #
# the game loop
# --------------------------------------------------------------------------- #
def play_game4(env: BalatroEnv, seed: int, policy: K.Policy,
               reward_rule: str = REWARD_SHARE,
               on_outcome: Optional[Callable[[dict, Optional[NDArray[np.int32]]], dict]] = None,
               fly=None, max_hands: int = 60) -> K.GameResult:
    """One ante-1 run, recording the blind-level quantities the pace rule needs.

    Structurally identical to :func:`plast_common.play_game`; the additions are
    ``required``, ``score_before``, ``score_after``, ``p0`` and both reward
    branches in every record. ``p0`` is the blind's starting play count, read
    from the game at the blind's first decision rather than assumed to be 4.
    """
    env.reset(seed=seed)
    records: List[dict] = []
    p0_by_round: Dict[int, int] = {}
    while not env.done and len(records) < max_hands:
        state = env.engine.state
        stage = int(state.stage.int())
        mask = env.action_mask()
        if stage == K.STAGE_PRE_BLIND:
            env.step(SELECT_BLIND_INDEX if mask[SELECT_BLIND_INDEX]
                     else int(np.flatnonzero(mask)[0]))
            continue
        if stage == K.STAGE_POST_BLIND:
            env.step(CASH_OUT_INDEX if mask[CASH_OUT_INDEX]
                     else int(np.flatnonzero(mask)[0]))
            continue
        if stage == K.STAGE_SHOP:
            env.step(NEXT_ROUND_INDEX if mask[NEXT_ROUND_INDEX]
                     else int(np.flatnonzero(mask)[0]))
            continue
        if stage not in K.STAGE_BLINDS:
            env.step(int(np.flatnonzero(mask)[0]))
            continue
        ctx = K.hand_context(state)
        if ctx is None:
            env.step(int(np.flatnonzero(mask)[0]))
            continue

        required = int(state.required_score)
        score_before = int(state.score)
        p0 = p0_by_round.setdefault(int(ctx.round), int(ctx.plays))

        action, extra = policy(ctx)
        if action == K.DISCARD and not ctx.discard_ok:
            action = K.PLAY
        target = ctx.best_slots if action == K.PLAY else ctx.dig
        act_index = PLAY_INDEX if action == K.PLAY else DISCARD_INDEX

        for slot in target:
            mask = env.action_mask()
            if slot < len(mask) and mask[slot] == 1:
                env.step(int(slot))
        mask = env.action_mask()
        if mask[act_index] != 1:
            other = DISCARD_INDEX if act_index == PLAY_INDEX else PLAY_INDEX
            if mask[other] == 1:
                act_index = other
                action = K.PLAY if other == PLAY_INDEX else K.DISCARD
            else:
                env.step(int(np.flatnonzero(mask)[0]))
                continue
        _f, _m, _r, done, info = env.step(act_index)

        out = resolve_outcome4(required, score_before, ctx.needed, ctx.plays, p0,
                               action, info, done, reward_rule)
        chips = out["chips_gained"]
        record = dict(
            seed=int(seed), round=ctx.round, hand_in_game=len(records),
            best_type=int(ctx.best_type), best_type_name=ctx.best_type_name,
            best_score=int(ctx.best_score), bucket=int(ctx.bucket),
            bucket_name=ctx.bucket_name, needed=int(ctx.needed),
            plays_before=int(ctx.plays), discards_before=int(ctx.discards),
            discard_ok=bool(ctx.discard_ok), bits_on=int(ctx.bits.sum()),
            odor=int(K._odor_key(ctx.bits)),
            action=action, chips_gained=chips, cleared=bool(out["cleared"]),
            lost=bool(out["lost"]), reward=out["reward"], punish=out["punish"],
            # v4 additions
            required=required, score_before=score_before,
            score_after=score_before + chips, p0=int(p0),
            reward_share=out["reward_share"], reward_pace=out["reward_pace"],
        )
        record.update(extra)
        if on_outcome is not None:
            kcc = fly.last_kc_counts if fly is not None else None
            record.update(on_outcome(record, kcc))
        records.append(record)

    return K.GameResult(
        seed=int(seed), hands=records, cleared=bool(env.is_win),
        chips=int(env.chips_total), rounds=int(env.engine.state.round),
        truncated=bool(env.steps >= 500),
    )


def evaluate_policy4(policy, seeds: Sequence[int], env=None,
                     reward_rule: str = REWARD_SHARE) -> List[K.GameResult]:
    e = env if env is not None else K.make_env()
    return [play_game4(e, int(s), policy, reward_rule=reward_rule) for s in seeds]


def dopamine_for4(record: dict) -> Optional[str]:
    """The immediate pulse, ``omission`` rule, on v4's ``reward`` field.

    Identical in effect to ``plast3_common.dopamine_for(record, "omission")``;
    spelled out here because v4's ``reward`` may come from either branch.
    """
    if record["action"] != K.PLAY:
        return None
    return "reward" if record["reward"] else "punish"


def write_json(path: Path, payload: dict) -> Path:
    return K.write_json(path, payload)


def rel_path(p: Path) -> str:
    return C3.rel_path(p)
