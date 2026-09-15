"""Thin, correct wrapper around ``pylatro.GameEngine``.

The shipped gym wrapper in ``vendor/balatro-rs/pylatro/gym/env.py`` is
self-described as broken (it ignores the seed, silently swallows illegal
actions, and replaces the chip reward with the win bonus instead of adding
to it). This module does not use it.

Action space
------------

``pylatro`` exposes a fixed length-109 legality mask. The index -> meaning
mapping is **fixed** for a given ``Config``: the vector is laid out as
concatenated per-action-kind sub-vectors whose lengths come only from
``Config`` (``available_max``, ``store_consumable_slots_max``,
``consumable_slots``, ``joker_slots``) plus two compile-time constants
(2 pack slots, 5 pack contents). Nothing about the *runtime* state changes
the layout, only which entries are unmasked. With the engine's default
config the layout is 109 wide and is what :data:`ACTION_NAMES` describes.

Card-relative indices (select / move) address a *slot position* in
``state.available``, not a card identity, so index 3 means "the card
currently sitting in hand slot 3" and its meaning tracks the hand contents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pylatro
from numpy.typing import NDArray

from .features import N_FEATURES, StateLike, encode

__all__ = [
    "BalatroEnv",
    "IllegalActionError",
    "RewardConfig",
    "ACTION_NAMES",
    "N_ACTIONS",
    "StepInfo",
]

N_ACTIONS: int = 109

# Sub-vector sizes for the engine's default Config. Asserted against the
# real mask length in BalatroEnv.__init__, because two of these
# (store_consumable_slots_max, and the pack constants) are not reachable
# from Python and so cannot be derived at runtime.
_AVAILABLE_MAX: int = 24
_SHOP_CARD_SLOTS: int = 4  # store_consumable_slots_max
_CONSUMABLE_SLOTS: int = 2
_JOKER_SLOTS: int = 5
_PACK_SLOTS: int = 2
_PACK_CONTENTS_MAX: int = 5


def _build_action_names() -> List[str]:
    names: List[str] = []
    for i in range(_AVAILABLE_MAX):
        names.append(f"select_card[{i}]")
    # Leftmost card cannot move left, so the block is shifted by one slot.
    for i in range(1, _AVAILABLE_MAX):
        names.append(f"move_card_left[{i}]")
    # Rightmost card cannot move right.
    for i in range(_AVAILABLE_MAX - 1):
        names.append(f"move_card_right[{i}]")
    names.append("play")
    names.append("discard")
    names.append("cash_out")
    for i in range(_SHOP_CARD_SLOTS):
        names.append(f"buy_joker[{i}]")
    names.append("next_round")
    names.append("select_blind")
    names.append("skip_blind")
    for i in range(_CONSUMABLE_SLOTS):
        names.append(f"buy_consumable[{i}]")
    names.append("buy_voucher")
    for i in range(_SHOP_CARD_SLOTS):
        names.append(f"buy_playing_card[{i}]")
    for i in range(_CONSUMABLE_SLOTS):
        names.append(f"use_consumable[{i}]")
    names.append("apply_tarot")
    for i in range(_JOKER_SLOTS):
        names.append(f"sell_joker[{i}]")
    for i in range(_CONSUMABLE_SLOTS):
        names.append(f"sell_consumable[{i}]")
    for i in range(_PACK_SLOTS):
        names.append(f"buy_pack[{i}]")
    for i in range(_PACK_CONTENTS_MAX):
        names.append(f"pick_pack_card[{i}]")
    names.append("skip_pack")
    names.append("sort_hand[rank]")
    names.append("sort_hand[suit]")
    names.append("reroll")
    names.append("apply_spectral")
    return names


ACTION_NAMES: List[str] = _build_action_names()

if len(ACTION_NAMES) != N_ACTIONS:  # pragma: no cover - structural invariant
    raise AssertionError(
        f"ACTION_NAMES has {len(ACTION_NAMES)} entries, expected {N_ACTIONS}"
    )

# Indices that are pure no-ops as far as scoring goes: reordering the hand
# and sorting it never change what a hand is worth in this engine. They are
# always legal during a blind, so a uniform-random policy spends most of its
# budget on them.
_MOVE_LEFT_START: int = _AVAILABLE_MAX
_MOVE_RIGHT_END: int = _AVAILABLE_MAX + 2 * (_AVAILABLE_MAX - 1)
SORT_HAND_INDICES: Tuple[int, ...] = (105, 106)
NOOP_ACTION_INDICES: Tuple[int, ...] = tuple(
    range(_MOVE_LEFT_START, _MOVE_RIGHT_END)
) + SORT_HAND_INDICES

PLAY_INDEX: int = 70
DISCARD_INDEX: int = 71
CASH_OUT_INDEX: int = 72
NEXT_ROUND_INDEX: int = 77
SELECT_BLIND_INDEX: int = 78
SKIP_BLIND_INDEX: int = 79


class IllegalActionError(ValueError):
    """Raised when ``step`` is given an index the legality mask forbids.

    Callers are expected to sample only from the mask returned by ``reset``
    or the previous ``step``. This is deliberately loud rather than a
    negative-reward no-op, so a masking bug in a policy cannot quietly
    degrade into a worse-performing agent.
    """


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping knobs.

    Attributes:
        chip_scale: Chips gained on a step are multiplied by this. The
            default of 1/100 keeps per-step rewards near unit scale.
        win_bonus: Added on the terminal step when the run is won. Added to,
            not substituted for, that step's chip reward.
        lose_penalty: Added on the terminal step when the run is lost.
        step_penalty: Added on every step. Negative values discourage
            burning turns on no-op reordering.
    """

    chip_scale: float = 1.0 / 100.0
    win_bonus: float = 20.0
    lose_penalty: float = 0.0
    step_penalty: float = 0.0


@dataclass(frozen=True)
class StepInfo:
    """Structured contents of the ``info`` dict returned by ``step``."""

    chips_gained: int
    chips_total: int
    score: int
    required_score: int
    stage: int
    plays: int
    discards: int
    money: int
    round: int
    n_jokers: int
    steps: int
    is_win: bool
    truncated: bool
    action_name: str

    def as_dict(self) -> Dict[str, Union[int, bool, str]]:
        return {
            "chips_gained": self.chips_gained,
            "chips_total": self.chips_total,
            "score": self.score,
            "required_score": self.required_score,
            "stage": self.stage,
            "plays": self.plays,
            "discards": self.discards,
            "money": self.money,
            "round": self.round,
            "n_jokers": self.n_jokers,
            "steps": self.steps,
            "is_win": self.is_win,
            "truncated": self.truncated,
            "action_name": self.action_name,
        }


class BalatroEnv:
    """A single-run Balatro environment over ``pylatro.GameEngine``.

    ``reset`` returns ``(features, mask)``; ``step`` returns
    ``(features, mask, reward, done, info)``. Illegal actions raise
    :class:`IllegalActionError`; the caller must filter with the mask.
    """

    n_actions: int = N_ACTIONS
    action_names: List[str] = ACTION_NAMES

    def __init__(
        self,
        ante_end: int = 1,
        reward: Optional[RewardConfig] = None,
        max_steps: int = 500,
        mask_noop_actions: bool = False,
        encoder: Optional[Callable[[StateLike], NDArray[np.float32]]] = None,
        n_features: Optional[int] = None,
    ) -> None:
        """
        Args:
            ante_end: Ante the run ends at. ``1`` (the default) means the run
                is won by clearing the Small, Big and Boss blinds of ante 1.
            reward: Reward shaping. Defaults to ``RewardConfig()``.
            max_steps: Hard cap on actions per episode. Hitting it ends the
                episode with ``done=True`` and ``info["truncated"]=True``.
            mask_noop_actions: When true, zero out the move-card and
                sort-hand entries in the returned mask. They are always legal
                during a blind and never change a hand's value, so they are
                noise; masking them is opt-in because it changes the task.
            encoder: State -> feature vector. Defaults to
                :func:`flybalatro.features.encode` (the 283-bit v1 encoding);
                pass :func:`flybalatro.features_v2.encode_v2` for the 315-bit
                v2 encoding. Injected rather than selected here so ``env.py``
                does not have to know about every encoding, and so ``step``
                still fetches ``engine.state`` exactly once (that getter clones
                the whole game and is the most expensive call in the loop).
            n_features: Length of ``encoder``'s output. Inferred from the first
                encode when omitted.
        """
        if ante_end < 1 or ante_end > 8:
            raise ValueError(f"ante_end must be in 1..8, got {ante_end}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")

        self._ante_end: int = ante_end
        self._reward_config: RewardConfig = reward if reward is not None else RewardConfig()
        self._max_steps: int = max_steps
        self._mask_noop: bool = mask_noop_actions
        self._encode: Callable[[StateLike], NDArray[np.float32]] = (
            encode if encoder is None else encoder
        )
        self.n_features: int = (
            int(n_features) if n_features is not None
            else (N_FEATURES if encoder is None else -1)
        )

        self._engine: Optional[pylatro.GameEngine] = None
        self._prev_score: int = 0
        self._chips_total: int = 0
        self._steps: int = 0
        self._done: bool = True
        self._last_action_name: str = ""

        # Layout sanity: the hardcoded ACTION_NAMES are only correct if the
        # engine's config produces a 109-wide vector.
        probe = pylatro.GameEngine(self._make_config(None))
        probe_len = len(probe.gen_action_space())
        if probe_len != N_ACTIONS:
            raise RuntimeError(
                f"engine action space is {probe_len} wide, expected {N_ACTIONS}; "
                "ACTION_NAMES no longer describes this build"
            )

    # -- construction helpers -------------------------------------------

    def _make_config(self, seed: Optional[int]) -> pylatro.Config:
        config = pylatro.Config()
        config.ante_end = self._ante_end
        if seed is not None:
            config.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        return config

    # -- public API ------------------------------------------------------

    @property
    def engine(self) -> pylatro.GameEngine:
        """The underlying engine. Raises if ``reset`` has not been called."""
        if self._engine is None:
            raise RuntimeError("reset() must be called before using the env")
        return self._engine

    @property
    def done(self) -> bool:
        return self._done

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def chips_total(self) -> int:
        """Sum of positive score deltas over the episode so far."""
        return self._chips_total

    @property
    def is_win(self) -> bool:
        return bool(self.engine.is_win)

    def reset(self, seed: Optional[int] = None) -> Tuple[NDArray[np.float32], NDArray[np.int8]]:
        """Start a fresh run. Returns ``(features, mask)``."""
        self._engine = pylatro.GameEngine(self._make_config(seed))
        self._prev_score = 0
        self._chips_total = 0
        self._steps = 0
        self._done = False
        self._last_action_name = ""
        state = self._engine.state
        features = self._encode(state)
        if self.n_features < 0:
            self.n_features = int(features.shape[0])
        return features, self.action_mask()

    def action_mask(self) -> NDArray[np.int8]:
        """Length-109 0/1 legality mask for the current state."""
        mask = np.asarray(self.engine.gen_action_space(), dtype=np.int8)
        if self._done:
            # A finished run has nothing legal; the engine already returns an
            # all-zero vector in End stage, but be explicit about it.
            mask = np.zeros(N_ACTIONS, dtype=np.int8)
        elif self._mask_noop:
            mask = mask.copy()
            for idx in NOOP_ACTION_INDICES:
                mask[idx] = 0
        return mask

    def legal_actions(self) -> NDArray[np.int64]:
        """Indices of currently legal actions."""
        return np.flatnonzero(self.action_mask()).astype(np.int64)

    def step(
        self, action_index: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.int8], float, bool, Dict[str, Union[int, bool, str]]]:
        """Take one action. Returns ``(features, mask, reward, done, info)``.

        Raises:
            IllegalActionError: if ``action_index`` is out of range or masked.
            RuntimeError: if the episode is already finished.
        """
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        index = int(action_index)
        if not 0 <= index < N_ACTIONS:
            raise IllegalActionError(
                f"action index {index} out of range 0..{N_ACTIONS - 1}"
            )
        mask = self.action_mask()
        if mask[index] == 0:
            legal = np.flatnonzero(mask).tolist()
            raise IllegalActionError(
                f"action {index} ({ACTION_NAMES[index]}) is masked; "
                f"legal indices are {legal}"
            )

        self._last_action_name = ACTION_NAMES[index]
        self.engine.handle_action_index(index)
        self._steps += 1

        # One .state fetch per step: the getter clones the whole Game
        # (action_history included), so it is the most expensive call here.
        state = self.engine.state
        score = int(state.score)
        # score resets to 0 when the next blind is dealt, so only count gains.
        chips_gained = score - self._prev_score
        if chips_gained < 0:
            chips_gained = 0
        self._prev_score = score
        self._chips_total += chips_gained

        terminated = bool(self.engine.is_over)
        truncated = (not terminated) and self._steps >= self._max_steps
        self._done = terminated or truncated

        cfg = self._reward_config
        reward = chips_gained * cfg.chip_scale + cfg.step_penalty
        is_win = bool(self.engine.is_win)
        if terminated:
            reward += cfg.win_bonus if is_win else cfg.lose_penalty

        features = self._encode(state)
        info = StepInfo(
            chips_gained=chips_gained,
            chips_total=self._chips_total,
            score=score,
            required_score=int(state.required_score),
            stage=int(state.stage.int()),
            plays=int(state.plays),
            discards=int(state.discards),
            money=int(state.money),
            round=int(state.round),
            n_jokers=len(state.jokers),
            steps=self._steps,
            is_win=is_win,
            truncated=truncated,
            action_name=self._last_action_name,
        ).as_dict()
        return features, self.action_mask(), float(reward), self._done, info
