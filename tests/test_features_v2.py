"""v2 encoding tests: the relay block, and that v1 is carried through untouched.

Covers the three places the hand-type relay has to line up:

* the block's own bits against a hand whose best subset is known by hand;
* the 283 v1 bits sitting unchanged behind it (so a v1 claim about the encoding
  is still true of v2);
* the real-game adapter's state objects, which must feed ``hands.analyse`` the
  same ``rank_index`` / ``suit_index`` / ``id`` fields ``pylatro`` does.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pytest

from flybalatro import hands
from flybalatro.encode import feature_map_for
from flybalatro.env import BalatroEnv, SELECT_BLIND_INDEX
from flybalatro.features import FEATURE_NAMES, N_FEATURES, encode
from flybalatro.features_v2 import (
    FEATURE_NAMES_V2,
    N_FEATURES_V2,
    N_HAND_BLOCK,
    OFF_BEST_MASK,
    OFF_BEST_TYPE,
    OFF_IS_BEST,
    OFF_SCORE_BUCKET,
    OFF_SELECTED_TYPE,
    encode_v2,
    encoder_for_version,
    hand_block_summary,
    n_features_for_version,
)
from tests.test_hands import Card, hand  # reuse the hand builders


class FakeStage:
    def __init__(self, value: int) -> None:
        self.value = value

    def int(self) -> int:
        return self.value


class FakeState:
    """The subset of a game state both encoders read."""

    def __init__(self, available: Sequence[Card], selected: Sequence[int] = (),
                 stage: int = 1, score: int = 0, required_score: int = 300,
                 plays: int = 4, discards: int = 3, money: int = 4) -> None:
        self.available = list(available)
        self.selected = [self.available[i] for i in selected]
        self.stage = FakeStage(stage)
        self.score = score
        self.required_score = required_score
        self.plays = plays
        self.discards = discards
        self.money = money
        self.jokers: List[object] = []


# -- layout -----------------------------------------------------------------


def test_layout_constants() -> None:
    assert N_HAND_BLOCK == 32
    assert N_FEATURES_V2 == N_HAND_BLOCK + N_FEATURES == 315
    assert len(FEATURE_NAMES_V2) == N_FEATURES_V2
    assert FEATURE_NAMES_V2[N_HAND_BLOCK:] == FEATURE_NAMES
    assert n_features_for_version(1) == N_FEATURES
    assert n_features_for_version(2) == N_FEATURES_V2
    assert encoder_for_version(1) is encode
    assert encoder_for_version(2) is encode_v2
    with pytest.raises(ValueError):
        encoder_for_version(3)


def test_v1_bits_are_carried_through_unchanged() -> None:
    env = BalatroEnv(ante_end=1, mask_noop_actions=True)
    env.reset(seed=7)
    env.step(SELECT_BLIND_INDEX)
    for _ in range(3):
        state = env.engine.state
        assert np.array_equal(encode_v2(state)[N_HAND_BLOCK:], encode(state))
        env.step(int(np.flatnonzero(env.action_mask())[0]))


def test_every_bit_is_zero_or_one() -> None:
    state = FakeState(hand("Ah Kh Qh Jh Th 2s 2c 2d"), selected=(0, 1))
    bits = encode_v2(state)
    assert bits.dtype == np.float32
    assert set(np.unique(bits).tolist()) <= {0.0, 1.0}


# -- the relay block --------------------------------------------------------


def test_relay_block_encodes_the_best_subset() -> None:
    cards = hand("Kh Kd Kc Qh Qd 2s 3s 4s")  # full house, slots 0-4
    state = FakeState(cards)
    bits = encode_v2(state)
    best = hands.best_subset(cards)
    assert best is not None and best.hand_type == hands.FULL_HOUSE

    assert bits[OFF_BEST_TYPE + hands.FULL_HOUSE] == 1.0
    assert bits[OFF_BEST_TYPE:OFF_BEST_TYPE + 9].sum() == 1.0
    assert np.array_equal(
        bits[OFF_BEST_MASK:OFF_BEST_MASK + 8],
        np.array(best.mask, dtype=np.float32),
    )
    # Nothing selected -> the "none" slot of the selected-type one-hot.
    assert bits[OFF_SELECTED_TYPE + hands.N_HAND_TYPES] == 1.0
    assert bits[OFF_IS_BEST] == 0.0


def test_selected_type_and_is_best_bit_follow_the_selection() -> None:
    cards = hand("Kh Kd Kc Qh Qd 2s 3s 4s")
    best = hands.best_subset(cards)
    assert best is not None

    partial = encode_v2(FakeState(cards, selected=(0, 1)))
    assert partial[OFF_SELECTED_TYPE + hands.PAIR] == 1.0
    assert partial[OFF_IS_BEST] == 0.0

    complete = encode_v2(FakeState(cards, selected=best.slots))
    assert complete[OFF_SELECTED_TYPE + hands.FULL_HOUSE] == 1.0
    assert complete[OFF_IS_BEST] == 1.0

    junk = encode_v2(FakeState(cards, selected=(5, 6)))
    assert junk[OFF_SELECTED_TYPE + hands.HIGH_CARD] == 1.0
    assert junk[OFF_IS_BEST] == 0.0


@pytest.mark.parametrize(
    "score,required,bucket",
    [(0, 3000, 0), (0, 800, 1), (0, 400, 2), (0, 300, 3), (300, 300, 3)],
)
def test_score_vs_needed_buckets(score: int, required: int, bucket: int) -> None:
    """Pair of kings scores (10 + 10 + 10) * 2 = 60... check against the real value."""
    cards = hand("Kh Kd 2s 3s 4s 5s 6h 8h")
    best = hands.best_subset(cards)
    assert best is not None
    bits = encode_v2(FakeState(cards, score=score, required_score=required))
    onehot = bits[OFF_SCORE_BUCKET:OFF_SCORE_BUCKET + 4]
    assert onehot.sum() == 1.0
    needed = max(0, required - score)
    ratio = best.score / needed if needed > 0 else float("inf")
    expected = 3 if needed <= 0 else (0 if ratio < 0.25 else 1 if ratio < 0.5 else 2 if ratio < 1.0 else 3)
    assert int(np.argmax(onehot)) == expected


def test_relay_block_is_silent_outside_a_blind() -> None:
    cards = hand("Kh Kd Kc Qh Qd")
    for stage in (0, 4, 5, 6, 7):
        bits = encode_v2(FakeState(cards, stage=stage))
        assert bits[:N_HAND_BLOCK].sum() == 0.0, f"stage {stage} drives the relay block"
    assert encode_v2(FakeState([], stage=1))[:N_HAND_BLOCK].sum() == 0.0


def test_hand_block_summary_is_labelled_and_complete() -> None:
    cards = hand("Ah Kh Qh Jh Th 2s 2c 2d")
    summary = hand_block_summary(FakeState(cards, selected=(0, 1, 2, 3, 4)))
    assert summary["active"] is True
    assert summary["best_type"] == "Straight Flush"
    assert summary["selected_is_best"] is True
    assert summary["needed"] == 300
    # best type + 5 best slots + selected type + is_best + score bucket
    assert summary["n_bits_on"] == 1 + 5 + 1 + 1 + 1
    idle = hand_block_summary(FakeState([], stage=5))
    assert idle["active"] is False
    assert idle["best_type"] == "none"


# -- the feature map --------------------------------------------------------


def test_v2_feature_map_puts_the_relay_block_on_real_orns() -> None:
    """``orn_first`` is what makes "the block is first" mean anything."""

    # 3,000 ORNs for 3,150 driven neurons, so the map has to overflow into the
    # other sensory neurons exactly as it does on the real graph (2,639 ORNs).
    class TinyGraph:
        n = 4000

        def orn_indices(self):
            return np.arange(0, 3000, dtype=np.int32)

        def sensory_indices(self):
            return np.arange(0, 4000, dtype=np.int32)

        type = np.array(["x"] * 4000, dtype=object)

    g = TinyGraph()
    fm = feature_map_for(g, 2)
    assert fm.n_features == N_FEATURES_V2
    assert fm.orn_first is True
    orn = set(g.orn_indices().tolist())
    relay = set(fm.groups[:N_HAND_BLOCK].ravel().tolist())
    assert fm.extended_with_sensory is True
    assert relay <= orn, "the relay block must land on real receptor neurons"
    # The overflow lands at the tail instead, among the v1 bits.
    assert not set(fm.groups[-1].tolist()) & orn

    v1 = feature_map_for(g, 1)
    assert v1.orn_first is False
    assert v1.n_features == N_FEATURES


# -- the real-game adapter --------------------------------------------------


def test_adapter_states_encode_and_analyse() -> None:
    from tests.test_realgame_adapter import card, payload
    from flybalatro.realgame.adapter import build_state

    cards = [card(i, r, s) for i, (r, s) in enumerate(
        [("K", "H"), ("K", "D"), ("K", "C"), ("Q", "H"), ("Q", "D"),
         ("2", "S"), ("3", "S"), ("4", "S")])]
    state = build_state(payload(hand=cards))
    analysis = hands.analyse_state(state)
    assert analysis.best_type == hands.FULL_HOUSE
    assert analysis.best_slots == (0, 1, 2, 3, 4)

    bits = encode_v2(state)
    assert bits.shape == (N_FEATURES_V2,)
    assert bits[OFF_BEST_TYPE + hands.FULL_HOUSE] == 1.0
    assert np.array_equal(bits[N_HAND_BLOCK:], encode(state))

    chosen = build_state(payload(hand=cards), selected_indices=(0, 1, 2, 3, 4))
    assert encode_v2(chosen)[OFF_IS_BEST] == 1.0
    summary = hand_block_summary(chosen)
    assert summary["best_type"] == "Full House"
    assert summary["selected_is_best"] is True

    shop = build_state(payload(hand=[], state="SHOP"))
    assert encode_v2(shop)[:N_HAND_BLOCK].sum() == 0.0
