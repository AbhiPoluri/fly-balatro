"""Hand-analysis tests against hand-constructed cases.

``hands.py`` is the thing the v2 encoder relays through the fly, so it has to
agree with ``vendor/balatro-rs`` rather than with poker intuition: only the
cards that make the hand score chips, and the "best subset" is chosen by score,
not by hand-type rank.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Sequence

import pytest

from flybalatro import hands
from flybalatro.hands import (
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    analyse,
    best_subset,
    classify_slots,
)

RANKS = {r: i for i, r in enumerate("2 3 4 5 6 7 8 9 T J Q K A".split())}
SUITS = {s: i for i, s in enumerate("s c h d".split())}


@dataclass(frozen=True)
class Card:
    """Minimal stand-in for ``pylatro.Card``."""

    rank_index: int
    suit_index: int
    id: int = 0


def hand(spec: str) -> List[Card]:
    """``"Ah Kh Qh Jh Th"`` -> cards, ids assigned by slot."""
    out: List[Card] = []
    for i, token in enumerate(spec.split()):
        out.append(Card(RANKS[token[0]], SUITS[token[1]], id=i))
    return out


def flags(n: int, selected: Sequence[int]) -> List[bool]:
    return [i in set(selected) for i in range(n)]


# -- classification ---------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("Ah", HIGH_CARD),
        ("Ah Kd", HIGH_CARD),
        ("Ah Ad", PAIR),
        ("Ah Ad Kh Kd", TWO_PAIR),
        ("Ah Ad Ac", THREE_OF_A_KIND),
        ("Ah Ad Ac As", FOUR_OF_A_KIND),
        ("5h 6d 7c 8s 9h", STRAIGHT),
        ("2h 7h 9h Jh Kh", FLUSH),
        ("Ah Ad Ac Kh Kd", FULL_HOUSE),
        ("5h 6h 7h 8h 9h", STRAIGHT_FLUSH),
        # Four of a kind beats the full house reading of the same five cards.
        ("Ah Ad Ac As Kh", FOUR_OF_A_KIND),
        # A pair with three kickers is still just a pair.
        ("Ah Ad 2c 3s 4h", PAIR),
    ],
)
def test_classify_types(spec: str, expected: int) -> None:
    cards = hand(spec)
    subset = classify_slots(cards, range(len(cards)))
    assert subset is not None
    assert subset.hand_type == expected


def test_ace_low_straight() -> None:
    cards = hand("Ah 2d 3c 4s 5h")
    subset = classify_slots(cards, range(5))
    assert subset is not None
    assert subset.hand_type == STRAIGHT
    # 30 base chips + (11 + 2 + 3 + 4 + 5), x4 mult.
    assert subset.score == (30 + 25) * 4


def test_ace_low_straight_flush() -> None:
    cards = hand("Ah 2h 3h 4h 5h")
    subset = classify_slots(cards, range(5))
    assert subset is not None
    assert subset.hand_type == STRAIGHT_FLUSH
    assert subset.score == (100 + 25) * 8


def test_ace_king_queen_jack_two_is_not_a_straight() -> None:
    cards = hand("Ah Kd Qc Js 2h")
    subset = classify_slots(cards, range(5))
    assert subset is not None
    assert subset.hand_type == HIGH_CARD


def test_full_house_beats_flush_within_one_five_card_set() -> None:
    """Precedence: five cards that are both a full house and a flush.

    A same-suit full house is a Flush House in real Balatro, which this module
    deliberately ignores (it needs enhancements to matter); the engine's own
    ordering puts Full House above Flush, and so does this.
    """
    cards = hand("Ah Ad Ac Kh Kd")  # full house, not a flush
    fh = classify_slots(cards, range(5))
    assert fh is not None and fh.hand_type == FULL_HOUSE

    same_suit = hand("Ah Ah Ah Kh Kh")  # impossible deal, tests precedence only
    made = classify_slots(same_suit, range(5))
    assert made is not None
    assert made.hand_type == FULL_HOUSE  # not FLUSH


def test_flush_can_outscore_a_full_house_across_subsets() -> None:
    """``best_subset`` ranks by score, not by hand-type tier.

    A high flush is worth (35 + 51) * 4 = 344; the low full house available in
    the same 8 cards is worth (40 + 12) * 4 = 208, so the flush is the best
    subset even though Full House is the higher type. This is the engine's
    arithmetic, not a bug in the ordering.
    """
    cards = hand("Ah Kh Qh Jh Th 2s 2c 2d")
    flush_score = (35 + 11 + 10 + 10 + 10 + 10) * 4
    trips_score = (30 + 2 + 2 + 2) * 3
    best = best_subset(cards)
    assert best is not None
    assert best.hand_type == STRAIGHT_FLUSH  # royal flush scores as a straight flush
    assert best.score == (100 + 51) * 8
    assert flush_score == 344
    assert trips_score == 108


def test_full_house_wins_when_it_outscores_the_alternatives() -> None:
    cards = hand("Kh Kd Kc Qh Qd 2s 3s 4s")
    best = best_subset(cards)
    assert best is not None
    assert best.hand_type == FULL_HOUSE
    assert best.score == (40 + 10 + 10 + 10 + 10 + 10) * 4
    assert best.slots == (0, 1, 2, 3, 4)


# -- scoring rule -----------------------------------------------------------


def test_kickers_do_not_add_chips() -> None:
    """The engine scores ``MadeHand.hand``, not every played card."""
    cards = hand("Ah Ad 2c 3s 4h")
    bare = classify_slots(cards, (0, 1))
    padded = classify_slots(cards, (0, 1, 2, 3, 4))
    assert bare is not None and padded is not None
    assert bare.score == padded.score == (10 + 11 + 11) * 2


def test_ace_low_straight_is_found_among_eight_cards() -> None:
    """A 2 3 4 5 (220) beats both 2-6 (200) and the pair of aces (44)."""
    cards = hand("Ah Ad 2c 3s 4h 5d 6c 7h")
    best = best_subset(cards)
    assert best is not None
    assert best.hand_type == STRAIGHT
    assert best.slots == (0, 2, 3, 4, 5)  # Ah 2c 3s 4h 5d
    assert best.score == (30 + 11 + 2 + 3 + 4 + 5) * 4 == 220
    six_high = classify_slots(cards, (2, 3, 4, 5, 6))
    assert six_high is not None and six_high.score == 200


def test_tie_break_is_fewest_cards_then_lowest_slots() -> None:
    cards = hand("Ah Ad 2c 3s")
    best = best_subset(cards)
    assert best is not None
    assert best.slots == (0, 1)  # the bare pair, not pair + kickers


# -- the optimality invariant ----------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "Ah Kh Qh Jh Th 2s 2c 2d",
        "2h 3d 4c 5s 6h 7d 8c 9s",
        "Kh Kd Kc Ks Qh Qd Qc 2s",
        "Ah 2h 3h 4h 5h Ad Ac As",
        "7h 7d 2c 9s Jh Qd 4c 5s",
        "Th Jh Qh Kh Ah Td Jd Qd",
    ],
)
def test_best_subset_scores_at_least_every_other_subset(spec: str) -> None:
    cards = hand(spec)
    best = best_subset(cards)
    assert best is not None
    for size in range(1, 6):
        for slots in itertools.combinations(range(len(cards)), size):
            other = classify_slots(cards, slots)
            assert other is not None
            assert best.score >= other.score, f"{slots} scores {other.score}"


def test_forced_inclusion_is_respected_and_never_better() -> None:
    cards = hand("Ah Kh Qh Jh Th 2s 2c 2d")
    free = best_subset(cards)
    forced = best_subset(cards, must_include=(5,))
    assert free is not None and forced is not None
    assert 5 in forced.slots
    assert forced.score <= free.score


def test_forced_inclusion_rejects_impossible_requests() -> None:
    cards = hand("Ah Kh Qh Jh Th")
    assert best_subset(cards, must_include=(0, 1, 2, 3, 4, 5)) is None
    assert best_subset(cards, must_include=(9,)) is None


# -- analyse ----------------------------------------------------------------


def test_analyse_reports_selection_state() -> None:
    cards = hand("Ah Ad 2c 3s 4h 5d 6c 8h")
    best = best_subset(cards)
    assert best is not None

    empty = analyse(cards, flags(8, ()))
    assert empty.n_selected == 0
    assert empty.selected_type is None
    assert empty.selected_is_best is False
    assert empty.best_slots == best.slots
    assert empty.best_mask == best.mask

    exact = analyse(cards, flags(8, best.slots))
    assert exact.selected_is_best is True
    assert exact.selected_type == best.hand_type
    assert exact.selected_score == best.score

    partial = analyse(cards, flags(8, best.slots[:1]))
    assert partial.selected_is_best is False
    assert partial.n_selected == 1


def test_analyse_with_no_cards() -> None:
    result = analyse([], [])
    assert result.n_available == 0
    assert result.best_type == -1
    assert result.best_slots == ()
    assert result.selected_type is None
    assert result.has_cards is False
    assert result.best_mask == (0,) * hands.HAND_SLOTS


def test_analyse_ignores_cards_past_slot_seven() -> None:
    cards = hand("2s 2c 3s 3c 4s 4c 5s 5c Ah Ad")
    result = analyse(cards, flags(10, (8, 9)))
    assert result.n_available == 8
    assert result.n_selected == 0  # the aces are outside the encoded slots
    assert all(slot < 8 for slot in result.best_slots)


def test_hand_type_name_round_trip() -> None:
    assert hands.hand_type_name(None) == "none"
    assert hands.hand_type_name(-1) == "none"
    assert hands.hand_type_name(FLUSH) == "Flush"
    assert len(hands.HAND_TYPE_NAMES) == hands.N_HAND_TYPES == 9


def test_selected_flags_from_state_matches_ids() -> None:
    class State:
        available = hand("Ah Ad 2c 3s 4h")
        selected = [available[1], available[3]]

    assert hands.selected_flags_from_state(State()) == (False, True, False, True, False)
    result = hands.analyse_state(State())
    assert result.n_selected == 2
    assert result.selected_slots == (1, 3)
