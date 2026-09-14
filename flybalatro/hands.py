"""Poker-hand analysis of a Balatro hand. **Computed outside the brain.**

Nothing in this module is biological. It is ordinary Python that enumerates
every subset of size 1..5 of the (up to 8) dealt cards, classifies each with
Balatro's base hand types, scores it at hand level 1, and reports the best one.
The point of the v2 experiment is to *relay* this analysis through the fly:
:mod:`flybalatro.features_v2` turns the result into extra binary channels that
become tonic current on olfactory receptor neurons, and the readout then has to
recover it from spike counts. The fly does not compute any of it.

Faithfulness to the engine
--------------------------

Classification order and the scoring rule follow ``vendor/balatro-rs``
(``core/src/hand.rs``, ``core/src/game.rs``) rather than a generic poker
library, because that is the engine the agent is scored by:

* Precedence is Straight Flush > Four of a Kind > Full House > Flush >
  Straight > Three of a Kind > Two Pair > Pair > High Card. A flush and a
  straight need exactly 5 cards; a full house needs 5; two pair needs 4.
* Ace-low straights (A 2 3 4 5) count; ace is rank index 12 and is the only
  rank allowed to wrap.
* **Only the cards that make the hand contribute chips.** ``calc_score_inner``
  iterates ``MadeHand.hand`` (the scoring subset), not ``MadeHand.all``, so a
  pair plus three kickers scores exactly the same as the bare pair. Card chips
  are 2..10 for Two..Ten, 10 for J/Q/K, 11 for Ace.
* Score at level 1 is ``(base_chips + sum(chips of the scoring cards)) *
  base_mult`` with the level-1 table from ``balatro-types/src/planet.rs``.
* Enhancement-only hands (Five of a Kind, Flush House, Flush Five) and the
  Royal Flush label are ignored: they are unreachable from a standard deck
  with no jokers or enhancements, which is the ante-1 setting used here.

Because kickers are worth nothing, ties are broken towards **fewer cards**,
then towards the lowest slot indices, so "the best subset" is the canonical
minimal one and the teacher issues as few select actions as possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Protocol, Sequence, Tuple

__all__ = [
    "HAND_TYPE_NAMES",
    "N_HAND_TYPES",
    "HAND_SLOTS",
    "MAX_SELECTED",
    "RANK_CHIPS",
    "LEVEL1_BASE",
    "Subset",
    "HandAnalysis",
    "analyse",
    "best_subset",
    "classify_slots",
    "hand_type_name",
    "analyse_state",
    "selected_flags_from_state",
]

#: Index -> name, in the engine's ``HandRank`` order restricted to the nine
#: base types reachable from a standard deck.
HAND_TYPE_NAMES: Tuple[str, ...] = (
    "High Card",
    "Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
)
N_HAND_TYPES: int = len(HAND_TYPE_NAMES)

HIGH_CARD: int = 0
PAIR: int = 1
TWO_PAIR: int = 2
THREE_OF_A_KIND: int = 3
STRAIGHT: int = 4
FLUSH: int = 5
FULL_HOUSE: int = 6
FOUR_OF_A_KIND: int = 7
STRAIGHT_FLUSH: int = 8

#: ``(chips, mult)`` at hand level 1, from ``Planetarium::new``.
LEVEL1_BASE: Tuple[Tuple[int, int], ...] = (
    (5, 1),    # High Card
    (10, 2),   # Pair
    (20, 2),   # Two Pair
    (30, 3),   # Three of a Kind
    (30, 4),   # Straight
    (35, 4),   # Flush
    (40, 4),   # Full House
    (60, 7),   # Four of a Kind
    (100, 8),  # Straight Flush
)

#: Chip value of a card by ``rank_index`` (0 = Two .. 12 = Ace).
RANK_CHIPS: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11)

#: Hand slots the encoding covers, matching ``features.HAND_SLOTS``.
HAND_SLOTS: int = 8

#: ``selected_max`` in the engine's default config.
MAX_SELECTED: int = 5

_ACE: int = 12
_TWO: int = 0


class CardLike(Protocol):
    """The two attributes this module reads off a card."""

    @property
    def rank_index(self) -> int: ...

    @property
    def suit_index(self) -> int: ...


@dataclass(frozen=True)
class Subset:
    """One candidate group of hand slots, classified and scored."""

    slots: Tuple[int, ...]
    hand_type: int
    score: int

    @property
    def hand_type_name(self) -> str:
        return HAND_TYPE_NAMES[self.hand_type]

    @property
    def mask(self) -> Tuple[int, ...]:
        """0/1 over :data:`HAND_SLOTS` slots."""
        out = [0] * HAND_SLOTS
        for slot in self.slots:
            if 0 <= slot < HAND_SLOTS:
                out[slot] = 1
        return tuple(out)


@dataclass(frozen=True)
class HandAnalysis:
    """What the hand-type relay knows about one game state's hand.

    ``best_*`` describes the highest-scoring playable subset of the dealt
    cards; ``selected_*`` describes what the current selection would score if
    played right now. ``selected_is_best`` is the single bit that tells the
    teacher (and, after the relay, the readout) "stop selecting and play".
    """

    n_available: int
    n_selected: int
    best_type: int
    best_slots: Tuple[int, ...]
    best_score: int
    selected_type: Optional[int]
    selected_slots: Tuple[int, ...]
    selected_score: int
    selected_is_best: bool

    @property
    def best_mask(self) -> Tuple[int, ...]:
        out = [0] * HAND_SLOTS
        for slot in self.best_slots:
            if 0 <= slot < HAND_SLOTS:
                out[slot] = 1
        return tuple(out)

    @property
    def best_type_name(self) -> str:
        return hand_type_name(self.best_type)

    @property
    def selected_type_name(self) -> str:
        return hand_type_name(self.selected_type)

    @property
    def has_cards(self) -> bool:
        return self.n_available > 0


def hand_type_name(index: Optional[int]) -> str:
    """``"none"`` for ``None`` or an out-of-range index."""
    if index is None or not 0 <= index < N_HAND_TYPES:
        return "none"
    return HAND_TYPE_NAMES[index]


# -- classification ---------------------------------------------------------


def _is_straight(ranks: Sequence[int]) -> bool:
    """Five distinct consecutive ranks, ace high or ace low."""
    if len(ranks) != 5:
        return False
    ordered = sorted(ranks)
    if all(ordered[i + 1] - ordered[i] == 1 for i in range(4)):
        return True
    # A 2 3 4 5: ace sorts last, drop it and check the rest.
    if ordered[4] == _ACE and ordered[0] == _TWO:
        return all(ordered[i + 1] - ordered[i] == 1 for i in range(3))
    return False


def _classify(cards: Tuple[Tuple[int, int], ...]) -> Tuple[int, Tuple[int, ...]]:
    """``(hand_type, positions that score)`` for 1..5 ``(rank, suit)`` cards.

    ``positions`` index into ``cards``. Mirrors ``SelectHand::best_hand`` and
    the per-rank ``cards_of_values`` scoring subsets.
    """
    n = len(cards)
    if n == 0 or n > MAX_SELECTED:
        raise ValueError(f"a played hand has 1..{MAX_SELECTED} cards, got {n}")

    by_rank: Dict[int, List[int]] = {}
    for i, (rank, _suit) in enumerate(cards):
        by_rank.setdefault(rank, []).append(i)
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]
    every = tuple(range(n))

    # Highest rank first, which is the order values_freq() iterates in.
    descending = sorted(by_rank.items(), key=lambda kv: -kv[0])

    def first_with(count: int, exclude: Optional[int] = None) -> Optional[int]:
        for rank, positions in descending:
            if rank != exclude and len(positions) >= count:
                return rank
        return None

    flush = n == 5 and len(set(suits)) == 1
    straight = _is_straight(ranks)

    if flush and straight:
        return STRAIGHT_FLUSH, every

    four = first_with(4)
    if four is not None:
        return FOUR_OF_A_KIND, tuple(by_rank[four])

    three = first_with(3)
    if n >= 5 and three is not None:
        pair = first_with(2, exclude=three)
        if pair is not None:
            return FULL_HOUSE, tuple(sorted(by_rank[three] + by_rank[pair]))

    if flush:
        return FLUSH, every
    if straight:
        return STRAIGHT, every
    if three is not None:
        return THREE_OF_A_KIND, tuple(by_rank[three])

    top_pair = first_with(2)
    if top_pair is not None:
        if n >= 4:
            second = first_with(2, exclude=top_pair)
            if second is not None:
                return TWO_PAIR, tuple(sorted(by_rank[top_pair] + by_rank[second]))
        return PAIR, tuple(by_rank[top_pair])

    highest = descending[0][0]
    return HIGH_CARD, tuple(by_rank[highest])


def _score(hand_type: int, cards: Tuple[Tuple[int, int], ...],
           scoring: Tuple[int, ...]) -> int:
    base_chips, base_mult = LEVEL1_BASE[hand_type]
    chips = base_chips
    for i in scoring:
        rank = cards[i][0]
        if not 0 <= rank < len(RANK_CHIPS):
            raise ValueError(f"rank_index out of range: {rank}")
        chips += RANK_CHIPS[rank]
    return chips * base_mult


# -- enumeration ------------------------------------------------------------


def _sort_key(subset: Subset) -> Tuple[int, int, Tuple[int, ...]]:
    """Highest score, then fewest cards, then lowest slots."""
    return (-subset.score, len(subset.slots), subset.slots)


# One entry is 218 Subset objects, ~35 kB, so the cache is deliberately small:
# a miss costs ~1 ms, which is nothing next to a 100 ms brain window, and four
# eval workers each holding a fat cache is not nothing next to 16 GB.
@lru_cache(maxsize=1024)
def _enumerate(cards: Tuple[Tuple[int, int], ...]) -> Tuple[Subset, ...]:
    """Every subset of size 1..5, classified and scored. 218 of them for 8 cards."""
    out: List[Subset] = []
    n = len(cards)
    for size in range(1, min(MAX_SELECTED, n) + 1):
        for slots in combinations(range(n), size):
            chosen = tuple(cards[i] for i in slots)
            hand_type, scoring = _classify(chosen)
            out.append(Subset(slots, hand_type, _score(hand_type, chosen, scoring)))
    return tuple(out)


@lru_cache(maxsize=4096)
def _best(cards: Tuple[Tuple[int, int], ...],
          must_include: FrozenSet[int]) -> Optional[Subset]:
    candidates = [
        s for s in _enumerate(cards) if must_include.issubset(s.slots)
    ]
    if not candidates:
        return None
    return min(candidates, key=_sort_key)


def _card_key(cards: Sequence[CardLike]) -> Tuple[Tuple[int, int], ...]:
    return tuple(
        (int(c.rank_index), int(c.suit_index)) for c in cards[:HAND_SLOTS]
    )


# -- public API -------------------------------------------------------------


def best_subset(
    available_cards: Sequence[CardLike], must_include: Sequence[int] = ()
) -> Optional[Subset]:
    """Highest-scoring playable subset, optionally forced to contain slots.

    Returns ``None`` when there are no cards, or when ``must_include`` cannot
    be part of any playable subset (more than 5 slots, or a slot outside the
    hand).
    """
    cards = _card_key(available_cards)
    if not cards:
        return None
    forced = frozenset(int(i) for i in must_include)
    if len(forced) > MAX_SELECTED or any(not 0 <= i < len(cards) for i in forced):
        return None
    return _best(cards, forced)


def classify_slots(
    available_cards: Sequence[CardLike], slots: Sequence[int]
) -> Optional[Subset]:
    """Classify an explicit group of slots, or ``None`` if it is not playable."""
    cards = _card_key(available_cards)
    chosen = sorted({int(i) for i in slots})
    if not chosen or len(chosen) > MAX_SELECTED:
        return None
    if any(not 0 <= i < len(cards) for i in chosen):
        return None
    picked = tuple(cards[i] for i in chosen)
    hand_type, scoring = _classify(picked)
    return Subset(tuple(chosen), hand_type, _score(hand_type, picked, scoring))


def analyse(
    available_cards: Sequence[CardLike], selected_flags: Sequence[bool]
) -> HandAnalysis:
    """Analyse one hand. ``selected_flags[i]`` is "slot ``i`` is selected".

    ``available_cards`` beyond slot 7 is ignored, matching the 8 hand slots the
    feature encoding covers. A state with no dealt cards (shop, cash-out) gives
    ``n_available == 0``, ``best_type == -1`` and no selection.
    """
    cards = _card_key(available_cards)
    selected = tuple(
        i for i in range(len(cards)) if i < len(selected_flags) and bool(selected_flags[i])
    )
    if not cards:
        return HandAnalysis(
            n_available=0, n_selected=0, best_type=-1, best_slots=(), best_score=0,
            selected_type=None, selected_slots=(), selected_score=0,
            selected_is_best=False,
        )

    best = _best(cards, frozenset())
    if best is None:  # pragma: no cover - unreachable with >=1 card
        raise RuntimeError("no playable subset for a non-empty hand")

    current = classify_slots(available_cards, selected) if selected else None
    return HandAnalysis(
        n_available=len(cards),
        n_selected=len(selected),
        best_type=best.hand_type,
        best_slots=best.slots,
        best_score=best.score,
        selected_type=current.hand_type if current is not None else None,
        selected_slots=selected,
        selected_score=current.score if current is not None else 0,
        selected_is_best=bool(selected == best.slots),
    )


def selected_flags_from_state(state: object) -> Tuple[bool, ...]:
    """``(available, selected)`` on a game state -> per-slot selected flags.

    Works for both ``pylatro.GameState`` (selection is a list of cards, matched
    by ``id``) and ``realgame.adapter.RealGameState`` (same shape).
    """
    available = getattr(state, "available")
    selected = getattr(state, "selected")
    selected_ids = frozenset(int(c.id) for c in selected)
    return tuple(
        int(c.id) in selected_ids for c in list(available)[:HAND_SLOTS]
    )


def analyse_state(state: object) -> HandAnalysis:
    """:func:`analyse` applied to a game state's hand and selection."""
    available = list(getattr(state, "available"))[:HAND_SLOTS]
    return analyse(available, selected_flags_from_state(state))
