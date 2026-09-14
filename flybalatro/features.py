"""Binary feature encoding of a Balatro game state.

The output of :func:`encode` is a fixed-length ``float32`` vector whose every
element is exactly ``0.0`` or ``1.0``. It is meant to be injected into the
fly connectome as olfactory receptor neuron (ORN) activity, one bit per
input channel, so the encoding is deliberately one-hot / sparse rather than
scalar-valued: no feature ever carries magnitude information in its value,
only in *which* bit is set.

Layout (283 bits total)
-----------------------

===== =========  ===========================================================
Start Length     Block
===== =========  ===========================================================
0     144        ``hand`` - 8 slots of ``available``, 18 bits each:
                 rank one-hot (13) + suit one-hot (4) + occupied (1)
144   8          ``hand_selected`` - one bit per hand slot, set when that
                 card is currently selected
152   90         ``selected`` - 5 slots of ``selected`` (``selected_max``),
                 18 bits each, same rank/suit/occupied layout. Redundant
                 with the two blocks above by construction, but gives the
                 readout a position-invariant view of the hand about to be
                 played at a fixed set of channels.
242   5          ``plays`` remaining one-hot: 0, 1, 2, 3, >=4
247   5          ``discards`` remaining one-hot: 0, 1, 2, 3, >=4
252   8          ``score / required_score`` ratio buckets
260   6          ``money`` buckets: 0, 1-2, 3-5, 6-9, 10-19, >=20
266   11         ``stage`` one-hot over ``Stage.int()`` (0..10)
277   6          joker count one-hot: 0, 1, 2, 3, 4, >=5
===== =========  ===========================================================

Notes
-----
* ``available`` is truncated to ``HAND_SLOTS`` (8). The engine's
  ``hand_size()`` is 8 by default but the Paint Brush / Palette vouchers
  raise it (up to ``available_max`` = 24), so a run that buys those would
  have cards beyond slot 7 invisible to the encoding.
* ``selected`` is truncated to ``SELECTED_SLOTS`` (5), which equals the
  engine's ``selected_max``, so no truncation happens in practice.
* The top bin of every count block is ">=", because vouchers can push
  plays / discards / joker slots past their config defaults.
* Rank index is 0..12 for Two..Ace; suit index is 0..3 for Spade, Club,
  Heart, Diamond. Both come straight from the ``Card.rank_index`` /
  ``Card.suit_index`` getters added to the Rust bindings.
"""

from __future__ import annotations

from typing import List, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "encode",
    "FEATURE_NAMES",
    "N_FEATURES",
    "HAND_SLOTS",
    "SELECTED_SLOTS",
    "N_RANKS",
    "N_SUITS",
    "RANK_NAMES",
    "SUIT_NAMES",
]

N_RANKS: int = 13
N_SUITS: int = 4
HAND_SLOTS: int = 8
SELECTED_SLOTS: int = 5
N_STAGES: int = 11
PLAY_BINS: int = 5
DISCARD_BINS: int = 5
RATIO_BINS: int = 8
MONEY_BINS: int = 6
JOKER_BINS: int = 6

RANK_NAMES: Tuple[str, ...] = (
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "T",
    "J",
    "Q",
    "K",
    "A",
)
SUIT_NAMES: Tuple[str, ...] = ("s", "c", "h", "d")

_STAGE_NAMES: Tuple[str, ...] = (
    "pre_blind",
    "blind_small",
    "blind_big",
    "blind_boss",
    "post_blind",
    "shop",
    "end_win",
    "end_lose",
    "tarot_hand",
    "spectral_hand",
    "pack_open",
)

# Upper edges of the score/required_score ratio buckets. A ratio of exactly
# zero always lands in bucket 0; anything at or above the last edge lands in
# the final bucket.
_RATIO_EDGES: Tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5)

# Inclusive upper bounds of the money buckets; the final bucket is open.
_MONEY_EDGES: Tuple[int, ...] = (0, 2, 5, 9, 19)

_CARD_BITS: int = N_RANKS + N_SUITS + 1

_OFF_HAND: int = 0
_OFF_HAND_SELECTED: int = _OFF_HAND + HAND_SLOTS * _CARD_BITS
_OFF_SELECTED: int = _OFF_HAND_SELECTED + HAND_SLOTS
_OFF_PLAYS: int = _OFF_SELECTED + SELECTED_SLOTS * _CARD_BITS
_OFF_DISCARDS: int = _OFF_PLAYS + PLAY_BINS
_OFF_RATIO: int = _OFF_DISCARDS + DISCARD_BINS
_OFF_MONEY: int = _OFF_RATIO + RATIO_BINS
_OFF_STAGE: int = _OFF_MONEY + MONEY_BINS
_OFF_JOKERS: int = _OFF_STAGE + N_STAGES
N_FEATURES: int = _OFF_JOKERS + JOKER_BINS


class _CardLike(Protocol):
    """The subset of ``pylatro.Card`` this module reads."""

    @property
    def rank_index(self) -> int: ...

    @property
    def suit_index(self) -> int: ...

    @property
    def id(self) -> int: ...


class _StageLike(Protocol):
    def int(self) -> int: ...


class StateLike(Protocol):
    """The subset of ``pylatro.GameState`` this module reads."""

    @property
    def available(self) -> Sequence[_CardLike]: ...

    @property
    def selected(self) -> Sequence[_CardLike]: ...

    @property
    def plays(self) -> int: ...

    @property
    def discards(self) -> int: ...

    @property
    def score(self) -> int: ...

    @property
    def required_score(self) -> int: ...

    @property
    def money(self) -> int: ...

    @property
    def stage(self) -> _StageLike: ...

    @property
    def jokers(self) -> Sequence[object]: ...


def _build_feature_names() -> List[str]:
    names: List[str] = []
    for slot in range(HAND_SLOTS):
        for rank in RANK_NAMES:
            names.append(f"hand{slot}_rank_{rank}")
        for suit in SUIT_NAMES:
            names.append(f"hand{slot}_suit_{suit}")
        names.append(f"hand{slot}_occupied")
    for slot in range(HAND_SLOTS):
        names.append(f"hand{slot}_selected")
    for slot in range(SELECTED_SLOTS):
        for rank in RANK_NAMES:
            names.append(f"sel{slot}_rank_{rank}")
        for suit in SUIT_NAMES:
            names.append(f"sel{slot}_suit_{suit}")
        names.append(f"sel{slot}_occupied")
    for i in range(PLAY_BINS):
        suffix = f"ge{i}" if i == PLAY_BINS - 1 else str(i)
        names.append(f"plays_{suffix}")
    for i in range(DISCARD_BINS):
        suffix = f"ge{i}" if i == DISCARD_BINS - 1 else str(i)
        names.append(f"discards_{suffix}")
    ratio_labels = (
        "zero",
        "lt0.125",
        "lt0.25",
        "lt0.5",
        "lt0.75",
        "lt1.0",
        "lt1.5",
        "ge1.5",
    )
    for label in ratio_labels:
        names.append(f"score_ratio_{label}")
    money_labels = ("0", "1_2", "3_5", "6_9", "10_19", "ge20")
    for label in money_labels:
        names.append(f"money_{label}")
    for label in _STAGE_NAMES:
        names.append(f"stage_{label}")
    for i in range(JOKER_BINS):
        suffix = f"ge{i}" if i == JOKER_BINS - 1 else str(i)
        names.append(f"jokers_{suffix}")
    return names


FEATURE_NAMES: List[str] = _build_feature_names()

if len(FEATURE_NAMES) != N_FEATURES:  # pragma: no cover - structural invariant
    raise AssertionError(
        f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries, expected {N_FEATURES}"
    )


def _bucket(value: int, edges: Sequence[int]) -> int:
    """Index of the first edge ``value`` does not exceed, else ``len(edges)``."""
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def _ratio_bucket(score: int, required: int) -> int:
    if score <= 0:
        return 0
    if required <= 0:
        return RATIO_BINS - 1
    ratio = score / required
    for i, edge in enumerate(_RATIO_EDGES):
        if ratio < edge:
            return i + 1
    return RATIO_BINS - 1


def _write_card_slot(out: NDArray[np.float32], base: int, card: _CardLike) -> None:
    rank = card.rank_index
    suit = card.suit_index
    if not 0 <= rank < N_RANKS:
        raise ValueError(f"rank_index out of range: {rank}")
    if not 0 <= suit < N_SUITS:
        raise ValueError(f"suit_index out of range: {suit}")
    out[base + rank] = 1.0
    out[base + N_RANKS + suit] = 1.0
    out[base + N_RANKS + N_SUITS] = 1.0


def encode(state: StateLike) -> NDArray[np.float32]:
    """Encode a ``pylatro.GameState`` as a length-283 binary vector.

    Every element is 0.0 or 1.0. See the module docstring for the layout.
    """
    out: NDArray[np.float32] = np.zeros(N_FEATURES, dtype=np.float32)

    available = state.available
    selected = state.selected
    selected_ids = frozenset(card.id for card in selected)

    for slot, card in enumerate(available):
        if slot >= HAND_SLOTS:
            break
        _write_card_slot(out, _OFF_HAND + slot * _CARD_BITS, card)
        if card.id in selected_ids:
            out[_OFF_HAND_SELECTED + slot] = 1.0

    for slot, card in enumerate(selected):
        if slot >= SELECTED_SLOTS:
            break
        _write_card_slot(out, _OFF_SELECTED + slot * _CARD_BITS, card)

    out[_OFF_PLAYS + min(state.plays, PLAY_BINS - 1)] = 1.0
    out[_OFF_DISCARDS + min(state.discards, DISCARD_BINS - 1)] = 1.0
    out[_OFF_RATIO + _ratio_bucket(state.score, state.required_score)] = 1.0
    out[_OFF_MONEY + _bucket(state.money, _MONEY_EDGES)] = 1.0

    stage_int = state.stage.int()
    if not 0 <= stage_int < N_STAGES:
        raise ValueError(f"stage int out of range: {stage_int}")
    out[_OFF_STAGE + stage_int] = 1.0

    out[_OFF_JOKERS + min(len(state.jokers), JOKER_BINS - 1)] = 1.0

    return out
