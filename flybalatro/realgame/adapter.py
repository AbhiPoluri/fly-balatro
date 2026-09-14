"""Bridge a BalatroBot ``gamestate`` payload to the pylatro-shaped interfaces.

Three jobs:

1. :class:`RealGameState` - a duck-typed stand-in for ``pylatro.GameState``
   that :func:`flybalatro.features.encode` accepts unchanged.
2. :func:`legality_mask` - a length-109 mask over ``env.ACTION_NAMES`` with
   ``BalatroEnv(mask_noop_actions=True)`` semantics.
3. :func:`plan_action` - one of the 109 action indices to the BalatroBot
   call(s) that realise it.

What ``features.encode`` actually reads
--------------------------------------

Only these, despite the wider ``pylatro`` surface: ``available`` and
``selected`` (each card's ``rank_index``, ``suit_index`` and ``id``),
``plays``, ``discards``, ``score``, ``required_score``, ``money``,
``stage.int()`` and ``len(jokers)``. It does **not** read ``deck``,
``discarded``, ``enhancement``, ``round`` or ``ante``, so those are carried on
:class:`RealGameState` for logging only and never affect a bit.

Selection is virtual
--------------------

The BalatroBot API has no card-selection endpoint: ``play`` and ``discard``
take the hand indices to act on. ``pylatro``, by contrast, has the agent toggle
``select_card[i]`` and then issue a bare ``play``. To keep the action space and
the encoded bits identical across both, :class:`RealGameState` carries its own
``selected_indices`` - the fly's selection-in-progress - and
:func:`plan_action` submits it when the fly finally plays or discards. The
game's own ``card.state.highlight`` is used only to seed that selection, since
nothing in the API can set it.

Selecting is one-way. ``pylatro`` drops ``select_card[i]`` from the action
space once slot ``i`` is selected -- there is no deselect, and the selection is
only released by ``play`` or ``discard``. :func:`legality_mask` matches that
exactly. Offering a deselect instead gives the readout an action it never saw
in training, and the v3 readout takes it forever: select slot 0, deselect slot
0, repeat. See ``docs/REALGAME_INSTALL.md`` section 4.

Hand indices address *positions*, exactly as in ``env.py``: index 3 means
"whatever is in hand slot 3 right now". The real game shows the hand sorted
rank-descending; ``pylatro`` returns draw order. That is immaterial because the
encoding is positional either way -- replaying a hand under the sorted order,
an id-hash permutation and four random permutations gives the same decision
sequence shape -- but it does mean a given index means different cards in the
two engines for the same deal.

Stage mapping
-------------

``features`` wants ``Stage.int()`` in 0..10 over
``pre_blind, blind_small, blind_big, blind_boss, post_blind, shop, end_win,
end_lose, tarot_hand, spectral_hand, pack_open``. BalatroBot reports a
``State`` string plus which blind is current:

===========================  ==========================================
BalatroBot ``state``         ``Stage.int()``
===========================  ==========================================
``BLIND_SELECT``             0 ``pre_blind``
``SELECTING_HAND``,          1 / 2 / 3 by the current blind's ``type``
``HAND_PLAYED``,             (``SMALL`` / ``BIG`` / ``BOSS``)
``DRAW_TO_HAND``,
``NEW_ROUND``
``ROUND_EVAL``               4 ``post_blind``
``SHOP``                     5 ``shop``
``GAME_OVER``                6 ``end_win`` if ``won`` else 7 ``end_lose``
``PLAY_TAROT``               8 ``tarot_hand``
``*_PACK``,                  10 ``pack_open``
``SMODS_BOOSTER_OPENED``
===========================  ==========================================

``MENU``, ``TUTORIAL``, ``SPLASH``, ``SANDBOX``, ``DEMO_CTA`` and ``UNKNOWN``
describe no run stage, so they raise :class:`UnmappableStateError` rather than
being folded into a stage the fly would misread. The loop handles them before
encoding. Nothing maps to 9 (``spectral_hand``): BalatroBot exposes spectral
targeting through ``SPECTRAL_PACK``, which is a pack-open state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..env import N_ACTIONS
from ..features import SELECTED_SLOTS

__all__ = [
    "CardGeometry",
    "ScreenFrame",
    "RealCard",
    "RealStage",
    "RealGameState",
    "ActionPlan",
    "UnmappableStateError",
    "build_state",
    "legality_mask",
    "plan_action",
    "RANK_TO_INDEX",
    "SUIT_TO_INDEX",
    "STAGE_PRE_BLIND",
    "STAGE_BLIND_SMALL",
    "STAGE_BLIND_BIG",
    "STAGE_BLIND_BOSS",
    "STAGE_POST_BLIND",
    "STAGE_SHOP",
    "STAGE_END_WIN",
    "STAGE_END_LOSE",
    "STAGE_TAROT_HAND",
    "STAGE_PACK_OPEN",
    "BLIND_STAGES",
]

# -- feature-space constants (mirrors features.py's documented ordering) ----

#: ``rank_index`` is 0..12 for Two..Ace.
RANK_TO_INDEX: Mapping[str, int] = {
    "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7,
    "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12,
}

#: ``suit_index`` is 0..3 for Spade, Club, Heart, Diamond.
SUIT_TO_INDEX: Mapping[str, int] = {"S": 0, "C": 1, "H": 2, "D": 3}

STAGE_PRE_BLIND: int = 0
STAGE_BLIND_SMALL: int = 1
STAGE_BLIND_BIG: int = 2
STAGE_BLIND_BOSS: int = 3
STAGE_POST_BLIND: int = 4
STAGE_SHOP: int = 5
STAGE_END_WIN: int = 6
STAGE_END_LOSE: int = 7
STAGE_TAROT_HAND: int = 8
STAGE_SPECTRAL_HAND: int = 9
STAGE_PACK_OPEN: int = 10

BLIND_STAGES: Tuple[int, ...] = (STAGE_BLIND_SMALL, STAGE_BLIND_BIG, STAGE_BLIND_BOSS)

_BLIND_TYPE_TO_STAGE: Mapping[str, int] = {
    "SMALL": STAGE_BLIND_SMALL,
    "BIG": STAGE_BLIND_BIG,
    "BOSS": STAGE_BLIND_BOSS,
}

#: States that are "inside a blind", including its animation states.
_IN_BLIND_STATES: FrozenSet[str] = frozenset(
    {"SELECTING_HAND", "HAND_PLAYED", "DRAW_TO_HAND", "NEW_ROUND"}
)

#: States where a booster pack is open.
_PACK_STATES: FrozenSet[str] = frozenset(
    {
        "TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK", "STANDARD_PACK",
        "BUFFOON_PACK", "SMODS_BOOSTER_OPENED",
    }
)

#: States that describe no run stage at all.
_NON_RUN_STATES: FrozenSet[str] = frozenset(
    {"MENU", "TUTORIAL", "SPLASH", "SANDBOX", "DEMO_CTA", "UNKNOWN"}
)

#: Shop card sets that count as "joker" for ``buy_joker[i]``.
_JOKER_SETS: FrozenSet[str] = frozenset({"JOKER"})

#: Shop card sets that count as "consumable" for ``buy_consumable[i]``.
_CONSUMABLE_SETS: FrozenSet[str] = frozenset({"TAROT", "PLANET", "SPECTRAL"})

#: Shop card sets that count as a plain playing card.
_PLAYING_CARD_SETS: FrozenSet[str] = frozenset({"DEFAULT", "ENHANCED", "EDITION"})

# -- action-space offsets (recomputed from env.ACTION_NAMES' documented layout)

_AVAILABLE_MAX: int = 24
_SELECT_CARD_START: int = 0
_MOVE_START: int = 24
_MOVE_END: int = 70  # exclusive; 24..69 are move_card_left / move_card_right
PLAY_INDEX: int = 70
DISCARD_INDEX: int = 71
CASH_OUT_INDEX: int = 72
_BUY_JOKER_START: int = 73
_BUY_JOKER_SLOTS: int = 4
NEXT_ROUND_INDEX: int = 77
SELECT_BLIND_INDEX: int = 78
SKIP_BLIND_INDEX: int = 79
_BUY_CONSUMABLE_START: int = 80
_BUY_CONSUMABLE_SLOTS: int = 2
BUY_VOUCHER_INDEX: int = 82
_BUY_PLAYING_CARD_START: int = 83
_BUY_PLAYING_CARD_SLOTS: int = 4
_USE_CONSUMABLE_START: int = 87
_USE_CONSUMABLE_SLOTS: int = 2
APPLY_TAROT_INDEX: int = 89
_SELL_JOKER_START: int = 90
_SELL_JOKER_SLOTS: int = 5
_SELL_CONSUMABLE_START: int = 95
_SELL_CONSUMABLE_SLOTS: int = 2
_BUY_PACK_START: int = 97
_BUY_PACK_SLOTS: int = 2
_PICK_PACK_START: int = 99
_PICK_PACK_SLOTS: int = 5
SKIP_PACK_INDEX: int = 104
_SORT_HAND_START: int = 105
REROLL_INDEX: int = 107
APPLY_SPECTRAL_INDEX: int = 108


class UnmappableStateError(ValueError):
    """The game is in a state that does not correspond to any run stage.

    Raised for ``MENU``, ``TUTORIAL``, ``SPLASH``, ``SANDBOX``, ``DEMO_CTA``
    and ``UNKNOWN``. Callers should handle these (start a run, or stop) rather
    than encode them.
    """


# -- JSON narrowing helpers -------------------------------------------------
#
# The payload is decoded JSON, so every field has to be checked rather than
# assumed. These raise ValueError with the offending path instead of silently
# coercing, because a wrong type here becomes a wrong bit downstream.


def _obj(container: Mapping[str, object], key: str, path: str) -> Dict[str, object]:
    """Narrow ``container[key]`` to a dict, treating an empty list as ``{}``.

    Lua has one table type, so the mod's JSON encoder cannot tell an empty
    object from an empty array and emits ``[]`` for both. A base playing card
    in the real game arrives with ``"modifier": []`` and ``"state": []``, which
    mean "no enhancement/seal/edition" and "not debuffed, not highlighted".
    A *non-empty* list is still a type error, because that would be real data
    in a shape the mapping below cannot read.
    """
    value = container.get(key)
    if value is None:
        return {}
    if isinstance(value, list) and not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{key}: expected an object, got {type(value).__name__}")
    return value


def _int(container: Mapping[str, object], key: str, path: str, default: Optional[int] = None) -> int:
    value = container.get(key)
    if value is None:
        if default is None:
            raise ValueError(f"{path}.{key}: missing required integer")
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}.{key}: expected an integer, got {type(value).__name__}")
    return value


def _str(container: Mapping[str, object], key: str, path: str, default: Optional[str] = None) -> str:
    value = container.get(key)
    if value is None:
        if default is None:
            raise ValueError(f"{path}.{key}: missing required string")
        return default
    if not isinstance(value, str):
        raise ValueError(f"{path}.{key}: expected a string, got {type(value).__name__}")
    return value


def _bool(container: Mapping[str, object], key: str, default: bool = False) -> bool:
    value = container.get(key)
    return value if isinstance(value, bool) else default


def _card_list(area: Mapping[str, object], path: str) -> List[Dict[str, object]]:
    value = area.get("cards")
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path}.cards: expected an array, got {type(value).__name__}")
    out: List[Dict[str, object]] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}.cards[{i}]: expected an object")
        out.append(item)
    return out


# -- the duck-typed state ---------------------------------------------------


@dataclass(frozen=True)
class CardGeometry:
    """Where one card actually is on screen, straight from the game.

    Balatro is LOVE and every card is a ``Moveable`` carrying two transforms in
    game units: ``T`` is the target and ``VT`` is the *visible* one, which eases
    toward ``T`` over a few frames. A card mid-deal, mid-discard or being raised
    is drawn at ``VT``. The mod reports both plus the room offset and the
    tile scaling, and does the arithmetic from ``engine/node.lua`` for us:

        pixel = (VT.<x|y> + G.ROOM.T.<x|y>) * (G.TILESCALE * G.TILESIZE)

    so :attr:`x` / :attr:`y` / :attr:`w` / :attr:`h` are LOVE **pixels** and
    :attr:`r` is a rotation in radians about the rect's own centre. This is the
    exact quantity the overlay needs and it is animation-accurate by
    construction -- there is no model and nothing to drift.

    Absent from every payload produced before the mod carried it, and absent
    from any payload where the game's internals moved, in which case the
    overlay falls back to its fitted static model.
    """

    x: float
    y: float
    w: float
    h: float
    r: float = 0.0
    #: ``VT`` has not caught up with ``T``: this card is still moving.
    moving: bool = False
    #: ``VT.scale``, which LOVE applies about the rect's own centre when it
    #: draws the card. The transform above is the **unscaled** box, so a hand
    #: card whose scale is 0.95 is drawn 5% smaller than ``w`` x ``h`` says and
    #: an outline that ignored this would stand off the card's border all the
    #: way round. Defaults to 1.0, so a payload that does not report it -- every
    #: log written before this field existed -- is unchanged.
    scale: float = 1.0

    def drawn(self) -> "CardGeometry":
        """The rect as it is actually drawn: scaled about its own centre."""
        if self.scale == 1.0:
            return self
        w, h = self.w * self.scale, self.h * self.scale
        return CardGeometry(x=self.cx - w / 2.0, y=self.cy - h / 2.0, w=w, h=h,
                            r=self.r, moving=self.moving, scale=1.0)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @classmethod
    def from_json(cls, payload: object) -> Optional["CardGeometry"]:
        """Parse a card's ``geometry`` block; ``None`` if it is absent or thin.

        Deliberately total: a geometry field that is missing, malformed or
        degenerate returns ``None`` rather than raising, because the overlay has
        a working fallback and a cosmetic field must never break a game call.
        """
        if not isinstance(payload, Mapping):
            return None
        rect = payload.get("rect")
        if not isinstance(rect, Mapping):
            return None
        try:
            w = float(rect["w"])  # type: ignore[arg-type]
            h = float(rect["h"])  # type: ignore[arg-type]
            if not (w > 0.0 and h > 0.0):
                return None
            return cls(
                x=float(rect["x"]),    # type: ignore[arg-type]
                y=float(rect["y"]),    # type: ignore[arg-type]
                w=w, h=h,
                r=float(rect.get("r", 0.0) or 0.0),
                moving=bool(payload.get("moving", False)),
                scale=_scale_of(payload),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def as_dict(self) -> Dict[str, object]:
        return {"x": round(self.x, 2), "y": round(self.y, 2),
                "w": round(self.w, 2), "h": round(self.h, 2),
                "r": round(self.r, 5), "moving": bool(self.moving),
                "scale": round(self.scale, 4)}


def _scale_of(payload: Mapping) -> float:
    """``VT.scale`` out of a geometry block, defaulting to 1.0.

    Total on purpose: a missing, zero or nonsensical scale means "drawn at the
    size the transform says", which is what every consumer assumed before this
    field existed.
    """
    vt = payload.get("vt")
    if not isinstance(vt, Mapping):
        return 1.0
    try:
        value = float(vt.get("scale", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return value if 0.05 <= value <= 4.0 else 1.0


@dataclass(frozen=True)
class ScreenFrame:
    """The frame the :class:`CardGeometry` rects live in.

    ``width`` / ``height`` are LOVE's own drawing dimensions and
    ``pixel_width`` / ``pixel_height`` the backing-store size; on a Retina
    display they differ by ``dpi_scale``. The overlay positions an ``NSWindow``
    in **points**, so it needs the ratio to convert.
    """

    width: float = 0.0
    height: float = 0.0
    pixel_width: float = 0.0
    pixel_height: float = 0.0
    dpi_scale: float = 1.0
    unit: float = 0.0

    @classmethod
    def from_json(cls, payload: object) -> Optional["ScreenFrame"]:
        if not isinstance(payload, Mapping):
            return None
        def num(key: str, default: float = 0.0) -> float:
            try:
                return float(payload[key])  # type: ignore[arg-type]
            except (KeyError, TypeError, ValueError):
                return default
        width, height = num("width"), num("height")
        if not (width > 0.0 and height > 0.0):
            return None
        return cls(width=width, height=height,
                   pixel_width=num("pixel_width", width),
                   pixel_height=num("pixel_height", height),
                   dpi_scale=num("dpi_scale", 1.0) or 1.0,
                   unit=num("unit"))

    def as_dict(self) -> Dict[str, object]:
        return {"width": self.width, "height": self.height,
                "pixel_width": self.pixel_width, "pixel_height": self.pixel_height,
                "dpi_scale": self.dpi_scale, "unit": self.unit}


@dataclass(frozen=True)
class RealCard:
    """One card, exposing the three attributes ``features.encode`` reads.

    ``enhancement``, ``seal``, ``edition``, ``debuff`` and ``cost_*`` are
    carried for logging and for shop legality; the encoding ignores them.
    """

    id: int
    rank_index: int
    suit_index: int
    key: str = ""
    card_set: str = ""
    label: str = ""
    enhancement: Optional[str] = None
    seal: Optional[str] = None
    edition: Optional[str] = None
    debuff: bool = False
    highlight: bool = False
    cost_buy: int = 0
    cost_sell: int = 0
    #: Where the card is on screen right now, when the mod reports it.
    #: Never read by the encoding; carried for the overlay only.
    geometry: Optional[CardGeometry] = None

    @classmethod
    def from_json(cls, payload: Mapping[str, object], path: str) -> "RealCard":
        """Build from a BalatroBot ``Card``. Non-playing cards get rank/suit -1."""
        value = _obj(payload, "value", path)
        modifier = _obj(payload, "modifier", path)
        state = _obj(payload, "state", path)
        cost = _obj(payload, "cost", path)

        rank_index = -1
        suit_index = -1
        raw_rank = value.get("rank")
        raw_suit = value.get("suit")
        if isinstance(raw_rank, str) and isinstance(raw_suit, str):
            if raw_rank not in RANK_TO_INDEX:
                raise ValueError(f"{path}.value.rank: unknown rank {raw_rank!r}")
            if raw_suit not in SUIT_TO_INDEX:
                raise ValueError(f"{path}.value.suit: unknown suit {raw_suit!r}")
            rank_index = RANK_TO_INDEX[raw_rank]
            suit_index = SUIT_TO_INDEX[raw_suit]

        enhancement = modifier.get("enhancement")
        seal = modifier.get("seal")
        edition = modifier.get("edition")
        return cls(
            id=_int(payload, "id", path),
            rank_index=rank_index,
            suit_index=suit_index,
            key=_str(payload, "key", path, ""),
            card_set=_str(payload, "set", path, ""),
            label=_str(payload, "label", path, ""),
            enhancement=enhancement if isinstance(enhancement, str) else None,
            seal=seal if isinstance(seal, str) else None,
            edition=edition if isinstance(edition, str) else None,
            debuff=_bool(state, "debuff"),
            highlight=_bool(state, "highlight"),
            cost_buy=_int(cost, "buy", path, 0),
            cost_sell=_int(cost, "sell", path, 0),
            geometry=CardGeometry.from_json(payload.get("geometry")),
        )

    @property
    def is_playing_card(self) -> bool:
        return self.rank_index >= 0 and self.suit_index >= 0


@dataclass(frozen=True)
class RealStage:
    """Stands in for ``pylatro.Stage``: ``features`` only calls ``.int()``."""

    value: int

    def int(self) -> int:
        return self.value


@dataclass(frozen=True)
class RealGameState:
    """A ``pylatro.GameState`` look-alike built from a BalatroBot payload.

    ``available`` is the hand in the game's own display order.
    ``selected_indices`` is the fly's virtual selection (positions into
    ``available``, in the order the fly picked them); ``selected`` resolves it
    to cards, which is what ``features.encode`` reads.
    """

    available: Tuple[RealCard, ...]
    plays: int
    discards: int
    score: int
    required_score: int
    money: int
    stage: RealStage
    jokers: Tuple[RealCard, ...]
    consumables: Tuple[RealCard, ...]
    shop: Tuple[RealCard, ...]
    vouchers: Tuple[RealCard, ...]
    packs: Tuple[RealCard, ...]
    pack: Tuple[RealCard, ...]
    selected_indices: Tuple[int, ...] = ()
    # Context, not encoded.
    raw_state: str = ""
    round_num: int = 0
    ante_num: int = 0
    won: bool = False
    blind_type: str = ""
    blind_name: str = ""
    reroll_cost: int = 0
    selected_max: int = SELECTED_SLOTS
    joker_slots: int = 5
    consumable_slots: int = 2
    #: The LOVE frame ``RealCard.geometry`` rects live in. ``None`` on any
    #: payload from a mod build that does not report it.
    screen: Optional[ScreenFrame] = None

    @property
    def selected(self) -> Tuple[RealCard, ...]:
        """The virtually-selected cards, in selection order."""
        return tuple(
            self.available[i] for i in self.selected_indices if 0 <= i < len(self.available)
        )

    @property
    def round(self) -> int:
        """Named for parity with ``pylatro``; not read by ``features``."""
        return self.round_num

    @property
    def ante(self) -> int:
        return self.ante_num

    def with_selection(self, indices: Sequence[int]) -> "RealGameState":
        """Copy carrying a different virtual selection."""
        deduped: List[int] = []
        for i in indices:
            index = int(i)
            if 0 <= index < len(self.available) and index not in deduped:
                deduped.append(index)
        return replace(self, selected_indices=tuple(deduped))

    def toggle_selection(self, index: int) -> "RealGameState":
        """Copy with hand slot ``index`` added to, or removed from, the selection."""
        current = list(self.selected_indices)
        if index in current:
            current.remove(index)
        elif len(current) < self.selected_max:
            current.append(index)
        return self.with_selection(current)

    @property
    def shop_joker_indices(self) -> Tuple[int, ...]:
        return tuple(i for i, c in enumerate(self.shop) if c.card_set in _JOKER_SETS)

    @property
    def shop_consumable_indices(self) -> Tuple[int, ...]:
        return tuple(i for i, c in enumerate(self.shop) if c.card_set in _CONSUMABLE_SETS)

    @property
    def shop_playing_card_indices(self) -> Tuple[int, ...]:
        return tuple(i for i, c in enumerate(self.shop) if c.card_set in _PLAYING_CARD_SETS)


def _stage_for(raw_state: str, blind_type: str, won: bool) -> int:
    """Map a BalatroBot ``State`` (+ current blind) onto ``Stage.int()``."""
    if raw_state in _NON_RUN_STATES:
        raise UnmappableStateError(f"state {raw_state!r} is not a run stage")
    if raw_state == "BLIND_SELECT":
        return STAGE_PRE_BLIND
    if raw_state in _IN_BLIND_STATES:
        return _BLIND_TYPE_TO_STAGE.get(blind_type, STAGE_BLIND_SMALL)
    if raw_state == "ROUND_EVAL":
        return STAGE_POST_BLIND
    if raw_state == "SHOP":
        return STAGE_SHOP
    if raw_state == "GAME_OVER":
        return STAGE_END_WIN if won else STAGE_END_LOSE
    if raw_state == "PLAY_TAROT":
        return STAGE_TAROT_HAND
    if raw_state in _PACK_STATES:
        return STAGE_PACK_OPEN
    raise UnmappableStateError(f"state {raw_state!r} has no stage mapping")


def _current_blind(blinds: Mapping[str, object]) -> Tuple[str, str, int]:
    """Return ``(type, name, score)`` for the blind now in play or up next.

    Prefers ``status == "CURRENT"`` (inside a round); falls back to
    ``status == "SELECT"`` (the blind offered in ``BLIND_SELECT``), which is
    the analogue of ``pylatro``'s pre-blind requirement.
    """
    chosen: Optional[Dict[str, object]] = None
    fallback: Optional[Dict[str, object]] = None
    for slot in ("small", "big", "boss"):
        entry = blinds.get(slot)
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if status == "CURRENT":
            chosen = entry
            break
        if status == "SELECT" and fallback is None:
            fallback = entry
    blind = chosen if chosen is not None else fallback
    if blind is None:
        return "", "", 0
    return (
        _str(blind, "type", "blind", ""),
        _str(blind, "name", "blind", ""),
        _int(blind, "score", "blind", 0),
    )


def build_state(
    payload: Mapping[str, object], selected_indices: Sequence[int] = ()
) -> RealGameState:
    """Convert a BalatroBot ``gamestate`` payload into a :class:`RealGameState`.

    Args:
        payload: the decoded ``gamestate`` result.
        selected_indices: the fly's virtual selection carried over from the
            previous decision. Indices outside the current hand are dropped,
            which is what should happen after a refill changes the hand.

    Raises:
        UnmappableStateError: the game is in a non-run state (e.g. ``MENU``).
        ValueError: the payload does not match the documented schema.
    """
    raw_state = _str(payload, "state", "gamestate")
    won = _bool(payload, "won")
    blinds = _obj(payload, "blinds", "gamestate")
    blind_type, blind_name, blind_score = _current_blind(blinds)
    stage_int = _stage_for(raw_state, blind_type, won)

    round_info = _obj(payload, "round", "gamestate")
    hand_area = _obj(payload, "hand", "gamestate")
    jokers_area = _obj(payload, "jokers", "gamestate")
    consumables_area = _obj(payload, "consumables", "gamestate")
    shop_area = _obj(payload, "shop", "gamestate")
    vouchers_area = _obj(payload, "vouchers", "gamestate")
    packs_area = _obj(payload, "packs", "gamestate")
    pack_area = _obj(payload, "pack", "gamestate")

    def cards(area: Mapping[str, object], path: str) -> Tuple[RealCard, ...]:
        return tuple(
            RealCard.from_json(c, f"{path}.cards[{i}]")
            for i, c in enumerate(_card_list(area, path))
        )

    available = cards(hand_area, "hand")
    highlighted_limit = _int(hand_area, "highlighted_limit", "hand", SELECTED_SLOTS)

    state = RealGameState(
        available=available,
        plays=_int(round_info, "hands_left", "round", 0),
        discards=_int(round_info, "discards_left", "round", 0),
        score=_int(round_info, "chips", "round", 0),
        required_score=blind_score,
        money=_int(payload, "money", "gamestate", 0),
        stage=RealStage(stage_int),
        jokers=cards(jokers_area, "jokers"),
        consumables=cards(consumables_area, "consumables"),
        shop=cards(shop_area, "shop"),
        vouchers=cards(vouchers_area, "vouchers"),
        packs=cards(packs_area, "packs"),
        pack=cards(pack_area, "pack"),
        raw_state=raw_state,
        round_num=_int(payload, "round_num", "gamestate", 0),
        ante_num=_int(payload, "ante_num", "gamestate", 0),
        won=won,
        blind_type=blind_type,
        blind_name=blind_name,
        reroll_cost=_int(round_info, "reroll_cost", "round", 0),
        selected_max=max(1, highlighted_limit),
        joker_slots=_int(jokers_area, "limit", "jokers", 5),
        consumable_slots=_int(consumables_area, "limit", "consumables", 2),
        screen=ScreenFrame.from_json(payload.get("screen")),
    )

    if selected_indices:
        return state.with_selection(selected_indices)
    # Nothing carried over: seed from the game's own highlights if it has any.
    # Nothing in the API can set these, so in practice this is empty.
    highlighted = [i for i, c in enumerate(available) if c.highlight]
    if highlighted:
        return state.with_selection(highlighted)
    return state


# -- legality ---------------------------------------------------------------


def legality_mask(state: RealGameState) -> NDArray[np.int8]:
    """Length-109 legality mask for ``state``, with noop actions masked.

    Mirrors ``BalatroEnv(mask_noop_actions=True)``: ``move_card_*`` (24..69)
    and ``sort_hand`` (105, 106) are always 0 because reordering a hand never
    changes what it scores.

    Per-stage rules:

    * ``pre_blind``: ``select_blind``; ``skip_blind`` unless the blind is BOSS.
    * blind stages: ``select_card[i]`` for every hand slot that is *not* already
      selected, while the selection has room -- there is no deselect, matching
      ``pylatro``; ``play`` and ``discard`` when 1..``selected_max`` cards are
      selected and the respective counter is above zero; ``use_consumable[i]``
      for a held consumable.
    * ``post_blind``: ``cash_out``.
    * ``shop``: ``next_round`` always; ``buy_*`` only when the item exists and
      its price is within ``money``; ``reroll`` only when affordable;
      ``sell_joker`` / ``sell_consumable`` for held cards.
    * ``pack_open``: ``pick_pack_card[i]`` for cards in the pack, ``skip_pack``.
    * ``end_win`` / ``end_lose``: nothing.

    ``apply_tarot`` (89) and ``apply_spectral`` (108) stay 0 throughout: they
    are ``pylatro`` staging steps for a targeting flow that BalatroBot performs
    atomically inside ``use`` / ``pack``, so there is no call to map them to.
    """
    mask: NDArray[np.int8] = np.zeros(N_ACTIONS, dtype=np.int8)
    stage = state.stage.int()

    if stage in (STAGE_END_WIN, STAGE_END_LOSE):
        return mask

    n_hand = len(state.available)
    n_selected = len(state.selected_indices)

    if stage == STAGE_PRE_BLIND:
        mask[SELECT_BLIND_INDEX] = 1
        if state.blind_type != "BOSS":
            mask[SKIP_BLIND_INDEX] = 1
        return mask

    if stage in BLIND_STAGES:
        # pylatro offers no deselect: once a slot is selected, its
        # ``select_card[i]`` leaves the action space until the selection is
        # consumed by ``play``/``discard``. Verified against
        # ``BalatroEnv.reset(seed=100000)``, where stepping ``select_card[1]``
        # twice raises IllegalActionError and the second mask is
        # ``[0, 2, 3, 4, 5, 6, 7, play, discard]``.
        #
        # Offering a deselect here would be a strictly larger action space than
        # the readout was trained and evaluated on, and the v3 readout walks
        # straight into it: select slot 0, deselect slot 0, forever. That is not
        # a policy quirk that also shows up headless -- ``outputs/bc3/eval.json``
        # records ``truncated_fraction 0.0`` over 400 episodes for
        # ``real_alpn_kc_dn/mlp`` -- it is this mask being wrong.
        room = n_selected < state.selected_max
        if room:
            for slot in range(min(n_hand, _AVAILABLE_MAX)):
                if slot not in state.selected_indices:
                    mask[_SELECT_CARD_START + slot] = 1
        if 1 <= n_selected <= state.selected_max:
            if state.plays > 0:
                mask[PLAY_INDEX] = 1
            if state.discards > 0:
                mask[DISCARD_INDEX] = 1
        for i in range(min(len(state.consumables), _USE_CONSUMABLE_SLOTS)):
            mask[_USE_CONSUMABLE_START + i] = 1
        return mask

    if stage == STAGE_POST_BLIND:
        mask[CASH_OUT_INDEX] = 1
        return mask

    if stage == STAGE_SHOP:
        # Always available, so the fly can never wedge itself in the shop.
        mask[NEXT_ROUND_INDEX] = 1
        money = state.money

        for i, shop_index in enumerate(state.shop_joker_indices[:_BUY_JOKER_SLOTS]):
            if state.shop[shop_index].cost_buy <= money and len(state.jokers) < state.joker_slots:
                mask[_BUY_JOKER_START + i] = 1
        for i, shop_index in enumerate(state.shop_consumable_indices[:_BUY_CONSUMABLE_SLOTS]):
            if (
                state.shop[shop_index].cost_buy <= money
                and len(state.consumables) < state.consumable_slots
            ):
                mask[_BUY_CONSUMABLE_START + i] = 1
        for i, shop_index in enumerate(state.shop_playing_card_indices[:_BUY_PLAYING_CARD_SLOTS]):
            if state.shop[shop_index].cost_buy <= money:
                mask[_BUY_PLAYING_CARD_START + i] = 1
        if state.vouchers and state.vouchers[0].cost_buy <= money:
            mask[BUY_VOUCHER_INDEX] = 1
        for i, pack in enumerate(state.packs[:_BUY_PACK_SLOTS]):
            if pack.cost_buy <= money:
                mask[_BUY_PACK_START + i] = 1
        if 0 < state.reroll_cost <= money:
            mask[REROLL_INDEX] = 1
        for i in range(min(len(state.jokers), _SELL_JOKER_SLOTS)):
            mask[_SELL_JOKER_START + i] = 1
        for i in range(min(len(state.consumables), _SELL_CONSUMABLE_SLOTS)):
            mask[_SELL_CONSUMABLE_START + i] = 1
        return mask

    if stage == STAGE_PACK_OPEN:
        for i in range(min(len(state.pack), _PICK_PACK_SLOTS)):
            mask[_PICK_PACK_START + i] = 1
        mask[SKIP_PACK_INDEX] = 1
        return mask

    if stage in (STAGE_TAROT_HAND, STAGE_SPECTRAL_HAND):
        # Targeting is driven through `use`/`pack` with explicit card lists, so
        # the only thing to do from here is pick targets in the hand.
        for slot in range(min(n_hand, _AVAILABLE_MAX)):
            mask[_SELECT_CARD_START + slot] = 1
        return mask

    return mask


# -- action routing ---------------------------------------------------------


@dataclass(frozen=True)
class ActionPlan:
    """How to realise one of the 109 action indices against BalatroBot.

    ``kind`` is ``"local"`` for the selection toggles, which change only the
    fly's virtual selection and issue no RPC; ``"rpc"`` otherwise.
    ``client_method`` names a :class:`~flybalatro.realgame.client.BalatroBot`
    method and ``kwargs`` are its keyword arguments.
    """

    index: int
    name: str
    kind: str
    description: str
    client_method: str = ""
    kwargs: Mapping[str, object] = field(default_factory=dict)
    clears_selection: bool = False
    new_selection: Optional[Tuple[int, ...]] = None


def plan_action(state: RealGameState, index: int) -> ActionPlan:
    """Map action ``index`` onto the BalatroBot call that performs it.

    Raises:
        ValueError: ``index`` is out of range, or is one of the indices that
            has no BalatroBot equivalent (the noop reorder/sort block, and the
            ``apply_tarot`` / ``apply_spectral`` staging steps).
    """
    if not 0 <= index < N_ACTIONS:
        raise ValueError(f"action index {index} out of range 0..{N_ACTIONS - 1}")

    from ..env import ACTION_NAMES  # local import keeps module import cheap

    name = ACTION_NAMES[index]

    if _SELECT_CARD_START <= index < _AVAILABLE_MAX:
        slot = index - _SELECT_CARD_START
        toggled = state.toggle_selection(slot)
        # The "deselect" arm is unreachable through the mask -- an already
        # selected slot is not legal, matching pylatro. It is kept so that
        # plan_action stays total over the 0..23 block for any caller that
        # bypasses the mask (the mask-invariant test, and --continue-run
        # seeding a selection from the game's own highlights).
        verb = "deselect" if slot in state.selected_indices else "select"
        return ActionPlan(
            index=index,
            name=name,
            kind="local",
            description=f"{verb} hand slot {slot}",
            new_selection=toggled.selected_indices,
        )

    if _MOVE_START <= index < _MOVE_END or _SORT_HAND_START <= index <= _SORT_HAND_START + 1:
        raise ValueError(
            f"action {index} ({name}) is a hand-reorder noop; it is masked under "
            "mask_noop_actions=True semantics and has no BalatroBot call"
        )

    if index == PLAY_INDEX:
        cards = sorted(state.selected_indices)
        if not cards:
            raise ValueError("play with an empty selection")
        return ActionPlan(
            index=index, name=name, kind="rpc",
            description=f"play hand slots {cards}",
            client_method="play", kwargs={"cards": cards}, clears_selection=True,
        )

    if index == DISCARD_INDEX:
        cards = sorted(state.selected_indices)
        if not cards:
            raise ValueError("discard with an empty selection")
        return ActionPlan(
            index=index, name=name, kind="rpc",
            description=f"discard hand slots {cards}",
            client_method="discard", kwargs={"cards": cards}, clears_selection=True,
        )

    if index == CASH_OUT_INDEX:
        return ActionPlan(index, name, "rpc", "cash out", "cash_out", {}, True)

    if _BUY_JOKER_START <= index < _BUY_JOKER_START + _BUY_JOKER_SLOTS:
        return _plan_shop_buy(
            index, name, state.shop_joker_indices, index - _BUY_JOKER_START, "joker"
        )

    if index == NEXT_ROUND_INDEX:
        return ActionPlan(index, name, "rpc", "leave the shop", "next_round", {}, True)

    if index == SELECT_BLIND_INDEX:
        return ActionPlan(index, name, "rpc", "select the blind", "select", {}, True)

    if index == SKIP_BLIND_INDEX:
        return ActionPlan(index, name, "rpc", "skip the blind", "skip", {}, True)

    if _BUY_CONSUMABLE_START <= index < _BUY_CONSUMABLE_START + _BUY_CONSUMABLE_SLOTS:
        return _plan_shop_buy(
            index, name, state.shop_consumable_indices,
            index - _BUY_CONSUMABLE_START, "consumable",
        )

    if index == BUY_VOUCHER_INDEX:
        if not state.vouchers:
            raise ValueError("buy_voucher with no voucher in the shop")
        return ActionPlan(
            index, name, "rpc", "buy the voucher", "buy_voucher", {"index": 0}
        )

    if _BUY_PLAYING_CARD_START <= index < _BUY_PLAYING_CARD_START + _BUY_PLAYING_CARD_SLOTS:
        return _plan_shop_buy(
            index, name, state.shop_playing_card_indices,
            index - _BUY_PLAYING_CARD_START, "playing card",
        )

    if _USE_CONSUMABLE_START <= index < _USE_CONSUMABLE_START + _USE_CONSUMABLE_SLOTS:
        slot = index - _USE_CONSUMABLE_START
        if slot >= len(state.consumables):
            raise ValueError(f"use_consumable[{slot}]: no consumable in that slot")
        kwargs: Dict[str, object] = {"consumable": slot}
        if state.selected_indices:
            kwargs["cards"] = sorted(state.selected_indices)
        return ActionPlan(
            index, name, "rpc",
            f"use consumable {slot} ({state.consumables[slot].label})",
            "use", kwargs, True,
        )

    if index in (APPLY_TAROT_INDEX, APPLY_SPECTRAL_INDEX):
        raise ValueError(
            f"action {index} ({name}) has no BalatroBot equivalent: targeting is "
            "applied atomically by `use`/`pack`, not staged as a separate step"
        )

    if _SELL_JOKER_START <= index < _SELL_JOKER_START + _SELL_JOKER_SLOTS:
        slot = index - _SELL_JOKER_START
        if slot >= len(state.jokers):
            raise ValueError(f"sell_joker[{slot}]: no joker in that slot")
        return ActionPlan(
            index, name, "rpc", f"sell joker {slot} ({state.jokers[slot].label})",
            "sell_joker", {"index": slot},
        )

    if _SELL_CONSUMABLE_START <= index < _SELL_CONSUMABLE_START + _SELL_CONSUMABLE_SLOTS:
        slot = index - _SELL_CONSUMABLE_START
        if slot >= len(state.consumables):
            raise ValueError(f"sell_consumable[{slot}]: no consumable in that slot")
        return ActionPlan(
            index, name, "rpc", f"sell consumable {slot}",
            "sell_consumable", {"index": slot},
        )

    if _BUY_PACK_START <= index < _BUY_PACK_START + _BUY_PACK_SLOTS:
        slot = index - _BUY_PACK_START
        if slot >= len(state.packs):
            raise ValueError(f"buy_pack[{slot}]: no pack in that slot")
        return ActionPlan(
            index, name, "rpc", f"buy pack {slot} ({state.packs[slot].label})",
            "buy_pack", {"index": slot},
        )

    if _PICK_PACK_START <= index < _PICK_PACK_START + _PICK_PACK_SLOTS:
        slot = index - _PICK_PACK_START
        if slot >= len(state.pack):
            raise ValueError(f"pick_pack_card[{slot}]: pack has no card there")
        # `pack` spells its hand-target list `targets`, unlike `use`'s `cards`.
        kwargs = {"index": slot}
        if state.selected_indices:
            kwargs["targets"] = sorted(state.selected_indices)
        return ActionPlan(
            index, name, "rpc", f"take pack card {slot} ({state.pack[slot].label})",
            "pack_select", kwargs, True,
        )

    if index == SKIP_PACK_INDEX:
        return ActionPlan(index, name, "rpc", "skip the pack", "pack_skip", {}, True)

    if index == REROLL_INDEX:
        return ActionPlan(index, name, "rpc", "reroll the shop", "reroll", {})

    raise ValueError(f"action {index} ({name}) has no mapping")


def _plan_shop_buy(
    index: int, name: str, candidates: Sequence[int], nth: int, what: str
) -> ActionPlan:
    """Map ``buy_*[nth]`` onto ``buy card <shop index>``.

    ``pylatro`` indexes shop slots per category; BalatroBot indexes the shop's
    single card array, so the nth joker (say) is at ``candidates[nth]``.
    """
    if nth >= len(candidates):
        raise ValueError(f"{name}: shop has no {what} in slot {nth}")
    shop_index = candidates[nth]
    return ActionPlan(
        index=index,
        name=name,
        kind="rpc",
        description=f"buy {what} at shop index {shop_index}",
        client_method="buy_card",
        kwargs={"index": shop_index},
    )
