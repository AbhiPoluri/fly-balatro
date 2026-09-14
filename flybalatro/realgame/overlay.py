"""A transparent, click-through detection overlay for the real Balatro window.

The point of this file, said once so the screen never has to imply otherwise:
**the fly does not see the game.** ``flybalatro/hands.py`` enumerates the
subsets of the dealt cards outside the brain, ``flybalatro/glomerular.py``
turns the answer into 32 relay bits on 32 whole ORN glomeruli, and the readout
on the spike counts picks an action. The boxes drawn here are annotations of
*that* data -- what the harness told the fly and what the fly decided -- pinned
to where the corresponding cards happen to be on screen. They are not
detections. :data:`CAPTION` says so inside the frame and is not optional.

Three pieces, deliberately separable:

``HandGeometry``
    slot index -> screen rect, from fitted parameters in ``pov_geometry.json``.
    Pure arithmetic, no AppKit, no Quartz; this is what the tests exercise and
    what ``scripts/pov_calibrate.py`` draws onto a real screenshot.
``Decision`` / :func:`read_log` / :func:`follow`
    the per-decision records of ``outputs/realgame/log.jsonl`` (schema in
    ``docs/REALGAME_INSTALL.md`` section 6), parsed into something the drawing
    code can use. Rejected-call records carry no ``state`` and are skipped.
``OverlayWindow``
    a borderless, clear, ``ignoresMouseEvents`` NSWindow at floating level over
    the bounds Quartz reports for the Balatro window. AppKit and Quartz are
    imported lazily *inside* this class so importing the module costs nothing
    and needs no GUI session.

Run it::

    python -m flybalatro.realgame.overlay --replay outputs/realgame/log.jsonl
    python -m flybalatro.realgame.overlay --calibrate
    python -m flybalatro.realgame.overlay            # live: follows latest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..features import RANK_NAMES

__all__ = [
    "CAPTION",
    "Rect",
    "HandGeometry",
    "Card",
    "CardRect",
    "ScreenFrame",
    "Decision",
    "Box",
    "Annotation",
    "parse_record",
    "read_log",
    "LogTail",
    "LatestFile",
    "follow",
    "build_annotation",
    "aligned_rects",
    "ALIGN_GAME",
    "ALIGN_MODEL",
    "balatro_window_bounds",
    "OverlayWindow",
    "main",
]

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_PATH = Path(__file__).with_name("pov_geometry.json")
DEFAULT_LOG = ROOT / "outputs" / "realgame" / "log.jsonl"
DEFAULT_LATEST = ROOT / "outputs" / "realgame" / "latest.json"

#: Mandatory, drawn inside the frame by every renderer.
CAPTION: str = (
    "The fly does not see pixels. Boxes show the hand analysis it is given "
    "(computed outside the brain) and what it decided."
)

SUIT_GLYPHS: Tuple[str, ...] = ("♠", "♣", "♥", "♦")
SUIT_WORDS: Tuple[str, ...] = ("spades", "clubs", "hearts", "diamonds")
#: ``suit_index >= 2`` is a red suit; same convention as ``viewer/server.py``.
SUIT_IS_RED: Tuple[bool, ...] = (False, False, True, True)

# Palette, matched to the viewer's dark terminal style. No neon.
COL_CARD = (125, 136, 148, 110)       # every card, only under --all-cards
COL_BEST = (240, 180, 41, 255)        # the best subset the harness relayed
COL_SELECTED = (62, 199, 179, 255)    # what the fly has selected so far
COL_TEXT = (232, 236, 240, 255)
COL_DIM = (150, 160, 170, 255)
COL_PANEL = (10, 11, 13, 208)
COL_CHROME = (8, 10, 12, 170)         # the one backdrop, on empty felt
COL_REWARD = (51, 209, 122, 255)
COL_PUNISH = (239, 68, 68, 255)
COL_SHORT = (240, 180, 41, 255)       # the best hand does not reach the blind
COL_ENOUGH = (51, 209, 122, 255)      # it clears it

#: Where the chrome goes: fractions of the game canvas, chosen to sit in felt
#: that Balatro leaves empty during a hand -- right of the run-info sidebar
#: (whose border is at x = 0.249), below the joker / consumable slot outlines
#: (they end at y = 0.31), above the hand fan (``top_y`` = 0.589) and left of
#: the deck (x = 0.847). Nothing the game draws is underneath it.
CHROME_RECT: Tuple[float, float, float, float] = (0.275, 0.330, 0.565, 0.225)

#: Where it goes when there is no hand on the table -- the shop and the
#: round-eval screen both fill the band above with their own panels, but the
#: strip above the joker slots (they start at y = 0.106) is empty on every
#: screen. Nothing big is drawn there: with no cards there is nothing to say
#: except that the relay block is silent.
CHROME_RECT_EMPTY: Tuple[float, float, float, float] = (0.275, 0.014, 0.565, 0.080)

_ACTION_LABELS: Dict[str, str] = {"play": "PLAY", "discard": "DIG", "cash_out": "CASH OUT"}

#: Where a frame's card rects came from. ``game`` is the transforms the game
#: itself reported for that frame; ``model`` is the fitted static fan. Drawn on
#: screen, because they are not equally trustworthy.
ALIGN_GAME: str = "game"
ALIGN_MODEL: str = "model"


# --------------------------------------------------------------------------- #
# geometry                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle in canvas pixels, top-left origin."""

    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def inside(self, width: float, height: float, slack: float = 0.0) -> bool:
        return (
            self.x >= -slack
            and self.y >= -slack
            and self.right <= width + slack
            and self.bottom <= height + slack
        )


@dataclass(frozen=True)
class HandGeometry:
    """Balatro's hand fan as a handful of fractions of the game canvas.

    Every field is a fraction so the same numbers work at any window size; they
    were fitted on ``outputs/realgame/balatro_fly_firsthand.png`` (2418x1570,
    a settled 8-card hand) by ``scripts/pov_calibrate.py``. See ``docs/POV.md``.

    The hand is a fan: cards evenly pitched left to right, the middle of the fan
    sitting slightly higher than the ends (``arc_lift``) and its bounding box
    slightly shorter there because the end cards are rotated (``arc_shrink``).
    A selected card is raised by ``raise_frac`` -- which is **not measured**,
    because the BalatroBot API has no selection endpoint and the real game never
    highlights the fly's selection. It is there to show the fly's *virtual*
    selection and the number is nominal.
    """

    center_x: float
    top_y: float
    card_w: float
    card_h: float
    span: float
    arc_lift: float
    arc_shrink: float
    raise_frac: float
    max_pitch_ratio: float
    reference_w: int
    reference_h: int
    #: Balatro keeps the same card pitch when the hand is short and centres the
    #: shorter fan, rather than spreading it to fill ``span``. Measured; see
    #: :meth:`pitch`. Defaults true, and false restores the old spread model.
    fixed_pitch: bool = True
    reference_image: str = ""
    n_reference: int = 8
    fit: Dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "HandGeometry":
        hand = obj["hand"]
        ref = obj.get("reference", {})
        return cls(
            center_x=float(hand["center_x"]),
            top_y=float(hand["top_y"]),
            card_w=float(hand["card_w"]),
            card_h=float(hand["card_h"]),
            span=float(hand["span"]),
            arc_lift=float(hand["arc_lift"]),
            arc_shrink=float(hand["arc_shrink"]),
            raise_frac=float(hand["raise"]),
            max_pitch_ratio=float(hand.get("max_pitch_ratio", 1.0)),
            fixed_pitch=bool(hand.get("fixed_pitch", True)),
            reference_w=int(ref.get("width", 0)),
            reference_h=int(ref.get("height", 0)),
            reference_image=str(ref.get("image", "")),
            n_reference=int(ref.get("n_cards", 8)),
            fit=dict(obj.get("fit", {})),
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "HandGeometry":
        with open(path or GEOMETRY_PATH, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- layout ------------------------------------------------------------
    #: Slots a full Balatro hand has; the pitch is fitted at this ``n``.
    FULL_HAND: int = 8

    def pitch(self, n_cards: int) -> float:
        """Centre-to-centre spacing, as a fraction of canvas width.

        With a full hand the fan is ``span`` wide and the cards overlap. **With
        fewer cards Balatro does not spread them out** -- it keeps the same
        pitch and centres the shorter fan.

        That is measured, not assumed. The original model spread the fan to fill
        ``span`` (clamped at ``max_pitch_ratio`` card widths), which is the
        natural guess and is wrong: on two independent settled 3-card hands
        pulled out of ``outputs/realgame/balatro_fly.mov`` the real pitch is
        165-171 px against the 8-card fit of 164.65 px, while the spread model
        wanted 225.4 px and put the end cards **60 px** off their cards
        (``outputs/pov/dynamic_align.json``, case B). Holding the pitch fixed
        brings that to about 4 px. ``max_pitch_ratio`` is kept as an upper
        clamp; with a fixed pitch it can no longer bind.
        """
        if n_cards <= 1:
            return 0.0
        even = (self.span - self.card_w) / (n_cards - 1)
        if self.fixed_pitch:
            full = (self.span - self.card_w) / (self.FULL_HAND - 1)
            # ``min`` so a hand with *more* than eight slots would still fit.
            even = min(full, even)
        return min(self.card_w * self.max_pitch_ratio, even)

    def slot_rect(
        self,
        slot: int,
        n_cards: int,
        *,
        selected: bool = False,
        width: float = 1.0,
        height: float = 1.0,
    ) -> Rect:
        """Screen rect of hand slot ``slot`` in a hand of ``n_cards``.

        ``width`` / ``height`` are the canvas size in pixels; leave them at 1.0
        to get fractions back.
        """
        if n_cards <= 0:
            raise ValueError("n_cards must be positive")
        if not 0 <= slot < n_cards:
            raise ValueError(f"slot {slot} out of range for a hand of {n_cards}")
        pitch = self.pitch(n_cards)
        fan = pitch * (n_cards - 1) + self.card_w
        left0 = self.center_x - fan / 2.0
        if n_cards == 1:
            bump = 1.0
        else:
            u = (slot - (n_cards - 1) / 2.0) / ((n_cards - 1) / 2.0)
            bump = 1.0 - u * u
        x = left0 + pitch * slot
        y = self.top_y - self.arc_lift * bump - (self.raise_frac if selected else 0.0)
        h = self.card_h - self.arc_shrink * bump
        return Rect(x * width, y * height, self.card_w * width, h * height)

    def hand_rects(
        self,
        n_cards: int,
        selected: Sequence[int] = (),
        *,
        width: float = 1.0,
        height: float = 1.0,
    ) -> List[Rect]:
        sel = set(int(s) for s in selected)
        return [
            self.slot_rect(i, n_cards, selected=(i in sel), width=width, height=height)
            for i in range(n_cards)
        ]


# --------------------------------------------------------------------------- #
# the log feed                                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CardRect:
    """Where the game itself says this card is, in LOVE pixels.

    The patched mod reports each hand card's *visible* transform -- the eased
    one the card is drawn at, not the target it is easing toward -- so a rect
    here is animation-accurate by construction. ``r`` is a rotation in radians
    about the rect's own centre; the Balatro fan runs about +/-5.5 degrees.
    """

    x: float
    y: float
    w: float
    h: float
    r: float = 0.0
    moving: bool = False
    #: ``VT.scale``, applied by LOVE about the rect's own centre. The reported
    #: transform is the box *before* it, so a hand card at 0.95 is drawn 5%
    #: smaller than ``w`` x ``h``. Measured on real pixels: the unscaled box is
    #: 225 px wide against a 213 px card. 1.0 for any log that predates the
    #: field, which then renders exactly as it did before.
    scale: float = 1.0

    def drawn(self) -> "CardRect":
        """This rect at the size it is actually drawn, same centre."""
        if self.scale == 1.0:
            return self
        w, h = self.w * self.scale, self.h * self.scale
        return CardRect(x=self.x + (self.w - w) / 2.0,
                        y=self.y + (self.h - h) / 2.0,
                        w=w, h=h, r=self.r, moving=self.moving)

    @classmethod
    def parse(cls, value: Any) -> Optional["CardRect"]:
        if not isinstance(value, dict):
            return None
        try:
            w, h = float(value["w"]), float(value["h"])
            if not (w > 0.0 and h > 0.0):
                return None
            try:
                scale = float(value.get("scale", 1.0) or 1.0)
            except (TypeError, ValueError):
                scale = 1.0
            return cls(x=float(value["x"]), y=float(value["y"]), w=w, h=h,
                       r=float(value.get("r", 0.0) or 0.0),
                       moving=bool(value.get("moving", False)),
                       scale=scale if 0.05 <= scale <= 4.0 else 1.0)
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class ScreenFrame:
    """The canvas the :class:`CardRect` values are expressed in.

    **These are LOVE's draw units, not the backing store.** A card rect is
    ``VT * G.TILESCALE * G.TILESIZE``, and ``G.TILESCALE`` is derived in
    ``love.resize(w, h)`` from the same ``w`` that ``love.graphics.getDimensions()``
    reports -- so the rects live in ``getDimensions()`` space by construction.
    On a high-DPI window ``getPixelDimensions()`` is a multiple of that and is
    the *wrong* denominator: dividing by it would halve every box and put it in
    the wrong place. The mod reports both; this reads ``width`` / ``height`` and
    keeps ``pixel_*`` only for diagnosis.
    """

    width: float
    height: float
    #: Backing-store size, when reported. Never used for scaling; kept so a
    #: mismatch with the overlay window is visible rather than silent.
    pixel_width: float = 0.0
    pixel_height: float = 0.0

    @classmethod
    def parse(cls, value: Any) -> Optional["ScreenFrame"]:
        if not isinstance(value, dict):
            return None
        def num(key: str) -> float:
            try:
                return float(value.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        w, h = num("width"), num("height")
        if not (w > 0.0 and h > 0.0):
            return None
        return cls(w, h, num("pixel_width") or w, num("pixel_height") or h)


@dataclass(frozen=True)
class Card:
    slot: int
    rank_index: int
    suit_index: int
    #: Reported by the game for this exact frame; ``None`` on a log written
    #: before the mod carried it, or by a mod build that does not.
    rect: Optional[CardRect] = None

    @property
    def rank(self) -> str:
        if 0 <= self.rank_index < len(RANK_NAMES):
            return RANK_NAMES[self.rank_index]
        return "?"

    @property
    def suit(self) -> str:
        if 0 <= self.suit_index < len(SUIT_GLYPHS):
            return SUIT_GLYPHS[self.suit_index]
        return "?"

    @property
    def red(self) -> bool:
        return bool(SUIT_IS_RED[self.suit_index]) if 0 <= self.suit_index < 4 else False

    @property
    def label(self) -> str:
        """``3<diamond> rank 1 suit 1`` -- the glyph plus the two indices the
        encoder actually used, so the label is the fly's input, not prettied."""
        return f"{self.rank}{self.suit} rank {self.rank_index} suit {self.suit_index}"


@dataclass(frozen=True)
class Decision:
    """One decision record, flattened to what the overlay draws."""

    index: int
    t: float
    action_name: str
    action_description: str
    cards: Tuple[Card, ...]
    selected: Tuple[int, ...]
    best_slots: Tuple[int, ...]
    best_type: str
    best_score: int
    needed: int
    score_bucket: str
    selected_type: str
    selected_is_best: bool
    relay_active: bool
    plays: int
    discards: int
    score: int
    required: int
    blind: str
    ante: int
    round: int
    stage: str
    policy_mode: str
    uses_brain: bool
    dopamine: Optional[str]
    rates: Dict[str, float]
    screen: Optional[ScreenFrame] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_cards(self) -> int:
        return len(self.cards)

    @property
    def has_rects(self) -> bool:
        """Every dealt slot carries a rect the game reported."""
        return bool(self.cards) and all(c.rect is not None for c in self.cards)

    @property
    def any_moving(self) -> bool:
        return any(c.rect is not None and c.rect.moving for c in self.cards)


def _as_obj(value: Any) -> Dict[str, Any]:
    """Lua has one table type, so the mod emits ``[]`` for an empty object."""
    return value if isinstance(value, dict) else {}


def parse_record(obj: Dict[str, Any]) -> Optional[Decision]:
    """One ``log.jsonl`` line -> :class:`Decision`, or ``None`` if it is not one.

    Rejected API calls log a short record with an ``error`` and no ``state``;
    those are skipped rather than drawn, because there is nothing to draw.
    """
    if not isinstance(obj, dict) or "error" in obj:
        return None
    state = _as_obj(obj.get("state"))
    if not state:
        return None
    relay = _as_obj(obj.get("relayed_hand_analysis"))
    cards = tuple(
        Card(
            slot=int(c.get("slot", i)),
            rank_index=int(c.get("rank_index", -1)),
            suit_index=int(c.get("suit_index", -1)),
            rect=CardRect.parse(c.get("rect")),
        )
        for i, c in enumerate(state.get("hand") or [])
        if isinstance(c, dict)
    )
    dopamine = obj.get("dopamine")
    if dopamine is None:
        dopamine = _as_obj(obj.get("mb")).get("dopamine")
    if dopamine in ("", "none"):
        dopamine = None
    return Decision(
        index=int(obj.get("decision", 0)),
        t=float(obj.get("t", 0.0)),
        action_name=str(obj.get("action_name", "?")),
        action_description=str(obj.get("action_description", "")),
        cards=cards,
        selected=tuple(int(i) for i in (state.get("selected_indices") or [])),
        best_slots=tuple(int(i) for i in (relay.get("best_slots") or [])),
        best_type=str(relay.get("best_type", "none")),
        best_score=int(relay.get("best_score", 0) or 0),
        needed=int(relay.get("needed", 0) or 0),
        score_bucket=str(relay.get("score_bucket", "")),
        selected_type=str(relay.get("selected_type", "none")),
        selected_is_best=bool(relay.get("selected_is_best", False)),
        relay_active=bool(relay.get("active", False)),
        plays=int(state.get("plays", 0) or 0),
        discards=int(state.get("discards", 0) or 0),
        score=int(state.get("score", 0) or 0),
        required=int(state.get("required_score", 0) or 0),
        blind=str(state.get("blind") or state.get("raw_state") or ""),
        ante=int(state.get("ante", 0) or 0),
        round=int(state.get("round", 0) or 0),
        stage=str(state.get("raw_state", "")),
        policy_mode=str(obj.get("policy_mode", "")),
        uses_brain=bool(obj.get("policy_uses_brain", False)),
        dopamine=str(dopamine) if dopamine else None,
        rates={k: float(v) for k, v in _as_obj(obj.get("spike_rates_hz")).items()},
        screen=ScreenFrame.parse(state.get("screen")),
        raw=obj,
    )


def read_log(path: Path, *, with_cards_only: bool = True) -> List[Decision]:
    """Every parseable decision in a ``.jsonl`` log, in order."""
    out: List[Decision] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-flushed final line while the game is running
            dec = parse_record(obj)
            if dec is None:
                continue
            if with_cards_only and not dec.cards:
                continue
            out.append(dec)
    return out


class LogTail:
    """Non-blocking reader for a growing ``.jsonl`` log.

    Non-blocking on purpose: the overlay has to keep pumping the AppKit event
    loop between decisions or the window stops repainting. Survives the file
    not existing yet, and being truncated or rotated between runs.
    """

    def __init__(self, path: Path, *, from_start: bool = False,
                 with_cards_only: bool = True) -> None:
        self.path = path
        self.with_cards_only = with_cards_only
        self._pending = ""
        self._inode: Optional[int] = None
        self._offset = 0
        if not from_start and path.exists():
            st = path.stat()
            self._offset, self._inode = st.st_size, st.st_ino

    def poll(self) -> List[Decision]:
        try:
            st = self.path.stat()
        except OSError:
            return []
        if self._inode is not None and (st.st_ino != self._inode or st.st_size < self._offset):
            self._offset, self._pending = 0, ""
        self._inode = st.st_ino
        if st.st_size <= self._offset:
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            fh.seek(self._offset)
            self._pending += fh.read()
            self._offset = fh.tell()
        *lines, self._pending = self._pending.split("\n")
        out: List[Decision] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            dec = parse_record(obj)
            if dec is not None and (dec.cards or not self.with_cards_only):
                out.append(dec)
        return out


class LatestFile:
    """Non-blocking reader for ``latest.json``.

    ``play.py`` writes it to a temp file and ``os.replace``s it, so a poller
    never sees a partial object. This is the feed ``docs/REALGAME_INSTALL.md``
    section 6 documents for exactly this purpose.
    """

    def __init__(self, path: Path, *, with_cards_only: bool = True) -> None:
        self.path = path
        self.with_cards_only = with_cards_only
        self._mtime = -1.0

    def poll(self) -> List[Decision]:
        try:
            mtime = self.path.stat().st_mtime
            if mtime == self._mtime:
                return []
            with open(self.path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            self._mtime = mtime
        except (OSError, json.JSONDecodeError):
            return []
        dec = parse_record(obj)
        if dec is None or not (dec.cards or not self.with_cards_only):
            return []
        return [dec]


def follow(path: Path, *, poll: float = 0.25, from_start: bool = False) -> Iterator[Decision]:
    """Blocking wrapper around :class:`LogTail`, for scripts and tests."""
    tail = LogTail(path, from_start=from_start)
    while True:
        got = tail.poll()
        for dec in got:
            yield dec
        if not got:
            time.sleep(poll)


def watch_latest(path: Path, *, poll: float = 0.2) -> Iterator[Decision]:
    """Blocking wrapper around :class:`LatestFile`."""
    latest = LatestFile(path)
    while True:
        got = latest.poll()
        for dec in got:
            yield dec
        if not got:
            time.sleep(poll)


# --------------------------------------------------------------------------- #
# annotation                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Box:
    rect: Rect
    label: str
    kind: str           # "card" | "best" | "selected"
    color: Tuple[int, int, int, int]
    thickness: int
    slot: int = 0
    #: Rotation in radians about the rect's centre. Balatro fans the hand from
    #: about -5.5 to +5.5 degrees, so an axis-aligned box on an end card has to
    #: be drawn oversized to cover it; with the real angle it can be drawn the
    #: size of the card. Zero whenever the angle is not known.
    angle: float = 0.0
    #: Fill the rect instead of stroking it. Used for the thin bar under a slot
    #: the fly has (virtually) selected -- a mark, deliberately not a box, so the
    #: amber best-subset outline stays the only box on screen.
    fill: bool = False


@dataclass(frozen=True)
class Annotation:
    """Everything both renderers draw, in canvas pixels.

    The screen answers three questions and nothing else: which cards the fly is
    about to play (``boxes``), what hand that is and whether it reaches the
    blind (``headline`` / ``headline_note`` / ``headline_ok``), and what the fly
    just decided (``decision`` / ``decision_sub``). ``hud`` is one dim line of
    context. All of it is drawn inside ``chrome``, which is empty felt, so the
    overlay never covers anything the game drew.
    """

    canvas: Tuple[float, float]
    boxes: Tuple[Box, ...]
    hud: Tuple[str, ...]
    decision: str
    decision_sub: str
    caption: str
    flash: Optional[str]
    source: str
    headline: str = ""
    headline_note: str = ""
    headline_ok: bool = False
    chrome: Rect = Rect(0.0, 0.0, 0.0, 0.0)
    #: ``game`` when the boxes came from the transforms the game reported for
    #: this exact frame, ``model`` when they came from the fitted static fan.
    #: Shown on screen, because the two are not equally trustworthy.
    align: str = ALIGN_MODEL


def action_label(action_name: str) -> Tuple[str, str]:
    """``select_card[3]`` -> ``("SELECT 3", "select hand slot 3")``."""
    name = action_name.strip()
    if name in _ACTION_LABELS:
        return _ACTION_LABELS[name], name
    if name.startswith("select_card[") and name.endswith("]"):
        return "SELECT " + name[len("select_card["):-1], name
    return name.replace("_", " ").upper(), name


def aligned_rects(
    dec: Decision,
    geom: HandGeometry,
    width: float,
    height: float,
) -> Tuple[Dict[int, Tuple[Rect, float]], str]:
    """Where each dealt slot is, and how we know.

    Two sources, preferred in this order:

    ``game`` -- **the game told us.** The patched BalatroBot mod reports every
        hand card's *visible* LOVE transform (``Card.VT``, the eased one the
        card is actually drawn at, not the ``T`` it is easing toward) together
        with the room offset and tile scaling, and the record carries the rect
        per slot plus the canvas it is in. Scaling that canvas onto the overlay
        window is the whole computation. It is exact, it costs nothing, and it
        is correct for **any** number of cards, for cards mid-deal or
        mid-discard, for a raised card and for a resized window, because there
        is no model to be wrong.
    ``model`` -- the fitted static fan in ``pov_geometry.json``. The fallback
        for a run recorded before the mod carried geometry, or against a mod
        build that does not. Accurate to a few pixels on a settled 8-card hand
        and progressively less so away from that; ``docs/POV.md`` section 4 has
        the measured residuals.

    Returns ``({slot: (rect, angle_radians)}, source)``. The angle is 0 under
    the model, which knows the fan exists but not where a given card is in it.
    """
    n = dec.n_cards
    if n <= 0:
        return {}, ALIGN_MODEL
    if dec.has_rects and dec.screen is not None:
        sx = float(width) / dec.screen.width
        sy = float(height) / dec.screen.height
        out: Dict[int, Tuple[Rect, float]] = {}
        for card in dec.cards:
            r = card.rect
            if r is None:  # pragma: no cover - has_rects guarantees otherwise
                continue
            # ``VT.scale`` before anything else: LOVE applies it about the
            # rect's centre when it draws, so the box that sits on the card's
            # border is the scaled one. Measured on the real window, taking it
            # out moves the outline from 225 px wide to 214 against a card
            # detected at 213 (docs/POV.md section 5).
            r = r.drawn()
            out[card.slot] = (
                Rect(r.x * sx, r.y * sy, r.w * sx, r.h * sy), float(r.r)
            )
        if out:
            return out, ALIGN_GAME
    selected = set(dec.selected)
    return (
        {
            slot: (
                geom.slot_rect(slot, n, selected=slot in selected,
                               width=width, height=height),
                0.0,
            )
            for slot in range(n)
        },
        ALIGN_MODEL,
    )


def build_annotation(
    dec: Decision,
    geom: HandGeometry,
    width: float,
    height: float,
    *,
    source: str = "",
    labels: bool = False,
    all_cards: bool = False,
) -> Annotation:
    """Boxes + chrome for one decision, over a canvas ``width`` x ``height``.

    By default the only box is the amber outline on the best subset -- the cards
    the fly is about to play. The cards themselves are legible on screen, so
    nothing repeats what they already say:

    ``all_cards``
        also outline every dealt slot in low-contrast grey. Off by default:
        three box styles at once is noise.
    ``labels``
        also write ``K<heart> rank 11 suit 2`` over every slot -- the glyph plus
        the two indices ``flybalatro/features.py`` encoded. Off by default
        because the card is right there; on for ``scripts/pov_calibrate.py``,
        where the point of the image is that the label matches the pixels.

    The hand type and score are drawn **once**, in the chrome, not under every
    card of the subset.
    """
    n = dec.n_cards
    best = set(dec.best_slots)
    selected = set(dec.selected)
    # An *outcome* record reports what the game paid for a hand that has already
    # been played. It carries the board the decision was taken on so that every
    # line of the log is self-contained -- but by the time it is written the
    # game is animating that hand away and dealing its replacements, so those
    # cards are no longer where the record says. Drawing boxes from it would put
    # an amber outline on stale positions for the whole pause between hands,
    # which is exactly the drift this overlay exists to remove. So an outcome
    # frame gets the dopamine flash, the headline and the decision, and no boxes.
    resolved = str(dec.raw.get("kind", "")) == "outcome"
    show_best = dec.relay_active and bool(best) and not resolved
    boxes: List[Box] = []
    placed, align = aligned_rects(dec, geom, width, height) if not resolved else ({}, ALIGN_MODEL)
    for card in dec.cards:
        slot = card.slot
        if not 0 <= slot < n or slot not in placed:
            continue
        rect, angle = placed[slot]
        if all_cards or labels:
            boxes.append(
                Box(rect, card.label if labels else "", "card", COL_CARD,
                    2 if all_cards else 0, slot, angle=angle)
            )
        if slot in selected:
            # A bar, not a box: the fly's virtual selection is worth showing but
            # must not compete with the one emphasis. Appended *before* the
            # amber outline so the outline draws on top of it -- painted the
            # other way round the bar hides the bottom edge of the box, which is
            # the one thing on screen that has to be unambiguous.
            bar = max(3.0, 0.006 * height)
            boxes.append(
                Box(Rect(rect.x, rect.bottom - bar, rect.w, bar), "", "selected",
                    COL_SELECTED, 0, slot, angle=angle, fill=True)
            )
        if slot in best and show_best:
            # No inset when the angle is known: the outline is drawn rotated
            # with the card, so it sits on the card's own border instead of
            # having to be oversized to cover a tilted rectangle with an
            # upright box.
            boxes.append(
                Box(rect if abs(angle) > 1e-4 else _inset(rect, 3),
                    "", "best", COL_BEST, 4, slot, angle=angle)
            )

    if resolved and dec.relay_active:
        flash = {"reward": "REWARD", "punish": "PUNISHMENT"}.get(dec.dopamine or "", "")
        headline = f"{dec.best_type.upper()} \u00b7 {dec.best_score} \u2014 played"
        ok = bool(dec.dopamine == "reward")
        outcome = _as_obj(dec.raw.get("outcome"))
        chips = outcome.get("chips_gained")
        note = (f"{chips} chips" if chips is not None else "")
        if flash:
            note = (note + " \u00b7 " if note else "") + f"dopamine: {flash}"
        elif note:
            note += " \u00b7 no dopamine"
    elif show_best:
        headline = f"{dec.best_type.upper()} \u00b7 {dec.best_score}"
        # ``needed`` is the chips still required for this blind, so the hand is
        # enough exactly when best_score >= needed.
        ok = dec.best_score >= dec.needed
        if dec.needed <= 0:
            note = "the blind is already met"
        elif ok:
            note = f"enough \u2014 clears the {dec.needed} still needed"
        else:
            note = f"{dec.needed - dec.best_score} short of the {dec.needed} still needed"
    elif dec.relay_active:
        headline = "NO SCORING HAND"
        ok, note = False, ""
    else:
        headline = "RELAY SILENT"
        ok, note = False, "no cards in play \u2014 the 32 relay bits are all off"

    label, _raw = action_label(dec.action_name)
    # Deliberately usually empty. The amber outline already says *which* cards
    # are being played, so "play hand slots [6, 7]" under a picture of exactly
    # those two cards is a line of chrome that tells the reader nothing. What
    # is worth a line is the surprising case: the fly playing something that is
    # not the best subset the harness found for it.
    sub = ""
    if show_best and dec.action_name == "play" and not dec.selected_is_best:
        sub = "NOT the best subset the harness found"
    pct = f"{dec.score}/{dec.required}" if dec.required else str(dec.score)
    hud = [
        f"round {pct} \u00b7 plays {dec.plays} \u00b7 discards {dec.discards} \u00b7 "
        f"{dec.blind} \u00b7 ante {dec.ante} \u00b7 decision {dec.index}"
    ]
    cx, cy, cw, ch = CHROME_RECT if n else CHROME_RECT_EMPTY
    return Annotation(
        canvas=(float(width), float(height)),
        boxes=tuple(boxes),
        hud=tuple(hud),
        decision=label,
        decision_sub=sub,
        caption=CAPTION,
        flash=dec.dopamine,
        source=source,
        headline=headline,
        headline_note=note,
        headline_ok=ok,
        chrome=Rect(cx * width, cy * height, cw * width, ch * height),
        align=align,
    )


def _inset(rect: Rect, by: float) -> Rect:
    return Rect(rect.x + by, rect.y + by, max(1.0, rect.w - 2 * by), max(1.0, rect.h - 2 * by))


# --------------------------------------------------------------------------- #
# finding the game window                                                      #
# --------------------------------------------------------------------------- #
#: Quartz reports the owner as "Balatro" for the Steam build and "love" when
#: LOVE is launched directly; Lovely does not change either.
OWNER_NAMES: Tuple[str, ...] = ("balatro", "love", "love.app")


def balatro_window_bounds(
    owners: Sequence[str] = OWNER_NAMES,
) -> Optional[Tuple[str, Tuple[float, float, float, float]]]:
    """``(owner, (x, y, w, h))`` of the game window in Quartz screen points.

    Quartz coordinates are top-left origin and the bounds include the title bar.
    Returns ``None`` when the game is not running (which is the normal case for
    ``--calibrate`` and ``--replay``). Imports Quartz lazily.
    """
    try:
        from Quartz import (  # type: ignore[import-not-found]
            CGWindowListCopyWindowInfo,
            kCGNullWindowID,
            kCGWindowListExcludeDesktopElements,
            kCGWindowListOptionOnScreenOnly,
        )
    except ImportError:  # pragma: no cover - depends on the machine
        return None
    options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
    wanted = {o.lower() for o in owners}
    best: Optional[Tuple[str, Tuple[float, float, float, float]]] = None
    best_area = 0.0
    for win in CGWindowListCopyWindowInfo(options, kCGNullWindowID) or []:
        owner = str(win.get("kCGWindowOwnerName") or "")
        if owner.lower() not in wanted:
            continue
        if int(win.get("kCGWindowLayer", 0) or 0) != 0:
            continue
        b = win.get("kCGWindowBounds") or {}
        rect = (
            float(b.get("X", 0.0)),
            float(b.get("Y", 0.0)),
            float(b.get("Width", 0.0)),
            float(b.get("Height", 0.0)),
        )
        area = rect[2] * rect[3]
        if area > best_area:
            best, best_area = (owner, rect), area
    return best


def content_rect(
    bounds: Tuple[float, float, float, float], titlebar: float
) -> Tuple[float, float, float, float]:
    """Strip the title bar off Quartz bounds to get the LOVE canvas rect."""
    x, y, w, h = bounds
    return (x, y + titlebar, w, max(1.0, h - titlebar))


# --------------------------------------------------------------------------- #
# the window                                                                   #
# --------------------------------------------------------------------------- #
class OverlayWindow:
    """A borderless clear NSWindow that draws an :class:`Annotation`.

    Click-through (``ignoresMouseEvents``), floating above normal windows, no
    shadow, not in the window list. AppKit is imported in ``__init__`` so the
    module stays importable in a test run with no GUI.
    """

    def __init__(self, rect: Tuple[float, float, float, float], *, title: str = "FLY POV") -> None:
        import AppKit  # type: ignore[import-not-found]

        self._AppKit = AppKit
        self.annotation: Optional[Annotation] = None
        self._flash_until = 0.0

        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        frame = self._to_cocoa(rect)
        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        window.setOpaque_(False)
        window.setBackgroundColor_(AppKit.NSColor.clearColor())
        window.setLevel_(AppKit.NSFloatingWindowLevel)
        window.setIgnoresMouseEvents_(True)
        window.setHasShadow_(False)
        window.setTitle_(title)
        window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
            | AppKit.NSWindowCollectionBehaviorIgnoresCycle
        )
        view = _view_class().alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, frame.size.width, frame.size.height)
        )
        view.owner = self
        window.setContentView_(view)
        window.orderFrontRegardless()
        self.app = app
        self.window = window
        self.view = view

    # -- geometry ----------------------------------------------------------
    def _to_cocoa(self, rect: Tuple[float, float, float, float]):
        """Quartz (top-left origin) -> Cocoa (bottom-left origin, main screen)."""
        AppKit = self._AppKit
        x, y, w, h = rect
        screens = AppKit.NSScreen.screens()
        screen_h = float(screens[0].frame().size.height) if screens else 0.0
        return AppKit.NSMakeRect(x, screen_h - y - h, w, h)

    def move_to(self, rect: Tuple[float, float, float, float]) -> None:
        self.window.setFrame_display_(self._to_cocoa(rect), True)

    # -- content -----------------------------------------------------------
    def show(self, annotation: Annotation) -> None:
        self.annotation = annotation
        if annotation.flash:
            self._flash_until = time.monotonic() + 0.9
        self.view.setNeedsDisplay_(True)

    @property
    def flashing(self) -> Optional[str]:
        if self.annotation and self.annotation.flash and time.monotonic() < self._flash_until:
            return self.annotation.flash
        return None

    def pump(self, seconds: float = 0.05) -> None:
        """Run the AppKit event loop briefly; the caller owns the real loop."""
        AppKit = self._AppKit
        deadline = AppKit.NSDate.dateWithTimeIntervalSinceNow_(seconds)
        while True:
            event = self.app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                AppKit.NSEventMaskAny, deadline, AppKit.NSDefaultRunLoopMode, True
            )
            if event is None:
                break
            self.app.sendEvent_(event)

    def close(self) -> None:
        self.window.orderOut_(None)


def _ns_color(AppKit, rgba: Tuple[int, int, int, int], alpha_scale: float = 1.0):
    r, g, b, a = rgba
    return AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
        r / 255.0, g / 255.0, b / 255.0, (a / 255.0) * alpha_scale
    )


_VIEW_CLASS = None


def _view_class():  # pragma: no cover - needs AppKit
    """Define (once) the NSView subclass that paints an :class:`Annotation`.

    Built lazily so importing this module never touches AppKit: the tests
    exercise the geometry and the log parser in a plain interpreter.
    """
    global _VIEW_CLASS
    if _VIEW_CLASS is not None:
        return _VIEW_CLASS
    import AppKit  # type: ignore[import-not-found]
    import objc  # type: ignore[import-not-found]

    class _OverlayView(AppKit.NSView):
        """Cocoa's origin is bottom-left; ``isFlipped`` makes ours top-left."""

        owner = objc.ivar()

        def isFlipped(self):  # noqa: N802 - Cocoa selector
            return True

        def drawRect_(self, _dirty):  # noqa: N802 - Cocoa selector
            owner = self.owner
            ann = getattr(owner, "annotation", None)
            if ann is None:
                return
            bounds = self.bounds()
            W = float(bounds.size.width)
            H = float(bounds.size.height)
            sx = W / max(1.0, ann.canvas[0])
            sy = H / max(1.0, ann.canvas[1])

            flash = owner.flashing
            if flash:
                colour = COL_REWARD if flash == "reward" else COL_PUNISH
                _ns_color(AppKit, colour, 0.16).setFill()
                AppKit.NSRectFillUsingOperation(
                    bounds, AppKit.NSCompositingOperationSourceOver
                )
                _ns_color(AppKit, colour, 0.9).setStroke()
                edge = AppKit.NSBezierPath.bezierPathWithRect_(
                    AppKit.NSInsetRect(bounds, 6, 6)
                )
                edge.setLineWidth_(10.0)
                edge.stroke()

            def font(frac, minimum, weight):
                return AppKit.NSFont.monospacedSystemFontOfSize_weight_(
                    max(minimum, frac * H), weight
                )

            head = font(0.040, 16.0, AppKit.NSFontWeightBold)
            verdict = font(0.055, 22.0, AppKit.NSFontWeightBold)
            small = font(0.017, 10.0, AppKit.NSFontWeightRegular)
            tiny = font(0.0135, 9.0, AppKit.NSFontWeightRegular)

            def attrs(f, colour):
                return {
                    AppKit.NSFontAttributeName: f,
                    AppKit.NSForegroundColorAttributeName: _ns_color(AppKit, colour),
                }

            def measure(s, f):
                return AppKit.NSString.stringWithString_(s).sizeWithAttributes_(
                    {AppKit.NSFontAttributeName: f}
                )

            def text(s, x, y, f, colour):
                AppKit.NSString.stringWithString_(s).drawAtPoint_withAttributes_(
                    AppKit.NSMakePoint(x, y), attrs(f, colour)
                )

            def panel(x, y, w, h, colour=COL_CHROME, radius=0.0):
                _ns_color(AppKit, colour).setFill()
                rect = AppKit.NSMakeRect(x, y, w, h)
                if radius > 0:
                    path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                        rect, radius, radius
                    )
                    path.fill()
                else:
                    AppKit.NSRectFillUsingOperation(
                        rect, AppKit.NSCompositingOperationSourceOver
                    )

            # -- the cards ---------------------------------------------------
            def tilted(rect, angle):
                """``rect`` rotated by ``angle`` radians about its own centre.

                The hand is a fan: Balatro tilts the end cards about +/-5.5
                degrees. When the game has told us each card's rotation there is
                no reason to draw an upright box oversized enough to cover a
                tilted card -- the outline can simply be tilted too, and then it
                sits on the card's own border.
                """
                path = AppKit.NSBezierPath.bezierPathWithRect_(
                    AppKit.NSMakeRect(-rect.size.width / 2.0,
                                      -rect.size.height / 2.0,
                                      rect.size.width, rect.size.height)
                )
                t = AppKit.NSAffineTransform.transform()
                t.translateXBy_yBy_(rect.origin.x + rect.size.width / 2.0,
                                    rect.origin.y + rect.size.height / 2.0)
                # The view is flipped (top-left origin), so a positive game-space
                # angle is a positive Cocoa rotation here.
                t.rotateByRadians_(angle)
                path.transformUsingAffineTransform_(t)
                return path

            for box in ann.boxes:
                r = box.rect
                rect = AppKit.NSMakeRect(r.x * sx, r.y * sy, r.w * sx, r.h * sy)
                rotated = abs(box.angle) > 1e-4
                if box.fill:
                    _ns_color(AppKit, box.color).setFill()
                    if rotated:
                        tilted(rect, box.angle).fill()
                    else:
                        AppKit.NSRectFillUsingOperation(
                            rect, AppKit.NSCompositingOperationSourceOver
                        )
                elif box.thickness > 0:
                    path = (tilted(rect, box.angle) if rotated
                            else AppKit.NSBezierPath.bezierPathWithRect_(rect))
                    path.setLineWidth_(max(1.0, box.thickness * sx))
                    _ns_color(AppKit, box.color).setStroke()
                    path.stroke()
                if box.label:
                    # --labels only. Labels are wider than the pitch, so odd
                    # slots sit one line higher or every other one is unreadable.
                    ly = rect.origin.y - (16 + 17 * (box.slot % 2)) * sy
                    size = measure(box.label, small)
                    panel(rect.origin.x - 3, ly - 2, size.width + 6, size.height + 4,
                          COL_PANEL)
                    text(box.label, rect.origin.x, ly, small, box.color)

            # -- the chrome, in empty felt -----------------------------------
            # One backdrop, sized to its own text, in the band the game leaves
            # blank during a hand. Three things, large: the hand and whether it
            # reaches the blind, what the fly decided, and one dim context line.
            cx = ann.chrome.x * sx
            cy = ann.chrome.y * sy
            gap = max(2.0, 0.006 * H)
            rows = []
            if ann.headline:
                rows.append((ann.headline, head,
                             COL_ENOUGH if ann.headline_ok else COL_SHORT))
            if ann.headline_note:
                rows.append((ann.headline_note, small, COL_DIM))
            rows.append((ann.decision, verdict, COL_TEXT))
            if ann.decision_sub:
                rows.append((ann.decision_sub, small, COL_DIM))
            for line in ann.hud:
                rows.append((line, small, COL_DIM))

            sizes = [measure(s, f) for s, f, _c in rows]
            box_w = min(max(sz.width for sz in sizes), ann.chrome.w * sx)
            box_h = sum(sz.height for sz in sizes) + gap * (len(rows) - 1)
            pad = max(8.0, 0.014 * H)
            panel(cx - pad, cy - pad, box_w + 2 * pad, box_h + 2 * pad,
                  COL_CHROME, radius=pad * 0.6)
            y = cy
            for (line, f, colour), sz in zip(rows, sizes):
                text(line, cx, y, f, colour)
                y += sz.height + gap

            # -- the caption, at the very bottom edge, occluding nothing ------
            size = measure(ann.caption, tiny)
            cap_y = H - size.height - 2.0
            panel(0.0, cap_y - 2.0, size.width + 2 * pad, size.height + 4.0,
                  COL_PANEL)
            text(ann.caption, pad, cap_y, tiny, COL_DIM)

    _VIEW_CLASS = _OverlayView
    return _OverlayView


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _parse_rect(text: str) -> Tuple[float, float, float, float]:
    parts = [float(p) for p in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--rect wants x,y,w,h")
    return (parts[0], parts[1], parts[2], parts[3])


def _demo_decision(geom: HandGeometry) -> Decision:
    """A synthetic hand for ``--calibrate`` with no log and no game.

    It is the hand in ``outputs/realgame/balatro_fly.png`` -- K C, Q H, J D,
    8 H, 8 C, 7 S, 5 S, 5 D -- so pointing ``--rect`` at that screenshot shows
    the boxes over the cards they describe.
    """
    ranks = (11, 10, 9, 6, 6, 5, 3, 3)
    suits = (1, 2, 3, 2, 1, 0, 0, 3)
    return Decision(
        index=0, t=0.0, action_name="select_card[6]",
        action_description="select hand slot 6",
        cards=tuple(Card(i, r, s) for i, (r, s) in enumerate(zip(ranks, suits))),
        selected=(3, 4), best_slots=(3, 4, 6, 7), best_type="Two Pair",
        best_score=92, needed=8, score_bucket="ge1.0", selected_type="Pair",
        selected_is_best=False, relay_active=True, plays=1, discards=2,
        score=292, required=300, blind="Small Blind", ante=1, round=1,
        stage="SELECTING_HAND", policy_mode="calibrate", uses_brain=False,
        dopamine=None, rates={},
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m flybalatro.realgame.overlay",
        description="Transparent detection-style overlay for the real Balatro window.",
    )
    ap.add_argument("--replay", type=Path, default=None,
                    help="step through a recorded log instead of following the live one")
    ap.add_argument("--calibrate", action="store_true",
                    help="draw the card boxes only, from a synthetic full hand")
    ap.add_argument("--rect", type=_parse_rect, default=None,
                    help="x,y,w,h in screen points; skips the Quartz window search")
    ap.add_argument("--titlebar", type=float, default=28.0,
                    help="title-bar height to strip off the Quartz bounds (0 if fullscreen)")
    ap.add_argument("--geometry", type=Path, default=GEOMETRY_PATH)
    ap.add_argument("--pause", type=float, default=1.0, help="seconds per replay step")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after this long (0 = run until interrupted)")
    ap.add_argument("--follow-log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--latest", type=Path, default=DEFAULT_LATEST,
                    help="live feed; latest.json is written atomically by play.py")
    ap.add_argument("--labels", action="store_true",
                    help="write the rank/suit label the encoder used over every "
                         "card. Off by default: the cards are legible on screen "
                         "and eight labels collide. This is the calibration view.")
    ap.add_argument("--all-cards", action="store_true",
                    help="also outline every dealt slot in low-contrast grey; "
                         "by default only the best subset gets a box")
    args = ap.parse_args(argv)

    geom = HandGeometry.load(args.geometry)

    if args.rect is not None:
        rect = args.rect
        source = "--rect"
    else:
        found = balatro_window_bounds()
        if found is None:
            print(
                "Balatro window not found (owner 'Balatro' or 'love').\n"
                "Start the modded game (docs/REALGAME_INSTALL.md section 4), or\n"
                "pass --rect x,y,w,h to place the overlay over something else.",
                file=sys.stderr,
            )
            return 2
        owner, bounds = found
        rect = content_rect(bounds, args.titlebar)
        source = f"{owner} window"
        print(f"found {owner} at {bounds}; canvas {rect}")

    try:
        window = OverlayWindow(rect)
    except ImportError:
        print("pyobjc is missing: uv pip install pyobjc-framework-Cocoa "
              "pyobjc-framework-Quartz", file=sys.stderr)
        return 3

    w, h = rect[2], rect[3]
    deadline = time.monotonic() + args.seconds if args.seconds > 0 else math.inf

    def draw(dec: Decision) -> None:
        window.show(build_annotation(dec, geom, w, h, source=source,
                                     labels=args.labels,
                                     all_cards=args.all_cards or args.labels))
        window.pump(0.02)

    if args.calibrate:
        draw(_demo_decision(geom))
        while time.monotonic() < deadline:
            window.pump(0.1)
        window.close()
        return 0

    if args.replay is not None:
        decisions = read_log(args.replay)
        if not decisions:
            print(f"no decisions with a hand in {args.replay}", file=sys.stderr)
            return 4
        print(f"replaying {len(decisions)} decisions from {args.replay}")
        for dec in decisions:
            if time.monotonic() >= deadline:
                break
            draw(dec)
            end = time.monotonic() + args.pause
            while time.monotonic() < end:
                window.pump(0.05)
        window.close()
        return 0

    # Card-less records (shop, round eval) are followed live, not filtered out:
    # they are how the overlay learns to stop drawing boxes over a screen that
    # no longer has the hand they described.
    feed = (LatestFile(args.latest, with_cards_only=False) if args.latest.exists()
            else LogTail(args.follow_log, with_cards_only=False))
    print(f"following {feed.path} — Ctrl-C to stop")
    try:
        while time.monotonic() < deadline:
            for dec in feed.poll():
                draw(dec)
            window.pump(0.1)
    except KeyboardInterrupt:
        pass
    window.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
