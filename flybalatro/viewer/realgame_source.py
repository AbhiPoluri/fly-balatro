"""Drive the viewer from the **real** Balatro game instead of its own headless one.

``python -m flybalatro.viewer.server --source realgame`` puts the point cloud on
the decisions ``flybalatro/realgame/play.py`` is making right now against the
Steam build; ``--replay outputs/realgame/log.jsonl`` puts it on a recorded run so
the brain view can be demonstrated with no game running at all.

Where the spikes come from, and how the screen says which
---------------------------------------------------------

``recorded``
    ``play.py`` ran the window as five 10 ms sub-steps and appended the neuron
    indices that fired in each to ``outputs/realgame/spikes.bin``; the record
    points at the offset. These are the spikes of the decision that was actually
    taken -- nothing is simulated here.
``re-simulated``
    The run predates spike recording (or it was off). Under the ``glomerular32``
    encoding the brain's *entire* input is the 32 relay bits, and those are
    reconstructable from the ``relayed_hand_analysis`` in the record; the brain
    is reset before every window and its seeds are fixed, so re-running it
    reproduces that decision's window rather than approximating it. The UI is
    told which mode each frame is, and says so.

Nothing is ever synthesised from the logged firing *rates*: a rate is not a
spike train and a picture built from one would be a drawing, not a measurement.

What the real game cannot tell us
---------------------------------

The real-game player logs its decision, not its readout, so there are no logits,
no per-slot confidences and no chosen-action confidence; it logs
``n_jokers`` but not joker names, and it logs neither per-card enhancements nor
the 283 game-state bits (which ``glomerular32`` does not send to the brain
anyway). Every one of those is emitted as ``null`` with a line of UI text saying
why, never as a plausible-looking number.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import hands
from ..features_v2 import (
    N_HAND_BLOCK,
    OFF_BEST_MASK,
    OFF_BEST_TYPE,
    OFF_IS_BEST,
    OFF_SCORE_BUCKET,
    OFF_SELECTED_TYPE,
    SCORE_BUCKET_LABELS,
)
from ..realgame.overlay import (
    SUIT_GLYPHS,
    Decision,
    HandGeometry,
    LatestFile,
    LogTail,
    build_annotation,
    read_log,
)
from ..realgame.play import SPIKES_BIN, SPIKES_WANT

__all__ = [
    "RealGameFrames",
    "RealGameSource",
    "pov_block",
    "relay_bits",
    "board_from_record",
    "normalize_mb",
    "read_sidecar",
]

JsonDict = Dict[str, object]

#: The four score-vs-needed buckets, in order, with the words the panel prints.
#: The keys are ``flybalatro.features_v2.SCORE_BUCKET_LABELS``; the labels are
#: the same ones ``flybalatro/viewer/plastic.py`` uses for the headless panel,
#: so one browser render path serves both modes.
BUCKET_LABELS: Tuple[Tuple[str, str], ...] = (
    ("lt0.25", "far behind"),
    ("lt0.5", "behind"),
    ("lt1.0", "close"),
    ("ge1.0", "enough"),
)

#: pylatro ``Stage.int()`` -> label, same table the headless panel uses.
STAGE_LABELS: Tuple[str, ...] = (
    "Pre-blind", "Small Blind", "Big Blind", "Boss Blind", "Cash out", "Shop",
    "Run won", "Run lost", "Tarot", "Spectral", "Booster pack",
)

RANK_NAMES: Tuple[str, ...] = (
    "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A",
)

#: Fields the real-game log does not carry, and the sentence the UI shows in
#: their place. Keep this list and ``docs/REALGAME_INSTALL.md`` section 6 in step.
MISSING: Dict[str, str] = {
    "logits": "the real-game player logs the action it took, not the readout's "
              "logits — no bars, no confidences",
    "jokers": "joker count is logged, names are not",
    "state_bits": "only the 32 relay bits are logged; under glomerular32 they "
                  "are the whole input, and the 283 game-state bits behind them "
                  "are not sent to the brain at all",
    "ante_end": "the run's ante target is not in the per-decision record",
}


# --------------------------------------------------------------------------- #
# the 32 relay bits, back out of the logged analysis                           #
# --------------------------------------------------------------------------- #
def relay_bits(dec: Decision) -> NDArray[np.float32]:
    """Rebuild the exact 32-bit odour the fly was driven with for ``dec``.

    ``flybalatro/features_v2.hand_block`` writes five one-hot / mask fields and
    ``hand_block_summary`` logs the *value* of every one of them, so this is a
    decode of the record rather than a guess. Under ``glomerular32`` these 32
    bits are the brain's entire input, so a window re-run from them is the same
    window.

    The reconstruction is checked against the record's own ``n_bits_on``; a
    mismatch raises rather than quietly feeding the brain a different odour.
    """
    out = np.zeros(N_HAND_BLOCK, dtype=np.float32)
    if not dec.relay_active:
        return out
    relay = dec.raw.get("relayed_hand_analysis") or {}
    best_type = str(relay.get("best_type", ""))
    if best_type in hands.HAND_TYPE_NAMES:
        out[OFF_BEST_TYPE + hands.HAND_TYPE_NAMES.index(best_type)] = 1.0
    for slot, bit in enumerate(relay.get("best_mask") or []):
        if bit and slot < 8:
            out[OFF_BEST_MASK + slot] = 1.0
    selected_type = str(relay.get("selected_type", "none"))
    if selected_type in hands.HAND_TYPE_NAMES:
        out[OFF_SELECTED_TYPE + hands.HAND_TYPE_NAMES.index(selected_type)] = 1.0
    else:
        out[OFF_SELECTED_TYPE + hands.N_HAND_TYPES] = 1.0
    if relay.get("selected_is_best"):
        out[OFF_IS_BEST] = 1.0
    bucket = relay.get("score_bucket")
    if bucket in SCORE_BUCKET_LABELS:
        out[OFF_SCORE_BUCKET + SCORE_BUCKET_LABELS.index(str(bucket))] = 1.0
    expected = relay.get("n_bits_on")
    if expected is not None and int(out.sum()) != int(expected):
        raise ValueError(
            f"decision {dec.index}: rebuilt {int(out.sum())} relay bits but the "
            f"record says {int(expected)} were on"
        )
    return out


# --------------------------------------------------------------------------- #
# the board panel                                                              #
# --------------------------------------------------------------------------- #
def board_from_record(dec: Decision) -> JsonDict:
    """The table panel's dict, from a log record, with holes marked as holes."""
    selected = set(dec.selected)
    cards: List[JsonDict] = []
    for card in dec.cards:
        cards.append({
            "rank_index": card.rank_index,
            "suit_index": card.suit_index,
            "rank": (RANK_NAMES[card.rank_index]
                     if 0 <= card.rank_index < len(RANK_NAMES) else "?"),
            "suit": (SUIT_GLYPHS[card.suit_index]
                     if 0 <= card.suit_index < len(SUIT_GLYPHS) else "?"),
            "red": card.red,
            "selected": card.slot in selected,
            # Not logged. None renders as blank, which is what "unknown" looks
            # like here; it is never filled in with a guess.
            "enhancement": None,
            "chips": None,
        })
    state = dec.raw.get("state") or {}
    n_jokers = int(state.get("n_jokers", 0) or 0)
    stage = dec.raw.get("state", {}).get("stage", 0)
    stage = int(stage) if isinstance(stage, (int, float)) else 0
    return {
        "cards": cards,
        "hand_count": len(cards),
        "hand_size": len(cards),
        "n_selected": len(selected),
        "chips": dec.score,
        "required": dec.required,
        "progress": min(1.0, dec.score / dec.required) if dec.required > 0 else 0.0,
        "plays": dec.plays,
        "discards": dec.discards,
        "money": int(state.get("money", 0) or 0),
        "stage": stage,
        "stage_name": (STAGE_LABELS[stage] if 0 <= stage < len(STAGE_LABELS)
                       else f"stage {stage}"),
        "blind_name": dec.blind or None,
        "ante": dec.ante,
        # The record has no ante target; "?" is the honest thing to print.
        "ante_end": "?",
        "round": dec.round,
        "jokers": ([] if n_jokers == 0
                   else [f"{n_jokers} joker{'s' if n_jokers != 1 else ''} "
                         "· names not logged"]),
    }


# --------------------------------------------------------------------------- #
# the boxes, for the embedded game panel                                       #
# --------------------------------------------------------------------------- #
#: Canvas used when the record carries no ``screen`` block. Under that branch
#: :func:`flybalatro.realgame.overlay.aligned_rects` falls back to the fitted
#: fan, which is expressed in *fractions* of the canvas -- so the number cancels
#: and any positive pair does. It is still sent, because the browser divides by
#: it to get those fractions back.
_FALLBACK_CANVAS: Tuple[float, float] = (1209.0, 785.0)


def pov_block(dec: Decision, geom: HandGeometry) -> JsonDict:
    """The fly's highlight for one decision, in the game's own canvas units.

    This is the **same** :func:`flybalatro.realgame.overlay.build_annotation`
    the transparent NSWindow overlay draws, called on the same record, and it is
    built in the canvas the game reported rather than in overlay-window points.
    The browser then scales it by ``displayed_width / canvas[0]`` onto the
    captured frame -- and because the captured frame *is* that canvas (cropped
    of its title bar and downscaled), the rects and the pixels are the same
    measurement of the same game and the box cannot be "nearly" right.

    ``align`` is carried through unchanged: ``game`` means every rect came from
    the card's own visible LOVE transform for that frame, ``model`` means the
    record predates the patched mod and the fitted fan was used instead. The two
    are not equally trustworthy and the panel says which it is looking at.
    """
    canvas = ((dec.screen.width, dec.screen.height) if dec.screen is not None
              else _FALLBACK_CANVAS)
    ann = build_annotation(dec, geom, canvas[0], canvas[1])
    boxes: List[JsonDict] = []
    for box in ann.boxes:
        if box.thickness <= 0 and not box.fill:
            continue
        boxes.append({
            "x": round(box.rect.x, 2), "y": round(box.rect.y, 2),
            "w": round(box.rect.w, 2), "h": round(box.rect.h, 2),
            "angle": round(float(box.angle), 5),
            "kind": box.kind,
            "fill": bool(box.fill),
            "slot": int(box.slot),
        })
    return {
        "canvas": [round(canvas[0], 2), round(canvas[1], 2)],
        "align": ann.align,
        "boxes": boxes,
        "headline": ann.headline,
        "note": ann.headline_note,
        "ok": bool(ann.headline_ok),
        "decision": ann.decision,
        "decision_sub": ann.decision_sub,
        # True while at least one card is still easing toward its target: the
        # rects are then correct for the frame the record was written on and the
        # capture is a few tens of ms later, so the box can lag the card.
        "moving": bool(dec.any_moving),
        "index": int(dec.index),
    }


# --------------------------------------------------------------------------- #
# the mushroom-body block                                                      #
# --------------------------------------------------------------------------- #
def _finite(value: object) -> object:
    """``None`` for anything that is not a finite number, the value otherwise.

    ``json.dumps`` writes a bare ``NaN`` / ``Infinity`` for those, which is not
    JSON and which ``JSON.parse`` rejects outright -- one empty bucket in one
    record would take the whole viewer down on the frame it arrived. Older
    ``plastic.py`` runs wrote exactly that for a bucket no hand has landed in
    yet, so this runs over every number on its way to the browser.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return None if not math.isfinite(float(value)) else value


def _clean(value: object) -> object:
    """:func:`_finite`, recursively, over dicts and lists."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return _finite(value)


def _spark(raw: object) -> List[float]:
    """A history array with every non-finite sample dropped, not nulled.

    A ``null`` in the middle of a line chart is a hole the renderer would have
    to invent a value for; a short history is simply a short history.
    """
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[float] = []
    for v in raw:
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out.append(float(v))
    return out


def normalize_mb(dec: Decision) -> Optional[JsonDict]:
    """The record's ``mb`` block in the shape the viewer's panel already renders.

    ``flybalatro/realgame/plastic.py`` and ``flybalatro/viewer/plastic.py`` are
    two different loops around the same fly, and they name the same quantities
    differently (``weights.n_synapses`` vs ``weights.n_edges``, a bucket dict vs
    a bucket list, ``mean_ratio_spark`` vs ``weight_history``). Rather than teach
    the browser two schemas, the real-game names are mapped onto the headless
    ones here, so ``TableView.mb`` stays one function with one contract.

    Returns ``None`` for a record with no ``mb`` block at all -- every run
    recorded before the plastic player existed.
    """
    raw = dec.raw.get("mb")
    if not isinstance(raw, dict):
        return None
    kind = str(dec.raw.get("kind") or "decision")
    weights = raw.get("weights") if isinstance(raw.get("weights"), dict) else {}
    assert isinstance(weights, dict)

    buckets: List[JsonDict] = []
    table = raw.get("buckets") if isinstance(raw.get("buckets"), dict) else {}
    assert isinstance(table, dict)
    for key, label in BUCKET_LABELS:
        row = table.get(key) if isinstance(table.get(key), dict) else {}
        assert isinstance(row, dict)
        buckets.append({
            "bucket": key,
            "label": label,
            "n": int(row.get("n", 0) or 0),
            "plays": int(row.get("plays", 0) or 0),
            # NaN for an empty bucket in the older schema, None in the newer;
            # both become None, which the panel prints as an em dash.
            "p_play": _finite(row.get("p_play")),
        })

    dopamine = str(raw.get("dopamine") or ("none" if kind == "decision" else "none"))
    if dopamine not in ("reward", "punish"):
        dopamine = "none"

    # ``p_play`` is this hand's probability and ``p_play_rolling`` is the rolling
    # play rate; runs recorded before the two were split carry only ``p_play``,
    # and there it is the rolling one. Reading it as the per-hand probability
    # would put a number on the P(play) bar that was never the fly's.
    has_rolling = "p_play_rolling" in raw
    p_play = _finite(raw.get("p_play")) if has_rolling else None
    p_rolling = _finite(raw.get("p_play_rolling") if has_rolling else raw.get("p_play"))

    out: JsonDict = {
        "source": "realgame",
        "record": kind,
        "action": str(raw.get("action") or dec.action_name or "?"),
        "explored": bool(raw.get("explored", False)),
        "approach_hz": _finite(raw.get("approach_hz")),
        "avoid_hz": _finite(raw.get("avoid_hz")),
        "raw_drive": _finite(raw.get("raw_drive")),
        "play_drive": _finite(raw.get("play_drive")),
        "p_play": p_play,
        "p_play_rolling": p_rolling,
        "p_play_is_rolling": not has_rolling,
        "bias": _finite(raw.get("bias")),
        "temperature": _finite(raw.get("temperature")),
        "kc_active": _finite(raw.get("kc_active")),
        "kc_spikes": _finite(raw.get("kc_spikes")),
        "hand": {
            "best_type": str(raw.get("best_type") or dec.best_type),
            "best_score": int(raw.get("best_score", dec.best_score) or 0),
            "needed": int(raw.get("needed", dec.needed) or 0),
            "bucket": str(raw.get("bucket") or dec.score_bucket),
            "bucket_label": dict(BUCKET_LABELS).get(
                str(raw.get("bucket") or dec.score_bucket), "—"),
            "discard_ok": bool(raw.get("discard_ok", True)),
            "plays": int(dec.plays),
            "discards": int(dec.discards),
        },
        "selected_slots": [int(s) for s in (raw.get("slots") or dec.selected or [])],
        "dopamine": dopamine,
        "dopamine_eta": _finite(raw.get("eta")),
        "dopamine_changed": _finite(raw.get("n_changed")),
        "dopamine_mean_abs_dw": _finite(raw.get("mean_abs_dw")),
        "dopamine_elig_kc": _finite(raw.get("elig_kc")),
        # The real-game player never freezes its synapses mid-run: a run is
        # either the learning one or the no-delivery control, and which it was
        # is in the pulse counts below, not in a switch the viewer owns.
        "learning": True,
        "pulses": {
            "reward": int(weights.get("n_reward_pulses", 0) or 0),
            "punish": int(weights.get("n_punish_pulses", 0) or 0),
        },
        "weights": {
            "n_edges": int(weights.get("n_synapses", 0) or 0),
            "n_changed": int(weights.get("n_changed", 0) or 0),
            "frac_changed": float(weights.get("frac_changed", 0.0) or 0.0),
            "mean_ratio": float(weights.get("mean_ratio", 1.0) or 0.0),
            "min_ratio": float(weights.get("min_ratio", 1.0) or 0.0),
            "n_at_floor": int(weights.get("n_at_floor", 0) or 0),
            "frac_at_floor": float(weights.get("frac_at_floor", 0.0) or 0.0),
        },
        # Mean weight *as a fraction of its original value*, so the dashed
        # baseline the strip chart draws is 1.0 -- where every synapse started.
        "weight_history": _spark(raw.get("mean_ratio_spark")),
        "weight_ref": 1.0,
        "changed_history": _spark(raw.get("frac_changed_spark")),
        "reward_history": _spark(raw.get("reward_spark")),
        "reward_rate": _finite(raw.get("reward_rate")),
        "hands": int(raw.get("n_hands", 0) or 0),
        "rolling_window": int(raw.get("rolling_window", 50) or 50),
        "buckets": buckets,
    }
    return out


# --------------------------------------------------------------------------- #
# the spike sidecar                                                            #
# --------------------------------------------------------------------------- #
def read_sidecar(path: Path, offset: int, n_bytes: int) -> List[NDArray[np.uint32]]:
    """The sub-step blobs at ``offset``, decoded back to neuron index arrays.

    Raises on a short or malformed read rather than returning a partial window:
    half a wave drawn as a whole one would be a lie about what fired.
    """
    with open(path, "rb") as fh:
        fh.seek(offset)
        blob = fh.read(n_bytes)
    if len(blob) != n_bytes:
        raise ValueError(f"{path}: wanted {n_bytes} bytes at {offset}, got {len(blob)}")
    out: List[NDArray[np.uint32]] = []
    pos = 0
    while pos < len(blob):
        _k, count = struct.unpack_from("<II", blob, pos)
        pos += 8
        end = pos + 4 * count
        if end > len(blob):
            raise ValueError(f"{path}: sub-step at {offset + pos} runs past the window")
        out.append(np.frombuffer(blob, dtype=np.uint32, count=count, offset=pos).copy())
        pos = end
    return out


# --------------------------------------------------------------------------- #
# the source                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class RealGameFrames:
    """One decision's worth of output, in the viewer's own wire shapes."""

    state: Optional[JsonDict] = None
    substeps: List[bytes] = field(default_factory=list)
    decision: Optional[JsonDict] = None
    #: An ``outcome`` record's frame: the dopamine pulse and the synapse state
    #: after it. It carries no board and no spikes, because a pulse is not a
    #: window -- nothing new fired, the fly was paid for the window it already
    #: ran, and drawing a wave here would be inventing one.
    outcome: Optional[JsonDict] = None
    notice: Optional[str] = None


#: ``bits -> (counts, fired point indices per sub-step, seconds)``. This is
#: ``Simulation.run_window``; it is injected rather than imported so this module
#: never imports the server it is imported by.
Resimulator = Callable[
    [NDArray[np.float32]],
    Tuple[NDArray[np.int32], List[NDArray[np.int32]], float],
]


class RealGameSource:
    """Turns ``outputs/realgame`` records into frames the browser already knows.

    Live, it polls ``latest.json`` -- which ``play.py`` writes to a temp file and
    ``os.replace``s, so a poller never sees half an object -- and touches
    ``spikes.want`` so the player knows to record its windows. Both are cheap
    file operations on the player's side (one ``stat`` per decision) and neither
    can block its loop.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        replay: Optional[Path] = None,
        point_index: Optional[NDArray[np.int32]] = None,
        resimulate: Optional[Resimulator] = None,
        want_interval: float = 5.0,
    ) -> None:
        self.out_dir = out_dir
        self.replay_path = replay
        self.point_index = point_index
        self.resimulate = resimulate
        self.want_path = out_dir / SPIKES_WANT
        self.sidecar_path = out_dir / SPIKES_BIN
        self.want_interval = want_interval
        self._last_want = 0.0
        self.n_decisions = 0
        self.n_recorded = 0
        self.n_resimulated = 0
        self.n_outcomes = 0
        self.last_mode: str = "none"
        #: The fitted fan, loaded once. Only the fallback path uses it, but it
        #: has to exist before the first record in case that record predates the
        #: mod's geometry block.
        self.geometry: HandGeometry = HandGeometry.load()
        #: The LOVE canvas the last record was drawn on, in draw units. The
        #: window-capture path needs it to tell the title bar from the game.
        self.last_screen: Optional[Tuple[float, float]] = None
        #: Wall time of the last :meth:`frames` call and an EMA of it, split by
        #: what dominated it. ``emit`` is the part the browser's picture costs --
        #: reading the sidecar, remapping neurons to points, packing the sub-step
        #: blobs and building the JSON; ``spikes`` is the numba window a run with
        #: no sidecar has to re-simulate, which is the brain, not the emit path.
        self.last_frame_ms: float = 0.0
        self.last_emit_ms: float = 0.0
        self.frame_ms_ema: float = 0.0
        self.emit_ms_ema: float = 0.0
        self.max_emit_ms: float = 0.0

        if replay is not None:
            self._replay: List[Decision] = read_log(replay, with_cards_only=False)
            if not self._replay:
                raise ValueError(f"no decisions in {replay}")
            self._cursor = 0
            self._feed = None
        else:
            self._replay = []
            self._cursor = 0
            latest = out_dir / "latest.json"
            self._feed = (LatestFile(latest, with_cards_only=False)
                          if latest.exists()
                          else LogTail(out_dir / "log.jsonl", with_cards_only=False))

    # -- the feed ----------------------------------------------------------
    @property
    def live(self) -> bool:
        return self._feed is not None

    @property
    def feed_path(self) -> Path:
        if self._feed is not None:
            return self._feed.path
        assert self.replay_path is not None
        return self.replay_path

    def request_spikes(self) -> None:
        """Tell a running ``play.py`` that someone is watching.

        Rate-limited to one touch per ``want_interval``; ``play.py`` treats the
        request as live for ``SPIKES_WANT_MAX_AGE`` seconds, so the player stops
        recording on its own shortly after the last browser goes away.
        """
        if not self.live:
            return
        now = time.time()
        if now - self._last_want < self.want_interval:
            return
        self._last_want = now
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.want_path.touch()
        except OSError:  # pragma: no cover - a read-only outputs dir
            pass

    def poll(self) -> List[Decision]:
        """Whatever is new, without blocking."""
        if self._feed is not None:
            self.request_spikes()
            return self._feed.poll()
        if self._cursor >= len(self._replay):
            self._cursor = 0            # loop the recording rather than go dark
        dec = self._replay[self._cursor]
        self._cursor += 1
        return [dec]

    # -- one record -> frames ----------------------------------------------
    def frames(self, dec: Decision) -> RealGameFrames:
        """Frames for one log record, which is a decision or an outcome.

        The plastic player writes two records per hand: the decision (one 50 ms
        window, one play-or-dig call) and the outcome (what the game then paid,
        and the dopamine pulse that followed). Only the first is a window, so
        only the first carries spikes; an outcome that went through the decision
        path would re-simulate a window the fly never ran.
        """
        t0 = time.perf_counter()
        if dec.screen is not None:
            self.last_screen = (dec.screen.width, dec.screen.height)
        if str(dec.raw.get("kind") or "") == "outcome":
            frames = RealGameFrames(outcome=self._outcome_frame(dec))
            self.n_outcomes += 1
            self._observe(t0, t0)
            return frames

        self.n_decisions += 1
        bits = relay_bits(dec)
        t_spikes = time.perf_counter()
        points, counts, mode = self._spikes(dec, bits)
        t_emit = time.perf_counter()
        self.last_mode = mode
        substeps = [
            struct.pack("<II", k, len(idx)) + np.asarray(idx, np.uint32).tobytes()
            for k, idx in enumerate(points)
        ]
        ref = dec.raw.get("spikes") or {}
        total = counts[0] if counts is not None else ref.get("total_spikes")
        n_spiking = counts[1] if counts is not None else ref.get("n_spiking")
        frames = RealGameFrames(
            state=self._state_frame(dec, bits),
            substeps=substeps,
            decision=self._decision_frame(dec, mode, total, n_spiking),
        )
        # A recorded window's sidecar read is part of the emit path (it is what
        # the picture costs); a re-simulated one is the brain and is timed apart.
        self._observe(t0, t_emit if mode == "re-simulated" else t_spikes)
        return frames

    def _observe(self, t0: float, emit_from: float) -> None:
        now = time.perf_counter()
        self.last_frame_ms = (now - t0) * 1000.0
        self.last_emit_ms = (now - emit_from) * 1000.0
        for name, value in (("frame_ms_ema", self.last_frame_ms),
                            ("emit_ms_ema", self.last_emit_ms)):
            prev = getattr(self, name)
            setattr(self, name, value if prev == 0.0 else 0.8 * prev + 0.2 * value)
        self.max_emit_ms = max(self.max_emit_ms, self.last_emit_ms)

    def _spikes(
        self, dec: Decision, bits: NDArray[np.float32]
    ) -> Tuple[List[NDArray[np.uint32]], Optional[Tuple[int, int]], str]:
        """``(point indices per sub-step, (total, distinct) or None, mode)``."""
        ref = dec.raw.get("spikes")
        if isinstance(ref, dict) and self.point_index is not None:
            path = self.out_dir / str(ref.get("file", SPIKES_BIN))
            try:
                neurons = read_sidecar(path, int(ref["offset"]), int(ref["bytes"]))
            except (OSError, KeyError, ValueError, TypeError):
                neurons = None
            if neurons is not None:
                self.n_recorded += 1
                return [self._to_points(idx) for idx in neurons], None, "recorded"
        if self.resimulate is not None and dec.relay_active:
            counts, fired, _secs = self.resimulate(bits)
            self.n_resimulated += 1
            return (list(fired),
                    (int(counts.sum()), int((counts > 0).sum())),
                    "re-simulated")
        return [], None, "none"

    def _to_points(self, neurons: NDArray[np.uint32]) -> NDArray[np.uint32]:
        assert self.point_index is not None
        if len(neurons) == 0:
            return np.zeros(0, dtype=np.uint32)
        points = self.point_index[np.asarray(neurons, np.int64)]
        return points[points >= 0].astype(np.uint32)

    # -- frames ------------------------------------------------------------
    def _state_frame(self, dec: Decision, bits: NDArray[np.float32]) -> JsonDict:
        return {
            "t": "state",
            "episode": 0,
            "step": max(0, dec.index - 1),
            "board": board_from_record(dec),
            "bits": [int(b) for b in bits.astype(np.uint8)],
            "n_bits": int(N_HAND_BLOCK),
            "n_bits_on": int(bits.sum()),
            "n_legal": int(dec.raw.get("n_legal", 0) or 0),
            "relay": dec.raw.get("relayed_hand_analysis"),
            "pov": pov_block(dec, self.geometry),
            "source": "realgame",
        }

    def _decision_frame(
        self,
        dec: Decision,
        mode: str,
        total_spikes: Optional[object],
        n_spiking: Optional[object],
    ) -> JsonDict:
        return {
            "t": "decision",
            "action": int(dec.raw.get("action_index", -1) or -1),
            "action_name": dec.action_name,
            "action_description": dec.action_description,
            # One row, the action taken. has_logits False makes the browser
            # print "pick" rather than a made-up score.
            "top": [{"name": dec.action_name, "logit": 0.0, "chosen": True}],
            "has_logits": False,
            "slots": [],
            "confidence": None,
            "rates": {k.upper(): v for k, v in dec.rates.items()},
            "total_spikes": total_spikes,
            "n_spiking": n_spiking,
            "chips_gained": None,
            "reward": None,
            "brain_ms": round(float(dec.raw.get("sim_seconds", 0.0) or 0.0) * 1000.0, 1),
            "done": False,
            "spike_mode": mode,
            "policy_mode": dec.policy_mode,
            "policy_uses_brain": dec.uses_brain,
            # The mushroom body, in the panel's own names. None for a run with
            # no plastic block -- every recording made before the learning fly
            # existed -- and the panel then simply does not appear.
            "mb": normalize_mb(dec),
            "source": "realgame",
        }

    def _outcome_frame(self, dec: Decision) -> JsonDict:
        """What the game paid for the hand, and the dopamine pulse it bought.

        Deliberately thin: no board, no bits, no spikes. The only new facts an
        outcome record carries are the pulse and the synapse state after it.
        """
        outcome = dec.raw.get("outcome")
        return {
            "t": "outcome",
            "dopamine": dec.dopamine,
            "action_name": dec.action_name,
            "hand": int(dec.raw.get("hand", 0) or 0),
            "outcome": _clean(outcome) if isinstance(outcome, dict) else None,
            "mb": normalize_mb(dec),
            "source": "realgame",
        }

    # -- what the header says ----------------------------------------------
    def describe(self) -> JsonDict:
        return {
            "live": self.live,
            "path": str(self.feed_path),
            "decisions": self.n_decisions,
            "outcomes": self.n_outcomes,
            "recorded": self.n_recorded,
            "resimulated": self.n_resimulated,
            "spike_mode": self.last_mode,
            "missing": dict(MISSING),
            # Records, not decisions: a plastic run writes two per hand.
            "replay_length": len(self._replay) or None,
            "replay_decisions": (
                sum(1 for d in self._replay
                    if str(d.raw.get("kind") or "decision") != "outcome")
                or None),
            "frame_ms": round(self.frame_ms_ema, 2),
            "emit_ms": round(self.emit_ms_ema, 2),
            "max_emit_ms": round(self.max_emit_ms, 2),
        }
