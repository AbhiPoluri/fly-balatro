#!/usr/bin/env python3
"""Measure the static hand-fan model against the *real* cards, in three regimes.

``flybalatro/realgame/overlay.py`` places its boxes from a fitted static model
(``pov_geometry.json``, fitted on one settled 8-card screenshot). This script
asks the honest question a dynamic-alignment path needs answered first:

(A) settled 8-card hand   -- what the model was fitted for
(B) hand with n < 8 cards -- the model *extrapolates* here and has never been
    checked, because every one of the 253 logged decisions has 8 cards or none
(C) hand mid-animation    -- cards still sliding after a deal or a discard

The detector is the one already in ``scripts/pov_calibrate.py``
(:class:`Shot.border_lines`, a sheared-column Hough over Balatro's flat
card-outline colour). Nothing here re-implements it; this module only feeds it
frames, pairs its output into cards, compares against
:class:`~flybalatro.realgame.overlay.HandGeometry`, and times it.

Frames for (B) and (C) come from ``outputs/realgame/balatro_fly.mov``, a
full-desktop screen recording (3024x1964) of the fly playing. The Balatro window
content inside it is **2418x1570 at scale 1.0** -- the same pixels as the
reference canvas -- at desktop offset :data:`CANVAS_IN_MOV`, so measured pixels
are already reference pixels and need no rescaling.

Reproduce the frames first -- into a scratch dir, **not** the repo; they are
~530 MB. 2 fps walks straight past a slide, so these are at the video's native
rate, already cropped to a window around the game canvas::

    S=/tmp/align; mkdir -p $S/seg1 $S/seg2
    ffmpeg -ss 19.0 -to 22.0  -i outputs/realgame/balatro_fly.mov \\
        -vf "crop=2478:1630:274:162" -fps_mode passthrough $S/seg1/s1_%04d.png
    ffmpeg -ss 23.5 -to 28.1 -i outputs/realgame/balatro_fly.mov \\
        -vf "crop=2478:1630:274:162" -fps_mode passthrough $S/seg2/s2_%04d.png

then::

    python scripts/pov_align_measure.py --frames-dir $S --crop-origin 274,162

Importing it has no side effects.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybalatro.realgame import overlay as ov  # noqa: E402
from scripts import pov_calibrate as pc  # noqa: E402

OUT = ROOT / "outputs" / "pov"
SHOTS = ROOT / "outputs" / "realgame"

#: Where the Balatro window's canvas sits inside a frame of ``balatro_fly.mov``.
#: ``(x, y, w, h)`` in desktop pixels. Found from the window's own border --
#: the macOS inner stroke is painted over canvas columns 0-1 and 2416-2417, and
#: its measured centres (302.5, 2718.5) put canvas x=0 at desktop x=302 with
#: the canvas exactly 2418 px wide, i.e. scale 1.0. Cross-correlating the
#: static sidebar UI against the reference screenshot agrees to +-2 px.
CANVAS_IN_MOV: Tuple[int, int, int, int] = (302, 196, 2418, 1570)
#: Honest uncertainty on that origin, in reference pixels. Every .mov number
#: carries it as a *common* offset across all slots of a frame.
REGISTRATION_SLOP_PX: Tuple[int, int] = (2, 3)

#: A wider vertical bracket than the shipped ``pc.BAND``. Mid-slide cards ride
#: above the settled fan; the shipped band can clip their tops. Going below
#: ~1250 drags the Play/Sort/Discard buttons into the white mask, so this stops
#: at 1250.
WIDE_BAND: Tuple[int, int] = (700, 1250)

FONT_PATH = pc.FONT_PATH
COL_MEASURED = (255, 90, 200, 230)
COL_MODEL = (125, 136, 148, 190)
COL_MODEL_HOT = (240, 180, 41, 235)


# --------------------------------------------------------------------------- #
# frames                                                                       #
# --------------------------------------------------------------------------- #
def canvas_from_desktop(path: Path,
                        rect: Tuple[int, int, int, int] = CANVAS_IN_MOV,
                        crop_origin: Tuple[int, int] = (0, 0)) -> Image.Image:
    """Cut the Balatro canvas out of a full-desktop frame.

    ``crop_origin`` is the desktop coordinate of the supplied image's own top
    left, so a frame that ffmpeg already cropped can still be addressed in
    desktop coordinates.
    """
    x, y, w, h = rect
    ox, oy = crop_origin
    im = Image.open(path).convert("RGB")
    return im.crop((x - ox, y - oy, x - ox + w, y - oy + h))


@contextmanager
def band(values: Tuple[int, int]):
    """Run the detector with a different vertical bracket.

    ``pc.BAND`` is read as a module global inside ``Shot.__init__``, so swapping
    it here changes what a ``Shot`` looks at without touching the shipped file.
    """
    old = pc.BAND
    pc.BAND = values
    try:
        yield
    finally:
        pc.BAND = old


def shot_of(img: Image.Image, scratch: Path,
            bounds: Optional[Tuple[int, int]] = None) -> pc.Shot:
    """A :class:`pc.Shot` over ``img``, optionally with a widened band."""
    scratch.parent.mkdir(parents=True, exist_ok=True)
    img.save(scratch)
    if bounds is None:
        return pc.Shot(scratch)
    with band(bounds):
        return pc.Shot(scratch)


# --------------------------------------------------------------------------- #
# pairing the Hough lines into cards                                           #
# --------------------------------------------------------------------------- #
def pair_lines(lines: Sequence[Tuple[int, float, int]],
               card_w: float = 211.0, tol: float = 14.0) -> Dict[str, object]:
    """Describe what ``border_lines()`` returned, without assuming n+1 lines.

    ``Shot.cards()`` assumes every card overlaps its neighbour, so the detector
    sees *n* left borders plus one right border. That is true of a settled
    Balatro fan at any size -- the game keeps a constant ~165 px pitch and does
    **not** spread a short hand -- but it is not true of a card with clear air
    on both sides, which shows both of its own borders. This returns the
    evidence either way so a failure can be named rather than silently chained.
    """
    xs = [int(l[0]) for l in lines]
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    # The last gap of a settled fan is the rightmost card's own width; every
    # other gap is the pitch. Anything else is a card out of formation.
    inner = gaps[:-1] if len(gaps) > 1 else []
    pitch = float(np.median(inner)) if inner else float("nan")
    anomalies = [{"after_line": i, "gap": g, "vs_pitch": round(g - pitch, 1)}
                 for i, g in enumerate(inner) if abs(g - pitch) > tol]
    isolated = [i for i, g in enumerate(gaps)
                if abs(g - card_w) <= tol
                and (i + 2 >= len(xs) or abs(xs[i + 2] - xs[i + 1]) > tol)]
    return {
        "n_lines": len(lines),
        "x": xs,
        "gaps": gaps,
        "deg": [round(float(l[1]), 1) for l in lines],
        "votes": [int(l[2]) for l in lines],
        "median_inner_gap": round(pitch, 1) if inner else None,
        "gap_anomalies": anomalies,
        "n_cardwidth_gaps": len(isolated),
        "tilt_monotone": all(float(lines[i][1]) <= float(lines[i + 1][1]) + 0.6
                             for i in range(len(lines) - 2)),
        "hough_saturated": [round(float(l[1]), 1) for l in lines
                            if abs(float(l[1])) >= pc.HOUGH_DEG - 1e-6],
    }


def cards_from_lines(lines: Sequence[Tuple[int, float, int]],
                     close_ratio: float = 0.85) -> Dict[str, object]:
    """Group the Hough lines into cards *without* assuming n+1 lines.

    ``Shot.cards()`` takes the last line as the rightmost card's right border
    and treats every other line as a left border. That is right for a settled
    fan, where each card is hidden behind its neighbour, and wrong the moment a
    card has clear air to its right: then its own right border shows up as an
    extra line and gets chained into a slot. The tell is the gap: two left
    borders are one pitch apart, so any line closer than ``close_ratio`` of the
    pitch to its predecessor is that predecessor's *right* border, not a new
    card. A card that owns both of its borders also reports its own width,
    which matters because Balatro squashes cards horizontally while they move.
    """
    xs = [int(l[0]) for l in lines]
    if len(xs) < 2:
        return {"n_cards": len(xs), "pitch": None, "cards": [
            {"left": x, "deg": round(float(l[1]), 1), "own_width": None}
            for x, l in zip(xs, lines)]}
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    pitch = float(np.median(gaps[:-1])) if len(gaps) > 1 else float(gaps[0])
    role = ["left"] * len(xs)
    for i in range(1, len(xs)):
        if xs[i] - xs[i - 1] < close_ratio * pitch:
            role[i] = "right"
    role[-1] = "right"
    cards: List[Dict[str, object]] = []
    for i, x in enumerate(xs):
        if role[i] == "right":
            continue
        own = xs[i + 1] - x if i + 1 < len(xs) and role[i + 1] == "right" else None
        cards.append({"left": x, "deg": round(float(lines[i][1]), 1),
                      "own_width": own})
    # The widest card that owns both borders. Not the median and not the last:
    # Balatro squashes a card horizontally while it moves, so a card in motion
    # under-reports its width, and a card is never drawn wider than settled.
    widths = [c["own_width"] for c in cards if c["own_width"]]
    shared = float(max(widths)) if widths else None
    for c in cards:
        w = c["own_width"] or shared
        c["centre_x"] = round(c["left"] + w / 2.0, 1) if w else None
        c["width_used"] = w
    return {"n_cards": len(cards), "pitch": round(pitch, 1),
            "shared_width": shared, "cards": cards}


def honest_compare(geom: ov.HandGeometry, grouped: Dict[str, object],
                   slot_map: Optional[Sequence[int]], n_cards: int,
                   width: int = 2418) -> Dict[str, object]:
    """The grouped cards against their modelled boxes, x only.

    ``slot_map`` says which hand slot each *detected* card belongs to; it is
    the identity unless the detector missed a card, which is exactly what
    happens mid-animation.
    """
    cards = grouped["cards"]
    slots = list(slot_map) if slot_map else list(range(len(cards)))
    boxes = geom.hand_rects(n_cards, width=width, height=1570)
    rows = []
    for card, slot in zip(cards, slots):
        if card["centre_x"] is None:
            continue
        rows.append({"slot": slot, "centre_x": card["centre_x"],
                     "own_width": card["own_width"],
                     "model_centre_x": round(boxes[slot].cx, 1),
                     "dx": round(boxes[slot].cx - float(card["centre_x"]), 1)})
    dxs = [abs(float(r["dx"])) for r in rows]
    return {
        "cards_detected": len(cards),
        "cards_on_screen": n_cards,
        "missed_slots": sorted(set(range(n_cards)) - set(slots)),
        "detector_pitch": grouped["pitch"],
        "slots": rows,
        "max_abs_dx": round(max(dxs), 1) if dxs else None,
        "median_abs_dx": round(float(np.median(dxs)), 1) if dxs else None,
    }


def scan_sequence(paths: Sequence[Path], scratch: Path,
                  crop_origin: Tuple[int, int] = (0, 0),
                  bounds: Optional[Tuple[int, int]] = None) -> Dict[str, object]:
    """Run the detector over a whole stretch of video and tabulate what it saw.

    One hand-picked frame flatters or damns a detector by accident; a run of
    consecutive frames across a deal shows what it actually does. ``cards(n)``
    is then tried for the ``n`` a settled fan would imply (``n_lines - 1``), and
    the frames where it raises are counted.
    """
    rows: List[Dict[str, object]] = []
    tmp = Path(scratch) / "_seq.png"
    for p in paths:
        img = canvas_from_desktop(p, crop_origin=crop_origin)
        sh = shot_of(img, tmp, bounds)
        lines = sh.border_lines()
        info = pair_lines(lines)
        n = len(lines) - 1
        ok, err = False, None
        if n >= 1:
            try:
                pc.measure(sh, n)
                ok = True
            except RuntimeError as exc:
                err = str(exc).split(": ", 1)[-1]
        rows.append({"frame": p.name, "n_lines": info["n_lines"],
                     "gaps": info["gaps"],
                     "n_gap_anomalies": len(info["gap_anomalies"]),
                     "hough_saturated": len(info["hough_saturated"]),
                     "measure_ok": ok, "measure_error": err})
    counts: Dict[str, int] = {}
    for r in rows:
        counts[str(r["n_lines"])] = counts.get(str(r["n_lines"]), 0) + 1
    return {
        "frames": len(rows),
        "line_count_histogram": dict(sorted(counts.items(), key=lambda kv: int(kv[0]))),
        "zero_line_frames": [r["frame"] for r in rows if r["n_lines"] == 0],
        "measure_failed_frames": [r["frame"] for r in rows if not r["measure_ok"]],
        "clean_frames": sum(1 for r in rows
                            if r["n_gap_anomalies"] == 0 and r["hough_saturated"] == 0
                            and r["measure_ok"] and r["n_lines"] >= 4),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# measure vs model                                                             #
# --------------------------------------------------------------------------- #
def measure(shot: pc.Shot, n_cards: int) -> pc.Measured:
    """The shipped measurement path, for ``n_cards`` cards."""
    return pc.measure(shot, n_cards)


def compare(geom: ov.HandGeometry, m: pc.Measured,
            width: int = 2418, height: int = 1570,
            bounds: Optional[Tuple[int, int]] = None) -> Dict[str, object]:
    """Measured card centres vs modelled box centres, per slot.

    Centres, not left edges: the cards are rotated, so a card's bounding-box
    left is not where its outline crosses mid-height, and the box is
    deliberately wider than the card. The centre of a rotated rect is the centre
    of its bounding box, so centres are the comparable quantity (docs/POV.md §3).
    """
    n = len(m.lefts)
    boxes = geom.hand_rects(n, width=width, height=height)
    # Which band the Shot was measured with -- NOT whatever pc.BAND happens to
    # be now, because the band() context manager has already put it back.
    used = bounds if bounds is not None else pc.BAND
    band_mid = (used[0] + used[1]) / 2.0
    rows: List[Dict[str, object]] = []
    for k in range(n):
        box = boxes[k]
        top, bot = m.updown[k]
        cy = (top + bot) / 2.0
        # border_lines() reports x where the outline crosses the *band's* mid
        # height. The outline is tilted, so that is the card's centre only if
        # the band happens to be centred on the cards. Projecting the line down
        # to the card's own mid height removes the band's choice from the
        # answer -- which is what a dynamic aligner should do.
        #
        # The sign: the Hough shears row ``r`` right by ``tan(deg)*(r - mid)``
        # before accumulating, so a line that piles into bin ``X`` satisfies
        # ``x(y) = X - tan(deg)*(y - mid)``. Checked against the raw mask --
        # slot 0's outline is at x=638 at y=960 and x=661 at y=1210 with
        # deg = -5.5. (``pc.render`` draws its pink line with the opposite
        # slant; that is cosmetic and only affects the shipped check images.)
        proj = (m.centres[k]
                - math.tan(math.radians(float(m.angles[k]))) * (cy - band_mid))
        rows.append({
            "slot": k,
            "measured_centre_x": round(m.centres[k], 1),
            "measured_centre_x_tilt_projected": round(proj, 1),
            "dx_tilt_projected": round(box.cx - proj, 1),
            "measured_centre_y": round(cy, 1),
            "measured_top": int(top),
            "measured_bottom": int(bot),
            "measured_tilt_deg": round(float(m.angles[k]), 1),
            "model_centre_x": round(box.cx, 1),
            "model_centre_y": round(box.y + box.h / 2.0, 1),
            "model_box": [round(v, 1) for v in box.as_tuple()],
            "dx": round(box.cx - m.centres[k], 1),
            "dy": round(box.y + box.h / 2.0 - cy, 1),
            "d_top": round(box.y - top, 1),
        })
    dxs = [abs(float(r["dx"])) for r in rows]
    dys = [abs(float(r["dy"])) for r in rows]
    pxs = [abs(float(r["dx_tilt_projected"])) for r in rows]
    return {
        "band_mid_height": band_mid,
        "max_abs_dx_tilt_projected": round(max(pxs), 1),
        "median_abs_dx_tilt_projected": round(float(np.median(pxs)), 1),
        "n_cards": n,
        "card_width_at_midheight": round(m.card_width, 1),
        "measured_pitch": round(
            float(np.polyfit(np.arange(n, dtype=float),
                             np.asarray(m.centres, dtype=float), 1)[0]), 2)
        if n > 1 else None,
        "model_pitch": round(geom.pitch(n) * width, 2),
        "slots": rows,
        "max_abs_dx": round(max(dxs), 1),
        "median_abs_dx": round(float(np.median(dxs)), 1),
        "max_abs_dy": round(max(dys), 1),
        "median_abs_dy": round(float(np.median(dys)), 1),
    }


# --------------------------------------------------------------------------- #
# cost                                                                         #
# --------------------------------------------------------------------------- #
def time_detector(path: Path, repeats: int = 7) -> Dict[str, float]:
    """Best-of-``repeats`` wall times for what a live snap would have to run.

    Screen capture itself is **not** measured: that needs the game running and
    a window id, which this machine deliberately does not have.
    """
    def best(fn) -> float:
        out = []
        for _ in range(repeats):
            t = time.perf_counter()
            fn()
            out.append((time.perf_counter() - t) * 1000.0)
        return min(out)

    decode = best(lambda: np.asarray(Image.open(path).convert("RGB")))
    build = best(lambda: pc.Shot(path))
    shot = pc.Shot(path)
    lines = best(shot.border_lines)
    detected = shot.border_lines()
    n = max(1, len(detected) - 1)
    full = best(lambda: pc.measure(pc.Shot(path), n))
    return {
        "repeats": repeats,
        "png_decode_ms": round(decode, 2),
        "shot_init_ms": round(build, 2),
        "shot_init_minus_decode_ms": round(build - decode, 2),
        "border_lines_ms": round(lines, 2),
        "full_snap_ms": round(full, 2),
        "snap_minus_decode_ms": round(full - decode, 2),
    }


# --------------------------------------------------------------------------- #
# the check image                                                              #
# --------------------------------------------------------------------------- #
def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:  # pragma: no cover
        return ImageFont.load_default()


def draw_check(base: Image.Image, geom: ov.HandGeometry, m: pc.Measured,
               title: str, notes: Sequence[str], out: Path) -> None:
    """Detected outlines (pink) and modelled boxes (grey) over the real frame.

    Same look as ``outputs/pov/overlay_calibration_check*.png``; drawn directly
    rather than through ``build_annotation`` because most of these frames have
    no matching log record -- they are *between* decisions.
    """
    im = base.convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    small, mono, big = _font(17), _font(21), _font(40)
    n = len(m.lefts)

    for k, box in enumerate(geom.hand_rects(n, width=im.width, height=im.height)):
        colour = COL_MODEL_HOT if k in (0, n - 1) else COL_MODEL
        d.rectangle([box.x, box.y, box.right, box.bottom], outline=colour, width=4)
        label = f"slot {k}"
        ly = box.y - 26 - 28 * (k % 2)
        tw = d.textlength(label, font=small)
        d.rectangle([box.x - 3, ly - 3, box.x + tw + 5, ly + 21], fill=ov.COL_PANEL)
        d.text((box.x, ly), label, font=small, fill=colour)

    y0, y1 = 880, 1240
    mid = (y0 + y1) / 2.0
    for k, (x, deg) in enumerate(zip(m.lefts, m.angles)):
        # x(y) = X - tan(deg)*(y - mid); see compare() for why.
        dx = math.tan(math.radians(deg)) * (y1 - mid)
        d.line([x + dx, y0, x - dx, y1], fill=COL_MEASURED, width=3)
        cx = m.centres[k]
        d.line([cx, mid - 30, cx, mid + 30], fill=COL_MEASURED, width=3)
        top, bot = m.updown[k]
        d.line([cx - 22, top, cx + 22, top], fill=COL_MEASURED, width=3)
        d.line([cx - 22, bot, cx + 22, bot], fill=COL_MEASURED, width=3)

    pad = 22
    lines = [(title, mono, ov.COL_TEXT)] + [(t, small, ov.COL_DIM) for t in notes]
    box_h = 26 + 30 * len(lines)
    d.rectangle([pad - 10, pad - 10, pad + 1180, pad + box_h], fill=ov.COL_PANEL)
    for i, (text, fnt, colour) in enumerate(lines):
        d.text((pad, pad + i * 30), text, font=fnt, fill=colour)

    cy = im.height - 44
    d.rectangle([0, cy - 12, im.width, im.height], fill=ov.COL_PANEL)
    d.text((pad, cy), ov.CAPTION, font=small, fill=ov.COL_DIM)
    note = "pink = detected outline, centre and top/bottom · grey/amber = modelled box"
    d.text((im.width - 22 - d.textlength(note, font=small), cy), note,
           font=small, fill=COL_MEASURED)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(im, layer).convert("RGB").save(out)


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
#: The three regimes, and the frame each was measured on. The ``.mov`` frames
#: are named relative to ``--frames-dir``; they were extracted at the video's
#: native rate (~57 fps) already cropped to ``--crop-origin``, because a slide
#: lasts a few hundred ms and 2 fps walks straight past it.
CASES: Dict[str, Dict[str, object]] = {
    "a_settled_8": {
        "label": "A - settled 8-card hand",
        "frame": "seg1/s1_0013.png",
        "t_sec": 19.225,
        "n_cards": 8,
        "image": "align_case_a.png",
        "how": "8/8 counter on screen; border_lines() output identical to "
               "+-1 px over 37 consecutive frames (0.65 s), gaps 163-167 px",
    },
    "b_short_3": {
        "label": "B - settled 3-card hand",
        "frame": "seg2/s2_0079.png",
        "t_sec": 24.875,
        "n_cards": 3,
        "image": "align_case_b.png",
        "how": "3/8 counter on screen; the three cards sit at exactly the same "
               "x for 25 consecutive frames (0.43 s) after a discard",
    },
    "b2_short_3_crosscheck": {
        "label": "B (cross-check) - a second settled 3-card hand",
        "frame": "seg1/s1_0091.png",
        "t_sec": 20.592,
        "n_cards": 3,
        "image": None,
        "how": "a different 3-card hand ~4.3 s earlier, also static for 25 "
               "frames; included so case B does not rest on one frame",
    },
    "c_midslide_8": {
        "label": "C - 8-card hand mid-animation",
        "frame": "seg2/s2_0133.png",
        "t_sec": 25.817,
        "n_cards": 8,
        "image": "align_case_c.png",
        # The detector finds 7 of the 8 cards; the one in the air is missing,
        # so the seven it does find are slots 0-3 and 5-7. Read off the frame
        # by eye (outputs/pov/align_case_c.png): four settled cards on the
        # left, a gap where the flying card is heading, three still sliding.
        "slot_map": [0, 1, 2, 3, 5, 6, 7],
        "caveat": (
            "READ 'detector_only', NOT the max/median in this block. Slot 4 "
            "here is a phantom: Shot.cards(8) chained slot 3's *right* border "
            "into it, so its 'measured centre' 1389.5 and 'measured centre y' "
            "1133.5 are the middle of a gap, and the 78.3 px d_y is not a "
            "measurement of any card. What the detector really did: 7 of the 8 "
            "cards found, the one in the air missed, max |dx| 26.0 px (on the "
            "card squashed to 133 px), median 10.4 px."
        ),
        "how": "8/8 counter, but one card is still in the air mid-frame and "
               "the right-hand group has not closed up: gaps run 133-193 px "
               "against a settled 163-167",
    },
}

#: The reference screenshot the model was fitted on, measured the same way, as
#: the baseline the .mov numbers have to be read against.
BASELINE = {
    "label": "A0 - the fitted reference screenshot",
    "path": SHOTS / "balatro_fly_firsthand.png",
    "n_cards": 8,
}


def _row_line(r: Dict[str, object]) -> str:
    return (f"   slot {r['slot']}  card {float(r['measured_centre_x']):7.1f},"
            f"{float(r['measured_centre_y']):7.1f}"
            f"  box {float(r['model_centre_x']):7.1f},{float(r['model_centre_y']):7.1f}"
            f"  dx {float(r['dx']):+6.1f}  dy {float(r['dy']):+6.1f}"
            f"  tilt {float(r['measured_tilt_deg']):+.1f}")


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-dir", type=Path, required=True,
                    help="scratch dir holding the frames extracted from the .mov")
    ap.add_argument("--crop-origin", default="0,0",
                    help="desktop coordinate of the extracted frames' top left")
    ap.add_argument("--scratch", type=Path, default=None,
                    help="where to drop the canvas crops (default: frames-dir)")
    ap.add_argument("--out", type=Path, default=OUT / "dynamic_align.json")
    args = ap.parse_args(argv)

    crop_origin = tuple(int(v) for v in args.crop_origin.split(","))
    scratch = args.scratch or args.frames_dir
    geom = ov.HandGeometry.load()
    # The model as it was when this measurement started: pitch spread to fill
    # ``span``. ``fixed_pitch`` was added to overlay.py in response to case B
    # below, so both are reported and neither claim is retro-fitted.
    spread = (dataclasses.replace(geom, fixed_pitch=False)
              if hasattr(geom, "fixed_pitch") else None)
    report: Dict[str, object] = {
        "_comment": (
            "Measured card positions vs the static model in pov_geometry.json, "
            "in pixels of the 2418x1570 reference canvas. Cases B and C come "
            "from outputs/realgame/balatro_fly.mov, whose Balatro canvas is "
            "2418x1570 at scale 1.0, so no rescaling is applied."
        ),
        "canvas_in_mov": {"rect": list(CANVAS_IN_MOV), "scale": 1.0,
                          "registration_slop_px": list(REGISTRATION_SLOP_PX)},
        "geometry": {"reference_image": geom.reference_image,
                     "n_reference": geom.n_reference,
                     "fitted_pitch_px": geom.fit.get("pitch_px"),
                     "max_pitch_ratio": geom.max_pitch_ratio,
                     "fixed_pitch": getattr(geom, "fixed_pitch", None),
                     "pitch_px_by_n": {str(n): round(geom.pitch(n) * 2418, 2)
                                       for n in range(2, 9)},
                     "spread_pitch_px_by_n": (
                         {str(n): round(spread.pitch(n) * 2418, 2)
                          for n in range(2, 9)} if spread else None)},
        "cases": {},
        "failures": {},
    }

    # -- the baseline, on the screenshot the model was fitted on ------------
    shot = pc.Shot(BASELINE["path"])
    m = measure(shot, int(BASELINE["n_cards"]))
    base = compare(geom, m)
    base["label"] = BASELINE["label"]
    base["source"] = BASELINE["path"].name
    base["band"] = list(pc.BAND)
    base["lines"] = pair_lines(shot.border_lines())
    base["detector_only"] = honest_compare(
        geom, cards_from_lines(shot.border_lines()), None,
        int(BASELINE["n_cards"]))
    report["cases"]["a0_reference_png"] = base
    print(f"{base['label']}: max|dx| {base['max_abs_dx']} px, "
          f"max|dy| {base['max_abs_dy']} px")
    for r in base["slots"]:
        print(_row_line(r))

    # -- the three regimes, off the recording -------------------------------
    for key, spec in CASES.items():
        src = args.frames_dir / str(spec["frame"])
        img = canvas_from_desktop(src, crop_origin=crop_origin)
        canvas_png = Path(scratch) / f"canvas_{key}.png"
        entry: Dict[str, object] = {
            "label": spec["label"], "source_frame": str(spec["frame"]),
            "t_sec": spec["t_sec"], "identified_by": spec["how"],
            **({"caveat": spec["caveat"]} if spec.get("caveat") else {}),
        }
        for tag, bounds in (("shipped_band", None), ("wide_band", WIDE_BAND)):
            sh = shot_of(img, canvas_png, bounds)
            lines = sh.border_lines()
            grouped = cards_from_lines(lines)
            block: Dict[str, object] = {
                "band": list(pc.BAND if bounds is None else bounds),
                "lines": pair_lines(lines),
                "detector_only": honest_compare(
                    geom, grouped, spec.get("slot_map"), int(spec["n_cards"])),
            }
            try:
                mm = measure(sh, int(spec["n_cards"]))
            except RuntimeError as exc:
                block["error"] = str(exc)
            else:
                block.update(compare(geom, mm, bounds=bounds))
                if spread is not None:
                    alt = compare(spread, mm, bounds=bounds)
                    block["spread_pitch_model"] = {
                        "note": "the model before fixed_pitch: pitch spread to "
                                "fill span, clamped at max_pitch_ratio card "
                                "widths. Identical to the fixed-pitch model at "
                                "n=8; this is what case B was measured against.",
                        "model_pitch": alt["model_pitch"],
                        "max_abs_dx": alt["max_abs_dx"],
                        "median_abs_dx": alt["median_abs_dx"],
                        "dx": [r["dx"] for r in alt["slots"]],
                    }
                if tag == "shipped_band" and spec.get("image"):
                    draw_check(
                        img, geom, mm, f"{spec['label']}  ·  {src.name}  ·  "
                        f"t={spec['t_sec']:.2f}s",
                        [f"max |dx| {block['max_abs_dx']} px   "
                         f"median |dx| {block['median_abs_dx']} px   "
                         f"max |dy| {block['max_abs_dy']} px   "
                         f"median |dy| {block['median_abs_dy']} px",
                         f"detector found {len(lines)} outlines for "
                         f"{spec['n_cards']} cards "
                         f"(a settled overlapping fan gives n+1)",
                         f"measured pitch {block['measured_pitch']} px   "
                         f"modelled pitch {block['model_pitch']} px",
                         str(spec["how"])],
                        OUT / str(spec["image"]))
            entry[tag] = block
        report["cases"][key] = entry
        good = entry["shipped_band"]
        if "max_abs_dx" in good:
            print(f"{spec['label']}: max|dx| {good['max_abs_dx']} px, "
                  f"max|dy| {good['max_abs_dy']} px")
            for r in good["slots"]:
                print(_row_line(r))

    # -- what the detector does across whole animations ---------------------
    import glob as _glob
    for tag, pattern in (("seg1_t19.0-22.0s", "seg1/s1_*.png"),
                         ("seg2_t23.5-28.1s", "seg2/s2_*.png")):
        paths = [Path(p) for p in sorted(_glob.glob(str(args.frames_dir / pattern)))]
        if not paths:
            continue
        seq = scan_sequence(paths, Path(scratch), crop_origin)
        rows = seq.pop("rows")
        (Path(scratch) / f"sequence_{tag}.json").write_text(
            json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        report["failures"][tag] = seq
        print(tag, seq["line_count_histogram"], "clean", seq["clean_frames"],
              "/", seq["frames"])

    report["failures"]["notes"] = [
        "Case C: the card in the air is NOT detected. It is rotated far past "
        "the Hough sweep's +-9 deg and squashed horizontally by Balatro's move "
        "animation, so it casts no near-vertical outline. What Shot.cards(8) "
        "assigns to slot 4 is the *right* border of the card in slot 3, which "
        "has clear air to its right because its neighbour left; the 133 px and "
        "193 px gaps around it are the tell.",
        "Case C: cards in motion are drawn narrower than 211 px (the settled "
        "width). Shot.cards() takes one width from the rightmost pair and "
        "applies it to every slot, so 'centre = left + width/2' is biased for "
        "any squashed card even when its outline is found correctly.",
        "The x a line reports is taken at BAND's mid height. The outlines are "
        "tilted, so widening BAND moves the reported centres: on case A the "
        "shipped band gives max |dx| 4.8 px and BAND=(700,1250) gives 11.8 px, "
        "purely from the band's mid height moving 73 px up. Projecting each "
        "line to its own card's mid height removes that ('tilt_projected').",
        "Shot.cards() assumes a settled overlapping fan: n left borders plus "
        "one right border. That happens to hold for a short hand too, because "
        "Balatro keeps a constant ~165 px pitch and does not spread the fan -- "
        "but it is an assumption, not a check, and case C shows what it does "
        "when it is wrong: it chains a right border into a slot silently.",
        "The zero-line frames are real blackouts, not empty tables. "
        "seg2/s2_0094.png has three cards plus a fourth arriving (the counter "
        "reads 4/8) and border_lines() returns nothing. It is not the colour "
        "mask: that frame has 15739 outline pixels against 13984 on the "
        "settled 3-card frame that works. It is the sweep. The best any column "
        "collects is 110 votes out of ~376 rows, against 272 when settled, "
        "because cards in a deal are rotated well past the +-9 deg the Hough "
        "shears over, so no shear straightens them. Raising HOUGH_DEG would "
        "cost time quadratically and start finding face-card art.",
        "Balatro does NOT spread a short hand: the measured pitch is 165 px at "
        "n=3 exactly as it is at n=8. The model as it stood when this was "
        "measured spread it to min(card_w, (span-card_w)/(n-1)) = 225.4 px for "
        "every n from 3 to 6 and 192 px at n=7, so the boxes walked outwards "
        "from the real cards by (pitch_model - 165)*(k - (n-1)/2): 56-64 px on "
        "the end slots of a 3-card hand. That was a model error, not a "
        "detector error -- the detector found all three cards to a pixel. "
        "pov_geometry.json has since gained fixed_pitch, which brings the same "
        "frames to ~4 px; 'spread_pitch_model' in each case keeps the before "
        "number so the claim is checkable.",
        "pc.render() draws its pink measured line with the slant inverted: the "
        "Hough shears row r right by tan(deg)*(r-mid), so the outline that "
        "piles into bin X runs x(y) = X - tan(deg)*(y-mid), and render draws "
        "x(y) = X + tan(deg)*(y-mid). Checked against the raw mask: slot 0 of "
        "the reference screenshot is at x=638 at y=960 and x=661 at y=1210 "
        "with deg=-5.5. Cosmetic -- it affects only the pink line's lean in "
        "outputs/pov/overlay_calibration_check*.png, not any fitted number -- "
        "and is left alone here rather than silently rewriting those images.",
        "Screen capture is not timed here. The game is not running and this "
        "measurement is offline on recorded pixels, so the numbers below are "
        "the detector only.",
    ]

    report["timing"] = time_detector(Path(scratch) / "canvas_a_settled_8.png")
    print("timing:", report["timing"])

    # -- the one number that would fix case B ------------------------------
    settled_pitch = float(geom.fit.get("pitch_px", 164.65))
    report["recommendation"] = {
        "pitch": {
            "status": ("already applied -- pov_geometry.json carries "
                       "fixed_pitch and HandGeometry.pitch() honours it"
                       if getattr(geom, "fixed_pitch", False) else
                       "NOT applied; the shipped model still spreads the fan"),
            "measured_pitch_px_at_n3": 165.0,
            "fitted_pitch_px_at_n8": settled_pitch,
            "equivalent_max_pitch_ratio": round(
                settled_pitch / (geom.card_w * 2418), 4),
            "why": "Balatro keeps a constant centre-to-centre pitch whatever "
                   "the hand size and centres the shorter fan; spreading it to "
                   "fill span put the end cards of a 3-card hand 56-64 px off.",
        },
        "arc_still_open": {
            "status": "not applied, and not enough material to fit",
            "what": "arc_lift / arc_shrink are indexed by u across the fan, so "
                    "a 3-card hand gets the full parabola over three slots. "
                    "Measured, the end cards of both 3-card hands sit 7-13 px "
                    "BELOW where the model puts their box centre, so the arc "
                    "is too deep at small n.",
            "why_not_fitted": "the only settled short hands in the 28 s "
                              "recording are two 3-card ones. One value of n "
                              "cannot separate 'arc scales with n' from 'arc "
                              "is a fixed pixel amount'.",
        },
        "detector_cost": {
            "status": "throttle, do not run per repaint",
            "what": "the overlay pumps its event queue every 0.1 s; a full "
                    "snap is ~81 ms of Python on top of an unmeasured screen "
                    "capture. Shot.__init__'s per-column loop (44 ms) is the "
                    "larger half and is vectorisable; border_lines() is 36 ms.",
        },
    }
    print("recommendation written")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
