#!/usr/bin/env python3
"""Fit the hand-fan geometry in ``flybalatro/realgame/pov_geometry.json``.

What it does, in order:

1. **Measure** the cards in a real screenshot, from pixels only. The cards are
   *rotated* (the fan runs -5.5 deg to +5.5 deg) so a column-wise search for
   a card's left edge is ill-posed and finds face-card art instead. What is
   well posed is the card's **outline**, which Balatro draws in one flat colour
   (a light blue-grey, ~(186,196,212)): low saturation, clearly darker than the
   white body and lighter than the art. A small Hough transform over that mask
   (shear the band by tan(theta) for theta in -9..+9 deg, sum columns, keep
   the peaks) returns each card's left border as a *line*: its x at the band's
   mid-height plus its angle. The per-card top/bottom still come from the white
   mask's profile over that card's own columns.
2. **Fit** ``left_k = left0 + pitch*k`` by least squares, and the parabolic arc
   ``top_k = top0 - lift*(1 - u^2)`` (``u`` = -1..1 across the fan), on
   ``balatro_fly_firsthand.png``, a *settled* 8-card hand.
3. **Check** the fitted parameters against every screenshot, including
   ``balatro_fly.png``, which caught the hand mid-slide (see ``docs/POV.md``),
   and write the residual table plus annotated PNGs to ``outputs/pov/``.

Run: ``python scripts/pov_calibrate.py``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybalatro.realgame import overlay as ov  # noqa: E402

SHOTS = ROOT / "outputs" / "realgame"
OUT = ROOT / "outputs" / "pov"

#: The fan lives in the bottom half; this band is above the "n/n" hand counter
#: and the Play Hand / Discard buttons, which are also white.
BAND = (860, 1236)
X_RANGE = (600, 2035)
#: Plausible centre-to-centre spacing, in pixels of the 2418-wide canvas. Used
#: only to keep the chain search on card outlines; 165 is the fitted answer.
GAP_RANGE = (120.0, 230.0)
#: Hough sweep. The fan runs about -5.5 to +5.5 degrees; +-9 leaves headroom.
HOUGH_DEG = 9.0
#: A card outline is ~300 px tall, so a real line collects most of that.
HOUGH_MIN_VOTES = 170

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"


# --------------------------------------------------------------------------- #
# measurement                                                                  #
# --------------------------------------------------------------------------- #
class Shot:
    """Pixel measurements of one screenshot's card band."""

    def __init__(self, path: Path) -> None:
        self.path = path
        im = np.asarray(Image.open(path).convert("RGB")).astype(np.int32)
        self.height, self.width, _ = im.shape
        y0, y1 = BAND
        band = im[y0:y1]
        lum = (band[:, :, 0] * 299 + band[:, :, 1] * 587 + band[:, :, 2] * 114) // 1000
        sat = band.max(axis=2) - band.min(axis=2)
        self._white = lum > 150
        # Balatro's card outline, as a flat colour. Art is saturated, the card
        # body is white, the felt is dark: this mask is almost pure outline.
        self._outline = ((sat < 45) & (lum >= 140) & (lum <= 232)).astype(np.int32)
        top = np.full(self.width, -1, dtype=np.int32)
        bot = np.full(self.width, -1, dtype=np.int32)
        for x in range(self.width):
            ys = np.flatnonzero(self._white[:, x])
            if ys.size <= 60:          # a column with a real card in it is tall
                continue
            top[x] = ys[0] + y0
            bot[x] = ys[-1] + y0
        self.top, self.bot = top, bot

    def extent(self) -> Tuple[int, int]:
        """Leftmost and rightmost pixel of the whole fan."""
        xs = np.flatnonzero(
            (self.top >= 0)
            & (np.arange(self.width) >= X_RANGE[0])
            & (np.arange(self.width) < X_RANGE[1])
        )
        return int(xs[0]), int(xs[-1])

    def border_lines(self) -> List[Tuple[int, float, int]]:
        """Every card outline in the band as ``(x at mid-height, degrees, votes)``."""
        mask = self._outline
        rows, W = mask.shape
        mid = rows / 2.0
        best_votes = np.zeros(W, dtype=np.int32)
        best_deg = np.zeros(W)
        for deg in np.arange(-HOUGH_DEG, HOUGH_DEG + 0.01, 0.5):
            t = math.tan(math.radians(float(deg)))
            acc = np.zeros(W, dtype=np.int32)
            for r in range(rows):
                sh = int(round(t * (r - mid)))
                if sh == 0:
                    acc += mask[r]
                elif sh > 0:
                    acc[sh:] += mask[r, : W - sh]
                else:
                    acc[: W + sh] += mask[r, -sh:]
            better = acc > best_votes
            best_deg[better] = float(deg)
            best_votes[better] = acc[better]

        peaks: List[Tuple[int, float, int]] = []
        for x in range(X_RANGE[0] + 10, X_RANGE[1] - 5):
            window = best_votes[x - 9:x + 10]
            if best_votes[x] == window.max() and best_votes[x] >= HOUGH_MIN_VOTES:
                peaks.append((x, float(best_deg[x]), int(best_votes[x])))
        merged: List[Tuple[int, float, int]] = []
        for peak in peaks:
            if merged and peak[0] - merged[-1][0] <= 10:
                if peak[2] > merged[-1][2]:
                    merged[-1] = peak
            else:
                merged.append(peak)
        return merged

    def cards(self, n_cards: int, overrides: Optional[Sequence[int]] = None
              ) -> Tuple[List[int], List[float], float]:
        """``(left border x per card, angle per card, card width)``.

        The last detected line is the rightmost card's *right* border, which is
        what gives the width; the ``n_cards`` left borders are the darkest
        increasing chain of the rest with a plausible gap.
        """
        lines = self.border_lines()
        if len(lines) < n_cards + 1:
            raise RuntimeError(f"{self.path.name}: only {len(lines)} card outlines found")
        right = lines[-1]
        rest = lines[:-1]
        if overrides:
            chosen = [next(l for l in rest if abs(l[0] - x) <= 6) for x in overrides]
        else:
            chain = _best_chain([(l[0], float(l[2])) for l in rest], n_cards)
            if chain is None:
                raise RuntimeError(f"{self.path.name}: no {n_cards}-card outline chain")
            chosen = [next(l for l in rest if l[0] == x) for x in chain]
        width = float(right[0] - chosen[-1][0])
        return [l[0] for l in chosen], [l[1] for l in chosen], width

    def vertical(self, lefts: Sequence[int], card_w: float) -> List[Tuple[int, int]]:
        """Measured ``(top, bottom)`` of each card, from the white mask."""
        x_end = self.extent()[1]
        out: List[Tuple[int, int]] = []
        for k, left in enumerate(lefts):
            nxt = lefts[k + 1] if k + 1 < len(lefts) else x_end + 1
            a, b = int(left) + 4, int(min(left + card_w, nxt)) - 4
            xs = [x for x in range(a, b + 1) if self.top[x] >= 0]
            if not xs:
                raise RuntimeError(f"{self.path.name}: card {k} has no visible columns")
            top = min(int(self.top[x]) for x in xs)
            # The "n/n" hand counter is white, sits just under the middle of the
            # fan and would drag that card's measured bottom 35 px down. Reject
            # columns more than 12 px below this card's own median bottom.
            bots = sorted(int(self.bot[x]) for x in xs)
            median = bots[len(bots) // 2]
            out.append((top, max(b for b in bots if b <= median + 12)))
        return out


def _best_chain(cands: Sequence[Tuple[int, float]], n: int) -> Optional[List[int]]:
    """Strongest increasing chain of ``n`` candidates with gaps in GAP_RANGE."""
    lo, hi = GAP_RANGE
    best_score: Dict[Tuple[int, int], float] = {}
    back: Dict[Tuple[int, int], int] = {}
    for i, (_x, dark) in enumerate(cands):
        best_score[(i, 1)] = dark
    for length in range(2, n + 1):
        for j, (xj, darkj) in enumerate(cands):
            for i, (xi, _) in enumerate(cands):
                if not lo <= xj - xi <= hi:
                    continue
                prev = best_score.get((i, length - 1))
                if prev is None:
                    continue
                score = prev + darkj
                if score > best_score.get((j, length), -1e18):
                    best_score[(j, length)] = score
                    back[(j, length)] = i
    ends = [(s, j) for (j, ln), s in best_score.items() if ln == n]
    if not ends:
        return None
    _, j = max(ends)
    chain = [cands[j][0]]
    length = n
    while length > 1:
        j = back[(j, length)]
        chain.append(cands[j][0])
        length -= 1
    return list(reversed(chain))


# --------------------------------------------------------------------------- #
# fit                                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Measured:
    """What the pixels say about one screenshot's hand."""

    shot: Shot
    lefts: List[int]
    angles: List[float]
    card_width: float          # card width measured at the band's mid-height
    updown: List[Tuple[int, int]]

    @property
    def centres(self) -> List[float]:
        """Each card's centre x. Rotation-invariant, so this is the quantity to
        compare a box against: the centre of a rotated rect is the centre of
        its bounding box."""
        return [x + self.card_width / 2.0 for x in self.lefts]


def measure(shot: Shot, n: int = 8,
            overrides: Optional[Sequence[int]] = None) -> Measured:
    lefts, angles, width = shot.cards(n, overrides)
    return Measured(shot, lefts, angles, width, shot.vertical(lefts, width))


def fit(m: Measured) -> Tuple[ov.HandGeometry, Dict[str, float]]:
    shot = m.shot
    n = len(m.lefts)
    k = np.arange(n, dtype=float)
    pitch, centre0 = np.polyfit(k, np.asarray(m.centres, dtype=float), 1)

    # The box has to *cover* a rotated card, so its width comes from the fan's
    # own pixel extent rather than from the card's width at mid-height.
    x0, x1 = shot.extent()
    card_w = float((x1 - x0 + 1) - pitch * (n - 1))

    u = (k - (n - 1) / 2.0) / ((n - 1) / 2.0)
    bump = 1.0 - u * u
    tops = np.array([t for t, _ in m.updown], dtype=float)
    bots = np.array([b for _, b in m.updown], dtype=float)
    ends = bump < 1e-9
    top0, bot0 = float(tops[ends].mean()), float(bots[ends].mean())
    mid = ~ends
    lift = float(np.median((top0 - tops[mid]) / bump[mid]))
    drop = float(np.median((bot0 - bots[mid]) / bump[mid]))

    W, H = float(shot.width), float(shot.height)
    span = pitch * (n - 1) + card_w
    geom = ov.HandGeometry(
        center_x=(centre0 + pitch * (n - 1) / 2.0) / W,
        top_y=top0 / H,
        card_w=card_w / W,
        card_h=(bot0 - top0) / H,
        span=span / W,
        arc_lift=lift / H,
        arc_shrink=(drop - lift) / H,
        raise_frac=0.030,
        max_pitch_ratio=1.0,
        reference_w=int(W),
        reference_h=int(H),
        reference_image=shot.path.name,
        n_reference=n,
    )
    stats = {"pitch_px": float(pitch), "centre0_px": float(centre0),
             "box_w_px": card_w, "card_w_at_midheight_px": m.card_width,
             "top0_px": top0, "bot0_px": bot0,
             "arc_lift_px": lift, "arc_drop_px": drop,
             "fan_angle_min_deg": min(m.angles), "fan_angle_max_deg": max(m.angles)}
    return geom, stats


def residuals(geom: ov.HandGeometry, m: Measured) -> Dict[str, object]:
    """Modelled box vs measured card, per slot.

    ``d_centre`` is the honest headline: the horizontal offset between the box
    and the card it is drawn on. Left edges are not comparable, because the cards are
    rotated, so a card's bounding-box left is not where its outline crosses
    mid-height, and the box is deliberately wider than the card.
    """
    shot = m.shot
    n = len(m.lefts)
    modelled = geom.hand_rects(n, width=shot.width, height=shot.height)
    rows = []
    for k in range(n):
        p = modelled[k]
        top, bot = m.updown[k]
        rows.append({
            "slot": k,
            "measured_centre": round(m.centres[k], 1),
            "measured_top": top,
            "measured_bottom": bot,
            "measured_angle_deg": m.angles[k],
            "modelled_box": [round(v, 1) for v in p.as_tuple()],
            "d_centre": round(p.cx - m.centres[k], 1),
            "d_top": round(p.y - top, 1),
            "d_bottom": round(p.bottom - bot, 1),
        })
    return {
        "image": shot.path.name,
        "canvas": [shot.width, shot.height],
        "extent": list(shot.extent()),
        "card_width_at_midheight": m.card_width,
        "cards": rows,
        "max_abs_d_centre": round(max(abs(r["d_centre"]) for r in rows), 1),
        "max_abs_d_top": round(max(abs(r["d_top"]) for r in rows), 1),
    }


# --------------------------------------------------------------------------- #
# the check images                                                             #
# --------------------------------------------------------------------------- #
def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:  # pragma: no cover - macOS always has Menlo
        return ImageFont.load_default()


def render(ann: ov.Annotation, base: Image.Image,
           measured: Optional["Measured"] = None) -> Image.Image:
    """Draw an annotation onto a copy of a screenshot.

    Same geometry the NSWindow uses (``build_annotation`` produced these
    rects) plus, in pink, what the pixels say: each card's measured
    outline as the slanted line it is, and a tick at its measured centre.
    """
    im = base.convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    small, mono, big = _font(17), _font(21), _font(52)

    if ann.flash:
        colour = ov.COL_REWARD if ann.flash == "reward" else ov.COL_PUNISH
        d.rectangle([0, 0, im.width - 1, im.height - 1], outline=colour, width=14)

    if measured is not None:
        y0, y1 = 900, 1200
        mid = (y0 + y1) / 2.0
        for k, (x, deg) in enumerate(zip(measured.lefts, measured.angles)):
            dx = math.tan(math.radians(deg)) * (y1 - mid)
            d.line([x - dx, y0, x + dx, y1], fill=(255, 90, 200, 220), width=3)
            cx = measured.centres[k]
            d.line([cx, mid - 26, cx, mid + 26], fill=(255, 90, 200, 220), width=3)

    for box in ann.boxes:
        r = box.rect
        if box.fill:
            d.rectangle([r.x, r.y, r.right, r.bottom], fill=box.color)
        elif box.thickness > 0:
            d.rectangle([r.x, r.y, r.right, r.bottom], outline=box.color,
                        width=box.thickness)
        if box.label:
            ly = r.y - 26 - 28 * (box.slot % 2)
            tw = d.textlength(box.label, font=small)
            d.rectangle([r.x - 3, ly - 3, r.x + tw + 5, ly + 21], fill=ov.COL_PANEL)
            d.text((r.x, ly), box.label, font=small, fill=box.color)

    # These images are a static document, not an overlay on a live game, so the
    # chrome stays in the corner where it does not sit over the cards.
    lines = [(ann.headline, mono, ov.COL_ENOUGH if ann.headline_ok else ov.COL_SHORT)]
    if ann.headline_note:
        lines.append((ann.headline_note, small, ov.COL_DIM))
    lines.extend((line, small, ov.COL_DIM) for line in ann.hud)
    pad = 22
    box_h = 26 + 30 * len(lines)
    d.rectangle([pad - 10, pad - 10, pad + 700, pad + box_h], fill=ov.COL_PANEL)
    for i, (line, fnt, colour) in enumerate(lines):
        d.text((pad, pad + i * 30), line, font=fnt, fill=colour)

    dy = im.height - 190
    d.rectangle([pad - 10, dy - 12, pad + 760, dy + 86], fill=ov.COL_PANEL)
    d.text((pad, dy), ann.decision, font=big, fill=ov.COL_TEXT)
    d.text((pad, dy + 60), ann.decision_sub, font=small, fill=ov.COL_DIM)

    cy = im.height - 44
    d.rectangle([0, cy - 12, im.width, im.height], fill=ov.COL_PANEL)
    d.text((pad, cy), ann.caption, font=small, fill=ov.COL_DIM)
    if measured is not None:
        note = "pink = measured card outline and centre · grey/amber = modelled box"
        d.text((im.width - 22 - d.textlength(note, font=small), cy), note,
               font=small, fill=(255, 90, 200, 220))
    return Image.alpha_composite(im, layer).convert("RGB")


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
#: Only needed if the outline chain ever picks the wrong line; see docs/POV.md.
#: Both shipped screenshots resolve unambiguously, so this is empty.
OVERRIDES: Dict[str, Sequence[int]] = {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detect", action="store_true",
                    help="print every detected card outline and stop")
    ap.add_argument("--no-write", dest="write", action="store_false", default=True)
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    fit_shot = Shot(SHOTS / "balatro_fly_firsthand.png")
    check_shot = Shot(SHOTS / "balatro_fly.png")

    if args.detect:
        for shot in (fit_shot, check_shot):
            print(shot.path.name, "extent", shot.extent())
            for x, deg, votes in shot.border_lines():
                print(f"   x={x:5d} angle={deg:+5.1f} deg  votes={votes}")
        return 0

    fitted = measure(fit_shot, 8, OVERRIDES.get(fit_shot.path.name))
    checked = measure(check_shot, 8, OVERRIDES.get(check_shot.path.name))
    for m in (fitted, checked):
        print(m.shot.path.name, "outlines", m.lefts,
              "angles", [round(a, 1) for a in m.angles],
              "card width", m.card_width)

    geom, stats = fit(fitted)
    print("fit:", {k: round(v, 2) for k, v in stats.items()})

    payload = {
        "_comment": (
            "Balatro hand-fan geometry, as fractions of the game canvas. Fitted "
            "by scripts/pov_calibrate.py; see docs/POV.md. 'raise' is NOT "
            "measured -- the API never highlights the fly's virtual selection."
        ),
        "reference": {
            "image": geom.reference_image,
            "width": geom.reference_w,
            "height": geom.reference_h,
            "n_cards": geom.n_reference,
        },
        "hand": {
            "center_x": round(geom.center_x, 6),
            "top_y": round(geom.top_y, 6),
            "card_w": round(geom.card_w, 6),
            "card_h": round(geom.card_h, 6),
            "span": round(geom.span, 6),
            "arc_lift": round(geom.arc_lift, 6),
            "arc_shrink": round(geom.arc_shrink, 6),
            "raise": round(geom.raise_frac, 6),
            "max_pitch_ratio": geom.max_pitch_ratio,
        },
        "fit": {k: round(v, 2) for k, v in stats.items()},
    }
    geom = ov.HandGeometry.from_dict(payload)  # round-trip: ship what we test

    report = {
        "fit_image": fit_shot.path.name,
        "residuals": [residuals(geom, fitted), residuals(geom, checked)],
    }
    for r in report["residuals"]:
        print(f"{r['image']}: max |d_centre| {r['max_abs_d_centre']} px, "
              f"max |d_top| {r['max_abs_d_top']} px")
        for row in r["cards"]:
            print(f"   slot {row['slot']}  card centre {row['measured_centre']:7.1f}"
                  f"  box centre {row['modelled_box'][0] + row['modelled_box'][2] / 2:7.1f}"
                  f"  dC {row['d_centre']:+6.1f}  dT {row['d_top']:+5.1f}"
                  f"  tilt {row['measured_angle_deg']:+.1f} deg")

    if args.write:
        ov.GEOMETRY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (OUT / "measured_rects.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")
        print("wrote", ov.GEOMETRY_PATH, "and", OUT / "measured_rects.json")

    # Pick the log record that *is* each screenshot, by matching the dealt hand
    # (rank_index, suit_index) against the glyphs read off the pixels. The
    # `decision` counter restarts every run and log.jsonl holds all three, so
    # indexing by it would pick the wrong run.
    HANDS = {
        "balatro_fly.png": (
            ((11, 1), (10, 2), (9, 3), (6, 2), (6, 1), (5, 0), (3, 0), (3, 3)), 292),
        "balatro_fly_firsthand.png": (
            ((12, 3), (9, 1), (7, 3), (6, 0), (5, 1), (1, 3), (0, 2), (0, 3)), 0),
    }
    decisions = ov.read_log(ROOT / "outputs" / "realgame" / "log.jsonl")

    def record_for(name: str) -> ov.Decision:
        want, score = HANDS[name]
        for d in decisions:
            if tuple((c.rank_index, c.suit_index) for c in d.cards) == want and d.score == score:
                return d
        raise RuntimeError(f"no log record matches {name}")

    for m, name in ((checked, "overlay_calibration_check.png"),
                    (fitted, "overlay_calibration_check_settled.png")):
        shot = m.shot
        base = Image.open(shot.path)
        # The check images are the calibration story: every measured card next
        # to its modelled box and the label the encoder produced for it. That is
        # the one place the dense view still earns its keep, so ask for it
        # explicitly; the live overlay's default is one box and no labels.
        ann = ov.build_annotation(record_for(shot.path.name), geom,
                                  shot.width, shot.height, source=shot.path.name,
                                  labels=True, all_cards=True)
        render(ann, base, m).save(OUT / name)
        print("wrote", OUT / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
