#!/usr/bin/env python
"""Do the fly's boxes land on the cards, in the frame the dashboard shows?

The embedded game panel draws the mod-reported card rects onto a frame captured
from the same window (``flybalatro/viewer/game_view.py``). Both come out of the
same game, so there is no fit to be wrong, but the *mod's* arithmetic from
``Card.VT`` to LOVE pixels had never been executed against real pixels until the
panel existed, and this measures it.

Method, per settled hand:

1. ask the running mod for the game state twice and keep it only if every card
   rect is stable between the reads (the hand is not mid-deal);
2. capture the window at **full** resolution, crop the title bar. The content is
   then 2418x1570, the same canvas ``scripts/pov_calibrate.py`` was tuned on,
   so its outline detector applies unchanged;
3. run that detector: each card's left border as a line, and the rightmost
   card's right border, which is where the measured card width comes from;
4. compare the detected card centres and tops with the reported rect's.

Numbers are pixels of the 2418-wide canvas. Divide by 2.69 for the 900 px
stream the browser is sent, or by 5.14 for the 470 CSS px the panel is drawn at
in a 1440x900 window.

    python scripts/pov_embed_measure.py --hands 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pov_calibrate as pc  # noqa: E402
from flybalatro.realgame.client import BalatroBot  # noqa: E402
from flybalatro.viewer.game_view import content_height, pick_window  # noqa: E402

OUT = ROOT / "outputs" / "pov"


def capture_full(window_id: int) -> Image.Image:
    """The window at backing-store resolution, framing excluded.

    Nominal resolution is what the viewer streams; this takes the full 2418 px
    one purely so the detector's tuning applies without rescaling it.
    """
    from Quartz import (CGDataProviderCopyData, CGImageGetBytesPerRow,
                        CGImageGetDataProvider, CGImageGetHeight,
                        CGImageGetWidth, CGRectNull,
                        CGWindowListCreateImage,
                        kCGWindowImageBoundsIgnoreFraming,
                        kCGWindowListOptionIncludingWindow)
    cg = CGWindowListCreateImage(CGRectNull, kCGWindowListOptionIncludingWindow,
                                 window_id, kCGWindowImageBoundsIgnoreFraming)
    w, h = CGImageGetWidth(cg), CGImageGetHeight(cg)
    stride = CGImageGetBytesPerRow(cg)
    data = CGDataProviderCopyData(CGImageGetDataProvider(cg))
    img = Image.frombuffer("RGB", (stride // 4, h), bytes(data), "raw", "BGRX",
                           stride, 1)
    return img.crop((0, 0, w, h))


def rects_of(state: Dict) -> Optional[List[Dict]]:
    cards = ((state.get("hand") or {}).get("cards")) or []
    geoms = [c.get("geometry") for c in cards]
    if not cards or not all(g and g.get("rect") for g in geoms):
        return None
    return [dict(g["rect"], scale=float((g.get("vt") or {}).get("scale", 1.0)))
            for g in geoms]


def settled(bot: BalatroBot, window_id: int, tol: float = 0.3):
    """``(rects, screen, content_image)`` for a hand that is not animating."""
    while True:
        a = bot.call("gamestate")
        ra = rects_of(a)
        if ra is None or len(ra) != 8:
            time.sleep(0.2)
            continue
        img = capture_full(window_id)
        b = bot.call("gamestate")
        rb = rects_of(b)
        if rb is None or len(rb) != len(ra):
            continue
        if any(abs(p["x"] - q["x"]) > tol or abs(p["y"] - q["y"]) > tol
               for p, q in zip(ra, rb)):
            continue
        screen = a["screen"]
        keep = content_height(img.height, screen["pixel_height"])
        content = img.crop((0, int(round(img.height - keep)), img.width, img.height))
        return ra, screen, content


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hands", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--json", type=Path, default=OUT / "embed_align.json")
    args = ap.parse_args(argv)

    from Quartz import (CGWindowListCopyWindowInfo, kCGNullWindowID,
                        kCGWindowListExcludeDesktopElements,
                        kCGWindowListOptionOnScreenOnly)
    infos = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID)
    ref = pick_window(infos or [])
    if ref is None:
        print("Balatro is not running")
        return 1
    bot = BalatroBot()
    rows: List[Dict] = []
    deadline = time.time() + args.timeout
    while len(rows) < args.hands and time.time() < deadline:
        rects, screen, content = settled(bot, ref.window_id)
        path = OUT / f"embed_align_{len(rows)}.png"
        content.save(path)
        try:
            m = pc.measure(pc.Shot(path), len(rects))
        except Exception as exc:
            print("  detector:", exc)
            continue
        s = content.width / screen["width"]
        dx, dy, dw = [], [], []
        for i, r in enumerate(rects):
            cx = (r["x"] + r["w"] / 2) * s
            # The card is drawn at VT.scale about its own centre, so the box
            # the mod reports (the unscaled VT rect) is larger than the card by
            # 1/scale. Compare the tops after taking that out, or the residual
            # is dominated by a size difference rather than a placement one.
            drawn_h = r["h"] * s * r["scale"]
            top = (r["y"] + r["h"] / 2) * s - drawn_h / 2
            dx.append(cx - m.centres[i])
            dy.append(top - m.updown[i][0])
            dw.append(r["w"] * s * r["scale"] - m.card_width)
        print(f"[{len(rows)}] pitch reported "
              f"{(rects[-1]['x'] - rects[0]['x']) * s / (len(rects) - 1):.2f} px, "
              f"detected {(m.centres[-1] - m.centres[0]) / (len(rects) - 1):.2f} px")
        print("    dx", [round(v, 1) for v in dx])
        print("    dy", [round(v, 1) for v in dy])
        print("    card width reported*scale %.1f detected %.1f"
              % (rects[0]["w"] * s * rects[0]["scale"], m.card_width))
        rows.append({"dx": dx, "dy": dy, "dw": dw,
                     "pitch_reported": (rects[-1]["x"] - rects[0]["x"]) * s / (len(rects) - 1),
                     "pitch_detected": (m.centres[-1] - m.centres[0]) / (len(rects) - 1),
                     "image": path.name})
        time.sleep(1.0)

    if rows:
        dx = np.array([v for r in rows for v in r["dx"]])
        dy = np.array([v for r in rows for v in r["dy"]])
        print("\n%d hands / %d cards, pixels of the 2418 canvas:" % (len(rows), dx.size))
        print("  |dx| max %.1f median %.1f    |dy| max %.1f median %.1f"
              % (np.abs(dx).max(), np.median(np.abs(dx)),
                 np.abs(dy).max(), np.median(np.abs(dy))))
        print("  pitch reported %.2f vs detected %.2f (%.2f%% short)"
              % (np.mean([r["pitch_reported"] for r in rows]),
                 np.mean([r["pitch_detected"] for r in rows]),
                 100 * (1 - np.mean([r["pitch_reported"] for r in rows])
                        / np.mean([r["pitch_detected"] for r in rows]))))
        args.json.write_text(json.dumps(rows, indent=1))
        print("  ->", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
