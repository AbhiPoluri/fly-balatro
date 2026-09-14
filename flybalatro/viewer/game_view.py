"""Capture the real Balatro window and stream it into the dashboard.

``python -m flybalatro.viewer.server --source realgame`` already shows the brain
that is deciding. This module puts the **game it is deciding about** in the same
browser window, so the whole demo is one page: the point cloud firing on the
left, Balatro playing small on the right, and the fly's amber outline drawn on
the captured pixels rather than on a second, transparent window floating over
the real screen.

How the pixels are taken
------------------------

``CGWindowListCreateImage(CGRectNull, kCGWindowListOptionIncludingWindow, wid,
kCGWindowImageBoundsIgnoreFraming | kCGWindowImageNominalResolution)``.

Three choices in that call, all of them load-bearing:

``kCGWindowListOptionIncludingWindow`` with an explicit window id captures *that
    window*, not the region of the screen it occupies -- so the frame is right
    even when the browser is on top of the game, which is the normal case here.
``kCGWindowImageBoundsIgnoreFraming`` drops the drop-shadow padding Quartz
    otherwise adds, making the image **exactly** the window bounds. Without it
    the image is 2554x1762 for an 1209x813 window and every rect is displaced.
``kCGWindowImageNominalResolution`` returns points rather than backing-store
    pixels: 1209x813 instead of 2418x1626. The panel is ~500 px wide, the
    stream is 900 px, so the extra resolution is thrown away immediately --
    and taking it costs 118 ms of capture plus 385 ms of JPEG on this M2 Pro
    against 22 ms + 2 ms for the nominal image. Measured, both, in docs/POV.md.

The image still includes the title bar, because the window bounds do. It is
cropped off using the game's own numbers: the mod reports ``screen.height`` (the
LOVE canvas in draw units) on every decision, and at nominal resolution the
captured height is the window height in the same points, so the title bar is
simply ``img_h - screen.height``. That is 28 on the windowed Steam build and 0
in fullscreen, with no constant to guess.

What that buys the boxes
------------------------

After the crop the captured image **is** the LOVE canvas, one image point per
draw unit. The card rects the patched mod reports live in exactly those units
(``docs/POV.md`` section 3), so mapping a rect onto the frame is one scalar --
``image_width / screen.width`` -- and no re-fit, no detector and no fan model is
involved. The rects and the pixels come out of the same game, and (modulo the
capture being taken a few tens of ms after the record) the same moment.

Why it renders into its own bitmap, rather than reading the CGImage
-------------------------------------------------------------------

The obvious route -- ``CGDataProviderCopyData`` on the captured image, then
``Image.frombuffer`` over those bytes -- **leaks 3.85 MB per frame**, measured:
400 captures take the process from 321 MB to 1,862 MB, exactly one window-sized
RGBA buffer each, and neither ``del``, ``gc.collect()`` nor an
``objc.autorelease_pool`` around the call recovers any of it. Taking a Python
buffer over the ``__NSCFData`` is what does it; capturing and copying the data
without ever touching its bytes does not grow at all. At 12 fps that is 46 MB a
second, which would have made the panel unusable within a minute.

So the capture is *drawn* instead: one ``CGBitmapContext`` backed by a
``bytearray`` this module owns, reused for the life of the stream, with
``CGContextDrawImage`` blitting each captured window into it. Nothing
CoreFoundation-owned is ever read from, the allocation happens once, and 200
frames move the process by 15 MB rather than 770. Drawing into a context sized
to the *output* also does the downscale and the title-bar crop in Quartz, in C,
which removes the PIL resize from the per-frame path entirely.

Wire format
-----------

Frames go down the **existing** viewer websocket as binary, beside the spike
sub-steps. A sub-step is ``<II``: sub-step index 0-4 then a count. A game frame
is ``<IIdII``: :data:`FRAME_TAG` (0x80000001, which no sub-step index can be),
a sequence number, the capture time, and the image size, followed by JPEG bytes.
The browser dispatches on the first word, so one socket carries both and the
spike path is untouched.
"""

from __future__ import annotations

import io
import logging
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..realgame.overlay import OWNER_NAMES

LOG = logging.getLogger("flyviewer.game")

__all__ = [
    "FRAME_TAG",
    "HEADER",
    "GameCapture",
    "GameFrame",
    "WindowRef",
    "content_height",
    "game_to_image",
    "is_game_frame",
    "pack_frame",
    "pick_window",
    "unpack_frame",
]

#: First word of a game-frame binary message. Deliberately far outside the
#: 0-4 range a spike sub-step index can take, so ``_onBinary`` can tell the two
#: apart by reading one ``uint32`` and never mistakes a JPEG for a sixth
#: sub-step (which would corrupt the 50 ms wave).
FRAME_TAG: int = 0x80000001

#: ``tag, seq, capture_time, width, height`` -- 24 bytes, no padding.
HEADER: str = "<IIdII"
HEADER_BYTES: int = struct.calcsize(HEADER)

#: Fallback title-bar height in points, used only when the game has not told us
#: its canvas size yet. 28 is what the windowed Steam build measures.
DEFAULT_TITLEBAR: float = 28.0

#: How far the measured title bar is allowed to be from plausible before it is
#: rejected in favour of the default. A fullscreen window gives 0.
MAX_TITLEBAR: float = 80.0

STATE_LIVE: str = "live"
STATE_NO_WINDOW: str = "no_window"
STATE_DENIED: str = "denied"
STATE_UNAVAILABLE: str = "unavailable"
STATE_OFF: str = "off"

#: The sentence the panel prints for each non-live state. Plain, and never a
#: stale frame pretending to be current.
STATE_TEXT: Dict[str, str] = {
    STATE_NO_WINDOW: "Balatro is not running — nothing to capture",
    STATE_DENIED: "screen-recording permission denied — grant it to this "
                  "terminal in System Settings › Privacy & Security",
    STATE_UNAVAILABLE: "window capture is unavailable on this machine "
                       "(pyobjc/Quartz did not import)",
    STATE_OFF: "game view is off",
}


# --------------------------------------------------------------------------- #
# pure parts: window lookup, geometry, framing                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WindowRef:
    """One on-screen window: its Quartz id, its owner and its bounds in points."""

    window_id: int
    owner: str
    bounds: Tuple[float, float, float, float]

    @property
    def area(self) -> float:
        return self.bounds[2] * self.bounds[3]


def pick_window(
    infos: Iterable[Dict[str, Any]],
    owners: Sequence[str] = OWNER_NAMES,
) -> Optional[WindowRef]:
    """The game window out of a ``CGWindowListCopyWindowInfo`` list.

    Same filter ``flybalatro.realgame.overlay.balatro_window_bounds`` uses --
    owner name in ``owners``, ``kCGWindowLayer == 0``, largest area wins -- but
    it also keeps the **window id**, which is what capturing one window by name
    rather than screen-scraping its rectangle requires. Takes plain dicts so it
    is testable without a window server.
    """
    wanted = {str(o).lower() for o in owners}
    best: Optional[WindowRef] = None
    for win in infos or []:
        owner = str(win.get("kCGWindowOwnerName") or "")
        if owner.lower() not in wanted:
            continue
        try:
            if int(win.get("kCGWindowLayer", 0) or 0) != 0:
                continue
            wid = int(win.get("kCGWindowNumber", 0) or 0)
        except (TypeError, ValueError):
            continue
        if wid <= 0:
            continue
        b = win.get("kCGWindowBounds") or {}
        try:
            rect = (float(b.get("X", 0.0)), float(b.get("Y", 0.0)),
                    float(b.get("Width", 0.0)), float(b.get("Height", 0.0)))
        except (TypeError, ValueError):
            continue
        if rect[2] <= 0.0 or rect[3] <= 0.0:
            continue
        ref = WindowRef(window_id=wid, owner=owner, bounds=rect)
        if best is None or ref.area > best.area:
            best = ref
    return best


def content_height(image_height: float, screen_height: Optional[float]) -> float:
    """Height of the LOVE canvas inside a nominal-resolution window capture.

    At nominal resolution the captured image is the window's bounds in points,
    and the window is the canvas plus a title bar. The game reports the canvas
    (``screen.height`` in ``docs/REALGAME_INSTALL.md`` section 2), so the title
    bar is the difference and nothing has to be assumed -- which matters because
    it is 28 windowed and 0 in fullscreen.

    Falls back to :data:`DEFAULT_TITLEBAR` when the game has said nothing yet,
    or when the difference is negative or implausibly large (a stale record from
    a differently-sized window).
    """
    h = float(image_height)
    if screen_height is not None:
        bar = h - float(screen_height)
        if 0.0 <= bar <= MAX_TITLEBAR:
            return float(screen_height)
    return max(1.0, h - DEFAULT_TITLEBAR)


def game_to_image(
    rect: Tuple[float, float, float, float],
    screen_size: Tuple[float, float],
    image_size: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    """A card rect in LOVE draw units -> pixels of the captured content frame.

    One scale factor per axis and nothing else. There is no fit here: the frame
    is the canvas, cropped and downscaled, so ``image_width / screen_width`` is
    the entire transform. The rotation is unchanged -- both axes scale by the
    same ratio whenever the aspect is preserved, which it is.
    """
    sw, sh = float(screen_size[0]), float(screen_size[1])
    if sw <= 0.0 or sh <= 0.0:
        raise ValueError("screen size must be positive")
    sx = float(image_size[0]) / sw
    sy = float(image_size[1]) / sh
    return (rect[0] * sx, rect[1] * sy, rect[2] * sx, rect[3] * sy)


def pack_frame(seq: int, captured_at: float, width: int, height: int,
               jpeg: bytes) -> bytes:
    """One game frame as a binary websocket message."""
    return struct.pack(HEADER, FRAME_TAG, int(seq) & 0xFFFFFFFF,
                       float(captured_at), int(width), int(height)) + jpeg


def is_game_frame(blob: bytes) -> bool:
    """Whether a binary message is a game frame rather than a spike sub-step.

    A sub-step's first word is its index, 0-4. :data:`FRAME_TAG` is not in that
    range, so one word decides it.
    """
    if len(blob) < 4:
        return False
    return struct.unpack_from("<I", blob, 0)[0] == FRAME_TAG


def unpack_frame(blob: bytes) -> Tuple[int, float, int, int, bytes]:
    """``(seq, captured_at, width, height, jpeg)``; raises on anything else."""
    if not is_game_frame(blob):
        raise ValueError("not a game frame")
    if len(blob) < HEADER_BYTES:
        raise ValueError("game frame truncated")
    _tag, seq, t, w, h = struct.unpack_from(HEADER, blob, 0)
    return int(seq), float(t), int(w), int(h), bytes(blob[HEADER_BYTES:])


# --------------------------------------------------------------------------- #
# the capture itself                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class GameFrame:
    """One captured frame, plus what it cost."""

    blob: bytes
    width: int
    height: int
    #: Size of the LOVE canvas this frame is a picture of, in draw units. The
    #: browser maps card rects with it; it is not the same as ``width`` once the
    #: frame has been downscaled.
    canvas: Tuple[float, float]
    capture_ms: float
    encode_ms: float
    captured_at: float

    @property
    def n_bytes(self) -> int:
        return len(self.blob)


class GameCapture:
    """Grabs the Balatro window, downscales it and JPEG-encodes it.

    Quartz and PIL are imported lazily, in :meth:`_quartz`, so this module is
    importable (and its pure parts testable) in a plain interpreter with no
    window server -- the same discipline ``overlay.OverlayWindow`` follows for
    AppKit.

    The window id is cached and re-looked-up whenever a capture comes back
    empty, because restarting the game gives it a new id and a capture loop that
    kept the old one would go dark for the rest of the session.
    """

    def __init__(self, width: int = 900, quality: int = 72,
                 owners: Sequence[str] = OWNER_NAMES) -> None:
        self.width: int = int(width)
        self.quality: int = int(quality)
        self.owners: Tuple[str, ...] = tuple(owners)
        self.window: Optional[WindowRef] = None
        self.state: str = STATE_NO_WINDOW
        self.seq: int = 0
        self.last_error: Optional[str] = None
        self._mods: Optional[Tuple[Any, Any]] = None
        self._checked_permission: bool = False
        self._ctx: Optional[Any] = None
        self._ctx_buf: Optional[Any] = None
        self._ctx_stride: int = 0
        self._ctx_size: Tuple[int, int] = (0, 0)

    # -- lazy imports ------------------------------------------------------
    def _quartz(self) -> Optional[Tuple[Any, Any]]:
        if self._mods is not None:
            return self._mods
        try:
            import Quartz  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the machine
            self.state = STATE_UNAVAILABLE
            self.last_error = str(exc)
            return None
        self._mods = (Quartz, Image)
        return self._mods

    def _permission_ok(self, Quartz: Any) -> bool:
        """``CGPreflightScreenCaptureAccess`` once, so a denial is named.

        Without the permission the capture does not fail -- it returns a picture
        of the desktop wallpaper with the window missing -- so a panel that only
        checked for ``None`` would silently show the wrong thing.
        """
        preflight = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
        if preflight is None:  # pragma: no cover - older macOS
            return True
        try:
            return bool(preflight())
        except Exception:  # pragma: no cover - defensive
            return True

    # -- window ------------------------------------------------------------
    def find_window(self) -> Optional[WindowRef]:
        mods = self._quartz()
        if mods is None:
            return None
        Quartz, _Image = mods
        options = (Quartz.kCGWindowListOptionOnScreenOnly
                   | Quartz.kCGWindowListExcludeDesktopElements)
        infos = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
        self.window = pick_window(infos or [], self.owners)
        return self.window

    # -- one frame ---------------------------------------------------------
    def grab(self, screen_size: Optional[Tuple[float, float]] = None
             ) -> Optional[GameFrame]:
        """One frame, or ``None`` with :attr:`state` saying why.

        ``screen_size`` is the canvas the game last reported; it is used only to
        find the title bar. Everything else about the frame is measured from the
        capture.
        """
        mods = self._quartz()
        if mods is None:
            return None
        Quartz, Image = mods
        if not self._checked_permission or self.state == STATE_DENIED:
            # Re-asked while denied, not once for the life of the process: the
            # user can grant screen recording in System Settings without
            # restarting this, and a panel that stayed dark afterwards would be
            # wrong about its own state. The idle tick is 1 Hz, so this costs
            # one call a second and only while it is failing.
            self._checked_permission = True
            if not self._permission_ok(Quartz):
                self.state = STATE_DENIED
                return None
        if self.window is None and self.find_window() is None:
            self.state = STATE_NO_WINDOW
            return None
        assert self.window is not None

        t0 = time.perf_counter()
        cg = self._capture(Quartz, self.window.window_id)
        if cg is None:
            # A stale id (the game restarted, or the window closed). One
            # re-lookup, then give up until the next tick.
            self.window = None
            if self.find_window() is None:
                self.state = STATE_NO_WINDOW
                return None
            cg = self._capture(Quartz, self.window.window_id)
            if cg is None:
                self.state = STATE_NO_WINDOW
                return None
        img_w = int(Quartz.CGImageGetWidth(cg))
        img_h = int(Quartz.CGImageGetHeight(cg))
        if img_w <= 0 or img_h <= 0:  # pragma: no cover - defensive
            self.state = STATE_NO_WINDOW
            return None
        screen_h = float(screen_size[1]) if screen_size else None
        keep = content_height(img_h, screen_h)
        # Never upscale: a nominal capture is already 1209 px wide and asking
        # for more than that would be inventing pixels and paying to send them.
        out_w = min(self.width, img_w)
        scale = out_w / float(img_w)
        out_h = max(1, int(round(keep * scale)))
        img = self._render(Quartz, Image, cg, out_w, out_h, img_h * scale)
        del cg
        t1 = time.perf_counter()

        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=self.quality, optimize=False,
                 subsampling=1)
        t2 = time.perf_counter()

        self.seq += 1
        self.state = STATE_LIVE
        self.last_error = None
        return GameFrame(
            blob=pack_frame(self.seq, time.time(), out_w, out_h, buf.getvalue()),
            width=out_w,
            height=out_h,
            # The canvas the boxes live in is the *content* of the window at
            # capture resolution -- the frame before it was downscaled. The
            # browser only ever needs the ratio, so any consistent pair works,
            # and this is the pair the game's own numbers are in.
            canvas=(float(img_w), float(keep)),
            capture_ms=(t1 - t0) * 1000.0,
            encode_ms=(t2 - t1) * 1000.0,
            captured_at=time.time(),
        )

    def _render(self, Quartz: Any, Image: Any, cg: Any, out_w: int, out_h: int,
                drawn_h: float) -> Any:
        """Blit the captured window into this stream's own bitmap, downscaled.

        ``drawn_h`` is the whole window scaled to ``out_w``; the context is only
        ``out_h`` tall, so drawing at the origin of a bottom-left coordinate
        system puts the title bar above the top edge, where it is clipped. That
        is the crop -- no second image, no copy.
        """
        ctx, buf, stride = self._context(Quartz, out_w, out_h)
        Quartz.CGContextClearRect(ctx, Quartz.CGRectMake(0, 0, out_w, out_h))
        Quartz.CGContextDrawImage(ctx, Quartz.CGRectMake(0, 0, out_w, drawn_h), cg)
        # A bitmap context's *buffer* is stored top row first even though its
        # coordinate system runs from the bottom left, so the rows come out in
        # PIL's own order and the orientation flag is ``1``. ``BGRX`` because
        # the context is 32-bit little-endian with the alpha byte first;
        # skipping it here means the JPEG encoder never sees a channel it would
        # have to drop.
        return Image.frombuffer("RGB", (out_w, out_h), bytes(memoryview(buf)),
                                "raw", "BGRX", stride, 1)

    def _context(self, Quartz: Any, out_w: int, out_h: int) -> Tuple[Any, Any, int]:
        """The reused bitmap context, rebuilt only when the size changes."""
        if self._ctx is not None and self._ctx_size == (out_w, out_h):
            return self._ctx, self._ctx_buf, self._ctx_stride
        stride = out_w * 4
        buf = bytearray(stride * out_h)
        ctx = Quartz.CGBitmapContextCreate(
            buf, out_w, out_h, 8, stride,
            Quartz.CGColorSpaceCreateDeviceRGB(),
            Quartz.kCGImageAlphaNoneSkipFirst | Quartz.kCGBitmapByteOrder32Little,
        )
        if ctx is None:  # pragma: no cover - would mean an invalid size
            raise RuntimeError(f"could not create a {out_w}x{out_h} bitmap context")
        Quartz.CGContextSetInterpolationQuality(
            ctx, Quartz.kCGInterpolationMedium)
        self._ctx, self._ctx_buf, self._ctx_stride = ctx, buf, stride
        self._ctx_size = (out_w, out_h)
        return ctx, buf, stride

    def _capture(self, Quartz: Any, window_id: int) -> Optional[Any]:
        try:
            return Quartz.CGWindowListCreateImage(
                Quartz.CGRectNull,
                Quartz.kCGWindowListOptionIncludingWindow,
                window_id,
                Quartz.kCGWindowImageBoundsIgnoreFraming
                | Quartz.kCGWindowImageNominalResolution,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.last_error = str(exc)
            return None

    # -- what the panel is told --------------------------------------------
    def describe(self) -> Dict[str, object]:
        return {
            "state": self.state,
            "text": STATE_TEXT.get(self.state, ""),
            "owner": self.window.owner if self.window else None,
            "window_id": self.window.window_id if self.window else None,
            "error": self.last_error,
        }
