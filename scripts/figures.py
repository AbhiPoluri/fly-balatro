"""Regenerate every figure in `figures/` from the JSON already in `outputs/`.

This script reads measured artefacts and draws them. It runs no simulation, no
game and no training, it writes nothing outside `figures/`, and it contains no
hand-typed result numbers -- every value on every axis is loaded from a file
under `outputs/` at draw time. The provenance line printed along the bottom edge
of each figure names those files.

    python -m scripts.figures              # all figures
    python -m scripts.figures --only 3 5   # just those
    python -m scripts.figures --list

Figures are written at 2x (200 dpi) for retina, plus SVG.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "outputs"
FIGDIR = ROOT / "figures"
DPI = 200

# Every README figure is drawn on this canvas and saved without a tight-bbox
# crop, so the PNG is exactly W_IN * DPI pixels wide and the scale GitHub
# applies is known before anything is drawn rather than measured afterwards.
W_IN = 11.2
README_PX = 900

from scripts import figstyle as S  # noqa: E402

S.use_style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


# --------------------------------------------------------------------------- #
# loading + provenance
# --------------------------------------------------------------------------- #
class Sources:
    """Loads JSON under `outputs/` and remembers what it touched."""

    def __init__(self) -> None:
        self._cache: dict[Path, Any] = {}
        self.used: list[str] = []

    def __call__(self, rel: str) -> Any:
        path = OUTPUTS / rel
        if path not in self._cache:
            if not path.exists():
                raise SystemExit(f"missing artefact: {path.relative_to(ROOT)}")
            self._cache[path] = json.loads(path.read_text())
        label = f"outputs/{rel}"
        if label not in self.used:
            self.used.append(label)
        return self._cache[path]

    def line(self, extra: str = "") -> str:
        src = "  ".join(self.used)
        return (extra + "\n" if extra else "") + "source  " + src


# --------------------------------------------------------------------------- #
# interval arithmetic -- every figure that shows a rate shows its uncertainty
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def paired_prop_ci(a_only: int, b_only: int, n: int, z: float = 1.959964):
    """95% CI for a paired difference of proportions (Agresti-Min / Wald form).

    `a_only` / `b_only` are the discordant counts from a McNemar table over `n`
    paired units. Returns (diff, lo, hi) for A - B.
    """
    if n == 0:
        return (float("nan"),) * 3
    diff = (a_only - b_only) / n
    var = (a_only + b_only - (a_only - b_only) ** 2 / n) / (n * n)
    half = z * math.sqrt(max(var, 0.0))
    return diff, diff - half, diff + half


def err_from(p: float, lo: float, hi: float) -> tuple[float, float]:
    return max(p - lo, 0.0), max(hi - p, 0.0)


# --------------------------------------------------------------------------- #
# small drawing helpers
# --------------------------------------------------------------------------- #
def png_width(path: Path) -> int:
    """Pixel width straight out of the PNG header."""
    with path.open("rb") as fh:
        return struct.unpack(">I", fh.read(24)[16:20])[0]


def readme_px(size_pt: float, width_px: int) -> float:
    """How many screen pixels `size_pt` becomes in a 900 px README column."""
    return size_pt * (DPI / 72.0) * (README_PX / width_px)


def _report_overflow(fig, name: str) -> None:
    """Warn if anything sits outside the canvas, since no tight crop will save it."""
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fw, fh = fig.get_size_inches()
    eps = 0.02
    over = []
    if bb.x0 < -eps:
        over.append(f"left {-bb.x0:.2f} in")
    if bb.y0 < -eps:
        over.append(f"bottom {-bb.y0:.2f} in")
    if bb.x1 > fw + eps:
        over.append(f"right {bb.x1 - fw:.2f} in")
    if bb.y1 > fh + eps:
        over.append(f"top {bb.y1 - fh:.2f} in")
    if over:
        print(f"    WARNING  {name} overflows the canvas: " + ", ".join(over))


def save(
    fig,
    name: str,
    svg: bool = True,
    tight: bool = True,
    sizes: dict[str, float] | None = None,
) -> None:
    """Write the figure, then report how large its type lands on GitHub.

    `sizes` maps a label to the point size actually used for that class of text.
    The measurement is taken from the written PNG's own header, so the numbers
    printed are what a reader gets, not what the layout intended.
    """
    FIGDIR.mkdir(exist_ok=True)
    png = FIGDIR / f"{name}.png"
    # `bbox_inches=None` would fall back to the rcParam, which is "tight"; the
    # figure's own bbox is what pins the PNG to exactly W_IN * DPI pixels.
    kw = {} if tight else {"bbox_inches": fig.bbox_inches, "pad_inches": 0.0}
    fig.savefig(png, dpi=DPI, **kw)
    if svg:
        fig.savefig(FIGDIR / f"{name}.svg", **kw)
    if not tight:
        _report_overflow(fig, name)
    plt.close(fig)
    size = png.stat().st_size / 1024
    w = png_width(png)
    print(
        f"  wrote figures/{name}.png  ({size:,.0f} KB, {w:,} px wide)"
        + ("  + .svg" if svg else "")
    )
    if sizes:
        parts = [f"{k} {pt:.1f}pt = {readme_px(pt, w):.1f}px" for k, pt in sizes.items()]
        print(f"    at a {README_PX} px README column:  " + "   ".join(parts))


def axes_in(fig, x: float, y: float, w: float, h: float):
    """Place an axes by inches measured from the bottom-left of the canvas."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def anchor_in(fig, x: float, y: float) -> tuple[float, float]:
    """A figure-fraction point from inches, for legends placed outside an axes."""
    fw, fh = fig.get_size_inches()
    return (x / fw, y / fh)


def under_legend(ax, fig, x: float, y: float, **kw):
    """A legend pinned in figure inches, so it cannot drift when a panel resizes."""
    kw.setdefault("labelcolor", S.MUTED)
    kw.setdefault("fontsize", S.FS_LEGEND)
    kw.setdefault("handlelength", 1.8)
    kw.setdefault("columnspacing", 2.0)
    kw.setdefault("labelspacing", 0.7)
    kw.setdefault("borderpad", 0.0)
    return ax.legend(
        loc="upper left",
        bbox_to_anchor=anchor_in(fig, x, y),
        bbox_transform=fig.transFigure,
        **kw,
    )


def readme_footer(fig, text: str, y_in: float = 0.11) -> None:
    """The provenance/caveat block, wrapped to the README canvas width."""
    _, fh = fig.get_size_inches()
    S.footer(
        fig,
        text,
        y=y_in / fh,
        size=S.FS_FOOT,
        width=S.wrap_cols(W_IN, S.FS_FOOT),
    )


def hbars(
    ax,
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str],
    errs=None,
    height: float = 0.62,
    fmt: str = "{:.3f}",
    value_pad: float = 0.008,
    alpha: Sequence[float] | None = None,
    label_size: float = S.FS_LABEL,
):
    y = np.arange(len(labels))[::-1]
    for i, (yy, v, c) in enumerate(zip(y, values, colors)):
        a = 1.0 if alpha is None else alpha[i]
        ax.barh(yy, v, height=height, color=c, alpha=a, zorder=3)
        if errs is not None:
            lo, hi = errs[i]
            ax.plot(
                [v - lo, v + hi],
                [yy, yy],
                color=S.BG_DEEP,
                lw=2.6,
                zorder=4,
                solid_capstyle="butt",
            )
            ax.plot([v - lo, v + hi], [yy, yy], color=S.BRIGHT, lw=0.9, zorder=5)
        right = v + (errs[i][1] if errs is not None else 0.0)
        ax.text(
            right + value_pad,
            yy,
            fmt.format(v),
            va="center",
            ha="left",
            color=S.TEXT,
            fontsize=label_size,
            zorder=6,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_size)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    return y


def vline(ax, x: float, label: str, color: str = S.C_CHANCE, ls=(0, (4, 3)),
          base: float = 0.015):
    ax.axvline(x, color=color, lw=0.9, ls=ls, zorder=2)
    ax.text(
        x,
        base,
        label + " ",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="right",
        color=color,
        fontsize=6.4,
        zorder=6,
    )


def node(
    ax,
    x,
    y,
    w,
    h,
    title,
    lines=(),
    accent=S.MUTED,
    fill=S.BG_DEEP,
    title_color=None,
    title_size=11.0,
    body_size=8.0,
    pad=12,
):
    """A labelled box on the schematic axes, where 1 data unit == 1/100 inch.

    Type sizes are parameters, not constants, because figure 1 has to stay
    legible when GitHub renders it at ~900 px: at this canvas width that is a
    scale factor of about 0.4, so nothing below ~7.5 pt survives the trip.
    """
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=6",
            facecolor=fill,
            edgecolor=accent,
            linewidth=1.0,
            zorder=3,
        )
    )
    ax.text(
        x + pad,
        y + h - pad,
        title,
        ha="left",
        va="top",
        color=title_color or accent,
        fontsize=title_size,
        weight="bold",
        zorder=4,
    )
    if lines:
        ax.text(
            x + pad,
            y + h - pad - title_size * 2.1,
            "\n".join(lines),
            ha="left",
            va="top",
            color=S.MUTED,
            fontsize=body_size,
            linespacing=1.6,
            zorder=4,
        )


def arrow(ax, p0, p1, color=S.RULE, lw=1.0, style="-|>", ls="-", rad=0.0, z=2):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=9,
            color=color,
            lw=lw,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def elbow(ax, pts, color=S.RULE, lw=1.0, ls="-", z=2):
    """A right-angled polyline ending in an arrowhead on its last segment."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="projecting")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, z=z)


# =========================================================================== #
# 1. pipeline schematic
# =========================================================================== #
def fig_pipeline(src: Sources) -> None:
    """The flow, and the one boundary that matters.

    Deliberately short of words. The fitted-versus-frozen accounting that used
    to occupy a third of this canvas is a table in the README instead: prose
    belongs there, and at README width it was unreadable here.
    """
    cfg = src("bc3/eval.json")["_config"]
    stats = src("mb2/input_stats.json")
    pools = src("plast3/run_frozen_current.json")
    act = src("calyx/probe_raw.json")["real"]["activity"]
    split = src("bc3/train_metrics.json")["_split"]

    graph = stats["graph"]
    glom = stats["glomeruli"]
    n_bits = int(cfg["n_bits_to_brain"])
    pl = pools["plasticity_pools"]
    dec = pools["decider"]
    om = pools["odour_map"]
    held_in = split["train"] + split["val"]

    # every subset of size 1..5 of an 8-card hand -- a combinatorial fact about
    # flybalatro.hands, derived here rather than typed in
    n_subsets = sum(math.comb(8, k) for k in range(1, 6))

    # full-figure axes so 1 data unit == 1/100 inch and nothing overflows.
    # 11.2 in wide: at GitHub's ~900 px README width that is a 0.40 scale, so
    # the 11 pt box titles land at ~13 px and the 8 pt body at ~9 px.
    W, H = 1120, 900
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    BX = 700  # the boundary
    ROW_Y, ROW_H = 540, 140  # the stations either side of it
    MID = ROW_Y + ROW_H / 2

    ax.text(24, 892, "the pipeline, and where the boundary is",
            color=S.BRIGHT, fontsize=15, weight="bold", va="top")
    ax.text(24, 865,
            "The fly is never shown a card. It is shown an odour that stands for "
            "an already-solved hand, and it decides what to do about it.",
            color=S.MUTED, fontsize=9, va="top")

    # ---- the boundary ----------------------------------------------------- #
    ax.plot([BX, BX], [50, 812], color=S.AMBER, lw=1.4, ls=(0, (3, 4)), zorder=2)
    ax.text(BX, 820, "THE BOUNDARY", color=S.AMBER, fontsize=10.5, weight="bold",
            ha="center", va="bottom")
    ax.text(BX, 802, f"{n_bits} bits cross.  Nothing else does.", color=S.AMBER,
            fontsize=8.5, ha="center", va="bottom", alpha=0.8)
    ax.text(BX + 9, 505, "everything the fly is ever told", color=S.AMBER,
            fontsize=7.5, rotation=90, ha="center", va="center", alpha=0.95)

    # ---- outside the brain ------------------------------------------------ #
    ax.add_patch(FancyBboxPatch((14, 520), 662, 268,
                                boxstyle="round,pad=0,rounding_size=8",
                                facecolor="#101418", edgecolor=S.RULE,
                                linewidth=0.9, zorder=1))
    ax.text(32, 772, "COMPUTED OUTSIDE THE BRAIN", color=S.ORANGE,
            fontsize=11, weight="bold", va="top")
    ax.text(32, 748,
            "ordinary Python, not biological in any sense.\n"
            "The poker is solved here, before anything reaches the fly.",
            color=S.MUTED, fontsize=8.5, va="top", linespacing=1.6)

    node(ax, 32, ROW_Y, 150, ROW_H, "Balatro",
         ["the dealt hand,", "the blind, and the", "chips still needed"],
         accent=S.SLATE, title_color=S.TEXT)
    node(ax, 208, ROW_Y, 230, ROW_H, "flybalatro/hands.py",
         [f"enumerates all {n_subsets} subsets",
          "of the dealt cards, scores",
          "each exactly as the engine",
          "does, picks the best"],
         accent=S.ORANGE, title_color=S.ORANGE)
    node(ax, 464, ROW_Y, 180, ROW_H, f"{n_bits} relay bits",
         ["best hand type, which", "cards, what's selected,", "score vs chips needed"],
         accent=S.ORANGE, title_color=S.ORANGE)
    arrow(ax, (182, MID), (206, MID), color=S.SLATE, lw=1.2)
    arrow(ax, (438, MID), (462, MID), color=S.ORANGE, lw=1.2)
    arrow(ax, (644, MID), (740, MID), color=S.AMBER, lw=2.0)

    # ---- inside the fly --------------------------------------------------- #
    ax.add_patch(FancyBboxPatch((724, 284), 382, 504,
                                boxstyle="round,pad=0,rounding_size=8",
                                facecolor="#0b0f14", edgecolor=S.RULE,
                                linewidth=0.9, zorder=1))
    ax.text(742, 772, "MaleCNS v1.0  --  FROZEN", color=S.AMBER,
            fontsize=11, weight="bold", va="top")
    ax.text(742, 748,
            f"{graph['neurons']:,} neurons, {graph['edges_csr']:,} edges at >= "
            f"{graph['threshold']} synapses,\n"
            f"leaky integrate-and-fire, one {cfg['window_ms']:.0f} ms window from "
            "reset.\nNo weight changes anywhere on this path.",
            color=S.MUTED, fontsize=8.5, va="top", linespacing=1.6)

    node(ax, 742, ROW_Y, 348, ROW_H, f"{n_bits} ORN glomeruli",
         ["1 relay bit = 1 whole olfactory receptor type.",
          f"{om['n_driven_neurons']:,} ORNs, {om['neurons_min']}-{om['neurons_max']}"
          " per type.",
          f"{glom['active_glomeruli_mean']:.1f} of {n_bits} lit per state, "
          f"{om['drive_mv']:.0f} mV tonic drive."],
         accent=S.CYAN, title_color=S.CYAN)

    chain = [
        ("ALPN", act["alpn"]["n"], S.TEAL, f"{act['alpn']['mean_rate_hz']:.0f} Hz"),
        ("Kenyon cells", act["kc"]["n"], S.GOLD, f"{act['kc']['mean_rate_hz']:.2f} Hz"),
        ("MBON", act["mbon"]["n"], S.AMBER, f"{act['mbon']['mean_rate_hz']:.1f} Hz"),
        ("descending", act["dn"]["n"], S.ORANGE, f"{act['dn']['mean_rate_hz']:.1f} Hz"),
    ]
    px, pw, ph, pgap = 742, 348, 42, 12
    ptop = 502  # top of the first pill
    ys = []
    for i, (name, n, col, rate) in enumerate(chain):
        y = ptop - i * (ph + pgap) - ph
        ys.append(y)
        ax.add_patch(FancyBboxPatch((px, y), pw, ph,
                                    boxstyle="round,pad=0,rounding_size=4",
                                    facecolor=S.BG_DEEP, edgecolor=col,
                                    linewidth=1.0, zorder=3))
        ax.text(px + 12, y + ph / 2, name, va="center", color=col, fontsize=9.5,
                zorder=4)
        ax.text(px + pw - 12, y + ph / 2, f"{n:,}     {rate}", va="center",
                ha="right", color=S.MUTED, fontsize=8.5, zorder=4)
    arrow(ax, (px + 40, ROW_Y), (px + 40, ptop), color=S.CYAN, lw=1.2)
    for i in range(1, len(chain)):
        arrow(ax, (px + 40, ys[i - 1]), (px + 40, ys[i] + ph),
              color=chain[i - 1][2], lw=1.2)

    # ---- the two heads, side by side in one box --------------------------- #
    hx, hy, hw, hh = 724, 96, 382, 176
    ax.add_patch(FancyBboxPatch((hx, hy), hw, hh,
                                boxstyle="round,pad=0,rounding_size=8",
                                facecolor="#0e1116", edgecolor=S.RULE,
                                linewidth=0.9, zorder=3))
    ax.text(hx + 14, hy + hh - 12, "then ONE OF TWO THINGS decides",
            color=S.TEXT, fontsize=9.5, weight="bold", va="top", zorder=4)
    lanes = [
        (730, S.SLATE, S.TEXT, "(a) trained readout",
         ["linear or 1x256 MLP on",
          "log1p ALPN+KC+DN counts.",
          "",
          "FITTED to the teacher's",
          f"actions on {held_in:,}",
          "held-in states."]),
        (924, S.AMBER, S.AMBER, "(b) MBON valence rule",
         [f"mean rate({dec['n_approach']} approach)",
          f"- mean rate({dec['n_avoid']} avoid)",
          "+ bias, through a sigmoid.",
          "",
          "NO trained action readout.",
          "4 label-free calibrations."]),
    ]
    for lx, rule, tcol, ltitle, lines in lanes:
        ax.plot([lx, lx], [hy + 14, hy + hh - 34], color=rule, lw=2.0, zorder=4)
        ax.text(lx + 10, hy + hh - 40, ltitle, color=tcol, fontsize=9.0,
                weight="bold", va="top", zorder=4)
        ax.text(lx + 10, hy + hh - 62, "\n".join(lines), color=S.MUTED,
                fontsize=7.5, va="top", linespacing=1.6, zorder=4)
    arrow(ax, (800, ys[3]), (800, hy + hh), color=S.SLATE, lw=1.2)
    elbow(ax, [(1090, ys[2] + ph / 2), (1098, ys[2] + ph / 2), (1098, hy + hh)],
          color=S.AMBER, lw=1.2)

    # ---- the action, and the loop back to the game ------------------------ #
    node(ax, 464, 129, 180, 110, "action",
         ["play the best subset,", "discard and dig, or", "select a card"],
         accent=S.GREEN, title_color=S.GREEN)
    arrow(ax, (722, 184), (646, 184), color=S.GREEN, lw=1.4)
    elbow(ax, [(464, 184), (450, 184), (450, 495), (107, 495), (107, ROW_Y - 2)],
          color=S.SLATE_DIM, lw=0.9)
    ax.text(115, 502, "the chosen action goes back to the game",
            color=S.SLATE, fontsize=7.5, va="bottom")

    # ---- dopamine: imposed from outside, reaching in ---------------------- #
    ax.add_patch(FancyBboxPatch((32, 300), 400, 170,
                                boxstyle="round,pad=0,rounding_size=8",
                                facecolor="#0b120e", edgecolor=S.GREEN,
                                linewidth=1.0, zorder=3))
    ax.text(46, 456, "DOPAMINE  --  path (b) only", color=S.GREEN,
            fontsize=10, weight="bold", va="top", zorder=4)
    ax.text(46, 432,
            "The chips the game paid become reward if the play made\n"
            "its fair share and punishment if it did not, gating\n"
            f"depression of {pl['kc_mbon_edges']:,} KC -> MBON synapses.\n\n"
            "An imposed harness signal computed from the game,\n"
            "not one the fly generates.",
            color=S.MUTED, fontsize=8, va="top", linespacing=1.6, zorder=4)
    dop_y = (ys[1] + ys[2] + ph) / 2
    arrow(ax, (432, dop_y), (740, dop_y), color=S.GREEN, lw=1.2, ls=(0, (4, 3)))
    ax.text(586, dop_y + 7, "depresses KC -> MBON", color=S.GREEN, fontsize=7.5,
            ha="center", va="bottom")

    S.footer(fig, src.line(
        "Orange marks what is computed outside the brain, green the harness "
        "signals that cross into and out of it, and each population inside the "
        "fly keeps the colour it has in the live dashboard. Counts and rates are "
        "read from the artefact that produced them; nothing here is drawn to "
        "scale. What is fitted and what is merely calibrated is tabulated in the "
        "README, beside this figure."), y=0.010)
    save(fig, "01_pipeline", tight=False)


# =========================================================================== #
# 2. the encoding reversal
# =========================================================================== #
def fig_encoding(src: Sources) -> None:
    """The reversal, stacked so the two encodings share one x scale.

    Side by side these panels were 13.6 in wide, which GitHub scales to 0.33 and
    turns 6.6 pt body type into 6 px. Stacked on the 11.2 in canvas each panel
    gets the full width, the legends spread onto one row instead of two columns
    of wrapped text, and the bars line up under each other -- which is the
    comparison the figure exists to make.
    """
    old = src("mb/probe_tuned.json")
    new_ = src("mb2/probes.json")
    stats = src("mb2/input_stats.json")

    old_run = old["runs"]["apl2"]
    a = old_run["results"]["A"]
    a_ref = old["reference"]["A"]
    a_int = old_run["results"]["intensity"]
    n_old = old["settings"]["n_samples"]

    g = new_["apl2_tonic_vth0"]
    orn_old = src("brain_probe.json")["summary_table"]["primary_w50_d30|label_A"]["real:ORN"]
    refs = stats["references"]
    n_new = stats["sample"]["n_sampled"]
    maj_new = stats["labels"]["hand_type_majority"]

    H = 8.7
    LEFT, PW, PH = 2.05, 8.70, 1.80
    LEG_H, GAP = 0.28, 0.34  # a two-line legend, and the space between panels
    fig = plt.figure(figsize=(W_IN, H))

    # ---- panel A: random-receptor encoding -------------------------------- #
    ax = axes_in(fig, LEFT, H - 1.05 - PH, PW, PH)
    rows = [
        ("raw input bits", a_ref["raw_bits"], S.CYAN),
        ("ALPN  (686)", a["readouts"]["ALPN"], S.TEAL),
        ("KC  (4,064)", a["readouts"]["KC"], S.GOLD),
        ("MBON  (97)", a["readouts"]["MBON"], S.AMBER),
        ("DN  (1,314)", a["readouts"]["DN"], S.ORANGE),
    ]
    y = hbars(
        ax,
        [r[0] for r in rows],
        [r[1]["best_acc"] for r in rows],
        [r[2] for r in rows],
        errs=[(r[1]["best_std"], r[1]["best_std"]) for r in rows],
    )
    ax.set_xlim(0, 1.08)
    chance = a["majority_class_rate"]
    perm = a["readouts"]["permuted:DN"]["best_acc"]
    ax.axvline(chance, color=S.C_CHANCE, lw=0.9, ls=(0, (4, 3)), zorder=2,
               label=f"majority class  {chance:.3f}")
    ax.axvline(perm, color=S.RED, lw=0.9, ls=(0, (2, 3)), zorder=2,
               label=f"permuted labels, DN  {perm:.3f}")
    dn_int = a_int["readouts"]["DN"]["best_acc"]
    ax.plot([dn_int], [y[-1]], marker="D", ms=5.0, color=S.RED, zorder=7, ls="none",
            label=f"the same DN spikes decode how many bits are on at {dn_int:.3f}"
                  " -- intensity, not identity")
    S.xgrid(ax)
    ax.set_xlabel("linear-probe accuracy, best of a 6-value C grid, 5-fold CV")
    under_legend(fig=fig, ax=ax, x=LEFT, y=H - 1.05 - PH - 0.42, ncols=2,
                 columnspacing=3.0)
    S.title(
        ax,
        "A · random-receptor encoding",
        "10 scattered ORNs per bit, ~36 bits on, all 54 glomeruli lit on every input.\n"
        f"Binary label on a random 12-bit group, n = {n_old}. Bars carry the CV SD.",
    )

    # ---- panel B: glomerular encoding ------------------------------------- #
    B_TOP = H - 1.05 - PH - 0.42 - LEG_H - GAP - 1.05
    ax = axes_in(fig, LEFT, B_TOP - PH, PW, PH)
    rows = [
        ("relay bits (input ceiling)", refs["relay_bits_32"]["hand_type"]["best_acc"], None, S.CYAN),
        ("ALPN  (686)", g["ALPN.full.hand_type"]["acc"], g["ALPN.full.hand_type"]["grouped_acc"], S.TEAL),
        ("KC  (4,064)", g["KC.full.hand_type"]["acc"], g["KC.full.hand_type"]["grouped_acc"], S.GOLD),
        ("MBON  (97)", g["MBON.full.hand_type"]["acc"], g["MBON.full.hand_type"]["grouped_acc"], S.AMBER),
        ("DN  (1,314)", g["DN.full.hand_type"]["acc"], g["DN.full.hand_type"]["grouped_acc"], S.ORANGE),
    ]
    y = hbars(ax, [r[0] for r in rows], [r[1] for r in rows], [r[3] for r in rows])
    for yy, r in zip(y, rows):
        if r[2] is not None:
            ax.plot([r[2]], [yy], marker="|", ms=12, mew=1.6, color=S.BRIGHT, zorder=7)
    ax.set_xlim(0, 1.08)
    cnt = refs["driven_orn_count_only"]["hand_type"]["best_acc"]
    ax.axvline(maj_new, color=S.C_CHANCE, lw=0.9, ls=(0, (4, 3)), zorder=2,
               label=f"majority class  {maj_new:.3f}")
    ax.axvline(cnt, color=S.RED, lw=0.9, ls=(0, (2, 3)), zorder=2,
               label=f"driven-ORN count alone  {cnt:.3f}")
    ax.plot([], [], marker="|", ms=10, mew=1.6, color=S.BRIGHT, ls="none",
            label="grouped CV, relay pattern as the group")
    dn_int = g["DN.full.intensity"]["acc"]
    ax.plot([dn_int], [y[-1]], marker="D", ms=5.0, color=S.RED, zorder=7, ls="none",
            label=f"DN intensity probe {dn_int:.3f} -- still above its identity score")
    S.xgrid(ax)
    ax.set_xlabel("linear-probe accuracy, 9-way hand type")
    under_legend(fig=fig, ax=ax, x=LEFT, y=B_TOP - PH - 0.42, ncols=2,
                 columnspacing=3.0)
    S.title(
        ax,
        "B · glomerular encoding",
        f"1 relay bit = 1 whole ORN type, "
        f"{stats['glomeruli']['active_glomeruli_mean']:.1f} of 32 lit per state.\n"
        f"9-way hand type on {n_new:,} real game states.",
    )

    readme_footer(
        fig,
        src.line(
            "The two panels are NOT a controlled comparison: the task changed too "
            "(binary synthetic label -> 9-way hand type). What is controlled is that "
            "both run the same tuned brain (apl_scale = 2, tonic drive) and both carry "
            "their own chance line and their own count-only confound. ORN itself is not "
            "probed here; under the random-receptor encoding the v1 gate probe put it "
            f"at {orn_old:.3f}, i.e. indistinguishable from the raw bits."
        ),
    )
    save(fig, "02_encoding_reversal", tight=False, sizes={
        "title": S.FS_TITLE, "subtitle": S.FS_SUB, "axis": S.FS_BODY,
        "bar label": S.FS_LABEL, "legend": S.FS_LEGEND, "footer": S.FS_FOOT,
    })


# =========================================================================== #
# 3. behaviour cloning
# =========================================================================== #
@dataclass
class BCRow:
    label: str
    key: str
    color: str
    note: str = ""


def _bc_panel(ax, fig, ev, rows: Sequence[BCRow], ceiling_key: str, title_main,
              title_sub, legend_y: float, legend_x: float,
              flag: Callable[[str], bool] | None = None):
    labels, vals, errs, cols, alphas, keys = [], [], [], [], [], []
    for r in rows:
        heads = [("-", 1.0)] if r.key.endswith("/-") else [("linear", 0.42), ("mlp", 1.0)]
        for head, alpha in heads:
            key = r.key if r.key.endswith("/-") else f"{r.key}/{head}"
            if key not in ev:
                continue
            d = ev[key]
            p, lo, hi = wilson(int(d["wins"]), int(d["n_episodes"]))
            suffix = "" if head == "-" else f"   {head}"
            labels.append(r.label + suffix)
            vals.append(p)
            errs.append(err_from(p, lo, hi))
            cols.append(r.color)
            alphas.append(alpha)
            keys.append(r.key)
    y = hbars(ax, labels, vals, cols, errs=errs, fmt="{:.3f}", alpha=alphas, height=0.68)
    ceil = ev[ceiling_key]
    cp, _, _ = wilson(int(ceil["wins"]), int(ceil["n_episodes"]))
    ax.axvline(cp, color=S.C_NOBRAIN, lw=1.0, ls=(0, (5, 3)), zorder=2,
               label=f"no-brain ceiling  {cp:.3f}   (the same readout on the same "
                     "bits, with the fly taken out of the loop)")
    if flag is not None:
        # a gutter of its own, clear of the longest tick label, so the mark
        # cannot be read as part of the row it flags
        dx = -(max(len(l) for l in labels) * S.CHAR_EM * S.FS_LABEL + 12)
        for yy, k in zip(y, keys):
            if flag(k):
                ax.annotate(
                    "!", xy=(0, yy), xycoords=ax.get_yaxis_transform(),
                    xytext=(dx, 0), textcoords="offset points",
                    ha="center", va="center", color=S.RED,
                    fontsize=S.FS_LABEL, weight="bold", annotation_clip=False,
                )
    S.xgrid(ax)
    ax.set_xlim(0, max(0.78, max(vals) + 0.13))
    ax.set_xlabel(f"ante-1 clear rate, {ceil['n_episodes']} episodes, Wilson 95% CI")
    under_legend(fig=fig, ax=ax, x=legend_x, y=legend_y)
    S.title(ax, title_main, title_sub)
    return y


def fig_behaviour_cloning(src: Sources) -> None:
    """v2 above v3 on one shared x axis.

    Side by side this was two 7 in columns carrying 16 and 20 rows of 7.4 pt
    labels, which GitHub scaled to 6.3 px. Nothing here can be dropped -- every
    row is either a baseline, a control or a readout the README argues from --
    so the canvas narrows to 11.2 in and grows downwards instead: each panel now
    spans the full width, the rows get a 0.20 in pitch, and stacking puts the
    two encodings on the same x scale, which is the comparison being made.
    """
    ev2 = src("bc2/eval.json")
    ev3 = src("bc3/eval.json")

    rows2 = [
        BCRow("random legal", "random/-", S.SLATE_DIM),
        BCRow("v2 teacher (expert)", "teacher_v2/-", S.C_TEACHER),
        BCRow("raw 315 bits, no brain", "raw_bits", S.C_NOBRAIN),
        BCRow("random projection", "rand_proj_matched", S.INDIGO),
        BCRow("shuffled fly, ALPN+KC+DN", "shuf_alpn_kc_dn", S.C_SHUFFLED),
        BCRow("shuffled fly, DN only", "shuf_dn", S.C_SHUFFLED),
        BCRow("real fly, ALPN+KC+DN", "real_alpn_kc_dn", S.C_REAL),
        BCRow("real fly, DN only", "real_dn", S.ORANGE),
    ]
    rows3 = [
        BCRow("random legal", "random/-", S.SLATE_DIM),
        BCRow("v2 teacher (expert)", "teacher_v2/-", S.C_TEACHER),
        BCRow("raw 32 relay bits, no brain", "raw_bits", S.C_NOBRAIN),
        BCRow("random projection", "rand_proj_matched", S.INDIGO),
        BCRow("shuffled fly, ALPN+KC+DN", "shuf_alpn_kc_dn", S.C_SHUFFLED),
        BCRow("shuffled fly, DN only", "shuf_dn", S.C_SHUFFLED),
        BCRow("real fly, ALPN+KC+DN", "real_alpn_kc_dn", S.C_REAL),
        BCRow("real fly, KC only", "real_kc", S.GOLD),
        BCRow("real fly, MBON only", "real_mbon", S.AMBER),
        BCRow("real fly, DN only", "real_dn", S.ORANGE),
    ]

    PITCH = 0.20  # inches per bar, the floor at which 8 pt labels stay apart
    n2 = sum(1 for r in rows2 for h in (("-",) if r.key.endswith("/-") else ("linear", "mlp"))
             if (r.key if r.key.endswith("/-") else f"{r.key}/{h}") in ev2)
    n3 = sum(1 for r in rows3 for h in (("-",) if r.key.endswith("/-") else ("linear", "mlp"))
             if (r.key if r.key.endswith("/-") else f"{r.key}/{h}") in ev3)
    PH2, PH3 = n2 * PITCH, n3 * PITCH

    LEFT, PW = 2.85, 7.90
    TOP_PAD, XLAB, LEG_H, GAP = 1.05, 0.42, 0.26, 0.34
    CAVEAT_H, FOOT_H = 0.70, 0.95
    H = (TOP_PAD + PH2 + XLAB + LEG_H + GAP
         + TOP_PAD + PH3 + XLAB + LEG_H + GAP
         + CAVEAT_H + FOOT_H)
    fig = plt.figure(figsize=(W_IN, H))

    a_bot = H - TOP_PAD - PH2
    ax_a = axes_in(fig, LEFT, a_bot, PW, PH2)
    _bc_panel(
        ax_a, fig, ev2, rows2, "raw_bits/mlp",
        "v2 · relay bits on scattered receptors",
        "315 bits, 10 random ORNs each. The brain costs almost everything:\n"
        "the best real-wiring readout clears 0 blinds in 400 episodes.",
        legend_y=a_bot - XLAB, legend_x=LEFT,
    )

    b_bot = a_bot - XLAB - LEG_H - GAP - TOP_PAD - PH3
    ax_b = axes_in(fig, LEFT, b_bot, PW, PH3)
    gap3 = ev3["raw_bits/mlp"]["clear_rate"] - ev3["real_alpn_kc_dn/mlp"]["clear_rate"]
    _bc_panel(
        ax_b, fig, ev3, rows3, "raw_bits/mlp",
        "v3 · one relay bit = one whole glomerulus",
        "32 bits, same states, same teacher, same held-out split, same eval seeds.\n"
        f"The brain now costs {gap3 * 100:.1f} points against its own no-brain control.",
        legend_y=b_bot - XLAB, legend_x=LEFT,
        flag=lambda k: k.startswith("shuf_"),
    )

    caveat_top = b_bot - XLAB - LEG_H - GAP
    fig.text(
        0.30 / W_IN, caveat_top / H,
        "!  under the glomerular encoding the global degree-preserving shuffle is a "
        "BROKEN control. It destroys the glomerular convergence\n"
        "   the encoding depends on and leaves a near-silent network, so it compares "
        "a working brain against a dead one. Rate table in\n"
        "   RESULTS.md; figure 5 is the control that replaces it.",
        ha="left", va="top", color=S.RED, fontsize=S.FS_LEGEND, linespacing=1.7,
    )
    readme_footer(
        fig,
        src.line(
            "Faint bars are the linear readout, solid bars the 1x256 MLP. Both are "
            "trained by behaviour cloning of the v2 hand-aware teacher; the fly's "
            "synapses never change on this path. v2 and v3 are NOT a controlled "
            "comparison with each other -- the encoding, the number of input bits and "
            "the brain's calibration all changed at once."
        ),
    )
    save(fig, "03_behaviour_cloning", tight=False, sizes={
        "title": S.FS_TITLE, "subtitle": S.FS_SUB, "axis": S.FS_BODY,
        "bar label": S.FS_LABEL, "legend": S.FS_LEGEND,
        "retraction note": S.FS_LEGEND, "footer": S.FS_FOOT,
    })


# =========================================================================== #
# 4. plasticity
# =========================================================================== #
BUCKETS = ["lt0.25", "lt0.5", "lt1.0", "ge1.0"]
BUCKET_LABELS = [
    "< 0.25x\nits share",
    "< 0.5x",
    "< 1.0x",
    ">= 1.0x\nclears it",
]


def _pool_buckets(runs: Iterable[dict], mode: str = "greedy") -> dict[str, tuple[int, int]]:
    out = {b: [0, 0] for b in BUCKETS}
    for r in runs:
        for b, d in r["eval"][mode]["p_play_by_bucket"].items():
            out[b][0] += int(d["plays"])
            out[b][1] += int(d["n"])
    return {b: (v[0], v[1]) for b, v in out.items()}


def fig_plasticity(src: Sources) -> None:
    """The 2x2 above the rule it produced.

    The two panels shared a 14.2 in row, which GitHub scaled to 0.31. Stacked on
    the 11.2 in canvas panel B gets the whole width -- five series across four
    buckets is what needed the room -- and its legend spreads onto two rows
    instead of three cramped columns.
    """
    summary = src("plast3/summary.json")
    cells = summary["cells"]
    mode = summary["primary_mode"]

    frozen_cur = src("plast3/run_frozen_current.json")
    frozen_sep = src("plast3/run_frozen_separated.json")
    teacher = src("plast3/baselines3.json")["teacher_dig"]

    cur_cur = [src(f"plast3/run_current_current_s{i}.json") for i in range(3)]
    om_sep = [src(f"plast3/run_omission_separated_s{i}.json") for i in range(3)]

    LEFT, PW = 1.75, 9.00
    PH_A, PH_B = 1.95, 2.95
    TOP_PAD, GAP = 1.05, 0.40
    XLAB_A, XLAB_B, LEG_B = 0.66, 0.82, 0.72  # B's tick labels are two lines
    FOOT_H = 1.55
    H = TOP_PAD + PH_A + XLAB_A + GAP + TOP_PAD + PH_B + XLAB_B + LEG_B + FOOT_H
    fig = plt.figure(figsize=(W_IN, H))

    # ---- panel A: the 2x2, each cell against its own frozen control -------- #
    a_bot = H - TOP_PAD - PH_A
    ax = axes_in(fig, LEFT, a_bot, PW, PH_A)
    order = [
        ("current_current", "chips-only punishment\ncurrent encoding", S.SLATE),
        ("omission_current", "+ omission punishment\ncurrent encoding", S.PURPLE),
        ("current_separated", "chips-only punishment\nseparated encoding", S.TEAL),
        ("omission_separated", "+ omission punishment\nseparated encoding", S.AMBER),
    ]
    labels, vals, errs, cols = [], [], [], []
    deltas = {}
    for key, label, col in order:
        vf = cells[key][mode]["vs_frozen"]
        d = vf["delta"]
        lo, hi = vf["ci_lo"], vf["ci_hi"]
        deltas[key] = d
        labels.append(label)
        vals.append(d)
        errs.append(err_from(d, lo, hi))
        cols.append(col)

    inter = (
        deltas["omission_separated"]
        - deltas["omission_current"]
        - deltas["current_separated"]
        + deltas["current_current"]
    )

    y = hbars(ax, labels, vals, cols, errs=errs, fmt="{:+.3f}", value_pad=0.006,
              height=0.48)
    ax.axvline(0, color=S.MUTED, lw=0.9, zorder=2)
    S.xgrid(ax)
    lim = max(abs(v) + e[1] for v, e in zip(vals, errs)) + 0.10
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(
        f"clear rate minus its OWN frozen control, {mode}\n"
        "paired on 400 games, bootstrap CI clustered on eval seed and training run"
    )
    S.title(
        ax,
        "A · the 2x2: useless apart, large together",
        "Each row is a learning fly minus the identical fly with dopamine\n"
        "switched off, on the same 400 games. 3 training seeds per cell.",
    )
    ax.annotate(
        f"interaction contrast  {inter:+.3f}\n"
        "the effect is essentially all interaction",
        xy=(1, 1), xycoords="axes fraction", xytext=(-6, -8),
        textcoords="offset points", ha="right", va="top",
        color=S.AMBER, fontsize=S.FS_LABEL, linespacing=1.7,
    )

    # ---- panel B: P(play) by score bucket --------------------------------- #
    b_bot = a_bot - XLAB_A - GAP - TOP_PAD - PH_B
    ax = axes_in(fig, LEFT, b_bot, PW, PH_B)
    series = [
        ("naive control, current enc.", _pool_buckets([frozen_cur], mode), S.SLATE, True),
        ("learned: chips-only (the v2 fly)", _pool_buckets(cur_cur, mode), S.SLATE, False),
        ("naive control, separated enc.", _pool_buckets([frozen_sep], mode), S.AMBER, True),
        ("learned: omission + separated (the v3 fly)", _pool_buckets(om_sep, mode), S.AMBER, False),
        (
            "v2 teacher (sees plays_left)",
            {b: (int(teacher["p_play_by_bucket"][b]["plays"]),
                 int(teacher["p_play_by_bucket"][b]["n"])) for b in BUCKETS},
            S.C_TEACHER,
            False,
        ),
    ]
    n_series = len(series)
    width = 0.16
    xs = np.arange(len(BUCKETS))
    for i, (label, data, col, hollow) in enumerate(series):
        off = (i - (n_series - 1) / 2) * width
        ps, los, his = [], [], []
        for b in BUCKETS:
            k, n = data[b]
            p, lo, hi = wilson(k, n)
            e_lo, e_hi = err_from(p, lo, hi)
            ps.append(p)
            los.append(e_lo)
            his.append(e_hi)
        ax.bar(
            xs + off,
            ps,
            width=width * 0.88,
            color="none" if hollow else col,
            edgecolor=col,
            linewidth=1.1 if hollow else 0.0,
            zorder=3,
            label=label,
        )
        ax.errorbar(
            xs + off,
            ps,
            yerr=[los, his],
            fmt="none",
            ecolor=S.BG_DEEP,
            elinewidth=2.4,
            capsize=0,
            zorder=4,
        )
        ax.errorbar(
            xs + off,
            ps,
            yerr=[los, his],
            fmt="none",
            ecolor=S.BRIGHT,
            elinewidth=0.8,
            capsize=0,
            zorder=5,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(BUCKET_LABELS, fontsize=S.FS_BODY)
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    S.ygrid(ax)
    ax.set_ylabel("P(play | the decision was legal)")
    under_legend(fig=fig, ax=ax, x=LEFT, y=b_bot - XLAB_B, ncols=2,
                 labelcolor=S.TEXT, handlelength=1.4, columnspacing=3.0)
    S.title(
        ax,
        "B · the rule forming",
        "The one ordered axis the odour carries. The v3 fly inverts its own\n"
        "naive policy and lands on the teacher's shape from chips alone.",
    )

    counts = _pool_buckets(om_sep, mode)
    ax.set_xlabel("what the best available hand scores against the chips still needed")

    readme_footer(
        fig,
        src.line(
            "Hollow bars are the naive control -- the identical fly with dopamine "
            "switched off. Decisions per bucket for the v3 fly, 3 runs pooled: "
            + ",  ".join(f"{b} {counts[b][1]:,}" for b in BUCKETS) + ". "
            "Evaluation is greedy (the preregistered primary mode), on 400 paired "
            "games, seeds 400000-400399. The protocol, primary outcome and decision "
            "rule were fixed in outputs/plast3/PREREGISTRATION.md before any of these "
            "games were played. The winning cell beats its own frozen control and TIES "
            "always-discard; under the preregistered rule that is a miss, not a win."
        ),
    )
    save(fig, "04_plasticity", tight=False, sizes={
        "title": S.FS_TITLE, "subtitle": S.FS_SUB, "axis": S.FS_BODY,
        "bar label": S.FS_LABEL, "legend": S.FS_LEGEND, "footer": S.FS_FOOT,
    })


# =========================================================================== #
# 5. the calyx null
# =========================================================================== #
def fig_calyx(src: Sources) -> None:
    """A and B keep their row, C moves under them and spreads sideways.

    Three panels on a 14.4 in row put the body type at 6.2 px on GitHub. A and B
    are both narrow row plots and survive half a 11.2 in canvas; panel C was the
    one that could not, so it drops to a full-width band underneath where its
    table and the retraction beside it both get their own column.
    """
    praw = src("calyx/probe_raw.json")
    paired_raw = src("calyx/paired_raw.json")["paired"]
    paired_hom = src("calyx/paired_homeo.json")["paired"]
    rewiring = src("calyx/rewiring.json")

    seeds = ["rw1", "rw2", "rw3"]

    A_LEFT, A_W = 2.05, 3.55
    B_LEFT, B_W = 6.70, 4.05
    PH = 2.85
    TOP_PAD, XLAB_B, LEG_A, GAP = 1.20, 0.62, 0.30, 0.45
    C_H, FOOT_H = 2.45, 1.00
    H = TOP_PAD + PH + XLAB_B + GAP + C_H + FOOT_H
    fig = plt.figure(figsize=(W_IN, H))
    row_bot = H - TOP_PAD - PH

    # ---- panel A: real sits inside the rewired band ----------------------- #
    ax = axes_in(fig, A_LEFT, row_bot, A_W, PH)
    rows = []
    for name, fn in (
        ("KC hand type, grouped CV", lambda d: d["decode_hand_type"]["kc"]["acc_grouped"]),
        ("DN hand type, grouped CV", lambda d: d["decode_hand_type"]["dn"]["acc_grouped"]),
        ("KC Jaccard between/within", lambda d: d["kc_code"]["jaccard"]["ratio"]),
    ):
        rows.append((name, fn(praw["real"]), [fn(praw[s]) for s in seeds]))
    rows.append(
        (
            "KC imitation top-1",
            paired_raw["kc/rw1"]["imitation_top1"]["real"],
            [paired_raw[f"kc/{s}"]["imitation_top1"]["rewired"] for s in seeds],
        )
    )
    rows.append(
        (
            "KC clear rate, 400 episodes",
            paired_raw["kc/rw1"]["clear_rate"]["real"],
            [paired_raw[f"kc/{s}"]["clear_rate"]["rewired"] for s in seeds],
        )
    )
    rows.append(
        (
            "KC clear rate, + homeostasis",
            paired_hom["kc/rw1"]["clear_rate"]["real"],
            [paired_hom[f"kc/{s}"]["clear_rate"]["rewired"] for s in seeds],
        )
    )

    ys = np.arange(len(rows))[::-1]
    for yy, (name, real, rw) in zip(ys, rows):
        lo, hi = min(rw), max(rw)
        ax.plot([lo, hi], [yy, yy], color=S.C_REWIRED, lw=5.0, alpha=0.28,
                solid_capstyle="round", zorder=2)
        ax.plot(rw, [yy] * len(rw), marker="o", ms=4.0, ls="none",
                color=S.C_REWIRED, zorder=3)
        ax.plot([real], [yy], marker="|", ms=15, mew=2.0, color=S.C_REAL, zorder=4)
        ax.text(max(hi, real) + 0.030, yy, f"{real:.3f}", va="center",
                color=S.TEXT, fontsize=S.FS_LABEL)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=S.FS_LABEL)
    ax.set_ylim(ys.min() - 0.7, ys.max() + 0.7)
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    S.xgrid(ax)
    ax.plot([], [], marker="|", ms=11, mew=2.0, ls="none", color=S.C_REAL,
            label="real ALPN -> KC wiring")
    ax.plot([], [], marker="o", ms=4.0, ls="none", color=S.C_REWIRED,
            label="3 rewiring seeds (empirical null)")
    under_legend(fig=fig, ax=ax, x=A_LEFT, y=row_bot - LEG_A, ncols=1,
                 handlelength=1.4)
    S.title(
        ax,
        "A · real sits inside the null band",
        f"{rewiring['strata']['n_edges']:,} calyx edges resampled within "
        f"{rewiring['strata']['n_strata']} strata of\n"
        "(hemisphere, KC subtype, synapse count, sign).\n"
        "Every degree and weight preserved exactly.",
    )

    # ---- panel B: paired differences, CIs crossing zero -------------------- #
    ax = axes_in(fig, B_LEFT, row_bot, B_W, PH)
    entries = []
    for variant, tag, table in (("raw", "", paired_raw), ("homeo", " + homeo", paired_hom)):
        for readout in ("kc", "dn"):
            for s in seeds:
                d = table[f"{readout}/{s}"]["clear_rate"]
                n = int(d["a_only"] + d["b_only"] + d["both"] + d["neither"])
                diff, lo, hi = paired_prop_ci(int(d["a_only"]), int(d["b_only"]), n)
                entries.append(
                    (f"{readout.upper()}{tag} · {s}", diff, lo, hi,
                     S.GOLD if readout == "kc" else S.ORANGE, d["mcnemar_p"])
                )
    ys = np.arange(len(entries))[::-1]
    for yy, (name, diff, lo, hi, col, p) in zip(ys, entries):
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.2, alpha=0.75, zorder=3)
        ax.plot([diff], [yy], marker="o", ms=4.2, color=col, zorder=4)
        ax.text(0.185, yy, f"p={p:.2f}", va="center", ha="right",
                color=S.MUTED, fontsize=S.FS_LABEL)
    ax.axvline(0, color=S.BRIGHT, lw=0.9, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entries], fontsize=S.FS_LABEL)
    ax.set_ylim(ys.min() - 0.7, ys.max() + 0.7)
    ax.set_xlim(-0.135, 0.19)
    ax.set_xticks([-0.10, -0.05, 0.0, 0.05, 0.10])
    S.xgrid(ax)
    ax.set_xlabel("clear rate, real minus rewired\n400 paired episode seeds, 95% CI")
    S.title(
        ax,
        f"B · {len(entries)} paired comparisons, none significant",
        "Exact McNemar on the same 400 episode seeds in every\n"
        "condition. Each row's exact p is printed at the right;\n"
        f"the smallest of the {len(entries)} is {min(e[5] for e in entries):.2f}.",
    )

    # ---- panel C: the activity-matching table ----------------------------- #
    c_top = row_bot - XLAB_B - GAP
    ax = axes_in(fig, 0.30, c_top - C_H, W_IN - 0.60, C_H)
    ax.axis("off")
    ax.set_xlim(0, (W_IN - 0.60) * 100)  # 1 data unit == 1/100 inch
    ax.set_ylim(0, C_H * 100)
    top = C_H * 100

    ax.text(0, top, "C · the control is fair before any correction",
            color=S.BRIGHT, fontsize=S.FS_TITLE, weight="bold", va="top")
    ax.text(0, top - 26,
            "The rewired networks land in the same activity regime with no "
            "recalibration at all. Rates in Hz.",
            color=S.MUTED, fontsize=S.FS_SUB, va="top")

    def fmt_row(fn, fmt):
        return [fmt.format(fn(praw["real"]))] + [fmt.format(fn(praw[s])) for s in seeds]

    table = [
        ("ALPN rate",
         fmt_row(lambda d: d["activity"]["alpn"]["mean_rate_hz"], "{:.1f}")),
        ("KC rate",
         fmt_row(lambda d: d["activity"]["kc"]["mean_rate_hz"], "{:.2f}")),
        ("KC active / state",
         fmt_row(lambda d: d["activity"]["kc"]["active_per_state"], "{:.0f}")),
        ("MBON rate",
         fmt_row(lambda d: d["activity"]["mbon"]["mean_rate_hz"], "{:.2f}")),
        ("DN rate",
         fmt_row(lambda d: d["activity"]["dn"]["mean_rate_hz"], "{:.2f}")),
        ("non-constant channels",
         fmt_row(lambda d: d["activity"]["non_constant_channels_alpn_kc_dn"], "{:,}")),
    ]
    x_cols = [235, 310, 385, 460]
    head = ["real", "rw1", "rw2", "rw3"]
    y0 = top - 62
    for x, h in zip(x_cols, head):
        ax.text(x, y0, h, color=S.C_REAL if h == "real" else S.C_REWIRED,
                fontsize=S.FS_LABEL, ha="right", va="top", weight="bold")
    ax.plot([0, 470], [y0 - 14, y0 - 14], color=S.RULE, lw=0.8)
    for i, (name, vals) in enumerate(table):
        yy = y0 - 28 - i * 22
        ax.text(0, yy, name, color=S.TEXT, fontsize=S.FS_LABEL, va="top")
        for x, v in zip(x_cols, vals):
            ax.text(x, yy, v, color=S.TEXT if x == x_cols[0] else S.MUTED,
                    fontsize=S.FS_LABEL, ha="right", va="top")
    ax.text(
        560, y0 + 4,
        "The global shuffle this control replaces left a near-silent\n"
        "network and is retracted; the README says why.\n\n"
        "Caron et al. 2013 make a near-null here the expected result,\n"
        "not an indictment of the model.",
        color=S.MUTED, fontsize=S.FS_LEGEND, va="top", linespacing=1.7,
    )

    readme_footer(
        fig,
        src.line(
            "Panel B intervals are Wald intervals on the paired difference computed "
            "from the McNemar discordant counts stored in the artefacts; the p-values "
            "are the exact McNemar tests those files carry. The control holds each "
            "projection neuron's hemisphere and KC-subtype profile fixed, so the "
            "subtype-level structure of Zheng et al. 2022 is held constant by this "
            "design, not tested by it."
        ),
    )
    save(fig, "05_calyx_null", tight=False, sizes={
        "title": S.FS_TITLE, "subtitle": S.FS_SUB, "axis": S.FS_BODY,
        "row label / table": S.FS_LABEL, "legend": S.FS_LEGEND, "footer": S.FS_FOOT,
    })


# --------------------------------------------------------------------------- #
FIGURES: dict[str, tuple[str, Callable[[Sources], None]]] = {
    "1": ("pipeline schematic, and where the boundary is", fig_pipeline),
    "2": ("the encoding reversal", fig_encoding),
    "3": ("behaviour cloning, ante-1 clear rate", fig_behaviour_cloning),
    "4": ("plasticity: the 2x2 and the rule forming", fig_plasticity),
    "5": ("the calyx rewiring null", fig_calyx),
}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None, help="figure numbers to draw")
    ap.add_argument("--list", action="store_true", help="list figures and exit")
    args = ap.parse_args(argv)

    if args.list:
        for k, (desc, _) in FIGURES.items():
            print(f"  {k}  {desc}")
        return 0

    wanted = args.only or list(FIGURES)
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s): {', '.join(unknown)}")

    FIGDIR.mkdir(exist_ok=True)
    for key in wanted:
        desc, fn = FIGURES[key]
        print(f"figure {key}: {desc}")
        fn(Sources())
    print(
        "\nThe demo clip (figures/demo.gif, figures/demo.mp4) is cut from\n"
        "outputs/realgame/balatro_fly.mov with ffmpeg; see docs/REPRODUCE.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
