"""Shared plotting style for `scripts/figures.py`.

Dark, monospace, restrained -- the same visual language as the live dashboard in
`outputs/pov/dashboard_embedded.png`, whose palette these constants were sampled
from. Nothing here reads data or draws a figure; it only configures matplotlib.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# --------------------------------------------------------------------------- #
# palette -- sampled from outputs/pov/dashboard_embedded.png
# --------------------------------------------------------------------------- #
BG = "#0e1013"  # panel background
BG_DEEP = "#08090b"  # inset / well background
RULE = "#1c2027"  # hairline separators, axis spines
GRID = "#161a20"  # the only gridlines we draw
MUTED = "#5d636e"  # captions, tick labels, provenance
TEXT = "#c2c7cf"  # body text
BRIGHT = "#e8ecf2"  # titles, headline numbers

# population / role accents, same assignment as the dashboard legend
AMBER = "#f0c96a"  # approach MBONs, headline series
GOLD = "#d3a14d"  # Kenyon cells
PURPLE = "#ad8af8"  # avoid MBONs
INDIGO = "#8e86d1"  # central complex
TEAL = "#69c4b4"  # ALPN (antennal lobe output)
CYAN = "#6cbad6"  # sensory / ORN
ORANGE = "#de7747"  # descending neurons
GREEN = "#69ce82"  # PAM, reward
RED = "#dd524c"  # PPL1, punishment
SLATE = "#515864"  # inert / control series
SLATE_DIM = "#363f4c"  # deep background series

# semantic aliases used across figures
C_REAL = AMBER
C_REWIRED = TEAL
C_SHUFFLED = SLATE
C_NOBRAIN = CYAN
C_TEACHER = BRIGHT
C_CHANCE = "#3a3f49"

MONO = ["SF Mono", "Menlo", "DejaVu Sans Mono", "monospace"]


def _available_mono() -> list[str]:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    picked = [n for n in MONO if n in installed]
    return picked + ["monospace"]


def use_style() -> None:
    """Apply the project figure style to the global matplotlib rcParams."""
    fam = _available_mono()
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "figure.edgecolor": BG,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.28,
            "axes.facecolor": BG,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": BRIGHT,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": TEXT,
            "ytick.labelcolor": TEXT,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "text.color": TEXT,
            "font.family": fam,
            "font.size": 8.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.0,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "figure.dpi": 100,
            "lines.solid_capstyle": "butt",
            "patch.linewidth": 0.0,
            "hatch.linewidth": 0.6,
        }
    )


def title(ax, text: str, sub: str | None = None) -> None:
    """A left-aligned title with an optional dimmer block beneath it.

    Both are placed with point offsets from the top-left of the axes, so the
    gap between them does not change when the axes is resized.
    """
    sub_lines = sub.count("\n") + 1 if sub else 0
    sub_h = sub_lines * 7.2 * 1.45
    ax.annotate(
        text,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 10 + sub_h + 6),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=BRIGHT,
        fontsize=10,
        weight="bold",
        annotation_clip=False,
    )
    if sub:
        ax.annotate(
            sub,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            color=MUTED,
            fontsize=7.2,
            linespacing=1.45,
            annotation_clip=False,
        )


def wrap(text: str, width: int = 132) -> str:
    """Hard-wrap a provenance / caption paragraph so it cannot widen a figure."""
    import textwrap

    return "\n".join(
        "\n".join(textwrap.wrap(para, width=width)) if para.strip() else ""
        for para in text.split("\n")
    )


def xgrid(ax) -> None:
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.yaxis.grid(False)


def ygrid(ax) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)


def footer(fig, text: str, y: float = 0.012) -> None:
    """Provenance line along the bottom edge: where every number came from."""
    fig.text(
        0.012,
        y,
        wrap(text, 150),
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=6.4,
        linespacing=1.6,
    )
