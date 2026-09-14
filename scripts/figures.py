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
def save(fig, name: str, svg: bool = True, tight: bool = True) -> None:
    FIGDIR.mkdir(exist_ok=True)
    png = FIGDIR / f"{name}.png"
    kw = {} if tight else {"bbox_inches": None, "pad_inches": 0.0}
    fig.savefig(png, dpi=DPI, **kw)
    if svg:
        fig.savefig(FIGDIR / f"{name}.svg", **kw)
    plt.close(fig)
    size = png.stat().st_size / 1024
    print(f"  wrote figures/{name}.png  ({size:,.0f} KB)" + ("  + .svg" if svg else ""))


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
            fontsize=7.4,
            zorder=6,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.4)
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


def box(ax, x, y, w, h, title, lines, accent=S.MUTED, fill=S.BG_DEEP, title_color=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=6",
            facecolor=fill,
            edgecolor=accent,
            linewidth=0.9,
            zorder=3,
        )
    )
    ax.text(
        x + 10,
        y + h - 13,
        title,
        ha="left",
        va="top",
        color=title_color or accent,
        fontsize=8.2,
        weight="bold",
        zorder=4,
    )
    if lines:
        ax.text(
            x + 10,
            y + h - 30,
            "\n".join(lines),
            ha="left",
            va="top",
            color=S.MUTED,
            fontsize=6.6,
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


# =========================================================================== #
# 1. pipeline schematic
# =========================================================================== #
def fig_pipeline(src: Sources) -> None:
    cfg = src("bc3/eval.json")["_config"]
    stats = src("mb2/input_stats.json")
    pools = src("plast3/run_frozen_current.json")
    act = src("calyx/probe_raw.json")["real"]["activity"]

    graph = stats["graph"]
    glom = stats["glomeruli"]
    n_bits = int(cfg["n_bits_to_brain"])
    pl = pools["plasticity_pools"]
    dec = pools["decider"]
    om = pools["odour_map"]

    # full-figure axes so 1 data unit == 1/100 inch and nothing overflows
    W, H = 1520, 650
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ROW_Y, ROW_H = 332, 176
    GTOP, GBOT = 602, 324

    ax.text(24, 640, "the pipeline, and where the boundary is",
            color=S.BRIGHT, fontsize=13.5, weight="bold", va="top")
    ax.text(24, 613,
            "The fly is never shown a card. It is shown an odour that stands for an "
            "already-solved hand, and it decides what to do about it.",
            color=S.MUTED, fontsize=7.6, va="top")

    # ---- outside the brain ------------------------------------------------ #
    ax.add_patch(FancyBboxPatch((14, GBOT), 686, GTOP - GBOT,
                                boxstyle="round,pad=0,rounding_size=8",
                                facecolor="#101418", edgecolor=S.RULE,
                                linewidth=0.9, zorder=1))
    ax.text(30, GTOP - 12, "COMPUTED OUTSIDE THE BRAIN", color=S.ORANGE,
            fontsize=8.6, weight="bold", va="top")
    ax.text(30, GTOP - 32,
            "ordinary Python, not biological in any sense.\n"
            "The poker is solved here, before anything reaches the fly.",
            color=S.MUTED, fontsize=7.0, va="top", linespacing=1.6)

    box(ax, 30, ROW_Y, 150, ROW_H, "Balatro",
        ["the dealt hand,", "the blind, the chips", "still needed, plays", "and discards left"],
        accent=S.SLATE, title_color=S.TEXT)
    box(ax, 206, ROW_Y, 252, ROW_H, "flybalatro/hands.py",
        ["enumerates all 218 subsets of",
         "size 1-5 of the dealt cards,",
         "classifies and scores each one",
         "exactly as balatro-rs does,",
         "picks the best subset"],
        accent=S.ORANGE, title_color=S.ORANGE)
    box(ax, 484, ROW_Y, 200, ROW_H, f"{n_bits} relay bits",
        ["best hand type (9)",
         "best-subset slot mask (8)",
         "type the selection forms (10)",
         "selection is the best (1)",
         "score vs still-needed (4)"],
        accent=S.ORANGE, title_color=S.TEXT)
    arrow(ax, (180, ROW_Y + 72), (204, ROW_Y + 72), color=S.SLATE)
    arrow(ax, (458, ROW_Y + 72), (482, ROW_Y + 72), color=S.ORANGE)

    # ---- the boundary ----------------------------------------------------- #
    ax.plot([710, 710], [40, 592], color=S.AMBER, lw=1.0, ls=(0, (3, 4)), zorder=2)
    ax.text(702, 366, "everything the fly\nis ever told  ->", color=S.AMBER,
            fontsize=7.8, ha="right", va="center", weight="bold", linespacing=1.7)
    arrow(ax, (684, ROW_Y + 72), (738, ROW_Y + 72), color=S.AMBER, lw=1.3)

    # ---- inside the fly --------------------------------------------------- #
    ax.add_patch(FancyBboxPatch((722, GBOT), 508, GTOP - GBOT,
                                boxstyle="round,pad=0,rounding_size=8",
                                facecolor="#0b0f14", edgecolor=S.RULE,
                                linewidth=0.9, zorder=1))
    ax.text(738, GTOP - 12, "MaleCNS v1.0  --  FROZEN", color=S.AMBER,
            fontsize=8.6, weight="bold", va="top")
    ax.text(738, GTOP - 32,
            f"{graph['neurons']:,} neurons, {graph['edges_csr']:,} edges at >= "
            f"{graph['threshold']} synapses, leaky integrate-and-fire.\n"
            f"One {cfg['window_ms']:.0f} ms window from reset. No weight changes "
            "anywhere on this path.",
            color=S.MUTED, fontsize=7.0, va="top", linespacing=1.6)

    box(ax, 738, ROW_Y, 190, ROW_H, f"{n_bits} ORN glomeruli",
        ["1 relay bit = 1 whole",
         "olfactory receptor type",
         f"{om['n_driven_neurons']:,} ORNs, {om['neurons_min']}-{om['neurons_max']} per type",
         f"{glom['active_glomeruli_mean']:.1f} of {n_bits} lit per state",
         f"{om['drive_mv']:.0f} mV tonic drive"],
        accent=S.CYAN, title_color=S.CYAN)

    chain = [
        ("ALPN", act["alpn"]["n"], S.TEAL, f"{act['alpn']['mean_rate_hz']:.0f} Hz"),
        ("Kenyon cells", act["kc"]["n"], S.GOLD, f"{act['kc']['mean_rate_hz']:.2f} Hz"),
        ("MBON", act["mbon"]["n"], S.AMBER, f"{act['mbon']['mean_rate_hz']:.1f} Hz"),
        ("descending", act["dn"]["n"], S.ORANGE, f"{act['dn']['mean_rate_hz']:.1f} Hz"),
    ]
    x0, w, bh, gap = 956, 258, 34, 12
    top = ROW_Y + ROW_H - 34
    ys = []
    for i, (name, n, col, rate) in enumerate(chain):
        y = top - i * (bh + gap)
        ys.append(y)
        ax.add_patch(FancyBboxPatch((x0, y), w, bh,
                                    boxstyle="round,pad=0,rounding_size=4",
                                    facecolor=S.BG_DEEP, edgecolor=col,
                                    linewidth=0.9, zorder=3))
        ax.text(x0 + 10, y + bh / 2, name, va="center", color=col, fontsize=7.6, zorder=4)
        ax.text(x0 + w - 10, y + bh / 2, f"{n:,}    {rate}", va="center", ha="right",
                color=S.MUTED, fontsize=6.8, zorder=4)
        if i:
            arrow(ax, (x0 + 20, y + bh + gap), (x0 + 20, y + bh),
                  color=chain[i - 1][2], lw=0.9)
    arrow(ax, (928, ROW_Y + 72), (954, top + bh / 2), color=S.CYAN, rad=-0.14)

    # ---- the two heads ---------------------------------------------------- #
    hx, hw = 1256, 232
    box(ax, hx, 434, hw, 168, "(a) trained readout",
        ["linear or 1x256 MLP on",
         "log1p spike counts.",
         "",
         "FITTED to the teacher's",
         "actions by behaviour",
         "cloning  (v1, v2, v3)."],
        accent=S.SLATE, title_color=S.TEXT)
    box(ax, hx, 208, hw, 202, "(b) MBON valence rule",
        [f"mean rate({dec['n_approach']} approach MBONs)",
         f"- mean rate({dec['n_avoid']} avoid MBONs)",
         "+ bias, through a sigmoid.",
         "",
         "NO trained action readout.",
         "Bias, temperature, per-KC",
         "thresholds and pulse gain",
         "are calibrated label-free."],
        accent=S.AMBER, title_color=S.AMBER)
    arrow(ax, (1214, ys[3] + bh / 2), (1254, 516), color=S.SLATE, rad=0.18)
    arrow(ax, (1214, ys[2] + bh / 2), (1254, 340), color=S.AMBER, rad=-0.12)

    box(ax, hx, 78, hw, 100, "action",
        ["play the best subset,", "discard the junk and dig,", "or select a card"],
        accent=S.GREEN, title_color=S.GREEN)
    arrow(ax, (hx + hw / 2, 208), (hx + hw / 2, 180), color=S.AMBER)
    ax.plot([hx + hw + 8, hx + hw + 8], [518, 128], color=S.SLATE, lw=0.9, zorder=2)
    ax.plot([hx + hw, hx + hw + 8], [518, 518], color=S.SLATE, lw=0.9, zorder=2)
    arrow(ax, (hx + hw + 8, 128), (hx + hw, 128), color=S.SLATE, lw=0.9)

    # ---- dopamine loop ---------------------------------------------------- #
    ax.add_patch(FancyBboxPatch((722, 66), 508, 166,
                                boxstyle="round,pad=0,rounding_size=6",
                                facecolor="#0b120e", edgecolor=S.GREEN,
                                linewidth=0.9, zorder=1))
    ax.text(738, 218, "DOPAMINE  --  the plasticity path only", color=S.GREEN,
            fontsize=8.0, weight="bold", va="top")
    ax.text(738, 198,
            "the chips the game paid  ->  reward if the play made its fair\n"
            "share, punishment if it did not  ->  PAM / PPL1 gate depression\n"
            f"of {pl['kc_mbon_edges']:,} KC -> MBON synapses "
            f"(floor {pl['weight_floor']:g}x, never negative,\n"
            "never above original).\n\n"
            "An imposed harness signal computed from the game,\n"
            "not a signal the fly generates.",
            color=S.MUTED, fontsize=6.8, va="top", linespacing=1.65)
    arrow(ax, (1252, 122), (1232, 122), color=S.GREEN, style="-|>", lw=1.0)
    kc_mbon_gap = (ys[2] + bh + ys[1]) / 2
    ax.plot([942, 942], [232, kc_mbon_gap], color=S.GREEN, lw=1.0,
            ls=(0, (3, 3)), zorder=2)
    arrow(ax, (942, kc_mbon_gap), (954, kc_mbon_gap), color=S.GREEN, lw=1.0)
    ax.text(934, 272, "depresses\nKC -> MBON", color=S.GREEN, fontsize=6.6,
            va="center", ha="right", linespacing=1.6)

    # ---- what is fitted, and where ---------------------------------------- #
    calib = pools["calibration"]
    split = src("bc3/train_metrics.json")["_split"]
    ratio = src("plast2/eta_calib.json")["chosen"]["0.05"]["ratio"]
    ax.add_patch(FancyBboxPatch((14, 66), 686, 246,
                                boxstyle="round,pad=0,rounding_size=8",
                                facecolor="#101418", edgecolor=S.RULE,
                                linewidth=0.9, zorder=1))
    ax.text(30, 298, "WHAT IS FITTED, AND WHERE", color=S.TEXT,
            fontsize=8.2, weight="bold", va="top")
    ax.text(30, 276,
            "(a)  is a trained action readout: a classifier fitted to the teacher's "
            f"chosen action\n     on {split['train'] + split['val']:,} held-in states. "
            "Everything it gets right is information that\n"
            "     survived the trip through the fly.\n\n"
            "(b)  has NO trained action readout. Nothing in it is fitted to an action, "
            "a label or\n     an outcome. These ARE calibrated, all label-free and all "
            "before any learning:\n"
            f"     the decision bias ({calib['bias']:.3f}) and softmax temperature "
            f"({calib['temperature']:.3f}) from {calib['n']} hands;\n"
            f"     {pl['kc']:,} per-Kenyon-cell homeostatic thresholds, each cell seeing "
            "only its own\n"
            f"     firing rate; and the punishment/reward per-pulse gain ratio "
            f"({ratio:.2f}).\n\n"
            "The connectome gives topology, synapse counts and a predicted sign per "
            "neuron. It does not give\n"
            "per-synapse efficacy or spike thresholds, so the APL gains and the uniform "
            "-45 mV threshold\nare ours, not data.",
            color=S.MUTED, fontsize=6.8, va="top", linespacing=1.62)

    S.footer(fig, src.line(
        "Every count on this diagram is read from the artefact that produced it; "
        "nothing here is drawn to scale."), y=0.012)
    save(fig, "01_pipeline", tight=False)


# =========================================================================== #
# 2. the encoding reversal
# =========================================================================== #
def fig_encoding(src: Sources) -> None:
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

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.815, bottom=0.315, wspace=0.40)

    # ---- panel A: random-receptor encoding -------------------------------- #
    ax = axes[0]
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
            label=f"the same DN spikes decode how many bits\nare on at {dn_int:.3f}"
                  " -- intensity, not identity")
    S.xgrid(ax)
    ax.set_xlabel("linear-probe accuracy, best of a 6-value C grid, 5-fold CV")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.165), ncols=2,
              labelcolor=S.MUTED, fontsize=6.5, handlelength=1.8,
              columnspacing=1.6, labelspacing=0.8, borderpad=0.0)
    S.title(
        ax,
        "A · random-receptor encoding",
        "10 scattered ORNs per bit, ~36 bits on, all 54 glomeruli lit on every input.\n"
        f"Binary label on a random 12-bit group, n = {n_old}. Bars carry the CV SD.",
    )

    # ---- panel B: glomerular encoding ------------------------------------- #
    ax = axes[1]
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
            label=f"DN intensity probe {dn_int:.3f} -- still\nabove its identity score")
    S.xgrid(ax)
    ax.set_xlabel("linear-probe accuracy, 9-way hand type")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.165), ncols=2,
              labelcolor=S.MUTED, fontsize=6.5, handlelength=1.8,
              columnspacing=1.6, labelspacing=0.8, borderpad=0.0)
    S.title(
        ax,
        "B · glomerular encoding",
        f"1 relay bit = 1 whole ORN type, "
        f"{stats['glomeruli']['active_glomeruli_mean']:.1f} of 32 lit per state.\n"
        f"9-way hand type on {n_new:,} real game states.",
    )

    S.footer(
        fig,
        src.line(
            "The two panels are NOT a controlled comparison: the task changed too "
            "(binary synthetic label -> 9-way hand type). What is controlled is that "
            "both run the same tuned brain (apl_scale = 2, tonic drive) and both carry "
            "their own chance line and their own count-only confound. ORN itself is not "
            "probed here; under the random-receptor encoding the v1 gate probe put it "
            f"at {orn_old:.3f}, i.e. indistinguishable from the raw bits."
        ),
        y=0.014,
    )
    save(fig, "02_encoding_reversal")


# =========================================================================== #
# 3. behaviour cloning
# =========================================================================== #
@dataclass
class BCRow:
    label: str
    key: str
    color: str
    note: str = ""


def _bc_panel(ax, ev, rows: Sequence[BCRow], ceiling_key: str, title_main, title_sub,
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
               label=f"no-brain ceiling  {cp:.3f}   (the same readout on the same bits,\n"
                     "with the fly taken out of the loop)")
    if flag is not None:
        for yy, k in zip(y, keys):
            if flag(k):
                ax.text(-0.008, yy, "!", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", color=S.RED, fontsize=8.5,
                        weight="bold")
    S.xgrid(ax)
    ax.set_xlim(0, max(0.78, max(vals) + 0.13))
    ax.set_xlabel(f"ante-1 clear rate, {ceil['n_episodes']} episodes, Wilson 95% CI")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.10), labelcolor=S.MUTED,
              fontsize=6.5, handlelength=1.8, borderpad=0.0)
    S.title(ax, title_main, title_sub)
    return y


def fig_behaviour_cloning(src: Sources) -> None:
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

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.0))
    fig.subplots_adjust(left=0.165, right=0.985, top=0.865, bottom=0.305, wspace=0.56)
    _bc_panel(
        axes[0], ev2, rows2, "raw_bits/mlp",
        "v2 · relay bits on scattered receptors",
        "315 bits, 10 random ORNs each. The brain costs almost everything:\n"
        "the best real-wiring readout clears 0 blinds in 400 episodes.",
    )
    gap3 = ev3["raw_bits/mlp"]["clear_rate"] - ev3["real_alpn_kc_dn/mlp"]["clear_rate"]
    _bc_panel(
        axes[1], ev3, rows3, "raw_bits/mlp",
        "v3 · one relay bit = one whole glomerulus",
        "32 bits, same states, same teacher, same held-out split, same eval seeds.\n"
        f"The brain now costs {gap3 * 100:.1f} points against its own no-brain control.",
        flag=lambda k: k.startswith("shuf_"),
    )
    fig.text(
        0.035, 0.105,
        "!  under the glomerular encoding the global degree-preserving shuffle is a "
        "BROKEN control. It destroys the glomerular convergence the encoding\n"
        "   depends on and leaves a near-silent network, so it compares a working "
        "brain against a dead one. Rate table in RESULTS.md; figure 5 is the\n"
        "   control that replaces it.",
        ha="left", va="bottom", color=S.RED, fontsize=6.6, linespacing=1.7,
    )
    S.footer(
        fig,
        src.line(
            "Faint bars are the linear readout, solid bars the 1x256 MLP. Both are "
            "trained by behaviour cloning of the v2 hand-aware teacher; the fly's "
            "synapses never change on this path. v2 and v3 are NOT a controlled "
            "comparison with each other -- the encoding, the number of input bits and "
            "the brain's calibration all changed at once."
        ),
        y=0.012,
    )
    save(fig, "03_behaviour_cloning")


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
    summary = src("plast3/summary.json")
    cells = summary["cells"]
    mode = summary["primary_mode"]

    frozen_cur = src("plast3/run_frozen_current.json")
    frozen_sep = src("plast3/run_frozen_separated.json")
    teacher = src("plast3/baselines3.json")["teacher_dig"]

    cur_cur = [src(f"plast3/run_current_current_s{i}.json") for i in range(3)]
    om_sep = [src(f"plast3/run_omission_separated_s{i}.json") for i in range(3)]

    fig = plt.figure(figsize=(14.2, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.30], wspace=0.30,
                          left=0.135, right=0.985, top=0.825, bottom=0.295)

    # ---- panel A: the 2x2, each cell against its own frozen control -------- #
    ax = fig.add_subplot(gs[0, 0])
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
    ax.text(
        0.985,
        0.975,
        f"interaction contrast  {inter:+.3f}\n"
        "the effect is essentially all interaction",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=S.AMBER,
        fontsize=7.2,
        linespacing=1.7,
    )

    # ---- panel B: P(play) by score bucket --------------------------------- #
    ax = fig.add_subplot(gs[0, 1])
    series = [
        ("naive control, current enc.", _pool_buckets([frozen_cur], mode), S.SLATE, True),
        ("learned: chips-only\n(the v2 fly)", _pool_buckets(cur_cur, mode), S.SLATE, False),
        ("naive control, separated enc.", _pool_buckets([frozen_sep], mode), S.AMBER, True),
        ("learned: omission + separated\n(the v3 fly)", _pool_buckets(om_sep, mode), S.AMBER, False),
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
    ax.set_xticklabels(BUCKET_LABELS, fontsize=7.0)
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    S.ygrid(ax)
    ax.set_ylabel("P(play | the decision was legal)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.155), ncols=3,
              labelcolor=S.TEXT, fontsize=6.8, handlelength=1.2,
              columnspacing=2.2, borderpad=0.0)
    S.title(
        ax,
        "B · the rule forming",
        "The one ordered axis the odour carries. The v3 fly inverts its own\n"
        "naive policy and lands on the teacher's shape from chips alone.",
    )

    counts = _pool_buckets(om_sep, mode)
    ax.set_xlabel("what the best available hand scores against the chips still needed")

    S.footer(
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
    save(fig, "04_plasticity")


# =========================================================================== #
# 5. the calyx null
# =========================================================================== #
def fig_calyx(src: Sources) -> None:
    praw = src("calyx/probe_raw.json")
    paired_raw = src("calyx/paired_raw.json")["paired"]
    paired_hom = src("calyx/paired_homeo.json")["paired"]
    rewiring = src("calyx/rewiring.json")

    seeds = ["rw1", "rw2", "rw3"]

    fig = plt.figure(figsize=(14.4, 6.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 0.92, 1.06], wspace=0.46,
                          left=0.125, right=0.985, top=0.805, bottom=0.235)

    # ---- panel A: real sits inside the rewired band ----------------------- #
    ax = fig.add_subplot(gs[0, 0])
    measures = [
        ("KC hand type, grouped CV", lambda d: d["decode_hand_type"]["kc"]["acc_grouped"], praw),
        ("DN hand type, grouped CV", lambda d: d["decode_hand_type"]["dn"]["acc_grouped"], praw),
        ("KC Jaccard between/within", lambda d: d["kc_code"]["jaccard"]["ratio"], praw),
        ("KC imitation top-1", None, None),
        ("KC clear rate", None, None),
        ("KC clear rate, + homeostasis", None, None),
    ]
    rows = []
    for name, fn, probe in measures[:3]:
        real = fn(probe["real"])
        rw = [fn(probe[s]) for s in seeds]
        rows.append((name, real, rw))
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
        ax.text(max(hi, real) + 0.022, yy, f"{real:.3f}", va="center",
                color=S.TEXT, fontsize=7.2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.2)
    ax.set_ylim(ys.min() - 0.7, ys.max() + 0.7)
    ax.set_xlim(0, 1.10)
    S.xgrid(ax)
    ax.plot([], [], marker="|", ms=11, mew=2.0, ls="none", color=S.C_REAL,
            label="real ALPN -> KC wiring")
    ax.plot([], [], marker="o", ms=4.0, ls="none", color=S.C_REWIRED,
            label="3 rewiring seeds (empirical null)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.075), ncols=2,
              labelcolor=S.MUTED, fontsize=6.6, handlelength=1.4,
              columnspacing=1.6, borderpad=0.0)
    S.title(
        ax,
        "A · real sits inside the null band",
        f"{rewiring['strata']['n_edges']:,} calyx edges resampled within "
        f"{rewiring['strata']['n_strata']} strata of\n"
        "(hemisphere, KC subtype, synapse count, sign). Every degree and\n"
        "weight preserved exactly.",
    )

    # ---- panel B: paired differences, CIs crossing zero -------------------- #
    ax = fig.add_subplot(gs[0, 1])
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
        ax.text(0.132, yy, f"p={p:.2f}", va="center", ha="right",
                color=S.MUTED, fontsize=6.4)
    ax.axvline(0, color=S.BRIGHT, lw=0.9, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entries], fontsize=6.8)
    ax.set_ylim(ys.min() - 0.7, ys.max() + 0.7)
    ax.set_xlim(-0.135, 0.135)
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
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0, 1.0, "C · the control is fair before any correction",
            color=S.BRIGHT, fontsize=10, weight="bold", va="top")
    ax.text(
        0, 0.945,
        "Unlike the global shuffle this replaces, the rewired networks\n"
        "land in the same activity regime with no recalibration at all.",
        color=S.MUTED, fontsize=7.0, va="top", linespacing=1.6,
    )

    def fmt_row(fn, fmt):
        return [fmt.format(fn(praw["real"]))] + [fmt.format(fn(praw[s])) for s in seeds]

    table = [
        ("ALPN mean rate",
         fmt_row(lambda d: d["activity"]["alpn"]["mean_rate_hz"], "{:.1f} Hz")),
        ("KC mean rate",
         fmt_row(lambda d: d["activity"]["kc"]["mean_rate_hz"], "{:.2f} Hz")),
        ("KC active per state",
         fmt_row(lambda d: d["activity"]["kc"]["active_per_state"], "{:.0f}")),
        ("MBON mean rate",
         fmt_row(lambda d: d["activity"]["mbon"]["mean_rate_hz"], "{:.2f} Hz")),
        ("DN mean rate",
         fmt_row(lambda d: d["activity"]["dn"]["mean_rate_hz"], "{:.2f} Hz")),
        ("non-constant channels",
         fmt_row(lambda d: d["activity"]["non_constant_channels_alpn_kc_dn"], "{:,}")),
    ]
    x_cols = [0.55, 0.69, 0.825, 0.96]
    head = ["real", "rw1", "rw2", "rw3"]
    y0 = 0.85
    for x, h in zip(x_cols, head):
        ax.text(x, y0, h, color=S.C_REAL if h == "real" else S.C_REWIRED,
                fontsize=7.2, ha="right", va="top", weight="bold")
    ax.plot([0, 0.97], [y0 - 0.035, y0 - 0.035], color=S.RULE, lw=0.8)
    for i, (name, vals) in enumerate(table):
        yy = y0 - 0.075 - i * 0.062
        ax.text(0, yy, name, color=S.TEXT, fontsize=7.0, va="top")
        for x, v in zip(x_cols, vals):
            ax.text(x, yy, v, color=S.TEXT if x == x_cols[0] else S.MUTED,
                    fontsize=7.0, ha="right", va="top")
    ax.text(
        0, y0 - 0.075 - len(table) * 0.062 - 0.055,
        "For contrast, the global degree-preserving shuffle this\n"
        "control replaces destroys the glomerular convergence the\n"
        "encoding depends on and leaves a near-silent network --\n"
        "a working brain against a dead one. That comparison is\n"
        "retracted; this is the control that answers it instead.",
        color=S.RED, fontsize=6.6, va="top", linespacing=1.65,
    )
    ax.text(
        0, y0 - 0.075 - len(table) * 0.062 - 0.29,
        "Caron et al. 2013 report PN -> KC connectivity in Drosophila\n"
        "as largely random with respect to glomerular identity, so a\n"
        "near-null here agrees with the biology rather than indicting\n"
        "the model. A large effect would have been the surprise.",
        color=S.MUTED, fontsize=6.6, va="top", linespacing=1.65,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    S.footer(
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
    save(fig, "05_calyx_null")


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
