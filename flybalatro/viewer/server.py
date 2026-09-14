"""Local demo viewer server: FastAPI + one websocket + one paced game loop.

Run::

    source .venv/bin/activate
    python -m flybalatro.viewer.server            # http://127.0.0.1:8765
    python -m flybalatro.viewer.server --condition shuffled

Per action the loop does: encode the game state to bits, turn those bits into
tonic current on olfactory receptor neurons, run the 50 ms window as five 10 ms
sub-steps (which is bit-identical to one 50 ms call, because the kernel carries
its own cursor), hand the spike counts to the readout, and step the environment.
The five sub-steps are sent as five binary frames so the browser can play the
window back as a propagating wave.

The encoding is whichever version the selected readout was trained on: 283 bits
(v1) or 315 (v2, the same bits behind a 32-bit hand-type relay block computed
by ``flybalatro.hands`` **outside the brain** and injected as receptor drive like
any other bit). The state frame carries that block decoded into labels so the
screen can show what was relayed and label it as externally computed.

Wire protocol
-------------

Text frames are JSON tagged by ``t``: ``hello`` (static setup), ``state`` (the
board the fly is about to act on, its input bits and, under encoding v2, the
relayed hand analysis they carry), ``decision`` (top-5
logits, chosen action, population rates, result), ``episode`` (run over) and
``status`` (controls, counters, RSS). Binary frames are spike sub-steps:
``<II`` (sub-step 0-4, count) followed by that many little-endian ``uint32``
*point* indices - already remapped from neuron index to point-cloud index, so
neurons with no soma in the volume simply do not appear.

Everything for one action is sent in a single burst and the browser schedules
the playback off its own clock; the numba kernel holds the GIL for ~150 ms per
action, so server-side pacing of individual sub-steps would visibly stutter.
"""

from __future__ import annotations

import os

# Pinned before numpy/numba import: a heavier experiment may be running beside
# this process and the viewer needs one core, not all of them.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import resource  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union  # noqa: E402

import numpy as np  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from .. import connectome as C  # noqa: E402
from .. import glomerular as G  # noqa: E402
from ..brain import Brain  # noqa: E402
from ..encode import FeatureMap, GlomerularMap  # noqa: E402
from ..encode import feature_map_for  # noqa: E402
from ..env import N_ACTIONS, BalatroEnv, IllegalActionError  # noqa: E402
from ..features import RANK_NAMES  # noqa: E402
from ..features_v2 import (  # noqa: E402
    HAND_BLOCK_OFFSETS,
    N_HAND_BLOCK,
    encoder_for_version,
    hand_block_summary,
    n_features_for_version,
)
from . import game_view  # noqa: E402
from . import soma  # noqa: E402
from .plastic import PlasticEngine, PlasticPolicy, resolve_spec  # noqa: E402
from .realgame_source import BUCKET_LABELS as REALGAME_BUCKET_LABELS  # noqa: E402
from .realgame_source import MISSING as REALGAME_MISSING  # noqa: E402
from .realgame_source import RealGameSource  # noqa: E402
from .policy import DecisionContext, Policy, select_policy  # noqa: E402

LOG = logging.getLogger("flyviewer")

STATIC_DIR = Path(__file__).resolve().parent / "static"
REALGAME_DIR = Path(__file__).resolve().parents[2] / "outputs" / "realgame"

SOURCES: Tuple[str, ...] = ("headless", "realgame")

WINDOW_MS: float = 50.0
N_SUBSTEPS: int = 5
SUBSTEP_MS: float = WINDOW_MS / N_SUBSTEPS

# Matching scripts/bc_brain_features.py so a brain-condition readout is fed
# exactly the features it was trained on.
BRAIN_SEED: int = 1
SHUFFLE_SEED: int = 0
FEATURE_MAP_SEED: int = 0
NEURONS_PER_FEATURE: int = 10
EDGE_THRESHOLD: int = 5

# v1 bit-strip blocks, from flybalatro/features.py's documented layout. v2
# prepends the relay block and shifts these by its width.
_V1_BLOCKS: Tuple[Tuple[str, str, int, int], ...] = (
    ("hand", "hand cards", 0, 144),
    ("hand_selected", "selected flags", 144, 152),
    ("selected", "selected cards", 152, 242),
    ("context", "plays / discards / score / stage", 242, 283),
)

_RELAY_LABELS: Dict[str, str] = {
    "best_type": "best hand type",
    "best_mask": "best-subset slots",
    "selected_type": "selected hand type",
    "selected_is_best": "selection is the best subset",
    "score_vs_needed": "best score vs still needed",
}


def bit_blocks(feature_version: int,
               encoding: str = G.ENCODING_FEATUREMAP,
               n_bits: Optional[int] = None) -> List[JsonDict]:
    """Named spans of the input vector, for the bit strip's tinting and key.

    ``sent`` is whether the block reaches the fly at all. Under ``glomerular32``
    only the 32 relay bits do: the 283 v1 state bits are dropped from the input
    entirely, and the strip has to say so rather than imply the brain sees them.

    ``n_bits`` caps the layout to the vector actually sent. The plastic mode
    encodes *only* the relay block (``hand_block``, 32 wide), so the 283 v1
    blocks behind it do not exist at all there and must not be drawn.
    """
    glom = encoding == G.ENCODING_GLOM32
    if feature_version == 1:
        blocks = [
            {"name": n, "label": label, "from": a, "to": b, "relay": False,
             "sent": True}
            for (n, label, a, b) in _V1_BLOCKS
        ]
    else:
        blocks = []
        for name, start, width in HAND_BLOCK_OFFSETS:
            if name == "v1_bits":
                continue
            blocks.append({
                "name": name, "label": _RELAY_LABELS.get(name, name),
                "from": start, "to": start + width, "relay": True, "sent": True,
            })
        for (n, label, a, b) in _V1_BLOCKS:
            blocks.append({
                "name": n,
                "label": label + (" (not sent to the brain)" if glom else ""),
                "from": a + N_HAND_BLOCK, "to": b + N_HAND_BLOCK, "relay": False,
                "sent": not glom,
            })
    if n_bits is not None:
        blocks = [b for b in blocks if int(b["from"]) < int(n_bits)]
    return blocks

CONDITIONS: Tuple[str, ...] = ("real", "shuffled")
SUIT_GLYPHS: Tuple[str, ...] = ("♠", "♣", "♥", "♦")
BLIND_NAMES: Tuple[str, ...] = ("Small Blind", "Big Blind", "Boss Blind")
STAGE_LABELS: Tuple[str, ...] = (
    "Pre-blind",
    "Small Blind",
    "Big Blind",
    "Boss Blind",
    "Cash out",
    "Shop",
    "Run won",
    "Run lost",
    "Tarot",
    "Spectral",
    "Booster pack",
)

JsonDict = Dict[str, object]


# --------------------------------------------------------------------------- #
# game-state snapshot                                                          #
# --------------------------------------------------------------------------- #
def _joker_name(joker: object) -> str:
    name = getattr(joker, "name", None)
    if isinstance(name, str) and name:
        return name
    try:
        return str(joker)
    except Exception:  # pragma: no cover - defensive, pylatro repr is opaque
        return type(joker).__name__


def board_snapshot(state: object) -> JsonDict:
    """Everything the right-hand table panel draws, as plain Python scalars.

    Takes the already-fetched state: ``engine.state`` clones the whole game, so
    the action loop fetches it once and passes it around.
    """
    selected_ids = frozenset(card.id for card in state.selected)
    cards: List[JsonDict] = []
    for card in state.available:
        enhancement = card.enhancement_name
        cards.append(
            {
                "rank_index": int(card.rank_index),
                "suit_index": int(card.suit_index),
                "rank": RANK_NAMES[int(card.rank_index)],
                "suit": SUIT_GLYPHS[int(card.suit_index)],
                "red": bool(int(card.suit_index) >= 2),
                "selected": bool(card.id in selected_ids),
                "enhancement": str(enhancement) if enhancement is not None else None,
                "chips": int(card.chips),
            }
        )

    stage = int(state.stage.int())
    blind = state.blind
    blind_name = (
        BLIND_NAMES[int(blind)] if isinstance(blind, int) and 0 <= int(blind) < 3 else None
    )
    required = int(state.required_score)
    score = int(state.score)
    return {
        "cards": cards,
        "hand_count": len(cards),
        "hand_size": int(state.hand_size),
        "n_selected": len(selected_ids),
        "chips": score,
        "required": required,
        "progress": min(1.0, score / required) if required > 0 else 0.0,
        "plays": int(state.plays),
        "discards": int(state.discards),
        "money": int(state.money),
        "stage": stage,
        "stage_name": STAGE_LABELS[stage] if 0 <= stage < len(STAGE_LABELS) else f"stage {stage}",
        "blind_name": blind_name,
        "ante": int(state.ante),
        "ante_end": int(state.ante_end),
        "round": int(state.round),
        "jokers": [_joker_name(j) for j in state.jokers],
    }


# --------------------------------------------------------------------------- #
# per-slot readout confidence, for the POV panel                               #
# --------------------------------------------------------------------------- #
def _softmax(values: NDArray[np.float32]) -> NDArray[np.float64]:
    z = np.asarray(values, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    total = e.sum()
    return e / total if total > 0 else np.full_like(e, 1.0 / max(1, e.size))


def slot_confidences(
    scores: NDArray[np.float32],
    mask: NDArray[np.int8],
    n_cards: int,
    has_logits: bool,
) -> List[JsonDict]:
    """``select_card[i]`` score per dealt hand slot, softmaxed over the legal ones.

    ``ACTION_NAMES[i] == f"select_card[{i}]"`` for the first 24 actions, so slot
    ``i`` is action ``i``. The confidence is a softmax over the *legal* select
    actions only -- it answers "which card would it pick next", not "will it
    play". Policies with no logits (random, heuristic) report ``conf: null`` so
    the screen can say so instead of drawing bars of noise.
    """
    slots: List[JsonDict] = []
    legal = [i for i in range(min(n_cards, N_ACTIONS)) if int(mask[i]) == 1]
    probs: Dict[int, float] = {}
    if has_logits and legal:
        p = _softmax(scores[legal])
        probs = {i: float(v) for i, v in zip(legal, p)}
    for i in range(n_cards):
        slots.append(
            {
                "slot": i,
                "legal": bool(int(mask[i]) == 1) if i < N_ACTIONS else False,
                "logit": float(scores[i]) if (has_logits and i < N_ACTIONS) else None,
                "conf": probs.get(i),
            }
        )
    return slots


def chosen_confidence(
    scores: NDArray[np.float32],
    mask: NDArray[np.int8],
    action: int,
    has_logits: bool,
) -> Optional[float]:
    """Softmax probability of the action taken, over every legal action."""
    if not has_logits:
        return None
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return None
    p = _softmax(scores[legal])
    where = np.flatnonzero(legal == action)
    return float(p[int(where[0])]) if where.size else None

# --------------------------------------------------------------------------- #
# simulation                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class LoopSettings:
    """Everything the controls can change at runtime."""

    pace_s: float = 1.2
    speed: float = 1.0
    paused: bool = False
    condition: str = "real"

    @property
    def interval_s(self) -> float:
        return max(0.08, self.pace_s / max(0.1, self.speed))


@dataclass
class Counters:
    episodes: int = 0
    wins: int = 0
    losses: int = 0
    actions: int = 0
    last_brain_ms: float = 0.0
    brain_ms_ema: float = 0.0


@dataclass
class ActionFrames:
    """One action's worth of output, ready to broadcast.

    ``state`` is ``None`` only for a bookkeeping tick in the plastic mode, where
    a game can end during the select-blind / cash-out housekeeping and there is
    no window to show.
    """

    state: Optional[JsonDict]
    substeps: List[bytes]
    decision: Optional[JsonDict]
    episode: Optional[JsonDict] = None
    #: Real-game mode only: the dopamine pulse a resolved hand bought, with no
    #: board and no spikes behind it. See ``RealGameFrames.outcome``.
    outcome: Optional[JsonDict] = None


@dataclass
class Simulation:
    """Graph, soma cloud, brains, feature map, env and policy in one place.

    The feature map, the state encoder and the env are rebuilt whenever the
    selected policy needs a different encoding version, because a readout
    trained on v2 bits has to be driven through a v2 feature map or it is being
    fed the wrong channels.
    """

    graph: C.Graph
    cloud: soma.SomaCloud
    settings: LoopSettings
    seed0: int
    prefer: str
    ante_end: int = 1
    max_steps: int = 500
    feature_version: int = 0
    encoding: str = G.ENCODING_FEATUREMAP
    feature_map: Optional[Union[FeatureMap, GlomerularMap]] = None
    encoder: Optional[Callable[[object], NDArray[np.float32]]] = None
    env: Optional[BalatroEnv] = None
    brains: Dict[str, Brain] = field(default_factory=dict)
    policy: Optional[Policy] = None
    policy_notes: List[str] = field(default_factory=list)
    counters: Counters = field(default_factory=Counters)
    populations: Dict[str, NDArray[np.int32]] = field(default_factory=dict)
    episode_index: int = 0
    #: Set in the live-learning mode; when it is here the game loop is the
    #: plasticity harness and there is no trained readout anywhere.
    plastic: Optional[PlasticEngine] = None
    n_features_override: Optional[int] = None
    #: ``headless`` drives the bundled pylatro game; ``realgame`` follows the
    #: decisions ``flybalatro/realgame/play.py`` is making against actual
    #: Balatro (or a recording of them). In that mode there is no env to step
    #: and no policy here -- the fly already decided, elsewhere.
    source: str = "headless"
    realgame: Optional[RealGameSource] = None
    #: Real-game mode only: the live window capture that puts Balatro itself in
    #: the dashboard, beside the brain. ``None`` in every other mode and under
    #: ``--no-game-view``; the panel then does not appear at all.
    game: Optional["GameStream"] = None
    #: Real-game mode only: the four populations the panel is about, as
    #: point-cloud indices, plus the counts the legend prints. Built once in
    #: :meth:`use_realgame` from the same brain ``flybalatro/realgame/plastic.py``
    #: ran, so the somata that flash are the ones the rule actually gates on.
    realgame_pools: Optional[JsonDict] = None

    # -- setup ------------------------------------------------------------
    @classmethod
    def build(
        cls,
        condition: str,
        ante_end: int,
        pace_s: float,
        prefer: str,
        seed0: int,
        max_steps: int,
        refresh_cloud: bool,
        plastic_config: str = "v2",
        plastic_weights: Optional[Path] = None,
        learning: bool = True,
        source: str = "headless",
        realgame_dir: Optional[Path] = None,
        replay: Optional[Path] = None,
        game_view_on: bool = True,
        game_fps: float = 12.0,
        game_width: int = 900,
    ) -> "Simulation":
        t0 = time.perf_counter()
        graph = C.load(EDGE_THRESHOLD)
        LOG.info("graph: %d neurons, %d edges (%.1fs)", graph.n, graph.m, time.perf_counter() - t0)

        cloud = soma.load_or_build(graph, refresh=refresh_cloud)
        LOG.info(
            "soma cloud: %d of %d neurons drawn (%d no soma, %d outside the brain box)",
            cloud.n_points,
            cloud.n_neurons,
            cloud.stats["n_no_soma"],
            cloud.stats["n_outside_brain_box"],
        )

        sim = cls(
            graph=graph,
            cloud=cloud,
            settings=LoopSettings(pace_s=pace_s, condition=condition),
            seed0=seed0,
            prefer=prefer,
            ante_end=ante_end,
            max_steps=max_steps,
            source=source,
            populations={
                "ORN": graph.orn_indices(),
                "ALPN": graph.alpn_indices(),
                "KC": graph.kc_indices(),
                "MBON": graph.mbon_indices(),
                "CX": graph.cx_indices(),
                "DN": graph.dn_indices(),
            },
        )
        if source == "realgame":
            sim.use_realgame(realgame_dir or REALGAME_DIR, replay)
            if game_view_on and replay is not None:
                # --replay steps through a *recording*. If Balatro happens to be
                # running, capturing it would stream a live window under boxes
                # taken from a hand that was played minutes or days ago -- a
                # picture that looks current and is not. There is no honest
                # frame to show for a recorded decision, so the panel is off.
                LOG.info("game view: off under --replay (a live capture would "
                         "not be the frame the recorded rects describe)")
            elif game_view_on:
                sim.use_game_view(fps=game_fps, width=game_width)
            return sim
        if prefer == "plastic":
            spec = resolve_spec(plastic_config)
            t1 = time.perf_counter()
            sim.plastic = PlasticEngine(
                graph, spec, learning=learning, weights=plastic_weights,
                seed0=None if seed0 == 0 else seed0,
            )
            LOG.info("plastic fly %s ready in %.1fs", spec.version,
                     time.perf_counter() - t1)
            sim.use_plastic()
        else:
            sim.use_condition(condition)
        sim.start_episode()
        return sim

    # -- real-game mode ----------------------------------------------------
    def use_realgame(self, out_dir: Path, replay: Optional[Path]) -> None:
        """Follow the real game instead of running one.

        The brain is still built, and built the same way ``play.py`` builds it
        (``glomerular32``, ``G.load_spec()``, ``shuffle_seed = 0``), because a
        run recorded before spike recording existed has to be re-simulated from
        its logged relay bits -- and that is only the same window if it is the
        same brain. No policy is selected: the fly's decision is in the record.
        """
        self.settings.condition = "real"
        self.policy = None
        self.policy_notes = []
        # Under this encoding the 32 relay bits are the entire input, and they
        # are the only bits the real-game log carries. Showing a 315-wide strip
        # would imply 283 bits were read and were off; they were neither.
        self.n_features_override = N_HAND_BLOCK
        self.use_encoding(2, G.ENCODING_GLOM32)
        brain = self.brain_for("real")
        self.realgame_pools = self.mushroom_body_pools(brain)
        self.realgame = RealGameSource(
            out_dir,
            replay=replay,
            point_index=self.cloud.point_index,
            resimulate=self.run_window,
        )
        LOG.info("real-game source: %s (%s)", self.realgame.feed_path,
                 "live" if self.realgame.live else "replay")

    def use_game_view(self, *, fps: float, width: int) -> None:
        """Put the real Balatro window in the dashboard, beside the brain.

        Constructed here rather than in ``create_app`` so a failure to import
        Quartz is reported at startup, in the log, next to everything else about
        the run -- and so the panel's state is part of ``status()`` from the
        first ``hello`` frame instead of appearing when the first capture lands.
        """
        capture = game_view.GameCapture(width=width)
        # One lookup now, purely so the log line is true. The stream re-looks-up
        # on every miss anyway, so starting with the game closed is fine.
        found = capture.find_window()
        self.game = GameStream(self, Hub(), capture, fps)
        LOG.info("game view: %.0f fps, %d px wide, %s", fps, width,
                 f"capturing {found.owner} window {found.window_id} "
                 f"({found.bounds[2]:.0f}x{found.bounds[3]:.0f} pt)" if found
                 else f"no window yet ({capture.state})")

    def mushroom_body_pools(self, brain: Brain) -> JsonDict:
        """The populations the plasticity rule runs on, as point-cloud indices.

        Derived here rather than read off the log: ``flybalatro/plasticity.py``
        assigns MBON valence from the brain's own weights and splits the KC ->
        MBON synapses by which dopamine cluster dominates each compartment, and
        this is that same code on the same brain -- ``brain_for("real")`` builds
        the fly ``flybalatro/realgame/plastic.py`` runs (``glomerular32``,
        ``G.load_spec()``, ``shuffle_seed = 0``).

        The pool sizes are the run's own fingerprint, and the record carries them
        too; :meth:`check_realgame_pools` compares the two rather than trusting
        that the two builds agreed.
        """
        from .. import plasticity as P  # noqa: PLC0415
        from .plastic import dan_indices  # noqa: PLC0415

        t0 = time.perf_counter()
        valence = P.mbon_valence_table(self.graph, brain.post, brain.weight,
                                       scheme="nt")
        plast = P.KcMbonPlasticity(brain, self.graph, valence)
        pam, ppl1 = dan_indices(self.graph)

        def pts(idx: NDArray[np.int32]) -> List[int]:
            p = self.cloud.point_index[np.asarray(idx, np.int64)]
            return [int(x) for x in p[p >= 0]]

        LOG.info(
            "mushroom body: %d approach / %d avoid MBONs, %d PAM / %d PPL1 "
            "DANs, %d KC -> MBON synapses (%.1fs)",
            len(valence.approach), len(valence.avoid), len(pam), len(ppl1),
            len(plast.all_edge), time.perf_counter() - t0,
        )
        return {
            "points": {
                "approach": pts(valence.approach),
                "avoid": pts(valence.avoid),
                "pam": pts(pam),
                "ppl1": pts(ppl1),
            },
            "n_approach": int(len(valence.approach)),
            "n_avoid": int(len(valence.avoid)),
            "n_pam": int(len(pam)),
            "n_ppl1": int(len(ppl1)),
            "n_reward_mbons": int(len(plast.targets["reward"].mbons)),
            "n_punish_mbons": int(len(plast.targets["punish"].mbons)),
            "kc_mbon_edges": int(len(plast.all_edge)),
            "n_kc": int(len(plast.kc)),
            "weight_floor": float(plast.floor_frac),
            "matches_log": None,
        }

    def check_realgame_pools(self, mb: JsonDict) -> None:
        """Compare the brain built here with the one the record was made on.

        The record carries the synapse count the player's own fly had. If it
        differs from this one's, the highlighted somata are a different fly's
        and the screen would be flashing the wrong neurons. Checked once, on the
        first record that carries the number, and reported either way rather
        than quietly assumed.
        """
        pools = self.realgame_pools
        if pools is None or pools.get("matches_log") is not None:
            return
        weights = mb.get("weights")
        logged = int(weights.get("n_edges", 0) or 0) if isinstance(weights, dict) else 0
        if logged <= 0:
            return
        ours = int(pools["kc_mbon_edges"])  # type: ignore[arg-type]
        pools["matches_log"] = bool(logged == ours)
        if logged == ours:
            LOG.info("mushroom body: %d KC -> MBON synapses, matching the record",
                     ours)
        else:
            LOG.error(
                "mushroom body MISMATCH: this brain has %d KC -> MBON synapses, "
                "the record was made on one with %d -- the highlighted somata "
                "are not the neurons that run's rule gated on", ours, logged,
            )

    # -- live-learning mode ------------------------------------------------
    def use_plastic(self) -> None:
        """Point the simulation at the plasticity harness instead of a readout."""
        engine = self.plastic
        if engine is None:  # pragma: no cover - build() guards this
            raise RuntimeError("no plastic engine")
        self.settings.condition = "real"
        self.policy = PlasticPolicy(engine)
        self.policy_notes = list(engine.notes)
        self.feature_version = 2
        self.encoding = G.ENCODING_GLOM32
        self.feature_map = engine.setup.gm
        self.encoder = None
        self.env = engine.env
        self.n_features_override = N_HAND_BLOCK
        for note in self.policy_notes:
            LOG.info("  plastic note: %s", note)

    def brain_for(self, condition: str) -> Brain:
        """The (cached) brain for a wiring condition; builds it on first use.

        Cached per ``(condition, encoding)``: ``glomerular32`` models were trained
        on a **tuned** brain (``apl_scale = 2``, ``outputs/mb2/tuned_config.json``)
        and feeding them counts from the untuned one is feeding them noise.
        """
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
        key = f"{condition}/{self.encoding}"
        brain = self.brains.get(key)
        if brain is not None:
            return brain
        t0 = time.perf_counter()
        if self.encoding == G.ENCODING_GLOM32:
            brain = G.brain_for(self.graph, condition, G.load_spec(),
                                shuffle_seed=SHUFFLE_SEED)
        else:
            real = self.brains.get(f"real/{self.encoding}")
            if real is None:
                real = Brain(self.graph, seed=BRAIN_SEED)
                real.warmup()
                self.brains[f"real/{self.encoding}"] = real
            brain = real if condition == "real" else real.shuffled(SHUFFLE_SEED)
        brain.warmup()
        self.brains[key] = brain
        LOG.info("brain[%s] ready in %.1fs", key, time.perf_counter() - t0)
        return brain

    def use_encoding(self, feature_version: int,
                     encoding: str = G.ENCODING_FEATUREMAP) -> None:
        """(Re)build the input map, the state encoder and the env for a policy."""
        if (feature_version == self.feature_version and encoding == self.encoding
                and self.env is not None):
            return
        self.feature_version = int(feature_version)
        self.encoding = str(encoding)
        if self.encoding == G.ENCODING_GLOM32:
            spec = G.load_spec()
            self.feature_map = G.map_for(self.graph, spec)
            detail = (f"32 relay bits -> 32 whole ORN glomeruli "
                      f"({sum(self.feature_map.group_sizes)} receptor neurons), "
                      f"{self.n_features - N_HAND_BLOCK} bits not sent, "
                      f"tuning {spec.tuning.label()}")
        else:
            self.feature_map = feature_map_for(
                self.graph, self.feature_version, seed=FEATURE_MAP_SEED,
                neurons_per_feature=NEURONS_PER_FEATURE,
            )
            detail = (f"{self.feature_map.groups.size} driven neurons "
                      f"(orn_first={self.feature_map.orn_first})")
        self.encoder = encoder_for_version(self.feature_version)
        self.env = BalatroEnv(
            ante_end=self.ante_end, max_steps=self.max_steps, mask_noop_actions=True,
            encoder=self.encoder, n_features=self.n_features,
        )
        LOG.info("encoding v%d / %s: %d bits -> %s", self.feature_version,
                 self.encoding, self.n_features, detail)

    @property
    def n_features(self) -> int:
        if self.n_features_override is not None:
            return int(self.n_features_override)
        return n_features_for_version(self.feature_version or 1)

    @property
    def n_bits_to_brain(self) -> int:
        return G.n_input_bits(self.encoding, self.n_features)

    def drive_bits(self, bits: NDArray[np.float32]) -> NDArray[np.float32]:
        """The slice of the encoded state that becomes receptor drive."""
        n = self.n_bits_to_brain
        return bits if len(bits) == n else bits[:n]

    def use_condition(self, condition: str) -> None:
        """Switch wiring condition and re-pick the policy for it."""
        if self.plastic is not None:
            # A degree-preserving shuffle leaves 664 of the 33,496 KC -> MBON
            # edges (outputs/plast/REPORT.md), so it deletes the pathway this
            # mode is about rather than rewiring it. There is nothing to show.
            LOG.info("plastic mode is real wiring only; ignoring %r", condition)
            self.settings.condition = "real"
            return
        self.settings.condition = condition
        self.policy, self.policy_notes = select_policy(
            self.graph, condition, prefer=self.prefer
        )
        LOG.info(
            "policy for %s wiring: %s (%s)", condition, self.policy.mode, self.policy.input_desc
        )
        for note in self.policy_notes:
            LOG.info("  policy note: %s", note)
        self.use_encoding(getattr(self.policy, "feature_version", 1),
                          getattr(self.policy, "encoding", G.ENCODING_FEATUREMAP))
        self.brain_for(condition)

    @property
    def brain(self) -> Brain:
        if self.plastic is not None:
            return self.plastic.setup.brain
        return self.brain_for(self.settings.condition)

    def start_episode(self) -> None:
        if self.plastic is not None:
            self.plastic.new_episode()
            return
        self.env.reset(seed=self.seed0 + self.episode_index)

    # -- one action -------------------------------------------------------
    def rates(self, counts: NDArray[np.int32]) -> Dict[str, float]:
        """Mean firing rate in Hz per population over the 50 ms window."""
        seconds = WINDOW_MS / 1000.0
        out: Dict[str, float] = {}
        for name, idx in self.populations.items():
            out[name] = float(counts[idx].mean() / seconds) if len(idx) else 0.0
        return out

    def run_window(
        self, bits: NDArray[np.float32]
    ) -> Tuple[NDArray[np.int32], List[NDArray[np.int32]], float]:
        """Run the 50 ms window as five 10 ms sub-steps.

        Returns final spike counts, the point indices that fired in each
        sub-step, and the wall-clock cost. Summing the sub-steps is identical
        to one 50 ms call; only the observation points differ.
        """
        brain = self.brain
        drive = self.feature_map.drive(self.drive_bits(bits))
        point_index = self.cloud.point_index
        brain.reset()
        previous = np.zeros(brain.n, dtype=np.int32)
        fired: List[NDArray[np.int32]] = []
        t0 = time.perf_counter()
        counts = previous
        for k in range(N_SUBSTEPS):
            counts, _ = brain.step(drive, SUBSTEP_MS, reset_counts=(k == 0))
            spiked = np.flatnonzero(counts > previous).astype(np.int32)
            points = point_index[spiked]
            fired.append(points[points >= 0].astype(np.uint32))
            previous = counts.copy()
        return counts, fired, time.perf_counter() - t0

    def act(self) -> ActionFrames:
        """One full action: encode, simulate, decide, step the environment."""
        if self.plastic is not None:
            return self.act_plastic()
        env = self.env
        mask = env.action_mask()
        state = env.engine.state
        bits = self.encoder(state)
        board = board_snapshot(state)
        relay = hand_block_summary(state) if self.feature_version >= 2 else None

        counts, fired, brain_seconds = self.run_window(bits)
        self.counters.last_brain_ms = brain_seconds * 1000.0
        ema = self.counters.brain_ms_ema
        self.counters.brain_ms_ema = (
            self.counters.last_brain_ms if ema == 0.0 else 0.8 * ema + 0.2 * self.counters.last_brain_ms
        )

        policy = self.policy
        if policy is None:  # pragma: no cover - build() always sets one
            raise RuntimeError("no policy selected")
        decision = policy.decide(
            DecisionContext(env=env, bits=bits, mask=mask, counts=counts)
        )

        state_frame: JsonDict = {
            "t": "state",
            "episode": self.counters.episodes,
            "step": int(env.steps),
            "board": board,
            "bits": [int(b) for b in bits.astype(np.uint8)],
            "n_bits": int(len(bits)),
            "n_bits_on": int(bits.sum()),
            "n_legal": int(mask.sum()),
            "relay": relay,
        }

        _features, _mask, reward, done, info = env.step(decision.action)
        self.counters.actions += 1

        substeps: List[bytes] = [
            struct.pack("<II", k, len(points)) + points.tobytes()
            for k, points in enumerate(fired)
        ]

        decision_frame: JsonDict = {
            "t": "decision",
            "action": int(decision.action),
            "action_name": str(info["action_name"]),
            "top": decision.top(5, mask),
            "has_logits": bool(decision.has_logits),
            "slots": slot_confidences(
                decision.scores, mask, len(board["cards"]), decision.has_logits
            ),
            "confidence": chosen_confidence(
                decision.scores, mask, decision.action, decision.has_logits
            ),
            "rates": self.rates(counts),
            "total_spikes": int(counts.sum()),
            "n_spiking": int((counts > 0).sum()),
            "chips_gained": int(info["chips_gained"]),
            "reward": float(reward),
            "brain_ms": round(brain_seconds * 1000.0, 1),
            "done": bool(done),
        }

        episode_frame: Optional[JsonDict] = None
        if done:
            won = bool(info["is_win"])
            self.counters.episodes += 1
            if won:
                self.counters.wins += 1
            else:
                self.counters.losses += 1
            episode_frame = {
                "t": "episode",
                "won": won,
                "truncated": bool(info["truncated"]),
                "chips_total": int(env.chips_total),
                "steps": int(env.steps),
                "seed": int(self.seed0 + self.episode_index),
                "episodes": self.counters.episodes,
                "wins": self.counters.wins,
                "losses": self.counters.losses,
            }
            self.episode_index += 1

        return ActionFrames(
            state=state_frame,
            substeps=substeps,
            decision=decision_frame,
            episode=episode_frame,
        )

    # -- one hand, live-learning mode -------------------------------------
    def act_plastic(self) -> ActionFrames:
        """One *hand*: the plasticity harness, with the fly's synapses moving.

        A hand is several environment steps (select each card of the target
        subset, then play or discard) but exactly one 50 ms brain window and one
        decision, so it is the unit this mode paces on.
        """
        engine = self.plastic
        if engine is None:  # pragma: no cover
            raise RuntimeError("no plastic engine")
        result = engine.step_hand(self.cloud.point_index, N_SUBSTEPS, SUBSTEP_MS)
        kind = str(result["kind"])
        if kind != "hand":
            # Housekeeping only: the game ended between hands, or the engine had
            # to take a repair step. Nothing to animate.
            episode = result.get("episode")
            if episode is not None:
                self.counters.episodes += 1
                if episode["won"]:
                    self.counters.wins += 1
                else:
                    self.counters.losses += 1
                episode = {"t": "episode", **episode, "episodes": self.counters.episodes,
                           "wins": self.counters.wins, "losses": self.counters.losses}
            return ActionFrames(state=None, substeps=[], decision=None,
                                episode=episode)

        state = result["state"]
        bits = result["bits"]
        counts = result["counts"]
        self.counters.last_brain_ms = float(result["brain_seconds"]) * 1000.0
        ema = self.counters.brain_ms_ema
        self.counters.brain_ms_ema = (
            self.counters.last_brain_ms if ema == 0.0
            else 0.8 * ema + 0.2 * self.counters.last_brain_ms
        )
        self.counters.actions += 1

        state_frame: JsonDict = {
            "t": "state",
            "episode": self.counters.episodes,
            "step": int(engine.env.steps),
            "board": board_snapshot(state),
            "bits": [int(b) for b in np.asarray(bits).astype(np.uint8)],
            "n_bits": int(len(bits)),
            "n_bits_on": int(np.asarray(bits).sum()),
            "n_legal": 2 if result["mb"]["hand"]["discard_ok"] else 1,
            "relay": hand_block_summary(state),
        }
        substeps: List[bytes] = [
            struct.pack("<II", k, len(points)) + points.tobytes()
            for k, points in enumerate(result["fired"])
        ]
        decision_frame: JsonDict = {
            "t": "decision",
            "action": int(result["action_index"]),
            "action_name": str(result["action_name"]),
            "top": [],
            "has_logits": False,
            "slots": [],
            "confidence": None,
            "rates": self.rates(counts),
            "total_spikes": int(counts.sum()),
            "n_spiking": int((counts > 0).sum()),
            "chips_gained": int(result["chips_gained"]),
            "reward": float(result["mb"]["reward"]),
            "brain_ms": round(float(result["brain_seconds"]) * 1000.0, 1),
            "done": bool(result["done"]),
            "mb": result["mb"],
        }

        episode_frame: Optional[JsonDict] = None
        episode = result.get("episode")
        if episode is not None:
            self.counters.episodes += 1
            if episode["won"]:
                self.counters.wins += 1
            else:
                self.counters.losses += 1
            episode_frame = {
                "t": "episode", **episode,
                "episodes": self.counters.episodes,
                "wins": self.counters.wins,
                "losses": self.counters.losses,
            }
        return ActionFrames(state=state_frame, substeps=substeps,
                            decision=decision_frame, episode=episode_frame)

    # -- frames -----------------------------------------------------------
    #: The footer for the live-learning mode. Four sentences, each of which is a
    #: claim about where a computation happens, and all four are checkable in
    #: the code: ``flybalatro/viewer/plastic.py`` (the loop),
    #: ``flybalatro/plasticity.py`` (the rule) and ``flybalatro/hands.py`` (the
    #: poker).
    PLASTIC_FOOTER: str = (
        "No trained readout. Decision from the fly's own MBONs. Synapses change "
        "by the fly's own dopamine rule. Hand analysis computed outside the brain."
    )

    def footer(self) -> str:
        if self.source == "realgame":
            src = self.realgame
            where = ("the live game" if src is not None and src.live
                     else "a recorded run")
            return (
                f"Real MaleCNS v1.0 wiring, {self.graph.n:,} neurons, leaky "
                "integrate-and-fire, frozen. These are the decisions the fly "
                f"made playing Balatro 1.0.1o itself, read from {where}; the "
                "hand analysis behind the 32 relay bits is computed outside the "
                "brain (flybalatro/hands.py) and those 32 bits are the whole "
                "input. Nothing here is simulated forward of the fly's own play."
            )
        if self.plastic is not None:
            engine = self.plastic
            return (
                f"{self.PLASTIC_FOOTER} Real MaleCNS v1.0 wiring, "
                f"{self.graph.n:,} neurons, leaky integrate-and-fire; the only "
                f"mutable quantity is the weight of "
                f"{engine.setup.plast.describe()['kc_mbon_edges']:,} "
                f"KC → MBON synapses."
            )
        wiring = (
            "Real MaleCNS v1.0 wiring"
            if self.settings.condition == "real"
            else "Degree-preserving shuffled MaleCNS v1.0 wiring (control)"
        )
        kind = getattr(self.policy, "readout_kind", "readout")
        relay = (
            " The hand-type block of the input is poker analysis computed outside "
            "the brain (flybalatro/hands.py) and relayed to it as receptor drive."
            if self.feature_version >= 2 else ""
        )
        if self.encoding == G.ENCODING_GLOM32:
            relay += (
                f" Under this encoding those {N_HAND_BLOCK} relay bits are the "
                "*only* input: each drives every receptor neuron of one ORN "
                f"glomerulus, and the {self.n_features - N_HAND_BLOCK} game-state "
                "bits behind them are not sent to the brain at all."
            )
        return (
            f"{wiring}, {self.graph.n:,} neurons, leaky integrate-and-fire, frozen. "
            f"Only the {kind} on top is trained.{relay}"
        )

    def caveat(self) -> Optional[str]:
        if self.source == "realgame":
            src = self.realgame
            if src is None:  # pragma: no cover - use_realgame always sets one
                return None
            if src.last_mode == "re-simulated":
                return (
                    "Spikes are RE-SIMULATED: this run was logged before the "
                    "player recorded its windows, so the 32 relay bits from the "
                    "record are driven through the same brain again. Same "
                    "input, same frozen wiring, same reset -- the same window, "
                    "not an impression of it. Attach to a live run (or one "
                    "recorded with --spikes on) to watch the original spikes."
                )
            if src.last_mode == "none":
                return (
                    "No spikes for this decision: the record carries no "
                    "sub-step sidecar and the relay block is silent, so there "
                    "is no odour to re-drive. The point cloud stays dark."
                )
            return None
        if self.plastic is not None:
            engine = self.plastic
            if engine.core.learning:
                return None
            return (
                "Learning is OFF: the dopamine rule is disarmed and every "
                "KC → MBON weight is frozen, so the fly plays greedily on "
                "whatever it had already learned. Nothing on screen will change "
                "in the weight panel."
            )
        policy = self.policy
        if policy is None or policy.brain_in_loop:
            return None
        return (
            f"The readout in use does not read the brain: {policy.input_desc}. "
            "The network is really being driven and really spiking, but it is not "
            "choosing these actions."
        )

    def hello(self) -> JsonDict:
        if self.policy is None and self.source != "realgame":  # pragma: no cover
            raise RuntimeError("no policy selected")
        meta = self.graph.meta
        status = dict(self.status())
        status.pop("t", None)
        return {
            "t": "hello",
            "n_neurons": int(self.graph.n),
            "n_edges": int(self.graph.m),
            "n_synapses": int(meta.get("synapses_retained", 0)),
            "n_points": int(self.cloud.n_points),
            "categories": list(soma.CATEGORY_NAMES),
            "category_counts": self.cloud.category_counts(),
            "cloud_stats": dict(self.cloud.stats),
            "populations": {k: int(len(v)) for k, v in self.populations.items()},
            "window_ms": WINDOW_MS,
            "substeps": N_SUBSTEPS,
            "substep_ms": SUBSTEP_MS,
            "n_features": int(self.n_features),
            "feature_version": int(self.feature_version),
            "encoding": str(self.encoding),
            "bit_blocks": bit_blocks(self.feature_version, self.encoding,
                                     self.n_features),
            "plastic": self.mb_panel(),
            "highlight": self.highlight_points(),
            "n_relay_bits": int(N_HAND_BLOCK) if self.feature_version >= 2 else 0,
            "n_bits_to_brain": int(self.n_bits_to_brain),
            "glomeruli": self.glomeruli(),
            "missing": dict(REALGAME_MISSING) if self.source == "realgame" else None,
            "drive_mv": float(self.feature_map.drive_mv),
            "footer": self.footer(),
            "caveat": self.caveat(),
            **status,
        }

    def highlight_points(self) -> Optional[JsonDict]:
        """Point-cloud indices of the four populations the panel is about.

        The same four in both modes that have a mushroom-body panel: the two
        MBON pools the decision is the difference of, and the two dopamine
        clusters that gate the two halves of the rule.
        """
        if self.plastic is not None:
            return self.plastic.highlight_points(self.cloud.point_index)
        if self.realgame_pools is not None:
            points = self.realgame_pools["points"]
            assert isinstance(points, dict)
            return dict(points)
        return None

    def mb_panel(self) -> Optional[JsonDict]:
        """The static half of the mushroom-body panel, or None if there is none.

        Real-game mode fills in only what this process can derive from the
        connectome. ``bias`` and ``temperature`` are the *player's* operating
        point and are in every record it wrote, so the panel takes them from the
        frame rather than being told a second, possibly different pair here.
        """
        if self.plastic is not None:
            return self.plastic.describe()
        pools = self.realgame_pools
        if pools is None:
            return None
        src = self.realgame
        return {
            "version": "realgame",
            "mode": "realgame",
            # Two words: it is a tag beside a heading in a 340px column, and
            # the sentence about what the mode is belongs in the footer.
            "label": ("live game" if src is not None and src.live else "replay"),
            "homeostasis": None,
            "bias": None,
            "temperature": None,
            "explore_floor": None,
            "n_approach": pools["n_approach"],
            "n_avoid": pools["n_avoid"],
            "n_pam": pools["n_pam"],
            "n_ppl1": pools["n_ppl1"],
            "n_kc": pools["n_kc"],
            "weight_floor": pools["weight_floor"],
            "pools": {"kc_mbon_edges": pools["kc_mbon_edges"],
                      "reward_mbons": pools["n_reward_mbons"],
                      "punish_mbons": pools["n_punish_mbons"]},
            "buckets": [{"bucket": key, "label": label}
                        for key, label in REALGAME_BUCKET_LABELS],
        }

    def glomeruli(self) -> Optional[List[JsonDict]]:
        """Bit -> glomerulus for the sensory panel, or None if not glomerular."""
        fm = self.feature_map
        if self.encoding != G.ENCODING_GLOM32 or not isinstance(fm, GlomerularMap):
            return None
        from ..features_v2 import FEATURE_NAMES_V2

        return [
            {"bit": i, "feature": FEATURE_NAMES_V2[i],
             "glomerulus": name, "n_orn": int(size)}
            for i, (name, size) in enumerate(zip(fm.type_names, fm.group_sizes))
        ]

    def status(self) -> JsonDict:
        policy = self.policy
        src = self.realgame
        realgame: Optional[JsonDict] = src.describe() if src is not None else None
        if realgame is not None and self.realgame_pools is not None:
            realgame["pools_match"] = self.realgame_pools["matches_log"]
            realgame["kc_mbon_edges"] = self.realgame_pools["kc_mbon_edges"]
        if self.source == "realgame":
            label = ("real Balatro \u00b7 live" if src is not None and src.live
                     else "real Balatro \u00b7 replay")
            policy_fields: JsonDict = {
                "policy_mode": "realgame",
                "policy_label": label,
                "policy_input": REALGAME_MISSING["logits"],
                "policy_readout": "not logged",
                "brain_in_loop": True,
            }
        else:
            policy_fields = {
                "policy_mode": policy.mode if policy else "none",
                "policy_label": policy.label if policy else "none",
                "policy_input": policy.input_desc if policy else "none",
                "policy_readout": (getattr(policy, "readout_kind", "none")
                                   if policy else "none"),
                "brain_in_loop": bool(policy.brain_in_loop) if policy else False,
            }
        return {
            "t": "status",
            "source": self.source,
            "realgame": realgame,
            **policy_fields,
            "condition": self.settings.condition,
            "paused": bool(self.settings.paused),
            "speed": float(self.settings.speed),
            "interval_ms": round(self.settings.interval_s * 1000.0, 1),
            "feature_version": int(self.feature_version),
            "encoding": str(self.encoding),
            "n_features": int(self.n_features),
            "n_bits_to_brain": int(self.n_bits_to_brain),
            "policy_notes": list(self.policy_notes),
            "plastic_version": self.plastic.spec.version if self.plastic else None,
            "plastic_homeostasis": (bool(self.plastic.spec.homeostasis)
                                    if self.plastic else None),
            "learning": bool(self.plastic.core.learning) if self.plastic else None,
            "footer": self.footer(),
            "caveat": self.caveat(),
            "episodes": self.counters.episodes,
            "wins": self.counters.wins,
            "losses": self.counters.losses,
            "actions": self.counters.actions,
            "brain_ms": round(self.counters.brain_ms_ema, 1),
            "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
            "game": self.game.describe() if self.game is not None else None,
        }


# --------------------------------------------------------------------------- #
# broadcast hub                                                                #
# --------------------------------------------------------------------------- #
class Hub:
    """Connected websockets, plus the run/pause gate the loop waits on."""

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self.has_clients: asyncio.Event = asyncio.Event()

    def add(self, ws: WebSocket) -> None:
        self._clients.add(ws)
        self.has_clients.set()

    def discard(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        if not self._clients:
            self.has_clients.clear()

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    async def send(self, payload: Union[JsonDict, bytes]) -> None:
        if not self._clients:
            return
        text = None if isinstance(payload, bytes) else json.dumps(payload, default=_json_default)
        dead: List[WebSocket] = []
        for ws in list(self._clients):
            try:
                if text is None:
                    await ws.send_bytes(payload)  # type: ignore[arg-type]
                else:
                    await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.discard(ws)


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


# --------------------------------------------------------------------------- #
# the embedded game view                                                       #
# --------------------------------------------------------------------------- #
class GameStream:
    """Captures the real Balatro window and pushes it down the viewer socket.

    A second asyncio task beside the decision loop, not a second socket and not
    a second process. Each tick does one capture in a worker thread (Quartz and
    PIL both drop the GIL, so the numba window the decision loop is running is
    not blocked by it), sends the JPEG as one binary frame, and sleeps whatever
    is left of the interval. **Nothing is queued**: if a tick runs long the next
    one starts late and a frame is simply never taken. A buffered video stream
    would drift behind the brain it is supposed to be next to, and the one thing
    the panel must never do is show an old frame as if it were current.

    While the game is not there the tick rate drops to :data:`IDLE_INTERVAL_S`,
    because the only thing to do then is re-check whether it came back.
    """

    #: How often to look for the window when there is none, in seconds.
    IDLE_INTERVAL_S: float = 1.0
    #: Re-announce the capture state at least this often, so a browser that
    #: connected after the last change still learns what it is looking at.
    ANNOUNCE_EVERY_S: float = 2.0

    def __init__(self, sim: "Simulation", hub: Hub,
                 capture: game_view.GameCapture, fps: float) -> None:
        self.sim: "Simulation" = sim
        self.hub: Hub = hub
        self.capture: game_view.GameCapture = capture
        self.fps: float = float(max(1.0, min(30.0, fps)))
        self.n_frames: int = 0
        self.capture_ms_ema: float = 0.0
        self.encode_ms_ema: float = 0.0
        self.bytes_ema: float = 0.0
        self.fps_measured: float = 0.0
        self.last_frame_at: float = 0.0
        self.canvas: Optional[Tuple[float, float]] = None
        self._announced: str = ""
        self._announced_at: float = 0.0

    @property
    def interval_s(self) -> float:
        return 1.0 / self.fps

    def describe(self) -> JsonDict:
        out: JsonDict = dict(self.capture.describe())
        out.update({
            "fps_target": round(self.fps, 1),
            "fps": round(self.fps_measured, 1),
            "frames": self.n_frames,
            "capture_ms": round(self.capture_ms_ema, 1),
            "encode_ms": round(self.encode_ms_ema, 1),
            "bytes": int(round(self.bytes_ema)),
            # Draw units of the LOVE canvas this stream is a picture of. The
            # browser divides the card rects by it; sending it with the stream
            # rather than only with each decision means a frame that arrives
            # before the first decision still knows its own scale.
            "canvas": list(self.canvas) if self.canvas else None,
        })
        return out

    def _observe(self, frame: game_view.GameFrame) -> None:
        self.n_frames += 1
        now = time.perf_counter()
        if self.last_frame_at:
            dt = now - self.last_frame_at
            inst = 1.0 / dt if dt > 0 else 0.0
            self.fps_measured = inst if self.fps_measured == 0.0 else (
                0.8 * self.fps_measured + 0.2 * inst)
        self.last_frame_at = now
        for name, value in (("capture_ms_ema", frame.capture_ms),
                            ("encode_ms_ema", frame.encode_ms),
                            ("bytes_ema", float(frame.n_bytes))):
            prev = getattr(self, name)
            setattr(self, name, value if prev == 0.0 else 0.8 * prev + 0.2 * value)
        self.canvas = frame.canvas

    async def _announce(self, force: bool = False) -> None:
        now = time.perf_counter()
        if (not force and self.capture.state == self._announced
                and now - self._announced_at < self.ANNOUNCE_EVERY_S):
            return
        self._announced = self.capture.state
        self._announced_at = now
        await self.hub.send({"t": "game", **self.describe()})

    async def run(self) -> None:
        src = self.sim.realgame
        while True:
            try:
                await self.hub.has_clients.wait()
                t0 = time.perf_counter()
                hint = src.last_screen if src is not None else None
                frame = await asyncio.to_thread(self.capture.grab, hint)
                if frame is not None:
                    self._observe(frame)
                    await self.hub.send(frame.blob)
                await self._announce()
                spent = time.perf_counter() - t0
                floor = self.interval_s if frame is not None else self.IDLE_INTERVAL_S
                await asyncio.sleep(max(0.005, floor - spent))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - keep the demo alive
                LOG.error("game capture error: %s\n%s", exc, traceback.format_exc())
                await asyncio.sleep(1.0)


# --------------------------------------------------------------------------- #
# game loop                                                                    #
# --------------------------------------------------------------------------- #
class GameLoop:
    def __init__(self, sim: Simulation, hub: Hub) -> None:
        self.sim: Simulation = sim
        self.hub: Hub = hub
        self._step_once: asyncio.Event = asyncio.Event()
        self._reset_requested: bool = False
        self._condition_request: Optional[str] = None
        self._lock: asyncio.Lock = asyncio.Lock()

    # -- controls ---------------------------------------------------------
    async def handle(self, message: JsonDict) -> None:
        cmd = str(message.get("cmd", ""))
        settings = self.sim.settings
        if self.sim.source == "realgame" and cmd in ("condition", "reset", "learning"):
            # The real game already happened. Its wiring was whatever play.py
            # ran, a reset would mean restarting Balatro, and there is no
            # plasticity here -- say so rather than pretend the button worked.
            await self.hub.send({
                "t": "notice",
                "text": f"{cmd}: not available while following the real game",
            })
            await self.hub.send(self.sim.status())
            return
        if cmd == "pause":
            settings.paused = True
        elif cmd == "play":
            settings.paused = False
        elif cmd == "toggle":
            settings.paused = not settings.paused
        elif cmd == "step":
            settings.paused = True
            self._step_once.set()
        elif cmd == "speed":
            settings.speed = float(min(8.0, max(0.1, float(message.get("value", 1.0)))))
        elif cmd == "pace":
            settings.pace_s = float(min(10.0, max(0.1, float(message.get("value", 1.2)))))
        elif cmd == "reset":
            self._reset_requested = True
            self._step_once.set()
        elif cmd == "condition":
            value = str(message.get("value", "real"))
            if value in CONDITIONS:
                self._condition_request = value
                self._step_once.set()
        elif cmd == "learning":
            # Only the plastic mode has anything to arm or disarm. Turning it
            # off freezes every KC -> MBON weight where it is; turning it back on
            # resumes from there, it does not restart.
            if self.sim.plastic is not None:
                self.sim.plastic.core.learning = bool(message.get("value", True))
                state_word = "on" if self.sim.plastic.core.learning else "off"
                await self.hub.send({"t": "notice",
                                     "text": f"learning {state_word}"})
        else:
            LOG.warning("unknown command %r", cmd)
        await self.hub.send(self.sim.status())

    async def _apply_pending(self) -> None:
        condition = self._condition_request
        if condition is not None:
            self._condition_request = None
            await self.hub.send(
                {"t": "notice", "text": f"switching to {condition} wiring ..."}
            )
            await asyncio.to_thread(self.sim.use_condition, condition)
            self._reset_requested = True
        if self._reset_requested:
            self._reset_requested = False
            self.sim.start_episode()
            await self.hub.send(self.sim.status())

    # -- main -------------------------------------------------------------
    async def run(self) -> None:
        if self.sim.source == "realgame":
            await self.run_realgame()
            return
        await self.run_headless()

    async def run_realgame(self) -> None:
        """Follow ``outputs/realgame``: poll, build frames, broadcast.

        Nothing here touches the game. The only thing it writes is
        ``spikes.want``, a zero-byte file whose mtime tells a running
        ``play.py`` that someone is watching; the player checks it with one
        ``stat`` per decision and records its window only while it is fresh.
        """
        sim = self.sim
        src = sim.realgame
        if src is None:  # pragma: no cover - use_realgame always sets one
            raise RuntimeError("no real-game source")
        while True:
            try:
                await self.hub.has_clients.wait()
                src.request_spikes()
                if sim.settings.paused and not self._step_once.is_set():
                    await asyncio.sleep(0.05)
                    continue
                self._step_once.clear()

                decisions = await asyncio.to_thread(src.poll)
                if not decisions:
                    await asyncio.sleep(0.15)
                    continue
                for dec in decisions:
                    t0 = time.perf_counter()
                    frames = await asyncio.to_thread(src.frames, dec)
                    for frame in (frames.decision, frames.outcome):
                        mb = frame.get("mb") if frame is not None else None
                        if isinstance(mb, dict):
                            sim.check_realgame_pools(mb)
                    await self._broadcast(ActionFrames(
                        state=frames.state, substeps=frames.substeps,
                        decision=frames.decision, outcome=frames.outcome,
                    ))
                    if not src.live:
                        # A recording has no pace of its own; use the control.
                        remaining = sim.settings.interval_s - (time.perf_counter() - t0)
                        if remaining > 0:
                            await asyncio.sleep(remaining)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - keep the demo alive
                LOG.error("real-game loop error: %s\n%s", exc, traceback.format_exc())
                await self.hub.send({"t": "notice", "text": f"feed error: {exc}"})
                await asyncio.sleep(0.5)

    async def run_headless(self) -> None:
        sim = self.sim
        while True:
            try:
                await self.hub.has_clients.wait()
                await self._apply_pending()

                if sim.settings.paused and not self._step_once.is_set():
                    await asyncio.sleep(0.05)
                    continue
                self._step_once.clear()

                # The plastic harness owns its own episode boundaries (it has to:
                # a hand is several env steps and the game can end inside one),
                # so the loop must not reset the env underneath it.
                if sim.plastic is None and sim.env.done:
                    sim.start_episode()
                    await self.hub.send(sim.status())

                t0 = time.perf_counter()
                frames = await asyncio.to_thread(sim.act)
                await self._broadcast(frames)

                if frames.episode is not None:
                    await asyncio.sleep(2.0)
                    if sim.plastic is None:
                        sim.start_episode()
                    await self.hub.send(sim.status())
                else:
                    remaining = sim.settings.interval_s - (time.perf_counter() - t0)
                    if remaining > 0:
                        await asyncio.sleep(remaining)
            except asyncio.CancelledError:
                raise
            except (IllegalActionError, RuntimeError, ValueError) as exc:
                LOG.error("action failed, resetting episode: %s\n%s", exc, traceback.format_exc())
                await self.hub.send({"t": "notice", "text": f"error: {exc}; episode reset"})
                sim.episode_index += 1
                sim.start_episode()
                await asyncio.sleep(0.5)
            except Exception as exc:  # pragma: no cover - keep the demo alive
                LOG.error("unexpected loop error: %s\n%s", exc, traceback.format_exc())
                await asyncio.sleep(0.5)

    async def _broadcast(self, frames: ActionFrames) -> None:
        if frames.state is not None:
            await self.hub.send(frames.state)
        for blob in frames.substeps:
            await self.hub.send(blob)
        if frames.decision is not None:
            await self.hub.send(frames.decision)
        if frames.outcome is not None:
            await self.hub.send(frames.outcome)
        if frames.episode is not None:
            await self.hub.send(frames.episode)
        await self.hub.send(self.sim.status())


# --------------------------------------------------------------------------- #
# app                                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class AppState:
    sim: Simulation
    hub: Hub
    loop: GameLoop
    task: Optional[asyncio.Task] = None
    #: The window-capture task, when the game view is on.
    game_task: Optional[asyncio.Task] = None


def create_app(sim: Simulation) -> FastAPI:
    hub = Hub()
    state = AppState(sim=sim, hub=hub, loop=GameLoop(sim, hub))
    app = FastAPI(title="fly-balatro viewer", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    async def _startup() -> None:
        state.task = asyncio.create_task(state.loop.run())
        if sim.game is not None:
            sim.game.hub = hub
            state.game_task = asyncio.create_task(sim.game.run())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        for task in (state.task, state.game_task):
            if task is not None:
                task.cancel()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    # Declared before the /static mount, which would otherwise swallow it.
    @app.get("/static/neurons.bin")
    async def neurons_bin() -> FileResponse:
        if not soma.BIN_PATH.exists():
            soma.BIN_PATH.write_bytes(sim.cloud.to_bytes())
        return FileResponse(
            soma.BIN_PATH,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/status")
    async def api_status() -> JsonDict:
        payload = dict(sim.status())
        payload["clients"] = hub.n_clients
        return payload

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        hub.add(ws)
        try:
            await ws.send_text(json.dumps(sim.hello(), default=_json_default))
            while True:
                raw = await ws.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    LOG.warning("bad websocket payload: %r", raw[:120])
                    continue
                if isinstance(message, dict):
                    await state.loop.handle(message)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # pragma: no cover - client-side breakage
            LOG.info("websocket closed: %s", exc)
        finally:
            hub.discard(ws)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--condition", choices=CONDITIONS, default="real")
    parser.add_argument(
        "--source", choices=SOURCES, default="headless",
        help="'headless' runs the bundled pylatro game in this process; "
             "'realgame' follows the decisions flybalatro/realgame/play.py is "
             "making against actual Balatro, via outputs/realgame/latest.json",
    )
    parser.add_argument(
        "--realgame-dir", type=Path, default=REALGAME_DIR,
        help="where latest.json, log.jsonl and spikes.bin live",
    )
    parser.add_argument(
        "--replay", type=Path, default=None,
        help="with --source realgame: step through a recorded log instead of "
             "following the live one, so the brain view works with no game "
             "running (outputs/realgame/log.jsonl)",
    )
    parser.add_argument(
        "--no-game-view", dest="game_view", action="store_false",
        help="do not capture the Balatro window into the dashboard. The game "
             "panel is on by default with --source realgame on a LIVE feed; it "
             "is off in every other case -- under --replay a live capture "
             "would not be the frame the recorded rects describe, and in the "
             "headless mode there is no real window at all",
    )
    parser.add_argument(
        "--game-fps", type=float, default=12.0,
        help="frames per second for the embedded game panel (1-30)",
    )
    parser.add_argument(
        "--game-width", type=int, default=900,
        help="width the captured frame is downscaled to before encoding; it is "
             "never upscaled past the window's own size",
    )
    parser.add_argument("--ante-end", type=int, default=1)
    parser.add_argument("--pace", type=float, default=1.2, help="seconds per action at speed 1")
    parser.add_argument(
        "--policy",
        choices=("auto", "brain", "raw", "heuristic", "random", "plastic"),
        default="auto",
        help="'plastic' is the live-learning mode: no readout, the decision "
             "comes from the fly's own MBONs and its KC -> MBON synapses change "
             "on screen under its own dopamine rule",
    )
    parser.add_argument(
        "--plastic-config", choices=("v1", "v2"), default="v2",
        help="v1 = outputs/plast/tuned_config.json (learned 'always play'), "
             "v2 = outputs/plast2/tuned_config.json (per-KC homeostasis, learns "
             "an odour-specific rule)",
    )
    parser.add_argument(
        "--plastic-weights", default=None,
        help="start from weights saved by scripts/plast{,2}_run.py "
             "--save-weights; omit to start from the naive fly. Pass 'learned' "
             "for the chosen config's reference run.",
    )
    parser.add_argument(
        "--learning", choices=("on", "off"), default="on",
        help="'off' freezes every KC -> MBON weight and plays greedily",
    )
    parser.add_argument("--seed", type=int, default=0, help="first episode's engine seed")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--refresh-cloud", action="store_true")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import uvicorn

    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    weights: Optional[Path] = None
    if args.policy == "plastic" and args.plastic_weights:
        if str(args.plastic_weights) == "learned":
            weights = resolve_spec(args.plastic_config).weights_path
        else:
            weights = Path(args.plastic_weights)

    if args.replay is not None and args.source != "realgame":
        parser_error = "--replay only means anything with --source realgame"
        raise SystemExit(parser_error)

    sim = Simulation.build(
        condition=args.condition,
        ante_end=args.ante_end,
        pace_s=args.pace,
        prefer=args.policy,
        seed0=args.seed,
        max_steps=args.max_steps,
        refresh_cloud=args.refresh_cloud,
        plastic_config=args.plastic_config,
        plastic_weights=weights,
        learning=(args.learning == "on"),
        source=args.source,
        realgame_dir=args.realgame_dir,
        replay=args.replay,
        game_view_on=bool(args.game_view),
        game_fps=args.game_fps,
        game_width=args.game_width,
    )
    LOG.info(
        "ready: http://%s:%d  (source=%s, policy=%s, condition=%s, RSS %.0f MB)",
        args.host,
        args.port,
        args.source,
        sim.policy.mode if sim.policy else "none",
        args.condition,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    )
    uvicorn.run(create_app(sim), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
