"""Drive the real Balatro game with the fly connectome.

One decision is:

    gamestate (JSON-RPC)  ->  RealGameState  ->  state bits
      ->  FeatureMap.drive  ->  Brain: reset + one `window` ms window
      ->  population spike counts  ->  readout  ->  masked argmax
      ->  action index  ->  BalatroBot call

The brain pipeline is identical to ``scripts/bc_brain_features.py`` (the same
input map, the same brain, ``brain.reset()`` before each window, ``log1p`` of the
readout's populations), so a readout trained by ``scripts/bc_train.py`` sees
exactly the feature distribution it was fit on.

Models are looked for in ``outputs/bc3/models`` first (the v3 glomerular run),
then ``outputs/bc2/models`` (v2), then ``outputs/bc/models``. Each export says
which state encoding it needs and which input path: ``feature_version`` 1 is the
283 game bits, 2 prepends 32 bits of poker analysis computed **outside the brain**
by :mod:`flybalatro.hands` from the very same card fields the adapter already
exposes (``rank_index`` / ``suit_index`` / ``id``). That block is relayed to the
fly as receptor drive; it is logged per decision so a run log shows what the fly
was told and what it then did.

``encoding`` picks the input path: ``featuremap`` gives every bit 10 scattered
ORNs on an untuned ``Brain(seed=1)``; ``glomerular32`` sends **only** those 32
relay bits, one whole ORN glomerulus each, into the brain tuned by
``outputs/mb2/tuned_config.json`` (``apl_scale = 2``), and drops the 283 state
bits from the fly entirely.

Policy modes, in the order they are tried (the chosen one is printed):

``brain``
    A readout trained on brain spikes (``outputs/bc/models/real_alpn_kc_dn_*.npz``,
    or ``shuf_*`` under ``--condition shuffled``). The spiking network is in the
    decision path: this is the fly playing.
``bits`` / ``rand_proj``
    A readout trained on the raw 283 bits, or on a fixed random projection of
    them (``raw_bits_*``, ``rand_proj_*``). The brain is still driven and logged,
    but it does **not** decide. Reported as such, loudly, because it would be
    easy to mistake for the mode above.
``heuristic``
    ``scripts/baseline_heuristic.HeuristicPolicy`` chooses; the brain is still
    driven and logged. Used when no readout has been exported yet.

``--plastic`` replaces all of the above with the **learning** fly
(:mod:`flybalatro.realgame.plastic`): no trained action readout at all, the
decision comes out of the fly's own approach-minus-avoidance MBON drive, and the
game's chips deliver dopamine that depresses the eligible Kenyon-cell -> MBON
synapses while it plays. That mode has its own hand-level loop, because its
decision is one binary call per hand rather than one action index per step.

Usage::

    source .venv/bin/activate
    python -m flybalatro.realgame.play --ante-end 1              # frozen readout
    python -m flybalatro.realgame.play --plastic --ante-end 1    # learning fly

Logs one JSON object per decision to ``outputs/realgame/log.jsonl`` and
mirrors the newest one to ``outputs/realgame/latest.json`` (written to a temp
file and renamed, so a reader never sees a half-written object). The schema is
documented in ``docs/REALGAME_INSTALL.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .. import glomerular as G
from ..features_v2 import (
    N_HAND_BLOCK,
    encoder_for_version,
    hand_block_summary,
    n_features_for_version,
)
from . import adapter as ad
from .client import TRANSIENT_STATES, BalatroBot, BalatroBotError, TransportError

__all__ = [
    "Readout",
    "Policy",
    "BrainRunner",
    "PopulationRates",
    "DecisionLogger",
    "SpikeSidecar",
    "play",
    "main",
]

ROOT: Path = Path(__file__).resolve().parents[2]
MODELS: Path = ROOT / "outputs" / "bc" / "models"
#: Searched in order, newest run first.
MODEL_DIRS: Tuple[Path, ...] = (
    ROOT / "outputs" / "bc3" / "models",
    ROOT / "outputs" / "bc2" / "models",
    ROOT / "outputs" / "bc" / "models",
)
OUT_DIR: Path = ROOT / "outputs" / "realgame"
#: Named here only so ``--warm-start``'s help text can show it without
#: importing the plasticity stack (which loads the connectome).
PLAST2_WEIGHTS: Path = ROOT / "outputs" / "plast2" / "weights_real_eta0.05_s0.npz"

NEG: float = -1e9

#: Sub-steps one decision's window is observed in, matching ``N_SUBSTEPS`` in
#: ``flybalatro/viewer/server.py`` so the browser can play the window back as a
#: propagating wave. Five 10 ms calls are bit-identical to one 50 ms call --
#: ``Brain.step`` carries its own kernel cursor and only ``reset_counts``
#: differs -- so recording spikes cannot change what the readout sees or which
#: action the fly takes. ``tests/test_realgame_spikes.py`` asserts exactly that.
N_SUBSTEPS: int = 5

#: Where the per-sub-step spikes go, and the file a consumer touches to ask for
#: them. Recording is off unless something is listening, because it costs a
#: copy of the spike-count vector per sub-step and tens of KB per decision.
SPIKES_BIN: str = "spikes.bin"
SPIKES_WANT: str = "spikes.want"
#: How stale the request file may be before the consumer counts as gone.
SPIKES_WANT_MAX_AGE: float = 20.0

#: Readout files to try, in preference order: (condition, feature space).
#: Names match ``scripts/bc_train.py``'s ``CONDITIONS`` x kind.
_BRAIN_CONDITIONS: Mapping[str, Tuple[str, ...]] = {
    "real": ("real_alpn_kc_dn", "real_kc", "real_alpn", "real_mbon", "real_dn"),
    "shuffled": ("shuf_alpn_kc_dn", "shuf_kc", "shuf_alpn", "shuf_mbon", "shuf_dn"),
}
#: Readout name -> the population blocks its input is, in concatenation order.
BRAIN_POPS: Mapping[str, Tuple[str, ...]] = {
    "real_alpn_kc_dn": ("alpn", "kc", "dn"),
    "real_alpn": ("alpn",),
    "real_kc": ("kc",),
    "real_mbon": ("mbon",),
    "real_dn": ("dn",),
    "shuf_alpn_kc_dn": ("alpn", "kc", "dn"),
    "shuf_alpn": ("alpn",),
    "shuf_kc": ("kc",),
    "shuf_mbon": ("mbon",),
    "shuf_dn": ("dn",),
}
_BITS_CONDITIONS: Tuple[str, ...] = ("rand_proj", "rand_proj_matched", "raw_bits")
_KINDS: Tuple[str, ...] = ("mlp", "linear")


def model_candidates(names: Sequence[str]) -> List[Path]:
    """Existing export paths for ``names`` x kinds, newest run directory first."""
    out: List[Path] = []
    for directory in MODEL_DIRS:
        for name in names:
            for kind in _KINDS:
                path = directory / f"{name}_{kind}.npz"
                if path.exists():
                    out.append(path)
    return out


def detect_encoding_config(condition: str, prefer: str = "auto") -> Tuple[int, str]:
    """``(feature_version, encoding)`` of the readout ``select_policy`` will pick.

    Needed before the brain is built, because both the input map *and* the brain's
    tuning depend on them. ``select_policy`` re-checks and the caller asserts the
    two agree.
    """
    names: List[str] = []
    if prefer in ("auto", "brain"):
        names.extend(_BRAIN_CONDITIONS[condition])
    if prefer in ("auto", "bits"):
        names.extend(_BITS_CONDITIONS)
    for path in model_candidates(names):
        payload = np.load(path, allow_pickle=False)
        version = (int(payload["feature_version"])
                   if "feature_version" in payload.files else 1)
        encoding = (str(payload["encoding"]) if "encoding" in payload.files
                    else G.ENCODING_FEATUREMAP)
        return version, encoding
    return 1, G.ENCODING_FEATUREMAP


def detect_feature_version(condition: str, prefer: str = "auto") -> int:
    """Just the state encoding version; kept for callers that only need it."""
    return detect_encoding_config(condition, prefer)[0]


# -- readout ----------------------------------------------------------------


class Readout:
    """Numpy forward pass of one readout exported by ``scripts/bc_train.py``.

    Mirrors ``scripts/bc_eval.Readout`` and the ``export_numpy`` key layout
    (``kind``, ``mean``, ``std``, ``keep``, then ``W``/``b`` or ``W1``/``b1``/
    ``W2``/``b2``). Reimplemented here rather than imported so that this module
    does not depend on a script that owns multiprocessing globals.
    """

    def __init__(self, path: Path) -> None:
        payload = np.load(path, allow_pickle=False)
        self.path: Path = path
        self.kind: str = str(payload["kind"])
        self.feature_version: int = (
            int(payload["feature_version"]) if "feature_version" in payload.files else 1
        )
        self.encoding: str = (
            str(payload["encoding"]) if "encoding" in payload.files
            else G.ENCODING_FEATUREMAP
        )
        self.mean: NDArray[np.float32] = payload["mean"].astype(np.float32)
        self.std: NDArray[np.float32] = payload["std"].astype(np.float32)
        self.keep: NDArray[np.bool_] = payload["keep"].astype(bool)
        if self.kind == "linear":
            self.W: NDArray[np.float32] = payload["W"].astype(np.float32)
            self.b: NDArray[np.float32] = payload["b"].astype(np.float32)
        else:
            self.W1: NDArray[np.float32] = payload["W1"].astype(np.float32)
            self.b1: NDArray[np.float32] = payload["b1"].astype(np.float32)
            self.W2: NDArray[np.float32] = payload["W2"].astype(np.float32)
            self.b2: NDArray[np.float32] = payload["b2"].astype(np.float32)

    @property
    def n_inputs(self) -> int:
        return int(self.mean.shape[0])

    def logits(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        z = ((x.astype(np.float32) - self.mean) / self.std)[self.keep]
        if self.kind == "linear":
            return self.W @ z + self.b
        h = self.W1 @ z + self.b1
        np.maximum(h, 0.0, out=h)
        return self.W2 @ h + self.b2

    def act(self, x: NDArray[np.float32], mask: NDArray[np.int8]) -> int:
        logits = self.logits(x)
        return int(np.argmax(np.where(mask > 0, logits, NEG)))


# -- brain ------------------------------------------------------------------


@dataclass(frozen=True)
class PopulationRates:
    """Mean firing rate per population over one window, in Hz."""

    orn: float
    alpn: float
    kc: float
    mbon: float
    dn: float

    def as_dict(self) -> Dict[str, float]:
        return {"orn": self.orn, "alpn": self.alpn, "kc": self.kc,
                "mbon": self.mbon, "dn": self.dn}


class BrainRunner:
    """The frozen spiking network plus the feature -> ORN drive map.

    Holds the exact configuration the readouts were trained under.
    """

    def __init__(
        self,
        condition: str = "real",
        window_ms: float = 50.0,
        threshold: int = 5,
        brain_seed: int = 1,
        shuffle_seed: int = 0,
        fm_seed: int = 0,
        neurons_per_feature: int = 10,
        verbose: bool = True,
        graph: Optional[object] = None,
        feature_version: int = 1,
        encoding: str = G.ENCODING_FEATUREMAP,
    ) -> None:
        """
        Args:
            graph: a prebuilt ``connectome.Graph``. Defaults to the cached
                MaleCNS graph; injectable so the spiking path can be tested on
                a small synthetic graph without a ~1.5 GB load.
            feature_version: state encoding the loaded readout was trained on.
                Decides both the encoder and the feature map's layout.
            encoding: input path -- ``featuremap`` (10 scattered ORNs per bit,
                untuned brain) or ``glomerular32`` (32 whole glomeruli from the
                relay block only, on the ``outputs/mb2`` tuned brain).
        """
        if condition not in ("real", "shuffled"):
            raise ValueError("condition must be 'real' or 'shuffled'")
        if encoding not in G.ENCODINGS:
            raise ValueError(f"encoding must be one of {G.ENCODINGS}, got {encoding!r}")
        from ..brain import Brain
        from ..connectome import load as load_graph
        from ..encode import feature_map_for

        if graph is None:
            if verbose:
                print(f"[brain] loading MaleCNS graph (threshold={threshold})...", flush=True)
            graph = load_graph(threshold=threshold)
        self.graph = graph
        self.condition: str = condition
        self.feature_version: int = int(feature_version)
        self.encoding: str = str(encoding)
        self.n_features: int = n_features_for_version(self.feature_version)
        self.n_bits_to_brain: int = G.n_input_bits(self.encoding, self.n_features)
        self.encode = encoder_for_version(self.feature_version)

        if self.encoding == G.ENCODING_GLOM32:
            spec = G.load_spec()
            self.window_ms: float = float(spec.window_ms)
            self.feature_map = G.map_for(graph, spec)
            self.brain = G.brain_for(graph, condition, spec, shuffle_seed=shuffle_seed)
            detail = (f"32 whole glomeruli, {sum(self.feature_map.group_sizes)} "
                      f"receptor neurons, tuning {spec.tuning.label()}, "
                      f"{self.n_features - self.n_bits_to_brain} bits not sent")
        else:
            self.window_ms = float(window_ms)
            self.feature_map = feature_map_for(
                graph, self.feature_version, seed=fm_seed,
                neurons_per_feature=neurons_per_feature,
            )
            base = Brain(graph, seed=brain_seed)
            self.brain = base.shuffled(shuffle_seed) if condition == "shuffled" else base
            detail = f"orn_first={self.feature_map.orn_first}"

        # Population index sets, in the order scripts/bc_brain_features.py
        # concatenates them for this encoding (v3 adds MBON).
        self._alpn = graph.alpn_indices().astype(np.int32)
        self._kc = graph.kc_indices().astype(np.int32)
        self._dn = graph.dn_indices().astype(np.int32)
        self._mbon = graph.mbon_indices().astype(np.int32)
        self._orn = graph.orn_indices().astype(np.int32)
        self.readout_idx, self.pop_slices = G.population_indices(graph, self.encoding)
        self.pop_sizes: Dict[str, int] = {
            "ORN": len(self._orn), "ALPN": len(self._alpn), "KC": len(self._kc),
            "MBON": len(self._mbon), "DN": len(self._dn),
        }
        self.brain.warmup()
        #: Neuron indices that fired in each of the :data:`N_SUBSTEPS` windows
        #: of the last :meth:`run`, or ``None`` if that call did not record.
        self.last_substeps: Optional[List[NDArray[np.int32]]] = None
        #: Totals over the last recorded window: spikes, and distinct neurons
        #: that fired. Not derivable from :attr:`last_substeps` (a neuron can
        #: fire twice inside one sub-step and the sub-step only records that it
        #: fired), so they are kept here for the consumer.
        self.last_total_spikes: Optional[int] = None
        self.last_n_spiking: Optional[int] = None
        if verbose:
            print(
                f"[brain] condition={condition} n={graph.n} "
                f"window={self.window_ms:g}ms populations={self.pop_sizes} "
                f"feature_version={self.feature_version} encoding={self.encoding} "
                f"({self.n_features} bits, {self.n_bits_to_brain} to the brain, "
                f"{detail})",
                flush=True,
            )

    def drive_bits(self, bits: NDArray[np.float32]) -> NDArray[np.float32]:
        """The slice of the encoded state that becomes receptor drive."""
        n = self.n_bits_to_brain
        return bits if len(bits) == n else bits[:n]

    def run(
        self,
        bits: NDArray[np.float32],
        *,
        record_substeps: bool = False,
    ) -> Tuple[NDArray[np.float32], PopulationRates, float]:
        """Drive one window. Returns ``(readout_features, rates, seconds)``.

        ``readout_features`` is ``log1p`` of the recorded populations' spike counts
        (ALPN+KC+DN, or ALPN+KC+MBON+DN under ``glomerular32``), the
        representation ``bc_train`` fits on.

        With ``record_substeps`` the same window is run as :data:`N_SUBSTEPS`
        equal slices and :attr:`last_substeps` is left holding the neuron
        indices that fired in each -- what the viewer animates. The kernel
        carries its own cursor across calls and only the first slice resets the
        counters, so the counts at the end are the same array either way; the
        readout, and therefore the fly's decision, cannot tell the difference.
        """
        self.brain.reset()
        drive = self.feature_map.drive(self.drive_bits(bits))
        if not record_substeps:
            self.last_substeps = None
            self.last_total_spikes = self.last_n_spiking = None
            counts, elapsed = self.brain.step(drive, self.window_ms)
        else:
            sub_ms = self.window_ms / N_SUBSTEPS
            previous = np.zeros(self.brain.n, dtype=np.int32)
            fired: List[NDArray[np.int32]] = []
            elapsed = 0.0
            counts = previous
            for k in range(N_SUBSTEPS):
                counts, dt = self.brain.step(drive, sub_ms, reset_counts=(k == 0))
                fired.append(np.flatnonzero(counts > previous).astype(np.int32))
                previous = counts.copy()
                elapsed += dt
            self.last_substeps = fired
            self.last_total_spikes = int(counts.sum())
            self.last_n_spiking = int((counts > 0).sum())
        features = np.log1p(counts[self.readout_idx].astype(np.float32))

        seconds = self.window_ms / 1000.0

        def rate(idx: NDArray[np.int32]) -> float:
            if len(idx) == 0:
                return 0.0
            return float(counts[idx].sum()) / (len(idx) * seconds)

        rates = PopulationRates(
            orn=rate(self._orn), alpn=rate(self._alpn), kc=rate(self._kc),
            mbon=rate(self._mbon), dn=rate(self._dn),
        )
        return features, rates, elapsed


# -- policies ---------------------------------------------------------------


class Policy:
    """Chooses an action index; optionally consumes brain features.

    ``mode`` is one of ``brain``, ``rand_proj``, ``bits``, ``heuristic``.
    ``uses_brain`` says whether the spiking network is actually in the decision
    path, which is the distinction worth being honest about in a demo.
    """

    def __init__(self, mode: str, detail: str, uses_brain: bool,
                 feature_version: int = 1,
                 encoding: str = G.ENCODING_FEATUREMAP) -> None:
        self.mode: str = mode
        self.detail: str = detail
        self.uses_brain: bool = uses_brain
        self.feature_version: int = int(feature_version)
        self.encoding: str = str(encoding)

    def act(
        self,
        state: ad.RealGameState,
        bits: NDArray[np.float32],
        brain_features: NDArray[np.float32],
        mask: NDArray[np.int8],
    ) -> int:
        raise NotImplementedError


class ReadoutPolicy(Policy):
    """Masked argmax over a trained readout."""

    def __init__(
        self,
        readout: Readout,
        mode: str,
        detail: str,
        uses_brain: bool,
        projection: Optional[NDArray[np.float32]] = None,
        pop_take: Optional[NDArray[np.int64]] = None,
    ) -> None:
        super().__init__(mode, detail, uses_brain, readout.feature_version,
                         readout.encoding)
        self._readout = readout
        self._projection = projection
        #: Columns of the brain feature vector this readout consumes, or None for
        #: all of them. An index array rather than a slice because ALPN+KC+DN is
        #: not contiguous under glomerular32 (MBON sits between KC and DN).
        self._pop_take = None if pop_take is None else np.asarray(pop_take, np.int64)
        # A brain-free readout trained under glomerular32 saw only the 32 relay
        # bits, because that is all the fly saw. Feed it the same 32.
        self._n_bits: Optional[int] = (
            None if readout.encoding != G.ENCODING_GLOM32 else int(N_HAND_BLOCK)
        )

    def act(
        self,
        state: ad.RealGameState,
        bits: NDArray[np.float32],
        brain_features: NDArray[np.float32],
        mask: NDArray[np.int8],
    ) -> int:
        if self.mode == "brain":
            x = (brain_features if self._pop_take is None
                 else brain_features[self._pop_take])
        else:
            b = bits if self._n_bits is None else bits[: self._n_bits]
            if self._projection is not None:
                h = b.astype(np.float32) @ self._projection
                x = np.maximum(h, 0.0, dtype=np.float32)
            else:
                x = b.astype(np.float32)
        return self._readout.act(x, mask)


class _EngineShim:
    """Gives ``HeuristicPolicy`` the ``env.engine.state`` it reaches for."""

    def __init__(self, state: ad.RealGameState) -> None:
        self.state = state


class _EnvShim:
    def __init__(self, state: ad.RealGameState) -> None:
        self.engine = _EngineShim(state)


class HeuristicThroughBrainPolicy(Policy):
    """The hand-crafted baseline decides; the brain is driven and logged anyway."""

    def __init__(self) -> None:
        scripts_dir = ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        try:
            from baseline_heuristic_v2 import HandAwarePolicy  # noqa: E402

            inner: object = HandAwarePolicy()
            detail = ("scripts/baseline_heuristic_v2.HandAwarePolicy, hand-aware "
                      "(brain driven but not deciding)")
        except ImportError:
            from baseline_heuristic import HeuristicPolicy  # noqa: E402

            inner = HeuristicPolicy()
            detail = ("scripts/baseline_heuristic.HeuristicPolicy "
                      "(brain driven but not deciding)")
        super().__init__("heuristic", detail, uses_brain=False)
        self._inner = inner

    def act(
        self,
        state: ad.RealGameState,
        bits: NDArray[np.float32],
        brain_features: NDArray[np.float32],
        mask: NDArray[np.int8],
    ) -> int:
        return int(self._inner(_EnvShim(state), mask))


def select_policy(
    condition: str,
    brain: Union["BrainRunner", int, None] = None,
    prefer: str = "auto",
    verbose: bool = True,
    n_brain_inputs: Optional[int] = None,
    pop_slices: Optional[Mapping[str, slice]] = None,
) -> Policy:
    """Pick the best available policy. ``prefer`` is ``auto`` or a mode name.

    ``brain`` is a :class:`BrainRunner`, whose population widths a brain readout
    has to match; an ``int`` (the ALPN+KC+DN width) is accepted in its place, as
    is passing ``n_brain_inputs`` / ``pop_slices`` directly.
    """
    if prefer not in ("auto", "brain", "bits", "heuristic"):
        raise ValueError(f"unknown --readout {prefer!r}")
    if isinstance(brain, int):
        n_brain_inputs = brain
    elif brain is not None:
        n_brain_inputs = len(brain.readout_idx)
        pop_slices = brain.pop_slices
    slices: Mapping[str, slice] = pop_slices or {}

    if prefer == "heuristic":
        policy: Policy = HeuristicThroughBrainPolicy()
        _announce(policy, verbose)
        return policy

    if prefer in ("auto", "brain"):
        for path in model_candidates(_BRAIN_CONDITIONS[condition]):
            name = path.stem.rsplit("_", 1)[0]
            pops = BRAIN_POPS.get(name, ("alpn", "kc", "dn"))
            pop_take: Optional[NDArray[np.int64]] = None
            width = n_brain_inputs
            if slices and tuple(slices) != pops:
                if any(p not in slices for p in pops):
                    continue
                pop_take = np.concatenate(
                    [np.arange(slices[p].start, slices[p].stop) for p in pops]
                ).astype(np.int64)
                width = int(len(pop_take))
            readout = Readout(path)
            if width is not None and readout.n_inputs != width:
                if verbose:
                    print(
                        f"[policy] skipping {path.name}: expects "
                        f"{readout.n_inputs} inputs, this brain gives {width}",
                        flush=True,
                    )
                continue
            policy = ReadoutPolicy(
                readout, "brain",
                f"{path.parent.parent.name}/{path.name} over "
                + "+".join(p.upper() for p in pops)
                + f" spikes (state encoding v{readout.feature_version}, "
                + f"encoding {readout.encoding})",
                True, pop_take=pop_take,
            )
            _announce(policy, verbose)
            return policy
        if prefer == "brain":
            raise FileNotFoundError(
                f"no brain readout for condition {condition!r} in "
                + ", ".join(str(d) for d in MODEL_DIRS)
                + "; run scripts/bc_train.py, or pass --readout bits / heuristic"
            )

    for path in model_candidates(_BITS_CONDITIONS):
        name = path.stem.rsplit("_", 1)[0]
        projection: Optional[NDArray[np.float32]] = None
        if name.startswith("rand_proj"):
            proj_path = path.parent / f"{name}_projection.npz"
            if not proj_path.exists():
                continue
            projection = np.load(proj_path, allow_pickle=False)["W"].astype(np.float32)
        mode = "rand_proj" if projection is not None else "bits"
        readout = Readout(path)
        policy = ReadoutPolicy(
            readout, mode,
            f"{path.parent.parent.name}/{path.name} over "
            f"{'a random projection of ' if projection is not None else ''}"
            f"the {readout.n_inputs if projection is None else 'game'} state bits "
            f"(v{readout.feature_version}) - THE BRAIN IS NOT IN THE DECISION PATH",
            uses_brain=False, projection=projection,
        )
        _announce(policy, verbose)
        return policy

    policy = HeuristicThroughBrainPolicy()
    _announce(policy, verbose)
    return policy


def _announce(policy: Policy, verbose: bool) -> None:
    if not verbose:
        return
    banner = "FLY DECIDES" if policy.uses_brain else "brain driven, does NOT decide"
    print(f"[policy] mode={policy.mode} ({banner})", flush=True)
    print(f"[policy] {policy.detail}", flush=True)


# -- logging ----------------------------------------------------------------


class DecisionLogger:
    """Appends one JSON object per decision and mirrors the latest atomically.

    ``log.jsonl`` is the full history; ``latest.json`` always holds one
    complete object (written to a sibling temp file and renamed), so another
    process can poll it without ever reading a partial write.
    """

    def __init__(self, out_dir: Path = OUT_DIR) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir: Path = out_dir
        self.jsonl: Path = out_dir / "log.jsonl"
        self.latest: Path = out_dir / "latest.json"
        self._handle = self.jsonl.open("a", encoding="utf-8")

    def write(self, record: Mapping[str, object]) -> None:
        line = json.dumps(record, default=str)
        self._handle.write(line + "\n")
        self._handle.flush()
        tmp = self.latest.with_suffix(".json.tmp")
        tmp.write_text(line + "\n", encoding="utf-8")
        os.replace(tmp, self.latest)

    def close(self) -> None:
        self._handle.close()


class SpikeSidecar:
    """Per-sub-step spiking neuron indices, in the viewer's own wire layout.

    One blob per sub-step: ``<II`` (sub-step index, count) then that many
    little-endian ``uint32`` **neuron** indices. That is byte-for-byte the
    binary frame ``flybalatro/viewer/static/app.js`` already decodes, except
    that the viewer's frames carry *point-cloud* indices; the remapping is
    ``flybalatro/viewer/soma.py``'s business and a game player has no reason to
    know about it, so what lands here is the neuron index and the consumer
    remaps.

    It is a sidecar and not a field of the JSON record because a window is tens
    of thousands of spikes: as text in ``log.jsonl`` it would multiply the log
    by an order of magnitude and put that cost on ``flush()`` inside the game
    loop. The blobs are written and flushed **before** the record that points at
    them, so a reader that has the record always has the bytes.

    Recording is off unless a consumer asks: something touches
    :data:`SPIKES_WANT` (the viewer does, while a browser is connected) and
    :meth:`wanted` goes true within :data:`SPIKES_WANT_MAX_AGE` seconds.
    """

    def __init__(self, out_dir: Path = OUT_DIR, mode: str = "auto") -> None:
        if mode not in ("auto", "on", "off"):
            raise ValueError("mode must be auto, on or off")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.mode: str = mode
        self.path: Path = out_dir / SPIKES_BIN
        self.want_path: Path = out_dir / SPIKES_WANT
        #: Opened on the first write, so a run that nobody watches leaves no
        #: file behind at all.
        self._handle = None
        self.n_written: int = 0

    def wanted(self) -> bool:
        """Should this decision be recorded?"""
        if self.mode == "off":
            return False
        if self.mode == "on":
            return True
        try:
            age = time.time() - self.want_path.stat().st_mtime
        except OSError:
            return False
        return age < SPIKES_WANT_MAX_AGE

    def write(
        self,
        substeps: Sequence[NDArray[np.int32]],
        substep_ms: float,
        n_neurons: int,
        *,
        total_spikes: Optional[int] = None,
        n_spiking: Optional[int] = None,
    ) -> Dict[str, object]:
        """Append one window; returns the stub that goes in the JSON record."""
        if self.mode == "off":  # pragma: no cover - guarded by wanted()
            raise RuntimeError("sidecar is in mode 'off'")
        if self._handle is None:
            self._handle = self.path.open("ab")
        offset = self._handle.tell()
        payload = b"".join(
            struct.pack("<II", k, len(idx)) + np.asarray(idx, np.uint32).tobytes()
            for k, idx in enumerate(substeps)
        )
        self._handle.write(payload)
        self._handle.flush()
        self.n_written += 1
        return {
            "file": SPIKES_BIN,
            "offset": int(offset),
            "bytes": len(payload),
            "substeps": len(substeps),
            "substep_ms": float(substep_ms),
            "index_space": "neuron",
            "n_neurons": int(n_neurons),
            # Over the whole window, not per sub-step: a neuron can fire twice
            # inside one sub-step, so neither of these is recoverable from the
            # blobs and both are None when the caller did not supply them.
            "total_spikes": None if total_spikes is None else int(total_spikes),
            "n_spiking": None if n_spiking is None else int(n_spiking),
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


# -- the loop ---------------------------------------------------------------


def _state_summary(state: ad.RealGameState) -> Dict[str, object]:
    return {
        "raw_state": state.raw_state,
        "stage": state.stage.int(),
        "ante": state.ante_num,
        "round": state.round_num,
        "blind": state.blind_name,
        "blind_type": state.blind_type,
        "score": state.score,
        "required_score": state.required_score,
        "plays": state.plays,
        "discards": state.discards,
        "money": state.money,
        "n_jokers": len(state.jokers),
        "hand": [
            {"slot": i, "rank_index": c.rank_index, "suit_index": c.suit_index,
             "label": c.label}
            for i, c in enumerate(state.available)
        ],
        "selected_indices": list(state.selected_indices),
    }


def play(
    client: BalatroBot,
    brain: BrainRunner,
    policy: Policy,
    logger: DecisionLogger,
    deck: str = "RED",
    stake: str = "WHITE",
    seed: Optional[str] = None,
    pause: float = 1.0,
    max_decisions: int = 400,
    stop_after_ante: int = 1,
    start_new_run: bool = True,
    max_consecutive_errors: int = 5,
    verbose: bool = True,
    spikes: Optional[SpikeSidecar] = None,
) -> Dict[str, object]:
    """Run the decision loop until the run ends, the ante is cleared, or the cap.

    Returns a summary dict, also written to ``outputs/realgame/summary.json``.
    """
    if start_new_run:
        if verbose:
            print(f"[run] starting {deck} deck / {stake} stake"
                  + (f" / seed {seed}" if seed else ""), flush=True)
        client.menu()
        client.start(deck=deck, stake=stake, seed=seed)

    selection: Tuple[int, ...] = ()
    decisions = 0
    blinds_attempted: List[str] = []
    best_score = 0
    last_state: Optional[ad.RealGameState] = None
    stop_reason = "max_decisions"
    t_start = time.time()

    consecutive_errors = 0
    while decisions < max_decisions:
        payload = client.wait_until_stable()
        raw_state = payload.get("state")
        if raw_state == "MENU":
            stop_reason = "back_at_menu"
            break
        if isinstance(raw_state, str) and raw_state in TRANSIENT_STATES:
            # wait_until_stable gave up: the game is wedged in an animation.
            # Deciding from here would burn a full poll timeout per iteration.
            stop_reason = f"stuck_in_{raw_state}"
            break

        try:
            state = ad.build_state(payload, selection)
        except ad.UnmappableStateError as exc:
            stop_reason = f"unmappable_state:{exc}"
            break
        last_state = state
        selection = state.selected_indices

        if state.blind_name and (not blinds_attempted or blinds_attempted[-1] != state.blind_name):
            if state.stage.int() in ad.BLIND_STAGES:
                blinds_attempted.append(state.blind_name)
        best_score = max(best_score, state.score)

        stage = state.stage.int()
        if stage in (ad.STAGE_END_WIN, ad.STAGE_END_LOSE):
            stop_reason = "game_over_win" if stage == ad.STAGE_END_WIN else "game_over_lose"
            break
        if state.ante_num > stop_after_ante:
            stop_reason = "ante_cleared"
            break

        mask = ad.legality_mask(state)
        if int(mask.sum()) == 0:
            stop_reason = f"no_legal_action_in_{state.raw_state}"
            break

        bits = brain.encode(state)
        # Recording is per decision, not per run: a consumer can attach or drop
        # away mid-game and the loop follows it without restarting.
        record_spikes = spikes is not None and spikes.wanted()
        t_window = time.perf_counter()
        brain_features, rates, sim_seconds = brain.run(
            bits, record_substeps=record_spikes)
        spike_ref: Optional[Dict[str, object]] = None
        if record_spikes and brain.last_substeps is not None and spikes is not None:
            spike_ref = spikes.write(
                brain.last_substeps, brain.window_ms / N_SUBSTEPS, brain.brain.n,
                total_spikes=brain.last_total_spikes,
                n_spiking=brain.last_n_spiking,
            )
        window_seconds = time.perf_counter() - t_window
        action = policy.act(state, bits, brain_features, mask)
        if not 0 <= action < len(mask) or mask[action] == 0:
            # A readout can only pick a masked action if the mask it saw and the
            # mask applied here disagree; fail loudly rather than play garbage.
            raise RuntimeError(
                f"policy chose illegal action {action}; legal were "
                f"{np.flatnonzero(mask).tolist()}"
            )

        plan = ad.plan_action(state, action)
        decisions += 1

        record: Dict[str, object] = {
            "t": time.time(),
            "decision": decisions,
            "condition": brain.condition,
            "policy_mode": policy.mode,
            "policy_uses_brain": policy.uses_brain,
            "action_index": action,
            "action_name": plan.name,
            "action_kind": plan.kind,
            "action_description": plan.description,
            "n_legal": int(mask.sum()),
            "n_bits_on": int(bits.sum()),
            "feature_version": brain.feature_version,
            # Poker analysis computed outside the brain and relayed to it as
            # receptor drive; None under the v1 encoding, which has no such block.
            "relayed_hand_analysis": (
                hand_block_summary(state) if brain.feature_version >= 2 else None
            ),
            "window_ms": brain.window_ms,
            "sim_seconds": round(sim_seconds, 4),
            # Everything the window cost this loop, including the sub-step
            # bookkeeping and the sidecar write when one is attached; compare
            # against sim_seconds to see what recording adds.
            "window_seconds": round(window_seconds, 4),
            "spikes": spike_ref,
            "spike_rates_hz": rates.as_dict(),
            "population_sizes": brain.pop_sizes,
            "state": _state_summary(state),
        }
        logger.write(record)
        if verbose:
            print(
                f"[{decisions:3d}] {state.raw_state:<14} "
                f"score={state.score}/{state.required_score} "
                f"h={state.plays} d={state.discards} $={state.money} "
                f"-> {plan.name} ({plan.description}) | "
                f"DN {rates.dn:.1f}Hz KC {rates.kc:.1f}Hz",
                flush=True,
            )

        if plan.kind == "local":
            selection = plan.new_selection if plan.new_selection is not None else ()
            continue

        method = getattr(client, plan.client_method)
        try:
            method(**plan.kwargs)
        except BalatroBotError as exc:
            # The mod refused. Log it, drop the selection and let the next poll
            # re-read the truth rather than guessing what the game did. A demo
            # should survive one bad call; a storm of them means the mask and
            # the game genuinely disagree, so cap it.
            consecutive_errors += 1
            logger.write({"t": time.time(), "decision": decisions,
                          "error": str(exc), "error_name": exc.name,
                          "consecutive_errors": consecutive_errors,
                          "action_index": action, "action_name": plan.name})
            if verbose:
                print(f"      !! {exc}", flush=True)
            selection = ()
            if consecutive_errors >= max_consecutive_errors:
                stop_reason = f"too_many_rejected_calls:{exc.name}"
                break
            continue
        consecutive_errors = 0
        if plan.clears_selection:
            selection = ()
        if pause > 0:
            time.sleep(pause)

    summary: Dict[str, object] = {
        "stop_reason": stop_reason,
        "decisions": decisions,
        "wall_seconds": round(time.time() - t_start, 1),
        "condition": brain.condition,
        "policy_mode": policy.mode,
        "policy_detail": policy.detail,
        "policy_uses_brain": policy.uses_brain,
        "spike_windows_recorded": spikes.n_written if spikes is not None else 0,
        "blinds_attempted": blinds_attempted,
        "best_round_score": best_score,
        "final": _state_summary(last_state) if last_state is not None else None,
    }
    (logger.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if verbose:
        print(f"[run] {stop_reason} after {decisions} decisions", flush=True)
    return summary


def _main_plastic(args: argparse.Namespace, client: BalatroBot) -> int:
    """``--plastic``: the learning fly drives the game instead of the readout.

    Kept beside :func:`main` rather than inside it so the readout path reads
    exactly as it did. The weights are saved in a ``finally``, so a Ctrl-C
    halfway through a run still leaves what the fly had learned by then.
    """
    from .plastic import LearningTrace, OperatingPoint, PlasticFly, play_plastic

    out_dir = Path(args.out_dir)
    op = OperatingPoint.load(eta_reward=args.eta, eta_punish=args.eta_punish)
    fly = PlasticFly(
        op=op,
        seed=args.fly_seed,
        warm_start=Path(args.warm_start) if args.warm_start else None,
        learning=not args.no_learning,
        n_substeps=N_SUBSTEPS,
    )
    logger = DecisionLogger(out_dir)
    spikes = SpikeSidecar(out_dir, mode=args.spikes)
    print(f"[spikes] mode={args.spikes} sidecar={spikes.path} "
          f"(wanted now: {spikes.wanted()})", flush=True)
    weights_path = Path(args.save_weights) if args.save_weights else out_dir / "weights.npz"
    trace = LearningTrace()
    try:
        play_plastic(
            client, fly, logger,
            deck=args.deck, stake=args.stake, seed=args.seed, pause=args.pause,
            max_hands=args.max_hands, stop_after_ante=args.ante_end,
            start_new_run=not args.continue_run,
            max_consecutive_errors=args.max_errors,
            spikes=spikes, trace=trace,
        )
    except KeyboardInterrupt:
        print("\n[run] interrupted", flush=True)
    finally:
        logger.close()
        spikes.close()
        saved = fly.save_weights(weights_path, meta={"hands": len(trace.actions)})
        print(f"[fly] weights written to {saved}", flush=True)
        w = fly.weight_state()
        print(f"[fly] {w['n_changed']}/{w['n_synapses']} KC -> MBON synapses "
              f"changed ({w['frac_changed']:.1%}), mean w/w0 {w['mean_ratio']:.5f}, "
              f"{w['n_at_floor']} at the floor; "
              f"{w['n_reward_pulses']} reward and {w['n_punish_pulses']} "
              f"punishment pulses", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12346)
    parser.add_argument("--condition", choices=("real", "shuffled"), default="real",
                        help="brain wiring: real MaleCNS, or degree-preserving shuffle")
    parser.add_argument("--readout", default="auto",
                        choices=("auto", "brain", "bits", "heuristic"))
    parser.add_argument("--window", type=float, default=50.0,
                        help="brain integration window per decision, ms")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="seconds to wait after each game-changing call")
    parser.add_argument("--deck", default="RED")
    parser.add_argument("--stake", default="WHITE")
    parser.add_argument("--seed", default=None, help="Balatro run seed, e.g. TEST123")
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument("--ante-end", type=int, default=1,
                        help="stop once the run passes this ante")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--brain-seed", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--continue-run", action="store_true",
                        help="play the run already in progress instead of starting one")
    parser.add_argument("--max-errors", type=int, default=5,
                        help="stop after this many consecutive rejected calls")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--spikes", choices=("auto", "on", "off"), default="auto",
        help="record the per-sub-step spiking neurons to spikes.bin so the "
             "viewer can animate the real window. 'auto' (default) records only "
             "while a consumer keeps outputs/realgame/spikes.want fresh, which "
             "the viewer does under --source realgame; 'off' never opens the file",
    )
    parser.add_argument("--wait-api", type=float, default=0.0,
                        help="seconds to wait for the API before giving up")

    plastic = parser.add_argument_group(
        "plasticity mode",
        "Play with the LEARNING fly instead of the frozen readout: no trained "
        "action readout at all, the decision comes out of the fly's own MBON "
        "approach-minus-avoidance drive, and the game's chips deliver dopamine "
        "that depresses the eligible KC -> MBON synapses while it plays. "
        "See flybalatro/realgame/plastic.py and RESULTS.md 'Plasticity v2'.",
    )
    plastic.add_argument("--plastic", action="store_true",
                         help="use the learning fly (default: the frozen v3 readout)")
    plastic.add_argument("--eta", type=float, default=0.05,
                         help="reward learning rate (default 0.05, the plast2 "
                              "operating point); the punishment rate is the "
                              "calibrated 3.81x of it")
    plastic.add_argument("--eta-punish", type=float, default=None,
                         help="override the calibrated punishment learning rate")
    plastic.add_argument("--warm-start", nargs="?", const=str(PLAST2_WEIGHTS),
                         default=None, metavar="WEIGHTS.npz",
                         help="start from already-learned KC -> MBON weights "
                              "instead of naive. Bare flag uses "
                              f"{PLAST2_WEIGHTS.name}")
    plastic.add_argument("--save-weights", default=None, metavar="PATH",
                         help="where to write the learned weights (default "
                              "<out-dir>/weights.npz); written even on Ctrl-C")
    plastic.add_argument("--no-learning", action="store_true",
                         help="the frozen plastic fly: same MBON decision rule, "
                              "dopamine computed and logged but never delivered. "
                              "The control for 'did the weights do anything'")
    plastic.add_argument("--max-hands", type=int, default=200,
                         help="stop after this many hands (plastic mode)")
    plastic.add_argument("--fly-seed", type=int, default=0,
                         help="seed for the exploration draw")
    args = parser.parse_args(argv)

    client = BalatroBot(args.host, args.port)
    if args.wait_api > 0:
        print(f"[api] waiting up to {args.wait_api:g}s for {client.url}...", flush=True)
        if not client.wait_for_api(args.wait_api):
            print(f"[api] no response from {client.url}", file=sys.stderr)
            return 2
    try:
        health = client.health()
    except (TransportError, BalatroBotError) as exc:
        print(f"[api] {exc}", file=sys.stderr)
        print("Is the modded game running? See docs/REALGAME_INSTALL.md",
              file=sys.stderr)
        return 2
    print(f"[api] {client.url} health={health}", flush=True)

    if args.plastic:
        return _main_plastic(args, client)

    feature_version, encoding = detect_encoding_config(args.condition, args.readout)
    brain = BrainRunner(
        condition=args.condition, window_ms=args.window, threshold=args.threshold,
        brain_seed=args.brain_seed, shuffle_seed=args.shuffle_seed,
        feature_version=feature_version, encoding=encoding,
    )
    policy = select_policy(args.condition, brain, prefer=args.readout)
    if (policy.feature_version != brain.feature_version
            or policy.encoding != brain.encoding):
        print(
            f"[policy] readout needs state encoding v{policy.feature_version} / "
            f"{policy.encoding} but the brain was built for "
            f"v{brain.feature_version} / {brain.encoding}; rebuilding the input path",
            file=sys.stderr,
        )
        brain = BrainRunner(
            condition=args.condition, window_ms=args.window, threshold=args.threshold,
            brain_seed=args.brain_seed, shuffle_seed=args.shuffle_seed,
            feature_version=policy.feature_version, encoding=policy.encoding,
            graph=brain.graph,
        )
    logger = DecisionLogger(Path(args.out_dir))
    spikes = SpikeSidecar(Path(args.out_dir), mode=args.spikes)
    print(f"[spikes] mode={args.spikes} sidecar={spikes.path} "
          f"(wanted now: {spikes.wanted()})", flush=True)
    try:
        play(
            client, brain, policy, logger,
            deck=args.deck, stake=args.stake, seed=args.seed, pause=args.pause,
            max_decisions=args.max_decisions, stop_after_ante=args.ante_end,
            start_new_run=not args.continue_run,
            max_consecutive_errors=args.max_errors,
            spikes=spikes,
        )
    finally:
        logger.close()
        spikes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
