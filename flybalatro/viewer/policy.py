"""Action selection for the viewer, and the honesty labels that go with it.

Four implementations of one interface, tried in this order at startup:

``brain_readout``
    A readout trained by ``scripts/bc_train.py`` on log1p spike counts of the
    ALPN + KC + DN populations (or ALPN / DN alone). The brain is in the
    decision path: this is the real thing.
``raw_bits``
    The same readout architecture trained on the game bits directly. The brain
    is still simulated and drawn, but it does **not** decide anything. The UI
    must say so.
``heuristic_through_brain``
    ``scripts/baseline_heuristic.py`` picks the action; the brain is still
    driven and drawn. Also not in the decision path.
``random_legal``
    Uniform over the legal mask. Last resort.

Features must match ``scripts/bc_brain_features.py`` exactly or a trained
readout is being fed the wrong thing: a 50 ms window from a fresh ``reset()`` and
``log1p(counts)`` rounded through float16, driven through the input path the model
was trained on -- ``encode.feature_map_for(graph, feature_version)`` (seed 0, 10
neurons per feature) on an untuned ``Brain(graph, seed=1)`` for ``featuremap``
models, or ``flybalatro.glomerular`` (32 whole ORN glomeruli, ``apl_scale = 2``)
for ``glomerular32`` ones.

Three model directories are searched, newest run first: ``outputs/bc3`` (the v3
glomerular run), then ``outputs/bc2`` (v2, which prepends a 32-bit hand-type
relay block to 283 state bits), then ``outputs/bc`` (the original 283-bit run).
Each export carries ``feature_version`` and ``encoding``; the policy exposes both
and the server builds the matching encoder, input map and brain. Missing fields
mean 1 and ``featuremap``, so nothing about the old runs had to change.

Within a directory the candidates are ordered by held-out imitation top-1 from
that run's ``train_metrics.json`` when it is there, so "the viewer shows the best
real-wiring readout" stays true without a hand-maintained list.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ..connectome import Graph
from ..env import ACTION_NAMES, N_ACTIONS, BalatroEnv
from ..glomerular import (
    ENCODING_FEATUREMAP,
    ENCODING_GLOM32,
    pops_for_encoding,
    relay_bits,
)

__all__ = [
    "DecisionContext",
    "Decision",
    "Policy",
    "ReadoutModel",
    "BrainReadoutPolicy",
    "RawBitsPolicy",
    "HeuristicThroughBrainPolicy",
    "RandomLegalPolicy",
    "select_policy",
    "MODELS_DIR",
]

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "outputs" / "bc" / "models"
#: Searched in order; the newest run wins when it is present.
MODEL_DIRS: Tuple[Path, ...] = (
    ROOT / "outputs" / "bc3" / "models",
    ROOT / "outputs" / "bc2" / "models",
    ROOT / "outputs" / "bc" / "models",
)
NEG: float = -1.0e9

# Wiring condition -> the trained model files that are valid for it, best first.
# The MLP goes first: with the v2/v3 relay encoding the linear readout cannot
# express "select the first unselected slot of the best subset" (it is a
# conjunction), and the viewer is there to show the behaviour, not to defend
# linearity. Within a directory this order is overridden by measured imitation
# accuracy when ``train_metrics.json`` is available.
_BRAIN_MODEL_ORDER: Dict[str, Tuple[str, ...]] = {
    "real": ("real_alpn_kc_dn_mlp", "real_alpn_kc_dn_linear",
             "real_kc_mlp", "real_kc_linear",
             "real_alpn_mlp", "real_alpn_linear",
             "real_mbon_mlp", "real_mbon_linear",
             "real_dn_mlp", "real_dn_linear"),
    "shuffled": ("shuf_alpn_kc_dn_mlp", "shuf_alpn_kc_dn_linear",
                 "shuf_kc_mlp", "shuf_kc_linear",
                 "shuf_mbon_mlp", "shuf_mbon_linear",
                 "shuf_dn_mlp", "shuf_dn_linear"),
}
_RAW_MODEL_ORDER: Tuple[str, ...] = ("raw_bits_mlp", "raw_bits_linear")


@dataclass(frozen=True)
class DecisionContext:
    """Everything a policy may look at for one action."""

    env: BalatroEnv
    bits: NDArray[np.float32]
    mask: NDArray[np.int8]
    counts: NDArray[np.int32]


@dataclass(frozen=True)
class Decision:
    """A chosen action plus what to show about how it was chosen."""

    action: int
    scores: NDArray[np.float32]
    has_logits: bool

    def top(self, k: int, mask: NDArray[np.int8]) -> List[Dict[str, float]]:
        """The ``k`` highest-scoring *legal* actions, descending."""
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return []
        order = legal[np.argsort(-self.scores[legal], kind="stable")][:k]
        return [
            {
                "index": int(i),
                "name": ACTION_NAMES[int(i)],
                "logit": float(self.scores[int(i)]),
                "chosen": bool(int(i) == self.action),
            }
            for i in order
        ]


class Policy(Protocol):
    """Picks one of 109 actions and describes itself for the screen."""

    mode: str
    label: str
    input_desc: str
    brain_in_loop: bool
    #: State encoding this policy needs: 1 = 283 bits, 2 = the hand-type relay.
    feature_version: int
    #: Input path into the brain: "featuremap" (v1/v2) or "glomerular32" (v3).
    encoding: str
    readout_kind: str

    def decide(self, ctx: DecisionContext) -> Decision: ...


class ReadoutModel:
    """The numpy export of a ``bc_train.py`` readout: standardise, then the net.

    Both exported kinds load: ``linear`` (one affine layer) and ``mlp``
    (1x256 ReLU). ``kind_label`` is what the UI footer must say about it, so the
    screen never claims a linear probe when a small MLP is doing the work.
    """

    def __init__(self, path: Path) -> None:
        z = np.load(path, allow_pickle=False)
        kind = str(z["kind"])
        if kind not in ("linear", "mlp"):
            raise ValueError(f"{path.name} is a {kind!r} readout, expected linear or mlp")
        self.path: Path = path
        self.name: str = path.stem
        self.kind: str = kind
        self.feature_version: int = (
            int(z["feature_version"]) if "feature_version" in z.files else 1
        )
        self.encoding: str = (
            str(z["encoding"]) if "encoding" in z.files else ENCODING_FEATUREMAP
        )
        self.cond: str = str(z["cond"]) if "cond" in z.files else ""
        self.mean: NDArray[np.float32] = z["mean"].astype(np.float32)
        self.std: NDArray[np.float32] = z["std"].astype(np.float32)
        self.keep: NDArray[np.bool_] = z["keep"].astype(np.bool_)
        n_used = int(self.keep.sum())
        if kind == "linear":
            self.weight: NDArray[np.float32] = z["W"].astype(np.float32)
            self.bias: NDArray[np.float32] = z["b"].astype(np.float32)
            if self.weight.shape != (N_ACTIONS, n_used):
                raise ValueError(
                    f"{path.name}: W is {self.weight.shape}, expected "
                    f"{(N_ACTIONS, n_used)}"
                )
        else:
            self.weight1: NDArray[np.float32] = z["W1"].astype(np.float32)
            self.bias1: NDArray[np.float32] = z["b1"].astype(np.float32)
            self.weight2: NDArray[np.float32] = z["W2"].astype(np.float32)
            self.bias2: NDArray[np.float32] = z["b2"].astype(np.float32)
            if self.weight1.shape[1] != n_used or self.weight2.shape[0] != N_ACTIONS:
                raise ValueError(
                    f"{path.name}: W1 is {self.weight1.shape}, W2 is "
                    f"{self.weight2.shape}, expected (h, {n_used}) and "
                    f"({N_ACTIONS}, h)"
                )

    @property
    def n_inputs(self) -> int:
        return int(len(self.mean))

    @property
    def n_used_inputs(self) -> int:
        return int(self.keep.sum())

    @property
    def kind_label(self) -> str:
        if self.kind == "linear":
            return "linear readout"
        return f"1x{self.weight1.shape[0]} ReLU readout"

    def logits(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        if features.shape != (self.n_inputs,):
            raise ValueError(f"expected {self.n_inputs} features, got {features.shape}")
        x = ((features - self.mean) / self.std)[self.keep]
        if self.kind == "linear":
            return (self.weight @ x + self.bias).astype(np.float32)
        h = self.weight1 @ x + self.bias1
        np.maximum(h, 0.0, out=h)
        return (self.weight2 @ h + self.bias2).astype(np.float32)


def _mask_and_argmax(
    logits: NDArray[np.float32], mask: NDArray[np.int8]
) -> Tuple[int, NDArray[np.float32]]:
    scores = np.where(mask > 0, logits, np.float32(NEG)).astype(np.float32)
    return int(np.argmax(scores)), logits.astype(np.float32)


class BrainReadoutPolicy:
    """Linear or MLP readout over log1p spike counts of real fly populations."""

    def __init__(self, model: ReadoutModel, graph: Graph) -> None:
        blocks = {
            "alpn": graph.alpn_indices(),
            "kc": graph.kc_indices(),
            "mbon": graph.mbon_indices(),
            "dn": graph.dn_indices(),
        }
        pops = _pops_for_model(model, blocks)
        self._indices: NDArray[np.int32] = np.concatenate(
            [blocks[p] for p in pops]
        ).astype(np.int32)
        self._model: ReadoutModel = model
        self.mode: str = "brain_readout"
        self.label: str = f"brain readout ({model.name})"
        self.input_desc: str = (
            f"{model.n_inputs:,}-d log1p spike counts from "
            + " + ".join(p.upper() for p in pops)
        )
        self.brain_in_loop: bool = True
        self.feature_version: int = model.feature_version
        self.encoding: str = model.encoding
        self.pops: Tuple[str, ...] = pops
        self.readout_kind: str = model.kind_label

    def features(self, counts: NDArray[np.int32]) -> NDArray[np.float32]:
        """Reproduce bc_brain_features' float16 round-trip exactly."""
        raw = np.log1p(counts[self._indices].astype(np.float32)).astype(np.float16)
        return raw.astype(np.float32)

    def decide(self, ctx: DecisionContext) -> Decision:
        logits = self._model.logits(self.features(ctx.counts))
        action, scores = _mask_and_argmax(logits, ctx.mask)
        return Decision(action=action, scores=scores, has_logits=True)


#: Trained condition name -> the population blocks it reads, in order. This is
#: the same table ``scripts/bc_train.py`` uses; the exported ``cond`` field makes
#: it unambiguous, so nothing has to be guessed from the input width.
_COND_POPS: Dict[str, Tuple[str, ...]] = {
    "real_alpn_kc_dn": ("alpn", "kc", "dn"),
    "real_alpn": ("alpn",), "real_kc": ("kc",),
    "real_mbon": ("mbon",), "real_dn": ("dn",),
    "shuf_alpn_kc_dn": ("alpn", "kc", "dn"),
    "shuf_alpn": ("alpn",), "shuf_kc": ("kc",),
    "shuf_mbon": ("mbon",), "shuf_dn": ("dn",),
}


def _pops_for_model(
    model: ReadoutModel, blocks: Dict[str, NDArray[np.int32]]
) -> Tuple[str, ...]:
    """Which populations a model reads: its ``cond`` if exported, else its width."""
    pops = _COND_POPS.get(model.cond)
    if pops is not None:
        got = sum(len(blocks[p]) for p in pops)
        if got != model.n_inputs:
            raise ValueError(
                f"{model.name}: cond {model.cond!r} means {'+'.join(pops)} = "
                f"{got} inputs, but the export wants {model.n_inputs}"
            )
        return pops
    return _pops_for_width(model.n_inputs, blocks)


def _pops_for_width(
    width: int, blocks: Dict[str, NDArray[np.int32]]
) -> Tuple[str, ...]:
    """Which population blocks sum to ``width``; raises if none do."""
    candidates = [pops_for_encoding(ENCODING_FEATUREMAP),
                  pops_for_encoding(ENCODING_GLOM32),
                  ("dn",), ("alpn",), ("kc",), ("mbon",)]
    for pops in candidates:
        if sum(len(blocks[p]) for p in pops) == width:
            return tuple(pops)
    have = {k: len(v) for k, v in blocks.items()}
    raise ValueError(f"no population combination of {have} has width {width}")


class RawBitsPolicy:
    """Readout over the game bits themselves. The brain is shown, not consulted.

    Under ``glomerular32`` the trained model saw only the 32 relay bits, because
    that is all the fly saw, so this control is narrowed to the same 32.
    """

    def __init__(self, model: ReadoutModel) -> None:
        self._model: ReadoutModel = model
        self._relay_only: bool = model.encoding == ENCODING_GLOM32
        self.mode: str = "raw_bits"
        self.label: str = f"raw-bits readout ({model.name})"
        which = "relay bits" if self._relay_only else "game feature bits"
        self.input_desc: str = f"{model.n_inputs}-d {which}, brain not consulted"
        self.brain_in_loop: bool = False
        self.feature_version: int = model.feature_version
        self.encoding: str = model.encoding
        self.readout_kind: str = model.kind_label

    def decide(self, ctx: DecisionContext) -> Decision:
        bits = relay_bits(ctx.bits) if self._relay_only else ctx.bits
        logits = self._model.logits(bits.astype(np.float32))
        action, scores = _mask_and_argmax(logits, ctx.mask)
        return Decision(action=action, scores=scores, has_logits=True)


class HeuristicThroughBrainPolicy:
    """Hand-written greedy policy. The brain is driven and drawn, not consulted."""

    def __init__(self) -> None:
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        try:
            from baseline_heuristic_v2 import HandAwarePolicy  # noqa: PLC0415

            self._inner = HandAwarePolicy()
            which = "scripts/baseline_heuristic_v2.py (hand-aware) picks the action"
        except ImportError:
            from baseline_heuristic import HeuristicPolicy  # noqa: PLC0415

            self._inner = HeuristicPolicy()
            which = "scripts/baseline_heuristic.py picks the action"
        self.mode: str = "heuristic_through_brain"
        self.label: str = "hand-written heuristic (brain shown, not consulted)"
        self.input_desc: str = which
        self.brain_in_loop: bool = False
        self.feature_version: int = 1
        self.encoding: str = ENCODING_FEATUREMAP
        self.readout_kind: str = "no readout"

    def decide(self, ctx: DecisionContext) -> Decision:
        action = int(self._inner(ctx.env, ctx.mask))
        scores = np.where(ctx.mask > 0, np.float32(0.0), np.float32(NEG)).astype(np.float32)
        scores[action] = 1.0
        return Decision(action=action, scores=scores, has_logits=False)


class RandomLegalPolicy:
    """Uniform over the legal mask."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self.mode: str = "random_legal"
        self.label: str = "uniform random legal action"
        self.input_desc: str = "nothing; the brain is shown but no readout exists"
        self.brain_in_loop: bool = False
        self.feature_version: int = 1
        self.encoding: str = ENCODING_FEATUREMAP
        self.readout_kind: str = "no readout"

    def decide(self, ctx: DecisionContext) -> Decision:
        legal = np.flatnonzero(ctx.mask)
        if legal.size == 0:
            raise RuntimeError("no legal actions but the episode is not finished")
        action = int(self._rng.choice(legal))
        scores = np.where(ctx.mask > 0, np.float32(0.0), np.float32(NEG)).astype(np.float32)
        scores[action] = 1.0
        return Decision(action=action, scores=scores, has_logits=False)


def _rank_by_accuracy(directory: Path, names: Sequence[str]) -> List[str]:
    """``names`` re-ordered by held-out imitation top-1 from ``train_metrics.json``.

    Falls back to the given order for anything the metrics file does not mention
    (and for the whole list when there is no metrics file), so the static
    preference still decides ties and unmeasured models.
    """
    metrics_path = directory.parent / "train_metrics.json"
    if not metrics_path.exists():
        return list(names)
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, ValueError):
        return list(names)
    scored: List[Tuple[float, int, str]] = []
    for i, name in enumerate(names):
        top1 = -1.0
        for kind in ("linear", "mlp"):
            if name.endswith("_" + kind):
                row = metrics.get(f"{name[: -(len(kind) + 1)]}/{kind}")
                if isinstance(row, dict):
                    test = row.get("test")
                    if isinstance(test, dict) and test.get("top1") is not None:
                        top1 = float(test["top1"])
        scored.append((-top1, i, name))
    scored.sort()
    return [name for _, _, name in scored]


def _try_load(names: Sequence[str], notes: List[str],
              build: Optional[Callable[[ReadoutModel], "Policy"]] = None):
    """First *usable* model, newest run directory first, best-measured first.

    ``build`` turns a loaded model into a policy and may raise ``ValueError`` if
    the model does not fit this brain (a width that contradicts its ``cond``, for
    instance); that candidate is then noted and skipped rather than sinking the
    whole stage, so a broken export in the newest run still falls through to an
    older one. Returns the built policy when ``build`` is given, else the model.
    """
    for directory in MODEL_DIRS:
        for name in _rank_by_accuracy(directory, names):
            path = directory / f"{name}.npz"
            if not path.exists():
                continue
            try:
                model = ReadoutModel(path)
                built = build(model) if build is not None else None
            except (ValueError, OSError) as exc:
                notes.append(f"{path.parent.name}/{path.name}: {exc}")
                continue
            notes.append(
                f"using {path.parent.parent.name}/{path.parent.name}/{path.name} "
                f"(feature_version {model.feature_version}, "
                f"encoding {model.encoding}, {model.kind_label})"
            )
            return built if build is not None else model
    notes.append("none of " + ", ".join(names) + " found in "
                 + ", ".join(str(d.parent.name) for d in MODEL_DIRS))
    return None


def select_policy(
    graph: Graph, condition: str, prefer: str = "auto", seed: int = 0
) -> Tuple[Policy, List[str]]:
    """Pick the best available policy for ``condition``.

    A readout trained on the *real* wiring is never silently applied to the
    shuffled brain: the shuffled condition only accepts ``shuf_*`` models and
    otherwise drops to a brain-free fallback.

    Returns the policy and a list of human-readable notes about what was tried,
    which the server sends to the browser so the screen can explain itself.
    """
    notes: List[str] = []
    order = ("brain", "raw", "heuristic", "random") if prefer == "auto" else (prefer,)

    for stage in order:
        if stage == "brain":
            built = _try_load(_BRAIN_MODEL_ORDER.get(condition, ()), notes,
                              build=lambda m: BrainReadoutPolicy(m, graph))
            if built is not None:
                return built, notes
        elif stage == "raw":
            model = _try_load(_RAW_MODEL_ORDER, notes)
            if model is not None:
                return RawBitsPolicy(model), notes
        elif stage == "heuristic":
            try:
                return HeuristicThroughBrainPolicy(), notes
            except ImportError as exc:
                notes.append(f"heuristic: {exc}")
        elif stage == "random":
            return RandomLegalPolicy(seed), notes
        else:
            raise ValueError(f"unknown policy preference {stage!r}")

    notes.append("all preferred policies unavailable; using random legal")
    return RandomLegalPolicy(seed), notes
