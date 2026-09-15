"""The v3 input path: 32 relay bits -> 32 whole ORN glomeruli, on a tuned brain.

One module every stage imports, so nothing in the pipeline can build a slightly
different version of this encoding. It is the glomerular twin of
``encode.feature_map_for``: collection is unchanged (``outputs/bc2/states.npz``
still holds 315-bit v2 rows), but only the leading 32 bits (the hand-type relay
block of :mod:`flybalatro.features_v2`) ever reach the fly, and each of them
drives *every* olfactory receptor neuron of one glomerulus instead of 10
scattered ones.

Why: ``outputs/mb/REPORT.md`` traced the v1/v2 information loss to the input, not
to the brain. With ~36 active bits x 10 random ORNs, all 54 glomeruli fired on
every state and the antennal-lobe total varied by a CV of 0.03; the Kenyon cells
had nothing to sparsify. ``outputs/mb2/REPORT.md`` ran the fix on 2,000 real
states: 6.8 of 32 glomeruli active per state, and a linear probe reads the 9-way
hand type at 1.000 from the Kenyon cells and 0.842 from the descending neurons,
against 0.598 / 0.610 before.

The spec (the glomerulus per bit, the tuning, the drive) is read from
``outputs/mb2/tuned_config.json``, the file the gated sweep chose, and checked
against :data:`GLOM32_TYPES` so a stage still works (and still agrees) if that
file is ever moved. Nothing here is measured: ``apl_scale = 2`` is a calibration
choice, documented in :mod:`flybalatro.tuning`.

Populations stored for v3 are ALPN, KC, **MBON**, DN, in that order. MBON is new
in v3: with a separable Kenyon-cell code there is a mushroom-body output
worth reading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .encode import DEFAULT_DRIVE_MV, GlomerularMap
from .features_v2 import N_HAND_BLOCK
from .tuning import Tuning

__all__ = [
    "ENCODING_FEATUREMAP",
    "ENCODING_GLOM32",
    "ENCODINGS",
    "GLOM32_TYPES",
    "TUNED_CONFIG_PATH",
    "GlomSpec",
    "load_spec",
    "map_for",
    "brain_for",
    "relay_bits",
    "pops_for_encoding",
    "population_indices",
    "population_slices",
    "n_input_bits",
]

ROOT = Path(__file__).resolve().parents[1]

#: ``--encoding`` values. ``featuremap`` is v1/v2 (``encode.feature_map_for``).
ENCODING_FEATUREMAP: str = "featuremap"
ENCODING_GLOM32: str = "glomerular32"
ENCODINGS: Tuple[str, ...] = (ENCODING_FEATUREMAP, ENCODING_GLOM32)

#: The file ``scripts/mb2_gate.py`` wrote when it picked ``apl2_tonic_vth0``.
TUNED_CONFIG_PATH: Path = ROOT / "outputs" / "mb2" / "tuned_config.json"

#: Glomerulus per relay bit, bit 0 first. Duplicated from ``tuned_config.json``
#: on purpose: :func:`load_spec` asserts the two agree, so a stage that reads the
#: file and a stage that cannot are guaranteed to drive the same receptors.
#: Bits 0-8 (the 9-way hand-type label) hold the nine smallest, tightest types so
#: the label cannot be read off the total drive; see ``scripts/mb2_assign.py``.
GLOM32_TYPES: Tuple[str, ...] = (
    "ORN_VM7d", "ORN_DA4m", "ORN_DC3", "ORN_DM5", "ORN_DA3", "ORN_VA4",
    "ORN_VC3", "ORN_VM5v", "ORN_VM6v",                      # best_type (9)
    "ORN_V", "ORN_DM2", "ORN_DA2", "ORN_VL2p", "ORN_DL5", "ORN_VM3",
    "ORN_VM2", "ORN_VC4",                                   # best_mask (8)
    "ORN_VL2a", "ORN_VM5d", "ORN_DL1", "ORN_VA2", "ORN_VL1", "ORN_VM4",
    "ORN_DM1", "ORN_DM3", "ORN_VA6", "ORN_DL4",             # selected_type (10)
    "ORN_DM6",                                              # selected_is_best
    "ORN_DA1", "ORN_VA1d", "ORN_VA1v", "ORN_DL3",           # score_vs_needed (4)
)

#: Population blocks stored per encoding, in concatenation order.
_POPS: Dict[str, Tuple[str, ...]] = {
    ENCODING_FEATUREMAP: ("alpn", "kc", "dn"),
    ENCODING_GLOM32: ("alpn", "kc", "mbon", "dn"),
}


@dataclass(frozen=True)
class GlomSpec:
    """Everything the v3 input path needs, as chosen by the mb2 gate."""

    type_names: Tuple[str, ...]
    tuning: Tuning
    drive_mv: float
    window_ms: float
    brain_seed: int
    v_jitter_mv: float
    label: str
    source: str

    @property
    def n_features(self) -> int:
        return len(self.type_names)

    def describe(self) -> dict:
        return dict(
            encoding=ENCODING_GLOM32, label=self.label, source=self.source,
            n_features=self.n_features, type_names=list(self.type_names),
            tuning=self.tuning.to_dict(), drive_mv=self.drive_mv,
            window_ms=self.window_ms, brain_seed=self.brain_seed,
            v_jitter_mv=self.v_jitter_mv,
        )


_SPEC_CACHE: Dict[str, GlomSpec] = {}


def load_spec(path: Optional[Path] = None) -> GlomSpec:
    """Read ``tuned_config.json``; fall back to the embedded constants.

    Cached, because every worker process asks for it once and the file never
    changes inside a run. Raises if the file disagrees with
    :data:`GLOM32_TYPES`, if the drive is not tonic, or if the block is not 32
    bits wide. All three would mean a stage is about to drive the wrong
    receptors with the wrong calibration.
    """
    p = Path(path) if path is not None else TUNED_CONFIG_PATH
    key = str(p)
    hit = _SPEC_CACHE.get(key)
    if hit is not None:
        return hit

    if not p.exists():
        spec = GlomSpec(
            type_names=GLOM32_TYPES, tuning=Tuning(apl_scale=2.0),
            drive_mv=DEFAULT_DRIVE_MV, window_ms=50.0, brain_seed=1,
            v_jitter_mv=6.9, label="apl2_tonic_vth0",
            source="flybalatro.glomerular.GLOM32_TYPES (tuned_config.json absent)",
        )
        _SPEC_CACHE[key] = spec
        return spec

    cfg = json.loads(p.read_text())
    enc = cfg.get("encoding", {})
    if str(enc.get("kind")) != "glomerular":
        raise ValueError(f"{p}: encoding kind is {enc.get('kind')!r}, expected 'glomerular'")
    drive = cfg.get("drive", {})
    if str(drive.get("mode")) != "tonic":
        raise ValueError(f"{p}: drive mode is {drive.get('mode')!r}; only tonic is wired up")
    names = tuple(str(a["glomerulus"]) for a in enc["assignment"])
    if len(names) != N_HAND_BLOCK:
        raise ValueError(f"{p}: {len(names)} glomeruli, expected {N_HAND_BLOCK}")
    if names != GLOM32_TYPES:
        raise ValueError(
            f"{p} disagrees with flybalatro.glomerular.GLOM32_TYPES; "
            "one of the two is stale, so refuse to guess"
        )
    spec = GlomSpec(
        type_names=names,
        tuning=Tuning.from_dict(cfg["tuning"]),
        drive_mv=float(drive.get("tonic_mv", DEFAULT_DRIVE_MV)),
        window_ms=float(cfg.get("window_ms", 50.0)),
        brain_seed=int(cfg.get("brain_seed", 1)),
        v_jitter_mv=float(cfg.get("v_jitter_mv", 6.9)),
        label=str(cfg.get("label", "")),
        source=str(p),
    )
    _SPEC_CACHE[key] = spec
    return spec


def map_for(graph, spec: Optional[GlomSpec] = None) -> GlomerularMap:
    """The one :class:`~flybalatro.encode.GlomerularMap` every stage must use."""
    s = spec if spec is not None else load_spec()
    return GlomerularMap(graph, s.type_names, drive_mv=s.drive_mv)


def brain_for(graph, wiring: str, spec: Optional[GlomSpec] = None,
              shuffle_seed: int = 0, brain_seed: Optional[int] = None):
    """A tuned ``Brain`` for ``wiring`` in ``{"real", "shuffled"}``.

    The tuning is handed to the ``Brain`` constructor, and ``Brain.shuffled``
    passes it down to the shuffled copy, where ``tuning.scale_out_edges`` walks
    ``graph.ptr``, i.e. it scales APL's outgoing synapses **by presynaptic
    identity**, which is exactly what a degree-preserving target permutation
    leaves intact.
    """
    from .brain import Brain

    s = spec if spec is not None else load_spec()
    seed = int(s.brain_seed if brain_seed is None else brain_seed)
    base = Brain(graph, seed=seed, v_jitter=s.v_jitter_mv, tuning=s.tuning)
    if wiring == "real":
        return base
    if wiring == "shuffled":
        out = base.shuffled(int(shuffle_seed))
        del base
        return out
    raise ValueError(f"wiring must be 'real' or 'shuffled', got {wiring!r}")


def relay_bits(bits: NDArray) -> NDArray:
    """The leading 32 relay bits of a v2 feature vector (rows or a single row).

    The other 283 bits are **not sent to the brain** under this encoding.
    """
    a = np.asarray(bits)
    if a.shape[-1] < N_HAND_BLOCK:
        raise ValueError(f"need at least {N_HAND_BLOCK} columns, got {a.shape[-1]}")
    return a[..., :N_HAND_BLOCK]


def n_input_bits(encoding: str, n_features: int) -> int:
    """Width of the vector the readout / the brain actually sees."""
    if encoding == ENCODING_GLOM32:
        return int(N_HAND_BLOCK)
    return int(n_features)


def pops_for_encoding(encoding: str) -> Tuple[str, ...]:
    """Population blocks stored by ``bc_brain_features``, in order."""
    try:
        return _POPS[encoding]
    except KeyError:
        raise ValueError(f"unknown encoding {encoding!r}; expected one of {ENCODINGS}") from None


def population_indices(graph, encoding: str) -> Tuple[NDArray[np.int32], Dict[str, slice]]:
    """Neuron indices to record, plus the slice of the result per population."""
    getters = {"alpn": graph.alpn_indices, "kc": graph.kc_indices,
               "mbon": graph.mbon_indices, "dn": graph.dn_indices}
    pops = pops_for_encoding(encoding)
    blocks = [getters[p]() for p in pops]
    slices: Dict[str, slice] = {}
    off = 0
    for p, b in zip(pops, blocks):
        slices[p] = slice(off, off + len(b))
        off += len(b)
    return np.concatenate(blocks).astype(np.int32), slices


def population_slices(sizes: Sequence[int], encoding: str) -> Dict[str, slice]:
    """Same slices from stored sizes, for code that has the widths but no graph."""
    pops = pops_for_encoding(encoding)
    if len(sizes) != len(pops):
        raise ValueError(f"{len(sizes)} sizes for {len(pops)} populations {pops}")
    out: Dict[str, slice] = {}
    off = 0
    for p, w in zip(pops, sizes):
        out[p] = slice(off, off + int(w))
        off += int(w)
    return out
