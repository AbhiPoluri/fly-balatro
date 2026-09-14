"""Soma positions for the viewer's point cloud.

The MaleCNS body annotations carry a ``somaLocation`` column of xyz voxel
coordinates. Two things make it unusable raw:

1. 20,156 of the 146,271 retained neurons have no soma location at all. That
   includes *every* one of the 2,639 ORNs we drive: an olfactory receptor
   neuron's cell body sits in the antenna, outside the imaged CNS volume, so
   only its axon is in the dataset. Those neurons are dropped from the cloud
   (``point_index == -1``) rather than invented somewhere.
2. Ascending neurons have their somata in the ventral nerve cord, which
   extends the z axis out to 134,088 - roughly three times the depth of the
   brain itself. Left in, the brain is squashed into a sixth of the frame.
   The bounding box is therefore taken from the populations that are
   unambiguously brain (optic-lobe and central-brain intrinsic neurons and
   visual projection/centrifugal neurons) and somata outside it are dropped,
   not clamped - clamping piles them into a flat wall at the posterior edge
   that reads as a rendering bug. 2,185 somata fall outside: 1,812 ascending
   and 8 efferent-ascending (the ventral-nerve-cord ones), plus 365 genuine
   brain neurons trimmed by the 0.05/99.95 percentile edges of the box.

Axes: ``x`` is left-right (the two optic lobes are the dense blobs at either
end), ``y`` increases ventrally, ``z`` runs anterior-posterior. The world
coordinates written to the cache negate ``y`` so that dorsal is +Y, and centre
and scale the box into roughly [-1, 1] so the browser needs no camera fitting.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from ..connectome import Graph

__all__ = [
    "CATEGORY_NAMES",
    "SomaCloud",
    "BIN_MAGIC",
    "build_cloud",
    "load_or_build",
]

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ANNOTATIONS = DATA_DIR / "malecns_v1" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
BIN_PATH = DATA_DIR / "viewer_neurons.bin"
NPZ_PATH = DATA_DIR / "viewer_neurons.npz"

BIN_MAGIC: int = 0x464C5931  # "FLY1"
BIN_VERSION: int = 1
CACHE_VERSION: int = 1

# Colour/role buckets. Order is the wire encoding; the browser has the same list.
CATEGORY_NAMES: Tuple[str, ...] = (
    "sensory",      # 0  olfactory / visual receptor neurons (cool)
    "ALPN",         # 1  antennal-lobe projection neurons
    "KC",           # 2  Kenyon cells (mushroom body)
    "MBON",         # 3  mushroom-body output neurons
    "CX",           # 4  central complex
    "DN",           # 5  descending neurons (warm)
    "ascending",    # 6  ascending from the VNC
    "optic",        # 7  optic-lobe intrinsic / visual projection
    "central",      # 8  everything else in the central brain
)

# Superclasses that are certainly inside the brain proper, used to derive the
# bounding box that ventral-nerve-cord somata are rejected against.
_BRAIN_SUPERCLASSES: Tuple[str, ...] = (
    "ol_intrinsic",
    "cb_intrinsic",
    "visual_projection",
    "visual_centrifugal",
)

_BOX_LO_PCT: float = 0.05
_BOX_HI_PCT: float = 99.95


@dataclass(frozen=True)
class SomaCloud:
    """Point cloud plus the mapping needed to turn spike indices into points.

    Attributes:
        xyz: float32 (n_points, 3) world coordinates, dorsal is +Y, roughly
            inside the unit cube.
        code: uint8 (n_points,) index into :data:`CATEGORY_NAMES`.
        point_index: int32 (n_neurons,) point index per connectome neuron, or
            ``-1`` when that neuron is not drawn.
        stats: counts describing what was kept and dropped, for the footer.
    """

    xyz: NDArray[np.float32]
    code: NDArray[np.uint8]
    point_index: NDArray[np.int32]
    stats: Dict[str, int]

    @property
    def n_points(self) -> int:
        return int(len(self.code))

    @property
    def n_neurons(self) -> int:
        return int(len(self.point_index))

    def to_bytes(self) -> bytes:
        """Binary blob served as ``/static/neurons.bin``.

        Layout: ``<IIII`` header (magic, version, n_points, n_neurons), then
        ``float32[n_points * 3]`` xyz, then ``uint8[n_points]`` category code.
        """
        header = struct.pack("<IIII", BIN_MAGIC, BIN_VERSION, self.n_points, self.n_neurons)
        return header + self.xyz.tobytes() + self.code.tobytes()

    def category_counts(self) -> Dict[str, int]:
        counts = np.bincount(self.code, minlength=len(CATEGORY_NAMES))
        return {name: int(counts[i]) for i, name in enumerate(CATEGORY_NAMES)}


def _read_soma_locations(body_id: NDArray[np.int64]) -> NDArray[np.float32]:
    """float32 (n, 3) soma voxel coordinates in connectome order, NaN if unknown.

    Read through pyarrow rather than pandas so the viewer process never pulls
    pandas in; the column is a fixed/variable-size list of ints with nulls.
    """
    import pyarrow.feather as feather

    table = feather.read_table(ANNOTATIONS, columns=["bodyId", "somaLocation"], memory_map=True)
    ids = table.column("bodyId").to_numpy(zero_copy_only=False).astype(np.int64)
    locs = table.column("somaLocation").to_pylist()

    order = np.argsort(ids, kind="stable")
    ids_sorted = ids[order]
    pos = np.searchsorted(ids_sorted, body_id)
    pos_clipped = np.minimum(pos, len(ids_sorted) - 1)
    hit = ids_sorted[pos_clipped] == body_id

    xyz = np.full((len(body_id), 3), np.nan, dtype=np.float32)
    for i in np.flatnonzero(hit):
        row = locs[int(order[int(pos_clipped[i])])]
        if row is None or len(row) != 3 or any(v is None for v in row):
            continue
        xyz[i] = (float(row[0]), float(row[1]), float(row[2]))
    return xyz


def _categorise(graph: Graph) -> NDArray[np.uint8]:
    """Per-neuron index into :data:`CATEGORY_NAMES`.

    Class-level populations (ALPN, KC, MBON, CX) win over superclass, because
    they are the ones the pipeline meters and the wave animation are about.
    """
    cls = graph.cls.astype(str)
    superclass = graph.superclass.astype(str)
    code = np.full(graph.n, CATEGORY_NAMES.index("central"), dtype=np.uint8)

    is_optic = np.char.startswith(superclass, "ol_") | np.char.startswith(superclass, "visual")
    code[is_optic] = CATEGORY_NAMES.index("optic")
    code[np.char.find(superclass, "ascending") >= 0] = CATEGORY_NAMES.index("ascending")
    code[np.char.find(superclass, "sensory") >= 0] = CATEGORY_NAMES.index("sensory")
    code[np.char.find(superclass, "descending_neuron") >= 0] = CATEGORY_NAMES.index("DN")
    code[cls == "CX"] = CATEGORY_NAMES.index("CX")
    code[cls == "ALPN"] = CATEGORY_NAMES.index("ALPN")
    code[cls == "Kenyon_Cell"] = CATEGORY_NAMES.index("KC")
    code[cls == "MBON"] = CATEGORY_NAMES.index("MBON")
    return code


def build_cloud(graph: Graph) -> SomaCloud:
    """Read the annotations and build the centred, scaled, filtered cloud."""
    raw = _read_soma_locations(graph.body_id)
    has_soma = np.isfinite(raw).all(axis=1)

    superclass = graph.superclass.astype(str)
    in_brain_pop = np.isin(superclass, np.asarray(_BRAIN_SUPERCLASSES))
    reference = raw[has_soma & in_brain_pop]
    if len(reference) < 1000:
        raise RuntimeError(
            f"only {len(reference)} reference somata found; the annotation file "
            "does not look like MaleCNS v1.0"
        )
    lo = np.percentile(reference, _BOX_LO_PCT, axis=0).astype(np.float32)
    hi = np.percentile(reference, _BOX_HI_PCT, axis=0).astype(np.float32)

    inside = has_soma & np.all((raw >= lo) & (raw <= hi), axis=1)
    n_outside = int(has_soma.sum() - inside.sum())

    centre = (lo + hi) * 0.5
    scale = float(np.max((hi - lo) * 0.5))
    if scale <= 0.0:
        raise RuntimeError("degenerate soma bounding box")

    kept = np.flatnonzero(inside).astype(np.int32)
    local = (raw[kept] - centre) / scale
    xyz = np.empty((len(kept), 3), dtype=np.float32)
    xyz[:, 0] = local[:, 0]
    xyz[:, 1] = -local[:, 1]  # soma y increases ventrally; flip so dorsal is +Y
    xyz[:, 2] = local[:, 2]

    point_index = np.full(graph.n, -1, dtype=np.int32)
    point_index[kept] = np.arange(len(kept), dtype=np.int32)
    code = _categorise(graph)[kept].astype(np.uint8)

    orn = graph.orn_indices()
    stats = {
        "cache_version": CACHE_VERSION,
        "n_neurons": int(graph.n),
        "n_points": int(len(kept)),
        "n_no_soma": int((~has_soma).sum()),
        "n_outside_brain_box": n_outside,
        "n_orn": int(len(orn)),
        "n_orn_drawn": int((point_index[orn] >= 0).sum()),
    }
    return SomaCloud(xyz=xyz, code=code, point_index=point_index, stats=stats)


def load_or_build(graph: Graph, refresh: bool = False) -> SomaCloud:
    """Cached :func:`build_cloud`; also (re)writes ``data/viewer_neurons.bin``."""
    if NPZ_PATH.exists() and not refresh:
        z = np.load(NPZ_PATH, allow_pickle=False)
        stats = {str(k): int(v) for k, v in zip(z["stat_keys"], z["stat_values"])}
        if stats.get("cache_version") == CACHE_VERSION and stats.get("n_neurons") == graph.n:
            cloud = SomaCloud(
                xyz=z["xyz"], code=z["code"], point_index=z["point_index"], stats=stats
            )
            if not BIN_PATH.exists():
                BIN_PATH.write_bytes(cloud.to_bytes())
            return cloud

    cloud = build_cloud(graph)
    keys: List[str] = list(cloud.stats.keys())
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        NPZ_PATH,
        xyz=cloud.xyz,
        code=cloud.code,
        point_index=cloud.point_index,
        stat_keys=np.asarray(keys),
        stat_values=np.asarray([cloud.stats[k] for k in keys], dtype=np.int64),
    )
    BIN_PATH.write_bytes(cloud.to_bytes())
    return cloud
