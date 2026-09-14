"""MaleCNS v1.0 loader: three feather files in, one brain-only signed CSR graph out.

Adapted from vendor/fly-craftax/flycraftax/data.py and vendor/doomfly/doom/connectome.py.

This module stores topology and annotations only. It does not infer receptors,
dynamics, muscle mappings, or behavior from the wiring graph.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as feather

_CACHE_VERSION = 2  # bump when the loader's output changes

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "malecns_v1"
CACHE_DIR = ROOT / "data"

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
EDGES = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# Shiu et al. 2024: mV of synaptic conductance per synapse of a connection.
W_SYN = 0.275

SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": 1.0,
    "serotonin": 1.0,
    "octopamine": 1.0,
}

# Exact annotation strings for the populations we read from / write to.
ORN_SUPERCLASS = "cb_sensory"
ORN_CLASS = "olfactory"
ORN_TYPE_PREFIX = "ORN_"
DN_SUPERCLASS = "descending_neuron"
MBON_CLASS = "MBON"
CX_CLASS = "CX"
ALPN_CLASS = "ALPN"          # antennal-lobe projection neurons (first hop past the ORNs)
KC_CLASS = "Kenyon_Cell"     # mushroom-body Kenyon cells


@dataclass
class Graph:
    """Outgoing CSR over brain-only neurons, with per-neuron annotation arrays."""

    ptr: np.ndarray  # int64 (n+1,) CSR row pointer over presynaptic neurons
    post: np.ndarray  # int32 (m,) postsynaptic index
    weight: np.ndarray  # float32 (m,) sign(pre) * synapse_count * W_SYN, in mV
    body_id: np.ndarray  # int64 (n,) sorted ascending
    type: np.ndarray  # object (n,)
    cls: np.ndarray  # object (n,)  ("class" column)
    superclass: np.ndarray  # object (n,)
    side: np.ndarray  # object (n,)
    sign: np.ndarray  # float32 (n,)  +1 / -1 / 0 (unknown)
    meta: dict

    @property
    def n(self) -> int:
        return len(self.body_id)

    @property
    def m(self) -> int:
        return len(self.post)

    # ---- population selectors ------------------------------------------------
    def _sel(self, mask) -> np.ndarray:
        return np.flatnonzero(mask).astype(np.int32)

    def orn_indices(self) -> np.ndarray:
        return self._sel((self.superclass == ORN_SUPERCLASS) & (self.cls == ORN_CLASS))

    def dn_indices(self) -> np.ndarray:
        return self._sel(self.superclass == DN_SUPERCLASS)

    def mbon_indices(self) -> np.ndarray:
        return self._sel(self.cls == MBON_CLASS)

    def cx_indices(self) -> np.ndarray:
        return self._sel(self.cls == CX_CLASS)

    def alpn_indices(self) -> np.ndarray:
        return self._sel(self.cls == ALPN_CLASS)

    def kc_indices(self) -> np.ndarray:
        return self._sel(self.cls == KC_CLASS)

    def sensory_indices(self) -> np.ndarray:
        """Every retained neuron whose superclass mentions 'sensory' (ORN fallback pool)."""
        sc = self.superclass.astype(str)
        return self._sel(np.char.find(sc, "sensory") >= 0)

    def validate(self) -> None:
        n, m = self.n, self.m
        if self.ptr.shape != (n + 1,) or self.ptr.dtype != np.int64:
            raise ValueError("bad ptr")
        if self.ptr[0] != 0 or self.ptr[-1] != m or np.any(np.diff(self.ptr) < 0):
            raise ValueError("bad CSR ptr")
        if self.post.dtype != np.int32 or self.weight.dtype != np.float32:
            raise ValueError("bad edge dtypes")
        if m and (self.post.min() < 0 or self.post.max() >= n):
            raise ValueError("post index out of range")
        if not np.isfinite(self.weight).all():
            raise ValueError("nonfinite weight")


def _neurons(data_dir: Path) -> tuple[dict, dict]:
    import pandas as pd

    ann = pd.read_feather(data_dir / ANNOTATIONS)
    n_raw = len(ann)
    sc = ann.superclass.astype(str)
    keep = ann.superclass.notna() & sc.ne("") & ~sc.str.startswith("vnc")
    keep &= ann.status.ne("Glia")  # glia are not neurons
    n_vnc = int((ann.superclass.notna() & sc.str.startswith("vnc")).sum())
    n_glia = int(ann.status.eq("Glia").sum())
    n_nosc = int((ann.superclass.isna() | sc.eq("")).sum())
    ann = ann[keep].copy()

    nt = pd.read_feather(
        data_dir / NEUROTRANSMITTERS,
        columns=["body", "ground_truth", "consensus_nt", "celltype_predicted_nt"],
    )
    if not nt.body.is_unique:
        raise ValueError("Duplicate neurotransmitter body ids.")
    nti = nt.set_index("body")
    s_gt = ann.bodyId.map(nti.ground_truth).map(SIGN)
    s_cons = ann.bodyId.map(nti.consensus_nt).map(SIGN)
    s_type = ann.bodyId.map(nti.celltype_predicted_nt).map(SIGN)
    tiers = {
        "ground_truth": int(s_gt.notna().sum()),
        "consensus_nt": int((s_gt.isna() & s_cons.notna()).sum()),
        "celltype_predicted_nt": int((s_gt.isna() & s_cons.isna() & s_type.notna()).sum()),
    }
    sign = s_gt.fillna(s_cons).fillna(s_type)
    n_unknown = int(sign.isna().sum())
    ann["sign"] = sign.fillna(0.0).astype(np.float32)
    ann["side_"] = ann.somaSide.fillna(ann.rootSide).fillna("unknown")
    ann = ann.sort_values("bodyId").reset_index(drop=True)

    cols = dict(
        body_id=ann.bodyId.values.astype(np.int64),
        type=ann.type.fillna("").values.astype(object),
        cls=ann["class"].fillna("").values.astype(object),
        superclass=ann.superclass.fillna("").values.astype(object),
        side=ann.side_.astype(str).values.astype(object),
        sign=ann["sign"].values.astype(np.float32),
    )
    stats = dict(
        annotation_rows=n_raw,
        dropped_no_superclass=n_nosc,
        dropped_vnc_superclass=n_vnc,
        dropped_glia=n_glia,
        neurons=len(ann),
        sign_tiers=tiers,
        neurons_unknown_sign=n_unknown,
    )
    del ann, nt, nti
    return cols, stats


def _edges(data_dir: Path, body_id: np.ndarray, threshold: int):
    """Filter in Arrow, index with searchsorted; never materialise 44M rows in pandas."""
    tbl = feather.read_table(
        data_dir / EDGES, columns=["body_pre", "body_post", "weight"], memory_map=True
    )
    n_rows = tbl.num_rows
    tbl = tbl.filter(pc.greater_equal(tbl.column("weight"), threshold))
    n_kept_thresh = tbl.num_rows
    pre_id = tbl.column("body_pre").to_numpy()
    post_id = tbl.column("body_post").to_numpy()
    count = tbl.column("weight").to_numpy().astype(np.int32)
    del tbl

    last = len(body_id) - 1
    i = np.searchsorted(body_id, pre_id)
    j = np.searchsorted(body_id, post_id)
    ok = body_id[np.minimum(i, last)] == pre_id
    ok &= body_id[np.minimum(j, last)] == post_id
    del pre_id, post_id
    pre = i[ok].astype(np.int32)
    post = j[ok].astype(np.int32)
    count = count[ok]
    del i, j, ok
    return pre, post, count, n_rows, n_kept_thresh


def build(threshold: int = 5, data_dir: Path = DATA_DIR, verbose: bool = True) -> Graph:
    t0 = time.perf_counter()
    cols, stats = _neurons(data_dir)
    body_id = cols["body_id"]
    sign = cols["sign"]
    pre, post, count, n_rows, n_kept_thresh = _edges(data_dir, body_id, threshold)
    n_in_brain = len(pre)

    keep = sign[pre] != 0.0  # craftax: an edge from an unsigned neuron has no sign
    pre, post, count = pre[keep], post[keep], count[keep]
    del keep
    n_signed = len(pre)

    order = np.argsort(pre, kind="stable")
    pre, post, count = pre[order], post[order], count[order]
    del order
    n = len(body_id)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=ptr[1:])
    weight = (sign[pre] * count.astype(np.float32) * W_SYN).astype(np.float32)

    meta = dict(
        threshold=threshold,
        w_syn=W_SYN,
        cache_version=_CACHE_VERSION,
        edge_rows_total=int(n_rows),
        edge_rows_ge_threshold=int(n_kept_thresh),
        edges_dropped_by_threshold=int(n_rows - n_kept_thresh),
        edges_both_endpoints_in_brain=int(n_in_brain),
        edges_dropped_endpoint_outside_brain=int(n_kept_thresh - n_in_brain),
        edges_dropped_unknown_pre_sign=int(n_in_brain - n_signed),
        edges=int(n_signed),
        synapses_retained=int(count.sum()),
        **stats,
    )
    del count
    g = Graph(
        ptr=ptr,
        post=post,
        weight=weight,
        body_id=body_id,
        type=cols["type"],
        cls=cols["cls"],
        superclass=cols["superclass"],
        side=cols["side"],
        sign=sign,
        meta=meta,
    )
    g.validate()
    meta["build_seconds"] = round(time.perf_counter() - t0, 2)
    if verbose:
        report(g)
    return g


def cache_path(threshold: int = 5) -> Path:
    return CACHE_DIR / f"connectome_v{_CACHE_VERSION}_t{threshold}.npz"


def save(g: Graph, path: Path | None = None) -> Path:
    import json

    path = path or cache_path(g.meta["threshold"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ptr=g.ptr,
        post=g.post,
        weight=g.weight,
        body_id=g.body_id,
        type=g.type.astype(str),
        cls=g.cls.astype(str),
        superclass=g.superclass.astype(str),
        side=g.side.astype(str),
        sign=g.sign,
        meta=np.asarray(json.dumps(g.meta)),
    )
    return path


def load(threshold: int = 5, data_dir: Path = DATA_DIR, verbose: bool = False) -> Graph:
    """Cached build. Returns the brain-only signed CSR graph."""
    import json

    path = cache_path(threshold)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        g = Graph(
            ptr=z["ptr"],
            post=z["post"],
            weight=z["weight"],
            body_id=z["body_id"],
            type=z["type"].astype(object),
            cls=z["cls"].astype(object),
            superclass=z["superclass"].astype(object),
            side=z["side"].astype(object),
            sign=z["sign"],
            meta=json.loads(str(z["meta"])),
        )
        g.validate()
        if verbose:
            report(g)
        return g
    g = build(threshold, data_dir, verbose=verbose)
    save(g, path)
    return g


def report(g: Graph) -> None:
    m = g.meta
    print("=" * 70)
    print(f"MaleCNS v1.0 brain-only signed graph (edge threshold = {m['threshold']} synapses)")
    print("=" * 70)
    print(f"annotation rows                 {m['annotation_rows']:>12,}")
    print(f"  dropped: no superclass        {m['dropped_no_superclass']:>12,}")
    print(f"  dropped: vnc* superclass      {m['dropped_vnc_superclass']:>12,}")
    print(f"  dropped: status == Glia       {m['dropped_glia']:>12,}")
    print(f"NEURONS retained                {m['neurons']:>12,}")
    print(f"  sign from ground_truth        {m['sign_tiers']['ground_truth']:>12,}")
    print(f"  sign from consensus_nt        {m['sign_tiers']['consensus_nt']:>12,}")
    print(f"  sign from celltype_pred_nt    {m['sign_tiers']['celltype_predicted_nt']:>12,}")
    print(f"  UNKNOWN sign                  {m['neurons_unknown_sign']:>12,}")
    print(f"edge rows in file               {m['edge_rows_total']:>12,}")
    print(f"  dropped by threshold          {m['edges_dropped_by_threshold']:>12,}")
    print(f"  dropped endpoint not in brain {m['edges_dropped_endpoint_outside_brain']:>12,}")
    print(f"  dropped unknown pre sign      {m['edges_dropped_unknown_pre_sign']:>12,}")
    print(f"EDGES retained                  {m['edges']:>12,}")
    print(f"  synapses retained             {m['synapses_retained']:>12,}")
    exc = int((g.weight > 0).sum())
    print(f"  excitatory / inhibitory       {exc:>12,} / {m['edges'] - exc:,}")
    print("-" * 70)
    for name, idx in [
        (f"ORN  (superclass={ORN_SUPERCLASS!r}, class={ORN_CLASS!r})", g.orn_indices()),
        (f"DN   (superclass={DN_SUPERCLASS!r})", g.dn_indices()),
        (f"MBON (class={MBON_CLASS!r})", g.mbon_indices()),
        (f"CX   (class={CX_CLASS!r})", g.cx_indices()),
        (f"ALPN (class={ALPN_CLASS!r})", g.alpn_indices()),
        (f"KC   (class={KC_CLASS!r})", g.kc_indices()),
        ("sensory (any 'sensory' superclass)", g.sensory_indices()),
    ]:
        print(f"{name:<52} {len(idx):>8,}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    p = cache_path(a.threshold)
    if a.rebuild and p.exists():
        p.unlink()
    g = load(a.threshold, verbose=True)
    print("cache:", cache_path(a.threshold))
    print("build seconds:", g.meta.get("build_seconds"))
