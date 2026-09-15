"""Degree- and weight-preserving rewiring of the ALPN -> Kenyon-cell stage only.

Why this exists
---------------

``RESULTS.md`` v3 compares the real MaleCNS wiring against
``Brain.shuffled(seed)``, a *global* permutation of every edge's postsynaptic
target. Under the glomerular encoding that control is broken: glomerular
convergence (~440 driven receptor neurons of one type onto the uniglomerular
projection neurons of that glomerulus) is exactly what a global target
permutation destroys, so the shuffled network barely fires (ALPN 71.3 Hz real
against 0.63 Hz shuffled) and the comparison is between a working brain and a
near-dead one.

This module builds the control that asks an answerable question instead: keep
the sensory front end, the APL loop, KC -> KC, KC -> MBON and every dopaminergic
edge **exactly as measured**, and randomise only *which* projection neurons
converge on each Kenyon cell. That is the mushroom-body calyx, the one stage
where the literature makes a prediction: PN -> KC connectivity is reported as
largely random in vivo (Caron et al. 2013), with later work finding partial
structure (Zheng et al. 2022), so a near-null validates the model against known
biology and a large effect is a finding.

The rewiring
------------

Double-edge swaps of postsynaptic endpoints, *within strata*. A stratum is

    (KC hemisphere, KC ``type`` string, synapse count, presynaptic sign)

i.e. exactly one edge weight value and one kind of target cell. Swapping the
posts of two edges in the same stratum therefore:

* leaves every presynaptic neuron's out-degree and out-weight untouched (the
  edge stays in its own CSR row with its own weight);
* leaves every postsynaptic neuron's in-degree and in-weight untouched (the two
  endpoints exchange edges of identical weight);
* leaves each ALPN's distribution over KC *subtype* and *hemisphere* untouched;
* changes only which specific Kenyon cell each projection neuron contacts.

Swaps that would create a duplicate (pre, post) pair or a self-loop are
rejected. Edges alone in their stratum can never move and are reported.

The result is a real :class:`flybalatro.connectome.Graph`, not a monkey-patched
index array, so ``Tuning.apply`` (which walks ``graph.ptr`` / ``graph.post``),
the per-KC homeostasis and every downstream stage operate on the true endpoints.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent
for _p in (str(ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flybalatro import connectome as C  # noqa: E402

OUT_DIR = ROOT / "outputs" / "calyx"

#: Attempted swaps per edge in a stratum. 20 is far past the point where the
#: measured rewiring depth stops moving (see ``rewiring_depth``).
ATTEMPTS_PER_EDGE: int = 20

ALPN_CLASS = C.ALPN_CLASS
KC_CLASS = C.KC_CLASS


# --------------------------------------------------------------------------- #
# the edge set
# --------------------------------------------------------------------------- #
def edge_pre(graph) -> NDArray[np.int32]:
    """Presynaptic neuron index per edge, expanded from the CSR row pointer."""
    counts = np.diff(graph.ptr)
    return np.repeat(np.arange(graph.n, dtype=np.int32), counts)


def alpn_kc_edges(graph) -> Dict[str, NDArray]:
    """Edge indices of the ALPN -> Kenyon-cell projection, plus their endpoints."""
    cls = graph.cls.astype(str)
    alpn = np.flatnonzero(cls == ALPN_CLASS).astype(np.int32)
    kc = np.flatnonzero(cls == KC_CLASS).astype(np.int32)
    kc_mask = np.zeros(graph.n, bool)
    kc_mask[kc] = True
    blocks: List[NDArray] = []
    for i in alpn.astype(np.int64):
        a, b = int(graph.ptr[i]), int(graph.ptr[i + 1])
        if b <= a:
            continue
        m = kc_mask[graph.post[a:b]]
        if m.any():
            blocks.append(np.arange(a, b, dtype=np.int64)[m])
    e = (np.concatenate(blocks) if blocks else np.zeros(0, np.int64))
    pre = edge_pre(graph)[e].astype(np.int32)
    return dict(edges=e, pre=pre, post=graph.post[e].astype(np.int32),
                weight=graph.weight[e].astype(np.float32),
                alpn=alpn, kc=kc)


def stratum_keys(graph, ek: Dict[str, NDArray]) -> Tuple[NDArray[np.int64], List[tuple]]:
    """``(key_id per edge, key list)``; key = (KC side, KC type, synapses, sign).

    The synapse count is recovered from the weight (``sign * count * W_SYN``),
    and the sign is carried separately so the 24 GABAergic ALPN -> KC edges can
    only ever swap with each other; an inhibitory edge landing where an
    excitatory one was would change that KC's weighted input.
    """
    side = graph.side.astype(str)
    typ = graph.type.astype(str)
    post = ek["post"]
    w = ek["weight"]
    count = np.rint(np.abs(w) / C.W_SYN).astype(np.int64)
    sign = np.sign(w).astype(np.int64)
    keys = [(str(side[p]), str(typ[p]), int(c), int(s))
            for p, c, s in zip(post, count, sign)]
    uniq: Dict[tuple, int] = {}
    ids = np.empty(len(keys), np.int64)
    order: List[tuple] = []
    for i, k in enumerate(keys):
        j = uniq.get(k)
        if j is None:
            j = len(order)
            uniq[k] = j
            order.append(k)
        ids[i] = j
    return ids, order


# --------------------------------------------------------------------------- #
# the swap
# --------------------------------------------------------------------------- #
@dataclass
class RewireResult:
    graph: object
    stats: dict


def rewire_alpn_kc(graph, seed: int,
                   attempts_per_edge: int = ATTEMPTS_PER_EDGE) -> RewireResult:
    """A new ``Graph`` with the ALPN -> KC postsynaptic endpoints swapped in strata.

    Everything except ``post`` at those edge positions is shared with / copied
    from ``graph`` unchanged.
    """
    t0 = time.perf_counter()
    ek = alpn_kc_edges(graph)
    e = ek["edges"]
    pre = ek["pre"]
    post = ek["post"].copy()
    ids, key_order = stratum_keys(graph, ek)

    n = int(graph.n)
    pairs = set((int(a) * n + int(b)) for a, b in zip(pre, ek["post"]))
    if len(pairs) != len(e):
        raise ValueError("ALPN -> KC edge set already contains duplicate (pre, post) "
                         "pairs; the no-duplicate invariant is undefined")

    rng = np.random.default_rng(seed)
    order = np.argsort(ids, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(ids[order]) != 0, True])
    attempted = accepted = 0
    rej_same_pre = rej_same_post = rej_dup = rej_self = 0
    singleton_edges = 0
    n_strata_movable = 0

    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        rows = order[b0:b1]
        k = len(rows)
        if k < 2:
            singleton_edges += k
            continue
        n_strata_movable += 1
        tries = int(attempts_per_edge * k)
        aa = rng.integers(0, k, size=tries)
        bb = rng.integers(0, k, size=tries)
        for x, y in zip(aa, bb):
            if x == y:
                continue
            i = int(rows[x])
            j = int(rows[y])
            attempted += 1
            p1, p2 = int(pre[i]), int(pre[j])
            q1, q2 = int(post[i]), int(post[j])
            if p1 == p2:
                rej_same_pre += 1
                continue
            if q1 == q2:
                rej_same_post += 1
                continue
            if p1 == q2 or p2 == q1:
                rej_self += 1
                continue
            k1 = p1 * n + q2
            k2 = p2 * n + q1
            if k1 in pairs or k2 in pairs:
                rej_dup += 1
                continue
            pairs.discard(p1 * n + q1)
            pairs.discard(p2 * n + q2)
            pairs.add(k1)
            pairs.add(k2)
            post[i], post[j] = q2, q1
            accepted += 1

    new_post = np.array(graph.post, dtype=np.int32, copy=True)
    new_post[e] = post
    meta = json.loads(json.dumps(graph.meta))
    depth = rewiring_depth(graph, ek, post)
    meta["calyx_rewiring"] = dict(
        stage="ALPN -> Kenyon_Cell",
        seed=int(seed),
        strata_definition="(KC side, KC type string, synapse count, presynaptic sign)",
        n_edges=int(len(e)),
        n_strata=int(len(key_order)),
        n_strata_with_2plus_edges=int(n_strata_movable),
        n_singleton_edges=int(singleton_edges),
        attempts_per_edge=int(attempts_per_edge),
        swaps_attempted=int(attempted),
        swaps_accepted=int(accepted),
        rejected_same_presynaptic=int(rej_same_pre),
        rejected_same_postsynaptic=int(rej_same_post),
        rejected_duplicate_edge=int(rej_dup),
        rejected_self_loop=int(rej_self),
        seconds=round(time.perf_counter() - t0, 2),
        **depth,
    )
    g2 = C.Graph(
        ptr=np.array(graph.ptr, copy=True),
        post=new_post,
        weight=np.array(graph.weight, copy=True),
        body_id=np.array(graph.body_id, copy=True),
        type=graph.type.copy(),
        cls=graph.cls.copy(),
        superclass=graph.superclass.copy(),
        side=graph.side.copy(),
        sign=np.array(graph.sign, copy=True),
        meta=meta,
    )
    g2.validate()
    return RewireResult(graph=g2, stats=meta["calyx_rewiring"])


def rewiring_depth(graph, ek: Dict[str, NDArray], new_post: NDArray) -> dict:
    """How much of the calyx moved. A near-null needs this number."""
    old = ek["post"].astype(np.int64)
    new = np.asarray(new_post, np.int64)
    pre = ek["pre"].astype(np.int64)
    n = int(graph.n)
    old_pairs = set((pre * n + old).tolist())
    novel = int(sum(1 for k in (pre * n + new).tolist() if k not in old_pairs))
    # Per-KC: what fraction of a Kenyon cell's ALPN inputs come from a different
    # projection neuron than before.
    per_kc: List[float] = []
    for kc in np.unique(old):
        a = set(pre[old == kc].tolist())
        b = set(pre[new == kc].tolist())
        if a:
            per_kc.append(1.0 - len(a & b) / len(a))
    per_kc_arr = np.asarray(per_kc) if per_kc else np.zeros(1)
    return dict(
        frac_edges_with_new_target=round(float((old != new).mean()), 4),
        frac_edges_novel_pair=round(novel / max(1, len(old)), 4),
        per_kc_frac_inputs_relocated_mean=round(float(per_kc_arr.mean()), 4),
        per_kc_frac_inputs_relocated_sd=round(float(per_kc_arr.std()), 4),
    )


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def rewired_path(seed: int, out_dir: Path = OUT_DIR) -> Path:
    return out_dir / f"graph_rewired_s{seed}.npz"


def build_or_load(seed: int, out_dir: Path = OUT_DIR, threshold: int = 5,
                  attempts_per_edge: int = ATTEMPTS_PER_EDGE, verbose: bool = True):
    """The rewired graph for ``seed``, cached on disk (resumable)."""
    path = rewired_path(seed, out_dir)
    if path.exists():
        g = _load_npz(path)
        if verbose:
            print(f"[calyx] loaded {path.name}", flush=True)
        return g
    base = C.load(threshold)
    res = rewire_alpn_kc(base, seed, attempts_per_edge)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.save(res.graph, path)
    if verbose:
        print(f"[calyx] wrote {path.name}: {json.dumps(res.stats)}", flush=True)
    return res.graph


def _load_npz(path: Path):
    z = np.load(path, allow_pickle=False)
    g = C.Graph(
        ptr=z["ptr"], post=z["post"], weight=z["weight"], body_id=z["body_id"],
        type=z["type"].astype(object), cls=z["cls"].astype(object),
        superclass=z["superclass"].astype(object), side=z["side"].astype(object),
        sign=z["sign"], meta=json.loads(str(z["meta"])),
    )
    g.validate()
    return g


def graph_for(condition: str, out_dir: Path = OUT_DIR, threshold: int = 5):
    """``"real"`` or ``"rw<seed>"`` (e.g. ``rw1``) -> a ``Graph``."""
    if condition == "real":
        return C.load(threshold)
    if condition.startswith("rw"):
        return build_or_load(int(condition[2:]), out_dir, threshold, verbose=False)
    raise ValueError(f"unknown condition {condition!r}")


# --------------------------------------------------------------------------- #
# invariant checks (also exercised by tests/test_calyx.py)
# --------------------------------------------------------------------------- #
def in_degree(graph) -> NDArray[np.int64]:
    return np.bincount(graph.post, minlength=graph.n).astype(np.int64)


def in_weight(graph) -> NDArray[np.float64]:
    out = np.zeros(graph.n, np.float64)
    np.add.at(out, graph.post, graph.weight.astype(np.float64))
    return out


def check_invariants(base, rw) -> dict:
    """Every invariant the rewiring promises, as a dict of booleans / counts."""
    ek = alpn_kc_edges(base)
    e = ek["edges"]
    mask = np.zeros(len(base.post), bool)
    mask[e] = True
    ids, _ = stratum_keys(base, ek)

    other_post_same = bool(np.array_equal(base.post[~mask], rw.post[~mask]))
    weights_same = bool(np.array_equal(base.weight, rw.weight))
    ptr_same = bool(np.array_equal(base.ptr, rw.ptr))
    indeg_same = bool(np.array_equal(in_degree(base), in_degree(rw)))
    inw_same = bool(np.allclose(in_weight(base), in_weight(rw), rtol=0, atol=1e-4))

    # strata respected: the multiset of postsynaptic targets inside each stratum
    # is unchanged.
    strata_ok = True
    for sid in np.unique(ids):
        sel = e[ids == sid]
        if not np.array_equal(np.sort(base.post[sel]), np.sort(rw.post[sel])):
            strata_ok = False
            break

    pre = edge_pre(rw)
    key = pre.astype(np.int64) * rw.n + rw.post.astype(np.int64)
    dup = int(len(key) - len(np.unique(key)))
    # The connectome itself contains 22 autapses, none of them ALPN -> KC; the
    # invariant is that the rewiring adds none.
    self_loops = int((pre == rw.post).sum())
    self_loops_base = int((edge_pre(base) == base.post).sum())
    self_loops_in_calyx = int((pre[e] == rw.post[e]).sum())
    return dict(
        non_alpn_kc_edges_identical=other_post_same,
        weights_identical=weights_same,
        ptr_identical=ptr_same,
        in_degree_preserved=indeg_same,
        in_weight_preserved=inw_same,
        strata_respected=strata_ok,
        duplicate_edges=dup,
        self_loops_total=self_loops,
        self_loops_in_original_graph=self_loops_base,
        self_loops_added=int(self_loops - self_loops_base),
        self_loops_in_alpn_kc=self_loops_in_calyx,
        n_alpn_kc_edges=int(len(e)),
        n_edges_moved=int((base.post[e] != rw.post[e]).sum()),
    )


def describe_strata(graph) -> dict:
    """Counts for the report: strata, sizes, singletons, per-type edge totals."""
    ek = alpn_kc_edges(graph)
    ids, keys = stratum_keys(graph, ek)
    sizes = np.bincount(ids)
    typ = graph.type.astype(str)
    side = graph.side.astype(str)
    post = ek["post"]
    by_type: Dict[str, int] = {}
    by_side_type: Dict[str, int] = {}
    for p in post:
        by_type[typ[p]] = by_type.get(typ[p], 0) + 1
        k = f"{side[p]}/{typ[p]}"
        by_side_type[k] = by_side_type.get(k, 0) + 1
    cnt = np.rint(np.abs(ek["weight"]) / C.W_SYN).astype(int)
    return dict(
        n_edges=int(len(post)),
        n_alpn=int(len(ek["alpn"])), n_kc=int(len(ek["kc"])),
        n_kc_with_alpn_input=int(len(np.unique(post))),
        n_strata=int(len(keys)),
        n_singleton_strata=int((sizes == 1).sum()),
        n_edges_in_singleton_strata=int(sizes[sizes == 1].sum()),
        stratum_size_median=float(np.median(sizes)),
        stratum_size_max=int(sizes.max()),
        synapse_count_min=int(cnt.min()), synapse_count_max=int(cnt.max()),
        synapse_count_mean=round(float(cnt.mean()), 2),
        n_inhibitory_edges=int((ek["weight"] < 0).sum()),
        edges_by_kc_type=dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        edges_by_side_and_kc_type=dict(sorted(by_side_type.items(),
                                              key=lambda kv: -kv[1])),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--attempts-per-edge", type=int, default=ATTEMPTS_PER_EDGE)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = C.load(5)
    info = dict(strata=describe_strata(base), rewirings={}, checks={})
    for s in [int(x) for x in a.seeds.split(",") if x.strip()]:
        g = build_or_load(s, out_dir, attempts_per_edge=a.attempts_per_edge)
        info["rewirings"][f"rw{s}"] = g.meta["calyx_rewiring"]
        info["checks"][f"rw{s}"] = check_invariants(base, g)
        print(f"[rw{s}] {json.dumps(info['checks'][f'rw{s}'])}", flush=True)
        del g
    (out_dir / "rewiring.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {out_dir / 'rewiring.json'}")


if __name__ == "__main__":
    main()
