"""Invariants of the ALPN -> Kenyon-cell calyx rewiring (scripts/calyx_common.py).

The point of this control is that *only* the calyx moves. Everything here is a
check that the rewired graph is the real one in every respect except which
Kenyon cell each projection neuron contacts:

* per-neuron in-degree and out-degree unchanged;
* per-neuron weighted in-total and out-total unchanged;
* strata respected (hemisphere, KC type string, synapse count, sign);
* no duplicate (pre, post) edge and no self-loop created;
* every edge that is not ALPN -> KC is bit-identical to the original.

The heavy checks run once on a small synthetic graph (fast, always) and once on
the real MaleCNS graph if its cache is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from flybalatro import connectome as C  # noqa: E402
from scripts import calyx_common as K  # noqa: E402


# --------------------------------------------------------------------------- #
# a small synthetic calyx
# --------------------------------------------------------------------------- #
def synthetic_graph(n_alpn: int = 12, n_kc: int = 40, seed: int = 0,
                    fan_out: int = 10):
    """ALPN -> KC plus decoy populations, built directly as a CSR ``Graph``."""
    rng = np.random.default_rng(seed)
    n_other = 6
    n = n_alpn + n_kc + n_other
    cls = np.array([C.ALPN_CLASS] * n_alpn + [C.KC_CLASS] * n_kc + [""] * n_other,
                   dtype=object)
    typ = np.array(
        [f"ALPN_{i}" for i in range(n_alpn)]
        + [("KCg-m" if i % 3 == 0 else "KCab-s" if i % 3 == 1 else "KCa'b'-m")
           for i in range(n_kc)]
        + ["APL"] * n_other, dtype=object)
    side = np.array(["L" if i % 2 == 0 else "R" for i in range(n)], dtype=object)
    sign = np.ones(n, np.float32)

    rows = []
    for i in range(n):
        if i < n_alpn:                       # ALPN -> KC, plus one non-KC edge
            k = min(fan_out, n_kc)
            targets = rng.choice(np.arange(n_alpn, n_alpn + n_kc),
                                 size=k, replace=False)
            counts = rng.integers(5, 9, size=k)
            extra = np.array([n_alpn + n_kc + (i % n_other)])
            targets = np.concatenate([targets, extra])
            counts = np.concatenate([counts, [7]])
        elif i < n_alpn + n_kc:              # KC -> KC and KC -> other
            k = min(3, n - n_alpn)
            targets = rng.choice(np.arange(n_alpn, n), size=k, replace=False)
            counts = rng.integers(5, 9, size=k)
        else:                                # other -> KC
            k = min(5, n_kc)
            targets = rng.choice(np.arange(n_alpn, n_alpn + n_kc), size=k,
                                 replace=False)
            counts = rng.integers(5, 9, size=k)
        # unique targets only: the real graph has no parallel edges
        uniq, keep = np.unique(targets, return_index=True)
        rows.append((uniq.astype(np.int32), counts[keep].astype(np.int32)))

    ptr = np.zeros(n + 1, np.int64)
    ptr[1:] = np.cumsum([len(t) for t, _ in rows])
    post = np.concatenate([t for t, _ in rows]).astype(np.int32)
    count = np.concatenate([c for _, c in rows]).astype(np.float32)
    weight = (count * C.W_SYN).astype(np.float32)
    g = C.Graph(ptr=ptr, post=post, weight=weight,
                body_id=np.arange(n, dtype=np.int64) + 1000,
                type=typ, cls=cls,
                superclass=np.array(["cb"] * n, dtype=object),
                side=side, sign=sign, meta=dict(threshold=5, w_syn=C.W_SYN))
    g.validate()
    return g


@pytest.fixture(scope="module")
def synth():
    return synthetic_graph()


@pytest.fixture(scope="module")
def synth_rw(synth):
    return K.rewire_alpn_kc(synth, seed=7).graph


def out_degree(g):
    return np.diff(g.ptr)


def out_weight(g):
    return np.array([g.weight[g.ptr[i]:g.ptr[i + 1]].sum() for i in range(g.n)],
                    np.float64)


# --------------------------------------------------------------------------- #
# structural invariants
# --------------------------------------------------------------------------- #
def test_out_degree_and_out_weight_preserved(synth, synth_rw):
    assert np.array_equal(out_degree(synth), out_degree(synth_rw))
    assert np.allclose(out_weight(synth), out_weight(synth_rw))
    assert np.array_equal(synth.ptr, synth_rw.ptr)
    assert np.array_equal(synth.weight, synth_rw.weight)


def test_in_degree_and_in_weight_preserved(synth, synth_rw):
    assert np.array_equal(K.in_degree(synth), K.in_degree(synth_rw))
    assert np.allclose(K.in_weight(synth), K.in_weight(synth_rw), atol=1e-5)


def test_non_alpn_kc_edges_bit_identical(synth, synth_rw):
    ek = K.alpn_kc_edges(synth)
    mask = np.zeros(len(synth.post), bool)
    mask[ek["edges"]] = True
    assert np.array_equal(synth.post[~mask], synth_rw.post[~mask])
    assert np.array_equal(synth.weight[~mask], synth_rw.weight[~mask])
    # and the ALPN -> KC edge positions are the same positions
    assert np.array_equal(K.alpn_kc_edges(synth_rw)["edges"], ek["edges"])


def test_strata_respected(synth, synth_rw):
    ek = K.alpn_kc_edges(synth)
    ids, keys = K.stratum_keys(synth, ek)
    for sid in np.unique(ids):
        sel = ek["edges"][ids == sid]
        assert np.array_equal(np.sort(synth.post[sel]), np.sort(synth_rw.post[sel])), (
            f"stratum {keys[sid]} changed its multiset of targets")


def test_strata_definition_is_side_type_count_sign(synth):
    """Two edges share a stratum id iff they share all four key components."""
    ek = K.alpn_kc_edges(synth)
    ids, keys = K.stratum_keys(synth, ek)
    side = synth.side.astype(str)
    typ = synth.type.astype(str)
    cnt = np.rint(np.abs(ek["weight"]) / C.W_SYN).astype(int)
    sgn = np.sign(ek["weight"]).astype(int)
    for i in range(len(ids)):
        p = ek["post"][i]
        assert keys[ids[i]] == (side[p], typ[p], int(cnt[i]), int(sgn[i]))


def test_no_duplicate_or_self_edges(synth, synth_rw):
    pre = K.edge_pre(synth_rw)
    key = pre.astype(np.int64) * synth_rw.n + synth_rw.post.astype(np.int64)
    assert len(np.unique(key)) == len(key), "rewiring created a parallel edge"
    ek = K.alpn_kc_edges(synth_rw)
    assert int((ek["pre"] == ek["post"]).sum()) == 0, "rewiring created a self-loop"
    base_self = int((K.edge_pre(synth) == synth.post).sum())
    assert int((pre == synth_rw.post).sum()) == base_self


def test_rewiring_actually_moved_edges(synth, synth_rw):
    ek = K.alpn_kc_edges(synth)
    moved = (synth.post[ek["edges"]] != synth_rw.post[ek["edges"]]).mean()
    assert moved > 0.5, f"only {moved:.2%} of calyx edges moved"
    st = synth_rw.meta["calyx_rewiring"]
    assert st["swaps_accepted"] > 0
    assert st["rejected_self_loop"] == 0


def test_deterministic_in_seed(synth):
    a = K.rewire_alpn_kc(synth, seed=11).graph
    b = K.rewire_alpn_kc(synth, seed=11).graph
    c = K.rewire_alpn_kc(synth, seed=12).graph
    assert np.array_equal(a.post, b.post)
    assert not np.array_equal(a.post, c.post)


def test_check_invariants_helper(synth, synth_rw):
    chk = K.check_invariants(synth, synth_rw)
    assert chk["non_alpn_kc_edges_identical"]
    assert chk["weights_identical"]
    assert chk["ptr_identical"]
    assert chk["in_degree_preserved"]
    assert chk["in_weight_preserved"]
    assert chk["strata_respected"]
    assert chk["duplicate_edges"] == 0
    assert chk["self_loops_added"] == 0
    assert chk["self_loops_in_alpn_kc"] == 0


def test_duplicate_rejection_is_exercised():
    """A dense synthetic calyx forces duplicate-edge rejections to happen."""
    g = synthetic_graph(n_alpn=6, n_kc=8, seed=3, fan_out=6)
    rw = K.rewire_alpn_kc(g, seed=5)
    assert rw.stats["rejected_duplicate_edge"] > 0
    K.check_invariants(g, rw.graph)


# --------------------------------------------------------------------------- #
# the real connectome, when its cache is present
# --------------------------------------------------------------------------- #
real_only = pytest.mark.skipif(
    not C.cache_path(5).exists(), reason="MaleCNS cache not built")


@pytest.fixture(scope="module")
def real_graph():
    return C.load(5)


@real_only
def test_real_graph_invariants(real_graph):
    rw = K.rewire_alpn_kc(real_graph, seed=1).graph
    chk = K.check_invariants(real_graph, rw)
    assert chk["non_alpn_kc_edges_identical"]
    assert chk["in_degree_preserved"]
    assert chk["in_weight_preserved"]
    assert chk["strata_respected"]
    assert chk["duplicate_edges"] == 0
    assert chk["self_loops_added"] == 0
    assert chk["self_loops_in_alpn_kc"] == 0
    assert chk["n_alpn_kc_edges"] == 19980
    assert chk["n_edges_moved"] / chk["n_alpn_kc_edges"] > 0.8


@real_only
def test_real_graph_preserves_per_alpn_target_type_profile(real_graph):
    """Each projection neuron keeps its distribution over KC subtype x side."""
    rw = K.rewire_alpn_kc(real_graph, seed=2).graph
    typ = real_graph.type.astype(str)
    side = real_graph.side.astype(str)
    ek = K.alpn_kc_edges(real_graph)
    ek2 = K.alpn_kc_edges(rw)
    from collections import Counter
    for alpn in np.unique(ek["pre"])[:60]:
        a = Counter(zip(side[ek["post"][ek["pre"] == alpn]],
                        typ[ek["post"][ek["pre"] == alpn]]))
        b = Counter(zip(side[ek2["post"][ek2["pre"] == alpn]],
                        typ[ek2["post"][ek2["pre"] == alpn]]))
        assert a == b


@real_only
def test_real_graph_non_calyx_pathways_untouched(real_graph):
    """ORN -> ALPN, APL both ways, KC -> KC, KC -> MBON, dopaminergic: identical."""
    rw = K.rewire_alpn_kc(real_graph, seed=3).graph
    cls = real_graph.cls.astype(str)
    typ = real_graph.type.astype(str)
    pre = K.edge_pre(real_graph)
    kc = np.zeros(real_graph.n, bool)
    kc[np.flatnonzero(cls == C.KC_CLASS)] = True
    alpn = np.zeros(real_graph.n, bool)
    alpn[np.flatnonzero(cls == C.ALPN_CLASS)] = True
    apl = np.zeros(real_graph.n, bool)
    apl[np.flatnonzero(typ == "APL")] = True
    orn = np.zeros(real_graph.n, bool)
    orn[real_graph.orn_indices()] = True
    mbon = np.zeros(real_graph.n, bool)
    mbon[real_graph.mbon_indices()] = True

    post = real_graph.post
    groups = {
        "ORN->ALPN": orn[pre] & alpn[post],
        "APL out": apl[pre],
        "APL in": apl[post],
        "KC->KC": kc[pre] & kc[post],
        "KC->MBON": kc[pre] & mbon[post],
        "everything not ALPN->KC": ~(alpn[pre] & kc[post]),
    }
    for name, m in groups.items():
        assert m.sum() > 0, name
        assert np.array_equal(real_graph.post[m], rw.post[m]), name


@real_only
def test_real_rewired_graph_builds_a_brain(real_graph):
    """The rewired graph is a real Graph: Tuning.apply and the LIF kernel run."""
    from flybalatro.brain import Brain
    from flybalatro.tuning import Tuning

    rw = K.rewire_alpn_kc(real_graph, seed=1).graph
    b = Brain(rw, seed=1, v_jitter=6.9, tuning=Tuning(apl_scale=2.0))
    assert b.n == real_graph.n
    info = b.tuning_info
    assert info["populations"]["KC"] == 4064
    drive = np.zeros(rw.n, np.float32)
    drive[rw.orn_indices()[:100]] = 30.0
    counts, _ = b.step(drive, 5.0)
    assert counts.sum() > 0
