"""Calibration knobs: the default must be a no-op, and nothing may leak.

Runs on the tiny synthetic graph from ``test_realgame_brain`` (no APL, no real
``ALPN`` -> ``Kenyon_Cell`` naming beyond the class strings it happens to use),
so it also pins the "empty selection is fine, do not raise" behaviour that the
real graph never exercises.
"""

from __future__ import annotations

import numpy as np

from flybalatro.brain import V_TH, Brain
from flybalatro.tuning import Tuning, apl_indices
from tests.test_realgame_brain import N_TOTAL, synthetic_graph


def _drive(graph, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = np.zeros(graph.n, np.float32)
    orn = np.flatnonzero(graph.cls.astype(str) == "olfactory")
    d[rng.choice(orn, size=300, replace=False)] = 30.0
    return d


def _counts(graph, tuning, seed: int = 0) -> np.ndarray:
    b = Brain(graph, seed=1, v_jitter=6.9, tuning=tuning)
    b.warmup()
    b.reset()
    counts, _ = b.step(_drive(graph, seed), 50.0)
    return counts


def test_default_tuning_is_a_no_op() -> None:
    g = synthetic_graph()
    np.testing.assert_array_equal(_counts(g, None), _counts(g, Tuning()))
    assert Tuning().is_default()


def test_tuning_never_mutates_the_graph() -> None:
    g = synthetic_graph()
    before = g.weight.copy()
    Brain(g, tuning=Tuning(apl_scale=10.0, alpn_kc_scale=0.25))
    np.testing.assert_array_equal(g.weight, before)


def test_untuned_brain_is_unaffected_by_a_tuned_sibling() -> None:
    """The weight array is a private copy, so the two cannot share storage."""
    g = synthetic_graph()
    plain = Brain(g, seed=1, v_jitter=6.9)
    plain.warmup()
    plain.reset()
    first, _ = plain.step(_drive(g), 50.0)
    first = first.copy()
    tuned = Brain(g, seed=1, v_jitter=6.9, tuning=Tuning(alpn_kc_scale=0.1))
    tuned.warmup()
    plain.reset()
    again, _ = plain.step(_drive(g), 50.0)
    np.testing.assert_array_equal(first, again)


def test_empty_population_is_tolerated() -> None:
    """The synthetic graph has no APL; scaling it must be a silent no-op."""
    g = synthetic_graph()
    assert len(apl_indices(g)) == 0
    b = Brain(g, seed=1, v_jitter=6.9, tuning=Tuning(apl_scale=50.0))
    assert b.tuning_info["populations"]["APL"] == 0
    assert b.tuning_info["apl_all_out"]["edges"] == 0
    np.testing.assert_array_equal(_counts(g, None), _counts(g, Tuning(apl_scale=50.0)))


def test_kc_threshold_offset_reaches_the_kernel() -> None:
    g = synthetic_graph()
    kc = np.flatnonzero(g.cls.astype(str) == "Kenyon_Cell")
    b = Brain(g, seed=1, v_jitter=6.9, tuning=Tuning(kc_vth_offset_mv=6.0))
    assert b.vth[kc[0]] == np.float32(V_TH + 6.0)
    assert b.vth[0] == np.float32(V_TH)
    # Drive the KCs directly with a bias so the test does not depend on the
    # synthetic graph's ORN -> ALPN -> KC gain being enough to reach threshold.
    loud = _counts(g, Tuning(kc_bias_mv=8.0))[kc].sum()
    quiet = _counts(g, Tuning(kc_bias_mv=8.0, kc_vth_offset_mv=40.0))[kc].sum()
    assert loud > 0 and quiet == 0


def test_alpn_kc_scale_only_touches_alpn_to_kc_edges() -> None:
    g = synthetic_graph()
    b = Brain(g, seed=1, v_jitter=6.9, tuning=Tuning(alpn_kc_scale=0.5))
    alpn = np.flatnonzero(g.cls.astype(str) == "ALPN")
    kc_mask = np.zeros(N_TOTAL, bool)
    kc_mask[g.cls.astype(str) == "Kenyon_Cell"] = True
    touched = 0
    for i in alpn:
        a, bb = int(g.ptr[i]), int(g.ptr[i + 1])
        m = kc_mask[g.post[a:bb]]
        np.testing.assert_allclose(b.weight[a:bb][m], g.weight[a:bb][m] * 0.5)
        np.testing.assert_allclose(b.weight[a:bb][~m], g.weight[a:bb][~m])
        touched += int(m.sum())
    assert touched > 0
    # Everything outside ALPN's rows is untouched.
    orn = np.flatnonzero(g.cls.astype(str) == "olfactory")
    a, bb = int(g.ptr[orn[0]]), int(g.ptr[orn[0] + 1])
    np.testing.assert_allclose(b.weight[a:bb], g.weight[a:bb])


def test_kc_bias_is_added_to_the_drive() -> None:
    g = synthetic_graph()
    kc = np.flatnonzero(g.cls.astype(str) == "Kenyon_Cell")
    b = Brain(g, seed=1, v_jitter=6.9, tuning=Tuning(kc_bias_mv=8.0))
    assert b._has_bias and b.bias[kc[0]] == np.float32(8.0)
    b.warmup()
    b.reset()
    counts, _ = b.step(np.zeros(g.n, np.float32), 50.0)
    assert counts[kc].sum() > 0, "a tonic KC bias with no input must still fire KCs"


def test_roundtrip_dict() -> None:
    t = Tuning(apl_scale=5.0, kc_vth_offset_mv=4.0)
    assert Tuning.from_dict(t.to_dict()) == t
    assert "apl5" in t.label()
