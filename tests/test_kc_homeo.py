"""Per-Kenyon-cell homeostatic thresholds: indexing, bounds, and the no-op default.

The mechanism is one array addition (``vth[kc[i]] += kc_vth_offsets[i]``) plus one
per-KC edge multiplier (``kc_mbon_out_scale``), so what has to be pinned is not
the arithmetic but the *contract*: the vectors are indexed by
``graph.kc_indices()``, a length mismatch is an error rather than a broadcast, and
an empty vector leaves the brain bit-identical to the model that has never heard
of them. A silently misaligned offset vector would look exactly like a
neuroscience result, which is why the length check is a hard error.

Runs on the tiny synthetic graph from ``test_realgame_brain`` (200 Kenyon cells,
20 MBONs) -- no MaleCNS load.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flybalatro.brain import V_TH, Brain
from flybalatro.tuning import KC_CLASS, MBON_CLASS, Tuning
from tests.test_realgame_brain import N_KC, N_TOTAL, synthetic_graph

ROOT = Path(__file__).resolve().parent.parent


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


# --------------------------------------------------------------------------- #
# the indexing contract
# --------------------------------------------------------------------------- #
def test_kc_indices_is_the_index_apply_uses() -> None:
    """``graph.kc_indices()`` and ``Tuning.apply``'s ``kc`` must be the same order.

    ``apply`` builds its own ``np.flatnonzero(cls == "Kenyon_Cell")``; the offset
    vector is documented as indexed by ``kc_indices()``. If the two ever diverge
    every offset lands on the wrong cell.
    """
    g = synthetic_graph()
    np.testing.assert_array_equal(
        g.kc_indices(),
        np.flatnonzero(g.cls.astype(str) == KC_CLASS).astype(np.int32),
    )


def test_offsets_land_on_the_right_cells() -> None:
    g = synthetic_graph()
    kc = g.kc_indices()
    off = np.zeros(len(kc), np.float32)
    off[3] = 7.5
    off[-1] = -2.0
    b = Brain(g, seed=1, v_jitter=6.9, tuning=Tuning(kc_vth_offsets=tuple(off)))
    assert b.vth[kc[3]] == np.float32(V_TH + 7.5)
    assert b.vth[kc[-1]] == np.float32(V_TH - 2.0)
    assert b.vth[kc[4]] == np.float32(V_TH)
    # nothing outside the KC population moved
    non_kc = np.setdiff1d(np.arange(N_TOTAL), kc)
    np.testing.assert_array_equal(b.vth[non_kc], np.full(len(non_kc), V_TH, np.float32))


def test_offsets_stack_on_the_uniform_offset() -> None:
    g = synthetic_graph()
    kc = g.kc_indices()
    off = np.zeros(len(kc), np.float32)
    off[0] = 2.0
    b = Brain(g, seed=1, v_jitter=6.9,
              tuning=Tuning(kc_vth_offset_mv=4.0, kc_vth_offsets=tuple(off)))
    assert b.vth[kc[0]] == np.float32(V_TH + 6.0)
    assert b.vth[kc[1]] == np.float32(V_TH + 4.0)


def test_offsets_reach_the_kernel() -> None:
    """A large positive offset must silence exactly the cells it is applied to."""
    g = synthetic_graph()
    kc = g.kc_indices()
    loud = _counts(g, Tuning(kc_bias_mv=8.0))[kc]
    assert loud.sum() > 0
    half = np.zeros(len(kc), np.float32)
    half[: len(kc) // 2] = 60.0
    quiet = _counts(g, Tuning(kc_bias_mv=8.0, kc_vth_offsets=tuple(half)))[kc]
    assert quiet[: len(kc) // 2].sum() == 0
    assert quiet[len(kc) // 2:].sum() > 0


def test_wrong_length_is_an_error_not_a_broadcast() -> None:
    g = synthetic_graph()
    for bad in ((0.0,), tuple(np.zeros(N_KC - 1)), tuple(np.zeros(N_KC + 1))):
        with pytest.raises(ValueError, match="Kenyon cells"):
            Brain(g, tuning=Tuning(kc_vth_offsets=bad))
    with pytest.raises(ValueError, match="Kenyon cells"):
        Brain(g, tuning=Tuning(kc_mbon_out_scale=(1.0, 1.0)))


def test_non_finite_offsets_are_rejected() -> None:
    g = synthetic_graph()
    bad = np.zeros(N_KC, np.float32)
    bad[5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Brain(g, tuning=Tuning(kc_vth_offsets=tuple(bad)))


def test_negative_out_scale_is_rejected() -> None:
    g = synthetic_graph()
    bad = np.ones(N_KC, np.float32)
    bad[2] = -0.5
    with pytest.raises(ValueError, match="non-negative"):
        Brain(g, tuning=Tuning(kc_mbon_out_scale=tuple(bad)))


# --------------------------------------------------------------------------- #
# no-op by default
# --------------------------------------------------------------------------- #
def test_empty_vectors_are_a_no_op() -> None:
    g = synthetic_graph()
    t = Tuning()
    assert t.is_default()
    assert t.label() == "default"
    assert "kc_vth_offsets" not in t.to_dict()
    assert "kc_mbon_out_scale" not in t.to_dict()
    np.testing.assert_array_equal(_counts(g, None), _counts(g, Tuning()))
    assert "kc_vth_offsets" not in Brain(g, tuning=Tuning()).tuning_info


def test_all_zero_offsets_change_nothing_in_the_kernel() -> None:
    g = synthetic_graph()
    zero = tuple(np.zeros(N_KC, np.float32))
    np.testing.assert_array_equal(_counts(g, None),
                                  _counts(g, Tuning(kc_vth_offsets=zero)))
    ones = tuple(np.ones(N_KC, np.float32))
    np.testing.assert_array_equal(_counts(g, None),
                                  _counts(g, Tuning(kc_mbon_out_scale=ones)))


def test_a_populated_vector_is_not_default() -> None:
    off = tuple(np.zeros(N_KC, np.float32))
    t = Tuning(kc_vth_offsets=off)
    assert not t.is_default()
    assert f"vtho{N_KC}" in t.label()
    t2 = Tuning(kc_mbon_out_scale=tuple(np.ones(N_KC, np.float32)))
    assert not t2.is_default()
    assert f"kcmb{N_KC}" in t2.label()


def test_roundtrip_with_vectors() -> None:
    off = tuple(float(x) for x in np.linspace(-3.0, 9.0, N_KC))
    sc = tuple(float(x) for x in np.linspace(0.1, 1.0, N_KC))
    t = Tuning(apl_scale=2.0, apl_kc_only=1.0, apl_mbon_scale=0.1,
               kc_vth_offsets=off, kc_mbon_out_scale=sc)
    d = t.to_dict()
    assert isinstance(d["kc_vth_offsets"], list) and len(d["kc_vth_offsets"]) == N_KC
    assert Tuning.from_dict(json.loads(json.dumps(d))) == t
    assert hash(t) == hash(Tuning.from_dict(d))


def test_unknown_keys_still_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tuning keys"):
        Tuning.from_dict({"kc_vth_offsetz": [1.0]})


# --------------------------------------------------------------------------- #
# the fallback multiplier
# --------------------------------------------------------------------------- #
def test_out_scale_touches_only_kc_to_mbon_edges() -> None:
    g = synthetic_graph()
    kc = g.kc_indices()
    mbon_mask = np.zeros(g.n, bool)
    mbon_mask[np.flatnonzero(g.cls.astype(str) == MBON_CLASS)] = True
    sc = np.ones(len(kc), np.float32)
    sc[::2] = 0.25
    b = Brain(g, seed=1, v_jitter=6.9, tuning=Tuning(kc_mbon_out_scale=tuple(sc)))
    touched = 0
    for row, i in enumerate(kc):
        a, bb = int(g.ptr[i]), int(g.ptr[i + 1])
        m = mbon_mask[g.post[a:bb]]
        np.testing.assert_allclose(b.weight[a:bb][m], g.weight[a:bb][m] * sc[row],
                                   rtol=1e-6)
        np.testing.assert_allclose(b.weight[a:bb][~m], g.weight[a:bb][~m])
        touched += int(m.sum())
    assert touched > 0
    alpn = np.flatnonzero(g.cls.astype(str) == "ALPN")
    a, bb = int(g.ptr[alpn[0]]), int(g.ptr[alpn[0] + 1])
    np.testing.assert_allclose(b.weight[a:bb], g.weight[a:bb])
    assert b.tuning_info["kc_mbon_out_scale"]["n_scaled_kc"] == int((sc != 1.0).sum())


def test_vectors_do_not_mutate_the_graph() -> None:
    g = synthetic_graph()
    w_before = g.weight.copy()
    Brain(g, tuning=Tuning(kc_vth_offsets=tuple(np.full(N_KC, 5.0)),
                           kc_mbon_out_scale=tuple(np.full(N_KC, 0.5))))
    np.testing.assert_array_equal(g.weight, w_before)


# --------------------------------------------------------------------------- #
# the shipped calibration
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (ROOT / "outputs" / "plast2" / "tuned_config.json").exists(),
                    reason="outputs/plast2/tuned_config.json not built yet")
def test_shipped_offsets_are_bounded_and_the_right_length() -> None:
    """The saved operating point must obey the clamp the calibration declares."""
    cfg = json.loads((ROOT / "outputs" / "plast2" / "tuned_config.json").read_text())
    t = Tuning.from_dict(cfg["tuning"])
    off = np.asarray(t.kc_vth_offsets, np.float64)
    assert len(off) == 4064, "MaleCNS v1.0 brain-only has 4064 Kenyon cells"
    assert np.isfinite(off).all()
    homeo = json.loads((ROOT / "outputs" / "plast2" / "kc_homeo.json").read_text())
    lo, hi = homeo["offset_clamp"]
    assert off.min() >= lo - 1e-6 and off.max() <= hi + 1e-6
    # and above every cell's own resting potential: v_th = -45 + offset > v_rest
    assert off.min() > -7.0, "a KC threshold below v_rest fires with no odour"
    if t.kc_mbon_out_scale:
        sc = np.asarray(t.kc_mbon_out_scale, np.float64)
        assert len(sc) == len(off)
        assert sc.min() >= 0.0 and sc.max() <= 1.0 + 1e-9
