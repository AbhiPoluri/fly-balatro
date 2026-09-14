"""Tests for the KC -> MBON plasticity rule: bounds, eligibility, compartment gating.

Everything here runs on a small synthetic graph so the tests are fast and the
expected answers are hand-computable; two tests at the end touch the real
connectome to pin the assignment that the experiment actually used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flybalatro import plasticity as P
from flybalatro.brain import Brain
from flybalatro.connectome import Graph

ROOT = Path(__file__).resolve().parents[1]
HAVE_CONNECTOME = (ROOT / "data" / "connectome_v2_t5.npz").exists()


# --------------------------------------------------------------------------- #
# a tiny hand-built mushroom body
# --------------------------------------------------------------------------- #
def _toy_graph() -> Graph:
    """4 KCs -> 4 MBONs, plus one PAM and one PPL1 neuron.

    indices: 0-3 KC, 4-7 MBON, 8 PAM01, 9 PPL101.
    MBON 4, 5 are glutamatergic (avoid) and innervated by PAM;
    MBON 6, 7 are cholinergic (approach) and innervated by PPL1.
    Every KC contacts every MBON, so the eligibility and compartment masks are
    the only things that can select edges.
    """
    n = 10
    edges = []
    for kc in range(4):
        for mb in range(4, 8):
            edges.append((kc, mb, 1.0 + kc + 0.1 * mb))
    edges.append((8, 4, 0.5))   # PAM -> MBON 4
    edges.append((8, 5, 0.5))   # PAM -> MBON 5
    edges.append((9, 6, 0.5))   # PPL1 -> MBON 6
    edges.append((9, 7, 0.5))   # PPL1 -> MBON 7
    edges.sort()
    pre = np.array([e[0] for e in edges], np.int64)
    post = np.array([e[1] for e in edges], np.int32)
    weight = np.array([e[2] for e in edges], np.float32)
    ptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=ptr[1:])
    return Graph(
        ptr=ptr, post=post, weight=weight,
        body_id=np.arange(n, dtype=np.int64),
        type=np.array(["KC", "KC", "KC", "KC", "MBON01", "MBON02", "MBON12",
                       "MBON14", "PAM01", "PPL101"], dtype=object),
        cls=np.array(["Kenyon_Cell"] * 4 + ["MBON"] * 4 + ["DAN", "DAN"],
                     dtype=object),
        superclass=np.array(["cb_intrinsic"] * n, dtype=object),
        side=np.array(["R"] * n, dtype=object),
        sign=np.ones(n, np.float32),
        meta=dict(threshold=5, w_syn=0.275),
    )


def _toy_valence(graph: Graph, post, weight) -> P.MbonValence:
    """The valence assignment the toy graph is meant to have, without the feather."""
    mbon = graph.mbon_indices()
    nt = np.array(["glutamate", "glutamate", "acetylcholine", "acetylcholine"],
                  dtype=object)
    valence = np.array([P.NT_VALENCE[str(x)] for x in nt], dtype=object)
    pam_in, ppl1_in = P.compartment_dominance(graph, post, weight, mbon)
    total = pam_in + ppl1_in
    dan = np.where(total <= 0, "",
                   np.where(pam_in >= 0.5 * total, "PAM", "PPL1")).astype(object)
    return P.MbonValence(
        mbon=mbon, types=graph.type.astype(str)[mbon],
        compartments=np.array(["c"] * len(mbon), dtype=object), nt=nt,
        valence=valence, dan=dan, pam_in=pam_in, ppl1_in=ppl1_in,
        used=(valence != "") & (dan != ""), scheme="toy",
    )


@pytest.fixture()
def toy():
    g = _toy_graph()
    brain = Brain(g, seed=0, v_jitter=0.0)
    val = _toy_valence(g, brain.post, brain.weight)
    return g, brain, val, P.KcMbonPlasticity(brain, g, val)


# --------------------------------------------------------------------------- #
def test_compartment_dominance_matches_construction(toy):
    _g, _brain, val, _pl = toy
    assert list(val.dan) == ["PAM", "PAM", "PPL1", "PPL1"]
    assert list(val.valence) == [P.AVOID, P.AVOID, P.APPROACH, P.APPROACH]
    assert val.used.all()


def test_target_sets_are_compartment_and_valence_gated(toy):
    g, _brain, val, pl = toy
    mbon = g.mbon_indices()
    # reward acts on avoidance MBONs in PAM compartments only
    assert sorted(pl.targets["reward"].mbons.tolist()) == [mbon[0], mbon[1]]
    # punishment acts on approach MBONs in PPL1 compartments only
    assert sorted(pl.targets["punish"].mbons.tolist()) == [mbon[2], mbon[3]]
    assert pl.targets["reward"].n == 4 * 2  # 4 KCs x 2 MBONs
    assert pl.targets["punish"].n == 4 * 2
    assert len(pl.all_edge) == 16


def test_only_eligible_synapses_change(toy):
    _g, brain, _val, pl = toy
    w0 = brain.weight.copy()
    counts = np.zeros(pl.n_kc, np.int32)
    counts[1] = 2  # only KC 1 fired, and fired to saturation
    pl.deliver("reward", counts, eta=0.5)
    changed = np.flatnonzero(brain.weight != w0)
    # exactly KC1 -> MBON 4,5
    assert len(changed) == 2
    assert set(pl.all_kc_row[np.isin(pl.all_edge, changed)].tolist()) == {1}
    assert set(brain.post[changed].tolist()) == set(pl.targets["reward"].mbons.tolist())


def test_punishment_does_not_touch_reward_compartment(toy):
    _g, brain, _val, pl = toy
    w0 = brain.weight.copy()
    counts = np.full(pl.n_kc, 4, np.int32)
    pl.deliver("punish", counts, eta=0.5)
    changed = np.flatnonzero(brain.weight != w0)
    assert set(brain.post[changed].tolist()) == set(pl.targets["punish"].mbons.tolist())


def test_silent_kc_never_changes(toy):
    _g, brain, _val, pl = toy
    w0 = brain.weight.copy()
    counts = np.zeros(pl.n_kc, np.int32)
    pl.deliver("reward", counts, eta=0.9)
    pl.deliver("punish", counts, eta=0.9)
    assert np.array_equal(brain.weight, w0)


def test_eligibility_saturates_at_kc_ref(toy):
    _g, _brain, _val, pl = toy
    e = pl.eligibility(np.array([0, 1, 2, 9], np.int32))
    assert e.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0])


def test_depression_factor_is_one_minus_eta_times_eligibility(toy):
    _g, brain, _val, pl = toy
    ts = pl.targets["reward"]
    counts = np.zeros(pl.n_kc, np.int32)
    counts[2] = 1  # eligibility 0.5
    before = brain.weight[ts.edge].copy()
    pl.deliver("reward", counts, eta=0.4)
    after = brain.weight[ts.edge]
    row2 = ts.kc_row == 2
    assert after[row2] == pytest.approx(before[row2] * (1 - 0.4 * 0.5))
    assert after[~row2] == pytest.approx(before[~row2])


def test_weights_stay_within_bounds_under_saturating_training(toy):
    _g, brain, _val, pl = toy
    counts = np.full(pl.n_kc, 10, np.int32)
    for _ in range(500):
        pl.deliver("reward", counts, eta=0.9)
        pl.deliver("punish", counts, eta=0.9)
    w = brain.weight[pl.all_edge]
    ratio = w / pl.all_w0
    assert (w > 0).all(), "depression must never make a weight negative"
    assert ratio.max() <= 1.0 + 1e-6, "depression must never exceed the original"
    assert ratio.min() >= P.WEIGHT_FLOOR - 1e-6
    stats = pl.weight_stats()
    assert not stats["any_negative"] and not stats["any_above_original"]
    assert stats["frac_at_floor"] == pytest.approx(1.0)


def test_reset_weights_restores_exactly(toy):
    _g, brain, _val, pl = toy
    w0 = brain.weight.copy()
    counts = np.full(pl.n_kc, 3, np.int32)
    pl.deliver("reward", counts, eta=0.3)
    pl.deliver("punish", counts, eta=0.3)
    assert not np.array_equal(brain.weight, w0)
    pl.reset_weights()
    assert np.array_equal(brain.weight, w0)


def test_zero_eta_is_a_no_op(toy):
    _g, brain, _val, pl = toy
    w0 = brain.weight.copy()
    pl.deliver("reward", np.full(pl.n_kc, 4, np.int32), eta=0.0)
    assert np.array_equal(brain.weight, w0)


def test_deliver_rejects_unknown_kind(toy):
    _g, _brain, _val, pl = toy
    with pytest.raises(ValueError):
        pl.deliver("shock", np.zeros(pl.n_kc, np.int32), eta=0.1)


def test_negative_kc_mbon_weight_is_rejected():
    g = _toy_graph()
    brain = Brain(g, seed=0, v_jitter=0.0)
    val = _toy_valence(g, brain.post, brain.weight)
    brain.weight[0] = -1.0
    with pytest.raises(ValueError, match="must be positive"):
        P.KcMbonPlasticity(brain, g, val)


# --------------------------------------------------------------------------- #
# decision rule
# --------------------------------------------------------------------------- #
def test_decider_drive_and_calibration():
    dec = P.Decider(approach=np.array([0, 1], np.int32),
                    avoid=np.array([2, 3], np.int32), window_ms=50.0)
    counts = np.array([2, 0, 1, 1], np.int32)
    # (1.0 - 1.0) spikes -> 0 Hz
    assert dec.raw_drive(counts) == pytest.approx(0.0)
    counts = np.array([4, 4, 0, 0], np.int32)
    assert dec.raw_drive(counts) == pytest.approx(4 * 1000 / 50.0)

    raw = np.linspace(-5, 5, 201)
    info = dec.calibrate(raw, explore=0.2)
    assert info["explore_achieved"] == pytest.approx(0.2, abs=0.02)
    # median state plays with probability play_bias_p
    p = 1.0 / (1.0 + np.exp(-(np.median(raw) + dec.bias) / dec.temperature))
    assert p == pytest.approx(dec.play_bias_p, abs=1e-6)


# --------------------------------------------------------------------------- #
# the real assignment the experiment used
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_CONNECTOME, reason="connectome cache not present")
def test_real_valence_assignment_is_the_documented_one():
    from flybalatro import connectome as C

    g = C.load(5)
    val = P.mbon_valence_table(g, g.post, g.weight, scheme="nt")
    s = val.summary()
    assert s["n_mbon"] == 97
    # Aso 2014: glutamatergic MBONs drive avoidance, GABA/ACh drive approach.
    assert set(s["avoid_types"]) >= {"MBON01", "MBON02", "MBON03", "MBON04",
                                     "MBON05", "MBON06", "MBON07"}
    assert {"MBON11", "MBON12", "MBON13", "MBON14", "MBON18"} <= set(s["approach_types"])
    # the reward arm is the glutamatergic PAM-compartment MBONs
    assert set(s["pam_avoid_types"]) == {"MBON01", "MBON02", "MBON03", "MBON04",
                                         "MBON05", "MBON06", "MBON07"}
    # MBON22 (calyx) and MBON34 have no dopaminergic input and are left out
    assert set(s["unassigned_types"]) == {"MBON22", "MBON34"}


@pytest.mark.skipif(not HAVE_CONNECTOME, reason="connectome cache not present")
def test_real_kc_mbon_weights_are_all_positive():
    from flybalatro import connectome as C

    g = C.load(5)
    brain = Brain(g, seed=1, v_jitter=0.0)
    val = P.mbon_valence_table(g, brain.post, brain.weight, scheme="nt")
    pl = P.KcMbonPlasticity(brain, g, val)
    assert len(pl.all_edge) > 30_000
    assert float(pl.all_w0.min()) > 0.0
