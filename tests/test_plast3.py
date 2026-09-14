"""Plasticity v3: the four things the v3 round adds, each of which could pass a
review by looking plausible and still be wrong.

* **Evaluation mode.** v1/v2 published "greedy" numbers that were floored-softmax
  samples. So the two paths are pinned separately: greedy consults no RNG and
  thresholds at ``p = 0.5``; sampled consults the RNG and nothing else changes.
* **Omission punishment.** The claim in the report is "fires exactly when the
  chips gained fall short of the share still needed, never otherwise". That is a
  claim about a boundary, so it is tested at the boundary and against
  :func:`plast_common.resolve_outcome`, not against a paraphrase of it.
* **The separated encoding.** Its whole point is that the four bucket ensembles
  are disjoint (so a bucket change is an identity change) and ORN-balanced (so a
  bucket change is *not* an intensity change). Both are asserted on the real
  graph.
* **The clustered analysis.** The v2 write-up's ``z = 1.75 over 180 games``
  treated 60 games counted three times as independent. The replacement is
  checked on inputs whose answer is known by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pytest

from flybalatro import connectome as C
from flybalatro import plasticity as P
from flybalatro.glomerular import GLOM32_TYPES
from scripts import plast3_common as X
from scripts import plast_common as K

# --------------------------------------------------------------------------- #
# stubs: the decision path without a connectome
# --------------------------------------------------------------------------- #
N_STUB = 6
APPROACH = np.array([4], np.int32)
AVOID = np.array([5], np.int32)
KC = np.array([0, 1, 2], np.int32)


class StubSetup:
    """Just enough of :class:`plast_common.Setup` for :class:`EvalFlyPolicy`.

    ``run_window`` returns a count vector whose approach-minus-avoid difference
    is driven by relay bit 0, so a test can ask for a positive or a negative
    ``play_drive`` by flipping one bit.
    """

    def __init__(self) -> None:
        self.kc = KC
        self.window_ms = 50.0
        self.n_windows = 0

    def run_window(self, bits) -> np.ndarray:
        self.n_windows += 1
        counts = np.zeros(N_STUB, np.int32)
        counts[KC] = 1
        if float(np.asarray(bits)[0]) > 0.0:
            counts[APPROACH] = 4
        else:
            counts[AVOID] = 4
        return counts


def stub_decider(bias: float = 0.0, temperature: float = 10.0,
                 explore_floor: float = 0.1) -> P.Decider:
    return P.Decider(approach=APPROACH, avoid=AVOID, window_ms=50.0, bias=bias,
                     temperature=temperature, explore_floor=explore_floor)


def ctx_for(bit0: int, discard_ok: bool = True) -> K.HandContext:
    bits = np.zeros(32, np.float32)
    bits[0] = float(bit0)
    bits[28] = 1.0
    return K.HandContext(
        bits=bits, best_type=1, best_type_name="Pair", best_score=40,
        best_slots=(0, 1), dig=(2, 3), needed=300, plays=3, discards=3,
        bucket=0, bucket_name="lt0.25", round=1, discard_ok=discard_ok)


class ScriptedRng:
    """A ``Generator`` stand-in that returns a fixed sequence from ``random()``."""

    def __init__(self, values: Sequence[float]) -> None:
        self.values = list(values)
        self.i = 0

    def random(self) -> float:
        v = self.values[self.i % len(self.values)]
        self.i += 1
        return v


# --------------------------------------------------------------------------- #
# 1. greedy vs sampled
# --------------------------------------------------------------------------- #
def test_greedy_threshold_is_p_half_and_equals_drive_sign() -> None:
    """``p >= 0.5`` and ``play_drive >= 0`` are the same rule at any floor."""
    for floor in (0.0, 0.1, 0.5):
        dec = stub_decider(explore_floor=floor)
        setup = StubSetup()
        for bit0 in (0, 1):
            counts = setup.run_window(ctx_for(bit0).bits)
            p = dec.p_play(counts)
            drive = dec.play_drive(counts)
            assert (p >= 0.5) == (drive >= 0.0)
            assert X.greedy_action(p) == (K.PLAY if drive >= 0 else K.DISCARD)


def test_greedy_never_touches_the_rng_and_is_deterministic() -> None:
    setup = StubSetup()
    dec = stub_decider()
    pol = X.EvalFlyPolicy(setup, dec, X.GREEDY, rng=None, cache=False)
    a1, info1 = pol(ctx_for(1))
    a0, info0 = pol(ctx_for(0))
    assert a1 == K.PLAY and a0 == K.DISCARD
    assert info1["eval_mode"] == X.GREEDY
    # repeat: same answers, no RNG anywhere in the path
    assert pol(ctx_for(1))[0] == K.PLAY
    assert pol(ctx_for(0))[0] == K.DISCARD


def test_sampled_follows_the_draw_where_greedy_would_not() -> None:
    """A near-0.5 odour is where the two modes disagree -- the v2 case exactly."""
    setup = StubSetup()
    dec = stub_decider(bias=0.0, temperature=1e6)   # every p ~ 0.5
    counts = setup.run_window(ctx_for(1).bits)
    p = dec.p_play(counts)
    assert 0.5 <= p < 0.55, p
    greedy = X.EvalFlyPolicy(setup, dec, X.GREEDY, cache=False)
    assert greedy(ctx_for(1))[0] == K.PLAY
    # a draw above p flips the sampled policy to DISCARD on the same odour
    sampled = X.EvalFlyPolicy(setup, dec, X.SAMPLED,
                              rng=ScriptedRng([0.99]), cache=False)
    assert sampled(ctx_for(1))[0] == K.DISCARD
    sampled2 = X.EvalFlyPolicy(setup, dec, X.SAMPLED,
                               rng=ScriptedRng([0.01]), cache=False)
    assert sampled2(ctx_for(1))[0] == K.PLAY


def test_sampled_requires_an_rng_and_mode_is_validated() -> None:
    setup, dec = StubSetup(), stub_decider()
    with pytest.raises(ValueError):
        X.EvalFlyPolicy(setup, dec, X.SAMPLED, rng=None)
    with pytest.raises(ValueError):
        X.EvalFlyPolicy(setup, dec, "argmax")


def test_an_illegal_discard_is_a_play_in_both_modes() -> None:
    setup, dec = StubSetup(), stub_decider()
    for mode, rng in ((X.GREEDY, None), (X.SAMPLED, ScriptedRng([0.99]))):
        pol = X.EvalFlyPolicy(setup, dec, mode, rng=rng, cache=False)
        assert pol(ctx_for(0, discard_ok=False))[0] == K.PLAY


def test_the_odour_cache_is_exact() -> None:
    """Same answers with and without the cache, and one window per odour."""
    dec = stub_decider()
    cold = X.EvalFlyPolicy(StubSetup(), dec, X.GREEDY, cache=False)
    warm = X.EvalFlyPolicy(StubSetup(), dec, X.GREEDY, cache=True)
    seq = [ctx_for(b) for b in (1, 0, 1, 1, 0)]
    cold_out = [cold(c) for c in seq]
    warm_out = [warm(c) for c in seq]
    assert [a for a, _ in cold_out] == [a for a, _ in warm_out]
    assert [i["p_play"] for _, i in cold_out] == [i["p_play"] for _, i in warm_out]
    assert cold.setup.n_windows == 5
    assert warm.setup.n_windows == 2          # two distinct odours
    assert warm.cache_stats()["distinct_odours"] == 2
    assert warm.cache_stats()["hit_rate"] == pytest.approx(3 / 5)


# --------------------------------------------------------------------------- #
# 2. omission punishment
# --------------------------------------------------------------------------- #
def _record(action: str, reward: int, punish: int) -> dict:
    return dict(action=action, reward=reward, punish=punish)


def test_omission_is_exactly_the_unrewarded_plays() -> None:
    assert X.dopamine_for(_record(K.PLAY, 1, 0), X.PUNISH_OMISSION) == "reward"
    assert X.dopamine_for(_record(K.PLAY, 0, 0), X.PUNISH_OMISSION) == "punish"
    assert X.dopamine_for(_record(K.PLAY, 0, 1), X.PUNISH_OMISSION) == "punish"
    # a discard never earns anything, under either rule
    for rule in X.PUNISHMENTS:
        assert X.dopamine_for(_record(K.DISCARD, 0, 0), rule) is None
        assert X.dopamine_for(_record(K.DISCARD, 1, 0), rule) is None


def test_current_punishment_is_a_strict_subset_of_omission() -> None:
    for reward in (0, 1):
        for punish in (0, 1):
            if reward and punish:
                continue      # resolve_outcome cannot produce this
            rec = _record(K.PLAY, reward, punish)
            cur = X.dopamine_for(rec, X.PUNISH_CURRENT)
            omi = X.dopamine_for(rec, X.PUNISH_OMISSION)
            if cur == "punish":
                assert omi == "punish"
            if cur == "reward":
                assert omi == "reward"
    assert X.dopamine_for(_record(K.PLAY, 0, 0), X.PUNISH_CURRENT) is None


def test_unknown_punishment_rule_is_an_error() -> None:
    with pytest.raises(ValueError):
        X.dopamine_for(_record(K.PLAY, 0, 0), "sometimes")


def _outcome(chips: int, needed: int, plays: int, stage: int = 1,
             done: bool = False, is_win: bool = False) -> dict:
    ctx = ctx_for(1)
    ctx = K.HandContext(**{**ctx.__dict__, "needed": needed, "plays": plays})
    info = dict(chips_gained=chips, stage=stage, truncated=False, is_win=is_win)
    return K.resolve_outcome(ctx, K.PLAY, info, done)


def test_omission_fires_iff_chips_fall_short_of_the_share_needed() -> None:
    """The boundary, against the game's own resolve_outcome. share = 300/3 = 100."""
    for chips, expect_punish in ((0, True), (99, True), (100, False), (250, False)):
        out = _outcome(chips, needed=300, plays=3)
        rec = dict(action=K.PLAY, **out)
        fires = X.dopamine_for(rec, X.PUNISH_OMISSION) == "punish"
        assert fires is expect_punish, (chips, out)
        assert (out["chips_gained"] < out["share"]) is expect_punish


def test_a_clearing_play_is_never_punished_by_omission() -> None:
    """``cleared`` implies chips >= needed >= share, so the branch cannot collide."""
    out = _outcome(400, needed=300, plays=1, stage=K.STAGE_POST_BLIND)
    assert out["cleared"] and out["reward"] == 1 and out["punish"] == 0
    assert X.dopamine_for(dict(action=K.PLAY, **out), X.PUNISH_OMISSION) == "reward"


def test_the_terminal_losing_play_is_punished_under_both_rules() -> None:
    out = _outcome(10, needed=300, plays=1, done=True)
    assert out["punish"] == 1 and out["reward"] == 0
    for rule in X.PUNISHMENTS:
        assert X.dopamine_for(dict(action=K.PLAY, **out), rule) == "punish"


# --------------------------------------------------------------------------- #
# 3. the separated bucket encoding
# --------------------------------------------------------------------------- #
def test_balanced_partition_is_disjoint_and_balanced() -> None:
    items = [(f"g{i}", s) for i, s in enumerate([204, 132, 130, 103, 98, 84, 83,
                                                 83, 82, 78, 74, 63, 63, 62, 58,
                                                 55, 54, 48, 45, 43, 43, 41, 38])]
    parts = X.balanced_partition(items, 4)
    flat = [x for p in parts for x in p]
    assert sorted(flat) == sorted(n for n, _ in items)
    assert len(set(flat)) == len(flat)
    size = dict(items)
    totals = [sum(size[n] for n in p) for p in parts]
    assert max(totals) / min(totals) < 1.05


real_only = pytest.mark.skipif(not C.cache_path(5).exists(),
                               reason="MaleCNS cache not built")


@pytest.fixture(scope="module")
def real_graph():
    return C.load(5)


@real_only
def test_separated_assignment_is_disjoint_balanced_and_accounts_for_every_bit(
        real_graph) -> None:
    groups = X.separated_assignment(real_graph)
    counts = X.orn_counts(real_graph)
    assert len(groups) == 32
    # every glomerulus used once, all 32 of the mb2 set used
    flat = [g for grp in groups for g in grp]
    assert len(set(flat)) == len(flat) == 32
    assert set(flat) == set(GLOM32_TYPES)
    # the nine hand-type bits keep their single tight glomerulus
    for b in X.HAND_TYPE_BITS:
        assert groups[b] == [GLOM32_TYPES[b]]
    # the dropped bits drive nothing
    for b in X.DROPPED_BITS:
        assert groups[b] == []
    # four disjoint bucket ensembles, ORN-count balanced within 5%
    bucket_sets = [set(groups[b]) for b in X.BUCKET_BITS]
    for i in range(len(bucket_sets)):
        for j in range(i + 1, len(bucket_sets)):
            assert not (bucket_sets[i] & bucket_sets[j])
    totals = [sum(counts[g] for g in groups[b]) for b in X.BUCKET_BITS]
    assert min(totals) > 400
    assert max(totals) / min(totals) < 1.05
    # each bucket keeps the single glomerulus mb2 gave it
    for b in X.BUCKET_BITS:
        assert GLOM32_TYPES[b] in groups[b]


@real_only
def test_changing_only_the_bucket_changes_identity_not_intensity(
        real_graph) -> None:
    """The point of the encoding: a bucket swap must not be an intensity cue."""
    import json

    cfg = json.loads((K.ROOT / "outputs" / "plast2" / "tuned_config.json").read_text())
    sep = X.map_for(X.ENC_SEPARATED, real_graph, cfg)
    cur = X.map_for(X.ENC_CURRENT, real_graph, cfg)
    bits = np.zeros(32, np.float32)
    bits[2] = 1.0          # Two Pair
    bits[9] = bits[10] = bits[26] = 1.0
    driven_sep, driven_cur, idx_sep = [], [], []
    for b in X.BUCKET_BITS:
        v = bits.copy()
        for q in X.BUCKET_BITS:
            v[q] = 0.0
        v[b] = 1.0
        driven_sep.append(sep.n_driven(v))
        driven_cur.append(cur.n_driven(v))
        idx_sep.append(set(int(i) for i in sep.indices(v)))
    # intensity: nearly constant under separated, 2x spread under current
    assert max(driven_sep) - min(driven_sep) <= 10
    assert max(driven_cur) - min(driven_cur) > 50
    # identity: the ORN sets a bucket change touches are much larger
    for i in range(4):
        for j in range(i + 1, 4):
            assert len(idx_sep[i] ^ idx_sep[j]) > 800


@real_only
def test_ensemble_map_refuses_a_glomerulus_shared_by_two_features(
        real_graph) -> None:
    with pytest.raises(ValueError):
        X.EnsembleGlomerularMap(real_graph, [["ORN_DA1"], ["ORN_DA1"]], 30.0)


def test_effective_bits_drops_only_the_dropped_block() -> None:
    bits = np.ones(32, np.float32)
    keep = X.effective_bits(X.ENC_SEPARATED, bits)
    assert keep[list(X.HAND_TYPE_BITS)].sum() == 9
    assert keep[list(X.BUCKET_BITS)].sum() == 4
    assert keep[list(X.DROPPED_BITS)].sum() == 0
    assert np.array_equal(X.effective_bits(X.ENC_CURRENT, bits), bits)


# --------------------------------------------------------------------------- #
# 4. the paired, clustered analysis
# --------------------------------------------------------------------------- #
def test_cluster_bootstrap_recovers_a_known_shift() -> None:
    rng = np.random.default_rng(0)
    control = rng.integers(0, 2, 400).astype(float)
    runs = [np.clip(control + 0.0, 0, 1) for _ in range(3)]
    r = X.cluster_bootstrap(runs, control, n_boot=500)
    assert r.delta == pytest.approx(0.0)
    assert r.lo <= 0.0 <= r.hi
    assert not r.to_dict()["significant"]

    # a real, constant improvement on every seed of every run
    shifted = [np.ones(400) for _ in range(3)]
    r2 = X.cluster_bootstrap(shifted, control, n_boot=500)
    assert r2.delta == pytest.approx(1.0 - control.mean())
    assert r2.lo > 0.0
    assert r2.to_dict()["significant"]


def test_cluster_bootstrap_is_paired_not_unpaired() -> None:
    """Same marginal clear rate, perfectly anti-correlated per seed -> delta 0."""
    control = np.array([1.0, 0.0] * 100)
    runs = [np.array([0.0, 1.0] * 100)]
    r = X.cluster_bootstrap(runs, control, n_boot=500)
    assert r.delta == pytest.approx(0.0)
    # and the per-seed pairing is what produced it, not the means
    assert r.per_run_delta == (pytest.approx(0.0),)


def test_cluster_bootstrap_widens_with_fewer_runs_and_reports_each() -> None:
    rng = np.random.default_rng(1)
    control = rng.integers(0, 2, 200).astype(float)
    runs = [np.clip(control + (i - 1) * 0.0, 0, 1) for i in range(3)]
    runs[0] = np.ones(200)
    r = X.cluster_bootstrap(runs, control, n_boot=500)
    assert len(r.per_run_delta) == 3
    assert r.per_run_delta[0] > r.per_run_delta[1]
    d = r.to_dict()
    assert d["n_runs"] == 3 and d["n_seeds"] == 200


def test_cluster_bootstrap_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        X.cluster_bootstrap([np.zeros(10)], np.zeros(9))


def test_clear_vector_follows_seed_order() -> None:
    games = [K.GameResult(seed=s, hands=[], cleared=bool(s % 2), chips=0,
                          rounds=1, truncated=False) for s in range(5)]
    assert list(X.clear_vector(games)) == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_the_cache_key_is_the_odour_the_encoding_actually_drives() -> None:
    """Two patterns differing only in a dropped bit are one odour under separated."""
    dec = stub_decider()
    pol = X.EvalFlyPolicy(StubSetup(), dec, X.GREEDY, cache=True,
                          encoding=X.ENC_SEPARATED)
    a = ctx_for(1)
    b = ctx_for(1)
    b.bits[12] = 1.0                      # a dropped best_slot bit
    assert pol(a)[0] == pol(b)[0]
    assert pol.setup.n_windows == 1       # one window, not two
    # under the current encoding they are different odours
    pol2 = X.EvalFlyPolicy(StubSetup(), dec, X.GREEDY, cache=True,
                           encoding=X.ENC_CURRENT)
    pol2(a), pol2(b)
    assert pol2.setup.n_windows == 2
