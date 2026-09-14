"""Tests for plasticity v4: the reward predicates, the eligibility trace, and the
reimplemented game loop.

Each new reward term is tested at its boundary conditions, as the brief for this
round requires: exactly on schedule, one chip under, the blind-clearing case
where the game reports ``chips_gained = 0``, and the last play of a blind.
"""

from __future__ import annotations

import numpy as np
import pytest

from flybalatro import plasticity as P
from scripts import plast3_common as C3
from scripts import plast4_common as C4
from scripts import plast_common as K


# --------------------------------------------------------------------------- #
# share / pace at their boundaries
# --------------------------------------------------------------------------- #
def test_share_ok_boundary():
    # needed 300 over 4 plays -> the bar is exactly 75.
    assert C4.share_ok(300, 4, 75)
    assert not C4.share_ok(300, 4, 74)
    # last play: the bar is the whole remaining requirement, i.e. clearing.
    assert C4.share_ok(96, 1, 96)
    assert not C4.share_ok(96, 1, 95)


def test_share_is_self_consistent_over_a_blind():
    """Exactly making the share every play clears the blind exactly."""
    required, plays = 300.0, 4
    needed, total = required, 0.0
    for p in range(plays, 0, -1):
        chips = needed / p
        assert C4.share_ok(int(round(needed)), p, chips)
        total += chips
        needed -= chips
    assert total == pytest.approx(required)
    assert needed == pytest.approx(0.0)


def test_pace_ok_boundary():
    # required 400, P0 = 4 -> schedule is 100 / 200 / 300 / 400.
    assert C4.pace_ok(400, 0, 4, 4, 100)          # exactly on schedule
    assert not C4.pace_ok(400, 0, 4, 4, 99)       # one chip under
    assert C4.pace_ok(400, 100, 3, 4, 100)        # still exactly on schedule
    assert not C4.pace_ok(400, 100, 3, 4, 99)
    # last play (plays = 1): the schedule demands the whole requirement.
    assert C4.pace_ok(400, 300, 1, 4, 100)
    assert not C4.pace_ok(400, 300, 1, 4, 99)


def test_pace_and_share_agree_on_the_first_play_of_a_blind():
    for required in (300, 450, 600):
        bar = required // 4
        assert C4.pace_ok(required, 0, 4, 4, bar) == C4.share_ok(required, 4, bar)
        assert (C4.pace_ok(required, 0, 4, 4, bar - 1)
                == C4.share_ok(required, 4, bar - 1))


def test_pace_is_stricter_when_behind_and_looser_when_ahead():
    required, p0 = 400, 4
    # behind schedule: banked 40 after one play, schedule wanted 100.
    assert C4.share_ok(360, 3, 120) and not C4.pace_ok(required, 40, 3, p0, 120)
    # ahead of schedule: banked 250 after one play, schedule wanted 100.
    assert C4.pace_ok(required, 250, 3, p0, 0) and not C4.share_ok(150, 3, 0)


def _info(chips: int, stage: int = 1, win: bool = False,
          truncated: bool = False) -> dict:
    return dict(chips_gained=chips, stage=stage, is_win=win, truncated=truncated)


def test_resolve_outcome4_clearing_play_is_rewarded_under_both_rules():
    """The game reports ``chips_gained = 0`` on the play that ends a blind."""
    out = C4.resolve_outcome4(required=300, score_before=204, needed=96, plays=2,
                              p0=4, action=K.PLAY,
                              info=_info(0, stage=K.STAGE_POST_BLIND), done=False,
                              reward_rule=C4.REWARD_SHARE)
    assert out["cleared"] and out["reward"] == 1
    assert out["reward_share"] == 1 and out["reward_pace"] == 1
    assert out["punish"] == 0


def test_resolve_outcome4_last_play_non_clearing_is_punished_and_never_rewarded():
    out = C4.resolve_outcome4(required=600, score_before=136, needed=464, plays=1,
                              p0=4, action=K.PLAY,
                              info=_info(28, stage=7), done=True,
                              reward_rule=C4.REWARD_SHARE)
    assert out["lost"] and out["reward"] == 0 and out["punish"] == 1
    # The arithmetic of section 2 of the preregistration: on the last play the
    # share IS the whole requirement, so reward-and-lost cannot happen.
    assert not (out["reward_share"] and out["lost"])
    assert not (out["reward_pace"] and out["lost"])


def test_resolve_outcome4_reward_rule_selects_the_branch():
    kw = dict(required=400, score_before=40, needed=360, plays=3, p0=4,
              action=K.PLAY, info=_info(120), done=False)
    s = C4.resolve_outcome4(reward_rule=C4.REWARD_SHARE, **kw)
    p = C4.resolve_outcome4(reward_rule=C4.REWARD_PACE, **kw)
    assert s["reward"] == 1 and p["reward"] == 0
    assert s["reward_share"] == p["reward_share"] == 1
    assert s["reward_pace"] == p["reward_pace"] == 0


def test_resolve_outcome4_discard_never_earns_anything():
    out = C4.resolve_outcome4(required=300, score_before=0, needed=300, plays=4,
                              p0=4, action=K.DISCARD, info=_info(0), done=False,
                              reward_rule=C4.REWARD_PACE)
    assert out["reward"] == 0 and out["punish"] == 0
    assert out["reward_share"] == 0 and out["reward_pace"] == 0


def test_resolve_outcome4_rejects_unknown_rule():
    with pytest.raises(ValueError):
        C4.resolve_outcome4(300, 0, 300, 4, 4, K.PLAY, _info(0), False, "teacher")


def test_share_branch_reproduces_plast_common_resolve_outcome():
    """v4's share branch must be v3's reward, bit for bit, on every case."""
    rng = np.random.default_rng(4)
    for _ in range(400):
        required = int(rng.choice([300, 450, 600]))
        plays = int(rng.integers(1, 5))
        score_before = int(rng.integers(0, required))
        needed = required - score_before
        chips = int(rng.integers(0, required))
        done = bool(rng.integers(0, 2))
        stage = int(rng.choice([1, K.STAGE_POST_BLIND, 7]))
        action = K.PLAY if rng.integers(0, 2) else K.DISCARD
        info = _info(chips, stage=stage)
        ctx = K.HandContext(
            bits=np.zeros(32, np.float32), best_type=0, best_type_name="x",
            best_score=chips, best_slots=(), dig=(), needed=needed, plays=plays,
            discards=1, bucket=0, bucket_name="lt0.25", round=0, discard_ok=True)
        old = K.resolve_outcome(ctx, action, info, done)
        new = C4.resolve_outcome4(required, score_before, needed, plays, 4,
                                  action, info, done, C4.REWARD_SHARE)
        for key in ("chips_gained", "cleared", "lost", "truncated", "share",
                    "reward", "punish"):
            assert old[key] == new[key], (key, old, new)


# --------------------------------------------------------------------------- #
# deliver_eligibility
# --------------------------------------------------------------------------- #
class _FakeBrain:
    def __init__(self, n_edges: int) -> None:
        self.weight = np.linspace(0.5, 2.0, n_edges).astype(np.float32)


class _FakePlast:
    """The two methods :class:`plast4_common.TraceRunner` actually needs."""

    def __init__(self, n_kc: int = 8) -> None:
        self.n_kc = n_kc
        self.calls: list = []

    def eligibility(self, kc_counts):
        return np.minimum(np.asarray(kc_counts, np.float32) / 2.0, 1.0)

    def deliver_eligibility(self, kind, elig, eta):
        self.calls.append((kind, np.asarray(elig, np.float32).copy(), float(eta)))
        return dict(kind=kind, n_edges=1, n_changed=1, mean_abs_dw=0.1,
                    sum_abs_dw=0.1, elig_kc=int((np.asarray(elig) > 0).sum()))


@pytest.fixture(scope="module")
def plast_setup():
    graph = K.C.load(5)
    import json
    cfg = json.loads((K.ROOT / "outputs" / "plast3" / "tuned_config_sep.json").read_text())
    return C3.build_setup3(C3.ENC_SEPARATED, cfg, graph)


def test_deliver_eligibility_matches_deliver(plast_setup):
    """The trace rule and the immediate rule are the same depression step."""
    plast = plast_setup.plast
    rng = np.random.default_rng(11)
    counts = rng.integers(0, 5, plast.n_kc).astype(np.int32)
    for kind in ("reward", "punish"):
        plast.reset_weights()
        a = plast.deliver(kind, counts, 0.05)
        wa = plast.brain.weight[plast.all_edge].copy()
        plast.reset_weights()
        b = plast.deliver_eligibility(kind, plast.eligibility(counts), 0.05)
        wb = plast.brain.weight[plast.all_edge].copy()
        assert np.array_equal(wa, wb)
        assert a["n_changed"] == b["n_changed"]
        assert a["mean_abs_dw"] == pytest.approx(b["mean_abs_dw"])
        assert a["elig_kc"] == b["elig_kc"]
    plast.reset_weights()


def test_deliver_eligibility_validates(plast_setup):
    plast = plast_setup.plast
    good = np.zeros(plast.n_kc, np.float32)
    with pytest.raises(ValueError):
        plast.deliver_eligibility("teacher", good, 0.05)
    with pytest.raises(ValueError):
        plast.deliver_eligibility("punish", np.zeros(3, np.float32), 0.05)
    bad = good.copy()
    bad[0] = 1.5
    with pytest.raises(ValueError):
        plast.deliver_eligibility("punish", bad, 0.05)
    bad[0] = -0.1
    with pytest.raises(ValueError):
        plast.deliver_eligibility("punish", bad, 0.05)


def test_deliver_eligibility_zero_eta_is_a_no_op(plast_setup):
    plast = plast_setup.plast
    plast.reset_weights()
    before = plast.brain.weight[plast.all_edge].copy()
    info = plast.deliver_eligibility("punish", np.ones(plast.n_kc, np.float32), 0.0)
    assert info["n_changed"] == 0
    assert np.array_equal(before, plast.brain.weight[plast.all_edge])


def test_deliver_eligibility_respects_the_weight_floor(plast_setup):
    plast = plast_setup.plast
    plast.reset_weights()
    full = np.ones(plast.n_kc, np.float32)
    for _ in range(400):
        plast.deliver_eligibility("punish", full, 0.5)
    stats = plast.weight_stats()
    assert stats["min_ratio"] >= P.WEIGHT_FLOOR - 1e-6
    assert not stats["any_negative"]
    plast.reset_weights()


# --------------------------------------------------------------------------- #
# the eligibility trace
# --------------------------------------------------------------------------- #
def test_trace_decays_geometrically_and_clips():
    t = C4.EligibilityTrace(4, 0.5)
    e = np.array([1.0, 0.5, 0.0, 0.25], np.float32)
    t.update(e, "lt0.5")
    assert t.trace == pytest.approx([1.0, 0.5, 0.0, 0.25])
    t.update(e, "lt1.0")
    # 0.5 * previous + e, clipped at 1
    assert t.trace == pytest.approx([1.0, 0.75, 0.0, 0.375])
    assert float(t.trace.max()) <= 1.0


def test_trace_gamma_zero_keeps_only_the_last_play():
    t = C4.EligibilityTrace(3, 0.0)
    t.update(np.array([1.0, 0.0, 0.0], np.float32), "a")
    t.update(np.array([0.0, 1.0, 0.0], np.float32), "b")
    assert t.trace == pytest.approx([0.0, 1.0, 0.0])
    assert t.weight_by_bucket() == {"a": 0.0, "b": 1.0}


def test_trace_weight_by_bucket_is_recency_weighted():
    t = C4.EligibilityTrace(2, 0.8)
    for b in ("lt0.25", "lt0.5", "lt0.25"):
        t.update(np.array([1.0, 0.0], np.float32), b)
    w = t.weight_by_bucket()
    assert w["lt0.25"] == pytest.approx(0.8 ** 2 + 1.0)
    assert w["lt0.5"] == pytest.approx(0.8)


def test_trace_reset_clears_everything():
    t = C4.EligibilityTrace(3, 0.9)
    t.update(np.ones(3, np.float32), "ge1.0")
    t.reset()
    assert not t.contributions
    assert float(t.trace.sum()) == 0.0


def test_trace_rejects_bad_gamma_and_shape():
    with pytest.raises(ValueError):
        C4.EligibilityTrace(3, 1.5)
    with pytest.raises(ValueError):
        C4.EligibilityTrace(3, -0.1)
    with pytest.raises(ValueError):
        C4.EligibilityTrace(3, 0.5).update(np.ones(4, np.float32))


# --------------------------------------------------------------------------- #
# TraceRunner: reset at a blind, PLAY only, fire iff lost
# --------------------------------------------------------------------------- #
def _rec(seed=1, rnd=0, action=K.PLAY, lost=False, bucket="lt0.5") -> dict:
    return dict(seed=seed, round=rnd, action=action, lost=lost,
                bucket_name=bucket)


def test_trace_runner_fires_only_on_lost():
    fp = _FakePlast()
    tr = C4.TraceRunner(fp, 0.8, 0.0797)
    counts = np.full(fp.n_kc, 2, np.int32)
    assert tr.on_record(_rec(), counts) == {}
    assert fp.calls == []
    out = tr.on_record(_rec(lost=True), counts)
    assert out["trace_pulse"] is True
    assert len(fp.calls) == 1 and fp.calls[0][0] == "punish"
    assert fp.calls[0][2] == pytest.approx(0.0797)


def test_trace_runner_resets_at_a_blind_boundary():
    fp = _FakePlast()
    tr = C4.TraceRunner(fp, 1.0, 0.05)
    counts = np.full(fp.n_kc, 2, np.int32)
    tr.on_record(_rec(rnd=0, bucket="lt0.25"), counts)
    tr.on_record(_rec(rnd=0, bucket="lt0.25"), counts)
    # new blind: the Small Blind's plays must not be punishable by a Big Blind loss
    out = tr.on_record(_rec(rnd=1, bucket="ge1.0", lost=True), counts)
    assert out["trace_by_bucket"] == {"ge1.0": 1.0}
    assert "lt0.25" not in tr.weight_by_bucket


def test_trace_runner_ignores_discards():
    fp = _FakePlast()
    tr = C4.TraceRunner(fp, 1.0, 0.05)
    counts = np.full(fp.n_kc, 2, np.int32)
    tr.on_record(_rec(action=K.DISCARD, bucket="lt0.25"), counts)
    out = tr.on_record(_rec(action=K.PLAY, bucket="lt1.0", lost=True), counts)
    assert out["trace_by_bucket"] == {"lt1.0": 1.0}


def test_trace_runner_does_not_fire_on_an_empty_trace():
    """A blind lost on a forced PLAY with no prior play still traces that play;
    a loss with nothing in the trace at all fires nothing."""
    fp = _FakePlast()
    tr = C4.TraceRunner(fp, 0.8, 0.05)
    counts = np.full(fp.n_kc, 2, np.int32)
    assert tr.on_record(_rec(action=K.DISCARD, lost=True), counts) == {}
    assert fp.calls == []


def test_trace_runner_zero_eta_fires_nothing():
    fp = _FakePlast()
    tr = C4.TraceRunner(fp, 0.8, 0.0)
    counts = np.full(fp.n_kc, 2, np.int32)
    tr.on_record(_rec(), counts)
    assert tr.on_record(_rec(lost=True), counts) == {}
    assert fp.calls == []


def test_trace_runner_resets_after_firing():
    fp = _FakePlast()
    tr = C4.TraceRunner(fp, 1.0, 0.05)
    counts = np.full(fp.n_kc, 2, np.int32)
    tr.on_record(_rec(lost=True), counts)
    assert not tr.trace.contributions
    assert tr.n_pulses == 1


# --------------------------------------------------------------------------- #
# the reimplemented game loop
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("policy_name", ["always_play", "always_discard",
                                         "bucket_ge_lt1"])
def test_play_game4_matches_play_game(policy_name):
    """The v4 loop must be the v3 loop plus extra recorded fields, nothing else."""
    policies = dict(
        always_play=K.always_play_policy,
        always_discard=K.make_always_discard_policy(),
        bucket_ge_lt1=K.make_bucket_policy(["ge1.0", "lt1.0"]),
    )
    pol = policies[policy_name]
    env_a, env_b = K.make_env(), K.make_env()
    for seed in (400000, 400001, 400002, 400017, 400099):
        old = K.play_game(env_a, seed, pol)
        new = C4.play_game4(env_b, seed, pol)
        assert old.cleared == new.cleared
        assert old.chips == new.chips
        assert old.rounds == new.rounds
        assert old.truncated == new.truncated
        assert len(old.hands) == len(new.hands)
        for a, b in zip(old.hands, new.hands):
            for key in a:
                assert a[key] == b[key], (seed, key, a, b)


def test_play_game4_records_the_blind_quantities():
    env = K.make_env()
    g = C4.play_game4(env, 400000, K.always_play_policy)
    assert g.hands
    for r in g.hands:
        assert r["required"] == r["score_before"] + r["needed"]
        assert r["p0"] >= r["plays_before"]
        assert r["score_after"] == r["score_before"] + r["chips_gained"]
    first = g.hands[0]
    assert first["score_before"] == 0 and first["p0"] == first["plays_before"]


def test_play_game4_pace_rule_changes_only_the_reward_field():
    env = K.make_env()
    a = C4.play_game4(env, 400003, K.always_play_policy, reward_rule=C4.REWARD_SHARE)
    b = C4.play_game4(env, 400003, K.always_play_policy, reward_rule=C4.REWARD_PACE)
    assert [h["action"] for h in a.hands] == [h["action"] for h in b.hands]
    assert a.cleared == b.cleared and a.chips == b.chips
    for x, y in zip(a.hands, b.hands):
        assert x["reward_share"] == y["reward_share"]
        assert x["reward_pace"] == y["reward_pace"]
        assert x["reward"] == x["reward_share"]
        assert y["reward"] == y["reward_pace"]


def test_dopamine_for4_is_the_omission_rule():
    assert C4.dopamine_for4(dict(action=K.DISCARD, reward=0)) is None
    assert C4.dopamine_for4(dict(action=K.PLAY, reward=1)) == "reward"
    assert C4.dopamine_for4(dict(action=K.PLAY, reward=0)) == "punish"


# --------------------------------------------------------------------------- #
# conditions and statistics
# --------------------------------------------------------------------------- #
def test_conditions_are_the_preregistered_set():
    names = [c.name for c in C4.conditions()]
    assert names == ["A", "B", "C50", "C80", "C100", "D"]
    by = {c.name: c for c in C4.conditions()}
    assert by["A"].reward_rule == C4.REWARD_SHARE and by["A"].gamma is None
    assert by["B"].reward_rule == C4.REWARD_PACE and by["B"].gamma is None
    assert by["C80"].gamma == 0.8 and by["C80"].reward_rule == C4.REWARD_SHARE
    assert by["D"].gamma == C4.TRACE_GAMMA_D and by["D"].reward_rule == C4.REWARD_PACE
    assert C4.NOVEL_FAMILIES == ("B", "C80", "D")


def test_holm_step_down():
    assert C4.holm([0.01]) == [0.01]
    adj = C4.holm([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)          # monotone: never below an earlier one
    assert all(0.0 <= p <= 1.0 for p in C4.holm([0.5, 0.6, 0.9]))


def test_mcnemar_exact_boundaries():
    assert C4.mcnemar_exact([1, 1, 0], [1, 1, 0])["p_exact"] == 1.0
    r = C4.mcnemar_exact([1] * 10 + [0] * 90, [0] * 100)
    assert r["b10"] == 10 and r["b01"] == 0
    assert r["p_exact"] == pytest.approx(2.0 / 1024)
    sym = C4.mcnemar_exact([1, 0], [0, 1])
    assert sym["p_exact"] == 1.0
    with pytest.raises(ValueError):
        C4.mcnemar_exact([1, 0], [1, 0, 1])


# --------------------------------------------------------------------------- #
# the constraint the whole project runs under
# --------------------------------------------------------------------------- #
def test_no_reward_term_consults_the_hand_analyser():
    """Every reward term must be computable from the game's payouts and costs.

    ``hands.py`` produces the odour the fly smells; it must never tell the reward
    whether the action was correct. Checked structurally: the reward path takes
    only integers out of the game state and the ``step`` return value.
    """
    import inspect

    src = "\n".join(inspect.getsource(f) for f in
                    (C4.share_ok, C4.pace_ok, C4.resolve_outcome4))
    for forbidden in ("hands.", "analyse", "best_type", "best_slots",
                      "teacher", "policy"):
        assert forbidden not in src, forbidden
    # and the signature carries only game quantities
    params = set(inspect.signature(C4.resolve_outcome4).parameters)
    assert params == {"required", "score_before", "needed", "plays", "p0",
                      "action", "info", "done", "reward_rule"}


# --------------------------------------------------------------------------- #
# the post-hoc capacity probe's pure helpers
# --------------------------------------------------------------------------- #
def test_capacity_by_bucket_groups_and_scores_play():
    from scripts import plast4_capacity as CAP

    table = {1: dict(bucket="lt0.5"), 2: dict(bucket="lt0.5"),
             3: dict(bucket="lt1.0")}
    out = CAP.by_bucket({1: -1.0, 2: +3.0, 3: -0.5}, table)
    assert out["lt0.5"]["n"] == 2
    assert out["lt0.5"]["mean"] == pytest.approx(1.0)
    assert out["lt0.5"]["frac_play"] == pytest.approx(0.5)   # drive >= 0 plays
    assert out["lt1.0"]["frac_play"] == 0.0
    # a drive of exactly zero counts as PLAY, matching greedy_action(p >= 0.5)
    assert CAP.by_bucket({3: 0.0}, table)["lt1.0"]["frac_play"] == 1.0


def test_capacity_crossings_reports_first_negative_pulse():
    from scripts import plast4_capacity as CAP

    arm = dict(trace=[
        dict(pulse=0, buckets=dict({"lt0.5": dict(mean=+1.0),
                                    "lt1.0": dict(mean=+1.0)})),
        dict(pulse=1, buckets=dict({"lt0.5": dict(mean=+0.5),
                                    "lt1.0": dict(mean=+1.0)})),
        dict(pulse=2, buckets=dict({"lt0.5": dict(mean=-0.1),
                                    "lt1.0": dict(mean=+1.0)})),
    ])
    c = CAP.crossings(arm)
    assert c["lt0.5"] == 2
    assert c["lt1.0"] is None
    assert c["ge1.0"] is None          # absent buckets never cross
