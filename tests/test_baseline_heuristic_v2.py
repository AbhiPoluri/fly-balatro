"""Teacher-v2 tests: the plan it makes, and the no-deselect repair rules.

The repair branches only fire on selections the teacher would never make itself
(random collection rollouts and learned readouts make them), so they are not
covered by just running episodes - hence the direct ``plan`` tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baseline_heuristic_v2 import HandAwarePolicy  # noqa: E402
from flybalatro import hands  # noqa: E402
from flybalatro.env import DISCARD_INDEX, PLAY_INDEX, BalatroEnv  # noqa: E402
from flybalatro.features_v2 import encode_v2, N_FEATURES_V2  # noqa: E402
from tests.test_features_v2 import FakeState  # noqa: E402
from tests.test_hands import hand  # noqa: E402


def plan(spec: str, selected: Sequence[int] = (), **kw):
    policy = HandAwarePolicy(play_share=kw.pop("play_share", 1.0))
    return policy.plan(FakeState(hand(spec), selected=selected, **kw))


# -- the main line ----------------------------------------------------------


def test_plays_the_best_subset_when_it_covers_its_share() -> None:
    # Full house for 400 against 300 still needed: play it.
    target, action = plan("Kh Kd Kc Qh Qd 2s 3s 4s", required_score=300)
    best = hands.best_subset(hand("Kh Kd Kc Qh Qd 2s 3s 4s"))
    assert best is not None
    assert action == PLAY_INDEX
    assert target == best.slots


def test_digs_when_the_best_hand_is_far_short_and_discards_remain() -> None:
    # Pair of twos (28) against 1200 still needed with 4 plays: dig.
    spec = "2h 2d 4s 5c 7h 9d Js Qc"
    target, action = plan(spec, required_score=1200, plays=4, discards=3)
    assert action == DISCARD_INDEX
    best = hands.best_subset(hand(spec))
    assert best is not None
    assert not (set(target) & set(best.slots)), "digging must keep the best cards"
    assert len(target) == hands.MAX_SELECTED


def test_plays_rather_than_digs_when_out_of_discards() -> None:
    spec = "2h 2d 4s 5c 7h 9d Js Qc"
    target, action = plan(spec, required_score=1200, discards=0)
    assert action == PLAY_INDEX
    best = hands.best_subset(hand(spec))
    assert best is not None and target == best.slots


def test_fair_share_scales_with_plays_left() -> None:
    spec = "Th Td 4s 5c 7h 9d Js Qc"  # pair of tens, (10+20)*2 = 60
    # One play left and 300 to go: 60 is nowhere near enough, dig.
    assert plan(spec, required_score=300, plays=1, discards=2)[1] == DISCARD_INDEX
    # Same hand against 60 still needed: it clears, play it.
    assert plan(spec, required_score=60, plays=1, discards=2)[1] == PLAY_INDEX


# -- the no-deselect repair rules -------------------------------------------


def test_partial_best_selection_is_finished_and_played() -> None:
    """Cards already committed to the best hand are never discarded."""
    spec = "2h 2d 4s 5c 7h 9d Js Qc"  # digging territory when nothing is selected
    best = hands.best_subset(hand(spec))
    assert best is not None
    assert plan(spec, required_score=1200)[1] == DISCARD_INDEX
    for partial in (best.slots[:1], best.slots):
        target, action = plan(spec, selected=partial, required_score=1200)
        assert action == PLAY_INDEX, f"{partial} is part of the best hand"
        assert target == best.slots


def test_junk_only_selection_is_discarded_away() -> None:
    """Only when the compromised hand is not worth a play.

    With slots 5 and 6 (a 2 and a 3) stuck in the selection, the best hand that
    still contains them is three kings for 180. Against 300 that is well over a
    quarter of the blind, so playing it is right; against 3,000 it is not, and
    the junk goes out with a discard instead.
    """
    spec = "Kh Kd Kc Qh Qd 2s 3s 4s"   # best hand is slots 0-4
    assert plan(spec, selected=(5, 6), required_score=300, discards=2)[1] == PLAY_INDEX

    target, action = plan(spec, selected=(5, 6), required_score=3000, discards=2)
    assert action == DISCARD_INDEX
    assert {5, 6} <= set(target)
    assert not (set(target) & {0, 1, 2, 3, 4})


def test_junk_selection_with_no_discards_left_plays_the_best_it_can() -> None:
    spec = "Kh Kd Kc Qh Qd 2s 3s 4s"
    target, action = plan(spec, selected=(5, 6), required_score=300, discards=0)
    assert action == PLAY_INDEX
    assert {5, 6} <= set(target)
    constrained = hands.best_subset(hand(spec), must_include=(5, 6))
    assert constrained is not None and target == constrained.slots


def test_mixed_selection_plays_the_best_hand_containing_it() -> None:
    spec = "Kh Kd Kc Qh Qd 2s 3s 4s"
    target, action = plan(spec, selected=(0, 5), required_score=3000, discards=2)
    assert action == PLAY_INDEX
    assert {0, 5} <= set(target)


def test_full_selection_only_plays_or_discards() -> None:
    spec = "Kh Kd Kc Qh Qd 2s 3s 4s"
    target, action = plan(spec, selected=(3, 4, 5, 6, 7), required_score=1200, discards=2)
    assert set(target) == {3, 4, 5, 6, 7}
    assert action in (PLAY_INDEX, DISCARD_INDEX)


# -- behaviour in the real engine -------------------------------------------


def test_policy_is_legal_and_plays_real_hands_in_the_engine() -> None:
    """Over real episodes: every action legal, and every play is the best subset."""
    policy = HandAwarePolicy()
    env = BalatroEnv(ante_end=1, max_steps=500, mask_noop_actions=True,
                     encoder=encode_v2, n_features=N_FEATURES_V2)
    from flybalatro import features_v2 as F2

    n_plays = 0
    n_plays_best = 0
    types = set()
    wins = 0
    for seed in range(12):
        bits, mask = env.reset(seed=seed)
        while not env.done:
            action = policy(env, mask)
            assert mask[action] == 1, f"illegal action {action}"
            assert action != 79, "the teacher must never skip a blind"
            if action == PLAY_INDEX:
                n_plays += 1
                if bits[F2.OFF_IS_BEST] > 0:
                    n_plays_best += 1
                sel = bits[F2.OFF_SELECTED_TYPE:F2.OFF_SELECTED_TYPE + F2.N_SELECTED_TYPE]
                types.add(int(np.argmax(sel)))
            bits, mask, _r, _done, _info = env.step(action)
        wins += int(env.is_win)

    assert n_plays > 20
    assert n_plays_best == n_plays, "the teacher only ever plays the best subset"
    assert len(types) >= 4, f"expected several hand types, saw {sorted(types)}"
    assert wins >= 4, f"expected to clear ante 1 often, cleared {wins}/12"
