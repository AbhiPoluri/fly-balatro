"""Tests for the Balatro environment wrapper and the binary feature encoding."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pylatro  # noqa: E402

from flybalatro.env import (  # noqa: E402
    ACTION_NAMES,
    N_ACTIONS,
    BalatroEnv,
    IllegalActionError,
    RewardConfig,
)
from flybalatro.features import (  # noqa: E402
    FEATURE_NAMES,
    HAND_SLOTS,
    N_FEATURES,
    N_RANKS,
    N_SUITS,
    SELECTED_SLOTS,
    encode,
)

EPISODE_SEEDS: Tuple[int, ...] = (0, 1, 2, 7, 13, 42, 99, 1234)


def _random_rollout(
    env: BalatroEnv, seed: int, rng: np.random.Generator
) -> Tuple[int, bool]:
    """Play one episode with a uniform legal policy. Returns (steps, done)."""
    _features, mask = env.reset(seed=seed)
    while not env.done:
        legal = np.flatnonzero(mask)
        assert legal.size > 0, "unfinished episode with no legal actions"
        action = int(legal[rng.integers(legal.size)])
        _features, mask, _reward, _done, _info = env.step(action)
    return env.steps, env.done


# -- mask ---------------------------------------------------------------


def test_mask_length_is_always_109() -> None:
    env = BalatroEnv()
    rng = np.random.default_rng(0)
    for seed in EPISODE_SEEDS[:4]:
        _features, mask = env.reset(seed=seed)
        assert mask.shape == (N_ACTIONS,)
        while not env.done:
            legal = np.flatnonzero(mask)
            action = int(legal[rng.integers(legal.size)])
            _features, mask, _reward, _done, _info = env.step(action)
            assert mask.shape == (N_ACTIONS,)
            assert set(np.unique(mask).tolist()) <= {0, 1}


def test_raw_engine_mask_is_109_wide() -> None:
    config = pylatro.Config()
    config.ante_end = 1
    engine = pylatro.GameEngine(config)
    assert len(engine.gen_action_space()) == N_ACTIONS
    assert len(ACTION_NAMES) == N_ACTIONS


def test_n_actions_attribute() -> None:
    env = BalatroEnv()
    assert env.n_actions == 109
    assert len(env.action_names) == 109


# -- stepping -----------------------------------------------------------


def test_legal_actions_never_raise() -> None:
    env = BalatroEnv()
    rng = np.random.default_rng(1)
    for seed in EPISODE_SEEDS:
        steps, done = _random_rollout(env, seed, rng)
        assert done
        assert steps > 0


def test_every_legal_action_is_takeable_from_a_fresh_state() -> None:
    """Each unmasked index must be acceptable to the engine, one at a time."""
    env = BalatroEnv()
    rng = np.random.default_rng(2)
    # Walk a few steps to reach a mid-blind state, then try every legal index
    # from a state restored by replaying the same seed and prefix.
    prefix: List[int] = []
    _features, mask = env.reset(seed=5)
    for _ in range(6):
        legal = np.flatnonzero(mask)
        action = int(legal[rng.integers(legal.size)])
        prefix.append(action)
        _features, mask, _reward, done, _info = env.step(action)
        if done:
            break

    target_mask = mask.copy()
    for index in np.flatnonzero(target_mask).tolist():
        replay = BalatroEnv()
        _f, _m = replay.reset(seed=5)
        for action in prefix:
            _f, _m, _r, _d, _i = replay.step(action)
        _f, _m, _r, _d, _i = replay.step(int(index))


def test_episode_terminates() -> None:
    env = BalatroEnv(max_steps=2000)
    rng = np.random.default_rng(3)
    for seed in EPISODE_SEEDS:
        steps, _done = _random_rollout(env, seed, rng)
        assert env.done
        assert steps < 2000, "episode hit the step cap instead of terminating"


def test_illegal_action_raises() -> None:
    env = BalatroEnv()
    _features, mask = env.reset(seed=0)
    illegal = int(np.flatnonzero(mask == 0)[0])
    with pytest.raises(IllegalActionError):
        env.step(illegal)


def test_out_of_range_action_raises() -> None:
    env = BalatroEnv()
    env.reset(seed=0)
    with pytest.raises(IllegalActionError):
        env.step(N_ACTIONS)
    with pytest.raises(IllegalActionError):
        env.step(-1)


def test_step_before_reset_raises() -> None:
    env = BalatroEnv()
    with pytest.raises(RuntimeError):
        env.step(78)


def test_step_after_done_raises() -> None:
    env = BalatroEnv()
    rng = np.random.default_rng(4)
    _random_rollout(env, 0, rng)
    with pytest.raises(RuntimeError):
        env.step(78)


def test_same_seed_is_deterministic() -> None:
    env_a = BalatroEnv()
    env_b = BalatroEnv()
    features_a, mask_a = env_a.reset(seed=321)
    features_b, mask_b = env_b.reset(seed=321)
    assert np.array_equal(features_a, features_b)
    assert np.array_equal(mask_a, mask_b)
    for action in (78, 0, 1, 70):
        fa, ma, ra, da, ia = env_a.step(action)
        fb, mb, rb, db, ib = env_b.step(action)
        assert np.array_equal(fa, fb)
        assert np.array_equal(ma, mb)
        assert ra == rb and da == db and ia == ib


def test_different_seeds_give_different_hands() -> None:
    env = BalatroEnv()
    hands: Set[Tuple[int, ...]] = set()
    for seed in range(6):
        env.reset(seed=seed)
        env.step(78)  # select blind, which deals the hand
        state = env.engine.state
        hands.add(tuple(card.rank_index * 4 + card.suit_index for card in state.available))
    assert len(hands) > 1


# -- reward shaping -----------------------------------------------------


def test_reward_is_chips_over_100() -> None:
    env = BalatroEnv(reward=RewardConfig(chip_scale=1.0 / 100.0, win_bonus=20.0))
    rng = np.random.default_rng(5)
    _features, mask = env.reset(seed=11)
    while not env.done:
        legal = np.flatnonzero(mask)
        action = int(legal[rng.integers(legal.size)])
        _features, mask, reward, done, info = env.step(action)
        expected = int(info["chips_gained"]) / 100.0
        if done and bool(info["is_win"]):
            expected += 20.0
        assert reward == pytest.approx(expected)


def test_reward_shaping_is_configurable() -> None:
    env = BalatroEnv(
        reward=RewardConfig(
            chip_scale=1.0, win_bonus=0.0, lose_penalty=-1.0, step_penalty=-0.5
        )
    )
    _features, mask = env.reset(seed=0)
    _f, _m, reward, _d, info = env.step(78)
    assert reward == pytest.approx(int(info["chips_gained"]) - 0.5)


def test_chips_total_is_non_decreasing() -> None:
    env = BalatroEnv()
    rng = np.random.default_rng(6)
    _features, mask = env.reset(seed=23)
    previous = 0
    while not env.done:
        legal = np.flatnonzero(mask)
        action = int(legal[rng.integers(legal.size)])
        _features, mask, _reward, _done, info = env.step(action)
        total = int(info["chips_total"])
        assert total >= previous
        previous = total


# -- features -----------------------------------------------------------


def test_feature_vector_is_fixed_length_and_binary() -> None:
    env = BalatroEnv()
    rng = np.random.default_rng(7)
    for seed in EPISODE_SEEDS[:4]:
        features, mask = env.reset(seed=seed)
        assert features.shape == (N_FEATURES,)
        assert features.dtype == np.float32
        assert np.isin(features, (0.0, 1.0)).all()
        while not env.done:
            legal = np.flatnonzero(mask)
            action = int(legal[rng.integers(legal.size)])
            features, mask, _reward, _done, _info = env.step(action)
            assert features.shape == (N_FEATURES,)
            assert features.dtype == np.float32
            assert np.isin(features, (0.0, 1.0)).all()


def test_feature_names_match_length() -> None:
    assert len(FEATURE_NAMES) == N_FEATURES
    assert len(set(FEATURE_NAMES)) == N_FEATURES
    assert 150 <= N_FEATURES <= 300


def test_feature_blocks_are_one_hot_per_slot() -> None:
    env = BalatroEnv()
    env.reset(seed=17)
    env.step(78)  # deal the hand
    env.step(0)  # select hand slot 0
    env.step(2)  # select hand slot 2
    state = env.engine.state
    features = encode(state)

    card_bits = N_RANKS + N_SUITS + 1
    for slot in range(HAND_SLOTS):
        base = slot * card_bits
        occupied = features[base + N_RANKS + N_SUITS]
        rank_sum = features[base : base + N_RANKS].sum()
        suit_sum = features[base + N_RANKS : base + N_RANKS + N_SUITS].sum()
        if slot < len(state.available):
            assert occupied == 1.0
            assert rank_sum == 1.0
            assert suit_sum == 1.0
        else:
            assert occupied == 0.0
            assert rank_sum == 0.0
            assert suit_sum == 0.0

    selected_flags_base = HAND_SLOTS * card_bits
    flags = features[selected_flags_base : selected_flags_base + HAND_SLOTS]
    assert flags.sum() == len(state.selected)
    assert flags[0] == 1.0
    assert flags[2] == 1.0
    assert flags[1] == 0.0

    selected_base = selected_flags_base + HAND_SLOTS
    for slot in range(SELECTED_SLOTS):
        base = selected_base + slot * card_bits
        occupied = features[base + N_RANKS + N_SUITS]
        expected = 1.0 if slot < len(state.selected) else 0.0
        assert occupied == expected


def test_feature_cards_match_state() -> None:
    env = BalatroEnv()
    env.reset(seed=77)
    env.step(78)
    state = env.engine.state
    features = encode(state)
    card_bits = N_RANKS + N_SUITS + 1
    for slot, card in enumerate(state.available[:HAND_SLOTS]):
        base = slot * card_bits
        assert int(np.argmax(features[base : base + N_RANKS])) == card.rank_index
        suits = features[base + N_RANKS : base + N_RANKS + N_SUITS]
        assert int(np.argmax(suits)) == card.suit_index


def test_stage_block_is_one_hot() -> None:
    env = BalatroEnv()
    rng = np.random.default_rng(8)
    stage_offset = FEATURE_NAMES.index("stage_pre_blind")
    n_stages = sum(1 for name in FEATURE_NAMES if name.startswith("stage_"))
    _features, mask = env.reset(seed=31)
    seen: Set[int] = set()
    while not env.done:
        legal = np.flatnonzero(mask)
        action = int(legal[rng.integers(legal.size)])
        features, mask, _reward, _done, info = env.step(action)
        block = features[stage_offset : stage_offset + n_stages]
        assert block.sum() == 1.0
        index = int(np.argmax(block))
        assert index == int(info["stage"])
        seen.add(index)
    assert len(seen) >= 2


# -- action semantics ---------------------------------------------------


def _action_kind(action: object) -> str:
    """pyo3 complex enum variants surface as classes named Action_<Variant>."""
    return type(action).__name__.removeprefix("Action_")


def _field_key(value: object) -> str:
    """Structural key for one payload field of an ``Action`` variant.

    ``Action`` is a pyo3 complex enum with no ``__repr__`` exposed, so
    ``str()`` on a variant gives an object address and cannot be compared.
    Cards are keyed by their unique ``id``; simple enums (MoveDirection,
    SortBy, Blind) stringify usefully; anything that still stringifies to an
    object address falls back to its type name.
    """
    card_id = getattr(value, "id", None)
    if isinstance(card_id, int):
        return f"card:{card_id}"
    text = str(value)
    if " object at 0x" in text:
        return f"type:{type(value).__name__}"
    return text


def _action_key(action: object) -> Tuple[str, ...]:
    """Comparable key for an ``Action``: variant name plus its payload."""
    key: List[str] = [_action_kind(action)]
    for attr in ("_0", "_1"):
        if hasattr(action, attr):
            key.append(_field_key(getattr(action, attr)))
    return tuple(key)


def test_action_index_semantics_match_action_names() -> None:
    """Compare every legal index against the engine's own resolution of it."""
    env = BalatroEnv()
    rng = np.random.default_rng(9)
    checked = 0
    for seed in (0, 3, 8):
        _features, mask = env.reset(seed=seed)
        while not env.done:
            state = env.engine.state
            available = state.available
            for index in np.flatnonzero(mask).tolist():
                action = env.engine.to_action(int(index))
                kind = _action_kind(action)
                name = ACTION_NAMES[index]
                checked += 1
                if name.startswith("select_card["):
                    slot = int(name[len("select_card[") : -1])
                    assert kind == "SelectCard"
                    assert action._0.id == available[slot].id
                elif name.startswith("move_card_left["):
                    slot = int(name[len("move_card_left[") : -1])
                    assert kind == "MoveCard"
                    assert str(action._0) == "MoveDirection.Left"
                    assert action._1.id == available[slot].id
                elif name.startswith("move_card_right["):
                    slot = int(name[len("move_card_right[") : -1])
                    assert kind == "MoveCard"
                    assert str(action._0) == "MoveDirection.Right"
                    assert action._1.id == available[slot].id
                elif name == "play":
                    assert kind == "Play"
                elif name == "discard":
                    assert kind == "Discard"
                elif name == "cash_out":
                    assert kind == "CashOut"
                elif name == "next_round":
                    assert kind == "NextRound"
                elif name == "select_blind":
                    assert kind == "SelectBlind"
                elif name == "skip_blind":
                    assert kind == "SkipBlind"
                elif name == "sort_hand[rank]":
                    assert kind == "SortHand"
                    assert str(action._0) == "SortBy.Rank"
                elif name == "sort_hand[suit]":
                    assert kind == "SortHand"
                    assert str(action._0) == "SortBy.Suit"
                elif name.startswith("buy_joker["):
                    assert kind == "BuyJoker"
                elif name.startswith("buy_consumable["):
                    assert kind == "BuyConsumable"
                elif name == "buy_voucher":
                    assert kind == "BuyVoucher"
                elif name.startswith("buy_playing_card["):
                    assert kind == "BuyPlayingCard"
                elif name.startswith("use_consumable["):
                    assert kind == "UseConsumable"
                elif name.startswith("sell_joker["):
                    assert kind == "SellJoker"
                elif name.startswith("sell_consumable["):
                    assert kind == "SellConsumable"
                elif name.startswith("buy_pack["):
                    assert kind == "BuyPack"
                elif name.startswith("pick_pack_card["):
                    assert kind == "PickPackCard"
                elif name == "skip_pack":
                    assert kind == "SkipPack"
                elif name == "reroll":
                    assert kind == "Reroll"
                elif name == "apply_tarot":
                    assert kind == "ApplyTarot"
                elif name == "apply_spectral":
                    assert kind == "ApplySpectral"
                else:  # pragma: no cover - guards against an unnamed index
                    raise AssertionError(f"unhandled action name {name!r}")
            legal = np.flatnonzero(mask)
            action_index = int(legal[rng.integers(legal.size)])
            _features, mask, _reward, _done, _info = env.step(action_index)
    assert checked > 200


def test_masked_indices_are_a_subset_of_gen_actions() -> None:
    """Every masked-legal action must appear in the engine's action list.

    The converse does not hold: ``gen_actions()`` emits ``DeselectCard``,
    which has no index in the 109-vector at all.
    """
    env = BalatroEnv()
    rng = np.random.default_rng(10)
    for seed in (1, 4):
        _features, mask = env.reset(seed=seed)
        while not env.done:
            listed = {_action_key(a) for a in env.engine.gen_actions()}
            for index in np.flatnonzero(mask).tolist():
                assert _action_key(env.engine.to_action(int(index))) in listed
            legal = np.flatnonzero(mask)
            _features, mask, _reward, _done, _info = env.step(
                int(legal[rng.integers(legal.size)])
            )


def test_mask_and_gen_actions_agree_on_count_modulo_deselect() -> None:
    env = BalatroEnv()
    _features, mask = env.reset(seed=2)
    assert int(mask.sum()) == len(env.engine.gen_actions())
    _features, mask, _r, _d, _i = env.step(78)
    assert int(mask.sum()) == len(env.engine.gen_actions())
    # Selecting a card adds a DeselectCard to gen_actions() with no index.
    _features, mask, _r, _d, _i = env.step(0)
    n_deselect = sum(
        1 for a in env.engine.gen_actions() if _action_kind(a) == "DeselectCard"
    )
    assert n_deselect == 1
    assert int(mask.sum()) == len(env.engine.gen_actions()) - n_deselect


def test_noop_masking_removes_move_and_sort() -> None:
    env = BalatroEnv(mask_noop_actions=True)
    env.reset(seed=0)
    _features, mask, _r, _d, _i = env.step(78)
    legal_names = [ACTION_NAMES[i] for i in np.flatnonzero(mask).tolist()]
    assert all(not n.startswith("move_card_") for n in legal_names)
    assert all(not n.startswith("sort_hand") for n in legal_names)
    assert any(n.startswith("select_card[") for n in legal_names)
