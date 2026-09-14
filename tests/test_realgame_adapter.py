"""Adapter tests: synthetic BalatroBot payloads -> exact 283-bit expectations.

The expectations are hand-computed from the layout documented in
``flybalatro/features.py``, not read back out of the adapter, so a change to
either side has to be made deliberately in both places.

Bit layout being asserted against:

    0..143    8 hand slots x 18 bits (rank 13 + suit 4 + occupied 1)
    144..151  hand_selected, one bit per hand slot
    152..241  5 selected slots x 18 bits
    242..246  plays remaining one-hot (0,1,2,3,>=4)
    247..251  discards remaining one-hot
    252..259  score/required ratio buckets
    260..265  money buckets (0, 1-2, 3-5, 6-9, 10-19, >=20)
    266..276  stage one-hot
    277..282  joker count one-hot (0,1,2,3,4,>=5)
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Set

import numpy as np
import pytest

from flybalatro.features import N_FEATURES, encode
from flybalatro.realgame.adapter import (
    STAGE_BLIND_BIG,
    STAGE_BLIND_BOSS,
    STAGE_BLIND_SMALL,
    STAGE_END_LOSE,
    STAGE_END_WIN,
    STAGE_PACK_OPEN,
    STAGE_POST_BLIND,
    STAGE_PRE_BLIND,
    STAGE_SHOP,
    ActionPlan,
    RealCard,
    UnmappableStateError,
    build_state,
    legality_mask,
    plan_action,
)

# -- layout constants, restated independently of the implementation ---------

CARD_BITS = 18
OFF_HAND = 0
OFF_HAND_SELECTED = 144
OFF_SELECTED = 152
OFF_PLAYS = 242
OFF_DISCARDS = 247
OFF_RATIO = 252
OFF_MONEY = 260
OFF_STAGE = 266
OFF_JOKERS = 277

RANKS = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7,
         "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
SUITS = {"S": 0, "C": 1, "H": 2, "D": 3}


# -- payload builders -------------------------------------------------------


def card(card_id: int, rank: str, suit: str, *, highlight: bool = False,
         buy: int = 0, sell: int = 1) -> Dict[str, object]:
    """A BalatroBot playing-card payload, shaped per docs/api.md."""
    return {
        "id": card_id,
        "key": f"{suit}_{rank}",
        "set": "DEFAULT",
        "label": f"{rank} of {suit}",
        "value": {"suit": suit, "rank": rank, "effect": ""},
        "modifier": {"seal": None, "edition": None, "enhancement": None,
                     "eternal": False, "perishable": None, "rental": False},
        "state": {"debuff": False, "hidden": False, "highlight": highlight},
        "cost": {"sell": sell, "buy": buy},
    }


def special(card_id: int, key: str, card_set: str, *, buy: int = 0,
            sell: int = 1, label: str = "") -> Dict[str, object]:
    """A non-playing card (joker, consumable, voucher, pack)."""
    return {
        "id": card_id,
        "key": key,
        "set": card_set,
        "label": label or key,
        "value": {"effect": ""},
        "modifier": {"seal": None, "edition": None, "enhancement": None,
                     "eternal": False, "perishable": None, "rental": False},
        "state": {"debuff": False, "hidden": False, "highlight": False},
        "cost": {"sell": sell, "buy": buy},
    }


def area(cards: Sequence[Mapping[str, object]], limit: int = 8,
         highlighted_limit: Optional[int] = 5) -> Dict[str, object]:
    out: Dict[str, object] = {"count": len(cards), "limit": limit,
                              "cards": list(cards)}
    if highlighted_limit is not None:
        out["highlighted_limit"] = highlighted_limit
    return out


def blind(blind_type: str, status: str, score: int) -> Dict[str, object]:
    return {"type": blind_type, "status": status, "name": f"{blind_type} Blind",
            "effect": "", "score": score}


def payload(
    *,
    state: str = "SELECTING_HAND",
    hand: Sequence[Mapping[str, object]] = (),
    money: int = 4,
    hands_left: int = 4,
    discards_left: int = 3,
    chips: int = 0,
    reroll_cost: int = 5,
    current_blind: str = "SMALL",
    blind_score: int = 300,
    jokers: Sequence[Mapping[str, object]] = (),
    consumables: Sequence[Mapping[str, object]] = (),
    shop: Sequence[Mapping[str, object]] = (),
    vouchers: Sequence[Mapping[str, object]] = (),
    packs: Sequence[Mapping[str, object]] = (),
    pack: Sequence[Mapping[str, object]] = (),
    won: bool = False,
    round_num: int = 1,
    ante_num: int = 1,
    highlighted_limit: Optional[int] = 5,
) -> Dict[str, object]:
    """A full BalatroBot gamestate payload."""
    status = "CURRENT" if state != "BLIND_SELECT" else "SELECT"
    blinds: Dict[str, object] = {
        "small": blind("SMALL", "UPCOMING", 300),
        "big": blind("BIG", "UPCOMING", 450),
        "boss": blind("BOSS", "UPCOMING", 600),
    }
    key = {"SMALL": "small", "BIG": "big", "BOSS": "boss"}[current_blind]
    blinds[key] = blind(current_blind, status, blind_score)
    return {
        "state": state,
        "round_num": round_num,
        "ante_num": ante_num,
        "money": money,
        "deck": "RED",
        "stake": "WHITE",
        "seed": "TEST123",
        "won": won,
        "used_vouchers": {},
        "hands": {},
        "round": {"hands_left": hands_left, "hands_played": 0,
                  "discards_left": discards_left, "discards_used": 0,
                  "reroll_cost": reroll_cost, "chips": chips},
        "blinds": blinds,
        "jokers": area(jokers, limit=5, highlighted_limit=None),
        "consumables": area(consumables, limit=2, highlighted_limit=None),
        "hand": area(hand, limit=8, highlighted_limit=highlighted_limit),
        "cards": area([], limit=52, highlighted_limit=None),
        "shop": area(shop, limit=4, highlighted_limit=None),
        "vouchers": area(vouchers, limit=1, highlighted_limit=None),
        "packs": area(packs, limit=2, highlighted_limit=None),
        "pack": area(pack, limit=5, highlighted_limit=None),
    }


def card_slot_bits(base: int, rank: str, suit: str) -> Set[int]:
    """The three bits one occupied card slot sets."""
    return {base + RANKS[rank], base + 13 + SUITS[suit], base + 17}


# -- the headline test: identical hands must give identical bits ------------


def test_encoded_bits_match_hand_computed_expectation() -> None:
    """A 5-card hand with 2 selected, checked bit by bit."""
    hand = [
        card(1, "A", "S"),
        card(2, "T", "H"),
        card(3, "7", "D"),
        card(4, "K", "C"),
        card(5, "2", "H"),
    ]
    state = build_state(payload(
        hand=hand, money=4, hands_left=3, discards_left=2,
        chips=150, blind_score=300, current_blind="SMALL",
    ), selected_indices=[1, 3])

    bits = encode(state)
    assert bits.shape == (N_FEATURES,)
    assert set(np.unique(bits)).issubset({0.0, 1.0})

    expected: Set[int] = set()
    # Hand slots 0..4 occupied, 5..7 empty.
    for slot, (rank, suit) in enumerate(
        [("A", "S"), ("T", "H"), ("7", "D"), ("K", "C"), ("2", "H")]
    ):
        expected |= card_slot_bits(OFF_HAND + slot * CARD_BITS, rank, suit)
    # hand_selected: slots 1 and 3.
    expected |= {OFF_HAND_SELECTED + 1, OFF_HAND_SELECTED + 3}
    # selected block, in selection order: T of H then K of C.
    expected |= card_slot_bits(OFF_SELECTED + 0 * CARD_BITS, "T", "H")
    expected |= card_slot_bits(OFF_SELECTED + 1 * CARD_BITS, "K", "C")
    # plays 3 -> bin 3; discards 2 -> bin 2.
    expected.add(OFF_PLAYS + 3)
    expected.add(OFF_DISCARDS + 2)
    # 150/300 = 0.5 -> first edge not satisfied by <0.5, so bucket for <0.75 = 4.
    expected.add(OFF_RATIO + 4)
    # money 4 -> bucket 3-5 = index 2.
    expected.add(OFF_MONEY + 2)
    # stage blind_small = 1.
    expected.add(OFF_STAGE + STAGE_BLIND_SMALL)
    # 0 jokers.
    expected.add(OFF_JOKERS + 0)

    assert set(np.flatnonzero(bits).tolist()) == expected


def test_selection_order_drives_the_selected_block() -> None:
    """The `selected` block is positional in selection order, not hand order."""
    hand = [card(1, "A", "S"), card(2, "T", "H"), card(3, "7", "D")]
    base = payload(hand=hand)
    forward = np.flatnonzero(encode(build_state(base, selected_indices=[0, 2])))
    reverse = np.flatnonzero(encode(build_state(base, selected_indices=[2, 0])))

    # Same hand bits and same hand_selected bits...
    assert set(forward.tolist()) & set(range(OFF_SELECTED)) == \
        set(reverse.tolist()) & set(range(OFF_SELECTED))
    # ...but the selected block differs.
    assert set(forward.tolist()) & set(range(OFF_SELECTED, OFF_PLAYS)) != \
        set(reverse.tolist()) & set(range(OFF_SELECTED, OFF_PLAYS))


def test_every_rank_and_suit_maps_to_the_right_bit() -> None:
    """Exhaustive check of the rank/suit tables, including T and the faces."""
    for rank, rank_index in RANKS.items():
        for suit, suit_index in SUITS.items():
            state = build_state(payload(hand=[card(1, rank, suit)]))
            bits = np.flatnonzero(encode(state)).tolist()
            assert OFF_HAND + rank_index in bits, f"{rank}{suit} rank bit"
            assert OFF_HAND + 13 + suit_index in bits, f"{rank}{suit} suit bit"
            assert OFF_HAND + 17 in bits


def test_hand_truncated_to_eight_slots() -> None:
    """A 10-card hand (Paint Brush etc.) fills only the 8 encodable slots."""
    hand = [card(i + 1, "2", "S") for i in range(10)]
    bits = np.flatnonzero(encode(build_state(payload(hand=hand)))).tolist()
    assert OFF_HAND + 7 * CARD_BITS + 17 in bits  # slot 7 occupied
    assert max(b for b in bits if b < OFF_HAND_SELECTED) < OFF_HAND_SELECTED


def test_empty_hand_sets_no_card_bits() -> None:
    bits = np.flatnonzero(encode(build_state(payload(hand=[], state="SHOP")))).tolist()
    assert all(b >= OFF_PLAYS for b in bits)


def test_money_and_count_buckets() -> None:
    cases = {0: 0, 1: 1, 2: 1, 3: 2, 5: 2, 6: 3, 9: 3, 10: 4, 19: 4, 20: 5, 99: 5}
    for money, bucket in cases.items():
        state = build_state(payload(hand=[card(1, "A", "S")], money=money))
        bits = np.flatnonzero(encode(state)).tolist()
        assert OFF_MONEY + bucket in bits, f"money {money}"

    # Counters saturate at the ">=4" bin.
    for plays, bucket in {0: 0, 1: 1, 4: 4, 9: 4}.items():
        state = build_state(payload(hand=[card(1, "A", "S")], hands_left=plays))
        assert OFF_PLAYS + bucket in np.flatnonzero(encode(state)).tolist()

    # Joker count saturates at ">=5".
    for n, bucket in {0: 0, 1: 1, 5: 5, 7: 5}.items():
        jokers = [special(100 + i, "j_joker", "JOKER") for i in range(n)]
        state = build_state(payload(hand=[card(1, "A", "S")], jokers=jokers))
        assert OFF_JOKERS + bucket in np.flatnonzero(encode(state)).tolist()


def test_score_ratio_buckets() -> None:
    cases = [(0, 300, 0), (1, 300, 1), (50, 300, 2), (100, 300, 3),
             (150, 300, 4), (250, 300, 5), (300, 300, 6), (600, 300, 7)]
    for chips, required, bucket in cases:
        state = build_state(payload(hand=[card(1, "A", "S")], chips=chips,
                                    blind_score=required))
        bits = np.flatnonzero(encode(state)).tolist()
        assert OFF_RATIO + bucket in bits, f"{chips}/{required}"


# -- stage mapping ----------------------------------------------------------


def test_stage_mapping() -> None:
    cases = [
        (dict(state="BLIND_SELECT"), STAGE_PRE_BLIND),
        (dict(state="SELECTING_HAND", current_blind="SMALL"), STAGE_BLIND_SMALL),
        (dict(state="SELECTING_HAND", current_blind="BIG"), STAGE_BLIND_BIG),
        (dict(state="SELECTING_HAND", current_blind="BOSS"), STAGE_BLIND_BOSS),
        (dict(state="HAND_PLAYED", current_blind="SMALL"), STAGE_BLIND_SMALL),
        (dict(state="DRAW_TO_HAND", current_blind="BIG"), STAGE_BLIND_BIG),
        (dict(state="ROUND_EVAL"), STAGE_POST_BLIND),
        (dict(state="SHOP"), STAGE_SHOP),
        (dict(state="GAME_OVER", won=True), STAGE_END_WIN),
        (dict(state="GAME_OVER", won=False), STAGE_END_LOSE),
        (dict(state="SMODS_BOOSTER_OPENED"), STAGE_PACK_OPEN),
        (dict(state="STANDARD_PACK"), STAGE_PACK_OPEN),
    ]
    for kwargs, expected in cases:
        state = build_state(payload(**kwargs))
        assert state.stage.int() == expected, kwargs
        # And the stage bit is the only one set in that block.
        bits = np.flatnonzero(encode(state)).tolist()
        in_block = [b for b in bits if OFF_STAGE <= b < OFF_JOKERS]
        assert in_block == [OFF_STAGE + expected]


def test_non_run_states_are_rejected() -> None:
    for name in ("MENU", "TUTORIAL", "SPLASH", "UNKNOWN"):
        with pytest.raises(UnmappableStateError):
            build_state(payload(state=name))


def test_malformed_payload_raises() -> None:
    bad = payload(hand=[card(1, "A", "S")])
    hand_area = bad["hand"]
    assert isinstance(hand_area, dict)
    cards = hand_area["cards"]
    assert isinstance(cards, list)
    first = cards[0]
    assert isinstance(first, dict)
    value = first["value"]
    assert isinstance(value, dict)
    value["rank"] = "11"  # not a Balatro rank
    with pytest.raises(ValueError):
        build_state(bad)


# -- selection bookkeeping --------------------------------------------------


def test_selection_respects_highlighted_limit_and_toggles() -> None:
    hand = [card(i + 1, "2", "S") for i in range(8)]
    state = build_state(payload(hand=hand))
    for i in range(6):
        state = state.toggle_selection(i)
    assert state.selected_indices == (0, 1, 2, 3, 4)  # capped at 5
    state = state.toggle_selection(2)
    assert state.selected_indices == (0, 1, 3, 4)
    state = state.toggle_selection(7)
    assert state.selected_indices == (0, 1, 3, 4, 7)


def test_stale_selection_indices_are_dropped_after_refill() -> None:
    """A selection pointing past the new hand must not leak into the bits."""
    small = payload(hand=[card(1, "A", "S"), card(2, "K", "H")])
    state = build_state(small, selected_indices=[0, 5, 9])
    assert state.selected_indices == (0,)


def test_highlight_seeds_selection_when_nothing_carried_over() -> None:
    hand = [card(1, "A", "S"), card(2, "K", "H", highlight=True)]
    state = build_state(payload(hand=hand))
    assert state.selected_indices == (1,)


def test_missing_highlighted_limit_falls_back_to_five() -> None:
    hand = [card(i + 1, "2", "S") for i in range(8)]
    state = build_state(payload(hand=hand, highlighted_limit=None))
    assert state.selected_max == 5


# -- legality mask ----------------------------------------------------------


def test_mask_masks_noop_actions_everywhere() -> None:
    states = [
        payload(state="BLIND_SELECT"),
        payload(hand=[card(1, "A", "S")]),
        payload(state="ROUND_EVAL"),
        payload(state="SHOP"),
    ]
    for raw in states:
        mask = legality_mask(build_state(raw))
        assert mask.shape == (109,)
        assert mask[24:70].sum() == 0, "move_card_* must be masked"
        assert mask[105] == 0 and mask[106] == 0, "sort_hand must be masked"
        assert mask[89] == 0 and mask[108] == 0, "apply_* have no mapping"


def test_mask_pre_blind() -> None:
    mask = legality_mask(build_state(payload(state="BLIND_SELECT")))
    assert mask[78] == 1  # select_blind
    assert mask[79] == 1  # skip_blind
    assert mask.sum() == 2

    boss = legality_mask(build_state(payload(state="BLIND_SELECT",
                                             current_blind="BOSS")))
    assert boss[78] == 1
    assert boss[79] == 0, "a boss blind cannot be skipped"


def test_mask_during_blind() -> None:
    hand = [card(i + 1, "2", "S") for i in range(5)]
    base = payload(hand=hand, hands_left=2, discards_left=1)

    empty = legality_mask(build_state(base))
    assert list(np.flatnonzero(empty[:24])) == [0, 1, 2, 3, 4]
    assert empty[70] == 0 and empty[71] == 0, "nothing selected -> no play/discard"

    chosen = legality_mask(build_state(base, selected_indices=[0, 1]))
    assert chosen[70] == 1 and chosen[71] == 1
    assert list(np.flatnonzero(chosen[:24])) == [2, 3, 4], (
        "an already-selected slot leaves the action space; pylatro has no deselect"
    )

    # A full selection offers no select_card at all -- nothing to extend, and
    # nothing to toggle off.
    full = legality_mask(build_state(base, selected_indices=[0, 1, 2, 3, 4]))
    assert list(np.flatnonzero(full[:24])) == []
    seven = payload(hand=[card(i + 1, "2", "S") for i in range(7)])
    full7 = legality_mask(build_state(seven, selected_indices=[0, 1, 2, 3, 4]))
    assert list(np.flatnonzero(full7[:24])) == [], "full selection, no room for 5,6"


def test_mask_respects_exhausted_counters() -> None:
    hand = [card(1, "A", "S")]
    no_discards = legality_mask(
        build_state(payload(hand=hand, discards_left=0), selected_indices=[0])
    )
    assert no_discards[70] == 1 and no_discards[71] == 0

    no_plays = legality_mask(
        build_state(payload(hand=hand, hands_left=0), selected_indices=[0])
    )
    assert no_plays[70] == 0 and no_plays[71] == 1


def test_mask_post_blind_and_terminal() -> None:
    post = legality_mask(build_state(payload(state="ROUND_EVAL")))
    assert post[72] == 1 and post.sum() == 1  # cash_out only

    for won in (True, False):
        over = legality_mask(build_state(payload(state="GAME_OVER", won=won)))
        assert over.sum() == 0


def test_mask_shop_affordability() -> None:
    shop = [
        special(10, "j_joker", "JOKER", buy=4, label="Joker"),
        special(11, "c_fool", "TAROT", buy=3, label="The Fool"),
        special(12, "j_sly", "JOKER", buy=9, label="Sly Joker"),
    ]
    state = build_state(payload(state="SHOP", money=5, shop=shop, reroll_cost=5))
    mask = legality_mask(state)

    assert mask[77] == 1, "next_round must always be legal in the shop"
    assert mask[73] == 1, "first joker costs 4 of 5 -> affordable"
    assert mask[74] == 0, "second joker costs 9 -> not affordable"
    assert mask[80] == 1, "the tarot costs 3 -> affordable"
    assert mask[107] == 1, "reroll costs exactly 5 -> affordable"

    broke = legality_mask(build_state(payload(state="SHOP", money=0, shop=shop)))
    assert broke[77] == 1
    assert broke[73] == 0 and broke[80] == 0 and broke[107] == 0


def test_mask_shop_joker_slots_full() -> None:
    shop = [special(10, "j_joker", "JOKER", buy=1)]
    jokers = [special(200 + i, "j_joker", "JOKER") for i in range(5)]
    mask = legality_mask(build_state(
        payload(state="SHOP", money=50, shop=shop, jokers=jokers)
    ))
    assert mask[73] == 0, "no free joker slot"
    assert mask[90] == 1, "but held jokers can be sold"


def test_mask_pack_open() -> None:
    pack = [special(30 + i, "c_fool", "TAROT") for i in range(3)]
    mask = legality_mask(build_state(payload(state="SMODS_BOOSTER_OPENED", pack=pack)))
    assert list(np.flatnonzero(mask[99:104])) == [0, 1, 2]
    assert mask[104] == 1  # skip_pack


# -- action routing ---------------------------------------------------------


def test_select_card_is_local_and_toggles() -> None:
    hand = [card(1, "A", "S"), card(2, "K", "H")]
    state = build_state(payload(hand=hand))

    plan = plan_action(state, 1)
    assert plan.kind == "local"
    assert plan.client_method == ""
    assert plan.new_selection == (1,)

    selected = state.with_selection([1])
    off = plan_action(selected, 1)
    assert off.new_selection == ()
    assert "deselect" in off.description


def test_play_and_discard_submit_sorted_indices() -> None:
    hand = [card(i + 1, "2", "S") for i in range(5)]
    state = build_state(payload(hand=hand), selected_indices=[3, 0, 2])

    play = plan_action(state, 70)
    assert play.kind == "rpc"
    assert play.client_method == "play"
    assert play.kwargs == {"cards": [0, 2, 3]}
    assert play.clears_selection

    discard = plan_action(state, 71)
    assert discard.client_method == "discard"
    assert discard.kwargs == {"cards": [0, 2, 3]}


def test_play_without_selection_is_an_error() -> None:
    state = build_state(payload(hand=[card(1, "A", "S")]))
    with pytest.raises(ValueError):
        plan_action(state, 70)


def test_stage_transition_actions() -> None:
    cases = {78: "select", 79: "skip", 72: "cash_out", 77: "next_round"}
    state = build_state(payload(state="SHOP"))
    for index, method in cases.items():
        plan = plan_action(state, index)
        assert plan.client_method == method
        assert plan.kwargs == {}


def test_buy_joker_maps_category_slot_to_shop_index() -> None:
    """buy_joker[1] is the *second joker*, which may be any shop index."""
    shop = [
        special(10, "c_fool", "TAROT", buy=3),
        special(11, "j_joker", "JOKER", buy=4),
        special(12, "c_venus", "PLANET", buy=3),
        special(13, "j_sly", "JOKER", buy=5),
    ]
    state = build_state(payload(state="SHOP", money=20, shop=shop))

    assert plan_action(state, 73).kwargs == {"index": 1}
    assert plan_action(state, 74).kwargs == {"index": 3}
    assert plan_action(state, 80).kwargs == {"index": 0}  # first consumable
    assert plan_action(state, 81).kwargs == {"index": 2}  # second consumable
    for plan_index in (73, 74, 80, 81):
        assert plan_action(state, plan_index).client_method == "buy_card"

    with pytest.raises(ValueError):
        plan_action(state, 75)  # only two jokers in the shop


def test_noop_and_unmappable_actions_raise() -> None:
    state = build_state(payload(hand=[card(1, "A", "S")]))
    for index in (24, 47, 69, 105, 106):
        with pytest.raises(ValueError, match="noop"):
            plan_action(state, index)
    for index in (89, 108):
        with pytest.raises(ValueError, match="no BalatroBot equivalent"):
            plan_action(state, index)
    for index in (-1, 109):
        with pytest.raises(ValueError, match="out of range"):
            plan_action(state, index)


def test_every_legal_action_has_a_plan() -> None:
    """The mask must never offer an action that plan_action refuses.

    This is the invariant that keeps the loop from crashing mid-demo.
    """
    scenarios = [
        payload(state="BLIND_SELECT"),
        payload(state="BLIND_SELECT", current_blind="BOSS"),
        payload(hand=[card(i + 1, "2", "S") for i in range(8)]),
        payload(state="ROUND_EVAL"),
        payload(
            state="SHOP", money=50,
            shop=[special(10, "j_joker", "JOKER", buy=4),
                  special(11, "c_fool", "TAROT", buy=3),
                  special(12, "S_A", "DEFAULT", buy=2)],
            vouchers=[special(20, "v_hone", "VOUCHER", buy=10)],
            packs=[special(21, "p_arcana_normal_1", "BOOSTER", buy=4)],
            jokers=[special(30, "j_joker", "JOKER")],
            consumables=[special(31, "c_fool", "TAROT")],
        ),
        payload(state="SMODS_BOOSTER_OPENED",
                pack=[special(40 + i, "c_fool", "TAROT") for i in range(3)]),
    ]
    for raw in scenarios:
        for selection in ([], [0]):
            state = build_state(raw, selected_indices=selection)
            mask = legality_mask(state)
            for index in np.flatnonzero(mask).tolist():
                plan = plan_action(state, int(index))
                assert isinstance(plan, ActionPlan)
                assert plan.kind in ("local", "rpc")
                if plan.kind == "rpc":
                    assert plan.client_method


def test_realcard_handles_non_playing_cards() -> None:
    joker = RealCard.from_json(special(1, "j_joker", "JOKER", buy=4), "shop.cards[0]")
    assert joker.rank_index == -1 and joker.suit_index == -1
    assert not joker.is_playing_card
    assert joker.cost_buy == 4
