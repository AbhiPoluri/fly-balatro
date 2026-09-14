"""Hand-crafted greedy baseline, an upper-ish reference for the learned readout.

Policy, in one sentence: pick the best made hand sitting in the 8 dealt cards
(flush > quads > trips > two pair), select exactly those cards and play them;
holding only a bare pair or less, keep the pair and discard the low cards
instead, and once out of discards play the pair (or the single highest card).
In the shop it buys nothing and moves on.

Playing a bare pair immediately is much worse than digging for a better hand:
holding the pair back until discards run out roughly triples the ante-1 clear
rate. ``MIN_PAIR_RANK_TO_PLAY`` is the knob - a pair at or above that rank
index gets played right away, and the default of 13 (above Ace) means never.

Run:
    source .venv/bin/activate
    python scripts/baseline_heuristic.py [n_episodes]

Writes outputs/baseline_heuristic.json.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Protocol, Sequence, Tuple

import sys
from pathlib import Path

# scripts/ is a package (it has __init__.py), so make the sibling helper
# importable whether this file is run as a script or as -m scripts.<name>.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
from numpy.typing import NDArray

from _baseline_common import report, run_episodes  # noqa: E402
from flybalatro.env import (
    CASH_OUT_INDEX,
    DISCARD_INDEX,
    NEXT_ROUND_INDEX,
    PLAY_INDEX,
    SELECT_BLIND_INDEX,
    BalatroEnv,
)

MAX_SELECTED = 5

# Rank index (0 = Two .. 12 = Ace) at or above which a bare pair is played
# immediately rather than held while discards remain. 13 is above every rank,
# so the default never plays a bare pair with a discard still in hand.
MIN_PAIR_RANK_TO_PLAY = 13

STAGE_PRE_BLIND = 0
STAGE_BLINDS = (1, 2, 3)
STAGE_POST_BLIND = 4
STAGE_SHOP = 5


class CardLike(Protocol):
    @property
    def rank_index(self) -> int: ...

    @property
    def suit_index(self) -> int: ...

    @property
    def id(self) -> int: ...


def _pick_target(
    available: Sequence[CardLike],
    discards_left: int,
    min_pair_rank_to_play: int = MIN_PAIR_RANK_TO_PLAY,
) -> Tuple[List[int], int]:
    """Choose which card ids to select and whether to play or discard them.

    Returns ``(card_ids, action_index)`` where ``action_index`` is
    :data:`PLAY_INDEX` or :data:`DISCARD_INDEX`.
    """
    by_rank: Dict[int, List[CardLike]] = defaultdict(list)
    by_suit: Dict[int, List[CardLike]] = defaultdict(list)
    for card in available:
        by_rank[card.rank_index].append(card)
        by_suit[card.suit_index].append(card)

    # Flush: five highest cards of any suit with at least five cards.
    for cards in by_suit.values():
        if len(cards) >= MAX_SELECTED:
            best = sorted(cards, key=lambda c: c.rank_index, reverse=True)[:MAX_SELECTED]
            return [c.id for c in best], PLAY_INDEX

    # Quads / trips: the biggest rank group, highest rank breaking ties.
    groups = sorted(
        by_rank.values(),
        key=lambda g: (len(g), g[0].rank_index),
        reverse=True,
    )
    top = groups[0]
    if len(top) >= 3:
        return [c.id for c in top[:MAX_SELECTED]], PLAY_INDEX

    if len(top) == 2:
        pairs = [g for g in groups if len(g) == 2]
        if len(pairs) >= 2:
            # Two pair, the two highest.
            chosen = pairs[0] + pairs[1]
            return [c.id for c in chosen], PLAY_INDEX
        if discards_left > 0 and top[0].rank_index < min_pair_rank_to_play:
            # Hold the pair, throw away the rest and dig for trips/two pair.
            keep = {c.id for c in top}
            rest = sorted(
                (c for c in available if c.id not in keep),
                key=lambda c: c.rank_index,
            )
            return [c.id for c in rest[:MAX_SELECTED]], DISCARD_INDEX
        return [c.id for c in top], PLAY_INDEX

    # Nothing made. Throw away the low cards if we still can.
    ordered = sorted(available, key=lambda c: c.rank_index)
    if discards_left > 0:
        return [c.id for c in ordered[:MAX_SELECTED]], DISCARD_INDEX
    return [ordered[-1].id], PLAY_INDEX


class HeuristicPolicy:
    """Greedy best-made-hand policy over the 109-index action space."""

    def __init__(self, min_pair_rank_to_play: int = MIN_PAIR_RANK_TO_PLAY) -> None:
        self._min_pair_rank_to_play: int = min_pair_rank_to_play

    def __call__(self, env: BalatroEnv, mask: NDArray[np.int8]) -> int:
        state = env.engine.state
        stage = int(state.stage.int())

        if stage == STAGE_PRE_BLIND:
            return self._or_fallback(SELECT_BLIND_INDEX, mask)
        if stage == STAGE_POST_BLIND:
            return self._or_fallback(CASH_OUT_INDEX, mask)
        if stage == STAGE_SHOP:
            return self._or_fallback(NEXT_ROUND_INDEX, mask)
        if stage not in STAGE_BLINDS:
            # Tarot / spectral / pack stages: unreachable without buying, but
            # do something legal rather than crash if we ever land there.
            return self._first_legal(mask)

        available: Sequence[CardLike] = state.available
        if not available:
            return self._first_legal(mask)

        target_ids, action_index = _pick_target(
            available, int(state.discards), self._min_pair_rank_to_play
        )
        selected_ids = frozenset(card.id for card in state.selected)

        # Select the next target card that is not already selected. Card
        # indices address hand slots, so the index is the card's position in
        # `available`.
        for slot, card in enumerate(available):
            if card.id in target_ids and card.id not in selected_ids:
                if slot < len(mask) and mask[slot] == 1:
                    return slot
                break

        return self._or_fallback(action_index, mask)

    @staticmethod
    def _or_fallback(index: int, mask: NDArray[np.int8]) -> int:
        if mask[index] == 1:
            return index
        return HeuristicPolicy._first_legal(mask)

    @staticmethod
    def _first_legal(mask: NDArray[np.int8]) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise RuntimeError("no legal actions but episode is not done")
        return int(legal[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_episodes", nargs="?", type=int, default=2000)
    parser.add_argument("--ante-end", type=int, default=1)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--min-pair-rank-to-play", type=int, default=MIN_PAIR_RANK_TO_PLAY
    )
    args = parser.parse_args()

    metrics = run_episodes(
        HeuristicPolicy(min_pair_rank_to_play=args.min_pair_rank_to_play),
        n_episodes=args.n_episodes,
        ante_end=args.ante_end,
        seed0=args.seed0,
        max_steps=args.max_steps,
    )
    report("baseline_heuristic", metrics)


if __name__ == "__main__":
    main()
