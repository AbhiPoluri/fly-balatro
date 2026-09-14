"""Hand-aware teacher (v2): plays the best-scoring subset the deal allows.

The v1 teacher (``scripts/baseline_heuristic.py``) only knows about flushes and
rank groups, so it cannot see a straight, cannot compare a low full house with a
high flush, and holds a pair with a fixed rule. This one asks
:mod:`flybalatro.hands` - which enumerates all 218 subsets of size 1..5 and
scores them exactly as ``vendor/balatro-rs`` would - and then:

* selects the cards of the best subset, one ``select_card[i]`` at a time;
* plays once the selection *is* the best subset;
* unless the best subset is worth little relative to what the blind still needs
  and a discard remains, in which case it selects the worst cards that are
  **not** in the best subset and discards them ("dig").

"Worth little" is a fair-share rule: play when ``best_score * plays_left >=
still_needed * --play-share``, i.e. when this hand is at least its share of the
remaining requirement. Default share is 1.0.

No deselect
-----------

The engine has no deselect: once ``select_card[i]`` is taken, that index is
masked out for the rest of the hand, and the only way to un-select a card is to
play it or discard it. The teacher must therefore be defined for selections it
would never have made itself (the collection script drives half its episodes
with random actions, and a learned readout selects whatever it likes). The rule:

* selection is empty -> decide play-or-dig from scratch;
* selection is part of the best subset -> finish that subset and **play** it
  (never dig away cards already committed to the best hand);
* selection is entirely outside the best subset, a discard remains, and the best
  hand reachable *including* those cards is below the fair share -> discard the
  selection plus the other worst cards, which throws the junk away;
* otherwise (junk mixed with good cards, or no discards left) -> play the best
  subset that contains the current selection.

``skip_blind`` is never chosen. In the shop it buys nothing and moves on.

Run:
    source .venv/bin/activate
    python scripts/baseline_heuristic_v2.py 1000

Writes outputs/bc2/baseline_heuristic_v2.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent
for _p in (str(ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _baseline_common import report, run_episodes  # noqa: E402
from flybalatro import hands  # noqa: E402
from flybalatro.env import (  # noqa: E402
    CASH_OUT_INDEX,
    DISCARD_INDEX,
    NEXT_ROUND_INDEX,
    PLAY_INDEX,
    SELECT_BLIND_INDEX,
    BalatroEnv,
)

STAGE_PRE_BLIND = 0
STAGE_BLINDS = (1, 2, 3)
STAGE_POST_BLIND = 4
STAGE_SHOP = 5

#: Play when ``best_score * plays_left >= still_needed * PLAY_SHARE``.
PLAY_SHARE: float = 1.0


class HandAwarePolicy:
    """Hand-aware greedy policy over the 109-index action space.

    Deterministic, stateless between calls: everything it needs is in
    ``env.engine.state``, which keeps it usable as a behaviour-cloning label
    function (``scripts/bc_collect.py`` re-derives the label per state).
    """

    def __init__(self, play_share: float = PLAY_SHARE) -> None:
        self.play_share: float = float(play_share)

    # -- planning ---------------------------------------------------------
    def plan(self, state: object) -> Tuple[Tuple[int, ...], int]:
        """``(target slots, PLAY_INDEX or DISCARD_INDEX)`` for a blind state."""
        available = list(state.available)[: hands.HAND_SLOTS]
        selected = tuple(
            i for i, on in enumerate(hands.selected_flags_from_state(state)) if on
        )
        plays = int(state.plays)
        discards = int(state.discards)
        needed = max(0, int(state.required_score) - int(state.score))

        best = hands.best_subset(available)
        if best is None:  # no cards to act on
            return (), PLAY_INDEX
        keep = set(best.slots)
        sel = set(selected)

        def fair_share_met(score: int) -> bool:
            """Is ``score`` at least this hand's share of what is still needed?"""
            if needed <= 0:
                return True
            return score * max(1, plays) >= needed * self.play_share

        def dig_target() -> Optional[Tuple[int, ...]]:
            """The worst cards outside the best subset, plus whatever is selected."""
            outside = [i for i in range(len(available)) if i not in keep]
            if not outside:
                return None
            outside.sort(key=lambda i: (hands.RANK_CHIPS[available[i].rank_index], i))
            target: List[int] = sorted(sel)
            for slot in outside:
                if len(target) >= hands.MAX_SELECTED:
                    break
                if slot not in sel:
                    target.append(slot)
            return tuple(sorted(target))

        # 1. Nothing selected yet: play the best hand, or dig for a better one.
        if not sel:
            if discards > 0 and not fair_share_met(best.score):
                target = dig_target()
                if target:
                    return target, DISCARD_INDEX
            return best.slots, PLAY_INDEX

        # 2. On track towards the best subset: finish it and play it.
        if sel <= keep:
            return best.slots, PLAY_INDEX

        # 3. Nothing but junk selected, and digging is still worth it: discard.
        constrained = hands.best_subset(available, must_include=selected)
        if (
            discards > 0
            and not (sel & keep)
            and (constrained is None or not fair_share_met(constrained.score))
        ):
            target = dig_target()
            if target:
                return target, DISCARD_INDEX

        # 4. Junk is mixed in with good cards, or discards are gone: play the
        #    best hand that still contains what has already been selected.
        if constrained is not None:
            return constrained.slots, PLAY_INDEX
        return tuple(sorted(sel)), PLAY_INDEX

    # -- acting -----------------------------------------------------------
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

        if not state.available:
            return self._first_legal(mask)

        target, action_index = self.plan(state)
        selected_flags = hands.selected_flags_from_state(state)

        # Card action indices address hand slots, so slot i is action i.
        for slot in target:
            if slot < len(selected_flags) and not selected_flags[slot]:
                if slot < len(mask) and mask[slot] == 1:
                    return slot
                break

        if mask[action_index] == 1:
            return action_index
        other = DISCARD_INDEX if action_index == PLAY_INDEX else PLAY_INDEX
        return self._or_fallback(other, mask)

    @staticmethod
    def _or_fallback(index: int, mask: NDArray[np.int8]) -> int:
        if mask[index] == 1:
            return index
        return HandAwarePolicy._first_legal(mask)

    @staticmethod
    def _first_legal(mask: NDArray[np.int8]) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise RuntimeError("no legal actions but episode is not done")
        return int(legal[0])


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_episodes", nargs="?", type=int, default=1000)
    parser.add_argument("--ante-end", type=int, default=1)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--play-share", type=float, default=PLAY_SHARE)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "bc2" / "baseline_heuristic_v2.json"))
    parser.add_argument("--name", default="baseline_heuristic_v2")
    args = parser.parse_args(argv)

    metrics = run_episodes(
        HandAwarePolicy(play_share=args.play_share),
        n_episodes=args.n_episodes,
        ante_end=args.ante_end,
        seed0=args.seed0,
        max_steps=args.max_steps,
    )
    metrics["play_share"] = args.play_share
    report(args.name, metrics, out_path=Path(args.out))


if __name__ == "__main__":
    main()
