"""Version 2 of the binary state encoding: the 283 v1 bits with a hand block in front.

The v1 encoding (:mod:`flybalatro.features`) hands the fly the *cards* and
nothing else: to play a flush it would have to learn poker from spike counts of
an untuned, frozen network, through a 144-glomerulus bottleneck. The v1 result
was that card identity is already ~gone two synapses in, so the readout learned
"select three slots, then play" and never a hand type.

v2 changes the question from "can the fly work out what a flush is" to "can the
fly *relay* a hand type". :mod:`flybalatro.hands` computes the poker analysis in
ordinary Python - **outside the brain, by code that is not biological in any
sense** - and this module turns the answer into 32 extra binary channels which
become tonic current on olfactory receptor neurons like every other bit. The
brain is a relay for that analysis, not its author. Anything the readout gets
right about hand types is information that survived the trip through the fly.

Layout (315 bits total)
-----------------------

The hand block is **first** so that it lands on real ORNs rather than on the
other sensory neurons the feature map overflows into (see
``FeatureMap(orn_first=True)``, which ``encode.feature_map_for(graph, 2)``
switches on).

===== ======  ============================================================
Start Length  Block
===== ======  ============================================================
0     9       best hand type one-hot, over ``hands.HAND_TYPE_NAMES``
9     8       slot mask of the best-scoring subset of the dealt cards
17    10      hand type the currently selected cards form: 9 types + "none"
27    1       the selection *is* the best subset (i.e. "play now")
28    4       best score vs. what is still needed: <0.25, <0.5, <1, >=1
32    283     the v1 encoding, unchanged and in the same order
===== ======  ============================================================

The whole 32-bit block is zero outside a blind (no dealt cards means no hand to
analyse), which also keeps the shop and cash-out states looking like v1.

v1 is untouched, so models in ``outputs/bc/`` keep working. Every v2 model
export carries ``feature_version = 2``; loaders default to 1 when the field is
absent.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from . import hands
from .features import N_FEATURES, StateLike
from .features import encode as encode_v1

__all__ = [
    "encode_v2",
    "FEATURE_NAMES_V2",
    "N_FEATURES_V2",
    "N_HAND_BLOCK",
    "HAND_BLOCK_OFFSETS",
    "SCORE_BUCKET_LABELS",
    "hand_block",
    "hand_block_summary",
    "encoder_for_version",
    "n_features_for_version",
]

N_BEST_TYPE: int = hands.N_HAND_TYPES          # 9
N_BEST_MASK: int = hands.HAND_SLOTS            # 8
N_SELECTED_TYPE: int = hands.N_HAND_TYPES + 1  # 9 + "none"
N_IS_BEST: int = 1
N_SCORE_BUCKETS: int = 4

OFF_BEST_TYPE: int = 0
OFF_BEST_MASK: int = OFF_BEST_TYPE + N_BEST_TYPE
OFF_SELECTED_TYPE: int = OFF_BEST_MASK + N_BEST_MASK
OFF_IS_BEST: int = OFF_SELECTED_TYPE + N_SELECTED_TYPE
OFF_SCORE_BUCKET: int = OFF_IS_BEST + N_IS_BEST
N_HAND_BLOCK: int = OFF_SCORE_BUCKET + N_SCORE_BUCKETS  # 32
OFF_V1: int = N_HAND_BLOCK

N_FEATURES_V2: int = N_HAND_BLOCK + N_FEATURES  # 315

#: Start offset of each named block, for the viewer's input panel.
HAND_BLOCK_OFFSETS: Tuple[Tuple[str, int, int], ...] = (
    ("best_type", OFF_BEST_TYPE, N_BEST_TYPE),
    ("best_mask", OFF_BEST_MASK, N_BEST_MASK),
    ("selected_type", OFF_SELECTED_TYPE, N_SELECTED_TYPE),
    ("selected_is_best", OFF_IS_BEST, N_IS_BEST),
    ("score_vs_needed", OFF_SCORE_BUCKET, N_SCORE_BUCKETS),
    ("v1_bits", OFF_V1, N_FEATURES),
)

#: Upper edges of the ``best_score / still_needed`` buckets; the last is open.
_SCORE_EDGES: Tuple[float, ...] = (0.25, 0.5, 1.0)
SCORE_BUCKET_LABELS: Tuple[str, ...] = ("lt0.25", "lt0.5", "lt1.0", "ge1.0")

#: Stages where cards are dealt and a hand can be played.
BLIND_STAGES: Tuple[int, ...] = (1, 2, 3)


def _build_feature_names() -> List[str]:
    from .features import FEATURE_NAMES

    names: List[str] = []
    for label in hands.HAND_TYPE_NAMES:
        names.append("best_" + label.lower().replace(" ", "_"))
    for slot in range(N_BEST_MASK):
        names.append(f"best_slot{slot}")
    for label in hands.HAND_TYPE_NAMES:
        names.append("sel_" + label.lower().replace(" ", "_"))
    names.append("sel_none")
    names.append("sel_is_best")
    for label in SCORE_BUCKET_LABELS:
        names.append(f"best_vs_needed_{label}")
    names.extend(FEATURE_NAMES)
    return names


FEATURE_NAMES_V2: List[str] = _build_feature_names()

if len(FEATURE_NAMES_V2) != N_FEATURES_V2:  # pragma: no cover - structural invariant
    raise AssertionError(
        f"FEATURE_NAMES_V2 has {len(FEATURE_NAMES_V2)} entries, expected {N_FEATURES_V2}"
    )


def _score_bucket(best_score: int, needed: int) -> int:
    """Which bucket ``best_score / needed`` falls in; ``needed <= 0`` is the top."""
    if needed <= 0:
        return N_SCORE_BUCKETS - 1
    ratio = best_score / needed
    for i, edge in enumerate(_SCORE_EDGES):
        if ratio < edge:
            return i
    return N_SCORE_BUCKETS - 1


def hand_block(state: StateLike) -> Tuple[NDArray[np.float32], Optional[hands.HandAnalysis], int]:
    """The 32 relay bits for ``state``, plus the analysis they encode.

    Returns ``(bits, analysis, needed)``. ``analysis`` is ``None`` outside a
    blind, in which case every bit is zero.
    """
    out: NDArray[np.float32] = np.zeros(N_HAND_BLOCK, dtype=np.float32)
    stage = int(state.stage.int())
    available = list(state.available)[: hands.HAND_SLOTS]
    needed = max(0, int(state.required_score) - int(state.score))
    if stage not in BLIND_STAGES or not available:
        return out, None, needed

    analysis = hands.analyse_state(state)
    out[OFF_BEST_TYPE + analysis.best_type] = 1.0
    for slot, bit in enumerate(analysis.best_mask):
        if bit:
            out[OFF_BEST_MASK + slot] = 1.0
    if analysis.selected_type is None:
        out[OFF_SELECTED_TYPE + hands.N_HAND_TYPES] = 1.0
    else:
        out[OFF_SELECTED_TYPE + analysis.selected_type] = 1.0
    if analysis.selected_is_best:
        out[OFF_IS_BEST] = 1.0
    out[OFF_SCORE_BUCKET + _score_bucket(analysis.best_score, needed)] = 1.0
    return out, analysis, needed


def encode_v2(state: StateLike) -> NDArray[np.float32]:
    """Encode a game state as a length-315 binary vector: hand block, then v1."""
    out: NDArray[np.float32] = np.empty(N_FEATURES_V2, dtype=np.float32)
    block, _analysis, _needed = hand_block(state)
    out[:N_HAND_BLOCK] = block
    out[N_HAND_BLOCK:] = encode_v1(state)
    return out


def hand_block_summary(state: StateLike) -> dict:
    """Human-readable version of the relay block, for the viewer and the logs.

    Every value here is computed outside the brain; the viewer labels it as
    such. The brain only ever sees the bits.
    """
    block, analysis, needed = hand_block(state)
    if analysis is None:
        return {
            "active": False,
            "best_type": "none",
            "best_mask": [0] * hands.HAND_SLOTS,
            "best_score": 0,
            "selected_type": "none",
            "selected_score": 0,
            "selected_is_best": False,
            "needed": needed,
            "score_bucket": None,
            "n_bits_on": int(block.sum()),
        }
    return {
        "active": True,
        "best_type": analysis.best_type_name,
        "best_mask": list(analysis.best_mask),
        "best_slots": list(analysis.best_slots),
        "best_score": int(analysis.best_score),
        "selected_type": analysis.selected_type_name,
        "selected_slots": list(analysis.selected_slots),
        "selected_score": int(analysis.selected_score),
        "selected_is_best": bool(analysis.selected_is_best),
        "needed": int(needed),
        "score_bucket": SCORE_BUCKET_LABELS[_score_bucket(analysis.best_score, needed)],
        "n_bits_on": int(block.sum()),
    }


def encoder_for_version(version: int):
    """``encode`` for version 1, ``encode_v2`` for version 2."""
    if int(version) == 1:
        return encode_v1
    if int(version) == 2:
        return encode_v2
    raise ValueError(f"unknown feature version {version!r}; expected 1 or 2")


def n_features_for_version(version: int) -> int:
    if int(version) == 1:
        return N_FEATURES
    if int(version) == 2:
        return N_FEATURES_V2
    raise ValueError(f"unknown feature version {version!r}; expected 1 or 2")
