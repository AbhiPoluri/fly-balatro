"""The glomerular assignment: 32 relay bits -> 32 distinct ORN glomeruli.

One relay bit drives every olfactory receptor neuron of one ``ORN_<glomerulus>``
type and nothing else. The other 283 v1 bits are dropped from the fly entirely.

Which glomerulus gets which bit matters, because the mushroom body in this model
is a very good total-drive meter (``outputs/mb/REPORT.md`` section 8: KC decodes
"how many bits were on" at 0.83 against ~0.59 for identity). Glomeruli here range
from 34 to 204 cells, so a careless assignment would let a 9-way hand-type probe
succeed by reading ORN *count*. The rule below is designed against that:

* the nine ``best_<hand type>`` bits -- the 9-way label -- get the nine smallest
  and tightest types (34-36 cells), so which hand type is active changes the
  total drive by at most 2 ORNs;
* the remaining 23 types are dealt largest-first to the remaining blocks in the
  order ``best_vs_needed`` (4), ``sel_<type>`` (10), ``sel_is_best`` (1),
  ``best_slot`` (8), so the four outliers (204/132/130/103 cells) land on the
  score-bucket block -- a nuisance dimension uncorrelated with either label.

That order was not picked by taste. One confound cannot be removed inside this
task: ``best_slot`` has exactly 1-5 bits on and *how many* is fixed by the hand
type (High Card 1, Pair 2, Trips 3, Two Pair 4, Straight/Flush/Full House 5), so
total driven-ORN count carries label information by construction. All 24 block
orders were scored on the sampled states by how well the driven-ORN *count alone*
(one feature, no brain, no glomerulus identity) predicts each label. The chosen
order is simultaneously the best on all three summary numbers: count-only 9-way
hand type 0.267 (worst order 0.492), count-only sel-is-best 0.890 -- exactly the
majority-class floor, i.e. no leakage at all -- and the smallest driven-ORN
standard deviation (50.3 ORNs). Putting the 204-cell ORN_DA1 on the 1-bit
``sel_is_best`` block, which minimises within-block size *spread*, instead makes
that label 0.994-decodable from total drive alone; that is the version this
replaces.

Those two count-only probes are reported in ``outputs/mb2/input_stats.json`` and
in the report, because 0.267 for the 9-way label is *above* the gate threshold of
permuted + 0.15 ~= 0.26. Any brain readout has to beat the count-only number, not
the gate, to mean anything.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from flybalatro.encode import orn_types_by_size
from flybalatro.features_v2 import HAND_BLOCK_OFFSETS, N_HAND_BLOCK

#: Blocks in the order they are dealt types, largest first. ``best_type`` is not
#: in the list: it takes the smallest nine.
DEAL_ORDER: Tuple[str, ...] = ("score_vs_needed", "selected_type",
                               "selected_is_best", "best_mask")
#: The block that gets the smallest, tightest types.
TIGHT_BLOCK: str = "best_type"


def blocks() -> Dict[str, Tuple[int, int]]:
    """Relay block name -> (offset, width), from ``features_v2`` itself."""
    out: Dict[str, Tuple[int, int]] = {}
    for name, off, width in HAND_BLOCK_OFFSETS:
        if off < N_HAND_BLOCK:
            out[name] = (int(off), int(width))
    return out


def assign(graph, n_bits: int = N_HAND_BLOCK) -> Tuple[List[str], List[int], dict]:
    """Return (type per bit, cell count per bit, provenance)."""
    blk = blocks()
    if sum(w for _, w in blk.values()) != n_bits:
        raise ValueError(f"relay blocks do not sum to {n_bits}: {blk}")
    ranked = orn_types_by_size(graph)
    if len(ranked) < n_bits:
        raise ValueError(f"only {len(ranked)} ORN types, need {n_bits}")
    chosen = ranked[:n_bits]                      # the 32 largest glomeruli

    tight_off, tight_w = blk[TIGHT_BLOCK]
    tight = chosen[-tight_w:]                     # the nine smallest of the 32
    pool = list(chosen[: n_bits - tight_w])       # everything else, largest first

    per_bit: List[str] = [""] * n_bits
    sizes: List[int] = [0] * n_bits
    block_of: List[str] = [""] * n_bits
    for i, (nm, c) in enumerate(tight):
        per_bit[tight_off + i] = nm
        sizes[tight_off + i] = c
        block_of[tight_off + i] = TIGHT_BLOCK
    k = 0
    for name in DEAL_ORDER:
        off, width = blk[name]
        for i in range(width):
            nm, c = pool[k]
            k += 1
            per_bit[off + i] = nm
            sizes[off + i] = c
            block_of[off + i] = name
    if k != len(pool):
        raise ValueError("assignment did not consume the pool")
    if any(x == "" for x in per_bit):
        raise ValueError("assignment left a bit unmapped")

    prov = dict(
        rule=("32 largest ORN_* types; the 9 smallest of them go to best_type "
              "(the 9-way label), the rest are dealt largest-first to "
              + ", ".join(DEAL_ORDER)),
        n_orn_types_available=len(ranked),
        n_types_used=n_bits,
        dropped_types=[nm for nm, _ in ranked[n_bits:]],
        block_of_bit=block_of,
        block_size_range={
            name: [min(sizes[blk[name][0]:blk[name][0] + blk[name][1]]),
                   max(sizes[blk[name][0]:blk[name][0] + blk[name][1]])]
            for name in blk
        },
        total_orns_mapped=int(sum(sizes)),
    )
    return per_bit, sizes, prov
