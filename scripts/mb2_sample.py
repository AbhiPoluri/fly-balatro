"""Stratified sample of real Balatro states + the glomerular assignment + the
input-level references every brain readout has to beat.

Writes, before any simulation runs:

    outputs/mb2/sample.npz        relay bits, labels, driven-ORN counts
    outputs/mb2/assignment.json   bit -> glomerulus, with cell counts
    outputs/mb2/input_stats.json  active-glomeruli stats, input Jaccard, references

The input Jaccard matters for reading the gate. The gate asks for
between-hand-type / within-hand-type Jaccard of active KC sets below 0.5. If the
*input* already sits near 1.0 -- the 32 relay bits of two states of different
hand types overlap almost as much as two states of the same hand type -- then the
gate is asking the mushroom body to *expand* a separation that is barely present
at the receptors, not to preserve one. Measured here so the verdict can say which.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from flybalatro import connectome as C
from flybalatro.encode import GlomerularMap
from flybalatro.features_v2 import FEATURE_NAMES_V2, N_HAND_BLOCK
from scripts.brain_probe import probe
from scripts.mb2_assign import assign, blocks

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "mb2"
STATES = ROOT / "outputs" / "bc2" / "states.npz"

#: Relay-block layout, re-derived from ``features_v2`` rather than hard-coded.
OFF_BEST_TYPE, N_BEST_TYPE = blocks()["best_type"]
OFF_IS_BEST, _ = blocks()["selected_is_best"]

HAND_TYPE_NAMES = ("High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
                   "Flush", "Full House", "Four of a Kind", "Straight Flush")

#: Short C grid: the full 6-value grid in ``brain_probe`` costs 2x the whole
#: simulation budget once it is run for 4 populations x 2 windows x 2 labels x
#: 2 (real/permuted) x 13 settings.
C_GRID_SHORT = [0.03, 0.3, 3.0]


# --- sampling -----------------------------------------------------------------
def water_fill(available: np.ndarray, total: int) -> np.ndarray:
    """As-equal-as-possible quotas per class, capped by availability."""
    avail = np.asarray(available, np.int64)
    quota = np.zeros(len(avail), np.int64)
    remaining = int(total)
    open_ = avail > 0
    while remaining > 0 and open_.any():
        share = max(1, remaining // int(open_.sum()))
        for c in np.flatnonzero(open_):
            if remaining <= 0:
                break
            take = min(share, int(avail[c] - quota[c]), remaining)
            quota[c] += take
            remaining -= take
            if quota[c] >= avail[c]:
                open_[c] = False
    return quota


def sample_states(n: int, seed: int) -> tuple[np.ndarray, dict]:
    """Stratified-by-hand-type sample of relay-active states; returns 32-bit rows."""
    z = np.load(STATES)
    n_feat = int(z["n_features"])
    bits = np.unpackbits(z["features_packed"], axis=1)[:, :n_feat]
    relay = bits[:, :N_HAND_BLOCK].astype(np.int8)
    bt = relay[:, OFF_BEST_TYPE:OFF_BEST_TYPE + N_BEST_TYPE]
    ok = np.flatnonzero(bt.sum(1) == 1)          # silent outside a blind
    ht = bt[ok].argmax(1)
    avail = np.array([int((ht == c).sum()) for c in range(N_BEST_TYPE)], np.int64)
    quota = water_fill(avail, n)

    rng = np.random.default_rng(seed)
    pick = []
    for c in range(N_BEST_TYPE):
        pool = ok[ht == c]
        pick.append(rng.choice(pool, size=int(quota[c]), replace=False))
    idx = np.sort(np.concatenate(pick))
    spec = dict(
        source=str(STATES.relative_to(ROOT)),
        n_states_total=int(len(bits)),
        n_states_relay_active=int(len(ok)),
        n_sampled=int(len(idx)),
        seed=int(seed),
        stratification="hand type only (bits 0-8); water-filled equal quotas",
        hand_type_available={HAND_TYPE_NAMES[c]: int(avail[c]) for c in range(9)},
        hand_type_quota={HAND_TYPE_NAMES[c]: int(quota[c]) for c in range(9)},
        state_rows=[int(x) for x in idx[:20]] + ["..."] if len(idx) > 20 else None,
    )
    return relay[idx], spec, idx


# --- input-level metrics ------------------------------------------------------
def jaccard_between_within(active: np.ndarray, labels: np.ndarray,
                           max_pairs: int = 400_000, seed: int = 0) -> dict:
    """Mean Jaccard of boolean sets for same-label vs different-label pairs.

    All pairs when ``n`` is small enough (2,000 states -> 2.0M pairs, fine as a
    dense int32 matmul), otherwise a random subsample of pairs.
    """
    A = active.astype(np.int32)
    n = len(A)
    inter = A @ A.T
    size = A.sum(1)
    union = size[:, None] + size[None, :] - inter
    iu = np.triu_indices(n, 1)
    num = inter[iu].astype(np.float64)
    den = np.maximum(union[iu], 1).astype(np.float64)
    j = np.where(union[iu] > 0, num / den, 0.0)
    same = labels[iu[0]] == labels[iu[1]]
    within = float(j[same].mean()) if same.any() else float("nan")
    between = float(j[~same].mean()) if (~same).any() else float("nan")
    del inter, union
    return dict(
        within=round(within, 4), between=round(between, 4),
        ratio=round(between / within, 4) if within > 0 else None,
        n_pairs_within=int(same.sum()), n_pairs_between=int((~same).sum()),
        mean_set_size=round(float(size.mean()), 2),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--threshold", type=int, default=5)
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = C.load(a.threshold)
    types, sizes, prov = assign(g)
    gm = GlomerularMap(g, types)

    relay, spec, rows = sample_states(a.n, a.seed)
    y_hand = relay[:, OFF_BEST_TYPE:OFF_BEST_TYPE + N_BEST_TYPE].argmax(1).astype(np.int8)
    y_best = relay[:, OFF_IS_BEST].astype(np.int8)
    n_glom = relay.sum(1).astype(np.int16)
    n_orn = np.array([gm.n_driven(r) for r in relay], np.int32)
    y_int = (n_orn > np.median(n_orn)).astype(np.int8)   # intensity control

    assignment = dict(
        provenance=prov,
        bits=[dict(bit=i, feature=FEATURE_NAMES_V2[i], glomerulus=types[i],
                   n_orn=sizes[i], block=prov["block_of_bit"][i])
              for i in range(N_HAND_BLOCK)],
        map=gm.describe(),
        dropped_from_the_fly=dict(
            v1_bits=int(len(FEATURE_NAMES_V2) - N_HAND_BLOCK),
            note="the 283 v1 state bits are not encoded at all in mb2",
        ),
    )
    (OUT_DIR / "assignment.json").write_text(json.dumps(assignment, indent=2))

    # Active-glomeruli stats.
    per_bit_rate = relay.mean(0)
    glom_stats = dict(
        active_glomeruli_mean=round(float(n_glom.mean()), 3),
        active_glomeruli_sd=round(float(n_glom.std()), 3),
        active_glomeruli_min=int(n_glom.min()),
        active_glomeruli_max=int(n_glom.max()),
        active_glomeruli_hist={int(k): int(v) for k, v in
                              zip(*np.unique(n_glom, return_counts=True))},
        silent_glomeruli_mean=round(float(N_HAND_BLOCK - n_glom.mean()), 3),
        driven_orn_mean=round(float(n_orn.mean()), 1),
        driven_orn_sd=round(float(n_orn.std()), 1),
        driven_orn_min=int(n_orn.min()), driven_orn_max=int(n_orn.max()),
        orns_mapped=int(sum(sizes)),
        orns_never_driven=int(len(g.orn_indices()) - sum(sizes)),
        per_bit_on_rate={FEATURE_NAMES_V2[i]: round(float(per_bit_rate[i]), 4)
                         for i in range(N_HAND_BLOCK)},
        bits_never_on=[FEATURE_NAMES_V2[i] for i in range(N_HAND_BLOCK)
                       if per_bit_rate[i] == 0.0],
    )

    # Input-level Jaccard, on the bits and on the driven-ORN sets.
    orn_all = gm.all_indices()
    pos = {int(x): k for k, x in enumerate(orn_all)}
    orn_sets = np.zeros((len(relay), len(orn_all)), bool)
    for i, r in enumerate(relay):
        orn_sets[i, [pos[int(x)] for x in gm.indices(r)]] = True
    jac = dict(
        relay_bits=jaccard_between_within(relay.astype(bool), y_hand),
        driven_orns=jaccard_between_within(orn_sets, y_hand),
    )

    # References: what the labels are worth without a brain.
    refs = {}
    for name, X in (("relay_bits_32", relay.astype(np.float64)),
                    ("bits_on_count_only", n_glom.reshape(-1, 1).astype(np.float64)),
                    ("driven_orn_count_only", n_orn.reshape(-1, 1).astype(np.float64))):
        refs[name] = {}
        for lab, y in (("hand_type", y_hand), ("sel_is_best", y_best),
                       ("intensity", y_int)):
            r = probe(X, y, c_grid=C_GRID_SHORT)
            rp = np.random.default_rng(3).permutation(y)
            r["permuted_acc"] = probe(X, rp, c_grid=C_GRID_SHORT)["best_acc"]
            refs[name][lab] = {k: r[k] for k in
                               ("best_acc", "permuted_acc", "n_used_features")}

    floors = dict(
        hand_type_majority=round(float(np.bincount(y_hand, minlength=9).max()
                                       / len(y_hand)), 4),
        hand_type_classes={HAND_TYPE_NAMES[c]: int((y_hand == c).sum())
                           for c in range(9)},
        sel_is_best_majority=round(float(max(y_best.mean(), 1 - y_best.mean())), 4),
        sel_is_best_positive_rate=round(float(y_best.mean()), 4),
        sel_is_best_positives=int(y_best.sum()),
        sel_is_best_positives_per_hand_type={
            HAND_TYPE_NAMES[c]: int(y_best[y_hand == c].sum()) for c in range(9)},
        intensity_positive_rate=round(float(y_int.mean()), 4),
    )

    out = dict(
        sample=spec, labels=floors, glomeruli=glom_stats,
        input_jaccard=jac, references=refs,
        c_grid=C_GRID_SHORT,
        machine=f"{platform.machine()} python {platform.python_version()}",
        graph=dict(neurons=g.n, edges_csr=g.m, threshold=g.meta["threshold"]),
    )
    (OUT_DIR / "input_stats.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(
        OUT_DIR / "sample.npz", relay=relay, y_hand=y_hand, y_best=y_best,
        y_intensity=y_int, n_glom=n_glom, n_orn=n_orn, state_row=rows,
        types=np.array(types, dtype=object), sizes=np.array(sizes, np.int32),
    )

    print(json.dumps(glom_stats, indent=2)[:1200])
    print("\ninput Jaccard:", json.dumps(jac, indent=2))
    print("\nlabel floors:", json.dumps(floors, indent=2))
    print("\nreferences:", json.dumps(refs, indent=2))
    print("wrote", OUT_DIR / "sample.npz", OUT_DIR / "assignment.json",
          OUT_DIR / "input_stats.json")


if __name__ == "__main__":
    main()
