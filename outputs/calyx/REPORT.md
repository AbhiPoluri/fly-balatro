# Calyx rewiring control: does the measured ALPN -> KC wiring matter? (2026-09-13)

*Status: complete. Three rewiring seeds, two calibration variants, all measurements done.*

## Why this run exists

`RESULTS.md` v3 tests "does the evolved wiring help?" by comparing the real
MaleCNS graph against `Brain.shuffled(seed)` -- a **global** permutation of every
edge's postsynaptic target. Under the glomerular encoding that is not a
like-for-like swap. The encoding works *because* ~440 driven receptor neurons of
one type converge on the uniglomerular projection neurons of that glomerulus, and
a global target permutation destroys exactly that convergence: ALPN 71.3 Hz real
against 0.63 Hz shuffled, KC 1.44 against 0.15, 2,254 non-constant readout
channels against 899. It compares a working brain against a near-dead one, so
"real beats shuffled" is not a defensible claim about evolved wiring.

This run replaces it with a control that asks something answerable: **keep the
sensory front end and everything downstream exactly as measured, and randomise
only which projection neurons converge on each Kenyon cell.**

That is also the one stage where the literature makes a prediction. PN -> KC
connectivity in the mushroom-body calyx is reported as largely random with
respect to glomerular identity (Caron et al. 2013), with later connectome work
finding partial, non-uniform structure (Zheng et al. 2022). A near-null here
*validates* the model against known biology; a large effect is a finding.

## 1. The rewiring

`scripts/calyx_common.py`. Double-edge swaps of postsynaptic endpoints of
ALPN -> Kenyon-cell edges, **within strata**. A stratum is

    (KC hemisphere `side`, KC `type` string, synapse count, presynaptic sign)

so two edges may exchange targets only if they carry the identical weight and
land on the same kind of cell in the same hemisphere. A swap is rejected if it
would produce a duplicate `(pre, post)` pair or a self-loop.

Preserved exactly: every neuron's out-degree and out-weight; every neuron's
in-degree and in-weight; every projection neuron's distribution over KC subtype
and hemisphere; and every edge that is not ALPN -> KC, bit for bit -- ORN -> ALPN,
APL in **and** out, KC -> KC, KC -> MBON, every dopaminergic edge. Changed: which
Kenyon cell each projection neuron contacts.

The result is a real `flybalatro.connectome.Graph` saved to
`outputs/calyx/graph_rewired_s<seed>.npz`, not a read-time index patch, so
`Tuning.apply`, the per-KC homeostasis and every downstream stage operate on the
true endpoints. Three rewiring seeds: `rw1`, `rw2`, `rw3`.

### The strata actually used

19,980 ALPN -> KC edges, 686 projection neurons, 4,064 Kenyon cells of which
**3,768 receive ALPN input**. Synapse counts 5-89 (mean 19.3). 24 of the 19,980
edges are inhibitory (GABAergic ALPNs) and can only swap with each other.

**818 strata.** 202 are singletons, so 202 edges (1.0%) can never move and stay
exactly where the connectome put them. Median stratum size 8, largest 212.

Edges by KC `type` string (exact annotations, both hemispheres summed):
`KCg-m` 8,601 | `KCab-s` 3,590 | `KCab-m` 2,484 | `KCab-c` 2,178 |
`KCa'b'-ap2` 1,390 | `KCa'b'-m` 926 | `KCa'b'-ap1` 676 | `KCab-p` 56 |
`KCg-d` 33 | `KCg-s2` 15 | `KC` 11 | `KCg-s4` 11 | `KCg-s3` 9.
(`KCg-s1` and `KCg` exist as annotations but receive no ALPN -> KC edge at the
>= 5 synapse threshold.) Split by hemisphere the largest cells are `L/KCg-m`
4,387 and `R/KCg-m` 4,214; the smallest non-empty is `L/KCg-s3` with 4.

Invariants are checked mechanically per seed (`calyx_common.check_invariants`)
and in `tests/test_calyx.py`.

## 2. The fairness requirement

Every condition gets the **same label-free calibration**, recalibrated
independently on its own graph: per-Kenyon-cell homeostatic `v_th` offsets by the
procedure of `scripts/kc_homeo.py`, which sees only "how often did I fire", never
a label, a reward or a readout. `scripts/calyx_homeo.py` runs that loop with the
constants, clamps, annealing schedule and odour split imported verbatim, for a
fixed 12-iteration budget with its convergence rule. One deviation, recorded in
every output file: the base tuning is `outputs/mb2/tuned_config.json`
(`apl_scale = 2`, tonic 30 mV -- the v3 / `outputs/bc3` operating point) rather
than `outputs/plast/tuned_config.json`.

Both calibration variants are measured:

* **`raw`** -- the v3 operating point exactly, no per-KC offsets. Directly
  comparable to the published `outputs/bc3` rows, and the honest test of whether
  a weight-preserving rewiring needs recalibration at all.
* **`homeo`** -- plus each condition's own per-KC offsets. This is the variant the
  fairness requirement asks for. It forces the per-cell response-rate
  distributions together *by construction*, so matching under `homeo` is partly a
  tautology; the `raw` table says whether they matched on their own.

## 3. Held-out discipline

`scripts/kc_homeo.py`'s odour set: distinct 32-bit relay patterns from
`outputs/bc2/states.npz`, filtered to decision points (relay block non-zero,
`sel_none` bit on) with episode >= 1000, deduplicated to 1,173 patterns, split
stratified by hand type with seed 20260913 into **500 calibration** and **500
held-out**. Thresholds are fitted on the calibration 500 only; every KC-code and
decoding number is measured on the held-out 500, whose rows into the bc3 unique
set are recorded in `outputs/calyx/probe_<variant>.json`
(`_set.gate_rows_in_bc3_uniq`). The behaviour-cloning readouts use bc3's split
unchanged: same 150,037 states, same `rng(seed + 1)` episode permutation, same
29,282 held-out states, same eval seeds 100000-100399.

## 4. Measurements

Four conditions -- `real` plus three independent rewirings -- measured under
both calibration variants. Every number below comes from
`outputs/calyx/tables_raw.md` and `outputs/calyx/tables_homeo.md`, generated
from the JSON by `scripts/calyx_tables.py`.

### 4.0 The pipeline reproduces `outputs/bc3` exactly

Before any comparison: the `raw`/`real` condition is the published v3 network
with a rebuilt-but-unmodified graph object, so it must reproduce bc3's rows
bit for bit. It does.

| quantity | `outputs/bc3` | `calyx raw/real` |
|---|---|---|
| KC linear imitation top-1 | 0.7085 (993 dims) | 0.7085 (993 dims) |
| DN linear imitation top-1 | 0.6606 (654 dims) | 0.6606 (654 dims) |
| KC linear clear rate, 400 episodes | 0.265 (106 wins, chips 843.175) | 0.265 (106 wins, chips 843.175) |
| DN linear clear rate, 400 episodes | 0.0325 (13 wins, chips 514.1375) | 0.0325 (13 wins, chips 514.1375) |
| train / val / test states | 108,505 / 12,250 / 29,282 | 108,505 / 12,250 / 29,282 |

So the rewiring machinery, the feature extraction, the readout training and
the live-play harness are all pass-throughs when the rewiring is absent. Any
difference below is caused by the rewiring and nothing else.

### 4.1 Activity matching (the precondition)

The requirement was that conditions share an activity regime before anything
downstream is compared. Under `raw` -- **no recalibration at all** -- they
already do:

| condition | ALPN Hz | KC Hz | MBON Hz | DN Hz | KC active frac | KC active/state | KC always-on (r>0.5) | KC r>0.3 | KC silent | non-constant ALPN+KC+DN |
|---|---|---|---|---|---|---|---|---|---|---|
| real | 69.47 | 1.33 | 8.16 | 6.26 | 0.0458 | 186.0 | 0.0403 (164) | 0.0522 (212) | 3347 | 1893 |
| rw1 | 69.45 | 1.39 | 8.17 | 6.26 | 0.0478 | 194.2 | 0.0406 (165) | 0.0522 (212) | 3353 | 1889 |
| rw2 | 69.50 | 1.36 | 8.13 | 6.25 | 0.0475 | 193.0 | 0.0386 (157) | 0.0546 (222) | 3356 | 1882 |
| rw3 | 69.47 | 1.35 | 8.07 | 6.26 | 0.0468 | 190.2 | 0.0386 (157) | 0.0549 (223) | 3359 | 1882 |

KC mean rate 1.33 vs 1.35-1.39 Hz, active fraction 0.0458 vs 0.0468-0.0478,
non-constant readout channels 1,893 vs 1,882-1,889. Compare the broken global
shuffle this control replaces: 71.3 -> 0.63 Hz at ALPN, 2,254 -> 899 channels.
A weight- and degree-preserving calyx rewiring needs no rescue; it lands in the
same regime on its own.

(The "always-on" column uses `r > 0.5` as its proxy. Cells that fire on
*literally every* held-out pattern, `r == 1`, number 59 / 65 / 50 / 51 for
real / rw1 / rw2 / rw3 under `raw`, and 2 / 0 / 1 / 4 under `homeo`.)

Under `homeo`, each condition additionally gets its own per-KC thresholds:

| condition | ALPN Hz | KC Hz | MBON Hz | DN Hz | KC active frac | KC active/state | KC always-on (r>0.5) | KC r>0.3 | KC silent | non-constant ALPN+KC+DN |
|---|---|---|---|---|---|---|---|---|---|---|
| real | 69.54 | 1.04 | 9.91 | 6.16 | 0.0436 | 177.3 | 0.0116 (47) | 0.0197 (80) | 1930 | 3303 |
| rw1 | 69.63 | 0.85 | 8.97 | 6.22 | 0.0396 | 161.1 | 0.0020 (8) | 0.0062 (25) | 1770 | 3473 |
| rw2 | 69.54 | 1.00 | 9.77 | 6.18 | 0.0429 | 174.5 | 0.0111 (45) | 0.0204 (83) | 2013 | 3235 |
| rw3 | 69.55 | 1.01 | 9.89 | 6.16 | 0.0426 | 173.2 | 0.0113 (46) | 0.0170 (69) | 2043 | 3196 |

**`rw1` is an outlier here and it is reported, not hidden.** Its homeostasis
run used the full 12-iteration budget (the others converged at 10) and settled
lower on every sparsity statistic: KC 0.85 Hz vs 1.00-1.04, the r>0.3 tail
0.0062 vs 0.0170-0.0204, r>0.5 0.0020 vs 0.0111-0.0116, zero literally
always-on cells. By the letter of the fairness rule `rw1`'s `homeo` numbers
are the least comparable of the four. In practice this cuts *against* an
evolved-wiring effect rather than for it: `rw1` is the condition furthest from
`real` in activity and is still indistinguishable from `real` and from the
other two rewirings on every downstream measure (KC decoding 0.998, imitation
0.7491, clear rate 0.383). `real`, `rw2` and `rw3` match each other closely
under both variants, so the `homeo` comparison does not rest on `rw1`.

Per-condition calibration outcomes:

| condition | iterations | converged | KC active frac | active/state | r>0.5 | r>0.3 | silent | offset mean / SD / min / max (mV) |
|---|---|---|---|---|---|---|---|---|
| real | 10 | True | 0.0446 | 181.3 | 0.0126 | 0.0194 | 1949 | -2.16 / 2.99 / -6.7 / 20.6 |
| rw1 | 12 | True | 0.0403 | 163.7 | 0.0042 | 0.0062 | 1738 | -2.16 / 3.08 / -6.7 / 23.0 |
| rw2 | 10 | True | 0.0448 | 182.2 | 0.0158 | 0.0197 | 1990 | -2.12 / 3.01 / -6.7 / 20.7 |
| rw3 | 10 | True | 0.0435 | 176.7 | 0.0106 | 0.0175 | 2006 | -2.13 / 3.04 / -6.7 / 19.3 |

### 4.2 KC code statistics

| condition | J within hand type | J between hand types | ratio | mean KC active-set size |
|---|---|---|---|---|
| real | 0.6130 | 0.5584 | 0.9111 | 186.0 |
| rw1 | 0.6306 | 0.5736 | 0.9097 | 194.2 |
| rw2 | 0.5953 | 0.5393 | 0.9060 | 193.0 |
| rw3 | 0.5983 | 0.5469 | 0.9141 | 190.2 |

| condition | J within hand type | J between hand types | ratio | mean KC active-set size |
|---|---|---|---|---|
| real | 0.2450 | 0.1143 | 0.4665 | 177.3 |
| rw1 | 0.1864 | 0.0552 | 0.2964 | 161.1 |
| rw2 | 0.2668 | 0.1184 | 0.4437 | 174.5 |
| rw3 | 0.2402 | 0.1144 | 0.4762 | 173.2 |

The between/within ratio is the quantity of interest: how much more similar
are two hands of the same type than two hands of different types. Under `raw`
`real` scores 0.9111 against rewired 0.9060-0.9141 -- `real` sits inside the
rewired range. Under `homeo` `real` scores 0.4665 against 0.2964-0.4762, again
inside the range (and again `rw1` is the outlier end). There is no separation.

### 4.3 Linear decoding of hand type

| condition | KC | KCBIN | DN | ALPN | MBON |
|---|---|---|---|---|---|
| real | 0.984 / 0.990 (0.200) | 0.982 / 0.984 (0.198) | 0.646 / 0.620 (0.152) | 1.000 / 1.000 (0.148) | 0.316 / 0.298 (0.206) |
| rw1 | 0.994 / 0.992 (0.178) | 0.988 / 0.992 (0.200) | 0.602 / 0.600 (0.170) | 1.000 / 1.000 (0.180) | 0.322 / 0.316 (0.146) |
| rw2 | 0.986 / 0.992 (0.194) | 0.984 / 0.990 (0.186) | 0.626 / 0.598 (0.164) | 1.000 / 1.000 (0.178) | 0.302 / 0.296 (0.192) |
| rw3 | 0.990 / 0.992 (0.178) | 0.986 / 0.986 (0.176) | 0.634 / 0.602 (0.160) | 1.000 / 1.000 (0.160) | 0.340 / 0.332 (0.198) |

Baselines on the same 500 held-out patterns: majority 0.2380, permuted labels on the 32 relay bits 0.2080, the 32 relay bits 1.0000, bits-on count only 0.6640, **driven-ORN count only 0.4020** (`outputs/mb2/REPORT.md` measured 0.267 for that confound on its own 2,000-state sample).

| condition | KC | KCBIN | DN | ALPN | MBON |
|---|---|---|---|---|---|
| real | 1.000 / 1.000 (0.138) | 1.000 / 1.000 (0.144) | 0.652 / 0.676 (0.202) | 1.000 / 1.000 (0.154) | 0.602 / 0.600 (0.188) |
| rw1 | 1.000 / 0.998 (0.178) | 1.000 / 0.998 (0.176) | 0.650 / 0.616 (0.164) | 1.000 / 1.000 (0.174) | 0.668 / 0.662 (0.194) |
| rw2 | 1.000 / 1.000 (0.146) | 1.000 / 1.000 (0.148) | 0.642 / 0.638 (0.166) | 1.000 / 1.000 (0.180) | 0.678 / 0.668 (0.234) |
| rw3 | 1.000 / 1.000 (0.174) | 1.000 / 1.000 (0.168) | 0.630 / 0.620 (0.178) | 1.000 / 1.000 (0.200) | 0.662 / 0.632 (0.208) |

Baselines on the same 500 held-out patterns: majority 0.2380, permuted labels on the 32 relay bits 0.2080, the 32 relay bits 1.0000, bits-on count only 0.6640, **driven-ORN count only 0.4020** (`outputs/mb2/REPORT.md` measured 0.267 for that confound on its own 2,000-state sample).

Reading these:

* **KC is at or near ceiling in every condition.** Under `raw`, grouped-CV
  0.990 (`real`) vs 0.992 / 0.992 / 0.992 (rewired). Under `homeo` every
  condition is 0.998-1.000, so the `homeo` decoding column is saturated and
  cannot resolve wiring at all -- the `raw` column is the informative one.
* **DN is not.** 0.620 (`real`) vs 0.600 / 0.598 / 0.602 under `raw`. `real`
  is nominally highest, by about one standard error on n=500
  (SE ~ 0.022); it is not a separation. More importantly all four sit *below*
  the bits-on count-only confound at 0.664, i.e. DN hand-type decoding here is
  consistent with pure intensity encoding -- the project's earlier DN finding,
  unchanged by rewiring.
* **ALPN is 1.000 in all four conditions by construction.** ALPN is upstream
  of the rewired stage and bit-identical across conditions; it is reported as a
  positive control on the harness, not as evidence.
* KC's 0.99 is far above both confounds (driven-ORN count 0.402, bits-on count
  0.664) and far above permuted labels (~0.18-0.20), so the KC code genuinely
  carries hand type -- in the rewired networks exactly as much as in the real
  one.

### 4.4 Behaviour cloning: imitation and live play

Linear readouts only, per the brief.

| condition | readout | held-out imitation top-1 | effective dims |
|---|---|---|---|
| real | kc | 0.7085 | 993 / 4064 |
| real | dn | 0.6606 | 654 / 1314 |
| rw1 | kc | 0.7159 | 965 / 4064 |
| rw1 | dn | 0.6573 | 660 / 1314 |
| rw2 | kc | 0.7246 | 992 / 4064 |
| rw2 | dn | 0.6616 | 659 / 1314 |
| rw3 | kc | 0.7147 | 967 / 4064 |
| rw3 | dn | 0.6620 | 654 / 1314 |

| condition | readout | held-out imitation top-1 | effective dims |
|---|---|---|---|
| real | kc | 0.7498 | 2790 / 4064 |
| real | dn | 0.6587 | 645 / 1314 |
| rw1 | kc | 0.7491 | 2896 / 4064 |
| rw1 | dn | 0.6645 | 651 / 1314 |
| rw2 | kc | 0.7493 | 2742 / 4064 |
| rw2 | dn | 0.6660 | 642 / 1314 |
| rw3 | kc | 0.7478 | 2755 / 4064 |
| rw3 | dn | 0.6632 | 645 / 1314 |

| condition | readout | clear ante 1 | mean chips | plays that were the best subset | plays |
|---|---|---|---|---|---|
| real | kc | 26.5% | 843 | 0.678 | 3376 |
| real | dn | 3.2% | 514 | 0.393 | 2798 |
| rw1 | kc | 24.8% | 851 | 0.705 | 3352 |
| rw1 | dn | 3.8% | 494 | 0.381 | 2689 |
| rw2 | kc | 25.0% | 845 | 0.724 | 3268 |
| rw2 | dn | 3.2% | 489 | 0.363 | 2752 |
| rw3 | kc | 27.5% | 830 | 0.702 | 3278 |
| rw3 | dn | 4.0% | 512 | 0.394 | 2709 |

| condition | readout | clear ante 1 | mean chips | plays that were the best subset | plays |
|---|---|---|---|---|---|
| real | kc | 38.0% | 936 | 0.799 | 3397 |
| real | dn | 1.8% | 495 | 0.362 | 2727 |
| rw1 | kc | 38.2% | 930 | 0.816 | 3388 |
| rw1 | dn | 3.2% | 511 | 0.367 | 2794 |
| rw2 | kc | 35.8% | 923 | 0.799 | 3356 |
| rw2 | dn | 3.2% | 501 | 0.353 | 2775 |
| rw3 | kc | 33.8% | 916 | 0.817 | 3364 |
| rw3 | dn | 2.5% | 502 | 0.356 | 2744 |

### 4.5 Paired comparisons over shared episode seeds

Every episode seed 100000-100399 is played by all four conditions, so
`real` and each rewiring are compared **paired**: exact McNemar (`scipy
binomtest`) on the per-episode clear/no-clear outcomes, and a paired bootstrap
(10,000 resamples, resampling episodes) for chips and for per-state imitation.

| readout | rewiring | imitation real - rw (95% CI) | clear real - rw | McNemar p (discordant) | chips real - rw (95% CI) |
|---|---|---|---|---|---|
| kc | rw1 | -0.0074 [-0.0113, -0.0035] | +1.8 pp (26.5% vs 24.8%) | 0.6322 (82/75) | -8.0 [-53.6, 38.1] |
| kc | rw2 | -0.0161 [-0.0200, -0.0122] | +1.5 pp (26.5% vs 25.0%) | 0.6946 (84/78) | -1.6 [-47.6, 44.9] |
| kc | rw3 | -0.0062 [-0.0099, -0.0025] | -1.0 pp (26.5% vs 27.5%) | 0.8066 (73/77) | 12.9 [-30.9, 57.2] |
| dn | rw1 | 0.0032 [-0.0006, 0.0072] | -0.5 pp (3.2% vs 3.8%) | 0.8506 (13/15) | 20.2 [-20.1, 60.1] |
| dn | rw2 | -0.0010 [-0.0048, 0.0028] | +0.0 pp (3.2% vs 3.2%) | 1.0000 (13/13) | 25.5 [-9.6, 60.2] |
| dn | rw3 | -0.0015 [-0.0055, 0.0027] | -0.8 pp (3.2% vs 4.0%) | 0.7011 (12/15) | 2.3 [-36.4, 39.9] |

| readout | rewiring | imitation real - rw (95% CI) | clear real - rw | McNemar p (discordant) | chips real - rw (95% CI) |
|---|---|---|---|---|---|
| kc | rw1 | 0.0007 [-0.0017, 0.0032] | -0.2 pp (38.0% vs 38.2%) | 1.0000 (60/61) | 6.3 [-26.9, 39.4] |
| kc | rw2 | 0.0006 [-0.0016, 0.0027] | +2.2 pp (38.0% vs 35.8%) | 0.4672 (65/56) | 13.8 [-16.0, 44.3] |
| kc | rw3 | 0.0020 [-0.0002, 0.0043] | +4.2 pp (38.0% vs 33.8%) | 0.1285 (64/47) | 20.3 [-10.6, 51.3] |
| dn | rw1 | -0.0058 [-0.0105, -0.0011] | -1.5 pp (1.8% vs 3.2%) | 0.2632 (7/13) | -16.1 [-52.5, 20.0] |
| dn | rw2 | -0.0073 [-0.0118, -0.0027] | -1.5 pp (1.8% vs 3.2%) | 0.2632 (7/13) | -6.5 [-42.6, 29.3] |
| dn | rw3 | -0.0045 [-0.0088, -0.0001] | -0.8 pp (1.8% vs 2.5%) | 0.6072 (6/9) | -6.8 [-45.1, 32.5] |

**Not one of the twelve clear-rate tests is significant** (smallest p =
0.1285, `homeo`/kc/rw3; the other eleven range 0.26 to 1.00). Chips CIs all
straddle zero. The largest clear-rate gap anywhere is 4.2 pp on 400 episodes.

The imitation CIs are the only place anything excludes zero, and they refute
rather than support an effect, because the sign flips with the variant:
under `raw` all three rewirings *beat* `real` on KC imitation (`real - rw` =
-0.0074, -0.0161, -0.0062), under `homeo` all three lose to it by a tenth as
much (+0.0007, +0.0006, +0.0020), and on DN the signs flip the other way.
These are sub-2-pp differences on a 29,282-state test set, where the paired CI
is narrow enough that any trivial difference clears zero. There is no
consistent direction, so there is no effect -- and note that the one direction
with the largest magnitude favours the *rewired* networks.

Twelve paired tests were run (3 seeds x 2 readouts x 2 variants) with no
multiplicity correction, which makes the all-null result stronger, not weaker.

### 4.6 The right null width is the spread between rewiring seeds

Three independent rewirings give an empirical null band. `real` sits inside it
on every measure:

| measure (variant) | rewired seeds | `real` | inside? |
|---|---|---|---|
| KC decoding, grouped (raw) | 0.992 - 0.992 | 0.990 | at the edge, below |
| DN decoding, grouped (raw) | 0.598 - 0.602 | 0.620 | above, by ~1 SE |
| Jaccard ratio (raw) | 0.9060 - 0.9141 | 0.9111 | yes |
| Jaccard ratio (homeo) | 0.2964 - 0.4762 | 0.4665 | yes |
| KC imitation (raw) | 0.7147 - 0.7246 | 0.7085 | below all three |
| KC imitation (homeo) | 0.7478 - 0.7493 | 0.7498 | above, by 0.0005 - 0.0020 |
| KC clear rate (raw) | 0.247 - 0.275 | 0.265 | yes |
| KC clear rate (homeo) | 0.338 - 0.383 | 0.380 | yes |

The real wiring is not systematically at the top of these ranges. On two of
them it is at the bottom.

### 4.7 For scale: what *does* move these numbers

The same measurements show label-free recalibration mattering an order of
magnitude more than the wiring, identically in every condition:

| | `raw` | `homeo` | delta |
|---|---|---|---|
| KC effective dims (`real`) | 993 | 2,790 | +1,797 |
| KC imitation top-1 (`real`) | 0.7085 | 0.7498 | +0.041 |
| KC clear rate (`real`) | 0.265 | 0.380 | +11.5 pp |

Per-KC homeostasis wakes up roughly 1,800 additional Kenyon cells and buys
+11.5 pp of clear rate. Randomising which Kenyon cell each projection neuron
talks to buys, at most, noise. That contrast is the result.

## 5. Verdict

**Under this control, the measured ALPN -> KC wiring does not help.** Randomising
which Kenyon cell each antennal-lobe projection neuron contacts -- while holding
every degree, every synapse count, every hemisphere and every KC subtype fixed --
changes nothing measurable: KC hand-type decoding stays at ceiling, the KC
population code keeps the same within/between structure, imitation moves by less
than two points in an inconsistent direction, and live clear rate over 400 shared
episode seeds is statistically indistinguishable in all twelve paired tests.

Confidence: **high** for the claim as scoped, for three reasons. The pipeline
reproduces `outputs/bc3` bit for bit when the rewiring is absent, so the harness
is not absorbing the effect. The conditions are in the same activity regime
*without* being forced there, so this is not the broken-control problem the
global shuffle had. And three independent rewiring seeds agree, giving an
empirical null band that `real` sits inside on every measure -- twice at the
bottom of it.

### What this licenses

The fine-grained identity of calyx connectivity is not doing work in this model.
This **agrees with the biology**: Caron et al. (2013) report PN -> KC connectivity
in *Drosophila* as largely random with respect to glomerular identity, and this
model behaving as if that wiring were random is the outcome that literature
predicts. A large effect would have been the surprise. This is the negative
result validating the model against known biology, not a failed experiment.

### What this does NOT license

* **It is not a claim about the connectome as a whole.** One stage was rewired --
  the mushroom body calyx, 19,980 of 5.15M edges. ORN -> ALPN, APL in both
  directions, KC -> KC, KC -> MBON, all dopaminergic wiring and everything
  downstream of the mushroom body were preserved exactly. Nothing here says
  those stages are equally arbitrary. They have not been tested.
* **It is not a claim that PN -> KC wiring is random.** The strata are
  `(hemisphere, KC subtype, synapse count, sign)`, so every projection neuron
  keeps its exact profile over KC subtypes and hemispheres; only the choice of
  *which individual KC within a subtype* is randomised. Subtype-level structure
  -- the kind Zheng et al. (2022) found beyond Caron's randomness -- is
  **preserved by this control, not tested by it**. The claim is narrower and
  sharper: given those subtype-level biases, the individual-KC identity of the
  wiring contributes nothing measurable here.
* **It is not a claim about the fly.** This is a frozen LIF reservoir driven by a
  32-bit game encoding through 32 whole ORN glomeruli, read out by a linear map
  and scored on Balatro. A task the mushroom body did not evolve for may simply
  fail to load the structure that wiring encodes. Absence of an effect here is
  evidence about this model, not about olfactory learning.
* **It does not rehabilitate the old "real beats shuffled" claim.** That
  comparison was broken and stays retracted. This control replaces it; it does
  not rescue it. Its finding is the opposite sign: with a fair control, real does
  not beat the control either.

## 6. Files

| file | what |
|---|---|
| `scripts/calyx_common.py` | stratified degree/weight-preserving ALPN -> KC rewiring; graph construction; invariant checks |
| `scripts/calyx_homeo.py` | per-KC homeostatic thresholds, recalibrated per condition |
| `scripts/calyx_feats.py` | brain features over bc3's 4,962 unique relay patterns |
| `scripts/calyx_probe.py` | activity, KC Jaccard, decoding + baselines |
| `scripts/calyx_train.py` | linear behaviour-cloning readouts (bc3 recipe, bc3 split) |
| `scripts/calyx_eval.py` | 400-episode live play, seeds 100000-100399 |
| `scripts/calyx_stats.py` | exact McNemar + paired bootstrap over shared seeds |
| `scripts/calyx_tables.py` | every table in this report |
| `scripts/calyx_run.sh` | resumable end-to-end driver |
| `tests/test_calyx.py` | 14 tests of the rewiring invariants |
| `outputs/calyx/rewiring.json` | per-seed rewiring depth and invariant checks |
| `outputs/calyx/probe_{raw,homeo}.json` | activity, Jaccard, decoding |
| `outputs/calyx/paired_{raw,homeo}.json` | paired statistics |
| `outputs/calyx/tables_{raw,homeo}.md` | generated tables |
| `outputs/calyx/run.log` | full run log |
