# Glomerular encoding: a gated test (2026-09-13)

`outputs/mb/REPORT.md` ended on a diagnosis, not a fix. At `apl_scale = 2` the
Kenyon cells were sparse (7.8% active) but not separable: 92% of the active cells
fired for every input, KC identity decoding sat at chance, and the reason was
upstream of the mushroom body. With the v1/v2 encoding, 36 active bits x 10
randomly scattered ORNs put ~360 receptor neurons into **all 54 glomeruli on
every input**, so the total ALPN spike count had a coefficient of variation of
0.031 across inputs. A mushroom body cannot make a discriminative sparse code out
of an input in which nothing is off.

This is the gated test of the fix: one relay bit = one whole ORN type. No
plasticity, no pipeline rebuild, no new knobs.

**Result in one line.** The fix works and it was entirely an encoding problem.
The 9-way hand-type label is now decodable from the Kenyon cells at **1.000**
(leakage-free grouped CV 0.995) against a permuted-label baseline of 0.108 and a
total-drive-only confound baseline of 0.267, where the previous round measured
0.598 against 0.573. **Zero of the twelve settings pass the four-criterion gate**,
all of them on the Jaccard criterion alone, and section 9 argues that criterion is
measuring the wrong thing here.

---

## 1. Method

Everything not listed here is unchanged from `RESULTS.md` and
`outputs/mb/REPORT.md`: MaleCNS v1.0 brain-only, 146,271 neurons, 5.15M edges at
>= 5 synapses, sign from neurotransmitter prediction, `W_SYN = 0.275` mV,
event-driven LIF (Shiu et al. 2024 constants, dt 0.1 ms), `v_jitter` 6.9 mV,
brain seed 1, stateless one 50 ms window from reset per state, readout on
`log1p` spike counts.

What changed:

* **Input.** `flybalatro.encode.GlomerularMap`: binary feature `f` drives *every*
  olfactory receptor neuron of one `ORN_<glomerulus>` type and nothing else.
  Groups are ragged (34-204 cells), which is why it is a new class rather than
  another `grouping=` value on `FeatureMap`.
* **Features.** Only the 32-bit relay block of `features_v2`. **The 283 v1 state
  bits are dropped from the fly entirely.**
* **States.** 2,000 real Balatro states sampled from `outputs/bc2/states.npz`
  (150,037 rows, `feature_version 2`), stratified by hand type.
* **Window split.** Two `step` calls per window (15 ms, then 35 ms with
  `reset_counts=False`) so a `t >= 15 ms` readout falls out for every population,
  not only for KCs.
* **Drive.** Tonic 30 mV as before, or Poisson via `Brain.set_poisson` at 100 Hz
  with `kick_mv = 68.75` (= 0.275 x 250, the value `scripts/brain_sweep.py`
  already used). The numba RNG is reseeded per state (`_seed_rng(5000 + i)`) so
  the Poisson realisation is identical across settings and reproducible.

Scripts: `scripts/mb2_assign.py` (assignment), `scripts/mb2_sample.py` (sample +
input references), `scripts/mb2_grid.py` (simulation), `scripts/mb2_probe.py`
(probes), `scripts/mb2_gate.py` (gate + pick), `scripts/mb2_mn9.py` (MN9 seeds).
`outputs/bc*/` and `outputs/mb/` are untouched.

---

## 2. The glomerulus assignment

The 32 largest `ORN_*` types out of 53 (the 4 olfactory neurons with an empty
`type` string are not assignable and are excluded). 2,075 of 2,639 ORNs are
mapped; 564 in the 21 smallest glomeruli are never driven by anything.

**Which glomerulus gets which bit is a confound decision, not bookkeeping.** The
previous round established that this mushroom body is an excellent total-drive
meter (KC decoded "how many bits were on" at 0.83 against 0.59 for identity).
Glomeruli here span 34 to 204 cells, so a careless assignment lets a 9-way probe
succeed by reading ORN *count*. Two rules:

1. The nine `best_<hand type>` bits -- the 9-way label -- get the nine **smallest
   and tightest** types, 34-36 cells, so which hand type is active changes the
   total drive by at most 2 ORNs.
2. The remaining 23 types are dealt largest-first to `best_vs_needed` (4),
   `sel_<type>` (10), `sel_is_best` (1), `best_slot` (8). That order was picked by
   scoring **all 24 block orders** on how well the driven-ORN count *alone* (one
   feature, no brain) predicts each label. The chosen order is best on all three
   summary numbers at once: count-only hand type 0.267 (worst order 0.492),
   count-only `sel_is_best` 0.890 = exactly the majority floor, i.e. no leakage,
   and the smallest driven-ORN spread (SD 50.3).

| bit | feature | glomerulus | ORNs | block |
|---|---|---|---|---|
| 0 | `best_high_card` | `ORN_VM7d` | 36 | best_type |
| 1 | `best_pair` | `ORN_DA4m` | 35 | best_type |
| 2 | `best_two_pair` | `ORN_DC3` | 35 | best_type |
| 3 | `best_three_of_a_kind` | `ORN_DM5` | 35 | best_type |
| 4 | `best_straight` | `ORN_DA3` | 34 | best_type |
| 5 | `best_flush` | `ORN_VA4` | 34 | best_type |
| 6 | `best_full_house` | `ORN_VC3` | 34 | best_type |
| 7 | `best_four_of_a_kind` | `ORN_VM5v` | 34 | best_type |
| 8 | `best_straight_flush` | `ORN_VM6v` | 34 | best_type |
| 9 | `best_slot0` | `ORN_V` | 55 | best_mask |
| 10 | `best_slot1` | `ORN_DM2` | 54 | best_mask |
| 11 | `best_slot2` | `ORN_DA2` | 48 | best_mask |
| 12 | `best_slot3` | `ORN_VL2p` | 45 | best_mask |
| 13 | `best_slot4` | `ORN_DL5` | 43 | best_mask |
| 14 | `best_slot5` | `ORN_VM3` | 43 | best_mask |
| 15 | `best_slot6` | `ORN_VM2` | 41 | best_mask |
| 16 | `best_slot7` | `ORN_VC4` | 38 | best_mask |
| 17 | `sel_high_card` | `ORN_VL2a` | 98 | selected_type |
| 18 | `sel_pair` | `ORN_VM5d` | 84 | selected_type |
| 19 | `sel_two_pair` | `ORN_DL1` | 83 | selected_type |
| 20 | `sel_three_of_a_kind` | `ORN_VA2` | 83 | selected_type |
| 21 | `sel_straight` | `ORN_VL1` | 82 | selected_type |
| 22 | `sel_flush` | `ORN_VM4` | 78 | selected_type |
| 23 | `sel_full_house` | `ORN_DM1` | 74 | selected_type |
| 24 | `sel_four_of_a_kind` | `ORN_DM3` | 63 | selected_type |
| 25 | `sel_straight_flush` | `ORN_VA6` | 63 | selected_type |
| 26 | `sel_none` | `ORN_DL4` | 62 | selected_type |
| 27 | `sel_is_best` | `ORN_DM6` | 58 | selected_is_best |
| 28 | `best_vs_needed_lt0.25` | `ORN_DA1` | 204 | score_vs_needed |
| 29 | `best_vs_needed_lt0.5` | `ORN_VA1d` | 132 | score_vs_needed |
| 30 | `best_vs_needed_lt1.0` | `ORN_VA1v` | 130 | score_vs_needed |
| 31 | `best_vs_needed_ge1.0` | `ORN_DL3` | 103 | score_vs_needed |

Per-block cell-count ranges: `best_type` 34-36, `best_mask` 38-55,
`selected_type` 62-98, `selected_is_best` 58, `score_vs_needed` 103-204.
Dropped (never driven): `ORN_DA4l`, `ORN_DP1l`, `ORN_VM1`, `ORN_DC1`, `ORN_DM4`,
`ORN_VC2`, `ORN_DP1m`, `ORN_VC5`, `ORN_VA3`, `ORN_VA7l`, `ORN_VC1`, `ORN_D`,
`ORN_VM6m`, `ORN_VM7v`, `ORN_VA7m`, `ORN_DC4`, `ORN_DL2v`, `ORN_DC2`, `ORN_VA5`,
`ORN_DL2d`, `ORN_VM6l`.

**One confound cannot be removed inside this task.** `best_slot` has exactly 1-5
bits on and *how many* is fixed by the hand type (High Card 1, Pair 2, Trips 3,
Two Pair 4, Straight / Flush / Full House 5). Total driven-ORN count therefore
carries label information by construction, which is why the count-only probe is
reported in section 4 and why it -- not the gate threshold -- is the bar.

### Active glomeruli per state (n = 2,000)

| metric | value |
|---|---|
| active glomeruli, mean (SD) | **6.83** (1.47) |
| min / max | 4 / 9 |
| histogram (4/5/6/7/8/9) | 220 / 235 / 212 / 450 / 767 / 116 |
| silent glomeruli of 32, mean | 25.17 |
| driven ORNs, mean (SD) | 441.2 (50.3) |
| driven ORNs, min / max | 242 / 582 |
| measured driven-ORN rate, tonic 30 mV | ~127 Hz (21.3 Hz over all 2,639 ORNs) |
| measured driven-ORN rate, Poisson 100 Hz | ~84 Hz (14.0 Hz over all 2,639 ORNs) |

Exactly the intended regime: 4-9 of 32 channels on, 23-28 off. Against the old
encoding's "all 54 glomeruli on every input", that is the whole intervention.

### Sample and labels

2,000 states drawn from the 138,337 rows whose relay block is non-silent (the
block is silent outside a blind, stages 1-3). Stratified **by hand type only**,
with water-filled equal quotas: 233 each for eight types and 136 for Straight
Flush, which has only 136 rows in the whole corpus. 1,169 distinct relay patterns
among the 2,000 states.

| label | definition | floors |
|---|---|---|
| hand type | 9-way, `argmax` of bits 0-8 | majority 0.1165, permuted ~0.11 |
| `sel_is_best` | binary, bit 27 (verified against `FEATURE_NAMES_V2`) | majority **0.890**, 220 positives (11.0%) |
| intensity (control) | driven-ORN count above the median | 0.4975 positive |

`sel_is_best` is naturally 11% positive and was **not** balanced: balancing it
would have been off-spec and would also have been a confound, because
`sel_is_best = 1` forces `sel_type = best_type` and duplicates the hand-type
glomerulus. Accuracy for that label is quoted next to the 0.890 floor, and a
balanced accuracy is recorded in `probes.json`.

---

## 3. The gate

Fixed by the task, applied mechanically in `scripts/mb2_gate.py`:

1. KC active-set Jaccard, between hand types / within hand type, **< 0.5**
2. KC always-on fraction (fires for > 50% of states) **< 0.05**
3. KC 9-way hand-type accuracy **>= permuted + 0.15**
4. MBON mean rate **< 30 Hz** and **>= 20** MBONs varying across states

Ranking, also fixed before the grid finished: criteria passed, then the KC
hand-type margin over permuted (grouped CV), then the lower Jaccard ratio. The
`kc_input_norm` retry is excluded from the pick by construction.

---

## 4. What the input is worth without a brain

`outputs/mb2/input_stats.json`. These are the numbers every brain readout has to
beat.

| features | hand type | `sel_is_best` | intensity |
|---|---|---|---|
| the 32 relay bits | **1.000** (perm 0.111) | **1.000** (perm 0.890) | 0.990 |
| bits-on count only (1 feature) | 0.547 | 0.890 | 0.842 |
| **driven-ORN count only (1 feature)** | **0.267** | **0.890** | 1.000 |

The first row is the glomerular ceiling, and it is *perfect*: with one whole ORN
type per bit there is one analog channel per bit, so nothing is lost at the
receptors by construction. That is the structural difference from the old
encoding, whose ceiling probe was 0.775.

**The third row is the real bar for criterion 3.** Total driven-ORN count alone
gives 0.267 on the 9-way label, while criterion 3 asks only for permuted + 0.15
~= 0.26. Every non-degenerate setting clears the gate threshold trivially; what
matters is whether it clears 0.267, and the chosen setting clears it by 0.73.

### Input-level Jaccard, which is how to read criterion 1

| sets | within hand type | between hand types | ratio |
|---|---|---|---|
| the 32 relay bits | 0.4331 | 0.2099 | **0.4846** |
| the driven-ORN sets | 0.4286 | 0.2160 | **0.5040** |

The input itself sits at 0.48-0.50, i.e. right *on* the threshold. Criterion 1 is
not asking the mushroom body to preserve a separation; it is asking it to be at
least as good as its own receptors on a metric the receptors barely pass. Any
blurring at all fails it.

---

## 5. The grid -- 12 settings, 2,000 states each

`apl_scale` {1, 2, 5} x drive {tonic 30 mV, Poisson ~100 Hz} x
`kc_vth_offset_mv` {0, 4}. Rows in ranking order; the `..._norm1` row is the
`kc_input_norm` retry from section 7. `a/a` = always-on / active, the diagnostic
that decided the previous round. "KC input-dep" = KCs whose per-cell active rate
is strictly between 0 and 1. Probe cells are `accuracy (grouped CV)`; "perm" is
the permuted-label accuracy. `gate` digits are criteria 1/2/3/4.

| setting | KC active | always-on | a/a | KC input-dep | J within | J between | J b/w | MBON Hz | MBON vary | ALPN Hz | ALPN CV | KC hand (grp) | perm | KCbin hand | KC intensity | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `apl2_tonic_vth0` | 0.0482 | 0.0406 | 0.842 | 838 (0.2062) | 0.6071 | 0.5337 | **0.8792** | 8.61 | 45 | 71.18 | 0.0284 | 1.0 (0.995) | 0.1075 | 1.0 (0.9945) | 0.959 | fail 0111 |
| `apl2_poisson_vth0` | 0.0450 | 0.0384 | 0.853 | 829 (0.204) | 0.6161 | 0.5611 | **0.9108** | 8.19 | 45 | 68.64 | 0.0252 | 0.995 (0.991) | 0.1115 | 0.989 (0.986) | 0.926 | fail 0111 |
| `apl5_tonic_vth0` | 0.0059 | 0.0020 | 0.339 | 415 (0.1021) | 0.2875 | 0.1691 | **0.5881** | 5.91 | 35 | 67.17 | 0.0314 | 0.984 (0.968) | 0.114 | 0.9845 (0.9685) | 0.936 | fail 0111 |
| `apl2_tonic_vth4` | 0.0227 | 0.0202 | 0.89 | 396 (0.0974) | 0.6077 | 0.5426 | **0.8929** | 8.06 | 29 | 71.36 | 0.0283 | 0.9725 (0.938) | 0.1165 | 0.9615 (0.913) | 0.9165 | fail 0111 |
| `apl2_poisson_vth4` | 0.0212 | 0.0189 | 0.892 | 350 (0.0861) | 0.6193 | 0.5707 | **0.9214** | 7.78 | 28 | 68.80 | 0.0251 | 0.9235 (0.911) | 0.1185 | 0.9105 (0.897) | 0.8635 | fail 0111 |
| `apl5_poisson_vth0` | 0.0047 | 0.0015 | 0.319 | 429 (0.1056) | 0.2565 | 0.1638 | **0.6385** | 5.73 | 35 | 64.49 | 0.0283 | 0.883 (0.8565) | 0.105 | 0.8835 (0.859) | 0.897 | fail 0111 |
| `apl2_tonic_vth0_norm1` | 0.0656 | 0.0605 | 0.922 | 892 (0.2195) | 0.6625 | 0.5871 | **0.8862** | 8.05 | 45 | 71.17 | 0.0283 | 1.0 (0.9995) | 0.1005 | 0.999 (0.999) | 0.9645 | fail 0011 |
| `apl1_tonic_vth4` | 0.2034 | 0.1922 | 0.945 | 1369 (0.3369) | 0.7645 | 0.7096 | **0.9282** | 25.50 | 64 | 72.49 | 0.0274 | 1.0 (0.9985) | 0.102 | 0.999 (0.9975) | 0.9685 | fail 0011 |
| `apl1_poisson_vth4` | 0.1920 | 0.1843 | 0.96 | 1268 (0.312) | 0.7714 | 0.7279 | **0.9436** | 23.71 | 63 | 69.92 | 0.0243 | 0.996 (0.997) | 0.1155 | 0.9925 (0.992) | 0.942 | fail 0011 |
| `apl5_poisson_vth4` | 0.0003 | 0.0002 | 0.667 | 12 (0.003) | 0.9135 | 0.9144 | **1.001** | 5.67 | 26 | 64.56 | 0.0285 | 0.1325 (0.135) | 0.1145 | 0.133 (0.1325) | 0.519 | fail 0101 |
| `apl5_tonic_vth4` | 0.0002 | 0.0002 | 1.0 | 8 (0.002) | 0.8934 | 0.8936 | **1.0002** | 5.86 | 26 | 67.28 | 0.0315 | 0.1355 (0.1285) | 0.12 | 0.1295 (0.121) | 0.509 | fail 0101 |
| `apl1_tonic_vth0` | 0.3421 | 0.3337 | 0.975 | 1499 (0.3688) | 0.8362 | 0.788 | **0.9424** | 48.54 | 71 | 72.32 | 0.0274 | 1.0 (1.0) | 0.106 | 0.9995 (1.0) | 0.9775 | fail 0010 |
| `apl1_poisson_vth0` | 0.3313 | 0.3233 | 0.976 | 1457 (0.3585) | 0.8405 | 0.7999 | **0.9517** | 46.80 | 71 | 69.75 | 0.0246 | 0.999 (0.999) | 0.111 | 0.997 (0.9965) | 0.9505 | fail 0010 |

ALPN mean rate stays within 4-13% of the untuned 72.3 Hz in every row, so nothing
here is perturbing the antennal lobe rather than the calyx. ALPN total-count CV
stays at 0.024-0.032 -- the same near-constant *total* as the old encoding -- which
is a measure that says nothing about separability; see section 9.

### Gate verdict, explicitly, per setting

| setting | crit 1 Jaccard < 0.5 | crit 2 always-on < 0.05 | crit 3 margin >= 0.15 | crit 4 MBON | verdict |
|---|---|---|---|---|---|
| `apl2_tonic_vth0` | 0.8792 **fail** | 0.0406 pass | 0.8925 pass | 8.6 Hz / 45 pass | **FAIL** (3/4) |
| `apl2_poisson_vth0` | 0.9108 **fail** | 0.0384 pass | 0.8835 pass | 8.2 Hz / 45 pass | **FAIL** (3/4) |
| `apl5_tonic_vth0` | 0.5881 **fail** | 0.0020 pass | 0.8700 pass | 5.9 Hz / 35 pass | **FAIL** (3/4) |
| `apl2_tonic_vth4` | 0.8929 **fail** | 0.0202 pass | 0.8560 pass | 8.1 Hz / 29 pass | **FAIL** (3/4) |
| `apl2_poisson_vth4` | 0.9214 **fail** | 0.0189 pass | 0.8050 pass | 7.8 Hz / 28 pass | **FAIL** (3/4) |
| `apl5_poisson_vth0` | 0.6385 **fail** | 0.0015 pass | 0.7780 pass | 5.7 Hz / 35 pass | **FAIL** (3/4) |
| `apl1_tonic_vth4` | 0.9282 **fail** | 0.1922 **fail** | 0.8980 pass | 25.5 Hz / 64 pass | **FAIL** (2/4) |
| `apl1_poisson_vth4` | 0.9436 **fail** | 0.1843 **fail** | 0.8805 pass | 23.7 Hz / 63 pass | **FAIL** (2/4) |
| `apl5_poisson_vth4` | 1.0010 **fail** | 0.0002 pass | 0.0180 **fail** | 5.7 Hz / 26 pass | **FAIL** (2/4) |
| `apl5_tonic_vth4` | 1.0002 **fail** | 0.0002 pass | 0.0155 **fail** | 5.9 Hz / 26 pass | **FAIL** (2/4) |
| `apl1_tonic_vth0` | 0.9424 **fail** | 0.3337 **fail** | 0.8940 pass | 48.5 Hz / 71 **fail** | **FAIL** (1/4) |
| `apl1_poisson_vth0` | 0.9517 **fail** | 0.3233 **fail** | 0.8880 pass | 46.8 Hz / 71 **fail** | **FAIL** (1/4) |
| `apl2_tonic_vth0_norm1` (retry) | 0.8862 **fail** | 0.0605 **fail** | 0.8995 pass | 8.1 Hz / 45 pass | **FAIL** (2/4) |

**0 of 12 settings pass all four criteria.** Per the task that is the result, and
no knob outside the named ones was added. Every setting fails criterion 1; six
fail *only* criterion 1. Section 9 is why that is not the same statement as
"the code is not discriminative".

The two `apl5 + vth4` rows are degenerate and their Jaccard, always-on and `a/a`
numbers are computed over a handful of cells: 0.8 active KCs per state, 8 cells
with any input-dependence at all. Same artefact `outputs/mb/REPORT.md` flagged.

---

## 6. Probe tables

Accuracy `full window / t >= 15 ms (permuted)`. `KCbin` is the KC matrix
binarised to active/inactive -- it separates "the graded rates are informative"
from "the active set is informative".

### hand type (9-way; majority 0.1165, count-only confound 0.267)

| setting | KC | KCbin | MBON | ALPN | DN |
|---|---|---|---|---|---|
| `apl2_tonic_vth0` | 1.0 / 0.998 (0.1075) | 1.0 / 0.9945 (0.108) | 0.511 / 0.5155 (0.1155) | 1.0 / 1.0 (0.0945) | 0.8415 / 0.8495 (0.1135) |
| `apl2_poisson_vth0` | 0.995 / 0.985 (0.1115) | 0.989 / 0.9715 (0.1035) | 0.394 / 0.3825 (0.111) | 1.0 / 1.0 (0.107) | 0.597 / 0.596 (0.121) |
| `apl5_tonic_vth0` | 0.984 / 0.201 (0.114) | 0.9845 / 0.185 (0.1155) | 0.466 / 0.4595 (0.1145) | 1.0 / 1.0 (0.1015) | 0.843 / 0.838 (0.118) |
| `apl2_tonic_vth4` | 0.9725 / 0.9685 (0.1165) | 0.9615 / 0.9615 (0.114) | 0.4465 / 0.441 (0.111) | 1.0 / 1.0 (0.105) | 0.839 / 0.8425 (0.1135) |
| `apl2_poisson_vth4` | 0.9235 / 0.92 (0.1185) | 0.9105 / 0.9115 (0.117) | 0.3725 / 0.3435 (0.111) | 1.0 / 1.0 (0.1235) | 0.608 / 0.6015 (0.127) |
| `apl5_poisson_vth0` | 0.883 / 0.19 (0.105) | 0.8835 / 0.1665 (0.1035) | 0.367 / 0.359 (0.1075) | 1.0 / 1.0 (0.1175) | 0.5885 / 0.5825 (0.119) |
| `apl2_tonic_vth0_norm1` | 1.0 / 0.995 (0.1005) | 0.999 / 0.9775 (0.104) | 0.512 / 0.485 (0.123) | 1.0 / 1.0 (0.1115) | 0.853 / 0.8435 (0.1135) |
| `apl1_tonic_vth4` | 1.0 / 1.0 (0.102) | 0.999 / 0.999 (0.1035) | 0.5955 / 0.5895 (0.13) | 1.0 / 1.0 (0.1035) | 0.852 / 0.8525 (0.107) |
| `apl1_poisson_vth4` | 0.996 / 0.997 (0.1155) | 0.9925 / 0.9925 (0.1195) | 0.4475 / 0.445 (0.1155) | 1.0 / 1.0 (0.1115) | 0.6135 / 0.6055 (0.109) |
| `apl5_poisson_vth4` | 0.1325 / 0.1265 (0.1145) | 0.133 / 0.1265 (0.1145) | 0.346 / 0.335 (0.1125) | 1.0 / 1.0 (0.119) | 0.5975 / 0.5875 (0.1195) |
| `apl5_tonic_vth4` | 0.1355 / 0.1345 (0.12) | 0.1295 / 0.1315 (0.12) | 0.475 / 0.4535 (0.106) | 1.0 / 1.0 (0.103) | 0.848 / 0.852 (0.0985) |
| `apl1_tonic_vth0` | 1.0 / 1.0 (0.106) | 0.9995 / 0.9995 (0.1075) | 0.6395 / 0.644 (0.1245) | 1.0 / 1.0 (0.1085) | 0.855 / 0.855 (0.109) |
| `apl1_poisson_vth0` | 0.999 / 0.9985 (0.111) | 0.997 / 0.9955 (0.108) | 0.4705 / 0.4625 (0.1115) | 1.0 / 1.0 (0.123) | 0.6255 / 0.6225 (0.1065) |

### `sel_is_best` (binary; majority 0.890, count-only confound 0.890)

| setting | KC | KCbin | MBON | ALPN | DN |
|---|---|---|---|---|---|
| `apl2_tonic_vth0` | 1.0 / 1.0 (0.86) | 1.0 / 1.0 (0.863) | 0.9205 / 0.9135 (0.889) | 1.0 / 1.0 (0.8745) | 0.9625 / 0.959 (0.8755) |
| `apl2_poisson_vth0` | 1.0 / 1.0 (0.861) | 1.0 / 1.0 (0.863) | 0.8905 / 0.888 (0.89) | 1.0 / 1.0 (0.864) | 0.9125 / 0.9105 (0.8715) |
| `apl5_tonic_vth0` | 1.0 / 0.9015 (0.8845) | 1.0 / 0.899 (0.885) | 0.8905 / 0.8895 (0.89) | 1.0 / 1.0 (0.8725) | 0.96 / 0.9545 (0.8775) |
| `apl2_tonic_vth4` | 1.0 / 1.0 (0.876) | 1.0 / 1.0 (0.8755) | 0.8885 / 0.89 (0.89) | 1.0 / 1.0 (0.876) | 0.964 / 0.9575 (0.878) |
| `apl2_poisson_vth4` | 0.998 / 0.998 (0.875) | 0.997 / 0.997 (0.8755) | 0.889 / 0.8895 (0.89) | 1.0 / 1.0 (0.867) | 0.9115 / 0.9125 (0.869) |
| `apl5_poisson_vth0` | 0.9985 / 0.8895 (0.8815) | 0.9985 / 0.8885 (0.8815) | 0.8905 / 0.89 (0.89) | 1.0 / 1.0 (0.87) | 0.915 / 0.911 (0.862) |
| `apl2_tonic_vth0_norm1` | 1.0 / 1.0 (0.868) | 1.0 / 1.0 (0.8695) | 0.9045 / 0.897 (0.8895) | 1.0 / 1.0 (0.8775) | 0.9545 / 0.9535 (0.8775) |
| `apl1_tonic_vth4` | 1.0 / 1.0 (0.864) | 1.0 / 1.0 (0.8615) | 0.9255 / 0.923 (0.89) | 1.0 / 1.0 (0.8775) | 0.96 / 0.9555 (0.8755) |
| `apl1_poisson_vth4` | 1.0 / 1.0 (0.861) | 1.0 / 1.0 (0.851) | 0.9095 / 0.908 (0.8705) | 1.0 / 1.0 (0.8705) | 0.9165 / 0.9105 (0.8715) |
| `apl5_poisson_vth4` | 0.89 / 0.8895 (0.89) | 0.89 / 0.89 (0.89) | 0.89 / 0.89 (0.89) | 1.0 / 1.0 (0.8655) | 0.9195 / 0.9185 (0.867) |
| `apl5_tonic_vth4` | 0.8905 / 0.8905 (0.89) | 0.8905 / 0.8905 (0.89) | 0.89 / 0.89 (0.89) | 1.0 / 1.0 (0.8825) | 0.9605 / 0.959 (0.876) |
| `apl1_tonic_vth0` | 1.0 / 1.0 (0.8675) | 1.0 / 1.0 (0.8575) | 0.9465 / 0.9465 (0.8905) | 1.0 / 1.0 (0.871) | 0.9555 / 0.956 (0.8815) |
| `apl1_poisson_vth0` | 1.0 / 1.0 (0.864) | 1.0 / 1.0 (0.8575) | 0.9195 / 0.917 (0.8895) | 1.0 / 1.0 (0.8655) | 0.9135 / 0.911 (0.871) |

The MBON and DN columns for this label are the only place the 0.890 majority floor
bites: `MBON 0.8905` is the floor, not a readout. KC and ALPN reach 1.000; DN
0.91-0.96, i.e. genuinely above floor.

### intensity control (driven-ORN count above median)

| setting | KC | KCbin | MBON | ALPN | DN |
|---|---|---|---|---|---|
| `apl2_tonic_vth0` | 0.959 / 0.9425 (0.5105) | 0.951 / 0.9335 (0.5035) | 0.7505 / 0.7315 (0.502) | 0.9755 / 0.977 (0.5155) | 0.9185 / 0.9195 (0.5345) |
| `apl2_poisson_vth0` | 0.926 / 0.889 (0.504) | 0.917 / 0.8855 (0.521) | 0.7225 / 0.674 (0.5015) | 0.9485 / 0.9435 (0.518) | 0.793 / 0.791 (0.509) |
| `apl5_tonic_vth0` | 0.936 / 0.5335 (0.497) | 0.935 / 0.5345 (0.4935) | 0.728 / 0.7135 (0.516) | 0.977 / 0.969 (0.5325) | 0.903 / 0.9095 (0.5105) |
| `apl2_tonic_vth4` | 0.9165 / 0.915 (0.54) | 0.909 / 0.9095 (0.539) | 0.7315 / 0.704 (0.525) | 0.978 / 0.9745 (0.504) | 0.911 / 0.9185 (0.5425) |
| `apl2_poisson_vth4` | 0.8635 / 0.858 (0.483) | 0.859 / 0.8575 (0.5) | 0.701 / 0.662 (0.5215) | 0.9555 / 0.952 (0.507) | 0.8 / 0.791 (0.503) |
| `apl5_poisson_vth0` | 0.897 / 0.538 (0.5035) | 0.8975 / 0.543 (0.5025) | 0.677 / 0.65 (0.535) | 0.9475 / 0.944 (0.4925) | 0.7785 / 0.7725 (0.517) |
| `apl2_tonic_vth0_norm1` | 0.9645 / 0.9495 (0.5025) | 0.9665 / 0.9425 (0.5145) | 0.7275 / 0.712 (0.535) | 0.9775 / 0.975 (0.5075) | 0.907 / 0.9095 (0.522) |
| `apl1_tonic_vth4` | 0.9685 / 0.9685 (0.522) | 0.962 / 0.962 (0.5285) | 0.7955 / 0.7835 (0.5275) | 0.9775 / 0.972 (0.531) | 0.916 / 0.9115 (0.5165) |
| `apl1_poisson_vth4` | 0.942 / 0.943 (0.4915) | 0.9325 / 0.9315 (0.5055) | 0.732 / 0.7275 (0.5305) | 0.9575 / 0.9505 (0.494) | 0.785 / 0.7895 (0.5235) |
| `apl5_poisson_vth4` | 0.519 / 0.5155 (0.5055) | 0.518 / 0.5145 (0.509) | 0.656 / 0.65 (0.5155) | 0.954 / 0.9515 (0.492) | 0.7815 / 0.7795 (0.5045) |
| `apl5_tonic_vth4` | 0.509 / 0.508 (0.5045) | 0.5105 / 0.5095 (0.509) | 0.729 / 0.7055 (0.516) | 0.9775 / 0.9755 (0.519) | 0.912 / 0.9085 (0.512) |
| `apl1_tonic_vth0` | 0.9775 / 0.974 (0.5085) | 0.97 / 0.97 (0.5165) | 0.8375 / 0.819 (0.501) | 0.974 / 0.974 (0.5115) | 0.924 / 0.9185 (0.529) |
| `apl1_poisson_vth0` | 0.9505 / 0.9485 (0.4935) | 0.93 / 0.924 (0.5105) | 0.761 / 0.763 (0.487) | 0.954 / 0.95 (0.488) | 0.796 / 0.788 (0.511) |

Four things in those tables.

* **ALPN is 1.000 on hand type in every single setting**, which is exactly right:
  no knob in `tuning.py` is upstream of the antennal lobe, and one whole
  glomerulus drives its own dedicated uniglomerular projection neurons. Compare
  0.695 under the old encoding. The bottleneck was the encoding.
* **The mushroom body now preserves that.** KC 1.000 at `apl1`/`apl2` tonic,
  0.984 at `apl5_tonic_vth0` with only 24 active KCs per state.
* **Intensity no longer dominates.** Last round KC decoded intensity at 0.830 and
  identity at 0.598. Intensity is still very decodable (0.959) -- it is not that
  the total-drive signal went away -- but identity now matches or exceeds it
  (1.000), and the identity margin over the count-only confound baseline is +0.73.
* **`t >= 15 ms` matters only at `apl5`.** There KC hand type collapses from 0.984
  to 0.201 -- but the late matrix has **7 usable columns** out of 4,064, so that is
  an empty matrix, not a late readout. Same artefact as last round. At
  `apl1`/`apl2` the late window is as good as the full one (0.998).

### Leakage check

The brain is a deterministic function of the 32 bits, and the 2,000 states contain
1,169 distinct relay patterns, so plain k-fold CV could put a bit-identical count
vector in both folds. Every probe therefore also reports `StratifiedGroupKFold`
accuracy with **relay pattern as the group**, so no pattern is shared between
train and test. At `apl2_tonic_vth0`, grouped CV gives KC 0.995 / ALPN 1.000 /
DN 0.7305 against plain 1.000 / 1.000 / 0.8415. The result is not memorisation.

---

## 7. The `kc_input_norm` retry

One retry under the chosen setting, since last round it failed only because the
old input had nothing to normalise toward. It fails again, and slightly worse.

| metric | `apl2_tonic_vth0` | `apl2_tonic_vth0_norm1` |
|---|---|---|
| KC active fraction | 0.0482 | **0.0656** (denser) |
| always-on fraction | 0.0406 (passes) | **0.0605** (now fails criterion 2) |
| always-on / active | 0.842 | **0.922** (worse) |
| Jaccard between/within | 0.8792 | **0.8862** (worse) |
| input-dependent KCs | 838 | 892 |
| KC hand type (grouped) | 1.000 (0.995) | 1.000 (0.9995) |
| MBON Hz / varying | 8.61 / 45 | 8.05 / 45 |
| gate | 3 of 4 | 2 of 4 |

Same direction as `outputs/mb/REPORT.md` section 9: equalising each KC's total
ALPN in-weight raises the active fraction and the always-on share and does not
improve separability. The new input does not change that, because the residual
between-KC spread is *which* ALPNs a KC samples, not how much weight it has -- and
now that the ALPNs genuinely differ across states, that spread matters more, not
less. Nothing left to retry here.

---

## 8. MN9 sugar check: tonic vs Poisson

`outputs/mb2/mn9.json`. The 165 labellar `LB*` gustatory receptor neurons, no
drive, and 165 size-matched random ORNs, 200 ms windows. Tonic is deterministic
(one run); Poisson is repeated over 5 noise seeds because MN9 is a two-neuron
readout.

| setting | GRN rate | sugar GRN -> MN9 | no drive | 165 matched random ORNs | selective? |
|---|---|---|---|---|---|
| `apl1` tonic | 126.6 Hz | 1 | 0 | 7 | no |
| `apl1` Poisson | 83.0 Hz | 3.2 (1-4) | 0 | 4.8 (4-6) | no |
| `apl2` tonic | 126.7 Hz | 1 | 0 | 5 | no |
| `apl2` Poisson | 83.0 Hz | 1.8 (1-3) | 0 | 6.0 (3-11) | no |
| `apl5` tonic | 126.8 Hz | 1 | 0 | 6 | no |
| `apl5` Poisson | 83.0 Hz | 1.2 (1-2) | 0 | 4.0 (2-8) | no |

**Poisson drive does not restore sugar specificity.** It does raise the sugar
response at `apl1` (1 -> 3.2 spikes) but it raises the matched-ORN control just as
much, and the control stays >= the sugar condition at every APL scale. MN9 is
silent with no drive under both modes, which is the one part of the check that
passes. This reproduces `outputs/mb/REPORT.md` section 7 -- MN9_R has 26 in-edges
summing to -8.2 mV and never fires -- and adds that the tonic-versus-Poisson
operating point is not the explanation. The Poisson spread (matched control 3-11
spikes at `apl2`) is also large enough that a 1-versus-5 difference in a
two-neuron readout would not have been resolvable with one seed.

---

## 9. Chosen setting

`outputs/mb2/tuned_config.json`. Picked by the pre-registered rule (criteria
passed, then KC hand-type margin on grouped CV, then lower Jaccard):

```json
{"tuning": {"apl_scale": 2.0, "alpn_kc_scale": 1.0, "kc_vth_offset_mv": 0.0,
            "kc_bias_mv": 0.0, "kc_kc_scale": 1.0, "kc_input_norm": 0.0},
 "drive": {"mode": "tonic", "tonic_mv": 30.0},
 "encoding": {"kind": "glomerular", "n_features": 32,
              "map_class": "flybalatro.encode.GlomerularMap",
              "assignment": [{"bit": 0, "glomerulus": "ORN_VM7d", "n_orn": 36}, ...]},
 "window_ms": 50.0, "brain_seed": 1, "v_jitter_mv": 6.9}
```

`apl_scale = 2`, tonic 30 mV, no threshold offset -- the same tuning the previous
round chose, now on the new input. 196 active KCs per state (4.8%), 838
input-dependent KCs, 146 of them active in an average state, 8.61 Hz MBON with 45
MBONs varying, ALPN unchanged at 71.2 Hz. It fails the gate on criterion 1 only.
The file carries the full 32-bit-to-glomerulus map so a later script reproduces
the input exactly:

```python
import json
from flybalatro import connectome as C
from flybalatro.brain import Brain
from flybalatro.encode import GlomerularMap
from flybalatro.tuning import Tuning

cfg = json.load(open("outputs/mb2/tuned_config.json"))
g = C.load(5)
gm = GlomerularMap(g, [b["glomerulus"] for b in cfg["encoding"]["assignment"]],
                   drive_mv=cfg["drive"]["tonic_mv"])
brain = Brain(g, seed=1, v_jitter=6.9, tuning=Tuning.from_dict(cfg["tuning"]))
```

**`apl5_tonic_vth0` is reported alongside it** as the sparse alternative, as
`apl5` was last round. It is the only setting that comes close to criterion 1
(0.5881, against an input-level 0.5040), has the best always-on / active of any
non-degenerate setting (0.339 versus 0.842), and still decodes hand type at 0.984
(grouped 0.968) from **23.8 active KCs per state**. Its costs: 415 input-dependent
KCs instead of 838, no late-window code at all (7 varying columns), and 0.6%
active, below the in vivo 5-10% band.

---

## 10. Judgment: is there a code a KC -> MBON rule could use?

**Yes, and the answer does not depend on the gate.**

The gate fails, uniformly and on one criterion: mean between-type Jaccard over
mean within-type Jaccard, 0.588 at best against a threshold of 0.5. **The number
is not wrong and the mushroom body really does blur the active set** -- the
driven-ORN sets go in at a ratio of 0.5040 and the KC sets come out at 0.8792 at
the chosen setting. Three things say that blurring is not what decides whether a
plasticity rule has something to work with.

1. **The binarised readout settles it.** `KCbin` -- the KC matrix reduced to
   active/inactive, every spike count thrown away -- decodes hand type at 1.000
   (grouped 0.9945) at the chosen setting and 0.9845 at `apl5_tonic_vth0`. The
   active *set* is fully discriminative at a mean pairwise overlap of 0.88. High
   mean overlap and perfect separability are compatible when the informative
   difference is a reliable handful of cells sitting on a large shared core, and
   mean Jaccard averages the two together.
2. **The antennal lobe fails the criterion too, while decoding perfectly.** ALPN
   active-set Jaccard ratio is 0.95 at the chosen setting -- essentially the same
   set of projection neurons fires for every state -- and ALPN 9-way hand-type
   accuracy is 1.000. One synapse from the receptors, mean set overlap and linear
   decodability are already close to independent.
3. **The input straddles the threshold.** Relay bits 0.4846, driven-ORN sets
   0.5040 -- the receptors themselves fail it on the set the fly actually sees. The
   criterion is asking the mushroom body to beat an input that does not pass.

What a KC -> MBON plasticity rule would actually have at the chosen setting: 4,064
Kenyon cells, 196 active in an average state, of which **146 belong to the 838
cells whose activation depends on the input**; 50 cells fire for literally every
state and 3,226 for none. That is a real, input-dependent, reliably addressable
substrate, which is exactly what was missing last round -- there, 837 cells fired
for every input and the KC identity probe was at chance.

Three qualifications, in order of size.

* **The relay is still computed outside the brain.** `flybalatro/hands.py`
  enumerates the subsets, classifies the hands and scores them in ordinary Python;
  the fly is handed the answer as 32 channels. What is measured here is whether
  the fly can *carry* that answer, not compute it. It now carries it intact to the
  Kenyon cells (1.000) and mostly intact to the descending neurons (0.8415 tonic,
  grouped 0.7305) -- against DN 0.61 identity versus 0.75 intensity in `RESULTS.md`.
* **The task became easier, and not only inside the fly.** The old encoding's
  glomerular ceiling was 0.775; the new one is 1.000 by construction, because 32
  bits get 32 dedicated channels. Some of the 0.598 -> 1.000 jump in KC accuracy
  is the fly and some is that the question changed from "recover one of 180 bits
  through a saturated 54-channel bottleneck" to "recover which of 32 channels is
  on". The controlled part of the claim is narrower and still holds: on *the same*
  32-bit relay, an input that leaves 25 of 32 glomeruli silent is decodable
  through the mushroom body and an input that lights all of them is not.
* **Poisson drive decodes worse than tonic at every stage past the antennal lobe**
  -- KC hand type 0.995 versus 1.000, MBON 0.394 versus 0.511, DN 0.597 versus
  0.8415 -- but the two modes are **not rate-matched** (84 Hz versus 127 Hz on the
  driven ORNs), so that comparison is confounded by total drive and is not a clean
  test of Poisson versus tonic. Nothing here measures synchrony directly, so this
  is not evidence for or against the hypothesis that constant current causes
  pathological synchrony; it only says that at these two operating points the
  tonic one relays more.

Also worth recording: MBON 9-way hand-type accuracy is 0.35-0.64 across settings
against a permuted 0.11, so the MBONs do carry partial hand-type information even
though the gate did not require it -- but the drop from KC 1.000 to MBON 0.511 at
the chosen setting is the largest single loss anywhere in the pathway, and the
MBON active-set Jaccard ratio is 0.96. Whatever a plasticity rule is meant to read
out, the 97 MBONs are where the code narrows.

---

## Artifacts

`outputs/mb2/`: `REPORT.md` (this file), `assignment.json`, `input_stats.json`,
`sample.npz`, `grid.json`, `probes.json`, `mn9.json`, `tuned_config.json`,
`counts_<setting>.npz` (13 files: KC / MBON / ALPN / DN spike counts per state,
full window and pre-15-ms), `grid.log`, `probe.log`.

Reproduce: `python -m scripts.mb2_sample`, then `python -m scripts.mb2_grid`
(~4 min per setting, resumable per setting), then `python -m scripts.mb2_probe`,
then `python -m scripts.mb2_gate --write-config --probe-tables`. The
`kc_input_norm` retry is
`python -m scripts.mb2_grid --extra apl2_tonic_vth0_norm1:2.0,tonic,0.0,1.0`
followed by `python -m scripts.mb2_probe --only apl2_tonic_vth0_norm1`.
Tests: `python -m pytest tests/`.
