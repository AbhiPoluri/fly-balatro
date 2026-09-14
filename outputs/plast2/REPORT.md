# Plasticity v2: per-Kenyon-cell homeostasis, and what the fly learns once credit is odour-specific (2026-09-13)

`outputs/plast/REPORT.md` ended with a clean diagnosis and a null result. The fly's
own dopamine-gated depression rule *did* learn from chips -- P(play | legal) went
0.527 -> 0.948 in 9 of 9 runs with a frozen control that did not move a decimal --
but it learned **one number for every odour**: the ante-1 clear rate fell 0.350 ->
0.033, worse than always-discard and close to always-play. The cause was measured,
not guessed: 188 of the ~211 Kenyon cells active per odour came from a pool that
responded to **more than half of all odours** (78 of them to every odour), and those
cells absorbed 70-75% of every weight change. A KC -> MBON synapse is a coincidence
detector; a presynaptic population that is active for everything makes its credit
assignment odour-independent by construction. Conditioning specificity failed exactly
as that predicts: unpaired odours moved 85-105% as far as the paired one.

This round changes the Kenyon-cell code and nothing else, and asks the same question
again.

---

## 1. The mechanism: per-Kenyon-cell homeostatic thresholds

### 1.1 Why no existing knob could do it

`flybalatro/tuning.py` already had four ways to make the mushroom body sparser --
`apl_scale`, `alpn_kc_scale`, `kc_vth_offset_mv`, `kc_kc_scale` -- and all four are
**population-wide rescalings**. They change how many Kenyon cells cross threshold;
they cannot change which. The reason is in the module's own docstring, measured on
this graph: the between-KC spread in total ALPN-driven conductance is 112.6 mV (set
by how many PN synapses that cell happens to have) against a within-KC, across-odour
spread of 9.5 mV, a ratio of 11.8. Threshold crossing is decided by KC in-degree, and
a uniform shift slides the cut along a ranking the wiring already fixed. That is why
`outputs/mb2` found every sparse setting producing a sparse code made of *the same
cells*, and why `apl5` -- the one setting whose code was specific -- had 18 active
KCs, too few to move an MBON at all.

### 1.2 The biology

Real Kenyon cells are not uniform-threshold cells. KC excitability is under
homeostatic control -- KC-intrinsic potassium conductances and the APL feedback loop
hold each cell near a set point -- and the population-level consequence is the
textbook sparse code: **an individual KC responds to roughly 5-10% of odours**, not to
half of them (Turner et al. 2008; Honegger et al. 2011; Lin et al. 2014). That
per-cell set point is a property of the animal that a synapse-count model has no way
to inherit, in exactly the way it has no way to inherit per-synapse efficacy, input
resistance or receptor identity. So it is calibrated the way the animal arrives at it.

### 1.3 The rule

`kc_vth_offsets: float32[n_kc]`, a new `Tuning` field indexed by `graph.kc_indices()`
and added on top of the existing per-neuron `vth` array (`flybalatro/tuning.py`,
`scripts/kc_homeo.py`):

```
r_i        = fraction of CALIBRATION odours for which KC i emits >= 1 spike
offset_i  += k * log((r_i + eps) / (target + eps))      target 0.07, eps 0.02
```

and nothing else. **The update sees only "how often did I fire."** It never sees the
hand type, the score bucket, the labels, the reward, the MBONs or the readout, so it
cannot smuggle the task into the encoding. The brain is rebuilt through `Tuning.apply`
each iteration, so the shipped `tuned_config.json` exercises the same code path every
later run does.

Settings: deadband -- cells with `r` in [0.04, 0.12] are not touched; `k` annealed
(4, 4, 3, 3, 2, 2, 1.5, 1.5, 1, 1, then 0.6, 0.6); per-iteration step capped
asymmetrically at 2.5 mV down and 4.0 mV up (lowering a threshold *recruits* a cell
and, because KCs inhibit each other through APL, changes every other cell's input;
raising one is locally stabilising and has tens of mV to cover).

### 1.4 Two deviations from the brief, both stated up front

**(a) The lower clamp is per cell, not -10 mV.** The brief's clamp is [-10, +30].
This kernel has `v_rest = -52` and a fixed per-neuron `v0 = -52 + U(0, 6.9)` mV, so a
threshold pushed below a cell's *own* `v0` makes it fire on the first tick of every
window regardless of the odour -- precisely the pathology homeostasis exists to
remove. The floor is therefore `max(-10, v0_i - V_TH + 0.25)`: "a spike threshold has
to sit above the cell's own resting potential". Measured, that floor lies above -10 mV
for **all 4,064 cells** (range -6.75 .. +0.15), and 2,249 cells end sitting on it.
Without it, the 3,299 cells that are silent at the starting point get driven to -10 mV
and the loop manufactures ~3,300 always-on cells -- the opposite of the goal. The
upper clamp is the brief's +30 and **no cell reached it** (max +21.4).

**(b) The stop rule was corrected mid-calibration.** Iterations 0-9 ran, converged on
the brief's criterion at iteration 9 (`frac(r > 0.3)` = 0.0138 < 0.02) -- and then
applied one more update and shipped *that*, which had never been measured:
re-measuring gave `frac(r > 0.3)` = 0.0541. The loop was changed to break **before**
applying an update once the measurement converges, so the shipped operating point is
one somebody looked at, and resumed for iterations 10-11 at gain 0.6. It converged
again at iteration 11, this time shipping the measured offsets. All 12 rows are in
`kc_homeo.json` with their `gain` and an `update_applied` flag; the resumed iterations
carry `gain_override = 0.6`.

### 1.5 The odour set

Real relay patterns from `outputs/bc2/states.npz`, first 32 columns = the
`features_v2` relay block. Filtered to rows inside a blind with a hand (block
non-zero) **and** with the `sel_none` bit on, so each pattern is one the fly actually
meets at a decision point, and to episodes >= 1000: 17,974 rows, **1,173 distinct
patterns**. Deduplicated on purpose -- "responds to 5-10% of odours" is a statement
about the odour set, not about how often the game deals each one. Split stratified by
hand type with a fixed seed (20260913) into **500 CALIBRATION** and **500 HELD-OUT**,
173 spare, no pattern in both. Straight Flush has only 15 distinct patterns in the
whole dump, which is why the episode cut is 1,000 and not 2,000: at 2,000 it drops to
6 and a 500/500 split leaves 3, below what stratified 5-fold CV needs.

The bc2 episodes come from seeds 0-1359 (heuristic teacher) and 500000-501358
(random), disjoint from the plasticity calibration (300000+), training (200000+) and
evaluation (100000-100059) seeds **by construction**, not merely by the episode cut.

### 1.6 The iteration record

Measured on the 500 calibration odours. `r>0.5` / `r>0.3` are counts of 4,064 KCs.

| iter | k | active frac | active/state | r>0.5 | r>0.3 | in band | silent | sign flips | at floor | at ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.0 | 0.0540 | 220 | 197 | 245 | 130 | 3299 | 0.00 | 1333 | 0 |
| 1 | 4.0 | 0.0612 | 249 | 154 | 269 | 371 | 2655 | 0.11 | 2319 | 0 |
| 2 | 3.0 | 0.0833 | 339 | 214 | 403 | 546 | 2157 | 0.27 | 2753 | 0 |
| 3 | 3.0 | 0.0778 | 316 | 179 | 367 | 613 | 2116 | 0.38 | 2704 | 0 |
| 4 | 2.0 | 0.0793 | 322 | 175 | 386 | 604 | 2110 | 0.39 | 2737 | 0 |
| 5 | 2.0 | 0.0737 | 299 | 155 | 330 | 656 | 2104 | 0.39 | 2701 | 0 |
| 6 | 1.5 | 0.0757 | 308 | 167 | 366 | 682 | 2093 | 0.39 | 2452 | 0 |
| 7 | 1.5 | 0.0570 | 232 | 36 | 154 | 788 | 1945 | 0.35 | 2531 | 0 |
| 8 | 1.0 | 0.0637 | 259 | 76 | 244 | 873 | 1941 | 0.36 | 2433 | 0 |
| 9 | 1.0 | 0.0464 | 188 | 25 | 56 | 1082 | 1843 | 0.29 | 2550 | 0 |
| 10 | 0.6 | 0.0648 | 263 | 71 | 220 | 1119 | 1769 | 0.29 | 2249 | 0 |
| **11** | **0.6** | **0.0454** | **184** | **19** | **34** | **1463** | **1599** | **0.24** | **2249** | **0** |

The loop is a damped oscillation, not a monotone descent, and the report should say
so: the sign-flip fraction (cells whose update reversed direction relative to the
previous iteration) sits at 0.24-0.39 throughout. That is a real property of the
system rather than a tuning failure -- APL makes every threshold change a change to
every other cell's input, so the population moves together -- and it is exactly why
the shipped operating point has to be a *measured* one. Iterations 7, 9 and 11 are the
troughs; 11 is what shipped.

Final offsets: **-6.74 .. +21.36 mV**, mean -2.12, sd 3.13, 4,031 of 4,064 non-zero,
resulting `v_th` range -51.74 .. -23.64 mV. Percentiles: 5% -6.04, 25% -4.22, 50%
-2.49, 75% -0.77, 95% +3.50, 99% +9.44. 139 cells above +5 mV, 36 above +10 mV, none
at the +30 ceiling, 2,249 at their own floor.

---

## 2. What it did to the Kenyon-cell code

Both columns measured on the **same 500 held-out odours**, neither of which the
calibration ever saw. Rates are over 4,064 Kenyon cells.

| | `outputs/plast` (before) | + KC homeostasis (after) |
|---|---|---|
| KC active fraction | 0.0538 | **0.0436** |
| KC active per odour | 218.7 | **177.2** |
| KC spikes per odour | 323.0 | 197.4 |
| cells with r > 0.5 ("always on") | **192** (0.0472) | **16** (0.0039) |
| cells with r > 0.3 | **242** (0.0595) | **31** (0.0076) |
| cells that ever respond | 769 | **2,457** |
| cells that never respond | 3,295 | 1,607 |
| cells inside the 4-12% band | 123 | **1,382** |
| median rate of responding cells | 0.088 | **0.066** |
| response-rate 95th / 99th percentile | 0.470 / 1.000 | 0.132 / 0.263 |
| 9-way hand type from KC counts, grouped CV | 0.986 | **0.996** |
| same, permuted labels | 0.120 | 0.134 |
| 9-way hand type from the binarised active set | 0.982 | 0.996 |
| MBON mean rate | 14.09 Hz | 13.18 Hz |
| MBONs that vary across odours | 47 of 97 | **68 of 97** |
| MBONs silent for every odour | 38 | **23** |
| reward-side MBON01-07 silent | 3 of 16 | **0 of 16** |
| approach - avoid drive range | 15.83 Hz (13.9 quanta) | 16.21 Hz (14.3 quanta) |

The code did not get sparser so much as it got **spread out**: three times as many
cells participate (769 -> 2,457), the broadly-tuned pool collapses by 12x (192 -> 16
cells above half the odours), and the median responding cell now answers to 6.6% of
odours -- the middle of the 5-10% the literature reports. Total activity actually
*fell* slightly, so this is redistribution, not amplification. Hand-type decoding did
not merely survive, it improved (0.986 -> 0.996 against a 0.134 permuted floor), and
1,599 cells remain permanently silent because their inputs are unreachable at any
threshold above their own resting potential -- a property of this wiring, reported
rather than tuned away.

A note on the CV: with 500 *distinct* patterns, "grouped by relay pattern" is 500
groups of one, so grouped CV and ordinary stratified 5-fold coincide. That is the
strictest form of the test -- no pattern appears in both a training and a test fold --
not a weaker one.

---

## 3. The gate: five criteria on 500 held-out odours

All five pre-registered in the brief. Criterion (v) is the `outputs/plast/probe`
protocol unchanged: the 8-odour panel picked by `pick_odor_panel` from 200 hands off
seeds 300000+, one odour per (score bucket, hand type) pair, 20 pulses at eta 0.05,
each odour conditioned in its own fresh run and the other seven measured as the
generalisation term.

| # | criterion | threshold | `outputs/plast` | + KC homeostasis |
|---|---|---|---|---|
| i | frac(r > 0.5) | < 0.02 | 0.0472 **FAIL** | **0.0039 PASS** |
| i | frac(r > 0.3) | < 0.05 | 0.0595 **FAIL** | **0.0076 PASS** |
| ii | KC active fraction | 3-10% | 0.0538 PASS | **0.0436 PASS** |
| iii | 9-way hand type from KC, grouped 5-fold CV | >= 0.85 | 0.986 PASS | **0.996 PASS** |
| iv | approach - avoid drive range | > 1 quantum (1.14 Hz) | 15.83 Hz PASS | **16.21 Hz PASS** |
| iv | MBON01-07 (reward side) not silent | > 0 firing | 13 of 16 fire PASS | **16 of 16 fire PASS** |
| v | reward: specificity share | >= 0.50 | 0.145 **FAIL** | **0.712 PASS** |
| v | punish: specificity share | >= 0.50 | -0.057 **FAIL** | **1.037 PASS** |
| | **overall** | | **FAIL** | **PASS** |

Criterion (v) in full, 20 pulses on the panel:

| | before: delta self | delta others | specific | after: delta self | delta others | specific |
|---|---|---|---|---|---|---|
| reward | +9.195 Hz | +7.860 | +1.335 (**15%**) | +4.659 Hz | +1.343 | +3.316 (**71%**) |
| punish | -2.329 Hz | -2.462 | +0.133 (**-6%**) | -1.051 Hz | +0.039 | -1.090 (**104%**) |

Direction is right in both cases for both flies (8/8 odours for reward, 7/8 for
punish). What changed is where the effect lands. Before, 20 punishment pulses on odour
A moved the other seven odours *further* than A itself; after, they do not move at all
(+0.039 Hz, within a tenth of one quantisation quantum), and the reward arm keeps 71%
of its effect on the paired odour.

**The fallback was not triggered.** The brief allowed one further mechanism if (v)
failed -- scaling each KC's outgoing MBON synapses by `target / max(r_i, target)`. It
is implemented (`Tuning.kc_mbon_out_scale`, `plast2_gate.fallback_out_scale`), covered
by tests, and `--fallback --write-config` will fold it in, but the homeostasis passes
on its own so the shipped `tuned_config.json` does not use it. Label:
`apl2_aplkc1_aplmb0.1_vtho4064`.

---

## 4. Dopamine pulse calibration

**This is calibration of the model, not shaping of the reward.** What the game gives
the fly is untouched: a reward pulse after a play that cleared the blind or made its
fair share `needed / plays_left`, a punishment pulse after a play that lost the blind,
nothing at all after a discard, at whatever *frequency* ante 1 produces. What is
calibrated is the per-pulse gain of the two arms of **our own rule**, which differed
by ~4x for a reason that has nothing to do with dopamine:

* the reward arm depresses KC -> (avoid, PAM) synapses. The neurotransmitter rule gives
  24 avoid MBONs, so one lost spike there moves `play_drive` by 1/24 x 20 = 0.83 Hz;
* the punishment arm depresses KC -> (approach, PPL1) synapses. There are 66 approach
  MBONs, so one lost spike moves it by 1/66 x 20 = 0.30 Hz.

The pool-size ratio alone is 66/24 = 2.75. A real PPL1 pulse is not weaker than a real
PAM pulse and nothing in the connectome says it should be, so `eta_punish` is scaled
until one punishment pulse moves the paired odour's drive as far as one reward pulse.

Measurement. The drive quantum is 1.136 Hz and a single pulse moves less than that, so
the single-pulse number is quantisation, not signal: at N = 1 every eta on the grid
returns 0.83-1.97 Hz with no ordering. The trajectory is therefore run out and
`|delta| / N` reported at N = 1 and 5. Two passes:

* **pass 1** (`eta_calib_pass1.json`): 8 odours, 20 pulses. Non-monotone in eta at
  every N -- the N=5 column went 0.314, 0.316, 0.261, 0.224, 0.316, 0.441, 0.487,
  0.794, 0.610 across the grid. Noise, so it was not used.
* **pass 2** (`eta_calib.json`): 23 odours (every distinct bucket x hand-type pair
  available), 8 pulses, matched at N = 5. Monotone:

| punish eta | 0.05 | 0.07 | 0.10 | 0.14 | 0.20 | 0.28 | 0.40 | 0.60 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|
| abs(delta)/pulse at N=5 (Hz) | 0.274 | 0.270 | 0.279 | 0.290 | 0.353 | 0.401 | 0.470 | 0.586 | 0.543 |

Reward reference: eta 0.05 -> 0.3439 Hz/pulse, eta 0.02 -> 0.2391 Hz/pulse.

Interpolating the monotone envelope at the eta_reward = 0.05 target gives
**eta_punish = 0.1907, ratio 3.81**. The eta_reward = 0.02 target (0.239) falls *below*
the smallest punishment eta on the grid, so its own match is a grid edge, not a
measurement; depression is multiplicative in `1 - eta * elig` and therefore linear in
eta to first order, so the single bracketed ratio is applied to both:

| | eta_reward | eta_punish | ratio |
|---|---|---|---|
| low | 0.02 | **0.0763** | 3.81 |
| control | 0.05 | **0.1907** | 3.81 |

3.81 against a pool-size ratio of 2.75 says the punishment arm's edges are also weaker
per edge, not only more numerous.

---

## 5. The protocol, rerun

Identical to `outputs/plast` except for the three things above: the homeostatic fly,
the calibrated punishment pulse, and 1,200 training hands instead of 600 (a code that
assigns credit per odour learns each odour separately, and so more slowly). Same
decision rule (`approach - avoid` mean MBON rate plus a bias and temperature
calibrated once on 200 hands from seeds 300000+ and frozen before any dopamine), same
10% exploration floor during training, greedy with plasticity frozen and no dopamine
at evaluation on seeds 100000-100059. Real wiring x 3 seeds x eta_reward in
{0.02, 0.05}, plus the frozen-fly control x 3 seeds. 9 runs, ~6.5 min each, 3
processes.

### 5.1 Evaluation: 60 ante-1 games on seeds 100000-100059, identical for every row

| row | clear before | clear after | chips after | P(play given legal) after |
|---|---|---|---|---|
| frozen fly, no dopamine (3 seeds) | 0.328 | 0.328 | 789 | 0.522 |
| real, eta_r 0.02 / eta_p 0.0763 (3 seeds) | 0.328 | 0.244 | 807 | **0.754** |
| real, eta_r 0.05 / eta_p 0.1907 (3 seeds) | 0.328 | 0.244 | 792 | **0.780** |
| always play | - | 0.000 | 519 | 1.000 |
| always discard | - | 0.400 | 868 | 0.000 |
| **best odour-only policy** (play iff bucket >= `lt1.0`) | - | **0.633** | 1001 | 0.385 |
| v2 teacher dig rule (sees `plays_left`) | - | **0.700** | 1088 | 0.569 |

The frozen control is bit-identical before and after in all three seeds, so every
change below is the weights. The pre-evaluation clear rate varies across seeds
(0.267 / 0.433 / 0.283) because the evaluation RNG is seeded per run; that spread is
the reason the clear-rate row cannot carry the conclusion. Pooled over 180 games,
0.328 -> 0.244 is **z = 1.75** on a two-proportion test: a nominal decline that 60
games per seed does not resolve. Chips are flat (789 -> 807 / 792). What *is* resolved
is P(play), the bucket gradient and the joint table.

### 5.2 P(play) by score-vs-needed bucket, and the gradient

| row | lt0.25 | lt0.5 | lt1.0 | ge1.0 | ge1.0 - lt0.25 |
|---|---|---|---|---|---|
| naive / frozen (identical) | 0.611 | 0.462 | 0.381 | 0.621 | **+0.010** |
| learned, eta_r 0.02 | 0.561 | 0.892 | 0.841 | 0.925 | **+0.364** |
| learned, eta_r 0.05 | 0.545 | 0.912 | 0.934 | 0.949 | **+0.404** |
| v1 (`outputs/plast`), eta 0.05 | 0.95 | 0.92 | 0.96 | 0.98 | +0.033 |
| v2 teacher dig rule | 0.000 | 0.693 | 0.981 | 1.000 | +1.000 |
| best odour-only policy | 0.000 | 0.000 | 1.000 | 1.000 | +1.000 |

The naive fly's gradient is +0.010 and non-monotone: it has essentially no opinion
about whether the hand can pay. After learning the gradient is +0.40 and monotone
across all four buckets, and -- this is the part v1 could not do -- the **lowest bucket
moves down while every other bucket moves up.** v1 moved all four to 0.92-0.98.

### 5.3 The joint table, P(play | hand type x score bucket)

Pooled over the three eta_reward = 0.05 seeds, cells with n >= 5 decisions, `(n)` is
the number of decisions in that cell. "naive" is the pre-evaluation of the same runs,
which is also exactly the frozen control.

**naive / frozen**

| hand type | lt0.25 | lt0.5 | lt1.0 | ge1.0 |
|---|---|---|---|---|
| High Card | 0.59 (136) | 0.83 (6) | - | - |
| Pair | 0.70 (361) | 0.72 (76) | 0.79 (19) | 0.79 (19) |
| Two Pair | 0.48 (218) | 0.40 (232) | 0.29 (126) | 0.62 (47) |
| Three of a Kind | 0.38 (8) | 0.54 (35) | 0.28 (25) | 0.71 (7) |
| Straight | - | 0.88 (24) | 0.54 (114) | 0.79 (66) |
| Flush | - | 0.24 (25) | 0.37 (121) | 0.52 (106) |
| Full House | - | 0.07 (40) | 0.32 (152) | 0.64 (131) |
| Four of a Kind | - | - | 0.00 (6) | 0.56 (34) |
| Straight Flush | - | - | - | 0.22 (9) |

**learned** (eta_r 0.05, 3 seeds)

| hand type | lt0.25 | lt0.5 | lt1.0 | ge1.0 |
|---|---|---|---|---|
| High Card | **0.37** (115) | 0.89 (9) | - | - |
| Pair | **0.52** (389) | 0.85 (116) | 0.88 (40) | 1.00 (31) |
| Two Pair | 0.69 (200) | 0.93 (250) | 0.93 (87) | 0.92 (63) |
| Three of a Kind | - | 0.94 (32) | 0.94 (35) | 0.94 (18) |
| Straight | - | 1.00 (11) | 0.91 (90) | 0.94 (64) |
| Flush | - | - | 1.00 (48) | 0.94 (70) |
| Full House | - | 1.00 (13) | 0.96 (52) | 0.96 (67) |
| Four of a Kind | - | - | - | 1.00 (14) |
| Straight Flush | - | - | - | - |

**v2 teacher** (for reference; it reads `plays_left`, the fly cannot)

| hand type | lt0.25 | lt0.5 | lt1.0 | ge1.0 |
|---|---|---|---|---|
| High Card | 0.00 (25) | 0.33 (3) | - | - |
| Pair | 0.00 (117) | 0.19 (31) | 0.85 (13) | 1.00 (19) |
| Two Pair | 0.00 (111) | 0.76 (113) | 0.97 (40) | 1.00 (45) |
| Three of a Kind | 0.00 (7) | 1.00 (9) | 1.00 (14) | 1.00 (6) |
| Straight | - | 1.00 (6) | 1.00 (16) | 1.00 (22) |
| Flush | - | 1.00 (4) | 1.00 (24) | 1.00 (16) |
| Full House | - | 1.00 (10) | 1.00 (48) | 1.00 (32) |
| Four of a Kind | - | - | 1.00 (1) | 1.00 (3) |

Read the two cells with the most decisions in them. **Pair / lt0.25**, 389 decisions:
0.70 -> **0.52**, and the teacher plays it 0.00 of the time. **Two Pair / lt0.5**, 250
decisions: 0.40 -> **0.93**, teacher 0.76. Those are opposite-signed moves on the two
sides of the pay line, from a rule with no access to which side it is on except through
the odour. v1 produced 0.46 -> 0.95 and 0.63 -> 0.92 -- same sign, everywhere.

### 5.4 Variance across cells, and why it is the wrong-signed metric

The brief asks for the variance of P(play) across hand-type x bucket cells. Here it is,
with the number that actually carries the claim next to it -- the cell-by-cell
correlation with the teacher's P(play), weighted by how many decisions the fly faced in
each cell.

| condition | cells (n>=5) | variance | sd | spread | mean | corr with teacher | weighted abs error |
|---|---|---|---|---|---|---|---|
| naive / frozen | 26 | 0.0549 | 0.234 | 0.875 | 0.511 | **-0.065** | 0.531 |
| learned (eta_r 0.05) | 22 | **0.0242** | 0.156 | 0.626 | 0.887 | **+0.881** | **0.300** |
| v2 teacher | 21 | 0.1634 | 0.404 | 1.000 | 0.751 | 1.000 | 0.000 |
| best odour-only policy | 21 | 0.2449 | 0.495 | 1.000 | 0.571 | +0.630 | 0.218 |
| always play | 15 | 0.0000 | 0.000 | 0.000 | 1.000 | n/a | 0.587 |

The variance **fell**, 0.0549 -> 0.0242, and taken alone that reads as a loss of
specificity. It is not, and the mean column says why: the cell mean moved 0.511 ->
0.887, so the distribution compressed against the ceiling at 1.0. Variance measures
spread, not direction, and `always play` scoring 0.0 while the best odour-only policy
scores 0.2449 shows it cannot separate a good policy from a degenerate one. The
direction is what changed: the naive fly's per-cell P(play) is **uncorrelated** with
what the situation deserves (-0.065, i.e. nothing), and the learned fly's is **+0.881**,
with the weighted mean absolute error against the teacher falling 0.531 -> 0.300.

### 5.5 Learning curves and the dopamine bookkeeping

Rolling window 50, over the 1,200 training hands, averaged over 3 seeds:

| | reward rate, first third | reward rate, last third | P(play), first third | P(play), last third |
|---|---|---|---|---|
| frozen (no dopamine) | 0.329 | 0.290 | 0.578 | 0.578 |
| real, eta_r 0.02 | 0.313 | **0.399** | 0.643 | 0.768 |
| real, eta_r 0.05 | 0.381 | **0.418** | 0.723 | 0.782 |
| v1 (`outputs/plast`) | 0.40 | 0.36 | ~0.95 by hand 150 | ~0.95 |

**The reward rate now rises during training and the frozen control's does not.** In v1
the reward rate *fell* from 0.40 to 0.36 while P(play) saturated at the exploration
ceiling inside 150 hands. Here P(play) climbs gradually to 0.75-0.78 (the ceiling with a
10% exploration floor is 0.95, so behaviour is *not* saturated) and the reward rate
improves by 0.09 / 0.04 over the run while the frozen fly's drifts down 0.04 on the same
games.

Dopamine per run, mean over seeds (1,200 hands):

| | reward pulses | punish pulses | no dopamine | ratio |
|---|---|---|---|---|
| real, eta_r 0.02 | 425.3 | 86.3 | 688.3 | 4.9 : 1 |
| real, eta_r 0.05 | 484.3 | 85.3 | 630.3 | 5.7 : 1 |
| v1 (`outputs/plast`) | 692 | 230 | - | 3.0 : 1 |

Weights after training, eta_r 0.05: mean ratio to original 0.809, 1.5% at the 0.05
floor, 61.0% unchanged, never negative, never above original. At eta_r 0.02: mean 0.903,
0.05% at floor. **This is not a saturated rule** -- three fifths of the 33,496
KC -> MBON synapses were never touched, so what limits it is credit assignment, not
headroom.

---

## 6. Where it now breaks: two measurements

### 6.1 The worst decisions generate no dopamine at all

From one training run (eta_r 0.05, seed 0; 1,200 hands, 915 plays, 285 discards):

| bucket | plays | rewarded | punished | **no dopamine** |
|---|---|---|---|---|
| lt0.25 | 282 | **0** | 28 | **254 (90%)** |
| lt0.5 | 216 | 115 | 30 | 71 |
| lt1.0 | 204 | 178 | 21 | 5 |
| ge1.0 | 213 | 211 | 2 | 0 |

`reward` fires when the play cleared the blind or made its fair share; `punish` fires
only when `done and not is_win`, i.e. on the **terminal** losing play. A wasted
`lt0.25` play in the middle of a game is neither: it produces no dopamine, and the rule
is depression-only, so there is nothing to undo the drift the *other* hands caused.
Nine times in ten, the decision the fly most needs to change is invisible to the only
signal it has. That is why P(play | lt0.25) only fell 0.611 -> 0.545 instead of toward
the teacher's 0.000, and it is the single biggest remaining gap.

### 6.2 The plasticity is specific, but not at the distance the task needs

`neighbour.json`. Condition one held-out odour with 20 pulses, then measure the drive
change on other held-out odours, binned by Hamming distance over the 32 relay bits.
`relative spread` = delta(probe) / delta(anchor); 1.0 means the conditioning
transferred completely, 0.0 means it did not transfer. 8 anchors, 24 probes per bin,
reward arm (the larger and cleaner of the two):

| Hamming distance | 2 | 4 | 6-8 | >= 10 |
|---|---|---|---|---|
| before, median ratio | 0.796 | 0.782 | 0.683 | 0.634 |
| before, ratio of means | 0.772 | 0.777 | 0.692 | 0.614 |
| **after, median ratio** | **0.836** | **0.348** | **0.157** | **0.000** |
| **after, ratio of means** | **0.803** | 0.484 | 0.322 | 0.177 |

Before, there is no gradient at all: conditioning one odour moved every other odour
about 70% as far whatever it was. After, there is a proper generalisation gradient --
complete transfer to a near neighbour, none at all to a distant one. That is what
criterion (v) measured and passed, because its 8-odour panel is one odour per
(bucket, hand type) pair and therefore mostly far apart.

But look at distance 2. **The score-vs-needed bucket is one bit of the 32-bit relay
block**, so "Two Pair that can clear this blind" and "Two Pair that cannot" differ in
exactly two bits (one off, one on) of the ~7 that are on -- and they are precisely the
pair the fly has to separate to stop wasting plays. At Hamming distance 2 the transfer
is 0.80-0.84: whatever the fly learns about one, it learns about the other. That is the
mechanism behind the one clearly wrong cell in the joint table, **Two Pair / lt0.25
going 0.48 -> 0.69** while Two Pair / lt0.5 (its distance-2 neighbour, 250 decisions,
heavily rewarded) went 0.40 -> 0.93.

So the bottleneck has moved again, and this time it is in the *encoding*, not the
plasticity: a decision axis that carries all of the task's information is given one bit
in 32, and no amount of Kenyon-cell specificity can separate two odours that are nearly
the same odour. The punishment arm shows the same profile more noisily (median ratio
0.263 / 0.146 / -0.089 / 0.000 by bin) because per-pulse punishment effects are small
enough that individual ratios are unstable.

---

## 7. Verdict

**Did the fly learn something odour-specific from chips? Yes, and this is the first
round where that is true.** Conditioning specificity went from 15% (reward) and -6%
(punish) to 71% and 104% on the held-out panel; the generalisation profile went from
flat-at-0.7 to a real distance gradient falling to zero; the bucket gradient in
behaviour went from +0.010 to +0.404; the per-cell correlation between what the fly
does and what the situation deserves went from -0.065 to +0.881; and the training
reward rate rose 0.31 -> 0.40 where v1's fell 0.40 -> 0.36. The frozen control did not
move a decimal in any of it, and nothing was fitted: the only mutable quantity is the
weight of 33,496 KC -> MBON synapses, changed by a dopamine-gated depression rule driven
by the game's own chips.

**Did it help it win? No, not measurably.** The ante-1 clear rate went 0.328 -> 0.244
(z = 1.75 over 180 games, not resolvable) and chips were flat, 789 -> 807 / 792. That is
a large improvement over v1's 0.350 -> 0.033 collapse, and the learned fly is no longer
near the worst policy in the game, but it is still at or below its own naive starting
point and well below the 0.633 a policy that only reads the score bucket achieves. The
fly learned the right *shape* and stopped short of the magnitudes that would pay:
P(play | lt0.25) reached 0.545 where it needs to reach ~0.0.

**The remaining bottleneck is credit, in two specific senses, both measured.** First,
90% of `lt0.25` plays -- the ones that waste a hand -- generate no dopamine at all,
because punishment only fires on the terminal losing play, and a depression-only rule
has no way to undo drift it did not cause. Reward outnumbers punishment 5:1 as a
consequence. Second, the relay encoding spends one bit of 32 on the only axis that
decides the action, so the two odours the fly must separate sit at Hamming distance 2,
where conditioning still transfers at 0.8. Neither is a Kenyon-cell problem any more.
The obvious next moves are both outside this round's scope: give the punishment signal
the mid-game wasted play (an omission or per-hand-value signal rather than
blind-loss-only), and widen the score-vs-needed axis in the relay block so adjacent
buckets are not adjacent odours.

### Caveats

* 60 evaluation games resolves 0.63 vs 0.24 but not 0.33 vs 0.24; the pre-evaluation
  clear rate alone spans 0.267-0.433 across the three seeds because the evaluation RNG
  is per-seed. Every claim above that matters is made on P(play), the bucket gradient,
  the joint table or the conditioning probe, all of which pool hundreds to thousands of
  decisions.
* The homeostasis loop oscillates (sign-flip fraction 0.24-0.39 at every iteration) and
  the shipped point is one of its troughs, chosen by the brief's own stop criterion on
  the calibration set and then confirmed on the held-out set. It is not a fixed point.
* 1,599 of 4,064 Kenyon cells are silent for every odour even at the lowest threshold
  their own resting potential allows. The effective population is ~2,465 cells.
* The lower clamp deviates from the brief as described in 1.4(a), and the stop rule was
  corrected mid-calibration as described in 1.4(b).
* The eta_reward = 0.02 punishment eta is the eta_reward = 0.05 ratio extrapolated, not
  its own measurement (section 4).
* The reward criterion is still the v2 teacher's fair-share inequality applied to the
  game's own chips, so "the right thing" and the teacher's rule coincide by construction
  -- the same caveat `outputs/plast/REPORT.md` records.
* Ante 1, no jokers, one connectome, one 50 ms window, one valence scheme (the Aso
  neurotransmitter rule). The shuffled-wiring control was not rerun: v1 established it
  is degenerate here, leaving 664 KC -> MBON edges instead of 33,496, so it deletes the
  pathway rather than rewiring it.

---

## Artifacts

Code (all additive; `flybalatro/plasticity.py` unchanged this round, `tuning.py`
extended with two per-KC vector knobs that default to no-ops):

* `flybalatro/tuning.py` -- `kc_vth_offsets`, `kc_mbon_out_scale`
* `scripts/kc_homeo.py` -- the homeostatic calibration and the odour-set split
* `scripts/plast2_gate.py` -- the five criteria, plus the unused fallback
* `scripts/plast2_eta.py` -- the pulse calibration
* `scripts/plast2_run.py` -- the protocol at the new operating point
* `scripts/plast2_neighbour.py` -- generalisation vs odour similarity
* `scripts/plast2_summary.py` -- the tables above
* `tests/test_kc_homeo.py` -- 15 tests; 221 pass in total, 1 skipped

Data, all under `outputs/plast2/`: `tuned_config.json` (the shipped fly, including the
4,064 offsets), `kc_homeo.json` + `kc_homeo.log`, `gate.json` + `gate.log`,
`eta_calib.json` + `eta_calib_pass1.json` + logs, `neighbour.json` + `neighbour.log`,
`baselines.json`, `run_<name>.json` x 9, `summary.json` + `summary.log`,
`run_{a,b,c,d}.log`. `outputs/bc*/`, `outputs/mb*/` and `outputs/plast/` are untouched.

Reproduce (~2 h on an M2 Pro, 3 processes for the runs):

```
python -m scripts.kc_homeo --iterations 12 --max-step-up 2.5
python -m scripts.kc_homeo --resume --iterations 18 --gain 0.6
python -m scripts.plast2_gate
python -m scripts.plast2_eta --n-odors 23 --n-pulses 8 --match-at 5
python -m scripts.plast2_eta --recompute --match-at 5
python -m scripts.plast2_run --baselines
python -m scripts.plast2_run --only real_eta0.02_s0,real_eta0.02_s1,real_eta0.02_s2 &
python -m scripts.plast2_run --only real_eta0.05_s0,real_eta0.05_s1,real_eta0.05_s2 &
python -m scripts.plast2_run --only frozen_s0,frozen_s1,frozen_s2 &
python -m scripts.plast2_neighbour
python -m scripts.plast2_summary
```

The two `kc_homeo` lines are what actually ran, and the shipped offsets are **not** a
clean single-invocation artefact -- the calibration was changed while it ran and the
record says so:

* iterations 0-9 ran with a **symmetric** 2.5 mV per-iteration step cap and the original
  stop rule (which applied one update past convergence and shipped it);
* iterations 10-11 ran after the stop rule was corrected to ship the *measured* offsets,
  and with the asymmetric cap (2.5 mV down, 4.0 mV up) that is now the default, at
  `--gain 0.6`.

So a fresh 12-iteration run on the current code uses the asymmetric cap from iteration 0
and will follow a different trajectory; `--max-step-up 2.5` above recovers the cap
iterations 0-9 actually used, but not the old stop rule. The authoritative record is
`kc_homeo.json`: rows 0-9 predate the `max_step_up_mv` / `update_applied` fields and rows
10-11 carry them, and the shipped operating point is `tuned_config.json`, whose
statistics are the iteration-11 row and the independently measured `gate.json` column.
