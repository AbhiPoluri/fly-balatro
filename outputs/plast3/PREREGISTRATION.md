# Plasticity v3: preregistration of the 2x2 rescue (2026-09-13)

Written **before any v3 game run**. Nothing below is chosen after seeing a v3
result. The only v3 numbers that exist at the time of writing are (a) the
evaluation-mode re-analysis of the v2 weights (Task 1, which touches no new
condition), and (b) the encoding audit in section 3, which is arithmetic over
relay bits and ORN counts and involves no brain and no game.

---

## 0. What v2 left on the table

v2 (`outputs/plast2/REPORT.md`) made the fly's dopamine-gated depression
odour-specific: conditioning specificity 0.145 -> 0.712 (reward arm), behavioural
bucket gradient +0.010 -> +0.404, per-cell correlation with the teacher's action
-0.065 -> +0.881. It did **not** improve the ante-1 clear rate (0.328 -> 0.244).
Its own post-mortem named two causes and measured neither against the alternative:

1. **Credit.** Punishment fires only on the terminal losing play, so 254 of 282
   `lt0.25` plays in a training run -- the decision the fly most needs to
   change -- earn no dopamine at all, and a depression-only rule cannot undo
   drift it did not cause.
2. **Encoding.** The score-vs-needed bucket is 1 relay bit of 32, so the two
   odours the fly must separate sit at Hamming distance 2, where v2's neighbour
   probe measured 0.80-0.84 transfer.

Cause 2 is **not established**: the v2 probe binned *arbitrary* held-out patterns
by Hamming distance, so its distance-2 bin mixes bucket changes with slot-mask
changes. Section 4 is the test that settles it.

## 1. Hypotheses

* **H-A (punishment).** Wiring up the negative branch of the same
  game-computable inequality that already defines reward -- punish every PLAY
  whose chips fall short of the share still needed, not only the terminal losing
  play -- raises the ante-1 clear rate of the learned fly above its frozen
  control.
* **H-B (encoding).** Giving each score bucket its own disjoint, ORN-count
  balanced glomerulus ensemble, instead of one bit in 32, raises the ante-1
  clear rate of the learned fly above its frozen control.
* **H-AB.** The two interact: the omission signal is what makes the low bucket
  punishable at all, and the separated encoding is what lets that punishment stay
  on the low bucket, so the (omission, separated) cell beats the sum of the two
  single-factor improvements.
* **H-0 (the null this round can return).** Neither factor moves the clear rate
  and the plasticity story fails for a reason that is now measured rather than
  asserted.

**Conditioning prediction (section 4), stated before the probe runs.** Under
`current`, a pulse on one bucket will transfer to the other three buckets of the
same hand at >= 0.6 relative spread; under `separated` it will transfer at
<= 0.3. If `current` shows *low* bucket-only transfer, the v2 report's
"Hamming distance 2 bottleneck" is **refuted** and H-B loses its stated
mechanism, whatever the game result.

**Direction prediction for H-A.** Omission punishment fires on ~5x more hands
than the current rule (v2 training run: 504 reward, 81 punish, 615 silent of
which ~411 were unrewarded plays). At the calibrated `eta_punish` this is a
strong pull toward DISCARD. A cell that collapses to near-always-discard
*counts as a pass only if it beats the always-discard baseline itself*; if it
merely matches it, the honest description is "the fly learned *don't play*, not
*when to play*", and it will be reported that way.

## 2. Design

Two factors, fully crossed, three training seeds per cell, plus one frozen
control per encoding (six cells' worth of comparison from 12 training runs + 2
controls).

| factor | level | meaning |
|---|---|---|
| A punishment | `current` | PPL1 pulse only on the terminal losing play (v2) |
| A punishment | `omission` | PPL1 pulse on **every** PLAY that was not rewarded |
| B encoding | `current` | the mb2 32-bit assignment, one glomerulus per bit (v2) |
| B encoding | `separated` | four disjoint ORN-balanced bucket ensembles (section 3) |

Reward is **unchanged** in every cell: a PLAY that cleared the blind or gained at
least `needed / plays_left` chips. Both punishment rules are computed from the
game's own chips by `plast_common.resolve_outcome`; no teacher, no label, no
readout. `omission` is a strict superset of `current` (`punish` already implies
`not reward`), so the two levels differ only in which unrewarded plays get a
pulse.

Frozen everywhere else: the Aso neurotransmitter valence rule, the fixed
approach-minus-avoidance MBON decision, the 50 ms window, real wiring, ante 1,
`eta_reward = 0.05`, 1,200 training decisions, 10% exploration floor during
training, bias and temperature calibrated once per encoding on 200 hands from
seeds 300000+ before any dopamine.

**Seeds, fixed now.**

| use | seeds |
|---|---|
| decider calibration | 300000+ (200 hands, coin-flip policy, as v1/v2) |
| training | 200000 + 1000 x s for s in {0, 1, 2}, identical across all four cells |
| **evaluation** | **400000-400399**, 400 games, identical across every condition |
| homeostasis / gate odours | `outputs/bc2` episodes >= 1000 (seeds 0-1359, 500000-501358) |

400000-400399 is fresh: it is disjoint from the v1/v2 evaluation range
(100000-100059), from training, from calibration and from the bc2 corpus. The
v2 evaluation seeds are never used for a v3 decision.

**Evaluation modes.** Every condition is evaluated twice on the same 400 games:
`greedy` (PLAY iff `p >= 0.5`, equivalently `play_drive >= 0`; no exploration
floor, no RNG) and `sampled` (`rng.random() < p` with the training floor, one
fixed evaluation RNG seed shared by every condition). **`greedy` is the primary
mode** -- it is what "the learned policy" means and it is what the v1/v2
write-ups claimed to report. `sampled` is secondary and is reported beside it
everywhere.

**Recalibration for the new encoding, label-free.** The separated encoding
changes the Kenyon-cell input statistics, so two things are recalibrated with the
**existing** procedures and no outcome labels:
* per-KC homeostatic thresholds (`scripts/kc_homeo.py`'s rule: each cell's own
  response rate towards target 0.07, deadband [0.04, 0.12], annealed gain, per-cell
  lower clamp), rerun for the separated odour set;
* the reward/punish per-pulse gain ratio (`scripts/plast2_eta.py`'s procedure:
  match `|delta play_drive|` per pulse between the two arms on a conditioning
  panel), rerun for the separated fly.
The `current` encoding keeps `outputs/plast2/tuned_config.json` and
`outputs/plast2/eta_calib.json` exactly as they are.

## 3. The separated encoding, and what is dropped

`separated` spends the same 32 glomeruli differently:

* bits 0-8 (best hand type) keep their single mb2 glomerulus, 34-36 ORNs each;
* bits 9-27 drive **nothing**;
* bits 28-31 (score vs needed) get the 23 glomeruli that frees, partitioned
  greedily largest-first into four **disjoint** ensembles of 5-6 glomeruli and
  **441 / 439 / 441 / 443 ORNs** (max/min = 1.009). Each ensemble keeps the single
  glomerulus mb2 had given that bucket, so it is a strict superset of the old
  channel.

**Why those 19 bits can go.** Measured over the 1,091 decision points of the 60
standard evaluation games:

| bits | block | behaviour at a decision point |
|---|---|---|
| 17-25, 27 | `sel_<type>`, `sel_is_best` | **never on** (0/1091): nothing is selected when the fly is asked |
| 26 | `sel_none` | **always on** (1091/1091): a constant |
| 9-16 | `best_slot` mask | varies, but the harness selects `best_slots` itself whichever way the fly decides, so it cannot change the value of PLAY vs DIG |

So 10 of the 19 carry literally zero bits of information at a decision point, and
the other 8 identify cards the fly has no control over.

**What it costs and what it buys.** The odour universe collapses to 9 hand types
x 4 buckets = 36 possible odours (27 of them occur in the 60 standard evaluation
games, against 393 distinct 32-bit patterns). That is the *whole* state the fly's
decision can depend on anyway -- it is exactly the joint table v2 already
reported -- so generalisation across odours is moot here and each odour gets
~33 training visits at 1,200 hands. The regime the brain sees barely moves:

| per decision point | `current` | `separated` |
|---|---|---|
| active glomeruli, mean | 7.04 | 6.77 |
| driven ORNs, mean (SD) | 430.5 (49.7) | **475.6 (1.3)** |

The near-zero SD is a second, deliberate consequence: `best_slot` has 1-5 bits on
and *how many* is fixed by the hand type, so under `current` the total drive
carries the label by construction (`outputs/mb2/REPORT.md` section 2 flags this).
Under `separated` the intensity is constant to +-3 ORNs and every difference
between two odours is an identity difference. **This is a confound in the
comparison and is declared here**: factor B changes three things at once (bucket
channels widened, irrelevant bits dropped, intensity cue removed). It is one
intervention -- "spend the encoding on the decision axis" -- not three separable
ones, and no claim will be made about which part did the work.

## 4. The bucket-only conditioning probe (runs before the games)

The measurement the v2 "Hamming distance 2" claim needed and never had. Take a
**fixed real decision-point pattern**, change **only** its score bucket (one
bucket bit off, one on, everything else identical), and measure how far a
conditioning pulse on the anchor transfers to those three variants.

Protocol, identical to `scripts/plast2_neighbour.py` except for the probe set:
16 anchors drawn from the 500 held-out bc2 relay patterns, 20 pulses,
`eta = 0.05`, both arms, `relative spread = delta(probe) / delta(anchor)`.
Two contrasts are measured with the same anchors: **bucket-only** (the 3
bucket variants) and **hand-type-only** (variants with the hand type changed and
the bucket held), which bounds how much of any measured transfer is just "a pulse
moves everything".

Reported for **both** encodings, before any game is played. Prediction and
falsification condition are in section 1.

## 5. Primary outcome, analysis, and decision rule

**Primary outcome.** The ante-1 **clear rate under greedy evaluation on the 400
paired seeds**, expressed as the paired improvement over the *same-encoding
frozen control on the same seeds*:

```
D_cell = mean over the 3 training runs of  mean over the 400 seeds of
         ( cleared[run, seed] - cleared[frozen_control(encoding), seed] )
```

**Uncertainty.** Two-way cluster bootstrap, 10,000 resamples, resampling the 400
evaluation seeds with replacement **and** the 3 training runs with replacement
(`plast3_common.cluster_bootstrap`). The v2 write-up's `z = 1.75 over 180 games`
treated 60 games counted three times as 180 independent games; it is not quoted
again. The three per-run paired deltas are printed beside every interval, since
three clusters cannot resolve much on their own and the reader should see that.

**Decision rule, fixed now.** A cell **succeeds** only if, in the primary mode:

1. its paired delta against its own frozen control is positive with a 95%
   clustered CI excluding 0; **and**
2. its paired delta against the **always-discard baseline evaluated on the same
   400 seeds** is positive with a 95% clustered CI excluding 0.

Condition 2 is the bar the task sets: a prettier bucket gradient is not a result.
Always-discard is re-measured on the 400 new seeds (its 0.400 in the v2 report is
a 60-seed number and is not carried over). A cell that passes 1 and fails 2 is
reported as "learned, still not competitive". A cell that fails 1 is reported as
"no effect". Secondary outcomes -- chips, P(play | legal), bucket gradient, the
joint P(play | hand type x bucket) table, sampled-mode versions of all of the
above -- are reported for every cell but cannot promote a cell to success.

**No cherry-picking clauses.** `eta_reward = 0.05` is fixed (v2 found the clear
rate identical at 0.02 and 0.05; only one is run here). Three training seeds per
cell, all three reported, no dropping. Both evaluation modes reported for every
cell. If a run crashes it is rerun with the same seed and the fact is recorded.
Every number in the final table comes from `outputs/plast3/run_<cell>_s<seed>.json`
written by `scripts/plast3_run.py`; nothing is recomputed by hand.

## 6. What would make this round a failure, stated in advance

* All four cells fail condition 1 -> the plasticity rule does not help this fly
  play Balatro, and the cause is not the two things v2 blamed.
* `omission` collapses every cell to always-discard and only ties the baseline ->
  the rule can learn "stop playing" and nothing finer.
* The bucket-only probe shows low transfer under `current` -> the v2 encoding
  diagnosis was wrong, and any H-B improvement needs a different explanation.

All three will be reported as stated if they occur.
