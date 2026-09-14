# Can the fly's own learning rule learn to play Balatro from chips alone? (2026-09-13)

Everything before this round put a *trained readout* on the fly: a linear model or
an MLP fitted by behaviour cloning to a hand-aware teacher. The fly was a
reservoir and the learning happened in scikit-learn. This round removes the
readout and the teacher. The decision comes out of the mushroom-body output
neurons by a fixed rule with no fitted weights, and the only thing that changes
during a run is the weight of the 33,496 Kenyon-cell -> MBON synapses, changed by
dopamine-gated depression driven by chips the game hands out.

**Result in one line.** The fly learned, from chips alone with no teacher and no
trained readout: P(play) went 0.527 -> 0.948 in 9 of 9 runs while a frozen fly with
the same decision rule did not move by a single decision. It learned the wrong
thing: the ante-1 clear rate fell from 0.350 to 0.033, worse than doing nothing,
because 70-75% of every weight change lands on the 188 Kenyon cells that fire for
more than half of the 200 odours (78 fire for every one of them), so the rule is a
nearly odour-independent ratchet toward approach.

---

## 1. What the fly decides, and what the harness does

Stated plainly, because the boundary is the whole point.

**Outside the fly, in ordinary Python that is not biological in any sense:**
`flybalatro/hands.py` enumerates all 218 subsets of size 1-5 of the dealt cards,
classifies and scores each one exactly as `balatro-rs` does, and reports the best.
`flybalatro/features_v2.py` turns that answer into the 32-bit relay block.
`flybalatro/encode.py`'s `GlomerularMap` turns those 32 bits into 30 mV of tonic
current on 32 whole ORN glomeruli -- the "odour" of this hand. After the fly has
chosen, the harness presses the `select_card[i]` keys and then `play` or
`discard`.

**Inside the fly:** one 50 ms window from reset, and a binary choice --
**play the best subset now, or discard the junk and dig.** Discard is offered only
while discards remain and there are cards outside the best subset.

So the fly is not learning poker and is not learning card selection. It is
learning a **valence**: which hand-odours mean approach (play) and which mean
avoid (dig). That is a computation a mushroom body demonstrably does, which is
why this is the first version of this project where "the fly learned" can be
literally true.

The odour carries the best hand type (9 bits), the best-subset slot mask (8), the
type the current selection forms (10, always `sel_none` here because the fly is
always asked with nothing selected), `sel_is_best` (1, always 0 for the same
reason) and the score-vs-needed bucket (4). It does **not** carry how many plays
or discards are left -- those live in the 283 v1 bits which the glomerular
encoding drops. That ceiling is measured in section 6 by the best possible
bucket-only policy.

### Why binary, and why this is the fly-level version of the task

The full game asks for one of 109 actions. A mushroom body does not pick one of
109 things; it assigns a value to an odour and an approach/avoid drive follows.
Making the harness do selection and leaving the fly a valence judgement is the
framing in which the fly's own plasticity rule is the right tool rather than a
mascot on a reservoir.

---

## 2. MBON valence: the assignment, and a conflict with the brief

### 2.1 Where the compartment names come from

They are annotation, not inference. MaleCNS v1.0's `instance` column carries the
Aso et al. 2014 compartment name in parentheses for every MBON and DAN:
`MBON11(y1pedc>a/B)_L`, `MBON18(a2sc)_R`, `PAM08(y4)_R`, `PPL101(y1ped)_L`.
Nothing in this report depends on remembering which number is which compartment.

### 2.2 Where the valence comes from

Aso et al. 2014, eLife 04580 ("Mushroom body output neurons encode valence and
guide memory-based action selection in *Drosophila*") reports the result as a
neurotransmitter rule: **every MBON whose activation drove avoidance was
glutamatergic, and every MBON whose activation drove attraction was GABAergic or
cholinergic** (the M4/M6 cluster -- MBON-g5b'2a, b'2mp, b'2mp_bilateral -- plus
MBON-g4>g1g2, b1>a, a1, b2b'2a on the avoidance side; the V2/V3 a'/a cluster and
the GABAergic MBON-g1pedc>a/b, g3, g3b'1 on the attraction side). MaleCNS ships
a per-neuron neurotransmitter prediction with ground truth for exactly the
classic types, so that rule assigns a valence to all 97 MBONs instead of to the
~9 that have been tested behaviourally.

**This disagrees with the lists in the task brief on 5 of the 9 types the brief
names.** The brief puts a2sc (MBON18), a3 (MBON14) and a'2 (MBON13) on the
avoidance side and g4>g1g2 (MBON05) and b'2mp (MBON03) on the approach side;
Aso 2014's activation experiments put the three cholinergic a'/a cells on the
*attraction* side and the two glutamatergic ones on the *aversion* side. The
brief's list reads like the **memory-expression** role (which MBON's output is
needed to express an aversive memory) rather than the **activation valence**
(which way the fly walks when you switch that MBON on); those two are
opposite-signed for the same cell, which is exactly how depression-based learning
works.

Three reasons the neurotransmitter rule is the one used here:

1. It is the paper's own summary of its own result, and it covers all 97 MBONs
   rather than 9.
2. It makes the circuit self-consistent with the *connectome's* compartment
   assignment (section 2.3): the glutamatergic avoidance MBONs are precisely the
   PAM-innervated ones and the GABA/ACh approach MBONs are precisely the
   PPL1-innervated ones. So reward -> PAM -> less avoidance -> approach, and
   punishment -> PPL1 -> less approach -> avoidance, with no cell in the wrong
   place. Under the brief's list, approach and avoidance MBONs are mixed inside
   the same dopaminergic compartment.
3. It reproduces the two best-known imaging results: aversive training depresses
   KC -> MBON-g1pedc>a/b (GABAergic, approach) and appetitive training depresses
   KC -> M4/M6 (glutamatergic, avoidance).

The brief's assignment is nevertheless run as a condition
(`briefValence_eta0.05_s0`) so the verdict does not rest on this choice.

### 2.3 Where the dopaminergic compartment comes from

Measured, not named: **a KC -> MBON synapse is in the PAM (reward) compartment set
if PAM neurons supply at least half of that MBON's total PAM+PPL1 input weight in
this graph, and in the PPL1 (punishment) set otherwise.** Both sums are taken over
the *brain's* own edge arrays, so the identical rule applies to the shuffled
control (where a global permutation scrambles which MBON each DAN contacts -- the
point of the control). Dopamine populations, exact strings and counts: 316 PAM
neurons in 15 types (`PAM01`-`PAM15`, class `DAN`, 4-50 cells each) and 16 PPL1
neurons in 8 types (`PPL101`-`PPL108`, 2 cells each); `PPL2` (8 neurons, 4 types)
and `PAL` (4 neurons, unsigned, class empty) exist in the graph and are not used.
There are 1,408 DAN -> MBON edges.

On the real graph the rule lands exactly where the anatomy says it should.

### 2.4 The table

97 MBON neurons, 37 types. `NT` is `consensus_nt`; `PAM in` / `PPL1 in` are total
|mV| of dopaminergic input summed over the cells of that type.

| MBON | n | compartment | NT | PAM in (mV) | PPL1 in (mV) | compartment set | valence (Aso NT rule) | brief |
|---|---|---|---|---|---|---|---|---|
| MBON01 | 2 | g5b'2a | glut | 611 | 0 | PAM | avoid | avoid |
| MBON02 | 2 | b2b'2a | glut | 331 | 0 | PAM | avoid | avoid |
| MBON03 | 2 | b'2mp | glut | 1274 | 0 | PAM | avoid | **approach** |
| MBON04 | 2 | b'2mp_bilateral | glut | 448 | 73 | PAM | avoid | - |
| MBON05 | 2 | g4>g1g2 | glut | 898 | 10 | PAM | avoid | **approach** |
| MBON06 | 2 | b1>a | glut | 1059 | 7 | PAM | avoid | - |
| MBON07 | 4 | a1 | glut | 540 | 0 | PAM | avoid | - |
| MBON09 | 4 | g3b'1 | gaba | 966 | 3 | PAM | approach | - |
| MBON10 | 9 | b'1 | gaba | 27 | 3 | PAM | approach | - |
| MBON11 | 2 | g1pedc>a/b | gaba | 28 | 672 | PPL1 | approach | approach |
| MBON12 | 4 | g2a'1 | acet | 0 | 224 | PPL1 | approach | approach |
| MBON13 | 2 | a'2 | acet | 0 | 181 | PPL1 | approach | **avoid** |
| MBON14 | 4 | a3 | acet | 0 | 236 | PPL1 | approach | **avoid** |
| MBON15 | 4 | a'1 | acet | 0 | 34 | PPL1 | approach | - |
| MBON15-like | 4 | a'1a'2 | acet | 0 | 59 | PPL1 | approach | - |
| MBON16 | 2 | a'3ap | acet | 0 | 83 | PPL1 | approach | - |
| MBON17 | 2 | a'3m | acet | 0 | 32 | PPL1 | approach | - |
| MBON17-like | 2 | a'2a'3 | acet | 0 | 15 | PPL1 | approach | - |
| MBON18 | 2 | a2sc | acet | 0 | 119 | PPL1 | approach | **avoid** |
| MBON19 | 4 | a2p3p | acet | 0 | 27 | PPL1 | approach | - |
| MBON20 | 2 | g1g2 | gaba | 0 | 38 | PPL1 | approach | - |
| MBON21 | 2 | g4g5 | acet | 239 | 3 | PAM | approach | - |
| MBON22 | 2 | calyx | acet | 0 | 0 | - | *unassigned* | - |
| MBON23 | 2 | a2sp | acet | 0 | 34 | PPL1 | approach | - |
| MBON24 | 2 | b2g5 | acet | 131 | 0 | PAM | approach | - |
| MBON25 | 2 | g1g2 | glut | 0 | 33 | PPL1 | avoid | - |
| MBON25-like | 4 | g2 | glut | 0 | 22 | PPL1 | avoid | - |
| MBON26 | 2 | b'2d | acet | 348 | 11 | PAM | approach | - |
| MBON27 | 2 | g5d | acet | 121 | 3 | PAM | approach | - |
| MBON28 | 2 | a'3a | acet | 0 | 46 | PPL1 | approach | - |
| MBON29 | 2 | g4g5 | acet | 89 | 0 | PAM | approach | - |
| MBON30 | 2 | g1g2g3 | glut | 2 | 87 | PPL1 | avoid | - |
| MBON31 | 2 | a'1a | gaba | 2 | 199 | PPL1 | approach | - |
| MBON32 | 2 | g2 | gaba | 0 | 293 | PPL1 | approach | - |
| MBON33 | 2 | g2g3 | acet | 14 | 94 | PPL1 | approach | - |
| MBON34 | 2 | g2 | glut | 0 | 0 | - | *unassigned* | - |
| MBON35 | 2 | g2 | acet | 0 | 280 | PPL1 | approach | - |

**Counts.** 66 approach neurons, 24 avoid neurons, **7 unassigned**: both
MBON22 (calyx) cells, both MBON34 cells, and 3 of the 9 MBON10 cells receive no
PAM or PPL1 input at all. Unassigned cells are excluded from the decision pools
and from plasticity.

Confidence: the compartment column and the neurotransmitter column are dataset
annotation (high); the valence column is a literature rule applied to that
annotation -- directly tested behaviourally for about 9 of the 37 types, and
extrapolated by neurotransmitter for the rest (medium for the tested types,
low-medium for the rest). The agreement of the PAM/PPL1 split with the valence
split is a *measurement on this graph*, and it is the strongest evidence that the
extrapolation is not arbitrary.

**Target sets of the two dopamine pulses.**

* Reward (PAM compartments, avoidance MBONs): MBON01-07, 16 neurons,
  **10,886 KC -> MBON edges.**
* Punishment (PPL1 compartments, approach MBONs): MBON11-20, 23, 28, 31, 32, 33,
  35 and the `-like` variants, 44 neurons, **13,582 KC -> MBON edges.**
* MBON09, MBON10, MBON21, MBON24, MBON26, MBON27 and MBON29 are approach-valence
  in PAM compartments, and MBON25, MBON25-like and MBON30 are avoidance-valence
  in PPL1 compartments: neither reinforcer touches them, because the rule is
  valence-*and*-compartment gated. They still contribute to the decision.

---

## 3. The rule

```
play_drive = mean rate(approach MBONs) - mean rate(avoid MBONs) + bias     [Hz]
P(play)    = eps/2 + (1 - eps) * sigmoid(play_drive / T)
```

No fitted weights: the pools come from section 2, the rates are spike counts in
the 50 ms window divided by the window, and `bias` and `T` are two scalars
calibrated **once, before any dopamine**, on 200 decision states drawn from
calibration seeds 300000+ that neither training nor evaluation ever visits.
`bias` puts the median naive state at P(play) = 0.55 (near 50/50, very slightly
play-biased, as specified); `T` is bisected so the mean probability of taking the
*minority* action over those 200 states is 0.20, i.e. the naive fly explores 20%
of the time. Both are frozen for the rest of the run, and both are calibrated
**per wiring**, because the shuffled brain has a different drive distribution and
a bias fitted on the real brain would make the shuffled fly trivially one-sided.

Plasticity, after the action resolves:

```
eligibility  e_k = min(1, KC k's spikes in the decision window / KC_REF),  KC_REF = 2
reward   -> every KC -> MBON edge with MBON in (avoid    x PAM ):  w *= (1 - eta * e_k)
punish   -> every KC -> MBON edge with MBON in (approach x PPL1):  w *= (1 - eta * e_k)
floor        w >= 0.05 * w_original,  and w never rises
```

Depression only, as in the biology and as specified. It is not a dead end even
though it only ever decreases: the two reinforcers act on *opposite* pools, so
`play_drive` moves both ways. `eta` is the single free learning parameter, swept
over {0.02, 0.05, 0.1}.

Reward and punishment come from the game and nothing else:

| event | signal |
|---|---|
| PLAY clears the blind | reward = 1 |
| PLAY gains `chips >= needed_before / plays_left_before` | reward = 1 |
| PLAY loses the blind (last play, still short) | punishment = 1 |
| PLAY, neither of the above | no dopamine |
| DISCARD | no dopamine, ever |

Discards get nothing because crediting an action whose payoff arrives after a
re-deal is delayed credit assignment, which is beyond a fly and beyond this rule.

**An identity worth stating up front.** The harness plays the best subset and
`hands.py` scores exactly as the engine does, so at ante 1 with no jokers
`chips_gained == best_score`, and therefore `reward = 1` is *exactly*
`best_score x plays_left >= needed` -- which is the v2 teacher's fair-share dig
rule. The signal is still entirely game-derived (chips, nothing else, no labels
and no teacher in the loop), but "did it learn the right thing?" is partly true by
construction of the reward. The question the experiment actually answers is
whether the mushroom body can learn the **odour -> reward** mapping with its own
plasticity rule.

---

## 4. Two deviations, both forced, both about APL

### 4.1 The reward arm was landing on cells that never spike

With `outputs/mb2/tuned_config.json` exactly as chosen (`apl_scale = 2`, tonic
30 mV), **every MBON in the reward target set is silent for every input.**
Measured over 40 real hands: MBON01-07 emit 0.00 spikes per state, and their
membrane potentials sit at -75 to -394 mV, i.e. 25 to 340 mV *below* rest, let
alone threshold.

The cause is one neuron. Tracing every synapse delivered to MBON01 in one 50 ms
window: **APL delivers -3,198 mV** against 259 mV of Kenyon-cell excitation and
~290 mV from everything else; for MBON05 it is -3,736 mV against 169 mV.
APL is the giant GABAergic interneuron of the mushroom body, and in the animal it
is *non-spiking* -- it releases GABA in a graded, compartmentalised way (Lin et
al. 2014; Amin et al. 2020). This kernel has no graded transmission, so it fires
APL at ~250 Hz and delivers its full synapse-count weight as spike-triggered
conductance. 94% of APL's output weight goes to Kenyon cells, which is the part
`apl_scale` was introduced to calibrate; the 6% that lands elsewhere includes 72
synapses onto MBONs, and doubling those is what closes the output stage.

Leaving it would have made this experiment a guaranteed null for a reason with
nothing to do with learning: the reward pulse would modify 10,886 synapses onto
neurons whose firing rate is identically zero.

**The correction** is two knobs added to `flybalatro/tuning.py`, both no-ops by
default so every earlier measurement is bit-identical
(`tests/test_tuning.py::test_default_tuning_is_a_no_op` still passes):

* `apl_kc_only = 1` -- `apl_scale` multiplies only APL -> Kenyon cell synapses,
  which is the documented role of the knob and of the neuron.
* `apl_mbon_scale = 0.1` -- a separate gain on the 72 APL -> MBON synapses.

Effect, on 40 real hands at 50 ms: MBON01-07 go from 0.00 to **12.8 Hz** with 11
of 16 cells firing and 8 varying with the input; the whole MBON population goes
from 8.7 to 14.3 Hz with 35 cells varying. The Kenyon-cell code barely moves:
212 active per state against 186, always-on fraction 0.046 against 0.042
(`kc_check.json` measures hand-type decodability under both tunings as well).

`apl5` was tested as an alternative (sparser and far more odour-specific
Kenyon-cell code, always-on/active 0.50 against 0.87) and rejected: with 18
active KCs per state, depressing their output changes MBON firing by exactly
zero, so the plasticity rule has no lever at all.

The mb2 tuning is still run as a condition (`specTuning_eta0.05_s0`) so the
silenced-reward-arm result is on the record rather than in a footnote.

### 4.2 Bookkeeping note on the shuffled control

`Tuning.apply` masks target populations with `graph.post`, not the brain's
permuted `post`, so under the shuffle `apl_kc_only` and `apl_mbon_scale` (like
the pre-existing `alpn_kc_scale` and `kc_kc_scale`) scale a *set of edge slots*
rather than the edges that actually reach KCs and MBONs in the shuffled brain.
This is the existing convention shared with every earlier round; under a global
permutation of 5.15M edge targets it makes little difference, but it is one more
reason the shuffled brain is a different dynamical regime rather than "the same
fly, rewired".

---

## 5. Sanity checks

All three were specified before the protocol ran, on 200 decision states from
calibration seeds 300000+ (137 distinct odours; 49 `ge1.0`, 57 `lt1.0`, 41
`lt0.5`, 53 `lt0.25`; discard legal in 176 of 200). `probe.json`, `probe.log`.

### 5.1 (a) The decision depends on the odour -- PASS

Real wiring, plasticity off. `play_drive` before the bias spans
**-5.45 to +7.80 Hz** (range 13.26 Hz, SD 2.28, 85 distinct values over 137
odours). The quantisation floor of the two pool means -- one extra spike in one
approach MBON plus one in one avoid MBON -- is 1.14 Hz, so the range is
**11.7 quanta wide**. The decision is not a coin flip dressed up: it is a graded
function of the odour.

Mean raw drive by score-vs-needed bucket, which is the decision-relevant axis:

| bucket | n | mean drive (Hz) | SD | naive P(play) |
|---|---|---|---|---|
| `lt0.25` | 53 | 0.53 | 2.28 | 0.393 |
| `lt0.5` | 41 | 1.83 | 2.12 | 0.635 |
| `lt1.0` | 57 | 1.01 | 2.10 | 0.496 |
| `ge1.0` | 49 | 2.46 | 2.11 | 0.683 |

By hand type: High Card 0.52, Two Pair 1.23, Full House 1.15, Straight 1.49,
Flush 1.67, Three of a Kind 1.89, Pair 2.11, Four of a Kind 3.60 (n = 2),
Straight Flush 1.17 (n = 2).

**The naive fly is not neutral**, and this matters for reading the before/after
table. Its P(play) already rises from 0.39 at the worst bucket to 0.68 at the
best -- a bucket gradient of +0.085 -- purely because the frozen connectome
happens to give strong hands a higher approach-minus-avoid drive. Learning has to
be measured against that prior, which is why every condition is reported against
the `frozen` runs on identical evaluation games rather than against 0.5.

Population state: 211.5 Kenyon cells active per state of 4,064; **188 of them
fire for more than half of the 200 odours** (89% of the average active set), and 78
of those fire for every one of the 200; 552 are input-dependent (fire for some but
not all), 3,434 never fire. MBONs 14.27 Hz overall (approach pool 14.69,
avoid pool 13.28).

### 5.2 (b) Synthetic conditioning -- PASSES on direction, FAILS on specificity

The classic result has two halves. Reporting one odour pair cannot separate
learning from a global shift, so each of eight odours (one per distinct
(bucket, hand type) pair) is conditioned in its own fresh run, 20 pulses at
eta = 0.05, and the change in that odour's `play_drive` is compared with the mean
change in the other seven.

| pulse | Δ drive, paired odour | Δ drive, other 7 | specificity | right direction |
|---|---|---|---|---|
| punishment (PPL1) | **-2.33 Hz** (SD 1.28) | -2.46 Hz | +0.13 Hz (**-6%** of the effect) | 7/8 |
| reward (PAM) | **+9.20 Hz** (SD 1.65) | +7.86 Hz | +1.34 Hz (**15%** of the effect) | 8/8 |

**Direction is right, every time.** Punishment paired with an odour lowers that
odour's play drive (7 of 8; the eighth moved by exactly 0, a quantisation floor
effect); reward raises it (8 of 8). Depression-only plasticity on two
compartment-gated pools does move the fly's valence both ways, which is the rule
working as designed. The single A/B pair the brief asks for is in
`probe.json` under `real.sanity_b`: paired odour (High Card / `lt0.25`) -2.33 Hz
versus unpaired (Full House / `ge1.0`).

**Specificity essentially fails.** An odour that was never paired moves 85-105%
as far as the paired one. The cause is measured, not guessed: **70-75% of the
total |Δw| lands on synapses whose presynaptic Kenyon cell fires for every
odour** (70.6% of the eligible edges in the reward set have an always-on
presynaptic KC). The mushroom-body code at this operating point is sparse (5.2%)
but not *specific*: 188 of the 211 active cells are drawn from the same 188 that
respond to more than half of all odours (78 of them to every odour). So a
dopamine pulse paired with one odour is very nearly a dopamine pulse paired with
all of them. This is the single most important limitation of the whole
experiment, and `outputs/mb2/REPORT.md` already flagged its cause
(always-on / active = 0.84 at this setting).

Reward is also **4x stronger than punishment** per pulse (+9.2 against -2.3 Hz),
because the reward pool (MBON01-07, 16 cells at ~12.8 Hz) can be driven to near
silence while the punishment pool includes MBON11, which is GABAergic: depressing
KC -> MBON11 releases its inhibition of the other approach MBONs, and the two
effects partly cancel inside the pool mean. Both facts are properties of the
wiring plus the readout, not free parameters.

**Window length does not fix specificity** (`real.window_scan`; the paired-odour
effect grows but the unpaired one grows with it):

| window | drive range | punishment Δ self / Δ other | reward Δ self / Δ other | reward specificity share |
|---|---|---|---|---|
| 50 ms (used) | 8.94 Hz | -2.33 / -2.46 | +9.20 / +7.86 | **15%** |
| 150 ms | 8.64 Hz | -3.91 / -3.74 | +15.24 / +14.08 | 8% |
| 300 ms | 6.67 Hz | -4.41 / -4.07 | +16.40 / +15.86 | 3% |

50 ms -- the frozen spec's window -- is also the most specific, so there was no
reason to deviate.

### 5.3 (c) Weight bounds -- PASS

After a deliberately saturating run (200 rounds of both pulses at eta = 0.5 on
one odour): no weight negative, none above its original value,
`min_ratio = 0.0500` exactly (the floor), 3.7% of the 33,496 edges at the floor,
96.0% untouched because their presynaptic KC never fired for that odour.
`tests/test_plasticity.py` pins all three invariants plus compartment gating,
eligibility saturation, and the exact depression factor.

### 5.4 The shuffled control is degenerate, and the reason is quantitative

Same tuning, same rule, `Brain.shuffled(0)`. Sanity (a) passes weakly: drive
spans 8.62 Hz = 4.08 quanta, and the naive shuffled fly even has a similar bucket
gradient (0.36 -> 0.58). But **the conditioning test returns exactly 0.000 Hz for
both pulses on all 8 odours.**

The reason is not subtle. A global permutation of 5.15M edge targets scatters the
Kenyon cells' 84,734 outgoing edges uniformly over 146,271 neurons, so the number
that happen to land on the 97 MBONs collapses:

| | real | shuffled |
|---|---|---|
| KC -> MBON edges | **33,496** | **664** |
| reward-gated edges (avoid x PAM) | 10,886 | 76 |
| punishment-gated edges (approach x PPL1) | 13,582 | 111 |
| Kenyon cells active per state | 211.5 | 22.9 |
| MBON mean rate | 14.27 Hz | 2.73 Hz |

With 664 KC -> MBON synapses total and 23 active Kenyon cells, the expected number
of eligible synapses per pulse is a handful, and depressing them changes no
MBON's spike count at all.

**So the real-versus-shuffled row of the eval table is uninformative about wiring
specificity.** The honest statement is that the shuffled fly cannot learn because
the shuffle deletes the pathway the rule acts on, not because the evolved wiring
is better at credit assignment. The 50x convergence of Kenyon-cell output onto 97
MBONs *is* a real property of the evolved connectome, but it is close to a
tautology (the mushroom body has an output stage; a random graph does not), and
it is a much weaker claim than the shuffled control was meant to test. This is
the reverse of the earlier rounds, where a degree-preserving shuffle *beat* the
real wiring because it created sensory -> DN shortcuts; here the same shuffle
destroys the one pathway that matters.

## 6. Results

18 runs: 3 eta x 3 seeds on real wiring, 3 seeds shuffled and 3 frozen at the
pre-registered eta = 0.05, plus one mb2-tuning run, one brief-valence run and one
reward-omission run. Every run: 200 calibration hands (no dopamine), a 60-game
evaluation, 600 training hands on its own seed block, then the *same* 60 games
again with plasticity frozen. ~200 s per run, 62 min total. `summary.json`.

One deviation from section 3 as written: `explore_floor = 0.1`. Sanity 5.1 found
that 30% of the 200 calibration odours put the naive fly outside P(play) in
[0.1, 0.9], and since a DISCARD delivers no dopamine, an odour the naive fly digs
on at p = 0.02 can never generate the experience that would change it. The floor
(`p = 0.05 + 0.9 * sigmoid(drive/T)`) keeps every odour learnable. It is applied
to every condition including the controls.

### 6.1 Harness checks, which all pass

* `frozen` pre and post are **bit-identical** (clear 0.350 / 0.350, chips 824 /
  824, P(play) 0.527 / 0.527, weight `mean_ratio` exactly 1.0000): with eta = 0
  nothing leaks into the eval.
* Naive P(play | discard legal) = 0.527 averaged over three seeds -- the
  calibration transfers from the calibration seeds to the evaluation games.
* No weight in any run went negative or above its original value; the minimum
  ratio is exactly the 0.05 floor in every learning run.

### 6.2 Evaluation table -- 60 ante-1 games, seeds 100000-100059, identical for every row

`grad` is P(play) at the two top score-vs-needed buckets minus the two bottom
ones: positive means the policy plays more when the hand is worth more, which is
the shape a correct valence has.

| row | clear before | **clear after** | chips before | **chips after** | P(play\|legal) before | after | grad before | after |
|---|---|---|---|---|---|---|---|---|
| **real, eta 0.02** (3 seeds) | 0.350 | **0.039** | 824 | **544** | 0.527 | 0.948 | 0.017 | 0.022 |
| **real, eta 0.05** (3 seeds) | 0.350 | **0.033** | 824 | **543** | 0.527 | 0.948 | 0.017 | 0.033 |
| **real, eta 0.1** (3 seeds) | 0.350 | **0.033** | 824 | **549** | 0.527 | 0.947 | 0.017 | 0.028 |
| frozen fly, no dopamine (3 seeds) | 0.350 | 0.350 | 824 | 824 | 0.527 | 0.527 | 0.017 | 0.017 |
| shuffled wiring, eta 0.05 (3 seeds) | 0.417 | 0.417 | 923 | 923 | 0.456 | 0.456 | 0.036 | 0.036 |
| mb2 tuning (reward arm silent), eta 0.05 | 0.383 | 0.250 | 827 | 829 | 0.408 | 0.380 | -0.109 | -0.116 |
| brief's valence lists, eta 0.05 | 0.417 | 0.333 | 869 | 870 | 0.565 | 0.487 | 0.104 | 0.017 |
| reward omission variant, eta 0.05 | 0.267 | 0.033 | 774 | 570 | 0.542 | 0.939 | 0.039 | 0.034 |

Policy baselines on exactly the same 60 games:

| policy | clear | chips | P(play\|legal) | what it is |
|---|---|---|---|---|
| always play | **0.000** | 519 | 1.000 | never digs |
| always discard | 0.400 | 868 | 0.000 | digs whenever it legally can |
| play iff bucket `ge1.0` | 0.383 | 833 | 0.079 | odour-only |
| **play iff bucket `lt1.0` or `ge1.0`** | **0.633** | **1001** | 0.385 | **best odour-only policy: the ceiling** |
| play iff bucket >= `lt0.5` | 0.500 | 1000 | 0.622 | odour-only |
| v2 teacher dig rule | **0.700** | 1088 | 0.569 | sees `plays_left`, which the odour does not |

The naive fly (0.350) sits between always-play (0.000) and always-discard
(0.400), so the game is genuinely discriminating: the ladder runs
0.000 < 0.350 < 0.400 < 0.633 < 0.700 and there is 0.28 of headroom between the
naive fly and the odour-only ceiling.

### 6.3 P(play) before and after, by score-vs-needed bucket

Pooled over the three seeds of each condition; decisions where a discard was
legal. The naive column is the frozen fly, which is the honest "before".

| bucket | n | naive / frozen | real eta 0.02 | real eta 0.05 | real eta 0.1 | shuffled | mb2 tuning | brief valence | teacher's answer |
|---|---|---|---|---|---|---|---|---|---|
| `lt0.25` | ~800 | 0.46 | 0.95 | 0.95 | 0.94 | 0.38 | 0.37 | 0.47 | **0.00** |
| `lt0.5` | ~520 | 0.63 | 0.93 | 0.92 | 0.93 | 0.54 | 0.55 | 0.49 | 0.69 |
| `lt1.0` | ~470 | 0.42 | 0.96 | 0.96 | 0.96 | 0.42 | 0.17 | 0.43 | 0.98 |
| `ge1.0` | ~430 | 0.69 | 0.97 | 0.98 | 0.97 | 0.54 | 0.62 | 0.57 | **1.00** |

And by hand type (real, eta 0.05; naive is the frozen fly on the same games):

| hand type | n | naive | after training |
|---|---|---|---|
| High Card | 152 | 0.395 | 0.939 |
| Pair | 511 | 0.615 | 0.949 |
| Two Pair | 616 | 0.503 | 0.934 |
| Three of a Kind | 75 | 0.547 | 0.950 |
| Straight | 199 | 0.507 | 0.970 |
| Flush | 233 | 0.433 | 0.989 |
| Full House | 279 | 0.548 | 0.935 |
| Four of a Kind | 31 | 0.742 | 1.000 |

The learned policy is flat: every bucket and every hand type ends between 0.93
and 1.00. The bucket gradient does move the right way -- 0.017 naive to 0.033 at
eta 0.05, and `ge1.0` ends 3-4 points above `lt0.25` -- so a small amount of
odour-specific learning is there. It is swamped by a global shift of +0.42 in
P(play). For scale, the same pooled gradient is 0.710 for the teacher rule
(`ge1.0` minus `lt0.25` = 1.00), 0.618 for the three-bucket policy, 1.000 for
`play iff bucket >= lt1.0`, 0.172 for `play iff ge1.0`, and exactly 0.000 for
always-play and always-discard.

These are the two marginals the brief asked for. The joint
P(play | hand type, score-vs-needed) was not produced; with every cell of both
marginals between 0.93 and 1.00 after training it carries no information the
marginals do not already carry, but it is a gap against the letter of the brief.

### 6.4 Learning curves and the mechanism

50-hand rolling windows over the 600 training hands, averaged across seeds
(`learning_curve_p_play`, `learning_curve_reward` in `summary.json`):

| condition | P(play) at hand 1 / 150 / 300 / 450 / 600 | reward rate over the same points |
|---|---|---|
| real, eta 0.05 | 0.67 / 0.95 / 0.96 / 0.93 / 0.95 | 0.67 / 0.37 / 0.41 / 0.31 / 0.40 |
| frozen | 0.67 / 0.63 / 0.66 / 0.57 / 0.55 | 0.67 / 0.35 / 0.27 / 0.37 / 0.34 |
| shuffled, eta 0.05 | 0.33 / 0.53 / 0.51 / 0.49 / 0.53 | 0.33 / 0.33 / 0.38 / 0.32 / 0.31 |

**The curve is a step, not a climb.** P(play) reaches its ceiling inside the
first ~150 hands and then sits there; the reward rate *falls* from 0.40 in the
first third to 0.36 in the last (eta 0.05) because the fly is now playing hands
that do not pay their way. The training dopamine budget for a typical eta = 0.05
run is 692 reward pulses, 230 punishment pulses, 878 hands with nothing -- reward
outnumbers punishment 3:1, and section 5.2 measured each reward pulse as 4x
stronger. That is the whole mechanism.

**0.948 is the exploration ceiling, not the fly's drive.** With
`explore_floor = 0.1` the decision is `0.05 + 0.9 * softmax`, so the maximum
attainable P(play) is 0.95. The trained fly sits on that bound for every hand
type and every bucket, which means behavioural saturation is complete and the
underlying softmax argument has gone further than the behaviour can show. It also
explains an otherwise suspicious coincidence: `omission_eta0.05_s0` is
bit-identical to `real_eta0.1_s0` (chips 570/570, P(play) 0.939/0.939). Diffing
the two runs' evaluation action sequences confirms all 478 decisions are the same
-- both policies are at p ~= 0.95 on every hand, so the shared evaluation RNG
draws the identical sequence. It is saturation, not a shared attractor, and it is
the strongest single piece of evidence that the learned policy has no
hand-dependence left in it.

Weights after 600 hands at eta = 0.05: mean ratio 0.950 over all 33,496
KC -> MBON edges, 2.2% at the 0.05 floor, minimum exactly 0.05. **eta makes
essentially no difference** (0.02 / 0.05 / 0.1 give clear 0.039 / 0.033 / 0.033
and final P(play) 0.948 / 0.948 / 0.947) because all three saturate the synapses
that matter well inside 600 hands; the sweep is flat, so there is no eta at which
this rule does better.

### 6.5 The three variant conditions

* **Shuffled wiring: nothing at all happens.** Pre and post are bit-identical in
  all three seeds. Section 5.4 gives the reason -- 664 KC -> MBON edges instead of
  33,496 -- and it means this row cannot be read as evidence about wiring
  specificity.
* **mb2 tuning (reward arm on silent cells): the fly learns the *other* way.**
  P(play) 0.408 -> 0.380, clear 0.383 -> 0.250. With MBON01-07 unable to spike,
  only punishment can act, so the fly digs slightly more -- and *still* gets
  worse, because its naive bucket gradient is **negative** (-0.109: it plays least
  at `lt1.0`, where the teacher plays 0.98) and punishment-only learning makes a
  wrong-shaped policy more extreme.
* **Reward omission (a full-strength PPL1 pulse after every unrewarded play):
  insufficient, as predicted in advance.** 270 omission pulses plus 77
  punishments against 226 rewards, and the result is indistinguishable from the
  plain rule (clear 0.033, P(play) 0.939). Given the measured +9.2 Hz per reward
  pulse against -2.3 Hz per punishment pulse, three times as many
  punishment-class pulses still loses.
* **The brief's valence lists** move P(play) *down* (0.565 -> 0.487) and lose
  clear rate (0.417 -> 0.333), and their naive bucket gradient (+0.104, the best
  of any condition) is destroyed by training (+0.017). So the verdict does not
  depend on the valence choice: neither assignment learns the right thing.

### 6.6 The Kenyon-cell code is not disturbed by the APL correction

`kc_check.json`, 200 calibration hands, 9-way hand type, logistic probe on
`log1p` spike counts, grouped CV by relay pattern:

| | mb2 tuning (`spec`) | this round (`plast`) |
|---|---|---|
| KC active per state | 183.5 (4.52%) | 211.5 (5.20%) |
| cells active on >50% of odours | 166 (4.08%) | 188 (4.63%) |
| input-dependent cells | 524 | 552 |
| KC hand type | 0.970 (grouped 0.94, permuted 0.235) | 0.960 (grouped 0.94, permuted 0.255) |
| KC binarised hand type | 0.970 | 0.965 |
| MBON rate / varying cells | 8.64 Hz / 33 | **14.27 Hz / 41** |
| MBON hand type | 0.610 | 0.635 |

The correction buys a working output stage (8.6 -> 14.3 Hz, 33 -> 41 varying
MBONs, MBON01-07 from 0.00 to 12.8 Hz) and costs nothing measurable in the
Kenyon-cell code the mb2 round validated.

---

## 7. Verdict

**Did the fly learn anything from chips alone? Yes, unambiguously.**
P(play | discard legal) goes 0.527 -> 0.948 in 9 of 9 real-wiring runs, and the
frozen fly with the identical decision rule and identical random seeds does not
move by a single decision. Nothing was fitted: no readout, no labels, no teacher
in the loop. The only thing that changed is the weight of KC -> MBON synapses,
changed by a coincidence-gated depression rule driven by whether the play made
its share of the chips. The synthetic conditioning test (section 5.2) shows the
same rule moving valence in the correct direction for 15 of 16
odour x reinforcer pairings. Dopamine-gated depression at KC -> MBON in this
connectome is a working learning mechanism.

**Was it the right thing? No. It made the fly worse at Balatro.** Ante-1 clear
rate falls 0.350 -> 0.033 and chips 824 -> 543; the trained fly is worse than the
naive fly, worse than always-discard (0.400), worse than every bucket-only policy
(0.383-0.633), and close to always-play (0.000), which is the worst policy in the
game. What the fly learned was one number -- "play" -- for every odour. The
evidence is in three places: P(play) ends between 0.93 and 1.00 for all four
score buckets and all eight hand types; the bucket gradient barely moves
(0.017 -> 0.033 against 0.710 for the teacher, or ge1.0-minus-lt0.25 of 1.00);
and the learning curve is a step
that finishes inside 150 hands and then flattens while the reward rate declines.

**Why, in one sentence, and it is not a bug:** the reward signal is available on
42% of plays and the punishment signal only when a blind is actually lost, each
reward pulse moves the play drive four times as far as each punishment pulse, and
**70-75% of every weight change lands on the 188 Kenyon cells that fire for more
than half of all odours** -- so the rule is a nearly odour-independent ratchet toward
approach. Only about 15% of the reward effect and ~0% of the punishment effect is
odour-specific (section 5.2), and 15% of a +0.42 shift is the +0.02 of gradient
that actually appeared. The mushroom-body code here is sparse (5.2% of cells) and
perfectly decodable (KC hand type 0.96-1.00) but **not specific**: 188 of the 211
active cells come from the same >50%-of-odours pool, and compartment-level credit
assignment
needs the opposite. The fix is upstream of plasticity, in the Kenyon-cell code,
and `apl5` -- the one setting in the mb2 grid with a genuinely specific code
(always-on / active 0.50 rather than 0.87) -- has only 18 active KCs, too few to
move an MBON at all. That is the real result of this round: **the plasticity rule
works and the code it has to work on does not support it.**

Three things worth keeping separate from that verdict.

* **The reward is the teacher's rule in disguise.** Because the harness plays the
  best subset and `hands.py` scores exactly as the engine does,
  `chips_gained == best_score` and `reward = 1` is exactly
  `best_score x plays_left >= needed`, which *is* the v2 teacher's fair-share
  criterion. No teacher is consulted at run time and the signal is nothing but
  chips, but "the right thing" was defined by the same inequality the teacher
  uses, so the experiment tests whether the mushroom body can learn that
  odour -> value mapping, not whether it can discover a good policy from scratch.
* **Real versus shuffled is not a usable comparison here.** The shuffle does not
  rewire the mushroom body, it deletes it: 33,496 KC -> MBON edges become 664, and
  the shuffled fly's weights and behaviour are bit-identical before and after
  training. The row belongs in the table because it was pre-registered, but the
  only thing it establishes is that the evolved connectome's Kenyon-cell output
  converges on the 97 MBONs about 50x more than chance -- true, and close to a
  tautology. Earlier rounds found shuffled wiring *beating* real wiring as a
  reservoir; here the same shuffle removes the pathway the experiment is about.
* **Two knobs were added to make the experiment possible** (section 4): at the
  mb2 operating point every MBON the reward pulse can reach is 25-340 mV below
  rest for every input, because a spiking model of the non-spiking APL neuron
  dumps -3,200 mV into MBON01 in 50 ms. That condition was run too, and it
  produces a *different* wrong answer rather than a null.

**Is the naive fly's valence map any good to begin with?** Slightly, and by
accident. The frozen connectome plays `ge1.0` hands at 0.69 and `lt0.25` at 0.46
with no learning at all, because strong-hand odours happen to drive a higher
approach-minus-avoid difference. Every learning condition destroyed or diluted
that accidental structure rather than sharpening it.

**What would have to change for this to work.** A Kenyon-cell code whose active
set is mostly odour-specific rather than mostly shared, which is a question about
the calyx and the APL feedback loop, not about the plasticity rule; a punishment
signal with the same event rate and per-pulse gain as the reward signal, which in
the game means punishing plays that under-deliver rather than only plays that lose
the blind; and probably a synaptic recovery term, since a depression-only rule
with a 20:1 imbalance between pool sensitivities has one absorbing state and this
fly walked straight into it.

## Artifacts

`outputs/plast/`: `REPORT.md` (this file), `tuned_config.json` (the frozen spec
plus the two APL knobs and the reason), `probe.json` + `probe.log` (sanity checks
and calibration), `calib.json` (the 200 calibration hands and the frozen
bias/temperature per wiring), `kc_check.json` (Kenyon-cell code under both
tunings), `baselines.json` (policy baselines on the eval games),
`run_<name>.json` (one per protocol cell), `summary.json`, `run.log`.

Code: `flybalatro/plasticity.py` (valence assignment, compartment gating, the
depression rule, the decision rule), `scripts/plast_common.py` (the harness),
`scripts/plast_probe.py`, `scripts/plast_run.py`, `scripts/plast_summary.py`,
`scripts/plast_kccheck.py`, `tests/test_plasticity.py`.

**Not done, and named as such.** (i) The brief's optional flagged variant -- a
mild punishment for discarding when the best available hand already exceeds the
blind -- was not run; reward-omission was run in that slot instead, on the
argument that the measured 4:1 per-pulse asymmetry makes any punishment-count fix
insufficient, which the omission run then confirmed. (ii) The
P(play | hand type, score-vs-needed) joint is not produced (section 6.3). (iii)
No condition was run at a Kenyon-cell operating point with a genuinely specific
code, because no such point exists in the mb2 grid that also drives MBONs -- that
is the follow-up this round points at, and it is an encoding change, not a
plasticity change.

**One file outside this round's own paths was touched.**
`flybalatro/tuning.py` gained two fields, `apl_kc_only` (default 0.0) and
`apl_mbon_scale` (default 1.0). Both are no-ops at their defaults, `from_dict`
still accepts every previously valid config dict unchanged, and the full suite is
206 passed / 1 skipped (the skip is pre-existing:
`tests/test_realgame_brain.py:200`). Nothing under `outputs/bc*/`, `outputs/mb/`,
`outputs/mb2/` or `scripts/bc_*` was modified.

Reproduce:

```
source .venv/bin/activate
python -m scripts.plast_probe --window-scan 50,150,300   # sanity + calibration
python -m scripts.plast_kccheck                          # KC code, both tunings
python -m scripts.plast_run --all                        # the protocol, resumable
python -m scripts.plast_summary
python -m pytest tests/
```
