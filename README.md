<p align="center">
  <img src="figures/demo.gif" width="720" alt="one browser window holding the whole demo: the MaleCNS point cloud firing on the left, the real Balatro window streamed into a panel on the right with an amber box on the cards the fly is about to play, and the mushroom-body readout under it">
</p>

<p align="center"><sub>
Left, the fly brain firing on the decision it just made. Right, the real
Balatro window streamed into the same page, the fly's pick boxed, its
mushroom body underneath. The box is a snapshot of one decision, so it trails
once the cards move. Fifteen seconds of the
<a href="RESULTS.md#the-learning-fly-on-the-real-game-and-an-overlay-that-tracks-the-cards-2026-09-13">learning fly</a>,
recorded for this clip.
</sub></p>

# fly-balatro

The complete male fruit-fly connectome, run as a frozen spiking network,
playing Balatro. The boundary between the fly and the harness is drawn where it
sits, and every control that would have made the headline smaller was run and
reported.

Google and Janelia released **MaleCNS v1.0** on 2026-09-03: 166,700 neurons,
the whole male *Drosophila* central nervous system. A wave of "fly brain plays
Doom / Mario 64 / Deadlock" followed. Nearly all of it is a **trained readout on
a frozen spiking simulation**, meaning a small classifier fitted outside the
brain that reads the spikes of a network whose wiring never changes, captioned
as though the neurons were taught something, and some of it is scripted.
Building that is worth doing. The caption is the problem.

This repository is the same genre with the caption corrected, then pushed until
it broke in ways worth recording.

---

## In plain terms

If you already work on connectomes or spiking networks, skip to
[The honest boundary](#the-honest-boundary), where the detail starts, and read
`RESULTS.md` for the full lab notebook. The rest of this section is for everyone
else.

A connectome is a map of a brain traced from electron-microscope images of a
real animal: every nerve cell, every place one cell passes a signal to another,
and how many of those contacts each pair shares. Google and Janelia finished one
for the whole central nervous system of a male fruit fly and put it online. What
they released is a wiring diagram. It says who connects to whom, how many
contacts they share, and a prediction of whether a cell excites or quietens what
it touches. It does not say how strongly any one contact pulls, and it is not a
recording of a living brain.

This project loads that diagram and runs it as a simulation. Each cell collects
charge from the cells feeding it, leaks some of it away, and fires when it
crosses a threshold. The wiring stays fixed while the fly plays. Under the first
of the two setups below, nothing inside the fly changes; under the second, one
set of synapses can weaken and nothing else moves. Neither one teaches the fly a
game.

Balatro is a single-player card game built out of poker hands. You hold a hand
of cards, choose up to five of them, and the game pays chips for whatever poker
hand they make. Reach the chip target before you run out of plays and you clear
the round. Miss it and the run is over. Each round is called a blind, and three
blinds in a row make an ante. You can also throw cards away and draw
replacements instead of playing, a limited number of times.

The fly never sees the cards. Ordinary code reads the hand, checks every
combination of cards that could be played, works out which one scores best, and
compresses that answer into a short list of yes-or-no facts: which poker hand is
the best one available, which cards make it, what is selected right now, and how
that best score compares with what the round still needs. Each fact switches on
one smell channel, and those channels are the only thing the fly receives.
Smell, because the fly's nose is the part of its brain that neuroscience
understands best, and because the circuit a fly uses to learn what to approach
and what to avoid sits one step downstream of it.

Something then has to turn spikes into a move, and this project tries two ways.

* **A small decoder trained outside the fly** watches which cells fired and
  picks the move. It was fitted on the moves an expert program chose, so
  whatever it gets right measures how much of the message survived the trip
  through the brain. It measures nothing the fly worked out.
* **No trained decoder for the move.** The fly's own output cells vote, under a
  fixed rule set by the chemical each of those cells releases, and the vote is a
  plain yes or no to the hand the program already picked. The chips the game pays
  become dopamine pulses, computed by the harness and not by the fly, and each
  pulse weakens particular connections: a reward weakens the cells voting avoid,
  a punishment weakens the cells voting approach. Four quantities on this path
  are still calibrated in advance. None of them sees a move, a label or an
  outcome, and they are named again at every claim that rests on them.

What came out of it:

* The fly does not play poker. The program plays the poker and the fly is
  handed the answer as a smell.
* The fly's particular wiring adds nothing measurable under the trained decoder.
  Swap the fly for a fixed random scramble of the same facts and the decoder
  does as well, or a little better.
* The fly's own learning rule does move its behaviour, by a wide margin over a
  frozen copy of itself. It still only ties a fixed rule that discards whenever
  discarding is legal and plays only when it must. The pass mark for that
  experiment, written down before a single game was played, required beating
  that rule too. It did not. That is a miss. What stops it is not the way the
  chips were turned into reward, which was tested and replaced: the rule can only
  ever *weaken* connections, and the ones it would have to weaken to do better
  are already as weak as the rule allows.
* Rewiring the one stage that was rewired fairly leaves every measurement where
  it was. An earlier claim that the real wiring beats a shuffled one is
  **retracted**: that shuffle left the network nearly silent, so it compared a
  working brain against a dead one.
* It does drive the real game on Steam, and it cleared the first ante in one of
  three runs.

<details>
<summary><b>Glossary</b>: the terms this file leans on</summary>

**The fly**

| term | what it is, and what it does here |
|---|---|
| **connectome** | A map of a brain traced from electron-microscope images: the cells, the contacts between them, and how many contacts each pair shares. |
| **glomerulus** (plural **glomeruli**) | One of the round hubs in the antennal lobe; all the receptor cells carrying the same odour channel converge on one hub, so lighting one glomerulus is close to presenting one smell. |
| **ORN**, olfactory receptor neuron | The sensory cells in the antenna, the fly's nose; each carries one odour channel into one glomerulus. |
| **ALPN**, antennal-lobe projection neuron | The relay out of a glomerulus, carrying the processed smell onward to the memory centre. |
| **mushroom body** | The fly's memory centre for smell, where an odour gets tied to something good or bad. |
| **calyx** | The input end of the mushroom body, where projection neurons hand over to Kenyon cells. |
| **Kenyon cell**, **KC** | Where the fly stores smell memories; each cell answers to a small, specific combination of odour channels, which is what keeps one smell distinguishable from another. |
| **APL**, anterior paired lateral neuron | One large inhibitory cell per side that damps the whole Kenyon-cell population, leaving only the strongest few firing. |
| **MBON**, mushroom-body output neuron | The output cells of the memory centre; they vote approach or avoid, and the balance between those votes is the decision. |
| **PAM** and **PPL1** | The two dopamine cell clusters, PAM signalling that something good happened and PPL1 something bad; they are what writes a memory into the Kenyon-cell-to-MBON connections. |
| **DN**, descending neuron | The only cells running from the brain down to the body, so every action the fly takes leaves through them. |
| **MN9** | The motor neuron that extends the proboscis to feed; the end point of the standard sugar sanity check. |
| **leaky integrate-and-fire**, **LIF** | The cell model used here: charge builds up from arriving spikes, leaks away over time, and the cell fires when it crosses a threshold. |
| **tonic drive** | An input held steadily on for the whole window, rather than a brief pulse. |
| **histaminergic** | Releasing histamine, which in a fly quietens the receiving cell instead of exciting it. |

**The game and the harness**

| term | what it is, and what it does here |
|---|---|
| **blind** | One round of Balatro: reach the chip target with the hands you have, or the run ends. |
| **ante** | A set of three blinds, small, big and boss. Clearing an ante means clearing all three. |
| **relay bits** | The yes-or-no facts the outside-the-brain hand analysis compresses into, and the whole of what the fly is given. |
| **reservoir** | Using a fixed, untrained network as a scrambler and training only a small readout on its output. The fly is the reservoir. |
| **readout** | The small trained model that turns spike counts into a move. |
| **behaviour cloning** | Fitting that readout to copy an expert program's move, hand by hand. |
| **no-brain control**, **no-brain ceiling** | The same readout given the same relay bits with the fly taken out, which bounds how well anything downstream of the relay can do. |
| **ablation** | Switching off part of the system to see what the rest manages without it. |
| **degree-preserving shuffle** | Rewiring at random while keeping every cell's number of connections and their strengths, so only the choice of partner changes. |

**Reading the statistics**

| term | what it is for |
|---|---|
| **grouped cross-validation** | Scoring a probe with whole groups of near-identical inputs held out together, so it cannot pass by memorising something it already saw. |
| **Wilson interval** | The bracketed range printed after a clear rate: the true rates consistent with that many wins out of that many episodes. |
| **cluster bootstrap** | The bracketed range printed after a plasticity difference; it resamples the evaluation seeds and the training runs both, so three runs over reused seeds do not get counted as independent games. |
| **exact McNemar test** | A test for two policies played on the same seeds, which looks only at the games where they disagreed and asks whether the disagreements fall one way more often than chance allows. |
| **interaction contrast** | How much two changes made together beat what you would get by adding up what each one gives alone. |
| **preregistration** | Writing the protocol and the pass mark into the repository before running the experiment, so the bar cannot move afterwards. |

</details>

---

## The honest boundary

| | |
|---|---|
| **The fly does not see the cards.** | `flybalatro/hands.py` enumerates all 218 subsets of size 1–5 of the dealt hand, classifies and scores each one exactly as the engine does, and picks the best. That is ordinary Python and it is not biological in any sense. |
| **The fly is told an odour.** | The answer becomes 32 binary relay bits, and each bit drives every olfactory receptor neuron (ORN, a sensory cell in the antenna) of one whole glomerulus, the antennal-lobe hub where a single odour channel arrives: ~6.8 of 32 glomeruli lit per state, 30 mV of tonic drive, held on for the whole window. That is the entire input. The other 283 state bits are dropped. |
| **The fly is frozen.** | 146,271 brain neurons, 5,146,572 edges at ≥5 synapses, leaky integrate-and-fire (charge in, leak out, spike at threshold) with Shiu et al. 2024 constants. One 50 ms window from reset per decision. No weight changes, except on the plasticity path below. |
| **Then one of two things decides.** | **(a)** a trained readout, linear or a 1×256 MLP (a small neural network) on log1p spike counts, fitted to 120,755 teacher actions. Or **(b)** no trained action readout at all: the fly's own mushroom-body output neurons (MBONs, the cells that vote approach or avoid), `mean rate(66 approach) − mean rate(24 avoid) + bias`, and dopamine-gated depression of its 33,496 synapses from Kenyon cells to those output neurons (KC→MBON) driven by the chips the game pays. |

![the pipeline, and where the boundary is](figures/01_pipeline.png)

### What is fitted, what is only calibrated, and what is ours

**"No trained action readout" is not "nothing is fitted."** Dopamine is an
**imposed harness signal** computed from the game's chips, not something the fly
generates, and four quantities on path (b) are calibrated before any learning
starts. None of them sees an action, a label or an outcome; all four are named
again at every claim that depends on them.

| | path | what it is fitted or calibrated to |
|---|---|---|
| **trained action readout**, linear or a 1×256 MLP on log1p spike counts | (a) | **Fitted.** The teacher's chosen action on **120,755** held-in states. Everything it gets right is information that survived the trip through the fly. |
| **decision bias** and **softmax temperature** (−2.616 and 0.838 for the v2 fly and the real game; −1.712 and 0.815 for the separated-encoding fly the headline plasticity numbers come from) | (b) | Calibrated, **label-free**, from 200 hands before any learning. |
| **4,064 per-Kenyon-cell homeostatic thresholds**, each cell's own spike threshold, set so it answers to a small fraction of odours the way a real Kenyon cell does | (b) | Calibrated, **label-free**: each cell sees only its own firing rate. |
| **punishment / reward per-pulse gain ratio** (3.81; 1.59 for the separated-encoding fly) | (b) | Calibrated, **label-free**, by matching absolute weight change per pulse. |
| **the gains on APL**, the anterior paired lateral cell that damps the whole Kenyon-cell population, **one global conductance, the uniform −45 mV spike threshold** | both | Neither fitted nor measured. **Ours**. The connectome gives topology, synapse counts and a predicted sign per neuron; it does not give per-synapse efficacy or spike thresholds. Listed in [`outputs/mb/REPORT.md`](outputs/mb/REPORT.md) §1. |
| **the 5,146,572 connectome edges themselves** | both | **Frozen.** Nothing moves on path (a) at all; on path (b) only the 33,496 KC→MBON weights move, under dopamine-gated depression. |

---

## Headline results, with their controls

Every row is a 400-episode ante-1 evaluation on shared seeds unless stated. A
blind is one chip target and an ante is three of them, small, big and boss, so
an episode clears the ante only by clearing all three in a row. The bracketed
ranges after a rate are Wilson intervals, the true rates consistent with that
many wins out of that many episodes. The bracketed ranges on the plasticity row
are cluster bootstraps of a paired difference, resampling the evaluation seeds
and the training runs both.

| | result | the control beside it |
|---|---|---|
| **The fly plays real Balatro** | 3 runs, **6 of 9 blinds cleared**, ante 1 cleared in run 3, **zero rejected API calls** | 3 runs is an existence proof, not a rate. No comparison with the headless number is claimed. |
| **Ante-1 clear rate, real wiring** (v3, reading the antennal-lobe projection neurons, the Kenyon cells and the descending neurons together) | linear **0.378** [0.331, 0.426], MLP **0.395** [0.348, 0.444] | the *same* MLP on the same 32 bits **without the fly**: 0.422. A random projection of those bits, linear: 0.438.   |
| **Plays the best available subset** (v3, real wiring, linear) | **0.810** of plays | the no-brain control on the same bits: 0.841. The fly costs 3 points. Under v2's encoding it was 0.112 against 0.927. |
| **Hand type decodable from Kenyon cells**, the cells that store the fly's smell memories | **1.000** (0.995 under grouped cross-validation, which holds near-identical inputs out together so nothing passes by memorising) | chance 0.117; driven-ORN-count-only confound 0.267. The old encoding's KC probe, on an *easier* binary label, managed 0.598 on a different task, so it is not a controlled comparison. |
| **…at the descending neurons** (DNs), the only cells carrying anything from brain to body | **0.842** | but the DN *intensity* probe is 0.918, so DNs still carry how much more cleanly than what |
| **Does evolved calyx wiring help?** The calyx is where smell relays hand over to the Kenyon cells | **No.** `real` sits inside the 3-seed rewired null on every measure | 12 paired comparisons, smallest p = **0.13** on an exact McNemar test, which counts only the games where two policies played the same seed and disagreed. Agrees with Caron et al. 2013. |
| **Does the fly's own learning rule learn?** | **Yes: +0.275** [+0.210, +0.338] over its own frozen control, under the best reward v4 found (v3's reward gives +0.234 [+0.141, +0.320]) | and it **ties** always-discard, a fixed rule that discards whenever that is legal and plays only when it must: +0.065 [+0.000, +0.130], p = 0.058, Holm-corrected 0.117. The preregistered rule, written down before any game was played, required beating both. **That is a miss.** |
| Label-free recalibration, for scale | per-KC homeostatic thresholds move clear rate 0.265 → **0.380** | in every wiring condition equally. Calibration matters more than wiring here. |

Two claims that were **retracted** rather than quietly dropped, and one bug that
is upstream's:

* **"Real wiring beats shuffled" is retracted.** Under the glomerular encoding a
  global degree-preserving shuffle, which rewires the whole brain at random
  while keeping every cell's connection count and strengths, destroys the
  convergence the encoding depends on and leaves a near-silent network, so it
  compares a working brain against a dead one. The [calyx control](#4-the-evolved-calyx-wiring-does-not-help)
  replaces it and answers the other way.
* **"The fly brain garbles its input" was an artefact of the encoding**, not a
  property of the fly. See [below](#1-the-encoding-reversal).
* The `balatro-rs` action mask offered an action its own handler rejected, about
  1 episode in 3,000. Fixed in `patches/balatro-rs.patch`, and we think it should
  go upstream.

---

## The findings

### 1. The encoding reversal

The first version gave each input bit 10 randomly scattered receptor neurons, so
~36 active bits drove ~360 receptors into **all 54 glomeruli on every input** and
the total antennal-lobe spike count varied by a coefficient of variation of
0.031. Under that encoding identity died one synapse in: ALPN (the projection
neurons leaving the antennal lobe) 0.705, Kenyon cells 0.598, descending
neurons 0.605 against a 0.593 majority floor, while the *same* DN spikes
decoded "how
many bits are on" at 0.728. The obvious reading was that the fly destroys what
you give it.

The input was the problem. One relay bit = one whole ORN type, ~6.8 lit per
state, and
hand identity survives to Kenyon cells at **1.000** (grouped by relay pattern,
0.995) and descending neurons at **0.842**, with the count-only confound at 0.267.
The confound number is the score a probe gets from how many receptors were
driven and nothing else, so beating it is what shows the fly carries *which*
hand rather than *how much* input.

![decoding by population under the two encodings](figures/02_encoding_reversal.png)

→ [RESULTS.md § Glomerular encoding](RESULTS.md#glomerular-encoding-2026-09-13) ·
full write-up [`outputs/mb2/REPORT.md`](outputs/mb2/REPORT.md) ·
prior round [`outputs/mb/REPORT.md`](outputs/mb/REPORT.md)

### 2. Behaviour cloning: what survives the trip

Behaviour cloning here means fitting the readout, the small model on top of the
spikes, to copy the move an expert program made, hand by hand. Same states, same
teacher, byte-identical episode split, same evaluation seeds across v2 and v3.
Only the encoding and the brain's calibration change.

Under v2, every real-wiring readout cleared **0 of 400 episodes** and found the
best available subset on 11 % of plays, where the same readout without the brain
found it on 93 %. Under v3 the best real-wiring readout clears **37.8 %** and
finds the best subset on **81 %**, against a no-brain control at 84 %. The
remaining gap is 3 points.

Read the **linear** column: at the MLP every condition is within 0.017 of the
32-bit input's ceiling, so the MLP column is saturated and cannot separate
wirings at all.

![ante-1 clear rate by condition, v2 and v3](figures/03_behaviour_cloning.png)

→ [RESULTS.md § v2](RESULTS.md#v2-hand-type-relay-2026-09-13) ·
[§ v3](RESULTS.md#v3-glomerular-encoding-2026-09-13)

### 3. Plasticity: the fly's own rule, and what it actually learned

No trained action readout, no teacher. The decision comes out of the
mushroom-body output neurons by a fixed valence rule, approach or avoid assigned
from the chemical each cell releases, and the only mutable thing in the fly is
the weight of its 33,496 KC→MBON synapses under dopamine-gated depression driven
by chips. Depression means a pulse only ever weakens a synapse: a reward pulse,
carried by the PAM dopamine cluster, weakens the path to the avoid cells, and a
punishment pulse, carried by PPL1, weakens the path to the approach cells.

Four rounds, and the frozen control is bit-identical every time (0 of 33,496
synapses move), so the learning effect is real:

* **v1** learned "always play", which is the worst policy in the game. Clear rate
  0.350 → 0.033.
* **v2** added per-Kenyon-cell homeostatic thresholds. This is the one
  intervention that made credit assignment odour-specific (reward specificity
  15 % → 71 %) and it worked: the fly learned the right *shape*, with cell-by-cell
  correlation against what the situation deserves going −0.065 → **+0.881**,
  without converting it into wins.
* **v3** was **preregistered**, with the protocol and the pass mark committed
  before the experiment ran ([`outputs/plast3/PREREGISTRATION.md`](outputs/plast3/PREREGISTRATION.md),
  written before a single game was played) as a 2×2: punish every play that fell
  short of its fair share, × give the score bucket its own disjoint glomerulus
  ensembles. **The two factors are useless apart and large together**:
  −0.023 and +0.022 alone, **+0.234** together, an interaction contrast of +0.171.
  An interaction contrast measures how much the two changes made together beat
  what adding up each change alone would give.
* **v4** was preregistered as well
  ([`outputs/plast4/PREREGISTRATION.md`](outputs/plast4/PREREGISTRATION.md),
  again written before any game here was played) and asked whether **any** reward
  computable from the game beats always-discard. None does. The best of them, an
  eligibility trace that carries a lost blind's punishment back over the plays
  that led into it, produces the best learned fly in the project, **0.448** in all
  three training seeds, **+0.275** [+0.210, +0.338] over its frozen control, and
  **+0.065** [+0.000, +0.130] against always-discard at p = 0.058,
  Holm-corrected 0.117. Another miss, by the width of a rounding.

v3 read the stall as a failure of the reward specification: its fly stopped at
the optimum of its own reward function, and the dopamine ledger said so, with the
`< 0.5×` bucket net reward-positive (139 rewards against 97 punishments). **v4
overturned that reading.** The thing the reward would have to say, "this hand paid
its share but the blind was still lost", cannot be expressed by any immediate
same-hand reward term at all, and it happens 0 times in 2,374 plays; only a trace
can reach it, and the trace flips that bucket's ledger without changing the policy
at all. A post-hoc probe then hands the rule the best signal any reward or credit
scheme could ever produce, punishment aimed at exactly the odours that need it,
and it still cannot get there: **83.6 % of the synapses such a pulse can reach are
already pinned at the 5 % weight floor**, 120 more pulses move the decision drive
by 0.5 %, and the drive that is left is there because the *reward* pulses
depressed the avoid side, which a depression-only rule has no operation to undo.
**The binding constraint is a capacity limit of depression-only KC→MBON
plasticity, not the reward specification and not credit assignment.** A
bidirectional rule would have the operation this one is missing; that was outside
the frozen spec every round ran under, and nothing was changed to chase it.

![the 2x2 interaction, and P(play) by score bucket](figures/04_plasticity.png)

→ [RESULTS.md § Plasticity](RESULTS.md#plasticity-does-the-flys-own-learning-rule-learn-balatro-from-chips-2026-09-13) ·
[§ v2](RESULTS.md#plasticity-v2-kc-homeostasis-2026-09-13) ·
[§ v3](RESULTS.md#plasticity-v3-the-audit-fixes-and-a-preregistered-2x2-2026-09-13) ·
[§ v4](RESULTS.md#plasticity-v4-can-any-game-computable-reward-beat-always-discard-2026-09-14) ·
[`outputs/plast/REPORT.md`](outputs/plast/REPORT.md) ·
[`outputs/plast2/REPORT.md`](outputs/plast2/REPORT.md)

### 4. The evolved calyx wiring does not help

The project's central claim used to be that evolved wiring beats a shuffle. That
control broke, so it was replaced with a narrow one: rewire **one stage**,
19,980 ALPN→Kenyon-cell edges, swapping endpoints within 818 strata of
(hemisphere, KC subtype, synapse count, sign), so every in-degree, out-degree,
in-weight, out-weight and per-projection-neuron subtype profile is preserved
exactly.

Two things make it worth more than the control it replaces. The harness
reproduces `outputs/bc3` **bit for bit** when the rewiring is absent, so nothing
is absorbing the effect; and unlike the global shuffle, the rewired networks land
in the same activity regime **with no recalibration at all**. `real` then sits
inside the three-seed null band, the range the rewired controls themselves
cover, on decoding, on Kenyon-cell code structure, on
imitation and on live clear rate, below all three on KC imitation, and none of
the twelve paired comparisons is significant.

Caron et al. 2013 report projection-neuron-to-Kenyon-cell (PN→KC) connectivity
in *Drosophila* as largely random with respect to glomerular identity. A
near-null here **agrees with the biology**, and a large effect would have been
the surprise.

![the calyx null](figures/05_calyx_null.png)

→ [RESULTS.md § Calyx rewiring control](RESULTS.md#calyx-rewiring-control) ·
full write-up [`outputs/calyx/REPORT.md`](outputs/calyx/REPORT.md)

### 5. Two defects in the model this whole genre uses

Both found here, both load-bearing for anyone building the same thing:

* **Fly photoreceptors are histaminergic, i.e. inhibitory.** In a simulation with
  no background activity, the eyes therefore emit *nothing* downstream, so a
  visual "fly brain plays X" drives a dark network unless it fixes this. (`doomfly`
  works around it with a tonic 12 mV bias on the lamina, the first processing
  layer behind the eye.) Driving the excitatory
  optic-lobe cells instead lands in the same regime as everything else: DN
  identity 0.60/0.54, intensity 0.67.
* **The standard sugar→MN9 sanity check fails on specificity**, before any tuning
  of ours. The check drives the taste neurons in the bristles on the fly's lip
  and expects MN9, the motor neuron that extends the proboscis to feed, to
  answer. Those gustatory neurons give MN9 1.2–3.2 spikes against
  4.0–6.0 for 165 matched random ORNs. Whatever that check is showing, it is not
  a sugar-specific pathway in this model. (MaleCNS v1.0 also carries no "sugar",
  "Gr64" or "Gr5" annotation string at all.)

→ [RESULTS.md § Gate probe](RESULTS.md#fly-brain-plays-balatro--results-2026-09-12) ·
[`outputs/mb/REPORT.md`](outputs/mb/REPORT.md) §2 ·
[`outputs/mb2/REPORT.md`](outputs/mb2/REPORT.md)

---

## What this is not

* **It is not a fly learning poker.** The poker is solved outside the brain in
  ordinary Python and relayed in as an odour. Every number here measures how
  much of that relay survives the trip, or what the fly's own valence rule does
  with it once it arrives.
* **It is not evidence that the connectome computes.** On the one stage tested
  properly, its individual wiring contributes nothing measurable, and a
  label-free recalibration of excitability moves the clear rate further (0.265 →
  0.380) than any wiring manipulation did.
* **It is not a simulation of a fly.** The connectome gives topology, synapse
  counts and a predicted sign. It does not give per-synapse efficacy, receptor
  identity, input resistance or spike threshold. One global conductance, a
  uniform −45 mV threshold and a handful of APL gains are ours, not data. They
  are listed in [`outputs/mb/REPORT.md`](outputs/mb/REPORT.md) §1.
* **It is not a benchmark.** Ante 1 only, no jokers, no enhancements (the
  modifiers a full Balatro run is built around), one connectome, one window
  length, one teacher. Three real-game runs.
* **It is not statistically powerful everywhere.** 400 episodes does not separate
  39.5 % from 40.5 %; 3 real-game runs separate nothing. Where an interval is not
  reported, assume it would cross zero.
* **The overlay's Lua side is only partly verified.** ~120 lines were written,
  installed and unit-tested against real pixels. A later session did launch the
  game with them, and every decision record in
  `outputs/realgame/log_gameview_runs.jsonl` carries the per-card `rect` and the
  `screen` block that only those lines emit, so the mod does emit the fields.
  What that run does not settle is whether the rects put the boxes in the right
  place on a live window. [`docs/POV.md`](docs/POV.md) §3 lists the five open
  items, and was written before that session.

---

## Layout

```
flybalatro/         the library: connectome loader, LIF brain, encodings,
                    hand analysis, plasticity, the viewer, the real-game driver
scripts/            one file per experiment stage; every one has a --help that
                    states the protocol it implements
tests/              467 tests, 466 passed and 1 skipped (a v1-era assertion
                    that a brain readout does NOT exist; one does, so it
                    cannot run)
outputs/            every artefact, with the config that produced it, plus five
                    long-form REPORT.md write-ups
figures/            regenerated from outputs/ by scripts/figures.py
patches/            our changes to two upstream projects, with upstream commits
docs/               REPRODUCE, DATA, REALGAME_INSTALL, POV
RESULTS.md          the running lab notebook, in the order things happened,
                    corrections marked in place
```

`RESULTS.md` is the primary record. Corrections to it are made **in place and
marked** rather than by deletion. The Plasticity v1 and v2 sections carry the
audit that overturned three of their own claims.

## Reproducing

**[`docs/REPRODUCE.md`](docs/REPRODUCE.md)** goes from clone to every headline
number: the connectome download, building the patched engine, the probes, the
behaviour-cloning pipeline, the plasticity protocol and the calyx control. It
carries measured runtimes for the machine it was produced on (M2 Pro, 16 GB) and
a note on which steps need the real game.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[viewer,figures,dev]"
python -m pytest                      # 466 passed, 1 skipped, ~36 s
python -m scripts.figures             # redraws every figure from outputs/
python -m flybalatro.viewer.server --port 8767 --policy brain
```

The live dashboard is what most of the debugging happened in; it is also the
project's least dishonest demo, because it labels the relayed hand analysis
"computed outside the brain" on screen. `outputs/pov/dashboard_embedded.png` is
what it looks like with the real game embedded in it.

* [`docs/DATA.md`](docs/DATA.md): MaleCNS v1.0 provenance, the exact files, and
  what this project filters out of them before anything runs.
* [`docs/REALGAME_INSTALL.md`](docs/REALGAME_INSTALL.md): driving the Steam
  build on macOS. The only part that cannot be reproduced headlessly.
* [`patches/README.md`](patches/README.md): the two patched upstreams, pinned
  commits, and why one of them is a real engine bug.

## Licence

MIT, see [`LICENSE`](LICENSE). Chosen to match the vendored upstreams, which
are MIT themselves (`balatro-rs`, `balatrobot`), so the patches in
[`patches/`](patches/) carry the same terms as the code they apply to.

No connectome data and no Balatro asset is redistributed here. See
[`docs/DATA.md`](docs/DATA.md) for the terms that attach to what you download
yourself.

Absolute paths in the artifacts under `outputs/` were redacted before the
first public commit: the author's home directory is written as `<repo>` and
`<home>`. Only the path prefix changed; no measurement, count or result was
touched, and every JSON still parses.

## Citing

[`CITATION.cff`](CITATION.cff). If you cite the connectome, cite the *Cell*
paper at doi:10.1016/j.cell.2026.08.015 and take its canonical author list from
the publisher record.
