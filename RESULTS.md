# Fly brain plays Balatro — results (2026-09-12)

Setup: MaleCNS v1.0 connectome (brain-only, 146,271 neurons, 5.15M edges ≥5 synapses,
signs from neurotransmitter prediction), leaky integrate-and-fire (Shiu et al. 2024 constants),
frozen. Balatro state → 283 binary bits → tonic drive on olfactory receptor neurons
(extended into other sensory neurons for the overflow). One 50 ms window from reset per decision.
Readout = linear or 1×256 MLP on log1p spike counts, trained by behavior cloning of a trivial
pair/flush heuristic (150k states). Headless Rust engine (balatro-rs, patched), ante 1 only.

## Gate probe: does input identity reach the descending neurons?
Linear probe on 600 random 180-bit inputs, chance ≈ 0.55.
raw bits 0.90 · ORN 0.90 · ALPN 0.70 · KC 0.59 · MBON 0.58 · CX 0.57 · DN 0.61 · DN (shuffled wiring) 0.66
DN "how many bits on" control: 0.75. → DNs encode input intensity, not identity.
Visual input: photoreceptors are histaminergic (inhibitory) → zero downstream spikes.
(doomfly works around this with a tonic 12 mV lamina bias, see its engine.py.)
Excitatory optic-lobe drive: DN identity 0.60/0.54, intensity 0.67. Same regime.
Kenyon cells fire ~50% vs ~5% in vivo: the untuned model's mushroom body is saturated.

## Behavior cloning + live play (400 episodes each, same seeds)
| condition                     | imitation top-1 | clear ante 1 | mean chips |
|-------------------------------|-----------------|--------------|------------|
| random legal                  |        –        |   0.0%       |  100       |
| heuristic (expert)            |        –        |  16.8%       |  931       |
| raw bits, linear / MLP        |  0.598 / 0.671  |   0 / 0      |  148 / 311 |
| random projection (6000), lin/MLP | 0.585 / 0.633 | 0 / 0      |  155 / 248 |
| shuffled fly, ALPN+KC+DN      |  0.553 / 0.581  |   0 / 0      |  146 / 158 |
| shuffled fly, DN only         |  0.527 / 0.556  |   0 / 0      |  137 / 152 |
| real fly, ALPN+KC+DN          |  0.478 / 0.503  |   0 / 0      |  131 / 126 |
| real fly, DN only             |  0.392 / 0.403  |   0 / 0      |  145 / 140 |
Most-frequent-legal-action baseline for imitation: 0.356.

## Reading

> **Correction (v3, 2026-09-13).** Two conclusions in this section were overturned
> later and are not live. (1) "Passing the state through the fly costs information
> at every stage", and the gate probe's reading that the fly garbles what it is
> given, were an artefact of **this encoding**, not a property of the fly. One
> relay bit per whole ORN type instead of 10 scattered receptors takes Kenyon-cell
> hand-type decoding to 1.000 and descending-neuron decoding to 0.842; see
> [Glomerular encoding](#glomerular-encoding-2026-09-13) and
> [v3](#v3-glomerular-encoding-2026-09-13). (2) "A degree-preserving shuffle is a
> better reservoir than the evolved wiring" does not survive either: the ordering
> reverses on the whole-brain linear readout in v3, and under the glomerular
> encoding the global shuffle is not a like-for-like control at all, as
> [that section's caveat](#did-real-wiring-stop-losing-to-shuffled) says. The
> opposite claim, that real wiring beats shuffled, is **also retracted**; the
> control that replaces it is the [calyx rewiring control](#calyx-rewiring-control),
> and it finds no effect either way.

The ordering raw bits > random projection > shuffled connectome > real connectome is established
by the probe and by imitation accuracy (n≈28.7k held-out states; the shuffled-vs-real gap is
0.553 vs 0.478 linear). In live play every brain condition lands between random (100 chips) and
the no-brain MLP (311), none clears ante 1, and the 10–30 chip differences among brain conditions
are not resolved at 400 episodes (no per-episode variance stored). Action histograms are similar
across conditions (mostly select-card, ~9–11% play), so chips track how often a readout presses
play on whatever is selected more than hand quality. Lead with imitation accuracy, not chips.
Passing the state through the fly costs information at every stage, and a degree-preserving
shuffle of its synapses is a better reservoir than the evolved wiring (shuffling creates direct
sensory→DN shortcuts). The "fly plays X" genre is reservoir computing with a mascot.

Caveats: 283 features into 144 glomerular channels is an encoding bottleneck (ceiling probe 0.78);
untuned LIF with no inhibitory calibration; stateless brain per decision (persistent was worse in
the probe); one heuristic expert; ante 1 only.

Artifacts: outputs/brain_probe*.json, outputs/bc/{train_metrics,eval}.json,
outputs/bc/eval_1000ep_nobrain.json, viewer: `python -m flybalatro.viewer.server --port 8766`.

---

# v2: hand-type relay (2026-09-13)

The v1 readout never played a poker hand: it learned "select three slots, then
play". v2 stops asking the fly to work out what a flush is and asks whether it
can **relay** one. `flybalatro/hands.py` enumerates all 218 subsets of size 1-5
of the dealt cards, classifies and scores each one exactly as `balatro-rs` does,
and `flybalatro/features_v2.py` turns the answer into 32 extra binary channels
placed **in front** of the unchanged 283 v1 bits (315 total). Those 32 bits are
injected as tonic current on olfactory receptor neurons like every other bit.
Everything else is the same: MaleCNS, frozen LIF, one 50 ms window from reset,
readout on log1p spike counts, ante 1, `mask_noop_actions=True`, 150k states,
same seeds.

Relay block: best hand type one-hot (9) · best-subset slot mask (8) · type the
current selection forms, 9 + "none" (10) · "selection **is** the best subset" (1)
· best score vs still-needed buckets (4). Silent outside a blind.

Two method changes come with it. `FeatureMap(orn_first=True)` lays real ORNs down
before the overflow sensory neurons, so the 32 relay bits sit on 320 genuine
receptor neurons (v1 pooled ORNs and overflow *then* permuted, so its channels
were ~93% ORN spread uniformly, and the v1 bits behind the relay block are now
driven through a slightly different map). And `hands.py` scores only the cards
that *make* the hand, not every played card, because `calc_score_inner` iterates
`MadeHand.hand`: a pair plus three kickers is worth exactly the bare pair.

## Teacher: hand-aware, and much stronger

`scripts/baseline_heuristic_v2.py` selects the best subset one `select_card[i]`
at a time and plays it, unless `best_score * plays_left < still_needed` and a
discard remains, in which case it discards the worst non-best cards. The engine
has no deselect, so it also needs a repair rule for selections it would never
make (half the collected episodes are random): a partial selection inside the
best subset is finished and played, an all-junk selection is discarded away when
the best hand containing it is below its fair share, and anything else is played
as the best subset containing what is already selected. It never skips a blind.

| teacher | seeds | clear ante 1 | mean chips |
|---------|-------|--------------|------------|
| v1 pair/flush heuristic | 0-999   | 17.7% | 942  |
| v2 hand-aware           | 0-999   | **70.1%** | **1111** |
| v1 pair/flush heuristic | eval 400 | 16.8% | 931 |
| v2 hand-aware           | eval 400 | **68.5%** | **1091** |

## Behaviour cloning of the v2 teacher (400 episodes, same eval seeds)

| condition                     | imitation top-1 | clear ante 1 | mean chips |
|-------------------------------|-----------------|--------------|------------|
| random legal                  |        –        |   0.0%       |  100       |
| v2 teacher (expert)           |        –        |  68.5%       | 1091       |
| raw v2 bits, linear / MLP     |  0.673 / 0.881  |  0.5% / 51.0% |  297 / 984 |
| random projection (3696) matched | 0.734 / 0.900 | 5.8% / 54.0% |  493 / 1001 |
| shuffled fly, ALPN+KC+DN      |  0.604 / 0.684  |   0 / 2.0%   |  238 / 404 |
| shuffled fly, DN only         |  0.578 / 0.651  |   0 / 0.7%   |  226 / 317 |
| real fly, ALPN+KC+DN          |  0.555 / 0.611  |   0 / 0      |  205 / 235 |
| real fly, ALPN only           |  0.542 / 0.604  |   0 / 0.5%   |  213 / 305 |
| real fly, DN only             |  0.422 / 0.439  |   0 / 0      |  139 / 131 |
Floors on the same held-out split (n = 29,282 states): most-frequent-legal-action
0.354, uniform-random-legal 0.205.

## Does the fly play real hand types?

> **Correction (v3, 2026-09-13).** "The relay arrives degraded, not intact" and
> "identity survives one synapse into the antennal lobe and is gone by the
> descending neurons" describe the v1/v2 **encoding**, not the fly. Under one
> relay bit per whole ORN glomerulus the same readout finds the best available
> subset on 81% of plays against a no-brain control's 84%, and the descending
> neurons decode hand type at 0.842. "Shuffled wiring beats real at every depth"
> is superseded too: it reverses on the whole-brain linear readout in v3, and
> under that encoding the global shuffle stops being a like-for-like control.
> See [v3](#v3-glomerular-encoding-2026-09-13), its
> [shuffle caveat](#did-real-wiring-stop-losing-to-shuffled), and the
> [calyx rewiring control](#calyx-rewiring-control) for the control that replaces
> the global shuffle.

Every row's plays, classified by the relay bits the brain was driven with
(`hand_evidence` in `outputs/bc2/eval.json`; "best" = fraction of plays where the
selection was the best available subset):

| row | plays | best | High Card | Pair | Two Pair | Trips | Straight | Flush | Full House |
|-----|-------|------|-----------|------|----------|-------|----------|-------|------------|
| random                   | 1613 | 0.009 | 75% | 22% |  2% | 1% |  0% | 0% |  0% |
| v2 teacher               | 3393 | 1.000 |  2% | 13% | 35% | 7% | 11% | 9% | 20% |
| raw v2 bits, MLP         | 3260 | 0.927 |  5% | 18% | 31% | 8% | 11% | 9% | 18% |
| random projection, MLP   | 3312 | 0.954 |  4% | 14% | 35% | 8% | 11% | 9% | 18% |
| shuffled ALPN+KC+DN, MLP | 2478 | 0.274 | 24% | 37% | 19% | 8% |  5% | 4% |  3% |
| shuffled DN, MLP         | 2193 | 0.180 | 29% | 40% | 16% | 7% |  3% | 3% |  2% |
| real ALPN, MLP           | 2141 | 0.142 | 24% | 47% | 15% | 6% |  3% | 2% |  1% |
| real ALPN+KC+DN, MLP     | 1937 | 0.112 | 32% | 47% | 13% | 4% |  2% | 2% |  1% |
| real DN, MLP             | 1641 | 0.010 | 60% | 35% |  2% | 2% |  0% | 0% |  0% |

Partly. Every real-wiring readout above the descending neurons plays real hand
types more often than random (Pair 47% vs 22%, Two Pair 13-15% vs 2%), and it
reaches trips, straights and flushes at all. But it finds the best available
subset on 11-14% of plays and never clears ante 1. The same readout trained
on the same bits *without* the brain finds it on 93-95% and clears half the time.
So the relay arrives degraded, not intact: enough to push the fly one rung up the
hand ladder, not enough to play the hand it was told about.

The controlled comparison is inside the brain. DN-only is the same architecture on
the same episodes and lands on random's histogram (High Card 60%, best 1.0%) with
*fewer* cards selected per play than random (4.9 vs 5.4), so the ALPN result is
not "it selected more cards": identity survives one synapse into the antennal
lobe and is gone by the descending neurons, which is what the v1 linear probe
(ALPN 0.70, DN 0.61, intensity control 0.75) predicted, now visible in behaviour.
The v1 ordering also survives: shuffled wiring beats real at every depth
(0.604 vs 0.555 imitation, 27% vs 11% best-subset plays), because a
degree-preserving shuffle creates direct sensory->DN shortcuts.

## What is computed outside the brain, and what the brain does

Outside the brain, in ordinary Python that is not biological in any sense:
subset enumeration, hand classification, level-1 scoring, the best-subset mask,
the type the current selection forms, the "selection is the best subset" bit and
the score-vs-needed bucket (`flybalatro/hands.py`, `flybalatro/features_v2.py`).
The teacher that generates the labels is also ordinary Python and uses the same
analysis directly.

The brain receives those 32 bits as receptor drive alongside the 283 state bits,
runs one 50 ms window, and produces spike counts. It computes no poker. The only
trained part is the readout (linear or 1x256 MLP) on those counts. Anything the
readout gets right about hand types is information that survived the trip; the
tables above measure how much did.

Caveats. The linear rows understate the encoding: "select the first unselected
best slot" is a conjunction a linear readout cannot express, which is why raw v2
bits are 0.673 linear vs 0.881 MLP. The v1->v2 jump in brain imitation
(0.473 -> 0.555 for real ALPN+KC+DN) is **not** a controlled comparison: the
teacher, the encoding and the feature map all changed. 400 episodes does not
resolve 0.0% vs 0.5% clear rates, though 1% vs 11% best-subset plays over
~1,600-3,400 plays per row is resolved. The teacher's fair-share threshold
(`--play-share`, default 1.0) was picked on a 300-episode sweep. Ante 1 only, no
jokers, no enhancements.

Artifacts: `outputs/bc2/{baseline_heuristic_v2,collect_meta,train_metrics,eval}.json`,
`outputs/bc2/viewer_v2.png`, the viewer running the v2 real-wiring readout, and
a picture of the failure mode rather than a success demo: the relay panel reads
"best hand Pair · 44, best slots 1 3" while the readout's top action is
`select_card[0]`, which is the 11% number above in one frame.
`outputs/bc/*` untouched. Viewer:
`python -m flybalatro.viewer.server --port 8767 --policy brain`. It prefers
`outputs/bc2` models, switches to `encode_v2` when a model says
`feature_version 2`, and shows the relayed analysis in its own panel labelled
"computed outside the brain". Pipeline:
`BC=outputs/bc2 TEACHER=v2 WORKERS=4 scripts/bc_pipeline.sh`.

---

# Glomerular encoding (2026-09-13)

`outputs/mb/REPORT.md` found the mushroom body sparse but not separable and
located the cause upstream: the v1/v2 encoding gives each bit 10 randomly
scattered ORNs, so ~36 active bits drove ~360 receptors into **all 54 glomeruli
on every input** and the total ALPN spike count varied by a CV of 0.031. This
tests the fix. One relay bit = one whole ORN type. No plasticity, no new knobs.

Encoding (`flybalatro.encode.GlomerularMap`): the 32 bits of the `features_v2`
relay block map to the 32 largest `ORN_*` glomeruli, one whole type each; the 283
v1 state bits are dropped from the fly entirely. The nine hand-type bits get the
nine smallest, tightest types (34-36 cells) so which hand type is active cannot be
read off the total drive; the remaining 23 are dealt largest-first in the block
order chosen by an explicit search over all 24 orders to minimise what driven-ORN
count alone reveals about the labels. 2,000 real states from `outputs/bc2`,
stratified by hand type; **6.83 of 32 glomeruli active per state** (4-9), 441
driven ORNs of 2,639.

| readout | old encoding (`outputs/mb/REPORT.md`) | glomerular, `apl2` tonic |
|---|---|---|
| glomerular ceiling, no simulation | 0.775 | **1.000** |
| ALPN, identity | 0.695 | **1.000** |
| KC, identity | 0.598 (permuted 0.573) | **1.000** (permuted 0.108) |
| KC, binarised active set | - | **1.000** |
| KC, intensity control | 0.830 | 0.959 |
| MBON, identity | 0.583 | 0.511 |
| DN, identity | 0.610 | 0.842 |

Labels here are 9-way hand type and `sel_is_best` on real states, not the old
synthetic bit-group labels, so the columns are not a controlled comparison: the
task got easier as well as the fly getting better. The controlled part: on *the
same* 32-bit relay, an input leaving 25 of 32 glomeruli silent is decodable
through the mushroom body and an input lighting all of them is not. Grouped CV
with relay pattern as the group (1,169 distinct patterns in 2,000 states) gives
KC 0.995, so it is not memorisation, and a driven-ORN-count-only probe gives
0.267, so it is not the total-drive confound.

**Gate: 0 of 12 settings pass** (`apl_scale` {1,2,5} x tonic/Poisson x
`kc_vth_offset_mv` {0,4}). Six fail *only* the between/within-type KC Jaccard
criterion (< 0.5; best 0.588 at `apl5` tonic). The mushroom body does blur the
active set (driven ORNs go in at 0.504, KCs come out at 0.879 at `apl2`), but the
binarised KC set still decodes hand type at 1.000 and the antennal lobe sits at
0.95 on the same metric while decoding perfectly, so mean set overlap is not what
decides separability here. `kc_input_norm = 1` fails again,
and worse (denser, always-on 0.041 -> 0.061). Poisson drive does not
restore MN9 sugar specificity (sugar 1.2-3.2 spikes vs 4.0-6.0 for 165 matched
random ORNs) and is worse than tonic at every stage past the antennal lobe.

Chosen, in `outputs/mb2/tuned_config.json`: `apl_scale = 2`, tonic 30 mV, no
threshold offset, 196 active KCs per state of which 146 are among the 838 whose
activation is input-dependent. That is a substrate a KC -> MBON plasticity rule
could use; the previous round had 837 cells firing for every input and identity at
chance. Full write-up: `outputs/mb2/REPORT.md`.

---

# v3: glomerular encoding (2026-09-13)

v2 asked whether the fly could **relay** a hand type and the answer was "partly":
the readout on real wiring found the best available subset on 11% of plays and
never cleared ante 1, where the same readout without the brain found it on 93%.
`outputs/mb2/REPORT.md` located the loss in the input, not the network, and fixed
it: one relay bit = one whole ORN glomerulus. This is that encoding run
through the behaviour-cloning pipeline, so the reservoir comparison
(real vs shuffled vs no brain) happens under an input the mushroom body
can do something with.

**Nothing was recollected.** `outputs/bc3/states.npz` is a symlink to the v2
file: the same 150,037 states, the same v2 teacher labels, the same
`rng(seed + 1)` episode split, so the held-out set is the same 29,282 states with
the same floors (most-frequent-legal 0.354, uniform-random-legal 0.205) and the
same eval seeds 100000-100399. The imitation columns below are directly
comparable to the v2 table.

## What changed

| | v2 | v3 |
|---|---|---|
| bits reaching the fly | all 315 | the **32** relay bits; 283 dropped |
| receptors per bit | 10, scattered across glomeruli | every ORN of one glomerulus (34-204) |
| glomeruli active per state | all 54 | **6.6** of 32 inside a blind (SD 1.4, 4-9); 6.1 over all states, 0 outside a blind |
| brain | untuned | `apl_scale = 2` (`outputs/mb2/tuned_config.json`) |
| populations recorded | ALPN, KC, DN | ALPN, KC, **MBON**, DN |
| unique inputs to simulate | 138,354 | **4,962** (96.7% duplicates) |

The assignment is `scripts/mb2_assign.py`'s, unchanged and re-checked against an
embedded constant at load time (`flybalatro/glomerular.py`): the nine
`best_<hand type>` bits get the nine smallest, tightest glomeruli so the 9-way
label cannot be read off total drive. MBON is recorded for the first time
because, per `outputs/mb2/REPORT.md`, there is finally a separable Kenyon-cell
code for it to read.

Deduplicating on 32 bits instead of 315 is what makes the stage cheap: 4,962
windows per wiring, 2.2 minutes on 4 workers against 63 minutes in v2. It also
means the 11,700 states (7.8%) outside a blind, where the relay block is silent,
collapse to **one** input, an unlit antennal lobe and zero spikes, so every
brain readout emits the same logits there. `truncated_fraction` is 0.000 on every
row, so that did not put anything in a loop.

## Imitation and live play (400 episodes, seeds 100000-100399, ante 1)

**Read the linear column.** The 32 relay bits fed straight to an MLP score 0.755,
and a random projection of the same 32 bits 0.757: that is the best achieved on
what the fly is told. Every brain MLP row except MBON-only lands within 0.017 of
it (0.740-0.750), so the MLP column is saturated and cannot separate wirings. The wiring argument is
in the linear column and in the hand-type table.

| condition | imitation top-1 lin / MLP | clear ante 1 lin / MLP | mean chips lin / MLP |
|---|---|---|---|
| random legal | - | 0.0% | 100 |
| v2 teacher (expert) | - | 68.5% | 1091 |
| raw 32 relay bits (no brain) | 0.549 / **0.755** | 0.5% / 42.2% | 413 / 956 |
| random projection (2254) matched | 0.754 / 0.757 | 43.8% / 41.8% | 942 / 961 |
| real fly, ALPN+KC+DN | **0.746** / 0.748 | **37.8%** / 39.5% | 922 / 937 |
| real fly, KC only | 0.709 / 0.750 | 26.5% / 39.2% | 843 / 929 |
| real fly, MBON only | 0.434 / 0.589 | 0.0% / 0.0% | 178 / 262 |
| real fly, DN only | 0.661 / 0.740 | 3.2% / 32.5% | 514 / 891 |
| shuffled fly, ALPN+KC+DN | 0.722 / 0.750 | 25.5% / 40.5% | 853 / 938 |
| shuffled fly, DN only | 0.695 / 0.749 | 20.0% / 41.0% | 762 / 944 |

The no-brain controls are **weaker** than they were in v2 (0.755 / 42% against
0.881 / 51%), because they too now see only 32 bits: the 283 dropped bits carried
things the teacher uses, such as how many plays are left for its discard rule. So
the fly did not get easier work, it got to the ceiling of the work it is given.
In v2 the best brain readout sat 0.27 below its no-brain control; in v3 it sits
0.007 below.

## Does the fly play real hand types?

Same `hand_evidence` construction as v2: every play classified by the relay bits
the brain was driven with, "best" = fraction of plays where the selection was the
best available subset.

| row | plays | best | High Card | Pair | Two Pair | Trips | Straight | Flush | Full House |
|---|---|---|---|---|---|---|---|---|---|
| random | 1613 | 0.009 | 75% | 22% | 2% | 1% | 0% | 0% | 0% |
| v2 teacher | 3393 | 1.000 | 2% | 13% | 35% | 7% | 11% | 9% | 20% |
| raw 32 bits, MLP | 3321 | 0.841 | 10% | 11% | 34% | 8% | 13% | 10% | 14% |
| random projection, MLP | 3346 | 0.834 | 10% | 10% | 34% | 8% | 12% | 9% | 16% |
| real ALPN+KC+DN, linear | 3372 | **0.810** | 12% | 11% | 34% | 7% | 12% | 9% | 14% |
| real ALPN+KC+DN, MLP | 3367 | 0.807 | 12% | 11% | 33% | 8% | 12% | 9% | 15% |
| real KC, MLP | 3344 | 0.803 | 11% | 11% | 34% | 7% | 12% | 9% | 14% |
| real KC, linear | 3376 | 0.678 | 14% | 16% | 31% | 9% | 10% | 8% | 11% |
| real DN, MLP | 3363 | 0.769 | 13% | 13% | 34% | 7% | 10% | 8% | 14% |
| real DN, linear | 2798 | 0.393 | 20% | 28% | 29% | 8% | 6% | 3% | 5% |
| real MBON, MLP | 1966 | 0.153 | 33% | 39% | 19% | 4% | 2% | 2% | 1% |
| real MBON, linear | 1712 | 0.014 | 42% | 46% | 9% | 2% | 1% | 0% | 0% |
| shuffled ALPN+KC+DN, MLP | 3325 | 0.814 | 9% | 12% | 34% | 8% | 12% | 9% | 15% |
| shuffled ALPN+KC+DN, linear | 3308 | 0.710 | 10% | 20% | 32% | 8% | 10% | 8% | 11% |
| shuffled DN, MLP | 3363 | 0.813 | 10% | 12% | 33% | 8% | 12% | 9% | 15% |
| shuffled DN, linear | 3189 | 0.623 | 11% | 25% | 29% | 7% | 10% | 8% | 9% |

The v2 sentence was "the relay arrives degraded, not intact". It now arrives
close to intact: real ALPN+KC+DN finds the best subset on **81%** of plays
against 11% in v2, and the no-brain control it is being compared against is 84%.
The remaining gap between fly and no-brain is 3 points, not 82. Its hand-type
histogram is within a few points of the teacher's from Pair through Flush; it
over-plays High Card (12% against 2%) and under-plays Full House (14% against
20%), which are the same two ranks the no-brain control misses in the same
direction (10% and 14%), so that residual is the 32-bit input rather than the
fly.

## Did real wiring stop losing to shuffled?

**Read this section's caveat before its numbers.** Under the glomerular
encoding the degree-preserving shuffle is no longer a like-for-like control: it
destroys the convergence the encoding depends on and leaves a network that
barely fires (table below). So the comparison below is between a working brain
and a near-silent one, and "real beats shuffled" is **not a defensible claim
about evolved wiring.** It is retracted.

**That control has now been run, and it answers the other way.** See
[Calyx rewiring control](#calyx-rewiring-control) below: a degree-, weight- and
type-preserving rewiring of the ALPN -> KC stage lands in the same activity
regime without any rescue, and real wiring then does **not** beat it on anything
-- KC decoding, KC code structure, imitation or live clear rate. Treat every
number in this section as describing a broken control, and the calyx section as
the real answer for that one stage.

On the whole-brain readout, **yes, and the v1/v2 ordering reversed.** Linear
imitation 0.746 real against 0.722 shuffled, live 37.8% against 25.5% clear,
best-subset plays 0.810 against 0.710. At the MLP they are tied because both are
at the input's ceiling.

On DN only, **no.** Shuffled still wins: 0.695 against 0.661 linear, 20.0%
against 3.2% clear, and at the MLP 41.0% against 32.5% live, about 2.5 standard
errors at 400 episodes, so ahead but not decisive.

The control also changed character, and that has to be said before the numbers
are read as a like-for-like swap. Glomerular convergence is the whole point of
the encoding: ~440 driven receptor neurons of one type each converge on the
uniglomerular projection neurons of that glomerulus, which is how ~441 driven
ORNs put ~480 of the 686 ALPNs above threshold. A degree-preserving target
permutation destroys exactly that convergence, so the shuffled brain barely
fires:

| population | real, mean rate | shuffled, mean rate | real, active per state | shuffled |
|---|---|---|---|---|
| ALPN (686) | 71.3 Hz | 0.63 Hz | 0.698 | 0.021 |
| KC (4064) | 1.44 Hz | 0.15 Hz | 0.049 | 0.006 |
| MBON (97) | 8.25 Hz | 2.84 Hz | 0.211 | 0.100 |
| DN (1314) | 6.03 Hz | 1.08 Hz | 0.182 | 0.040 |

Non-constant channels available to the readout: 2,254 real, 899 shuffled. The
shuffled network is now close to a sparse direct receptor-to-descending-neuron
projection with almost nothing in between. That is the same "shuffling creates
shortcuts" story as v1 and v2, except that under the old encoding the shortcuts
were an *addition* to a working network and here they are nearly all that is
left. That is why shuffled still wins at DN: a shortcut is the best thing a DN
can read, and the real fly's DNs are several synapses downstream of the input.

## Did the DN-only readout become a real player?

Yes, with the MLP. Descending neurons are the fly's actual motor output and in v2
they were the clearest failure in the project: 0.439 imitation, 0% clear, 131
chips, best-subset on 1.0% of plays, a histogram indistinguishable from random.
Under the glomerular encoding the same population, same architecture, same
episodes:

| | v2 | v3 |
|---|---|---|
| imitation top-1, linear / MLP | 0.422 / 0.439 | **0.661 / 0.740** |
| clear ante 1, MLP | 0 | **32.5%** |
| mean chips, MLP | 131 | **891** |
| best-subset plays, MLP | 0.010 | **0.769** |
| High Card share of plays, MLP | 60% | 13% |

The linear DN readout is transformed too (0.393 best-subset against 0.010) but
clears only 3.2%, so "real player" is an MLP statement.

Two smaller results. **KC only** reaches 0.750 / 39.2% / 0.803 best-subset, i.e.
the full ALPN+KC+DN set adds nothing on top of the Kenyon cells alone: the
mb2 probe that read hand type from KCs at 1.000 showing up in behaviour rather
than in a classifier. **MBON only** is the weak link: 0.589 / 0% / 0.153, with 49
of 97 cells non-constant. Note that `apl_scale = 2` scales every APL output
synapse, the ~2% that lands on MBONs included; whether restricting it to APL->KC
helps the MBON readout is not tested here.

## Caveats

Not a controlled comparison with v2: the encoding, the number of input bits
(315 -> 32), the brain's calibration (`apl_scale` 1 -> 2) and the recorded
populations all changed at once. What *is* controlled is everything the labels
depend on (same states, same teacher, byte-identical episode split, same eval
seeds) and the real-vs-shuffled-vs-no-brain comparison inside this run, where
only the wiring differs.

The MLP rows are at the ceiling of a 32-bit input and cannot resolve wiring; 400
episodes does not separate 39.5% from 40.5%. The shuffled control is no longer
rate-matched to the real brain (table above). The teacher, the `--play-share`
threshold, ante 1 only, no jokers, no enhancements: all unchanged from v2. And
the poker is still computed outside the brain by `flybalatro/hands.py`; the fly
relays it, and this section measures how much of it survives the trip, which is
now most of it.

## Artifacts

`outputs/bc3/{feats_real,feats_shuffled}.npz` (log1p spike counts, float16, ALPN
+ KC + MBON + DN), `outputs/bc3/{train_metrics,eval}.json`,
`outputs/bc3/models/*.npz` (now carrying `encoding: "glomerular32"` alongside
`feature_version`), `outputs/bc3/viewer_v3.png`. `outputs/bc/`, `outputs/bc2/`,
`outputs/mb/` and `outputs/mb2/` are untouched; `outputs/bc3/states.npz` is a
symlink into `bc2`.

Pipeline: `BC=outputs/bc3 TEACHER=v2 WORKERS=4 ENCODING=glomerular32
CACHE_CAP=6000 scripts/bc_pipeline.sh`. `scripts/bc_consistency_check.py`
reproduces the stored rows from a live tuned brain exactly, for both wirings; it
was re-run after the fact because another session added an `apl_kc_only` knob to
`flybalatro/tuning.py` mid-eval (default 0.0, a no-op on the `apl_scale` path,
which is why `eval.json._config` shows a seven-key tuning dict).

Viewer: `python -m flybalatro.viewer.server --port 8767 --policy brain`. It
prefers `outputs/bc3` models and, within a run directory, orders candidates by
held-out imitation top-1 from that run's `train_metrics.json`, so the screen
shows the best-measured real-wiring readout rather than a hand-maintained
favourite: here `real_kc_mlp` (0.7495) over `real_alpn_kc_dn_mlp` (0.7484), a
tie in both imitation and live play. The sensory panel draws the 32 relay bits as
named glomeruli with their ORN counts, lights the ~6 that are driven, and marks
the other 283 bits as not sent to the brain.
`outputs/bc3/viewer_v3.png` is that screen mid-Boss-Blind at 429/600 with five
glomeruli lit, ALPN at 70 Hz and the Kenyon cells at 1.2 Hz.

---

# Plasticity: does the fly's own learning rule learn Balatro from chips? (2026-09-13)

Every earlier round put a *trained readout* on the fly and the learning happened
in scikit-learn. This round removes the **trained action readout** and the
teacher. The decision comes out of the mushroom-body output neurons by a fixed
rule with no fitted action weights, and the only mutable thing in the fly is the
weight of the 33,496 Kenyon-cell -> MBON synapses, changed by dopamine-gated
depression driven by chips.

> **Correction (v3, 2026-09-13).** Three claims in this section and the next were
> audited and are corrected in place below; the full accounting is in
> `outputs/plast3/`. (1) **Evaluation was not greedy.** `plast_run.py:174/205`
> evaluates with the *training* decider, which samples `rng.random() < p`, and
> these runs carry `explore_floor = 0.1`, so every "after" number here is a
> floored-softmax sample. Both modes are now reported (`outputs/plast3/eval_mode.json`,
> and the v2 table below). (2) **"No readout / nothing fitted" is too strong.**
> There *is* a fixed readout (`approach - avoid` over named MBON pools), and
> the fly carries a calibrated bias, a calibrated softmax temperature, 4,064
> per-Kenyon-cell thresholds (v2) and a calibrated per-pulse gain. What is absent
> is a *trained action* readout: nothing is fitted to actions, labels or
> outcomes. Dopamine here is an imposed update signal computed by the harness
> from the game's chips, not a signal the fly generates. (3) **"The bottleneck
> moved" was an inference, not a measurement**, in both sections; see the notes
> at each claim.

**The fly's decision is binary**: play the best subset now, or discard the junk
and dig (legal while discards remain). `flybalatro/hands.py` and
`features_v2.py` still compute the poker outside the brain and `GlomerularMap`
still turns the 32 relay bits into tonic drive on 32 whole ORN glomeruli; the
harness presses the keys. What the fly can learn is a **valence**: which
hand-odours mean approach and which mean avoid.

```
play_drive = mean rate(approach MBONs) - mean rate(avoid MBONs) + bias      [Hz]
P(play)    = 0.05 + 0.9 * sigmoid(play_drive / T)
reward (PAM)   -> depress KC -> (avoid    x PAM ) by (1 - eta * elig), floor 0.05x
punish (PPL1)  -> depress KC -> (approach x PPL1) by (1 - eta * elig), floor 0.05x
```

`bias` and `T` are two scalars calibrated once on 200 hands from seeds 300000+
before any dopamine (median state at P(play) = 0.55, exploration 20%) and frozen.
Reward = the play cleared the blind or gained >= `needed / plays_left`;
punishment = the play lost the blind; discards get no dopamine ever.

**Valence** is Aso et al. 2014 (eLife 04580) as a neurotransmitter rule
(glutamatergic MBONs drive avoidance, GABAergic and cholinergic ones drive
approach), applied to MaleCNS's own `consensus_nt`: 24 avoid neurons, 66
approach, 7 unassigned. **This disagrees with the compartment lists in the task
brief on 5 of the 9 types they name** (the brief's list looks like the
memory-expression role, not the activation valence); the brief's assignment was
run as its own condition and does no better. **Compartment** is measured, not
named: a KC -> MBON synapse is PAM-gated if PAM supplies >= half of that MBON's
PAM+PPL1 input weight. On the real graph that lands exactly where the anatomy
says: MBON01-07 (all glutamatergic) are 90-100% PAM, MBON11-20 100% PPL1.

## Two knobs added, both about APL, both forced

At `outputs/mb2/tuned_config.json` exactly as chosen, **every MBON the reward
pulse can reach is silent for every input**: MBON01-07 emit 0.00 spikes and sit
25-340 mV *below* rest. One neuron does it. APL delivers **-3,198 mV** to MBON01
in a 50 ms window against 259 mV of Kenyon-cell excitation (MBON05: -3,736
against 169). APL is non-spiking in vivo; this kernel has no graded transmission,
fires it at ~250 Hz and applies its full synapse-count weight. So
`apl_kc_only = 1` (`apl_scale` hits only APL -> KC, its documented role and 94% of
its output) and `apl_mbon_scale = 0.1` (the other 72 synapses). Both default to
no-ops, every earlier number is bit-identical, and the Kenyon-cell code barely
moves: 211.5 vs 183.5 active per state, cells active on >50% of odours 188 vs 166
(78 vs 66 active on every odour), hand-type decoding
0.960 vs 0.970 (grouped 0.94 both). MBONs go 8.6 -> 14.3 Hz, 33 -> 41 varying.
The mb2 tuning was run anyway as a condition.

## Sanity checks

| check | result |
|---|---|
| (a) decision depends on the odour | **pass**: raw drive spans -5.45..+7.80 Hz, 11.7 quantisation quanta, 85 distinct values over 137 odours |
| (b) conditioning, direction | **pass**: 20 punishment pulses lower the paired odour's drive by 2.33 Hz (7/8 odours), 20 reward pulses raise it by 9.20 Hz (8/8) |
| (b) conditioning, specificity | **fail**: unpaired odours move 85-105% as far; 70-75% of the weight change lands on the 188 Kenyon cells active on >50% of odours |
| (c) weight bounds | **pass**: never negative, never above original, min ratio exactly the 0.05 floor |

The naive fly is not neutral: its P(play) already runs 0.46 (`lt0.25`) to 0.69
(`ge1.0`), so "before" is the frozen fly on identical games, not 0.5.

## Eval, 60 ante-1 games on seeds 100000-100059, identical for every row

| row | clear before | clear after | chips after | P(play\|legal) after |
|---|---|---|---|---|
| real wiring, eta 0.02 / 0.05 / 0.1 (3 seeds each) | 0.350 | **0.039 / 0.033 / 0.033** | 544 / 543 / 549 | 0.948 / 0.948 / 0.947 |
| frozen fly, no dopamine | 0.350 | 0.350 | 824 | 0.527 |
| shuffled wiring | 0.417 | 0.417 (bit-identical) | 923 | 0.456 |
| mb2 tuning, reward arm silent | 0.383 | 0.250 | 829 | 0.380 |
| brief's valence lists | 0.417 | 0.333 | 870 | 0.487 |
| reward-omission variant | 0.267 | 0.033 | 570 | 0.939 |
| always play | - | 0.000 | 519 | 1.000 |
| always discard | - | 0.400 | 868 | 0.000 |
| **best odour-only policy** (play iff bucket >= `lt1.0`) | - | **0.633** | 1001 | 0.385 |
| v2 teacher dig rule (sees `plays_left`) | - | 0.700 | 1088 | 0.569 |

P(play) by score-vs-needed bucket, naive -> learned (real, eta 0.05), with what
the teacher does: `lt0.25` 0.46 -> 0.95 (teacher 0.00) · `lt0.5` 0.63 -> 0.92
(0.69) · `lt1.0` 0.42 -> 0.96 (0.98) · `ge1.0` 0.69 -> 0.98 (1.00). By hand type
every category ends between 0.93 and 1.00 (the two marginals are reported; the
P(play | hand type x bucket) joint was not produced, and with every cell of both
marginals flat it would add nothing). The bucket gradient goes 0.017 -> 0.033
against 0.710 for the teacher. 0.948 sits at the exploration ceiling: with
`explore_floor = 0.1` the maximum P(play) is 0.95, so behavioural saturation is
complete and the underlying drive is past it. That is also why the
reward-omission run is bit-identical to `real_eta0.1_s0`: both sit at p ~ 0.95
for every hand, so the same eval RNG draws the same 478 actions. The learning
curve is a step: P(play) hits its ceiling inside 150 hands and flattens while the
reward rate *falls* from 0.40 to 0.36. eta is irrelevant (all three saturate), and 692 reward pulses
against 230 punishments per run, at 4x the per-pulse gain, accounts for it.

## Reading

**The fly learned, and it learned the wrong thing.** P(play) 0.527 -> 0.948 in 9
of 9 runs with zero movement in the frozen control is a real learning effect from
a real biological rule with **no trained action readout** (a fixed MBON rule plus
a calibrated bias, temperature and pulse gain; see the correction above). But
the clear rate fell from 0.350 to
0.033 (worse than the naive fly, worse than always-discard, close to always-play,
which is the worst policy in the game) because the fly learned a single number
for every odour. Only ~15% of the reward effect and ~0% of the punishment effect
is odour-specific, and 15% of a +0.42 shift is the +0.02 of bucket gradient that
appeared. The mushroom-body code at this operating point is sparse (5.2%)
and perfectly decodable (KC hand type 0.96) but **not specific**: 188 of the 211
active cells are drawn from the same pool that responds to more than half of all
odours (78 of them to every odour), and compartment-level credit assignment needs
the opposite. `apl5`, the one mb2 setting with a specific code,
has 18 active KCs, too few to move an MBON at all.

The plasticity rule works and the code it has to work on does not support it.
The most likely bottleneck is one stage on, at the *specificity* of the
Kenyon-cell active set rather than at the encoding (fixed in the glomerular
round). *Corrected:* the original wording, "the bottleneck moved one stage", was
a diagnosis from a correlation (unspecific credit, unspecific behaviour), not
a measurement. v2 then intervened on exactly that and made the credit specific,
which is the evidence for this sentence; it did not make the fly win, so
"bottleneck" in the singular was wrong either way.

Caveats. Because the harness plays the best subset and `hands.py` scores as the
engine does, `chips_gained == best_score`, so `reward = 1` is exactly the v2
teacher's fair-share inequality: game-derived, but "the right thing" is defined
by the same rule the teacher uses. The shuffled control is uninformative: a global
permutation leaves 664 KC -> MBON edges instead of 33,496, so it deletes the
pathway rather than rewiring it (and pre == post exactly). 60 games resolves 0.03
vs 0.35 but not 0.03 vs 0.00. Ante 1, no jokers, one connectome, one window
length; `Tuning.apply` masks populations with `graph.post` rather than the brain's
permuted `post`, the pre-existing convention, which is one more reason the
shuffled brain is a different regime rather than the same fly rewired.

Artifacts: `outputs/plast/REPORT.md` (full write-up), `tuned_config.json`,
`probe.json` + `probe.log`, `calib.json`, `kc_check.json`, `baselines.json`,
`run_<name>.json` x18, `summary.json`, `run.log`. Code: `flybalatro/plasticity.py`,
`scripts/plast_{common,probe,run,summary,kccheck}.py`, `tests/test_plasticity.py`.
`outputs/bc*/`, `outputs/mb/` and `outputs/mb2/` untouched. Reproduce:
`python -m scripts.plast_probe --window-scan 50,150,300`, then
`python -m scripts.plast_kccheck`, then
`python -m scripts.plast_run --all --explore-floor 0.1`, then
`python -m scripts.plast_summary`.

---

# Plasticity v2: KC homeostasis (2026-09-13)

The previous section's verdict was "the plasticity rule works and the code it has to
work on does not support it": 188 of the ~211 Kenyon cells active per odour came from a
pool responding to more than half of all odours, those cells took 70-75% of every weight
change, and conditioning generalised completely (unpaired odours moved 85-105% as far as
the paired one). This round rebuilds the Kenyon-cell code.

**What the intervention did.** Giving each
Kenyon cell its own homeostatic threshold took the number of cells that *ever* respond
from **769 to 2,457** while mean activity **fell** from 218.7 to 177.2 cells per odour,
and the broadly-tuned pool collapsed 12x (192 -> 16 cells answering to more than half of
all odours). Three times the participating population on less total spiking is a
redistribution of the code, not an amplification of it, and it made
dopamine-gated credit odour-specific (15% -> 71% on the reward arm). Hand-type decoding
went *up*, 0.986 -> 0.996.

*Corrected (v3):* this round did **not** change "nothing else". It also (a) gave the
punishment pulse its own calibrated `eta` (a ~3.8x gain change on one arm) and (b)
doubled training from 600 to 1,200 hands. So the homeostasis effect is isolated by the
**conditioning** comparison (same odours, same pulses, one fly with per-KC thresholds
and one without) and **not** by the gameplay comparison against `outputs/plast`,
which differs in three ways at once.

**The mechanism is per-cell homeostatic thresholds**, which the animal has and a
synapse-count model cannot inherit: KC excitability is homeostatically regulated and a
single Kenyon cell answers to ~5-10% of odours (Turner et al. 2008; Honegger et al. 2011;
Lin et al. 2014). None of the existing knobs can produce that, because `apl_scale`,
`alpn_kc_scale`, `kc_vth_offset_mv` and `kc_kc_scale` are population-wide rescalings:
they change how many KCs cross threshold, never which, since between-KC input spread
(SD 112.6 mV, set by in-degree) is 11.8x the within-KC across-odour spread. New knob
`Tuning.kc_vth_offsets: float32[4064]`, indexed by `graph.kc_indices()`, defaults to a
no-op, calibrated by `offset_i += k * log((r_i + eps)/(target + eps))` with target 0.07
on 500 distinct real relay patterns from `outputs/bc2` (episodes >= 1000, decision-point
patterns only, held-out half never touched). **The update sees only each cell's own
firing rate**: no labels, no reward, no MBONs, no readout. 12 iterations, annealed
gain, deadband [0.04, 0.12], converged on the brief's criterion at iteration 11. Offsets
end at -6.74 .. +21.36 mV (none at the +30 ceiling; 2,249 at their own floor).

Two deviations, both in `outputs/plast2/REPORT.md` 1.4: the lower clamp is per cell,
`max(-10, v0_i - V_TH + 0.25)`, because this kernel's fixed `v0 = -52 + U(0, 6.9)` means
a threshold below a cell's own `v0` makes it fire on tick 1 for *every* odour (without
it the loop manufactures ~3,300 always-on cells); and the stop rule was corrected
mid-calibration to ship the offsets whose statistics are on the record rather than an
unmeasured post-convergence update.

## What it did to the code, and the gate (500 held-out odours)

| criterion | limit | before | after | |
|---|---|---|---|---|
| (i) frac KC with r > 0.5 | < 0.02 | 0.0472 (192 cells) | **0.0039 (16)** | PASS |
| (i) frac KC with r > 0.3 | < 0.05 | 0.0595 (242) | **0.0076 (31)** | PASS |
| (ii) KC active fraction | 3-10% | 0.0538 | **0.0436** | PASS |
| (iii) 9-way hand type from KC, grouped 5-fold | >= 0.85 | 0.986 | **0.996** (permuted 0.134) | PASS |
| (iv) approach - avoid drive range | > 1 quantum | 15.83 Hz | **16.21 Hz** (14.3 quanta) | PASS |
| (iv) MBON01-07 silent | 0 of 16 | 3 of 16 | **0 of 16** | PASS |
| (v) reward specificity share | >= 0.50 | 0.145 | **0.712** | PASS |
| (v) punish specificity share | >= 0.50 | -0.057 | **1.037** | PASS |

Cells that ever respond 769 -> **2,457**; median rate of a responding cell 0.088 ->
**0.066**; MBONs that vary 47 -> 68 of 97. The fallback mechanism the brief allowed
(scaling KC -> MBON output by residual response rate) is implemented and tested but
**was not needed**. With 500 distinct patterns, "grouped by relay pattern" CV is 500
groups of one, the strictest form of the test.

## Pulse calibration (model calibration, not reward shaping)

The two arms of our own rule differed ~4x per pulse because the neurotransmitter rule
gives 66 approach MBONs against 24 avoid ones, so one lost spike moves `play_drive` by
0.30 vs 0.83 Hz. Frequency is left exactly as the game gives it; only the per-pulse gain
is matched. Measured on a 23-odour panel at 5 pulses (an 8-odour first pass was
non-monotone in eta: quantisation, not signal): **eta_punish / eta_reward = 3.81**, so
eta_punish = 0.0763 and 0.1907 for eta_reward 0.02 and 0.05.

## The rerun: 60 ante-1 games on seeds 100000-100059, 1,200 training hands, 3 seeds

| row | clear before | clear after | chips after | P(play) after | bucket gradient |
|---|---|---|---|---|---|
| frozen fly, no dopamine | 0.328 | 0.328 | 789 | 0.522 | +0.010 |
| real, eta_r 0.02 / eta_p 0.0763 | 0.328 | 0.244 | 807 | 0.754 | **+0.364** |
| real, eta_r 0.05 / eta_p 0.1907 | 0.328 | 0.244 | 792 | 0.780 | **+0.404** |
| v1 (`outputs/plast`) | 0.350 | 0.033 | 543 | 0.948 | +0.033 |
| always play | - | 0.000 | 519 | 1.000 | n/a |
| always discard | - | 0.400 | 868 | 0.000 | n/a |
| best odour-only policy | - | 0.633 | 1001 | 0.385 | +1.000 |
| v2 teacher dig rule | - | 0.700 | 1088 | 0.569 | +1.000 |

**Corrected (v3): every "after" number in the table above is a floored-softmax
*sample*, not the greedy policy.** `plast_run.py:174/205` evaluates with the training
decider and these runs carry `explore_floor = 0.1`. Both modes, same weights, same 60
games (`outputs/plast3/eval_mode.json`; `outputs/plast3/weights_current_current_s0.npz`
is bit-identical to the published `weights_real_eta0.05_s0.npz`, and the sampled column
reproduces the published run to every decimal):

| row (same weights) | clear, **greedy** | clear, sampled | chips greedy/sampled | P(play\|legal) greedy | bucket gradient greedy |
|---|---|---|---|---|---|
| frozen fly, no dopamine | **0.267** | 0.328 (0.267 / 0.433 / 0.283 by eval RNG) | 688 / 803 | 0.474 | -0.009 |
| real, eta_r 0.02 (2 seeds) | **0.183** | 0.192 | 744 / 785 | 0.793 | +0.428 |
| real, eta_r 0.05 (3 seeds) | **0.178** | 0.244 | 742 / 792 | 0.807 | +0.459 |
| always discard | 0.400 | 0.400 | | 0.000 | n/a |

Greedy does **not** change the conclusion, and it does not rescue it: the learned fly is
still below its own frozen control and still far below always-discard. Paired on the
same games and bootstrapped with clusters on both the evaluation seed and the training
run: learned minus frozen is **-0.089 [-0.217, +0.033]** greedy (p = 0.17) and
**-0.083 [-0.222, +0.067]** sampled (p = 0.26), no resolvable movement either way;
learned minus always-discard is **-0.222 [-0.367, -0.083]** greedy (p = 0.003) and
-0.156 [-0.300, -0.011] sampled. The mode *does* change the level of both rows:
the naive fly's 0.328 was itself a sampled number (its three eval RNGs give 0.267,
0.433, 0.283 on the same 60 games), and greedy pulls the naive fly to 0.267 and the
learned fly to 0.178. The learned low bucket sits at P(play) = 0.51-0.57, so greedy
commits to *playing* most `lt0.25` odours where sampling threw about half of them
away. That is why greedy is the harsher and more honest number here.

"bucket gradient" is P(play | ge1.0) - P(play | lt0.25), the one ordered axis the odour
carries. P(play) by bucket, naive -> learned (eta 0.05): `lt0.25` 0.611 -> **0.545**,
`lt0.5` 0.462 -> 0.912, `lt1.0` 0.381 -> 0.934, `ge1.0` 0.621 -> 0.949. **The lowest
bucket moves down while the other three move up**. In v1 all four went to 0.92-0.98.
The joint P(play | hand type x bucket) table shows the same thing per cell: Pair/lt0.25
(389 decisions) 0.70 -> 0.52 where the teacher plays 0.00, Two Pair/lt0.5 (250) 0.40 ->
0.93 where the teacher plays 0.76.

Variance of P(play) across the 22-26 hand-type x bucket cells with n >= 5: naive
**0.0549** -> learned **0.0242** (teacher 0.1634, best odour-only 0.2449, always-play
0.0). It **fell**, and that is a ceiling artefact, not a loss of specificity: the cell
mean moved 0.511 -> 0.887, and variance measures spread, not direction (always-play
scores 0.0). The directional version is the cell-by-cell correlation with the teacher's
P(play), decision-weighted: **-0.065 -> +0.881**, weighted mean absolute error
0.531 -> 0.300.

Learning curves: the rolling reward rate **rises** 0.313 -> 0.399 (eta 0.02) and
0.381 -> 0.418 (eta 0.05) over 1,200 hands, while the frozen control drifts 0.329 ->
0.290 on the same games. In v1 it fell 0.40 -> 0.36 while P(play) hit the exploration
ceiling inside 150 hands; here P(play) climbs gradually to 0.75-0.78 against a 0.95
ceiling, so behaviour is not saturated. 61% of the 33,496 KC -> MBON synapses were never
touched and 1.5% reached the floor, so the rule is not saturated either.

## Reading

**The fly learned something odour-specific this time.** Conditioning specificity 15% ->
71% (reward) and -6% -> 104% (punish); behavioural bucket gradient +0.010 -> +0.404;
per-cell correlation with what the situation deserves -0.065 -> +0.881; training reward
rate up instead of down; frozen control unmoved. **No trained action readout**: the
only mutable quantity *during learning* is the weight of 33,496 KC -> MBON synapses
under a dopamine-gated depression rule driven by the game's chips. (What *is* fitted,
before learning and without ever seeing an action or an outcome: the decision bias and
temperature, the 4,064 per-KC thresholds, and the per-pulse gain ratio.)

**It did not help it win.** Clear 0.328 -> 0.244 and chips are flat. That is far better
than v1's 0.350 -> 0.033 collapse, but the fly is still at or below its naive starting
point and well short of what reading the score bucket alone buys. *Corrected (v3): the
original text quoted `z = 1.75 over 180 games`. There were 60 games, evaluated by three
training runs (the same 60 seeds counted three times), so that z treats one game as
three independent observations and is not a valid interval. The paired, seed- and
run-clustered version is in `outputs/plast3/eval_mode.json`. The "0.633 the score
bucket alone buys" is also a 60-seed number from a search over three nested bucket
policies; on 400 fresh seeds the best of those policies gets 0.530 and always-discard
gets 0.383.*

**Two candidate causes, one measured and one not.** *Corrected (v3): the original
heading here read "the bottleneck moved out of the mushroom body, and both halves of it
are measured". Only the first half was measured; the second was an inference from a
probe that could not support it, and neither establishes that the mushroom body has
stopped being a limit.*
(1) **Measured.** Punishment fires only on the *terminal* losing play, so of 282
`lt0.25` plays in a training run, 0 were rewarded, 28 punished and **254 (90%) got no
dopamine at all**. The decision the fly most needs to change is invisible, and a
depression-only rule has no way to undo drift it did not cause (reward outnumbers
punishment 5:1 as a result).
(2) **Not established.** `outputs/plast2/neighbour.json`: after homeostasis,
conditioning transfer falls with odour distance as it should: Hamming 2 / 4 / 6-8 /
>=10 gives median relative spread 0.836 / 0.348 / 0.157 / 0.000, against a flat
0.796 / 0.782 / 0.683 / 0.634 before. The v2 reading was that because the
score-vs-needed bucket is **one bit of 32**, the two odours the fly must separate
("this Two Pair clears, that one doesn't") sit at Hamming distance 2 where transfer is
still 0.80-0.84. But that probe's distance-2 bin was **arbitrary pairs of held-out
patterns** (mostly best-slot-mask changes, not bucket changes), so it never measured
the discrimination it was used to explain. The test that does hold a hand fixed and
changes only its bucket is `outputs/plast3/transfer.json`; see the v3 section for what
it found.

Artifacts: `outputs/plast2/REPORT.md` (full write-up), `tuned_config.json`,
`kc_homeo.json`, `gate.json`, `eta_calib.json` (+ `_pass1`), `neighbour.json`,
`baselines.json`, `run_<name>.json` x9, `summary.json`, logs. Code:
`flybalatro/tuning.py` (two additive per-KC knobs, defaults no-op),
`scripts/kc_homeo.py`, `scripts/plast2_{gate,eta,run,neighbour,summary}.py`,
`tests/test_kc_homeo.py`. 221 tests pass. `outputs/bc*/`, `outputs/mb*/` and
`outputs/plast/` untouched. Reproduce: see `outputs/plast2/REPORT.md`.

---

# Plasticity v3: the audit fixes, and a preregistered 2x2 (2026-09-13)

An external review of the two sections above found one bug in the evaluation, three
overclaims, and one diagnosis the evidence did not support. This round fixes the bug,
corrects the record in place above, measures the unsupported diagnosis, and
runs the rescue it implies. **The protocol, the primary outcome, the analysis and the
decision rule were written down in `outputs/plast3/PREREGISTRATION.md` before any game
here was played.** Nothing under `outputs/bc*/`, `outputs/mb*/`, `outputs/plast/` or
`outputs/plast2/` was modified.

## 1. The evaluation was a sample, not a greedy policy

Covered in the corrections above and in `outputs/plast3/eval_mode.json`. Greedy
does **not** change the v2 conclusion. It moves the naive fly 0.328 ->
0.267 and the learned fly 0.244 -> 0.178 on the same 60 games, leaving the learned-minus
-naive gap at -0.089 [-0.217, +0.033] (it was -0.083 sampled), and widening the gap to
always-discard from -0.156 to -0.222 [-0.367, -0.083]. Provenance: the v3 cell
`current_current_s0` is protocol-identical to v2's `real_eta0.05_s0` and its weights
come out **bit-identical** to the published file (0 of 33,496 synapses differ), a
re-run of `real_eta0.02_s0` reproduces its json to every decimal, and the sampled
column reproduces the published clear rates exactly.

## 2. The bucket-only conditioning probe (`outputs/plast3/transfer.json`)

v2 blamed "Hamming distance 2" using a probe that binned *arbitrary* held-out patterns,
so its distance-2 bin was mostly best-slot-mask changes. This one holds a real
decision-point pattern fixed and changes **only** the score bucket: 16 anchors, 20
pulses at eta 0.05, with a hand-type-only contrast on the same anchors.
`relative spread = delta(probe)/delta(anchor)`; headline is the ratio of means, medians
in brackets.

| encoding | arm | bucket-only (the pair the fly must separate) | hand-type-only (contrast) |
|---|---|---|---|
| `current` | reward | **0.615** (0.686) | 0.498 (0.435) |
| `current` | punish | 0.915 (0.072) | 0.888 (0.303) |
| `separated` | reward | **0.388** (0.331) | 0.658 (0.625) |
| `separated` | punish | 0.465 (0.082) | 0.593 (0.276) |

Anchor effects are the same size in both flies (reward `|delta_self|` 4.81 vs 4.67 Hz),
so the ratios compare. **The v2 diagnosis is confirmed in direction**: under the current
encoding, conditioning one hand transfers to *the same hand in a different score bucket*
more than it transfers to *a different hand type in the same bucket* (0.615 vs 0.498),
which is exactly backwards from what the task needs. Under the separated encoding the
ordering reverses (0.388 vs 0.658). The preregistered thresholds were `current >= 0.6`
(**met**, 0.615) and `separated <= 0.3` (**missed**, 0.388). Reported as a miss.

## 3. The separated encoding

Four disjoint, ORN-count-balanced glomerulus ensembles for the four score buckets
(441 / 439 / 441 / 443 ORNs, max/min 1.009), paid for by the 19 relay bits the
plasticity harness cannot use: bits 17-25 and 27 are **never** on at a decision point
(0 of 1,091 in the 60 standard games), bit 26 is **always** on, and bits 9-16 name the
cards the harness selects itself whichever way the fly decides. Per decision point the
regime barely moves (6.77 vs 7.04 active glomeruli), but the driven-ORN count goes
from 430.5 (SD 49.7) to 475.6 (SD **1.3**), because `best_slot` carried the hand-type
label in the total drive by construction and now nothing does. **Declared confound:
factor B changes three things at once** (wider bucket channels, dead bits dropped,
intensity cue removed); it is one intervention and no claim is made about which part did
the work. The odour universe collapses to 9 x 4, of which 28 occur, exactly
the joint table v2 already reported, so nothing the fly's binary decision could use is
lost.

Both recalibrations used the existing label-free procedures. Per-KC homeostasis for the
separated fly converged at iteration 11 like v2's: cells that ever respond 600 ->
**1,781**, KC active per odour 237.8 -> **201.9**, cells answering to more than half the
odours 182 -> **26**, offsets -6.74 .. +19.92 mV. Its pulse-gain ratio is
`eta_punish / eta_reward = 1.59` against the current encoding's 3.81.

## 4. The 2x2, 400 paired games (seeds 400000-400399), 3 training seeds per cell

Reward unchanged everywhere. `omission` punishes **every** PLAY that did not make its
fair share `needed / plays_left`, the negative branch of the inequality that already
defines reward, computed from the game's own chips. It is engineered reward shaping
computed from the game only; there is no teacher in it. It fires on 3-4x as many hands
(258-346 punishment pulses per run against 81-109).

| cell | mode | clear | its frozen control | delta vs frozen [95% CI] | delta vs always-discard [95% CI] | P(play\|legal) | bucket gradient |
|---|---|---|---|---|---|---|---|
| current / current (= the v2 fly) | greedy | 0.223 | 0.287 | -0.064 [-0.125, -0.003] | -0.159 [-0.223, -0.097] | 0.790 | +0.498 |
| current / current | sampled | 0.233 | 0.362 | -0.129 [-0.188, -0.070] | -0.149 [-0.210, -0.087] | 0.763 | +0.424 |
| current / separated | greedy | 0.195 | 0.172 | +0.022 [-0.077, +0.147] | -0.188 [-0.297, -0.058] | 0.835 | +0.407 |
| current / separated | sampled | 0.210 | 0.217 | -0.007 [-0.067, +0.052] | -0.172 [-0.237, -0.106] | 0.775 | +0.416 |
| omission / current | greedy | 0.264 | 0.287 | -0.023 [-0.083, +0.036] | -0.118 [-0.180, -0.057] | 0.758 | +0.536 |
| omission / current | sampled | 0.272 | 0.362 | -0.091 [-0.154, -0.029] | -0.111 [-0.174, -0.048] | 0.726 | +0.468 |
| **omission / separated** | greedy | **0.407** | 0.172 | **+0.234 [+0.141, +0.320]** | +0.024 [-0.072, +0.113] | 0.646 | **+0.938** |
| **omission / separated** | sampled | **0.413** | 0.217 | **+0.196 [+0.120, +0.269]** | +0.031 [-0.048, +0.109] | 0.640 | +0.831 |

Baselines on the same 400 games: always-discard **0.383**, always-play 0.037, play iff
`ge1.0` 0.403, play iff `>= lt1.0` **0.530**, play iff `>= lt0.5` 0.448, v2 teacher dig
rule 0.672. Intervals are 10,000-resample bootstraps clustering on **both** the
evaluation seed and the training run, on the per-seed paired difference.

**The two factors are useless apart and large together.** Deltas against each cell's own
frozen control, greedy: punishment alone -0.023, encoding alone +0.022, both **+0.234**.
The interaction contrast is **+0.171**, so the effect is essentially all interaction,
which is the mechanism H-AB predicted: the omission signal is what makes a wasted
`lt0.25` play punishable at all, and the separated encoding is what keeps that
punishment off the buckets the fly should still play.

**What the winning cell learned.** Per run its greedy clear rate is 0.448 / 0.325 /
0.448 and its mean chips 966 / 831 / 966 against always-discard's 835. Two of the three
runs converge on one policy: `P(play) = 0.00` for `lt0.25` and ~1.00 for `lt0.5`,
`lt1.0` and `ge1.0`, the hand-written `bucket_ge_lt1_lt05` baseline, whose 400 game
outcomes they reproduce **bit-for-bit** (0 of 400 differ). The naive separated fly does the
**opposite** (plays `lt0.25` 100% of the time, `lt0.5` 10%, bucket gradient **-0.62**),
so this is a full inversion driven by chips alone. It is not a collapse to
always-discard: it plays 65% of the hands where discarding was legal.

## 5. Verdict, against the rule fixed in advance

A cell succeeded only if it beat **both** its frozen control and always-discard with a
95% clustered CI excluding zero, under greedy evaluation.

**No cell succeeds. `omission / separated` passes the first condition decisively
(+0.234 [+0.141, +0.320]) and fails the second: +0.024 [-0.072, +0.113] against
always-discard is a statistical tie.** It is the first configuration in this project
where the fly's own learning rule produces a policy that is not worse than doing
nothing. It beats its own naive control by 0.234, beats the *other* encoding's naive
control by +0.119 [+0.024, +0.208], earns more chips than always-discard (921 vs 835
mean) and has a near-perfect bucket gradient. But it does not clear more blinds than a
fly that never plays until it must, and it sits **-0.123 [-0.217, -0.035]** against the
best policy that reads only the score bucket (0.530).

So the plasticity story **does not win, and it now fails for a reason that is measured
rather than asserted: the reward, not the learning.**

*Corrected (v4, 2026-09-14): that diagnosis did not survive.
[Plasticity v4](#plasticity-v4-can-any-game-computable-reward-beat-always-discard-2026-09-14)
replaced the reward twice and neither replacement beats always-discard; "paid its
share but still lost the blind", the thing the reward would have to say, turns
out to be inexpressible in **any** immediate same-hand reward term (0 of 2,374
plays), and an eligibility trace that can reach it flips the `lt0.5` ledger from
+115 to -85 without changing the policy at all. A post-hoc probe (v4 section 6)
then hands the rule a perfectly targeted punishment and it still cannot reach the
0.530 policy: 83.6% of the synapses such a pulse can touch are already at the 5%
weight floor. The binding constraint is the capacity of depression-only
KC -> MBON plasticity, not the reward specification. The ledger below is
unchanged; the conclusion drawn from it is superseded.*

The dopamine ledger of the winning cell:

| bucket | plays | rewarded | punished | learned P(play), greedy |
|---|---|---|---|---|
| `lt0.25` | 133 | 0 | **133** | **0.00** |
| `lt0.5` | 236 | **139** | 97 | 1.00 |
| `lt1.0` | 213 | 186 | 27 | 1.00 |
| `ge1.0` | 199 | 198 | 1 | 1.00 |

(v2's cell, same table, for contrast: `lt0.25` 282 plays, 0 rewarded, 28 punished and
**254 with no dopamine at all**; the omission signal turned those 254 into
133 punishments and drove P(play | lt0.25) from 0.54 to 0.00.)

`lt0.5` comes out **net reward-positive** (139 vs 97), so the reward function *tells*
the fly to play it, and the fly obeys. That is not a learning failure; it is the reward
specification. *Corrected (v4): the ledger is right and that inference is not; see
the note above and v4 sections 6 and 7.* `reward = chips_gained >= needed / plays_left` means "make your fair
share of this blind", and with three or four plays left a hand scoring 0.25-0.5x what is
still needed usually does make its share, while still not clearing the blind. The fly
converged on the optimum of the signal it was given, and that optimum is
`play iff bucket >= lt0.5`, i.e. `bucket_ge_lt1_lt05` at 0.448. Two of the three runs
reproduce that hand-written baseline's 400 game outcomes **bit-for-bit** (0 of 400
differ); the third plays `lt0.25` 17.6% of the time and scores 0.325. The 0.530 policy
needs "this hand pays its share but still loses the blind" to be *punished*, and nothing
in the current reward says that. The v1 and v2 caveats already flagged that "the right
thing" is defined by the same fair-share inequality the teacher uses; this round is
where that caveat becomes the binding constraint.

Two claims elsewhere in this file should be read with the same caution. The 0.757 in
the v3 imitation table is an MLP on a **random projection** of the 32 relay bits
(0.755 is the MLP on the bits themselves), an **achieved baseline**, not an
information-theoretic ceiling; and the "0.633 best odour-only policy" of the v1/v2
sections was a search over three nested bucket policies on 60 seeds, which on 400 fresh
seeds is 0.530.

Artifacts: `outputs/plast3/PREREGISTRATION.md`, `eval_mode.json`, `transfer.json`,
`kc_homeo_sep.json`, `tuned_config_sep.json`, `eta_calib_sep.json`, `baselines3.json`,
`run_<cell>.json` x14, `weights_<cell>.npz` x12, `summary.json`, `summary.md`, logs,
and `v2retrain/` (the four v2 weight sets that were never saved). Code:
`scripts/plast3_{common,evalmode,homeo,eta,transfer,run,summary}.py`,
`tests/test_plast3.py` (23 new tests; 395 pass, 1 skipped, with
`tests/test_pov.py` excluded because it reads the live
`outputs/realgame/log.jsonl` fixture, which a concurrently running real-game
session rewrote). Reproduce:
`python -m scripts.plast3_evalmode`, then `plast3_homeo`, `plast3_eta`,
`plast3_transfer`, `plast3_run --all`, `plast3_summary`.


# Real game: the fly on actual Balatro (2026-09-13)

Everything above is `pylatro`, a Rust reimplementation. This is the same v3
readout driving **Balatro 1.0.1o** itself (the Steam build, LÖVE 11.5, running
in a window) through the BalatroBot mod's JSON-RPC API. Install and launch:
`docs/REALGAME_INSTALL.md`.

Same weights, unchanged: `outputs/bc3/models/real_alpn_kc_dn_mlp.npz` over
ALPN+KC+DN spike counts, `glomerular32` input, 50 ms window, `condition=real`.
`auto` selects it and prints `FLY DECIDES`. Nothing was retrained, refit or
tuned for the real game.

## The bug the real game found

The first attempt never played a card. It toggled `select_card[0]` on and off
for 222 decisions.

The cause was in `flybalatro/realgame/adapter.py`, not in the readout.
`legality_mask` allowed `select_card[i]` on an already-selected slot so the
selection could be toggled off. **`pylatro` has no deselect**: once slot `i`
is selected its action leaves the mask until `play` or `discard` consumes the
selection. So the adapter was offering the readout an action that does not
exist in the action space it was trained and evaluated on, and the readout took
it forever.

That this was a mask bug rather than a weak policy is settled by
`outputs/bc3/eval.json`: `real_alpn_kc_dn/mlp` has `truncated_fraction 0.0`
over 400 headless episodes at `max_steps 200`. It never stalls where the
deselect action does not exist.

A plausible-looking alternative was tested and rejected first: the real game
sorts the hand rank-descending while `pylatro` returns draw order, so the
positional encoding is off-manifold. Replaying the stuck hand under the game's
sorted order, an id-hash permutation of it, and four random permutations gave
the **identical 2-cycle in all six**; after the mask fix all six reach a
`discard`. Hand order is immaterial.

One smaller seam, same session: Lua has a single table type, so the mod emits
`[]` for an empty object, and every base playing card arrives with
`"modifier": []`. `_obj` now reads an empty list as `{}`.

## Three runs, `--ante-end 1`, RED / WHITE, unseeded

| Run | Blinds attempted | Blinds cleared | Chips per blind cleared | Ended | Decisions | Wall |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Small, Big | **1 / 3** | Small 384 | lost the Big Blind at 407/450 | 77 | 134 s |
| 2 | Small, Big, The Goad | **2 / 3** | Small 300, Big 580 | lost the boss at 441/600 | 100 | 194 s |
| 3 | Small, Big, The Hook | **3 / 3, ante 1 cleared** | Small 404, Big 529, Hook 644 | ante 2 reached | 76 | 161 s |

Run 3 beat The Hook (which discards two random cards from hand after every
played hand) with a single four-card play for 644 against a 600 requirement.

Across the three runs: **6 of 9 blinds cleared**, 27 hands played, 17 hands
discarded, and **zero rejected API calls**: every action the mask allowed, the
game accepted, which is the invariant `tests/test_realgame_adapter.py` asserts,
now confirmed against the real thing. The fly bought nothing; it left the shop
immediately in all five visits.

Three runs is an existence proof, not a rate. It is too small a sample to
compare against the headless 39.5% ante-1 clear over 400 episodes, and no such
comparison is claimed here.

## Artifacts

`outputs/realgame/`: `log.jsonl` (the three runs are its first 253 per-decision
records; later sessions the same evening appended to the same file, which now
holds 412),
`log_run{1,2,3}.jsonl`, `summary_run{1,2,3}.json`, `run{1,2,3}.out`,
`logs/<ts>/12346.log` (the mod chain loading), and the evidence:
`balatro_fly.png` (Small Blind, round score 292/300, one hand and two discards
left), `balatro_fly_round_eval.png`, `balatro_fly_shop.png`,
`balatro_fly_firsthand.png`, and `balatro_fly.mov` (28 s, full desktop).

Screenshots come from the API's own `screenshot` endpoint, which needs no
window id and no screen-recording permission. Note that the fly's card
selection is **virtual**: the API has no selection endpoint, so the adapter
carries the selection and submits the indices at play time. The cards are never
highlighted on screen no matter how many `select_card` decisions have been
made, so a screenshot is framed on game state, not on selection.

Code: `flybalatro/realgame/{client,adapter,play}.py`. `tests/` is 246 passed,
1 skipped. `outputs/bc*/`, `outputs/mb*/` and `outputs/plast*/` untouched.

---

# The learning fly on the real game, and an overlay that tracks the cards (2026-09-13)

Two things, both about the same complaint: the demo was showing a *frozen* fly
and a *mock* table.

## 1. `--plastic`: the fly whose synapses change drives real Balatro

`flybalatro/realgame/play.py` drove the Steam build with the **frozen v3
readout**, a scikit-learn model over brain spikes that never changes while it
plays. `flybalatro/realgame/plastic.py` adds `--plastic`, which replaces it with
the fly from "Plasticity v2" above:

* the decision is `mean rate(approach MBONs) - mean rate(avoid MBONs) + bias`,
  a fixed rule over neurotransmitter-assigned pools of the fly's own MBONs, with
  **no trained action readout**: nothing fitted to actions, labels or outcomes;
* the only mutable state is the weight of the 33,496 KC → MBON synapses, moved
  by dopamine-gated depression driven by the chips the game pays;
* the operating point is **loaded, not refitted**: the 4,064 per-Kenyon-cell
  homeostatic thresholds, `apl_kc_only` and `apl_mbon_scale` from
  `outputs/plast2/tuned_config.json`, and the two calibrated scalars `bias`
  = −2.6160 and `temperature` = 0.8383 from `run_real_eta0.05_s0.json`, with
  `eta_punish / eta_reward` = 3.81 from `eta_calib.json`. So a real-game fly and
  the 1,200-hand headless runs above are the same fly at the same operating
  point.

The decision is per **hand**, not per action, so this is its own loop rather
than a policy inside `play()`, which would have run four brain windows to
select four cards. It reuses `scripts/plast_common.py` (`build_setup`,
`hand_context`, `dig_slots`, `resolve_outcome`) and `flybalatro/plasticity.py`
unchanged; `resolve_outcome` in particular stays the **only** definition of
reward and punishment in the project, so the real game and the headless runs
cannot drift apart. Weights start naive by default and are written out on exit,
including on Ctrl-C; `--warm-start` loads
`outputs/plast2/weights_real_eta0.05_s0.npz` instead.

**It has not yet played actual Balatro.**
*Corrected (audit, 2026-09-15): it did, later the same evening. The game was
relaunched at 18:51:50 (`outputs/realgame/logs/2026-09-13T18-51-50/12346.log`)
and `outputs/realgame/log_gameview_runs.jsonl` records seven `--plastic` sessions
against it between 19:07 and 19:33: 105 decisions (55 play, 50 discard), 104
outcomes, 26 reward and 5 punishment pulses, 15 blinds cleared and 5 lost, with
the weights moving (1,149 of 33,496 synapses changed by the end of the longest
session). Those runs were never written up and no clear rate is claimed from
them; what follows describes the offline harness only.*
The game exited before this work
started and the brief said not to relaunch it, so what is on the record is the
same loop against `scripts/realgame_plastic_offline.py`: a real `http.server`
speaking the mod's JSON-RPC dialect, dealing random hands from a real 52-card
deck and scoring played hands with `flybalatro/hands.py`: Balatro's own rules,
the same function that tells the fly what its options are worth. The real
client, the real adapter, the real connectome, the real window, the real
depression, the real loop; a simulated game, with no jokers and no blind
effects.

| run | hands | ended | dopamine | KC → MBON synapses changed | mean w/w0 |
|---|---|---|---|---|---|
| learning, eta 0.05 | 30 | ante cleared | **12 reward**, 0 punish | **2,992 / 33,496 (8.9%)** | 0.99598 |
| `--no-learning` control | 29 | lost | 10 reward, 1 punish *computed* | **0 (0.0%)** | **1.000000** |
| unwinnable blinds | 7 | lost | 0 reward, **1 punish** | 477 (1.4%) | 0.99846 |

The control is the point of that table: identical decisions, dopamine computed
and logged, never delivered, and **not one synapse moves**. The third row exists
because punishment fires only on the *terminal* losing play, so a run that keeps
clearing never exercises that arm at all: the first row never lost, which is
why its punish count is 0.

Nothing here is a clear rate and none of it is comparable with the tables
above: 30 hands against a simulated game is an existence proof that the loop
closes (hand → odour → MBON drive → key press → chips → dopamine → weight
change) and nothing more.

Reproduce: `python -m scripts.realgame_plastic_offline --hands 40`, and
`--no-learning` for the control.

## 2. The overlay lines up with where the cards actually are

The overlay placed its boxes from a **fitted static model** of Balatro's card
fan, which drifts on short hands, mid-animation frames, raised cards and window
resizes. Balatro is LÖVE, so the game knows the answer exactly: every card is a
`Moveable` with a target transform `T` and a *visible* one `VT` that eases
toward it, and the draw path uses `VT`. ~120 additive, `pcall`-wrapped lines in
the BalatroBot mod's `gamestate.lua` now report each hand card's `VT`-derived
screen rect and rotation plus the room/tile scaling, and `overlay.aligned_rects`
prefers them, falling back to the fitted fan when they are absent. Exact by
construction, animation-accurate, and correct for any number of cards. No model
is left to be wrong.

**Written, installed and unit-tested; never executed.** A Lua file is only read
when the game launches, and the game has not launched since.
*Corrected (audit, 2026-09-15): it launched at 18:51:50 that evening, after this
Lua was installed, and the Lua did execute. Every decision record in
`outputs/realgame/log_gameview_runs.jsonl` carries the per-card `rect`
(`x/y/w/h/r/moving`) and the `screen` block that only these additions emit, so
the sentence below, "what is not verified is that the mod emits the fields at
all", is settled, as is the first of the five open items in `docs/POV.md` §3. The
`screen` block answers part of that list's fourth item too: it reports `width`
1209 against `pixel_width` 2418, so the two dimension sources do differ on this
window. The rest of that list is still open, and nothing here establishes that
the boxes land in the right place on a live window. `docs/POV.md` §3 was written
before that session and still says the Lua has never run.*
What is verified is
the Python consumer (10 tests over parsing, fallback, scaling and hands of
1–8 cards) and, on real pixels, that feeding it the card positions *measured
off* `balatro_fly_firsthand.png` puts the boxes on the card borders where the
fitted model is 4.26 px out. What is not verified is that the mod emits the
fields at all. `docs/POV.md` §3 is explicit about which is which.

The overlay itself was also cut back to what the brief asked for: one amber
outline on the cards about to be played (now drawn *rotated with the card*, so
it sits on the card's own border instead of being oversized to cover a tilt),
one line naming the hand and whether it clears the blind, the decision, one dim
context line, and the honesty caption at the bottom edge. Gone: the per-card
rank/suit labels, the grey box on every card, the repeated per-card hand-name
captions, the stray amber line at the window edge, the tint over the game, and
the big verdict block that used to cover the Options / Ante / Round buttons.
Evidence: `outputs/pov/overlay_v2.png` against `outputs/pov/live_test_2.png`.

Code: `flybalatro/realgame/plastic.py`, `scripts/realgame_plastic_offline.py`,
additions to `flybalatro/realgame/{play,adapter,overlay}.py`,
`vendor/balatrobot/src/lua/utils/gamestate.lua`,
`tests/test_realgame_plastic.py` (46 tests). `outputs/plast*/`, `outputs/bc*/`
and `outputs/mb*/` untouched.

## Calyx rewiring control

*Full report: `outputs/calyx/REPORT.md`. Run 2026-09-13.*

The project's central claim, that the evolved wiring helps, rested on a
degree-preserving *global* shuffle that, under the v3 glomerular encoding,
compares a working brain against a near-dead one (ALPN 71.3 Hz real against 0.63
Hz shuffled). That claim is retracted above. This is the control that replaces
it.

**The control.** Rewire one stage only: ALPN -> KC, the mushroom body calyx,
19,980 edges from 686 projection neurons onto 4,064 Kenyon cells. Swap edge
endpoints within strata of `(hemisphere, KC subtype, synapse count, sign)` --
818 strata (median 8 edges, max 212, 202 singletons), spanning subtype cells
from `L/KCg-m` at 4,387 edges down to `L/KCg-s3` at 4, rejecting self-loops
and duplicates. Every in-degree, out-degree, in-weight, out-weight
and per-projection-neuron subtype profile is preserved exactly; ORN -> ALPN, APL
in both directions, KC -> KC, KC -> MBON and all dopaminergic wiring are
untouched. Three independent seeds relocate 86-87% of each Kenyon cell's inputs.
A real rewired graph is built, so tuning and homeostasis operate on the correct
endpoints. Invariants are checked per seed and in `tests/test_calyx.py`.

**The answer: the evolved ALPN -> KC wiring does not help.**

Two things make that statement worth more than the one it replaces. First, the
harness reproduces `outputs/bc3` bit for bit when the rewiring is absent --
imitation 0.7085 / 0.6606, clear rate 0.265 / 0.0325, so nothing is absorbing
the effect. Second, and unlike the global shuffle, the rewired networks land in
the same activity regime *with no recalibration at all*: KC 1.33 Hz real against
1.35-1.39 rewired, 1,893 non-constant readout channels against 1,882-1,889. The
comparison is fair before any correction is applied. It largely stays fair
after per-KC homeostatic thresholds are refitted independently for each graph,
with one exception reported in full there: one rewiring (`rw1`) settles sparser
under homeostasis (KC 0.85 Hz, r>0.3 tail 0.0062 against ~0.02 elsewhere), so
its homeostatic numbers are the least comparable of the four. It is also the
condition furthest from `real` in activity and still matches it downstream, so
the null does not depend on it.

| measure | `real` | three rewirings |
|---|---|---|
| KC hand-type decoding, grouped CV | 0.990 | 0.992 / 0.992 / 0.992 |
| DN hand-type decoding, grouped CV | 0.620 | 0.600 / 0.598 / 0.602 |
| KC Jaccard between/within ratio | 0.9111 | 0.9097 / 0.9060 / 0.9141 |
| KC linear imitation top-1 | 0.7085 | 0.7159 / 0.7246 / 0.7147 |
| KC linear clear rate, 400 episodes | 0.265 | 0.247 / 0.250 / 0.275 |
| KC clear rate, with per-KC homeostasis | 0.380 | 0.383 / 0.357 / 0.338 |

Across 400 episode seeds shared by all conditions, **none of the twelve paired
clear-rate comparisons is significant** (exact McNemar, smallest p = 0.13). The
three rewiring seeds give an empirical null band, and `real` sits inside it on
every measure; on KC imitation it sits *below* all three. DN decoding is below
the bits-on count-only confound (0.664) in every condition, so it remains
consistent with pure intensity encoding, as found earlier.

For scale, in the same measurements a label-free recalibration (per-KC
homeostatic thresholds, no labels, no rewards) moves `real` from 993 to 2,790
effective Kenyon dimensions and from 26.5% to 38.0% clear, in every condition
equally. Calibration matters; which Kenyon cell a projection neuron talks to
does not.

**Against the literature, this is the expected result.** Caron et al. (2013)
report PN -> KC connectivity in *Drosophila* as largely random with respect to
glomerular identity. A model that behaves as if that wiring were random is
agreeing with the biology, so a near-null here **validates** the model rather
than indicting it; a large effect would have been the surprise.

**What this does not license.** One stage was rewired, 19,980 of 5.15M edges;
nothing here says the rest of the connectome is equally arbitrary, and it has
not been tested. Nor is this a claim that PN -> KC wiring is random: the strata
preserve each projection neuron's hemisphere and KC-subtype profile exactly, so
the subtype-level structure that Zheng et al. (2022) found beyond Caron's
randomness is **held fixed by this control, not tested by it**. The claim is
narrower: given those subtype-level biases, the individual-Kenyon-cell identity
of calyx wiring contributes nothing measurable in this model, on this task.


# Plasticity v4: can any game-computable reward beat always-discard? (2026-09-14)

v3 ended with a measured diagnosis rather than a result: the fly converged on the
exact optimum of the reward it was given (`chips_gained >= needed / plays_left`,
"did this play earn its share") and that optimum *ties* always-discard. This
round tests two replacement reward specifications against it. **The protocol, the
primary outcome, the analysis and the decision rule were written down in
`outputs/plast4/PREREGISTRATION.md` before any game here was played** (file
timestamp 00:33:12; first run 00:35:35). Nothing under `outputs/bc*/`, `mb*/`,
`plast/`, `plast2/`, `plast3/`, `calyx/`, `realgame/` or `pov/` was modified;
`flybalatro/plasticity.py` and `scripts/plast_common.py` received additive
changes only.

**Nothing beats always-discard.** The eligibility trace produces the best learned
fly in the project (0.448 greedy against a frozen control of 0.172,
**identical in all three training seeds at all three trace lengths**) and lands
at +0.065 [**+0.000**, +0.130] against always-discard, p = 0.058, Holm-corrected
0.117. That misses the bar fixed in advance by the width of a rounding, and it is
reported as a miss. The pace reward is measurably *worse* than the baseline, as the
preregistration predicted from arithmetic before it was run.

**And the reward is no longer what is stopping the fly.** A post-hoc probe
(section 6, added after the result and marked as such) hands the system the best
signal any reward or credit rule could produce (punishment aimed at exactly the
`lt0.5` odours) and it still cannot reach the 0.530 policy: from the trained
fly, 83.6% of the synapses such a pulse can touch are already at the weight
floor and 120 more pulses move the drive by 0.5%; from a naive fly it drives
`lt0.5` negative but drags `lt1.0` and `ge1.0` down with it. **The binding
constraint is a capacity limit of depression-only KC -> MBON plasticity, not the
reward specification and not credit assignment.**

## 1. The off-by-one audit: there isn't one, and the arithmetic says more

`state.plays` reads 4, 3, 2, 1 over a blind, so it counts the play about to be
made and `share = needed / plays` is correct. Satisfying it exactly every play
clears the blind exactly (`share_{k+1} = share_k`, and the four shares sum to
`required`), and on the last play `plays = 1`, so the rule demands
`chips >= needed`, i.e. it demands clearing. The `max(1, ctx.plays)` guard in
`plast_common.resolve_outcome` is dead code on this harness; left untouched.

**The sharper consequence, and the real finding of the audit.** Because the last
play's share *is* the whole remaining requirement, `reward AND lost` is
impossible by construction, and it is 0 of 2,374 v3 training plays, and 0 in
every v4 condition. So the gap v3 diagnosed, "paid its share but still lost the
blind", **cannot be expressed by any immediate same-hand reward term at all.** It
lives in the *earlier* plays of a blind later lost, and only a trace can reach
them. That argument was not available to v3.

## 2. The candidates, and the constraint every one of them is audited against

Every condition is v3's winning cell: `separated` encoding, `omission`
immediate punishment, `eta_reward = 0.05` / `eta_punish = 0.0797`, 1,200 training
decisions, 10% exploration floor, the same 400 paired evaluation seeds
400000-400399. **Only the reward specification differs.**

| cond | reward | trace | reads | hard constraint |
|---|---|---|---|---|
| A | share (v3, unchanged) | -- | `chips_gained`, `needed`, `plays`, terminal flags | admissible |
| B | pace | -- | `required_score`, `score`, `plays`, `chips_gained`, terminal flags | admissible |
| C | share | gamma 0.5 / 0.8 / 1.0 | the above, plus KC spike counts (the brain) and `lost` (the game) | admissible |
| D | pace | gamma 0.8 (fixed in advance) | as B plus as C | admissible |

No reward term consults `flybalatro/hands.py` about whether the action was
*correct*; `hands.py` only enumerates and scores subsets, which is what produces
the odour. `tests/test_plast4.py::test_no_reward_term_consults_the_hand_analyser`
asserts this structurally over the reward path's source and signature.

**Pace**, spelled out without division:
`(score_before + chips) * P0 >= required * (P0 - plays + 1)`, with `P0` read from
the game at each blind's first decision (4 in 100% of 1,024 observed blinds). It
does not re-baseline: writing `delta` for the deficit against schedule, pace
demands `chips >= required/P0 + delta` and share demands
`chips >= required/P0 + delta/plays`, so pace is stricter when behind, looser when
ahead, and identical on the first play of every blind.

**The trace**: `trace <- min(1, gamma * trace + eligibility(kc_counts))` on each
PLAY, reset at every blind boundary, and one extra PPL1 pulse against that trace
when a blind is lost. PLAY decisions only (a DISCARD earns no dopamine anywhere in
this project, and tracing discards would push the fly toward the degenerate
always-discard policy); the losing play is hit twice when unrewarded; the same
calibrated `eta_punish`; no new scalar. Trace length is justified against
Galili et al. 2011 (a *Drosophila* olfactory trace survives a **15 s** gap),
Handler et al. 2019 (mushroom-body plasticity is sensitive to the order and timing
of odour and dopamine) and Cassenaer & Laurent 2012 (a neuromodulator arriving
*after* Kenyon-cell spiking still gates the change). At roughly 10-20 s per
Balatro hand, gamma 0.5 is a one-hand half-life and gamma 1.0 a flat trace over a
~1 min blind; the sweep brackets the fly's range and the mapping to seconds is
approximate and stated as such in the prereg.

## 3. The provenance gate

`plast4_common.play_game4` reimplements the game loop, because the pace rule needs
`required_score` and the blind's starting play count and `plast_common.py` was
additive-only this round. The preregistration gated the round on that
reimplementation, and nothing else was launched until it passed:

| v4 | v3 | greedy 400 | sampled 400 | chips | weights |
|---|---|---|---|---|---|
| `run_frozen` | `run_frozen_separated` | identical | identical | identical | (v3 saved none) |
| `run_A_s0` | `run_omission_separated_s0` | identical | identical | identical | **0 of 33,496 synapses differ** |
| `run_A_s1` | `run_omission_separated_s1` | identical | identical | identical | **0 of 33,496 differ** |
| `run_A_s2` | `run_omission_separated_s2` | identical | identical | identical | **0 of 33,496 differ** |

## 4. The result

Intervals are 10,000-resample two-way cluster bootstraps over evaluation seed
**and** training run, on the per-seed paired difference. Greedy is the primary
mode.

| cond | reward | gamma | mode | clear | per run | vs frozen [95% CI] | vs always-discard [95% CI] | p | P(play\|legal) | grad | chips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | share | -- | greedy | 0.407 | .448/.325/.448 | +0.234 [+0.141, +0.320] | +0.024 [-0.073, +0.113] | 0.618 | 0.646 | +0.938 | 921 |
| A | share | -- | sampled | 0.413 | .425/.355/.460 | +0.196 [+0.120, +0.269] | +0.031 [-0.048, +0.109] | 0.457 | 0.640 | +0.831 | 939 |
| B | pace | -- | greedy | 0.203 | .113/.448/.048 | +0.030 [-0.133, +0.265] | -0.180 [-0.349, +0.053] | 0.094 | 0.839 | +0.415 | 696 |
| B | pace | -- | sampled | 0.236 | .172/.400/.135 | +0.018 [-0.103, +0.175] | -0.147 [-0.273, +0.010] | 0.064 | 0.756 | +0.463 | 779 |
| **C50** | share | 0.5 | greedy | **0.448** | .448/.448/.448 | **+0.275 [+0.210, +0.338]** | +0.065 [+0.000, +0.130] | 0.058 | 0.631 | **+1.000** | 966 |
| C50 | share | 0.5 | sampled | 0.431 | .427/.438/.427 | +0.213 [+0.158, +0.268] | +0.048 [-0.012, +0.108] | 0.128 | 0.636 | +0.833 | 945 |
| **C80** | share | 0.8 | greedy | **0.448** | .448/.448/.448 | **+0.275 [+0.210, +0.338]** | +0.065 [+0.000, +0.130] | 0.058 | 0.631 | **+1.000** | 966 |
| C80 | share | 0.8 | sampled | 0.431 | .440/.445/.407 | +0.213 [+0.153, +0.272] | +0.048 [-0.017, +0.111] | 0.148 | 0.630 | +0.852 | 954 |
| **C100** | share | 1.0 | greedy | **0.448** | .448/.448/.448 | **+0.275 [+0.210, +0.338]** | +0.065 [+0.000, +0.130] | 0.058 | 0.631 | **+1.000** | 966 |
| C100 | share | 1.0 | sampled | 0.433 | .443/.425/.430 | +0.215 [+0.158, +0.273] | +0.050 [-0.013, +0.113] | 0.126 | 0.629 | +0.848 | 964 |
| D | pace | 0.8 | greedy | 0.207 | .048/.343/.230 | +0.034 [-0.113, +0.175] | -0.176 [-0.325, -0.032] | 0.013 | 0.827 | +0.442 | 716 |
| D | pace | 0.8 | sampled | 0.185 | .110/.247/.198 | -0.033 [-0.113, +0.048] | **-0.198 [-0.283, -0.114]** | 0.000 | 0.794 | +0.365 | 746 |

Shared frozen control (the reward only affects learning, so every condition has
the same naive fly): greedy **0.1725**, sampled 0.2175. Baselines on the same 400
games: always-discard **0.3825**, always-play 0.0375, `bucket_ge1` 0.4025,
`bucket_ge_lt1_lt05` 0.4475, **`bucket_ge_lt1` 0.5300**, teacher-dig 0.6725.

**The decision, against the rule fixed in advance.** Success required a positive
greedy delta against the frozen control with a 95% clustered CI excluding 0,
**and** a positive delta against always-discard with a CI excluding 0 and a
Holm-corrected p < 0.05 across the three novel families `{B, C80, D}`.

| condition | beats frozen | beats always-discard (CI) | p | Holm p | verdict |
|---|---|---|---|---|---|
| A | yes | no | 0.618 | -- | reference |
| B | no | no | 0.094 | 0.117 | no effect |
| C50 | yes | no | 0.058 | -- | sensitivity only |
| **C80** | yes | **no** | 0.058 | 0.117 | **learned, still not competitive** |
| C100 | yes | no | 0.058 | -- | sensitivity only |
| D | no | no (**significantly worse**) | 0.013 | 0.039 | no effect |

**No condition succeeds.** C's lower bound is exactly **+0.000** at the 2.5th
percentile. D's Holm-corrected 0.039 is significance in the **wrong direction**:
it is 0.176 worse than always-discard greedy and -0.198 [-0.283, -0.114] sampled.
Exact McNemar per training run, greedy, never pooled: every C run is 106 / 80
against always-discard (p = 0.067) and 152 / 42 against the frozen control
(p = 7.9e-16).

## 5. What each condition learned

### B and D: the preregistered pace prediction came out right

The prereg predicted, from re-scoring v3's 2,374 logged plays before any v4 run,
that pace would reward **more** `lt0.5` plays than share (524 against 431) and
would reintroduce reward on `lt0.25` (21 against 0), so B would be no better than
A and probably worse. On its own trajectory B rewarded 505 of 763 `lt0.5` plays
and 19 of 501 `lt0.25` plays, drove P(play | `lt0.25`) from 0.06 to **0.58** and
P(play | legal) from 0.646 to 0.839, and scored 0.203. **Confirmed with the sign
and the mechanism.** The two rules disagree on only 4.4-8.8% of plays per run, and
that minority costs 0.24 clear rate because it sits exactly where the decision is
marginal.

### C: the trace does what it was predicted to do, and it is not enough

Dopamine ledger, pooled over the three training runs. One lost blind fires **one**
pulse over the union of that blind's recent plays, so the summed `gamma^k` weight
is an **upper** bound on what a bucket receives and the number of pulses
containing the bucket is a **lower** bound; both are shown.

**C80** (`gamma = 0.8`), the prespecified central cell:

| bucket | plays | reward | punish immediate | trace pulses | trace weight | rewarded-but-lost | net (lower) | net (upper) | P(play) greedy |
|---|---|---|---|---|---|---|---|---|---|
| `lt0.25` | 305 | 0 | 305 | 88 | 127.3 | 0 | -393 | -432 | **0.00** |
| `lt0.5` | 727 | 426 | 301 | 134 | 209.7 | 0 | **-9** | **-85** | **1.00** |
| `lt1.0` | 650 | 579 | 71 | 103 | 125.0 | 0 | +405 | +383 | 1.00 |
| `ge1.0` | 613 | 610 | 3 | 5 | 4.4 | 0 | +602 | +603 | 1.00 |

Against A's ledger, same reward, no trace:

| bucket | A reward / punish | A net | A P(play) | C80 net (upper) | C80 P(play) |
|---|---|---|---|---|---|
| `lt0.25` | 0 / 387 | -387 | 0.06 | -432 | **0.00** |
| `lt0.5` | 431 / 316 | **+115** | 1.00 | **-85** | 1.00 |
| `lt1.0` | 560 / 93 | +467 | 1.00 | +383 | 1.00 |
| `ge1.0` | 583 / 4 | +579 | 1.00 | +603 | 1.00 |

Scoring the three preregistered predictions:

1. **"gamma 0.8 and 1.0 flip `lt0.5` net-negative."** *Right on the ledger, wrong
   on the behaviour.* `lt0.5` goes from +115 in A to -85 (C80) and -134 (C100) on
   the upper bound, and **P(play | lt0.5) stays 1.00 in all nine C runs.** The
   signal flipped sign; the policy did not.
2. **"`lt1.0` stays net-positive, so the fly converges on `play iff >= lt1.0`
   = 0.530."** Half right. `lt1.0` does stay strongly positive (+383), but the fly
   converged on `play iff >= lt0.5` = 0.448. The 0.530 policy appeared in no run.
3. **"P(play | ge1.0) should not move, because a `ge1.0` hand clears the blind and
   so cannot sit in a lost blind's trace."** *Right:* `ge1.0` appears
   in **0-3%** of terminal pulses against `lt0.5`'s 81-85%, and P(play | ge1.0) is
   1.00 everywhere. The implementation check passes.

**What the trace did buy: reproducibility.** A converged on the
`bucket_ge_lt1_lt05` policy in **2 of 3** seeds; its s1 run plays `lt0.25` 17.6%
of the time and scores 0.325. **All 9** C runs converge on it, driving
P(play | `lt0.25`) to exactly 0.000 in every seed at every gamma, and **all nine
reproduce that hand-written baseline's 400 game outcomes bit-for-bit (0 of 400
differ)**, with a bucket gradient of exactly +1.000. That is why C is 0.448 with
zero spread across seeds where A is 0.407 with a spread of 0.12.

### Why the trace could not go further: it is the least selective signal in the system

| condition | terminal pulses | pulses containing `lt0.25` | `lt0.5` | `lt1.0` | `ge1.0` |
|---|---|---|---|---|---|
| C50 | 171 | 0.68 | 0.82 | 0.64 | 0.02 |
| C80 | 158 | 0.56 | **0.85** | **0.65** | 0.03 |
| C100 | 155 | 0.58 | 0.81 | 0.67 | 0.00 |
| D | 220 | 0.80 | 0.78 | 0.56 | 0.01 |

**85% of terminal pulses contain an `lt0.5` play and 65% contain an `lt1.0`
play.** The two buckets the fly must separate to reach 0.530 are punished together
by nearly every terminal pulse. The immediate omission rule is perfectly
selective (an `lt0.25` play earns an `lt0.25` punishment, which is why that
bucket goes to exactly 0.000), and the terminal pulse is not selective at all,
because a blind that was lost contains plays from every bucket in roughly the
proportion the fly plays them. **Terminal credit supplies the right sign and
cannot supply the right target**, and no setting of its strength or its decay
changes that: all three gammas give 0.4475 to four decimal places.

That was going to be this round's verdict. Section 6 is the probe that checks
whether the *alternative*, a perfectly targeted signal, would have worked,
and it shows that it would not, which moves the binding constraint one stage
further back. The selectivity measurement above stands; it is no longer the
explanation.

## 6. Post-hoc probe: can this readout express the 0.530 policy at all?

**Added after the result, labelled post-hoc everywhere it is quoted, and it
changes the verdict below.** It is not a preregistered condition, it plays no
game, it uses no labels and no outcomes, and it promotes nothing: it takes a
trained fly, delivers punishment pulses against a bucket's *own* odours, and
reads `play_drive` off the MBONs. Same category as v3's transfer probe.

Section 5 concluded that terminal credit fails because it cannot target one
bucket. That presumes the alternative: that a *perfectly* selective punishment
**would** put `lt0.5` below zero and leave `lt1.0` above it. The fly never
demonstrated that. What it demonstrated was `lt0.25 -> 0.00` with `lt1.0` at
1.00, a separation between **non-adjacent** buckets, which is an easier problem.
The probe gives the system the best signal any reward or credit rule could ever
produce: 120 punishment pulses aimed at exactly the seven `lt0.5` odours seen at
real decision points. `eta_punish = 0.0797`, the calibrated value, unchanged.

Mean `play_drive` in Hz per bucket (a bucket is played when its drive is >= 0),
24 distinct separated odours drawn from the calibration seed range:

| arm | start | pulse 0 | pulse 120 | `lt0.5` reaches 0? | what happened to the others |
|---|---|---|---|---|---|
| one `lt0.5` odour | C80 weights | `lt0.5` **+9.045** | **+9.002** | **no** | `ge1.0` +6.44 -> +6.82, `lt0.25` and `lt1.0` do not move at all |
| all 7 `lt0.5` odours | C80 weights | `lt0.5` **+9.045** | **+8.494** | **no** | `lt1.0` +4.51 -> +4.63, `ge1.0` +6.44 -> +6.53 |
| all 7 `lt0.5` odours | naive weights | `lt0.5` -0.002 | **-3.877** | yes | **`ge1.0` +1.50 -> -0.17, crossing zero at pulse 38**; `lt1.0` -2.47 -> -3.16; `lt0.25` +1.75 -> **+2.40** |

And the synapses those pulses can reach, the KC -> **approach** MBON
edges gated by the Kenyon cells the target odours drive, which is the only thing
a punishment pulse can move:

| arm | reachable edges | mean w/w0 before | at the floor before | mean w/w0 after | at the floor after |
|---|---|---|---|---|---|
| one `lt0.5` odour, C80 | 1,079 | **0.063** | **83.6%** | **0.050** | **100%** |
| all 7, C80 | 2,171 | 0.304 | 45.9% | 0.148 | 60.4% |
| all 7, naive | 2,072 | 1.000 | 0.0% | 0.306 | 22.4% |

**Two things, and both of them are hard limits.**

1. **From the trained fly the punishment arm is exhausted.** For a single
   `lt0.5` odour, 83.6% of every synapse a punishment pulse can reach is already
   pinned at the 5% weight floor, and 120 more pulses take that to **100%** while
   the drive falls by **0.5%** and stays 9 Hz the wrong side of zero. The
   remaining drive is not reachable by KC -> MBON depression at all: it is there
   because the 426 *reward* pulses that bucket earned depressed its **avoid**
   side, and a depression-only rule has no operation that puts that back. By the
   time any terminal pulse arrives (trace or not, gamma 0.5 or 1.0) there is
   nothing left for it to depress. That is why all three gammas give 0.4475 to
   four decimal places.

2. **From a naive fly the punishment operation is not bucket-selective, even
   when the signal is.** A perfectly targeted `lt0.5` punishment does drive
   `lt0.5` from -0.002 to -3.877, and **`ge1.0` crosses zero with it, at pulse
   38, ending at -0.17 from +1.50**, while `lt0.25` drifts the *wrong* way, up to
   +2.40. `ge1.0` is the bucket that must always be played and its own odours
   were never targeted; it fell anyway. `lt1.0` also falls (-2.47 to -3.16),
   though it starts negative in the naive fly (the naive separated fly's bucket
   gradient is -0.62, as v3 reported), so the informative casualty is `ge1.0`.
   The four buckets have disjoint *glomerulus* ensembles by construction, but
   their Kenyon-cell populations overlap enough that depressing one bucket's
   approach pathway drags an unrelated bucket across the decision boundary.

This readout cannot express the 0.530 policy. Under a
depression-only KC -> MBON rule read out by a fixed approach-minus-avoidance
difference, `lt0.5 -> 0` with `lt1.0 -> 1` is not reachable from either end.

## 7. Verdict

Three candidate diagnoses were live at the start of this round. The round
eliminates two of them and leaves the third, measured.

**Not a reward-specification problem.** That was v3's diagnosis and it does not
survive. Two replacement specifications were tested. The pace rule is measurably
*worse* (0.203 against 0.407; -0.176 [-0.325, -0.032] against always-discard),
exactly as this round's preregistration predicted from arithmetic before it ran.
The eligibility trace, the only mechanism that can reach "paid its share but
still lost the blind" by the section 1 arithmetic, flips that bucket's ledger
from +115 to -85 and **does not change the policy at all**.

**Not, in the end, a credit-assignment problem either, though the trace is too
blunt.** 85% of terminal pulses contain an `lt0.5` play and 65% contain an
`lt1.0` play, and that is measured, so terminal credit
cannot target one bucket. But section 6 hands the system a *perfectly* targeted
signal, which is the ceiling on what any credit rule could deliver, and it still
cannot produce the 0.530 policy from either the trained or the naive fly; from
the naive fly it drags `ge1.0`, whose odours it never touched, across the
decision boundary. Selectivity of the *signal* is not the binding constraint;
the *operation* is not selective either, and it runs out first.

**It is a capacity limit of depression-only plasticity at KC -> MBON.** The rule
has one operation, lowering a weight; it bottoms out at 5% of the original, and
adjacent buckets share enough Kenyon cells that using it on one moves its
neighbour. The fly can express `lt0.25 -> 0.00` while three buckets sit at 1.00,
because `lt0.25` is the *extreme* of the axis and is the only bucket that earns
zero rewards. It cannot express a cut *between* `lt0.5` and `lt1.0`, which is
what 0.530 requires. The synapses, not the reward, capped it at 0.448.

This is a narrower claim than "the fly cannot learn Balatro", and it is the one
the measurements support. It also says what a fifth round would have to change,
and it is not the reward: a bidirectional rule (Handler et al. 2019 report
DopR1/DopR2 directing depression **or** potentiation depending on timing) has the
operation this one is missing. That was outside the frozen spec every round of
this project has run under, and nothing here was changed to chase it.

**The fly reached the optimum of every reward we could compute from the game,
and that optimum is a tie with always-discard.** It reaches it now in every seed
rather than two in three, it earns more chips than always-discard (966 against
835), and at +0.065 [+0.000, +0.130] it is not distinguishable from doing
nothing. The reward is no longer what is stopping it.

Nothing was tuned after seeing a result. The gamma sweep was preregistered as a
sweep; all three values are reported; all three give the same greedy number. The
section 6 probe was added after the result, is marked as such, and promoted no
condition; it only changed the wording of this verdict, from a claim the data
did not support to one it does.

## 8. Two corrections to this section's own framing

* **D's Holm-corrected p of 0.039 is significance in the wrong direction.** The
  decision rule is one-sided (`D_discard > 0`) while the preregistered p is
  two-sided, so a small p on a *negative* delta means D is reliably **worse**
  than always-discard, not nearly better. No verdict changes (D fails on the CI
  direction regardless), but the table row should not be read as a near-miss.
* **The preregistration says a *Drosophila* olfactory trace "persists for at
  least that long" of 15 s.** That overstates Galili et al. 2011, which tested a
  15 s gap and found learning across it; it does not establish 15 s as a lower
  bound on the trace. The correct statement is "survives a 15 s gap". The
  preregistration is left unedited, because editing a preregistration after
  seeing results is worse than carrying a correction beside it.
* **A concurrent process running in the same working copy rewrote every file
  under `outputs/` at 01:09:55, after this round's runs finished.** It was not
  this round's work and it was cosmetic: `outputs/plast3/`'s numbers are
  unchanged (`run_omission_separated_s0.json` still reports clear rate 0.4475
  over 1,200 training records), the provenance gate of section 3 **re-passes
  against the rewritten files on all four cells**, and `plast4_summary`
  regenerates `summary.json` byte-identically afterwards. Recorded because
  section 3's claim rests on those files.

## 9. Artifacts

`outputs/plast4/`: `PREREGISTRATION.md`, `audit.json`, `gate.json`,
`gate_check.log`, `baselines4.json`, `run_{frozen,A,B,C50,C80,C100,D}_s*.json`
(19), `weights_*.npz` (18), `summary.json`, `summary.md`, `capacity.json`,
`capacity.log`, `logs/`.

Code: `scripts/plast4_{common,audit,run,summary,capacity}.py`,
`scripts/plast4_all.sh`,
`tests/test_plast4.py` (38 tests), and one additive method on the learning rule,
`flybalatro.plasticity.KcMbonPlasticity.deliver_eligibility`, which applies the
identical depression step to a caller-maintained eligibility vector so the trace
rule and the immediate rule cannot drift apart. Suite: 466 passed, 1 skipped.

Reproduce:

    python -m scripts.plast4_audit
    python -m scripts.plast4_run --gate        # must pass before anything else
    python -m scripts.plast4_run --baselines
    bash scripts/plast4_all.sh
    python -m scripts.plast4_summary
    python -m scripts.plast4_capacity   # post-hoc, added after the result
