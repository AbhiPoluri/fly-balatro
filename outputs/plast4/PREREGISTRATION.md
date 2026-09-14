# Plasticity v4: preregistration of the reward-function comparison (2026-09-14)

Written **before any v4 game is played**. The only v4 numbers that exist at the
time of writing are the two arithmetic audits in sections 2 and 4, which are
re-analyses of the **already published** training logs in
`outputs/plast3/run_omission_separated_s{0,1,2}.json`. They involve no brain, no
new game and no new condition; they exist so that this document can state
falsifiable, signed predictions rather than vague ones. Both audits are
reproducible by `python -m scripts.plast4_audit`.

Nothing under `outputs/bc*/`, `outputs/mb*/`, `outputs/plast/`, `outputs/plast2/`,
`outputs/plast3/`, `outputs/calyx/`, `outputs/realgame/` or `outputs/pov/` is
modified by anything in this round. `flybalatro/plasticity.py` and
`scripts/plast_common.py` receive **additive** changes only.

---

## 0. What v3 left on the table

`outputs/plast3/` ran a preregistered 2x2 and found that the cell
(`omission` punishment, `separated` encoding) raises the ante-1 greedy clear
rate from its frozen control's 0.172 to **0.407**, +0.234 [+0.141, +0.320]
paired and clustered -- and **ties** always-discard (0.383) at
+0.024 [-0.072, +0.113], against a best-bucket-only-policy reference of 0.530
and a teacher at 0.672. It failed the preregistered success bar, which required
beating always-discard.

Its diagnosis, from its own dopamine ledger, was that this is a **reward
specification** failure and not a learning failure. The reward is

```
reward = cleared  or  chips_gained >= needed / plays_left          (the "share" rule)
```

and in the `lt0.5` score bucket that rule is net reward-positive
(139 rewarded against 97 punished in training run s0), so the signal tells the
fly to play hands worth a quarter to a half of what the blind still needs, and
the fly obeys. Two of the three runs converged on the exact optimum of that
signal -- `play iff bucket >= lt0.5`, the hand-written `bucket_ge_lt1_lt05`
baseline at 0.448 -- reproducing its 400 game outcomes bit-for-bit.

This round asks: **is there a reward, computable from the game alone, whose
optimum beats always-discard?**

## 1. The hard constraint, carried from the whole project

> **Every reward term must be computable from the GAME's own payouts and costs.
> Never from a policy, a teacher, or the hand analyser's opinion about what to
> play. If a term needs `flybalatro/hands.py` to decide what was "correct", it
> is out.**

`hands.py` is used, as everywhere in this project, to *enumerate and score the
subsets* -- that is the harness pressing keys, and it is what produces the odour
the fly smells. It must never be used to say whether the chosen action was the
right one. Each candidate below carries an explicit audit line naming exactly
which game quantities its reward reads.

The quantities the game itself exposes at a decision point, all of them from
`pylatro`'s own state or from the `step` return value, are:

| quantity | source | meaning |
|---|---|---|
| `state.required_score` | game | this blind's chip requirement |
| `state.score` | game | chips banked in this blind so far |
| `state.plays` | game | plays **remaining, including the one about to be made** |
| `state.discards` | game | discards remaining |
| `info["chips_gained"]` | game | chips this play scored (0 when the play ended the blind) |
| `info["stage"]`, `done`, `info["is_win"]`, `info["truncated"]` | game | did the blind clear, did the run end, did it end in a loss |

Nothing else is admissible as a reward term.

## 2. Audit of the existing share inequality (the off-by-one check)

**Finding: there is no off-by-one.** Probed directly on seed 400000, `state.plays`
reads 4, 3, 2, 1 on successive plays of a blind, i.e. it counts the play that is
about to be made. Spreading the remaining requirement over the plays that remain
*including this one* is therefore `needed / plays`, which is what
`plast_common.resolve_outcome` computes. Two consequences confirm it:

* if every play makes exactly its share, the blind clears exactly. With
  `chips_k = needed_k / plays_k`, `needed_{k+1} = needed_k (1 - 1/plays_k)` and
  `plays_{k+1} = plays_k - 1`, so `share_{k+1} = share_k`, and the four shares
  sum to `required`;
* on the **last** play `plays = 1`, so the rule demands `chips >= needed`, i.e.
  it demands clearing. That is the correct terminal case.

The `max(1, ctx.plays)` guard in `resolve_outcome` is therefore dead code on the
ante-1 harness: `plays >= 1` at every decision point where a hand exists. It is
left untouched.

**A second, sharper consequence of the same arithmetic, which is the real
finding of this audit.** Because the last play's share *is* the whole remaining
requirement, a play that "paid its share" on the last play has cleared the blind.
So `reward AND lost` is **impossible by construction**, and the audit confirms it
empirically: across the 2,374 training plays of the three v3 runs, the number of
plays that were rewarded and also carried `lost = True` is **0, 0, 0**.

The diagnosed gap -- "paid its share but still lost the blind" -- therefore
**cannot be expressed by any immediate, same-hand reward term at all**. It lives
entirely in the *earlier* plays of a blind that was later lost, and the only
mechanism that can reach those synapses is a trace. This is the strongest
argument in this document for candidate C, and it was not available to v3.

## 3. Candidates

All four conditions share: the `separated` encoding, the `omission` immediate
punishment rule (every PLAY that was not rewarded earns a PPL1 pulse), the
calibrated `eta_reward = 0.05` / `eta_punish = 0.0797`, the frozen approach-minus
-avoidance MBON decision, ante 1, 1,200 training decisions, 10% exploration
floor, `outputs/plast3/tuned_config_sep.json` and `eta_calib_sep.json` loaded
unchanged. **Only the reward predicate and the presence of the trace differ.**

### A. Baseline -- the v3 `omission / separated` cell, re-run here

Reward: `cleared or chips_gained >= needed / plays`. Unchanged. Re-run inside the
v4 harness so that every comparison in this round is paired against a number
produced by the same code path, and so that the v4 harness's provenance can be
established (section 7).

*Constraint audit*: reads `chips_gained`, `required_score - score` (= `needed`),
`plays`, and the blind/run terminal flags. No `hands.py` verdict. **Admissible.**

### B. Pace -- the cumulative schedule, not the re-baselined share

Reward: the blind is cleared, **or** the cumulative round score after this play
is at or above the linear pace that clears the blind in the plays the blind
started with:

```
score_after * P0 >= required * plays_used_after
```

where `P0` is `state.plays` at the blind's first decision (4 throughout ante 1;
the code reads it rather than assuming it and records its observed distribution),
`plays_used_after = P0 - plays + 1`, `score_after = score_before + chips_gained`,
and `score_before = required - needed`.

The two rules coincide exactly when the blind is on schedule, and on the first
play of every blind. Writing `delta` for the deficit against schedule before the
play, pace demands `chips >= required/P0 + delta` and share demands
`chips >= required/P0 + delta/plays`. So **pace is stricter when the blind is
behind schedule and looser when it is ahead**, and the share rule's re-baselining
is what makes it demand a contribution from a play made when the blind is
already comfortable.

*Constraint audit*: reads `required_score`, `score`, `plays`, `chips_gained`,
terminal flags. No `hands.py` verdict. **Admissible.**

**Signed prediction, from the audit (section 2's script, run on v3's logs).**
Re-scoring all 2,374 v3 training plays under the pace rule flips **5.5-6.9%** of
them, and the flips are lopsided in the *wrong* direction: 131 plays go
share-punished -> pace-rewarded and only 21 go the other way. Per bucket, summed
over the three runs:

| bucket | plays | rewarded by share | rewarded by pace |
|---|---|---|---|
| `lt0.25` | 387 | **0** | **21** |
| `lt0.5` | 747 | **431** | **524** |
| `lt1.0` | 653 | 560 | 556 |
| `ge1.0` | 587 | 583 | 583 |

**So B is predicted to be no better than A and probably worse**: it rewards
*more* `lt0.5` plays than the rule whose over-rewarding of `lt0.5` is the
diagnosed problem, and it reintroduces reward on `lt0.25`, which A drove to
P(play) = 0.00. B is run anyway because it was specified in advance and because a
signed prediction that comes out right is worth more than a candidate quietly
dropped.

### C. Eligibility trace -- terminal credit for a lost blind

The immediate signal is **unchanged from A**. Added: a decaying presynaptic
eligibility trace over the Kenyon cells, and one extra PPL1 (punishment) pulse
delivered against that trace when a blind is **lost**.

```
on each PLAY:   trace <- min(1, gamma * trace + eligibility(kc_counts))
on blind LOSS:  deliver("punish", trace, eta_punish)      # one extra pulse
at each blind boundary: trace <- 0
```

`eligibility(kc_counts)` is the existing presynaptic factor of the learning rule
(`plasticity.KcMbonPlasticity.eligibility`: spikes in the 50 ms decision window,
saturating at `KC_REF = 2`). The clip at 1 keeps the trace inside the same
`[0, 1]` range a single-hand eligibility occupies, so the depression step
`w <- max(floor, w (1 - eta f))` keeps every invariant it already has.

**Fixed design choices, with reasons, stated now:**

* **PLAY decisions only enter the trace.** A DISCARD earns no dopamine anywhere
  in this project; putting discards in the trace would punish the odours the fly
  *discarded* on after a loss, which pushes P(play) down on exactly those odours
  and drives the fly toward always-discard -- the degenerate solution this round
  is trying to beat.
* **The trace resets at every blind boundary.** Losing the Boss Blind must not
  punish the plays that cleared the Small Blind.
* **The losing play is hit twice when it was unrewarded** -- once by the
  immediate omission rule at full eligibility, once by the terminal trace pulse
  at weight ~1. Declared, not corrected: it both failed its own share and lost
  the blind.
* **The terminal pulse uses the same calibrated `eta_punish = 0.0797`.** No new
  scalar is introduced and nothing is tuned.
* **`gamma` is swept, not tuned**: `{0.5, 0.8, 1.0}`, three training seeds each,
  all reported.

*Constraint audit*: the trace reads Kenyon-cell **spike counts** -- the brain,
not a teacher -- and the pulse is gated by `lost`, which is the game's own
terminal outcome. No `hands.py` verdict. **Admissible.**

**Justification of the trace length against the fly literature.** *Drosophila*
supports olfactory **trace** conditioning: Galili et al. (2011, J Neurosci
31:7240) show flies form an odour-shock association across a **15 s** stimulus-free
gap, so an odour trace usable by dopamine persists for at least that long.
Handler et al. (2019, Cell 178:60) show that mushroom-body plasticity is
sensitive to the *order and timing* of odour and dopaminergic reinforcement on a
trial-by-trial basis, with DopR1 and DopR2 directing depression or potentiation
-- i.e. a real coincidence window, not an instantaneous one. Cassenaer & Laurent
(2012, Nature 482:47) demonstrate in the locust mushroom body that a
neuromodulatory signal arriving after Kenyon-cell spiking still gates the
plasticity, which is an eligibility trace in the literal sense.

Mapping that to decisions: one Balatro hand takes roughly 10-20 s of real play,
so `gamma = 0.5` is a half-life of about one hand (~10-20 s, the conservative
end of the fly range), `gamma = 0.8` a half-life of about three hands
(~30-60 s), and `gamma = 1.0` no decay within a blind, which given at most four
plays per blind is a flat trace over about a minute. The sweep therefore
brackets the range the biology supports; the mapping from hands to seconds is
approximate and is stated as such.

### D. B + C

Pace reward and the eligibility trace together, at the **prespecified central**
`gamma = 0.8`. Chosen before any run; `gamma` is not selected for D from C's
results.

### Dropped

Nothing. The set is A, B, C x 3 gammas, D = 18 training runs plus one frozen
control.

## 4. Audit of what the trace would have to do (off-policy, on v3's logs)

Recomputed on the v3 `omission_separated` training logs, grouping plays by blind,
marking blinds that contain a `lost` hand, and summing `gamma^k` over the plays
of those blinds (`k` = plays made after this one within the blind). This is the
weight the terminal pulse would have put on each bucket **had the trace existed
on A's own trajectory**. It is off-policy -- the fly's behaviour changes as it
learns, so this is the *direction of the initial signal*, not the converged
ledger -- and the sum ignores the clip at 1, so it is an upper bound on delivered
weight.

The two dopamine arms are calibrated to equal `|delta play_drive|` per pulse
(`eta_punish / eta_reward = 1.5943`, `eta_calib_sep.json`), so
`net = rewarded - punished - trace_weight` is, to first order in `eta`, the sign
and size of the pressure on P(play) for that bucket. Run s0:

| bucket | rewarded | punished | net (A) | trace g=0.5 | net' | trace g=0.8 | net' | trace g=1.0 | net' |
|---|---|---|---|---|---|---|---|---|---|
| `lt0.25` | 0 | 133 | -133 | 33.1 | -166.1 | 59.8 | -192.8 | 86.0 | -219.0 |
| `lt0.5` | 139 | 97 | **+42** | 37.4 | **+4.6** | 61.1 | **-19.1** | 84.0 | **-42.0** |
| `lt1.0` | 186 | 27 | +159 | 35.4 | +123.6 | 46.4 | +112.6 | 57.0 | +102.0 |
| `ge1.0` | 198 | 1 | +197 | 1.0 | +196.0 | 1.0 | +196.0 | 1.0 | +196.0 |

s1 and s2 give the same picture (`lt0.5` net +20 and +53 under A, going to
-20.9 / +1.2 at `gamma = 0.5`, -49.0 / -35.0 at 0.8, -76.0 / -69.0 at 1.0).

**Signed predictions for C, stated now:**

1. **`gamma = 0.8` and `gamma = 1.0` flip `lt0.5` net-negative in all three runs;
   `gamma = 0.5` is borderline (net +4.6 / -20.9 / +1.2, i.e. it flips in one run
   of three).** So P(play | `lt0.5`) should fall towards 0 at 0.8 and 1.0 and
   should be unstable across seeds at 0.5.
2. **`lt1.0` stays strongly net-positive at every gamma** (+97 to +124), so the
   fly should keep playing it. The predicted converged policy is therefore
   `play iff bucket >= lt1.0` -- which is the hand-written `bucket_ge_lt1`
   baseline, measured at **0.530** on these same 400 seeds, i.e. **above
   always-discard's 0.383**. This is the concrete mechanism by which C could pass.
3. **P(play | `ge1.0`) should not move.** A `ge1.0` hand scores at least what the
   blind still needs, so playing it clears the blind, so it essentially cannot
   appear inside a lost blind: its trace weight is 1.0, 1.0 and 2.0 across the
   three runs, against 133, 134 and 120 immediate punishments in `lt0.25`. **If
   P(play | ge1.0) moves materially under C, something is wrong with the
   implementation** and it will be reported as such.
4. **The risk, stated in advance:** the trace adds punishment and nothing else,
   so C may simply drive P(play) down everywhere and collapse to always-discard.
   Carrying the v3 clause forward: *a cell that collapses to near-always-discard
   counts as a pass only if it beats the always-discard baseline itself; if it
   merely matches it, the honest description is "the fly learned don't play, not
   when to play", and it will be reported that way.*

## 5. Protocol

Identical to v3 wherever v3 fixed something, so the numbers are comparable.

| item | value |
|---|---|
| encoding | `separated` (v3 factor B), for every condition |
| immediate punishment | `omission` (v3 factor A), for every condition |
| config | `outputs/plast3/tuned_config_sep.json`, loaded unchanged |
| eta | `eta_reward = 0.05`, `eta_punish = 0.0797` from `outputs/plast3/eta_calib_sep.json` |
| decider calibration | 200 hands from seed 300000+, coin-flip policy, before any dopamine; recomputed per process by the same code as v3 |
| training seeds | `200000 + 1000 * s`, `s in {0, 1, 2}`, identical across all conditions |
| training decisions | 1,200 per run |
| exploration floor | 0.1 during training |
| **evaluation** | **the same 400 paired game seeds as v3: 400000-400399** |
| evaluation modes | `greedy` (primary) and `sampled` (secondary, fixed RNG seed 50000) |
| frozen control | one, shared: the naive `separated` fly. The reward function only affects learning, so all six conditions have the *same* frozen control by construction; it is re-run here and checked bit-identical to `outputs/plast3/run_frozen_separated.json` |
| baselines | always-discard, always-play, the three bucket policies, teacher-dig, re-measured on the same 400 seeds |

## 6. Primary outcome, analysis, decision rule

**Primary outcome.** Ante-1 **clear rate under greedy evaluation on the 400
paired seeds**, for each condition, expressed twice:

```
D_frozen(cond) = mean over 3 runs of mean over 400 seeds of
                 ( cleared[run, seed] - cleared[frozen, seed] )
D_discard(cond) = the same against the always-discard policy on the same seeds
```

**Uncertainty.** Two-way cluster bootstrap, 10,000 resamples, resampling the 400
evaluation seeds with replacement **and** the 3 training runs with replacement
(`plast3_common.cluster_bootstrap`, reused unchanged). The three per-run paired
deltas are printed beside every interval. Additionally, **exact McNemar** per
training run against always-discard and against the frozen control on the 400
paired 0/1 outcomes. McNemar is **not** pooled across the three runs -- they
share the 400 seeds, so a pooled table is not a valid 2x2.

**Multiplicity.** The three **novel** condition families are
`{B, C@gamma=0.8, D}`. A is the reference, not a test. `gamma in {0.5, 1.0}` are
**sensitivity analyses** and cannot promote anything. Holm's step-down correction
is applied across the three novel families to the two-sided bootstrap p-value of
`D_discard` under greedy evaluation.

**Decision rule, fixed now.** A condition **succeeds** only if, under greedy
evaluation:

1. `D_frozen > 0` with a 95% clustered CI excluding 0; **and**
2. `D_discard > 0` with a 95% clustered CI excluding 0 **and** a Holm-corrected
   two-sided bootstrap p < 0.05 across the three novel families.

A condition that passes 1 and fails 2 is reported as "learned, still not
competitive". A condition that fails 1 is reported as "no effect". Secondary
outcomes -- chips, P(play | legal), the bucket gradient, the per-bucket dopamine
ledger, the joint `P(play | hand type x bucket)` table, and the sampled-mode
version of everything -- are reported for every condition and **cannot promote a
condition to success**.

**The ledger reported for every condition** separates the two punishment
channels, because a trace hit is not a pulse:

| column | meaning |
|---|---|
| `plays` | PLAY decisions in that bucket during training |
| `reward` | immediate reward pulses |
| `punish_immediate` | immediate omission punish pulses |
| `punish_trace_weight` | summed eligibility weight delivered by terminal trace pulses, attributed to the bucket of the play that contributed it |
| `n_terminal_pulses` | number of terminal pulses that included any weight from this bucket |
| `rewarded_but_lost` | plays that were rewarded and carried `lost` (predicted 0 everywhere, see section 2) |
| `p_play` | learned P(play \| bucket) at greedy evaluation |

**No cherry-picking clauses.** Three training seeds per condition, all three
reported, none dropped. Both evaluation modes reported for every condition. All
three gammas reported. `eta` is fixed and not re-calibrated per condition. If a
run crashes it is rerun with the same seed and the fact is recorded. Every number
in the final table comes from `outputs/plast4/run_<cond>_s<seed>.json` written by
`scripts/plast4_run.py`; nothing is recomputed by hand.

## 7. Provenance gate, before anything else runs

The v4 harness reimplements the game loop (`plast4_common.play_game4`) because it
must record `required_score`, `score` and `P0`, and must fire a pulse at a blind
boundary -- none of which `plast_common.play_game` exposes, and
`plast_common.py` is additive-only this round. That reimplementation is a risk,
so it is gated:

1. run the frozen `separated` control and all three A runs through the v4
   harness;
2. require the 400 greedy outcomes, the 400 sampled outcomes, the 400 chip
   totals and the saved `.npz` weights to be **bit-identical** to
   `outputs/plast3/run_frozen_separated.json` and
   `outputs/plast3/run_omission_separated_s{0,1,2}.json` /
   `weights_omission_separated_s{0,1,2}.npz`.

**Nothing else is launched until that gate passes.** If it does not pass, the
discrepancy is found and fixed before any B/C/D run exists, and the fact is
recorded in the report.

## 8. What would make this round a failure, stated in advance

* **No condition beats always-discard.** Then the honest sentence is: *the fly
  reached the optimum of every reward we could compute from the game, and that
  optimum is a tie with always-discard.* It will be written exactly that way.
* **C collapses to always-discard.** Then the trace supplies punishment and not
  credit, and the conclusion is that terminal credit assignment in a
  depression-only rule can only ever subtract.
* **B beats A.** Then the section 3 prediction was wrong and the pace/share
  arithmetic above is missing something; it will be reported as a failed
  prediction with the number that refutes it.
* **P(play | ge1.0) moves under C.** Implementation bug; reported as such.

No condition is tuned after seeing its result. If nothing passes, nothing passes.
