### Conditions x evaluation mode, 400 paired seeds (400000-400399)

| condition | reward | gamma | mode | clear | per-run | vs frozen [95% CI] | vs always-discard [95% CI] | P(play\|legal) | grad |
|---|---|---|---|---|---|---|---|---|---|
| A | share | - | greedy | **0.407** | 0.448 / 0.325 / 0.448 | +0.234 [+0.141, +0.320] | +0.024 [-0.072, +0.113] | 0.646 | +0.938 |
| A | share | - | sampled | **0.413** | 0.425 / 0.355 / 0.460 | +0.196 [+0.120, +0.269] | +0.031 [-0.048, +0.109] | 0.640 | +0.831 |
| B | pace | - | greedy | **0.203** | 0.113 / 0.448 / 0.048 | +0.030 [-0.133, +0.265] | -0.180 [-0.349, +0.052] | 0.839 | +0.415 |
| B | pace | - | sampled | **0.236** | 0.172 / 0.400 / 0.135 | +0.018 [-0.102, +0.175] | -0.147 [-0.273, +0.010] | 0.756 | +0.463 |
| C50 | share | 0.5 | greedy | **0.448** | 0.448 / 0.448 / 0.448 | +0.275 [+0.210, +0.338] | +0.065 [+0.000, +0.130] | 0.631 | +1.000 |
| C50 | share | 0.5 | sampled | **0.431** | 0.427 / 0.438 / 0.427 | +0.213 [+0.158, +0.268] | +0.048 [-0.012, +0.107] | 0.636 | +0.833 |
| C80 | share | 0.8 | greedy | **0.448** | 0.448 / 0.448 / 0.448 | +0.275 [+0.210, +0.338] | +0.065 [+0.000, +0.130] | 0.631 | +1.000 |
| C80 | share | 0.8 | sampled | **0.431** | 0.440 / 0.445 / 0.407 | +0.213 [+0.153, +0.272] | +0.048 [-0.017, +0.111] | 0.630 | +0.852 |
| C100 | share | 1.0 | greedy | **0.448** | 0.448 / 0.448 / 0.448 | +0.275 [+0.210, +0.338] | +0.065 [+0.000, +0.130] | 0.631 | +1.000 |
| C100 | share | 1.0 | sampled | **0.432** | 0.443 / 0.425 / 0.430 | +0.215 [+0.158, +0.273] | +0.050 [-0.013, +0.113] | 0.629 | +0.848 |
| D | pace | 0.8 | greedy | **0.207** | 0.048 / 0.343 / 0.230 | +0.034 [-0.113, +0.175] | -0.176 [-0.325, -0.032] | 0.827 | +0.442 |
| D | pace | 0.8 | sampled | **0.185** | 0.110 / 0.247 / 0.198 | -0.033 [-0.113, +0.048] | -0.198 [-0.282, -0.114] | 0.794 | +0.365 |

Shared frozen control: greedy 0.1725, sampled 0.2175.

### Baselines on the same 400 games

| baseline | clear |
|---|---|
| always_discard | 0.3825 |
| always_play | 0.0375 |
| bucket_ge1 | 0.4025 |
| bucket_ge_lt1 | 0.5300 |
| bucket_ge_lt1_lt05 | 0.4475 |
| teacher_dig | 0.6725 |

### Decision (greedy, preregistered)

| condition | beats frozen | beats always-discard (CI) | bootstrap p | Holm p | verdict |
|---|---|---|---|---|---|
| A | yes | no | - | - | reference |
| B | no | no | 0.0938 | 0.1168 | no effect |
| C50 | yes | no | - | - | sensitivity only |
| C80 | yes | no | 0.0584 | 0.1168 | learned, still not competitive |
| C100 | yes | no | - | - | sensitivity only |
| D | no | no | 0.0130 | 0.0390 | no effect |

**Anything beat always-discard: False**

### Dopamine ledger per score bucket (pooled over the 3 training runs)

**A** -- baseline: v3 omission/separated, share reward

| bucket | plays | reward | punish (immediate) | punish (trace weight) | terminal pulses | rewarded-but-lost | net | P(play) greedy | P(play) sampled |
|---|---|---|---|---|---|---|---|---|---|
| `lt0.25` | 387 | 0 | 387 | 0.0 | 0 | 0 | -387.0 | 0.06 | 0.12 |
| `lt0.5` | 747 | 431 | 316 | 0.0 | 0 | 0 | +115.0 | 1.00 | 0.95 |
| `lt1.0` | 653 | 560 | 93 | 0.0 | 0 | 0 | +467.0 | 1.00 | 0.95 |
| `ge1.0` | 587 | 583 | 4 | 0.0 | 0 | 0 | +579.0 | 1.00 | 0.95 |

**B** -- pace reward: cumulative schedule, no re-baselining

| bucket | plays | reward | punish (immediate) | punish (trace weight) | terminal pulses | rewarded-but-lost | net | P(play) greedy | P(play) sampled |
|---|---|---|---|---|---|---|---|---|---|
| `lt0.25` | 501 | 19 | 482 | 0.0 | 0 | 0 | -463.0 | 0.58 | 0.47 |
| `lt0.5` | 763 | 505 | 258 | 0.0 | 0 | 0 | +247.0 | 1.00 | 0.95 |
| `lt1.0` | 601 | 520 | 81 | 0.0 | 0 | 0 | +439.0 | 1.00 | 0.95 |
| `ge1.0` | 596 | 593 | 3 | 0.0 | 0 | 0 | +590.0 | 1.00 | 0.94 |

**C50** -- share reward + eligibility trace, gamma=0.5

| bucket | plays | reward | punish (immediate) | punish (trace weight) | terminal pulses | rewarded-but-lost | net | P(play) greedy | P(play) sampled |
|---|---|---|---|---|---|---|---|---|---|
| `lt0.25` | 351 | 0 | 351 | 85.2 | 117 | 0 | -436.2 | 0.00 | 0.12 |
| `lt0.5` | 724 | 406 | 318 | 138.2 | 141 | 0 | -50.2 | 1.00 | 0.95 |
| `lt1.0` | 618 | 545 | 73 | 93.1 | 110 | 0 | +378.9 | 1.00 | 0.94 |
| `ge1.0` | 598 | 594 | 4 | 4.0 | 4 | 0 | +586.0 | 1.00 | 0.95 |

**C80** -- share reward + eligibility trace, gamma=0.8

| bucket | plays | reward | punish (immediate) | punish (trace weight) | terminal pulses | rewarded-but-lost | net | P(play) greedy | P(play) sampled |
|---|---|---|---|---|---|---|---|---|---|
| `lt0.25` | 305 | 0 | 305 | 127.3 | 88 | 0 | -432.3 | 0.00 | 0.09 |
| `lt0.5` | 727 | 426 | 301 | 209.7 | 134 | 0 | -84.7 | 1.00 | 0.95 |
| `lt1.0` | 650 | 579 | 71 | 125.0 | 103 | 0 | +383.0 | 1.00 | 0.95 |
| `ge1.0` | 613 | 610 | 3 | 4.4 | 5 | 0 | +602.6 | 1.00 | 0.95 |

**C100** -- share reward + eligibility trace, gamma=1

| bucket | plays | reward | punish (immediate) | punish (trace weight) | terminal pulses | rewarded-but-lost | net | P(play) greedy | P(play) sampled |
|---|---|---|---|---|---|---|---|---|---|
| `lt0.25` | 301 | 0 | 301 | 199.0 | 90 | 0 | -500.0 | 0.00 | 0.09 |
| `lt0.5` | 745 | 445 | 300 | 279.0 | 126 | 0 | -134.0 | 1.00 | 0.95 |
| `lt1.0` | 633 | 558 | 75 | 142.0 | 104 | 0 | +341.0 | 1.00 | 0.94 |
| `ge1.0` | 607 | 607 | 0 | 0.0 | 0 | 0 | +607.0 | 1.00 | 0.94 |

**D** -- pace reward + eligibility trace, gamma=0.8

| bucket | plays | reward | punish (immediate) | punish (trace weight) | terminal pulses | rewarded-but-lost | net | P(play) greedy | P(play) sampled |
|---|---|---|---|---|---|---|---|---|---|
| `lt0.25` | 547 | 31 | 516 | 254.7 | 177 | 0 | -739.7 | 0.56 | 0.58 |
| `lt0.5` | 709 | 458 | 251 | 242.7 | 172 | 0 | -35.7 | 1.00 | 0.94 |
| `lt1.0` | 615 | 508 | 107 | 148.4 | 124 | 0 | +252.6 | 1.00 | 0.95 |
| `ge1.0` | 575 | 573 | 2 | 3.6 | 3 | 0 | +567.4 | 1.00 | 0.94 |

### How bucket-selective the terminal pulse is

One lost blind fires one pulse over the union of that blind's recent plays, so `pulse share` is the fraction of terminal pulses that contain **any** play from that bucket.

| condition | terminal pulses | `lt0.25` | `lt0.5` | `lt1.0` | `ge1.0` |
|---|---|---|---|---|---|
| C50 | 171 | 0.68 | 0.82 | 0.64 | 0.02 |
| C80 | 158 | 0.56 | 0.85 | 0.65 | 0.03 |
| C100 | 155 | 0.58 | 0.81 | 0.67 | 0.00 |
| D | 220 | 0.80 | 0.78 | 0.56 | 0.01 |

### Exact McNemar per training run (greedy, 400 paired games)

| condition | run | vs frozen b10/b01, p | vs always-discard b10/b01, p |
|---|---|---|---|
| A | s0 | 152/42, p=7.89e-16 | 106/80, p=6.65e-02 |
| A | s1 | 112/51, p=2.01e-06 | 77/100, p=9.79e-02 |
| A | s2 | 152/42, p=7.89e-16 | 106/80, p=6.65e-02 |
| B | s0 | 33/57, p=1.49e-02 | 30/138, p=9.36e-18 |
| B | s1 | 152/42, p=7.89e-16 | 106/80, p=6.65e-02 |
| B | s2 | 11/61, p=1.55e-09 | 15/149, p=6.27e-29 |
| C50 | s0 | 152/42, p=7.89e-16 | 106/80, p=6.65e-02 |
| C50 | s1 | 152/42, p=7.89e-16 | 106/80, p=6.65e-02 |
| C50 | s2 | 152/42, p=7.89e-16 | 106/80, p=6.65e-02 |
| C80 | s0 | 152/42, p=7.89e-16 | 106/80, p=6.65e-02 |
| C80 | s1 | 152/42, p=7.89e-16 | 106/80, p=6.65e-02 |
| C80 | s2 | 152/42, p=7.89e-16 | 106/80, p=6.65e-02 |
| C100 | s0 | 152/42, p=7.89e-16 | 106/80, p=6.65e-02 |
| C100 | s1 | 152/42, p=7.89e-16 | 106/80, p=6.65e-02 |
| C100 | s2 | 152/42, p=7.89e-16 | 106/80, p=6.65e-02 |
| D | s0 | 11/61, p=1.55e-09 | 15/149, p=6.27e-29 |
| D | s1 | 112/44, p=5.04e-08 | 86/102, p=2.74e-01 |
| D | s2 | 74/51, p=4.87e-02 | 56/117, p=4.08e-06 |

