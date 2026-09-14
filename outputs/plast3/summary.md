### 2x2, 400 paired evaluation games (seeds 400000-400399)

| cell | mode | clear | frozen | delta vs frozen [95% CI] | delta vs always-discard [95% CI] | P(play\|legal) | grad |
|---|---|---|---|---|---|---|---|
| current_current | greedy | **0.223** | 0.287 | -0.064 [-0.125, -0.003] | -0.159 [-0.223, -0.097] | 0.790 | +0.498 |
| current_current | sampled | **0.233** | 0.362 | -0.129 [-0.188, -0.070] | -0.149 [-0.210, -0.087] | 0.763 | +0.424 |
| current_separated | greedy | **0.195** | 0.172 | +0.022 [-0.077, +0.147] | -0.188 [-0.297, -0.058] | 0.835 | +0.407 |
| current_separated | sampled | **0.210** | 0.217 | -0.007 [-0.067, +0.052] | -0.172 [-0.237, -0.106] | 0.775 | +0.416 |
| omission_current | greedy | **0.264** | 0.287 | -0.023 [-0.083, +0.036] | -0.118 [-0.180, -0.057] | 0.758 | +0.536 |
| omission_current | sampled | **0.272** | 0.362 | -0.091 [-0.154, -0.029] | -0.111 [-0.174, -0.048] | 0.726 | +0.468 |
| omission_separated | greedy | **0.407** | 0.172 | +0.234 [+0.141, +0.320] | +0.024 [-0.072, +0.113] | 0.646 | +0.938 |
| omission_separated | sampled | **0.413** | 0.217 | +0.196 [+0.120, +0.269] | +0.031 [-0.048, +0.109] | 0.640 | +0.831 |

| baseline | clear |
|---|---|
| always_discard | 0.383 |
| always_play | 0.037 |
| bucket_ge1 | 0.403 |
| bucket_ge_lt1 | 0.530 |
| bucket_ge_lt1_lt05 | 0.448 |
| teacher_dig | 0.672 |
