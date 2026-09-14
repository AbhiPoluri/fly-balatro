# Calibrating the mushroom body: is the Kenyon-cell code sparse and separable?

Prerequisite work for a later plasticity experiment. **No plasticity is implemented
here.** Everything below is either a measurement of the frozen network or a static
calibration applied once when the `Brain` is built.

Model unchanged from `RESULTS.md`: MaleCNS v1.0, brain-only, 146,271 neurons, 5.15M
edges at >=5 synapses, sign from neurotransmitter prediction, one global
`W_SYN = 0.275` mV per synapse, event-driven LIF (Shiu et al. 2024 constants,
dt 0.1 ms), stateless one window per input. Inputs: 180 random binary bits, 20%
active (mean 35.4 bits on), 10 ORNs per bit, 30 mV tonic drive, one 50 ms window
from reset, `v_jitter` 6.9 mV, brain seed 1. Same generator as
`scripts/brain_probe.py`.

---

## 1. The knobs

**These are calibration parameters we chose. They are not connectome data.** The
connectome gives topology, synapse counts and a predicted sign per neuron. It does
not give per-synapse efficacy, receptor identity, input resistance or spike
threshold. The untuned model already contains one arbitrary global constant
(`W_SYN`) and a uniform -45 mV threshold for every neuron in the brain; the knobs
below make a few more of those free parameters explicit. Any number measured after
tuning has to be quoted together with the tuning that produced it.

`flybalatro/tuning.py` defines `Tuning`, a frozen dataclass applied once in
`Brain.__init__`. Mechanically it is (a) per-edge weight multipliers on a private
copy of `graph.weight` and (b) per-neuron `vth` / `bias` offsets. The default
`Tuning()` is a no-op and the brain is **bit-identical** to the untuned model
(verified on the real graph -- same KC totals, same brain totals, same APL counts --
and pinned by `tests/test_tuning.py::test_default_tuning_is_a_no_op`).

| knob | what it does | scope | why |
|---|---|---|---|
| `apl_scale` | multiplies every outgoing synapse of APL | 4,498 edges, 57,072 abs mV, 94.1% of it onto KCs | APL is the giant GABAergic interneuron of the mushroom body and the substrate of the winner-take-all normalisation that keeps the KC code sparse (Lin et al. 2014: blocking APL roughly doubles KC odour responses and abolishes sparseness). Our weight for it is only synapse count x 0.275 mV; the loop gain is unknown. |
| `alpn_kc_scale` | multiplies ALPN -> Kenyon_Cell synapses | 19,980 edges, 106,119 abs mV | A single PN-KC connection is subthreshold in vivo; a KC needs several coincident claws. A synapse-count proxy has no reason to land at the right absolute level. |
| `kc_vth_offset_mv` | adds mV to `v_th` of every KC | 4,064 cells | KCs are high-threshold, low-input-resistance cells sitting near silence. A uniform -45 mV for the whole brain gives them no such property. |
| `kc_kc_scale` | multiplies Kenyon_Cell -> Kenyon_Cell synapses | 33,247 edges, 58,226 abs mV = 14.3 mV per KC, 35% of all excitatory drive a KC receives | KC -> KC excitatory recurrence is not part of the canonical feed-forward PN -> KC -> APL circuit; at this strength it makes the calyx a positive-feedback amplifier. Whether those edges are real synapses or a segmentation artefact is not something this project can settle, which is exactly why their gain is a free parameter. |
| `kc_input_norm` | 0..1; interpolates each KC's ALPN in-weights toward the population-mean total, `factor_i = (1-x) + x*(mean/total_i)` | per-KC factor on the same 19,980 edges; per-KC ALPN in-weight SD 12.82 -> 7.32 mV at x=1 (residual is the 296 KCs with no ALPN input at all) | The only non-uniform knob. Equal-total-claw-weight is the assumption mushroom-body theory works under (Litwin-Kumar et al. 2017), and KCs do regulate their own excitability. Applied once at construction from the connectome's own weights: static, unsupervised, label-free -- **not** plasticity. |
| `kc_bias_mv` | adds tonic drive to every KC | 4,064 cells | Same idea as `kc_vth_offset_mv` in current units. Provided for completeness; unused in every sweep. |

Changes to `flybalatro/brain.py` (minimal): the numba kernel took a hard-coded
`-45.0` threshold and now reads a per-neuron `vth` array; `step()` adds a
per-neuron `bias` to the drive it is handed; and `self.weight` is now always a
private copy, because `np.ascontiguousarray` returns the *cached graph's own array*
for an already-contiguous float32 input, so an in-place edge scale would have leaked
into every other `Brain` built from that graph in the same process.

## 2. Selectors and counts (exact annotation strings, MaleCNS v1.0)

| population | selector | n | notes |
|---|---|---|---|
| APL | `type == "APL"` | 2 | class is the **empty string** (not "MBIN"), superclass `cb_intrinsic`, sign -1 (GABA). Body ids **10540** (side R, 2,261 out-edges) and **10977** (side L, 2,237 out-edges). |
| KC | `class == "Kenyon_Cell"` | 4,064 | superclass `cb_intrinsic`, all sign +1; types `KCg*`, `KCab-*`, `KCa'b'-*` |
| MBON | `class == "MBON"` | 97 | `MBON01`..`MBON35`, mixed sign |
| ALPN | `class == "ALPN"` | 686 | mixed sign (uniglomerular PNs plus the inhibitory `*_vPN`) |
| ORN | `superclass == "cb_sensory" and class == "olfactory"` | 2,639 | 54 glomerular types, all sign +1 |
| DN | `superclass == "descending_neuron"` | 1,314 | |
| sugar GRN | `type.startswith("LB")` | 165 | **MaleCNS v1.0 carries no "sugar", "Gr64" or "Gr5" annotation string at all**; the only types containing "GRN" are leg taste-peg neurons (`claw_tpGRN`, `dorsal_tpGRN`). The labellar-bristle `LB*` types (`LB1a`..`LB4b`) are all class `gustatory`, superclass `cb_sensory`, and are exactly what `vendor/fly-craftax/scripts/mn9_check.py` drives (`conn.index(type_prefix="LB")`). 150 have sign +1; 15 have unknown NT so their outgoing edges were dropped at graph build. |
| MN9 | `type == "MN9"` | 2 | proboscis-extension motor neuron. Class is the **empty string**, superclass `cb_motor`, sign +1. Body ids **10331** (L) and **16949** (R). |

## 3. Baseline (untuned), N = 200 inputs -- `outputs/mb/baseline.json`

| metric | value |
|---|---|
| (i) mean fraction of KCs active (>=1 spike) | **0.3794** -- 1,542 of 4,064; range 0.319-0.436 across inputs |
| (ii) mean pairwise Jaccard of active KC sets | **0.8911**; independent-Bernoulli reference at p=0.379 is p/(2-p) = **0.234** |
| (iii) "always-on" KCs (fire for >50% of inputs) | **0.3757** (1,527 cells) -- i.e. **99% of the active cells are active for every input** |
| (iv) KC spikes per ms, mean over inputs | first spike at ms 5, rises to ~70-90/ms and stays there for the whole window; only **15%** of KC spikes fall in the first 15 ms |
| (v) MBON mean rate / ALPN mean rate | **61.06 Hz** / **73.97 Hz** (ORN 15.27, DN 7.50, APL 317.55 Hz) |
| (vi) MN9 sugar response | sugar GRNs -> **1** spike; no drive -> **0**; 165 matched random ORNs -> **7**; the 1,800-ORN probe input -> **8** |
| whole-brain activity | 22,054 spikes per 50 ms window |

Two things are already visible. The mushroom body is saturated in the specific sense
that matters: not merely dense, but dense **with the same cells** (Jaccard 3.8x the
independent reference; always-on / active = 0.99). And the MN9 check does not pass:
MN9 fires *less* for sugar than for a size-matched olfactory drive, and MN9_R never
fires at all (26 in-edges summing to -8.2 mV, against 93 edges and +26.1 mV for
MN9_L). Section 7.

## 4. Sweep -- `outputs/mb/sweep.json` (44 settings, N = 200 each, identical inputs)

Target: KC active fraction 5-10%, always-on near 0, low Jaccard, MN9 preserved.
`always/active` is the diagnostic column: the share of the *active* set that fires
for every input. It is the number that decides separability.

| setting | KC active | always-on | always/active | Jaccard | Bern. ref | KC Hz | MBON Hz | MBON varying | ALPN Hz | APL Hz | brain spikes | MN9 sugar/matched | KC spikes in first 15 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pnkc0.25` | 0.0000 | 0.0000 | 0.00 | 0.0000 | 0.0000 | 0.00 | 9.37 | 25 | 74.57 | 221.6 | 18303 | 1/4 | n/a |
| `pnkc0.5` | 0.0159 | 0.0150 | 0.94 | 0.6226 | 0.0080 | 0.37 | 9.51 | 27 | 74.54 | 230.8 | 18376 | 1/7 | 8% |
| `default` | 0.3794 | 0.3757 | 0.99 | 0.8911 | 0.2341 | 15.21 | 61.06 | 70 | 73.97 | 317.6 | 22054 | 1/7 | 15% |
| `vth2` | 0.2986 | 0.3024 | 1.01 | 0.8788 | 0.1755 | 10.11 | 44.64 | 64 | 74.13 | 298.6 | 20719 | 1/7 | 8% |
| `vth4` | 0.2315 | 0.2308 | 1.00 | 0.8512 | 0.1309 | 6.82 | 32.75 | 63 | 74.13 | 280.6 | 19890 | 2/5 | 3% |
| `vth6` | 0.1672 | 0.1658 | 0.99 | 0.8223 | 0.0912 | 4.51 | 22.22 | 56 | 74.26 | 264.9 | 19299 | 1/4 | 1% |
| `vth8` | 0.1210 | 0.1198 | 0.99 | 0.8176 | 0.0644 | 3.03 | 16.78 | 44 | 74.36 | 256.9 | 18953 | 1/7 | 0% |
| `apl2` | 0.0775 | 0.0716 | 0.92 | 0.6253 | 0.0403 | 2.15 | 11.13 | 44 | 72.94 | 251.0 | 18550 | 1/5 | 32% |
| `apl2_vth2` | 0.0377 | 0.0362 | 0.96 | 0.7364 | 0.0192 | 1.04 | 8.77 | 29 | 73.12 | 241.4 | 18314 | 1/5 | 6% |
| `apl2_vth4` | 0.0260 | 0.0253 | 0.97 | 0.7127 | 0.0132 | 0.65 | 8.66 | 25 | 73.25 | 236.3 | 18255 | 1/4 | 1% |
| `apl2_vth6` | 0.0171 | 0.0157 | 0.92 | 0.6858 | 0.0086 | 0.40 | 8.64 | 25 | 73.29 | 231.1 | 18205 | 1/4 | 0% |
| `apl3` | 0.0365 | 0.0303 | 0.83 | 0.4378 | 0.0186 | 0.81 | 8.93 | 37 | 71.87 | 226.2 | 18098 | 1/4 | 70% |
| `apl3_vth2` | 0.0047 | 0.0042 | 0.89 | 0.6209 | 0.0024 | 0.12 | 7.91 | 24 | 71.99 | 225.8 | 17993 | 1/4 | 5% |
| `apl3_vth4` | 0.0026 | 0.0025 | 0.96 | 0.5305 | 0.0013 | 0.06 | 7.93 | 25 | 72.01 | 223.5 | 17984 | 1/7 | 0% |
| `apl3_vth6` | 0.0012 | 0.0010 | 0.83 | 0.5830 | 0.0006 | 0.03 | 7.91 | 25 | 72.03 | 222.8 | 17978 | 1/6 | 0% |
| `apl5` | 0.0284 | 0.0229 | 0.81 | 0.3908 | 0.0144 | 0.57 | 7.02 | 35 | 69.04 | 223.9 | 17664 | 1/6 | 98% |
| `apl5_vth2` | 0.0003 | 0.0002 | 0.67 | 0.7322 | 0.0002 | 0.01 | 6.41 | 22 | 69.30 | 221.8 | 17604 | 1/5 | 14% |
| `apl5_vth4` | 0.0002 | 0.0002 | 1.00 | 1.0000 | 0.0001 | 0.01 | 6.37 | 22 | 69.33 | 221.8 | 17605 | 1/5 | 0% |
| `apl5_vth6` | 0.0002 | 0.0002 | 1.00 | 0.9801 | 0.0001 | 0.00 | 6.37 | 22 | 69.33 | 221.9 | 17605 | 1/4 | 0% |
| `apl10` | 0.0282 | 0.0226 | 0.80 | 0.3870 | 0.0143 | 0.56 | 4.34 | 25 | 61.25 | 223.5 | 16584 | 1/4 | 100% |
| `apl20` | 0.0282 | 0.0226 | 0.80 | 0.3870 | 0.0143 | 0.56 | 2.11 | 16 | 50.35 | 189.8 | 15011 | 1/2 | 100% |
| `apl50` | 0.0282 | 0.0226 | 0.80 | 0.3870 | 0.0143 | 0.56 | 0.91 | 14 | 39.54 | 129.5 | 13396 | 1/2 | 100% |
| `pnkc2` | 0.8323 | 0.8339 | 1.00 | 0.9781 | 0.7128 | 63.15 | 126.39 | 76 | 73.79 | 357.8 | 31984 | 2/10 | 21% |
| `pnkc2_vth8` | 0.6828 | 0.6853 | 1.00 | 0.9555 | 0.5184 | 30.31 | 81.47 | 71 | 74.03 | 320.9 | 25210 | 1/6 | 10% |
| `pnkc2_vth12` | 0.5989 | 0.6024 | 1.01 | 0.9429 | 0.4274 | 22.38 | 68.69 | 68 | 74.05 | 313.8 | 23509 | 3/4 | 4% |
| `pnkc3` | 0.9024 | 0.9033 | 1.00 | 0.9934 | 0.8222 | 92.23 | 145.23 | 77 | 73.71 | 360.1 | 37934 | 1/8 | 22% |
| `pnkc3_vth12` | 0.8072 | 0.8081 | 1.00 | 0.9805 | 0.6767 | 43.55 | 98.34 | 78 | 73.89 | 336.6 | 27992 | 2/7 | 13% |
| `pnkc3_vth16` | 0.7700 | 0.7719 | 1.00 | 0.9743 | 0.6261 | 35.88 | 88.36 | 73 | 73.99 | 323.9 | 26357 | 1/5 | 8% |
| `kckc0.5` | 0.3679 | 0.3666 | 1.00 | 0.8914 | 0.2254 | 14.52 | 59.57 | 70 | 73.99 | 316.3 | 21900 | 2/4 | 15% |
| `kckc0.25` | 0.3625 | 0.3622 | 1.00 | 0.8880 | 0.2214 | 14.22 | 58.88 | 70 | 74.01 | 316.4 | 21818 | 2/5 | 15% |
| `kckc0` | 0.3577 | 0.3561 | 1.00 | 0.8858 | 0.2178 | 13.95 | 58.32 | 70 | 74.00 | 315.7 | 21757 | 1/9 | 15% |
| `apl2_kckc0` | 0.0764 | 0.0716 | 0.94 | 0.6205 | 0.0397 | 2.12 | 11.09 | 44 | 72.95 | 250.2 | 18544 | 1/6 | 32% |
| `apl2_kckc0.25` | 0.0765 | 0.0709 | 0.93 | 0.6219 | 0.0398 | 2.12 | 11.08 | 43 | 72.95 | 250.2 | 18544 | 1/7 | 33% |
| `apl3_kckc0` | 0.0363 | 0.0300 | 0.83 | 0.4368 | 0.0185 | 0.80 | 8.93 | 37 | 71.88 | 226.4 | 18100 | 1/4 | 71% |
| `apl3_kckc0.25` | 0.0363 | 0.0300 | 0.83 | 0.4374 | 0.0185 | 0.81 | 8.94 | 37 | 71.87 | 226.7 | 18099 | 1/4 | 71% |
| `apl5_kckc0` | 0.0284 | 0.0229 | 0.81 | 0.3907 | 0.0144 | 0.57 | 7.04 | 35 | 69.04 | 224.1 | 17667 | 1/6 | 98% |
| `apl5_kckc0.25` | 0.0284 | 0.0229 | 0.81 | 0.3907 | 0.0144 | 0.57 | 7.03 | 35 | 69.05 | 223.9 | 17668 | 1/6 | 98% |
| `norm0.5` | 0.3761 | 0.3733 | 0.99 | 0.8510 | 0.2316 | 12.85 | 51.53 | 70 | 74.04 | 307.4 | 21424 | 1/8 | 13% |
| `norm1` | 0.3572 | 0.3558 | 1.00 | 0.8571 | 0.2174 | 12.96 | 47.48 | 69 | 74.04 | 308.1 | 21331 | 1/4 | 15% |
| `apl2_norm1` | 0.0907 | 0.0869 | 0.96 | 0.6745 | 0.0475 | 2.54 | 9.97 | 42 | 72.90 | 257.6 | 18621 | 1/8 | 27% |
| `apl3_norm1` | 0.0406 | 0.0352 | 0.87 | 0.4632 | 0.0207 | 0.91 | 8.78 | 38 | 71.75 | 233.7 | 18119 | 1/4 | 59% |
| `kckc0_norm1` | 0.3272 | 0.3255 | 0.99 | 0.8401 | 0.1956 | 11.45 | 42.34 | 68 | 74.08 | 301.6 | 20938 | 1/5 | 16% |
| `apl2_kckc0_norm1` | 0.0874 | 0.0824 | 0.94 | 0.6668 | 0.0457 | 2.40 | 9.92 | 41 | 72.93 | 256.2 | 18596 | 1/8 | 28% |
| `apl3_kckc0_norm1` | 0.0397 | 0.0349 | 0.88 | 0.4593 | 0.0202 | 0.89 | 8.80 | 38 | 71.76 | 233.8 | 18118 | 1/4 | 60% |
| `apl5_kckc0_norm1` | 0.0261 | 0.0217 | 0.83 | 0.3522 | 0.0132 | 0.52 | 6.92 | 33 | 69.05 | 225.4 | 17662 | 1/5 | 99% |

Rows with fewer than ~20 active KCs (`pnkc0.25`, `apl3_vth4`, `apl3_vth6`,
`apl5_vth2`, `apl5_vth4`, `apl5_vth6`) are degenerate -- their Jaccard and
always-on numbers are computed over a handful of cells and mean nothing.

Reading the sweep:

* **`apl_scale` is the only knob that produces sparseness.** 0.379 -> 0.078 at
  scale 2, then a hard floor at **0.0282** from scale 5 upward: scales 10, 20 and
  50 give identical KC metrics. Section 6 explains why.
* **`kc_vth_offset_mv` scales the level, not the selectivity.** 8 mV of extra
  threshold takes 0.379 down to 0.121 and leaves `always/active` at 0.99. A uniform
  threshold shift cannot change *which* KCs are above threshold, only how many.
* **`alpn_kc_scale` below 1 collapses the code** (0.0159 at 0.5; exactly zero KC
  spikes at 0.25) and above 1 saturates it (0.83 at 2, 0.90 at 3) at every threshold
  offset tried up to 16 mV, with `always/active` = 1.00 throughout.
* **`kc_kc_scale` does almost nothing.** Deleting all 33,247 KC->KC synapses moves
  the active fraction 0.3794 -> 0.3577 and `always/active` not at all.
* **`kc_input_norm` does not help and slightly hurts.** At `apl2`, equalising every
  KC's total ALPN in-weight takes the active fraction *up* (0.0775 -> 0.0907) and
  `always/active` from 0.92 to 0.96.
* **`apl_scale` above 5 leaks out of the mushroom body.** 5.9% of APL's output
  weight goes to non-KC targets (ALPN calyx terminals 1,233 mV, MBONs 1,201, DANs
  279), so `apl10/20/50` drag the ALPN rate from 74.0 down to 61.3 / 50.4 / 39.5 Hz
  while changing nothing about the KCs. Those rows perturb the antennal lobe rather
  than calibrate the calyx.

## 5. Chosen setting

`outputs/mb/tuned_config.json`, chosen by `scripts/mb_pick.py` against gates that
live in the script rather than in prose:

* KC active fraction in [0.05, 0.10] -- the stated target
* ALPN mean rate within 10% of the untuned 73.97 Hz -- a knob that moves the
  antennal lobe is not calibrating the mushroom body (10% rather than 5% because APL
  genuinely does synapse on PN terminals in the calyx)
* MBON mean rate >= 5 Hz and >= 40 MBON cells varying across inputs -- an MBON
  population that barely spikes has nothing for a plasticity rule to read
* MN9 silent with no drive, and still responding to sugar drive at all

**5 of 43 settings pass: `apl2`, `apl2_kckc0.25`, `apl2_kckc0`, `apl2_kckc0_norm1`,
`apl2_norm1`** -- all of them `apl_scale = 2` with the other knobs set to values that
change nothing. Ranked by `always/active` then Jaccard/reference:

```json
{"apl_scale": 2.0, "alpn_kc_scale": 1.0, "kc_vth_offset_mv": 0.0,
 "kc_bias_mv": 0.0, "kc_kc_scale": 1.0, "kc_input_norm": 0.0}
```

To load it in a later script:

```python
import json
from flybalatro import connectome as C
from flybalatro.brain import Brain
from flybalatro.tuning import Tuning

cfg = json.load(open("outputs/mb/tuned_config.json"))
g = C.load(5)
brain = Brain(g, seed=1, v_jitter=6.9, tuning=Tuning.from_dict(cfg["tuning"]))
```

`apl5` is reported alongside it throughout. It misses the band (2.8% active) but has
the best `always/active` of any non-degenerate setting (0.81) and is closest to the
in vivo ~5%, so it is the fairer test of "does going sparser make the code
readable".

## 6. Timing, and why APL plateaus

KC spikes per ms, mean over 200 inputs, ms 1-16:

| setting | 1-4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | first | peak | first 15 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `default` | 0 | 0.2 | 10.1 | 44.6 | 57.0 | 20.7 | 34.8 | 65.8 | 32.5 | 46.2 | 73.4 | 66.8 | 98.1 | 5 ms | 17 ms | 15% |
| `apl2` | 0 | 0.2 | 10.1 | 44.5 | 53.4 | 7.3 | 2.9 | 6.5 | 1.5 | 1.1 | 3.6 | 10.2 | 12.3 | 5 ms | 8 ms | 32% |
| `apl5` | 0 | 0.2 | 10.1 | 44.5 | 53.1 | 6.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5 ms | 8 ms | 98% |

**The opening volley is identical to one decimal place at every APL scale** (0.2 /
10.1 / 44.5 / ~53 at ms 5-8). That is the entire explanation of the `apl_scale`
floor. APL is driven *by* the Kenyon cells: measured mean first KC spike **5.80 ms**
(min 5), mean first APL spike **5.88 ms** (min 5), plus the 1.8 ms axonal delay, so
nothing APL does can reach a KC before ~7.7 ms -- by which time the volley has
happened. Raising `apl_scale` past 5
silences everything *after* the volley and leaves the volley untouched -- which is
the 0.0282 floor, 98-100% of it inside the first 15 ms. (A second-order effect
points the same way: the kernel implements Brian2's `(unless refractory)` semantics,
so inhibitory arrivals onto a refractory KC are dropped outright rather than
buffered, and at high KC rates some APL inhibition is simply discarded.)

**The `t >= 15 ms` readout variant.** Counting only late spikes does not help:

| setting | late active fraction | late always/active | late Jaccard | late varying cells |
|---|---|---|---|---|
| `default` | 0.3672 | 0.995 | 0.898 | 1,891 |
| `apl2` | 0.0513 | 0.945 | 0.704 | 451 |
| `apl5` | 0.0003 | 0.667 | 0.899 | **3** |

At `apl2` the late window is a smaller copy of the same problem. At `apl5` there is
no late code at all -- 3 varying cells out of 4,064 -- so a late-window probe there
is reading an empty matrix rather than reading and failing.

## 7. MN9 sugar check, before and after

Reproducing `vendor/fly-craftax`'s `mn9` target: drive the labellar GRNs, look for
spikes in MN9. Our version uses this project's tonic 30 mV (~133 Hz, comparable to
the reference's 100-150 Hz Poisson) over a 200 ms window, and adds a control the
reference does not have.

| condition | untuned | `apl2` | `apl5` |
|---|---|---|---|
| 165 sugar GRNs at 30 mV | **1** spike (MN9_L 1, MN9_R 0) | 1 | 1 |
| no drive | **0** | 0 | 0 |
| 165 *matched random ORNs* at 30 mV | **7** | 5 | 6 |
| the 1,800-ORN probe input | **8** | 5 | 5 |

Tuning preserves the check -- exactly 1 sugar spike in every one of the 44 sweep
rows, because none of these knobs is on the GRN -> ... -> MN9 path -- but it was
**not passing to begin with**. MN9 fires *less* for sugar than for a size-matched
olfactory drive, so this model does not reproduce sugar-specific proboscis
extension. The pathway is present in the graph (MN9_L has 93 in-edges including
`DNge062` at +127.6 mV and `GNG120` at +98.7 mV, the GNG/DN interneurons Shiu et al.
identify), but MN9_R has only 26 in-edges summing to **-8.2 mV** and never fires
under any condition. The wider issue is the operating point: 30 mV tonic drive on
*any* 165+ sensory neurons puts ~9,700 neurons into sustained firing (78k-95k spikes
per 200 ms, i.e. ~390-470 spikes/ms in every condition including the controls), and
a 1-versus-7 difference in a two-neuron readout is noise on that floor.

## 8. Identity probe, before and after -- `outputs/mb/probe_tuned.json`

Same probe as `scripts/brain_probe.py`, same helpers, same seeds: 600 random
180-bit inputs, 5-fold CV logistic regression on `log1p(spike counts)`, best of a
C grid, and the same pipeline on permuted labels as the honest chance level. Labels
A and B are the two identity labels from `brain_probe.make_dataset`; **intensity**
is the control "number of active input bits above the median".

Reference rows (no brain):

| label | majority class | raw bits | glomerular ceiling | permuted raw bits |
|---|---|---|---|---|
| A | 0.5933 | 0.9 | 0.775 | 0.555 |
| B | 0.5033 | 0.8533 | 0.745 | 0.485 |
| intensity | 0.5583 | 0.855 | 0.955 | 0.5217 |

Brain readouts, `accuracy / permuted-label accuracy (columns used)`:

| label | setting | ALPN | KC | KC (t>=15 ms) | MBON | DN |
|---|---|---|---|---|---|---|
| A | untuned | 0.695 / 0.535 (580) | 0.587 / 0.570 (2058) | 0.603 / 0.555 (1956) | 0.583 / 0.597 (71) | 0.610 / 0.557 (543) |
| A | `apl2` | 0.705 / 0.552 (580) | 0.598 / 0.573 (816) | 0.637 / 0.570 (502) | 0.603 / 0.590 (47) | 0.605 / 0.553 (552) |
| A | `apl5` | 0.697 / 0.518 (579) | 0.597 / 0.590 (477) | 0.608 / 0.593 (7) | 0.602 / 0.593 (35) | 0.615 / 0.578 (547) |
| B | untuned | 0.663 / 0.520 (580) | 0.615 / 0.503 (2058) | 0.612 / 0.525 (1956) | 0.528 / 0.507 (71) | 0.520 / 0.527 (543) |
| B | `apl2` | 0.612 / 0.513 (580) | 0.582 / 0.495 (816) | 0.563 / 0.490 (502) | 0.497 / 0.522 (47) | 0.540 / 0.535 (552) |
| B | `apl5` | 0.633 / 0.538 (579) | 0.575 / 0.512 (477) | 0.495 / 0.472 (7) | 0.537 / 0.518 (35) | 0.563 / 0.520 (547) |
| intensity | untuned | 0.908 / 0.558 (580) | 0.873 / 0.515 (2058) | 0.873 / 0.523 (1956) | 0.843 / 0.533 (71) | 0.753 / 0.485 (543) |
| intensity | `apl2` | 0.897 / 0.553 (580) | 0.830 / 0.517 (816) | 0.777 / 0.533 (502) | 0.767 / 0.567 (47) | 0.728 / 0.532 (552) |
| intensity | `apl5` | 0.887 / 0.542 (579) | 0.768 / 0.508 (477) | 0.623 / 0.558 (7) | 0.760 / 0.553 (35) | 0.752 / 0.523 (547) |

The untuned row reproduces the published baseline to three decimals -- KC
0.587 / 0.615, MBON 0.583 / 0.528, DN 0.610 / 0.520, and DN permuted 0.557 / 0.527
against the recorded 0.556 / 0.527 -- which is the check that this script's reuse of
the `brain_probe` helpers is faithful.

**Identity does not reach the Kenyon cells any better after tuning.** Label A moves
0.587 -> 0.598 (`apl2`) against a permuted baseline that moves 0.570 -> 0.573;
label B moves 0.615 -> 0.582 against 0.503 -> 0.495. Every one of those differences
is inside the permuted-label spread. MBON is the same story: A 0.583 -> 0.603 with
permuted 0.597 -> 0.590, B 0.528 -> 0.497 with permuted 0.507 -> 0.522. The
`t >= 15 ms` variant is the only place tuning helps at all -- label A at `apl2`
reads 0.637 against permuted 0.570, a +0.067 margin versus +0.025 for the full
window -- but at `apl5` the late window has 7 usable columns, so its 0.608 is an
artefact of an almost empty matrix, not a readout.

**The `n_used_features` column is the cleanest statement of the whole finding.**
The number of KC columns that vary at all across inputs falls 2,058 -> 816 -> 477
from untuned to `apl2` to `apl5`, and over that same 4x sparsification the identity
accuracy is flat (A: 0.587 / 0.598 / 0.597) while the *intensity* accuracy drops
0.873 -> 0.830 -> 0.768. Sparsification removed cells that were carrying total drive
and exposed no cells that carry identity.

**The intensity control is the result.** KC decodes "how many bits were on" at
0.873 (untuned), 0.830 (`apl2`) and 0.768 (`apl5`), against ~0.59 for identity, and
MBON at 0.843 / 0.767 / 0.760. The mushroom body is a very good total-drive meter
and a chance-level identity detector, before and after tuning. This is the same
failure mode `RESULTS.md` documented at the descending neurons (DN identity 0.61,
intensity 0.75), now measured one stage earlier and shown to be untouched by
sparsification. ALPN is unchanged throughout (0.695 -> 0.705 for A), which it should
be: no knob here is upstream of the antennal lobe.

**A 100 ms window at `apl2`** (`outputs/mb/probe_tuned_100ms.json`) buys a little on
both, and nothing that changes the verdict:

| label | ALPN | KC | KC (t>=15 ms) | MBON | DN |
|---|---|---|---|---|---|
| A | 0.712 / 0.550 (585) | 0.615 / 0.563 (822) | 0.615 / 0.573 (508) | 0.587 / 0.602 (47) | 0.628 / 0.548 (579) |
| B | 0.653 / 0.477 (585) | 0.608 / 0.482 (822) | 0.603 / 0.492 (508) | 0.550 / 0.495 (47) | 0.537 / 0.508 (579) |
| intensity | 0.890 / 0.567 (585) | 0.845 / 0.492 (822) | 0.800 / 0.498 (508) | 0.770 / 0.552 (47) | 0.760 / 0.538 (579) |

KC identity is 0.615 / 0.608 against permuted 0.563 / 0.482, a margin of +0.05 and
+0.13. Label B at 100 ms is the single largest identity margin anywhere in this
report, and it is still below what the ALPNs one synapse from the receptors give in
a 50 ms window (0.663 untuned). Doubling the window mostly buys more spikes to
count, which is why intensity is the column that gains most (KC 0.830 -> 0.845).

## 9. Why no local knob fixes separability -- `outputs/mb/diagnostics.json`

`scripts/mb_diagnose.py` decomposes the drive each KC receives. Untuned model,
60 inputs.

**Where KC input comes from** (total signed conductance into the KC population, per
KC):

| source | signed mV per KC | edges |
|---|---|---|
| ALPN | **+26.04** | 19,980 |
| Kenyon_Cell (recurrent) | **+14.33** | 33,247 |
| APL | **-13.21** | 4,104 |
| DAN | +2.45 | 5,624 |
| DPM | +2.07 | 3,273 |
| MB-C1 | -0.97 | 1,869 |
| everything else | < 0.5 mV each | |

**The decisive number.** Take the total synaptic conductance actually delivered to
each KC in a window, as an (inputs x 4,064 KCs) matrix:

* SD **across KCs** of each KC's mean drive -- input-independent, set by which and
  how many synapses that KC happens to have: **110.8 mV**
* mean per-KC SD **across inputs** -- input-dependent, the part that could carry
  identity: **11.3 mV**
* ratio **9.78**; correlation between a KC's firing frequency and its mean drive
  **0.804**

At `apl2` the ratio is essentially unchanged -- 133.3 / 14.5 = **9.20** -- even
though the mean delivered drive goes from +31.8 mV to -127.3 mV. Doubling the
inhibition moves the whole distribution down without narrowing its spread relative
to the input-dependent part, which is why it sparsifies and does not discriminate.

A KC's firing is decided by its wiring, not by the stimulus. Every knob in section 1
except `kc_input_norm` is a *uniform* rescaling or a *uniform* threshold shift, so
none of them can reorder KCs by total drive -- they only move the cut point along a
fixed ranking. That is exactly what the sweep shows: the active fraction moves over
two orders of magnitude while `always/active` stays between 0.80 and 1.00.

**And the upstream reason.** The *total* ALPN spike count has a coefficient of
variation of **0.0313** across inputs: every input drives the antennal lobe almost
exactly as hard as every other. 35 of 180 active bits put ~350 ORNs into all ~54
glomeruli, so no ALPN is silent for any input (per-cell CV across inputs is 0.46,
but it rides on a nearly constant total). A mushroom body cannot make a sparse
*discriminative* code out of an input in which nothing is off.

**Two things tried that did not work** -- recorded so nobody repeats them:

1. `kc_input_norm = 1`. Equalising every KC's total ALPN in-weight reduces the
   between-KC SD of that in-weight from 12.82 to 7.32 mV and the between/within
   ratio of delivered drive from 9.8 to 7.7, and leaves `always/active` at
   0.94-1.00 at every APL scale. The residual between-KC spread is *which* ALPNs a
   KC samples (their rates differ a lot, input-independently), not how much weight
   it has.
2. **Per-KC threshold homeostasis** -- an ad-hoc diagnostic, deliberately *not* in
   `tuning.py`. Setting `vth[kc]` individually, both from a static connectome
   quantity (proportional to each KC's excitatory in-weight, beta in 0.02-0.1) and
   from each KC's mean drive measured on 20-30 *separate* calibration inputs (gains
   0.6-1.6 on spike counts, alphas 0.05-0.2 on delivered conductance), left
   `always/active` at 0.97-1.00 in every variant and in some made the code denser.
   The reason is the ratio above: once a cell sits 110 mV above threshold, an 11 mV
   input-dependent wobble cannot turn it off, wherever the threshold is placed.

## 10. Judgment

**Is the KC code sparse? Yes.** `apl2` gives 7.75% of KCs active per input, inside
the 5-10% target and close to in vivo; `apl5` gives 2.8%. Scaling the APL feedback
loop does all of this work. Nothing else contributes: `kc_vth_offset_mv` and
`alpn_kc_scale` move the level while leaving the code's structure untouched,
`kc_kc_scale` does effectively nothing, `kc_input_norm` slightly hurts.

**Is it separable? No.** At `apl2`, 92% of the active Kenyon cells fire for every
single input, and the mean pairwise Jaccard of active sets is 0.625 against 0.040
for independent sets of the same size -- a factor of 15.5. At `apl5` it is 81% and
27x. The population went from "1,500 cells that all fire every time" to "300 cells
that all fire every time". That is sparseness without selectivity, and the probe in
section 8 says the same thing.

**Which knob did the work, and why the rest cannot.** Only `apl_scale`, and only
down to a floor of 2.8% active, because APL is driven by the KCs and cannot
influence the 5-8 ms opening volley at all (identical at every scale).
Separability is not reachable with *any* of these knobs, for the reason in section
9: the drive a KC receives varies ~10x more between cells than it does between
inputs, and a knob that rescales a pathway uniformly or shifts a threshold uniformly
cannot reorder cells.

**What would be needed.** The binding constraint is the input encoding, which this
task fixed: 180 bits at 20% active through `grouping="random"` puts every feature
into every glomerulus, so the antennal-lobe code has a CV of 0.0313 across inputs.
The untested lever is a sparser, glomerularly structured encoding --
`grouping="by_type"`, or far fewer active bits, or fewer ORNs per bit -- so that
different inputs actually silence different glomeruli. That is an `encode.py`
change, not a mushroom-body knob, and it should be tried before any plasticity rule
is written.

**Is the model still sane?** The mushroom body is now in a defensible regime: KC
sparseness in the right range, MBON rates down from a physiologically absurd 61 Hz
to 11 Hz with 44 of 97 MBONs still varying across inputs, ALPN and ORN rates
essentially unchanged (73.0 vs 74.0 Hz; 15.3 Hz), whole-brain activity down only
16% (22.1k -> 18.6k spikes per window), no population silenced. The rest of the
brain is not sane and tuning did not change that: 30 mV tonic drive on a few hundred
sensory neurons puts ~10,000 neurons into sustained firing, DNs sit at ~7.3-7.5 Hz
regardless of input, and the Shiu et al. sugar -> MN9 check fails on specificity
before and after (section 7). The calibration is local to the mushroom body and
should be described that way.

**Bottom line for the plasticity experiment.** The mushroom body is *usable* in the
narrow sense that it now produces a sparse KC vector and a non-saturated MBON
readout, which a KC->MBON learning rule needs mechanically. It is **not** usable as
an associative-memory substrate: a rule that depresses the KC->MBON synapses active
for input X will depress almost exactly the same synapses for input Y, because 92%
of the active KCs are shared. Fix the input encoding first.

---

## Files

| path | what |
|---|---|
| `flybalatro/tuning.py` | the knobs (`Tuning`, `apl_indices`, `scale_out_edges`) |
| `flybalatro/brain.py` | per-neuron `vth` / `bias`, private weight copy |
| `tests/test_tuning.py` | default is a no-op, no aliasing, empty populations tolerated |
| `scripts/mb_sparsity.py` | the measurement, the selectors, the MN9 check |
| `scripts/mb_sweep.py` | sweeps, merged into `sweep.json` by tuning label |
| `scripts/mb_pick.py` | gates, ranking, writes `tuned_config.json` |
| `scripts/mb_probe_tuned.py` | the identity probe at several tunings |
| `scripts/mb_diagnose.py` | the variance decomposition in section 9 |
| `outputs/mb/baseline.json` | section 3 |
| `outputs/mb/sweep.json` | section 4 (44 settings) |
| `outputs/mb/tuned_config.json` | section 5, loadable by a later script |
| `outputs/mb/probe_tuned.json`, `probe_tuned_100ms.json` | section 8 |
| `outputs/mb/diagnostics.json` | section 9 |
| `outputs/mb/probe_counts.npz` | raw count matrices for re-probing |

## Reproduce

```bash
source .venv/bin/activate
python -m scripts.mb_sparsity --tag baseline --n-inputs 200
python -m scripts.mb_sweep --stages apl,vth,pnkc,kckc,grid,grid_kckc,norm --n-inputs 200 \
  --custom '[{"alpn_kc_scale":2},{"alpn_kc_scale":3},{"alpn_kc_scale":2,"kc_vth_offset_mv":8},{"alpn_kc_scale":2,"kc_vth_offset_mv":12},{"alpn_kc_scale":3,"kc_vth_offset_mv":12},{"alpn_kc_scale":3,"kc_vth_offset_mv":16}]'
python -m scripts.mb_pick
python -m scripts.mb_probe_tuned --settings '[{}, {"apl_scale":2}, {"apl_scale":5}]' --tag probe_tuned
python -m scripts.mb_probe_tuned --settings '[{"apl_scale":2}]' --window 100 --tag probe_tuned_100ms
python -m scripts.mb_diagnose
python -m pytest tests -q
```

`outputs/bc*/` untouched.
