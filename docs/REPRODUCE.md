# Reproducing every headline number

Clone to result. Runtimes below are **measured**, not estimated: they are read
off the logs in `outputs/` that the runs themselves wrote, on the machine
everything in `RESULTS.md` was produced on:

> **Apple M2 Pro, 16 GB, macOS 26 (Darwin 25.6.0), CPython 3.11.3, `arm64`.**
> Everything runs on CPU. Nothing needs a GPU; `torch` is used for the readout
> MLP and runs fine on the CPU device.

Two ways to waste an afternoon here:

* **The connectome download is 1.1 GB** and the first `Graph` build takes a
  couple of minutes; after that a cache at `data/connectome_v2_t5.npz` makes it
  seconds.
* **Only §7 needs the real game.** Everything else is headless and
  deterministic given the seeds recorded in each artefact.

---

## 0. The repository

```bash
git clone <this repo> fly-balatro && cd fly-balatro
REPO=$(git rev-parse --show-toplevel)

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[viewer,figures,dev]"      # add ,realgame on macOS for §7
```

Then check it:

```bash
python -m pytest            # 428 passed, 1 skipped in ~31 s
```

The one skip (`pytest -rs` names it) is a v1-era test asserting that no brain
readout exists. One does, so it cannot run.

## 1. The connectome

Download the three MaleCNS v1.0 feather files listed in
[`docs/DATA.md`](DATA.md) into `data/malecns_v1/`:

```bash
mkdir -p data/malecns_v1 && cd data/malecns_v1
BASE=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
curl -O $BASE/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl -O $BASE/body-neurotransmitters-male-cns-v1.0.feather
curl -O $BASE/connectome-weights-male-cns-v1.0-minconf-0.5.feather
cd "$REPO"
```

Build and cache the signed CSR graph:

```bash
python -c "from flybalatro.connectome import build; g = build(); print(g.n, g.m)"
# 146271 5146572      ~2 min the first time, ~3 s from the cache afterwards
```

If those two numbers do not match, stop: every table downstream is keyed to
them. `docs/DATA.md` lists exactly which entries the loader drops and why.

## 2. The engine (`pylatro`)

`pylatro` is **not** installed from PyPI. It is built from the patched
`balatro-rs` checkout; `patches/README.md` explains what the patches do and why
one of them is a real engine bug.

```bash
# apply the patches (patches/README.md has the full commands)
git clone https://github.com/evanofslack/balatro-rs "$REPO/vendor/balatro-rs"
git -C "$REPO/vendor/balatro-rs" checkout 05de340fb075b94f63ccbb4ff6b68304f1ddfb0c
git -C "$REPO/vendor/balatro-rs" apply "$REPO/patches/balatro-rs.patch"

# build the bindings into this venv  (~3 min, needs a Rust toolchain)
pip install maturin
cd "$REPO/vendor/balatro-rs/pylatro" && maturin develop --release && cd "$REPO"

python -c "import pylatro; print(pylatro.__name__)"
```

Sanity check the engine wrapper and the hand analysis, which is the part of this
project that is *not* biological:

```bash
python -m pytest tests/test_env.py tests/test_hands.py -q      # 58 tests, ~2 s
```

## 3. Probes: does input identity reach the descending neurons?

The v1 gate probe: the one behind "the fly brain garbles its input", and the
one the glomerular encoding later reverses.

```bash
python -m scripts.brain_probe                 # ~12 min, writes outputs/probe_*.json
python -m scripts.brain_probe_merge           # -> outputs/brain_probe.json
```

(`brain_probe` takes `--n-samples`, `--window`, `--threshold`; the defaults are
what produced the stored file. `--help` on any script in `scripts/` prints the
full protocol it implements; several of them are long.)

Then the mushroom-body calibration that located the loss in the *input* rather
than the network, and chose the operating point everything after v3 uses:

```bash
python -m scripts.mb2_sample --n 2000        # real states, stratified by hand type
python -m scripts.mb2_assign                 # 32 relay bits -> 32 ORN glomeruli
python -m scripts.mb2_grid                   # 12 settings x 4 populations, ~55 min
python -m scripts.mb2_probe                  # -> outputs/mb2/probes.json
python -m scripts.mb2_gate --write-config    # -> outputs/mb2/tuned_config.json
```

Headline: KC hand-type decoding 0.598 → **1.000**, DN 0.610 → **0.842**, with the
driven-ORN-count-only confound at 0.267. Figure 2.

## 4. Behaviour cloning (v1, v2, v3)

One pipeline, three configurations. **v3 is the one to run**: it is 30× cheaper
than v2 because deduplicating on 32 bits instead of 315 collapses 138,354 inputs
to 4,962 unique ones.

```bash
# 1. collect states with the teacher and a random policy   (45 s, 150,037 states)
python -m scripts.bc_collect --teacher v2 --feature-version 2 \
       --out outputs/bc2/states.npz
mkdir -p outputs/bc3
ln -s ../bc2/states.npz outputs/bc3/states.npz   # v3 recollects nothing

# 2. run the brain once per unique input, both wirings
python -m scripts.bc_brain_features --bc-dir outputs/bc3 --workers 4 \
       --encoding glomerular32 --states outputs/bc2/states.npz
#    real       4,962 windows, 2.2 min   (v2's 138,354 windows took 69.8 min)
#    shuffled   4,962 windows, 1.8 min   (v2: 63.2 min)

# 3. train + evaluate everything, chained
BC=outputs/bc3 TEACHER=v2 WORKERS=4 ENCODING=glomerular32 CACHE_CAP=6000 \
  scripts/bc_pipeline.sh
#    ~35 min total; the 400-episode live evaluation is ~5 min per wiring group
```

For v2 substitute `BC=outputs/bc2` and drop `ENCODING`; for v1 also use
`TEACHER=v1`. Expect **~4 hours** for a full v2 run, almost all of it in step 2.

Headline: real ALPN+KC+DN clears ante 1 on **37.8 %** of 400 episodes (linear
readout) against a 42.2 % no-brain ceiling, where under v2's encoding the same
readout cleared **0**. Figure 3.

## 5. Plasticity: no trained action readout

Per-Kenyon-cell homeostasis first (label-free: each cell sees only its own
firing rate), then the pulse-gain calibration, then the runs.

```bash
python -m scripts.plast_probe --window-scan 50,150,300   # v1 operating point
python -m scripts.plast_kccheck
python -m scripts.kc_homeo                 # 12 iterations, 205 s
python -m scripts.plast2_gate --write-config   # the 8-criterion gate, all PASS
python -m scripts.plast2_eta               # eta_punish / eta_reward = 3.81
python -m scripts.plast2_run --all
python -m scripts.plast2_summary
```

The preregistered 2×2 (**read
[`outputs/plast3/PREREGISTRATION.md`](../outputs/plast3/PREREGISTRATION.md)
first**; the protocol, primary outcome and decision rule were fixed before any
of these games were played):

```bash
python -m scripts.plast3_evalmode          # greedy vs sampled, on the v2 weights
python -m scripts.plast3_homeo             # homeostasis for the separated encoding
python -m scripts.plast3_eta               # its own pulse-gain ratio (1.59)
python -m scripts.plast3_transfer          # the bucket-only conditioning probe
python -m scripts.plast3_run --all         # 4 cells x 3 seeds + 2 frozen controls
python -m scripts.plast3_summary
```

Runtimes, from the run logs: a frozen control is **69–341 s** for 400 games, a
learning cell **206–270 s** per training seed. The whole 2×2 is about **75 min**.

Headline: `omission / separated` reaches **+0.234 [+0.141, +0.320]** over its own
frozen control and **ties** always-discard (+0.024 [−0.072, +0.113]). Under the
preregistered rule that is a **miss**, not a win. Figure 4.

## 6. The calyx rewiring control

The control that replaces the retracted "real beats shuffled" claim.

```bash
scripts/calyx_run.sh          # ~3 h end to end on 4 workers
```

or stage by stage:

```bash
C=real,rw1,rw2,rw3
python -m scripts.calyx_common --seeds 1,2,3          # 3 rewired graphs, 86 MB apiece
python -m scripts.calyx_feats  --variant raw   --conditions $C --workers 2
python -m scripts.calyx_homeo  --conditions $C        # per-graph homeostasis, ~7 min each
python -m scripts.calyx_feats  --variant homeo --conditions $C --workers 2
python -m scripts.calyx_probe  --variant raw   --conditions $C   # decoding + code structure
python -m scripts.calyx_train  --variant raw   --conditions $C --readouts kc,dn
python -m scripts.calyx_eval   --variant raw   --conditions $C --readouts kc,dn --workers 2
python -m scripts.calyx_stats  --variant raw   --conditions $C --readouts kc,dn
python -m scripts.calyx_tables --variant raw   --conditions $C --readouts kc,dn
```

`scripts/calyx_run.sh` is the same sequence for both variants, resumable: every
stage skips work already on disk.

The harness reproducing `outputs/bc3` **bit for bit** when the rewiring is
absent (imitation 0.7085 / 0.6606, clear 0.265 / 0.0325) is the check that says
nothing is absorbing the effect. Figure 5.

Headline: none of the twelve paired clear-rate comparisons is significant
(smallest exact-McNemar p = 0.13), and `real` sits inside the three-seed null
band on every measure.

## 7. The real game — the only step that needs Balatro

Full install: [`docs/REALGAME_INSTALL.md`](REALGAME_INSTALL.md). It is macOS-
specific, needs a Steam copy of Balatro 1.0.1o, Lovely, Steamodded and the
patched BalatroBot mod, and it is the only part of this project that cannot be
reproduced headlessly.

```bash
# terminal 1: the game with the JSON-RPC API on 127.0.0.1:12346
uvx --from "$REPO/vendor/balatrobot" balatrobot serve \
    --gamespeed 1 --animation-fps 60 --no-reduced-motion \
    --logs-path "$REPO/outputs/realgame/logs"

# terminal 2: the fly
python -m flybalatro.realgame.play --ante-end 1 --pause 1.5      # frozen v3 readout
python -m flybalatro.realgame.plastic --hands 40                 # the learning fly
```

> **Do not re-run this into `outputs/realgame/`.** Two tests in
> `tests/test_pov.py` assert over the existing `outputs/realgame/log.jsonl`
> (253 decision records from the three recorded runs) and will fail if it is
> regenerated. Point `--out-dir` somewhere else.

Three runs cleared 6 of 9 blinds and all of ante 1 in run 3. That is an
existence proof, not a rate, and it is not comparable with the headless 39.5 %.

The learning fly's real-game loop has **not** run against actual Balatro; what
is on the record is the same loop against a local server speaking the mod's
dialect:

```bash
python -m scripts.realgame_plastic_offline --hands 40
python -m scripts.realgame_plastic_offline --hands 40 --no-learning   # 0 synapses move
```

## 8. Figures

Every figure regenerates from the JSON already in `outputs/`. No simulation, no
game, no training: it reads artefacts and draws them, and there is not a single
hand-typed result number in the script:

```bash
python -m scripts.figures            # ~20 s, writes figures/*.png at 200 dpi + .svg
python -m scripts.figures --list
python -m scripts.figures --only 3 5
```

The demo clip is cut from the 156 MB screen recording (gitignored, in
`outputs/realgame/balatro_fly.mov`, the Big Blind of run 1, the one it lost at
407/450) with `ffmpeg`. The crop is the Balatro window only: the recording is of
a whole desktop, and the first second of it is not the game, so both the crop
rectangle and the `-ss 14.6` start offset are load-bearing.

```bash
ffmpeg -ss 14.6 -t 8 -i outputs/realgame/balatro_fly.mov \
  -vf "crop=2414:1634:301:136,fps=10,scale=680:-1:flags=lanczos,\
palettegen=max_colors=96:stats_mode=diff" -y /tmp/pal.png
ffmpeg -ss 14.6 -t 8 -i outputs/realgame/balatro_fly.mov -i /tmp/pal.png \
  -lavfi "crop=2414:1634:301:136,fps=10,scale=680:-1:flags=lanczos[x];\
[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -y figures/demo.gif
ffmpeg -ss 14.6 -t 8 -i outputs/realgame/balatro_fly.mov \
  -vf "crop=2414:1634:301:136,fps=24,scale=1200:-2:flags=lanczos" \
  -c:v libx264 -pix_fmt yuv420p -crf 28 -an -movflags +faststart -y figures/demo.mp4
```

## 9. The live dashboard

Not a result, but it is how most of the debugging happened, and
`outputs/pov/dashboard_embedded.png` is what it looks like.

```bash
python -m flybalatro.viewer.server --port 8767 --policy brain     # the v3 readout
python -m flybalatro.viewer.server --port 8769 --policy plastic \
       --plastic-config v2 --learning on                          # the learning fly
```

`.claude/launch.json` carries these as named entries. It is checkout-relative:
the commands are run from the repository root.

---

## What is expensive, ranked

| stage | wall clock | why |
|---|---|---|
| v2 brain features | 69.8 min real + 63.2 min shuffled | 138,354 unique 315-bit inputs |
| calyx, end to end | ~3 h | 8 feature runs (4 graphs × 2 variants) + 8 trainings |
| plasticity 2×2 | ~75 min | 12 training runs + 2 frozen controls, 400 games each |
| mb2 grid | ~55 min | 12 settings × 2,000 states |
| v1 gate probe | ~12 min | 600 inputs × 8 configurations |
| v3 brain features | **4.0 min** | 4,962 unique 32-bit inputs, 96.7 % duplicates |
| connectome build | ~2 min, then cached | 1.05 GB of edges |
| the whole test suite | 34 s | |
| all five figures | ~20 s | reads JSON, draws |

## If a number does not reproduce

Each artefact records the configuration that produced it under a `_config` /
`settings` / `meta` key: seeds, window length, worker count, tuning dict. Diff
that against yours first. `scripts/bc_consistency_check.py` exists for exactly
this: it rebuilds the stored `outputs/bc3` feature rows from a live tuned brain
and asserts they match, for both wirings.
