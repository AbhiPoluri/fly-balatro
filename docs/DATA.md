# Data provenance

Everything in this project except the Balatro engine comes from one dataset.

## MaleCNS v1.0

The complete male *Drosophila melanogaster* central nervous system connectome,
released 2026-09-03 by Janelia / FlyEM and Google Research — 166,700 neurons
across brain and ventral nerve cord.

| | |
|---|---|
| Release | MaleCNS v1.0, 2026-09-03 |
| Download portal | <https://male-cns.janelia.org/download/> |
| Announcement | <https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/> |
| Paper | doi:[10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015) (*Cell*) |
| Files used | the flat-connectome release, `minconf-0.5` |

Three files, ~1.1 GB, fetched from the public bucket the portal points at:

```
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
  body-annotations-male-cns-v1.0-minconf-0.5.feather        # 14 MB
  body-neurotransmitters-male-cns-v1.0.feather              # 43 MB
  connectome-weights-male-cns-v1.0-minconf-0.5.feather      # 1.05 GB
```

They belong in `data/malecns_v1/` and are gitignored — re-downloadable, and far
too large to track. `flybalatro/connectome.py` reads them once and caches a
derived signed CSR graph at `data/connectome_v2_t5.npz` (86 MB, also gitignored,
`_CACHE_VERSION = 2`).

**Licence.** The URLs above are served without a licence file alongside them,
and the release notes on the portal are the authority. This repository ships
**no** connectome data — only code that reads what you download yourself — so
whatever terms Janelia/FlyEM attach apply directly to you. Check the portal
before redistributing anything derived from the raw files. If you cite the data,
cite the *Cell* paper above.

## What this project does to it, before anything else happens

Recorded in `Graph.meta` on every build and reproduced in
`outputs/*/REPORT.md`:

| step | effect |
|---|---|
| keep entries with an assigned `superclass` | drops entries with none |
| drop `superclass` starting with `vnc` | **brain only** — the ventral nerve cord is not simulated |
| drop `status == "Glia"` | glia are not neurons |
| **net effect of those three node filters** | 166,700 released entries → **146,271** simulated neurons |
| keep edges with `weight >= 5` synapses | 5,146,572 edges retained |
| both endpoints must survive the node filters | drops edges leaving the brain |
| sign each neuron from its predicted neurotransmitter | ACh/DA/5-HT/OA → +1, GABA/Glu/His → −1; unknown → the neuron's outgoing edges are dropped |
| edge weight | `sign(pre) × synapse_count × W_SYN`, `W_SYN = 0.275 mV` (Shiu et al. 2024) |

So the "166,700 neurons" of the release and the "146,271 neurons" this project
simulates are different numbers on purpose: the ventral nerve cord is not
simulated.

**The connectome gives topology, synapse counts and a predicted sign. It does
not give per-synapse efficacy, receptor identity, input resistance or spike
threshold.** Every such value in this project is a parameter we chose, and
`outputs/mb/REPORT.md` §1 lists them one by one.

## Balatro

Two of them, and they are not the same program:

* **`pylatro` / `balatro-rs`** — a Rust reimplementation
  (<https://github.com/evanofslack/balatro-rs>, MIT), patched here; see
  `patches/README.md`. Every headless number in `RESULTS.md` comes from this.
* **Balatro 1.0.1o** — the actual Steam game by LocalThunk, driven through the
  BalatroBot mod's JSON-RPC API. No game asset, binary or content is
  redistributed by this repository; you supply your own copy. Only the
  real-game section of `RESULTS.md` uses it.

## Prior art this borrows from

`flybalatro/connectome.py` says so in its own docstring: the loader is adapted
from `fly-craftax` (<https://github.com/liuzihe02/fly-craftax>) and `doomfly`
(<https://github.com/nftechie/doomfly>), two of the "fly brain plays X" projects
that followed the MaleCNS release. `doomfly`'s `datasets.json` is where the
exact file URLs above come from. Neither is vendored into a tracked path.
