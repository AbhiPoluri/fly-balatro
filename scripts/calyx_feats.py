"""Brain features for the calyx control: one condition x one calibration variant.

Same stage as ``scripts/bc_brain_features.py`` and the same contract (for every
*unique* 32-bit relay pattern, ``reset -> drive -> one 50 ms window ->
log1p(spike counts)`` for ALPN, KC, MBON, DN) but over an arbitrary wiring
condition (``real`` or a rewired graph ``rw<seed>``) and an arbitrary
calibration variant:

``raw``    the v3 operating point exactly (``outputs/mb2/tuned_config.json``);
``homeo``  the same plus that condition's own per-KC homeostatic ``v_th``
           offsets from ``scripts/calyx_homeo.py``.

The unique-pattern set is bc3's own (``outputs/bc3/uniq_index.npz``), so the
output directory can be handed to ``scripts/bc_train.py`` unchanged.

Run::

    python -m scripts.calyx_feats --variant raw --conditions real,rw1,rw2,rw3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent
for _p in (str(ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_DIR = ROOT / "outputs" / "calyx"
BC3 = ROOT / "outputs" / "bc3"
VARIANTS = ("raw", "homeo")

_W: dict = {}


def cond_dir(variant: str, cond: str, out_dir: Path = OUT_DIR) -> Path:
    return out_dir / variant / cond


def tuning_dict(variant: str, cond: str, out_dir: Path = OUT_DIR) -> dict:
    """The ``Tuning`` kwargs for this condition / variant, as a plain dict."""
    cfg = json.loads((ROOT / "outputs" / "mb2" / "tuned_config.json").read_text())
    t = dict(cfg["tuning"])
    if variant == "homeo":
        h = json.loads((out_dir / f"homeo_{cond}.json").read_text())
        if not h.get("complete"):
            raise SystemExit(f"homeo_{cond}.json is not complete yet")
        t["kc_vth_offsets"] = [float(x) for x in h["offsets"]]
    elif variant != "raw":
        raise ValueError(variant)
    return t


# ---- worker ---------------------------------------------------------------
def _init(cond: str, tuning: dict, parts_dir: str, window_ms: float,
          drive_mv: float, brain_seed: int, v_jitter: float, gloms) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"
    for p in (str(ROOT), str(_SCRIPT_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from flybalatro import glomerular as G
    from flybalatro.brain import Brain
    from flybalatro.encode import GlomerularMap
    from flybalatro.tuning import Tuning
    from scripts import calyx_common as K

    g = K.graph_for(cond)
    brain = Brain(g, seed=int(brain_seed), v_jitter=float(v_jitter),
                  tuning=Tuning.from_dict(tuning))
    brain.warmup()
    idx, _sl = G.population_indices(g, G.ENCODING_GLOM32)
    _W.update(brain=brain, fm=GlomerularMap(g, list(gloms), drive_mv=float(drive_mv)),
              idx=idx, window=float(window_ms), parts=Path(parts_dir), cond=cond)


def _run_chunk(job):
    chunk_id, packed = job
    brain, fm, idx, window = _W["brain"], _W["fm"], _W["idx"], _W["window"]
    bits = np.unpackbits(packed, axis=1)[:, :32].astype(np.float32)
    out = np.empty((len(bits), len(idx)), np.float16)
    t0 = time.perf_counter()
    for i in range(len(bits)):
        brain.reset()
        counts, _ = brain.step(fm.drive(bits[i]), window)
        out[i] = np.log1p(counts[idx].astype(np.float32)).astype(np.float16)
    tmp = _W["parts"] / f"{chunk_id:05d}.tmp.npy"
    np.save(tmp, out)
    tmp.replace(_W["parts"] / f"{chunk_id:05d}.npy")
    return chunk_id, len(bits), time.perf_counter() - t0


# ---- driver ---------------------------------------------------------------
def run(variant: str, cond: str, workers: int = 2, chunk: int = 250,
        out_dir: Path = OUT_DIR, force: bool = False) -> dict:
    import multiprocessing as mp

    from flybalatro import glomerular as G
    from scripts import calyx_common as K

    d = cond_dir(variant, cond, out_dir)
    out_path = d / "feats_real.npz"
    if out_path.exists() and not force:
        print(f"[{variant}/{cond}] exists; skipping", flush=True)
        return {"skipped": True}
    parts = d / "_parts"
    parts.mkdir(parents=True, exist_ok=True)

    spec = G.load_spec()
    uniq = np.load(BC3 / "uniq_index.npz", allow_pickle=False)["uniq_packed"]
    n = len(uniq)
    tuning = tuning_dict(variant, cond, out_dir)
    bounds = list(range(0, n, chunk)) + [n]
    jobs = [(c, uniq[bounds[c]:bounds[c + 1]]) for c in range(len(bounds) - 1)
            if not (parts / f"{c:05d}.npy").exists()]
    done0 = n - sum(len(j[1]) for j in jobs)
    print(f"[{variant}/{cond}] {n:,} unique patterns, {len(jobs)} chunks to run "
          f"({done0:,} rows already on disk)", flush=True)

    t0 = time.perf_counter()
    if jobs:
        ctx = mp.get_context("spawn")
        args = (cond, tuning, str(parts), spec.window_ms, spec.drive_mv,
                spec.brain_seed, spec.v_jitter_mv, list(spec.type_names))
        with ctx.Pool(max(1, workers), initializer=_init, initargs=args) as pool:
            done = 0
            for k, (cid, cn, _s) in enumerate(pool.imap_unordered(_run_chunk, jobs), 1):
                done += cn
                el = time.perf_counter() - t0
                if k % 5 == 0 or k == len(jobs):
                    rate = done / el
                    print(f"[{variant}/{cond}] {done:,}/{n - done0:,} rows "
                          f"{el/60:.1f} min  {rate:.1f}/s  "
                          f"ETA {(n - done0 - done)/max(rate,1e-9)/60:.1f} min",
                          flush=True)

    g = K.graph_for(cond)
    pops = G.pops_for_encoding(G.ENCODING_GLOM32)
    sizes = [len({"alpn": g.alpn_indices, "kc": g.kc_indices, "mbon": g.mbon_indices,
                  "dn": g.dn_indices}[p]()) for p in pops]
    slices = G.population_slices(sizes, G.ENCODING_GLOM32)
    M = np.empty((n, sum(sizes)), np.float16)
    for c in range(len(bounds) - 1):
        part = np.load(parts / f"{c:05d}.npy")
        if len(part) != bounds[c + 1] - bounds[c] or part.shape[1] != M.shape[1]:
            raise RuntimeError(f"part {c} has shape {part.shape}")
        M[bounds[c]:bounds[c + 1]] = part
    meta = dict(condition="real", calyx_condition=cond, calyx_variant=variant,
                encoding=G.ENCODING_GLOM32, window_ms=spec.window_ms,
                feature_version=2, n_features=32, brain_seed=spec.brain_seed,
                shuffle_seed=None, threshold=5, n_unique=int(n),
                pops=list(pops), pop_sizes={p.upper(): int(w)
                                            for p, w in zip(pops, sizes)},
                tuning=tuning if variant == "raw" else
                {k: v for k, v in tuning.items() if k != "kc_vth_offsets"},
                kc_vth_offsets_source=(f"outputs/calyx/homeo_{cond}.json"
                                       if variant == "homeo" else None),
                graph_provenance=g.meta.get("calyx_rewiring", "real MaleCNS v1.0"),
                wall_seconds=round(time.perf_counter() - t0, 1))
    np.savez_compressed(out_path, **{p: M[:, slices[p]] for p in pops},
                        uniq_packed=uniq, meta=np.array(json.dumps(meta)))
    del M
    # bc_train.py needs these two next to the features.
    link = d / "states.npz"
    if not link.exists():
        link.symlink_to(Path("../../../bc2/states.npz"))
    idxp = d / "uniq_index.npz"
    if not idxp.exists():
        import shutil
        shutil.copy(BC3 / "uniq_index.npz", idxp)
    for p in parts.glob("*.npy"):
        p.unlink()
    parts.rmdir()
    print(f"[{variant}/{cond}] wrote {out_path} "
          f"({out_path.stat().st_size/1e6:.0f} MB) in "
          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return meta


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="raw", choices=VARIANTS)
    ap.add_argument("--conditions", default="real,rw1,rw2,rw3")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for cond in [c.strip() for c in a.conditions.split(",") if c.strip()]:
        run(a.variant, cond, a.workers, a.chunk, Path(a.out_dir), a.force)


if __name__ == "__main__":
    main()
