"""Stage 2: push every collected state through the frozen fly brain.

For each *unique* input vector (the brain is a deterministic, stateless function
of the drive, so duplicates are free), under one wiring condition:

    brain.reset()  ->  drive = <the encoding>.drive(bits)
    ->  one window  ->  log1p(spike counts) per recorded population

Two encodings, selected with ``--encoding``:

``featuremap`` (default, v1/v2)
    ``flybalatro.encode.feature_map_for(g, version)``: every one of the 283 / 315
    bits drives 10 ORNs drawn at random across glomeruli. Records ALPN (686),
    KC (4064), DN (1314).
``glomerular32`` (v3)
    ``flybalatro.glomerular``: only the leading 32 bits -- the hand-type relay
    block -- reach the fly, each driving *every* ORN of one glomerulus, on a
    brain tuned per ``outputs/mb2/tuned_config.json`` (apl_scale 2, tonic 30 mV).
    The other 283 bits are dropped. Deduplication therefore runs on the 32 relay
    bits alone, which collapses 150,037 states to 4,962 inputs. Records ALPN,
    KC, MBON (97) and DN.

Both come out of one shared constructor so that this stage, ``bc_eval``, the
consistency check, the viewer and the real-game player cannot drift apart.
``--feature-version`` must match the one stored in ``states.npz``; version 2 lays
its 32-bit hand-type block on real ORNs.

Conditions: ``real`` (MaleCNS wiring) and ``shuffled`` (``Brain.shuffled(0)``,
edge targets globally permuted, degrees preserved).

Parallelism: multiprocessing with the ``spawn`` context (numba + fork on macOS
is unsafe), one Brain per worker, BLAS/numba pinned to one thread each. Work is
handed out in chunks and every finished chunk is written to
``outputs/bc/_parts/<cond>_<chunk>.npy``, so a crash or a Ctrl-C costs at most
one chunk per worker. Re-running skips parts that already exist and skips a
condition whose final npz exists.

Run:
    source .venv/bin/activate
    nohup python scripts/bc_brain_features.py --workers 6 --window 50 \
        > outputs/bc/brain_features.log 2>&1 &
    python scripts/bc_brain_features.py --bc-dir outputs/bc2 --workers 4 \
        --conditions real
    python scripts/bc_brain_features.py --bc-dir outputs/bc3 --workers 4 \
        --encoding glomerular32 --states outputs/bc2/states.npz

Writes <bc-dir>/feats_real.npz, <bc-dir>/feats_shuffled.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BC = ROOT / "outputs" / "bc"
BC = DEFAULT_BC
PARTS = BC / "_parts"

# ---- worker ---------------------------------------------------------------
_W: dict = {}


def _init(condition: str, window_ms: float, n_features: int, brain_seed: int,
          shuffle_seed: int, fm_seed: int, npf: int, threshold: int,
          feature_version: int, encoding: str) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from flybalatro import connectome as C
    from flybalatro import glomerular as G
    from flybalatro.brain import Brain
    from flybalatro.encode import feature_map_for

    g = C.load(threshold)
    if condition not in ("real", "shuffled"):
        raise ValueError(condition)
    if encoding == G.ENCODING_GLOM32:
        spec = G.load_spec()
        fm = G.map_for(g, spec)
        brain = G.brain_for(g, condition, spec, shuffle_seed=shuffle_seed,
                            brain_seed=brain_seed)
    else:
        fm = feature_map_for(g, feature_version, seed=fm_seed, neurons_per_feature=npf)
        base = Brain(g, seed=brain_seed)
        brain = base if condition == "real" else base.shuffled(shuffle_seed)
        if condition == "shuffled":
            del base
    if fm.n_features != n_features:
        raise RuntimeError(
            f"{encoding} map wants {fm.n_features} features, the packed rows have "
            f"{n_features}; --feature-version / --encoding do not match the states"
        )
    brain.warmup()
    idx, _slices = G.population_indices(g, encoding)
    _W.update(brain=brain, fm=fm, idx=idx, window=float(window_ms))


def _run_chunk(job):
    """job = (chunk_id, packed_rows uint8[n, k], n_features). Returns (chunk_id, n, secs)."""
    chunk_id, packed, n_features = job
    brain, fm, idx, window = _W["brain"], _W["fm"], _W["idx"], _W["window"]
    parts = _W["parts"]
    path = parts / f"{_W['cond']}_{chunk_id:05d}.npy"
    bits = np.unpackbits(packed, axis=1)[:, :n_features]
    out = np.empty((len(bits), len(idx)), np.float16)
    t0 = time.perf_counter()
    for i in range(len(bits)):
        brain.reset()
        counts, _ = brain.step(fm.drive(bits[i]), window)
        out[i] = np.log1p(counts[idx].astype(np.float32)).astype(np.float16)
    tmp = parts / f"{_W['cond']}_{chunk_id:05d}.tmp.npy"
    np.save(tmp, out)
    tmp.replace(path)
    return chunk_id, len(bits), time.perf_counter() - t0


def _init_wrap(cond: str, parts_dir: str, *args) -> None:
    from pathlib import Path as _Path

    _W["cond"] = cond
    _W["parts"] = _Path(parts_dir)
    _init(cond, *args)


# ---- driver ---------------------------------------------------------------
def worker_rss_mb(pids) -> float:
    import subprocess
    if not pids:
        return 0.0
    try:
        out = subprocess.run(["ps", "-o", "rss=", *[f"-p{p}" for p in pids]],
                             capture_output=True, text=True, timeout=5).stdout
        return sum(int(x) for x in out.split()) / 1024.0
    except Exception:
        return 0.0


def run_condition(cond: str, uniq_packed: np.ndarray, n_features: int, a,
                  feature_version: int = 1, encoding: str = "featuremap") -> dict:
    import multiprocessing as mp

    from flybalatro import connectome as C
    from flybalatro import glomerular as G

    out_path = BC / f"feats_{cond}.npz"
    if out_path.exists() and not a.force:
        print(f"[{cond}] {out_path} exists; skipping.", flush=True)
        return {"skipped": True}

    PARTS.mkdir(parents=True, exist_ok=True)
    n = len(uniq_packed)
    bounds = list(range(0, n, a.chunk)) + [n]
    jobs = []
    done_rows = 0
    for c in range(len(bounds) - 1):
        p = PARTS / f"{cond}_{c:05d}.npy"
        if p.exists():
            done_rows += bounds[c + 1] - bounds[c]
            continue
        jobs.append((c, uniq_packed[bounds[c]:bounds[c + 1]], n_features))
    print(f"[{cond}] {n:,} unique states, {len(bounds)-1} chunks of {a.chunk}; "
          f"{done_rows:,} rows already on disk, {n-done_rows:,} to do", flush=True)

    t0 = time.perf_counter()
    rss_peak = 0.0
    if jobs:
        ctx = mp.get_context("spawn")
        init_args = (cond, str(PARTS), a.window, n_features, a.brain_seed,
                     a.shuffle_seed, a.fm_seed, a.neurons_per_feature, a.threshold,
                     feature_version, encoding)
        with ctx.Pool(a.workers, initializer=_init_wrap, initargs=init_args) as pool:
            pids = [p.pid for p in pool._pool]
            done = 0
            for k, (cid, cn, secs) in enumerate(pool.imap_unordered(_run_chunk, jobs), 1):
                done += cn
                el = time.perf_counter() - t0
                rate = done / el
                eta = (n - done_rows - done) / rate if rate > 0 else float("nan")
                if k <= 3 or k % 20 == 0 or k == len(jobs):
                    rss = worker_rss_mb(pids)
                    rss_peak = max(rss_peak, rss)
                    print(f"[{cond}] {done:,}/{n-done_rows:,} rows  {el/60:.1f} min  "
                          f"{rate:.1f} states/s  ETA {eta/60:.1f} min  "
                          f"workers RSS {rss:.0f} MB", flush=True)

    # assemble
    print(f"[{cond}] assembling ...", flush=True)
    total_secs = time.perf_counter() - t0
    g = C.load(a.threshold)
    pops = G.pops_for_encoding(encoding)
    sizes = [len({"alpn": g.alpn_indices, "kc": g.kc_indices, "mbon": g.mbon_indices,
                  "dn": g.dn_indices}[p]()) for p in pops]
    slices = G.population_slices(sizes, encoding)
    M = np.empty((n, sum(sizes)), np.float16)
    for c in range(len(bounds) - 1):
        part = np.load(PARTS / f"{cond}_{c:05d}.npy")
        if len(part) != bounds[c + 1] - bounds[c]:
            raise RuntimeError(f"part {c} has {len(part)} rows, expected "
                               f"{bounds[c+1]-bounds[c]}")
        if part.shape[1] != M.shape[1]:
            raise RuntimeError(f"part {c} is {part.shape[1]} wide, expected "
                               f"{M.shape[1]}; stale _parts from another encoding?")
        M[bounds[c]:bounds[c + 1]] = part
    spec_info = (G.load_spec().describe() if encoding == G.ENCODING_GLOM32 else None)
    meta = dict(condition=cond, encoding=encoding, window_ms=a.window,
                feature_version=feature_version,
                n_features=int(n_features), brain_seed=a.brain_seed,
                shuffle_seed=(a.shuffle_seed if cond == "shuffled" else None),
                fm_seed=a.fm_seed, neurons_per_feature=a.neurons_per_feature,
                threshold=a.threshold, n_unique=int(n), workers=a.workers,
                chunk=a.chunk, wall_seconds=round(total_secs, 1),
                states_per_second=round(n / total_secs, 2) if total_secs > 0 else None,
                worker_rss_peak_mb=round(rss_peak, 1),
                pops=list(pops),
                pop_sizes={p.upper(): int(w) for p, w in zip(pops, sizes)},
                glomerular_spec=spec_info,
                extended_with_sensory=(encoding != G.ENCODING_GLOM32))
    np.savez_compressed(
        out_path,
        **{p: M[:, slices[p]] for p in pops},
        uniq_packed=uniq_packed, meta=np.array(json.dumps(meta)),
    )
    del M
    print(f"[{cond}] wrote {out_path} ({out_path.stat().st_size/1e6:.0f} MB) "
          f"in {total_secs/60:.1f} min", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--window", type=float, default=50.0)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--max-states", type=int, default=0,
                    help="subsample whole episodes down to about this many states (0 = all)")
    ap.add_argument("--brain-seed", type=int, default=1)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--fm-seed", type=int, default=0)
    ap.add_argument("--neurons-per-feature", type=int, default=10)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--conditions", default="real,shuffled")
    ap.add_argument("--bc-dir", default=str(DEFAULT_BC))
    ap.add_argument("--states", default="")
    ap.add_argument("--feature-version", type=int, default=0,
                    help="0 = take it from states.npz")
    ap.add_argument("--encoding", default="featuremap",
                    choices=("featuremap", "glomerular32"),
                    help="glomerular32 drives 32 whole ORN glomeruli from the "
                         "relay block only, on the outputs/mb2 tuned brain")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    global BC, PARTS
    BC = Path(a.bc_dir)
    BC.mkdir(parents=True, exist_ok=True)
    PARTS = BC / "_parts"
    states_path = Path(a.states) if a.states else BC / "states.npz"

    from flybalatro import glomerular as G

    if a.encoding == G.ENCODING_GLOM32:
        spec = G.load_spec()
        a.window = spec.window_ms
        a.brain_seed = spec.brain_seed
        print(f"encoding={a.encoding} spec={spec.label} from {spec.source}\n"
              f"  tuning={spec.tuning.to_dict()} tonic {spec.drive_mv:g} mV "
              f"window {spec.window_ms:g} ms brain_seed {spec.brain_seed}", flush=True)

    z = np.load(states_path, allow_pickle=False)
    packed = z["features_packed"]
    episode = z["episode"]
    n_features = int(z["n_features"])
    stored_version = int(z["feature_version"]) if "feature_version" in z.files else 1
    feature_version = a.feature_version or stored_version
    if feature_version != stored_version:
        raise SystemExit(
            f"--feature-version {feature_version} but {states_path} was collected "
            f"with version {stored_version}"
        )
    print(f"{states_path}: {len(packed):,} states, {n_features} features, "
          f"feature_version={feature_version}", flush=True)

    keep = np.ones(len(packed), bool)
    if a.max_states and a.max_states < len(packed):
        eps = np.unique(episode)
        rng = np.random.default_rng(5)
        order = rng.permutation(eps)
        sizes = np.bincount(episode, minlength=eps.max() + 1)
        cum, chosen = 0, []
        for e in order:
            chosen.append(e)
            cum += sizes[e]
            if cum >= a.max_states:
                break
        keep = np.isin(episode, np.asarray(chosen))
        print(f"subsampled to {keep.sum():,} states from {len(chosen):,} episodes")

    # Under glomerular32 the brain only ever sees the 32 relay bits, so two states
    # that differ anywhere else are the *same* input and must not be simulated
    # twice. Re-pack to 32 bits and dedupe on that.
    n_input = G.n_input_bits(a.encoding, n_features)
    if n_input != n_features:
        bits_all = np.unpackbits(packed, axis=1)[:, :n_input]
        packed_in = np.packbits(bits_all, axis=1)
        del bits_all
        print(f"encoding {a.encoding}: the brain sees {n_input} of {n_features} bits "
              f"({n_features - n_input} not sent)", flush=True)
    else:
        packed_in = packed

    uniq_packed, inverse = np.unique(packed_in[keep], axis=0, return_inverse=True)
    inverse = inverse.astype(np.int32).ravel()
    idx_map = np.full(len(packed_in), -1, np.int32)
    idx_map[np.flatnonzero(keep)] = inverse
    np.savez_compressed(BC / "uniq_index.npz", uniq_packed=uniq_packed,
                        state_to_uniq=idx_map, kept=keep,
                        n_input_bits=np.int32(n_input),
                        encoding=np.array(a.encoding))
    print(f"{keep.sum():,} states -> {len(uniq_packed):,} unique input vectors "
          f"({1 - len(uniq_packed)/keep.sum():.1%} duplicates)", flush=True)
    est = len(uniq_packed) * 0.096 / max(a.workers, 1) / 60
    print(f"rough single-condition estimate at {a.window} ms, {a.workers} workers, "
          f"perfect scaling: {est:.0f} min", flush=True)

    metas = {}
    for cond in a.conditions.split(","):
        metas[cond] = run_condition(cond.strip(), uniq_packed, n_input, a,
                                    feature_version, a.encoding)
    (BC / "brain_features_meta.json").write_text(json.dumps(metas, indent=2, default=str) + "\n")
    print("done")


if __name__ == "__main__":
    main()
