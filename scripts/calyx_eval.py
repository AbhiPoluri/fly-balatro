"""Live play for the calyx conditions, with per-episode records kept for pairing.

Identical to ``scripts/bc_eval.py`` in everything that matters: same env, same
400 episodes on seeds 100000-100399, same ante-1 cut, same masked-argmax policy,
same ``reset -> one 50 ms window -> log1p counts`` brain call, and this module
imports that module's worker functions rather than reimplementing them. Two
differences:

* the brain is built from a calyx condition's graph (``real`` or ``rw<seed>``)
  and that condition's own calibration variant;
* every episode's outcome is **kept**, not just the aggregate, so real and
  rewired can be compared paired over the shared episode seed (McNemar on the
  clear-rate, paired bootstrap on chips) instead of as two independent rates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent
for _p in (str(ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import calyx_common as K  # noqa: E402
from scripts import calyx_feats as F  # noqa: E402
from scripts import calyx_train as T  # noqa: E402

OUT_DIR = K.OUT_DIR
SEED0, N_EPISODES = 100_000, 400


def _init(variant: str, cond: str, readouts: Sequence[str], models_dir: str,
          tuning: dict, cache_cap: int, ante_end: int, max_steps: int) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"
    for p in (str(ROOT), str(_SCRIPT_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from baseline_heuristic_v2 import HandAwarePolicy

    from flybalatro import glomerular as G
    from flybalatro.brain import Brain
    from flybalatro.encode import GlomerularMap
    from flybalatro.env import BalatroEnv
    from flybalatro.features_v2 import encoder_for_version
    from flybalatro.tuning import Tuning
    from scripts import bc_eval as E

    spec = G.load_spec()
    g = K.graph_for(cond)
    brain = Brain(g, seed=spec.brain_seed, v_jitter=spec.v_jitter_mv,
                  tuning=Tuning.from_dict(tuning))
    brain.warmup()
    idx, slices = G.population_indices(g, G.ENCODING_GLOM32)
    E._W.update(
        models=Path(models_dir), version=2, encoding=G.ENCODING_GLOM32, n_input=32,
        references={"random", "teacher_v2"},
        env=BalatroEnv(ante_end=ante_end, max_steps=max_steps,
                       mask_noop_actions=True, encoder=encoder_for_version(2),
                       n_features=315),
        teacher=HandAwarePolicy(), cache={}, cache_cap=int(cache_cap),
        hits=0, misses=0, readouts={}, proj={},
        fm=G.map_for(g, spec), brain=brain, idx=idx, slices=slices,
        window=spec.window_ms)
    for name in readouts:
        ro = name.rsplit("_", 1)[-1]
        E.READOUT_SOURCE[name] = ("real", T.READOUTS[ro])
        E._W["readouts"][(name, "linear")] = E.Readout(
            Path(models_dir) / f"{name}_linear.npz")


def run(variant: str, cond: str, readouts: Sequence[str], workers: int = 2,
        episodes: int = N_EPISODES, seed0: int = SEED0, cache_cap: int = 6000,
        out_dir: Path = OUT_DIR, force: bool = False) -> dict:
    import multiprocessing as mp

    from scripts import bc_eval as E

    d = F.cond_dir(variant, cond, out_dir)
    outp = d / "eval.json"
    out = json.loads(outp.read_text()) if (outp.exists() and not force) else {}
    names = [f"calyx_{cond}_{ro}" for ro in readouts]
    todo = [n for n in names if n not in out]
    if not todo:
        print(f"[{variant}/{cond}] eval present; skipping", flush=True)
        return out
    tuning = F.tuning_dict(variant, cond, out_dir)
    seeds = list(range(seed0, seed0 + episodes))
    jobs = [(n, "linear", int(s), int(s) * 7919 + 13) for n in todo for s in seeds]
    print(f"[{variant}/{cond}] {len(todo)} readouts x {episodes} episodes "
          f"= {len(jobs)} jobs", flush=True)
    ctx = mp.get_context("spawn")
    args = (variant, cond, todo, str(d / "models"), tuning, cache_cap, 1, 200)
    by: Dict[str, List[dict]] = {}
    t0 = time.perf_counter()
    with ctx.Pool(max(1, workers), initializer=_init, initargs=args) as pool:
        for i, rec in enumerate(pool.imap_unordered(E._episode, jobs, chunksize=2), 1):
            by.setdefault(rec["name"], []).append(rec)
            if i % max(1, len(jobs) // 8) == 0:
                el = time.perf_counter() - t0
                print(f"[{variant}/{cond}] {i}/{len(jobs)}  {el/60:.1f} min  "
                      f"ETA {(len(jobs)-i)*el/i/60:.1f} min", flush=True)
    for name, recs in by.items():
        recs.sort(key=lambda r: r["seed"])
        out[name] = E.summarise(recs)
        np.savez_compressed(
            d / f"episodes_{name}.npz",
            seed=np.array([r["seed"] for r in recs], np.int64),
            win=np.array([r["win"] for r in recs], bool),
            chips=np.array([r["chips"] for r in recs], np.int64),
            steps=np.array([r["steps"] for r in recs], np.int64),
            plays=np.array([r["ev"]["plays"] for r in recs], np.int64),
            plays_best=np.array([r["ev"]["plays_best"] for r in recs], np.int64))
        s = out[name]
        print(f"  {name}: clear {s['clear_rate']:.3f} chips {s['chips_mean']:.0f} "
              f"best-subset {s['hand_evidence']['plays_that_were_the_best_subset']}",
              flush=True)
    out["_config"] = dict(variant=variant, condition=cond, episodes=episodes,
                          seed0=seed0, ante_end=1, max_steps=200, teacher="v2",
                          cache_cap=cache_cap, tuning_keys=sorted(tuning))
    outp.write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="raw", choices=F.VARIANTS)
    ap.add_argument("--conditions", default="real,rw1,rw2,rw3")
    ap.add_argument("--readouts", default="kc,dn")
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cache-cap", type=int, default=6000)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    ros = [r.strip() for r in a.readouts.split(",") if r.strip()]
    for cond in [c.strip() for c in a.conditions.split(",") if c.strip()]:
        run(a.variant, cond, ros, a.workers, a.episodes, SEED0, a.cache_cap,
            Path(a.out_dir), a.force)


if __name__ == "__main__":
    main()
