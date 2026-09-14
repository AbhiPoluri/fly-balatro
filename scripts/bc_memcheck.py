"""Peak-RSS + throughput probe: how many brain worker processes fit in ~10 GB?

Loads the cached Graph, builds one real Brain and one shuffled Brain, runs a few
50 ms and 20 ms windows, and reports peak RSS at each stage plus wall time per
window. Run this in a *fresh* process; the numbers are used to pick the worker
count for scripts/bc_brain_features.py.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybalatro import connectome as C
from flybalatro.brain import Brain
from flybalatro.encode import FeatureMap


def rss_mb() -> float:
    # ru_maxrss is bytes on macOS, kB on Linux.
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024.0 * 1024.0) if sys.platform == "darwin" else v / 1024.0


def main() -> None:
    stages = {"import": round(rss_mb(), 1)}
    g = C.load(5)
    stages["graph_loaded"] = round(rss_mb(), 1)
    fm = FeatureMap.for_graph(g, 283, 10, seed=0)
    print("feature map:", fm.describe(), flush=True)

    real = Brain(g, seed=1)
    real.warmup()
    stages["real_brain"] = round(rss_mb(), 1)

    rng = np.random.default_rng(0)
    F = (rng.random((6, 283)) < 0.2).astype(np.int8)
    pops = {"ALPN": g.alpn_indices(), "KC": g.kc_indices(), "DN": g.dn_indices()}
    print("pop sizes:", {k: len(v) for k, v in pops.items()}, flush=True)

    timings = {}
    for w in (50.0, 20.0):
        t0 = time.perf_counter()
        for f in F:
            real.reset()
            real.step(fm.drive(f), w)
        timings[f"real_{int(w)}ms_per_window_s"] = round((time.perf_counter() - t0) / len(F), 4)
    stages["after_real_windows"] = round(rss_mb(), 1)

    shuf = Brain(g, seed=1).shuffled(seed=0)
    shuf.warmup()
    stages["plus_shuffled_brain"] = round(rss_mb(), 1)
    for w in (50.0, 20.0):
        t0 = time.perf_counter()
        for f in F:
            shuf.reset()
            shuf.step(fm.drive(f), w)
        timings[f"shuffled_{int(w)}ms_per_window_s"] = round((time.perf_counter() - t0) / len(F), 4)
    stages["peak"] = round(rss_mb(), 1)

    out = {"rss_mb_by_stage": stages, "timings": timings,
           "pop_sizes": {k: int(len(v)) for k, v in pops.items()},
           "feature_map": fm.describe()}
    print(json.dumps(out, indent=2))
    p = ROOT / "outputs" / "bc" / "memcheck.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
