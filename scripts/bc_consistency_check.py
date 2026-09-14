"""Assert the live brain in bc_eval reproduces the stored bc_brain_features rows.

If this fails the online policy is being fed out-of-distribution inputs and the
eval table means nothing, so the pipeline runs it before training each wiring.

The input path comes from the same shared constructors every other stage uses,
keyed by the ``encoding`` and ``feature_version`` stored in the feature file:
``flybalatro.encode.feature_map_for`` for v1/v2, ``flybalatro.glomerular`` for
``glomerular32`` (which also means a tuned brain, ``apl_scale = 2``). That is the
whole point of the check: a v2 map laid out ``orn_first`` drives different
neurons than a v1 map, a glomerular map drives different neurons again, an
untuned brain answers differently from a tuned one, and nothing else in the
pipeline would notice.

Run:
    python scripts/bc_consistency_check.py real
    python scripts/bc_consistency_check.py real --bc-dir outputs/bc2
    python scripts/bc_consistency_check.py real --bc-dir outputs/bc3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from flybalatro import connectome as C
from flybalatro import glomerular as G
from flybalatro.brain import Brain
from flybalatro.encode import feature_map_for

DEFAULT_BC = ROOT / "outputs" / "bc"


def check(cond: str, bc: Path = DEFAULT_BC, n_rows: int = 5, window_ms: float = 50.0,
          brain_seed: int = 1, shuffle_seed: int = 0, fm_seed: int = 0,
          npf: int = 10, feature_version: int = 0) -> int:
    z = np.load(bc / f"feats_{cond}.npz", allow_pickle=False)
    uniq = z["uniq_packed"]
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    encoding = str(meta.get("encoding", G.ENCODING_FEATUREMAP))
    pops = G.pops_for_encoding(encoding)
    stored = np.concatenate([z[p] for p in pops], axis=1)
    version = feature_version or int(meta.get("feature_version", 1))
    window_ms = float(meta.get("window_ms", window_ms))
    g = C.load(5)
    if encoding == G.ENCODING_GLOM32:
        spec = G.load_spec()
        fm = G.map_for(g, spec)
        brain = G.brain_for(g, cond, spec, shuffle_seed=shuffle_seed)
        window_ms = float(meta.get("window_ms", spec.window_ms))
        extra = f"glomeruli={fm.n_features} tuning={spec.tuning.label()}"
    else:
        fm = feature_map_for(g, version, seed=fm_seed, neurons_per_feature=npf)
        base = Brain(g, seed=brain_seed)
        brain = base if cond == "real" else base.shuffled(shuffle_seed)
        extra = f"orn_first={fm.orn_first}"
    n_features = fm.n_features
    brain.warmup()
    idx, _ = G.population_indices(g, encoding)
    rows = np.linspace(0, len(uniq) - 1, n_rows).astype(int)
    print(f"[{cond}] encoding={encoding} feature_version={version} "
          f"n_features={n_features} window={window_ms:g}ms {extra} "
          f"pops={'+'.join(p.upper() for p in pops)}")
    bad = 0
    for r in rows:
        bits = np.unpackbits(uniq[r : r + 1], axis=1)[0, :n_features]
        brain.reset()
        counts, _ = brain.step(fm.drive(bits), window_ms)
        live = np.log1p(counts[idx].astype(np.float32)).astype(np.float16)
        ok = np.array_equal(live, stored[r])
        if not ok:
            bad += 1
            d = np.abs(live.astype(np.float32) - stored[r].astype(np.float32))
            print(f"  row {r}: MISMATCH  max|d|={d.max():.4f}  n_diff={(d>0).sum()}")
        else:
            print(f"  row {r}: ok  ({int((live>0).sum())} active units)")
    print(f"[{cond}] {n_rows - bad}/{n_rows} rows reproduce exactly")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("condition", nargs="?", default="real", choices=("real", "shuffled"))
    ap.add_argument("--bc-dir", default=str(DEFAULT_BC))
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--feature-version", type=int, default=0)
    a = ap.parse_args()
    sys.exit(
        1
        if check(a.condition, Path(a.bc_dir), n_rows=a.rows,
                 feature_version=a.feature_version)
        else 0
    )
