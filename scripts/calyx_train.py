"""Behaviour-cloning readouts for the calyx conditions -- linear only.

Same recipe as ``scripts/bc_train.py`` (same states, same episode split, same
optimiser, same seeds: this module imports and calls that module's own
functions), restricted to the **linear** readout. The MLP is deliberately not
trained: ``outputs/bc3`` shows every MLP row within 0.017 of the no-brain control
because a 1x256 MLP saturates the 32-bit input's ceiling, so it cannot resolve
wiring and would only cost compute.

Two readouts per condition:

``kc``   the 4,064 Kenyon cells -- the population immediately downstream of the
         rewired synapses, and the one the control is about;
``dn``   the 1,314 descending neurons -- the fly's motor output, several synapses
         further on.

``alpn_kc_dn`` is available but is *not* the discriminating readout: the antennal
lobe is upstream of the calyx, is bit-identical in every condition, and already
decodes the 9-way hand type at 1.000, so a readout that can see it can be flat
across conditions by construction.

Beyond ``bc_train``'s aggregates this writes the **per-test-state** correctness
vector, so the imitation comparison between conditions can be paired over shared
test episodes rather than compared as two independent proportions.
"""

from __future__ import annotations

import argparse
import json
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

from scripts import bc_train as B  # noqa: E402
from scripts import calyx_common as K  # noqa: E402
from scripts import calyx_feats as F  # noqa: E402

OUT_DIR = K.OUT_DIR
READOUTS = {"kc": ("kc",), "dn": ("dn",), "alpn_kc_dn": ("alpn", "kc", "dn")}
EPOCHS, BATCH, LR, WD, SEED = 15, 512, 1e-3, 1e-4, 0


def splits_for(episode: np.ndarray, seed: int = SEED) -> Dict[str, np.ndarray]:
    """Byte-identical to ``bc_train.main``'s split: rng(seed + 1) over episodes."""
    eps = np.unique(episode)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(eps)
    n_test_ep = int(round(0.2 * len(eps)))
    test_eps = set(perm[:n_test_ep].tolist())
    rest = perm[n_test_ep:]
    n_val_ep = max(1, int(round(0.1 * len(rest))))
    val_eps = set(rest[:n_val_ep].tolist())
    which = np.zeros(len(episode), np.int8)
    which[np.isin(episode, list(val_eps))] = 1
    which[np.isin(episode, list(test_eps))] = 2
    return {"train": np.flatnonzero(which == 0), "val": np.flatnonzero(which == 1),
            "test": np.flatnonzero(which == 2)}


def per_state_correct(model, U, s2u, states, masks, labels, mean, std, keep,
                      device, batch: int = 8192) -> np.ndarray:
    """Masked-argmax correctness for every test state (the paired unit)."""
    import torch

    model.eval()
    out = np.zeros(len(states), bool)
    mean_t, std_t, keep_t = (torch.from_numpy(mean), torch.from_numpy(std),
                             torch.from_numpy(keep))
    with torch.no_grad():
        for i in range(0, len(states), batch):
            st = states[i:i + batch]
            X = torch.from_numpy(U[s2u[st]].astype(np.float32))
            X = ((X - mean_t) / std_t)[:, keep_t].to(device)
            m = torch.from_numpy(masks[st].astype(np.int8)).to(device)
            lg = model(X).masked_fill(m == 0, B.NEG)
            pred = lg.argmax(1).cpu().numpy()
            out[i:i + batch] = pred == labels[st]
    return out


def train_condition(variant: str, cond: str, readout: str, out_dir: Path = OUT_DIR,
                    force: bool = False) -> dict:
    import torch

    d = F.cond_dir(variant, cond, out_dir)
    models = d / "models"
    models.mkdir(parents=True, exist_ok=True)
    mpath = d / "train_metrics.json"
    metrics = json.loads(mpath.read_text()) if (mpath.exists() and not force) else {}
    key = f"{readout}/linear"
    if key in metrics and not force:
        print(f"[{variant}/{cond}/{readout}] present; skipping", flush=True)
        return metrics

    B.BC = d
    B.MODELS = models
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    S = B.load_states("glomerular32")
    labels, masks, episode = S["label"], S["masks"], S["episode"]
    splits = splits_for(episode)
    U, s2u = B.load_brain("real", READOUTS[readout])
    mean, std, keep = B.train_stats(U, s2u[splits["train"]])
    std = np.where(keep, std, 1.0).astype(np.float32)
    t0 = time.perf_counter()
    model, res = B.train_one("linear", U, s2u, splits, masks, labels, S["behaviour"],
                             mean, std, keep, device, EPOCHS, BATCH, LR, WD, SEED,
                             log=lambda *_a, **_k: None)
    res["train_seconds"] = round(time.perf_counter() - t0, 1)
    res["raw_dims"] = int(U.shape[1])
    res["effective_dims"] = int(keep.sum())
    correct = per_state_correct(model, U, s2u, splits["test"], masks, labels,
                                mean, std, keep, device)
    np.savez_compressed(d / f"test_correct_{readout}.npz",
                        state=splits["test"].astype(np.int32),
                        episode=episode[splits["test"]].astype(np.int64),
                        correct=correct)
    name = f"calyx_{cond}_{readout}"
    B.export_numpy(model, "linear", mean, std, keep, models / f"{name}_linear.npz",
                   feature_version=2, cond=name, encoding="glomerular32")
    metrics[key] = res
    metrics["_split"] = {k: int(len(v)) for k, v in splits.items()}
    metrics["_condition"] = cond
    metrics["_variant"] = variant
    mpath.write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    print(f"[{variant}/{cond}/{readout}] top1 {res['test']['top1']:.4f} "
          f"(d={int(keep.sum())}/{U.shape[1]}, {res['train_seconds']:.0f}s)",
          flush=True)
    del U
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="raw", choices=F.VARIANTS)
    ap.add_argument("--conditions", default="real,rw1,rw2,rw3")
    ap.add_argument("--readouts", default="kc,dn")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for cond in [c.strip() for c in a.conditions.split(",") if c.strip()]:
        for ro in [r.strip() for r in a.readouts.split(",") if r.strip()]:
            train_condition(a.variant, cond, ro, Path(a.out_dir), a.force)


if __name__ == "__main__":
    main()
