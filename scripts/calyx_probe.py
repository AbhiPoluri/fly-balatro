"""Activity matching + KC code statistics + linear decoding, per calyx condition.

Everything here reads the stored ``log1p`` spike counts written by
``scripts/calyx_feats.py`` (no simulation) and is measured on the **held-out**
half of ``scripts/kc_homeo.py``'s odour split (the 500 ``gate`` patterns), never
on the 500 ``calib`` patterns the per-KC thresholds were fitted to.

Three blocks, in the order the report needs them:

1. **Activity matching.** Mean firing rate per population, active fraction, the
   per-KC response-rate distribution (always-on ``r > 0.5``, the ``r > 0.3``
   tail, silent cells) and the number of non-constant readout channels. If two
   conditions do not match here, their downstream numbers are not comparable and
   the report has to say so instead of quoting them.
2. **KC code statistics.** Active-set Jaccard between hand types vs within hand
   type, same construction as ``scripts/mb2_grid.jaccard_between_within``.
3. **Decoding.** 9-way hand type from KC / KCbin / DN / ALPN / MBON, 5-fold CV
   grouped by relay pattern (``scripts.mb2_probe.probe_grouped``) plus the plain
   stratified number, against a permuted-label baseline and the count-only
   confound baseline (driven-ORN count, the one feature that carries label
   information by construction; ``outputs/mb2/REPORT.md`` measured 0.267 for it
   on its own sample).
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

from scripts import calyx_common as K  # noqa: E402
from scripts import calyx_feats as F  # noqa: E402
from scripts import kc_homeo as H  # noqa: E402
from scripts.brain_probe import probe  # noqa: E402
from scripts.mb2_grid import jaccard_between_within  # noqa: E402
from scripts.mb2_probe import probe_grouped  # noqa: E402
from scripts.mb2_sample import C_GRID_SHORT  # noqa: E402

OUT_DIR = K.OUT_DIR
BC3 = ROOT / "outputs" / "bc3"
POPS = ("alpn", "kc", "mbon", "dn")
WINDOW_MS = 50.0


# --------------------------------------------------------------------------- #
# pattern -> feature row
# --------------------------------------------------------------------------- #
def pattern_rows() -> dict:
    """Map ``kc_homeo``'s calib / gate patterns onto rows of the bc3 unique set."""
    uniq = np.load(BC3 / "uniq_index.npz", allow_pickle=False)["uniq_packed"]
    index = {uniq[i].tobytes(): i for i in range(len(uniq))}
    sets = H.relay_patterns()
    out = {}
    for name in ("calib", "gate"):
        pats = sets[name].astype(np.uint8)
        rows = []
        for p in pats:
            key = np.packbits(p).tobytes()
            j = index.get(key)
            if j is None:
                raise RuntimeError("a kc_homeo pattern is not in the bc3 unique set")
            rows.append(j)
        out[name] = np.asarray(rows, np.int64)
    bits = np.unpackbits(uniq, axis=1)[:, :32]
    out["uniq_bits"] = bits
    out["meta"] = sets["meta"]
    return out


def driven_orn_count(bits: np.ndarray) -> np.ndarray:
    """Total driven receptor neurons per pattern: the count-only confound."""
    from flybalatro import glomerular as G

    g = K.C.load(5)
    typ = g.type.astype(str)
    sizes = np.array([int((typ == t).sum()) for t in G.GLOM32_TYPES], np.float64)
    return bits.astype(np.float64) @ sizes


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #
def counts_from_feats(path: Path, rows: np.ndarray) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    out = {}
    for p in POPS:
        f = z[p][rows].astype(np.float32)
        out[p] = np.rint(np.expm1(f)).astype(np.int32)
    return out


def activity_stats(counts: Dict[str, np.ndarray]) -> dict:
    out = {}
    for p, M in counts.items():
        fired = M > 0
        r = fired.mean(0)
        out[p] = dict(
            n=int(M.shape[1]),
            mean_rate_hz=round(float(M.mean() / (WINDOW_MS / 1000.0)), 3),
            active_frac_per_state=round(float(fired.mean()), 5),
            active_per_state=round(float(fired.sum(1).mean()), 1),
            active_per_state_sd=round(float(fired.sum(1).std()), 1),
            n_non_constant=int((M.std(0) > 0).sum()),
            n_ever_active=int(fired.any(0).sum()),
            n_silent=int((r == 0.0).sum()),
            frac_r_gt_0_5=round(float((r > 0.5).mean()), 5),
            n_r_gt_0_5=int((r > 0.5).sum()),
            frac_r_gt_0_3=round(float((r > 0.3).mean()), 5),
            n_r_gt_0_3=int((r > 0.3).sum()),
            n_r_eq_1=int((r >= 1.0).sum()),
        )
    out["non_constant_channels_alpn_kc_dn"] = int(
        sum(out[p]["n_non_constant"] for p in ("alpn", "kc", "dn")))
    return out


def kc_code_stats(counts: Dict[str, np.ndarray], hand: np.ndarray) -> dict:
    active = (counts["kc"] > 0).astype(np.float32)
    return dict(jaccard=jaccard_between_within(active, hand))


def decode(counts: Dict[str, np.ndarray], bits: np.ndarray, hand: np.ndarray,
           pops: Sequence[str], perm_seed: int = 3) -> dict:
    rng = np.random.default_rng(perm_seed)
    perm = rng.permutation(hand)
    _, groups = np.unique(bits, axis=0, return_inverse=True)
    out: Dict[str, dict] = {}
    for name in pops:
        src = "kc" if name == "kcbin" else name
        M = counts[src]
        X = np.log1p(M.astype(np.float64))
        if name == "kcbin":
            X = (M > 0).astype(np.float64)
        t0 = time.perf_counter()
        r = probe(X, hand, c_grid=C_GRID_SHORT)
        rp = probe(X, perm, c_grid=C_GRID_SHORT)
        gr = probe_grouped(X, hand, groups, c_grid=C_GRID_SHORT)
        out[name] = dict(acc=r["best_acc"], acc_grouped=gr, permuted=rp["best_acc"],
                         margin=(round(r["best_acc"] - rp["best_acc"], 4)
                                 if r["best_acc"] is not None else None),
                         n_used_features=r["n_used_features"],
                         seconds=round(time.perf_counter() - t0, 1))
        print(f"      {name:6s} acc {r['best_acc']} grouped {gr} "
              f"perm {rp['best_acc']} ({out[name]['seconds']:.0f}s)", flush=True)
    return out


def baselines(bits: np.ndarray, hand: np.ndarray, perm_seed: int = 3) -> dict:
    rng = np.random.default_rng(perm_seed)
    perm = rng.permutation(hand)
    orn = driven_orn_count(bits)[:, None]
    bitcount = bits.sum(1, keepdims=True).astype(np.float64)
    return dict(
        majority=round(float(np.bincount(hand, minlength=9).max() / len(hand)), 4),
        relay_bits_32=probe(bits.astype(np.float64), hand, c_grid=C_GRID_SHORT)["best_acc"],
        driven_orn_count_only=probe(orn, hand, c_grid=C_GRID_SHORT)["best_acc"],
        bits_on_count_only=probe(bitcount, hand, c_grid=C_GRID_SHORT)["best_acc"],
        permuted_labels_on_bits=probe(bits.astype(np.float64), perm,
                                      c_grid=C_GRID_SHORT)["best_acc"],
        mb2_reference_driven_orn_count_only=0.267,
    )


# --------------------------------------------------------------------------- #
def run(variant: str, conditions: Sequence[str], out_dir: Path = OUT_DIR,
        pops: Sequence[str] = ("kc", "kcbin", "dn", "alpn", "mbon"),
        force: bool = False) -> dict:
    pr = pattern_rows()
    gate = pr["gate"]
    bits = pr["uniq_bits"][gate]
    hand = bits[:, :9].argmax(1)
    outp = out_dir / f"probe_{variant}.json"
    out = json.loads(outp.read_text()) if (outp.exists() and not force) else {}
    if "_baselines" not in out:
        out["_baselines"] = baselines(bits, hand)
        out["_set"] = dict(
            set="kc_homeo held-out gate patterns", n=int(len(gate)),
            source="outputs/bc2/states.npz relay block, deduplicated",
            split_seed=H.SPLIT_SEED, n_calib=H.N_CALIB, n_gate=H.N_GATE,
            hand_type_counts=np.bincount(hand, minlength=9).tolist(),
            calib_rows_in_bc3_uniq=pr["calib"].tolist(),
            gate_rows_in_bc3_uniq=gate.tolist(),
            filter=pr["meta"]["filter"])
        outp.write_text(json.dumps(out, indent=2, default=float) + "\n")
    for cond in conditions:
        if cond in out and not force:
            print(f"[{variant}/{cond}] present; skipping", flush=True)
            continue
        path = F.cond_dir(variant, cond, out_dir) / "feats_real.npz"
        if not path.exists():
            print(f"[{variant}/{cond}] no features yet; skipping", flush=True)
            continue
        print(f"[{variant}/{cond}]", flush=True)
        counts = counts_from_feats(path, gate)
        rec = dict(activity=activity_stats(counts),
                   kc_code=kc_code_stats(counts, hand))
        rec["decode_hand_type"] = decode(counts, bits, hand, pops)
        out[cond] = rec
        outp.write_text(json.dumps(out, indent=2, default=float) + "\n")
        del counts
    print(f"wrote {outp}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="raw", choices=F.VARIANTS)
    ap.add_argument("--conditions", default="real,rw1,rw2,rw3")
    ap.add_argument("--pops", default="kc,kcbin,dn,alpn,mbon")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    run(a.variant, [c.strip() for c in a.conditions.split(",") if c.strip()],
        Path(a.out_dir), [p.strip() for p in a.pops.split(",") if p.strip()],
        a.force)


if __name__ == "__main__":
    main()
