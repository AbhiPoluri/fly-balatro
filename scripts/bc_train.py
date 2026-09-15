"""Stage 3: train masked-cross-entropy readouts to imitate the heuristic expert.

Conditions (each trained with a linear readout and a 1x256 ReLU MLP, identical
optimiser recipe everywhere so the comparison is about the representation):

    real_alpn_kc_dn    6064-d log1p spike counts, MaleCNS wiring
    real_alpn          686-d, antennal-lobe projection neurons only (one synapse
                       from the driven receptor neurons, where the v1 probe still
                       found input identity)
    real_kc            4064-d, Kenyon cells only
    real_mbon          97-d, mushroom-body output neurons only
    real_dn            1314-d, descending neurons only
    shuf_alpn_kc_dn    6064-d, degree-preserving edge-target permutation
    shuf_dn            1314-d
    raw_bits            the input bits themselves, no brain at all (283 under
                        --feature-version 1, 315 under 2, and **32** under
                        --encoding glomerular32, where only the relay block is
                        sent to the fly, so the no-brain control sees exactly
                        what the fly saw)
    rand_proj           bits -> fixed Gaussian projection -> ReLU, 6000 units
    rand_proj_matched   same, width matched to the *effective* (non-constant)
                        width of real_alpn_kc_dn, because most of KC is silent

``--encoding`` defaults to whatever ``feats_real.npz`` was built with, so the
no-brain rows are automatically narrowed to the 32 relay bits in a glomerular
run. Every export carries both ``feature_version`` and ``encoding``.

Splits are by episode (never by state): consecutive states inside a blind are
near-duplicates, so a state-level split would leak. 80% of episodes train
(of which 10% are held back as a validation set for epoch selection), 20% test.

Loss: cross-entropy over the 109 logits with illegal actions filled to -1e9,
which is what the online policy in bc_eval.py argmaxes over.

``--bc-dir`` selects the run directory, so the v2 run lives in ``outputs/bc2``
alongside the v1 models rather than replacing them. The input width and the
``feature_version`` stamped into every export come from ``states.npz``, so the
viewer and the real-game player can tell which encoder a model needs.

Run:
    source .venv/bin/activate
    python scripts/bc_train.py
    python scripts/bc_train.py --bc-dir outputs/bc2 \
        --conditions real_alpn,real_alpn_kc_dn,real_dn,raw_bits,rand_proj_matched
    python scripts/bc_train.py --bc-dir outputs/bc3 \
        --conditions real_alpn_kc_dn,real_kc,real_mbon,real_dn,raw_bits,rand_proj_matched

Writes outputs/bc/models/*.pt, outputs/bc/models/*.npz (numpy export used by
bc_eval.py so the eval workers never import torch) and
outputs/bc/train_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BC = ROOT / "outputs" / "bc"
BC = DEFAULT_BC
MODELS = BC / "models"
N_ACTIONS = 109
NEG = -1e9

CONDITIONS = ("real_alpn_kc_dn", "real_alpn", "real_kc", "real_mbon", "real_dn",
              "shuf_alpn_kc_dn", "shuf_dn", "raw_bits", "rand_proj",
              "rand_proj_matched")

# condition -> (wiring or None, population blocks)
WIRING = {
    "real_alpn_kc_dn": ("real", ("alpn", "kc", "dn")),
    "real_alpn": ("real", ("alpn",)),
    "real_kc": ("real", ("kc",)),
    "real_mbon": ("real", ("mbon",)),
    "real_dn": ("real", ("dn",)),
    "shuf_alpn_kc_dn": ("shuffled", ("alpn", "kc", "dn")),
    "shuf_alpn": ("shuffled", ("alpn",)),
    "shuf_kc": ("shuffled", ("kc",)),
    "shuf_mbon": ("shuffled", ("mbon",)),
    "shuf_dn": ("shuffled", ("dn",)),
}


# ---- feature sources -------------------------------------------------------
def detect_encoding(bc: Path) -> str:
    """The encoding ``bc_brain_features`` used, from whichever feats file exists."""
    from flybalatro import glomerular as G

    for cond in ("real", "shuffled"):
        path = bc / f"feats_{cond}.npz"
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=False)
        if "meta" in z.files:
            return str(json.loads(str(z["meta"])).get("encoding", G.ENCODING_FEATUREMAP))
    idx = bc / "uniq_index.npz"
    if idx.exists():
        z = np.load(idx, allow_pickle=False)
        if "encoding" in z.files:
            return str(z["encoding"])
    return G.ENCODING_FEATUREMAP


def load_states(encoding: str = "featuremap"):
    """Bits, masks, labels. ``bits`` is narrowed to what the brain was given.

    Under ``glomerular32`` only the 32-bit relay block is sent to the fly, so the
    brain-free controls (``raw_bits``, ``rand_proj*``) must see those 32 bits and
    nothing else, or they are not comparable to the fly.
    """
    from flybalatro import glomerular as G

    z = np.load(BC / "states.npz", allow_pickle=False)
    n_features = int(z["n_features"])
    n_input = G.n_input_bits(encoding, n_features)
    bits = np.unpackbits(z["features_packed"], axis=1)[:, :n_input]
    masks = np.unpackbits(z["mask_packed"], axis=1)[:, : int(z["n_actions"])]
    version = int(z["feature_version"]) if "feature_version" in z.files else 1
    return dict(bits=bits.astype(np.uint8), masks=masks.astype(np.uint8),
                label=z["label"].astype(np.int64), episode=z["episode"],
                behaviour=z["behaviour"], feature_version=version,
                n_features=n_features, n_input_bits=n_input)


def load_brain(cond_wiring: str, pops: tuple[str, ...]):
    """Returns (unique_matrix float16[n_uniq, d], state_to_uniq int32[n_states])."""
    idx = np.load(BC / "uniq_index.npz", allow_pickle=False)
    s2u = idx["state_to_uniq"]
    z = np.load(BC / f"feats_{cond_wiring}.npz", allow_pickle=False)
    missing = [p for p in pops if p not in z.files]
    if missing:
        raise SystemExit(
            f"feats_{cond_wiring}.npz has no {missing} block(s); it stores "
            f"{sorted(k for k in z.files if k not in ('uniq_packed', 'meta'))}. "
            "Rerun bc_brain_features for this encoding."
        )
    parts = [z[p] for p in pops]
    if len(parts) == 1:
        M = np.ascontiguousarray(parts[0])
    else:
        n = len(parts[0])
        d = sum(p.shape[1] for p in parts)
        M = np.empty((n, d), np.float16)
        o = 0
        for p in parts:
            M[:, o:o + p.shape[1]] = p
            o += p.shape[1]
            del p
    if (s2u < 0).any():
        raise RuntimeError("uniq_index.npz has unmapped states; rerun bc_brain_features")
    return M, s2u.astype(np.int64)


def rand_proj_matrix(width: int, n_in: int, seed: int = 12345):
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, width))).astype(np.float32)


def apply_rand_proj(bits: np.ndarray, W: np.ndarray, chunk: int = 20000) -> np.ndarray:
    out = np.empty((len(bits), W.shape[1]), np.float16)
    for i in range(0, len(bits), chunk):
        h = bits[i:i + chunk].astype(np.float32) @ W
        np.maximum(h, 0.0, out=h)
        out[i:i + chunk] = h.astype(np.float16)
    return out


# ---- stats / normalisation -------------------------------------------------
def train_stats(M: np.ndarray, rows: np.ndarray, chunk: int = 20000):
    """Exact mean/std over the training rows of M (gathered in chunks)."""
    d = M.shape[1]
    s = np.zeros(d, np.float64)
    ss = np.zeros(d, np.float64)
    n = 0
    for i in range(0, len(rows), chunk):
        X = M[rows[i:i + chunk]].astype(np.float32)
        s += X.sum(0, dtype=np.float64)
        ss += (X.astype(np.float64) ** 2).sum(0)
        n += len(X)
    mean = s / n
    var = np.maximum(ss / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    keep = std > 1e-6
    return mean.astype(np.float32), std.astype(np.float32), keep


# ---- models ----------------------------------------------------------------
class Linear(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, N_ACTIONS)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, N_ACTIONS))

    def forward(self, x):
        return self.net(x)


def masked_metrics(logits: torch.Tensor, mask: torch.Tensor, y: torch.Tensor):
    masked = logits.masked_fill(mask == 0, NEG)
    pred_m = masked.argmax(1)
    pred_u = logits.argmax(1)
    legal_u = mask.gather(1, pred_u[:, None]).squeeze(1) > 0
    n_legal = mask.sum(1)
    nontriv = n_legal > 1
    return dict(
        correct_masked=int((pred_m == y).sum()),
        correct_unmasked=int((pred_u == y).sum()),
        legal_unmasked=int(legal_u.sum()),
        n=int(len(y)),
        correct_masked_nontrivial=int(((pred_m == y) & nontriv).sum()),
        n_nontrivial=int(nontriv.sum()),
    )


def evaluate(model, U, s2u, states, masks, labels, mean, std, keep, device,
             behaviour=None, batch=4096):
    """``states`` indexes masks/labels/behaviour; ``s2u[states]`` indexes U."""
    model.eval()
    agg = None
    per_beh = {}
    with torch.no_grad():
        for i in range(0, len(states), batch):
            st = states[i:i + batch]
            X = torch.from_numpy(U[s2u[st]].astype(np.float32))
            X = ((X - mean) / std)[:, keep].to(device)
            m = torch.from_numpy(masks[st].astype(np.int8)).to(device)
            y = torch.from_numpy(labels[st]).to(device)
            out = masked_metrics(model(X), m, y)
            agg = out if agg is None else {k: agg[k] + v for k, v in out.items()}
            if behaviour is not None:
                b = behaviour[st]
                for bv in (0, 1):
                    sel = np.flatnonzero(b == bv)
                    if len(sel) == 0:
                        continue
                    t = torch.from_numpy(sel).to(device)
                    o = masked_metrics(model(X[t]), m[t], y[t])
                    prev = per_beh.get(bv)
                    per_beh[bv] = o if prev is None else {k: prev[k] + v for k, v in o.items()}
    res = dict(
        top1=agg["correct_masked"] / agg["n"],
        top1_nontrivial=(agg["correct_masked_nontrivial"] / agg["n_nontrivial"]
                         if agg["n_nontrivial"] else None),
        top1_unmasked=agg["correct_unmasked"] / agg["n"],
        unmasked_argmax_legal_rate=agg["legal_unmasked"] / agg["n"],
        n=agg["n"], n_nontrivial=agg["n_nontrivial"],
    )
    for bv, o in per_beh.items():
        res[f"top1_behaviour_{'heuristic' if bv == 0 else 'random'}"] = (
            o["correct_masked"] / o["n"])
    return res


def train_one(kind, U, s2u, splits, masks, labels, behaviour, mean, std, keep, device,
              epochs, batch, lr, wd, seed, log=print):
    torch.manual_seed(seed)
    d = int(keep.sum())
    model = (Linear(d) if kind == "linear" else MLP(d)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()
    mean_t, std_t = torch.from_numpy(mean), torch.from_numpy(std)
    keep_t = torch.from_numpy(keep)

    tr, va, te = splits["train"], splits["val"], splits["test"]
    rng = np.random.default_rng(seed)
    best = (-1.0, None, -1)
    hist = []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr))
        tot = 0.0
        nb = 0
        for i in range(0, len(order), batch):
            st = tr[order[i:i + batch]]
            X = torch.from_numpy(U[s2u[st]].astype(np.float32))
            X = ((X - mean_t) / std_t)[:, keep_t].to(device)
            m = torch.from_numpy(masks[st].astype(np.int8)).to(device)
            y = torch.from_numpy(labels[st]).to(device)
            logits = model(X).masked_fill(m == 0, NEG)
            loss = lossf(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        vm = evaluate(model, U, s2u, va, masks, labels, mean_t, std_t, keep_t, device)
        hist.append(dict(epoch=ep, train_loss=round(tot / max(nb, 1), 4),
                         val_top1=round(vm["top1"], 4)))
        if vm["top1"] > best[0]:
            best = (vm["top1"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
        log(f"      ep{ep:02d} loss {tot/max(nb,1):.4f} val {vm['top1']:.4f}")
    model.load_state_dict(best[1])
    test = evaluate(model, U, s2u, te, masks, labels, mean_t, std_t, keep_t, device,
                    behaviour=behaviour)
    return model, dict(test=test, val_best=best[0], best_epoch=best[2], history=hist,
                       n_input_dims=d, n_params=sum(p.numel() for p in model.parameters()))


def export_numpy(model, kind, mean, std, keep, path: Path, feature_version: int = 1,
                 cond: str = "", encoding: str = "featuremap"):
    """Numpy export read by bc_eval, the viewer and the real-game player.

    ``feature_version`` tells those loaders which state encoder to build and
    ``encoding`` tells them which input path to drive the brain through; missing
    fields mean 1 and ``featuremap``, so v1/v2 exports stay readable unchanged.
    """
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    out = dict(kind=np.array(kind), mean=mean, std=std, keep=keep.astype(np.bool_),
               feature_version=np.int32(feature_version), cond=np.array(cond),
               encoding=np.array(encoding))
    if kind == "linear":
        out.update(W=sd["fc.weight"], b=sd["fc.bias"])
    else:
        out.update(W1=sd["net.0.weight"], b1=sd["net.0.bias"],
                   W2=sd["net.2.weight"], b2=sd["net.2.bias"])
    np.savez(path, **out)


# ---- prior baselines -------------------------------------------------------
def prior_baselines(labels, masks, splits):
    tr, te = splits["train"], splits["test"]
    cnt = np.bincount(labels[tr], minlength=N_ACTIONS).astype(np.float64)
    score = np.log(cnt + 1e-9)
    ml = masks[te].astype(bool)
    pred = np.where(ml, score[None, :], -np.inf).argmax(1)
    y = labels[te]
    n_legal = masks[te].sum(1)
    nt = n_legal > 1
    return dict(
        most_frequent_legal_label_top1=float((pred == y).mean()),
        most_frequent_legal_label_top1_nontrivial=float((pred == y)[nt].mean()),
        majority_label_top1=float((y == int(cnt.argmax())).mean()),
        trivial_state_fraction=float((~nt).mean()),
        uniform_random_legal_top1=float(np.mean(1.0 / n_legal)),
        n_test=int(len(y)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rp-width", type=int, default=6000)
    ap.add_argument("--bc-dir", default=str(DEFAULT_BC))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--encoding", default="auto",
                    choices=("auto", "featuremap", "glomerular32"),
                    help="auto = read it off feats_*.npz")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    global BC, MODELS
    BC = Path(a.bc_dir)
    MODELS = BC / "models"
    MODELS.mkdir(parents=True, exist_ok=True)
    metrics_path = BC / "train_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if (metrics_path.exists() and not a.force) else {}

    if a.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = a.device
    print(f"device = {device}", flush=True)

    encoding = detect_encoding(BC) if a.encoding == "auto" else a.encoding
    S = load_states(encoding)
    labels, masks, episode, behaviour = S["label"], S["masks"], S["episode"], S["behaviour"]
    n_states = len(labels)
    feature_version = int(S["feature_version"])
    n_bits = int(S["bits"].shape[1])
    print(f"states: {n_states:,}  feature_version={feature_version}  "
          f"encoding={encoding}  bits into the readout={n_bits} "
          f"(of {S['n_features']} encoded)", flush=True)
    metrics["_feature_version"] = feature_version
    metrics["_encoding"] = encoding
    metrics["_n_bits"] = n_bits
    metrics["_n_features_encoded"] = int(S["n_features"])

    eps = np.unique(episode)
    rng = np.random.default_rng(a.seed + 1)
    perm = rng.permutation(eps)
    n_test_ep = int(round(0.2 * len(eps)))
    test_eps = set(perm[:n_test_ep].tolist())
    rest = perm[n_test_ep:]
    n_val_ep = max(1, int(round(0.1 * len(rest))))
    val_eps = set(rest[:n_val_ep].tolist())
    which = np.zeros(n_states, np.int8)  # 0 train, 1 val, 2 test
    which[np.isin(episode, list(val_eps))] = 1
    which[np.isin(episode, list(test_eps))] = 2
    splits = {"train": np.flatnonzero(which == 0), "val": np.flatnonzero(which == 1),
              "test": np.flatnonzero(which == 2)}
    split_info = {k: int(len(v)) for k, v in splits.items()}
    split_info["episodes"] = dict(total=int(len(eps)), test=len(test_eps), val=len(val_eps))
    for k, v in splits.items():
        b = behaviour[v]
        split_info[f"{k}_heuristic_frac"] = round(float((b == 0).mean()), 4)
    print("split:", json.dumps(split_info), flush=True)
    metrics["_split"] = split_info
    metrics["_priors"] = prior_baselines(labels, masks, splits)
    print("priors:", json.dumps(metrics["_priors"], indent=2), flush=True)

    # effective width of the real full condition, for the matched control
    eff_width = None
    if (BC / "feats_real.npz").exists():
        U, s2u = load_brain("real", ("alpn", "kc", "dn"))
        m_, s_, k_ = train_stats(U, s2u[splits["train"]])
        eff_width = int(k_.sum())
        print(f"real_alpn_kc_dn effective (non-constant) width: {eff_width}/{U.shape[1]}", flush=True)
        del U, m_, s_, k_
        metrics["_effective_width_real_alpn_kc_dn"] = eff_width

    def source(cond):
        if cond in WIRING:
            wiring, pops = WIRING[cond]
            U, s2u = load_brain(wiring, pops)
        elif cond == "raw_bits":
            U, s2u = S["bits"].astype(np.float16), np.arange(n_states)
        elif cond in ("rand_proj", "rand_proj_matched"):
            w = a.rp_width if cond == "rand_proj" else (eff_width or a.rp_width)
            W = rand_proj_matrix(w, n_bits)
            U, s2u = apply_rand_proj(S["bits"], W), np.arange(n_states)
            np.savez_compressed(MODELS / f"{cond}_projection.npz", W=W)
        else:
            raise ValueError(cond)
        return U, s2u

    for cond in a.conditions.split(","):
        cond = cond.strip()
        if all(f"{cond}/{k}" in metrics for k in ("linear", "mlp")) and not a.force:
            print(f"[{cond}] already in train_metrics.json; skipping", flush=True)
            continue
        need = (f"feats_{WIRING[cond][0]}.npz" if cond in WIRING else None)
        if need and not (BC / need).exists():
            print(f"[{cond}] {need} missing; skipping", flush=True)
            continue
        t0 = time.perf_counter()
        U, s2u = source(cond)
        mean, std, keep = train_stats(U, s2u[splits["train"]])
        std = np.where(keep, std, 1.0).astype(np.float32)
        print(f"[{cond}] d={U.shape[1]} effective={int(keep.sum())} "
              f"({time.perf_counter()-t0:.0f}s to build)", flush=True)
        for kind in ("linear", "mlp"):
            print(f"    {cond} / {kind}", flush=True)
            t1 = time.perf_counter()
            model, res = train_one(kind, U, s2u, splits, masks, labels, behaviour,
                                   mean, std, keep, device, a.epochs, a.batch, a.lr,
                                   a.weight_decay, a.seed)
            res["train_seconds"] = round(time.perf_counter() - t1, 1)
            res["raw_dims"] = int(U.shape[1])
            torch.save({"state_dict": model.state_dict(), "kind": kind, "cond": cond,
                        "keep": keep, "mean": mean, "std": std,
                        "feature_version": feature_version, "encoding": encoding},
                       MODELS / f"{cond}_{kind}.pt")
            export_numpy(model, kind, mean, std, keep, MODELS / f"{cond}_{kind}.npz",
                         feature_version=feature_version, cond=cond,
                         encoding=encoding)
            metrics[f"{cond}/{kind}"] = res
            print(f"      test top1 {res['test']['top1']:.4f}  "
                  f"nontrivial {res['test']['top1_nontrivial']}  "
                  f"({res['train_seconds']:.0f}s)", flush=True)
            metrics_path.write_text(json.dumps(metrics, indent=2, default=float) + "\n")
        del U
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
