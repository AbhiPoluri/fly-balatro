"""Smoke tests for the behaviour-cloning pipeline (collect -> train -> eval).

Deliberately tiny: they check wiring and shapes, plus that a readout trained on
2k states actually learns something, and that the online eval loop closes with a
real brain in it. They are not a substitute for the full runs in outputs/bc/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from flybalatro.env import N_ACTIONS  # noqa: E402
from flybalatro.features import N_FEATURES  # noqa: E402

import bc_collect  # noqa: E402
import bc_eval  # noqa: E402


# ---- stage 1 ---------------------------------------------------------------
def test_collect_shapes_and_labels():
    data, meta = bc_collect.collect(target_states=400, seed0_heur=0, seed0_rand=500_000,
                                    max_steps=500, ante_end=1, rng_seed=3, verbose=False)
    n = len(data["label"])
    assert n >= 400
    assert meta["n_states"] == n
    bits = np.unpackbits(data["features_packed"], axis=1)[:, :N_FEATURES]
    masks = np.unpackbits(data["mask_packed"], axis=1)[:, :N_ACTIONS]
    assert bits.shape == (n, N_FEATURES)
    assert masks.shape == (n, N_ACTIONS)
    assert set(np.unique(bits)) <= {0, 1}
    for key in ("label", "episode", "behaviour", "step_in_episode", "stage"):
        assert len(data[key]) == n, key
    # every stored expert label is legal in its own state
    assert masks[np.arange(n), data["label"]].all()
    # at least one legal action everywhere, and no no-op actions are legal
    assert (masks.sum(1) > 0).all()
    from flybalatro.env import NOOP_ACTION_INDICES
    assert masks[:, list(NOOP_ACTION_INDICES)].sum() == 0
    # both behaviours present, and heuristic-driven episodes take expert actions
    assert set(np.unique(data["behaviour"])) == {0, 1}


def test_collect_is_deterministic():
    a, _ = bc_collect.collect(200, 0, 500_000, 500, 1, 3, verbose=False)
    b, _ = bc_collect.collect(200, 0, 500_000, 500, 1, 3, verbose=False)
    assert np.array_equal(a["features_packed"], b["features_packed"])
    assert np.array_equal(a["label"], b["label"])


# ---- stage 3 ---------------------------------------------------------------
def test_readout_beats_chance_on_heldout():
    """A linear masked-CE readout on 2k states must beat both chance floors."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    data, _ = bc_collect.collect(2000, 0, 500_000, 500, 1, 3, verbose=False)
    bits = np.unpackbits(data["features_packed"], axis=1)[:, :N_FEATURES].astype(np.float32)
    masks = np.unpackbits(data["mask_packed"], axis=1)[:, :N_ACTIONS].astype(np.int8)
    y = data["label"].astype(np.int64)
    ep = data["episode"]

    eps = np.unique(ep)
    rng = np.random.default_rng(0)
    test_eps = rng.permutation(eps)[: max(1, len(eps) // 5)]
    te = np.isin(ep, test_eps)
    tr = ~te
    assert tr.sum() > 100 and te.sum() > 50

    mu, sd = bits[tr].mean(0), bits[tr].std(0) + 1e-6
    X = torch.from_numpy((bits - mu) / sd)
    M = torch.from_numpy(masks)
    Y = torch.from_numpy(y)
    itr = torch.from_numpy(np.flatnonzero(tr))
    ite = torch.from_numpy(np.flatnonzero(te))

    torch.manual_seed(0)
    model = nn.Linear(N_FEATURES, N_ACTIONS)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(150):
        perm = itr[torch.randperm(len(itr))]
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            lg = model(X[b]).masked_fill(M[b] == 0, -1e9)
            loss = lossf(lg, Y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(X[ite]).masked_fill(M[ite] == 0, -1e9).argmax(1)
    acc = float((pred == Y[ite]).float().mean())

    n_legal = masks[te].sum(1)
    uniform_chance = float(np.mean(1.0 / n_legal))
    counts = np.bincount(y[tr], minlength=N_ACTIONS).astype(np.float64)
    prior_pred = np.where(masks[te].astype(bool), np.log(counts + 1e-9)[None, :],
                          -np.inf).argmax(1)
    prior_acc = float((prior_pred == y[te]).mean())

    assert acc > uniform_chance + 0.10, (acc, uniform_chance)
    assert acc > prior_acc, (acc, prior_acc)


# ---- stage 4 ---------------------------------------------------------------
def _fake_readout(tmp_path: Path, d: int, kind: str = "linear") -> Path:
    rng = np.random.default_rng(0)
    p = tmp_path / f"fake_{kind}.npz"
    payload = dict(kind=np.array(kind), mean=np.zeros(d, np.float32),
                   std=np.ones(d, np.float32), keep=np.ones(d, bool))
    if kind == "linear":
        payload.update(W=rng.normal(0, 0.1, (N_ACTIONS, d)).astype(np.float32),
                       b=np.zeros(N_ACTIONS, np.float32))
    else:
        payload.update(W1=rng.normal(0, 0.1, (16, d)).astype(np.float32),
                       b1=np.zeros(16, np.float32),
                       W2=rng.normal(0, 0.1, (N_ACTIONS, 16)).astype(np.float32),
                       b2=np.zeros(N_ACTIONS, np.float32))
    np.savez(p, **payload)
    return p


def _setup_worker(readouts: dict, brain=None, encoding=None, n_input=N_FEATURES):
    """The worker globals ``bc_eval._init`` would have set, for a v1 run.

    ``encoding`` / ``n_input`` are what narrow the state to the bits the brain is
    actually given; under ``featuremap`` that is all of them.
    """
    from flybalatro.env import BalatroEnv
    from flybalatro.glomerular import ENCODING_FEATUREMAP
    from baseline_heuristic import HeuristicPolicy
    bc_eval._W.clear()
    bc_eval._W.update(env=BalatroEnv(ante_end=1, max_steps=200, mask_noop_actions=True),
                      teacher=HeuristicPolicy(), cache={}, cache_cap=5000,
                      hits=0, misses=0, readouts=readouts, proj={}, brain=brain,
                      version=1, references={"random", "heuristic"},
                      encoding=encoding or ENCODING_FEATUREMAP,
                      n_input=int(n_input))


def test_eval_two_episodes_references_and_readout(tmp_path):
    ro = bc_eval.Readout(_fake_readout(tmp_path, N_FEATURES, "mlp"))
    _setup_worker({("raw_bits", "mlp"): ro})
    recs = [bc_eval._episode(("heuristic", "-", 100_000 + i, 1)) for i in range(2)]
    recs += [bc_eval._episode(("random", "-", 100_000 + i, 5)) for i in range(2)]
    recs += [bc_eval._episode(("raw_bits", "mlp", 100_000 + i, 1)) for i in range(2)]
    for r in recs:
        assert r["steps"] >= 1
        assert r["hist"].sum() == r["steps"]
        assert r["chips"] >= 0
    s = bc_eval.summarise(recs[:2])
    assert s["n_episodes"] == 2
    assert 0.0 <= s["clear_rate"] <= 1.0
    assert s["total_steps"] == sum(r["steps"] for r in recs[:2])


@pytest.mark.parametrize("wiring", ["real"])
def test_eval_two_episodes_with_brain_in_the_loop(tmp_path, wiring):
    """The expensive one: a live DN-only readout driving 2 real episodes (20 ms window)."""
    from flybalatro import connectome as C
    from flybalatro.brain import Brain
    from flybalatro.encode import FeatureMap

    if not C.cache_path(5).exists():
        pytest.skip("connectome cache missing")
    g = C.load(5)
    brain = Brain(g, seed=1)
    brain.warmup()
    n_alpn, n_kc, n_dn = len(g.alpn_indices()), len(g.kc_indices()), len(g.dn_indices())
    ro = bc_eval.Readout(_fake_readout(tmp_path, n_dn, "linear"))
    _setup_worker({("real_dn", "linear"): ro}, brain=brain)
    bc_eval._W.update(
        fm=FeatureMap.for_graph(g, N_FEATURES, 10, seed=0), window=20.0,
        idx=np.concatenate([g.alpn_indices(), g.kc_indices(), g.dn_indices()]).astype(np.int32),
        slices={"alpn": slice(0, n_alpn), "kc": slice(n_alpn, n_alpn + n_kc),
                "dn": slice(n_alpn + n_kc, n_alpn + n_kc + n_dn)},
    )
    recs = [bc_eval._episode(("real_dn", "linear", 100_000 + i, 1)) for i in range(2)]
    for r in recs:
        assert r["steps"] >= 1
        assert r["hist"].sum() == r["steps"]
    assert bc_eval._W["misses"] > 0
    s = bc_eval.summarise(recs)
    assert s["cache_hit_rate"] is not None and 0.0 <= s["cache_hit_rate"] <= 1.0


def test_stored_brain_features_match_the_live_loop():
    """The online eval must see exactly the features the readout was trained on."""
    import bc_consistency_check
    for cond in ("real", "shuffled"):
        if not (ROOT / "outputs" / "bc" / f"feats_{cond}.npz").exists():
            pytest.skip(f"feats_{cond}.npz not built yet")
        assert bc_consistency_check.check(cond, n_rows=2) == 0
