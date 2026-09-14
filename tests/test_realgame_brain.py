"""The brain half of the real-game path, on a small synthetic connectome.

``BrainRunner`` is exercised with a real ``FeatureMap`` and a real spiking
``Brain``, just over a ~4k-neuron synthetic graph instead of MaleCNS, so the
drive -> spike -> population-rate -> readout-feature chain is genuinely run
without a multi-gigabyte load. The MaleCNS path differs only in which graph is
passed in.

Also checks that :class:`flybalatro.realgame.play.Readout` can consume the npz
files ``scripts/bc_train.py`` actually exported, and that ``select_policy``
reports honestly which mode it picked.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from flybalatro.connectome import Graph
from flybalatro.features import N_FEATURES
from flybalatro.realgame.play import (
    MODELS,
    BrainRunner,
    HeuristicThroughBrainPolicy,
    Readout,
    ReadoutPolicy,
    select_policy,
)

# 283 features x 10 neurons each needs 2830 drivable sensory neurons.
N_ORN = 3000
N_ALPN = 60
N_KC = 200
N_MBON = 20
N_DN = 40
N_TOTAL = N_ORN + N_ALPN + N_KC + N_MBON + N_DN


def synthetic_graph(seed: int = 0) -> Graph:
    """A tiny feed-forward stand-in: ORN -> ALPN -> KC -> MBON -> DN."""
    rng = np.random.default_rng(seed)

    cls: List[str] = (["olfactory"] * N_ORN + ["ALPN"] * N_ALPN
                      + ["Kenyon_Cell"] * N_KC + ["MBON"] * N_MBON
                      + ["other"] * N_DN)
    superclass: List[str] = (["cb_sensory"] * N_ORN + ["central_brain"] * N_ALPN
                            + ["central_brain"] * N_KC + ["central_brain"] * N_MBON
                            + ["descending_neuron"] * N_DN)

    orn = np.arange(N_ORN)
    alpn = np.arange(N_ORN, N_ORN + N_ALPN)
    kc = np.arange(N_ORN + N_ALPN, N_ORN + N_ALPN + N_KC)
    mbon = np.arange(N_ORN + N_ALPN + N_KC, N_ORN + N_ALPN + N_KC + N_MBON)
    dn = np.arange(N_TOTAL - N_DN, N_TOTAL)

    # Each neuron gets a fixed out-degree into the next layer, so `ptr` is
    # trivially monotonic and every population actually receives input.
    targets: List[np.ndarray] = []
    fan = 6
    for source_layer, target_layer in ((orn, alpn), (alpn, kc), (kc, mbon), (mbon, dn)):
        for _ in source_layer:
            targets.append(rng.choice(target_layer, size=fan, replace=True))
    for _ in dn:
        targets.append(np.empty(0, dtype=np.int64))

    degrees = np.array([len(t) for t in targets], dtype=np.int64)
    ptr = np.zeros(N_TOTAL + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(degrees)
    post = (np.concatenate(targets) if len(targets) else np.empty(0)).astype(np.int32)
    # Strong positive weights so a 50 ms window actually produces spikes.
    weight = np.full(len(post), 1.2, dtype=np.float32)

    return Graph(
        ptr=ptr, post=post, weight=weight,
        body_id=np.arange(N_TOTAL, dtype=np.int64),
        type=np.array([f"ORN_T{i % 54}" if i < N_ORN else "x" for i in range(N_TOTAL)],
                      dtype=object),
        cls=np.array(cls, dtype=object),
        superclass=np.array(superclass, dtype=object),
        side=np.array(["L"] * N_TOTAL, dtype=object),
        sign=np.ones(N_TOTAL, dtype=np.int8),
        meta={"synthetic": True},
    )


@pytest.fixture(scope="module")
def runner() -> BrainRunner:
    return BrainRunner(condition="real", window_ms=50.0, verbose=False,
                       graph=synthetic_graph())


def test_populations_are_found(runner: BrainRunner) -> None:
    assert runner.pop_sizes == {"ORN": N_ORN, "ALPN": N_ALPN, "KC": N_KC,
                                "MBON": N_MBON, "DN": N_DN}
    # The readout input is ALPN+KC+DN, in that order.
    assert len(runner.readout_idx) == N_ALPN + N_KC + N_DN


def test_drive_produces_spikes_and_rates(runner: BrainRunner) -> None:
    bits = np.zeros(N_FEATURES, dtype=np.float32)
    bits[[0, 17, 100, 242, 260, 266, 277]] = 1.0

    features, rates, seconds = runner.run(bits)

    assert features.shape == (len(runner.readout_idx),)
    assert np.isfinite(features).all()
    assert (features >= 0).all(), "log1p of counts is non-negative"
    assert seconds >= 0.0
    # Driven ORNs must fire; with tonic 30 mV drive they fire tonically.
    assert rates.orn > 0.0, "driven ORNs did not spike"
    for value in rates.as_dict().values():
        assert value >= 0.0 and np.isfinite(value)


def test_run_is_deterministic_for_the_same_bits(runner: BrainRunner) -> None:
    """`brain.reset()` per window is what makes a decision reproducible."""
    bits = np.zeros(N_FEATURES, dtype=np.float32)
    bits[[1, 20, 150, 244, 262, 267, 278]] = 1.0
    first, rates_a, _ = runner.run(bits)
    second, rates_b, _ = runner.run(bits)
    np.testing.assert_array_equal(first, second)
    assert rates_a.as_dict() == rates_b.as_dict()


def test_different_bits_give_different_brain_features(runner: BrainRunner) -> None:
    a = np.zeros(N_FEATURES, dtype=np.float32)
    a[[0, 1, 2, 242, 260, 266, 277]] = 1.0
    b = np.zeros(N_FEATURES, dtype=np.float32)
    b[[80, 81, 82, 245, 263, 270, 280]] = 1.0
    features_a, _, _ = runner.run(a)
    features_b, _, _ = runner.run(b)
    assert not np.array_equal(features_a, features_b), \
        "the reservoir collapsed distinct states onto identical features"


def test_shuffled_condition_changes_the_response() -> None:
    graph = synthetic_graph()
    real = BrainRunner(condition="real", verbose=False, graph=graph)
    shuffled = BrainRunner(condition="shuffled", verbose=False, graph=graph)
    assert shuffled.brain.shuffle_seed == 0

    bits = np.zeros(N_FEATURES, dtype=np.float32)
    bits[[0, 30, 200, 243, 261, 268, 279]] = 1.0
    features_real, _, _ = real.run(bits)
    features_shuf, _, _ = shuffled.run(bits)
    assert features_real.shape == features_shuf.shape
    assert not np.array_equal(features_real, features_shuf)


# -- readout / policy selection ---------------------------------------------


def _exported_models() -> List[Path]:
    if not MODELS.exists():
        return []
    return sorted(p for p in MODELS.glob("*.npz") if not p.name.endswith("_projection.npz"))


@pytest.mark.skipif(not _exported_models(), reason="no readouts exported yet")
def test_readout_loads_real_exports_and_gives_109_logits() -> None:
    for path in _exported_models():
        readout = Readout(path)
        assert readout.kind in ("linear", "mlp")
        x = np.zeros(readout.n_inputs, dtype=np.float32)
        logits = readout.logits(x)
        assert logits.shape == (109,), f"{path.name} -> {logits.shape}"
        assert np.isfinite(logits).all()

        # A mask of one action must select exactly that action.
        mask = np.zeros(109, dtype=np.int8)
        mask[77] = 1
        assert readout.act(x, mask) == 77


def test_select_policy_reports_its_mode(runner: BrainRunner) -> None:
    policy = select_policy("real", len(runner.readout_idx), prefer="auto", verbose=False)
    assert policy.mode in ("brain", "rand_proj", "bits", "heuristic")
    # Only the brain mode may claim the network is deciding.
    assert policy.uses_brain == (policy.mode == "brain")
    assert policy.detail

    heuristic = select_policy("real", 1, prefer="heuristic", verbose=False)
    assert isinstance(heuristic, HeuristicThroughBrainPolicy)
    assert not heuristic.uses_brain


def test_select_policy_rejects_mismatched_brain_readout() -> None:
    """A readout trained on 6064 inputs must not be fed a 300-unit brain."""
    policy = select_policy("real", 300, prefer="auto", verbose=False)
    if isinstance(policy, ReadoutPolicy):
        assert policy.mode != "brain"


def test_brain_readout_absent_is_an_explicit_error() -> None:
    """`--readout brain` must fail loudly, not silently downgrade."""
    if any((MODELS / f"real_alpn_kc_dn_{k}.npz").exists() for k in ("mlp", "linear")):
        pytest.skip("a brain readout exists, so this cannot be tested")
    with pytest.raises(FileNotFoundError, match="no brain readout"):
        select_policy("real", 6064, prefer="brain", verbose=False)


# --------------------------------------------------------------------------- #
# v3: the glomerular input path                                                #
# --------------------------------------------------------------------------- #
def _v3_model(path: Path, width: int, cond: str, kind: str = "mlp",
              hidden: int = 8) -> Path:
    """A ``bc_train``-shaped export stamped ``encoding=glomerular32``."""
    rng = np.random.default_rng(0)
    payload = {
        "kind": np.array(kind), "mean": np.zeros(width, np.float32),
        "std": np.ones(width, np.float32), "keep": np.ones(width, np.bool_),
        "feature_version": np.int32(2), "cond": np.array(cond),
        "encoding": np.array("glomerular32"),
    }
    if kind == "linear":
        payload.update(W=rng.normal(0, 0.1, (109, width)).astype(np.float32),
                       b=np.zeros(109, np.float32))
    else:
        payload.update(W1=rng.normal(0, 0.1, (hidden, width)).astype(np.float32),
                       b1=np.zeros(hidden, np.float32),
                       W2=rng.normal(0, 0.1, (109, hidden)).astype(np.float32),
                       b2=np.zeros(109, np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


def _synthetic_spec(monkeypatch: "pytest.MonkeyPatch"):
    """The real spec's tuning and drive, on the synthetic graph's ORN type names.

    The 32 MaleCNS glomeruli do not exist in a 3.3k-neuron stand-in, so only the
    names are swapped; everything the code under test reads -- tuning, window,
    seed, drive -- is the one ``outputs/mb2/tuned_config.json`` chose.
    """
    import dataclasses

    from flybalatro import glomerular as G

    spec = dataclasses.replace(
        G.load_spec(), type_names=tuple(f"ORN_T{i}" for i in range(32)),
        source="synthetic")
    monkeypatch.setattr(G, "load_spec", lambda path=None: spec)
    return spec


def test_brain_runner_drives_whole_glomeruli_and_records_mbon(
    monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Only the 32 relay bits reach the fly, and MBON joins the readout vector."""
    from flybalatro import glomerular as G
    from flybalatro.features_v2 import N_FEATURES_V2

    spec = _synthetic_spec(monkeypatch)
    graph = synthetic_graph()
    runner = BrainRunner(condition="real", verbose=False, graph=graph,
                         feature_version=2, encoding=G.ENCODING_GLOM32)
    assert runner.n_bits_to_brain == 32
    assert runner.window_ms == spec.window_ms
    assert runner.feature_map.type_names == list(spec.type_names)
    assert tuple(runner.pop_slices) == ("alpn", "kc", "mbon", "dn")
    assert len(runner.readout_idx) == N_ALPN + N_KC + N_MBON + N_DN

    bits = np.zeros(N_FEATURES_V2, np.float32)
    bits[3] = bits[17] = 1.0
    features, rates, _secs = runner.run(bits)
    assert features.shape == (len(runner.readout_idx),)
    # a state that differs only outside the relay block is the same input
    other = bits.copy()
    other[200:] = 1.0
    np.testing.assert_array_equal(features, runner.run(other)[0])
    # and a silent relay block is a silent fly
    assert runner.run(np.zeros(N_FEATURES_V2, np.float32))[1].alpn == 0.0
    assert rates.alpn >= 0.0


def test_select_policy_gathers_alpn_kc_dn_around_mbon(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """The v3 readout vector is ALPN+KC+MBON+DN; an ALPN+KC+DN readout must skip
    the MBON block rather than take a contiguous slice through it."""
    from flybalatro import glomerular as G
    from flybalatro.realgame import play as pl

    models = tmp_path / "bc3" / "models"
    _v3_model(models / "real_alpn_kc_dn_mlp.npz", N_ALPN + N_KC + N_DN,
              "real_alpn_kc_dn")
    monkeypatch.setattr(pl, "MODEL_DIRS", (models,))

    slices = G.population_slices([N_ALPN, N_KC, N_MBON, N_DN], G.ENCODING_GLOM32)
    policy = select_policy("real", n_brain_inputs=N_ALPN + N_KC + N_MBON + N_DN,
                           pop_slices=slices, prefer="brain", verbose=False)
    take = policy._pop_take
    assert take is not None and len(take) == N_ALPN + N_KC + N_DN
    mbon = set(range(slices["mbon"].start, slices["mbon"].stop))
    assert not (set(take.tolist()) & mbon)
    assert take[0] == 0 and take[-1] == slices["dn"].stop - 1

    features = np.arange(N_ALPN + N_KC + N_MBON + N_DN, dtype=np.float32)
    action = policy.act(None, np.zeros(315, np.float32), features,
                        np.ones(109, np.int8))
    assert 0 <= action < 109


def test_a_glomerular_raw_bits_readout_is_fed_only_32_bits(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from flybalatro.realgame import play as pl

    models = tmp_path / "bc3" / "models"
    _v3_model(models / "raw_bits_mlp.npz", 32, "raw_bits")
    monkeypatch.setattr(pl, "MODEL_DIRS", (models,))
    assert pl.detect_encoding_config("real", "bits") == (2, "glomerular32")
    policy = select_policy("real", n_brain_inputs=None, prefer="bits", verbose=False)
    assert policy.uses_brain is False
    bits = np.zeros(315, np.float32)
    bits[32:] = 1.0                        # only not-sent bits are on
    assert 0 <= policy.act(None, bits, np.zeros(0, np.float32),
                           np.ones(109, np.int8)) < 109
