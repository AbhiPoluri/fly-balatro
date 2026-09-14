"""The v3 glomerular input path: spec, tuned brain, and every stage that reads it.

The point of :mod:`flybalatro.glomerular` is that one module decides which
receptors a relay bit drives and which brain answers, so collection, feature
extraction, training, evaluation, the viewer and the real-game player cannot
drift apart. These tests pin that contract from both ends: the spec file agrees
with the embedded constant, and each consumer narrows the 315-bit state to the
same 32 bits and reads the same population blocks.

Mostly runs on the ~3.3k-neuron synthetic graph from ``test_realgame_brain``; the
two checks that need the real glomerulus sizes load MaleCNS, like
``test_glomerular_encode`` already does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flybalatro import glomerular as G  # noqa: E402
from flybalatro.brain import Brain  # noqa: E402
from flybalatro.encode import GlomerularMap  # noqa: E402
from flybalatro.features_v2 import N_FEATURES_V2, N_HAND_BLOCK  # noqa: E402
from flybalatro.tuning import Tuning, apl_indices  # noqa: E402
from tests.test_realgame_brain import N_ALPN, N_DN, N_KC, N_MBON, synthetic_graph  # noqa: E402


# --------------------------------------------------------------------------- #
# the spec                                                                     #
# --------------------------------------------------------------------------- #
def test_spec_matches_the_tuned_config_the_gate_wrote() -> None:
    """The one non-negotiable: the file and the embedded constant agree."""
    spec = G.load_spec()
    assert spec.source.endswith("tuned_config.json"), "outputs/mb2 spec not found"
    assert spec.type_names == G.GLOM32_TYPES
    assert spec.n_features == N_HAND_BLOCK == 32
    assert spec.tuning == Tuning(apl_scale=2.0)
    assert spec.drive_mv == 30.0
    assert spec.window_ms == 50.0
    assert spec.brain_seed == 1
    assert spec.v_jitter_mv == 6.9
    assert spec.label == "apl2_tonic_vth0"


def test_spec_is_cached_per_path() -> None:
    assert G.load_spec() is G.load_spec()


def _write_spec(tmp_path: Path, **patch) -> Path:
    cfg = json.loads(G.TUNED_CONFIG_PATH.read_text())
    for key, value in patch.items():
        head, _, tail = key.partition(".")
        if tail:
            cfg[head][tail] = value
        else:
            cfg[head] = value
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "tuned_config.json"
    path.write_text(json.dumps(cfg))
    return path


def test_spec_refuses_a_stale_or_wrong_config(tmp_path: Path) -> None:
    cfg = json.loads(G.TUNED_CONFIG_PATH.read_text())
    swapped = [dict(a) for a in cfg["encoding"]["assignment"]]
    swapped[0]["glomerulus"], swapped[1]["glomerulus"] = (
        swapped[1]["glomerulus"], swapped[0]["glomerulus"])
    with pytest.raises(ValueError, match="disagrees"):
        G.load_spec(_write_spec(tmp_path / "a", **{"encoding.assignment": swapped}))
    with pytest.raises(ValueError, match="only tonic"):
        G.load_spec(_write_spec(tmp_path / "b", **{"drive.mode": "poisson"}))
    with pytest.raises(ValueError, match="expected 'glomerular'"):
        G.load_spec(_write_spec(tmp_path / "c", **{"encoding.kind": "featuremap"}))
    with pytest.raises(ValueError, match="8 glomeruli, expected 32"):
        G.load_spec(_write_spec(tmp_path / "d",
                                **{"encoding.assignment": cfg["encoding"]["assignment"][:8]}))


def test_spec_falls_back_to_the_embedded_constants(tmp_path: Path) -> None:
    spec = G.load_spec(tmp_path / "absent.json")
    assert spec.type_names == G.GLOM32_TYPES
    assert spec.tuning == Tuning(apl_scale=2.0)
    assert "GLOM32_TYPES" in spec.source


def test_the_nine_label_glomeruli_are_the_tight_ones() -> None:
    """Bits 0-8 are the 9-way hand-type label, so total drive must not give it away."""
    from flybalatro import connectome as C

    graph = C.load(5)
    gm = G.map_for(graph)
    label_sizes = gm.group_sizes[:9]
    assert max(label_sizes) - min(label_sizes) <= 2
    assert max(label_sizes) <= min(gm.group_sizes[9:])
    assert sum(gm.group_sizes) == len(gm.all_indices()) == 2075


# --------------------------------------------------------------------------- #
# bits in, populations out                                                     #
# --------------------------------------------------------------------------- #
def test_only_the_relay_block_is_sent() -> None:
    assert G.n_input_bits(G.ENCODING_GLOM32, N_FEATURES_V2) == 32
    assert G.n_input_bits(G.ENCODING_FEATUREMAP, N_FEATURES_V2) == N_FEATURES_V2
    bits = np.arange(N_FEATURES_V2)
    np.testing.assert_array_equal(G.relay_bits(bits), np.arange(32))
    rows = np.zeros((4, N_FEATURES_V2), np.uint8)
    rows[:, 40] = 1
    assert G.relay_bits(rows).shape == (4, 32)
    assert G.relay_bits(rows).sum() == 0, "a v1 bit leaked into the relay slice"
    with pytest.raises(ValueError):
        G.relay_bits(np.zeros(8))


def test_population_order_adds_mbon_only_for_v3() -> None:
    assert G.pops_for_encoding(G.ENCODING_FEATUREMAP) == ("alpn", "kc", "dn")
    assert G.pops_for_encoding(G.ENCODING_GLOM32) == ("alpn", "kc", "mbon", "dn")
    with pytest.raises(ValueError):
        G.pops_for_encoding("nope")

    graph = synthetic_graph()
    idx, slices = G.population_indices(graph, G.ENCODING_GLOM32)
    assert len(idx) == N_ALPN + N_KC + N_MBON + N_DN
    assert slices == G.population_slices([N_ALPN, N_KC, N_MBON, N_DN],
                                         G.ENCODING_GLOM32)
    # MBON sits *between* KC and DN, so ALPN+KC+DN is not a contiguous slice.
    assert slices["mbon"].start == slices["kc"].stop
    assert slices["dn"].start == slices["mbon"].stop
    idx2, _ = G.population_indices(graph, G.ENCODING_FEATUREMAP)
    assert len(idx2) == N_ALPN + N_KC + N_DN


# --------------------------------------------------------------------------- #
# the tuned brain, real and shuffled                                           #
# --------------------------------------------------------------------------- #
def _apl_graph():
    """The synthetic graph with one ALPN relabelled ``APL`` so tuning has a target."""
    graph = synthetic_graph()
    types = np.array(graph.type, dtype=object)
    apl = int(np.flatnonzero(graph.cls.astype(str) == "ALPN")[0])
    types[apl] = "APL"
    graph.type = types
    return graph, apl


def _apl_out_weight(brain: Brain, apl: int) -> float:
    a, b = int(brain.ptr[apl]), int(brain.ptr[apl + 1])
    return float(np.abs(brain.weight[a:b]).sum())


def test_apl_scale_applies_by_presynaptic_identity_in_both_wirings() -> None:
    """The shuffle permutes edge *targets*; APL's own out-edges are still its own.

    ``Tuning.apply`` walks ``graph.ptr``, so scaling APL's output does not depend
    on where those edges land -- which is what makes the shuffled control a
    control rather than a differently-tuned brain.
    """
    graph, apl = _apl_graph()
    assert apl_indices(graph).tolist() == [apl]
    untuned = _apl_out_weight(Brain(graph, seed=1), apl)
    spec = G.load_spec()
    for wiring in ("real", "shuffled"):
        brain = G.brain_for(graph, wiring, spec)
        assert _apl_out_weight(brain, apl) == pytest.approx(2.0 * untuned), wiring
        assert brain.tuning == spec.tuning
        assert brain.seed == spec.brain_seed
    # and the graph itself is untouched
    assert _apl_out_weight(Brain(graph, seed=1), apl) == pytest.approx(untuned)


def test_brain_for_rejects_an_unknown_wiring() -> None:
    graph, _ = _apl_graph()
    with pytest.raises(ValueError, match="real.*shuffled"):
        G.brain_for(graph, "scrambled", G.load_spec())


def test_shuffled_brain_keeps_its_degrees_and_differs_from_real() -> None:
    graph, _ = _apl_graph()
    spec = G.load_spec()
    real = G.brain_for(graph, "real", spec)
    shuf = G.brain_for(graph, "shuffled", spec)
    np.testing.assert_array_equal(real.ptr, shuf.ptr)
    np.testing.assert_array_equal(np.sort(real.post), np.sort(shuf.post))
    assert not np.array_equal(real.post, shuf.post)


def test_a_silent_relay_block_produces_no_spikes() -> None:
    """Outside a blind the relay block is all zeros, so the fly is told nothing.

    11,700 of the 150,037 collected states are like this; they all collapse to one
    input under this encoding, and that input is an unlit antennal lobe.
    """
    graph, _ = _apl_graph()
    gm = GlomerularMap(graph, [f"ORN_T{i}" for i in range(32)])
    brain = G.brain_for(graph, "real", G.load_spec())
    brain.warmup()
    brain.reset()
    counts, _ = brain.step(gm.drive(np.zeros(32)), 50.0)
    assert counts.sum() == 0


# --------------------------------------------------------------------------- #
# bc_brain_features / bc_train / bc_eval agree on the narrowing                #
# --------------------------------------------------------------------------- #
def test_dedupe_collapses_states_that_differ_outside_the_relay_block() -> None:
    """This is the whole reason the v3 feature stage is cheap: 150k -> 4,962."""
    rows = np.zeros((3, N_FEATURES_V2), np.uint8)
    rows[0, 1] = rows[1, 1] = rows[2, 5] = 1
    rows[1, 100] = 1                     # differs only in a bit that is not sent
    packed_in = np.packbits(rows[:, :32], axis=1)
    uniq, inverse = np.unique(packed_in, axis=0, return_inverse=True)
    assert len(uniq) == 2
    assert inverse[0] == inverse[1] != inverse[2]
    assert packed_in.shape[1] == 4       # 32 bits, not 40


def test_bc_train_narrows_the_no_brain_controls(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: Path) -> None:
    import bc_train

    n = 6
    rows = np.zeros((n, N_FEATURES_V2), np.uint8)
    rows[:, 3] = 1
    rows[:, 200] = 1
    np.savez(
        tmp_path / "states.npz",
        features_packed=np.packbits(rows, axis=1),
        mask_packed=np.packbits(np.ones((n, 109), np.uint8), axis=1),
        label=np.zeros(n, np.int64), episode=np.zeros(n, np.int32),
        behaviour=np.zeros(n, np.int8), n_features=np.int32(N_FEATURES_V2),
        n_actions=np.int32(109), feature_version=np.int32(2),
    )
    monkeypatch.setattr(bc_train, "BC", tmp_path)
    glom = bc_train.load_states(G.ENCODING_GLOM32)
    assert glom["bits"].shape == (n, 32)
    assert glom["bits"].sum() == n, "a not-sent bit reached the no-brain control"
    assert glom["n_input_bits"] == 32 and glom["n_features"] == N_FEATURES_V2
    full = bc_train.load_states(G.ENCODING_FEATUREMAP)
    assert full["bits"].shape == (n, N_FEATURES_V2)


def test_bc_train_detects_the_encoding_from_the_feature_file(tmp_path: Path) -> None:
    import bc_train

    assert bc_train.detect_encoding(tmp_path) == G.ENCODING_FEATUREMAP
    np.savez(tmp_path / "feats_real.npz",
             meta=np.array(json.dumps({"encoding": G.ENCODING_GLOM32})))
    assert bc_train.detect_encoding(tmp_path) == G.ENCODING_GLOM32


def test_bc_train_knows_the_new_single_population_conditions() -> None:
    import bc_train

    assert bc_train.WIRING["real_kc"] == ("real", ("kc",))
    assert bc_train.WIRING["real_mbon"] == ("real", ("mbon",))
    for cond in ("real_kc", "real_mbon"):
        assert cond in bc_train.CONDITIONS


def _stub_eval_worker(encoding: str, sizes: Dict[str, int]) -> None:
    import bc_eval

    order = G.pops_for_encoding(encoding)
    bc_eval._W.clear()
    bc_eval._W.update(
        encoding=encoding,
        n_input=G.n_input_bits(encoding, N_FEATURES_V2),
        slices=G.population_slices([sizes[p] for p in order], encoding),
        proj={"rand_proj_matched": np.ones((32, 5), np.float32)},
        cache={}, cache_cap=8, hits=0, misses=0,
    )


def test_bc_eval_feeds_the_same_32_bits_to_brain_free_rows() -> None:
    import bc_eval

    sizes = {"alpn": 4, "kc": 5, "mbon": 2, "dn": 3}
    _stub_eval_worker(G.ENCODING_GLOM32, sizes)
    bits = np.zeros(N_FEATURES_V2, np.float32)
    bits[7] = 1.0
    bits[300] = 1.0                       # not sent
    raw = bc_eval._features_for("raw_bits", bits)
    assert raw.shape == (32,) and raw.sum() == 1.0
    proj = bc_eval._features_for("rand_proj_matched", bits)
    assert proj.shape == (5,) and proj[0] == pytest.approx(1.0)


def test_bc_eval_gathers_alpn_kc_dn_around_mbon() -> None:
    import bc_eval

    sizes = {"alpn": 4, "kc": 5, "mbon": 2, "dn": 3}
    _stub_eval_worker(G.ENCODING_GLOM32, sizes)
    full = np.arange(sum(sizes.values()), dtype=np.float16)
    bc_eval._W["cache"][np.packbits(np.zeros(32, np.uint8)).tobytes()] = full
    bits = np.zeros(N_FEATURES_V2, np.float32)
    got = bc_eval._features_for("real_alpn_kc_dn", bits)
    # 0-3 ALPN, 4-8 KC, [9-10 MBON skipped], 11-13 DN
    np.testing.assert_array_equal(got, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13])
    np.testing.assert_array_equal(bc_eval._features_for("real_mbon", bits), [9, 10])
    np.testing.assert_array_equal(bc_eval._features_for("real_kc", bits),
                                  [4, 5, 6, 7, 8])


def test_bc_eval_caches_on_the_relay_bits_alone() -> None:
    """Two states differing outside the relay block must be one cache entry."""
    import bc_eval

    sizes = {"alpn": 1, "kc": 1, "mbon": 1, "dn": 1}
    _stub_eval_worker(G.ENCODING_GLOM32, sizes)
    calls = {"n": 0}

    class _FakeBrain:
        def reset(self) -> None:
            pass

        def step(self, drive, window):  # noqa: ANN001, ANN202
            calls["n"] += 1
            return np.zeros(4, np.int32), 0.0

    bc_eval._W.update(brain=_FakeBrain(), window=50.0,
                      fm=None, idx=np.arange(4, dtype=np.int32))

    class _FakeMap:
        def drive(self, bits):  # noqa: ANN001, ANN202
            assert len(bits) == 32
            return bits

    bc_eval._W["fm"] = _FakeMap()
    a = np.zeros(N_FEATURES_V2, np.float32)
    a[2] = 1.0
    b = a.copy()
    b[250] = 1.0
    bc_eval._features_for("real_dn", a)
    bc_eval._features_for("real_dn", b)
    assert calls["n"] == 1 and bc_eval._W["hits"] == 1


def test_bc_eval_knows_the_new_readout_sources() -> None:
    import bc_eval

    assert bc_eval.READOUT_SOURCE["real_kc"] == ("real", ("kc",))
    assert bc_eval.READOUT_SOURCE["real_mbon"] == ("real", ("mbon",))
