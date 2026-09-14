"""Viewer policy selection: model directory order, kinds, encoding, and ranking.

The viewer has to pick a readout *and* the state encoder plus input path that
readout was trained on. Getting the pairing wrong is silent - the brain would be
driven through the wrong channels, or an untuned brain would answer a readout fit
on a tuned one, and the screen would still look fine - so the pairing is tested
rather than eyeballed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from flybalatro.features import N_FEATURES
from flybalatro.features_v2 import N_FEATURES_V2, N_HAND_BLOCK
from flybalatro.glomerular import ENCODING_FEATUREMAP, ENCODING_GLOM32
from flybalatro.viewer import policy as vp

N_ALPN, N_KC, N_MBON, N_DN = 686, 4064, 97, 1314
N_ACTIONS = 109


class FakeGraph:
    """Only the index accessors ``BrainReadoutPolicy`` calls."""

    def alpn_indices(self):
        return np.arange(N_ALPN, dtype=np.int32)

    def kc_indices(self):
        return np.arange(N_ALPN, N_ALPN + N_KC, dtype=np.int32)

    def mbon_indices(self):
        return np.arange(N_ALPN + N_KC, N_ALPN + N_KC + N_MBON, dtype=np.int32)

    def dn_indices(self):
        start = N_ALPN + N_KC + N_MBON
        return np.arange(start, start + N_DN, dtype=np.int32)


def write_model(path: Path, width: int, kind: str, feature_version: int,
                hidden: int = 8, encoding: str = ENCODING_FEATUREMAP) -> Path:
    rng = np.random.default_rng(0)
    # ``bc_train.export_numpy`` stores the condition without the kind suffix.
    cond = path.stem[: -(len(kind) + 1)] if path.stem.endswith("_" + kind) else path.stem
    payload = {
        "kind": np.array(kind),
        "mean": np.zeros(width, np.float32),
        "std": np.ones(width, np.float32),
        "keep": np.ones(width, np.bool_),
        "feature_version": np.int32(feature_version),
        "encoding": np.array(encoding),
        "cond": np.array(cond),
    }
    if kind == "linear":
        payload["W"] = rng.normal(size=(N_ACTIONS, width)).astype(np.float32)
        payload["b"] = np.zeros(N_ACTIONS, np.float32)
    else:
        payload["W1"] = rng.normal(size=(hidden, width)).astype(np.float32)
        payload["b1"] = np.zeros(hidden, np.float32)
        payload["W2"] = rng.normal(size=(N_ACTIONS, hidden)).astype(np.float32)
        payload["b2"] = np.zeros(N_ACTIONS, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


@pytest.fixture()
def model_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tuple[Path, Path]:
    """A fake (bc2, bc) pair, in that search order."""
    new, old = tmp_path / "bc2" / "models", tmp_path / "bc" / "models"
    new.mkdir(parents=True)
    old.mkdir(parents=True)
    monkeypatch.setattr(vp, "MODEL_DIRS", (new, old))
    return new, old


@pytest.fixture()
def v3_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tuple[Path, Path]:
    """A fake (bc3, bc2) pair: the glomerular run must win."""
    new, old = tmp_path / "bc3" / "models", tmp_path / "bc2" / "models"
    new.mkdir(parents=True)
    old.mkdir(parents=True)
    monkeypatch.setattr(vp, "MODEL_DIRS", (new, old))
    return new, old


def test_prefers_the_newer_run_and_reports_its_encoding(model_dirs) -> None:
    new, old = model_dirs
    write_model(old / "real_alpn_kc_dn_linear.npz", N_ALPN + N_KC + N_DN, "linear", 1)
    write_model(new / "real_alpn_kc_dn_mlp.npz", N_ALPN + N_KC + N_DN, "mlp", 2)

    chosen, notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    assert chosen.mode == "brain_readout"
    assert chosen.feature_version == 2
    assert chosen.readout_kind.endswith("ReLU readout")
    assert any("bc2" in note for note in notes)


def test_falls_back_to_the_old_run_when_the_new_one_is_empty(model_dirs) -> None:
    _new, old = model_dirs
    write_model(old / "real_dn_linear.npz", N_DN, "linear", 1)
    chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    assert chosen.mode == "brain_readout"
    assert chosen.feature_version == 1
    assert "DN" in chosen.input_desc


def test_shuffled_never_loads_a_real_wiring_model(model_dirs) -> None:
    new, _old = model_dirs
    write_model(new / "real_alpn_kc_dn_mlp.npz", N_ALPN + N_KC + N_DN, "mlp", 2)
    write_model(new / "raw_bits_mlp.npz", N_FEATURES_V2, "mlp", 2)
    chosen, _notes = vp.select_policy(FakeGraph(), "shuffled", prefer="auto")
    assert chosen.mode == "raw_bits"
    assert chosen.feature_version == 2


def test_raw_bits_policy_carries_its_own_encoding_version(model_dirs) -> None:
    new, old = model_dirs
    write_model(old / "raw_bits_linear.npz", N_FEATURES, "linear", 1)
    chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer="raw")
    assert chosen.feature_version == 1

    write_model(new / "raw_bits_mlp.npz", N_FEATURES_V2, "mlp", 2)
    chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer="raw")
    assert chosen.feature_version == 2


def test_alpn_only_readout_is_accepted_and_sliced(model_dirs) -> None:
    new, _old = model_dirs
    write_model(new / "real_alpn_linear.npz", N_ALPN, "linear", 2)
    chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    assert chosen.mode == "brain_readout"
    assert "ALPN" in chosen.input_desc

    counts = np.arange(N_ALPN + N_KC + N_DN, dtype=np.int32)
    features = chosen.features(counts)
    assert features.shape == (N_ALPN,)
    assert np.allclose(features, np.log1p(np.arange(N_ALPN, dtype=np.float32)).astype(np.float16))


def test_fallbacks_declare_the_brain_is_not_deciding(model_dirs) -> None:
    for prefer in ("heuristic", "random"):
        chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer=prefer)
        assert chosen.brain_in_loop is False
        assert chosen.feature_version == 1
    chosen, notes = vp.select_policy(FakeGraph(), "real", prefer="auto")
    assert chosen.mode in ("heuristic_through_brain", "random_legal")
    assert notes


def test_a_corrupt_export_is_skipped_not_fatal(model_dirs, tmp_path: Path) -> None:
    new, old = model_dirs
    bad = new / "real_alpn_kc_dn_mlp.npz"
    np.savez(bad, kind=np.array("linear"), mean=np.zeros(4, np.float32),
             std=np.ones(4, np.float32), keep=np.ones(4, np.bool_),
             W=np.zeros((3, 4), np.float32), b=np.zeros(3, np.float32))
    write_model(old / "real_dn_linear.npz", N_DN, "linear", 1)
    chosen, notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    assert chosen.mode == "brain_readout"
    assert chosen.feature_version == 1
    assert any("real_alpn_kc_dn_mlp.npz" in note for note in notes)


# --------------------------------------------------------------------------- #
# v3: the glomerular run, its extra populations, and accuracy-based ranking     #
# --------------------------------------------------------------------------- #
def test_the_real_model_dirs_put_bc3_first() -> None:
    """Not a fixture: the shipped default must prefer the newest run."""
    assert [d.parent.name for d in vp.MODEL_DIRS] == ["bc3", "bc2", "bc"]


def test_bc3_wins_over_bc2_and_reports_its_encoding(v3_dirs) -> None:
    new, old = v3_dirs
    write_model(old / "real_alpn_kc_dn_mlp.npz", N_ALPN + N_KC + N_DN, "mlp", 2)
    write_model(new / "real_alpn_kc_dn_mlp.npz", N_ALPN + N_KC + N_DN, "mlp", 2,
                encoding=ENCODING_GLOM32)
    chosen, notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    assert chosen.encoding == ENCODING_GLOM32
    assert chosen.pops == ("alpn", "kc", "dn")
    assert any("bc3" in note and ENCODING_GLOM32 in note for note in notes)


def test_kc_only_and_mbon_only_readouts_are_sliced_correctly(v3_dirs) -> None:
    new, _old = v3_dirs
    write_model(new / "real_mbon_mlp.npz", N_MBON, "mlp", 2, encoding=ENCODING_GLOM32)
    chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    assert chosen.pops == ("mbon",)
    counts = np.arange(N_ALPN + N_KC + N_MBON + N_DN, dtype=np.int32)
    got = chosen.features(counts)
    assert got.shape == (N_MBON,)
    expected = np.log1p(
        np.arange(N_ALPN + N_KC, N_ALPN + N_KC + N_MBON, dtype=np.float32)
    ).astype(np.float16)
    np.testing.assert_allclose(got, expected)

    (new / "real_mbon_mlp.npz").unlink()
    write_model(new / "real_kc_linear.npz", N_KC, "linear", 2,
                encoding=ENCODING_GLOM32)
    chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    assert chosen.pops == ("kc",)
    assert chosen.features(counts).shape == (N_KC,)


def test_alpn_kc_dn_is_gathered_around_mbon(v3_dirs) -> None:
    """ALPN+KC+DN is not contiguous once MBON is recorded between them."""
    new, _old = v3_dirs
    write_model(new / "real_alpn_kc_dn_linear.npz", N_ALPN + N_KC + N_DN, "linear",
                2, encoding=ENCODING_GLOM32)
    chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    counts = np.arange(N_ALPN + N_KC + N_MBON + N_DN, dtype=np.int32)
    got = chosen.features(counts)
    assert got.shape == (N_ALPN + N_KC + N_DN,)
    # the indices gathered skip the MBON block entirely
    want = np.concatenate([np.arange(N_ALPN + N_KC),
                           np.arange(N_ALPN + N_KC + N_MBON,
                                     N_ALPN + N_KC + N_MBON + N_DN)])
    np.testing.assert_allclose(got, np.log1p(want.astype(np.float32)).astype(np.float16))


def test_a_cond_that_contradicts_the_width_is_refused(v3_dirs) -> None:
    new, old = v3_dirs
    write_model(new / "real_kc_mlp.npz", N_DN, "mlp", 2, encoding=ENCODING_GLOM32)
    write_model(old / "real_dn_linear.npz", N_DN, "linear", 2)
    chosen, notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    assert chosen.pops == ("dn",)
    assert any("real_kc" in note and "inputs" in note for note in notes)


def test_raw_bits_under_glomerular32_sees_only_the_relay_block(v3_dirs) -> None:
    new, _old = v3_dirs
    write_model(new / "raw_bits_mlp.npz", N_HAND_BLOCK, "mlp", 2,
                encoding=ENCODING_GLOM32)
    chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer="raw")
    assert chosen.mode == "raw_bits"
    assert chosen.encoding == ENCODING_GLOM32
    assert "relay bits" in chosen.input_desc
    bits = np.zeros(N_FEATURES_V2, np.float32)
    bits[N_HAND_BLOCK:] = 1.0             # only not-sent bits are set
    ctx = vp.DecisionContext(env=None, bits=bits,
                             mask=np.ones(N_ACTIONS, np.int8),
                             counts=np.zeros(1, np.int32))
    decision = chosen.decide(ctx)         # must not raise on a 315-wide vector
    assert 0 <= decision.action < N_ACTIONS


def test_candidates_are_ranked_by_measured_imitation_accuracy(v3_dirs) -> None:
    """DN-only beats the full set here, so the viewer must show DN-only."""
    new, _old = v3_dirs
    write_model(new / "real_alpn_kc_dn_mlp.npz", N_ALPN + N_KC + N_DN, "mlp", 2,
                encoding=ENCODING_GLOM32)
    write_model(new / "real_dn_mlp.npz", N_DN, "mlp", 2, encoding=ENCODING_GLOM32)
    (new.parent / "train_metrics.json").write_text(json.dumps({
        "real_alpn_kc_dn/mlp": {"test": {"top1": 0.40}},
        "real_dn/mlp": {"test": {"top1": 0.90}},
    }))
    chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    assert chosen.pops == ("dn",)
    # without the metrics file the static order wins again
    (new.parent / "train_metrics.json").unlink()
    chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    assert chosen.pops == ("alpn", "kc", "dn")


def test_an_unreadable_metrics_file_does_not_break_selection(v3_dirs) -> None:
    new, _old = v3_dirs
    write_model(new / "real_dn_mlp.npz", N_DN, "mlp", 2, encoding=ENCODING_GLOM32)
    (new.parent / "train_metrics.json").write_text("{not json")
    chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    assert chosen.pops == ("dn",)


def test_a_model_without_an_encoding_field_is_still_featuremap(model_dirs) -> None:
    new, _old = model_dirs
    path = write_model(new / "real_dn_linear.npz", N_DN, "linear", 2)
    z = dict(np.load(path, allow_pickle=False))
    z.pop("encoding")
    np.savez(path, **z)
    chosen, _notes = vp.select_policy(FakeGraph(), "real", prefer="brain")
    assert chosen.encoding == ENCODING_FEATUREMAP
