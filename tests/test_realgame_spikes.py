"""Spike recording on the real-game path, and the viewer source that eats it.

Two claims are load-bearing and both are checked here rather than asserted in
prose:

1. Recording the window as five sub-steps is **free of consequence**. The
   readout features, the population rates and therefore the action are
   bit-identical to the un-recorded single 50 ms call, so attaching the viewer
   cannot change how the fly plays.
2. The 32 relay bits the viewer re-simulates from a log record are the bits the
   fly was actually driven with. ``features_v2.hand_block`` builds them and
   ``hand_block_summary`` logs every field of them, so this is a decode, not a
   reconstruction by resemblance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from flybalatro.features import N_FEATURES
from flybalatro.features_v2 import N_HAND_BLOCK, hand_block, hand_block_summary
from flybalatro.realgame import overlay as ov
from flybalatro.realgame.play import N_SUBSTEPS, BrainRunner, SpikeSidecar
from flybalatro.viewer.realgame_source import (
    RealGameSource,
    board_from_record,
    read_sidecar,
    relay_bits,
)

from test_realgame_brain import synthetic_graph

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "realgame" / "log.jsonl"


@pytest.fixture(scope="module")
def runner() -> BrainRunner:
    return BrainRunner(condition="real", window_ms=50.0, verbose=False,
                       graph=synthetic_graph())


def _bits() -> np.ndarray:
    bits = np.zeros(N_FEATURES, dtype=np.float32)
    bits[[0, 17, 100, 242, 260, 266, 277]] = 1.0
    return bits


# ------------------------------------------------- recording changes nothing
def test_recording_substeps_gives_the_same_window(runner: BrainRunner) -> None:
    """Five 10 ms calls == one 50 ms call: the kernel carries its own cursor."""
    bits = _bits()
    plain, rates_plain, _ = runner.run(bits)
    recorded, rates_rec, _ = runner.run(bits, record_substeps=True)

    np.testing.assert_array_equal(plain, recorded)
    assert rates_plain.as_dict() == rates_rec.as_dict()


def test_recording_yields_one_frame_per_substep(runner: BrainRunner) -> None:
    runner.run(_bits(), record_substeps=True)
    subs = runner.last_substeps
    assert subs is not None and len(subs) == N_SUBSTEPS
    assert all(idx.dtype == np.int32 for idx in subs)
    assert all(0 <= int(idx.max(initial=0)) < runner.brain.n for idx in subs)
    # Every sub-step's indices are distinct and sorted: they come from
    # flatnonzero over "fired since the last sub-step".
    for idx in subs:
        assert np.all(np.diff(idx) > 0)
    assert sum(len(idx) for idx in subs) > 0, "nothing fired at all"
    # The totals are over the window, not the sum of the sub-steps: a neuron can
    # fire twice inside one 10 ms slice.
    assert runner.last_n_spiking == len(np.unique(np.concatenate(subs)))
    assert runner.last_total_spikes >= runner.last_n_spiking


def test_not_recording_clears_the_last_window(runner: BrainRunner) -> None:
    runner.run(_bits(), record_substeps=True)
    runner.run(_bits())
    assert runner.last_substeps is None
    assert runner.last_total_spikes is None


# ------------------------------------------------------------- the sidecar
def test_sidecar_round_trips(tmp_path: Path, runner: BrainRunner) -> None:
    runner.run(_bits(), record_substeps=True)
    subs = runner.last_substeps
    assert subs is not None
    sidecar = SpikeSidecar(tmp_path, mode="on")
    first = sidecar.write(subs, 10.0, runner.brain.n,
                          total_spikes=runner.last_total_spikes,
                          n_spiking=runner.last_n_spiking)
    second = sidecar.write(subs, 10.0, runner.brain.n)
    sidecar.close()

    assert first["offset"] == 0 and second["offset"] == first["bytes"]
    assert first["index_space"] == "neuron"
    assert first["n_spiking"] == runner.last_n_spiking
    assert second["n_spiking"] is None
    back = read_sidecar(sidecar.path, int(first["offset"]), int(first["bytes"]))
    assert len(back) == N_SUBSTEPS
    for got, want in zip(back, subs):
        np.testing.assert_array_equal(got.astype(np.int64), want.astype(np.int64))


def test_sidecar_rejects_a_short_read(tmp_path: Path) -> None:
    (tmp_path / "spikes.bin").write_bytes(b"\x00" * 8)
    with pytest.raises(ValueError):
        read_sidecar(tmp_path / "spikes.bin", 0, 64)


def test_recording_is_off_until_a_consumer_asks(tmp_path: Path) -> None:
    auto = SpikeSidecar(tmp_path, mode="auto")
    assert not auto.wanted(), "nothing is listening yet"
    (tmp_path / "spikes.want").touch()
    assert auto.wanted()
    auto.close()

    off = SpikeSidecar(tmp_path, mode="off")
    assert not off.wanted()
    assert not (tmp_path / "spikes.bin").exists(), "mode=off must not open the file"

    on = SpikeSidecar(tmp_path, mode="on")
    assert on.wanted()
    on.close()


def test_recording_cost_is_small(runner: BrainRunner) -> None:
    """The loop must not stall when a consumer attaches.

    A ratio, not a wall clock: the sub-step path runs the same kernel steps and
    adds four ``counts`` copies and five ``flatnonzero`` passes. The two are
    timed interleaved and compared on their *best* run, so a busy machine
    perturbs both equally instead of failing the cheaper one. The bound is
    loose on purpose -- this graph is ~4k neurons, where five Python-level
    kernel calls cost proportionally far more than they do on MaleCNS's 146k.
    """
    import time

    bits = _bits()
    runner.run(bits)                       # warm
    runner.run(bits, record_substeps=True)

    best = {False: float("inf"), True: float("inf")}
    for _ in range(9):
        for record in (False, True):
            t0 = time.perf_counter()
            runner.run(bits, record_substeps=record)
            best[record] = min(best[record], time.perf_counter() - t0)
    assert best[True] < 4.0 * best[False]


# ------------------------------------------------ the 32 bits, back out again
def test_relay_bits_match_the_encoder_that_wrote_them() -> None:
    """Decode a logged analysis and compare against ``hand_block`` itself."""
    from flybalatro.env import BalatroEnv

    env = BalatroEnv(ante_end=1, max_steps=50)
    env.reset(seed=7)
    state = env.engine.state
    block, _analysis, _needed = hand_block(state)
    summary = hand_block_summary(state)

    record = {
        "decision": 1,
        "relayed_hand_analysis": summary,
        "state": {"hand": [{"slot": i, "rank_index": int(c.rank_index),
                            "suit_index": int(c.suit_index)}
                           for i, c in enumerate(state.available)],
                  "selected_indices": [], "raw_state": "SELECTING_HAND"},
    }
    dec = ov.parse_record(record)
    assert dec is not None
    np.testing.assert_array_equal(relay_bits(dec), block)


def test_relay_bits_agree_with_every_shipped_record() -> None:
    """``n_bits_on`` is logged, so the decode is self-checking on real data."""
    decisions = ov.read_log(LOG, with_cards_only=False)
    assert len(decisions) > 200
    active = 0
    for dec in decisions:
        bits = relay_bits(dec)          # raises if the popcount disagrees
        assert bits.shape == (N_HAND_BLOCK,)
        active += int(dec.relay_active)
    assert active > 100, "the shipped log should be mostly in-blind decisions"


def test_relay_bits_are_all_off_outside_a_blind() -> None:
    dec = next(d for d in ov.read_log(LOG, with_cards_only=False)
               if not d.relay_active)
    assert relay_bits(dec).sum() == 0.0


# ---------------------------------------------------------- the board panel
def test_board_marks_what_the_real_game_does_not_log() -> None:
    dec = next(d for d in ov.read_log(LOG) if d.cards)
    board = board_from_record(dec)
    assert board["ante_end"] == "?", "the record has no ante target"
    assert all(card["enhancement"] is None for card in board["cards"])
    assert all(card["chips"] is None for card in board["cards"])
    assert len(board["cards"]) == dec.n_cards
    assert board["n_selected"] == len(set(dec.selected))
    assert board["blind_name"] == dec.blind


# --------------------------------------------------------------- the source
def _source(tmp_path: Path) -> RealGameSource:
    return RealGameSource(tmp_path, replay=LOG)


def test_replay_source_produces_frames_the_browser_understands(tmp_path: Path) -> None:
    src = _source(tmp_path)
    assert not src.live
    frames = src.frames(src.poll()[0])
    assert frames.state is not None and frames.decision is not None
    assert frames.state["t"] == "state"
    assert len(frames.state["bits"]) == N_HAND_BLOCK
    # No readout was logged, so none is invented.
    assert frames.decision["has_logits"] is False
    assert frames.decision["confidence"] is None
    assert frames.decision["slots"] == []
    assert frames.decision["chips_gained"] is None
    assert set(frames.decision["rates"]) <= {"ORN", "ALPN", "KC", "MBON", "DN"}


def test_replay_without_a_resimulator_says_it_has_no_spikes(tmp_path: Path) -> None:
    src = _source(tmp_path)
    frames = src.frames(src.poll()[0])
    assert frames.substeps == []
    assert frames.decision is not None
    assert frames.decision["spike_mode"] == "none"
    assert frames.decision["total_spikes"] is None


def test_source_reads_a_sidecar_when_the_record_points_at_one(
    tmp_path: Path, runner: BrainRunner
) -> None:
    runner.run(_bits(), record_substeps=True)
    subs = runner.last_substeps
    assert subs is not None
    sidecar = SpikeSidecar(tmp_path, mode="on")
    ref = sidecar.write(subs, 10.0, runner.brain.n,
                        total_spikes=runner.last_total_spikes,
                        n_spiking=runner.last_n_spiking)
    sidecar.close()

    obj = json.loads(next(iter(open(LOG, encoding="utf-8"))))
    obj["spikes"] = ref
    log = tmp_path / "one.jsonl"
    log.write_text(json.dumps(obj) + "\n", encoding="utf-8")

    # Identity remap, so points == neurons and the comparison is direct.
    point_index = np.arange(runner.brain.n, dtype=np.int32)
    src = RealGameSource(tmp_path, replay=log, point_index=point_index)
    frames = src.frames(src.poll()[0])
    assert frames.decision is not None
    assert frames.decision["spike_mode"] == "recorded"
    assert frames.decision["n_spiking"] == runner.last_n_spiking
    assert len(frames.substeps) == N_SUBSTEPS
    import struct

    for blob, want in zip(frames.substeps, subs):
        k, count = struct.unpack_from("<II", blob, 0)
        got = np.frombuffer(blob, np.uint32, count=count, offset=8)
        np.testing.assert_array_equal(got.astype(np.int64), want.astype(np.int64))
        assert 0 <= k < N_SUBSTEPS


def test_source_resimulates_when_there_is_no_sidecar(tmp_path: Path) -> None:
    """The fallback drives the logged bits, and says which mode it used."""
    calls: List[np.ndarray] = []

    def fake(bits: np.ndarray):
        calls.append(bits.copy())
        counts = np.zeros(16, dtype=np.int32)
        counts[[1, 2, 3]] = 2
        fired = [np.array([1, 2], np.int32), np.array([3], np.int32)]
        return counts, fired, 0.01

    src = RealGameSource(tmp_path, replay=LOG,
                         point_index=np.arange(16, dtype=np.int32),
                         resimulate=fake)
    dec = next(d for d in src._replay if d.relay_active)
    frames = src.frames(dec)
    assert frames.decision is not None
    assert frames.decision["spike_mode"] == "re-simulated"
    assert frames.decision["total_spikes"] == 6
    assert frames.decision["n_spiking"] == 3
    assert len(frames.substeps) == 2
    np.testing.assert_array_equal(calls[0], relay_bits(dec))


def test_live_source_asks_for_spikes_and_never_blocks(tmp_path: Path) -> None:
    (tmp_path / "latest.json").write_text(
        next(iter(open(LOG, encoding="utf-8"))), encoding="utf-8"
    )
    src = RealGameSource(tmp_path, want_interval=0.0)
    assert src.live
    assert src.poll(), "the first poll should see the file"
    assert (tmp_path / "spikes.want").exists()
    assert src.poll() == [], "unchanged latest.json must return immediately"
