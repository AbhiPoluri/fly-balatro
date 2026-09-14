"""The viewer's live-learning mode: weight persistence, the decision/dopamine
core, and the launch entries that start the two dashboards.

The expensive half of the mode (the connectome, the glomerular map, the Balatro
environment) is deliberately not exercised here. What *is* exercised is
everything that could be silently wrong:

* a weight file that loads onto the wrong synapses would look fine on screen, so
  the round trip and the alignment check are tested rather than trusted;
* the mode's whole claim is "the fly chooses play or dig, and only its own
  dopamine changes its synapses", so the core is tested on the same ten-neuron
  mushroom body ``tests/test_plasticity.py`` uses, where the right answer is
  hand-computable;
* the launch entries are the documented way to open the dashboards, so a typo in
  a port or a flag is a broken deliverable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from flybalatro import plasticity as P
from flybalatro.brain import Brain
from flybalatro.viewer import plastic as VP
from test_plasticity import _toy_graph, _toy_valence

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_FILES = (
    ROOT / ".claude" / "launch.json",
    Path.home() / ".claude" / "launch.json",
)
PLASTIC_ENTRIES = {"fly-plastic-v1": 8768, "fly-plastic-v2": 8769}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def toy() -> Tuple:
    g = _toy_graph()
    brain = Brain(g, seed=0, v_jitter=0.0)
    val = _toy_valence(g, brain.post, brain.weight)
    return g, brain, val, P.KcMbonPlasticity(brain, g, val)


def _core(toy, learning: bool = True, eta_reward: float = 0.4,
          eta_punish: float = 0.4, bias: float = 0.0,
          temperature: float = 1.0, explore_floor: float = 0.1,
          rng_seed: int = 0) -> VP.PlasticCore:
    g, brain, val, plast = toy
    decider = P.Decider(
        approach=val.approach, avoid=val.avoid, window_ms=50.0,
        bias=bias, temperature=temperature, explore_floor=explore_floor,
    )
    return VP.PlasticCore(
        plast=plast, decider=decider, kc_indices=g.kc_indices(),
        rng=np.random.default_rng(rng_seed), learning=learning,
        eta_reward=eta_reward, eta_punish=eta_punish,
    )


def _counts(g, kc_spikes=(2, 0, 1, 0), approach_hz=4, avoid_hz=1):
    """Spike counts over the toy graph: KCs as given, MBONs by valence."""
    c = np.zeros(g.n, np.int32)
    c[g.kc_indices()] = np.asarray(kc_spikes, np.int32)
    # MBON 4,5 are avoid (glutamate), 6,7 approach (acetylcholine).
    mbon = g.mbon_indices()
    c[mbon[:2]] = avoid_hz
    c[mbon[2:]] = approach_hz
    return c


# --------------------------------------------------------------------------- #
# 1. weight persistence
# --------------------------------------------------------------------------- #
def test_weights_round_trip_restores_every_synapse(toy, tmp_path: Path) -> None:
    g, brain, val, plast = toy
    plast.deliver("reward", np.array([2, 0, 1, 0]), 0.3)
    plast.deliver("punish", np.array([0, 1, 2, 2]), 0.2)
    learned = brain.weight[plast.all_edge].copy()
    assert not np.allclose(learned, plast.all_w0)

    path = plast.save_weights(tmp_path / "w.npz",
                              meta={"bias": -2.5, "temperature": 0.84})
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp*")), "temporary file left behind"

    plast.reset_weights()
    assert np.allclose(brain.weight[plast.all_edge], plast.all_w0)

    loaded = plast.load_weights(path)
    assert np.array_equal(brain.weight[plast.all_edge], learned)
    assert loaded["meta"] == {"bias": -2.5, "temperature": 0.84}
    assert loaded["n_edges"] == len(plast.all_edge)


def test_read_weights_reports_the_stats_without_a_brain(toy, tmp_path: Path) -> None:
    g, brain, val, plast = toy
    # Saturate one KC's synapses so the floor count is non-zero and known.
    for _ in range(400):
        plast.deliver("reward", np.array([2, 0, 0, 0]), 0.9)
    path = plast.save_weights(tmp_path / "w.npz")
    d = P.read_weights(path)
    stats = plast.weight_stats()
    assert d["n_edges"] == stats["n_edges"]
    assert d["mean_ratio"] == pytest.approx(stats["mean_ratio"], abs=1e-6)
    assert d["frac_at_floor"] == pytest.approx(stats["frac_at_floor"], abs=1e-9)
    assert d["frac_unchanged"] == pytest.approx(stats["frac_unchanged"], abs=1e-9)
    assert d["min_ratio"] == pytest.approx(P.WEIGHT_FLOOR, abs=1e-6)
    assert d["n_deliveries"]["reward"] == 400


def test_a_weight_file_from_another_fly_is_refused(toy, tmp_path: Path) -> None:
    g, brain, val, plast = toy
    path = plast.save_weights(tmp_path / "w.npz")
    payload = dict(np.load(path, allow_pickle=False))

    # Same number of edges, different indices: the values would land on the
    # wrong synapses and nothing about the screen would look wrong.
    shifted = dict(payload)
    shifted["edge"] = payload["edge"] + 1
    np.savez_compressed(tmp_path / "shifted.npz", **shifted)
    with pytest.raises(ValueError, match="edge indices do not match"):
        plast.load_weights(tmp_path / "shifted.npz")

    # Same edges, different construction weights: a differently tuned brain.
    retuned = dict(payload)
    retuned["w0"] = payload["w0"] * np.float32(1.5)
    np.savez_compressed(tmp_path / "retuned.npz", **retuned)
    with pytest.raises(ValueError, match="original weights differ"):
        plast.load_weights(tmp_path / "retuned.npz")

    # Wrong format tag.
    bad = dict(payload)
    bad["format"] = np.array("something/else")
    np.savez_compressed(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError, match="format is"):
        P.read_weights(tmp_path / "bad.npz")


def test_loading_weights_leaves_non_kc_mbon_edges_alone(toy, tmp_path: Path) -> None:
    g, brain, val, plast = toy
    plast.deliver("reward", np.array([2, 2, 2, 2]), 0.5)
    path = plast.save_weights(tmp_path / "w.npz")
    plast.reset_weights()

    others = np.setdiff1d(np.arange(len(brain.weight)), plast.all_edge)
    before = brain.weight[others].copy()
    plast.load_weights(path)
    assert np.array_equal(brain.weight[others], before)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_the_shipped_weight_files_match_their_published_runs(version: str) -> None:
    """The saved weights are the run on the record, not a new training run.

    ``weight_stats_after_training`` in the published ``run_*.json`` and the
    statistics of the ``.npz`` have to agree exactly: the training is
    deterministic given the seeds, so anything else means the rerun that produced
    the file diverged from the run the report describes.
    """
    spec = VP.PLASTIC_CONFIGS[version]
    weights = spec.weights_path
    run = spec.out_dir / f"run_{spec.reference_run}.json"
    if not weights.exists() or not run.exists():
        pytest.skip(f"{weights.name} / {run.name} not present")
    d = P.read_weights(weights)
    published = json.loads(run.read_text())["weight_stats_after_training"]
    assert d["n_edges"] == published["n_edges"]
    for key in ("mean_ratio", "min_ratio", "frac_at_floor", "frac_unchanged"):
        assert d[key] == pytest.approx(published[key], rel=0, abs=1e-9), key
    cal = json.loads(run.read_text())["calibration"]
    assert d["meta"]["bias"] == pytest.approx(cal["bias"], abs=1e-9)
    assert d["meta"]["temperature"] == pytest.approx(cal["temperature"], abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. the decision
# --------------------------------------------------------------------------- #
def test_the_decision_is_play_or_dig_and_nothing_else(toy) -> None:
    g = toy[0]
    core = _core(toy)
    seen = set()
    for approach in range(0, 6):
        for avoid in range(0, 6):
            c = _counts(g, approach_hz=approach, avoid_hz=avoid)
            for discard_ok in (True, False):
                out = core.decide(c, discard_ok)
                seen.add(out["action"])
    assert seen <= {VP.PLAY, VP.DIG}
    assert seen == {VP.PLAY, VP.DIG}, "neither branch was ever taken"


def test_with_no_discard_left_the_only_legal_action_is_play(toy) -> None:
    g = toy[0]
    # An odour the fly would certainly dig on: avoidance far above approach.
    # No exploration floor here, so "certainly" is deterministic.
    core = _core(toy, explore_floor=0.0)
    c = _counts(g, approach_hz=0, avoid_hz=5)
    assert core.decide(c, discard_ok=True)["action"] == VP.DIG
    forced = core.decide(c, discard_ok=False)
    assert forced["action"] == VP.PLAY and forced["forced"] is True


def test_learning_off_is_greedy_and_never_explores(toy) -> None:
    g = toy[0]
    core = _core(toy, learning=False, temperature=1.0)
    c = _counts(g, approach_hz=4, avoid_hz=1)
    out = [core.decide(c, True) for _ in range(30)]
    assert {o["action"] for o in out} == {VP.PLAY}
    assert not any(o["explored"] for o in out)
    assert out[0]["p_play"] > 0.5


def test_drive_is_approach_minus_avoid_plus_bias(toy) -> None:
    g = toy[0]
    core = _core(toy, bias=-1.5)
    c = _counts(g, approach_hz=3, avoid_hz=1)
    read = core.read(c)
    # 50 ms window: one spike is 20 Hz.
    assert read["approach_hz"] == pytest.approx(60.0)
    assert read["avoid_hz"] == pytest.approx(20.0)
    assert read["raw_drive"] == pytest.approx(40.0)
    assert read["play_drive"] == pytest.approx(38.5)


# --------------------------------------------------------------------------- #
# 3. the dopamine rule as the viewer drives it
# --------------------------------------------------------------------------- #
def test_learning_off_leaves_every_weight_untouched(toy) -> None:
    g, brain, val, plast = toy
    core = _core(toy, learning=False)
    before = brain.weight.copy()
    for outcome in ({"reward": 1, "punish": 0}, {"reward": 0, "punish": 1}):
        pulse = core.learn(VP.PLAY, outcome, np.array([2, 2, 2, 2]))
        assert pulse.kind == "none"
    assert np.array_equal(brain.weight, before)
    assert core.pulses == {"reward": 0, "punish": 0}


def test_a_reward_pulse_changes_only_eligible_reward_compartment_synapses(toy) -> None:
    g, brain, val, plast = toy
    core = _core(toy, eta_reward=0.4)
    before = brain.weight.copy()

    # KC 0 and KC 2 fired; KC 1 and KC 3 were silent.
    kc_counts = np.array([2, 0, 1, 0])
    pulse = core.learn(VP.PLAY, {"reward": 1, "punish": 0}, kc_counts)
    assert pulse.kind == "reward" and core.pulses == {"reward": 1, "punish": 0}

    changed = np.flatnonzero(brain.weight != before)
    reward_set = plast.targets["reward"]
    punish_set = plast.targets["punish"]
    assert set(changed.tolist()) <= set(reward_set.edge.tolist()), \
        "a reward pulse touched a synapse outside the PAM compartments"
    assert not np.any(brain.weight[punish_set.edge] != before[punish_set.edge])

    # Within the reward compartments, only the KCs that fired moved.
    fired_rows = {0, 2}
    for edge, row in zip(reward_set.edge, reward_set.kc_row):
        moved = brain.weight[edge] != before[edge]
        assert moved == (int(row) in fired_rows), (int(edge), int(row))

    # And depression only: never up, never negative, never past the floor.
    stats = plast.weight_stats()
    assert not stats["any_above_original"] and not stats["any_negative"]
    assert stats["min_ratio"] >= P.WEIGHT_FLOOR - 1e-6


def test_a_punish_pulse_hits_the_other_compartment(toy) -> None:
    g, brain, val, plast = toy
    core = _core(toy, eta_punish=0.4)
    before = brain.weight.copy()
    pulse = core.learn(VP.PLAY, {"reward": 0, "punish": 1}, np.array([2, 2, 2, 2]))
    assert pulse.kind == "punish" and core.pulses == {"reward": 0, "punish": 1}
    changed = set(np.flatnonzero(brain.weight != before).tolist())
    assert changed and changed <= set(plast.targets["punish"].edge.tolist())


def test_a_dig_and_an_unremarkable_play_deliver_nothing(toy) -> None:
    g, brain, val, plast = toy
    core = _core(toy)
    before = brain.weight.copy()
    assert core.learn(VP.DIG, {"reward": 1, "punish": 0},
                      np.array([2, 2, 2, 2])).kind == "none"
    assert core.learn(VP.PLAY, {"reward": 0, "punish": 0},
                      np.array([2, 2, 2, 2])).kind == "none"
    assert np.array_equal(brain.weight, before)


# --------------------------------------------------------------------------- #
# 4. what the panel is fed
# --------------------------------------------------------------------------- #
def test_bucket_table_has_one_row_per_bucket_in_order(toy) -> None:
    core = _core(toy)
    rows = core.bucket_table()
    assert [r["bucket"] for r in rows] == [b for b, _ in VP.BUCKET_LABELS]
    assert [r["label"] for r in rows] == ["far behind", "behind", "close", "enough"]
    assert all(r["n"] == 0 and r["p_play"] is None for r in rows)


def test_recorded_decisions_land_in_their_bucket_and_only_when_dig_was_legal(toy) -> None:
    core = _core(toy)
    core.record(VP.PLAY, "ge1.0", True, {"reward": 1})
    core.record(VP.DIG, "lt0.25", True, {"reward": 0})
    core.record(VP.PLAY, "lt0.25", True, {"reward": 0})
    # Forced play: no choice was made, so it must not move a bucket bar.
    core.record(VP.PLAY, "lt0.5", False, {"reward": 1})
    rows = {r["bucket"]: r for r in core.bucket_table()}
    assert rows["ge1.0"]["n"] == 1 and rows["ge1.0"]["p_play"] == 1.0
    assert rows["lt0.25"]["n"] == 2 and rows["lt0.25"]["p_play"] == 0.5
    assert rows["lt0.5"]["n"] == 0 and rows["lt0.5"]["p_play"] is None
    assert core.hands == 4 and core.plays == 3
    assert core.reward_rate() == pytest.approx(2 / 3)


def test_weight_summary_tracks_the_depression(toy) -> None:
    g, brain, val, plast = toy
    core = _core(toy)
    naive = core.weight_summary()
    assert naive["frac_changed"] == 0.0 and naive["n_changed"] == 0
    assert naive["mean_ratio"] == pytest.approx(1.0)
    assert naive["mean_weight"] == pytest.approx(naive["mean_w0"])

    core.learn(VP.PLAY, {"reward": 1, "punish": 0}, np.array([2, 0, 0, 0]))
    after = core.weight_summary()
    assert 0.0 < after["frac_changed"] < 1.0
    assert after["n_changed"] == int(round(after["frac_changed"] * after["n_edges"]))
    assert after["mean_weight"] < naive["mean_weight"]
    assert after["mean_ratio"] < 1.0
    assert not after["any_above_original"] and not after["any_negative"]


def test_history_grows_one_point_per_recorded_hand(toy) -> None:
    core = _core(toy)
    start = len(core.mean_weight_history)
    for i in range(5):
        core.record(VP.PLAY, "ge1.0", True, {"reward": i % 2})
    assert len(core.mean_weight_history) == start + 5
    assert len(core.reward_history) == 5
    assert core.mean_weight_history.maxlen == VP.HISTORY_POINTS


# --------------------------------------------------------------------------- #
# 5. config specs
# --------------------------------------------------------------------------- #
def test_both_plastic_configs_point_at_files_that_exist() -> None:
    assert set(VP.PLASTIC_CONFIGS) == {"v1", "v2"}
    v1, v2 = VP.PLASTIC_CONFIGS["v1"], VP.PLASTIC_CONFIGS["v2"]
    assert v1.config_path.name == "tuned_config.json"
    assert v1.config_path.parent.name == "plast"
    assert v2.config_path.parent.name == "plast2"
    assert v1.homeostasis is False and v2.homeostasis is True
    # v2's punishment arm was calibrated to match its reward arm per pulse.
    assert v2.eta_punish > v2.eta_reward
    assert v1.eta_punish == v1.eta_reward
    for spec in (v1, v2):
        assert spec.config_path.exists(), spec.config_path
        cfg = json.loads(spec.config_path.read_text())
        has_offsets = isinstance(cfg["tuning"].get("kc_vth_offsets"), list)
        assert has_offsets == spec.homeostasis


def test_resolve_spec_refuses_an_unknown_version() -> None:
    assert VP.resolve_spec("v2").version == "v2"
    with pytest.raises(ValueError, match="plastic-config"):
        VP.resolve_spec("v3")


def test_the_bit_strip_is_capped_to_the_bits_that_exist() -> None:
    """The plastic mode encodes only the 32-bit relay block.

    The readout modes send 315 bits with the trailing 283 drawn faint; the
    plastic harness never computes them, so a layout that mentions them would
    index past the vector.
    """
    from flybalatro.viewer.server import bit_blocks

    full = bit_blocks(2, "glomerular32")
    capped = bit_blocks(2, "glomerular32", 32)
    assert len(capped) < len(full)
    assert all(b["relay"] for b in capped)
    assert max(int(b["to"]) for b in capped) == 32
    assert all(b["sent"] for b in capped)


# --------------------------------------------------------------------------- #
# 6. the launch entries that start the dashboards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", LAUNCH_FILES, ids=lambda p: p.parent.parent.name)
def test_launch_json_starts_both_dashboards(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path} not present")
    data = json.loads(path.read_text())
    assert isinstance(data.get("configurations"), list)
    by_name = {c["name"]: c for c in data["configurations"]}

    # Nothing else in the file was clobbered by the append.
    assert len(by_name) == len(data["configurations"]), "duplicate names"
    assert "fly-viewer" in by_name, "the taught fly's entry is gone"

    for name, port in PLASTIC_ENTRIES.items():
        cfg = by_name[name]
        assert cfg["runtimeExecutable"] == "bash"
        args = cfg["runtimeArgs"]
        assert args[0] == "-c" and len(args) == 2
        cmd = args[1]
        assert cfg["port"] == port
        assert f"--port {port}" in cmd, cmd
        # The repo's own file is checkout-relative (run from the repo root); a
        # user-level file may still cd into an absolute checkout first. Both are
        # accepted, an unrelated absolute path is not.
        assert cmd.startswith(".venv/bin/python") or cmd.startswith(f"cd {ROOT} &&"), cmd
        assert ".venv/bin/python -m flybalatro.viewer.server" in cmd
        assert "--policy plastic" in cmd
        assert f"--plastic-config {name.rsplit('-', 1)[1]}" in cmd
        assert "--learning on" in cmd

    # The two dashboards must not collide with each other or with fly A.
    ports = [c["port"] for c in data["configurations"]]
    assert len(ports) == len(set(ports)), f"duplicate ports in {path}"


def test_the_server_accepts_the_launch_flags() -> None:
    """The flags in launch.json are the flags the server parses."""
    from flybalatro.viewer.server import parse_args

    for name, port in PLASTIC_ENTRIES.items():
        version = name.rsplit("-", 1)[1]
        args = parse_args(["--port", str(port), "--policy", "plastic",
                           "--plastic-config", version, "--learning", "on"])
        assert args.port == port
        assert args.policy == "plastic"
        assert args.plastic_config == version
        assert args.learning == "on"
        assert args.plastic_weights is None


def test_plastic_weights_learned_resolves_to_the_shipped_file() -> None:
    for version in ("v1", "v2"):
        spec = VP.resolve_spec(version)
        assert spec.weights_path.name == f"weights_{spec.reference_run}.npz"
        assert spec.weights_path.parent == spec.out_dir
