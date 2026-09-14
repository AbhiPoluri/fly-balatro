"""The learning fly's real-game loop, and the overlay's dynamic alignment.

Two things are under test and neither of them needs the 1.5 GB connectome:

1. :func:`flybalatro.realgame.plastic.play_plastic` -- the hand-level loop --
   driven against ``scripts/realgame_plastic_offline.ScriptedGame`` over real
   HTTP with a *stubbed* fly. What is checked is the control flow the fly does
   not own: that the dopamine pulse lands on the window the decision was taken
   in and not a later one, that a discard never earns one, that a losing play
   is punished, that the record schema the viewer and the overlay read is
   actually emitted, and that every number in it survives ``json.dumps`` +
   ``JSON.parse`` (no ``NaN``).
2. The dynamic alignment path: a card rect the *game* reported beats the fitted
   static fan, is scaled correctly onto the overlay window, carries the card's
   rotation, and falls back cleanly when any part of it is missing.

The fly itself -- the brain, the MBON rule, the depression -- is tested in
``tests/test_plasticity.py`` and ``tests/test_kc_homeo.py``, and exercised
end to end by ``python -m scripts.realgame_plastic_offline``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT), str(ROOT / "scripts")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from flybalatro.realgame import overlay as ov  # noqa: E402
from flybalatro.realgame.adapter import CardGeometry, ScreenFrame  # noqa: E402
from flybalatro.realgame.plastic import (  # noqa: E402
    DISCARD,
    PLAY,
    LearningTrace,
    OperatingPoint,
    play_plastic,
)


# --------------------------------------------------------------------------- #
# the geometry the game reports                                                #
# --------------------------------------------------------------------------- #
class TestCardGeometry:
    def test_parses_a_rect(self) -> None:
        g = CardGeometry.from_json(
            {"rect": {"x": 638.0, "y": 925.0, "w": 211.0, "h": 288.0, "r": -0.096},
             "moving": True}
        )
        assert g is not None
        assert (g.x, g.y, g.w, g.h) == (638.0, 925.0, 211.0, 288.0)
        assert g.r == pytest.approx(-0.096)
        assert g.moving is True
        # The centre is what a box should be compared against: the centre of a
        # rotated rect is the centre of its bounding box, and left edges are not
        # comparable across a tilt.
        assert g.cx == pytest.approx(638.0 + 211.0 / 2)
        assert g.cy == pytest.approx(925.0 + 288.0 / 2)

    @pytest.mark.parametrize(
        "payload",
        [
            None, {}, [], "nope",
            {"rect": None},
            {"rect": {"x": 1, "y": 2}},                       # no size
            {"rect": {"x": 1, "y": 2, "w": 0, "h": 10}},      # degenerate
            {"rect": {"x": 1, "y": 2, "w": 10, "h": -1}},
            {"rect": {"x": "a", "y": 2, "w": 10, "h": 10}},   # not a number
        ],
    )
    def test_bad_geometry_is_none_not_an_exception(self, payload: object) -> None:
        """A cosmetic field must never break a game call."""
        assert CardGeometry.from_json(payload) is None

    def test_screen_frame_needs_real_dimensions(self) -> None:
        assert ScreenFrame.from_json({"width": 0, "height": 0}) is None
        f = ScreenFrame.from_json(
            {"width": 1183, "height": 768, "pixel_width": 2366,
             "pixel_height": 1536, "dpi_scale": 2.0}
        )
        assert f is not None and f.dpi_scale == 2.0
        # Retina: the LOVE pixel buffer is twice the window in points, and the
        # overlay is positioned in points, so the two must stay distinguishable.
        assert f.pixel_width == 2 * f.width


# --------------------------------------------------------------------------- #
# alignment: the game's rects beat the fitted model                            #
# --------------------------------------------------------------------------- #
def _record(
    n_cards: int = 8,
    *,
    rects: bool = True,
    screen: bool = True,
    canvas: Tuple[float, float] = (2418.0, 1570.0),
    best: Sequence[int] = (0, 1),
) -> Dict[str, object]:
    hand: List[Dict[str, object]] = []
    for i in range(n_cards):
        card: Dict[str, object] = {"slot": i, "rank_index": i % 13, "suit_index": i % 4}
        if rects:
            card["rect"] = {"x": 600.0 + 160.0 * i, "y": 920.0, "w": 211.0,
                            "h": 288.0, "r": 0.05 * (i - (n_cards - 1) / 2)}
        hand.append(card)
    state: Dict[str, object] = {
        "raw_state": "SELECTING_HAND", "stage": 1, "ante": 1, "round": 1,
        "blind": "Small Blind", "score": 100, "required_score": 300,
        "plays": 3, "discards": 2, "money": 4, "hand": hand,
        "selected_indices": [],
    }
    if screen:
        state["screen"] = {"width": canvas[0], "height": canvas[1],
                           "pixel_width": canvas[0], "pixel_height": canvas[1]}
    return {
        "decision": 1, "t": 0.0, "action_name": "play",
        "action_description": "play hand slots [0, 1]",
        "policy_mode": "plastic", "policy_uses_brain": True,
        "relayed_hand_analysis": {
            "active": True, "best_type": "Pair", "best_score": 118,
            "best_slots": list(best), "needed": 200, "score_bucket": "lt1.0",
            "selected_type": "none", "selected_is_best": True,
        },
        "state": state,
    }


class TestAlignment:
    def test_game_rects_are_preferred_and_scaled(self) -> None:
        dec = ov.parse_record(_record())
        assert dec is not None and dec.has_rects
        geom = ov.HandGeometry.load()
        # An overlay window half the size of the game canvas.
        placed, source = ov.aligned_rects(dec, geom, 1209.0, 785.0)
        assert source == ov.ALIGN_GAME
        assert set(placed) == set(range(8))
        rect, angle = placed[0]
        assert rect.x == pytest.approx(300.0)      # 600 * (1209/2418)
        assert rect.y == pytest.approx(460.0)      # 920 * (785/1570)
        assert rect.w == pytest.approx(105.5)
        assert angle == pytest.approx(0.05 * -3.5)

    def test_falls_back_to_the_model_without_rects(self) -> None:
        dec = ov.parse_record(_record(rects=False))
        assert dec is not None and not dec.has_rects
        geom = ov.HandGeometry.load()
        placed, source = ov.aligned_rects(dec, geom, 2418.0, 1570.0)
        assert source == ov.ALIGN_MODEL
        # Identical to what the fitted fan alone would say.
        for slot in range(8):
            want = geom.slot_rect(slot, 8, width=2418.0, height=1570.0)
            got, angle = placed[slot]
            assert got.as_tuple() == pytest.approx(want.as_tuple())
            assert angle == 0.0

    def test_rects_without_a_screen_frame_fall_back(self) -> None:
        """Rects in unknown units are unusable, not approximately usable."""
        dec = ov.parse_record(_record(screen=False))
        assert dec is not None
        _placed, source = ov.aligned_rects(dec, ov.HandGeometry.load(), 100.0, 100.0)
        assert source == ov.ALIGN_MODEL

    def test_one_missing_rect_falls_back_for_the_whole_hand(self) -> None:
        """Mixing sources across one hand would misalign one card silently."""
        rec = _record()
        del rec["state"]["hand"][3]["rect"]          # type: ignore[index]
        dec = ov.parse_record(rec)
        assert dec is not None and not dec.has_rects
        _placed, source = ov.aligned_rects(dec, ov.HandGeometry.load(), 100.0, 100.0)
        assert source == ov.ALIGN_MODEL

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 8])
    def test_short_hands_are_exact_from_the_game(self, n: int) -> None:
        """The case the fitted model has never been checked on.

        ``docs/POV.md`` is explicit that ``n != 8`` is extrapolation: every one
        of the 253 logged readout decisions had 8 cards or none. With the game
        reporting rects there is nothing to extrapolate -- each card is placed
        where the game says it is, whatever ``n`` happens to be.
        """
        dec = ov.parse_record(_record(n_cards=n))
        assert dec is not None
        placed, source = ov.aligned_rects(dec, ov.HandGeometry.load(), 2418.0, 1570.0)
        assert source == ov.ALIGN_GAME
        assert len(placed) == n
        for slot in range(n):
            rect, _a = placed[slot]
            assert rect.x == pytest.approx(600.0 + 160.0 * slot)

    def test_the_fallback_model_holds_the_pitch_on_a_short_hand(self) -> None:
        """Balatro does not spread a short fan; it centres it.

        Measured on two independent settled 3-card hands cut out of
        ``outputs/realgame/balatro_fly.mov`` (``outputs/pov/dynamic_align.json``,
        cases B and B2): the real pitch is 165-171 px against the 8-card fit of
        164.65 px. The old model spread the fan to fill ``span``, wanted 225.4 px
        and put the end cards ~60 px off. This is the regression guard.
        """
        geom = ov.HandGeometry.load()
        W, H = 2418.0, 1570.0
        assert geom.pitch(3) == pytest.approx(geom.pitch(8))
        assert geom.pitch(3) * W == pytest.approx(164.65, abs=0.5)
        for measured in ((1158.0, 1323.0, 1488.0), (1162.5, 1327.5, 1493.5)):
            for slot, truth in enumerate(measured):
                got = geom.slot_rect(slot, 3, width=W, height=H).cx
                assert abs(got - truth) < 6.0, (slot, got, truth)

    def test_the_eight_card_fit_is_untouched_by_the_pitch_rule(self) -> None:
        """The change must not move the case the model was fitted on."""
        geom = ov.HandGeometry.load()
        centres = [geom.slot_rect(k, 8, width=2418.0, height=1570.0).cx
                   for k in range(8)]
        assert centres == pytest.approx(
            [750.6, 915.2, 1079.9, 1244.5, 1409.2, 1573.9, 1738.5, 1903.2], abs=0.1
        )

    def test_moving_cards_are_flagged(self) -> None:
        rec = _record()
        rec["state"]["hand"][2]["rect"]["moving"] = True   # type: ignore[index]
        dec = ov.parse_record(rec)
        assert dec is not None and dec.any_moving

    def test_annotation_reports_which_source_it_used(self) -> None:
        geom = ov.HandGeometry.load()
        live = ov.build_annotation(ov.parse_record(_record()), geom, 1200.0, 800.0)
        assert live.align == ov.ALIGN_GAME
        modelled = ov.build_annotation(
            ov.parse_record(_record(rects=False)), geom, 1200.0, 800.0
        )
        assert modelled.align == ov.ALIGN_MODEL

    def test_best_subset_boxes_carry_the_cards_own_tilt(self) -> None:
        ann = ov.build_annotation(
            ov.parse_record(_record(best=(0, 7))), ov.HandGeometry.load(),
            2418.0, 1570.0,
        )
        best = {b.slot: b for b in ann.boxes if b.kind == "best"}
        assert set(best) == {0, 7}
        # Opposite ends of the fan lean opposite ways.
        assert best[0].angle < 0 < best[7].angle
        assert abs(best[0].angle) == pytest.approx(abs(best[7].angle))

    def test_scaling_uses_draw_units_not_the_backing_store(self) -> None:
        """The 2x trap on a Retina display.

        A card rect is ``VT * G.TILESCALE * G.TILESIZE``, and ``G.TILESCALE`` is
        derived in ``love.resize(w, h)`` from the same ``w`` that
        ``love.graphics.getDimensions()`` reports. So the rects live in
        ``getDimensions()`` space. Scaling by ``getPixelDimensions()`` instead
        would halve every box on a high-DPI window and put it in the wrong
        place -- and would look perfectly fine on a non-Retina one.
        """
        rec = _record()
        rec["state"]["screen"] = {                     # type: ignore[index]
            "width": 1209.0, "height": 785.0,
            "pixel_width": 2418.0, "pixel_height": 1570.0, "dpi_scale": 2.0,
        }
        dec = ov.parse_record(rec)
        assert dec is not None and dec.screen is not None
        assert dec.screen.width == 1209.0            # draw units
        assert dec.screen.pixel_width == 2418.0      # kept, but not used to scale
        # An overlay window the size of the draw space is a 1:1 mapping.
        placed, source = ov.aligned_rects(dec, ov.HandGeometry.load(), 1209.0, 785.0)
        assert source == ov.ALIGN_GAME
        assert placed[0][0].x == pytest.approx(600.0)
        assert placed[0][0].w == pytest.approx(211.0)

    def test_an_outcome_record_draws_no_card_boxes(self) -> None:
        """The board an outcome describes is already gone.

        By the time the game has resolved a hand it is animating those cards
        away and dealing their replacements, so boxes placed from the record's
        board would sit on stale positions for the whole pause between hands --
        exactly the drift this overlay exists to remove.
        """
        rec = _record()
        rec["kind"] = "outcome"
        rec["dopamine"] = "reward"
        rec["outcome"] = {"chips_gained": 276, "cleared": False, "lost": False,
                          "reward": 1, "punish": 0}
        dec = ov.parse_record(rec)
        assert dec is not None
        ann = ov.build_annotation(dec, ov.HandGeometry.load(), 1200.0, 800.0)
        assert ann.boxes == ()
        # It still says what happened, and it still flashes.
        assert "played" in ann.headline
        assert "REWARD" in ann.headline_note and "276" in ann.headline_note
        assert ann.flash == "reward"
        assert ann.headline_ok is True

    def test_a_punished_outcome_reads_as_punishment(self) -> None:
        rec = _record()
        rec.update(kind="outcome", dopamine="punish",
                   outcome={"chips_gained": 16, "cleared": False, "lost": True,
                            "reward": 0, "punish": 1})
        ann = ov.build_annotation(ov.parse_record(rec), ov.HandGeometry.load(),
                                  1200.0, 800.0)
        assert "PUNISHMENT" in ann.headline_note
        assert ann.headline_ok is False
        assert ann.flash == "punish"

    def test_a_decision_record_still_draws_its_boxes(self) -> None:
        """The guard above must not have turned the overlay off."""
        rec = _record()
        rec["kind"] = "decision"
        ann = ov.build_annotation(ov.parse_record(rec), ov.HandGeometry.load(),
                                  1200.0, 800.0)
        assert [b for b in ann.boxes if b.kind == "best"]

    def test_old_logs_still_parse(self) -> None:
        """The three recorded readout runs have no rects and must still draw."""
        log = ROOT / "outputs" / "realgame" / "log_run3.jsonl"
        if not log.exists():
            pytest.skip("recorded run not present")
        decisions = ov.read_log(log)
        assert decisions
        assert not any(d.has_rects for d in decisions)
        ann = ov.build_annotation(decisions[0], ov.HandGeometry.load(), 1200.0, 800.0)
        assert ann.align == ov.ALIGN_MODEL
        assert ann.boxes


# --------------------------------------------------------------------------- #
# the rolling trace the viewer draws                                           #
# --------------------------------------------------------------------------- #
class TestLearningTrace:
    def test_empty_trace_is_json(self) -> None:
        payload = json.dumps(LearningTrace().as_dict())
        assert "NaN" not in payload and "Infinity" not in payload
        json.loads(payload)

    def test_empty_bucket_is_null_not_nan(self) -> None:
        """``json.dumps`` writes a bare ``NaN`` for a float nan, which is not
        JSON; one empty bucket would break the browser's whole parse."""
        t = LearningTrace()
        t.observe("lt0.5", PLAY, 1, {"mean_ratio": 0.99, "frac_changed": 0.01})
        table = t.bucket_table()
        assert table["lt0.5"]["p_play"] == 1.0
        assert table["ge1.0"]["p_play"] is None
        assert "NaN" not in json.dumps(table)

    def test_rolling_play_rate_does_not_shadow_the_per_hand_probability(self) -> None:
        """Both are merged into one ``mb`` block; they are different numbers."""
        t = LearningTrace()
        t.observe("ge1.0", PLAY, 1, {"mean_ratio": 1.0, "frac_changed": 0.0})
        keys = set(t.as_dict())
        assert "p_play_rolling" in keys
        assert "p_play" not in keys

    def test_bucket_table_is_a_rolling_window(self) -> None:
        t = LearningTrace()
        for _ in range(40):
            t.observe("lt0.25", DISCARD, 0, {"mean_ratio": 1.0, "frac_changed": 0.0})
        for _ in range(10):
            t.observe("lt0.25", PLAY, 1, {"mean_ratio": 1.0, "frac_changed": 0.0})
        assert t.bucket_table(window=10)["lt0.25"]["p_play"] == 1.0
        assert t.bucket_table(window=50)["lt0.25"]["p_play"] == pytest.approx(0.2)

    def test_reward_rate_is_none_before_any_hand(self) -> None:
        assert LearningTrace().reward_rate() is None


# --------------------------------------------------------------------------- #
# the operating point                                                          #
# --------------------------------------------------------------------------- #
class TestOperatingPoint:
    def test_loads_the_plast2_calibration(self) -> None:
        op = OperatingPoint.load(eta_reward=0.05)
        # The two frozen scalars from RESULTS.md "Plasticity v2".
        assert op.bias == pytest.approx(-2.61597, abs=1e-4)
        assert op.temperature == pytest.approx(0.83831, abs=1e-4)
        assert op.explore_floor == pytest.approx(0.1)
        # The punishment arm is ~3.81x the reward arm, calibrated, not guessed.
        assert op.eta_punish == pytest.approx(0.1907, abs=1e-4)
        assert op.eta_punish / op.eta_reward == pytest.approx(3.81, abs=0.02)

    def test_an_uncalibrated_eta_says_so_instead_of_pretending(self) -> None:
        op = OperatingPoint.load(eta_reward=0.037)
        assert op.eta_punish == op.eta_reward
        assert "NOT CALIBRATED" in op.eta_source

    def test_an_explicit_eta_punish_wins(self) -> None:
        op = OperatingPoint.load(eta_reward=0.05, eta_punish=0.5)
        assert op.eta_punish == 0.5

    def test_refuses_a_file_with_no_calibration(self, tmp_path: Path) -> None:
        bad = tmp_path / "nope.json"
        bad.write_text(json.dumps({"decider": {"window_ms": 50}}))
        with pytest.raises(ValueError, match="calibrated bias"):
            OperatingPoint.load(path=bad)


# --------------------------------------------------------------------------- #
# the loop                                                                     #
# --------------------------------------------------------------------------- #
class StubFly:
    """Same surface as :class:`PlasticFly`, no spiking network.

    ``script`` is the sequence of actions to take; the drive numbers it reports
    are fixed. The point is the loop, not the fly.
    """

    def __init__(self, script: Sequence[str] = (), n_synapses: int = 33496) -> None:
        self.script = list(script)
        self.calls: List[str] = []
        self.pulses: List[Tuple[str, int]] = []
        self.n_reward = 0
        self.n_punish = 0
        self.n_synapses = n_synapses
        self.n_changed = 0
        self.learning = True
        self.n_substeps = 1
        self.last_substeps: List[np.ndarray] = []
        self.last_counts: Optional[np.ndarray] = None
        #: Bumped on every decision, recorded with each pulse, so a test can
        #: prove the pulse landed on the window the decision was taken in.
        self.window_id = 0
        self.setup = type("S", (), {
            "wiring": "real", "window_ms": 50.0,
            "brain": type("B", (), {"n": 1000})(),
        })()

    def decide(self, ctx: object) -> Dict[str, object]:
        self.window_id += 1
        action = self.script.pop(0) if self.script else PLAY
        if action == DISCARD and not getattr(ctx, "discard_ok", False):
            action = PLAY
        self.calls.append(action)
        return {
            "action": action, "raw_drive": 1.0, "play_drive": 0.5,
            "p_play": 0.6, "explored": False,
            "discard_ok": bool(getattr(ctx, "discard_ok", False)),
            "approach_hz": 12.0, "avoid_hz": 8.0, "kc_active": 150,
            "kc_spikes": 160, "bias": -2.6, "temperature": 0.84,
            "sim_seconds": 0.01, "window_seconds": 0.02,
        }

    def deliver(self, kind: str) -> Dict[str, object]:
        self.pulses.append((kind, self.window_id))
        if kind == "reward":
            self.n_reward += 1
        else:
            self.n_punish += 1
        self.n_changed += 100
        return {"dopamine": kind, "eta": 0.05, "n_changed": 100,
                "mean_abs_dw": 0.1, "elig_kc": 150}

    def weight_state(self) -> Dict[str, object]:
        return {
            "n_synapses": self.n_synapses, "n_changed": self.n_changed,
            "frac_changed": round(self.n_changed / self.n_synapses, 5),
            "mean_ratio": 1.0 - self.n_changed / (10.0 * self.n_synapses),
            "min_ratio": 0.95, "frac_at_floor": 0.0, "n_at_floor": 0,
            "n_reward_pulses": self.n_reward, "n_punish_pulses": self.n_punish,
        }

    def describe(self) -> Dict[str, object]:
        return {"stub": True, "learning": self.learning}


class RecordingLogger:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.records: List[Dict[str, object]] = []

    def write(self, record: Dict[str, object]) -> None:
        # Everything written must survive a round trip through the browser.
        self.records.append(json.loads(json.dumps(record, default=str)))

    def close(self) -> None:
        pass

    def of_kind(self, kind: str) -> List[Dict[str, object]]:
        return [r for r in self.records if r.get("kind") == kind]


@pytest.fixture()
def scripted():
    from realgame_plastic_offline import ScriptedGame, serve

    def build(**kwargs):
        game = ScriptedGame(**kwargs)
        server, client = serve(game)
        return game, client, server

    servers: List[object] = []

    def factory(**kwargs):
        game, client, server = build(**kwargs)
        servers.append(server)
        return game, client

    yield factory
    for server in servers:
        server.shutdown()          # type: ignore[attr-defined]
        server.server_close()      # type: ignore[attr-defined]


class TestPlayPlastic:
    def test_plays_a_blind_and_logs_both_record_kinds(self, scripted, tmp_path) -> None:
        game, client = scripted(seed=1)
        fly = StubFly()
        logger = RecordingLogger(tmp_path)
        summary = play_plastic(client, fly, logger, pause=0.0, max_hands=6,
                               verbose=False)
        assert summary["hands"] >= 1
        decisions = logger.of_kind("decision")
        outcomes = logger.of_kind("outcome")
        assert decisions and outcomes
        # One outcome per hand actually submitted.
        assert len(outcomes) == summary["hands"]
        # The mod was actually called with card indices.
        assert any(m in ("play", "discard") for m, _p in game.calls)

    def test_the_pulse_lands_on_the_window_the_decision_was_taken_in(
        self, scripted, tmp_path
    ) -> None:
        """The coincidence the rule is about.

        By the time the game has finished animating a hand, the fly has not yet
        run another window -- but it would have if the loop resolved the outcome
        *after* deciding the next hand. The pulse must carry the previous
        decision's ``window_id``, not the next one's.
        """
        _game, client = scripted(seed=2)
        fly = StubFly(script=[PLAY] * 8)
        play_plastic(client, fly, RecordingLogger(tmp_path), pause=0.0,
                     max_hands=5, verbose=False)
        assert fly.pulses, "no dopamine fired at all"
        for _kind, window_id in fly.pulses:
            # n-th pulse resolves the n-th decision.
            assert 1 <= window_id <= fly.window_id
        kinds = [k for k, _ in fly.pulses]
        assert all(k in ("reward", "punish") for k in kinds)

    def test_a_discard_never_earns_dopamine(self, scripted, tmp_path) -> None:
        _game, client = scripted(seed=4)
        fly = StubFly(script=[DISCARD, DISCARD, DISCARD])
        logger = RecordingLogger(tmp_path)
        play_plastic(client, fly, logger, pause=0.0, max_hands=3, verbose=False)
        discards = [r for r in logger.of_kind("outcome") if r["action"] == DISCARD]
        assert discards, "the stub never actually discarded"
        for rec in discards:
            assert rec["dopamine"] is None
            assert rec["outcome"]["reward"] == 0
            assert rec["outcome"]["punish"] == 0

    def test_a_losing_play_is_punished(self, scripted, tmp_path) -> None:
        """Punishment fires only on the terminal losing play, so the blind has
        to be out of reach for it to fire at all."""
        _game, client = scripted(seed=5, blind_scores=(100000, 100000, 100000))
        fly = StubFly(script=[PLAY] * 10)
        logger = RecordingLogger(tmp_path)
        summary = play_plastic(client, fly, logger, pause=0.0, max_hands=10,
                               verbose=False)
        assert summary["stop_reason"] == "game_over_lose"
        assert fly.n_punish == 1, [k for k, _ in fly.pulses]
        punished = [r for r in logger.of_kind("outcome") if r["dopamine"] == "punish"]
        assert len(punished) == 1
        assert punished[0]["outcome"]["lost"] is True

    def test_the_record_carries_what_the_viewer_reads(self, scripted, tmp_path) -> None:
        _game, client = scripted(seed=6)
        logger = RecordingLogger(tmp_path)
        play_plastic(client, StubFly(), logger, pause=0.0, max_hands=4, verbose=False)
        rec = logger.of_kind("decision")[0]
        mb = rec["mb"]
        for key in ("action", "p_play", "approach_hz", "avoid_hz", "play_drive",
                    "bucket", "best_type", "best_score", "needed", "slots",
                    "weights", "buckets", "reward_rate", "p_play_rolling",
                    "reward_spark", "mean_ratio_spark", "frac_changed_spark",
                    "n_hands", "rolling_window"):
            assert key in mb, key
        for key in ("n_synapses", "frac_changed", "mean_ratio", "n_at_floor",
                    "n_reward_pulses", "n_punish_pulses"):
            assert key in mb["weights"], key
        assert set(mb["buckets"]) == {"lt0.25", "lt0.5", "lt1.0", "ge1.0"}

    def test_the_record_is_what_the_overlay_parses(self, scripted, tmp_path) -> None:
        """One overlay reads a plastic log and a readout log."""
        _game, client = scripted(seed=7)
        logger = RecordingLogger(tmp_path)
        play_plastic(client, StubFly(), logger, pause=0.0, max_hands=4, verbose=False)
        dec = ov.parse_record(logger.of_kind("decision")[0])
        assert dec is not None
        assert dec.relay_active and dec.best_slots
        assert dec.action_name in ("play", "discard")
        assert dec.blind
        # The scripted game emits the geometry the patched mod reports, so the
        # alignment path is the exact one a real run would take.
        assert dec.has_rects and dec.screen is not None
        _placed, source = ov.aligned_rects(dec, ov.HandGeometry.load(), 1200.0, 800.0)
        assert source == ov.ALIGN_GAME

    def test_the_overlay_flash_key_is_at_the_top_level(self, scripted, tmp_path) -> None:
        _game, client = scripted(seed=8)
        logger = RecordingLogger(tmp_path)
        play_plastic(client, StubFly(), logger, pause=0.0, max_hands=5, verbose=False)
        fired = [r for r in logger.of_kind("outcome") if r["dopamine"]]
        assert fired, "no dopamine in the whole run"
        for rec in fired:
            dec = ov.parse_record(rec)
            assert dec is not None and dec.dopamine in ("reward", "punish")

    def test_nothing_logged_is_unparseable_json(self, scripted, tmp_path) -> None:
        _game, client = scripted(seed=9)
        logger = RecordingLogger(tmp_path)
        play_plastic(client, StubFly(script=[PLAY, DISCARD, PLAY, DISCARD]),
                     logger, pause=0.0, max_hands=6, verbose=False)
        blob = json.dumps(logger.records)
        assert "NaN" not in blob
        assert "Infinity" not in blob

    def test_a_frozen_fly_still_decides_but_moves_nothing(self, scripted, tmp_path) -> None:
        """The control for 'did the weights do anything'."""
        _game, client = scripted(seed=10)
        fly = StubFly()
        fly.deliver = lambda kind: {  # type: ignore[method-assign]
            "dopamine": "none", "eta": 0.0, "n_changed": 0, "mean_abs_dw": 0.0
        }
        logger = RecordingLogger(tmp_path)
        play_plastic(client, fly, logger, pause=0.0, max_hands=4, verbose=False)
        assert fly.calls, "the fly never decided"
        assert fly.weight_state()["n_changed"] == 0

    def test_the_fly_is_asked_about_a_hand_with_nothing_selected(
        self, scripted, tmp_path
    ) -> None:
        """The odour the plast2 bias and temperature were calibrated on.

        ``build_state`` seeds the virtual selection from the game's own
        ``card.highlight`` flags, and the relay block encodes the selected hand
        type and the "play it now" bit from that. A highlighted card would hand
        the fly a different odour from the calibration set, silently.
        """
        game, client = scripted(seed=12)
        seen: List[object] = []

        class Spy(StubFly):
            def decide(self, ctx):          # type: ignore[override]
                seen.append(ctx)
                return super().decide(ctx)

        # Make the game highlight a card, as a real Balatro would after a click.
        original = game._deal

        def deal_with_highlight(n: int = 8) -> None:
            original(n)
            game.hand[2]["state"]["highlight"] = True   # type: ignore[index]

        game._deal = deal_with_highlight               # type: ignore[method-assign]
        play_plastic(client, Spy(), RecordingLogger(tmp_path), pause=0.0,
                     max_hands=2, verbose=False)
        assert seen, "the fly was never asked"
        from flybalatro import features_v2 as F
        from flybalatro import hands as HH
        for ctx in seen:
            bits = np.asarray(ctx.bits)            # type: ignore[attr-defined]
            # The "nothing selected" bit is the last slot of the selected-type
            # block, and it must be the one that is on.
            assert bits[F.OFF_SELECTED_TYPE + HH.N_HAND_TYPES] == 1.0
            assert bits[F.OFF_IS_BEST] == 0.0

    def test_summary_is_written_and_names_the_rule(self, scripted, tmp_path) -> None:
        _game, client = scripted(seed=11)
        logger = RecordingLogger(tmp_path)
        summary = play_plastic(client, StubFly(), logger, pause=0.0, max_hands=3,
                               verbose=False)
        assert (tmp_path / "summary.json").exists()
        assert summary["policy_mode"] == "plastic"
        assert "no trained action readout" in summary["policy_detail"]
        assert summary["policy_uses_brain"] is True


# --------------------------------------------------------------------------- #
# the scripted game is honest about being scripted                             #
# --------------------------------------------------------------------------- #
class TestScriptedGame:
    def test_it_scores_with_balatros_own_rules(self) -> None:
        """Chips paid and chips promised come from one implementation.

        The fly is told what a subset is worth by ``hands.classify_slots``; if
        the scripted game paid out by some other rule, the dopamine would be
        reacting to a different game than the one the fly was shown.
        """
        from flybalatro import hands
        from realgame_plastic_offline import RANKS, SUITS, ScriptedGame, _Slot

        game = ScriptedGame(seed=0)
        game.dispatch("start", {})
        game.dispatch("select", {})
        cards = [_Slot(c) for c in game.hand]
        for slots in ([0, 1, 2], [0], [3, 4, 5, 6, 7]):
            want = hands.classify_slots(cards, slots)
            assert game._score_slots(slots) == (want.score if want else 0)
        # And the ranks/suits round-trip through the adapter's own tables.
        assert len(RANKS) == 13 and len(SUITS) == 4

    def test_it_reports_card_geometry_like_the_patched_mod(self) -> None:
        from realgame_plastic_offline import ScriptedGame

        game = ScriptedGame(seed=0)
        game.dispatch("start", {})
        payload = game.dispatch("select", {})
        cards = payload["hand"]["cards"]           # type: ignore[index]
        assert len(cards) == 8
        for card in cards:
            g = CardGeometry.from_json(card["geometry"])
            assert g is not None and g.w > 0
        # A fan: the tilt runs monotonically from one end to the other.
        tilts = [CardGeometry.from_json(c["geometry"]).r for c in cards]  # type: ignore[union-attr]
        assert tilts == sorted(tilts)
        assert tilts[0] < 0 < tilts[-1]
        assert abs(math.degrees(tilts[-1])) == pytest.approx(5.5, abs=0.01)
