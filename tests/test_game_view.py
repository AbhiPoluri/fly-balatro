"""The pure parts of the embedded game view.

Everything here runs without a window server, without Quartz and without
Balatro: the window *lookup* is a filter over plain dicts, the geometry is
arithmetic, and the wire framing is ``struct``. What cannot be tested this way
-- that ``CGWindowListCreateImage`` returns the game's pixels -- is verified by
hand and recorded in ``docs/POV.md``.
"""

from __future__ import annotations

import math
import struct

import pytest

from flybalatro.realgame import overlay as ov
from flybalatro.viewer import game_view as gv


# --------------------------------------------------------------------------- #
# window lookup                                                                #
# --------------------------------------------------------------------------- #
def win(owner: str, wid: int, x: float = 0.0, y: float = 0.0,
        w: float = 100.0, h: float = 100.0, layer: int = 0) -> dict:
    return {
        "kCGWindowOwnerName": owner,
        "kCGWindowNumber": wid,
        "kCGWindowLayer": layer,
        "kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h},
    }


class TestPickWindow:
    def test_finds_the_steam_build(self):
        ref = gv.pick_window([win("Finder", 1), win("Balatro", 42, 151, 70, 1209, 813)])
        assert ref is not None
        assert (ref.window_id, ref.owner) == (42, "Balatro")
        assert ref.bounds == (151.0, 70.0, 1209.0, 813.0)

    def test_finds_bare_love(self):
        ref = gv.pick_window([win("love", 7, 0, 0, 800, 600)])
        assert ref is not None and ref.window_id == 7

    def test_owner_match_is_case_insensitive(self):
        assert gv.pick_window([win("BALATRO", 3)]) is not None

    def test_ignores_other_apps(self):
        assert gv.pick_window([win("Safari", 1), win("Terminal", 2)]) is None

    def test_ignores_non_zero_layers(self):
        """Layer 0 is a normal window; the overlay itself floats above it and
        must never be picked as the thing to capture."""
        assert gv.pick_window([win("Balatro", 9, layer=3)]) is None

    def test_largest_area_wins(self):
        infos = [win("Balatro", 1, w=400, h=300), win("Balatro", 2, w=1209, h=813)]
        ref = gv.pick_window(infos)
        assert ref is not None and ref.window_id == 2
        assert gv.pick_window(list(reversed(infos))).window_id == 2

    def test_zero_sized_and_idless_windows_are_skipped(self):
        assert gv.pick_window([win("Balatro", 5, w=0, h=0)]) is None
        assert gv.pick_window([win("Balatro", 0)]) is None

    def test_malformed_entries_do_not_raise(self):
        infos = [
            {"kCGWindowOwnerName": "Balatro"},                       # no id
            {"kCGWindowOwnerName": "Balatro", "kCGWindowNumber": "x"},
            {"kCGWindowOwnerName": "Balatro", "kCGWindowNumber": 4,
             "kCGWindowBounds": {"Width": "wide"}},
            win("Balatro", 11, w=640, h=480),
        ]
        ref = gv.pick_window(infos)
        assert ref is not None and ref.window_id == 11

    def test_empty_and_none_lists(self):
        assert gv.pick_window([]) is None
        assert gv.pick_window(None) is None

    def test_owner_names_are_the_overlay_s(self):
        """One list, so the capture and the NSWindow overlay can never disagree
        about which window is the game."""
        assert gv.OWNER_NAMES is ov.OWNER_NAMES


# --------------------------------------------------------------------------- #
# title bar                                                                    #
# --------------------------------------------------------------------------- #
class TestContentHeight:
    def test_uses_the_canvas_the_game_reported(self):
        # The measured Steam build: 813 pt of window, 785 pt of canvas.
        assert gv.content_height(813, 785) == 785.0

    def test_fullscreen_has_no_title_bar(self):
        assert gv.content_height(785, 785) == 785.0

    def test_falls_back_when_the_game_has_said_nothing(self):
        assert gv.content_height(813, None) == pytest.approx(813 - gv.DEFAULT_TITLEBAR)

    def test_rejects_a_stale_canvas_from_a_resized_window(self):
        """A record from a much smaller window would carve a 300 px 'title bar'
        out of the picture; the default is wrong by a few pixels, that is wrong
        by a third of the screen."""
        assert gv.content_height(813, 500) == pytest.approx(813 - gv.DEFAULT_TITLEBAR)

    def test_rejects_a_canvas_taller_than_the_window(self):
        assert gv.content_height(785, 900) == pytest.approx(785 - gv.DEFAULT_TITLEBAR)

    def test_never_returns_a_non_positive_height(self):
        assert gv.content_height(10, None) >= 1.0


# --------------------------------------------------------------------------- #
# game units -> captured pixels                                                #
# --------------------------------------------------------------------------- #
class TestGameToImage:
    def test_identity_when_the_frame_is_the_canvas(self):
        rect = (317.96, 459.03, 112.59, 151.19)
        assert gv.game_to_image(rect, (1209.0, 785.0), (1209.0, 785.0)) == \
            pytest.approx(rect)

    def test_downscaled_frame(self):
        """900 px of frame for 1209 draw units: everything scales by 0.7444."""
        out = gv.game_to_image((100.0, 200.0, 50.0, 60.0), (1209.0, 785.0),
                               (900.0, 584.0))
        s = 900.0 / 1209.0
        assert out[0] == pytest.approx(100.0 * s)
        assert out[2] == pytest.approx(50.0 * s)
        assert out[1] == pytest.approx(200.0 * 584.0 / 785.0)

    def test_aspect_is_preserved_by_the_real_pipeline(self):
        """The capture crops only the title bar and resizes by one factor, so
        both axes share a scale and a rotated box stays a rectangle."""
        sx = 900.0 / 1209.0
        sy = 584.0 / 785.0
        assert sx == pytest.approx(sy, abs=1e-3)

    def test_a_card_centre_maps_to_the_same_fraction(self):
        rect = (317.96, 459.03, 112.59, 151.19)
        out = gv.game_to_image(rect, (1209.0, 785.0), (900.0, 584.0))
        assert (out[0] + out[2] / 2) / 900.0 == \
            pytest.approx((rect[0] + rect[2] / 2) / 1209.0)

    def test_rejects_a_degenerate_canvas(self):
        with pytest.raises(ValueError):
            gv.game_to_image((0, 0, 1, 1), (0.0, 785.0), (900.0, 584.0))


# --------------------------------------------------------------------------- #
# wire framing                                                                 #
# --------------------------------------------------------------------------- #
class TestFraming:
    def test_roundtrip(self):
        blob = gv.pack_frame(17, 1234.5, 900, 584, b"\xff\xd8jpegbytes")
        assert gv.unpack_frame(blob) == (17, 1234.5, 900, 584, b"\xff\xd8jpegbytes")

    def test_header_is_24_bytes(self):
        assert gv.HEADER_BYTES == 24
        assert len(gv.pack_frame(0, 0.0, 1, 1, b"")) == 24

    def test_a_game_frame_is_recognised(self):
        assert gv.is_game_frame(gv.pack_frame(1, 0.0, 4, 4, b"xx"))

    @pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
    def test_a_spike_substep_is_not_mistaken_for_one(self, index):
        """The collision that would matter: the spike path sends ``<II``
        (sub-step index, count) on the same socket, and a JPEG pushed into the
        sub-step list would corrupt the 50 ms wave. The tag is outside the
        index range, so one word separates them."""
        substep = struct.pack("<II", index, 3) + b"\x00" * 12
        assert not gv.is_game_frame(substep)
        with pytest.raises(ValueError):
            gv.unpack_frame(substep)

    def test_the_tag_is_not_a_plausible_substep_index(self):
        assert gv.FRAME_TAG > 4
        assert gv.FRAME_TAG <= 0xFFFFFFFF

    def test_short_and_empty_blobs(self):
        assert not gv.is_game_frame(b"")
        assert not gv.is_game_frame(b"\x01\x02")
        with pytest.raises(ValueError):
            gv.unpack_frame(struct.pack("<I", gv.FRAME_TAG))

    def test_sequence_wraps_rather_than_raising(self):
        seq, _t, _w, _h, _j = gv.unpack_frame(
            gv.pack_frame(0x1_0000_0003, 0.0, 1, 1, b""))
        assert seq == 3


# --------------------------------------------------------------------------- #
# what the panel is told when there is nothing to show                         #
# --------------------------------------------------------------------------- #
class TestDegradation:
    def test_every_non_live_state_has_a_sentence(self):
        for state in (gv.STATE_NO_WINDOW, gv.STATE_DENIED, gv.STATE_UNAVAILABLE,
                      gv.STATE_OFF):
            assert gv.STATE_TEXT[state].strip()

    def test_a_fresh_capture_reports_no_window(self):
        cap = gv.GameCapture()
        assert cap.state == gv.STATE_NO_WINDOW
        assert cap.describe()["window_id"] is None
        assert cap.describe()["text"] == gv.STATE_TEXT[gv.STATE_NO_WINDOW]

    def test_describe_is_json_safe(self):
        import json
        json.dumps(gv.GameCapture().describe())


# --------------------------------------------------------------------------- #
# the boxes, in the canvas the capture is a picture of                         #
# --------------------------------------------------------------------------- #
def _record(rects: bool = True) -> dict:
    """A minimal decision record with the mod's geometry block."""
    hand = []
    for slot in range(3):
        card = {"slot": slot, "rank_index": 8 + slot, "suit_index": 0}
        if rects:
            card["rect"] = {"x": 300.0 + 120.0 * slot, "y": 459.0,
                            "w": 112.59, "h": 151.19, "r": -0.068 + 0.068 * slot,
                            "moving": False}
        hand.append(card)
    return {
        "index": 4,
        "t": 0.0,
        "action_name": "play",
        "action_description": "play hand slots [0, 1]",
        "relayed_hand_analysis": {
            "active": True, "best_type": "Pair", "best_score": 60,
            "best_mask": [1, 1, 0], "best_slots": [0, 1],
            "selected_type": "none", "selected_slots": [0],
            "selected_score": 0, "selected_is_best": False,
            "needed": 300, "score_bucket": "lt0.25", "n_bits_on": 5,
        },
        "state": {
            "hand": hand, "selected_indices": [0], "plays": 3, "discards": 3,
            "score": 0, "required_score": 300, "blind": "Small Blind",
            "ante": 1, "round": 1, "stage": 1, "money": 4, "n_jokers": 0,
            "screen": {"width": 1209.0, "height": 785.0,
                       "pixel_width": 2418.0, "pixel_height": 1570.0},
        },
    }


class TestPovBlock:
    def _block(self, rec: dict) -> dict:
        from flybalatro.viewer.realgame_source import pov_block
        dec = ov.parse_record(rec)
        assert dec is not None
        return pov_block(dec, ov.HandGeometry.load())

    def test_boxes_are_in_the_canvas_the_game_reported(self):
        block = self._block(_record())
        assert block["canvas"] == [1209.0, 785.0]
        assert block["align"] == "game"
        best = [b for b in block["boxes"] if b["kind"] == "best"]
        assert len(best) == 2                      # best_mask is slots 0 and 1
        assert best[0]["x"] == pytest.approx(300.0)
        assert best[0]["w"] == pytest.approx(112.59)

    def test_the_outline_is_rotated_with_the_card(self):
        best = [b for b in self._block(_record())["boxes"] if b["kind"] == "best"]
        assert best[0]["angle"] == pytest.approx(-0.068)
        assert best[1]["angle"] == pytest.approx(0.0, abs=1e-6)

    def test_the_selected_slot_gets_a_bar_not_a_box(self):
        bars = [b for b in self._block(_record())["boxes"] if b["kind"] == "selected"]
        assert len(bars) == 1 and bars[0]["fill"] is True
        assert bars[0]["h"] < 20.0

    def test_falls_back_to_the_model_without_rects(self):
        block = self._block(_record(rects=False))
        assert block["align"] == "model"
        assert block["boxes"], "the fitted fan still places every slot"
        # Fractions of the canvas, so they are still drawable on the capture.
        for box in block["boxes"]:
            assert 0.0 <= box["x"] < block["canvas"][0]

    def test_headline_says_whether_the_hand_clears_the_blind(self):
        block = self._block(_record())
        assert block["headline"] == "PAIR · 60"
        assert block["ok"] is False and "short" in block["note"]

    def test_an_outcome_record_draws_no_boxes(self):
        """The game is animating those cards away by then; a box from it would
        sit on a position nothing is at."""
        rec = _record()
        rec["kind"] = "outcome"
        rec["outcome"] = {"chips_gained": 60}
        assert self._block(rec)["boxes"] == []

    def test_scaling_a_box_onto_a_downscaled_frame(self):
        """What the browser does: divide by the canvas, multiply by the frame."""
        block = self._block(_record())
        best = [b for b in block["boxes"] if b["kind"] == "best"][0]
        out = gv.game_to_image((best["x"], best["y"], best["w"], best["h"]),
                               tuple(block["canvas"]), (900.0, 584.0))
        assert out[0] == pytest.approx(300.0 * 900.0 / 1209.0)
        assert math.isclose(out[2] / out[3], best["w"] / best["h"], rel_tol=1e-3)

    def test_block_is_json_safe(self):
        import json
        json.dumps(self._block(_record()))
