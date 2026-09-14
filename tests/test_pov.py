"""The POV overlay: fan geometry, the log feed, and the calibration.

Nothing here touches AppKit, Quartz or a running game. The geometry is the same
code the NSWindow draws with and the same code ``scripts/pov_calibrate.py``
paints onto a real screenshot, so a green run here is what makes
``outputs/pov/overlay_calibration_check.png`` evidence about the live overlay
rather than about a separate drawing routine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flybalatro.realgame import overlay as ov
from flybalatro.viewer.server import chosen_confidence, slot_confidences

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "realgame" / "log.jsonl"
MEASURED = ROOT / "outputs" / "pov" / "measured_rects.json"

#: Box centre vs measured card centre, in pixels of the 2418 px canvas. The
#: settled hand is what the live overlay meets; ``balatro_fly.png`` caught one
#: card mid-slide and that card alone is ~27 px out. See docs/POV.md.
TOL_SETTLED_PX = 5.0
TOL_MIDSLIDE_PX = 28.0
TOL_TOP_PX = 6.0


@pytest.fixture(scope="module")
def geom() -> ov.HandGeometry:
    return ov.HandGeometry.load()


# ----------------------------------------------------------------- geometry
def test_geometry_loads_with_a_reference_canvas(geom: ov.HandGeometry) -> None:
    assert geom.reference_w > 0 and geom.reference_h > 0
    assert geom.reference_image.endswith(".png")
    assert 0.0 < geom.card_w < 0.25
    assert 0.0 < geom.center_x < 1.0


def test_slot_rects_are_monotone_in_slot_index(geom: ov.HandGeometry) -> None:
    for n in range(1, 9):
        rects = geom.hand_rects(n, width=1920, height=1080)
        xs = [r.x for r in rects]
        assert xs == sorted(xs)
        if n > 1:
            assert all(b.x > a.x for a, b in zip(rects, rects[1:]))


def test_every_slot_rect_is_inside_the_window(geom: ov.HandGeometry) -> None:
    for w, h in ((2418, 1570), (1280, 720), (3840, 2160), (960, 600)):
        for n in range(1, 9):
            for slot in range(n):
                for selected in (False, True):   # a raised card must fit too
                    rect = geom.slot_rect(slot, n, selected=selected, width=w, height=h)
                    assert rect.inside(w, h), (w, h, n, slot, selected, rect)


def test_selected_cards_are_raised(geom: ov.HandGeometry) -> None:
    for slot in range(8):
        plain = geom.slot_rect(slot, 8, width=2418, height=1570)
        up = geom.slot_rect(slot, 8, selected=True, width=2418, height=1570)
        assert up.y < plain.y                       # smaller y == higher on screen
        assert up.x == pytest.approx(plain.x)
        assert up.w == pytest.approx(plain.w)


def test_the_fan_is_centred_and_arched(geom: ov.HandGeometry) -> None:
    rects = geom.hand_rects(8, width=2418, height=1570)
    mid = (rects[0].x + rects[-1].right) / 2.0
    assert mid == pytest.approx(geom.center_x * 2418, abs=1.0)
    # the middle of the fan sits higher than its ends
    assert rects[3].y < rects[0].y
    assert rects[4].y < rects[7].y
    assert rects[0].y == pytest.approx(rects[7].y, abs=0.5)


def test_rects_scale_with_the_window(geom: ov.HandGeometry) -> None:
    small = geom.slot_rect(2, 8, width=1209, height=785)
    big = geom.slot_rect(2, 8, width=2418, height=1570)
    assert big.x == pytest.approx(small.x * 2, rel=1e-9)
    assert big.h == pytest.approx(small.h * 2, rel=1e-9)


def test_fewer_cards_never_spread_past_one_card_width(geom: ov.HandGeometry) -> None:
    for n in range(2, 9):
        assert geom.pitch(n) <= geom.card_w * geom.max_pitch_ratio + 1e-12
    assert geom.pitch(8) < geom.card_w          # a full hand overlaps
    assert geom.pitch(1) == 0.0


def test_bad_slots_are_rejected(geom: ov.HandGeometry) -> None:
    with pytest.raises(ValueError):
        geom.slot_rect(8, 8)
    with pytest.raises(ValueError):
        geom.slot_rect(-1, 8)
    with pytest.raises(ValueError):
        geom.slot_rect(0, 0)


# -------------------------------------------------------------- calibration
def test_calibration_reproduces_the_measured_rects() -> None:
    """The shipped parameters against the pixel measurements, both images."""
    report = json.loads(MEASURED.read_text())
    by_image = {r["image"]: r for r in report["residuals"]}
    settled = by_image["balatro_fly_firsthand.png"]
    midslide = by_image["balatro_fly.png"]

    assert settled["max_abs_d_centre"] <= TOL_SETTLED_PX
    assert settled["max_abs_d_top"] <= TOL_TOP_PX
    # balatro_fly.png caught one card still sliding into its slot after a
    # discard; every other slot there is within the settled tolerance too.
    assert midslide["max_abs_d_centre"] <= TOL_MIDSLIDE_PX
    assert midslide["max_abs_d_top"] <= TOL_TOP_PX
    settled_others = sorted(abs(r["d_centre"]) for r in midslide["cards"])[:-1]
    assert max(settled_others) <= 8.0
    for row in settled["cards"] + midslide["cards"]:
        assert len(row["modelled_box"]) == 4

    # The measured tilts are the fan: monotone, symmetric, about +-5.5 degrees.
    tilts = [r["measured_angle_deg"] for r in settled["cards"]]
    assert tilts == sorted(tilts)
    assert tilts[0] < -4.0 and tilts[-1] > 4.0


def test_calibration_report_matches_the_shipped_parameters(geom: ov.HandGeometry) -> None:
    """The residual table was computed with the parameters that ship."""
    report = json.loads(MEASURED.read_text())
    settled = next(r for r in report["residuals"]
                   if r["image"] == geom.reference_image)
    w, h = settled["canvas"]
    modelled = geom.hand_rects(len(settled["cards"]), width=w, height=h)
    for row, rect in zip(settled["cards"], modelled):
        assert rect.x == pytest.approx(row["modelled_box"][0], abs=0.1)
        assert rect.y == pytest.approx(row["modelled_box"][1], abs=0.1)


# -------------------------------------------------------------- the log feed
def test_parse_record_skips_rejected_calls() -> None:
    assert ov.parse_record({"t": 1.0, "decision": 3, "error": "bad", "action_index": 7}) is None
    assert ov.parse_record({"t": 1.0, "decision": 3}) is None
    assert ov.parse_record([]) is None                      # type: ignore[arg-type]
    # Lua emits [] for an empty object; that is not a state.
    assert ov.parse_record({"decision": 1, "state": []}) is None


def test_read_log_parses_the_real_run() -> None:
    decisions = ov.read_log(LOG)
    assert len(decisions) >= 200
    assert all(d.n_cards == 8 for d in decisions)
    assert {d.action_name for d in decisions} & {"play", "discard"}
    first = decisions[0]
    assert first.blind
    assert first.required > 0
    assert 0 <= first.plays <= 8


def test_cards_decode_to_the_glyphs_on_the_screenshot() -> None:
    """The hand in balatro_fly.png, read off the pixels: K C Q H J D 8 H 8 C 7 S 5 S 5 D."""
    want = ((11, 1), (10, 2), (9, 3), (6, 2), (6, 1), (5, 0), (3, 0), (3, 3))
    dec = next(
        d for d in ov.read_log(LOG)
        if tuple((c.rank_index, c.suit_index) for c in d.cards) == want and d.score == 292
    )
    assert [c.rank for c in dec.cards] == ["K", "Q", "J", "8", "8", "7", "5", "5"]
    assert [c.suit for c in dec.cards] == ["♣", "♥", "♦", "♥", "♣", "♠", "♠", "♦"]
    assert [c.red for c in dec.cards] == [False, True, True, True, False, False, False, True]
    assert dec.cards[0].label == "K♣ rank 11 suit 1"
    assert dec.best_type == "Two Pair"
    assert dec.best_slots == (3, 4, 6, 7)


def test_dopamine_is_optional_and_read_from_either_place() -> None:
    base = json.loads(next(iter(open(LOG, encoding="utf-8"))))
    assert ov.parse_record(base).dopamine is None
    assert ov.parse_record({**base, "dopamine": "reward"}).dopamine == "reward"
    assert ov.parse_record({**base, "mb": {"dopamine": "punish"}}).dopamine == "punish"
    assert ov.parse_record({**base, "dopamine": "none"}).dopamine is None


def test_follow_reads_appended_lines(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    lines = [l for l in open(LOG, encoding="utf-8")][:3]
    path.write_text("".join(lines), encoding="utf-8")
    feed = ov.follow(path, poll=0.0, from_start=True)
    got = [next(feed) for _ in range(3)]
    assert [d.index for d in got] == [1, 2, 3]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(lines[0])
        fh.flush()
    assert next(feed).index == 1


def test_follow_survives_a_truncated_final_line(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    line = next(iter(open(LOG, encoding="utf-8")))
    path.write_text(line + line[:40], encoding="utf-8")   # half-flushed tail
    feed = ov.follow(path, poll=0.0, from_start=True)
    assert next(feed).index == 1


def test_watch_latest_reads_one_record(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    obj = json.loads(next(iter(open(LOG, encoding="utf-8"))))
    path.write_text(json.dumps(obj), encoding="utf-8")
    assert next(ov.watch_latest(path, poll=0.0)).index == 1


# -------------------------------------------------------------- annotations
def test_default_annotation_boxes_only_the_best_subset(geom: ov.HandGeometry) -> None:
    """One emphasis. No grey box on every card, no label over every card, and
    the hand type is named once in the chrome instead of under each card."""
    dec = next(d for d in ov.read_log(LOG) if d.best_slots and d.relay_active)
    ann = ov.build_annotation(dec, geom, 2418, 1570)
    best = [b for b in ann.boxes if b.kind == "best"]
    assert len(best) == len(dec.best_slots)
    assert not [b for b in ann.boxes if b.kind == "card"]
    assert not [b for b in ann.boxes if b.label]
    assert ann.headline == f"{dec.best_type.upper()} \u00b7 {dec.best_score}"
    assert ann.caption == ov.CAPTION
    assert all(b.rect.inside(2418, 1570, slack=16) for b in ann.boxes)


def test_labels_flag_restores_the_calibration_view(geom: ov.HandGeometry) -> None:
    dec = next(d for d in ov.read_log(LOG) if d.best_slots and d.relay_active)
    ann = ov.build_annotation(dec, geom, 2418, 1570, labels=True, all_cards=True)
    cards = [b for b in ann.boxes if b.kind == "card"]
    assert len(cards) == dec.n_cards
    assert all(b.label == c.label for b, c in zip(cards, dec.cards))
    assert all(b.thickness > 0 for b in cards)


def test_headline_says_whether_the_best_hand_reaches_the_blind(
    geom: ov.HandGeometry,
) -> None:
    for dec in ov.read_log(LOG):
        if not (dec.relay_active and dec.best_slots):
            continue
        ann = ov.build_annotation(dec, geom, 2418, 1570)
        assert ann.headline_ok == (dec.best_score >= dec.needed)


def test_chrome_sits_in_felt_the_game_leaves_empty(geom: ov.HandGeometry) -> None:
    """Above the hand fan, right of the sidebar, left of the deck."""
    dec = next(d for d in ov.read_log(LOG) if d.best_slots)
    ann = ov.build_annotation(dec, geom, 2418, 1570)
    top_card = min(r.y for r in
                   geom.hand_rects(dec.n_cards, width=2418, height=1570))
    assert ann.chrome.bottom < top_card
    assert ann.chrome.x > 0.249 * 2418          # right of the run-info sidebar
    assert ann.chrome.right < 0.847 * 2418      # left of the deck
    assert ann.chrome.y > 0.31 * 1570           # below the joker slot outlines


def test_chrome_moves_out_of_the_way_when_there_is_no_hand(
    geom: ov.HandGeometry,
) -> None:
    """Shop and round-eval fill the middle band with their own panels; with no
    cards there is nothing to draw there anyway."""
    dec = next(d for d in ov.read_log(LOG, with_cards_only=False) if not d.cards)
    ann = ov.build_annotation(dec, geom, 2418, 1570)
    assert ann.boxes == ()
    assert ann.chrome.bottom < 0.106 * 1570     # above the joker slot outlines
    assert ann.headline == "RELAY SILENT"


def test_live_feeds_can_pass_through_card_less_records(tmp_path: Path) -> None:
    obj = next(json.loads(l) for l in open(LOG, encoding="utf-8")
               if json.loads(l).get("state") and not json.loads(l)["state"]["hand"])
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    assert ov.LatestFile(path).poll() == []
    assert len(ov.LatestFile(path, with_cards_only=False).poll()) == 1


def test_selected_slots_get_a_bar_not_a_box(geom: ov.HandGeometry) -> None:
    dec = next(d for d in ov.read_log(LOG) if d.selected)
    ann = ov.build_annotation(dec, geom, 2418, 1570)
    sel = [b for b in ann.boxes if b.kind == "selected"]
    assert len(sel) == len(set(dec.selected))
    assert all(b.fill and b.thickness == 0 for b in sel)
    slots = geom.hand_rects(dec.n_cards, dec.selected, width=2418, height=1570)
    for bar in sel:
        assert bar.rect.h < 0.25 * slots[bar.slot].h


def test_action_labels() -> None:
    assert ov.action_label("play")[0] == "PLAY"
    assert ov.action_label("discard")[0] == "DIG"
    assert ov.action_label("select_card[2]")[0] == "SELECT 2"
    assert ov.action_label("next_round")[0] == "NEXT ROUND"


def test_content_rect_strips_the_titlebar() -> None:
    assert ov.content_rect((100.0, 50.0, 800.0, 628.0), 28.0) == (100.0, 78.0, 800.0, 600.0)
    assert ov.content_rect((0.0, 0.0, 640.0, 360.0), 0.0) == (0.0, 0.0, 640.0, 360.0)


# ------------------------------------------------- viewer per-slot readout
def test_slot_confidences_are_a_softmax_over_legal_select_actions() -> None:
    scores = np.zeros(109, dtype=np.float32)
    scores[:8] = np.array([3.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mask = np.zeros(109, dtype=np.int8)
    mask[[0, 1, 3]] = 1
    slots = slot_confidences(scores, mask, 8, True)
    assert len(slots) == 8
    confs = [s["conf"] for s in slots if s["conf"] is not None]
    assert len(confs) == 3
    assert sum(confs) == pytest.approx(1.0)
    assert slots[0]["conf"] > slots[3]["conf"] > slots[1]["conf"]
    assert slots[2]["conf"] is None and slots[2]["legal"] is False


def test_slot_confidences_are_none_without_logits() -> None:
    scores = np.random.default_rng(0).normal(size=109).astype(np.float32)
    mask = np.ones(109, dtype=np.int8)
    slots = slot_confidences(scores, mask, 8, False)
    assert all(s["conf"] is None and s["logit"] is None for s in slots)
    assert all(s["legal"] for s in slots)


def test_chosen_confidence() -> None:
    scores = np.full(109, -50.0, dtype=np.float32)
    scores[70] = 10.0
    mask = np.zeros(109, dtype=np.int8)
    mask[[70, 71]] = 1
    assert chosen_confidence(scores, mask, 70, True) == pytest.approx(1.0, abs=1e-6)
    assert chosen_confidence(scores, mask, 71, True) < 1e-6
    assert chosen_confidence(scores, mask, 70, False) is None
    assert chosen_confidence(scores, np.zeros(109, dtype=np.int8), 70, True) is None


def test_log_tail_is_non_blocking(tmp_path: Path) -> None:
    """The overlay polls between AppKit pumps; an empty poll must return, not wait."""
    path = tmp_path / "log.jsonl"
    lines = [l for l in open(LOG, encoding="utf-8")][:2]
    path.write_text(lines[0], encoding="utf-8")
    tail = ov.LogTail(path, from_start=True)
    assert [d.index for d in tail.poll()] == [1]
    assert tail.poll() == []                      # nothing new, returns at once
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(lines[1])
    assert [d.index for d in tail.poll()] == [2]


def test_log_tail_handles_a_rotated_file(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    lines = [l for l in open(LOG, encoding="utf-8")][:3]
    path.write_text("".join(lines), encoding="utf-8")
    tail = ov.LogTail(path)                       # starts at the end
    assert tail.poll() == []
    path.write_text(lines[0], encoding="utf-8")   # a new run truncates it
    assert [d.index for d in tail.poll()] == [1]


def test_latest_file_reads_once_per_write(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    obj = json.loads(next(iter(open(LOG, encoding="utf-8"))))
    path.write_text(json.dumps(obj), encoding="utf-8")
    latest = ov.LatestFile(path)
    assert [d.index for d in latest.poll()] == [1]
    assert latest.poll() == []
    assert ov.LatestFile(tmp_path / "nope.json").poll() == []


def test_calibrate_demo_hand_is_the_screenshot_hand() -> None:
    """--calibrate over balatro_fly.png must show that screenshot's own cards."""
    dec = ov._demo_decision(ov.HandGeometry.load())
    assert [c.rank + c.suit for c in dec.cards] == [
        "K♣", "Q♥", "J♦", "8♥", "8♣", "7♠", "5♠", "5♦"]
    assert dec.best_slots == (3, 4, 6, 7) and dec.best_type == "Two Pair"
    assert set(dec.selected) <= set(dec.best_slots)
    assert dec.action_name == "select_card[6]"
