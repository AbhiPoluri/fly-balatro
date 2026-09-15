# The fly's POV: detection boxes, and what they honestly are

Three screens, one idea, one caption that none of them will let you turn off:

> **The fly does not see pixels. Boxes show the hand analysis it is given
> (computed outside the brain) and what it decided.**

1. **Viewer panel**: `FLY POV` in the web viewer, a mock camera frame over the
   eight hand cards. Toggle with the `TABLE / POV` switch in the header, or `v`.
2. **Real-game overlay**: `flybalatro/realgame/overlay.py`, a transparent,
   always-on-top, click-through window that draws the same boxes over the actual
   Balatro window while the fly plays it.
3. **The embedded game panel** (section 5): the real Balatro window captured
   and streamed *into* the dashboard, boxes drawn on the captured pixels, so the
   whole demo is one browser window instead of a game, a browser and a floating
   overlay. This is the one to run.

> Shell blocks below use `$REPO` for this checkout: set
> `REPO=$(git rev-parse --show-toplevel)` once per shell.

None of them is computer vision. `flybalatro/hands.py` enumerates every subset of the
dealt cards and scores it with Balatro's rules, **outside the brain**;
`flybalatro/glomerular.py` turns the answer into 32 relay bits and drives 32
whole ORN glomeruli with them; the readout on the spike counts (or, in
live-learning mode, the fly's own MBONs) returns an action. A box on a card
means *the harness told the fly something about that slot* or *the fly's output
named that slot*. The YOLO look is a joke about the Deadlock meme; the data
under it is the real thing, and the frame says so.

---

## 1. The viewer panel

Run it the usual way; nothing about the launch changes.

```bash
source .venv/bin/activate
python -m flybalatro.viewer.server --port 8770                 # v3 readout
python -m flybalatro.viewer.server --port 8770 --policy plastic # live learning
```

Open `http://127.0.0.1:8770/`, click **POV** in the header (or press `v`). The
POV panel replaces the table panel in the same grid cell and widens the column
to an even split, so the brain point cloud stays on screen.

What is drawn, top to bottom:

| Element | What it is |
| --- | --- |
| HUD, top left | `best: <hand type> · <score> / <needed> needed`, then plays / discards / round score, then blind, ante and how many of the 32 relay bits are lit. All of it is the relayed analysis, not a reading of the screen. |
| thin grey box + label on every card | one per dealt slot, labelled `K♣ rank 11 suit 1`: the glyph plus the two indices `flybalatro/features.py` encoded. Odd slots take the upper label row so neighbouring labels never collide. |
| amber box + `TWO PAIR · 118` | the best subset `hands.py` found, and its score. Every card of the subset carries the label. |
| teal inner box | slots the fly has already selected. |
| bar under each slot | **readout modes:** softmax over the *legal* `select_card` logits: "which card would it pick next". A slot already selected has no such action left in the legal mask, so it shows a full teal bar marked `picked` rather than a silent gap. A policy with no logits (random, heuristic) shows no bars and says so.<br>**live-learning mode:** there are no per-slot logits at all (the fly makes one binary play/dig call), so `P(play)` is painted on the best subset and `1 − P(play)` on the slots the dig rule throws, with a caption saying it is one call, not eight. |
| big label, bottom left | `PLAY`, `DIG`, `select 6`, and the confidence: softmax of the chosen action over every legal action (readout), or `P(play)` from the MBON drive (plastic). |
| caption | drawn by the canvas code on every frame. It is not markup and cannot be styled away. |
| glomerulus strip | below the frame: the 32 relay bits as named ORN glomeruli with their receptor counts, lit when driven. That strip **is** the whole input. |

Server side, `flybalatro/viewer/server.py` adds two fields to the `decision`
frame: `slots` (per dealt slot: `legal`, `logit`, `conf`) and `confidence`.
Both are `null` when the policy has no logits.

Verified at 1440×900: `document.documentElement.scrollWidth == clientWidth`,
no horizontal scroll, no console errors.
`outputs/pov/viewer_pov.png` is the panel in readout mode and
`outputs/pov/viewer_pov_plastic.png` in live-learning mode.

---

## 2. The real-game overlay

```bash
cd "$REPO" && source .venv/bin/activate

# boxes only, over a synthetic full hand: the hand in balatro_fly.png
python -m flybalatro.realgame.overlay --calibrate

# step through a recorded run; no game, no server needed
python -m flybalatro.realgame.overlay --replay outputs/realgame/log.jsonl --pause 1.0

# a run by the LEARNING fly, so the dopamine flash is live
python -m flybalatro.realgame.overlay --replay outputs/realgame_offline/log.jsonl --pause 1.0

# live, while the fly plays: follows outputs/realgame/latest.json
python -m flybalatro.realgame.overlay
```

### What it draws, and what it stopped drawing

Four things and nothing else:

1. **one amber outline** on the cards the fly is about to play, rotated with
   the card, so it sits on the card's own border;
2. **one line** naming the hand and whether it clears what is still needed,
   amber when it is short and green when it is enough;
3. **the decision**, `PLAY` or `DIG`, large;
4. one dim line of context, and the honesty caption at the very bottom edge.

All of 1–4 except the outlines are drawn inside `CHROME_RECT`, a band of felt
that Balatro leaves empty during a hand: right of the run-info sidebar (whose
border is at x = 0.249), below the joker and consumable slot outlines (they end
at y = 0.31), above the hand fan (`top_y` = 0.589) and left of the deck
(x = 0.847), so the overlay never covers anything the game drew. On a screen
with no hand (shop, cash-out) it moves to the strip above the joker slots, which
is empty on every screen.

Deleted, because the earlier version was unreadable
(`outputs/pov/live_test_2.png`): the `K♣ rank 11 suit 1` label over every card,
the thin grey box on every card, the hand-name caption repeated under every card
of the subset, the stray amber line at the window edge, the tint over the whole
game, and the giant verdict block that sat over the Options / Ante / Round
buttons. `--labels` and `--all-cards` still exist for
`scripts/pov_calibrate.py`'s check images, where the point of the picture *is*
that every label matches its pixels; they are off by default.

An **outcome** record, the one the plastic loop writes when the game has
resolved a hand, draws the flash, the headline and no card boxes at all. It
carries the board the decision was taken on, but by the time it is written the
game is animating that hand away and dealing its replacements, so boxes from it
would sit on stale positions for the whole pause between hands.

Useful flags: `--rect x,y,w,h` places the window by hand instead of searching for
the game (screen points, top-left origin); `--titlebar N` is how much of the
Quartz bounds is title bar (default 28, use `0` for fullscreen); `--seconds N`
stops after N seconds; `--geometry PATH` swaps the calibration file.

**Finding the window.** `CGWindowListCopyWindowInfo` with
`kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements`, owner
name `Balatro` or `love`, `kCGWindowLayer == 0`, largest area wins. Those bounds
are Quartz: **top-left origin, and they include the title bar**, so
`content_rect()` strips `--titlebar` off the top and `OverlayWindow._to_cocoa()`
flips y against the main `NSScreen` height for Cocoa's bottom-left origin.

**The window.** Borderless `NSWindow`, `clearColor` background, `setOpaque_(False)`,
`NSFloatingWindowLevel`, `setIgnoresMouseEvents_(True)` (clicks go through to the
game), no shadow, `CanJoinAllSpaces | Stationary | FullScreenAuxiliary |
IgnoresCycle`, and the app runs at `NSApplicationActivationPolicyAccessory` so it
never takes focus or shows in the Dock. AppKit and Quartz are imported **lazily**,
inside the window class and the lookup function, so `tests/test_pov.py` imports
the geometry and the log parser in a plain interpreter.

**Where the data comes from.** Live, it polls `outputs/realgame/latest.json`,
which `play.py` writes to a temp file and `os.replace`s: atomic, so a poller
never sees half an object (`docs/REALGAME_INSTALL.md` §6 documents it as the
subscribe mechanism). If that file is absent it tails `log.jsonl` instead. Both
readers (`LatestFile`, `LogTail`) are **non-blocking** on purpose: the loop has
to keep pumping the AppKit event queue between decisions or the window stops
repainting. Records with an `error` and no `state` (rejected API calls) are
skipped.

**Dopamine flash.** If a record carries `dopamine: "reward"` or `"punish"` (top
level, or inside `mb`), the frame flashes green or red for ~0.9 s. The three
recorded v3-readout runs emit no dopamine, so it is dormant on those; on a
`--plastic` run it is not. Checked on both arms against the offline runs:
`outputs/realgame_offline/log.jsonl` fires 12 green flashes over 30 hands and
`outputs/realgame_offline_lose/log.jsonl` fires the red one, on the terminal
losing play.

---

## 3. Dynamic alignment: asking the game where the cards are

The overlay used to place every box from a **fitted static model** of Balatro's
card fan (section 4). A model drifts whenever reality differs from it: fewer
than eight cards in hand, cards mid-deal or mid-discard, a raised card, a
resized window. The user's requirement was that it "dynamically line up", so the
model is now the *fallback*, and the first choice is to ask the game.

### The game already knows, and now it says

Balatro is LÖVE. Every card is a `Moveable` carrying two transforms in game
units: **`T` is the target** and **`VT` is the visible one**, which eases toward
`T` over a few frames. `engine/node.lua`'s draw path takes `self.VT or self.T`,
so a card that is mid-deal, mid-discard or being raised is *drawn* at `VT`, and
an overlay that wants to line up with what is on screen has to read
`VT`, not `T`.

Game units become LOVE pixels exactly as `Node:put_focused_cursor` does it:

```
pixel = (VT.<x|y> + G.ROOM.T.<x|y>) * (G.TILESCALE * G.TILESIZE)
size  =  VT.<w|h>                   * (G.TILESCALE * G.TILESIZE)
```

with `VT.r` a rotation in radians about the rect's own centre. The stock
BalatroBot mod does not report any of it, so **~120 additive lines** were added
to one file of it, `src/lua/utils/gamestate.lua` (both the installed copy and
the vendored one, byte-identical). Every card gains an optional `geometry`
block carrying `rect` (that arithmetic already done) plus the raw `vt`, `t`,
`room` and `unit` values so the mapping can be re-derived or re-fitted on the
Python side without another Lua change, and a `moving` flag that is true while
`VT` has not caught up with `T`. The state gains an optional `screen` block with
the tile scaling, the room rect and LÖVE's `getDimensions()` /
`getPixelDimensions()` / `getDPIScale()`, because the overlay is positioned in
window **points** while the rects are in **pixels**. Both are `pcall`-wrapped
and optional; nothing existing was removed, renamed or reordered.
`docs/REALGAME_INSTALL.md` §2 records the edit and how to revert just it.

**A Lua file is only read when the game launches**, so this takes effect on the
next start of Balatro and not before.

### What the overlay does with it

`overlay.aligned_rects(dec, geom, width, height)` returns
`({slot: (rect, angle)}, source)` with two sources tried in order:

| source | when | what it costs | how good |
| --- | --- | --- | --- |
| `game` | every dealt slot in the record carries a `rect`, and the record carries a `screen` | one multiply per card | exact by construction; there is no model to be wrong |
| `model` | anything is missing: a log recorded before the mod carried geometry, or a stock mod | the fitted fan in `pov_geometry.json` | 4.8 px on a settled 8-card hand, 4.2 px on a short one, **20 px** mid-animation and 78 px on a card in the air (measured below) |

It is deliberately all-or-nothing per hand: if even one card lacks a rect the
whole hand falls back, because mixing the two sources across one fan would
misalign exactly one card and say nothing about it. The chosen source is carried
on the `Annotation` as `align`, because the two are not equally trustworthy.

Knowing the angle also lets the box be drawn **rotated with the card**, so it
sits on the card's own border. The old axis-aligned box had to be 225 px wide to
cover a 211 px card tilted 5.5°; the tilted box is the card's width.

### What is and is not verified

Balatro was **not running** for any of the work described in this section (it
exited before it started), so the Lua was written against the game's own
decompiled `engine/node.lua` and `engine/moveable.lua`, syntax-checked, and
installed, but was not executed at the time of writing. What that left:

> **Correction (2026-09-13).** The Lua has since executed. The game was
> relaunched at 18:51:50 and seven `--plastic` sessions ran against it between
> 19:07 and 19:33 (`outputs/realgame/logs/2026-09-13T18-51-50/`,
> `outputs/realgame/log_gameview_runs.jsonl`). Every decision record from those
> sessions carries the per-card `rect {x, y, w, h, r, moving}` and the `screen`
> block that `patches/balatrobot-gamestate.patch` alone adds, so open item 1
> below, whether the mod emits the fields at all, is settled. Item 4 is settled
> in part: `width` 1209 against `pixel_width` 2418, so the two sources do differ.
> Nothing here shows that the boxes land correctly on a moving card; that is
> still open.

* **Verified.** The Python consumer, on rects fed in by hand: parsing,
  the all-or-nothing fallback, hands of 1/2/3/5/7/8 cards, malformed and
  missing geometry, and the scaling arithmetic onto an overlay window of a
  different size. `tests/test_realgame_plastic.py::TestAlignment`, 10 tests.
* **Verified on real pixels.** `outputs/pov/overlay_v2.png` is the real
  `NSWindow` over the real screenshot `outputs/realgame/balatro_fly_firsthand.png`,
  driven through the `game` path with the rects set to the card positions
  **measured off those pixels** by `scripts/pov_calibrate.py`. The boxes land on
  the card borders (max centre error 0.00 px, which is tautological, since the
  rects *are* the measured positions; what it shows is that the parse-and-scale
  path adds no error of its own, and what the result looks like when the rects
  are right). On the same frame the fitted model is 4.26 px out.
* **NOT verified.** Five things, all of which need the game running *and* a
  restart to load the new Lua:
  1. that the mod emits the fields at all;
  2. that `VT` (rather than `T`) is what a hand card is drawn at in
     practice;
  3. that `card.container` is `G.ROOM` for hand cards; if Balatro's `CardArea`
     sets the container to itself, the Lua adds the room offset twice and every
     box is displaced by a constant. The raw `room` values are exported
     alongside `rect` precisely so this is diagnosable from the first live
     record rather than needing another Lua change;
  4. that the room→pixel formula holds on a live window. In particular the mod
     reports both `getDimensions()` and `getPixelDimensions()`, and the overlay
     scales by the **former**: a card rect is `VT * G.TILESCALE * G.TILESIZE`,
     and `G.TILESCALE` is derived in `love.resize(w, h)` from the same `w` that
     `getDimensions()` returns, so the rects live in draw units. Scaling by the
     backing store instead would halve every box on a high-DPI window and look
     perfectly correct on a non-Retina one. `conf.lua` does not set
     `t.window.highdpi`, but the mod's own `screenshot` endpoint returns a
     2418 px image for what looks like a ~1209 pt window, so the two probably
     *do* differ here and this is the branch that matters;
  5. that `VT.r`'s sign convention matches the overlay's rotation.

  Until the game is restarted the live overlay falls back to the model, exactly
  as it did before.

### Measured: how far off each path is

Frames were cut out of `outputs/realgame/balatro_fly.mov` (28 s of the fly
actually playing, whose Balatro canvas is the 2418×1570 reference at scale 1.0)
and the cards found with `scripts/pov_calibrate.py`'s own outline detector.
`scripts/pov_align_measure.py` does it; `outputs/pov/dynamic_align.json` has the
per-slot table and `outputs/pov/align_case_{a,b,c}.png` draw it, pink for the
measured outline and centre, grey/amber for the modelled box. **Those three
images were rendered against the model as it was *before* the short-hand fix
below**, which is what makes `align_case_b.png` worth looking at: three cards
packed at a 165 px pitch with the modelled boxes spread across 225 px and the
outer two sitting visibly off their cards. Numbers are the
**fitted model's** box centre minus the **measured** card centre, in pixels of
the 2418 px canvas:

| case | what | max \|dx\| | median \|dx\| | max \|dy\| |
| --- | --- | --- | --- | --- |
| A | settled 8-card hand | **4.8** | 3.8 | 2.3 |
| A0 | the screenshot the model was fitted on | 4.3 | 2.4 | 2.0 |
| B | settled **3-card** hand | **64.3** → **4.2** | 56.5 → 3.9 | 10.7 |
| B2 | a second 3-card hand, cross-check | 61.0 → **2.0** | 58.8 → 0.6 | 12.7 |
| C | 8-card hand **mid-animation** | **19.7** | 10.8 | **78.3** on the airborne card |

**Case B was the model's worst failure and it is now fixed.** `docs/POV.md` used
to say that `n != 8` was untested extrapolation, and it was wrong in the
expected direction but by a lot: the old model *spread* a short fan to fill
`span`, wanting a 225.4 px pitch, while Balatro **keeps the 164.65 px
pitch and centres the shorter fan**. Two independent 3-card hands measure
165–171 px. `HandGeometry.pitch` now holds the pitch fixed, which takes the
short-hand error from 64 px to 4.2 px and leaves every 8-card number
bit-identical. `max_pitch_ratio` survives as an upper clamp that can no longer
bind.

Case C is what no model can fix. One card was in the air; its measured centre is
**78 px** above where any fan model puts it, and the seven that had landed are
still 10–20 px out because the fan had not closed up. That is the case the
game-reported `VT` path answers exactly and neither a model nor a detector can.

### Measured: what the live detector costs, and where it fails

Running the outline detector on every overlay frame was option (b). It works,
and it is not good enough to be the primary path:

| | ms, best of 7, on this M2 Pro |
| --- | --- |
| decode one 2418×1570 PNG | 60.3 |
| `Shot.border_lines()` alone | 33.4 |
| a full snap (decode + mask + Hough + per-card top/bottom) | 140.3 |
| a full snap excluding the decode | 80.0 |

80 ms of work per frame, against a game capture that would have to be taken
first, is affordable at 2–3 Hz and not at frame rate. Worse, it is least
reliable exactly when it is most needed. Over 430 frames spanning two
animations, the detector returned the wrong number of outlines on about 30% of
them (25 frames found *zero*), and **the airborne card in case C is not
detected at all**: Balatro's move animation rotates it far past the sweep's
±9° and squashes it horizontally, so it casts no near-vertical outline, and the
detector silently assigns the *right* border of its neighbour to that slot. A
detector that fails on moving cards cannot be the fix for moving cards.

So the shipped order is (a) then (c): ask the game, and fall back to the model.
The detector stays where it belongs: offline calibration, in
`scripts/pov_calibrate.py`.

### What emitting the spikes costs

The viewer animates the decision window as five 10 ms sub-steps. The real-game
loop already computes that window, so it **emits the neuron indices that fired
in each sub-step** rather than re-simulating anything. Measured on this M2 Pro,
40 paired windows on the same brain, the two arms interleaved so background load
hits both equally (two other compute-heavy jobs were saturating the machine
throughout, so the absolute numbers are inflated and only the paired difference
is meaningful):

| | median |
| --- | --- |
| one 50 ms window | 168.0 ms |
| five 10 ms sub-steps | 167.7 ms |
| **paired difference** | **+0.9 ms (+0.5%)** |

Writing the sidecar adds ~0.1 ms and 58 KB per decision. Against a loop that
pauses a full second between hands, neither is close to a bottleneck.

And the split is not an approximation: `Brain.step` carries its own kernel
cursor across calls and only `reset_counts` differs, so the spike counts at the
end are **bit-identical** either way, verified directly, including that the
resulting MBON drive is identical to six decimal places. Recording cannot change
what the fly decides.

## 4. Calibration: the fallback model, and where the cards are

### The parameters

`flybalatro/realgame/pov_geometry.json`, all **fractions of the game canvas** so
one file works at any window size:

| key | value | meaning |
| --- | --- | --- |
| `center_x` | 0.548749 | centre of the fan. Not 0.5: the deck sits to the right of the hand. |
| `span` | 0.569892 | left edge of slot 0's box to the right edge of slot 7's |
| `card_w` | 0.093224 | width of the box (225.4 px on the reference canvas) |
| `card_h` | 0.190127 | box height at the ends of the fan |
| `top_y` | 0.589490 | top of an end card's box |
| `arc_lift` | 0.009753 | how much higher the middle of the fan sits (15.3 px) |
| `arc_shrink` | 0.005917 | the middle boxes are shorter: the end cards are rotated ±5.5°, so their axis-aligned box is taller |
| `raise` | 0.030 | **not measured**, see below |
| `max_pitch_ratio` | 1.0 | with fewer than 8 cards the fan spreads, but never past one card width of pitch |

`pitch = (span − card_w) / (n − 1)`, clamped by `max_pitch_ratio`. At n = 8 that
is 164.65 px on the 2418 px reference canvas; the cards are 211 px wide, so they
overlap by 46 px, which is what the game does. Slot *k*'s box is
`x = centre − fan/2 + pitch·k`, `y = top_y − arc_lift·(1 − u²) − raise·selected`
with `u` running −1 … +1 across the fan. The box is deliberately **wider than
the card** (225 vs 211): it is axis-aligned and the end cards are tilted 5.5°,
so it has to cover a rotated rectangle.

**`raise` cannot be measured.** The BalatroBot API has no card-selection
endpoint, so the adapter carries the fly's selection itself and submits the
indices at play time; the real game **never highlights** it
(`docs/REALGAME_INSTALL.md` §4). The number is nominal and exists so the overlay
can show the fly's *virtual* selection. Nothing in the game moves.

**n ≠ 8 was extrapolation, and it was wrong.** Every one of the 253 logged
decisions has 8 cards or none, so a short hand's layout was never checked
against the real game. When it finally was, by cutting 3-card frames out of
`balatro_fly.mov` (§3), the spread-out model turned out to be 60 px off. Balatro
holds the pitch and centres the shorter fan. `pitch()` now does the same and the
error is 4.2 px. `max_pitch_ratio` is kept as an upper clamp and no longer
binds.

### The measurement

**The cards are rotated.** That is the fact the whole calibration turns on: the
fan runs from −5.5° at slot 0 to +5.5° at slot 7, measured, monotone. A
column-wise search for "the left edge of card *k*" is therefore ill-posed (a
tilted border is not a column), and a first attempt at it found face-card art
instead of card edges on two slots.

What *is* well posed is the card's **outline**, which Balatro draws in one flat
colour, a light blue-grey near `(186, 196, 212)`: low saturation, clearly darker
than the white body, clearly lighter than the art and the felt. `Shot.border_lines()`
masks exactly that, then runs a small Hough transform: shear the band by
`tan θ` for θ ∈ [−9°, +9°] in half-degree steps, sum columns, keep peaks with
≥ 170 votes out of ~300 rows. Each peak is one card outline as a **line**: its x
at the band's mid-height and its angle. On both screenshots that returns nine
lines: the eight left borders and the rightmost card's right border, which is
where the card's width comes from.

The rest is arithmetic on well-defined quantities:

* **card centre** = left border x at mid-height + width/2. Rotation-invariant
  (the centre of a rotated rect is the centre of its bounding box), so this is
  the quantity a box should be compared against. Left edges are not comparable.
* `centre_k = centre0 + pitch·k` by least squares.
* box width = the fan's pixel extent (from the white-card mask) minus `7·pitch`.
* top and bottom per card from the white mask's profile over that card's own
  columns, discarding columns more than 12 px below the card's median bottom;
  otherwise the white `8/8` hand counter drags the middle card's box 35 px down.
* the arc from the median of the per-card lift estimates.

```bash
python scripts/pov_calibrate.py            # measure, fit, write JSON + check images
python scripts/pov_calibrate.py --detect   # print every detected outline and stop
```

### The fit, and what it is worth

Fitted on **`outputs/realgame/balatro_fly_firsthand.png`**, a *settled* 8-card
hand. Checked against **`outputs/realgame/balatro_fly.png`**, which the brief
named and which turns out to have caught one card **mid-slide**: the 8♥ in slot
3 sits 27 px right of its slot and is tilted **+3.5°** where its neighbours are
−1.5° and −0.5°, i.e. it is out of the fan's own arc. Its left *and* right
borders are both detected, because it has a 30 px gap on either side while every
other card overlaps its neighbour. It was still easing into place after a
discard. Fitting on that frame would have baked a transient into the parameters,
so it is the check, not the fit.

Measured card centre vs modelled box centre, in pixels of the 2418 px canvas
(`outputs/pov/measured_rects.json` has the full table with tops, bottoms and
tilts):

| slot | tilt | settled: card → box | Δ | mid-slide: card → box | Δ |
| --- | --- | --- | --- | --- | --- |
| 0 | −5.5° | 750.5 → 750.6 | +0.1 | 745.5 → 750.6 | +5.1 |
| 1 | −2.0° | 919.5 → 915.2 | −4.3 | 909.5 → 915.2 | +5.7 |
| 2 | −1.5° | 1079.5 → 1079.9 | +0.4 | 1074.5 → 1079.9 | +5.4 |
| 3 | −1.5° | 1241.5 → 1244.5 | +3.0 | 1271.5 → 1244.5 | **−27.0** |
| 4 | +0.0° | 1407.5 → 1409.2 | +1.7 | 1401.5 → 1409.2 | +7.7 |
| 5 | +2.5° | 1570.5 → 1573.8 | +3.4 | 1567.5 → 1573.8 | +6.4 |
| 6 | +5.0° | 1739.5 → 1738.5 | −1.0 | 1733.5 → 1738.5 | +5.0 |
| 7 | +5.5° | 1906.5 → 1903.2 | −3.3 | 1902.5 → 1903.2 | +0.7 |

**Stated tolerance.** Settled hand: **≤ 4.3 px** horizontally and **≤ 3.0 px**
vertically: 0.18 % of the canvas width, 2 % of a card. That is what the live
overlay meets, because the live overlay meets settled hands. Mid-slide frame:
**≤ 7.7 px** on seven slots and **27.0 px** on the one card that was still
moving, with **≤ 5.0 px** vertically throughout; even at 27 px the box still
covers 88 % of that card. `tests/test_pov.py` asserts 5 px (settled), 28 px
(mid-slide worst slot), 8 px (mid-slide, every other slot) and 6 px vertical,
and checks that the measured tilts are monotone and reach ±4°. If any of those
fail, Balatro's layout or the window size changed.

Evidence: `outputs/pov/overlay_calibration_check.png` (mid-slide, the frame the
brief named) and `outputs/pov/overlay_calibration_check_settled.png`. Both draw
the **measured** outline as the slanted pink line it actually is, plus a pink
tick at the measured card centre, with the **modelled** box in grey/amber on
top of the real screenshot, so the residual is something you can look at. Both
use the real log record for that exact hand, matched by
`(rank_index, suit_index)` against the glyphs on screen, so the labels are
checked against pixels too.

### Re-measuring when the window size changes

The parameters are fractions, so a plain resize of the Balatro window needs
nothing. Re-measure when the **aspect ratio** changes (x is normalised by width
and y by height independently, and that is exact only at the reference aspect,
2418 : 1570), or when a Balatro update moves the hand.

1. Take a screenshot of a **settled** full hand through the mod's own endpoint;
   it needs no window id and no screen-recording permission:
   ```bash
   curl -s -X POST http://127.0.0.1:12346 -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"screenshot","params":{"path":"'"$REPO"'/outputs/realgame/newhand.png"},"id":1}'
   ```
   Settled matters: wait a second after the cards land. `--detect` will tell you
   if it was not: a settled fan has monotone tilts and evenly spaced outlines.
2. Check `BAND` in `scripts/pov_calibrate.py` still brackets the cards (it must
   exclude the `n/n` hand counter and the Play/Discard buttons, which are also
   white), point the two `Shot(...)` lines at the new file, and add its hand to
   the `HANDS` table so the check image gets the right labels.
3. `python scripts/pov_calibrate.py`, read the printed residuals, and **look at**
   `outputs/pov/overlay_calibration_check.png`. If the outline chain ever picks
   a wrong line, run `--detect`, choose the eight by hand and put them in the
   `OVERRIDES` table, which exists for that. Neither shipped screenshot needs it.
4. `pytest tests/test_pov.py`, adjusting `TOL_*` only if the new residuals are
   different, not to make a bad fit pass.

---

## 5. The game inside the dashboard

The demo used to be three windows: Balatro, a browser with the brain in it, and
a transparent overlay floating over the game. It is now one browser window. The
server captures the Balatro window and streams it into the page as a panel, and
the fly's amber outline is drawn **on the captured pixels**.

```bash
cd "$REPO" && source .venv/bin/activate

# 1. the game, with the mod (docs/REALGAME_INSTALL.md section 4)
# 2. the fly
python -m flybalatro.realgame.play --plastic --ante-end 8 --pause 2.0
# 3. the one window
python -m flybalatro.viewer.server --port 8772 --source realgame
```

Then open `http://127.0.0.1:8772/`. Flags: `--game-fps` (default 12),
`--game-width` (default 900, never upscaled past the window's own size) and
`--no-game-view`.

The panel exists only under `--source realgame` on a **live** feed. In the
headless mode there is no real window to capture. Under `--replay` there is
often one, and capturing it would put a live picture of Balatro under boxes
taken from a hand played minutes or days ago, which looks current and is not; so
`--replay` turns the panel off and says so in the log. Either way the section is
not rendered at all rather than rendered empty.

### How the pixels are taken, and why that way

`CGWindowListCreateImage(CGRectNull, kCGWindowListOptionIncludingWindow, wid,
kCGWindowImageBoundsIgnoreFraming | kCGWindowImageNominalResolution)`, in
`flybalatro/viewer/game_view.py`. Capturing **a window by id** rather than a
screen rectangle is the whole point: the browser is on top of the game for the
entire demo and the frames are still correct. The window is found the same way
`overlay.py` finds it (owner `Balatro` / `love`, layer 0, largest area), and
`pick_window()` keeps the window **id** as well as the bounds.

The two image flags are measured choices. All three rows below produce the same
900 px stream through the same code; only the source image differs (median of
60 frames, this M2 Pro):

| option | source image | capture + scale + crop | JPEG |
| --- | --- | --- | --- |
| default | 2554×1734, and the drop shadow is **in** the image, so every rect is displaced | 58.7 ms | 1.9 ms |
| `IgnoreFraming` | 2418×1598, the window, at backing-store resolution | 54.1 ms | 1.9 ms |
| `IgnoreFraming \| NominalResolution` | **1209×785**, the window in points | **34.2 ms** | 1.9 ms |

The panel is ~470 CSS px wide and the stream is 900, so the Retina backing store
is downscaled away immediately; capturing it costs 20 ms a frame to do that. It
also breaks the title-bar crop, which is the second reason to prefer points: the
game reports `screen.height` in points, and against a backing-store image the
difference is 841 rather than 28, which is rejected and falls back to the
default; the middle row above does come out 1598 tall where the canvas
is 1570.

**The title bar comes off using the game's own number.** At nominal resolution
the captured image is the window bounds in points, and the mod reports
`screen.height` (the LOVE canvas in the same points) on every decision, so the
title bar is `image_height − screen.height`: 28 on the windowed Steam build and
0 in fullscreen, with nothing assumed. After the crop the frame **is** the LOVE
canvas, one image point per draw unit, which is what makes the boxes a
multiplication instead of a fit.

**It renders into its own bitmap rather than reading the CGImage.** The obvious
route, `CGDataProviderCopyData` then `Image.frombuffer` over those bytes,
**leaks 3.85 MB per frame**: 400 captures take the process from 321 MB to
1,862 MB, one window-sized RGBA buffer each, and neither `del`, `gc.collect()`
nor an `objc.autorelease_pool` recovers any of it. Capturing without ever
touching the data's bytes does not grow at all, so it is taking a Python buffer
over the `__NSCFData` that does it. At 12 fps that is 46 MB/s and the panel
would have been unusable inside a minute. So there is one `CGBitmapContext`
backed by a `bytearray` this module owns, reused for the life of the stream, and
`CGContextDrawImage` blits each capture into it: 200 frames move the process by
9 MB. Drawing into a context sized to the *output* also does the downscale and
the title-bar crop in Quartz, in C, which removes PIL's resize from the
per-frame path.

### Measured cost

12 fps target, 900 px wide, q72, over 2,954 frames of a live run:

| | |
| --- | --- |
| capture + downscale + crop | **40.7 ms** (34.2 ms unloaded) |
| JPEG encode | **1.8 ms** |
| bytes per frame | **35–55 KB** (scene-dependent; 54 KB on a full table) |
| achieved | **11.7 fps** of a 12 fps target |
| server RSS | 789 MB, flat over 3,000 frames (780 MB without the panel) |
| browser | 30 fps, the cap |

**It costs the brain path nothing.** The control is exact: a second viewer was
already following the same live feed with no game view. Over the same run it
reports `frame_ms` 0.32 / `emit_ms` 0.31 / `max_emit_ms` 2.04 ms; the one with
the capture running reports 0.31 / 0.30 / 1.36. The capture runs as its own
asyncio task, one frame in flight, never queued (a tick that runs long does
not take a frame), and both Quartz and PIL drop the GIL, so the ~150 ms
numba window is not blocked by it.

### The boxes

The server calls the **same** `overlay.build_annotation` the NSWindow overlay
calls, on the same record, and serialises the result into a `pov` block on the
`state` frame in the game's own canvas units
(`realgame_source.pov_block`). The browser scales it by
`frame_width / canvas_width` and strokes each box rotated about its centre. No
second geometry, no re-fit, and `align` (`game` / `model`) rides along so the
panel says which it is looking at.

Two things are deliberately timed rather than immediate: the boxes and the
headline go up when the `state` frame lands (they are the harness's analysis,
which exists before the fly runs), and the `PLAY` / `DIG` verdict is held until
the 50 ms wave has finished playing in the cloud, so the game panel cannot
announce the decision before the brain that made it has finished firing. An
`outcome` frame clears the boxes, for the same reason `overlay.py` draws none on
one: the hand is being animated away.

### Do they line up? Measured.

`python scripts/pov_embed_measure.py --hands 4` captures the window at full
resolution while the mod reports a settled 8-card hand, runs
`scripts/pov_calibrate.py`'s outline detector on those exact pixels, and
compares. **This is the first time the mod's `VT` → pixels arithmetic has ever
been run against real pixels**; section 3 listed five things it could not
verify without the game. Four hands, 32 cards, in pixels of the 2418-wide
canvas (divide by 5.14 for the 470 CSS px the panel is drawn at):

| | max | median |
| --- | --- | --- |
| horizontal, box centre vs card centre | **22.3** | 5.6 |
| vertical, box top vs card top | **5.2** | 2.3 |

So: **vertical is exact** (2 px of 2418 is under half a CSS pixel on the panel),
and horizontally the box is on the card at the left of the fan and drifts to
about 4 CSS px by the eighth. What the four unverified assumptions turned into:

* `card.container` **is** `G.ROOM` for hand cards: the reported `room` matches
  `screen.room` exactly, so the offset is not added twice. Assumption 3 holds.
* The rects are in `getDimensions()` draw units, not the backing store: the
  frame is 2418 px for a 1209-unit canvas and the ratio is exactly 2.000.
  Assumption 4 holds, and it was the branch that mattered.
* `VT.r`'s sign is right: detected outline angles run −5.0° … +6.5° against
  reported −5.6° … +6.2°, monotone and matching within the detector's 0.5° step.
  Assumption 5 holds.
* **`VT.scale` was missing, and it was wrong by 5%.** Hand cards are drawn at
  `VT.scale = 0.95` about their own centre, and the mod reports the *unscaled*
  transform: a 225.2 px box on a card measured at 213. `CardGeometry` now
  carries `scale` (defaulting to 1.0, so every log written before this renders
  exactly as it did), `as_dict` emits it, and `aligned_rects` applies it before
  anything else. The outline is now 213.9 px against a detected 213.
* **The reported pitch is ~1.9% short and that is not fixed.** The mod places
  the fan at a 161.65 px pitch; the pixels measure 164.82, and
  `pov_geometry.json`'s independently fitted model measured 164.65 on a
  different screenshot months of work earlier. Two independent measurements
  agree against the transform. About a quarter of the gap is an artifact of
  measuring a rotated border at a fixed band height (±2.9 px at the ends of the
  fan); the rest is real and unexplained. It is the residual the table above
  reports, and on a settled hand it makes the game path *worse* than the fitted
  model at the ends of the fan (4.8 px), while remaining the only path that is
  right for a short hand, a card in the air, or a resized window, which is what
  it exists for.

`outputs/pov/embed_align.json` has the per-card table and
`outputs/pov/embed_align_*.png` the frames it was measured on.

### What the panel says when there is no game

Four states, each a plain sentence in place of the picture, never a stale frame
pretending to be current: `no_window` ("Balatro is not running — nothing to
capture"), `denied` (`CGPreflightScreenCaptureAccess` says no), `unavailable`
(pyobjc did not import) and `off`. `denied` is re-asked on every idle tick
rather than once per process, because the permission can be granted in System
Settings without restarting the viewer. A frame that stops arriving is labelled
`STALE · N.Ns` over the last one rather than left up silently, and the window id
is re-looked-up on every miss so restarting the game brings the panel back
without restarting the viewer.

The boxes carry a `decision N` tag for the opposite reason: the frame under them
is live and the record is not, so between a decision and the next capture the
game can raise a selected card or animate a played one out from under its box.
Naming the decision is how the panel says the boxes are a snapshot of one
instant and the picture is not.

### Layout

`body.gameview` widens the right column from 340 px to 500 px, and only while
the capture is on. 340 px puts Balatro's own UI text at 0.28× and it is not
readable; 500 px gives a 470 × 305 CSS panel backed by a 900 × 584 frame, which
is legible down to the card ranks and the blind requirement. The point cloud
keeps ~940 px at 1440 and stays the hero. Verified in a 1440×900 window:
`scrollWidth == clientWidth == 1440`, `scrollHeight == clientHeight == 900`, no
scrollbar on the page; the panel column scrolls inside itself as it already did.
`outputs/pov/dashboard_embedded.png` is the whole thing during a live run.

### Does this replace the NSWindow overlay?

For the dashboard demo, yes, and `overlay.py` is untouched and still works.
The embedded panel is better where it counts (one window, the rects and the
pixels from the same frame, boxes that cannot be knocked out of alignment by
moving the game window), and worse where the overlay is the point: full-screen,
native-resolution boxes on the real game for someone who wants to watch Balatro
rather than watch a brain. Keep both.

---

## 6. Verification without the game

Done on a machine with Balatro **not** running:

* `outputs/pov/viewer_pov.png` and `outputs/pov/viewer_pov_plastic.png`: the
  POV canvas at its native 1384×778, against the headless `pylatro` game on port
  8770, in the readout and live-learning modes. The page layout was checked
  programmatically at 1440×900 (`scrollWidth == clientWidth == 1440`,
  `scrollHeight == clientHeight == 900`, no horizontal scroll); the only console
  entries were WebSocket errors from restarting the server mid-session, with
  `ws.readyState === 1` afterwards; nothing from the POV code. The server was
  stopped afterwards.
* `outputs/pov/overlay_calibration_check.png` and `…_settled.png`: the primary
  alignment evidence, above.
* `outputs/pov/overlay_live.png`: the actual NSWindow, transparent and
  click-through, floating over `balatro_fly.png` opened in Preview:
  ```bash
  open -a Preview outputs/realgame/balatro_fly.png
  # Preview's Quartz bounds minus its 52 pt title+toolbar
  python -m flybalatro.realgame.overlay --calibrate --rect 4,89,1183,768 --seconds 30 &
  screencapture -x -R4,37,1183,820 outputs/pov/overlay_live.png
  ```
  This demonstrates the window mechanics (borderless, clear, floating,
  click-through, correctly sized to another app's window), and `screencapture`
  needed no new permission. Its alignment is **approximate**: Preview resamples
  the 2418 px image down to 1183 pt and adds its own chrome, so this is not the
  alignment evidence. The check images are.
* **`outputs/pov/overlay_v2.png`**: the same trick, but showing the *current*
  overlay over the settled 8-card frame, driven through the `game` alignment
  path with the rects set to the card positions measured off those exact pixels:

  ```bash
  open -a Preview outputs/realgame/balatro_fly_firsthand.png
  python -m flybalatro.realgame.overlay --replay <one-record.jsonl> \
      --rect 4,89,1183,768 --titlebar 0 --pause 30 --seconds 40 &
  screencapture -x -R4,37,1183,820 outputs/pov/overlay_v2.png
  ```

  Preview shows the 2418 px image at exactly the 1183 × 768 content rect
  (aspect 1.5404 against the image's 1.5401), so the boxes land where the model
  says they should. Compare it against `outputs/pov/live_test_2.png`, which is
  the overlay over the *real running game* before this work: that frame carries
  a `K♣ rank 11 suit 1` label on every card, a grey box on every card, `FULL
  HOUSE · 312` repeated under five of them, a green tint over the whole game, a
  stray amber rule down the window, and a `SELECT 0` block sitting on top of the
  Options / Ante / Round buttons. The new one carries two amber outlines, three
  lines of text in empty felt, and the caption.

What has **not** been verified, and cannot be without the game running: the live
follow path end to end (`LatestFile.poll()` against a `latest.json` that
`play.py` is actively writing), the Quartz lookup against a real Balatro window,
the `--titlebar 28` default for the LÖVE window, and everything in §3's
five-item list, above all whether the patched mod emits card geometry at all.
The dopamine flash *is* now verified on both arms, against the offline plastic
runs (§2).

---

## 7. Files

| Path | What |
| --- | --- |
| `flybalatro/realgame/overlay.py` | geometry, log feed, annotation, the NSWindow, the CLI |
| `flybalatro/realgame/pov_geometry.json` | the fitted parameters |
| `scripts/pov_calibrate.py` | measure → fit → write JSON and the check images |
| `flybalatro/viewer/static/app.js` | `PovView`, the panel |
| `flybalatro/viewer/static/index.html`, `style.css` | its markup and styles |
| `flybalatro/viewer/server.py` | `slot_confidences()`, `chosen_confidence()`, `GameStream` |
| `flybalatro/viewer/game_view.py` | window lookup by id, capture, downscale, JPEG, wire framing |
| `flybalatro/viewer/realgame_source.py` | `pov_block()`, the boxes, in the game's canvas units |
| `scripts/pov_embed_measure.py` | do the boxes land on the cards of the captured frame |
| `tests/test_pov.py` | 33 tests: geometry, calibration, feed, annotation, confidences |
| `tests/test_game_view.py` | 43 tests: window lookup, title bar, rect transform, frame framing, degradation, `pov_block` |
| `outputs/pov/` | screenshots, `measured_rects.json`, `embed_align.json` |

Dependencies: `pyobjc-core`, `pyobjc-framework-Cocoa`, `pyobjc-framework-Quartz`
(12.2.2) were already in `.venv`. `pillow 12.3.0` was added
(`uv pip install --python .venv/bin/python pillow`) for the calibration script's
measurement and check images; nothing the viewer or the overlay needs at runtime.
If pyobjc is ever missing:
`uv pip install pyobjc-framework-Cocoa pyobjc-framework-Quartz`.
