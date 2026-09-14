# Patches to vendored upstreams

`vendor/` is gitignored — it holds upstream checkouts that you clone yourself
(see [`docs/REALGAME_INSTALL.md`](../docs/REALGAME_INSTALL.md)). Two of them are
patched, and those patches are the part of this project that lives in someone
else's tree. They are kept here so a fresh clone can reproduce them.

Both are `git diff` output taken against the pinned upstream commit, so they
apply with `git apply` from the root of the respective checkout.

| patch | upstream | pinned commit |
|---|---|---|
| `balatro-rs.patch` | <https://github.com/evanofslack/balatro-rs> | `05de340fb075b94f63ccbb4ff6b68304f1ddfb0c` (2026-08-23, "Merge pull request #55 from evanofslack/vouchers") |
| `balatrobot-gamestate.patch` | <https://github.com/coder/balatrobot> | `e7c6db8a9ad88318f6e4128eefd6e61aafc94885` (2026-06-17, "docs(ci): version docs with mike…") |

## Applying

```bash
REPO=$(git rev-parse --show-toplevel)

git clone https://github.com/evanofslack/balatro-rs "$REPO/vendor/balatro-rs"
git -C "$REPO/vendor/balatro-rs" checkout 05de340fb075b94f63ccbb4ff6b68304f1ddfb0c
git -C "$REPO/vendor/balatro-rs" apply "$REPO/patches/balatro-rs.patch"

git clone https://github.com/coder/balatrobot "$REPO/vendor/balatrobot"
git -C "$REPO/vendor/balatrobot" checkout e7c6db8a9ad88318f6e4128eefd6e61aafc94885
git -C "$REPO/vendor/balatrobot" apply "$REPO/patches/balatrobot-gamestate.patch"
```

Then build the Python bindings (`maturin develop` inside
`vendor/balatro-rs/pylatro`) and install the Lua mod, both covered in
`docs/REPRODUCE.md`.

## `balatro-rs.patch` — 3 files, +148 / −16

### 1. `core/src/card.rs` — pyo3 getters (+97)

`Card` crossed the FFI boundary as an opaque handle, so Python could not read
what a card *was*. The patch adds `#[getter]`s for `rank_index`, `suit_index`,
`value`, `suit`, `chips` and `enhancement`, plus the small conversions behind
them. `flybalatro/hands.py` needs exactly these: it enumerates all 218 subsets
of the dealt cards and has to classify and score each one the way the engine
does, which means reading rank, suit and chip value off every card.

Purely additive to the Rust side. Nothing existing changes behaviour.

### 2. `pylatro/src/lib.rs` — `GameEngine.to_action` and GameState fields (+39)

`to_action` turns an action index from the mask back into the engine's `Action`,
which is what makes a *masked* policy expressible from Python: without it you
can sample an index but you cannot tell what you sampled. The extra GameState
fields (`stage`, blind and score state) are what the viewer and the real-game
adapter display and what `flybalatro/env.py` builds its 283 state bits from.

Also additive.

### 3. `core/src/generator.rs` — a real engine bug (+12 / −16)

**This one is a genuine upstream bug and we think it is worth upstreaming.**

`Game` has two places that decide whether a consumable is usable right now:

* `gen_actions_use_consumable`, the action *generator*, and
* `unmask_action_space_use_consumable`, the action *mask*.

They are supposed to agree. Upstream, the generator already reads

```rust
Stage::End(_) | Stage::TarotHand(_) | Stage::SpectralHand(_) | Stage::PackOpen()
…
!t.requires_hand() || (self.stage.is_blind() && selected_count >= t.min_targets() && …)
```

while the mask had drifted to

```rust
Stage::End(_) | Stage::TarotHand(_) | Stage::SpectralHand(_)      // PackOpen missing
…
if !t.requires_targets() { true } else if self.stage.is_blind() { … } else { true }
```

Two divergences follow. `PackOpen` is missing from the mask's early return, and
the `else { true }` arm unmasks a hand-touching Tarot or Spectral *outside* a
Blind — where `Game::use_consumable` (`core/src/game.rs:1167`, which uses
`requires_hand()`) then rejects it as `InvalidAction`. So the mask offers an
action the handler refuses, and a masked agent that trusts the mask crashes or
stalls. We measured it at roughly **1 in 3,000 episodes** in ante-1 play; it
would be much more frequent in a run that reaches the shop often.

The patch makes the mask a copy of the generator's own predicate, including
`PackOpen`. It removes a divergence rather than introducing a rule, and the
correct version is already in the file a few hundred lines above, which is the
strongest argument for sending it upstream: the project already believes the
patched logic, in the other half of the same pair.

## `balatrobot-gamestate.patch` — 1 file, +125 / −1

`src/lua/utils/gamestate.lua`, ~120 additive lines, every block `pcall`-wrapped
so a failure degrades to the mod's existing behaviour instead of breaking the
game.

Balatro is LÖVE, and every card is a `Moveable` carrying a target transform `T`
and a *visible* transform `VT` that eases toward it — `VT` is what the draw path
actually uses. The patch reports, for each card in hand, the `VT`-derived screen
rectangle and rotation plus the room/tile scaling the game is currently using.

This replaces a fitted static model of Balatro's card fan in
`flybalatro/realgame/overlay.py`, which drifted on short hands, mid-animation
frames, raised cards and window resizes. With the mod's own numbers the overlay
is exact by construction: `overlay.aligned_rects` prefers them and falls back to
the fitted fan when the fields are absent, so the mod and the consumer can be
upgraded independently.

Caveat, carried over from `docs/POV.md` §3: **the Lua side has been written,
installed and unit-tested but never executed.** A Lua file is only read when the
game launches and the game has not launched since. What is verified is the
Python consumer (10 tests over parsing, fallback, scaling and hands of 1–8
cards) and, on real pixels, that feeding it card positions measured off
`outputs/realgame/balatro_fly_firsthand.png` puts the boxes on the card borders
where the fitted model is 4.26 px out. What is *not* verified is that the mod
emits the fields at all.
