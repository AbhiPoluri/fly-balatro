"""Drive the real Balatro game (via the BalatroBot mod) from the fly connectome.

Three pieces:

* :mod:`flybalatro.realgame.client` - a typed JSON-RPC 2.0 client over the
  BalatroBot HTTP API. Method names mirror ``vendor/balatrobot/docs/api.md``.
* :mod:`flybalatro.realgame.adapter` - converts a BalatroBot ``gamestate``
  payload into the duck-typed object :func:`flybalatro.features.encode`
  consumes, maps the 109 pylatro action indices onto BalatroBot calls, and
  builds a legality mask with ``env.py``'s ``mask_noop_actions=True``
  semantics.
* :mod:`flybalatro.realgame.play` - the decision loop: read state, encode to
  state bits, drive the brain for a 50 ms window, read the readout, act.

Nothing here imports :mod:`pylatro`, so the real-game path runs even where the
Rust engine is unavailable.
"""

from __future__ import annotations

__all__ = ["client", "adapter", "play"]
