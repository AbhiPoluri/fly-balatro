"""Local demo viewer: watch the frozen MaleCNS network play Balatro in a browser.

Three pieces:

* :mod:`flybalatro.viewer.soma` - soma xyz from the MaleCNS body annotations,
  turned into a centred/scaled point cloud plus the neuron -> point index map.
* :mod:`flybalatro.viewer.policy` - the ``Policy`` interface and its four
  implementations (brain readout, raw-bits readout, heuristic-through-brain,
  random legal), plus the startup auto-selection.
* :mod:`flybalatro.viewer.server` - FastAPI app, websocket protocol and the
  paced game loop.

Run with ``python -m flybalatro.viewer.server``.
"""

from __future__ import annotations

__all__ = ["soma", "policy", "server"]
