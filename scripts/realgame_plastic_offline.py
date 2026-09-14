"""Run the real-game plasticity loop with no Balatro, against a scripted game.

Why this exists
---------------

``flybalatro/realgame/plastic.py`` is the learning fly driving **actual
Balatro** through the BalatroBot mod. Actual Balatro is not always running, and
a claim that the loop works should not depend on whether it is. So this script
stands a real ``http.server`` up on localhost speaking the mod's own JSON-RPC
dialect, and points the real loop at it.

Everything except the game is the real thing:

* the real :class:`~flybalatro.realgame.client.BalatroBot` over real HTTP,
  including its error mapping and ``wait_until_stable`` polling;
* the real :func:`flybalatro.realgame.adapter.build_state` parsing real
  mod-shaped payloads;
* the real MaleCNS connectome at the real ``outputs/plast2`` operating point,
  with the real 4,064 per-Kenyon-cell thresholds;
* the real 50 ms window, the real approach-minus-avoidance MBON rule, the real
  :class:`flybalatro.plasticity.KcMbonPlasticity` depression;
* the real :func:`flybalatro.realgame.plastic.play_plastic` loop, unmodified.

What is simulated is the **game**: :class:`ScriptedGame` below deals random
hands from a real 52-card deck and scores a played hand with
:mod:`flybalatro.hands`, which is Balatro's own scoring. So the chips are a
genuine function of the cards the fly was shown, and therefore so is the
dopamine. What it does not simulate is Balatro's jokers, its blind effects or
its animations.

This is **not** evidence that the fly can beat Balatro, and nothing here should
be read as a clear rate. It is evidence that the loop closes: that a hand
becomes an odour, that the odour becomes an MBON drive, that the drive becomes
a key press, that the chips become dopamine, and that the dopamine moves
synapses. Run it against the real window for the real claim.

Usage::

    source .venv/bin/activate
    python -m scripts.realgame_plastic_offline --hands 40
    python -m scripts.realgame_plastic_offline --hands 40 --no-learning   # control
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybalatro import hands  # noqa: E402
from flybalatro.realgame.client import BalatroBot  # noqa: E402
from flybalatro.realgame.play import DecisionLogger, SpikeSidecar  # noqa: E402
from flybalatro.realgame.plastic import (  # noqa: E402
    LearningTrace,
    OperatingPoint,
    PlasticFly,
    play_plastic,
)

__all__ = ["ScriptedGame", "serve", "main"]

RANKS: Tuple[str, ...] = ("2", "3", "4", "5", "6", "7", "8", "9", "T",
                          "J", "Q", "K", "A")
SUITS: Tuple[str, ...] = ("S", "C", "H", "D")


class _Slot:
    """``hands`` reads ``rank_index`` / ``suit_index``; the payload has strings."""

    __slots__ = ("rank_index", "suit_index")

    def __init__(self, card: Mapping[str, object]) -> None:
        value = card["value"]
        self.rank_index = RANKS.index(str(value["rank"]))      # type: ignore[index]
        self.suit_index = SUITS.index(str(value["suit"]))      # type: ignore[index]


class _RpcError(Exception):
    def __init__(self, code: int, message: str, name: str) -> None:
        super().__init__(message)
        self.code, self.message, self.name = code, message, name


class ScriptedGame:
    """A Balatro-shaped state machine that deals real cards and scores for real.

    One ante of three blinds. A played hand scores with
    ``flybalatro.hands.best_subset`` over the submitted slots -- the same code
    that tells the fly what its options are worth -- so "chips gained" is what
    Balatro would pay for that subset with no jokers.

    The card geometry the real mod now reports is emitted too, from a simple
    fan layout, so the overlay's dynamic-alignment path has something to parse
    in a log produced without the game.
    """

    def __init__(self, seed: int = 0, blind_scores: Sequence[int] = (300, 450, 600),
                 emit_geometry: bool = True) -> None:
        self.rng = np.random.default_rng(seed)
        self.blind_scores = tuple(int(s) for s in blind_scores)
        self.emit_geometry = bool(emit_geometry)
        self.state = "MENU"
        self.blind_index = 0
        self.chips = 0
        self.hands_left = 4
        self.discards_left = 3
        self.money = 4
        self.ante = 1
        self.round = 0
        self.next_id = 1
        self.hand: List[Dict[str, object]] = []
        self.deck: List[Tuple[str, str]] = []
        self.calls: List[Tuple[str, Mapping[str, object]]] = []
        self.hands_scored: List[int] = []

    # -- cards -------------------------------------------------------------
    def _shuffle(self) -> None:
        deck = [(r, s) for s in SUITS for r in RANKS]
        order = self.rng.permutation(len(deck))
        self.deck = [deck[i] for i in order]

    def _draw(self) -> Dict[str, object]:
        if not self.deck:
            self._shuffle()
        rank, suit = self.deck.pop()
        card_id = self.next_id
        self.next_id += 1
        return {
            "id": card_id, "key": f"{suit}_{rank}", "set": "DEFAULT",
            "label": f"{rank}{suit}",
            "value": {"suit": suit, "rank": rank, "effect": ""},
            "modifier": [],
            "state": {"debuff": False, "hidden": False, "highlight": False},
            "cost": {"sell": 1, "buy": 0},
        }

    def _deal(self, n: int = 8) -> None:
        self.hand = [self._draw() for _ in range(n)]

    def _with_geometry(self, cards: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
        """Attach the ``geometry`` block the patched mod reports.

        A plain arithmetic fan in LOVE pixels: eight cards across the middle of
        a 2418 x 1570 canvas with the same +/-5.5 degree tilt the real game
        uses. Not a claim about real positions -- it exists so a consumer of
        this log exercises the same parsing path as a real one.
        """
        out: List[Dict[str, object]] = []
        n = max(1, len(cards))
        card_w, card_h = 211.0, 288.0
        span = 1378.0
        pitch = (span - card_w) / max(1, n - 1) if n > 1 else 0.0
        left = 1327.0 - (pitch * (n - 1) + card_w) / 2.0
        for i, card in enumerate(cards):
            entry = dict(card)
            u = (2.0 * i / (n - 1) - 1.0) if n > 1 else 0.0
            entry["geometry"] = {
                "rect": {
                    "x": round(left + pitch * i, 2),
                    "y": round(925.0 - 15.0 * (1.0 - u * u), 2),
                    "w": card_w, "h": card_h,
                    "r": round(np.radians(5.5) * u, 5),
                },
                "moving": False,
            }
            out.append(entry)
        return out

    # -- payload -----------------------------------------------------------
    def _area(self, cards: Sequence[Mapping[str, object]], limit: int,
              highlighted: Optional[int] = None, geometry: bool = False) -> Dict[str, object]:
        listed = (self._with_geometry(cards)
                  if geometry and self.emit_geometry else list(cards))
        out: Dict[str, object] = {"count": len(listed), "limit": limit, "cards": listed}
        if highlighted is not None:
            out["highlighted_limit"] = highlighted
        return out

    @property
    def blind_score(self) -> int:
        return self.blind_scores[min(self.blind_index, len(self.blind_scores) - 1)]

    def gamestate(self) -> Dict[str, object]:
        in_blind = self.state == "SELECTING_HAND"
        names = ("Small Blind", "Big Blind", "The Hook")
        types = ("SMALL", "BIG", "BOSS")
        blinds: Dict[str, object] = {}
        for i, key in enumerate(("small", "big", "boss")):
            if i < self.blind_index:
                status = "DEFEATED"
            elif i == self.blind_index:
                status = "CURRENT" if self.state not in ("BLIND_SELECT", "MENU") else "SELECT"
            else:
                status = "UPCOMING"
            blinds[key] = {"type": types[i], "status": status, "name": names[i],
                           "effect": "", "score": self.blind_scores[i]}
        payload: Dict[str, object] = {
            "state": self.state,
            "round_num": self.round, "ante_num": self.ante, "money": self.money,
            "deck": "RED", "stake": "WHITE", "seed": "OFFLINE", "won": False,
            "used_vouchers": {}, "hands": {},
            "round": {"hands_left": self.hands_left, "hands_played": 0,
                      "discards_left": self.discards_left, "discards_used": 0,
                      "reroll_cost": 5, "chips": self.chips},
            "blinds": blinds,
            "jokers": self._area([], 5), "consumables": self._area([], 2),
            "hand": self._area(self.hand if in_blind else [], 8, 5, geometry=True),
            "cards": self._area([], 52),
            "shop": self._area([], 4),
            "vouchers": self._area([], 1), "packs": self._area([], 2),
            "pack": self._area([], 5),
        }
        if self.emit_geometry:
            payload["screen"] = {
                "tilesize": 20, "tilescale": 3.65, "unit": 73.0,
                "room": {"x": 0.0, "y": 0.0, "w": 33.0, "h": 21.5},
                "width": 2418, "height": 1570,
                "pixel_width": 2418, "pixel_height": 1570, "dpi_scale": 1.0,
            }
        return payload

    # -- scoring -----------------------------------------------------------
    def _score_slots(self, indices: Sequence[int]) -> int:
        """What Balatro pays for those slots, via the project's own scorer.

        ``hands.classify_slots`` is the same function that tells the fly what
        its options are worth, so the chips the fly is paid and the chips it was
        told about come from one implementation of Balatro's rules.
        """
        cards = [_Slot(c) for c in self.hand]
        subset = hands.classify_slots(cards, list(indices))
        return int(subset.score) if subset is not None else 0

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, method: str, params: Mapping[str, object]) -> object:
        self.calls.append((method, dict(params)))
        if method == "health":
            return {"status": "ok", "scripted": True}
        if method == "gamestate":
            return self.gamestate()
        if method == "menu":
            self.state = "MENU"
            return self.gamestate()
        if method == "start":
            self.state = "BLIND_SELECT"
            self.round = 1
            self.blind_index = 0
            self._shuffle()
            return self.gamestate()
        if method == "select":
            self._require("BLIND_SELECT", method)
            self.state = "SELECTING_HAND"
            self._deal()
            return self.gamestate()
        if method in ("play", "discard"):
            self._require("SELECTING_HAND", method)
            raw = params.get("cards")
            if not isinstance(raw, list) or not raw:
                raise _RpcError(-32001, "cards must be a non-empty array", "BAD_REQUEST")
            indices = sorted(int(i) for i in raw)
            if indices[-1] >= len(self.hand) or indices[0] < 0:
                raise _RpcError(-32001, "card index out of range", "BAD_REQUEST")
            if method == "discard":
                if self.discards_left <= 0:
                    raise _RpcError(-32003, "no discards left", "NOT_ALLOWED")
                self.discards_left -= 1
            else:
                if self.hands_left <= 0:
                    raise _RpcError(-32003, "no hands left", "NOT_ALLOWED")
                self.hands_left -= 1
                gained = self._score_slots(indices)
                self.hands_scored.append(gained)
                self.chips += gained
            for i in indices:
                self.hand[i] = self._draw()
            if self.chips >= self.blind_score:
                self.state = "ROUND_EVAL"
            elif self.hands_left == 0:
                self.state = "GAME_OVER"
            return self.gamestate()
        if method == "cash_out":
            self._require("ROUND_EVAL", method)
            self.state = "SHOP"
            self.money += 4
            return self.gamestate()
        if method == "next_round":
            self._require("SHOP", method)
            self.blind_index += 1
            self.chips = 0
            self.hands_left = 4
            self.discards_left = 3
            self.round += 1
            if self.blind_index >= len(self.blind_scores):
                # Ante cleared.
                self.ante += 1
                self.blind_index = 0
            self.state = "BLIND_SELECT"
            return self.gamestate()
        if method == "skip":
            self._require("BLIND_SELECT", method)
            return self.gamestate()
        raise _RpcError(-32601, f"unknown method {method}", "BAD_REQUEST")

    def _require(self, expected: str, method: str) -> None:
        if self.state != expected:
            raise _RpcError(-32002,
                            f"{method} requires {expected}, game is in {self.state}",
                            "INVALID_STATE")


class _Handler(BaseHTTPRequestHandler):
    game: ScriptedGame

    def do_POST(self) -> None:  # noqa: N802 - http.server's spelling
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        request_id = request.get("id")
        try:
            result = type(self).game.dispatch(request.get("method", ""),
                                              request.get("params") or {})
            body: Dict[str, object] = {"jsonrpc": "2.0", "result": result, "id": request_id}
        except _RpcError as exc:
            body = {"jsonrpc": "2.0", "id": request_id,
                    "error": {"code": exc.code, "message": exc.message,
                              "data": {"name": exc.name}}}
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        pass


def serve(game: ScriptedGame) -> Tuple[HTTPServer, BalatroBot]:
    """Stand the scripted game up on a free localhost port."""
    handler = type("_BoundHandler", (_Handler,), {"game": game})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, BalatroBot("127.0.0.1", server.server_address[1], timeout=10.0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", type=int, default=40)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fly-seed", type=int, default=0)
    ap.add_argument("--ante-end", type=int, default=2)
    ap.add_argument("--no-learning", action="store_true",
                    help="the control: same decisions, dopamine computed but "
                         "never delivered, so the weights must not move")
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "realgame_offline"))
    ap.add_argument("--pause", type=float, default=0.0)
    ap.add_argument("--spikes", choices=("auto", "on", "off"), default="auto")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    game = ScriptedGame(seed=a.seed)
    server, client = serve(game)
    t0 = time.time()
    try:
        op = OperatingPoint.load(eta_reward=a.eta)
        fly = PlasticFly(op=op, seed=a.fly_seed, learning=not a.no_learning,
                         warm_start=Path(a.warm_start) if a.warm_start else None,
                         n_substeps=5)
        before = fly.weight_state()
        logger = DecisionLogger(out_dir)
        spikes = SpikeSidecar(out_dir, mode=a.spikes)
        trace = LearningTrace()
        try:
            summary = play_plastic(
                client, fly, logger, pause=a.pause, max_hands=a.hands,
                stop_after_ante=a.ante_end, spikes=spikes, trace=trace,
            )
        finally:
            logger.close()
            spikes.close()
        after = fly.weight_state()
        fly.save_weights(out_dir / "weights.npz", meta={"offline": True})
        report = {
            "scripted": True,
            "note": ("the GAME is simulated; the brain, the plasticity rule, the "
                     "client, the adapter and the loop are the real ones"),
            "hands_requested": a.hands,
            "wall_seconds": round(time.time() - t0, 1),
            "learning": not a.no_learning,
            "weights_before": before,
            "weights_after": after,
            "weights_moved": before != after,
            "chips_per_scored_hand": game.hands_scored,
            "rpc_calls": len(game.calls),
            "summary": summary,
        }
        (out_dir / "offline_report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print("\n== offline plasticity check ==", flush=True)
        print(f"  hands            {summary['hands']}", flush=True)
        print(f"  stop             {summary['stop_reason']}", flush=True)
        print(f"  dopamine         {summary['dopamine']}", flush=True)
        print(f"  synapses changed {after['n_changed']}/{after['n_synapses']} "
              f"({after['frac_changed']:.2%})  mean w/w0 {after['mean_ratio']:.6f}  "
              f"at floor {after['n_at_floor']}", flush=True)
        print(f"  P(play) by bucket "
              f"{ {k: v['p_play'] for k, v in trace.bucket_table().items()} }", flush=True)
        print(f"  report           {out_dir / 'offline_report.json'}", flush=True)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
