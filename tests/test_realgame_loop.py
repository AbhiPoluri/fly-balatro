"""End-to-end test of the real-game loop against a scripted fake BalatroBot.

The fake is a real ``http.server`` speaking JSON-RPC 2.0, so the client's
transport, error mapping and state handling are all exercised; only the game
itself is simulated. The brain is stubbed (a full MaleCNS ``Brain`` is ~1.5 GB
and irrelevant to the control flow), while the policy is the real
``HeuristicPolicy``, so the path under test is:

    HTTP gamestate -> adapter -> 283 bits -> policy -> adapter action plan
      -> HTTP call -> log line

which is everything except the spiking network and the real game.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from flybalatro.realgame.client import BalatroBot, BalatroBotError, TransportError
from flybalatro.realgame.play import (
    DecisionLogger,
    HeuristicThroughBrainPolicy,
    PopulationRates,
    play,
)

RANKS: Tuple[str, ...] = ("A", "K", "Q", "J", "T", "9", "8", "7")
SUITS: Tuple[str, ...] = ("S", "H", "D", "C")


# -- the fake game ----------------------------------------------------------


class FakeGame:
    """A minimal Balatro state machine: one blind, then the shop, then ante 2."""

    def __init__(self, blind_score: int = 300, chips_per_play: int = 120) -> None:
        self.state: str = "MENU"
        self.blind_score: int = blind_score
        self.chips_per_play: int = chips_per_play
        self.chips: int = 0
        self.hands_left: int = 4
        self.discards_left: int = 3
        self.money: int = 4
        self.ante: int = 1
        self.round: int = 0
        self.next_id: int = 1
        self.hand: List[Dict[str, object]] = []
        self.calls: List[Tuple[str, Mapping[str, object]]] = []

    # -- helpers
    def _card(self, rank: str, suit: str) -> Dict[str, object]:
        card_id = self.next_id
        self.next_id += 1
        return {
            "id": card_id, "key": f"{suit}_{rank}", "set": "DEFAULT",
            "label": f"{rank}{suit}",
            "value": {"suit": suit, "rank": rank, "effect": ""},
            "modifier": {"seal": None, "edition": None, "enhancement": None,
                         "eternal": False, "perishable": None, "rental": False},
            "state": {"debuff": False, "hidden": False, "highlight": False},
            "cost": {"sell": 1, "buy": 0},
        }

    def _deal(self, n: int = 8) -> None:
        # A deliberately boring hand: one pair of Aces, no flush, so the
        # heuristic has to discard before it plays.
        pattern = [("A", "S"), ("A", "H"), ("K", "D"), ("Q", "C"),
                   ("J", "S"), ("9", "H"), ("8", "D"), ("7", "C")]
        self.hand = [self._card(r, s) for r, s in pattern[:n]]

    def _area(self, cards: Sequence[Mapping[str, object]], limit: int,
              highlighted: Optional[int] = None) -> Dict[str, object]:
        out: Dict[str, object] = {"count": len(cards), "limit": limit,
                                  "cards": list(cards)}
        if highlighted is not None:
            out["highlighted_limit"] = highlighted
        return out

    def gamestate(self) -> Dict[str, object]:
        in_blind = self.state in ("SELECTING_HAND",)
        status = "CURRENT" if self.state not in ("BLIND_SELECT", "MENU") else "SELECT"
        return {
            "state": self.state,
            "round_num": self.round, "ante_num": self.ante, "money": self.money,
            "deck": "RED", "stake": "WHITE", "seed": "FAKE1", "won": False,
            "used_vouchers": {}, "hands": {},
            "round": {"hands_left": self.hands_left, "hands_played": 0,
                      "discards_left": self.discards_left, "discards_used": 0,
                      "reroll_cost": 5, "chips": self.chips},
            "blinds": {
                "small": {"type": "SMALL", "status": status,
                          "name": "Small Blind", "effect": "",
                          "score": self.blind_score},
                "big": {"type": "BIG", "status": "UPCOMING", "name": "Big Blind",
                        "effect": "", "score": 450},
                "boss": {"type": "BOSS", "status": "UPCOMING", "name": "The Hook",
                         "effect": "", "score": 600},
            },
            "jokers": self._area([], 5), "consumables": self._area([], 2),
            "hand": self._area(self.hand if in_blind else [], 8, 5),
            "cards": self._area([], 52),
            "shop": self._area(
                [{"id": 900, "key": "j_joker", "set": "JOKER", "label": "Joker",
                  "value": {"effect": ""},
                  "modifier": {"seal": None, "edition": None, "enhancement": None,
                               "eternal": False, "perishable": None, "rental": False},
                  "state": {"debuff": False, "hidden": False, "highlight": False},
                  "cost": {"sell": 2, "buy": 4}}] if self.state == "SHOP" else [],
                4,
            ),
            "vouchers": self._area([], 1), "packs": self._area([], 2),
            "pack": self._area([], 5),
        }

    # -- dispatch
    def dispatch(self, method: str, params: Mapping[str, object]) -> object:
        self.calls.append((method, dict(params)))
        if method == "health":
            return {"status": "ok"}
        if method == "gamestate":
            return self.gamestate()
        if method == "menu":
            self.state = "MENU"
            return self.gamestate()
        if method == "start":
            self.state = "BLIND_SELECT"
            self.round = 1
            return self.gamestate()
        if method == "select":
            self._require("BLIND_SELECT", method)
            self.state = "SELECTING_HAND"
            self._deal()
            return self.gamestate()
        if method == "skip":
            self._require("BLIND_SELECT", method)
            self.state = "BLIND_SELECT"
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
                self.chips += self.chips_per_play
            # Refill the played/discarded slots with fresh cards.
            for i in indices:
                self.hand[i] = self._card("6", "S")
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
            self.state = "BLIND_SELECT"
            self.ante = 2
            self.round += 1
            self.chips = 0
            self.hands_left = 4
            self.discards_left = 3
            return self.gamestate()
        if method == "buy":
            self._require("SHOP", method)
            self.money -= 4
            return self.gamestate()
        raise _RpcError(-32601, f"unknown method {method}", "BAD_REQUEST")

    def _require(self, expected: str, method: str) -> None:
        if self.state != expected:
            raise _RpcError(
                -32002, f"{method} requires {expected}, game is in {self.state}",
                "INVALID_STATE",
            )


class _RpcError(Exception):
    def __init__(self, code: int, message: str, name: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.name = name


class _Handler(BaseHTTPRequestHandler):
    game: FakeGame

    def do_POST(self) -> None:  # noqa: N802 - http.server's required spelling
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}
        try:
            result = type(self).game.dispatch(method, params)
            body = {"jsonrpc": "2.0", "result": result, "id": request_id}
            code = 200
        except _RpcError as exc:
            body = {"jsonrpc": "2.0", "id": request_id,
                    "error": {"code": exc.code, "message": exc.message,
                              "data": {"name": exc.name}}}
            code = 200
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # keep pytest output clean


@pytest.fixture()
def fake_game() -> Tuple[FakeGame, BalatroBot]:
    game = FakeGame()
    handler = type("_BoundHandler", (_Handler,), {"game": game})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = BalatroBot("127.0.0.1", server.server_address[1], timeout=5.0)
    try:
        yield game, client
    finally:
        server.shutdown()
        server.server_close()


# -- a stub brain -----------------------------------------------------------


class StubBrain:
    """Same surface as ``BrainRunner``, no spiking network."""

    def __init__(self, feature_version: int = 1) -> None:
        from flybalatro.features_v2 import encoder_for_version, n_features_for_version

        self.condition = "real"
        self.window_ms = 50.0
        self.pop_sizes = {"ORN": 2639, "ALPN": 686, "KC": 4064, "MBON": 94, "DN": 1314}
        self.readout_idx = np.arange(6064, dtype=np.int32)
        self.calls = 0
        # The loop encodes through the brain runner, because the encoding version
        # is a property of the readout the runner was built for.
        self.feature_version = int(feature_version)
        self.n_features = n_features_for_version(feature_version)
        self.encode = encoder_for_version(feature_version)
        self.last_substeps = None
        self.last_total_spikes = None
        self.last_n_spiking = None
        # Only the real BrainRunner has one; the loop reads brain.brain.n when
        # it writes a spike sidecar, which this stub never triggers.
        self.brain = None

    def run(
        self,
        bits: NDArray[np.float32],
        *,
        record_substeps: bool = False,
    ) -> Tuple[NDArray[np.float32], PopulationRates, float]:
        self.calls += 1
        # Something bits-dependent, so a log reader can tell decisions apart.
        level = float(bits.sum())
        features = np.full(len(self.readout_idx), np.log1p(level), dtype=np.float32)
        rates = PopulationRates(orn=level, alpn=level / 2, kc=level / 4,
                                mbon=level / 8, dn=level / 16)
        return features, rates, 0.001


# -- client tests -----------------------------------------------------------


def test_client_health_and_gamestate(fake_game: Tuple[FakeGame, BalatroBot]) -> None:
    _game, client = fake_game
    assert client.health() == {"status": "ok"}
    assert client.is_up()
    state = client.gamestate()
    assert state["state"] == "MENU"


def test_client_maps_rpc_errors(fake_game: Tuple[FakeGame, BalatroBot]) -> None:
    _game, client = fake_game
    with pytest.raises(BalatroBotError) as excinfo:
        client.select()  # requires BLIND_SELECT, game is at MENU
    assert excinfo.value.name == "INVALID_STATE"
    assert excinfo.value.code == -32002
    assert excinfo.value.method == "select"


def test_client_reports_transport_failure() -> None:
    client = BalatroBot("127.0.0.1", 1, timeout=2.0)
    assert not client.is_up()
    with pytest.raises(TransportError):
        client.gamestate()


def test_client_sends_documented_params(fake_game: Tuple[FakeGame, BalatroBot]) -> None:
    game, client = fake_game
    client.start(deck="BLUE", stake="RED", seed="ABC")
    client.select()
    client.play([2, 0])
    methods = {name: params for name, params in game.calls}
    assert methods["start"] == {"deck": "BLUE", "stake": "RED", "seed": "ABC"}
    assert methods["play"] == {"cards": [2, 0]}


# -- loop tests -------------------------------------------------------------


def test_loop_plays_a_blind_and_reaches_the_shop(
    fake_game: Tuple[FakeGame, BalatroBot], tmp_path: Path
) -> None:
    game, client = fake_game
    logger = DecisionLogger(tmp_path)
    summary = play(
        client, StubBrain(), HeuristicThroughBrainPolicy(), logger,
        pause=0.0, max_decisions=200, stop_after_ante=1, verbose=False,
    )
    logger.close()

    assert summary["stop_reason"] == "ante_cleared", summary
    called = [name for name, _ in game.calls]
    assert "select" in called, "must select the blind"
    assert "play" in called, "must play at least one hand"
    assert "cash_out" in called, "must cash out"
    assert "next_round" in called, "must leave the shop"
    assert summary["blinds_attempted"] == ["Small Blind"]
    assert isinstance(summary["best_round_score"], int)
    assert summary["best_round_score"] >= game.chips_per_play


def test_loop_log_schema(fake_game: Tuple[FakeGame, BalatroBot], tmp_path: Path) -> None:
    _game, client = fake_game
    logger = DecisionLogger(tmp_path)
    play(client, StubBrain(), HeuristicThroughBrainPolicy(), logger,
         pause=0.0, max_decisions=200, stop_after_ante=1, verbose=False)
    logger.close()

    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines
    records = [json.loads(line) for line in lines]
    decisions = [r for r in records if "action_index" in r and "error" not in r]
    assert decisions

    for record in decisions:
        for key in ("t", "decision", "condition", "policy_mode", "action_index",
                    "action_name", "action_kind", "n_legal", "window_ms",
                    "spike_rates_hz", "population_sizes", "state"):
            assert key in record, f"missing {key}"
        rates = record["spike_rates_hz"]
        assert set(rates) == {"orn", "alpn", "kc", "mbon", "dn"}
        assert record["action_kind"] in ("local", "rpc")
        state = record["state"]
        for key in ("raw_state", "stage", "ante", "score", "required_score",
                    "plays", "discards", "money", "hand", "selected_indices"):
            assert key in state, f"missing state.{key}"

    # latest.json always holds one complete record.
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest == records[-1]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["decisions"] == len(decisions)


def test_loop_drives_the_brain_once_per_decision(
    fake_game: Tuple[FakeGame, BalatroBot], tmp_path: Path
) -> None:
    _game, client = fake_game
    brain = StubBrain()
    logger = DecisionLogger(tmp_path)
    summary = play(client, brain, HeuristicThroughBrainPolicy(), logger,
                   pause=0.0, max_decisions=200, stop_after_ante=1, verbose=False)
    logger.close()
    assert brain.calls == summary["decisions"]


def test_loop_selection_is_cleared_after_playing(
    fake_game: Tuple[FakeGame, BalatroBot], tmp_path: Path
) -> None:
    """A stale selection after a refill would make the fly play the wrong cards."""
    game, client = fake_game
    logger = DecisionLogger(tmp_path)
    play(client, StubBrain(), HeuristicThroughBrainPolicy(), logger,
         pause=0.0, max_decisions=200, stop_after_ante=1, verbose=False)
    logger.close()

    records = [json.loads(line) for line
               in (tmp_path / "log.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    # Every play/discard is immediately followed by an empty selection.
    for i, record in enumerate(records[:-1]):
        if record.get("action_name") in ("play", "discard"):
            following = records[i + 1]
            if "state" in following:
                assert following["state"]["selected_indices"] == []

    # And the indices actually submitted were within the hand at the time.
    for name, params in game.calls:
        if name in ("play", "discard"):
            cards = params["cards"]
            assert isinstance(cards, list) and cards
            assert all(0 <= int(c) < 8 for c in cards)
            assert len(set(int(c) for c in cards)) == len(cards)


def test_loop_stops_on_game_over(tmp_path: Path) -> None:
    """A blind that cannot be beaten ends the run rather than spinning."""
    game = FakeGame(blind_score=10_000, chips_per_play=1)
    handler = type("_BoundHandler2", (_Handler,), {"game": game})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = BalatroBot("127.0.0.1", server.server_address[1], timeout=5.0)
    logger = DecisionLogger(tmp_path)
    try:
        summary = play(client, StubBrain(), HeuristicThroughBrainPolicy(), logger,
                       pause=0.0, max_decisions=300, stop_after_ante=1, verbose=False)
    finally:
        logger.close()
        server.shutdown()
        server.server_close()
    assert summary["stop_reason"] == "game_over_lose", summary


def test_every_state_the_fake_reaches_is_encodable(
    fake_game: Tuple[FakeGame, BalatroBot], tmp_path: Path
) -> None:
    """No decision may be taken from a state the adapter cannot map."""
    _game, client = fake_game
    logger = DecisionLogger(tmp_path)
    play(client, StubBrain(), HeuristicThroughBrainPolicy(), logger,
         pause=0.0, max_decisions=200, stop_after_ante=1, verbose=False)
    logger.close()
    records = [json.loads(line) for line
               in (tmp_path / "log.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    seen = {r["state"]["raw_state"] for r in records if "state" in r}
    assert {"BLIND_SELECT", "SELECTING_HAND", "ROUND_EVAL", "SHOP"} <= seen, seen
    for record in records:
        if "state" in record:
            assert 0 <= record["state"]["stage"] <= 10


def test_pack_uses_targets_not_cards(fake_game: Tuple[FakeGame, BalatroBot]) -> None:
    """`pack` spells hand targets `targets`; `use` spells them `cards`.

    Regression guard: sending `cards` to `pack` is a BAD_REQUEST, and the two
    endpoints genuinely disagree on the key name.
    """
    game, client = fake_game
    try:
        client.pack(card=0, targets=[1, 2])
    except BalatroBotError:
        pass  # the fake rejects unknown methods; only the params matter here
    try:
        client.pack(skip=True)
    except BalatroBotError:
        pass
    try:
        client.use(0, cards=[1])
    except BalatroBotError:
        pass

    sent = {name: params for name, params in game.calls}
    assert sent["pack"] == {"skip": True}
    pack_calls = [p for n, p in game.calls if n == "pack"]
    assert pack_calls[0] == {"card": 0, "targets": [1, 2]}
    assert "cards" not in pack_calls[0]
    assert sent["use"] == {"consumable": 0, "cards": [1]}


def test_exactly_one_of_params_is_enforced(fake_game: Tuple[FakeGame, BalatroBot]) -> None:
    _game, client = fake_game
    for call in (
        lambda: client.buy(),
        lambda: client.buy(card=0, voucher=1),
        lambda: client.sell(),
        lambda: client.sell(joker=0, consumable=0),
        lambda: client.rearrange(),
        lambda: client.rearrange(hand=[0], jokers=[0]),
        lambda: client.pack(),
    ):
        with pytest.raises(ValueError):
            call()


def test_convenience_wrappers_hit_the_documented_wire_form(
    fake_game: Tuple[FakeGame, BalatroBot]
) -> None:
    game, client = fake_game
    client.start()
    client.select()
    for call in (lambda: client.buy_card(2), lambda: client.buy_voucher(0),
                 lambda: client.buy_pack(1)):
        try:
            call()
        except BalatroBotError:
            pass
    buys = [p for n, p in game.calls if n == "buy"]
    assert buys == [{"card": 2}, {"voucher": 0}, {"pack": 1}]


def test_loop_relays_the_hand_analysis_under_the_v2_encoding(
    fake_game: Tuple[FakeGame, BalatroBot], tmp_path: Path
) -> None:
    """v2: the loop encodes 315 bits and logs what it relayed to the fly.

    The adapter's cards expose the same ``rank_index`` / ``suit_index`` / ``id``
    fields ``pylatro`` does, which is all ``flybalatro.hands`` reads, so the
    analysis in the log is the same function of the board that the headless
    pipeline was trained on.
    """
    from flybalatro.features_v2 import N_FEATURES_V2
    from flybalatro.hands import HAND_TYPE_NAMES

    _game, client = fake_game
    brain = StubBrain(feature_version=2)
    logger = DecisionLogger(tmp_path)
    summary = play(client, brain, HeuristicThroughBrainPolicy(), logger,
                   pause=0.0, max_decisions=200, stop_after_ante=1, verbose=False)
    logger.close()
    assert summary["stop_reason"] == "ante_cleared", summary

    records = [json.loads(line) for line
               in (tmp_path / "log.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    decisions = [r for r in records if "action_index" in r and "error" not in r]
    assert decisions
    assert brain.n_features == N_FEATURES_V2

    in_blind = [r for r in decisions if r["relayed_hand_analysis"]["active"]]
    assert in_blind, "no decision was taken inside a blind"
    names = set(HAND_TYPE_NAMES) | {"none"}
    for record in decisions:
        assert record["feature_version"] == 2
        relay = record["relayed_hand_analysis"]
        assert set(relay) >= {"active", "best_type", "selected_type", "n_bits_on"}
        assert relay["best_type"] in names
        assert relay["selected_type"] in names
    for record in in_blind:
        relay = record["relayed_hand_analysis"]
        assert relay["best_type"] != "none"
        assert 0 <= len(relay["best_slots"]) <= 5
        assert relay["best_score"] > 0
        assert 1 <= relay["n_bits_on"] <= 32

    played = [r for r in in_blind if r["action_name"] == "play"]
    assert played, "the hand-aware heuristic must play at least once"
    for record in played:
        assert record["relayed_hand_analysis"]["selected_is_best"] is True
