"""Typed JSON-RPC 2.0 client for the BalatroBot mod's HTTP API.

Transport and framing follow ``vendor/balatrobot/docs/api.md``: a single POST
of ``{"jsonrpc": "2.0", "method": ..., "params": ..., "id": ...}`` to
``http://127.0.0.1:12346``, answered with either ``result`` or ``error``.

Method names here mirror the API's method names exactly - ``health``,
``gamestate``, ``start``, ``menu``, ``save``, ``load``, ``select``, ``skip``,
``buy``, ``pack``, ``sell``, ``reroll``, ``cash_out``, ``next_round``,
``play``, ``discard``, ``rearrange``, ``use``, ``screenshot`` - with one
unavoidable spelling change: the RPC method ``set`` is exposed as
:meth:`BalatroBot.set_`, because a method literally named ``set`` reads as the
builtin at every call site. :data:`METHOD_NAMES` is the authoritative list of
wire names.

The methods whose wire form is "exactly one of these keys" (``buy``, ``sell``,
``pack``, ``rearrange``) additionally have unambiguous convenience wrappers -
``buy_card``, ``sell_joker``, ``pack_skip``, ``rearrange_hand`` and friends -
that just delegate. Watch the one inconsistency in the API itself: ``use``
takes its hand targets as ``cards``, while ``pack`` takes them as ``targets``.

Two facts about this API shape the rest of the package:

1. **There is no card-selection endpoint.** ``play`` and ``discard`` take the
   0-based hand indices to act on. A bot therefore holds its own notion of
   "currently selected" and submits it at play/discard time; see
   :class:`flybalatro.realgame.adapter.RealGameState`.
2. **Every mutating method returns the resulting GameState**, and the docs
   pin the state it returns in (``select`` -> ``SELECTING_HAND``, ``cash_out``
   -> ``SHOP``, ``next_round`` -> ``BLIND_SELECT``). The endpoints block until
   the game settles, so polling is a safety net rather than the main mechanism
   - :meth:`BalatroBot.wait_until_stable` exists for the transient animation
   states (``HAND_PLAYED``, ``DRAW_TO_HAND``, ``NEW_ROUND``) in case a build
   returns early.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "BalatroBot",
    "BalatroBotError",
    "TransportError",
    "GameStateJson",
    "JsonValue",
    "ParamValue",
    "METHOD_NAMES",
    "ERROR_NAMES",
    "TRANSIENT_STATES",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
]

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 12346

# A decoded JSON document. Recursive aliases are not expressible in the typing
# vocabulary this project uses, so the nesting bottoms out at `object` and the
# adapter narrows explicitly rather than casting.
JsonValue = Union[None, bool, int, float, str, List[object], Dict[str, object]]

#: A BalatroBot ``gamestate`` payload, exactly as decoded from the wire.
GameStateJson = Dict[str, object]

#: Values accepted in an RPC ``params`` object.
ParamValue = Union[None, bool, int, float, str, List[int], List[str]]

#: Wire names of every method in ``docs/api.md`` / ``src/lua/utils/openrpc.json``.
METHOD_NAMES: Tuple[str, ...] = (
    "health",
    "gamestate",
    "rpc.discover",
    "start",
    "menu",
    "save",
    "load",
    "select",
    "skip",
    "buy",
    "pack",
    "sell",
    "reroll",
    "cash_out",
    "next_round",
    "play",
    "discard",
    "rearrange",
    "use",
    "add",
    "screenshot",
    "set",
)

#: ``data.name`` values the mod reports, mapped to their JSON-RPC codes.
ERROR_NAMES: Mapping[str, int] = {
    "INTERNAL_ERROR": -32000,
    "BAD_REQUEST": -32001,
    "INVALID_STATE": -32002,
    "NOT_ALLOWED": -32003,
}

#: States that mean "an animation is playing"; no decision can be taken in them.
TRANSIENT_STATES: FrozenSet[str] = frozenset(
    {"HAND_PLAYED", "DRAW_TO_HAND", "NEW_ROUND", "PLAY_TAROT", "UNKNOWN"}
)


class TransportError(RuntimeError):
    """The HTTP request itself failed (game not running, port closed, timeout)."""


class BalatroBotError(RuntimeError):
    """The API returned a JSON-RPC ``error`` object.

    Attributes:
        code: JSON-RPC error code, e.g. ``-32002``.
        name: The mod's symbolic name from ``error.data.name``, e.g.
            ``"INVALID_STATE"``. Empty string when the payload omits it.
        method: The RPC method that failed.
    """

    def __init__(self, method: str, code: int, message: str, name: str) -> None:
        super().__init__(f"{method}: {name or code} - {message}")
        self.code: int = code
        self.name: str = name
        self.method: str = method


class BalatroBot:
    """Thin, synchronous JSON-RPC client.

    Uses only the standard library, so it does not pull in the ``balatrobot``
    PyPI package (which requires Python 3.13; this project's venv is 3.11).
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = 30.0,
    ) -> None:
        self.host: str = host
        self.port: int = port
        self.timeout: float = float(timeout)
        self.url: str = f"http://{host}:{port}"
        self._next_id: int = 1

    # -- transport -------------------------------------------------------

    def call(self, method: str, params: Optional[Mapping[str, ParamValue]] = None) -> object:
        """Invoke ``method`` and return its ``result``.

        Raises:
            TransportError: the request never reached a JSON-RPC reply.
            BalatroBotError: the mod answered with an ``error`` object.
        """
        request_id = self._next_id
        self._next_id += 1
        payload: Dict[str, object] = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id,
        }
        if params is not None:
            payload["params"] = dict(params)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:  # the mod still answers JSON on 4xx/5xx
            raw = exc.read()
            if not raw:
                raise TransportError(f"{method}: HTTP {exc.code} with empty body") from exc
        except urllib.error.URLError as exc:
            raise TransportError(f"{method}: cannot reach {self.url} ({exc.reason})") from exc
        except TimeoutError as exc:
            raise TransportError(f"{method}: timed out after {self.timeout}s") from exc

        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransportError(f"{method}: reply was not JSON ({raw[:200]!r})") from exc
        if not isinstance(decoded, dict):
            raise TransportError(f"{method}: reply was not a JSON object")

        error = decoded.get("error")
        if error is not None:
            if not isinstance(error, dict):
                raise TransportError(f"{method}: malformed error object")
            raw_code = error.get("code")
            code = raw_code if isinstance(raw_code, int) else 0
            raw_message = error.get("message")
            message = raw_message if isinstance(raw_message, str) else ""
            data = error.get("data")
            name = ""
            if isinstance(data, dict):
                raw_name = data.get("name")
                if isinstance(raw_name, str):
                    name = raw_name
            raise BalatroBotError(method, code, message, name)

        if "result" not in decoded:
            raise TransportError(f"{method}: reply had neither result nor error")
        return decoded["result"]

    def _call_state(
        self, method: str, params: Optional[Mapping[str, ParamValue]] = None
    ) -> GameStateJson:
        """Invoke a method documented to return a GameState."""
        result = self.call(method, params)
        if not isinstance(result, dict):
            raise TransportError(f"{method}: expected a GameState object, got {type(result).__name__}")
        return result

    # -- lifecycle -------------------------------------------------------

    def health(self) -> Dict[str, object]:
        """RPC ``health``. Returns ``{"status": "ok"}``."""
        result = self.call("health")
        if not isinstance(result, dict):
            raise TransportError("health: expected an object")
        return result

    def is_up(self) -> bool:
        """True when ``health`` answers, False on any transport failure."""
        try:
            return self.health().get("status") == "ok"
        except (TransportError, BalatroBotError):
            return False

    def wait_for_api(self, timeout: float = 120.0, interval: float = 1.0) -> bool:
        """Block until ``health`` answers ok, or ``timeout`` elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_up():
                return True
            time.sleep(interval)
        return False

    def gamestate(self) -> GameStateJson:
        """RPC ``gamestate``. Works in any state."""
        return self._call_state("gamestate")

    def menu(self) -> GameStateJson:
        """RPC ``menu``. Returns to the main menu from any state."""
        return self._call_state("menu")

    def start(self, deck: str = "RED", stake: str = "WHITE", seed: Optional[str] = None) -> GameStateJson:
        """RPC ``start``. Requires ``MENU``; returns in ``BLIND_SELECT``."""
        params: Dict[str, ParamValue] = {"deck": deck, "stake": stake}
        if seed is not None:
            params["seed"] = seed
        return self._call_state("start", params)

    def save(self, path: str) -> Dict[str, object]:
        """RPC ``save``."""
        result = self.call("save", {"path": path})
        if not isinstance(result, dict):
            raise TransportError("save: expected an object")
        return result

    def load(self, path: str) -> Dict[str, object]:
        """RPC ``load``."""
        result = self.call("load", {"path": path})
        if not isinstance(result, dict):
            raise TransportError("load: expected an object")
        return result

    # -- blind selection -------------------------------------------------

    def select(self) -> GameStateJson:
        """RPC ``select``: play the current blind. ``BLIND_SELECT`` -> ``SELECTING_HAND``."""
        return self._call_state("select")

    def skip(self) -> GameStateJson:
        """RPC ``skip``: skip the current blind. Small/Big only."""
        return self._call_state("skip")

    # -- the round -------------------------------------------------------

    def play(self, cards: Sequence[int]) -> GameStateJson:
        """RPC ``play``: play the cards at these 0-based hand indices (1-5 of them)."""
        return self._call_state("play", {"cards": [int(i) for i in cards]})

    def discard(self, cards: Sequence[int]) -> GameStateJson:
        """RPC ``discard``: discard the cards at these 0-based hand indices."""
        return self._call_state("discard", {"cards": [int(i) for i in cards]})

    def rearrange(
        self,
        hand: Optional[Sequence[int]] = None,
        jokers: Optional[Sequence[int]] = None,
        consumables: Optional[Sequence[int]] = None,
    ) -> GameStateJson:
        """RPC ``rearrange``: exactly one of ``hand`` / ``jokers`` / ``consumables``."""
        given = [(k, v) for k, v in
                 (("hand", hand), ("jokers", jokers), ("consumables", consumables))
                 if v is not None]
        if len(given) != 1:
            raise ValueError("rearrange takes exactly one of hand, jokers, consumables")
        key, value = given[0]
        return self._call_state("rearrange", {key: [int(i) for i in value]})

    def rearrange_hand(self, order: Sequence[int]) -> GameStateJson:
        """``rearrange(hand=order)``."""
        return self.rearrange(hand=order)

    def cash_out(self) -> GameStateJson:
        """RPC ``cash_out``. ``ROUND_EVAL`` -> ``SHOP``."""
        return self._call_state("cash_out")

    # -- the shop --------------------------------------------------------

    def next_round(self) -> GameStateJson:
        """RPC ``next_round``: leave the shop. ``SHOP`` -> ``BLIND_SELECT``."""
        return self._call_state("next_round")

    def buy(
        self,
        card: Optional[int] = None,
        voucher: Optional[int] = None,
        pack: Optional[int] = None,
    ) -> GameStateJson:
        """RPC ``buy``: exactly one of ``card`` / ``voucher`` / ``pack``."""
        given = [(k, v) for k, v in
                 (("card", card), ("voucher", voucher), ("pack", pack)) if v is not None]
        if len(given) != 1:
            raise ValueError("buy takes exactly one of card, voucher, pack")
        key, value = given[0]
        return self._call_state("buy", {key: int(value)})

    def buy_card(self, index: int) -> GameStateJson:
        """``buy(card=index)``."""
        return self.buy(card=index)

    def buy_voucher(self, index: int) -> GameStateJson:
        """``buy(voucher=index)``."""
        return self.buy(voucher=index)

    def buy_pack(self, index: int) -> GameStateJson:
        """``buy(pack=index)``."""
        return self.buy(pack=index)

    def reroll(self) -> GameStateJson:
        """RPC ``reroll``: reroll the shop (costs money)."""
        return self._call_state("reroll")

    def sell(
        self, joker: Optional[int] = None, consumable: Optional[int] = None
    ) -> GameStateJson:
        """RPC ``sell``: exactly one of ``joker`` / ``consumable``."""
        given = [(k, v) for k, v in
                 (("joker", joker), ("consumable", consumable)) if v is not None]
        if len(given) != 1:
            raise ValueError("sell takes exactly one of joker, consumable")
        key, value = given[0]
        return self._call_state("sell", {key: int(value)})

    def sell_joker(self, index: int) -> GameStateJson:
        """``sell(joker=index)``."""
        return self.sell(joker=index)

    def sell_consumable(self, index: int) -> GameStateJson:
        """``sell(consumable=index)``."""
        return self.sell(consumable=index)

    # -- packs and consumables -------------------------------------------

    def pack(
        self,
        card: Optional[int] = None,
        targets: Optional[Sequence[int]] = None,
        skip: bool = False,
    ) -> GameStateJson:
        """RPC ``pack``: take ``card`` from the open pack, or ``skip`` it.

        ``targets`` are hand indices for a consumable that needs them (The
        Magician and friends). Note the wire name is ``targets``, not ``cards``
        - ``use`` spells the same idea ``cards``. Source of truth is
        ``src/lua/endpoints/pack.lua``, since ``pack`` is absent from
        ``openrpc.json``.
        """
        if skip:
            return self._call_state("pack", {"skip": True})
        if card is None:
            raise ValueError("pack needs either card= or skip=True")
        params: Dict[str, ParamValue] = {"card": int(card)}
        if targets is not None:
            params["targets"] = [int(i) for i in targets]
        return self._call_state("pack", params)

    def pack_select(
        self, index: int, targets: Optional[Sequence[int]] = None
    ) -> GameStateJson:
        """``pack(card=index, targets=targets)``."""
        return self.pack(card=index, targets=targets)

    def pack_skip(self) -> GameStateJson:
        """``pack(skip=True)``."""
        return self.pack(skip=True)

    def use(self, consumable: int, cards: Optional[Sequence[int]] = None) -> GameStateJson:
        """RPC ``use``: use consumable ``consumable``, optionally on hand ``cards``."""
        params: Dict[str, ParamValue] = {"consumable": int(consumable)}
        if cards is not None:
            params["cards"] = [int(i) for i in cards]
        return self._call_state("use", params)

    # -- debug / capture -------------------------------------------------

    def screenshot(self, path: str) -> Dict[str, object]:
        """RPC ``screenshot``: have the game write a PNG to ``path``."""
        result = self.call("screenshot", {"path": path})
        if not isinstance(result, dict):
            raise TransportError("screenshot: expected an object")
        return result

    def set_(self, **values: ParamValue) -> GameStateJson:
        """RPC ``set`` (debug): ``money``, ``chips``, ``ante``, ``round``, ``hands``, ``discards``, ``shop``."""
        return self._call_state("set", values)

    def discover(self) -> Dict[str, object]:
        """RPC ``rpc.discover``: the mod's own OpenRPC document."""
        result = self.call("rpc.discover")
        if not isinstance(result, dict):
            raise TransportError("rpc.discover: expected an object")
        return result

    # -- idling ----------------------------------------------------------

    def wait_until_stable(
        self, timeout: float = 20.0, interval: float = 0.1
    ) -> GameStateJson:
        """Poll ``gamestate`` until the game is out of an animation state.

        The mutating endpoints already block until the game settles, so this
        normally returns on its first poll. It matters only if a build returns
        while ``HAND_PLAYED`` / ``DRAW_TO_HAND`` / ``NEW_ROUND`` is still on
        screen. On timeout it returns the last state read rather than raising,
        so the caller can decide with whatever the game is showing.
        """
        deadline = time.monotonic() + timeout
        state = self.gamestate()
        while time.monotonic() < deadline:
            name = state.get("state")
            if not isinstance(name, str) or name not in TRANSIENT_STATES:
                return state
            time.sleep(interval)
            state = self.gamestate()
        return state
