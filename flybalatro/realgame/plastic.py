"""The *learning* fly on the real Balatro window.

``flybalatro/realgame/play.py`` drives the game with the frozen v3 readout: a
scikit-learn model fitted on brain spikes, which never changes while it plays.
This module drives the same game with the fly from ``RESULTS.md`` "Plasticity
v2" instead, and the difference is the whole point:

* **No trained action readout.** The decision is
  ``mean rate(approach MBONs) - mean rate(avoid MBONs) + bias``, a fixed rule
  over neurotransmitter-assigned pools of the fly's own mushroom-body output
  neurons. Nothing is fitted to actions, labels or outcomes.
* **The synapses change while you watch.** The only mutable state in the fly is
  the weight of the 33,496 Kenyon-cell -> MBON edges, moved by dopamine-gated
  depression (:mod:`flybalatro.plasticity`) driven by the chips the *real game*
  pays out.

What is still computed outside the brain is exactly what it was before, and no
more: :mod:`flybalatro.hands` enumerates the subsets of the dealt cards and
scores them with Balatro's rules, and :mod:`flybalatro.glomerular` relays the
answer as 32 bits on 32 whole ORN glomeruli. The harness then presses the keys
for whichever of the two things the fly chose.

The decision is binary -- **play the best subset now, or discard the junk and
dig** -- so this loop is per *hand*, not per action. That is why it does not
reuse :func:`flybalatro.realgame.play.play`, whose loop is one iteration per
action index and would run four brain windows to select four cards. Everything
below the loop is shared: the same :class:`~flybalatro.realgame.client.BalatroBot`,
the same :func:`flybalatro.realgame.adapter.build_state`, the same
``DecisionLogger`` and ``SpikeSidecar``.

Operating point
---------------

Loaded, not re-derived:

``outputs/plast2/tuned_config.json``
    the brain -- 4,064 per-Kenyon-cell homeostatic thresholds, ``apl_kc_only``,
    ``apl_mbon_scale``, the glomerulus assignment, the 50 ms window.
``outputs/plast2/run_real_eta0.05_s0.json``
    the decision rule's two calibrated scalars, ``bias`` and ``temperature``,
    plus ``explore_floor``. Calibrated once on 200 pre-dopamine hands from
    headless seeds 300000+ and frozen; see ``RESULTS.md`` "Plasticity v2".
``outputs/plast2/eta_calib.json``
    ``eta_punish`` for the chosen ``eta_reward``: the two arms of the rule
    differ ~3.8x per pulse because the neurotransmitter rule gives 66 approach
    MBONs against 24 avoid ones, and only that per-pulse gain is matched.

Weights start **naive** by default, which is the interesting thing to watch;
``--warm-start`` loads ``outputs/plast2/weights_real_eta0.05_s0.npz`` instead.
Either way they are written out when the run ends, including on Ctrl-C.

Run it::

    source .venv/bin/activate
    python -m flybalatro.realgame.play --plastic --ante-end 1
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import plasticity as P
from ..features_v2 import SCORE_BUCKET_LABELS
from . import adapter as ad
from .client import TRANSIENT_STATES, BalatroBot, BalatroBotError

__all__ = [
    "PLAST2_CONFIG",
    "PLAST2_OPERATING_POINT",
    "PLAST2_ETA_CALIB",
    "PLAST2_WEIGHTS",
    "OperatingPoint",
    "PlasticFly",
    "LearningTrace",
    "run_window_substeps",
    "play_plastic",
]

ROOT: Path = Path(__file__).resolve().parents[2]
PLAST2: Path = ROOT / "outputs" / "plast2"
PLAST2_CONFIG: Path = PLAST2 / "tuned_config.json"
PLAST2_OPERATING_POINT: Path = PLAST2 / "run_real_eta0.05_s0.json"
PLAST2_ETA_CALIB: Path = PLAST2 / "eta_calib.json"
PLAST2_WEIGHTS: Path = PLAST2 / "weights_real_eta0.05_s0.npz"

PLAY: str = "play"
DISCARD: str = "discard"

#: Hands kept in the rolling windows the viewer draws. One blind is ~4 hands,
#: so 50 is about a dozen blinds -- long enough to show a trend, short enough
#: that a change in behaviour is visible while it happens.
ROLLING: int = 50


def _scripts_on_path() -> None:
    """``scripts/`` is not a package; the batch runs import it the same way."""
    d = str(ROOT / "scripts")
    if d not in sys.path:
        sys.path.insert(0, d)


# --------------------------------------------------------------------------- #
# the operating point                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OperatingPoint:
    """The calibrated scalars of the plast2 fly, read from disk, never refitted.

    ``bias`` and ``temperature`` are the decision rule's two free numbers.
    Refitting them here would make every run incomparable with ``RESULTS.md``
    and would cost 200 headless hands at startup, so they are loaded.
    """

    bias: float
    temperature: float
    explore_floor: float
    play_bias_p: float
    eta_reward: float
    eta_punish: float
    source: str
    eta_source: str

    @classmethod
    def load(
        cls,
        eta_reward: float = 0.05,
        eta_punish: Optional[float] = None,
        path: Path = PLAST2_OPERATING_POINT,
        eta_path: Path = PLAST2_ETA_CALIB,
    ) -> "OperatingPoint":
        payload = json.loads(Path(path).read_text())
        dec = payload.get("decider") or payload.get("calibration") or {}
        if "bias" not in dec or "temperature" not in dec:
            raise ValueError(
                f"{path} carries no calibrated bias/temperature; it is not a "
                "plast2 run json"
            )
        eta_src = "given on the command line"
        if eta_punish is None:
            eta_punish, eta_src = _eta_punish_for(eta_reward, eta_path)
        return cls(
            bias=float(dec["bias"]),
            temperature=float(dec["temperature"]),
            explore_floor=float(dec.get("explore_floor", 0.1)),
            play_bias_p=float(dec.get("play_bias_p", 0.55)),
            eta_reward=float(eta_reward),
            eta_punish=float(eta_punish),
            source=str(path),
            eta_source=eta_src,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "bias": self.bias, "temperature": self.temperature,
            "explore_floor": self.explore_floor, "play_bias_p": self.play_bias_p,
            "eta_reward": self.eta_reward, "eta_punish": self.eta_punish,
            "source": self.source, "eta_source": self.eta_source,
        }


def _eta_punish_for(eta_reward: float, path: Path = PLAST2_ETA_CALIB) -> Tuple[float, str]:
    """The calibrated punishment eta, or the reward eta if none was measured."""
    try:
        chosen = json.loads(Path(path).read_text()).get("chosen", {})
    except (OSError, ValueError):
        chosen = {}
    row = chosen.get(f"{eta_reward:g}")
    if isinstance(row, Mapping) and "eta_punish" in row:
        return float(row["eta_punish"]), f"{path} (ratio {row.get('ratio')})"
    # Honest fallback: no calibration for this eta, so the two arms are equal
    # and the run is *not* the plast2 operating point. Said out loud.
    return float(eta_reward), (
        f"NOT CALIBRATED: {path} has no entry for eta_reward={eta_reward:g}, "
        "so both arms use the same gain"
    )


# --------------------------------------------------------------------------- #
# the fly                                                                      #
# --------------------------------------------------------------------------- #
def run_window_substeps(
    setup: object,
    bits: NDArray[np.float32],
    n_substeps: int = 1,
) -> Tuple[NDArray[np.int32], List[NDArray[np.int32]], float]:
    """One decision window, optionally split so the viewer can replay it.

    Returns ``(counts, per_substep_fired, elapsed)``. ``n_substeps <= 1`` runs
    the window in one call and returns no sub-steps.

    Splitting is not an approximation: ``Brain.step`` carries its own kernel
    cursor across calls and only ``reset_counts`` differs, so the counts at the
    end -- and therefore the MBON drive, the decision and every weight change
    that follows -- are bit-identical either way. ``tests/test_realgame_spikes.py``
    asserts exactly that for the readout path; the same argument holds here.
    """
    brain = setup.brain                       # type: ignore[attr-defined]
    drive = setup.odor_drive(bits)            # type: ignore[attr-defined]
    window_ms = float(setup.window_ms)        # type: ignore[attr-defined]
    brain.reset()
    if n_substeps <= 1:
        counts, elapsed = brain.step(drive, window_ms)
        return counts, [], float(elapsed)

    sub_ms = window_ms / n_substeps
    previous = np.zeros(brain.n, dtype=np.int32)
    fired: List[NDArray[np.int32]] = []
    elapsed = 0.0
    counts = previous
    for k in range(n_substeps):
        counts, dt = brain.step(drive, sub_ms, reset_counts=(k == 0))
        fired.append(np.flatnonzero(counts > previous).astype(np.int32))
        previous = counts.copy()
        elapsed += float(dt)
    return counts, fired, elapsed


class PlasticFly:
    """The plast2 fly: frozen wiring and encoding, mutable KC -> MBON weights.

    Thin on purpose. The brain, the valence table and the plasticity object all
    come from ``scripts/plast_common.build_setup`` -- the same call the batch
    runs in ``RESULTS.md`` make -- so a fly playing the real game and a fly
    playing 1,200 headless hands are the same object at the same operating
    point, and the two sets of numbers are comparable.
    """

    def __init__(
        self,
        op: Optional[OperatingPoint] = None,
        config_path: Path = PLAST2_CONFIG,
        graph: Optional[object] = None,
        wiring: str = "real",
        valence_scheme: str = "nt",
        seed: int = 0,
        warm_start: Optional[Path] = None,
        learning: bool = True,
        n_substeps: int = 1,
        verbose: bool = True,
    ) -> None:
        _scripts_on_path()
        import plast_common as K  # noqa: E402  (scripts/ is not a package)

        self.K = K
        self.op = op or OperatingPoint.load()
        self.learning = bool(learning)
        self.n_substeps = max(1, int(n_substeps))
        cfg = json.loads(Path(config_path).read_text())
        self.config_path = Path(config_path)
        if verbose:
            print(f"[fly] plast2 operating point from {self.op.source}", flush=True)
            print(f"[fly] brain config {config_path}"
                  f" (label {cfg.get('label')})", flush=True)
        self.setup = K.build_setup(
            wiring=wiring, valence_scheme=valence_scheme, graph=graph, cfg=cfg
        )
        self.decider = P.Decider(
            approach=self.setup.valence.approach,
            avoid=self.setup.valence.avoid,
            window_ms=self.setup.window_ms,
            bias=self.op.bias,
            temperature=self.op.temperature,
            play_bias_p=self.op.play_bias_p,
            explore_floor=self.op.explore_floor,
        )
        self.rng = np.random.default_rng(10_000 + int(seed))
        self.warm_started: Optional[Dict[str, object]] = None
        if warm_start is not None:
            self.warm_started = self.setup.plast.load_weights(Path(warm_start))
            if verbose:
                print(f"[fly] WARM START from {warm_start}", flush=True)
        elif verbose:
            print("[fly] naive: every KC -> MBON synapse at its original weight",
                  flush=True)
        #: Kenyon-cell spike counts of the most recent decision window -- the
        #: presynaptic half of the coincidence, needed when dopamine arrives
        #: some seconds later once the game has resolved the hand.
        self.last_kc_counts: Optional[NDArray[np.int32]] = None
        self.last_counts: Optional[NDArray[np.int32]] = None
        self.last_substeps: List[NDArray[np.int32]] = []
        self.n_reward = 0
        self.n_punish = 0
        if verbose:
            d = self.setup.plast.describe()
            print(f"[fly] {d['kc_mbon_edges']} KC -> MBON synapses "
                  f"({d['reward_edges']} PAM-gated, {d['punish_edges']} PPL1-gated), "
                  f"{len(self.decider.approach)} approach / "
                  f"{len(self.decider.avoid)} avoid MBONs", flush=True)
            print(f"[fly] eta_reward={self.op.eta_reward:g} "
                  f"eta_punish={self.op.eta_punish:g} "
                  f"({self.op.eta_source})", flush=True)

    # -- deciding ----------------------------------------------------------
    def decide(self, ctx: object) -> Dict[str, object]:
        """One 50 ms window from reset, then the fly's own play/dig call.

        Mirrors ``plast_common.FlyPolicy.__call__`` exactly -- same drive, same
        rule, same exploration -- but keeps the raw counts and the per-sub-step
        spikes, which the batch runs have no use for and the viewer does.
        """
        bits = np.asarray(ctx.bits, dtype=np.float32)  # type: ignore[attr-defined]
        t0 = time.perf_counter()
        counts, fired, sim_seconds = run_window_substeps(
            self.setup, bits, self.n_substeps
        )
        wall = time.perf_counter() - t0
        kcc = counts[self.setup.kc]
        self.last_counts = counts
        self.last_kc_counts = kcc.copy()
        self.last_substeps = fired

        raw = self.decider.raw_drive(counts)
        drive = raw + self.decider.bias
        p = self.decider.p_play(counts)
        discard_ok = bool(getattr(ctx, "discard_ok", False))
        if not discard_ok:
            action, explored = PLAY, False
        else:
            action = PLAY if float(self.rng.random()) < p else DISCARD
            explored = (action == PLAY) != (p >= 0.5)

        scale = 1000.0 / self.setup.window_ms
        return {
            "action": action,
            "raw_drive": round(raw, 4),
            "play_drive": round(drive, 4),
            "p_play": round(p, 4),
            "explored": bool(explored),
            "discard_ok": discard_ok,
            "approach_hz": round(float(counts[self.decider.approach].mean()) * scale, 3),
            "avoid_hz": round(float(counts[self.decider.avoid].mean()) * scale, 3),
            "kc_active": int((kcc > 0).sum()),
            "kc_spikes": int(kcc.sum()),
            "bias": round(self.decider.bias, 4),
            "temperature": round(self.decider.temperature, 4),
            "sim_seconds": round(float(sim_seconds), 4),
            "window_seconds": round(wall, 4),
        }

    # -- learning ----------------------------------------------------------
    def deliver(self, kind: str) -> Dict[str, object]:
        """One dopamine pulse against the Kenyon cells of the *decided* hand.

        The eligibility trace is :attr:`last_kc_counts`, which is the window the
        fly took the decision in -- not a fresh window on the post-play state.
        That is the coincidence the rule is about.
        """
        eta = self.op.eta_reward if kind == "reward" else self.op.eta_punish
        if not self.learning or eta <= 0.0 or self.last_kc_counts is None:
            return {"dopamine": "none", "eta": 0.0, "n_changed": 0,
                    "mean_abs_dw": 0.0}
        info = self.setup.plast.deliver(kind, self.last_kc_counts, eta)
        if kind == "reward":
            self.n_reward += 1
        else:
            self.n_punish += 1
        return {
            "dopamine": kind,
            "eta": eta,
            "n_changed": int(info["n_changed"]),
            "mean_abs_dw": round(float(info["mean_abs_dw"]), 8),
            "elig_kc": int(info["elig_kc"]),
        }

    def weight_state(self) -> Dict[str, object]:
        """What the synapses look like now, in the four numbers worth watching."""
        s = self.setup.plast.weight_stats()
        n = int(s["n_edges"])
        return {
            "n_synapses": n,
            "frac_changed": round(1.0 - float(s["frac_unchanged"]), 5),
            "n_changed": int(round((1.0 - float(s["frac_unchanged"])) * n)),
            "mean_ratio": round(float(s["mean_ratio"]), 6),
            "min_ratio": round(float(s["min_ratio"]), 6),
            "frac_at_floor": round(float(s["frac_at_floor"]), 6),
            "n_at_floor": int(round(float(s["frac_at_floor"]) * n)),
            "n_reward_pulses": self.n_reward,
            "n_punish_pulses": self.n_punish,
        }

    def save_weights(self, path: Path, meta: Optional[Dict[str, object]] = None) -> Path:
        payload = {"source": "flybalatro.realgame.plastic", "learning": self.learning}
        payload.update(meta or {})
        payload.update(self.op.as_dict())
        return self.setup.plast.save_weights(Path(path), meta=payload)

    def describe(self) -> Dict[str, object]:
        d = dict(self.setup.plast.describe())
        d.update({
            "config": str(self.config_path),
            "operating_point": self.op.as_dict(),
            "n_approach_mbons": int(len(self.decider.approach)),
            "n_avoid_mbons": int(len(self.decider.avoid)),
            "window_ms": float(self.setup.window_ms),
            "learning": self.learning,
            "warm_started": self.warm_started is not None,
        })
        return d


# --------------------------------------------------------------------------- #
# what the run looks like over time                                            #
# --------------------------------------------------------------------------- #
@dataclass
class LearningTrace:
    """Running summaries the viewer draws, kept here so it need not recompute.

    All of it is a function of the hands already played, so a replay of
    ``log.jsonl`` shows exactly what the live run showed.
    """

    rewards: List[int] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    buckets: List[str] = field(default_factory=list)
    mean_ratio: List[float] = field(default_factory=list)
    frac_changed: List[float] = field(default_factory=list)

    def observe(self, bucket: str, action: str, reward: int,
                weights: Mapping[str, object]) -> None:
        self.buckets.append(bucket)
        self.actions.append(action)
        self.rewards.append(int(reward))
        self.mean_ratio.append(float(weights["mean_ratio"]))
        self.frac_changed.append(float(weights["frac_changed"]))

    def bucket_table(self, window: int = ROLLING) -> Dict[str, Dict[str, object]]:
        """Rolling P(play) per score-vs-needed bucket.

        The direct picture of whether the rule is forming: the odour carries the
        bucket, so a fly that has learned anything ordered should play more as
        the bucket rises.
        """
        # ``None``, never ``float("nan")``: ``json.dumps`` writes a bare ``NaN``
        # for that, which is not JSON and which ``JSON.parse`` in the browser
        # rejects outright -- one empty bucket would break the whole viewer.
        out: Dict[str, Dict[str, object]] = {
            b: {"n": 0, "plays": 0, "p_play": None}
            for b in SCORE_BUCKET_LABELS
        }
        for bucket, action in zip(self.buckets[-window:], self.actions[-window:]):
            row = out.get(bucket)
            if row is None:
                continue
            row["n"] += 1
            row["plays"] += int(action == PLAY)
        for row in out.values():
            n = int(row["n"])          # type: ignore[arg-type]
            if n:
                row["p_play"] = round(int(row["plays"]) / n, 4)  # type: ignore[arg-type]
        return out

    def reward_rate(self, window: int = ROLLING) -> Optional[float]:
        tail = self.rewards[-window:]
        if not tail:
            return None
        return round(sum(tail) / len(tail), 4)

    def spark(self, values: Sequence[float], n: int = 60) -> List[float]:
        return [round(float(v), 6) for v in list(values)[-n:]]

    def as_dict(self) -> Dict[str, object]:
        # Every key here is merged into the record's ``mb`` block alongside the
        # per-decision fields, so none of them may collide with one. The rolling
        # play rate is ``p_play_rolling`` for exactly that reason: ``p_play`` is
        # this hand's probability and the two are different numbers.
        return {
            "n_hands": len(self.actions),
            "reward_rate": self.reward_rate(),
            "p_play_rolling": (round(sum(a == PLAY for a in self.actions[-ROLLING:])
                                     / len(self.actions[-ROLLING:]), 4)
                               if self.actions else None),
            "buckets": self.bucket_table(),
            "reward_spark": self.spark(
                [float(x) for x in self.rewards], 60),
            "mean_ratio_spark": self.spark(self.mean_ratio, 60),
            "frac_changed_spark": self.spark(self.frac_changed, 60),
            "rolling_window": ROLLING,
        }


# --------------------------------------------------------------------------- #
# the loop                                                                     #
# --------------------------------------------------------------------------- #
def _pending_outcome(
    ctx: object,
    action: str,
    before: ad.RealGameState,
    after: ad.RealGameState,
) -> Dict[str, object]:
    """Resolve one played hand against what the real game then showed.

    Delegates the *definition* of reward and punishment to
    ``plast_common.resolve_outcome``, which is the only place in the project
    that defines them, so the real game and the headless runs cannot drift
    apart. Everything here is the translation of a BalatroBot state into the
    ``info`` dict that function expects.
    """
    _scripts_on_path()
    import plast_common as K  # noqa: E402

    stage_after = int(after.stage.int())
    chips_gained = max(0, int(after.score) - int(before.score))
    if stage_after in (ad.STAGE_POST_BLIND, ad.STAGE_SHOP) or after.ante_num > before.ante_num:
        # The blind is over: the round score is reset or banked, so the
        # difference above is meaningless. What the hand scored is what was
        # still needed, at least.
        chips_gained = max(chips_gained, int(getattr(ctx, "needed", 0)))
    done = stage_after in (ad.STAGE_END_WIN, ad.STAGE_END_LOSE)
    info = {
        "chips_gained": chips_gained,
        "stage": stage_after,
        "truncated": False,
        "is_win": stage_after == ad.STAGE_END_WIN,
    }
    return K.resolve_outcome(ctx, action, info, done)


def play_plastic(
    client: BalatroBot,
    fly: PlasticFly,
    logger: object,
    deck: str = "RED",
    stake: str = "WHITE",
    seed: Optional[str] = None,
    pause: float = 1.0,
    max_hands: int = 200,
    stop_after_ante: int = 1,
    start_new_run: bool = True,
    max_consecutive_errors: int = 5,
    verbose: bool = True,
    spikes: object = None,
    trace: Optional[LearningTrace] = None,
) -> Dict[str, object]:
    """Play real Balatro, one hand per iteration, learning from the chips.

    Each iteration:

    1. read the game, build a :class:`~flybalatro.realgame.adapter.RealGameState`;
    2. **resolve the previous hand** -- reward if it cleared the blind or paid
       its fair share, punishment if it lost the run -- and deliver the dopamine
       pulse against the Kenyon cells that were active when the fly decided;
    3. analyse the new hand outside the brain, relay it as 32 glomerular bits,
       run one 50 ms window, and let the approach-minus-avoidance MBON drive
       choose PLAY or DIG;
    4. submit the best subset (play) or the junk (dig) through the mod.

    Step 2 comes before step 3 deliberately: the pulse must land on the window
    the decision was taken in, and by the time the game has resolved a hand that
    window is one iteration old.
    """
    _scripts_on_path()
    import plast_common as K  # noqa: E402

    trace = trace if trace is not None else LearningTrace()
    if start_new_run:
        if verbose:
            print(f"[run] starting {deck} deck / {stake} stake"
                  + (f" / seed {seed}" if seed else ""), flush=True)
        client.menu()
        client.start(deck=deck, stake=stake, seed=seed)

    hands_played = 0
    decisions = 0
    blinds_attempted: List[str] = []
    best_score = 0
    dopamine_counts: Dict[str, int] = {"reward": 0, "punish": 0, "none": 0}
    stop_reason = "max_hands"
    consecutive_errors = 0
    t_start = time.time()
    last_state: Optional[ad.RealGameState] = None
    #: (hand context, action, state before the call) of a hand the game has not
    #: resolved yet.
    pending: Optional[Tuple[object, str, ad.RealGameState, Dict[str, object]]] = None

    def resolve(after: ad.RealGameState) -> Optional[Dict[str, object]]:
        """Settle ``pending`` against ``after`` and fire the dopamine pulse."""
        nonlocal pending
        if pending is None:
            return None
        ctx, action, before, _mb = pending
        pending = None
        outcome = _pending_outcome(ctx, action, before, after)
        kind = "reward" if outcome["reward"] else ("punish" if outcome["punish"] else "none")
        pulse = fly.deliver(kind) if kind != "none" else {
            "dopamine": "none", "eta": 0.0, "n_changed": 0, "mean_abs_dw": 0.0
        }
        dopamine_counts[kind] = dopamine_counts.get(kind, 0) + 1
        weights = fly.weight_state()
        trace.observe(str(getattr(ctx, "bucket_name", "")), action,
                      int(outcome["reward"]), weights)
        record = {
            "t": time.time(),
            "kind": "outcome",
            "hand": hands_played,
            "action": action,
            "action_name": action,
            # Top level as well as inside ``mb``: the overlay flashes on this
            # key and must not have to know about the plastic schema.
            "dopamine": kind if kind != "none" else None,
            "outcome": outcome,
            "mb": {**pulse, "weights": weights, **trace.as_dict()},
            # The board is unchanged by a pulse, but carrying it keeps every
            # record self-contained so a reader never has to join two lines.
            "state": _state_summary(before, ()),
            "relayed_hand_analysis": {
                "active": True,
                "best_type": str(getattr(ctx, "best_type_name", "")),
                "best_score": int(getattr(ctx, "best_score", 0)),
                "best_slots": [int(s) for s in getattr(ctx, "best_slots", ())],
                "needed": int(getattr(ctx, "needed", 0)),
                "score_bucket": str(getattr(ctx, "bucket_name", "")),
                "selected_type": "none",
                "selected_is_best": bool(action == PLAY),
            },
            "policy_mode": "plastic",
            "policy_uses_brain": True,
        }
        logger.write(record)                                  # type: ignore[attr-defined]
        if verbose:
            flash = {"reward": "DA+ reward", "punish": "DA- punish"}.get(kind, "no pulse")
            print(f"      -> {flash}; chips {outcome['chips_gained']} "
                  f"cleared={outcome['cleared']} lost={outcome['lost']} | "
                  f"synapses changed {weights['frac_changed']:.1%} "
                  f"mean w/w0 {weights['mean_ratio']:.4f}", flush=True)
        return record

    while hands_played < max_hands:
        payload = client.wait_until_stable()
        raw_state = payload.get("state")
        if raw_state == "MENU":
            stop_reason = "back_at_menu"
            break
        if isinstance(raw_state, str) and raw_state in TRANSIENT_STATES:
            stop_reason = f"stuck_in_{raw_state}"
            break
        try:
            state = ad.build_state(payload, ())
        except ad.UnmappableStateError as exc:
            stop_reason = f"unmappable_state:{exc}"
            break
        last_state = state
        if state.blind_name and (not blinds_attempted or blinds_attempted[-1] != state.blind_name):
            if state.stage.int() in ad.BLIND_STAGES:
                blinds_attempted.append(state.blind_name)
        best_score = max(best_score, state.score)

        # (2) The previous hand, now that the game has finished animating it.
        resolve(state)

        stage = state.stage.int()
        if stage in (ad.STAGE_END_WIN, ad.STAGE_END_LOSE):
            stop_reason = "game_over_win" if stage == ad.STAGE_END_WIN else "game_over_lose"
            break
        if state.ante_num > stop_after_ante:
            stop_reason = "ante_cleared"
            break

        # Non-hand screens: advance them without involving the fly. The fly's
        # decision is about a hand of cards; nothing here is one, and buying
        # jokers is outside this experiment exactly as it was in the readout
        # runs (which left every shop immediately).
        try:
            if stage == ad.STAGE_PRE_BLIND:
                client.select()
                continue
            if stage == ad.STAGE_POST_BLIND:
                client.cash_out()
                continue
            if stage == ad.STAGE_SHOP:
                client.next_round()
                continue
            if stage not in ad.BLIND_STAGES:
                stop_reason = f"unhandled_stage_{state.raw_state}"
                break
        except BalatroBotError as exc:
            consecutive_errors += 1
            logger.write({"t": time.time(), "error": str(exc),   # type: ignore[attr-defined]
                          "error_name": exc.name, "stage": stage,
                          "consecutive_errors": consecutive_errors})
            if consecutive_errors >= max_consecutive_errors:
                stop_reason = f"too_many_rejected_calls:{exc.name}"
                break
            continue

        # (3) A hand. Analyse it outside the brain, relay it, let the fly decide.
        # Explicitly with NOTHING selected. ``build_state`` seeds the virtual
        # selection from the game's own ``card.highlight`` flags when any card
        # carries one, and ``features_v2.hand_block`` encodes the selected hand
        # type and the "play it now" bit from that -- so a stray highlight would
        # hand the fly a different odour from the one the plast2 bias and
        # temperature were calibrated on (which used the sel-none bit).
        ctx = K.hand_context(state.with_selection(()))
        if ctx is None:
            stop_reason = f"no_hand_in_{state.raw_state}"
            break
        mb = fly.decide(ctx)
        action = str(mb["action"])
        slots = tuple(ctx.best_slots) if action == PLAY else tuple(ctx.dig)
        if not slots:
            # Nothing to throw: the whole hand is the best subset. Play it.
            action, slots = PLAY, tuple(ctx.best_slots)
            mb["action"] = action
            mb["forced"] = "no cards outside the best subset to discard"
        decisions += 1

        spike_ref = None
        if spikes is not None and getattr(spikes, "wanted", lambda: False)():
            if fly.last_substeps:
                spike_ref = spikes.write(                      # type: ignore[attr-defined]
                    fly.last_substeps,
                    fly.setup.window_ms / max(1, fly.n_substeps),
                    fly.setup.brain.n,
                    total_spikes=(int(fly.last_counts.sum())
                                  if fly.last_counts is not None else None),
                    n_spiking=(int((fly.last_counts > 0).sum())
                               if fly.last_counts is not None else None),
                )

        record: Dict[str, object] = {
            "t": time.time(),
            "kind": "decision",
            "decision": decisions,
            "hand": hands_played + 1,
            "condition": fly.setup.wiring,
            "policy_mode": "plastic",
            "policy_uses_brain": True,
            "action_name": action,
            "action_kind": "rpc",
            "action_description": f"{action} hand slots {list(slots)}",
            "window_ms": float(fly.setup.window_ms),
            "sim_seconds": mb["sim_seconds"],
            "window_seconds": mb["window_seconds"],
            "spikes": spike_ref,
            # Same keys ``features_v2.hand_block_summary`` writes for the
            # readout runs, so one overlay and one viewer read both.
            "relayed_hand_analysis": {
                "active": True,
                "best_type": ctx.best_type_name,
                "best_score": int(ctx.best_score),
                "best_slots": [int(s) for s in ctx.best_slots],
                "needed": int(ctx.needed),
                "score_bucket": ctx.bucket_name,
                "selected_type": "none",
                "selected_is_best": bool(action == PLAY),
            },
            "n_bits_on": int(np.asarray(ctx.bits).sum()),
            "mb": {
                **{k: v for k, v in mb.items()
                   if k not in ("sim_seconds", "window_seconds")},
                "bucket": ctx.bucket_name,
                "best_type": ctx.best_type_name,
                "best_score": int(ctx.best_score),
                "needed": int(ctx.needed),
                "slots": [int(s) for s in slots],
                "weights": fly.weight_state(),
                **trace.as_dict(),
            },
            "state": _state_summary(state, slots),
        }
        logger.write(record)                                   # type: ignore[attr-defined]
        if verbose:
            print(
                f"[{decisions:3d}] {state.raw_state:<14} "
                f"score={state.score}/{state.required_score} "
                f"h={state.plays} d={state.discards} | "
                f"{ctx.best_type_name} {ctx.best_score} vs {ctx.needed} "
                f"({ctx.bucket_name}) | app {mb['approach_hz']:.1f} - "
                f"avo {mb['avoid_hz']:.1f} Hz -> P(play) {mb['p_play']:.2f} "
                f"=> {action.upper()} {list(slots)}"
                + ("  [explored]" if mb["explored"] else ""),
                flush=True,
            )

        # (4) Press the keys.
        try:
            if action == PLAY:
                client.play(cards=sorted(int(s) for s in slots))
            else:
                client.discard(cards=sorted(int(s) for s in slots))
        except BalatroBotError as exc:
            consecutive_errors += 1
            logger.write({"t": time.time(), "decision": decisions,   # type: ignore[attr-defined]
                          "error": str(exc), "error_name": exc.name,
                          "action_name": action,
                          "consecutive_errors": consecutive_errors})
            if verbose:
                print(f"      !! {exc}", flush=True)
            if consecutive_errors >= max_consecutive_errors:
                stop_reason = f"too_many_rejected_calls:{exc.name}"
                break
            continue
        consecutive_errors = 0
        hands_played += 1
        # A DISCARD gets no dopamine ever -- crediting a re-deal across a
        # discard is delayed credit assignment and beyond a fly -- but it still
        # has to be settled so the next hand starts clean.
        pending = (ctx, action, state, mb)
        if pause > 0:
            time.sleep(pause)

    # The run is over: settle anything outstanding against the last state seen.
    if pending is not None and last_state is not None:
        resolve(last_state)

    summary: Dict[str, object] = {
        "stop_reason": stop_reason,
        "mode": "plastic",
        "hands": hands_played,
        "decisions": decisions,
        "wall_seconds": round(time.time() - t_start, 1),
        "condition": fly.setup.wiring,
        "policy_mode": "plastic",
        "policy_detail": (
            "MBON approach - avoidance drive, no trained action readout; "
            "KC -> MBON synapses depressed by dopamine computed from chips"
        ),
        "policy_uses_brain": True,
        "blinds_attempted": blinds_attempted,
        "best_round_score": best_score,
        "dopamine": dopamine_counts,
        "fly": fly.describe(),
        "weights": fly.weight_state(),
        "learning": trace.as_dict(),
        "final": _state_summary(last_state, ()) if last_state is not None else None,
    }
    out_dir = Path(getattr(logger, "out_dir", ROOT / "outputs" / "realgame"))
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if verbose:
        print(f"[run] {stop_reason} after {hands_played} hands "
              f"({dopamine_counts['reward']} reward, "
              f"{dopamine_counts['punish']} punishment pulses)", flush=True)
    return summary


def _state_summary(state: Optional[ad.RealGameState],
                   slots: Sequence[int]) -> Optional[Dict[str, object]]:
    """The subset of the state the overlay and the viewer read.

    Deliberately the *same keys* ``play._state_summary`` produces -- ``hand``,
    ``selected_indices``, ``blind`` -- so ``overlay.parse_record`` reads a
    plastic log and a readout log with the same code. Two keys are added, both
    optional and both ignored by an older reader: each card's ``rect``, which is
    where the game says that card actually is on screen right now, and
    ``screen``, the frame those rects live in.
    """
    if state is None:
        return None
    hand: List[Dict[str, object]] = []
    for i, card in enumerate(state.available):
        entry: Dict[str, object] = {
            "slot": i, "rank_index": card.rank_index,
            "suit_index": card.suit_index, "label": card.label,
        }
        if card.geometry is not None:
            entry["rect"] = card.geometry.as_dict()
        hand.append(entry)
    out: Dict[str, object] = {
        "raw_state": state.raw_state,
        "stage": int(state.stage.int()),
        "ante": int(state.ante_num),
        "round": int(state.round_num),
        "blind": state.blind_name,
        "blind_type": state.blind_type,
        "score": int(state.score),
        "required_score": int(state.required_score),
        "plays": int(state.plays),
        "discards": int(state.discards),
        "money": int(state.money),
        "n_jokers": len(state.jokers),
        "hand": hand,
        # The fly's selection is virtual: the API has no selection endpoint, so
        # this is the subset it is about to submit, not something the game has
        # highlighted. Same convention as the readout runs.
        "selected_indices": [int(s) for s in slots],
    }
    if state.screen is not None:
        out["screen"] = state.screen.as_dict()
    return out
