"""Live-learning mode: the plasticity harness driving the viewer, hand by hand.

The other viewer modes (``flybalatro/viewer/policy.py``) put a **trained
readout** on a frozen brain and ask it to pick one of 109 actions. This mode has
no readout and nothing fitted. It is ``scripts/plast_common.py`` -- the harness
behind ``outputs/plast`` and ``outputs/plast2`` -- stepped one hand at a time so
the browser can watch each stage:

1. ``flybalatro/hands.py`` enumerates all 218 subsets of the dealt cards,
   classifies and scores them with Balatro's own rules, and reports the best
   one. **That is computed outside the brain** and the screen says so.
2. ``flybalatro/features_v2.hand_block`` turns the answer into 32 binary
   channels and ``flybalatro.encode.GlomerularMap`` drives each of them onto
   every olfactory receptor neuron of one whole ORN glomerulus: the hand's
   "odour".
3. One 50 ms window from reset, run as five 10 ms sub-steps so the point cloud
   can play the wave back.
4. The decision comes out of the mushroom-body output neurons by a fixed rule:
   ``play_drive = mean rate(approach MBONs) - mean rate(avoid MBONs) + bias``,
   ``P(play) = floor/2 + (1 - floor) * sigmoid(drive / T)``. ``bias`` and ``T``
   are the two scalars the training run calibrated once before any dopamine and
   then froze; they arrive with the weights file or are read from the run json.
   Nothing else about the decision is learned or fitted.
5. The harness presses the keys -- select the best subset and play, or select
   the junk and discard -- and the game resolves the hand.
6. ``plast_common.resolve_outcome`` says whether that play paid its way. If it
   did, a PAM (reward) pulse; if it lost the blind, a PPL1 (punishment) pulse;
   a discard gets nothing. The pulse depresses the KC -> MBON synapses of the
   Kenyon cells that fired in step 3, in the compartments that dopamine
   population gates -- :class:`flybalatro.plasticity.KcMbonPlasticity`.

So the only thing that changes on screen is the weight of 33,496 real synapses,
changed by the fly's own rule from the game's own chips.

Two flies live here. ``v1`` is ``outputs/plast/tuned_config.json``, which learned
"always play" -- one valence for every odour. ``v2`` is
``outputs/plast2/tuned_config.json``, the same fly plus per-Kenyon-cell
homeostatic thresholds, which learns an odour-specific rule; its bucket bars
separate on screen and v1's do not. Both are documented in ``RESULTS.md``.

Layering, because one of the two halves has to be testable without a connectome:

:class:`PlasticCore`
    decision + dopamine + statistics. Takes spike counts and a resolved
    outcome; knows nothing about Balatro, the environment or the point cloud.
:class:`PlasticEngine`
    the harness: config, ``Setup`` (graph, glomerular map, brain, valence,
    plastic synapses), the environment, hand stepping and the wire frames.
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
# scripts/ is not a package on the path when the viewer is started as
# ``python -m flybalatro.viewer.server``; plast_common itself inserts both ROOT
# and scripts/ into sys.path, so importing it as ``scripts.plast_common`` is
# enough once ROOT is there.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .. import plasticity as P  # noqa: E402
from ..features_v2 import N_HAND_BLOCK, SCORE_BUCKET_LABELS  # noqa: E402

__all__ = [
    "BUCKET_LABELS",
    "PLASTIC_CONFIGS",
    "PlasticSpec",
    "PlasticCore",
    "PlasticEngine",
    "PlasticPolicy",
    "resolve_spec",
    "dan_indices",
]

#: The score-vs-needed bucket, in plain words. ``lt0.25`` means the best hand
#: available scores less than a quarter of what is still needed.
BUCKET_LABELS: Tuple[Tuple[str, str], ...] = (
    ("lt0.25", "far behind"),
    ("lt0.5", "behind"),
    ("lt1.0", "close"),
    ("ge1.0", "enough"),
)

#: Rolling windows, in hands: long enough that a bucket bar is not one decision
#: wide, short enough that the screen shows the fly as it is now.
BUCKET_WINDOW: int = 240
REWARD_WINDOW: int = 50
HISTORY_POINTS: int = 160


# --------------------------------------------------------------------------- #
# which fly
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlasticSpec:
    """One of the two plasticity flies, and where its numbers come from."""

    version: str                 # "v1" | "v2"
    config_path: Path
    out_dir: Path
    label: str
    homeostasis: bool
    #: Reward learning rate, and the separately calibrated punishment one. v1
    #: used a single eta for both arms; v2 matched them per pulse
    #: (``outputs/plast2/eta_calib.json``, ratio 3.81).
    eta_reward: float
    eta_punish: float
    #: The run whose calibration (and saved weights) this mode defaults to.
    reference_run: str

    @property
    def weights_path(self) -> Path:
        return self.out_dir / f"weights_{self.reference_run}.npz"


PLASTIC_CONFIGS: Dict[str, PlasticSpec] = {
    "v1": PlasticSpec(
        version="v1",
        config_path=ROOT / "outputs" / "plast" / "tuned_config.json",
        out_dir=ROOT / "outputs" / "plast",
        label="plasticity v1 (no KC homeostasis)",
        homeostasis=False,
        eta_reward=0.05,
        eta_punish=0.05,
        reference_run="real_eta0.05_s0",
    ),
    "v2": PlasticSpec(
        version="v2",
        config_path=ROOT / "outputs" / "plast2" / "tuned_config.json",
        out_dir=ROOT / "outputs" / "plast2",
        label="plasticity v2 (per-KC homeostatic thresholds)",
        homeostasis=True,
        eta_reward=0.05,
        eta_punish=0.1907,
        reference_run="real_eta0.05_s0",
    ),
}


def resolve_spec(version: str) -> PlasticSpec:
    try:
        return PLASTIC_CONFIGS[version]
    except KeyError:
        raise ValueError(
            f"--plastic-config must be one of {sorted(PLASTIC_CONFIGS)}, "
            f"got {version!r}"
        ) from None


def dan_indices(graph) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    """(PAM, PPL1) neuron indices, by ``type`` prefix.

    The same test :func:`flybalatro.plasticity.compartment_dominance` uses to
    decide which compartment a KC -> MBON synapse sits in, so the populations
    the screen flashes are exactly the ones the rule gates on.
    """
    types = graph.type.astype(str)
    pam = np.flatnonzero(np.char.startswith(types, "PAM")).astype(np.int32)
    ppl1 = np.flatnonzero(np.char.startswith(types, "PPL1")).astype(np.int32)
    return pam, ppl1


def calibration_from_runs(spec: PlasticSpec) -> Optional[dict]:
    """``bias`` / ``temperature`` from any run json in this fly's output dir.

    The calibration is cached per ``(tuning, wiring, valence, window)`` in
    ``scripts/plast_run.py`` and computed on the *naive* fly, so every run of one
    config shares it and reading it back costs nothing. Preferred over
    recomputing it live, which is 200 more 50 ms windows at startup.
    """
    preferred = spec.out_dir / f"run_{spec.reference_run}.json"
    candidates = [preferred] if preferred.exists() else []
    candidates += sorted(p for p in spec.out_dir.glob("run_*.json")
                         if p != preferred)
    for path in candidates:
        try:
            cal = json.loads(path.read_text()).get("calibration")
        except (OSError, ValueError):
            continue
        if isinstance(cal, dict) and "bias" in cal and "temperature" in cal:
            out = dict(cal)
            out["source"] = str(path.relative_to(ROOT))
            return out
    return None


# --------------------------------------------------------------------------- #
# decision + dopamine, with no game and no connectome
# --------------------------------------------------------------------------- #
PLAY = "play"
DIG = "dig"


@dataclass
class PulseInfo:
    kind: str = "none"          # "reward" | "punish" | "none"
    n_changed: int = 0
    mean_abs_dw: float = 0.0
    eta: float = 0.0
    elig_kc: int = 0


class PlasticCore:
    """The mushroom-body decision and the dopamine rule, plus rolling stats.

    Deliberately free of Balatro, the environment and the point cloud, so the
    two claims that matter -- "the decision is play or dig and nothing else",
    "learning off leaves every synapse alone, a reward pulse moves only eligible
    ones" -- are testable on a ten-neuron synthetic graph.
    """

    def __init__(self, plast: P.KcMbonPlasticity, decider: P.Decider,
                 kc_indices: NDArray[np.int32], rng: np.random.Generator,
                 learning: bool = True, eta_reward: float = 0.05,
                 eta_punish: float = 0.05) -> None:
        self.plast = plast
        self.decider = decider
        self.kc = np.asarray(kc_indices, np.int32)
        self.rng = rng
        self.learning = bool(learning)
        self.eta_reward = float(eta_reward)
        self.eta_punish = float(eta_punish)

        self.hands: int = 0
        self.plays: int = 0
        self.pulses: Dict[str, int] = {"reward": 0, "punish": 0}
        self.mean_weight_history: Deque[float] = deque(maxlen=HISTORY_POINTS)
        self.reward_history: Deque[float] = deque(maxlen=HISTORY_POINTS)
        self._recent_rewards: Deque[int] = deque(maxlen=REWARD_WINDOW)
        self._recent_buckets: Deque[Tuple[str, int]] = deque(maxlen=BUCKET_WINDOW)
        self.mean_weight_history.append(self.mean_weight())

    # -- read the mushroom body -------------------------------------------
    def read(self, counts: NDArray[np.int_]) -> dict:
        """Approach / avoid drive and P(play) for one window's spike counts."""
        dec = self.decider
        scale = 1000.0 / dec.window_ms
        c = np.asarray(counts)
        approach_hz = float(c[dec.approach].mean()) * scale if len(dec.approach) else 0.0
        avoid_hz = float(c[dec.avoid].mean()) * scale if len(dec.avoid) else 0.0
        raw = approach_hz - avoid_hz
        drive = raw + dec.bias
        return dict(
            approach_hz=approach_hz,
            avoid_hz=avoid_hz,
            raw_drive=raw,
            play_drive=drive,
            p_play=dec.p_play(c),
            bias=float(dec.bias),
            temperature=float(dec.temperature),
            explore_floor=float(dec.explore_floor),
        )

    def decide(self, counts: NDArray[np.int_], discard_ok: bool) -> dict:
        """PLAY or DIG. Sampled at P(play) while learning, greedy when frozen.

        The exploration floor is what keeps an odour learnable at all: a DIG
        delivers no dopamine, so an odour the fly is confident about digging can
        never generate the experience that would change its mind. With learning
        off there is nothing to learn, so the fly acts on its best guess.
        """
        info = self.read(counts)
        p = info["p_play"]
        if not discard_ok:
            info.update(action=PLAY, explored=False, forced=True)
            return info
        if self.learning:
            action = PLAY if float(self.rng.random()) < p else DIG
            explored = (action == PLAY) != (p >= 0.5)
        else:
            action = PLAY if p >= 0.5 else DIG
            explored = False
        info.update(action=action, explored=bool(explored), forced=False)
        return info

    # -- the dopamine rule -------------------------------------------------
    def learn(self, action: str, outcome: dict,
              kc_counts: NDArray[np.int_]) -> PulseInfo:
        """Deliver the pulse this outcome earns, if learning is on."""
        if not self.learning:
            return PulseInfo()
        if action == PLAY and outcome.get("reward") and self.eta_reward > 0.0:
            info = self.plast.deliver("reward", kc_counts, self.eta_reward)
            self.pulses["reward"] += 1
            return PulseInfo("reward", info["n_changed"], info["mean_abs_dw"],
                             self.eta_reward, info["elig_kc"])
        if action == PLAY and outcome.get("punish") and self.eta_punish > 0.0:
            info = self.plast.deliver("punish", kc_counts, self.eta_punish)
            self.pulses["punish"] += 1
            return PulseInfo("punish", info["n_changed"], info["mean_abs_dw"],
                             self.eta_punish, info["elig_kc"])
        return PulseInfo()

    # -- bookkeeping -------------------------------------------------------
    def record(self, action: str, bucket: str, discard_ok: bool,
               outcome: dict) -> None:
        self.hands += 1
        if action == PLAY:
            self.plays += 1
            self._recent_rewards.append(int(bool(outcome.get("reward"))))
        if discard_ok:
            self._recent_buckets.append((bucket, int(action == PLAY)))
        self.mean_weight_history.append(self.mean_weight())
        self.reward_history.append(
            float(np.mean(self._recent_rewards)) if self._recent_rewards else 0.0
        )

    def mean_weight(self) -> float:
        w = self.plast.brain.weight[self.plast.all_edge]
        return float(w.mean()) if len(w) else 0.0

    def mean_w0(self) -> float:
        w0 = self.plast.all_w0
        return float(w0.mean()) if len(w0) else 0.0

    def weight_summary(self) -> dict:
        """What the weight panel shows, plus the invariant it has to hold."""
        stats = self.plast.weight_stats()
        floor = self.plast.floor_frac
        ratio = (self.plast.brain.weight[self.plast.all_edge]
                 / self.plast.all_w0) if stats["n_edges"] else np.zeros(0)
        return dict(
            n_edges=stats["n_edges"],
            frac_changed=1.0 - stats["frac_unchanged"],
            n_changed=int(round((1.0 - stats["frac_unchanged"]) * stats["n_edges"])),
            frac_at_floor=stats["frac_at_floor"],
            n_at_floor=int((ratio <= floor + 1e-6).sum()) if len(ratio) else 0,
            mean_ratio=stats["mean_ratio"],
            min_ratio=stats["min_ratio"],
            mean_weight=self.mean_weight(),
            mean_w0=self.mean_w0(),
            floor=floor,
            any_negative=stats["any_negative"],
            any_above_original=stats["any_above_original"],
        )

    def bucket_table(self) -> List[dict]:
        """Rolling P(play) per score-vs-needed bucket, decisions where digging
        was legal. This is the direct read of "did it learn the rule": the fly's
        odour carries the bucket, so a fly that learned a valence *per odour*
        separates these four bars and a fly that learned one number does not."""
        rows: List[dict] = []
        for key, label in BUCKET_LABELS:
            n = plays = 0
            for bucket, played in self._recent_buckets:
                if bucket == key:
                    n += 1
                    plays += played
            rows.append(dict(bucket=key, label=label, n=n, plays=plays,
                             p_play=(plays / n) if n else None))
        return rows

    def reward_rate(self) -> Optional[float]:
        if not self._recent_rewards:
            return None
        return float(np.mean(self._recent_rewards))


# --------------------------------------------------------------------------- #
# the harness
# --------------------------------------------------------------------------- #
class PlasticEngine:
    """One hand of Balatro per call, with the fly's synapses changing live."""

    def __init__(self, graph, spec: PlasticSpec, learning: bool = True,
                 weights: Optional[Path] = None, seed0: Optional[int] = None,
                 explore_floor: float = 0.1, rng_seed: int = 0,
                 max_hands_per_game: int = 60) -> None:
        import scripts.plast_common as K  # noqa: PLC0415

        self.K = K
        self.spec = spec
        self.cfg = json.loads(spec.config_path.read_text())
        self.setup = K.build_setup(wiring="real", valence_scheme="nt",
                                   graph=graph, cfg=self.cfg)
        self.decider = K.make_decider(self.setup, explore_floor=explore_floor)
        self.notes: List[str] = []

        # Calibration: the two frozen scalars the training run measured on 200
        # naive hands. They must be the run's, not a fresh guess, or the saved
        # weights are being read by a different decision rule.
        self.weights_meta: Optional[dict] = None
        self.weights_path: Optional[Path] = None
        loaded = self._load_weights(weights)
        cal = (loaded or {}).get("meta") or {}
        if "bias" in cal and "temperature" in cal:
            self.decider.bias = float(cal["bias"])
            self.decider.temperature = float(cal["temperature"])
            self.calibration_source = f"{Path(str(loaded['path'])).name} metadata"
        else:
            run_cal = calibration_from_runs(spec)
            if run_cal is None:
                raise RuntimeError(
                    f"no calibration available for {spec.version}: neither a "
                    f"weights file nor a run json under {spec.out_dir}"
                )
            self.decider.bias = float(run_cal["bias"])
            self.decider.temperature = float(run_cal["temperature"])
            self.calibration_source = str(run_cal.get("source", "run json"))
        self.notes.append(
            f"decision bias {self.decider.bias:+.3f} Hz, temperature "
            f"{self.decider.temperature:.3f} Hz from {self.calibration_source}"
        )

        eta_r = float(cal.get("eta_reward", spec.eta_reward))
        eta_p = float(cal.get("eta_punish", spec.eta_punish))
        self.core = PlasticCore(
            plast=self.setup.plast, decider=self.decider,
            kc_indices=self.setup.kc, rng=np.random.default_rng(rng_seed),
            learning=learning, eta_reward=eta_r, eta_punish=eta_p,
        )

        # Training seeds by default, so the dashboard never plays the games the
        # published before/after numbers are measured on.
        self.seed0 = int(K.TRAIN_SEED0 if seed0 is None else seed0)
        self.env = K.make_env()
        self.max_hands_per_game = int(max_hands_per_game)
        self.episode_index = 0
        self.hands_this_game = 0
        self.games: int = 0
        self.cleared_games: int = 0
        self.pam, self.ppl1 = dan_indices(graph)
        self._started = False

    # -- weights -----------------------------------------------------------
    def _load_weights(self, weights: Optional[Path]) -> Optional[dict]:
        if weights is None:
            self.notes.append(
                "starting from the naive fly: every KC -> MBON synapse at its "
                "connectome weight, nothing learned yet"
            )
            return None
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(f"--plastic-weights {path} does not exist")
        loaded = self.setup.plast.load_weights(path)
        self.weights_path = path
        self.weights_meta = loaded.get("meta") or {}
        self.notes.append(
            f"loaded learned weights from {path.name}: mean "
            f"{loaded['mean_ratio']:.3f} of the original, "
            f"{1.0 - loaded['frac_unchanged']:.1%} of "
            f"{loaded['n_edges']:,} synapses changed, "
            f"{loaded['frac_at_floor']:.1%} at the floor"
        )
        return loaded

    # -- the fly's own numbers --------------------------------------------
    def describe(self) -> dict:
        """Static facts about this fly, for ``hello``."""
        v = self.setup.valence
        return dict(
            version=self.spec.version,
            label=self.spec.label,
            homeostasis=self.spec.homeostasis,
            learning=bool(self.core.learning),
            config=str(self.spec.config_path.relative_to(ROOT)),
            config_label=str(self.cfg.get("label", "")),
            weights=str(self.weights_path.relative_to(ROOT))
            if self.weights_path else None,
            weights_meta=self.weights_meta,
            calibration_source=self.calibration_source,
            bias=float(self.decider.bias),
            temperature=float(self.decider.temperature),
            explore_floor=float(self.decider.explore_floor),
            eta_reward=float(self.core.eta_reward),
            eta_punish=float(self.core.eta_punish),
            weight_floor=float(self.setup.plast.floor_frac),
            kc_ref=float(self.setup.plast.kc_ref),
            n_approach=int(len(v.approach)),
            n_avoid=int(len(v.avoid)),
            n_pam=int(len(self.pam)),
            n_ppl1=int(len(self.ppl1)),
            pools=self.setup.plast.describe(),
            buckets=[dict(bucket=k, label=lbl) for k, lbl in BUCKET_LABELS],
            seed0=self.seed0,
            n_kc=int(len(self.setup.kc)),
            notes=list(self.notes),
        )

    def highlight_points(self, point_index: NDArray[np.int32]) -> Dict[str, List[int]]:
        """Point-cloud indices of the populations this mode highlights.

        Kenyon cells and the rest keep their category colour; the MBONs the
        decision is read from and the two dopamine clusters that gate the two
        halves of the rule get their own, because they are what the panel is
        about. Neurons with no soma in the imaged volume are simply absent.
        """
        def pts(idx: NDArray[np.int32]) -> List[int]:
            p = point_index[np.asarray(idx, np.int64)]
            return [int(x) for x in p[p >= 0]]

        v = self.setup.valence
        return dict(
            approach=pts(v.approach),
            avoid=pts(v.avoid),
            pam=pts(self.pam),
            ppl1=pts(self.ppl1),
        )

    # -- episodes ----------------------------------------------------------
    def new_episode(self) -> None:
        self.env.reset(seed=self.seed0 + self.episode_index)
        self.hands_this_game = 0
        self._started = True

    def _housekeeping(self) -> Optional[object]:
        """Advance select-blind / cash-out / shop until a hand is on the table.

        Returns the ``HandContext`` for that hand, or ``None`` if the game ended
        first. Identical to the stage handling in ``plast_common.play_game``.
        """
        K = self.K
        for _ in range(64):
            if self.env.done:
                return None
            state = self.env.engine.state
            stage = int(state.stage.int())
            mask = self.env.action_mask()
            if stage == K.STAGE_PRE_BLIND:
                self.env.step(K.SELECT_BLIND_INDEX if mask[K.SELECT_BLIND_INDEX]
                              else int(np.flatnonzero(mask)[0]))
                continue
            if stage == K.STAGE_POST_BLIND:
                self.env.step(K.CASH_OUT_INDEX if mask[K.CASH_OUT_INDEX]
                              else int(np.flatnonzero(mask)[0]))
                continue
            if stage == K.STAGE_SHOP:
                self.env.step(K.NEXT_ROUND_INDEX if mask[K.NEXT_ROUND_INDEX]
                              else int(np.flatnonzero(mask)[0]))
                continue
            if stage not in K.STAGE_BLINDS:
                self.env.step(int(np.flatnonzero(mask)[0]))
                continue
            ctx = K.hand_context(state)
            if ctx is None:
                self.env.step(int(np.flatnonzero(mask)[0]))
                continue
            return ctx
        return None

    def _finish_episode(self) -> dict:
        won = bool(self.env.is_win)
        self.games += 1
        self.cleared_games += int(won)
        frame = dict(
            won=won,
            truncated=bool(self.env.steps >= 500),
            chips_total=int(self.env.chips_total),
            steps=int(self.env.steps),
            seed=int(self.seed0 + self.episode_index),
            games=self.games,
            cleared=self.cleared_games,
            clear_rate=self.cleared_games / max(1, self.games),
        )
        self.episode_index += 1
        return frame

    # -- the window --------------------------------------------------------
    def run_window(self, bits: NDArray[np.float32],
                   point_index: NDArray[np.int32], n_substeps: int,
                   substep_ms: float) -> Tuple[NDArray[np.int32],
                                               List[NDArray[np.uint32]], float]:
        """The decision window, observed in ``n_substeps`` slices.

        Summing the slices is identical to one call of the whole window -- the
        kernel carries its own cursor -- so this is the same 50 ms the batch runs
        use, only watched.
        """
        import time  # noqa: PLC0415

        brain = self.setup.brain
        drive = self.setup.odor_drive(bits)
        brain.reset()
        previous = np.zeros(brain.n, dtype=np.int32)
        fired: List[NDArray[np.uint32]] = []
        t0 = time.perf_counter()
        counts = previous
        for k in range(n_substeps):
            counts, _ = brain.step(drive, substep_ms, reset_counts=(k == 0))
            spiked = np.flatnonzero(counts > previous).astype(np.int64)
            pts = point_index[spiked]
            fired.append(pts[pts >= 0].astype(np.uint32))
            previous = counts.copy()
        return counts, fired, time.perf_counter() - t0

    # -- one hand ----------------------------------------------------------
    def step_hand(self, point_index: NDArray[np.int32], n_substeps: int = 5,
                  substep_ms: float = 10.0) -> dict:
        """Play one hand: odour, window, MBON decision, keys, outcome, dopamine.

        Returns a dict with everything the browser needs, or
        ``{"kind": "episode"}`` when the game ended during housekeeping.
        """
        K = self.K
        if not self._started or self.env.done:
            self.new_episode()

        ctx = self._housekeeping()
        if ctx is None:
            return dict(kind="episode", episode=self._finish_episode())
        if self.hands_this_game >= self.max_hands_per_game:
            # The harness's own cap, so a pathological game cannot stall the
            # screen. Closed out *before* the reset, or the episode frame would
            # report the next game's result.
            frame = self._finish_episode()
            self.new_episode()
            return dict(kind="episode", episode=frame)

        state = self.env.engine.state
        counts, fired, seconds = self.run_window(
            ctx.bits, point_index, n_substeps, substep_ms)
        kc_counts = counts[self.setup.kc].copy()

        info = self.core.decide(counts, ctx.discard_ok)
        action = str(info["action"])

        # Press the keys: select the target slots, then play or discard. Exactly
        # plast_common.play_game's sequence, including its fallback when the
        # engine refuses the action we intended.
        target = ctx.best_slots if action == PLAY else ctx.dig
        act_index = K.PLAY_INDEX if action == PLAY else K.DISCARD_INDEX
        for slot in target:
            mask = self.env.action_mask()
            if slot < len(mask) and mask[slot] == 1:
                self.env.step(int(slot))
        mask = self.env.action_mask()
        if mask[act_index] != 1:
            other = K.DISCARD_INDEX if act_index == K.PLAY_INDEX else K.PLAY_INDEX
            if mask[other] == 1:
                act_index = other
                action = PLAY if other == K.PLAY_INDEX else DIG
            else:
                self.env.step(int(np.flatnonzero(mask)[0]))
                return dict(kind="skip")

        board_selected = [int(s) for s in target]
        _f, _m, _r, done, env_info = self.env.step(act_index)
        self.hands_this_game += 1

        # ``resolve_outcome`` takes the harness's own PLAY/DISCARD spelling.
        outcome = K.resolve_outcome(
            ctx, K.PLAY if action == PLAY else K.DISCARD, env_info, done)
        pulse = self.core.learn(action, outcome, kc_counts)
        self.core.record(action, ctx.bucket_name, ctx.discard_ok, outcome)

        mb = dict(
            hand=dict(
                best_type=ctx.best_type_name,
                best_score=int(ctx.best_score),
                best_slots=list(ctx.best_slots),
                dig_slots=list(ctx.dig),
                needed=int(ctx.needed),
                bucket=ctx.bucket_name,
                bucket_label=dict(BUCKET_LABELS).get(ctx.bucket_name, ctx.bucket_name),
                plays=int(ctx.plays),
                discards=int(ctx.discards),
                discard_ok=bool(ctx.discard_ok),
                share=round(float(outcome["share"]), 1),
                n_bits_on=int(ctx.bits.sum()),
            ),
            approach_hz=round(info["approach_hz"], 3),
            avoid_hz=round(info["avoid_hz"], 3),
            raw_drive=round(info["raw_drive"], 3),
            play_drive=round(info["play_drive"], 3),
            p_play=round(info["p_play"], 4),
            bias=round(info["bias"], 3),
            temperature=round(info["temperature"], 3),
            explore_floor=info["explore_floor"],
            action=action,
            explored=bool(info["explored"]),
            forced=bool(info["forced"]),
            selected_slots=board_selected,
            kc_active=int((kc_counts > 0).sum()),
            kc_spikes=int(kc_counts.sum()),
            chips_gained=int(outcome["chips_gained"]),
            cleared=bool(outcome["cleared"]),
            lost=bool(outcome["lost"]),
            reward=int(outcome["reward"]),
            punish=int(outcome["punish"]),
            outcome=_outcome_text(action, outcome),
            dopamine=pulse.kind,
            dopamine_eta=round(pulse.eta, 4),
            dopamine_changed=pulse.n_changed,
            dopamine_mean_abs_dw=round(pulse.mean_abs_dw, 6),
            dopamine_elig_kc=pulse.elig_kc,
            pulses=dict(self.core.pulses),
            learning=bool(self.core.learning),
            hands=self.core.hands,
            plays=self.core.plays,
            weights=self.core.weight_summary(),
            weight_history=[round(x, 5) for x in self.core.mean_weight_history],
            reward_history=[round(x, 4) for x in self.core.reward_history],
            reward_rate=self.core.reward_rate(),
            buckets=self.core.bucket_table(),
            games=self.games,
            games_cleared=self.cleared_games,
            clear_rate=(self.cleared_games / self.games) if self.games else None,
        )
        return dict(
            kind="hand",
            state=state,
            bits=ctx.bits,
            counts=counts,
            fired=fired,
            brain_seconds=seconds,
            action_name=str(env_info["action_name"]),
            action_index=int(act_index),
            chips_gained=int(outcome["chips_gained"]),
            done=bool(done),
            mb=mb,
            episode=self._finish_episode() if done else None,
        )


class PlasticPolicy:
    """The :class:`flybalatro.viewer.policy.Policy` face of this mode.

    The server's ``status`` / ``footer`` / ``hello`` code asks a policy what it
    is and what it reads, and there is a true answer here: the MBONs. It does
    **not** pick one of 109 actions -- the loop is hand-level, not action-level,
    because one decision costs several key presses -- so ``decide`` is not part
    of this mode's path and refuses rather than inventing a Decision.
    """

    def __init__(self, engine: "PlasticEngine") -> None:
        from ..glomerular import ENCODING_GLOM32  # noqa: PLC0415

        v = engine.setup.valence
        self.engine = engine
        self.mode: str = "plastic"
        self.label: str = (
            f"{engine.spec.label} · learning "
            f"{'ON' if engine.core.learning else 'off'}"
        )
        self.input_desc: str = (
            f"its own mushroom-body output neurons: {len(v.approach)} approach "
            f"minus {len(v.avoid)} avoid, mean rate over the 50 ms window"
        )
        self.brain_in_loop: bool = True
        self.feature_version: int = 2
        self.encoding: str = ENCODING_GLOM32
        self.readout_kind: str = "no readout"

    def decide(self, ctx):  # pragma: no cover - not on this mode's path
        raise NotImplementedError(
            "the plastic mode decides per hand in PlasticEngine.step_hand, not "
            "per action"
        )


def _outcome_text(action: str, outcome: dict) -> str:
    """One clause describing what the game did with the hand."""
    if action == DIG:
        return "dug — threw away the junk, no dopamine either way"
    chips = outcome["chips_gained"]
    if outcome["cleared"]:
        # ``chips_gained`` is the score delta and the engine zeroes the score
        # when the blind ends, so the winning play often reports 0. Saying
        # "cleared the blind with 0 chips" would be reporting that artefact as
        # if it were the hand.
        return (f"cleared the blind with {chips:,} chips" if chips
                else "cleared the blind")
    if outcome["lost"]:
        return f"lost the blind — {chips:,} chips was not enough"
    if outcome["reward"]:
        return f"{chips:,} chips, paid its share ({outcome['share']:.0f} needed)"
    return f"{chips:,} chips, short of its share ({outcome['share']:.0f} needed)"
