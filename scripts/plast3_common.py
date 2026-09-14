"""Shared pieces for plasticity v3: evaluation modes, the separated bucket
encoding, omission punishment, and the paired/clustered analysis.

Three things live here because more than one v3 script needs them and because an
external audit of v1/v2 found each of them either wrong or unestablished:

1. **Evaluation mode.** ``plast_run.Runner._evaluate`` reuses the *training*
   decider, so with ``explore_floor = 0.1`` the published "greedy" evaluation was
   really a floored-softmax **sample**. :class:`EvalFlyPolicy` does both, and
   names which: ``greedy`` takes PLAY iff ``p >= 0.5`` -- equivalently iff
   ``play_drive >= 0``, since ``p = e/2 + (1 - e) * sigmoid(d / T)`` crosses 0.5
   exactly at ``d = 0`` for any floor ``e < 1`` -- and ``sampled`` draws
   ``rng.random() < p`` as before.

2. **The separated bucket encoding.** The v2 report blamed a "Hamming distance 2"
   bottleneck: the score-vs-needed bucket is one bit of 32, so the two odours the
   fly must separate are nearly the same odour. :func:`separated_assignment`
   gives each of the four buckets its own disjoint, ORN-count-balanced ensemble
   of glomeruli, paid for by dropping the relay bits that carry nothing at a
   plasticity decision point (see the module constants below).

3. **The analysis.** v2 quoted ``z = 1.75 over 180 games`` for 3 training runs
   over 60 reused evaluation seeds, which treats the same game counted three
   times as three independent games. :func:`cluster_bootstrap` resamples both
   clusters -- evaluation seed and training run -- and works on the *paired*
   per-seed difference against a control evaluated on the same seeds.

Nothing here mutates anything under ``outputs/plast``, ``outputs/plast2`` or
``outputs/mb2``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from flybalatro import plasticity as P
from flybalatro.brain import Brain
from flybalatro.encode import GlomerularMap
from flybalatro.features_v2 import N_HAND_BLOCK
from flybalatro.glomerular import GLOM32_TYPES
from flybalatro.tuning import Tuning
from scripts import plast_common as K

OUT_DIR = K.ROOT / "outputs" / "plast3"

# --------------------------------------------------------------------------- #
# encodings
# --------------------------------------------------------------------------- #
#: Factor B levels.
ENC_CURRENT: str = "current"
ENC_SEPARATED: str = "separated"
ENCODINGS: Tuple[str, ...] = (ENC_CURRENT, ENC_SEPARATED)

#: Relay bits 0-8 -- the 9-way best-hand-type one-hot. Kept in both encodings, on
#: the same nine small, tight glomeruli ``outputs/mb2`` chose so that *which*
#: hand type is on changes the driven-ORN count by at most 2 cells.
HAND_TYPE_BITS: Tuple[int, ...] = tuple(range(0, 9))
#: Relay bits 28-31 -- ``best score vs what is still needed``, the axis that
#: decides the action.
BUCKET_BITS: Tuple[int, ...] = (28, 29, 30, 31)
#: Relay bits 9-27, dropped by the separated encoding. Measured over the 1,091
#: decision points of the 60 standard evaluation games (``plast3_encoding``
#: section of the report): bits 17-25 and 27 are **never** on, bit 26
#: (``sel_none``) is on at **every** decision point, and bits 9-16 are the slot
#: mask of the best subset -- which the harness selects itself whichever way the
#: fly decides, so it cannot change the value of PLAY vs DIG.
DROPPED_BITS: Tuple[int, ...] = tuple(range(9, 28))

#: How many disjoint ensembles the freed glomeruli are split into.
N_BUCKETS: int = len(BUCKET_BITS)


class EnsembleGlomerularMap:
    """Binary feature -> tonic drive on every ORN of a *set* of glomeruli.

    :class:`flybalatro.encode.GlomerularMap` is one glomerulus per feature and
    refuses duplicates, which is exactly right for the v3 encoding and cannot
    express this one. Same interface (``drive``/``indices``/``n_driven``) so
    :class:`plast_common.Setup` does not know the difference; a feature may map
    to the empty set, which is how a dropped relay bit is wired to nothing.
    """

    def __init__(self, graph, groups: Sequence[Sequence[str]],
                 drive_mv: float) -> None:
        t = graph.type.astype(str)
        orn = graph.orn_indices()
        is_orn = np.zeros(graph.n, bool)
        is_orn[orn] = True
        seen: Dict[str, int] = {}
        self.group_names: List[List[str]] = []
        self.groups: List[NDArray[np.int32]] = []
        for f, names in enumerate(groups):
            idx: List[NDArray[np.int32]] = []
            for nm in names:
                nm = str(nm)
                if nm in seen:
                    raise ValueError(
                        f"glomerulus {nm!r} used by feature {seen[nm]} and {f}; "
                        "ensembles must be disjoint")
                seen[nm] = f
                hit = np.flatnonzero((t == nm) & is_orn).astype(np.int32)
                if len(hit) == 0:
                    raise ValueError(f"no olfactory receptor neurons of type {nm!r}")
                idx.append(hit)
            self.group_names.append([str(x) for x in names])
            self.groups.append(np.concatenate(idx).astype(np.int32) if idx
                               else np.zeros(0, np.int32))
        self.n_features = len(self.groups)
        self.drive_mv = float(drive_mv)
        self.n_total = int(graph.n)
        self.group_sizes = [int(len(g)) for g in self.groups]
        self._buf = np.zeros(self.n_total, np.float32)
        nonempty = [g for g in self.groups if len(g)]
        self._all = (np.unique(np.concatenate(nonempty)).astype(np.int32)
                     if nonempty else np.zeros(0, np.int32))

    def indices(self, features) -> NDArray[np.int32]:
        f = np.asarray(features)
        if f.shape != (self.n_features,):
            raise ValueError(f"expected {self.n_features} features, got {f.shape}")
        on = [self.groups[i] for i in np.flatnonzero(f.astype(bool))
              if len(self.groups[i])]
        if not on:
            return np.zeros(0, np.int32)
        return np.concatenate(on).astype(np.int32)

    def drive(self, features, drive_mv: Optional[float] = None) -> NDArray[np.float32]:
        mv = self.drive_mv if drive_mv is None else float(drive_mv)
        self._buf.fill(0.0)
        idx = self.indices(features)
        if len(idx):
            self._buf[idx] = mv
        return self._buf

    def n_driven(self, features) -> int:
        f = np.asarray(features).astype(bool)
        return int(sum(self.group_sizes[i] for i in np.flatnonzero(f)))

    def all_indices(self) -> NDArray[np.int32]:
        return self._all

    def describe(self) -> dict:
        return dict(
            kind="glomerular_ensemble",
            n_features=self.n_features,
            drive_mv=self.drive_mv,
            groups=[list(g) for g in self.group_names],
            group_sizes=list(self.group_sizes),
            n_driven_neurons=int(len(self._all)),
        )


def orn_counts(graph, names: Sequence[str] = GLOM32_TYPES) -> Dict[str, int]:
    """ORNs per glomerulus type, from the graph rather than from the report."""
    t = graph.type.astype(str)
    orn = graph.orn_indices()
    is_orn = np.zeros(graph.n, bool)
    is_orn[orn] = True
    return {str(nm): int(((t == str(nm)) & is_orn).sum()) for nm in names}


def balanced_partition(items: Sequence[Tuple[str, int]], n_parts: int
                       ) -> List[List[str]]:
    """Greedy largest-first partition into ``n_parts`` near-equal total sizes.

    Deterministic: ties broken by the input order, which is
    :data:`flybalatro.glomerular.GLOM32_TYPES`.
    """
    parts: List[List[str]] = [[] for _ in range(n_parts)]
    totals = [0] * n_parts
    for name, size in sorted(items, key=lambda x: (-x[1], x[0])):
        j = int(np.argmin(totals))
        parts[j].append(name)
        totals[j] += int(size)
    return parts


def separated_assignment(graph) -> List[List[str]]:
    """32 relay bits -> glomerulus ensembles, buckets separated.

    * bits 0-8 keep their single mb2 glomerulus (34-36 ORNs each);
    * bits 9-27 drive nothing (see :data:`DROPPED_BITS`);
    * bits 28-31 get the 23 glomeruli that frees, split into four disjoint
      ensembles of near-equal ORN count, each keeping the single glomerulus mb2
      had given that bucket so the new ensemble is a strict superset of the old
      channel.
    """
    counts = orn_counts(graph)
    bucket_singletons = [GLOM32_TYPES[b] for b in BUCKET_BITS]
    freed = [nm for nm in GLOM32_TYPES[9:]]
    parts = balanced_partition([(nm, counts[nm]) for nm in freed], N_BUCKETS)
    # Order the parts so each bucket keeps its own mb2 glomerulus. The greedy
    # pass seeds each part with one of the four largest glomeruli, which are
    # exactly the four mb2 bucket channels, so this is a permutation, not a move.
    order: List[int] = []
    for nm in bucket_singletons:
        hit = [i for i, p in enumerate(parts) if nm in p and i not in order]
        if not hit:
            raise AssertionError(f"{nm} did not seed its own part")
        order.append(hit[0])
    groups: List[List[str]] = [[] for _ in range(N_HAND_BLOCK)]
    for b in HAND_TYPE_BITS:
        groups[b] = [GLOM32_TYPES[b]]
    for b, j in zip(BUCKET_BITS, order):
        groups[b] = list(parts[j])
    return groups


def assignment_for(encoding: str, graph) -> List[List[str]]:
    if encoding == ENC_CURRENT:
        return [[nm] for nm in GLOM32_TYPES]
    if encoding == ENC_SEPARATED:
        return separated_assignment(graph)
    raise ValueError(f"unknown encoding {encoding!r}; expected one of {ENCODINGS}")


def map_for(encoding: str, graph, cfg: dict):
    """The odour map for one encoding, built from the config's own drive."""
    drive_mv = float(cfg["drive"]["tonic_mv"])
    if encoding == ENC_CURRENT:
        gloms = [b["glomerulus"] for b in cfg["encoding"]["assignment"]]
        return GlomerularMap(graph, gloms, drive_mv=drive_mv)
    return EnsembleGlomerularMap(graph, assignment_for(encoding, graph), drive_mv)


def build_setup3(encoding: str, cfg: dict, graph, valence_scheme: str = "nt",
                 window_ms: Optional[float] = None) -> K.Setup:
    """:func:`plast_common.build_setup` with a pluggable odour map.

    Real wiring only: v1 established that the shuffled control deletes the
    KC -> MBON pathway here rather than rewiring it.
    """
    window = float(cfg["window_ms"] if window_ms is None else window_ms)
    gm = map_for(encoding, graph, cfg)
    brain = Brain(graph, seed=int(cfg["brain_seed"]),
                  v_jitter=float(cfg["v_jitter_mv"]),
                  tuning=Tuning.from_dict(cfg["tuning"]))
    brain.warmup()
    valence = P.mbon_valence_table(graph, brain.post, brain.weight,
                                   scheme=valence_scheme)
    plast = P.KcMbonPlasticity(brain, graph, valence)
    return K.Setup(graph=graph, cfg=cfg, gm=gm, brain=brain, valence=valence,
                   plast=plast, kc=graph.kc_indices(), window_ms=window,
                   wiring="real")


def effective_bits(encoding: str, bits: NDArray[np.float32]) -> NDArray[np.float32]:
    """The relay bits this encoding actually drives receptors with."""
    b = np.asarray(bits, np.float32)[:N_HAND_BLOCK].copy()
    if encoding == ENC_SEPARATED:
        b[list(DROPPED_BITS)] = 0.0
    return b


# --------------------------------------------------------------------------- #
# evaluation modes
# --------------------------------------------------------------------------- #
GREEDY: str = "greedy"
SAMPLED: str = "sampled"
EVAL_MODES: Tuple[str, ...] = (GREEDY, SAMPLED)


def greedy_action(p: float) -> str:
    """PLAY iff ``p >= 0.5``. Floor-invariant: ``p >= 0.5`` iff ``drive >= 0``."""
    return K.PLAY if p >= 0.5 else K.DISCARD


class EvalFlyPolicy:
    """The learned policy at evaluation, in one of the two modes.

    The brain is deterministic and stateless across windows (``Brain.reset`` then
    one ``step`` of ``window_ms`` from fixed ``v0``), so with the weights frozen
    the whole decision is a pure function of the 32 relay bits. ``cache=True``
    memoises it per odour, which is exact and is the only reason 400-game
    evaluations are affordable; it must never be used while weights are moving.
    """

    def __init__(self, setup: K.Setup, decider: P.Decider, mode: str,
                 rng: Optional[np.random.Generator] = None,
                 cache: bool = True, encoding: str = ENC_CURRENT) -> None:
        if mode not in EVAL_MODES:
            raise ValueError(f"mode must be one of {EVAL_MODES}, got {mode!r}")
        if mode == SAMPLED and rng is None:
            raise ValueError("sampled evaluation needs an rng")
        if encoding not in ENCODINGS:
            raise ValueError(f"unknown encoding {encoding!r}")
        self.setup = setup
        self.decider = decider
        self.mode = mode
        self.rng = rng
        #: Cache key. Under ``separated`` the dropped relay bits drive no
        #: receptor at all, so two patterns that differ only there are the same
        #: odour and the key must say so -- otherwise the cache is merely
        #: correct-but-slow (28 odours stored as ~400 keys).
        self.encoding = encoding
        self.use_cache = bool(cache)
        self._cache: Dict[int, Tuple[float, float, int, int]] = {}
        self.n_hit = 0
        self.n_miss = 0
        self.last_kc_counts: Optional[NDArray[np.int32]] = None

    def _features(self, ctx: K.HandContext) -> Tuple[float, float, int, int]:
        key = int(K._odor_key(effective_bits(self.encoding, ctx.bits)))
        if self.use_cache:
            hit = self._cache.get(key)
            if hit is not None:
                self.n_hit += 1
                return hit
        self.n_miss += 1
        counts = self.setup.run_window(ctx.bits)
        kcc = counts[self.setup.kc]
        self.last_kc_counts = kcc.copy()
        raw = self.decider.raw_drive(counts)
        out = (raw, self.decider.p_play(counts), int((kcc > 0).sum()),
               int(kcc.sum()))
        if self.use_cache:
            self._cache[key] = out
        return out

    def __call__(self, ctx: K.HandContext) -> Tuple[str, dict]:
        raw, p, kc_active, kc_spikes = self._features(ctx)
        if not ctx.discard_ok:
            action = K.PLAY
        elif self.mode == GREEDY:
            action = greedy_action(p)
        else:
            action = K.PLAY if float(self.rng.random()) < p else K.DISCARD
        return action, dict(
            raw_drive=round(raw, 4),
            play_drive=round(raw + self.decider.bias, 4),
            p_play=round(p, 4),
            eval_mode=self.mode,
            kc_active=kc_active,
            kc_spikes=kc_spikes,
        )

    def cache_stats(self) -> dict:
        n = self.n_hit + self.n_miss
        return dict(distinct_odours=len(self._cache), n_decisions=int(n),
                    hit_rate=round(self.n_hit / n, 4) if n else 0.0)


# --------------------------------------------------------------------------- #
# punishment conditions (factor A)
# --------------------------------------------------------------------------- #
#: Factor A levels.
PUNISH_CURRENT: str = "current"
PUNISH_OMISSION: str = "omission"
PUNISHMENTS: Tuple[str, ...] = (PUNISH_CURRENT, PUNISH_OMISSION)


def dopamine_for(record: dict, punishment: str) -> Optional[str]:
    """Which dopamine pulse one resolved hand earns, ``None`` for silence.

    ``reward`` is untouched: a PLAY that cleared the blind or made its fair share
    ``needed / plays_left`` (:func:`plast_common.resolve_outcome`). ``current``
    punishment is the existing one, the terminal losing play. ``omission``
    punishes **every** PLAY that was not rewarded -- the same game-computable
    inequality with its negative branch wired up, and a strict superset of
    ``current`` because ``punish`` already implies ``not reward``. A DISCARD
    still earns nothing either way.
    """
    if punishment not in PUNISHMENTS:
        raise ValueError(f"punishment must be one of {PUNISHMENTS}")
    if record["action"] != K.PLAY:
        return None
    if record["reward"]:
        return "reward"
    if punishment == PUNISH_CURRENT:
        return "punish" if record["punish"] else None
    return "punish"


# --------------------------------------------------------------------------- #
# paired, clustered analysis
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClusterResult:
    delta: float
    lo: float
    hi: float
    p_two_sided: float
    n_seeds: int
    n_runs: int
    per_run_delta: Tuple[float, ...]

    def to_dict(self) -> dict:
        return dict(delta=round(self.delta, 4), ci_lo=round(self.lo, 4),
                    ci_hi=round(self.hi, 4), p_two_sided=round(self.p_two_sided, 4),
                    n_seeds=self.n_seeds, n_runs=self.n_runs,
                    per_run_delta=[round(x, 4) for x in self.per_run_delta],
                    significant=bool(self.lo > 0.0 or self.hi < 0.0))


def cluster_bootstrap(runs: Sequence[Sequence[float]], control: Sequence[float],
                      n_boot: int = 10_000, seed: int = 20260913,
                      alpha: float = 0.05) -> ClusterResult:
    """Paired difference vs ``control``, resampling seeds **and** training runs.

    ``runs[r][i]`` is the outcome (here: did the game clear) of training run
    ``r`` on evaluation seed ``i``; ``control[i]`` is the same seed under the
    control policy. The statistic is
    ``mean_over_runs mean_over_seeds (runs[r][i] - control[i])``.

    Both clusters are resampled with replacement, which is the correction the v2
    write-up's ``z = 1.75 over 180 games`` needs: the 180 were 60 distinct games
    counted three times, and the seed-to-seed variance of a 5-hand Balatro ante
    dwarfs the run-to-run variance. Resampling three runs is coarse -- three
    clusters cannot resolve much on their own -- so the per-run deltas are
    reported beside the interval rather than hidden inside it.
    """
    A = np.asarray([np.asarray(r, np.float64) for r in runs], np.float64)
    c = np.asarray(control, np.float64)
    if A.ndim != 2 or A.shape[1] != len(c):
        raise ValueError(f"runs {A.shape} do not align with control {c.shape}")
    D = A - c[None, :]
    n_runs, n_seeds = D.shape
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, np.float64)
    for b in range(n_boot):
        ri = rng.integers(0, n_runs, n_runs)
        si = rng.integers(0, n_seeds, n_seeds)
        stats[b] = D[np.ix_(ri, si)].mean()
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    obs = float(D.mean())
    tail = min(float((stats <= 0).mean()), float((stats >= 0).mean()))
    return ClusterResult(delta=obs, lo=float(lo), hi=float(hi),
                         p_two_sided=min(1.0, 2.0 * tail), n_seeds=int(n_seeds),
                         n_runs=int(n_runs),
                         per_run_delta=tuple(float(x) for x in D.mean(axis=1)))


def clear_vector(games: Sequence[K.GameResult]) -> NDArray[np.float64]:
    """Per-seed 0/1 clear outcomes, in seed order."""
    return np.asarray([1.0 if g.cleared else 0.0 for g in games], np.float64)


# --------------------------------------------------------------------------- #
# protocol constants (pre-registered; see outputs/plast3/PREREGISTRATION.md)
# --------------------------------------------------------------------------- #
#: 400 fresh paired evaluation seeds, disjoint from every seed range used so far
#: (eval 100000-100059, train 200000+, calibration 300000+, bc2 0-1359 and
#: 500000-501358).
EVAL3_SEED0: int = 400_000
EVAL3_GAMES: int = 400
TRAIN3_HANDS: int = 1_200
TRAIN3_SEEDS: Tuple[int, ...] = (0, 1, 2)
ETA_REWARD3: float = 0.05
EXPLORE_FLOOR3: float = 0.1
#: One evaluation RNG for every sampled evaluation, so the sampled comparison is
#: paired on the draw as well as on the game.
EVAL3_RNG_SEED: int = 50_000


def eval_seeds(n: int = EVAL3_GAMES, seed0: int = EVAL3_SEED0) -> List[int]:
    return [seed0 + i for i in range(n)]


def evaluate_policy(policy, seeds: Sequence[int], env=None) -> List[K.GameResult]:
    e = env if env is not None else K.make_env()
    return [K.play_game(e, int(s), policy) for s in seeds]


def write_json(path: Path, payload: dict) -> Path:
    return K.write_json(path, payload)


def rel_path(p: Path) -> str:
    """Project-relative when possible; absolute otherwise (smoke runs to /tmp)."""
    try:
        return str(Path(p).relative_to(K.ROOT))
    except ValueError:
        return str(p)
