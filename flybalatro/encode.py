"""Binary game features -> tonic current on olfactory receptor neurons.

Each binary feature owns a disjoint group of ORNs. An active feature depolarises
its group, exactly like doomfly's 30 mV retina drive.

Bottleneck worth knowing: MaleCNS has 2,639 ORNs but only 54 glomerular types
(ORN_DA1, ORN_VA1d, ...). Every ORN of a type converges on the same handful of
antennal-lobe projection neurons, so N features are ultimately projected into
~54 x 2 sides analog channels no matter how the groups are drawn.

  grouping='random'  (default) -- each feature gets 10 ORNs scattered across
      glomeruli, so each PN channel sees a random weighted sum of features:
      a random linear projection, which a linear readout can decode.
  grouping='by_type' -- contiguous chunks within a glomerulus; features sharing a
      glomerulus become nearly indistinguishable downstream, but a label defined
      as "count of bits in a group" maps onto one channel cleanly.

``orn_first`` (v2 encoding) -- by default the ORNs and the overflow sensory
neurons are pooled and then permuted *together*, so a low feature index is no
more likely to sit on a real ORN than a high one: with 283 features only 93% of
the 2,830 driven neurons are ORNs at all, spread uniformly. The v2 encoding puts
its hand-type block at the front specifically so it drives the olfactory
pathway, which only means something if "first" is honoured, so ``orn_first=True``
permutes the real ORNs and the overflow separately and lays the ORNs down first.
Default stays ``False`` so v1 feature maps (and the models trained on them) are
bit-identical.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DRIVE_MV = 30.0


class FeatureMap:
    def __init__(self, n_features: int, orn_indices, neurons_per_feature: int = 10,
                 seed: int = 0, n_total: int | None = None, grouping: str = "random",
                 types=None, fallback_indices=None, drive_mv: float = DEFAULT_DRIVE_MV,
                 orn_first: bool = False):
        if grouping not in ("random", "by_type"):
            raise ValueError("grouping must be 'random' or 'by_type'")
        if orn_first and grouping != "random":
            raise ValueError("orn_first only applies to grouping='random'")
        orn = np.asarray(orn_indices, np.int32)
        need = n_features * neurons_per_feature
        self.extended_with_sensory = False
        self.orn_first = bool(orn_first)
        self.n_true_orn = int(len(orn))
        if need > len(orn):
            if fallback_indices is None:
                raise ValueError(
                    f"{n_features} features x {neurons_per_feature} needs {need} ORNs, "
                    f"only {len(orn)} available; pass fallback_indices to extend."
                )
            extra = np.setdiff1d(np.asarray(fallback_indices, np.int32), orn)
            rng0 = np.random.default_rng(seed)
            extra = rng0.permutation(extra)[: need - len(orn)]
            if len(orn) + len(extra) < need:
                raise ValueError("Not enough sensory neurons even after extension.")
            if orn_first:
                # Shuffle within each tier, then concatenate: the low feature
                # indices get real ORNs, the overflow lands at the tail.
                rng1 = np.random.default_rng(seed)
                orn = np.concatenate(
                    [rng1.permutation(orn), extra]
                ).astype(np.int32)
            else:
                orn = np.concatenate([orn, extra]).astype(np.int32)
            self.extended_with_sensory = True

        rng = np.random.default_rng(seed)
        if grouping == "random":
            pool = orn[:need] if orn_first else rng.permutation(orn)[:need]
        else:
            if types is None:
                raise ValueError("grouping='by_type' needs the per-neuron type array")
            t = np.asarray(types, dtype=object)[orn].astype(str)
            pool = orn[np.argsort(t, kind="stable")][:need]
        self.groups = pool.reshape(n_features, neurons_per_feature).astype(np.int32)

        self.n_features = int(n_features)
        self.neurons_per_feature = int(neurons_per_feature)
        self.seed = int(seed)
        self.grouping = grouping
        self.drive_mv = float(drive_mv)
        self.n_orn_available = int(len(orn))
        self.n_total = int(n_total) if n_total is not None else int(pool.max()) + 1
        self._buf = np.zeros(self.n_total, np.float32)

    @classmethod
    def for_graph(cls, graph, n_features: int, neurons_per_feature: int = 10,
                  seed: int = 0, grouping: str = "random",
                  drive_mv: float = DEFAULT_DRIVE_MV,
                  orn_first: bool = False) -> "FeatureMap":
        return cls(
            n_features, graph.orn_indices(), neurons_per_feature, seed,
            n_total=graph.n, grouping=grouping, types=graph.type,
            fallback_indices=graph.sensory_indices(), drive_mv=drive_mv,
            orn_first=orn_first,
        )

    def drive(self, features, drive_mv: float | None = None) -> np.ndarray:
        """features: bool/0-1 array of length n_features -> float32[n_total] tonic current."""
        f = np.asarray(features)
        if f.shape != (self.n_features,):
            raise ValueError(f"expected {self.n_features} features, got {f.shape}")
        mv = self.drive_mv if drive_mv is None else float(drive_mv)
        self._buf.fill(0.0)
        on = np.flatnonzero(f.astype(bool))
        if len(on):
            self._buf[self.groups[on].ravel()] = mv
        return self._buf

    def all_indices(self) -> np.ndarray:
        return np.unique(self.groups.ravel())

    def describe(self) -> dict:
        return dict(
            n_features=self.n_features,
            neurons_per_feature=self.neurons_per_feature,
            grouping=self.grouping,
            seed=self.seed,
            drive_mv=self.drive_mv,
            n_orn_available=self.n_orn_available,
            n_driven_neurons=int(self.groups.size),
            extended_with_sensory=self.extended_with_sensory,
            orn_first=self.orn_first,
            n_true_orn=self.n_true_orn,
        )


#: Neurons driven per feature, everywhere. Changing it invalidates every model.
NEURONS_PER_FEATURE = 10

#: ``FeatureMap`` seed used by every stage of the BC pipeline.
FEATURE_MAP_SEED = 0


def feature_map_for(graph, feature_version: int = 1, seed: int = FEATURE_MAP_SEED,
                    neurons_per_feature: int = NEURONS_PER_FEATURE) -> "FeatureMap":
    """The one constructor every stage must use, keyed by encoding version.

    Collection, ``bc_brain_features``, ``bc_eval``, the consistency check, the
    viewer and the real-game player all have to build the *same* map or a
    trained readout is silently fed the wrong channels. v1 keeps the original
    pooled-and-permuted behaviour; v2 is ``orn_first=True`` so its hand-type
    block drives real olfactory receptor neurons.
    """
    from .features import N_FEATURES
    from .features_v2 import N_FEATURES_V2

    version = int(feature_version)
    if version == 1:
        return FeatureMap.for_graph(graph, N_FEATURES, neurons_per_feature, seed=seed)
    if version == 2:
        return FeatureMap.for_graph(
            graph, N_FEATURES_V2, neurons_per_feature, seed=seed, orn_first=True
        )
    raise ValueError(f"unknown feature version {feature_version!r}; expected 1 or 2")


# --------------------------------------------------------------------------- #
# Glomerular encoding (v3 / mb2): one binary feature -> one whole ORN type.
# --------------------------------------------------------------------------- #
#: Every olfactory receptor neuron type in MaleCNS v1.0 is named ``ORN_<glom>``.
ORN_TYPE_PREFIX = "ORN_"


def orn_types_by_size(graph) -> list[tuple[str, int]]:
    """Every ``ORN_*`` type with its cell count, largest first, name as tie-break.

    The 4 olfactory neurons whose ``type`` string is empty are excluded: they are
    not assignable to a glomerulus.
    """
    orn = graph.orn_indices()
    t = graph.type.astype(str)[orn]
    keep = np.char.startswith(t, ORN_TYPE_PREFIX)
    names, counts = np.unique(t[keep], return_counts=True)
    order = np.lexsort((names, -counts))
    return [(str(names[i]), int(counts[i])) for i in order]


class GlomerularMap:
    """Binary feature -> tonic drive on *every* ORN of one glomerular type.

    The v1/v2 :class:`FeatureMap` gives each feature 10 ORNs drawn at random
    across glomeruli, so ~36 active features put ~360 ORNs into all 54 glomeruli
    and the antennal-lobe output is nearly constant across inputs (measured ALPN
    total-count CV 0.031, ``outputs/mb/REPORT.md`` section 9). This map instead
    makes one feature one *channel*: feature ``f`` drives all ~34-204 ORNs of
    type ``type_names[f]`` and nothing else, so an input with k active features
    activates exactly k glomeruli and leaves the other ~21 silent.

    Groups are ragged (glomeruli differ in size), which is why this is a separate
    class rather than another ``grouping=`` value on ``FeatureMap``.
    """

    def __init__(self, graph, type_names, drive_mv: float = DEFAULT_DRIVE_MV):
        names = [str(x) for x in type_names]
        if len(set(names)) != len(names):
            raise ValueError("type_names must be distinct (one glomerulus per feature)")
        t = graph.type.astype(str)
        orn = graph.orn_indices()
        is_orn = np.zeros(graph.n, bool)
        is_orn[orn] = True
        self.groups: list[np.ndarray] = []
        for nm in names:
            idx = np.flatnonzero((t == nm) & is_orn).astype(np.int32)
            if len(idx) == 0:
                raise ValueError(f"no olfactory receptor neurons of type {nm!r}")
            self.groups.append(idx)
        self.type_names = names
        self.n_features = len(names)
        self.drive_mv = float(drive_mv)
        self.n_total = int(graph.n)
        self.group_sizes = [int(len(g)) for g in self.groups]
        self._buf = np.zeros(self.n_total, np.float32)
        self._all = np.unique(np.concatenate(self.groups)).astype(np.int32)

    # ---- drive ---------------------------------------------------------------
    def indices(self, features) -> np.ndarray:
        """Indices of every ORN driven by this feature vector."""
        f = np.asarray(features)
        if f.shape != (self.n_features,):
            raise ValueError(f"expected {self.n_features} features, got {f.shape}")
        on = np.flatnonzero(f.astype(bool))
        if len(on) == 0:
            return np.zeros(0, np.int32)
        return np.concatenate([self.groups[i] for i in on]).astype(np.int32)

    def drive(self, features, drive_mv: float | None = None) -> np.ndarray:
        """float32[n_total] tonic current: ``drive_mv`` on every driven ORN."""
        mv = self.drive_mv if drive_mv is None else float(drive_mv)
        self._buf.fill(0.0)
        idx = self.indices(features)
        if len(idx):
            self._buf[idx] = mv
        return self._buf

    def n_driven(self, features) -> int:
        f = np.asarray(features).astype(bool)
        return int(sum(self.group_sizes[i] for i in np.flatnonzero(f)))

    def all_indices(self) -> np.ndarray:
        return self._all

    def describe(self) -> dict:
        return dict(
            kind="glomerular",
            n_features=self.n_features,
            drive_mv=self.drive_mv,
            type_names=list(self.type_names),
            group_sizes=list(self.group_sizes),
            n_driven_neurons=int(len(self._all)),
            neurons_min=min(self.group_sizes),
            neurons_max=max(self.group_sizes),
        )
