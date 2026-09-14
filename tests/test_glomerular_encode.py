"""GlomerularMap and the 32-bit relay assignment (mb2 encoding)."""

from __future__ import annotations

import numpy as np
import pytest

from flybalatro.encode import GlomerularMap, orn_types_by_size
from flybalatro.features_v2 import N_HAND_BLOCK


class FakeGraph:
    """Three glomeruli of 3/2/1 ORNs plus one non-ORN neuron of the same type."""

    def __init__(self) -> None:
        self.type = np.array(["ORN_A", "ORN_A", "ORN_A", "ORN_B", "ORN_B",
                              "ORN_C", "ORN_A", ""], dtype=object)
        self.cls = np.array(["olfactory"] * 6 + ["other", "olfactory"], dtype=object)
        self.superclass = np.array(["cb_sensory"] * 6 + ["cb_intrinsic",
                                                         "cb_sensory"], dtype=object)

    @property
    def n(self) -> int:
        return len(self.type)

    def orn_indices(self) -> np.ndarray:
        return np.flatnonzero((self.superclass == "cb_sensory")
                              & (self.cls == "olfactory")).astype(np.int32)


def test_orn_types_by_size_orders_and_excludes_untyped() -> None:
    got = orn_types_by_size(FakeGraph())
    assert got == [("ORN_A", 3), ("ORN_B", 2), ("ORN_C", 1)]


def test_groups_are_whole_types_and_exclude_non_orns() -> None:
    gm = GlomerularMap(FakeGraph(), ["ORN_A", "ORN_B"])
    assert [len(g) for g in gm.groups] == [3, 2]
    assert 6 not in gm.all_indices()          # the non-ORN ORN_A-typed neuron
    assert gm.group_sizes == [3, 2]


def test_drive_hits_exactly_the_active_glomeruli() -> None:
    gm = GlomerularMap(FakeGraph(), ["ORN_A", "ORN_B", "ORN_C"], drive_mv=30.0)
    d = gm.drive(np.array([0, 1, 0]))
    assert np.flatnonzero(d).tolist() == [3, 4]
    assert d[3] == 30.0
    assert gm.n_driven(np.array([1, 0, 1])) == 4
    assert gm.indices(np.array([0, 0, 0])).size == 0
    with pytest.raises(ValueError):
        gm.drive(np.array([1, 0]))


def test_duplicate_or_missing_types_rejected() -> None:
    with pytest.raises(ValueError):
        GlomerularMap(FakeGraph(), ["ORN_A", "ORN_A"])
    with pytest.raises(ValueError):
        GlomerularMap(FakeGraph(), ["ORN_ZZ"])


def test_assignment_is_a_bijection_onto_32_glomeruli() -> None:
    graph = pytest.importorskip("flybalatro.connectome")
    from scripts.mb2_assign import assign

    g = graph.load(5)
    types, sizes, prov = assign(g)
    assert len(types) == N_HAND_BLOCK == 32
    assert len(set(types)) == 32
    ranked = dict(orn_types_by_size(g))
    assert all(ranked[t] == s for t, s in zip(types, sizes))
    # the 32 chosen are the 32 largest
    assert min(sizes) >= max(ranked[t] for t in ranked if t not in set(types))
    # the 9-way label block is the tight one
    assert max(sizes[0:9]) - min(sizes[0:9]) <= 2
