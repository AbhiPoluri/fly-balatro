"""Dopamine-gated depression at Kenyon-cell -> MBON synapses.

This is the only part of the project where anything in the fly changes. The
connectome, the LIF constants, the calibration in :mod:`flybalatro.tuning` and
the glomerular encoding are all frozen exactly as ``outputs/mb2/tuned_config.json``
left them; the *only* mutable state is the weight of the 33,496 KC -> MBON edges.

The circuit being modelled
--------------------------

The canonical mushroom-body learning rule (Aso et al. 2014 eLife 04577/04580;
Hige et al. 2015; Owald et al. 2015; Cohn et al. 2015): a Kenyon cell active at
the same time as dopamine release in its compartment has its output synapse onto
that compartment's MBON *depressed*. Reward (PAM cluster) and punishment (PPL1
cluster) innervate disjoint compartments, so the two reinforcers depress
different halves of the MBON population and the animal's approach/avoid balance
shifts.

Three things have to be pinned down before that rule can be written down here,
and all three come from data in the repository rather than from a guess:

**Valence.** Aso et al. 2014 (eLife 04580, "Mushroom body output neurons encode
valence and guide memory-based action selection in Drosophila") reports that
*every* MBON whose activation drove avoidance was glutamatergic and every MBON
whose activation drove attraction was GABAergic or cholinergic. MaleCNS ships a
per-neuron neurotransmitter prediction (with ground truth for the classic types),
so that rule assigns a valence to all 97 MBONs rather than to the ~9 that have
been tested optogenetically. See :func:`mbon_valence_table`.

**Compartment.** MaleCNS ``instance`` strings carry the Aso compartment name in
parentheses -- ``MBON11(y1pedc>a/B)_L``, ``PAM08(y4)_R``, ``PPL101(y1ped)_L`` --
so the compartment names are annotation, not inference. The *gating* rule is
stronger than name matching and does not need it: a KC -> MBON synapse is in the
PAM (reward) compartment set if PAM neurons supply at least half of that MBON's
total PAM+PPL1 input weight in this graph, and in the PPL1 (punishment) set
otherwise. Measured on the real graph that lands exactly where the anatomy says
it should: MBON01-07 (all glutamatergic) are 90-100% PAM, MBON11-20 are 100%
PPL1.

**Eligibility.** The presynaptic side of the coincidence. A KC is eligible in
proportion to how hard it fired in the decision window, saturating at
``KC_REF`` spikes (2 in a 50 ms window; the observed range is 0-4).

Everything mutable lives in :class:`KcMbonPlasticity`, which owns a private copy
of the original KC -> MBON weights so a run can be reset and so the
"weights never leave ``[floor * w0, w0]``" invariant is checkable at any time.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from . import connectome as C

__all__ = [
    "APPROACH",
    "AVOID",
    "KC_REF",
    "WEIGHT_FLOOR",
    "WEIGHTS_FORMAT",
    "MbonValence",
    "KcMbonPlasticity",
    "Decider",
    "mbon_valence_table",
    "compartment_dominance",
    "read_instances",
    "read_neurotransmitters",
    "read_weights",
    "BRIEF_VALENCE",
]

#: Valence labels.
APPROACH = "approach"
AVOID = "avoid"

#: Spikes in the decision window at which a Kenyon cell is fully eligible.
KC_REF: float = 2.0

#: A depressed synapse never falls below this multiple of its original weight.
WEIGHT_FLOOR: float = 0.05

#: Version tag written into every ``.npz`` produced by
#: :meth:`KcMbonPlasticity.save_weights`, so a loader can refuse a file it does
#: not understand instead of silently mis-indexing 33,496 synapses.
WEIGHTS_FORMAT: str = "kc_mbon_weights/1"

#: Aso et al. 2014 (eLife 04580) neurotransmitter rule for MBON activation valence.
NT_VALENCE: Dict[str, str] = {
    "glutamate": AVOID,
    "gaba": APPROACH,
    "acetylcholine": APPROACH,
}

#: The per-compartment lists named in the task brief, for the comparison table
#: and for the ``--valence brief`` robustness run. These are *activation*
#: valences as the brief states them; they disagree with the neurotransmitter
#: rule on 6 of 9 entries (see ``outputs/plast/REPORT.md``).
BRIEF_VALENCE: Dict[str, str] = {
    "MBON11": APPROACH,   # y1pedc>a/B
    "MBON05": APPROACH,   # y4>y1y2
    "MBON03": APPROACH,   # B'2mp
    "MBON12": APPROACH,   # y2a'1
    "MBON18": AVOID,      # a2sc
    "MBON14": AVOID,      # a3
    "MBON13": AVOID,      # a'2
    "MBON01": AVOID,      # y5B'2a
    "MBON02": AVOID,      # B2B'2a
}

_INSTANCE_RE = re.compile(r"\((.*)\)")


# --------------------------------------------------------------------------- #
# annotation helpers
# --------------------------------------------------------------------------- #
def read_neurotransmitters(graph) -> NDArray[np.str_]:
    """Per-neuron ``consensus_nt`` string for every neuron in ``graph`` order.

    The same file :mod:`flybalatro.connectome` already reads for the sign; this
    keeps the string instead of collapsing it to +1/-1, because GABA and
    glutamate are both inhibitory here but have *opposite* behavioural valence.
    """
    import pandas as pd

    nt = pd.read_feather(
        C.DATA_DIR / C.NEUROTRANSMITTERS, columns=["body", "consensus_nt"]
    )
    lookup = dict(zip(nt["body"].to_numpy(), nt["consensus_nt"].astype(str).to_numpy()))
    return np.array([lookup.get(int(b), "") for b in graph.body_id], dtype=object)


def read_instances(graph) -> NDArray[np.str_]:
    """Per-neuron ``instance`` string, which carries the compartment name."""
    import pandas as pd

    ann = pd.read_feather(C.DATA_DIR / C.ANNOTATIONS, columns=["bodyId", "instance"])
    lookup = dict(zip(ann["bodyId"].to_numpy(), ann["instance"].astype(str).to_numpy()))
    return np.array([lookup.get(int(b), "") for b in graph.body_id], dtype=object)


def compartment_from_instance(instance: str) -> str:
    """``MBON11(y1pedc>a/B)_L`` -> ``y1pedc>a/B``; empty when unparenthesised."""
    m = _INSTANCE_RE.search(str(instance))
    return m.group(1) if m else ""


def compartment_dominance(
    graph, post: NDArray[np.int32], weight: NDArray[np.float32],
    targets: NDArray[np.int32],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Total |weight| each target receives from PAM and from PPL1 neurons.

    ``post`` / ``weight`` are the *brain's* edge arrays, so the same rule applies
    unchanged to a shuffled brain (where the permutation scrambles which MBON
    each dopaminergic neuron innervates -- exactly what the control is for).
    """
    n = graph.n
    types = graph.type.astype(str)
    pre_of = np.repeat(np.arange(n, dtype=np.int64), np.diff(graph.ptr))
    is_pam = np.char.startswith(types, "PAM")
    is_ppl1 = np.char.startswith(types, "PPL1")
    pam_in = np.zeros(n, np.float64)
    ppl1_in = np.zeros(n, np.float64)
    m = is_pam[pre_of]
    np.add.at(pam_in, post[m], np.abs(weight[m], dtype=np.float64))
    m = is_ppl1[pre_of]
    np.add.at(ppl1_in, post[m], np.abs(weight[m], dtype=np.float64))
    return pam_in[targets], ppl1_in[targets]


# --------------------------------------------------------------------------- #
# valence assignment
# --------------------------------------------------------------------------- #
@dataclass
class MbonValence:
    """Valence and dopaminergic compartment for every MBON, with provenance.

    ``valence`` is per-MBON-neuron (length = number of MBONs); ``dan`` is
    ``"PAM"`` / ``"PPL1"`` / ``""`` for the dominant dopaminergic input.
    MBONs with neither PAM nor PPL1 input, or with no assignable
    neurotransmitter, are dropped from every pool and reported as unassigned.
    """

    mbon: NDArray[np.int32]
    types: NDArray[np.str_]
    compartments: NDArray[np.str_]
    nt: NDArray[np.str_]
    valence: NDArray[np.str_]
    dan: NDArray[np.str_]
    pam_in: NDArray[np.float64]
    ppl1_in: NDArray[np.float64]
    used: NDArray[np.bool_]
    scheme: str = "nt"

    # -- pools -------------------------------------------------------------
    @property
    def approach(self) -> NDArray[np.int32]:
        return self.mbon[self.used & (self.valence == APPROACH)]

    @property
    def avoid(self) -> NDArray[np.int32]:
        return self.mbon[self.used & (self.valence == AVOID)]

    def pool(self, valence: str, dan: str) -> NDArray[np.int32]:
        """MBONs of one valence whose dominant dopaminergic input is ``dan``."""
        return self.mbon[self.used & (self.valence == valence) & (self.dan == dan)]

    # -- reporting ---------------------------------------------------------
    def table(self) -> List[dict]:
        """One row per MBON *type* (both hemispheres pooled)."""
        rows: List[dict] = []
        for ty in sorted(set(self.types.tolist())):
            m = self.types == ty
            rows.append(dict(
                type=str(ty),
                n=int(m.sum()),
                compartment=str(self.compartments[m][0]),
                nt=str(self.nt[m][0]),
                pam_in=round(float(self.pam_in[m].sum()), 1),
                ppl1_in=round(float(self.ppl1_in[m].sum()), 1),
                dan=str(self.dan[m][0]),
                valence=str(self.valence[m][0]) or "unassigned",
                used=bool(self.used[m][0]),
                brief_valence=BRIEF_VALENCE.get(str(ty), ""),
            ))
        return rows

    def summary(self) -> dict:
        rows = self.table()
        return dict(
            scheme=self.scheme,
            n_mbon=int(len(self.mbon)),
            n_types=len(rows),
            approach_neurons=int(len(self.approach)),
            avoid_neurons=int(len(self.avoid)),
            unassigned_neurons=int((~self.used).sum()),
            unassigned_types=[r["type"] for r in rows if not r["used"]],
            approach_types=[r["type"] for r in rows
                            if r["used"] and r["valence"] == APPROACH],
            avoid_types=[r["type"] for r in rows
                         if r["used"] and r["valence"] == AVOID],
            pam_avoid_types=[r["type"] for r in rows
                             if r["used"] and r["valence"] == AVOID
                             and r["dan"] == "PAM"],
            ppl1_approach_types=[r["type"] for r in rows
                                 if r["used"] and r["valence"] == APPROACH
                                 and r["dan"] == "PPL1"],
            brief_agreement={
                r["type"]: dict(nt_rule=r["valence"], brief=r["brief_valence"],
                                agree=r["valence"] == r["brief_valence"])
                for r in rows if r["brief_valence"]
            },
        )


def mbon_valence_table(graph, post, weight, scheme: str = "nt") -> MbonValence:
    """Build the valence / compartment assignment.

    ``scheme="nt"``: Aso et al. 2014's neurotransmitter rule over all 97 MBONs.
    ``scheme="brief"``: only the 9 compartments the task brief names, everything
    else unassigned -- the robustness control for the valence choice.
    """
    mbon = graph.mbon_indices()
    types = graph.type.astype(str)[mbon]
    inst = read_instances(graph)[mbon]
    comp = np.array([compartment_from_instance(s) for s in inst], dtype=object)
    nt = read_neurotransmitters(graph)[mbon]

    if scheme == "nt":
        valence = np.array([NT_VALENCE.get(str(x), "") for x in nt], dtype=object)
    elif scheme == "brief":
        valence = np.array([BRIEF_VALENCE.get(str(t), "") for t in types], dtype=object)
    else:
        raise ValueError(f"unknown valence scheme {scheme!r}")

    pam_in, ppl1_in = compartment_dominance(graph, post, weight, mbon)
    total = pam_in + ppl1_in
    dan = np.where(total <= 0.0, "",
                   np.where(pam_in >= 0.5 * total, "PAM", "PPL1")).astype(object)
    used = (valence != "") & (dan != "")
    return MbonValence(
        mbon=mbon, types=types, compartments=comp, nt=nt, valence=valence,
        dan=dan, pam_in=pam_in, ppl1_in=ppl1_in, used=used, scheme=scheme,
    )


# --------------------------------------------------------------------------- #
# the plastic synapses
# --------------------------------------------------------------------------- #
@dataclass
class _TargetSet:
    """Edge slice for one (dopamine population, MBON valence) pair."""

    edge: NDArray[np.int64]       # index into brain.weight
    kc_row: NDArray[np.int32]     # index into the KC index array
    w0: NDArray[np.float32]       # original weight
    floor: NDArray[np.float32]    # WEIGHT_FLOOR * w0
    mbons: NDArray[np.int32]

    @property
    def n(self) -> int:
        return int(len(self.edge))


class KcMbonPlasticity:
    """Coincidence-gated depression of KC -> MBON weights on a live ``Brain``.

    Owns nothing of the brain but its ``weight`` array, which it mutates in
    place. ``reset_weights`` puts every KC -> MBON edge back to its construction
    value, so one ``Brain`` can serve many runs.
    """

    def __init__(self, brain, graph, valence: MbonValence,
                 kc_ref: float = KC_REF, floor: float = WEIGHT_FLOOR) -> None:
        self.brain = brain
        self.graph = graph
        self.valence = valence
        self.kc_ref = float(kc_ref)
        self.floor_frac = float(floor)
        self.kc = graph.kc_indices()
        self.n_kc = int(len(self.kc))

        post = brain.post
        ptr = graph.ptr
        is_mbon = np.zeros(graph.n, bool)
        is_mbon[valence.mbon] = True

        # Every KC -> MBON edge, with the presynaptic KC's row in self.kc.
        edges: List[np.ndarray] = []
        rows: List[np.ndarray] = []
        for row, i in enumerate(self.kc):
            a, b = int(ptr[i]), int(ptr[i + 1])
            if b <= a:
                continue
            sl = np.arange(a, b, dtype=np.int64)
            m = is_mbon[post[a:b]]
            if not m.any():
                continue
            e = sl[m]
            edges.append(e)
            rows.append(np.full(len(e), row, np.int32))
        self.all_edge = (np.concatenate(edges) if edges
                         else np.zeros(0, np.int64))
        self.all_kc_row = (np.concatenate(rows) if rows
                           else np.zeros(0, np.int32))
        self.all_w0 = brain.weight[self.all_edge].copy()
        if len(self.all_w0) and float(self.all_w0.min()) <= 0.0:
            raise ValueError(
                "KC -> MBON weights must be positive for depression-only "
                f"plasticity; min is {float(self.all_w0.min())}"
            )
        self.all_floor = (self.all_w0 * self.floor_frac).astype(np.float32)
        self.all_post = post[self.all_edge]

        # Split into the two dopamine-gated target sets.
        self.targets: Dict[str, _TargetSet] = {}
        for kind, dan, val in (("reward", "PAM", AVOID),
                               ("punish", "PPL1", APPROACH)):
            mbons = valence.pool(val, dan)
            keep = np.isin(self.all_post, mbons)
            self.targets[kind] = _TargetSet(
                edge=self.all_edge[keep], kc_row=self.all_kc_row[keep],
                w0=self.all_w0[keep], floor=self.all_floor[keep], mbons=mbons,
            )
        self.n_deliveries: Dict[str, int] = {"reward": 0, "punish": 0}

    # -- state -------------------------------------------------------------
    def reset_weights(self) -> None:
        self.brain.weight[self.all_edge] = self.all_w0
        self.n_deliveries = {"reward": 0, "punish": 0}

    def eligibility(self, kc_counts: NDArray[np.int_]) -> NDArray[np.float32]:
        """Presynaptic trace: spikes in the decision window, saturating at ``kc_ref``."""
        e = np.asarray(kc_counts, np.float32) / np.float32(self.kc_ref)
        return np.minimum(e, np.float32(1.0))

    # -- the rule ----------------------------------------------------------
    def deliver(self, kind: str, kc_counts: NDArray[np.int_], eta: float) -> dict:
        """One dopamine pulse. Returns ``{n_edges, n_changed, mean_abs_dw, ...}``.

        ``kind`` is ``"reward"`` (PAM compartments, depress KC -> avoidance MBONs)
        or ``"punish"`` (PPL1 compartments, depress KC -> approach MBONs).
        """
        if kind not in self.targets:
            raise ValueError(f"kind must be 'reward' or 'punish', got {kind!r}")
        ts = self.targets[kind]
        if eta <= 0.0 or ts.n == 0:
            return dict(kind=kind, n_edges=ts.n, n_changed=0, mean_abs_dw=0.0,
                        sum_abs_dw=0.0, elig_kc=0)
        elig = self.eligibility(kc_counts)
        f = elig[ts.kc_row]
        active = f > 0.0
        if not active.any():
            return dict(kind=kind, n_edges=ts.n, n_changed=0, mean_abs_dw=0.0,
                        sum_abs_dw=0.0, elig_kc=0)
        e = ts.edge[active]
        before = self.brain.weight[e].copy()
        after = before * (np.float32(1.0) - np.float32(eta) * f[active])
        after = np.maximum(after, ts.floor[active])
        self.brain.weight[e] = after
        dw = np.abs(after - before, dtype=np.float64)
        self.n_deliveries[kind] += 1
        return dict(
            kind=kind,
            n_edges=int(ts.n),
            n_changed=int((dw > 0).sum()),
            mean_abs_dw=float(dw.mean()),
            sum_abs_dw=float(dw.sum()),
            elig_kc=int((elig > 0).sum()),
        )

    def deliver_eligibility(self, kind: str, elig: NDArray[np.float32],
                            eta: float) -> dict:
        """One dopamine pulse against a **precomputed** per-Kenyon-cell eligibility.

        :meth:`deliver` derives its presynaptic factor from the spike counts of
        the decision window that just happened, which is the right thing when
        dopamine arrives with the outcome of that same decision. A *trace* rule
        needs the other case: dopamine arrives later, and the presynaptic factor
        is what is left of several earlier windows (Cassenaer & Laurent 2012;
        Galili et al. 2011 for the behavioural gap in *Drosophila*). This method
        applies the identical depression step to an eligibility vector the caller
        maintains, so the two rules cannot drift apart.

        ``elig`` must be per-Kenyon-cell in :data:`self.kc` order and inside
        ``[0, 1]``, which is the range :meth:`eligibility` produces; the clip is
        what keeps ``w (1 - eta f)`` positive and the weight floor meaningful.
        ``deliver_eligibility(kind, self.eligibility(counts), eta)`` is exactly
        ``deliver(kind, counts, eta)``.
        """
        if kind not in self.targets:
            raise ValueError(f"kind must be 'reward' or 'punish', got {kind!r}")
        e = np.asarray(elig, np.float32)
        if e.shape != (self.n_kc,):
            raise ValueError(
                f"eligibility must have one entry per Kenyon cell "
                f"({self.n_kc}), got {e.shape}")
        if float(e.min(initial=0.0)) < 0.0 or float(e.max(initial=0.0)) > 1.0:
            raise ValueError("eligibility must lie in [0, 1]")
        ts = self.targets[kind]
        if eta <= 0.0 or ts.n == 0:
            return dict(kind=kind, n_edges=ts.n, n_changed=0, mean_abs_dw=0.0,
                        sum_abs_dw=0.0, elig_kc=0)
        f = e[ts.kc_row]
        active = f > 0.0
        if not active.any():
            return dict(kind=kind, n_edges=ts.n, n_changed=0, mean_abs_dw=0.0,
                        sum_abs_dw=0.0, elig_kc=0)
        edge = ts.edge[active]
        before = self.brain.weight[edge].copy()
        after = before * (np.float32(1.0) - np.float32(eta) * f[active])
        after = np.maximum(after, ts.floor[active])
        self.brain.weight[edge] = after
        dw = np.abs(after - before, dtype=np.float64)
        self.n_deliveries[kind] += 1
        return dict(
            kind=kind,
            n_edges=int(ts.n),
            n_changed=int((dw > 0).sum()),
            mean_abs_dw=float(dw.mean()),
            sum_abs_dw=float(dw.sum()),
            elig_kc=int((e > 0).sum()),
        )

    # -- persistence -------------------------------------------------------
    def save_weights(self, path, meta: Optional[dict] = None) -> "Path":
        """Write the learned KC -> MBON weights to ``path`` as ``.npz``.

        Three arrays are enough to restore a trained fly onto a freshly built
        brain, and all three are needed:

        ``edge``
            index into ``brain.weight`` of every KC -> MBON edge, in the order
            :class:`KcMbonPlasticity` enumerates them. This is the alignment
            key -- a loader that does not check it can put 33,496 numbers on the
            wrong synapses and nothing will look broken.
        ``weight``
            the current (learned) weight of each of those edges.
        ``w0``
            the weight the same edge had at construction, so "changed / at the
            floor / mean ratio" stay computable from the file alone and a run
            can be inspected without rebuilding the brain.

        ``meta`` is any JSON-serialisable dict -- the calibrated ``bias`` and
        ``temperature`` of the :class:`Decider` that produced these weights
        belong here, because the weights are meaningless to a decision rule
        calibrated differently.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            format=np.array(WEIGHTS_FORMAT),
            edge=self.all_edge.astype(np.int64),
            weight=self.brain.weight[self.all_edge].astype(np.float32),
            w0=self.all_w0.astype(np.float32),
            post=self.all_post.astype(np.int32),
            kc_row=self.all_kc_row.astype(np.int32),
            kc=self.kc.astype(np.int32),
            kc_ref=np.float32(self.kc_ref),
            weight_floor=np.float32(self.floor_frac),
            n_deliveries_reward=np.int64(self.n_deliveries["reward"]),
            n_deliveries_punish=np.int64(self.n_deliveries["punish"]),
            meta=np.array(json.dumps(meta or {}, sort_keys=True, default=str)),
        )
        # ``np.savez_compressed`` appends ``.npz`` to any name that lacks it, so
        # the temporary has to end in ``.npz`` or the rename target never exists.
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(p)
        return p

    def load_weights(self, path) -> dict:
        """Restore weights written by :meth:`save_weights` onto this brain.

        Refuses a file whose ``edge`` array is not exactly this object's, which
        is the only way the values could land on the wrong synapses: the edge
        order depends on the graph, the edge threshold and the MBON set, so a
        mismatch means the file was made against a different fly.
        """
        d = read_weights(path)
        edge = d["edge"]
        if edge.shape != self.all_edge.shape or not np.array_equal(edge, self.all_edge):
            raise ValueError(
                f"{path}: KC -> MBON edge indices do not match this brain "
                f"({len(edge)} saved vs {len(self.all_edge)} here); the file was "
                "made against a different graph, edge threshold or MBON set"
            )
        w0 = d["w0"]
        if not np.allclose(w0, self.all_w0, rtol=0, atol=1e-6):
            raise ValueError(
                f"{path}: original weights differ from this brain's, so the "
                "learned weights belong to a differently tuned fly"
            )
        self.brain.weight[self.all_edge] = d["weight"].astype(np.float32)
        return d

    # -- diagnostics -------------------------------------------------------
    def weight_stats(self) -> dict:
        w = self.brain.weight[self.all_edge]
        ratio = w / self.all_w0
        return dict(
            n_edges=int(len(w)),
            min_ratio=float(ratio.min()) if len(w) else 1.0,
            max_ratio=float(ratio.max()) if len(w) else 1.0,
            mean_ratio=float(ratio.mean()) if len(w) else 1.0,
            frac_at_floor=float((ratio <= self.floor_frac + 1e-6).mean()) if len(w) else 0.0,
            frac_unchanged=float((ratio >= 1.0 - 1e-6).mean()) if len(w) else 0.0,
            any_negative=bool((w < 0).any()),
            any_above_original=bool((ratio > 1.0 + 1e-6).any()),
        )

    def describe(self) -> dict:
        return dict(
            kc=self.n_kc,
            kc_mbon_edges=int(len(self.all_edge)),
            kc_ref=self.kc_ref,
            weight_floor=self.floor_frac,
            reward_edges=self.targets["reward"].n,
            reward_mbons=int(len(self.targets["reward"].mbons)),
            punish_edges=self.targets["punish"].n,
            punish_mbons=int(len(self.targets["punish"].mbons)),
        )


# --------------------------------------------------------------------------- #
# saved weights
# --------------------------------------------------------------------------- #
def read_weights(path) -> dict:
    """Read a :meth:`KcMbonPlasticity.save_weights` file without a brain.

    Returns the arrays plus ``meta`` decoded back into a dict and the derived
    statistics (``mean_ratio``, ``frac_unchanged``, ``frac_at_floor``) so a
    weight file can be inspected, diffed or shown in a UI on its own.
    """
    p = Path(path)
    z = np.load(p, allow_pickle=False)
    fmt = str(z["format"]) if "format" in z.files else ""
    if fmt != WEIGHTS_FORMAT:
        raise ValueError(
            f"{p}: format is {fmt!r}, expected {WEIGHTS_FORMAT!r}"
        )
    weight = z["weight"].astype(np.float32)
    w0 = z["w0"].astype(np.float32)
    floor = float(z["weight_floor"]) if "weight_floor" in z.files else WEIGHT_FLOOR
    ratio = weight / w0
    try:
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    except ValueError:
        meta = {}
    return dict(
        path=str(p),
        format=fmt,
        edge=z["edge"].astype(np.int64),
        weight=weight,
        w0=w0,
        post=z["post"].astype(np.int32) if "post" in z.files else None,
        kc_row=z["kc_row"].astype(np.int32) if "kc_row" in z.files else None,
        kc=z["kc"].astype(np.int32) if "kc" in z.files else None,
        kc_ref=float(z["kc_ref"]) if "kc_ref" in z.files else KC_REF,
        weight_floor=floor,
        n_deliveries=dict(
            reward=int(z["n_deliveries_reward"]) if "n_deliveries_reward" in z.files else 0,
            punish=int(z["n_deliveries_punish"]) if "n_deliveries_punish" in z.files else 0,
        ),
        meta=meta,
        n_edges=int(len(weight)),
        mean_ratio=float(ratio.mean()) if len(weight) else 1.0,
        min_ratio=float(ratio.min()) if len(weight) else 1.0,
        frac_unchanged=float((ratio >= 1.0 - 1e-6).mean()) if len(weight) else 1.0,
        frac_at_floor=float((ratio <= floor + 1e-6).mean()) if len(weight) else 0.0,
    )


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #
@dataclass
class Decider:
    """MBON rates -> a play/discard probability. No trained weights.

    ``play_drive = mean rate(approach MBONs) - mean rate(avoid MBONs) + bias``,
    in Hz. ``bias`` and ``temperature`` are calibrated once on pre-learning
    decision states (see :meth:`calibrate`) and then frozen: bias so the naive
    fly is near 50/50, temperature so it explores ~``explore`` of the time.
    """

    approach: NDArray[np.int32]
    avoid: NDArray[np.int32]
    window_ms: float
    bias: float = 0.0
    temperature: float = 1.0
    play_bias_p: float = 0.55
    #: Floor on exploration: ``p = eps/2 + (1 - eps) * sigmoid(drive / T)``. With
    #: eps = 0 an odour whose drive sits 2 SD from the median is taken almost
    #: deterministically, and since a DISCARD delivers no dopamine, such an odour
    #: can never generate the experience that would change it. The floor keeps
    #: every odour learnable.
    explore_floor: float = 0.0

    def raw_drive(self, counts: NDArray[np.int_]) -> float:
        """Approach minus avoidance mean rate, in Hz, before the bias."""
        scale = 1000.0 / self.window_ms
        a = float(np.asarray(counts)[self.approach].mean()) * scale
        b = float(np.asarray(counts)[self.avoid].mean()) * scale
        return a - b

    def play_drive(self, counts: NDArray[np.int_]) -> float:
        return self.raw_drive(counts) + self.bias

    def p_play(self, counts: NDArray[np.int_]) -> float:
        d = self.play_drive(counts)
        t = max(self.temperature, 1e-9)
        p = 1.0 / (1.0 + math.exp(-d / t))
        e = self.explore_floor
        return e / 2.0 + (1.0 - e) * p

    # -- calibration -------------------------------------------------------
    def calibrate(self, raw: Sequence[float], explore: float = 0.20) -> dict:
        """Freeze ``bias`` and ``temperature`` from pre-learning raw drives.

        ``bias`` is set so the median naive state sits at ``play_bias_p`` (a
        slight play bias, as specified), i.e. the naive fly is near 50/50 and
        cannot win by being one-sided. ``temperature`` is chosen so the mean
        probability of taking the *minority* action over these same states is
        ``explore`` -- the exploration rate.
        """
        r = np.asarray(list(raw), np.float64)
        if len(r) == 0:
            raise ValueError("no calibration samples")
        med = float(np.median(r))
        spread = float(np.percentile(r, 84) - np.percentile(r, 16)) or 1.0

        e = self.explore_floor

        def mean_explore(t: float, b: float) -> float:
            p = 1.0 / (1.0 + np.exp(-(r + b) / max(t, 1e-9)))
            p = e / 2.0 + (1.0 - e) * p
            return float(np.minimum(p, 1.0 - p).mean())

        # temperature first (bias only shifts the centre a little), bisecting on
        # a monotone function of t.
        lo, hi = 1e-4, max(spread * 100.0, 1.0)
        b0 = -med
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            if mean_explore(mid, b0) < explore:
                lo = mid
            else:
                hi = mid
        t = math.sqrt(lo * hi)
        # bias so the median state plays with probability play_bias_p (after the
        # floor is applied, so the requested bias is the one the fly acts on)
        q = (self.play_bias_p - e / 2.0) / (1.0 - e) if e < 1.0 else 0.5
        q = min(max(q, 1e-6), 1.0 - 1e-6)
        target = math.log(q / (1.0 - q)) * t
        self.temperature = float(t)
        self.bias = float(target - med)
        return dict(
            n=int(len(r)),
            raw_median=med,
            raw_p16=float(np.percentile(r, 16)),
            raw_p84=float(np.percentile(r, 84)),
            raw_min=float(r.min()),
            raw_max=float(r.max()),
            raw_sd=float(r.std()),
            bias=self.bias,
            temperature=self.temperature,
            explore_target=explore,
            explore_achieved=mean_explore(self.temperature, self.bias),
            play_bias_p=self.play_bias_p,
            explore_floor=e,
        )

    def to_dict(self) -> dict:
        return dict(
            n_approach=int(len(self.approach)),
            n_avoid=int(len(self.avoid)),
            window_ms=self.window_ms,
            bias=self.bias,
            temperature=self.temperature,
            play_bias_p=self.play_bias_p,
            explore_floor=self.explore_floor,
        )
