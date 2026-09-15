"""Calibration knobs for the frozen LIF network. NOT connectome data.

Everything in this module is a **free parameter we chose**, exactly like the one
global ``W_SYN = 0.275`` already folded into ``Graph.weight``. None of it is
measured, and none of it comes out of MaleCNS. The connectome supplies topology,
synapse counts and a predicted sign per neuron; it does not supply per-synapse
efficacy, receptor identity, input resistance, spike threshold, or adaptation.
The untuned model therefore has a free scale factor per pathway and a free
threshold per cell type, and picking values for a few of them is calibration, not
a finding. Anything measured after tuning has to be reported together with the
tuning that produced it.

Why these three knobs in particular (all standard mushroom-body physiology, none
of it new to this project):

``apl_scale``
    The APL neuron is the giant GABAergic interneuron of the mushroom body, one
    per hemisphere, and the substrate of the winner-take-all normalisation that
    keeps the Kenyon-cell code sparse (Lin et al. 2014: blocking APL roughly
    doubles KC odour responses and destroys sparseness). Our sign for it is only
    "GABA -> inhibitory, weight = synapse count x 0.275 mV"; the actual gain of
    that feedback loop is unknown, so it gets a scale factor. Applies to **every
    outgoing synapse of APL**, not only APL->KC (its output is dominated by KCs
    anyway; ``apply`` reports the split).

``alpn_kc_scale``
    Each KC samples ~6 of ~50 glomeruli and needs several coincident PN inputs
    to spike; a single PN-KC connection is subthreshold in vivo. A synapse-count
    proxy for efficacy has no reason to land at the right absolute level, so the
    PN->KC drive gets its own scale.

``kc_vth_offset_mv``
    KCs are high-threshold, low-input-resistance cells sitting near
    silence. A uniform -45 mV threshold for every neuron in the brain gives them
    no such property, so we allow raising v_th for KCs specifically.

``kc_kc_scale``
    A fourth knob, added after the 1-D sweeps below plateaued. In this graph the
    Kenyon cells make 33,247 excitatory synapses **on each other** (14.3 mV of
    conductance per KC, 35% of all excitatory drive a KC receives, second only to
    ALPN's 26.1 mV). The canonical mushroom body is a feed-forward PN -> KC
    divergence normalised by APL; KC -> KC excitatory recurrence is not part of
    that circuit model, and at this strength it makes the calyx a positive-feedback
    amplifier whose winners are set by KC in-degree rather than by the input.
    (Whether those edges are real chemical synapses between KC axons or an
    artefact of segmentation is not something this project can settle, which is
    exactly why their gain is a free parameter here.)

``apl_kc_only``
    A flag, not a gain: when set, ``apl_scale`` multiplies only the APL -> Kenyon
    cell synapses instead of every outgoing APL synapse. APL's documented role
    (Lin et al. 2014) is the winner-take-all normalisation *of the Kenyon cells*,
    and 94% of its output weight goes there, so scaling the other 6% was
    incidental. It stops mattering only if nothing downstream reads the MBONs:
    APL supplies -1.9 to -160 mV of direct inhibition to individual MBONs, and at
    ``apl_scale = 2`` that is enough to hold MBON01-07 (the glutamatergic,
    PAM-innervated output neurons) hundreds of mV below threshold for every
    input. Default 0.0, so every earlier measurement is unchanged.

``apl_mbon_scale``
    A separate gain on the 72 APL -> MBON synapses, because APL's effect on the
    MBONs is where this model is furthest from the animal. APL is a *non-spiking*
    neuron in vivo, a giant interneuron that releases GABA in a graded,
    compartmentalised way (Lin et al. 2014; Amin et al. 2020), and this kernel
    has no graded transmission, so it fires APL at ~250 Hz and delivers its full
    synapse-count weight as spike-triggered conductance. Measured in one 50 ms
    window at ``apl_scale = 2``: APL puts -3,198 mV into MBON01 and -3,736 mV into
    MBON05 against 259 mV and 169 mV of Kenyon-cell excitation, so every
    glutamatergic PAM-compartment MBON sits 60-390 mV below rest for every input
    and never spikes. That is not sparseness, it is a silenced output stage, and
    it makes any MBON readout (and therefore any KC -> MBON plasticity rule) blind
    on that side. Default 1.0, so every earlier measurement is unchanged.

``kc_input_norm``
    A fifth knob, and the only one that is not a uniform rescaling. The four above
    all multiply a pathway or shift a threshold by the *same* amount for every
    Kenyon cell, so none of them can change which KCs cross threshold first --
    only how many do. Measured on this graph, the total ALPN-driven conductance a
    KC receives has a between-KC SD of 112.6 mV (set by how many PN synapses that
    KC happens to have) against a within-KC, across-input SD of 9.5 mV: a ratio of
    11.8. Threshold crossing is therefore decided by KC in-degree, not by the
    odour, which is why the sparsity knobs above produce a sparse code made of the
    *same* cells every time. ``kc_input_norm`` interpolates each KC's ALPN input
    weights toward the value that gives every KC the population-mean total:

        factor_i = (1 - x) + x * (mean_total / total_i),  x = kc_input_norm

    This is the equal-total-claw-weight assumption that mushroom-body theory works
    under (Litwin-Kumar et al. 2017; each KC samples ~6 random glomeruli and the
    model assumes comparable total drive), plus the observation that KCs
    homeostatically regulate their own excitability. It is applied **once at
    construction from the connectome's own weights**, a static calibration, not
    plasticity: nothing about it changes while the network runs, and it does not
    depend on the input, the labels, or any measured activity.

``kc_vth_offsets``
    A **per-Kenyon-cell** threshold offset, in mV, indexed by the KC's row in
    ``graph.kc_indices()`` and added on top of the uniform ``kc_vth_offset_mv``.
    This is the one knob in this module that is not a population-wide constant,
    and it exists because a population-wide constant cannot fix the problem
    measured in ``outputs/plast/REPORT.md``: at the chosen operating point 188 of
    the ~211 Kenyon cells active per odour come from a pool that responds to more
    than half of *all* odours, so dopamine-gated depression assigns credit to the
    same cells whatever the fly smelled, and conditioning generalises completely.
    Raising or lowering every KC's threshold by the same amount changes how many
    cells cross it, never which ones, the same argument the ``kc_input_norm``
    paragraph above makes.

    Real Kenyon cells do have individually set thresholds. KC excitability is
    under homeostatic control: the KC-intrinsic potassium conductances and the
    APL feedback loop together hold each cell near a set point, and the
    population-level consequence is the textbook one: an individual KC responds
    to roughly 5-10% of odours, not to half of them (Turner et al. 2008; Honegger
    et al. 2011; Lin et al. 2014). That per-cell set point is a *property of the
    animal* a synapse-count model has no way to inherit, in exactly the way it has
    no way to inherit per-synapse efficacy. So it is calibrated the way the animal
    arrives at it: iteratively, from each cell's own measured response rate, with
    no reference to the labels, the task, the reward, or the readout, only to
    "how often did I fire". See ``scripts/kc_homeo.py``.

    Mechanically: ``vth[kc[i]] += kc_vth_offsets[i]``. Stored as a tuple so
    ``Tuning`` stays a frozen, hashable, JSON-round-trippable dataclass; ``()``
    (the default) is a no-op and every earlier measurement is bit-identical.

``kc_mbon_out_scale``
    A per-Kenyon-cell multiplier on that KC's outgoing **KC -> MBON** synapses,
    same indexing as ``kc_vth_offsets``. A static calibration applied once at
    ``Brain`` construction (so ``KcMbonPlasticity`` picks it up as its ``w0`` and
    the weight-bound invariant is unaffected), provided as the documented fallback
    if per-cell thresholds alone leave a residual broadly-tuned population
    dominating the credit assignment: a cell that still answers to 2x its target
    share of odours contributes half as much to the output stage. Default ``()``
    is a no-op.

``kc_bias_mv`` is the same idea as ``kc_vth_offset_mv`` in current units (a tonic
offset added to the drive) and is provided for completeness; the sweeps do not
use it.

Mechanically these are (a) per-edge weight multipliers and (b) per-neuron
threshold / bias offsets, applied once at ``Brain`` construction. The default
``Tuning()`` is a no-op and the brain is bit-identical to the untuned model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np

#: Exact annotation strings used to find the tuned populations in MaleCNS v1.0.
APL_TYPE = "APL"                 # type == "APL"; class is the empty string,
                                 # superclass == "cb_intrinsic", 1 per hemisphere
KC_CLASS = "Kenyon_Cell"         # class == "Kenyon_Cell"
ALPN_CLASS = "ALPN"              # class == "ALPN"
MBON_CLASS = "MBON"              # class == "MBON"


def apl_indices(graph) -> np.ndarray:
    """The two APL neurons (one per hemisphere), by exact type string."""
    return np.flatnonzero(graph.type.astype(str) == APL_TYPE).astype(np.int32)


def scale_out_edges(graph, weight: np.ndarray, pre: np.ndarray, scale: float,
                    post_mask: np.ndarray | None = None) -> dict:
    """Multiply the weight of every edge pre -> (post_mask) by `scale`, in place.

    Returns provenance: how many edges and how much total |mV| were touched.
    Tolerates an empty selection (e.g. the synthetic test graph has no APL).
    """
    n_edges = 0
    total_before = 0.0
    n_targets = 0
    for i in np.asarray(pre, np.int64):
        a, b = int(graph.ptr[i]), int(graph.ptr[i + 1])
        if b <= a:
            continue
        if post_mask is None:
            sl = slice(a, b)
            n_edges += b - a
            total_before += float(np.abs(weight[sl]).sum())
            n_targets += int(np.unique(graph.post[a:b]).size)
            weight[sl] *= scale
        else:
            m = post_mask[graph.post[a:b]]
            if not m.any():
                continue
            idx = np.arange(a, b)[m]
            n_edges += int(m.sum())
            total_before += float(np.abs(weight[idx]).sum())
            n_targets += int(np.unique(graph.post[idx]).size)
            weight[idx] *= scale
    return dict(edges=int(n_edges), weight_abs_mv_before=round(total_before, 2),
                weight_abs_mv_after=round(total_before * float(scale), 2),
                distinct_targets=int(n_targets), scale=float(scale))


@dataclass(frozen=True)
class Tuning:
    """Calibration knobs. Defaults = untuned model, bit for bit.

    apl_scale          multiplier on every outgoing synapse of APL (inhibitory)
    alpn_kc_scale      multiplier on ALPN -> Kenyon_Cell synapses
    kc_vth_offset_mv   mV added to v_th of every Kenyon cell (raises threshold)
    kc_bias_mv         mV of tonic drive added to every Kenyon cell
    kc_kc_scale        multiplier on Kenyon_Cell -> Kenyon_Cell synapses
    kc_input_norm      0..1; equalise each KC's total ALPN in-weight (1 = fully)
    """

    apl_scale: float = 1.0
    alpn_kc_scale: float = 1.0
    kc_vth_offset_mv: float = 0.0
    kc_bias_mv: float = 0.0
    kc_kc_scale: float = 1.0
    kc_input_norm: float = 0.0
    apl_kc_only: float = 0.0
    apl_mbon_scale: float = 1.0
    #: per-KC, indexed by ``graph.kc_indices()``; () = no-op. See the module
    #: docstring: the only non-uniform threshold knob in this module.
    kc_vth_offsets: Tuple[float, ...] = ()
    #: per-KC multiplier on that KC's outgoing KC -> MBON synapses; () = no-op.
    kc_mbon_out_scale: Tuple[float, ...] = ()

    #: field -> (short name for `label`, default value)
    KNOBS = dict(apl_scale=("apl", 1.0), alpn_kc_scale=("pnkc", 1.0),
                 kc_vth_offset_mv=("vth", 0.0), kc_bias_mv=("bias", 0.0),
                 kc_kc_scale=("kckc", 1.0), kc_input_norm=("norm", 0.0),
                 apl_kc_only=("aplkc", 0.0), apl_mbon_scale=("aplmb", 1.0),
                 kc_vth_offsets=("vtho", ()), kc_mbon_out_scale=("kcmb", ()))

    #: The knobs whose value is a per-KC vector rather than a scalar.
    VECTOR_KNOBS = ("kc_vth_offsets", "kc_mbon_out_scale")

    def is_default(self) -> bool:
        return all(tuple(getattr(self, f)) == tuple(d)
                   if f in self.VECTOR_KNOBS else getattr(self, f) == d
                   for f, (_, d) in self.KNOBS.items())

    def to_dict(self) -> dict:
        """JSON-ready. The per-KC vectors are omitted entirely when empty, so a
        default ``Tuning`` serialises exactly as it did before they existed."""
        out = {k: float(v) for k, v in asdict(self).items()
               if k not in self.VECTOR_KNOBS}
        for f in self.VECTOR_KNOBS:
            v = tuple(getattr(self, f))
            if v:
                out[f] = [float(x) for x in v]
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Tuning":
        fields = {"apl_scale", "alpn_kc_scale", "kc_vth_offset_mv", "kc_bias_mv",
                  "kc_kc_scale", "kc_input_norm", "apl_kc_only",
                  "apl_mbon_scale", "kc_vth_offsets", "kc_mbon_out_scale"}
        unknown = set(d) - fields
        if unknown:
            raise ValueError(f"unknown tuning keys: {sorted(unknown)}")
        kw = {k: (tuple(float(x) for x in v) if k in cls.VECTOR_KNOBS
                  else float(v)) for k, v in d.items()}
        return cls(**kw)

    def label(self) -> str:
        """Compact, stable name: only the knobs that differ from the default."""
        parts = []
        for f, (short, d) in self.KNOBS.items():
            v = getattr(self, f)
            if f in self.VECTOR_KNOBS:
                if tuple(v):
                    parts.append(f"{short}{len(tuple(v))}")
            elif v != d:
                parts.append(f"{short}{v:g}")
        return "_".join(parts) if parts else "default"

    # ---- application ---------------------------------------------------------
    def apply(self, graph, weight: np.ndarray, vth: np.ndarray,
              bias: np.ndarray) -> dict:
        """Mutate a Brain's private weight / vth / bias arrays. Returns provenance.

        Called by ``Brain.__init__``; `weight` must already be a private copy of
        ``graph.weight`` (it is).
        """
        cls = graph.cls.astype(str)
        kc = np.flatnonzero(cls == KC_CLASS).astype(np.int32)
        alpn = np.flatnonzero(cls == ALPN_CLASS).astype(np.int32)
        apl = apl_indices(graph)
        kc_mask = np.zeros(graph.n, bool)
        kc_mask[kc] = True
        mbon_mask = np.zeros(graph.n, bool)
        mbon_mask[np.flatnonzero(cls == MBON_CLASS)] = True

        info: dict = dict(
            tuning=self.to_dict(),
            is_default=self.is_default(),
            populations=dict(APL=int(len(apl)), KC=int(len(kc)), ALPN=int(len(alpn))),
            apl_body_ids=[int(x) for x in graph.body_id[apl]],
        )
        if self.apl_scale != 1.0:
            if self.apl_kc_only:
                # APL's documented role (Lin et al. 2014) is Kenyon-cell
                # sparsening. Restricting the scale to APL -> KC leaves APL's
                # (much smaller, 2% of its output weight) direct inhibition of
                # the MBONs at the connectome's own level, which matters when
                # the MBONs are the readout rather than a diagnostic.
                info["apl_to_kc_only"] = scale_out_edges(
                    graph, weight, apl, self.apl_scale, post_mask=kc_mask)
            else:
                info["apl_all_out"] = scale_out_edges(graph, weight, apl,
                                                      self.apl_scale)
            # Report (do not re-scale) how much of that lands on KCs.
            info["apl_to_kc_share"] = _share(graph, apl, kc_mask)
        if self.apl_mbon_scale != 1.0:
            info["apl_to_mbon"] = scale_out_edges(
                graph, weight, apl, self.apl_mbon_scale, post_mask=mbon_mask)
        if self.alpn_kc_scale != 1.0 or self.kc_input_norm != 0.0:
            info["alpn_to_kc"] = self._scale_alpn_to_kc(graph, weight, alpn, kc,
                                                        kc_mask)
        if self.kc_kc_scale != 1.0:
            info["kc_to_kc"] = scale_out_edges(
                graph, weight, kc, self.kc_kc_scale, post_mask=kc_mask)
        if self.kc_vth_offset_mv != 0.0:
            vth[kc] += np.float32(self.kc_vth_offset_mv)
            info["kc_vth_mv"] = float(vth[kc[0]]) if len(kc) else None
        if self.kc_bias_mv != 0.0:
            bias[kc] += np.float32(self.kc_bias_mv)
            info["kc_bias_mv"] = float(self.kc_bias_mv)
        if tuple(self.kc_vth_offsets):
            info["kc_vth_offsets"] = self._apply_kc_vth_offsets(vth, kc)
        if tuple(self.kc_mbon_out_scale):
            info["kc_mbon_out_scale"] = self._apply_kc_mbon_out_scale(
                graph, weight, kc, mbon_mask)
        return info

    # ---- per-KC vectors ------------------------------------------------------
    def _kc_vector(self, name: str, kc: np.ndarray) -> np.ndarray:
        """Validate a per-KC vector against the KC population and return float32.

        The index is the KC's row in ``graph.kc_indices()``, which is
        ``np.flatnonzero(cls == "Kenyon_Cell")``, the same expression ``apply``
        uses to build ``kc``, so a length match is the whole contract. A
        mismatch is a hard error rather than a broadcast, because a silently
        misaligned offset vector would look exactly like a biological result.
        """
        v = np.asarray(tuple(getattr(self, name)), np.float32)
        if v.shape != (len(kc),):
            raise ValueError(
                f"{name} has length {v.shape[0]}, but this graph has "
                f"{len(kc)} Kenyon cells; it must be indexed by "
                "graph.kc_indices()"
            )
        if not np.isfinite(v).all():
            raise ValueError(f"{name} must be finite")
        return v

    def _apply_kc_vth_offsets(self, vth: np.ndarray, kc: np.ndarray) -> dict:
        off = self._kc_vector("kc_vth_offsets", kc)
        vth[kc] += off
        return dict(
            n_kc=int(len(kc)), n_nonzero=int((off != 0.0).sum()),
            mean_mv=round(float(off.mean()), 4),
            sd_mv=round(float(off.std()), 4),
            min_mv=round(float(off.min()), 4), max_mv=round(float(off.max()), 4),
            vth_min=round(float(vth[kc].min()), 4),
            vth_max=round(float(vth[kc].max()), 4),
        )

    def _apply_kc_mbon_out_scale(self, graph, weight: np.ndarray,
                                 kc: np.ndarray, mbon_mask: np.ndarray) -> dict:
        sc = self._kc_vector("kc_mbon_out_scale", kc)
        if float(sc.min()) < 0.0:
            raise ValueError("kc_mbon_out_scale must be non-negative")
        n_edges = 0
        before = after = 0.0
        for row, i in enumerate(np.asarray(kc, np.int64)):
            f = float(sc[row])
            if f == 1.0:
                continue
            a, b = int(graph.ptr[i]), int(graph.ptr[i + 1])
            if b <= a:
                continue
            m = mbon_mask[graph.post[a:b]]
            if not m.any():
                continue
            idx = np.arange(a, b)[m]
            n_edges += int(m.sum())
            before += float(np.abs(weight[idx]).sum())
            weight[idx] *= np.float32(f)
            after += float(np.abs(weight[idx]).sum())
        return dict(
            n_kc=int(len(kc)), n_scaled_kc=int((sc != 1.0).sum()),
            edges=int(n_edges),
            weight_abs_mv_before=round(before, 2),
            weight_abs_mv_after=round(after, 2),
            scale_min=round(float(sc.min()), 4),
            scale_max=round(float(sc.max()), 4),
            scale_mean=round(float(sc.mean()), 4),
        )


    def _scale_alpn_to_kc(self, graph, weight, alpn, kc, kc_mask) -> dict:
        """ALPN -> KC edges: global `alpn_kc_scale` x per-KC normalisation factor.

        The factor is derived from ``graph.weight`` (the untouched connectome), so
        the result does not depend on the order in which knobs are applied.
        """
        totals = np.zeros(graph.n, np.float64)
        for i in np.asarray(alpn, np.int64):
            a, b = int(graph.ptr[i]), int(graph.ptr[i + 1])
            m = kc_mask[graph.post[a:b]]
            if m.any():
                np.add.at(totals, graph.post[a:b][m], np.abs(graph.weight[a:b][m]))
        kc_tot = totals[kc]
        have = kc_tot > 0
        mean_tot = float(kc_tot[have].mean()) if have.any() else 0.0
        factor = np.ones(graph.n, np.float64)
        x = float(self.kc_input_norm)
        if x != 0.0:
            eq = np.ones(len(kc))
            eq[have] = mean_tot / kc_tot[have]
            factor[kc] = (1.0 - x) + x * eq
        factor *= float(self.alpn_kc_scale)

        n_edges = 0
        before = after = 0.0
        for i in np.asarray(alpn, np.int64):
            a, b = int(graph.ptr[i]), int(graph.ptr[i + 1])
            p = graph.post[a:b]
            m = kc_mask[p]
            if not m.any():
                continue
            idx = np.arange(a, b)[m]
            n_edges += int(m.sum())
            before += float(np.abs(weight[idx]).sum())
            weight[idx] *= factor[p[m]].astype(np.float32)
            after += float(np.abs(weight[idx]).sum())
        scaled = totals[kc] * factor[kc]
        return dict(
            edges=int(n_edges), weight_abs_mv_before=round(before, 2),
            weight_abs_mv_after=round(after, 2),
            alpn_kc_scale=float(self.alpn_kc_scale),
            kc_input_norm=x,
            kc_alpn_in_weight_mean_before=round(float(kc_tot.mean()), 2),
            kc_alpn_in_weight_sd_before=round(float(kc_tot.std()), 2),
            kc_alpn_in_weight_mean_after=round(float(scaled.mean()), 2),
            kc_alpn_in_weight_sd_after=round(float(scaled.std()), 2),
            kc_without_alpn_input=int((~have).sum()),
        )


def _share(graph, pre: np.ndarray, post_mask: np.ndarray) -> dict:
    """Fraction of `pre`'s outgoing edges / |mV| that land on `post_mask`."""
    e_all = e_sel = 0
    w_all = w_sel = 0.0
    for i in np.asarray(pre, np.int64):
        a, b = int(graph.ptr[i]), int(graph.ptr[i + 1])
        if b <= a:
            continue
        w = np.abs(graph.weight[a:b])
        m = post_mask[graph.post[a:b]]
        e_all += b - a
        w_all += float(w.sum())
        e_sel += int(m.sum())
        w_sel += float(w[m].sum())
    return dict(edges_total=e_all, edges_to_target=e_sel,
                weight_abs_mv_total=round(w_all, 1),
                weight_abs_mv_to_target=round(w_sel, 1),
                fraction_of_weight_to_target=round(w_sel / w_all, 4) if w_all else None)


DEFAULT = Tuning()
