"""Event-driven leaky integrate-and-fire over the MaleCNS brain-only CSR graph.

Kernel ported from vendor/doomfly/doom/engine.py (audited numba `advance`), with the
Shiu et al. 2024 constants from vendor/fly-craftax/flycraftax/brain.py:

    dt 0.1 ms | v_rest -52 mV | v_reset -52 mV | v_th -45 mV
    membrane tau 20 ms | synaptic tau 5 ms | refractory 2.2 ms | delay 1.8 ms
    w_syn 0.275 mV per synapse (already folded into Graph.weight)

Brian2 refractory semantics reproduced from the reference: `v` and `g` carry
`(unless refractory)`, so EVERY write to them is skipped while a neuron is
refractory -- a synaptic `g += w` landing on a refractory target is dropped
outright, not buffered. Reset runs after the synapse slot.

`drive` is a tonic current expressed as mV of steady-state depolarisation above
rest (doomfly's `drive*(1-av)` form), so it is dt-invariant in meaning: drive=30
settles v at -22 mV, i.e. tonic firing with ISI ~= 2.2 + 5.3 = 7.5 ms (~133 Hz).

`vth` is a per-neuron threshold array (default V_TH everywhere) and `bias` a
per-neuron tonic-current offset added to whatever `step` is given. Both exist so
`flybalatro.tuning` can calibrate a population without touching the connectome;
with the default `Tuning()` they are exactly the old constants and the old
behaviour, bit for bit.

This is a coarse spiking approximation. It does not model ion channels,
receptors, neuromodulation, graded transmission, or plasticity.
"""

from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

V_REST = -52.0
V_RESET = -52.0
V_TH = -45.0
T_MBR = 20.0
TAU_SYN = 5.0
T_RFC = 2.2
T_DLY = 1.8


@njit(cache=True)
def _seed_rng(seed):  # numba keeps its own RNG state; seed it inside jitted code
    np.random.seed(seed)


@njit(cache=True)
def advance(
    ptr, post, weight, v, g, refractory, drive, kick_prob, kick_mv,
    queue, queue_count, cursor, steps, dt, counts, active, active_flag, nactive,
    vth,
):
    av = math.exp(-dt / 20.0)
    ag = math.exp(-dt / 5.0)
    coupling = (av - ag) / 3.0
    delay_slots = queue.shape[0]
    n_dly = int(round(1.8 / dt))
    n_rfc = int(round(2.2 / dt))
    use_kick = kick_mv > 0.0
    for _step in range(steps):
        # Delivery occurs after integration/threshold and before reset, matching the
        # reference schedule. A spike at tick t arrives at t+18 for dt=.1.
        slot = cursor % delay_slots
        for k in range(nactive[0]):
            i = active[k]
            if refractory[i] > 0:
                refractory[i] -= 1
            if refractory[i] == 0:
                v[i] = -52.0 + (v[i] + 52.0) * av + drive[i] * (1.0 - av) + g[i] * coupling
                g[i] *= ag
                if use_kick and kick_prob[i] > 0.0:
                    if np.random.random() < kick_prob[i]:
                        v[i] += kick_mv
                if v[i] > vth[i]:
                    counts[i] += 1
                    future = (cursor + n_dly) % delay_slots
                    queue[future, queue_count[future]] = i
                    queue_count[future] += 1
        for q in range(queue_count[slot]):
            i = queue[slot, q]
            for e in range(ptr[i], ptr[i + 1]):
                j = post[e]
                # Brian2's (unless refractory) makes g read-only, including
                # synaptic writes. Do not save arrivals for a later release.
                if refractory[j] > 0:
                    continue
                g[j] += weight[e]
                if active_flag[j] == 0:
                    active_flag[j] = 1
                    active[nactive[0]] = j
                    nactive[0] += 1
        queue_count[slot] = 0
        future = (cursor + n_dly) % delay_slots
        for q in range(queue_count[future]):
            i = queue[future, q]
            v[i] = -52.0
            g[i] = 0.0
            refractory[i] = n_rfc
        cursor += 1
    return cursor


class Brain:
    """Frozen spiking reservoir over a `flybalatro.connectome.Graph`."""

    def __init__(self, graph, dt: float = 0.1, seed: int = 0, v_jitter: float = 6.9,
                 kick_mv: float = 0.0, _post_override=None, tuning=None):
        if dt != 0.1:
            raise ValueError("This audited kernel supports only dt=0.1 ms.")
        graph.validate()
        self.graph = graph
        self.dt = dt
        self.seed = int(seed)
        self.v_jitter = float(v_jitter)
        self.kick_mv = float(kick_mv)
        self.shuffle_seed = None

        self.ptr = np.ascontiguousarray(graph.ptr, dtype=np.int64)
        self.post = np.ascontiguousarray(
            graph.post if _post_override is None else _post_override, dtype=np.int32
        )
        # Always a private copy: np.ascontiguousarray would alias the cached
        # graph's array, so a Tuning scaling edges in place would leak into every
        # other Brain built from the same graph.
        self.weight = np.array(graph.weight, dtype=np.float32, order="C", copy=True)
        self.n = graph.n
        if len(self.post) != len(self.weight) or int(self.ptr[-1]) != len(self.post):
            raise ValueError("Invalid CSR graph")
        if self.n and (self.post.min() < 0 or self.post.max() >= self.n):
            raise ValueError("Graph index out of bounds")

        # Calibration arrays (see flybalatro.tuning). Defaults reproduce the
        # hard-coded Shiu constants exactly.
        self.vth = np.full(self.n, V_TH, np.float32)
        self.bias = np.zeros(self.n, np.float32)
        self.tuning = tuning
        self.tuning_info: dict = {}
        if tuning is not None:
            self.tuning_info = tuning.apply(graph, self.weight, self.vth, self.bias)
        self._has_bias = bool(np.any(self.bias != 0.0))

        n_slots = int(round(T_DLY / dt)) + 1
        self.v = np.empty(self.n, np.float32)
        self.g = np.zeros(self.n, np.float32)
        self.drive = np.zeros(self.n, np.float32)
        self.kick_prob = np.zeros(self.n, np.float32)
        self.refractory = np.zeros(self.n, np.int16)
        self.queue = np.zeros((n_slots, self.n), np.int32)
        self.queue_count = np.zeros(n_slots, np.int32)
        self.counts = np.zeros(self.n, np.int32)
        self.active = np.zeros(self.n, np.int32)
        self.active_flag = np.zeros(self.n, np.uint8)
        self.nactive = np.zeros(1, np.int32)
        self.cursor = 0
        self.total_spikes = 0
        self.sim_ms = 0.0
        self._rng = np.random.default_rng(self.seed)
        _seed_rng(self.seed)
        # Fixed, drawn once: heterogeneous v0 breaks synchrony, and keeping it
        # constant across resets makes the reservoir a deterministic function of
        # the input (no trial-to-trial noise for the linear probe to fight).
        self._v0 = np.full(self.n, V_REST, np.float32)
        if self.v_jitter > 0:
            self._v0 += self._rng.uniform(0.0, self.v_jitter, self.n).astype(np.float32)
        self.reset()

    # ---- construction --------------------------------------------------------
    def shuffled(self, seed: int) -> "Brain":
        """Control graph: globally permute edge targets.

        `ptr` and `weight` are untouched, so per-presynaptic out-degree, synapse
        counts and the sign attached to each presynaptic neuron are preserved;
        `post` as a multiset is unchanged, so every neuron keeps its exact
        in-degree. Self-loops and parallel edges become possible -- acceptable
        for a wiring control.
        """
        rng = np.random.default_rng(seed)
        post = rng.permutation(self.graph.post).astype(np.int32)
        b = Brain(self.graph, dt=self.dt, seed=self.seed, v_jitter=self.v_jitter,
                  kick_mv=self.kick_mv, _post_override=post, tuning=self.tuning)
        b.shuffle_seed = int(seed)
        return b

    # ---- state ---------------------------------------------------------------
    def reset(self) -> None:
        """Clear all dynamic state back to the fixed jittered v0.

        The jitter is not cosmetic: with identical v0 every driven neuron crosses
        threshold on the same tick and the whole brain locks into synchronous
        volleys, which destroys any rate code the readout could use.
        """
        self.v[:] = self._v0
        self.g.fill(0.0)
        self.refractory.fill(0)
        self.queue_count.fill(0)
        self.counts.fill(0)
        self.active_flag.fill(0)
        self.nactive[0] = 0
        self.cursor = 0

    def _activate(self, idx) -> None:
        idx = np.asarray(idx, np.int32)
        if len(idx) == 0:
            return
        new = idx[self.active_flag[idx] == 0]
        if len(new) == 0:
            return
        k = int(self.nactive[0])
        self.active[k:k + len(new)] = new
        self.active_flag[new] = 1
        self.nactive[0] = k + len(new)

    # ---- simulation ----------------------------------------------------------
    def step(self, drive, duration_ms: float, reset_counts: bool = True):
        """Run `duration_ms` of brain time under tonic `drive`; return spike counts.

        drive: float32[n], mV of steady-state depolarisation above rest per neuron.
        Returns (counts int32[n], elapsed_seconds).
        """
        drive = np.asarray(drive, np.float32)
        if drive.shape != (self.n,) or not np.isfinite(drive).all():
            raise ValueError("drive must be a finite float32 array of length n")
        steps = int(round(duration_ms / self.dt))
        if steps < 1:
            raise ValueError("Duration too short")
        self.drive[:] = drive
        if self._has_bias:
            self.drive += self.bias
        if reset_counts:
            self.counts.fill(0)
        # Drive-only neurons are never reached by synaptic delivery, so they would
        # never enter the active set on their own.
        self._activate(np.flatnonzero(self.drive != 0.0))
        if self.kick_mv > 0:
            self._activate(np.flatnonzero(self.kick_prob > 0.0))
        t0 = time.perf_counter()
        self.cursor = advance(
            self.ptr, self.post, self.weight, self.v, self.g, self.refractory,
            self.drive, self.kick_prob, self.kick_mv, self.queue, self.queue_count,
            self.cursor, steps, self.dt, self.counts, self.active, self.active_flag,
            self.nactive, self.vth,
        )
        elapsed = time.perf_counter() - t0
        self.total_spikes += int(self.counts.sum())
        self.sim_ms += steps * self.dt
        return self.counts.copy(), elapsed

    def set_poisson(self, idx, hz: float, kick_mv: float | None = None) -> None:
        """Poisson-input alternative to tonic drive: per-step Bernoulli kicks to `idx`.

        Use instead of (or alongside) `drive` if constant current produces
        pathological synchrony. `kick_mv` must be set at construction (>0) for the
        kernel to sample at all.
        """
        if kick_mv is not None:
            self.kick_mv = float(kick_mv)
        if self.kick_mv <= 0:
            raise ValueError("kick_mv must be > 0 to use Poisson input")
        self.kick_prob.fill(0.0)
        self.kick_prob[np.asarray(idx, np.int32)] = float(hz) * self.dt / 1000.0

    @property
    def n_active(self) -> int:
        return int(self.nactive[0])

    def warmup(self) -> None:
        """Compile the kernel outside any timed region (numba cache=True still JITs once)."""
        d = np.zeros(self.n, np.float32)
        self.step(d, self.dt)
        self.reset()
