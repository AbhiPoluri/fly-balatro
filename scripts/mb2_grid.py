"""Gated test of the glomerular encoding: 12 settings x 2,000 real Balatro states.

Simulation phase only. Per setting it runs every sampled state through one 50 ms
window from reset, splits the window at 15 ms, and saves the spike-count matrices
for KC / MBON / ALPN / DN to ``outputs/mb2/counts_<label>.npz``. Metrics that need
no classifier (sparsity, always-on, between/within-type Jaccard, rates, ALPN total
CV, the MN9 sugar check) go straight into ``outputs/mb2/grid.json``; the probes are
a separate phase (``scripts/mb2_probe.py``) reading the saved matrices, so a probe
change never costs a resimulation.

Grid: ``apl_scale`` {1, 2, 5} x drive {tonic 30 mV, Poisson ~100 Hz} x
``kc_vth_offset_mv`` {0, 4}.

No plasticity. Nothing here adapts, and nothing depends on the labels.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from flybalatro import connectome as C
from flybalatro.brain import Brain, _seed_rng
from flybalatro.encode import GlomerularMap
from flybalatro.tuning import Tuning
from scripts.mb2_assign import assign
from scripts.mb_sparsity import kc_metrics, pop_rate, populations

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "mb2"
GRID = OUT_DIR / "grid.json"

#: Populations whose per-state spike counts are saved for the probe phase.
KEEP: Tuple[str, ...] = ("KC", "MBON", "ALPN", "DN")

#: Poisson drive. ``kick_mv`` is a construction-time constant, so a Poisson
#: setting needs its own Brain. 68.75 mV = 0.275 x 250, the value
#: ``scripts/brain_sweep.py`` already used: far above threshold, so one kick is
#: one spike and the driven ORN rate tracks the requested Hz (measured ~80 Hz at
#: 100 Hz requested, because a kick landing inside the 2.2 ms refractory period is
#: dropped). Tonic 30 mV gives ~133 Hz, so the two modes are NOT rate-matched;
#: the measured ORN rate is recorded per setting.
POISSON_KICK_MV = 0.275 * 250
POISSON_HZ = 100.0

#: Grid axes.
APL_SCALES: Tuple[float, ...] = (1.0, 2.0, 5.0)
VTH_OFFSETS: Tuple[float, ...] = (0.0, 4.0)
DRIVES: Tuple[str, ...] = ("tonic", "poisson")


def label_for(apl: float, drive: str, vth: float) -> str:
    return f"apl{apl:g}_{drive}_vth{vth:g}"


def grid_settings() -> List[dict]:
    out = []
    for apl in APL_SCALES:
        for drive in DRIVES:
            for vth in VTH_OFFSETS:
                out.append(dict(label=label_for(apl, drive, vth), apl_scale=apl,
                                drive=drive, kc_vth_offset_mv=vth,
                                kc_input_norm=0.0))
    return out


def build_brain(graph, cfg: dict, seed: int, v_jitter: float) -> Brain:
    t = Tuning(apl_scale=cfg["apl_scale"],
               kc_vth_offset_mv=cfg["kc_vth_offset_mv"],
               kc_input_norm=cfg.get("kc_input_norm", 0.0))
    kick = POISSON_KICK_MV if cfg["drive"] == "poisson" else 0.0
    return Brain(graph, seed=seed, v_jitter=v_jitter, tuning=t, kick_mv=kick)


# --- one window ---------------------------------------------------------------
def run_state(brain: Brain, gm: GlomerularMap, row: np.ndarray, mode: str,
              window_ms: float, split_ms: float, zeros: np.ndarray,
              noise_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """One window from reset; returns (full-window counts, counts before split)."""
    brain.reset()
    if mode == "tonic":
        d = gm.drive(row)
    else:
        brain.kick_prob.fill(0.0)
        brain.set_poisson(gm.indices(row), POISSON_HZ)
        # Reseed per state so the Poisson realisation is identical across
        # settings and reproducible across runs.
        _seed_rng(noise_seed)
        d = zeros
    brain.step(d, split_ms, reset_counts=True)
    early = brain.counts.copy()
    brain.step(d, window_ms - split_ms, reset_counts=False)
    return brain.counts.copy(), early


def simulate(brain: Brain, gm: GlomerularMap, relay: np.ndarray, pops: dict,
             mode: str, window_ms: float, split_ms: float, noise_base: int,
             label: str) -> dict:
    n = len(relay)
    full = {k: np.zeros((n, len(pops[k])), np.int16) for k in KEEP}
    early = {k: np.zeros((n, len(pops[k])), np.int16) for k in KEEP}
    orn_tot = np.zeros(n, np.int32)
    apl_tot = np.zeros(n, np.int32)
    brain_tot = np.zeros(n, np.int32)
    zeros = np.zeros(brain.n, np.float32)
    brain.warmup()
    t0 = time.perf_counter()
    for i, row in enumerate(relay):
        c, e = run_state(brain, gm, row, mode, window_ms, split_ms, zeros,
                         noise_base + i)
        for k in KEEP:
            full[k][i] = c[pops[k]]
            early[k][i] = e[pops[k]]
        orn_tot[i] = c[pops["ORN"]].sum()
        apl_tot[i] = c[pops["APL"]].sum()
        brain_tot[i] = c.sum()
        if (i + 1) % 250 == 0:
            print(f"    [{label}] {i+1}/{n}  {time.perf_counter()-t0:.0f}s",
                  flush=True)
    return dict(full=full, early=early, orn_tot=orn_tot, apl_tot=apl_tot,
                brain_tot=brain_tot, seconds=time.perf_counter() - t0)


# --- metrics ------------------------------------------------------------------
def jaccard_between_within(active: np.ndarray, labels: np.ndarray) -> dict:
    """Mean Jaccard of active sets for same-label vs different-label state pairs.

    float32 rather than int32 for the intersection matmul: counts here are well
    under 2**24 so it is exact, and it goes through BLAS instead of numpy's
    integer loop (seconds instead of minutes at 2,000 states).
    """
    A = np.ascontiguousarray(active, np.float32)
    n = len(A)
    inter = A @ A.T
    size = A.sum(1)
    union = size[:, None] + size[None, :] - inter
    iu = np.triu_indices(n, 1)
    u = union[iu]
    j = np.where(u > 0, inter[iu] / np.maximum(u, 1.0), 0.0)
    same = labels[iu[0]] == labels[iu[1]]
    within = float(j[same].mean())
    between = float(j[~same].mean())
    return dict(within=round(within, 4), between=round(between, 4),
                ratio=round(between / within, 4) if within > 0 else None,
                mean_set_size=round(float(size.mean()), 1),
                states_with_empty_set=int((size == 0).sum()))


def pop_extra(M: np.ndarray, labels: np.ndarray) -> dict:
    """Population diagnostics beyond ``mb_sparsity.pop_rate``: is anything off?"""
    mean = M.mean(0)
    have = mean > 0
    cv = M.std(0)[have] / mean[have]
    tot = M.sum(1).astype(np.float64)
    return dict(
        fraction_of_cells_ever_firing=round(float((M > 0).any(0).mean()), 4),
        cells_never_firing=int((~have).sum()),
        mean_per_cell_cv_across_states=round(float(cv.mean()), 4) if have.any() else None,
        total_count_mean=round(float(tot.mean()), 1),
        total_count_cv=round(float(tot.std() / tot.mean()), 4) if tot.mean() > 0 else None,
        jaccard_by_type=jaccard_between_within(M > 0, labels),
    )


def sim_metrics(got: dict, y_hand: np.ndarray, window_ms: float,
                split_ms: float) -> dict:
    kc = got["full"]["KC"].astype(np.int32)
    kc_late = (got["full"]["KC"].astype(np.int32)
               - got["early"]["KC"].astype(np.int32))
    m = dict(
        KC=kc_metrics(kc, window_ms),
        KC_late=kc_metrics(kc_late, window_ms - split_ms),
        KC_jaccard_by_type=jaccard_between_within(kc > 0, y_hand),
        KC_late_jaccard_by_type=jaccard_between_within(kc_late > 0, y_hand),
    )
    m["KC"]["always_on_over_active"] = (
        round(m["KC"]["always_on_fraction"] / m["KC"]["mean_active_fraction"], 3)
        if m["KC"]["mean_active_fraction"] > 0 else None)
    for k in ("MBON", "ALPN", "DN"):
        X = got["full"][k].astype(np.int32)
        m[k] = pop_rate(X, window_ms)
        m[k].update(pop_extra(X, y_hand))
    a = got["apl_tot"].astype(np.float64)
    m["ORN_mean_rate_hz"] = round(
        float(got["orn_tot"].mean() / 2639 * 1000.0 / window_ms), 2)
    m["APL_mean_rate_hz"] = round(float(a.mean() / 2 * 1000.0 / window_ms), 2)
    m["brain_spikes_per_state"] = round(float(got["brain_tot"].mean()), 1)
    m["sim_seconds"] = round(got["seconds"], 1)
    return m


# --- MN9, tonic and Poisson ---------------------------------------------------
def mn9_check(brain: Brain, pops: dict, n_total: int, mode: str,
              drive_mv: float = 30.0, window_ms: float = 200.0,
              matched_seed: int = 0, noise_seed: int = 991) -> dict:
    """Sugar GRNs -> MN9, with the two controls, under tonic or Poisson drive."""
    grn, mn9, orn = pops["GRN"], pops["MN9"], pops["ORN"]
    rng = np.random.default_rng(matched_seed)
    matched = rng.choice(orn, size=min(len(grn), len(orn)), replace=False)
    zeros = np.zeros(n_total, np.float32)
    out: dict = dict(mode=mode, window_ms=window_ms,
                     drive=(f"tonic {drive_mv} mV" if mode == "tonic"
                            else f"Poisson {POISSON_HZ} Hz, kick {POISSON_KICK_MV:g} mV"),
                     n_grn=int(len(grn)),
                     matched_control=f"{len(matched)} random ORNs, seed {matched_seed}")
    for name, idx in (("sugar_grn_drive", grn), ("no_drive", None),
                      ("matched_orn_control", matched)):
        brain.reset()
        if mode == "tonic":
            brain.kick_prob.fill(0.0)
            d = zeros.copy()
            if idx is not None:
                d[idx] = drive_mv
        else:
            brain.kick_prob.fill(0.0)
            if idx is not None:
                brain.set_poisson(idx, POISSON_HZ)
            _seed_rng(noise_seed)
            d = zeros
        counts, _ = brain.step(d, window_ms)
        out[name] = dict(
            mn9_spikes=[int(x) for x in counts[mn9]],
            mn9_spikes_total=int(counts[mn9].sum()),
            grn_rate_hz=round(float(counts[grn].mean() * 1000.0 / window_ms), 2),
            brain_spikes=int(counts.sum()),
            neurons_firing=int((counts > 0).sum()),
        )
    s = out["sugar_grn_drive"]["mn9_spikes_total"]
    z = out["no_drive"]["mn9_spikes_total"]
    mc = out["matched_orn_control"]["mn9_spikes_total"]
    out["verdict"] = dict(
        mn9_responds_to_sugar=bool(s > 0), silent_without_drive=bool(z == 0),
        sugar_selective=bool(s > max(1, 2 * mc)),
        summary=f"sugar {s} / no-drive {z} / matched-ORN {mc}",
    )
    brain.kick_prob.fill(0.0)
    return out


# --- driver -------------------------------------------------------------------
def load_grid() -> dict:
    if GRID.exists():
        return json.loads(GRID.read_text())
    return dict(meta={}, settings={})


def save_grid(d: dict) -> None:
    GRID.write_text(json.dumps(d, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--window", type=float, default=50.0)
    ap.add_argument("--split", type=float, default=15.0)
    ap.add_argument("--drive-mv", type=float, default=30.0)
    ap.add_argument("--brain-seed", type=int, default=1)
    ap.add_argument("--v-jitter", type=float, default=6.9)
    ap.add_argument("--noise-base", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=0, help="first N states (bench)")
    ap.add_argument("--only", default="", help="comma-separated setting labels")
    ap.add_argument("--extra", default="", help="label:apl,drive,vth,norm to append")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    z = np.load(OUT_DIR / "sample.npz", allow_pickle=True)
    relay = z["relay"]
    y_hand = z["y_hand"]
    if a.limit:
        relay, y_hand = relay[:a.limit], y_hand[:a.limit]
    g = C.load(a.threshold)
    pops = populations(g)
    types, sizes, prov = assign(g)
    gm = GlomerularMap(g, types, drive_mv=a.drive_mv)

    settings = grid_settings()
    if a.extra:
        lbl, rest = a.extra.split(":")
        apl, drive, vth, norm = rest.split(",")
        settings = [dict(label=lbl, apl_scale=float(apl), drive=drive,
                         kc_vth_offset_mv=float(vth), kc_input_norm=float(norm))]
    if a.only:
        want = set(a.only.split(","))
        settings = [s for s in settings if s["label"] in want]

    d = load_grid()
    d["meta"] = dict(
        n_states=int(len(relay)), window_ms=a.window, split_ms=a.split,
        drive_mv=a.drive_mv, poisson_hz=POISSON_HZ,
        poisson_kick_mv=POISSON_KICK_MV, brain_seed=a.brain_seed,
        v_jitter_mv=a.v_jitter, noise_base=a.noise_base,
        encoding="glomerular: 32 relay bits -> 32 ORN types, 283 v1 bits dropped",
        assignment=[dict(bit=i, glomerulus=types[i], n_orn=sizes[i]) for i in range(32)],
        graph=dict(neurons=g.n, edges_csr=g.m, threshold=g.meta["threshold"]),
        machine=f"{platform.machine()} python {platform.python_version()}",
    )
    save_grid(d)

    for cfg in settings:
        lbl = cfg["label"]
        cpath = OUT_DIR / f"counts_{lbl}.npz"
        if (not a.force and lbl in d["settings"]
                and d["settings"][lbl].get("sim") and cpath.exists()):
            print(f"skip {lbl} (done)")
            continue
        print(f"=== {lbl}  {cfg}", flush=True)
        brain = build_brain(g, cfg, a.brain_seed, a.v_jitter)
        got = simulate(brain, gm, relay, pops, cfg["drive"], a.window, a.split,
                       a.noise_base, lbl)
        m = sim_metrics(got, y_hand, a.window, a.split)
        m["mn9"] = mn9_check(brain, pops, g.n, cfg["drive"], drive_mv=a.drive_mv)
        arrays = {}
        for k in KEEP:
            arrays[k] = got["full"][k]
            arrays[k + "_early"] = got["early"][k]
        np.savez_compressed(cpath, orn_tot=got["orn_tot"], apl_tot=got["apl_tot"],
                            brain_tot=got["brain_tot"], **arrays)
        d = load_grid()
        d["settings"].setdefault(lbl, {})["config"] = cfg
        d["settings"][lbl]["tuning"] = brain.tuning.to_dict()
        d["settings"][lbl]["sim"] = m
        d["settings"][lbl]["counts_file"] = cpath.name
        save_grid(d)
        k = m["KC"]
        print(f"  KC active {k['mean_active_fraction']:.4f} always-on "
              f"{k['always_on_fraction']:.4f} (a/a {k['always_on_over_active']}) "
              f"J between/within {m['KC_jaccard_by_type']['ratio']}  "
              f"MBON {m['MBON']['mean_rate_hz']:.2f} Hz vary "
              f"{m['MBON']['nonzero_variance_cells']}  ALPN CV "
              f"{m['ALPN']['total_count_cv']}  {m['sim_seconds']:.0f}s", flush=True)
    print("wrote", GRID)


if __name__ == "__main__":
    main()
