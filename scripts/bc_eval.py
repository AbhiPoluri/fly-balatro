"""Stage 4: online evaluation with the fly brain in the loop.

Each trained readout plays real episodes: the env's state bits drive the matching
brain condition (stateless -- reset, one window, log1p counts), the readout scores
the 109 actions, illegal ones are set to -inf, and argmax is played. Reference
rows for ``random`` (uniform over the same masked action set) and the teacher
(the cloning target) use the identical seeds.

``--encoding`` selects the input path and must match what the models were trained
on (``auto`` reads it off their exports). Under ``glomerular32`` the brain is the
tuned one from ``outputs/mb2/tuned_config.json`` and only the 32-bit relay block
reaches it, so the brain-free rows are scored on those 32 bits too; the env still
encodes all 315, which is what ``hand_evidence`` below reads.

Under ``--feature-version 2`` every row also reports what the fly actually
*played*: the histogram of poker hand types it put down, the fraction of plays
where the selection was the best available subset, and the fraction of
select_card actions that landed on a best-subset slot. Those come straight out of
the relay bits the brain was driven with (offsets in
``flybalatro.features_v2``), so they measure the same thing the fly was told,
not a second opinion computed after the fact.

Inference is plain numpy from the ``outputs/bc/models/*.npz`` exports, so the
spawned workers never import torch.

Two things make this affordable. Readouts that share a wiring condition are
evaluated in the *same* worker over the same episode chunk, so they share one
Brain and one ``packed bits -> log1p counts`` cache; and a policy that gets stuck
in a select/deselect loop revisits identical states, which the cache then serves
for free.

Run:
    source .venv/bin/activate
    nohup python scripts/bc_eval.py --episodes 400 --workers 6 \
        > outputs/bc/eval.log 2>&1 &
    python scripts/bc_eval.py --bc-dir outputs/bc2 --teacher v2 --episodes 400 \
        --workers 4
    python scripts/bc_eval.py --bc-dir outputs/bc3 --teacher v2 --episodes 400 \
        --workers 4 --encoding glomerular32 --cache-cap 6000

Writes <bc-dir>/eval.json incrementally (one group at a time, resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_BC = ROOT / "outputs" / "bc"
BC = DEFAULT_BC
MODELS = BC / "models"
NEG = -1e30
SKIP_BLIND_INDEX = 79
PLAY_INDEX = 70
DISCARD_INDEX = 71
N_SELECT_ACTIONS = 24

# readout name -> (wiring condition or None, populations used)
READOUT_SOURCE = {
    "real_alpn_kc_dn": ("real", ("alpn", "kc", "dn")),
    "real_alpn": ("real", ("alpn",)),
    "real_kc": ("real", ("kc",)),
    "real_mbon": ("real", ("mbon",)),
    "real_dn": ("real", ("dn",)),
    "shuf_alpn_kc_dn": ("shuffled", ("alpn", "kc", "dn")),
    "shuf_alpn": ("shuffled", ("alpn",)),
    "shuf_kc": ("shuffled", ("kc",)),
    "shuf_mbon": ("shuffled", ("mbon",)),
    "shuf_dn": ("shuffled", ("dn",)),
    "raw_bits": (None, None),
    "rand_proj": (None, None),
    "rand_proj_matched": (None, None),
}
# Cheap first: the two references and the brain-free controls finish in seconds,
# so an interrupted run still holds the rows the comparison needs most.
GROUP_ORDER = ("references", "nobrain", "real", "shuffled")


def _detect_feature_version(bc: Path, models: Path) -> int:
    """Version stamped on the exports, else on states.npz, else 1."""
    for path in sorted(models.glob("*_linear.npz")) + sorted(models.glob("*_mlp.npz")):
        z = np.load(path, allow_pickle=False)
        if "feature_version" in z.files:
            return int(z["feature_version"])
    states = bc / "states.npz"
    if states.exists():
        z = np.load(states, allow_pickle=False)
        if "feature_version" in z.files:
            return int(z["feature_version"])
    return 1


def _detect_encoding(bc: Path, models: Path) -> str:
    """Encoding stamped on the exports, else on feats_*.npz, else featuremap."""
    from flybalatro import glomerular as G

    for path in sorted(models.glob("*_linear.npz")) + sorted(models.glob("*_mlp.npz")):
        z = np.load(path, allow_pickle=False)
        if "encoding" in z.files:
            return str(z["encoding"])
    for cond in ("real", "shuffled"):
        path = bc / f"feats_{cond}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            if "meta" in z.files:
                return str(json.loads(str(z["meta"])).get("encoding",
                                                          G.ENCODING_FEATUREMAP))
    return G.ENCODING_FEATUREMAP


class Readout:
    """Numpy forward pass of one exported linear / MLP readout."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.kind = str(z["kind"])
        self.mean = z["mean"].astype(np.float32)
        self.std = z["std"].astype(np.float32)
        self.keep = z["keep"].astype(bool)
        if self.kind == "linear":
            self.W = z["W"].astype(np.float32)
            self.b = z["b"].astype(np.float32)
        else:
            self.W1 = z["W1"].astype(np.float32); self.b1 = z["b1"].astype(np.float32)
            self.W2 = z["W2"].astype(np.float32); self.b2 = z["b2"].astype(np.float32)

    def logits(self, x: np.ndarray) -> np.ndarray:
        z = ((x.astype(np.float32) - self.mean) / self.std)[self.keep]
        if self.kind == "linear":
            return self.W @ z + self.b
        h = self.W1 @ z + self.b1
        np.maximum(h, 0.0, out=h)
        return self.W2 @ h + self.b2

    def act(self, x: np.ndarray, mask: np.ndarray) -> int:
        lg = self.logits(x)
        lg = np.where(mask > 0, lg, NEG)
        return int(np.argmax(lg))


# ---- worker ---------------------------------------------------------------
_W: dict = {}


def _init(wiring, readouts, window_ms, n_features, brain_seed, shuffle_seed,
          fm_seed, npf, threshold, ante_end, max_steps, cache_cap,
          feature_version, teacher, models_dir, reference, encoding):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"
    for p in (str(ROOT), str(ROOT / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from flybalatro.env import BalatroEnv
    from flybalatro.features_v2 import encoder_for_version

    from flybalatro import glomerular as G

    _W["models"] = Path(models_dir)
    _W["version"] = int(feature_version)
    _W["encoding"] = str(encoding)
    # Bits the readouts and the brain actually see: all of them under the v1/v2
    # feature map, only the 32-bit relay block under glomerular32.
    _W["n_input"] = G.n_input_bits(str(encoding), int(n_features))
    # Spawned workers re-import this module, so the reference row's name has to
    # be handed over explicitly rather than read off the parent's globals.
    _W["references"] = {"random", str(reference)}
    _W["env"] = BalatroEnv(ante_end=ante_end, max_steps=max_steps, mask_noop_actions=True,
                           encoder=encoder_for_version(feature_version),
                           n_features=n_features)
    if teacher == "v2":
        from baseline_heuristic_v2 import HandAwarePolicy

        _W["teacher"] = HandAwarePolicy()
    else:
        from baseline_heuristic import HeuristicPolicy

        _W["teacher"] = HeuristicPolicy()
    _W["cache"] = {}
    _W["cache_cap"] = int(cache_cap)
    _W["hits"] = 0
    _W["misses"] = 0
    _W["readouts"] = {}
    _W["proj"] = {}
    for name, kind in readouts:
        if name in _W["references"]:
            continue
        _W["readouts"][(name, kind)] = Readout(_W["models"] / f"{name}_{kind}.npz")
        if name.startswith("rand_proj"):
            _W["proj"][name] = np.load(
                _W["models"] / f"{name}_projection.npz")["W"].astype(np.float32)

    if wiring is None:
        _W["brain"] = None
        return
    from flybalatro import connectome as C
    from flybalatro.brain import Brain
    from flybalatro.encode import feature_map_for
    g = C.load(threshold)
    if encoding == G.ENCODING_GLOM32:
        spec = G.load_spec()
        _W["fm"] = G.map_for(g, spec)
        brain = G.brain_for(g, wiring, spec, shuffle_seed=shuffle_seed,
                            brain_seed=brain_seed)
        window_ms = spec.window_ms
    else:
        _W["fm"] = feature_map_for(g, feature_version, seed=fm_seed,
                                   neurons_per_feature=npf)
        base = Brain(g, seed=brain_seed)
        brain = base if wiring == "real" else base.shuffled(shuffle_seed)
        if wiring != "real":
            del base
    if _W["fm"].n_features != _W["n_input"]:
        raise RuntimeError(
            f"{encoding} map wants {_W['fm'].n_features} features, the encoder "
            f"gives {n_features} of which {_W['n_input']} are sent to the brain"
        )
    brain.warmup()
    _W["idx"], _W["slices"] = G.population_indices(g, encoding)
    _W["brain"] = brain
    _W["window"] = float(window_ms)


def _input_bits(bits: np.ndarray) -> np.ndarray:
    """The slice of the encoded state this run feeds to readouts and the brain."""
    n = _W["n_input"]
    return bits if len(bits) == n else bits[:n]


def _brain_feat(bits: np.ndarray) -> np.ndarray:
    key = np.packbits(bits.astype(np.uint8)).tobytes()
    c = _W["cache"]
    v = c.get(key)
    if v is not None:
        _W["hits"] += 1
        return v
    _W["misses"] += 1
    brain = _W["brain"]
    brain.reset()
    counts, _ = brain.step(_W["fm"].drive(bits), _W["window"])
    v = np.log1p(counts[_W["idx"]].astype(np.float32)).astype(np.float16)
    if len(c) < _W["cache_cap"]:
        c[key] = v
    return v


def _features_for(name, bits):
    x = _input_bits(bits)
    if name == "raw_bits":
        return x.astype(np.float32)
    if name.startswith("rand_proj"):
        h = x.astype(np.float32) @ _W["proj"][name]
        np.maximum(h, 0.0, out=h)
        return h
    full = _brain_feat(x)
    pops = READOUT_SOURCE[name][1]
    if len(pops) == 1:
        return full[_W["slices"][pops[0]]]
    return np.concatenate([full[_W["slices"][p]] for p in pops])


def _play_evidence(bits, action, ev):
    """Accumulate "did it play a real hand type" counters from the relay bits.

    Only meaningful for feature version 2, whose first 32 bits *are* the hand
    analysis the brain was driven with (computed outside the brain, in
    ``flybalatro.hands``). Reading them back here measures what the fly was told
    and then did, with no second analysis of the board.
    """
    from flybalatro import features_v2 as F2

    sel = bits[F2.OFF_SELECTED_TYPE:F2.OFF_SELECTED_TYPE + F2.N_SELECTED_TYPE]
    if sel.sum() < 1:  # block inactive: not inside a blind
        return
    if action == PLAY_INDEX:
        ev["plays"] += 1
        ev["play_types"][int(np.argmax(sel))] += 1
        if bits[F2.OFF_IS_BEST] > 0:
            ev["plays_best"] += 1
    elif action == DISCARD_INDEX:
        ev["discards"] += 1
        ev["discard_types"][int(np.argmax(sel))] += 1
    elif action < N_SELECT_ACTIONS:
        ev["selects"] += 1
        if action < F2.N_BEST_MASK and bits[F2.OFF_BEST_MASK + action] > 0:
            ev["selects_on_best"] += 1


def _episode(job):
    """job = (name, kind, seed, rng_seed) -> per-episode record."""
    name, kind, seed, rng_seed = job
    env = _W["env"]
    features, mask = env.reset(seed=seed)
    hist = np.zeros(109, np.int32)
    rng = np.random.default_rng(rng_seed) if name == "random" else None
    ro = _W["readouts"].get((name, kind))
    version = _W["version"]
    ev = dict(plays=0, plays_best=0, selects=0, selects_on_best=0, discards=0,
              play_types=np.zeros(10, np.int32), discard_types=np.zeros(10, np.int32))
    h0, m0 = _W["hits"], _W["misses"]
    t0 = time.perf_counter()
    while not env.done:
        if name == "random":
            legal = np.flatnonzero(mask)
            action = int(legal[rng.integers(len(legal))])
        elif name in _W["references"]:
            action = _W["teacher"](env, mask)
        else:
            action = ro.act(_features_for(name, features), mask)
        hist[action] += 1
        if version == 2:
            _play_evidence(features, action, ev)
        features, mask, _r, _d, info = env.step(action)
    return dict(name=name, kind=kind, seed=int(seed), win=bool(env.is_win),
                chips=int(env.chips_total), steps=int(env.steps),
                truncated=bool(info["truncated"]), rounds=int(info["round"]),
                score=int(info["score"]), hist=hist,
                secs=time.perf_counter() - t0,
                hits=_W["hits"] - h0, misses=_W["misses"] - m0, ev=ev)


# ---- driver ---------------------------------------------------------------
def _summarise_evidence(recs) -> dict:
    from flybalatro import hands

    labels = list(hands.HAND_TYPE_NAMES) + ["none"]
    plays = sum(r["ev"]["plays"] for r in recs)
    selects = sum(r["ev"]["selects"] for r in recs)
    discards = sum(r["ev"]["discards"] for r in recs)
    play_types = np.sum([r["ev"]["play_types"] for r in recs], axis=0)
    discard_types = np.sum([r["ev"]["discard_types"] for r in recs], axis=0)
    return dict(
        plays=int(plays),
        discards=int(discards),
        selects=int(selects),
        plays_that_were_the_best_subset=(
            sum(r["ev"]["plays_best"] for r in recs) / plays if plays else None),
        selects_on_a_best_subset_slot=(
            sum(r["ev"]["selects_on_best"] for r in recs) / selects if selects else None),
        played_hand_types={labels[i]: int(play_types[i])
                           for i in range(len(labels)) if play_types[i] > 0},
        played_hand_type_fractions={labels[i]: round(float(play_types[i] / plays), 4)
                                    for i in range(len(labels)) if play_types[i] > 0}
        if plays else {},
        discarded_hand_types={labels[i]: int(discard_types[i])
                              for i in range(len(labels)) if discard_types[i] > 0},
    )


def summarise(recs) -> dict:
    chips = [r["chips"] for r in recs]
    steps = [r["steps"] for r in recs]
    hist = np.sum([r["hist"] for r in recs], axis=0)
    total = int(hist.sum())
    top = np.argsort(hist)[::-1][:10]
    from flybalatro.env import ACTION_NAMES
    evidence = (_summarise_evidence(recs)
                if recs and isinstance(recs[0].get("ev"), dict) else None)
    return dict(
        n_episodes=len(recs),
        clear_rate=sum(r["win"] for r in recs) / len(recs),
        wins=int(sum(r["win"] for r in recs)),
        chips_mean=float(statistics.fmean(chips)),
        chips_median=float(statistics.median(chips)),
        chips_max=int(max(chips)),
        episode_len_mean=float(statistics.fmean(steps)),
        episode_len_median=float(statistics.median(steps)),
        truncated_fraction=sum(r["truncated"] for r in recs) / len(recs),
        rounds_reached_mean=float(statistics.fmean([r["rounds"] for r in recs])),
        skip_blind_fraction=float(hist[SKIP_BLIND_INDEX] / total) if total else 0.0,
        total_steps=total,
        action_histogram_top10={ACTION_NAMES[i]: int(hist[i]) for i in top if hist[i] > 0},
        n_distinct_actions_used=int((hist > 0).sum()),
        wall_seconds=round(sum(r["secs"] for r in recs), 1),
        cache_hit_rate=(float(sum(r["hits"] for r in recs)
                              / max(1, sum(r["hits"] + r["misses"] for r in recs)))
                        if recs and recs[0].get("misses") is not None else None),
        hand_evidence=evidence,
    )


def run_group(group, readouts, seeds, a, out: dict, outp: Path,
              feature_version: int, n_features: int, encoding: str) -> None:
    import multiprocessing as mp
    todo = [(n, k) for (n, k) in readouts if f"{n}/{k}" not in out]
    if not todo:
        print(f"[{group}] all rows present; skipping", flush=True)
        return
    wiring = {"real": "real", "shuffled": "shuffled"}.get(group)
    jobs = [(n, k, int(s), int(s) * 7919 + 13) for (n, k) in todo for s in seeds]
    init_args = (wiring, todo, a.window, n_features, a.brain_seed, a.shuffle_seed,
                 a.fm_seed, a.neurons_per_feature, a.threshold, a.ante_end, a.max_steps,
                 a.cache_cap, feature_version, a.teacher, str(MODELS), a.reference,
                 encoding)
    print(f"[{group}] {len(todo)} readouts x {len(seeds)} episodes = {len(jobs)} jobs, "
          f"wiring={wiring}", flush=True)
    t0 = time.perf_counter()
    by: dict = {}
    ctx = mp.get_context("spawn")
    nw = 1 if a.workers <= 1 else a.workers
    with ctx.Pool(nw, initializer=_init, initargs=init_args) as pool:
        for i, rec in enumerate(pool.imap_unordered(_episode, jobs, chunksize=2), 1):
            key = (rec["name"], rec["kind"])
            by.setdefault(key, []).append(rec)
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                print(f"[{group}] {i}/{len(jobs)} episodes  {el/60:.1f} min  "
                      f"ETA {(len(jobs)-i)*el/i/60:.1f} min", flush=True)
            # Finish and persist each readout the moment its last episode lands,
            # so an interrupted group keeps the rows it already completed.
            if len(by[key]) == len(seeds):
                n, k = key
                out[f"{n}/{k}"] = summarise(by[key])
                sm = out[f"{n}/{k}"]
                print(f"  {n}/{k}: clear {sm['clear_rate']:.3f}  "
                      f"chips {sm['chips_mean']:.0f}  len {sm['episode_len_mean']:.1f}  "
                      f"skip {sm['skip_blind_fraction']:.4f}  "
                      f"trunc {sm['truncated_fraction']:.3f}  "
                      f"cache {sm['cache_hit_rate']:.3f}", flush=True)
                outp.write_text(json.dumps(out, indent=2, default=float) + "\n")
                del by[key]
    outp.write_text(json.dumps(out, indent=2, default=float) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--window", type=float, default=50.0)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--ante-end", type=int, default=1)
    ap.add_argument("--cache-cap", type=int, default=2_000)
    ap.add_argument("--brain-seed", type=int, default=1)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--fm-seed", type=int, default=0)
    ap.add_argument("--neurons-per-feature", type=int, default=10)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--groups", default=",".join(GROUP_ORDER))
    ap.add_argument("--bc-dir", default=str(DEFAULT_BC))
    ap.add_argument("--teacher", default="v1", choices=("v1", "v2"))
    ap.add_argument("--feature-version", type=int, default=0,
                    help="0 = take it from the trained models / states.npz")
    ap.add_argument("--encoding", default="auto",
                    choices=("auto", "featuremap", "glomerular32"),
                    help="auto = take it from the trained models / feats_*.npz")
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    global BC, MODELS
    BC = Path(a.bc_dir)
    MODELS = BC / "models"
    if not a.out:
        a.out = str(BC / "eval.json")

    feature_version = a.feature_version or _detect_feature_version(BC, MODELS)
    encoding = _detect_encoding(BC, MODELS) if a.encoding == "auto" else a.encoding
    from flybalatro import glomerular as G
    from flybalatro.features_v2 import n_features_for_version
    n_features = n_features_for_version(feature_version)
    n_input = G.n_input_bits(encoding, n_features)
    if encoding == G.ENCODING_GLOM32:
        spec = G.load_spec()
        a.window = spec.window_ms
        a.brain_seed = spec.brain_seed
        print(f"glomerular spec {spec.label} from {spec.source}: "
              f"tuning={spec.tuning.to_dict()} tonic {spec.drive_mv:g} mV", flush=True)
    reference = "heuristic" if a.teacher == "v1" else "teacher_v2"
    a.reference = reference
    print(f"feature_version={feature_version} n_features={n_features} "
          f"encoding={encoding} bits_to_brain={n_input} "
          f"teacher={a.teacher} bc_dir={BC}", flush=True)

    outp = Path(a.out)
    out = json.loads(outp.read_text()) if (outp.exists() and not a.force) else {}
    seeds = list(range(a.seed0, a.seed0 + a.episodes))

    trained = {p.name[: -len(".npz")] for p in MODELS.glob("*_linear.npz")}
    trained |= {p.name[: -len(".npz")] for p in MODELS.glob("*_mlp.npz")}
    have = set()
    for stem in trained:
        for kind in ("linear", "mlp"):
            if stem.endswith("_" + kind):
                have.add((stem[: -(len(kind) + 1)], kind))
    groups = {
        "references": [("random", "-"), (reference, "-")],
        "nobrain": sorted([(n, k) for (n, k) in have if READOUT_SOURCE.get(n, (0,))[0] is None]),
        "real": sorted([(n, k) for (n, k) in have if READOUT_SOURCE.get(n, (0,))[0] == "real"]),
        "shuffled": sorted([(n, k) for (n, k) in have if READOUT_SOURCE.get(n, (0,))[0] == "shuffled"]),
    }
    out["_config"] = dict(episodes=a.episodes, seed0=a.seed0, window_ms=a.window,
                          max_steps=a.max_steps, ante_end=a.ante_end,
                          mask_noop_actions=True, workers=a.workers,
                          cache_cap=a.cache_cap, brain_seed=a.brain_seed,
                          shuffle_seed=a.shuffle_seed, fm_seed=a.fm_seed,
                          feature_version=feature_version, encoding=encoding,
                          n_bits_to_brain=n_input, teacher=a.teacher,
                          reference_row=reference,
                          glomerular_spec=(G.load_spec().describe()
                                           if encoding == G.ENCODING_GLOM32 else None))
    for g in a.groups.split(","):
        g = g.strip()
        if not groups.get(g):
            print(f"[{g}] nothing to run", flush=True)
            continue
        run_group(g, groups[g], seeds, a, out, outp, feature_version, n_features,
                  encoding)
    outp.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
