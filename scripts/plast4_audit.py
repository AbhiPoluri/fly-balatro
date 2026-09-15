"""The two arithmetic audits quoted in ``outputs/plast4/PREREGISTRATION.md``.

Both are re-analyses of the **already published** v3 training logs
(``outputs/plast3/run_omission_separated_s{0,1,2}.json``). No brain, no game, no
new condition: this is log arithmetic, which is why the preregistration is
allowed to quote it and still be a preregistration.

1. **The share inequality.** Confirms there is no off-by-one (``state.plays``
   counts the play about to be made, so ``needed / plays`` demands clearing on
   the last play) and counts the plays where ``reward`` and ``lost`` are both
   true, which the arithmetic says is impossible. Then re-scores every v3
   training play under the ``pace`` rule and reports the flips per bucket.

2. **The trace target.** Groups plays by blind, marks blinds containing a ``lost``
   hand, and sums ``gamma^k`` over their plays per bucket: the weight a terminal
   pulse would have delivered on v3's own trajectory. Off-policy, and the sum
   ignores the clip at 1, so it is an upper bound and a direction, not a
   prediction of the converged ledger.

Run::

    python -m scripts.plast4_audit
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from scripts import plast4_common as C4
from scripts import plast_common as K

V3 = K.ROOT / "outputs" / "plast3"
V3_RUNS = tuple(V3 / f"run_omission_separated_s{s}.json" for s in (0, 1, 2))
BUCKETS = ("lt0.25", "lt0.5", "lt1.0", "ge1.0")


def _blind_key(rec: dict) -> tuple:
    return (int(rec["seed"]), int(rec["round"]))


def blind_tables(records: Sequence[dict]) -> tuple:
    """``required`` and ``p0`` per blind, reconstructed from the first record.

    The first record of a blind has ``score = 0``, so its ``needed`` is the
    blind's requirement and its ``plays_before`` is the blind's play budget.
    """
    req: Dict[tuple, int] = {}
    p0: Dict[tuple, int] = {}
    for r in records:
        k = _blind_key(r)
        if k not in req:
            req[k] = int(r["needed"])
            p0[k] = int(r["plays_before"])
    return req, p0


def audit_reward(records: Sequence[dict]) -> dict:
    """Share vs pace on one v3 run's plays, plus the rewarded-and-lost count."""
    req, p0 = blind_tables(records)
    per_bucket = {b: dict(plays=0, share=0, pace=0, rewarded_but_lost=0)
                  for b in BUCKETS}
    flips = collections.Counter()
    n_plays = 0
    mismatch = 0
    for r in records:
        if r["action"] != K.PLAY:
            continue
        n_plays += 1
        k = _blind_key(r)
        required, budget = req[k], p0[k]
        needed, plays, chips = int(r["needed"]), int(r["plays_before"]), int(r["chips_gained"])
        cleared = bool(r["cleared"])
        s_ok = cleared or C4.share_ok(needed, plays, chips)
        p_ok = cleared or C4.pace_ok(required, required - needed, plays, budget, chips)
        if s_ok != bool(r["reward"]):
            mismatch += 1
        b = str(r["bucket_name"])
        per_bucket[b]["plays"] += 1
        per_bucket[b]["share"] += int(s_ok)
        per_bucket[b]["pace"] += int(p_ok)
        per_bucket[b]["rewarded_but_lost"] += int(s_ok and bool(r["lost"]))
        flips[(s_ok, p_ok)] += 1
    return dict(
        n_plays=n_plays,
        share_reproduces_v3_reward=bool(mismatch == 0),
        n_mismatch=mismatch,
        p0_values=sorted(set(p0.values())),
        share_rewarded_pace_punished=int(flips[(True, False)]),
        share_punished_pace_rewarded=int(flips[(False, True)]),
        n_flips=int(flips[(True, False)] + flips[(False, True)]),
        flip_fraction=round((flips[(True, False)] + flips[(False, True)]) / n_plays, 5)
        if n_plays else 0.0,
        per_bucket=per_bucket,
        rewarded_but_lost=int(sum(v["rewarded_but_lost"] for v in per_bucket.values())),
    )


def audit_trace(records: Sequence[dict],
                gammas: Sequence[float] = C4.TRACE_GAMMAS) -> dict:
    """Per-bucket terminal-pulse weight a trace would have delivered, per gamma."""
    blinds: Dict[tuple, List[dict]] = collections.defaultdict(list)
    for r in records:
        blinds[_blind_key(r)].append(r)
    weight = {f"{g:g}": {b: 0.0 for b in BUCKETS} for g in gammas}
    terminal_bucket = collections.Counter()
    n_lost = 0
    plays_in_lost = 0
    for rs in blinds.values():
        if not any(r["lost"] for r in rs):
            continue
        n_lost += 1
        plays = [r for r in rs if r["action"] == K.PLAY]
        plays_in_lost += len(plays)
        if plays:
            terminal_bucket[str(plays[-1]["bucket_name"])] += 1
        n = len(plays)
        for j, r in enumerate(plays):
            back = n - 1 - j
            for g in gammas:
                weight[f"{g:g}"][str(r["bucket_name"])] += float(g) ** back
    immediate = {b: dict(reward=0, punish=0) for b in BUCKETS}
    for r in records:
        if r["action"] != K.PLAY:
            continue
        b = str(r["bucket_name"])
        immediate[b]["reward" if r["reward"] else "punish"] += 1
    net = {g: {b: immediate[b]["reward"] - immediate[b]["punish"] - weight[g][b]
               for b in BUCKETS} for g in weight}
    return dict(n_blinds=len(blinds), n_lost_blinds=n_lost,
                plays_in_lost_blinds=plays_in_lost,
                terminal_bucket=dict(terminal_bucket),
                immediate=immediate,
                trace_weight={g: {b: round(v, 2) for b, v in d.items()}
                              for g, d in weight.items()},
                net_with_trace={g: {b: round(v, 2) for b, v in d.items()}
                                for g, d in net.items()})


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(C4.OUT_DIR / "audit.json"))
    a = ap.parse_args(argv)

    payload: dict = dict(source=[C4.rel_path(p) for p in V3_RUNS], runs={})
    for p in V3_RUNS:
        recs = json.loads(p.read_text())["train"]["records"]
        name = p.stem.replace("run_", "")
        rw = audit_reward(recs)
        tr = audit_trace(recs)
        payload["runs"][name] = dict(reward=rw, trace=tr)
        print(f"== {name}: {rw['n_plays']} plays, p0 {rw['p0_values']}, "
              f"share reproduces v3 reward: {rw['share_reproduces_v3_reward']}")
        print(f"   rewarded-and-lost: {rw['rewarded_but_lost']}   "
              f"pace flips {rw['n_flips']} ({rw['flip_fraction']:.3%}): "
              f"share->punish {rw['share_rewarded_pace_punished']}, "
              f"punish->share {rw['share_punished_pace_rewarded']}")
        for b in BUCKETS:
            v = rw["per_bucket"][b]
            cells = "  ".join(
                f"g={g}: T {tr['trace_weight'][g][b]:7.1f} net {tr['net_with_trace'][g][b]:+8.1f}"
                for g in tr["trace_weight"])
            print(f"   {b:8s} plays {v['plays']:4d} share {v['share']:4d} "
                  f"pace {v['pace']:4d} | {cells}")

    totals = {b: dict(plays=0, share=0, pace=0) for b in BUCKETS}
    for d in payload["runs"].values():
        for b in BUCKETS:
            for k in ("plays", "share", "pace"):
                totals[b][k] += d["reward"]["per_bucket"][b][k]
    payload["pooled_reward_by_bucket"] = totals
    print("\npooled over the three runs:")
    for b in BUCKETS:
        v = totals[b]
        print(f"   {b:8s} plays {v['plays']:4d} share-rewarded {v['share']:4d} "
              f"pace-rewarded {v['pace']:4d}")
    C4.write_json(Path(a.out), payload)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
