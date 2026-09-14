"""Print the imitation-accuracy and online-eval tables from the saved metrics.

    python scripts/bc_summary.py                # outputs/bc
    python scripts/bc_summary.py outputs/bc2    # the v2 run, with the hand table
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "bc"
if not BC.is_absolute():
    BC = ROOT / BC
ORDER = ["real_alpn_kc_dn", "real_alpn", "real_dn", "shuf_alpn_kc_dn", "shuf_alpn",
         "shuf_dn", "raw_bits", "rand_proj", "rand_proj_matched"]

m = json.loads((BC / "train_metrics.json").read_text()) if (BC / "train_metrics.json").exists() else {}
print("=== imitation (held-out, masked top-1) ===")
p = m.get("_priors", {})
print(f"  floor: most-frequent-legal-label {p.get('most_frequent_legal_label_top1', float('nan')):.4f}   "
      f"uniform-random-legal {p.get('uniform_random_legal_top1', float('nan')):.4f}   "
      f"majority {p.get('majority_label_top1', float('nan')):.4f}")
print(f"  {'condition':22s} {'d':>6s} {'eff':>6s} {'linear':>8s} {'mlp':>8s} {'lin_nt':>8s} {'mlp_nt':>8s}")
for c in ORDER:
    L, M_ = m.get(f"{c}/linear"), m.get(f"{c}/mlp")
    if not L and not M_:
        continue
    ref = L or M_
    print(f"  {c:22s} {ref['raw_dims']:>6d} {ref['n_input_dims']:>6d} "
          f"{L['test']['top1']:>8.4f} {M_['test']['top1']:>8.4f} "
          f"{L['test']['top1_nontrivial']:>8.4f} {M_['test']['top1_nontrivial']:>8.4f}")

e = json.loads((BC / "eval.json").read_text()) if (BC / "eval.json").exists() else {}
print("\n=== online eval ===", json.dumps(e.get("_config", {})))
print(f"  {'readout':28s} {'clear':>7s} {'chips':>8s} {'chips_md':>9s} {'len':>6s} {'skip':>7s} {'trunc':>6s} {'cache':>6s}")
rows = [k for k in ("random/-", "heuristic/-", "teacher_v2/-")] + \
       [f"{c}/{k}" for c in ORDER for k in ("linear", "mlp")]
for r in rows:
    v = e.get(r)
    if not v:
        continue
    ch = v.get("cache_hit_rate")
    print(f"  {r:28s} {v['clear_rate']:>7.3f} {v['chips_mean']:>8.1f} {v['chips_median']:>9.1f} "
          f"{v['episode_len_mean']:>6.1f} {v['skip_blind_fraction']:>7.4f} "
          f"{v['truncated_fraction']:>6.3f} {(f'{ch:.2f}' if ch is not None else '-'):>6s}")

# -- what the fly actually put on the table ---------------------------------
if any(isinstance(v, dict) and v.get("hand_evidence") for k, v in e.items() if k != "_config"):
    print("\n=== hand types played (relay bits, feature version 2) ===")
    print(f"  {'readout':28s} {'plays':>6s} {'=best':>7s} {'sel@best':>9s}  top hand types")
    for r in rows:
        v = e.get(r)
        if not v or not v.get("hand_evidence"):
            continue
        h = v["hand_evidence"]
        frac = h.get("played_hand_type_fractions", {})
        top = sorted(frac.items(), key=lambda kv: -kv[1])[:4]
        best = h.get("plays_that_were_the_best_subset")
        sel = h.get("selects_on_a_best_subset_slot")
        print(f"  {r:28s} {h['plays']:>6d} "
              f"{(f'{best:.3f}' if best is not None else '-'):>7s} "
              f"{(f'{sel:.3f}' if sel is not None else '-'):>9s}  "
              + ", ".join(f"{k} {p:.0%}" for k, p in top))
