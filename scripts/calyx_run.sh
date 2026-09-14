#!/usr/bin/env bash
# Drives the whole calyx control end to end, resumable: every stage skips work
# that is already on disk, so re-running after an interruption costs nothing.
#
#   bash scripts/calyx_run.sh > outputs/calyx/run.log 2>&1 &
#
# Assumes scripts/calyx_homeo.py and the raw-variant scripts/calyx_feats.py run
# are already running (or done); it waits for their outputs.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
O=outputs/calyx
CONDS=real,rw1,rw2,rw3

say() { echo "=== $(date '+%H:%M:%S') $*"; }
wait_for() { while [ ! -f "$1" ]; do sleep 30; done; say "have $1"; }

say "waiting for the raw-variant features"
for c in real rw1 rw2 rw3; do wait_for "$O/raw/$c/feats_real.npz"; done

say "probe raw"
python -m scripts.calyx_probe --variant raw --conditions $CONDS
say "train raw"
python -m scripts.calyx_train --variant raw --conditions $CONDS --readouts kc,dn

say "waiting for the per-KC homeostasis"
for c in real rw1 rw2 rw3; do wait_for "$O/tuned_$c.json"; done

say "features homeo"
python -m scripts.calyx_feats --variant homeo --conditions $CONDS --workers 2
say "probe homeo"
python -m scripts.calyx_probe --variant homeo --conditions $CONDS
say "train homeo"
python -m scripts.calyx_train --variant homeo --conditions $CONDS --readouts kc,dn

say "live play, homeo"
python -m scripts.calyx_eval --variant homeo --conditions $CONDS --readouts kc,dn --workers 2
say "live play, raw"
python -m scripts.calyx_eval --variant raw --conditions $CONDS --readouts kc,dn --workers 2

for v in homeo raw; do
  say "paired stats $v"
  python -m scripts.calyx_stats --variant $v --conditions $CONDS --readouts kc,dn
  python -m scripts.calyx_tables --variant $v --conditions $CONDS --readouts kc,dn \
      > "$O/tables_$v.log" 2>&1
done
say "CALYX RUN COMPLETE"
