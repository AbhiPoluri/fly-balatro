#!/bin/bash
# Chains the remaining BC stages as each dependency completes, so the machine is
# never idle. Waits on the log line bc_brain_features prints *after*
# np.savez_compressed returns -- file existence is not enough, the npz is
# streamed and a partial zip makes bc_train raise BadZipFile.
#
# Run the v1 pipeline (283 bits, pair/flush teacher):
#   scripts/bc_pipeline.sh
# Run the v2 pipeline (315 bits with the hand-type relay, hand-aware teacher):
#   BC=outputs/bc2 TEACHER=v2 WORKERS=4 scripts/bc_pipeline.sh
# Run the v3 pipeline (the 32 relay bits only, one whole ORN glomerulus each, on
# the outputs/mb2 tuned brain; same states and same teacher as v2):
#   BC=outputs/bc3 TEACHER=v2 WORKERS=4 ENCODING=glomerular32 scripts/bc_pipeline.sh
#
# bc_brain_features is expected to be running (or finished) against the same
# directory; this script only waits for it, trains and evaluates.
set -eu
# resolve the repo root from this script's own location, so the pipeline runs
# from any checkout. Override PY to use a different interpreter.
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-$ROOT/.venv/bin/python}
BC=${BC:-outputs/bc}
TEACHER=${TEACHER:-v1}
EPISODES=${EPISODES:-1000}
WORKERS=${WORKERS:-4}
EPOCHS=${EPOCHS:-15}
ENCODING=${ENCODING:-featuremap}
CACHE_CAP=${CACHE_CAP:-2000}
# v3 reads the mushroom-body output too (MBON is worth a row only once the Kenyon
# cell code is separable); real_alpn is only trained in the v2 run, where the
# relay block lands on the receptor neurons one synapse upstream of it.
if [ "$ENCODING" = "glomerular32" ]; then
  REAL_CONDS=${REAL_CONDS:-real_alpn_kc_dn,real_kc,real_mbon,real_dn,raw_bits,rand_proj_matched}
elif [ "$TEACHER" = "v2" ]; then
  REAL_CONDS=${REAL_CONDS:-real_alpn_kc_dn,real_alpn,real_dn,raw_bits,rand_proj_matched}
else
  REAL_CONDS=${REAL_CONDS:-real_alpn_kc_dn,real_dn,rand_proj_matched}
fi
SHUF_CONDS=${SHUF_CONDS:-shuf_alpn_kc_dn,shuf_dn}

wait_for () {  # wait_for <log-regex> <label>
  until grep -qE "$1" $BC/brain_features.log; do sleep 20; done
  echo "=== $(date '+%H:%M:%S') $2 complete ==="
}

wait_for '^\[real\] wrote' "feats_real.npz"
$PY scripts/bc_consistency_check.py real --bc-dir "$BC"
$PY scripts/bc_train.py --bc-dir "$BC" --conditions "$REAL_CONDS" --epochs "$EPOCHS" \
  --encoding "$ENCODING"
echo "=== $(date '+%H:%M:%S') real-wiring readouts trained ==="

wait_for '^\[shuffled\] wrote' "feats_shuffled.npz"
$PY scripts/bc_consistency_check.py shuffled --bc-dir "$BC"
$PY scripts/bc_train.py --bc-dir "$BC" --conditions "$SHUF_CONDS" --epochs "$EPOCHS" \
  --encoding "$ENCODING"
echo "=== $(date '+%H:%M:%S') shuffled-wiring readouts trained ==="

# eval.json is no longer deleted first: bc_eval skips rows it already has, which
# is what makes an interrupted eval resumable. Delete it by hand for a clean rerun.
$PY scripts/bc_eval.py --bc-dir "$BC" --teacher "$TEACHER" \
  --episodes "$EPISODES" --workers "$WORKERS" --encoding "$ENCODING" \
  --cache-cap "$CACHE_CAP"
echo "=== $(date '+%H:%M:%S') pipeline done ==="
