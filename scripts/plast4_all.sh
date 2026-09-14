#!/usr/bin/env bash
# Run every plast4 cell that does not already have a json, four at a time.
#
#   bash scripts/plast4_all.sh            # the 15 B/C/D runs
#   bash scripts/plast4_all.sh B_s0 D_s2  # a named subset
#
# Each cell is its own process, so a crash loses one cell and nothing else, and
# `python -m scripts.plast4_run --only <cell>` skips any cell whose json exists.
# Liveness: pgrep -f plast4_
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

CELLS=("$@")
if [ ${#CELLS[@]} -eq 0 ]; then
    CELLS=(B_s0 B_s1 B_s2 C50_s0 C50_s1 C50_s2 C80_s0 C80_s1 C80_s2 \
           C100_s0 C100_s1 C100_s2 D_s0 D_s1 D_s2)
fi

mkdir -p outputs/plast4/logs
printf '%s\n' "${CELLS[@]}" | xargs -P 4 -I{} sh -c \
    'python -m scripts.plast4_run --only "$1" > outputs/plast4/logs/"$1".log 2>&1' _ {}

echo "== all cells done =="
tail -n 2 outputs/plast4/logs/*.log
