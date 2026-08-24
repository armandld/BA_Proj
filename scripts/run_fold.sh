#!/usr/bin/env bash
# Run one confirmatory Level-3 LOSO fold.
set -euo pipefail

FOLD="${1:?usage: run_fold.sh <fold> [qaoa_trials] [classical_trials]}"
QAOA_TRIALS="${2:-170}"
CLASSICAL_TRIALS="${3:-85}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
export QHAS_RESULTS_DIR="${QHAS_RESULTS_DIR:-$ROOT/results/campaigns/current}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 2
fi
if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=normal)" ]; then
    echo "Refusing a confirmatory fold from a dirty worktree." >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"
"$PYTHON_BIN" study/closed_loop/closed_loop_campaign.py \
    --folds "$FOLD" --n-trials "$QAOA_TRIALS" \
    --n-trials-classical "$CLASSICAL_TRIALS"
"$PYTHON_BIN" study/closed_loop/closed_loop_budget_matched.py \
    --fold "$FOLD" --max-iter 8
"$PYTHON_BIN" study/closed_loop/closed_loop_run_variance.py \
    --fold "$FOLD" --repeats 3 --seed 0 --qaoa-seed 0
