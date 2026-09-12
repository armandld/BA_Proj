#!/usr/bin/env bash
# One phase-1 worker. Use run_rented_campaign.sh to start the worker pool.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_TRIALS="${1:-600}"
WORKER_SEED="${2:-0}"

if ! [[ "$TARGET_TRIALS" =~ ^[1-9][0-9]*$ ]]; then
    echo "target_trials must be an integer >= 1" >&2
    exit 2
fi
if ! [[ "$WORKER_SEED" =~ ^[0-9]+$ ]]; then
    echo "worker_seed must be an integer >= 0" >&2
    exit 2
fi

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export QHAS_CAMPAIGN_DIR="${QHAS_CAMPAIGN_DIR:-$ROOT_DIR/results/hyperparams/reoptimisation}"
export QHAS_JOURNAL_DIR="${QHAS_JOURNAL_DIR:-$QHAS_CAMPAIGN_DIR/journal}"
RESULT_PATH="${CAMPAIGN_RESULT_PATH:-$QHAS_CAMPAIGN_DIR/candidate_phase1.json}"
mkdir -p "$QHAS_CAMPAIGN_DIR" "$QHAS_JOURNAL_DIR"

if [ "${QHAS_PREFLIGHT_DONE:-0}" != "1" ]; then
    if [ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=normal)" ]; then
        echo "Refusing to run from a dirty worktree; commit the campaign code first." >&2
        exit 1
    fi
    "$PYTHON_BIN" "$ROOT_DIR/study/common/preflight_coefficients.py"
    "$PYTHON_BIN" "$ROOT_DIR/src/train_hyperparams.py" --print-space
    "$PYTHON_BIN" "$ROOT_DIR/src/train_hyperparams.py" \
        --phase 1 --n-trials "$TARGET_TRIALS" --prepare-only
fi

cd "$ROOT_DIR/src"
WORKER_TRIALS="${WORKER_TRIALS:-$TARGET_TRIALS}" \
    "$PYTHON_BIN" train_hyperparams.py \
        --phase 1 --seed "$WORKER_SEED" --n-trials "$TARGET_TRIALS" \
        --result-path "$RESULT_PATH"
