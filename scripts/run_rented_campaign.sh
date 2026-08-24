#!/usr/bin/env bash
# Run phase 1 on one rented multi-core machine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

DEFAULT_WORKERS="$($PYTHON_BIN -c 'import os; print(os.cpu_count() or 1)')"
N_WORKERS="${1:-$DEFAULT_WORKERS}"
TARGET_TRIALS="${2:-600}"
BASE_SEED="${3:-0}"

for value in "$N_WORKERS" "$TARGET_TRIALS"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "workers and target_trials must be integers >= 1" >&2
        exit 2
    fi
done
if ! [[ "$BASE_SEED" =~ ^[0-9]+$ ]]; then
    echo "base_seed must be an integer >= 0" >&2
    exit 2
fi
if [ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=normal)" ]; then
    echo "Refusing to run from a dirty worktree; commit the campaign code first." >&2
    exit 1
fi

export QHAS_CAMPAIGN_DIR="${QHAS_CAMPAIGN_DIR:-$ROOT_DIR/results/hyperparams/reoptimisation}"
export QHAS_JOURNAL_DIR="${QHAS_JOURNAL_DIR:-$QHAS_CAMPAIGN_DIR/journal}"
export CAMPAIGN_RESULT_PATH="${CAMPAIGN_RESULT_PATH:-$QHAS_CAMPAIGN_DIR/candidate_phase1.json}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$QHAS_CAMPAIGN_DIR/logs/$RUN_ID"
mkdir -p "$QHAS_JOURNAL_DIR" "$LOG_DIR"

echo "Campaign directory : $QHAS_CAMPAIGN_DIR"
echo "Workers            : $N_WORKERS"
echo "Global trial target: $TARGET_TRIALS"
echo "Worker logs        : $LOG_DIR"

"$PYTHON_BIN" "$ROOT_DIR/study/common/preflight_coefficients.py"
"$PYTHON_BIN" "$ROOT_DIR/src/train_hyperparams.py" --print-space
"$PYTHON_BIN" "$ROOT_DIR/src/train_hyperparams.py" \
    --phase 1 --n-trials "$TARGET_TRIALS" --prepare-only

# Numerical libraries stay single-threaded; parallelism is managed here.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export QHAS_PREFLIGHT_DONE=1

pids=()
stop_workers() {
    trap - INT TERM
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait || true
    exit 130
}
trap stop_workers INT TERM

for ((worker=0; worker<N_WORKERS; worker++)); do
    seed=$((BASE_SEED + worker))
    bash "$ROOT_DIR/scripts/run_reoptimisation.sh" \
        "$TARGET_TRIALS" "$seed" \
        >"$LOG_DIR/worker_${worker}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
trap - INT TERM
if [ "$status" -ne 0 ]; then
    echo "At least one worker failed; inspect $LOG_DIR" >&2
    exit "$status"
fi

"$PYTHON_BIN" "$ROOT_DIR/src/train_hyperparams.py" \
    --phase 1 --n-trials "$TARGET_TRIALS" --finalize-only \
    --result-path "$CAMPAIGN_RESULT_PATH"
"$PYTHON_BIN" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "complete", d["trial_states"]' \
    "$CAMPAIGN_RESULT_PATH"

echo "Campaign complete: $CAMPAIGN_RESULT_PATH"
