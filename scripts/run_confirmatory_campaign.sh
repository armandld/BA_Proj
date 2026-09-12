#!/usr/bin/env bash
# Run the eight independent confirmatory folds on one multi-core machine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
MAX_PARALLEL="${1:-4}"
QAOA_TRIALS="${2:-170}"
CLASSICAL_TRIALS="${3:-85}"
export QHAS_RESULTS_DIR="${QHAS_RESULTS_DIR:-$ROOT/results/campaigns/current}"
FOLDS=(kh vortex tearing coalescence double_tearing magnetic_twist ot rotor)

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "max_parallel must be an integer >= 1" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=normal)" ]]; then
    echo "Refusing a confirmatory campaign from a dirty worktree." >&2
    exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$QHAS_RESULTS_DIR/logs/closed_loop/$STAMP"
mkdir -p "$LOG_DIR"

pids=()
names=()
status=0
wait_oldest() {
    local pid="${pids[0]}"
    local fold="${names[0]}"
    if ! wait "$pid"; then
        echo "fold $fold failed; see $LOG_DIR/$fold.log" >&2
        status=1
    fi
    pids=("${pids[@]:1}")
    names=("${names[@]:1}")
}

for fold in "${FOLDS[@]}"; do
    while [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; do
        wait_oldest
    done
    bash "$ROOT/scripts/run_fold.sh" "$fold" \
        "$QAOA_TRIALS" "$CLASSICAL_TRIALS" \
        >"$LOG_DIR/$fold.log" 2>&1 &
    pids+=("$!")
    names+=("$fold")
    echo "started $fold -> $LOG_DIR/$fold.log"
done
while [[ ${#pids[@]} -gt 0 ]]; do
    wait_oldest
done
if [[ $status -ne 0 ]]; then
    exit "$status"
fi

cd "$ROOT"
"$PYTHON_BIN" study/closed_loop/closed_loop_headline_counts.py
"$PYTHON_BIN" study/closed_loop/closed_loop_fold_synthesis.py
echo "Confirmatory campaign complete."
