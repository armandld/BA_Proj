#!/usr/bin/env bash
# Generate the registered DNS panel in parallel on one machine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
MAX_PARALLEL="${1:-4}"
N="${2:-256}"
export QHAS_RESULTS_DIR="${QHAS_RESULTS_DIR:-$ROOT/results/campaigns/current}"
SCENARIOS=(
    orszag_tang harris_tearing kelvin_helmholtz mhd_rotor
    lamb_oseen island_coalescence double_tearing magnetic_twist
)
RE_VALUES=(400 800 1200 1600)
SEEDS=(0 1 2 3 4)

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "max_parallel must be an integer >= 1" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=normal)" ]]; then
    echo "Refusing a DNS campaign from a dirty worktree." >&2
    exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$QHAS_RESULTS_DIR/logs/dns/$STAMP"
mkdir -p "$LOG_DIR"
pids=()
names=()
status=0

wait_oldest() {
    local pid="${pids[0]}"
    local name="${names[0]}"
    if ! wait "$pid"; then
        echo "$name failed; see $LOG_DIR/$name.log" >&2
        status=1
    fi
    pids=("${pids[@]:1}")
    names=("${names[@]:1}")
}

for scenario in "${SCENARIOS[@]}"; do
    for re in "${RE_VALUES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            while [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; do
                wait_oldest
            done
            name="${scenario}_Re${re}_seed${seed}"
            "$PYTHON_BIN" "$ROOT/study/pipeline/dns_sweep.py" \
                --scenario "$scenario" --re "$re" --phys-seed "$seed" \
                --N "$N" --labels-dim 4 \
                >"$LOG_DIR/$name.log" 2>&1 &
            pids+=("$!")
            names+=("$name")
            echo "started $name"
        done
    done
done
while [[ ${#pids[@]} -gt 0 ]]; do
    wait_oldest
done
if [[ $status -ne 0 ]]; then
    exit "$status"
fi

cd "$ROOT"
"$PYTHON_BIN" study/pipeline/dns_validation.py --N "$N"
"$PYTHON_BIN" study/pipeline/data_catalog.py --N "$N" --dim 4
echo "DNS campaign complete."
