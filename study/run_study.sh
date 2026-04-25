#!/bin/bash
# =============================================================================
# Master script for the Q-HAS performance evaluation study.
#
# Phases:
#   1. DNS sweep at multiple Re (N=256)
#   2. Hard patch identification (L2 error criterion)
#   3. Hamiltonian coefficient analysis + threshold stability
#   4. Exact diagonalization on hard patches
#   5. QAOA evaluation on promising patches
#
# Usage:
#   ./study/run_study.sh              # run all phases
#   ./study/run_study.sh 3 4 5        # run specific phases
#   ./study/run_study.sh --quick      # quick run (Re=400 only, dim=2)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# defaults
RE_ARGS=""
DIM_ARGS=""
PHASES=""
QUICK=0

# parse args
for arg in "$@"; do
    case "$arg" in
        --quick)
            QUICK=1
            ;;
        [1-5])
            PHASES="$PHASES $arg"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--quick] [1] [2] [3] [4] [5]"
            exit 1
            ;;
    esac
done

# default: all phases
if [ -z "$PHASES" ]; then
    PHASES="1 2 3 4 5"
fi

if [ "$QUICK" -eq 1 ]; then
    RE_ARGS="--re 400"
    DIM_ARGS="--dim 2"
    echo "=== QUICK MODE: Re=400 only, dim=2 ==="
fi

echo "============================================"
echo "  Q-HAS Performance Evaluation Study"
echo "  Phases: $PHASES"
echo "============================================"
echo ""

run_phase() {
    local phase=$1
    local script=$2
    shift 2
    echo ""
    echo "============================================"
    echo "  PHASE $phase"
    echo "============================================"
    echo ""
    python "$script" "$@"
    echo ""
    echo "  Phase $phase finished."
    echo ""
}

for phase in $PHASES; do
    case "$phase" in
        1)
            run_phase 1 study/phase1_dns_sweep.py $RE_ARGS
            ;;
        2)
            run_phase 2 study/phase2_hard_patches.py $RE_ARGS $DIM_ARGS
            ;;
        3)
            run_phase 3 study/phase3_coefficients.py $RE_ARGS $DIM_ARGS
            ;;
        4)
            run_phase 4 study/phase4_exact_diag.py $RE_ARGS $DIM_ARGS
            ;;
        5)
            run_phase 5 study/phase5_qaoa_eval.py $RE_ARGS $DIM_ARGS
            ;;
    esac
done

echo "============================================"
echo "  ALL PHASES COMPLETE"
echo "============================================"
echo ""
echo "Results saved in: study/results/"
ls -lh study/results/*.npz 2>/dev/null || echo "  (no results yet)"
