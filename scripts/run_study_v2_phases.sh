#!/bin/bash
# D-76 : la reorganisation `17d983d` a deplace ET renomme chaque script
# invoque ci-dessous ; ce lanceur portait encore les chemins d'avant et
# mourait sur son PREMIER appel Python (`can't open file ...`, code 2).
# Meme defaut que D-71, qui n'avait couvert que `run_fold.sh` et
# `run_leak_free_campaign.sh`. Chemins verifies fichier par fichier, et
# chaque drapeau CLI passe ici confirme present dans le `--help` de sa
# nouvelle cible.
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
            run_phase 1 study/pipeline/dns_sweep.py $RE_ARGS
            ;;
        2)
            run_phase 2 study/pipeline/hard_patch_labels.py $RE_ARGS $DIM_ARGS
            ;;
        3)
            run_phase 3 study/pipeline/hamiltonian_coefficients.py $RE_ARGS $DIM_ARGS
            ;;
        4)
            run_phase 4 study/pipeline/exact_diagonalisation.py $RE_ARGS $DIM_ARGS
            ;;
        5)
            run_phase 5 study/common/qaoa_inputs.py $RE_ARGS $DIM_ARGS
            ;;
    esac
done

echo "============================================"
echo "  ALL PHASES COMPLETE"
echo "============================================"
echo ""
echo "Results saved in: results/"
ls -lh results/*.npz 2>/dev/null || echo "  (no results yet)"
