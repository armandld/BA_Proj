#!/bin/bash
# ============================================================
# Reproduce the static and predictive study from a complete DNS panel.
#
# Pre-requis : results/ contient les donnees phases 1-2
# (dns_*.npz + patches_*.npz). La regeneration des DONNEES n'est
# pas incluse (heures de DNS) : utiliser study/pipeline/dns_sweep.py.
#
# Usage :
#   bash scripts/run_study_v3.sh --all          # toute l'analyse
#   bash scripts/run_study_v3.sh --skip-t6      # sans le pilote t6
#   bash scripts/run_study_v3.sh --only t4 t9   # sous-ensemble
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export QHAS_RESULTS_DIR="${QHAS_RESULTS_DIR:-$ROOT_DIR/results/campaigns/current}"
RESULTS_DIR="$QHAS_RESULTS_DIR"
LOG_DIR="$QHAS_RESULTS_DIR/logs/study"
STAMP="$(date +'%Y-%m-%d_%H-%M-%S')"
mkdir -p "$LOG_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

N=256
DIM=4
SKIP_T6=false
ONLY=()
HYPERPARAMS_FILE=""
ALLOW_DIRTY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) shift ;;
        --skip-t6) SKIP_T6=true; shift ;;
        --N) N="$2"; shift 2 ;;
        --dim) DIM="$2"; shift 2 ;;
        --hyperparams-file) HYPERPARAMS_FILE="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=true; shift ;;
        --only) shift; while [[ $# -gt 0 && "$1" != --* ]]; do
                    ONLY+=("$1"); shift; done ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

if ! $ALLOW_DIRTY && [[ -n "$(git -C "$ROOT_DIR" status --porcelain)" ]]; then
    echo "ERROR: refusing a scientific study run from a dirty tree." >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -n "$HYPERPARAMS_FILE" ]]; then
    if [[ ! -f "$HYPERPARAMS_FILE" ]]; then
        echo "ERROR: hyperparameter artifact not found: $HYPERPARAMS_FILE" >&2
        exit 1
    fi
    export QHAS_HYPERPARAMS_PATH="$(cd "$(dirname "$HYPERPARAMS_FILE")" && pwd)/$(basename "$HYPERPARAMS_FILE")"
    PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c \
      'from hyperparams_loader import load_hyperparams; p=load_hyperparams(); print("Validated campaign parameters:", ", ".join(sorted(p)))'
fi

want() {
    # execute l'etape si --only absent ou si elle y figure
    if [[ ${#ONLY[@]} -eq 0 ]]; then return 0; fi
    for o in "${ONLY[@]}"; do [[ "$o" == "$1" ]] && return 0; done
    return 1
}

run_step() {
    local name="$1"; shift
    local log="$LOG_DIR/${STAMP}_${name}.log"
    echo ""
    echo "=== [$name] $* ==="
    echo "    log: $log"
    "$@" 2>&1 | tee "$log"
}

# ---- garde : panel DNS + labels complet ----
run_step data_gate "$PYTHON_BIN" "$ROOT_DIR/study/pipeline/data_catalog.py" \
    --N "$N" --dim "$DIM"

echo "V3 regeneration run — N=$N dim=$DIM  (commit $(git -C "$ROOT_DIR" rev-parse --short HEAD))"

# ---- gate : suite de tests ----
want tests && run_step tests "$PYTHON_BIN" -m pytest "$ROOT_DIR/tests/study" -q

# ---- taches v3 (t2/t3 sont des bibliotheques : validees par pytest) ----
want t1  && run_step t1  "$PYTHON_BIN" "$ROOT_DIR/study/h2b_prediction/h2b_feature_selection.py" --N "$N" --dim "$DIM"
want t1b && run_step t1b "$PYTHON_BIN" "$ROOT_DIR/study/h2b_prediction/h2b_neighbour_cone_curve.py" --N "$N" --dim "$DIM"
want t4  && run_step t4  "$PYTHON_BIN" "$ROOT_DIR/study/h2b_prediction/h2b_blocked_split.py" --N "$N" --dim "$DIM"
want t5  && run_step t5  "$PYTHON_BIN" "$ROOT_DIR/study/h2b_prediction/h2b_psi_feature_loso.py" --N "$N" --dim "$DIM"
if ! $SKIP_T6 && want t6; then
    run_step t6 "$PYTHON_BIN" "$ROOT_DIR/study/pipeline/dynamic_patch_labels.py" \
        --scenario orszag_tang --re 400 --N "$N" --dim "$DIM" --snaps 2
fi
want t7 && run_step t7 "$PYTHON_BIN" "$ROOT_DIR/study/h2b_prediction/h2b_prediction_horizon.py" --N "$N" --dim "$DIM"
want t9 && run_step t9 "$PYTHON_BIN" "$ROOT_DIR/study/h3_representation/h3_locality_proposition.py" --N "$N" --dim 2 4
want t29 && run_step t29 "$PYTHON_BIN" "$ROOT_DIR/study/h2b_prediction/h2b_loso_delta_ci.py" --N "$N" --dim "$DIM"

# ---- table maitresse auto-verifiante ----
want t10 && run_step t10 "$PYTHON_BIN" "$ROOT_DIR/study/common/aggregate_v3.py" --N "$N" --dim "$DIM" --strict

echo ""
echo "V3 regeneration complete. Master table: $RESULTS_DIR/v3_master_table.md"
