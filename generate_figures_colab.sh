#!/bin/bash
set -o pipefail
export LC_NUMERIC=C
# ============================================================
# Generate Figures on Google Colab — wrapper for generate_figures.sh
# ============================================================
#
# Usage:
#   bash generate_figures_colab.sh --phase <1|2|3> [--lambda <value>]
#
# This script:
#   1. Runs each figure script individually with per-script logging
#   2. Saves figures to figures/phase<N>/
#   3. Saves logs to figures/phase<N>/logs/
#   4. Creates a summary JSON for the notebook to parse
#
# Designed to be called from the Colab notebook.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
FIGURES_CODE_DIR="$ROOT_DIR/figures_code"

# Defaults
LAMBDA_COST=0.40
PHASE=0
TARGET_FIGURES=()

# ── Argument parsing ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2 ;;
        --lambda) LAMBDA_COST="$2"; shift 2 ;;
        --figures)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                TARGET_FIGURES+=("$1"); shift
            done
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ $PHASE != 1 && $PHASE != 2 && $PHASE != 3 ]]; then
    echo "Error: --phase must be 1, 2, or 3 (got '$PHASE')"
    exit 1
fi

OUTPUT_DIR="$ROOT_DIR/figures/phase${PHASE}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

# ── Step 1: Extract best hyperparams ──
log "Step 1: Extracting best hyperparameters for phase ${PHASE}..."

case $PHASE in
    1) Q_PHASE_FILTER="phase1"; C_PHASE_FILTER="phase1"
       export FIGURE_Q_PHASE_FILTER="1"; export FIGURE_C_PHASE_FILTER="1" ;;
    2) Q_PHASE_FILTER="phase2";  C_PHASE_FILTER="phase2"
       export FIGURE_Q_PHASE_FILTER="2";  export FIGURE_C_PHASE_FILTER="2" ;;
    3) Q_PHASE_FILTER="phase3";  C_PHASE_FILTER="phase3"
       export FIGURE_Q_PHASE_FILTER="3";  export FIGURE_C_PHASE_FILTER="3" ;;
esac

export FIGURE_PHASE="$PHASE"

TRAIN_RESULTS_DIR="$ROOT_DIR/Train_results"
if [[ -d "$TRAIN_RESULTS_DIR" ]] && [[ -f "$ROOT_DIR/scripts/extract_best_hyperparams.py" ]]; then
    python "$ROOT_DIR/scripts/extract_best_hyperparams.py" \
        --train-dir "$TRAIN_RESULTS_DIR" \
        --output "$ROOT_DIR/best_hyperparams.json" \
        --lambda-cost "$LAMBDA_COST" \
        --top-k 3 \
        --quantum-phase-filter "$Q_PHASE_FILTER" \
        --classical-phase-filter "$C_PHASE_FILTER" \
        2>&1 | tee "$LOG_DIR/extract_hyperparams.log"
    log "  best_hyperparams.json updated."
else
    log "  WARNING: Train_results/ or extract script not found. Using existing best_hyperparams.json"
fi

# ── Step 2: Run figure scripts one by one with logging ──
log "Step 2: Generating figures (phase ${PHASE})..."

FIGURE_SCRIPTS=(
    "fig0_pareto_lambda.py"
    "fig1_noise_robustness.py"
    "fig2_early_detection.py"
    "fig3_spatial_coherence.py"
    "fig4_comprehensive_comparison.py"
    "fig5_qaoa_detailed_analysis.py"
    "fig6_statistical_validation.py"
    "fig7_physical_fidelity.py"
    "fig8_hierarchical_comparison.py"
    "fig9_synthetic_unit_tests.py"
    "fig10_grid_scaling.py"
    "fig11_hamiltonian_design.py"
    "fig12_depth_analysis.py"
    "fig13_sigma_ablation.py"
    "fig14_boundary_correction.py"
    "fig15_decision_flip_analysis.py"
    "fig16_decision_landscape.py"
    "fig17_topological_attribution.py"
)

if [ ${#TARGET_FIGURES[@]} -gt 0 ]; then
    FILTERED_SCRIPTS=()
    for f_num in "${TARGET_FIGURES[@]}"; do
        match=$(printf "%s\n" "${FIGURE_SCRIPTS[@]}" | grep -E "^fig${f_num}_")
        [[ -n "$match" ]] && FILTERED_SCRIPTS+=("$match")
    done
    FIGURE_SCRIPTS=("${FILTERED_SCRIPTS[@]}")
fi

FAILED=0
SUCCEEDED=0
SUMMARY_FILE="$OUTPUT_DIR/generation_summary.json"
echo '{"phase": '$PHASE', "lambda": '$LAMBDA_COST', "scripts": [' > "$SUMMARY_FILE"
FIRST=true

for script in "${FIGURE_SCRIPTS[@]}"; do
    script_path="$FIGURES_CODE_DIR/$script"
    script_name="${script%.py}"
    log_file="$LOG_DIR/${script_name}.log"

    if [[ ! -f "$script_path" ]]; then
        log "  SKIP: $script (not found)"
        continue
    fi

    log "  Running $script ..."
    START_TIME=$(date +%s)

    if python "$script_path" > "$log_file" 2>&1; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        SUCCEEDED=$((SUCCEEDED + 1))
        STATUS="ok"
        log "  OK: $script (${DURATION}s)"
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        FAILED=$((FAILED + 1))
        STATUS="failed"
        log "  FAILED: $script (${DURATION}s) — see $log_file"
        # Print last few lines of error
        tail -5 "$log_file" 2>/dev/null
    fi

    # Append to summary JSON
    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo ',' >> "$SUMMARY_FILE"
    fi
    echo "  {\"script\": \"$script\", \"status\": \"$STATUS\", \"duration_s\": $DURATION}" >> "$SUMMARY_FILE"
done

echo '],' >> "$SUMMARY_FILE"
echo "\"succeeded\": $SUCCEEDED, \"failed\": $FAILED" >> "$SUMMARY_FILE"
echo '}' >> "$SUMMARY_FILE"

# ── Summary ──
log "============================================================"
log "Figure generation complete for Phase ${PHASE}"
log "  Succeeded: $SUCCEEDED"
log "  Failed:    $FAILED"
log "  Figures:   $OUTPUT_DIR/"
log "  Logs:      $LOG_DIR/"
log "  Summary:   $SUMMARY_FILE"
log "============================================================"

# List generated figures
FIGURE_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
log "  Generated $FIGURE_COUNT PNG files"

exit $FAILED
