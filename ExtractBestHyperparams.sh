#!/bin/bash
set -e
set -o pipefail

# ============================================================
# Q-HAS: Extract Best Hyperparameters from Rescore Results
# ============================================================

# -----------------------------
# Path Configuration
# -----------------------------
ENV_FILE="environment.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_LOC="$SCRIPT_DIR/scripts"
ROOT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
# Default input/output
IN_DIR="$SCRIPT_DIR/Train_results"
OUT_FILE="$SCRIPT_DIR/best_hyperparams.json"

# -----------------------------
# Default Configurations
# -----------------------------
TOP_K=3
LAMBDA_COST=0.40

# -----------------------------
# Conda Environment Detection
# -----------------------------
if [ -f "$ROOT_DIR/$ENV_FILE" ]; then
    ENV_NAME=$(grep 'name:' $ENV_FILE | cut -d ' ' -f 2)

    if [ -z "$ENV_NAME" ]; then
        echo "⚠️ Could not detect Conda environment from $ENV_FILE. Please activate manually."
    else
        echo "🔹 Detected Conda environment: $ENV_NAME"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
    fi
else
    echo "⚠️ $ENV_FILE not found. Make sure the Conda environment is active."
fi

# -----------------------------
# Help Function
# -----------------------------
display_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Extract top-K hyperparameters from rescore CSVs into best_hyperparams.json."
    echo "Scans Train_results/rescore_q_has_v2_phase*_lambda*/ directories."
    echo ""
    echo "Options:"
    echo "  --top-k <int>        Number of top trials per phase/lambda (default: 3)"
    echo "  --lambda-cost <float> Lambda cost for default selection (default: 0.40)"
    echo "  --in-dir <dir>       Input directory with rescore results"
    echo "                       (default: Train_results/)"
    echo "  --output <file>      Output JSON file path"
    echo "                       (default: best_hyperparams.json)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                           # defaults (top 3)"
    echo "  $0 --top-k 5                 # top 5 per phase/lambda"
    echo "  $0 --output out.json         # custom output path"
}

# -----------------------------
# Run Stage Helper
# -----------------------------
run_stage() {
    local stage_name="$1"
    shift
    local cmd=("$@")

    echo "➡️ Stage: $stage_name"
    echo "Running: ${cmd[*]}"

    "${cmd[@]}"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        echo "❌ Stage '$stage_name' failed with exit code $exit_code!"
        exit $exit_code
    fi

    echo "✅ Stage '$stage_name' completed successfully."
}

# -----------------------------
# Argument Parsing
# -----------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --top-k) TOP_K="$2"; shift 2 ;;
        --lambda-cost) LAMBDA_COST="$2"; shift 2 ;;
        --in-dir) IN_DIR="$2"; shift 2 ;;
        --output) OUT_FILE="$2"; shift 2 ;;
        -h|--help) display_help; exit 0 ;;
        *) echo "❌ Unknown option: $1"; display_help; exit 1 ;;
    esac
done

echo "=============================================================="
echo "Q-HAS Extract Best Hyperparameters"
echo "Script Path: $SCRIPTS_LOC"
echo "Input Dir:   $IN_DIR"
echo "Output:      $OUT_FILE"
echo "Top-K:       $TOP_K"
echo "Lambda:      $LAMBDA_COST"
echo "=============================================================="

# -----------------------------
# Pipeline Execution
# -----------------------------
run_stage "Extract best hyperparameters" python "$SCRIPTS_LOC/extract_best_hyperparams.py" \
    --train-dir "$IN_DIR" \
    --output "$OUT_FILE" \
    --top-k "$TOP_K" \
    --lambda-cost "$LAMBDA_COST"

# -----------------------------
# ✅ Completion
# -----------------------------
echo "=============================================================="
echo "🎉 Hyperparameter extraction completed successfully!"
echo "   Output: $OUT_FILE"
echo "=============================================================="
exit 0
