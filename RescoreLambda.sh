#!/bin/bash
set -e
set -o pipefail

# ============================================================
# Q-HAS: Rescore Optuna Trials with Different Lambda Values
# ============================================================

# -----------------------------
# Path Configuration
# -----------------------------
ENV_FILE="environment.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_LOC="$SCRIPT_DIR/src"
ROOT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
# Default input/output
IN_DIR="$SCRIPT_DIR/Train_results"

# -----------------------------
# Default Configurations
# -----------------------------
LAMBDA_COST=""
LAMBDA_SWEEP=""
STUDY_NAME=""
DB_FILE=""
PHASE="all"

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
    echo "Recompute Optuna trial scores with different lambda_cost values."
    echo "Uses existing training results — no re-simulation needed."
    echo ""
    echo "Options:"
    echo "  --lambda <float>                Single lambda_cost value (e.g. 0.3)"
    echo "  --sweep <float> [<float> ...]   Multiple lambda values for sweep"
    echo "                                  (e.g. --sweep 0.0 0.1 0.3 0.5 1.0)"
    echo "  --phase <1|1b|2|3|c1|c2|all>    Which training phase to rescore (default: all)"
    echo "  --db <path>                     Specific .db file (overrides --phase)"
    echo "  --study <name>                  Study name (required with --db)"
    echo "  --in-dir <dir>                  Input directory with .db files"
    echo "                                  (default: Train_results/)"
    echo "  -h, --help                      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --lambda 0.3"
    echo "  $0 --sweep 0.0 0.1 0.2 0.3 0.5 1.0"
    echo "  $0 --lambda 0.5 --phase 1"
    echo "  $0 --sweep 0.0 0.1 0.5 1.0 --phase 3"
    echo "  $0 --db Train_results/q_has_v2_phase1.db --study q_has_v2_phase1 --lambda 0.3"
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
        --lambda) LAMBDA_COST="$2"; shift 2 ;;
        --sweep)
            shift
            LAMBDA_SWEEP=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LAMBDA_SWEEP="$LAMBDA_SWEEP $1"
                shift
            done
            ;;
        --phase) PHASE="$2"; shift 2 ;;
        --db) DB_FILE="$2"; shift 2 ;;
        --study) STUDY_NAME="$2"; shift 2 ;;
        --in-dir) IN_DIR="$2"; shift 2 ;;
        -h|--help) display_help; exit 0 ;;
        *) echo "❌ Unknown option: $1"; display_help; exit 1 ;;
    esac
done

# -----------------------------
# Validation
# -----------------------------
if [ -z "$LAMBDA_COST" ] && [ -z "$LAMBDA_SWEEP" ]; then
    echo "❌ Error: Provide --lambda <value> or --sweep <values>"
    echo ""
    display_help
    exit 1
fi

# Build the lambda arguments for the Python script
LAMBDA_ARGS=""
if [ -n "$LAMBDA_COST" ]; then
    LAMBDA_ARGS="--lambda-cost $LAMBDA_COST"
fi
if [ -n "$LAMBDA_SWEEP" ]; then
    LAMBDA_ARGS="$LAMBDA_ARGS --lambda-sweep $LAMBDA_SWEEP"
fi

echo "=============================================================="
echo "Q-HAS Lambda Rescore"
echo "Script Path: $SCRIPTS_LOC"
echo "Input Dir:   $IN_DIR"
if [ -n "$LAMBDA_COST" ]; then
    echo "Lambda:      $LAMBDA_COST"
fi
if [ -n "$LAMBDA_SWEEP" ]; then
    echo "Sweep:      $LAMBDA_SWEEP"
fi
echo "=============================================================="

# -----------------------------
# Pipeline Execution
# -----------------------------

run_rescore() {
    local db_path="$1"
    local study_name="$2"

    if [ ! -f "$db_path" ]; then
        echo "⚠️ Skipping $db_path (file not found)"
        return
    fi

    run_stage "Rescore $study_name" python "$SCRIPTS_LOC/recompute_lambda_scores.py" \
        --db-path "$db_path" \
        --study-name "$study_name" \
        $LAMBDA_ARGS
}

if [ -n "$DB_FILE" ]; then
    # Explicit db file
    if [ -z "$STUDY_NAME" ]; then
        echo "❌ Error: --study is required when using --db"
        exit 1
    fi
    run_rescore "$DB_FILE" "$STUDY_NAME"
else
    # Phase-based
    case $PHASE in
        1)
            run_rescore "$IN_DIR/q_has_v2_phase1.db" "q_has_v2_phase1"
            ;;
        1_agressive)
            run_rescore "$IN_DIR/q_has_v2_phase1_agr.db" "q_has_v2_phase1_agr"
            ;;
        1b)
            run_rescore "$IN_DIR/q_has_v2_phase1b.db" "q_has_v2_phase1b"
            ;;
        1b_agressive)
            run_rescore "$IN_DIR/q_has_v2_phase1b_agr.db" "q_has_v2_phase1b_agr"
            ;;
        2)
            run_rescore "$IN_DIR/q_has_v2_phase2.db" "q_has_v2_phase2"
            ;;
        2_agressive)
            run_rescore "$IN_DIR/q_has_v2_phase2_agr.db" "q_has_v2_phase2_agr"
            ;;
        3)
            run_rescore "$IN_DIR/q_has_v2_phase3.db" "q_has_v2_phase3"
            ;;
        c1)
            run_rescore "$IN_DIR/classical_v2_phase1.db" "classical_v2_phase1"
            ;;
        c2)
            run_rescore "$IN_DIR/classical_v2_phase2.db" "classical_v2_phase2"
            ;;
        c3)
            run_rescore "$IN_DIR/classical_v2_phase3.db" "classical_v2_phase3"
            ;;
        all)
            run_rescore "$IN_DIR/q_has_v2_phase1.db" "q_has_v2_phase1"
            run_rescore "$IN_DIR/q_has_v2_phase1_agr.db" "q_has_v2_phase1_agr"
            run_rescore "$IN_DIR/q_has_v2_phase1b.db" "q_has_v2_phase1b"
            run_rescore "$IN_DIR/q_has_v2_phase1b_agr.db" "q_has_v2_phase1b_agr"
            run_rescore "$IN_DIR/q_has_v2_phase2.db" "q_has_v2_phase2"
            run_rescore "$IN_DIR/q_has_v2_phase2_agr.db" "q_has_v2_phase2_agr"
            run_rescore "$IN_DIR/q_has_v2_phase3.db" "q_has_v2_phase3"
            run_rescore "$IN_DIR/classical_v2_phase1.db" "classical_v2_phase1"
            run_rescore "$IN_DIR/classical_v2_phase2.db" "classical_v2_phase2"
            run_rescore "$IN_DIR/classical_v2_phase3.db" "classical_v2_phase3"
            ;;
        *)
            echo "❌ Unknown phase: $PHASE (use 1, 1b, 2, 3, c1, c2, or all)"
            exit 1
            ;;
    esac
fi

# -----------------------------
# ✅ Completion
# -----------------------------
echo "=============================================================="
echo "🎉 Lambda rescore completed successfully!"
echo "=============================================================="
exit 0
