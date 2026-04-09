#!/bin/bash
set -e
set -o pipefail

# ============================================================
# Import Data from Neon Script
# This script imports Optuna study data from a Neon database to local storage.
# ============================================================

# -----------------------------
# Default Configurations
# -----------------------------
RESET_NEON=false
RESET_LOCAL=false
LocalToNeon=false


# -----------------------------
# Path Configuration
# -----------------------------
ENV_FILE="environment.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_LOC="$SCRIPT_DIR/src"
ROOT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
# Output and log paths
TRAIN_DIR="$SCRIPT_DIR/Train_results"
IN_URL="postgresql://neondb_owner:npg_osTe7ENJpZz5@ep-patient-hall-abitnl4g-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
LOG_DIR="$TRAIN_DIR/../logs"
LOG_FILE="$LOG_DIR/import_data_[$(date +'%Y-%m-%d_%H-%M-%S')].log"

# Make sure directories exist
mkdir -p "$TRAIN_DIR"
mkdir -p "$LOG_DIR"

# Now you can safely write logs to LOG_FILE
> "$LOG_FILE"

# -----------------------------
# Logging Helper
# -----------------------------
pipeline_log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# -----------------------------
# Conda Environment Detection
# -----------------------------
if [ -f "$ROOT_DIR/$ENV_FILE" ]; then
    # Extract the ENV_NAME variable from setup_env.sh
    ENV_NAME=$(grep 'name:' $ENV_FILE | cut -d ' ' -f 2)
    
    if [ -z "$ENV_NAME" ]; then
        echo "⚠️ Could not detect Conda environment from $ENV_FILE. Please activate manually."
    else
        echo "🔹 Detected Conda environment: $ENV_NAME"
        # Activate it
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
    echo "Options:"
    echo "  --train-dir <dir>                 Training directory (default: Train_results)"
    echo "  --in-url <url>                    Input URL for Neon database (default: postgresql://neondb_owner:...)"
    echo "  --reset                             Reset existing study in Neon before import"
    echo "  --LocalToNeon                       Import from local Optuna to Neon (default: false)"
}

# -----------------------------
# Logging Helper
# -----------------------------
log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# -----------------------------
# Run Stage Helper (Array Version)
# -----------------------------
run_stage() {
    local stage_name="$1"
    shift
    local cmd=("$@")

    log "➡️ Stage: $stage_name"
    log "Running: ${cmd[*]}"

    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}  # exit code of the Python command

    if [ $exit_code -ne 0 ]; then
        log "❌ Stage '$stage_name' failed with exit code $exit_code!"
        exit $exit_code
    fi

    log "✅ Stage '$stage_name' completed successfully."
}


# -----------------------------
# Argument Parsing
# -----------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-dir) TRAIN_DIR="$2"; shift 2 ;;
        --in-url) IN_URL="$2"; shift 2 ;;
        --ResetNeon) RESET_NEON=true; shift ;;
        --ResetLocal) RESET_LOCAL=true; shift ;;
        --LocalToNeon) LocalToNeon=true; shift ;;
        -h|--help) display_help; exit 0 ;;
        *) echo "❌ Unknown option: $1"; display_help; exit 1 ;;
    esac
done


# -----------------------------
# Environment Preparation
# -----------------------------
mkdir -p "$TRAIN_DIR"
> "$LOG_FILE"

log "=============================================================="
log "IMPORTING DATA FROM NEON - Q-HAS Pipeline"
log "Script Path: $SCRIPTS_LOC"
log "Train Results Dir: $TRAIN_DIR"
log "Input File: $IN_URL"
log "=============================================================="

# -----------------------------
# Pipeline Execution
# -----------------------------
echo "ROOT DIR: $ROOT_DIR"

run_stage "IMPORT" python "$SCRIPTS_LOC/import_Neon_data_to_local.py" \
    --train-dir "$TRAIN_DIR" \
    --in-url "$IN_URL" \
    $([ "$LocalToNeon" = true ] && echo "--LocalToNeon") \
    $([ "$RESET_NEON" = true ] && echo "--ResetNeon") \
    $([ "$RESET_LOCAL" = true ] && echo "--ResetLocal") \
    2>&1 | tee -a "$LOG_FILE"

# -----------------------------
# ✅ Completion
# -----------------------------
log "=============================================================="
log "IMPORT completed successfully!"
log "Full log: $LOG_FILE"
log "=============================================================="
exit 0
