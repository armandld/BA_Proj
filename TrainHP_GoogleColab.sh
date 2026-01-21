#!/bin/bash
set -e
set -o pipefail

# ============================================================
# Q-HAS: Quantum-Hierarchical Adaptive Steering Pipeline - Main Workflow
# ============================================================

# -----------------------------
# Default Configurations
# -----------------------------
BACKEND="aer"
MODE="simulator"
METHOD="COBYLA"
NUM_SHOTS=100000
VERBOSE=false
SKIP_CLEANUP=false
ONLY_MAPPING=false
ONLY_OPTIMIZE=false
ONLY_EXECUTE=false
ONLY_POSTPROCESS=false
NUM_QBITS=4
DEPTH=2
OPT_LEVEL=3
GRID_SIZE=16
DNS_RESOLUTION=256
T_MAX=1.0
DT=1e-4
HYBRID_DT=0.1
ADVANCED_ANOMALIES_ENABLE=false

# -----------------------------
# Path Configuration
# -----------------------------
ENV_FILE="environment.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_LOC="$SCRIPT_DIR/src"
ROOT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
# Output and log paths
OUT_DIR="$SCRIPT_DIR/data"
IN_FILE="$SCRIPT_DIR/input/mapping_input.json"
LOG_DIR="$OUT_DIR/../logs"
LOG_FILE="$LOG_DIR/pipeline[$(date +'%Y-%m-%d_%H-%M-%S')].log"

# Make sure directories exist
mkdir -p "$OUT_DIR"
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
    echo "  --backend <aer|estimator>       Quantum backend (default: aer)"
    echo "  --mode <simulator|hardware>     Simulator or IBM Quantum (default: simulator)"
    echo "  --shots <int>                   Number of shots (default: 1024)"
    echo "  --numqbits <int>                Number of qubits (default: 4)"
    echo "  --depth <int>                  Depth of the ULA ansatz (default: 2)"
    echo "  --opt_level <0|1|2|3>          Optimization level for transpiler (default: 3)"
    echo "  --out-dir <dir>                 Output directory (default: data)"
    echo "  --verbose                       Enable verbose logging"
    echo "  --skip-cleanup                  Skip deleting previous data"
    echo "  --method <COBYLA|Nelder-Mead|Powell|L-BFGS-B>               Optimization method for minimize (default: COBYLA)"
    echo "Custom Domain Parameters:"
    echo "  --grid-size <int>               Coarse grid dimension N (NxN) (default: 16)"
    echo "  --dns-resolution <int>          High-Res Grid for Ground Truth (default: 256)"
    echo "  --t-max <float>                 Simulation end time (default: 1.0)"
    echo "  --dt <float>                    Time step size (default: 0.01)"
    echo "  --hybrid-dt <float>               Hybrid simulation time step size (default: 0.1)"
    echo "  --AdvAnomaliesEnable            Enable advanced anomaly handling in mapping"
    echo ""
    echo "Stage control (choose one):"
    echo "  --only-mapping                  Run mapping stage only"
    echo "  --only-optimize                 Run optimization stage only"
    echo "  --only-execute                  Run execution stage only"
    echo "  --only-postprocess              Run post-processing stage only"
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
        --backend) BACKEND="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --shots) NUM_SHOTS="$2"; shift 2 ;;
        --in-file) IN_DIR="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --verbose) VERBOSE=true; shift ;;
        --skip-cleanup) SKIP_CLEANUP=true; shift ;;
        --depth) DEPTH="$2"; shift 2 ;;
        --opt-level) OPT_LEVEL="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --grid-size) GRID_SIZE="$2"; shift 2 ;;
        --dns-resolution) DNS_RESOLUTION="$2"; shift 2 ;;
        --t-max) T_MAX="$2"; shift 2 ;;
        --dt) DT="$2"; shift 2 ;;
        --hybrid-dt) HYBRID_DT="$2"; shift 2 ;;
        --AdvAnomaliesEnable) ADVANCED_ANOMALIES_ENABLE=true; shift ;;
        --only-mapping) ONLY_MAPPING=true; shift ;;
        --only-optimize) ONLY_OPTIMIZE=true; shift ;;
        --only-execute) ONLY_EXECUTE=true; shift ;;
        --only-postprocess) ONLY_POSTPROCESS=true; shift ;;
        -h|--help) display_help; exit 0 ;;
        *) echo "❌ Unknown option: $1"; display_help; exit 1 ;;
    esac
done

# -----------------------------
# Stage Validation
# -----------------------------
STAGE_ONLY_COUNT=0
$ONLY_MAPPING && ((STAGE_ONLY_COUNT++))
$ONLY_OPTIMIZE && ((STAGE_ONLY_COUNT++))
$ONLY_EXECUTE && ((STAGE_ONLY_COUNT++))
$ONLY_POSTPROCESS && ((STAGE_ONLY_COUNT++))

if [ $STAGE_ONLY_COUNT -gt 1 ]; then
    echo "❌ Error: Multiple stage-only flags provided."
    exit 1
fi

# -----------------------------
# Environment Preparation
# -----------------------------
mkdir -p "$OUT_DIR"
> "$LOG_FILE"

log "=============================================================="
log "🚀 Starting Training Q-HAS' Hyper Parameters: Quantum-Hierarchical Adaptive Steering Pipeline - Variational Workflow"
log "Mode: $MODE | Backend: $BACKEND | Shots: $NUM_SHOTS | Method: $METHOD"
log "Script Path: $SCRIPTS_LOC"
log "Output Dir: $OUT_DIR"
log "Input File: $IN_FILE"
log "=============================================================="

if [ "$SKIP_CLEANUP" = false ]; then
    log "🧹 Cleaning previous results..."
    rm -rf "$OUT_DIR"/*
fi

# -----------------------------
# Pipeline Execution
# -----------------------------
echo "ROOT DIR: $ROOT_DIR"
run_stage "Training Hyperparameters" python "$SCRIPTS_LOC/TrainHyperParam.py" \
    2>&1 | tee -a "$LOG_FILE"


# -----------------------------
# ✅ Completion
# -----------------------------
log "=============================================================="
log "🎉 Q-HAS completed successfully!"
log "Full log: $LOG_FILE"
log "=============================================================="
exit 0
