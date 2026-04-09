#!/bin/bash
set -e
set -o pipefail

# ============================================================
# Q-HAS: Quantum-Hierarchical Adaptive Steering Visualization of Hyperparameters
# ============================================================

# -----------------------------
# Path Configuration
# -----------------------------
ENV_FILE="environment.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_LOC="$SCRIPT_DIR/src"
ROOT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
# Output and echo paths
IN_FILE="$SCRIPT_DIR/Train_results"



FULL=false

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
    echo "  --full                          Enable all graph constructions"
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
# Run Stage Helper (Array Version)
# -----------------------------
run_stage() {
    local stage_name="$1"
    shift
    local cmd=("$@")

    echo "➡️ Stage: $stage_name"
    echo "Running: ${cmd[*]}"

    "${cmd[@]}" 
    local exit_code=${PIPESTATUS[0]}  # exit code of the Python command

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
        --in-file) IN_DIR="$2"; shift 2 ;;
        --full) FULL=true; shift ;;
        -h|--help) display_help; exit 0 ;;
        *) echo "❌ Unknown option: $1"; display_help; exit 1 ;;
    esac
done


echo "=============================================================="
echo "Visual of the Training Q-HAS' Hyper Parameters: Quantum-Hierarchical Adaptive Steering Pipeline - Variational Workflow"
echo "Script Path: $SCRIPTS_LOC"
echo "Input File: $IN_FILE"
echo "=============================================================="

# -----------------------------
# Pipeline Execution
# -----------------------------
run_stage "Analyze hyperparameters of phase 1" python "$SCRIPTS_LOC/analyze_hyperparams.py" \
    --db-path "$IN_FILE"/q_has_v2_phase1.db \
    --study-name q_has_v2_phase1 \
    $([ "$FULL" = true ] && echo "--full")
run_stage "Analyze hyperparameters of phase 1 agressive" python "$SCRIPTS_LOC/analyze_hyperparams.py" \
    --db-path "$IN_FILE"/q_has_v2_phase1_agr.db \
    --study-name q_has_v2_phase1_agr \
    $([ "$FULL" = true ] && echo "--full")
run_stage "Analyze hyperparameters of phase 1b" python "$SCRIPTS_LOC/analyze_hyperparams.py" \
    --db-path "$IN_FILE"/q_has_v2_phase1b.db \
    --study-name q_has_v2_phase1b \
    $([ "$FULL" = true ] && echo "--full")
run_stage "Analyze hyperparameters of phase 1b agressive" python "$SCRIPTS_LOC/analyze_hyperparams.py" \
    --db-path "$IN_FILE"/q_has_v2_phase1b_agr.db \
    --study-name q_has_v2_phase1b_agr \
    $([ "$FULL" = true ] && echo "--full")
run_stage "Analyze hyperparameters of phase 2" python "$SCRIPTS_LOC/analyze_hyperparams.py" \
    --db-path "$IN_FILE"/q_has_v2_phase2.db \
    --study-name q_has_v2_phase2 \
    $([ "$FULL" = true ] && echo "--full")
run_stage "Analyze hyperparameters of phase 2 agressive" python "$SCRIPTS_LOC/analyze_hyperparams.py" \
    --db-path "$IN_FILE"/q_has_v2_phase2_agr.db \
    --study-name q_has_v2_phase2_agr \
    $([ "$FULL" = true ] && echo "--full")
run_stage "Analyze hyperparameters of phase 3" python "$SCRIPTS_LOC/analyze_hyperparams.py" \
    --db-path "$IN_FILE"/q_has_v2_phase3.db \
    --study-name q_has_v2_phase3 \
    $([ "$FULL" = true ] && echo "--full")
run_stage "Analyze hyperparameters of classical phase 1" python "$SCRIPTS_LOC/analyze_hyperparams.py" \
    --db-path "$IN_FILE"/classical_v2_phase1.db \
    --study-name classical_v2_phase1 \
    $([ "$FULL" = true ] && echo "--full")
run_stage "Analyze hyperparameters of classical phase 2" python "$SCRIPTS_LOC/analyze_hyperparams.py" \
    --db-path "$IN_FILE"/classical_v2_phase2.db \
    --study-name classical_v2_phase2 \
    $([ "$FULL" = true ] && echo "--full")
run_stage "Analyze hyperparameters of classical phase 3" python "$SCRIPTS_LOC/analyze_hyperparams.py" \
    --db-path "$IN_FILE"/classical_v2_phase3.db \
    --study-name classical_v2_phase3 \
    $([ "$FULL" = true ] && echo "--full")

# -----------------------------
# ✅ Completion
# -----------------------------
echo "=============================================================="
echo "🎉 Visual completed successfully!"
echo "=============================================================="
exit 0
