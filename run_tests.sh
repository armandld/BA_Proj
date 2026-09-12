#!/bin/bash
set -e
set -o pipefail

# ============================================================
# Q-HAS: Test Runner Workflow
# ============================================================

# -----------------------------
# Default Configurations
# -----------------------------
RUN_ALL=true
RUN_SOLVER=false
RUN_VQA=false
RUN_QAOA=false
RUN_V9=false
RUN_MODULES=false
RUN_FIGURES=false
RUN_DIAGNOSE=false

# -----------------------------
# Path Configuration
# -----------------------------
ENV_FILE="environment.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR" && pwd)"

# Output and log paths
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/test_runner[$(date +'%Y-%m-%d_%H-%M-%S')].log"

# Make sure directories exist
mkdir -p "$LOG_DIR"
> "$LOG_FILE"

# -----------------------------
# Logging Helper
# -----------------------------
log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# -----------------------------
# Conda Environment Detection
# -----------------------------
if [ -f "$ROOT_DIR/$ENV_FILE" ]; then
    ENV_NAME=$(grep 'name:' "$ENV_FILE" | cut -d ' ' -f 2)

    if [ -z "$ENV_NAME" ]; then
        echo "Warning: Could not detect Conda environment from $ENV_FILE. Please activate manually."
    else
        echo "Detected Conda environment: $ENV_NAME"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
    fi
else
    echo "Warning: $ENV_FILE not found. Make sure the Conda environment is active."
fi

# -----------------------------
# Help Function
# -----------------------------
display_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "If no option is provided, ALL tests are run (default)."
    echo ""
    echo "Test targeting options:"
    echo "  --solver        MHD solver convergence (test_solver_convergence.py)"
    echo "  --vqa           VQA anomaly detection chain (test_vqa_anomaly_cases.py)"
    echo "  --qaoa          QAOA end-to-end pipeline (QAOA_test.py)"
    echo "  --v9            v9 Hamiltonian diagnostics + metrics (test_hamiltonian_v9_diagnostic.py,"
    echo "                  test_v9_metrics.py)"
    echo "  --modules       Module-by-module validation (test_module_validation.py,"
    echo "                  test_signal_contribution.py, diag_hamiltonian_balance.py)"
    echo "  --figures       Figure evaluation tests (test_qaoa_advantage.py,"
    echo "                  test_qaoa_noise_and_early.py, test_qaoa_scaling_and_hparams.py,"
    echo "                  test_qaoa_decisions.py, test_qaoa_physics_decision.py)"
    echo "  --diagnose      Diagnostic convergence (diagnose_convergence.py)"
    echo ""
    echo "Other options:"
    echo "  -h, --help      Show this help message"
}

# -----------------------------
# Argument Parsing
# -----------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --solver)
            RUN_SOLVER=true
            RUN_ALL=false
            shift
            ;;
        --vqa)
            RUN_VQA=true
            RUN_ALL=false
            shift
            ;;
        --qaoa)
            RUN_QAOA=true
            RUN_ALL=false
            shift
            ;;
        --v9)
            RUN_V9=true
            RUN_ALL=false
            shift
            ;;
        --modules)
            RUN_MODULES=true
            RUN_ALL=false
            shift
            ;;
        --figures)
            RUN_FIGURES=true
            RUN_ALL=false
            shift
            ;;
        --diagnose)
            RUN_DIAGNOSE=true
            RUN_ALL=false
            shift
            ;;
        -h|--help)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
done

# -----------------------------
# Run Stage Helper
# -----------------------------
run_stage() {
    local stage_name="$1"
    shift
    local cmd=("$@")

    log "Stage: $stage_name"
    log "Running: ${cmd[*]}"

    # Execute command, capturing output
    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        log "FAILED: Stage '$stage_name' failed with exit code $exit_code!"
        exit $exit_code
    fi

    log "PASSED: Stage '$stage_name' completed successfully."
    echo "--------------------------------------------------------------"
}

# -----------------------------
# Pipeline Execution
# -----------------------------
log "=============================================================="
log "Starting Q-HAS Test Suite"
log "Root Dir: $ROOT_DIR"
log "=============================================================="

# ── DEFAULT: Run all test suites ──
if $RUN_ALL; then
    # Tier 0: Fast structural tests (< 5s)
    run_stage "v9 Hamiltonian diagnostics" python -m pytest tests/mapping/test_hamiltonian_v9_diagnostic.py -v
    run_stage "v9 metrics (Tier 0)" python -m pytest tests/pipeline/test_v9_metrics.py -v

    # Tier 1: Module validation (< 30s)
    run_stage "Module validation" python -m pytest tests/pipeline/test_module_validation.py -v
    run_stage "Signal contribution" python -m pytest tests/mapping/test_signal_contribution.py -v

    # Tier 1b: Additional structural tests
    run_stage "Shock gradient / X-point" python -m pytest tests/mapping/test_shock_gradient_proposal.py -v
    run_stage "Beta X-point sensitivity" python -m pytest tests/mapping/test_beta_xpoint.py -v

    # Tier 1c: Diagnostic scripts (standalone, not pytest-compatible)
    run_stage "Hamiltonian balance diagnostic" bash -c "cd tests/tools && python diag_hamiltonian_balance.py"
    run_stage "QAOA contribution diagnostic" bash -c "cd tests/tools && python diag_qaoa_contribution.py"

    # Tier 2: VQA chain (< 60s)
    run_stage "VQA anomaly cases" python -m pytest tests/quantum/test_vqa_anomaly_cases.py -v
    run_stage "QAOA end-to-end pipeline" python -m pytest tests/quantum/QAOA_test.py -v

    # Tier 3: Physics decision tests (< 120s)
    run_stage "QAOA physics decisions" python -m pytest tests/quantum/test_qaoa_physics_decision.py -v
    run_stage "QAOA controlled decisions" bash -c "cd tests/quantum && python test_qaoa_decisions.py"

    # Tier 4: Solver convergence (slow, ~8min)
    run_stage "MHD solver convergence" python -m pytest tests/solver/test_solver_convergence.py -v

    # Tier 5: Evaluation figures (slow, ~5-10min each)
    run_stage "QAOA advantage" python tests/quantum/test_qaoa_advantage.py
    run_stage "QAOA noise & early detection" python -m pytest tests/quantum/test_qaoa_noise_and_early.py -v
    run_stage "QAOA scaling & hparams" python -m pytest tests/quantum/test_qaoa_scaling_and_hparams.py -v

    # Tier 6: Diagnostic convergence (standalone scripts)
    run_stage "Diagnostic convergence" bash -c "cd tests/tools && python diagnose_convergence.py"
fi

# ── TARGETED: Individual test groups ──

if $RUN_SOLVER; then
    run_stage "MHD solver convergence" python -m pytest tests/solver/test_solver_convergence.py -v
fi

if $RUN_VQA; then
    run_stage "VQA anomaly cases" python -m pytest tests/quantum/test_vqa_anomaly_cases.py -v
fi

if $RUN_QAOA; then
    run_stage "QAOA end-to-end pipeline" python -m pytest tests/quantum/QAOA_test.py -v
fi

if $RUN_V9; then
    run_stage "v9 Hamiltonian diagnostics" python -m pytest tests/mapping/test_hamiltonian_v9_diagnostic.py -v
    run_stage "v9 metrics (Tier 0)" python -m pytest tests/pipeline/test_v9_metrics.py -v
fi

if $RUN_MODULES; then
    run_stage "Module validation" python -m pytest tests/pipeline/test_module_validation.py -v
    run_stage "Signal contribution" python -m pytest tests/mapping/test_signal_contribution.py -v
    run_stage "Hamiltonian balance diagnostic" bash -c "cd tests/tools && python diag_hamiltonian_balance.py"
fi

if $RUN_FIGURES; then
    run_stage "QAOA advantage" python tests/quantum/test_qaoa_advantage.py
    run_stage "QAOA noise & early detection" python -m pytest tests/quantum/test_qaoa_noise_and_early.py -v
    run_stage "QAOA scaling & hparams" python -m pytest tests/quantum/test_qaoa_scaling_and_hparams.py -v
    run_stage "QAOA controlled decisions" bash -c "cd tests/quantum && python test_qaoa_decisions.py"
    run_stage "QAOA physics decisions" python -m pytest tests/quantum/test_qaoa_physics_decision.py -v
fi

if $RUN_DIAGNOSE; then
    run_stage "Diagnostic convergence" bash -c "cd tests/tools && python diagnose_convergence.py"
    run_stage "QAOA contribution diagnostic" bash -c "cd tests/tools && python diag_qaoa_contribution.py"
fi

# -----------------------------
# Completion
# -----------------------------
log "=============================================================="
log "All requested tests completed successfully!"
log "Full log: $LOG_FILE"
log "=============================================================="
exit 0
