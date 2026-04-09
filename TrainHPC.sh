#!/bin/bash
set -e
set -o pipefail

# ============================================================
# Q-HAS: Distributed Hyperparameter Training — HPC (SLURM)
# ============================================================
#
# This script orchestrates the 3-phase training on an HPC cluster.
# It submits SLURM job arrays and waits for each phase to complete
# before launching the next one.
#
# Usage:
#   bash TrainHP_HPC.sh                     # run all 3 phases (default)
#   bash TrainHP_HPC.sh --phase 1           # run only phase 1
#   bash TrainHP_HPC.sh --workers 10        # 10 workers per phase
#   bash TrainHP_HPC.sh --setup             # only create conda env, don't train
#   bash TrainHP_HPC.sh -h                  # help
# ============================================================

# -----------------------------
# Default Configurations
# -----------------------------
NUM_WORKERS=10
TRIALS_P1=40
TRIALS_P2=50
TRIALS_P3=60
PHASE="all"
SETUP_ONLY=false

# -----------------------------
# Path Configuration
# -----------------------------
ENV_FILE="environment.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
SLURM_SCRIPT="$ROOT_DIR/hpc/submit_training.sh"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/hpc_orchestrator_$(date +'%Y-%m-%d_%H-%M-%S').log"

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
# Help Function
# -----------------------------
display_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --workers <int>        Number of SLURM workers per phase (default: $NUM_WORKERS)"
    echo "  --trials-p1 <int>      Trials per worker for Phase 1 (default: $TRIALS_P1)"
    echo "  --trials-p2 <int>      Trials per worker for Phase 2 (default: $TRIALS_P2)"
    echo "  --trials-p3 <int>      Trials per worker for Phase 3 (default: $TRIALS_P3)"
    echo "  --phase <1|2|3|all>    Run only this phase, or all (default: all)"
    echo "  --setup                Only create conda environment, don't train"
    echo "  -h, --help             Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                  # 5 workers, all phases"
    echo "  $0 --workers 10 --phase 1           # 10 workers, phase 1 only"
    echo "  $0 --workers 20 --trials-p2 10      # 20 workers x 10 trials = 200 total for P2"
}

# -----------------------------
# Argument Parsing
# -----------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers) NUM_WORKERS="$2"; shift 2 ;;
        --trials-p1) TRIALS_P1="$2"; shift 2 ;;
        --trials-p2) TRIALS_P2="$2"; shift 2 ;;
        --trials-p3) TRIALS_P3="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --setup) SETUP_ONLY=true; shift ;;
        -h|--help) display_help; exit 0 ;;
        *) echo "Unknown option: $1"; display_help; exit 1 ;;
    esac
done

# -----------------------------
# Validation
# -----------------------------
if [ ! -f "$SLURM_SCRIPT" ]; then
    log "SLURM script not found at $SLURM_SCRIPT"
    exit 1
fi

# -----------------------------
# Conda Environment Setup
# -----------------------------
log "=============================================================="
log "Q-HAS HPC Training Orchestrator"
log "  Workers per phase: $NUM_WORKERS"
log "  Trials:  P1=$TRIALS_P1  P2=$TRIALS_P2  P3=$TRIALS_P3"
log "  Phase:   $PHASE"
log "=============================================================="

if [ -f "$ROOT_DIR/$ENV_FILE" ]; then
    ENV_NAME=$(grep '^name:' "$ROOT_DIR/$ENV_FILE" | awk '{print $2}')

    # Check if env already exists
    if conda env list 2>/dev/null | grep -q "$ENV_NAME"; then
        log "Conda environment '$ENV_NAME' already exists."
    else
        log "Creating conda environment '$ENV_NAME' from $ENV_FILE..."
        conda env create -f "$ROOT_DIR/$ENV_FILE"
        log "Conda environment '$ENV_NAME' created successfully."
    fi
else
    log "$ENV_FILE not found. Make sure the conda environment is set up manually."
fi

if [ "$SETUP_ONLY" = true ]; then
    log "Setup complete. Exiting (--setup flag was set)."
    exit 0
fi

# -----------------------------
# Submit Phase Helper
# -----------------------------
submit_phase() {
    local phase_num="$1"
    local trials="$2"
    local max_idx=$((NUM_WORKERS - 1))

    log "--------------------------------------------------------------"
    log "Submitting Phase $phase_num: $NUM_WORKERS workers x $trials trials"
    log "  Total trials target: $((NUM_WORKERS * trials))"
    log "--------------------------------------------------------------"

    local job_id
    job_id=$(sbatch \
        --array=0-"$max_idx" \
        --export=PHASE="$phase_num",TRIALS="$trials" \
        --parsable \
        "$SLURM_SCRIPT")

    log "Phase $phase_num submitted — SLURM Job ID: $job_id"
    log "  Monitor: squeue -u $USER"
    log "  Logs:    tail -f $LOG_DIR/qhas_${job_id}_0.out"

    echo "$job_id"
}

wait_for_job() {
    local job_id="$1"
    local phase_num="$2"

    log "Waiting for Phase $phase_num (job $job_id) to complete..."

    while true; do
        # Check if any tasks from this job array are still running/pending
        local remaining
        remaining=$(squeue -j "$job_id" -h 2>/dev/null | wc -l)

        if [ "$remaining" -eq 0 ]; then
            break
        fi

        sleep 30
    done

    log "Phase $phase_num (job $job_id) completed."
}

# -----------------------------
# Pipeline Execution
# -----------------------------

run_phase() {
    local phase_num="$1"
    local trials="$2"

    local job_id
    job_id=$(submit_phase "$phase_num" "$trials")
    wait_for_job "$job_id" "$phase_num"
}

if [ "$PHASE" = "all" ]; then
    run_phase 1 "$TRIALS_P1"
    run_phase 2 "$TRIALS_P2"
    run_phase 3 "$TRIALS_P3"

elif [ "$PHASE" = "1" ]; then
    run_phase 1 "$TRIALS_P1"
elif [ "$PHASE" = "2" ]; then
    run_phase 2 "$TRIALS_P2"
elif [ "$PHASE" = "3" ]; then
    run_phase 3 "$TRIALS_P3"
else
    log "Invalid phase: $PHASE (expected 1, 2, 3, or all)"
    exit 1
fi

# -----------------------------
# Completion
# -----------------------------
log "=============================================================="
log "Q-HAS HPC Training completed!"
log "  Results: Train_results/journal/"
log "  Full log: $LOG_FILE"
log "=============================================================="
exit 0
