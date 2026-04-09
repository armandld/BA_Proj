#!/bin/bash
#SBATCH --job-name=qhas-train
#SBATCH --array=0-4                    # 5 workers in parallel (adjust as needed)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qhas_%A_%a.out
#SBATCH --error=logs/qhas_%A_%a.err

# ============================================================
# Q-HAS Distributed Hyperparameter Training — Imperial HPC
# ============================================================
#
# Usage:
#   # Phase 1 (5 workers, 50 trials each)
#   sbatch --export=PHASE=1,TRIALS=50 hpc/submit_training.sh
#
#   # Phase 2 (after phase 1 is done)
#   sbatch --export=PHASE=2,TRIALS=50 hpc/submit_training.sh
#
#   # Phase 3
#   sbatch --export=PHASE=3,TRIALS=30 hpc/submit_training.sh
#
#   # Full sequential run (single worker, all phases)
#   sbatch --array=0 --export=PHASE=all hpc/submit_training.sh
#
# Monitor:
#   squeue -u $USER                     # see running jobs
#   tail -f logs/qhas_<jobid>_0.out     # follow worker 0
#   scancel <jobid>                     # cancel all workers
# ============================================================

set -euo pipefail

# ── Paths ──
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOURNAL_DIR="${REPO_DIR}/Train_results/journal"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "$JOURNAL_DIR" "$LOG_DIR"

# ── Environment setup ──
# Imperial HPC uses 'module load' — adjust the module name to what's
# available on your cluster (run 'module avail anaconda' to check).
module load anaconda3/personal 2>/dev/null \
    || module load anaconda3 2>/dev/null \
    || module load python/3.11 2>/dev/null \
    || echo "[WARN] No Python module found — using system Python"

# Activate the conda environment (create it first with:
#   conda env create -f environment.yaml)
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate qiskit-project 2>/dev/null \
    || echo "[WARN] Could not activate conda env 'qiskit-project'"

# ── Distributed config ──
# OPTUNA_JOURNAL tells TrainHyperParam.py to use JournalFileStorage
# instead of SQLite — safe for concurrent writes over NFS.
export OPTUNA_JOURNAL="$JOURNAL_DIR"

# Phase and trials come from --export or default to all
export WORKER_PHASE="${PHASE:-}"
if [ "$WORKER_PHASE" = "all" ]; then
    unset WORKER_PHASE
fi

if [ -n "${TRIALS:-}" ]; then
    export WORKER_TRIALS="$TRIALS"
fi

# ── Run ──
echo "============================================================"
echo "Q-HAS Training — Worker ${SLURM_ARRAY_TASK_ID:-0} / Job ${SLURM_ARRAY_JOB_ID:-local}"
echo "  Phase:   ${WORKER_PHASE:-all (sequential)}"
echo "  Trials:  ${WORKER_TRIALS:-all remaining}"
echo "  Journal: $JOURNAL_DIR"
echo "  Host:    $(hostname)"
echo "  Date:    $(date)"
echo "============================================================"

cd "$REPO_DIR/src"
python TrainHyperParam.py

echo "[DONE] Worker ${SLURM_ARRAY_TASK_ID:-0} finished at $(date)"
