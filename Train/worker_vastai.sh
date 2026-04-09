#!/usr/bin/env bash
# ============================================================
#  Q-HAS Distributed Worker — Vast.ai / Any Linux VM
#
#  Usage:
#    export OPTUNA_STORAGE="postgresql://user:pass@host/db?sslmode=require"
#    bash worker_vastai.sh 1 50    # Phase 1, 50 trials
#    bash worker_vastai.sh 2 100   # Phase 2, 100 trials
# ============================================================
set -e

PHASE="${1:-1}"
TRIALS="${2:-50}"
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/BA_Proj.git}"
BRANCH="${BRANCH:-main}"

if [ -z "$OPTUNA_STORAGE" ]; then
    echo "ERROR: Set OPTUNA_STORAGE env var first."
    echo "  export OPTUNA_STORAGE=\"postgresql://user:pass@host/db?sslmode=require\""
    exit 1
fi

echo "=== Q-HAS Worker ==="
echo "Phase: $PHASE | Trials: $TRIALS"
echo "Storage: ${OPTUNA_STORAGE##*@}"

# Install system deps if needed
if ! python3 -c "import optuna" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install --quiet optuna psycopg2-binary qiskit qiskit-aer numpy scipy
fi

# Clone or update repo
if [ ! -d "/workspace/BA_Proj" ]; then
    git clone -b "$BRANCH" "$REPO_URL" /workspace/BA_Proj
else
    cd /workspace/BA_Proj && git pull origin "$BRANCH"
fi

# Run worker
cd /workspace/BA_Proj/src
export WORKER_PHASE="$PHASE"
export WORKER_TRIALS="$TRIALS"

echo "Starting worker..."
python3 TrainHyperParam.py
echo "Worker done."
