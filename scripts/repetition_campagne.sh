#!/usr/bin/env bash
# Fast rehearsal of the journal, shared budget, and interruption recovery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export QHAS_CAMPAIGN_DIR="$TMP_DIR"
export QHAS_JOURNAL_DIR="$TMP_DIR/journal"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - <<'PY'
import multiprocessing as mp
import optuna
import train_hyperparams as training

optuna.logging.set_verbosity(optuna.logging.WARNING)
config = {"study_name": "rented_machine_rehearsal", "n_trials": 8}


def objective(trial):
    return trial.suggest_float("x", 0.0, 1.0)


def worker(seed):
    training.WORKER_TRIALS = None
    training.run_phase("rehearsal", config, objective, seed=seed)


context = mp.get_context("fork")
workers = [context.Process(target=worker, args=(seed,)) for seed in (10, 11)]
for process in workers:
    process.start()
for process in workers:
    process.join()
assert all(process.exitcode == 0 for process in workers)

study = optuna.load_study(
    study_name=config["study_name"], storage=training._get_storage(config))
assert training.trials_done(study) == config["n_trials"]
assert not [trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.RUNNING]
assert len({trial.number for trial in study.trials}) == len(study.trials)

study.ask()
assert training.fail_interrupted_trials(study) == 1
assert training.trials_done(study) == config["n_trials"]
print("Rehearsal passed: concurrent journal, global budget, resume recovery.")
PY
