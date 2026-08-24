"""Contracts for the single-machine campaign workflow."""
import os
import subprocess

import optuna

import train_hyperparams as training


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
LAUNCHER = os.path.join(SCRIPTS, "run_rented_campaign.sh")
WORKER = os.path.join(SCRIPTS, "run_reoptimisation.sh")


def test_campaign_scripts_are_valid_bash():
    result = subprocess.run(
        ["bash", "-n", LAUNCHER, WORKER,
         os.path.join(SCRIPTS, "repetition_campagne.sh")],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_launcher_is_one_machine_and_one_journal():
    source = open(LAUNCHER, encoding="utf-8").read()
    assert "QHAS_JOURNAL_DIR" in source
    assert "OMP_NUM_THREADS=1" in source
    assert "--prepare-only" in source and "--finalize-only" in source
    forbidden = ("#PBS", "#SBATCH", "sbatch", "qsub", "postgresql://",
                 "OPTUNA_STORAGE", "OPTUNA_JOURNAL", "google.colab")
    assert not [token for token in forbidden if token in source]


def test_invalid_worker_arguments_fail_before_any_campaign_work():
    result = subprocess.run(
        ["bash", WORKER, "0", "0"], capture_output=True, text=True,
        timeout=30, cwd=ROOT)
    assert result.returncode == 2


def test_interrupted_journal_trial_is_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "data_dir", str(tmp_path))
    monkeypatch.setattr(training, "JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.setattr(training, "_DIRS_READY", False)
    config = {"study_name": "interrupted", "n_trials": 2}
    study = optuna.create_study(
        study_name=config["study_name"],
        storage=training._get_storage(config), load_if_exists=True)
    study.ask()
    assert training.trials_done(study) == 1
    assert training.fail_interrupted_trials(study) == 1
    assert training.trials_done(study) == 0
