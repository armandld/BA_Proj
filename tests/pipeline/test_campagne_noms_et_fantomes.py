"""Contracts for the single-machine campaign workflow."""
import json
import os
import subprocess

import optuna

import train_hyperparams as training
from hyperparams_loader import load_hyperparams


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


def _fake_campaign_result(quantum_params, classical_params):
    return {
        "provenance": {"git_commit": "deadbeef", "git_dirty": False},
        "quantum": {"phase3": {"best_params": quantum_params}},
        "classical": {"phase3": {"best_params": classical_params}},
        "deploy": {"quantum": quantum_params, "classical": classical_params},
    }


def test_deploy_writes_where_hyperparams_loader_actually_reads(
        tmp_path, monkeypatch):
    """D-22 : `_save_results` et `hyperparams_loader` lisaient deux chemins
    DIFFERENTS par defaut (`CAMPAIGN_DIR/best_hyperparams.json` contre
    `results/hyperparams/best_hyperparams.json`) ; rien ne copiait l'un vers
    l'autre. Une campagne pouvait tourner jusqu'au bout sans que `study/`
    n'en voie jamais le resultat. Round-trip par le VRAI `load_hyperparams`,
    pas seulement une comparaison de chemins : c'est la garantie qui compte."""
    staged = tmp_path / "staged" / "best_hyperparams.json"
    staged.parent.mkdir()
    quantum_params = {"beta": 0.31, "threshold_amr": 0.19}
    classical_params = {"threshold_amr": 0.22}
    staged.write_text(json.dumps(
        _fake_campaign_result(quantum_params, classical_params)))

    deploy_target = tmp_path / "deployed" / "best_hyperparams.json"
    monkeypatch.setenv("QHAS_HYPERPARAMS_PATH", str(deploy_target))

    returned_path = training._deploy(str(staged))

    assert os.path.abspath(returned_path) == os.path.abspath(str(deploy_target))
    assert deploy_target.exists()
    assert load_hyperparams(method="quantum") == quantum_params
    assert load_hyperparams(method="classical") == classical_params


def test_deploy_is_a_noop_when_staged_and_deploy_paths_already_coincide(
        tmp_path, monkeypatch):
    """If `CAMPAIGN_DIR` is ever pointed straight at the deploy path (e.g. a
    manual `QHAS_CAMPAIGN_DIR` override), `_deploy` must not fail trying to
    copy a file onto itself."""
    same_path = tmp_path / "best_hyperparams.json"
    same_path.write_text(json.dumps(
        _fake_campaign_result({"beta": 0.1}, {"threshold_amr": 0.2})))
    monkeypatch.setenv("QHAS_HYPERPARAMS_PATH", str(same_path))

    returned_path = training._deploy(str(same_path))

    assert os.path.abspath(returned_path) == os.path.abspath(str(same_path))


def test_main_all_phase_deploys_by_default_but_no_deploy_flag_skips_it():
    """The CLI wiring, not just the function: `--phase all` must reach
    `_deploy` unless `--no-deploy` is passed."""
    import inspect
    source = inspect.getsource(training.main)
    save_idx = source.index("_save_results(")
    tail = source[save_idx:]
    assert "_deploy(" in tail, (
        "main() calls _save_results but never _deploy() afterwards — a "
        "completed campaign would stay orphaned again (D-22)")
    assert "no_deploy" in tail
