"""The rented-machine instructions and launchers must agree."""
import os
import re
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANUAL = os.path.join(ROOT, "docs", "MODE_EMPLOI_CAMPAGNE.md")
LAUNCHER = os.path.join(ROOT, "scripts", "run_rented_campaign.sh")
WORKER = os.path.join(ROOT, "scripts", "run_reoptimisation.sh")


def _manual():
    return open(MANUAL, encoding="utf-8").read()


def test_requirements_cover_the_campaign_stack():
    path = os.path.join(ROOT, "requirements.txt")
    content = open(path, encoding="utf-8").read().lower()
    for package in ("optuna", "qiskit", "qiskit-aer", "numpy", "scipy",
                    "scikit-learn", "cma", "qiskit-machine-learning"):
        assert package in content


def test_every_script_named_by_the_manual_exists():
    named = set(re.findall(r"scripts/[A-Za-z0-9_]+\.(?:sh|py)", _manual()))
    assert named
    missing = [path for path in sorted(named)
               if not os.path.exists(os.path.join(ROOT, path))]
    assert not missing


def test_manual_and_launchers_have_no_removed_platform_paths():
    combined = _manual() + open(LAUNCHER, encoding="utf-8").read() + \
        open(WORKER, encoding="utf-8").read()
    forbidden = ("#PBS", "#SBATCH", "sbatch", "qsub", "SLURM",
                 "PostgreSQL", "OPTUNA_STORAGE", "OPTUNA_JOURNAL",
                 "Google Colab", "Google Drive")
    assert not [token for token in forbidden if token.lower() in combined.lower()]


def test_launchers_parse_and_separate_global_budget_from_workers():
    check = subprocess.run(
        ["bash", "-n", LAUNCHER, WORKER], capture_output=True,
        text=True, timeout=30)
    assert check.returncode == 0, check.stderr
    launcher = open(LAUNCHER, encoding="utf-8").read()
    worker = open(WORKER, encoding="utf-8").read()
    assert 'TARGET_TRIALS="${2:-600}"' in launcher
    assert 'N_WORKERS="${1:-$DEFAULT_WORKERS}"' in launcher
    assert '--n-trials "$TARGET_TRIALS"' in worker
    assert 'WORKER_TRIALS="${WORKER_TRIALS:-$TARGET_TRIALS}"' in worker


def test_launcher_rejects_invalid_capacity_without_starting_workers():
    result = subprocess.run(
        ["bash", LAUNCHER, "0", "10", "0"], capture_output=True,
        text=True, timeout=30, cwd=ROOT)
    assert result.returncode == 2
