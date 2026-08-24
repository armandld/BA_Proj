"""Phase-10 artifacts expose the executed protocol and untouched test split."""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in [os.path.join(ROOT, "src")] + [
        os.path.join(ROOT, "study", folder) for folder in (
            "pipeline", "h2b_prediction", "common")]:
    if path not in sys.path:
        sys.path.insert(0, path)

DNS = os.path.join(ROOT, "results", "dns_orszag_tang_Re400_N64.npz")
PATCHES = os.path.join(
    ROOT, "results", "patches_orszag_tang_Re400_N64_dim2.npz")


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    if not (os.path.exists(DNS) and os.path.exists(PATCHES)):
        pytest.skip("phase-10 inputs are absent")

    import h2b_train_linear_hamiltonian as phase10

    out = tmp_path_factory.mktemp("phase10")
    for source in (DNS, PATCHES):
        os.symlink(source, os.path.join(out, os.path.basename(source)))

    previous_dir, previous_argv = phase10.RESULTS_DIR, sys.argv
    phase10.RESULTS_DIR = str(out)
    sys.argv = [
        "phase10", "--modes", "joint", "--n-iters", "3",
        "--sweeps", "20", "--n-restarts", "1", "--dim", "2",
        "--N", "64", "--max-batch", "2", "--max-val", "2",
        "--max-test", "2", "--scenario", "orszag_tang", "--re", "400",
        "--analytical-init", "none", "--optimiser", "nelder-mead",
    ]
    try:
        phase10.main()
    finally:
        phase10.RESULTS_DIR, sys.argv = previous_dir, previous_argv

    return (
        np.load(out / "train_joint_N64_dim2.npz", allow_pickle=False),
        np.load(out / "train_COMPARE_N64_dim2.npz", allow_pickle=False),
        phase10,
    )


def test_run_artifact_names_the_optimizer_that_ran(artifacts):
    run, compare, _ = artifacts
    assert str(run["optimiser"]) == "nelder-mead"
    assert set(map(str, compare["optimiser"])) == {"nelder-mead"}


def test_run_artifact_separates_selection_from_final_test(artifacts):
    run, compare, _ = artifacts
    assert str(run["split_strategy"]) == "chronological_per_configuration"
    assert set(run["train_pairs"][:, 1]).isdisjoint(run["val_pairs"][:, 1])
    assert set(run["train_pairs"][:, 1]).isdisjoint(run["test_pairs"][:, 1])
    assert set(run["val_pairs"][:, 1]).isdisjoint(run["test_pairs"][:, 1])
    assert "best_f1_test" in run.files
    assert "f1_test" in compare.files
    assert float(run["delta_test"]) == pytest.approx(
        float(run["best_f1_test"] - run["classical_f1_test"]))


def test_run_artifact_records_seed_arguments_and_git_state(artifacts):
    run, compare, _ = artifacts
    args = json.loads(str(run["cli_args"]))
    assert args["seed"] == int(run["seed"]) == 0
    for artifact in (run, compare):
        assert "git_hash_at_start" in artifact.files
        assert "dirty_at_start" in artifact.files


def test_requested_cma_cannot_silently_fall_back(monkeypatch, artifacts):
    _, _, phase10 = artifacts
    monkeypatch.setattr(phase10, "HAS_CMA", False)
    with pytest.raises(RuntimeError, match="not installed"):
        phase10._load_cma()
