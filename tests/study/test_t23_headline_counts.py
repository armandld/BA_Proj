"""Matched-budget headline counts are derived from paired runs only."""

import json

import pytest

from closed_loop_headline_counts import fold_counts, totals


def _run(delta, *, completed=True, converged=True):
    return {
        "completed": completed,
        "qaoa_seed": 0,
        "phys_score": 0.2,
        "patch_ratio": 0.4,
        "budget_match": ({
            "converged": converged,
            "delta_phys": delta,
            "classical": {"phys_score": 0.2 - delta,
                          "patch_ratio": 0.4},
        } if completed else None),
    }


def _write(tmp_path, fold, runs, status="complete"):
    for seed, run in enumerate(runs):
        run.setdefault("physics_seed", seed)
    payload = {
        "schema": 2,
        "replication_unit": "trajectory",
        "status": status,
        "qhas_runs": runs,
        "parent_campaign_contract_sha256": "a" * 64,
    }
    (tmp_path / f"t20_qhas_run_variance_{fold}.json").write_text(
        json.dumps(payload), encoding="utf-8")


def test_fold_counts_excludes_aborted_and_unmatched_runs(tmp_path):
    _write(tmp_path, "ot", [
        _run(-0.1), _run(0.2), _run(0.0),
        _run(0.3, completed=False), _run(-0.2, converged=False),
    ])
    row = fold_counts(str(tmp_path), "ot")
    assert row["n_runs"] == 5
    assert row["n_completed"] == 4
    assert row["n_aborted"] == 1
    assert row["n_paired"] == 3
    assert row["n_unmatched"] == 1
    assert row["qhas_lower_error"] == 1
    assert row["classical_lower_error"] == 1
    assert row["ties"] == 1


def test_obsolete_single_reference_schema_is_refused(tmp_path):
    (tmp_path / "t20_qhas_run_variance_ot.json").write_text(
        json.dumps({"qhas_runs": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="obsolete schema"):
        fold_counts(str(tmp_path), "ot")


def test_quantum_seed_must_be_fixed_across_physics_trajectories(tmp_path):
    runs = [_run(-0.1), _run(0.2)]
    runs[1]["qaoa_seed"] = 1
    _write(tmp_path, "ot", runs)
    with pytest.raises(RuntimeError, match="QAOA seed fixed"):
        fold_counts(str(tmp_path), "ot")


def test_fold_inference_uses_trajectory_bootstrap(tmp_path):
    _write(tmp_path, "ot", [_run(0.1), _run(0.2), _run(0.3)])
    row = fold_counts(str(tmp_path), "ot", n_boot=500, seed=7)
    assert row["inference"]["n_trajectories"] == 3
    assert row["inference"]["ci_low"] > 0
    assert row["inference"]["classical_confirmed"]
    assert row["inference"]["sign_flip_p"] == pytest.approx(0.25)


def test_totals_weight_the_mean_by_paired_run_count():
    rows = [
        {"n_runs": 2, "n_completed": 2, "n_aborted": 0,
         "n_paired": 2, "n_unmatched": 0, "qhas_lower_error": 2,
         "classical_lower_error": 0, "ties": 0, "mean_delta_phys": -0.1},
        {"n_runs": 1, "n_completed": 1, "n_aborted": 0,
         "n_paired": 1, "n_unmatched": 0, "qhas_lower_error": 0,
         "classical_lower_error": 1, "ties": 0, "mean_delta_phys": 0.2},
    ]
    result = totals(rows)
    assert result["n_paired"] == 3
    assert result["mean_delta_phys"] == pytest.approx(0.0)
