"""Budget matching never promotes a failed evaluation to a conclusion."""

import numpy as np
import pytest

import closed_loop_budget_matched as module


def _run_with_failed_low_endpoint(
        _T, _key, _cfg, _dns, hyperparams, _is_classical,
        lambda_cost=None, verbose=False, seed=0):
    threshold = hyperparams["threshold_amr"]
    if threshold == pytest.approx(0.05):
        return {
            "patch_ratio": np.nan, "phys_score": np.nan,
            "combined": np.nan, "wall_s": 0.0,
        }
    patch = 1.0 - threshold
    return {
        "patch_ratio": patch, "phys_score": 1.0 - patch,
        "combined": 0.0, "wall_s": 0.0,
    }


def test_failed_endpoint_is_excluded_from_the_best_candidate(monkeypatch):
    monkeypatch.setattr(module, "run_arm", _run_with_failed_low_endpoint)
    best, trace = module.bisect_threshold_for_budget(
        None, "case", None, None, {}, target_patch=0.5,
        max_iter=4, tol=0.02,
    )
    assert any(not np.isfinite(row["patch_ratio"]) for row in trace)
    assert np.isfinite(best["patch_ratio"])
    assert abs(best["patch_ratio"] - 0.5) <= 0.02


def test_no_finite_candidate_raises_instead_of_writing_nan(monkeypatch):
    monkeypatch.setattr(
        module, "run_arm",
        lambda *args, **kwargs: {
            "patch_ratio": np.nan, "phys_score": np.nan,
            "combined": np.nan, "wall_s": 0.0,
        },
    )
    with pytest.raises(RuntimeError, match="no finite evaluation"):
        module.bisect_threshold_for_budget(
            None, "case", None, None, {}, target_patch=0.5,
        )


def test_non_converged_match_has_no_scientific_conclusion():
    message = module.budget_match_reading(delta_phys=-0.3, converged=False)
    assert message.startswith("INCONCLUSIVE")


def test_converged_match_can_support_either_registered_reading():
    assert "Q-HAS" in module.budget_match_reading(-0.02, True)
    assert "equal" in module.budget_match_reading(0.0, True)
    assert "classical" in module.budget_match_reading(0.02, True)
