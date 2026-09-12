"""Well-posed search bounds for the V2 mean-field initialization."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in [os.path.join(ROOT, "src")] + [
        os.path.join(ROOT, "study", folder) for folder in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if path not in sys.path:
        sys.path.insert(0, path)

from h2b_analytical_solution import (  # noqa: E402
    best_threshold,
    c_bias_grid,
    require_interior_optima,
    summarize_curve,
)
from h2b_train_linear_hamiltonian import (  # noqa: E402
    THETA_BOUNDS,
    build_init_map,
    build_init_map_from_artifact,
    chronological_split_indices,
    decode_theta,
    evenly_subsample,
    hits_bound,
)


def test_analytical_grid_and_optimizer_share_one_search_box():
    grid = c_bias_grid()
    assert np.log10(grid[[0, -1]]) == pytest.approx(THETA_BOUNDS[0])
    assert decode_theta(THETA_BOUNDS[:, 0])[0] == pytest.approx(grid[0])
    assert decode_theta(THETA_BOUNDS[:, 1])[0] == pytest.approx(grid[-1])


def test_score_threshold_bounds_cover_the_full_score_domain():
    assert THETA_BOUNDS[1] == pytest.approx([0.0, 1.0])
    scores = np.linspace(0.0, 1.0, 200)
    threshold, _ = best_threshold(scores, scores > 0.69)
    assert THETA_BOUNDS[1, 0] <= threshold <= THETA_BOUNDS[1, 1]


def test_threshold_search_is_exact_for_observed_score_breakpoints():
    scores = np.array([0.1, 0.2, 0.21, 0.9])
    threshold, f1 = best_threshold(scores, [False, False, True, True])
    assert threshold == pytest.approx(0.205)
    assert f1 == pytest.approx(1.0)


def test_flat_curve_is_degenerate_not_an_optimum():
    summary = summarize_curve(np.full(7, 0.4), np.logspace(-1, 5, 7))
    assert summary["degenerate"] is True
    assert summary["at_left_edge"] is True


def test_informative_edge_curve_is_rejected_before_artifact_creation():
    row = dict(
        scenario="rotor",
        Re=400,
        **summarize_curve(np.linspace(0.1, 0.8, 7), np.logspace(-1, 5, 7)),
    )
    with pytest.raises(RuntimeError, match="grid edge"):
        require_interior_optima([row])


def test_short_right_plateau_does_not_masquerade_as_bias_only_limit():
    grid = np.logspace(-1, 1, 9)
    summary = summarize_curve(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8], grid)
    assert summary["right_plateau_decades"] < 1.0
    assert summary["bias_only_limit"] is False
    with pytest.raises(RuntimeError, match="grid edge"):
        require_interior_optima([dict(scenario="kh", Re=400, **summary)])


def test_interior_plateau_selects_the_smallest_maximizer():
    grid = np.logspace(-1, 5, 7)
    summary = summarize_curve([0.1, 0.2, 0.5, 0.8, 0.8, 0.8, 0.8], grid)
    assert summary["c_bias_star"] == pytest.approx(grid[3])
    assert summary["at_left_edge"] is False
    assert summary["at_right_edge"] is True
    assert summary["bias_only_limit"] is True
    assert summary["c_bias_identifiable"] is False
    assert summary["right_plateau_decades"] == pytest.approx(3.0)
    require_interior_optima([dict(scenario="ot", Re=400, **summary)])


def test_disconnected_maximum_is_not_counted_as_part_of_the_right_plateau():
    grid = np.logspace(-1, 5, 7)
    summary = summarize_curve([0.1, 0.8, 0.2, 0.3, 0.8, 0.8, 0.8], grid)
    assert summary["right_plateau_start_index"] == 4
    assert summary["right_plateau_decades"] == pytest.approx(2.0)


def test_chronological_split_and_subsample_never_cross_blocks():
    train, validation, test = chronological_split_indices(30, 0.6, 0.2)
    assert np.array_equal(train, np.arange(18))
    assert np.array_equal(validation, np.arange(18, 24))
    assert np.array_equal(test, np.arange(24, 30))
    assert np.array_equal(evenly_subsample(train, 3), [0, 8, 17])


def test_degenerate_edge_does_not_trigger_the_edge_guard():
    summary = summarize_curve(np.zeros(7), np.logspace(-1, 5, 7))
    require_interior_optima([dict(scenario="harris", Re=400, **summary)])


def test_analytical_initialization_is_interior_for_the_observed_plateau(capsys):
    init, skipped = build_init_map(
        ["scenario:harris_tearing"], [0.69], [251.0], [False])
    theta = init["scenario:harris_tearing"]
    assert skipped == 0
    assert decode_theta(theta) == pytest.approx([251.0, 0.69])
    assert hits_bound(theta) is False
    assert "RABOTE" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [(-1.0, 10.0, 5), (10.0, 1.0, 5), (0.1, 10.0, 2)],
)
def test_invalid_grids_are_rejected(args):
    with pytest.raises(ValueError):
        c_bias_grid(*args)


def _artifact(**updates):
    artifact = {
        "tags": np.array(["scenario:ot"]),
        "thr_star": np.array([0.2]),
        "c_bias_star": np.array([1000.0]),
        "degenerate": np.array([False]),
        "theta_bounds": THETA_BOUNDS.copy(),
        "at_left_edge": np.array([False]),
        "at_right_edge": np.array([True]),
        "bias_only_limit": np.array([True]),
        "split_strategy": np.array("chronological_per_configuration"),
        "train_fraction": np.array(0.6),
        "validation_fraction": np.array(0.2),
    }
    artifact.update(updates)
    return artifact


def test_bias_only_plateau_is_a_valid_analytical_initialization():
    init, skipped = build_init_map_from_artifact(_artifact())
    assert skipped == 0
    assert init["scenario:ot"] == pytest.approx([3.0, 0.2])


def test_obsolete_or_unresolved_analytical_artifacts_are_rejected():
    with pytest.raises(RuntimeError, match="obsolete"):
        build_init_map_from_artifact({"tags": np.array(["scenario:ot"])})
    with pytest.raises(RuntimeError, match="different search bounds"):
        build_init_map_from_artifact(
            _artifact(theta_bounds=np.array([[-1.0, 2.0], [0.02, 0.6]])))
    with pytest.raises(RuntimeError, match="unresolved edge"):
        build_init_map_from_artifact(
            _artifact(bias_only_limit=np.array([False])))
    with pytest.raises(RuntimeError, match="different temporal split"):
        build_init_map_from_artifact(
            _artifact(train_fraction=np.array(0.7)))
