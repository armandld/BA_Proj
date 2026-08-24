"""Degenerate V2 mean-field sweeps cannot seed closed-loop training."""

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
    F1_SPAN_TOL,
    mean_over_informative,
    summarize_curve,
)
from h2b_train_linear_hamiltonian import build_init_map  # noqa: E402


def test_curve_span_separates_flat_and_informative_sweeps():
    grid = np.logspace(-1, 5, 7)
    flat = summarize_curve(np.full(7, 0.4), grid)
    informative = summarize_curve([0.1, 0.2, 0.5, 0.8, 0.8, 0.8, 0.8], grid)
    assert flat["f1_span"] == 0.0 and flat["degenerate"] is True
    assert informative["f1_span"] > F1_SPAN_TOL
    assert informative["degenerate"] is False


def test_degenerate_rows_do_not_bias_aggregates():
    rows = [
        {"c_bias_star": 250.0, "degenerate": False},
        {"c_bias_star": 1000.0, "degenerate": False},
        {"c_bias_star": 0.1, "degenerate": True},
    ]
    assert mean_over_informative(rows, "c_bias_star") == pytest.approx(625.0)


def test_all_degenerate_rows_produce_no_plausible_number():
    rows = [{"c_bias_star": 0.1, "degenerate": True} for _ in range(4)]
    assert np.isnan(mean_over_informative(rows, "c_bias_star"))


def test_closed_loop_initialization_skips_degenerate_and_nan_rows():
    init, skipped = build_init_map(
        ["cfg:bad", "scenario:missing", "scenario:good"],
        [0.2, np.nan, 0.69],
        [0.1, np.nan, 251.0],
        [True, True, False],
    )
    assert skipped == 2
    assert set(init) == {"scenario:good"}


def test_non_degenerate_low_bias_value_is_not_silently_discarded():
    init, skipped = build_init_map(["cfg:valid"], [0.2], [0.1], [False])
    assert skipped == 0
    assert init["cfg:valid"] == pytest.approx([-1.0, 0.2])
