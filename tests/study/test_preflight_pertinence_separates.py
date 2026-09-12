"""La porte de pertinence compare le coefficient à la baseline classique."""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "study", "common"), _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import preflight_coefficients as P  # noqa: E402


@pytest.mark.parametrize(
    "nominal,baseline,best,expected",
    [
        (0.75, 0.70, 0.72, True),
        (0.75, 0.80, 0.805, False),
        (0.55, 0.40, 0.90, False),
        (np.nan, 0.40, 0.90, False),
        (0.75, np.nan, 0.90, False),
    ],
)
def test_relevance_verdict_requires_incremental_value(
        nominal, baseline, best, expected):
    assert P.relevance_is_sufficient(nominal, baseline, best) is expected


def test_margin_is_strict():
    baseline = 0.80
    assert not P.relevance_is_sufficient(
        0.70, baseline, baseline + P.RELEVANCE_MARGIN)


@pytest.mark.slow
def test_real_control_finds_an_attainable_improvement():
    ok, measured = P.controle_pertinence()

    assert measured["rho_nominal"] > 0.6
    assert measured["rho_best_probe"] > (
        measured["rho_classical"] + measured["required_margin"])
    assert measured["gap_best_vs_classical"] == pytest.approx(
        measured["rho_best_probe"] - measured["rho_classical"])
    assert measured["n_probes"] == P.RELEVANCE_PROBES
    assert ok

    for name, value in measured["best_probe_params"].items():
        low, high, _ = P.SEARCH_SPACE[name]
        assert low <= value <= high, (name, value, low, high)
