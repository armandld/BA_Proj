"""Tests V3 Task 2 : exemples 6 patches calcules a la main + planchers
verifies analytiquement (critere d'acceptation du protocole)."""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os
import sys

import numpy as np
import pytest
from sklearn.metrics import f1_score

from metrics import (

    captured_error_at_budget,
    ce_curve,
    degeneracy_flag,
    degeneracy_floors,
    spearman,
)

# exemple 6 patches calcule a la main :
# classement par score : idx 0 (e=5), 1 (e=1), 3 (e=0), 5 (e=1),
#                        4 (e=1), 2 (e=2) ; somme e = 10
SCORES6 = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.3])
E6 = np.array([5.0, 1.0, 2.0, 0.0, 1.0, 1.0])


# ------------------------- CE(b) a la main ---------------------------

def test_ce_budgets_hand_computed():
    ce, _ = captured_error_at_budget(SCORES6, E6)
    # b=0.10 -> ceil(0.6)=1 patch  -> 5/10
    # b=0.25 -> ceil(1.5)=2 patches -> 6/10
    # b=0.50 -> ceil(3.0)=3 patches -> 6/10
    assert ce[0.10] == pytest.approx(0.5)
    assert ce[0.25] == pytest.approx(0.6)
    assert ce[0.50] == pytest.approx(0.6)


def test_ce_curve_hand_computed():
    curve = ce_curve(SCORES6, E6)
    np.testing.assert_allclose(curve, [0.5, 0.6, 0.6, 0.7, 0.8, 1.0])


def test_ce_auc_hand_computed():
    # trapezes avec CE(0)=0 :
    # (1/12) * [(0+.5)+(.5+.6)+(.6+.6)+(.6+.7)+(.7+.8)+(.8+1)] = 7.4/12
    _, auc = captured_error_at_budget(SCORES6, E6)
    assert auc == pytest.approx(7.4 / 12)


def test_ce_perfect_ranking_dominates():
    rng = np.random.default_rng(0)
    e = rng.exponential(size=50)
    _, auc_perfect = captured_error_at_budget(e, e)  # score = e
    _, auc_shuffled = captured_error_at_budget(rng.permutation(e), e)
    assert auc_perfect >= auc_shuffled
    curve = ce_curve(e, e)
    assert np.all(np.diff(curve) >= -1e-12)  # cumulatif croissant
    assert curve[-1] == pytest.approx(1.0)


def test_ce_uniform_errors_equal_budget_fraction():
    # e uniforme -> CE(b) = ceil(b*n)/n quel que soit le classement
    n = 8
    scores = np.random.default_rng(1).normal(size=n)
    ce, _ = captured_error_at_budget(scores, np.ones(n))
    for b, v in ce.items():
        assert v == pytest.approx(np.ceil(b * n) / n)


def test_ce_zero_total_error_is_nan():
    ce, auc = captured_error_at_budget(SCORES6, np.zeros(6))
    assert all(np.isnan(v) for v in ce.values())
    assert np.isnan(auc)


def test_ce_invalid_budget_raises():
    with pytest.raises(ValueError):
        captured_error_at_budget(SCORES6, E6, budgets=(0.0,))


# ----------------------------- Spearman ------------------------------

def test_spearman_monotone_and_antitone():
    x = np.linspace(0, 1, 20)
    assert spearman(x, np.exp(x)) == pytest.approx(1.0)
    assert spearman(x, -x ** 3) == pytest.approx(-1.0)


# ------------------- planchers de degenerescence ---------------------

def test_floors_analytical_formulas():
    assert degeneracy_floors(0.25)["refine_all"] == pytest.approx(0.4)
    assert degeneracy_floors(0.25)["refine_none"] == 0.0
    p = 0.319
    assert degeneracy_floors(p)["refine_all"] == pytest.approx(
        2 * p / (1 + p))


@pytest.mark.parametrize("p", [0.25, 0.319])
def test_refine_all_floor_matches_sklearn_f1(p):
    # verification analytique du plancher : tout-positif -> F1 = 2p/(1+p)
    n = 1000
    k = int(round(p * n))
    gt = np.array([1] * k + [0] * (n - k))
    pred = np.ones(n, dtype=int)
    f1 = f1_score(gt, pred)
    assert f1 == pytest.approx(2 * (k / n) / (1 + k / n))
    assert degeneracy_flag(pred, k / n, gt=gt) is True


def test_refine_none_floor():
    gt = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    pred = np.zeros(8, dtype=int)
    assert f1_score(gt, pred, zero_division=0) == 0.0
    assert degeneracy_flag(pred, 0.25, gt=gt) is True


def test_informative_predictor_not_flagged():
    gt = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    pred = gt.copy()  # F1 = 1, loin des planchers 0.4 et 0
    assert degeneracy_flag(pred, 0.25, gt=gt) is False


def test_flag_without_gt_detects_constant_predictions():
    assert degeneracy_flag(np.ones(100), 0.25) is True
    assert degeneracy_flag(np.zeros(100), 0.25) is True
    mixed = np.array([0, 1] * 50)
    assert degeneracy_flag(mixed, 0.25) is False


def test_flag_tolerance_boundary():
    # F1 a exactement tol du plancher refine-all : flagge
    gt = np.array([1] * 25 + [0] * 75)
    pred = np.ones(100, dtype=int)        # F1 = 0.4 = plancher exact
    assert degeneracy_flag(pred, 0.25, tol=0.005, gt=gt) is True
