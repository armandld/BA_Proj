"""Tests V3 Task 4 : helpers du split bloque (sans donnees DNS).

Le critere d'acceptation chiffre (reproduction des nombres de phase 11A
sur le split aleatoire : classique 0.475, GBT 0.980) est valide par
l'execution sur les vraies donnees.
"""

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

from h2b_blocked_split import (

    apply_per_config_threshold,
    per_config_thresholds,
    ranking_metrics_per_snapshot,
    replace_score_column,
    split_indices_blocked,
    split_indices_random,
)


def _thr_fn(scores, gt, grid=None):
    """Mini best_threshold_f1 injectable (evite la chaine qiskit)."""
    if grid is None:
        grid = np.linspace(scores.min(), scores.max(), 101)
    best = (float(grid[0]), -1.0)
    for t in grid:
        f1 = f1_score(gt, (scores > t).astype(int), zero_division=0)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


# ------------------------------ splits -------------------------------

def test_random_split_reproduces_phase11a_permutation():
    n, seed, frac = 440, 0, 0.7
    tr, va = split_indices_random(n, seed, frac)
    perm = np.random.default_rng(seed).permutation(n)
    assert tr == list(perm[:308])
    assert va == list(perm[308:])
    assert len(tr) == 308 and len(va) == 132  # 440 snaps de phase 11A


def test_blocked_split_per_config_60_40():
    cfg = ["a"] * 10 + ["b"] * 5
    tr, va = split_indices_blocked(cfg, 0.6)
    assert tr == [0, 1, 2, 3, 4, 5, 10, 11, 12]   # 6 de a + 3 de b
    assert va == [6, 7, 8, 9, 13, 14]
    # train strictement avant val a l'interieur de chaque config
    for c in ("a", "b"):
        t_pos = [i for i in tr if cfg[i] == c]
        v_pos = [i for i in va if cfg[i] == c]
        assert max(t_pos) < min(v_pos)


# --------------------------- colonne score ---------------------------

def test_replace_score_column_no_mutation():
    X = np.arange(12, dtype=float).reshape(4, 3)
    X0 = X.copy()
    s = np.array([9.0, 9.0, 9.0, 9.0])
    X2 = replace_score_column(X, s)
    np.testing.assert_array_equal(X, X0)          # pas de mutation
    np.testing.assert_array_equal(X2[:, 0], s)
    np.testing.assert_array_equal(X2[:, 1:], X[:, 1:])


# ----------------------- seuils par config ---------------------------

def test_per_config_threshold_recovers_scale_shift():
    """Mecanisme branche 2 : memes classements, echelles decalees par
    config -> le seuil global echoue, les seuils par config recuperent
    un F1 parfait."""
    rng = np.random.default_rng(0)
    n = 300
    s_raw_a = rng.uniform(size=n)
    s_raw_b = rng.uniform(size=n)
    y_a = (s_raw_a > 0.75).astype(int)
    y_b = (s_raw_b > 0.75).astype(int)
    scores = np.concatenate([s_raw_a, s_raw_b + 10.0])  # decalage d'echelle
    y = np.concatenate([y_a, y_b])
    cfg = np.array(["A"] * n + ["B"] * n)

    thr_g, _ = _thr_fn(scores, y)
    pred_g = (scores > thr_g).astype(int)
    f1_global = f1_score(y, pred_g, zero_division=0)

    thr_map = per_config_thresholds(scores, y, cfg, _thr_fn)
    pred_p = apply_per_config_threshold(scores, cfg, thr_map, thr_g)
    f1_per = f1_score(y, pred_p, zero_division=0)

    assert f1_per > 0.99          # calibration par config : parfait
    assert f1_per > f1_global + 0.2


def test_apply_per_config_threshold_fallback():
    p = np.array([0.1, 0.9])
    cfg = np.array(["seen", "unseen"])
    pred = apply_per_config_threshold(p, cfg, {"seen": 0.5}, 0.8)
    assert list(pred) == [0, 1]   # unseen -> seuil de repli 0.8


# ------------------- metriques par snapshot --------------------------

def test_ranking_metrics_means_hand_computed():
    # snapshot 1 = l'exemple 6 patches de Task 2 ; snapshot 2 = un seul
    # patch porte toute l'erreur et il est classe premier
    s1 = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.3])
    e1 = np.array([5.0, 1.0, 2.0, 0.0, 1.0, 1.0])
    s2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    e2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    ce, auc, _ = ranking_metrics_per_snapshot([s1, s2], [e1, e2])
    assert ce[0.10] == pytest.approx((0.5 + 1.0) / 2)
    assert ce[0.25] == pytest.approx((0.6 + 1.0) / 2)
    assert ce[0.50] == pytest.approx((0.6 + 1.0) / 2)
    assert auc == pytest.approx((7.4 / 12 + 11.0 / 12) / 2)


def test_ranking_metrics_nan_safe_spearman():
    # un snapshot a score constant (rho indefini) -> ignore par nanmean
    s1 = np.array([1.0, 2.0, 3.0, 4.0])
    e1 = np.array([1.0, 2.0, 3.0, 4.0])     # rho = 1
    s2 = np.full(4, 0.5)                     # constant -> NaN
    e2 = np.array([1.0, 2.0, 3.0, 4.0])
    _, _, rho = ranking_metrics_per_snapshot([s1, s2], [e1, e2])
    assert rho == pytest.approx(1.0)
