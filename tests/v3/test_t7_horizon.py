"""Tests V3 Task 7 : helpers du dataset predictif (sans qiskit).

Les chiffres (tables h, matrice k x h, deltas psi) sont valides par
l'execution sur les vraies donnees.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "study", "v3"))

from t7_horizon import (
    blocked_pair_split,
    capture_at_budget,
    common_traj_values,
    enhl_mask,
    finite_diff_features,
    horizon_pairs,
    method_features,
    subset_captured_error,
)


# ------------------------------ paires --------------------------------

def test_horizon_pairs_range_and_count():
    pairs = horizon_pairs(10, 2, min_t=1)
    assert pairs[0] == (1, 3) and pairs[-1] == (7, 9)
    assert len(pairs) == 7
    assert all(th == t + 2 and t >= 1 and th <= 9 for t, th in pairs)


def test_blocked_pair_split_drops_straddlers():
    # n=10, frac 0.6 -> t0 = 6 ; train: t+h < 6 ; val: t >= 6 ;
    # les paires a cheval (t < 6 <= t+h) sont abandonnees
    tr, va = blocked_pair_split(10, 2, train_frac=0.6, min_t=1)
    assert tr == [(1, 3), (2, 4), (3, 5)]
    assert va == [(6, 8), (7, 9)]
    assert max(th for _, th in tr) < 6 <= min(t for t, _ in va)
    dropped = set(horizon_pairs(10, 2)) - set(tr) - set(va)
    assert dropped == {(4, 6), (5, 7)}


def test_blocked_pair_split_large_horizon_can_empty_train():
    tr, va = blocked_pair_split(10, 8, train_frac=0.6, min_t=1)
    assert tr == []          # aucune paire entierement dans le train
    assert va == []          # t >= 6 exige t+8 <= 9 : impossible
    # h=4 : il reste au moins une paire de chaque cote
    tr4, va4 = blocked_pair_split(20, 4, train_frac=0.6, min_t=1)
    assert tr4 and va4


# ---------------------------- sous-ensemble ---------------------------

def test_enhl_mask():
    y_t = np.array([0, 0, 1, 1])
    y_th = np.array([1, 0, 1, 0])
    np.testing.assert_array_equal(enhl_mask(y_t, y_th),
                                  [True, False, False, False])


def test_subset_captured_error_hand_computed():
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    e = np.array([4.0, 3.0, 2.0, 1.0])
    subset = np.array([False, True, False, True])
    # b=0.5 -> top-2 = {0, 1} ; subset∩top = {1} -> 3 / (3+1)
    assert subset_captured_error(scores, e, subset, 0.5) == \
        pytest.approx(0.75)
    assert np.isnan(subset_captured_error(scores, e,
                                          np.zeros(4, bool), 0.5))
    assert np.isnan(subset_captured_error(scores, np.zeros(4),
                                          subset, 0.5))


def test_capture_at_budget_hand_computed():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    y_f = np.array([1, 0, 1, 0])
    # top-2 = {0, 1} ; futurs-difficiles = {0, 2} -> capture 1/2
    assert capture_at_budget(scores, y_f, 0.5) == pytest.approx(0.5)
    assert capture_at_budget(scores, y_f, 1.0) == pytest.approx(1.0)
    assert np.isnan(capture_at_budget(scores, np.zeros(4), 0.5))


# --------------------------- features ---------------------------------

def test_finite_diff_features_alignment():
    seq = np.array([[[1.0, 2.0]], [[3.0, 5.0]], [[6.0, 9.0]]])
    out = finite_diff_features(seq)
    assert np.all(np.isnan(out[0]))
    np.testing.assert_allclose(out[1], [[2.0, 3.0]])
    np.testing.assert_allclose(out[2], [[3.0, 4.0]])


def _toy_seq(T=3, dim=4):
    rng = np.random.default_rng(0)
    feats2d = rng.normal(size=(T, dim, dim, 9))
    F9 = feats2d.reshape(T, dim * dim, 9)
    return dict(
        F9=F9, FEATS2D=feats2d, D9=finite_diff_features(F9),
        PSI4=rng.normal(size=(T, dim * dim)),
        PSIV2=rng.normal(size=(T, dim * dim)),
    )


def test_method_features_shapes_and_content():
    seq = _toy_seq()
    t = 1
    assert method_features(seq, t, "base9").shape == (16, 9)
    assert method_features(seq, t, "base9+D9").shape == (16, 18)
    assert method_features(seq, t, "base9+psi4").shape == (16, 10)
    assert method_features(seq, t, "base9+psiv2").shape == (16, 10)
    full = method_features(seq, t, "full (base9+D9+psi4+psiv2)")
    assert full.shape == (16, 20)
    # contenu : colonnes 0-8 = F9(t) ; col 9 du jeu psi4 = PSI4(t)
    np.testing.assert_array_equal(full[:, :9], seq["F9"][t])
    np.testing.assert_array_equal(
        method_features(seq, t, "base9+psi4")[:, 9], seq["PSI4"][t])
    # k-hop : reutilise la construction Task 1b
    assert method_features(seq, t, "khop0").shape == (16, 9)
    assert method_features(seq, t, "khop1").shape == (16, 45)
    assert method_features(seq, t, "khop2").shape == (16, 225)


def test_method_features_unknown_name_raises():
    with pytest.raises(ValueError):
        method_features(_toy_seq(), 0, "nope")


# ----------------- trajectoires communes (bootstrap) -------------------

def test_common_traj_values_intersection_and_nan_filter():
    # une trajectoire courte peut manquer dans un bras (p.ex. harris a
    # h=8 bloque : aucune paire val) ou valoir NaN -> intersection
    cfgs = [("a", 1), ("a", 2), ("b", 1), ("b", 2)]
    ra = {("a", 1): 0.5, ("a", 2): 0.6, ("b", 1): np.nan}
    rb = {("a", 1): 0.4, ("a", 2): 0.7, ("b", 1): 0.3, ("b", 2): 0.2}
    va, vb, common = common_traj_values(ra, rb, cfgs)
    assert common == [("a", 1), ("a", 2)]
    np.testing.assert_allclose(va, [0.5, 0.6])
    np.testing.assert_allclose(vb, [0.4, 0.7])
