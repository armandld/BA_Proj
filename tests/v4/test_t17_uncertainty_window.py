"""Tests V4 T17 : fenetre d'incertitude sur la famille ZZ.

Fonctions pures uniquement — la partie qui evolue le solveur V1 est
couverte par les criteres du script lui-meme. Ce qui est verifie ici est
la correction de l'instrument : la fenetre doit reproduire exactement la
formule de `HamiltParams.compute_coefficients`, distinguer les deux
familles d'aretes, et se neutraliser quand sigma -> +inf (c'est cette
neutralisation qui sert a attribuer la suppression a la fenetre plutot
qu'aux gates amont).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("v4", "v3", ""):
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "study", sub))

from t17_uncertainty_window import PARAM_SETS, SCENARIOS, uncertainty_window


def test_window_is_one_at_the_threshold():
    """w(score = threshold) = 1 : c'est la definition de la gaussienne
    centree sur le seuil de decision."""
    score = np.full((4, 4), 0.25)
    w = uncertainty_window(score, threshold_amr=0.25, sigma=0.05)
    assert np.allclose(w, 1.0)


def test_window_decays_as_the_score_leaves_the_threshold():
    score = np.array([[0.25, 0.35, 0.45, 0.55]])
    # champ constant par colonne -> on lit la decroissance via un seul axe
    w = uncertainty_window(np.repeat(score, 4, axis=0), 0.25, 0.10, axis=0)
    assert np.allclose(w[:, 0], 1.0)
    assert w[0, 1] > w[0, 2] > w[0, 3]
    assert w[0, 3] < 0.05          # 3 sigma


def test_window_matches_the_closed_form_on_edge_averages():
    rng = np.random.default_rng(0)
    score = rng.random((6, 6))
    thr, sig = 0.15, 0.19
    w = uncertainty_window(score, thr, sig, axis=1)
    expected = np.exp(-((0.5 * (score + np.roll(score, -1, axis=1)) - thr)
                        / sig) ** 2)
    assert np.allclose(w, expected)


def test_horizontal_and_vertical_windows_differ():
    """Les deux familles d'aretes ne partagent pas la meme fenetre : les
    apparier serait une erreur de mesure, pas un detail."""
    rng = np.random.default_rng(1)
    score = rng.random((8, 8))
    w_h = uncertainty_window(score, 0.2, 0.1, axis=1)
    w_v = uncertainty_window(score, 0.2, 0.1, axis=0)
    assert not np.allclose(w_h, w_v)


def test_large_sigma_neutralises_the_window():
    """sigma -> +inf doit rendre w == 1 partout : c'est l'hypothese qui
    permet d'attribuer la suppression a la fenetre."""
    rng = np.random.default_rng(2)
    score = rng.random((8, 8))
    w = uncertainty_window(score, 0.1496, 1e9)
    assert w.min() > 1.0 - 1e-9


def test_window_underflows_when_every_score_is_far_from_threshold():
    """Le cas Orszag-Tang : un score confine loin du seuil eteint toute la
    famille ZZ, quelle que soit la physique du champ."""
    score = np.random.default_rng(3).uniform(0.5057, 0.8748, (16, 16))
    w = uncertainty_window(score, threshold_amr=0.0, sigma=0.05)
    assert w.max() < 1e-40
    # avec le sigma reellement entraine, la fenetre ne sous-deborde plus
    # mais reste tres petite : les deux regimes doivent rester distincts
    w_tr = uncertainty_window(score, threshold_amr=0.1496, sigma=0.1888)
    assert 1e-10 < w_tr.max() < 1e-1


def test_sigma_is_floored_rather_than_dividing_by_zero():
    score = np.full((3, 3), 0.4)
    w = uncertainty_window(score, 0.1, sigma=0.0)
    assert np.all(np.isfinite(w))
    assert w.max() == pytest.approx(0.0, abs=1e-12)


def test_declared_param_sets_and_scenarios_are_coherent():
    assert set(PARAM_SETS) == {"v1_test_default", "level3_trained"}
    for ps in PARAM_SETS.values():
        assert ps["sigma"] > 0 and ps["threshold_amr"] >= 0
    # les quatre classes des folds Level-3, sous leurs noms V1 exacts
    assert len(SCENARIOS) == 4
    assert all(s.startswith("init_") for s in SCENARIOS)


def test_scenario_names_exist_on_the_v1_solver():
    """Garde-fou : un nom errone ferait silencieusement sauter une classe
    (le script se contente de la signaler comme indisponible)."""
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
    from Simulation.solver import MHDSolver
    for s in SCENARIOS:
        assert hasattr(MHDSolver, s), s
