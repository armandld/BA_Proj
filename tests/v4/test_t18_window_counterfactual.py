"""Tests V4 T18 : contrefactuel « ZZ sans la fenetre ».

L'enjeu de cette tache est une conclusion NEGATIVE (« annuler ZZ ne change
rien, meme a couplage plein »). Une conclusion negative n'a de valeur que
si l'instrument sait produire un positif : les tests ci-dessous verifient
donc surtout que `ablate_all` DETECTE un changement quand il y en a un, et
que la neutralisation de la fenetre est bien verifiee plutot que supposee.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("v4", "v3", ""):
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "study", sub))

from t13_term_ablation import ABLATIONS
from t18_window_counterfactual import HUGE_SIGMA, _c_amplitude, ablate_all


def _hp(dim=2, c=0.0, k=0.0, z=None):
    """Hamiltonien jouet au format attendu par `ground_state_mask`."""
    z = np.zeros((dim, dim)) if z is None else np.asarray(z, dtype=float)
    return {
        "H_edges": (z, z),
        "C_edges": (np.full((dim, dim), c), np.full((dim, dim), c)),
        "K_plaquettes": np.full((dim, dim), k),
        "threshold_amr": 0.0,
        "w_z_frac": 0.15,
    }


def test_c_amplitude_handles_tuple_and_array():
    assert _c_amplitude(_hp(c=-3.0)) == pytest.approx(3.0)
    hp = _hp()
    hp["C_edges"] = np.array([[-2.0, 1.0], [0.5, 0.0]])
    assert _c_amplitude(hp) == pytest.approx(2.0)


def test_huge_sigma_neutralises_the_gaussian():
    """La constante utilisee pour le bras `no_window` doit vraiment rendre
    la fenetre unitaire en double precision, sinon le contrefactuel
    compare deux versions attenuees."""
    from t17_uncertainty_window import uncertainty_window
    score = np.random.default_rng(0).random((8, 8))
    w = uncertainty_window(score, 0.1496, HUGE_SIGMA)
    assert w.min() > 1.0 - 1e-12


def test_control_ablation_changes_nothing():
    """`full` est le controle de la chaine de mesure : il compare le
    Hamiltonien a lui-meme et doit rendre exactement 0."""
    gt = np.array([[True, False], [False, True]])
    res, _ = ablate_all(_hp(c=-1.0, k=-0.5, z=[[0.3, -0.2], [0.1, 0.4]]),
                        2, gt)
    full = [r for r in res if r["ablation"] == "full"][0]
    assert full["changed"] == 0.0


def test_ablation_detects_a_real_change():
    """Test de sensibilite : avec un biais Z qui porte seul la decision,
    retirer Z DOIT changer le masque. Sans ce controle positif, un
    'changed = 0' partout ne prouverait rien."""
    gt = np.zeros((2, 2), dtype=bool)
    hp = _hp(c=0.0, k=0.0, z=[[0.9, 0.8], [0.7, 0.6]])
    res, _ = ablate_all(hp, 2, gt)
    by = {r["ablation"]: r for r in res}
    assert by["no_Z"]["changed"] > 0.0
    assert by["full"]["changed"] == 0.0


def test_dropping_an_all_zero_family_is_a_no_op():
    """Si ZZ est deja nul, l'annuler ne peut rien changer — coherence
    interne de l'instrument."""
    gt = np.zeros((2, 2), dtype=bool)
    res, _ = ablate_all(_hp(c=0.0, k=-0.4, z=[[0.5, -0.5], [0.5, -0.5]]),
                        2, gt)
    by = {r["ablation"]: r for r in res}
    assert by["no_ZZ"]["changed"] == 0.0


def test_every_declared_ablation_is_reported():
    gt = np.zeros((2, 2), dtype=bool)
    res, _ = ablate_all(_hp(c=-1.0, k=-0.5), 2, gt)
    assert {r["ablation"] for r in res} == {n for n, _ in ABLATIONS}
    for r in res:
        assert 0.0 <= r["changed"] <= 1.0
        assert 0.0 <= r["refined"] <= 1.0


def test_reported_fields_are_present_and_finite():
    gt = np.zeros((2, 2), dtype=bool)
    res, uni = ablate_all(_hp(c=-1.0, k=-0.5, z=[[0.2, 0.1], [0.0, 0.3]]),
                          2, gt)
    assert isinstance(uni, bool)
    for r in res:
        for key in ("changed", "refined", "f1", "dE", "n_optima"):
            assert np.isfinite(r[key]), key
