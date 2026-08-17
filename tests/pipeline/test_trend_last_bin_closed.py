"""D-61 — la ligne de tendance jetait toujours le point extreme.

`analyze_hyperparams._add_trend` trace une mediane par classe. Ses bornes
sortent de `linspace(x.min(), x.max())`, donc la derniere borne EST `x.max()`
— et le masque `x < bins[k+1]` excluait cette valeur de toute classe. L'essai
qui porte la plus grande valeur du parametre n'entrait dans aucune mediane,
sur les quatre figures qui appellent cette fonction.

Sur quelle entree ces tests echouent-ils ? Sur la version d'avant D-61 :
`test_extreme_point_enters_the_last_bin` n'y trace aucune ligne du tout
(deux classes sur trois seulement, sous le minimum de trois), et
`test_frozen_study_last_median` y rend 0,258164 au lieu de 0,258670.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_ROOT = _repo_root()
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_Q_DB = os.path.join(_ROOT, "results", "hyperparams", "optuna_studies",
                     "q_has_v2_phase1.db")
_Q_STUDY = "q_has_v2_phase1"

#: Mesure du 13 aout 2026 sur la base gelee (178 essais complets finis),
#: parametre `beta` contre la perte composite, `n_bins` par defaut.
#: Ecrites pour qu'une derive se voie.
_LAST_MEDIAN_BETA = 0.258670        # avant D-61 : 0.258164
_N_TREND_POINTS_BETA = 14


@pytest.fixture(scope="module")
def analyzer():
    return pytest.importorskip("analyze_hyperparams")


@pytest.fixture(scope="module")
def rescorer():
    return pytest.importorskip("recompute_lambda_scores")


def _trend_points(analyzer, x, y, **kw):
    """Ce que la fonction TRACE, pas ce que son source dit."""
    fig, ax = plt.subplots()
    try:
        before = len(ax.get_lines())
        analyzer._add_trend(ax, x, y, **kw)
        lines = ax.get_lines()
        if len(lines) == before:
            return None
        return [np.asarray(a) for a in lines[-1].get_data()]
    finally:
        plt.close(fig)


def test_extreme_point_enters_the_last_bin(analyzer):
    """Entree qui SEPARE : le point extreme decide la mediane du bord.

    Trois classes, bornes 0 / 4 / 8 / 12. La derniere, [8, 12], contient
    x = 8 (y = 1) et deux fois x = 12 (y = 100). Fermee, sa mediane vaut
    100 ; ouverte, elle ne retient qu'un seul point, tombe sous le minimum
    de deux, et la ligne entiere disparait faute de trois classes.
    """
    x = [0.0, 1.0, 4.0, 5.0, 8.0, 12.0, 12.0]
    y = [0.0, 0.0, 1.0, 1.0, 1.0, 100.0, 100.0]

    points = _trend_points(analyzer, x, y, n_bins=3)
    assert points is not None, "aucune ligne de tendance tracee"
    centers, medians = points
    assert len(centers) == 3
    assert centers[-1] == pytest.approx(10.0)
    assert medians[-1] == pytest.approx(100.0)


def test_interior_bins_stay_half_open(analyzer):
    """Epingle ce qui NE change pas : seule la derniere classe est fermee.

    x = 4 est la borne entre la premiere et la deuxieme classe ; il doit
    rester dans la deuxieme. S'il comptait dans les deux, la premiere
    mediane passerait de 0,0 a 0,5 — c'est ce que ce test interdit.
    """
    x = [0.0, 1.0, 4.0, 5.0, 8.0, 11.0, 12.0]
    y = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    centers, medians = _trend_points(analyzer, x, y, n_bins=3)
    assert medians[0] == pytest.approx(0.0)


def test_too_few_points_still_draws_nothing(analyzer):
    """Garde inchangee : moins de 5 essais, pas de tendance."""
    assert _trend_points(analyzer, [0.0, 1.0, 2.0, 3.0],
                         [0.0, 1.0, 2.0, 3.0]) is None


def test_both_copies_agree(analyzer, rescorer):
    """`_add_trend` existe en DEUX exemplaires, mot pour mot.

    D-61 vivait dans les deux. Corriger une copie et pas l'autre laisse le
    défaut en place là où on ne le cherche plus — et rien ne le dirait. Ce
    test compare ce que les deux TRACENT sur l'entrée qui sépare, pas leur
    source : une réécriture de l'une reste permise tant que le résultat
    tient.
    """
    x = [0.0, 1.0, 4.0, 5.0, 8.0, 12.0, 12.0]
    y = [0.0, 0.0, 1.0, 1.0, 1.0, 100.0, 100.0]

    a = _trend_points(analyzer, x, y, n_bins=3)
    b = _trend_points(rescorer, x, y, n_bins=3)
    assert a is not None and b is not None
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])
    assert b[1][-1] == pytest.approx(100.0)


def test_frozen_study_last_median(analyzer):
    """La mesure, sur la base gelee : le nombre bouge, et il est ecrit."""
    if not os.path.exists(_Q_DB):
        pytest.skip("base gelee absente : " + _Q_DB)
    _study, completed = analyzer.load_study(_Q_DB, _Q_STUDY)
    x = [t.params["beta"] for t in completed]
    y = [t.value for t in completed]

    centers, medians = _trend_points(analyzer, x, y)
    assert len(centers) == _N_TREND_POINTS_BETA
    assert medians[-1] == pytest.approx(_LAST_MEDIAN_BETA, abs=1e-6)
