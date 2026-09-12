"""D-62 — le nuage de Pareto du rescore coupait ses propres points.

`plot_pareto_with_isocost` fixait sa fenêtre verticale a (-0,05 ; 0,40), en
dur. C'est la figure qui porte le front de Pareto et les lignes d'iso-score :
un point hors cadre n'est pas signale, il disparait, et le front semble
s'arreter la ou le cadre s'arrete.

Mesure sur les bases gelees, erreur physique moyennee sur les scenarios :

    q_has_v2_phase1      phys dans [0,0348 ; 0,2997]   0/178 hors cadre
    classical_v2_phase1  phys dans [0,0114 ; 2,2749]   9/125 hors cadre,
                         dont 3 des 46 points du front de Pareto

Sur quelle entree ce test echoue-t-il ? Sur la version d'avant D-62, avec la
base classique : la borne haute y vaut 0,40 pour un maximum trace a 2,2749.
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

_STUDIES = os.path.join(_ROOT, "results", "hyperparams", "optuna_studies")

#: Mesures du 13 aout 2026, ecrites pour qu'une derive se voie.
_PHYS_MAX = {"q_has_v2_phase1": 0.2997, "classical_v2_phase1": 2.2749}
_PHYS_MIN = {"q_has_v2_phase1": 0.0348, "classical_v2_phase1": 0.0114}


@pytest.fixture(scope="module")
def rescorer():
    return pytest.importorskip("recompute_lambda_scores")


def _phys_and_window(rescorer, study, monkeypatch, tmp_path):
    db = os.path.join(_STUDIES, study + ".db")
    if not os.path.exists(db):
        pytest.skip("base gelee absente : " + db)
    _s, completed = rescorer.load_completed_trials(db, study)

    keys = rescorer._detect_scenario_keys(completed)
    phys = np.array([rescorer._get_global_phys_patch(t, keys)[0]
                     for t in completed])

    seen = {}

    def _capture(fig, output_dir, name):
        seen["ylim"] = fig.axes[0].get_ylim()
        plt.close(fig)

    monkeypatch.setattr(rescorer, "_save", _capture)
    rescorer.plot_pareto_with_isocost(completed, 0.4, str(tmp_path))
    assert "ylim" in seen, "aucune figure produite"
    return phys, seen["ylim"]


@pytest.mark.parametrize("study", ["q_has_v2_phase1", "classical_v2_phase1"])
def test_window_contains_every_plotted_point(rescorer, study, monkeypatch,
                                             tmp_path):
    phys, (lo, hi) = _phys_and_window(rescorer, study, monkeypatch, tmp_path)
    assert phys.max() == pytest.approx(_PHYS_MAX[study], abs=1e-4)
    assert phys.min() == pytest.approx(_PHYS_MIN[study], abs=1e-4)
    hidden = int(((phys < lo) | (phys > hi)).sum())
    assert hidden == 0, (
        "%d points sur %d hors cadre : la figure en cache sans le dire"
        % (hidden, len(phys)))


def test_window_unchanged_when_everything_fits(rescorer, monkeypatch,
                                               tmp_path):
    """Épingle l'intention : la fenêtre par défaut reste celle d'origine.

    Sur l'étude quantique tout entrait déjà dans (-0,05 ; 0,40) — la
    correction ne doit pas avoir change cette figure.
    """
    _phys, (lo, hi) = _phys_and_window(rescorer, "q_has_v2_phase1",
                                       monkeypatch, tmp_path)
    assert (lo, hi) == pytest.approx((-0.05, 0.4))


def test_pareto_front_points_are_all_visible(rescorer, monkeypatch, tmp_path):
    """Le front lui-meme : 3 de ses 46 points tombaient hors cadre."""
    study = "classical_v2_phase1"
    db = os.path.join(_STUDIES, study + ".db")
    if not os.path.exists(db):
        pytest.skip("base gelee absente : " + db)
    _s, completed = rescorer.load_completed_trials(db, study)
    keys = rescorer._detect_scenario_keys(completed)
    phys = np.array([rescorer._get_global_phys_patch(t, keys)[0]
                     for t in completed])
    patch = np.array([rescorer._get_global_phys_patch(t, keys)[1]
                      for t in completed])
    front = np.column_stack([patch, phys])[
        rescorer._pareto_front(np.column_stack([patch, phys]))]
    assert len(front) == 46

    _phys, (lo, hi) = _phys_and_window(rescorer, study, monkeypatch, tmp_path)
    hidden = int(((front[:, 1] < lo) | (front[:, 1] > hi)).sum())
    assert hidden == 0, "%d points du front hors cadre" % hidden
