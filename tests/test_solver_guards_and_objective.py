"""Trois corrections mesurees : le garde de divergence, l'espace de recherche
reel de l'objectif, et l'origine de sigma.

Chacune ferme la meme forme de defaut : une valeur qui se substitue en
silence a une valeur absente, et qu'aucun aval ne peut distinguer d'une
valeur choisie.
"""

import os
import sys
import warnings

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.grid import PeriodicGrid  # noqa: E402
from Simulation.solver import MHDSolver  # noqa: E402


# ══════════════════════════════════════════════════════════════════════
#  1. is_diverged : un seuil qui attrape enfin quelque chose
# ══════════════════════════════════════════════════════════════════════
#
# Le seuil valait 1e100. `float64` ne deborde qu'au-dela de ~1e154, donc un
# champ a 1e50 — physiquement mort, 1e49 fois l'echelle du probleme —
# passait sans un mot. Mesure sur les quatre scenarios, 200 pas a CFL 0.4 :
# pic a 3.85. Le nouveau seuil de 1e8 laisse une marge de 2.6e7.

def _sim(n=32):
    s = MHDSolver(PeriodicGrid(n), dt=1e-4, Re=400, Rm=400)
    s.init_orszag_tang()
    return s


def test_a_healthy_field_is_not_flagged():
    assert not _sim().is_diverged()


@pytest.mark.parametrize("scenario", ["orszag_tang", "kelvin_helmholtz",
                                      "mhd_rotor", "harris_tearing"])
def test_no_scenario_comes_anywhere_near_the_threshold(scenario):
    """La marge doit rester enorme sur le comportement REEL, sinon le seuil
    abregerait des runs viables."""
    s = MHDSolver(PeriodicGrid(64), dt=1e-4, Re=400, Rm=400)
    getattr(s, f"init_{scenario}")()
    for _ in range(50):
        s.adapt_dt(cfl_target=0.4)
        s.step_full(record_stats=False)
    peak = max(np.max(np.abs(getattr(s, k))) for k in ("vx", "vy", "Bx", "By"))
    assert peak < 1e3, f"{scenario} atteint {peak:.3g}"
    assert not s.is_diverged()


def test_a_field_that_is_physically_dead_is_now_caught():
    """1e50 passait avant : c'est 1e49 fois l'echelle du probleme."""
    s = _sim()
    s.vx = np.full_like(s.vx, 1e50)
    assert s.is_diverged()


@pytest.mark.parametrize("mag", [1e9, 1e20, 1e50, 1e100])
def test_every_absurd_amplitude_is_caught(mag):
    s = _sim()
    s.By = np.full_like(s.By, mag)
    assert s.is_diverged()


def test_the_old_threshold_would_have_missed_all_of_them():
    """Le defaut lui-meme, fige : avec 1e100 rien de tout cela ne sortait."""
    s = _sim()
    s.vx = np.full_like(s.vx, 1e50)
    assert not s.is_diverged(max_value=1e100)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_and_inf_are_still_caught(bad):
    s = _sim()
    s.Bx[0, 0] = bad
    assert s.is_diverged()


def test_the_caller_can_still_widen_the_threshold_explicitly():
    """Un garde qu'on ne peut pas desserrer finit par etre contourne."""
    s = _sim()
    s.vx = np.full_like(s.vx, 1e10)
    assert s.is_diverged()
    assert not s.is_diverged(max_value=1e20)


# ══════════════════════════════════════════════════════════════════════
#  2. L'objectif : ce qu'il explore vraiment
# ══════════════════════════════════════════════════════════════════════
#
# Quatre parametres etaient ecrits `if "x" not in frozen:` — la forme d'un
# parametre conditionnel — alors que ce sont des constantes. Une campagne
# pouvait croire optimiser kappa sans le faire.

def _T():
    import TrainHyperParam_v2 as T
    return T


def test_the_search_space_is_five_parameters_not_nine():
    T = _T()
    assert set(T.search_space(True)) == {"beta", "w_z_frac", "sigma",
                                         "beta_curl", "beta_xpoint"}
    assert len(T.search_space(True)) == 5


def test_the_first_phase_searches_the_shared_michelson_instead():
    T = _T()
    assert "beta_michelson" in T.search_space(False)
    assert "sigma" not in T.search_space(False)


@pytest.mark.parametrize("name", ["threshold_amr", "gamma_hydro",
                                  "gamma_mag", "kappa"])
def test_the_fixed_parameters_are_named_and_never_searched(name):
    T = _T()
    assert name in T.FIXED_PARAMS
    assert name not in T.search_space(True)
    assert name not in T.search_space(False)


def test_the_frozen_threshold_is_the_measured_classical_best():
    """0.14959824837662078 est la valeur du meilleur essai classique (#42).
    Le lien est verifie contre la base dans
    tests/test_hyperparams_provenance_break.py."""
    assert _T().CLASSICAL_BEST_THRESHOLD == pytest.approx(
        0.14959824837662078, abs=1e-15)


def test_the_search_space_matches_what_the_objective_actually_suggests():
    """Le test croise : la liste declaree doit coincider avec les
    `trial.suggest_*` du code. Une liste qui derive du code serait pire
    qu'une absence de liste."""
    import inspect
    import re
    T = _T()
    src = inspect.getsource(T.make_composite_objective)
    suggested = set(re.findall(r'trial\.suggest_\w+\("(\w+)"', src))
    declared = set(T.search_space(True)) | set(T.search_space(False))
    assert suggested == declared, (
        f"le code propose {sorted(suggested)}, la liste annonce "
        f"{sorted(declared)}")


def test_no_fixed_parameter_is_ever_suggested():
    import inspect
    import re
    T = _T()
    src = inspect.getsource(T.make_composite_objective)
    suggested = set(re.findall(r'trial\.suggest_\w+\("(\w+)"', src))
    assert not (suggested & set(T.FIXED_PARAMS)), (
        "un parametre declare fixe est propose a Optuna")


# ══════════════════════════════════════════════════════════════════════
#  3. sigma : le repli ne doit plus etre silencieux
# ══════════════════════════════════════════════════════════════════════

def test_the_pipeline_warns_when_sigma_has_to_be_defaulted():
    """0.05 n'est pas un defaut raisonnable : c'est une valeur qu'aucun essai
    n'a choisie, appliquee au parametre au coeur de D-9."""
    src = open(os.path.join(_SRC, "pipeline.py"), encoding="utf-8").read()
    assert "_sigma_defaulted = 'sigma' not in hp" in src
    assert "RuntimeWarning" in src
    assert "D-22" in src


def test_the_run_details_record_where_sigma_came_from():
    """Sans cette trace, un artefact ne permet pas de dire si sigma vient de
    l'entrainement ou d'un repli."""
    src = open(os.path.join(_SRC, "pipeline.py"), encoding="utf-8").read()
    assert "_out['sigma_source']" in src
    assert "'default' if _sigma_defaulted else 'loaded'" in src


def test_sigma_is_currently_absent_from_the_loaded_hyperparameters():
    """L'etat d'aujourd'hui, epingle. A retourner apres reoptimisation."""
    from hyperparams_loader import load_hyperparams
    assert "sigma" not in load_hyperparams(), (
        "sigma est de retour : verifier qu'il vaut ce que la campagne a "
        "trouve, et retourner ce test")
