"""D-98 — CORRIGE. Le « controle negatif » de fig9 ne compte plus de P/R/F1.

`fig9_synthetic_unit_tests.py` annonce sa 4e ligne comme
« Uniform noise : negative control -> false positive rate ». `pixel_prf`
definit sa reference RELATIVEMENT au champ mesure (`needs = gt > gt.mean()`)
: sur un champ sans anomalie, la moyenne coupe le bruit en deux et pres de
la moitie des pixels seraient declares « a raffiner » si on l'utilisait
comme verite terrain. Il n'y a alors pas de faux positif a compter contre
elle.

Correction : le controle negatif utilise desormais `false_flag_rate`, qui
ne compare a AUCUNE reference — juste la fraction du champ que chaque bras
marque a raffiner. `pixel_prf` reste inchangee et correcte pour les trois
lignes a signal (vortex, current sheet, X-point), ou les deux bras
partagent le meme `needs` et la comparaison reste valide.

Ces tests verifient la correction : `false_flag_rate` existe, ne prend pas
de verite terrain en argument, et la ligne de controle negatif de fig9
l'utilise au lieu de `pixel_prf`.
"""
import ast
import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_V1_LEGACY = os.path.join(_REPO_ROOT, "figures", "v1_legacy")
if _V1_LEGACY not in sys.path:
    sys.path.insert(0, _V1_LEGACY)

from fig_utils import ground_truth_errors  # noqa: E402
from Simulation.grid import PeriodicGrid  # noqa: E402
from Simulation.solver import MHDSolver  # noqa: E402


def _module_source():
    path = os.path.join(_V1_LEGACY, "fig9_synthetic_unit_tests.py")
    with open(path, encoding="utf-8") as f:
        return f.read(), path


def _pixel_prf_reference(gt):
    """La reference relative de `pixel_prf`, isolee — toujours vraie pour
    les trois lignes a signal, voir `fig9.pixel_prf`."""
    return gt > gt.mean()


def test_la_reference_relative_ne_porte_aucune_information_absolue():
    """Le champ qui SEPARE : multiplier gt par 1000 doit tout changer si la
    reference etait absolue. Elle ne change rien — donc `pixel_prf` reste
    defendable seulement pour comparer deux bras sur le MEME champ, jamais
    pour un controle negatif seul (c'est `false_flag_rate` qui en tient
    lieu desormais).
    """
    rng = np.random.default_rng(0)
    gt = np.abs(rng.standard_normal((32, 32)))
    a = _pixel_prf_reference(gt)
    b = _pixel_prf_reference(gt * 1000.0)
    assert np.array_equal(a, b), "la reference reagit a l'echelle"
    c = _pixel_prf_reference(gt + 5.0)
    assert np.array_equal(a, c), "la reference reagit a un decalage constant"


def test_false_flag_rate_ne_prend_aucune_verite_terrain():
    """Le champ qui SEPARE la correction de l'ancien defaut : `false_flag_rate`
    ne doit meme pas pouvoir recevoir un `gt`, faute de quoi rien
    n'empecherait de la re-brancher sur la meme reference relative que
    `pixel_prf` et de reintroduire D-98 sous un autre nom.
    """
    src, _ = _module_source()
    arbre = ast.parse(src)
    fn = next((n for n in ast.walk(arbre)
               if isinstance(n, ast.FunctionDef) and n.name == "false_flag_rate"),
              None)
    assert fn is not None, "false_flag_rate a disparu de fig9 : D-98 rouvert"
    params = [a.arg for a in fn.args.args]
    assert params == ["patches", "N"], (
        f"false_flag_rate prend {params} : un parametre `gt` la ferait "
        "retomber dans le defaut D-98 qu'elle corrige")
    doc = ast.get_docstring(fn)
    assert doc and "D-98" in doc, "le renvoi D-98 a quitte false_flag_rate"


def test_le_flux_principal_utilise_false_flag_rate_pour_le_controle_negatif():
    """Verifie que la ligne `make_uniform_noise` du script principal appelle
    `false_flag_rate`, pas `pixel_prf` — comportement, pas juste presence
    de la fonction (qui pourrait exister sans etre branchee).
    """
    src, path = _module_source()
    assert "is_negative_control" in src, (
        f"{path} ne distingue plus la ligne de controle negatif des trois "
        "lignes a signal")
    assert "false_flag_rate(" in src, (
        f"{path} n'appelle plus false_flag_rate")


def test_le_controle_negatif_declare_la_moitie_du_domaine_par_reference_relative():
    """Fait historique qui motive la correction, toujours vrai : SI on
    utilisait `needs = gt > gt.mean()` comme verite terrain sur le champ
    SANS anomalie de fig9, pres de la moitie du domaine serait declaree
    « a raffiner ». C'est cette lecture-la que `false_flag_rate` evite —
    ce test ne verifie plus ce que fig9 affiche (voir les deux tests
    ci-dessus), seulement que la raison de la correction reste mesurable.

    Mesure consignee : 0,4709 a N=64 (0,466 a N=256, 50 pas).
    """
    N = 64
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    rng = np.random.default_rng(0)   # meme graine que make_uniform_noise
    sim.vx = 0.01 * rng.standard_normal((N, N))
    sim.vy = 0.01 * rng.standard_normal((N, N))
    sim.Bx = 1.0 + 0.01 * rng.standard_normal((N, N))
    sim.By = 0.01 * rng.standard_normal((N, N))
    for _ in range(50):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)

    gt = ground_truth_errors(sim, N, 2)
    frac = float(np.mean(_pixel_prf_reference(gt)))
    assert frac == pytest.approx(0.4709, abs=0.01), (
        "fraction declaree = %.4f, consignee 0,4709 — remesurer, ne pas "
        "retoucher le seuil" % frac)
    assert frac > 0.25, (
        "la reference relative ne declare plus la moitie du domaine : "
        "revoir si la correction D-98 est toujours motivee")
