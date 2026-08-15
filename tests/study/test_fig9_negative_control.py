"""D-97 — le « controle negatif » de fig9 ne peut pas echouer.

`fig9_synthetic_unit_tests.py` annonce sa 4e ligne comme
« Uniform noise : negative control -> false positive rate ». Mais
`pixel_prf` definit sa reference RELATIVEMENT au champ mesure
(`needs = gt > gt.mean()`) : sur un champ sans anomalie, la moyenne coupe le
bruit en deux et pres de la moitie des pixels sont declares « a raffiner ».
Il n'y a alors pas de faux positif a compter.

Defaut RAPPORTE, non corrige : choisir une reference absolue changerait les
quatre lignes de la figure. Ces tests verrouillent la DEVIATION — sa mesure
et la mention qui la porte dans le code — pour qu'elle ne soit ni oubliee ni
recorrigee en silence.
"""
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


def _pixel_prf_reference(gt):
    """La reference de fig9, isolee. Voir `fig9.pixel_prf`."""
    return gt > gt.mean()


def test_la_reference_ne_porte_aucune_information_absolue():
    """Le champ qui SEPARE : multiplier gt par 1000 doit tout changer si la
    reference etait absolue. Elle ne change rien — donc elle ne l'est pas.

    C'est la preuve structurelle, sans solveur et sans tirage.
    """
    rng = np.random.default_rng(0)
    gt = np.abs(rng.standard_normal((32, 32)))
    a = _pixel_prf_reference(gt)
    b = _pixel_prf_reference(gt * 1000.0)
    assert np.array_equal(a, b), "la reference reagit a l'echelle : D-97 serait ferme"
    c = _pixel_prf_reference(gt + 5.0)
    assert np.array_equal(a, c), "la reference reagit a un decalage constant"


def test_le_controle_negatif_declare_la_moitie_du_domaine():
    """Sur le champ SANS anomalie de fig9, la fraction declaree « a raffiner ».

    Mesure consignee : 0,4709 a N=64 (0,466 a N=256, 50 pas). Le nombre est
    ecrit pour qu'une derive se voie. Un controle negatif utile rendrait une
    fraction proche de 0.
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
        "le controle negatif ne declare plus la moitie du domaine : D-97 a "
        "peut-etre ete tranche, mettre a jour DEFAUTS.md")


def test_la_deviation_reste_ecrite_dans_le_fichier_concerne():
    """Une deviation connue non consignee la ou elle vit se fait recorriger.

    On interroge la docstring du module (son contrat lisible), pas la mise
    en forme du source.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_fig9_doc", os.path.join(_V1_LEGACY, "fig9_synthetic_unit_tests.py"))
    # On ne charge PAS le module (son bloc principal lance des simulations) :
    # on lit la docstring de la fonction via l'AST, ce qui est son contrat.
    import ast
    with open(spec.origin, encoding="utf-8") as f:
        arbre = ast.parse(f.read())
    doc = None
    for n in ast.walk(arbre):
        if isinstance(n, ast.FunctionDef) and n.name == "pixel_prf":
            doc = ast.get_docstring(n)
    assert doc is not None, "pixel_prf a disparu de fig9"
    assert "D-97" in doc, "la mention de la deviation D-97 a quitte pixel_prf"
