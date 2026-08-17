"""D-100 — le poids d'incertitude du panneau D de fig11 n'est pas celui du hamiltonien.

`fig11_hamiltonian_design.py:102` recalcule `w = exp(-((score - thr)/sigma)^2)`
sur le score PAR CELLULE. `HamiltParams.py:469-473` le calcule sur le score
moyenne PAR ARETE et en produit deux champs distincts (horizontal, vertical).

Defaut RAPPORTE, non corrige : choisir ce qu'on affiche (w_h, w_v, leur moyenne,
leur max) est un choix de presentation. Ces tests verrouillent la DEVIATION —
sa mesure et la mention qui la porte dans le code.
"""
import ast
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_V1_LEGACY = os.path.join(_REPO_ROOT, "figures", "v1_legacy")
if _V1_LEGACY not in sys.path:
    sys.path.insert(0, _V1_LEGACY)

import matplotlib  # noqa: E402
matplotlib.use("Agg")


def test_le_moyennage_par_arete_change_le_poids():
    """Le champ qui SEPARE : un score qui varie le long d'UN SEUL axe.

    Sur un tel champ, la moyenne par arete horizontale deplace le score dans
    la bande d'incertitude alors que la moyenne verticale ne le deplace pas —
    exactement ce que la nappe de Harris produit. Sur un score CONSTANT, les
    trois versions coincident : ce serait le champ qui ne separe pas.
    """
    thr, sigma = 0.3044, 0.05
    n = 64
    # rampe le long de l'axe 1 seulement
    x = np.linspace(0.0, 0.6, n)
    score = np.tile(x, (n, 1))

    w_cell = np.exp(-((score - thr) / sigma) ** 2)
    s_h = 0.5 * (score + np.roll(score, -1, axis=1))
    s_v = 0.5 * (score + np.roll(score, -1, axis=0))
    w_h = np.exp(-((s_h - thr) / sigma) ** 2)
    w_v = np.exp(-((s_v - thr) / sigma) ** 2)

    assert not np.allclose(w_cell, w_h), (
        "le moyennage horizontal ne change rien : ce champ ne separe pas")
    assert np.allclose(w_cell, w_v), (
        "le moyennage vertical ne devrait rien changer sur une rampe en axe 1")

    # champ qui NE SEPARE PAS, garde explicitement
    plat = np.full((n, n), thr)
    assert np.allclose(np.exp(-((plat - thr) / sigma) ** 2),
                       np.exp(-((0.5 * (plat + np.roll(plat, -1, axis=1)) - thr) / sigma) ** 2))


def test_le_mappeur_produit_bien_deux_champs_de_poids():
    """Interroge le module, pas le texte : `compute_coefficients` doit rendre
    des aretes horizontales et verticales distinctes, ce que le panneau
    unique de fig11 ne peut pas representer.
    """
    from Simulation.HamiltParams import PhysicalMapper
    src = sys.modules["Simulation.HamiltParams"].__file__
    with open(src, encoding="utf-8") as f:
        arbre = ast.parse(f.read())
    noms = {n.id for n in ast.walk(arbre) if isinstance(n, ast.Name)}
    assert {"uncertainty_h", "uncertainty_v"} <= noms, (
        "le mappeur ne calcule plus deux poids d'arete : D-100 a peut-etre "
        "ete tranche, mettre a jour DEFAUTS.md")
    assert PhysicalMapper is not None


def test_la_deviation_reste_ecrite_dans_le_fichier_concerne():
    """Une deviation connue non consignee la ou elle vit se fait recorriger."""
    chemin = os.path.join(_V1_LEGACY, "fig11_hamiltonian_design.py")
    with open(chemin, encoding="utf-8") as f:
        src = f.read()
    assert "D-100" in src, "la mention de la deviation D-100 a quitte fig11"
    # la mention doit accompagner le calcul concerne, pas vivre en tete de fichier
    i_mention = src.index("D-100")
    i_calcul = src.index("uncertainty = np.exp(")
    assert 0 < i_calcul - i_mention < 1500, (
        "la mention de D-100 s'est eloignee du calcul qu'elle documente")
