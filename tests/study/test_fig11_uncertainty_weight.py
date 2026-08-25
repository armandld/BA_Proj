"""D-100 — CORRIGE. Le panneau d'incertitude de fig11 affiche h ET v.

`fig11_hamiltonian_design.py` recalculait `w = exp(-((score - thr)/sigma)^2)`
sur le score PAR CELLULE, en un seul panneau. `HamiltParams.py:533-546` le
calcule sur le score moyenne PAR ARETE, avec un axe de roulement different
par direction, et en produit DEUX champs distincts (horizontal, vertical)
qui pesent `C_horiz`/`C_vert` separement — l'anisotropie mesuree a la
decouverte (aretes horizontales 4,3x plus actives que le panneau unique ne
le montrait sur `harris_tearing`) n'apparaissait pas du tout.

Correction : fig11 reproduit exactement les deux moyennes par arete du
mappeur (`uncertainty_h`, `uncertainty_v`) et affiche les deux comme deux
panneaux separes (D et E) plutot que de choisir une seule combinaison qui
masquerait l'anisotropie. Ces tests verifient la correction.
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


def _fig11_source():
    path = os.path.join(_V1_LEGACY, "fig11_hamiltonian_design.py")
    with open(path, encoding="utf-8") as f:
        return f.read(), path


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
    des aretes horizontales et verticales distinctes — ce que fig11 doit
    reproduire pour ses deux panneaux.
    """
    from Simulation.HamiltParams import PhysicalMapper
    src = sys.modules["Simulation.HamiltParams"].__file__
    with open(src, encoding="utf-8") as f:
        arbre = ast.parse(f.read())
    noms = {n.id for n in ast.walk(arbre) if isinstance(n, ast.Name)}
    assert {"uncertainty_h", "uncertainty_v"} <= noms, (
        "le mappeur ne calcule plus deux poids d'arete : la correction "
        "D-100 de fig11 n'a plus de reference a reproduire")
    assert PhysicalMapper is not None


def test_fig11_reproduit_les_deux_moyennes_par_arete_du_mappeur():
    """Verifie que fig11 calcule bien `uncertainty_h`/`uncertainty_v` avec
    les MEMES axes de roulement que `HamiltParams.py` (axis=1 horizontal,
    axis=0 vertical) — pas juste des noms qui y ressemblent.
    """
    src, path = _fig11_source()
    assert "uncertainty_h" in src and "uncertainty_v" in src, (
        f"{path} ne calcule plus deux poids d'incertitude separes : "
        "D-100 rouvert")
    assert "np.roll(score, -1, axis=1)" in src, (
        f"{path} : la moyenne par arete horizontale ne suit plus la "
        "convention de HamiltParams.py (axis=1)")
    assert "np.roll(score, -1, axis=0)" in src, (
        f"{path} : la moyenne par arete verticale ne suit plus la "
        "convention de HamiltParams.py (axis=0)")


def test_fig11_affiche_les_deux_champs_comme_deux_panneaux():
    """Le grief d'origine etait qu'un panneau UNIQUE ne peut pas montrer
    l'anisotropie h/v. Verifie que la grille est passee a 5 colonnes et que
    les deux champs sont bien affiches (deux `ax.imshow` sur les deux
    variables), pas fusionnes en un seul avant affichage.
    """
    src, path = _fig11_source()
    assert "n_scenarios, 5" in src, (
        f"{path} n'a pas de cinquieme colonne : les deux poids ne peuvent "
        "pas etre affiches separement")
    assert "imshow(field" in src or (
        "imshow(uncertainty_h" in src and "imshow(uncertainty_v" in src), (
        f"{path} n'affiche plus explicitement uncertainty_h et "
        "uncertainty_v")


def test_la_correction_D100_reste_ecrite_dans_le_fichier_concerne():
    """Une correction non consignee la ou elle vit se fait re-casser sans
    que personne ne le remarque."""
    src, path = _fig11_source()
    assert "D-100" in src, "la mention de la correction D-100 a quitte fig11"
    assert "CORRIGE" in src, (
        "fig11 ne dit plus explicitement que D-100 est corrige, pas "
        "seulement rapporte")
