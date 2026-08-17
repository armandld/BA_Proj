"""D-99 — les metriques de coherence de fig3 supposaient un domaine BORNE.

La grille de ce depot est periodique (`PeriodicGrid`). `compactness` remplissait
le pourtour avec « rien n'est raffine » (`mode='constant'`) et `component_density`
appelait `label()` sans refermer les bords : toute structure traversant un bord
etait comptee comme exposee des deux cotes, et coupee en deux composantes.

Le champ qui SEPARE : un masque qui TOUCHE un bord. Sur un bloc central, les
deux conventions rendent exactement la meme valeur — un test ecrit la-dessus
passerait sans rien verifier. C'est la derniere fonction de test ci-dessous,
gardee explicitement pour cette raison.

Ces tests echouent sur la version d'avant la correction.
"""
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


def _fig3():
    """Charge les helpers de fig3 sans executer son bloc principal.

    fig3 lance des simulations au niveau module ; on n'importe donc que les
    definitions, en executant le source jusqu'au marqueur de la boucle
    principale. Les fonctions obtenues sont les VRAIES (operateur assorti),
    pas une reecriture.
    """
    chemin = os.path.join(_V1_LEGACY, "fig3_spatial_coherence.py")
    with open(chemin, encoding="utf-8") as f:
        src = f.read()
    marqueur = "#  MAIN"
    assert marqueur in src, "le marqueur de section MAIN a disparu de fig3"
    tete = src.split(marqueur)[0]
    ns = {"__name__": "_fig3_helpers", "__file__": chemin}
    exec(compile(tete, chemin, "exec"), ns)  # noqa: S102
    return ns


@pytest.fixture(scope="module")
def f3():
    ns = _fig3()
    for nom in ("compactness", "component_density"):
        assert nom in ns, "%s a disparu de fig3" % nom
    return ns


N = 256


def _bande_traversante():
    """Nappe de courant : bande verticale qui traverse tout le domaine."""
    m = np.zeros((N, N), dtype=bool)
    m[:, 112:144] = True
    return m


def _bloc_a_cheval():
    """Un bloc coupe en deux par le bord haut/bas — recolle en periodique."""
    m = np.zeros((N, N), dtype=bool)
    m[:32, :32] = True
    m[-32:, :32] = True
    return m


def _bloc_central():
    """Ne touche aucun bord : le champ qui NE SEPARE PAS."""
    m = np.zeros((N, N), dtype=bool)
    m[100:150, 100:150] = True
    return m


def test_bande_traversante_nest_plus_penalisee(f3):
    """Avant D-99 : 0,0698. Verite periodique : 0,0625 (2 cotes exposes sur 256
    colonnes, soit 512 pixels de bord pour 8192 de surface)."""
    mesure = f3["compactness"](_bande_traversante())
    assert mesure == pytest.approx(0.0625, abs=1e-4), (
        "compactness = %.4f ; 0,0625 attendu, 0,0698 etait l'ancienne valeur" % mesure)


def test_bloc_a_cheval_compte_pour_une_seule_region(f3):
    """Avant D-99 : 2 composantes. Le bloc est CONTINU a travers le bord."""
    n_comp, densite = f3["component_density"](_bloc_a_cheval())
    assert n_comp == 1, "%d composantes pour une region continue" % n_comp
    aire = 2 * 32 * 32
    assert densite == pytest.approx(1 / (aire / 1000), rel=1e-9)


def test_bloc_a_cheval_compactness(f3):
    """Avant D-99 : 0,1211 (+31,9 %). Verite periodique : 0,0918."""
    mesure = f3["compactness"](_bloc_a_cheval())
    assert mesure == pytest.approx(0.0918, abs=5e-4), (
        "compactness = %.4f ; 0,0918 attendu, 0,1211 etait l'ancienne valeur" % mesure)


def test_le_bloc_central_ne_separe_pas(f3):
    """Garde-fou explicite : sur cette entree les deux conventions coincident.

    Ecrit pour que personne ne « valide » D-99 sur un bloc central et croie
    avoir verifie quelque chose. Si cette valeur bougeait, ce serait la
    correction qui aurait deborde sur un cas qu'elle ne devait pas toucher.
    """
    assert f3["compactness"](_bloc_central()) == pytest.approx(0.0784, abs=1e-4)
    n_comp, _ = f3["component_density"](_bloc_central())
    assert n_comp == 1


def test_domaine_entierement_raffine_na_aucun_bord(f3):
    """Le cas limite qui tranche le plus nettement les deux conventions.

    Tout le domaine raffine : en periodique il n'existe AUCUN pixel de bord
    (perimetre 0). En borne, les 4 cotes du carre comptaient — 1020 pixels.
    """
    plein = np.ones((N, N), dtype=bool)
    assert f3["compactness"](plein) == 0.0
    n_comp, _ = f3["component_density"](plein)
    assert n_comp == 1


def test_masque_vide(f3):
    """Les cas degeneres restent inchanges."""
    vide = np.zeros((N, N), dtype=bool)
    assert f3["compactness"](vide) == 0.0
    assert f3["component_density"](vide) == (0, 0.0)


def test_deux_regions_reellement_disjointes_restent_deux(f3):
    """La correction ne doit pas fusionner ce qui est vraiment separe."""
    m = np.zeros((N, N), dtype=bool)
    m[40:60, 40:60] = True
    m[150:170, 150:170] = True
    n_comp, _ = f3["component_density"](m)
    assert n_comp == 2
