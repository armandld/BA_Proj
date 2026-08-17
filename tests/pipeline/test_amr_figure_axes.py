"""D-68 — l'orientation de la figure AMR que la boucle fermée publie.

`plot_amr_state` est la seule fonction de `src/visual.py` et de
`src/help_visual.py` qui s'exécute en production : `pipeline.py` l'appelle
quatre fois par pas de verrouillage et sauve un PNG à chaque fois.

Ce que ces tests séparent :

- ce qui était FAUX — les deux étiquettes nommaient l'axe de l'autre ;
- ce qui était JUSTE et doit le rester — le champ et les cadres sont
  cohérents entre eux, le cadre tombe bien sur la structure qu'il désigne.

**D-68 est clos par transposition** (décision de USER). `plot_amr_state`
était le seul des trois traceurs du dépôt à mettre Y en horizontal ; il met
désormais X, comme `plot_recursive_state` et `help_visual.plot_field`. Le
test qui gardait la frontière de décision a été REMESURÉ, pas supprimé : il
dit maintenant l'inverse, avec la mesure avant/après.

Le champ d'essai SÉPARE : une structure en (X=10, Y=40) est asymétrique
sous transposition. Sur un champ symétrique (une gaussienne centrée, un
Taylor-Green), les deux orientations rendent la même image et ces tests
passeraient sans rien vérifier.

Les assertions portent sur les objets matplotlib RENDUS — l'`Axes` que la
fonction a réellement construit — et non sur le texte du source : trois
tests du dépôt ont déjà cassé sur des changements voulus pour avoir lu des
chaînes dans un fichier.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from Simulation.grid import AXIS_X, AXIS_Y
import visual
from visual import plot_amr_state


N = 64
X_BLOB, Y_BLOB = 10, 40          # au sens de grid.py : axis0 = X, axis1 = Y


class _FakeSim:
    """Le seul contrat que `plot_amr_state` consomme de `sim`."""

    def __init__(self, Jz):
        self._Jz = Jz

    def get_fluxes(self):
        return {'vx': None, 'vy': None, 'Bx': None, 'By': None, 'Jz': self._Jz}


def _champ_qui_separe():
    """Une seule cellule brillante, hors diagonale : Jz != Jz.T."""
    Jz = np.zeros((N, N))
    Jz[X_BLOB, Y_BLOB] = 1.0
    return Jz


def _bounds_autour_de_la_structure():
    # bounds = (a0_s, a0_e, a1_s, a1_e) — l'ordre que `get_periodic_patch`
    # consomme : les deux premiers indexent l'axe 0, les deux autres l'axe 1.
    return (X_BLOB - 2, X_BLOB + 2, Y_BLOB - 2, Y_BLOB + 2)


@pytest.fixture
def axes_rendus(tmp_path, monkeypatch):
    """Rejoue l'appel de `pipeline.py` et rend l'`Axes` réellement construit.

    `plot_amr_state` ferme sa figure quand `save_dir` est fourni sans
    `verbose` : on neutralise la fermeture le temps de la mesure, sans
    toucher au reste du chemin.
    """
    monkeypatch.setattr(visual.plt, "close", lambda *a, **k: None)
    out = str(tmp_path / "figs")
    patch = {'bounds': _bounds_autour_de_la_structure(), 'depth': 1}

    plot_amr_state(_FakeSim(_champ_qui_separe()), [patch], 0.0, 3,
                   verbose=False, save_dir=out, suffix="probe")

    assert os.listdir(out), "aucun PNG produit : la mesure ne porterait sur rien"
    ax = plt.gcf().axes[0]
    yield ax
    plt.close("all")


def _quantite_de_l_etiquette(label):
    """« Grid Y (axe 1 du tableau) » → 'Y'. Lève si l'étiquette ne tranche pas."""
    a_x, a_y = "Grid X" in label, "Grid Y" in label
    assert a_x != a_y, f"étiquette qui ne nomme pas un axe unique : {label!r}"
    return "X" if a_x else "Y"


# ═══════════════════════════════════════════════════════════════════════
#  1. Ce qui était faux : les étiquettes nommaient l'axe de l'autre
# ═══════════════════════════════════════════════════════════════════════

def test_une_structure_se_relit_a_la_position_ou_on_l_a_mise(axes_rendus):
    """ÉCHOUE sur l'ancienne version : X=10 s'y lisait « X=40 ».

    Lecture bout en bout, telle qu'un humain la fait sur le PNG : on prend
    la position de la structure sur l'axe horizontal, on lit le nom que
    l'axe porte, et on compare à la position où grid.py l'a mise.
    """
    ligne, colonne = np.unravel_index(np.argmax(_champ_qui_separe()), (N, N))
    vrai = {"X": ligne, "Y": colonne}            # grid.py : axis0 = X, axis1 = Y

    # Où l'image place réellement la structure, mesuré sur l'AxesImage rendu.
    (image,) = axes_rendus.get_images()
    tableau = np.asarray(image.get_array())
    v_rendu, h_rendu = np.unravel_index(np.argmax(tableau), tableau.shape)

    lu = {_quantite_de_l_etiquette(axes_rendus.get_xlabel()): h_rendu,
          _quantite_de_l_etiquette(axes_rendus.get_ylabel()): v_rendu}

    assert lu == vrai, (
        f"structure posée en X={vrai['X']}, Y={vrai['Y']} ; "
        f"relue sur la figure X={lu.get('X')}, Y={lu.get('Y')}")


def test_les_deux_etiquettes_ne_nomment_pas_le_meme_axe(axes_rendus):
    """Un garde-fou : la correction ne doit pas écrire « Y » des deux côtés."""
    assert (_quantite_de_l_etiquette(axes_rendus.get_xlabel())
            != _quantite_de_l_etiquette(axes_rendus.get_ylabel()))


# ═══════════════════════════════════════════════════════════════════════
#  2. Ce qui était juste : le cadre tombe sur la structure — à ne pas casser
# ═══════════════════════════════════════════════════════════════════════

def test_le_cadre_encadre_la_structure_qu_il_designe(axes_rendus):
    """ÉPINGLE l'ancien comportement : la géométrie ne doit PAS bouger.

    Ce test échoue si quelqu'un « corrige » D-68 en transposant l'image
    sans transposer les cadres — ou l'inverse. C'est la moitié saine du
    module, vérifiée avant d'être laissée en place.
    """
    (image,) = axes_rendus.get_images()
    tableau = np.asarray(image.get_array())
    v_struct, h_struct = np.unravel_index(np.argmax(tableau), tableau.shape)

    cadres = [p for p in axes_rendus.patches
              if isinstance(p, matplotlib.patches.Rectangle)
              and p.get_width() and p.get_height()]
    assert cadres, "aucun cadre dessiné : le test ne mesurerait rien"

    (cadre,) = cadres
    gauche, bas = cadre.get_xy()
    droite = gauche + cadre.get_width()
    haut = bas + cadre.get_height()

    assert gauche <= h_struct <= droite and bas <= v_struct <= haut, (
        f"cadre h[{gauche},{droite}] v[{bas},{haut}] ne tombe pas sur la "
        f"structure h={h_struct} v={v_struct} : champ et cadres ont cessé "
        "d'être cohérents")


def test_le_champ_est_transpose_pour_mettre_X_en_horizontal(axes_rendus):
    """SEUIL REMESURÉ — la décision a été prise, ce test dit l'inverse.

    Ce test exigeait auparavant que le champ ne soit **pas** transposé : il
    gardait la frontière de décision, pour qu'elle ne se franchisse pas en
    silence. Elle a été franchie explicitement par USER, donc le seuil se
    **remesure**, il ne se supprime pas.

    Ce qui a motivé la décision : `plot_amr_state` était le seul des trois
    traceurs du dépôt à mettre Y en horizontal. `plot_recursive_state`
    (même fichier) trace `state['Jz'].T` et `help_visual.plot_field` trace
    `grid.X.T` en étiquetant « X » l'axe horizontal. L'objection « cela
    change des PNG publiés » ne tient plus : toutes les figures sont
    regénérées après la campagne.

    Mesure, structure placée en X=10, Y=40 au sens de `grid.py` :

        avant  — lue « X=40, Y=10 » sur la figure
        après  — lue « X=10, Y=40 »

    L'autre moitié du module reste vraie et est vérifiée juste au-dessus :
    champ et cadres restent cohérents entre eux. Transposer l'un sans
    l'autre ferait tomber ce test-là.
    """
    (image,) = axes_rendus.get_images()
    rendu = np.asarray(image.get_array())
    brut = _champ_qui_separe()

    assert np.array_equal(rendu, brut.T), (
        "le champ passé à imshow n'est pas la transposée du tableau brut : "
        "D-68 a été résolu par transposition, voir DEFAUTS.md")
    assert not np.array_equal(rendu, brut), (
        "rendu identique au brut : le champ d'essai ne sépare plus les deux "
        "orientations, ce test ne vérifierait rien")

    # La structure doit se lire à sa vraie place : X en horizontal.
    v, h = np.unravel_index(np.argmax(rendu), rendu.shape)
    assert (h, v) == (X_BLOB, Y_BLOB), (
        f"structure lue en X={h}, Y={v} ; attendu X={X_BLOB}, Y={Y_BLOB}")


# ═══════════════════════════════════════════════════════════════════════
#  3. La déviation reste écrite là où elle vit
# ═══════════════════════════════════════════════════════════════════════

def test_la_deviation_reste_consignee_dans_le_fichier():
    """Une déviation connue non écrite se fait recorriger par erreur.

    Seul test de ce fichier qui lise le source, et il le fait exprès : la
    règle « ne jamais laisser une déviation connue non écrite » demande
    qu'un test vérifie que la mention reste dans le fichier concerné.
    """
    src = open(visual.__file__, encoding="utf-8").read()
    assert "D-68" in src, (
        "la mention de D-68 a disparu de src/visual.py : la raison de ne "
        "pas transposer n'est plus lisible là où elle vit")


# ═══════════════════════════════════════════════════════════════════════
#  4. Le champ d'essai sépare vraiment — sinon rien de ce qui précède ne vaut
# ═══════════════════════════════════════════════════════════════════════

def test_le_champ_d_essai_distingue_les_deux_orientations():
    """Sur un champ symétrique, tous les tests ci-dessus passeraient à vide."""
    Jz = _champ_qui_separe()
    assert not np.array_equal(Jz, Jz.T), "champ symétrique : ne sépare rien"
    ligne, colonne = np.unravel_index(np.argmax(Jz), Jz.shape)
    assert ligne != colonne, "structure sur la diagonale : ne sépare rien"
    assert (AXIS_X, AXIS_Y) == (0, 1), (
        "la convention de grid.py a changé : ces tests décodent X depuis "
        "l'axe 0 et doivent être remesurés")
