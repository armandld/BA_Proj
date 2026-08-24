"""Que fait la plaquette d'un signal NUMERIQUEMENT NEGLIGEABLE ?

Sous `norm="max"` — le defaut depuis le 21 aout 2026 — chacune des deux
magnitudes de la plaquette est divisee par SON PROPRE maximum :

    K_p = -w_ZZZZ * (|omega|/max|omega| + |J|/max|J|) / max(la somme)

C'est ce qui rend a la structure faible le poids que le denominateur commun
lui refusait (facteur 179 sur `harris_tearing`). Mais la normalisation n'a
aucune notion de « ce signal a-t-il un sens physique ? » : elle remet a
l'echelle ce qu'elle trouve.

LE DEFAUT, MESURE
-----------------
Le seul garde est `EPS = 1e-10`, qui est un garde de DIVISION PAR ZERO, pas
un seuil physique. Il produit une marche :

    max|omega| = 4,65e-15  ->  K au pic d'omega = 0,000000   (sous EPS)
    max|omega| = 4,65e-10  ->  K au pic d'omega = 0,999998   (sur EPS)
    max|omega| = 4,65e-01  ->  K au pic d'omega = 0,999998

Une vorticite de 1e-9 — de la poussiere numerique — pese donc AUTANT qu'une
vorticite de 1. Sous `legacy`, le denominateur commun l'ecrasait
naturellement (0,000000 contre 0,500000) : le defaut est le revers exact de
la correction.

POURQUOI RIEN N'EST CORRIGE ICI
-------------------------------
Le corpus n'entre pas dans la bande. Balaye sur les 24 artefacts DNS,
instantane par instantane : **aucun** `max|omega|` ni `max|J|` ne tombe dans
`(1e-10, 1e-6)`. Les valeurs sont soit EXACTEMENT nulles (v ou B identiquement
nul a t=0), ce qui est sur, soit >= 4,9e-02 — quatre ordres de grandeur
au-dessus de la bande. `harris_tearing` passe de 0 exact a 1,29e-04 en un
instantane.

Corriger demanderait de choisir un plancher physique (par exemple « au-dessus
du niveau d'arrondi du champ », `pic > k * eps_machine * max|v| / dx`), et
c'est une decision de conception sur `src/`, pas une retouche. Ce fichier
epingle la marche pour qu'elle ne surprenne personne, et surveille le corpus
pour qu'un futur artefact dans la bande fasse rougir la suite au lieu
d'entrer en silence.
"""
import glob
import os
import sys

import numpy as np
import pytest

_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_RACINE, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_RACINE, "src"))

from Simulation.HamiltParams_v2 import PhysicalMapperV2       # noqa: E402
from Simulation.grid import curl_z                            # noqa: E402

N = 16
#: La bande ou un signal est trop grand pour etre eteint par EPS et trop
#: petit pour etre physique. Bornes MESUREES, pas choisies : la basse est
#: `PhysicalMapperV2.EPS`, la haute est deux ordres sous le plus petit pic
#: non nul du corpus (1,29e-04 sur harris).
_BANDE = (PhysicalMapperV2.EPS, 1e-6)


def _grille():
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.meshgrid(x, x, indexing="ij")


def _bosse(Y, centre, sigma=0.45):
    return np.exp(-((Y - centre) ** 2) / (2 * sigma ** 2))


def _K_au_pic_domega(amplitude_v, norm):
    """Valeur de la plaquette la ou omega culmine, rapportee a son pic.

    Tourbillon en y=pi/2, nappe en y=3pi/2 : supports DISJOINTS, donc la
    valeur lue au pic d'omega ne doit rien a J.
    """
    X, Y = _grille()
    Z = np.zeros((N, N))
    vx = -amplitude_v * _bosse(Y, np.pi / 2)
    Bx = _bosse(Y, 3 * np.pi / 2)
    om = np.abs(curl_z(vx, Z, True))
    m = PhysicalMapperV2(dx=2 * np.pi / N, norm=norm)
    k = np.abs(m.compute_coefficients(
        None, np.full((N, N), 0.5),
        dict(vx=vx, vy=Z, Bx=Bx, By=Z, Jz=Z), 0.15)["K_plaquettes"])
    if om.max() == 0.0:
        return 0.0, 0.0
    ou = np.unravel_index(np.argmax(om), om.shape)
    return float(k[ou]) / float(k.max()), float(om.max())


# ------------------------------------------------------------------
#  1. la marche, epinglee
# ------------------------------------------------------------------
def test_sous_max_un_signal_juste_au_dessus_de_EPS_est_promu_a_pleine_echelle():
    """Le defaut, tel qu'il est. Mesure, pas suppose.

    Sur quelle entree ce test echoue : le jour ou quelqu'un ajoute un
    plancher physique a la normalisation par signal — auquel cas ce fichier
    doit etre reecrit, et `DEFAUTS.md` mis a jour.
    """
    sous, pic_sous = _K_au_pic_domega(1e-14, "max")
    sur, pic_sur = _K_au_pic_domega(1e-9, "max")
    fort, _ = _K_au_pic_domega(1.0, "max")

    assert pic_sous < PhysicalMapperV2.EPS < pic_sur, (
        f"le cas teste n'encadre plus EPS : {pic_sous:.2e} et {pic_sur:.2e}")
    assert sous == pytest.approx(0.0, abs=1e-9), (
        f"sous EPS la vorticite pese {sous:.6f} au lieu de 0")
    assert sur > 0.99, (
        f"juste au-dessus d'EPS la vorticite pese {sur:.6f} : la marche a "
        "disparu, un plancher a peut-etre ete ajoute — relire ce fichier")
    assert sur == pytest.approx(fort, abs=1e-6), (
        f"1e-9 pese {sur:.6f} et 1,0 pese {fort:.6f} : ils ne sont plus "
        "indiscernables, la marche a change de forme")


def test_sous_legacy_le_meme_signal_negligeable_reste_negligeable():
    """Le champ qui SEPARE : la marche est propre a `max`.

    Sous `legacy` le denominateur commun ecrase naturellement la structure
    faible — c'est le defaut que `max` corrige, et c'est aussi ce qui la
    protegeait de la poussiere numerique. Les deux faits sont le meme fait.
    """
    negligeable, _ = _K_au_pic_domega(1e-9, "legacy")
    fort, _ = _K_au_pic_domega(1.0, "legacy")
    assert negligeable < 1e-6, (
        f"sous `legacy` une vorticite de 1e-9 pese {negligeable:.3e} : le "
        "denominateur commun ne l'ecrase plus, les deux modes ont converge")
    assert fort > 0.4, (
        f"sous `legacy` une vorticite de 1,0 ne pese que {fort:.3f} : ce "
        "test ne compare plus rien")


# ------------------------------------------------------------------
#  2. le corpus reste-t-il hors de la bande ?
# ------------------------------------------------------------------
def _pics_du_corpus(repertoire=None):
    """Rend (liste des pics dans la bande, nombre d'instantanes balayes).

    `repertoire` est parametrable pour que le PLANCHER soit lui-meme
    testable : un plancher qu'on ne peut pas faire tomber n'est pas un garde.
    """
    racine = os.path.join(_RACINE, "results") if repertoire is None else repertoire
    dans_la_bande, n_instantanes = [], 0
    for chemin in sorted(glob.glob(os.path.join(racine, "dns_*.npz"))):
        try:
            d = np.load(chemin)
        except Exception:
            continue
        for si in range(len(d["vx"])):
            n_instantanes += 1
            for nom, (a, b) in (("omega", ("vx", "vy")), ("J", ("Bx", "By"))):
                pic = float(np.max(np.abs(curl_z(d[a][si].astype(float),
                                                 d[b][si].astype(float), True))))
                if _BANDE[0] < pic < _BANDE[1]:
                    dans_la_bande.append(
                        (os.path.basename(chemin), si, nom, pic))
    return dans_la_bande, n_instantanes


def _verifie_plancher(n, plancher=300):
    assert n >= plancher, (
        f"{n} instantanes balayes ; 480 mesures le 22 aout 2026 — le "
        "balayage a retreci, il ne prouve plus ce qu'il prouvait")


def test_le_plancher_de_balayage_tombe_sur_un_repertoire_vide(tmp_path):
    """Le garde du garde."""
    _, n = _pics_du_corpus(str(tmp_path))
    assert n == 0
    with pytest.raises(AssertionError, match="le balayage a retreci"):
        _verifie_plancher(n)


@pytest.mark.slow
def test_aucun_instantane_du_corpus_nentre_dans_la_bande():
    """Le garde qui compte : un futur artefact dans la bande doit crier.

    Mesure le 22 aout 2026 sur les 24 artefacts DNS : 0 instantane dans
    `(1e-10, 1e-6)`. Les pics sont soit exactement nuls, soit >= 4,9e-02.
    """
    dans, n = _pics_du_corpus()
    _verifie_plancher(n)
    assert not dans, (
        f"{len(dans)} pic(s) dans la bande ou EPS promeut la poussiere "
        f"numerique a pleine echelle : {dans[:5]}. Sous `norm=\"max\"` ce "
        "signal pese autant qu'une structure reelle — voir DEFAUTS.md.")


@pytest.mark.slow
def test_les_pics_nuls_du_corpus_sont_EXACTEMENT_nuls():
    """Ce qui rend la bande inatteignable aujourd'hui : les champs vides le
    sont exactement (v = 0 a t=0), pas approximativement. Un `1e-12` a la
    place d'un `0` serait sur ; un `1e-9` ne le serait pas.
    """
    racine = os.path.join(_RACINE, "results")
    fichiers = sorted(glob.glob(os.path.join(racine, "dns_*_Re400_N96.npz")))
    if not fichiers:
        pytest.skip("artefacts DNS N=96 absents")
    vus_nuls = 0
    for chemin in fichiers:
        d = np.load(chemin)
        for si in range(len(d["vx"])):
            for a, b in (("vx", "vy"), ("Bx", "By")):
                pic = float(np.max(np.abs(curl_z(d[a][si].astype(float),
                                                 d[b][si].astype(float), True))))
                if pic == 0.0:
                    vus_nuls += 1
                else:
                    assert pic >= _BANDE[1], (
                        f"{os.path.basename(chemin)} instantane {si} : pic "
                        f"{pic:.3e}, ni nul ni franchement physique")
    assert vus_nuls > 0, (
        "aucun champ identiquement nul dans le corpus : ce test ne verifie "
        "plus l'hypothese qu'il pretend verifier")
