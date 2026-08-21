#!/usr/bin/env python3
"""D-86 — la phase 10a rendait le bord gauche de sa grille comme un optimum.

`analyse_snapshots` choisissait `c_bias*` par `np.argmax(f1_grid)`. Sur une
courbe PLATE, `argmax` rend l'indice 0 : `c_bias* = 0.1`, la borne basse de
`np.logspace(-1, 2, 25)` — le point le plus domine par les couplages, donc
l'oppose de ce que le balayage cherche. Mesure : 14 balayages plats sur 52
configurations parcourues (0/16 a dim=2, 5/16 a dim=4 N=96, 8/16 a dim=8,
1/4 a dim=4 N=256), tous avec un ecart max-min exactement nul, contre 0,125
a 0,433 pour les 38 informatifs.

Ces tests interrogent le comportement du module, pas le texte de son source.
Ils echouent tous sur la version d'avant D-86 sauf le dernier, qui epingle
l'ancien comportement d'`argmax` et passe des deux cotes : c'est lui qui
empeche la correction d'etre defaite en silence.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h2b_analytical_solution as p10a
from h2b_analytical_solution import mean_over_informative
from h2b_train_linear_hamiltonian import build_init_map


C_GRID = np.logspace(-1.0, 2.0, 25)


# ------------------------------------------------------------------
#  Le drapeau lui-meme
# ------------------------------------------------------------------

def _fake_result(f1_grid):
    """Rejoue la seule partie de `analyse_snapshots` qui decide, sans
    reconstruire un hamiltonien : la lecture de la courbe."""
    f1_grid = np.asarray(f1_grid, dtype=float)
    span = float(f1_grid.max() - f1_grid.min())
    return span, bool(span <= p10a.F1_SPAN_TOL), float(C_GRID[int(np.argmax(f1_grid))])


def test_une_courbe_plate_est_declaree_degeneree():
    """Le cas mesure : F1(c) identiquement nul sur toute la grille."""
    span, degen, c_star = _fake_result(np.zeros(25))
    assert span == 0.0
    assert degen is True
    # et c'est bien le bord GAUCHE que l'argmax rendait
    assert c_star == pytest.approx(0.1)


def test_une_courbe_plate_non_nulle_est_aussi_degeneree():
    """Le critere est la platitude, pas la nullite : un plateau a F1=0,4 ne
    designe pas davantage un optimum qu'un plateau a 0."""
    _, degen, _ = _fake_result(np.full(25, 0.4))
    assert degen is True


def test_une_courbe_informative_ne_l_est_pas():
    """Le plus petit ecart mesure parmi les 38 balayages informatifs vaut
    0,125 — quatre ordres au-dessus de la tolerance."""
    g = np.zeros(25)
    g[17] = 0.125
    span, degen, c_star = _fake_result(g)
    assert span == pytest.approx(0.125)
    assert degen is False
    assert c_star == pytest.approx(float(C_GRID[17]))


def test_la_tolerance_laisse_la_separation_mesuree_intacte():
    """Aucune zone grise : la tolerance est a douze ordres sous le plus
    petit ecart informatif observe."""
    assert p10a.F1_SPAN_TOL < 0.125 / 1e10


# ------------------------------------------------------------------
#  L'agregation — c'est la que le mauvais nombre se propageait
# ------------------------------------------------------------------

def test_un_balayage_degenere_ne_tire_plus_la_moyenne_du_scenario():
    """Cas mesure sur mhd_rotor (N=96, dim=4) : c* par Re valait
    [74,99 ; 100 ; 100 ; 0,10], la derniere ligne etant degeneree."""
    rows = [dict(c_bias_star=c, degenerate=d) for c, d in
            [(74.9894, False), (100.0, False), (100.0, False), (0.1, True)]]

    ancien = float(np.mean([r["c_bias_star"] for r in rows]))   # avant D-86
    nouveau = mean_over_informative(rows, "c_bias_star")

    assert ancien == pytest.approx(68.7724, abs=1e-3)
    assert nouveau == pytest.approx(91.6631, abs=1e-3)
    # l'ecart se voit surtout dans l'espace ou la phase 10 lit ce nombre
    assert np.log10(nouveau) - np.log10(ancien) == pytest.approx(0.1246, abs=1e-3)


def test_un_scenario_entierement_degenere_rend_NaN_et_pas_un_nombre():
    """harris_tearing : 4 Re sur 4 degeneres. L'ancienne moyenne rendait
    0,1000 — un nombre fini, plausible, et qui ne mesure rien."""
    rows = [dict(c_bias_star=0.1, degenerate=True) for _ in range(4)]
    assert float(np.mean([r["c_bias_star"] for r in rows])) == pytest.approx(0.1)
    assert np.isnan(mean_over_informative(rows, "c_bias_star"))


# ------------------------------------------------------------------
#  Le consommateur — sans lui le drapeau serait decoratif
# ------------------------------------------------------------------

def test_la_phase_10_n_initialise_pas_sur_une_ligne_degeneree():
    tags = ["cfg:harris_tearing_Re400", "cfg:orszag_tang_Re400"]
    init, n_skip = build_init_map(tags, [0.661, 0.160], [0.1, 100.0], [True, False])
    assert "cfg:harris_tearing_Re400" not in init
    assert "cfg:orszag_tang_Re400" in init
    assert init["cfg:orszag_tang_Re400"][0] == pytest.approx(2.0)


def test_la_phase_10_ecarte_aussi_un_NaN_de_scenario():
    init, n_skip = build_init_map(["scenario:harris_tearing"], [np.nan], [np.nan], [True])
    assert init == {}


def test_epingle_l_ancienne_init_au_bord_du_domaine():
    """EPINGLE l'ancien comportement, par la fonction reelle : sans drapeau
    (l'artefact d'avant D-86 n'en portait pas), la ligne harris_tearing
    entrait dans `init_map` et y valait `log10 c_bias = -1`, la borne basse
    du domaine. Passe des deux cotes — c'est ce qui fait echouer la suite le
    jour ou le filtre disparait, en montrant ce qu'on retrouverait."""
    init, n_skip = build_init_map(
        ["cfg:harris_tearing_Re400"], [0.661], [0.1], [False])
    assert n_skip == 0
    assert init["cfg:harris_tearing_Re400"][0] == pytest.approx(-1.0)
    # ... et avec le drapeau, la meme ligne ne produit plus d'init du tout
    init2, n_skip2 = build_init_map(
        ["cfg:harris_tearing_Re400"], [0.661], [0.1], [True])
    assert (init2, n_skip2) == ({}, 1)


# ------------------------------------------------------------------
#  Le garde de campagne, sur les vraies donnees
# ------------------------------------------------------------------

def _artefacts_presents(scenario, res, N, dim):
    from config import RESULTS_DIR
    return all(os.path.exists(os.path.join(RESULTS_DIR, f))
               for r in res
               for f in (f"dns_{scenario}_Re{r}_N{N}.npz",
                         f"patches_{scenario}_Re{r}_N{N}_dim{dim}.npz"))


@pytest.mark.skipif(not _artefacts_presents("harris_tearing", [400], 96, 4),
                    reason="artefacts DNS/patches harris_tearing N=96 absents")
def _balayage_harris(norm, c_grid=None, re=400):
    """Le balayage reel de D-86, dans la normalisation demandee."""
    import Simulation.HamiltParams_v2 as HP2
    from config import RESULTS_DIR
    anciens = HP2.PhysicalMapperV2.__init__.__defaults__
    HP2.PhysicalMapperV2.__init__.__defaults__ = tuple(
        norm if d in ("legacy", "max") else d for d in anciens)
    # Le garde du garde : si le forcage devenait un no-op — signature
    # changee, defaut deplace — tous les tests de ce fichier passeraient en
    # mesurant deux fois la MEME normalisation, sans que rien ne crie.
    assert HP2.PhysicalMapperV2().norm == norm, (
        f"le forcage de norm={norm!r} n'a pas pris : la signature de "
        "PhysicalMapperV2.__init__ a change, ce fichier ne mesure plus rien")
    try:
        return p10a.analyse_snapshots(
            os.path.join(RESULTS_DIR, f"dns_harris_tearing_Re{re}_N96.npz"),
            os.path.join(RESULTS_DIR,
                         "patches_harris_tearing_Re400_N96_dim4.npz"),
            4, c_grid=C_GRID if c_grid is None else c_grid,
            max_snaps=8, seed=0)
    finally:
        HP2.PhysicalMapperV2.__init__.__defaults__ = anciens


def test_le_cas_mesure_sur_les_vraies_donnees():
    """Le balayage reel qui a revele D-86, rejoue de bout en bout.

    harris_tearing Re=400, N=96, dim=4, 8 instantanes, graine 0. C'est le
    seul test du fichier qui traverse la construction du hamiltonien et la
    descente de champ moyen ; les autres portent sur ce qu'on en fait.

    **La degenerescence est propre a `norm="legacy"`.** Mesure le 21 aout
    2026, apres le basculement du defaut : sous `max`, les 4 Re rendent
    `f1_span` = 0,550 a 0,566 et AUCUN n'est degenere. Le balayage plat de
    D-86 etait donc, sur cette configuration, un artefact de la
    normalisation historique — le hamiltonien de champ moyen n'y separait
    rien parce que ses coefficients ne separaient rien.
    """
    r = _balayage_harris("legacy")

    # la courbe ne separe rien : F1 identiquement nul sur les 25 points
    assert r["f1_span"] == 0.0
    assert r["f1_grid"].max() == 0.0
    assert r["degenerate"] is True

    # le nombre que l'ancienne version publiait comme optimum, inchange :
    # c'est bien le bord GAUCHE de la grille, pas un maximum
    assert r["c_bias_star"] == pytest.approx(0.1)
    assert r["f1_mf"] == 0.0

    # et pourtant l'indicateur classique, lui, mesure quelque chose sur la
    # meme configuration — la degenerescence est celle du champ moyen
    assert r["classical_f1"] == pytest.approx(0.745, abs=5e-3)


def test_sous_le_defaut_actuel_le_balayage_nest_plus_degenere():
    """Le champ qui SEPARE : sans lui, le test ci-dessus laisserait croire
    que la platitude est une propriete de la configuration.

    Sur quelle entree ce test echoue : si `max` cessait de donner de la
    resolution a ce balayage, la justification du basculement du defaut
    perdrait son argument le plus fort.
    """
    r = _balayage_harris("max")
    assert r["degenerate"] is False, "le balayage est redevenu plat sous `max`"
    assert r["f1_span"] > 0.4, (
        f"f1_span = {r['f1_span']:.4f} sous `max`, mesure 0,566")
    # le hamiltonien mesure enfin quelque chose — et reste SOUS le classique
    assert r["f1_grid"].max() < r["classical_f1"], (
        "le hamiltonien de champ moyen depasse la baseline classique : "
        "c'est un resultat, il se publie, il ne se glisse pas dans un test")


def test_loptimum_du_balayage_est_AU_BORD_de_la_grille_donc_illisible():
    """D-86 en MIROIR, trouve en corrigeant D-86.

    Sous `max` la courbe monte de facon monotone et `argmax` rend le bord
    DROIT de `logspace(-1, 2)` — 100,0 sur 3 Re sur 4. Un optimum au bord
    n'est pas un optimum, exactement comme le bord gauche ne l'etait pas.

    Mesure en elargissant a `logspace(-1, 5)` : F1 sature a 0,6333 des
    `c_bias` ~ 251, et les six derniers points sont identiques. L'optimum
    est donc la limite BIAIS SEUL — les couplages n'apportent rien de
    positif — et il reste sous la baseline classique (0,745).

    Ce test epingle le defaut, il ne le corrige pas : elargir la grille est
    une decision qui change tous les `c_bias*` publies.
    """
    r = _balayage_harris("max")
    assert np.isclose(r["c_bias_star"], C_GRID[-1]), (
        f"c_bias* = {r['c_bias_star']:.4g} n'est plus au bord droit "
        f"({C_GRID[-1]:.4g}) : la grille a change, remesurer")

    large = np.logspace(-1.0, 5.0, 31)
    rl = _balayage_harris("max", c_grid=large)
    assert rl["f1_grid"].max() > r["f1_grid"].max(), (
        "elargir la grille n'ameliore plus le F1 : le bord n'etait donc pas "
        "contraignant, relire ce test")
    assert np.allclose(rl["f1_grid"][-6:], rl["f1_grid"][-1]), (
        "F1 ne sature plus en haut de grille : la limite biais-seul a bouge")
    assert rl["f1_grid"].max() < rl["classical_f1"], (
        "meme a la saturation le hamiltonien depasse le classique — resultat "
        "majeur, a publier et non a epingler ici")


@pytest.mark.skipif(not _artefacts_presents("harris_tearing",
                                            [400, 800, 1200, 1600], 96, 4),
                    reason="artefacts DNS/patches harris_tearing N=96 absents")
def test_une_campagne_entierement_degeneree_leve(monkeypatch):
    """Meme regle que D-56, un cran plus bas : les entrees existaient, c'est
    le BALAYAGE qui n'a rien mesure. L'artefact s'ecrivait quand meme, avec
    `c_bias* = 0,1` sur chaque ligne.

    harris_tearing a dim=4, N=96 est le cas mesure : 4 Re sur 4 degeneres
    SOUS `norm="legacy"`, que ce test force explicitement. Sous le defaut
    actuel (`max`) les 4 Re sont informatifs et la campagne n'a plus a lever :
    la garde reste necessaire, le cas qui la declenche a change.

    La campagne leve avant d'ecrire, donc ce test ne touche pas `results/`.
    """
    import Simulation.HamiltParams_v2 as HP2
    anciens = HP2.PhysicalMapperV2.__init__.__defaults__
    HP2.PhysicalMapperV2.__init__.__defaults__ = tuple(
        "legacy" if d in ("legacy", "max") else d for d in anciens)
    assert HP2.PhysicalMapperV2().norm == "legacy", (
        "le forcage de norm='legacy' n'a pas pris : ce test ne mesure plus "
        "le cas degenere qu'il pretend mesurer")
    monkeypatch.setattr(sys, "argv", [
        "h2b_analytical_solution", "--scenario", "harris_tearing",
        "--dim", "4", "--N", "96", "--max-snaps", "8", "--seed", "0"])
    try:
        with pytest.raises(RuntimeError, match="balayage vide"):
            p10a.main()
    finally:
        HP2.PhysicalMapperV2.__init__.__defaults__ = anciens


def test_le_module_expose_bien_la_tolerance_et_pas_un_nombre_magique():
    assert isinstance(p10a.F1_SPAN_TOL, float)
    assert p10a.F1_SPAN_TOL > 0.0
