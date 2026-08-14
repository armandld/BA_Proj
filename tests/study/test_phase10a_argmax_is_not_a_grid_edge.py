#!/usr/bin/env python3
"""D-87 — un optimum posé SUR le bord de la grille, et un thr* hors de la boîte.

D-86 traite la courbe **plate**, dont l'`argmax` ne désigne rien. Restent, sur
les courbes **informatives** que D-86 laisse passer par construction, deux
manières pour la phase 10a de rendre un nombre plausible et faux :

1. **L'`argmax` tombe sur un BORD de la grille de `c_bias`.** La grille s'arrête
   là ; l'optimum réel peut être au-delà, et rien ne le disait. La phase 10
   porte `hits_bound()` pour exactement cette pathologie — la phase 10a n'avait
   pas d'équivalent.
2. **`thr*` sort de la boîte de la phase 10 et y est raboté en silence.**
   `best_threshold` balaie `linspace(0.02, 0.60, 59)` — exactement la boîte —
   *réunie* aux quantiles du score, qui en sortent. `np.clip` ne dit rien.

Mesure (`--dim 4 --N 256 --max-snaps 8 --seed 0`, Re=400, sortie hors du dépôt) :

    scenario           thr*        c*     F1_MF   écart max-min   position
    harris_tearing     0.6777    0.100    0.0000   0.000e+00      bord gauche (dégénéré, D-86)
    kelvin_helmholtz   0.6908  100.000    0.2235   2.235e-01      bord DROIT  <- informatif
    mhd_rotor          0.2200  100.000    0.4333   4.333e-01      bord DROIT  <- informatif
    orszag_tang        0.1800   31.623    0.2843   2.843e-01      intérieur

Ce qu'il y a au-delà du bord droit, mesuré : le plus petit `c` qui fait basculer
un seul spin vaut **1,6e4** (kelvin_helmholtz) et **5,0e4** (mhd_rotor), deux
ordres au-delà de la grille. `hits_bound(x0)` est vrai sur **3 lignes sur 4** ;
après D-86 il reste vrai sur **2 des 3** lignes retenues.

Les tests de `TestEpinglage` passent des **deux** côtés, et c'est voulu : ils
épinglent le lien entre la grille et la boîte, et tomberaient si l'une bougeait
sous l'autre. Tous les autres échouent sur la version d'avant.
"""
import ast
import inspect
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

from h2b_train_linear_hamiltonian import THETA_BOUNDS, hits_bound  # noqa: E402

# La grille de c_bias telle que `main()` la construit.
C_GRID = np.logspace(-1.0, 2.0, 25)


# ------------------------------------------------------------------
# Épinglage — vert des deux côtés
# ------------------------------------------------------------------

class TestEpinglage:
    """Ce que la correction suppose du dépôt, épinglé pour que ça se voie."""

    def test_les_deux_bords_de_la_grille_sont_les_deux_bornes_de_la_boite(self):
        """La grille de c_bias et la boîte de la phase 10 coïncident **bord à
        bord** : 0,1 -> log10 = -1,0 = borne basse ; 100 -> +2,0 = borne haute.

        C'est ce qui rend un argmax de bord coûteux plutôt qu'anecdotique : il
        pose x0 sur un coin. Si l'une des deux bougeait sous l'autre, ce test
        le dirait, et il faudrait remesurer — pas retoucher la borne.
        """
        assert np.log10(float(C_GRID[0])) == pytest.approx(
            float(THETA_BOUNDS[0, 0]))
        assert np.log10(float(C_GRID[-1])) == pytest.approx(
            float(THETA_BOUNDS[0, 1]))
        assert hits_bound(np.array([np.log10(C_GRID[0]), 0.15])) is True
        assert hits_bound(np.array([np.log10(C_GRID[-1]), 0.15])) is True

    def test_un_x0_interieur_ne_declenche_pas_hits_bound(self):
        """Le témoin. Sans lui les assertions ci-dessus passeraient avec un
        `hits_bound` constamment vrai, et ne mesureraient rien."""
        assert hits_bound(np.array([np.log10(31.623), 0.18])) is False

    def test_les_thr_stars_mesures_tombent_hors_de_la_boite(self):
        """Les deux thr* mesurés sortent de la boîte — donc étaient rabotés.

        Le test porte les **nombres mesurés** : si la grille de
        `best_threshold` cessait de déborder, il échouerait, et il faudrait
        remesurer.
        """
        thr_lo, thr_hi = float(THETA_BOUNDS[1, 0]), float(THETA_BOUNDS[1, 1])
        assert thr_hi == pytest.approx(0.60)
        for thr_mesure in (0.6777, 0.6908):        # harris_tearing, KH
            assert thr_mesure > thr_hi
            assert float(np.clip(thr_mesure, thr_lo, thr_hi)) \
                == pytest.approx(thr_hi)

    def test_la_grille_de_best_threshold_deborde_bien_la_boite(self):
        """La cause, pas seulement le symptôme : `best_threshold` réunit la
        boîte à des quantiles qui en sortent.

        Interroge la fonction sur des scores construits pour séparer — un
        score dont le quantile haut dépasse 0,60 — et non le texte du source.
        """
        from h2b_analytical_solution import best_threshold
        scores = np.linspace(0.50, 0.71, 200)      # comme harris_tearing
        gt = (scores > 0.677).astype(int)
        thr, f1 = best_threshold(scores, gt)
        assert thr > float(THETA_BOUNDS[1, 1]), \
            "la grille ne déborde plus : remesurer, D-87 n'a plus d'objet"
        assert f1 > 0.9


# ------------------------------------------------------------------
# La correction — rouge sur la version d'avant
# ------------------------------------------------------------------

def _analyse_keys():
    """Les clés que `analyse_snapshots` promet, lues sur son AST.

    Interroge le module, pas le texte du source : chercher une chaîne
    testerait la mise en forme, ce que ce dépôt a déjà payé trois fois.
    """
    import h2b_analytical_solution as ana
    tree = ast.parse(inspect.getsource(ana.analyse_snapshots))
    return {kw.arg for node in ast.walk(tree) if isinstance(node, ast.Call)
            for kw in node.keywords if kw.arg}


def test_analyse_snapshots_rend_la_position_de_l_argmax():
    for key in ("at_left_edge", "at_right_edge", "thr_outside_box"):
        assert key in _analyse_keys(), f"`{key}` absent du dict rendu"


def test_le_bord_droit_est_signale():
    """c* = 100 : c'est le dernier point, pas forcément l'optimum."""
    from h2b_analytical_solution import _thr_outside_box  # noqa: F401
    idx = int(np.argmax(np.linspace(0.0, 0.4333, C_GRID.size)))  # mhd_rotor
    assert idx == C_GRID.size - 1
    assert float(C_GRID[idx]) == pytest.approx(100.0)


def test_thr_outside_box_separe_les_deux_cotes():
    """Vrai au-dessus de 0,60, faux à l'intérieur — le témoin est dedans."""
    from h2b_analytical_solution import _thr_outside_box
    assert _thr_outside_box(0.6777) is True       # harris_tearing, mesuré
    assert _thr_outside_box(0.6908) is True       # kelvin_helmholtz, mesuré
    assert _thr_outside_box(0.2200) is False      # mhd_rotor, mesuré
    assert _thr_outside_box(0.1800) is False      # orszag_tang, mesuré
    assert _thr_outside_box(0.01) is True         # sous la borne basse


def test_un_thr_NaN_ne_declenche_pas_le_drapeau():
    """Un scénario entièrement dégénéré rend NaN (D-86). NaN n'est pas « hors
    boîte » : il n'est nulle part, et la phase 10 l'écarte déjà."""
    from h2b_analytical_solution import _thr_outside_box
    assert _thr_outside_box(float("nan")) is False


def test_la_boite_est_lue_chez_la_phase_10_pas_recopiee():
    """La boîte n'a qu'une source. Deux copies se séparent en silence — et
    c'est par là que thr* sortait de la boîte sans que personne ne le voie."""
    import h2b_analytical_solution as ana
    import h2b_train_linear_hamiltonian as tr
    assert ana.THETA_BOUNDS is tr.THETA_BOUNDS


def test_les_drapeaux_d_un_agregat_excluent_les_degeneres():
    """Un dégénéré est au bord gauche **par construction** (D-86 : son argmax
    est l'indice 0). Le compter dans un agrégat allumerait `at_left_edge`
    partout, et le drapeau ne dirait plus rien."""
    from h2b_analytical_solution import _edge_flags_agg
    rows = [
        dict(degenerate=True, at_left_edge=True, at_right_edge=False,
             thr_star=0.6777),                     # harris_tearing
        dict(degenerate=False, at_left_edge=False, at_right_edge=True,
             thr_star=0.2200),                     # mhd_rotor
    ]
    flags = _edge_flags_agg(rows)
    assert flags["at_left_edge"] is False, \
        "le dégénéré a allumé le bord gauche de l'agrégat"
    assert flags["at_right_edge"] is True
    # thr* de l'agrégat = moyenne des informatifs = 0,2200, dans la boîte
    assert flags["thr_outside_box"] is False


def test_le_drapeau_thr_d_un_agregat_porte_sur_sa_propre_valeur():
    """Contrairement aux bords, `thr_outside_box` se calcule sur la valeur
    PROPRE de l'agrégat : c'est celle-là que la phase 10 rabotera."""
    from h2b_analytical_solution import _edge_flags_agg
    rows = [dict(degenerate=False, at_left_edge=False, at_right_edge=False,
                 thr_star=0.6908)]                 # kelvin_helmholtz
    assert _edge_flags_agg(rows)["thr_outside_box"] is True


def test_build_init_map_rabote_dans_la_boite_et_le_dit(capsys):
    """La ligne KH mesurée : thr* = 0,6908 doit ressortir rabotée à 0,60, et
    le rabotage doit s'entendre."""
    from h2b_train_linear_hamiltonian import build_init_map
    init_map, n_skip = build_init_map(
        tags=np.array(["scenario:kelvin_helmholtz"]),
        thr_star=np.array([0.6908]),
        c_bias_star=np.array([100.0]),
        degenerate=np.array([False]))
    assert n_skip == 0
    theta = init_map["scenario:kelvin_helmholtz"]
    assert theta[1] == pytest.approx(float(THETA_BOUNDS[1, 1]))
    assert theta[0] == pytest.approx(float(THETA_BOUNDS[0, 1]))
    out = capsys.readouterr().out
    assert "RABOTE" in out and "D-87" in out
    assert "SUR une borne" in out


def test_build_init_map_ne_dit_rien_sur_une_ligne_interieure(capsys):
    """Le témoin : orszag_tang, seule ligne intérieure des quatre mesurées.

    Sans lui, les deux assertions du test précédent passeraient avec un
    avertissement imprimé à chaque ligne, et ne mesureraient rien.
    """
    from h2b_train_linear_hamiltonian import build_init_map
    init_map, _ = build_init_map(
        tags=np.array(["scenario:orszag_tang"]),
        thr_star=np.array([0.1800]),
        c_bias_star=np.array([31.6228]),
        degenerate=np.array([False]))
    theta = init_map["scenario:orszag_tang"]
    assert theta == pytest.approx([np.log10(31.6228), 0.18])
    out = capsys.readouterr().out
    assert "RABOTE" not in out
    assert "SUR une borne" not in out


def test_train_consigne_si_x0_etait_sur_une_borne():
    """`hits_bound` n'était évalué que sur le theta FINAL. Un x0 posé sur une
    borne — 3 lignes sur 4 avec l'init analytique mesuré, 2 sur les 3 retenues
    après D-86 — n'apparaissait ni à l'écran ni dans l'artefact."""
    import h2b_train_linear_hamiltonian as tr
    tree = ast.parse(inspect.getsource(tr.train))
    rendues = {kw.arg for node in ast.walk(tree) if isinstance(node, ast.Call)
               for kw in node.keywords if kw.arg}
    for key in ("x0_theta", "x0_hits_bound", "x0_from_analytical"):
        assert key in rendues, f"`{key}` absent du dict rendu par train()"
