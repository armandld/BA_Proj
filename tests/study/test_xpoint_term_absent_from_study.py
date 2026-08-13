"""D-51 — `study/` mesure un Hamiltonien sans le terme ZZZZ de point X.

La campagne d'entraînement met `AdvAnomaliesEnable = True` sur 6/6 scénarios
(D-33, `RESULTS.md`) ; les deux seuls sites de `study/` qui mentionnent le
drapeau le codent en dur à `False`, et `build_ising_terms` ne sait de toute
façon pas représenter `K_xpoint`.

**Épinglage de déviation, pas verrouillage d'une correction** : rien n'est
corrigé, la mesure dit que le terme vaut exactement zéro à `dim = 2`, donc
aucun nombre publié n'en dépend. Ces tests font échouer la suite le jour où
quelqu'un change de côté sur cet axe — ce qui est précisément le moment où il
faut remesurer, pas glisser.

Sur quelle entrée chacun échoue est écrit dans sa docstring.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

DIM = 2


def _params(with_xpoint, k_xpoint=7.0):
    """Coefficients synthétiques ; `K_xpoint` DOMINE tous les autres blocs.

    Choisi pour SÉPARER : si un chemin lisait `K_xpoint`, un terme à 7,0
    contre des blocs à 0,1 ne pourrait pas passer inaperçu.
    """
    z = lambda v: np.full((DIM, DIM), v, dtype=float)
    p = {"H_edges": (z(0.1), z(0.1)),
         "C_edges": (z(0.1), z(0.1)),
         "K_plaquettes": z(0.1)}
    if with_xpoint:
        p["K_xpoint"] = z(k_xpoint)
    return p


def test_build_ising_terms_ignores_xpoint():
    """`build_ising_terms` rend exactement la même chose avec et sans.

    Échoue si quelqu'un implémente `K_xpoint` côté Ising — ce qui est une
    des trois directions de D-51, et demande alors de rejouer phase 4, T13
    et T26 avant de mettre ce test à jour.
    """
    from ising_terms_and_annealing import build_ising_terms

    h_a, (ei_a, ec_a), (pi_a, pc_a) = build_ising_terms(_params(False), DIM)
    h_b, (ei_b, ec_b), (pi_b, pc_b) = build_ising_terms(_params(True), DIM)

    np.testing.assert_array_equal(h_a, h_b)
    np.testing.assert_array_equal(ec_a, ec_b)
    np.testing.assert_array_equal(pc_a, pc_b)
    np.testing.assert_array_equal(pi_a, pi_b)


def test_cost_hamiltonian_does_honour_xpoint_when_enabled():
    """Le vrai Hamiltonien QAOA, lui, le lit — mais seulement drapeau levé.

    C'est la moitié qui rend l'écart réel : les deux chemins que le dépôt
    présente comme encodant « the EXACT SAME Hamiltonian » divergent dès que
    `K_xpoint` est non nul et le drapeau vrai. Échoue si le gate disparaît
    d'un côté ou de l'autre.
    """
    from VQA.cost_hamiltonian import create_period_hamiltonian

    off = create_period_hamiltonian(_params(True), DIM, False)
    on = create_period_hamiltonian(_params(True), DIM, True)

    assert len(on) > len(off), (
        "drapeau levé, le terme de point X n'entre plus dans le Hamiltonien "
        "de coût : l'écart que D-51 documente aurait disparu — remesurer")
    assert float(np.sum(np.abs(on.coeffs))) > float(np.sum(np.abs(off.coeffs)))

    # et sans K_xpoint du tout, le drapeau ne change rien
    a = create_period_hamiltonian(_params(False), DIM, False)
    b = create_period_hamiltonian(_params(False), DIM, True)
    assert len(a) == len(b)


def test_study_encoder_never_produces_xpoint():
    """`prepare_qaoa_inputs` ne demande jamais les anomalies avancées.

    On interroge le comportement, pas le source : le mappeur ne produit
    `K_xpoint` que si le drapeau est vrai, donc son absence de la sortie
    prouve que `study/` est du côté `False` de l'axe. Échoue le jour où
    `study/` bascule — direction 2 ou 3 de D-51.
    """
    from qaoa_inputs import prepare_qaoa_inputs

    N = 16
    x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    vx, vy = np.sin(Y), np.sin(X)
    Bx, By = np.cos(Y), np.cos(X)

    for use_v2 in (False, True):
        _, hp, _ = prepare_qaoa_inputs(vx, vy, Bx, By, N, DIM, 400,
                                       use_v2=use_v2)
        assert "K_xpoint" not in hp, (
            f"use_v2={use_v2} : `study/` produit maintenant K_xpoint. C'est "
            "peut-être voulu (D-51 direction 2/3) — mais alors phase 4, T13 "
            "et T26 doivent être rejouées et leurs nombres remesurés.")


def test_ablation_zeroes_a_key_nothing_reads():
    """`h3_term_ablation` annule `K_xpoint` sur `no_ZZZZ`… pour rien.

    Le geste est correct en intention et sans effet en pratique, puisque
    `ground_state_mask` passe par `build_ising_terms`. On épingle que
    l'ablation `no_ZZZZ` donne le MÊME masque que `K_xpoint` soit présent ou
    non : tant que c'est vrai, T13 ne teste qu'une des deux familles ZZZZ.
    """
    from h3_term_ablation import zero_hamiltonian_terms, ground_state_mask

    a = ground_state_mask(zero_hamiltonian_terms(_params(False), ("ZZZZ",)),
                          DIM)
    b = ground_state_mask(zero_hamiltonian_terms(_params(True), ("ZZZZ",)),
                          DIM)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == pytest.approx(b[1])

    # et même SANS ablation, la présence de K_xpoint ne change rien
    c = ground_state_mask(_params(False), DIM)
    d = ground_state_mask(_params(True), DIM)
    np.testing.assert_array_equal(c[0], d[0])
    assert c[1] == pytest.approx(d[1]), (
        "un K_xpoint 70x plus grand que les autres blocs ne déplace pas "
        "l'énergie du fondamental : c'est exactement la cécité que D-51 "
        "décrit")
