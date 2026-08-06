"""T24 — la comparaison sans fuite ne doit pas confondre regle et budget.

En mode `leak-free` les deux bras tournent a des seuils DIFFERENTS (le bras
QAOA au seuil classique regle du fold, le controle au seuil budget-apparie
force par `--matched-reference`). Sur `rotor` : 0.5864 contre 0.0969.

Mon propre code affichait « at the SAME operating point the classical arm
completed » quand le bras Q-HAS mourait. C'etait faux, et c'est le motif de
la campagne dans sa forme la plus pure : une ligne de sortie qui ne decrit
pas le calcul qu'elle accompagne.
"""
import ast
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.abspath(os.path.join(_HERE, "..", "..", "study", "v4"))
RESULTS = os.path.abspath(os.path.join(_HERE, "..", "..", "study", "results"))
sys.path.insert(0, os.path.join(V4, "..", "v3"))
sys.path.insert(0, V4)

from t24_leak_free_summary import analyse, frontier, frontier_at

FOLDS = ("ot", "kh", "rotor", "tearing")


def _have(fold):
    return os.path.exists(os.path.join(
        RESULTS, f"t22_unseen_leak-free_{fold}.json"))


def test_no_claim_of_a_shared_operating_point():
    """t22 ne doit plus affirmer un point de fonctionnement commun.

    En mode leak-free il n'y en a pas ; l'affirmer effacait la seule
    reserve qui empeche de lire l'avortement de Q-HAS comme une
    instabilite propre au bras a budget egal."""
    src = open(os.path.join(V4, "t22_unseen_conditions.py"),
               encoding="utf-8").read()
    assert "at the SAME operating point" not in src, (
        "t22 affirme un point de fonctionnement commun que le mode "
        "leak-free n'a pas")
    assert "DIFFERENT operating points" in src


@pytest.mark.parametrize("fold", FOLDS)
def test_thresholds_are_recorded_and_differ(fold):
    """Les deux seuils doivent etre lisibles dans le resume, pas devines."""
    if not _have(fold):
        pytest.skip("artefact leak-free absent")
    r = analyse(RESULTS, fold)
    assert r["qaoa_threshold"] is not None
    assert r["classical_threshold"] is not None
    # en leak-free ils different par construction ; si un jour ils
    # coincidaient, le drapeau doit le dire plutot que de rester implicite
    assert r["thresholds_match"] is not None


def test_frontier_refuses_to_extrapolate():
    """Hors de la plage balayee, aucun rapport — pas une valeur de bord.

    `np.interp` rend silencieusement l'extremite : un nombre d'apparence
    normale pour une comparaison qui n'existe pas. C'est precisement le
    motif traque."""
    front = ([0.2, 0.5, 0.9], [1.0, 0.5, 0.1])
    assert frontier_at(front, 0.05) is None      # sous la plage
    assert frontier_at(front, 0.95) is None      # au-dessus
    assert frontier_at(front, 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("fold", FOLDS)
def test_out_of_range_budgets_carry_no_ratio(fold):
    """Un budget hors plage ne doit jamais porter de rapport."""
    if not _have(fold):
        pytest.skip("artefact leak-free absent")
    r = analyse(RESULTS, fold)
    for cond, rec in r["conditions"].items():
        if rec.get("out_of_swept_range"):
            assert rec["ratio_vs_frontier"] is None, (
                f"{fold}/{cond}: budget hors de la frontiere balayee mais "
                f"un rapport est publie")


@pytest.mark.parametrize("fold", FOLDS)
def test_aborted_draws_excluded_from_the_leak_free_means(fold):
    if not _have(fold):
        pytest.skip("artefact leak-free absent")
    d = json.load(open(os.path.join(
        RESULTS, f"t22_unseen_leak-free_{fold}.json")))
    r = analyse(RESULTS, fold)
    for cond in ("canonical", "unseen"):
        runs = d["arms"]["qhas"].get(f"{cond}_runs", [])
        n_ok = sum(1 for x in runs if x["completed"])
        assert r["conditions"][cond]["n_completed"] == n_ok
        if n_ok == 0:
            assert "qhas_phys" not in r["conditions"][cond], (
                f"{fold}/{cond}: une moyenne est publiee alors qu'aucun "
                f"tirage n'a abouti")


def test_summary_script_declares_it_is_only_a_bound():
    """Le mode ne re-regle pas le bras QAOA : le module doit le dire.

    Sans cette reserve, « la fuite retiree, Q-HAS empire » se lirait comme
    le test definitif, alors que le reglage n'a pas ete refait."""
    src = open(os.path.join(V4, "t24_leak_free_summary.py"),
               encoding="utf-8").read()
    assert "BORNE" in src and "Optuna" in src
