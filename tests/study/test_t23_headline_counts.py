"""T23 — le decompte de tete doit etre CALCULE, et sur la bonne reference.

Le nombre le plus cite de l'etude etait ecrit a la main et ne se reproduisait
pas depuis les artefacts. Ces tests pinnent les deux erreurs qui s'y etaient
glissees, et la convention qui les evite.
"""
import json
import os
import sys

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.join(_REPO_ROOT, "study")
RESULTS = os.path.join(_REPO_ROOT, "results")

from closed_loop_headline_counts import fold_counts, matched_reference, totals

FOLDS = ("ot", "kh", "rotor", "tearing")


def _have(fold):
    return (os.path.exists(os.path.join(
                RESULTS, f"t15b_budget_matched_{fold}.json"))
            and os.path.exists(os.path.join(
                RESULTS, f"t20_qhas_run_variance_{fold}.json")))


@pytest.mark.parametrize("fold", FOLDS)
def test_aborted_draws_leave_the_denominator(fold):
    """Un tirage avorte ne compte NI au numerateur NI au denominateur.

    C'est l'erreur exacte du tableau publie : `rotor` avait 2 tirages
    avortes et etait malgre tout rapporte sur 5, d'ou un total sur 20."""
    if not _have(fold):
        pytest.skip("artefacts absents")
    r = fold_counts(RESULTS, fold)
    t = json.load(open(os.path.join(
        RESULTS, f"t20_qhas_run_variance_{fold}.json")))
    n_ok = sum(1 for x in t["qhas_runs"] if x["completed"])
    assert r["n_completed"] == n_ok
    assert r["n_completed"] + r["n_aborted"] == r["n_runs"]
    for k in ("less_faithful", "costlier", "dominated"):
        assert r[k] <= r["n_completed"], (
            f"{fold}: {k}={r[k]} depasse le nombre d'executions valides "
            f"({r['n_completed']}) — des tirages avortes sont comptes")


@pytest.mark.parametrize("fold", FOLDS)
def test_dominated_is_the_conjunction_never_larger(fold):
    """`domine` = les deux conditions a la fois, donc <= chacune.

    Sur `kh` le tableau publie donnait moins-fidele 4/5 et plus-couteux 5/5
    avec domine 4/5 : les deux colonnes etaient transposees. Cette borne
    n'aurait pas suffi a l'attraper, mais elle rend impossible toute
    transposition qui ferait depasser le minimum."""
    if not _have(fold):
        pytest.skip("artefacts absents")
    r = fold_counts(RESULTS, fold)
    assert r["dominated"] <= min(r["less_faithful"], r["costlier"])


@pytest.mark.parametrize("fold", FOLDS)
def test_reference_is_the_matched_point_not_the_t20_control(fold):
    """La reference doit venir de T15b, pas du controle classique de T20.

    Sur `ot` et `kh` ce controle a tourne au seuil REGLE et non au seuil
    apparie (D14). Sur `ot` il rend phys = 0.4845 la ou le point apparie
    vaut 0.0827 : prendre l'un pour l'autre inverse le sens du resultat sur
    ce fold. Le test verifie l'origine des deux coordonnees."""
    if not _have(fold):
        pytest.skip("artefacts absents")
    m = json.load(open(os.path.join(
        RESULTS, f"t15b_budget_matched_{fold}.json")))["matched_classical"]
    ref = matched_reference(RESULTS, fold)
    assert ref == (float(m["phys_score"]), float(m["patch_ratio"]))
    r = fold_counts(RESULTS, fold)
    assert r["ref_phys"] == float(m["phys_score"])
    assert r["ref_patch"] == float(m["patch_ratio"])


def test_totals_are_the_sum_of_the_folds():
    rows = [r for r in (fold_counts(RESULTS, f) for f in FOLDS) if r]
    if not rows:
        pytest.skip("artefacts absents")
    t = totals(rows)
    for k in ("n_completed", "less_faithful", "costlier", "dominated"):
        assert t[k] == sum(r[k] for r in rows)
    assert t["dominated"] <= min(t["less_faithful"], t["costlier"])


def test_published_counts_match_the_artifacts():
    """Regression sur la valeur publiee apres correction : 18/18, 16/18, 16/18.

    Si ce test casse, c'est soit qu'un artefact a change, soit que le
    decompte a re-derive — dans les deux cas RESULTS.md doit etre repris
    en meme temps, jamais l'un sans l'autre."""
    rows = [r for r in (fold_counts(RESULTS, f) for f in FOLDS) if r]
    if len(rows) < 4:
        pytest.skip("les quatre folds ne sont pas tous presents")
    t = totals(rows)
    assert (t["n_completed"], t["less_faithful"], t["costlier"],
            t["dominated"]) == (18, 18, 16, 16)
