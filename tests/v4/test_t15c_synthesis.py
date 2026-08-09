"""Tests V4 T15c : synthese inter-folds du niveau 3.

Aucune simulation. On verifie que l'agregation lit fidelement les JSON de
fold, que la convention de signe est la bonne (`combined` est un COUT,
donc delta < 0 signifie Q-HAS meilleur), que la marge d'equivalence suit
la formule pre-enregistree, et que la detection de domination de Pareto
n'est pas laxiste.
"""
import json
import os
import sys

import numpy as np
import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HERE = os.path.dirname(os.path.abspath(__file__))

from closed_loop_fold_synthesis import (
    TOST_MARGIN_FRAC, WIN_RULE_MIN, format_table, interp_frontier, load_fold,
    primary_analysis, secondary_analysis,
)


def _mk_fold(dirpath, fold, q_comb, c_comb, q_patch=0.6, q_phys=0.2,
             matched=None, trace=None):
    """Ecrit une paire de JSON de fold minimale mais structurellement
    identique a celle produite par t15 / t15b."""
    d15 = {
        "fold": fold, "scenario": f"scen_{fold}", "train_on": ["a", "b"],
        "n_trials": 4,
        "qhas": {"combined": q_comb, "phys_score": q_phys,
                 "patch_ratio": q_patch, "wall_s": 1.0},
        "classical": {"combined": c_comb, "phys_score": 0.4,
                      "patch_ratio": 0.3, "wall_s": 1.0},
        "hyperparams": {"threshold_amr": 0.1496},
        "classical_params": {"threshold_amr": 0.46},
        "t_tune": 10.0, "git_hash": "deadbeef",
    }
    json.dump(d15, open(os.path.join(
        dirpath, f"t15_level3_fold_{fold}.json"), "w"))
    if matched is not None:
        d15b = {
            "fold": fold, "target_patch": q_patch,
            "qhas": d15["qhas"],
            "tuned_classical": d15["classical"],
            "matched_classical": matched,
            "trace": trace or [matched],
            "delta_phys_matched": q_phys - matched["phys_score"],
        }
        json.dump(d15b, open(os.path.join(
            dirpath, f"t15b_budget_matched_{fold}.json"), "w"))


def test_load_fold_missing_returns_none(tmp_path):
    assert load_fold(str(tmp_path), "nope") is None


def test_load_fold_without_t15b_has_no_budget(tmp_path):
    _mk_fold(str(tmp_path), "ot", 0.33, 0.44)
    rec = load_fold(str(tmp_path), "ot")
    assert rec is not None and rec["budget"] is None
    assert rec["qhas"]["combined"] == pytest.approx(0.33)


def test_sign_convention_lower_combined_is_better(tmp_path):
    # combined est un cout : Q-HAS 0.33 < classique 0.44 => Q-HAS gagne
    _mk_fold(str(tmp_path), "ot", 0.33, 0.44)
    _mk_fold(str(tmp_path), "kh", 0.50, 0.40)
    recs = [load_fold(str(tmp_path), f) for f in ("ot", "kh")]
    pr = primary_analysis(recs)
    assert pr["n_qhas_better"] == 1
    assert pr["n_classical_better"] == 1
    assert pr["delta"][0] < 0 and pr["delta"][1] > 0


def test_win_rule_requires_three_of_four(tmp_path):
    for f, (q, c) in zip(("ot", "kh", "rotor", "tearing"),
                         [(0.1, 0.2), (0.1, 0.2), (0.1, 0.2), (0.3, 0.2)]):
        _mk_fold(str(tmp_path), f, q, c)
    recs = [load_fold(str(tmp_path), f)
            for f in ("ot", "kh", "rotor", "tearing")]
    pr = primary_analysis(recs)
    assert pr["n_qhas_better"] == 3 >= WIN_RULE_MIN
    assert pr["qhas_wins_rule"] is True
    assert pr["classical_wins_rule"] is False


def test_tost_margin_follows_preregistered_formula(tmp_path):
    _mk_fold(str(tmp_path), "ot", 0.33, 0.40)
    _mk_fold(str(tmp_path), "kh", 0.35, 0.60)
    recs = [load_fold(str(tmp_path), f) for f in ("ot", "kh")]
    pr = primary_analysis(recs)
    expected = TOST_MARGIN_FRAC * np.mean([0.40, 0.60])
    assert pr["margin"] == pytest.approx(expected)
    # la marge ne depend PAS des ecarts observes : meme classique, meme marge
    _mk_fold(str(tmp_path), "rotor", 9.99, 0.40)
    _mk_fold(str(tmp_path), "tearing", 9.99, 0.60)
    recs2 = [load_fold(str(tmp_path), f) for f in ("rotor", "tearing")]
    assert primary_analysis(recs2)["margin"] == pytest.approx(expected)


def test_single_fold_yields_no_paired_statistic(tmp_path):
    _mk_fold(str(tmp_path), "ot", 0.33, 0.44)
    pr = primary_analysis([load_fold(str(tmp_path), "ot")])
    assert pr["n_folds"] == 1
    assert pr["tost"] is None and pr["paired_t_p"] is None
    assert "note_underpowered" in pr


def test_interp_frontier_matches_trace_endpoints():
    trace = [{"patch_ratio": 0.2, "phys_score": 0.5},
             {"patch_ratio": 0.9, "phys_score": 0.01}]
    assert interp_frontier(trace, 0.2) == pytest.approx(0.5)
    assert interp_frontier(trace, 0.9) == pytest.approx(0.01)
    mid = interp_frontier(trace, 0.55)
    assert 0.01 < mid < 0.5
    # trace desordonnee : le tri interne doit donner le meme resultat
    assert interp_frontier(trace[::-1], 0.55) == pytest.approx(mid)


def test_domination_requires_both_coordinates(tmp_path):
    # classique moins cher ET plus fidele => Q-HAS domine
    _mk_fold(str(tmp_path), "ot", 0.33, 0.44, q_patch=0.68, q_phys=0.194,
             matched={"threshold": 0.19, "patch_ratio": 0.64,
                      "phys_score": 0.083, "combined": 0.24},
             trace=[{"patch_ratio": 0.64, "phys_score": 0.083},
                    {"patch_ratio": 0.95, "phys_score": 0.011}])
    # classique moins cher mais MOINS fidele => pas de domination
    _mk_fold(str(tmp_path), "kh", 0.33, 0.44, q_patch=0.68, q_phys=0.194,
             matched={"threshold": 0.19, "patch_ratio": 0.64,
                      "phys_score": 0.500, "combined": 0.55},
             trace=[{"patch_ratio": 0.64, "phys_score": 0.500},
                    {"patch_ratio": 0.95, "phys_score": 0.30}])
    recs = [load_fold(str(tmp_path), f) for f in ("ot", "kh")]
    sec = secondary_analysis(recs)
    by = {r["fold"]: r for r in sec["rows"]}
    assert by["ot"]["qhas_dominated"] is True
    assert by["kh"]["qhas_dominated"] is False
    assert sec["n_qhas_dominated"] == 1
    # delta = qhas_phys - matched_phys ; > 0 => Q-HAS pire a cout egal
    assert by["ot"]["delta_phys_matched"] > 0
    assert by["kh"]["delta_phys_matched"] < 0


def test_secondary_skips_folds_without_budget_run(tmp_path):
    _mk_fold(str(tmp_path), "ot", 0.33, 0.44)          # pas de t15b
    _mk_fold(str(tmp_path), "kh", 0.33, 0.44, q_phys=0.2,
             matched={"threshold": 0.2, "patch_ratio": 0.6,
                      "phys_score": 0.1, "combined": 0.3},
             trace=[{"patch_ratio": 0.6, "phys_score": 0.1}])
    recs = [load_fold(str(tmp_path), f) for f in ("ot", "kh")]
    sec = secondary_analysis(recs)
    assert sec["n_folds"] == 1 and sec["rows"][0]["fold"] == "kh"
    # le critere primaire, lui, utilise bien les DEUX folds
    assert primary_analysis(recs)["n_folds"] == 2


def test_ratio_vs_frontier_reproduces_published_ot_value(tmp_path):
    """Le fold `ot` publie : Q-HAS phys 0.194 a patch 0.680, frontiere
    classique 0.0827 au meme budget => 2.35x (borne basse, la frontiere
    interpolee est legerement au-dessus du point apparie)."""
    _mk_fold(str(tmp_path), "ot", 0.3328, 0.4386, q_patch=0.67966,
             q_phys=0.19403,
             matched={"threshold": 0.190625, "patch_ratio": 0.64117,
                      "phys_score": 0.08270, "combined": 0.24226},
             trace=[{"patch_ratio": 0.64117, "phys_score": 0.08270},
                    {"patch_ratio": 0.94803, "phys_score": 0.01111}])
    sec = secondary_analysis([load_fold(str(tmp_path), "ot")])
    r = sec["rows"][0]
    assert r["ratio_vs_frontier"] > 2.0
    assert r["qhas_dominated"] is True


def test_format_table_is_renderable_with_one_fold(tmp_path):
    _mk_fold(str(tmp_path), "ot", 0.33, 0.44)
    recs = [load_fold(str(tmp_path), "ot")]
    txt = format_table(recs, primary_analysis(recs), secondary_analysis(recs))
    assert "Primary endpoint" in txt and "Secondary" in txt
    assert "no fold has a budget-matched run yet" in txt
