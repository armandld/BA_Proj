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


#: `secondary_analysis` reproduisait le meme defaut que D-92
#: (`figures/pareto_frontier.py`, retracte dans RESULTS.md) avant d'etre
#: corrige a son tour : point Q-HAS = tirage unique non verifie plutot que
#: la moyenne des tirages T20 acheves, frontiere non purgee des points
#: avortes. Corrige : `secondary_analysis(records, results_dir)` reprend
#: `verified_qhas_point`/`load_trace_audit`/`drop_aborted` de
#: `figures/pareto_frontier.py` (une seule definition, importee, pas une
#: nouvelle copie).
_RETRACTED_RATIOS = {"kh": 4.41, "ot": 2.57, "rotor": 3.62, "tearing": 4.38}
_CORRECTED_RATIOS = {"kh": 2.10, "ot": 1.79, "rotor": 2.49, "tearing": 1.98}


@pytest.mark.parametrize("fold", ["kh", "ot", "rotor", "tearing"])
def test_real_data_no_longer_reproduces_the_retracted_ratio(fold):
    """Rejoue `secondary_analysis` sur les artefacts geles de `results/`
    (memes 4 folds que `test_pareto_frontier_retracted_ratio.py`) et
    verifie qu'elle rend les memes ratios CORRIGES, pas les retractes."""
    _repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    results_dir = os.path.join(_repo_root, "results")
    path = os.path.join(results_dir, f"t15b_budget_matched_{fold}.json")
    if not os.path.exists(path):
        pytest.skip("artefact gele absent : " + path)

    rec = load_fold(results_dir, fold)
    sec = secondary_analysis([rec], results_dir)
    row = sec["rows"][0]

    assert row["ratio_vs_frontier"] == pytest.approx(
        _CORRECTED_RATIOS[fold], abs=0.01)
    assert row["ratio_vs_frontier"] != pytest.approx(
        _RETRACTED_RATIOS[fold], abs=0.05)


def test_rotor_flips_from_dominated_to_not_once_qhas_point_is_verified():
    """Teste `secondary_analysis` isolement, PAS l'artefact publie : dans
    `main()`, `rotor` est deja exclu par l'audit T19 (pre-registration
    §5, raison sans rapport avec ce correctif) avant meme d'atteindre
    cette fonction, donc `results/t15c_fold_synthesis.json` ne le montre
    jamais. Ici, sans ce filtre, `rotor` change de verdict de domination
    (pas seulement de magnitude) sous le point T20 verifie -- preuve que
    la correction a une vraie prise sur au moins un pli (mesure, pas
    supposee -- voir RESULTS.md)."""
    _repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    results_dir = os.path.join(_repo_root, "results")
    recs = [r for r in (load_fold(results_dir, f)
                        for f in ("kh", "ot", "rotor", "tearing"))
            if r is not None and r["budget"] is not None]
    if len(recs) < 4:
        pytest.skip("artefacts geles t15b incomplets")
    sec = secondary_analysis(recs, results_dir)
    by = {r["fold"]: r["qhas_dominated"] for r in sec["rows"]}
    assert by["rotor"] is False
    assert by["kh"] is True and by["ot"] is True and by["tearing"] is True
    assert sec["n_qhas_dominated"] == 3


def test_interp_frontier_refuses_to_extrapolate():
    """D-92 : extrapoler hors de la trace balayee rendrait sans broncher
    la valeur du bord -- `interp_frontier` doit rendre None plutot que ce
    nombre d'apparence normale pour une comparaison qui n'existe pas."""
    trace = [{"patch_ratio": 0.2, "phys_score": 0.5},
             {"patch_ratio": 0.9, "phys_score": 0.01}]
    assert interp_frontier(trace, 0.1) is None
    assert interp_frontier(trace, 0.95) is None
    assert interp_frontier(trace, 0.5) is not None


def test_secondary_analysis_falls_back_to_the_single_draw_without_results_dir(
        tmp_path):
    """Sans `results_dir` (les tests synthetiques existants, par ex.), le
    comportement precedent est preserve a l'identique : repli explicite
    sur le tirage unique, pas un crash ni un silence."""
    _mk_fold(str(tmp_path), "ot", 0.33, 0.44, q_patch=0.68, q_phys=0.194,
             matched={"threshold": 0.19, "patch_ratio": 0.64,
                      "phys_score": 0.083, "combined": 0.24},
             trace=[{"patch_ratio": 0.64, "phys_score": 0.083},
                    {"patch_ratio": 0.95, "phys_score": 0.011}])
    rec = load_fold(str(tmp_path), "ot")
    sec_no_dir = secondary_analysis([rec])
    sec_with_dir = secondary_analysis([rec], str(tmp_path))
    assert sec_no_dir["rows"][0]["qhas_phys"] == pytest.approx(0.194)
    assert sec_with_dir["rows"][0]["qhas_phys"] == pytest.approx(0.194)
