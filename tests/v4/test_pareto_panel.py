"""Tests V4 : planche multi-folds de la frontiere erreur-cout.

On ne teste pas l'esthetique mais ce dont depend la correction de la
lecture : la selection des folds reellement disponibles, la coherence des
nombres annotes avec les JSON sources, et le fait que la planche se rende
sans exception pour 1 comme pour 4 panneaux (le cas a 4 panneaux ne doit
pas etre decouvert apres des heures de calcul).
"""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("v4", "v3", ""):
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "study", sub))

from make_pareto_figure import interp_frontier, load_points
from make_pareto_panel import FOLD_TITLES, available_folds, build_panel


def _mk_budget_json(dirpath, fold, q_patch=0.68, q_phys=0.194, scale=1.0):
    trace = [
        {"threshold": 0.05, "patch_ratio": 0.95, "phys_score": 0.011 * scale,
         "combined": 0.28, "wall_s": 1.0},
        {"threshold": 0.19, "patch_ratio": 0.64, "phys_score": 0.083 * scale,
         "combined": 0.24, "wall_s": 1.0},
        {"threshold": 0.46, "patch_ratio": 0.32, "phys_score": 0.485 * scale,
         "combined": 0.44, "wall_s": 1.0},
    ]
    d = {
        "fold": fold, "target_patch": q_patch,
        "qhas": {"combined": 0.33, "phys_score": q_phys,
                 "patch_ratio": q_patch, "wall_s": 1.0},
        "tuned_classical": {"combined": 0.44, "phys_score": 0.485 * scale,
                            "patch_ratio": 0.32, "wall_s": 1.0},
        "matched_classical": {"threshold": 0.19, "patch_ratio": 0.64,
                              "phys_score": 0.083 * scale, "combined": 0.24,
                              "wall_s": 1.0},
        "trace": trace, "delta_phys_matched": q_phys - 0.083 * scale,
    }
    json.dump(d, open(os.path.join(
        dirpath, f"t15b_budget_matched_{fold}.json"), "w"))


def test_available_folds_filters_to_existing_runs(tmp_path):
    _mk_budget_json(str(tmp_path), "ot")
    _mk_budget_json(str(tmp_path), "rotor")
    got = available_folds(str(tmp_path), ["ot", "kh", "rotor", "tearing"])
    assert got == ["ot", "rotor"]          # ordre demande preserve
    assert available_folds(str(tmp_path), ["kh"]) == []


def test_every_fold_code_has_a_readable_title():
    for f in ("ot", "kh", "rotor", "tearing"):
        assert f in FOLD_TITLES and FOLD_TITLES[f] != f


def test_single_panel_renders_files(tmp_path):
    _mk_budget_json(str(tmp_path), "ot")
    front, q, tuned, _ = load_points(str(tmp_path), "ot")
    out = tmp_path / "fig"
    base, rows = build_panel([{"fold": "ot", "front": front, "q": q}],
                             str(out), ncols=2)
    assert os.path.exists(base + ".pdf") and os.path.exists(base + ".png")
    assert len(rows) == 1 and rows[0]["fold"] == "ot"


def test_four_panels_render_with_disparate_error_scales(tmp_path):
    """Le cas reel de la campagne : 4 folds dont les erreurs couvrent deux
    ordres de grandeur. Les axes etant independants, aucun panneau ne doit
    faire echouer le rendu."""
    scales = {"ot": 1.0, "kh": 0.05, "rotor": 1.6, "tearing": 0.4}
    recs = []
    for f, s in scales.items():
        _mk_budget_json(str(tmp_path), f, scale=s, q_phys=0.19 * s)
        front, q, _, _ = load_points(str(tmp_path), f)
        recs.append({"fold": f, "front": front, "q": q})
    base, rows = build_panel(recs, str(tmp_path / "fig"), ncols=2)
    assert len(rows) == 4
    assert os.path.exists(base + ".png")


def test_annotated_ratio_matches_the_source_json(tmp_path):
    """Le rapport affiche doit etre exactement phys(Q-HAS) / frontiere
    interpolee au meme budget — pas une quantite recalculee autrement."""
    _mk_budget_json(str(tmp_path), "ot", q_patch=0.68, q_phys=0.194)
    front, q, _, _ = load_points(str(tmp_path), "ot")
    base, rows = build_panel([{"fold": "ot", "front": front, "q": q}],
                             str(tmp_path / "fig"))
    r = rows[0]
    assert r["q_ref"] == pytest.approx(interp_frontier(front, 0.68))
    assert r["ratio"] == pytest.approx(0.194 / r["q_ref"])
    assert r["ratio"] > 1.0        # Q-HAS au-dessus de la frontiere


def test_missing_fold_raises_rather_than_plotting_nothing(tmp_path):
    with pytest.raises(SystemExit):
        load_points(str(tmp_path), "kh")
