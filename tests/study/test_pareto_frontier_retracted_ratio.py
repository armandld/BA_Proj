"""D-92 — `pareto_frontier.py`, execute seul, reproduisait les rapports
Q-HAS RETRACTES : 2,57x / 4,41x / 3,62x / 4,38x sur ot/kh/rotor/tearing.

`docs/RESULTS.md` ("Figure updated") documente deja que ces quatre nombres
ont ete retires : ils venaient d'un tirage UNIQUE d'un bras QAOA non
deterministe (D11), gonfles de 1,1 a 2,2x contre la moyenne sur 5 tirages,
et d'une frontiere qui pouvait inclure un point issu d'une trace avortee.
`pareto_panel.py` (la planche V4 multi-folds) a ete corrigee pour prendre le
point Q-HAS moyenne (`verified_qhas_point`, sur `t20_qhas_run_variance_*`)
et retirer les points avortes (`drop_aborted`, audit `t19`). Mais
`pareto_frontier.py::main()` — le script MONO-fold, dont `pareto_panel.py`
importe les fonctions de base et dont il reprend "exactement la meme
grammaire" par sa propre docstring — n'a jamais recu ces deux corrections :
lance seul (`python figures/pareto_frontier.py --fold X`), il rendait
encore le tirage unique, sans jamais consulter T19 ni T20.

Sur quelle entree ces tests echouent-ils ? Sur la version d'avant D-92 :
`test_main_uses_the_averaged_qhas_point_not_the_single_draw` et
`test_main_drops_frontier_points_marked_aborted_by_the_t19_audit` y
echouent — la premiere parce que `main()` n'appelait jamais
`verified_qhas_point`, la seconde parce qu'il n'appelait jamais
`drop_aborted`. `test_the_three_selection_functions_are_not_duplicated`
y echoue aussi : `pareto_panel.py` definissait alors sa propre copie de
ces trois fonctions, exactement la forme de defaut (D-60/D-61, deux copies
qui divergent) que ce depot a deja rencontree deux fois.
"""
import csv
import json
import os
import sys

import pytest


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_FIGURES = os.path.join(_REPO_ROOT, "figures")
if _FIGURES not in sys.path:
    sys.path.insert(0, _FIGURES)

import config  # noqa: E402  (study/pipeline/config.py, via sys.path above)
import pareto_frontier  # noqa: E402
import pareto_panel  # noqa: E402

_RESULTS_DIR = config.RESULTS_DIR

#: Mesure du 15 aout 2026 sur les artefacts geles de `results/` : t15b (un
#: seul tirage) contre t20 (5 tirages, 3 pour `rotor` dont 2 avortes).
#: Ecrit ici pour qu'une derive se voie.
_RETRACTED_RATIOS = {"ot": 2.57, "kh": 4.41, "rotor": 3.62, "tearing": 4.38}
_CORRECTED_RATIOS = {"ot": 1.79, "kh": 2.10, "rotor": 2.49, "tearing": 1.98}
_CORRECTED_RATIOS_VS_MATCHED = {
    "ot": 1.30, "kh": 1.90, "rotor": 2.74, "tearing": 1.81,
}


def _mk_budget_json(dirpath, fold, q_patch, single_phys, matched_thr,
                     matched_patch, matched_phys, trace):
    json.dump({
        "fold": fold, "target_patch": q_patch,
        "qhas": {"combined": 0.0, "phys_score": single_phys,
                 "patch_ratio": q_patch, "wall_s": 1.0},
        "tuned_classical": {"combined": 0.0, "phys_score": 999.0,
                            "patch_ratio": 0.01, "wall_s": 1.0},
        "matched_classical": {"threshold": matched_thr,
                              "patch_ratio": matched_patch,
                              "phys_score": matched_phys, "combined": 0.0,
                              "wall_s": 1.0},
        "trace": trace,
        "delta_phys_matched": single_phys - matched_phys,
    }, open(os.path.join(dirpath, f"t15b_budget_matched_{fold}.json"), "w"))


def _mk_variance_json(dirpath, fold, phys_values, patch_values,
                       n_aborted=0):
    runs = [{"completed": True, "phys_score": p, "patch_ratio": q}
            for p, q in zip(phys_values, patch_values)]
    runs += [{"completed": False} for _ in range(n_aborted)]
    json.dump({"fold": fold, "qhas_runs": runs},
              open(os.path.join(dirpath, f"t20_qhas_run_variance_{fold}.json"),
                   "w"))


def _mk_trace_audit_json(dirpath, fold, aborted_thresholds):
    json.dump({"traces": [{"fold": fold,
                           "points": [{"threshold": t, "completed": False}
                                      for t in aborted_thresholds]}]},
              open(os.path.join(dirpath, "t19_budget_trace_audit.json"), "w"))


def _read_csv_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_the_three_selection_functions_are_not_duplicated():
    """Une seule definition de chaque : `pareto_panel` importe la sienne
    d'ici plutot que d'en tenir sa propre copie — c'est la forme meme de
    defaut (D-60/D-61) qui a laisse `main()` diverger de la planche."""
    assert pareto_panel.verified_qhas_point is pareto_frontier.verified_qhas_point
    assert pareto_panel.load_trace_audit is pareto_frontier.load_trace_audit
    assert pareto_panel.drop_aborted is pareto_frontier.drop_aborted


def test_main_uses_the_averaged_qhas_point_not_the_single_draw(
        tmp_path, monkeypatch):
    fold = "synthfold"
    trace = [
        {"threshold": 0.05, "patch_ratio": 0.10, "phys_score": 0.50,
         "combined": 0.0, "wall_s": 1.0},
        {"threshold": 0.50, "patch_ratio": 0.90, "phys_score": 0.02,
         "combined": 0.0, "wall_s": 1.0},
    ]
    # tirage unique (t15b) tres au-dessus de la moyenne des 5 tirages (t20)
    _mk_budget_json(str(tmp_path), fold, q_patch=0.60, single_phys=0.30,
                    matched_thr=0.30, matched_patch=0.55, matched_phys=0.10,
                    trace=trace)
    _mk_variance_json(str(tmp_path), fold,
                      phys_values=[0.10, 0.11, 0.09, 0.10, 0.10],
                      patch_values=[0.60] * 5)

    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv",
                        ["pareto_frontier.py", "--fold", fold,
                         "--out-dir", str(tmp_path / "fig")])
    pareto_frontier.main()

    rows = _read_csv_rows(str(tmp_path / "fig" / f"pareto_frontier_{fold}.csv"))
    qhas_row = next(r for r in rows if r["series"] == "qhas")
    # single_phys=0.30 (t15b) ; moyenne des 5 tirages t20 = 0.10 -- si
    # `main()` prenait encore le tirage unique, ce test echoue.
    assert float(qhas_row["phys_score"]) == pytest.approx(0.10, abs=1e-9)
    assert float(qhas_row["phys_score"]) != pytest.approx(0.30, abs=1e-2)


def test_main_falls_back_to_the_single_draw_when_no_repeats_exist(
        tmp_path, monkeypatch, capsys):
    fold = "synthfold2"
    trace = [
        {"threshold": 0.05, "patch_ratio": 0.10, "phys_score": 0.50,
         "combined": 0.0, "wall_s": 1.0},
        {"threshold": 0.50, "patch_ratio": 0.90, "phys_score": 0.02,
         "combined": 0.0, "wall_s": 1.0},
    ]
    _mk_budget_json(str(tmp_path), fold, q_patch=0.60, single_phys=0.30,
                    matched_thr=0.30, matched_patch=0.55, matched_phys=0.10,
                    trace=trace)
    # pas de t20_qhas_run_variance_{fold}.json : aucun tirage repete

    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv",
                        ["pareto_frontier.py", "--fold", fold,
                         "--out-dir", str(tmp_path / "fig")])
    pareto_frontier.main()

    rows = _read_csv_rows(str(tmp_path / "fig" / f"pareto_frontier_{fold}.csv"))
    qhas_row = next(r for r in rows if r["series"] == "qhas")
    assert float(qhas_row["phys_score"]) == pytest.approx(0.30, abs=1e-9)
    assert "SINGLE t15b draw" in capsys.readouterr().out


def test_main_drops_frontier_points_marked_aborted_by_the_t19_audit(
        tmp_path, monkeypatch, capsys):
    fold = "synthfold3"
    trace = [
        {"threshold": 0.05, "patch_ratio": 0.10, "phys_score": 0.50,
         "combined": 0.0, "wall_s": 1.0},
        {"threshold": 0.30, "patch_ratio": 0.55, "phys_score": 0.10,
         "combined": 0.0, "wall_s": 1.0},
        {"threshold": 0.50, "patch_ratio": 0.90, "phys_score": 0.02,
         "combined": 0.0, "wall_s": 1.0},
    ]
    _mk_budget_json(str(tmp_path), fold, q_patch=0.60, single_phys=0.30,
                    matched_thr=0.30, matched_patch=0.55, matched_phys=0.10,
                    trace=trace)
    _mk_trace_audit_json(str(tmp_path), fold, aborted_thresholds=[0.05])

    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv",
                        ["pareto_frontier.py", "--fold", fold,
                         "--out-dir", str(tmp_path / "fig")])
    pareto_frontier.main()

    rows = _read_csv_rows(str(tmp_path / "fig" / f"pareto_frontier_{fold}.csv"))
    classical_thrs = {float(r["threshold"]) for r in rows
                      if r["series"] == "classical"}
    assert 0.05 not in classical_thrs
    assert classical_thrs == {0.30, 0.50}
    assert "dropped 1 frontier point" in capsys.readouterr().out


def test_main_writes_the_matched_classical_reference_row(tmp_path, monkeypatch):
    fold = "synthfold4"
    trace = [
        {"threshold": 0.05, "patch_ratio": 0.10, "phys_score": 0.50,
         "combined": 0.0, "wall_s": 1.0},
        {"threshold": 0.50, "patch_ratio": 0.90, "phys_score": 0.02,
         "combined": 0.0, "wall_s": 1.0},
    ]
    _mk_budget_json(str(tmp_path), fold, q_patch=0.60, single_phys=0.30,
                    matched_thr=0.30, matched_patch=0.55, matched_phys=0.10,
                    trace=trace)

    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv",
                        ["pareto_frontier.py", "--fold", fold,
                         "--out-dir", str(tmp_path / "fig")])
    pareto_frontier.main()

    rows = _read_csv_rows(str(tmp_path / "fig" / f"pareto_frontier_{fold}.csv"))
    mc_row = next(r for r in rows if r["series"] == "matched_classical")
    assert float(mc_row["phys_score"]) == pytest.approx(0.10, abs=1e-9)
    assert float(mc_row["threshold"]) == pytest.approx(0.30, abs=1e-9)


@pytest.mark.parametrize("fold", ["ot", "kh", "rotor", "tearing"])
def test_real_data_no_longer_reproduces_the_retracted_ratio(fold, tmp_path,
                                                             monkeypatch):
    """Epingle le nombre publie : rejoue `main()` sur les artefacts geles
    et compare aux tables corrigees de `docs/RESULTS.md`."""
    path = os.path.join(_RESULTS_DIR, f"t15b_budget_matched_{fold}.json")
    if not os.path.exists(path):
        pytest.skip("artefact gele absent : " + path)

    monkeypatch.setattr(sys, "argv",
                        ["pareto_frontier.py", "--fold", fold,
                         "--out-dir", str(tmp_path)])
    pareto_frontier.main()

    rows = _read_csv_rows(str(tmp_path / f"pareto_frontier_{fold}.csv"))
    qhas_phys = float(next(r for r in rows if r["series"] == "qhas")
                      ["phys_score"])
    frontier_phys = pareto_frontier.interp_frontier(
        [{"patch": float(r["patch_ratio"]), "phys": float(r["phys_score"])}
         for r in rows if r["series"] == "classical"],
        float(next(r for r in rows if r["series"] == "qhas")["patch_ratio"]))
    matched_phys = float(next(r for r in rows
                              if r["series"] == "matched_classical")
                         ["phys_score"])

    ratio = qhas_phys / frontier_phys
    ratio_vs_matched = qhas_phys / matched_phys

    assert ratio == pytest.approx(_CORRECTED_RATIOS[fold], abs=0.01)
    assert ratio != pytest.approx(_RETRACTED_RATIOS[fold], abs=0.05)
    assert ratio_vs_matched == pytest.approx(
        _CORRECTED_RATIOS_VS_MATCHED[fold], abs=0.01)
