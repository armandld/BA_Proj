"""D-196 (audit H1/H3/H4, 26 aout) : `rows_t15c` rendait un affichage
« OK, sans reference » pour 3 metriques d'ISSUE (pas de completude) tant
que la campagne LOSO n'avait pas fourni tous les folds -- exactement le
defaut que ce module corrige deja pour les deux compteurs de completude de
la meme fonction (voir le commentaire « Ces lignes n'avaient AUCUNE
reference » dans `aggregate_master_table.py::rows_t15c`).
`test_master_table_is_pinned.py` l'a detecte en pratique : 3 lignes t15c
sans reference dans la table committee (elles feraient passer
« 4/4 folds domines » quel qu'en soit le contenu), et par la meme
occasion `KNOWN_DIFF` y etait reste a 4 alors que la table portait deja
6 DIFF depuis la fermeture de D-158 (voir DEFAUTS.md, note du 25 aout).

Ce banc verifie `rows_t15c` isolement, avec un jeu de folds synthetique
minimal (pas de DNS, pas de campagne reelle) qui SEPARE le cas complet du
cas partiel -- sans separation le bug reste invisible : un jeu de folds
constamment complet ou constamment vide ne le distingue pas.
"""
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aggregate_master_table import rows_t15c


def _write_fold(results_dir, name, qhas_combined, classical_combined,
                patch_ratio=0.5, phys_score=0.1):
    p = os.path.join(str(results_dir), f"t15_level3_fold_{name}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({
            "scenario": name, "train_on": [], "n_trials": 1,
            "qhas": {"combined": qhas_combined, "patch_ratio": patch_ratio,
                     "phys_score": phys_score},
            "classical": {"combined": classical_combined},
            "hyperparams": {}, "classical_params": {},
        }, fh)


def _write_budget(results_dir, name, patch_ratio=0.5, phys_score=0.1,
                  delta=0.01):
    """`t15b_budget_matched_{name}.json` minimal, suffisant pour
    `secondary_analysis`/`interp_frontier`."""
    p = os.path.join(str(results_dir), f"t15b_budget_matched_{name}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({
            "target_patch": patch_ratio,
            "matched_classical": {"threshold": 0.1,
                                  "patch_ratio": patch_ratio,
                                  "phys_score": phys_score},
            "delta_phys_matched": delta,
            "trace": [{"patch_ratio": 0.0, "phys_score": 1.0},
                     {"patch_ratio": 1.0, "phys_score": 0.0}],
        }, fh)


def test_outcome_rows_are_missing_while_the_fold_set_is_partial(tmp_path):
    """2 folds sur 4 : la campagne LOSO n'est pas terminee."""
    _write_fold(tmp_path, "a", 1.0, 2.0)
    _write_fold(tmp_path, "b", 1.0, 2.0)
    rows = {r["metric"]: r for r in
            rows_t15c(str(tmp_path), ["a", "b", "c", "d"])}

    for m in ("folds where Q-HAS better (combined)",
             "folds where Q-HAS Pareto-dominated at equal budget"):
        assert rows[m]["status"] == "MISSING", (
            f"{m} ne doit pas afficher de valeur sur un sous-ensemble "
            "partiel de folds")
        assert rows[m]["value"] is None

    # Les deux compteurs de completude, eux, ONT une cible connue (4) et
    # doivent rester visibles -- DIFF, pas MISSING : "2 sur 4" est une
    # information, pas une absence de mesure.
    assert rows["folds completed"]["value"] == 2.0
    assert rows["folds completed"]["ref"] == 4.0
    assert rows["folds completed"]["status"] == "DIFF"


def test_outcome_rows_are_populated_once_every_fold_is_present(tmp_path):
    """4 folds sur 4 : comportement inchange, non-regression."""
    for i, name in enumerate(("a", "b", "c", "d")):
        _write_fold(tmp_path, name, 1.0, 2.0 + 0.1 * i)
    rows = {r["metric"]: r for r in
            rows_t15c(str(tmp_path), ["a", "b", "c", "d"])}

    m = "folds where Q-HAS better (combined)"
    assert rows[m]["status"] == "OK"          # Q-HAS gagne toujours ici
    assert rows[m]["value"] == 4.0
    assert rows["folds completed"]["status"] == "OK"


def test_mean_delta_phys_is_missing_not_silently_dropped_when_partial(tmp_path):
    """Cas reel du 26 aout : primaire partiel (4/8 en production, 2/4 ici)
    mais du budget-matched (t15b) existe deja pour les folds presents --
    la ligne doit rester visible en MISSING, pas disparaitre du tableau
    (sinon le nombre de lignes varie avec l'etat de la campagne, moins
    visible qu'un MISSING nomme)."""
    _write_fold(tmp_path, "a", 1.0, 2.0)
    _write_fold(tmp_path, "b", 1.0, 2.0)
    _write_budget(tmp_path, "a")
    _write_budget(tmp_path, "b")
    rows = {r["metric"]: r for r in
            rows_t15c(str(tmp_path), ["a", "b", "c", "d"])}
    m = "mean delta phys at equal budget (>0 = Q-HAS worse)"
    assert m in rows, "la ligne ne doit pas disparaitre silencieusement"
    assert rows[m]["status"] == "MISSING"
    assert rows["budget-matched folds"]["value"] == 2.0


def test_no_fold_at_all_still_returns_the_three_original_missing_rows(tmp_path):
    """Non-regression du cas deja gere avant D-196 (aucun fold present)."""
    rows = {r["metric"]: r for r in
            rows_t15c(str(tmp_path), ["a", "b", "c", "d"])}
    assert len(rows) == 3
    for r in rows.values():
        assert r["status"] == "MISSING"
