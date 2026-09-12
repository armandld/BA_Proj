"""D-90 : `h4_transfer_summary.py` lisait la cle `total_abort_arm`
(singulier) alors que `h4_unseen_conditions.py:512` ecrit
`total_abort_arms` (PLURIEL, une liste — un fold peut voir les deux bras
avorter). La cle lue n'existe jamais dans l'artefact reel : `.get()` rend
systematiquement `None`, sans lever, et le message affichait
« the None arm aborted... » au lieu du nom du bras reellement fautif — un
repli silencieux qui montre une valeur plausible et fausse plutot que de
crier.

`load()` prend `results_dir` en parametre (pas `config.RESULTS_DIR` en dur),
donc testable avec `tmp_path` sans jamais toucher `results/` du depot —
l'erreur qui a produit cette mesure elle-meme (voir docs/RESULTS.md, D-89 et
D-90 : un essai reel avec `--folds tearing` a ecrase
`results/t22c_transfer_summary.json`, restaure depuis git avant de pousser
ce correctif).
"""
import json
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_H4_DIR = os.path.join(_REPO_ROOT, "study", "h4_transfer")
if _H4_DIR not in sys.path:
    sys.path.insert(0, _H4_DIR)

from h4_transfer_summary import load  # noqa: E402


def _write_total_abort_record(results_dir, fold, arms):
    """Forme exacte qu'ecrit `h4_unseen_conditions.py` pour un fold dont un
    ou plusieurs bras ont avorte sur la totalite d'une condition."""
    rec = {
        "fold": fold, "scenario": "harris_tearing", "mode": "unseen-ic",
        "status": "total_abort", "total_abort_arms": list(arms),
        "arms": {
            "qhas": {"status": "total_abort" if "qhas" in arms
                     else "completed",
                     "n_runs": 3,
                     **({} if "qhas" in arms else
                        {"canonical": {"phys_score": 0.02},
                         "unseen": {"phys_score": 0.03}})},
            "classical": {"status": "total_abort" if "classical" in arms
                          else "completed",
                          "n_runs": 2,
                          **({} if "classical" in arms else
                             {"canonical": {"phys_score": 0.01},
                              "unseen": {"phys_score": 0.02}})},
        },
    }
    path = os.path.join(results_dir, f"t22_unseen_unseen-ic_{fold}.json")
    json.dump(rec, open(path, "w"))
    return path


def test_total_abort_arms_key_matches_the_producer(tmp_path):
    """Mesure avant (D-90) : `total_abort_arm` (singulier) n'existe pas
    dans l'artefact reel -> `None`. Mesure apres : `total_abort_arms`
    (pluriel) rend la vraie liste."""
    results_dir = str(tmp_path)
    _write_total_abort_record(results_dir, "zz", ["qhas"])
    rec = load(results_dir, "zz")
    assert rec["total_abort"] is True
    assert rec["total_abort_arms"] == ["qhas"], (
        "la cle lue par load() ne correspond plus a celle produite par "
        "h4_unseen_conditions.py -- c'est exactement la regression D-90")
    # la mesure AVANT, ecrite explicitement : la cle que l'ancien code lisait
    with open(os.path.join(results_dir, "t22_unseen_unseen-ic_zz.json")) as f:
        raw = json.load(f)
    assert raw.get("total_abort_arm") is None, (
        "cette cle (singulier) n'a jamais existe dans l'artefact reel : "
        "c'est la lire qui produisait le 'None' affiche")


def test_both_arms_dead_reports_both_names(tmp_path):
    """Deux bras morts sur le meme fold : la liste porte les deux noms,
    pas seulement le premier."""
    results_dir = str(tmp_path)
    _write_total_abort_record(results_dir, "zz2", ["qhas", "classical"])
    rec = load(results_dir, "zz2")
    assert set(rec["total_abort_arms"]) == {"qhas", "classical"}


def test_main_prints_the_real_arm_name_not_none(tmp_path, capsys):
    """Bout en bout via `main()`, isole dans `tmp_path` (jamais
    `results/` du depot — voir la note en tete de fichier sur D-89/D-90)."""
    import argparse
    from unittest.mock import patch
    results_dir = str(tmp_path)
    _write_total_abort_record(results_dir, "zz3", ["qhas"])

    import h4_transfer_summary as mod
    with patch.object(mod, "RESULTS_DIR", results_dir, create=True), \
         patch("sys.argv", ["h4_transfer_summary.py", "--folds", "zz3"]), \
         patch("config.RESULTS_DIR", results_dir):
        try:
            mod.main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "the None arm" not in out, (
        "le message affiche encore 'None' au lieu du nom du bras (D-90)")
    assert "qhas" in out
