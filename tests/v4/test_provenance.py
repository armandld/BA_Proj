"""D15 — le tampon de provenance doit decrire le code qui a TOURNE.

`git_commit_hash()` etait appele au moment de sauvegarder. Une execution
d'une heure recevait donc le hash de ce qui avait ete commite *pendant*
qu'elle tournait. Les artefacts T20 de `ot` et `kh` en portent la trace :
leur hash est posterieur au commit qui a change leur propre controle
classique, alors qu'ils ont execute la version d'avant.
"""
import ast
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.abspath(os.path.join(_HERE, "..", "..", "study", "v4"))
sys.path.insert(0, V4)

import provenance

# Les taches longues : celles ou un commit peut tomber pendant l'execution.
# T20 dure ~50 min par fold, T22 ~1 h. Ce sont exactement celles qui ont
# ete mal estampillees.
LONG_TASKS = ("t20_qhas_run_variance.py", "t22_unseen_conditions.py")


def test_start_then_finish_keeps_the_starting_hash():
    """`git_hash` doit valoir le hash de DEPART, pas celui de fin.

    C'est la cle que lisent les agregateurs ; si elle prenait la valeur de
    fin, le correctif ne servirait a rien."""
    p = provenance.start()
    out = provenance.finish(p)
    assert out["git_hash"] == p["git_hash_at_start"]
    assert out["git_hash_at_start"] == p["git_hash_at_start"]
    assert "git_hash_at_save" in out
    assert "head_moved_during_run" in out


def test_a_moved_head_is_reported_not_hidden():
    """Si HEAD bouge pendant l'execution, l'artefact doit le DIRE.

    Sans ce drapeau, un artefact estampille d'un hash qui n'a pas tourne est
    indiscernable d'un artefact correct — le motif meme de cette campagne."""
    fake = {"git_hash_at_start": "a" * 40, "dirty_at_start": False}
    out = provenance.finish(fake)
    # HEAD reel != le faux hash de depart
    assert out["head_moved_during_run"] is True
    assert out["git_hash"] == "a" * 40


def test_unknown_hash_does_not_claim_a_move():
    """Sans git, on ne peut rien affirmer : ne pas inventer un deplacement."""
    out = provenance.finish({"git_hash_at_start": "unknown",
                             "dirty_at_start": None})
    assert out["head_moved_during_run"] is False


@pytest.mark.parametrize("script", LONG_TASKS)
def test_long_tasks_capture_the_hash_before_computing(script):
    """`provenance.start()` doit preceder le premier `run_arm`.

    Prendre le hash apres coup, meme via ce module, reproduirait D15."""
    src = open(os.path.join(V4, script), encoding="utf-8").read()
    tree = ast.parse(src)
    start_line = run_line = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == "start"
                and isinstance(f.value, ast.Name)
                and f.value.id == "provenance"):
            start_line = node.lineno if start_line is None else min(
                start_line, node.lineno)
        if isinstance(f, ast.Name) and f.id == "run_arm":
            run_line = node.lineno if run_line is None else min(
                run_line, node.lineno)
    assert start_line is not None, f"{script}: provenance.start() absent"
    if run_line is not None:
        assert start_line < run_line, (
            f"{script}: provenance.start() ligne {start_line} vient APRES le "
            f"premier run_arm ligne {run_line} — le hash ne decrirait pas le "
            f"code execute")


@pytest.mark.parametrize("script", LONG_TASKS)
def test_long_tasks_no_longer_stamp_at_save_time(script):
    """Plus aucun appel a `git_commit_hash()` dans les taches longues."""
    src = open(os.path.join(V4, script), encoding="utf-8").read()
    assert "git_commit_hash()" not in src, (
        f"{script}: appelle encore git_commit_hash() — le tampon serait "
        f"pris a la sauvegarde (D15)")
