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


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _study_file(name):
    """Chemin d'un module de study/ quel que soit son dossier d'hypothese."""
    for _d in ("pipeline", "h0_selection", "h1_solver", "h2b_prediction",
               "h3_representation", "h4_transfer", "closed_loop", "common"):
        _c = os.path.join(_REPO_ROOT, "study", _d, name)
        if os.path.exists(_c):
            return _c
    raise FileNotFoundError(name)

_HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.join(_REPO_ROOT, "study")

import provenance

# Les taches longues : celles ou un commit peut tomber pendant l'execution.
# T20 dure ~50 min par fold, T22 ~1 h. Ce sont exactement celles qui ont
# ete mal estampillees.
LONG_TASKS = ("closed_loop_run_variance.py", "h4_unseen_conditions.py")


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
    src = open(_study_file(script), encoding="utf-8").read()
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
    """Plus aucun appel a `git_commit_hash()` dans les taches longues.

    Ce test cherche une CHAINE. Il est garde tel quel — il n'est pas faux —
    mais il ne mesure pas sa promesse : D-133 l'a montre par mutation. Le
    garde structurel est `test_long_tasks_never_call_the_stamp_under_any_name`.
    """
    src = open(_study_file(script), encoding="utf-8").read()
    assert "git_commit_hash()" not in src, (
        f"{script}: appelle encore git_commit_hash() — le tampon serait "
        f"pris a la sauvegarde (D15)")


@pytest.mark.parametrize("script", LONG_TASKS)
def test_long_tasks_never_call_the_stamp_under_any_name(script):
    """D-133 : la promesse mesuree sur l'AST, sous n'importe quel nom.

    Le test ci-dessus fait `assert "git_commit_hash()" not in src`. Mesure
    par mutation : D15 REINTRODUIT dans `h4_unseen_conditions.py` — une
    ligne `out["git_hash"] = _gch()` APRES `provenance.finish(prov)`,
    l'import ecrit `from provenance import git_commit_hash as _gch` — rend
    `grep -c "git_commit_hash()"` egal a **0** et laisse le fichier a
    **7 passed**. Le tampon redeviendrait celui de la SAUVEGARDE, c'est-a-dire
    exactement D15, et le garde ne le verrait pas.

    Ce test resout d'abord les alias d'import, puis cherche l'appel dans
    l'AST. Il couvre `git_commit_hash()`, `provenance.git_commit_hash()` et
    tout alias.

    LIMITE DECLAREE, pour qu'elle ne soit pas redecouverte : un
    `subprocess` qui appellerait `git rev-parse HEAD` a la main
    reintroduirait D15 sans passer par ce helper, et echapperait aux deux
    gardes. La promesse ecrite ici porte sur le HELPER ; l'invariant plus
    large est couvert par `test_start_then_finish_keeps_the_starting_hash`
    et `test_a_moved_head_is_reported_not_hidden`, qui mesurent ce que
    `provenance` rend.
    """
    src = open(_study_file(script), encoding="utf-8").read()
    tree = ast.parse(src)

    #  Tous les noms sous lesquels le tampon peut etre appele ici.
    names = {"git_commit_hash"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "provenance":
            for a in node.names:
                if a.name == "git_commit_hash":
                    names.add(a.asname or a.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        called = (f.id if isinstance(f, ast.Name)
                  else f.attr if isinstance(f, ast.Attribute) else None)
        assert called not in names, (
            f"{script}:{node.lineno} appelle le tampon sous le nom "
            f"`{called}` — le hash serait pris a la SAUVEGARDE, pas au "
            "depart (D15). Seul `provenance.start()` doit le prendre.")
