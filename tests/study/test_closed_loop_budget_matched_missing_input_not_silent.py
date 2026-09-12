"""D-74 : `closed_loop_budget_matched.py` rendait la main avec le code 0 sur
un fold manquant ou inconnu, sans rien ecrire — un balayage vide (CLAUDE.md :
« un balayage vide doit crier ») indiscernable d'une execution reussie.

Meme famille que D-56, deja corrigee sur les 11 autres sites de `study/`
listes dans `docs/RESULTS.md` — `closed_loop_campaign.py`, son voisin direct
dans ce dossier, compris. Ce fichier-ci n'avait ni `assert` ni `raise`
(mesure AST) et etait reste hors de la correction : ses deux gardes
faisaient `print(...); return`.

Mesure avant / apres, meme commande
(`--fold no_such_scenario_xyz`, fold manquant) : code 0 -> 1.
"""
import ast
import json
import os
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_SCRIPT = os.path.join(_REPO_ROOT, "study", "closed_loop",
                       "closed_loop_budget_matched.py")
_RESULTS_DIR = os.path.join(_REPO_ROOT, "results")


def _run(args, timeout=60):
    return subprocess.run(
        [sys.executable, _SCRIPT, *args],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=timeout)


def test_missing_fold_artifact_is_not_silent():
    """Fold sans artefact `t15_level3_fold_*.json` : doit lever, pas rendre
    la main avec le code 0. C'est la mesure « avant » de D-74 : 0 -> non-0."""
    r = _run(["--fold", "no_such_scenario_xyz"])
    assert r.returncode != 0, (
        "le balayage vide (fold manquant) est redevenu silencieux : "
        "code de sortie 0")
    assert "D-74" in (r.stdout + r.stderr) or "balayage vide" in (
        r.stdout + r.stderr)


def test_unknown_fold_with_a_stray_artifact_is_not_silent():
    """Un fold dont l'artefact t15 existe mais qui ne correspond a AUCUN
    scenario connu de `train_hyperparams` (faute de frappe, artefact perime)
    doit lui aussi lever plutot que rendre la main : c'est la seconde garde
    silencieuse que D-74 corrige, distincte de la premiere (celle-ci est
    atteinte APRES la lecture du fold, donc ne peut pas etre exercee par un
    fold simplement absent)."""
    fake_fold = "not_a_real_scenario_zzz"
    path = os.path.join(_RESULTS_DIR, f"t15_level3_fold_{fake_fold}.json")
    assert not os.path.exists(path), (
        "un artefact de test porte deja ce nom ; en choisir un autre pour "
        "ne pas ecraser un resultat reel")
    json.dump({"qhas": {"patch_ratio": 0.5}}, open(path, "w"))
    try:
        r = _run(["--fold", fake_fold])
    finally:
        os.remove(path)
    assert r.returncode != 0, (
        "le balayage vide (fold inconnu de train_hyperparams) est redevenu "
        "silencieux : code de sortie 0")
    assert "D-74" in (r.stdout + r.stderr) or "balayage vide" in (
        r.stdout + r.stderr)


def _has_silent_print_then_return(path):
    """AST : un bloc `if ...: print(...); return` sans aucun `raise`."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        calls_print = any(
            isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"
            for n in ast.walk(node))
        returns = any(isinstance(n, ast.Return) for n in ast.walk(node))
        raises = any(isinstance(n, ast.Raise) for n in ast.walk(node))
        if calls_print and returns and not raises:
            out.append(node.lineno)
    return out


def test_no_print_then_return_guard_remains_in_the_source():
    """Verification statique, independante de l'execution : aucune garde
    `print(...); return` sans `raise` ne doit rester dans ce fichier."""
    silent = _has_silent_print_then_return(_SCRIPT)
    assert not silent, (
        f"closed_loop_budget_matched.py rend encore la main sans lever sur "
        f"un balayage vide (lignes : {silent})")


def test_the_detector_itself_can_fail():
    """Le detecteur AST doit pouvoir trouver quelque chose, sinon les deux
    tests ci-dessus passent sans avoir rien verifie."""
    import tempfile
    src = ("def main():\n"
           "    if not True:\n"
           "        print('x')\n"
           "        return\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        tmp = fh.name
    try:
        assert _has_silent_print_then_return(tmp) == [2]
    finally:
        os.unlink(tmp)
