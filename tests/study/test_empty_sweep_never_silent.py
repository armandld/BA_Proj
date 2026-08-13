"""D-56 : huit modules de `study/` sortaient avec le code 0 sur un balayage
vide, sans ecrire d'artefact — donc en laissant en place celui de la
campagne precedente.

`CLAUDE.md` : « Un test qui ne peut pas echouer est un defaut. Tout script de
`study/` ou de `tests/` porte une assertion, et un balayage vide doit
crier. » Onze modules levaient deja ; huit imprimaient `no input.` et
rendaient la main. D-55 en avait corrige un neuvieme
(`h3_term_ablation.py`) ; ceci ferme la famille.

Mesure avant / apres, meme commande
(`--scenario no_such_scenario --N 64`) :

    h3_locality_proposition.py   code 0 -> 1
    h3_equivariance.py           code 0 -> 1
    h2b_learned_meanfield_h.py   code 0 -> 1
"""
import ast
import glob
import os
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

#: Noms d'accumulateurs derriere lesquels un `if not <x>:` garde la fin
#: d'un balayage. Tenu a la main : c'est la liste des formes reellement
#: rencontrees dans `study/`.
ACCUMULATORS = {"rows", "records", "results", "configs", "by_scene",
                "per_cfg", "out_rows", "all_rows", "entries"}


def _silent_empty_sweeps(path):
    """Gardes `if not <accumulateur>:` dont le corps ne fait que rendre la
    main. Interroge l'AST, pas le texte du source : un test qui cherche la
    chaine « no input. » casserait sur une reformulation sans qu'aucun
    defaut n'existe."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If)
                and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and getattr(node.test.operand, "id", None) in ACCUMULATORS):
            continue
        raises = any(isinstance(n, ast.Raise) for n in ast.walk(node))
        returns = any(isinstance(n, ast.Return) for n in ast.walk(node))
        if returns and not raises:
            out.append((node.test.operand.id, node.lineno))
    return out


STUDY_FILES = sorted(glob.glob(os.path.join(_REPO_ROOT, "study", "**", "*.py"),
                               recursive=True))


@pytest.mark.parametrize("path", STUDY_FILES,
                         ids=[os.path.relpath(p, _REPO_ROOT) for p in STUDY_FILES])
def test_no_study_module_returns_silently_on_an_empty_sweep(path):
    silent = _silent_empty_sweeps(path)
    assert not silent, (
        f"{os.path.relpath(path, _REPO_ROOT)} rend la main sans lever sur un "
        f"balayage vide (accumulateur/ligne : {silent}). Le script sortirait "
        "avec le code 0 sans ecrire d'artefact, en laissant celui de la "
        "campagne precedente en place. Voir D-55 / D-56.")


def test_the_guard_actually_bites_on_a_real_module():
    """L'AST dit ce que le code CONTIENT ; ceci verifie ce qu'il FAIT.
    `h3_locality_proposition` rendait 0 sur ce meme appel."""
    r = subprocess.run(
        [sys.executable,
         os.path.join(_REPO_ROOT, "study", "h3_representation",
                      "h3_locality_proposition.py"),
         "--scenario", "no_such_scenario", "--N", "64"],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=600)
    assert r.returncode != 0, (
        "le balayage vide est redevenu silencieux : code de sortie 0")
    assert "balayage vide" in (r.stderr + r.stdout)


def test_the_detector_itself_can_fail():
    """Un balayage vide doit crier — y compris celui-ci. Si `_silent_empty_sweeps`
    ne detectait plus rien, les tests ci-dessus passeraient sans mesurer quoi
    que ce soit (le piege du balayage vide, dans le fichier cense le
    detecter)."""
    assert len(STUDY_FILES) > 40, (
        f"seulement {len(STUDY_FILES)} modules de study/ collectes : le "
        "balayage du detecteur est vide ou tronque")

    import tempfile
    src = "def main():\n    rows = []\n    if not rows:\n        print('x')\n        return\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        tmp = fh.name
    try:
        assert _silent_empty_sweeps(tmp) == [("rows", 3)]
    finally:
        os.unlink(tmp)
