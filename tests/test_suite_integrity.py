"""La suite se teste elle-meme : ses imports croises resolvent-ils ?

Ecrit apres une panne silencieuse de deux commits. La reorganisation de
`tests/` a deplace `test_qaoa_scaling_and_hparams.py` de `tests/` vers
`tests/quantum/`, et le script a mis a jour les imports DANS les fichiers
deplaces — pas ceux qui les designaient depuis un autre fichier.

`tests/quantum/test_qaoa_arm_is_sampled.py` importait encore
`tests.test_qaoa_scaling_and_hparams`, dans le corps d'une fixture. Six
tests echouaient donc a la PREPARATION, pas a la collecte : `--collect-only`
restait a zero erreur, et l'echec ne se voyait qu'en lancant la suite
entiere jusqu'au bout.

Le controle est bon marche : on n'importe rien, on demande seulement a
`importlib` si le module existe.
"""
import ast
import importlib.util
import os
import sys

import pytest


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_REPO_ROOT = _repo_root()
_TESTS = os.path.join(_REPO_ROOT, "tests")


def _test_files():
    out = []
    for base, dirs, files in os.walk(_TESTS):
        dirs[:] = [d for d in dirs
                   if d not in ("__pycache__", "tools") and not d.startswith(".")]
        out.extend(os.path.join(base, f) for f in files
                   if f.endswith(".py") and f != "__init__.py")
    return sorted(out)


def _internal_imports(path):
    """Les `import tests…` / `from tests… import …` du fichier, ou qu'ils
    soient — y compris a l'interieur d'une fonction ou d'une fixture."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    noms = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "tests" or node.module.startswith("tests."):
                noms.append((node.module, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tests" or alias.name.startswith("tests."):
                    noms.append((alias.name, node.lineno))
    return noms


def test_there_are_test_files_to_check():
    """Un balayage vide sort en vert et ne prouve rien."""
    assert len(_test_files()) > 40, f"{len(_test_files())} fichiers trouves"


@pytest.mark.parametrize("path", _test_files(), ids=lambda p: os.path.relpath(p, _TESTS))
def test_every_cross_test_import_resolves(path):
    """Un import vers un fichier de test deplace ne se voit qu'a
    l'execution, et seulement si le test concerne tourne jusqu'au bout."""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    manquants = []
    for module, ligne in _internal_imports(path):
        try:
            trouve = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            trouve = False
        if not trouve:
            manquants.append(f"ligne {ligne} : {module}")
    assert not manquants, (
        f"{os.path.relpath(path, _TESTS)} designe des modules de test qui "
        f"n'existent pas — {manquants}")


def test_every_package_directory_carries_its_init():
    """Les imports croises passent par le paquet `tests.` : un dossier sans
    `__init__.py` les rendrait irresolvables."""
    sans = []
    for base, dirs, _ in os.walk(_TESTS):
        dirs[:] = [d for d in dirs
                   if d not in ("__pycache__", "tools") and not d.startswith(".")]
        if not os.path.exists(os.path.join(base, "__init__.py")):
            sans.append(os.path.relpath(base, _REPO_ROOT))
    assert not sans, f"dossiers de test sans __init__.py : {sans}"
