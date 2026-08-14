import pathlib
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


def _dirs_missing_init(racine):
    """Dossiers portant du `.py` mais pas d'`__init__.py`, sous `racine`."""
    sans = []
    for base, dirs, files in os.walk(racine):
        dirs[:] = [d for d in dirs
                   if d not in ("__pycache__", "tools") and not d.startswith(".")]
        if not any(f.endswith(".py") for f in files):
            continue          # residu, pas un paquet manquant — voir ci-dessous
        if not os.path.exists(os.path.join(base, "__init__.py")):
            sans.append(base)
    return sans


def test_every_package_directory_carries_its_init():
    """Les imports croises passent par le paquet `tests.` : un dossier sans
    `__init__.py` les rendrait irresolvables.

    Un dossier qui ne contient AUCUN `.py` n'est pas un paquet manquant :
    c'est un residu. Apres la reorganisation de `tests/` par sous-systeme,
    `tests/v3/` et `tests/v4/` sont restes sur les copies de travail avec
    leur seul `__pycache__`. Git ne suit pas les dossiers vides : un clone
    neuf ne les a jamais eus, et les effacer ne tient pas — ils reviennent
    avec le repertoire de travail.

    Le test echouait donc sur l'etat d'une machine et non sur le contenu du
    depot : vrai localement, inexistant a l'arrivee. Le critere porte
    desormais sur les dossiers qui portent effectivement du code. Un vrai
    sous-dossier de test ajoute sans `__init__.py` en contient par
    construction : il reste attrape — c'est ce que verifie le test suivant.
    """
    sans = [os.path.relpath(d, _REPO_ROOT) for d in _dirs_missing_init(_TESTS)]
    assert not sans, f"dossiers de test sans __init__.py : {sans}"


def test_the_init_check_can_still_fail(tmp_path):
    """Garde-fou : sans lui, l'assouplissement ci-dessus pourrait rendre le
    test precedent incapable d'echouer, et personne ne le verrait."""
    vrai_manque = tmp_path / "sous_dossier"
    vrai_manque.mkdir()
    (vrai_manque / "test_quelque_chose.py").write_text("def test_x():\n    pass\n")

    residu = tmp_path / "residu" / "__pycache__"
    residu.mkdir(parents=True)
    (residu / "vieux.cpython-311.pyc").write_bytes(b"\x00")

    trouves = _dirs_missing_init(tmp_path)
    assert trouves == [str(vrai_manque)], (
        "un dossier portant un .py sans __init__.py doit etre signale, et le "
        f"residu sans .py doit etre ignore ; obtenu {trouves}")


# ══════════════════════════════════════════════════════════════════════
#  Aucun test ne doit etre incapable d'echouer
# ══════════════════════════════════════════════════════════════════════

#: Tests sans assertion explicite, mais qui echouent quand meme — chacun
#: avec la raison qui le rend legitime.
SANS_ASSERTION_LEGITIMES = {
    "tests/pipeline/test_src_coverage_inventory.py::test_every_module_imports_cleanly":
        "l'import EST la verification : un module casse leve ImportError",
    "tests/pipeline/test_src_coverage_inventory.py::test_every_entry_point_parses":
        "`ast.parse` EST la verification : une syntaxe invalide leve",
    "tests/quantum/test_qaoa_physics_decision.py::test_k_opt_30_with_psi_zero_limited":
        "diagnostic assume : son propre commentaire dit « we just measure "
        "and report ». A convertir en mesure epinglee ou a sortir de tests/",
}


def _porte_une_verification(fn, helpers):
    """Vrai si la fonction assertit, leve, ou appelle quelque chose qui le
    fait. Resout les helpers du module d'un niveau — le motif `_uniform(...)`
    de `test_analytic_fields.py` sans quoi 12 tests sains seraient signales."""
    for x in ast.walk(fn):
        if isinstance(x, (ast.Assert, ast.Raise)):
            return True
        if isinstance(x, ast.Call):
            nom = getattr(x.func, "attr", "") or getattr(x.func, "id", "")
            if nom.startswith(("assert", "check", "verifie", "_assert", "_check")):
                return True
            if nom in ("raises", "fail", "warns", "xfail", "exit"):
                return True
            if nom in helpers:
                return True
    return False


def _helpers_verificateurs(arbre):
    out = set()
    for n in arbre.body:
        if not isinstance(n, ast.FunctionDef):
            continue
        for x in ast.walk(n):
            if isinstance(x, (ast.Assert, ast.Raise)):
                out.add(n.name)
                break
            if isinstance(x, ast.Call):
                nm = getattr(x.func, "attr", "") or getattr(x.func, "id", "")
                if nm.startswith(("assert", "check")):
                    out.add(n.name)
                    break
    return out


def test_aucun_test_n_est_incapable_d_echouer():
    """« Un test qui ne peut pas echouer est un defaut » — CLAUDE.md.

    Trouve par ce balayage : `test_xpoint_selectivity` imprimait un
    rapport, branchait sur `if/else` pour choisir la phrase affichee, puis
    rendait un tuple. Zero assertion, sur le sujet meme — la selectivite du
    detecteur de points X — que le papier revendique. Il portait donc son
    verdict dans du texte, pas dans un resultat.
    """
    coupables = []
    for chemin in sorted(pathlib.Path(_TESTS).rglob("test_*.py")):
        try:
            arbre = ast.parse(chemin.read_text())
        except SyntaxError:
            continue
        helpers = _helpers_verificateurs(arbre)
        rel = os.path.relpath(chemin, _REPO_ROOT)
        for n in ast.walk(arbre):
            if not (isinstance(n, ast.FunctionDef) and n.name.startswith("test")):
                continue
            if _porte_une_verification(n, helpers):
                continue
            cle = f"{rel}::{n.name}"
            if cle not in SANS_ASSERTION_LEGITIMES:
                coupables.append(cle)

    assert not coupables, (
        "test(s) sans aucune verification — ils passent quoi qu'il arrive :\n  "
        + "\n  ".join(coupables))


def test_chaque_exemption_porte_sa_raison_et_existe_encore():
    """Une liste d'exemptions qui pourrit est pire que pas de liste."""
    for cle, raison in SANS_ASSERTION_LEGITIMES.items():
        assert len(raison) > 30, f"{cle} exempte sans raison lisible"
        fichier = cle.split("::")[0]
        assert os.path.exists(os.path.join(_REPO_ROOT, fichier)), (
            f"{fichier} n'existe plus — retirer son exemption")
