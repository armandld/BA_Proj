"""D-75 : quatorze fichiers de `study/` rendaient encore la main avec le code 0
sur un balayage vide — la forme sœur de D-56 et de D-74, invisible au
detecteur ecrit pour D-56.

`tests/study/test_empty_sweep_never_silent.py` (D-56) ne reconnait qu'une
seule forme syntaxique : `if not <accumulateur>:` ou `<accumulateur>` doit en
plus figurer dans une liste de noms tenue a la main (`rows`, `configs`,
`by_scene`, ...). Les gardes reelles de `study/h2b_prediction/` s'ecrivent
`if len(by_scene) < 2:`, `if len(set(...)) < 2:`, `if not Xs:`,
`if not all_d:`, `if not cfgs:` — aucune ne correspond. D-74 avait deja
paye ce prix dans `study/closed_loop/`, un site a la fois.

Mesure avant / apres, meme commande
(`--scenario no_such_scenario --N 64`), code de sortie du processus :

    h2b_feature_selection.py        0 -> 1
    h2b_loso_transfer.py            0 -> 1
    h2b_loso_bootstrap.py           0 -> 1
    h2b_neighbour_cone_curve.py     0 -> 1
    h2b_prediction_horizon.py       0 -> 1
    h2b_psi_feature_loso.py         0 -> 1
    h2b_v1_hamiltonian_loso.py      0 -> 1
    h2b_multiseed.py                0 -> 1
    h2b_random_split_bootstrap.py   0 -> 1
    h2b_scenario_ablation.py        0 -> 1
    h2b_dynamic_ground_truth.py     0 -> 1
    label_percentile_sensitivity.py 0 -> 1
    hard_patch_labels.py            0 -> 1   (mesure : `results/` vide)

Ce detecteur-ci ne connait aucune liste de noms : il flaire la FORME
« garde de donnees qui rend la main sans lever », quelle que soit
l'expression testee. Les sorties anticipees pilotees par un DRAPEAU CLI
(`if args.list:`, `if args.dry_run:`) sont exemptees — elles sont voulues,
et leur condition ne parle que de `args`.
"""
import ast
import glob
import os
import subprocess
import sys
import textwrap

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _condition_is_cli_flag_only(test):
    """Vrai si la condition ne parle que des arguments de la ligne de commande.

    `if args.dry_run:` est une sortie anticipee VOULUE : l'utilisateur l'a
    demandee, rien n'a echoue. `if len(by_scene) < 2:` parle des donnees.
    """
    roots = set()
    for node in ast.walk(test):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            roots.add(node.value.id)
        elif isinstance(node, ast.Name):
            roots.add(node.id)
    return bool(roots) and roots <= {"args", "os", "sys"}


def _statements_of(node):
    """Descend dans `node` sans entrer dans une fonction imbriquee.

    Une fermeture definie dans `main` a son propre `return` : le compter
    ferait crier le detecteur sur du code sain (`h2b_neighbour_cone_curve`
    en porte une)."""
    out = []
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        out.append(n)
        stack.extend(ast.iter_child_nodes(n))
    return out


def _is_zero_exit(node):
    """`sys.exit()` ou `sys.exit(0)` — un `sys.exit(1)` est un cri, pas un defaut."""
    if not (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "exit"):
        return False
    if not node.args:
        return True
    a = node.args[0]
    return isinstance(a, ast.Constant) and a.value in (0, None)


def silent_data_guards(path):
    """Gardes de `main()` qui rendent la main avec le code 0 sur des donnees
    manquantes ou degenerees. Interroge l'AST, jamais le texte du source."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "main"]:
        for node in _statements_of(fn):
            if not isinstance(node, ast.If):
                continue
            if _condition_is_cli_flag_only(node.test):
                continue
            body = [n for stmt in node.body for n in [stmt] + _statements_of(stmt)]
            if any(isinstance(n, ast.Raise) for n in body):
                continue
            leaves = (any(isinstance(n, ast.Return) for n in body)
                      or any(_is_zero_exit(n) for n in body))
            if leaves:
                out.append((ast.unparse(node.test)[:60], node.lineno))
    return out


STUDY_FILES = sorted(glob.glob(os.path.join(_REPO_ROOT, "study", "**", "*.py"),
                               recursive=True))


@pytest.mark.parametrize("path", STUDY_FILES,
                         ids=[os.path.relpath(p, _REPO_ROOT) for p in STUDY_FILES])
def test_no_main_leaves_with_code_zero_on_missing_or_degenerate_data(path):
    silent = silent_data_guards(path)
    assert not silent, (
        f"{os.path.relpath(path, _REPO_ROOT)} : garde(s) de donnees qui rendent "
        f"la main sans lever (condition/ligne : {silent}). Le processus sort "
        "avec le code 0 sans ecrire d'artefact — indiscernable d'une campagne "
        "reussie. Voir D-56, D-74, D-75.")


@pytest.mark.parametrize("relpath", [
    "study/h2b_prediction/h2b_loso_transfer.py",
    "study/h2b_prediction/h2b_multiseed.py",
    "study/pipeline/label_percentile_sensitivity.py",
])
def test_the_guard_actually_bites_on_a_real_module(relpath):
    """L'AST dit ce que le code CONTIENT ; ceci verifie ce qu'il FAIT.
    Les trois rendaient 0 sur ce meme appel."""
    r = subprocess.run(
        [sys.executable, os.path.join(_REPO_ROOT, relpath),
         "--scenario", "no_such_scenario", "--N", "64"],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=900)
    assert r.returncode != 0, (
        f"{relpath} : le balayage vide est redevenu silencieux (code 0)")
    assert "balayage vide" in (r.stdout + r.stderr)
    assert "D-75" in (r.stdout + r.stderr)


def test_the_detector_sees_the_shapes_that_were_actually_there():
    """Epingle l'ancien comportement : les cinq formes reellement rencontrees
    dans `study/h2b_prediction/`. Le detecteur de D-56 n'en voyait aucune —
    c'est ce trou qui a laisse quatorze fichiers dehors."""
    src = textwrap.dedent("""
        def main():
            if len(by_scene) < 2:
                print("need >=2 scenarios."); return
            if len(set(c[0] for c in configs)) < 2:
                print("Need >=2 scenarios for LOSO."); return
            if not Xs:
                print("no inputs."); return
            if not all_d:
                print("no output."); return
            if len(np.unique(Ytr)) < 2:
                print("degenerate training set."); return
    """)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        tmp = fh.name
    try:
        found = silent_data_guards(tmp)
    finally:
        os.unlink(tmp)
    assert len(found) == 5, f"le detecteur ne voit que {found}"

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "_d56_detector",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "test_empty_sweep_never_silent.py"))
    _d56 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_d56)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        tmp = fh.name
    try:
        d56_found = _d56._silent_empty_sweeps(tmp)
    finally:
        os.unlink(tmp)
    assert d56_found == [], (
        "le detecteur de D-56 voit maintenant ces formes : cette assertion "
        "documente POURQUOI ce fichier existe — si elle tombe, fusionner les "
        "deux detecteurs plutot que de la relacher")


def test_the_detector_does_not_cry_on_a_cli_flag_or_a_loud_exit():
    """Un faux positif coute plus cher qu'un defaut manque : il envoie
    corriger du code correct. `--dry-run` et `sys.exit(1)` sont sains."""
    src = textwrap.dedent("""
        def main():
            if args.dry_run:
                print("dry run"); return
            if args.strict and n_diff:
                sys.exit(1)
            def _inner(k):
                if k is None:
                    return 0
                return k
            if not rows:
                raise RuntimeError("balayage vide")
    """)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        tmp = fh.name
    try:
        assert silent_data_guards(tmp) == []
    finally:
        os.unlink(tmp)


def test_the_sweep_itself_is_not_empty():
    """Un balayage vide doit crier — y compris celui-ci."""
    assert len(STUDY_FILES) > 40, (
        f"seulement {len(STUDY_FILES)} modules de study/ collectes : le "
        "balayage du detecteur est vide ou tronque")
    assert sum(1 for p in STUDY_FILES
               if "main" in open(p, encoding="utf-8").read()) > 30
