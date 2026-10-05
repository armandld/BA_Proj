"""D-202 — les bancs synthetiques comparent QAOA sur V1 a l'optimum exact sur V2.

`h3_toy_model_check.py` (statique) et `h3_toy_instability_check.py`
(dynamique) ont ete ecrits pour repliquer H0a et H0b hors des huit
scenarios : ils comparent la decision du QAOA a l'optimum exact « de son
propre hamiltonien ». Trouve a la revue d'octobre : l'optimum exact y est
calcule sur V2 (`build_patch_hamiltonian(..., use_v2=True)`), le QAOA sur le
hamiltonien que `prepare_qaoa_inputs` construit par defaut, V1. Les nombres
publies (« accord QAOA/exact » 0,622 et 0,719 ; « QAOA bat l'exact »)
comparent donc deux hamiltoniens differents et ne repliquent ni H0a ni
H0b.

Le defaut est OUVERT (`docs/DEFAUTS.md`, D-202). Il est epingle ici par un
xfail STRICT : le jour ou les deux harnais construisent les deux
hamiltoniens avec le meme mappeur, le test passe en XPASS, donc echoue, et
oblige a retirer le marqueur, a relancer les deux bancs et a mettre a jour
`RESULTS.md` et le preprint. Les deux autres tests, toujours verts,
verifient que la detection mord et decrivent l'etat exact du defaut.
"""
import ast
import inspect
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
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

HARNESSES = ("h3_toy_model_check.py", "h3_toy_instability_check.py")
BUILDERS = ("build_patch_hamiltonian", "prepare_qaoa_inputs")


def _defaults():
    """Valeur par defaut de `use_v2`, lue sur les VRAIES signatures."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        from exact_diagonalisation import build_patch_hamiltonian
        from qaoa_inputs import prepare_qaoa_inputs
    return {
        f.__name__: inspect.signature(f).parameters["use_v2"].default
        for f in (build_patch_hamiltonian, prepare_qaoa_inputs)}


def mappers_used(src, defaults):
    """{constructeur: ensemble des valeurs effectives de use_v2} dans `src`.

    Un `use_v2` absent vaut le defaut de la signature ; une valeur non
    constante vaut "?" (inconnue) plutot que d'etre devinee."""
    out = {b: set() for b in BUILDERS}
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        if name not in out:
            continue
        value = defaults[name]
        for kw in node.keywords:
            if kw.arg == "use_v2":
                value = (bool(kw.value.value)
                         if isinstance(kw.value, ast.Constant) else "?")
        out[name].add(value)
    return out


def _harness_source(name):
    path = os.path.join(_REPO_ROOT, "study", "h3_representation", name)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


@pytest.mark.xfail(strict=True, reason=(
    "D-202 ouvert : l'optimum exact est calcule sur V2 et le QAOA sur V1. "
    "XPASS = corrige : retirer ce marqueur, relancer les deux bancs, mettre "
    "a jour RESULTS.md, DEFAUTS.md et le preprint."))
@pytest.mark.parametrize("harness", HARNESSES)
def test_qaoa_and_exact_optimum_use_the_same_mapper(harness):
    used = mappers_used(_harness_source(harness), _defaults())
    assert all(used[b] for b in BUILDERS), used
    assert used["build_patch_hamiltonian"] == used["prepare_qaoa_inputs"], used


@pytest.mark.parametrize("harness", HARNESSES)
def test_the_open_defect_is_exactly_the_documented_one(harness):
    """Exact sur V2, QAOA sur V1 : l'etat decrit dans DEFAUTS.md, ni plus
    ni moins. Une modification partielle doit se voir ici."""
    used = mappers_used(_harness_source(harness), _defaults())
    assert used == {"build_patch_hamiltonian": {True},
                    "prepare_qaoa_inputs": {False}}, used


def test_the_detection_bites():
    """Le detecteur distingue un harnais coherent d'un harnais incoherent."""
    d = _defaults()
    assert d == {"build_patch_hamiltonian": False,
                 "prepare_qaoa_inputs": False}
    coherent = ("build_patch_hamiltonian(a, use_v2=True)\n"
                "prepare_qaoa_inputs(a, use_v2=True)\n")
    split = ("build_patch_hamiltonian(a, use_v2=True)\n"
             "prepare_qaoa_inputs(a)\n")
    unknown = "prepare_qaoa_inputs(a, use_v2=flag)\n"
    c = mappers_used(coherent, d)
    assert c["build_patch_hamiltonian"] == c["prepare_qaoa_inputs"] == {True}
    s = mappers_used(split, d)
    assert s["build_patch_hamiltonian"] != s["prepare_qaoa_inputs"]
    assert mappers_used(unknown, d)["prepare_qaoa_inputs"] == {"?"}
