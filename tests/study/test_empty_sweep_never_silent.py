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


# ══════════════════════════════════════════════════════════════════
#  D-148 — le meme garde, par le COMPORTEMENT
# ══════════════════════════════════════════════════════════════════
#
# Le detecteur AST ci-dessus cherche UNE forme (`if not <nom>: ... return`)
# avec une liste de noms tenue a la main. Balayage generalise a tout nom et
# a toute forme : 30 sites de `study/` repondent, AUCUN n'est dans
# `ACCUMULATORS` — le detecteur en voyait zero. Et six modules sortaient
# reellement avec le code 0 sans rien ecrire, dont les phases 2, 3, 4, 5, 7
# et 8 du pipeline, alors que l'en-tete de ce fichier annonce « ceci ferme
# la famille ».
#
# La lecon est celle de D-56 elle-meme, d'un cran plus haut : D-56 avait
# trouve trois modules par l'AST que la recherche de la chaine « no input. »
# aurait manques, parce que leur MESSAGE differait. Ici c'est la FORME qui
# differe — `if <accumulateur>:` autour du resume, puis on tombe en bas de
# `main()`. Aucune recherche de forme ne ferme une famille ; seul le
# comportement le fait.
#
# Ce test ne cherche donc aucune forme : il execute chaque module sur une
# demande qui ne correspond a rien et exige un code de sortie non nul.
# ~2 min pour 61 modules.

#: Modules exemptes, avec leur raison. Une exemption sans raison ecrite se
#: fait etendre par erreur.
_EXEMPTIONS = {
    "study/pipeline/dns_validation.py": (
        "GELE — ses artefacts sont publies, et une correction y a deja ete "
        "annulee. Mesure D-148 : il sort bien avec le code 0 sur "
        "`--scenario no_such_scenario`, MAIS il ECRIT "
        "`results/dns_validation_N64.npz` — sa sortie n'est donc pas "
        "indiscernable d'une campagne reussie faute d'artefact, elle EST "
        "une campagne qui a produit quelque chose. Non corrige, "
        "volontairement."),
}


def _modules_lancables():
    out = []
    for p in STUDY_FILES:
        src = open(p, encoding="utf-8").read()
        if "__main__" in src and "argparse" in src:
            out.append(p)
    return out


_LANCABLES = _modules_lancables()


def test_le_balayage_comportemental_nest_pas_vide():
    """Un balayage vide doit crier — y compris celui-ci."""
    assert len(_LANCABLES) > 50, (
        f"{len(_LANCABLES)} modules lancables collectes : le balayage de "
        "D-148 est vide ou tronque, et les tests ci-dessous ne prouveraient "
        "rien")


def test_chaque_exemption_de_D148_porte_sa_raison_et_existe_encore():
    for rel, raison in _EXEMPTIONS.items():
        assert os.path.exists(os.path.join(_REPO_ROOT, rel)), (
            f"{rel} est exempte mais n'existe plus : retirer l'exemption")
        assert len(raison) > 80, f"{rel} : exemption sans raison ecrite"


@pytest.mark.parametrize(
    "path", _LANCABLES,
    ids=[os.path.relpath(p, _REPO_ROOT) for p in _LANCABLES])
def test_aucun_module_de_study_ne_sort_zero_sur_un_balayage_vide(path):
    """`--scenario no_such_scenario` ne correspond a rien : un module qui
    rend 0 la-dessus est indiscernable d'une campagne reussie."""
    rel = os.path.relpath(path, _REPO_ROOT)
    if rel in _EXEMPTIONS:
        pytest.skip(_EXEMPTIONS[rel])
    r = subprocess.run(
        [sys.executable, path, "--scenario", "no_such_scenario", "--N", "64"],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=300)
    assert r.returncode != 0, (
        f"{rel} sort avec le code 0 sur un balayage qui ne correspond a "
        f"rien, sans ecrire d'artefact : il laisse en place ceux de la "
        f"campagne precedente et ne se distingue pas d'une campagne "
        f"reussie. Dernieres lignes :\n"
        + "\n".join((r.stdout + r.stderr).strip().split("\n")[-5:]))


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
