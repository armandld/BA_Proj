"""Balayage systematique du motif de defaillance silencieuse (study/v4).

Neuf instances du MEME motif ont ete trouvees au fil de l'eau pendant cette
campagne : *un calcul qui echoue, ou ne fait pas ce qu'il annonce, et rend
une valeur indiscernable d'une valeur valide*.

  1-4. le garde-fou de divergence de V1 rend un score PARTIEL dont les cles
       sont identiques a celles d'une execution complete (T15, T20, T22 x2)
  5-7. des noms de fichiers de sortie fixes ecrasant silencieusement un
       resultat precedent (T13 mappers, T19 folds, T20 passe non protegee)
  8.   l'agregateur T16 moyennant les tirages avortes avec les valides
  9.   `--mode no-leak` accepte, documente, jamais implemente : seul le nom
       du fichier changeait

Quatre des neuf etaient dans le code de VERIFICATION ecrit pour attraper les
autres. Chercher au fil de l'eau ne suffit donc pas : ce module balaie les
formes verifiables mecaniquement, pour qu'aucune ne revienne en silence.

Ce balayage ne pretend pas etre exhaustif — il couvre les formes qui ont
reellement mordu. Ce qu'il garantit, c'est la NON-REGRESSION.
"""
import ast
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.abspath(os.path.join(_HERE, "..", "..", "study", "v4"))

TASK_SCRIPTS = sorted(
    f for f in os.listdir(V4)
    if f.startswith("t") and f.endswith(".py")
)


def _source(name):
    return open(os.path.join(V4, name), encoding="utf-8").read()


def _tree(name):
    return ast.parse(_source(name))


# ---------------------------------------------------------------- (a)
# Une option CLI acceptee doit AGIR. Le cas `--mode no-leak` ne changeait
# que le nom du fichier de sortie : l'artefact portait un nom affirmant un
# calcul qui n'avait pas eu lieu.

def _choice_literals(tree):
    """Valeurs de `choices=[...]` passees a add_argument, par script."""
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg == "choices" and isinstance(kw.value, (ast.List,
                                                            ast.Tuple)):
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(
                            elt.value, str):
                        out.append(elt.value)
    return out


def _choice_groups(tree):
    """Groupes de valeurs, un par `choices=[...]`."""
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg == "choices" and isinstance(kw.value, (ast.List,
                                                             ast.Tuple)):
                vals = [e.value for e in kw.value.elts
                        if isinstance(e, ast.Constant)
                        and isinstance(e.value, str)]
                if vals:
                    out.append(vals)
    return out


@pytest.mark.parametrize("script", TASK_SCRIPTS)
def test_every_cli_choice_group_is_acted_upon(script):
    """Chaque GROUPE d'options a choix doit etre teste dans le corps.

    On n'exige pas que chaque valeur soit comparee : dans un choix binaire
    l'une des deux est legitimement le `else` implicite
    (`use_v2=(args.mapper == "v2")`). Ce qui doit etre vrai, c'est qu'au
    moins une valeur du groupe soit comparee quelque part — sinon l'option
    entiere est decorative, comme l'etait `--mode no-leak`, et nomme un
    artefact pour un calcul qui n'a pas eu lieu."""
    src = _source(script)
    for group in _choice_groups(_tree(script)):
        acted = any(
            re.search(rf'==\s*["\']{re.escape(c)}["\']'
                      rf'|["\']{re.escape(c)}["\']\s*==', src)
            for c in group)
        assert acted, (
            f"{script}: aucune valeur de {group} n'est comparee dans le "
            f"corps — l'option ne peut rien changer au calcul tout en "
            f"nommant un artefact")


# ---------------------------------------------------------------- (b)
# Toute agregation sur des executions doit exclure les avortees.

AGG = re.compile(r"np\.(mean|std|median|sum)\s*\(")


@pytest.mark.parametrize("script", TASK_SCRIPTS)
def test_aggregations_over_runs_filter_completed(script):
    """Une moyenne prise sur une liste `*_runs` doit filtrer `completed`.

    C'est le defaut qui a fait publier a T16 une moyenne de 0.3328 pour
    `rotor` la ou les tirages valides donnaient 0.1473."""
    src = _source(script)
    for i, line in enumerate(src.splitlines(), 1):
        if not AGG.search(line):
            continue
        if "_runs" not in line:
            continue
        # la ligne agrege des executions : le filtre doit etre visible
        # dans un voisinage proche (meme ligne ou definition juste avant)
        window = "\n".join(src.splitlines()[max(0, i - 12):i])
        assert "completed" in window or "_ok" in line, (
            f"{script}:{i} agrege des executions sans filtrer les avortees:\n"
            f"    {line.strip()}")


# ---------------------------------------------------------------- (c)
# Un chemin de sortie doit porter ce qui distingue son contenu.

OUTPATH = re.compile(r'f"(t\d+[a-z]*_[^"]*)"')


@pytest.mark.parametrize("script", TASK_SCRIPTS)
def test_output_filenames_carry_their_distinguishing_arguments(script):
    """Si un script accepte une option a choix, son fichier de sortie doit
    porter cette option — sinon deux modes s'ecrasent l'un l'autre.

    C'est le defaut D9 : `t13` ecrivait le meme nom pour les mappeurs v1 et
    v2, donc la comparaison qui justifiait la tache ne pouvait pas tenir
    dans l'artefact."""
    src = _source(script)
    choices = _choice_literals(_tree(script))
    if not choices:
        pytest.skip("pas d'option a choix")
    paths = OUTPATH.findall(src)
    if not paths:
        pytest.skip("pas de chemin de sortie litteral")
    # Le nom de sortie doit varier avec l'option a choix. Soit il
    # l'interpole directement, soit le script le suffixe conditionnellement
    # (forme retenue quand un nom historique doit rester valide).
    interpolates = any(re.search(r"\{args\.(mapper|mode)", p) for p in paths)
    conditional = re.search(
        r'\+\s*\(""\s+if\s+args\.(mapper|mode)\s*==', src)
    assert interpolates or conditional, (
        f"{script}: le(s) nom(s) de sortie {paths} ne varient pas avec "
        f"{choices} — deux modes ecriraient le meme fichier et le second "
        f"effacerait le premier (defaut D9)")


# ---------------------------------------------------------------- (d)
# Pas d'exception avalee sans trace.

@pytest.mark.parametrize("script", TASK_SCRIPTS)
def test_no_bare_except_that_silently_passes(script):
    """`except: pass` masque une defaillance et laisse le calcul continuer
    avec un etat inconnu. Les seuls tolerees ici sont celles dont le corps
    est explicitement commente comme sans consequence."""
    tree = _tree(script)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = node.body
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            src_lines = _source(script).splitlines()
            ctx = "\n".join(src_lines[max(0, node.lineno - 12):node.lineno])
            # tolere : fermeture des figures matplotlib, dont l'echec ne
            # peut pas fausser un resultat (le mode verbeux de V1 ouvre des
            # figures qu'il ne ferme jamais, cf. la fuite memoire corrigee)
            if "_plt.close" in ctx or "matplotlib" in ctx:
                continue
            assert "#" in ctx, (
                f"{script}:{node.lineno} `except: pass` sans justification "
                f"ecrite — une defaillance y disparait sans trace")
