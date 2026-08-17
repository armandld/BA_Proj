"""Balayage systematique du motif de defaillance silencieuse (study/v4).

DIX-SEPT instances du MEME motif ont ete trouvees pendant cette campagne :
*un calcul qui echoue, ou ne fait pas ce qu'il annonce, et rend une valeur
indiscernable d'une valeur valide*. Neuf au fil de l'eau, trois par ce
balayage, et CINQ de plus en verifiant les NOMBRES PUBLIES contre leurs
artefacts — dont le decompte de tete de l'etude, compose a la main et faux
(cf. `study/v4/t23_headline_counts.py`).

  1-4. le garde-fou de divergence de V1 rend un score PARTIEL dont les cles
       sont identiques a celles d'une execution complete (T15, T20, T22 x2)
  5-7. des noms de fichiers de sortie fixes ecrasant silencieusement un
       resultat precedent (T13 mappers, T19 folds, T20 passe non protegee)
  8.   l'agregateur T16 moyennant les tirages avortes avec les valides
  9.   `--mode no-leak` accepte, documente, jamais implemente : seul le nom
       du fichier changeait

Une part d'entre elles etait dans le code de VERIFICATION ecrit pour
attraper les autres. Chercher au fil de l'eau ne suffit donc pas : ce module
balaie les formes verifiables mecaniquement, pour qu'aucune ne revienne en
silence.

La lecon la plus large ne se teste pas ici : TOUT nombre qu'aucun script ne
produit s'est revele faux, et tout nombre recalcule par
`t16_aggregate_v4.py` s'est revele juste. La parade n'est pas la relecture
— elle a ete appliquee et a laisse passer — mais le fait de rendre le
nombre fonction de l'artefact.

Ce balayage ne pretend pas etre exhaustif — il couvre les formes qui ont
reellement mordu. Ce qu'il garantit, c'est la NON-REGRESSION.
"""

import os
import sys

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

import ast
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.join(_REPO_ROOT, "study")

_STUDY_DIRS = ("pipeline", "h0_selection", "h1_solver", "h2b_prediction",
               "h3_representation", "h4_transfer", "closed_loop", "common")

TASK_SCRIPTS = sorted({
    f
    for _d in _STUDY_DIRS
    for f in os.listdir(os.path.join(_REPO_ROOT, "study", _d))
    if f.endswith(".py") and f not in ("__init__.py", "config.py")
})

# Un balayage qui ne balaie rien est exactement le motif que ce fichier
# traque : il doit crier plutot que passer.
assert len(TASK_SCRIPTS) >= 45, (
    f"seulement {len(TASK_SCRIPTS)} scripts balayes — la selection est "
    "cassee, pas le depot"
)


def _source(name):
    return open(_study_file(name), encoding="utf-8").read()


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

#: Les reducteurs numpy qui, appliques a une liste de tirages, produisent le
#: nombre publie. `np.sum` compte, les trois autres moyennent.
_REDUCTEURS = ("mean", "std", "median", "sum")


def _aggregations_sur_tirages(tree):
    """Appels `np.<reducteur>(...)` dont l'argument mentionne une liste de
    tirages, avec la ligne ou ils commencent.

    D-128 — la version precedente cherchait `np.mean(` et `_runs` sur la
    MEME ligne du source. Aucune des deux agregations reelles du depot n'est
    ecrite ainsi : elles tiennent sur deux lignes, la liste etant nommee un
    peu plus haut. Mesure : **0 ligne selectionnee sur 65 scripts**, pour
    **2 agregations reelles** — un balayage vide, dans le fichier qui existe
    pour detecter les balayages vides. L'AST voit la structure, pas la mise
    en forme : un retour a la ligne ne le desarme pas.
    """
    trouves = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in _REDUCTEURS
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "np"):
            continue
        noms = set()
        for c in ast.walk(n):
            if isinstance(c, ast.Name):
                noms.add(c.id)
            elif isinstance(c, ast.Attribute):
                noms.add(c.attr)
            elif isinstance(c, ast.Constant) and isinstance(c.value, str):
                noms.add(c.value)
        if any(x == "runs" or x.endswith("_runs") for x in noms):
            trouves.append((n, n.lineno))
    return trouves


def _mentionne_completed(node):
    for c in ast.walk(node):
        if isinstance(c, ast.Constant) and c.value == "completed":
            return True
        if isinstance(c, (ast.Name, ast.Attribute)):
            nom = getattr(c, "id", None) or getattr(c, "attr", "")
            if nom in ("completed", "n_ok", "runs_ok"):
                return True
    return False


def _liste_agregee_est_filtree(appel, tree):
    """Le filtre `completed` est-il DANS l'appel, ou dans la liaison du nom
    que l'appel agrege ?

    D-128 — la version precedente cherchait la chaine `completed` dans les
    12 lignes precedentes. Ce voisinage contient `\"n_completed\": len(runs)`,
    un champ de COMPTE-RENDU qui n'a rien d'un filtre : retirer le vrai
    filtre laissait le garde vert. On remonte donc du nom agrege a sa
    liaison, et on exige le filtre la.
    """
    if _mentionne_completed(appel):
        return True
    # noms iteres par les comprehensions de l'appel, ou passes directement
    sources = set()
    for c in ast.walk(appel):
        if isinstance(c, ast.comprehension) and isinstance(c.iter, ast.Name):
            sources.add(c.iter.id)
        elif isinstance(c, ast.Name):
            sources.add(c.id)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Assign):
            continue
        cibles = {t.id for t in n.targets if isinstance(t, ast.Name)}
        if cibles & sources and _mentionne_completed(n.value):
            return True
    return False


@pytest.mark.parametrize("script", TASK_SCRIPTS)
def test_aggregations_over_runs_filter_completed(script):
    """Une moyenne prise sur une liste `*_runs` doit filtrer `completed`.

    C'est le defaut qui a fait publier a T16 une moyenne de 0.3328 pour
    `rotor` la ou les tirages valides donnaient 0.1473."""
    tree = _tree(script)
    lignes = _source(script).splitlines()
    for appel, i in _aggregations_sur_tirages(tree):
        assert _liste_agregee_est_filtree(appel, tree), (
            f"{script}:{i} agrege des executions sans filtrer les avortees:\n"
            f"    {lignes[i - 1].strip()}")


def test_the_aggregation_sweep_is_not_empty():
    """Un balayage qui ne selectionne rien doit crier.

    D-128 : le garde ci-dessus passait sur les 65 scripts en n'examinant
    AUCUN site. Ce test compte ce qui est reellement examine, pour qu'une
    reecriture qui le redesarme se voie — c'est la regle « verifier le
    nombre de cas SELECTIONNES, pas seulement le code de retour ».

    Le nombre est ecrit ici pour qu'une derive se voie : **2** agregations
    sur tirages a la date de D-128, toutes deux dans
    `closed_loop_leak_free_summary.py` (lignes 126 et 127). Il peut monter
    legitimement ; il ne doit pas tomber a zero.
    """
    total = sum(len(_aggregations_sur_tirages(_tree(s))) for s in TASK_SCRIPTS)
    assert total >= 2, (
        f"le balayage des agregations sur tirages n'examine plus que "
        f"{total} site(s) sur {len(TASK_SCRIPTS)} scripts — il passerait "
        "au vert sans rien verifier, ce qui est le defaut D-128")


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
