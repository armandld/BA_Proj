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

# D-128 (suite). Le garde ci-dessus (premiere passe de D-128) ne voyait un
# site que si le mot "runs"/"_runs" apparaissait LITTERALEMENT dans les
# tokens de l'appel `np.<reducteur>(...)` lui-meme -- Name, Attribute ou
# chaine. Mesure sur ce depot : **2 sites sur 65 scripts**, tous deux dans
# `closed_loop_leak_free_summary.py`. Mais le motif dominant du depot est le
# passage par un PARAMETRE DE FONCTION : `summarise(q_ok)` filtre
# `q_ok` chez l'APPELANT, puis calcule `np.mean(v)` DANS `summarise`, sur un
# nom `v` qui ne contient ni "runs" ni "_runs" -- invisible au premier
# garde. Rejoue avec une trace de provenance (variable -> conteneur
# `*_runs`, a travers UNE indirection d'appel de fonction locale) : **20
# sites sur 5 scripts** (`aggregate_master_table.py`: 2,
# `closed_loop_leak_free_summary.py`: 2, `closed_loop_run_variance.py`: 5
# via `summarise()`, `h4_physics_robustness.py`: 5,
# `h4_unseen_conditions.py`: 6). Les 2 sites d'origine sont un
# sous-ensemble exact des 20 -- confirme, pas contredit.
#
# Garde aussi contre l'inversion : `if not r["completed"]` GARDE les
# tirages avortes, l'oppose de l'intention, et ne doit pas compter comme un
# filtre valide (deja rencontre comme classe de defaut sous le nom
# "convention d'axes inversee").

_RUNS_SRC = re.compile(r"_runs\b")


def _a_filtre_completed_non_nie(gen_ifs):
    for f in gen_ifs:
        if isinstance(f, ast.UnaryOp) and isinstance(f.op, ast.Not):
            continue
        if "completed" in ast.unparse(f):
            return True
    return False


def _statut_source_iteree(iter_node, statuts_locaux):
    """'raw' (vient d'un `*_runs` non filtre) / 'filtered' / None (rien a
    voir avec des tirages), pour la source iteree par une comprehension."""
    if isinstance(iter_node, ast.Name) and iter_node.id in statuts_locaux:
        return statuts_locaux[iter_node.id]
    if isinstance(iter_node, ast.BinOp):
        cotes = [_statut_source_iteree(iter_node.left, statuts_locaux),
                 _statut_source_iteree(iter_node.right, statuts_locaux)]
        cotes = [s for s in cotes if s]
        if not cotes:
            return None
        return "raw" if "raw" in cotes else "filtered"
    if _RUNS_SRC.search(ast.unparse(iter_node)):
        return "raw"
    return None


def _statuts_locaux(fonction, graine_params=None):
    """Statuts locaux (nom -> 'raw'/'filtered') d'une fonction du module,
    pour du code lineaire -- vrai de tous les scripts de `study/`."""
    statuts = dict(graine_params or {})
    for n in ast.walk(fonction):
        if not (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)):
            continue
        nom = n.targets[0].id
        rhs = n.value
        comp = rhs
        if isinstance(comp, ast.Call):
            # np.array([... for x in NOM], dtype=float) : la comprehension
            # est un ARGUMENT de l'appel, pas le rhs lui-meme.
            comp = next(
                (a for a in comp.args
                 if isinstance(a, (ast.ListComp, ast.GeneratorExp,
                                   ast.SetComp))), None)
        derive = None
        if isinstance(comp, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            gen = comp.generators[0]
            base = _statut_source_iteree(gen.iter, statuts)
            if base is not None:
                derive = ("filtered" if _a_filtre_completed_non_nie(gen.ifs)
                          else base)
        elif isinstance(rhs, ast.Name) and rhs.id in statuts:
            derive = statuts[rhs.id]
        elif (isinstance(rhs, ast.List) and not rhs.elts
              and _RUNS_SRC.search(nom)):
            derive = "raw"  # accumulateur : `q_runs = []` puis `.append`
        elif _RUNS_SRC.search(ast.unparse(rhs)):
            derive = "raw"  # ex. `t["qhas_runs"]`, `q.get(f"{c}_runs", [])`
        if derive is not None:
            statuts[nom] = derive
    return statuts


def _agregations_avec_statut(tree):
    """(call_node, lineno, statut ou None) pour chaque
    `np.<reducteur>(...)` du module, en tracant sa provenance a travers UNE
    indirection d'appel de fonction locale (le cas `summarise(q_ok)`)."""
    fonctions = {n.name: n for n in tree.body
                 if isinstance(n, ast.FunctionDef)}
    statuts = {n: _statuts_locaux(f) for n, f in fonctions.items()}

    graine = {n: {} for n in fonctions}
    for nom_appelant, appelant in fonctions.items():
        for n in ast.walk(appelant):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in fonctions and n.args):
                continue
            params_callee = [a.arg for a in fonctions[n.func.id].args.args]
            if not params_callee:
                continue
            arg0 = n.args[0]
            st = (statuts[nom_appelant].get(arg0.id)
                  if isinstance(arg0, ast.Name) else None)
            if st is None:
                continue
            prev = graine[n.func.id].get(params_callee[0])
            graine[n.func.id][params_callee[0]] = (
                st if prev in (None, st) else "mixed")

    for nom, f in fonctions.items():
        g = {k: v for k, v in graine[nom].items() if v != "mixed"}
        if g:
            statuts[nom] = _statuts_locaux(f, graine_params=g)

    resultats = []
    for nom, f in fonctions.items():
        st = statuts[nom]
        for n in ast.walk(f):
            if not (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr in _REDUCTEURS
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "np"
                    and n.args):
                continue
            noms = {c.id for c in ast.walk(n.args[0])
                    if isinstance(c, ast.Name)}
            statuts_vus = {st[x] for x in noms if x in st}
            if not statuts_vus:
                continue  # rien a voir avec des tirages
            resultats.append(
                (n, n.lineno,
                 "raw" if "raw" in statuts_vus else "filtered"))
    return resultats


@pytest.mark.parametrize("script", TASK_SCRIPTS)
def test_aggregations_over_runs_filter_completed(script):
    """Une moyenne prise sur une liste de tirages doit exclure les tirages
    avortes de sa provenance.

    C'est le defaut qui a fait publier a T16 une moyenne de 0.3328 pour
    `rotor` la ou les tirages valides donnaient 0.1473."""
    lignes = _source(script).splitlines()
    for appel, i, statut in _agregations_avec_statut(_tree(script)):
        assert statut != "raw", (
            f"{script}:{i} agrege des tirages sans filtrer les avortees:\n"
            f"    {lignes[i - 1].strip()}")


def test_the_aggregation_sweep_is_not_empty():
    """Un balayage qui ne selectionne rien doit crier.

    D-128 (suite) : la premiere passe de D-128 corrigeait le balayage vide
    d'origine mais n'en voyait encore que 2 sites sur 20 reels -- toujours
    aveugle a `summarise(q_ok)` et a ses semblables, ou l'agregation vit
    dans une fonction DIFFERENTE de celle qui filtre. Mesure a cette passe :
    **20** agregations tracees et confirmees filtrees, sur 5 scripts. Le
    nombre peut monter legitimement ; il ne doit pas retomber a 2 (le
    plancher de la premiere passe) ni a 0.
    """
    total = sum(1 for s in TASK_SCRIPTS
                for _, _, st in _agregations_avec_statut(_tree(s))
                if st == "filtered")
    assert total >= 20, (
        f"le balayage des agregations sur tirages n'examine plus que "
        f"{total} site(s) confirmes filtres sur {len(TASK_SCRIPTS)} "
        "scripts (mesure a l'ecriture : 20) — il est redevenu aveugle a "
        "une partie du motif reel du depot")


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
