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
    """Un balayage vide sort en vert et ne prouve rien.

    D-166 : le plancher a 40 ne detectait plus rien -- 153 fichiers mesures
    a `bfe4c46` (18 aout 2026), le balayage pouvait fondre de 112 fichiers
    sur 153 sans que ce test ne bouge."""
    assert len(_test_files()) >= 153, (
        f"{len(_test_files())} fichiers trouves ; 153 mesures a `bfe4c46` "
        "(18 aout 2026) : le balayage a retreci, il ne prouve plus ce "
        "qu'il prouvait")


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
    """Une liste d'exemptions qui pourrit est pire que pas de liste.

    D-163 : ce controle ne verifiait que l'existence du FICHIER. Or ce que
    l'exemption designe n'est pas un fichier, c'est une FONCTION de test —
    et ce qu'elle fait n'est pas « exister », c'est « supprimer un
    signalement ». Un fichier peut survivre a la fonction qu'il portait ;
    une fonction peut gagner une assertion sans que son exemption parte
    avec. Dans les deux cas l'entree devient une permission dormante,
    accordee d'avance a la prochaine fonction qui prendra ce nom. Meme forme
    que D-161, ou 2 exemptions sur 4 etaient deja mortes sans que le
    controle de peremption le voie.

    Les trois criteres, du plus faible au plus fort : le fichier existe, la
    fonction existe, et elle serait ENCORE signalee sans son exemption.
    """
    for cle, raison in SANS_ASSERTION_LEGITIMES.items():
        assert len(raison) > 30, f"{cle} exempte sans raison lisible"
        fichier, nom = cle.split("::")
        chemin = os.path.join(_REPO_ROOT, fichier)
        assert os.path.exists(chemin), (
            f"{fichier} n'existe plus — retirer son exemption")
        arbre = ast.parse(open(chemin, encoding="utf-8").read())
        fns = [n for n in ast.walk(arbre)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == nom]
        assert fns, (
            f"{cle} : le fichier existe mais la fonction exemptee n'y est "
            "plus. L'exemption ne protege plus rien — elle attend la "
            "prochaine fonction qui prendra ce nom. La retirer.")
        assert not _porte_une_verification(fns[0], _helpers_verificateurs(arbre)), (
            f"{cle} porte desormais une verification : son exemption ne "
            "supprime plus aucun signalement. La retirer, sinon elle "
            "couvrira le jour ou cette verification disparaîtra.")


def test_le_controle_de_peremption_peut_echouer():
    """Epingle D-163 : sur quelle entree l'ancien controle echouait-il ?

    Sur aucune de celles qui comptent. Il faisait `os.path.exists(fichier)`
    — vrai pour les trois entrees, et il le resterait apres la suppression
    ou le renommage de la fonction exemptee, comme apres l'ajout d'une
    assertion dedans. Les trois criteres sont mesures ici sur un arbre
    fabrique, pour que chacun puisse rendre faux.
    """
    src = ("def test_sans_assertion():\n"
           "    calcule()\n\n\n"
           "def test_avec_assertion():\n"
           "    assert calcule()\n")
    arbre = ast.parse(src)
    helpers = _helpers_verificateurs(arbre)
    par_nom = {n.name: n for n in ast.walk(arbre)
               if isinstance(n, ast.FunctionDef)}

    #  le critere qui mord : une fonction qui a GAGNE une verification
    assert not _porte_une_verification(par_nom["test_sans_assertion"], helpers)
    assert _porte_une_verification(par_nom["test_avec_assertion"], helpers), (
        "le critere de peremption ne distingue plus une fonction verifiante "
        "d'une fonction muette : D-163 est rouvert")
    #  le critere le plus faible : une fonction ABSENTE de l'arbre
    assert "test_disparu_d163" not in par_nom


# ══════════════════════════════════════════════════════════════════════
#  D-154 — le meme controle, sur les 349 imports internes qu'il ignorait
# ══════════════════════════════════════════════════════════════════════
#
#  `test_every_cross_test_import_resolves` ci-dessus existe parce qu'un
#  fichier deplace ne se voit qu'a l'EXECUTION. Il ne regardait pourtant que
#  les imports dont le module commence par `tests.` : **3 sites sur 480**
#  qui designent un module du depot (1347 sites d'import en tout).
#
#  Les 477 autres nomment `src/`, `study/` et `figures/` par leur nom de
#  module (`Simulation`, `VQA`, `pipeline`, `train_hyperparams`, `config`,
#  `fig_utils`...), et **381 sites sont ecrits dans le corps d'une fonction
#  ou d'une fixture** — exactement la position qui rend l'echec invisible a
#  `--collect-only`.
#
#  Mesure du 18 aout 2026, avant correction. Trois modules de `src/`
#  renommes (`analyze_hyperparams`, `recompute_lambda_scores`,
#  `compare_rotor_budget`), sur les 7 fichiers qui les emploient :
#
#      arbre sain                     62 passed
#      arbre mute      6 failed, 11 passed, 45 SKIPPED
#      ce fichier                    158 passed   <- vert
#
#  **45 tests disparaissent en silence** : les fixtures concernees passent
#  par `pytest.importorskip("<module de src/>")`, qui transforme un fichier
#  absent en `skip`, pas en echec. Un module deplace (meme nom de fichier,
#  autre dossier) donne la meme chose : 3 failed, 1 passed, 19 skipped, et
#  ce fichier toujours **158 passed**.
#
#  Le correctif ne touche pas ces 14 `importorskip` : leur raison d'etre est
#  legitime (`analyze_hyperparams` importe `optuna`, absent d'un
#  environnement minimal). Il rend seulement le GARDE capable de rougir.

import importlib.machinery  # noqa: E402

#: Les racines d'import de la suite. Les dix premieres sont celles que
#: `tests/conftest.py` pose sur `sys.path` — `test_les_racines_declarees_sont_bien_celles_de_la_suite`
#: verifie qu'elles y sont encore. `figures/` s'y ajoute : il est pose par
#: les fichiers de test qui en ont besoin (`test_pareto_panel.py`,
#: `test_pareto_frontier_retracted_ratio.py`), donc apres l'import de
#: celui-ci — on ne peut pas le lire dans `sys.path`, il se declare.
_RACINES_CONFTEST = (
    os.path.join(_REPO_ROOT, "src"),
    _REPO_ROOT,
    os.path.join(_REPO_ROOT, "study", "pipeline"),
    os.path.join(_REPO_ROOT, "study", "common"),
    os.path.join(_REPO_ROOT, "study", "h0_selection"),
    os.path.join(_REPO_ROOT, "study", "h1_solver"),
    os.path.join(_REPO_ROOT, "study", "h2b_prediction"),
    os.path.join(_REPO_ROOT, "study", "h3_representation"),
    os.path.join(_REPO_ROOT, "study", "h4_transfer"),
    os.path.join(_REPO_ROOT, "study", "closed_loop"),
)
#: Les deux racines suivantes sont posees par les fichiers de test qui en
#: ont besoin : `figures/` (`test_pareto_panel.py:44`) et
#: `figures/v1_legacy/` (`test_fig0_classical_truncation.py:29` et sept
#: autres). Une racine oubliee ici ne rend rien silencieux : les modules
#: qu'elle porte cessent de resoudre et le balayage ci-dessous ROUGIT — son
#: message dit alors de declarer la racine plutot que de deplacer un
#: fichier. L'oubli se voit, il ne se tait pas.
_RACINES = _RACINES_CONFTEST + (
    os.path.join(_REPO_ROOT, "figures"),
    os.path.join(_REPO_ROOT, "figures", "v1_legacy"),
)


def _sites_d_import(path):
    """(nom de module, ligne, sous_fonction) pour tout ce que le fichier
    importe — `import x`, `from x import y`, et le nom litteral passe a
    `pytest.importorskip(...)`.

    `importorskip` compte : c'est par lui que 45 tests se sont tus a la
    mesure ci-dessus. Un import relatif (`from . import x`) est rendu
    absolu contre le paquet du fichier ; aucun n'existe aujourd'hui dans
    `tests/`, et le test de plancher le dirait s'il en apparaissait un qui
    ne resout pas."""
    arbre = ast.parse(open(path, encoding="utf-8").read())
    dans_fonction = set()
    for fn in ast.walk(arbre):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for x in ast.walk(fn):
                dans_fonction.add(id(x))
    out = []
    for n in ast.walk(arbre):
        sous = id(n) in dans_fonction
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            out.append((n.module, n.lineno, sous))
        elif isinstance(n, ast.Import):
            for a in n.names:
                out.append((a.name, n.lineno, sous))
        elif (isinstance(n, ast.Call)
              and getattr(n.func, "attr", "") == "importorskip"
              and n.args and isinstance(n.args[0], ast.Constant)
              and isinstance(n.args[0].value, str)):
            out.append((n.args[0].value, n.lineno, sous))
    return out


def _resout(nom, racines):
    """Le module existe-t-il sous l'une des racines ?

    Ecrit a la main plutot qu'avec `find_spec` : `find_spec` IMPORTE les
    paquets parents, et l'import de `Simulation` ou de `pipeline` coute des
    secondes et peut lever pour une dependance absente — on mesurerait
    alors l'environnement, pas l'emplacement des fichiers.

    `study/` n'a pas d'`__init__.py` : un dossier nu est un paquet
    d'espace de noms, et `from study.common import qaoa_inputs` passe par
    lui. Le refuser rendrait ce test rouge sur un depot sain."""
    courant = list(racines)
    morceaux = nom.split(".")
    for i, part in enumerate(morceaux):
        dernier = i == len(morceaux) - 1
        suivant = None
        for r in courant:
            dossier = os.path.join(r, part)
            if os.path.isdir(dossier):
                suivant = [dossier]
                break
            if dernier and os.path.isfile(dossier + ".py"):
                suivant = []
                break
        if suivant is None:
            return False
        courant = suivant
    return True


def _est_externe(sommet):
    """Vrai si le nom vient de la bibliotheque standard ou d'un paquet
    installe — donc pas du depot. Les racines du depot sont RETIREES du
    chemin de recherche, sans quoi tout module local passerait pour
    externe."""
    if sommet in sys.stdlib_module_names:
        return True
    abs_racines = {os.path.abspath(r) for r in _RACINES}
    dehors = [p for p in sys.path
              if p and os.path.abspath(p) not in abs_racines]
    try:
        return importlib.machinery.PathFinder.find_spec(sommet, dehors) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _tous_les_sites():
    for p in _test_files():
        for nom, ligne, sous in _sites_d_import(p):
            yield p, nom, ligne, sous


def test_les_racines_declarees_sont_bien_celles_de_la_suite():
    """Deux chemins censes coincider : les racines ecrites ici et celles que
    `conftest.py` pose. Si conftest en retire une, ce test le dit — sinon le
    balayage ci-dessous declarerait « externe » un module devenu
    introuvable, et redeviendrait incapable de rougir."""
    sur_le_chemin = {os.path.abspath(p) for p in sys.path}
    manquantes = [r for r in _RACINES_CONFTEST
                  if os.path.isdir(r) and os.path.abspath(r) not in sur_le_chemin]
    assert not manquantes, (
        "racines declarees ici mais absentes de sys.path a l'execution — "
        f"conftest.py a change : {manquantes}")


def test_tout_import_interne_de_la_suite_resout():
    """Le controle que D-154 ouvre : un module du depot designe par un test
    doit exister la ou la suite le cherche.

    Rouge sur un fichier DEPLACE comme sur un fichier RENOMME — les deux
    laissaient l'ancien controle a 158 passed."""
    casses = []
    for p, nom, ligne, sous in _tous_les_sites():
        sommet = nom.split(".")[0]
        if _est_externe(sommet):
            continue
        if not _resout(nom, _RACINES):
            ou = "dans une fonction" if sous else "au niveau module"
            casses.append(f"{os.path.relpath(p, _REPO_ROOT)}:{ligne} ({ou}) : {nom}")
    assert not casses, (
        "module(s) introuvable(s) sous les racines de la suite. Si le nom "
        "existe encore ailleurs dans le depot, le fichier a ete deplace ; "
        "sinon c'est une dependance absente de l'environnement — dans les "
        "deux cas les tests concernes se taisent (skip) au lieu "
        f"d'echouer :\n  " + "\n  ".join(casses))


def test_la_partition_local_externe_n_est_pas_degeneree():
    """`_est_externe` decide qui est controle et qui ne l'est pas : s'il
    rendait `True` partout, le balayage precedent serait vert quoi qu'il
    arrive — un balayage vide qui ne crie pas.

    Mesure du 18 aout 2026 : 104 noms distincts, dont **67 locaux**."""
    vus = {nom.split(".")[0] for _p, nom, _l, _s in _tous_les_sites()}
    locaux = {s for s in vus if not _est_externe(s)}
    assert len(locaux) >= 60, (
        f"{len(locaux)} noms locaux sur {len(vus)} — mesure du 18 aout : "
        "67 sur 104. La partition s'est degradee, le balayage ne controle "
        "presque plus rien")
    #  Deux temoins nommes, un de chaque cote : le jour ou l'un bascule,
    #  c'est la partition qui est fausse, pas le depot.
    assert not _est_externe("Simulation"), (
        "`Simulation` est un paquet de src/, pas une dependance installee")
    assert _est_externe("numpy") and _est_externe("os"), (
        "une dependance installee et un module de la bibliotheque standard "
        "doivent rester hors du controle")


def test_le_balayage_couvre_bien_ce_qu_il_annonce():
    """Epingle l'ancien perimetre : **3 sites sur 480**.

    Si quelqu'un restreint de nouveau le balayage aux modules `tests.`, ce
    test tombe. Un balayage vide doit crier — y compris celui-ci.

    Mesure du 18 aout 2026 : 153 fichiers, 1347 sites d'import, dont 480
    designent un module du depot et 381 sont ecrits dans le corps d'une
    fonction."""
    sites = [(nom, sous) for _p, nom, _l, sous in _tous_les_sites()]
    locaux = [(n, s) for n, s in sites if not _est_externe(n.split(".")[0])]
    dans_fonction = [n for n, s in sites if s]
    assert len(_test_files()) >= 153, (
        f"{len(_test_files())} fichiers balayes ; 153 mesures a `bfe4c46` "
        "(18 aout 2026, meme quantite que test_there_are_test_files_to_check, "
        "D-166)")
    assert len(sites) >= 1200, (
        f"{len(sites)} sites d'import balayes — mesure du 18 aout : 1347. "
        "Le parseur a perdu des formes")
    assert len(locaux) >= 400, (
        f"seulement {len(locaux)} sites internes controles — mesure du "
        "18 aout : 480. L'ancien perimetre (3 sites, les seuls `tests.`) "
        "est de retour")
    assert len(dans_fonction) >= 300, (
        f"{len(dans_fonction)} imports dans le corps d'une fonction — "
        "mesure du 18 aout : 381. C'est la position qui echappe a "
        "`--collect-only`, et la raison d'etre de ce fichier")


def test_le_resolveur_voit_un_fichier_deplace(tmp_path):
    """Sur quelle entree ce balayage echouerait-il ? Celle-ci.

    Deux arbres identiques a un deplacement pres — le champ qui SEPARE.
    Sans lui, `_resout` pourrait rendre `True` partout et tout ce qui
    precede serait vert quoi qu'il arrive."""
    racine = tmp_path / "r"
    (racine / "src").mkdir(parents=True)
    (racine / "src" / "analyze.py").write_text("X = 1\n")
    (racine / "src" / "paquet").mkdir()
    (racine / "src" / "paquet" / "__init__.py").write_text("")
    (racine / "src" / "paquet" / "grille.py").write_text("Y = 2\n")
    (racine / "espace_de_noms").mkdir()          # sans __init__.py, comme study/
    (racine / "espace_de_noms" / "m.py").write_text("Z = 3\n")
    racines = (str(racine / "src"), str(racine))

    assert _resout("analyze", racines)
    assert _resout("paquet.grille", racines)
    assert _resout("espace_de_noms.m", racines), (
        "un dossier sans __init__.py est un paquet d'espace de noms : "
        "study/ en est un, et le refuser rendrait la suite rouge a tort")
    assert not _resout("absent", racines)
    assert not _resout("paquet.absent", racines)

    (racine / "src" / "outils").mkdir()
    os.rename(str(racine / "src" / "analyze.py"),
              str(racine / "src" / "outils" / "analyze.py"))
    assert not _resout("analyze", racines), (
        "un fichier deplace dans un sous-dossier doit cesser de resoudre — "
        "c'est le defaut que D-154 ferme")


def test_importorskip_est_bien_dans_le_balayage():
    """`pytest.importorskip("<module du depot>")` est la forme par laquelle
    45 tests se sont tus. Elle doit etre vue par le balayage, pas seulement
    les `import`.

    Mesure du 18 aout : 15 sites, dont 14 nomment un module du depot."""
    src = ('import pytest\n'
           'mod = pytest.importorskip("analyze_hyperparams")\n'
           'def f():\n'
           '    autre = pytest.importorskip("recompute_lambda_scores")\n')
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        tmp = fh.name
    try:
        noms = [n for n, _l, _s in _sites_d_import(tmp)]
    finally:
        os.unlink(tmp)
    assert "analyze_hyperparams" in noms and "recompute_lambda_scores" in noms, (
        f"les noms passes a importorskip echappent au balayage : {noms}")

    reels = 0
    for p in _test_files():
        arbre = ast.parse(open(p, encoding="utf-8").read())
        reels += sum(1 for n in ast.walk(arbre)
                     if isinstance(n, ast.Call)
                     and getattr(n.func, "attr", "") == "importorskip")
    assert reels >= 14, (
        f"{reels} appels a importorskip balayes — mesure du 18 aout : 15. "
        "Un balayage vide doit crier")
