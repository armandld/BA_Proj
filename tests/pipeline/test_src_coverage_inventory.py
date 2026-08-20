"""Aucun module de `src/` ne doit rester sans test.

Ce fichier fait deux choses que les autres ne font pas :

  1. il importe CHAQUE module de `src/`, ce qui attrape les erreurs de
     syntaxe, les imports circulaires et les dépendances manquantes — un
     module cassé qui n'est jamais importé par la suite de tests reste
     invisible jusqu'au jour où une campagne s'en sert ;

  2. il tient l'inventaire de ce qui est testé et de ce qui ne l'est pas,
     et **échoue quand un nouveau module apparaît sans entrée**. Sans ce
     garde-fou, la couverture se dégrade silencieusement à chaque ajout.

L'inventaire distingue trois catégories, et chacune doit être justifiée :
`COVERED` (des tests visent ses fonctions), `ENTRY_POINTS` (scripts de
lancement, testés par leur `--help` et leur contrat d'arguments) et
`EXCLUDED` (visualisation et pilotes d'entraînement, avec la raison).
"""

import ast
import importlib
import os
import re
import subprocess
import sys

import pytest



def _repo_root():
    """Racine du depot : on remonte jusqu'au dossier qui contient `src/`.

    Un calcul par `dirname` repete depend de la profondeur du fichier et
    casse au premier deplacement — souvent en silence, en pointant vers un
    chemin qui n'existe pas.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_REPO_ROOT = _repo_root()
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


#  Modules dont les fonctions sont visées par des tests dédiés.
COVERED = {
    "Simulation/grid.py",
    "Simulation/solver.py",
    "Simulation/PhysToAngle.py",
    "Simulation/HamiltParams.py",
    "Simulation/HamiltParams_v2.py",
    "Simulation/RescaleArrays.py",
    "Simulation/refinement.py",
    "Simulation/utils.py",
    "Simulation/pre_compute_dns.py",
    "VQA/cost_hamiltonian.py",
    "VQA/init_qbits_state.py",
    "VQA/mapping.py",
    "VQA/postprocess.py",
    "VQA/execute.py",
    "VQA/optimize.py",
    "VQA/runtime.py",
    "pipeline.py",
    "hyperparams_loader.py",
    "visual.py",
    #  D-162 : etait declare ENTRY_POINTS alors qu'il ne porte aucun bloc
    #  `__main__` — c'est un module d'UNE fonction, importee par
    #  `Simulation/refinement.py` (le chemin de decision deploye),
    #  `compare_rotor_budget.py` et deux figures. Cinq fichiers de
    #  `tests/quantum/` l'importent et l'appellent : sa place est ici.
    "call_vqa_shell.py",
}

#  Scripts lancés en ligne de commande : on teste qu'ils s'importent et
#  que leur interface d'arguments tient.
#
#  D-162 — cette categorie n'est PAS une trappe de sortie. Elle dispense de
#  l'import propre et, avant cette correction, du controle « nomme par la
#  suite » : deux vrais controles, en echange desquels elle n'exigeait
#  rien. Mesure du 18 aout 2026 : **19 des 19 modules de `COVERED`**
#  passaient tels quels son unique controle — n'importe lequel pouvait y
#  etre deplace sans qu'un test ne bouge. Desormais un module declare ici
#  doit PORTER le bloc `__main__` dont le controle porte le nom, et reste
#  soumis au « nomme par la suite ».
ENTRY_POINTS = {
    "train_hyperparams.py",
    "compare_rotor_budget.py",
    "analyze_hyperparams.py",
    "recompute_lambda_scores.py",
}

#  Exclus, avec la raison. Toute entrée ici doit être justifiable.
EXCLUDED = {
    #  D-68 : `help_visual.py` reste exclu, mais la raison a changé. L'ancienne
    #  — « aucune valeur numérique produite » — était vraie et insuffisante :
    #  aucune de ces fonctions ne rend de valeur, et pourtant `plot_amr_state`
    #  annonçait ses axes à l'envers pendant toute la vie du dépôt. Une figure
    #  porte une convention, et une convention peut être fausse.
    #  `visual.py` est passé dans COVERED : tests/pipeline/test_amr_figure_axes.py
    #  vise `plot_amr_state`, la seule de ses fonctions qui s'exécute.
    "help_visual.py": (
        "tracé matplotlib ; aucune de ses 5 fonctions n'a d'appelant — "
        "`visualize_vqa_step` est importée par refinement.py sans être "
        "appelée. Voir COUVERTURE.md §1a quater"
    ),
}


def _modules():
    out = []
    for dirpath, _dirs, names in os.walk(_SRC):
        if "__pycache__" in dirpath:
            continue
        for n in sorted(names):
            if n.endswith(".py") and n != "__init__.py":
                out.append(os.path.relpath(os.path.join(dirpath, n), _SRC))
    return sorted(p.replace(os.sep, "/") for p in out)


# ═══════════════════════════════════════════════════════════════════════
#  1. L'inventaire est complet
# ═══════════════════════════════════════════════════════════════════════

def test_every_module_is_accounted_for():
    """Un nouveau module sans entree fait tomber ce test — c'est le but."""
    declared = COVERED | ENTRY_POINTS | set(EXCLUDED)
    actual = set(_modules())
    assert actual == declared, (
        "inventaire desynchronise.\n"
        f"  modules non declares : {sorted(actual - declared)}\n"
        f"  declares mais absents : {sorted(declared - actual)}")


def test_the_three_categories_do_not_overlap():
    assert not (COVERED & ENTRY_POINTS)
    assert not (COVERED & set(EXCLUDED))
    assert not (ENTRY_POINTS & set(EXCLUDED))


def test_every_exclusion_carries_a_reason():
    for mod, reason in EXCLUDED.items():
        assert isinstance(reason, str) and len(reason) > 20, (
            f"{mod} est exclu sans justification lisible")


def test_the_exclusion_list_stays_small():
    """Exclure est une exception, pas une strategie."""
    assert len(EXCLUDED) <= 3, (
        f"{len(EXCLUDED)} modules exclus : la couverture se degrade")


# ═══════════════════════════════════════════════════════════════════════
#  2. Tout s'importe
# ═══════════════════════════════════════════════════════════════════════

def _import_name(rel):
    return rel[:-3].replace("/", ".")


@pytest.mark.parametrize("rel", sorted(COVERED | set(EXCLUDED)))
def test_every_module_imports_cleanly(rel):
    """Un import cassé reste invisible jusqu'a ce qu'une campagne s'en serve."""
    import matplotlib
    matplotlib.use("Agg")
    importlib.import_module(_import_name(rel))


@pytest.mark.parametrize("rel", sorted(ENTRY_POINTS))
def test_every_entry_point_parses(rel):
    """Les pilotes sont lourds a importer : on verifie au moins la syntaxe."""
    src = open(os.path.join(_SRC, rel), encoding="utf-8").read()
    ast.parse(src, filename=rel)


def _porte_un_bloc_main(src):
    """Vrai si le module porte un `if __name__ == '__main__':` au niveau
    module. Interroge l'AST, pas le texte : une chaîne `"__main__"` dans un
    commentaire ou un message ne compte pas."""
    for n in ast.parse(src).body:
        if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test):
            return True
    return False


@pytest.mark.parametrize("rel", sorted(ENTRY_POINTS))
def test_every_entry_point_guards_its_main(rel):
    """Un script sans `if __name__ == '__main__'` s'execute a l'import.

    Importer un module d'entrainement lancerait alors une campagne entiere.

    D-162 : ce test ne verifiait QUE l'absence de travail au niveau module.
    Il ne verifiait pas la presence du bloc `__main__` dont il porte le nom
    et que sa docstring annonce — or un module qui n'en a aucun le passe
    trivialement. Mesure du 18 aout 2026 : `call_vqa_shell.py` etait declare
    ENTRY_POINTS, ne porte aucun bloc `__main__`, et ce test etait vert
    dessus. C'est la question 2 — ce que la fonction PROMET, verifie point
    par point.
    """
    #  Configuration inoffensive tolérée a l'import : choix du backend
    #  graphique, niveau de journalisation, tampon de sortie.
    ALLOWED = ("matplotlib.use", "logging.set_verbosity",
               "logging.basicConfig", "stdout.reconfigure",
               "warnings.filterwarnings", "set_start_method")

    src = open(os.path.join(_SRC, rel), encoding="utf-8").read()
    tree = ast.parse(src)
    offenders = []
    for n in tree.body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            text = ast.unparse(n)
            if not any(a in text for a in ALLOWED):
                offenders.append(text[:70])
    assert not offenders, (
        f"{rel} execute du travail au niveau module — l'importer suffirait "
        f"a le declencher : {offenders}")
    assert _porte_un_bloc_main(src), (
        f"{rel} est declare ENTRY_POINTS mais ne porte aucun bloc "
        "`if __name__ == '__main__':` — ce n'est pas un point d'entree. "
        "Cette categorie dispense de l'import propre : un module de "
        "bibliotheque range ici perd son controle sans rien gagner. Le "
        "deplacer vers `COVERED` (D-162).")


# ═══════════════════════════════════════════════════════════════════════
#  3. Chaque module couvert est réellement visé par un test
# ═══════════════════════════════════════════════════════════════════════

#  D-159 — pourquoi ce n'est plus une recherche de chaine dans le texte.
#
#  `_test_corpus` concatenait le TEXTE de tous les fichiers de `tests/` — y
#  compris CELUI-CI. Les deux tests ci-dessous cherchaient un nom dans ce
#  texte ; or c'est ici que les noms sont declares : `"Simulation/grid.py"`
#  dans `COVERED`, `"project_divergence_free"` dans `critical`. Chaque
#  recherche trouvait donc sa propre declaration, et **les deux tests
#  etaient incapables d'echouer**.
#
#  Mesure du 18 aout 2026 : un module neuf `src/Simulation/zzz_untested.py`,
#  ajoute a `COVERED` ET a `critical` avec une fonction qu'aucun test ne
#  nomme nulle part — **102 passed**. Le fichier dont la docstring dit
#  « echoue quand un nouveau module apparait sans entree » acceptait une
#  entree qui ne couvrait rien.
#
#  Deux corrections, et la seconde est la vraie : ce fichier sort du
#  corpus, et le corpus n'est plus du texte mais des IDENTIFIANTS lus dans
#  l'AST des autres tests. Un nom cite dans un commentaire ou dans une
#  phrase de docstring ne compte plus — « l'assertion porte sur le
#  comportement, pas sur le texte du source », retournee contre la suite.
#  Les litteraux de chaine qui sont des identifiants valides comptent, eux :
#  `getattr(mod, "compute_coefficients")` et `monkeypatch.setattr(m, "score", …)`
#  sont des references reelles. Verifie apres coup : les 19 modules de
#  `COVERED` et les 49 fonctions critiques restent tous vus (4531
#  identifiants collectes), donc aucun faux rouge.

_IDENTIFIANTS = None
_MODULES_IMPORTES = None


def _identifiants_du_corpus():
    """Identifiants reellement employes par les AUTRES fichiers de `tests/`."""
    global _IDENTIFIANTS
    if _IDENTIFIANTS is not None:
        return _IDENTIFIANTS
    moi = os.path.abspath(__file__)
    vus = set()
    for dirpath, _dirs, names in os.walk(os.path.join(_REPO_ROOT, "tests")):
        if "__pycache__" in dirpath:
            continue
        for n in names:
            if not n.endswith(".py"):
                continue
            chemin = os.path.join(dirpath, n)
            if os.path.abspath(chemin) == moi:
                continue          # l'inventaire ne se compte pas lui-meme
            try:
                arbre = ast.parse(open(chemin, encoding="utf-8").read())
            except SyntaxError:
                continue
            for x in ast.walk(arbre):
                if isinstance(x, ast.Name):
                    vus.add(x.id)
                elif isinstance(x, ast.Attribute):
                    vus.add(x.attr)
                elif isinstance(x, ast.ImportFrom) and x.module:
                    vus.update(x.module.split("."))
                    vus.update(a.name for a in x.names)
                elif isinstance(x, ast.Import):
                    for a in x.names:
                        vus.update(a.name.split("."))
                elif (isinstance(x, ast.Constant) and isinstance(x.value, str)
                      and x.value.isidentifier()):
                    vus.add(x.value)
    _IDENTIFIANTS = vus
    return vus


def _modules_importes_du_corpus():
    """D-164 : les stems de MODULE reellement IMPORTES par `tests/` — pas
    n'importe quel identifiant.

    `_identifiants_du_corpus` sert deux usages differents avec la meme
    largeur, et un seul des deux le supporte. Pour les FONCTIONS
    (`test_the_public_surface_of_the_physics_path_is_exercised`), un nom
    passe a `getattr`/`monkeypatch.setattr` est une reference reelle — la
    largeur est voulue, D-159 l'a mesuree. Pour les MODULES de `COVERED`,
    le stem lui-meme (`grid`, `solver`, `execute`, `optimize`, `pipeline`…)
    est un mot assez commun pour apparaitre comme attribut SANS AUCUN
    RAPPORT avec le module de `src/` — `study.optimize(objective)` sur un
    objet Optuna rend `optimize` present dans `_identifiants_du_corpus()`
    sans qu'aucun test ne touche `VQA/optimize.py`. Mesure : 5 des 19
    modules de `COVERED` (`grid`, `solver`, `execute`, `optimize`,
    `pipeline`) restent presents dans `_identifiants_du_corpus()` meme
    apres avoir retire TOUS les fichiers qui les importent reellement.

    Le remede ne restreint que la provenance : seuls les stems tires d'un
    `import x` / `from x import y` reel comptent. Les 19 modules de
    `COVERED` s'y retrouvent tous (verifie avant d'ecrire la correction) —
    aucun faux rouge."""
    global _MODULES_IMPORTES
    if _MODULES_IMPORTES is not None:
        return _MODULES_IMPORTES
    moi = os.path.abspath(__file__)
    vus = set()
    for dirpath, _dirs, names in os.walk(os.path.join(_REPO_ROOT, "tests")):
        if "__pycache__" in dirpath:
            continue
        for n in names:
            if not n.endswith(".py"):
                continue
            chemin = os.path.join(dirpath, n)
            if os.path.abspath(chemin) == moi:
                continue          # l'inventaire ne se compte pas lui-meme
            try:
                arbre = ast.parse(open(chemin, encoding="utf-8").read())
            except SyntaxError:
                continue
            vus.update(_stems_importes(arbre))
    _MODULES_IMPORTES = vus
    return vus


def _stems_importes(arbre):
    """Les stems de MODULE d'un arbre AST deja parse — factorise pour que
    D-164 soit testable sur une source synthetique, sans marcher `tests/`."""
    vus = set()
    for x in ast.walk(arbre):
        if isinstance(x, ast.ImportFrom) and x.module:
            vus.update(x.module.split("."))
        elif isinstance(x, ast.Import):
            for a in x.names:
                vus.update(a.name.split("."))
    return vus


@pytest.mark.parametrize("rel", sorted(COVERED))
def test_each_covered_module_is_named_by_the_test_suite(rel):
    """« Couvert » doit vouloir dire qu'un test l'IMPORTE vraiment — pas
    qu'un attribut homonyme, sans rapport, traine ailleurs dans `tests/`
    (D-164).

    D-162 — pourquoi `ENTRY_POINTS` n'entre PAS dans ce controle, et c'est
    une decision mesuree, pas un oubli. La categorie dispense de l'import
    propre et de ce controle-ci : elle etait donc une trappe, et l'y
    soumettre semblait la fermer. Mesure faite avant d'ecrire la
    correction : sous le critere de D-164 (un import REEL, pas un
    homonyme), **3 des 4 pilotes ne sont pas importes** par `tests/` —
    `compare_rotor_budget`, `analyze_hyperparams`,
    `recompute_lambda_scores`. C'est exactement ce que la docstring de
    `test_every_entry_point_parses` annonce : « les pilotes sont lourds a
    importer ». Les y soumettre fabriquerait **3 faux rouges** sur du code
    sain. La trappe est fermee autrement, et sans faux positif : par
    l'assertion de `test_every_entry_point_guards_its_main`, qui exige
    desormais le bloc `__main__` — on ne peut plus garer un module de
    bibliotheque ici.
    """
    stem = os.path.basename(rel)[:-3]
    assert stem in _modules_importes_du_corpus(), (
        f"{rel} est declare couvert mais aucun test ne l'IMPORTE — le nom "
        "n'apparaît dans aucun `import`/`from … import` de `tests/` (ce "
        "fichier exclu). Un attribut homonyme sans rapport (ex. "
        "`study.optimize(...)` d'Optuna pour `VQA/optimize.py`) ne compte "
        "pas : voir D-164")


def test_le_controle_du_bloc_main_peut_echouer():
    """Epingle D-162 : sur quelle entree l'ancien controle echouait-il ?

    Sur aucune. Il ne cherchait que du travail au niveau module ; un module
    sans aucun bloc `__main__` — donc pas un point d'entree du tout — le
    passait. Le champ qui SEPARE : un module d'une seule fonction, ce
    qu'etait exactement `call_vqa_shell.py`.
    """
    ancien = "import os\n\n\ndef f():\n    return 1\n"
    assert not _porte_un_bloc_main(ancien), (
        "un module sans bloc `__main__` est declare comme en portant un : "
        "le controle de D-162 ne mesure plus rien")
    assert _porte_un_bloc_main(ancien + "\n\nif __name__ == '__main__':\n    f()\n")
    #  Une chaîne "__main__" hors d'un `if` de niveau module ne compte pas :
    #  sinon un simple message d'aide suffirait a satisfaire le controle.
    assert not _porte_un_bloc_main('MSG = "lance-moi avec __main__"\n')

    #  Et le module reellement en cause : il ne porte toujours pas de bloc
    #  `__main__`, c'est pourquoi il a quitte ENTRY_POINTS.
    src = open(os.path.join(_SRC, "call_vqa_shell.py"), encoding="utf-8").read()
    assert not _porte_un_bloc_main(src), (
        "call_vqa_shell.py porte desormais un bloc `__main__` : il peut "
        "revenir dans ENTRY_POINTS, et cet epinglage doit etre remesure")


def test_the_public_surface_of_the_physics_path_is_exercised():
    """Les fonctions du chemin physique doivent apparaitre dans les tests.

    On se limite au chemin qui produit des nombres publies — pas aux
    pilotes ni aux traces.
    """
    vus = _identifiants_du_corpus()
    critical = {
        "Simulation/grid.py": ["project_divergence_free", "_compute_q_criterion",
                               "grad", "div", "laplacian", "smooth_field",
                               "extract_patch_data", "curl_z", "divergence"],
        "Simulation/solver.py": ["_fd_grad", "_fd_laplacian", "_compute_rhs_fd",
                                 "_rk4_step", "_rk2_step", "adapt_dt",
                                 "check_cfl", "is_diverged", "step_full",
                                 "step_layered", "get_fluxes",
                                 "_downsample_local", "_upsample_local",
                                 "_upsample_global"],
        "Simulation/PhysToAngle.py": ["classical_score", "compute_stress_flux",
                                      "_lohner_estimator", "map_to_angles"],
        "Simulation/HamiltParams.py": ["_f_gate", "_g_rot", "_g_strain",
                                       "_g_mag", "_michelson_relu",
                                       "_threshold_contrast",
                                       "_compute_det_jacobian_B",
                                       "physical_score", "compute_coefficients"],
        "Simulation/RescaleArrays.py": ["_maxabs_pool_2d", "_maxabs_pool_1d",
                                        "_resize_padded_bilinear",
                                        "_resize_padded_maxpool",
                                        "get_adaptive_flux", "_process_score"],
        "VQA/cost_hamiltonian.py": ["create_bounded_hamiltonian",
                                    "create_period_hamiltonian",
                                    "get_expected_Z", "NullHamiltonianError",
                                    "COEFF_MIN"],
        "pipeline.py": ["score", "instability_weight_map",
                        "weighted_relative_error"],
        "hyperparams_loader.py": ["load_hyperparams"],
    }
    missing = []
    for mod, names in critical.items():
        for name in names:
            if name not in vus:
                missing.append(f"{mod}::{name}")
    assert not missing, (
        f"jamais employees par un test (ce fichier exclu) : {missing}")


# ═══════════════════════════════════════════════════════════════════════
#  4. Hygiène du code source
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("rel", sorted(COVERED | ENTRY_POINTS | set(EXCLUDED)))
def test_no_module_swallows_exceptions_silently(rel):
    """`except: pass` et `except Exception: pass` cachent les pannes.

    Un filet qui ne journalise rien rend une erreur de programmation
    indiscernable d'un fonctionnement normal — le motif que ce depot
    traque.
    """
    src = open(os.path.join(_SRC, rel), encoding="utf-8").read()
    tree = ast.parse(src)

    def _is_broad(handler):
        """`except:` nu ou `except Exception:` — tout est avale."""
        if handler.type is None:
            return True
        return (isinstance(handler.type, ast.Name)
                and handler.type.id in ("Exception", "BaseException"))

    offenders = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.ExceptHandler)
                 and _is_broad(n)
                 and all(isinstance(b, ast.Pass) for b in n.body)]
    #  Une capture ETROITE et documentee reste licite : attraper un
    #  KeyError attendu n'a rien d'un filet aveugle. C'est la capture
    #  LARGE qui rend une erreur de programmation indiscernable d'un
    #  fonctionnement normal.
    assert not offenders, (
        f"{rel} : capture large suivie de `pass` aux lignes {offenders}")


@pytest.mark.parametrize("rel", sorted(COVERED))
def test_no_module_defines_the_same_constant_twice(rel):
    """Une constante redefinie localement masque celle du module.

    C'etait le cas de DIVERGENCE_PENALTY dans pipeline.py : quatre
    definitions, dont trois masquaient la premiere.
    """
    src = open(os.path.join(_SRC, rel), encoding="utf-8").read()
    names = re.findall(r"^\s*([A-Z][A-Z0-9_]{3,})\s*=", src, flags=re.M)
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"{rel} : constantes redefinies {sorted(dupes)}"


def test_le_corpus_ne_compte_ni_lui_meme_ni_les_commentaires(tmp_path):
    """Epingle D-159 : sur quelle entree l'ancien critere echouait-il ?

    Sur aucune — c'est le defaut. L'ancien cherchait une chaine dans le
    TEXTE de tous les fichiers de `tests/`, ce fichier compris, ou les noms
    sont declares. Ce test verifie les deux moities du remede : le fichier
    s'exclut, et un nom cite en prose ne compte pas."""
    corpus = _identifiants_du_corpus()

    #  1. Ce fichier ne se compte pas. `zzz_temoin_d159` n'apparaît QUE
    #     dans la ligne ci-dessous, dans ce fichier : il ne doit pas etre vu.
    temoin = "zzz_temoin_d159"
    assert temoin not in corpus, (
        "ce fichier est de nouveau dans son propre corpus : tout nom qu'il "
        "declare se validerait lui-meme, et les deux tests ci-dessus "
        "redeviendraient incapables d'echouer (D-159)")

    #  2. Un nom cite en commentaire ou en prose de docstring ne compte pas ;
    #     un nom employe comme identifiant, ou passe a getattr, compte.
    src = (
        '"""Un module qui parle de prose_seulement sans jamais l\'employer."""\n'
        "# et un commentaire qui cite commentaire_seulement\n"
        "from Simulation.grid import curl_z\n"
        "def test_x(monkeypatch):\n"
        "    monkeypatch.setattr(mod, 'chaine_identifiant', None)\n"
        "    return curl_z(1) + objet.attribut_employe\n"
    )
    arbre = ast.parse(src)
    vus = set()
    for x in ast.walk(arbre):
        if isinstance(x, ast.Name):
            vus.add(x.id)
        elif isinstance(x, ast.Attribute):
            vus.add(x.attr)
        elif isinstance(x, ast.ImportFrom) and x.module:
            vus.update(x.module.split("."))
            vus.update(a.name for a in x.names)
        elif (isinstance(x, ast.Constant) and isinstance(x.value, str)
              and x.value.isidentifier()):
            vus.add(x.value)
    assert {"curl_z", "grid", "Simulation", "attribut_employe",
            "chaine_identifiant"} <= vus
    assert "commentaire_seulement" not in vus, (
        "un nom cite en commentaire ne prouve aucune couverture")
    assert "prose_seulement" not in vus, (
        "un nom cite dans une phrase de docstring ne prouve aucune "
        "couverture — la docstring n'est pas un identifiant valide, donc "
        "elle ne doit pas entrer")


def test_le_corpus_dexamen_nest_pas_vide():
    """Un balayage vide doit crier — y compris celui-ci.

    Mesure du 18 aout 2026 : 4531 identifiants collectes hors ce fichier."""
    corpus = _identifiants_du_corpus()
    assert len(corpus) >= 3000, (
        f"{len(corpus)} identifiants collectes dans tests/ — mesure du "
        "18 aout : 4531. Le corpus s'est vide, et les deux tests de "
        "couverture ne prouveraient plus rien")


# ═══════════════════════════════════════════════════════════════════════
#  D-164 — un homonyme d'ATTRIBUT ne doit pas compter pour un IMPORT
# ═══════════════════════════════════════════════════════════════════════
#
#  Mesure, avant correction : `VQA/optimize.py` n'a qu'UN fichier genuin
#  (`tests/quantum/test_vqa_chain_contracts.py`, `from VQA.optimize import
#  optimize`). Le retirer du corpus laisse `"optimize" in
#  _identifiants_du_corpus()` a **True** quand meme — 11 sites
#  `study.optimize(objective, ...)` dans `tests/pipeline/
#  test_train_hyperparams_*.py` appellent `.optimize()` sur un objet
#  Optuna, sans lien avec `VQA/optimize.py`. `test_each_covered_module_is_
#  named_by_the_test_suite["VQA/optimize.py"]` restait donc vert meme sans
#  aucun test qui touche `VQA/optimize.py`.
#
#  Balaye les 19 modules de `COVERED` : 5 sur 19 (`grid`, `solver`,
#  `execute`, `optimize`, `pipeline`) survivent au retrait de TOUS leurs
#  importateurs genuins, `_identifiants_du_corpus()` interroge. Aucun ne
#  survit sur `_modules_importes_du_corpus()` (mesure ci-dessous).

def test_un_homonyme_d_attribut_ne_compte_pas_comme_import():
    """Epingle D-164 : sur quelle entree l'ancien corpus (large,
    `_identifiants_du_corpus`) se laissait-il tromper, la ou le nouveau
    (`_modules_importes_du_corpus`) ne se laisse pas faire ?

    `obj.optimize(...)` SANS AUCUN `import` d'un module `optimize` — la
    forme exacte des 11 sites `study.optimize(...)` d'Optuna."""
    leurre = ast.parse(
        "def test_x():\n"
        "    study = make_study()\n"
        "    study.optimize(objective, n_trials=1)\n"
    )
    #  ANCIEN comportement, rejoue explicitement : un `ast.Attribute` suffit.
    ancien = {x.attr for x in ast.walk(leurre) if isinstance(x, ast.Attribute)}
    assert "optimize" in ancien, (
        "le leurre doit reproduire la forme exacte qui trompait l'ancien "
        "corpus — sinon ce test ne prouve rien sur D-164")

    #  NOUVEAU comportement : aucun `import` dans le leurre, donc rien.
    assert "optimize" not in _stems_importes(leurre), (
        "un attribut homonyme, sans aucun `import` de ce nom, est encore "
        "compte comme une preuve de couverture — D-164 n'est plus corrige")

    #  Le VRAI import, lui, doit continuer a compter — sinon le remede est
    #  plus destructeur que le defaut.
    vrai = ast.parse("from VQA.optimize import optimize\n")
    assert "optimize" in _stems_importes(vrai)


def test_les_19_modules_couverts_survivent_au_remede():
    """Pas de faux rouge : le remede de D-164 ne doit priver aucun des 19
    modules de `COVERED` d'une preuve reelle. Mesure avant d'ecrire la
    correction : les 19 s'y retrouvent."""
    manquants = [rel for rel in COVERED
                 if os.path.basename(rel)[:-3] not in _modules_importes_du_corpus()]
    assert not manquants, (
        f"modules de COVERED sans import genuin qui les nomme : {manquants} "
        "— le remede de D-164 est trop strict")


def test_le_corpus_des_modules_importes_nest_pas_vide():
    """Un balayage vide doit crier — y compris celui-ci.

    D-168 : le plancher a 50 ne detectait plus rien -- 130 stems mesures
    a `51e36ab` (18 aout 2026)."""
    assert len(_modules_importes_du_corpus()) >= 130, (
        f"{len(_modules_importes_du_corpus())} stems de module importes ; "
        "130 mesures a `51e36ab` (18 aout 2026) : le corpus etroit de "
        "D-164 a retreci, et le test de couverture ne prouverait plus rien")
