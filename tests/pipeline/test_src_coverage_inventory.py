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
}

#  Scripts lancés en ligne de commande : on teste qu'ils s'importent et
#  que leur interface d'arguments tient.
ENTRY_POINTS = {
    "train_hyperparams.py",
    "compare_rotor_budget.py",
    "analyze_hyperparams.py",
    "recompute_lambda_scores.py",
    "call_vqa_shell.py",
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


@pytest.mark.parametrize("rel", sorted(ENTRY_POINTS))
def test_every_entry_point_guards_its_main(rel):
    """Un script sans `if __name__ == '__main__'` s'execute a l'import.

    Importer un module d'entrainement lancerait alors une campagne entiere.
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


# ═══════════════════════════════════════════════════════════════════════
#  3. Chaque module couvert est réellement visé par un test
# ═══════════════════════════════════════════════════════════════════════

def _test_corpus():
    text = ""
    for dirpath, _dirs, names in os.walk(os.path.join(_REPO_ROOT, "tests")):
        if "__pycache__" in dirpath:
            continue
        for n in names:
            if n.endswith(".py"):
                text += open(os.path.join(dirpath, n), encoding="utf-8").read()
    return text


@pytest.mark.parametrize("rel", sorted(COVERED))
def test_each_covered_module_is_named_by_the_test_suite(rel):
    """« Couvert » doit vouloir dire qu'un test le nomme vraiment."""
    corpus = _test_corpus()
    stem = os.path.basename(rel)[:-3]
    assert re.search(rf"\b{re.escape(stem)}\b", corpus), (
        f"{rel} est declare couvert mais aucun test ne le mentionne")


def test_the_public_surface_of_the_physics_path_is_exercised():
    """Les fonctions du chemin physique doivent apparaitre dans les tests.

    On se limite au chemin qui produit des nombres publies — pas aux
    pilotes ni aux traces.
    """
    corpus = _test_corpus()
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
            if not re.search(rf"\b{re.escape(name)}\b", corpus):
                missing.append(f"{mod}::{name}")
    assert not missing, f"jamais nommees dans les tests : {missing}"


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
