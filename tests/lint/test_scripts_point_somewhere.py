"""D-116 — les lanceurs de `scripts/` nomment-ils des chemins qui existent ?

Trouve en executant la recette de `CLAUDE.md`. Sa deuxieme commande,
`python -m pytest tests/v3 tests/v4 -q`, rendait :

    no tests ran in 0.01s          [exited with code 0]

`tests/v3/` et `tests/v4/` avaient ete vides par la reorganisation des
tests par sous-systeme (commit b60dc39) ; git ne suit pas les dossiers
vides, seuls leurs `__pycache__` subsistaient sur les copies de travail.
Le test de non-regression annonce par CLAUDE.md ne verifiait donc plus
RIEN, et le disait avec un code de retour 0.

Le meme balayage vide vivait dans `scripts/run_study_v3.sh` :

    want tests && run_step tests python -m pytest "$ROOT_DIR/tests/v3" -q

Et il n'etait pas seul. Les QUINZE chemins de ce lanceur pointaient dans
le vide -- `study/v3/`, `study/results/`, `study/phase11_upper_bound.py`
-- parce que le script datait de deux reorganisations en arriere. En
prime `ROOT_DIR` remontait de deux crans (`../..`) alors que le script
etait descendu dans `scripts/`, un seul cran sous la racine : ROOT_DIR
designait le PARENT du depot.

`scripts/generate_figures_v1.sh` portait la version la plus dangereuse.
Son `FIGURES_CODE_DIR` pointait sur un `figures_code/` disparu (les 17
scripts vivent dans `figures/v1_legacy/`), et sa boucle disait :

    if [[ ! -f "$script_path" ]]; then log "  SKIP: ..."; continue; fi

Les 17 scripts tombaient donc dans la branche SKIP, le lanceur annoncait
« Succeeded: 0  Failed: 0 » et rendait 0. Une campagne de figures verte
qui ne produisait aucune figure.

C'est le piege numero 3 de COUVERTURE, « balayage vide », applique aux
lanceurs plutot qu'aux tests. Un lanceur qui ne lance rien ressemble
exactement a un lanceur qui a tout lance.
"""

import os
import re
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_REPO, "scripts")

#: Chemins que les lanceurs CREENT (`mkdir -p`) ou PRODUISENT : leur
#: absence est normale sur un depot neuf. Tout le reste est une ENTREE,
#: et une entree qui n'existe pas est un lanceur casse.
_SORTIES = ("logs", "results/logs_v3", "results/figures",
            "results/hyperparams/best_hyperparams.regenerated.json",
            # cree par `mkdir -p` dans run_reoptimisation.sh : c'est la
            # base Optuna que la campagne PRODUIT, pas une entree.
            "results/hyperparams/reoptimisation")


def _shell_scripts():
    return sorted(f for f in os.listdir(_SCRIPTS) if f.endswith(".sh"))


def _chemins_du_depot(source):
    """Les chemins litteraux construits sous `$ROOT_DIR` / `$REPO`."""
    return {m.group(1) for m in re.finditer(
        r'\$(?:\{)?(?:ROOT_DIR|REPO|REPO_ROOT)(?:\})?/'
        r'([A-Za-z0-9_][A-Za-z0-9_/.-]*)', source)}


def _collecte(cible):
    """Nombre de tests que pytest collecte reellement sous `cible`."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         os.path.join(_REPO, cible)],
        capture_output=True, text=True, cwd=_REPO)
    return len(re.findall(r"::", r.stdout)), r.stdout


def test_il_y_a_bien_des_lanceurs_a_verifier():
    """Un balayage vide doit crier -- y compris celui-ci."""
    assert _shell_scripts(), "aucun .sh dans scripts/ : le test ne verifie rien"


@pytest.mark.parametrize("script", _shell_scripts())
def test_chaque_chemin_dun_lanceur_existe(script):
    source = open(os.path.join(_SCRIPTS, script)).read()
    morts = [c for c in sorted(_chemins_du_depot(source))
             if not c.startswith(_SORTIES)
             and not os.path.exists(os.path.join(_REPO, c))]
    assert not morts, (
        f"{script} nomme des chemins inexistants : {morts}. Un lanceur qui "
        f"pointe dans le vide echoue a la premiere etape -- ou pire, passe "
        f"en ne faisant rien.")


@pytest.mark.parametrize("script", _shell_scripts())
def test_aucune_passe_pytest_ne_balaie_le_vide(script):
    """`pytest <dossier vide>` rend « no tests ran » et un code 0.

    Le pire des deux mondes : l'etape est verte et n'a rien verifie. On
    collecte pour de vrai, on exige au moins un test.
    """
    source = open(os.path.join(_SCRIPTS, script)).read()
    cibles = re.findall(
        r'pytest\s+"?\$(?:\{)?(?:ROOT_DIR|REPO|REPO_ROOT)(?:\})?/'
        r'([A-Za-z0-9_][A-Za-z0-9_/.-]*)"?', source)
    for cible in cibles:
        n, sortie = _collecte(cible)
        assert n > 0, (
            f"{script} lance pytest sur `{cible}`, qui ne collecte AUCUN "
            f"test : l'etape rendrait 0 sans rien verifier.\n{sortie[-800:]}")


# ══════════════════════════════════════════════════════════════════
#  La recette de CLAUDE.md
# ══════════════════════════════════════════════════════════════════

def _cibles_pytest_de_la_recette():
    source = open(os.path.join(_REPO, "CLAUDE.md"), encoding="utf-8").read()
    cibles = []
    for ligne in source.splitlines():
        if "pytest" not in ligne:
            continue
        for mot in ligne.split():
            if mot.startswith("tests") and not mot.startswith("--"):
                cibles.append(mot)
    return cibles


def test_la_recette_de_claude_md_nomme_des_dossiers_qui_existent():
    cibles = _cibles_pytest_de_la_recette()
    assert cibles, "aucune cible pytest lue dans CLAUDE.md — recette illisible"
    morts = [c for c in cibles if not os.path.exists(os.path.join(_REPO, c))]
    assert not morts, (
        f"la recette de CLAUDE.md nomme {morts}, qui n'existe(nt) pas. "
        f"C'est le test de non-regression du depot : il doit collecter.")


def test_chaque_cible_de_la_recette_collecte_au_moins_un_test():
    for cible in _cibles_pytest_de_la_recette():
        n, _ = _collecte(cible)
        assert n > 0, (
            f"la recette de CLAUDE.md balaie `{cible}` : 0 test collecte.")


def test_les_modules_cites_par_claude_md_existent():
    """CLAUDE.md est lu comme une instruction, pas comme de la prose.

    Il designait `study/h2b_prediction/phase11_upper_bound.py` comme la
    source a reutiliser pour `build_dataset` / `extract_features_2d` /
    `make_model` / `fit_eval` / `best_threshold_f1`. Ce fichier n'existe
    pas : les cinq fonctions vivent dans `h2b_ceiling_random_split.py`.
    Une consigne « reutiliser avant de reecrire » qui nomme un fichier
    absent produit exactement ce qu'elle veut empecher -- une
    reimplementation.
    """
    source = open(os.path.join(_REPO, "CLAUDE.md"), encoding="utf-8").read()
    cites = set(re.findall(r'`((?:src|study|tests|figures|scripts|docs|results)'
                           r'/[A-Za-z0-9_/.-]*\.(?:py|md|json|sh))`', source))
    assert cites, "aucun module cite dans CLAUDE.md — lecture cassee"
    morts = sorted(c for c in cites
                   if not os.path.exists(os.path.join(_REPO, c)))
    assert not morts, f"CLAUDE.md nomme des fichiers inexistants : {morts}"


def test_les_fonctions_que_claude_md_dit_de_reutiliser_sont_importables():
    """Le chemin peut exister et ne pas porter les fonctions annoncees."""
    src = os.path.join(_REPO, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    etude = os.path.join(_REPO, "study", "h2b_prediction")
    if etude not in sys.path:
        sys.path.insert(0, etude)
    mod = pytest.importorskip("h2b_ceiling_random_split")
    for nom in ("build_dataset", "extract_features_2d", "make_model",
                "fit_eval", "best_threshold_f1"):
        assert hasattr(mod, nom), (
            f"CLAUDE.md dit d'importer `{nom}` depuis ce module : absent.")
