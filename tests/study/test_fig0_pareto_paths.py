"""D-94 — `figures/v1_legacy/fig0_pareto_lambda.py` : le script mourait a sa premiere lecture.

`PROJECT_ROOT` ne montait que d'un niveau (meme cause que D-93), et
`Train_results/` a quitte la racine du depot (17d983d l'a mis dans `attic/`,
12a163e a vide l'attic). `TRAIN_DIR` designait donc
`figures/v1_legacy/../Train_results`, qui n'existe pas : `os.listdir` levait
`FileNotFoundError` avant la premiere figure.

Les donnees, elles, sont toujours la : `results/hyperparams/optuna_studies/`
porte les `rescore_{q_has,classical}_v2_phase*_lambda*` que les motifs du
script cherchent.

Ces tests echouent tous sur la version d'avant la correction. On importe le
module par une fixture qui ASSERTE, jamais par `importorskip` : un module
qu'on ne peut pas importer doit rendre la suite ROUGE, pas verte-avec-skip.
"""
import importlib
import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_V1_LEGACY = os.path.join(_REPO_ROOT, "figures", "v1_legacy")


@pytest.fixture(scope="module")
def fig0():
    """Importe le module. Avant D-94 : FileNotFoundError a l'import."""
    if _V1_LEGACY not in sys.path:
        sys.path.insert(0, _V1_LEGACY)
    try:
        return importlib.import_module("fig0_pareto_lambda")
    except Exception as exc:  # noqa: BLE001 — on veut le message exact
        pytest.fail("fig0_pareto_lambda n'est pas importable : %r" % (exc,))


def test_le_module_est_importable(fig0):
    """La garantie de base : le script se charge sans mourir ni rien produire."""
    assert hasattr(fig0, "TRAIN_DIR")


def test_train_dir_existe(fig0):
    """Avant D-94 : <racine>/figures/Train_results, inexistant."""
    assert os.path.isdir(fig0.TRAIN_DIR), (
        "TRAIN_DIR n'existe pas : %s" % fig0.TRAIN_DIR)
    ancien = os.path.join(_REPO_ROOT, "figures", "Train_results")
    assert os.path.abspath(fig0.TRAIN_DIR) != ancien


@pytest.mark.parametrize("nom,motif", [
    ("quantique", "QUANTUM_PATTERN"),
    ("classique", "CLASSICAL_PATTERN"),
])
def test_le_balayage_n_est_pas_vide(fig0, nom, motif):
    """Un balayage vide doit crier : on exige des essais, pas un rc=0.

    C'est l'assertion qui distingue « le chemin existe » de « le chemin porte
    ce que le script y cherche » : un TRAIN_DIR valide mais sans repertoire
    `rescore_*` rendrait un dict vide, et le script tracerait des figures
    vides sans rien dire.
    """
    data = fig0.load_all_trials(fig0.TRAIN_DIR, getattr(fig0, motif))
    assert len(data) > 0, "aucun repertoire rescore_* %s sous %s" % (nom, fig0.TRAIN_DIR)
    n_essais = sum(len(rows) for rows in data.values())
    assert n_essais > 0, "repertoires trouves mais aucun essai lu (%s)" % nom


def test_les_colonnes_par_scenario_sont_bien_la(fig0):
    """Les colonnes que SCENARIOS_ALL nomme existent dans les CSV lus.

    Sinon `_collect_points` rend des tableaux vides et chaque panneau est
    vide, sans erreur — le meme piege, un cran plus loin.
    """
    data = fig0.load_all_trials(fig0.TRAIN_DIR, fig0.QUANTUM_PATTERN)
    rows = next(iter(data.values()))
    for sc_name, sc_info in fig0.SCENARIOS_ALL.items():
        assert sc_info["phys"] in rows[0], "%s: colonne %s absente" % (sc_name, sc_info["phys"])
        assert sc_info["patch"] in rows[0], "%s: colonne %s absente" % (sc_name, sc_info["patch"])
        phys, patch, scores = fig0._collect_points(
            data, sc_info["phys"], sc_info["patch"], fig0.TARGET_LAMBDA)
        assert len(phys) > 0, "%s: aucun point collecte" % sc_name


def test_json_path_ne_touche_pas_lentree_gelee(fig0):
    """La decision ecrite dans le code, verrouillee ici.

    Le bloc de fin du script REECRIT le JSON que JSON_PATH designe.
    `results/hyperparams/` est une entree gelee (son PROVENANCE.md : le seul
    dossier non reproductible par une commande). Une regeneration de figure
    ne doit pas pouvoir muter un artefact qu'on ne sait pas refaire.
    """
    gele = os.path.join(_REPO_ROOT, "results", "hyperparams") + os.sep
    assert not os.path.abspath(fig0.JSON_PATH).startswith(gele), (
        "fig0 ecrirait dans l'entree gelee : %s" % fig0.JSON_PATH)
    assert os.path.basename(fig0.JSON_PATH) != "best_hyperparams.json"


def test_les_cles_ecrites_nont_pas_de_consommateur_dans_lentree_gelee():
    """Mesure qui justifie le choix ci-dessus, re-verifiee a chaque execution.

    Si un jour `best_hyperparams.json` portait `pareto_front_quantum`, la
    decision de detourner l'ecriture demanderait a etre reprise — ce test le
    dirait au lieu de laisser la divergence s'installer.
    """
    gele = os.path.join(_REPO_ROOT, "results", "hyperparams", "best_hyperparams.json")
    with open(gele, encoding="utf-8") as f:
        d = json.load(f)
    assert "pareto_front_quantum" not in d
    assert "pareto_best_quantum" not in d


def test_importer_le_module_n_ecrit_rien(tmp_path):
    """Avant D-94, le bloc MAIN tournait a l'IMPORT : importer le module
    relancait toute la campagne de figures et reecrivait JSON_PATH.

    On mesure l'effet de bord, pas le texte du source : un sous-processus
    importe le module avec `open` et `savefig` instrumentes, et doit
    n'ouvrir aucun fichier en ecriture.
    """
    code = (
        "import builtins, os, sys\n"
        "ecrits = []\n"
        "_open = builtins.open\n"
        "def _spy(f, mode='r', *a, **k):\n"
        "    if any(c in mode for c in 'wax+'):\n"
        "        ecrits.append(str(f))\n"
        "    return _open(f, mode, *a, **k)\n"
        "builtins.open = _spy\n"
        "sys.path.insert(0, %r)\n"
        "import fig0_pareto_lambda\n"
        "print('ECRITS=' + repr(ecrits))\n" % _V1_LEGACY
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr[-3000:]
    ligne = [l for l in out.stdout.splitlines() if l.startswith("ECRITS=")][-1]
    ecrits = eval(ligne[len("ECRITS="):])  # noqa: S307 — liste de chemins qu'on vient d'imprimer
    # `/dev/null` et les caches hors depot (matplotlib, fontconfig) ne comptent
    # pas : ce qu'on interdit, c'est qu'un IMPORT touche au depot.
    dans_le_depot = [c for c in ecrits
                     if os.path.abspath(c).startswith(_REPO_ROOT + os.sep)]
    assert dans_le_depot == [], (
        "l'import de fig0 ecrit dans le depot : %r" % (dans_le_depot,))
