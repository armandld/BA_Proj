"""D-63 — `recompute_lambda_scores` sortait avec le code 0 quoi qu'il arrive.

Le corps entier de `main` vivait dans un `try / except Exception` dont la
branche d'erreur imprimait « Erreur lors du chargement » et **ne sortait
pas**. Deux conséquences, mesurées avant correction :

  * une étude inexistante, une base absente : code **0**, aucun fichier
    écrit. Une chaîne de campagne ne pouvait pas distinguer un rescore
    réussi d'un rescore qui n'avait rien produit — et les artefacts du run
    précédent restaient en place ;
  * un échec survenu **après** le chargement (écriture, tracé) était
    annoncé comme une erreur de chargement, alors que la ligne
    `Loaded study ... 178 completed` venait d'être imprimée.

C'est la forme que `CLAUDE.md` interdit — « un balayage vide doit crier » —
et que D-55 et D-56 ont déjà corrigée onze fois dans `study/`.

Sur quelle entrée ces tests échouent-ils ? Sur la version d'avant D-63 :
les deux premiers y voient un code 0.
"""

import os
import subprocess
import sys

import pytest


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_ROOT = _repo_root()
_SCRIPT = os.path.join(_ROOT, "src", "recompute_lambda_scores.py")
_DB = os.path.join(_ROOT, "results", "hyperparams", "optuna_studies",
                   "classical_v2_phase1.db")
_STUDY = "classical_v2_phase1"


def _run(*args, timeout=900):
    return subprocess.run([sys.executable, _SCRIPT, *args],
                          capture_output=True, text=True, timeout=timeout)


@pytest.fixture(autouse=True)
def _needs_frozen_db():
    if not os.path.exists(_DB):
        pytest.skip("base gelee absente : " + _DB)


def test_unknown_study_exits_nonzero(tmp_path):
    out = _run("--db-path", _DB, "--study-name", "pas_une_etude",
               "--lambda-cost", "0.3", "--output-dir", str(tmp_path))
    assert out.returncode != 0, "un rescore qui n'a rien mesure sortait a 0"
    assert "Erreur lors du chargement" in out.stderr
    assert os.listdir(str(tmp_path)) == []


def test_failure_after_loading_is_not_called_a_loading_error(tmp_path):
    """La cause annoncée doit être la vraie.

    On force l'échec APRÈS le chargement : le dossier de sortie est un
    fichier, donc `os.makedirs` lève. L'ancienne version imprimait « Erreur
    lors du chargement » juste après avoir imprimé « Loaded study ».
    """
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")

    out = _run("--db-path", _DB, "--study-name", _STUDY,
               "--lambda-cost", "0.3", "--output-dir", str(blocker))
    assert out.returncode != 0
    assert "Loaded study" in out.stdout, "le chargement devait reussir"
    assert "Erreur lors du chargement" not in (out.stdout + out.stderr)
    assert "NotADirectoryError" in out.stderr


def test_healthy_run_still_exits_zero_and_writes(tmp_path):
    """Épingle le chemin qui marche : la correction ne doit rien casser."""
    out = _run("--db-path", _DB, "--study-name", _STUDY,
               "--lambda-cost", "0.3", "--output-dir", str(tmp_path))
    assert out.returncode == 0, out.stderr[-2000:]
    produced = os.path.join(str(tmp_path), "rescore_%s_lambda0.3000" % _STUDY)
    names = sorted(os.listdir(produced))
    assert "trials_lambda0.3000.csv" in names
    assert "summary_lambda0.3000.txt" in names
