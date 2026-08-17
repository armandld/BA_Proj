"""D-102 — la conclusion de fig15 citait le sigma d'un AUTRE module.

`fig15_decision_flip_analysis.py` construit son `HamiltMapper` via
`_hamilt_mapper_kwargs` (`fig_utils.py`), dont le repli pour `sigma` vaut
0,05 (`TRAINED_PARAMS.get('sigma', 0.05)`) : `'sigma'` n'est echantillonne
par aucune entree de `results/hyperparams/best_hyperparams.json` (D-22), le
repli s'applique donc inconditionnellement dans ce fichier.

Le bloc "CONCLUSION" (imprime quand `flip_rate < 0.05 et mean_ratio < 0.5`)
citait pourtant `sigma=0.023` en dur — la valeur de `TRAINED_SIGMA` dans
`study/pipeline/config.py`, un module que ce fichier n'importe pas et dont
le pipeline (ferme) est distinct de celui de `figures/v1_legacy/`. Question
4 de `VIGIL.md` : deux chemins censes decrire le meme "sigma trained" ne
coincidaient pas.

Ce test n'importe pas `fig15_decision_flip_analysis.py` lui-meme : le
fichier execute sa campagne complete (VQA sur 4 scenarios) a l'IMPORT, sans
garde `if __name__ == "__main__"` (meme contrainte que
`test_v1_legacy_instrumented_bfs_score_grid.py` pour D-96). Il verifie donc
la grandeur reellement en jeu (le repli de TRAINED_PARAMS, importable seul
et sans simulation) et le texte source.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_V1_LEGACY = os.path.join(_REPO_ROOT, "figures", "v1_legacy")
if _V1_LEGACY not in sys.path:
    sys.path.insert(0, _V1_LEGACY)

from fig_utils import TRAINED_PARAMS  # noqa: E402

_FIG15 = os.path.join(_V1_LEGACY, "fig15_decision_flip_analysis.py")


def test_sigma_key_is_still_absent_from_deployed_hyperparams():
    """Precondition du defaut (D-22) : si elle cesse d'etre vraie, le repli
    0,05 ci-dessous cesse d'etre le nombre reellement utilise et ce test
    doit etre remesure, pas simplement mis a jour."""
    assert "sigma" not in TRAINED_PARAMS


def test_the_actual_fallback_this_file_uses_is_0_05_not_0_023():
    """C'est cette valeur, et non 0,023, que `_hamilt_mapper_kwargs`
    applique reellement dans fig15 (et fig11/fig16/fig17, meme helper)."""
    assert TRAINED_PARAMS.get("sigma", 0.05) == 0.05
    assert TRAINED_PARAMS.get("sigma", 0.05) != 0.023


def test_source_no_longer_hardcodes_the_other_modules_sigma():
    src = open(_FIG15, encoding="utf-8").read()
    assert "σ=0.023" not in src, (
        "fig15 cite encore le TRAINED_SIGMA de study/pipeline/config.py, "
        "pas le repli 0,05 que son propre HamiltMapper utilise (D-102)")


def test_source_prints_the_dynamically_computed_sigma():
    src = open(_FIG15, encoding="utf-8").read()
    assert "sigma_trained = TRAINED_PARAMS.get('sigma', 0.05)" in src
    assert "{sigma_trained:" in src
