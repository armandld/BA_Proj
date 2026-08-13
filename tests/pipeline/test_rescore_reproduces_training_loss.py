"""Le contrat central de `recompute_lambda_scores`, verrouille.

La promesse du module : rejouer le score d'un essai sous un autre
`lambda_cost`. Elle n'a de valeur que si, **au lambda de l'entrainement**,
elle rend exactement ce que l'entrainement a mesure. Sinon toute
comparaison « ancien contre nouveau » melange un changement de lambda avec
un ecart de formule, sans qu'on puisse les separer.

Deux chemins doivent donc coincider :

    pipeline.score        combined = (phys + λ·patch) / (1 + λ)
    _composite_loop       perte de l'essai = moyenne des sous-pertes
    recompute_score       meme formule, meme moyenne

Verifie ici sur les 303 essais complets des deux bases gelees, au
`LAMBDA_COST_SOFT = 0.4` de la campagne. Mesure du 13 aout 2026 : ecart
maximal **5,6e-17** (quantique) et **2,2e-16** (classique).

Ce test n'a jamais echoue : il epingle un resultat sain, pour qu'un
changement de formule d'un cote sans l'autre se voie tout de suite.
"""

import os
import sys

import numpy as np
import pytest


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_ROOT = _repo_root()
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_STUDIES = os.path.join(_ROOT, "results", "hyperparams", "optuna_studies")
_N_TRIALS = {"q_has_v2_phase1": 178, "classical_v2_phase1": 125}


@pytest.fixture(scope="module")
def rescorer():
    return pytest.importorskip("recompute_lambda_scores")


@pytest.mark.parametrize("study", sorted(_N_TRIALS))
def test_recompute_at_training_lambda_is_the_training_loss(rescorer, study):
    db = os.path.join(_STUDIES, study + ".db")
    if not os.path.exists(db):
        pytest.skip("base gelee absente : " + db)

    train = pytest.importorskip("train_hyperparams")
    lam = train.LAMBDA_COST_SOFT
    assert lam == 0.4, "le lambda de la campagne a change : remesurer"

    _s, completed = rescorer.load_completed_trials(db, study)
    assert len(completed) == _N_TRIALS[study]

    delta = np.array([rescorer.recompute_score(t, lam) - t.value
                      for t in completed])
    assert np.abs(delta).max() < 1e-12, (
        "ecart max %.3e : la formule du rescore et celle de l'entrainement "
        "ont divergé" % np.abs(delta).max())


def test_a_different_lambda_actually_moves_the_score(rescorer):
    """Le test precedent ne prouve rien si le rescore ignore son lambda.

    Sur quelle entree separe-t-il ? Un lambda de 0,0 (cout ignore) contre
    1,0 (cout a poids egal) : les deux doivent s'ecarter de l'entrainement,
    et dans des sens opposes des que l'essai a un patch_ratio non nul.
    """
    study = "classical_v2_phase1"
    db = os.path.join(_STUDIES, study + ".db")
    if not os.path.exists(db):
        pytest.skip("base gelee absente : " + db)
    _s, completed = rescorer.load_completed_trials(db, study)

    keys = rescorer._detect_scenario_keys(completed)
    phys = np.array([rescorer._get_global_phys_patch(t, keys)[0]
                     for t in completed])
    patch = np.array([rescorer._get_global_phys_patch(t, keys)[1]
                      for t in completed])
    zero = np.array([rescorer.recompute_score(t, 0.0) for t in completed])
    one = np.array([rescorer.recompute_score(t, 1.0) for t in completed])
    orig = np.array([t.value for t in completed])

    assert np.abs(zero - orig).max() > 1e-3
    assert np.abs(one - orig).max() > 1e-3

    # λ = 0 ne garde que la physique : `zero` DOIT valoir `phys`.
    np.testing.assert_allclose(zero, phys, rtol=0, atol=1e-15)

    # Et le sens du deplacement est celui de (phys − patch), exactement :
    #     zero − orig = (phys − patch) · λ/(1+λ) au lambda d'entrainement.
    # Ce n'est pas « ca bouge », c'est le sens PREVU, essai par essai.
    # Mesure du 13 aout 2026 : 118 essais baissent, 7 montent — ces 7 sont
    # ceux dont l'erreur physique depasse le cout, la premiere version de ce
    # test les avait oublies.
    assert np.array_equal(np.sign(zero - orig), np.sign(phys - patch))
    assert int((zero < orig).sum()) == 118
    assert int((zero > orig).sum()) == 7
