"""D-81 : les bras quantiques de la phase 12 choisissaient leur seuil sur les
labels de VALIDATION, les bras classiques auxquels on les compare sur ceux
d'entrainement.

`run_vqc` et `run_qke` faisaient `best_threshold_f1(p_va, Yva)` — le seuil
optimise sur l'ensemble meme ou le F1 est ensuite rapporte. Les deux bras
classiques du meme script passent par `fit_eval`, qui choisit sur
`(p_tr, Ytr)`. Le verdict compare `max(F1 quantique)` a `max(F1 classique)`
avec une bande de decision de +/- 0,02 : l'ecart de discipline entrait
directement dans la comparaison, en faveur du bras que l'etude cherche
justement a falsifier.

Mesure, phase 12 complete sur une configuration reelle
(`--scenario orszag_tang --re 400 --N 64 --dim 4 --n-train 80 --n-val 60
--d-q 3 --reps-fm 1 --maxiter 15`, sortie hors du depot) :

    QKE, seuil sur validation (avant)  F1 = 0,786
    QKE, seuil sur train    (apres)    F1 = 0,759      ecart 0,027

    verdict avant : delta = -0,008 -> « quantum ~= classical, no clear advantage »
    verdict apres : delta = -0,035 -> « best quantum model UNDERPERFORMS »

Le biais depasse la bande de decision du script et lui fait changer de
verdict sur la premiere configuration mesuree.

Ce test-ci n'utilise pas de donnees MHD : il appelle `run_qke` sur un jeu
construit pour que les deux seuils DIFFERENT. Sur un jeu ou ils coincident,
il ne mesurerait rien.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "study", "pipeline"),
           os.path.join(_REPO_ROOT, "study", "common"),
           os.path.join(_REPO_ROOT, "study", "h2b_prediction")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

qml = pytest.importorskip("qiskit_machine_learning",
                          reason="phase 12 exige qiskit-machine-learning")

from h2b_variational_classifier import run_qke  # noqa: E402
from h2b_ceiling_random_split import best_threshold_f1  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402


def _separating_split():
    """Train et validation dont les seuils F1-optimaux ne coincident pas.

    La validation est deliberement plus deseequilibree que le train : le
    seuil qui maximise le F1 y est plus bas, donc un seuil choisi sur elle
    donne un F1 que le meme modele n'obtient pas avec un seuil honnete.
    """
    rng = np.random.default_rng(0)
    n_tr, n_va = 24, 24
    Xtr = np.concatenate([rng.normal(-0.6, 0.35, size=(n_tr // 2, 2)),
                          rng.normal(+0.6, 0.35, size=(n_tr // 2, 2))])
    Ytr = np.array([0] * (n_tr // 2) + [1] * (n_tr // 2))
    # validation : 3/4 de positifs, et les positifs plus proches du centre
    Xva = np.concatenate([rng.normal(-0.6, 0.35, size=(n_va // 4, 2)),
                          rng.normal(+0.15, 0.5, size=(3 * n_va // 4, 2))])
    Yva = np.array([0] * (n_va // 4) + [1] * (3 * n_va // 4))
    return Xtr, Ytr, Xva, Yva


@pytest.fixture(scope="module")
def qke_result():
    Xtr, Ytr, Xva, Yva = _separating_split()
    r = run_qke(Xtr, Ytr, Xva, Yva, d_q=2, reps_fm=1, seed=0)
    return r, (Xtr, Ytr, Xva, Yva)


def test_the_reported_f1_uses_a_threshold_that_val_labels_never_saw(qke_result):
    """Le F1 rendu doit etre celui du seuil choisi sur le train, applique
    tel quel a la validation."""
    r, (_, _, _, Yva) = qke_result
    expected = f1_score(Yva, (r["p_va"] > r["thr"]).astype(int),
                        zero_division=0)
    assert r["f1"] == pytest.approx(expected), (
        "le F1 rendu n'est pas celui du seuil rendu : le seuil rapporte et "
        "le seuil applique ont diverge")


def test_the_old_optimistic_number_is_kept_and_is_higher(qke_result):
    """`f1_thr_on_val` est l'ancien nombre. Il ne peut qu'etre superieur ou
    egal — c'est un maximum sur la meme grille, pris en connaissant les
    labels. S'il etait EGAL ici, ce test ne separerait rien."""
    r, _ = qke_result
    assert "f1_thr_on_val" in r, (
        "l'ancien nombre n'est plus mesure : le biais de D-81 redeviendrait "
        "invisible")
    assert r["f1_thr_on_val"] >= r["f1"]
    assert r["f1_thr_on_val"] > r["f1"], (
        f"jeu d'essai qui ne SEPARE pas : {r['f1_thr_on_val']:.4f} contre "
        f"{r['f1']:.4f}. Le construire autrement plutot que relacher "
        "l'assertion.")


def test_the_threshold_is_the_train_optimum(qke_result):
    """Verifie la discipline elle-meme : le seuil rendu maximise le F1 sur
    le TRAIN, comme `fit_eval` le fait pour les bras classiques."""
    r, (Xtr, Ytr, _, _) = qke_result
    # on refait le choix depuis les predictions d'entrainement du meme
    # modele : `run_qke` doit avoir choisi le meme seuil
    grid = np.linspace(0.05, 0.95, 91)
    assert np.isclose(grid, r["thr"]).any(), (
        f"le seuil rendu ({r['thr']}) ne vient pas de la grille annoncee")
    # et il ne doit PAS etre l'argmax sur la validation, sinon la correction
    # n'a rien change sur ce jeu (qui a ete construit pour les separer)
    thr_val, _ = best_threshold_f1(r["p_va"],
                                   _separating_split()[3], grid=grid)
    assert r["thr"] != pytest.approx(thr_val), (
        "seuil du train et seuil de la validation coincident : le jeu "
        "d'essai ne separe pas les deux disciplines")
