"""Epingle l'artefact de `h2b_toy_ceiling.py` -- test de deviation, pas de
regression : casse s'il faut le remesurer, pas parce qu'une valeur precise a
change de quelques millièmes.

Ce que ce test verifie n'est PAS "le F1 vaut tant" mais l'absence de fuite
train/val (`CLAUDE.md` : « aucun reglage ne voit le scenario tenu ou les
labels d'evaluation ») : un seuil ajuste sur le train qui se degraderait
fortement sur le val, ou qui varierait beaucoup d'un partage a l'autre,
signalerait un ajustement qui a memorise le train plutot qu'appris un vrai
plafond classique.
"""
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARTIFACT = os.path.join(_REPO_ROOT, "results", "h2b_toy_ceiling_N48_dim3_n300.npz")


@pytest.fixture(scope="module")
def artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip(f"artefact absent : {os.path.basename(ARTIFACT)}")
    return np.load(ARTIFACT, allow_pickle=True)


def test_ran_on_the_expected_configuration(artifact):
    assert int(artifact["n_instances"]) == 300
    assert int(artifact["n_splits"]) == 10
    assert float(artifact["train_frac"]) == pytest.approx(0.7)
    assert int(artifact["N"]) == 48
    assert int(artifact["n_patches"]) == 3


def test_threshold_is_stable_across_disjoint_splits(artifact):
    """Un seuil qui bouge beaucoup d'un partage a l'autre ne serait pas un
    plafond classique reel mais un artefact du tirage train/val precis."""
    thr = artifact["thresholds"]
    assert thr.std() < 0.02
    assert (thr.max() - thr.min()) < 0.05


def test_validation_f1_does_not_collapse_relative_to_train(artifact):
    """Signature de memorisation du train : F1 val très inferieur au F1
    train. Ici l'ecart doit rester dans le bruit d'echantillonnage."""
    f1_train = artifact["f1_trains"]
    f1_val = artifact["f1_vals"]
    gap = f1_train - f1_val
    assert gap.mean() < 0.05
    assert f1_val.min() > 0.5


def test_validation_f1_beats_trivial_baselines(artifact):
    """L2_PERCENTILE_HARD=75 -> "tout raffiner" donne F1=0.4 (precision
    0.25, rappel 1.0) ; le score classique doit faire mieux que ce
    plancher, sans pour autant atteindre 1.0 (qui trahirait une fuite)."""
    f1_val = artifact["f1_vals"]
    assert f1_val.mean() > 0.55
    assert f1_val.mean() < 0.95
