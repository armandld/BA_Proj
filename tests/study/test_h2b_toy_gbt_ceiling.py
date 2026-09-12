"""Epingle l'artefact de `h2b_toy_gbt_ceiling.py` -- test de deviation, pas
de regression : casse s'il faut le remesurer, pas parce qu'une valeur
precise a change de quelques millièmes.

Deux choses comptent ici :
  1. Absence de fuite train/val, meme discipline que
     `test_h2b_toy_ceiling.py` (`CLAUDE.md` : « aucun reglage ne voit le
     scenario tenu ou les labels d'evaluation »).
  2. Le seuil classique retrouve ici (~0.50) doit rester coherent avec
     celui mesure independamment par `h3_toy_model_check.py` (0.4920) et
     `h2b_toy_ceiling.py` (0.4980 +/- 0.0040) -- trois pools d'instances
     jouets differents, trois scripts differents, le meme score classique.
"""
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARTIFACT = os.path.join(_REPO_ROOT, "results", "h2b_toy_gbt_ceiling_N48_dim3_n300.npz")


@pytest.fixture(scope="module")
def artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip(f"artefact absent : {os.path.basename(ARTIFACT)}")
    return np.load(ARTIFACT, allow_pickle=True)


def test_ran_on_the_expected_configuration(artifact):
    assert int(artifact["n_instances"]) == 300
    assert int(artifact["n_splits"]) == 5
    assert int(artifact["N"]) == 48
    assert int(artifact["n_patches"]) == 3


def test_classical_threshold_agrees_with_the_other_two_toy_pools(artifact):
    """Meme score classique, trois pools jouets independants : h3_toy_model
    _check.py (0.4920), h2b_toy_ceiling.py (0.4980 +/- 0.0040), et ici."""
    thr = artifact["thr"]
    assert thr.std() < 0.02
    assert 0.45 < thr.mean() < 0.55


def test_classical_val_f1_does_not_collapse_relative_to_train(artifact):
    gap = artifact["f1_class_train"] - artifact["f1_class_val"]
    assert abs(gap.mean()) < 0.05


def test_site_ceiling_is_a_real_but_modest_gain_over_classical(artifact):
    """Ni degrade (le meilleur des 3 modeles doit au moins egaler le seuil
    brut), ni suspicieusement parfait (signe de fuite)."""
    f1_site_best = artifact["f1_site_best"]
    f1_class_val = artifact["f1_class_val"]
    assert f1_site_best.mean() >= f1_class_val.mean() - 0.02
    assert f1_site_best.mean() < 0.95


def test_stencil_ceiling_does_not_blow_far_past_site_ceiling(artifact):
    """Q2 (voisinage) proche de Q1 (site) est un resultat attendu et deja
    documente sur DNS reelles (H0b : les couplages n'ajoutent pas grand
    chose) -- une derive large signalerait un vrai changement a comprendre,
    pas seulement un chiffre a remettre a jour."""
    delta = artifact["f1_stencil_gbt"] - artifact["f1_site_best"]
    assert abs(delta.mean()) < 0.10
