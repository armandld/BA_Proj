"""Epingle l'artefact de `h3_toy_model_check.py` -- test de deviation, pas
de regression : casse s'il faut le remesurer, pas parce qu'une valeur
precise a change de quelques millièmes.

Les deux faits qui comptent, tous les deux des repliques de resultats deja
etablis sur DNS reelles (H0a, H0b -- voir docs/PLAN_PREPRINT.md) :
  1. L'optimum exact du hamiltonien est un masque constant sur ces 20
     instantanes jouets, comme T26 le mesure a 50% du temps sur des
     scenarios reels a la meme taille (dim=3, hyperparametres de
     reference) -- PAS un defaut du generateur (verifie contre 5
     instantanes DNS reels, meme resultat).
  2. QAOA, en ne resolvant PAS parfaitement ce hamiltonien, produit une
     MEILLEURE decision que l'optimum exact (H0b : mieux resoudre H
     degrade la decision).
"""
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARTIFACT = os.path.join(_REPO_ROOT, "results", "h3_toy_model_check_N48_dim3_n20.npz")


@pytest.fixture(scope="module")
def artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip(f"artefact absent : {os.path.basename(ARTIFACT)}")
    return np.load(ARTIFACT, allow_pickle=True)


def test_ran_on_the_expected_configuration(artifact):
    assert int(artifact["n_seeds"]) == 20
    assert int(artifact["N"]) == 48
    assert int(artifact["n_patches"]) == 3


def test_exact_optimum_is_uniform_like_on_real_dns_at_this_size(artifact):
    """Pas un bug du modele jouet : T26 mesure la meme chose sur de vraies
    DNS a dim=3, aux memes hyperparametres de reference."""
    assert np.all(artifact["exact_frac"] == 1.0)
    assert np.all(artifact["degeneracy"] == 1)  # unique, pas une egalite


def test_qaoa_does_not_collapse_onto_the_trivial_optimum(artifact):
    """H0a : QAOA n'atteint pas simplement son propre optimum."""
    agree = artifact["agree_qaoa_exact"]
    assert agree.mean() < 0.95
    assert agree.min() < 1.0


def test_qaoa_beats_the_exact_optimum_on_ground_truth(artifact):
    """H0b, repliquee sur donnees jouets : ne pas pleinement resoudre H
    donne une MEILLEURE decision que le resoudre parfaitement."""
    f1_exact = artifact["f1_exact"]
    f1_qaoa = artifact["f1_qaoa"]
    assert np.all(f1_exact == 0.5)
    assert f1_qaoa.mean() > f1_exact.mean()
