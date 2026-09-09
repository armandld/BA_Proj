"""Epingle l'artefact de `h3_toy_instability_check.py` -- test de deviation,
pas de regression : casse s'il faut le remesurer, pas parce qu'une valeur
precise a change de quelques millièmes.

Meme mesure que `test_h3_toy_model_check.py` (H0a, H0b), mais sur un
instantane jouet DYNAMIQUE (evolution DNS reelle depuis une recette
d'instabilite a parametres randomises), pas le champ statique synthetique.
"""
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARTIFACT = os.path.join(_REPO_ROOT, "results", "h3_toy_instability_check_N256_dim2_n8.npz")


@pytest.fixture(scope="module")
def artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip(f"artefact absent : {os.path.basename(ARTIFACT)}")
    return np.load(ARTIFACT, allow_pickle=True)


def test_ran_on_the_expected_configuration(artifact):
    assert int(artifact["n_seeds"]) == 8
    assert int(artifact["N"]) == 256
    assert int(artifact["n_patches"]) == 2


def test_every_seed_produced_a_finite_well_shaped_decision(artifact):
    """Chaque instantane a bien traverse tout le pipeline (Hamiltonien,
    optimum exact, QAOA) sans NaN ni fraction hors [0, 1]."""
    for key in ("exact_frac", "qaoa_frac", "f1_exact", "f1_qaoa",
               "f1_classical", "agree_qaoa_exact", "agree_qaoa_classical",
               "agree_classical_exact"):
        arr = artifact[key]
        assert np.all(np.isfinite(arr)), f"{key} contient une valeur non finie"
        assert arr.shape == (8,)
    assert np.all((artifact["exact_frac"] >= 0.0) & (artifact["exact_frac"] <= 1.0))
    assert np.all((artifact["qaoa_frac"] >= 0.0) & (artifact["qaoa_frac"] <= 1.0))


def test_exact_optimum_is_near_uniform_but_not_pinned_to_it(artifact):
    """Comme sur DNS reelles (T26) et le jouet statique : l'optimum exact
    du hamiltonien est presque toujours un masque quasi-uniforme (3 ou 4
    patches/4) -- mais pas fige a 1,0 partout comme sur le jouet statique
    a dim=3 : ici 2/8 graines s'en ecartent (0,75)."""
    assert np.all(artifact["exact_frac"] >= 0.75)
    assert artifact["exact_frac"].min() == pytest.approx(0.75)
    assert np.all(artifact["degeneracy"] == 1)  # unique, pas une egalite


def test_qaoa_does_not_collapse_onto_the_trivial_optimum(artifact):
    """H0a, repliquee sur le jouet dynamique."""
    agree = artifact["agree_qaoa_exact"]
    assert agree.mean() < 0.95
    assert agree.min() < 1.0


def test_qaoa_beats_the_exact_optimum_on_ground_truth_on_average(artifact):
    """H0b, repliquee sur le jouet dynamique -- en MOYENNE seulement.
    Contrairement au jouet statique (f1_exact constant a 0,5, f1_qaoa
    toujours superieur), cet echantillon est plus bruite (n=8, plusieurs
    recettes) : l'inegalite ne tient pas graine par graine (ex. graine 6 :
    f1_qaoa=0 < f1_exact=0,4)."""
    f1_exact = artifact["f1_exact"]
    f1_qaoa = artifact["f1_qaoa"]
    assert np.all(np.isin(np.round(f1_exact, 6), [0.4, 0.5]))
    assert f1_qaoa.mean() > f1_exact.mean()
