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
