"""Epingle l'artefact de `h2b_v2_hamiltonian_vs_gbt_loso.py` -- test de
deviation, pas de regression : casse s'il faut le remesurer.

V2 (hamiltonien sans parametre, aucun entrainement) contre GBT et seuil
classique, LOSO sur 4 scenarios DNS reels (dim=3, la seule taille QAOA
certifiee non degeneree ici).
"""
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARTIFACT = os.path.join(
    _REPO_ROOT, "results", "h2b_v2_hamiltonian_vs_gbt_loso_N96_dim3_n5.npz")


@pytest.fixture(scope="module")
def artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip(f"artefact absent : {os.path.basename(ARTIFACT)}")
    return np.load(ARTIFACT, allow_pickle=True)


def test_ran_on_the_expected_configuration(artifact):
    assert int(artifact["N"]) == 96
    assert int(artifact["dim"]) == 3
    assert int(artifact["n_snaps"]) == 5
    assert list(artifact["held"]) == [
        "harris_tearing", "kelvin_helmholtz", "mhd_rotor", "orszag_tang"]


def test_every_fold_produced_a_finite_f1_in_range(artifact):
    for key in ("f1_classical", "f1_gbt", "f1_qaoa"):
        arr = artifact[key]
        assert arr.shape == (4,)
        assert np.all(np.isfinite(arr))
        assert np.all((arr >= 0.0) & (arr <= 1.0))
