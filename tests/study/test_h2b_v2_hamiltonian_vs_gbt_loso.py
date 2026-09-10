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


def test_qaoa_never_strictly_beats_gbt(artifact):
    """QAOA(V2) egale GBT a 3 plis sur 4 (harris_tearing, mhd_rotor,
    orszag_tang) et perd sur le 4e (kelvin_helmholtz) -- jamais mieux."""
    f1_gbt = artifact["f1_gbt"]
    f1_qaoa = artifact["f1_qaoa"]
    assert np.all(f1_qaoa <= f1_gbt + 1e-9)
    assert np.sum(np.isclose(f1_qaoa, f1_gbt)) == 3


def test_mhd_rotor_favours_the_classical_threshold_over_both_learners(artifact):
    """Seul pli ou classique bat QAOA ET GBT. Ne pas confondre avec la
    degenerescence de `init_mhd_rotor` documentee ailleurs (score
    classique uniforme dans le pipeline complet a N=256) : ici classique
    est au contraire le MEILLEUR, pas degenere -- mecanisme different,
    non explique, seulement constate."""
    held = list(artifact["held"])
    i = held.index("mhd_rotor")
    assert artifact["f1_classical"][i] > artifact["f1_gbt"][i]
    assert artifact["f1_classical"][i] > artifact["f1_qaoa"][i]


def test_qaoa_beats_classical_on_average_without_any_training(artifact):
    """V2 n'a aucun parametre a regler (D-22 ne le concerne pas) et bat
    quand meme le seuil classique en moyenne LOSO."""
    assert artifact["f1_qaoa"].mean() > artifact["f1_classical"].mean()
