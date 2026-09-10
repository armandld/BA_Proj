"""Epingle l'artefact de `h2b_v2_hamiltonian_vs_gbt_loso.py` -- test de
deviation, pas de regression : casse s'il faut le remesurer.

V2 (hamiltonien sans parametre, aucun entrainement) contre GBT et seuil
classique, contre son propre optimum exact (H0a/H0b) et contre lui-meme
sans couplages (H3), LOSO sur 4 scenarios DNS reels (dim=3, la seule
taille QAOA certifiee non degeneree ici).
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
    for key in ("f1_classical", "f1_gbt", "f1_qaoa", "f1_exact",
               "agree_qaoa_exact", "f1_qaoa_zonly", "f1_exact_zonly"):
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


def test_qaoa_does_not_collapse_onto_the_trivial_optimum(artifact):
    """H0a, ici : accord QAOA/optimum exact tres variable d'un scenario a
    l'autre (0,067 a mhd_rotor, 0,978 a kelvin_helmholtz) -- QAOA
    n'atteint pas de facon fiable l'optimum de son propre hamiltonien."""
    agree = artifact["agree_qaoa_exact"]
    assert agree.mean() < 0.95
    assert agree.min() < 0.5


def test_qaoa_beats_the_exact_optimum_on_ground_truth(artifact):
    """H0b, ici : F1(QAOA) > F1(exact) en moyenne et sur 3 plis/4 --
    resoudre parfaitement le hamiltonien fait PIRE que l'optimisation
    ratee de QAOA, comme sur DNS reelle (T26/D-53/D-200), le jouet
    statique et le jouet dynamique. Seul orszag_tang fait exception
    (exact bat QAOA)."""
    f1_exact = artifact["f1_exact"]
    f1_qaoa = artifact["f1_qaoa"]
    held = list(artifact["held"])
    assert f1_qaoa.mean() > f1_exact.mean()
    n_qaoa_beats_exact = int(np.sum(f1_qaoa > f1_exact))
    assert n_qaoa_beats_exact == 3
    assert f1_exact[held.index("orszag_tang")] > f1_qaoa[held.index("orszag_tang")]


def test_couplings_never_help_the_exact_optimum(artifact):
    """H3, ici, pour l'optimum EXACT (pas QAOA) : le hamiltonien complet
    ne bat jamais le biais Z seul (couplages ZZ/ZZZZ annules) -- 0/4
    plis, comme T26 (dim=3 exhaustif : le complet est INFERIEUR au biais
    Z seul)."""
    f1_exact = artifact["f1_exact"]
    f1_exact_zonly = artifact["f1_exact_zonly"]
    assert np.all(f1_exact <= f1_exact_zonly + 1e-9)
    assert f1_exact.mean() < f1_exact_zonly.mean()


def test_couplings_never_hurt_qaoa_here_unlike_the_exact_optimum(artifact):
    """H3, ici, pour QAOA : nuance par rapport a l'optimum exact --
    jamais de perte (egalite ou mieux sur les 4 plis), et legerement
    mieux en moyenne. Ne PAS lire comme "H3 refute" : QAOA ne resout pas
    le hamiltonien (H0a), donc ce que "les couplages aident QAOA" mesure
    ici, c'est l'effet des couplages sur une recherche ratee, pas sur la
    representation elle-meme -- c'est l'optimum exact (test ci-dessus)
    qui repond a la question de H3."""
    f1_qaoa = artifact["f1_qaoa"]
    f1_qaoa_zonly = artifact["f1_qaoa_zonly"]
    assert np.all(f1_qaoa >= f1_qaoa_zonly - 1e-9)
    assert f1_qaoa.mean() > f1_qaoa_zonly.mean()
