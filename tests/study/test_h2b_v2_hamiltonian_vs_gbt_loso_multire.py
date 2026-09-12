"""Epingle l'artefact multi-Re + bootstrap de `h2b_v2_hamiltonian_vs_gbt_loso.py`
-- test de deviation, pas de regression : casse s'il faut le remesurer.

Suite directe de `test_h2b_v2_hamiltonian_vs_gbt_loso.py` (Re=400, n=5,
sans IC -- SUPERSEDED, voir RESULTS.md) : 4 regimes Re empiles par pli
LOSO, n=40 instantanes tenus/pli, IC95 bootstrap (n_boot=1000) sur chaque
delta. Le verdict QAOA-contre-classique s'INVERSE a cette echelle par
rapport a la mesure Re=400/n=5 -- voir RESULTS.md, section "V2 contre
GBT, multi-Re + IC95 bootstrap", pour la lecture complete.
"""
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARTIFACT = os.path.join(
    _REPO_ROOT, "results",
    "h2b_v2_hamiltonian_vs_gbt_loso_N96_dim3_n10_multire.npz")

F1_KEYS = ("f1_classical", "f1_gbt", "f1_qaoa", "f1_exact",
           "f1_qaoa_zonly", "f1_exact_zonly", "agree_qaoa_exact")
CI_PAIRS = (
    ("ci_qaoa_classical_lo", "ci_qaoa_classical_hi", "p_qaoa_ge_classical"),
    ("ci_qaoa_exact_lo", "ci_qaoa_exact_hi", "p_qaoa_ge_exact"),
    ("ci_exact_full_zonly_lo", "ci_exact_full_zonly_hi", "p_exact_full_ge_zonly"),
    ("ci_qaoa_full_zonly_lo", "ci_qaoa_full_zonly_hi", "p_qaoa_full_ge_zonly"),
)


@pytest.fixture(scope="module")
def artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip(f"artefact absent : {os.path.basename(ARTIFACT)}")
    return np.load(ARTIFACT, allow_pickle=True)


def test_ran_on_the_expected_configuration(artifact):
    assert int(artifact["N"]) == 96
    assert int(artifact["dim"]) == 3
    assert int(artifact["n_snaps"]) == 10
    assert int(artifact["n_boot"]) == 1000
    assert int(artifact["k_opt"]) == 60
    assert int(artifact["shots"]) == 4096
    assert list(artifact["re_values"]) == [400, 800, 1200, 1600]
    assert list(artifact["held"]) == [
        "harris_tearing", "kelvin_helmholtz", "mhd_rotor", "orszag_tang"]
    assert list(artifact["n_val_snaps"]) == [40, 40, 40, 40]


def test_every_f1_array_is_finite_and_in_range(artifact):
    for key in F1_KEYS:
        arr = artifact[key]
        assert arr.shape == (4,)
        assert np.all(np.isfinite(arr))
        assert np.all((arr >= 0.0) & (arr <= 1.0))


def test_every_bootstrap_ci_is_finite_ordered_and_a_valid_probability(artifact):
    for lo_key, hi_key, p_key in CI_PAIRS:
        lo, hi, p = artifact[lo_key], artifact[hi_key], artifact[p_key]
        assert lo.shape == hi.shape == p.shape == (4,)
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
        assert np.all(lo <= hi + 1e-9)
        assert np.all((p >= 0.0) & (p <= 1.0))


def test_classical_beats_or_ties_both_learners_on_every_fold(artifact):
    """Le resultat central de cette mesure : a 4 Re et n=40/pli, le seuil
    classique bat ou egale QAOA(V2) ET GBT sur les 4 plis LOSO, jamais
    l'inverse -- INVERSE la mesure Re=400/n=5 ("QAOA bat classique sur
    3 plis/4"), voir RESULTS.md."""
    f1_classical = artifact["f1_classical"]
    f1_qaoa = artifact["f1_qaoa"]
    f1_gbt = artifact["f1_gbt"]
    assert np.all(f1_classical >= f1_qaoa - 1e-9)
    assert np.all(f1_classical >= f1_gbt - 1e-9)
    assert f1_classical.mean() > f1_qaoa.mean()
    assert f1_classical.mean() > f1_gbt.mean()


def test_qaoa_loses_to_classical_with_confidence_except_a_degenerate_tie(artifact):
    """IC95 bootstrap du delta F1(QAOA)-F1(classique) : negatif avec
    confiance (p=0) sur kelvin_helmholtz, mhd_rotor, orszag_tang.
    harris_tearing est une egalite EXACTE (IC de largeur nulle) -- les
    trois arbitres rendent le meme F1 sur les 40 instantanes tenus, pas
    une victoire QAOA."""
    held = list(artifact["held"])
    p = artifact["p_qaoa_ge_classical"]
    ci_hi = artifact["ci_qaoa_classical_hi"]
    for sc in ("kelvin_helmholtz", "mhd_rotor", "orszag_tang"):
        i = held.index(sc)
        assert p[i] == 0.0
        assert ci_hi[i] < 0.0
    i = held.index("harris_tearing")
    assert p[i] == 1.0
    lo, hi = artifact["ci_qaoa_classical_lo"][i], artifact["ci_qaoa_classical_hi"][i]
    assert lo == 0.0 and hi == 0.0


def test_qaoa_beats_gbt_strictly_on_exactly_one_fold(artifact):
    """A cette echelle, QAOA bat GBT strictement sur mhd_rotor (0,476
    contre 0,456) -- nuance la mesure Re=400/n=5 ("jamais strictement
    mieux que GBT")."""
    held = list(artifact["held"])
    f1_gbt = artifact["f1_gbt"]
    f1_qaoa = artifact["f1_qaoa"]
    i = held.index("mhd_rotor")
    assert f1_qaoa[i] > f1_gbt[i]
    n_qaoa_strictly_beats_gbt = int(np.sum(f1_qaoa > f1_gbt + 1e-9))
    assert n_qaoa_strictly_beats_gbt == 1


def test_h0a_agreement_still_highly_variable_by_scenario(artifact):
    """H0a, repliquee : l'accord QAOA/optimum exact du hamiltonien
    complet reste tres heterogene par scenario (10,6% a mhd_rotor, 99,4%
    a kelvin_helmholtz) -- meme direction qu'a Re=400/n=5."""
    agree = artifact["agree_qaoa_exact"]
    assert agree.min() < 0.2
    assert agree.max() > 0.9


def test_h0b_holds_with_confidence_on_three_folds_orszag_tang_the_exception(artifact):
    """H0b, avec IC95 : F1(QAOA) >= F1(exact), confiant (p=1) sur
    harris_tearing, kelvin_helmholtz, mhd_rotor. orszag_tang reste la
    seule exception, confiante dans l'autre sens (p=0) -- meme exception
    qu'a Re=400/n=5, cinquieme contexte independant ou H0b tient."""
    held = list(artifact["held"])
    p = artifact["p_qaoa_ge_exact"]
    for sc in ("harris_tearing", "kelvin_helmholtz", "mhd_rotor"):
        assert p[held.index(sc)] == 1.0
    assert p[held.index("orszag_tang")] == 0.0
    assert artifact["f1_qaoa"].mean() > artifact["f1_exact"].mean()


def test_couplings_never_help_the_exact_optimum_with_confidence(artifact):
    """H3 pour l'exact, avec IC95 : le hamiltonien complet ne bat jamais
    le biais Z seul -- 2 egalites exactes (kelvin_helmholtz,
    orszag_tang), 2 infériorites confiantes (p=0, harris_tearing,
    mhd_rotor). Confirme T26 et Re=400/n=5."""
    f1_exact = artifact["f1_exact"]
    f1_exact_zonly = artifact["f1_exact_zonly"]
    assert np.all(f1_exact <= f1_exact_zonly + 1e-9)
    assert f1_exact.mean() < f1_exact_zonly.mean()
    p = artifact["p_exact_full_ge_zonly"]
    assert int(np.sum(p == 0.0)) == 2
    assert int(np.sum(p == 1.0)) == 2


def test_couplings_no_longer_help_qaoa_either(artifact):
    """H3 pour QAOA : la nuance Re=400/n=5 ("QAOA profite legerement des
    couplages") s'INVERSE ici -- aucune victoire significative du complet
    (1 egalite quasi exacte, 1 non-conclusif p=0,408, 1 defaite confiante
    p=0). QAOA rejoint l'exact : les couplages n'aident nulle part."""
    p = artifact["p_qaoa_full_ge_zonly"]
    held = list(artifact["held"])
    assert p[held.index("mhd_rotor")] == pytest.approx(0.408)
    assert p[held.index("orszag_tang")] == 0.0
    ci_lo = artifact["ci_qaoa_full_zonly_lo"]
    ci_hi = artifact["ci_qaoa_full_zonly_hi"]
    for i in range(4):
        assert not (ci_lo[i] > 1e-9)


def test_ground_state_is_always_unique(artifact):
    """dim=3 (18 qubits) reste non degenere sur les 160 instantanes
    mesures (4 scenarios x 4 Re x 10) -- meme conclusion que T26, verifiee
    ici sur un echantillon 8x plus grand."""
    deg = artifact["degeneracy"]
    assert deg.shape == (160,)
    assert np.all(deg == 1)


def test_provenance_head_move_is_accounted_for(artifact):
    """head_moved_during_run=True est attendu ici (mesure de ~78,5 min) :
    verifie que l'arbre etait propre au depart, pas que HEAD n'a pas
    bouge -- voir RESULTS.md pour la verification des 2 commits
    concernes."""
    assert bool(artifact["dirty_at_start"]) is False
    assert str(artifact["git_hash_at_start"]) == "3d5fe73a085a7e001913fbbc71afdcc688b4e5e9"
