"""Tests V3 Task 3 : bootstrap par blocs trajectoire.

Critere d'acceptation du protocole : distribution d'echantillonnage
connue sur donnees synthetiques + demonstration que les CI trajectoire
sont plus larges que les CI snapshot sur entree autocorrelee.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "study", "v3"))
from stats import bootstrap_by_trajectory, paired_delta_bootstrap


def _autocorrelated_data(T=50, m=10, seed=0):
    """T trajectoires de m snapshots chacune, valeur CONSTANTE par
    trajectoire (autocorrelation intra-trajectoire maximale)."""
    rng = np.random.default_rng(seed)
    effects = rng.normal(size=T)
    values = np.repeat(effects, m)
    traj_ids = np.repeat(np.arange(T), m)
    return values, traj_ids, effects


def _snapshot_bootstrap_ci(values, B=1000, seed=0):
    """Bootstrap naif au niveau snapshot (interdit pour les chiffres
    titres, §1.5) — uniquement pour la comparaison de largeur de CI."""
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([values[rng.integers(0, n, size=n)].mean()
                     for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return lo, hi


# ------------------ distribution d'echantillonnage -------------------

def test_point_estimate_is_full_sample_statistic():
    values, traj_ids, _ = _autocorrelated_data()
    res = bootstrap_by_trajectory(values, traj_ids, B=200, seed=1)
    assert res["estimate"] == pytest.approx(values.mean())
    assert res["n_traj"] == 50


def test_bootstrap_std_matches_known_sampling_distribution():
    # valeurs constantes par trajectoire : la moyenne globale a pour
    # erreur-type analytique std(effets)/sqrt(T)
    values, traj_ids, effects = _autocorrelated_data(T=50, m=10, seed=2)
    res = bootstrap_by_trajectory(values, traj_ids, B=2000, seed=3)
    se_true = effects.std(ddof=0) / np.sqrt(len(effects))
    assert res["boot"].std() == pytest.approx(se_true, rel=0.2)


def test_ci_covers_true_mean():
    values, traj_ids, _ = _autocorrelated_data(T=80, m=5, seed=4)
    res = bootstrap_by_trajectory(values, traj_ids, B=1000, seed=5)
    assert res["ci_low"] <= 0.0 <= res["ci_high"]  # vraie moyenne = 0
    assert res["ci_low"] < res["estimate"] < res["ci_high"]


def test_trajectory_ci_wider_than_snapshot_ci_on_autocorrelated_input():
    # acceptation protocole : sur entree autocorrelee (m copies par
    # trajectoire), le bootstrap snapshot sous-estime la variance d'un
    # facteur ~sqrt(m) ; la CI trajectoire doit etre nettement plus large
    values, traj_ids, _ = _autocorrelated_data(T=50, m=10, seed=6)
    res = bootstrap_by_trajectory(values, traj_ids, B=1000, seed=7)
    width_traj = res["ci_high"] - res["ci_low"]
    lo_s, hi_s = _snapshot_bootstrap_ci(values, B=1000, seed=7)
    width_snap = hi_s - lo_s
    assert width_traj > 1.5 * width_snap  # ratio theorique ~ sqrt(10)


def test_deterministic_given_seed():
    values, traj_ids, _ = _autocorrelated_data(seed=8)
    r1 = bootstrap_by_trajectory(values, traj_ids, B=100, seed=9)
    r2 = bootstrap_by_trajectory(values, traj_ids, B=100, seed=9)
    np.testing.assert_array_equal(r1["boot"], r2["boot"])
    r3 = bootstrap_by_trajectory(values, traj_ids, B=100, seed=10)
    assert not np.array_equal(r1["boot"], r3["boot"])


def test_input_validation():
    with pytest.raises(ValueError):
        bootstrap_by_trajectory([1.0, 2.0], [0], B=10)
    with pytest.raises(ValueError):
        bootstrap_by_trajectory([], [], B=10)


# ------------------------- variante appariee --------------------------

def test_paired_delta_constant_shift():
    values, traj_ids, _ = _autocorrelated_data(T=40, m=8, seed=11)
    res = paired_delta_bootstrap(values + 1.0, values, traj_ids,
                                 B=500, seed=12)
    assert res["mean_delta"] == pytest.approx(1.0)
    assert res["frac_positive"] == 1.0
    assert res["ci_low"] == pytest.approx(1.0)   # deltas tous exactement 1
    assert res["ci_high"] == pytest.approx(1.0)
    assert res["ci_low"] > 0.0                   # CI exclut 0


def test_paired_delta_sign_convention_and_fraction():
    # delta_t = stat(a_t) - stat(b_t) ; la moitie des trajectoires
    # favorise a, l'autre b, d'une marge identique -> delta moyen ~ 0
    T, m = 30, 4
    traj_ids = np.repeat(np.arange(T), m)
    base = np.zeros(T * m)
    shift = np.repeat(np.where(np.arange(T) % 2 == 0, 1.0, -1.0), m)
    res = paired_delta_bootstrap(base + shift, base, traj_ids,
                                 B=500, seed=13)
    assert res["mean_delta"] == pytest.approx(0.0)
    assert res["frac_positive"] == pytest.approx(0.5)
    assert res["ci_low"] < 0.0 < res["ci_high"]  # CI contient 0


def test_paired_delta_length_mismatch_raises():
    values, traj_ids, _ = _autocorrelated_data(T=10, m=3, seed=14)
    with pytest.raises(ValueError):
        paired_delta_bootstrap(values, values[:-1], traj_ids, B=10)


def test_paired_delta_unequal_trajectory_sizes():
    # tailles de trajectoires inegales : stat par trajectoire d'abord
    traj_ids = np.array([0, 0, 0, 1, 1, 2])
    a = np.array([1.0, 1.0, 1.0, 3.0, 3.0, 5.0])
    b = np.zeros(6)
    res = paired_delta_bootstrap(a, b, traj_ids, B=200, seed=15)
    np.testing.assert_allclose(sorted(res["deltas"]), [1.0, 3.0, 5.0])
    assert res["mean_delta"] == pytest.approx(3.0)  # moyenne non ponderee
    assert res["frac_positive"] == 1.0
