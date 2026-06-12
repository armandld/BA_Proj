"""Tests V3 Task 9 : condition de la Proposition 2.

La comptabilite par site est verifiee contre un minimiseur Ising
exhaustif (dim=2, 8 qubits, 256 etats) qui reproduit la liste de
termes de create_period_hamiltonian — la Proposition elle-meme est
ainsi testee numeriquement : condition satisfaite partout => l'etat
fondamental exact est s* = -sign(h).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "study", "v3"))

from t9_prop2_check import mean_field_state, per_site_condition

DIM = 2
N_Q = 2 * DIM * DIM


def _hp(H0=None, H1=None, C0=None, C1=None, K=None, KX=None):
    z = np.zeros((DIM, DIM))
    d = dict(H_edges=[H0 if H0 is not None else z.copy(),
                      H1 if H1 is not None else z.copy()],
             C_edges=[C0 if C0 is not None else z.copy(),
                      C1 if C1 is not None else z.copy()],
             K_plaquettes=K if K is not None else z.copy())
    if KX is not None:
        d["K_xpoint"] = KX
    return d


def _idx_h(y, x):
    return (y % DIM) * DIM + (x % DIM)


def _idx_v(y, x):
    return DIM * DIM + (y % DIM) * DIM + (x % DIM)


def _energy(hp, s):
    """Energie Ising, miroir de la liste de termes de
    create_period_hamiltonian (convention Z|0> = +1)."""
    E = 0.0
    K = hp["K_plaquettes"]
    KX = hp.get("K_xpoint")
    for i in range(DIM):
        for j in range(DIM):
            E += hp["H_edges"][0][i, j] * s[_idx_h(i, j)]
            E += hp["H_edges"][1][i, j] * s[_idx_v(i, j)]
            E += hp["C_edges"][0][i, j] * s[_idx_h(i, j)] * s[_idx_h(i, j + 1)]
            E += hp["C_edges"][1][i, j] * s[_idx_v(i, j)] * s[_idx_v(i + 1, j)]
            plaq = (s[_idx_h(i, j)] * s[_idx_v(i, j + 1)]
                    * s[_idx_h(i + 1, j)] * s[_idx_v(i, j)])
            E += K[i, j] * plaq
            if KX is not None:
                E += KX[i, j] * plaq
    return E


def _brute_force_ground(hp):
    best_s, best_E = None, np.inf
    for bits in range(2 ** N_Q):
        s = np.array([1 - 2 * ((bits >> q) & 1) for q in range(N_Q)])
        E = _energy(hp, s)
        if E < best_E - 1e-12:
            best_E, best_s = E, s.copy()
    return best_s, best_E


# ------------------------ comptabilite par site ------------------------

def test_single_coupling_hits_exactly_two_sites():
    C0 = np.zeros((DIM, DIM)); C0[0, 0] = 0.5
    lhs, h, sat = per_site_condition(_hp(C0=C0), DIM)
    expected = np.zeros(N_Q)
    expected[_idx_h(0, 0)] = 1.0   # 2|C|
    expected[_idx_h(0, 1)] = 1.0
    np.testing.assert_allclose(lhs, expected)
    assert not sat.any()           # h = 0 partout : condition stricte


def test_single_plaquette_hits_exactly_four_sites():
    K = np.zeros((DIM, DIM)); K[0, 0] = 0.3
    lhs, _, _ = per_site_condition(_hp(K=K), DIM)
    qs = [_idx_h(0, 0), _idx_v(0, 1), _idx_h(1, 0), _idx_v(0, 0)]
    expected = np.zeros(N_Q)
    for q in qs:
        expected[q] = 1.2          # 4|K|
    np.testing.assert_allclose(lhs, expected)


def test_periodic_wrap_of_couplings():
    C0 = np.zeros((DIM, DIM)); C0[0, DIM - 1] = 1.0   # H(0,1)-H(0,0)
    lhs, _, _ = per_site_condition(_hp(C0=C0), DIM)
    assert lhs[_idx_h(0, DIM - 1)] == 2.0
    assert lhs[_idx_h(0, 0)] == 2.0
    assert lhs.sum() == 4.0


def test_xpoint_included_and_excludable():
    K = np.zeros((DIM, DIM)); K[0, 0] = 0.1
    KX = np.zeros((DIM, DIM)); KX[0, 0] = 0.2
    lhs_with, _, _ = per_site_condition(_hp(K=K, KX=KX), DIM)
    lhs_wo, _, _ = per_site_condition(_hp(K=K, KX=KX), DIM,
                                      include_xpoint=False)
    assert lhs_with[_idx_h(0, 0)] == pytest.approx(4 * 0.3)
    assert lhs_wo[_idx_h(0, 0)] == pytest.approx(4 * 0.1)


def test_mean_field_state_is_minus_sign_h():
    H0 = np.array([[+1.0, -2.0], [+0.5, -0.5]])
    H1 = -H0
    s = mean_field_state(_hp(H0=H0, H1=H1), DIM)
    for i in range(DIM):
        for j in range(DIM):
            assert s[_idx_h(i, j)] == -np.sign(H0[i, j])
            assert s[_idx_v(i, j)] == -np.sign(H1[i, j])


# ------------------- Proposition 2 contre force brute ------------------

@pytest.mark.parametrize("rng_seed", [0, 1, 2])
def test_condition_satisfied_implies_mean_field_ground_state(rng_seed):
    rng = np.random.default_rng(rng_seed)
    # |h| dans [1, 2], couplages minuscules -> condition vraie partout
    H0 = rng.choice([-1, 1], (DIM, DIM)) * rng.uniform(1, 2, (DIM, DIM))
    H1 = rng.choice([-1, 1], (DIM, DIM)) * rng.uniform(1, 2, (DIM, DIM))
    C0 = rng.uniform(-1e-2, 1e-2, (DIM, DIM))
    C1 = rng.uniform(-1e-2, 1e-2, (DIM, DIM))
    K = rng.uniform(-1e-2, 1e-2, (DIM, DIM))
    hp = _hp(H0, H1, C0, C1, K)
    _, _, sat = per_site_condition(hp, DIM)
    assert sat.all()
    s_star, _ = _brute_force_ground(hp)
    np.testing.assert_array_equal(s_star, mean_field_state(hp, DIM))


def test_strong_couplings_break_condition_and_mean_field():
    # h faible uniforme, ZZ antiferro fort : retournement collectif
    H0 = np.full((DIM, DIM), 0.1)
    H1 = np.full((DIM, DIM), 0.1)
    C0 = np.full((DIM, DIM), 2.0)
    C1 = np.full((DIM, DIM), 2.0)
    hp = _hp(H0, H1, C0, C1)
    _, _, sat = per_site_condition(hp, DIM)
    assert not sat.any()
    s_star, E_star = _brute_force_ground(hp)
    s_mf = mean_field_state(hp, DIM)
    assert not np.array_equal(s_star, s_mf)
    assert E_star < _energy(hp, s_mf)   # le champ moyen n'est pas optimal
