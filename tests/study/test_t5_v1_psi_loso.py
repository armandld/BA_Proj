"""Tests V3 Task 5 : helpers psi signe / agregation (sans qiskit).

La reproduction des nombres publies de phase 11e (bras legacy block_avg
a beta=0.5495) est validee par l'execution sur les vraies donnees.
"""
import os
import sys

import numpy as np
import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HERE = os.path.dirname(os.path.abspath(__file__))

from h2b_psi_feature_loso import (
    BETA_TRIAL4,
    THR_TRIAL4,
    block_agg,
    psi_abs_v1,
    psi_signed_v1,
    score_aug_legacy,
    score_aug_signed,
    signed_combine,
)
from Simulation.HamiltParams_v2 import compute_psi_v2


def test_trial4_params_come_from_config():
    assert BETA_TRIAL4 == pytest.approx(9.94)
    assert THR_TRIAL4 == pytest.approx(0.1496)


# -------------------------- combinaison signee ------------------------

def test_signed_combine_keeps_sign_of_larger_magnitude():
    psi_h = np.array([+0.2, -0.9, +0.5])
    psi_v = np.array([-0.7, +0.1, -0.5])   # egalite au 3e -> h
    out = signed_combine(psi_h, psi_v)
    np.testing.assert_allclose(out, [-0.7, -0.9, +0.5])


def test_abs_of_signed_equals_legacy_abs():
    rng = np.random.default_rng(0)
    dph = rng.normal(size=(8, 8))
    dpv = rng.normal(size=(8, 8))
    for beta in (0.5495, 9.94):
        s = psi_signed_v1(dph, dpv, beta)
        a = psi_abs_v1(dph, dpv, beta)
        np.testing.assert_allclose(np.abs(s), a)


def test_psi_abs_matches_phase11e_formula():
    # replique independante de v1_psi_field (phase 11e)
    rng = np.random.default_rng(1)
    dph = rng.normal(size=(6, 6))
    dpv = rng.normal(size=(6, 6))
    beta = 0.5495
    avg = float(np.mean(np.abs(dph) + np.abs(dpv))) / 2.0
    ref = np.maximum(np.abs((np.pi / 2) * np.tanh(beta * dph / avg)),
                     np.abs((np.pi / 2) * np.tanh(beta * dpv / avg)))
    np.testing.assert_allclose(psi_abs_v1(dph, dpv, beta), ref)


def test_psi_signed_sign_and_saturation():
    dph = np.array([[+1.0, -1.0], [0.0, 0.0]])
    dpv = np.zeros((2, 2))
    psi = psi_signed_v1(dph, dpv, beta=1000.0)  # tanh sature
    assert psi[0, 0] == pytest.approx(+np.pi / 2, rel=1e-6)
    assert psi[0, 1] == pytest.approx(-np.pi / 2, rel=1e-6)
    assert psi[1, 0] == 0.0


def test_psi_zero_flux_edge_case():
    z = np.zeros((4, 4))
    np.testing.assert_array_equal(psi_signed_v1(z, z, 9.94), z)
    np.testing.assert_array_equal(psi_abs_v1(z, z, 9.94), z)


# ----------------------------- score_aug ------------------------------

def test_score_aug_signed_identity_at_zero_psi():
    s = np.array([0.0, 0.3, 1.0])
    np.testing.assert_allclose(score_aug_signed(s, np.zeros(3)), s)


def test_score_aug_signed_damping_reduces_urgency_and_clips():
    s = np.array([0.4, 0.05, 0.9])
    psi = np.array([-np.pi / 2, -np.pi / 2, +np.pi / 2])
    out = score_aug_signed(s, psi)
    np.testing.assert_allclose(out, [0.0, 0.0, 1.0])  # clip [0, 1]
    psi_small = np.array([-0.1, 0.0, 0.1])
    out2 = score_aug_signed(s, psi_small)
    assert out2[0] < s[0] and out2[2] > s[2]   # signe preserve


def test_score_aug_legacy_matches_phase11e():
    s = np.array([0.4, 0.9])
    psi_abs = np.array([np.pi / 4, np.pi / 2])
    np.testing.assert_allclose(score_aug_legacy(s, psi_abs),
                               [0.4 + 0.5, 0.9 + 1.0])


# ---------------------------- agregation ------------------------------

def test_block_agg_avg_and_max():
    f = np.arange(16, dtype=float).reshape(4, 4)
    np.testing.assert_allclose(block_agg(f, 2, "avg"),
                               [[2.5, 4.5], [10.5, 12.5]])
    np.testing.assert_allclose(block_agg(f, 2, "max"),
                               [[5.0, 7.0], [13.0, 15.0]])


# ------------------------- compute_psi_v2 (src) -----------------------

def test_compute_psi_v2_sign_and_range():
    rng = np.random.default_rng(2)
    phi_prev = rng.uniform(size=(8, 8))
    phi = phi_prev + rng.normal(size=(8, 8))
    psi = compute_psi_v2(phi_prev, phi)
    assert np.all(np.abs(psi) <= np.pi / 2)
    np.testing.assert_array_equal(np.sign(psi),
                                  np.sign(phi - phi_prev))
