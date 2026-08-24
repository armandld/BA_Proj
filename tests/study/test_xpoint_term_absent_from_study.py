"""Contrats du terme ZZZZ de point X dans les consommateurs Study/QAOA."""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

DIM = 2


def _params(with_xpoint, k_xpoint=-7.0):
    z = lambda v: np.full((DIM, DIM), v, dtype=float)
    params = {
        "H_edges": (z(0.1), z(0.1)),
        "C_edges": (z(-0.1), z(-0.1)),
        "K_plaquettes": z(-0.1),
    }
    if with_xpoint:
        params["K_xpoint"] = z(k_xpoint)
    return params


def test_qaoa_operator_consumes_xpoint_when_present():
    from VQA.cost_hamiltonian import create_period_hamiltonian

    without = create_period_hamiltonian(_params(False), DIM)
    with_xpoint = create_period_hamiltonian(_params(True), DIM)

    assert len(with_xpoint) == len(without) + DIM * DIM
    assert float(np.sum(np.abs(with_xpoint.coeffs))) > float(
        np.sum(np.abs(without.coeffs)))


def test_study_ising_terms_consume_xpoint_when_present():
    from ising_terms_and_annealing import build_ising_terms

    _, _, without = build_ising_terms(_params(False), DIM)
    _, _, with_xpoint = build_ising_terms(_params(True), DIM)

    assert len(with_xpoint[1]) == len(without[1]) + DIM * DIM
    assert np.sum(with_xpoint[1]) < np.sum(without[1])


def test_zzzz_ablation_removes_both_plaquette_families():
    from h3_term_ablation import ground_state_mask, zero_hamiltonian_terms

    without_xpoint = ground_state_mask(
        zero_hamiltonian_terms(_params(False), ("ZZZZ",)), DIM)
    with_xpoint = ground_state_mask(
        zero_hamiltonian_terms(_params(True), ("ZZZZ",)), DIM)

    np.testing.assert_array_equal(without_xpoint[0], with_xpoint[0])
    assert without_xpoint[1] == with_xpoint[1]
