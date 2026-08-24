"""Exact phase-4 solving exploits the diagonal Ising structure."""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in [os.path.join(ROOT, "src")] + [
        os.path.join(ROOT, "study", folder) for folder in (
            "pipeline", "common")]:
    if path not in sys.path:
        sys.path.insert(0, path)

from exact_diagonalisation import exact_diag, ground_state_decisions  # noqa: E402
from qaoa_inputs import full_comparison  # noqa: E402
from VQA.cost_hamiltonian import create_period_hamiltonian  # noqa: E402


def _hamiltonian(dim, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "H_edges": [rng.normal(size=(dim, dim)), rng.normal(size=(dim, dim))],
        "C_edges": [rng.normal(size=(dim, dim)), rng.normal(size=(dim, dim))],
        "K_plaquettes": rng.normal(size=(dim, dim)),
        "K_xpoint": rng.normal(size=(dim, dim)),
    }


def test_enumerated_spectrum_matches_the_qiskit_operator():
    hp = _hamiltonian(2)
    state, energy, spectrum, gap = exact_diag(hp, 2)
    reference = np.sort(np.real(np.diag(
        create_period_hamiltonian(hp, 2).to_matrix())))
    np.testing.assert_allclose(spectrum, reference, rtol=0.0, atol=1e-10)
    assert energy == spectrum[0]
    assert gap == spectrum[1] - spectrum[0]
    assert np.count_nonzero(state) == 1


def test_ground_state_vector_and_decisions_use_the_same_bit_order():
    dim = 2
    hp = {
        "H_edges": [np.full((dim, dim), 1.0), np.full((dim, dim), -1.0)],
        "C_edges": [np.zeros((dim, dim)), np.zeros((dim, dim))],
        "K_plaquettes": np.zeros((dim, dim)),
    }
    state, _, _, _ = exact_diag(hp, dim)
    _, horizontal, vertical = ground_state_decisions(state, dim)
    assert horizontal.all()
    assert not vertical.any()


def test_degenerate_ground_states_are_visible_in_the_spectrum():
    dim = 2
    zeros = np.zeros((dim, dim))
    hp = {
        "H_edges": [zeros.copy(), zeros.copy()],
        "C_edges": [zeros.copy(), zeros.copy()],
        "K_plaquettes": zeros.copy(),
    }
    _, energy, spectrum, gap = exact_diag(hp, dim)
    assert energy == 0.0
    assert gap == 0.0
    assert np.count_nonzero(np.isclose(spectrum, energy)) == 2 ** (2 * dim * dim)


def test_qaoa_agreement_is_not_interpreted_against_an_arbitrary_ground_state():
    qaoa_h = np.array([[True, False], [False, True]])
    qaoa_v = np.zeros((2, 2), dtype=bool)
    exact_h = np.array([[True, True], [False, False]])
    exact_v = np.zeros((2, 2), dtype=bool)
    result = full_comparison(
        qaoa_h, qaoa_v, exact_h, exact_v,
        gt_refine=np.zeros((2, 2), dtype=bool),
        score_patch=np.zeros((2, 2)), threshold_amr=0.5,
        exact_ground_degeneracy=2,
    )
    assert result["qaoa_exact_agreement_raw"] == 0.5
    assert np.isnan(result["qaoa_exact_agreement"])
    assert result["exact_ground_degeneracy"] == 2
