"""
V1 guards — the checks that keep a V1 result from being read as something
it is not.

`src/` is frozen, so nothing here changes V1's behaviour. What these tests
do is make three of its silent behaviours *loud*, so that no present or
future V1 test can mistake them for a normal outcome:

  A. The placeholder Hamiltonian. `cost_hamiltonian.py` prunes every
     coefficient below 1e-6, and when that empties the term list it injects
     ("Z", [0], 1e-3) so Qiskit does not crash on an empty observable. The
     object returned is then a *signal that no Hamiltonian was built*, not a
     Hamiltonian. `is_null_placeholder` is the detector; use it before
     interpreting any operator coming out of V1.

  B. The pruning threshold. The Gaussian uncertainty window drives C_edges
     to ~1e-42, which is 36 orders of magnitude below the 1e-6 cut, so the
     ZZ terms are never constructed. "Ablating ZZ changes no decision" is
     therefore not a statement about the optimiser: there is nothing to
     ablate. This is pinned here so the chain window -> pruning -> empty
     ZZ family is checked mechanically rather than remembered.

  C. The swallowed exception. `execute.py:184` sets the sampler's shot count
     for the MPS final readout inside `try/except Exception: pass`. If that
     assignment ever stops working, the readout silently runs at the wrong
     shot count and every marginal downstream is quietly noisier. The test
     below performs the same assignment on the same primitive objects, so
     the failure surfaces here instead of hiding there.

Run with:
    python -m pytest tests/test_v1_guards.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.quantum_info import SparsePauliOp

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.HamiltParams import PhysicalMapper
from VQA.cost_hamiltonian import create_period_hamiltonian

DIM = 2                      # deployed size: 2*DIM^2 = 8 qubits
PRUNE_THRESHOLD = 1e-6       # cost_hamiltonian.py:117 etc.
PLACEHOLDER_COEFF = 1e-3     # cost_hamiltonian.py:215 / :295


# ═══════════════════════════════════════════════════════════════════════
#  THE DETECTOR
# ═══════════════════════════════════════════════════════════════════════

def is_null_placeholder(op, coeff=PLACEHOLDER_COEFF, tol=1e-12):
    """True iff `op` is the placeholder V1 substitutes for an empty term list.

    The placeholder is a single ("Z", [0], 1e-3). Qiskit writes qubit 0 as
    the RIGHTMOST character of the label, so the label is 'I...IZ'.

    Any test that receives a Hamiltonian from V1 should call this first: a
    placeholder means the patch produced no coefficient above the pruning
    threshold, which is a different event from "the Hamiltonian is weak".
    """
    terms = op.to_list()
    if len(terms) != 1:
        return False
    label, c = terms[0]
    if label.count("Z") != 1 or set(label) - {"I", "Z"}:
        return False
    if not label.endswith("Z"):
        return False
    return abs(complex(c) - coeff) < tol


def _flat_params(value, dim=DIM):
    """hamilt_params with every coefficient set to `value`."""
    a = np.full((dim, dim), float(value))
    return {
        'H_edges': (a.copy(), a.copy()),
        'C_edges': (a.copy(), a.copy()),
        'K_plaquettes': a.copy(),
    }


def _diagonal_energies(op):
    """Energies of every computational basis state (the operator is diagonal)."""
    return np.real(np.diag(op.to_matrix()))


# ═══════════════════════════════════════════════════════════════════════
#  A. THE PLACEHOLDER MUST BE DETECTABLE
# ═══════════════════════════════════════════════════════════════════════

class TestNullPlaceholder:

    def test_all_coefficients_pruned_yields_the_placeholder(self):
        """Every coefficient below 1e-6 -> the operator is the placeholder."""
        tiny = PRUNE_THRESHOLD / 1000.0          # 1e-9
        op = create_period_hamiltonian(_flat_params(tiny), DIM)

        assert is_null_placeholder(op), (
            f"expected the injected placeholder, got {op.to_list()}"
        )
        assert op.num_qubits == 2 * DIM * DIM

    def test_the_placeholder_dominates_what_it_replaced(self):
        """The substitute is not small — it is 1e6 times the pruned signal.

        This is why it must be detected rather than tolerated: reading the
        placeholder as a Hamiltonian overstates the physics by six orders
        of magnitude.
        """
        tiny = PRUNE_THRESHOLD / 1000.0
        op = create_period_hamiltonian(_flat_params(tiny), DIM)
        placeholder = abs(complex(op.to_list()[0][1]))

        assert placeholder / tiny == pytest.approx(1e6, rel=1e-9)

    def test_the_placeholder_is_not_physically_neutral(self):
        """It biases qubit 0, deterministically.

        The source comment says the injected term "has no physical effect".
        It has one: ("Z", [0], +1e-3) is minimised by Z = -1 on qubit 0,
        i.e. the ground state refines edge 0 and only edge 0. Half the
        spectrum is strictly preferred over the other half.
        """
        op = create_period_hamiltonian(
            _flat_params(PRUNE_THRESHOLD / 1000.0), DIM)
        energies = _diagonal_energies(op)
        n = op.num_qubits

        # basis index -> bit of qubit 0 (Qiskit: qubit 0 is the LSB)
        q0 = np.array([(i >> 0) & 1 for i in range(2 ** n)])

        assert energies.min() < energies.max(), (
            "the placeholder is claimed to be neutral, but a strictly flat "
            "spectrum would make this assertion fail — check the source"
        )
        best = np.flatnonzero(energies == energies.min())
        assert set(q0[best]) == {1}, (
            "every ground state must have qubit 0 excited — the placeholder "
            "is a refine-edge-0 bias"
        )
        assert energies.min() == pytest.approx(-PLACEHOLDER_COEFF, rel=1e-12)

    def test_placeholder_escapes_the_null_hamiltonian_shortcut(self):
        """`execute()` skips COBYLA only for an all-zero operator.

        Its test is `np.allclose(np.abs(coeffs), 0.0)`, whose default atol
        is 1e-8. The placeholder sits at 1e-3, so it does NOT trigger the
        shortcut: a patch with no surviving coefficient runs a full
        variational optimisation against a fabricated single-Z operator.
        Pinned here because the two thresholds live in different files and
        nothing else connects them.
        """
        op = create_period_hamiltonian(
            _flat_params(PRUNE_THRESHOLD / 1000.0), DIM)

        assert not np.allclose(np.abs(op.coeffs), 0.0), (
            "if this ever becomes True the shortcut fires and the "
            "placeholder path changes meaning"
        )
        # ... whereas a genuinely zero operator does trigger it
        zero_op = SparsePauliOp.from_sparse_list(
            [("Z", [0], 0.0)], num_qubits=op.num_qubits)
        assert np.allclose(np.abs(zero_op.coeffs), 0.0)

    def test_a_real_hamiltonian_is_not_flagged(self):
        """The detector must not fire on ordinary operators."""
        op = create_period_hamiltonian(_flat_params(0.5), DIM)
        assert not is_null_placeholder(op)
        assert len(op.to_list()) > 1

        single_real_term = SparsePauliOp.from_sparse_list(
            [("Z", [0], 0.7)], num_qubits=2 * DIM * DIM)
        assert not is_null_placeholder(single_real_term), (
            "a single Z with a physical coefficient is not the placeholder"
        )


# ═══════════════════════════════════════════════════════════════════════
#  B. WINDOWED ZZ NEVER REACHES THE OPERATOR
# ═══════════════════════════════════════════════════════════════════════

class TestPruningThreshold:

    def test_windowed_zz_is_pruned_before_the_solver(self):
        """C_edges is computed, is nonzero, and is dropped by the 1e-6 cut.

        The three assertions are ordered so that the failure tells you which
        link broke: the coupling exists, it is below the cut, and no ZZ term
        appears in the operator.
        """
        N = DIM
        fields = {
            'vx': np.array([[2.0, 2.0], [0.0, 0.0]], dtype=float),
            'vy': np.zeros((N, N)),
            'Bx': np.zeros((N, N)),
            'By': np.zeros((N, N)),
            'Jz': np.zeros((N, N)),
        }
        hm = PhysicalMapper(cs=1.0, eta_mhd=0.01)
        sim = MHDSolver(PeriodicGrid(N))
        score = hm.physical_score(fields)
        hp = hm.compute_coefficients(sim, score, fields, 0.0)

        C_h, C_v = hp['C_edges']
        c_max = max(np.max(np.abs(C_h)), np.max(np.abs(C_v)))

        assert c_max > 0.0, "the gradient coupling must be computed, not skipped"
        assert c_max < PRUNE_THRESHOLD, (
            f"windowed coupling {c_max:.3e} is expected far below the "
            f"{PRUNE_THRESHOLD:g} pruning cut"
        )

        op = create_period_hamiltonian(hp, DIM)
        zz_terms = [t for t in op.to_list() if t[0].count("Z") == 2]
        assert zz_terms == [], (
            f"no ZZ term can reach the solver at this size, found {zz_terms}"
        )

    def test_a_coupling_above_the_cut_does_reach_the_operator(self):
        """Control: the ZZ family is not structurally absent."""
        hp = _flat_params(0.0)
        hp['C_edges'][0][:] = 0.5
        op = create_period_hamiltonian(hp, DIM)
        zz_terms = [t for t in op.to_list() if t[0].count("Z") == 2]
        assert len(zz_terms) == DIM * DIM, (
            "with a coupling above the cut, one ZZ term per site is expected"
        )


# ═══════════════════════════════════════════════════════════════════════
#  C. THE SWALLOWED EXCEPTION
# ═══════════════════════════════════════════════════════════════════════

class TestSamplerShotOption:
    """`execute.py:182-185` swallows any failure of

        sampler.options.default_shots = mps_shots

    If it fails, the MPS final readout runs at the previous shot count and
    nothing says so. The same assignment is exercised here, un-caught, on
    both primitive construction paths.
    """

    def test_legacy_path_accepts_the_shot_option(self):
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        sampler = Sampler(mode=AerSimulator(method='matrix_product_state'))
        sampler.options.default_shots = 4096
        assert sampler.options.default_shots == 4096

        sampler.options.default_shots = 8192          # what execute() does
        assert sampler.options.default_shots == 8192, (
            "the assignment execute.py performs inside try/except must take "
            "effect, otherwise the MPS readout is silently under-sampled"
        )

    def test_runtime_path_accepts_the_shot_option(self):
        from VQA.runtime import VQARuntime

        rt = VQARuntime(backend_name="matrix_product_state", mode="local",
                        shots=4096, opt_level=1)
        assert rt.sampler.options.default_shots == 4096

        rt.sampler.options.default_shots = 8192
        assert rt.sampler.options.default_shots == 8192


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
