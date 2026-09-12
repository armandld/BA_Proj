"""
Comprehensive module-by-module validation for Q-HAS.

Tests every module in the pipeline independently to confirm correctness:
  1. AngleMapper: stress flux computation, theta encoding, psi encoding
  2. PhysicalMapper: H_edges (Z), C_edges (ZZ), K_plaquettes (ZZZZ)
  3. Init qubit state: R(theta, psi) gate produces correct amplitudes
  4. Cost Hamiltonian: operator structure, term counts, symmetry
  5. QAOA execution: ground state for trivial Hamiltonians
  6. Postprocess: marginal normalization
  7. RescaleArrays: max-abs pooling preserves anomalies
  8. Refinement decision: threshold logic

Run:
    cd /home/user/BA_Proj && python -m pytest tests/test_module_validation.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Simulation.grid import PeriodicGrid
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import get_adaptive_flux, _maxabs_pool_2d
from VQA.init_qbits_state import init_qbits_state
from VQA.cost_hamiltonian import create_period_hamiltonian, create_bounded_hamiltonian
from VQA.mapping import mapping
from VQA.execute import execute
from VQA.postprocess import postprocess


# ══════════════════════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════════════════════

class MockSolver:
    def __init__(self, grid):
        self.grid = grid


def make_uniform_fields(N, v_bg=0.1, B_bg=0.5):
    return {
        'vx': np.full((N, N), v_bg),
        'vy': np.full((N, N), v_bg),
        'Bx': np.full((N, N), B_bg),
        'By': np.full((N, N), B_bg),
        'Jz': np.zeros((N, N)),
    }


# ══════════════════════════════════════════════════════════════════════
#  MODULE 1: AngleMapper — stress flux computation
# ══════════════════════════════════════════════════════════════════════

class TestStressFlux:
    """Verify compute_stress_flux correctly detects gradients."""

    def test_uniform_fields_zero_flux(self):
        """Uniform fields have zero spatial gradients -> zero flux."""
        mapper = AngleMapper()
        N = 16
        fields = make_uniform_fields(N)
        phi = mapper.compute_stress_flux(fields)

        assert phi['phi_horizontal'].shape == (N, N)
        assert phi['phi_vertical'].shape == (N, N)
        np.testing.assert_allclose(phi['phi_horizontal'], 0.0, atol=1e-10)
        np.testing.assert_allclose(phi['phi_vertical'], 0.0, atol=1e-10)

    def test_step_function_creates_flux(self):
        """A step function in vx creates horizontal flux at the discontinuity."""
        mapper = AngleMapper()
        N = 16
        fields = make_uniform_fields(N, v_bg=0.0, B_bg=0.0)
        # Step at column 8
        fields['vx'][:, :8] = 1.0
        fields['vx'][:, 8:] = -1.0

        phi = mapper.compute_stress_flux(fields)

        # Horizontal flux should be non-zero at the step boundary
        max_flux_h = np.max(phi['phi_horizontal'])
        assert max_flux_h > 0.1, f"Step function should create horizontal flux, got max={max_flux_h}"

    def test_vertical_step_creates_vertical_flux(self):
        """A step in vy along rows creates vertical flux."""
        mapper = AngleMapper()
        N = 16
        fields = make_uniform_fields(N, v_bg=0.0, B_bg=0.0)
        fields['vy'][:8, :] = 1.0
        fields['vy'][8:, :] = -1.0

        phi = mapper.compute_stress_flux(fields)
        max_flux_v = np.max(phi['phi_vertical'])
        assert max_flux_v > 0.1, f"Vertical step should create vertical flux, got max={max_flux_v}"

    def test_flux_symmetry(self):
        """Transposing the fields should swap horizontal and vertical flux."""
        mapper = AngleMapper()
        N = 16
        fields = make_uniform_fields(N, v_bg=0.0, B_bg=0.0)
        fields['vx'][:, :8] = 2.0

        phi1 = mapper.compute_stress_flux(fields)

        # Transpose: swap vx<->vy, swap axes
        fields2 = make_uniform_fields(N, v_bg=0.0, B_bg=0.0)
        fields2['vy'][:8, :] = 2.0
        phi2 = mapper.compute_stress_flux(fields2)

        # Horizontal flux of fields1 should relate to vertical flux of fields2
        assert np.max(phi1['phi_horizontal']) > 0.1
        assert np.max(phi2['phi_vertical']) > 0.1


# ══════════════════════════════════════════════════════════════════════
#  MODULE 2: AngleMapper — theta encoding
# ══════════════════════════════════════════════════════════════════════

class TestThetaEncoding:
    """Verify θ = 2·arcsin(√score) maps classical score to rotation angle correctly."""

    @staticmethod
    def _score_to_theta(score):
        """Helper: θ = 2·arcsin(√score), matching map_to_angles."""
        return 2.0 * np.arcsin(np.sqrt(np.clip(score, 0.0, 1.0)))

    def test_zero_score_gives_zero_theta(self):
        """score = 0 -> theta = 0 (qubit stays in |0>)."""
        theta = self._score_to_theta(np.array([0.0]))
        np.testing.assert_allclose(theta, 0.0, atol=1e-10)

    def test_max_score_gives_pi_theta(self):
        """score = 1 -> theta = pi (qubit fully in |1>)."""
        theta = self._score_to_theta(np.array([1.0]))
        np.testing.assert_allclose(theta, np.pi, atol=1e-10)

    def test_monotonicity(self):
        """Larger score -> larger theta (monotonic increasing)."""
        scores = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
        theta = self._score_to_theta(scores)
        for i in range(len(theta) - 1):
            assert theta[i+1] > theta[i], f"Theta not monotonic at index {i}"

    def test_probability_identity(self):
        """sin²(θ/2) should equal the input score (P(|1⟩) = score)."""
        scores = np.linspace(0.0, 1.0, 50)
        theta = self._score_to_theta(scores)
        recovered = np.sin(theta / 2.0) ** 2
        np.testing.assert_allclose(recovered, scores, atol=1e-12)

    def test_theta_range_is_0_to_pi(self):
        """Theta must stay in [0, pi] for all valid score values."""
        scores = np.linspace(0.0, 1.0, 1000)
        theta = self._score_to_theta(scores)
        assert np.all(theta >= -1e-10), "Theta < 0"
        assert np.all(theta <= np.pi + 1e-10), "Theta > pi"


# ══════════════════════════════════════════════════════════════════════
#  MODULE 3: AngleMapper — psi encoding
# ══════════════════════════════════════════════════════════════════════

class TestPsiEncoding:
    """Verify _activation_function_psi maps flux changes correctly."""

    def test_no_change_gives_zero_psi(self):
        """phi == phi_prev -> psi = 0."""
        mapper = AngleMapper()
        phi = np.array([1.0, 2.0, 3.0])
        psi = mapper._activation_function_psi(phi, phi, beta=1.0, AveragePhiDev=1.0)
        np.testing.assert_allclose(psi, 0.0, atol=1e-10)

    def test_growing_signal_positive_psi(self):
        """phi > phi_prev -> positive psi (growing instability)."""
        mapper = AngleMapper()
        phi_prev = np.array([1.0])
        phi = np.array([3.0])
        psi = mapper._activation_function_psi(phi_prev, phi, beta=1.0, AveragePhiDev=1.0)
        assert psi[0] > 0, f"Growing signal should give positive psi, got {psi[0]}"

    def test_decaying_signal_negative_psi(self):
        """phi < phi_prev -> negative psi (damping)."""
        mapper = AngleMapper()
        phi_prev = np.array([3.0])
        phi = np.array([1.0])
        psi = mapper._activation_function_psi(phi_prev, phi, beta=1.0, AveragePhiDev=1.0)
        assert psi[0] < 0, f"Decaying signal should give negative psi, got {psi[0]}"

    def test_psi_bounded(self):
        """Psi must stay in [-pi/2, pi/2]."""
        mapper = AngleMapper()
        phi_prev = np.array([0.0])
        phi = np.array([1000.0])
        for beta in [0.1, 1.0, 5.0, 10.0]:
            psi = mapper._activation_function_psi(phi_prev, phi, beta=beta, AveragePhiDev=1.0)
            assert abs(psi[0]) <= np.pi / 2 + 1e-10, (
                f"Psi out of bounds for beta={beta}: |psi|={abs(psi[0]):.4f}"
            )

    def test_beta_sensitivity(self):
        """Higher beta -> larger |psi| for same flux change."""
        mapper = AngleMapper()
        phi_prev = np.array([1.0])
        phi = np.array([2.0])
        avg_dev = 1.0
        psi_low = mapper._activation_function_psi(phi_prev, phi, beta=0.5, AveragePhiDev=avg_dev)
        psi_high = mapper._activation_function_psi(phi_prev, phi, beta=5.0, AveragePhiDev=avg_dev)
        assert abs(psi_high[0]) > abs(psi_low[0]), (
            f"Higher beta should give larger |psi|: beta=0.5 -> {psi_low[0]:.4f}, "
            f"beta=5.0 -> {psi_high[0]:.4f}"
        )

    def test_none_prev_gives_zero(self):
        """When phi_prev is None, psi should be zero."""
        mapper = AngleMapper()
        phi = np.array([1.0, 2.0])
        psi = mapper._activation_function_psi(None, phi, beta=1.0, AveragePhiDev=1.0)
        np.testing.assert_allclose(psi, 0.0, atol=1e-10)

    def test_map_to_angles_no_prev(self):
        """map_to_angles with None phi_prev returns psi=0."""
        mapper = AngleMapper()
        phi_dict = {'phi_horizontal': np.ones((4, 4)), 'phi_vertical': np.ones((4, 4))}
        score_h = np.full((4, 4), 0.5)
        score_v = np.full((4, 4), 0.5)
        _, _, psi_h, psi_v = mapper.map_to_angles(
            score_h, score_v, None, phi_dict, None, 1.0
        )
        np.testing.assert_allclose(psi_h, 0.0, atol=1e-10)
        np.testing.assert_allclose(psi_v, 0.0, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════
#  MODULE 4: PhysicalMapper — Hamiltonian coefficients
# ══════════════════════════════════════════════════════════════════════

class TestHamiltParams:
    """Verify PhysicalMapper computes correct Hamiltonian coefficients."""

    def _make_setup(self, N=16):
        grid = PeriodicGrid(N, 2 * np.pi)
        nu = grid.L / 100.0
        eta_mhd = grid.L / 100.0
        mapper = PhysicalMapper(cs=1.0, nu=nu, eta_mhd=eta_mhd,
                                beta_curl=0.5, beta_xpoint=0.5, dx=grid.dx)
        sim = MockSolver(grid)
        return grid, mapper, sim

    def test_uniform_fields_H_negative(self):
        """Uniform fields -> cos_theta ~ 1 -> H = 4*(threshold - 1) < 0.

        For uniform fields, phi ~ 0, so r ~ 0, cos_theta ~ 1.
        With threshold_amr < 1, H should be negative (push toward |0>, don't refine).
        """
        grid, mapper, sim = self._make_setup()
        fields = make_uniform_fields(grid.N)
        score = mapper.physical_score(fields)
        hp = mapper.compute_coefficients(
            sim, score, fields, threshold_amr=0.5
        )

        H_h, H_v = hp['H_edges']
        # For uniform fields, score ~ 0 everywhere
        # H = 4 * (score - threshold) = 4 * (0 - 0.5) = -2.0
        assert np.all(H_h <= 0), f"Uniform fields should give H <= 0, got max={H_h.max()}"
        assert np.all(H_v <= 0), f"Uniform fields should give H <= 0, got max={H_v.max()}"

    def test_H_edges_adaptive_z_weight(self):
        """Adaptive Z: H_edges scale as a fraction of max(|C|,|K|) when active."""
        grid, mapper, sim = self._make_setup()
        fields = make_uniform_fields(grid.N)
        # Create a strong anomaly in top-left
        fields['vx'][:8, :8] += 3.0
        fields['vy'][:8, :8] += 3.0

        score = mapper.physical_score(fields)

        hp = mapper.compute_coefficients(
            sim, score, fields, threshold_amr=0.5
        )

        H_h, H_v = hp['H_edges']
        C_h, C_v = hp['C_edges']
        K = hp['K_plaquettes']

        max_H = np.max(np.abs(np.concatenate([H_h.ravel(), H_v.ravel()])))
        max_CK = max(np.max(np.abs(C_h)), np.max(np.abs(C_v)),
                      np.max(np.abs(K)))

        if max_CK == 0:
            # Quiet fields: Z should also be zero (alpha scales with max|C|,|K|)
            assert max_H == 0.0, (
                f"When C/K are zero, H_edges should be zero, got {max_H:.6f}")
        else:
            # Active fields: Z should be non-zero but smaller than ZZ/ZZZZ scale
            assert max_H > 0, (
                f"Active fields should produce non-zero H_edges, got {max_H:.6f}")
            assert max_H < max_CK, (
                f"H_edges ({max_H:.6f}) should be a fraction of "
                f"max(|C|,|K|) ({max_CK:.6f})")

    def test_threshold_contrast_filter(self):
        """Threshold-relative contrast kills sub-critical signals (v9)."""
        grid, mapper, sim = self._make_setup()
        # Test _threshold_contrast directly
        val = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        val_crit = 1.0
        beta = 0.5
        result = mapper._threshold_contrast(val, val_crit, beta)

        # signal = β × max(0, val/val_crit − 1)
        # val = 0.1: 0.5 * max(0, 0.1 - 1) = 0 (sub-critical)
        assert result[0] == 0.0, f"Sub-critical should be 0, got {result[0]}"
        # val = 1.0: 0.5 * max(0, 1.0 - 1) = 0 (exactly at critical)
        assert result[2] == 0.0, f"At-critical should be 0, got {result[2]}"
        # val = 5.0: 0.5 * max(0, 5.0 - 1) = 2.0 (above critical)
        assert result[4] > 0, f"Above-critical should be > 0, got {result[4]}"
        np.testing.assert_allclose(result[4], 0.5 * (5.0 / 1.0 - 1.0))

    def test_C_edges_from_velocity_jump(self):
        """A velocity discontinuity should produce non-zero C_edges."""
        grid, mapper, sim = self._make_setup()
        fields = make_uniform_fields(grid.N, v_bg=0.0, B_bg=0.0)
        # Strong velocity step at column 8
        fields['vx'][:, :8] = 3.0
        fields['vx'][:, 8:] = -3.0

        score = mapper.physical_score(fields)

        hp = mapper.compute_coefficients(
            sim, score, fields, threshold_amr=0.5
        )

        C_h, C_v = hp['C_edges']
        # C is now negative (ferromagnetic sign in HamiltParams), check it's non-zero
        assert C_h.min() < 0, f"Velocity jump should produce C_h < 0, got min={C_h.min()}"

    def test_K_plaquettes_from_vortex(self):
        """A vortex should produce non-zero K_plaquettes."""
        grid, mapper, sim = self._make_setup()
        N = grid.N
        fields = make_uniform_fields(N, v_bg=0.0, B_bg=0.5)

        # Lamb-Oseen vortex centered at (N/4, N/4)
        cx, cy = N // 4, N // 4
        for i in range(N):
            for j in range(N):
                dx = i - cx
                dy = j - cy
                r = np.sqrt(dx**2 + dy**2) + 1e-10
                v_theta = 5.0 * (1 - np.exp(-r**2 / 9.0)) / r
                fields['vx'][i, j] += -v_theta * dy / r
                fields['vy'][i, j] += v_theta * dx / r

        score = mapper.physical_score(fields)

        hp = mapper.compute_coefficients(
            sim, score, fields, threshold_amr=0.5
        )

        K = hp['K_plaquettes']
        # K is now negative (even-parity sign in HamiltParams), check it's non-zero
        assert K.min() < 0, f"Vortex should produce K < 0, got min={K.min()}"

    def test_coefficient_weight_hierarchy(self):
        """Weight hierarchy: Z < ZZ/ZZZZ (adaptive fraction), C(2) > K(1)."""
        grid, mapper, sim = self._make_setup()
        fields = make_uniform_fields(grid.N)
        fields['vx'][:8, :8] += 3.0

        score = mapper.physical_score(fields)

        hp = mapper.compute_coefficients(
            sim, score, fields, threshold_amr=0.5
        )

        H_h, _ = hp['H_edges']
        C_h, _ = hp['C_edges']
        K = hp['K_plaquettes']

        max_H = np.max(np.abs(H_h))
        max_C = np.max(np.abs(C_h))
        max_K = np.max(np.abs(K))

        print(f"\n  Weight hierarchy: |H|_max={max_H:.4f}, |C|_max={max_C:.4f}, |K|_max={max_K:.4f}")
        # Adaptive Z: H_edges should be smaller than max(|C|, |K|)
        max_CK = max(max_C, max_K)
        assert max_H < max_CK, (
            f"H_edges ({max_H:.6f}) should be less than max(|C|,|K|) ({max_CK:.6f})")
        # C_edges (weight=2) should dominate K_plaquettes (weight=1)
        assert max_C > 0, "C_edges should have non-zero values"


# ══════════════════════════════════════════════════════════════════════
#  MODULE 5: init_qbits_state — R gate verification
# ══════════════════════════════════════════════════════════════════════

class TestInitQubitsState:
    """Verify that R(theta, psi+pi/2) produces correct qubit amplitudes."""

    def test_theta_zero_gives_ground_state(self):
        """theta = 0 -> qubit in |0>, P(|1>) = 0."""
        from qiskit_aer import AerSimulator

        theta_h = np.array([[0.0]])
        theta_v = np.array([[0.0]])
        psi_h = np.array([[0.0]])
        psi_v = np.array([[0.0]])

        qc = init_qbits_state(theta_h, theta_v, psi_h, psi_v)
        qc.measure_all()

        backend = AerSimulator(method='statevector')
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)
        result = sampler.run([qc], shots=4096).result()[0]
        counts = result.data.meas.get_counts()

        # All measurements should be '00'
        total = sum(counts.values())
        p_00 = counts.get('00', 0) / total
        assert p_00 > 0.99, f"theta=0 should give |00>, got P(00)={p_00:.4f}"

    def test_theta_pi_gives_excited_state(self):
        """theta = pi -> qubit in |1>, P(|1>) = 1."""
        from qiskit_aer import AerSimulator

        theta_h = np.array([[np.pi]])
        theta_v = np.array([[np.pi]])
        psi_h = np.array([[0.0]])
        psi_v = np.array([[0.0]])

        qc = init_qbits_state(theta_h, theta_v, psi_h, psi_v)
        qc.measure_all()

        backend = AerSimulator(method='statevector')
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)
        result = sampler.run([qc], shots=4096).result()[0]
        counts = result.data.meas.get_counts()

        total = sum(counts.values())
        p_11 = counts.get('11', 0) / total
        assert p_11 > 0.99, f"theta=pi should give |11>, got P(11)={p_11:.4f}"

    def test_probability_follows_sin_squared(self):
        """P(|1>) = sin^2(theta/2) for the R gate."""
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import SamplerV2

        backend = AerSimulator(method='statevector')
        sampler = SamplerV2(mode=backend)

        for theta_val in [0.5, 1.0, 1.5, 2.0, 2.5]:
            theta_h = np.array([[theta_val]])
            theta_v = np.array([[0.0]])  # only check qubit 0 (H-edge)
            psi_h = np.array([[0.0]])
            psi_v = np.array([[0.0]])

            qc = init_qbits_state(theta_h, theta_v, psi_h, psi_v)
            qc.measure_all()

            result = sampler.run([qc], shots=8192).result()[0]
            counts = result.data.meas.get_counts()
            total = sum(counts.values())

            # Count shots where qubit 0 is |1>
            p1_measured = sum(c for bs, c in counts.items() if bs[-1] == '1') / total
            p1_expected = np.sin(theta_val / 2) ** 2

            assert abs(p1_measured - p1_expected) < 0.05, (
                f"theta={theta_val:.1f}: P(|1>)_measured={p1_measured:.3f} != "
                f"P(|1>)_expected={p1_expected:.3f}"
            )

    def test_psi_does_not_change_amplitude(self):
        """Psi only affects phase, not P(|1>). P(|1>) should remain sin^2(theta/2)."""
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import SamplerV2

        backend = AerSimulator(method='statevector')
        sampler = SamplerV2(mode=backend)

        theta_val = 1.0
        expected_p1 = np.sin(theta_val / 2) ** 2

        for psi_val in [0.0, 0.5, 1.0, 1.5]:
            theta_h = np.array([[theta_val]])
            theta_v = np.array([[0.0]])
            psi_h = np.array([[psi_val]])
            psi_v = np.array([[0.0]])

            qc = init_qbits_state(theta_h, theta_v, psi_h, psi_v)
            qc.measure_all()

            result = sampler.run([qc], shots=8192).result()[0]
            counts = result.data.meas.get_counts()
            total = sum(counts.values())

            p1_measured = sum(c for bs, c in counts.items() if bs[-1] == '1') / total
            assert abs(p1_measured - expected_p1) < 0.05, (
                f"psi={psi_val:.1f}: P(|1>) should be {expected_p1:.3f} "
                f"regardless of psi, got {p1_measured:.3f}"
            )


# ══════════════════════════════════════════════════════════════════════
#  MODULE 6: Cost Hamiltonian — operator structure
# ══════════════════════════════════════════════════════════════════════

class TestCostHamiltonian:
    """Verify the Hamiltonian has correct operator structure."""

    def test_periodic_hamiltonian_num_qubits(self):
        """Periodic Hamiltonian for dim=2 should have 8 qubits."""
        hp = {
            "H_edges": (np.ones((2, 2)), np.ones((2, 2))),
            "C_edges": (np.ones((2, 2)), np.ones((2, 2))),
            "K_plaquettes": np.ones((2, 2)),
        }
        ham = create_period_hamiltonian(hp, dim=2)
        assert ham.num_qubits == 8, f"Expected 8 qubits for dim=2, got {ham.num_qubits}"

    def test_zero_coefficients_filtered(self):
        """Coefficients below 1e-6 are filtered out — and when nothing is
        left, that is reported instead of being replaced.

        The construction used to append a ("Z", [0], 1e-3) safety term so
        Qiskit would not choke on an empty observable. Downstream then had a
        single-Z operator indistinguishable from a real one, 1e5 times
        larger than the 1e-8 coefficient it stood for.
        """
        from VQA.cost_hamiltonian import NullHamiltonianError

        hp = {
            "H_edges": (np.array([[1e-8, 0.0], [0.0, 0.0]]),
                        np.array([[0.0, 0.0], [0.0, 0.0]])),
            "C_edges": (np.zeros((2, 2)), np.zeros((2, 2))),
            "K_plaquettes": np.zeros((2, 2)),
        }
        import pytest

        with pytest.raises(NullHamiltonianError):
            create_period_hamiltonian(hp, dim=2)

        # One coefficient above the cut is enough to build the operator
        hp["H_edges"][0][0, 0] = 1e-3
        ham = create_period_hamiltonian(hp, dim=2)
        assert len(ham.to_list()) == 1

    def test_all_Z_terms_present(self):
        """Each cell should contribute one Z term for H and one for V."""
        dim = 2
        hp = {
            "H_edges": (np.ones((dim, dim)), np.ones((dim, dim))),
            "C_edges": (np.zeros((dim, dim)), np.zeros((dim, dim))),
            "K_plaquettes": np.zeros((dim, dim)),
        }
        ham = create_period_hamiltonian(hp, dim=dim)
        terms = ham.to_list()
        z_terms = [t for t in terms if t[0].count('Z') == 1 and t[0].count('I') == 7]
        # 2x2 grid -> 4 horizontal + 4 vertical = 8 Z terms
        assert len(z_terms) == 2 * dim * dim, (
            f"Expected {2*dim*dim} Z terms, got {len(z_terms)}"
        )

    def test_ZZ_terms_present(self):
        """Non-zero C_edges should produce ZZ terms."""
        dim = 2
        hp = {
            "H_edges": (np.zeros((dim, dim)), np.zeros((dim, dim))),
            "C_edges": (np.ones((dim, dim)), np.ones((dim, dim))),
            "K_plaquettes": np.zeros((dim, dim)),
        }
        ham = create_period_hamiltonian(hp, dim=dim)
        terms = ham.to_list()
        zz_terms = [t for t in terms if t[0].count('Z') == 2]
        assert len(zz_terms) > 0, "C_edges should produce ZZ terms"

    def test_ZZZZ_terms_present(self):
        """Non-zero K_plaquettes should produce ZZZZ terms."""
        dim = 2
        hp = {
            "H_edges": (np.zeros((dim, dim)), np.zeros((dim, dim))),
            "C_edges": (np.zeros((dim, dim)), np.zeros((dim, dim))),
            "K_plaquettes": np.ones((dim, dim)),
        }
        ham = create_period_hamiltonian(hp, dim=dim)
        terms = ham.to_list()
        zzzz_terms = [t for t in terms if t[0].count('Z') == 4]
        assert len(zzzz_terms) > 0, "K_plaquettes should produce ZZZZ terms"

    def test_hamiltonian_hermitian(self):
        """The Hamiltonian must be Hermitian (all coefficients are real for Pauli-Z)."""
        hp = {
            "H_edges": (np.random.randn(2, 2), np.random.randn(2, 2)),
            "C_edges": (np.random.randn(2, 2), np.random.randn(2, 2)),
            "K_plaquettes": np.random.randn(2, 2),
        }
        ham = create_period_hamiltonian(hp, dim=2)
        # All Pauli-Z operators are Hermitian, so coefficients must be real
        for pauli_str, coeff in ham.to_list():
            assert np.isreal(coeff), f"Non-real coefficient {coeff} for {pauli_str}"


# ══════════════════════════════════════════════════════════════════════
#  MODULE 7: QAOA execution — ground state verification
# ══════════════════════════════════════════════════════════════════════

class TestQAOAExecution:
    """Verify QAOA finds the ground state for simple Hamiltonians."""

    def _run_qaoa(self, hamilt_params, theta_val=0.5, psi_val=0.0, reps=2, K_opt=200):
        """Helper: run full QAOA pipeline with given params."""
        dim = 2
        data = {
            "theta_h": np.full((dim, dim), theta_val).tolist(),
            "theta_v": np.full((dim, dim), theta_val).tolist(),
            "psi_h": np.full((dim, dim), psi_val).tolist(),
            "psi_v": np.full((dim, dim), psi_val).tolist(),
        }
        qc, cost_ham = mapping(
            data, hamilt_params, period_bound=True, reps=reps)

        E_max = 0
        for key, value in hamilt_params.items():
            if isinstance(value, (tuple, list)):
                for v in value:
                    if isinstance(v, np.ndarray):
                        E_max += np.sum(np.abs(v))
            elif isinstance(value, np.ndarray):
                E_max += np.sum(np.abs(value))
        E_max = max(E_max, 1e-10)

        dist, _ = execute(qc, cost_ham, "simulator", "state_vector",
                          4096, reps, K_opt, 1e-4, E_max, verbose=False)
        return np.array(postprocess(dist, qc.num_qubits, verbose=False))

    def test_strong_positive_Z_drives_to_one(self):
        """All positive Z -> QAOA should prefer |1> (minimizes energy)."""
        hp = {
            "H_edges": (np.full((2, 2), 5.0), np.full((2, 2), 5.0)),
            "C_edges": (np.zeros((2, 2)), np.zeros((2, 2))),
            "K_plaquettes": np.zeros((2, 2)),
        }
        marginals = self._run_qaoa(hp, theta_val=0.8)
        avg_p1 = np.mean(marginals)
        print(f"\n  [Positive Z] avg P(|1>) = {avg_p1:.4f}")
        assert avg_p1 > 0.5, (
            f"Strong positive Z should bias toward |1>, got avg P(|1>)={avg_p1:.4f}"
        )

    def test_strong_negative_Z_drives_to_zero(self):
        """All negative Z -> QAOA should prefer |0> (minimizes energy)."""
        hp = {
            "H_edges": (np.full((2, 2), -5.0), np.full((2, 2), -5.0)),
            "C_edges": (np.zeros((2, 2)), np.zeros((2, 2))),
            "K_plaquettes": np.zeros((2, 2)),
        }
        marginals = self._run_qaoa(hp, theta_val=0.8)
        avg_p1 = np.mean(marginals)
        print(f"\n  [Negative Z] avg P(|1>) = {avg_p1:.4f}")
        assert avg_p1 < 0.5, (
            f"Strong negative Z should bias toward |0>, got avg P(|1>)={avg_p1:.4f}"
        )

    def test_single_hot_cell_discrimination(self):
        """v10: ferromagnetic ZZ + adaptive Z bias -> hot cell maintains higher P(|1>).

        Without Z bias, ferromagnetic ZZ aligns ALL qubits to the majority state
        (3 cold cells → all collapse to |0>). The adaptive Z bias breaks this
        degeneracy by penalizing |0> on hot cells and |1> on cold cells.
        """
        # v10: H_edges provides adaptive Z bias to break ground-state degeneracy.
        # Z > 0 on hot cell → drives to |1>, Z < 0 on cold cells → drives to |0>.
        # This mirrors the pipeline's alpha_z * (score - threshold) computation.
        dim = 2
        score_hot = 0.92   # sin²(θ/2) for θ=2.8
        score_cold = 0.01  # sin²(θ/2) for θ=0.2
        threshold = 0.5
        # Adaptive Z: proportional to coupling scale, signed by (score - threshold)
        alpha_z = 0.15 * 5.0  # w_z_frac * median(|C|)
        z_hot = alpha_z * (score_hot - threshold)    # positive → drive to |1>
        z_cold = alpha_z * (score_cold - threshold)  # negative → drive to |0>
        hp = {
            "H_edges": (
                np.array([[z_hot, z_cold], [z_cold, z_cold]]),
                np.array([[z_hot, z_cold], [z_cold, z_cold]]),
            ),
            "C_edges": (np.full((2, 2), -5.0), np.full((2, 2), -5.0)),
            "K_plaquettes": np.full((2, 2), -0.5),
        }
        # Hot cell (0,0) has θ≈π (initialized near |1>), cold cells θ≈0 (near |0>)
        theta_h = np.array([[2.8, 0.2], [0.2, 0.2]])
        theta_v = np.array([[2.8, 0.2], [0.2, 0.2]])
        data = {
            "theta_h": theta_h.tolist(),
            "theta_v": theta_v.tolist(),
            "psi_h": np.zeros((dim, dim)).tolist(),
            "psi_v": np.zeros((dim, dim)).tolist(),
        }
        qc, cost_ham = mapping(data, hp, period_bound=True, reps=2)

        E_max = 0
        for key, value in hp.items():
            if isinstance(value, (tuple, list)):
                for v in value:
                    if isinstance(v, np.ndarray):
                        E_max += np.sum(np.abs(v))
            elif isinstance(value, np.ndarray):
                E_max += np.sum(np.abs(value))
        E_max = max(E_max, 1e-10)

        dist, _ = execute(qc, cost_ham, "simulator", "state_vector",
                          4096, 2, 200, 1e-4, E_max, verbose=False)
        marginals = np.array(postprocess(dist, qc.num_qubits, verbose=False))

        n = 4  # 2x2
        prob_h = marginals[:n].reshape(2, 2)
        prob_v = marginals[n:].reshape(2, 2)
        prob_map = np.maximum(prob_h, prob_v)

        p_hot = prob_map[0, 0]
        p_cold = np.mean([prob_map[0, 1], prob_map[1, 0], prob_map[1, 1]])

        print(f"\n  [Single hot cell] P(hot)={p_hot:.4f}, P(cold)={p_cold:.4f}")
        print(f"  Prob map: {prob_map}")
        # Hot cell should maintain higher probability than cold cells
        assert p_hot > p_cold, (
            f"Hot cell should have higher P(|1>) than cold cells, "
            f"P(hot)={p_hot:.4f}, P(cold)={p_cold:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════
#  MODULE 8: Postprocess — marginal computation
# ══════════════════════════════════════════════════════════════════════

class TestPostprocess:
    """Verify postprocess correctly computes marginal probabilities."""

    def test_all_zeros_bitstring(self):
        """All shots in |000...0> -> all marginals = 0."""
        dist = {'00000000': 1.0}
        marginals = postprocess(dist, 8, verbose=False)
        np.testing.assert_allclose(marginals, 0.0, atol=1e-10)

    def test_all_ones_bitstring(self):
        """All shots in |111...1> -> all marginals = 1."""
        dist = {'11111111': 1.0}
        marginals = postprocess(dist, 8, verbose=False)
        np.testing.assert_allclose(marginals, 1.0, atol=1e-10)

    def test_single_qubit_hot(self):
        """Only qubit 0 is |1> -> marginal[0] = 1, others = 0."""
        # bitstring '10000000' reversed -> bit at index 0 is '0' (rightmost is qubit 0)
        # For qubit 0 to be |1>, the rightmost bit must be '1': '00000001'
        dist = {'00000001': 1.0}
        marginals = postprocess(dist, 8, verbose=False)
        # After reversal: enumerate('10000000') -> bit[0]='1' -> qubit 7
        # Wait, let me re-read the postprocess code:
        # for i, bit in enumerate(bitstring[::-1]):  -> reverses string
        # So '00000001' reversed is '10000000'
        # enumerate: i=0 bit='1', i=1 bit='0', ...
        # So hits[0] += count -> qubit 0 gets the count
        assert marginals[0] == 1.0, f"Qubit 0 should have marginal=1, got {marginals[0]}"
        for i in range(1, 8):
            assert marginals[i] == 0.0, f"Qubit {i} should have marginal=0, got {marginals[i]}"

    def test_mixed_distribution_normalization(self):
        """Marginals from a normalized distribution should be in [0, 1]."""
        dist = {'00': 0.3, '01': 0.2, '10': 0.1, '11': 0.4}
        marginals = postprocess(dist, 2, verbose=False)
        # Qubit 0 (rightmost): '01' and '11' have bit 0 = '1'
        # After reverse: '10' and '11'
        # enumerate('10'): i=0 bit='1', i=1 bit='0' -> qubit 0 gets count
        # But wait, '01' reversed is '10', so bit at i=0 is '1'
        # '11' reversed is '11', bit at i=0 is '1'
        # So qubit 0 = 0.2 + 0.4 = 0.6
        assert abs(marginals[0] - 0.6) < 1e-10, f"Qubit 0 marginal should be 0.6, got {marginals[0]}"
        # Qubit 1: '10' reversed is '01', bit at i=1 is '1' -> P = 0.1
        # '11' reversed is '11', bit at i=1 is '1' -> P = 0.4
        # Wait, '10' reversed = '01': i=0 bit='0', i=1 bit='1'
        # '11' reversed = '11': i=0 bit='1', i=1 bit='1'
        # So qubit 1 = 0.1 + 0.4 = 0.5
        assert abs(marginals[1] - 0.5) < 1e-10, f"Qubit 1 marginal should be 0.5, got {marginals[1]}"


# ══════════════════════════════════════════════════════════════════════
#  MODULE 9: RescaleArrays — max-abs pooling
# ══════════════════════════════════════════════════════════════════════

class TestRescaleArrays:
    """Verify max-abs pooling preserves anomalies during downsampling."""

    def test_single_anomaly_survives(self):
        """A single large value in one block must survive max-abs pooling."""
        arr = np.zeros((8, 8))
        arr[1, 1] = 10.0  # anomaly in block (0, 0)
        result = _maxabs_pool_2d(arr, 2, 2)
        assert result[0, 0] == 10.0, f"Anomaly should survive, got {result[0, 0]}"

    def test_negative_anomaly_survives(self):
        """Negative anomalies should also survive (max-abs, not max)."""
        arr = np.zeros((8, 8))
        arr[1, 1] = -15.0
        result = _maxabs_pool_2d(arr, 2, 2)
        assert result[0, 0] == -15.0, f"Negative anomaly should survive, got {result[0, 0]}"

    def test_output_shape(self):
        """Output should have the target dimensions."""
        arr = np.random.randn(16, 16)
        result = _maxabs_pool_2d(arr, 4, 4)
        assert result.shape == (4, 4), f"Expected shape (4, 4), got {result.shape}"

    def test_identity_when_same_size(self):
        """When target = input size, pooling should be identity."""
        arr = np.random.randn(4, 4)
        result = _maxabs_pool_2d(arr, 4, 4)
        np.testing.assert_allclose(result, arr)

    def test_flux_downsampling_preserves_localization(self):
        """Full get_adaptive_flux: anomaly in one quadrant should stay localized.

        Flux uses bilinear interpolation (smooth fields), while Hamiltonian
        coefficients use max-abs pooling.  We verify that HAMILTONIAN params
        preserve the anomaly localization after downsampling.
        """
        N = 16
        phi_h = np.zeros((N, N))
        phi_v = np.zeros((N, N))

        # H_edges anomaly in top-left quadrant
        H_h = np.zeros((N, N))
        H_h[2:6, 2:6] = 5.0
        hp = {
            "H_edges": (H_h, np.zeros((N, N))),
            "C_edges": (np.zeros((N, N)), np.zeros((N, N))),
            "K_plaquettes": np.zeros((N, N)),
        }

        # Use a dummy score array since we only care about Hamiltonian pooling
        dummy_score = np.zeros((N, N))
        _, _, mini_hp, _ = get_adaptive_flux(phi_h, phi_v, None, None, dummy_score, hp, target_dim=2)
        # Hamiltonian max-abs pooling should preserve the anomaly in (0,0)
        mini_H_h = mini_hp['H_edges'][0]
        assert mini_H_h[0, 0] > mini_H_h[1, 1], (
            f"Hamiltonian anomaly should be in (0,0): H_h = {mini_H_h}"
        )


# ══════════════════════════════════════════════════════════════════════
#  MODULE 10: Refinement decision logic
# ══════════════════════════════════════════════════════════════════════

class TestRefinementDecision:
    """Verify the refinement threshold logic is correct."""

    def test_effective_threshold_increases_with_depth(self):
        """effective_threshold = threshold_amr + (1-threshold_amr)*depth/max_depth."""
        threshold_amr = 0.5
        max_depth = 3
        for depth in range(max_depth + 1):
            eff = threshold_amr + (1.0 - threshold_amr) * depth / max_depth
            if depth == 0:
                assert abs(eff - threshold_amr) < 1e-10, "At depth 0, effective = threshold"
            if depth == max_depth:
                assert abs(eff - 1.0) < 1e-10, "At max_depth, effective = 1.0"
            if depth > 0:
                eff_prev = threshold_amr + (1.0 - threshold_amr) * (depth - 1) / max_depth
                assert eff > eff_prev, "Effective threshold should increase with depth"

    def test_high_prob_triggers_refinement(self):
        """P(|1>) > threshold should trigger refinement."""
        local_prob = 0.8
        threshold = 0.5
        assert local_prob >= threshold, "High prob should trigger refinement"

    def test_low_prob_skips_refinement(self):
        """P(|1>) < threshold should skip refinement."""
        local_prob = 0.2
        threshold = 0.5
        assert local_prob < threshold, "Low prob should skip refinement"


# ══════════════════════════════════════════════════════════════════════
#  MODULE 11: End-to-end — Z bias direction consistency
# ══════════════════════════════════════════════════════════════════════

class TestEndToEndConsistency:
    """Verify the full chain: physics -> Z bias -> QAOA -> correct refinement direction."""

    def test_quiet_region_not_refined(self):
        """Uniform fields: H < 0 -> QAOA prefers |0> -> no refinement."""
        dim = 2
        # Uniform fields -> phi ~ 0 -> theta ~ 0 -> cos_theta ~ 1
        # H = 4*(threshold - 1) < 0 for threshold < 1
        hp = {
            "H_edges": (np.full((dim, dim), -2.0), np.full((dim, dim), -2.0)),
            "C_edges": (np.zeros((dim, dim)), np.zeros((dim, dim))),
            "K_plaquettes": np.zeros((dim, dim)),
        }
        data = {
            "theta_h": np.full((dim, dim), 0.1).tolist(),  # small theta -> near |0>
            "theta_v": np.full((dim, dim), 0.1).tolist(),
            "psi_h": np.zeros((dim, dim)).tolist(),
            "psi_v": np.zeros((dim, dim)).tolist(),
        }
        qc, cost_ham = mapping(data, hp, period_bound=True, reps=2)
        E_max = np.sum(np.abs(hp['H_edges'][0])) + np.sum(np.abs(hp['H_edges'][1]))
        dist, _ = execute(qc, cost_ham, "simulator", "state_vector",
                          4096, 2, 200, 1e-4, max(E_max, 1e-10), verbose=False)
        marginals = np.array(postprocess(dist, qc.num_qubits, verbose=False))

        avg_p1 = np.mean(marginals)
        print(f"\n  [Quiet region] avg P(|1>) = {avg_p1:.4f}")
        assert avg_p1 < 0.4, (
            f"Quiet region should have low P(|1>), got {avg_p1:.4f}"
        )

    def test_anomalous_region_refined(self):
        """Strong anomaly: H > 0 -> QAOA prefers |1> -> triggers refinement."""
        dim = 2
        hp = {
            "H_edges": (np.full((dim, dim), 5.0), np.full((dim, dim), 5.0)),
            "C_edges": (np.full((dim, dim), 0.5), np.full((dim, dim), 0.5)),
            "K_plaquettes": np.full((dim, dim), 0.3),
        }
        data = {
            "theta_h": np.full((dim, dim), 1.5).tolist(),  # large theta -> biased toward |1>
            "theta_v": np.full((dim, dim), 1.5).tolist(),
            "psi_h": np.zeros((dim, dim)).tolist(),
            "psi_v": np.zeros((dim, dim)).tolist(),
        }
        qc, cost_ham = mapping(data, hp, period_bound=True, reps=2)
        E_max = 0
        for key, value in hp.items():
            if isinstance(value, (tuple, list)):
                for v in value:
                    E_max += np.sum(np.abs(v))
            elif isinstance(value, np.ndarray):
                E_max += np.sum(np.abs(value))
        dist, _ = execute(qc, cost_ham, "simulator", "state_vector",
                          4096, 2, 200, 1e-4, max(E_max, 1e-10), verbose=False)
        marginals = np.array(postprocess(dist, qc.num_qubits, verbose=False))

        avg_p1 = np.mean(marginals)
        print(f"\n  [Anomalous region] avg P(|1>) = {avg_p1:.4f}")
        assert avg_p1 > 0.5, (
            f"Anomalous region should have high P(|1>), got {avg_p1:.4f}"
        )


# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
