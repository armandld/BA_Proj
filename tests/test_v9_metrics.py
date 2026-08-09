"""
V9 Metrics — Tier 0 unit-level validation tests for the v9 Hamiltonian.

These tests validate structural properties of v9 independently of trained
hyperparameters. They should pass for ANY valid configuration and serve
as a sanity check before running expensive training.

Tests:
  Tier0A: Coefficient survival — ZZ/ZZZZ stay active across timesteps
  Tier0B: Ground state structure — all-|0⟩ energy = sum of all (≤0) coefficients
  Tier0C: Classical init fidelity — p=0 output equals classical score
  Tier0D: Non-trivial correction — p>0 actually modifies the classical score

Run with:
    cd tests && python -m pytest test_v9_metrics.py -v
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from VQA.cost_hamiltonian import create_period_hamiltonian

# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

DIM = 2  # 2x2 core = 8 qubits (fast)


def _make_sim_and_fields(scenario='init_kelvin_helmholtz', N=64, n_steps=50):
    """Create a simulation and evolve it for n_steps to develop features."""
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=400, Rm=400)
    getattr(sim, scenario)()
    for _ in range(n_steps):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    return sim, sim.get_fluxes()


def _compute_hamiltonian_coefficients(sim, fields, **kwargs):
    """Compute v9 Hamiltonian coefficients."""
    hm = PhysicalMapper(cs=1.0, eta_mhd=0.01, **kwargs)
    score = hm.physical_score(fields)
    hp = hm.compute_coefficients(sim, score, fields, 0.0)
    return hp, score


# ═══════════════════════════════════════════════════════════════════════
#  TIER 0A — COEFFICIENT SURVIVAL
# ═══════════════════════════════════════════════════════════════════════

class TestCoefficientSurvival(unittest.TestCase):
    """
    Verify that threshold-contrast keeps ZZ/ZZZZ coefficients active
    throughout simulation evolution (unlike Michelson which could drop to 0).
    """

    def test_coefficients_survive_kh_evolution(self):
        """C_edges and K_plaquettes stay non-zero through KH evolution."""
        grid = PeriodicGrid(resolution_N=64)
        sim = MHDSolver(grid, dt=1e-3, Re=400, Rm=400)
        sim.init_kelvin_helmholtz()

        hm = PhysicalMapper(cs=1.0, eta_mhd=0.01)
        c_max_history = []
        k_max_history = []

        for step in range(5):
            # Evolve 20 steps at a time
            for _ in range(20):
                sim.adapt_dt(cfl_target=0.4)
                sim.step_full(record_stats=False)

            fields = sim.get_fluxes()
            score = hm.physical_score(fields)
            hp = hm.compute_coefficients(sim, score, fields, 0.0)

            c_h, c_v = hp['C_edges']
            c_max = max(np.max(np.abs(c_h)), np.max(np.abs(c_v)))
            k_max = np.max(np.abs(hp['K_plaquettes']))

            c_max_history.append(c_max)
            k_max_history.append(k_max)

        print(f"\n[SURVIVAL-KH] C_edges max over time: {c_max_history}")
        print(f"[SURVIVAL-KH] K_plaq  max over time: {k_max_history}")

        # At least one of ZZ or ZZZZ should be non-zero at every step
        for i, (c, k) in enumerate(zip(c_max_history, k_max_history)):
            self.assertGreater(
                max(c, k), 1e-6,
                f"Step {i}: both C_edges and K_plaquettes are zero — "
                "threshold-contrast killed all signal"
            )

    def test_orszag_tang_zz_dies_while_zzzz_survives(self):
        """
        On Orszag-Tang the ZZ coupling does NOT survive, and the ZZZZ
        plaquettes do. The asymmetry is structural: the Gaussian
        uncertainty window multiplies C_edges only.

        On this snapshot the score spans [0.51, 0.87] with threshold_amr
        = 0, so every site is many sigma away from the boundary and the
        window is ~1e-48. Re-running with a wide sigma (window ~ 1)
        restores C_edges to O(60) while leaving K_plaquettes bit-identical
        — which is what makes the attribution airtight.
        """
        sim, fields = _make_sim_and_fields('init_orszag_tang', N=64, n_steps=30)

        hp, score = _compute_hamiltonian_coefficients(sim, fields)
        c_h, c_v = hp['C_edges']
        c_max = max(np.max(np.abs(c_h)), np.max(np.abs(c_v)))
        k_max = np.max(np.abs(hp['K_plaquettes']))

        # counterfactual: same fields, window effectively disabled
        hp_open, _ = _compute_hamiltonian_coefficients(sim, fields, sigma=10.0)
        c_h_o, c_v_o = hp_open['C_edges']
        c_max_open = max(np.max(np.abs(c_h_o)), np.max(np.abs(c_v_o)))
        k_max_open = np.max(np.abs(hp_open['K_plaquettes']))

        print(f"\n[SURVIVAL-OT] score range     : "
              f"[{score.min():.4f}, {score.max():.4f}]")
        print(f"[SURVIVAL-OT] max|C| sigma=0.05: {c_max:.6e}")
        print(f"[SURVIVAL-OT] max|C| sigma=10  : {c_max_open:.6e}")
        print(f"[SURVIVAL-OT] max|K| sigma=0.05: {k_max:.6e}")
        print(f"[SURVIVAL-OT] max|K| sigma=10  : {k_max_open:.6e}")

        self.assertLess(c_max, 1e-30,
                        "ZZ is expected to be annihilated at the deployed "
                        "sigma — the recorded V1 behaviour has changed")
        self.assertGreater(c_max_open, 1e-3,
                           "with the window open the ZZ coupling must return")
        self.assertGreater(k_max, 1e-3,
                           "ZZZZ is not windowed and must survive")
        self.assertAlmostEqual(k_max, k_max_open, places=12,
                               msg="sigma must not touch K_plaquettes — if it "
                                   "does, the ZZ/ZZZZ asymmetry is not clean")


# ═══════════════════════════════════════════════════════════════════════
#  TIER 0B — GROUND STATE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

class TestGroundStateStructure(unittest.TestCase):
    """
    Verify that the ferromagnetic Hamiltonian has all-|0⟩ as its
    ground state (sum of all coefficients ≤ 0).
    """

    def _all_zero_energy(self, hamiltonian):
        """Energy of all-|0⟩: every Z → +1, ZZ → +1, ZZZZ → +1."""
        energy = 0.0
        for pauli_label, coeff in hamiltonian.to_list():
            energy += np.real(coeff)
        return energy

    def test_ground_state_is_all_zero(self):
        """For ferromagnetic H, E(all-|0⟩) = sum(coefficients) ≤ 0."""
        N = DIM
        fields = {
            'vx': np.array([[1.5, -0.5], [-1.0, 0.0]], dtype=float),
            'vy': np.array([[0.0, 0.5], [-0.5, 0.0]], dtype=float),
            'Bx': np.zeros((N, N)),
            'By': np.zeros((N, N)),
            'Jz': np.zeros((N, N)),
        }

        hm = PhysicalMapper(cs=1.0, eta_mhd=0.01)
        grid = PeriodicGrid(N)
        sim = MHDSolver(grid)
        score = hm.physical_score(fields)
        hp = hm.compute_coefficients(sim, score, fields, 0.0)

        hamiltonian = create_period_hamiltonian(hp, N)
        E0 = self._all_zero_energy(hamiltonian)

        print(f"\n[GROUND-STATE] E(all-|0⟩) = {E0:.4f}")
        self.assertLessEqual(E0, 0.0,
                             "Ferromagnetic Hamiltonian: E(all-|0⟩) must be ≤ 0")

    def test_stronger_anomaly_more_negative_energy(self):
        """Stronger fields should produce more negative ground state energy."""
        N = DIM
        grid = PeriodicGrid(N)
        sim = MHDSolver(grid)
        hm = PhysicalMapper(cs=1.0, eta_mhd=0.01)

        energies = []
        for amplitude in [0.1, 0.5, 1.0, 2.0]:
            fields = {
                'vx': np.array([[amplitude, amplitude],
                                [-amplitude, -amplitude]], dtype=float),
                'vy': np.zeros((N, N)),
                'Bx': np.zeros((N, N)),
                'By': np.zeros((N, N)),
                'Jz': np.zeros((N, N)),
            }
            score = hm.physical_score(fields)
            hp = hm.compute_coefficients(sim, score, fields, 0.0)
            H = create_period_hamiltonian(hp, N)
            E = self._all_zero_energy(H)
            energies.append(E)

        print(f"\n[GROUND-SCALE] Energies: {energies}")

        # Energy should become more negative with stronger fields
        for i in range(len(energies) - 1):
            self.assertLessEqual(
                energies[i + 1], energies[i] + 1e-6,
                f"E(amplitude={[0.1, 0.5, 1.0, 2.0][i+1]}) should be ≤ "
                f"E(amplitude={[0.1, 0.5, 1.0, 2.0][i]})"
            )


# ═══════════════════════════════════════════════════════════════════════
#  TIER 0C — H_EDGES ADAPTIVE-Z VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

class TestHEdgesAdaptiveZ(unittest.TestCase):
    """Verify H_edges uses adaptive Z weight proportional to max(|C|, |K|).

    For quiet (zero) fields, max(|C|, |K|) = 0 so H_edges = 0.
    For active fields, H_edges may be non-zero but must remain subordinate
    to the dominant interaction term (|H| < |C| or |H| < |K|).
    """

    def test_h_edges_zero_on_quiet_fields(self):
        """H_edges = 0 for quiet fields (adaptive alpha scales with max(|C|,|K|) = 0)."""
        N = DIM
        grid = PeriodicGrid(N)
        sim = MHDSolver(grid)
        hm = PhysicalMapper(cs=1.0, eta_mhd=0.01)

        fields = {'vx': np.zeros((N, N)), 'vy': np.zeros((N, N)),
                  'Bx': np.zeros((N, N)), 'By': np.zeros((N, N)),
                  'Jz': np.zeros((N, N))}

        score = hm.physical_score(fields)
        hp = hm.compute_coefficients(sim, score, fields, 0.0)
        H_h, H_v = hp['H_edges']
        h_max = max(np.max(np.abs(H_h)), np.max(np.abs(H_v)))
        print(f"[H_EDGES-ZERO] quiet: max |H_edges| = {h_max}")
        self.assertEqual(h_max, 0.0,
                         "H_edges must be zero for quiet fields (alpha ~ max(|C|,|K|) = 0)")

    def test_h_edges_subordinate_on_active_fields(self):
        """H_edges is subordinate to max(|C|, |K|) for active fields (adaptive Z)."""
        N = DIM
        grid = PeriodicGrid(N)
        sim = MHDSolver(grid)
        hm = PhysicalMapper(cs=1.0, eta_mhd=0.01)

        configs = [
            ("shear", {'vx': np.array([[2, 2], [-2, -2]], dtype=float),
                       'vy': np.zeros((N, N)), 'Bx': np.zeros((N, N)),
                       'By': np.zeros((N, N)), 'Jz': np.zeros((N, N))}),
            ("supersonic", {'vx': np.full((N, N), 3.0),
                            'vy': np.zeros((N, N)), 'Bx': np.zeros((N, N)),
                            'By': np.zeros((N, N)), 'Jz': np.zeros((N, N))}),
        ]

        for name, fields in configs:
            score = hm.physical_score(fields)
            hp = hm.compute_coefficients(sim, score, fields, 0.0)
            H_h, H_v = hp['H_edges']
            C_h, C_v = hp['C_edges']
            h_max = max(np.max(np.abs(H_h)), np.max(np.abs(H_v)))
            c_max = max(np.max(np.abs(C_h)), np.max(np.abs(C_v)))
            # Also check K_plaquettes if present
            k_max = 0.0
            if 'K_plaquettes' in hp:
                k_max = np.max(np.abs(hp['K_plaquettes']))
            dominant = max(c_max, k_max)
            print(f"[H_EDGES-ADAPTIVE] {name}: max |H| = {h_max:.4f}, "
                  f"max |C| = {c_max:.4f}, max |K| = {k_max:.4f}")
            if dominant > 0:
                self.assertLess(h_max, dominant,
                                f"H_edges must be subordinate to max(|C|,|K|) for {name} fields")


# ═══════════════════════════════════════════════════════════════════════
#  TIER 0D — CORRECTION MAP STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

class TestCorrectionMapStructure(unittest.TestCase):
    """
    Structure of the correction δ(i,j) = P_QAOA(i,j) - score(i,j).

    The v9 design intent was that the ZZ coupling creates spatial
    correlations in the correction map, i.e. that the Hamiltonian carries
    information beyond the classical score used for θ initialisation.

    The tests below record what the implementation actually does: the
    Gaussian uncertainty window multiplying C_edges annihilates the ZZ
    coupling wherever the score sits away from threshold_amr, which is
    the generic case. The gradient signal is computed correctly and then
    multiplied by ~0. This is the mechanism, measured, not an accident of
    a particular field.
    """

    def test_zz_coupling_is_annihilated_by_the_uncertainty_window(self):
        """
        On a 2x2 periodic grid with a sharp velocity boundary, the
        gradient coupling is real and O(48). The uncertainty window
        reduces it to O(1e-42).

        Setup: score is spatially uniform at 0.5 (all indicators wrap
        around on 2x2), threshold_amr = 0, sigma = 0.05. The window is
        exp(-((score - thr)/sigma)^2) = exp(-100) = 3.7e-44.

        Three assertions:
          1. the delivered coupling is numerically dead;
          2. the same call with thr = score (window = 1) returns O(48),
             so the signal exists and it is the window that removes it;
          3. the ratio of the two equals the window value exactly, which
             attributes the annihilation to the window and to nothing else.
        """
        N = DIM
        # Sharp velocity boundary: top row fast, bottom row slow
        fields = {
            'vx': np.array([[2.0, 2.0], [0.0, 0.0]], dtype=float),
            'vy': np.zeros((N, N)),
            'Bx': np.zeros((N, N)),
            'By': np.zeros((N, N)),
            'Jz': np.zeros((N, N)),
        }

        hm = PhysicalMapper(cs=1.0, eta_mhd=0.01)
        grid = PeriodicGrid(N)
        sim = MHDSolver(grid)
        score = hm.physical_score(fields)

        self.assertAlmostEqual(float(score.min()), float(score.max()), places=12,
                               msg="2x2 periodic score must be uniform for "
                                   "this test to isolate the window")
        s = float(score.flat[0])

        def max_abs_C(threshold_amr):
            C_h, C_v = hm.compute_coefficients(
                sim, score, fields, threshold_amr)['C_edges']
            return max(np.max(np.abs(C_h)), np.max(np.abs(C_v)))

        c_windowed = max_abs_C(0.0)      # deployed setting
        c_open = max_abs_C(s)            # window = exp(0) = 1
        window = np.exp(-((s - 0.0) / hm.sigma) ** 2)

        print(f"\n[WINDOW] score (uniform)      : {s}")
        print(f"[WINDOW] sigma                : {hm.sigma}")
        print(f"[WINDOW] max|C| at thr=0      : {c_windowed:.6e}")
        print(f"[WINDOW] max|C| at thr=score  : {c_open:.6e}")
        print(f"[WINDOW] exp(-((s-thr)/sig)^2): {window:.6e}")

        # 1. what the pipeline actually receives is numerically dead
        self.assertLess(c_windowed, 1e-30,
                        "the ZZ coupling delivered at threshold_amr=0 is not "
                        "dead — the recorded V1 behaviour has changed")
        # 2. the underlying gradient signal is present and large
        self.assertGreater(c_open, 1.0,
                           "with the window open the gradient coupling must "
                           "be O(1) or larger — otherwise the annihilation "
                           "is not attributable to the window")
        # 3. the window, and only the window, explains the gap
        self.assertAlmostEqual(c_windowed / c_open, window, delta=1e-6 * window,
                               msg="the suppression factor must equal the "
                                   "Gaussian window exactly")


# ═══════════════════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    unittest.main(verbosity=2)
