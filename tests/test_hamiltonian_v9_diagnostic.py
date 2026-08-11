"""
Diagnostic tests for Hamiltonian architecture:
  1. Z-terms use ADAPTIVE weight (fraction of max|C|,|K|) to break ground state degeneracy
  2. Threshold-relative contrast replaces Michelson for ZZ/ZZZZ
  3. θ = classical anchor, Z = degeneracy breaker, ZZ/ZZZZ = spatial correlations

These tests validate that:
  A. H_edges is non-zero and proportional to (score - threshold) × max(|C|,|K|)
  B. ZZ/ZZZZ signal survives in spatially uniform regions (Michelson would kill it)
  C. Noise immunity — sub-critical noise produces weak coefficients
  D. Information orthogonality — ZZ and ZZZZ respond to different physics
  E. Gradient layer scaling — more layers = more QAOA correction capacity

Run with:
    cd tests && python -m pytest test_hamiltonian_v9_diagnostic.py -v
"""

import sys, os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

N_TEST = 16       # Grid resolution for coefficient tests
NU = 1e-3
ETA = 1e-3
DX_16 = 2 * np.pi / N_TEST


def _make_fields(N, vx=0.0, vy=0.0, Bx=0.0, By=0.0, Jz=0.0):
    """Uniform MHD fields."""
    return {
        'vx': np.full((N, N), vx),
        'vy': np.full((N, N), vy),
        'Bx': np.full((N, N), Bx),
        'By': np.full((N, N), By),
        'Jz': np.full((N, N), Jz),
    }


def _make_solver(N):
    """Create a solver for coefficient computation."""
    grid = PeriodicGrid(N)
    return MHDSolver(grid)


def _make_mapper(**kwargs):
    """Create PhysicalMapper with test-friendly defaults."""
    defaults = dict(cs=1.0, nu=NU, eta_mhd=ETA, dx=DX_16,
                    gamma_hydro=0.5, gamma_mag=0.5, kappa=5.0, sigma=0.10,
                    w_z_frac=0.15)
    defaults.update(kwargs)
    return PhysicalMapper(**defaults)


# ═══════════════════════════════════════════════════════════════════════
#  TEST A: Z-terms use adaptive weight (fraction of max|C|,|K|)
# ═══════════════════════════════════════════════════════════════════════

class TestA_ZTermsAdaptive(unittest.TestCase):
    """
    Z-terms use adaptive weight α = w_z_frac × max(|C|, |K|).
    H_edges = α × (score − threshold_amr).
    This breaks ground state degeneracy while keeping Z smaller than ZZ.
    """

    def test_z_nonzero_when_zz_active(self):
        """H_edges > 0 when ZZ/ZZZZ coupling is active (global median > 0)."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper(w_z_frac=0.15)

        # Alternating rows → gradient at EVERY row boundary
        fields = _make_fields(N, Bx=0.5)
        for i in range(N):
            fields['vx'][i, :] = 5.0 if i % 2 == 0 else -5.0

        score = AngleMapper.classical_score(fields)
        threshold = 0.3
        result = hm.compute_coefficients(sim, score, fields, threshold)

        H_h, H_v = result['H_edges']
        C_h, C_v = result['C_edges']

        # ZZ should be active (alternating shear)
        C_max = max(np.max(np.abs(C_h)), np.max(np.abs(C_v)))
        self.assertGreater(C_max, 0.01, "ZZ must be active for this test")

        # Global alpha → Z should be non-zero everywhere score != threshold
        self.assertGreater(np.max(np.abs(H_h)), 1e-6,
                           msg="H must be non-zero when ZZ is active")

    def test_z_proportional_to_zz_scale(self):
        """alpha = w_z_frac × median(nonzero |C|,|K|), so H ≤ w_z_frac × max."""
        N = N_TEST
        sim = _make_solver(N)
        w_z_frac = 0.15
        hm = _make_mapper(w_z_frac=w_z_frac)

        # Alternating rows → ZZ active everywhere
        fields = _make_fields(N, Bx=0.5)
        for i in range(N):
            fields['vx'][i, :] = 5.0 if i % 2 == 0 else -5.0

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        H_h, _ = result['H_edges']
        C_h, C_v = result['C_edges']
        K = result['K_plaquettes']

        # median ≤ max, so max|H| ≤ w_z_frac × max(all |C|,|K|) still holds
        C_max = max(np.max(np.abs(C_h)), np.max(np.abs(C_v)),
                    np.max(np.abs(K)), 1e-10)
        self.assertLessEqual(np.max(np.abs(H_h)), w_z_frac * C_max * 1.0 + 1e-8,
                             msg="Z must be bounded by w_z_frac × max(|C|,|K|)")

    def test_z_zero_when_zz_zero(self):
        """H_edges = 0 when ZZ/ZZZZ are all zero (quiet fields)."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper(w_z_frac=0.15)

        fields = _make_fields(N, vx=0.0, vy=0.0, Bx=0.0, By=0.0)
        score = AngleMapper.classical_score(fields)

        result = hm.compute_coefficients(sim, score, fields, 0.3)
        H_h, H_v = result['H_edges']
        # With quiet fields, C and K are zero → α = w_z_frac × 0 = 0 → H = 0
        self.assertAlmostEqual(np.max(np.abs(H_h)), 0.0, places=8)
        self.assertAlmostEqual(np.max(np.abs(H_v)), 0.0, places=8)

    def test_z_sign_convention(self):
        """H > 0 where score > threshold (bias toward refinement)."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper(w_z_frac=0.15)

        # Alternating rows → ZZ active everywhere → global alpha > 0
        fields = _make_fields(N, Bx=0.5)
        for i in range(N):
            fields['vx'][i, :] = 5.0 if i % 2 == 0 else -5.0

        score = AngleMapper.classical_score(fields)
        threshold = 0.3
        result = hm.compute_coefficients(sim, score, fields, threshold)

        H_h, _ = result['H_edges']
        # Where score > threshold, H should be positive
        above = score > threshold
        if np.any(above):
            self.assertGreater(np.mean(H_h[above]), 0.0,
                               msg="H > 0 where score > threshold")

    def test_backward_compat_kwargs_ignored(self):
        """Old callers passing psi_h/psi_v should not crash."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, vx=1.0)
        score = AngleMapper.classical_score(fields)

        # Old call signature with psi — should be silently ignored
        result = hm.compute_coefficients(sim, score, fields, 0.3,
                                          psi_h=np.ones((N, N)),
                                          psi_v=np.ones((N, N)))
        self.assertIn('H_edges', result)


# ═══════════════════════════════════════════════════════════════════════
#  TEST B: ZZ/ZZZZ survive in spatially uniform regions
# ═══════════════════════════════════════════════════════════════════════

class TestB_ThresholdContrastSurvival(unittest.TestCase):
    """
    The old Michelson filter killed the signal when the domain was
    spatially uniform (val ≈ avg → Mic → 0). The new threshold-relative
    contrast should survive as long as val > critical threshold.
    """

    def test_uniform_shear_above_critical(self):
        """
        Uniform shear layer everywhere (same jump at every point).
        Old Michelson: mic = 0 (val = avg). New threshold-contrast: signal > 0.
        """
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper(sigma=0.10)

        # Every other row has different velocity → uniform gradient everywhere
        fields = _make_fields(N, Bx=0.5)
        for i in range(N):
            fields['vx'][i, :] = 3.0 if i % 2 == 0 else 0.0

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        C_h, C_v = result['C_edges']
        max_C = max(np.max(np.abs(C_h)), np.max(np.abs(C_v)))
        self.assertGreater(max_C, 0.01,
                           "ZZ coefficient should survive in uniform shear "
                           "(Michelson would have killed it)")

    def test_uniform_vorticity_above_critical(self):
        """
        Uniform vorticity everywhere. Old Michelson: 0. New: signal > 0.
        """
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper(beta_curl=0.5)

        # Solid body rotation: vx = -y, vy = x → uniform ωz
        # CONVENTION : AXIS_X = 0 dans grid.py, donc l'axe 0 porte x.
        # `np.mgrid[0:N, 0:N]` rend (axe0, axe1) : le premier est donc
        # X et le second Y, a l'inverse du nommage historique. Sous la
        # convention corrigee, le champ ci-dessous est une rotation
        # solide ; nomme dans l'autre sens, c'etait une deformation
        # pure, de vorticite exactement nulle.
        X, Y = np.mgrid[0:N, 0:N]
        fields = _make_fields(N, Bx=0.5)
        fields['vx'] = -3.0 * (Y - N/2) / N
        fields['vy'] = 3.0 * (X - N/2) / N

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        K = result['K_plaquettes']
        max_K = np.max(np.abs(K))
        self.assertGreater(max_K, 0.001,
                           "ZZZZ coefficient should survive in uniform vorticity "
                           "(Michelson would have killed it)")

    def test_zero_signal_below_critical(self):
        """
        Very weak fields (below all critical thresholds) → signal ≈ 0.
        The threshold-contrast correctly filters out sub-critical physics.
        """
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        # Tiny fields — well below critical thresholds
        fields = _make_fields(N, vx=1e-6, vy=1e-6, Bx=1e-6, By=1e-6)

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        C_h, C_v = result['C_edges']
        K = result['K_plaquettes']
        self.assertLess(np.max(np.abs(C_h)), 1e-3)
        self.assertLess(np.max(np.abs(C_v)), 1e-3)
        self.assertLess(np.max(np.abs(K)), 1e-3)


# ═══════════════════════════════════════════════════════════════════════
#  TEST C: Noise immunity — noise weaker than real anomalies
# ═══════════════════════════════════════════════════════════════════════

class TestC_NoiseImmunity(unittest.TestCase):
    """
    Random noise should produce weaker Hamiltonian coefficients than
    a genuine anomaly.
    """

    def test_noise_weaker_than_anomaly(self):
        """
        Gaussian noise ZZ/ZZZZ should be MUCH weaker than real shear layer.
        """
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()
        rng = np.random.RandomState(42)

        # Noise fields
        amp = 0.01
        fields_noise = {
            'vx': amp * rng.randn(N, N),
            'vy': amp * rng.randn(N, N),
            'Bx': amp * rng.randn(N, N),
            'By': amp * rng.randn(N, N),
            'Jz': amp * rng.randn(N, N),
        }
        score_noise = AngleMapper.classical_score(fields_noise)
        result_noise = hm.compute_coefficients(sim, score_noise, fields_noise, 0.3)

        # Real anomaly fields
        fields_anom = _make_fields(N, Bx=0.5)
        fields_anom['vx'][:N//2, :] = 3.0
        fields_anom['vx'][N//2:, :] = -3.0
        score_anom = AngleMapper.classical_score(fields_anom)
        result_anom = hm.compute_coefficients(sim, score_anom, fields_anom, 0.3)

        noise_C = max(np.max(np.abs(result_noise['C_edges'][0])),
                      np.max(np.abs(result_noise['C_edges'][1])))
        anom_C = max(np.max(np.abs(result_anom['C_edges'][0])),
                     np.max(np.abs(result_anom['C_edges'][1])))
        self.assertLess(noise_C, anom_C,
                        f"Noise ZZ ({noise_C:.3f}) should be weaker than "
                        f"anomaly ZZ ({anom_C:.3f})")


# ═══════════════════════════════════════════════════════════════════════
#  TEST D: Information orthogonality — ZZ and ZZZZ respond differently
# ═══════════════════════════════════════════════════════════════════════

class TestD_InformationOrthogonality(unittest.TestCase):
    """
    Verify that ZZ and ZZZZ respond to different physics:
    - ZZ responds to spatial gradients (shear layers, jumps)
    - ZZZZ responds to circulation (vortex cores, current sheets)
    """

    def test_shear_activates_zz(self):
        """Sharp velocity jump → ZZ fires."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, Bx=0.5)
        fields['vx'][:N//2, :] = 3.0
        fields['vx'][N//2:, :] = -3.0

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        C_h, C_v = result['C_edges']
        self.assertGreater(max(np.max(np.abs(C_h)), np.max(np.abs(C_v))), 0.1,
                           "ZZ should fire on shear layer")

    def test_vortex_activates_zzzz(self):
        """Solid rotation → ZZZZ fires (rotation-dominated Q > 0)."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        # Lamb-Oseen-like vortex
        Y, X = np.mgrid[0:N, 0:N]
        cx, cy = N/2, N/2
        r = np.sqrt((X - cx)**2 + (Y - cy)**2) + 1e-10
        r_core = N / 6
        v_theta = 5.0 * (1 - np.exp(-r**2 / r_core**2)) / r
        cos_t = -(Y - cy) / r
        sin_t = (X - cx) / r

        fields = _make_fields(N, Bx=0.5)
        fields['vx'] = v_theta * cos_t
        fields['vy'] = v_theta * sin_t

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        K = result['K_plaquettes']
        self.assertGreater(np.max(np.abs(K)), 0.01,
                           "ZZZZ should fire on vortex")

    def test_quiet_fields_no_coupling(self):
        """Very quiet fields → all coupling terms near zero."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, vx=0.01, vy=0.01, Bx=0.01, By=0.01)
        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        C_h, C_v = result['C_edges']
        K = result['K_plaquettes']
        self.assertLess(np.max(np.abs(C_h)), 0.1)
        self.assertLess(np.max(np.abs(K)), 0.01)


# ═══════════════════════════════════════════════════════════════════════
#  TEST E: Structural properties — Hamiltonian has correct form
# ═══════════════════════════════════════════════════════════════════════

class TestE_StructuralProperties(unittest.TestCase):
    """
    Verify structural properties of the v9 Hamiltonian.
    """

    def test_zz_is_ferromagnetic(self):
        """ZZ coefficients should be negative (ferromagnetic: neighbors align)."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, Bx=0.5)
        fields['vx'][:N//2, :] = 3.0
        fields['vx'][N//2:, :] = -3.0

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        C_h, C_v = result['C_edges']
        # Non-zero elements should be negative (ferromagnetic)
        nonzero_h = C_h[np.abs(C_h) > 1e-6]
        if len(nonzero_h) > 0:
            self.assertTrue(np.all(nonzero_h <= 0),
                            "ZZ should be ferromagnetic (C <= 0)")

    def test_zzzz_is_negative(self):
        """ZZZZ coefficients should be negative (even-parity)."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        Y, X = np.mgrid[0:N, 0:N]
        fields = _make_fields(N, Bx=0.5)
        fields['vx'] = -3.0 * (Y - N/2) / N
        fields['vy'] = 3.0 * (X - N/2) / N

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        K = result['K_plaquettes']
        nonzero_K = K[np.abs(K) > 1e-6]
        if len(nonzero_K) > 0:
            self.assertTrue(np.all(nonzero_K <= 0),
                            "ZZZZ should be negative (even-parity)")

    def test_z_subordinate_to_correlations(self):
        """
        Z bias should be a small fraction of the ZZ/ZZZZ energy scale.
        The multi-body terms must dominate the Hamiltonian.
        """
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper(w_z_frac=0.15)

        # Alternating rows → ZZ active everywhere
        fields = _make_fields(N, Bx=0.5)
        for i in range(N):
            fields['vx'][i, :] = 5.0 if i % 2 == 0 else -5.0

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        H_h, H_v = result['H_edges']
        C_h, C_v = result['C_edges']
        K = result['K_plaquettes']

        # ZZ and/or ZZZZ > 0 (multi-body terms exist)
        total_coupling = (np.sum(np.abs(C_h)) + np.sum(np.abs(C_v))
                          + np.sum(np.abs(K)))
        self.assertGreater(total_coupling, 0.1,
                           "Hamiltonian should have non-zero multi-body terms")

        # Z energy is much smaller than ZZ+ZZZZ energy
        total_z = np.sum(np.abs(H_h)) + np.sum(np.abs(H_v))
        self.assertLess(total_z, total_coupling,
                        "Z bias must be subordinate to multi-body terms")


if __name__ == '__main__':
    unittest.main()
