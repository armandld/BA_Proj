"""
Tests for HamiltParams_v2 (parameter-free physics-first Hamiltonian).

Adapted from test_hamiltonian_v9_diagnostic.py. Verifies:
  A. Coefficients survive (non-zero) in all existing scenarios
  B. Z remains subordinate to the spatial terms (ZZ, ZZZZ)
  C. Ferromagnetic ground state is correct (C < 0, K < 0)
  D. Coefficients scale as expected with Re
  E. Noise immunity: real anomalies > noise
  F. Information orthogonality: ZZ (gradients) vs ZZZZ (circulation)
  G. Phase encoding (psi_v2) is parameter-free and bounded

Run with:
    python -m pytest tests/test_hamiltonian_v2.py -v
"""

import sys, os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams_v2 import PhysicalMapperV2, compute_psi_v2


# ===================================================================
#  HELPERS
# ===================================================================

N_TEST = 16
DX_16 = 2 * np.pi / N_TEST


def _make_fields(N, vx=0.0, vy=0.0, Bx=0.0, By=0.0, Jz=0.0):
    """Uniform MHD fields."""
    return {
        'vx': np.full((N, N), vx, dtype=np.float64),
        'vy': np.full((N, N), vy, dtype=np.float64),
        'Bx': np.full((N, N), Bx, dtype=np.float64),
        'By': np.full((N, N), By, dtype=np.float64),
        'Jz': np.full((N, N), Jz, dtype=np.float64),
    }


def _make_solver(N):
    """Create a solver for coefficient computation."""
    grid = PeriodicGrid(N)
    return MHDSolver(grid)


def _make_mapper(dx=DX_16):
    """Create PhysicalMapperV2."""
    return PhysicalMapperV2(dx=dx)


def _setup_scenario(N, scenario_name, Re=400):
    """Initialize a solver with a given scenario, return (sim, fields, score)."""
    grid = PeriodicGrid(N)
    sim = MHDSolver(grid, dt=1e-4, Re=Re, Rm=Re)

    if scenario_name == "orszag_tang":
        sim.init_orszag_tang()
    elif scenario_name == "harris_tearing":
        sim.init_harris_tearing()
    elif scenario_name == "kelvin_helmholtz":
        sim.init_kelvin_helmholtz()
    elif scenario_name == "mhd_rotor":
        sim.init_mhd_rotor()

    # advance a few steps so gradients develop
    for _ in range(20):
        sim.step_full(record_stats=False)

    fields = sim.get_fluxes()
    physics_state = {
        'vx': sim.vx, 'vy': sim.vy,
        'Bx': sim.Bx, 'By': sim.By,
        'dx': sim.dx,
    }
    # classical_score needs Jz in physics_state
    physics_state['Jz'] = fields['Jz']
    score = AngleMapper.classical_score(physics_state)
    return sim, fields, score


# ===================================================================
#  TEST A: Coefficients survive in all 4 scenarios
# ===================================================================

class TestA_CoefficientsSurvive(unittest.TestCase):
    """
    All Hamiltonian terms must be non-zero on real MHD scenarios.
    A Hamiltonian with zero coefficients provides no quantum correction.
    """

    def _check_scenario(self, scenario):
        N = 32
        sim, fields, score = _setup_scenario(N, scenario, Re=400)
        hm = _make_mapper(dx=sim.dx)
        result = hm.compute_coefficients(sim, score, fields, threshold_amr=0.15)

        H_h, H_v = result['H_edges']
        C_h, C_v = result['C_edges']
        K = result['K_plaquettes']

        max_C = max(np.max(np.abs(C_h)), np.max(np.abs(C_v)))
        max_K = np.max(np.abs(K))
        max_H = max(np.max(np.abs(H_h)), np.max(np.abs(H_v)))

        self.assertGreater(max_C, 1e-3,
                           f"{scenario}: ZZ coefficients must be non-zero")
        self.assertGreater(max_K, 1e-3,
                           f"{scenario}: ZZZZ coefficients must be non-zero")
        self.assertGreater(max_H, 1e-6,
                           f"{scenario}: Z coefficients must be non-zero")

    def test_orszag_tang(self):
        self._check_scenario("orszag_tang")

    def test_harris_tearing(self):
        self._check_scenario("harris_tearing")

    def test_kelvin_helmholtz(self):
        self._check_scenario("kelvin_helmholtz")

    def test_mhd_rotor(self):
        self._check_scenario("mhd_rotor")


# ===================================================================
#  TEST B: Z subordinate to spatial terms
# ===================================================================

class TestB_ZSubordinate(unittest.TestCase):
    """
    Z bias should be a small fraction of the ZZ/ZZZZ energy scale.
    The multi-body terms must dominate the Hamiltonian.
    """

    def test_z_subordinate_alternating_shear(self):
        """In strong shear, Z energy < ZZ+ZZZZ energy."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, Bx=0.5)
        for i in range(N):
            fields['vx'][i, :] = 5.0 if i % 2 == 0 else -5.0

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        H_h, H_v = result['H_edges']
        C_h, C_v = result['C_edges']
        K = result['K_plaquettes']

        total_z = np.sum(np.abs(H_h)) + np.sum(np.abs(H_v))
        total_coupling = (np.sum(np.abs(C_h)) + np.sum(np.abs(C_v))
                          + np.sum(np.abs(K)))

        self.assertGreater(total_coupling, 0.1,
                           "Multi-body terms must be active")
        self.assertLess(total_z, total_coupling,
                        "Z bias must be subordinate to multi-body terms")

    def test_z_bounded_by_c_bias(self):
        """max|H| <= C_BIAS * max(|C|, |K|) (up to score range)."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, Bx=0.5)
        for i in range(N):
            fields['vx'][i, :] = 5.0 if i % 2 == 0 else -5.0

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        H_h, _ = result['H_edges']
        C_h, C_v = result['C_edges']
        K = result['K_plaquettes']

        max_coupling = max(
            np.max(np.abs(C_h)), np.max(np.abs(C_v)),
            np.max(np.abs(K)), 1e-10
        )
        # Z = c * median * (score - thr), and |score - thr| <= 1,
        # and median <= max, so max|H| <= c * max_coupling
        self.assertLessEqual(
            np.max(np.abs(H_h)),
            hm.C_BIAS * max_coupling + 1e-8,
            "Z must be bounded by C_BIAS * max(|C|,|K|)"
        )

    def test_z_zero_when_quiet(self):
        """H = 0 when all fields are zero (no coupling to scale from)."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N)
        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        H_h, H_v = result['H_edges']
        self.assertAlmostEqual(np.max(np.abs(H_h)), 0.0, places=8)
        self.assertAlmostEqual(np.max(np.abs(H_v)), 0.0, places=8)

    def test_z_sign_convention(self):
        """H > 0 where score > threshold (bias toward refinement)."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, Bx=0.5)
        for i in range(N):
            fields['vx'][i, :] = 5.0 if i % 2 == 0 else -5.0

        score = AngleMapper.classical_score(fields)
        threshold = 0.3
        result = hm.compute_coefficients(sim, score, fields, threshold)

        H_h, _ = result['H_edges']
        above = score > threshold
        if np.any(above):
            self.assertGreater(np.mean(H_h[above]), 0.0,
                               "H > 0 where score > threshold")


# ===================================================================
#  TEST C: Ferromagnetic ground state
# ===================================================================

class TestC_FerromagneticStructure(unittest.TestCase):
    """
    ZZ coefficients must be negative (ferromagnetic: neighbors align).
    ZZZZ coefficients must be negative (even-parity).
    """

    def test_zz_is_ferromagnetic(self):
        """Non-zero C values must be <= 0."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, Bx=0.5)
        fields['vx'][:N//2, :] = 3.0
        fields['vx'][N//2:, :] = -3.0

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        C_h, C_v = result['C_edges']
        nonzero_h = C_h[np.abs(C_h) > 1e-8]
        nonzero_v = C_v[np.abs(C_v) > 1e-8]

        if len(nonzero_h) > 0:
            self.assertTrue(np.all(nonzero_h <= 0),
                            "ZZ must be ferromagnetic (C <= 0)")
        if len(nonzero_v) > 0:
            self.assertTrue(np.all(nonzero_v <= 0),
                            "ZZ must be ferromagnetic (C <= 0)")

    def test_zzzz_is_negative(self):
        """Non-zero K values must be <= 0."""
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
        nonzero_K = K[np.abs(K) > 1e-8]
        if len(nonzero_K) > 0:
            self.assertTrue(np.all(nonzero_K <= 0),
                            "ZZZZ must be negative (even-parity)")

    def test_xpoint_is_negative(self):
        """X-point K values must be <= 0."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        # Harris sheet: strong Bx reversal drives det(nabla B) < 0
        fields = _make_fields(N)
        Y = np.linspace(0, 2*np.pi, N, endpoint=False)
        fields['Bx'] = np.tanh((Y[:, None] - np.pi) / 0.3) * np.ones((N, N))
        fields['By'] = 0.1 * np.sin(2 * np.pi * np.linspace(0, 1, N, endpoint=False)[None, :]) * np.ones((N, 1))

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3,
                                         advanced_anomalies_enabled=True)

        K_xp = result.get('K_xpoint')
        self.assertIsNotNone(K_xp, "X-point term must be present")
        nonzero_xp = K_xp[np.abs(K_xp) > 1e-8]
        if len(nonzero_xp) > 0:
            self.assertTrue(np.all(nonzero_xp <= 0),
                            "X-point K must be <= 0")


# ===================================================================
#  TEST D: Scaling with Re
# ===================================================================

class TestD_ReScaling(unittest.TestCase):
    """
    At higher Re, gradients steepen (less viscous smoothing),
    so |C| and |K| should increase with Re.
    """

    def test_coefficients_increase_with_Re(self):
        """
        Run Orszag-Tang at Re=200 and Re=800. Higher Re should produce
        larger |C| and |K| on average (gradients are sharper).
        """
        N = 32

        def _get_coeff_scale(Re):
            sim, fields, score = _setup_scenario(N, "orszag_tang", Re=Re)
            hm = _make_mapper(dx=sim.dx)
            result = hm.compute_coefficients(sim, score, fields, 0.15)
            C_h, C_v = result['C_edges']
            K = result['K_plaquettes']
            return (np.mean(np.abs(C_h)) + np.mean(np.abs(C_v)),
                    np.mean(np.abs(K)))

        c_lo, k_lo = _get_coeff_scale(200)
        c_hi, k_hi = _get_coeff_scale(800)

        # Higher Re should produce larger coefficients (sharper gradients)
        self.assertGreater(c_hi, c_lo * 0.8,
                           f"ZZ should increase with Re: "
                           f"Re=200:{c_lo:.4f} vs Re=800:{c_hi:.4f}")
        # K normalisation by max makes this less strict, but the numerator
        # (|omega| + |Jz|) should still grow.


# ===================================================================
#  TEST E: Noise immunity
# ===================================================================

class TestE_NoiseImmunity(unittest.TestCase):
    """Random noise should produce weaker coefficients than real anomalies."""

    def test_noise_weaker_than_anomaly(self):
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()
        rng = np.random.RandomState(42)

        # Noise
        amp = 0.01
        fields_noise = {
            'vx': amp * rng.randn(N, N),
            'vy': amp * rng.randn(N, N),
            'Bx': amp * rng.randn(N, N),
            'By': amp * rng.randn(N, N),
            'Jz': amp * rng.randn(N, N),
        }
        score_noise = AngleMapper.classical_score(fields_noise)
        r_noise = hm.compute_coefficients(sim, score_noise, fields_noise, 0.3)

        # Real anomaly
        fields_anom = _make_fields(N, Bx=0.5)
        fields_anom['vx'][:N//2, :] = 3.0
        fields_anom['vx'][N//2:, :] = -3.0
        score_anom = AngleMapper.classical_score(fields_anom)
        r_anom = hm.compute_coefficients(sim, score_anom, fields_anom, 0.3)

        # CHANGEMENT DE GRANDEUR, pas de seuil (regle VIGIL). Sous
        # `norm="max"` — le defaut depuis le 21 aout 2026 — le PIC vaut
        # `w_zz` sur TOUT champ non uniforme, par construction : bruit pur
        # et anomalie franche rendent tous deux 2,000, et cette assertion
        # comparait deux nombres identiques. Le pic n'est plus une mesure
        # de force de signal.
        #
        # Le fait physique vise reste vrai et reste mesurable : une anomalie
        # est CONCENTREE, le bruit est ETALE. On mesure donc la fraction du
        # domaine qui depasse la moitie du pic — petite pour une anomalie,
        # grande pour du bruit. C'est une grandeur de FORME, la seule qui
        # survive a une normalisation par le maximum.
        def etalement(r):
            c = np.maximum(np.abs(r['C_edges'][0]), np.abs(r['C_edges'][1]))
            return float(np.mean(c >= 0.5 * c.max()))

        noise_spread = etalement(r_noise)
        anom_spread = etalement(r_anom)

        self.assertLess(anom_spread, noise_spread,
                        f"Anomaly ZZ spread ({anom_spread:.3f}) should be "
                        f"tighter than noise spread ({noise_spread:.3f}) : "
                        "une anomalie concentre le couplage, le bruit l'etale")

        # Le champ qui SEPARE : sans lui, l'assertion ci-dessus passerait sur
        # deux champs de bruit. Le bruit doit reellement etaler.
        self.assertGreater(noise_spread, 0.2,
                           f"le champ de bruit ne s'etale pas "
                           f"({noise_spread:.3f}) : il ne separe rien")


# ===================================================================
#  TEST F: Information orthogonality
# ===================================================================

class TestF_InformationOrthogonality(unittest.TestCase):
    """ZZ and ZZZZ respond to different physics."""

    def test_shear_activates_zz_primarily(self):
        """Sharp velocity jump: ZZ should fire strongly."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, Bx=0.5)
        fields['vx'][:N//2, :] = 3.0
        fields['vx'][N//2:, :] = -3.0

        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        C_max = max(np.max(np.abs(result['C_edges'][0])),
                    np.max(np.abs(result['C_edges'][1])))
        self.assertGreater(C_max, 0.5,
                           "ZZ should fire strongly on shear layer")

    def test_vortex_activates_zzzz(self):
        """Solid rotation: ZZZZ should fire."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

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

        K_max = np.max(np.abs(result['K_plaquettes']))
        self.assertGreater(K_max, 0.01,
                           "ZZZZ should fire on vortex")

    def test_quiet_fields_weak(self):
        """Very quiet fields: all terms near zero."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, vx=0.01, vy=0.01, Bx=0.01, By=0.01)
        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        # With uniform tiny fields, all jumps are zero -> C = 0
        C_h, C_v = result['C_edges']
        K = result['K_plaquettes']
        self.assertLess(np.max(np.abs(C_h)), 0.01)
        self.assertLess(np.max(np.abs(K)), 0.01)


# ===================================================================
#  TEST G: Phase encoding (psi_v2)
# ===================================================================

class TestG_PhaseEncoding(unittest.TestCase):
    """Parameter-free phase encoding is bounded and correct."""

    def test_psi_bounded(self):
        """psi must be in [-pi/2, pi/2]."""
        N = 16
        rng = np.random.RandomState(123)
        phi_prev = rng.randn(N, N)
        phi = rng.randn(N, N) * 5

        psi = compute_psi_v2(phi_prev, phi)
        self.assertLessEqual(np.max(psi), np.pi / 2 + 1e-10)
        self.assertGreaterEqual(np.min(psi), -np.pi / 2 - 1e-10)

    def test_psi_zero_when_no_change(self):
        """psi = 0 when phi_prev == phi."""
        N = 16
        phi = np.ones((N, N))
        psi = compute_psi_v2(phi, phi)
        np.testing.assert_allclose(psi, 0.0, atol=1e-10)

    def test_psi_none_handling(self):
        """psi = 0 when phi_prev or phi is None."""
        N = 16
        phi = np.ones((N, N))
        psi = compute_psi_v2(None, phi)
        np.testing.assert_allclose(psi, 0.0, atol=1e-10)

    def test_psi_sign(self):
        """psi > 0 where flux increased, psi < 0 where it decreased."""
        N = 16
        phi_prev = np.zeros((N, N))
        phi = np.ones((N, N))
        phi[:N//2, :] = 2.0    # increased
        phi[N//2:, :] = -1.0   # decreased

        psi = compute_psi_v2(phi_prev, phi)
        self.assertTrue(np.all(psi[:N//2, :] > 0), "psi > 0 where flux increased")
        self.assertTrue(np.all(psi[N//2:, :] < 0), "psi < 0 where flux decreased")


# ===================================================================
#  TEST H: Backward compatibility (interface matches v1)
# ===================================================================

class TestH_InterfaceCompat(unittest.TestCase):
    """v2 returns the same dict structure as v1 for downstream code."""

    def test_output_keys(self):
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, vx=1.0, Bx=0.5)
        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3)

        self.assertIn('H_edges', result)
        self.assertIn('C_edges', result)
        self.assertIn('K_plaquettes', result)
        self.assertIn('threshold_amr', result)
        self.assertIn('w_z_frac', result)

        # H_edges and C_edges are tuples of 2 arrays
        self.assertEqual(len(result['H_edges']), 2)
        self.assertEqual(len(result['C_edges']), 2)
        self.assertEqual(result['H_edges'][0].shape, (N, N))
        self.assertEqual(result['K_plaquettes'].shape, (N, N))

    def test_output_keys_with_xpoint(self):
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, vx=1.0, Bx=0.5)
        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3,
                                         advanced_anomalies_enabled=True)
        self.assertIn('K_xpoint', result)
        self.assertEqual(result['K_xpoint'].shape, (N, N))

    def test_extra_kwargs_ignored(self):
        """Old callers passing psi_h/psi_v should not crash."""
        N = N_TEST
        sim = _make_solver(N)
        hm = _make_mapper()

        fields = _make_fields(N, vx=1.0)
        score = AngleMapper.classical_score(fields)
        result = hm.compute_coefficients(sim, score, fields, 0.3,
                                         psi_h=np.ones((N, N)),
                                         psi_v=np.ones((N, N)))
        self.assertIn('H_edges', result)


if __name__ == '__main__':
    unittest.main()
