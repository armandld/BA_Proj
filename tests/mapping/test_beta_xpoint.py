"""
Tests for the beta_xpoint hyperparameter — X-point reconnection sensitivity.

Validates that:
  A. K_xpoint detects X-points independently of K_plaquettes (Jz channel)
  B. beta_xpoint controls sensitivity: higher beta → stronger K_xpoint
  C. X-points (det < 0) are distinguished from O-points (det > 0)
  D. When X-points and current sheets co-locate, both channels fire
  E. On a real Orszag-Tang snapshot, K_xpoint adds information beyond K_plaquettes

Key physics:
  K_plaquettes uses curl(B) = Jz  (antisymmetric part of ∇B)
  K_xpoint    uses det(∇B)        (full Jacobian, sym + antisym)

  A curl-free potential field (Bx = y, By = x) has:
    curl(B) = dBy/dx − dBx/dy = 1 − 1 = 0   → K_plaquettes mag component = 0
    det(∇B) = dBx/dx·dBy/dy − dBx/dy·dBy/dx = 0·0 − 1·1 = −1   → X-point!

  This is the regime where beta_xpoint matters: reconnection without strong Jz.

Run:
    cd /home/user/BA_Proj && python -m pytest tests/test_beta_xpoint.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.HamiltParams import PhysicalMapper


# ── Helpers ──────────────────────────────────────────────────────────

def curl_free_xpoint_fields(N=8):
    """Potential field Bx=y, By=x with zero velocity.

    Properties on the discrete grid:
      - det(∇B) < 0 at half the cells (X-points)
      - det(∇B) > 0 at the other half (O-points)
      - Jz_curl = 0 at interior cells (curl-free)
      - Jz_curl ≠ 0 at periodic-wrap cells (boundary artefact)
    """
    # CONVENTION : `grid.py` declare AXIS_X = 0, donc x vit sur l'axe 0
    # et y sur l'axe 1. Ces champs posaient l'inverse, ce qui les
    # transposait par rapport au reste du depot. L'erreur etait
    # invisible tant que les mappeurs portaient la meme inversion :
    # test et code s'accordaient dans la meme confusion. Sous la
    # convention corrigee, une nappe de Harris ainsi construite
    # porte un courant Jz EXACTEMENT nul.
    yc = np.linspace(-1.5, 1.5, N)
    xc = np.linspace(-1.5, 1.5, N)
    Bx = np.tile(yc[None, :], (N, 1))     # Bx = y, y sur l'axe 1
    By = np.tile(xc[:, None], (1, N))     # By = x, x sur l'axe 0
    return {
        'vx': np.zeros((N, N)),
        'vy': np.zeros((N, N)),
        'Bx': Bx,
        'By': By,
        'Jz': np.zeros((N, N)),
    }


def current_sheet_fields(N=8):
    """Harris current sheet: Bx = tanh(y), By = 0.

    Properties:
      - Strong Jz = dBx/dy at the sheet center
      - det(∇B) ≈ 0 (By = 0 → dBy/dx = dBy/dy = 0)
      - K_plaquettes should fire (via Jz), K_xpoint should NOT
    """
    # CONVENTION : `grid.py` declare AXIS_X = 0, donc x vit sur l'axe 0
    # et y sur l'axe 1. Ces champs posaient l'inverse, ce qui les
    # transposait par rapport au reste du depot. L'erreur etait
    # invisible tant que les mappeurs portaient la meme inversion :
    # test et code s'accordaient dans la meme confusion. Sous la
    # convention corrigee, une nappe de Harris ainsi construite
    # porte un courant Jz EXACTEMENT nul.
    y = np.linspace(-3, 3, N)
    Bx = np.tanh(np.tile(y[None, :], (N, 1)))   # varie selon y = axe 1
    return {
        'vx': np.zeros((N, N)),
        'vy': np.zeros((N, N)),
        'Bx': Bx,
        'By': np.zeros((N, N)),
        'Jz': np.zeros((N, N)),
    }


def xpoint_with_current_fields(N=8):
    """X-point topology WITH co-located current sheet.

    Combines a potential-field X-point with a Harris sheet:
      Bx = y + tanh(y),  By = x
    Both K_plaquettes (Jz) and K_xpoint (det) should fire.
    """
    # CONVENTION : `grid.py` declare AXIS_X = 0, donc x vit sur l'axe 0
    # et y sur l'axe 1. Ces champs posaient l'inverse, ce qui les
    # transposait par rapport au reste du depot. L'erreur etait
    # invisible tant que les mappeurs portaient la meme inversion :
    # test et code s'accordaient dans la meme confusion. Sous la
    # convention corrigee, une nappe de Harris ainsi construite
    # porte un courant Jz EXACTEMENT nul.
    yc = np.linspace(-1.5, 1.5, N)
    xc = np.linspace(-1.5, 1.5, N)
    y2d = np.tile(yc[None, :], (N, 1))          # y sur l'axe 1
    Bx = y2d + 1.5 * np.tanh(y2d * 2)
    By = np.tile(xc[:, None], (1, N))           # x sur l'axe 0
    return {
        'vx': np.zeros((N, N)),
        'vy': np.zeros((N, N)),
        'Bx': Bx,
        'By': By,
        'Jz': np.zeros((N, N)),
    }


def make_mapper(**kwargs):
    defaults = dict(cs=1.0, eta_mhd=0.01, beta_curl=1.0, beta_xpoint=1.0)
    defaults.update(kwargs)
    return PhysicalMapper(**defaults)


def compute_hp(fields, N=8, adv=True, **mapper_kwargs):
    """Compute Hamiltonian coefficients for given fields."""
    mapper = make_mapper(**mapper_kwargs)
    grid = PeriodicGrid(N)
    sim = MHDSolver(grid)
    score = mapper.physical_score(fields)
    return mapper.compute_coefficients(
        sim, score, fields, 0.0,
        advanced_anomalies_enabled=adv,
    )


# ══════════════════════════════════════════════════════════════════════
#  TEST A: X-point detection independent of K_plaquettes
# ══════════════════════════════════════════════════════════════════════

class TestXpointIndependence:
    """Curl-free field → K_xpoint fires, K_plaquettes is weak."""

    def test_curl_free_activates_K_xpoint(self):
        """A curl-free X-point field should produce significant K_xpoint."""
        N = 8
        fields = curl_free_xpoint_fields(N)
        hp = compute_hp(fields, N, beta_xpoint=1.0)

        Kx = hp['K_xpoint']
        Kx_max = np.max(np.abs(Kx))
        print(f"\n  [CURL-FREE] max |K_xpoint| = {Kx_max:.4f}")
        assert Kx_max > 0.1, f"K_xpoint should detect X-points, got max={Kx_max:.4f}"

    def test_curl_free_weak_K_plaquettes_interior(self):
        """Interior cells of a curl-free field should have weak K_plaquettes.

        Boundary cells may have non-zero K_plaquettes from periodic wrap,
        but interior cells (where Jz_curl=0) should be near-zero.
        """
        N = 8
        fields = curl_free_xpoint_fields(N)
        hp = compute_hp(fields, N, beta_xpoint=1.0)

        K = hp['K_plaquettes']
        # Interior: rows 2:-2, cols 2:-2 (away from periodic wrap)
        K_interior = K[2:-2, 2:-2]
        K_int_max = np.max(np.abs(K_interior))
        Kx_max = np.max(np.abs(hp['K_xpoint']))

        print(f"\n  [CURL-FREE] max |K_plaq interior| = {K_int_max:.4f}")
        print(f"  [CURL-FREE] max |K_xpoint|        = {Kx_max:.4f}")
        assert Kx_max > K_int_max, (
            f"K_xpoint should dominate over K_plaq interior for curl-free field: "
            f"Kx={Kx_max:.4f}, K_interior={K_int_max:.4f}"
        )

    def test_current_sheet_weak_K_xpoint(self):
        """A pure Harris sheet (By=0) should have negligible K_xpoint."""
        N = 8
        fields = current_sheet_fields(N)
        hp = compute_hp(fields, N, beta_xpoint=1.0)

        K_max = np.max(np.abs(hp['K_plaquettes']))
        Kx_max = np.max(np.abs(hp.get('K_xpoint', np.zeros((N, N)))))

        print(f"\n  [HARRIS] max |K_plaquettes| = {K_max:.4f}")
        print(f"  [HARRIS] max |K_xpoint|     = {Kx_max:.4f}")

        # K_plaquettes should dominate (Jz channel fires on current sheet)
        # K_xpoint should be small (no X-point topology: By=0 → det≈0)
        assert K_max > Kx_max, (
            f"Harris sheet: K_plaquettes should dominate K_xpoint: "
            f"K={K_max:.4f}, Kx={Kx_max:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════
#  TEST B: beta_xpoint sensitivity
# ══════════════════════════════════════════════════════════════════════

class TestBetaXpointSensitivity:
    """Higher beta_xpoint → more aggressive threshold-contrast → stronger K_xpoint."""

    def test_higher_beta_gives_stronger_K_xpoint(self):
        """Increasing beta_xpoint should increase |K_xpoint| magnitude.

        Uses a weak B-field so the threshold-contrast signal doesn't
        saturate the tc_max=10.0 clamp for the low-beta case.
        """
        N = 8
        # Weak field: scale down B so the signal is near the critical value
        fields = curl_free_xpoint_fields(N)
        fields['Bx'] *= 0.05
        fields['By'] *= 0.05

        hp_low = compute_hp(fields, N, beta_xpoint=0.1)
        hp_high = compute_hp(fields, N, beta_xpoint=2.0)

        Kx_low = np.max(np.abs(hp_low['K_xpoint']))
        Kx_high = np.max(np.abs(hp_high['K_xpoint']))

        print(f"\n  [SENSITIVITY] beta_xpoint=0.1 → max|Kx| = {Kx_low:.4f}")
        print(f"  [SENSITIVITY] beta_xpoint=2.0 → max|Kx| = {Kx_high:.4f}")

        assert Kx_high > Kx_low, (
            f"Higher beta_xpoint should give stronger K_xpoint: "
            f"low={Kx_low:.4f}, high={Kx_high:.4f}"
        )

    def test_beta_xpoint_does_not_affect_K_plaquettes(self):
        """beta_xpoint is independent of K_plaquettes computation."""
        N = 8
        fields = curl_free_xpoint_fields(N)

        hp_a = compute_hp(fields, N, beta_xpoint=0.3)
        hp_b = compute_hp(fields, N, beta_xpoint=3.0)

        K_a = hp_a['K_plaquettes']
        K_b = hp_b['K_plaquettes']

        assert np.allclose(K_a, K_b, atol=1e-10), (
            "beta_xpoint should not change K_plaquettes"
        )


# ══════════════════════════════════════════════════════════════════════
#  TEST C: X-point vs O-point discrimination
# ══════════════════════════════════════════════════════════════════════

class TestXpointVsOpoint:
    """K_xpoint = max(0, -det(∇B)): positive only at X-points (det < 0)."""

    def test_xpoint_signal_spatial_pattern(self):
        """The K_xpoint map should have a checkerboard-like pattern:
        X-points and O-points alternate on the potential-field grid.
        """
        N = 8
        fields = curl_free_xpoint_fields(N)
        hp = compute_hp(fields, N, beta_xpoint=1.0)

        Kx = hp['K_xpoint']
        # K_xpoint = -f_Rm * mic_xpoint, so active cells are negative
        n_active = np.sum(Kx < -0.01)
        n_inactive = np.sum(np.abs(Kx) < 0.01)
        n_total = N * N

        print(f"\n  [PATTERN] Active cells (Kx < -0.01): {n_active}/{n_total}")
        print(f"  [PATTERN] Inactive cells (|Kx| < 0.01): {n_inactive}/{n_total}")

        # Should have roughly half active (X-points) and half inactive (O-points)
        assert n_active > 0, "Some cells should be X-point active"
        assert n_inactive > 0, "Some cells should be O-point inactive"
        assert n_active < n_total, "Not all cells should be active (O-points exist)"


# ══════════════════════════════════════════════════════════════════════
#  TEST D: Co-located X-point + current sheet
# ══════════════════════════════════════════════════════════════════════

class TestColocated:
    """When both X-points and Jz co-exist, both channels should fire."""

    def test_both_channels_fire(self):
        """X-point + current sheet field should activate both K_xpoint and K_plaquettes."""
        N = 8
        fields = xpoint_with_current_fields(N)
        hp = compute_hp(fields, N, beta_xpoint=1.0)

        K_max = np.max(np.abs(hp['K_plaquettes']))
        Kx_max = np.max(np.abs(hp['K_xpoint']))

        print(f"\n  [CO-LOCATED] max |K_plaquettes| = {K_max:.4f}")
        print(f"  [CO-LOCATED] max |K_xpoint|     = {Kx_max:.4f}")

        assert K_max > 0.1, f"K_plaquettes should be active, got {K_max:.4f}"
        assert Kx_max > 0.1, f"K_xpoint should be active, got {Kx_max:.4f}"

    def test_xpoint_adds_information(self):
        """K_xpoint should not be a simple scalar multiple of K_plaquettes.

        If they carry different spatial information, their correlation
        should be imperfect (|r| < 0.95).
        """
        N = 8
        fields = xpoint_with_current_fields(N)
        hp = compute_hp(fields, N, beta_xpoint=1.0)

        K = hp['K_plaquettes'].ravel()
        Kx = hp['K_xpoint'].ravel()

        # Only compare cells where at least one channel is active
        mask = (np.abs(K) > 0.01) | (np.abs(Kx) > 0.01)
        if np.sum(mask) < 4:
            pytest.skip("Not enough active cells for correlation test")

        corr = np.corrcoef(K[mask], Kx[mask])[0, 1]
        print(f"\n  [INFO] Correlation(K_plaq, K_xpoint) = {corr:.4f}")
        print(f"  [INFO] Active cells: {np.sum(mask)}/{N*N}")

        assert abs(corr) < 0.95, (
            f"K_xpoint should add independent info (|corr| < 0.95), "
            f"got {corr:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════
#  TEST E: Orszag-Tang — real physics scenario
# ══════════════════════════════════════════════════════════════════════

class TestOrszagTang:
    """On a real OT snapshot, K_xpoint should add spatial information."""

    def test_ot_xpoint_nonzero(self):
        """Orszag-Tang develops X-points by t≈0.5 — K_xpoint should be active."""
        N = 32
        grid = PeriodicGrid(N)
        sim = MHDSolver(grid, dt=5e-3, Re=800, Rm=800)

        # Orszag-Tang initial conditions
        X, Y = grid.X, grid.Y
        sim.vx = -np.sin(Y)
        sim.vy = np.sin(X)
        sim.Bx = -np.sin(Y)
        sim.By = np.sin(2 * X)
        sim.enforce_incompressibility()
        sim.record_energy()

        # Evolve to develop X-points
        for _ in range(100):
            sim.step_full()

        fields = {
            'vx': sim.vx, 'vy': sim.vy,
            'Bx': sim.Bx, 'By': sim.By,
            'Jz': np.zeros_like(sim.vx),
        }

        hp = compute_hp(fields, N, beta_xpoint=1.0,
                         eta_mhd=grid.L / 800, beta_curl=1.0)

        Kx = hp['K_xpoint']
        K = hp['K_plaquettes']
        Kx_max = np.max(np.abs(Kx))
        K_max = np.max(np.abs(K))

        print(f"\n  [OT] max |K_plaquettes| = {K_max:.4f}")
        print(f"  [OT] max |K_xpoint|     = {Kx_max:.4f}")
        print(f"  [OT] K_xpoint active cells: {np.sum(np.abs(Kx) > 0.01)}/{N*N}")

        assert Kx_max > 0.01, (
            f"Orszag-Tang should develop X-points by t=0.5, got max|Kx|={Kx_max:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
