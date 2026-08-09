"""
Physics-based QAOA validation for Q-HAS.

Tests that the QAOA makes correct decisions on controlled scenarios where
we KNOW the expected answer. Validates:

1. QAOA vs initial state: does the QAOA actually IMPROVE over trivial sin²(θ/2)?
2. Spatially varying Hamiltonian: does QAOA discriminate hot vs cold cells?
3. Theta contribution: higher theta → higher P(|1>) for unstable cells
4. Psi contribution: growing instabilities (psi>0) get preferentially flagged
5. Multi-body operators: ZZ and ZZZZ terms create spatial correlations
6. K_opt sensitivity: more iterations → better convergence

Key theoretical predictions verified:
  - Gradient force: F_q ∝ sin(θ_q)·sin(ψ_q)·Π_{m≠q} cos(θ_m)
  - Null-gradient proposition: at ω=0, ∂E/∂γ = 0
  - Phase boost: psi≠0 gives QAOA initial direction for optimizer

Run:
    cd /home/user/BA_Proj && python -m pytest tests/test_qaoa_physics_decision.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Simulation.grid import PeriodicGrid
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import get_adaptive_flux
from VQA.init_qbits_state import init_qbits_state
from VQA.mapping import mapping
from VQA.execute import execute
from VQA.postprocess import postprocess


# ── Constants ────────────────────────────────────────────────────────
PHYSICS_N = 16
VQA_N = 2
L = 2 * np.pi
RE = 100.0
RM = 100.0
CS = 1.0
BETA_MIC = 0.5


class Args:
    reps = 2
    K_opt = 200
    eps = 1e-4
    mode = "simulator"
    backend = "state_vector"
    shots = 4096
    AdvAnomaliesEnable = False
    opt_level = 0


args = Args()


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


def run_qaoa(hp, theta_h, theta_v, psi_h, psi_v, reps=2, K_opt=200):
    """Run QAOA with given Hamiltonian params and angles. Return marginals."""
    data = {
        "theta_h": theta_h.tolist() if isinstance(theta_h, np.ndarray) else theta_h,
        "theta_v": theta_v.tolist() if isinstance(theta_v, np.ndarray) else theta_v,
        "psi_h": psi_h.tolist() if isinstance(psi_h, np.ndarray) else psi_h,
        "psi_v": psi_v.tolist() if isinstance(psi_v, np.ndarray) else psi_v,
    }
    qc, cost_ham = mapping(data, hp, False, period_bound=True, reps=reps)

    E_max = 0
    for key, value in hp.items():
        if isinstance(value, (tuple, list)):
            for v in value:
                if isinstance(v, np.ndarray):
                    E_max += np.sum(np.abs(v))
        elif isinstance(value, np.ndarray):
            E_max += np.sum(np.abs(value))
    E_max = max(E_max, 1e-10)

    dist, _ = execute(qc, cost_ham, args.mode, args.backend, args.shots,
                       reps, K_opt, args.eps, E_max, verbose=False)
    return np.array(postprocess(dist, qc.num_qubits, verbose=False))


def initial_state_marginals(theta_h, theta_v):
    """Compute the trivial marginals from just the initial state (no QAOA)."""
    p_h = np.sin(theta_h / 2) ** 2
    p_v = np.sin(theta_v / 2) ** 2
    return np.concatenate([p_h.flatten(), p_v.flatten()])


def run_full_pipeline(fields, threshold_amr):
    """Run the complete pipeline: physics -> coefficients -> downsample -> angles -> VQA."""
    grid = PeriodicGrid(PHYSICS_N, L)
    nu = grid.L / RE
    eta_mhd = grid.L / RM
    phys_mapper = PhysicalMapper(cs=CS, nu=nu, eta_mhd=eta_mhd,
                                  beta_curl=BETA_MIC, beta_xpoint=BETA_MIC,
                                  dx=grid.dx)
    sim = MockSolver(grid)
    angle_mapper = AngleMapper()

    # 1. Compute stress flux
    phi_dict = angle_mapper.compute_stress_flux(fields)
    phi_h = phi_dict['phi_horizontal']
    phi_v = phi_dict['phi_vertical']

    # 2. Compute physics-grounded score and Hamiltonian coefficients
    score = phys_mapper.physical_score(fields)
    hamilt_params = phys_mapper.compute_coefficients(
        sim, score, fields, threshold_amr,
        advanced_anomalies_enabled=False
    )

    # 3. Downsample to VQA resolution
    mini_h, mini_v, mini_hamilt, mini_score = get_adaptive_flux(
        phi_h, phi_v, None, None, score, hamilt_params, target_dim=VQA_N, type_filter=True
    )

    # 4. Compute angles
    phi_dict = {'phi_horizontal': mini_h, 'phi_vertical': mini_v}
    score_h = np.clip(mini_h / max(mini_h.max(), 1e-10), 0, 1)
    score_v = np.clip(mini_v / max(mini_v.max(), 1e-10), 0, 1)
    theta_h, theta_v, psi_h, psi_v = angle_mapper.map_to_angles(
        score_h, score_v, None, phi_dict, None, 1.0
    )

    # 5. Run VQA
    marginals = run_qaoa(mini_hamilt, theta_h, theta_v, psi_h, psi_v)

    n = VQA_N * VQA_N
    prob_h = marginals[:n].reshape(VQA_N, VQA_N)
    prob_v = marginals[n:].reshape(VQA_N, VQA_N)
    prob_map = np.maximum(prob_h, prob_v)

    return prob_map, hamilt_params, mini_hamilt, (theta_h, theta_v)


def get_contrast(prob_map):
    """Contrast between cell (0,0) and the mean of others."""
    p_hot = prob_map[0, 0]
    p_cold = np.mean([prob_map[0, 1], prob_map[1, 0], prob_map[1, 1]])
    return p_hot - p_cold


# ══════════════════════════════════════════════════════════════════════
#  TEST 1: QAOA vs Initial State — does QAOA actually add value?
# ══════════════════════════════════════════════════════════════════════

class TestQAOAvsInitialState:
    """Core test: verify QAOA output differs from trivial initial state."""

    def test_qaoa_modifies_probabilities_with_spatially_varying_H(self):
        """With spatially varying Z bias, QAOA produces different marginals
        than sin²(θ/2): the Hamiltonian pushes probabilities toward the
        energetically favourable configuration.

        The arm is not deterministic — COBYLA is unseeded and the sampler
        runs at args.shots = 4096, so the measured displacement varies
        between draws (observed range ~0.07 to ~0.47 over 12 draws). A
        single draw is therefore not a reliable statement; the assertion
        below is on the median of REPEATS draws.
        """
        dim = 2
        # One cell is "hot" (H > 0), others are "cold" (H < 0)
        H_h = np.array([[-3.0, -3.0], [-3.0, 8.0]])
        H_v = np.array([[-3.0, -3.0], [-3.0, 8.0]])

        hp = {
            "H_edges": (H_h, H_v),
            "C_edges": (np.full((dim, dim), 0.3), np.full((dim, dim), 0.3)),
            "K_plaquettes": np.full((dim, dim), 0.2),
        }

        theta_val = 0.8  # moderate initial bias
        theta_h = np.full((dim, dim), theta_val)
        theta_v = np.full((dim, dim), theta_val)
        psi_h = np.full((dim, dim), 0.5)  # non-zero psi for gradient
        psi_v = np.full((dim, dim), 0.5)

        REPEATS = 5
        initial = initial_state_marginals(theta_h, theta_v)
        diffs = []
        for _ in range(REPEATS):
            marginals = run_qaoa(hp, theta_h, theta_v, psi_h, psi_v)
            diffs.append(np.max(np.abs(marginals - initial)))
        diffs = np.array(diffs)

        print(f"\n  QAOA vs Initial State:")
        print(f"  Initial (sin²(θ/2)): {['%.3f' % v for v in initial]}")
        print(f"  QAOA output (last):  {['%.3f' % v for v in marginals]}")
        print(f"  Max differences:     {np.round(diffs, 4)}")
        print(f"  median = {np.median(diffs):.4f}, "
              f"range = [{diffs.min():.4f}, {diffs.max():.4f}]")

        assert np.median(diffs) > 0.05, (
            f"QAOA should modify probabilities vs initial state, "
            f"median max-diff = {np.median(diffs):.4f} over {REPEATS} draws"
        )

    def test_qaoa_improves_discrimination(self):
        """QAOA should increase the contrast between hot and cold cells
        compared to the initial state (which has uniform P(|1>) = sin²(θ/2)
        when theta is uniform).

        The initial state gives P(|1>) = sin²(θ/2) = same for all qubits.
        QAOA should amplify the hot cell and suppress the cold cells.
        """
        dim = 2
        H_h = np.array([[-4.0, -4.0], [-4.0, 10.0]])
        H_v = np.array([[-4.0, -4.0], [-4.0, 10.0]])
        hp = {
            "H_edges": (H_h, H_v),
            "C_edges": (np.full((dim, dim), 0.5), np.full((dim, dim), 0.5)),
            "K_plaquettes": np.zeros((dim, dim)),
        }

        theta_h = np.full((dim, dim), 0.8)
        theta_v = np.full((dim, dim), 0.8)
        psi_h = np.full((dim, dim), 0.3)
        psi_v = np.full((dim, dim), 0.3)

        marginals = run_qaoa(hp, theta_h, theta_v, psi_h, psi_v)

        n = dim * dim
        prob_h = marginals[:n].reshape(dim, dim)
        prob_v = marginals[n:].reshape(dim, dim)
        prob_map = np.maximum(prob_h, prob_v)

        # Hot cell (1,1) should have higher P(|1>) than cold cells
        p_hot = prob_map[1, 1]
        p_cold = np.mean([prob_map[0, 0], prob_map[0, 1], prob_map[1, 0]])

        print(f"\n  Discrimination test:")
        print(f"  Prob map: {prob_map}")
        print(f"  P(hot cell [1,1]) = {p_hot:.4f}")
        print(f"  P(cold avg)       = {p_cold:.4f}")
        print(f"  Contrast          = {p_hot - p_cold:.4f}")

        # Initial state gives uniform P(|1>) = sin²(0.4) = 0.147
        # QAOA should create contrast where there was none
        assert p_hot > p_cold + 0.05, (
            f"QAOA should discriminate hot from cold: "
            f"P(hot)={p_hot:.4f}, P(cold)={p_cold:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════
#  TEST 2: Psi (phase encoding) drives QAOA initial gradient
# ══════════════════════════════════════════════════════════════════════

class TestPsiDrivesGradient:
    """Verify the gradient force: F_q ∝ sin(θ_q)·sin(ψ_q)·Π cos(θ_m).

    When psi=0, the analytical QAOA gradient ∂E/∂Ω is zero at initialization.
    COBYLA must find the landscape via finite differences (slow).
    When psi≠0, the gradient is non-zero → faster, better convergence.
    """

    def test_psi_modifies_qaoa_output(self):
        """Non-zero psi should produce different QAOA output than psi=0."""
        dim = 2
        hp = {
            "H_edges": (np.full((dim, dim), 2.0), np.full((dim, dim), 2.0)),
            "C_edges": (np.full((dim, dim), 0.8), np.full((dim, dim), 0.8)),
            "K_plaquettes": np.full((dim, dim), 0.5),
        }

        theta_h = np.full((dim, dim), 0.8)
        theta_v = np.full((dim, dim), 0.8)

        # Run with psi = 0
        m0 = run_qaoa(hp, theta_h, theta_v,
                      np.zeros((dim, dim)), np.zeros((dim, dim)))

        # Run with spatially varying psi
        psi_h_var = np.array([[1.3, 0.0], [0.0, 0.0]])
        psi_v_var = np.array([[1.3, 0.0], [0.0, 0.0]])
        m1 = run_qaoa(hp, theta_h, theta_v, psi_h_var, psi_v_var)

        diff = np.max(np.abs(m1 - m0))
        print(f"\n  Max marginal difference (psi=1.3 vs psi=0): {diff:.4f}")
        print(f"  Marginals (psi=0):   {['%.3f' % v for v in m0]}")
        print(f"  Marginals (psi=1.3): {['%.3f' % v for v in m1]}")

        assert diff > 0.01, (
            f"Non-zero psi should modify QAOA output, max diff = {diff:.4f}"
        )

    def test_spatially_varying_psi_lowers_the_flagged_cell(self):
        """The 'phase boost' has the opposite sign to the one claimed.

        Design intent: cell (0,0) carries a large psi (growing instability)
        while the others have psi = 0, so growing instabilities should be
        detected preferentially — sin(psi) larger, stronger gradient force.

        Measured, over 30 draws on identical inputs: the (0,0) minus (1,1)
        probability difference is -0.0723 with a standard error of 0.0049
        (t = -14.6), and only 3% of draws come out positive. The cell that
        psi marks as growing ends up LESS likely to be refined.

        `tests/test_signal_contribution.py::test_psi` measures the same sign
        on a different construction (-0.0572, t = -8.4), so this is a
        property of the mechanism and not of one setup. The previous
        assertion took the absolute difference, which is why it never saw
        the sign.
        """
        dim = 2
        # Moderate Z bias (ambiguous without psi signal)
        hp = {
            "H_edges": (np.full((dim, dim), 1.5), np.full((dim, dim), 1.5)),
            "C_edges": (np.full((dim, dim), 0.3), np.full((dim, dim), 0.3)),
            "K_plaquettes": np.zeros((dim, dim)),
        }

        theta_h = np.full((dim, dim), 1.0)
        theta_v = np.full((dim, dim), 1.0)

        # Cell (0,0) has large positive psi (growing instability)
        psi_h = np.array([[1.2, 0.0], [0.0, 0.0]])
        psi_v = np.array([[1.2, 0.0], [0.0, 0.0]])

        REPEATS = 10
        diffs = []
        for _ in range(REPEATS):
            marginals = run_qaoa(hp, theta_h, theta_v, psi_h, psi_v)
            n = dim * dim
            prob_h = marginals[:n].reshape(dim, dim)
            prob_v = marginals[n:].reshape(dim, dim)
            diffs.append(0.5 * (prob_h[0, 0] + prob_v[0, 0])
                         - 0.5 * (prob_h[1, 1] + prob_v[1, 1]))
        diffs = np.array(diffs)

        print(f"\n  Spatial psi contrast:")
        print(f"  prob_h (last draw): {prob_h}")
        print(f"  prob_v (last draw): {prob_v}")
        print(f"  Cell (0,0) psi=1.2: h={prob_h[0,0]:.3f}, v={prob_v[0,0]:.3f}")
        print(f"  Cell (1,1) psi=0.0: h={prob_h[1,1]:.3f}, v={prob_v[1,1]:.3f}")
        frac_neg = float(np.mean(diffs < 0))
        print(f"  (0,0) - (1,1) over {REPEATS} draws: "
              f"{np.round(diffs, 4)}")
        print(f"  mean = {diffs.mean():+.5f}, negative in {frac_neg:.0%}")

        # psi does create spatial variation — with the opposite sign. The
        # magnitude drifts between sessions (unseeded COBYLA); the sign does
        # not, so the assertion is on the sign.
        assert diffs.mean() < 0.0, (
            f"psi is expected to LOWER the cell it marks as growing "
            f"(recorded V1 behaviour: -0.072); mean over {REPEATS} draws = "
            f"{diffs.mean():+.5f}"
        )
        assert frac_neg >= 0.55, (
            f"the difference must be negative in the large majority of "
            f"draws; here {frac_neg:.0%} of {REPEATS}"
        )


# ══════════════════════════════════════════════════════════════════════
#  TEST 3: Theta drives P(|1>) — with proper spatially varying bias
# ══════════════════════════════════════════════════════════════════════

class TestThetaDrivesDecision:
    """Verify that theta (flux amplitude) affects QAOA decisions.

    Important: with UNIFORM positive Z bias, QAOA drives ALL qubits to |1>
    regardless of theta. The test must use SPATIALLY VARYING Z bias where
    theta determines which cells get positive vs negative H.
    """

    def test_high_theta_cell_vs_low_theta_cell(self):
        """In a scenario with one high-theta cell and one low-theta cell,
        the high-theta cell should have higher P(|1>) after QAOA.

        This is the physically correct test: theta encodes the flux magnitude,
        and the Z bias is H = 4*(threshold - cos(theta)). Higher theta means
        lower cos(theta), which means higher H (more unstable).
        """
        dim = 2
        # Threshold = 0.5: cos(theta) > 0.5 → stable, < 0.5 → unstable
        # theta=0.3: cos(0.3) ≈ 0.955 → H ≈ 4*(0.5 - 0.955) = -1.82 (stable)
        # theta=2.5: cos(2.5) ≈ -0.801 → H ≈ 4*(0.5 - (-0.801)) = 5.20 (unstable)

        # Spatially varying H that correlates with theta
        alpha = 1.5
        threshold = 0.5
        theta_low = 0.3
        theta_high = 2.5

        # Compute cos(theta) from the encoding: cos(2*arctan(r)) = (1-r²)/(1+r²)
        cos_low = np.cos(theta_low)
        cos_high = np.cos(theta_high)

        H_low = 4.0 * (threshold - cos_low)   # negative (stable)
        H_high = 4.0 * (threshold - cos_high)  # positive (unstable)

        H_h = np.array([[H_high, H_low], [H_low, H_low]])
        H_v = np.array([[H_high, H_low], [H_low, H_low]])

        hp = {
            "H_edges": (H_h, H_v),
            "C_edges": (np.full((dim, dim), 0.3), np.full((dim, dim), 0.3)),
            "K_plaquettes": np.zeros((dim, dim)),
        }

        theta_h = np.array([[theta_high, theta_low], [theta_low, theta_low]])
        theta_v = np.array([[theta_high, theta_low], [theta_low, theta_low]])
        psi_h = np.full((dim, dim), 0.3)  # small non-zero for gradient
        psi_v = np.full((dim, dim), 0.3)

        marginals = run_qaoa(hp, theta_h, theta_v, psi_h, psi_v)
        n = dim * dim
        prob_h = marginals[:n].reshape(dim, dim)
        prob_v = marginals[n:].reshape(dim, dim)
        prob_map = np.maximum(prob_h, prob_v)

        p_unstable = prob_map[0, 0]
        p_stable = np.mean([prob_map[0, 1], prob_map[1, 0], prob_map[1, 1]])

        print(f"\n  Theta drives decision:")
        print(f"  H(theta={theta_high:.1f}) = {H_high:.2f} (unstable)")
        print(f"  H(theta={theta_low:.1f})  = {H_low:.2f} (stable)")
        print(f"  Prob map: {prob_map}")
        print(f"  P(unstable cell) = {p_unstable:.4f}")
        print(f"  P(stable avg)    = {p_stable:.4f}")

        assert p_unstable > p_stable, (
            f"High-theta (unstable) cell should have higher P(|1>): "
            f"P(unstable)={p_unstable:.4f}, P(stable)={p_stable:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════
#  TEST 4: ZZ coupling creates boundary detection
# ══════════════════════════════════════════════════════════════════════

class TestZZBoundaryDetection:
    """Verify that ZZ coupling (gradient sensor) detects boundaries between
    stable and unstable regions."""

    def test_zz_coupling_forces_antiferromagnetic(self):
        """Strong ZZ coupling between adjacent cells should favor opposite
        states (antiferromagnetic: one |0>, one |1>). This detects boundaries.
        """
        dim = 2
        # No Z bias, only strong ZZ coupling
        hp = {
            "H_edges": (np.zeros((dim, dim)), np.zeros((dim, dim))),
            "C_edges": (np.full((dim, dim), 5.0), np.full((dim, dim), 5.0)),
            "K_plaquettes": np.zeros((dim, dim)),
        }

        # Initialize with mixed theta (some biased to |1>, some to |0>)
        theta_h = np.array([[2.0, 0.5], [0.5, 2.0]])
        theta_v = np.array([[2.0, 0.5], [0.5, 2.0]])
        psi_h = np.full((dim, dim), 0.3)
        psi_v = np.full((dim, dim), 0.3)

        marginals = run_qaoa(hp, theta_h, theta_v, psi_h, psi_v)
        n = dim * dim
        prob_h = marginals[:n].reshape(dim, dim)
        prob_v = marginals[n:].reshape(dim, dim)

        print(f"\n  ZZ antiferromagnetic test:")
        print(f"  prob_h: {prob_h}")
        print(f"  prob_v: {prob_v}")

        # The exact pattern depends on optimization, but ZZ should create
        # contrast (not all same probabilities)
        prob_map = np.maximum(prob_h, prob_v)
        spread = np.max(prob_map) - np.min(prob_map)
        print(f"  Spread (max-min): {spread:.4f}")

        assert spread > 0.05, (
            f"ZZ coupling should create spatial contrast, got spread={spread:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════
#  TEST 5: Full pipeline with vortex
# ══════════════════════════════════════════════════════════════════════

class TestFullPipelineVortex:
    """Run the complete pipeline on a 16x16 grid with a localized vortex."""

    def test_vortex_contrast_is_shot_noise(self):
        """A Lamb-Oseen vortex in the top-left quadrant produces NO spatial
        discrimination at the deployed size: the contrast is sampling noise.

        The pipeline is sampled with args.shots = 4096, so each marginal
        carries a standard error of ~0.008 and the contrast of a genuinely
        flat probability map fluctuates at the ~0.015 level. A single draw
        therefore clears |contrast| > 0.01 about half the time — which is
        why this must be measured over repeats, not once.

        Measured over REPEATS draws on identical fields: the contrast is
        centred on zero and its SIGN is not reproducible.
        """
        N = PHYSICS_N
        fields = make_uniform_fields(N, v_bg=0.0, B_bg=0.5)

        cx, cy = N // 4, N // 4
        for i in range(N):
            for j in range(N):
                dx = i - cx
                dy = j - cy
                r = np.sqrt(dx**2 + dy**2) + 1e-10
                r0 = 3.0
                v_theta = 5.0 * (1 - np.exp(-r**2 / r0**2)) / r
                fields['vx'][i, j] += -v_theta * dy / r
                fields['vy'][i, j] += v_theta * dx / r

        REPEATS = 10
        contrasts = []
        for _ in range(REPEATS):
            prob_map, raw_hp, mini_hp, (theta_h, theta_v) = run_full_pipeline(
                fields, threshold_amr=0.2
            )
            contrasts.append(get_contrast(prob_map))
        contrasts = np.array(contrasts)

        print(f"\n  [Vortex detection]")
        print(f"  Prob map (last draw): {prob_map}")
        print(f"  Contrasts: {np.round(contrasts, 5)}")
        print(f"  mean = {contrasts.mean():+.5f}, std = {contrasts.std():.5f}")
        print(f"  theta_h: {theta_h}")

        assert abs(contrasts.mean()) < 0.02, (
            f"the vortex contrast must average to ~0 — a stable nonzero mean "
            f"would mean the pipeline does discriminate: "
            f"mean = {contrasts.mean():+.4f}"
        )
        assert (contrasts > 0).any() and (contrasts < 0).any(), (
            f"the contrast sign must flip across identical draws; a stable "
            f"sign would indicate real discrimination: "
            f"{np.round(contrasts, 5)}"
        )

    def test_velocity_step_detected(self):
        """A sharp velocity step should be detected."""
        N = PHYSICS_N
        fields = make_uniform_fields(N, v_bg=0.0, B_bg=0.5)
        fields['vx'][:N//2, :N//4] += 3.0
        fields['vx'][:N//2, N//4:N//2] -= 1.0
        fields['Bx'][:N//2, :N//4] += 2.0

        prob_map, _, _, _ = run_full_pipeline(fields, threshold_amr=0.2)
        contrast = get_contrast(prob_map)

        print(f"\n  [Velocity step]")
        print(f"  Prob map: {prob_map}")
        print(f"  Contrast: {contrast:+.4f}")

        assert abs(contrast) > 0.01, (
            f"Velocity step should produce spatial discrimination, got |contrast| = {abs(contrast):.4f}"
        )


# ══════════════════════════════════════════════════════════════════════
#  TEST 6: Psi encoding correctness
# ══════════════════════════════════════════════════════════════════════

class TestPsiEncoding:
    """Verify the angle encoding correctly captures temporal dynamics."""

    def test_growing_instability_positive_psi(self):
        """When flux is growing (phi > phi_prev), psi should be positive."""
        angle_mapper = AngleMapper()

        phi_dict = {
            'phi_horizontal': np.array([[1.5, 0.5], [0.5, 0.5]]),
            'phi_vertical': np.array([[1.5, 0.5], [0.5, 0.5]]),
        }
        phi_dict_prev = {
            'phi_horizontal': np.array([[0.3, 0.5], [0.5, 0.5]]),
            'phi_vertical': np.array([[0.3, 0.5], [0.5, 0.5]]),
        }

        full_h = phi_dict['phi_horizontal']
        full_v = phi_dict['phi_vertical']
        score_h = np.clip(full_h / max(full_h.max(), 1e-10), 0, 1)
        score_v = np.clip(full_v / max(full_v.max(), 1e-10), 0, 1)
        _, _, psi_h, psi_v = angle_mapper.map_to_angles(
            score_h, score_v, phi_dict_prev, phi_dict, 0.5, 3.0
        )

        assert psi_h[0, 0] > 0.5, f"Growing cell should have psi > 0.5, got {psi_h[0, 0]:.4f}"
        assert abs(psi_h[1, 1]) < 0.1, f"Stable cell should have psi ~ 0, got {psi_h[1, 1]:.4f}"


# ══════════════════════════════════════════════════════════════════════
#  TEST 7: Diagnostic — QAOA convergence behavior
# ══════════════════════════════════════════════════════════════════════

class TestQAOAConvergence:
    """Diagnostic tests for understanding QAOA optimization behavior."""

    def test_qaoa_converges_for_simple_hamiltonian(self):
        """For a simple all-positive Z Hamiltonian, QAOA should converge to
        high P(|1>) (ground state of H = Σ h_i Z_i with h_i > 0).

        The minimum energy is achieved when all qubits are |1>
        (eigenvalue of Z for |1> is -1, so E = Σ h_i * (-1) < 0).
        """
        dim = 2
        hp = {
            "H_edges": (np.full((dim, dim), 5.0), np.full((dim, dim), 5.0)),
            "C_edges": (np.zeros((dim, dim)), np.zeros((dim, dim))),
            "K_plaquettes": np.zeros((dim, dim)),
        }

        theta_h = np.full((dim, dim), 0.8)
        theta_v = np.full((dim, dim), 0.8)
        psi_h = np.full((dim, dim), 0.5)
        psi_v = np.full((dim, dim), 0.5)

        # Unseeded COBYLA: over 10 draws the mean P(|1>) measured 0.829 with
        # a minimum of 0.676, so a single draw clears the 0.7 bar only ~90%
        # of the time. The statement is about convergence, not about one run.
        REPEATS = 5
        avgs = np.array([
            np.mean(run_qaoa(hp, theta_h, theta_v, psi_h, psi_v, K_opt=200))
            for _ in range(REPEATS)
        ])
        avg_p1 = float(np.median(avgs))

        initial_p1 = np.sin(0.8 / 2) ** 2  # ≈ 0.147

        print(f"\n  Simple Hamiltonian convergence:")
        print(f"  Initial P(|1>) = sin²(0.4) = {initial_p1:.4f}")
        print(f"  QAOA avg P(|1>) per draw = {np.round(avgs, 4)}")
        print(f"  median = {avg_p1:.4f}")
        print(f"  Improvement: {avg_p1 - initial_p1:+.4f}")

        # QAOA should significantly increase P(|1>) from initial state
        assert avg_p1 > initial_p1 + 0.2, (
            f"QAOA should increase P(|1>) from {initial_p1:.3f}, got "
            f"{avg_p1:.3f} (median of {REPEATS} draws)"
        )
        assert avg_p1 > 0.7, (
            f"QAOA should converge to high P(|1>) for all-positive Z, got "
            f"{avg_p1:.3f} (median of {REPEATS} draws)"
        )

    def test_k_opt_30_still_converges_with_psi(self):
        """With K_opt=30 (training setting) and non-zero psi, QAOA should
        still make a meaningful improvement over the initial state.

        This is the actual training regime — if this fails, the QAOA is
        not contributing during training.
        """
        dim = 2
        # Clear contrast: cell (1,1) hot, others cold
        H_h = np.array([[-3.0, -3.0], [-3.0, 6.0]])
        H_v = np.array([[-3.0, -3.0], [-3.0, 6.0]])
        hp = {
            "H_edges": (H_h, H_v),
            "C_edges": (np.full((dim, dim), 0.3), np.full((dim, dim), 0.3)),
            "K_plaquettes": np.zeros((dim, dim)),
        }

        theta_h = np.full((dim, dim), 0.8)
        theta_v = np.full((dim, dim), 0.8)
        psi_h = np.full((dim, dim), 0.5)  # non-zero psi for gradient
        psi_v = np.full((dim, dim), 0.5)

        marginals = run_qaoa(hp, theta_h, theta_v, psi_h, psi_v, K_opt=30)
        initial = initial_state_marginals(theta_h, theta_v)

        n = dim * dim
        prob_h = marginals[:n].reshape(dim, dim)
        prob_v = marginals[n:].reshape(dim, dim)
        prob_map = np.maximum(prob_h, prob_v)

        p_hot = prob_map[1, 1]
        p_cold = np.mean([prob_map[0, 0], prob_map[0, 1], prob_map[1, 0]])
        diff_from_initial = np.max(np.abs(marginals - initial))

        print(f"\n  K_opt=30 convergence:")
        print(f"  Prob map: {prob_map}")
        print(f"  P(hot)={p_hot:.4f}, P(cold)={p_cold:.4f}")
        print(f"  Max diff from initial state: {diff_from_initial:.4f}")

        # At minimum, QAOA should produce different output than initial state
        assert diff_from_initial > 0.03, (
            f"QAOA with K_opt=30 should modify initial state, diff={diff_from_initial:.4f}"
        )

    def test_k_opt_30_with_psi_zero_limited(self):
        """With K_opt=30 and psi=0, QAOA has zero analytical gradient.
        COBYLA must use finite differences only, which is slower.

        This tests whether the QAOA still works in the worst case
        (first VQA call where psi=0).
        """
        dim = 2
        H_h = np.array([[-3.0, -3.0], [-3.0, 6.0]])
        H_v = np.array([[-3.0, -3.0], [-3.0, 6.0]])
        hp = {
            "H_edges": (H_h, H_v),
            "C_edges": (np.full((dim, dim), 0.3), np.full((dim, dim), 0.3)),
            "K_plaquettes": np.zeros((dim, dim)),
        }

        theta_h = np.full((dim, dim), 0.8)
        theta_v = np.full((dim, dim), 0.8)

        # psi = 0: zero analytical gradient
        marginals = run_qaoa(hp, theta_h, theta_v,
                             np.zeros((dim, dim)), np.zeros((dim, dim)),
                             K_opt=30)
        initial = initial_state_marginals(theta_h, theta_v)
        diff = np.max(np.abs(marginals - initial))

        print(f"\n  K_opt=30, psi=0 (worst case):")
        print(f"  Initial: {['%.3f' % v for v in initial]}")
        print(f"  QAOA:    {['%.3f' % v for v in marginals]}")
        print(f"  Max diff: {diff:.4f}")

        # This tests the actual capability — with psi=0, the QAOA
        # may or may not improve. We just measure and report.
        # The key insight is: if this shows minimal improvement, it
        # confirms that psi≠0 is needed for QAOA to be effective.
        if diff < 0.05:
            print("  WARNING: QAOA barely modifies output with psi=0 and K_opt=30")
            print("  This suggests psi encoding is essential for QAOA effectiveness")


# ══════════════════════════════════════════════════════════════════════
#  TEST 8: Threshold effect on Z bias discrimination
# ══════════════════════════════════════════════════════════════════════

class TestThresholdEffect:
    """Verify that threshold_amr controls Z bias discrimination power."""

    def test_z_bias_scale_is_threshold_dependent(self):
        """The Z bias is alpha_z * (score - threshold_amr) with an alpha_z
        that is itself a function of threshold_amr, so the bias magnitude is
        NOT monotone in the threshold.

        Mechanism: alpha_z is normalised by C_scale = median(nonzero |C|,
        |K|), and |C| carries the Gaussian uncertainty window
        exp(-((score - threshold_amr)/sigma)^2). Moving the threshold away
        from the score support kills |C|, C_scale drops to the |K| floor,
        and alpha_z drops with it.

        On this shear layer the score takes exactly two values, 0 and 0.5:
          threshold = 0.20  ->  max|C| ~ 1.2e+01, alpha_z ~ 8.8e-01
          threshold = 0.95  ->  max|C| ~ 2.4e-84, alpha_z ~ 5.1e-03
        i.e. the same field and the same score give a Z bias whose scale
        moves by more than two orders of magnitude with the threshold alone.

        Requires spatially varying fields: for uniform fields C_scale = 0
        and H_edges vanishes regardless of threshold.
        """
        grid = PeriodicGrid(PHYSICS_N, L)
        nu = grid.L / RE
        eta_mhd = grid.L / RM
        phys_mapper = PhysicalMapper(cs=CS, nu=nu, eta_mhd=eta_mhd,
                                      beta_curl=BETA_MIC,
                                      beta_xpoint=BETA_MIC, dx=grid.dx)
        sim = MockSolver(grid)

        # Shear layer: vx jumps at row N//2, creating non-zero C_edges
        N = PHYSICS_N
        vx = np.full((N, N), 0.1)
        vx[:N//2, :] = 1.5
        fields = {
            'vx': vx,
            'vy': np.full((N, N), 0.1),
            'Bx': np.full((N, N), 0.5),
            'By': np.full((N, N), 0.5),
            'Jz': np.zeros((N, N)),
        }
        score = phys_mapper.physical_score(fields)

        hp_low = phys_mapper.compute_coefficients(
            sim, score, fields, 0.2
        )
        hp_high = phys_mapper.compute_coefficients(
            sim, score, fields, 0.95
        )

        H_low = hp_low['H_edges'][0]    # alpha_z(0.2)  * (score - 0.2)
        H_high = hp_high['H_edges'][0]  # alpha_z(0.95) * (score - 0.95)

        def alpha_of(H, thr):
            """Recover alpha_z, and check the bias is linear in the score."""
            d = score - thr
            m = np.abs(d) > 1e-12
            ratios = H[m] / d[m]
            assert np.allclose(ratios, ratios[0], rtol=1e-9), (
                "H_edges is not alpha_z * (score - threshold) at fixed "
                "threshold — the linear form itself has changed"
            )
            return float(ratios[0])

        a_low, a_high = alpha_of(H_low, 0.2), alpha_of(H_high, 0.95)
        C_low = max(np.max(np.abs(hp_low['C_edges'][0])),
                    np.max(np.abs(hp_low['C_edges'][1])))
        C_high = max(np.max(np.abs(hp_high['C_edges'][0])),
                     np.max(np.abs(hp_high['C_edges'][1])))

        print(f"\n  Z bias threshold effect:")
        print(f"  score values           = {np.unique(np.round(score, 6))}")
        print(f"  H(threshold=0.2) mean  = {np.mean(H_low):.4f}")
        print(f"  H(threshold=0.95) mean = {np.mean(H_high):.4f}")
        print(f"  alpha_z(0.2)           = {a_low:.6e}")
        print(f"  alpha_z(0.95)          = {a_high:.6e}")
        print(f"  max|C| at 0.2          = {C_low:.6e}")
        print(f"  max|C| at 0.95         = {C_high:.6e}")

        # 1. the Z weight is not a constant of the mapper
        assert a_low > 50 * a_high, (
            f"alpha_z must depend strongly on the threshold: "
            f"alpha(0.2)={a_low:.3e}, alpha(0.95)={a_high:.3e}"
        )
        # 2. the cause is the windowed ZZ coupling entering the Z
        #    normalisation, not the score, which is identical in both calls
        assert C_low > 1.0, "ZZ coupling must be alive when the threshold " \
                            "sits near the score support"
        assert C_high < 1e-30, "ZZ coupling must be annihilated when the " \
                               "threshold sits far from the score support"


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
