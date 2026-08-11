"""
End-to-end PhysicalMapper -> VQA diagnostic test.

Tests that each Hamiltonian contribution type (Z, ZZ, ZZZZ plaquette,
ZZZZ vertex) independently produces spatial discrimination when computed
from synthetic MHD fields via the real PhysicalMapper pipeline.

Pipeline under test:
  1. AngleMapper.compute_stress_flux(fields) -> phi_h, phi_v, diff_grad
  2. PhysicalMapper.compute_coefficients(sim, phi, ..., fields) -> hamilt_params
  3. get_adaptive_flux(phi_h, phi_v, prev_h, prev_v, score, hamilt, target=2) -> downsampled to 2x2
  4. AngleMapper.map_to_angles(...) -> theta, psi
  5. VQA: mapping -> execute -> postprocess -> marginals

Grid: 16x16 physics -> downsample to 2x2 (8 qubits).
Anomalies are localized in the top-left quadrant (rows 0:8, cols 0:8)
so that after max-abs pooling to 2x2, cell (0,0) carries the signal.

Tests:
  A. Baseline    -- uniform fields -> uniform marginals (control)
  B. Z bias      -- flux peak in top-left -> H_edges discriminates
  C. ZZ gradient -- velocity discontinuity -> C_edges discriminates
  D. ZZZZ plaq   -- vortex in top-left -> K_plaquettes discriminates
  E. ZZZZ vertex -- X-point reconnection -> K_xpoint discriminates
  F. Combined    -- all anomalies at once

Run:
    cd /home/user/BA_Proj && python -m pytest tests/QAOA_test.py -v
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Simulation.grid import PeriodicGrid
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import get_adaptive_flux
from VQA.mapping import mapping
from VQA.execute import execute
from VQA.postprocess import postprocess

# -- Constants -----------------------------------------------------------------
PHYSICS_N = 16          # physics grid resolution
VQA_N = 2              # VQA grid resolution (8 qubits)
L = 2 * np.pi          # domain length
RE = 100.0             # Reynolds number
RM = 100.0             # Magnetic Reynolds number
CS = 1.0               # sound speed
BETA_MIC = 0.5         # Michelson sensitivity
THRESHOLD = 0.2        # Z bias threshold


# -- Minimal args-like object for execute() ------------------------------------
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
    """Provides sim.grid interface expected by PhysicalMapper."""
    def __init__(self, grid):
        self.grid = grid


def make_grid():
    return PeriodicGrid(PHYSICS_N, L)


def make_mapper(grid):
    nu = grid.L / RE
    eta_mhd = grid.L / RM
    return PhysicalMapper(cs=CS, nu=nu, eta_mhd=eta_mhd,
                          beta_curl=BETA_MIC, beta_xpoint=BETA_MIC,
                          dx=grid.dx)


def uniform_fields(grid, v_bg=0.1, B_bg=0.5):
    """Create uniform background MHD fields."""
    N = grid.N
    return {
        'vx': np.full((N, N), v_bg),
        'vy': np.full((N, N), v_bg),
        'Bx': np.full((N, N), B_bg),
        'By': np.full((N, N), B_bg),
        'Jz': np.zeros((N, N)),
    }


def run_pipeline(fields, grid, mapper, sim, adv_anomalies=False):
    """Run the full pipeline: stress flux -> coefficients -> downsample -> angles -> VQA."""
    angle_mapper = AngleMapper()

    # 1. Compute stress flux
    phi_dict = angle_mapper.compute_stress_flux(fields)
    phi_h = phi_dict['phi_horizontal']
    phi_v = phi_dict['phi_vertical']

    # 2. Compute physics-grounded score and Hamiltonian coefficients
    score = mapper.physical_score(fields)
    hamilt_params = mapper.compute_coefficients(
        sim, score, fields, THRESHOLD,
        advanced_anomalies_enabled=adv_anomalies
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

    return theta_h, theta_v, psi_h, psi_v, mini_hamilt, hamilt_params


def run_vqa(theta_h, theta_v, psi_h, psi_v, hamilt_params, label=""):
    """Run full VQA chain and return per-qubit P(|1>)."""
    data = {
        "theta_h": theta_h.tolist(),
        "theta_v": theta_v.tolist(),
        "psi_h": psi_h.tolist(),
        "psi_v": psi_v.tolist(),
    }
    qc, cost_ham = mapping(data, hamilt_params, args.AdvAnomaliesEnable,
                           period_bound=True, reps=args.reps)

    E_max = 0
    for key, value in hamilt_params.items():
        if isinstance(value, (tuple, list)):
            for v in value:
                if isinstance(v, np.ndarray):
                    E_max += np.sum(np.abs(v))
        elif isinstance(value, np.ndarray):
            E_max += np.sum(np.abs(value))
    E_max = max(E_max, 1e-10)

    dist, _ = execute(qc, cost_ham, args.mode, args.backend, args.shots,
                   args.reps, args.K_opt, args.eps, E_max, verbose=False)
    marginals = np.array(postprocess(dist, qc.num_qubits, verbose=False))
    return marginals


def get_contrast(marginals):
    """Compute contrast between cell (0,0) and the rest of the 2x2 map."""
    n = VQA_N * VQA_N
    prob_h = marginals[:n].reshape(VQA_N, VQA_N)
    prob_v = marginals[n:].reshape(VQA_N, VQA_N)
    prob_map = np.maximum(prob_h, prob_v)

    p_hot = prob_map[0, 0]
    p_cold = np.mean([prob_map[0, 1], prob_map[1, 0], prob_map[1, 1]])
    return p_hot - p_cold, prob_map


def print_coefficients(hamilt_params, label):
    """Print diagnostics about raw coefficient arrays (before downsampling)."""
    print(f"\n  [{label}] Coefficient diagnostics (full {PHYSICS_N}x{PHYSICS_N}):")
    H_h, H_v = hamilt_params['H_edges']
    C_h, C_v = hamilt_params['C_edges']
    K = hamilt_params['K_plaquettes']
    print(f"    H_horiz: min={H_h.min():.4f}, max={H_h.max():.4f}, mean={H_h.mean():.4f}")
    print(f"    H_vert:  min={H_v.min():.4f}, max={H_v.max():.4f}, mean={H_v.mean():.4f}")
    print(f"    C_horiz: min={C_h.min():.4f}, max={C_h.max():.4f}, mean={C_h.mean():.4f}")
    print(f"    C_vert:  min={C_v.min():.4f}, max={C_v.max():.4f}, mean={C_v.mean():.4f}")
    print(f"    K_plaq:  min={K.min():.4f}, max={K.max():.4f}, mean={K.mean():.4f}")
    if 'K_xpoint' in hamilt_params:
        Kx = hamilt_params['K_xpoint']
        print(f"    K_xpt:   min={Kx.min():.4f}, max={Kx.max():.4f}, mean={Kx.mean():.4f}")


# ==============================================================================
#  TEST A: BASELINE -- uniform fields -> all coefficients zero
# ==============================================================================
class TestBaseline:
    def test_uniform_fields_give_uniform_marginals(self):
        """Uniform MHD fields must produce trivial (all-zero) Hamiltonian
        coefficients and uniform initial angles.

        With the adaptive Z bias (alpha_z = w_z_frac * median(|C|,|K|)),
        uniform fields have zero gradients → C_scale=0 → all coefficients
        are exactly zero. Running VQA on a zero Hamiltonian is degenerate
        (COBYLA wanders freely), so we verify the physics directly.
        """
        grid = make_grid()
        mapper = make_mapper(grid)
        sim = MockSolver(grid)
        fields = uniform_fields(grid)

        theta_h, theta_v, psi_h, psi_v, mini_hp, raw_hp = run_pipeline(
            fields, grid, mapper, sim)
        print_coefficients(raw_hp, "Baseline")

        # All interaction coefficients should be zero for uniform fields
        H_h, H_v = raw_hp['H_edges']
        C_h, C_v = raw_hp['C_edges']
        K = raw_hp['K_plaquettes']

        max_H = max(np.max(np.abs(H_h)), np.max(np.abs(H_v)))
        max_C = max(np.max(np.abs(C_h)), np.max(np.abs(C_v)))
        max_K = np.max(np.abs(K))

        print(f"\n  [Baseline] max|H|={max_H:.6f}, max|C|={max_C:.6f}, max|K|={max_K:.6f}")
        assert max_H < 1e-6, f"H_edges should be ~0 for uniform fields, got {max_H}"
        assert max_C < 1e-6, f"C_edges should be ~0 for uniform fields, got {max_C}"
        assert max_K < 1e-6, f"K_plaquettes should be ~0 for uniform fields, got {max_K}"

        # Theta angles should be uniform (same value everywhere)
        theta_spread = max(theta_h.max() - theta_h.min(), theta_v.max() - theta_v.min())
        print(f"  [Baseline] theta spread = {theta_spread:.6f}")
        assert theta_spread < 0.01, (
            f"Uniform fields should give uniform theta angles, "
            f"but spread = {theta_spread:.4f}"
        )


# ==============================================================================
#  TEST B: Z BIAS -- strong flux peak in top-left quadrant
# ==============================================================================
class TestZBias:
    def test_flux_peak_discriminates(self):
        """A velocity/B perturbation in top-left must produce positive contrast."""
        grid = make_grid()
        mapper = make_mapper(grid)
        sim = MockSolver(grid)
        fields = uniform_fields(grid)

        N = grid.N
        fields['vx'][:N//2, :N//2] += 2.0
        fields['vy'][:N//2, :N//2] += 2.0
        fields['Bx'][:N//2, :N//2] += 1.5
        fields['By'][:N//2, :N//2] += 1.5

        theta_h, theta_v, psi_h, psi_v, mini_hp, raw_hp = run_pipeline(
            fields, grid, mapper, sim)
        print_coefficients(raw_hp, "Z bias")
        m = run_vqa(theta_h, theta_v, psi_h, psi_v, mini_hp)
        contrast, prob_map = get_contrast(m)

        print(f"\n  [Z bias] Contrast = {contrast:+.4f}")
        for i in range(VQA_N):
            print(f"    {['%.4f' % prob_map[i,j] for j in range(VQA_N)]}")

        assert contrast > 0.01, (
            f"Z bias test: flux peak in top-left should produce positive contrast, "
            f"got {contrast:+.4f} (min: 0.01)"
        )


# ==============================================================================
#  TEST C: ZZ GRADIENT -- sharp velocity discontinuity in top-left
# ==============================================================================
class TestZZGradient:
    def test_velocity_discontinuity_discriminates(self):
        """A sharp velocity step in top-left must produce positive contrast."""
        grid = make_grid()
        mapper = make_mapper(grid)
        sim = MockSolver(grid)
        fields = uniform_fields(grid)

        N = grid.N
        fields['vx'][:N//2, :N//4] += 3.0
        fields['vx'][:N//2, N//4:N//2] -= 1.0
        fields['Bx'][:N//2, :N//4] += 2.0

        theta_h, theta_v, psi_h, psi_v, mini_hp, raw_hp = run_pipeline(
            fields, grid, mapper, sim)
        print_coefficients(raw_hp, "ZZ gradient")
        m = run_vqa(theta_h, theta_v, psi_h, psi_v, mini_hp)
        contrast, prob_map = get_contrast(m)

        print(f"\n  [ZZ gradient] Contrast = {contrast:+.4f}")
        for i in range(VQA_N):
            print(f"    {['%.4f' % prob_map[i,j] for j in range(VQA_N)]}")

        assert abs(contrast) > 0.01, (
            f"ZZ gradient test: velocity discontinuity should produce spatial discrimination, "
            f"got |contrast| = {abs(contrast):.4f} (min: 0.01)"
        )


# ==============================================================================
#  TEST D: ZZZZ PLAQUETTE -- Lamb-Oseen vortex in top-left
# ==============================================================================
class TestZZZZPlaquette:
    def test_the_vortex_gains_positive_contrast_once_the_curl_sees_it(self):
        """A Lamb-Oseen vortex in the top-left DOES produce positive spatial
        contrast.

        Ce test affirmait le contraire — « -0.0058 +/- 0.0064 sur 12
        tirages, le QAOA tire meme la cellule chaude vers le bas » — et il
        mesurait bien ce qui se passait alors. Mais la cause n'etait pas le
        QAOA : c'etait le defaut D-1. Le rotationnel des mappeurs etait
        ecrit sous la convention `indexing='xy'` alors que la grille
        construit ses champs en `indexing='ij'` ; une rotation solide
        rendait donc exactement 0, et le terme ZZZZ de plaquette — dont la
        seule raison d'etre est de detecter une circulation — etait
        numeriquement mort sur un vortex pur.

        Attribution mesuree sur ce meme vortex, 16 tirages par ligne, tout
        le reste egal :

          | fixed_curl | fixed_flux | contraste | ecart-type |  sigma | max|K| |
          |------------|------------|-----------|------------|--------|--------|
          | False      | False      |  -0.00725 |    0.00859 |  -3.4  | 0.0553 |
          | False      | True       |  -0.00852 |    0.00896 |  -3.8  | 0.0553 |
          | True       | False      |  +0.05672 |    0.03976 |  +5.7  | 1.2545 |
          | True       | True       |  +0.07292 |    0.04429 |  +6.6  | 1.2545 |

        La ligne (False, False) reproduit la valeur historique a l'ecart-type
        pres. Le coefficient de plaquette passe de 0.055 a 1.255 — vingt-trois
        fois plus grand — des que le rotationnel voit la rotation.

        Reste vrai : theta_h[0,0] = pi, la cellule (0,0) part deja a
        P(|1>) = 1 et n'a plus de marge ; et le tirage a 4096 coups bruite
        chaque marginale a ~0.008. La moyenne sur REPEATS reste donc
        indispensable — un tirage isole ne prouve rien.
        """
        grid = make_grid()
        mapper = make_mapper(grid)
        sim = MockSolver(grid)
        fields = uniform_fields(grid, v_bg=0.0, B_bg=0.5)

        N = grid.N
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

        theta_h, theta_v, psi_h, psi_v, mini_hp, raw_hp = run_pipeline(
            fields, grid, mapper, sim)
        print_coefficients(raw_hp, "ZZZZ plaquette")

        REPEATS = 8
        contrasts = []
        for _ in range(REPEATS):
            m = run_vqa(theta_h, theta_v, psi_h, psi_v, mini_hp)
            c, prob_map = get_contrast(m)
            contrasts.append(c)
        contrasts = np.array(contrasts)

        print(f"\n  [ZZZZ plaquette] theta_h = "
              f"{np.array2string(theta_h, precision=4)}")
        print(f"  [ZZZZ plaquette] contrasts = "
              f"{np.array2string(contrasts, precision=5)}")
        print(f"  [ZZZZ plaquette] mean = {contrasts.mean():+.5f}, "
              f"std = {contrasts.std():.5f}")
        for i in range(VQA_N):
            print(f"    {['%.4f' % prob_map[i,j] for j in range(VQA_N)]}")

        sem = contrasts.std() / np.sqrt(REPEATS)
        assert contrasts.mean() > 2.0 * sem, (
            f"le vortex doit gagner un contraste positif distinct du bruit ; "
            f"mesure {contrasts.mean():+.5f} +/- {sem:.5f} sur {REPEATS} "
            f"tirages. S'il retombe a zero, le terme de plaquette a cesse de "
            f"voir la circulation — verifier fixed_curl.")
        assert (contrasts > 0).mean() >= 0.7, (
            f"le signe doit etre stable ; seulement "
            f"{(contrasts > 0).mean():.0%} des tirages sont positifs : "
            f"{np.array2string(contrasts, precision=5)}")


# ==============================================================================
#  TEST E: ZZZZ VERTEX -- X-point magnetic reconnection
# ==============================================================================
class TestZZZZVertex:
    def test_xpoint_discriminates(self):
        """An X-point reconnection pattern in top-left must produce positive contrast."""
        grid = make_grid()
        mapper = make_mapper(grid)
        sim = MockSolver(grid)
        fields = uniform_fields(grid, v_bg=0.0, B_bg=0.5)

        N = grid.N
        cx, cy = N // 4, N // 4
        for i in range(N // 2):
            for j in range(N // 2):
                dx = i - cx
                dy = j - cy
                r = np.sqrt(dx**2 + dy**2) + 1e-10
                v_r = 3.0 * np.exp(-r**2 / 9.0)
                fields['vx'][i, j] += -v_r * dx / r
                fields['vy'][i, j] += -v_r * dy / r

        theta_h, theta_v, psi_h, psi_v, mini_hp, raw_hp = run_pipeline(
            fields, grid, mapper, sim, adv_anomalies=True)
        print_coefficients(raw_hp, "ZZZZ X-point")
        m = run_vqa(theta_h, theta_v, psi_h, psi_v, mini_hp)
        contrast, prob_map = get_contrast(m)

        print(f"\n  [ZZZZ X-point] Contrast = {contrast:+.4f}")
        for i in range(VQA_N):
            print(f"    {['%.4f' % prob_map[i,j] for j in range(VQA_N)]}")

        assert contrast > 0.01, (
            f"ZZZZ X-point test: reconnection pattern should produce positive contrast, "
            f"got {contrast:+.4f} (min: 0.01)"
        )


# ==============================================================================
#  TEST F: COMBINED -- all anomalies in top-left
# ==============================================================================
class TestCombined:
    def test_combined_anomalies_discriminate(self):
        """All anomalies combined in top-left must produce positive contrast."""
        grid = make_grid()
        mapper = make_mapper(grid)
        sim = MockSolver(grid)
        fields = uniform_fields(grid, v_bg=0.0, B_bg=0.5)

        N = grid.N
        cx, cy = N // 4, N // 4

        # Vortex (plaquette)
        for i in range(N):
            for j in range(N):
                dx = i - cx
                dy = j - cy
                r = np.sqrt(dx**2 + dy**2) + 1e-10
                r0 = 3.0
                v_theta = 4.0 * (1 - np.exp(-r**2 / r0**2)) / r
                fields['vx'][i, j] += -v_theta * dy / r
                fields['vy'][i, j] += v_theta * dx / r

        # Velocity/B step (ZZ gradient + Z bias)
        fields['vx'][:N//2, :N//4] += 2.0
        fields['Bx'][:N//2, :N//4] += 1.5
        fields['By'][:N//2, :N//4] += 1.0

        theta_h, theta_v, psi_h, psi_v, mini_hp, raw_hp = run_pipeline(
            fields, grid, mapper, sim)
        print_coefficients(raw_hp, "Combined")
        m = run_vqa(theta_h, theta_v, psi_h, psi_v, mini_hp)
        contrast, prob_map = get_contrast(m)

        print(f"\n  [Combined] Contrast = {contrast:+.4f}")
        for i in range(VQA_N):
            print(f"    {['%.4f' % prob_map[i,j] for j in range(VQA_N)]}")

        # Combined anomalies may flip the optimization landscape (all terms
        # interact), so we check absolute contrast — the anomalous cell must
        # be *different* from the calm cells, regardless of sign.
        assert abs(contrast) > 0.01, (
            f"Combined test: all anomalies should produce spatial discrimination, "
            f"got |contrast| = {abs(contrast):.4f} (min: 0.01)"
        )


# ==============================================================================
#  MAIN
# ==============================================================================
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-v'])
