"""Shared utilities for all publication figure scripts."""
import sys, os
import numpy as np
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter, label

# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------

from types import SimpleNamespace
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import get_adaptive_flux
from VQA.runtime import VQARuntime
from call_vqa_shell import call_vqa_shell
from Simulation.refinement import run_adaptive_vqa, run_adaptive_classical
from hyperparams_loader import load_hyperparams
from math import log

# ── Publication style ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLE = {
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 10,
    'legend.framealpha': 0.8,
    'legend.edgecolor': '0.8',
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

def apply_style():
    plt.rcParams.update(STYLE)

COLORS = {
    'classical': '#4878CF',   # muted blue
    'qaoa': '#D65F5F',        # muted red
    'ground_truth': '#59A14F', # muted green
    'tie': '#9E9E9E',
    'smoothed': '#ECA63D',    # muted orange
    'dns': '#333333',
}

# Short labels for publication
LABELS = {
    'classical': 'Classical',
    'qaoa': 'Q-HAS',
}


# ── Phase-aware scenario filtering ──
# When FIGURE_PHASE is set (by generate_figures.sh), figure scripts should
# only run scenarios appropriate for that training phase.
#   Phase 1: KH, Tearing, Orszag-Tang, MHD Rotor
#   Phase 2: (reserved)
#   Phase 3: all scenarios

SCENARIOS_PHASE1 = {
    'init_kelvin_helmholtz', 'init_harris_tearing',
    'init_orszag_tang', 'init_mhd_rotor',
}
SCENARIOS_PHASE2 = {
    'init_orszag_tang', 'init_mhd_rotor',
}
SCENARIOS_PHASE3 = SCENARIOS_PHASE1 | SCENARIOS_PHASE2

_PHASE_SCENARIOS = {
    '1': SCENARIOS_PHASE1,
    '2': SCENARIOS_PHASE2,
    '3': SCENARIOS_PHASE3,
}

FIGURE_PHASE = os.environ.get('FIGURE_PHASE', None)

# ── Figure output directory ──
# When FIGURE_PHASE is set, save directly to figures/phase<N>/ instead of figures/
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
if FIGURE_PHASE:
    FIG_DIR = os.path.join(_PROJECT_ROOT, 'figures', f'phase{FIGURE_PHASE}')
else:
    FIG_DIR = os.path.join(_PROJECT_ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def _extract_init_name(item, item_key=None):
    """Extract the init function name from various scenario formats."""
    if isinstance(item, dict):
        return item.get('init', item.get('scenario', ''))
    elif isinstance(item, (list, tuple)):
        # ('init_func', n_steps) or ('label', 'init_func', n_steps)
        for elem in item:
            if isinstance(elem, str) and elem.startswith('init_'):
                return elem
        return item[0] if item else ''
    elif isinstance(item, str):
        return item
    return ''


def _is_scenario_allowed(init_name, allowed):
    """Check if init_name is in the allowed set."""
    return init_name in allowed


def filter_scenarios(scenario_list):
    """Filter a list of scenario tuples by active phase.

    Accepts various formats:
      - [('Label', 'init_func', n_steps), ...]
      - [('init_func', n_steps), ...]

    When FIGURE_PHASE is not set, returns the list unchanged.
    """
    if FIGURE_PHASE is None:
        return scenario_list

    allowed = _PHASE_SCENARIOS.get(FIGURE_PHASE)
    if allowed is None:
        return scenario_list

    filtered = []
    for item in scenario_list:
        name = _extract_init_name(item)
        if _is_scenario_allowed(name, allowed):
            filtered.append(item)

    if not filtered:
        print(f"[WARN] Phase {FIGURE_PHASE} filter removed all scenarios — "
              f"running unfiltered")
        return scenario_list

    skipped = len(scenario_list) - len(filtered)
    if skipped > 0:
        print(f"[Phase {FIGURE_PHASE}] Keeping {len(filtered)}/{len(scenario_list)} "
              f"scenarios (skipped {skipped})")
    return filtered


def filter_scenarios_dict(scenario_dict):
    """Filter a dict of scenarios by active phase.

    Accepts various dict formats:
      - {'Label': ('init_func', n_steps)}
      - {'Label': 'init_func'}
      - {'Label': {'init': 'init_func', ...}}

    When FIGURE_PHASE is not set, returns the dict unchanged.
    """
    if FIGURE_PHASE is None:
        return scenario_dict

    allowed = _PHASE_SCENARIOS.get(FIGURE_PHASE)
    if allowed is None:
        return scenario_dict

    filtered = {}
    for key, val in scenario_dict.items():
        name = _extract_init_name(val, item_key=key)
        if _is_scenario_allowed(name, allowed) or _is_scenario_allowed(key, allowed):
            filtered[key] = val

    if not filtered:
        print(f"[WARN] Phase {FIGURE_PHASE} filter removed all scenarios — "
              f"running unfiltered")
        return scenario_dict

    skipped = len(scenario_dict) - len(filtered)
    if skipped > 0:
        print(f"[Phase {FIGURE_PHASE}] Keeping {len(filtered)}/{len(scenario_dict)} "
              f"scenarios (skipped {skipped})")
    return filtered


def phase_allows_scenario(init_name):
    """Check if a single scenario is allowed by the current phase filter.

    Use this for scripts that hardcode a single scenario inline.
    Returns True if no filter is active or the scenario is allowed.
    """
    if FIGURE_PHASE is None:
        return True
    allowed = _PHASE_SCENARIOS.get(FIGURE_PHASE)
    if allowed is None:
        return True
    return init_name in allowed


# ── Simulation helpers ──

def make_sim(N, scenario, n_steps, Re=800, Rm=800):
    """Run *n_steps* of MHD and return (sim, Phi_prev).

    Phi_prev is the stress-flux snapshot taken just before the last
    time-step, so that callers can compute a meaningful temporal
    derivative  ΔΦ = Φ_current − Φ_prev  for the Phase Boost ψ.
    """
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=Re, Rm=Rm)
    getattr(sim, scenario)()
    _mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    Phi_prev = None
    for i in range(n_steps):
        if i == n_steps - 1:                       # penultimate state
            Phi_prev = _mapper.compute_stress_flux(sim.get_fluxes())
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    return sim, Phi_prev


def make_sim_with_history(N, scenario, n_steps, Re=800, Rm=800):
    """Run simulation recording energy at each step. Returns (sim, energy_history)."""
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=Re, Rm=Rm)
    getattr(sim, scenario)()
    for _ in range(n_steps):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=True)
    return sim


def ground_truth_errors(sim, N, n_blocks=None):
    """Pixel-level error indicator map (N x N).

    Combines gradient magnitude (1st derivative — shear layers, jumps)
    and Laplacian magnitude (2nd derivative — oscillations, under-resolution)
    across all 4 MHD fields.

    Returns an (N, N) array of error magnitudes at every pixel.
    The n_blocks argument is accepted for backward compatibility but ignored.
    """
    state = sim.get_fluxes()
    total = np.zeros((N, N))
    for key in ['vx', 'vy', 'Bx', 'By']:
        f = state[key]
        # Gradient magnitude (1st derivative) — detects jumps, shear layers
        grad_x = np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)  # central diff
        grad_y = np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)
        grad_mag = grad_x**2 + grad_y**2
        # Laplacian magnitude (2nd derivative) — detects oscillations
        fpp_xx = np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)
        fpp_yy = np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)
        lap_mag = fpp_xx**2 + fpp_yy**2
        total += grad_mag + lap_mag
    return np.sqrt(total)


def classical_block_scores(sim, N, n_blocks):
    state = sim.get_fluxes()
    full_score = AngleMapper.classical_score(state)
    bh, bw = N // n_blocks, N // n_blocks
    scores = np.zeros((n_blocks, n_blocks))
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            scores[bi, bj] = np.max(full_score[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw])
    return scores


def noisy_classical_scores(sim, N, n_blocks, noise_std, rng):
    state = sim.get_fluxes()
    full_score = AngleMapper.classical_score(state)
    noisy = full_score + noise_std * rng.standard_normal(full_score.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    bh, bw = N // n_blocks, N // n_blocks
    scores = np.zeros((n_blocks, n_blocks))
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            scores[bi, bj] = np.max(noisy[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw])
    return scores


def smoothed_classical_scores(sim, N, n_blocks, sigma_smooth=3.0,
                              noise_std=0.0, rng=None):
    """Stronger classical baseline: Gaussian blur + block max.
    This is a classical enrichment that adds spatial regularization,
    similar in spirit to what the QAOA Hamiltonian does."""
    state = sim.get_fluxes()
    full_score = AngleMapper.classical_score(state)
    if noise_std > 0 and rng is not None:
        full_score = full_score + noise_std * rng.standard_normal(full_score.shape)
        full_score = np.clip(full_score, 0.0, 1.0)
    smoothed = gaussian_filter(full_score, sigma=sigma_smooth, mode='wrap')
    bh, bw = N // n_blocks, N // n_blocks
    scores = np.zeros((n_blocks, n_blocks))
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            scores[bi, bj] = np.max(smoothed[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw])
    return scores


# ── Best hyperparameters loaded from best_hyperparams.json ──
# Default: last phase, lambda_cost=0.20, rank 0 (best trial).
# Override with: load_hyperparams(phase='phase1', lambda_cost='lambda_0.30', rank=1)
# Regenerate JSON: ./scripts/extract_best_hyperparams.sh
_phase_arg = f"phase{FIGURE_PHASE}" if FIGURE_PHASE else None
TRAINED_PARAMS = load_hyperparams(phase=_phase_arg)
CLASSICAL_PARAMS = load_hyperparams(phase=_phase_arg, method='classical')

# If params were frozen during quantum training, they won't appear in the
# quantum params.  Fill in defaults matching what was frozen in TrainHyperParam_v2.
_FROZEN_DEFAULTS = {
    'threshold_amr': CLASSICAL_PARAMS.get('threshold_amr', 0.15),
    'gamma_hydro': 2.0,
    'gamma_mag': 0.5,
    'kappa': 10.0,
}
for _k, _v in _FROZEN_DEFAULTS.items():
    if _k not in TRAINED_PARAMS:
        TRAINED_PARAMS[_k] = _v

def _hamilt_mapper_kwargs(grid):
    """Build PhysicalMapper keyword arguments from TRAINED_PARAMS.

    Handles both single-beta (Phase 1: beta_michelson) and split-beta
    (Phase 1b+: sigma, beta_curl, beta_xpoint) parameter sets.
    """
    nu = grid.L / 800
    eta = grid.L / 800
    kwargs = dict(
        cs=1.0, nu=nu, eta_mhd=eta, dx=grid.dx,
        gamma_hydro=TRAINED_PARAMS['gamma_hydro'],
        gamma_mag=TRAINED_PARAMS['gamma_mag'],
        kappa=TRAINED_PARAMS['kappa'],
        w_z_frac=TRAINED_PARAMS['w_z_frac'],
    )
    # sigma replaces beta_grad (uncertainty width for ZZ coupling)
    kwargs['sigma'] = TRAINED_PARAMS.get('sigma', 0.05)
    kwargs['beta_curl'] = TRAINED_PARAMS['beta_curl']
    kwargs['beta_xpoint'] = TRAINED_PARAMS['beta_xpoint']
    return kwargs


def qaoa_block_scores(sim, N, n_blocks, threshold=CLASSICAL_PARAMS['threshold_amr'],
                      w_z_frac=TRAINED_PARAMS['w_z_frac'],
                      K_opt=40, noise_std=0.0, rng=None, reps=None,
                      RE_CRIT=None, RM_CRIT=None, method="COBYLA",
                      Phi_prev=None):
    """Evaluate Q-HAS on *sim* and return (n_blocks, n_blocks) QAOA scores.

    Parameters
    ----------
    Phi_prev : dict or None
        Stress-flux snapshot from the previous time-step (keys:
        ``phi_horizontal``, ``phi_vertical``).  When provided the
        Phase Boost ψ = (π/2)·tanh(β·ΔΦ/⟨|ΔΦ|⟩) is activated,
        giving the QAOA access to temporal rate-of-change information.
    """
    grid = sim.grid
    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    hm_kwargs = _hamilt_mapper_kwargs(grid)
    hm_kwargs['w_z_frac'] = w_z_frac  # override with function argument
    if RE_CRIT is not None or RM_CRIT is not None:
        HamiltMapper = PhysicalMapper(**hm_kwargs)
        if RE_CRIT is not None:
            HamiltMapper.RE_CRIT = RE_CRIT
        if RM_CRIT is not None:
            HamiltMapper.RM_CRIT = RM_CRIT
    else:
        HamiltMapper = PhysicalMapper(**hm_kwargs)
    _reps = reps if reps is not None else (n_blocks - 1) * 2
    args = SimpleNamespace(
        reps=_reps,
        mode="simulator", backend="state_vector",
        shots=1024, method=method, opt_level=1,
        AdvAnomaliesEnable=True, K_opt=K_opt, eps=1e-2,
    )
    vqa_runtime = VQARuntime(
        backend_name="state_vector", mode="simulator",
        shots=1024, opt_level=1,
    )
    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)
    full_score = AngleMapper.classical_score(physics_state)
    if noise_std > 0 and rng is not None:
        full_score = full_score + noise_std * rng.standard_normal(full_score.shape)
        full_score = np.clip(full_score, 0.0, 1.0)
    hamilt_params = HamiltMapper.compute_coefficients(
        sim, full_score, physics_state, threshold,
        advanced_anomalies_enabled=True,
    )
    phi_h, phi_v = Phi['phi_horizontal'], Phi['phi_vertical']

    # ── Phase Boost: compute ψ from temporal derivative ΔΦ ──────────
    prev_h = Phi_prev['phi_horizontal'] if Phi_prev is not None else None
    prev_v = Phi_prev['phi_vertical']   if Phi_prev is not None else None

    if prev_h is not None:
        AveragePhiDev = 0.5 * (np.mean(np.abs(phi_h - prev_h))
                                + np.mean(np.abs(phi_v - prev_v)))
        # Downsample with previous flux → get 6-tuple
        mini_h, mini_v, mini_prev_h, mini_prev_v, mini_hp, mini_score = \
            get_adaptive_flux(
                phi_h, phi_v, prev_h, prev_v, full_score, hamilt_params,
                target_dim=n_blocks, type_filter=True,
            )
        mini_Phi_prev = {'phi_horizontal': mini_prev_h,
                         'phi_vertical':   mini_prev_v}
    else:
        AveragePhiDev = None
        mini_h, mini_v, mini_hp, mini_score = get_adaptive_flux(
            phi_h, phi_v, None, None, full_score, hamilt_params,
            target_dim=n_blocks, type_filter=True,
        )
        mini_Phi_prev = None

    mini_score = np.clip(mini_score, 0.0, 1.0)
    angles = mapper.map_to_angles(
        score_h=mini_score, score_v=mini_score,
        phi_dict_prev=mini_Phi_prev,
        phi_dict={'phi_horizontal': mini_h, 'phi_vertical': mini_v},
        AveragePhiDev=AveragePhiDev, beta=1.0,
    )
    result = call_vqa_shell(
        angles, mini_hp, False, args,
        period_bound=True, vqa_runtime=vqa_runtime,
    )
    ne = n_blocks * n_blocks
    if result is not None:
        probs, _ = result
        ph = probs[:ne].reshape(n_blocks, n_blocks)
        pv = probs[ne:].reshape(n_blocks, n_blocks)
        return 0.5 * (ph + pv)
    return mini_score.copy()


def select_top_k(scores, k):
    flat = np.argsort(scores.ravel())[::-1][:k]
    nc = scores.shape[1]
    return set((idx // nc, idx % nc) for idx in flat)


# captured_fraction() removed — pixel-level overlap is computed
# by _patches_overlap_with_gt() in patches_to_metrics().


# ═══════════════════════════════════════════════════════════════════════
#  PHYSICAL METRICS — global fidelity measures (not local block overlap)
# ═══════════════════════════════════════════════════════════════════════

def compute_kinetic_energy(sim):
    """E_k = 0.5 * integral(vx^2 + vy^2) dA"""
    dx = sim.grid.dx
    return 0.5 * np.sum(sim.vx**2 + sim.vy**2) * dx**2


def compute_magnetic_energy(sim):
    """E_m = 0.5 * integral(Bx^2 + By^2) dA"""
    dx = sim.grid.dx
    return 0.5 * np.sum(sim.Bx**2 + sim.By**2) * dx**2


def compute_enstrophy(sim):
    """Omega = integral(omega_z^2) dA  where omega_z = dvy/dx - dvx/dy"""
    # Convention du depot : axis=0 est x, axis=1 est y (indexing='ij').
    # Les deux lignes lisaient l'inverse, si bien qu'omega_z valait en fait
    # dvy/dy - dvx/dx — une combinaison de deformation, nulle sur une
    # rotation solide. L'« enstrophie » tracee ne mesurait donc pas une
    # enstrophie. Le carre ne rattrape rien : ce n'est pas un signe oppose.
    dx = sim.grid.dx
    dvydx = (np.roll(sim.vy, -1, axis=0) - np.roll(sim.vy, 1, axis=0)) / (2 * dx)
    dvxdy = (np.roll(sim.vx, -1, axis=1) - np.roll(sim.vx, 1, axis=1)) / (2 * dx)
    omega_z = dvydx - dvxdy
    return np.sum(omega_z**2) * dx**2


def compute_mean_jz_squared(sim):
    """Mean square current density: <Jz^2> (proxy for magnetic dissipation)."""
    state = sim.get_fluxes()
    return np.mean(state['Jz']**2)


def field_l2_error(sim_test, sim_ref):
    """Relative L2 error across all 4 MHD fields vs reference (DNS)."""
    dx = sim_test.grid.dx
    total_err = 0.0
    total_ref = 0.0
    for f_test, f_ref in [(sim_test.vx, sim_ref.vx), (sim_test.vy, sim_ref.vy),
                           (sim_test.Bx, sim_ref.Bx), (sim_test.By, sim_ref.By)]:
        total_err += np.sum((f_test - f_ref)**2) * dx**2
        total_ref += np.sum(f_ref**2) * dx**2
    return np.sqrt(total_err / (total_ref + 1e-30))


# ═══════════════════════════════════════════════════════════════════════
#  SPATIAL COHERENCE METRICS — patch structure analysis
# ═══════════════════════════════════════════════════════════════════════

def selection_to_mask(selection, n_blocks):
    """Convert a set of (i,j) block indices to a binary mask."""
    mask = np.zeros((n_blocks, n_blocks), dtype=int)
    for i, j in selection:
        mask[i, j] = 1
    return mask


def count_connected_components(mask):
    """Number of 4-connected components in a binary block mask."""
    labeled, n_components = label(mask)
    return n_components


def compute_fragmentation(mask):
    """Fragmentation = n_components / n_selected.
    1.0 = maximally fragmented (every block is isolated).
    Lower = more spatially coherent."""
    n_selected = np.sum(mask)
    if n_selected == 0:
        return 0.0
    n_comp = count_connected_components(mask)
    return n_comp / n_selected


def compute_perimeter_area_ratio(mask):
    """Perimeter / area ratio of the selected blocks.
    Lower = more compact/coherent selection."""
    n_selected = np.sum(mask)
    if n_selected == 0:
        return 0.0
    # Count exposed edges (4-connectivity)
    perimeter = 0
    rows, cols = mask.shape
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] == 1:
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                        perimeter += 1  # boundary edge
                    elif mask[ni, nj] == 0:
                        perimeter += 1
    return perimeter / n_selected


# ═══════════════════════════════════════════════════════════════════════
#  TEMPORAL STABILITY — for early detection analysis
# ═══════════════════════════════════════════════════════════════════════

def selection_jaccard(sel_a, sel_b):
    """Jaccard similarity between two block selections."""
    if len(sel_a) == 0 and len(sel_b) == 0:
        return 1.0
    intersection = len(sel_a & sel_b)
    union = len(sel_a | sel_b)
    return intersection / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
#  HIERARCHICAL AMR — run_adaptive_vqa vs run_adaptive_classical
# ═══════════════════════════════════════════════════════════════════════

def _compute_depths(N, target_dim=2, min_size=6):
    """Compute solve_max_depth for given grid size."""
    return max(1, int(log(N / min_size) / log(target_dim)))


def _patches_total_fine_pixels(patches, N, target_dim=2):
    """Count total effective fine-resolution pixels selected by AMR patches.

    leaf_depth patches are at full resolution (local_factor=1).
    coarse_leaf patches are at coarse resolution (local_factor from depth).
    Returns (n_fine_pixels, n_total_pixels, fine_fraction).
    """
    from Simulation.utils import compute_local_factor

    max_depth = max((p['depth'] for p in patches), default=0)
    if max_depth == 0:
        max_depth = 1
    n_fine = 0
    n_total = 0
    for p in patches:
        bounds = p['bounds']
        H = bounds[1] - bounds[0]
        W = bounds[3] - bounds[2]
        area = H * W
        n_total += area
        if p.get('type') in ('leaf_depth', 'leaf_limit'):
            n_fine += area  # full resolution
        else:
            lf = compute_local_factor(H, W, p['depth'], max_depth, target_dim)
            n_fine += area / (lf ** 2) if lf > 0 else area
    return n_fine, n_total, n_fine / (N * N) if N > 0 else 0.0


def _patches_overlap_with_gt(patches, gt_pixel_map, N, n_blocks=None):
    """Measure how well AMR patches capture ground-truth error at pixel level.

    gt_pixel_map: (N, N) array of pixel-level error magnitudes.
    n_blocks: ignored (kept for backward compatibility).

    For each pixel, we compute a resolution weight based on which patch
    covers it:
      - leaf_depth / leaf_limit patches: weight = 1.0 (full DNS resolution)
      - coarse_leaf patches: weight = 1/local_factor^2 (reduced resolution)
      - uncovered pixels: weight = 0

    captured_fraction = sum(gt * weight) / sum(gt)
    """
    from Simulation.utils import compute_local_factor

    max_depth = max((p['depth'] for p in patches), default=1)
    if max_depth == 0:
        max_depth = 1

    # Infer target_dim from patches: typical patch at depth d covers N/target_dim^d
    # Use depth-0 patch size if available, else default to 2
    target_dim = 2
    for p in patches:
        if p['depth'] == 0:
            H = p['bounds'][1] - p['bounds'][0]
            td_candidate = N // H
            if 1 < td_candidate <= 8:
                target_dim = td_candidate
            break

    weight_map = np.zeros((N, N))
    for p in patches:
        bounds = p['bounds']
        ptype = p.get('type', 'unknown')
        depth = p['depth']
        H = bounds[1] - bounds[0]
        W = bounds[3] - bounds[2]

        if ptype in ('leaf_depth', 'leaf_limit'):
            w = 1.0
        elif ptype == 'fallback':
            continue
        else:
            lf = compute_local_factor(H, W, depth, max_depth, target_dim)
            w = 1.0 / (lf ** 2)

        y0, y1, x0, x1 = bounds[0], bounds[1], bounds[2], bounds[3]
        # Take the max weight if patches overlap
        region = weight_map[y0:y1, x0:x1]
        weight_map[y0:y1, x0:x1] = np.maximum(region, w)

    total = np.sum(gt_pixel_map)
    if total < 1e-12:
        return 1.0
    captured = np.sum(gt_pixel_map * weight_map)
    return captured / total


def run_hierarchical_comparison(sim, N, Phi_prev=None,
                                 threshold=CLASSICAL_PARAMS['threshold_amr'],
                                 threshold_qa=None, threshold_cl=None,
                                 target_dim=2, min_size=6,
                                 max_depth_override=None,
                                 noise_std=0.0, rng=None,
                                 K_opt=40, verbose=False):
    """Run both hierarchical Q-HAS and classical AMR on the same sim state.

    threshold    : default for Q-HAS (backward compat)
    threshold_qa : override threshold for Q-HAS (QAOA probabilities)
    threshold_cl : override threshold for classical (raw scores).
                   Defaults to CLASSICAL_PARAMS['threshold_amr'] for fairness.

    Returns dict with:
        'qaoa_patches': list of patches from run_adaptive_vqa
        'classical_patches': list of patches from run_adaptive_classical
        'qaoa_patches_wo_vqa': classical baseline from inside run_adaptive_vqa
        'Phi': current stress flux (for chaining)
    """
    thr_qa = threshold_qa if threshold_qa is not None else threshold
    thr_cl = threshold_cl if threshold_cl is not None else CLASSICAL_PARAMS['threshold_amr']

    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    grid = sim.grid
    nu = grid.L / 800
    HamiltMapper = PhysicalMapper(**_hamilt_mapper_kwargs(grid))

    solve_max_depth = _compute_depths(N, target_dim, min_size)
    scan_max_depth = solve_max_depth
    if max_depth_override is not None:
        scan_max_depth = min(solve_max_depth, max_depth_override)

    reps = (target_dim - 1) * 2
    args = SimpleNamespace(
        reps=reps,
        mode="simulator", backend="state_vector",
        shots=1024, method="COBYLA", opt_level=1,
        AdvAnomaliesEnable=True, K_opt=K_opt, eps=1e-2,
    )
    vqa_runtime = VQARuntime(
        backend_name="state_vector", mode="simulator",
        shots=1024, opt_level=1,
    )

    # Run Q-HAS (hierarchical VQA)
    qaoa_patches, qaoa_patches_wo_vqa, Phi = run_adaptive_vqa(
        sim, mapper, HamiltMapper, args, Phi_prev,
        beta=TRAINED_PARAMS['beta'],
        threshold_amr=thr_qa,
        target_dim=target_dim,
        max_depth=scan_max_depth,
        solve_max_depth=solve_max_depth,
        min_size=min_size,
        verbose=verbose,
        vqa_runtime=vqa_runtime,
    )

    # Run classical AMR (same BFS structure, possibly different threshold)
    classical_patches, _ = run_adaptive_classical(
        sim, mapper,
        threshold_amr=thr_cl,
        target_dim=target_dim,
        max_depth=scan_max_depth,
        solve_max_depth=solve_max_depth,
        min_size=min_size,
        verbose=verbose,
    )

    return {
        'qaoa_patches': qaoa_patches,
        'classical_patches': classical_patches,
        'qaoa_patches_wo_vqa': qaoa_patches_wo_vqa,
        'Phi': Phi,
    }


def run_single_method(sim, N, method='qaoa', Phi_prev=None,
                      threshold=CLASSICAL_PARAMS['threshold_amr'],
                      target_dim=2, min_size=6,
                      max_depth_override=None,
                      K_opt=40, verbose=False):
    """Run a single AMR method. Useful for independent threshold sweeps.

    method: 'qaoa' or 'classical'
    Returns: (patches, Phi_or_None)
    """
    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    grid = sim.grid
    nu = grid.L / 800
    HamiltMapper = PhysicalMapper(**_hamilt_mapper_kwargs(grid))

    solve_max_depth = _compute_depths(N, target_dim, min_size)
    scan_max_depth = solve_max_depth
    if max_depth_override is not None:
        scan_max_depth = min(solve_max_depth, max_depth_override)

    if method == 'qaoa':
        reps = (target_dim - 1) * 2
        args = SimpleNamespace(
            reps=reps,
            mode="simulator", backend="state_vector",
            shots=1024, method="COBYLA", opt_level=1,
            AdvAnomaliesEnable=True, K_opt=K_opt, eps=1e-2,
        )
        vqa_runtime = VQARuntime(
            backend_name="state_vector", mode="simulator",
            shots=1024, opt_level=1,
        )
        patches, _, Phi = run_adaptive_vqa(
            sim, mapper, HamiltMapper, args, Phi_prev,
            beta=TRAINED_PARAMS['beta'],
            threshold_amr=threshold,
            target_dim=target_dim,
            max_depth=scan_max_depth,
            solve_max_depth=solve_max_depth,
            min_size=min_size,
            verbose=verbose,
            vqa_runtime=vqa_runtime,
        )
        return patches, Phi
    else:
        patches, _ = run_adaptive_classical(
            sim, mapper,
            threshold_amr=threshold,
            target_dim=target_dim,
            max_depth=scan_max_depth,
            solve_max_depth=solve_max_depth,
            min_size=min_size,
            verbose=verbose,
        )
        return patches, None


def find_optimal_threshold(sim, N, method, gt_errors, Phi_prev=None,
                           target_dim=2, min_size=6, K_opt=40,
                           capture_margin=0.05, verbose=False):
    """Find the optimal threshold for a given AMR method via fine-grained sweep.

    Strategy:
      1. Coarse sweep (0.15 to 0.85, step 0.10) to find the region of interest
      2. Fine sweep (step 0.025) around the best region
      3. Selection criterion: find the "knee" — the highest threshold that
         still achieves near-maximum captured fraction.

    The objective is:
      - PRIMARY: maximize captured_fraction (within capture_margin of the best)
      - SECONDARY: among those, minimize compute_ratio

    capture_margin: acceptable loss in captured_fraction relative to the best
                    (e.g., 0.05 means we accept up to 5% less capture for
                     a significant compute saving)

    Returns: dict with 'threshold', 'captured', 'compute', 'efficiency',
             'sweep_data' (list of all evaluated points)
    """
    # ── Phase 1: Coarse sweep ──
    coarse_thrs = [t / 100 for t in range(15, 90, 10)]
    sweep_data = []

    for thr in coarse_thrs:
        patches, _ = run_single_method(
            sim, N, method=method, Phi_prev=Phi_prev,
            threshold=thr, target_dim=target_dim, min_size=min_size, K_opt=K_opt,
        )
        m = patches_to_metrics(patches, gt_errors, N, target_dim)
        sweep_data.append({
            'threshold': thr,
            'captured': m['captured_fraction'],
            'compute': m['compute_ratio'],
            'n_fine': m['n_fine'],
            'n_total': m['n_total'],
        })

    # Find the coarse region with best captured fraction
    best_cap = max(d['captured'] for d in sweep_data)
    # Find thresholds within margin of best capture
    good_coarse = [d for d in sweep_data
                   if d['captured'] >= best_cap - capture_margin]
    # The region of interest is around the highest good threshold
    # (where we still capture well but use less compute)
    if good_coarse:
        center_thr = max(d['threshold'] for d in good_coarse)
    else:
        center_thr = sweep_data[0]['threshold']

    # ── Phase 2: Fine sweep around center ──
    fine_lo = max(0.10, center_thr - 0.15)
    fine_hi = min(0.90, center_thr + 0.15)
    fine_thrs = [t / 1000 for t in range(int(fine_lo * 1000),
                                          int(fine_hi * 1000) + 1, 25)]
    # Remove already-evaluated thresholds
    already = {d['threshold'] for d in sweep_data}
    fine_thrs = [t for t in fine_thrs if t not in already]

    for thr in fine_thrs:
        patches, _ = run_single_method(
            sim, N, method=method, Phi_prev=Phi_prev,
            threshold=thr, target_dim=target_dim, min_size=min_size, K_opt=K_opt,
        )
        m = patches_to_metrics(patches, gt_errors, N, target_dim)
        sweep_data.append({
            'threshold': thr,
            'captured': m['captured_fraction'],
            'compute': m['compute_ratio'],
            'n_fine': m['n_fine'],
            'n_total': m['n_total'],
        })

    # Sort by threshold for readability
    sweep_data.sort(key=lambda d: d['threshold'])

    # ── Selection: find the knee ──
    best_cap = max(d['captured'] for d in sweep_data)

    # Candidates: all points within capture_margin of the best
    candidates = [d for d in sweep_data
                  if d['captured'] >= best_cap - capture_margin]

    if not candidates:
        # Fallback: just pick the best capture
        candidates = [d for d in sweep_data if d['captured'] == best_cap]

    # Among candidates, pick the one with lowest compute_ratio
    # (this is the "knee" — best trade-off)
    best = min(candidates, key=lambda d: d['compute'])

    if verbose:
        print(f"    [{method}] Sweep: {len(sweep_data)} points, "
              f"best_cap={best_cap:.4f}")
        print(f"    [{method}] Selected: thr={best['threshold']:.3f}, "
              f"cap={best['captured']:.4f}, comp={best['compute']:.4f}, "
              f"fine={best['n_fine']}/{best['n_total']}")

    return {
        'threshold': best['threshold'],
        'captured': best['captured'],
        'compute': best['compute'],
        'efficiency': best['captured'] / max(best['compute'], 1e-6),
        'sweep_data': sweep_data,
        'best_capture': best_cap,
    }


def patches_to_metrics(patches, gt_pixel_map, N, target_dim=2):
    """Extract comparison metrics from a patch list.

    gt_pixel_map: (N, N) pixel-level error map from ground_truth_errors().

    Returns dict with:
        'n_fine': number of fine-resolution patches
        'n_coarse': number of coarse patches
        'n_total': total patches
        'captured_fraction': fraction of GT error captured (pixel-level)
        'compute_ratio': effective pixels / N^2
    """
    n_fine = sum(1 for p in patches
                 if p.get('type') in ('leaf_depth', 'leaf_limit'))
    n_coarse = sum(1 for p in patches
                   if p.get('type') == 'coarse_leaf')
    captured = _patches_overlap_with_gt(patches, gt_pixel_map, N)
    _, _, compute_ratio = _patches_total_fine_pixels(patches, N, target_dim)

    return {
        'n_fine': n_fine,
        'n_coarse': n_coarse,
        'n_total': len(patches),
        'captured_fraction': captured,
        'compute_ratio': compute_ratio,
    }


def print_patch_summary(label, patches, gt_errors, N, target_dim=2):
    """Print a human-readable summary of patch selection."""
    m = patches_to_metrics(patches, gt_errors, N, target_dim)
    types = {}
    for p in patches:
        t = p.get('type', 'unknown')
        types[t] = types.get(t, 0) + 1
    depth_counts = {}
    for p in patches:
        d = p['depth']
        depth_counts[d] = depth_counts.get(d, 0) + 1
    print(f"  [{label}] {m['n_total']} patches "
          f"(fine={m['n_fine']}, coarse={m['n_coarse']}), "
          f"captured={m['captured_fraction']:.3f}, "
          f"compute_ratio={m['compute_ratio']:.4f}")
    print(f"    types: {types}")
    print(f"    depths: {depth_counts}")
