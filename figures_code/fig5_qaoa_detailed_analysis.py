"""Fig 5 — Detailed QAOA Analysis on Hierarchical AMR
=====================================================
Analyzes what the QAOA actually does at each level of the hierarchy:
- What probabilities does it assign to each cell?
- How do they differ from the classical score?
- What Hamiltonian coefficients are active?
- Does the QAOA correct classical false negatives/positives?

IMPORTANT: Analyzes both the FULL-GRID depth-0 call AND deeper
patches where the effective dx is small enough to trigger physical
thresholds (Re_cell, Rm_cell > critical).

Uses 256×256 grid with 2×2 VQA patches (8 qubits).
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fig_utils import (
    apply_style, COLORS, TRAINED_PARAMS, CLASSICAL_PARAMS,
    make_sim, ground_truth_errors, _compute_depths,
    filter_scenarios_dict,
    _hamilt_mapper_kwargs,
    FIG_DIR,
)
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import get_adaptive_flux
from Simulation.refinement import run_adaptive_vqa, run_adaptive_classical
from VQA.runtime import VQARuntime
from call_vqa_shell import call_vqa_shell
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

apply_style()

# ── Configuration ──
N = 256
TARGET_DIM = 2
MIN_SIZE = 6

SCENARIOS = filter_scenarios_dict({
    'init_kelvin_helmholtz':   {'label': 'Kelvin-Helmholtz', 'n_steps': 400},
    'init_harris_tearing':     {'label': 'Harris Tearing',   'n_steps': 300},
    'init_mhd_rotor':          {'label': 'MHD Rotor',        'n_steps': 300},
    'init_orszag_tang':        {'label': 'Orszag-Tang',      'n_steps': 500},
})

n_scen = len(SCENARIOS)
if n_scen == 0:
    print("Aucun scénario pour cette phase.")
    sys.exit(0)

solve_md = _compute_depths(N, TARGET_DIM, MIN_SIZE)

print("=" * 70)
print(f"Fig 5: Detailed QAOA Analysis (N={N}, VQA={TARGET_DIM}×{TARGET_DIM})")
print(f"  solve_max_depth = {solve_md}")
print("=" * 70)


def analyze_vqa_at_patch(sim, N, Phi_prev, threshold, bounds=None, target_dim=2):
    """Run a VQA call on a specific patch and analyze coefficients.

    bounds: (y0, y1, x0, x1) — if None, use full domain (depth 0).
    """
    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    grid = sim.grid
    HamiltMapper = PhysicalMapper(**_hamilt_mapper_kwargs(grid))

    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)
    full_score = AngleMapper.classical_score(physics_state)

    # Extract sub-domain if bounds specified
    if bounds is not None:
        y0, y1, x0, x1 = bounds
        local_state = {k: v[y0:y1, x0:x1] for k, v in physics_state.items()}
        local_score = full_score[y0:y1, x0:x1]
        local_phi_h = Phi['phi_horizontal'][y0:y1, x0:x1]
        local_phi_v = Phi['phi_vertical'][y0:y1, x0:x1]
        patch_size = (y1 - y0)
        # Compute effective dx for this patch level
        dx_eff = (patch_size / N) * grid.L / target_dim
        is_periodic = False
    else:
        local_state = physics_state
        local_score = full_score
        local_phi_h = Phi['phi_horizontal']
        local_phi_v = Phi['phi_vertical']
        dx_eff = None  # use grid.dx
        is_periodic = True

    # Compute Hamiltonian coefficients with proper dx
    hamilt_params = HamiltMapper.compute_coefficients(
        sim, local_score, local_state, threshold,
        advanced_anomalies_enabled=True,
        dx_override=dx_eff,
    )

    # Extract coefficient magnitudes BEFORE downsampling
    full_H_energy = _safe_energy(hamilt_params.get('H_edges', None))
    full_C_energy = _safe_energy(hamilt_params.get('C_edges', None))
    full_K_energy = _safe_energy(hamilt_params.get('K_plaquettes', None))

    # Downsample for VQA
    phi_h = local_phi_h
    phi_v = local_phi_v

    prev_h = Phi_prev['phi_horizontal'] if Phi_prev else None
    prev_v = Phi_prev['phi_vertical'] if Phi_prev else None
    if bounds is not None and prev_h is not None:
        y0, y1, x0, x1 = bounds
        prev_h = prev_h[y0:y1, x0:x1]
        prev_v = prev_v[y0:y1, x0:x1]

    type_filter = is_periodic

    if prev_h is not None:
        AveragePhiDev = 0.5 * (np.mean(np.abs(phi_h - prev_h))
                                + np.mean(np.abs(phi_v - prev_v)))
        mini_h, mini_v, mini_prev_h, mini_prev_v, mini_hp, mini_score = \
            get_adaptive_flux(
                phi_h, phi_v, prev_h, prev_v, local_score, hamilt_params,
                target_dim=target_dim, type_filter=type_filter,
            )
        mini_Phi_prev = {'phi_horizontal': mini_prev_h,
                         'phi_vertical': mini_prev_v}
    else:
        AveragePhiDev = None
        mini_h, mini_v, mini_hp, mini_score = get_adaptive_flux(
            phi_h, phi_v, None, None, local_score, hamilt_params,
            target_dim=target_dim, type_filter=type_filter,
        )
        mini_Phi_prev = None

    mini_score_clipped = np.clip(mini_score, 0.0, 1.0)
    angles = mapper.map_to_angles(
        score_h=mini_score_clipped, score_v=mini_score_clipped,
        phi_dict_prev=mini_Phi_prev,
        phi_dict={'phi_horizontal': mini_h, 'phi_vertical': mini_v},
        AveragePhiDev=AveragePhiDev, beta=1.0,
    )

    # Extract downsampled coefficient magnitudes
    ds_H_energy = _safe_energy(mini_hp.get('H_edges', None))
    ds_C_energy = _safe_energy(mini_hp.get('C_edges', None))
    ds_K_energy = _safe_energy(mini_hp.get('K_plaquettes', None))

    reps = (target_dim - 1) * 2
    args = SimpleNamespace(
        reps=reps,
        mode="simulator", backend="state_vector",
        shots=1024, method="COBYLA", opt_level=1,
        AdvAnomaliesEnable=True, K_opt=40, eps=1e-2,
    )
    vqa_runtime = VQARuntime(
        backend_name="state_vector", mode="simulator",
        shots=1024, opt_level=1,
    )

    result = call_vqa_shell(
        angles, mini_hp, False, args,
        period_bound=is_periodic, vqa_runtime=vqa_runtime,
    )

    ne = target_dim * target_dim
    if result is not None:
        probs, opt_params = result
        ph = probs[:ne].reshape(target_dim, target_dim)
        pv = probs[ne:].reshape(target_dim, target_dim)
        qaoa_score = 0.5 * (ph + pv)
    else:
        qaoa_score = mini_score_clipped.copy()
        opt_params = None

    theta_h, theta_v, psi_h, psi_v = angles
    classical_prob_h = np.sin(theta_h / 2) ** 2
    classical_prob_v = np.sin(theta_v / 2) ** 2
    # Angles may be larger than target_dim when map_to_angles keeps full resolution;
    # downsample to (target_dim, target_dim) to match VQA output shape.
    if classical_prob_h.size > target_dim * target_dim:
        ph_2d = classical_prob_h.reshape(int(np.sqrt(classical_prob_h.size)), -1)
        pv_2d = classical_prob_v.reshape(int(np.sqrt(classical_prob_v.size)), -1)
        bk = ph_2d.shape[0] // target_dim
        ph_2d = ph_2d.reshape(target_dim, bk, target_dim, bk).mean(axis=(1, 3))
        pv_2d = pv_2d.reshape(target_dim, bk, target_dim, bk).mean(axis=(1, 3))
        classical_prob = 0.5 * (ph_2d + pv_2d)
    else:
        classical_prob = 0.5 * (classical_prob_h.reshape(target_dim, target_dim)
                                + classical_prob_v.reshape(target_dim, target_dim))

    return {
        'classical_score': mini_score_clipped,
        'classical_prob': classical_prob,
        'qaoa_prob': qaoa_score,
        'correction': qaoa_score - classical_prob,
        'theta_h': theta_h, 'theta_v': theta_v,
        'psi_h': psi_h, 'psi_v': psi_v,
        'full_H_energy': full_H_energy,
        'full_C_energy': full_C_energy,
        'full_K_energy': full_K_energy,
        'ds_H_energy': ds_H_energy,
        'ds_C_energy': ds_C_energy,
        'ds_K_energy': ds_K_energy,
        'opt_params': opt_params,
    }


def _safe_energy(val):
    """Safely compute total |coefficient| energy from various data types."""
    if val is None:
        return 0.0
    if isinstance(val, (tuple, list)):
        return sum(np.sum(np.abs(a)) for a in val if isinstance(a, np.ndarray))
    if isinstance(val, dict):
        return sum(np.sum(np.abs(v)) for v in val.values() if isinstance(v, np.ndarray))
    if isinstance(val, np.ndarray):
        return np.sum(np.abs(val))
    return 0.0


def analyze_hamiltonian_by_depth(sim, N, Phi_prev, threshold, target_dim, max_depth,
                                 min_size):
    """Compute Hamiltonian coefficient magnitudes at each BFS depth.

    At depth d, the domain is divided into 4^d patches, each of size N/2^d.
    We sample the highest-GT-error patch at each depth to get representative
    coefficient values.

    Returns dict: depth -> {'H': float, 'C': float, 'K': float}
    """
    gt = ground_truth_errors(sim, N, target_dim)
    result = {}

    # Depth 0: full domain
    ana = analyze_vqa_at_patch(sim, N, Phi_prev, threshold,
                                bounds=None, target_dim=target_dim)
    result[0] = {
        'H': ana['full_H_energy'], 'C': ana['full_C_energy'],
        'K': ana['full_K_energy'], 'correction': ana['correction'],
    }
    print(f"    Depth 0: H={ana['full_H_energy']:.4f} C={ana['full_C_energy']:.4f} "
          f"K={ana['full_K_energy']:.4f}")

    # Deeper depths: subdivide and pick highest-error patch
    for d in range(1, min(max_depth, 4)):  # cap at depth 3 for speed
        patch_size = N // (2 ** d)
        if patch_size < min_size:
            break
        best_err = -1
        best_bounds = None
        n_patches = 2 ** d
        for i in range(n_patches):
            for j in range(n_patches):
                y0, y1 = i * patch_size, (i + 1) * patch_size
                x0, x1 = j * patch_size, (j + 1) * patch_size
                err = gt[y0:y1, x0:x1].sum()
                if err > best_err:
                    best_err = err
                    best_bounds = (y0, y1, x0, x1)
        try:
            ana = analyze_vqa_at_patch(sim, N, Phi_prev, threshold,
                                        bounds=best_bounds, target_dim=target_dim)
            result[d] = {
                'H': ana['full_H_energy'], 'C': ana['full_C_energy'],
                'K': ana['full_K_energy'], 'correction': ana['correction'],
            }
            print(f"    Depth {d}: H={ana['full_H_energy']:.4f} "
                  f"C={ana['full_C_energy']:.4f} K={ana['full_K_energy']:.4f}")
        except Exception as e:
            print(f"    Depth {d}: FAILED ({e})")
            result[d] = {'H': 0, 'C': 0, 'K': 0, 'correction': np.zeros((target_dim, target_dim))}

    return result


# ── Run analysis ──
all_results = {}
for scenario_init, cfg in SCENARIOS.items():
    label = cfg['label']
    print(f"\n{'─'*50}")
    print(f"Analyzing: {label}")
    print(f"{'─'*50}")

    sim, Phi_prev = make_sim(N, scenario_init, cfg['n_steps'])
    gt = ground_truth_errors(sim, N, TARGET_DIM)

    print(f"  Computing Hamiltonian at multiple depths...")
    depth_hamilt = analyze_hamiltonian_by_depth(
        sim, N, Phi_prev, TRAINED_PARAMS['threshold_amr'],
        TARGET_DIM, solve_md, MIN_SIZE)

    # Also get depth-0 and depth-1 for backward compat
    analysis_d0 = analyze_vqa_at_patch(sim, N, Phi_prev,
                                        TRAINED_PARAMS['threshold_amr'],
                                        bounds=None, target_dim=TARGET_DIM)
    # Find best depth-1 quadrant
    scores_quadrant = []
    quadrants = [(0, N//2, 0, N//2), (0, N//2, N//2, N),
                 (N//2, N, 0, N//2), (N//2, N, N//2, N)]
    for qb in quadrants:
        y0, y1, x0, x1 = qb
        scores_quadrant.append(np.max(gt[y0:y1, x0:x1]))
    best_quad = quadrants[np.argmax(scores_quadrant)]
    analysis_d1 = analyze_vqa_at_patch(sim, N, Phi_prev,
                                        TRAINED_PARAMS['threshold_amr'],
                                        bounds=best_quad, target_dim=TARGET_DIM)

    # Full hierarchical comparison
    from fig_utils import run_hierarchical_comparison, patches_to_metrics, print_patch_summary
    best_qa_thr = TRAINED_PARAMS['threshold_amr']
    best_cl_thr = CLASSICAL_PARAMS['threshold_amr']
    comp = run_hierarchical_comparison(
        sim, N, Phi_prev=Phi_prev,
        threshold_qa=best_qa_thr, threshold_cl=best_cl_thr,
        target_dim=TARGET_DIM, min_size=MIN_SIZE, K_opt=40, verbose=True,
    )
    qaoa_m = patches_to_metrics(comp['qaoa_patches'], gt, N, TARGET_DIM)
    cl_m = patches_to_metrics(comp['classical_patches'], gt, N, TARGET_DIM)

    all_results[label] = {
        'analysis_d0': analysis_d0,
        'analysis_d1': analysis_d1,
        'depth_hamilt': depth_hamilt,
        'gt': gt,
        'qaoa_metrics': qaoa_m,
        'cl_metrics': cl_m,
        'best_quad': best_quad,
    }


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS — per-quadrant GT error share
# ═══════════════════════════════════════════════════════════════════════

def _gt_error_share(gt, target_dim):
    """Compute per-quadrant error share from the full-resolution GT map.

    For each cell (i,j) in the target_dim × target_dim VQA grid,
    returns sum(GT in quadrant) / sum(GT over all quadrants).
    This is the physically meaningful ground truth at the VQA decision
    resolution: "what fraction of total error lives here?"
    """
    bk = gt.shape[0] // target_dim
    shares = np.zeros((target_dim, target_dim))
    total = gt.sum() + 1e-12
    for i in range(target_dim):
        for j in range(target_dim):
            shares[i, j] = gt[i*bk:(i+1)*bk, j*bk:(j+1)*bk].sum()
    return shares / total


def _gt_quadrant_above_threshold(gt, target_dim, threshold):
    """Binary: does each quadrant's mean error exceed the threshold?

    Used for TP/FP/FN/TN classification of QAOA vs classical decisions.
    """
    bk = gt.shape[0] // target_dim
    above = np.zeros((target_dim, target_dim), dtype=bool)
    for i in range(target_dim):
        for j in range(target_dim):
            above[i, j] = gt[i*bk:(i+1)*bk, j*bk:(j+1)*bk].mean() > threshold
    return above


# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

SHORT = {'Kelvin-Helmholtz': 'KH', 'Harris Tearing': 'Tearing',
         'MHD Rotor': 'Rotor', 'Orszag-Tang': 'OT'}

# Depth colors (deeper = darker)
DEPTH_COLORS = ['#FFB3B3', '#E57373', '#C62828', '#4A0000']

n_scenarios = len(all_results)
fig, axes = plt.subplots(n_scenarios, 3, figsize=(14, 3.4 * n_scenarios))
if n_scenarios == 1:
    axes = axes[np.newaxis, :]

for row, (label, data) in enumerate(all_results.items()):
    ana_d0 = data['analysis_d0']
    ana_d1 = data['analysis_d1']
    depth_hamilt = data['depth_hamilt']
    gt = data['gt']
    sn = SHORT.get(label, label)

    # ── Col 0: GT error map ──
    ax = axes[row, 0]
    im = ax.imshow(gt, cmap='hot', origin='lower', aspect='equal')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Error', fontsize=8)
    ax.set_title(f'{sn}: GT Error Map', fontsize=10, pad=10)
    ax.tick_params(labelsize=8)

    # ── Col 1: QAOA correction δ at depth 0 and depth 1 ──
    ax = axes[row, 1]
    cell_labels = [f'({i},{j})' for i in range(TARGET_DIM) for j in range(TARGET_DIM)]
    correction_d0 = ana_d0['correction'].ravel()
    correction_d1 = ana_d1['correction'].ravel()
    x2 = np.arange(TARGET_DIM * TARGET_DIM)
    w2 = 0.3
    ax.bar(x2 - w2/2, correction_d0, w2, label='Depth 0',
           color=COLORS['qaoa'], alpha=0.7)
    ax.bar(x2 + w2/2, correction_d1, w2, label='Depth 1',
           color=COLORS['smoothed'], alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, lw=0.8)
    ax.set_xticks(x2)
    ax.set_xticklabels(cell_labels, fontsize=8)
    ax.set_ylabel(r'$\delta$ (QAOA $-$ Classical)', fontsize=9)
    ax.set_title(f'{sn}: QAOA Correction $\\delta$', fontsize=10, pad=10)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

    # ── Col 2: Depth-grouped Hamiltonian energy (log-scale) ──
    # Grouped bars: one group per Hamiltonian term (H, C, K), one bar
    # per depth.  Area-weighted (depth d × 4^(max_d−d)) so shallower
    # depths reflect the larger region they cover.
    # Log-scale y-axis keeps all bars readable regardless of magnitude.
    ax = axes[row, 2]
    terms = ['H(Z)', 'C(ZZ)', 'K(ZZZZ)']
    keys = ['H', 'C', 'K']
    depths_available = sorted(depth_hamilt.keys())
    max_d = max(depths_available) if depths_available else 0
    n_depths = len(depths_available)

    x_e = np.arange(len(terms))
    total_w = 0.7
    bar_w = total_w / max(n_depths, 1)

    for di, d in enumerate(depths_available):
        weight = 4 ** (max_d - d) if max_d > 0 else 1.0
        vals = [max(depth_hamilt[d][k] * weight, 1e-6) for k in keys]
        offset = (di - (n_depths - 1) / 2) * bar_w
        color = DEPTH_COLORS[min(d, len(DEPTH_COLORS) - 1)]
        bars = ax.bar(x_e + offset, vals, bar_w * 0.88, label=f'd={d}',
                      color=color, alpha=0.85, edgecolor='white',
                      linewidth=0.3)
        for b, v in zip(bars, vals):
            if v > 1e-5:
                ax.text(b.get_x() + b.get_width() / 2,
                        v * 1.15, f'{v:.0f}', ha='center', va='bottom',
                        fontsize=6, rotation=45)

    ax.set_yscale('log')
    ax.set_xticks(x_e)
    ax.set_xticklabels(terms, fontsize=9)
    ax.set_ylabel('Weighted |coeff|', fontsize=9)
    ax.set_title(f'{sn}: Hamiltonian by Depth', fontsize=10, pad=10)
    ax.legend(fontsize=6, loc='lower right', ncol=2,
              bbox_to_anchor=(0.98, 0.15))
    ax.tick_params(labelsize=8)

fig.suptitle('QAOA Detailed Analysis', fontsize=11, fontweight='bold')
fig.subplots_adjust(top=0.92, hspace=0.45, wspace=0.60)
out = os.path.join(FIG_DIR, 'fig5_qaoa_detailed_analysis.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"\nSaved -> {out}")

# ── Print detailed summary ──
print("\n" + "=" * 70)
print("DETAILED SUMMARY — What does the QAOA actually change?")
print("=" * 70)
for label, data in all_results.items():
    ana_d0 = data['analysis_d0']
    ana_d1 = data['analysis_d1']
    gt = data['gt']
    gt_share = _gt_error_share(gt, TARGET_DIM)

    print(f"\n{label}:")
    print(f"  GT error: min={gt.min():.4f} max={gt.max():.4f}")
    print(f"  Per-quadrant GT error share: {gt_share.ravel()}")
    print(f"  Depth 0 — Classical prob: {ana_d0['classical_prob'].ravel()}")
    print(f"  Depth 0 — QAOA prob:      {ana_d0['qaoa_prob'].ravel()}")
    print(f"  Depth 0 — Max |delta|: {np.max(np.abs(ana_d0['correction'])):.4f}")
    print(f"  Depth 0 — Hamiltonian: H={ana_d0['full_H_energy']:.2f}, "
          f"C={ana_d0['full_C_energy']:.2f}, K={ana_d0['full_K_energy']:.2f}")
    print(f"  Depth 1 — Max |delta|: {np.max(np.abs(ana_d1['correction'])):.4f}")
    print(f"  Depth 1 — Hamiltonian: H={ana_d1['full_H_energy']:.2f}, "
          f"C={ana_d1['full_C_energy']:.2f}, K={ana_d1['full_K_energy']:.2f}")
    ratio_d0 = (ana_d0['full_C_energy'] + ana_d0['full_K_energy']) / \
               (ana_d0['full_H_energy'] + ana_d0['full_C_energy'] + ana_d0['full_K_energy'] + 1e-12)
    ratio_d1 = (ana_d1['full_C_energy'] + ana_d1['full_K_energy']) / \
               (ana_d1['full_H_energy'] + ana_d1['full_C_energy'] + ana_d1['full_K_energy'] + 1e-12)
    print(f"  Multi-body ratio (C+K)/(H+C+K): d0={ratio_d0:.3f}, d1={ratio_d1:.3f}")

    # Rank agreement: does QAOA's top quadrant match the GT's top quadrant?
    qa_top = np.argmax(ana_d0['qaoa_prob'].ravel())
    cl_top = np.argmax(ana_d0['classical_prob'].ravel())
    gt_top = np.argmax(gt_share.ravel())
    print(f"  Top quadrant — GT: {gt_top}, QAOA: {qa_top}, Classical: {cl_top}"
          f"  {'(QAOA matches GT)' if qa_top == gt_top else '(QAOA misses)'}"
          f"  {'(CL matches GT)' if cl_top == gt_top else '(CL misses)'}")

    # TP/FP/FN analysis at threshold
    thr = TRAINED_PARAMS['threshold_amr']
    gt_above = _gt_quadrant_above_threshold(gt, TARGET_DIM, thr)
    qa_refine = ana_d0['qaoa_prob'] > 0.5
    cl_refine = ana_d0['classical_prob'] > 0.5
    n_cells = TARGET_DIM * TARGET_DIM
    qa_tp = int(np.sum(qa_refine & gt_above))
    qa_fp = int(np.sum(qa_refine & ~gt_above))
    qa_fn = int(np.sum(~qa_refine & gt_above))
    cl_tp = int(np.sum(cl_refine & gt_above))
    cl_fp = int(np.sum(cl_refine & ~gt_above))
    cl_fn = int(np.sum(~cl_refine & gt_above))
    print(f"  Decision quality (P>0.5 = refine):")
    print(f"    QAOA:      TP={qa_tp} FP={qa_fp} FN={qa_fn} / {n_cells} cells")
    print(f"    Classical: TP={cl_tp} FP={cl_fp} FN={cl_fn} / {n_cells} cells")
    print(f"  Hierarchical: Q-HAS={data['qaoa_metrics']['captured_fraction']:.3f} "
          f"vs CL={data['cl_metrics']['captured_fraction']:.3f}")
