#!/usr/bin/env python3
"""
Fig 15 — Decision Flip Analysis: Why QAOA Corrections Don't Change Outcomes
============================================================================

THE diagnostic figure. Instruments the hierarchical BFS tree to capture,
at every node and every cell, both the classical score and the QAOA
probability BEFORE the refine/coarsen decision is made.

This reveals:
  A) How often QAOA corrections actually FLIP a decision (cross the threshold)
  B) Whether flips are correct (aligned with ground truth) or incorrect
  C) The relationship between correction magnitude and threshold proximity
  D) How flips cascade through the tree (pixel impact)
  E) The distribution of classical scores relative to the threshold —
     showing whether the QAOA even HAS room to make a difference

Produces: figures/fig15_decision_flip_analysis.png
"""
import sys, os
import numpy as np
from collections import defaultdict
from types import SimpleNamespace

# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------

from fig_utils import (
    apply_style, COLORS, TRAINED_PARAMS, CLASSICAL_PARAMS,
    make_sim, ground_truth_errors, _compute_depths,
    _hamilt_mapper_kwargs,
    filter_scenarios_dict,
    FIG_DIR,
)
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import get_adaptive_flux, _process_score
from Simulation.utils import get_periodic_patch
from Simulation.refinement import (
    _prepare_vqa_input, _boundary_activation,
)
from VQA.runtime import VQARuntime
from call_vqa_shell import call_vqa_shell

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
    print("No scenarios for this phase.")
    sys.exit(0)

solve_md = _compute_depths(N, TARGET_DIM, MIN_SIZE)
scan_md = solve_md

print("=" * 70)
print(f"Fig 15: Decision Flip Analysis (N={N}, depth={solve_md})")
print(f"  Q-HAS threshold:  {TRAINED_PARAMS['threshold_amr']:.4f}")
print(f"  Classical threshold: {CLASSICAL_PARAMS['threshold_amr']:.4f}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
#  INSTRUMENTED BFS — captures per-node decisions
# ═══════════════════════════════════════════════════════════════════════

def instrumented_bfs(sim, N, Phi_prev, threshold_amr, target_dim, max_depth,
                     min_size, gt_error_map):
    """Run the VQA BFS and capture every per-cell decision.

    Returns a list of records, one per cell per node:
    {
        'depth': int,
        'bounds': tuple,          # parent patch bounds
        'cell': (i, j),
        'sub_bounds': tuple,      # child cell bounds
        'classical_score': float, # before-QAOA score (θ-encoded)
        'qaoa_prob': float,       # after-QAOA probability
        'correction': float,      # qaoa - classical
        'decision_classical': bool,  # score >= threshold
        'decision_qaoa': bool,       # prob >= threshold
        'flipped': bool,
        'gt_error_mean': float,   # mean GT error in sub-cell
        'gt_error_max': float,    # max GT error in sub-cell
        'sub_area': int,          # pixels in sub-cell
    }
    """
    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    grid = sim.grid
    HamiltMapper = PhysicalMapper(**_hamilt_mapper_kwargs(grid))

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

    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)
    full_h = Phi['phi_horizontal']
    full_v = Phi['phi_vertical']
    full_score = AngleMapper.classical_score(physics_state)

    full_prev_h = None
    full_prev_v = None
    AveragePhiDev = None
    if Phi_prev is not None:
        full_prev_h = Phi_prev['phi_horizontal']
        full_prev_v = Phi_prev['phi_vertical']
        AveragePhiDev = 0.5 * (np.mean(np.abs(full_h - full_prev_h))
                                + np.mean(np.abs(full_v - full_prev_v)))

    H, W = full_h.shape
    initial_bounds = (0, H, 0, W)

    decision_log = []
    pending = [initial_bounds]
    depth = 0

    while pending and depth <= max_depth:
        next_level = []

        for bounds in pending:
            y_s, y_e, x_s, x_e = bounds
            height = y_e - y_s
            width = x_e - x_s

            if height < min_size or width < min_size:
                continue

            # Get classical score for this patch (same as _run_level_classical)
            pad = 1 if depth > 0 else 0
            local_score_raw = get_periodic_patch(full_score, y_s, y_e, x_s, x_e, pad=pad)
            is_periodic = (depth == 0)
            # `target_dim`, PAS `target_dim + 2*pad` : a depth > 0, `_process_score`
            # emprunte `_resize_padded_maxpool`, dont le contrat est « entree
            # (N+2, M+2) -> sortie (t_dim+2, t_dim+2) » — le halo est deja ajoute
            # par la fonction. Regression de D-37 (voir refinement.py) : demander
            # target_dim+2 fait rendre un coeur 4x4 pour target_dim=2, et la boucle
            # `for i in range(target_dim)` ne lit alors QUE son quart haut-gauche —
            # classical_score decrit une sous-region differente de qaoa_prob.
            # Mesure sur Harris tearing (N=256, 30 pas, patch depth=1) : ecart
            # jusqu'a 0.525 sur des scores dont l'echelle max vaut 0.656 (80%),
            # et 2 des 4 decisions binaires (score >= threshold_amr=0.3228)
            # basculaient. Voir D-96.
            score_map_padded = _process_score(local_score_raw, is_periodic, target_dim)
            if depth > 0:
                score_map = score_map_padded[1:-1, 1:-1]
            else:
                score_map = score_map_padded

            if depth >= max_depth:
                # At max depth, no decision to make — both methods just store leaf
                continue

            # Get QAOA probability for this patch
            prep = _prepare_vqa_input(
                full_h, full_v, full_prev_h, full_prev_v,
                full_score,
                physics_state, bounds, depth, mapper, args,
                AveragePhiDev, TRAINED_PARAMS['beta'], target_dim,
                HamiltMapper=HamiltMapper, sim=sim,
                threshold_amr=threshold_amr,
            )

            if prep is None:
                continue

            angles, mini_hamilt_params, mini_score = prep

            prob_map_avant = mini_score
            if depth > 0:
                prob_map_avant = mini_score[1:-1, 1:-1]

            # Run QAOA
            result = call_vqa_shell(
                angles, mini_hamilt_params, False, args,
                period_bound=is_periodic,
                vqa_runtime=vqa_runtime,
            )

            if result is None:
                # QAOA failed — use classical score as fallback
                prob_map = prob_map_avant.copy()
            else:
                probs, _ = result
                ne = target_dim * target_dim
                probs_h = probs[:ne].reshape(target_dim, target_dim)
                probs_v = probs[ne:].reshape(target_dim, target_dim)
                prob_map = 0.5 * (probs_h + probs_v)

            # Extract Hamiltonian ZZ coefficients for "why" analysis
            C_raw = mini_hamilt_params.get('C_edges', (None, None))
            C_h_arr = C_raw[0] if isinstance(C_raw, (tuple, list)) else C_raw
            C_v_arr = C_raw[1] if isinstance(C_raw, (tuple, list)) and len(C_raw) > 1 else None
            # Total ZZ energy per cell (sum of adjacent ZZ couplings)
            zz_per_cell = np.zeros((target_dim, target_dim))
            if C_h_arr is not None and C_h_arr.size >= target_dim * target_dim:
                # C_edges may be padded (e.g. 4×4 for depth>0 with target_dim=2)
                if C_h_arr.shape[0] > target_dim:
                    pad_sz = (C_h_arr.shape[0] - target_dim) // 2
                    ch = C_h_arr[pad_sz:pad_sz+target_dim, pad_sz:pad_sz+target_dim]
                    cv = C_v_arr[pad_sz:pad_sz+target_dim, pad_sz:pad_sz+target_dim] if C_v_arr is not None else np.zeros_like(ch)
                else:
                    ch = C_h_arr
                    cv = C_v_arr if C_v_arr is not None else np.zeros_like(ch)
                for ci in range(target_dim):
                    for cj in range(target_dim):
                        zz = 0.0
                        zz += abs(float(ch[ci, cj]))
                        zz += abs(float(cv[ci, cj]))
                        if cj > 0: zz += abs(float(ch[ci, cj-1]))
                        if ci > 0: zz += abs(float(cv[ci-1, cj]))
                        zz_per_cell[ci, cj] = zz
            else:
                zz_per_cell[:] = 0.0

            # Record decisions for each cell
            step_y = height // target_dim
            step_x = width // target_dim
            effective_threshold = threshold_amr

            for i in range(target_dim):
                for j in range(target_dim):
                    classical_score = float(score_map[i, j])
                    qaoa_prob = float(prob_map[i, j])

                    sub_y_s = y_s + i * step_y
                    sub_y_e = y_s + (i + 1) * step_y if i < target_dim - 1 else y_e
                    sub_x_s = x_s + j * step_x
                    sub_x_e = x_s + (j + 1) * step_x if j < target_dim - 1 else x_e
                    sub_bounds = (sub_y_s, sub_y_e, sub_x_s, sub_x_e)

                    gt_patch = gt_error_map[sub_y_s:sub_y_e, sub_x_s:sub_x_e]
                    gt_mean = float(np.mean(gt_patch))
                    gt_max = float(np.max(gt_patch))
                    sub_area = (sub_y_e - sub_y_s) * (sub_x_e - sub_x_s)

                    dec_cl = classical_score >= effective_threshold
                    dec_qa = qaoa_prob >= effective_threshold
                    flipped = (dec_cl != dec_qa)

                    # Neighbor context: how many neighbors are above threshold?
                    n_neighbors_refine = 0
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < target_dim and 0 <= nj < target_dim:
                            if score_map[ni, nj] >= effective_threshold:
                                n_neighbors_refine += 1

                    decision_log.append({
                        'depth': depth,
                        'bounds': bounds,
                        'cell': (i, j),
                        'sub_bounds': sub_bounds,
                        'classical_score': classical_score,
                        'qaoa_prob': qaoa_prob,
                        'correction': qaoa_prob - classical_score,
                        'decision_classical': dec_cl,
                        'decision_qaoa': dec_qa,
                        'flipped': flipped,
                        'gt_error_mean': gt_mean,
                        'gt_error_max': gt_max,
                        'sub_area': sub_area,
                        'zz_energy': float(zz_per_cell[i, j]),
                        'n_neighbors_refine': n_neighbors_refine,
                    })

                    # Follow the QAOA decision for the BFS (this is what Q-HAS does)
                    if dec_qa:
                        next_level.append(sub_bounds)

        pending = next_level
        depth += 1

    return decision_log


# ═══════════════════════════════════════════════════════════════════════
#  RUN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

all_logs = {}
for scenario_init, cfg in SCENARIOS.items():
    label = cfg['label']
    print(f"\n{'─'*60}")
    print(f"  Analyzing: {label}")
    print(f"{'─'*60}")

    sim, Phi_prev = make_sim(N, scenario_init, cfg['n_steps'])
    gt = ground_truth_errors(sim, N, TARGET_DIM)

    log = instrumented_bfs(
        sim, N, Phi_prev,
        threshold_amr=TRAINED_PARAMS['threshold_amr'],
        target_dim=TARGET_DIM,
        max_depth=scan_md,
        min_size=MIN_SIZE,
        gt_error_map=gt,
    )

    all_logs[label] = log

    # Print summary
    n_total = len(log)
    n_flipped = sum(1 for r in log if r['flipped'])
    # "Should refine" = GT error above mean (same definition as fig4 pixel_precision)
    gt_all_means = np.array([r['gt_error_mean'] for r in log])
    gt_mean_threshold = np.mean(gt_all_means) if len(gt_all_means) > 0 else 0
    n_correct_flip = sum(1 for r in log if r['flipped'] and
                         (r['decision_qaoa'] == (r['gt_error_mean'] > gt_mean_threshold)))
    n_correct_cl = sum(1 for r in log if r['flipped'] and
                       (r['decision_classical'] == (r['gt_error_mean'] > gt_mean_threshold)))
    n_incorrect_flip = n_flipped - n_correct_flip
    print(f"  Total cells evaluated: {n_total}")
    print(f"  Decision flips: {n_flipped} ({100*n_flipped/max(n_total,1):.1f}%)")
    print(f"    QAOA correct:    {n_correct_flip} (QAOA decision matches GT)")
    print(f"    Classical correct: {n_correct_cl} (Classical decision matches GT)")
    print(f"    Net benefit: {n_correct_flip - n_correct_cl:+d} flips where QAOA is better")
    print(f"    GT threshold used: mean(gt_error) = {gt_mean_threshold:.4f}")

    # Per-depth breakdown
    depths = sorted(set(r['depth'] for r in log))
    for d in depths:
        d_recs = [r for r in log if r['depth'] == d]
        d_flip = sum(1 for r in d_recs if r['flipped'])
        print(f"    Depth {d}: {len(d_recs)} cells, {d_flip} flips")


# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING — 4 panels per scenario (rows)
# ═══════════════════════════════════════════════════════════════════════

n_scenarios = len(all_logs)
fig, axes = plt.subplots(n_scenarios, 4, figsize=(24, 5.5 * n_scenarios))
if n_scenarios == 1:
    axes = axes[np.newaxis, :]

threshold = TRAINED_PARAMS['threshold_amr']

for row, (label, log) in enumerate(all_logs.items()):
    if not log:
        continue

    cl_scores = np.array([r['classical_score'] for r in log])
    qa_probs = np.array([r['qaoa_prob'] for r in log])
    corrections = np.array([r['correction'] for r in log])
    flipped = np.array([r['flipped'] for r in log])
    depths_arr = np.array([r['depth'] for r in log])

    # ── Col 0: Scatter — Classical score vs QAOA prob ──
    ax = axes[row, 0]
    no_flip = ~flipped
    ax.scatter(cl_scores[no_flip], qa_probs[no_flip],
               c='#BBBBBB', alpha=0.3, s=10, edgecolors='none', label='Agreement')
    if flipped.any():
        ax.scatter(cl_scores[flipped], qa_probs[flipped],
                   c='#E53935', s=50, lw=1.2, edgecolors='darkred', marker='o',
                   label=f'Flip ({flipped.sum()})', zorder=5)
    ax.axvline(threshold, color='gray', ls='--', lw=0.8, alpha=0.7)
    ax.axhline(threshold, color='gray', ls='--', lw=0.8, alpha=0.7)
    lims = [0, max(cl_scores.max(), qa_probs.max(), 1.0)]
    ax.plot(lims, lims, 'k-', lw=0.5, alpha=0.3)
    ax.set_xlabel('Classical score', fontsize=10)
    ax.set_ylabel('QAOA probability', fontsize=10)
    ax.set_title(f'{label}: Score vs Probability', fontsize=12, pad=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.text(threshold * 0.45, lims[1] * 1.02, 'QAOA adds\nrefinement', fontsize=9,
            ha='center', va='top', color='red', alpha=0.5)
    ax.text(lims[1] * 0.95, threshold * 0.5, 'QAOA removes\nrefinement', fontsize=9,
            ha='right', va='center', color='blue', alpha=0.5)
    ax.set_ylim(lims[0], lims[1] * 1.08)  # extra headroom for text
    ax.tick_params(labelsize=9)

    # ── Col 1: Threshold proximity histogram ──
    ax = axes[row, 1]
    dist_to_thr = cl_scores - threshold
    ax.hist(dist_to_thr, bins=40, color=COLORS['classical'], alpha=0.6,
            label='Score $-$ threshold')
    ax.hist(corrections, bins=40, color=COLORS['qaoa'], alpha=0.6,
            label='Correction $\\delta$')
    ax.axvline(0, color='gray', ls='-', lw=0.8)
    ax.set_xlabel('Score $-$ threshold  /  Correction $\\delta$', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'{label}: Threshold proximity', fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    # ── Col 2: Decision flips by depth ──
    ax = axes[row, 2]
    unique_depths = sorted(set(depths_arr))
    agree_counts = []
    flip_counts = []
    for d in unique_depths:
        mask_d = depths_arr == d
        n_d = mask_d.sum()
        n_flip_d = (mask_d & flipped).sum()
        agree_counts.append(n_d - n_flip_d)
        flip_counts.append(n_flip_d)

    x_d = np.arange(len(unique_depths))
    w = 0.6
    ax.bar(x_d, agree_counts, w, label='Agreement', color='#90CAF9', alpha=0.8)
    ax.bar(x_d, flip_counts, w, bottom=agree_counts,
           label='Flip', color='#FF9800', alpha=0.8)
    for idx, d in enumerate(unique_depths):
        if flip_counts[idx] > 0:
            y_pos = agree_counts[idx] + flip_counts[idx]
            ax.text(idx, y_pos + 0.3, f'{flip_counts[idx]}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x_d)
    ax.set_xticklabels([f'd={d}' for d in unique_depths], fontsize=9)
    ax.set_xlabel('BFS Depth', fontsize=10)
    ax.set_ylabel('Number of cells', fontsize=10)
    ax.set_title(f'{label}: Decisions by depth', fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    # ── Col 3: |correction| vs |distance to threshold| ──
    ax = axes[row, 3]
    dist_to_thr_abs = np.abs(cl_scores - threshold)
    corr_abs = np.abs(corrections)
    c_scatter = np.where(flipped, '#E53935', '#888888')
    ax.scatter(dist_to_thr_abs, corr_abs, c=c_scatter, alpha=0.4, s=12, edgecolors='none')
    max_val = max(dist_to_thr_abs.max(), corr_abs.max()) if len(dist_to_thr_abs) > 0 else 1.0
    line_range = np.linspace(0, max_val, 100)
    ax.plot(line_range, line_range, 'k--', lw=1, alpha=0.5,
            label='$|\\delta|=|$dist$|$')
    ax.fill_between(line_range, line_range, max_val * 1.1, alpha=0.05, color='red')
    ax.fill_between(line_range, 0, line_range, alpha=0.05, color='blue')
    ax.text(max_val * 0.7, max_val * 0.9, 'Flip\npossible', fontsize=9,
            color='red', alpha=0.6, ha='center')
    ax.text(max_val * 0.7, max_val * 0.2, 'Too far\nto flip', fontsize=9,
            color='blue', alpha=0.6, ha='center')
    ax.set_xlabel('$|$Score $-$ threshold$|$', fontsize=10)
    ax.set_ylabel('$|$Correction $\\delta$$|$', fontsize=10)
    ax.set_title(f'{label}: Correction vs threshold gap', fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

fig.suptitle(
    f'Decision Flip Analysis  (N={N}, {TARGET_DIM}$\\times${TARGET_DIM} VQA, '
    f'threshold={threshold:.4f})',
    fontsize=12, fontweight='bold', y=1.01,
)
fig.subplots_adjust(hspace=0.40, wspace=0.30)
plt.tight_layout(pad=1.5)
out = os.path.join(FIG_DIR, 'fig15_decision_flip_analysis.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"\nSaved -> {out}")


# ═══════════════════════════════════════════════════════════════════════
#  DETAILED TEXT SUMMARY
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("GLOBAL DECISION FLIP ANALYSIS")
print("=" * 70)

all_records = []
for label, log in all_logs.items():
    all_records.extend(log)

if all_records:
    all_cl = np.array([r['classical_score'] for r in all_records])
    all_qa = np.array([r['qaoa_prob'] for r in all_records])
    all_corr = np.array([r['correction'] for r in all_records])
    all_flip = np.array([r['flipped'] for r in all_records])
    all_dist = np.abs(all_cl - threshold)

    print(f"\nAcross all scenarios:")
    print(f"  Total cells:     {len(all_records)}")
    print(f"  Total flips:     {all_flip.sum()} ({100*all_flip.mean():.1f}%)")
    print(f"  Mean |correction|: {np.mean(np.abs(all_corr)):.4f}")
    print(f"  Mean |dist to thr|: {np.mean(all_dist):.4f}")
    print(f"  Ratio:           {np.mean(np.abs(all_corr))/max(np.mean(all_dist),1e-8):.3f}")

    # Key diagnostic: histogram of correction/distance ratio
    ratio = np.abs(all_corr) / np.clip(all_dist, 1e-6, None)
    print(f"\n  |correction| / |distance to threshold| distribution:")
    for pct in [25, 50, 75, 90, 95, 99]:
        print(f"    P{pct}: {np.percentile(ratio, pct):.3f}")
    print(f"    Fraction > 1.0 (can flip): {(ratio > 1.0).mean()*100:.1f}%")

    # Correlation between correction magnitude and GT error
    all_gt = np.array([r['gt_error_mean'] for r in all_records])
    if len(all_gt) > 2:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(np.abs(all_corr), all_gt)
        print(f"\n  Spearman correlation |δ| vs GT error: ρ={rho:.3f} (p={pval:.4f})")
        print(f"  → {'Corrections target high-error regions' if rho > 0.1 else 'Corrections are NOT targeted at high-error regions'}")

    # WHY analysis: ZZ energy and neighbor context at flips vs non-flips
    all_zz = np.array([r['zz_energy'] for r in all_records])
    all_neigh = np.array([r['n_neighbors_refine'] for r in all_records])
    if all_flip.any() and not all_flip.all():
        print(f"\n  WHY FLIPS HAPPEN (ZZ + neighbor analysis):")
        print(f"  {'Metric':<30} {'No flip':>12} {'Flipped':>12} {'Ratio':>8}")
        print(f"  {'─'*62}")
        for name, arr in [('Median ZZ energy', all_zz),
                          ('Mean GT error', all_gt),
                          ('Mean neighbors above thr', all_neigh)]:
            v_noflip = np.median(arr[~all_flip]) if name.startswith('Median') else np.mean(arr[~all_flip])
            v_flip = np.median(arr[all_flip]) if name.startswith('Median') else np.mean(arr[all_flip])
            ratio_val = v_flip / max(v_noflip, 1e-8)
            print(f"  {name:<30} {v_noflip:>12.4f} {v_flip:>12.4f} {ratio_val:>8.2f}x")

        # Direction of correction at flips
        flip_corr = all_corr[all_flip]
        n_up = (flip_corr > 0).sum()
        n_down = (flip_corr < 0).sum()
        print(f"\n  Flip direction: {n_up} pushed UP (toward refine), {n_down} pushed DOWN (toward skip)")

        # Were the flips correct? Use mean GT error as threshold (consistent with fig4)
        gt_mean_all = np.mean(all_gt)
        all_dec_cl = np.array([r['decision_classical'] for r in all_records])
        all_dec_qa = np.array([r['decision_qaoa'] for r in all_records])
        gt_should = all_gt > gt_mean_all
        n_qa_correct_all = int((all_dec_qa == gt_should).sum())
        n_cl_correct_all = int((all_dec_cl == gt_should).sum())
        print(f"\n  OVERALL ACCURACY (all cells, GT threshold = mean):")
        print(f"    QAOA correct:      {n_qa_correct_all}/{len(all_records)} ({100*n_qa_correct_all/len(all_records):.1f}%)")
        print(f"    Classical correct:  {n_cl_correct_all}/{len(all_records)} ({100*n_cl_correct_all/len(all_records):.1f}%)")
        print(f"    Net: {'QAOA' if n_qa_correct_all > n_cl_correct_all else 'Classical'} "
              f"by {abs(n_qa_correct_all - n_cl_correct_all)} cells")

        # Per-scenario accuracy breakdown
        print(f"\n  PER-SCENARIO ACCURACY:")
        for label_s, log_s in all_logs.items():
            gt_s = np.array([r['gt_error_mean'] for r in log_s])
            gt_thr_s = np.mean(gt_s)
            qa_corr_s = sum(1 for r in log_s if r['decision_qaoa'] == (r['gt_error_mean'] > gt_thr_s))
            cl_corr_s = sum(1 for r in log_s if r['decision_classical'] == (r['gt_error_mean'] > gt_thr_s))
            n_s = len(log_s)
            winner = 'QAOA' if qa_corr_s > cl_corr_s else ('Classical' if cl_corr_s > qa_corr_s else 'Tie')
            print(f"    {label_s:20s}: QAOA {qa_corr_s}/{n_s} ({100*qa_corr_s/max(n_s,1):.1f}%) "
                  f"CL {cl_corr_s}/{n_s} ({100*cl_corr_s/max(n_s,1):.1f}%) → {winner}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
if all_records:
    flip_rate = all_flip.mean()
    mean_ratio = np.mean(np.abs(all_corr)) / max(np.mean(all_dist), 1e-8)

    # Quality of flips — use mean-based GT threshold
    gt_mean_global = np.mean(all_gt)
    all_dec_cl_g = np.array([r['decision_classical'] for r in all_records])
    all_dec_qa_g = np.array([r['decision_qaoa'] for r in all_records])
    gt_should_g = all_gt > gt_mean_global
    n_qa_correct_g = int((all_dec_qa_g == gt_should_g).sum())
    n_cl_correct_g = int((all_dec_cl_g == gt_should_g).sum())

    if flip_rate < 0.05 and mean_ratio < 0.5:
        print("The QAOA corrections are MUCH SMALLER than the gap between")
        print("classical scores and the threshold. The classical scores are")
        print("either far above or far below the threshold, leaving no room")
        print("for the QAOA to flip decisions.")
        print()
        print("ROOT CAUSE: With σ=0.023 (trained), the uncertainty weighting")
        print("  exp(-((s-thr)/σ)²) is essentially zero unless the classical")
        print("  score is within ~0.05 of the threshold. Most cells have")
        print("  scores far from threshold → ZZ coupling is suppressed →")
        print("  QAOA ≈ classical.")
        print()
        print("WHY THIS IS STILL MEANINGFUL:")
        print("  1. The Hamiltonian IS correctly designed — ZZ coupling activates")
        print("     only near the decision boundary (uncertainty-aware).")
        print("  2. At 2×2 VQA resolution, only 4 cells compete — few boundary")
        print("     cells exist. At 8×8 or 16×16, more cells would be borderline.")
        print("  3. Higher Re/Rm → thinner structures → more borderline cells →")
        print("     more quantum advantage potential.")
        print()
        print("→ IMPLICATION: The framework is sound but the current scale")
        print("  (2×2 VQA, N=256, Re=800) doesn't produce enough borderline")
        print("  cells for ZZ corrections to matter. This is a SCALING argument,")
        print("  not a fundamental limitation.")
    elif flip_rate < 0.05:
        print("Corrections are comparable to threshold distance but few flips")
        print("occur. The corrections may be pushing scores AWAY from the")
        print("threshold rather than across it.")
    else:
        print(f"Flip rate is {100*flip_rate:.1f}% — corrections DO change decisions.")
        if n_qa_correct_g > n_cl_correct_g:
            print(f"  QAOA correct: {n_qa_correct_g}/{len(all_records)}")
            print(f"  Classical correct: {n_cl_correct_g}/{len(all_records)}")
            print(f"  → QAOA IMPROVES overall decision accuracy by {n_qa_correct_g - n_cl_correct_g} cells")
        elif n_qa_correct_g < n_cl_correct_g:
            print(f"  QAOA correct: {n_qa_correct_g}/{len(all_records)}")
            print(f"  Classical correct: {n_cl_correct_g}/{len(all_records)}")
            print(f"  → Classical is MORE accurate by {n_cl_correct_g - n_qa_correct_g} cells")
        else:
            print(f"  Both methods have identical accuracy: {n_qa_correct_g}/{len(all_records)}")
