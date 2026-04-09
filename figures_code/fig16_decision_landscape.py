#!/usr/bin/env python3
"""
Fig 16 — Decision Landscape: Classical Score vs QAOA Probability
================================================================

For every cell in the hierarchical BFS tree, plots the classical
score (x) against the QAOA probability (y), coloured by ground-truth
label (should-refine vs should-not).

Key reading guide:
  - Points on the diagonal → QAOA agrees with classical
  - Off-diagonal points   → QAOA corrections
  - Green off-diagonal in correct quadrant → QAOA improves decision
  - Green off-diagonal in wrong  quadrant → QAOA hurts decision

Each scenario gets one panel (1 × 4 layout).  Threshold lines and
quadrant counts provide immediate quantitative summary.

Reuses the instrumented_bfs() engine from fig15_decision_flip_analysis.
"""
import sys, os
import numpy as np
from collections import defaultdict
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fig_utils import (
    apply_style, COLORS, TRAINED_PARAMS, CLASSICAL_PARAMS,
    make_sim, ground_truth_errors, _compute_depths,
    _hamilt_mapper_kwargs,
    filter_scenarios_dict,
    FIG_DIR,
)
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import _process_score
from Simulation.utils import get_periodic_patch
from Simulation.refinement import _prepare_vqa_input
from VQA.runtime import VQARuntime
from call_vqa_shell import call_vqa_shell

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

apply_style()

# ── Configuration ──
N = 256
TARGET_DIM = 2
MIN_SIZE = 6

SCENARIOS = filter_scenarios_dict({
    'init_kelvin_helmholtz': {'label': 'Kelvin-Helmholtz', 'n_steps': 400},
    'init_harris_tearing':   {'label': 'Harris Tearing',   'n_steps': 300},
    'init_mhd_rotor':        {'label': 'MHD Rotor',        'n_steps': 300},
    'init_orszag_tang':       {'label': 'Orszag-Tang',      'n_steps': 500},
})

SHORT = {'Kelvin-Helmholtz': 'KH', 'Harris Tearing': 'Tearing',
         'MHD Rotor': 'Rotor', 'Orszag-Tang': 'OT'}

solve_md = _compute_depths(N, TARGET_DIM, MIN_SIZE)
threshold = TRAINED_PARAMS['threshold_amr']


# ═══════════════════════════════════════════════════════════════════════
#  INSTRUMENTED BFS — capture per-cell decisions
# ═══════════════════════════════════════════════════════════════════════

def instrumented_bfs(sim, N, Phi_prev, threshold_amr, target_dim, max_depth,
                     min_size, gt_error_map):
    """Run VQA BFS and capture every per-cell decision.

    Returns list of dicts with classical_score, qaoa_prob, gt_error_mean,
    depth, correction, decision flags, etc.
    """
    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    grid = sim.grid
    HamiltMapper = PhysicalMapper(**_hamilt_mapper_kwargs(grid))

    reps = (target_dim - 1) * 2
    args = SimpleNamespace(
        reps=reps, mode="simulator", backend="state_vector",
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

    full_prev_h = full_prev_v = None
    AveragePhiDev = None
    if Phi_prev is not None:
        full_prev_h = Phi_prev['phi_horizontal']
        full_prev_v = Phi_prev['phi_vertical']
        AveragePhiDev = 0.5 * (np.mean(np.abs(full_h - full_prev_h))
                                + np.mean(np.abs(full_v - full_prev_v)))

    H, W = full_h.shape
    decision_log = []
    pending = [(0, H, 0, W)]
    depth = 0

    while pending and depth <= max_depth:
        next_level = []
        for bounds in pending:
            y_s, y_e, x_s, x_e = bounds
            height, width = y_e - y_s, x_e - x_s
            if height < min_size or width < min_size:
                continue

            pad = 1 if depth > 0 else 0
            local_score_raw = get_periodic_patch(full_score, y_s, y_e, x_s, x_e, pad=pad)
            is_periodic = (depth == 0)
            score_map_padded = _process_score(
                local_score_raw, is_periodic,
                target_dim + 2 * pad if pad > 0 else target_dim)
            score_map = score_map_padded[1:-1, 1:-1] if depth > 0 else score_map_padded

            if depth >= max_depth:
                continue

            prep = _prepare_vqa_input(
                full_h, full_v, full_prev_h, full_prev_v,
                full_score, physics_state, bounds, depth, mapper, args,
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

            result = call_vqa_shell(
                angles, mini_hamilt_params, False, args,
                period_bound=is_periodic, vqa_runtime=vqa_runtime,
            )

            if result is None:
                prob_map = prob_map_avant.copy()
            else:
                probs, _ = result
                ne = target_dim * target_dim
                probs_h = probs[:ne].reshape(target_dim, target_dim)
                probs_v = probs[ne:].reshape(target_dim, target_dim)
                prob_map = 0.5 * (probs_h + probs_v)

            step_y = height // target_dim
            step_x = width // target_dim

            for i in range(target_dim):
                for j in range(target_dim):
                    cl_s = float(score_map[i, j])
                    qa_p = float(prob_map[i, j])
                    sub_y_s = y_s + i * step_y
                    sub_y_e = y_s + (i + 1) * step_y if i < target_dim - 1 else y_e
                    sub_x_s = x_s + j * step_x
                    sub_x_e = x_s + (j + 1) * step_x if j < target_dim - 1 else x_e

                    gt_patch = gt_error_map[sub_y_s:sub_y_e, sub_x_s:sub_x_e]
                    gt_mean = float(np.mean(gt_patch))

                    dec_cl = cl_s >= threshold_amr
                    dec_qa = qa_p >= threshold_amr

                    decision_log.append({
                        'depth': depth,
                        'classical_score': cl_s,
                        'qaoa_prob': qa_p,
                        'correction': qa_p - cl_s,
                        'decision_classical': dec_cl,
                        'decision_qaoa': dec_qa,
                        'flipped': dec_cl != dec_qa,
                        'gt_error_mean': gt_mean,
                    })

                    if dec_qa:
                        next_level.append((sub_y_s, sub_y_e, sub_x_s, sub_x_e))

        pending = next_level
        depth += 1

    return decision_log


# ═══════════════════════════════════════════════════════════════════════
#  COLLECT DATA
# ═══════════════════════════════════════════════════════════════════════

n_scen = len(SCENARIOS)
if n_scen == 0:
    print("No scenarios for this phase.")
    sys.exit(0)

print("=" * 70)
print(f"Fig 16: Decision Landscape (N={N}, depth={solve_md})")
print("=" * 70)

all_logs = {}
for scenario_init, cfg in SCENARIOS.items():
    label = cfg['label']
    print(f"\n  [{label}] Running instrumented BFS...")
    sim, Phi_prev = make_sim(N, scenario_init, cfg['n_steps'])
    gt = ground_truth_errors(sim, N, TARGET_DIM)
    log = instrumented_bfs(
        sim, N, Phi_prev, threshold_amr=threshold,
        target_dim=TARGET_DIM, max_depth=solve_md,
        min_size=MIN_SIZE, gt_error_map=gt,
    )
    all_logs[label] = log
    n_flip = sum(1 for r in log if r['flipped'])
    print(f"    {len(log)} cells, {n_flip} flips ({100*n_flip/max(len(log),1):.1f}%)")


# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING — 1 × n_scen
# ═══════════════════════════════════════════════════════════════════════

n_scenarios = len(all_logs)
fig, axes = plt.subplots(1, n_scenarios, figsize=(3.5 * n_scenarios, 3.8))
if n_scenarios == 1:
    axes = [axes]

# Colours
C_REFINE = '#59A14F'    # green — should refine
C_SKIP   = '#A0A0A0'    # gray  — should not refine
C_FLIP   = '#D65F5F'    # red edge for flipped points

log_lines = [f"Fig 16 — Decision Landscape\n"
             f"N={N}, threshold={threshold:.4f}\n"]

for idx, (label, log) in enumerate(all_logs.items()):
    ax = axes[idx]
    sn = SHORT.get(label, label)

    if not log:
        ax.set_title(sn, fontsize=10)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='gray')
        continue

    cl = np.array([r['classical_score'] for r in log])
    qa = np.array([r['qaoa_prob'] for r in log])
    gt_err = np.array([r['gt_error_mean'] for r in log])
    flipped = np.array([r['flipped'] for r in log])

    # GT label: "should refine" if error above scenario mean
    gt_thr = np.mean(gt_err)
    should_refine = gt_err > gt_thr

    # ── Background: non-flipped cells ──
    nf = ~flipped
    # should-not-refine, no flip (majority, draw first)
    mask = nf & ~should_refine
    ax.scatter(cl[mask], qa[mask], s=8, alpha=0.25, c=C_SKIP,
               edgecolors='none', zorder=2, rasterized=True)
    # should-refine, no flip
    mask = nf & should_refine
    ax.scatter(cl[mask], qa[mask], s=8, alpha=0.35, c=C_REFINE,
               edgecolors='none', zorder=3, rasterized=True)

    # ── Flipped cells — larger, with edge ──
    # should-not-refine, flipped
    mask = flipped & ~should_refine
    ax.scatter(cl[mask], qa[mask], s=28, alpha=0.8, c=C_SKIP,
               edgecolors=C_FLIP, linewidths=0.8, zorder=5,
               marker='D')
    # should-refine, flipped
    mask = flipped & should_refine
    ax.scatter(cl[mask], qa[mask], s=28, alpha=0.8, c=C_REFINE,
               edgecolors=C_FLIP, linewidths=0.8, zorder=5,
               marker='D')

    # ── Reference lines ──
    ax.axvline(threshold, color='#555555', ls='--', lw=0.7, alpha=0.6)
    ax.axhline(threshold, color='#555555', ls='--', lw=0.7, alpha=0.6)
    # Diagonal
    lim_max = max(cl.max(), qa.max(), 0.5) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color='black', ls='-',
            lw=0.5, alpha=0.25, zorder=1)
    ax.set_xlim(-0.02, lim_max)
    ax.set_ylim(-0.02, lim_max)

    # ── Quadrant counts (flipped cells only) ──
    # Upper-left: QAOA refines, Classical skips
    ul = int(((qa >= threshold) & (cl < threshold)).sum())
    # Lower-right: QAOA skips, Classical refines
    lr = int(((qa < threshold) & (cl >= threshold)).sum())
    # For those flips, how many are correct?
    ul_correct = int(((qa >= threshold) & (cl < threshold) & should_refine).sum())
    lr_correct = int(((qa < threshold) & (cl >= threshold) & ~should_refine).sum())

    # ── Quadrant annotations ──
    n_flip_here = int(flipped.sum())

    # Compute overall accuracy for this scenario
    dec_qa_arr = np.array([r['decision_qaoa'] for r in log])
    dec_cl_arr = np.array([r['decision_classical'] for r in log])
    qa_acc = int((dec_qa_arr == should_refine).sum())
    cl_acc = int((dec_cl_arr == should_refine).sum())
    winner = 'Q-HAS' if qa_acc > cl_acc else ('Classical' if cl_acc > qa_acc else 'Tie')
    net = abs(qa_acc - cl_acc)

    # Threshold fraction in axes coordinates
    thr_frac = (threshold + 0.02) / (lim_max + 0.02)  # account for -0.02 xlim

    # Upper-left quadrant, just right of vertical threshold line
    if ul > 0:
        ax.text(thr_frac + 0.03, 0.96,
                f'QAOA adds +{ul}\n{ul_correct}/{ul} correct',
                fontsize=6.5, ha='left', va='top',
                color='#2E7D32',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#AAAAAA', alpha=0.9))
    # Lower-right quadrant, just above horizontal threshold line
    if lr > 0:
        ax.text(0.96, thr_frac + 0.03,
                f'QAOA removes \u2212{lr}\n{lr_correct}/{lr} correct',
                fontsize=6.5, ha='right', va='bottom',
                color='#C62828',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#AAAAAA', alpha=0.9))
    # Top-right: net accuracy summary (Déplacé du bas-gauche vers le HAUT-DROIT)
    ax.text(0.98, 0.98, # Modifié: coordonnées en haut à droite
            f'{winner} +{net}',
            fontsize=7, ha='right', va='top', # Modifié: alignement pour le coin haut-droit
            color='#333333', fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.2', fc='#F5F5F5',
                      ec='#999999', alpha=0.9))
    # ── Axes ──
    ax.set_xlabel('Classical Score', fontsize=9)
    if idx == 0:
        ax.set_ylabel('QAOA Probability', fontsize=9)
    else:
        ax.set_ylabel('')
    flip_pct = 100 * n_flip_here / max(len(log), 1)
    ax.set_title(f'{sn}  ({n_flip_here} flips, {flip_pct:.0f}%)',
                 fontsize=10, pad=12) 
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=8)

    # ── Per-scenario stats for log ──
    n_total = len(log)

    log_lines.append(f"\n{'='*50}")
    log_lines.append(f"{label} ({sn}):")
    log_lines.append(f"  Cells: {n_total}  |  Flips: {n_flip_here} ({100*n_flip_here/n_total:.1f}%)")
    log_lines.append(f"  QAOA correct: {qa_acc}/{n_total} ({100*qa_acc/n_total:.1f}%)")
    log_lines.append(f"  Classical correct: {cl_acc}/{n_total} ({100*cl_acc/n_total:.1f}%)")
    log_lines.append(f"  Net: {winner} by {net} cells")
    log_lines.append(f"  QAOA adds refinement: {ul} ({ul_correct} correct)")
    log_lines.append(f"  QAOA removes refinement: {lr} ({lr_correct} correct)")
    log_lines.append(f"  GT threshold (mean error): {gt_thr:.4f}")

# ── Shared legend (below figure) ──
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_REFINE,
           markersize=6, label='Should refine'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_SKIP,
           markersize=6, label='Should not refine'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
           markeredgecolor=C_FLIP, markeredgewidth=1.0,
           markersize=6, label='Decision flip'),
    Line2D([0], [0], color='#555555', ls='--', lw=0.7,
           label=f'Threshold ({threshold:.3f})'),
]
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.80,
                    wspace=0.28)
fig.legend(handles=legend_elements, loc='lower center',
           ncol=4, fontsize=8, frameon=True, framealpha=0.9,
           edgecolor='0.8', bbox_to_anchor=(0.5, 0.01))

fig.suptitle('Decision Landscape: Classical Score vs QAOA Probability',
             fontsize=11, fontweight='bold', y = 0.98)

out = os.path.join(FIG_DIR, 'fig16_decision_landscape.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved → {out}")

# ── Log ──
log_path = os.path.join(FIG_DIR, 'fig16_decision_landscape.log')
with open(log_path, 'w') as f:
    f.write('\n'.join(log_lines))
print(f"Log  → {log_path}")
