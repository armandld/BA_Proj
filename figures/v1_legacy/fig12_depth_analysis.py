#!/usr/bin/env python3
"""
Fig 12 — Depth-Resolved AMR Analysis

Breaks down Q-HAS vs Classical performance by AMR tree depth.
Shows where quantum spatial correlations matter most (depth>0 bounded patches).

Panels:
  Row per scenario, 4 columns:
  A) Patch count by depth (stacked bar: Q-HAS vs Classical)
  B) Captured error fraction by depth (grouped bar)
  C) Decision agreement rate by depth (% of patches where both methods agree)
  D) Compute budget by depth (fraction of total refined pixels per depth)
"""
import sys, os
import numpy as np
from collections import defaultdict

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
    apply_style, COLORS, FIG_DIR, TRAINED_PARAMS, CLASSICAL_PARAMS,
    make_sim, ground_truth_errors,
    run_hierarchical_comparison, _compute_depths,
    filter_scenarios_dict,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

apply_style()

# ── Configuration ──
N = 256
TARGET_DIM = 2
MIN_SIZE = 6
N_TRIALS = 3

SCENARIOS = filter_scenarios_dict({
    'Kelvin-Helmholtz':   {'init': 'init_kelvin_helmholtz',   'n_steps': 400},
    'Harris Tearing':     {'init': 'init_harris_tearing',     'n_steps': 300},
    'MHD Rotor':          {'init': 'init_mhd_rotor',          'n_steps': 300},
    'Orszag-Tang':        {'init': 'init_orszag_tang',        'n_steps': 500},
})


def _patches_by_depth(patches, gt, N):
    """Group patches by depth and compute per-depth metrics.

    Returns dict: depth → {
        'count': int,
        'captured': float (GT error covered),
        'pixels': int (fine-resolution pixels),
        'bounds_list': list of bounds,
    }
    """
    from Simulation.utils import compute_local_factor

    max_depth = max((p['depth'] for p in patches), default=1)
    if max_depth == 0:
        max_depth = 1

    gt_total = gt.sum() + 1e-10
    by_depth = defaultdict(lambda: {'count': 0, 'captured': 0.0, 'pixels': 0,
                                     'bounds_list': []})

    for p in patches:
        d = p['depth']
        bounds = p['bounds']
        y_s, y_e, x_s, x_e = bounds
        ptype = p.get('type', 'leaf_depth')

        by_depth[d]['count'] += 1
        by_depth[d]['bounds_list'].append(bounds)

        # Compute effective pixels
        H, W = y_e - y_s, x_e - x_s
        area = H * W
        if ptype == 'coarse_leaf':
            local_factor = compute_local_factor(H, W, d, max_depth)
            eff_pixels = area / (local_factor ** 2) if local_factor > 0 else 0
        else:
            eff_pixels = area
        by_depth[d]['pixels'] += int(eff_pixels)

        # Compute captured GT in this patch's bounds
        # Handle periodic wrapping
        gt_patch = _extract_patch_gt(gt, y_s, y_e, x_s, x_e, N)
        if ptype != 'coarse_leaf':
            by_depth[d]['captured'] += gt_patch.sum() / gt_total

    return dict(by_depth)


def _extract_patch_gt(gt, y_s, y_e, x_s, x_e, N):
    """Extract GT values for a patch, handling periodic wrapping."""
    rows = np.arange(y_s, y_e) % N
    cols = np.arange(x_s, x_e) % N
    return gt[np.ix_(rows, cols)]


def _agreement_by_depth(qa_patches, cl_patches, N):
    """Compute agreement rate between Q-HAS and classical by depth.

    For each depth level, compare refined regions pixel-by-pixel.
    Returns dict: depth → agreement_rate (0-1).
    """
    max_depth_qa = max((p['depth'] for p in qa_patches), default=1)
    max_depth_cl = max((p['depth'] for p in cl_patches), default=1)
    max_depth = max(max_depth_qa, max_depth_cl, 1)

    agreement = {}
    for d in range(max_depth + 1):
        qa_mask = np.zeros((N, N), dtype=bool)
        cl_mask = np.zeros((N, N), dtype=bool)

        for p in qa_patches:
            if p['depth'] == d and p.get('type', '') != 'coarse_leaf':
                y_s, y_e, x_s, x_e = p['bounds']
                rows = np.arange(y_s, y_e) % N
                cols = np.arange(x_s, x_e) % N
                qa_mask[np.ix_(rows, cols)] = True

        for p in cl_patches:
            if p['depth'] == d and p.get('type', '') != 'coarse_leaf':
                y_s, y_e, x_s, x_e = p['bounds']
                rows = np.arange(y_s, y_e) % N
                cols = np.arange(x_s, x_e) % N
                cl_mask[np.ix_(rows, cols)] = True

        # Agreement = fraction of pixels where both agree
        total = N * N
        agree = np.sum(qa_mask == cl_mask)
        agreement[d] = agree / total if total > 0 else 1.0

    return agreement


def main():
    n_scenarios = len(SCENARIOS)
    if n_scenarios == 0:
        print("No scenarios for this phase — skipping fig12")
        return

    fig, axes = plt.subplots(n_scenarios, 4, figsize=(14, 3 * n_scenarios),
                              squeeze=False)

    max_depth = _compute_depths(N, TARGET_DIM, MIN_SIZE)
    log_lines = [f"Fig 12 — Depth-Resolved Analysis\nN={N}, max_depth={max_depth}\n"]

    for row, (label, cfg) in enumerate(SCENARIOS.items()):
        print(f"  [{label}] Running {N_TRIALS} trials...")
        all_qa_by_depth = defaultdict(lambda: {'count': [], 'captured': [], 'pixels': []})
        all_cl_by_depth = defaultdict(lambda: {'count': [], 'captured': [], 'pixels': []})
        all_agreement = defaultdict(list)

        for trial in range(N_TRIALS):
            sim, Phi_prev = make_sim(N, cfg['init'], cfg['n_steps'])
            gt = ground_truth_errors(sim, N)

            result = run_hierarchical_comparison(
                sim, N, Phi_prev=Phi_prev,
                target_dim=TARGET_DIM, min_size=MIN_SIZE,
            )

            qa_bd = _patches_by_depth(result['qaoa_patches'], gt, N)
            cl_bd = _patches_by_depth(result['classical_patches'], gt, N)
            agree = _agreement_by_depth(
                result['qaoa_patches'], result['classical_patches'], N
            )

            for d in range(max_depth + 1):
                qa_d = qa_bd.get(d, {'count': 0, 'captured': 0.0, 'pixels': 0})
                cl_d = cl_bd.get(d, {'count': 0, 'captured': 0.0, 'pixels': 0})
                all_qa_by_depth[d]['count'].append(qa_d['count'])
                all_qa_by_depth[d]['captured'].append(qa_d['captured'])
                all_qa_by_depth[d]['pixels'].append(qa_d['pixels'])
                all_cl_by_depth[d]['count'].append(cl_d['count'])
                all_cl_by_depth[d]['captured'].append(cl_d['captured'])
                all_cl_by_depth[d]['pixels'].append(cl_d['pixels'])
                if d in agree:
                    all_agreement[d].append(agree[d])

        depths = list(range(max_depth + 1))
        x = np.arange(len(depths))
        w = 0.35

        SHORT = {'Kelvin-Helmholtz': 'KH', 'Harris Tearing': 'Tearing',
                 'MHD Rotor': 'Rotor', 'Orszag-Tang': 'OT'}
        sn = SHORT.get(label, label)

        # ── Panel A: Patch count ──
        ax = axes[row, 0]
        qa_counts = [np.mean(all_qa_by_depth[d]['count']) for d in depths]
        cl_counts = [np.mean(all_cl_by_depth[d]['count']) for d in depths]
        qa_err = [np.std(all_qa_by_depth[d]['count']) for d in depths]
        cl_err = [np.std(all_cl_by_depth[d]['count']) for d in depths]
        ax.bar(x - w/2, qa_counts, w, yerr=qa_err, label='Q-HAS',
               color=COLORS['qaoa'], alpha=0.8, capsize=2)
        ax.bar(x + w/2, cl_counts, w, yerr=cl_err, label='Classical',
               color=COLORS['classical'], alpha=0.8, capsize=2)
        ax.set_ylabel('Patch Count', fontsize=8)
        ax.set_title(f'{sn}: Patches', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f'd={d}' for d in depths], fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('BFS Depth', fontsize=8)

        # ── Panel B: Captured GT fraction ──
        ax = axes[row, 1]
        qa_cap = [np.mean(all_qa_by_depth[d]['captured']) * 100 for d in depths]
        cl_cap = [np.mean(all_cl_by_depth[d]['captured']) * 100 for d in depths]
        qa_cap_err = [np.std(all_qa_by_depth[d]['captured']) * 100 for d in depths]
        cl_cap_err = [np.std(all_cl_by_depth[d]['captured']) * 100 for d in depths]
        ax.bar(x - w/2, qa_cap, w, yerr=qa_cap_err, label='Q-HAS',
               color=COLORS['qaoa'], alpha=0.8, capsize=2)
        ax.bar(x + w/2, cl_cap, w, yerr=cl_cap_err, label='Classical',
               color=COLORS['classical'], alpha=0.8, capsize=2)
        ax.set_ylabel('GT Captured (%)', fontsize=8)
        ax.set_title(f'{sn}: Error Captured', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f'd={d}' for d in depths], fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('BFS Depth', fontsize=8)

        # ── Panel C: Agreement rate ──
        # NOTE: High agreement (>90%) is expected — most BFS decisions are
        # far from the threshold, so QAOA corrections don't flip them.
        # Disagreements at the deepest level show where corrections matter.
        ax = axes[row, 2]
        agree_mean = [np.mean(all_agreement[d]) * 100 if all_agreement[d] else 100
                      for d in depths]
        agree_std = [np.std(all_agreement[d]) * 100 if all_agreement[d] else 0
                     for d in depths]
        bar_colors = ['#59A14F' if a > 95 else '#ECA63D' if a > 85 else '#D65F5F'
                      for a in agree_mean]
        bars = ax.bar(x, agree_mean, 0.6, yerr=agree_std,
                      color=bar_colors, alpha=0.8, capsize=2)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3, lw=0.8)
        ax.set_ylabel('Agreement (%)', fontsize=8)
        ax.set_title(f'{sn}: Decision Agreement', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f'd={d}' for d in depths], fontsize=7)
        ax.set_ylim(0, 105)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('BFS Depth', fontsize=8)
        # Annotate lowest-agreement depth
        min_agree = min(agree_mean)
        min_idx = agree_mean.index(min_agree)
        if min_agree < 100:
            ax.text(x[min_idx], min_agree - 2, f'{min_agree:.0f}%',
                    ha='center', va='top', fontsize=7, fontweight='bold',
                    color='#D65F5F')

        # ── Panel D: Compute budget ──
        ax = axes[row, 3]
        qa_px = [np.mean(all_qa_by_depth[d]['pixels']) for d in depths]
        cl_px = [np.mean(all_cl_by_depth[d]['pixels']) for d in depths]
        qa_total = sum(qa_px) + 1e-10
        cl_total = sum(cl_px) + 1e-10
        qa_frac = [p / qa_total * 100 for p in qa_px]
        cl_frac = [p / cl_total * 100 for p in cl_px]
        ax.bar(x - w/2, qa_frac, w, label='Q-HAS',
               color=COLORS['qaoa'], alpha=0.8)
        ax.bar(x + w/2, cl_frac, w, label='Classical',
               color=COLORS['classical'], alpha=0.8)
        ax.set_ylabel('Budget (%)', fontsize=8)
        ax.set_title(f'{sn}: Compute Budget', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f'd={d}' for d in depths], fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('BFS Depth', fontsize=8)

        # Log
        log_lines.append(f"\n{'='*50}")
        log_lines.append(f"Scenario: {label}")
        log_lines.append(f"{'='*50}")
        for d in depths:
            qa_c = np.mean(all_qa_by_depth[d]['count'])
            cl_c = np.mean(all_cl_by_depth[d]['count'])
            qa_cap_v = np.mean(all_qa_by_depth[d]['captured']) * 100
            cl_cap_v = np.mean(all_cl_by_depth[d]['captured']) * 100
            agr = np.mean(all_agreement[d]) * 100 if all_agreement[d] else 100
            log_lines.append(
                f"  Depth {d}: QA={qa_c:.1f} patches ({qa_cap_v:.1f}% GT), "
                f"CL={cl_c:.1f} patches ({cl_cap_v:.1f}% GT), "
                f"Agreement={agr:.1f}%"
            )

    fig.suptitle('Depth-Resolved AMR Analysis',
                 fontsize=11, fontweight='bold')
    fig.subplots_adjust(top=0.94, hspace=0.35, wspace=0.30)
    out = os.path.join(FIG_DIR, 'fig12_depth_analysis.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out}")

    log_path = os.path.join(FIG_DIR, 'fig12_depth_analysis.log')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"Log  → {log_path}")


if __name__ == '__main__':
    main()
