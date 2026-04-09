"""
Figure 9: Synthetic Unit Tests — Controlled patterns for Q-HAS.
Produces: figures/fig9_synthetic_unit_tests.png

Tests Q-HAS on synthetic MHD fields with known anomaly topology,
designed to exercise specific Hamiltonian terms:
  - Vortex core:    smooth curl structure → ZZ edge (curl) terms
  - Current sheet:  thin gradient layer   → ZZ edge (gradient) terms
  - X-point:        reconnection topology → ZZZZ plaquette terms
  - Uniform noise:  negative control      → false positive rate

Multiple trials for statistical confidence.
"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fig_utils import (
    apply_style, COLORS, TRAINED_PARAMS, CLASSICAL_PARAMS, ground_truth_errors,
    run_hierarchical_comparison, patches_to_metrics, _compute_depths,
    FIG_DIR,
)
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from matplotlib.lines import Line2D

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

apply_style()

N = 256
TARGET_DIM = 2
MIN_SIZE = 6
N_TRIALS = 3


def patches_to_fine_mask(patches, N):
    mask = np.zeros((N, N), dtype=bool)
    for p in patches:
        if p.get('type') in ('leaf_depth', 'leaf_limit'):
            y0, y1, x0, x1 = p['bounds']
            mask[y0:y1, x0:x1] = True
    return mask


def pixel_prf(patches, gt, N):
    """Pixel-level precision, recall, F1 against GT > mean."""
    refined = patches_to_fine_mask(patches, N)
    needs = gt > gt.mean()
    tp = np.sum(refined & needs)
    prec = tp / max(np.sum(refined), 1)
    rec = tp / max(np.sum(needs), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return prec, rec, f1


def draw_patch_boundaries(ax, patches, N, color, lw=1.0):
    for p in patches:
        if p.get('type') in ('leaf_depth', 'leaf_limit'):
            y0, y1, x0, x1 = p['bounds']
            rect = plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                  lw=lw, edgecolor=color, facecolor='none',
                                  alpha=0.8)
            ax.add_patch(rect)


# ── Synthetic pattern generators ──

def make_vortex_core(N):
    """Smooth vortex flow — curl anomaly at center."""
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    y, x = np.mgrid[0:N, 0:N] / N * 2 * np.pi
    sim.vx = -np.sin(y)
    sim.vy = np.sin(x)
    sim.Bx = np.cos(y) * 0.5
    sim.By = -np.cos(x) * 0.5
    for _ in range(50):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    return sim, 'Vortex Core'


def make_current_sheet(N):
    """Harris-like current sheet — thin gradient at x=N/2.

    Uses stronger amplitude and fewer evolution steps to preserve the
    sharp structure that exercises gradient-based detection.
    """
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    x = np.arange(N) / N
    # Strong sheared flow + magnetic field with sharp gradient
    sim.vy = 0.5 * np.tanh((x - 0.5) * 40)[np.newaxis, :] * np.ones((N, 1))
    sim.vx[:] = 0
    sim.By[:] = np.tanh((x - 0.5) * 40)[np.newaxis, :]
    sim.Bx[:] = 0.3
    for _ in range(20):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    return sim, 'Current Sheet'


def make_xpoint(N):
    """Reconnection X-point — complex topology with strong gradients."""
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    y_arr, x_arr = np.mgrid[0:N, 0:N] / N
    # Stronger fields for clearer signal
    sim.By = 1.5 * (np.tanh((x_arr - 0.25) * 30) - np.tanh((x_arr - 0.75) * 30) - 1.0)
    sim.Bx = 1.5 * np.tanh((y_arr - 0.5) * 30)
    sim.vx = 0.3 * np.sin(2 * np.pi * x_arr) * np.cos(2 * np.pi * y_arr)
    sim.vy = -0.3 * np.cos(2 * np.pi * x_arr) * np.sin(2 * np.pi * y_arr)
    for _ in range(20):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    return sim, 'X-point'


def make_uniform_noise(N):
    """Uniform fields + tiny noise — negative control (no anomaly)."""
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    rng = np.random.default_rng(0)
    sim.vx = 0.01 * rng.standard_normal((N, N))
    sim.vy = 0.01 * rng.standard_normal((N, N))
    sim.Bx = 1.0 + 0.01 * rng.standard_normal((N, N))
    sim.By = 0.01 * rng.standard_normal((N, N))
    for _ in range(50):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    return sim, 'Uniform Noise'


# ── Main ──
patterns = [make_vortex_core, make_current_sheet, make_xpoint, make_uniform_noise]
n_patterns = len(patterns)

print(f"Fig 9: Synthetic Unit Tests (N={N}, {N_TRIALS} trials)")

fig, axes = plt.subplots(n_patterns, 3, figsize=(11, 3.2 * n_patterns))

for row, make_fn in enumerate(patterns):
    sim, pattern_label = make_fn(N)
    gt = ground_truth_errors(sim, N, TARGET_DIM)
    print(f"\n--- {pattern_label} ---")
    print(f"  GT: min={gt.min():.4f} max={gt.max():.4f} mean={gt.mean():.4f}")

    best_qa_thr = TRAINED_PARAMS['threshold_amr']
    best_cl_thr = CLASSICAL_PARAMS['threshold_amr']

    qa_prec_all, cl_prec_all = [], []
    qa_rec_all, cl_rec_all = [], []
    qa_f1_all, cl_f1_all = [], []
    qa_patches_vis, cl_patches_vis = None, None

    for trial in range(N_TRIALS):
        comp = run_hierarchical_comparison(
            sim, N, Phi_prev=None,
            threshold_qa=best_qa_thr, threshold_cl=best_cl_thr,
            target_dim=TARGET_DIM, min_size=MIN_SIZE, K_opt=40,
        )
        if trial == 0:
            qa_patches_vis = comp['qaoa_patches']
            cl_patches_vis = comp['classical_patches']

        qa_p, qa_r, qa_f = pixel_prf(comp['qaoa_patches'], gt, N)
        cl_p, cl_r, cl_f = pixel_prf(comp['classical_patches'], gt, N)
        qa_prec_all.append(qa_p); qa_rec_all.append(qa_r); qa_f1_all.append(qa_f)
        cl_prec_all.append(cl_p); cl_rec_all.append(cl_r); cl_f1_all.append(cl_f)

    print(f"  QA: P={np.mean(qa_prec_all):.3f} R={np.mean(qa_rec_all):.3f} "
          f"F1={np.mean(qa_f1_all):.3f}")
    print(f"  CL: P={np.mean(cl_prec_all):.3f} R={np.mean(cl_rec_all):.3f} "
          f"F1={np.mean(cl_f1_all):.3f}")

    # Col 0: Field visualization (|B| magnitude or vorticity)
    ax = axes[row, 0]
    B_mag = np.sqrt(sim.Bx**2 + sim.By**2)
    im = ax.imshow(B_mag, cmap='inferno', origin='lower', aspect='equal')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('|B|', fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax.set_title(f'{pattern_label} — |B|', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # Col 1: GT error + patch boundaries overlaid
    ax = axes[row, 1]
    im = ax.imshow(gt, cmap='hot', origin='lower', aspect='equal')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Error', fontsize=6)
    cb.ax.tick_params(labelsize=5)
    draw_patch_boundaries(ax, qa_patches_vis, N, color='cyan', lw=1.0)
    draw_patch_boundaries(ax, cl_patches_vis, N, color='#00FF00', lw=0.7)
    ax.set_title('GT Error + Patches', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[
        Line2D([0], [0], color='cyan', lw=1.5, label='Q-HAS'),
        Line2D([0], [0], color='#00FF00', lw=1.5, label='Classical'),
    ], fontsize=5, loc='upper right', framealpha=0.7)

    # Col 2: Precision / Recall / F1 bars with value annotations
    ax = axes[row, 2]
    labels = ['Prec.', 'Rec.', 'F1']
    qa_vals = [np.mean(qa_prec_all), np.mean(qa_rec_all), np.mean(qa_f1_all)]
    cl_vals = [np.mean(cl_prec_all), np.mean(cl_rec_all), np.mean(cl_f1_all)]
    qa_errs = [np.std(qa_prec_all), np.std(qa_rec_all), np.std(qa_f1_all)]
    cl_errs = [np.std(cl_prec_all), np.std(cl_rec_all), np.std(cl_f1_all)]

    x = np.arange(len(labels))
    w = 0.32
    bars_qa = ax.bar(x - w/2, qa_vals, w, yerr=qa_errs, capsize=2,
                     color=COLORS['qaoa'], label='Q-HAS', alpha=0.85)
    bars_cl = ax.bar(x + w/2, cl_vals, w, yerr=cl_errs, capsize=2,
                     color=COLORS['classical'], label='Classical', alpha=0.85)
    # Value annotations — smaller font, 2 decimal places
    for bar, val in zip(bars_qa, qa_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=5.5)
    for bar, val in zip(bars_cl, cl_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.2)
    ax.set_title('Decision Quality', fontsize=8)
    ax.legend(fontsize=6, loc='upper right')

    # Summary text box
    delta_f1 = np.mean(qa_f1_all) - np.mean(cl_f1_all)
    if abs(delta_f1) < 0.01:
        verdict = "Equivalent"
    elif delta_f1 > 0:
        verdict = f"Q-HAS +{delta_f1:.3f}"
    else:
        verdict = f"Classical +{-delta_f1:.3f}"
    ax.text(0.98, 0.02, verdict, transform=ax.transAxes,
            fontsize=6.5, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.85))

fig.suptitle('Synthetic Unit Tests', fontsize=11, fontweight='bold')
fig.subplots_adjust(top=0.93, bottom=0.03, left=0.04, right=0.97,
                    hspace=0.30, wspace=0.25)
out = os.path.join(FIG_DIR, 'fig9_synthetic_unit_tests.png')
plt.savefig(out, dpi=300)
print(f"\nSaved: {out}")
