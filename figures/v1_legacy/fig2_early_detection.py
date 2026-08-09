"""
Figure 2: Early Detection & Temporal Stability — Hierarchical AMR.
Produces: figures/fig2_early_detection.png

Tests whether Q-HAS's spatial correlations provide:
- Better prediction of where instabilities will develop
  (AMR at early time captures late-time GT errors)
- Higher precision (of what you refine, how much actually matters)
- More stable patch selections over time (pixel-level IoU)

Key hypothesis: The QAOA Hamiltonian encodes topology (vortex cores,
current sheets) that are PRECURSORS to instabilities, while the
classical score only detects CURRENT gradients.

Uses hierarchical AMR on N=256 grids with 2x2 VQA (8 qubits).
"""
import numpy as np
import os, sys
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
    apply_style, COLORS, TRAINED_PARAMS, CLASSICAL_PARAMS, make_sim, ground_truth_errors,
    run_hierarchical_comparison, patches_to_metrics,
    _compute_depths, filter_scenarios_dict,
    FIG_DIR,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

apply_style()

N = 256
TARGET_DIM = 2
MIN_SIZE = 6
N_POINTS = 6       # evaluation timesteps (enough for a smooth curve)
N_TRIALS = 1       # single VQA run per timestep
START_FRAC = 0.10  # earliest timestep = 10% of late_steps

# Scenario configs: use SHORTER simulation periods to reduce warmup time.
# The "late_steps" are reduced from training values to keep total runtime
# manageable on Colab while providing more temporal resolution (6 points).
configs = filter_scenarios_dict({
    'Kelvin-Helmholtz': {'init': 'init_kelvin_helmholtz',   'late_steps': 100},
    'Harris Tearing':   {'init': 'init_harris_tearing',     'late_steps': 80},
    'MHD Rotor':        {'init': 'init_mhd_rotor',          'late_steps': 80},
    'Orszag-Tang':      {'init': 'init_orszag_tang',        'late_steps': 120},
})

SHORT_NAMES = {
    'Kelvin-Helmholtz': 'KH',
    'Harris Tearing':   'Tearing',
    'MHD Rotor':        'Rotor',
    'Orszag-Tang':      'OT',
}

n_cols = len(configs)
if n_cols == 0:
    print("Aucun scenario pour cette phase.")
    sys.exit(0)

solve_md = _compute_depths(N, TARGET_DIM, MIN_SIZE)
print(f"Fig 2: Early Detection (N={N}, depth={solve_md}, "
      f"{N_POINTS} time-points, {N_TRIALS} trials)")


# ══════════════════════════════════════════════════════════════════════
#  HELPERS — pixel-level patch evaluation
# ══════════════════════════════════════════════════════════════════════

def patches_to_binary_mask(patches, N):
    """Convert hierarchical patches to an N x N binary refinement mask.

    A pixel is marked 1 if it falls inside any leaf_depth or leaf_limit
    patch (= full resolution). Coarse patches are ignored since they
    represent reduced-resolution regions, not true refinement decisions.
    """
    mask = np.zeros((N, N), dtype=bool)
    for p in patches:
        if p.get('type') in ('leaf_depth', 'leaf_limit'):
            y0, y1, x0, x1 = p['bounds']
            mask[y0:y1, x0:x1] = True
    return mask


def gt_needs_refinement(gt, threshold='mean'):
    """Binary mask of pixels that truly need refinement.

    Uses mean GT error as the threshold: pixels above the mean
    correspond to regions where the simulation under-resolves the
    physics. This matches the AMR decision: refine where error is high.
    """
    if threshold == 'mean':
        thr = gt.mean()
    else:
        thr = float(threshold)
    return gt > thr


def pixel_precision_recall_f1(patches, gt, N):
    """Compute pixel-level precision, recall, and F1.

    - refined_mask: pixels inside fine-resolution AMR patches
    - gt_mask:      pixels where GT error > mean (need refinement)

    Precision = |refined & needs_ref| / |refined|
      → Of what you refined, how much actually needed it?
    Recall    = |refined & needs_ref| / |needs_ref|
      → Of what needed refinement, how much did you catch?
    F1        = harmonic mean
    """
    refined = patches_to_binary_mask(patches, N)
    needs_ref = gt_needs_refinement(gt)

    tp = np.sum(refined & needs_ref)
    n_refined = np.sum(refined)
    n_needs = np.sum(needs_ref)

    precision = tp / max(n_refined, 1)
    recall = tp / max(n_needs, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def pixel_iou(patches_a, patches_b, N):
    """Pixel-level IoU between two patch sets.

    Measures spatial overlap of refinement decisions:
    IoU = |mask_a & mask_b| / |mask_a | mask_b|

    Unlike exact-bound Jaccard, this correctly handles partial overlaps
    and depth differences in the hierarchical tree.
    """
    mask_a = patches_to_binary_mask(patches_a, N)
    mask_b = patches_to_binary_mask(patches_b, N)
    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    return intersection / max(union, 1)


# ══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 10))
if n_cols == 1:
    axes = axes[:, np.newaxis]

for col_idx, (scen_name, cfg) in enumerate(configs.items()):
    print(f"\n{'='*50}")
    print(f"--- {scen_name} ---")

    late = cfg['late_steps']
    start = max(int(START_FRAC * late), 10)
    # Round to integers, exclude the final step (that's what we predict)
    steps = np.unique(np.linspace(start, int(0.9 * late), N_POINTS).astype(int))
    n_steps_actual = len(steps)

    # Compute late-time ground truth ONCE
    sim_late, Phi_late = make_sim(N, cfg['init'], late)
    gt_late = ground_truth_errors(sim_late, N, TARGET_DIM)

    best_qa_thr = TRAINED_PARAMS['threshold_amr']
    best_cl_thr = CLASSICAL_PARAMS['threshold_amr']

    # Storage: per-trial arrays [N_TRIALS, n_steps]
    qa_recall_all = np.zeros((N_TRIALS, n_steps_actual))
    cl_recall_all = np.zeros((N_TRIALS, n_steps_actual))
    qa_prec_all = np.zeros((N_TRIALS, n_steps_actual))
    cl_prec_all = np.zeros((N_TRIALS, n_steps_actual))
    qa_f1_all = np.zeros((N_TRIALS, n_steps_actual))
    cl_f1_all = np.zeros((N_TRIALS, n_steps_actual))

    # IoU arrays: [N_TRIALS, n_steps-1]
    qa_iou_all = np.zeros((N_TRIALS, max(n_steps_actual - 1, 1)))
    cl_iou_all = np.zeros((N_TRIALS, max(n_steps_actual - 1, 1)))

    for trial in range(N_TRIALS):
        print(f"  Trial {trial+1}/{N_TRIALS}")
        qa_patches_prev = None
        cl_patches_prev = None

        for si, es in enumerate(steps):
            sim_early, Phi_prev = make_sim(N, cfg['init'], es)

            comp = run_hierarchical_comparison(
                sim_early, N, Phi_prev=Phi_prev,
                threshold_qa=best_qa_thr, threshold_cl=best_cl_thr,
                target_dim=TARGET_DIM, min_size=MIN_SIZE, K_opt=20,
            )

            # Precision / Recall / F1 against late GT
            qa_p, qa_r, qa_f = pixel_precision_recall_f1(
                comp['qaoa_patches'], gt_late, N)
            cl_p, cl_r, cl_f = pixel_precision_recall_f1(
                comp['classical_patches'], gt_late, N)

            qa_recall_all[trial, si] = qa_r
            cl_recall_all[trial, si] = cl_r
            qa_prec_all[trial, si] = qa_p
            cl_prec_all[trial, si] = cl_p
            qa_f1_all[trial, si] = qa_f
            cl_f1_all[trial, si] = cl_f

            # Pixel IoU vs previous timestep
            if qa_patches_prev is not None:
                qa_iou_all[trial, si - 1] = pixel_iou(
                    qa_patches_prev, comp['qaoa_patches'], N)
                cl_iou_all[trial, si - 1] = pixel_iou(
                    cl_patches_prev, comp['classical_patches'], N)

            qa_patches_prev = comp['qaoa_patches']
            cl_patches_prev = comp['classical_patches']

            if trial == 0:
                print(f"    t={es:4d}: QA R={qa_r:.3f} P={qa_p:.3f} F1={qa_f:.3f}"
                      f"  |  CL R={cl_r:.3f} P={cl_p:.3f} F1={cl_f:.3f}")

    # ── Aggregate: mean ± std ──
    qa_recall_mu, qa_recall_std = qa_recall_all.mean(0), qa_recall_all.std(0)
    cl_recall_mu, cl_recall_std = cl_recall_all.mean(0), cl_recall_all.std(0)
    qa_prec_mu, qa_prec_std = qa_prec_all.mean(0), qa_prec_all.std(0)
    cl_prec_mu, cl_prec_std = cl_prec_all.mean(0), cl_prec_all.std(0)
    qa_f1_mu, qa_f1_std = qa_f1_all.mean(0), qa_f1_all.std(0)
    cl_f1_mu, cl_f1_std = cl_f1_all.mean(0), cl_f1_all.std(0)
    qa_iou_mu, qa_iou_std = qa_iou_all.mean(0), qa_iou_all.std(0)
    cl_iou_mu, cl_iou_std = cl_iou_all.mean(0), cl_iou_all.std(0)

    mid_steps = 0.5 * (steps[:-1] + steps[1:])

    sn = SHORT_NAMES.get(scen_name, scen_name)

    def _plot_with_band(ax, x, mu, std, color, marker, label):
        ax.plot(x, mu, f'{marker}-', color=color, ms=3, lw=1.2, label=label)
        if N_TRIALS > 1:
            ax.fill_between(x, mu - std, mu + std, color=color, alpha=0.15)

    # ── Row 0: Recall (captured fraction of late GT error) ──
    ax = axes[0, col_idx]
    _plot_with_band(ax, steps, cl_recall_mu, cl_recall_std,
                    COLORS['classical'], 's', 'Classical AMR')
    _plot_with_band(ax, steps, qa_recall_mu, qa_recall_std,
                    COLORS['qaoa'], 'o', 'Q-HAS')
    ax.set_title(f'{sn}: Recall', fontweight='bold')
    ax.set_ylabel('Recall', fontsize=9)
    ax.legend(fontsize=7, loc='best')
    all_v = np.concatenate([qa_recall_mu, cl_recall_mu])
    ymin = max(0, all_v.min() - 0.05)
    ax.set_ylim(ymin, min(all_v.max() + 0.05, 1.05))

    # ── Row 1: Precision (of refined pixels, fraction that matters) ──
    ax = axes[1, col_idx]
    _plot_with_band(ax, steps, cl_prec_mu, cl_prec_std,
                    COLORS['classical'], 's', 'Classical AMR')
    _plot_with_band(ax, steps, qa_prec_mu, qa_prec_std,
                    COLORS['qaoa'], 'o', 'Q-HAS')
    ax.set_title(f'{sn}: Precision')
    ax.set_ylabel('Precision', fontsize=9)
    ax.legend(fontsize=7, loc='best')
    all_p = np.concatenate([qa_prec_mu, cl_prec_mu])
    ax.set_ylim(max(0, all_p.min() - 0.05), min(all_p.max() + 0.05, 1.05))

    # ── Row 2: Pixel-level IoU stability (consecutive timesteps) ──
    ax = axes[2, col_idx]
    _plot_with_band(ax, mid_steps, cl_iou_mu, cl_iou_std,
                    COLORS['classical'], 's', 'Classical AMR')
    _plot_with_band(ax, mid_steps, qa_iou_mu, qa_iou_std,
                    COLORS['qaoa'], 'o', 'Q-HAS')
    ax.axhline(y=1.0, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Pixel IoU', fontsize=9)
    ax.set_title(f'{sn}: IoU Stability')
    ax.legend(fontsize=7, loc='best')
    ax.set_ylim(-0.05, 1.1)

    # Print scenario summary
    print(f"  Summary ({N_TRIALS} trials, {n_steps_actual} timesteps):")
    print(f"    Recall  — QA: {qa_recall_mu.mean():.3f} +/- {qa_recall_std.mean():.3f}"
          f"  CL: {cl_recall_mu.mean():.3f} +/- {cl_recall_std.mean():.3f}")
    print(f"    Precision — QA: {qa_prec_mu.mean():.3f} +/- {qa_prec_std.mean():.3f}"
          f"  CL: {cl_prec_mu.mean():.3f} +/- {cl_prec_std.mean():.3f}")
    print(f"    F1      — QA: {qa_f1_mu.mean():.3f}  CL: {cl_f1_mu.mean():.3f}")
    print(f"    IoU     — QA: {qa_iou_mu.mean():.3f} +/- {qa_iou_std.mean():.3f}"
          f"  CL: {cl_iou_mu.mean():.3f} +/- {cl_iou_std.mean():.3f}")

fig.suptitle('Early Detection & Temporal Stability',
             fontsize=13, fontweight='bold')

# X-axis label on ALL rows
for row in range(3):
    for col in range(n_cols):
        axes[row, col].set_xlabel('Timestep', fontsize=9)
        axes[row, col].tick_params(labelsize=8)

fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.07,
                    hspace=0.35, wspace=0.30)
out = os.path.join(FIG_DIR, 'fig2_early_detection.png')
plt.savefig(out, dpi=300)
print(f"\nSaved: {out}")
