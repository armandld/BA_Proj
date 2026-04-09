"""
Figure 4: Comprehensive Performance Comparison — Q-HAS vs Classical AMR.
Produces: figures/fig4_comprehensive_comparison.png

Shows side-by-side performance across all scenarios using each method's
own trained threshold. Metrics:
  - Captured fraction (recall of GT error)
  - Pixel precision (of refined pixels, fraction above GT mean)
  - Compute ratio (fraction of domain at full resolution)
  - Efficiency = captured / compute (quality per unit cost)

Each method uses its independently trained threshold (quantum vs classical),
ensuring a fair comparison at their respective operating points.
"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fig_utils import (
    apply_style, COLORS, TRAINED_PARAMS, CLASSICAL_PARAMS, make_sim, ground_truth_errors,
    run_hierarchical_comparison, patches_to_metrics, _compute_depths,
    filter_scenarios,
    FIG_DIR,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

apply_style()

N = 256
TARGET_DIM = 2
MIN_SIZE = 6
N_TRIALS = 3

scenarios = filter_scenarios([
    ('Kelvin-Helmholtz',  'init_kelvin_helmholtz',   400),
    ('Harris Tearing',    'init_harris_tearing',     300),
    ('MHD Rotor',         'init_mhd_rotor',          300),
    ('Orszag-Tang',       'init_orszag_tang',        500),
])

n_scen = len(scenarios)
if n_scen == 0:
    print("Aucun scenario pour cette phase.")
    sys.exit(0)

best_qa_thr = TRAINED_PARAMS['threshold_amr']
best_cl_thr = CLASSICAL_PARAMS['threshold_amr']

print(f"Fig 4: Comprehensive Comparison (N={N}, {N_TRIALS} trials)")
print(f"  Q-HAS threshold:     {best_qa_thr:.4f}")
print(f"  Classical threshold:  {best_cl_thr:.4f}")


def pixel_precision(patches, gt, N):
    """Of refined pixels, fraction where GT error > mean."""
    mask = np.zeros((N, N), dtype=bool)
    for p in patches:
        if p.get('type') in ('leaf_depth', 'leaf_limit'):
            y0, y1, x0, x1 = p['bounds']
            mask[y0:y1, x0:x1] = True
    n_refined = np.sum(mask)
    if n_refined == 0:
        return 0.0
    return np.sum(mask & (gt > gt.mean())) / n_refined


def pixel_recall(patches, gt, N):
    """Of GT-high pixels, fraction covered by refinement."""
    mask = np.zeros((N, N), dtype=bool)
    for p in patches:
        if p.get('type') in ('leaf_depth', 'leaf_limit'):
            y0, y1, x0, x1 = p['bounds']
            mask[y0:y1, x0:x1] = True
    gt_mask = gt > gt.mean()
    n_needs = np.sum(gt_mask)
    if n_needs == 0:
        return 1.0
    return np.sum(mask & gt_mask) / n_needs


# ── Collect metrics ──
metric_names = ['Captured Fraction', 'Precision', 'Recall', 'Compute Ratio']
n_metrics = len(metric_names)

qa_data = np.zeros((n_scen, N_TRIALS, n_metrics))
cl_data = np.zeros((n_scen, N_TRIALS, n_metrics))
scen_names = []

for si, (name, init, n_steps) in enumerate(scenarios):
    print(f"\n--- {name} ---")
    scen_names.append(name)
    sim, Phi_prev = make_sim(N, init, n_steps)
    gt = ground_truth_errors(sim, N, TARGET_DIM)

    for trial in range(N_TRIALS):
        comp = run_hierarchical_comparison(
            sim, N, Phi_prev=Phi_prev,
            threshold_qa=best_qa_thr, threshold_cl=best_cl_thr,
            target_dim=TARGET_DIM, min_size=MIN_SIZE, K_opt=40,
        )
        qa_m = patches_to_metrics(comp['qaoa_patches'], gt, N, TARGET_DIM)
        cl_m = patches_to_metrics(comp['classical_patches'], gt, N, TARGET_DIM)

        qa_data[si, trial, 0] = qa_m['captured_fraction']
        cl_data[si, trial, 0] = cl_m['captured_fraction']
        qa_data[si, trial, 1] = pixel_precision(comp['qaoa_patches'], gt, N)
        cl_data[si, trial, 1] = pixel_precision(comp['classical_patches'], gt, N)
        qa_data[si, trial, 2] = pixel_recall(comp['qaoa_patches'], gt, N)
        cl_data[si, trial, 2] = pixel_recall(comp['classical_patches'], gt, N)
        qa_data[si, trial, 3] = qa_m['compute_ratio']
        cl_data[si, trial, 3] = cl_m['compute_ratio']

    # Print trial-0 summary
    print(f"  QA: cap={qa_data[si,:,0].mean():.3f} prec={qa_data[si,:,1].mean():.3f} "
          f"rec={qa_data[si,:,2].mean():.3f} comp={qa_data[si,:,3].mean():.3f}")
    print(f"  CL: cap={cl_data[si,:,0].mean():.3f} prec={cl_data[si,:,1].mean():.3f} "
          f"rec={cl_data[si,:,2].mean():.3f} comp={cl_data[si,:,3].mean():.3f}")

# ── Plot: grouped bar chart ──
short_names = ['KH', 'Tearing', 'Rotor', 'OT']
# Use only as many short names as we have scenarios
short_labels = short_names[:n_scen]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

x = np.arange(n_scen)
w = 0.35

for mi in range(n_metrics):
    ax = axes[mi]
    qa_mu = qa_data[:, :, mi].mean(axis=1)
    qa_std = qa_data[:, :, mi].std(axis=1)
    cl_mu = cl_data[:, :, mi].mean(axis=1)
    cl_std = cl_data[:, :, mi].std(axis=1)

    bars_cl = ax.bar(x - w/2, cl_mu, w, yerr=cl_std, capsize=3,
                     color=COLORS['classical'], alpha=0.85, label='Classical')
    bars_qa = ax.bar(x + w/2, qa_mu, w, yerr=qa_std, capsize=3,
                     color=COLORS['qaoa'], alpha=0.85, label='Q-HAS')

    # Value annotations
    for bar, val in zip(bars_cl, cl_mu):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
    for bar, val in zip(bars_qa, qa_mu):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9, rotation=0, ha='center')
    ax.set_ylabel(metric_names[mi])
    ax.set_title(metric_names[mi], fontweight='bold', pad=12)
    ax.legend(fontsize=8)

    if mi != 3:  # compute ratio can exceed 1
        ax.set_ylim(0, 1.15)

fig.suptitle('Q-HAS vs Classical AMR Comparison',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig4_comprehensive_comparison.png')
plt.savefig(out, dpi=300)
print(f"\nSaved: {out}")

# ── Print summary table ──
print("\n" + "=" * 70)
print("COMPREHENSIVE COMPARISON SUMMARY")
print(f"{'Scenario':<20} {'Metric':<15} {'Q-HAS':>8} {'Classical':>10} {'Delta':>8}")
print("-" * 70)
for si in range(n_scen):
    for mi, mname in enumerate(['Captured', 'Precision', 'Recall', 'Compute']):
        qa_v = qa_data[si, :, mi].mean()
        cl_v = cl_data[si, :, mi].mean()
        delta = qa_v - cl_v
        prefix = scen_names[si] if mi == 0 else ''
        print(f"{prefix:<20} {mname:<15} {qa_v:>8.3f} {cl_v:>10.3f} {delta:>+8.3f}")
    print()
