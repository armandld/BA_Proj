"""
Figure 3: Spatial Coherence & Topology Detection — Patch structure analysis.
Produces: figures/fig3_spatial_coherence.png

Analyzes whether Q-HAS produces more spatially coherent and topologically
meaningful patch layouts than classical AMR.

Key metrics:
  - Compactness: perimeter-to-area ratio of the refined region (lower = better)
  - Component density: connected components per unit refined area (lower = better)
  - GT precision: of refined pixels, fraction above GT error percentile
  - GT alignment curve: precision at multiple severity thresholds

Hypothesis: Q-HAS's ZZ/ZZZZ spatial correlations enforce spatial coherence
(neighbors align), producing fewer isolated patches and better alignment
with physical structures.
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
    run_hierarchical_comparison, patches_to_metrics, _compute_depths,
    filter_scenarios,
    FIG_DIR,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import label

apply_style()

N = 256
TARGET_DIM = 2
MIN_SIZE = 6
N_TRIALS = 3
GT_PERCENTILES = [50, 60, 70, 75, 80, 85, 90, 95]

scenarios = filter_scenarios([
    ('KH',      'init_kelvin_helmholtz',   400),
    ('Tearing', 'init_harris_tearing',     300),
    ('Rotor',   'init_mhd_rotor',          300),
    ('OT',      'init_orszag_tang',        500),
])

n_scen = len(scenarios)
if n_scen == 0:
    print("Aucun scenario pour cette phase.")
    sys.exit(0)

print(f"Fig 3: Spatial Coherence (N={N}, {N_TRIALS} trials)")


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def patches_to_fine_mask(patches, N):
    """Binary N x N mask of fine-resolution patches (leaf_depth/leaf_limit)."""
    mask = np.zeros((N, N), dtype=bool)
    for p in patches:
        if p.get('type') in ('leaf_depth', 'leaf_limit'):
            y0, y1, x0, x1 = p['bounds']
            mask[y0:y1, x0:x1] = True
    return mask


def compactness(mask):
    """Perimeter-to-area ratio of the refined region.

    Perimeter = number of boundary pixels (refined pixel adjacent to
    non-refined pixel). Lower ratio = more compact, coherent patches.
    Returns 0 if no refined pixels.

    D-99 : le remplissage etait `mode='constant', constant_values=False`,
    c'est-a-dire « hors du domaine, rien n'est raffine ». Le domaine de ce
    depot est PERIODIQUE (`PeriodicGrid`) : un pixel raffine du bord haut a
    pour voisin le bord bas. Toute structure qui traverse le bord etait donc
    comptee comme exposee des deux cotes, et rendue moins compacte qu'elle
    ne l'est. Mesure (N=256) : bande verticale traversante, du type d'une
    nappe de courant, 0,0698 contre 0,0625 en periodique (**+11,7 %**) ;
    bloc a cheval sur le bord, 0,1211 contre 0,0918 (**+31,9 %**) ; bloc
    central, identique — c'est le champ qui NE SEPARE PAS.
    """
    area = np.sum(mask)
    if area == 0:
        return 0.0
    # Count boundary pixels: refined pixel where at least one 4-neighbor is not refined
    padded = np.pad(mask, 1, mode='wrap')
    boundary = mask & (
        ~padded[:-2, 1:-1] | ~padded[2:, 1:-1] |   # top, bottom
        ~padded[1:-1, :-2] | ~padded[1:-1, 2:]      # left, right
    )
    perimeter = np.sum(boundary)
    return perimeter / area


def _label_periodic(mask):
    """Etiquetage 4-connexe sur un tore : compte les composantes du domaine
    PERIODIQUE, pas celles d'une image bornee.

    D-99 : `label(mask)` seul coupait toute region traversant un bord en deux
    composantes. Mesure : bloc a cheval sur le bord haut/bas, **2 composantes
    au lieu de 1** — `component_density` doublait pour cette region.

    On etiquette d'abord normalement, puis on fusionne (union-find) les
    etiquettes qui se font face de part et d'autre des deux paires de bords.
    La fusion est faite cellule a cellule en vis-a-vis : c'est exactement la
    4-connexite refermee sur le tore, la meme que celle de `label`.
    """
    labeled, n = label(mask)
    if n <= 1:
        return labeled, n
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for bord_a, bord_b in ((labeled[0, :], labeled[-1, :]),
                           (labeled[:, 0], labeled[:, -1])):
        for a, b in zip(bord_a, bord_b):
            if a and b:
                union(int(a), int(b))

    racines = {find(i) for i in range(1, n + 1)}
    return labeled, len(racines)


def component_density(mask):
    """Connected components per unit refined area, on the periodic domain.

    Lower = fewer, larger contiguous regions = more coherent.
    """
    area = np.sum(mask)
    if area == 0:
        return 0, 0.0
    labeled, n_comp = _label_periodic(mask)
    # Normalize: components per 1000 refined pixels (avoids tiny numbers)
    density = n_comp / (area / 1000)
    return n_comp, density


def gt_precision_at_percentile(mask, gt, percentile):
    """Of refined pixels, fraction where GT error > given percentile.

    High precision = the method targets genuinely high-error regions.
    """
    n_refined = np.sum(mask)
    if n_refined == 0:
        return 0.0
    threshold = np.percentile(gt, percentile)
    hot = gt >= threshold
    return np.sum(mask & hot) / n_refined


def draw_patch_boundaries(ax, patches, N, color, lw=1.0):
    """Draw fine-patch boundaries on an axes (rectangular outlines)."""
    for p in patches:
        if p.get('type') in ('leaf_depth', 'leaf_limit'):
            y0, y1, x0, x1 = p['bounds']
            rect = plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                  linewidth=lw, edgecolor=color,
                                  facecolor='none', alpha=0.8)
            ax.add_patch(rect)


# ══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(n_scen, 3, figsize=(16, 3.2 * n_scen))
if n_scen == 1:
    axes = axes[np.newaxis, :]

all_metrics = {'qa': [], 'cl': []}

for row, (name, init, n_steps) in enumerate(scenarios):
    print(f"\n{'='*50}")
    print(f"--- {name} ---")

    sim, Phi_prev = make_sim(N, init, n_steps)
    gt = ground_truth_errors(sim, N, TARGET_DIM)

    best_qa_thr = TRAINED_PARAMS['threshold_amr']
    best_cl_thr = CLASSICAL_PARAMS['threshold_amr']

    # Accumulate metrics over trials
    qa_compact_trials, cl_compact_trials = [], []
    qa_compdens_trials, cl_compdens_trials = [], []
    qa_ncomp_trials, cl_ncomp_trials = [], []
    # GT precision at each percentile: [N_TRIALS, len(GT_PERCENTILES)]
    qa_prec_trials = np.zeros((N_TRIALS, len(GT_PERCENTILES)))
    cl_prec_trials = np.zeros((N_TRIALS, len(GT_PERCENTILES)))

    # Store first trial's patches for visualization
    qa_patches_vis = None
    cl_patches_vis = None

    for trial in range(N_TRIALS):
        comp = run_hierarchical_comparison(
            sim, N, Phi_prev=Phi_prev,
            threshold_qa=best_qa_thr, threshold_cl=best_cl_thr,
            target_dim=TARGET_DIM, min_size=MIN_SIZE, K_opt=40,
        )

        if trial == 0:
            qa_patches_vis = comp['qaoa_patches']
            cl_patches_vis = comp['classical_patches']

        qa_mask = patches_to_fine_mask(comp['qaoa_patches'], N)
        cl_mask = patches_to_fine_mask(comp['classical_patches'], N)

        qa_compact_trials.append(compactness(qa_mask))
        cl_compact_trials.append(compactness(cl_mask))

        qa_nc, qa_cd = component_density(qa_mask)
        cl_nc, cl_cd = component_density(cl_mask)
        qa_ncomp_trials.append(qa_nc)
        cl_ncomp_trials.append(cl_nc)
        qa_compdens_trials.append(qa_cd)
        cl_compdens_trials.append(cl_cd)

        for pi, pct in enumerate(GT_PERCENTILES):
            qa_prec_trials[trial, pi] = gt_precision_at_percentile(qa_mask, gt, pct)
            cl_prec_trials[trial, pi] = gt_precision_at_percentile(cl_mask, gt, pct)

    # Aggregate
    qa_compact_mu, qa_compact_std = np.mean(qa_compact_trials), np.std(qa_compact_trials)
    cl_compact_mu, cl_compact_std = np.mean(cl_compact_trials), np.std(cl_compact_trials)
    qa_compdens_mu, qa_compdens_std = np.mean(qa_compdens_trials), np.std(qa_compdens_trials)
    cl_compdens_mu, cl_compdens_std = np.mean(cl_compdens_trials), np.std(cl_compdens_trials)
    qa_prec_mu = qa_prec_trials.mean(axis=0)
    qa_prec_std = qa_prec_trials.std(axis=0)
    cl_prec_mu = cl_prec_trials.mean(axis=0)
    cl_prec_std = cl_prec_trials.std(axis=0)

    # Also compute captured fraction for summary (from first trial)
    qa_m = patches_to_metrics(qa_patches_vis, gt, N, TARGET_DIM)
    cl_m = patches_to_metrics(cl_patches_vis, gt, N, TARGET_DIM)

    all_metrics['qa'].append({
        'compact': qa_compact_mu, 'compdens': qa_compdens_mu,
        'ncomp': np.mean(qa_ncomp_trials),
        'prec75': qa_prec_mu[GT_PERCENTILES.index(75)],
        'cap': qa_m['captured_fraction'],
    })
    all_metrics['cl'].append({
        'compact': cl_compact_mu, 'compdens': cl_compdens_mu,
        'ncomp': np.mean(cl_ncomp_trials),
        'prec75': cl_prec_mu[GT_PERCENTILES.index(75)],
        'cap': cl_m['captured_fraction'],
    })

    print(f"  Compactness (P/A):    QA={qa_compact_mu:.3f}+/-{qa_compact_std:.3f}  "
          f"CL={cl_compact_mu:.3f}+/-{cl_compact_std:.3f}  (lower=better)")
    print(f"  Component density:    QA={qa_compdens_mu:.2f}+/-{qa_compdens_std:.2f}  "
          f"CL={cl_compdens_mu:.2f}+/-{cl_compdens_std:.2f}  (lower=better)")
    print(f"  Components:           QA={np.mean(qa_ncomp_trials):.1f}  "
          f"CL={np.mean(cl_ncomp_trials):.1f}")
    print(f"  GT precision (p75):   QA={qa_prec_mu[GT_PERCENTILES.index(75)]:.3f}  "
          f"CL={cl_prec_mu[GT_PERCENTILES.index(75)]:.3f}")

    # ── Col 0: GT heatmap + both methods' patch boundaries overlaid ──
    ax = axes[row, 0]
    im = ax.imshow(gt, cmap='hot', origin='lower', aspect='equal')
    draw_patch_boundaries(ax, qa_patches_vis, N, color='cyan', lw=1.2)
    draw_patch_boundaries(ax, cl_patches_vis, N, color='#00FF00', lw=0.8)
    ax.set_title(f'{name}: Patches', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    legend_elements = [
        Line2D([0], [0], color='cyan', lw=2, label='Q-HAS'),
        Line2D([0], [0], color='#00FF00', lw=2, label='Classical'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

    # ── Col 1: Spatial coherence metrics (bar chart) ──
    ax = axes[row, 1]
    labels = ['Compactness\n(P/A ratio)', 'Component\ndensity']
    qa_vals = [qa_compact_mu, qa_compdens_mu]
    cl_vals = [cl_compact_mu, cl_compdens_mu]
    qa_errs = [qa_compact_std, qa_compdens_std]
    cl_errs = [cl_compact_std, cl_compdens_std]

    x = np.arange(len(labels))
    w = 0.35
    bars_qa = ax.bar(x - w/2, qa_vals, w, yerr=qa_errs, capsize=3,
                      label='Q-HAS', color=COLORS['qaoa'], alpha=0.85)
    bars_cl = ax.bar(x + w/2, cl_vals, w, yerr=cl_errs, capsize=3,
                      label='Classical', color=COLORS['classical'], alpha=0.85)
    # Value annotations — shifted left/right to avoid error bars
    for bar, val, err in zip(bars_qa, qa_vals, qa_errs):
        y_ann = val + err + 0.015  # above error bar
        ax.text(bar.get_x() + bar.get_width() * 0.3, y_ann,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7,
                color=COLORS['qaoa'])
    for bar, val, err in zip(bars_cl, cl_vals, cl_errs):
        y_ann = val + err + 0.015
        ax.text(bar.get_x() + bar.get_width() * 0.7, y_ann,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7,
                color=COLORS['classical'])
    # Ensure annotations stay inside the plot
    all_tops = [v + e + 0.08 for v, e in zip(qa_vals + cl_vals, qa_errs + cl_errs)]
    ax.set_ylim(0, max(all_tops) * 1.15)
    # Compactness ratio annotation
    compact_ratio = qa_compact_mu / max(cl_compact_mu, 1e-8)
    top_val = max(qa_compact_mu + qa_compact_std, cl_compact_mu + cl_compact_std)
    ax.text(x[0], top_val + 0.06, f'ratio {compact_ratio:.2f}\u00d7',
            ha='center', va='bottom', fontsize=7, fontstyle='italic', color='#555555')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Score', fontsize=9)
    ax.set_title(f'{name}: Coherence metrics')
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # ── Col 2: GT alignment curve at multiple percentile thresholds ──
    ax = axes[row, 2]
    ax.plot(GT_PERCENTILES, cl_prec_mu, 's-', color=COLORS['classical'],
            ms=5, lw=1.8, label='Classical')
    ax.fill_between(GT_PERCENTILES, cl_prec_mu - cl_prec_std,
                     cl_prec_mu + cl_prec_std,
                     color=COLORS['classical'], alpha=0.15)
    ax.plot(GT_PERCENTILES, qa_prec_mu, 'o-', color=COLORS['qaoa'],
            ms=5, lw=1.8, label='Q-HAS')
    ax.fill_between(GT_PERCENTILES, qa_prec_mu - qa_prec_std,
                     qa_prec_mu + qa_prec_std,
                     color=COLORS['qaoa'], alpha=0.15)
    ax.set_xlabel('GT error percentile threshold', fontsize=9)
    ax.set_ylabel('Precision (refined above threshold)', fontsize=9)
    ax.set_title(f'{name}: GT alignment')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=8)

fig.suptitle('Spatial Coherence Analysis', fontsize=13, fontweight='bold')
fig.subplots_adjust(top=0.93, bottom=0.05, hspace=0.35, wspace=0.28)
out = os.path.join(FIG_DIR, 'fig3_spatial_coherence.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"\nSaved: {out}")

# Print summary
print("\n" + "=" * 60)
print("SPATIAL COHERENCE SUMMARY (mean across scenarios)")
qa_compact_mean = np.mean([m['compact'] for m in all_metrics['qa']])
cl_compact_mean = np.mean([m['compact'] for m in all_metrics['cl']])
qa_compdens_mean = np.mean([m['compdens'] for m in all_metrics['qa']])
cl_compdens_mean = np.mean([m['compdens'] for m in all_metrics['cl']])
qa_prec75_mean = np.mean([m['prec75'] for m in all_metrics['qa']])
cl_prec75_mean = np.mean([m['prec75'] for m in all_metrics['cl']])
qa_ncomp_mean = np.mean([m['ncomp'] for m in all_metrics['qa']])
cl_ncomp_mean = np.mean([m['ncomp'] for m in all_metrics['cl']])
print(f"  Compactness (P/A):  QA={qa_compact_mean:.3f} vs CL={cl_compact_mean:.3f} (lower=better)")
print(f"  Component density:  QA={qa_compdens_mean:.2f} vs CL={cl_compdens_mean:.2f} (lower=better)")
print(f"  Components:         QA={qa_ncomp_mean:.1f} vs CL={cl_ncomp_mean:.1f}")
print(f"  GT precision (p75): QA={qa_prec75_mean:.3f} vs CL={cl_prec75_mean:.3f} (higher=better)")
