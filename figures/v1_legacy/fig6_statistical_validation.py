"""
Figure 6: Statistical Validation — Hierarchical Q-HAS vs Classical AMR.
Produces: figures/fig6_statistical_validation.png

FAIR PROTOCOL: Both methods run on identical sim states.
Each seed creates a genuinely different simulation (tiny initial
perturbation) so the captured-fraction samples are independent.

Reports: bootstrap 95% CI, permutation p-value, and Cohen's d.
"""
import numpy as np
import os, sys
import json
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
from matplotlib.patches import Patch

apply_style()

N = 256
TARGET_DIM = 2
MIN_SIZE = 6
N_SEEDS = 10
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000

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

print(f"Fig 6: Statistical Validation (N={N}, {N_SEEDS} seeds)")


def perturb_sim(sim, seed, eps=1e-4):
    """Add tiny perturbation to sim fields for independent samples.

    The perturbation is small enough not to change the physics
    qualitatively, but large enough to produce different VQA decisions
    and different GT error maps — giving genuinely independent samples.
    """
    rng = np.random.default_rng(seed)
    for field_name in ['vx', 'vy', 'Bx', 'By']:
        field = getattr(sim, field_name)
        rms = max(np.std(field), 1e-10)
        setattr(sim, field_name,
                field + eps * rms * rng.standard_normal(field.shape))


def permutation_pvalue(deltas, n_perm, rng):
    """One-sided sign-flip permutation test."""
    observed = np.mean(deltas)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(deltas))
        if np.mean(signs * deltas) >= observed:
            count += 1
    return count / n_perm


CACHE_PATH = os.path.join(FIG_DIR, '.fig6_cache.json')

use_cache = os.path.exists(CACHE_PATH) and '--recompute' not in sys.argv

if use_cache:
    with open(CACHE_PATH, 'r') as f:
        cached = json.load(f)
    all_results = {}
    for scen_name, data in cached.items():
        all_results[scen_name] = {
            'qa': np.array(data['qa']),
            'cl': np.array(data['cl']),
            'delta': np.array(data['delta']),
            'ci': np.array(data['ci']),
            'p_val': float(data['p_val']),
            'cohens_d': float(data['cohens_d']),
            'qa_compute': np.array(data['qa_compute']),
            'cl_compute': np.array(data['cl_compute']),
        }
    print("Loaded from cache \u2014 replotting only. Use --recompute to force.")
else:
    all_results = {}
    for scen_name, init_method, n_steps in scenarios:
        print(f"\n--- {scen_name} ---")

        best_qa_thr = TRAINED_PARAMS['threshold_amr']
        best_cl_thr = CLASSICAL_PARAMS['threshold_amr']

        qa_samples, cl_samples = [], []
        qa_compute, cl_compute = [], []
        for seed in range(N_SEEDS):
            # Create a fresh sim with small perturbation for independence
            sim, Phi_prev = make_sim(N, init_method, n_steps)
            perturb_sim(sim, seed)
            gt = ground_truth_errors(sim, N, TARGET_DIM)

            comp = run_hierarchical_comparison(
                sim, N, Phi_prev=Phi_prev,
                threshold_qa=best_qa_thr, threshold_cl=best_cl_thr,
                target_dim=TARGET_DIM, min_size=MIN_SIZE, K_opt=40,
            )
            qa_m = patches_to_metrics(comp['qaoa_patches'], gt, N, TARGET_DIM)
            cl_m = patches_to_metrics(comp['classical_patches'], gt, N, TARGET_DIM)
            qa_samples.append(qa_m['captured_fraction'])
            cl_samples.append(cl_m['captured_fraction'])
            qa_compute.append(qa_m['compute_ratio'])
            cl_compute.append(cl_m['compute_ratio'])
            if seed == 0:
                print(f"  seed 0: QA={qa_m['captured_fraction']:.4f} "
                      f"CL={cl_m['captured_fraction']:.4f}")

        qa_arr, cl_arr = np.array(qa_samples), np.array(cl_samples)
        delta = qa_arr - cl_arr

        # Bootstrap CI
        rng_boot = np.random.default_rng(42)
        boot_delta = np.array([
            np.mean(delta[rng_boot.choice(N_SEEDS, size=N_SEEDS, replace=True)])
            for _ in range(N_BOOTSTRAP)
        ])
        ci = np.percentile(boot_delta, [2.5, 97.5])

        # Permutation test
        rng_perm = np.random.default_rng(123)
        p_val = permutation_pvalue(delta, N_PERMUTATIONS, rng_perm)

        # Cohen's d effect size
        d_std = np.std(delta, ddof=1)
        cohens_d = np.mean(delta) / d_std if d_std > 1e-12 else 0.0

        all_results[scen_name] = {
            'qa': qa_arr, 'cl': cl_arr, 'delta': delta,
            'ci': ci, 'p_val': p_val, 'cohens_d': cohens_d,
            'qa_compute': np.array(qa_compute), 'cl_compute': np.array(cl_compute),
        }
        print(f"  delta={np.mean(delta):+.4f} CI=[{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"p={p_val:.3f} d={cohens_d:+.2f}")

    # Save cache
    cache_data = {}
    for scen_name, data in all_results.items():
        cache_data[scen_name] = {
            'qa': data['qa'].tolist(),
            'cl': data['cl'].tolist(),
            'delta': data['delta'].tolist(),
            'ci': data['ci'].tolist(),
            'p_val': data['p_val'],
            'cohens_d': data['cohens_d'],
            'qa_compute': data['qa_compute'].tolist(),
            'cl_compute': data['cl_compute'].tolist(),
        }
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"\nCache saved to {CACHE_PATH}")

# ── Plot ──
SHORT_NAMES = {
    'Kelvin-Helmholtz': 'KH',
    'Harris Tearing': 'Tearing',
    'MHD Rotor': 'Rotor',
    'Orszag-Tang': 'OT',
}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 2.6),
                                     constrained_layout=True)
scen_names = [s[0] for s in scenarios]
short_labels = [SHORT_NAMES.get(s, s) for s in scen_names]
y_pos = np.arange(len(scen_names))

# Left: Forest plot (delta + CI)
means = [np.mean(all_results[s]['delta']) for s in scen_names]
ci_lo = [all_results[s]['ci'][0] for s in scen_names]
ci_hi = [all_results[s]['ci'][1] for s in scen_names]
p_vals = [all_results[s]['p_val'] for s in scen_names]
ds = [all_results[s]['cohens_d'] for s in scen_names]

colors = [COLORS['qaoa'] if lo > 0 else
          (COLORS['classical'] if hi < 0 else COLORS['tie'])
          for lo, hi in zip(ci_lo, ci_hi)]
xerr = [[m - lo for m, lo in zip(means, ci_lo)],
        [hi - m for m, hi in zip(means, ci_hi)]]

ax1.barh(y_pos, means, xerr=xerr, height=0.5, color=colors,
         capsize=3, alpha=0.85)
ax1.axvline(0, color='black', lw=1)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(short_labels)
ax1.set_xlabel(r'$\Delta$ captured fraction')
ax1.set_title('Forest Plot (95% CI)')
for i, (m, p, d) in enumerate(zip(means, p_vals, ds)):
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    ax1.text(max(ci_hi[i], 0) + 0.005, i,
             f'p={p:.3f}{sig}  d={d:+.2f}',
             va='center', fontsize=6)
ax1.legend(handles=[
    Patch(facecolor=COLORS['qaoa'], label='Q-HAS better'),
    Patch(facecolor=COLORS['classical'], label='Classical better'),
    Patch(facecolor=COLORS['tie'], label='Not significant'),
], loc='lower left', fontsize=6)

# Middle: Absolute values with individual data points
ax2.barh(y_pos - 0.2, [np.mean(all_results[s]['qa']) for s in scen_names],
         height=0.35, color=COLORS['qaoa'], label='Q-HAS', alpha=0.85)
ax2.barh(y_pos + 0.2, [np.mean(all_results[s]['cl']) for s in scen_names],
         height=0.35, color=COLORS['classical'], label='Classical', alpha=0.85)
for i, s in enumerate(scen_names):
    ax2.scatter(all_results[s]['qa'], [i - 0.2] * N_SEEDS,
                color=COLORS['qaoa'], s=10, alpha=0.4, zorder=5)
    ax2.scatter(all_results[s]['cl'], [i + 0.2] * N_SEEDS,
                color=COLORS['classical'], s=10, alpha=0.4, zorder=5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(short_labels)
ax2.set_xlabel('Captured fraction')
ax2.set_title('Absolute Performance')
ax2.legend(loc='lower left')

# Right: Compute ratio comparison
ax3.barh(y_pos - 0.2, [np.mean(all_results[s]['qa_compute']) for s in scen_names],
         height=0.35, color=COLORS['qaoa'], label='Q-HAS', alpha=0.85)
ax3.barh(y_pos + 0.2, [np.mean(all_results[s]['cl_compute']) for s in scen_names],
         height=0.35, color=COLORS['classical'], label='Classical', alpha=0.85)
for i, s in enumerate(scen_names):
    ax3.scatter(all_results[s]['qa_compute'], [i - 0.2] * N_SEEDS,
                color=COLORS['qaoa'], s=10, alpha=0.4, zorder=5)
    ax3.scatter(all_results[s]['cl_compute'], [i + 0.2] * N_SEEDS,
                color=COLORS['classical'], s=10, alpha=0.4, zorder=5)
ax3.axvline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5, label='Full DNS')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(short_labels)
ax3.set_xlabel('Compute ratio (pixels/$N^2$)')
ax3.set_title('Compute Cost')
ax3.legend(loc='lower left')

fig.suptitle('Statistical Validation', fontsize=11, fontweight='bold')
out = os.path.join(FIG_DIR, 'fig6_statistical_validation.png')
plt.savefig(out, dpi=300)
plt.savefig(out.replace('.png', '.pdf'))
print(f"\nSaved: {out}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
for s in scen_names:
    r = all_results[s]
    sig = '***' if r['p_val'] < 0.001 else ('**' if r['p_val'] < 0.01 else
          ('*' if r['p_val'] < 0.05 else 'n.s.'))
    print(f"  {s:25s}: delta={np.mean(r['delta']):+.4f} "
          f"CI=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
          f"p={r['p_val']:.3f} {sig} d={r['cohens_d']:+.2f}")
