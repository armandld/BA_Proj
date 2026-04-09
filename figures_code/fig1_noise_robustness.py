"""
Figure 1: Noise Robustness — Q-HAS vs Classical AMR (fair comparison).
Produces: figures/fig1_noise_robustness.png

FAIR PROTOCOL: Each method uses its own optimal threshold (determined
from clean-condition sweep). Then we test noise robustness at those
operating points by injecting Gaussian noise into the MHD fields.

The key hypothesis: Q-HAS's ZZ/ZZZZ spatial correlations in the
Hamiltonian act as a topological denoiser — correlated anomaly signals
survive while uncorrelated noise is suppressed.

Panels: 2 scenarios × 2 columns (captured fraction + compute ratio)
"""
import numpy as np
import copy
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fig_utils import (
    apply_style, COLORS, TRAINED_PARAMS, CLASSICAL_PARAMS, make_sim, ground_truth_errors,
    run_single_method, run_hierarchical_comparison, patches_to_metrics,
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
N_TRIALS = 5
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

scenarios = filter_scenarios_dict({
    'Kelvin-Helmholtz': ('init_kelvin_helmholtz',   400),
    'Harris Tearing':   ('init_harris_tearing',     300),
    'MHD Rotor':        ('init_mhd_rotor',          300),
    'Orszag-Tang':      ('init_orszag_tang',        500),
})

n_scen = len(scenarios)
if n_scen == 0:
    print("Aucun scénario pour cette phase.")
    sys.exit(0)


def inject_field_noise(sim, noise_std, rng):
    """Inject additive Gaussian noise into all 4 MHD fields.

    Noise is scaled relative to each field's RMS amplitude so that
    the noise level is physically meaningful (SNR-based).
    Returns a copy of the sim with noisy fields.
    """
    if noise_std <= 0:
        return sim

    for field_name in ['vx', 'vy', 'Bx', 'By']:
        field = getattr(sim, field_name)
        rms = max(np.std(field), 1e-10)
        noise = noise_std * rms * rng.standard_normal(field.shape)
        setattr(sim, field_name, field + noise)
    return sim


print("=" * 70)
print(f"Fig 1: Noise Robustness (N={N}, VQA={TARGET_DIM}x{TARGET_DIM}, 8 qubits)")
print(f"  solve_max_depth = {_compute_depths(N, TARGET_DIM, MIN_SIZE)}")
print(f"  Noise injected into MHD fields (relative to field RMS)")
print(f"  Finding optimal threshold for each method first...")
print("=" * 70)

fig, axes = plt.subplots(n_scen, 2, figsize=(12, 4.5 * n_scen))
if n_scen == 1:
    axes = axes[np.newaxis, :]

for scen_idx, (scen_name, (init_method, n_steps)) in enumerate(scenarios.items()):
    print(f"\n{'='*50}")
    print(f"--- {scen_name} ---")

    # STEP 1: Run clean simulation once for GT and threshold optimization
    sim_clean, Phi_prev_clean = make_sim(N, init_method, n_steps)
    gt_clean = ground_truth_errors(sim_clean, N, TARGET_DIM)

    best_qa_thr = TRAINED_PARAMS['threshold_amr']
    best_cl_thr = CLASSICAL_PARAMS['threshold_amr']

    # Step 2: Noise robustness at each method's optimal threshold
    qa_cap_means, qa_cap_stds = [], []
    cl_cap_means, cl_cap_stds = [], []
    qa_comp_means, qa_comp_stds = [], []
    cl_comp_means, cl_comp_stds = [], []

    for sigma in NOISE_LEVELS:
        qa_caps, cl_caps, qa_comps, cl_comps = [], [], [], []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial + int(sigma * 1000))

            # Create a fresh sim and inject noise into the FIELDS
            sim_noisy, Phi_prev_noisy = make_sim(N, init_method, n_steps)
            inject_field_noise(sim_noisy, sigma, rng)

            # GT is computed on CLEAN sim (we measure how well noisy
            # decisions capture clean ground-truth errors)
            comp = run_hierarchical_comparison(
                sim_noisy, N, Phi_prev=Phi_prev_noisy,
                threshold_qa=best_qa_thr,
                threshold_cl=best_cl_thr,
                target_dim=TARGET_DIM, min_size=MIN_SIZE,
                K_opt=40, verbose=False,
            )
            qa_m = patches_to_metrics(comp['qaoa_patches'], gt_clean, N, TARGET_DIM)
            cl_m = patches_to_metrics(comp['classical_patches'], gt_clean, N, TARGET_DIM)
            qa_caps.append(qa_m['captured_fraction'])
            cl_caps.append(cl_m['captured_fraction'])
            qa_comps.append(qa_m['compute_ratio'])
            cl_comps.append(cl_m['compute_ratio'])

        qa_cap_means.append(np.mean(qa_caps)); qa_cap_stds.append(np.std(qa_caps))
        cl_cap_means.append(np.mean(cl_caps)); cl_cap_stds.append(np.std(cl_caps))
        qa_comp_means.append(np.mean(qa_comps)); qa_comp_stds.append(np.std(qa_comps))
        cl_comp_means.append(np.mean(cl_comps)); cl_comp_stds.append(np.std(cl_comps))
        print(f"  sigma={sigma:.2f}: QA cap={qa_cap_means[-1]:.3f}±{qa_cap_stds[-1]:.3f}, "
              f"CL cap={cl_cap_means[-1]:.3f}±{cl_cap_stds[-1]:.3f} | "
              f"QA comp={qa_comp_means[-1]:.3f}, CL comp={cl_comp_means[-1]:.3f}")

    # --- Plotting ---
    ax = axes[scen_idx, 0]
    ax.errorbar(NOISE_LEVELS, cl_cap_means, yerr=cl_cap_stds, color=COLORS['classical'],
                marker='s', ms=6, capsize=3, lw=1.5,
                label=f'Classical (thr={best_cl_thr:.2f})')
    ax.errorbar(NOISE_LEVELS, qa_cap_means, yerr=qa_cap_stds, color=COLORS['qaoa'],
                marker='o', ms=6, capsize=3, lw=1.5,
                label=f'Q-HAS (thr={best_qa_thr:.2f})')
    ax.set_xlabel('Noise level (σ, relative to field RMS)')
    ax.set_ylabel('Captured Error Fraction')
    ax.set_title(f'{scen_name}: Error Capture under Noise')
    ax.legend(fontsize=8)
    # Auto-scale y-axis with some padding
    all_vals = qa_cap_means + cl_cap_means
    ymax = max(all_vals) * 1.15 if max(all_vals) > 0.01 else 1.0
    ax.set_ylim(0, min(ymax, 1.1))

    ax = axes[scen_idx, 1]
    ax.errorbar(NOISE_LEVELS, cl_comp_means, yerr=cl_comp_stds, color=COLORS['classical'],
                marker='s', ms=6, capsize=3, lw=1.5,
                label=f'Classical (thr={best_cl_thr:.2f})')
    ax.errorbar(NOISE_LEVELS, qa_comp_means, yerr=qa_comp_stds, color=COLORS['qaoa'],
                marker='o', ms=6, capsize=3, lw=1.5,
                label=f'Q-HAS (thr={best_qa_thr:.2f})')
    ax.axhline(y=1.0, color='gray', ls='--', alpha=0.5, label='Full DNS')
    ax.set_xlabel('Noise level (σ, relative to field RMS)')
    ax.set_ylabel('Compute Ratio')
    ax.set_title(f'{scen_name}: Compute Cost under Noise')
    ax.legend(fontsize=8)

fig.suptitle(f'Noise Robustness: Q-HAS vs Classical AMR (fair: separate thresholds)\n'
             f'(N={N}, {TARGET_DIM}x{TARGET_DIM} VQA, 8 qubits, {N_TRIALS} trials)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig1_noise_robustness.png')
plt.savefig(out, dpi=300); print(f"\nSaved: {out}")
