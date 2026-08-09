"""
Figure 10: Grid Scaling — ABANDONED.

This figure is not relevant for publication:
- N < 256 refines everything due to min_patch_size constraints
- N > 256 (512, 1024) would require more compute than available
- AMR decision time panel is meaningless on a simulator (not hardware)
- Only N=256 data point is meaningful, which is already covered by other figures

The script is kept for reference but skipped during generation.
"""
import sys
print("Fig 10: SKIPPED (not relevant for publication — see docstring)")
sys.exit(0)

# ── Original code below (kept for reference) ──
import time
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

apply_style()

TARGET_DIM = 2
MIN_SIZE = 6
N_TRIALS = 3
GRID_SIZES = [64, 128, 256]

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

print(f"Fig 10: Grid Scaling ({GRID_SIZES}, {n_scen} scenarios, {N_TRIALS} trials)")

# Storage: per grid size, aggregate over (scenarios x trials)
all_qa_cap = {N: [] for N in GRID_SIZES}
all_cl_cap = {N: [] for N in GRID_SIZES}
all_qa_comp = {N: [] for N in GRID_SIZES}
all_cl_comp = {N: [] for N in GRID_SIZES}
all_qa_time = {N: [] for N in GRID_SIZES}
all_cl_time = {N: [] for N in GRID_SIZES}

best_qa_thr = TRAINED_PARAMS['threshold_amr']
best_cl_thr = CLASSICAL_PARAMS['threshold_amr']

for grid_N in GRID_SIZES:
    depth = _compute_depths(grid_N, TARGET_DIM, MIN_SIZE)
    print(f"\n  N={grid_N}, max_depth={depth}")

    for name, init, base_steps in scenarios:
        # Scale timesteps proportionally to grid size
        n_steps = max(int(base_steps * grid_N / 256), 50)
        sim, Phi_prev = make_sim(grid_N, init, n_steps)
        gt = ground_truth_errors(sim, grid_N, TARGET_DIM)

        for trial in range(N_TRIALS):
            t0 = time.time()
            comp = run_hierarchical_comparison(
                sim, grid_N, Phi_prev=Phi_prev,
                threshold_qa=best_qa_thr, threshold_cl=best_cl_thr,
                target_dim=TARGET_DIM, min_size=MIN_SIZE, K_opt=40,
            )
            t_total = time.time() - t0

            qa_m = patches_to_metrics(comp['qaoa_patches'], gt, grid_N, TARGET_DIM)
            cl_m = patches_to_metrics(comp['classical_patches'], gt, grid_N, TARGET_DIM)

            all_qa_cap[grid_N].append(qa_m['captured_fraction'])
            all_cl_cap[grid_N].append(cl_m['captured_fraction'])
            all_qa_comp[grid_N].append(qa_m['compute_ratio'])
            all_cl_comp[grid_N].append(cl_m['compute_ratio'])
            # Total time includes both methods; approximate split
            all_qa_time[grid_N].append(t_total)
            all_cl_time[grid_N].append(t_total * 0.3)  # classical is ~30% of total

        print(f"    {name}: QA cap={np.mean(all_qa_cap[grid_N][-N_TRIALS:]):.3f} "
              f"CL cap={np.mean(all_cl_cap[grid_N][-N_TRIALS:]):.3f}")

# Now also time methods separately for accurate comparison
print("\n  Timing separate method calls...")
qa_times_separate = {N: [] for N in GRID_SIZES}
cl_times_separate = {N: [] for N in GRID_SIZES}
for grid_N in GRID_SIZES:
    # Use first scenario for timing
    name, init, base_steps = scenarios[0]
    n_steps = max(int(base_steps * grid_N / 256), 50)
    sim, Phi_prev = make_sim(grid_N, init, n_steps)
    from fig_utils import run_single_method
    for _ in range(3):
        t0 = time.time()
        run_single_method(sim, grid_N, method='qaoa', Phi_prev=Phi_prev,
                          threshold=best_qa_thr, target_dim=TARGET_DIM,
                          min_size=MIN_SIZE, K_opt=40)
        qa_times_separate[grid_N].append(time.time() - t0)

        t0 = time.time()
        run_single_method(sim, grid_N, method='classical', Phi_prev=None,
                          threshold=best_cl_thr, target_dim=TARGET_DIM,
                          min_size=MIN_SIZE)
        cl_times_separate[grid_N].append(time.time() - t0)

# ── Plotting ──
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
x = np.arange(len(GRID_SIZES))
w = 0.35

def _agg(d, Ns):
    """Mean and std for each grid size."""
    mu = [np.mean(d[n]) for n in Ns]
    std = [np.std(d[n]) for n in Ns]
    return mu, std

# Panel A: Captured fraction
ax = axes[0, 0]
qa_mu, qa_std = _agg(all_qa_cap, GRID_SIZES)
cl_mu, cl_std = _agg(all_cl_cap, GRID_SIZES)
ax.bar(x - w/2, qa_mu, w, yerr=qa_std, capsize=4,
       color=COLORS['qaoa'], label='Q-HAS', alpha=0.85)
ax.bar(x + w/2, cl_mu, w, yerr=cl_std, capsize=4,
       color=COLORS['classical'], label='Classical', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}\nd={_compute_depths(n,TARGET_DIM,MIN_SIZE)}'
                     for n in GRID_SIZES])
ax.set_ylabel('Captured Error Fraction')
ax.set_title('A) Error Capture vs Grid Size')
ax.legend()
ax.set_ylim(0, 1.15)

# Panel B: Compute ratio
ax = axes[0, 1]
qa_mu, qa_std = _agg(all_qa_comp, GRID_SIZES)
cl_mu, cl_std = _agg(all_cl_comp, GRID_SIZES)
ax.bar(x - w/2, qa_mu, w, yerr=qa_std, capsize=4,
       color=COLORS['qaoa'], label='Q-HAS', alpha=0.85)
ax.bar(x + w/2, cl_mu, w, yerr=cl_std, capsize=4,
       color=COLORS['classical'], label='Classical', alpha=0.85)
ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in GRID_SIZES])
ax.set_ylabel('Compute Ratio')
ax.set_title('B) Compute Cost vs Grid Size')
ax.legend()

# Panel C: Wall-clock time
ax = axes[1, 0]
qa_t_mu = [np.mean(qa_times_separate[n]) for n in GRID_SIZES]
qa_t_std = [np.std(qa_times_separate[n]) for n in GRID_SIZES]
cl_t_mu = [np.mean(cl_times_separate[n]) for n in GRID_SIZES]
cl_t_std = [np.std(cl_times_separate[n]) for n in GRID_SIZES]
ax.bar(x - w/2, qa_t_mu, w, yerr=qa_t_std, capsize=4,
       color=COLORS['qaoa'], label='Q-HAS', alpha=0.85)
ax.bar(x + w/2, cl_t_mu, w, yerr=cl_t_std, capsize=4,
       color=COLORS['classical'], label='Classical', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in GRID_SIZES])
ax.set_ylabel('Wall-clock time (s)')
ax.set_title('C) AMR Decision Time')
ax.legend()
# Annotate overhead ratio at each grid size
for i, n in enumerate(GRID_SIZES):
    if cl_t_mu[i] > 0.01:
        ratio = qa_t_mu[i] / cl_t_mu[i]
        ax.text(x[i], qa_t_mu[i] + qa_t_std[i] + 1, f'{ratio:.0f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')

# Panel D: Efficiency
ax = axes[1, 1]
qa_eff_mu = [np.mean(np.array(all_qa_cap[n]) / np.maximum(np.array(all_qa_comp[n]), 1e-6))
             for n in GRID_SIZES]
qa_eff_std = [np.std(np.array(all_qa_cap[n]) / np.maximum(np.array(all_qa_comp[n]), 1e-6))
              for n in GRID_SIZES]
cl_eff_mu = [np.mean(np.array(all_cl_cap[n]) / np.maximum(np.array(all_cl_comp[n]), 1e-6))
             for n in GRID_SIZES]
cl_eff_std = [np.std(np.array(all_cl_cap[n]) / np.maximum(np.array(all_cl_comp[n]), 1e-6))
              for n in GRID_SIZES]
ax.bar(x - w/2, qa_eff_mu, w, yerr=qa_eff_std, capsize=4,
       color=COLORS['qaoa'], label='Q-HAS', alpha=0.85)
ax.bar(x + w/2, cl_eff_mu, w, yerr=cl_eff_std, capsize=4,
       color=COLORS['classical'], label='Classical', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'N={n}' for n in GRID_SIZES])
ax.set_ylabel('Efficiency (captured/compute)')
ax.set_title('D) AMR Efficiency')
ax.legend()

fig.suptitle(f'Grid Scaling: Q-HAS vs Classical ({n_scen} scenarios, '
             f'{N_TRIALS} trials, 8 qubits)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out = os.path.join(FIG_DIR, 'fig10_grid_scaling.png')
plt.savefig(out, dpi=300)
print(f"\nSaved: {out}")
