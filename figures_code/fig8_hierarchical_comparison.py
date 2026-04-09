"""Fig 8 — Hierarchical AMR Comparison: Q-HAS vs Classical
==========================================================
Compares run_adaptive_vqa (quantum) vs run_adaptive_classical on real
MHD grid (256×256) using 2×2 VQA patches (8 qubits).

This is the first figure to test the *hierarchical* AMR approach
rather than flat block selection. Both methods use the same BFS
tree structure — only the decision engine differs.

Panels:
  A) Per-scenario bar chart: captured fraction (Q-HAS vs Classical)
  B) Patch depth distribution comparison
  C) Compute ratio (effective pixels / N²)
  D) Physical fidelity after time evolution with step_layered
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fig_utils import (
    apply_style, COLORS, TRAINED_PARAMS, CLASSICAL_PARAMS,
    make_sim, ground_truth_errors,
    run_single_method, run_hierarchical_comparison, patches_to_metrics,
    print_patch_summary, _compute_depths,
    compute_kinetic_energy, compute_magnetic_energy,
    compute_enstrophy, field_l2_error,
    filter_scenarios_dict,
    FIG_DIR,
)
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

apply_style()

# ── Configuration ──
N = 256
TARGET_DIM = 2  # 2×2 VQA grid → 8 qubits
MIN_SIZE = 6
N_TRIALS = 2       # 2 trials for error bars on panels A-C
# Shortened warmup to stay within Colab timeout
SCENARIOS = filter_scenarios_dict({
    'init_kelvin_helmholtz':   {'label': 'Kelvin-Helmholtz', 'n_steps': 100},
    'init_harris_tearing':     {'label': 'Harris Tearing',   'n_steps': 80},
    'init_mhd_rotor':          {'label': 'MHD Rotor',        'n_steps': 80},
    'init_orszag_tang':        {'label': 'Orszag-Tang',      'n_steps': 120},
})

n_scen = len(SCENARIOS)
if n_scen == 0:
    print("Aucun scénario pour cette phase.")
    sys.exit(0)

print("=" * 70)
print(f"Fig 8: Hierarchical AMR Comparison (N={N}, VQA={TARGET_DIM}×{TARGET_DIM})")
print(f"  solve_max_depth = {_compute_depths(N, TARGET_DIM, MIN_SIZE)}")
print(f"  n_trials = {N_TRIALS}")
print("=" * 70)

# ── Data caching ──
CACHE_PATH = os.path.join(FIG_DIR, '.fig8_cache.json')
use_cache = os.path.exists(CACHE_PATH) and '--recompute' not in sys.argv

if use_cache:
    print("Loaded from cache — replotting only. Use --recompute to force.")
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    results = {}
    for label, trials in cache['results'].items():
        results[label] = []
        for t in trials:
            t['qaoa_depths'] = {int(k): v for k, v in t['qaoa_depths'].items()}
            t['cl_depths'] = {int(k): v for k, v in t['cl_depths'].items()}
            results[label].append(t)
    fidelity_results = cache['fidelity_results']
else:
    # ── Run comparisons ──
    results = {}  # scenario → list of dicts per trial

    for scenario_init, cfg in SCENARIOS.items():
        print(f"\n{'─'*50}")
        print(f"Scenario: {cfg['label']} ({scenario_init})")
        print(f"{'─'*50}")
        scenario_results = []

        # Find optimal thresholds via fine-grained search
        sim0, Phi0 = make_sim(N, scenario_init, cfg['n_steps'])
        gt0 = ground_truth_errors(sim0, N, TARGET_DIM)
        best_qa_thr = TRAINED_PARAMS['threshold_amr']
        best_cl_thr = CLASSICAL_PARAMS['threshold_amr']

        for trial in range(N_TRIALS):
            print(f"\n  Trial {trial+1}/{N_TRIALS}")
            sim, Phi_prev = make_sim(N, scenario_init, cfg['n_steps'])
            gt = ground_truth_errors(sim, N, TARGET_DIM)
            result = run_hierarchical_comparison(
                sim, N, Phi_prev=Phi_prev,
                threshold_qa=best_qa_thr, threshold_cl=best_cl_thr,
                target_dim=TARGET_DIM,
                min_size=MIN_SIZE,
                K_opt=40,
                verbose=True,
            )

            qaoa_m = patches_to_metrics(result['qaoa_patches'], gt, N, TARGET_DIM)
            cl_m = patches_to_metrics(result['classical_patches'], gt, N, TARGET_DIM)

            print_patch_summary("Q-HAS", result['qaoa_patches'], gt, N, TARGET_DIM)
            print_patch_summary("Classical", result['classical_patches'], gt, N, TARGET_DIM)

            # Depth distribution
            qaoa_depths = {}
            for p in result['qaoa_patches']:
                d = p['depth']
                qaoa_depths[d] = qaoa_depths.get(d, 0) + 1
            cl_depths = {}
            for p in result['classical_patches']:
                d = p['depth']
                cl_depths[d] = cl_depths.get(d, 0) + 1

            scenario_results.append({
                'qaoa_captured': qaoa_m['captured_fraction'],
                'cl_captured': cl_m['captured_fraction'],
                'qaoa_compute': qaoa_m['compute_ratio'],
                'cl_compute': cl_m['compute_ratio'],
                'qaoa_n_fine': qaoa_m['n_fine'],
                'cl_n_fine': cl_m['n_fine'],
                'qaoa_n_total': qaoa_m['n_total'],
                'cl_n_total': cl_m['n_total'],
                'qaoa_depths': qaoa_depths,
                'cl_depths': cl_depths,
            })

        results[cfg['label']] = scenario_results

    # ── Panel D: Physical fidelity via step_layered ──
    print(f"\n{'='*50}")
    print("Panel D: Physical Fidelity (step_layered evolution)")
    print(f"{'='*50}")

    fidelity_results = {}
    N_FIDELITY = 256
    N_EVOLVE_STEPS = 10  # number of MHD steps with AMR

    for scenario_init, cfg in SCENARIOS.items():
        label = cfg['label']
        print(f"\n  {label}: evolving {N_EVOLVE_STEPS} steps with AMR...")

        # Create 3 copies: DNS reference, Q-HAS, Classical
        grid_ref = PeriodicGrid(resolution_N=N_FIDELITY)
        sim_ref = MHDSolver(grid_ref, dt=1e-3, Re=800, Rm=800)
        getattr(sim_ref, scenario_init)()

        grid_q = PeriodicGrid(resolution_N=N_FIDELITY)
        sim_q = MHDSolver(grid_q, dt=1e-3, Re=800, Rm=800)
        getattr(sim_q, scenario_init)()

        grid_c = PeriodicGrid(resolution_N=N_FIDELITY)
        sim_c = MHDSolver(grid_c, dt=1e-3, Re=800, Rm=800)
        getattr(sim_c, scenario_init)()

        # Warmup (all identical)
        warmup = cfg['n_steps']
        from Simulation.PhysToAngle import AngleMapper as AM
        _m = AM(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
        Phi_prev = None
        for i in range(warmup):
            if i == warmup - 1:
                Phi_prev = _m.compute_stress_flux(sim_ref.get_fluxes())
            dt = sim_ref.adapt_dt(cfl_target=0.4)
            sim_ref.step_full(record_stats=False)
            sim_q.dt = dt
            sim_q.step_full(record_stats=False)
            sim_c.dt = dt
            sim_c.step_full(record_stats=False)

        # Find optimal thresholds for fidelity
        gt_fid = ground_truth_errors(sim_ref, N_FIDELITY, TARGET_DIM)
        best_qa_thr = TRAINED_PARAMS['threshold_amr']
        best_cl_thr = CLASSICAL_PARAMS['threshold_amr']

        # Now evolve with AMR — FAIR: each method sees its OWN state
        solve_md = _compute_depths(N_FIDELITY, TARGET_DIM, MIN_SIZE)
        Phi_prev_qa = Phi_prev
        for step in range(N_EVOLVE_STEPS):
            # Each method gets patches from its own simulation state
            qa_patches, Phi_new = run_single_method(
                sim_q, N_FIDELITY, method='qaoa', Phi_prev=Phi_prev_qa,
                threshold=best_qa_thr, target_dim=TARGET_DIM, min_size=MIN_SIZE, K_opt=40,
            )
            cl_patches, _ = run_single_method(
                sim_c, N_FIDELITY, method='classical', Phi_prev=None,
                threshold=best_cl_thr, target_dim=TARGET_DIM, min_size=MIN_SIZE,
            )
            Phi_prev_qa = Phi_new

            dt = sim_ref.adapt_dt(cfl_target=0.4)
            sim_ref.step_full(record_stats=False)

            sim_q.dt = dt
            sim_q.tau_buffer = {}
            sim_q.step_layered(qa_patches, max_depth=solve_md, target_dim=TARGET_DIM)

            sim_c.dt = dt
            sim_c.tau_buffer = {}
            sim_c.step_layered(cl_patches, max_depth=solve_md, target_dim=TARGET_DIM)

        l2_qaoa = field_l2_error(sim_q, sim_ref)
        l2_cl = field_l2_error(sim_c, sim_ref)
        print(f"    L2 error: Q-HAS={l2_qaoa:.6f}, Classical={l2_cl:.6f}")
        fidelity_results[label] = {'qaoa_l2': l2_qaoa, 'cl_l2': l2_cl}

    # ── Save cache ──
    cache_data = {
        'results': {},
        'fidelity_results': fidelity_results,
    }
    for label, trials in results.items():
        cache_data['results'][label] = []
        for t in trials:
            t_copy = dict(t)
            t_copy['qaoa_depths'] = {str(k): v for k, v in t['qaoa_depths'].items()}
            t_copy['cl_depths'] = {str(k): v for k, v in t['cl_depths'].items()}
            cache_data['results'][label].append(t_copy)
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache_data, f)
    print(f"Cache saved → {CACHE_PATH}")


# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

SHORT_LABELS = {
    'Kelvin-Helmholtz': 'KH',
    'Harris Tearing': 'Tearing',
    'MHD Rotor': 'Rotor',
    'Orszag-Tang': 'OT',
}

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
scenarios = list(results.keys())
short_scenarios = [SHORT_LABELS.get(s, s) for s in scenarios]

# Panel A: Captured fraction bar chart
ax = axes[0, 0]
x = np.arange(len(scenarios))
w = 0.35
qaoa_caps = [np.mean([r['qaoa_captured'] for r in results[s]]) for s in scenarios]
cl_caps = [np.mean([r['cl_captured'] for r in results[s]]) for s in scenarios]
qaoa_err = [np.std([r['qaoa_captured'] for r in results[s]]) for s in scenarios]
cl_err = [np.std([r['cl_captured'] for r in results[s]]) for s in scenarios]

bars_q = ax.bar(x - w/2, qaoa_caps, w, yerr=qaoa_err, label='Q-HAS',
                color=COLORS['qaoa'], capsize=3, alpha=0.85)
bars_c = ax.bar(x + w/2, cl_caps, w, yerr=cl_err, label='Classical',
                color=COLORS['classical'], capsize=3, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(short_scenarios, fontsize=8)
ax.set_ylabel('Captured Error Fraction')
ax.set_title('A) Captured Error', fontsize=9)
ax.legend(fontsize=7, framealpha=0.7)
ax.set_ylim(0, 1.1)

# Panel B: Number of fine patches
ax = axes[0, 1]
qaoa_fine = [np.mean([r['qaoa_n_fine'] for r in results[s]]) for s in scenarios]
cl_fine = [np.mean([r['cl_n_fine'] for r in results[s]]) for s in scenarios]
qaoa_total = [np.mean([r['qaoa_n_total'] for r in results[s]]) for s in scenarios]
cl_total = [np.mean([r['cl_n_total'] for r in results[s]]) for s in scenarios]

bars_q = ax.bar(x - w/2, qaoa_fine, w, label='Q-HAS fine',
                color=COLORS['qaoa'], alpha=0.85)
ax.bar(x - w/2, [t - f for t, f in zip(qaoa_total, qaoa_fine)], w,
       bottom=qaoa_fine, color=COLORS['qaoa'], alpha=0.3, label='Q-HAS coarse')
bars_c = ax.bar(x + w/2, cl_fine, w, label='Classical fine',
                color=COLORS['classical'], alpha=0.85)
ax.bar(x + w/2, [t - f for t, f in zip(cl_total, cl_fine)], w,
       bottom=cl_fine, color=COLORS['classical'], alpha=0.3, label='Classical coarse')
ax.set_xticks(x)
ax.set_xticklabels(short_scenarios, fontsize=8)
ax.set_ylabel('Number of Patches')
ax.set_title('B) Patch Count', fontsize=9)
ax.legend(fontsize=7, framealpha=0.7)

# Panel C: Compute ratio
ax = axes[1, 0]
qaoa_comp = [np.mean([r['qaoa_compute'] for r in results[s]]) for s in scenarios]
cl_comp = [np.mean([r['cl_compute'] for r in results[s]]) for s in scenarios]

ax.bar(x - w/2, qaoa_comp, w, label='Q-HAS', color=COLORS['qaoa'], alpha=0.85)
ax.bar(x + w/2, cl_comp, w, label='Classical', color=COLORS['classical'], alpha=0.85)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Full DNS')
ax.set_xticks(x)
ax.set_xticklabels(short_scenarios, fontsize=8)
ax.set_ylabel('Compute Ratio (pixels/N²)')
ax.set_title('C) Compute Cost', fontsize=9)
ax.legend(fontsize=7, framealpha=0.7)

# Panel D: Physical fidelity (L2 error after evolution)
ax = axes[1, 1]
fid_scenarios = list(fidelity_results.keys())
x_fid = np.arange(len(fid_scenarios))
qaoa_l2 = [fidelity_results[s]['qaoa_l2'] for s in fid_scenarios]
cl_l2 = [fidelity_results[s]['cl_l2'] for s in fid_scenarios]

short_fid = [SHORT_LABELS.get(s, s) for s in fid_scenarios]
ax.bar(x_fid - w/2, qaoa_l2, w, label='Q-HAS', color=COLORS['qaoa'], alpha=0.85)
ax.bar(x_fid + w/2, cl_l2, w, label='Classical', color=COLORS['classical'], alpha=0.85)
ax.set_xticks(x_fid)
ax.set_xticklabels(short_fid, fontsize=8)
ax.set_ylabel('Relative L2 Error vs DNS')
ax.set_title('D) Physical Fidelity', fontsize=9)
ax.legend(fontsize=7, framealpha=0.7)

fig.suptitle('Hierarchical AMR Overview',
             fontsize=11, fontweight='bold')
fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.35, wspace=0.30)
out = os.path.join(FIG_DIR, 'fig8_hierarchical_comparison.png')
plt.savefig(out, dpi=300)
print(f"\nSaved → {out}")

# ── Print summary ──
print("\n" + "=" * 70)
print("SUMMARY: Hierarchical AMR Comparison")
print("=" * 70)
for s in scenarios:
    trials = results[s]
    qa_mean = np.mean([r['qaoa_captured'] for r in trials])
    cl_mean = np.mean([r['cl_captured'] for r in trials])
    delta = qa_mean - cl_mean
    winner = "Q-HAS" if delta > 0.005 else ("Classical" if delta < -0.005 else "Tie")
    print(f"  {s:20s}: Q-HAS={qa_mean:.3f}  Classical={cl_mean:.3f}  "
          f"Δ={delta:+.3f}  → {winner}")

print("\nPhysical fidelity (L2 vs DNS):")
for s in fid_scenarios:
    qa = fidelity_results[s]['qaoa_l2']
    cl = fidelity_results[s]['cl_l2']
    winner = "Q-HAS" if qa < cl else "Classical"
    print(f"  {s:20s}: Q-HAS={qa:.6f}  Classical={cl:.6f}  → {winner}")
