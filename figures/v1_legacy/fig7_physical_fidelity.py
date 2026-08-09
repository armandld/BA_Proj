"""
Figure 7: Physical Fidelity — Simulation quality with hierarchical AMR.
Produces: figures/fig7_physical_fidelity.png

THE key figure: evolves MHD with step_layered using Q-HAS vs Classical
AMR patches and measures L2 error vs DNS, kinetic energy conservation,
and enstrophy conservation.

Each method sees its OWN simulation state for patch decisions.
Multiple trials with error bands for statistical confidence.
"""
import numpy as np
import os, sys, json
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
    apply_style, COLORS, TRAINED_PARAMS, CLASSICAL_PARAMS, ground_truth_errors,
    _compute_depths, run_single_method, patches_to_metrics,
    compute_kinetic_energy, compute_enstrophy, field_l2_error,
    filter_scenarios_dict,
    FIG_DIR,
)
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

apply_style()

N = 256
TARGET_DIM = 2
MIN_SIZE = 6
N_EVOLVE = 15      # enough points for a meaningful curve
N_TRIALS = 1

# Shortened warmup (100/80/80/120) to stay within Colab timeout.
# The instabilities are already visible at these shorter times.
scenarios = filter_scenarios_dict({
    'Kelvin-Helmholtz': {'init': 'init_kelvin_helmholtz',   'n_steps': 100},
    'Harris Tearing':   {'init': 'init_harris_tearing',     'n_steps': 80},
    'MHD Rotor':        {'init': 'init_mhd_rotor',          'n_steps': 80},
    'Orszag-Tang':      {'init': 'init_orszag_tang',        'n_steps': 120},
})

n_scen = len(scenarios)
if n_scen == 0:
    print("Aucun scenario pour cette phase.")
    sys.exit(0)

solve_md = _compute_depths(N, TARGET_DIM, MIN_SIZE)
print(f"Fig 7: Physical Fidelity (N={N}, {N_EVOLVE} AMR steps, "
      f"{N_TRIALS} trials, depth={solve_md})")


SHORT_NAMES = {
    'Kelvin-Helmholtz': 'KH',
    'Harris Tearing': 'Tearing',
    'MHD Rotor': 'Rotor',
    'Orszag-Tang': 'OT',
}


def _plot_with_band(ax, x, all_curves, color, label, ls='-', lw=0.9, zorder=2):
    """Plot mean line with std band from [N_TRIALS, n_steps] array."""
    mu = np.mean(all_curves, axis=0)
    std = np.std(all_curves, axis=0)
    ax.plot(x, mu, color=color, lw=lw, label=label, ls=ls, markersize=3, zorder=zorder)
    ax.fill_between(x, mu - std, mu + std, color=color, alpha=0.12, zorder=zorder)


CACHE_PATH = os.path.join(FIG_DIR, '.fig7_cache.json')
use_cache = os.path.exists(CACHE_PATH) and '--recompute' not in sys.argv

if use_cache:
    print("Loaded from cache — replotting only. Use --recompute to force.")
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    # reconstruct cached_data dict with numpy arrays
    cached_data = {}
    for scen_name, d in cache.items():
        cached_data[scen_name] = {k: np.array(v) for k, v in d.items()}
else:
    cached_data = {}
    for row, (scen_name, cfg) in enumerate(scenarios.items()):
        init_method = cfg['init']
        warmup_steps = cfg['n_steps']
        print(f"\n--- {scen_name} (warmup={warmup_steps}) ---")

        best_qa_thr = TRAINED_PARAMS['threshold_amr']
        best_cl_thr = CLASSICAL_PARAMS['threshold_amr']

        # Storage: [N_TRIALS, N_EVOLVE]
        l2_qa_all = np.zeros((N_TRIALS, N_EVOLVE))
        l2_cl_all = np.zeros((N_TRIALS, N_EVOLVE))
        ek_dns_all = np.zeros((N_TRIALS, N_EVOLVE))
        ek_qa_all = np.zeros((N_TRIALS, N_EVOLVE))
        ek_cl_all = np.zeros((N_TRIALS, N_EVOLVE))
        ens_dns_all = np.zeros((N_TRIALS, N_EVOLVE))
        ens_qa_all = np.zeros((N_TRIALS, N_EVOLVE))
        ens_cl_all = np.zeros((N_TRIALS, N_EVOLVE))

        for trial in range(N_TRIALS):
            print(f"  Trial {trial+1}/{N_TRIALS}")

            # Create 3 identical sims
            sims = {}
            for lbl in ['dns', 'qaoa', 'classical']:
                g = PeriodicGrid(resolution_N=N)
                s = MHDSolver(g, dt=1e-3, Re=800, Rm=800)
                getattr(s, init_method)()
                sims[lbl] = s

            # Add tiny perturbation for trial independence
            if trial > 0:
                rng = np.random.default_rng(trial)
                for lbl in sims:
                    for fn in ['vx', 'vy', 'Bx', 'By']:
                        f = getattr(sims[lbl], fn)
                        rms = max(np.std(f), 1e-10)
                        setattr(sims[lbl], fn,
                                f + 1e-5 * rms * rng.standard_normal(f.shape))

            # Warmup — all sims evolve identically
            _m = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
            Phi_prev_qa = None
            for i in range(warmup_steps):
                if i == warmup_steps - 1:
                    Phi_prev_qa = _m.compute_stress_flux(sims['dns'].get_fluxes())
                dt = sims['dns'].adapt_dt(cfl_target=0.4)
                for s in sims.values():
                    s.dt = dt
                    s.step_full(record_stats=False)

            # AMR evolution
            nan_count = {'qaoa': 0, 'classical': 0}
            for step in range(N_EVOLVE):
                if step % 50 == 0:
                    print(f"    step {step}/{N_EVOLVE}")

                qa_patches, Phi_qa_new = run_single_method(
                    sims['qaoa'], N, method='qaoa', Phi_prev=Phi_prev_qa,
                    threshold=best_qa_thr, target_dim=TARGET_DIM,
                    min_size=MIN_SIZE, K_opt=40,
                )
                cl_patches, _ = run_single_method(
                    sims['classical'], N, method='classical', Phi_prev=None,
                    threshold=best_cl_thr, target_dim=TARGET_DIM,
                    min_size=MIN_SIZE,
                )
                Phi_prev_qa = Phi_qa_new

                dt = sims['dns'].adapt_dt(cfl_target=0.3)
                sims['dns'].step_full(record_stats=False)

                for lbl, patches in [('qaoa', qa_patches), ('classical', cl_patches)]:
                    s = sims[lbl]
                    s.dt = dt
                    s.tau_buffer = {}
                    s.step_layered(patches, max_depth=solve_md, target_dim=TARGET_DIM)
                    if np.any(np.isnan(s.vx)) or np.any(np.isnan(s.vy)):
                        nan_count[lbl] += 1
                        s.vx = sims['dns'].vx.copy()
                        s.vy = sims['dns'].vy.copy()
                        s.Bx = sims['dns'].Bx.copy()
                        s.By = sims['dns'].By.copy()

                l2_qa_all[trial, step] = field_l2_error(sims['qaoa'], sims['dns'])
                l2_cl_all[trial, step] = field_l2_error(sims['classical'], sims['dns'])
                ek_dns_all[trial, step] = compute_kinetic_energy(sims['dns'])
                ek_qa_all[trial, step] = compute_kinetic_energy(sims['qaoa'])
                ek_cl_all[trial, step] = compute_kinetic_energy(sims['classical'])
                ens_dns_all[trial, step] = compute_enstrophy(sims['dns'])
                ens_qa_all[trial, step] = compute_enstrophy(sims['qaoa'])
                ens_cl_all[trial, step] = compute_enstrophy(sims['classical'])

            if any(v > 0 for v in nan_count.values()):
                print(f"  WARNING NaN resets (trial {trial+1}): {nan_count}")

        cached_data[scen_name] = {
            'l2_qa': l2_qa_all.tolist(), 'l2_cl': l2_cl_all.tolist(),
            'ek_dns': ek_dns_all.tolist(), 'ek_qa': ek_qa_all.tolist(), 'ek_cl': ek_cl_all.tolist(),
            'ens_dns': ens_dns_all.tolist(), 'ens_qa': ens_qa_all.tolist(), 'ens_cl': ens_cl_all.tolist(),
        }

    # save cache
    with open(CACHE_PATH, 'w') as f:
        json.dump({k: {kk: v.tolist() if hasattr(v, 'tolist') else v
                       for kk, v in d.items()}
                   for k, d in cached_data.items()}, f)

# ---------- Plotting (works with both cached and fresh data) ----------
fig, axes = plt.subplots(n_scen, 3, figsize=(10, 2.4 * n_scen))
if n_scen == 1:
    axes = axes[np.newaxis, :]

for row, scen_name in enumerate(scenarios):
    d = cached_data[scen_name]
    l2_qa_all  = np.asarray(d['l2_qa'])
    l2_cl_all  = np.asarray(d['l2_cl'])
    ek_dns_all = np.asarray(d['ek_dns'])
    ek_qa_all  = np.asarray(d['ek_qa'])
    ek_cl_all  = np.asarray(d['ek_cl'])
    ens_dns_all = np.asarray(d['ens_dns'])
    ens_qa_all  = np.asarray(d['ens_qa'])
    ens_cl_all  = np.asarray(d['ens_cl'])

    steps_arr = np.arange(l2_qa_all.shape[1])
    short = SHORT_NAMES.get(scen_name, scen_name)

    # Col 0: L2 error vs DNS
    ax = axes[row, 0]
    _plot_with_band(ax, steps_arr, l2_cl_all, COLORS['classical'], 'Classical AMR', ls='-.', lw=1.2, zorder=4)
    _plot_with_band(ax, steps_arr, l2_qa_all, COLORS['qaoa'], 'Q-HAS', lw=1.0, zorder=3)
    ax.set_xlabel('AMR Step')
    ax.set_ylabel('Rel. L2 Error')
    ax.set_title(f'{short}: L2 Error', fontsize=9)
    ax.legend(fontsize=7, loc='best', framealpha=0.7)
    ax.set_yscale('log')

    # Col 1: Kinetic energy conservation
    ax = axes[row, 1]
    _plot_with_band(ax, steps_arr, ek_dns_all, COLORS['dns'], 'DNS', ls='--', lw=0.9, zorder=2)
    _plot_with_band(ax, steps_arr, ek_qa_all, COLORS['qaoa'], 'Q-HAS', lw=1.0, zorder=3)
    _plot_with_band(ax, steps_arr, ek_cl_all, COLORS['classical'], 'Classical AMR', ls='-.', lw=1.2, zorder=4)
    ax.set_xlabel('AMR Step')
    ax.set_ylabel('Kinetic Energy')
    ax.set_title(f'{short}: Energy', fontsize=9)
    ax.legend(fontsize=7, loc='best', framealpha=0.7)

    # Col 2: Enstrophy conservation
    ax = axes[row, 2]
    _plot_with_band(ax, steps_arr, ens_dns_all, COLORS['dns'], 'DNS', ls='--', lw=0.9, zorder=2)
    _plot_with_band(ax, steps_arr, ens_qa_all, COLORS['qaoa'], 'Q-HAS', lw=1.0, zorder=3)
    _plot_with_band(ax, steps_arr, ens_cl_all, COLORS['classical'], 'Classical AMR', ls='-.', lw=1.2, zorder=4)
    ax.set_xlabel('AMR Step')
    ax.set_ylabel('Enstrophy')
    ax.set_title(f'{short}: Enstrophy', fontsize=9)
    ax.legend(fontsize=7, loc='best', framealpha=0.7)

    # Print scenario summary
    print(f"  Final L2: QA={l2_qa_all[:, -1].mean():.6f}+/-{l2_qa_all[:, -1].std():.6f}  "
          f"CL={l2_cl_all[:, -1].mean():.6f}+/-{l2_cl_all[:, -1].std():.6f}")

fig.suptitle('Physical Fidelity: AMR Evolution',
             fontsize=11, fontweight='bold')
fig.subplots_adjust(top=0.92, bottom=0.07, hspace=0.38, wspace=0.30)
out = os.path.join(FIG_DIR, 'fig7_physical_fidelity.png')
plt.savefig(out, dpi=300)
print(f"\nSaved: {out}")
