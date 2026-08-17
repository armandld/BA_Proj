#!/usr/bin/env python3
"""
Fig 11 — Hamiltonian Design Visualization

Demonstrates the uncertainty-weighted ZZ coupling on a real MHD snapshot.
Shows how sigma concentrates coupling near the decision boundary.

Panels (per scenario, 1 row):
  A) Classical score map with threshold contour
  B) ZZ coupling magnitude WITHOUT uncertainty weighting (sigma→∞)
  C) ZZ coupling magnitude WITH trained sigma
  D) Uncertainty weight map: exp(-((score-thr)/sigma)^2)
  E) Z-bias map: alpha_z * (score - threshold)
"""
import sys, os
import numpy as np

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
    make_sim, _hamilt_mapper_kwargs, filter_scenarios_dict,
)
from Simulation.HamiltParams import PhysicalMapper
from Simulation.PhysToAngle import AngleMapper

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

apply_style()

# ── Configuration ──
N = 256
SCENARIOS = filter_scenarios_dict({
    'Kelvin-Helmholtz':   {'init': 'init_kelvin_helmholtz',   'n_steps': 400},
    'Harris Tearing':     {'init': 'init_harris_tearing',     'n_steps': 300},
    'MHD Rotor':          {'init': 'init_mhd_rotor',          'n_steps': 300},
    'Orszag-Tang':        {'init': 'init_orszag_tang',        'n_steps': 500},
})

sigma_trained = TRAINED_PARAMS.get('sigma', 0.05)
threshold = TRAINED_PARAMS['threshold_amr']


def compute_zz_maps(sim, score, threshold, sigma_val):
    """Compute ZZ coupling magnitudes for given sigma."""
    grid = sim.grid
    hm_kwargs = _hamilt_mapper_kwargs(grid)
    hm_kwargs['sigma'] = sigma_val
    hm = PhysicalMapper(**hm_kwargs)

    state = sim.get_fluxes()
    result = hm.compute_coefficients(
        sim, score, state, threshold,
        advanced_anomalies_enabled=True,
    )
    C_h, C_v = result['C_edges']
    H_h, H_v = result['H_edges']
    K = result['K_plaquettes']

    zz_mag = np.sqrt(C_h**2 + C_v**2)
    z_bias = 0.5 * (H_h + H_v)  # average of horiz/vert
    return zz_mag, z_bias, C_h, C_v, K


def main():
    n_scenarios = len(SCENARIOS)
    if n_scenarios == 0:
        print("No scenarios for this phase — skipping fig11")
        return

    SHORT = {'Kelvin-Helmholtz': 'KH', 'Harris Tearing': 'Tearing',
             'MHD Rotor': 'Rotor', 'Orszag-Tang': 'OT'}

    # 4 panels: Score, |ZZ| always-on, |ZZ| weighted, Uncertainty weight
    # Removed Z-bias panel (nearly zero for all scenarios due to sigma suppression)
    fig, axes = plt.subplots(n_scenarios, 4, figsize=(14, 3.2 * n_scenarios),
                              squeeze=False)

    for row, (label, cfg) in enumerate(SCENARIOS.items()):
        sn = SHORT.get(label, label)
        print(f"  [{label}] Running simulation...")
        sim, Phi_prev = make_sim(N, cfg['init'], cfg['n_steps'])
        state = sim.get_fluxes()
        score = AngleMapper.classical_score(state)

        print(f"  [{label}] Computing Hamiltonian coefficients...")
        zz_trained, z_bias, _, _, _ = compute_zz_maps(sim, score, threshold, sigma_trained)
        zz_alwayson, _, _, _, _     = compute_zz_maps(sim, score, threshold, 100.0)

        # D-100 — DEVIATION MESUREE, NON CORRIGEE (decision en attente, voir
        # docs/DEFAUTS.md). Ce poids est recalcule ICI a partir du score
        # PAR CELLULE. Ce n'est pas celui que le hamiltonien applique :
        # `HamiltParams.py:469-473` le calcule sur le score moyenne PAR ARETE
        # (`0.5 * (s + roll(s, -1, axis))`), et en produit DEUX champs
        # distincts, horizontal et vertical. Mesure (N=64, sigma=0,0500,
        # threshold=0,3044), part des cellules a w > 0,1 :
        #   Kelvin-Helmholtz  panneau 9,89 %  |  aretes h 10,40 % / v 9,91 %
        #   Harris Tearing    panneau 1,27 %  |  aretes h  5,52 % / v 1,27 %
        # Sur la nappe de tearing, les aretes horizontales sont donc 4,3x plus
        # actives que ce que le panneau affiche, et l'anisotropie que le
        # hamiltonien voit (h != v) n'apparait pas du tout ici.
        # Ne PAS corriger sans trancher : afficher w_h, w_v, leur moyenne ou
        # leur max est un choix de presentation, pas une correction mecanique.
        # (Le nombre « ZZ reduced by X% » du panneau C, lui, vient bien du
        # mappeur reel via compute_zz_maps : il n'est pas concerne.)
        uncertainty = np.exp(-((score - threshold) / max(sigma_trained, 1e-6)) ** 2)

        # ── Panel A: Classical score ──
        ax = axes[row, 0]
        im = ax.imshow(score, origin='lower', cmap='inferno', vmin=0, vmax=1)
        cs = ax.contour(score, levels=[threshold], colors='cyan', linewidths=1.0,
                        linestyles='--')
        ax.set_title(f'{sn}: Classical Score', fontsize=10)
        cb = _add_colorbar(ax, im)
        cb.ax.tick_params(labelsize=7)
        ax.tick_params(labelsize=7)

        # ── Panel B: |ZZ| always-on ──
        ax = axes[row, 1]
        vmax_zz = max(np.percentile(zz_alwayson, 99), 1e-6)
        im = ax.imshow(zz_alwayson, origin='lower', cmap='hot', vmin=0, vmax=vmax_zz)
        ax.set_title(f'{sn}: |ZZ| (no gating)', fontsize=10)
        cb = _add_colorbar(ax, im)
        cb.ax.tick_params(labelsize=7)
        ax.tick_params(labelsize=7)

        # ── Panel C: |ZZ| weighted ──
        ax = axes[row, 2]
        im = ax.imshow(zz_trained, origin='lower', cmap='hot', vmin=0, vmax=vmax_zz)
        ax.set_title(f'{sn}: |ZZ| (σ={sigma_trained:.3f})', fontsize=10)
        cb = _add_colorbar(ax, im)
        cb.ax.tick_params(labelsize=7)
        ax.tick_params(labelsize=7)
        # Annotate ZZ reduction — clearer label
        zz_reduction = 1 - zz_trained.sum() / max(zz_alwayson.sum(), 1e-10)
        ax.text(0.02, 0.98, f'ZZ reduced\nby {zz_reduction:.0%}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=8, color='white', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, pad=2))

        # ── Panel D: Uncertainty weight ──
        ax = axes[row, 3]
        im = ax.imshow(uncertainty, origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_title(f'{sn}: Uncertainty w(s)', fontsize=10)
        cb = _add_colorbar(ax, im)
        cb.ax.tick_params(labelsize=7)
        ax.tick_params(labelsize=7)
        pct_active = np.mean(uncertainty > 0.1) * 100
        ax.text(0.02, 0.98, f'{pct_active:.0f}% of cells\nw > 0.1',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=8, color='black',
                bbox=dict(facecolor='white', alpha=0.8, pad=2))

        print(f"  [{label}] ZZ reduction: {zz_reduction:.1%}, "
              f"Active cells (w>0.1): {pct_active:.1f}%")

    fig.suptitle('Hamiltonian Design: Uncertainty-Weighted ZZ Coupling',
                 fontsize=11, fontweight='bold')
    fig.subplots_adjust(top=0.93, hspace=0.25, wspace=0.25)
    out = os.path.join(FIG_DIR, 'fig11_hamiltonian_design.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out}")

    # ── Log file ──
    log_path = os.path.join(FIG_DIR, 'fig11_hamiltonian_design.log')
    with open(log_path, 'w') as f:
        f.write(f"Fig 11 — Hamiltonian Design Visualization\n")
        f.write(f"threshold_amr = {threshold}\n")
        f.write(f"sigma = {sigma_trained}\n")
        f.write(f"w_z_frac = {TRAINED_PARAMS['w_z_frac']}\n")
        f.write(f"N = {N}\n")
        f.write(f"Scenarios: {list(SCENARIOS.keys())}\n")
    print(f"Log  → {log_path}")


def _add_colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return plt.colorbar(im, cax=cax)


if __name__ == '__main__':
    main()
