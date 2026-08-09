#!/usr/bin/env python3
"""
Fig 13 — Uncertainty Weighting (σ) Ablation Study

Shows how σ controls which cells receive quantum spatial correlations.
Small σ concentrates ZZ coupling on cells near the decision boundary;
large σ activates coupling everywhere.

Panels:
  A) Score distance from threshold — histogram per scenario with σ-window
     Gaussian overlays.  Visually explains WHY some scenarios have near-zero
     coupling (scores far from threshold) while others activate.
  B) Active edge fraction vs σ — what fraction of ZZ edges have
     uncertainty weight > 0.1 at each σ.
  C) ZZ retention vs σ — mean |ZZ(σ)| / mean |ZZ(σ→∞)| per scenario,
     showing how much coupling survives the gating.
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
    make_sim, ground_truth_errors, _hamilt_mapper_kwargs,
    filter_scenarios_dict,
)
from Simulation.HamiltParams import PhysicalMapper
from Simulation.PhysToAngle import AngleMapper

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

apply_style()

# ── Configuration ──
N = 256
TARGET_DIM = 2
MIN_SIZE = 6

SIGMA_VALUES = [0.023, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]
SIGMA_BASELINE = 100.0  # effectively no gating

SCENARIOS = filter_scenarios_dict({
    'Kelvin-Helmholtz': {'init': 'init_kelvin_helmholtz', 'n_steps': 400},
    'Harris Tearing':   {'init': 'init_harris_tearing',   'n_steps': 300},
    'MHD Rotor':        {'init': 'init_mhd_rotor',        'n_steps': 300},
    'Orszag-Tang':      {'init': 'init_orszag_tang',       'n_steps': 500},
})

threshold = TRAINED_PARAMS['threshold_amr']
sigma_trained = TRAINED_PARAMS.get('sigma', 0.023)

SHORT = {'Kelvin-Helmholtz': 'KH', 'Orszag-Tang': 'OT',
         'Harris Tearing': 'Tearing', 'MHD Rotor': 'Rotor'}
SCEN_COLORS = {'KH': '#D65F5F', 'OT': '#4878CF',
               'Tearing': '#59A14F', 'Rotor': '#ECA63D'}
MARKERS = {'KH': 'o', 'OT': 's', 'Tearing': 'D', 'Rotor': '^'}


def analyze_coefficients(sim, score, sigma_val):
    """Compute ZZ/Z stats for a given sigma."""
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

    zz_mean = 0.5 * (np.abs(C_h).mean() + np.abs(C_v).mean())
    z_mean = np.abs(H_h).mean()

    # Active edge fraction (uncertainty weight > 0.1)
    uncertainty = np.exp(-((score - threshold) / max(sigma_val, 1e-6)) ** 2)
    active_frac = np.mean(uncertainty > 0.1)

    return {
        'zz_mean': zz_mean,
        'z_mean': z_mean,
        'active_frac': active_frac,
    }


def main():
    n_scenarios = len(SCENARIOS)
    if n_scenarios == 0:
        print("No scenarios for this phase — skipping fig13")
        return

    # ── Collect data ──
    print("Collecting coefficient data across σ values...")
    coeff_data = {}   # scenario → {sigma → stats}
    score_data = {}   # scenario → score array (for histogram)

    for label, cfg in SCENARIOS.items():
        print(f"  [{label}] Simulating...")
        sim, _ = make_sim(N, cfg['init'], cfg['n_steps'])
        state = sim.get_fluxes()
        score = AngleMapper.classical_score(state)
        score_data[label] = score

        coeff_data[label] = {}
        # Baseline (no gating)
        coeff_data[label]['baseline'] = analyze_coefficients(sim, score, SIGMA_BASELINE)
        print(f"    baseline: |ZZ|={coeff_data[label]['baseline']['zz_mean']:.6f}")

        for sigma in SIGMA_VALUES:
            coeff_data[label][sigma] = analyze_coefficients(sim, score, sigma)
            print(f"    σ={sigma:.3f}: active={coeff_data[label][sigma]['active_frac']:.1%}, "
                  f"|ZZ|={coeff_data[label][sigma]['zz_mean']:.6f}")

    # ── 3-panel figure ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    # ════════════════════════════════════════════════════════════════
    # Panel A: Score distance from threshold + σ Gaussian windows
    # ════════════════════════════════════════════════════════════════
    ax = axes[0]
    bins = np.linspace(0, 0.5, 50)

    # Draw histograms first to establish y-limits
    for label in SCENARIOS:
        sn = SHORT.get(label, label)
        score = score_data[label]
        dist = np.abs(score.ravel() - threshold)
        ax.hist(dist, bins=bins, alpha=0.45, label=sn,
                color=SCEN_COLORS.get(sn, 'gray'), density=True,
                edgecolor='none')

    # Now overlay Gaussian windows, scaled to the histogram y-range
    ymax = ax.get_ylim()[1]
    x_g = np.linspace(0, 0.5, 200)
    sigma_show = [float(f"{sigma_trained:.3f}"), 0.05, 0.10, 0.20]
    linestyles = ['-', '--', '-.', ':']
    for sigma, ls in zip(sigma_show, linestyles):
        gauss = np.exp(-(x_g / sigma) ** 2)
        ax.plot(x_g, gauss * ymax * 0.75, color='black',
                linestyle=ls, linewidth=1.0, alpha=0.7,
                label=f'σ={sigma}')

    ax.set_xlabel('|score − threshold|', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.set_title('A) Score Distance from Threshold', fontsize=9)
    ax.legend(fontsize=5.5, loc='upper right', ncol=2)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 0.5)

    # ════════════════════════════════════════════════════════════════
    # Panel B: Active edge fraction vs σ
    # ════════════════════════════════════════════════════════════════
    ax = axes[1]
    for label in SCENARIOS:
        sn = SHORT.get(label, label)
        fracs = [coeff_data[label][s]['active_frac'] * 100 for s in SIGMA_VALUES]
        ax.plot(SIGMA_VALUES, fracs, f'-{MARKERS.get(sn, "o")}',
                color=SCEN_COLORS.get(sn, 'gray'),
                label=sn, markersize=4, linewidth=1.2)
    ax.axvline(x=sigma_trained, color='#D65F5F', linestyle=':', alpha=0.7,
               linewidth=1.2, label=f'σ*={sigma_trained:.3f}')
    ax.set_xlabel('σ (uncertainty width)', fontsize=8)
    ax.set_ylabel('Active Edges (%)', fontsize=8)
    ax.set_title('B) Fraction of Active ZZ Coupling', fontsize=9)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, max(SIGMA_VALUES) + 0.02)
    ax.set_ylim(0, 105)

    # ════════════════════════════════════════════════════════════════
    # Panel C: ZZ retention % vs σ
    # mean|ZZ(σ)| / mean|ZZ(baseline)| × 100
    # ════════════════════════════════════════════════════════════════
    ax = axes[2]
    for label in SCENARIOS:
        sn = SHORT.get(label, label)
        baseline_zz = coeff_data[label]['baseline']['zz_mean']
        if baseline_zz < 1e-12:
            # Skip scenarios with zero baseline (shouldn't happen)
            continue
        retention = [coeff_data[label][s]['zz_mean'] / baseline_zz * 100
                     for s in SIGMA_VALUES]
        ax.plot(SIGMA_VALUES, retention, f'-{MARKERS.get(sn, "o")}',
                color=SCEN_COLORS.get(sn, 'gray'),
                label=sn, markersize=4, linewidth=1.2)
    ax.axvline(x=sigma_trained, color='#D65F5F', linestyle=':', alpha=0.7,
               linewidth=1.2, label=f'σ*={sigma_trained:.3f}')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('σ (uncertainty width)', fontsize=8)
    ax.set_ylabel('ZZ Retention (%)', fontsize=8)
    ax.set_title('C) ZZ Coupling Retained vs σ', fontsize=9)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, max(SIGMA_VALUES) + 0.02)
    ax.set_ylim(-5, 110)

    fig.suptitle('Uncertainty Weighting Ablation',
                 fontsize=11, fontweight='bold')
    fig.subplots_adjust(top=0.85, wspace=0.30)
    out = os.path.join(FIG_DIR, 'fig13_sigma_ablation.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out}")

    # ── Log ──
    log_path = os.path.join(FIG_DIR, 'fig13_sigma_ablation.log')
    with open(log_path, 'w') as f:
        f.write(f"Fig 13 — Sigma Ablation Study\n")
        f.write(f"threshold = {threshold}\n")
        f.write(f"trained σ = {sigma_trained}\n\n")
        for label in SCENARIOS:
            sn = SHORT.get(label, label)
            baseline_zz = coeff_data[label]['baseline']['zz_mean']
            f.write(f"\n{label} ({sn}):\n")
            f.write(f"  Baseline |ZZ| = {baseline_zz:.6f}\n")
            f.write(f"  {'sigma':>8}  {'Active%':>8}  {'|ZZ|':>12}  {'Retention%':>12}\n")
            for s in SIGMA_VALUES:
                d = coeff_data[label][s]
                ret = d['zz_mean'] / max(baseline_zz, 1e-12) * 100
                f.write(f"  {s:>8.3f}  "
                        f"{d['active_frac']*100:>8.1f}  "
                        f"{d['zz_mean']:>12.6f}  "
                        f"{ret:>12.1f}\n")
            # Score distribution stats
            score = score_data[label]
            dist = np.abs(score.ravel() - threshold)
            f.write(f"  Score distance: median={np.median(dist):.4f}, "
                    f"mean={np.mean(dist):.4f}, "
                    f"frac<σ*={np.mean(dist < sigma_trained):.3f}\n")
    print(f"Log  → {log_path}")


if __name__ == '__main__':
    main()
