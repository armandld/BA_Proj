"""
Figure 0: Pareto Front — Physics Error vs Compute Cost
Produces: figures/fig0_pareto_*.png

Plots phys_score vs patch_ratio across lambda_cost values for both
quantum (Q-HAS) and classical AMR.  Generates:
  - Per-scenario graphs (1 per scenario)
  - Grouped graphs (simple 4-scenario, complex 2-scenario)
  - Combined overview

Reads directly from Train_results/rescore_*_lambda*/trials_*.csv
to access per-scenario breakdown columns.
"""
import csv
import json
import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fig_utils import apply_style, COLORS, FIG_DIR

apply_style()

# ── Paths ──
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'Train_results')
JSON_PATH = os.path.join(PROJECT_ROOT, 'best_hyperparams.json')
## FIG_DIR imported from fig_utils (phase-aware)

# ── Scenario definitions ──
# Auto-detect which scenario columns exist in the CSV data.
# Old training: kh, vortex, tearing, coalescence
# New training: kh, tearing, ot, rotor
SCENARIOS_ALL = {
    'kelvin_helmholtz':   {'phys': 'phys_kh',      'patch': 'patch_kh',      'label': 'Kelvin-Helmholtz'},
    'harris_tearing':     {'phys': 'phys_tearing',  'patch': 'patch_tearing',  'label': 'Harris Tearing'},
    'orszag_tang':        {'phys': 'phys_ot',       'patch': 'patch_ot',       'label': 'Orszag-Tang'},
    'mhd_rotor':          {'phys': 'phys_rotor',    'patch': 'patch_rotor',    'label': 'MHD Rotor'},
}

# Patterns
QUANTUM_PATTERN = re.compile(r'^rescore_q_has_v2_phase([\w]+)_lambda([\d.]+)$')
CLASSICAL_PATTERN = re.compile(r'^rescore_classical_v2_phase([\w]+)_lambda([\d.]+)$')

# Target lambda for "interesting region" centering
TARGET_LAMBDA = 0.40


def load_all_trials(train_dir, pattern, phase_filter=None):
    """Load all trials from rescore dirs matching pattern.

    Returns dict: { (phase, lambda_val): [list of row dicts] }

    Parameters
    ----------
    phase_filter : str or None
        If set, only include rescore dirs whose phase matches exactly
        (e.g. "1b" to match phase1b, "2" to match phase2).
    """
    results = {}
    for entry in sorted(os.listdir(train_dir)):
        dirpath = os.path.join(train_dir, entry)
        if not os.path.isdir(dirpath):
            continue
        m = pattern.match(entry)
        if not m:
            continue
        phase = m.group(1)
        if phase_filter is not None and phase != phase_filter:
            continue
        lam_str = m.group(2)
        try:
            lam_val = float(lam_str)
        except ValueError:
            continue

        csvs = [f for f in os.listdir(dirpath) if f.endswith('.csv')]
        if not csvs:
            continue

        rows = []
        csvpath = os.path.join(dirpath, csvs[0])
        with open(csvpath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        results[(phase, lam_val)] = rows
    return results


def extract_pareto_front(phys, patch):
    """Return boolean mask of Pareto-optimal points (minimize both)."""
    n = len(phys)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if (phys[j] <= phys[i] and patch[j] <= patch[i] and
                    (phys[j] < phys[i] or patch[j] < patch[i])):
                is_pareto[i] = False
                break
    return is_pareto


def _safe_float(val):
    """Convert to float, return None on failure."""
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _collect_points(data, phys_col, patch_col, lam_cost):
    """Collect (phys, patch, combined_score) from trial data."""
    phys, patch, scores = [], [], []
    for (phase, lam_val), rows in data.items():
        for row in rows:
            p = _safe_float(row.get(phys_col))
            r = _safe_float(row.get(patch_col))
            if p is not None and r is not None and p < 5.0 and r < 1.5:
                phys.append(p)
                patch.append(r)
                scores.append(p + lam_cost * r)
    return np.array(phys), np.array(patch), np.array(scores)


def _plot_isocost_lines(ax, scores, lam_cost, y_max):
    """Draw iso-score lines: phys = S - lambda * patch."""
    patch_range = np.linspace(0, 1, 100)
    if len(scores) == 0:
        return
    score_levels = np.quantile(scores, [0.05, 0.1, 0.25, 0.5])
    for s in score_levels:
        iso_phys = s - lam_cost * patch_range
        ax.plot(patch_range, iso_phys, "--", color="gray", alpha=0.4, linewidth=0.8)
        y_at_1 = s - lam_cost
        if 0 <= y_at_1 <= y_max:
            ax.text(1.01, y_at_1, f"S={s:.3f}", fontsize=7, color="gray", va="center")


def _plot_method(fig, ax, phys, patch, scores, method, lam_cost, vmin=None, vmax=None):
    """Plot one method's points with viridis color, Pareto front, and best star."""
    if len(phys) == 0:
        return

    cmap = "magma_r" if method == 'classical' else "viridis_r"
    marker = "D" if method == 'classical' else "o"
    label = "Classical" if method == 'classical' else "Q-HAS"
    front_color = COLORS['classical'] if method == 'classical' else COLORS['qaoa']

    # All points colored by combined score
    sc = ax.scatter(patch, phys, c=scores, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=40, alpha=0.7, edgecolors="k", linewidth=0.3,
                    marker=marker)
    fig.colorbar(sc, ax=ax, label=f"{label} Score ($\\lambda$={lam_cost:.2f})",
                 shrink=0.8, pad=0.02)

    # Pareto front
    pareto = extract_pareto_front(phys, patch)
    pidx = np.where(pareto)[0]
    if len(pidx) >= 2:
        pts = np.column_stack([patch[pidx], phys[pidx]])
        order = np.argsort(pts[:, 0])
        ax.plot(pts[order, 0], pts[order, 1],
                color=front_color, linestyle='--', linewidth=2, alpha=0.7,
                label=f'{label} Pareto front', zorder=4)

    # Best trial star
    best_idx = np.argmin(scores)
    ax.scatter(patch[best_idx], phys[best_idx], s=250, c="red", marker="*",
               zorder=5, edgecolors="k", linewidth=0.5,
               label=f"Best {label} (S={scores[best_idx]:.4f})")


def plot_pareto_scenario(quantum_data, classical_data,
                         phys_col, patch_col, title,
                         center_lambda=TARGET_LAMBDA):
    """Plot quantum + classical Pareto front for one scenario.

    Returns the figure (caller must savefig).
    Uses recompute_lambda_scores-style: viridis colormap by combined score,
    iso-score lines, red best-trial star, dashed Pareto front.
    """

    q_phys, q_patch, q_scores = _collect_points(quantum_data, phys_col, patch_col, center_lambda)
    c_phys, c_patch, c_scores = _collect_points(classical_data, phys_col, patch_col, center_lambda)

    v_min, v_max = None, None
    if len(q_scores) > 0:
        v_min = np.min(q_scores)
        v_max = np.max(q_scores)

    if v_min is not None and len(c_scores) > 0:
        mask = (c_scores >= v_min) & (c_scores <= v_max)
        c_phys = c_phys[mask]
        c_patch = c_patch[mask]
        c_scores = c_scores[mask]

    has_q = len(q_phys) > 0
    has_c = len(c_phys) > 0
    ncols = int(has_q) + int(has_c)

    if ncols == 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(title)
        return fig

    fig, axes = plt.subplots(1, ncols, figsize=(12 * ncols, 9))
    if ncols == 1:
        axes = [axes]

    y_max = max(np.percentile(q_phys, 95) * 1.3, 0.4)

    col = 0
    if has_q:
        all_scores = np.concatenate([q_scores, c_scores]) if has_c else q_scores
        _plot_isocost_lines(axes[col], all_scores, center_lambda, y_max)
        _plot_method(fig, axes[col], q_phys, q_patch, q_scores, 'quantum', center_lambda, vmin=v_min, vmax=v_max)
        axes[col].set_xlabel("Patch Ratio (computational cost)", fontsize=12)
        axes[col].set_ylabel("Physics Score (L2 error)", fontsize=12)
        axes[col].set_title(f"{title} — Q-HAS", fontsize=13)
        axes[col].set_ylim(-0.05, y_max)
        axes[col].set_xlim(-0.02, 1.05)
        axes[col].legend(loc='upper right', fontsize=9)
        axes[col].grid(True, alpha=0.3)
        col += 1

    if has_c:
        all_scores = np.concatenate([q_scores, c_scores]) if has_q else c_scores
        _plot_isocost_lines(axes[col], all_scores, center_lambda, y_max)
        _plot_method(fig, axes[col], c_phys, c_patch, c_scores, 'classical', center_lambda, vmin=v_min, vmax=v_max)
        axes[col].set_xlabel("Patch Ratio (computational cost)", fontsize=12)
        axes[col].set_ylabel("Physics Score (L2 error)", fontsize=12)
        axes[col].set_title(f"{title} — Classical", fontsize=13)
        axes[col].set_ylim(-0.05, y_max)
        axes[col].set_xlim(-0.02, 1.05)
        axes[col].legend(loc='upper right', fontsize=9)
        axes[col].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _collect_grouped_points(data, scenario_dict, lam_cost):
    """Collect averaged (phys, patch, score) across a group of scenarios."""
    phys, patch, scores = [], [], []
    for (phase, lam_val), rows in data.items():
        for row in rows:
            physes, patches = [], []
            for sc_name, sc_info in scenario_dict.items():
                p = _safe_float(row.get(sc_info['phys']))
                r = _safe_float(row.get(sc_info['patch']))
                if p is not None and r is not None:
                    physes.append(p)
                    patches.append(r)
            if len(physes) == len(scenario_dict):
                avg_p = np.mean(physes)
                avg_r = np.mean(patches)
                phys.append(avg_p)
                patch.append(avg_r)
                scores.append(avg_p + lam_cost * avg_r)
    return np.array(phys), np.array(patch), np.array(scores)


def plot_grouped_pareto(quantum_data, classical_data,
                        scenario_dict, title, lam_cost=TARGET_LAMBDA):
    """Plot aggregate Pareto for a group of scenarios using average phys/patch.

    Returns the figure (caller must savefig).
    """
    q_phys, q_patch, q_scores = _collect_grouped_points(quantum_data, scenario_dict, lam_cost)
    c_phys, c_patch, c_scores = _collect_grouped_points(classical_data, scenario_dict, lam_cost)

    v_min, v_max = None, None
    if len(q_scores) > 0:
        v_min = np.min(q_scores)
        v_max = np.max(q_scores)

    if v_min is not None and len(c_scores) > 0:
        mask = (c_scores >= v_min) & (c_scores <= v_max)
        c_phys = c_phys[mask]
        c_patch = c_patch[mask]
        c_scores = c_scores[mask]

    has_q = len(q_phys) > 0
    has_c = len(c_phys) > 0
    ncols = int(has_q) + int(has_c)
    if ncols == 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(title)
        return fig

    fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 8))
    if ncols == 1:
        axes = [axes]

    y_max = max(np.percentile(q_phys, 95) * 1.3, 0.4)

    col = 0
    if has_q:
        all_scores = np.concatenate([q_scores, c_scores]) if has_c else q_scores
        _plot_isocost_lines(axes[col], all_scores, lam_cost, y_max)
        _plot_method(fig, axes[col], q_phys, q_patch, q_scores, 'quantum', lam_cost, vmin=v_min, vmax=v_max)
        axes[col].set_xlabel("Avg Patch Ratio (computational cost)", fontsize=12)
        axes[col].set_ylabel("Avg Physics Score (L2 error)", fontsize=12)
        axes[col].set_title(f"{title}\nQ-HAS", fontsize=13)
        axes[col].set_ylim(-0.05, y_max)
        axes[col].set_xlim(-0.02, 1.05)
        axes[col].legend(loc='upper right', fontsize=9)
        axes[col].grid(True, alpha=0.3)
        col += 1

    if has_c:
        all_scores = np.concatenate([q_scores, c_scores]) if has_q else c_scores
        _plot_isocost_lines(axes[col], all_scores, lam_cost, y_max)
        _plot_method(fig, axes[col], c_phys, c_patch, c_scores, 'classical', lam_cost, vmin=v_min, vmax=v_max)
        axes[col].set_xlabel("Avg Patch Ratio (computational cost)", fontsize=12)
        axes[col].set_ylabel("Avg Physics Score (L2 error)", fontsize=12)
        axes[col].set_title(f"{title}\nClassical", fontsize=13)
        axes[col].set_ylim(-0.05, y_max)
        axes[col].set_xlim(-0.02, 1.05)
        axes[col].legend(loc='upper right', fontsize=9)
        axes[col].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

print("Loading trial data from CSVs...")
# Optional phase filters from environment (set by generate_figures.sh)
Q_PHASE_FILTER = os.environ.get('FIGURE_Q_PHASE_FILTER', None)
C_PHASE_FILTER = os.environ.get('FIGURE_C_PHASE_FILTER', None)
if Q_PHASE_FILTER:
    print(f"[fig0] Quantum phase filter: {Q_PHASE_FILTER}")
if C_PHASE_FILTER:
    print(f"[fig0] Classical phase filter: {C_PHASE_FILTER}")

quantum_data = load_all_trials(TRAIN_DIR, QUANTUM_PATTERN, phase_filter=Q_PHASE_FILTER)
classical_data = load_all_trials(TRAIN_DIR, CLASSICAL_PATTERN, phase_filter=C_PHASE_FILTER)

print(f"  Quantum:   {len(quantum_data)} phase/lambda combos, "
      f"{sum(len(r) for r in quantum_data.values())} total trials")
print(f"  Classical: {len(classical_data)} phase/lambda combos, "
      f"{sum(len(r) for r in classical_data.values())} total trials")

# ── 1. Per-scenario graphs ──
for sc_name, sc_info in SCENARIOS_ALL.items():
    fig = plot_pareto_scenario(quantum_data, classical_data,
                               sc_info['phys'], sc_info['patch'],
                               f"Pareto Front: {sc_info['label']}")
    out = os.path.join(FIG_DIR, f'fig0_pareto_{sc_name}.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")

# ── 2. Grouped: all scenarios ──
fig = plot_grouped_pareto(quantum_data, classical_data,
                          SCENARIOS_ALL,
                          'Pareto Front: All Scenarios (KH + Tearing + OT + Rotor)')
out = os.path.join(FIG_DIR, 'fig0_pareto_all_combined.png')
fig.savefig(out, dpi=300)
plt.close(fig)
print(f"  Saved: {out}")

# ── 5. Update best_hyperparams.json with Pareto front data ──
# Compute overall Pareto front from quantum trials (phys_score vs patch_ratio)
all_q_phys, all_q_patch, all_q_trials = [], [], []
for (phase, lam_val), rows in quantum_data.items():
    for row in rows:
        p = _safe_float(row.get('phys_score'))
        r = _safe_float(row.get('patch_ratio'))
        if p is not None and r is not None and p < 5.0:
            all_q_phys.append(p)
            all_q_patch.append(r)
            all_q_trials.append({
                'trial': int(row.get('trial', -1)),
                'phase': phase,
                'lambda': lam_val,
                'phys_score': p,
                'patch_ratio': r,
            })

if all_q_phys:
    q_phys_arr = np.array(all_q_phys)
    q_patch_arr = np.array(all_q_patch)
    pareto_mask = extract_pareto_front(q_phys_arr, q_patch_arr)

    pareto_front = []
    for i, is_p in enumerate(pareto_mask):
        if is_p:
            pareto_front.append(all_q_trials[i])
    pareto_front.sort(key=lambda x: x['phys_score'])

    # Update JSON
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH, 'r') as f:
            json_data = json.load(f)
    else:
        json_data = {}

    json_data['pareto_front_quantum'] = pareto_front

    # Pareto best = lowest combined score
    if pareto_front:
        best = min(pareto_front, key=lambda x: x['phys_score'] + TARGET_LAMBDA * x['patch_ratio'])
        json_data['pareto_best_quantum'] = best

    with open(JSON_PATH, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"\nUpdated {JSON_PATH} with Pareto front ({len(pareto_front)} points).")

print("\nDone. All fig0_pareto_*.png saved to figures/")
