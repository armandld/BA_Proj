#!/usr/bin/env python3
"""
Fig 17 — Topological Correction Attribution
=============================================

Decomposes QAOA corrections by Hamiltonian term to answer:
"When Q-HAS uses its unique topology features, does it make
 better decisions than when it doesn't?"

Panels:
  A) QAOA vs Classical accuracy binned by ZZ coupling strength
     → Do spatial correlations improve QAOA decisions?
  B) Correct-flip rate by dominant Hamiltonian term per scenario
     → Which quantum feature drives the best corrections?
  C) QAOA accuracy advantage in topology-rich vs smooth regions
     → Does topology awareness help where it matters?

Reuses instrumented BFS from fig16, extended to capture per-cell
Hamiltonian decomposition (ZZ, ZZZZ, K_xpoint, Z).
"""
import sys, os, json
import numpy as np
from collections import defaultdict
from types import SimpleNamespace

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
    apply_style, COLORS, TRAINED_PARAMS, CLASSICAL_PARAMS,
    make_sim, ground_truth_errors, _compute_depths,
    _hamilt_mapper_kwargs,
    filter_scenarios_dict,
    FIG_DIR,
)
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import _process_score
from Simulation.utils import get_periodic_patch
from Simulation.refinement import _prepare_vqa_input
from VQA.runtime import VQARuntime
from call_vqa_shell import call_vqa_shell

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

apply_style()

# ── Configuration ──
N = 256
TARGET_DIM = 2
MIN_SIZE = 6

SCENARIOS = filter_scenarios_dict({
    'init_kelvin_helmholtz': {'label': 'Kelvin-Helmholtz', 'n_steps': 400},
    'init_harris_tearing':   {'label': 'Harris Tearing',   'n_steps': 300},
    'init_mhd_rotor':        {'label': 'MHD Rotor',        'n_steps': 300},
    'init_orszag_tang':       {'label': 'Orszag-Tang',      'n_steps': 500},
})

SHORT = {'Kelvin-Helmholtz': 'KH', 'Harris Tearing': 'Tearing',
         'MHD Rotor': 'Rotor', 'Orszag-Tang': 'OT'}

solve_md = _compute_depths(N, TARGET_DIM, MIN_SIZE)
threshold = TRAINED_PARAMS['threshold_amr']


# ═════════════════════════════════════════════════════════════════════
#  INSTRUMENTED BFS — extended with Hamiltonian decomposition
# ═════════════════════════════════════════════════════════════════════

def instrumented_bfs_hamilt(sim, N, Phi_prev, threshold_amr, target_dim,
                            max_depth, min_size, gt_error_map):
    """Run VQA BFS capturing per-cell decisions AND Hamiltonian terms."""
    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    grid = sim.grid
    HamiltMapper = PhysicalMapper(**_hamilt_mapper_kwargs(grid))

    reps = (target_dim - 1) * 2
    args = SimpleNamespace(
        reps=reps, mode="simulator", backend="state_vector",
        shots=1024, method="COBYLA", opt_level=1,
        AdvAnomaliesEnable=True, K_opt=40, eps=1e-2,
    )
    vqa_runtime = VQARuntime(
        backend_name="state_vector", mode="simulator",
        shots=1024, opt_level=1,
    )

    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)
    full_h = Phi['phi_horizontal']
    full_v = Phi['phi_vertical']
    full_score = AngleMapper.classical_score(physics_state)

    full_prev_h = full_prev_v = None
    AveragePhiDev = None
    if Phi_prev is not None:
        full_prev_h = Phi_prev['phi_horizontal']
        full_prev_v = Phi_prev['phi_vertical']
        AveragePhiDev = 0.5 * (np.mean(np.abs(full_h - full_prev_h))
                                + np.mean(np.abs(full_v - full_prev_v)))

    H, W = full_h.shape
    decision_log = []
    pending = [(0, H, 0, W)]
    depth = 0

    while pending and depth <= max_depth:
        next_level = []
        for bounds in pending:
            y_s, y_e, x_s, x_e = bounds
            height, width = y_e - y_s, x_e - x_s
            if height < min_size or width < min_size:
                continue

            pad = 1 if depth > 0 else 0
            local_score_raw = get_periodic_patch(full_score, y_s, y_e,
                                                 x_s, x_e, pad=pad)
            is_periodic = (depth == 0)
            score_map_padded = _process_score(
                local_score_raw, is_periodic,
                target_dim + 2 * pad if pad > 0 else target_dim)
            score_map = (score_map_padded[1:-1, 1:-1]
                         if depth > 0 else score_map_padded)

            if depth >= max_depth:
                continue

            prep = _prepare_vqa_input(
                full_h, full_v, full_prev_h, full_prev_v,
                full_score, physics_state, bounds, depth, mapper, args,
                AveragePhiDev, TRAINED_PARAMS['beta'], target_dim,
                HamiltMapper=HamiltMapper, sim=sim,
                threshold_amr=threshold_amr,
            )
            if prep is None:
                continue

            angles, mini_hamilt_params, mini_score = prep
            prob_map_avant = mini_score
            if depth > 0:
                prob_map_avant = mini_score[1:-1, 1:-1]

            # ── Extract Hamiltonian terms ──
            C_h, C_v = mini_hamilt_params['C_edges']
            K_plaq = mini_hamilt_params['K_plaquettes']
            K_xpt = mini_hamilt_params.get('K_xpoint')
            H_h, H_v = mini_hamilt_params['H_edges']

            # Depad for depth > 0
            if depth > 0:
                C_h = C_h[1:-1, 1:-1]
                C_v = C_v[1:-1, 1:-1]
                K_plaq = K_plaq[1:-1, 1:-1]
                H_h = H_h[1:-1, 1:-1]
                H_v = H_v[1:-1, 1:-1]
                if K_xpt is not None:
                    K_xpt = K_xpt[1:-1, 1:-1]

            # ── VQA call ──
            result = call_vqa_shell(
                angles, mini_hamilt_params, False, args,
                period_bound=is_periodic, vqa_runtime=vqa_runtime,
            )

            if result is None:
                prob_map = prob_map_avant.copy()
            else:
                probs, _ = result
                ne = target_dim * target_dim
                probs_h = probs[:ne].reshape(target_dim, target_dim)
                probs_v = probs[ne:].reshape(target_dim, target_dim)
                prob_map = 0.5 * (probs_h + probs_v)

            step_y = height // target_dim
            step_x = width // target_dim

            for i in range(target_dim):
                for j in range(target_dim):
                    cl_s = float(score_map[i, j])
                    qa_p = float(prob_map[i, j])
                    sub_y_s = y_s + i * step_y
                    sub_y_e = (y_s + (i + 1) * step_y
                               if i < target_dim - 1 else y_e)
                    sub_x_s = x_s + j * step_x
                    sub_x_e = (x_s + (j + 1) * step_x
                               if j < target_dim - 1 else x_e)

                    gt_patch = gt_error_map[sub_y_s:sub_y_e,
                                            sub_x_s:sub_x_e]
                    gt_mean = float(np.mean(gt_patch))

                    dec_cl = cl_s >= threshold_amr
                    dec_qa = qa_p >= threshold_amr

                    # ── Per-cell Hamiltonian decomposition ──
                    # ZZ: sum of adjacent edge couplings
                    zz = abs(float(C_h[i, j])) + abs(float(C_v[i, j]))
                    if j > 0:
                        zz += abs(float(C_h[i, j - 1]))
                    if i > 0:
                        zz += abs(float(C_v[i - 1, j]))

                    # ZZZZ (circulation plaquette)
                    zzzz = abs(float(K_plaq[i, j]))

                    # ZZZZ (X-point)
                    xpt = (abs(float(K_xpt[i, j]))
                           if K_xpt is not None else 0.0)

                    # Z (activity bias)
                    z_bias = 0.5 * (abs(float(H_h[i, j]))
                                    + abs(float(H_v[i, j])))

                    decision_log.append({
                        'depth': depth,
                        'classical_score': cl_s,
                        'qaoa_prob': qa_p,
                        'correction': qa_p - cl_s,
                        'decision_classical': dec_cl,
                        'decision_qaoa': dec_qa,
                        'flipped': dec_cl != dec_qa,
                        'gt_error_mean': gt_mean,
                        'zz_strength': zz,
                        'zzzz_strength': zzzz,
                        'xpoint_strength': xpt,
                        'z_bias': z_bias,
                    })

                    if dec_qa:
                        next_level.append(
                            (sub_y_s, sub_y_e, sub_x_s, sub_x_e))

        pending = next_level
        depth += 1

    return decision_log


# ═════════════════════════════════════════════════════════════════════
#  DATA COLLECTION (with caching)
# ═════════════════════════════════════════════════════════════════════

n_scen = len(SCENARIOS)
if n_scen == 0:
    print("No scenarios for this phase.")
    sys.exit(0)

CACHE_PATH = os.path.join(FIG_DIR, '.fig17_cache.json')
use_cache = os.path.exists(CACHE_PATH) and '--recompute' not in sys.argv

if use_cache:
    print("Loaded from cache — replotting only. Use --recompute to force.")
    with open(CACHE_PATH) as f:
        all_logs = json.load(f)
else:
    print("=" * 70)
    print(f"Fig 17: Topological Attribution (N={N}, depth={solve_md})")
    print("=" * 70)

    all_logs = {}
    for scenario_init, cfg in SCENARIOS.items():
        label = cfg['label']
        print(f"\n  [{label}] Running instrumented BFS with Hamiltonian...")
        sim, Phi_prev = make_sim(N, scenario_init, cfg['n_steps'])
        gt = ground_truth_errors(sim, N, TARGET_DIM)
        log = instrumented_bfs_hamilt(
            sim, N, Phi_prev, threshold_amr=threshold,
            target_dim=TARGET_DIM, max_depth=solve_md,
            min_size=MIN_SIZE, gt_error_map=gt,
        )
        all_logs[label] = log
        n_flip = sum(1 for r in log if r['flipped'])
        print(f"    {len(log)} cells, {n_flip} flips "
              f"({100 * n_flip / max(len(log), 1):.1f}%)")

    with open(CACHE_PATH, 'w') as f:
        json.dump(all_logs, f)
    print(f"\nCache saved → {CACHE_PATH}")


# ═════════════════════════════════════════════════════════════════════
#  ANALYSIS — prepare arrays for plotting
# ═════════════════════════════════════════════════════════════════════

# Aggregate data across all scenarios
all_zz = []
all_qaoa_correct = []
all_cl_correct = []
per_scenario = {}

for label, log in all_logs.items():
    if not log:
        continue

    gt_arr = np.array([r['gt_error_mean'] for r in log])
    gt_thr = np.mean(gt_arr)
    should_refine = gt_arr > gt_thr

    cl_dec = np.array([r['decision_classical'] for r in log])
    qa_dec = np.array([r['decision_qaoa'] for r in log])
    flipped = np.array([r['flipped'] for r in log])

    qa_correct = (qa_dec == should_refine)
    cl_correct = (cl_dec == should_refine)

    zz = np.array([r['zz_strength'] for r in log])
    zzzz = np.array([r['zzzz_strength'] for r in log])
    xpt = np.array([r['xpoint_strength'] for r in log])
    z_b = np.array([r['z_bias'] for r in log])

    # Topology signal = ZZ + ZZZZ + X-point
    topo = zz + zzzz + xpt

    # Dominant term per cell
    terms = np.column_stack([zz, zzzz + xpt, z_b])
    dominant = np.argmax(terms, axis=1)  # 0=ZZ, 1=ZZZZ, 2=Z

    all_zz.extend(zz.tolist())
    all_qaoa_correct.extend(qa_correct.tolist())
    all_cl_correct.extend(cl_correct.tolist())

    # Per-scenario analysis
    # Among flipped cells, accuracy by dominant term
    flip_mask = flipped
    n_flip = int(flip_mask.sum())

    term_names = ['ZZ (spatial)', 'ZZZZ (topology)', 'Z (bias)']
    flip_acc_by_term = {}
    flip_count_by_term = {}
    for ti, tn in enumerate(term_names):
        mask = flip_mask & (dominant == ti)
        n = int(mask.sum())
        correct = int((mask & qa_correct).sum())
        flip_acc_by_term[tn] = correct / max(n, 1)
        flip_count_by_term[tn] = n

    # Topology-rich vs smooth split (median of topo signal)
    topo_med = np.median(topo) if len(topo) > 0 else 0.0
    rich_mask = topo > topo_med
    smooth_mask = ~rich_mask

    qa_acc_rich = float(qa_correct[rich_mask].mean()) if rich_mask.any() else 0.0
    cl_acc_rich = float(cl_correct[rich_mask].mean()) if rich_mask.any() else 0.0
    qa_acc_smooth = float(qa_correct[smooth_mask].mean()) if smooth_mask.any() else 0.0
    cl_acc_smooth = float(cl_correct[smooth_mask].mean()) if smooth_mask.any() else 0.0

    per_scenario[label] = {
        'flip_acc_by_term': flip_acc_by_term,
        'flip_count_by_term': flip_count_by_term,
        'adv_rich': qa_acc_rich - cl_acc_rich,
        'adv_smooth': qa_acc_smooth - cl_acc_smooth,
        'qa_acc_rich': qa_acc_rich,
        'cl_acc_rich': cl_acc_rich,
        'qa_acc_smooth': qa_acc_smooth,
        'cl_acc_smooth': cl_acc_smooth,
        'n_rich': int(rich_mask.sum()),
        'n_smooth': int(smooth_mask.sum()),
    }

# Aggregate ZZ quartile analysis
all_zz = np.array(all_zz)
all_qaoa_correct = np.array(all_qaoa_correct)
all_cl_correct = np.array(all_cl_correct)

# 4 bins by ZZ strength
if len(all_zz) > 0:
    quartiles = np.percentile(all_zz, [25, 50, 75])
    bins = np.digitize(all_zz, quartiles)  # 0,1,2,3
    bin_labels = ['Q1\n(weakest)', 'Q2', 'Q3', 'Q4\n(strongest)']
    qa_acc_per_bin = []
    cl_acc_per_bin = []
    for b in range(4):
        mask = bins == b
        qa_acc_per_bin.append(
            float(all_qaoa_correct[mask].mean()) if mask.any() else 0.0)
        cl_acc_per_bin.append(
            float(all_cl_correct[mask].mean()) if mask.any() else 0.0)


# ═════════════════════════════════════════════════════════════════════
#  PLOTTING — 1 × 3
# ═════════════════════════════════════════════════════════════════════

fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14, 4.2))

# ── Panel A: Accuracy by ZZ Coupling Quartile ──
x_q = np.arange(4)
w = 0.35
bars_qa = ax_a.bar(x_q - w / 2, qa_acc_per_bin, w,
                   color=COLORS['qaoa'], alpha=0.85, label='Q-HAS')
bars_cl = ax_a.bar(x_q + w / 2, cl_acc_per_bin, w,
                   color=COLORS['classical'], alpha=0.85, label='Classical')

# Value annotations
for bars in [bars_qa, bars_cl]:
    for bar in bars:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                  f'{h:.2f}', ha='center', va='bottom', fontsize=7)

ax_a.set_xticks(x_q)
ax_a.set_xticklabels(bin_labels, fontsize=8)
ax_a.set_ylabel('Decision Accuracy', fontsize=9)
ax_a.set_xlabel('ZZ Coupling Strength Quartile', fontsize=9)
ax_a.set_title('A) Accuracy by Spatial Correlation\n    Strength',
               fontsize=10, pad=10)
ax_a.set_ylim(0, 1.1)
ax_a.legend(fontsize=8, loc='lower right')
ax_a.tick_params(labelsize=8)

# ── Panel B: Correct Flip Rate by Dominant Hamiltonian Term ──
scen_labels_ordered = [s for s in SHORT.values()
                       if any(k for k in per_scenario
                              if SHORT.get(k, k) == s)]
scen_keys_ordered = [k for k in per_scenario]
n_s = len(scen_keys_ordered)
x_s = np.arange(n_s)
term_names = ['ZZ (spatial)', 'ZZZZ (topology)', 'Z (bias)']
term_colors = ['#D65F5F', '#59A14F', '#ECA63D']  # red, green, orange

bar_w = 0.25
for ti, tn in enumerate(term_names):
    vals = []
    counts = []
    for key in scen_keys_ordered:
        vals.append(per_scenario[key]['flip_acc_by_term'].get(tn, 0.0))
        counts.append(per_scenario[key]['flip_count_by_term'].get(tn, 0))
    offset = (ti - 1) * bar_w
    bars = ax_b.bar(x_s + offset, vals, bar_w * 0.88,
                    color=term_colors[ti], alpha=0.85, label=tn)
    # Annotate with count
    for bi, (bar, c) in enumerate(zip(bars, counts)):
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                  f'n={c}', ha='center', va='bottom', fontsize=6,
                  color='#555555')

ax_b.set_xticks(x_s)
ax_b.set_xticklabels([SHORT.get(k, k) for k in scen_keys_ordered],
                     fontsize=8)
ax_b.set_ylabel('Fraction of Correct Flips', fontsize=9)
ax_b.set_xlabel('Scenario', fontsize=9)
ax_b.set_title('B) Flip Accuracy by Dominant\n    Hamiltonian Term',
               fontsize=10, pad=10)
ax_b.set_ylim(0, 1.15)
ax_b.legend(fontsize=7, loc='upper right', ncol=1)
ax_b.tick_params(labelsize=8)

# ── Panel C: QAOA Advantage in Topology-Rich vs Smooth Regions ──
x_c = np.arange(n_s)
w_c = 0.35
rich_vals = [per_scenario[k]['adv_rich'] for k in scen_keys_ordered]
smooth_vals = [per_scenario[k]['adv_smooth'] for k in scen_keys_ordered]

bars_r = ax_c.bar(x_c - w_c / 2, rich_vals, w_c,
                  color='#59A14F', alpha=0.85, label='Topology-rich')
bars_s = ax_c.bar(x_c + w_c / 2, smooth_vals, w_c,
                  color='#A0A0A0', alpha=0.85, label='Smooth')

# Annotate with actual percentages
for bars in [bars_r, bars_s]:
    for bar in bars:
        h = bar.get_height()
        sign = '+' if h >= 0 else ''
        offset = 0.002 
        y_pos = h + (offset if h >= 0 else -offset)
        v_align = 'bottom' if h >= 0 else 'top'
        
        ax_c.text(bar.get_x() + bar.get_width() / 2,
                  y_pos,
                  f'{sign}{h:.2f}', ha='center',
                  va=v_align,
                  fontsize=7)

ax_c.axhline(0, color='black', lw=0.8)
ax_c.set_xticks(x_c)
ax_c.set_xticklabels([SHORT.get(k, k) for k in scen_keys_ordered],
                     fontsize=8)
ax_c.set_ylabel('Accuracy Advantage\n(Q-HAS $-$ Classical)', fontsize=9)
ax_c.set_xlabel('Scenario', fontsize=9)
ax_c.set_title('C) QAOA Advantage by\n    Topological Regime',
               fontsize=10, pad=10)
ax_c.legend(fontsize=8, loc='best')
ax_c.tick_params(labelsize=8)

# ── Layout ──
fig.suptitle('Topological Correction Attribution',
             fontsize=11, fontweight='bold', y=0.99)
fig.subplots_adjust(top=0.82, bottom=0.14, left=0.06, right=0.97,
                    wspace=0.32)

out = os.path.join(FIG_DIR, 'fig17_topological_attribution.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved → {out}")

# ── Log ──
log_lines = [
    "Fig 17 — Topological Correction Attribution",
    f"N={N}, threshold={threshold:.4f}",
    "",
    "Panel A: Accuracy by ZZ Coupling Quartile (all scenarios pooled)",
]
for bi, bl in enumerate(bin_labels):
    log_lines.append(
        f"  {bl.replace(chr(10), ' ')}: Q-HAS={qa_acc_per_bin[bi]:.3f}  "
        f"Classical={cl_acc_per_bin[bi]:.3f}")

log_lines.append("\nPanel B: Flip Accuracy by Dominant Term")
for key in scen_keys_ordered:
    sn = SHORT.get(key, key)
    log_lines.append(f"\n  {sn}:")
    for tn in term_names:
        acc = per_scenario[key]['flip_acc_by_term'].get(tn, 0.0)
        cnt = per_scenario[key]['flip_count_by_term'].get(tn, 0)
        log_lines.append(f"    {tn}: {acc:.3f} ({cnt} flips)")

log_lines.append("\nPanel C: QAOA Advantage by Topological Regime")
for key in scen_keys_ordered:
    sn = SHORT.get(key, key)
    p = per_scenario[key]
    log_lines.append(
        f"  {sn}: rich={p['adv_rich']:+.3f} (n={p['n_rich']}), "
        f"smooth={p['adv_smooth']:+.3f} (n={p['n_smooth']})")
    log_lines.append(
        f"        Q-HAS: rich={p['qa_acc_rich']:.3f} smooth={p['qa_acc_smooth']:.3f}  "
        f"Classical: rich={p['cl_acc_rich']:.3f} smooth={p['cl_acc_smooth']:.3f}")

log_path = os.path.join(FIG_DIR, 'fig17_topological_attribution.log')
with open(log_path, 'w') as f:
    f.write('\n'.join(log_lines))
print(f"Log  → {log_path}")
