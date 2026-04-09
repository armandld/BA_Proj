#!/usr/bin/env python3
"""
Figure 14 — Boundary Correction Proof-of-Concept — ABANDONED.

This figure is abandoned because:
- At 2×2 VQA resolution, all block scores are far above/below threshold,
  leaving no natural borderline cells to demonstrate boundary corrections.
- The Hamiltonian coefficient visualization (Panel B) shows near-zero values
  due to σ=0.023 suppression, making it uninformative.
- Panel D's scaling argument is theoretical with no empirical data.
- The figure doesn't effectively communicate the boundary correction mechanism.

The boundary correction concept is better explained in text (README) with
reference to Figs 11 (Hamiltonian design) and 15 (decision flip analysis).
"""
import sys
print("Fig 14: SKIPPED (abandoned — see docstring)")
sys.exit(0)

# ── Original code below (kept for reference) ──
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

from fig_utils import (
    apply_style, COLORS, FIG_DIR, TRAINED_PARAMS, CLASSICAL_PARAMS,
    make_sim, ground_truth_errors, qaoa_block_scores, _hamilt_mapper_kwargs,
)
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper

apply_style()

# ── Configuration ─────────────────────────────────────────────────────
N = 128                # Simulation resolution
N_BLOCKS = 2           # VQA grid size (2x2 = 8 qubits)
K_OPT = 60             # COBYLA iterations (higher for better convergence)
THRESHOLD = TRAINED_PARAMS.get('threshold_amr', 0.30)
SIGMA = TRAINED_PARAMS.get('sigma', 0.05)

# Scenarios to search for natural boundary correction cases
# Search at many timesteps to maximize chance of finding borderline scores
SCENARIOS = [
    ('KH (very early)', 'init_kelvin_helmholtz', 3),
    ('KH (early)', 'init_kelvin_helmholtz', 5),
    ('KH (mid-early)', 'init_kelvin_helmholtz', 8),
    ('KH (mid)', 'init_kelvin_helmholtz', 15),
    ('KH (late)', 'init_kelvin_helmholtz', 25),
    ('OT (very early)', 'init_orszag_tang', 3),
    ('OT (early)', 'init_orszag_tang', 5),
    ('OT (mid-early)', 'init_orszag_tang', 8),
    ('OT (mid)', 'init_orszag_tang', 15),
    ('Tearing (very early)', 'init_harris_tearing', 3),
    ('Tearing (early)', 'init_harris_tearing', 5),
    ('Tearing (mid)', 'init_harris_tearing', 15),
    ('Rotor (very early)', 'init_mhd_rotor', 3),
    ('Rotor (early)', 'init_mhd_rotor', 5),
    ('Rotor (mid)', 'init_mhd_rotor', 8),
    ('Rotor (mid)', 'init_mhd_rotor', 15),
]


def find_boundary_cases():
    """Search MHD scenarios for natural boundary correction cases.

    A boundary case is one where:
    - At least one cell has score within +/- margin of threshold
    - That cell has neighbors above threshold (majority-refine context)

    Uses _process_score (same max-pool as the actual BFS) instead of raw
    block-max so that scores reflect what the VQA/classical AMR really sees.
    Progressively widens the margin if nothing is found.
    """
    from Simulation.RescaleArrays import _process_score

    cases = []

    # Try progressively wider margins until we find something
    for margin in [0.15, 0.30, 0.50, 0.80, 1.0]:
        cases.clear()
        for label, scenario, n_steps in SCENARIOS:
            print(f"  Searching {label} (n_steps={n_steps})...")
            try:
                sim, Phi_prev = make_sim(N, scenario, n_steps)
            except Exception as e:
                print(f"    Skip: {e}")
                continue

            # Use the SAME downsampling as the actual BFS (_process_score)
            physics_state = sim.get_fluxes()
            full_score = AngleMapper.classical_score(physics_state)
            scores = _process_score(full_score, True, N_BLOCKS)
            gt_err = ground_truth_errors(sim, N)

            # Block-level ground truth: average error per block
            bh, bw = N // N_BLOCKS, N // N_BLOCKS
            gt_blocks = np.zeros((N_BLOCKS, N_BLOCKS))
            for bi in range(N_BLOCKS):
                for bj in range(N_BLOCKS):
                    gt_blocks[bi, bj] = np.mean(
                        gt_err[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw]
                    )

            # GT decision: refine if error above median
            gt_median = np.median(gt_blocks)
            gt_refine = gt_blocks > gt_median

            # Classical decision
            cl_refine = scores > THRESHOLD

            # Find borderline cells
            for i in range(N_BLOCKS):
                for j in range(N_BLOCKS):
                    dist = abs(scores[i, j] - THRESHOLD)
                    if dist < margin:
                        # Count neighbors above threshold
                        neighbors = []
                        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ni = (i + di) % N_BLOCKS
                            nj = (j + dj) % N_BLOCKS
                            neighbors.append(scores[ni, nj])
                        n_above = sum(1 for s in neighbors if s > THRESHOLD)

                        # Is this a correction opportunity?
                        classical_wrong = cl_refine[i,j] != gt_refine[i,j]

                        cases.append({
                            'label': label,
                            'scenario': scenario,
                            'n_steps': n_steps,
                            'sim': sim,
                            'Phi_prev': Phi_prev,
                            'scores': scores,
                            'gt_blocks': gt_blocks,
                            'gt_refine': gt_refine,
                            'gt_err': gt_err,
                            'cell': (i, j),
                            'cell_score': scores[i, j],
                            'dist_to_threshold': dist,
                            'n_neighbors_above': n_above,
                            'classical_wrong': classical_wrong,
                            'neighbor_scores': neighbors,
                        })

        if cases:
            if margin > 0.15:
                print(f"  (widened margin to {margin:.2f} to find cases)")
            break

    # Sort: prefer cases where classical is wrong and cell is borderline
    cases.sort(key=lambda c: (
        -int(c['classical_wrong']),   # classical wrong first
        c['dist_to_threshold'],        # closest to threshold
        -c['n_neighbors_above'],       # more refine neighbors
    ))
    return cases


def compute_coefficients_for_case(sim, scores):
    """Compute full Hamiltonian coefficients for a 2x2 case."""
    grid = sim.grid
    hm_kwargs = _hamilt_mapper_kwargs(grid)
    HamiltMapper = PhysicalMapper(**hm_kwargs)
    physics_state = sim.get_fluxes()
    full_score = AngleMapper.classical_score(physics_state)
    hp = HamiltMapper.compute_coefficients(
        sim, full_score, physics_state, THRESHOLD,
        advanced_anomalies_enabled=True,
    )
    return hp, full_score


def run_qaoa_for_case(case):
    """Run QAOA on a specific case and return probabilities."""
    sim = case['sim']
    Phi_prev = case['Phi_prev']
    try:
        qaoa_scores = qaoa_block_scores(
            sim, N, N_BLOCKS,
            threshold=THRESHOLD,
            K_opt=K_OPT,
            Phi_prev=Phi_prev,
        )
        return qaoa_scores
    except Exception as e:
        print(f"    QAOA failed: {e}")
        return None


def make_figure(case, qaoa_scores):
    """Create the 4-panel boundary correction figure."""
    scores = case['scores']
    gt_refine = case['gt_refine']
    gt_err = case['gt_err']
    gt_blocks = case['gt_blocks']
    cell = case['cell']
    label = case['label']

    cl_refine = scores > THRESHOLD
    qa_refine = qaoa_scores > THRESHOLD if qaoa_scores is not None else cl_refine.copy()

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                           left=0.06, right=0.94, top=0.93, bottom=0.06)

    # ── Panel A: Classical Score Map ──────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    im = ax_a.imshow(gt_err, cmap='hot', origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax_a, label='Error indicator', shrink=0.8)

    # Overlay 2x2 grid
    bh, bw = N // N_BLOCKS, N // N_BLOCKS
    for bi in range(N_BLOCKS):
        for bj in range(N_BLOCKS):
            x0, y0 = bj * bw, bi * bh
            color = 'lime' if scores[bi, bj] > THRESHOLD else 'cyan'
            lw = 3 if (bi, bj) == cell else 1.5
            ls = '--' if (bi, bj) == cell else '-'
            rect = plt.Rectangle((x0, y0), bw, bh,
                                  linewidth=lw, edgecolor=color,
                                  facecolor='none', linestyle=ls)
            ax_a.add_patch(rect)
            # Score label
            ax_a.text(x0 + bw/2, y0 + bh/2,
                      f'{scores[bi,bj]:.2f}',
                      ha='center', va='center', fontsize=12,
                      fontweight='bold', color='white',
                      bbox=dict(boxstyle='round,pad=0.2',
                                facecolor='black', alpha=0.7))

    ax_a.axhline(y=bh, color='white', linewidth=0.5, alpha=0.5)
    ax_a.axvline(x=bw, color='white', linewidth=0.5, alpha=0.5)
    ax_a.set_title(f'A) {label}: Score Map (thr={THRESHOLD:.2f})', fontsize=9)
    ax_a.set_xlabel('x')
    ax_a.set_ylabel('y')

    # ── Panel B: Coefficient Analysis ─────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    # Build a visual representation of the 2x2 Hamiltonian
    # Show cells as boxes with Z bias, edges as lines with ZZ strength
    cell_positions = {
        (0,0): (0.2, 0.7), (0,1): (0.8, 0.7),
        (1,0): (0.2, 0.2), (1,1): (0.8, 0.2),
    }

    # Compute uncertainty weights for each cell
    sigma_val = max(SIGMA, 1e-6)
    uncertainty = np.exp(-((scores - THRESHOLD) / sigma_val) ** 2)

    # Z bias
    hp, full_score = compute_coefficients_for_case(case['sim'], scores)
    H_h, H_v = hp['H_edges']
    C_h, C_v = hp['C_edges']
    K_plaq = hp['K_plaquettes']

    # Downsample coefficients to 2x2 for visualization
    from Simulation.RescaleArrays import _maxabs_pool_2d
    if H_h.shape[0] > N_BLOCKS:
        H_h_ds = _maxabs_pool_2d(H_h, N_BLOCKS, N_BLOCKS)
        H_v_ds = _maxabs_pool_2d(H_v, N_BLOCKS, N_BLOCKS)
        C_h_ds = _maxabs_pool_2d(C_h, N_BLOCKS, N_BLOCKS)
        C_v_ds = _maxabs_pool_2d(C_v, N_BLOCKS, N_BLOCKS)
        K_ds = _maxabs_pool_2d(K_plaq, N_BLOCKS, N_BLOCKS)
    else:
        H_h_ds, H_v_ds = H_h, H_v
        C_h_ds, C_v_ds = C_h, C_v
        K_ds = K_plaq

    ax_b.set_xlim(-0.05, 1.05)
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.set_aspect('equal')
    ax_b.axis('off')

    # Draw cells
    for (i, j), (cx, cy) in cell_positions.items():
        is_borderline = (i, j) == cell
        s = scores[i, j]
        u = uncertainty[i, j]
        z = 0.5 * (H_h_ds[i, j] + H_v_ds[i, j])

        # Color by score relative to threshold
        if s > THRESHOLD:
            fcolor = '#E91E63' if not is_borderline else '#FF5722'
            decision = 'REFINE'
        else:
            fcolor = '#2196F3' if not is_borderline else '#FF9800'
            decision = 'SKIP'

        box_size = 0.22
        rect = plt.Rectangle((cx - box_size/2, cy - box_size/2),
                              box_size, box_size,
                              facecolor=fcolor, edgecolor='black',
                              linewidth=3 if is_borderline else 1.5,
                              linestyle='--' if is_borderline else '-',
                              alpha=0.8)
        ax_b.add_patch(rect)
        ax_b.text(cx, cy + 0.04, f's={s:.2f}', ha='center', va='center',
                  fontsize=9, fontweight='bold', color='white')
        ax_b.text(cx, cy - 0.04, f'u={u:.2f}', ha='center', va='center',
                  fontsize=8, color='white')
        ax_b.text(cx, cy - 0.16, f'{decision}', ha='center', va='top',
                  fontsize=8, fontweight='bold',
                  color=fcolor)

    # Draw ZZ edges with thickness proportional to coupling strength
    edges = [
        ((0,0), (0,1), 'h'), ((1,0), (1,1), 'h'),  # horizontal
        ((0,0), (1,0), 'v'), ((0,1), (1,1), 'v'),   # vertical
    ]
    max_C = max(abs(C_h_ds).max(), abs(C_v_ds).max(), 1e-10)
    for (i1,j1), (i2,j2), direction in edges:
        p1 = cell_positions[(i1,j1)]
        p2 = cell_positions[(i2,j2)]
        if direction == 'h':
            c_val = C_h_ds[i1, j1]
        else:
            c_val = C_v_ds[i1, j1]
        lw = 1 + 5 * abs(c_val) / max_C
        color = '#4CAF50' if abs(c_val) > 0.1 * max_C else '#BDBDBD'
        ax_b.plot([p1[0], p2[0]], [p1[1], p2[1]],
                  linewidth=lw, color=color, alpha=0.7, zorder=0)
        # Label edge strength
        mx, my = 0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1])
        offset = 0.04 if direction == 'h' else -0.08
        ax_b.text(mx + (0 if direction == 'h' else offset),
                  my + (offset if direction == 'h' else 0),
                  f'ZZ={c_val:.1f}', fontsize=7, ha='center', va='center',
                  color='darkgreen',
                  bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))

    ax_b.set_title(f'B) Hamiltonian (2×2, σ={SIGMA:.2f})', fontsize=9)
    # Legend
    ax_b.text(0.5, -0.02, 'Dashed box = borderline cell | '
              'u = uncertainty weight | Edge width ∝ |ZZ|',
              ha='center', fontsize=8, style='italic',
              transform=ax_b.transAxes)

    # ── Panel C: Decision Comparison ──────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])

    # 3 sub-grids: Classical | QAOA | Ground Truth
    methods = [
        ('Classical', cl_refine, COLORS['classical']),
        ('Q-HAS (QAOA)', qa_refine, COLORS['qaoa']),
        ('Ground Truth', gt_refine, COLORS['ground_truth']),
    ]

    for idx, (name, decision, base_color) in enumerate(methods):
        x_off = idx * 3.2
        for i in range(N_BLOCKS):
            for j in range(N_BLOCKS):
                x = x_off + j * 1.1
                y = (N_BLOCKS - 1 - i) * 1.1
                if decision[i, j]:
                    fc = '#E91E63'
                    txt = 'R'
                else:
                    fc = '#2196F3'
                    txt = 'S'
                # Highlight if this is the borderline cell
                lw = 3 if (i, j) == cell else 1
                ec = 'gold' if (i, j) == cell else 'black'
                rect = plt.Rectangle((x, y), 1.0, 1.0,
                                      facecolor=fc, edgecolor=ec,
                                      linewidth=lw, alpha=0.8)
                ax_c.add_patch(rect)
                ax_c.text(x + 0.5, y + 0.5, txt,
                          ha='center', va='center',
                          fontsize=14, fontweight='bold', color='white')

                # Show score/probability
                if idx == 0:
                    val = scores[i, j]
                elif idx == 1:
                    val = qaoa_scores[i, j] if qaoa_scores is not None else scores[i, j]
                else:
                    val = gt_blocks[i, j]
                ax_c.text(x + 0.5, y + 0.15, f'{val:.2f}',
                          ha='center', va='center', fontsize=8, color='white')

        ax_c.text(x_off + 0.55, -0.4, name, ha='center', fontsize=10,
                  fontweight='bold', color=base_color)

    # Check corrections
    n_corrections = 0
    correction_details = []
    for i in range(N_BLOCKS):
        for j in range(N_BLOCKS):
            cl_correct = cl_refine[i,j] == gt_refine[i,j]
            qa_correct = qa_refine[i,j] == gt_refine[i,j]
            if qa_correct and not cl_correct:
                n_corrections += 1
                correction_details.append(f"({i},{j})")

    ax_c.set_xlim(-0.3, 9.8)
    ax_c.set_ylim(-0.8, N_BLOCKS * 1.1 + 0.3)
    ax_c.set_aspect('equal')
    ax_c.axis('off')

    corr_text = (f'QAOA corrections: {n_corrections} cell(s) {", ".join(correction_details)}'
                 if n_corrections > 0
                 else 'No corrections in this case (QAOA and classical agree)')
    color_corr = '#4CAF50' if n_corrections > 0 else '#FF9800'
    ax_c.set_title(f'C) Decisions (R=Refine, S=Skip)\n{corr_text}',
                   fontsize=9, color=color_corr)
    ax_c.text(0.5, -0.08,
              'Gold border = borderline cell | '
              'Bottom number = score (Classical/QAOA) or error (GT)',
              ha='center', fontsize=8, style='italic',
              transform=ax_c.transAxes)

    # ── Panel D: Scaling Argument ─────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])

    grid_sizes = np.arange(2, 17)
    # At NxN grid:
    # - Total cells: N^2
    # - Boundary cells (score within sigma of threshold): ~4*(N-1) for a
    #   contiguous anomaly region (perimeter scales as O(N))
    # - ZZ edges: 2*N^2 (periodic) or 2*N*(N-1) (open)
    # - Each boundary cell has 2-4 ZZ-coupled neighbors
    # - Correction potential: proportional to boundary cells

    total_cells = grid_sizes ** 2
    total_edges = 2 * grid_sizes ** 2  # periodic BC
    # Model: anomaly region is ~40% of domain (circular blob)
    # Boundary length ~ pi * sqrt(0.4) * N ~ 1.12 * N
    boundary_cells = np.minimum(1.12 * grid_sizes, total_cells * 0.3)
    boundary_cells = np.maximum(boundary_cells, 1)  # at least 1
    correction_potential = boundary_cells / total_cells * 100  # percentage

    # Classical error rate at boundary: each boundary cell has ~50% chance
    # of wrong decision (score near threshold → coin flip)
    classical_errors = 0.5 * boundary_cells
    # QAOA correction rate: ~60-80% of boundary errors corrected by ZZ
    qaoa_corrections = 0.7 * classical_errors

    ax_d.plot(grid_sizes, classical_errors, 'o-',
              color=COLORS['classical'], linewidth=2, markersize=6,
              label='Classical boundary errors')
    ax_d.plot(grid_sizes, qaoa_corrections, 's-',
              color=COLORS['qaoa'], linewidth=2, markersize=6,
              label='QAOA corrections (est. 70%)')
    ax_d.fill_between(grid_sizes,
                       0.5 * classical_errors, 0.9 * classical_errors,
                       color=COLORS['qaoa'], alpha=0.15,
                       label='Correction range (50-90%)')

    # Annotate key points
    ax_d.axvline(x=2, color='gray', linestyle=':', alpha=0.5)
    ax_d.text(2.1, classical_errors[0] + 0.3,
              f'2×2\n(this work)\n{int(boundary_cells[0])} boundary cell',
              fontsize=8, color='gray')

    idx_8 = 6  # grid_size = 8
    ax_d.axvline(x=8, color='gray', linestyle=':', alpha=0.5)
    ax_d.text(8.1, classical_errors[idx_8] + 0.5,
              f'8×8\n{int(boundary_cells[idx_8])} boundary cells\n'
              f'~{int(qaoa_corrections[idx_8])} corrections',
              fontsize=8, color='gray')

    ax_d.set_xlabel('VQA Grid Size (N×N)', fontsize=8)
    ax_d.set_ylabel('Number of Cells', fontsize=8)
    ax_d.set_title('D) Scaling: Boundary Corrections vs Grid Size', fontsize=9)
    ax_d.legend(fontsize=6, loc='upper left')
    ax_d.set_xlim(1.5, 16.5)
    ax_d.tick_params(labelsize=7)

    # Add qubit count on secondary axis
    ax_d2 = ax_d.twiny()
    qubit_counts = 2 * grid_sizes ** 2
    ax_d2.set_xlim(ax_d.get_xlim())
    tick_positions = [2, 4, 8, 12, 16]
    ax_d2.set_xticks(tick_positions)
    ax_d2.set_xticklabels([f'{2*n**2}q' for n in tick_positions], fontsize=6)
    ax_d2.set_xlabel('Qubits (2N²)', fontsize=7)

    fig.suptitle('Boundary Correction Proof-of-Concept',
                 fontsize=11, fontweight='bold', y=0.97)

    return fig


def main():
    print("=" * 60)
    print("Figure 14: Boundary Correction Proof-of-Concept")
    print("=" * 60)

    # ── Step 1: Search for natural boundary correction cases ──
    print("\n[1/3] Searching MHD scenarios for boundary cases...")
    cases = find_boundary_cases()

    if not cases:
        print("  No borderline cases found even with widened margin.")
        print("  This likely means all block scores are far from threshold.")
        print("  Generating figure with the closest-to-threshold cell anyway.")
        # Fallback: pick the scenario with the lowest max-score cell
        fallback_cases = []
        for label, scenario, n_steps in SCENARIOS:
            try:
                sim, Phi_prev = make_sim(N, scenario, n_steps)
            except Exception:
                continue
            physics_state = sim.get_fluxes()
            full_score = AngleMapper.classical_score(physics_state)
            from Simulation.RescaleArrays import _process_score
            scores = _process_score(full_score, True, N_BLOCKS)
            gt_err = ground_truth_errors(sim, N)
            bh, bw = N // N_BLOCKS, N // N_BLOCKS
            gt_blocks = np.zeros((N_BLOCKS, N_BLOCKS))
            for bi in range(N_BLOCKS):
                for bj in range(N_BLOCKS):
                    gt_blocks[bi, bj] = np.mean(gt_err[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw])
            gt_median = np.median(gt_blocks)
            gt_refine = gt_blocks > gt_median
            cl_refine = scores > THRESHOLD
            # Pick cell closest to threshold
            for i in range(N_BLOCKS):
                for j in range(N_BLOCKS):
                    dist = abs(scores[i, j] - THRESHOLD)
                    neighbors = []
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = (i+di) % N_BLOCKS, (j+dj) % N_BLOCKS
                        neighbors.append(scores[ni, nj])
                    fallback_cases.append({
                        'label': label, 'scenario': scenario, 'n_steps': n_steps,
                        'sim': sim, 'Phi_prev': Phi_prev, 'scores': scores,
                        'gt_blocks': gt_blocks, 'gt_refine': gt_refine,
                        'gt_err': gt_err, 'cell': (i, j),
                        'cell_score': float(scores[i, j]),
                        'dist_to_threshold': dist,
                        'n_neighbors_above': sum(1 for s in neighbors if s > THRESHOLD),
                        'classical_wrong': cl_refine[i,j] != gt_refine[i,j],
                        'neighbor_scores': neighbors,
                    })
        fallback_cases.sort(key=lambda c: c['dist_to_threshold'])
        cases = fallback_cases

    # Report findings
    print(f"\n  Found {len(cases)} borderline cases")
    for i, c in enumerate(cases[:5]):
        print(f"  #{i}: {c['label']} cell={c['cell']} "
              f"score={c['cell_score']:.3f} "
              f"dist={c['dist_to_threshold']:.3f} "
              f"neighbors_above={c['n_neighbors_above']} "
              f"classical_wrong={c['classical_wrong']}")

    # Pick best case (classical wrong AND closest to threshold)
    best = cases[0]
    print(f"\n  Selected: {best['label']} cell={best['cell']} "
          f"score={best['cell_score']:.3f}")

    # ── Step 2: Run QAOA ──
    print("\n[2/3] Running QAOA on selected case...")
    qaoa_scores = run_qaoa_for_case(best)
    if qaoa_scores is not None:
        print(f"  QAOA scores:\n{qaoa_scores}")
        print(f"  Classical scores:\n{best['scores']}")
        qa_refine = qaoa_scores > THRESHOLD
        cl_refine = best['scores'] > THRESHOLD
        print(f"  Classical decisions:\n{cl_refine}")
        print(f"  QAOA decisions:\n{qa_refine}")
        print(f"  Ground truth:\n{best['gt_refine']}")

        # Count corrections
        for i in range(N_BLOCKS):
            for j in range(N_BLOCKS):
                cl_ok = cl_refine[i,j] == best['gt_refine'][i,j]
                qa_ok = qa_refine[i,j] == best['gt_refine'][i,j]
                if qa_ok and not cl_ok:
                    print(f"  ** CORRECTION at ({i},{j}): "
                          f"score={best['scores'][i,j]:.3f} -> "
                          f"QAOA={qaoa_scores[i,j]:.3f}")
    else:
        print("  QAOA failed, using coefficient analysis only")

    # ── Step 3: Generate figure ──
    print("\n[3/3] Generating figure...")
    fig = make_figure(best, qaoa_scores)
    out_path = os.path.join(FIG_DIR, 'fig14_boundary_correction.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Also try a second-best case if available
    if len(cases) > 1 and cases[1]['classical_wrong']:
        alt = cases[1]
        print(f"\n  Also running alternative: {alt['label']} "
              f"cell={alt['cell']} score={alt['cell_score']:.3f}")
        alt_qaoa = run_qaoa_for_case(alt)
        if alt_qaoa is not None:
            fig2 = make_figure(alt, alt_qaoa)
            out2 = os.path.join(FIG_DIR, 'fig14_boundary_correction_alt.png')
            fig2.savefig(out2, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print(f"  Saved: {out2}")

    print("\nDone.")


if __name__ == '__main__':
    main()
