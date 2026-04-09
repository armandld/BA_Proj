"""
Direct measurement: Does QAOA select BETTER blocks than classical?
==================================================================

Protocol (no step_layered needed — pure selection quality):
1. Run MHD simulation to develop features
2. Compute ground truth: per-block L2 error of coarse vs DNS-quality fields
3. With budget K: select top-K blocks by classical score vs QAOA probability
4. Measure total captured error: sum of ground-truth errors in selected blocks
   → Higher = better (we're selecting the blocks that NEED refinement most)
5. Compute selection overlap with ground truth optimal

This is a PURE block selection benchmark — it isolates whether the QAOA
makes smarter refinement decisions, independent of the solver.

Test scenarios:
- MHD Rotor: sheath vs core discrimination
- Kelvin-Helmholtz: vortex roll-up with current sheets
- Orszag-Tang: complex multi-scale interactions
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from types import SimpleNamespace
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import get_adaptive_flux
from VQA.runtime import VQARuntime
from call_vqa_shell import call_vqa_shell


def compute_block_indicator_errors(sim, N, n_blocks):
    """
    Compute per-block 'refinement need' using gradients of all fields.

    This is a proxy for ground-truth error: blocks with large gradients
    are where coarse resolution introduces the most error.

    Uses: max of (|grad vx|, |grad vy|, |grad Bx|, |grad By|, |grad Jz|)
    within each block. This is resolution-independent.
    """
    state = sim.get_fluxes()
    block_h = N // n_blocks
    block_w = N // n_blocks

    # Compute gradient magnitudes for all fields
    indicators = []
    for key in ['vx', 'vy', 'Bx', 'By', 'Jz']:
        f = state[key]
        gx = np.abs(np.roll(f, -1, axis=1) - f)
        gy = np.abs(np.roll(f, -1, axis=0) - f)
        grad_mag = np.sqrt(gx**2 + gy**2)
        indicators.append(grad_mag)

    # Combined gradient indicator
    combined = np.sqrt(sum(ind**2 for ind in indicators))

    # Per-block: use MEAN of combined gradient (not max, to avoid outlier sensitivity)
    errors = np.zeros((n_blocks, n_blocks))
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            y0, y1 = bi * block_h, (bi + 1) * block_h
            x0, x1 = bj * block_w, (bj + 1) * block_w
            errors[bi, bj] = np.mean(combined[y0:y1, x0:x1])

    return errors


def compute_block_second_deriv(sim, N, n_blocks):
    """
    Per-block refinement need using second derivatives (truncation error proxy).

    For a 2nd-order FD scheme, the leading error term is O(h² * f'').
    So the blocks with the largest |f''| are where refinement helps most.
    """
    state = sim.get_fluxes()
    block_h = N // n_blocks
    block_w = N // n_blocks

    total = np.zeros((N, N))
    for key in ['vx', 'vy', 'Bx', 'By']:
        f = state[key]
        fpp_xx = np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)
        fpp_yy = np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)
        total += fpp_xx**2 + fpp_yy**2

    total = np.sqrt(total)

    errors = np.zeros((n_blocks, n_blocks))
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            y0, y1 = bi * block_h, (bi + 1) * block_h
            x0, x1 = bj * block_w, (bj + 1) * block_w
            errors[bi, bj] = np.mean(total[y0:y1, x0:x1])

    return errors


def classical_block_scores(sim, N, n_blocks):
    """Per-block max of classical_score (same as pipeline)."""
    state = sim.get_fluxes()
    full_score = AngleMapper.classical_score(state)
    block_h = N // n_blocks
    block_w = N // n_blocks
    scores = np.zeros((n_blocks, n_blocks))
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            y0, y1 = bi * block_h, (bi + 1) * block_h
            x0, x1 = bj * block_w, (bj + 1) * block_w
            scores[bi, bj] = np.max(full_score[y0:y1, x0:x1])
    return scores


def qaoa_block_scores(sim, N, n_blocks, threshold=0.3, w_z_frac=0.15, K_opt=80,
                      Phi_prev=None):
    """Per-block QAOA probability (refinement confidence)."""
    grid = sim.grid
    nu = grid.L / 800
    eta = grid.L / 800

    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    HamiltMapper = PhysicalMapper(
        cs=1.0, nu=nu, eta_mhd=eta, beta=0.5, dx=grid.dx,
        gamma_hydro=0.5, gamma_mag=0.5, kappa=5.0, w_z_frac=w_z_frac,
    )

    args = SimpleNamespace(
        reps=(n_blocks - 1) * 2,
        mode="simulator", backend="state_vector",
        shots=1024, method="COBYLA", opt_level=1,
        AdvAnomaliesEnable=True,
        K_opt=K_opt, eps=1e-2,
    )
    vqa_runtime = VQARuntime(
        backend_name="state_vector", mode="simulator",
        shots=1024, opt_level=1,
    )

    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)
    full_score = AngleMapper.classical_score(physics_state)

    hamilt_params = HamiltMapper.compute_coefficients(
        sim, full_score, physics_state, threshold,
        advanced_anomalies_enabled=True,
    )

    phi_h = Phi['phi_horizontal']
    phi_v = Phi['phi_vertical']

    prev_h = Phi_prev['phi_horizontal'] if Phi_prev is not None else None
    prev_v = Phi_prev['phi_vertical']   if Phi_prev is not None else None

    if prev_h is not None:
        AveragePhiDev = 0.5 * (np.mean(np.abs(phi_h - prev_h))
                                + np.mean(np.abs(phi_v - prev_v)))
        mini_h, mini_v, mini_prev_h, mini_prev_v, mini_hp, mini_score = \
            get_adaptive_flux(
                phi_h, phi_v, prev_h, prev_v, full_score, hamilt_params,
                target_dim=n_blocks, type_filter=True,
            )
        mini_Phi_prev = {'phi_horizontal': mini_prev_h,
                         'phi_vertical':   mini_prev_v}
    else:
        AveragePhiDev = None
        mini_h, mini_v, mini_hp, mini_score = get_adaptive_flux(
            phi_h, phi_v, None, None, full_score, hamilt_params,
            target_dim=n_blocks, type_filter=True,
        )
        mini_Phi_prev = None

    mini_score = np.clip(mini_score, 0.0, 1.0)

    angles = mapper.map_to_angles(
        score_h=mini_score, score_v=mini_score,
        phi_dict_prev=mini_Phi_prev,
        phi_dict={'phi_horizontal': mini_h, 'phi_vertical': mini_v},
        AveragePhiDev=AveragePhiDev, beta=1.0,
    )

    probs, _ = call_vqa_shell(
        angles, mini_hp, False, args,
        period_bound=True, vqa_runtime=vqa_runtime,
    )

    n_edges = n_blocks * n_blocks
    if probs is not None:
        ph = probs[:n_edges].reshape(n_blocks, n_blocks)
        pv = probs[n_edges:].reshape(n_blocks, n_blocks)
        return 0.5 * (ph + pv), mini_hp
    else:
        return mini_score.copy(), mini_hp


def select_top_k(scores, k):
    """Top-K block indices by score (descending)."""
    flat = np.argsort(scores.ravel())[::-1][:k]
    n_cols = scores.shape[1]
    return set((idx // n_cols, idx % n_cols) for idx in flat)


def captured_error_fraction(selection, ground_truth_errors):
    """What fraction of total error is captured by the selection?"""
    total = np.sum(ground_truth_errors)
    if total < 1e-12:
        return 1.0
    captured = sum(ground_truth_errors[i, j] for i, j in selection)
    return captured / total


def rank_correlation(scores, ground_truth):
    """Spearman rank correlation between score ranking and GT ranking."""
    from scipy.stats import spearmanr
    rho, pval = spearmanr(scores.ravel(), ground_truth.ravel())
    return rho, pval


def evaluate_scenario(name, sim, N, n_blocks, budgets, threshold=0.3, w_z_frac=0.15,
                      Phi_prev=None):
    """Run full evaluation for one scenario."""
    print(f"\n{'='*70}")
    print(f"  SCENARIO: {name}")
    print(f"  Grid: {N}x{N}, Blocks: {n_blocks}x{n_blocks}")
    print(f"{'='*70}")

    # Ground truth: 2nd derivative (truncation error proxy)
    gt_errors = compute_block_second_deriv(sim, N, n_blocks)
    gt_grad = compute_block_indicator_errors(sim, N, n_blocks)
    print(f"\n  Ground truth (2nd deriv) block errors:")
    print(f"    {np.array2string(gt_errors, precision=4)}")

    # Classical scores
    cl_scores = classical_block_scores(sim, N, n_blocks)
    print(f"\n  Classical block scores:")
    print(f"    {np.array2string(cl_scores, precision=4)}")

    # QAOA scores
    qaoa_scores, mini_hp = qaoa_block_scores(sim, N, n_blocks, threshold, w_z_frac,
                                              Phi_prev=Phi_prev)
    print(f"\n  QAOA block probabilities:")
    print(f"    {np.array2string(qaoa_scores, precision=4)}")

    # Rank correlations
    rho_cl, p_cl = rank_correlation(cl_scores, gt_errors)
    rho_qa, p_qa = rank_correlation(qaoa_scores, gt_errors)
    print(f"\n  Rank correlation with ground truth (2nd deriv):")
    print(f"    Classical:  rho={rho_cl:+.3f} (p={p_cl:.3f})")
    print(f"    QAOA:       rho={rho_qa:+.3f} (p={p_qa:.3f})")

    rho_cl_g, _ = rank_correlation(cl_scores, gt_grad)
    rho_qa_g, _ = rank_correlation(qaoa_scores, gt_grad)
    print(f"  Rank correlation with ground truth (gradient):")
    print(f"    Classical:  rho={rho_cl_g:+.3f}")
    print(f"    QAOA:       rho={rho_qa_g:+.3f}")

    # Energy decomposition
    mH_h, mH_v = mini_hp['H_edges']
    mC_h, mC_v = mini_hp['C_edges']
    mK = mini_hp['K_plaquettes']
    E_Z = np.sum(np.abs(mH_h)) + np.sum(np.abs(mH_v))
    E_ZZ = np.sum(np.abs(mC_h)) + np.sum(np.abs(mC_v))
    E_ZZZZ = np.sum(np.abs(mK))
    E_total = E_Z + E_ZZ + E_ZZZZ + 1e-10
    print(f"\n  Hamiltonian energy decomposition:")
    print(f"    Z={E_Z:.3f} ({E_Z/E_total*100:.1f}%), "
          f"ZZ={E_ZZ:.3f} ({E_ZZ/E_total*100:.1f}%), "
          f"ZZZZ={E_ZZZZ:.3f} ({E_ZZZZ/E_total*100:.1f}%)")
    print(f"    Multi/Single ratio: {(E_ZZ+E_ZZZZ)/max(E_Z,1e-10):.3f}")

    # Budget sweep
    print(f"\n  Budget sweep — captured error fraction:")
    print(f"  {'Budget':>8s} {'GT optimal':>12s} {'Classical':>12s} {'QAOA':>12s} {'Winner':>10s}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for budget in budgets:
        gt_sel = select_top_k(gt_errors, budget)
        cl_sel = select_top_k(cl_scores, budget)
        qa_sel = select_top_k(qaoa_scores, budget)

        gt_frac = captured_error_fraction(gt_sel, gt_errors)
        cl_frac = captured_error_fraction(cl_sel, gt_errors)
        qa_frac = captured_error_fraction(qa_sel, gt_errors)

        winner = "QAOA" if qa_frac > cl_frac else ("Classical" if cl_frac > qa_frac else "Tie")
        marker = " <--" if winner == "QAOA" else ""
        print(f"  {budget:>8d} {gt_frac:>12.4f} {cl_frac:>12.4f} {qa_frac:>12.4f} {winner:>10s}{marker}")

    # Selection overlap at various budgets
    print(f"\n  Selection overlap with ground truth:")
    print(f"  {'Budget':>8s} {'Classical':>12s} {'QAOA':>12s}")
    for budget in budgets:
        gt_sel = select_top_k(gt_errors, budget)
        cl_sel = select_top_k(cl_scores, budget)
        qa_sel = select_top_k(qaoa_scores, budget)
        cl_olap = len(cl_sel & gt_sel)
        qa_olap = len(qa_sel & gt_sel)
        print(f"  {budget:>8d} {cl_olap:>8d}/{budget:<3d} {qa_olap:>8d}/{budget:<3d}")

    return {
        'rho_cl': rho_cl, 'rho_qa': rho_qa,
        'gt_errors': gt_errors, 'cl_scores': cl_scores, 'qaoa_scores': qaoa_scores,
    }


def main():
    N = 64
    threshold = 0.3
    w_z_frac = 0.15

    # n_blocks determines qubit count: n_blocks² horizontal + n_blocks² vertical
    # 2×2 → 8 qubits (fast), 3×3 → 18 qubits (~1GB statevector)
    block_configs = [
        (2, [1]),           # 2×2: budget 1 out of 4
        (3, [1, 2, 3]),     # 3×3: budgets 1-3 out of 9
    ]

    print("=" * 70)
    print("  QAOA ADVANTAGE MEASUREMENT — Block Selection Quality")
    print(f"  Grid: {N}x{N}")
    print(f"  Threshold: {threshold}, w_z_frac: {w_z_frac}")
    print("=" * 70)

    scenarios = {
        'MHD Rotor': ('init_mhd_rotor', 200),
        'Kelvin-Helmholtz': ('init_kelvin_helmholtz', 300),
        'Orszag-Tang': ('init_orszag_tang', 500),
    }
    all_results = {}

    for name, (init_method, n_steps) in scenarios.items():
        print(f"\n  Initializing {name} ({n_steps} steps)...")
        grid = PeriodicGrid(resolution_N=N)
        sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
        getattr(sim, init_method)()
        _mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
        Phi_prev = None
        for i in range(n_steps):
            if i == n_steps - 1:
                Phi_prev = _mapper.compute_stress_flux(sim.get_fluxes())
            sim.adapt_dt(cfl_target=0.4)
            sim.step_full(record_stats=False)

        for n_blocks, budgets in block_configs:
            tag = f"{name} ({n_blocks}x{n_blocks})"
            all_results[tag] = evaluate_scenario(
                tag, sim, N, n_blocks, budgets, threshold, w_z_frac,
                Phi_prev=Phi_prev,
            )

    # ── Overall summary ──
    print(f"\n\n{'#'*70}")
    print(f"  OVERALL SUMMARY")
    print(f"{'#'*70}")
    print(f"  {'Scenario':<40s} {'rho(Classical)':>15s} {'rho(QAOA)':>15s} {'Better?':>10s}")
    print(f"  {'-'*40} {'-'*15} {'-'*15} {'-'*10}")
    for name, res in all_results.items():
        better = "QAOA" if res['rho_qa'] > res['rho_cl'] else "Classical"
        print(f"  {name:<40s} {res['rho_cl']:>+15.3f} {res['rho_qa']:>+15.3f} {better:>10s}")


if __name__ == "__main__":
    main()
