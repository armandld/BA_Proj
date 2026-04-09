"""
Test: Can QAOA beat classical under noise or for early anomaly detection?
=========================================================================

Hypothesis 1 — NOISE ROBUSTNESS:
  When classical scores are noisy (sensor noise, under-resolved grid),
  the QAOA's ZZ coupling enforces spatial coherence: "neighbors should
  agree on refinement." This acts as a denoiser. Test: add increasing
  noise to score maps and compare selection quality.

Hypothesis 2 — EARLY DETECTION:
  When an anomaly is growing fast, the current classical score is weak
  but the spatial PATTERN (expanding ring, front) is detectable by
  multi-body terms. Test: evaluate selection at early time t₁, compare
  against ground truth at later time t₂ when the anomaly is fully developed.

For both: measure Spearman ρ and captured error fraction vs ground truth.
"""
import sys, os
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from types import SimpleNamespace
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import get_adaptive_flux
from VQA.runtime import VQARuntime
from call_vqa_shell import call_vqa_shell


# ═════════════════════════════════════════════════════════════════════
#  Shared utilities
# ═════════════════════════════════════════════════════════════════════

def ground_truth_errors(sim, N, n_blocks):
    """Per-block 2nd derivative (truncation error proxy)."""
    state = sim.get_fluxes()
    bh = N // n_blocks
    bw = N // n_blocks
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
            y0, y1 = bi * bh, (bi + 1) * bh
            x0, x1 = bj * bw, (bj + 1) * bw
            errors[bi, bj] = np.mean(total[y0:y1, x0:x1])
    return errors


def classical_block_scores(sim, N, n_blocks):
    """Per-block max of classical_score."""
    state = sim.get_fluxes()
    full_score = AngleMapper.classical_score(state)
    bh = N // n_blocks
    bw = N // n_blocks
    scores = np.zeros((n_blocks, n_blocks))
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            y0, y1 = bi * bh, (bi + 1) * bh
            x0, x1 = bj * bw, (bj + 1) * bw
            scores[bi, bj] = np.max(full_score[y0:y1, x0:x1])
    return scores


def noisy_classical_scores(sim, N, n_blocks, noise_std, rng):
    """Classical scores with additive Gaussian noise on the FULL grid before pooling."""
    state = sim.get_fluxes()
    full_score = AngleMapper.classical_score(state)
    # Add noise at cell level (before block pooling)
    noisy = full_score + noise_std * rng.standard_normal(full_score.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    bh = N // n_blocks
    bw = N // n_blocks
    scores = np.zeros((n_blocks, n_blocks))
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            y0, y1 = bi * bh, (bi + 1) * bh
            x0, x1 = bj * bw, (bj + 1) * bw
            scores[bi, bj] = np.max(noisy[y0:y1, x0:x1])
    return scores


def qaoa_block_scores(sim, N, n_blocks, threshold=0.3, w_z_frac=0.15,
                      K_opt=80, noise_std=0.0, rng=None, Phi_prev=None):
    """
    Per-block QAOA probability.
    If noise_std > 0, adds noise to the classical score BEFORE computing
    Hamiltonian Z biases (simulating noisy sensor input).
    """
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

    # Inject noise into score map (same noise the classical selector would see)
    if noise_std > 0 and rng is not None:
        full_score = full_score + noise_std * rng.standard_normal(full_score.shape)
        full_score = np.clip(full_score, 0.0, 1.0)

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
        return 0.5 * (ph + pv)
    else:
        return mini_score.copy()


def select_top_k(scores, k):
    flat = np.argsort(scores.ravel())[::-1][:k]
    nc = scores.shape[1]
    return set((idx // nc, idx % nc) for idx in flat)


def captured_fraction(selection, gt_errors):
    total = np.sum(gt_errors)
    if total < 1e-12:
        return 1.0
    return sum(gt_errors[i, j] for i, j in selection) / total


# ═════════════════════════════════════════════════════════════════════
#  TEST 1: NOISE ROBUSTNESS
# ═════════════════════════════════════════════════════════════════════

def test_noise_robustness():
    print("\n" + "=" * 70)
    print("  TEST 1: NOISE ROBUSTNESS")
    print("  Can QAOA's spatial coupling denoise corrupted score maps?")
    print("=" * 70)

    N = 64
    n_blocks = 3
    threshold = 0.3
    w_z_frac = 0.15
    budget = 2
    n_trials = 5  # average over random seeds

    scenarios = {
        'MHD Rotor': ('init_mhd_rotor', 200),
        'Orszag-Tang': ('init_orszag_tang', 500),
    }

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    for scen_name, (init_method, n_steps) in scenarios.items():
        print(f"\n  --- {scen_name} ({n_blocks}x{n_blocks}, budget={budget}) ---")

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

        gt = ground_truth_errors(sim, N, n_blocks)
        gt_sel = select_top_k(gt, budget)
        gt_frac = captured_fraction(gt_sel, gt)

        print(f"  Ground truth errors:\n    {np.array2string(gt, precision=4)}")
        print(f"  Optimal captured fraction (budget={budget}): {gt_frac:.4f}")
        print()
        print(f"  {'Noise σ':>10s} {'Classical ρ':>13s} {'QAOA ρ':>13s} "
              f"{'Cl. captured':>13s} {'QA. captured':>13s} {'Winner':>10s}")
        print(f"  {'-'*10} {'-'*13} {'-'*13} {'-'*13} {'-'*13} {'-'*10}")

        for noise_std in noise_levels:
            cl_rhos = []
            qa_rhos = []
            cl_fracs = []
            qa_fracs = []

            for seed in range(n_trials):
                rng = np.random.default_rng(42 + seed)

                # Classical with noise
                cl_scores = noisy_classical_scores(sim, N, n_blocks, noise_std, rng)
                rho_cl, _ = spearmanr(cl_scores.ravel(), gt.ravel())
                cl_sel = select_top_k(cl_scores, budget)
                cl_rhos.append(rho_cl)
                cl_fracs.append(captured_fraction(cl_sel, gt))

                # QAOA with same noise
                rng2 = np.random.default_rng(42 + seed)  # same noise realization
                qa_scores = qaoa_block_scores(
                    sim, N, n_blocks, threshold, w_z_frac, K_opt=80,
                    noise_std=noise_std, rng=rng2, Phi_prev=Phi_prev,
                )
                rho_qa, _ = spearmanr(qa_scores.ravel(), gt.ravel())
                qa_rhos.append(rho_qa)
                qa_sel = select_top_k(qa_scores, budget)
                qa_fracs.append(captured_fraction(qa_sel, gt))

            mean_cl_rho = np.mean(cl_rhos)
            mean_qa_rho = np.mean(qa_rhos)
            mean_cl_frac = np.mean(cl_fracs)
            mean_qa_frac = np.mean(qa_fracs)
            winner = "QAOA" if mean_qa_frac > mean_cl_frac + 0.005 else (
                "Classical" if mean_cl_frac > mean_qa_frac + 0.005 else "Tie")
            marker = " <--" if winner == "QAOA" else ""
            print(f"  {noise_std:>10.2f} {mean_cl_rho:>+13.3f} {mean_qa_rho:>+13.3f} "
                  f"{mean_cl_frac:>13.4f} {mean_qa_frac:>13.4f} {winner:>10s}{marker}")


# ═════════════════════════════════════════════════════════════════════
#  TEST 2: EARLY ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════

def test_early_detection():
    print("\n\n" + "=" * 70)
    print("  TEST 2: EARLY ANOMALY DETECTION")
    print("  Can QAOA predict where refinement will be needed in the FUTURE?")
    print("=" * 70)

    N = 64
    n_blocks = 3
    threshold = 0.3
    w_z_frac = 0.15
    budget = 2

    # For each scenario: evaluate at early time, compare against late-time GT
    scenarios = {
        'MHD Rotor': {
            'init': 'init_mhd_rotor',
            'early_steps': [20, 50, 100],
            'late_steps': 300,  # fully developed sheath
        },
        'Kelvin-Helmholtz': {
            'init': 'init_kelvin_helmholtz',
            'early_steps': [30, 80, 150],
            'late_steps': 400,
        },
    }

    for scen_name, cfg in scenarios.items():
        print(f"\n  --- {scen_name} ({n_blocks}x{n_blocks}, budget={budget}) ---")

        # First run to late time for ground truth
        grid = PeriodicGrid(resolution_N=N)
        sim_late = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
        getattr(sim_late, cfg['init'])()
        for _ in range(cfg['late_steps']):
            sim_late.adapt_dt(cfl_target=0.4)
            sim_late.step_full(record_stats=False)

        gt_late = ground_truth_errors(sim_late, N, n_blocks)
        gt_sel = select_top_k(gt_late, budget)
        gt_frac = captured_fraction(gt_sel, gt_late)

        print(f"  Late-time GT errors (t={cfg['late_steps']} steps):")
        print(f"    {np.array2string(gt_late, precision=4)}")
        print(f"  Optimal captured fraction: {gt_frac:.4f}")
        print()
        print(f"  {'Early steps':>12s} {'Classical ρ':>13s} {'QAOA ρ':>13s} "
              f"{'Cl. captured':>13s} {'QA. captured':>13s} {'Winner':>10s}")
        print(f"  {'-'*12} {'-'*13} {'-'*13} {'-'*13} {'-'*13} {'-'*10}")

        for early_steps in cfg['early_steps']:
            # Run to early time (capture Phi_prev for Phase Boost)
            grid_e = PeriodicGrid(resolution_N=N)
            sim_early = MHDSolver(grid_e, dt=1e-3, Re=800, Rm=800)
            getattr(sim_early, cfg['init'])()
            _mapper_e = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
            Phi_prev_e = None
            for i in range(early_steps):
                if i == early_steps - 1:
                    Phi_prev_e = _mapper_e.compute_stress_flux(sim_early.get_fluxes())
                sim_early.adapt_dt(cfl_target=0.4)
                sim_early.step_full(record_stats=False)

            # Classical at early time → rank against late-time GT
            cl_scores = classical_block_scores(sim_early, N, n_blocks)
            rho_cl, _ = spearmanr(cl_scores.ravel(), gt_late.ravel())
            cl_sel = select_top_k(cl_scores, budget)
            cl_frac = captured_fraction(cl_sel, gt_late)

            # QAOA at early time → rank against late-time GT
            qa_scores = qaoa_block_scores(sim_early, N, n_blocks, threshold, w_z_frac,
                                          Phi_prev=Phi_prev_e)
            rho_qa, _ = spearmanr(qa_scores.ravel(), gt_late.ravel())
            qa_sel = select_top_k(qa_scores, budget)
            qa_frac = captured_fraction(qa_sel, gt_late)

            winner = "QAOA" if qa_frac > cl_frac + 0.005 else (
                "Classical" if cl_frac > qa_frac + 0.005 else "Tie")
            marker = " <--" if winner == "QAOA" else ""
            print(f"  {early_steps:>12d} {rho_cl:>+13.3f} {rho_qa:>+13.3f} "
                  f"{cl_frac:>13.4f} {qa_frac:>13.4f} {winner:>10s}{marker}")

        # Also show what classical and QAOA see at earliest time
        grid_e = PeriodicGrid(resolution_N=N)
        sim_e = MHDSolver(grid_e, dt=1e-3, Re=800, Rm=800)
        getattr(sim_e, cfg['init'])()
        _mapper_e2 = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
        Phi_prev_e2 = None
        es0 = cfg['early_steps'][0]
        for i in range(es0):
            if i == es0 - 1:
                Phi_prev_e2 = _mapper_e2.compute_stress_flux(sim_e.get_fluxes())
            sim_e.adapt_dt(cfl_target=0.4)
            sim_e.step_full(record_stats=False)

        print(f"\n  At earliest time (step {es0}):")
        cl = classical_block_scores(sim_e, N, n_blocks)
        print(f"    Classical scores: {np.array2string(cl, precision=3)}")
        qa = qaoa_block_scores(sim_e, N, n_blocks, threshold, w_z_frac,
                               Phi_prev=Phi_prev_e2)
        print(f"    QAOA probs:       {np.array2string(qa, precision=3)}")
        gt_e = ground_truth_errors(sim_e, N, n_blocks)
        print(f"    Early GT errors:  {np.array2string(gt_e, precision=4)}")
        print(f"    Late GT errors:   {np.array2string(gt_late, precision=4)}")


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  QAOA vs CLASSICAL: Noise Robustness + Early Detection")
    print("=" * 70)

    test_noise_robustness()
    test_early_detection()

    print("\n\n" + "#" * 70)
    print("  CONCLUSION")
    print("#" * 70)
    print("  Check the tables above:")
    print("  - If QAOA wins at high noise → quantum denoising advantage exists")
    print("  - If QAOA wins at early steps → quantum predictive advantage exists")
    print("  - If Classical wins everywhere → no quantum advantage for AMR")
    print("#" * 70)


if __name__ == "__main__":
    main()
