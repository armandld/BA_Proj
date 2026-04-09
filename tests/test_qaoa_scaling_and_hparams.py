"""
Scaling + Hyperparameter sweep: Is the 3×3 result valid at larger grids?
========================================================================

Addresses two concerns:
1. "3×3 is too small" → test with N=32, 64, 128 (same 3×3 blocks, larger cells)
2. "Maybe different hyperparameters help" → sweep w_z_frac and threshold

The idea: with more cells per block, the max-abs pooling and Hamiltonian
coefficients have richer data to work with. If QAOA's advantage scales
with physical grid resolution, it matters.

Also tests: can lowering w_z_frac (making multi-body terms more dominant)
shift the noise crossover to lower σ where it's more practically relevant?

Note: PhysicalMapper uses an adaptive Z weight controlled by w_z_frac
(default 0.15). The actual Z bias applied is w_z_frac × max(|C|, |K|) ×
(score - threshold), scaling automatically with Hamiltonian coefficient
magnitudes. The legacy w_z parameter still exists for backward
compatibility but w_z_frac is the preferred knob.
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


def ground_truth_errors(sim, N, n_blocks):
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
            errors[bi, bj] = np.mean(total[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw])
    return errors


def classical_block_scores(sim, N, n_blocks):
    state = sim.get_fluxes()
    full_score = AngleMapper.classical_score(state)
    bh = N // n_blocks
    bw = N // n_blocks
    scores = np.zeros((n_blocks, n_blocks))
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            scores[bi, bj] = np.max(full_score[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw])
    return scores


def noisy_classical_scores(sim, N, n_blocks, noise_std, rng):
    state = sim.get_fluxes()
    full_score = AngleMapper.classical_score(state)
    noisy = full_score + noise_std * rng.standard_normal(full_score.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    bh = N // n_blocks
    bw = N // n_blocks
    scores = np.zeros((n_blocks, n_blocks))
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            scores[bi, bj] = np.max(noisy[bi*bh:(bi+1)*bh, bj*bw:(bj+1)*bw])
    return scores


def qaoa_block_scores(sim, N, n_blocks, threshold, w_z_frac, K_opt=80,
                      noise_std=0.0, rng=None, Phi_prev=None):
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
        AdvAnomaliesEnable=True, K_opt=K_opt, eps=1e-2,
    )
    vqa_runtime = VQARuntime(
        backend_name="state_vector", mode="simulator",
        shots=1024, opt_level=1,
    )

    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)
    full_score = AngleMapper.classical_score(physics_state)

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

    ne = n_blocks * n_blocks
    if probs is not None:
        ph = probs[:ne].reshape(n_blocks, n_blocks)
        pv = probs[ne:].reshape(n_blocks, n_blocks)
        return 0.5 * (ph + pv)
    return mini_score.copy()


def select_top_k(scores, k):
    flat = np.argsort(scores.ravel())[::-1][:k]
    nc = scores.shape[1]
    return set((idx // nc, idx % nc) for idx in flat)


def captured_fraction(selection, gt):
    total = np.sum(gt)
    if total < 1e-12:
        return 1.0
    return sum(gt[i, j] for i, j in selection) / total


# ═════════════════════════════════════════════════════════════════════
#  TEST A: GRID RESOLUTION SCALING (clean + noisy)
# ═════════════════════════════════════════════════════════════════════

def test_resolution_scaling():
    print("\n" + "=" * 75)
    print("  TEST A: DOES GRID RESOLUTION CHANGE THE CONCLUSION?")
    print("  Same 3×3 blocks, but N=32, 64, 128 (cells per block: 10², 21², 42²)")
    print("=" * 75)

    n_blocks = 3
    budget = 2
    w_z_frac = 0.15
    threshold = 0.3
    noise_levels = [0.0, 0.2, 0.3]
    n_trials = 3

    for N, n_steps in [(32, 100), (64, 200), (128, 400)]:
        cells_per_block = (N // n_blocks) ** 2
        print(f"\n  --- N={N} ({cells_per_block} cells/block, {n_steps} steps) ---")

        grid = PeriodicGrid(resolution_N=N)
        sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
        sim.init_mhd_rotor()
        _mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
        Phi_prev = None
        for i in range(n_steps):
            if i == n_steps - 1:
                Phi_prev = _mapper.compute_stress_flux(sim.get_fluxes())
            sim.adapt_dt(cfl_target=0.4)
            sim.step_full(record_stats=False)

        gt = ground_truth_errors(sim, N, n_blocks)
        gt_sel = select_top_k(gt, budget)

        print(f"  GT errors: {np.array2string(gt, precision=4)}")
        print(f"  GT top-{budget}: {gt_sel}")

        # Clean comparison
        cl = classical_block_scores(sim, N, n_blocks)
        rho_cl, _ = spearmanr(cl.ravel(), gt.ravel())
        cl_frac = captured_fraction(select_top_k(cl, budget), gt)

        qa = qaoa_block_scores(sim, N, n_blocks, threshold, w_z_frac,
                               Phi_prev=Phi_prev)
        rho_qa, _ = spearmanr(qa.ravel(), gt.ravel())
        qa_frac = captured_fraction(select_top_k(qa, budget), gt)

        print(f"\n  Clean (σ=0):")
        print(f"    Classical: ρ={rho_cl:+.3f}, captured={cl_frac:.4f}")
        print(f"    QAOA:      ρ={rho_qa:+.3f}, captured={qa_frac:.4f}")
        winner = "QAOA" if qa_frac > cl_frac + 0.005 else (
            "Classical" if cl_frac > qa_frac + 0.005 else "Tie")
        print(f"    Winner: {winner}")

        # Noisy comparison
        for sigma in noise_levels[1:]:
            cl_fracs = []
            qa_fracs = []
            for seed in range(n_trials):
                rng = np.random.default_rng(42 + seed)
                cl_n = noisy_classical_scores(sim, N, n_blocks, sigma, rng)
                cl_fracs.append(captured_fraction(select_top_k(cl_n, budget), gt))

                rng2 = np.random.default_rng(42 + seed)
                qa_n = qaoa_block_scores(sim, N, n_blocks, threshold, w_z_frac,
                                         noise_std=sigma, rng=rng2,
                                         Phi_prev=Phi_prev)
                qa_fracs.append(captured_fraction(select_top_k(qa_n, budget), gt))

            mcl = np.mean(cl_fracs)
            mqa = np.mean(qa_fracs)
            winner = "QAOA" if mqa > mcl + 0.005 else (
                "Classical" if mcl > mqa + 0.005 else "Tie")
            marker = " <--" if winner == "QAOA" else ""
            print(f"  Noisy (σ={sigma}): Classical={mcl:.4f}, QAOA={mqa:.4f} → {winner}{marker}")


# ═════════════════════════════════════════════════════════════════════
#  TEST B: HYPERPARAMETER SWEEP (w_z_frac, threshold)
# ═════════════════════════════════════════════════════════════════════

def test_hyperparameter_sweep():
    print("\n\n" + "=" * 75)
    print("  TEST B: CAN HYPERPARAMETERS SHIFT THE CROSSOVER?")
    print("  Sweep w_z_frac ∈ {0.05, 0.10, 0.15, 0.30} × threshold ∈ {0.2, 0.3, 0.5}")
    print("  on MHD Rotor 3×3, budget=2")
    print("=" * 75)

    N = 64
    n_blocks = 3
    budget = 2

    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    sim.init_mhd_rotor()
    _mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    Phi_prev = None
    for i in range(200):
        if i == 199:
            Phi_prev = _mapper.compute_stress_flux(sim.get_fluxes())
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)

    gt = ground_truth_errors(sim, N, n_blocks)
    cl_clean = classical_block_scores(sim, N, n_blocks)
    rho_cl_clean, _ = spearmanr(cl_clean.ravel(), gt.ravel())
    cl_frac_clean = captured_fraction(select_top_k(cl_clean, budget), gt)

    print(f"\n  Classical baseline (no noise): ρ={rho_cl_clean:+.3f}, "
          f"captured={cl_frac_clean:.4f}")
    print(f"  GT errors: {np.array2string(gt, precision=4)}")

    # Clean sweep
    print(f"\n  --- CLEAN (σ=0) ---")
    print(f"  {'w_z_frac':>10s} {'thresh':>8s} {'QAOA ρ':>10s} {'captured':>10s} {'vs Classical':>14s}")
    print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*14}")
    for w_z_frac in [0.05, 0.10, 0.15, 0.30]:
        for threshold in [0.2, 0.3, 0.5]:
            qa = qaoa_block_scores(sim, N, n_blocks, threshold, w_z_frac,
                                   Phi_prev=Phi_prev)
            rho_qa, _ = spearmanr(qa.ravel(), gt.ravel())
            qa_frac = captured_fraction(select_top_k(qa, budget), gt)
            delta = qa_frac - cl_frac_clean
            marker = " *** BETTER" if delta > 0.005 else ""
            print(f"  {w_z_frac:>10.2f} {threshold:>8.1f} {rho_qa:>+10.3f} "
                  f"{qa_frac:>10.4f} {delta:>+14.4f}{marker}")

    # Noisy sweep (σ=0.3 — where we saw the crossover)
    sigma = 0.3
    n_trials = 3
    print(f"\n  --- NOISY (σ={sigma}) ---")
    print(f"  {'w_z_frac':>10s} {'thresh':>8s} {'QAOA capt':>12s} {'Cl. capt':>12s} {'Winner':>10s}")
    print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
    for w_z_frac in [0.05, 0.10, 0.15, 0.30]:
        for threshold in [0.2, 0.3, 0.5]:
            cl_fracs = []
            qa_fracs = []
            for seed in range(n_trials):
                rng = np.random.default_rng(42 + seed)
                cl_n = noisy_classical_scores(sim, N, n_blocks, sigma, rng)
                cl_fracs.append(captured_fraction(select_top_k(cl_n, budget), gt))

                rng2 = np.random.default_rng(42 + seed)
                qa_n = qaoa_block_scores(sim, N, n_blocks, threshold, w_z_frac,
                                         noise_std=sigma, rng=rng2,
                                         Phi_prev=Phi_prev)
                qa_fracs.append(captured_fraction(select_top_k(qa_n, budget), gt))

            mcl = np.mean(cl_fracs)
            mqa = np.mean(qa_fracs)
            winner = "QAOA" if mqa > mcl + 0.005 else (
                "Classical" if mcl > mqa + 0.005 else "Tie")
            marker = " <--" if winner == "QAOA" else ""
            print(f"  {w_z_frac:>10.2f} {threshold:>8.1f} {mqa:>12.4f} "
                  f"{mcl:>12.4f} {winner:>10s}{marker}")


# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 75)
    print("  QAOA SCALING + HYPERPARAMETER ANALYSIS")
    print("=" * 75)

    test_resolution_scaling()
    test_hyperparameter_sweep()

    print("\n\n" + "#" * 75)
    print("  KEY QUESTIONS ANSWERED:")
    print("  1. Does larger grid change the conclusion?")
    print("  2. Can hyperparameters shift the noise crossover?")
    print("#" * 75)


if __name__ == "__main__":
    main()
