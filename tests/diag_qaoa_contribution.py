"""
Diagnostic: Does the QAOA actually change refinement decisions?

Runs the VQA pipeline on each scenario at a single time step and compares:
1. Classical score map (θ-only, before QAOA)
2. QAOA probability map (after optimization)
3. Refinement decisions with threshold

Reports:
- Number of cells flipped by QAOA (from refine→skip or skip→refine)
- Energy decomposition: Z vs ZZ vs ZZZZ after downsampling
- Multi-body/single-body energy ratio
- QAOA improvement: cost before vs after optimization
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


def diagnose_scenario(scenario_name, sim, N, VQA_N=2, threshold=0.3,
                      w_z_frac=0.15, beta_mic=0.5, K_opt=50, Phi_prev=None):
    """Run one diagnostic cycle on a developed simulation state."""
    grid = sim.grid
    nu = grid.L / 800
    eta = grid.L / 800

    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    HamiltMapper = PhysicalMapper(
        cs=1.0, nu=nu, eta_mhd=eta, beta=beta_mic, dx=grid.dx,
        gamma_hydro=0.5, gamma_mag=0.5, kappa=5.0, w_z_frac=w_z_frac,
    )

    argus = SimpleNamespace(
        reps=(VQA_N - 1) * 2,
        mode="simulator", backend="state_vector",
        shots=1024, method="COBYLA", opt_level=1,
        AdvAnomaliesEnable=True,
        K_opt=K_opt, eps=1e-2,
        eta=0.001, Bz_guide=0.1, c_s=1.0, Re=800, Rm=800,
    )

    vqa_runtime = VQARuntime(
        backend_name="state_vector", mode="simulator",
        shots=1024, opt_level=1,
    )

    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)

    # Use classical_score (same as the fixed pipeline)
    full_score = AngleMapper.classical_score(physics_state)
    hamilt_params = HamiltMapper.compute_coefficients(
        sim, full_score, physics_state, threshold,
        advanced_anomalies_enabled=True,
    )

    # Downsample to VQA_N × VQA_N
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
                target_dim=VQA_N, type_filter=True,
            )
        mini_Phi_prev = {'phi_horizontal': mini_prev_h,
                         'phi_vertical':   mini_prev_v}
    else:
        AveragePhiDev = None
        mini_h, mini_v, mini_hp, mini_score = get_adaptive_flux(
            phi_h, phi_v, None, None, full_score, hamilt_params,
            target_dim=VQA_N, type_filter=True,
        )
        mini_Phi_prev = None

    mini_score = np.clip(mini_score, 0.0, 1.0)

    # Angles (classical → θ, Phase Boost → ψ)
    angles = mapper.map_to_angles(
        score_h=mini_score, score_v=mini_score,
        phi_dict_prev=mini_Phi_prev,
        phi_dict={'phi_horizontal': mini_h, 'phi_vertical': mini_v},
        AveragePhiDev=AveragePhiDev, beta=1.0,
    )

    # Before QAOA: classical score = P(|1⟩) from θ
    prob_before = mini_score

    # Energy decomposition
    mH_h, mH_v = mini_hp['H_edges']
    mC_h, mC_v = mini_hp['C_edges']
    mK = mini_hp['K_plaquettes']
    E_Z = np.sum(np.abs(mH_h)) + np.sum(np.abs(mH_v))
    E_ZZ = np.sum(np.abs(mC_h)) + np.sum(np.abs(mC_v))
    E_ZZZZ = np.sum(np.abs(mK))
    mKx = mini_hp.get('K_xpoint', np.zeros_like(mK))
    E_xpoint = np.sum(np.abs(mKx))
    E_total = E_Z + E_ZZ + E_ZZZZ + E_xpoint
    ratio = (E_ZZ + E_ZZZZ + E_xpoint) / max(E_Z, 1e-10)

    # Run QAOA
    probs, _ = call_vqa_shell(
        angles, mini_hp, False, argus,
        period_bound=True, vqa_runtime=vqa_runtime,
    )

    if probs is not None:
        n_edges = VQA_N * VQA_N
        ph = probs[:n_edges].reshape(VQA_N, VQA_N)
        pv = probs[n_edges:].reshape(VQA_N, VQA_N)
        prob_after = 0.5 * (ph + pv)
    else:
        prob_after = prob_before.copy()

    # Decision comparison
    refine_before = prob_before >= threshold
    refine_after = prob_after >= threshold
    flipped = np.sum(refine_before != refine_after)
    total_cells = VQA_N * VQA_N

    # New refinements by QAOA
    added = np.sum(~refine_before & refine_after)
    removed = np.sum(refine_before & ~refine_after)

    print(f"\n{'='*60}")
    print(f"  SCENARIO: {scenario_name} (threshold={threshold}, w_z_frac={w_z_frac})")
    print(f"{'='*60}")
    print(f"  Score map (classical, {VQA_N}×{VQA_N}):")
    print(f"    {np.array2string(mini_score, precision=3)}")
    print(f"\n  Before QAOA (θ-only):")
    print(f"    P(|1⟩) = {np.array2string(prob_before, precision=3)}")
    print(f"    Refine: {refine_before.astype(int)}")
    print(f"\n  After QAOA:")
    print(f"    P(|1⟩) = {np.array2string(prob_after, precision=3)}")
    print(f"    Refine: {refine_after.astype(int)}")
    print(f"\n  Decision changes: {flipped}/{total_cells} cells flipped")
    print(f"    Added by QAOA: {added}, Removed by QAOA: {removed}")
    print(f"\n  Energy decomposition (after downsampling to {VQA_N}×{VQA_N}):")
    print(f"    Z  (1-body): E={E_Z:.3f} ({E_Z/E_total*100:.1f}%)")
    print(f"    ZZ (2-body): E={E_ZZ:.3f} ({E_ZZ/E_total*100:.1f}%)")
    print(f"    ZZZZ (4-body): E={E_ZZZZ:.3f} ({E_ZZZZ/E_total*100:.1f}%)")
    if E_xpoint > 0:
        print(f"    X-point (4-body): E={E_xpoint:.3f} ({E_xpoint/E_total*100:.1f}%)")
    print(f"    Multi/Single ratio: {ratio:.3f}", end="")
    if ratio < 0.1:
        print(" ⚠ SINGLE-BODY PROBLEM (no quantum advantage)")
    elif ratio < 0.3:
        print(" ⚠ Weak multi-body")
    else:
        print(" ✓ Meaningful multi-body")

    print(f"\n  Hamiltonian coefficients ({VQA_N}×{VQA_N}):")
    print(f"    H_h (Z):  {np.array2string(mH_h, precision=3)}")
    print(f"    C_h (ZZ): {np.array2string(mC_h, precision=3)}")
    print(f"    K (ZZZZ): {np.array2string(mK, precision=3)}")

    # Check if all Z biases have the same sign (no discrimination)
    all_positive = np.all(mH_h > 0) and np.all(mH_v > 0)
    all_negative = np.all(mH_h < 0) and np.all(mH_v < 0)
    if all_positive:
        print(f"\n  ⚠ ALL Z biases positive → QAOA ground state = refine everything")
        print(f"    (Score range [{mini_score.min():.3f}, {mini_score.max():.3f}] "
              f"all > threshold={threshold})")
    elif all_negative:
        print(f"\n  ⚠ ALL Z biases negative → QAOA ground state = refine nothing")
    else:
        z_pos = np.sum(mH_h > 0) + np.sum(mH_v > 0)
        z_neg = np.sum(mH_h < 0) + np.sum(mH_v < 0)
        print(f"\n  ✓ Mixed Z biases: {z_pos} positive, {z_neg} negative "
              f"→ genuine decision problem")

    return {
        'flipped': flipped, 'total': total_cells,
        'ratio': ratio, 'prob_before': prob_before, 'prob_after': prob_after,
    }


def main():
    N = 64
    VQA_N = 2

    scenarios = {
        'kelvin_helmholtz': ('init_kelvin_helmholtz', 500),
        'lamb_oseen_vortex': ('init_lamb_oseen_vortex', 300),
        'harris_tearing': ('init_harris_tearing', 600),
        'island_coalescence': ('init_island_coalescence', 300),
        'orszag_tang': ('init_orszag_tang', 2000),
        'mhd_rotor': ('init_mhd_rotor', 300),
    }

    print("=" * 60)
    print("  DIAGNOSTIC: QAOA Contribution Analysis")
    print("=" * 60)
    print(f"  Grid: {N}×{N}, VQA: {VQA_N}×{VQA_N}")

    for threshold in [0.3, 0.5]:
        for w_z_frac in [0.15, 0.10]:
            print(f"\n\n{'#'*60}")
            print(f"  CONFIGURATION: threshold={threshold}, w_z_frac={w_z_frac}")
            print(f"{'#'*60}")

            for name, (init_method, n_steps) in scenarios.items():
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

                diagnose_scenario(name, sim, N, VQA_N, threshold, w_z_frac,
                                  Phi_prev=Phi_prev)


if __name__ == "__main__":
    main()
