"""
Controlled QAOA Decision Validation
====================================
Tests whether the Q-HAS Hamiltonian + QAOA produces correct refinement
decisions in synthetic cases with KNOWN ground truth.

Test cases:
1. Uniform quiet field     → all scores << threshold → refine NOTHING
2. Uniform active field    → all scores >> threshold → refine EVERYTHING
3. Single hot cell         → one cell above threshold → refine ONLY that cell
4. Checkerboard (XOR)      → anti-correlated vorticity/Jz → discriminate
5. Gradient field          → smooth gradient across domain → refine high end only
6. MHD Rotor (real)        → sheath = high, core = moderate → refine sheath

For each case, we check:
- Z bias signs match ground truth (positive = should refine, negative = should not)
- QAOA probability map respects ground truth after optimization
- Multi-body terms (ZZ, ZZZZ) have meaningful magnitude when appropriate
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


def make_args(VQA_N=2, K_opt=80):
    return SimpleNamespace(
        reps=(VQA_N - 1) * 2,
        mode="simulator", backend="state_vector",
        shots=1024, method="COBYLA", opt_level=1,
        AdvAnomaliesEnable=True,
        K_opt=K_opt, eps=1e-2,
        eta=0.001, Bz_guide=0.1, c_s=1.0, Re=800, Rm=800,
    )


def run_qaoa_on_sim(sim, VQA_N=2, threshold=0.3, w_z_frac=0.15, K_opt=80,
                    Phi_prev=None):
    """Run full pipeline: score → Hamiltonian → downsample → QAOA → probabilities."""
    grid = sim.grid
    N = grid.resolution
    nu = grid.L / 800
    eta = grid.L / 800

    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    HamiltMapper = PhysicalMapper(
        cs=1.0, nu=nu, eta_mhd=eta, beta_curl=0.5, beta_xpoint=0.5,
        dx=grid.dx,
        gamma_hydro=0.5, gamma_mag=0.5, kappa=5.0, w_z_frac=w_z_frac,
    )

    args = make_args(VQA_N, K_opt)
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

    angles = mapper.map_to_angles(
        score_h=mini_score, score_v=mini_score,
        phi_dict_prev=mini_Phi_prev,
        phi_dict={'phi_horizontal': mini_h, 'phi_vertical': mini_v},
        AveragePhiDev=AveragePhiDev, beta=1.0,
    )

    # Extract Z biases (downsampled)
    mH_h, mH_v = mini_hp['H_edges']
    mC_h, mC_v = mini_hp['C_edges']
    mK = mini_hp['K_plaquettes']

    # Run QAOA
    probs, _ = call_vqa_shell(
        angles, mini_hp, False, args,
        period_bound=True, vqa_runtime=vqa_runtime,
    )

    n_edges = VQA_N * VQA_N
    if probs is not None:
        ph = probs[:n_edges].reshape(VQA_N, VQA_N)
        pv = probs[n_edges:].reshape(VQA_N, VQA_N)
        prob_after = 0.5 * (ph + pv)
    else:
        prob_after = mini_score.copy()

    return {
        'score': mini_score,
        'prob_after': prob_after,
        'H_h': mH_h, 'H_v': mH_v,
        'C_h': mC_h, 'C_v': mC_v,
        'K': mK,
        'threshold': threshold,
        'full_score': full_score,
    }


def create_quiet_sim(N=32):
    """Uniform weak field — no features, nothing to refine."""
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    # Very weak uniform field + tiny noise
    rng = np.random.default_rng(42)
    sim.vx = 0.01 * rng.standard_normal(grid.X.shape)
    sim.vy = 0.01 * rng.standard_normal(grid.X.shape)
    sim.Bx = 1.0 * np.ones_like(grid.X)
    sim.By = np.zeros_like(grid.X)
    sim.enforce_incompressibility()
    sim.record_energy()
    return sim


def create_active_sim(N=32):
    """Strong everywhere — everything should be refined."""
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    # Strong shear flow + strong current everywhere
    X, Y = grid.X, grid.Y
    sim.vx = 2.0 * np.sin(Y)
    sim.vy = 2.0 * np.sin(X)
    sim.Bx = 2.0 * np.sin(Y)
    sim.By = 2.0 * np.cos(X)
    sim.enforce_incompressibility()
    sim.record_energy()
    return sim


def create_localized_anomaly_sim(N=32):
    """One quadrant has a strong vortex + current sheet, rest is quiet."""
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    X, Y = grid.X, grid.Y
    cx, cy = grid.L / 4, grid.L / 4  # Top-left quadrant center

    # Background: weak uniform
    sim.vx = 0.01 * np.ones_like(X)
    sim.vy = 0.01 * np.ones_like(X)
    sim.Bx = 1.0 * np.ones_like(X)
    sim.By = np.zeros_like(X)

    # Localized vortex in top-left quadrant
    r = np.sqrt((X - cx)**2 + (Y - cy)**2 + 1e-12)
    mask = (r < grid.L / 4)
    strength = 5.0 * np.exp(-r**2 / (grid.L / 8)**2)
    sim.vx += -strength * (Y - cy) / r * mask
    sim.vy +=  strength * (X - cx) / r * mask
    # Current sheet in same region
    sim.By += 3.0 * np.tanh((X - cx) / (grid.L / 16)) * mask

    sim.enforce_incompressibility()
    sim.record_energy()
    return sim


def create_gradient_sim(N=32):
    """Linear gradient: left side quiet, right side active."""
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    X, Y = grid.X, grid.Y

    # Gradient: intensity increases left to right
    intensity = X / grid.L  # 0 to 1
    sim.vx = 3.0 * intensity * np.sin(4 * Y)
    sim.vy = 3.0 * intensity * np.cos(4 * X)
    sim.Bx = 1.0 * np.ones_like(X)
    sim.By = 2.0 * intensity * np.sin(2 * X)

    sim.enforce_incompressibility()
    sim.record_energy()
    return sim


def create_rotor_sim(N=32, n_steps=100):
    """Real MHD rotor — sheath should be refined, quiet exterior not."""
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
    return sim, Phi_prev


def print_result(name, res, ground_truth_desc):
    """Pretty-print a test result."""
    VQA_N = res['score'].shape[0]
    threshold = res['threshold']

    refine_classical = res['score'] >= threshold
    refine_qaoa = res['prob_after'] >= threshold

    E_Z = np.sum(np.abs(res['H_h'])) + np.sum(np.abs(res['H_v']))
    E_ZZ = np.sum(np.abs(res['C_h'])) + np.sum(np.abs(res['C_v']))
    E_ZZZZ = np.sum(np.abs(res['K']))
    E_total = E_Z + E_ZZ + E_ZZZZ + 1e-10

    print(f"\n{'='*65}")
    print(f"  TEST: {name}")
    print(f"  Expected: {ground_truth_desc}")
    print(f"{'='*65}")
    print(f"  Score map ({VQA_N}x{VQA_N}, threshold={threshold}):")
    print(f"    {np.array2string(res['score'], precision=3)}")
    print(f"  Z bias H_h: {np.array2string(res['H_h'], precision=3)}")
    print(f"  Z bias H_v: {np.array2string(res['H_v'], precision=3)}")

    z_pos = np.sum(res['H_h'] > 0) + np.sum(res['H_v'] > 0)
    z_neg = np.sum(res['H_h'] < 0) + np.sum(res['H_v'] < 0)
    print(f"  Z sign split: {z_pos} positive (refine), {z_neg} negative (skip)")

    print(f"\n  Classical decision: {refine_classical.astype(int)}")
    print(f"  QAOA decision:     {refine_qaoa.astype(int)}")
    flipped = np.sum(refine_classical != refine_qaoa)
    print(f"  Cells flipped by QAOA: {flipped}/{VQA_N*VQA_N}")

    print(f"\n  QAOA P(|1>): {np.array2string(res['prob_after'], precision=3)}")
    print(f"  Energy: Z={E_Z:.3f} ({E_Z/E_total*100:.1f}%), "
          f"ZZ={E_ZZ:.3f} ({E_ZZ/E_total*100:.1f}%), "
          f"ZZZZ={E_ZZZZ:.3f} ({E_ZZZZ/E_total*100:.1f}%)")
    print(f"  Multi/Single ratio: {(E_ZZ+E_ZZZZ)/max(E_Z,1e-10):.3f}")

    return {
        'z_pos': z_pos, 'z_neg': z_neg,
        'flipped': flipped, 'E_Z': E_Z, 'E_ZZ': E_ZZ, 'E_ZZZZ': E_ZZZZ,
        'refine_classical': refine_classical, 'refine_qaoa': refine_qaoa,
    }


# ══════════════════════════════════════════════════════════════════════
#  ACCEPTANCE — what V1 is expected to do
# ══════════════════════════════════════════════════════════════════════
#
# Five of the seven checks hold; two do NOT, and both are real V1 defects
# rather than noise:
#
#   quiet_no_refine   — on a quiet uniform field `physical_score` returns
#                       ~0.49-0.58, i.e. above threshold=0.3, so the
#                       pipeline refines all four cells. The score is not
#                       calibrated to the threshold it is compared against.
#   active_z_positive — on a strong active field only 4 of the 8 Z biases
#                       are positive, and they sit at ~1e-4.
#
# The expected pattern is therefore pinned as a pattern, not as "all pass".
# The stage fails if V1 moves in EITHER direction — a repair is a change of
# behaviour and must be noticed too.
EXPECTED_CHECKS = {
    'quiet_z_negative': True,
    'quiet_no_refine': False,     # known defect: quiet field scores ~0.55
    'active_z_positive': False,   # known defect: only 4/8 biases positive
    'active_all_refine': True,
    'localized_mixed': True,
    'gradient_mixed': True,
    'rotor_multibody': True,
}


def check_expected_behaviour(checks):
    """Compare the outcome pattern against the recorded one and exit non-zero
    on any difference.

    Without this the script prints "Some checks FAILED" and returns 0, so
    `run_tests.sh` reports the stage as PASSED.
    """
    missing = set(EXPECTED_CHECKS) - set(checks)
    extra = set(checks) - set(EXPECTED_CHECKS)
    assert not missing and not extra, (
        f"the set of checks changed (missing={sorted(missing)}, "
        f"extra={sorted(extra)}); update EXPECTED_CHECKS deliberately"
    )

    diffs = {k: (bool(checks[k]), EXPECTED_CHECKS[k])
             for k in EXPECTED_CHECKS if bool(checks[k]) != EXPECTED_CHECKS[k]}
    assert not diffs, (
        "V1 decision behaviour departed from the recorded pattern "
        f"(check: got, expected): {diffs}"
    )

    n_defects = sum(1 for v in EXPECTED_CHECKS.values() if not v)
    print(f"\n  [ACCEPTANCE] outcome pattern matches the recorded one "
          f"({len(EXPECTED_CHECKS) - n_defects} hold, {n_defects} known "
          f"defects) -> OK")


def main():
    N = 32
    VQA_N = 2
    threshold = 0.3
    w_z_frac = 0.15

    print("=" * 65)
    print("  QAOA DECISION VALIDATION — Controlled Synthetic Cases")
    print(f"  Grid: {N}x{N}, VQA: {VQA_N}x{VQA_N}, threshold={threshold}, w_z_frac={w_z_frac}")
    print("=" * 65)

    results = {}
    checks = {}

    # ── Test 1: Quiet field → no refinement ──
    print("\n[1/5] Running quiet field test...")
    sim = create_quiet_sim(N)
    res = run_qaoa_on_sim(sim, VQA_N, threshold, w_z_frac)
    stats = print_result(
        "Quiet Uniform Field", res,
        "ALL Z biases negative, QAOA refines NOTHING"
    )
    # Check: all Z biases should be negative (score << threshold)
    checks['quiet_z_negative'] = stats['z_neg'] == 2 * VQA_N * VQA_N
    checks['quiet_no_refine'] = np.sum(res['prob_after'] >= threshold) == 0
    results['quiet'] = stats

    # ── Test 2: Active field → refine everything ──
    print("\n[2/5] Running active field test...")
    sim = create_active_sim(N)
    res = run_qaoa_on_sim(sim, VQA_N, threshold, w_z_frac)
    stats = print_result(
        "Strong Active Field", res,
        "ALL Z biases positive, QAOA refines EVERYTHING"
    )
    checks['active_z_positive'] = stats['z_pos'] == 2 * VQA_N * VQA_N
    checks['active_all_refine'] = np.sum(res['prob_after'] >= threshold) == VQA_N * VQA_N
    results['active'] = stats

    # ── Test 3: Localized anomaly → refine only anomaly region ──
    print("\n[3/5] Running localized anomaly test...")
    sim = create_localized_anomaly_sim(N)
    res = run_qaoa_on_sim(sim, VQA_N, threshold, w_z_frac)
    stats = print_result(
        "Localized Anomaly (top-left)", res,
        "Mixed Z biases: anomaly region positive, quiet regions negative"
    )
    checks['localized_mixed'] = stats['z_pos'] > 0 and stats['z_neg'] > 0
    results['localized'] = stats

    # ── Test 4: Gradient → refine high end only ──
    print("\n[4/5] Running gradient field test...")
    sim = create_gradient_sim(N)
    res = run_qaoa_on_sim(sim, VQA_N, threshold, w_z_frac)
    stats = print_result(
        "Left-to-Right Gradient", res,
        "Right side (high) refined, left side (low) not refined"
    )
    checks['gradient_mixed'] = stats['z_pos'] > 0 and stats['z_neg'] > 0
    results['gradient'] = stats

    # ── Test 5: Real MHD Rotor ──
    print("\n[5/5] Running MHD Rotor test (100 steps)...")
    sim, Phi_prev_rotor = create_rotor_sim(N, n_steps=100)
    res = run_qaoa_on_sim(sim, VQA_N, threshold, w_z_frac, Phi_prev=Phi_prev_rotor)
    stats = print_result(
        "MHD Rotor (real physics)", res,
        "Sheath cells refined, quiet exterior not refined"
    )
    # Rotor should have multi-body terms (ZZ/ZZZZ) meaningful
    checks['rotor_multibody'] = (stats['E_ZZ'] + stats['E_ZZZZ']) / max(stats['E_Z'], 1e-10) > 0.1
    results['rotor'] = stats

    # ── Summary ──
    print(f"\n\n{'#'*65}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'#'*65}")
    n_pass = sum(checks.values())
    n_total = len(checks)
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")
    print(f"\n  Result: {n_pass}/{n_total} checks passed")

    if n_pass == n_total:
        print("  => Hamiltonian Z biases are CONSISTENT with ground truth.")
        print("  => QAOA has meaningful multi-body terms in non-trivial cases.")
    else:
        print("  => Some checks FAILED. Investigate the cases above.")

    check_expected_behaviour(checks)

    # ── Threshold sensitivity analysis ──
    print(f"\n\n{'#'*65}")
    print(f"  THRESHOLD SENSITIVITY (MHD Rotor)")
    print(f"{'#'*65}")
    sim, Phi_prev_rotor2 = create_rotor_sim(N, n_steps=100)
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        res = run_qaoa_on_sim(sim, VQA_N, th, w_z_frac, Phi_prev=Phi_prev_rotor2)
        n_refine_classical = np.sum(res['score'] >= th)
        n_refine_qaoa = np.sum(res['prob_after'] >= th)
        flipped = np.sum((res['score'] >= th) != (res['prob_after'] >= th))
        print(f"  threshold={th:.1f}: classical refines {n_refine_classical}/{VQA_N**2}, "
              f"QAOA refines {n_refine_qaoa}/{VQA_N**2}, "
              f"flipped={flipped}")


if __name__ == "__main__":
    main()
