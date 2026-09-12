#!/usr/bin/env python3
"""Compare the V1 and parameter-free V2 Hamiltonians on every scenario.

This is a functional diagnostic, not a performance benchmark.
"""
import argparse, os, sys, time
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

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.HamiltParams import PhysicalMapper
from Simulation.HamiltParams_v2 import PhysicalMapperV2
from Simulation.PhysToAngle import AngleMapper
from VQA.cost_hamiltonian import create_period_hamiltonian
from VQA.mapping import mapping
from VQA.execute import execute
from VQA.postprocess import postprocess
from config import (
    SCENARIOS, TRAINED_THRESHOLD, V2_THRESHOLD, trained_mapper_params,
)


# -------------------------------------------------------------------
# Scenario setup
# -------------------------------------------------------------------

ALL_SCENARIOS = list(SCENARIOS)
RE = 400
N = 32   # small grid for fast sanity check
DIM = 2  # 2x2 patches -> 8 qubits (fast exact diag + QAOA)
WARMUP_STEPS = 50
REPS = 2
K_OPT = 60

V1_PARAMS = trained_mapper_params()
V1_THRESHOLD = TRAINED_THRESHOLD


def marginals_converged(marg, tol=0.01):
    """
    Le QAOA a-t-il converge ? Le critere est : les marginales ne sont pas
    TOUTES a 0.5 — donc la distance a 0.5, pas la dispersion.

    Rend (converged, max|m-0.5|, min|m-0.5|, std(m)).

    Fonction de module, et non une fermeture dans `run_scenario`, pour
    qu'un test puisse l'interroger sans rejouer un scenario complet.
    """
    marg = np.asarray(marg, dtype=float)
    dist = np.abs(marg - 0.5)
    return (bool(dist.max() > tol), float(dist.max()),
            float(dist.min()), float(np.std(marg)))


def run_scenario(scenario, Re=RE, N_grid=N, dim=DIM):
    """
    Run one scenario and compare v1 vs v2 Hamiltonian.
    """
    print(f"\n{'='*60}")
    print(f"  {scenario}  Re={Re}  N={N_grid}  dim={dim}")
    print(f"{'='*60}")

    # ---- 1. Setup simulation ----
    grid = PeriodicGrid(N_grid)
    sim = MHDSolver(grid, dt=1e-3, Re=Re, Rm=Re)

    inits = {
        "orszag_tang": sim.init_orszag_tang,
        "harris_tearing": sim.init_harris_tearing,
        "kelvin_helmholtz": sim.init_kelvin_helmholtz,
        "mhd_rotor": sim.init_mhd_rotor,
        "lamb_oseen": sim.init_lamb_oseen_vortex,
        "island_coalescence": sim.init_island_coalescence,
        "double_tearing": sim.init_double_tearing,
        "magnetic_twist": sim.init_magnetic_twist,
    }
    if scenario not in inits:
        raise ValueError(f"unknown scenario: {scenario}")
    inits[scenario]()

    # warmup
    for _ in range(WARMUP_STEPS):
        sim.dt = sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)

    dx = sim.dx
    vx, vy, Bx, By = sim.vx.copy(), sim.vy.copy(), sim.Bx.copy(), sim.By.copy()
    fields = sim.get_fluxes()

    # classical score
    physics_state = {"vx": vx, "vy": vy, "Bx": Bx, "By": By, "dx": dx}
    physics_state["Jz"] = fields["Jz"]
    score = AngleMapper.classical_score(physics_state)

    # ---- 2. Compute coefficients: v1 vs v2 ----
    nu = 1.0 / Re
    eta = 1.0 / Re

    mapper_v1 = PhysicalMapper(
        cs=1.0, nu=nu, eta_mhd=eta, dx=dx, **V1_PARAMS)
    mapper_v2 = PhysicalMapperV2(dx=dx)

    coeffs_v1 = mapper_v1.compute_coefficients(
        sim, score, fields, V1_THRESHOLD,
        advanced_anomalies_enabled=True, verbose=False)
    coeffs_v2 = mapper_v2.compute_coefficients(
        sim, score, fields, V2_THRESHOLD,
        advanced_anomalies_enabled=True, verbose=False)

    # ---- 3. Report coefficient statistics ----
    def coeff_stats(coeffs, label):
        H_h, H_v = coeffs['H_edges']
        C_h, C_v = coeffs['C_edges']
        K = coeffs['K_plaquettes']

        max_H = max(np.max(np.abs(H_h)), np.max(np.abs(H_v)))
        max_C = max(np.max(np.abs(C_h)), np.max(np.abs(C_v)))
        max_K = np.max(np.abs(K))
        mean_C = 0.5 * (np.mean(np.abs(C_h)) + np.mean(np.abs(C_v)))
        mean_K = np.mean(np.abs(K))
        total_coupling = (np.sum(np.abs(C_h)) + np.sum(np.abs(C_v))
                          + np.sum(np.abs(K)))
        total_z = np.sum(np.abs(H_h)) + np.sum(np.abs(H_v))
        z_ratio = total_z / max(total_coupling, 1e-10)

        # check ferromagnetic
        nz_C = C_h[np.abs(C_h) > 1e-8]
        nz_K = K[np.abs(K) > 1e-8]
        ferro_C = np.all(nz_C <= 0) if len(nz_C) > 0 else True
        ferro_K = np.all(nz_K <= 0) if len(nz_K) > 0 else True

        print(f"\n  [{label}]")
        print(f"    max|H| = {max_H:.6f}  (Z bias)")
        print(f"    max|C| = {max_C:.6f}  mean|C| = {mean_C:.6f}  (ZZ)")
        print(f"    max|K| = {max_K:.6f}  mean|K| = {mean_K:.6f}  (ZZZZ)")
        print(f"    |Z|/|ZZ+ZZZZ| = {z_ratio:.4f}  "
              f"({'OK: Z subordinate' if z_ratio < 1 else 'WARNING: Z dominant'})")
        print(f"    Ferromagnetic C: {'YES' if ferro_C else 'NO (BUG!)'}  "
              f"Negative K: {'YES' if ferro_K else 'NO (BUG!)'}")
        print(f"    C non-zero: {np.sum(np.abs(C_h) > 1e-8) + np.sum(np.abs(C_v) > 1e-8)}"
              f"/{C_h.size + C_v.size}  "
              f"K non-zero: {np.sum(np.abs(K) > 1e-8)}/{K.size}")

        return {
            "max_H": max_H, "max_C": max_C, "max_K": max_K,
            "mean_C": mean_C, "mean_K": mean_K,
            "z_ratio": z_ratio, "ferro_C": ferro_C, "ferro_K": ferro_K,
            "total_coupling": total_coupling, "total_z": total_z,
            "survived": max_C > 1e-6 and max_K > 1e-6,
        }

    stats_v1 = coeff_stats(coeffs_v1, "v1 (trained)")
    stats_v2 = coeff_stats(coeffs_v2, "v2 (parameter-free)")

    # ---- 4. QAOA comparison ----
    # downsample to VQA resolution
    patch_size = N_grid // dim
    dx_vqa = dx * patch_size

    def block_avg(f):
        return f.reshape(dim, patch_size, dim, patch_size).mean(axis=(1, 3))

    def block_max(f):
        return f.reshape(dim, patch_size, dim, patch_size).max(axis=(1, 3))

    vx_vqa = block_avg(vx)
    vy_vqa = block_avg(vy)
    Bx_vqa = block_avg(Bx)
    By_vqa = block_avg(By)
    score_vqa = block_max(score)

    grid_vqa = PeriodicGrid(dim, length_L=2*np.pi)
    sim_vqa = MHDSolver(grid_vqa, dt=1e-4, Re=Re, Rm=Re)
    sim_vqa.vx = vx_vqa
    sim_vqa.vy = vy_vqa
    sim_vqa.Bx = Bx_vqa
    sim_vqa.By = By_vqa
    fields_vqa = sim_vqa.get_fluxes()

    def run_qaoa(mapper, threshold, label):
        coeffs = mapper.compute_coefficients(
            sim_vqa, score_vqa, fields_vqa, threshold,
            advanced_anomalies_enabled=True,
            dx_override=dx_vqa, verbose=False)

        # angles
        theta_h = 2.0 * np.arcsin(np.sqrt(np.clip(score_vqa, 0, 1)))
        theta_v = theta_h.copy()
        psi_h = np.zeros_like(theta_h)
        psi_v = np.zeros_like(theta_v)

        data_in = {
            "theta_h": theta_h, "theta_v": theta_v,
            "psi_h": psi_h, "psi_v": psi_v,
        }

        t0 = time.time()
        qc, cost_ham = mapping(data_in, coeffs, period_bound=True, reps=REPS)

        c_abs = np.abs(cost_ham.coeffs)
        E_max = max(float(np.sum(c_abs)), 1.0)

        dist, params = execute(
            qc, cost_ham, mode="simulator", backend_name="state_vector",
            shots=0, reps=REPS, K_opt=K_OPT, eps=1e-3, E_max=E_max,
            verbose=False)

        n_qubits = 2 * dim * dim
        marginals = postprocess(dist, n_qubits, verbose=False)
        elapsed = time.time() - t0

        marg = np.array(marginals)
        n_cells = dim * dim
        marg_h = marg[:n_cells].reshape(dim, dim)
        marg_v = marg[n_cells:].reshape(dim, dim)
        decisions = (marg_h > 0.5) | (marg_v > 0.5)
        classical_decisions = score_vqa > threshold

        print(f"\n  [{label} QAOA] ({elapsed:.1f}s)")
        print(f"    Marginals H: {marg_h.flatten()}")
        print(f"    Marginals V: {marg_v.flatten()}")
        print(f"    QAOA decisions:      {decisions.flatten().astype(int)}")
        print(f"    Classical decisions:  {classical_decisions.flatten().astype(int)}")
        print(f"    Score per patch:      {score_vqa.flatten()}")

        # A uniform but decisive vector is converged; distance from 0.5 is
        # therefore the relevant diagnostic, not dispersion across qubits.
        converged, decisiveness, weakest, spread = marginals_converged(marg)
        print(f"    Distance to 0.5: max={decisiveness:.4f} min={weakest:.4f}  "
              f"(std={spread:.4f})  "
              f"({'converged' if converged else 'NOT converged (all marginals at 0.5)'})")

        return {
            "marginals": marg,
            "decisions": decisions,
            "classical": classical_decisions,
            "converged": converged,
            "decisiveness": decisiveness,
            "weakest": weakest,
            "spread": spread,
        }

    qaoa_v1 = run_qaoa(mapper_v1, V1_THRESHOLD, "v1")
    qaoa_v2 = run_qaoa(mapper_v2, V2_THRESHOLD, "v2")

    # ---- 5. Comparison ----
    agree = np.mean(qaoa_v1["decisions"] == qaoa_v2["decisions"])
    print(f"\n  v1 vs v2 QAOA decision agreement: {agree:.1%}")

    return {
        "stats_v1": stats_v1, "stats_v2": stats_v2,
        "qaoa_v1": qaoa_v1, "qaoa_v2": qaoa_v2,
        "agreement": agree,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0.3: v2 Hamiltonian sanity check")
    parser.add_argument("--scenario", nargs="+", default=ALL_SCENARIOS)
    parser.add_argument("--re", type=int, default=RE)
    parser.add_argument("--N", type=int, default=N)
    parser.add_argument("--dim", type=int, default=DIM)
    args = parser.parse_args()

    print("Phase 0.3: v2 Hamiltonian sanity check")
    print(f"  Scenarios: {args.scenario}")
    print(f"  Re={args.re}, N={args.N}, dim={args.dim}")

    all_results = {}
    for sc in args.scenario:
        all_results[sc] = run_scenario(sc, Re=args.re, N_grid=args.N, dim=args.dim)

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print("PHASE 0.3 SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Scenario':<20} {'v1 survived':>12} {'v2 survived':>12} "
          f"{'v1 converged':>13} {'v2 converged':>13} {'agree':>6}")

    all_ok = True
    for sc, r in all_results.items():
        s1 = r["stats_v1"]["survived"]
        s2 = r["stats_v2"]["survived"]
        c1 = r["qaoa_v1"]["converged"]
        c2 = r["qaoa_v2"]["converged"]
        ag = r["agreement"]

        print(f"  {sc:<20} {'YES' if s1 else 'NO':>12} {'YES' if s2 else 'NO':>12} "
              f"{'YES' if c1 else 'NO':>13} {'YES' if c2 else 'NO':>13} "
              f"{ag:>5.0%}")

        if not s2:
            all_ok = False
            print(f"    ** v2 coefficients DEAD in {sc} -- investigate! **")

    print()
    if all_ok:
        print("  All v2 coefficients survived. Hamiltonian is functional.")
    else:
        print("  WARNING: Some v2 coefficients are zero. Check the formulas.")

    # ---- Qualitative differences note ----
    print(f"\n{'='*70}")
    print("QUALITATIVE DIFFERENCES v1 vs v2")
    print(f"{'='*70}")
    print("""
  v1 (HamiltParams.py, 8 trained parameters):
    - ZZ: f_gate * g_strain * threshold_contrast * Gaussian_uncertainty
    - ZZZZ: g_rot * f_gate * threshold_contrast
    - Complex physics-informed gates with trainable growth rates
    - Sigma controls uncertainty width near decision boundary
    - Coefficients can be zero far from threshold (Gaussian suppression)

  v2 (HamiltParams_v2.py, 0 trained parameters):
    - ZZ: simple domain-normalized gradient ratio
    - ZZZZ: max-normalized circulation + current density
    - No gates, no sigmoids, no Gaussians
    - Coefficients are always non-zero where gradients exist
    - Only thr_amr remains as a physical choice

  Key trade-offs:
    - v2 is simpler but may over-couple in calm regions
    - v1 has uncertainty weighting that focuses quantum corrections
      near the decision boundary (but requires sigma tuning)
    - v2 normalises by domain statistics (mean/max), which means
      coefficients are relative rather than absolute
    - v1 uses absolute physical thresholds (Re_crit, Rm_crit)
""")


if __name__ == "__main__":
    main()
