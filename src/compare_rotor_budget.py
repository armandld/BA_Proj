#!/usr/bin/env python3
"""
Budget-Constrained AMR Comparison: Q-HAS vs Classical on MHD Rotor
==================================================================

Proof-of-concept demonstrating quantum advantage in a specific regime:
when the number of blocks that can be refined is **limited** (budget),
the classical linear indicator misses the optimal selection because it
cannot distinguish "high vorticity, no Jz" (rotor core, smooth)
from "high vorticity AND high Jz" (magnetic sheath, needs refinement).

The QAOA circuit, through entangled qubits encoding vort and Jz,
captures this XOR-like correlation and selects the correct blocks.

Protocol
--------
1. Run MHD Rotor at high resolution (DNS reference) to time T_eval
2. Run the same simulation at coarse resolution (baseline)
3. Partition the domain into N_blocks × N_blocks macro-blocks
4. Compute ground truth: per-block L2 error (DNS vs coarse)
5. Rank blocks by "refinement need" (ground truth error)
6. With budget K (can only refine K out of N_blocks² blocks):
   - Classical: top-K by linear indicator score
   - Q-HAS: QAOA selects K blocks via cost Hamiltonian
7. For each selection, run step_layered with only those K blocks active
8. Compare resulting L2 errors against DNS

Usage
-----
    cd src/
    python compare_rotor_budget.py [--resolution 96] [--n-blocks 3] [--budget 3]

Le circuit porte 2*n_blocks^2 qubits : n_blocks=4 en demanderait 32,
soit 69 Go de statevector. Voir `verifier_taille_circuit`.
"""

import argparse
import sys
import os
import numpy as np
from types import SimpleNamespace

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from VQA.runtime import VQARuntime
from call_vqa_shell import call_vqa_shell
from hyperparams_loader import load_hyperparams


def compute_block_errors(dns_fluxes, coarse_fluxes, N, n_blocks):
    """
    Compute per-block L2 error between DNS and coarse solution.

    Returns (n_blocks, n_blocks) array of relative L2 errors.
    """
    block_h = N // n_blocks
    block_w = N // n_blocks
    errors = np.zeros((n_blocks, n_blocks))
    fields = ['vx', 'vy', 'Bx', 'By', 'Jz']

    for bi in range(n_blocks):
        for bj in range(n_blocks):
            y0, y1 = bi * block_h, (bi + 1) * block_h
            x0, x1 = bj * block_w, (bj + 1) * block_w

            total_err = 0.0
            for var in fields:
                dns_block = dns_fluxes[var][y0:y1, x0:x1]
                coarse_block = coarse_fluxes[var][y0:y1, x0:x1]
                diff = np.sqrt(np.mean((dns_block - coarse_block) ** 2))
                ref = np.sqrt(np.mean(dns_block ** 2)) + 1e-10
                total_err += diff / ref
            errors[bi, bj] = total_err / len(fields)

    return errors


def classical_block_scores(physics_state, N, n_blocks):
    """
    Compute per-block classical indicator score.
    Uses the same multi-indicator RMS as the pipeline.

    Returns (n_blocks, n_blocks) array of scores.
    """
    full_score = AngleMapper.classical_score(physics_state)
    block_h = N // n_blocks
    block_w = N // n_blocks
    scores = np.zeros((n_blocks, n_blocks))

    for bi in range(n_blocks):
        for bj in range(n_blocks):
            y0, y1 = bi * block_h, (bi + 1) * block_h
            x0, x1 = bj * block_w, (bj + 1) * block_w
            # Max-pool: use max score in block (same as AMR refinement logic)
            scores[bi, bj] = np.max(full_score[y0:y1, x0:x1])

    return scores


def qhas_block_scores(sim, n_blocks, argus, Phi_prev=None):
    """
    Run Q-HAS VQA on the full domain partitioned into n_blocks × n_blocks.
    Returns per-block refinement probabilities from the QAOA circuit.
    """
    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    N = sim.grid.N
    nu = sim.grid.L / argus.Re
    eta_mhd = sim.grid.L / argus.Rm

    # D-10 : `PhysicalMapper(..., beta=0.5, ...)` levait `TypeError`.
    # `beta` a quitte le constructeur du mapper pour devenir un argument de
    # `run_adaptive_vqa` ; ce script n'en a jamais eu besoin ici, il passe
    # deja son propre beta a `map_to_angles` plus bas. Le script mourait a
    # l'etape 4 sur 5 et n'a donc JAMAIS produit son .npz.
    #
    # Les trois constantes 0.5 / 0.5 / 5.0 sont remplacees par les
    # hyperparametres REELLEMENT deployes : une demonstration d'avantage
    # quantique sur des parametres que personne n'utilise ne demontre rien
    # sur le critere de ce depot. Les quatre cles que la signature actuelle
    # attend (sigma, beta_curl, beta_xpoint, w_z_frac) etaient absentes de
    # l'appel, donc silencieusement remplacees par les defauts du mapper.
    hp = load_hyperparams()
    HamiltMapper = PhysicalMapper(
        cs=argus.c_s, nu=nu, eta_mhd=eta_mhd,
        dx=sim.grid.dx,
        gamma_hydro=hp["gamma_hydro"],
        gamma_mag=hp["gamma_mag"],
        kappa=hp["kappa"],
        sigma=hp.get("sigma", 0.05),
        beta_curl=hp["beta_curl"],
        beta_xpoint=hp["beta_xpoint"],
        w_z_frac=hp["w_z_frac"],
    )

    vqa_runtime = VQARuntime(
        backend_name=argus.backend,
        mode=argus.mode,
        shots=argus.shots,
        opt_level=argus.opt_level,
    )

    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)

    # Use classical_score (same as the pipeline) for fair comparison
    full_score = AngleMapper.classical_score(physics_state)
    hamilt_params = HamiltMapper.compute_coefficients(
        sim, full_score, physics_state,
        threshold_amr=0.0,
        advanced_anomalies_enabled=argus.AdvAnomaliesEnable,
    )

    # Downsample to n_blocks × n_blocks and run VQA
    from Simulation.RescaleArrays import get_adaptive_flux, _process_score

    target_dim = n_blocks

    from Simulation.utils import slice_hamiltonian_params

    local_h = Phi['phi_horizontal']
    local_v = Phi['phi_vertical']
    local_score = full_score

    prev_h = Phi_prev['phi_horizontal'] if Phi_prev is not None else None
    prev_v = Phi_prev['phi_vertical']   if Phi_prev is not None else None

    if prev_h is not None:
        AveragePhiDev = 0.5 * (np.mean(np.abs(local_h - prev_h))
                                + np.mean(np.abs(local_v - prev_v)))
        mini_h, mini_v, mini_prev_h, mini_prev_v, mini_hamilt_params, mini_score = \
            get_adaptive_flux(
                local_h, local_v, prev_h, prev_v, local_score,
                hamilt_params, target_dim=target_dim, type_filter=True,
            )
        mini_Phi_prev = {'phi_horizontal': mini_prev_h,
                         'phi_vertical':   mini_prev_v}
    else:
        AveragePhiDev = None
        mini_h, mini_v, mini_hamilt_params, mini_score = get_adaptive_flux(
            local_h, local_v, None, None, local_score,
            hamilt_params, target_dim=target_dim, type_filter=True,
        )
        mini_Phi_prev = None

    mini_score = np.clip(mini_score, 0.0, 1.0)

    mini_angles = mapper.map_to_angles(
        score_h=mini_score, score_v=mini_score,
        phi_dict_prev=mini_Phi_prev,
        phi_dict={'phi_horizontal': mini_h, 'phi_vertical': mini_v},
        AveragePhiDev=AveragePhiDev, beta=1.0,
    )

    # Adjust reps for the grid size
    reps = (target_dim - 1) * 2
    argus_vqa = SimpleNamespace(**vars(argus))
    argus_vqa.reps = reps

    probs, _ = call_vqa_shell(
        mini_angles, mini_hamilt_params, False, argus_vqa,
        period_bound=True,
        vqa_runtime=vqa_runtime,
    )

    if probs is None:
        print("[WARNING] VQA returned None, falling back to uniform scores")
        return np.ones((n_blocks, n_blocks)) * 0.5

    num_edges = target_dim * target_dim
    probs_h = probs[:num_edges].reshape(target_dim, target_dim)
    probs_v = probs[num_edges:].reshape(target_dim, target_dim)
    prob_map = 0.5 * (probs_h + probs_v)

    return prob_map


#: Au-dela, le statevector ne tient plus en memoire sur une machine
#: ordinaire. 18 qubits coutent 4 Mo, 32 en coutent 69 Go.
QUBITS_MAX_STATEVECTOR = 20


def qubits_requis(n_blocks):
    """Nombre de qubits du circuit de selection : un par arete du pavage.

    `call_vqa_shell` rend `2 * n_blocks**2` probabilites (horizontales et
    verticales), donc le circuit porte autant de qubits.
    """
    return 2 * n_blocks ** 2


def verifier_taille_circuit(n_blocks, backend, qubits_max=QUBITS_MAX_STATEVECTOR):
    """Refuse AVANT le DNS une taille de circuit qui ne tiendra pas.

    Le defaut d'origine (`--n-blocks 4`) demande 32 qubits, soit 69 Go de
    statevector. Il echouait avec un `QiskitError` de qiskit-aer -- mais
    seulement a l'etape 4 sur 5, apres avoir paye la simulation DNS et la
    reference grossiere. Une configuration impossible doit etre refusee au
    premier instant, avec le chiffre qui explique pourquoi.
    """
    q = qubits_requis(n_blocks)
    if backend == "state_vector" and q > qubits_max:
        octets = (2 ** q) * 16
        raise ValueError(
            f"n_blocks={n_blocks} demande {q} qubits "
            f"(2*{n_blocks}^2), soit {octets / 1e9:.1f} Go de statevector. "
            f"Maximum retenu : {qubits_max} qubits. "
            f"Utiliser n_blocks <= {int((qubits_max / 2) ** 0.5)} avec "
            f"`state_vector`, ou --backend aer.")


def select_top_k(scores, budget):
    """Select indices of top-K blocks by score. Returns list of (i, j)."""
    flat_indices = np.argsort(scores.ravel())[::-1][:budget]
    n_cols = scores.shape[1]
    return [(idx // n_cols, idx % n_cols) for idx in flat_indices]


def build_patches_from_selection(selection, N, n_blocks, max_depth):
    """
    Build active patch list from selected block indices.
    Selected blocks get full-depth (leaf_depth), others get depth=0 (coarse).
    """
    block_h = N // n_blocks
    block_w = N // n_blocks
    patches = []

    selected_set = set(selection)
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            y0, y1 = bi * block_h, (bi + 1) * block_h
            x0, x1 = bj * block_w, (bj + 1) * block_w
            if (bi, bj) in selected_set:
                patches.append({
                    'bounds': (y0, y1, x0, x1),
                    'depth': max_depth,
                    'type': 'leaf_depth',
                })
            else:
                patches.append({
                    'bounds': (y0, y1, x0, x1),
                    'depth': 0,
                    'type': 'coarse_leaf',
                })
    return patches


def compute_solution_error(sim_fluxes, dns_fluxes):
    """Global weighted L2 error across all fields."""
    fields = ['vx', 'vy', 'Bx', 'By', 'Jz']
    total_err = 0.0
    for var in fields:
        diff = np.sqrt(np.mean((sim_fluxes[var] - dns_fluxes[var]) ** 2))
        ref = np.sqrt(np.mean(dns_fluxes[var] ** 2)) + 1e-10
        total_err += diff / ref
    return total_err / len(fields)


def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Budget-constrained AMR comparison: Q-HAS vs Classical on MHD Rotor"
    )
    # 96 = 3 x 32 : divisible par le defaut `--n-blocks 3`. Le couple
    # d'origine (128, 4) violait DEUX contraintes a la fois -- 128 n'est pas
    # divisible par 3 une fois n_blocks corrige, et 4 blocs demandent
    # 32 qubits. Les defauts d'un script doivent former une configuration
    # qui tourne.
    parser.add_argument("--resolution", type=int, default=96,
                        help="DNS grid resolution (NxN), divisible par --n-blocks")
    # Defaut ramene de 4 a 3 : le circuit porte 2*n_blocks^2 qubits, et le
    # simulateur `state_vector` alloue 2^q nombres complexes.
    #
    #   n_blocks=2 ->  8 qubits ->   4 ko
    #   n_blocks=3 -> 18 qubits ->   4 Mo
    #   n_blocks=4 -> 32 qubits ->  69 Go   <- le defaut d'origine
    parser.add_argument("--n-blocks", type=int, default=3,
                        help="Partition into n_blocks × n_blocks macro-blocks "
                             "(cout: 2*n^2 qubits ; 4 est hors de portee memoire)")
    parser.add_argument("--budget", type=int, default=3,
                        help="Number of blocks allowed to refine (budget K)")
    parser.add_argument("--t-eval", type=float, default=0.5,
                        help="Time at which to evaluate (rotor must be developed)")
    parser.add_argument("--dt", type=float, default=1e-3,
                        help="Time step")
    parser.add_argument("--Re", type=int, default=800)
    parser.add_argument("--Rm", type=int, default=800)
    parser.add_argument("--backend", default="state_vector",
                        choices=["aer", "state_vector"])
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--K-opt", type=int, default=50,
                        help="Max QAOA optimizer iterations")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-dir", default="../data/rotor_budget",
                        help="Directory to save results")
    args = parser.parse_args()

    N = args.resolution
    n_blocks = args.n_blocks
    budget = args.budget
    T_eval = args.t_eval

    assert N % n_blocks == 0, (
        f"Resolution {N} must be divisible by n_blocks {n_blocks} — "
        f"essayer --resolution {n_blocks * round(N / n_blocks)}")
    assert budget <= n_blocks ** 2, f"Budget {budget} exceeds total blocks {n_blocks**2}"
    verifier_taille_circuit(n_blocks, args.backend)

    print("=" * 70)
    print("  BUDGET-CONSTRAINED AMR: Q-HAS vs CLASSICAL on MHD ROTOR")
    print("=" * 70)
    print(f"  Resolution:  {N}×{N}")
    print(f"  Blocks:      {n_blocks}×{n_blocks} = {n_blocks**2} total")
    print(f"  Budget:      {budget} blocks (out of {n_blocks**2})")
    print(f"  T_eval:      {T_eval}")
    print(f"  Re={args.Re}, Rm={args.Rm}")
    print("=" * 70)

    # ── Step 1: Run DNS reference ──
    print("\n[1/5] Running DNS reference simulation...")
    grid = PeriodicGrid(resolution_N=N)
    sim_dns = MHDSolver(grid, dt=args.dt, Re=args.Re, Rm=args.Rm)
    sim_dns.init_mhd_rotor()

    t = 0.0
    step = 0
    while t < T_eval:
        dt = sim_dns.adapt_dt(cfl_target=0.4)
        dt = min(dt, T_eval - t)
        sim_dns.dt = dt
        sim_dns.step_full(record_stats=False)
        t += dt
        step += 1
        if step % 100 == 0:
            print(f"  DNS step {step}, t={t:.4f}")

    dns_fluxes = sim_dns.get_fluxes()
    print(f"  DNS complete: {step} steps, t={t:.4f}")

    # ── Step 2: Run coarse baseline (no refinement) ──
    print("\n[2/5] Running coarse baseline (no AMR)...")
    sim_coarse = MHDSolver(grid, dt=args.dt, Re=args.Re, Rm=args.Rm)
    sim_coarse.init_mhd_rotor()

    # Use very coarse stepping (downsample factor for step_layered)
    max_depth = 3
    target_dim = 2
    # step_layered with all patches at depth=0 → maximally coarse everywhere
    all_coarse_patches = []
    block_h = N // n_blocks
    block_w = N // n_blocks
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            y0, y1 = bi * block_h, (bi + 1) * block_h
            x0, x1 = bj * block_w, (bj + 1) * block_w
            all_coarse_patches.append({
                'bounds': (y0, y1, x0, x1),
                'depth': 0,
                'type': 'coarse_leaf',
            })

    t = 0.0
    step = 0
    _mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    Phi_prev_coarse = None
    while t < T_eval:
        Phi_prev_coarse = _mapper.compute_stress_flux(sim_coarse.get_fluxes())
        dt = sim_coarse.adapt_dt(cfl_target=0.4)
        dt = min(dt, T_eval - t)
        sim_coarse.dt = dt
        sim_coarse.step_layered(all_coarse_patches, max_depth=max_depth,
                                target_dim=target_dim)
        t += dt
        step += 1

    coarse_fluxes = sim_coarse.get_fluxes()
    coarse_error = compute_solution_error(coarse_fluxes, dns_fluxes)
    print(f"  Coarse baseline error: {coarse_error:.6f}")

    # ── Step 3: Compute ground truth block errors and block scores ──
    print("\n[3/5] Computing per-block errors and indicator scores...")
    block_errors = compute_block_errors(dns_fluxes, coarse_fluxes, N, n_blocks)

    # Ground truth: which blocks truly need refinement (top-K by error)
    gt_selection = select_top_k(block_errors, budget)
    print(f"  Ground truth top-{budget} blocks (by error): {gt_selection}")
    print(f"  Block errors:\n{np.array2string(block_errors, precision=4)}")

    # Classical selection
    classical_scores = classical_block_scores(
        sim_coarse.get_fluxes(), N, n_blocks
    )
    classical_selection = select_top_k(classical_scores, budget)
    print(f"\n  Classical scores:\n{np.array2string(classical_scores, precision=4)}")
    print(f"  Classical top-{budget} blocks: {classical_selection}")

    # ── Step 4: Q-HAS selection ──
    print("\n[4/5] Running Q-HAS VQA for block selection...")
    argus = SimpleNamespace(
        reps=(n_blocks - 1) * 2,
        mode="simulator",
        backend=args.backend,
        shots=args.shots,
        method="COBYLA",
        opt_level=1,
        AdvAnomaliesEnable=True,
        K_opt=args.K_opt,
        eps=1e-2,
        eta=0.001,
        Bz_guide=0.1,
        c_s=1.0,
        Re=args.Re,
        Rm=args.Rm,
    )

    qhas_scores = qhas_block_scores(sim_coarse, n_blocks, argus,
                                     Phi_prev=Phi_prev_coarse)
    qhas_selection = select_top_k(qhas_scores, budget)
    print(f"  Q-HAS scores:\n{np.array2string(qhas_scores, precision=4)}")
    print(f"  Q-HAS top-{budget} blocks: {qhas_selection}")

    # ── Step 5: Run step_layered with each selection and compare ──
    print("\n[5/5] Running budget-constrained AMR with each selection...")

    results = {}
    for label, selection in [("Ground Truth", gt_selection),
                             ("Classical", classical_selection),
                             ("Q-HAS", qhas_selection)]:
        sim_test = MHDSolver(grid, dt=args.dt, Re=args.Re, Rm=args.Rm)
        sim_test.init_mhd_rotor()

        patches = build_patches_from_selection(selection, N, n_blocks, max_depth)

        t = 0.0
        step = 0
        total_pixels = 0
        while t < T_eval:
            dt = sim_test.adapt_dt(cfl_target=0.4)
            dt = min(dt, T_eval - t)
            sim_test.dt = dt
            pixels = sim_test.step_layered(patches, max_depth=max_depth,
                                           target_dim=target_dim)
            total_pixels += pixels
            t += dt
            step += 1

        test_fluxes = sim_test.get_fluxes()
        error = compute_solution_error(test_fluxes, dns_fluxes)
        avg_pixels = total_pixels / max(step, 1)
        results[label] = {
            'error': error,
            'selection': selection,
            'avg_pixels': avg_pixels,
            'steps': step,
        }
        print(f"  {label:15s}: L2 error = {error:.6f}, "
              f"avg pixels/step = {avg_pixels:.0f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  {'Method':<15s} {'L2 Error':>12s} {'vs Coarse':>12s} "
          f"{'Selection':>30s}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*30}")
    print(f"  {'No AMR':<15s} {coarse_error:>12.6f} {'(baseline)':>12s} "
          f"{'none':>30s}")
    for label in ["Ground Truth", "Classical", "Q-HAS"]:
        r = results[label]
        improvement = (1.0 - r['error'] / coarse_error) * 100
        sel_str = str(r['selection'])
        print(f"  {label:<15s} {r['error']:>12.6f} {improvement:>+11.1f}% "
              f"{sel_str:>30s}")

    # Highlight agreement with ground truth
    gt_set = set(gt_selection)
    cl_overlap = len(set(classical_selection) & gt_set)
    qh_overlap = len(set(qhas_selection) & gt_set)
    print(f"\n  Agreement with ground truth:")
    print(f"    Classical: {cl_overlap}/{budget} blocks correct")
    print(f"    Q-HAS:     {qh_overlap}/{budget} blocks correct")

    if results["Q-HAS"]["error"] < results["Classical"]["error"]:
        advantage = (1.0 - results["Q-HAS"]["error"] / results["Classical"]["error"]) * 100
        print(f"\n  >>> Q-HAS ADVANTAGE: {advantage:.1f}% lower error than Classical <<<")
    else:
        disadvantage = (results["Q-HAS"]["error"] / results["Classical"]["error"] - 1.0) * 100
        print(f"\n  >>> Classical wins by {disadvantage:.1f}% "
              f"(Q-HAS needs more optimization or different hyperparams) <<<")

    print("=" * 70)

    # ── Save results ──
    os.makedirs(args.save_dir, exist_ok=True)
    np.savez(
        os.path.join(args.save_dir, "rotor_budget_results.npz"),
        block_errors=block_errors,
        classical_scores=classical_scores,
        qhas_scores=qhas_scores,
        gt_selection=np.array(gt_selection),
        classical_selection=np.array(classical_selection),
        qhas_selection=np.array(qhas_selection),
        results_errors={k: v['error'] for k, v in results.items()},
        coarse_error=coarse_error,
        N=N, n_blocks=n_blocks, budget=budget, T_eval=T_eval,
    )
    print(f"\nResults saved to {args.save_dir}/rotor_budget_results.npz")


if __name__ == "__main__":
    main()
