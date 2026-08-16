#!/usr/bin/env python3
"""
Phase 5 - QAOA evaluation on promising patches.

Only runs on patches where Phase 4 showed the Hamiltonian is correct
(exact ground state agrees with L2 ground truth). Compares:
  1. QAOA decisions (optimized circuit)
  2. Classical threshold decisions
  3. Exact ground state decisions (from Phase 4)
  4. L2 ground truth

This is the final validation: does the QAOA circuit actually find
decisions close to the exact ground state, and does it beat classical?

Input:  results/dns_{scenario}_Re{Re}_N{N}.npz
        results/patches_{scenario}_Re{Re}_N{N}_dim{D}.npz
        results/exact_diag_{scenario}_Re{Re}_N{N}_dim{D}.npz
Output: results/qaoa_eval_{scenario}_Re{Re}_N{N}_dim{D}.npz

Usage:
  python study/qaoa_inputs.py
  python study/qaoa_inputs.py --re 800 --dim 2 --reps 2
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
from config import (
    RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N, VQA_DIMS,
    TRAINED_SIGMA, TRAINED_BETA_CURL, TRAINED_BETA_XPOINT,
    TRAINED_W_Z_FRAC, TRAINED_THRESHOLD, TRAINED_GAMMA_HYDRO,
    TRAINED_GAMMA_MAG, TRAINED_KAPPA, TRAINED_BETA,
    V2_THRESHOLD,
)
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.HamiltParams import PhysicalMapper
from Simulation.HamiltParams_v2 import PhysicalMapperV2
from Simulation.PhysToAngle import AngleMapper
from VQA.cost_hamiltonian import create_period_hamiltonian
from VQA.mapping import mapping
from VQA.execute import execute
from VQA.postprocess import postprocess


# -------------------------------------------------------------------
# Coefficient pruning (path C)
# -------------------------------------------------------------------

def prune_hamilt_params(hamilt_params, eps):
    """Zero out coefficients with |coeff| < eps * max(|coeff|) in each block.

    Returns a new dict with H_edges, C_edges, K_plaquettes pruned. This
    yields sparser Hamiltonians → shorter compiled circuits → faster
    simulation. We prune each block independently so a very strong H_i
    doesn't kill all C/K terms (they live on different scales).
    """
    import copy
    hp = copy.deepcopy(hamilt_params)

    def _prune_block(arr):
        if arr is None:
            return arr
        m = float(np.max(np.abs(arr)))
        if m == 0.0:
            return arr
        arr[np.abs(arr) < eps * m] = 0.0
        return arr

    # H_edges: tuple/list of 2 arrays
    if "H_edges" in hp and hp["H_edges"] is not None:
        H0, H1 = hp["H_edges"]
        hp["H_edges"] = (_prune_block(np.asarray(H0, dtype=float).copy()),
                         _prune_block(np.asarray(H1, dtype=float).copy()))

    # C_edges: tuple/list of 2 arrays
    if "C_edges" in hp and hp["C_edges"] is not None:
        C0, C1 = hp["C_edges"]
        hp["C_edges"] = (_prune_block(np.asarray(C0, dtype=float).copy()),
                         _prune_block(np.asarray(C1, dtype=float).copy()))

    # K_plaquettes: single array
    if "K_plaquettes" in hp and hp["K_plaquettes"] is not None:
        hp["K_plaquettes"] = _prune_block(
            np.asarray(hp["K_plaquettes"], dtype=float).copy())

    return hp


# -------------------------------------------------------------------
# Warm-start from classical score (path D)
# -------------------------------------------------------------------

def classical_warm_start_params(score_vqa, threshold_amr, reps):
    """Schedule (beta, gamma) CONSTANT — ne lit NI `score_vqa` NI `threshold_amr`.

    D-48. Le nom, l'ancienne docstring (« from the classical AMR decision »)
    et l'aide CLI de `--warm-start` (« classical-score-derived ») annoncaient
    un warm start derive de la decision classique. Il n'en est rien : le
    corps ne consomme aucun des deux premiers arguments et rend
    `beta = 0.05` partout, `gamma = 0.15 / k`, pour tout champ et tout seuil.

    Mesure du contrat (6 entrees couvrant tout l'intervalle : score nul,
    score unite, score aleatoire, seuil 0 / 1 / 1e9) : sortie identique
    BIT-A-BIT, ecart maximal **0,0e+00**. Les deux arguments sont morts.

    Cela compte parce que les appelants les passent a cote de warm starts
    qui, eux, sont reels — `sa_warm` et `greedy` demarrent sur
    `classical_init_spins(score_vqa, thr_amr, dim)` dans
    `h0_optimiser_equivalence.solver_panel` — et parce que la fiche d'audit
    compte « warm start present » parmi les axes que les etudes h0/h3
    traversent. Elles ne le traversent pas : elles traversent une
    initialisation constante.

    NON CORRIGE — decision, pas defaut de code. Rendre le schedule
    reellement dependant du score deplacerait `progress` (T11b,
    `RESULTS.md`), un nombre publie : voir la mesure dans `DEFAUTS.md` D-48.
    Le schedule reste donc bit-a-bit celui sur lequel les nombres publies
    ont ete obtenus. `tests/study/test_warm_start_is_constant.py` epingle
    cette independance : le jour ou quelqu'un la lie au score, le test
    tombe et la mesure doit etre refaite.

    Les deux arguments sont conserves dans la signature parce que les quatre
    sites d'appel les passent ; les retirer serait un changement d'API sans
    rapport avec la question posee.
    """
    del score_vqa, threshold_amr        # D-48 : jamais lus, explicitement.
    # Small ramp: large-ish gamma on first layer so the cost Hamiltonian
    # nudges amplitudes in the classical direction; tiny beta to avoid
    # flattening the distribution.
    beta = np.full(reps, 0.05)
    # gamma ramp: geometric decay, first layer dominant
    k = np.arange(1, reps + 1)
    gamma = 0.15 / k
    return np.concatenate([beta, gamma])


# -------------------------------------------------------------------
# Build QAOA inputs for a snapshot
# -------------------------------------------------------------------


def _psi_from_pipeline(vx, vy, Bx, By, prev_fields, N, n_patches, Re, dx,
                       HamiltMapper, threshold_amr, beta, fixed_curl=False):
    """Angles (theta, psi) tels que les calcule le pipeline DEPLOYE.

    On ne recalcule rien : on appelle `refinement._prepare_vqa_input`, qui
    est l'encodeur reellement utilise par V1. Reimplementer donnerait un psi
    qui RESSEMBLE au vrai sans l'etre — et un psi vraisemblable mais faux
    serait indiscernable du bon, ce qui est precisement le defaut que cette
    etude traque.

    Le scan de profondeur 0 est periodique et couvre tout le domaine, donc
    bounds = (0, N, 0, N) et depth = 0.
    """
    from types import SimpleNamespace

    from Simulation.PhysToAngle import AngleMapper
    from Simulation.refinement import _prepare_vqa_input

    grid = PeriodicGrid(N)
    sim = MHDSolver(grid, dt=1e-4, Re=Re, Rm=Re)
    sim.vx, sim.vy, sim.Bx, sim.By = vx, vy, Bx, By
    physics_state = sim.get_fluxes()

    mapper = AngleMapper()
    Phi = mapper.compute_stress_flux(physics_state)
    Phi_prev = mapper.compute_stress_flux(prev_fields)

    full_h, full_v = Phi["phi_horizontal"], Phi["phi_vertical"]
    prev_h, prev_v = Phi_prev["phi_horizontal"], Phi_prev["phi_vertical"]

    # Meme definition que refinement.py:602-603.
    average_phi_dev = 0.5 * (np.mean(np.abs(full_h - prev_h))
                             + np.mean(np.abs(full_v - prev_v)))

    full_score = AngleMapper.classical_score(physics_state,
                                             fixed_curl=fixed_curl)

    prep = _prepare_vqa_input(
        full_h, full_v, prev_h, prev_v, full_score,
        physics_state, (0, N, 0, N), 0, mapper,
        SimpleNamespace(AdvAnomaliesEnable=False),
        average_phi_dev, beta, n_patches,
        HamiltMapper=HamiltMapper, sim=sim, threshold_amr=threshold_amr,
    )
    if prep is None:
        raise RuntimeError(
            "l'encodeur du pipeline a refuse ce patch ; psi ne peut pas etre "
            "reconstruit et il ne sera pas fabrique")
    angles, _mini_hp, _mini_score = prep
    return angles


def prepare_qaoa_inputs(vx, vy, Bx, By, N, n_patches, Re,
                        use_v2=False, prev_fields=None,
                        with_psi=False, beta=1.0, fixed_curl=False):
    """
    Prepare all inputs needed for QAOA on one snapshot.

    Returns:
      data_in: dict with theta_h, theta_v, psi_h, psi_v
      hamilt_params: dict for Hamiltonian construction
      score_vqa: (n_patches, n_patches) classical score

    prev_fields / with_psi
    ----------------------
    psi encode la DERIVEE TEMPORELLE du flux de contrainte. Il vaut zero par
    defaut ici (voir plus bas), alors que le pipeline deploye le calcule
    (refinement.py:181) et que la campagne Optuna a regle les hyperparametres
    avec lui actif. Passer prev_fields (l'instantane precedent) et
    with_psi=True rebranche cet encodage, en DELEGUANT a l'encodeur du
    pipeline plutot qu'en le reimplementant.

    fixed_curl
    ----------
    Les mappeurs forment leur rotationnel et leur divergence sous la
    convention indexing='xy' alors que `grid.py` declare indexing='ij'. Sous
    la convention du depot, leur « vorticite » vaut dv_y/dy - dv_x/dx : elle
    est aveugle a la rotation solide (voir `tests/test_analytic_fields.py`).
    fixed_curl=True applique la convention declaree. Le defaut reste False,
    bit-a-bit identique au chemin sur lequel les hyperparametres ont ete
    optimises.
    """
    dx = 2 * np.pi / N
    nu = 1.0 / Re
    eta = 1.0 / Re
    patch_size = N // n_patches

    if use_v2:
        mapper = PhysicalMapperV2(dx=dx, fixed_curl=fixed_curl)
    else:
        mapper = PhysicalMapper(
            cs=1.0, nu=nu, eta_mhd=eta, dx=dx,
            gamma_hydro=TRAINED_GAMMA_HYDRO, gamma_mag=TRAINED_GAMMA_MAG,
            kappa=TRAINED_KAPPA, sigma=TRAINED_SIGMA,
            beta_curl=TRAINED_BETA_CURL, beta_xpoint=TRAINED_BETA_XPOINT,
            w_z_frac=TRAINED_W_Z_FRAC, fixed_curl=fixed_curl,
        )

    # compute Jz at full resolution for classical score
    grad_By_x_full = (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)) / (2.0 * dx)
    grad_Bx_y_full = (np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1)) / (2.0 * dx)
    Jz_full = grad_By_x_full - grad_Bx_y_full

    # full-resolution classical score (needs Jz)
    physics_state = {"vx": vx, "vy": vy, "Bx": Bx, "By": By,
                     "Jz": Jz_full, "dx": dx}
    full_score = AngleMapper.classical_score(physics_state,
                                             fixed_curl=fixed_curl)

    # downsample to VQA resolution
    def block_avg(f):
        return f.reshape(n_patches, patch_size, n_patches, patch_size).mean(axis=(1, 3))

    def block_max(f):
        return f.reshape(n_patches, patch_size, n_patches, patch_size).max(axis=(1, 3))

    vx_vqa = block_avg(vx)
    vy_vqa = block_avg(vy)
    Bx_vqa = block_avg(Bx)
    By_vqa = block_avg(By)
    score_vqa = block_max(full_score)

    # compute coefficients at VQA resolution
    dx_vqa = dx * patch_size
    grid_vqa = PeriodicGrid(n_patches, length_L=2*np.pi)
    sim_vqa = MHDSolver(grid_vqa, dt=1e-4, Re=Re, Rm=Re)
    sim_vqa.vx = vx_vqa
    sim_vqa.vy = vy_vqa
    sim_vqa.Bx = Bx_vqa
    sim_vqa.By = By_vqa

    fields_vqa = sim_vqa.get_fluxes()
    threshold_amr = V2_THRESHOLD if use_v2 else TRAINED_THRESHOLD
    hamilt_params = mapper.compute_coefficients(
        sim_vqa, score_vqa, fields_vqa, threshold_amr,
        dx_override=dx_vqa, verbose=False,
    )

    # compute angles for qubit initialization
    # theta = 2*arcsin(sqrt(score)), psi = 0 (no temporal flux in study)
    score_h = score_vqa  # horizontal edges get the score
    score_v = score_vqa  # vertical edges get the score
    theta_h = 2.0 * np.arcsin(np.sqrt(np.clip(score_h, 0, 1)))
    theta_v = 2.0 * np.arcsin(np.sqrt(np.clip(score_v, 0, 1)))
    psi_h = np.zeros_like(theta_h)
    psi_v = np.zeros_like(theta_v)

    if with_psi:
        if prev_fields is None:
            raise ValueError(
                "with_psi=True exige prev_fields : psi est une derivee "
                "temporelle, il ne peut pas etre calcule sur un instantane "
                "isole. Un psi fabrique sans instantane precedent serait "
                "indiscernable du vrai.")
        theta_h, theta_v, psi_h, psi_v = _psi_from_pipeline(
            vx, vy, Bx, By, prev_fields, N, n_patches, Re, dx,
            mapper, threshold_amr, beta, fixed_curl=fixed_curl,
        )

    data_in = {
        "theta_h": theta_h,
        "theta_v": theta_v,
        "psi_h": psi_h,
        "psi_v": psi_v,
    }

    return data_in, hamilt_params, score_vqa


# -------------------------------------------------------------------
# Run QAOA and extract decisions
# -------------------------------------------------------------------

def run_qaoa_on_snapshot(data_in, hamilt_params, dim, reps=2,
                         K_opt=100, shots=8192,
                         backend_name="state_vector",
                         warm_start_params=None,
                         prune_eps=0.0):
    """
    Build and run QAOA circuit, return decisions.

    Returns:
      marginals: list of P(qi=1) for each qubit
      decisions_h: (dim, dim) bool
      decisions_v: (dim, dim) bool
      optimal_params: optimized QAOA parameters
      wall_time: seconds
    """
    t0 = time.time()

    # optional coefficient pruning: drop coefficients below prune_eps * max
    # (only for QAOA execution — the un-pruned Hamiltonian is still used
    #  for E_patch analysis in phase 6).
    if prune_eps > 0.0:
        hamilt_params = prune_hamilt_params(hamilt_params, prune_eps)

    # build circuit
    qc, cost_hamiltonian = mapping(
        data_in, hamilt_params,
        advanced_anomalies_enabled=True,
        period_bound=True,
        reps=reps,
    )

    # estimate E_max for initial parameter setup
    coeffs = np.abs(cost_hamiltonian.coeffs)
    E_max = float(np.sum(coeffs)) if len(coeffs) > 0 else 1.0
    E_max = max(E_max, 1.0)

    # run QAOA (state_vector or matrix_product_state for scaling)
    distribution, optimal_params = execute(
        qc, cost_hamiltonian,
        mode="simulator",
        backend_name=backend_name,
        shots=shots,
        reps=reps,
        K_opt=K_opt,
        eps=1e-3,
        E_max=E_max,
        verbose=False,
        warm_start_params=warm_start_params,
    )

    n_qubits = 2 * dim * dim
    marginals = postprocess(distribution, n_qubits, verbose=False)

    # convert marginals to decisions
    marg_array = np.array(marginals)
    n_cells = dim * dim
    marg_h = marg_array[:n_cells].reshape(dim, dim)
    marg_v = marg_array[n_cells:].reshape(dim, dim)

    decisions_h = marg_h > 0.5
    decisions_v = marg_v > 0.5

    wall_time = time.time() - t0

    return marginals, decisions_h, decisions_v, optimal_params, wall_time


# -------------------------------------------------------------------
# Comparison
# -------------------------------------------------------------------

def full_comparison(qaoa_h, qaoa_v, exact_h, exact_v,
                    gt_refine, score_patch, threshold_amr):
    """
    Compare QAOA vs exact diag vs classical vs ground truth.
    """
    qaoa_refine = qaoa_h | qaoa_v
    exact_refine = exact_h | exact_v
    classical_refine = score_patch > threshold_amr

    def metrics(pred, gt):
        tp = np.sum(pred & gt)
        fp = np.sum(pred & ~gt)
        fn = np.sum(~pred & gt)
        tn = np.sum(~pred & ~gt)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # agreement between QAOA and exact ground state
    qaoa_exact_agree = np.mean(qaoa_refine == exact_refine)

    return {
        "qaoa": metrics(qaoa_refine, gt_refine),
        "exact": metrics(exact_refine, gt_refine),
        "classical": metrics(classical_refine, gt_refine),
        "qaoa_exact_agreement": qaoa_exact_agree,
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def run_phase5(dns_path, patches_path, ed_path, n_patches,
               reps=2, K_opt=100, use_v2=False,
               backend_name="state_vector",
               warm_start_classical=False,
               prune_eps=0.0):
    """
    Run Phase 5 for one (scenario, Re, dim) combination.
    """
    dns = np.load(dns_path)
    patches = np.load(patches_path)
    ed = np.load(ed_path)

    vx_all = dns["vx"]
    vy_all = dns["vy"]
    Bx_all = dns["Bx"]
    By_all = dns["By"]
    N = vx_all.shape[1]
    Re = int(dns.get("meta_Re", 800))
    scenario = str(dns.get("meta_scenario", "unknown"))

    l2_all = patches["l2_errors"]
    l2_threshold = float(patches["l2_threshold"])
    is_hard_all = patches["is_hard"]

    # Phase 4 results
    promising = ed["promising"]
    snap_indices = ed["snap_indices"]
    ed_decisions_h = ed["decisions_h"]
    ed_decisions_v = ed["decisions_v"]
    ed_gt_refine = ed["gt_refine"]

    n_promising = np.sum(promising)
    print(f"  {scenario} Re={Re} dim={n_patches}: "
          f"{n_promising}/{len(promising)} promising snapshots")

    if n_promising == 0:
        print("  No promising patches -- skipping QAOA.")
        return None, None

    all_results = []
    warm_start = None

    for idx in range(len(snap_indices)):
        if not promising[idx]:
            continue

        si = snap_indices[idx]
        vx = vx_all[si].astype(np.float64)
        vy = vy_all[si].astype(np.float64)
        Bx = Bx_all[si].astype(np.float64)
        By = By_all[si].astype(np.float64)

        # prepare inputs
        data_in, hamilt_params, score_vqa = prepare_qaoa_inputs(
            vx, vy, Bx, By, N, n_patches, Re, use_v2=use_v2,
        )

        # pick warm-start:
        #   - previous optimal params if available (warm across snapshots)
        #   - else the FIXED schedule if enabled (D-48 : « classical » de nom
        #     seulement, il ne lit pas score_vqa)
        #   - else None (use linear ramp default)
        this_warm = warm_start
        if this_warm is None and warm_start_classical:
            thr = V2_THRESHOLD if use_v2 else TRAINED_THRESHOLD
            this_warm = classical_warm_start_params(score_vqa, thr, reps)

        # run QAOA
        marginals, qaoa_h, qaoa_v, optimal_params, wall_time = \
            run_qaoa_on_snapshot(
                data_in, hamilt_params, n_patches,
                reps=reps, K_opt=K_opt,
                backend_name=backend_name,
                warm_start_params=this_warm,
                prune_eps=prune_eps,
            )

        # warm start for next snapshot
        warm_start = optimal_params

        # ground truth
        gt_refine = l2_all[si] >= l2_threshold

        # comparison
        comp = full_comparison(
            qaoa_h, qaoa_v,
            ed_decisions_h[idx], ed_decisions_v[idx],
            gt_refine, score_vqa,
            V2_THRESHOLD if use_v2 else TRAINED_THRESHOLD,
        )

        print(f"    snap {si:3d}: "
              f"QAOA_F1={comp['qaoa']['f1']:.3f} "
              f"exact_F1={comp['exact']['f1']:.3f} "
              f"class_F1={comp['classical']['f1']:.3f} "
              f"QAOA-exact agree={comp['qaoa_exact_agreement']:.2f} "
              f"({wall_time:.1f}s)")

        all_results.append({
            "snap_idx": si,
            "marginals": np.array(marginals),
            "qaoa_h": qaoa_h,
            "qaoa_v": qaoa_v,
            "optimal_params": optimal_params,
            "wall_time": wall_time,
            "comparison": comp,
        })

    if not all_results:
        return None, None

    # summary
    qaoa_f1s = [r["comparison"]["qaoa"]["f1"] for r in all_results]
    exact_f1s = [r["comparison"]["exact"]["f1"] for r in all_results]
    class_f1s = [r["comparison"]["classical"]["f1"] for r in all_results]
    agrees = [r["comparison"]["qaoa_exact_agreement"] for r in all_results]

    print(f"\n  Summary ({len(all_results)} promising snapshots):")
    print(f"    QAOA F1:       mean={np.mean(qaoa_f1s):.3f} "
          f"std={np.std(qaoa_f1s):.3f}")
    print(f"    Exact diag F1: mean={np.mean(exact_f1s):.3f} "
          f"std={np.std(exact_f1s):.3f}")
    print(f"    Classical F1:  mean={np.mean(class_f1s):.3f} "
          f"std={np.std(class_f1s):.3f}")
    print(f"    QAOA-exact agreement: mean={np.mean(agrees):.3f}")

    # verdict
    qaoa_wins = sum(q > c for q, c in zip(qaoa_f1s, class_f1s))
    print(f"\n    QAOA beats classical: {qaoa_wins}/{len(all_results)} snapshots")

    meta = {
        "scenario": scenario, "Re": Re, "N": N,
        "n_patches": n_patches, "reps": reps, "K_opt": K_opt,
        "suffix": "_v2" if use_v2 else "",
    }

    return all_results, meta


def save_results(all_results, meta, outdir=RESULTS_DIR):
    """Save Phase 5 results."""
    if all_results is None:
        return None

    suffix = meta.get("suffix", "")
    fname = (f"qaoa_eval_{meta['scenario']}_Re{meta['Re']}"
             f"_N{meta['N']}_dim{meta['n_patches']}{suffix}.npz")
    path = os.path.join(outdir, fname)

    n = len(all_results)
    dim = meta["n_patches"]

    snap_indices = np.array([r["snap_idx"] for r in all_results])
    marginals = np.array([r["marginals"] for r in all_results])
    qaoa_h = np.array([r["qaoa_h"] for r in all_results])
    qaoa_v = np.array([r["qaoa_v"] for r in all_results])
    wall_times = np.array([r["wall_time"] for r in all_results])
    qaoa_f1 = np.array([r["comparison"]["qaoa"]["f1"] for r in all_results])
    exact_f1 = np.array([r["comparison"]["exact"]["f1"] for r in all_results])
    class_f1 = np.array([r["comparison"]["classical"]["f1"] for r in all_results])
    agreement = np.array([r["comparison"]["qaoa_exact_agreement"]
                          for r in all_results])

    np.savez_compressed(
        path,
        snap_indices=snap_indices,
        marginals=marginals,
        qaoa_decisions_h=qaoa_h,
        qaoa_decisions_v=qaoa_v,
        wall_times=wall_times,
        qaoa_f1=qaoa_f1,
        exact_f1=exact_f1,
        classical_f1=class_f1,
        qaoa_exact_agreement=agreement,
        scenario=meta["scenario"],
        Re=meta["Re"],
        N=meta["N"],
        n_patches=meta["n_patches"],
        reps=meta["reps"],
    )
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {fname} ({size_kb:.0f} KB)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: QAOA evaluation on promising patches")
    parser.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    parser.add_argument("--scenario", nargs="+", default=SCENARIOS)
    parser.add_argument("--dim", nargs="+", type=int, default=VQA_DIMS)
    parser.add_argument("--N", type=int, default=DNS_N)
    parser.add_argument("--reps", type=int, default=2,
                        help="QAOA circuit depth (number of layers)")
    parser.add_argument("--K_opt", type=int, default=100,
                        help="Max COBYLA iterations")
    parser.add_argument("--v2", action="store_true",
                        help="Use parameter-free v2 Hamiltonian")
    parser.add_argument("--backend", default="state_vector",
                        choices=["state_vector", "matrix_product_state",
                                 "aer"],
                        help="Aer simulation method (use matrix_product_state "
                             "for dim>=3 scaling)")
    parser.add_argument("--warm-start", action="store_true",
                        help="Initialise QAOA from a FIXED (beta, gamma) "
                             "schedule instead of execute()'s E_max-scaled "
                             "ramp. Ne derive pas du score classique (D-48)")
    parser.add_argument("--prune-eps", type=float, default=0.0,
                        help="Prune |coeff| < eps * max per block before "
                             "QAOA (0 = no pruning)")
    args = parser.parse_args()

    version = "v2" if args.v2 else "v1"
    print(f"Phase 5: QAOA evaluation on promising patches ({version})")
    print(f"  Patch dims: {args.dim}, reps={args.reps}, K_opt={args.K_opt}")
    print(f"  Backend:    {args.backend}")
    if args.warm_start:
        print(f"  Warm-start: classical-derived schedule")
    if args.prune_eps > 0:
        print(f"  Pruning:    |coeff| < {args.prune_eps} * max (per block)")
    print()

    final_summary = {}

    for sc in args.scenario:
        for re in args.re:
            for dim in args.dim:
                n_qubits = 2 * dim * dim
                # raise ceiling for MPS backend
                qubit_cap = 40 if args.backend == "matrix_product_state" else 20
                if n_qubits > qubit_cap:
                    continue

                suffix = "_v2" if args.v2 else ""
                dns_path = os.path.join(
                    RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
                patches_path = os.path.join(
                    RESULTS_DIR,
                    f"patches_{sc}_Re{re}_N{args.N}_dim{dim}.npz")
                ed_path = os.path.join(
                    RESULTS_DIR,
                    f"exact_diag_{sc}_Re{re}_N{args.N}_dim{dim}{suffix}.npz")

                for p in [dns_path, patches_path, ed_path]:
                    if not os.path.exists(p):
                        print(f"  SKIP: {os.path.basename(p)} not found")
                        break
                else:
                    print(f"[{sc} Re={re} dim={dim}]")
                    results, meta = run_phase5(
                        dns_path, patches_path, ed_path, dim,
                        reps=args.reps, K_opt=args.K_opt,
                        use_v2=args.v2,
                        backend_name=args.backend,
                        warm_start_classical=args.warm_start,
                        prune_eps=args.prune_eps,
                    )
                    if results is not None:
                        save_results(results, meta)
                        final_summary[(sc, re, dim)] = {
                            "qaoa_f1": np.mean([r["comparison"]["qaoa"]["f1"]
                                                for r in results]),
                            "exact_f1": np.mean([r["comparison"]["exact"]["f1"]
                                                 for r in results]),
                            "class_f1": np.mean([r["comparison"]["classical"]["f1"]
                                                 for r in results]),
                            "n_promising": len(results),
                        }
                    print()

    # final verdict
    if final_summary:
        print("=" * 70)
        print("PHASE 5 FINAL VERDICT")
        print("=" * 70)
        print(f"  {'Scenario':<16} {'Re':>4} {'dim':>4}  "
              f"{'QAOA_F1':>8} {'Exact_F1':>8} {'Class_F1':>8} "
              f"{'N_prom':>6}")
        for (sc, re, dim), s in sorted(final_summary.items()):
            print(f"  {sc:<16} {re:>4} {dim:>4}  "
                  f"{s['qaoa_f1']:>8.3f} {s['exact_f1']:>8.3f} "
                  f"{s['class_f1']:>8.3f} {s['n_promising']:>6}")

        # overall: does QAOA beat classical on average?
        all_qaoa = [s["qaoa_f1"] for s in final_summary.values()]
        all_class = [s["class_f1"] for s in final_summary.values()]
        print(f"\n  Overall QAOA F1: {np.mean(all_qaoa):.3f}")
        print(f"  Overall Classical F1: {np.mean(all_class):.3f}")
        if np.mean(all_qaoa) > np.mean(all_class):
            print("  >> QAOA provides quantum advantage on hard patches.")
        else:
            print("  >> Classical threshold remains competitive.")

    print("\nPhase 5 complete.")


if __name__ == "__main__":
    main()
