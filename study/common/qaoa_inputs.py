#!/usr/bin/env python3
"""Phase 5: QAOA benchmark on every snapshot evaluated by phase 4.

QAOA, the exact Ising solution and the classical threshold are compared to
the same L2-hard labels. ``promising`` is retained only as a diagnostic and
never filters the panel. Agreement with an arbitrary exact state is omitted
when the ground-state manifold is degenerate.

Input:  results/dns_{scenario}_Re{Re}_N{N}.npz
        results/patches_{scenario}_Re{Re}_N{N}_dim{D}.npz
        results/exact_diag_{scenario}_Re{Re}_N{N}_dim{D}.npz
Output: results/qaoa_eval_{scenario}_Re{Re}_N{N}_dim{D}.npz

Usage:
  python study/common/qaoa_inputs.py --re 800 --dim 2 --reps 2
"""
import argparse, json, os, sys, time
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
    TRAINED_THRESHOLD, TRAINED_BETA, trained_mapper_params,
    V2_THRESHOLD,
)
from Simulation.grid import AXIS_X, AXIS_Y, PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.HamiltParams import PhysicalMapper
from Simulation.HamiltParams_v2 import PhysicalMapperV2
from Simulation.PhysToAngle import AngleMapper
from VQA.cost_hamiltonian import create_period_hamiltonian
from VQA.mapping import mapping
from VQA.execute import execute
from VQA.postprocess import postprocess
import provenance


# -------------------------------------------------------------------
# Coefficient pruning (path C)
# -------------------------------------------------------------------

def prune_hamilt_params(hamilt_params, eps):
    """Zero out coefficients with |coeff| < eps * max(|coeff|) in each block.

    Returns a new dict with H_edges, C_edges, K_plaquettes and K_xpoint
    pruned. Blocks are thresholded independently because they represent
    distinct operator families.
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

    if "K_xpoint" in hp and hp["K_xpoint"] is not None:
        hp["K_xpoint"] = _prune_block(
            np.asarray(hp["K_xpoint"], dtype=float).copy())

    return hp


# -------------------------------------------------------------------
# Deterministic QAOA parameter initialisation
# -------------------------------------------------------------------

def constant_initial_params(reps):
    """Return a deterministic ``(beta, gamma)`` schedule for QAOA.

    This is an optimiser initialisation, not a classical warm start. The
    encoded classical score already enters the circuit through ``theta``.
    """
    if reps < 1:
        raise ValueError("reps doit etre >= 1")
    beta = np.full(reps, 0.05)
    k = np.arange(1, reps + 1)
    gamma = 0.15 / k
    return np.concatenate([beta, gamma])


# -------------------------------------------------------------------
# Build QAOA inputs for a snapshot
# -------------------------------------------------------------------


def _psi_from_pipeline(vx, vy, Bx, By, prev_fields, N, n_patches, Re, dx,
                       HamiltMapper, threshold_amr, beta, fixed_curl=True,
                       prev_phi=None):
    """Delegate theta/psi encoding to the deployed depth-zero encoder.

    ``prev_phi`` accepts the pipeline's exponential moving average directly.
    ``prev_fields`` is retained for snapshot-pair studies.
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
    if prev_phi is None:
        if prev_fields is None:
            raise ValueError("prev_fields or prev_phi is required")
        prev_state = dict(prev_fields)
        prev_state["Jz"] = (
            (np.roll(prev_state["By"], -1, axis=AXIS_X)
             - np.roll(prev_state["By"], 1, axis=AXIS_X))
            - (np.roll(prev_state["Bx"], -1, axis=AXIS_Y)
               - np.roll(prev_state["Bx"], 1, axis=AXIS_Y))
        ) / (2.0 * dx)
        Phi_prev = mapper.compute_stress_flux(prev_state)
    else:
        missing = {"phi_horizontal", "phi_vertical"} - set(prev_phi)
        if missing:
            raise KeyError(f"prev_phi is missing {sorted(missing)}")
        Phi_prev = prev_phi

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
        SimpleNamespace(AdvAnomaliesEnable=True),
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
                        with_psi=False, beta=1.0, fixed_curl=True,
                        prev_phi=None):
    """
    Prepare all inputs needed for QAOA on one snapshot.

    Returns:
      data_in: dict with theta_h, theta_v, psi_h, psi_v
      hamilt_params: dict for Hamiltonian construction
      score_vqa: (n_patches, n_patches) classical score

    With ``with_psi=True``, ``prev_phi`` should be the deployed EMA of the
    stress flux. A raw ``prev_fields`` snapshot is also accepted for paired
    representation studies.

    ``fixed_curl=True`` applique la convention ``indexing='ij'`` de la
    grille. Une ablation explicite peut passer ``False``.
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
            fixed_curl=fixed_curl,
            **trained_mapper_params(),
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
        advanced_anomalies_enabled=True,
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
        if prev_fields is None and prev_phi is None:
            raise ValueError(
                "with_psi=True requires prev_fields or prev_phi")
        theta_h, theta_v, psi_h, psi_v = _psi_from_pipeline(
            vx, vy, Bx, By, prev_fields, N, n_patches, Re, dx,
            mapper, threshold_amr, beta, fixed_curl=fixed_curl,
            prev_phi=prev_phi,
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
                         prune_eps=0.0,
                         seed=0):
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
        seed=seed,
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
                    gt_refine, score_patch, threshold_amr,
                    exact_ground_degeneracy=1):
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
    qaoa_exact_agree_raw = float(np.mean(qaoa_refine == exact_refine))
    qaoa_exact_agree = (
        qaoa_exact_agree_raw if exact_ground_degeneracy == 1 else np.nan)

    return {
        "qaoa": metrics(qaoa_refine, gt_refine),
        "exact": metrics(exact_refine, gt_refine),
        "classical": metrics(classical_refine, gt_refine),
        "qaoa_exact_agreement": qaoa_exact_agree,
        "qaoa_exact_agreement_raw": qaoa_exact_agree_raw,
        "exact_ground_degeneracy": int(exact_ground_degeneracy),
    }


def _stress_flux_for_snapshot(vx, vy, Bx, By, N):
    """Compute the stress flux with the deployed grid convention."""
    dx = 2 * np.pi / N
    jz = (
        (np.roll(By, -1, axis=AXIS_X) - np.roll(By, 1, axis=AXIS_X))
        - (np.roll(Bx, -1, axis=AXIS_Y) - np.roll(Bx, 1, axis=AXIS_Y))
    ) / (2.0 * dx)
    return AngleMapper().compute_stress_flux({
        "vx": vx, "vy": vy, "Bx": Bx, "By": By, "Jz": jz,
    })


def _ema_update(previous, current, alpha=0.3):
    if previous is None:
        return {key: np.asarray(value) for key, value in current.items()}
    return {
        key: alpha * np.asarray(current[key])
        + (1.0 - alpha) * np.asarray(previous[key])
        for key in current
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def run_phase5(dns_path, patches_path, ed_path, n_patches,
               reps=2, K_opt=100, use_v2=False,
               backend_name="state_vector",
               constant_initialisation=False,
               prune_eps=0.0,
               seed=0,
               zero_psi=False):
    """
    Run Phase 5 for one (scenario, Re, dim) combination.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if not 0 <= int(seed) <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
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
    # Phase 4 results
    promising = ed["promising"]
    snap_indices = ed["snap_indices"]
    ed_decisions_h = ed["decisions_h"]
    ed_decisions_v = ed["decisions_v"]
    ed_keys = set(ed.files) if hasattr(ed, "files") else set(ed)
    if "ground_degeneracy" not in ed_keys:
        raise RuntimeError(
            "exact-diagonalization artifact lacks ground_degeneracy; "
            "rerun phase 4 before phase 5")
    ground_degeneracy = np.asarray(ed["ground_degeneracy"], dtype=np.int64)
    # ``promising`` remains a diagnostic; every aligned snapshot is evaluated.
    n_promising = int(np.sum(promising))
    print(f"  {scenario} Re={Re} dim={n_patches}: "
          f"{n_promising}/{len(promising)} snapshots 'promising' "
          "(diagnostic; all snapshots are evaluated)")

    all_results = []
    phi_ema = None
    flux_cursor = 0

    for idx in range(len(snap_indices)):
        si = int(snap_indices[idx])
        snapshot_seed = (int(seed) + idx) % (2**32)

        # Advance the same EMA used by the deployed pipeline, including DNS
        # snapshots that are not part of the phase-4 evaluation subset.
        while flux_cursor < si:
            phi = _stress_flux_for_snapshot(
                vx_all[flux_cursor], vy_all[flux_cursor],
                Bx_all[flux_cursor], By_all[flux_cursor], N,
            )
            phi_ema = _ema_update(phi_ema, phi)
            flux_cursor += 1

        vx = vx_all[si].astype(np.float64)
        vy = vy_all[si].astype(np.float64)
        Bx = Bx_all[si].astype(np.float64)
        By = By_all[si].astype(np.float64)

        # prepare inputs
        data_in, hamilt_params, score_vqa = prepare_qaoa_inputs(
            vx, vy, Bx, By, N, n_patches, Re, use_v2=use_v2,
            prev_phi=phi_ema,
            with_psi=(phi_ema is not None and not zero_psi),
        )

        # Each snapshot is an independent solver benchmark. This avoids an
        # order-dependent optimiser state in the QAOA-vs-exact comparison.
        initial_params = (
            constant_initial_params(reps)
            if constant_initialisation else None
        )

        # run QAOA
        marginals, qaoa_h, qaoa_v, optimal_params, wall_time = \
            run_qaoa_on_snapshot(
                data_in, hamilt_params, n_patches,
                reps=reps, K_opt=K_opt,
                backend_name=backend_name,
                warm_start_params=initial_params,
                prune_eps=prune_eps,
                seed=snapshot_seed,
            )

        phi_ema = _ema_update(
            phi_ema, _stress_flux_for_snapshot(vx, vy, Bx, By, N))
        flux_cursor = si + 1

        # ground truth
        gt_refine = l2_all[si] >= l2_threshold

        # comparison
        comp = full_comparison(
            qaoa_h, qaoa_v,
            ed_decisions_h[idx], ed_decisions_v[idx],
            gt_refine, score_vqa,
            V2_THRESHOLD if use_v2 else TRAINED_THRESHOLD,
            int(ground_degeneracy[idx]),
        )
        degeneracy = int(ground_degeneracy[idx])
        comp.setdefault("exact_ground_degeneracy", degeneracy)
        comp.setdefault(
            "qaoa_exact_agreement_raw",
            float(comp.get("qaoa_exact_agreement", np.nan)))
        if degeneracy != 1:
            comp["qaoa_exact_agreement"] = np.nan

        print(f"    snap {si:3d}: "
              f"QAOA_F1={comp['qaoa']['f1']:.3f} "
              f"exact_F1={comp['exact']['f1']:.3f} "
              f"class_F1={comp['classical']['f1']:.3f} "
              f"QAOA-exact agree={comp['qaoa_exact_agreement']:.2f} "
              f"ground degeneracy={degeneracy} "
              f"({wall_time:.1f}s)")

        all_results.append({
            "snap_idx": si,
            "marginals": np.array(marginals),
            "qaoa_h": qaoa_h,
            "qaoa_v": qaoa_v,
            "optimal_params": optimal_params,
            "wall_time": wall_time,
            "seed": snapshot_seed,
            "comparison": comp,
        })

    if not all_results:
        return None, None

    # summary
    qaoa_f1s = [r["comparison"]["qaoa"]["f1"] for r in all_results]
    exact_f1s = [r["comparison"]["exact"]["f1"] for r in all_results]
    class_f1s = [r["comparison"]["classical"]["f1"] for r in all_results]
    agrees = [r["comparison"]["qaoa_exact_agreement"] for r in all_results]

    print(f"\n  Summary ({len(all_results)} evaluated snapshots):")
    print(f"    QAOA F1:       mean={np.mean(qaoa_f1s):.3f} "
          f"std={np.std(qaoa_f1s):.3f}")
    print(f"    Exact diag F1: mean={np.mean(exact_f1s):.3f} "
          f"std={np.std(exact_f1s):.3f}")
    print(f"    Classical F1:  mean={np.mean(class_f1s):.3f} "
          f"std={np.std(class_f1s):.3f}")
    if np.any(np.isfinite(agrees)):
        print(f"    QAOA-exact agreement (unique ground states only): "
              f"mean={np.nanmean(agrees):.3f}")
    else:
        print("    QAOA-exact agreement: undefined (all ground states degenerate)")

    # verdict
    qaoa_wins = sum(q > c for q, c in zip(qaoa_f1s, class_f1s))
    print(f"\n    QAOA beats classical: {qaoa_wins}/{len(all_results)} snapshots")

    meta = {
        "scenario": scenario, "Re": Re, "N": N,
        "n_patches": n_patches, "reps": reps, "K_opt": K_opt,
        "seed": int(seed),
        "backend": backend_name,
        "constant_initialisation": bool(constant_initialisation),
        "prune_eps": float(prune_eps),
        "zero_psi": bool(zero_psi),
        "suffix": ("_zeropsi" if zero_psi else "")
                  + ("_v2" if use_v2 else ""),
    }

    return all_results, meta


def save_results(all_results, meta, outdir=RESULTS_DIR, *,
                 run_provenance=None, cli_args=None):
    """Save Phase 5 results."""
    if all_results is None:
        return None

    variant = (
        f"_p{meta['reps']}_k{meta['K_opt']}_{meta['backend']}"
        f"_seed{meta['seed']}"
        + ("_constinit" if meta.get("constant_initialisation") else "")
        + (f"_prune{meta['prune_eps']:g}"
           if meta.get("prune_eps", 0.0) > 0.0 else "")
    )
    suffix = variant + meta.get("suffix", "")
    fname = (f"qaoa_eval_{meta['scenario']}_Re{meta['Re']}"
             f"_N{meta['N']}_dim{meta['n_patches']}{suffix}.npz")
    path = os.path.join(outdir, fname)

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
    agreement_raw = np.array(
        [r["comparison"]["qaoa_exact_agreement_raw"] for r in all_results])
    ground_degeneracy = np.array(
        [r["comparison"]["exact_ground_degeneracy"] for r in all_results],
        dtype=np.int64)
    seeds = np.array([r["seed"] for r in all_results], dtype=np.uint32)

    provenance_fields = (
        provenance.finish(run_provenance)
        if run_provenance is not None else {})
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
        qaoa_exact_agreement_raw=agreement_raw,
        exact_ground_degeneracy=ground_degeneracy,
        seeds=seeds,
        seed=meta["seed"],
        zero_psi=meta.get("zero_psi", False),
        scenario=meta["scenario"],
        Re=meta["Re"],
        N=meta["N"],
        n_patches=meta["n_patches"],
        reps=meta["reps"],
        K_opt=meta["K_opt"],
        backend=meta["backend"],
        constant_initialisation=meta.get("constant_initialisation", False),
        prune_eps=meta.get("prune_eps", 0.0),
        cli_args="" if cli_args is None else cli_args,
        **provenance_fields,
    )
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {fname} ({size_kb:.0f} KB)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: QAOA evaluation on the phase-4 panel")
    parser.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    parser.add_argument("--scenario", nargs="+", default=SCENARIOS)
    parser.add_argument("--dim", nargs="+", type=int, default=VQA_DIMS)
    parser.add_argument("--N", type=int, default=DNS_N)
    parser.add_argument("--reps", type=int, default=2,
                        help="QAOA circuit depth (number of layers)")
    parser.add_argument("--K_opt", type=int, default=100,
                        help="Max COBYLA iterations")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for transpilation and quantum shots")
    parser.add_argument("--v2", action="store_true",
                        help="Use the a-priori v2 Hamiltonian")
    parser.add_argument("--backend", default="state_vector",
                        choices=["state_vector", "matrix_product_state",
                                 "aer"],
                        help="Aer simulation method (use matrix_product_state "
                             "for dim>=3 scaling)")
    parser.add_argument("--constant-init", action="store_true",
                        help="Initialise each independent QAOA optimisation "
                             "with a fixed (beta, gamma) schedule")
    parser.add_argument("--zero-psi", action="store_true",
                        help="Ablate the temporal stress-flux phase")
    parser.add_argument("--prune-eps", type=float, default=0.0,
                        help="Prune |coeff| < eps * max per block before "
                             "QAOA (0 = no pruning)")
    args = parser.parse_args()
    run_provenance = provenance.start()
    cli_args = json.dumps(vars(args), sort_keys=True)

    version = "v2" if args.v2 else "v1"
    print(f"Phase 5: QAOA evaluation on the phase-4 snapshot panel ({version})")
    print(f"  Patch dims: {args.dim}, reps={args.reps}, K_opt={args.K_opt}")
    print(f"  Backend:    {args.backend}")
    print(f"  Seed:       {args.seed}")
    if args.constant_init:
        print("  QAOA initialisation: fixed schedule per snapshot")
    print(f"  Temporal phase: {'ablated' if args.zero_psi else 'deployed EMA'}")
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
                        constant_initialisation=args.constant_init,
                        prune_eps=args.prune_eps,
                        seed=args.seed,
                        zero_psi=args.zero_psi,
                    )
                    if results is not None:
                        save_results(
                            results, meta,
                            run_provenance=run_provenance,
                            cli_args=cli_args)
                        final_summary[(sc, re, dim)] = {
                            "qaoa_f1": np.mean([r["comparison"]["qaoa"]["f1"]
                                                for r in results]),
                            "exact_f1": np.mean([r["comparison"]["exact"]["f1"]
                                                 for r in results]),
                            "class_f1": np.mean([r["comparison"]["classical"]["f1"]
                                                 for r in results]),
                            "n_evaluated": len(results),
                        }
                    print()

    if not final_summary:
        raise RuntimeError(
            "empty sweep: no (scenario, Re, dim) produced a QAOA result")

    if final_summary:
        print("=" * 70)
        print("PHASE 5 SUMMARY")
        print("=" * 70)
        print(f"  {'Scenario':<16} {'Re':>4} {'dim':>4}  "
              f"{'QAOA_F1':>8} {'Exact_F1':>8} {'Class_F1':>8} "
              f"{'N_eval':>6}")
        for (sc, re, dim), s in sorted(final_summary.items()):
            print(f"  {sc:<16} {re:>4} {dim:>4}  "
                  f"{s['qaoa_f1']:>8.3f} {s['exact_f1']:>8.3f} "
                  f"{s['class_f1']:>8.3f} {s['n_evaluated']:>6}")

        all_qaoa = [s["qaoa_f1"] for s in final_summary.values()]
        all_class = [s["class_f1"] for s in final_summary.values()]
        print(f"\n  Overall QAOA F1: {np.mean(all_qaoa):.3f}")
        print(f"  Overall Classical F1: {np.mean(all_class):.3f}")
        print(f"  Descriptive delta: "
              f"{np.mean(all_qaoa) - np.mean(all_class):+.3f}")
        print("  No advantage claim is made without paired uncertainty and "
              "budget-matched closed-loop evidence.")

    print("\nPhase 5 complete.")


if __name__ == "__main__":
    main()
