#!/usr/bin/env python3
"""
Phase 3 - Compute Hamiltonian coefficients on all patches, analyze
correlation with L2 error, and test threshold stability across Re.

For each patch at each snapshot:
  1. Compute C_ij (ZZ), K_ijkl (ZZZZ), H_i (Z) coefficients
  2. Compute per-cell energy: E_i = |H_i| + sum_j|C_ij| + sum_p|K_p|
  3. Correlate E_i with L2 error (is the Hamiltonian pointing at hard patches?)
  4. Find optimal threshold that best separates hard from easy patches
  5. Test whether this threshold is stable across Re values

Input:  results/dns_{scenario}_Re{Re}_N{N}.npz
        results/patches_{scenario}_Re{Re}_N{N}_dim{D}.npz
Output: results/coefficients_{scenario}_Re{Re}_N{N}_dim{D}.npz
        Printed analysis and correlation tables.

Usage:
  python study/hamiltonian_coefficients.py
  python study/hamiltonian_coefficients.py --re 800 --dim 4
"""
import argparse, os, sys, glob
import numpy as np
from scipy.stats import spearmanr, pearsonr

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
    TRAINED_SIGMA, TRAINED_THRESHOLD, trained_mapper_params,
    V2_THRESHOLD,
)
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.HamiltParams import PhysicalMapper
from Simulation.HamiltParams_v2 import PhysicalMapperV2
from Simulation.PhysToAngle import AngleMapper


def compute_patch_coefficients(vx, vy, Bx, By, N, n_patches,
                               Re, threshold_amr,
                               sigma=None, beta_curl=None, beta_xpoint=None,
                               w_z_frac=None,
                               gamma_hydro=None, gamma_mag=None, kappa=None,
                               relative_percentile=None,
                               use_v2=False):
    """
    Compute Hamiltonian coefficients for each patch.

    If use_v2=True, uses the a-priori PhysicalMapperV2 constants.
    Otherwise uses the trained PhysicalMapper (v1).

    Returns dict with:
      H_all:  (n_patches, n_patches) - mean |H_i| per patch (Z bias)
      C_all:  (n_patches, n_patches) - mean |C_ij| per patch (ZZ coupling)
      K_all:  (n_patches, n_patches) - mean |K_p| per patch (ZZZZ plaquette)
      E_all:  (n_patches, n_patches) - total energy per patch
      score_all: (n_patches, n_patches) - classical score per patch
    """
    dx = 2 * np.pi / N
    patch_size = N // n_patches
    nu = 1.0 / Re
    eta = 1.0 / Re  # Rm = Re

    if use_v2:
        mapper = PhysicalMapperV2(dx=dx)
    else:
        mapper = PhysicalMapper(
            cs=1.0, nu=nu, eta_mhd=eta, dx=dx,
            gamma_hydro=gamma_hydro, gamma_mag=gamma_mag,
            kappa=kappa, sigma=sigma,
            beta_curl=beta_curl, beta_xpoint=beta_xpoint,
            w_z_frac=w_z_frac,
            relative_percentile=relative_percentile,
        )

    # full-resolution coefficients
    grid = PeriodicGrid(N)
    sim = MHDSolver(grid, dt=1e-4, Re=Re, Rm=Re)
    sim.vx, sim.vy, sim.Bx, sim.By = vx, vy, Bx, By

    fields = sim.get_fluxes()

    # full-resolution classical score (needs Jz)
    physics_state = {"vx": vx, "vy": vy, "Bx": Bx, "By": By,
                     "Jz": fields["Jz"], "dx": dx}
    full_score = AngleMapper.classical_score(physics_state)
    coeffs = mapper.compute_coefficients(
        sim, full_score, fields, threshold_amr,
        advanced_anomalies_enabled=True,
        verbose=False,
    )

    H_h, H_v = coeffs["H_edges"]       # (N, N) each
    C_h, C_v = coeffs["C_edges"]       # (N, N) each
    K_plaq = coeffs["K_plaquettes"]    # (N, N)
    K_xp = coeffs.get("K_xpoint")     # (N, N) or None

    # combine into per-cell energy
    H_mag = np.abs(H_h) + np.abs(H_v)                           # Z bias
    C_mag = np.abs(C_h) + np.abs(C_v)                           # ZZ coupling
    K_mag = np.abs(K_plaq) + (np.abs(K_xp) if K_xp is not None else 0)  # ZZZZ

    E_cell = H_mag + C_mag + K_mag  # total Hamiltonian energy per cell

    # block-reduce to patch resolution
    H_all = np.zeros((n_patches, n_patches))
    C_all = np.zeros_like(H_all)
    K_all = np.zeros_like(H_all)
    E_all = np.zeros_like(H_all)
    score_all = np.zeros_like(H_all)

    for pi in range(n_patches):
        for pj in range(n_patches):
            i0, i1 = pi * patch_size, (pi + 1) * patch_size
            j0, j1 = pj * patch_size, (pj + 1) * patch_size

            H_all[pi, pj] = np.mean(H_mag[i0:i1, j0:j1])
            C_all[pi, pj] = np.mean(C_mag[i0:i1, j0:j1])
            K_all[pi, pj] = np.mean(K_mag[i0:i1, j0:j1])
            E_all[pi, pj] = np.mean(E_cell[i0:i1, j0:j1])
            score_all[pi, pj] = np.max(full_score[i0:i1, j0:j1])

    return {
        "H_patch": H_all,
        "C_patch": C_all,
        "K_patch": K_all,
        "E_patch": E_all,
        "score_patch": score_all,
        # raw per-cell data
        "H_cell": H_mag,
        "C_cell": C_mag,
        "K_cell": K_mag,
        "E_cell": E_cell,
        "score_cell": full_score,
    }


def find_optimal_threshold(energy_values, is_hard):
    """
    Sweep thresholds on the energy/score to find the one that best
    separates hard from easy patches. Uses F1 score as the objective.

    Returns: (best_threshold, best_f1, all_thresholds, all_f1s)

    Si `energy_values` est CONSTANT, aucun seuil ne separe quoi que ce
    soit : la fonction rend NaN, pas un F1.
    """
    flat_e = energy_values.flatten()
    flat_h = is_hard.flatten()

    # D-43 : sur une entree constante, les 100 percentiles valent tous la
    # meme chose, et `flat_e >= thr` predit alors TOUS les patchs durs. Le
    # F1 rendu etait celui du classifieur tout-positif, 2p/(p+1) : mesure
    # 0.400 sur harris_tearing (prevalence 0.250) et 0.376 sur
    # kelvin_helmholtz (0.231) a partir des artefacts reels
    # coefficients_*_Re400_N256_dim4.npz, ou l'energie v1 est
    # identiquement nulle (D-40/D-41). A cote du 0.519 authentique
    # d'orszag_tang, 0.400 se lit comme un signal reel un peu plus faible.
    # `labels_global_threshold.py` nomme deja 0.400 comme la signature du
    # classifieur constant tout-positif ; le sibling de cette fonction
    # dans pipeline_verification.py rend 0.000 et signale la degenerescence
    # sur exactement les memes donnees.
    if np.ptp(flat_e) < 1e-12:
        nan_sweep = np.full(100, np.nan)
        return np.nan, np.nan, nan_sweep, nan_sweep.copy()

    # sweep thresholds from 5th to 95th percentile
    thresholds = np.percentile(flat_e, np.linspace(5, 95, 100))
    f1_scores = []

    for thr in thresholds:
        pred = flat_e >= thr
        tp = np.sum(pred & flat_h)
        fp = np.sum(pred & ~flat_h)
        fn = np.sum(~pred & flat_h)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx], thresholds, f1_scores


def analyze_one(dns_path, patches_path, n_patches,
                sigma_values=None, use_v2=False):
    """
    Analyze coefficients for one (scenario, Re) combination.
    """
    dns = np.load(dns_path)
    patches = np.load(patches_path)

    vx_all = dns["vx"]
    vy_all = dns["vy"]
    Bx_all = dns["Bx"]
    By_all = dns["By"]
    N = vx_all.shape[1]
    Re = int(dns.get("meta_Re", 800))
    scenario = str(dns.get("meta_scenario", "unknown"))

    is_hard_all = patches["is_hard"]   # (n_snaps, n_p, n_p)
    l2_all = patches["l2_errors"]

    n_snaps = len(vx_all)
    # use a subset of snapshots for speed (every 3rd, min 5)
    snap_indices = list(range(0, n_snaps, max(1, n_snaps // 10)))
    if len(snap_indices) < 3:
        snap_indices = list(range(n_snaps))

    print(f"  Computing coefficients on {len(snap_indices)} snapshots "
          f"({'v2' if use_v2 else 'v1'})...")

    threshold_amr = V2_THRESHOLD if use_v2 else TRAINED_THRESHOLD

    # for v2 we run a single pass (no sigma sweep); for v1, sweep sigma
    if use_v2:
        sigma_values = [0.0]  # placeholder key, v2 ignores sigma
    elif sigma_values is None:
        sigma_values = [0.023, 0.05, 0.10, 0.15, 0.20, 0.30]

    results = {}
    for sigma in sigma_values:
        all_E = []
        all_C = []
        all_K = []
        all_H = []
        all_scores = []

        for si in snap_indices:
            vx = vx_all[si].astype(np.float64)
            vy = vy_all[si].astype(np.float64)
            Bx = Bx_all[si].astype(np.float64)
            By = By_all[si].astype(np.float64)

            if use_v2:
                coeffs = compute_patch_coefficients(
                    vx, vy, Bx, By, N, n_patches, Re,
                    threshold_amr=threshold_amr,
                    use_v2=True,
                )
            else:
                mapper_params = trained_mapper_params()
                mapper_params["sigma"] = sigma
                coeffs = compute_patch_coefficients(
                    vx, vy, Bx, By, N, n_patches, Re,
                    threshold_amr=threshold_amr,
                    **mapper_params,
                )
            all_E.append(coeffs["E_patch"])
            all_C.append(coeffs["C_patch"])
            all_K.append(coeffs["K_patch"])
            all_H.append(coeffs["H_patch"])
            all_scores.append(coeffs["score_patch"])

        all_E = np.array(all_E)
        all_C = np.array(all_C)
        all_K = np.array(all_K)
        all_H = np.array(all_H)
        all_scores = np.array(all_scores)

        # subset is_hard to match snap_indices
        is_hard_sub = is_hard_all[snap_indices]
        l2_sub = l2_all[snap_indices]

        # correlations
        e_flat = all_E.flatten()
        l2_flat = l2_sub.flatten()
        hard_flat = is_hard_sub.flatten()

        rho_e, p_e = spearmanr(e_flat, l2_flat)
        rho_c, p_c = spearmanr(all_C.flatten(), l2_flat)
        rho_k, p_k = spearmanr(all_K.flatten(), l2_flat)
        rho_s, p_s = spearmanr(all_scores.flatten(), l2_flat)

        # optimal threshold on E
        best_thr_e, best_f1_e, _, _ = find_optimal_threshold(all_E, is_hard_sub)
        # optimal threshold on classical score
        best_thr_s, best_f1_s, _, _ = find_optimal_threshold(all_scores, is_hard_sub)

        # fraction of nonzero coefficients
        frac_C_active = np.mean(all_C.flatten() > 1e-10)
        frac_K_active = np.mean(all_K.flatten() > 1e-10)

        # D-43 : une energie constante ne classe rien ; le dire au lieu de
        # laisser un NaN passer pour un accident de calcul.
        degenerate_E = bool(np.ptp(all_E) < 1e-12)

        label = "v2" if use_v2 else f"sigma={sigma:.3f}"
        print(f"    {label}:")
        if degenerate_E:
            print(f"      DEGENERATE: E is constant over all patches and "
                  f"snapshots -- no threshold separates anything "
                  f"(no coefficient crossed a critical threshold)")
        print(f"      Spearman E vs L2:     rho={rho_e:.3f} (p={p_e:.1e})")
        print(f"      Spearman C(ZZ) vs L2: rho={rho_c:.3f} (p={p_c:.1e})")
        print(f"      Spearman K(ZZZZ) vs L2: rho={rho_k:.3f} (p={p_k:.1e})")
        print(f"      Spearman Score vs L2: rho={rho_s:.3f} (p={p_s:.1e})")
        print(f"      Best F1 (E threshold={best_thr_e:.4f}): {best_f1_e:.3f}")
        print(f"      Best F1 (Score thr={best_thr_s:.4f}):   {best_f1_s:.3f}")
        print(f"      Active C edges: {100*frac_C_active:.1f}%, "
              f"Active K plaquettes: {100*frac_K_active:.1f}%")

        results[sigma] = {
            "E": all_E, "C": all_C, "K": all_K, "H": all_H,
            "scores": all_scores, "l2": l2_sub, "is_hard": is_hard_sub,
            "rho_e": rho_e, "rho_c": rho_c, "rho_k": rho_k, "rho_s": rho_s,
            "best_thr_e": best_thr_e, "best_f1_e": best_f1_e,
            "best_thr_s": best_thr_s, "best_f1_s": best_f1_s,
            "frac_C_active": frac_C_active, "frac_K_active": frac_K_active,
            "degenerate_E": degenerate_E,
        }

    return results, {"scenario": scenario, "Re": Re, "N": N, "n_patches": n_patches}


def threshold_stability_report(all_results):
    """
    Print a table showing how optimal threshold varies with Re.
    If it's unstable, the Hamiltonian may need rethinking.
    """
    print("\n" + "=" * 70)
    print("THRESHOLD STABILITY ACROSS Re")
    print("=" * 70)

    # group by scenario
    by_scenario = {}
    for key, (results, meta) in all_results.items():
        sc = meta["scenario"]
        re = meta["Re"]
        if sc not in by_scenario:
            by_scenario[sc] = {}
        by_scenario[sc][re] = (results, meta)

    for sc, re_dict in sorted(by_scenario.items()):
        print(f"\n--- {sc} ---")
        print(f"  {'Re':>6}  {'sigma':>6}  {'Thr(E)':>8}  {'F1(E)':>6}  "
              f"{'Thr(Score)':>10}  {'F1(Score)':>9}  "
              f"{'rho(E,L2)':>9}  {'C_active':>8}")

        for re in sorted(re_dict.keys()):
            results, meta = re_dict[re]
            # report for trained sigma
            for sigma in [0.023, 0.10]:
                if sigma not in results:
                    continue
                r = results[sigma]
                # D-43 : une ligne degeneree affichait thr=0.0000 / F1=0.400
                # identiques a tous les Re, ce qui se lisait comme un seuil
                # PARFAITEMENT STABLE — la conclusion meme que cette table
                # existe pour produire.
                flag = "  <- DEGENERATE (E constant)" if r.get("degenerate_E") else ""
                print(f"  {re:>6}  {sigma:>6.3f}  {r['best_thr_e']:>8.4f}  "
                      f"{r['best_f1_e']:>6.3f}  "
                      f"{r['best_thr_s']:>10.4f}  {r['best_f1_s']:>9.3f}  "
                      f"{r['rho_e']:>9.3f}  {100*r['frac_C_active']:>7.1f}%{flag}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Coefficient analysis")
    parser.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    parser.add_argument("--scenario", nargs="+", default=SCENARIOS)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--N", type=int, default=DNS_N)
    parser.add_argument("--v2", action="store_true",
                        help="Use the a-priori v2 Hamiltonian")
    args = parser.parse_args()

    version = "v2" if args.v2 else "v1"
    print(f"Phase 3: Hamiltonian coefficient analysis ({version})")
    print(f"  Patch dim: {args.dim}")
    print()

    all_results = {}

    for sc in args.scenario:
        for re in args.re:
            dns_path = os.path.join(
                RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            patches_path = os.path.join(
                RESULTS_DIR,
                f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")

            if not os.path.exists(dns_path):
                print(f"  SKIP: {dns_path} not found")
                continue
            if not os.path.exists(patches_path):
                print(f"  SKIP: {patches_path} not found")
                continue

            print(f"[{sc} Re={re}]")
            results, meta = analyze_one(
                dns_path, patches_path, args.dim, use_v2=args.v2)
            all_results[(sc, re)] = (results, meta)

            # save
            suffix = "_v2" if args.v2 else ""
            fname = (f"coefficients_{sc}_Re{re}_N{args.N}"
                     f"_dim{args.dim}{suffix}.npz")
            path = os.path.join(RESULTS_DIR, fname)
            save_data = {}
            for sigma, r in results.items():
                prefix = f"s{sigma:.3f}_"
                save_data[prefix + "E"] = r["E"]
                save_data[prefix + "C"] = r["C"]
                save_data[prefix + "K"] = r["K"]
                save_data[prefix + "rho_e"] = r["rho_e"]
                save_data[prefix + "best_thr_e"] = r["best_thr_e"]
                save_data[prefix + "best_f1_e"] = r["best_f1_e"]
            np.savez_compressed(path, **save_data)
            print(f"  Saved: {fname}")
            print()

    if not all_results:
        # D-148 : meme famille que D-55/D-56/D-75, sur la phase 3. Mesure :
        # `--scenario no_such_scenario --N 64` sortait avec le code 0 apres
        # avoir imprime « Phase 3 complete. », sans ecrire d'artefact — donc
        # en laissant en place ceux de la campagne precedente. Le rapport de
        # stabilite etait deja garde par `if all_results:`, mais rien ne
        # criait quand la condition etait fausse.
        raise RuntimeError(
            "balayage vide : aucun couple (scenario, Re) n'a d'artefact "
            "d'entree pour les arguments donnes. Le script sortait ici avec "
            "le code 0 et sans artefact (D-148).")

    threshold_stability_report(all_results)

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
