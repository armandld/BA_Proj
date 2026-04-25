#!/usr/bin/env python3
"""
Phase 2 - Identify hard patches using an objective L2 criterion.

For each DNS snapshot, divide the domain into coarse patches and compute:
  1. L2 error of each patch = how much information is lost by coarsening
  2. Classical AMR score of each patch
  3. "Hard" classification: patches where L2 error is high (needs refinement)
     but the classical score is ambiguous (near the decision boundary)

This gives an objective, physics-based ground truth for what "needs refinement"
means, independent of any threshold choice.

Input:  results/dns_{scenario}_Re{Re}_N{N}.npz  (from Phase 1)
Output: results/patches_{scenario}_Re{Re}_N{N}_dim{D}.npz

Usage:
  python study/phase2_hard_patches.py
  python study/phase2_hard_patches.py --re 800 --dim 4
"""
import argparse, os, sys, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import (
    RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N, VQA_DIMS, L2_PERCENTILE_HARD,
)
from Simulation.grid import PeriodicGrid
from Simulation.PhysToAngle import AngleMapper


# -------------------------------------------------------------------
# L2 error computation
# -------------------------------------------------------------------

def coarsen_field(field, factor):
    """
    Coarsen a 2D field by block-averaging with the given factor.
    Then prolong back to original resolution by nearest-neighbor.
    Returns the prolonged coarse field (same shape as input).
    """
    N = field.shape[0]
    n_coarse = N // factor
    # block average
    coarse = field.reshape(n_coarse, factor, n_coarse, factor).mean(axis=(1, 3))
    # prolong back (nearest neighbor = repeat)
    prolonged = np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)
    return prolonged


def patch_l2_errors(vx, vy, Bx, By, n_patches):
    """
    Compute per-patch L2 error: how much info is lost by coarsening to
    n_patches x n_patches resolution.

    For each patch, L2 = sqrt(mean((fine - coarse_prolonged)^2)) over all
    4 fields, normalized by the global RMS of the fine fields.

    Returns: (n_patches, n_patches) array of L2 errors.
    """
    N = vx.shape[0]
    patch_size = N // n_patches
    assert N % n_patches == 0, f"N={N} not divisible by n_patches={n_patches}"

    # coarsen each field
    vx_c = coarsen_field(vx, patch_size)
    vy_c = coarsen_field(vy, patch_size)
    Bx_c = coarsen_field(Bx, patch_size)
    By_c = coarsen_field(By, patch_size)

    # global normalization
    rms_global = np.sqrt(np.mean(vx**2 + vy**2 + Bx**2 + By**2))
    if rms_global < 1e-15:
        rms_global = 1.0

    # per-patch L2
    errors = np.zeros((n_patches, n_patches))
    for pi in range(n_patches):
        for pj in range(n_patches):
            i0, i1 = pi * patch_size, (pi + 1) * patch_size
            j0, j1 = pj * patch_size, (pj + 1) * patch_size

            diff_sq = (
                (vx[i0:i1, j0:j1] - vx_c[i0:i1, j0:j1]) ** 2
                + (vy[i0:i1, j0:j1] - vy_c[i0:i1, j0:j1]) ** 2
                + (Bx[i0:i1, j0:j1] - Bx_c[i0:i1, j0:j1]) ** 2
                + (By[i0:i1, j0:j1] - By_c[i0:i1, j0:j1]) ** 2
            )
            errors[pi, pj] = np.sqrt(np.mean(diff_sq)) / rms_global

    return errors


# -------------------------------------------------------------------
# Classical score computation
# -------------------------------------------------------------------

def patch_classical_scores(vx, vy, Bx, By, n_patches, dx):
    """
    Compute classical AMR score per patch using the 4-indicator RMS.
    Returns: (n_patches, n_patches) array of scores in [0, 1].
    """
    N = vx.shape[0]
    patch_size = N // n_patches

    # compute Jz = curl(B).z = dBy/dx - dBx/dy
    grad_By_x = (np.roll(By, -1, axis=0) - np.roll(By, 1, axis=0)) / (2.0 * dx)
    grad_Bx_y = (np.roll(Bx, -1, axis=1) - np.roll(Bx, 1, axis=1)) / (2.0 * dx)
    Jz = grad_By_x - grad_Bx_y

    # compute full-resolution indicators
    physics_state = {"vx": vx, "vy": vy, "Bx": Bx, "By": By,
                     "Jz": Jz, "dx": dx}
    full_score = AngleMapper.classical_score(physics_state)

    # block-max pool to patch resolution
    scores = np.zeros((n_patches, n_patches))
    for pi in range(n_patches):
        for pj in range(n_patches):
            i0, i1 = pi * patch_size, (pi + 1) * patch_size
            j0, j1 = pj * patch_size, (pj + 1) * patch_size
            scores[pi, pj] = np.max(full_score[i0:i1, j0:j1])

    return scores


# -------------------------------------------------------------------
# Per-cell gradient-based error (finer than patch L2)
# -------------------------------------------------------------------

def cell_gradient_error(vx, vy, Bx, By, dx):
    """
    Pixel-level error indicator based on gradient magnitude of all fields.
    This is a proxy for "how much fine structure exists at this location."
    Returns: (N, N) array.
    """
    err = np.zeros_like(vx)
    for f in [vx, vy, Bx, By]:
        gx = np.gradient(f, dx, axis=0)
        gy = np.gradient(f, dx, axis=1)
        err += gx**2 + gy**2
    return np.sqrt(err)


# -------------------------------------------------------------------
# Main analysis
# -------------------------------------------------------------------

def analyze_dns_file(dns_path, n_patches_list):
    """
    Analyze one DNS file, return patch data for all requested patch dims.
    """
    data = np.load(dns_path)
    vx_all = data["vx"]
    vy_all = data["vy"]
    Bx_all = data["Bx"]
    By_all = data["By"]
    t_all = data["t"]
    N = vx_all.shape[1]
    dx = 2 * np.pi / N

    scenario = str(data.get("meta_scenario", "unknown"))
    Re = int(data.get("meta_Re", 0))
    n_snaps = len(t_all)

    print(f"  Loaded: {os.path.basename(dns_path)} "
          f"({scenario} Re={Re} N={N}, {n_snaps} snapshots)")

    results_by_dim = {}

    for n_p in n_patches_list:
        if N % n_p != 0:
            print(f"    SKIP dim={n_p}: N={N} not divisible")
            continue

        all_l2 = []          # (n_snaps, n_p, n_p)
        all_scores = []      # (n_snaps, n_p, n_p)
        all_cell_err = []    # (n_snaps, N, N)

        for si in range(n_snaps):
            vx = vx_all[si].astype(np.float64)
            vy = vy_all[si].astype(np.float64)
            Bx = Bx_all[si].astype(np.float64)
            By = By_all[si].astype(np.float64)

            l2 = patch_l2_errors(vx, vy, Bx, By, n_p)
            sc = patch_classical_scores(vx, vy, Bx, By, n_p, dx)
            ce = cell_gradient_error(vx, vy, Bx, By, dx)

            all_l2.append(l2)
            all_scores.append(sc)
            all_cell_err.append(ce)

        all_l2 = np.array(all_l2)          # (n_snaps, n_p, n_p)
        all_scores = np.array(all_scores)
        all_cell_err = np.array(all_cell_err)

        # classify hard patches (top percentile by L2)
        l2_flat = all_l2.flatten()
        threshold_l2 = np.percentile(l2_flat, L2_PERCENTILE_HARD)
        is_hard = all_l2 >= threshold_l2    # (n_snaps, n_p, n_p) bool

        n_hard = is_hard.sum()
        n_total = is_hard.size
        frac_hard = n_hard / n_total

        # score distribution at hard vs easy patches
        hard_scores = all_scores[is_hard]
        easy_scores = all_scores[~is_hard]

        print(f"    dim={n_p}: L2 threshold={threshold_l2:.6f}, "
              f"hard={n_hard}/{n_total} ({100*frac_hard:.1f}%)")
        print(f"      Hard patch scores:  mean={hard_scores.mean():.4f} "
              f"std={hard_scores.std():.4f} "
              f"[{hard_scores.min():.4f}, {hard_scores.max():.4f}]")
        print(f"      Easy patch scores:  mean={easy_scores.mean():.4f} "
              f"std={easy_scores.std():.4f}")

        results_by_dim[n_p] = {
            "l2_errors": all_l2,
            "classical_scores": all_scores,
            "cell_gradient_error": all_cell_err,
            "is_hard": is_hard,
            "l2_threshold": threshold_l2,
        }

    return results_by_dim, {
        "scenario": scenario, "Re": Re, "N": N,
        "t": t_all, "n_snaps": n_snaps,
    }


def save_patches(results_by_dim, meta, outdir=RESULTS_DIR):
    """Save patch analysis for each dim."""
    paths = []
    for n_p, res in results_by_dim.items():
        fname = (f"patches_{meta['scenario']}_Re{meta['Re']}"
                 f"_N{meta['N']}_dim{n_p}.npz")
        path = os.path.join(outdir, fname)
        np.savez_compressed(
            path,
            l2_errors=res["l2_errors"],
            classical_scores=res["classical_scores"],
            is_hard=res["is_hard"],
            l2_threshold=res["l2_threshold"],
            t=meta["t"],
            scenario=meta["scenario"],
            Re=meta["Re"],
            N=meta["N"],
            n_patches=n_p,
        )
        size_kb = os.path.getsize(path) / 1024
        print(f"    Saved: {fname} ({size_kb:.0f} KB)")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Hard patch identification")
    parser.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    parser.add_argument("--scenario", nargs="+", default=SCENARIOS)
    parser.add_argument("--dim", nargs="+", type=int, default=VQA_DIMS)
    parser.add_argument("--N", type=int, default=DNS_N)
    args = parser.parse_args()

    print("Phase 2: Hard patch identification")
    print(f"  Patch dims: {args.dim}")
    print()

    dns_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "dns_*.npz")))
    if not dns_files:
        print("No DNS files found. Run phase1_dns_sweep.py first.")
        return

    # filter by requested scenario/Re
    for dns_path in dns_files:
        data = np.load(dns_path, allow_pickle=True)
        sc = str(data.get("meta_scenario", ""))
        re = int(data.get("meta_Re", 0))
        if sc not in args.scenario or re not in args.re:
            continue

        print(f"Analyzing: {os.path.basename(dns_path)}")
        results_by_dim, meta = analyze_dns_file(dns_path, args.dim)
        save_patches(results_by_dim, meta)
        print()

    print("Phase 2 complete.")


if __name__ == "__main__":
    main()
