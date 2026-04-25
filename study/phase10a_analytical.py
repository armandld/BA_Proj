#!/usr/bin/env python3
"""
Phase 10a - Analytical / mean-field derivation of (c_bias*, thr_amr*).

Rather than running a noisy closed-loop optimiser blind from x0=(1, 0.15),
phase 10a exploits the fact that:

  (1) The Hamiltonian's Z bias is h_i = c_bias * M * (score_i - thr).
      If c_bias is large enough that h_i dominates the couplings, the
      per-site ground state equals the classical indicator (score > thr).
      So thr* can be found by a 1-D F1 sweep on the classical score
      against the L2-hard mask — NO optimiser, no SA.

  (2) c_bias controls the trade-off: too small -> FM couplings win,
      uniform state; too large -> biases win, = classical indicator.
      The useful regime is where strong-signal sites follow their bias
      and weak-signal sites get corrected by their neighbours.
      We locate it by a zero-temperature mean-field (MF) iteration on
      the actual Ising + 4-body graph, with c_bias swept on a log-grid,
      using thr* from step (1). The MF prediction is compared to GT,
      F1 maximised -> c_bias*.

Output: results/analytical_N{N}_dim{D}.npz
          keys: tag, thr_star, c_bias_star, f1_mf, classical_f1
                (one row per scenario + one 'joint' row)

Phase 10 (closed-loop training) reads this file if it exists and uses
(log10 c_bias*, thr*) as THETA_INIT per mode.

Convention (as in phase 7): spin +1 = "don't refine", -1 = "refine".
GT is +1 when L2 error >= threshold (refine). So h_i > 0 wants s_i=-1
(refine), which matches score > thr <=> refine.

Usage:
  python study/phase10a_analytical.py --dim 4
  python study/phase10a_analytical.py --dim 4 --scenario mhd_rotor
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N

from phase4_exact_diag import build_patch_hamiltonian
from phase7_sa_baseline import (
    build_ising_terms, spins_to_decisions, _metrics, _build_incidence,
)


# -------------------------------------------------------------------
# 1D threshold sweep (classical indicator F1 vs L2-hard mask)
# -------------------------------------------------------------------

def best_threshold(scores_pool, gt_pool, grid=None):
    """Return (thr*, F1*) from a 1-D grid sweep."""
    if grid is None:
        # mix of uniform + score-quantiles for robustness
        q = np.quantile(scores_pool, np.linspace(0.05, 0.95, 19))
        grid = np.unique(np.concatenate([
            np.linspace(0.02, 0.60, 59), q
        ]))
    best_thr, best_f1 = float(grid[0]), -1.0
    for thr in grid:
        pred = scores_pool > thr
        tp = int(((pred == 1) & (gt_pool == 1)).sum())
        fp = int(((pred == 1) & (gt_pool == 0)).sum())
        fn = int(((pred == 0) & (gt_pool == 1)).sum())
        denom = (2 * tp + fp + fn)
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


# -------------------------------------------------------------------
# Zero-temperature MF iteration on the Ising + 4-body graph
# -------------------------------------------------------------------

def mean_field_ground(h_bias, edges, plaqs, n_q,
                      max_iter=200, init_spins=None, rng=None):
    """Asynchronous zero-T Glauber (per-site greedy) until convergence."""
    edge_idx, edge_coef = edges
    plaq_idx, plaq_coef = plaqs
    edges_by_q, plaqs_by_q = _build_incidence(n_q, edges, plaqs)

    if init_spins is None:
        spins = np.ones(n_q, dtype=np.int8)     # start at "don't refine"
    else:
        spins = init_spins.astype(np.int8).copy()

    # random scan order for symmetry-breaking
    if rng is None:
        rng = np.random.default_rng(0)

    for it in range(max_iter):
        changed = False
        order = rng.permutation(n_q)
        for q in order:
            h_eff = h_bias[q]
            for e_idx in edges_by_q[q]:
                i, j = edge_idx[e_idx]
                other = j if i == q else i
                h_eff += edge_coef[e_idx] * spins[other]
            for p_idx in plaqs_by_q[q]:
                i, j, k, l = plaq_idx[p_idx]
                prod = 1
                for qq in (i, j, k, l):
                    if qq != q:
                        prod *= spins[qq]
                h_eff += plaq_coef[p_idx] * prod
            # minimise h_eff * s_q  ->  s_q = -sign(h_eff), ties: keep
            if h_eff > 0 and spins[q] != -1:
                spins[q] = -1; changed = True
            elif h_eff < 0 and spins[q] != 1:
                spins[q] = 1; changed = True
        if not changed:
            break
    return spins, it + 1


# -------------------------------------------------------------------
# MF F1 as a function of c_bias for one snapshot
# -------------------------------------------------------------------

def mf_f1_curve(vx, vy, Bx, By, N, dim, Re,
                thr_amr, c_bias_grid, gt_mask, rng):
    """For one snapshot, return F1(c_bias) via MF decoding."""
    # Build with c_bias=1; scale h_bias afterwards (linear in c_bias).
    hp_unit, _, _ = build_patch_hamiltonian(
        vx, vy, Bx, By, N, dim, Re,
        threshold_amr=thr_amr, use_v2=True, c_bias=1.0,
    )
    h_unit, edges, plaqs = build_ising_terms(hp_unit, dim)
    n_q = 2 * dim * dim

    f1s = []
    for c in c_bias_grid:
        h_bias = h_unit * float(c)
        spins, _ = mean_field_ground(h_bias, edges, plaqs, n_q,
                                     max_iter=100, rng=rng)
        dec_h, dec_v = spins_to_decisions(spins, dim)
        refine = dec_h | dec_v
        f1s.append(_metrics(refine, gt_mask)["f1"])
    return np.array(f1s)


# -------------------------------------------------------------------
# Per-scenario analysis
# -------------------------------------------------------------------

def analyse_snapshots(dns_path, patches_path, dim, *,
                       c_grid, max_snaps, seed):
    dns = np.load(dns_path)
    patches = np.load(patches_path)
    vx = dns["vx"]; vy = dns["vy"]
    Bx = dns["Bx"]; By = dns["By"]
    N  = vx.shape[1]
    Re = int(dns.get("meta_Re", 800))
    scenario = str(dns.get("meta_scenario", "unknown"))

    l2_all = patches["l2_errors"]
    l2_thr = float(patches["l2_threshold"])

    n_snaps = len(vx)
    step = max(1, n_snaps // max_snaps)
    snap_indices = list(range(0, n_snaps, step))[:max_snaps]

    # ---- pool scores & GT for threshold sweep ----
    scores_pool = []
    gt_pool     = []
    rng = np.random.default_rng(seed)

    # we also remember (vx, vy, Bx, By, gt_mask) per snap for the MF
    # pass so we build the patch Hamiltonian only once per snap
    snap_cache = []
    for si in snap_indices:
        hp, score_vqa, _ = build_patch_hamiltonian(
            vx[si].astype(np.float64), vy[si].astype(np.float64),
            Bx[si].astype(np.float64), By[si].astype(np.float64),
            N, dim, Re, threshold_amr=0.15, use_v2=True, c_bias=1.0,
        )
        gt_mask = l2_all[si] >= l2_thr
        scores_pool.append(score_vqa.ravel())
        gt_pool.append(gt_mask.ravel().astype(int))
        snap_cache.append(dict(
            vx=vx[si].astype(np.float64), vy=vy[si].astype(np.float64),
            Bx=Bx[si].astype(np.float64), By=By[si].astype(np.float64),
            gt_mask=gt_mask, score=score_vqa,
        ))
    scores_pool = np.concatenate(scores_pool)
    gt_pool     = np.concatenate(gt_pool)

    # ---- (1) thr* ----
    thr_star, f1_class = best_threshold(scores_pool, gt_pool)

    # ---- (2) c_bias* via MF with thr_star ----
    f1_grid = np.zeros_like(c_grid)
    for sc in snap_cache:
        f1_grid += mf_f1_curve(
            sc["vx"], sc["vy"], sc["Bx"], sc["By"],
            N, dim, Re,
            thr_star, c_grid, sc["gt_mask"], rng,
        )
    f1_grid /= len(snap_cache)

    bi = int(np.argmax(f1_grid))
    c_star = float(c_grid[bi])
    f1_mf  = float(f1_grid[bi])

    return dict(
        scenario=scenario, Re=Re, dim=dim, N=N,
        thr_star=thr_star, c_bias_star=c_star,
        f1_mf=f1_mf, classical_f1=f1_class,
        f1_grid=f1_grid, c_grid=np.array(c_grid),
        snap_indices=np.array(snap_indices),
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Phase 10a: MF-analytical init for (c_bias, thr)")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=8,
                   help="snapshots per (scenario, Re) for MF curve")
    p.add_argument("--n-cgrid", type=int, default=25,
                   help="c_bias points on log grid [0.1, 100]")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    c_grid = np.logspace(-1.0, 2.0, args.n_cgrid)
    print("=" * 88)
    print("  Phase 10a: MF-analytical derivation of (c_bias*, thr*)")
    print(f"  c_bias grid: log10 [-1, 2]   ({args.n_cgrid} points)")
    print(f"  snaps/config: {args.max_snaps}")
    print("=" * 88)
    print()

    # Group by scenario: one (thr*, c_bias*) per scenario, pooled over Re
    # First compute per-config rows, then aggregate.
    per_cfg = []
    for sc in args.scenario:
        for re in args.re:
            dns_path = os.path.join(
                RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            patches_path = os.path.join(
                RESULTS_DIR,
                f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if not (os.path.exists(dns_path) and os.path.exists(patches_path)):
                print(f"  SKIP {sc} Re={re}: missing input")
                continue
            t0 = time.time()
            res = analyse_snapshots(
                dns_path, patches_path, args.dim,
                c_grid=c_grid, max_snaps=args.max_snaps, seed=args.seed)
            dt = time.time() - t0
            per_cfg.append(res)
            print(f"  [{sc} Re={re}] thr*={res['thr_star']:.3f} "
                  f"c_bias*={res['c_bias_star']:.2f}  "
                  f"F1_MF={res['f1_mf']:.3f}  "
                  f"classical={res['classical_f1']:.3f}   [{dt:.1f}s]")

    if not per_cfg:
        print("no input.")
        return

    # ---- per-scenario aggregation ----
    by_scene = {}
    for r in per_cfg:
        by_scene.setdefault(r["scenario"], []).append(r)

    scenario_rows = []
    print("\n  per-scenario (mean over Re):")
    for sc, rows in by_scene.items():
        thr = float(np.mean([r["thr_star"] for r in rows]))
        c   = float(np.mean([r["c_bias_star"] for r in rows]))
        f1m = float(np.mean([r["f1_mf"] for r in rows]))
        f1c = float(np.mean([r["classical_f1"] for r in rows]))
        scenario_rows.append(dict(
            tag=f"scenario:{sc}", thr_star=thr, c_bias_star=c,
            f1_mf=f1m, classical_f1=f1c))
        print(f"    {sc:<18} thr*={thr:.3f}  c_bias*={c:.2f}  "
              f"F1_MF={f1m:.3f}  classical={f1c:.3f}")

    # ---- joint (mean over everything) ----
    thr_j = float(np.mean([r["thr_star"]    for r in per_cfg]))
    c_j   = float(np.mean([r["c_bias_star"] for r in per_cfg]))
    f1mj  = float(np.mean([r["f1_mf"]       for r in per_cfg]))
    f1cj  = float(np.mean([r["classical_f1"] for r in per_cfg]))
    joint_row = dict(tag="joint", thr_star=thr_j, c_bias_star=c_j,
                     f1_mf=f1mj, classical_f1=f1cj)
    print(f"\n  joint  thr*={thr_j:.3f}  c_bias*={c_j:.2f}  "
          f"F1_MF={f1mj:.3f}  classical={f1cj:.3f}")

    # ---- per-config (for reference) ----
    cfg_rows = []
    for r in per_cfg:
        cfg_rows.append(dict(
            tag=f"cfg:{r['scenario']}_Re{r['Re']}",
            thr_star=r["thr_star"], c_bias_star=r["c_bias_star"],
            f1_mf=r["f1_mf"], classical_f1=r["classical_f1"],
        ))

    # ---- save ----
    all_rows = scenario_rows + [joint_row] + cfg_rows
    out = os.path.join(
        RESULTS_DIR, f"analytical_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        tags=np.array([r["tag"]         for r in all_rows]),
        thr_star=np.array([r["thr_star"]     for r in all_rows]),
        c_bias_star=np.array([r["c_bias_star"] for r in all_rows]),
        f1_mf=np.array([r["f1_mf"]       for r in all_rows]),
        classical_f1=np.array([r["classical_f1"] for r in all_rows]),
        c_grid=c_grid,
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 10a complete.")


if __name__ == "__main__":
    main()
