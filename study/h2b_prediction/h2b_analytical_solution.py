#!/usr/bin/env python3
"""Phase 10a: deterministic initialisation for the V2 rescue fit.

The classical-score threshold is selected exactly on the chronological
training prefix. ``c_bias`` is swept on the Ising graph and decoded with
deterministic zero-temperature mean-field updates. Flat and unresolved
edge sweeps are rejected instead of being reported as optima.

Spin ``+1`` means "do not refine" and spin ``-1`` means "refine".

Usage:
  python study/h2b_prediction/h2b_analytical_solution.py --dim 4
"""
import argparse
import json
import os
import sys
import time
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
from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N, V2_THRESHOLD

from exact_diagonalisation import build_patch_hamiltonian
from ising_terms_and_annealing import (
    build_ising_terms, spins_to_decisions, _metrics, _build_incidence,
)
from h2b_train_linear_hamiltonian import (
    THETA_BOUNDS,
    chronological_split_indices,
    evenly_subsample,
)
import provenance


F1_SPAN_TOL = 1e-12
DEFAULT_C_MIN = 10.0 ** float(THETA_BOUNDS[0, 0])
DEFAULT_C_MAX = 10.0 ** float(THETA_BOUNDS[0, 1])


def c_bias_grid(c_min=DEFAULT_C_MIN, c_max=DEFAULT_C_MAX, n_points=31):
    """Logarithmic grid sharing the closed-loop optimizer's bounds."""
    if not (np.isfinite(c_min) and np.isfinite(c_max)
            and 0.0 < c_min < c_max):
        raise ValueError("c_min and c_max must be finite with 0 < min < max")
    if not isinstance(n_points, (int, np.integer)) or n_points < 3:
        raise ValueError("n_points must be an integer >= 3")
    return np.logspace(np.log10(c_min), np.log10(c_max), int(n_points))


def summarize_curve(f1_grid, c_grid):
    """Describe a sweep without presenting a flat or edge value as an optimum."""
    f1_grid = np.asarray(f1_grid, dtype=float)
    c_grid = np.asarray(c_grid, dtype=float)
    if (f1_grid.ndim != 1 or c_grid.ndim != 1
            or f1_grid.size != c_grid.size or f1_grid.size < 3
            or not np.all(np.isfinite(f1_grid))
            or not np.all(np.isfinite(c_grid))):
        raise ValueError("finite one-dimensional grids of equal length are required")
    span = float(np.ptp(f1_grid))
    degenerate = bool(span <= F1_SPAN_TOL)
    is_max = np.isclose(
        f1_grid, np.max(f1_grid), rtol=0.0, atol=F1_SPAN_TOL)
    maximizers = np.flatnonzero(is_max)
    best_index = int(maximizers[0])
    reaches_right = bool(maximizers[-1] == f1_grid.size - 1)
    suffix_start = f1_grid.size - 1
    if reaches_right:
        while suffix_start > 0 and is_max[suffix_start - 1]:
            suffix_start -= 1
    right_plateau_decades = (
        float(np.log10(c_grid[-1] / c_grid[suffix_start]))
        if reaches_right else 0.0
    )
    return {
        "f1_span": span,
        "degenerate": degenerate,
        "best_index": best_index,
        "c_bias_star": float(c_grid[best_index]),
        "f1_mf": float(f1_grid[best_index]),
        "at_left_edge": bool(maximizers[0] == 0),
        "at_right_edge": reaches_right,
        "n_maximizers": int(maximizers.size),
        "right_plateau_start_index": int(suffix_start),
        "right_plateau_decades": right_plateau_decades,
        "bias_only_limit": bool(
            not degenerate and reaches_right
            and right_plateau_decades >= 1.0),
        "c_bias_identifiable": bool(not degenerate and maximizers.size == 1),
    }


def require_interior_optima(rows):
    """Reject informative curves whose optimum is outside the explored grid."""
    unresolved = [
        f"{r['scenario']}:Re{r['Re']}"
        for r in rows
        if not r["degenerate"]
        and (r["at_left_edge"]
             or (r["at_right_edge"] and not r.get("bias_only_limit", False)))
    ]
    if unresolved:
        raise RuntimeError(
            "c_bias sweep unresolved at a grid edge for "
            + ", ".join(unresolved)
            + "; widen --c-min/--c-max before producing an artifact")


# -------------------------------------------------------------------
# 1D threshold sweep (classical indicator F1 vs L2-hard mask)
# -------------------------------------------------------------------

def best_threshold(scores_pool, gt_pool, grid=None):
    """Return the exact best ``score > threshold`` rule within the bounds."""
    scores_pool = np.asarray(scores_pool, dtype=float).ravel()
    gt_pool = np.asarray(gt_pool, dtype=bool).ravel()
    if scores_pool.size == 0 or scores_pool.size != gt_pool.size:
        raise ValueError("scores and labels must be non-empty and aligned")
    if not np.all(np.isfinite(scores_pool)):
        raise ValueError("scores must be finite")
    lo, hi = map(float, THETA_BOUNDS[1])
    if np.any((scores_pool < lo) | (scores_pool > hi)):
        raise ValueError(f"scores must lie in [{lo:g}, {hi:g}]")
    exact_grid = grid is None
    if exact_grid:
        # Predictions change only when the threshold crosses an observed
        # score, so this finite set is exhaustive for the strict ``>`` rule.
        grid = np.unique(np.concatenate(([lo, hi], scores_pool)))
    else:
        grid = np.asarray(grid, dtype=float).ravel()
        if (grid.size == 0 or not np.all(np.isfinite(grid))
                or np.any((grid < lo) | (grid > hi))):
            raise ValueError(
                f"threshold grid must be finite and in [{lo:g}, {hi:g}]")
    best_index, best_f1 = 0, -1.0
    for index, thr in enumerate(grid):
        pred = scores_pool > thr
        tp = int(((pred == 1) & (gt_pool == 1)).sum())
        fp = int(((pred == 1) & (gt_pool == 0)).sum())
        fn = int(((pred == 0) & (gt_pool == 1)).sum())
        denom = (2 * tp + fp + fn)
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_index = f1, index
    best_thr = float(grid[best_index])
    if exact_grid and best_index + 1 < len(grid):
        # Any value up to the next observed score produces the same strict
        # decision. Its midpoint avoids an arbitrary breakpoint or bound.
        best_thr = 0.5 * (best_thr + float(grid[best_index + 1]))
    return best_thr, best_f1


# -------------------------------------------------------------------
# Zero-temperature MF iteration on the Ising + 4-body graph
# -------------------------------------------------------------------

def mean_field_decode(h_bias, edges, plaqs, n_q,
                      max_iter=200, init_spins=None, rng=None):
    """Asynchronous zero-T Glauber (per-site greedy) until convergence."""
    edge_idx, edge_coef = edges
    plaq_idx, plaq_coef = plaqs
    edges_by_q, plaqs_by_q = _build_incidence(n_q, edges, plaqs)

    if init_spins is None:
        spins = np.ones(n_q, dtype=np.int8)     # start at "don't refine"
    else:
        spins = init_spins.astype(np.int8).copy()

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
                thr_amr, c_bias_grid, gt_mask, seed):
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
        # Common random numbers make every c_bias value use the same scan
        # sequence; curve differences therefore come from the Hamiltonian.
        spins, _ = mean_field_decode(
            h_bias, edges, plaqs, n_q, max_iter=100,
            rng=np.random.default_rng(seed))
        dec_h, dec_v = spins_to_decisions(spins, dim)
        refine = dec_h | dec_v
        f1s.append(_metrics(refine, gt_mask)["f1"])
    return np.array(f1s)


# -------------------------------------------------------------------
# Per-scenario analysis
# -------------------------------------------------------------------

def analyse_snapshots(dns_path, patches_path, dim, *,
                       c_grid, max_snaps, seed, train_frac, val_frac):
    dns = np.load(dns_path)
    patches = np.load(patches_path)
    vx = dns["vx"]; vy = dns["vy"]
    Bx = dns["Bx"]; By = dns["By"]
    N  = vx.shape[1]
    Re = int(dns.get("meta_Re", 800))
    scenario = str(dns.get("meta_scenario", "unknown"))

    l2_all = patches["l2_errors"]
    l2_thr = float(patches["l2_threshold"])

    train_indices, _, _ = chronological_split_indices(
        len(vx), train_frac, val_frac)
    snap_indices = evenly_subsample(train_indices, max_snaps).tolist()

    # ---- pool scores & GT for threshold sweep ----
    scores_pool = []
    gt_pool     = []
    snap_cache = []
    for si in snap_indices:
        _, score_vqa, _ = build_patch_hamiltonian(
            vx[si].astype(np.float64), vy[si].astype(np.float64),
            Bx[si].astype(np.float64), By[si].astype(np.float64),
            N, dim, Re, threshold_amr=V2_THRESHOLD, use_v2=True,
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
    for snapshot_rank, sc in enumerate(snap_cache):
        f1_grid += mf_f1_curve(
            sc["vx"], sc["vy"], sc["Bx"], sc["By"],
            N, dim, Re,
            thr_star, c_grid, sc["gt_mask"], seed + snapshot_rank,
        )
    f1_grid /= len(snap_cache)
    curve = summarize_curve(f1_grid, c_grid)
    return dict(
        scenario=scenario, Re=Re, dim=dim, N=N,
        thr_star=thr_star, c_bias_star=curve["c_bias_star"],
        f1_mf=curve["f1_mf"], classical_f1=f1_class,
        f1_grid=f1_grid, c_grid=np.array(c_grid),
        snap_indices=np.array(snap_indices),
        f1_span=curve["f1_span"], degenerate=curve["degenerate"],
        at_left_edge=curve["at_left_edge"],
        at_right_edge=curve["at_right_edge"],
        n_maximizers=curve["n_maximizers"],
        right_plateau_start_index=curve["right_plateau_start_index"],
        right_plateau_decades=curve["right_plateau_decades"],
        bias_only_limit=curve["bias_only_limit"],
        c_bias_identifiable=curve["c_bias_identifiable"],
    )


def mean_over_informative(rows, key):
    """Average informative rows, or return NaN when none is available."""
    vals = [r[key] for r in rows if not r["degenerate"]]
    return float(np.mean(vals)) if vals else float("nan")


# -------------------------------------------------------------------
# Search-domain diagnostics
# -------------------------------------------------------------------

def _edge_flags(row):
    """Render unresolved search-domain diagnostics for one row."""
    agg = "f1_span" not in row
    suffix = "-COMPONENT" if agg else ""
    f = []
    if row.get("at_left_edge"):
        f.append(f"LEFT-EDGE{suffix}")
    if row.get("at_right_edge"):
        label = "BIAS-ONLY-LIMIT" if row.get("bias_only_limit") else "RIGHT-EDGE"
        f.append(f"{label}{suffix}")
    return ("   << " + " | ".join(f)) if f else ""


def _edge_flags_agg(rows):
    """Aggregate edge diagnostics over informative component rows."""
    live = [r for r in rows if not r["degenerate"]]
    right = [r for r in live if r["at_right_edge"]]
    return dict(
        at_left_edge=bool(any(r["at_left_edge"] for r in live)),
        at_right_edge=bool(right),
        bias_only_limit=bool(
            right and all(r.get("bias_only_limit", False) for r in right)),
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
    p.add_argument("--n-cgrid", type=int, default=31,
                   help="number of logarithmic c_bias points")
    p.add_argument("--c-min", type=float, default=DEFAULT_C_MIN)
    p.add_argument("--c-max", type=float, default=DEFAULT_C_MAX)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.6,
                   help="chronological training fraction per trajectory")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="chronological validation fraction used by phase 10")
    args = p.parse_args()

    try:
        c_grid = c_bias_grid(args.c_min, args.c_max, args.n_cgrid)
        chronological_split_indices(3, args.train_frac, args.val_frac)
        if args.max_snaps < 1:
            raise ValueError("max_snaps must be positive")
    except ValueError as exc:
        p.error(str(exc))
    run_provenance = provenance.start()
    print("=" * 88)
    print("  Phase 10a: MF-analytical derivation of (c_bias*, thr*)")
    print(f"  c_bias grid: [{args.c_min:g}, {args.c_max:g}]   "
          f"({args.n_cgrid} points)")
    print(f"  snaps/config: {args.max_snaps}")
    print(f"  chronological training prefix: {args.train_frac:.1%}")
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
                c_grid=c_grid, max_snaps=args.max_snaps, seed=args.seed,
                train_frac=args.train_frac, val_frac=args.val_frac)
            dt = time.time() - t0
            per_cfg.append(res)
            print(f"  [{sc} Re={re}] thr*={res['thr_star']:.3f} "
                  f"c_bias*={res['c_bias_star']:.2f}  "
                  f"F1_MF={res['f1_mf']:.3f}  "
                  f"classical={res['classical_f1']:.3f}   [{dt:.1f}s]"
                  + ("   UNINFORMATIVE: flat F1(c)"
                     if res["degenerate"] else "")
                  + (f"   BIAS-ONLY plateau >= "
                     f"{res['right_plateau_decades']:.1f} decades"
                     if res["bias_only_limit"] else "")
                  + _edge_flags(res))

    if not per_cfg:
        raise RuntimeError(
            "empty sweep: no configuration has both DNS and patch inputs")

    n_degen = sum(r["degenerate"] for r in per_cfg)
    if n_degen == len(per_cfg):
        raise RuntimeError(
            f"all {len(per_cfg)} configurations have a flat F1(c_bias) "
            "curve; widen the c_bias grid")

    require_interior_optima(per_cfg)

    # ---- per-scenario aggregation ----
    by_scene = {}
    for r in per_cfg:
        by_scene.setdefault(r["scenario"], []).append(r)

    scenario_rows = []
    print("\n  per-scenario (mean over Re, degenerate rows excluded):")
    for sc, rows in by_scene.items():
        n_d = sum(r["degenerate"] for r in rows)
        row = dict(
            tag=f"scenario:{sc}",
            thr_star=mean_over_informative(rows, "thr_star"),
            c_bias_star=mean_over_informative(rows, "c_bias_star"),
            f1_mf=mean_over_informative(rows, "f1_mf"),
            classical_f1=mean_over_informative(rows, "classical_f1"),
            degenerate=(n_d == len(rows)),
            **_edge_flags_agg(rows),
        )
        scenario_rows.append(row)
        print(f"    {sc:<18} thr*={row['thr_star']:.3f}  "
              f"c_bias*={row['c_bias_star']:.2f}  "
              f"F1_MF={row['f1_mf']:.3f}  "
              f"classical={row['classical_f1']:.3f}"
              + (f"   ({n_d}/{len(rows)} degeneres exclus)" if n_d else "")
              + ("   UNINFORMATIVE: no non-flat sweep"
                 if row["degenerate"] else "")
              + _edge_flags(row))

    # ---- joint (mean over everything) ----
    joint_row = dict(
        tag="joint",
        thr_star=mean_over_informative(per_cfg, "thr_star"),
        c_bias_star=mean_over_informative(per_cfg, "c_bias_star"),
        f1_mf=mean_over_informative(per_cfg, "f1_mf"),
        classical_f1=mean_over_informative(per_cfg, "classical_f1"),
        degenerate=False,
        **_edge_flags_agg(per_cfg),
    )
    print(f"\n  joint  thr*={joint_row['thr_star']:.3f}  "
          f"c_bias*={joint_row['c_bias_star']:.2f}  "
          f"F1_MF={joint_row['f1_mf']:.3f}  "
          f"classical={joint_row['classical_f1']:.3f}"
          + (f"   ({n_degen}/{len(per_cfg)} degeneres exclus)"
             if n_degen else "")
          + _edge_flags(joint_row))

    # ---- per-config (for reference) ----
    cfg_rows = []
    for r in per_cfg:
        cfg_rows.append(dict(
            tag=f"cfg:{r['scenario']}_Re{r['Re']}",
            thr_star=r["thr_star"], c_bias_star=r["c_bias_star"],
            f1_mf=r["f1_mf"], classical_f1=r["classical_f1"],
            degenerate=r["degenerate"], f1_span=r["f1_span"],
            at_left_edge=r["at_left_edge"],
            at_right_edge=r["at_right_edge"],
            n_maximizers=r["n_maximizers"],
            right_plateau_start_index=r["right_plateau_start_index"],
            right_plateau_decades=r["right_plateau_decades"],
            bias_only_limit=r["bias_only_limit"],
            c_bias_identifiable=r["c_bias_identifiable"],
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
        degenerate=np.array([bool(r["degenerate"]) for r in all_rows]),
        f1_span=np.array([float(r.get("f1_span", np.nan)) for r in all_rows]),
        at_left_edge=np.array([bool(r["at_left_edge"]) for r in all_rows]),
        at_right_edge=np.array([bool(r["at_right_edge"]) for r in all_rows]),
        n_maximizers=np.array(
            [float(r.get("n_maximizers", np.nan)) for r in all_rows]),
        right_plateau_start_index=np.array(
            [float(r.get("right_plateau_start_index", np.nan))
             for r in all_rows]),
        right_plateau_decades=np.array(
            [float(r.get("right_plateau_decades", np.nan))
             for r in all_rows]),
        bias_only_limit=np.array(
            [bool(r.get("bias_only_limit", False)) for r in all_rows]),
        c_bias_identifiable=np.array(
            [bool(r.get("c_bias_identifiable", False)) for r in all_rows]),
        theta_bounds=np.asarray(THETA_BOUNDS, dtype=float),
        c_grid=c_grid,
        split_strategy="chronological_per_configuration",
        train_fraction=args.train_frac,
        validation_fraction=args.val_frac,
        cli_args=json.dumps(vars(args), sort_keys=True),
        seed=args.seed,
        **provenance.finish(run_provenance),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 10a complete.")


if __name__ == "__main__":
    main()
