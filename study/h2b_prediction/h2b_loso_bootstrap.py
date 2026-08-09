#!/usr/bin/env python3
"""
Phase 11B-2 - LOSO with bootstrap confidence intervals.

Phase 11B reported the LOSO mean F1_site = 0.19+/-0.15 across 4
scenarios. That variance bar comes from an n=4 inter-scenario std --
it has no statistical power. A reviewer will ask "what's the
snapshot-level CI on each fold?"

This phase answers by:

  1. Loading ALL snapshots (max-snaps bumped from 30 -> 80 per config
     by default) so each held-out scenario has O(10^4) cells / O(10^3)
     snapshots of evaluation coverage.
  2. For each LOSO fold, computing F1_site and F1_class.
  3. Bootstrapping the held-out scenario's snapshots B=500 times
     (resampling with replacement at the SNAPSHOT level, not the cell
     level -- cells within a snapshot are spatially correlated, so
     cell-level bootstrap underestimates variance) and reporting a
     95% percentile CI on each F1.
  4. Computing a paired bootstrap p-value for H0: F1_site >= F1_class,
     H1: F1_site  < F1_class (the hypothesis we want to REJECT in
     order to save Q-HAS).

Output columns:
  held                 F1_class CI_class F1_site CI_site  delta CI_delta p(H0)

Input:  results/dns_{sc}_Re{re}_N{N}.npz
        results/patches_{sc}_Re{re}_N{N}_dim{D}.npz
Output: results/loso_bootstrap_N{N}_dim{D}.npz

Usage:
  python study/phase11b2_bootstrap.py --dim 4 --max-snaps 80 --n-boot 500
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
from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N

from h2b_ceiling_random_split import (
    FEATURE_NAMES, N_FEATS,
    extract_features_2d, best_threshold_f1, make_model, fit_eval,
)

from sklearn.metrics import f1_score


def gather_by_scenario(configs, dim, max_snaps):
    """Keep features per snapshot so we can bootstrap snapshots later."""
    per_sc = {}
    for sc, re, dns_path, patches_path in configs:
        dns = np.load(dns_path)
        patches = np.load(patches_path)
        vx = dns["vx"].astype(np.float64); vy = dns["vy"].astype(np.float64)
        Bx = dns["Bx"].astype(np.float64); By = dns["By"].astype(np.float64)
        N = vx.shape[1]
        l2_all = patches["l2_errors"]; thr = float(patches["l2_threshold"])

        n_snaps = len(vx)
        step = max(1, n_snaps // max_snaps)
        idx = list(range(0, n_snaps, step))[:max_snaps]

        for si in idx:
            feats_2d, score = extract_features_2d(
                vx[si], vy[si], Bx[si], By[si], N, dim, re)
            per_sc.setdefault(sc, dict(Xs=[], Ys=[], Ss=[]))
            per_sc[sc]["Xs"].append(feats_2d.reshape(-1, N_FEATS))
            per_sc[sc]["Ys"].append((l2_all[si] >= thr).ravel().astype(int))
            per_sc[sc]["Ss"].append(score.ravel())
    return per_sc


def snapshot_f1(Y_list, P_list, thr):
    """F1 aggregated over snapshots with a given threshold on probabilities."""
    Y = np.concatenate(Y_list)
    P = np.concatenate(P_list)
    return f1_score(Y, (P > thr).astype(int), zero_division=0)


def bootstrap_ci(Y_list, P_list, thr, n_boot, rng, alpha=0.05):
    """Percentile CI on F1 via snapshot-level bootstrap."""
    n = len(Y_list)
    f1s = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = np.concatenate([Y_list[i] for i in idx])
        p = np.concatenate([P_list[i] for i in idx])
        f1s[b] = f1_score(y, (p > thr).astype(int), zero_division=0)
    lo = float(np.quantile(f1s, alpha / 2))
    hi = float(np.quantile(f1s, 1 - alpha / 2))
    return lo, hi, f1s


def paired_bootstrap_delta(
    Y_list, P_site_list, thr_site,
    S_list, thr_cls, n_boot, rng,
):
    """Paired bootstrap on delta = F1_site - F1_class."""
    n = len(Y_list)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = np.concatenate([Y_list[i] for i in idx])
        ps = np.concatenate([P_site_list[i] for i in idx])
        sc = np.concatenate([S_list[i] for i in idx])
        f_s = f1_score(y, (ps > thr_site).astype(int), zero_division=0)
        f_c = f1_score(y, (sc > thr_cls).astype(int), zero_division=0)
        deltas[b] = f_s - f_c
    lo = float(np.quantile(deltas, 0.025))
    hi = float(np.quantile(deltas, 0.975))
    # p-value for H1: delta < 0 (site fails to beat classical)
    # one-sided: fraction of bootstrap replicates where delta >= 0
    # (low = strong evidence site < classical)
    p_site_gte_class = float((deltas >= 0).mean())
    return lo, hi, deltas, p_site_gte_class


def main():
    p = argparse.ArgumentParser(
        description="Phase 11B-2: LOSO with bootstrap CIs")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=80)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  Phase 11B-2: LOSO with snapshot-level bootstrap CIs")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}  "
          f"n_boot={args.n_boot}")
    print("=" * 88)
    print()

    # -- discover configs --
    configs = []
    for sc in args.scenario:
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(
                RESULTS_DIR, f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if os.path.exists(dp) and os.path.exists(pp):
                configs.append((sc, re, dp, pp))
    if len(set(c[0] for c in configs)) < 2:
        print("Need >=2 scenarios for LOSO."); return

    t0 = time.time()
    per_sc = gather_by_scenario(configs, args.dim, args.max_snaps)
    print(f"  built dataset in {time.time()-t0:.1f}s")
    for sc in per_sc:
        n_snaps = len(per_sc[sc]["Xs"])
        n_cells = sum(len(y) for y in per_sc[sc]["Ys"])
        pr = np.concatenate(per_sc[sc]["Ys"]).mean()
        print(f"    {sc:<18} snaps={n_snaps:>4}  cells={n_cells:>6}  "
              f"pos_rate={pr:.3f}")
    print()

    rng = np.random.default_rng(args.seed)

    header = (f"  {'held':<18} {'F1_class':>9} {'CI95_class':>18} "
              f"{'F1_site':>9} {'CI95_site':>18} "
              f"{'delta':>7} {'CI95_delta':>18} {'p(H0)':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for held in per_sc:
        # -- train pool = all scenarios != held --
        tr_sc = [s for s in per_sc if s != held]
        Xtr = np.concatenate([x for s in tr_sc for x in per_sc[s]["Xs"]])
        Ytr = np.concatenate([y for s in tr_sc for y in per_sc[s]["Ys"]])
        Str = np.concatenate([z for s in tr_sc for z in per_sc[s]["Ss"]])

        # -- validation pool (kept as per-snapshot lists for bootstrap) --
        Xv_list = per_sc[held]["Xs"]
        Yv_list = per_sc[held]["Ys"]
        Sv_list = per_sc[held]["Ss"]
        Xv_flat = np.concatenate(Xv_list)

        # classical thr*
        thr_cls, _ = best_threshold_f1(Str, Ytr)

        # site GBT: train + get per-snap probabilities
        if len(np.unique(Ytr)) < 2:
            print(f"  {held:<18} degenerate (train all-one-class)")
            continue
        r = fit_eval(make_model("gbt", args.seed), Xtr, Ytr, Xv_flat,
                     np.concatenate(Yv_list))
        # recompute probs per snapshot using model -> threshold best
        # fit_eval already searches thr_grid; use its thr
        thr_site = r["thr"]
        # get per-snapshot predictions:
        # fit the model once; we need the same fitted instance to predict
        # per-snap. We'll re-fit once more via fit_eval's API (fit_eval
        # returns p-flat but not the model); refit here quickly:
        model = make_model("gbt", args.seed)
        model.fit(Xtr, Ytr)
        P_site_list = [model.predict_proba(x)[:, 1] for x in Xv_list]

        # point estimates
        f1_class = snapshot_f1(Yv_list, Sv_list, thr_cls)
        f1_site  = snapshot_f1(Yv_list, P_site_list, thr_site)

        # bootstrap CIs
        c_lo, c_hi, _ = bootstrap_ci(Yv_list, Sv_list, thr_cls,
                                      args.n_boot, rng)
        s_lo, s_hi, _ = bootstrap_ci(Yv_list, P_site_list, thr_site,
                                      args.n_boot, rng)
        d_lo, d_hi, _, p_H0 = paired_bootstrap_delta(
            Yv_list, P_site_list, thr_site, Sv_list, thr_cls,
            args.n_boot, rng)

        delta = f1_site - f1_class

        print(f"  {held:<18} "
              f"{f1_class:>9.3f} [{c_lo:>5.3f}, {c_hi:>5.3f}]   "
              f"{f1_site:>9.3f} [{s_lo:>5.3f}, {s_hi:>5.3f}]   "
              f"{delta:>+7.3f} [{d_lo:>+6.3f},{d_hi:>+6.3f}]  "
              f"{p_H0:>7.3f}")

        rows.append(dict(
            held=held,
            f1_class=f1_class, f1_class_ci=(c_lo, c_hi),
            f1_site=f1_site,   f1_site_ci=(s_lo, s_hi),
            delta=delta,       delta_ci=(d_lo, d_hi),
            p_site_beats_class=p_H0,
            thr_class=thr_cls, thr_site=thr_site,
            n_val_snaps=len(Yv_list),
            n_val_cells=len(Xv_flat),
        ))

    # -- aggregate --
    if rows:
        mean_delta = float(np.mean([r["delta"] for r in rows]))
        ps = [r["p_site_beats_class"] for r in rows]
        print("  " + "-" * (len(header) - 2))
        print(f"  mean delta across folds = {mean_delta:+.3f}")
        print(f"  folds where bootstrap p(F1_site >= F1_class) < 0.05: "
              f"{sum(p < 0.05 for p in ps)} / {len(ps)}  "
              f"(i.e. site significantly WORSE than classical)")

        print()
        print("  INTERPRETATION:")
        if mean_delta < 0:
            k = sum(p < 0.05 for p in ps)
            total = len(ps)
            if k == total:
                print("  * In every fold, the mean-field ceiling is "
                      "significantly BELOW the classical indicator "
                      "(bootstrap p < 0.05).")
                print("  * This is a definitive rejection of the "
                      "'local-H is scenario-universal' hypothesis at "
                      "the 95% confidence level.")
            else:
                print(f"  * The mean delta is {mean_delta:+.3f} and "
                      f"{k}/{total} folds reject site >= classical at "
                      f"p < 0.05. The falsification is robust in "
                      f"aggregate.")
        else:
            print(f"  * Mean delta = {mean_delta:+.3f} >= 0. "
                  f"The ceiling does NOT collapse in aggregate -- "
                  f"investigate what was different in phase 11b.")

    # -- save --
    out = os.path.join(RESULTS_DIR,
                       f"loso_bootstrap_N{args.N}_dim{args.dim}.npz")
    if rows:
        np.savez_compressed(
            out,
            held=np.array([r["held"] for r in rows]),
            f1_class=np.array([r["f1_class"] for r in rows]),
            f1_class_ci_lo=np.array([r["f1_class_ci"][0] for r in rows]),
            f1_class_ci_hi=np.array([r["f1_class_ci"][1] for r in rows]),
            f1_site =np.array([r["f1_site"] for r in rows]),
            f1_site_ci_lo=np.array([r["f1_site_ci"][0] for r in rows]),
            f1_site_ci_hi=np.array([r["f1_site_ci"][1] for r in rows]),
            delta   =np.array([r["delta"] for r in rows]),
            delta_ci_lo=np.array([r["delta_ci"][0] for r in rows]),
            delta_ci_hi=np.array([r["delta_ci"][1] for r in rows]),
            p_site_beats_class=np.array(
                [r["p_site_beats_class"] for r in rows]),
            n_val_snaps=np.array([r["n_val_snaps"] for r in rows]),
            n_val_cells=np.array([r["n_val_cells"] for r in rows]),
        )
        print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 11B-2 complete.")


if __name__ == "__main__":
    main()
