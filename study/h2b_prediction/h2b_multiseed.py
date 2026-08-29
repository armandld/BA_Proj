#!/usr/bin/env python3
"""
Phase 11F - Multi-seed wrapper for the headline phase 11 / 11B numbers.

V1's Fig. 6 reports 10-seed permutation tests. V2's headline phase 11
(random split) and 11B (LOSO) numbers were originally seed=0 only.
This phase re-runs both at seeds 0..N-1 and reports mean +/- std for:

  - random-split mean-field GBT F1   (was: 0.989 single-seed)
  - random-split stencil GBT F1       (was: 0.991 single-seed)
  - LOSO mean-field GBT F1 per fold   (was: 0.191 +/- 0.152, fold-std only)
  - LOSO stencil GBT F1 per fold      (was: 0.215 +/- 0.142, fold-std only)
  - learned-linear-H F1 (random + LOSO)

The fold-std reports across-scenario variability; this phase's
seed-std reports across-seed variability of each fold. Together
they decompose the noise budget the V1 protocol expects.

Output: results/multiseed_N{N}_dim{D}.npz

Usage:
  python study/phase11f_multiseed.py --dim 4 --n-seeds 10 --max-snaps 30
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
    FEATURE_NAMES, N_FEATS, extract_features_2d, stencil_features,
    make_model, fit_eval, best_threshold_f1,
)
from sklearn.metrics import f1_score


def discover_configs(scenarios, res, N, dim):
    cfgs = []
    for sc in scenarios:
        for re in res:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{N}.npz")
            pp = os.path.join(RESULTS_DIR, f"patches_{sc}_Re{re}_N{N}_dim{dim}.npz")
            if os.path.exists(dp) and os.path.exists(pp):
                cfgs.append((sc, re, dp, pp))
    return cfgs


def gather_per_snapshot(configs, dim, max_snaps):
    """Return parallel lists of per-snapshot arrays + scenario tags."""
    Xs, Xst, Ys, Ss, tags = [], [], [], [], []
    for sc, re, dp, pp in configs:
        dns = np.load(dp); patches = np.load(pp)
        vx = dns["vx"].astype(np.float64); vy = dns["vy"].astype(np.float64)
        Bx = dns["Bx"].astype(np.float64); By = dns["By"].astype(np.float64)
        N = vx.shape[1]
        l2 = patches["l2_errors"]; thr = float(patches["l2_threshold"])
        n_snap = len(vx); step = max(1, n_snap // max_snaps)
        idx = list(range(0, n_snap, step))[:max_snaps]
        for si in idx:
            f2d, sc_v = extract_features_2d(vx[si], vy[si], Bx[si], By[si], N, dim, re)
            Xs.append(f2d.reshape(-1, N_FEATS))
            Xst.append(stencil_features(f2d))
            Ys.append((l2[si] >= thr).ravel().astype(int))
            Ss.append(sc_v.ravel())
            tags.append(sc)
    return Xs, Xst, Ys, Ss, tags


def random_split_seed(Xs, Xst, Ys, Ss, val_frac, seed):
    """One random-by-snapshot split, fit GBT site + stencil, return F1."""
    rng = np.random.default_rng(seed)
    n = len(Xs); perm = rng.permutation(n)
    n_va = max(1, int(val_frac * n))
    va = perm[:n_va]; tr = perm[n_va:]

    Xtr_s  = np.concatenate([Xs[i]  for i in tr])
    Xtr_st = np.concatenate([Xst[i] for i in tr])
    Ytr    = np.concatenate([Ys[i]  for i in tr])
    Str    = np.concatenate([Ss[i]  for i in tr])
    Xv_s   = np.concatenate([Xs[i]  for i in va])
    Xv_st  = np.concatenate([Xst[i] for i in va])
    Yv     = np.concatenate([Ys[i]  for i in va])
    Sv     = np.concatenate([Ss[i]  for i in va])

    if len(np.unique(Ytr)) < 2 or len(np.unique(Yv)) < 2:
        return dict(f1_site=float("nan"), f1_sten=float("nan"),
                    f1_class=float("nan"))
    r_site = fit_eval(make_model("gbt", seed), Xtr_s,  Ytr, Xv_s,  Yv)
    r_sten = fit_eval(make_model("gbt", seed), Xtr_st, Ytr, Xv_st, Yv)
    thr_c, _ = best_threshold_f1(Str, Ytr)
    f1_c = f1_score(Yv, (Sv > thr_c).astype(int), zero_division=0)
    return dict(f1_site=r_site["f1"], f1_sten=r_sten["f1"], f1_class=f1_c)


def loso_seed(Xs, Xst, Ys, Ss, tags, seed):
    """LOSO with seed used in model RNG. Returns per-fold dicts."""
    scs = sorted(set(tags))
    rows = []
    by_sc = {sc: [i for i, t in enumerate(tags) if t == sc] for sc in scs}
    for held in scs:
        tr_idx = [i for sc, ix in by_sc.items() if sc != held for i in ix]
        va_idx = by_sc[held]
        Xtr_s  = np.concatenate([Xs[i]  for i in tr_idx])
        Xtr_st = np.concatenate([Xst[i] for i in tr_idx])
        Ytr    = np.concatenate([Ys[i]  for i in tr_idx])
        Str    = np.concatenate([Ss[i]  for i in tr_idx])
        Xv_s   = np.concatenate([Xs[i]  for i in va_idx])
        Xv_st  = np.concatenate([Xst[i] for i in va_idx])
        Yv     = np.concatenate([Ys[i]  for i in va_idx])
        Sv     = np.concatenate([Ss[i]  for i in va_idx])
        if len(np.unique(Ytr)) < 2 or len(np.unique(Yv)) < 2:
            rows.append(dict(held=held, f1_site=float("nan"),
                             f1_sten=float("nan"), f1_class=float("nan")))
            continue
        thr_c, _ = best_threshold_f1(Str, Ytr)
        f1_c = f1_score(Yv, (Sv > thr_c).astype(int), zero_division=0)
        r_site = fit_eval(make_model("gbt", seed), Xtr_s,  Ytr, Xv_s,  Yv)
        r_sten = fit_eval(make_model("gbt", seed), Xtr_st, Ytr, Xv_st, Yv)
        rows.append(dict(held=held, f1_site=r_site["f1"],
                         f1_sten=r_sten["f1"], f1_class=f1_c))
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Phase 11F: multi-seed wrapper for phase 11 / 11B headlines")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.30)
    args = p.parse_args()

    print("=" * 88)
    print(f"  Phase 11F: multi-seed (n={args.n_seeds}) re-run of phase 11 + 11B")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}")
    print("=" * 88)
    print()

    cfgs = discover_configs(args.scenario, args.re, args.N, args.dim)
    if not cfgs:
        # Cette garde faisait auparavant `print(...); return` : code 0,
        # aucun artefact ecrit, indiscernable d'une campagne reussie. Le
        # detecteur AST qui traque ce motif ailleurs dans study/ ne
        # reconnait que la forme `if not <accumulateur nomme>:` ; celle-ci
        # lui echappait.
        raise RuntimeError(
            "balayage vide : aucune configuration (scenario, Re) n'a d'artefact "
            "d'entree pour les arguments donnes. Le script sortait ici avec le "
            "code 0 et sans artefact (D-75).")
    Xs, Xst, Ys, Ss, tags = gather_per_snapshot(cfgs, args.dim, args.max_snaps)
    print(f"  built dataset: {len(Xs)} snapshots across "
          f"{len(set(tags))} scenarios\n")

    # --- random split, multiple seeds ---
    rs_site, rs_sten, rs_class = [], [], []
    for seed in range(args.n_seeds):
        r = random_split_seed(Xs, Xst, Ys, Ss, args.val_frac, seed)
        rs_site.append(r["f1_site"]); rs_sten.append(r["f1_sten"])
        rs_class.append(r["f1_class"])
    rs_site = np.array(rs_site); rs_sten = np.array(rs_sten)
    rs_class = np.array(rs_class)

    print("  RANDOM SPLIT (val_frac={:.0%}, by snapshot)".format(args.val_frac))
    print(f"    F1 mean-field GBT : {np.nanmean(rs_site):.3f} "
          f"+/- {np.nanstd(rs_site):.3f}   (n={args.n_seeds} seeds)")
    print(f"    F1 stencil   GBT  : {np.nanmean(rs_sten):.3f} "
          f"+/- {np.nanstd(rs_sten):.3f}")
    print(f"    F1 classical thr  : {np.nanmean(rs_class):.3f} "
          f"+/- {np.nanstd(rs_class):.3f}")
    print()

    # --- LOSO, multiple seeds ---
    scs = sorted(set(tags))
    loso_site = {sc: [] for sc in scs}
    loso_sten = {sc: [] for sc in scs}
    loso_class = {sc: [] for sc in scs}
    for seed in range(args.n_seeds):
        rows = loso_seed(Xs, Xst, Ys, Ss, tags, seed)
        for r in rows:
            loso_site[r["held"]].append(r["f1_site"])
            loso_sten[r["held"]].append(r["f1_sten"])
            loso_class[r["held"]].append(r["f1_class"])

    print(f"  LOSO ({len(scs)} folds)")
    print(f"  {'held-out':<18} {'F1_class +/- s':>16} "
          f"{'F1_site +/- s':>16} {'F1_sten +/- s':>16}")
    for sc in scs:
        a = np.array(loso_site[sc]); b = np.array(loso_sten[sc])
        c = np.array(loso_class[sc])
        print(f"  {sc:<18} "
              f"{np.nanmean(c):>7.3f} +/- {np.nanstd(c):.3f}   "
              f"{np.nanmean(a):>7.3f} +/- {np.nanstd(a):.3f}   "
              f"{np.nanmean(b):>7.3f} +/- {np.nanstd(b):.3f}")
    site_means = np.array([np.nanmean(loso_site[sc]) for sc in scs])
    site_stds  = np.array([np.nanstd (loso_site[sc]) for sc in scs])
    cls_means  = np.array([np.nanmean(loso_class[sc]) for sc in scs])
    print()
    print(f"  LOSO mean-field across folds (n={len(scs)}): "
          f"{np.nanmean(site_means):.3f} +/- {np.nanstd(site_means):.3f}")
    print(f"  LOSO classical  across folds (n={len(scs)}): "
          f"{np.nanmean(cls_means):.3f} +/- {np.nanstd(cls_means):.3f}")
    print(f"  mean across-seed std per fold: "
          f"{np.nanmean(site_stds):.3f}   "
          f"(decomposes solver noise vs scenario noise)")

    out = os.path.join(RESULTS_DIR,
                       f"multiseed_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        n_seeds=args.n_seeds,
        rs_site=rs_site, rs_sten=rs_sten, rs_class=rs_class,
        loso_scenarios=np.array(scs),
        loso_site=np.array([loso_site[s]  for s in scs]),
        loso_sten=np.array([loso_sten[s]  for s in scs]),
        loso_class=np.array([loso_class[s] for s in scs]),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 11F complete.")


if __name__ == "__main__":
    main()
