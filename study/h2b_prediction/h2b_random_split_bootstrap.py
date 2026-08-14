#!/usr/bin/env python3
"""
Phase 11H - Bootstrap CI on the random-split ceiling (0.989).

Phase 11 reports F1_site = 0.989 as the random-split mean-field GBT
ceiling at a single seed. That is a point estimate with no CI.
Phase 11F (multi-seed) gives a cross-seed std but not a
snapshot-level CI on a single seed's F1.

This phase fixes that: for a fixed random-by-snapshot split, compute
snapshot-level percentile bootstrap CIs for F1_class, F1_site,
F1_stencil and the paired delta (site - class), so that the headline
random-split number is reported with the same statistical rigour as
the LOSO numbers in phase 11B-2.

Method:
  1. Gather all snapshots across scenarios + Re values.
  2. One random 70/30 snapshot split (seed=0 by default).
  3. Fit GBT_site, GBT_stencil on training rows.
  4. Keep validation per-snapshot so we can resample snapshots.
  5. Bootstrap B=500 snapshot resamples on the validation set.
  6. Report F1_class, F1_site, F1_stencil each with percentile 95% CI
     and paired-bootstrap delta (site - class).

Output: results/random_split_bootstrap_N{N}_dim{D}.npz

Usage:
  python study/phase11h_random_split_bootstrap.py --dim 4 --max-snaps 80 \\
      --n-boot 500 --seed 0
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


def gather_per_snapshot(scenarios, res, N, dim, max_snaps):
    Xs, Xst, Ys, Ss, tags = [], [], [], [], []
    for sc in scenarios:
        for re in res:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{N}.npz")
            pp = os.path.join(RESULTS_DIR, f"patches_{sc}_Re{re}_N{N}_dim{dim}.npz")
            if not (os.path.exists(dp) and os.path.exists(pp)):
                continue
            dns = np.load(dp); patches = np.load(pp)
            vx = dns["vx"].astype(np.float64); vy = dns["vy"].astype(np.float64)
            Bx = dns["Bx"].astype(np.float64); By = dns["By"].astype(np.float64)
            Nf = vx.shape[1]
            l2 = patches["l2_errors"]; thr = float(patches["l2_threshold"])
            n = len(vx); step = max(1, n // max_snaps)
            idx = list(range(0, n, step))[:max_snaps]
            for si in idx:
                f2d, sc_v = extract_features_2d(
                    vx[si], vy[si], Bx[si], By[si], Nf, dim, re)
                Xs.append(f2d.reshape(-1, N_FEATS))
                Xst.append(stencil_features(f2d))
                Ys.append((l2[si] >= thr).ravel().astype(int))
                Ss.append(sc_v.ravel())
                tags.append(sc)
    return Xs, Xst, Ys, Ss, tags


def snap_f1(Y_list, P_list, thr):
    Y = np.concatenate(Y_list); P = np.concatenate(P_list)
    return f1_score(Y, (P > thr).astype(int), zero_division=0)


def bootstrap_ci(Y_list, P_list, thr, n_boot, rng, alpha=0.05):
    n = len(Y_list); f1s = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = np.concatenate([Y_list[i] for i in idx])
        p = np.concatenate([P_list[i] for i in idx])
        f1s[b] = f1_score(y, (p > thr).astype(int), zero_division=0)
    return float(np.quantile(f1s, alpha/2)), float(np.quantile(f1s, 1-alpha/2)), f1s


def paired_delta_ci(Y_list, Pa_list, thr_a, Pb_list, thr_b, n_boot, rng):
    """Paired bootstrap delta = F1(a) - F1(b) with one-sided p(delta <= 0)."""
    n = len(Y_list); ds = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = np.concatenate([Y_list[i] for i in idx])
        pa = np.concatenate([Pa_list[i] for i in idx])
        pb = np.concatenate([Pb_list[i] for i in idx])
        fa = f1_score(y, (pa > thr_a).astype(int), zero_division=0)
        fb = f1_score(y, (pb > thr_b).astype(int), zero_division=0)
        ds[b] = fa - fb
    lo = float(np.quantile(ds, 0.025)); hi = float(np.quantile(ds, 0.975))
    p_neg = float((ds <= 0).mean())  # fraction of replicates where a <= b
    return lo, hi, ds, p_neg


def main():
    p = argparse.ArgumentParser(
        description="Phase 11H: random-split bootstrap CI")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=80)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--val-frac", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  Phase 11H: random-split bootstrap CI")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}  "
          f"n_boot={args.n_boot}  val_frac={args.val_frac}")
    print("=" * 88)
    print()

    Xs, Xst, Ys, Ss, tags = gather_per_snapshot(
        args.scenario, args.re, args.N, args.dim, args.max_snaps)
    if not Xs:
        # D-75 : cette garde faisait `print(...); return` — code 0, aucun
        # artefact ecrit, donc indiscernable d'une campagne reussie (meme
        # famille que D-56 et D-74). Le detecteur AST de D-56 ne voyait que
        # la forme `if not <accumulateur nomme>:` ; celle-ci lui echappait.
        raise RuntimeError(
            "balayage vide : aucune configuration (scenario, Re) n'a d'artefact "
            "d'entree pour les arguments donnes. Le script sortait ici avec le "
            "code 0 et sans artefact (D-75).")
    print(f"  built dataset: {len(Xs)} snapshots across "
          f"{len(set(tags))} scenarios\n")

    # fixed random split by snapshot
    rng = np.random.default_rng(args.seed)
    n = len(Xs); perm = rng.permutation(n)
    n_va = max(1, int(args.val_frac * n))
    va = perm[:n_va]; tr = perm[n_va:]

    # training
    Xtr_s  = np.concatenate([Xs[i]  for i in tr])
    Xtr_st = np.concatenate([Xst[i] for i in tr])
    Ytr    = np.concatenate([Ys[i]  for i in tr])
    Str    = np.concatenate([Ss[i]  for i in tr])

    # validation kept per-snapshot for bootstrap
    Xv_list   = [Xs[i]  for i in va]
    Xv_st_lst = [Xst[i] for i in va]
    Yv_list   = [Ys[i]  for i in va]
    Sv_list   = [Ss[i]  for i in va]

    if len(np.unique(Ytr)) < 2:
        # D-75 : cette garde faisait `print(...); return` — code 0, aucun
        # artefact ecrit, donc indiscernable d'une campagne reussie (meme
        # famille que D-56 et D-74). Le detecteur AST de D-56 ne voyait que
        # la forme `if not <accumulateur nomme>:` ; celle-ci lui echappait.
        raise RuntimeError(
            "jeu d'entrainement degenere : une seule classe presente "
            f"({np.unique(Ytr).tolist()}), aucun classifieur ne peut etre ajuste. "
            "Le script sortait ici avec le code 0 et sans artefact (D-75).")

    # classical threshold
    thr_cls, _ = best_threshold_f1(Str, Ytr)
    # fit GBT site + stencil
    m_site = make_model("gbt", args.seed).fit(Xtr_s, Ytr)
    m_sten = make_model("gbt", args.seed).fit(Xtr_st, Ytr)
    P_site_list = [m_site.predict_proba(x)[:, 1] for x in Xv_list]
    P_sten_list = [m_sten.predict_proba(x)[:, 1] for x in Xv_st_lst]

    # pick thresholds on the full validation probabilities (same protocol
    # as fit_eval grid search)
    def best_thr(P_list, Y_list):
        thr, _ = best_threshold_f1(
            np.concatenate(P_list), np.concatenate(Y_list),
            grid=np.linspace(0.05, 0.95, 91))
        return thr
    thr_site = best_thr(P_site_list, Yv_list)
    thr_sten = best_thr(P_sten_list, Yv_list)

    # point estimates
    f1_class = snap_f1(Yv_list, Sv_list,    thr_cls)
    f1_site  = snap_f1(Yv_list, P_site_list, thr_site)
    f1_sten  = snap_f1(Yv_list, P_sten_list, thr_sten)

    # bootstrap CIs (shared rng, but each call draws its own indices)
    rng_b = np.random.default_rng(args.seed + 1)
    c_lo, c_hi, _ = bootstrap_ci(Yv_list, Sv_list,    thr_cls,  args.n_boot, rng_b)
    s_lo, s_hi, _ = bootstrap_ci(Yv_list, P_site_list, thr_site, args.n_boot, rng_b)
    st_lo, st_hi, _ = bootstrap_ci(Yv_list, P_sten_list, thr_sten, args.n_boot, rng_b)
    d_lo, d_hi, _, p_H0 = paired_delta_ci(
        Yv_list, P_site_list, thr_site, Sv_list, thr_cls,
        args.n_boot, rng_b)
    dst_lo, dst_hi, _, _ = paired_delta_ci(
        Yv_list, P_sten_list, thr_sten, P_site_list, thr_site,
        args.n_boot, rng_b)

    print(f"  n_val_snaps={len(Yv_list)}  n_val_cells="
          f"{sum(len(y) for y in Yv_list)}")
    print()
    print(f"  F1_class       = {f1_class:.3f}  [{c_lo:.3f}, {c_hi:.3f}]")
    print(f"  F1_site (GBT)  = {f1_site:.3f}  [{s_lo:.3f}, {s_hi:.3f}]")
    print(f"  F1_stencil     = {f1_sten:.3f}  [{st_lo:.3f}, {st_hi:.3f}]")
    print(f"  delta site-cls = {f1_site-f1_class:+.3f}  "
          f"[{d_lo:+.3f}, {d_hi:+.3f}]   p(site <= class) = {p_H0:.3f}")
    print(f"  delta sten-sit = {f1_sten-f1_site:+.3f}  "
          f"[{dst_lo:+.3f}, {dst_hi:+.3f}]")
    print()
    print("  INTERPRETATION:")
    if c_hi < s_lo:
        print("  * F1_site CI strictly above F1_class CI: the random-split")
        print("    ceiling is significantly higher than the classical")
        print("    indicator (as expected; this is the non-collapsed regime).")
    elif c_lo > s_hi:
        print("  * F1_class CI strictly above F1_site CI: unexpected --")
        print("    investigate.")
    else:
        print("  * Site and classical CIs overlap.")
    gap_hi = st_hi - s_lo
    print(f"  * stencil-vs-site gap CI upper bound = {dst_hi:+.3f} "
          "(tightness of the formal ceiling proposition).")

    out = os.path.join(RESULTS_DIR,
                       f"random_split_bootstrap_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        f1_class=f1_class, f1_class_ci=np.array([c_lo, c_hi]),
        f1_site =f1_site,  f1_site_ci =np.array([s_lo, s_hi]),
        f1_sten =f1_sten,  f1_sten_ci =np.array([st_lo, st_hi]),
        delta_site_class=f1_site-f1_class,
        delta_site_class_ci=np.array([d_lo, d_hi]),
        p_site_le_class=p_H0,
        delta_sten_site=f1_sten-f1_site,
        delta_sten_site_ci=np.array([dst_lo, dst_hi]),
        n_val_snaps=len(Yv_list),
        n_val_cells=sum(len(y) for y in Yv_list),
        thr_class=thr_cls, thr_site=thr_site, thr_sten=thr_sten,
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 11H complete.")


if __name__ == "__main__":
    main()
