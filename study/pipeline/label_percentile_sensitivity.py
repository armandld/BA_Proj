#!/usr/bin/env python3
"""
Phase 2B - Sensitivity of the falsification result to L2_PERCENTILE_HARD.

Phase 2 tags the top 25% of cells (by L2 coarsening error) as "hard".
A reviewer could reasonably ask: "does the ceiling collapse in phase 11B
(LOSO F1_site ~= 0.19) depend on that 25% cut-off?"

We sweep the percentile across a physically meaningful range
{60, 70, 75, 80, 85, 90} and for each value:

  (a) re-threshold the existing L2 error tensors into a new "is_hard" mask
      (cheap: no re-running of phase 2)
  (b) re-build the learned-H dataset and the classical score
  (c) run LOSO on the new labels to get F1_site and F1_class
  (d) report the LOSO delta (F1_site - F1_class) as a function of p

Expectation:
  - If F1_site - F1_class < 0 across ALL percentiles, the falsification
    is robust: no cut-off choice saves the local-Hamiltonian hypothesis.
  - If there is a magic percentile p* where the gap turns positive,
    that's a signal the L2 hard-patch label was pathological at p=25%.

We re-use the 9-feature mean-field ceiling (GBT) from phase 11 since
the stencil ceiling tracks it within ~0.002 everywhere.

Input:  results/dns_{sc}_Re{re}_N{N}.npz
        results/patches_{sc}_Re{re}_N{N}_dim{D}.npz  (for l2_errors)
Output: results/percentile_sensitivity_N{N}_dim{D}.npz

Usage:
  python study/label_percentile_sensitivity.py --dim 4
  python study/label_percentile_sensitivity.py --dim 4 \\
         --percentiles 50 60 70 75 80 85 90
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


def build_percentile_dataset(configs, dim, max_snaps, percentile):
    """Re-threshold L2 per-(scenario, Re) with the requested percentile.

    Matches phase 2's convention (threshold is per-config), which keeps
    the per-scenario pos_rate exactly at (100 - percentile)%. Then the
    LOSO fold sees balanced classes regardless of which scenario is held
    out -- which is the only fair way to measure sensitivity.
    """
    by_sc = {}
    thr_by_cfg = {}
    pos_rate_by_cfg = {}
    for sc, re, dns_path, patches_path in configs:
        dns = np.load(dns_path)
        patches = np.load(patches_path)
        vx = dns["vx"].astype(np.float64); vy = dns["vy"].astype(np.float64)
        Bx = dns["Bx"].astype(np.float64); By = dns["By"].astype(np.float64)
        N = vx.shape[1]
        l2_all = patches["l2_errors"]

        # per-config threshold (phase 2 convention)
        thr = float(np.percentile(l2_all.ravel(), percentile))
        thr_by_cfg[(sc, re)] = thr
        pos_rate_by_cfg[(sc, re)] = float((l2_all >= thr).mean())

        n_snaps = len(vx)
        step = max(1, n_snaps // max_snaps)
        idx = list(range(0, n_snaps, step))[:max_snaps]

        by_sc.setdefault(sc, dict(X=[], Y=[], S=[]))
        for si in idx:
            feats_2d, score = extract_features_2d(
                vx[si], vy[si], Bx[si], By[si], N, dim, re)
            by_sc[sc]["X"].append(feats_2d.reshape(-1, N_FEATS))
            by_sc[sc]["Y"].append((l2_all[si] >= thr).ravel().astype(int))
            by_sc[sc]["S"].append(score.ravel())

    for sc in by_sc:
        by_sc[sc]["X"] = np.concatenate(by_sc[sc]["X"])
        by_sc[sc]["Y"] = np.concatenate(by_sc[sc]["Y"])
        by_sc[sc]["S"] = np.concatenate(by_sc[sc]["S"])
    avg_pos = float(np.mean(list(pos_rate_by_cfg.values())))
    return by_sc, thr_by_cfg, avg_pos


def loso_site_vs_class(by_sc, seed):
    """LOSO for mean-field GBT (site) vs classical thr-sweep."""
    rows = []
    for held in by_sc:
        Xtr = np.concatenate([by_sc[s]["X"] for s in by_sc if s != held])
        Ytr = np.concatenate([by_sc[s]["Y"] for s in by_sc if s != held])
        Str = np.concatenate([by_sc[s]["S"] for s in by_sc if s != held])
        Xv, Yv, Sv = by_sc[held]["X"], by_sc[held]["Y"], by_sc[held]["S"]

        # classical
        thr_cls, _ = best_threshold_f1(Str, Ytr)
        f1_cls = f1_score(Yv, (Sv > thr_cls).astype(int), zero_division=0)

        # mean-field site ceiling
        # degenerate: all one class in train OR val -> skip model, report nan
        if len(np.unique(Ytr)) < 2 or len(np.unique(Yv)) < 2:
            f1_site = float("nan")
        else:
            r = fit_eval(make_model("gbt", seed), Xtr, Ytr, Xv, Yv)
            f1_site = r["f1"]

        rows.append(dict(held=held, n=len(Yv),
                         f1_class=f1_cls, f1_site=f1_site))
    return rows


def interpretation_message(rows_summary, percentiles):
    """Interpret the swept deltas against the module's own robustness
    criterion: robust iff the gap never turns positive (delta < 0
    everywhere), not merely small. Returns None if there is nothing to
    interpret.
    """
    if not rows_summary:
        return None
    deltas = [r["delta"] for r in rows_summary if np.isfinite(r["delta"])]
    if not deltas:
        return None
    if max(deltas) < 0:
        return (f"  * max(F1_site - F1_class) = {max(deltas):+.3f} "
                f"over p in {percentiles}  "
                f"==> the LOSO collapse is ROBUST to the percentile "
                f"choice. The local-Hamiltonian hypothesis fails for "
                f"ANY reasonable hard-patch definition.")
    i = int(np.argmax(deltas))
    best_p = rows_summary[i]["percentile"]
    return (f"  * at p={best_p:.0f}, F1_site beats classical by "
            f"{max(deltas):+.3f}  ==> the result is SENSITIVE to the "
            f"percentile cut-off. Investigate whether that specific "
            f"percentile corresponds to a physically meaningful "
            f"L2 error scale.")


def main():
    p = argparse.ArgumentParser(
        description="Phase 2B: L2_PERCENTILE sensitivity of the LOSO result")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--percentiles", nargs="+", type=float,
                   default=[60, 70, 75, 80, 85, 90])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  Phase 2B: Sensitivity of LOSO ceiling to L2_PERCENTILE_HARD")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}")
    print(f"  percentiles: {args.percentiles}")
    print("=" * 88)
    print()

    # -- discover which configs have data --
    configs = []
    for sc in args.scenario:
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(
                RESULTS_DIR, f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if os.path.exists(dp) and os.path.exists(pp):
                configs.append((sc, re, dp, pp))
    if len(configs) < 2:
        # D-75 : cette garde faisait `print(...); return` — code 0, aucun
        # artefact ecrit, donc indiscernable d'une campagne reussie (meme
        # famille que D-56 et D-74). Le detecteur AST de D-56 ne voyait que
        # la forme `if not <accumulateur nomme>:` ; celle-ci lui echappait.
        raise RuntimeError(
            "balayage vide : la sensibilite au percentile exige au moins 2 "
            f"configurations avec artefacts d'entree, {len(configs)} trouvee(s). "
            "Le script sortait ici avec le code 0 et sans artefact (D-75).")
    print(f"  {len(configs)} configs available across "
          f"{len(set(sc for sc,*_ in configs))} scenario(s)")

    header = (f"  {'percentile':>10} {'pos_rate':>8} "
              f"{'f1_class':>9} {'f1_site':>9} {'delta':>9}")
    rows_summary = []
    per_fold_log = []

    for pct in args.percentiles:
        t0 = time.time()
        by_sc, thr_by_cfg, pos_rate = build_percentile_dataset(
            configs, args.dim, args.max_snaps, pct)
        thr = float(np.mean(list(thr_by_cfg.values())))

        # If fewer than 2 scenarios available, LOSO is undefined; emit NaN.
        if len(by_sc) < 2:
            print(f"  p={pct:>4.0f}: only {len(by_sc)} scenario(s) -- "
                  f"LOSO undefined")
            continue

        rows = loso_site_vs_class(by_sc, args.seed)
        f1c = np.nanmean([r["f1_class"] for r in rows])
        f1s = np.nanmean([r["f1_site"]  for r in rows])
        delta = f1s - f1c
        dt = time.time() - t0

        rows_summary.append(dict(
            percentile=pct, threshold=thr, pos_rate=pos_rate,
            f1_class=f1c, f1_site=f1s, delta=delta,
            per_fold=rows, time_s=dt,
        ))
        per_fold_log.append((pct, rows))

        if len(rows_summary) == 1:
            print(header)
            print("  " + "-" * (len(header) - 2))
        print(f"  {pct:>10.0f} {pos_rate:>8.3f} "
              f"{f1c:>9.3f} {f1s:>9.3f} {delta:>+9.3f}   "
              f"[{dt:.1f}s]")

    # -- interpretation --
    print()
    print("  per-fold breakdown:")
    print(f"  {'p':>4}  {'held':<18} {'n_val':>7} "
          f"{'f1_class':>9} {'f1_site':>9} {'delta':>9}")
    for pct, rows in per_fold_log:
        for r in rows:
            f1c = r["f1_class"]; f1s = r["f1_site"]
            f1s_str = "      nan" if not np.isfinite(f1s) else f"{f1s:>9.3f}"
            d_str = ("      nan" if not np.isfinite(f1s)
                     else f"{f1s - f1c:>+9.3f}")
            print(f"  {pct:>4.0f}  {r['held']:<18} {r['n']:>7d} "
                  f"{f1c:>9.3f} {f1s_str} {d_str}")

    print()
    print("  INTERPRETATION:")
    msg = interpretation_message(rows_summary, args.percentiles)
    if msg:
        print(msg)

    # -- save --
    out = os.path.join(RESULTS_DIR,
                       f"percentile_sensitivity_N{args.N}_dim{args.dim}.npz")
    if rows_summary:
        np.savez_compressed(
            out,
            percentiles=np.array([r["percentile"] for r in rows_summary]),
            thresholds =np.array([r["threshold"]  for r in rows_summary]),
            pos_rates  =np.array([r["pos_rate"]   for r in rows_summary]),
            f1_class   =np.array([r["f1_class"]   for r in rows_summary]),
            f1_site    =np.array([r["f1_site"]    for r in rows_summary]),
            deltas     =np.array([r["delta"]      for r in rows_summary]),
        )
        print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 2B complete.")


if __name__ == "__main__":
    main()
