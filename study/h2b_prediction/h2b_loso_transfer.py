#!/usr/bin/env python3
"""
Phase 11B - Leave-One-Scenario-Out validation of the upper bound.

Phase 11 (phase11_upper_bound.py) reports F1_site ~= 0.99 with a random
snapshot split. A reviewer will ask: "does the classifier memorise
signatures per scenario via |B|^2 / Re, or is the ceiling a real
per-site property?"

This phase resolves the ambiguity. For each fold:
  - all snapshots of ONE scenario  -> validation set
  - all snapshots of the 3 others  -> training set
  -> strict generalisation across physics regimes.

The F1 we report (site-LOSO, stencil-LOSO) is the honest ceiling
for ANY mean-field / neighbourhood Hamiltonian deployed on an
UNSEEN MHD instability class -- which is the practical use case.

Interpretation:
  - F1_site_LOSO >> F1_classical  ->  a learned local H generalises.
  - F1_site_LOSO ~= F1_classical  ->  the phase 11 ceiling was
      inter-scenario memorisation; real per-site signal is weak.

Uses the same 9 features and stencil construction as phase 11.

Output: results/upper_bound_loso_N{N}_dim{D}.npz

Usage:
  python study/phase11b_loso.py --dim 4
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
    extract_features_2d, stencil_features,
    make_model, fit_eval, best_threshold_f1,
)


def _gather_scenario(dns_paths_for_scen, dim, max_snaps):
    """Return (X_site, X_sten, Y, S) stacked for one scenario."""
    X_site, X_sten, Y, S = [], [], [], []
    for re, dns_path, patches_path in dns_paths_for_scen:
        dns = np.load(dns_path)
        patches = np.load(patches_path)
        vx_all = dns["vx"].astype(np.float64)
        vy_all = dns["vy"].astype(np.float64)
        Bx_all = dns["Bx"].astype(np.float64)
        By_all = dns["By"].astype(np.float64)
        N = vx_all.shape[1]
        l2_all = patches["l2_errors"]; l2_thr = float(patches["l2_threshold"])

        n_snaps = len(vx_all)
        step = max(1, n_snaps // max_snaps)
        idx = list(range(0, n_snaps, step))[:max_snaps]

        for si in idx:
            feats_2d, score = extract_features_2d(
                vx_all[si], vy_all[si], Bx_all[si], By_all[si],
                N, dim, re,
            )
            X_site.append(feats_2d.reshape(-1, N_FEATS))
            X_sten.append(stencil_features(feats_2d))
            Y.append((l2_all[si] >= l2_thr).ravel().astype(int))
            S.append(score.ravel())
    return (np.concatenate(X_site) if X_site else np.zeros((0, N_FEATS)),
            np.concatenate(X_sten) if X_sten else np.zeros((0, 5 * N_FEATS)),
            np.concatenate(Y) if Y else np.zeros(0, dtype=int),
            np.concatenate(S) if S else np.zeros(0))


def main():
    p = argparse.ArgumentParser(
        description="Phase 11B: Leave-one-scenario-out ceiling")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--label-suffix", default="",
                   help="variante de label, ex. _globalthr (T28). Le suffixe est repercute dans le nom de sortie pour qu'une variante n'ecrase jamais l'autre.")

    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  Phase 11B: Leave-One-Scenario-Out validation")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}")
    print("  Folds:", ", ".join(args.scenario))
    print("=" * 88)
    print()

    # ---- gather per-scenario data ----
    by_scene = {}
    for sc in args.scenario:
        rows = []
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(RESULTS_DIR,
                              f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}{args.label_suffix}.npz")
            if os.path.exists(dp) and os.path.exists(pp):
                rows.append((re, dp, pp))
        if rows:
            by_scene[sc] = rows

    if len(by_scene) < 2:
        print("need >=2 scenarios with data to run LOSO."); return

    # pre-build per-scenario feature matrices
    print("  building per-scenario feature matrices...")
    t0 = time.time()
    data = {}
    for sc, rows in by_scene.items():
        Xs, Xn, Y, S = _gather_scenario(rows, args.dim, args.max_snaps)
        data[sc] = dict(X_site=Xs, X_sten=Xn, Y=Y, S=S)
        print(f"    {sc:<18} cells={len(Y):>6}  pos_rate={Y.mean():.3f}")
    print(f"  done in {time.time() - t0:.1f}s\n")

    # ---- LOSO folds ----
    rows = []
    header = f"  {'held-out':<18} {'n_val':>7} {'F1_class':>9} " \
             f"{'F1_site':>9} {'F1_sten':>9} {'d_site':>8} {'d_sten':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for held in by_scene:
        Xtr_site = np.concatenate(
            [data[sc]["X_site"] for sc in by_scene if sc != held])
        Xtr_sten = np.concatenate(
            [data[sc]["X_sten"] for sc in by_scene if sc != held])
        Ytr = np.concatenate(
            [data[sc]["Y"] for sc in by_scene if sc != held])
        Str = np.concatenate(
            [data[sc]["S"] for sc in by_scene if sc != held])

        Xva_site = data[held]["X_site"]
        Xva_sten = data[held]["X_sten"]
        Yva      = data[held]["Y"]
        Sva      = data[held]["S"]

        # classical baseline: best thr on train, applied on val
        thr_star, _ = best_threshold_f1(Str, Ytr)
        from sklearn.metrics import f1_score
        f1_cls = f1_score(Yva, (Sva > thr_star).astype(int),
                          zero_division=0)

        # site GBT
        r_site = fit_eval(make_model("gbt", args.seed),
                          Xtr_site, Ytr, Xva_site, Yva)

        # stencil GBT
        r_sten = fit_eval(make_model("gbt", args.seed),
                          Xtr_sten, Ytr, Xva_sten, Yva)

        print(f"  {held:<18} {len(Yva):>7d} {f1_cls:>9.3f} "
              f"{r_site['f1']:>9.3f} {r_sten['f1']:>9.3f} "
              f"{r_site['f1'] - f1_cls:>+8.3f} "
              f"{r_sten['f1'] - r_site['f1']:>+8.3f}")

        rows.append(dict(
            held=held, n_val=len(Yva), thr_class=thr_star,
            f1_class=f1_cls,
            f1_site=r_site["f1"], auc_site=r_site["auc"],
            f1_sten=r_sten["f1"], auc_sten=r_sten["auc"],
        ))

    # ---- aggregate ----
    f1_cls_mean  = float(np.mean([r["f1_class"] for r in rows]))
    f1_cls_std   = float(np.std( [r["f1_class"] for r in rows]))
    f1_site_mean = float(np.mean([r["f1_site"] for r in rows]))
    f1_site_std  = float(np.std( [r["f1_site"] for r in rows]))
    f1_sten_mean = float(np.mean([r["f1_sten"] for r in rows]))
    f1_sten_std  = float(np.std( [r["f1_sten"] for r in rows]))

    print("  " + "-" * (len(header) - 2))
    print(f"  {'MEAN +/- STD':<18} {'':>7} "
          f"{f1_cls_mean:.3f}+/-{f1_cls_std:.3f}  "
          f"{f1_site_mean:.3f}+/-{f1_site_std:.3f}  "
          f"{f1_sten_mean:.3f}+/-{f1_sten_std:.3f}")

    print("\n  INTERPRETATION:")
    d_site_loso = f1_site_mean - f1_cls_mean
    d_sten_loso = f1_sten_mean - f1_site_mean
    if d_site_loso < 0.05:
        print(f"  * site-LOSO ~= classical (delta = {d_site_loso:+.3f})  "
              f"==> the phase 11 ceiling was largely INTER-SCENARIO "
              f"memorisation. For unseen scenarios, a local H barely "
              f"beats the classical indicator.")
    elif d_site_loso < 0.15:
        print(f"  * site-LOSO > classical by {d_site_loso:+.3f}  "
              f"==> modest cross-scenario transfer; a learned local H "
              f"offers real but limited generalisation.")
    else:
        print(f"  * site-LOSO > classical by {d_site_loso:+.3f}  "
              f"==> STRONG cross-scenario transfer; the local-field "
              f"signal generalises. A learned mean-field H is viable.")

    if d_sten_loso < 0.02:
        print(f"  * stencil ~= site even under LOSO (delta = "
              f"{d_sten_loso:+.3f})  ==> ZZ/ZZZZ couplings remain "
              f"unnecessary across scenarios.")
    else:
        print(f"  * stencil > site by {d_sten_loso:+.3f} under LOSO  "
              f"==> neighbourhood couplings help for transfer.")

    # ---- save ----
    out = os.path.join(RESULTS_DIR,
                       f"upper_bound_loso_N{args.N}_dim{args.dim}{args.label_suffix}.npz")
    np.savez_compressed(
        out,
        held=np.array([r["held"] for r in rows]),
        n_val=np.array([r["n_val"] for r in rows]),
        f1_class=np.array([r["f1_class"] for r in rows]),
        f1_site =np.array([r["f1_site"]  for r in rows]),
        f1_sten =np.array([r["f1_sten"]  for r in rows]),
        auc_site=np.array([r["auc_site"] for r in rows]),
        auc_sten=np.array([r["auc_sten"] for r in rows]),
        f1_class_mean=f1_cls_mean, f1_class_std=f1_cls_std,
        f1_site_mean =f1_site_mean, f1_site_std =f1_site_std,
        f1_sten_mean =f1_sten_mean, f1_sten_std =f1_sten_std,
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 11B complete.")


if __name__ == "__main__":
    main()
