#!/usr/bin/env python3
"""
Phase 11C - Learned mean-field Hamiltonian with physical features.

Phase 11 showed that a GBT on 9 per-site features reaches F1 ~= 0.99,
while the v2 Hamiltonian (h_i = c_bias * M * (score_i - thr)) plateaus
around F1 ~= 0.48. The gap comes from the feature set: v2 only sees
score_classical, whereas the ceiling model exploits |B|^2, |grad_B|^2,
|J_z|, ...

This phase materialises the learned mean-field Hamiltonian:

    h_i = w . phi_i - b

where phi_i is the standardised 9-feature vector at site i. We fit
(w, b) by logistic regression on the L2-hard label; the decision
"refine i" corresponds to h_i > 0 (spin convention: +1 = don't refine,
-1 = refine, so we want h_i > 0  =>  s_i = -1 via argmin h_i*s_i).

Couplings ZZ / ZZZZ are kept at the v2 parameter-free values
(w_ZZ = 2, w_ZZZZ = 1). Phase 11 proved they cannot add more than
+0.002 F1 over the mean-field ceiling, so their exact magnitude
is irrelevant for this part of the study; they matter only for the
QAOA / SA comparison on the resulting H.

Pipeline:
  1. Build the feature pool over training snapshots.
  2. Fit logistic regression -> (w, b).
  3. Evaluate the resulting per-site decision rule on held-out snapshots.
  4. Optionally compare against the classical indicator and v2 baseline.

Output: results/learned_h_N{N}_dim{D}.npz
          keys: w, b, feature_names, f1_val, f1_class_val,
                per_scene_f1_learned, per_scene_f1_class

Usage:
  python study/phase11c_learned_h.py --dim 4
  python study/phase11c_learned_h.py --dim 4 --loso
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
    extract_features_2d, build_dataset, best_threshold_f1,
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


def fit_learned_h(Xtr, Ytr, seed=0, C_reg=1.0):
    """Fit logistic regression -> effective (w, b) in standardised feature
    space. Decision: h_i = w . phi_i - b > 0  <=>  refine."""
    scaler = StandardScaler().fit(Xtr)
    Zt = scaler.transform(Xtr)
    lr = LogisticRegression(
        max_iter=1000, C=C_reg, class_weight="balanced",
        random_state=seed,
    ).fit(Zt, Ytr)
    # p(refine = 1) = sigmoid(w.z + b_lr); h_i = w.z + b_lr (match
    # Hamiltonian convention: h_i > 0 -> prefer s_i = -1 = refine).
    w_std = lr.coef_.ravel().copy()
    b_std = float(lr.intercept_.ravel()[0])
    # unfold standardisation back into the raw-feature space:
    # h_i = sum_k w_std_k * (x_k - mu_k) / sigma_k + b_std
    #     = sum_k (w_std_k / sigma_k) * x_k  +  (b_std - sum_k w_std_k * mu_k / sigma_k)
    mu = scaler.mean_; sigma = scaler.scale_
    w_raw = w_std / sigma
    b_raw = b_std - np.sum(w_std * mu / sigma)
    return dict(scaler=scaler, lr=lr, w_raw=w_raw, b_raw=b_raw,
                w_std=w_std, b_std=b_std)


def predict_h(model, X):
    """Raw Hamiltonian-convention field h_i; decision = (h_i > thr_best)."""
    Z = model["scaler"].transform(X)
    return Z @ model["w_std"] + model["b_std"]


def main():
    p = argparse.ArgumentParser(
        description="Phase 11C: learned mean-field Hamiltonian (w, b)")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--C-reg", type=float, default=1.0,
                   help="LR inverse regularisation")
    p.add_argument("--loso", action="store_true",
                   help="also report leave-one-scenario-out F1")
    args = p.parse_args()

    print("=" * 88)
    print("  Phase 11C: learned mean-field Hamiltonian h_i = w.phi_i - b")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}")
    print("=" * 88)
    print()

    # ---- gather data (reuse phase 11 builder) ----
    configs = []
    for sc in args.scenario:
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(RESULTS_DIR,
                              f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if os.path.exists(dp) and os.path.exists(pp):
                configs.append((sc, re, dp, pp))
    if not configs:
        print("no input."); return

    t0 = time.time()
    X_site, X_sten, Y_snap, S_snap, tags = build_dataset(
        configs, args.dim, args.max_snaps)
    print(f"  built: {len(X_site)} snaps in {time.time()-t0:.1f}s")

    # ---- random split ----
    rng = np.random.default_rng(args.seed)
    n_snaps = len(X_site)
    perm = rng.permutation(n_snaps)
    n_tr = max(1, int(args.train_frac * n_snaps))
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]

    def stack(pool, idxs):
        return np.concatenate([pool[i] for i in idxs], axis=0)

    Xtr = stack(X_site, tr_idx); Xva = stack(X_site, va_idx)
    Ytr = stack(Y_snap, tr_idx); Yva = stack(Y_snap, va_idx)
    Str = stack(S_snap, tr_idx); Sva = stack(S_snap, va_idx)
    Tva = np.concatenate([
        np.full(args.dim * args.dim, tags[i]) for i in va_idx])

    # ---- fit learned H ----
    t0 = time.time()
    model = fit_learned_h(Xtr, Ytr, seed=args.seed, C_reg=args.C_reg)
    h_tr = predict_h(model, Xtr)
    h_va = predict_h(model, Xva)
    # best threshold on h_tr against Ytr
    thr_star, f1_tr = best_threshold_f1(
        h_tr, Ytr,
        grid=np.linspace(h_tr.min(), h_tr.max(), 201))
    pred_va = (h_va > thr_star).astype(int)
    f1_val  = f1_score(Yva, pred_va, zero_division=0)
    dt = time.time() - t0

    # classical baseline on same split
    thr_cls, _ = best_threshold_f1(Str, Ytr)
    f1_cls_val = f1_score(Yva, (Sva > thr_cls).astype(int),
                           zero_division=0)

    print(f"\n  [random split] learned-H F1_train={f1_tr:.3f}  "
          f"F1_val={f1_val:.3f}  "
          f"classical F1_val={f1_cls_val:.3f}   [{dt:.1f}s]")

    print("\n  learned weights (in standardised feature space):")
    print(f"    {'feature':<18} {'w_std':>10} {'w_raw':>12}")
    order = np.argsort(-np.abs(model["w_std"]))
    for k in order:
        print(f"    {FEATURE_NAMES[k]:<18} "
              f"{model['w_std'][k]:>+10.3f} "
              f"{model['w_raw'][k]:>+12.3e}")
    print(f"    {'(bias b)':<18} {model['b_std']:>+10.3f} "
          f"{model['b_raw']:>+12.3e}")

    # ---- per-scenario breakdown ----
    print("\n  per-scenario (val):")
    print(f"  {'scenario':<18} {'n':>6} {'class':>8} {'learn':>8} "
          f"{'delta':>8}")
    per_scene = {}
    for sc in sorted(set(Tva)):
        mask = Tva == sc
        if not mask.any():
            continue
        f1c = f1_score(Yva[mask], (Sva[mask] > thr_cls).astype(int),
                       zero_division=0)
        f1l = f1_score(Yva[mask], (h_va[mask] > thr_star).astype(int),
                       zero_division=0)
        per_scene[sc] = (f1c, f1l)
        print(f"  {sc:<18} {int(mask.sum()):>6d} {f1c:>8.3f} "
              f"{f1l:>8.3f} {f1l-f1c:>+8.3f}")

    # ---- optional LOSO ----
    loso_rows = []
    if args.loso:
        print("\n  [leave-one-scenario-out]")
        print(f"  {'held':<18} {'n_val':>7} {'class':>8} {'learn':>8} "
              f"{'delta':>8}")
        # rebuild per-scenario stacks once
        X_by_sc, Y_by_sc, S_by_sc = {}, {}, {}
        for i in range(n_snaps):
            sc = tags[i]
            X_by_sc.setdefault(sc, []).append(X_site[i])
            Y_by_sc.setdefault(sc, []).append(Y_snap[i])
            S_by_sc.setdefault(sc, []).append(S_snap[i])
        for sc in X_by_sc:
            X_by_sc[sc] = np.concatenate(X_by_sc[sc])
            Y_by_sc[sc] = np.concatenate(Y_by_sc[sc])
            S_by_sc[sc] = np.concatenate(S_by_sc[sc])

        for held in X_by_sc:
            Xt = np.concatenate([X_by_sc[s] for s in X_by_sc if s != held])
            Yt = np.concatenate([Y_by_sc[s] for s in X_by_sc if s != held])
            St = np.concatenate([S_by_sc[s] for s in X_by_sc if s != held])
            Xv, Yv, Sv = X_by_sc[held], Y_by_sc[held], S_by_sc[held]

            m = fit_learned_h(Xt, Yt, seed=args.seed, C_reg=args.C_reg)
            h_t = predict_h(m, Xt); h_v = predict_h(m, Xv)
            thr_l, _ = best_threshold_f1(
                h_t, Yt, grid=np.linspace(h_t.min(), h_t.max(), 201))
            thr_c, _ = best_threshold_f1(St, Yt)
            f1l = f1_score(Yv, (h_v > thr_l).astype(int),
                           zero_division=0)
            f1c = f1_score(Yv, (Sv > thr_c).astype(int),
                           zero_division=0)
            print(f"  {held:<18} {len(Yv):>7d} {f1c:>8.3f} "
                  f"{f1l:>8.3f} {f1l-f1c:>+8.3f}")
            loso_rows.append(dict(held=held, n_val=len(Yv),
                                   f1_class=f1c, f1_learned=f1l,
                                   w_std=m["w_std"], b_std=m["b_std"]))
        if loso_rows:
            m_cls = float(np.mean([r["f1_class"]   for r in loso_rows]))
            m_lrn = float(np.mean([r["f1_learned"] for r in loso_rows]))
            print(f"  {'MEAN':<18} {'':>7} {m_cls:>8.3f} {m_lrn:>8.3f} "
                  f"{m_lrn-m_cls:>+8.3f}")

    # ---- save ----
    out = os.path.join(RESULTS_DIR, f"learned_h_N{args.N}_dim{args.dim}.npz")
    save_kw = dict(
        feature_names=np.array(FEATURE_NAMES),
        w_std=model["w_std"], b_std=model["b_std"],
        w_raw=model["w_raw"], b_raw=model["b_raw"],
        scaler_mean=model["scaler"].mean_,
        scaler_scale=model["scaler"].scale_,
        thr_star=thr_star,
        f1_train=f1_tr, f1_val=f1_val,
        f1_class_val=f1_cls_val,
        thr_class=thr_cls,
        per_scene_tags=np.array(list(per_scene.keys())),
        per_scene_f1_class  =np.array([v[0] for v in per_scene.values()]),
        per_scene_f1_learned=np.array([v[1] for v in per_scene.values()]),
    )
    if loso_rows:
        save_kw.update(dict(
            loso_held=np.array([r["held"] for r in loso_rows]),
            loso_f1_class  =np.array([r["f1_class"]   for r in loso_rows]),
            loso_f1_learned=np.array([r["f1_learned"] for r in loso_rows]),
        ))
    np.savez_compressed(out, **save_kw)
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 11C complete.")


if __name__ == "__main__":
    main()
