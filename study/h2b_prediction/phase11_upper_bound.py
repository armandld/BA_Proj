#!/usr/bin/env python3
"""
Phase 11A - Upper bounds on per-site classifiers from local MHD features.

Answers two strategic questions:

  Q1 (mean-field ceiling): what is the best possible F1 for a decision
     rule of the form  refine(i) = f(local fields at cell i)?
     This is the ceiling for ANY Ising Hamiltonian whose Z-bias is a
     local function of the fields and whose couplings are negligible
     (i.e. where only h_i matters). If this ceiling is <= the classical
     indicator F1, no local-bias Hamiltonian can do better than classical.

  Q2 (neighbourhood ceiling): what is the best F1 for a decision rule
     of the form  refine(i) = g(local fields at i AND its 4 neighbours)?
     This is the ceiling for an Ising Hamiltonian WITH ZZ and ZZZZ
     couplings: the minimiser propagates information one hop per step,
     but even an optimal such Hamiltonian cannot beat a classifier that
     reads the full stencil directly.

Three models for Q1 provide a robust ceiling (different inductive biases):
   - Logistic Regression   (linear, sanity baseline)
   - Random Forest         (high-variance non-linear)
   - Hist Gradient Boost   (low-variance non-linear)
If all three converge to ~the same F1, the ceiling is credible.

One model for Q2 suffices: GBT on stencil features (self + 4 neighbours).
The delta (Q2 - Q1) quantifies the residual value of couplings in
ANY local Hamiltonian -- regardless of the specific H.

Features per cell (at dim x dim VQA resolution):
   1. score_classical           (already-computed AMR indicator)
   2. |v|^2                      (kinetic energy density)
   3. |B|^2                      (magnetic energy density)
   4. |omega_z|                  (vorticity magnitude)
   5. |J_z|                      (current magnitude)
   6. |grad v|^2                 (velocity gradient norm)
   7. |grad B|^2                 (B-field gradient norm)
   8. det(grad B)                (X-point indicator)
   9. Re                         (Reynolds, scalar broadcast)

For Q2, stencil features concatenate the 9 features of the cell itself
with the 9 features of its 4 periodic neighbours (N, S, E, W) -> 45 feats.

Split: train/val by SNAPSHOT (not by cell) to avoid per-snapshot leakage.

Input:  results/dns_{sc}_Re{re}_N{N}.npz
        results/patches_{sc}_Re{re}_N{N}_dim{D}.npz
Output: results/upper_bound_N{N}_dim{D}.npz

Usage:
  python study/phase11_upper_bound.py --dim 4
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

from phase4_exact_diag import build_patch_hamiltonian

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score


FEATURE_NAMES = [
    "score_classical", "|v|^2", "|B|^2", "|omega_z|", "|J_z|",
    "|grad_v|^2", "|grad_B|^2", "det_grad_B", "Re",
]
N_FEATS = len(FEATURE_NAMES)


# -------------------------------------------------------------------
# Feature extraction per snapshot at dim x dim resolution
# -------------------------------------------------------------------

def _block_avg(f, patch_size, dim):
    return f.reshape(dim, patch_size, dim, patch_size).mean(axis=(1, 3))


def extract_features_2d(vx, vy, Bx, By, N, dim, Re):
    """Return (dim, dim, n_features) feature tensor + (dim, dim) score."""
    hp, score_vqa, _ = build_patch_hamiltonian(
        vx, vy, Bx, By, N, dim, Re,
        threshold_amr=0.15, use_v2=True, c_bias=1.0,
    )
    ps = N // dim

    vx_d = _block_avg(vx, ps, dim); vy_d = _block_avg(vy, ps, dim)
    Bx_d = _block_avg(Bx, ps, dim); By_d = _block_avg(By, ps, dim)

    dxvy = np.roll(vy_d, -1, axis=1) - vy_d
    dyvx = np.roll(vx_d, -1, axis=0) - vx_d
    dxBy = np.roll(By_d, -1, axis=1) - By_d
    dyBx = np.roll(Bx_d, -1, axis=0) - Bx_d
    dxvx = np.roll(vx_d, -1, axis=1) - vx_d
    dyvy = np.roll(vy_d, -1, axis=0) - vy_d
    dxBx = np.roll(Bx_d, -1, axis=1) - Bx_d
    dyBy = np.roll(By_d, -1, axis=0) - By_d

    omega_z = dxvy - dyvx
    J_z     = dxBy - dyBx
    grad_v2 = dxvx**2 + dyvx**2 + dxvy**2 + dyvy**2
    grad_B2 = dxBx**2 + dyBx**2 + dxBy**2 + dyBy**2
    det_gB  = dxBx * dyBy - dyBx * dxBy

    feats = np.stack([
        score_vqa,
        vx_d**2 + vy_d**2,
        Bx_d**2 + By_d**2,
        np.abs(omega_z),
        np.abs(J_z),
        grad_v2,
        grad_B2,
        det_gB,
        np.full((dim, dim), float(Re)),
    ], axis=-1)  # shape (dim, dim, 9)
    return feats, score_vqa


def stencil_features(feats_2d):
    """Augment (dim, dim, F) to (dim*dim, 5*F): self + N/S/E/W periodic."""
    f_n = np.roll(feats_2d, -1, axis=0)
    f_s = np.roll(feats_2d, +1, axis=0)
    f_e = np.roll(feats_2d, -1, axis=1)
    f_w = np.roll(feats_2d, +1, axis=1)
    cat = np.concatenate([feats_2d, f_n, f_s, f_e, f_w], axis=-1)
    return cat.reshape(-1, cat.shape[-1])


# -------------------------------------------------------------------
# Dataset build
# -------------------------------------------------------------------

def build_dataset(configs, dim, max_snaps_per_cfg):
    """Return parallel lists, one entry per snapshot.
    X_site: (dim*dim, 9)     X_sten: (dim*dim, 45)
    Y:      (dim*dim,)       S: (dim*dim,) classical score
    """
    X_site, X_sten, Y_snap, S_snap, tags = [], [], [], [], []

    for sc, re, dns_path, patches_path in configs:
        dns = np.load(dns_path)
        patches = np.load(patches_path)
        vx_all = dns["vx"].astype(np.float64)
        vy_all = dns["vy"].astype(np.float64)
        Bx_all = dns["Bx"].astype(np.float64)
        By_all = dns["By"].astype(np.float64)
        N = vx_all.shape[1]
        l2_all = patches["l2_errors"]; l2_thr = float(patches["l2_threshold"])

        n_snaps = len(vx_all)
        step = max(1, n_snaps // max_snaps_per_cfg)
        idx = list(range(0, n_snaps, step))[:max_snaps_per_cfg]

        for si in idx:
            feats_2d, score = extract_features_2d(
                vx_all[si], vy_all[si], Bx_all[si], By_all[si],
                N, dim, re,
            )
            X_site.append(feats_2d.reshape(-1, N_FEATS))
            X_sten.append(stencil_features(feats_2d))
            Y_snap.append((l2_all[si] >= l2_thr).ravel().astype(int))
            S_snap.append(score.ravel())
            tags.append(sc)

    return X_site, X_sten, Y_snap, S_snap, tags


# -------------------------------------------------------------------
# Best-F1 threshold sweep on a score vector
# -------------------------------------------------------------------

def best_threshold_f1(scores, gt, grid=None):
    if grid is None:
        q = np.quantile(scores, np.linspace(0.05, 0.95, 19))
        grid = np.unique(np.concatenate([np.linspace(0.02, 0.6, 59), q]))
    best = (float(grid[0]), 0.0)
    for thr in grid:
        pred = (scores > thr).astype(int)
        f1 = f1_score(gt, pred, zero_division=0)
        if f1 > best[1]:
            best = (float(thr), float(f1))
    return best


# -------------------------------------------------------------------
# Model factory
# -------------------------------------------------------------------

def make_model(name, seed):
    if name == "lr":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, C=1.0,
                                        class_weight="balanced",
                                        random_state=seed)),
        ])
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, class_weight="balanced", random_state=seed,
        )
    if name == "gbt":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=20,
            random_state=seed,
        )
    raise ValueError(name)


def fit_eval(model, Xtr, Ytr, Xva, Yva, thr_grid=None):
    t0 = time.time()
    model.fit(Xtr, Ytr)
    grid = (thr_grid if thr_grid is not None
            else np.linspace(0.05, 0.95, 91))
    p_tr = model.predict_proba(Xtr)[:, 1]
    p = model.predict_proba(Xva)[:, 1]
    thr, _ = best_threshold_f1(p_tr, Ytr, grid=grid)
    f1 = f1_score(Yva, (p > thr).astype(int), zero_division=0)
    try:
        auc = roc_auc_score(Yva, p)
    except Exception:
        auc = float("nan")
    return dict(p=p, thr=thr, f1=f1, auc=auc, fit_s=time.time() - t0)


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Phase 11A: per-site upper bounds (mean-field + stencil)")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--label-suffix", default="",
                   help="variante de label, ex. _globalthr (T28). Le suffixe est repercute dans le nom de sortie pour qu'une variante n'ecrase jamais l'autre.")

    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30,
                   help="snapshots per (scenario, Re) included in dataset")
    p.add_argument("--train-frac", type=float, default=0.7,
                   help="train split by SNAPSHOT (avoids cell leakage)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  Phase 11A: per-site (mean-field) and stencil (neighbourhood) ceilings")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}")
    print("=" * 88)
    print()

    configs = []
    for sc in args.scenario:
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(RESULTS_DIR,
                              f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}{args.label_suffix}.npz")
            if os.path.exists(dp) and os.path.exists(pp):
                configs.append((sc, re, dp, pp))
            else:
                print(f"  SKIP {sc} Re={re}: missing input")
    if not configs:
        print("no input."); return

    # -- build dataset --
    t0 = time.time()
    X_site, X_sten, Y_snap, S_snap, tags = build_dataset(
        configs, args.dim, args.max_snaps)
    dt = time.time() - t0
    n_snaps = len(X_site)
    n_cells = n_snaps * args.dim * args.dim
    print(f"  built dataset: {n_snaps} snaps, {n_cells} cells"
          f"   site-feats=9, stencil-feats={X_sten[0].shape[1]}  [{dt:.1f}s]")

    # -- split by snapshot --
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_snaps)
    n_tr = max(1, int(args.train_frac * n_snaps))
    tr_idx = perm[:n_tr]; va_idx = perm[n_tr:]
    print(f"  split: {len(tr_idx)} train snaps, {len(va_idx)} val snaps")

    def _stack(pool, idxs):
        return np.concatenate([pool[i] for i in idxs], axis=0)

    Xtr_site = _stack(X_site, tr_idx); Xva_site = _stack(X_site, va_idx)
    Xtr_sten = _stack(X_sten, tr_idx); Xva_sten = _stack(X_sten, va_idx)
    Ytr      = _stack(Y_snap, tr_idx); Yva      = _stack(Y_snap, va_idx)
    Str      = _stack(S_snap, tr_idx); Sva      = _stack(S_snap, va_idx)
    Tva      = np.concatenate([
        np.full(args.dim * args.dim, tags[i]) for i in va_idx])

    # -- classical baseline --
    thr_star, f1_class_tr = best_threshold_f1(Str, Ytr)
    f1_class_va = f1_score(Yva, (Sva > thr_star).astype(int),
                           zero_division=0)
    print(f"\n  [baseline] classical: thr*={thr_star:.3f}  "
          f"F1_train={f1_class_tr:.3f}  F1_val={f1_class_va:.3f}")

    # -- Q1: mean-field ceiling (3 models on per-site features) --
    print("\n  [Q1] mean-field ceiling (per-site features, 9 inputs):")
    print(f"  {'model':<8} {'F1_val':>8} {'AUC':>8} {'thr':>6} {'fit_s':>8}")
    site_res = {}
    for name in ("lr", "rf", "gbt"):
        r = fit_eval(make_model(name, args.seed),
                     Xtr_site, Ytr, Xva_site, Yva)
        site_res[name] = r
        print(f"  {name:<8} {r['f1']:>8.3f} {r['auc']:>8.3f} "
              f"{r['thr']:>6.2f} {r['fit_s']:>8.1f}")

    f1_site_best = max(r["f1"] for r in site_res.values())
    best_site_model = max(site_res, key=lambda k: site_res[k]["f1"])
    print(f"  -> mean-field ceiling = {f1_site_best:.3f}  "
          f"(best model: {best_site_model})")

    # -- Q2: neighbourhood ceiling (GBT on stencil) --
    print("\n  [Q2] neighbourhood ceiling (stencil features, 45 inputs):")
    r_sten = fit_eval(make_model("gbt", args.seed),
                      Xtr_sten, Ytr, Xva_sten, Yva)
    print(f"  {'gbt-sten':<8} {r_sten['f1']:>8.3f} "
          f"{r_sten['auc']:>8.3f} {r_sten['thr']:>6.2f} "
          f"{r_sten['fit_s']:>8.1f}")

    # -- verdicts --
    delta_site_class = f1_site_best - f1_class_va
    delta_sten_site  = r_sten["f1"] - f1_site_best
    delta_sten_class = r_sten["f1"] - f1_class_va

    print("\n  " + "-" * 84)
    print(f"  classical baseline      F1 = {f1_class_va:.3f}")
    print(f"  mean-field ceiling      F1 = {f1_site_best:.3f}   "
          f"(delta vs classical = {delta_site_class:+.3f})")
    print(f"  neighbourhood ceiling   F1 = {r_sten['f1']:.3f}   "
          f"(delta vs mean-field = {delta_sten_site:+.3f})")
    print("  " + "-" * 84)

    print("\n  INTERPRETATION:")
    if delta_site_class < 0.02:
        print("  * mean-field ceiling ~= classical  ==>  NO local-bias "
              "Hamiltonian (h_i = f(local)) can beat classical.")
    else:
        print(f"  * mean-field ceiling beats classical by "
              f"{delta_site_class:+.3f}  ==>  a learned bias could help.")

    if delta_sten_site < 0.02:
        print("  * stencil ~= mean-field  ==>  ZZ/ZZZZ couplings cannot "
              "add value: no local Hamiltonian (even with couplings) "
              "will exceed the mean-field ceiling by more than noise.")
    elif delta_sten_site < 0.10:
        print(f"  * stencil beats mean-field by {delta_sten_site:+.3f}  "
              f"==>  modest room for couplings; phase 11B might pick "
              f"this up.")
    else:
        print(f"  * stencil beats mean-field by {delta_sten_site:+.3f}  "
              f"==>  SUBSTANTIAL room for couplings; a learned "
              f"Hamiltonian with ZZ/ZZZZ is worth building (phase 11B).")

    if delta_sten_class < 0.02:
        print("  * CONCLUSION: no local Hamiltonian (with or without "
              "couplings) can beat classical. Pivot to non-Hamiltonian "
              "quantum paradigms (VQC, QKE, QBM).")

    # -- per-scenario breakdown on val --
    print("\n  per-scenario (val set only):")
    print(f"  {'scenario':<18} {'n':>6} {'class':>7} "
          f"{'site':>7} {'sten':>7}  {'d_site':>7} {'d_sten':>7}")
    per_scene = {}
    p_site_best = site_res[best_site_model]["p"]
    thr_site_best = site_res[best_site_model]["thr"]
    for sc in sorted(set(Tva)):
        mask = (Tva == sc)
        if not mask.any():
            continue
        ycl = (Sva[mask] > thr_star).astype(int)
        ys  = (p_site_best[mask] > thr_site_best).astype(int)
        yn  = (r_sten["p"][mask] > r_sten["thr"]).astype(int)
        f1c = f1_score(Yva[mask], ycl, zero_division=0)
        f1s = f1_score(Yva[mask], ys,  zero_division=0)
        f1n = f1_score(Yva[mask], yn,  zero_division=0)
        per_scene[sc] = (f1c, f1s, f1n)
        print(f"  {sc:<18} {int(mask.sum()):>6d} "
              f"{f1c:>7.3f} {f1s:>7.3f} {f1n:>7.3f}  "
              f"{f1s - f1c:>+7.3f} {f1n - f1s:>+7.3f}")

    # -- permutation importance for best site model --
    print(f"\n  feature importance (permutation on {best_site_model}, val):")
    base_f1 = site_res[best_site_model]["f1"]
    best_model_obj = make_model(best_site_model, args.seed)
    best_model_obj.fit(Xtr_site, Ytr)
    imp = []
    for k, name in enumerate(FEATURE_NAMES):
        Xp = Xva_site.copy()
        rng.shuffle(Xp[:, k])
        pp = best_model_obj.predict_proba(Xp)[:, 1]
        f1p = f1_score(Yva, (pp > thr_site_best).astype(int),
                       zero_division=0)
        imp.append(base_f1 - f1p)
        print(f"    {name:<18} drop = {base_f1 - f1p:+.3f}")

    # -- save --
    out = os.path.join(RESULTS_DIR, f"upper_bound_N{args.N}_dim{args.dim}{args.label_suffix}.npz")
    np.savez_compressed(
        out,
        f1_class_val=f1_class_va,
        f1_site_lr=site_res["lr"]["f1"],
        f1_site_rf=site_res["rf"]["f1"],
        f1_site_gbt=site_res["gbt"]["f1"],
        f1_site_best=f1_site_best,
        best_site_model=best_site_model,
        f1_stencil_gbt=r_sten["f1"],
        auc_site_gbt=site_res["gbt"]["auc"],
        auc_stencil_gbt=r_sten["auc"],
        delta_site_vs_class=delta_site_class,
        delta_stencil_vs_site=delta_sten_site,
        delta_stencil_vs_class=delta_sten_class,
        thr_star_class=thr_star,
        feature_names=np.array(FEATURE_NAMES),
        permutation_importance=np.array(imp),
        per_scene_tags=np.array(list(per_scene.keys())),
        per_scene_f1_class=np.array([v[0] for v in per_scene.values()]),
        per_scene_f1_site =np.array([v[1] for v in per_scene.values()]),
        per_scene_f1_sten =np.array([v[2] for v in per_scene.values()]),
        n_train_snaps=len(tr_idx), n_val_snaps=len(va_idx),
        n_train_cells=len(Xtr_site), n_val_cells=len(Xva_site),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 11A complete.")


if __name__ == "__main__":
    main()
