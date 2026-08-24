#!/usr/bin/env python3
"""
Phase 12 - Quantum classifier baselines (VQC and QKE).

Phase 11 ruled out QAOA (the optimiser paradigm): the optimal
Hamiltonian is separable, so QAOA offers no advantage over
site-wise argmin. This phase probes the *other* family of quantum
ML paradigms -- quantum classifiers that bypass the Hamiltonian
entirely.

Two models are tested on the same 9 per-cell features as phase 11
(reduced to d_q features via PCA for circuit tractability):

  (A) VQC = Variational Quantum Classifier
      ZZFeatureMap (feature embedding via entanglement) +
      RealAmplitudes (variational ansatz), trained with COBYLA
      against cross-entropy loss. Output is p(refine | features).

  (B) QKE = Quantum Kernel Estimation + classical SVM
      Fidelity kernel K(x, y) = |<phi(x) | phi(y)>|^2 computed
      from ZZFeatureMap states; fed to an SVC.

Both are compared against classical baselines on the same
d_q-dimensional PCA features (LR, GBT) so the comparison isolates
the effect of the quantum kernel / circuit.

If VQC or QKE surpass the classical F1 ceiling from phase 11,
*that* is a quantum advantage outside the Hamiltonian paradigm --
the kind of result that keeps a quantum angle viable in a paper
whose headline is "QAOA does not help here".

Output: results/vqc_N{N}_dim{D}.npz

Usage:
  python study/phase12_vqc.py --dim 4 --n-train 1500 --n-val 500

Requires: qiskit-machine-learning (already in environment.yaml).
"""
import argparse, os, sys, time, warnings
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
    FEATURE_NAMES, N_FEATS, build_dataset, make_model, fit_eval,
    best_threshold_f1,
)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import SVC


# -------------------------------------------------------------------
# Stratified subsample
# -------------------------------------------------------------------

def stratified_subsample(X, Y, n, rng):
    """Return (n,) indices stratified by Y (pos/neg)."""
    pos = np.where(Y == 1)[0]; neg = np.where(Y == 0)[0]
    n_pos = min(len(pos), n // 2); n_neg = min(len(neg), n - n_pos)
    take_p = rng.choice(pos, n_pos, replace=False)
    take_n = rng.choice(neg, n_neg, replace=False)
    sel = np.concatenate([take_p, take_n])
    rng.shuffle(sel)
    return sel


# -------------------------------------------------------------------
# VQC
# -------------------------------------------------------------------

def run_vqc(Xtr, Ytr, Xva, Yva, d_q, reps_fm, reps_ansatz,
            maxiter, seed):
    """Train a VQC and return (f1, auc, thr, fit_s, model)."""
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.algorithms.classifiers import VQC
    from qiskit.primitives import StatevectorSampler

    # Scale to [-pi, pi] for ZZFeatureMap numerical stability
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    Xtr_s = (Xtr - lo) / span * np.pi - np.pi / 2
    Xva_s = (Xva - lo) / span * np.pi - np.pi / 2
    Xva_s = np.clip(Xva_s, -np.pi, np.pi)

    fm  = ZZFeatureMap(feature_dimension=d_q, reps=reps_fm,
                        entanglement="linear")
    ans = RealAmplitudes(num_qubits=d_q, reps=reps_ansatz,
                         entanglement="linear")

    from qiskit_machine_learning.optimizers import COBYLA
    opt = COBYLA(maxiter=maxiter)

    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(-np.pi, np.pi, ans.num_parameters)

    vqc = VQC(feature_map=fm, ansatz=ans, optimizer=opt,
              sampler=StatevectorSampler(seed=seed),
              initial_point=initial_point)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vqc.fit(Xtr_s, Ytr)
    fit_s = time.time() - t0

    # predict_proba via the score marginal
    def _proba(X):
        try:
            return vqc.predict_proba(X)[:, 1]
        except Exception:
            return vqc.predict(X).astype(float)

    p_va = _proba(Xva_s)
    p_tr = _proba(Xtr_s)
    # Select the decision threshold on training data only.
    grid = np.linspace(0.05, 0.95, 91)
    thr, _ = best_threshold_f1(p_tr, Ytr, grid=grid)
    f1 = float(f1_score(Yva, (p_va > thr).astype(int), zero_division=0))
    _, f1_thr_on_val = best_threshold_f1(p_va, Yva, grid=grid)
    try:
        auc = roc_auc_score(Yva, p_va)
    except Exception:
        auc = float("nan")
    return dict(f1=f1, f1_thr_on_val=f1_thr_on_val, auc=auc, thr=thr,
                fit_s=fit_s, p_va=p_va)


# -------------------------------------------------------------------
# QKE
# -------------------------------------------------------------------

def run_qke(Xtr, Ytr, Xva, Yva, d_q, reps_fm, seed):
    """Quantum kernel + SVC."""
    from qiskit.circuit.library import ZZFeatureMap
    try:
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
    except ImportError:
        from qiskit_machine_learning.kernels import QuantumKernel as FidelityQuantumKernel

    # Scale to [-pi, pi]
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    Xtr_s = (Xtr - lo) / span * np.pi - np.pi / 2
    Xva_s = (Xva - lo) / span * np.pi - np.pi / 2
    Xva_s = np.clip(Xva_s, -np.pi, np.pi)

    fm = ZZFeatureMap(feature_dimension=d_q, reps=reps_fm,
                      entanglement="linear")
    qk = FidelityQuantumKernel(feature_map=fm)

    t0 = time.time()
    K_tr = qk.evaluate(x_vec=Xtr_s)
    K_va = qk.evaluate(x_vec=Xva_s, y_vec=Xtr_s)
    svc = SVC(kernel="precomputed", probability=True,
              class_weight="balanced", random_state=seed).fit(K_tr, Ytr)
    fit_s = time.time() - t0

    p_va = svc.predict_proba(K_va)[:, 1]
    p_tr = svc.predict_proba(K_tr)[:, 1]
    # D-81 : le seuil se choisit sur le TRAIN, comme `fit_eval` le fait pour
    # les bras classiques auxquels ce F1 est compare. Il etait choisi sur
    # `(p_va, Yva)` — les labels de validation — ce qui donnait au bras
    # quantique un avantage que son concurrent n'avait pas.
    grid = np.linspace(0.05, 0.95, 91)
    thr, _ = best_threshold_f1(p_tr, Ytr, grid=grid)
    f1 = float(f1_score(Yva, (p_va > thr).astype(int), zero_division=0))
    # l'ancien nombre, garde pour que le biais reste mesurable
    _, f1_thr_on_val = best_threshold_f1(p_va, Yva, grid=grid)
    try:
        auc = roc_auc_score(Yva, p_va)
    except Exception:
        auc = float("nan")
    return dict(f1=f1, f1_thr_on_val=f1_thr_on_val, auc=auc, thr=thr,
                fit_s=fit_s, p_va=p_va)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Phase 12: VQC + QKE quantum classifier baselines")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--n-train", type=int, default=1500,
                   help="VQC/QKE train subsample (stratified)")
    p.add_argument("--n-val", type=int, default=500,
                   help="VQC/QKE val subsample (stratified)")
    p.add_argument("--d-q", type=int, default=4,
                   help="# qubits (= PCA-reduced feature dim)")
    p.add_argument("--reps-fm", type=int, default=2,
                   help="ZZFeatureMap repetitions")
    p.add_argument("--reps-ansatz", type=int, default=2,
                   help="RealAmplitudes repetitions")
    p.add_argument("--maxiter", type=int, default=80,
                   help="COBYLA maxiter for VQC")
    p.add_argument("--skip-vqc", action="store_true")
    p.add_argument("--skip-qke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  Phase 12: VQC + QKE quantum classifier baselines")
    print(f"  dim={args.dim}  N={args.N}  d_q={args.d_q}  "
          f"n_train={args.n_train}  n_val={args.n_val}")
    print("=" * 88)
    print()

    # ---- data ----
    configs = []
    for sc in args.scenario:
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(RESULTS_DIR,
                              f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if os.path.exists(dp) and os.path.exists(pp):
                configs.append((sc, re, dp, pp))
    if not configs:
        # D-56 : ce garde imprimait « no input. » et rendait la main avec le
        # code 0, sans ecrire d'artefact — donc en laissant en place celui de
        # la campagne precedente. Une campagne qui n'avait rien mesure etait
        # indiscernable d'une campagne reussie. Onze autres modules de
        # `study/` levaient deja ici ; ceux-ci ne le faisaient pas.
        raise RuntimeError(
            "balayage vide : aucune configurations n'a d'artefact d'entree pour les "
            "arguments donnes. Le script sortait ici avec le code 0 et sans "
            "artefact, donc sans se distinguer d'une campagne reussie.")

    X_site, _, Y_snap, S_snap, tags = build_dataset(
        configs, args.dim, args.max_snaps)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X_site))
    n_tr_sn = max(1, int(args.train_frac * len(X_site)))
    tr_sn, va_sn = perm[:n_tr_sn], perm[n_tr_sn:]

    def stack(pool, idxs):
        return np.concatenate([pool[i] for i in idxs], axis=0)
    Xtr_full = stack(X_site, tr_sn); Ytr_full = stack(Y_snap, tr_sn)
    Xva_full = stack(X_site, va_sn); Yva_full = stack(Y_snap, va_sn)
    Sva_full = stack(S_snap, va_sn); Str_full = stack(S_snap, tr_sn)

    # subsample stratified
    tr_sel = stratified_subsample(Xtr_full, Ytr_full, args.n_train, rng)
    va_sel = stratified_subsample(Xva_full, Yva_full, args.n_val, rng)
    Xtr = Xtr_full[tr_sel]; Ytr = Ytr_full[tr_sel]
    Xva = Xva_full[va_sel]; Yva = Yva_full[va_sel]
    print(f"  subsampled: train={len(Xtr)}  val={len(Xva)}  "
          f"pos_tr={Ytr.mean():.3f}  pos_va={Yva.mean():.3f}")

    # ---- PCA -> d_q ----
    scaler = StandardScaler().fit(Xtr)
    Ztr_full = scaler.transform(Xtr); Zva_full = scaler.transform(Xva)
    pca = PCA(n_components=args.d_q, random_state=args.seed).fit(Ztr_full)
    Ptr = pca.transform(Ztr_full); Pva = pca.transform(Zva_full)
    print(f"  PCA {N_FEATS}->{args.d_q}  "
          f"explained variance ratio: "
          f"{pca.explained_variance_ratio_.sum():.3f}")

    # ---- classical baselines on PCA features ----
    print("\n  [classical on PCA features]")
    results = {}
    for name in ("lr", "gbt"):
        r = fit_eval(make_model(name, args.seed), Ptr, Ytr, Pva, Yva)
        results[f"class_{name}_pca"] = r
        print(f"    {name:<8}  F1={r['f1']:.3f}  AUC={r['auc']:.3f}  "
              f"fit={r['fit_s']:.1f}s")

    # classical indicator baseline on val subsample
    # re-use: best threshold on FULL train, apply on subsampled val
    thr_cls, _ = best_threshold_f1(Str_full, Ytr_full)
    # match va_sel indices back to Sva_full
    f1_cls_sub = f1_score(Yva, (Sva_full[va_sel] > thr_cls).astype(int),
                           zero_division=0)
    print(f"    classical-score (thr*={thr_cls:.3f}): "
          f"F1={f1_cls_sub:.3f}")

    # ---- QKE ----
    if not args.skip_qke:
        print("\n  [QKE] quantum kernel + SVC")
        try:
            r_qke = run_qke(Ptr, Ytr, Pva, Yva,
                             d_q=args.d_q, reps_fm=args.reps_fm,
                             seed=args.seed)
            results["qke"] = r_qke
            print(f"    qke     F1={r_qke['f1']:.3f}  "
                  f"AUC={r_qke['auc']:.3f}  fit={r_qke['fit_s']:.1f}s"
                  f"   [seuil sur val, biaise, D-81 : "
                  f"{r_qke['f1_thr_on_val']:.3f}]")
        except Exception as e:
            print(f"    QKE failed: {e!r}")

    # ---- VQC ----
    if not args.skip_vqc:
        print("\n  [VQC] variational quantum classifier")
        try:
            r_vqc = run_vqc(Ptr, Ytr, Pva, Yva,
                             d_q=args.d_q,
                             reps_fm=args.reps_fm,
                             reps_ansatz=args.reps_ansatz,
                             maxiter=args.maxiter, seed=args.seed)
            results["vqc"] = r_vqc
            print(f"    vqc     F1={r_vqc['f1']:.3f}  "
                  f"AUC={r_vqc['auc']:.3f}  fit={r_vqc['fit_s']:.1f}s"
                  f"   [seuil sur val, biaise, D-81 : "
                  f"{r_vqc['f1_thr_on_val']:.3f}]")
        except Exception as e:
            print(f"    VQC failed: {e!r}")

    # ---- summary & verdict ----
    print("\n  " + "-" * 84)
    print(f"  classical score only  F1 = {f1_cls_sub:.3f}  (AMR baseline)")
    print(f"  LR (PCA)              F1 = {results['class_lr_pca']['f1']:.3f}")
    print(f"  GBT (PCA)             F1 = {results['class_gbt_pca']['f1']:.3f}")
    if "qke" in results:
        print(f"  QKE                   F1 = {results['qke']['f1']:.3f}")
    if "vqc" in results:
        print(f"  VQC                   F1 = {results['vqc']['f1']:.3f}")
    print("  " + "-" * 84)

    # verdict: does quantum beat classical on the same features?
    f1_cl_best = max(results["class_lr_pca"]["f1"],
                      results["class_gbt_pca"]["f1"])
    f1_q_best  = max(
        (results[k]["f1"] for k in ("qke", "vqc") if k in results),
        default=-1.0)
    quantum_arms = [k for k in ("qke", "vqc") if k in results]
    if len(quantum_arms) < 2:
        print(f"\n  ATTENTION : bras quantiques effectivement executes = "
              f"{quantum_arms or 'aucun'} — le verdict ci-dessous ne porte "
              f"que sur ceux-la (D-81).")
    if f1_q_best < 0:
        print("\n  (no quantum model ran; skipping verdict)")
    else:
        delta = f1_q_best - f1_cl_best
        if delta < -0.02:
            print(f"\n  VERDICT: best quantum model UNDERPERFORMS best "
                  f"classical by {delta:+.3f}  ==>  no advantage outside "
                  f"Hamiltonian paradigm either.")
        elif delta < 0.02:
            print(f"\n  VERDICT: quantum ~= classical (delta = {delta:+.3f}) "
                  f" ==>  no clear advantage; publishable as 'both paradigms "
                  f"tested, neither helps'.")
        else:
            print(f"\n  VERDICT: best quantum beats best classical by "
                  f"{delta:+.3f}  ==>  quantum advantage found in the "
                  f"CLASSIFIER paradigm (not the optimiser). Worth a full "
                  f"chapter.")

    # ---- save ----
    out = os.path.join(RESULTS_DIR, f"vqc_N{args.N}_dim{args.dim}.npz")
    save_kw = dict(
        d_q=args.d_q, n_train=len(Xtr), n_val=len(Xva),
        pca_explained_var=pca.explained_variance_ratio_,
        f1_classical_score=f1_cls_sub,
        f1_lr_pca =results["class_lr_pca"]["f1"],
        f1_gbt_pca=results["class_gbt_pca"]["f1"],
    )
    # D-81 : `f1_*_thr_on_val` est l'ancien nombre — seuil choisi sur les
    # labels de validation. Il est conserve pour que l'ecart reste
    # mesurable, jamais compare aux bras classiques.
    if "qke" in results:
        save_kw["f1_qke"]  = results["qke"]["f1"]
        save_kw["auc_qke"] = results["qke"]["auc"]
        save_kw["f1_qke_thr_on_val"] = results["qke"]["f1_thr_on_val"]
    if "vqc" in results:
        save_kw["f1_vqc"]  = results["vqc"]["f1"]
        save_kw["auc_vqc"] = results["vqc"]["auc"]
        save_kw["f1_vqc_thr_on_val"] = results["vqc"]["f1_thr_on_val"]
    # quels bras ont reellement tourne : un bras qui a leve laisse le verdict
    # se calculer sur les autres sans le dire (mesure : VQC est tombe sur
    # `qiskit.algorithms` absent, et le verdict s'est imprime quand meme).
    save_kw["arms_that_ran"] = np.array(sorted(results))
    np.savez_compressed(out, **save_kw)
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 12 complete.")


if __name__ == "__main__":
    main()
