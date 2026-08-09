#!/usr/bin/env python3
"""
Phase 6 - Verification: Does the v2 Hamiltonian identify hard patches?

Key question: when we rank patches by their Hamiltonian "energy"
  E_i = |H_i| + sum_j |C_ij| + sum_p |K_p|
do the top-ranked patches coincide with the L2-hard patches?

This is the direct analog of "the ground state of the quantum optimisation
problem is the hard-patch mask". If the Hamiltonian is well-posed, then
the highest-E patches (which cost the most to keep in the low-refinement
|0> state) are exactly the ones that most need refinement.

For each (scenario, Re, dim) we report:
  - F1 score of Hamiltonian energy ranking vs L2 ground truth
  - F1 score of classical indicator vs L2 ground truth
  - Top-K overlap between them
  - ROC-AUC for Hamiltonian energy as hard-patch detector
  - Same for classical score (for comparison)

Usage:
  python study/phase6_verify.py
  python study/phase6_verify.py --dim 4 --v2
"""
import argparse, os, sys
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve

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


def analyze(scenario, Re, dim, N, use_v2=True):
    suffix = "_v2" if use_v2 else ""
    coef_path = os.path.join(
        RESULTS_DIR,
        f"coefficients_{scenario}_Re{Re}_N{N}_dim{dim}{suffix}.npz")
    patch_path = os.path.join(
        RESULTS_DIR, f"patches_{scenario}_Re{Re}_N{N}_dim{dim}.npz")

    if not (os.path.exists(coef_path) and os.path.exists(patch_path)):
        return None

    patches = np.load(patch_path)
    coefs = np.load(coef_path)

    l2_full = patches["l2_errors"]          # (n_snaps_full, dim, dim)
    is_hard_full = patches["is_hard"]
    l2_thr = float(patches["l2_threshold"])

    # coefficients were computed only on every k-th snapshot; figure out
    # which snapshots are there by reading the C array shape
    # v2 uses key s0.000_E
    E_key = "s0.000_E" if use_v2 else None
    if use_v2:
        E = coefs[E_key]
    else:
        # pick the first sigma key
        for k in coefs.files:
            if k.endswith("_E"):
                E = coefs[k]
                break

    n_snaps_sub = E.shape[0]
    n_snaps_full = l2_full.shape[0]
    # reconstruct snap indices (must match analyze_one logic)
    step = max(1, n_snaps_full // 10)
    snap_indices = list(range(0, n_snaps_full, step))
    if len(snap_indices) < 3:
        snap_indices = list(range(n_snaps_full))
    snap_indices = snap_indices[:n_snaps_sub]

    l2 = l2_full[snap_indices]          # (n_sub, dim, dim)
    is_hard = is_hard_full[snap_indices]

    # flatten for classification metrics
    E_flat = E.flatten()
    l2_flat = l2.flatten()
    hard_flat = is_hard.flatten()

    # classical scores are stored in the patches file
    if "classical_scores" in patches.files:
        classical_full = patches["classical_scores"]
        classical = classical_full[snap_indices].flatten()
    else:
        classical = None

    # AUC for Hamiltonian energy ranking
    try:
        auc_E = roc_auc_score(hard_flat, E_flat)
    except ValueError:
        auc_E = np.nan

    # F1 at optimal threshold (scan)
    def best_f1(score, labels):
        if len(set(labels.astype(int))) < 2:
            return 0.0, 0.0
        thrs = np.quantile(score, np.linspace(0.05, 0.95, 40))
        best = 0.0
        best_thr = 0.0
        for t in thrs:
            pred = score > t
            tp = np.sum(pred & labels)
            fp = np.sum(pred & ~labels)
            fn = np.sum(~pred & labels)
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2*p*r / max(p + r, 1e-10)
            if f1 > best:
                best = f1
                best_thr = t
        return best, best_thr

    f1_E, thr_E = best_f1(E_flat, hard_flat)

    # top-25% overlap (Hamiltonian-hard vs L2-hard)
    # the hard rate is 25% by construction of the L2 threshold
    n_hard = int(np.sum(hard_flat))
    topK_E = np.argsort(E_flat)[::-1][:n_hard]
    topK_hard = np.where(hard_flat)[0]
    overlap_E = len(set(topK_E) & set(topK_hard))
    recall_E = overlap_E / max(n_hard, 1)

    result = {
        "scenario": scenario, "Re": Re, "dim": dim,
        "n_snaps": n_snaps_sub,
        "n_patches": len(E_flat),
        "n_hard": int(n_hard),
        "auc_E": auc_E, "f1_E": f1_E,
        "recall_E_topK": recall_E,
    }

    # classical for comparison (from the saved scores)
    if classical is not None:
        try:
            auc_c = roc_auc_score(hard_flat, classical)
        except ValueError:
            auc_c = np.nan
        f1_c, _ = best_f1(classical, hard_flat)
        topK_c = np.argsort(classical)[::-1][:n_hard]
        overlap_c = len(set(topK_c) & set(topK_hard))
        recall_c = overlap_c / max(n_hard, 1)
        result.update({
            "auc_c": auc_c, "f1_c": f1_c,
            "recall_c_topK": recall_c,
        })

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: verify v2 Hamiltonian finds hard patches")
    parser.add_argument("--scenario", nargs="+", default=SCENARIOS)
    parser.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    parser.add_argument("--dim", nargs="+", type=int, default=[4])
    parser.add_argument("--N", type=int, default=DNS_N)
    parser.add_argument("--v1", action="store_true",
                        help="Use v1 coefficients file instead of v2")
    args = parser.parse_args()

    use_v2 = not args.v1
    version = "v2 (parameter-free)" if use_v2 else "v1 (trained)"

    print("=" * 78)
    print(f"  Phase 6: Hard-patch detection via {version} Hamiltonian energy")
    print("=" * 78)
    print()
    print("  Hypothesis: the patch-level Hamiltonian energy")
    print("    E_patch = <|H_i|> + <|C_ij|> + <|K_p|>")
    print("  ranks patches such that high-E patches = hard-to-simulate patches.")
    print()
    print(f"  {'Scenario':<18} {'Re':>5} {'dim':>4} {'N_hard':>7} "
          f"{'AUC(E)':>7} {'F1(E)':>7} {'Recall@K(E)':>12} "
          f"{'AUC(cl)':>8} {'F1(cl)':>7} {'Recall@K(cl)':>13}")
    print("  " + "-" * 76)

    rows = []
    for sc in args.scenario:
        for re in args.re:
            for dim in args.dim:
                r = analyze(sc, re, dim, args.N, use_v2=use_v2)
                if r is None:
                    continue
                rows.append(r)
                print(f"  {sc:<18} {re:>5} {dim:>4} {r['n_hard']:>7} "
                      f"{r['auc_E']:>7.3f} {r['f1_E']:>7.3f} "
                      f"{r['recall_E_topK']:>12.3f} "
                      f"{r.get('auc_c', np.nan):>8.3f} "
                      f"{r.get('f1_c', np.nan):>7.3f} "
                      f"{r.get('recall_c_topK', np.nan):>13.3f}")

    if not rows:
        print("  No coefficient files found. Run Phase 3 first.")
        return

    print()
    print("=" * 78)
    print("  INTERPRETATION")
    print("=" * 78)

    mean_auc_E = np.nanmean([r['auc_E'] for r in rows])
    mean_f1_E = np.nanmean([r['f1_E'] for r in rows])
    mean_recall_E = np.nanmean([r['recall_E_topK'] for r in rows])

    auc_c = [r.get('auc_c', np.nan) for r in rows]
    f1_c = [r.get('f1_c', np.nan) for r in rows]
    recall_c = [r.get('recall_c_topK', np.nan) for r in rows]
    mean_auc_c = np.nanmean(auc_c)
    mean_f1_c = np.nanmean(f1_c)
    mean_recall_c = np.nanmean(recall_c)

    print(f"\n  Hamiltonian energy:  AUC = {mean_auc_E:.3f}  "
          f"F1 = {mean_f1_E:.3f}  Recall@K = {mean_recall_E:.3f}")
    print(f"  Classical score:     AUC = {mean_auc_c:.3f}  "
          f"F1 = {mean_f1_c:.3f}  Recall@K = {mean_recall_c:.3f}")

    print()
    if mean_auc_E > 0.5 + 0.05:
        print(f"  PASS: Hamiltonian energy ranks hard patches above chance "
              f"(AUC={mean_auc_E:.3f} > 0.5).")
        print(f"        Minimizing H (i.e. picking low-E states) does identify "
              f"non-hard patches.")
        print(f"        Maximizing E identifies the hard patches.")
    else:
        print(f"  FAIL: Hamiltonian energy does not rank hard patches "
              f"(AUC={mean_auc_E:.3f}).")

    if mean_f1_E > mean_f1_c + 0.02:
        print(f"\n  PASS: Hamiltonian F1 ({mean_f1_E:.3f}) > "
              f"Classical F1 ({mean_f1_c:.3f}).")
        print(f"        The quantum Hamiltonian adds discrimination power beyond "
              f"the classical score.")
    elif mean_f1_E > mean_f1_c - 0.02:
        print(f"\n  TIE:  Hamiltonian F1 ({mean_f1_E:.3f}) ~= "
              f"Classical F1 ({mean_f1_c:.3f}).")
        print(f"        The Hamiltonian matches the classical baseline.")
    else:
        print(f"\n  WARN: Hamiltonian F1 ({mean_f1_E:.3f}) < "
              f"Classical F1 ({mean_f1_c:.3f}).")


if __name__ == "__main__":
    main()
