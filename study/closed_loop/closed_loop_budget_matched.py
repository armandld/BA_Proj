#!/usr/bin/env python3
"""Compare Q-HAS with classical AMR at the same realised patch budget.

The classical threshold is bisected until its ``patch_ratio`` matches the
Q-HAS fold result within tolerance. Non-finite evaluations are rejected and
an unconverged search is saved as inconclusive with a non-zero exit status.
"""
import argparse, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------

from closed_loop_campaign import (
    _atomic_json, _load_v1_training_module, fold_scenarios, run_arm,
)
import provenance


def bisect_threshold_for_budget(T, key, cfg, dns_held, base_hp, target_patch,
                                lo=0.05, hi=0.80, max_iter=5, tol=0.02,
                                lambda_cost=None, verbose=False, seed=0):
    """Cherche le seuil classique reproduisant `target_patch`.

    `patch_ratio` decroit avec le seuil (seuil haut -> on raffine moins),
    la bissection exploite cette monotonie. Chaque evaluation est UN run
    complet du bras classique sur la classe tenue.

    Retourne (best, trace) ou best est l'evaluation la plus proche de la
    cible et trace la liste des evaluations.
    """
    if not np.isfinite(target_patch):
        raise ValueError("target_patch must be finite")
    if not (0 <= target_patch <= 1):
        raise ValueError("target_patch must be in [0, 1]")
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError("lo and hi must be finite with lo < hi")
    if max_iter < 0 or tol < 0:
        raise ValueError("max_iter and tol must be non-negative")

    trace = []

    def _finite_evaluations():
        return [
            row for row in trace
            if np.isfinite(row["patch_ratio"])
            and np.isfinite(row["phys_score"])
        ]

    def _best_finite():
        finite = _finite_evaluations()
        if not finite:
            raise RuntimeError(
                "budget matching produced no finite evaluation")
        return min(
            finite,
            key=lambda row: abs(row["patch_ratio"] - target_patch),
        )

    def _eval(thr):
        hp = dict(base_hp)
        hp["threshold_amr"] = float(thr)
        r = run_arm(T, key, cfg, dns_held, hp, True,
                    lambda_cost=lambda_cost, verbose=verbose, seed=seed)
        completed = bool(r.get("completed", True))
        rec = dict(threshold=float(thr),
                   patch_ratio=(float(r.get("patch_ratio", np.nan))
                                if completed else np.nan),
                   phys_score=(float(r.get("phys_score", np.nan))
                               if completed else np.nan),
                   combined=(float(r.get("combined", np.nan))
                             if completed else np.nan),
                   wall_s=float(r.get("wall_s", np.nan)),
                   completed=completed,
                   abort=r.get("abort"))
        trace.append(rec)
        print(f"    thr={thr:.4f} -> patch={rec['patch_ratio']:.4f} "
              f"phys={rec['phys_score']:.4f} "
              f"(target patch {target_patch:.4f})", flush=True)
        return rec

    r_lo, r_hi = _eval(lo), _eval(hi)
    endpoint_ratios = [
        row["patch_ratio"] for row in (r_lo, r_hi)
        if np.isfinite(row["patch_ratio"])
    ]
    if (len(endpoint_ratios) < 2
            or not (min(endpoint_ratios) - tol <= target_patch
                    <= max(endpoint_ratios) + tol)):
        print("    [warn] target budget outside the bracket; returning the "
              "closest evaluation", flush=True)
    for _ in range(max_iter):
        best = _best_finite()
        if abs(best["patch_ratio"] - target_patch) <= tol:
            break
        mid = 0.5 * (lo + hi)
        r_mid = _eval(mid)
        if not (np.isfinite(r_mid["patch_ratio"])
                and np.isfinite(r_mid["phys_score"])):
            raise RuntimeError(
                f"non-finite budget evaluation at threshold {mid:.6g}; "
                "bisection direction is undefined")
        # patch_ratio decroissant en thr
        if r_mid["patch_ratio"] > target_patch:
            lo = mid
        else:
            hi = mid
    best = _best_finite()
    return best, trace


def budget_match_reading(delta_phys, converged):
    """Describe the observed direction only after budget convergence."""
    if not converged or not np.isfinite(delta_phys):
        return ("INCONCLUSIVE: the classical arm did not reach the target "
                "budget within tolerance.")
    if delta_phys < 0:
        return "Q-HAS has lower observed error at the matched budget."
    if delta_phys > 0:
        return "classical AMR has lower observed error at the matched budget."
    return "the observed errors are equal at the matched budget."


def main():
    p = argparse.ArgumentParser(
        description="V4 Task 15b: budget-matched classical comparison")
    from config import RESULTS_DIR

    p.add_argument("--fold", required=True, help="cle du fold (ex: ot)")
    p.add_argument("--prefix", default="t15_level3")
    p.add_argument("--max-iter", type=int, default=8)
    p.add_argument("--tol", type=float, default=0.02,
                   help="tolerance sur le patch_ratio cible")
    p.add_argument("--lambda-cost", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    prov = provenance.start()

    T = _load_v1_training_module()
    scen = dict(fold_scenarios(T))
    if args.fold not in scen:
        raise SystemExit(f"unknown fold {args.fold} (balayage vide)")
    cfg = scen[args.fold]

    fold_path = os.path.join(RESULTS_DIR,
                             f"{args.prefix}_fold_{args.fold}.json")
    if not os.path.exists(fold_path):
        raise SystemExit(
            f"missing {fold_path}; run the closed-loop fold first")
    with open(fold_path, encoding="utf-8") as stream:
        rec = json.load(stream)
    if not rec.get("campaign_contract_sha256"):
        raise RuntimeError(
            "fold artifact has no campaign contract; refusing an "
            "unverifiable budget match")
    target = float(rec["qhas"]["patch_ratio"])

    print("=" * 88)
    print(f"  V4 Task 15b: budget-matched comparison, fold {args.fold}")
    print(f"  Q-HAS realised patch_ratio = {target:.4f}  "
          f"(phys {rec['qhas']['phys_score']:.4f})")
    print(f"  tuned classical            = "
          f"{rec['classical']['patch_ratio']:.4f}  "
          f"(phys {rec['classical']['phys_score']:.4f}, thr "
          f"{rec['classical_params']['threshold_amr']:.4f})")
    print("  searching the classical threshold that matches the Q-HAS budget")
    print("=" * 88, flush=True)

    dns_held = T._precompute_dns_for([(args.fold, cfg)],
                                     label=f"held/{args.fold}")
    t0 = time.time()
    best, trace = bisect_threshold_for_budget(
        T, args.fold, cfg, dns_held, rec["hyperparams"], target,
        max_iter=args.max_iter, tol=args.tol,
        lambda_cost=args.lambda_cost, seed=args.seed)
    wall = time.time() - t0

    d_phys = rec["qhas"]["phys_score"] - best["phys_score"]
    mismatch = best["patch_ratio"] - target
    converged = bool(np.isfinite(d_phys) and abs(mismatch) <= args.tol)
    print("\n  " + "=" * 84)
    print(f"  budget-matched classical: thr={best['threshold']:.4f}  "
          f"patch={best['patch_ratio']:.4f}  phys={best['phys_score']:.4f}")
    print(f"  Q-HAS                   : "
          f"patch={target:.4f}  phys={rec['qhas']['phys_score']:.4f}")
    print(f"  delta phys at matched budget = {d_phys:+.4f} "
          f"(negative favours Q-HAS)")
    print(f"  budget mismatch remaining    = "
          f"{best['patch_ratio'] - target:+.4f}")
    print("\n  READING: " + budget_match_reading(d_phys, converged))

    output_prefix = ("t15b_budget_matched" if args.prefix == "t15_level3"
                     else f"{args.prefix}_budget_matched")
    out = os.path.join(RESULTS_DIR, f"{output_prefix}_{args.fold}.json")
    payload = dict(
        artifact="closed_loop_budget_match", schema=2,
        fold=args.fold, target_patch=target,
        qhas=rec["qhas"], tuned_classical=rec["classical"],
        matched_classical=best, trace=trace,
        delta_phys_matched=d_phys,
        budget_mismatch=mismatch,
        converged=converged,
        parent_campaign_contract_sha256=rec["campaign_contract_sha256"],
        wall_s=wall, cli_args=vars(args),
    )
    payload.update(provenance.finish(prov))
    _atomic_json(out, payload)
    print(f"\n  saved: {os.path.basename(out)}")
    if not converged:
        raise SystemExit(2)
    print("\nV4 Task 15b complete.")


if __name__ == "__main__":
    main()
