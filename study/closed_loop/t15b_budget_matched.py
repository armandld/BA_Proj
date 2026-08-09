#!/usr/bin/env python3
"""
V4 Task 15b - Comparaison a BUDGET APPARIE (audit, Priorite 0).

MOTIVATION. Le fold Level-3 `ot` donne Q-HAS meilleur en fidelite
(phys 0.194 contre 0.485) mais a 2.1x le cout (patch 0.680 contre 0.324).
Les deux bras ne sont donc pas au meme point de la frontiere erreur-cout, et
l'ecart de fidelite n'est pas interpretable tel quel. La cause est une
asymetrie de reglage heritee du module V1 : dans
`make_composite_objective`, le seuil du bras QAOA est CODE EN DUR
(`HyperParams["threshold_amr"] = 0.1496`, jamais propose a Optuna), alors
que `make_classical_composite_objective` l'optimise librement sur [0.05,
0.8] et retient ici 0.4616.

PROTOCOLE. On fixe le budget et on compare la fidelite :
  1. lire le `patch_ratio` realise par Q-HAS sur la classe tenue ;
  2. chercher par bissection le seuil classique qui reproduit ce meme
     `patch_ratio` (a `--tol` pres) ;
  3. comparer `phys_score` A COUT EGAL.
Le bras classique est le seul re-execute : la trajectoire Q-HAS, la trace
DNS, le hot start et le budget hybride sont ceux du fold deja calcule.

LECTURE PRE-SPECIFIEE.
  - si, a budget egal, la fidelite classique rejoint celle de Q-HAS, le
    gain observe au fold Level-3 est un simple deplacement le long de la
    frontiere de Pareto, pas une amelioration de la regle de decision ;
  - si Q-HAS conserve un avantage de fidelite a cout egal, le gain est
    attribuable a la regle de decision et doit etre rapporte comme tel.

Sortie : results/t15b_budget_matched_{fold}.json
Usage :
  python study/v4/t15b_budget_matched.py --fold ot --max-iter 5
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

from t1_feature_selection import git_commit_hash
from t15_level3_closed_loop import (
    _load_v1_training_module, fold_scenarios, run_arm,
)


def bisect_threshold_for_budget(T, key, cfg, dns_held, base_hp, target_patch,
                                lo=0.05, hi=0.80, max_iter=5, tol=0.02,
                                lambda_cost=None, verbose=False):
    """Cherche le seuil classique reproduisant `target_patch`.

    `patch_ratio` decroit avec le seuil (seuil haut -> on raffine moins),
    la bissection exploite cette monotonie. Chaque evaluation est UN run
    complet du bras classique sur la classe tenue.

    Retourne (best, trace) ou best est l'evaluation la plus proche de la
    cible et trace la liste des evaluations.
    """
    trace = []

    def _eval(thr):
        hp = dict(base_hp)
        hp["threshold_amr"] = float(thr)
        r = run_arm(T, key, cfg, dns_held, hp, True,
                    lambda_cost=lambda_cost, verbose=verbose)
        rec = dict(threshold=float(thr),
                   patch_ratio=float(r.get("patch_ratio", np.nan)),
                   phys_score=float(r.get("phys_score", np.nan)),
                   combined=float(r.get("combined", np.nan)),
                   wall_s=float(r.get("wall_s", np.nan)))
        trace.append(rec)
        print(f"    thr={thr:.4f} -> patch={rec['patch_ratio']:.4f} "
              f"phys={rec['phys_score']:.4f} "
              f"(target patch {target_patch:.4f})", flush=True)
        return rec

    r_lo, r_hi = _eval(lo), _eval(hi)
    if not (min(r_lo["patch_ratio"], r_hi["patch_ratio"]) - tol
            <= target_patch
            <= max(r_lo["patch_ratio"], r_hi["patch_ratio"]) + tol):
        print("    [warn] target budget outside the bracket; returning the "
              "closest evaluation", flush=True)
    for _ in range(max_iter):
        best = min(trace, key=lambda r: abs(r["patch_ratio"] - target_patch))
        if abs(best["patch_ratio"] - target_patch) <= tol:
            break
        mid = 0.5 * (lo + hi)
        r_mid = _eval(mid)
        # patch_ratio decroissant en thr
        if r_mid["patch_ratio"] > target_patch:
            lo = mid
        else:
            hi = mid
    best = min(trace, key=lambda r: abs(r["patch_ratio"] - target_patch))
    return best, trace


def main():
    p = argparse.ArgumentParser(
        description="V4 Task 15b: budget-matched classical comparison")
    from config import RESULTS_DIR

    p.add_argument("--fold", required=True, help="cle du fold (ex: ot)")
    p.add_argument("--prefix", default="t15_level3")
    p.add_argument("--max-iter", type=int, default=5)
    p.add_argument("--tol", type=float, default=0.02,
                   help="tolerance sur le patch_ratio cible")
    p.add_argument("--lambda-cost", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    fold_path = os.path.join(RESULTS_DIR,
                             f"{args.prefix}_fold_{args.fold}.json")
    if not os.path.exists(fold_path):
        print(f"missing {fold_path}; run t15 for this fold first.")
        return
    rec = json.load(open(fold_path))
    target = float(rec["qhas"]["patch_ratio"])

    T = _load_v1_training_module()
    scen = dict(fold_scenarios(T, warn=False))
    if args.fold not in scen:
        print(f"unknown fold {args.fold}"); return
    cfg = scen[args.fold]

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
        lambda_cost=args.lambda_cost)
    wall = time.time() - t0

    d_phys = rec["qhas"]["phys_score"] - best["phys_score"]
    print("\n  " + "=" * 84)
    print(f"  budget-matched classical: thr={best['threshold']:.4f}  "
          f"patch={best['patch_ratio']:.4f}  phys={best['phys_score']:.4f}")
    print(f"  Q-HAS                   : "
          f"patch={target:.4f}  phys={rec['qhas']['phys_score']:.4f}")
    print(f"  delta phys at matched budget = {d_phys:+.4f} "
          f"(negative favours Q-HAS)")
    print(f"  budget mismatch remaining    = "
          f"{best['patch_ratio'] - target:+.4f}")
    print("\n  READING: " + (
        "the Level-3 fidelity gap survives at equal compute; it is "
        "attributable to the decision rule."
        if d_phys < -0.01 else
        "at equal compute the classical arm recovers the fidelity; the "
        "Level-3 gap was a move along the error-cost frontier, not a "
        "better decision rule."))

    out = os.path.join(RESULTS_DIR,
                       f"t15b_budget_matched_{args.fold}.json")
    json.dump(dict(fold=args.fold, target_patch=target,
                   qhas=rec["qhas"], tuned_classical=rec["classical"],
                   matched_classical=best, trace=trace,
                   delta_phys_matched=d_phys, wall_s=wall,
                   git_hash=git_commit_hash(), cli_args=vars(args)),
              open(out, "w"), indent=1, default=float)
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nV4 Task 15b complete.")


if __name__ == "__main__":
    main()
