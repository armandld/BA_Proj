#!/usr/bin/env python3
"""
V4 Task 22d - Le plancher atteignable sur la condition inedite.

LE CONFONDANT QUE CETTE TACHE MESURE. T22c constate que le rapport entre
bras se resserre sur la condition inedite (`tearing` : 3.27x -> 1.39x,
|z| = 3.45). Deux lectures incompatibles :

  (a) la regle QAOA transfere mieux ; ou
  (b) la condition inedite est simplement PLUS FACILE, les deux bras
      butent sur un plancher commun (discretisation, budget maximal), et
      deux methodes quelconques y convergeraient — le resserrement ne dit
      alors rien de la qualite des decisions.

On ne peut pas trancher sans connaitre ce plancher. Il se mesure en UNE
execution par fold : le bras classique a seuil tres bas raffine presque
tout le domaine et donne l'erreur residuelle du solveur a cette resolution,
c'est-a-dire le meilleur resultat qu'une regle de decision, quelle qu'elle
soit, peut esperer.

ATTENTION — CE « PLANCHER » N'EST PAS UNE BORNE INFERIEURE. C'est l'erreur
obtenue en raffinant presque tout, donc une ESTIMATION de l'optimum
atteignable, pas un optimum certifie. La mesure sur `rotor` le prouve : en
condition inedite le bras classique regle atteint 0.98x cette valeur, il la
BAT. Raffiner davantage n'ameliore donc pas toujours l'erreur. Les rapports
ci-dessous se lisent « par rapport au raffinement quasi-complet », et non
« par rapport au mieux possible ».

LECTURE. Pour chaque condition on rapporte l'ecart de chaque bras AU
PLANCHER, phys / phys_floor :
  - si les deux bras sont a ~1.0x du plancher sur la condition inedite, le
    resserrement observe est l'effet (b) et ne mesure aucun transfert ;
  - s'ils en restent nettement au-dessus, ils ont encore de la marge et
    leur ecart reflete bien leurs regles de decision.

Le seuil plancher (0.05) est le point le plus bas deja utilise par les
bissections de t15b, donc ni choisi ni ajuste ici.

Sortie : results/t22d_unseen_floor_{fold}.json
Usage :
  python study/v4/t22d_unseen_floor.py --fold tearing
"""
import argparse, contextlib, io, json, os, sys, time

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

from h2b_feature_selection import git_commit_hash
from closed_loop_campaign import (_load_v1_training_module, fold_scenarios,
                                    run_arm)
from closed_loop_divergence_audit import parse_abort
from h4_unseen_conditions import build_traces, unseen_config

# Seuil de « raffiner presque tout » : borne basse deja balayee par t15b,
# donc pas un parametre choisi pour cette tache.
FLOOR_THRESHOLD = 0.05


def measure(T, fold, cfg, dns, hp, tag):
    """Une execution classique, avec capture de l'avortement."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = run_arm(T, fold, cfg, dns, hp, True, verbose=True)
    ab = parse_abort(buf.getvalue())
    try:
        import matplotlib.pyplot as _plt
        _plt.close("all")
    except Exception:
        pass
    out = {k: float(r.get(k, np.nan))
           for k in ("combined", "phys_score", "patch_ratio")}
    out["completed"] = ab is None
    print(f"  {tag}: phys={out['phys_score']:.5f} "
          f"patch={out['patch_ratio']:.4f}"
          f"{'' if ab is None else '   **ABORTED**'}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(
        description="V4 T22d: attainable floor on the unseen condition")
    from config import RESULTS_DIR

    p.add_argument("--fold", required=True)
    p.add_argument("--threshold", type=float, default=FLOOR_THRESHOLD)
    args = p.parse_args()

    rec = json.load(open(os.path.join(
        RESULTS_DIR, f"t15_level3_fold_{args.fold}.json")))
    scenario = rec["scenario"]
    tpath = os.path.join(RESULTS_DIR,
                         f"t22_unseen_unseen-ic_{args.fold}.json")
    if not os.path.exists(tpath):
        raise SystemExit(f"run t22 for fold {args.fold} first")
    t22 = json.load(open(tpath))

    print("=" * 80)
    print(f"  V4 T22d - attainable floor, fold {args.fold} ({scenario})")
    print(f"  classical arm at threshold {args.threshold} = refine almost "
          f"everything")
    print("=" * 80, flush=True)

    T = _load_v1_training_module()
    cfg = dict(fold_scenarios(T, warn=False))[args.fold]
    hp = dict(rec["hyperparams"])
    hp["threshold_amr"] = args.threshold

    dns_can = build_traces(T, args.fold, cfg, scenario, unseen=False)
    dns_uns = build_traces(T, args.fold, cfg, scenario, unseen=True)
    floor_can = measure(T, args.fold, cfg, dns_can, hp, "floor canonical")
    floor_uns = measure(T, args.fold, unseen_config(cfg, scenario), dns_uns,
                        hp, "floor unseen   ")

    if not (floor_can["completed"] and floor_uns["completed"]):
        raise SystemExit("a floor run aborted; the floor is not measured")

    out = {"fold": args.fold, "scenario": scenario,
           "threshold": args.threshold,
           "floor_canonical": floor_can, "floor_unseen": floor_uns,
           "arms": {}}

    print("\n  distance to the attainable floor (phys / phys_floor)")
    print(f"  {'arm':<11}{'canonical':>22}{'unseen':>22}")
    for arm in ("qhas", "classical"):
        a = t22["arms"][arm]
        dc = a["canonical"]["phys_score"] / floor_can["phys_score"]
        du = a["unseen"]["phys_score"] / floor_uns["phys_score"]
        out["arms"][arm] = {"canonical_over_floor": float(dc),
                            "unseen_over_floor": float(du)}
        print(f"  {arm:<11}{dc:>21.2f}x{du:>21.2f}x")

    dq = out["arms"]["qhas"]["unseen_over_floor"]
    dc_ = out["arms"]["classical"]["unseen_over_floor"]
    # un bras SOUS 1.0 signale que la reference n'est pas une borne
    below = min(dq, dc_) < 1.0
    at_floor = dq < 1.5 and dc_ < 1.5
    out["both_arms_at_floor_on_unseen"] = bool(at_floor)
    print("\n  " + "-" * 74)
    if below:
        print("  NOTE: an arm scores BELOW 1.00x, so near-full refinement is")
        print("  not optimal here and this reference is not a lower bound.")
    if at_floor:
        print("  => BOTH arms sit within 1.5x of the attainable floor on the")
        print("     unseen condition. The narrowing of their ratio is floor")
        print("     convergence, NOT evidence of better transfer.")
    else:
        print("  => at least one arm remains well above the floor, so the")
        print("     arms still have room to differ: their ratio reflects the")
        print("     decision rules, not a shared ceiling on achievable error.")

    op = os.path.join(RESULTS_DIR, f"t22d_unseen_floor_{args.fold}.json")
    out["git_hash"] = git_commit_hash()
    out["cli_args"] = vars(args)
    json.dump(out, open(op, "w"), indent=1)
    print(f"\n  saved: {os.path.basename(op)}")


if __name__ == "__main__":
    main()
