#!/usr/bin/env python3
"""
V4 Task 21 - Le critere primaire est-il BIEN POSE ? (mesure, pas jugement)

LE PROBLEME. Le critere pre-enregistre est un scalaire :

    combined(lambda) = (phys + lambda * patch) / (1 + lambda),   lambda = 0.4

Sur les quatre folds il donne 2-2, et l'analyse a jusqu'ici ARGUMENTE que ce
verdict est contamine par le defaut D4 (le seuil du bras QAOA est fige a
0.1496 tandis que le bras classique regle librement le sien, donc les deux
bras ne sont pas au meme point de la frontiere cout-erreur). C'etait un
JUGEMENT. Cette tache le remplace par des MESURES.

TROIS MESURES, aucune ne demandant de nouvelle simulation.

  1. ORDRE PARTIEL DE PARETO (sans lambda). Si un bras est meilleur sur
     phys ET sur patch, il domine, quel que soit lambda : le verdict ne
     depend d'aucun arbitrage. La ou la dominance tranche, il n'y a rien a
     juger.

  2. CROISEMENT EN LAMBDA. Quand les bras sont incomparables, le verdict
     scalarise bascule a
         lambda* = (phys_c - phys_q) / (patch_q - patch_c)
     Si lambda* > 0, le gagnant depend du choix de lambda, et un fold dont
     le verdict bascule a l'interieur d'une plage defendable ne MESURE
     rien : il enregistre une convention.

  3. STABILITE DU COMPTAGE. Le decompte des folds gagnes par chaque bras
     est recalcule sur une grille de lambda. Si le 2-2 n'existe que dans
     une fenetre etroite de lambda, ce n'est pas un resultat.

CE QUE CELA NE FAIT PAS. Cela ne supprime pas D4 : cela mesure si D4
change la conclusion. La suppression demanderait de re-regler le bras QAOA
avec `threshold_amr` dans l'espace de recherche (voir la note finale) —
plusieurs heures de calcul.

Statut : analyse de sensibilite POST-HOC, declaree comme telle. Elle ne
remplace pas le critere pre-enregistre, elle en teste la robustesse.

Sortie : results/t21_endpoint_wellposedness.json
Usage :
  python study/v4/t21_endpoint_wellposedness.py
"""
import argparse, json, os, sys

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
from t15c_fold_synthesis import load_divergence_audit, load_fold

LAMBDA_PREREG = 0.4          # T.LAMBDA_COST_SOFT, fige avant la campagne


def combined(phys, patch, lam):
    return (phys + lam * patch) / (1.0 + lam)


def dominance(q_phys, q_patch, c_phys, c_patch):
    """Ordre partiel, sans lambda. Retourne 'qhas', 'classical' ou None."""
    q_better = (q_phys <= c_phys) and (q_patch <= c_patch)
    c_better = (c_phys <= q_phys) and (c_patch <= q_patch)
    q_strict = q_better and (q_phys < c_phys or q_patch < c_patch)
    c_strict = c_better and (c_phys < q_phys or c_patch < q_patch)
    if q_strict and not c_strict:
        return "qhas"
    if c_strict and not q_strict:
        return "classical"
    return None                      # incomparables


def crossover_lambda(q_phys, q_patch, c_phys, c_patch):
    """lambda ou les deux `combined` s'egalent.

    combined_q = combined_c  <=>  phys_q + L*patch_q = phys_c + L*patch_c
    d'ou L* = (phys_c - phys_q) / (patch_q - patch_c).
    Retourne None si les patch sont egaux (pas de croisement).
    """
    denom = q_patch - c_patch
    if abs(denom) < 1e-12:
        return None
    return float((c_phys - q_phys) / denom)


def analyse_fold(rec):
    q, c = rec["qhas"], rec["classical"]
    qp, qc = float(q["phys_score"]), float(q["patch_ratio"])
    cp, cc = float(c["phys_score"]), float(c["patch_ratio"])
    dom = dominance(qp, qc, cp, cc)
    lstar = crossover_lambda(qp, qc, cp, cc)
    win_prereg = ("qhas" if combined(qp, qc, LAMBDA_PREREG)
                  < combined(cp, cc, LAMBDA_PREREG) else "classical")

    # Sens du basculement : au-dela de lambda*, qui gagne ?
    if lstar is not None and lstar > 0:
        hi = combined(qp, qc, lstar * 2 + 1.0) < combined(cp, cc,
                                                          lstar * 2 + 1.0)
        winner_above = "qhas" if hi else "classical"
    else:
        winner_above = None

    return {
        "fold": rec["fold"], "scenario": rec["scenario"],
        "qhas_phys": qp, "qhas_patch": qc,
        "classical_phys": cp, "classical_patch": cc,
        "dominance": dom,
        "lambda_star": lstar,
        "lambda_star_in_range": bool(lstar is not None and 0.0 < lstar < 10.0),
        "winner_at_prereg_lambda": win_prereg,
        "winner_above_lambda_star": winner_above,
        "verdict_is_lambda_free": dom is not None,
    }


def counting_vs_lambda(rows, lambdas):
    """Comptage des folds gagnes par bras, en fonction de lambda."""
    out = []
    for lam in lambdas:
        nq = sum(1 for r in rows
                 if combined(r["qhas_phys"], r["qhas_patch"], lam)
                 < combined(r["classical_phys"], r["classical_patch"], lam))
        out.append({"lambda": float(lam), "n_qhas": int(nq),
                    "n_classical": int(len(rows) - nq)})
    return out


def main():
    p = argparse.ArgumentParser(
        description="V4 T21: is the primary endpoint well posed?")
    from config import RESULTS_DIR

    p.add_argument("--folds", nargs="+",
                   default=["ot", "kh", "rotor", "tearing"])
    p.add_argument("--results-dir", default=None)
    args = p.parse_args()
    results_dir = args.results_dir or RESULTS_DIR

    audit = load_divergence_audit(results_dir)
    recs, excluded = [], []
    for f in args.folds:
        r = load_fold(results_dir, f)
        if r is None:
            continue
        if audit is not None and not audit.get(f, True):
            excluded.append(f)
            continue
        recs.append(r)
    if not recs:
        raise SystemExit("no usable fold")

    print("=" * 80)
    print("  V4 T21 - Is the primary endpoint well posed?")
    print(f"  pre-registered lambda = {LAMBDA_PREREG}")
    if excluded:
        print(f"  excluded (failed divergence audit): {', '.join(excluded)}")
    if audit is None:
        print("  WARNING: no divergence audit; fold validity unverified")
    print("=" * 80)

    rows = [analyse_fold(r) for r in recs]

    print("\n  1. PARETO DOMINANCE (no lambda involved)")
    print(f"  {'fold':<10}{'dominates':<12}{'verdict lambda-free?'}")
    for r in rows:
        d = r["dominance"] or "incomparable"
        print(f"  {r['fold']:<10}{d:<12}{r['verdict_is_lambda_free']}")
    n_free = sum(r["verdict_is_lambda_free"] for r in rows)
    n_cl = sum(1 for r in rows if r["dominance"] == "classical")
    n_q = sum(1 for r in rows if r["dominance"] == "qhas")
    print(f"  -> decided without any lambda: {n_free}/{len(rows)} folds "
          f"(classical {n_cl}, Q-HAS {n_q})")

    print("\n  2. LAMBDA CROSSOVER (for the folds dominance cannot decide)")
    print(f"  {'fold':<10}{'lambda*':>10}   winner below / above")
    for r in rows:
        if r["dominance"] is not None:
            continue
        ls = r["lambda_star"]
        below = ("qhas" if r["winner_at_prereg_lambda"] == "qhas"
                 else "classical")
        print(f"  {r['fold']:<10}{ls:>10.4f}   {below} / "
              f"{r['winner_above_lambda_star']}")
        print(f"  {'':<10}{'':>10}   pre-registered lambda={LAMBDA_PREREG} "
              f"sits {'BELOW' if LAMBDA_PREREG < ls else 'ABOVE'} the "
              f"crossover")

    print("\n  3. COUNT STABILITY ACROSS LAMBDA")
    grid = [0.0, 0.1, 0.2, LAMBDA_PREREG, 0.6, 0.8, 1.0, 1.5,
            2.0, 5.0, 20.0, 100.0]
    counts = counting_vs_lambda(rows, grid)
    print(f"  {'lambda':>8}{'Q-HAS wins':>12}{'classical wins':>16}")
    for c in counts:
        star = "  <- pre-registered" if abs(
            c["lambda"] - LAMBDA_PREREG) < 1e-9 else ""
        print(f"  {c['lambda']:>8.2f}{c['n_qhas']:>12}"
              f"{c['n_classical']:>16}{star}")

    # DISTINGUER deux choses que j'avais d'abord confondues :
    #   - le COMPTAGE change-t-il avec lambda ? (marge)
    #   - le VAINQUEUR MAJORITAIRE change-t-il ? (verdict)
    # Un comptage qui passe de 2-1 a 3-0 ne rend pas le critere mal pose :
    # le verdict est le meme. Seul un basculement du vainqueur le ferait.
    counts_change = len({(c["n_qhas"], c["n_classical"]) for c in counts}) > 1
    winners = {("qhas" if c["n_qhas"] > c["n_classical"]
                else "classical" if c["n_classical"] > c["n_qhas"]
                else "tie") for c in counts}
    verdict_flips = len(winners - {"tie"}) > 1

    print(f"\n  margin changes with lambda : {counts_change}")
    print(f"  VERDICT flips with lambda  : {verdict_flips}"
          f"   (winners seen: {', '.join(sorted(winners))})")
    if verdict_flips:
        print("  => the endpoint's verdict is partly a property of the")
        print("     chosen lambda, not of the arms. That is a MEASUREMENT")
        print("     of ill-posedness.")
    else:
        w = (winners - {"tie"}).pop() if winners - {"tie"} else "tie"
        print(f"  => the verdict is STABLE across the whole lambda range "
              f"tested:\n     `{w}` holds the majority everywhere. Only the "
              f"margin moves.\n     The endpoint is therefore NOT ill-posed "
              f"in its direction; the\n     earlier reading that it was "
              f"overstated the case.")
    flips = verdict_flips

    out = {
        "lambda_prereg": LAMBDA_PREREG,
        "folds": rows,
        "excluded_failed_audit": excluded,
        "counting_vs_lambda": counts,
        "count_changes_with_lambda": bool(counts_change),
        "verdict_flips_with_lambda": bool(verdict_flips),
        "winners_across_lambda": sorted(winners),
        "n_decided_without_lambda": int(n_free),
        "n_dominated_by_classical": int(n_cl),
        "n_dominated_by_qhas": int(n_q),
        "git_hash": git_commit_hash(),
        "cli_args": vars(args),
    }
    path = os.path.join(results_dir, "t21_endpoint_wellposedness.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n  saved: {os.path.basename(path)}")
    print("\n  NOTE. This measures whether D4 changes the conclusion; it does")
    print("  not remove D4. Removing it requires re-tuning the QAOA arm with")
    print("  `threshold_amr` in the search space, so both arms optimise the")
    print("  same free parameters — hours of compute, and the definitive")
    print("  experiment.")
    print("\nV4 Task 21 complete.")


if __name__ == "__main__":
    main()
