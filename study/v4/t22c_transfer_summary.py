#!/usr/bin/env python3
"""
V4 Task 22c - Synthese du test de transfert (conditions initiales inedites).

NE CALCULE AUCUNE SIMULATION : agrege les sorties de `t22_unseen_conditions`
lancees avec `--repeats N --matched-reference`.

LA QUESTION. La passe a un seul tirage suggerait que Q-HAS se comporte
RELATIVEMENT MIEUX que la regle classique quand la condition initiale est
inedite : le rapport phys Q/C se resserrait sur les quatre folds (ot
0.22->0.17, kh 2.52->1.81, rotor 3.67->1.88, tearing 2.94->1.01). Si c'etait
reel, cela contredirait la lecture « l'apprentissage ne transfere pas ».

POURQUOI ELLE N'ETAIT PAS TRANCHEE. Un tirage par condition, contre un
coefficient de variation de 17 a 49 % mesure par T20 (defaut D11, chaine
VQA sans germe). Les variations observees etaient du meme ordre que le
bruit. De plus la comparaison utilisait le bras classique REGLE, a un autre
budget (defaut D4).

CE QUE CETTE SYNTHESE FAIT. Avec N tirages par condition :
  - le ratio de degradation de Q-HAS devient une moyenne assortie d'un
    ecart-type propage ;
  - celui du bras classique est EXACT (T20 : etendue exactement nulle sur
    8 rejeux), donc toute l'incertitude est du cote Q-HAS ;
  - le test decisif est |deg_Q - deg_C| / sd(deg_Q) : sous 2, le
    resserrement observe n'est pas separable du bruit.

LECTURE PRINCIPALE. Le comptage de DOMINANCE sur la condition inedite, qui
ne depend d'aucun appariement de budget. C'est important car l'appariement
a ete calibre sur la condition CANONIQUE : sur la condition inedite le
`patch` de Q-HAS bouge, donc l'egalite de budget n'y est qu'approchee.

Sortie : results/t22c_transfer_summary.json
Usage :
  python study/v4/t22c_transfer_summary.py
"""
import argparse, json, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "v3"))
sys.path.insert(0, _HERE)

from t1_feature_selection import git_commit_hash

FOLDS = ("ot", "kh", "rotor", "tearing")


def ratio_sd(mu_u, sd_u, mu_c, sd_c):
    """Ecart-type du rapport mu_u/mu_c par propagation au premier ordre.

    Les deux moyennes proviennent de tirages INDEPENDANTS (executions
    distinctes), donc les termes croises s'annulent.
    """
    if mu_c == 0:
        return float("nan")
    rel = (sd_u / mu_u) ** 2 if mu_u else 0.0
    rel += (sd_c / mu_c) ** 2
    return float(abs(mu_u / mu_c) * np.sqrt(rel))


def load(results_dir, fold, mode="unseen-ic"):
    p = os.path.join(results_dir, f"t22_unseen_{mode}_{fold}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    # un enregistrement `total_abort` ne porte AUCUNE moyenne : le bras n'a
    # pas de point de fonctionnement sur cette condition. Il ne doit ni etre
    # analyse comme les autres, ni disparaitre du compte rendu.
    if d.get("status") == "partial":
        # point de reprise : aucune statistique exploitable
        return {"fold": fold, "underpowered": True, "partial": True,
                "raw": d}
    if d.get("status") == "total_abort":
        return {"fold": fold, "underpowered": True, "total_abort": True,
                "total_abort_arm": d.get("total_abort_arm"), "raw": d}
    if d["arms"]["qhas"].get("n_runs", 1) < 2:
        return {"fold": fold, "underpowered": True, "raw": d}
    return {"fold": fold, "underpowered": False, "raw": d}


def analyse(rec):
    d = rec["raw"]
    q, c = d["arms"]["qhas"], d["arms"]["classical"]
    qc, qu = q["canonical"]["phys_score"], q["unseen"]["phys_score"]
    cc, cu = c["canonical"]["phys_score"], c["unseen"]["phys_score"]
    sqc = q.get("canonical_phys_sd", 0.0)
    squ = q.get("unseen_phys_sd", 0.0)

    deg_q = qu / qc if qc else float("nan")
    deg_c = cu / cc if cc else float("nan")
    sd_deg_q = ratio_sd(qu, squ, qc, sqc)

    # test decisif : les deux degradations sont-elles separables ?
    z = (abs(deg_q - deg_c) / sd_deg_q
         if sd_deg_q and np.isfinite(sd_deg_q) else float("nan"))

    # DEPLACEMENT DU RAPPORT ENTRE BRAS :
    #     R(cond) = phys_qhas(cond) / phys_classical(cond)
    #     shift   = R_unseen / R_canonical
    #
    # ATTENTION — je l'avais d'abord presente comme une mesure « non
    # confondue », par opposition au rapport des degradations. C'EST FAUX :
    #     R_uns/R_can = (qu/cu)/(qc/cc) = (qu/qc)/(cu/cc) = deg_q/deg_c
    # les deux quantites sont ALGEBRIQUEMENT IDENTIQUES, d'ou le meme |z|.
    # Ce n'est donc qu'une reformulation, pas un controle supplementaire.
    #
    # Le confondant demeure et n'est PAS traite ici : si la condition
    # inedite est plus facile et que les deux bras butent sur un plancher
    # commun (discretisation, budget), ils convergent quelle que soit la
    # qualite de leur regle de decision, et le rapport se resserre
    # mecaniquement. Distinguer « plancher commun » de « meilleur
    # transfert » demanderait de mesurer ce plancher — non fait.
    # On rapporte donc le nombre, et le seul enonce qui n'en depende pas
    # reste le comptage de dominance.
    r_can = qc / cc if cc else float("nan")
    r_uns = qu / cu if cu else float("nan")
    shift = r_uns / r_can if r_can else float("nan")
    sd_r_can = ratio_sd(qc, sqc, cc, 0.0)     # bras classique deterministe
    sd_r_uns = ratio_sd(qu, squ, cu, 0.0)
    sd_shift = ratio_sd(r_uns, sd_r_uns, r_can, sd_r_can)
    # le rapport se deplace-t-il significativement par rapport a 1 ?
    z_shift = (abs(shift - 1.0) / sd_shift
               if sd_shift and np.isfinite(sd_shift) else float("nan"))

    # dominance sur la condition inedite, tirage par tirage (sans budget).
    # Les tirages AVORTES sont exclus : un plantage n'est pas une decision.
    qu_runs = [r for r in d["arms"]["qhas"].get("unseen_runs", [])
               if r.get("completed", True)]
    worse = sum(1 for r in qu_runs if r["phys_score"] > cu)
    costlier = sum(1 for r in qu_runs
                   if r["patch_ratio"] > c["unseen"]["patch_ratio"])
    dominated = sum(1 for r in qu_runs
                    if r["phys_score"] > cu
                    and r["patch_ratio"] > c["unseen"]["patch_ratio"])
    return {
        "fold": rec["fold"],
        "n_runs": q.get("n_runs"),
        "n_aborted": q.get("n_aborted", 0),
        "n_usable_unseen": len(qu_runs),
        "dns_relative_shift": d.get("dns_relative_shift"),
        "weak_condition": bool(d.get("unseen_condition_is_weak", False)),
        "classical_reference": d.get("classical_reference_source"),
        "qhas_canonical": qc, "qhas_canonical_sd": sqc,
        "qhas_unseen": qu, "qhas_unseen_sd": squ,
        "classical_canonical": cc, "classical_unseen": cu,
        "qhas_patch_unseen": q["unseen"]["patch_ratio"],
        "classical_patch_unseen": c["unseen"]["patch_ratio"],
        "deg_qhas": float(deg_q), "deg_qhas_sd": float(sd_deg_q),
        "deg_classical": float(deg_c),
        "separation_z": float(z),
        "separable": bool(np.isfinite(z) and z >= 2.0),
        "arm_ratio_shift": float(shift),
        "arm_ratio_shift_sd": float(sd_shift),
        "arm_ratio_shift_z": float(z_shift),
        "arm_ratio_shift_significant": bool(
            np.isfinite(z_shift) and z_shift >= 2.0),
        "n_worse_on_unseen": int(worse),
        "n_costlier_on_unseen": int(costlier),
        "n_dominated_on_unseen": int(dominated),
        "ratio_canonical": float(qc / cc) if cc else float("nan"),
        "ratio_unseen": float(qu / cu) if cu else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(
        description="V4 T22c: transfer test summary across folds")
    from config import RESULTS_DIR
    p.add_argument("--folds", nargs="+", default=list(FOLDS))
    p.add_argument("--mode", default="unseen-ic")
    args = p.parse_args()

    recs, missing, weak, dead = [], [], [], []
    for f in args.folds:
        r = load(RESULTS_DIR, f, args.mode)
        if r is None:
            missing.append(f)
        elif r.get("total_abort"):
            dead.append((f, r["total_abort_arm"]))
        elif r["underpowered"]:
            weak.append(f)
        else:
            recs.append(analyse(r))

    print("=" * 84)
    print("  V4 T22c - transfer to unseen initial conditions")
    print("=" * 84)
    if missing:
        print(f"  not run: {', '.join(missing)}")
    if weak:
        print(f"  single-run only (cannot separate from D11 noise): "
              f"{', '.join(weak)}")
    if dead:
        for f, arm in dead:
            print(f"  {f}: NO OPERATING POINT — the {arm} arm aborted on "
                  f"every draw of one condition; no ratio is defined, and "
                  f"this fold cannot be scored for or against transfer")
    if not recs:
        raise SystemExit("no fold with repeated draws; run t22 --repeats 5")

    print(f"\n  {'fold':<9}{'deg Q-HAS':>18}{'deg classical':>15}"
          f"{'|z|':>7}{'separable':>11}")
    for r in recs:
        print(f"  {r['fold']:<9}{r['deg_qhas']:>10.3f}+-{r['deg_qhas_sd']:<6.3f}"
              f"{r['deg_classical']:>15.3f}{r['separation_z']:>7.2f}"
              f"{str(r['separable']):>11}")

    print(f"\n  {'fold':<9}{'arm-ratio shift R_uns/R_can':>30}{'|z| vs 1':>10}"
          f"{'significant':>13}   (= deg_Q/deg_C, same number)")
    for r in recs:
        print(f"  {r['fold']:<9}{r['arm_ratio_shift']:>19.3f}"
              f"+-{r['arm_ratio_shift_sd']:<8.3f}{r['arm_ratio_shift_z']:>10.2f}"
              f"{str(r['arm_ratio_shift_significant']):>13}")

    print(f"\n  {'fold':<9}{'phys ratio Q/C':>26}{'dominated on unseen':>22}")
    for r in recs:
        print(f"  {r['fold']:<9}{r['ratio_canonical']:>11.2f}x ->"
              f"{r['ratio_unseen']:>10.2f}x"
              f"{r['n_dominated_on_unseen']:>15}/{r['n_usable_unseen']}"
              f"  (worse {r['n_worse_on_unseen']}, "
              f"costlier {r['n_costlier_on_unseen']})")

    weak = [r["fold"] for r in recs if r.get("weak_condition")]
    if weak:
        print(f"\n  NEARLY VACUOUS unseen condition (<1% trajectory shift): "
              f"{', '.join(weak)}")
        print("  Those folds cannot support a transfer claim: the 'unseen'")
        print("  condition is barely distinguishable from the canonical one.")

    n_sep = sum(r["separable"] for r in recs)
    tot_dom = sum(r["n_dominated_on_unseen"] for r in recs)
    tot_run = sum(r["n_usable_unseen"] for r in recs)
    tot_ab = sum(r.get("n_aborted", 0) for r in recs)
    tot_worse = sum(r["n_worse_on_unseen"] for r in recs)

    print("\n  " + "-" * 78)
    print(f"  folds where the two degradations are separable (|z| >= 2): "
          f"{n_sep}/{len(recs)}")
    if n_sep == 0:
        print("  => the apparent narrowing of the Q-HAS/classical ratio on")
        print("     unseen conditions is NOT separable from the sampling")
        print("     noise of the unseen QAOA chain (D11). No transfer claim")
        print("     in either direction is supported.")
    if tot_ab:
        print(f"  aborted Q-HAS runs excluded across folds: {tot_ab}")
    print(f"  Q-HAS less faithful on the unseen condition: "
          f"{tot_worse}/{tot_run} runs")
    print(f"  Q-HAS strictly dominated on the unseen condition: "
          f"{tot_dom}/{tot_run} runs")
    print("\n  CAVEAT: budget matching was calibrated on the CANONICAL")
    print("  condition, so equality of budget on the unseen condition is")
    print("  approximate. The dominance count above needs no matching.")

    out = {"folds": recs, "missing": missing, "single_run_only": weak,
           "n_separable": int(n_sep), "n_runs_total": int(tot_run),
           "n_worse_total": int(tot_worse), "n_dominated_total": int(tot_dom),
           "git_hash": git_commit_hash(), "cli_args": vars(args)}
    path = os.path.join(RESULTS_DIR, "t22c_transfer_summary.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n  saved: {os.path.basename(path)}")


if __name__ == "__main__":
    main()
