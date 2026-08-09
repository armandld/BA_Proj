"""V4 Tache 23 — le decompte de tete, CALCULE et non recopie.

POURQUOI CETTE TACHE EXISTE
---------------------------
Le resultat le plus cite de l'etude — « Q-HAS est moins fidele que la regle
classique appariee en budget sur 19/20 executions, plus couteux sur 18/20,
strictement Pareto-domine sur 17/20 » — n'etait produit par AUCUN script.
Il avait ete compose a la main dans `RESULTS_V4.md`, et ne se reproduisait
pas depuis les artefacts. Deux ecarts :

  1. sur `kh`, les colonnes « moins fidele » et « plus couteux » etaient
     TRANSPOSEES (4/5 et 5/5 au lieu de 5/5 et 4/5) ;
  2. sur `rotor`, les 2 tirages AVORTES etaient comptes au denominateur,
     d'ou un total sur 20 quand seules 18 executions ont abouti. C'est
     exactement le defaut deja recense (une agregation qui melange les
     tirages avortes aux valides), reapparu dans le texte apres avoir ete
     corrige dans le code.

Le decompte correct est plus NET sur la fidelite et plus faible sur le
cout : 18/18, 16/18, 16/18.

LA CONVENTION, EXPLICITE
------------------------
Reference = le point BUDGET-APPARIE mesure par T15b
(`t15b_budget_matched_<fold>.json::matched_classical`), sur SES DEUX
coordonnees : `phys_score` et `patch_ratio`.

  - « moins fidele »  : phys_score(tirage Q-HAS) > phys_score(apparie)
  - « plus couteux »  : patch_ratio(tirage Q-HAS) > patch_ratio(apparie)
  - « domine »        : les deux a la fois

Ne PAS utiliser `t20::classical_stats` comme reference. Ce bloc est le
controle de determinisme du bras classique, et sur `ot` et `kh` il a tourne
au seuil REGLE, pas au seuil apparie (defaut D14) : sur `ot` il rend
phys = 0.4845 la ou le point apparie vaut 0.0827, ce qui inverserait le sens
du resultat sur ce fold.

Seuls les tirages `completed` comptent. Un tirage avorte n'est pas un point
de mesure, ni au numerateur ni au denominateur.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------

from t1_feature_selection import git_commit_hash    # v3, reutilise

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
FOLDS = ("ot", "kh", "rotor", "tearing")


def matched_reference(results_dir, fold):
    """Le point budget-apparie de T15b : (phys_score, patch_ratio)."""
    p = os.path.join(results_dir, f"t15b_budget_matched_{fold}.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))["matched_classical"]
    return float(m["phys_score"]), float(m["patch_ratio"])


def fold_counts(results_dir, fold):
    """Decompte d'un fold, ou None si un artefact manque.

    Retourne un dict avec n_completed, n_aborted et les trois compteurs.
    """
    ref = matched_reference(results_dir, fold)
    tp = os.path.join(results_dir, f"t20_qhas_run_variance_{fold}.json")
    if ref is None or not os.path.exists(tp):
        return None
    ref_phys, ref_patch = ref
    t = json.load(open(tp))
    ok = [r for r in t["qhas_runs"] if r["completed"]]
    less = sum(1 for r in ok if r["phys_score"] > ref_phys)
    cost = sum(1 for r in ok if r["patch_ratio"] > ref_patch)
    dom = sum(1 for r in ok
              if r["phys_score"] > ref_phys and r["patch_ratio"] > ref_patch)
    # Le controle classique du meme artefact : combien de ses rejeux ont
    # avorte AU POINT COMPARE. C'est la seule forme sous laquelle
    # l'asymetrie d'avortement peut etre affirmee — le bras classique
    # diverge lui aussi a d'AUTRES seuils (T19 : le seuil regle de `rotor`,
    # et 2 de ses 6 points de bissection).
    cr = t.get("classical_runs", [])
    c_ab = sum(1 for r in cr if not r.get("completed", True))
    return {
        "fold": fold,
        "ref_phys": ref_phys, "ref_patch": ref_patch,
        "n_runs": len(t["qhas_runs"]),
        "n_completed": len(ok),
        "n_aborted": len(t["qhas_runs"]) - len(ok),
        "n_classical_runs": len(cr),
        "n_classical_aborted": c_ab,
        "less_faithful": less, "costlier": cost, "dominated": dom,
    }


def totals(rows):
    t = {"n_completed": 0, "n_aborted": 0,
         "n_classical_runs": 0, "n_classical_aborted": 0,
         "less_faithful": 0, "costlier": 0, "dominated": 0}
    for r in rows:
        for k in t:
            t[k] += r[k]
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds", nargs="+", default=list(FOLDS))
    p.add_argument("--seed", type=int, default=0,
                   help="sans effet : cette tache ne simule rien, elle "
                        "relit des artefacts. Present pour l'uniformite.")
    args = p.parse_args()

    rows, missing = [], []
    for f in args.folds:
        r = fold_counts(RESULTS_DIR, f)
        (rows.append(r) if r else missing.append(f))

    print("=" * 84)
    print("  V4 T23 - headline counts, recomputed from the artifacts")
    print("=" * 84)
    if missing:
        print(f"  missing artifacts for: {', '.join(missing)}")
    if not rows:
        raise SystemExit("no fold has both t15b and t20 artifacts")

    print("\n  reference = t15b budget-matched point, BOTH coordinates")
    print("  aborted draws excluded from numerator AND denominator\n")
    print("  | fold | n | aborted | less faithful | costlier | dominated |")
    print("  |---|---|---|---|---|---|")
    for r in rows:
        n = r["n_completed"]
        print(f"  | `{r['fold']}` | {n} | {r['n_aborted']} | "
              f"{r['less_faithful']}/{n} | {r['costlier']}/{n} | "
              f"**{r['dominated']}/{n}** |")
    t = totals(rows)
    n = t["n_completed"]
    print(f"  | **total** | **{n}** | {t['n_aborted']} | "
          f"**{t['less_faithful']}/{n}** | **{t['costlier']}/{n}** | "
          f"**{t['dominated']}/{n}** |")

    print(f"\n  Over {n} completed closed-loop runs across "
          f"{len(rows)} held-out classes, Q-HAS is less faithful than the")
    print(f"  budget-matched classical rule on {t['less_faithful']}/{n}, "
          f"more expensive on {t['costlier']}/{n}, and strictly")
    print(f"  Pareto-dominated on {t['dominated']}/{n}.")
    print(f"\n  aborts at the compared operating point: "
          f"Q-HAS {t['n_aborted']}/{n + t['n_aborted']}, "
          f"classical {t['n_classical_aborted']}/{t['n_classical_runs']}.")
    print("  This is NOT a claim that the classical rule never diverges: "
          "T19 records\n  its tuned threshold aborting on `rotor`, and 2 of "
          "that fold's 6 bisection\n  points. Divergence is a property of "
          "the threshold, and both arms have\n  thresholds that diverge.")

    out = {"folds": rows, "total": t, "n_folds": len(rows),
           "missing": missing, "git_hash": git_commit_hash(),
           "cli_args": vars(args)}
    op = os.path.join(RESULTS_DIR, "t23_headline_counts.json")
    json.dump(out, open(op, "w"), indent=1)
    print(f"\n  saved: {os.path.basename(op)}")
    print("\nV4 Task 23 complete.")


if __name__ == "__main__":
    main()
