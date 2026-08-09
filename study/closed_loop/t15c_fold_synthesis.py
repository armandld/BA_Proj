#!/usr/bin/env python3
"""
V4 Task 15c - Synthese inter-folds du niveau 3 (closed loop).

Ce module NE CALCULE AUCUNE SIMULATION. Il agrege les sorties deja
produites par t15 (`t15_level3_fold_{f}.json`) et t15b
(`t15b_budget_matched_{f}.json`) et applique, sans les modifier, les regles
de decision figees dans `docs/level3_preregistration.md` §4.

DEUX NIVEAUX D'ANALYSE, explicitement separes :

  A. PRIMAIRE, PRE-ENREGISTRE (`docs/level3_preregistration.md` §4)
     Critere `combined` = (phys + lambda*patch)/(1+lambda), apparie par
     fold, Q-HAS contre le bras classique REGLE (celui du fold). Regles :
       - comptage : un bras gagnant sur >= 3/4 folds ;
       - TOST au seuil d'equivalence = 5% du `combined` classique moyen,
         marge fixee avant tout calcul (elle est ici RELUE des donnees mais
         par la formule pre-enregistree, jamais choisie a la vue des ecarts);
       - Holm-Bonferroni sur la famille des tests rapportes.

  B. SECONDAIRE, POST-HOC ET DECLARE (defaut D4)
     Comparaison a BUDGET APPARIE (t15b). Cette analyse a ete ajoutee
     APRES lecture du fold `ot`, parce que le bras QAOA de V1 fige son
     `threshold_amr` alors que le bras classique le regle librement : les
     deux bras du critere primaire ne sont donc pas au meme point de la
     frontiere cout-erreur. Elle est rapportee comme secondaire et
     exploratoire, jamais comme confirmation du plan pre-enregistre.

Sorties : results/t15c_fold_synthesis.json  (+ table markdown sur stdout)
Usage :
  python study/v4/t15c_fold_synthesis.py
  python study/v4/t15c_fold_synthesis.py --folds ot kh rotor tearing
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
from stats_confirmatory import holm_correction, tost_equivalence

ALL_FOLDS = ("ot", "kh", "rotor", "tearing")

# Marge d'equivalence pre-enregistree : 5% du `combined` classique moyen.
TOST_MARGIN_FRAC = 0.05

# Regle de comptage pre-enregistree : un bras doit gagner sur au moins
# 3 folds sur 4 pour que la difference soit declaree consistante.
WIN_RULE_MIN = 3


def load_divergence_audit(results_dir):
    """Lit l'audit T19 s'il existe : {fold: fold_usable}.

    Sans audit on ne PEUT PAS savoir si un bras a fini sa trajectoire (V1
    renvoie les memes cles dans les deux cas). L'absence d'audit est donc
    signalee, jamais interpretee comme « tout va bien ».
    """
    p = os.path.join(results_dir, "t19_arm_divergence_audit.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {r["fold"]: bool(r["fold_usable"]) for r in d.get("results", [])}


def load_fold(results_dir, fold, prefix="t15_level3"):
    """Charge un fold. Retourne None si le fold n'a pas (encore) tourne.

    `budget` vaut None si t15b n'a pas encore tourne pour ce fold : le
    critere primaire reste calculable sans lui.
    """
    p15 = os.path.join(results_dir, f"{prefix}_fold_{fold}.json")
    if not os.path.exists(p15):
        return None
    d = json.load(open(p15))
    rec = {
        "fold": fold,
        "scenario": d["scenario"],
        "train_on": d["train_on"],
        "n_trials": d["n_trials"],
        "qhas": d["qhas"],
        "classical": d["classical"],
        "hyperparams": d["hyperparams"],
        "classical_params": d["classical_params"],
        "t_tune": d.get("t_tune"),
        "git_hash": d.get("git_hash"),
        "budget": None,
    }
    p15b = os.path.join(results_dir, f"t15b_budget_matched_{fold}.json")
    if os.path.exists(p15b):
        b = json.load(open(p15b))
        rec["budget"] = {
            "target_patch": b["target_patch"],
            "matched": b["matched_classical"],
            "delta_phys_matched": b["delta_phys_matched"],
            "trace": b["trace"],
        }
    return rec


def interp_frontier(trace, patch):
    """Erreur classique attendue au budget `patch`, par interpolation
    lineaire de la trace de bissection (reprise de make_pareto_figure pour
    garder une definition unique)."""
    pts = sorted(({"patch": r["patch_ratio"], "phys": r["phys_score"]}
                  for r in trace), key=lambda r: r["patch"])
    xs = np.array([r["patch"] for r in pts])
    ys = np.array([r["phys"] for r in pts])
    return float(np.interp(patch, xs, ys))


def primary_analysis(records, margin_frac=TOST_MARGIN_FRAC):
    """Critere primaire pre-enregistre : `combined` apparie par fold.

    Convention de signe : delta = Q-HAS - classique. `combined` est un
    cout, donc delta < 0 signifie Q-HAS meilleur.
    """
    q = np.array([r["qhas"]["combined"] for r in records], dtype=float)
    c = np.array([r["classical"]["combined"] for r in records], dtype=float)
    d = q - c
    n = len(d)

    # marge d'equivalence par la formule pre-enregistree
    margin = float(margin_frac * np.mean(c))

    out = {
        "n_folds": n,
        "folds": [r["fold"] for r in records],
        "qhas_combined": q.tolist(),
        "classical_combined": c.tolist(),
        "delta": d.tolist(),
        "mean_delta": float(np.mean(d)),
        "n_qhas_better": int(np.sum(d < 0)),
        "n_classical_better": int(np.sum(d > 0)),
        "win_rule_min": WIN_RULE_MIN,
        "margin": margin,
        "margin_frac": margin_frac,
    }
    out["qhas_wins_rule"] = bool(out["n_qhas_better"] >= WIN_RULE_MIN)
    out["classical_wins_rule"] = bool(
        out["n_classical_better"] >= WIN_RULE_MIN)

    if n >= 2:
        from scipy import stats
        t_res = stats.ttest_rel(q, c)
        out["paired_t_p"] = float(t_res.pvalue)
        # test des signes exact (binomial), bilateral : ne suppose pas la
        # normalite, mais avec n=4 son p minimal atteignable est 0.125
        k = int(np.sum(d < 0))
        out["sign_test_p"] = float(
            stats.binomtest(k, n, 0.5).pvalue) if k + int(
            np.sum(d > 0)) == n else None
        out["sign_test_p_min_attainable"] = float(
            stats.binomtest(0, n, 0.5).pvalue)
        tost = tost_equivalence(q, c, margin=margin, paired=True)
        out["tost"] = {k2: (bool(v) if isinstance(v, (bool, np.bool_))
                            else float(v))
                       for k2, v in tost.items()}
        holm = holm_correction([out["paired_t_p"], tost["p_tost"]])
        out["holm"] = {"labels": ["paired_t(difference)", "TOST(equivalence)"],
                       "p_adjusted": holm["p_adjusted"].tolist(),
                       "reject": holm["reject"].tolist()}
    else:
        out["paired_t_p"] = None
        out["sign_test_p"] = None
        out["tost"] = None
        out["holm"] = None
        out["note_underpowered"] = (
            "n < 2 folds: no paired statistic is defined.")
    return out


def secondary_analysis(records):
    """Analyse post-hoc a budget apparie (defaut D4). Ne porte que sur les
    folds pour lesquels t15b a tourne."""
    rows, deltas = [], []
    for r in records:
        b = r["budget"]
        if b is None:
            continue
        q_patch = r["qhas"]["patch_ratio"]
        q_phys = r["qhas"]["phys_score"]
        m = b["matched"]
        front_at_q = interp_frontier(b["trace"], q_patch)
        rows.append({
            "fold": r["fold"],
            "qhas_patch": q_patch,
            "qhas_phys": q_phys,
            "matched_threshold": m["threshold"],
            "matched_patch": m["patch_ratio"],
            "matched_phys": m["phys_score"],
            "delta_phys_matched": b["delta_phys_matched"],
            "frontier_phys_at_qhas_budget": front_at_q,
            "ratio_vs_frontier": (float(q_phys / front_at_q)
                                  if front_at_q > 0 else None),
            "budget_gap": abs(m["patch_ratio"] - q_patch),
            "qhas_dominated": bool(m["patch_ratio"] <= q_patch
                                   and m["phys_score"] <= q_phys),
        })
        deltas.append(b["delta_phys_matched"])
    out = {"n_folds": len(rows), "rows": rows,
           "n_qhas_dominated": int(sum(r["qhas_dominated"] for r in rows))}
    if deltas:
        out["mean_delta_phys_matched"] = float(np.mean(deltas))
        # delta = qhas_phys - matched_phys ; > 0 => Q-HAS pire a cout egal
        out["n_qhas_worse_at_equal_budget"] = int(
            sum(x > 0 for x in deltas))
    return out


def format_table(records, primary, secondary):
    """Table markdown ; toute valeur affichee provient des JSON de fold."""
    L = []
    L.append("### Primary endpoint (pre-registered): paired `combined`")
    L.append("")
    L.append("| fold | scenario | Q-HAS combined | classical combined | "
             "delta (Q-HAS-cl) | better |")
    L.append("|---|---|---|---|---|---|")
    for r in records:
        q, c = r["qhas"]["combined"], r["classical"]["combined"]
        L.append(f"| {r['fold']} | {r['scenario']} | {q:.4f} | {c:.4f} | "
                 f"{q - c:+.4f} | {'Q-HAS' if q < c else 'classical'} |")
    L.append("")
    if primary.get("excluded_failed_folds"):
        L.append(f"- **excluded as failed** (pre-registration §5, an arm did "
                 f"not complete its trajectory): "
                 f"{', '.join(primary['excluded_failed_folds'])}")
    if not primary.get("audit_present", False):
        L.append("- **validity unaudited**: run `t19_arm_divergence_audit.py`"
                 " — an aborted arm is indistinguishable from a completed one"
                 " in the stored output")
    L.append(f"- folds usable: {primary['n_folds']}/4 — "
             f"Q-HAS better on {primary['n_qhas_better']}, "
             f"classical better on {primary['n_classical_better']} "
             f"(pre-registered rule: >= {WIN_RULE_MIN}/4)")
    if primary.get("tost"):
        t = primary["tost"]
        L.append(f"- TOST margin (5% of mean classical combined) = "
                 f"{primary['margin']:.4f}; diff = {t['diff']:+.4f}; "
                 f"p_TOST = {t['p_tost']:.4f} => equivalence "
                 f"{'ESTABLISHED' if t['equivalent'] else 'NOT established'}")
        L.append(f"- paired t p = {primary['paired_t_p']:.4f}; "
                 f"Holm-adjusted = "
                 f"{primary['holm']['p_adjusted'][0]:.4f} (difference), "
                 f"{primary['holm']['p_adjusted'][1]:.4f} (equivalence)")
        if primary.get("sign_test_p") is not None:
            L.append(f"- exact sign test p = {primary['sign_test_p']:.4f} "
                     f"(minimum attainable at n={primary['n_folds']}: "
                     f"{primary['sign_test_p_min_attainable']:.4f})")

    L.append("")
    L.append("### Secondary (post-hoc, defect D4): equal-budget comparison")
    L.append("")
    if secondary["n_folds"] == 0:
        L.append("_no fold has a budget-matched run yet_")
        return "\n".join(L)
    L.append("| fold | Q-HAS patch | Q-HAS phys | matched thr | "
             "matched patch | matched phys | Q-HAS/frontier | dominated? |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in secondary["rows"]:
        ratio = ("n/a" if r["ratio_vs_frontier"] is None
                 else f"{r['ratio_vs_frontier']:.2f}x")
        L.append(f"| {r['fold']} | {r['qhas_patch']:.4f} | "
                 f"{r['qhas_phys']:.4f} | {r['matched_threshold']:.4f} | "
                 f"{r['matched_patch']:.4f} | {r['matched_phys']:.4f} | "
                 f"{ratio} | {'yes' if r['qhas_dominated'] else 'no'} |")
    L.append("")
    L.append(f"- Q-HAS strictly Pareto-dominated on "
             f"{secondary['n_qhas_dominated']}/{secondary['n_folds']} "
             f"budget-matched folds")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(
        description="V4: cross-fold synthesis of the Level-3 closed loop")
    from config import RESULTS_DIR
    p.add_argument("--folds", nargs="+", default=list(ALL_FOLDS))
    p.add_argument("--prefix", default="t15_level3")
    p.add_argument("--results-dir", default=None)
    args = p.parse_args()
    results_dir = args.results_dir or RESULTS_DIR

    records, missing = [], []
    for f in args.folds:
        r = load_fold(results_dir, f, args.prefix)
        (records.append(r) if r is not None else missing.append(f))

    print("=" * 78)
    print("  V4 T15c - Level-3 cross-fold synthesis")
    print("=" * 78)
    if missing:
        print(f"  folds not yet available: {', '.join(missing)}")
    if not records:
        raise SystemExit("no Level-3 fold available; run t15 first.")

    # Pre-registration §5 : un fold dont un bras n'a pas fini sa
    # trajectoire est un ECHEC, pas un resultat. Sans lui, un bras
    # classique qui diverge se lit comme une victoire de Q-HAS.
    audit = load_divergence_audit(results_dir)
    excluded = []
    if audit is None:
        print("  WARNING: no T19 divergence audit found. V1 returns the same")
        print("  keys for an aborted and a completed run, so fold validity")
        print("  CANNOT be assumed. Run t19_arm_divergence_audit.py.")
    else:
        keep = []
        for r in records:
            if audit.get(r["fold"], True):
                keep.append(r)
            else:
                excluded.append(r["fold"])
        if excluded:
            print(f"  EXCLUDED as failed (pre-registration §5, an arm did "
                  f"not complete): {', '.join(excluded)}")
        unaudited = [r["fold"] for r in keep if r["fold"] not in audit]
        if unaudited:
            print(f"  WARNING: not audited, validity unknown: "
                  f"{', '.join(unaudited)}")
        records = keep
        if not records:
            raise SystemExit(
                "every available fold failed its divergence audit; "
                "no paired statistic is defined.")

    primary = primary_analysis(records)
    primary["excluded_failed_folds"] = excluded
    primary["audit_present"] = audit is not None
    secondary = secondary_analysis(records)
    table = format_table(records, primary, secondary)
    print()
    print(table)
    print()

    out = {
        "folds_available": [r["fold"] for r in records],
        "folds_missing": missing,
        "records": records,
        "primary": primary,
        "secondary": secondary,
        "git_hash": git_commit_hash(),
        "cli_args": vars(args),
    }
    path = os.path.join(results_dir, "t15c_fold_synthesis.json")
    os.makedirs(results_dir, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    md = os.path.join(results_dir, "t15c_fold_synthesis.md")
    open(md, "w").write(table + "\n")
    print(f"  saved: {os.path.basename(path)} / {os.path.basename(md)}")


if __name__ == "__main__":
    main()
