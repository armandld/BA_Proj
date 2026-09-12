"""V4 Tache 24 — ce que devient Q-HAS quand on retire la fuite D13.

CE QUE MESURE `--mode leak-free`, ET CE QU'IL NE MESURE PAS
-----------------------------------------------------------
Le mode remplace le seuil FUITE du bras QAOA (0.14959824837662078, ajuste
sur les quatre classes y compris la tenue) par le seuil classique REGLE du
fold, issu de `train_classical_threshold_excluding` donc ajuste sur les
seules classes d'entrainement. La fuite disparait.

Mais la substitution ne re-regle PAS le bras QAOA. Un test definitif
remettrait `threshold_amr` dans l'espace de recherche Optuna, hors classe
tenue — c'est l'experience declaree non tentee. Ce que cette tache mesure
est donc une BORNE : Q-HAS survit-il au retrait du seuil fuite sans
re-reglage ? Pas : quel est le meilleur Q-HAS sans fuite ?

LE PIEGE A EVITER ICI
---------------------
Le bras QAOA et le bras classique de controle NE TOURNENT PAS au meme
seuil (`--matched-reference` force le controle au point budget-apparie).
Sur `rotor` : 0.5864 contre 0.0969. Comparer directement leurs erreurs
melangerait donc l'effet de la regle de decision et celui du budget.

La comparaison honnete passe par la FRONTIERE classique de T15b,
interpolee au budget que Q-HAS a REELLEMENT realise. Quand ce budget tombe
hors de la plage balayee, aucune interpolation n'est legitime et la tache
le dit au lieu d'extrapoler.
"""
import argparse
import json
import os
import sys

import numpy as np

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

import provenance

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
def frontier(results_dir, fold):
    """(patch, phys) tries de la trace de bissection classique de T15b."""
    p = os.path.join(results_dir, f"t15b_budget_matched_{fold}.json")
    if not os.path.exists(p):
        return None
    tr = sorted(json.load(open(p))["trace"], key=lambda r: r["patch_ratio"])
    return ([r["patch_ratio"] for r in tr], [r["phys_score"] for r in tr])


def frontier_at(front, patch):
    """Erreur classique au budget `patch`, ou None HORS de la plage balayee.

    On refuse d'extrapoler : sur `rotor` le budget realise sans fuite
    (0.056-0.138) tombe sous le point le plus bas de la frontiere (0.152),
    et `np.interp` y rendrait sans broncher la valeur du bord — un nombre
    d'apparence normale pour une comparaison qui n'existe pas.
    """
    xs, ys = front
    if patch < xs[0] or patch > xs[-1]:
        return None
    return float(np.interp(patch, xs, ys))


def analyse(results_dir, fold):
    p = os.path.join(results_dir, f"t22_unseen_leak-free_{fold}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    # Un point de reprise n'est PAS un resultat. Ses moyennes portent sur
    # les tirages faits jusque-la et n'ont aucune validite. Les lire comme
    # les autres serait le motif de la campagne introduit par la mesure
    # censee l'eviter.
    if d.get("status") == "partial":
        return {"fold": fold, "status": "partial",
                "partial_stage": d.get("partial_stage"),
                "qaoa_threshold": None, "classical_threshold": None,
                "thresholds_match": None, "conditions": {}}
    fr = frontier(results_dir, fold)
    out = {
        "fold": fold,
        "status": d.get("status"),
        "qaoa_threshold": d.get("qaoa_threshold_amr"),
        "classical_threshold": d.get("classical_threshold_amr"),
        "thresholds_match": d.get("thresholds_match"),
        "conditions": {},
    }
    # Les artefacts produits avant l'ajout des champs de seuil ne les
    # portent pas. On les RECONSTITUE depuis leurs sources d'origine
    # plutot que d'afficher un trou : en mode leak-free le seuil QAOA est
    # `classical_params.threshold_amr` du fold, et le controle tourne au
    # seuil budget-apparie de T15b.
    if out["qaoa_threshold"] is None:
        fp = os.path.join(results_dir, f"t15_level3_fold_{fold}.json")
        if os.path.exists(fp):
            out["qaoa_threshold"] = float(json.load(open(fp))
                                          ["classical_params"]["threshold_amr"])
        out["threshold_source"] = "reconstructed from t15 fold record"
    if out["classical_threshold"] is None:
        bp = os.path.join(results_dir, f"t15b_budget_matched_{fold}.json")
        if os.path.exists(bp):
            out["classical_threshold"] = float(json.load(open(bp))
                                               ["matched_classical"]["threshold"])
    if out["thresholds_match"] is None and None not in (
            out["qaoa_threshold"], out["classical_threshold"]):
        out["thresholds_match"] = bool(
            abs(out["qaoa_threshold"] - out["classical_threshold"]) < 1e-12)

    q, c = d["arms"]["qhas"], d["arms"]["classical"]
    for cond in ("canonical", "unseen"):
        runs = [r for r in q.get(f"{cond}_runs", []) if r["completed"]]
        n_all = len(q.get(f"{cond}_runs", []))
        rec = {"n_runs": n_all, "n_completed": len(runs),
               "n_aborted": n_all - len(runs)}
        if runs:
            qp = float(np.mean([r["patch_ratio"] for r in runs]))
            qe = float(np.mean([r["phys_score"] for r in runs]))
            rec.update(qhas_patch=qp, qhas_phys=qe)
            ref = frontier_at(fr, qp) if fr else None
            rec["frontier_phys_at_qhas_budget"] = ref
            rec["ratio_vs_frontier"] = (qe / ref if ref else None)
            rec["out_of_swept_range"] = ref is None
            if c.get("status") == "completed":
                rec["classical_patch"] = c[cond]["patch_ratio"]
                rec["classical_phys"] = c[cond]["phys_score"]
        out["conditions"][cond] = rec
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    from config import FOLD_KEYS
    p.add_argument("--folds", nargs="+", default=list(FOLD_KEYS))
    p.add_argument("--seed", type=int, default=0,
                   help="sans effet : relit des artefacts, ne simule rien")
    args = p.parse_args()
    prov = provenance.start()

    rows, missing = [], []
    for f in args.folds:
        r = analyse(RESULTS_DIR, f)
        (rows.append(r) if r else missing.append(f))

    print("=" * 84)
    print("  V4 T24 - Q-HAS with the D13 leak removed")
    print("=" * 84)
    if missing:
        print(f"  not run yet: {', '.join(missing)}")
    if not rows:
        raise SystemExit("no leak-free artifact; run t22 --mode leak-free")

    print("\n  The QAOA arm runs at the fold's own classical tuned threshold")
    print("  (training classes only). The classical control runs at the")
    print("  budget-matched threshold. THESE DIFFER — the comparison that")
    print("  controls for budget is against the T15b frontier interpolated")
    print("  at the budget Q-HAS actually realised.\n")

    partial = [r for r in rows if r.get("status") == "partial"]
    rows = [r for r in rows if r.get("status") != "partial"]
    for r in partial:
        print(f"  {r['fold']}: INCOMPLETE (checkpoint after "
              f"{r['partial_stage']}) — not analysed; the run was "
              f"interrupted and its partial means are not results")
    if not rows:
        raise SystemExit("no COMPLETE leak-free artifact yet")

    for r in rows:
        qt = ("%.4f" % r["qaoa_threshold"] if r["qaoa_threshold"] is not None
              else "?")
        ct = ("%.4f" % r["classical_threshold"]
              if r["classical_threshold"] is not None else "?")
        note = (" [thresholds reconstructed]"
                if r.get("threshold_source") else "")
        print(f"  {r['fold']}  (QAOA thr {qt} vs "
              f"classical control {ct}){note}")
        for cond, rec in r["conditions"].items():
            if not rec["n_completed"]:
                print(f"    {cond:9s}: ALL {rec['n_runs']} draws ABORTED — "
                      f"no operating point")
                continue
            line = (f"    {cond:9s}: n={rec['n_completed']}"
                    f"/{rec['n_runs']}  patch={rec['qhas_patch']:.4f}  "
                    f"phys={rec['qhas_phys']:.5f}")
            if rec.get("out_of_swept_range"):
                line += "   frontier: BUDGET OUTSIDE SWEPT RANGE, no ratio"
            elif rec.get("ratio_vs_frontier"):
                line += (f"   frontier={rec['frontier_phys_at_qhas_budget']:.5f}"
                         f"  ratio={rec['ratio_vs_frontier']:.1f}x")
            print(line)
        print()

    out = {"folds": rows, "missing": missing, "cli_args": vars(args)}
    out.update(provenance.finish(prov))
    op = os.path.join(RESULTS_DIR, "t24_leak_free_summary.json")
    json.dump(out, open(op, "w"), indent=1)
    print(f"  saved: {os.path.basename(op)}")
    print("\nV4 Task 24 complete.")


if __name__ == "__main__":
    main()
