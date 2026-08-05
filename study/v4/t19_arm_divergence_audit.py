#!/usr/bin/env python3
"""
V4 Task 19 - Audit : quels bras Level-3 ont REELLEMENT fini leur trajectoire ?

MOTIVATION. `src/pipeline.py` contient un garde-fou de divergence
(ligne ~499) : si le CFL depasse 1 ou si un champ devient non fini, la
simulation est INTERROMPUE et la fonction renvoie un score PARTIEL, calcule
a l'instant de l'arret. Le dictionnaire retourne a exactement les memes
cles qu'une execution complete : rien, dans la valeur de retour, ne
distingue une trajectoire terminee d'une trajectoire avortee.

Le pilote `t15` n'a donc aucun moyen de le voir — il ecarte meme
`field_errors`, la seule cle qui aurait pu servir d'indice. Un fold dont le
bras CLASSIQUE diverge apparait alors comme une VICTOIRE de Q-HAS, alors
qu'aucune comparaison n'a eu lieu.

C'est precisement le cas prevu par la pre-registration
(`docs/level3_preregistration.md` §5) :

  « Folds that fail to complete (divergence penalty, solver failure) are
    reported as failures and excluded from the paired statistics; the count
    is stated. »

...mais rien ne l'implementait. Cette tache fournit la detection.

METHODE. Pour chaque bras d'un fold deja calcule, on rejoue l'execution
avec `verbose=True` en capturant la sortie standard, et on cherche la
marque `[ABORT] Divergence detected at step N (t=...)` emise par V1. Le
pipeline etant deterministe (meme trace DNS, meme hot start, memes
hyperparametres), le rejeu reproduit l'execution d'origine ; on verifie
d'ailleurs que le `combined` rejoue coincide avec celui stocke, et on le
signale sinon.

V1 n'est ni modifie ni contourne : on lit sa propre trace d'execution.

Sortie : results/t19_arm_divergence_audit.json
Usage :
  python study/v4/t19_arm_divergence_audit.py --folds ot kh rotor tearing
  python study/v4/t19_arm_divergence_audit.py --folds rotor --arms classical
"""
import argparse, contextlib, io, json, os, re, sys, time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "v3"))
sys.path.insert(0, _HERE)

from t1_feature_selection import git_commit_hash
from t15_level3_closed_loop import (_load_v1_training_module, fold_scenarios,
                                    run_arm)

# Marques emises par le garde-fou de V1 (src/pipeline.py).
ABORT_RE = re.compile(r"\[ABORT\] Divergence detected at step (\d+) "
                      r"\(t=([-\d.eE+]+)\)")
DIVERGE_RE = re.compile(r"\[DIVERGE\] Partial score: combined=([-\d.eE+]+), "
                        r"diverged_fields=(\d+)/(\d+)")
# Penalite forfaitaire de V1 : un bras qui la renvoie n'a produit aucune
# information physique du tout.
DIVERGENCE_PENALTY = 10.0


def parse_abort(text):
    """Retourne None si la trajectoire est allee au bout, sinon un dict."""
    m = ABORT_RE.search(text)
    if not m:
        return None
    out = {"abort_step": int(m.group(1)), "abort_time": float(m.group(2))}
    d = DIVERGE_RE.search(text)
    if d:
        out["partial_combined"] = float(d.group(1))
        out["diverged_fields"] = int(d.group(2))
        out["n_fields"] = int(d.group(3))
    return out


def audit_arm(T, key, cfg, dns_held, hyperparams, classical_only):
    """Rejoue un bras en capturant la trace de V1.

    La sortie de V1 est volumineuse : on la capture puis on ne conserve que
    la marque d'arret. Rien n'est reinjecte dans stdout.
    """
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        res = run_arm(T, key, cfg, dns_held, hyperparams, classical_only,
                      verbose=True)
    txt = buf.getvalue()
    ab = parse_abort(txt)
    return {
        "completed": ab is None,
        "abort": ab,
        "combined": float(res.get("combined", np.nan)),
        "phys_score": float(res.get("phys_score", np.nan)),
        "patch_ratio": float(res.get("patch_ratio", np.nan)),
        "flat_penalty": bool(
            abs(float(res.get("phys_score", np.nan)) - DIVERGENCE_PENALTY)
            < 1e-9),
        "replay_wall_s": time.time() - t0,
    }


def audit_fold(T, all_scen, results_dir, fold, arms=("qhas", "classical"),
               prefix="t15_level3"):
    path = os.path.join(results_dir, f"{prefix}_fold_{fold}.json")
    if not os.path.exists(path):
        return None
    rec = json.load(open(path))
    cfg = dict(all_scen)[fold]
    dns_held = T._precompute_dns_for([(fold, cfg)], label=f"audit/{fold}")

    hp_q = dict(rec["hyperparams"])
    hp_c = dict(rec["hyperparams"])
    hp_c.update(rec["classical_params"])

    out = {"fold": fold, "scenario": rec["scenario"], "arms": {}}
    for arm in arms:
        hp = hp_q if arm == "qhas" else hp_c
        a = audit_arm(T, fold, cfg, dns_held, hp, arm == "classical")
        stored = rec["qhas" if arm == "qhas" else "classical"]
        a["stored_combined"] = float(stored.get("combined", np.nan))
        # le rejeu doit reproduire l'execution d'origine ; sinon la lecture
        # de cet audit ne vaut rien et il faut le dire
        a["replay_matches_stored"] = bool(
            np.isfinite(a["combined"]) and np.isfinite(a["stored_combined"])
            and abs(a["combined"] - a["stored_combined"]) < 1e-6)
        out["arms"][arm] = a
    out["fold_usable"] = all(v["completed"] for v in out["arms"].values())
    return out


def audit_budget_trace(T, all_scen, results_dir, fold):
    """Audite les points de la trace de bissection t15b.

    Enjeu : la courbe tracee par les figures est presentee comme la
    frontiere ATTEIGNABLE du cout classique. Un point issu d'une
    trajectoire avortee n'est pas un point de fonctionnement — l'inclure
    fait passer un plantage pour une option disponible, et deforme
    l'echelle. Les executions divergentes s'arretant tot, cet audit est
    bon marche.
    """
    p = os.path.join(results_dir, f"t15b_budget_matched_{fold}.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p))
    rec = json.load(open(os.path.join(
        results_dir, f"t15_level3_fold_{fold}.json")))
    cfg = dict(all_scen)[fold]
    dns_held = T._precompute_dns_for([(fold, cfg)], label=f"trace/{fold}")

    pts = []
    for r in b["trace"]:
        hp = dict(rec["hyperparams"])
        hp["threshold_amr"] = r["threshold"]
        a = audit_arm(T, fold, cfg, dns_held, hp, True)
        pts.append({
            "threshold": r["threshold"],
            "patch_ratio": r["patch_ratio"],
            "phys_score": r["phys_score"],
            "stored_wall_s": r.get("wall_s"),
            "completed": a["completed"],
            "abort": a["abort"],
            "replay_phys": a["phys_score"],
        })
        tag = "ok" if a["completed"] else "ABORTED"
        print(f"    thr={r['threshold']:.4f}  patch={r['patch_ratio']:.4f}  "
              f"phys={r['phys_score']:.4f}  -> {tag}", flush=True)
    return {"fold": fold, "points": pts,
            "n_aborted": int(sum(not x["completed"] for x in pts))}


def main():
    p = argparse.ArgumentParser(
        description="V4 T19: did each Level-3 arm finish its trajectory?")
    from config import RESULTS_DIR

    p.add_argument("--folds", nargs="+",
                   default=["ot", "kh", "rotor", "tearing"])
    p.add_argument("--arms", nargs="+", default=["qhas", "classical"])
    p.add_argument("--prefix", default="t15_level3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trace-only", action="store_true",
                   help="n'auditer que les points de bissection t15b")
    args = p.parse_args()

    print("=" * 84)
    print("  V4 T19 - Level-3 arm divergence audit")
    print("  V1 aborts on CFL>1 or non-finite fields and returns a PARTIAL")
    print("  score with identical keys. This replays each arm to detect it.")
    print("=" * 84)

    T = _load_v1_training_module()
    all_scen = fold_scenarios(T, warn=False)

    results, t0 = [], time.time()
    if args.trace_only:
        traces = []
        for f in args.folds:
            print(f"\n  fold {f}: auditing t15b bisection points")
            tr = audit_budget_trace(T, all_scen, RESULTS_DIR, f)
            if tr is None:
                print("    no t15b output, skipped")
                continue
            traces.append(tr)
            print(f"    -> {tr['n_aborted']}/{len(tr['points'])} points "
                  f"came from an aborted run")
        tp = os.path.join(RESULTS_DIR, "t19_budget_trace_audit.json")
        json.dump({"traces": traces, "git_hash": git_commit_hash(),
                   "cli_args": vars(args), "wall_s": time.time() - t0},
                  open(tp, "w"), indent=1)
        print(f"\n  saved: {os.path.basename(tp)}")
        return
    for f in args.folds:
        r = audit_fold(T, all_scen, RESULTS_DIR, f, tuple(args.arms),
                       args.prefix)
        if r is None:
            print(f"\n  {f}: fold not computed yet, skipped")
            continue
        results.append(r)
        print(f"\n  fold {f} ({r['scenario']})")
        for arm, a in r["arms"].items():
            if a["completed"]:
                print(f"    {arm:<10} COMPLETED   combined={a['combined']:.4f}"
                      f"  phys={a['phys_score']:.4f}")
            else:
                ab = a["abort"]
                print(f"    {arm:<10} **ABORTED** at step {ab['abort_step']} "
                      f"(t={ab['abort_time']:.4f})  "
                      f"partial phys={a['phys_score']:.4f}"
                      f"{'  [FLAT PENALTY]' if a['flat_penalty'] else ''}")
            if not a["replay_matches_stored"]:
                print(f"      WARNING: replay {a['combined']:.6f} != stored "
                      f"{a['stored_combined']:.6f} — audit unreliable "
                      f"for this arm")
        print(f"    -> fold {'USABLE' if r['fold_usable'] else 'NOT USABLE'}"
              f" for the paired statistics")

    if not results:
        raise SystemExit("no computed fold to audit")

    usable = [r["fold"] for r in results if r["fold_usable"]]
    failed = [r["fold"] for r in results if not r["fold_usable"]]
    print("\n  " + "-" * 80)
    print(f"  usable folds : {', '.join(usable) if usable else 'none'}")
    print(f"  failed folds : {', '.join(failed) if failed else 'none'}")
    print("  Pre-registration §5: failed folds are reported as failures and")
    print("  EXCLUDED from the paired statistics, with the count stated.")

    path = os.path.join(RESULTS_DIR, "t19_arm_divergence_audit.json")

    # FUSION, jamais ecrasement. Ecrire tel quel ferait perdre les folds
    # audites lors d'un appel precedent des qu'on relance la tache sur un
    # sous-ensemble — c'est exactement le defaut D9, ici dans du code ecrit
    # APRES l'avoir diagnostique. Un audit qui perd silencieusement ses
    # propres resultats est pire qu'absent : les folds disparus
    # redeviennent « non audites », donc traites comme valides par defaut.
    merged = {}
    if os.path.exists(path):
        try:
            prev = json.load(open(path))
            for r in prev.get("results", []):
                merged[r["fold"]] = r
        except (ValueError, KeyError):
            print("  WARNING: existing audit unreadable, starting fresh")
    replaced = [r["fold"] for r in results if r["fold"] in merged]
    for r in results:
        merged[r["fold"]] = r
    if replaced:
        print(f"  refreshed folds: {', '.join(replaced)}")
    kept = [f for f in merged if f not in {r['fold'] for r in results}]
    if kept:
        print(f"  preserved from earlier audits: {', '.join(sorted(kept))}")

    all_results = [merged[f] for f in sorted(merged)]
    out = {
        "results": all_results,
        "usable_folds": [r["fold"] for r in all_results if r["fold_usable"]],
        "failed_folds": [r["fold"] for r in all_results
                         if not r["fold_usable"]],
        "git_hash": git_commit_hash(),
        "cli_args": vars(args),
        "wall_s": time.time() - t0,
    }
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n  saved: {os.path.basename(path)} ({time.time() - t0:.0f}s)")
    print("\nV4 Task 19 complete.")


if __name__ == "__main__":
    main()


def safe_classical_hyperparams(rec, results_dir, fold):
    """Hyperparametres du bras classique SANS point de fonctionnement divergent.

    Le seuil classique regle d'un fold peut tomber dans une bande instable :
    sur `rotor` il vaut 0.4616, entre les deux points de bissection qui
    avortent (0.4250 et 0.8000), et le bras diverge alors au pas 208. Le
    comparer a Q-HAS produit des ecarts enormes et vides de sens — le piege
    a mordu T15 (le fold comptait comme une victoire Q-HAS), T20 (gap/sd
    15.9) puis T22 (comparaison sur une trajectoire tronquee).

    Cette fonction centralise la regle : si l'audit T19 dit que le bras
    classique regle a termine, on prend ses parametres ; sinon on prend le
    seuil BUDGET-APPARIE, dont l'audit de trace a verifie qu'il termine.

    Retourne (hyperparams, source, tuned_arm_completed).
    """
    hp = dict(rec["hyperparams"])
    hp.update(rec.get("classical_params", {}))
    completed = None
    apath = os.path.join(results_dir, "t19_arm_divergence_audit.json")
    if os.path.exists(apath):
        try:
            au = json.load(open(apath))
            for r in au.get("results", []):
                if r["fold"] == fold:
                    completed = bool(r["arms"]["classical"]["completed"])
        except (ValueError, KeyError):
            pass
    if completed is False:
        bpath = os.path.join(results_dir,
                             f"t15b_budget_matched_{fold}.json")
        if os.path.exists(bpath):
            thr = float(json.load(open(bpath))
                        ["matched_classical"]["threshold"])
            hp["threshold_amr"] = thr
            return hp, f"budget-matched threshold {thr:.4f} " \
                       f"(tuned arm ABORTED)", completed
        return hp, "tuned arm ABORTED and no t15b available", completed
    src = ("tuned classical arm" if completed
           else "tuned classical arm [T19 audit absent]")
    return hp, src, completed
