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


def main():
    p = argparse.ArgumentParser(
        description="V4 T19: did each Level-3 arm finish its trajectory?")
    from config import RESULTS_DIR

    p.add_argument("--folds", nargs="+",
                   default=["ot", "kh", "rotor", "tearing"])
    p.add_argument("--arms", nargs="+", default=["qhas", "classical"])
    p.add_argument("--prefix", default="t15_level3")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 84)
    print("  V4 T19 - Level-3 arm divergence audit")
    print("  V1 aborts on CFL>1 or non-finite fields and returns a PARTIAL")
    print("  score with identical keys. This replays each arm to detect it.")
    print("=" * 84)

    T = _load_v1_training_module()
    all_scen = fold_scenarios(T, warn=False)

    results, t0 = [], time.time()
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

    out = {
        "results": results,
        "usable_folds": usable,
        "failed_folds": failed,
        "git_hash": git_commit_hash(),
        "cli_args": vars(args),
        "wall_s": time.time() - t0,
    }
    path = os.path.join(RESULTS_DIR, "t19_arm_divergence_audit.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n  saved: {os.path.basename(path)} ({time.time() - t0:.0f}s)")
    print("\nV4 Task 19 complete.")


if __name__ == "__main__":
    main()
