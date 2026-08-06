#!/usr/bin/env python3
"""
V4 Task 20 - Variance d'execution du bras Q-HAS (defaut D11).

CE QUI A DECLENCHE CETTE TACHE. L'audit T19 rejoue chaque bras d'un fold
Level-3 avec des entrees identiques et verifie qu'il reproduit la valeur
stockee. Le bras CLASSIQUE reproduit au bit pres. Le bras Q-HAS, non :
sur le fold `ot`, le rejeu donne combined = 0.3108 / phys = 0.1345 la ou
l'execution d'origine avait donne 0.3328 / 0.1940 — soit 44 % d'ecart sur
la fidelite, a entrees stricitement identiques.

CAUSE. Aucun germe n'est fixe dans toute la chaine VQA de V1 :
`AerSimulator` est construit sans `seed_simulator`, et `Estimator` comme
`Sampler` tournent a `default_shots = 256`. Le bras Q-HAS est donc
doublement stochastique :
  1. l'objectif optimise par COBYLA est estime sur 256 tirages, donc
     l'optimiseur suit une trajectoire differente a chaque execution ;
  2. la lecture finale des marginales est elle-meme un tirage a 256 coups.
Le bras classique ne comporte aucun echantillonnage : d'ou son
determinisme, qui sert ici de CONTROLE de la chaine de mesure.

Consequence : chaque nombre Q-HAS publie au niveau 3 est UN tirage d'une
distribution dont la dispersion n'a jamais ete mesuree. Aucune comparaison
mono-execution n'est interpretable tant que cette dispersion est inconnue.

CE QUE MESURE LA TACHE. K executions du bras Q-HAS sur la classe tenue,
entrees identiques (meme trace DNS, meme hot start, memes hyperparametres),
puis K executions du bras classique comme controle de determinisme. On
rapporte moyenne, ecart-type, etendue, et surtout la comparaison de cette
dispersion a l'ECART mesure entre les deux bras : si l'ecart-type de Q-HAS
est du meme ordre que l'ecart Q-HAS/classique, la conclusion du fold ne
tient pas sur une seule execution.

V1 n'est pas modifie : on ne peut pas fixer le germe sans le toucher, et
c'est precisement le defaut a documenter.

Sortie : results/t20_qhas_run_variance_{fold}.json
Usage :
  python study/v4/t20_qhas_run_variance.py --fold kh --repeats 5
"""
import argparse, contextlib, io, json, os, sys, time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "v3"))
sys.path.insert(0, _HERE)

from t1_feature_selection import git_commit_hash
from t15_level3_closed_loop import (_load_v1_training_module, fold_scenarios,
                                    run_arm)
from t19_arm_divergence_audit import parse_abort

METRICS = ("combined", "phys_score", "patch_ratio")


def summarise(runs):
    """Statistiques descriptives par metrique."""
    out = {}
    for m in METRICS:
        v = np.array([r[m] for r in runs], dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            out[m] = None
            continue
        out[m] = {
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "range": float(np.max(v) - np.min(v)),
            "cv": (float(np.std(v, ddof=1) / abs(np.mean(v)))
                   if v.size > 1 and np.mean(v) != 0 else None),
            "values": v.tolist(),
        }
    return out


def main():
    p = argparse.ArgumentParser(
        description="V4 T20: run-to-run variance of the Q-HAS arm")
    from config import RESULTS_DIR

    p.add_argument("--fold", default="kh")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--classical-repeats", type=int, default=2,
                   help="controle de determinisme ; 2 suffisent a le montrer")
    p.add_argument("--prefix", default="t15_level3")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    np.random.seed(args.seed)   # sans effet sur Aer : c'est le point

    path = os.path.join(RESULTS_DIR, f"{args.prefix}_fold_{args.fold}.json")
    if not os.path.exists(path):
        raise SystemExit(f"fold {args.fold} not computed yet ({path})")
    rec = json.load(open(path))

    print("=" * 84)
    print(f"  V4 T20 - Q-HAS run-to-run variance, fold {args.fold}")
    print(f"  {args.repeats} Q-HAS runs + {args.classical_repeats} classical "
          f"(determinism control), identical inputs")
    print("  V1 fixes no RNG seed: Estimator/Sampler run at 256 shots.")
    print("=" * 84, flush=True)

    T = _load_v1_training_module()
    all_scen = fold_scenarios(T, warn=False)
    cfg = dict(all_scen)[args.fold]
    dns_held = T._precompute_dns_for([(args.fold, cfg)],
                                     label=f"variance/{args.fold}")

    hp_q = dict(rec["hyperparams"])
    # Le CONTROLE classique doit tourner a un point de fonctionnement QUI
    # TERMINE. Le seuil regle diverge sur `rotor` : les deux executions de
    # controle avortaient, `c_ok` etait vide et la tache s'arretait sans
    # rien sauvegarder — T20 ne pouvait donc structurellement pas aboutir
    # sur ce fold. On prend le seuil budget-apparie, le meme que celui deja
    # utilise comme valeur de reference.
    from t19_arm_divergence_audit import safe_classical_hyperparams
    hp_c, hp_c_src, _ = safe_classical_hyperparams(
        rec, RESULTS_DIR, args.fold, always_matched=True)
    print(f"  classical control runs at: {hp_c_src}", flush=True)

    t0 = time.time()
    def guarded(hp, only):
        """Une execution, avec son statut d'avortement CAPTURE.

        Indispensable ici : le bras Q-HAS n'est pas deterministe (D11), donc
        un tirage divergent ne peut pas etre identifie apres coup en le
        rejouant. T22 l'a paye — un tirage a phys=0.601 contre 0.002 chez
        ses voisins avait ete moyenne avec eux.
        """
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = run_arm(T, args.fold, cfg, dns_held, hp, only, verbose=True)
        ab = parse_abort(buf.getvalue())
        # V1 en mode verbeux ouvre des figures matplotlib et ne les
        # ferme jamais (src/visual.py). Sur des dizaines d'executions
        # cela epuise la memoire et le processus est tue. Le mode
        # verbeux est pourtant obligatoire ici : c'est lui qui emet le
        # marqueur d'avortement qu'on capture.
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception:
            pass
        d = {m: float(r.get(m, np.nan)) for m in METRICS}
        d["completed"] = ab is None
        d["abort"] = ab
        return d

    q_runs = []
    for i in range(args.repeats):
        q_runs.append(guarded(hp_q, False))
        print(f"  Q-HAS run {i + 1}/{args.repeats}: "
              f"combined={q_runs[-1]['combined']:.4f} "
              f"phys={q_runs[-1]['phys_score']:.4f} "
              f"patch={q_runs[-1]['patch_ratio']:.4f}"
              f"{'' if q_runs[-1]['completed'] else '   **ABORTED**'}",
              flush=True)

    c_runs = []
    for i in range(args.classical_repeats):
        c_runs.append(guarded(hp_c, True))
        print(f"  classical run {i + 1}/{args.classical_repeats}: "
              f"combined={c_runs[-1]['combined']:.4f} "
              f"phys={c_runs[-1]['phys_score']:.4f}", flush=True)

    # une trajectoire avortee n'est pas un point de mesure
    n_ab = sum(1 for r in q_runs + c_runs if not r["completed"])
    if n_ab:
        print(f"\n  {n_ab} run(s) ABORTED — excluded from the statistics",
              flush=True)
    q_ok = [r for r in q_runs if r["completed"]]
    c_ok = [r for r in c_runs if r["completed"]]
    if len(q_ok) < 2 or not c_ok:
        raise SystemExit("too few completed runs to summarise")
    q_stats, c_stats = summarise(q_ok), summarise(c_ok)

    print("\n  " + "-" * 80)
    print(f"  {'metric':<14}{'Q-HAS mean':>12}{'std':>10}{'range':>10}"
          f"{'CV':>8}   | classical range")
    for m in METRICS:
        qs, cs = q_stats[m], c_stats[m]
        cv = "n/a" if not qs or qs["cv"] is None else f"{qs['cv']:.3f}"
        print(f"  {m:<14}{qs['mean']:>12.4f}{qs['std']:>10.4f}"
              f"{qs['range']:>10.4f}{cv:>8}   | {cs['range']:.2e}")

    # Le controle : le bras classique doit etre EXACTEMENT reproductible.
    c_det = all(c_stats[m]["range"] == 0.0 for m in METRICS)
    print(f"\n  classical arm deterministic: {c_det}  "
          f"(control — if False, the measurement chain itself is suspect)")

    # La question qui decide de la lisibilite du fold.
    #
    # ATTENTION AU BIAIS : mesurer l'ecart a partir de la valeur STOCKEE du
    # fold revient a utiliser UN tirage, et rien ne garantit qu'il soit
    # representatif — sur `kh` il s'est trouve etre le maximum des six
    # tirages connus, ce qui gonfle mecaniquement l'ecart. On rapporte donc
    # les DEUX lectures, et c'est celle fondee sur la MOYENNE qui fait foi.
    # QUELLE reference classique ? Si le bras classique du fold a AVORTE
    # (audit T19), sa valeur est un score partiel de trajectoire tronquee :
    # la comparer a Q-HAS n'a aucun sens et produit un ecart enorme et
    # trompeur — sur `rotor`, gap/sd = 15.9 contre une execution plantee.
    # Dans ce cas on prend le point classique BUDGET-APPARIE, dont l'audit
    # a verifie qu'il termine.
    # La reference est TOUJOURS le point classique BUDGET-APPARIE, jamais
    # le bras classique regle. Deux raisons distinctes :
    #   - si le bras regle a avorte (audit T19), sa valeur est un score
    #     partiel : sur `rotor` la comparer donnait gap/sd = 15.9 contre une
    #     execution plantee ;
    #   - meme quand il termine, il tourne a un AUTRE budget (defaut D4) :
    #     sur `ot` il refine 0.324 contre 0.680 pour Q-HAS, et l'ecart
    #     mesure alors le point de fonctionnement, pas la regle de decision.
    # Seul le point apparie compare les deux bras a cout egal.
    stored_q = rec["qhas"]["phys_score"]
    stored_c = rec["classical"]["phys_score"]
    ref_source = "tuned classical arm"
    audit_path = os.path.join(RESULTS_DIR, "t19_arm_divergence_audit.json")
    classical_completed = None
    if os.path.exists(audit_path):
        try:
            au = json.load(open(audit_path))
            for r in au.get("results", []):
                if r["fold"] == args.fold:
                    classical_completed = bool(
                        r["arms"]["classical"]["completed"])
        except (ValueError, KeyError) as exc:
            # NE PAS avaler : un audit illisible ferait retomber sur
            # `completed = None`, c'est-a-dire « bras suppose termine » —
            # exactement la defaillance silencieuse que cet audit existe
            # pour empecher. On le signale bruyamment.
            print(f"  WARNING: divergence audit unreadable ({exc}); the "
                  f"tuned arm's completion is UNVERIFIED", flush=True)
    bpath = os.path.join(
        RESULTS_DIR, f"t15b_budget_matched_{args.fold}.json")
    if os.path.exists(bpath):
        stored_c = float(json.load(open(bpath))
                         ["matched_classical"]["phys_score"])
        ref_source = ("budget-matched classical"
                      + ("" if classical_completed is not False
                         else " (tuned arm ABORTED)"))
    elif classical_completed is False:
        stored_c = float("nan")
        ref_source = "UNAVAILABLE (tuned arm aborted, no t15b)"
    print(f"\n  classical reference: {ref_source}"
          f"{'' if classical_completed is not None else '  [T19 audit absent]'}")

    sd = q_stats["phys_score"]["std"]
    mean_q = q_stats["phys_score"]["mean"]
    gap_stored = abs(stored_q - stored_c)
    gap_mean = abs(mean_q - stored_c)
    r_stored = gap_stored / sd if sd > 0 else float("inf")
    r_mean = gap_mean / sd if sd > 0 else float("inf")

    draws = np.array(q_stats["phys_score"]["values"], dtype=float)
    pct = 100.0 * float(np.mean(np.append(draws, stored_q) <= stored_q))

    print(f"\n  Q-HAS run-to-run std on phys : {sd:.5f}  "
          f"(CV {q_stats['phys_score']['cv']:.3f})")
    print(f"  stored fold value {stored_q:.5f} sits at the {pct:.0f}th "
          f"percentile of the draws")
    print(f"  gap / std, from the STORED draw : {r_stored:.2f}")
    print(f"  gap / std, from the MEAN draw   : {r_mean:.2f}   <- the one "
          f"to quote")
    if r_mean < 2.0:
        print("  => against sampling noise the between-arm gap is NOT "
              "large;\n     a single run per arm cannot support a "
              "directional claim on\n     the magnitude. Report the "
              "dominance count instead.")
    else:
        print("  => the between-arm gap is large against sampling noise; "
              "the\n     direction survives, though the magnitude carries "
              "this spread.")
    ratio = r_mean

    out = {
        "fold": args.fold,
        "scenario": rec["scenario"],
        "repeats": args.repeats,
        "n_aborted": int(n_ab),
        "n_completed_qhas": len(q_ok),
        "qhas_runs": q_runs,
        "classical_runs": c_runs,
        "qhas_stats": q_stats,
        "classical_stats": c_stats,
        "classical_deterministic": bool(c_det),
        "stored_qhas_phys": float(stored_q),
        "stored_classical_phys": float(stored_c),
        "gap_phys_from_stored": float(gap_stored),
        "gap_phys_from_mean": float(gap_mean),
        "gap_over_std_from_stored": float(r_stored),
        "gap_over_std_from_mean": float(r_mean),
        "gap_over_std": float(ratio),          # = version moyenne
        "stored_percentile_among_draws": float(pct),
        "classical_reference_source": ref_source,
        "classical_arm_completed": classical_completed,
        "shots": cfg.get("shots"),
        "git_hash": git_commit_hash(),
        "cli_args": vars(args),
        "wall_s": time.time() - t0,
    }
    op = os.path.join(RESULTS_DIR,
                      f"t20_qhas_run_variance_{args.fold}.json")
    json.dump(out, open(op, "w"), indent=1)
    print(f"\n  saved: {os.path.basename(op)} ({time.time() - t0:.0f}s)")
    print("\nV4 Task 20 complete.")


if __name__ == "__main__":
    main()
