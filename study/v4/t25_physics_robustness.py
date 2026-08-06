#!/usr/bin/env python3
"""
V4 Tache 25 - Robustesse de la direction a la PHYSIQUE (le « >=3 seeds »).

CE QUE J'AI TROUVE EN VOULANT FAIRE VARIER LA GRAINE
----------------------------------------------------
La pre-inscription demande >= 3 graines physiques par classe, et l'etude
declare partout « 1 seed par classe » comme limite. Les DEUX enonces sont
mal specifies : dans trois classes sur quatre il n'existe aucune graine.

  init_kelvin_helmholtz   aucun RNG. `noise_amplitude` multiplie sin(X),
                          c'est un MODE deterministe, pas du bruit.
  init_harris_tearing     aucun RNG. `perturbation` multiplie cos(k*X).
  init_orszag_tang        aucun RNG, et aucun parametre.
  init_mhd_rotor          RNG reel, mais `np.random.default_rng(42)` est
                          ECRIT EN DUR — la graine n'est pas un parametre.

Donc « relancer avec d'autres graines » ne peut pas etre fait tel quel :
il n'y a rien a faire varier dans trois classes, et dans la quatrieme il
faudrait modifier `src/`, qui est en lecture seule.

CE QUE CETTE TACHE FAIT A LA PLACE
----------------------------------
Ce que la pre-inscription VOULAIT est que la direction du resultat ne
tienne pas a un etat initial arbitraire. Le levier qui existe reellement,
c'est le PARAMETRE physique de la condition initiale. On evalue donc chaque
fold sur plusieurs conditions initiales distinctes, et pour `rotor` on fait
en plus varier sa vraie graine RNG en substituant temporairement
`np.random.default_rng` — sans toucher a V1.

`orszag_tang` n'expose aucun parametre : son seul levier est le nombre de
Reynolds, ce qui est declare comme tel et NON compte comme une variation de
condition initiale.

LA MESURE
---------
Pour chaque (fold, condition) : n tirages Q-HAS (avec la garde de
divergence) et une petite FRONTIERE classique de quelques seuils encadrant
le budget realise par Q-HAS. La comparaison se fait contre la frontiere
interpolee a ce budget — jamais bras contre bras, dont les budgets
different (lecon de T24). Hors de la plage balayee : aucun rapport.

La sortie est un DECOMPTE DE DIRECTION : sur combien de (fold, condition)
Q-HAS est-il au-dessus de la frontiere classique a budget egal ? C'est la
question a laquelle plusieurs etats initiaux peuvent repondre ; la
magnitude, elle, reste hors de portee (n petit, cf. T20).

Sortie : results/t25_physics_robustness_{fold}.json
Usage :
  python study/v4/t25_physics_robustness.py --fold kh --repeats 3
"""
import argparse, contextlib, io, json, os, sys, time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "v3"))
sys.path.insert(0, _HERE)

import provenance
from t15_level3_closed_loop import (_load_v1_training_module, fold_scenarios,
                                    run_arm)
from t19_arm_divergence_audit import parse_abort, safe_classical_hyperparams
from t22_unseen_conditions import init_override

# Conditions initiales SUPPLEMENTAIRES, distinctes de la canonique ET de
# celle de T22. Elles restent dans un domaine physiquement raisonnable :
# c'est la meme instabilite, vue depuis un autre etat initial.
#
#   'ic'   -> arguments passes a la fonction init_* de V1
#   'rng'  -> graine substituee a `np.random.default_rng` (rotor seulement)
#   're'   -> Reynolds substitue (orszag_tang seulement, faute de mieux)
CONDITIONS = {
    "kelvin_helmholtz": [
        {"tag": "kh_b", "ic": dict(shear_width=0.65, noise_amplitude=0.15,
                                   drift_velocity=0.35),
         "note": "wider shear layer, stronger seed mode, slower drift"},
        {"tag": "kh_c", "ic": dict(shear_width=0.42, noise_amplitude=0.07,
                                   drift_velocity=0.60),
         "note": "intermediate layer, weaker seed, faster drift"},
    ],
    "harris_tearing": [
        {"tag": "tearing_b", "ic": dict(B0=1.25, shear_width=0.38,
                                        perturbation=0.005, k_mode=1.0),
         "note": "stronger field, thicker sheet, weaker seed"},
        {"tag": "tearing_c", "ic": dict(B0=0.9, shear_width=0.26,
                                        perturbation=0.015, k_mode=3.0),
         "note": "thinner sheet, mode 3"},
    ],
    "mhd_rotor": [
        # `rotor` est la SEULE classe ou une graine existe reellement.
        {"tag": "rotor_seed7", "ic": {}, "rng": 7,
         "note": "canonical parameters, RNG seed 42 -> 7 (a TRUE seed "
                 "change, the only one available anywhere in the suite)"},
        {"tag": "rotor_b", "ic": dict(omega=12.0, r0=0.85,
                                      taper_width=0.12, perturbation=0.008),
         "note": "faster, larger rotor with a sharper taper"},
    ],
    "orszag_tang": [
        # Aucun parametre : le seul levier est le Reynolds, et il ne
        # constitue PAS une variation de condition initiale. Declare.
        {"tag": "ot_re900", "ic": {}, "re": 900,
         "note": "init_orszag_tang takes NO parameters; this varies the "
                 "Reynolds number, which is NOT an initial-condition "
                 "variation and must not be counted as one"},
    ],
}


class rng_override:
    """Substitue `np.random.default_rng` pendant l'initialisation.

    `init_mhd_rotor` appelle `np.random.default_rng(42)` en dur. C'est la
    seule vraie graine physique de toute la suite, et elle n'est pas un
    parametre. On la rend variable sans modifier V1 : la substitution est
    locale a l'initialisation et numpy est rendu intact quoi qu'il arrive.
    """

    def __init__(self, seed):
        self.seed = seed
        self._saved = None

    def __enter__(self):
        self._saved = np.random.default_rng
        seed = self.seed
        np.random.default_rng = lambda *a, **k: self._saved(seed)
        return self

    def __exit__(self, *exc):
        np.random.default_rng = self._saved
        return False


def build_trace(T, key, cfg, scenario, cond):
    """Trace DNS pour une condition, avec ses substitutions."""
    cfg2 = dict(cfg)
    if "re" in cond:
        cfg2["Re"] = cond["re"]
        cfg2["Rm"] = cond["re"]
    label = f"robust/{cond['tag']}"
    if "rng" in cond:
        with rng_override(cond["rng"]):
            with init_override(scenario, cond.get("ic")):
                return T._precompute_dns_for([(key, cfg2)], label=label), cfg2
    with init_override(scenario, cond.get("ic")):
        return T._precompute_dns_for([(key, cfg2)], label=label), cfg2


def signature(trace, key):
    hs = trace[key][1]
    return float(np.sum([np.sum(np.abs(hs[k]))
                         for k in ("vx", "vy", "Bx", "By")]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    from config import RESULTS_DIR

    p.add_argument("--fold", required=True)
    p.add_argument("--repeats", type=int, default=3,
                   help="tirages Q-HAS par condition (bras non "
                        "deterministe, D11)")
    p.add_argument("--frontier-points", type=int, default=3,
                   help="seuils classiques encadrant le budget Q-HAS")
    p.add_argument("--prefix", default="t15_level3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    path = os.path.join(RESULTS_DIR, f"{args.prefix}_fold_{args.fold}.json")
    if not os.path.exists(path):
        raise SystemExit(f"fold {args.fold} not computed yet")
    rec = json.load(open(path))

    T = _load_v1_training_module()
    scen = dict(fold_scenarios(T, warn=False))
    cfg = scen[args.fold]
    scenario = rec.get("scenario") or args.fold
    conds = CONDITIONS.get(scenario, [])
    if not conds:
        raise SystemExit(f"no extra condition defined for {scenario}")

    hp_q = dict(rec["hyperparams"])
    hp_c, cls_src, _ = safe_classical_hyperparams(
        rec, RESULTS_DIR, args.fold, always_matched=True)
    thr_matched = float(hp_c["threshold_amr"])

    print("=" * 84)
    print(f"  V4 T25 - physics robustness, fold {args.fold} ({scenario})")
    print(f"  {len(conds)} extra initial condition(s), "
          f"{args.repeats} Q-HAS draws each")
    print(f"  classical frontier anchored at the matched threshold "
          f"{thr_matched:.4f}")
    print("=" * 84, flush=True)

    op = os.path.join(RESULTS_DIR,
                      f"t25_physics_robustness_{args.fold}.json")
    prev = {}
    if os.path.exists(op) and not args.no_resume:
        try:
            d = json.load(open(op))
            if d.get("fold") == args.fold and d.get(
                    "cli_args", {}).get("repeats") == args.repeats:
                prev = {c["tag"]: c for c in d.get("conditions", [])}
                print(f"  RESUMING: {len(prev)} condition(s) already done",
                      flush=True)
        except ValueError as exc:
            # NE PAS avaler : un point de reprise illisible ferait repartir
            # de zero en silence, et l'artefact final serait indiscernable
            # d'une execution qui n'a jamais ete interrompue. On le dit.
            print(f"  WARNING: checkpoint at {os.path.basename(op)} is "
                  f"unreadable ({exc}); starting over from the first "
                  f"condition", flush=True)

    # Reference CANONIQUE : sans elle, « on a fait varier la physique » est
    # une affirmation invérifiée. Une condition qui ne deplace pas la
    # trajectoire ne teste rien, et son resultat serait indiscernable d'un
    # vrai test de robustesse — le motif de cette campagne. On mesure donc
    # le deplacement et on le RECORD, comme T22 le fait pour ses conditions
    # inedites.
    dns_can = T._precompute_dns_for([(args.fold, cfg)],
                                    label=f"robust/canonical-{args.fold}")
    sig_can = signature(dns_can, args.fold)
    print(f"  canonical DNS signature = {sig_can:.6e}", flush=True)

    prov = provenance.start()
    t0 = time.time()
    out = {"fold": args.fold, "scenario": scenario,
           "classical_reference_source": cls_src,
           "matched_threshold": thr_matched, "conditions": []}

    def guarded(hp, only, cfg_, dns_):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = run_arm(T, args.fold, cfg_, dns_, hp, only, verbose=True)
        ab = parse_abort(buf.getvalue())
        # V1 en mode verbeux ouvre des figures matplotlib et ne les ferme
        # jamais (src/visual.py). Sur des dizaines d'executions cela epuise
        # la memoire et le processus est tue. Le mode verbeux est pourtant
        # obligatoire ici : c'est lui qui emet le marqueur d'avortement.
        # L'echec de cette fermeture ne peut fausser AUCUN resultat — elle
        # ne touche qu'a l'etat graphique — d'ou le `pass`, justifie ici et
        # non laisse muet.
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception:
            pass
        return {"phys_score": float(r.get("phys_score", np.nan)),
                "patch_ratio": float(r.get("patch_ratio", np.nan)),
                "completed": ab is None, "abort": ab}

    def checkpoint(stage):
        snap = dict(out)
        snap["status"] = "partial"
        snap["partial_stage"] = stage
        snap["partial_warning"] = (
            "INCOMPLETE — resume point, not a result")
        snap.update(provenance.finish(prov))
        snap["cli_args"] = vars(args)
        json.dump(snap, open(op, "w"), indent=1)
        print(f"    [checkpoint] {stage}", flush=True)

    for cond in conds:
        if cond["tag"] in prev:
            out["conditions"].append(prev[cond["tag"]])
            print(f"\n  {cond['tag']}: reused from checkpoint", flush=True)
            continue
        print(f"\n  --- {cond['tag']}: {cond['note']} ---", flush=True)
        dns, cfg2 = build_trace(T, args.fold, cfg, scenario, cond)
        sig = signature(dns, args.fold)
        shift = abs(sig - sig_can) / max(abs(sig_can), 1e-30)
        weak = shift < 0.01
        print(f"    DNS signature {sig:.6e}  shift vs canonical "
              f"{100 * shift:.4f}%", flush=True)
        if weak:
            print(f"    WARNING: this condition moves the trajectory by "
                  f"only {100 * shift:.4f}% (< 1%) — it tests almost "
                  f"nothing and must be reported as such", flush=True)

        q = []
        for i in range(args.repeats):
            q.append(guarded(hp_q, False, cfg2, dns))
            tail = ("" if q[-1]["completed"]
                    else "   **ABORTED step %d**" % q[-1]["abort"]["abort_step"])
            print(f"    [qhas] {i + 1}/{args.repeats}: "
                  f"phys={q[-1]['phys_score']:.5f} "
                  f"patch={q[-1]['patch_ratio']:.4f}{tail}", flush=True)
        q_ok = [r for r in q if r["completed"]]

        # frontiere classique : le seuil apparie, plus des seuils
        # l'encadrant, pour pouvoir interpoler au budget realise par Q-HAS
        thrs = [thr_matched]
        for k in range(1, args.frontier_points):
            thrs += [thr_matched * (1.0 + 0.6 * k),
                     thr_matched * max(0.15, 1.0 - 0.45 * k)]
        thrs = sorted({round(t, 6) for t in thrs})[:args.frontier_points + 2]
        front = []
        for t in thrs:
            hpc = dict(hp_c); hpc["threshold_amr"] = float(t)
            r = guarded(hpc, True, cfg2, dns)
            front.append({"threshold": float(t), **r})
            print(f"    [classical] thr={t:.4f}: "
                  f"phys={r['phys_score']:.5f} patch={r['patch_ratio']:.4f}"
                  f"{'' if r['completed'] else '   **ABORTED**'}", flush=True)

        f_ok = sorted([r for r in front if r["completed"]],
                      key=lambda r: r["patch_ratio"])
        entry = {"tag": cond["tag"], "note": cond["note"],
                 "is_ic_variation": "re" not in cond,
                 "is_true_seed_change": "rng" in cond,
                 "dns_signature": sig,
                 "dns_signature_canonical": sig_can,
                 "dns_relative_shift": float(shift),
                 "condition_is_weak": bool(weak),
                 "qhas_runs": q, "classical_frontier": front,
                 "n_qhas_completed": len(q_ok),
                 "n_qhas_aborted": len(q) - len(q_ok),
                 "n_classical_completed": len(f_ok),
                 "n_classical_aborted": len(front) - len(f_ok)}
        if q_ok and len(f_ok) >= 2:
            qp = float(np.mean([r["patch_ratio"] for r in q_ok]))
            qe = float(np.mean([r["phys_score"] for r in q_ok]))
            xs = [r["patch_ratio"] for r in f_ok]
            ys = [r["phys_score"] for r in f_ok]
            entry.update(qhas_patch=qp, qhas_phys=qe)
            if xs[0] <= qp <= xs[-1]:
                ref = float(np.interp(qp, xs, ys))
                entry["frontier_at_qhas_budget"] = ref
                entry["ratio_vs_frontier"] = qe / ref if ref else None
                entry["qhas_worse"] = bool(ref and qe > ref)
                entry["out_of_swept_range"] = False
                print(f"    => Q-HAS {qe:.5f} vs frontier {ref:.5f} "
                      f"at budget {qp:.4f}  ratio={qe / ref:.2f}x")
            else:
                # jamais d'extrapolation : `np.interp` rendrait le bord
                entry["out_of_swept_range"] = True
                entry["ratio_vs_frontier"] = None
                entry["qhas_worse"] = None
                print(f"    => budget {qp:.4f} OUTSIDE the swept range "
                      f"[{xs[0]:.4f}, {xs[-1]:.4f}] — no ratio")
        else:
            entry["ratio_vs_frontier"] = None
            entry["qhas_worse"] = None
            print("    => no comparison: "
                  f"{len(q_ok)} Q-HAS and {len(f_ok)} classical completed")
        out["conditions"].append(entry)
        checkpoint(cond["tag"])

    # Une condition qui ne deplace pas la trajectoire ne peut ni confirmer
    # ni infirmer la direction : elle sort du decompte, dans les deux sens.
    dec = [c for c in out["conditions"]
           if c.get("qhas_worse") is not None and not c.get("condition_is_weak")]
    weak_n = sum(1 for c in out["conditions"] if c.get("condition_is_weak"))
    out["n_decidable"] = len(dec)
    out["n_qhas_worse"] = sum(1 for c in dec if c["qhas_worse"])
    out["n_weak_excluded"] = weak_n
    if weak_n:
        print(f"  {weak_n} condition(s) excluded as vacuous "
              f"(trajectory shift < 1%)")
    out["status"] = "completed"
    out.update(provenance.finish(prov))
    out["cli_args"] = vars(args)
    out["wall_s"] = time.time() - t0
    out.pop("partial_stage", None)
    out.pop("partial_warning", None)
    json.dump(out, open(op, "w"), indent=1)

    print("\n  " + "-" * 78)
    print(f"  direction held on {out['n_qhas_worse']}/{out['n_decidable']} "
          f"decidable condition(s) for fold {args.fold}")
    print(f"  saved: {os.path.basename(op)} ({time.time() - t0:.0f}s)")
    print("\nV4 Task 25 complete.")


if __name__ == "__main__":
    main()
