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



# Facteur maximal tolere entre les deux points encadrants. Au-dela,
# l'interpolation traverse une variation trop brutale pour que « l'erreur
# classique a ce budget » ait un sens.
FRONTIER_MAX_LOCAL_RATIO = 5.0


def frontier_verdict(f_ok, qp, qe):
    """Erreur classique au budget `qp`, ou un refus motive.

    LE PIEGE, TROUVE EN LISANT LES PREMIERES SORTIES. Sur les conditions
    initiales alternatives, la relation budget -> erreur du bras classique
    n'est PAS monotone. Sur `tearing_b` :

        patch 0.0156 -> phys 0.043     patch 0.6250 -> phys 0.012
        patch 0.2297 -> phys 9.659     patch 0.8742 -> phys 1.289

    Raffiner davantage y donne parfois une erreur 30x pire. Interpoler
    « la frontiere atteignable » sur de tels points produit un nombre
    d'apparence normale qui ne mesure rien — exactement le motif traque
    par cette campagne, et il aurait ete publie comme un ratio.

    On n'accepte donc l'interpolation que si le voisinage encadrant est
    LOCALEMENT SAIN : erreur non croissante avec le budget, et les deux
    points a moins d'un facteur `FRONTIER_MAX_LOCAL_RATIO`. Sinon on rend
    le motif du refus, jamais un chiffre.
    """
    xs = [r["patch_ratio"] for r in f_ok]
    ys = [r["phys_score"] for r in f_ok]
    if len(f_ok) < 2:
        return None, "fewer than 2 completed classical runs"
    if not (xs[0] <= qp <= xs[-1]):
        return None, (f"budget {qp:.4f} outside the swept range "
                      f"[{xs[0]:.4f}, {xs[-1]:.4f}]")
    i = max(1, min(len(xs) - 1,
                   next(k for k in range(1, len(xs)) if xs[k] >= qp)))
    lo_x, lo_y, hi_x, hi_y = xs[i - 1], ys[i - 1], xs[i], ys[i]
    if hi_y > lo_y:
        return None, (f"frontier NOT monotone in the bracketing interval: "
                      f"budget {lo_x:.4f}->{hi_x:.4f} gives error "
                      f"{lo_y:.5f}->{hi_y:.5f} (more refinement, worse "
                      f"error) — no attainable error is defined here")
    lo_m, hi_m = max(lo_y, 1e-30), max(hi_y, 1e-30)
    ratio = max(lo_m, hi_m) / min(lo_m, hi_m)
    if ratio > FRONTIER_MAX_LOCAL_RATIO:
        return None, (f"bracketing points differ by {ratio:.1f}x "
                      f"(> {FRONTIER_MAX_LOCAL_RATIO:.0f}x) over a budget "
                      f"gap of {hi_x - lo_x:.4f} — too steep to interpolate")
    t = (qp - lo_x) / (hi_x - lo_x) if hi_x > lo_x else 0.0
    return float(lo_y + t * (hi_y - lo_y)), None


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
    p.add_argument("--run-weak", action="store_true",
                   help="simuler quand meme les conditions vacues (< 1 %% de "
                        "deplacement). Par defaut on ne les simule pas : "
                        "elles sont exclues du decompte de toute facon")
    p.add_argument("--recompute", action="store_true",
                   help="re-derive the verdicts from a stored artifact "
                        "WITHOUT simulating anything. Sert quand la regle "
                        "de verdict change : les tirages et la frontiere "
                        "sont deja la, seule leur lecture evolue.")
    args = p.parse_args()

    if args.recompute:
        op = os.path.join(RESULTS_DIR,
                          f"t25_physics_robustness_{args.fold}.json")
        if not os.path.exists(op):
            raise SystemExit(f"no artifact for fold {args.fold}")
        d = json.load(open(op))
        changed = 0
        for c in d.get("conditions", []):
            if c.get("skipped_as_vacuous"):
                continue
            f_ok = sorted([r for r in c.get("classical_frontier", [])
                           if r["completed"]], key=lambda r: r["patch_ratio"])
            q_ok = [r for r in c.get("qhas_runs", []) if r["completed"]]
            if not q_ok or len(f_ok) < 2:
                continue
            qp = float(np.mean([r["patch_ratio"] for r in q_ok]))
            qe = float(np.mean([r["phys_score"] for r in q_ok]))
            ref, why = frontier_verdict(f_ok, qp, qe)
            before = c.get("ratio_vs_frontier")
            c["frontier_at_qhas_budget"] = ref
            c["frontier_refusal"] = why
            c["ratio_vs_frontier"] = (qe / ref) if ref else None
            c["qhas_worse"] = bool(qe > ref) if ref else None
            after = c["ratio_vs_frontier"]
            if (before is None) != (after is None):
                changed += 1
                print(f"  {c['tag']}: "
                      + (f"{before:.2f}x -> NO VERDICT ({why})" if before
                         else f"NO VERDICT -> {after:.2f}x"))
        dec = [c for c in d["conditions"]
               if c.get("qhas_worse") is not None
               and not c.get("condition_is_weak")]
        d["n_decidable"] = len(dec)
        d["n_qhas_worse"] = sum(1 for c in dec if c["qhas_worse"])
        d["verdicts_recomputed"] = True
        json.dump(d, open(op, "w"), indent=1)
        print(f"  {changed} verdict(s) changed; direction now "
              f"{d['n_qhas_worse']}/{d['n_decidable']}")
        raise SystemExit(0)

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
            if not args.run_weak:
                # Une condition vacue est deja exclue du decompte : y
                # consacrer ~1 h 30 de simulation produirait un resultat qui
                # ne peut ni confirmer ni infirmer quoi que ce soit, tout en
                # RESSEMBLANT a une confirmation de robustesse. On enregistre
                # le constat — qui est lui-meme le resultat — et on passe.
                out["conditions"].append({
                    "tag": cond["tag"], "note": cond["note"],
                    "is_ic_variation": "re" not in cond,
                    "is_true_seed_change": "rng" in cond,
                    "dns_signature": sig,
                    "dns_signature_canonical": sig_can,
                    "dns_relative_shift": float(shift),
                    "condition_is_weak": True,
                    "skipped_as_vacuous": True,
                    "qhas_worse": None, "ratio_vs_frontier": None,
                })
                print(f"    => SKIPPED: no draws run. The finding is the "
                      f"shift itself, not a comparison.", flush=True)
                checkpoint(cond["tag"])
                continue

        q = []
        for i in range(args.repeats):
            q.append(guarded(hp_q, False, cfg2, dns))
            tail = ("" if q[-1]["completed"]
                    else "   **ABORTED step %d**" % q[-1]["abort"]["abort_step"])
            print(f"    [qhas] {i + 1}/{args.repeats}: "
                  f"phys={q[-1]['phys_score']:.5f} "
                  f"patch={q[-1]['patch_ratio']:.4f}{tail}", flush=True)
        q_ok = [r for r in q if r["completed"]]

        # FRONTIERE CLASSIQUE, PLACEE PAR BISSECTION SUR LE BUDGET VISE.
        #
        # La premiere version balayait des seuils derives du seuil apparie,
        # c'est-a-dire calibre sur la condition CANONIQUE. Sur une autre
        # condition initiale le budget realise par Q-HAS se deplace, et la
        # plage balayee ne l'encadrait souvent pas : `tearing_c` a rendu un
        # budget de 0.7689 pour une plage [0.0156, 0.6250]. Deux conditions
        # sur trois sortaient donc sans verdict — la tache refusait
        # correctement, mais ne mesurait rien.
        #
        # On cible desormais le budget que Q-HAS vient REELLEMENT de
        # realiser sur CETTE condition. `patch_ratio` decroit avec le seuil,
        # ce qui rend la bissection valide ; chaque evaluation est gardee
        # contre la divergence, et toutes sont conservees : la frontiere est
        # la trace complete, pas seulement le point le plus proche.
        target = (float(np.mean([r["patch_ratio"] for r in q_ok]))
                  if q_ok else None)
        front = []

        def _eval(thr):
            hpc = dict(hp_c); hpc["threshold_amr"] = float(thr)
            r = guarded(hpc, True, cfg2, dns)
            front.append({"threshold": float(thr), **r})
            print(f"    [classical] thr={thr:.4f}: "
                  f"phys={r['phys_score']:.5f} patch={r['patch_ratio']:.4f}"
                  f"{'' if r['completed'] else '   **ABORTED**'}", flush=True)
            return r

        if target is None:
            # aucun tirage Q-HAS valide : rien a encadrer, on se rabat sur
            # le seuil apparie pour garder une trace exploitable
            _eval(thr_matched)
        else:
            lo, hi = 0.02, 0.95           # patch eleve <-> seuil bas
            r_lo, r_hi = _eval(lo), _eval(hi)
            for _ in range(args.frontier_points):
                # arret des que le budget vise est encadre serre
                below = [r for r in front
                         if r["completed"] and r["patch_ratio"] <= target]
                above = [r for r in front
                         if r["completed"] and r["patch_ratio"] >= target]
                if below and above:
                    gap = (min(above, key=lambda r: r["patch_ratio"])["patch_ratio"]
                           - max(below, key=lambda r: r["patch_ratio"])["patch_ratio"])
                    if gap <= 0.06:
                        break
                mid = 0.5 * (lo + hi)
                r_mid = _eval(mid)
                if r_mid["patch_ratio"] > target:
                    lo = mid
                else:
                    hi = mid

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
            ref, why = frontier_verdict(f_ok, qp, qe)
            entry["frontier_at_qhas_budget"] = ref
            entry["frontier_refusal"] = why
            if ref:
                entry["ratio_vs_frontier"] = qe / ref
                entry["qhas_worse"] = bool(qe > ref)
                print(f"    => Q-HAS {qe:.5f} vs frontier {ref:.5f} "
                      f"at budget {qp:.4f}  ratio={qe / ref:.2f}x")
            else:
                entry["ratio_vs_frontier"] = None
                entry["qhas_worse"] = None
                print(f"    => NO VERDICT: {why}")
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
