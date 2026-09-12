#!/usr/bin/env python3
"""
V4 Task 22 - Fuite d'apprentissage et conditions initiales INEDITES.

DEUX PROBLEMES DISTINCTS, dont un seul etait traite jusqu'ici.

(1) FUITE DE PARAMETRE (defaut D13, decouvert ici).
    Le protocole Level-3 pretend que la classe tenue n'intervient dans
    AUCUN reglage. C'est faux pour le bras QAOA. Dans
    `train_hyperparams.make_composite_objective` le seuil de decision est
    code en dur :

        HyperParams["threshold_amr"] = 0.14959824837662078  # le meilleur classique

    et cette valeur vient de `_run_classical_phase1`, qui l'ajuste sur
    « KH + OT + Tearing + Rotor » — les QUATRE classes. Mon propre pilote
    reproduisait la fuite (`best.setdefault("threshold_amr", 0.1495...)`).
    Le bras classique, lui, re-regle son seuil par fold sur les seules
    classes d'entrainement. La fuite est donc ASYMETRIQUE et joue EN FAVEUR
    de Q-HAS.

(2) CONDITION INITIALE DEJA VUE.
    Meme sans fuite de parametre, la classe tenue est evaluee sur la meme
    condition initiale que celle qui sert partout ailleurs : les fonctions
    `init_*` de V1 sont appelees sans argument, donc toujours avec leurs
    valeurs par defaut. Un modele qui generalise doit affronter une
    condition qu'il n'a jamais rencontree, pas la trajectoire canonique.

CE QUE FAIT CETTE TACHE.

  mode `unseen-ic`  : reprend les hyperparametres deja regles de chaque
                      fold et evalue les DEUX bras sur une condition
                      initiale INEDITE de la classe tenue. Compare la
                      degradation des deux bras. Bon marche.
  mode `leak-free`  : evalue Q-HAS avec le seuil du bras CLASSIQUE de ce
                      fold, issu de `train_classical_threshold_excluding`
                      donc regle sur les seules classes d'entrainement.
                      Supprime D13 sans nouveau reglage Optuna.

Le mode `no-leak` a existe comme OPTION ACCEPTEE ET NON IMPLEMENTEE : il ne
changeait que le nom du fichier de sortie, produisant un artefact nomme
comme si la fuite avait ete supprimee alors que le calcul etait identique.
C'est la neuvieme instance du motif que cette campagne traque — un calcul
qui ne fait pas ce qu'il annonce et rend un resultat indiscernable d'un
resultat valide. L'option est retiree et remplacee par `leak-free`, qui est
reellement implemente ci-dessous.

Les conditions inedites sont obtenues en substituant temporairement
`_init_dns_scenario` de V1 pour transmettre des parametres physiques aux
`init_*`, puis en la restaurant. V1 n'est pas modifie, et la substitution
est verifiee (la trajectoire DOIT differer de la canonique).

`orszag_tang` n'expose AUCUN parametre : sa condition initiale ne peut pas
varier sans toucher V1. Pour cette classe seule, la condition inedite est
un nombre de Reynolds different, ce qui est declare comme tel.

Sortie : results/t22_unseen_{mode}_{fold}.json
Usage :
  python study/h4_transfer/h4_unseen_conditions.py --fold kh --mode unseen-ic
"""
import argparse, contextlib, io, json, os, sys, time

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

import provenance
from closed_loop_campaign import (_load_v1_training_module, fold_scenarios,
                                    run_arm)
from closed_loop_divergence_audit import (parse_abort,
                                      safe_classical_hyperparams)

# La valeur fuitee : ajustee sur les quatre classes, puis figee comme seuil
# du bras QAOA quelle que soit la classe tenue.
LEAKED_THRESHOLD = 0.14959824837662078

# Conditions initiales INEDITES, une par classe. Les valeurs sont choisies
# loin des defauts mais dans un domaine physiquement raisonnable : il s'agit
# de la meme instabilite, pas d'un autre probleme.
#   'ic'  -> arguments passes a la fonction init_* de V1
#   're'  -> Reynolds substitue (seul levier pour orszag_tang)
UNSEEN_CONDITIONS = {
    "kelvin_helmholtz": {
        "ic": dict(shear_width=0.35, noise_amplitude=0.05,
                   drift_velocity=0.7),
        "note": "narrower shear layer, weaker seed, faster drift",
    },
    "harris_tearing": {
        "ic": dict(B0=0.8, shear_width=0.22, perturbation=0.02, k_mode=2.0),
        "note": "thinner current sheet, mode 2 instead of mode 1",
    },
    "mhd_rotor": {
        "ic": dict(omega=8.0, r0=0.60, taper_width=0.20, perturbation=0.01),
        "note": "slower, smaller rotor with a wider taper",
    },
    "orszag_tang": {
        "ic": {},
        "re": 600,
        "note": "init_orszag_tang takes NO parameters; the only available "
                "unseen condition is a different Reynolds number",
    },
}


def unseen_config(cfg, scenario):
    """Config du fold avec la condition inedite appliquee."""
    spec = UNSEEN_CONDITIONS.get(scenario, {})
    out = dict(cfg)
    if "re" in spec:
        out["Re"] = spec["re"]
        out["Rm"] = spec["re"]
    return out


class init_override:
    """Substitue `_init_dns_scenario` pour passer des arguments aux init_*.

    Context manager : V1 est rendu a son etat initial quoi qu'il arrive.
    """

    def __init__(self, scenario, ic_params):
        self.scenario = scenario
        self.ic_params = dict(ic_params or {})
        self._saved = None

    def __enter__(self):
        import Simulation.pre_compute_dns as pcd
        self._mod = pcd
        self._saved = pcd._init_dns_scenario
        scenario, ic = self.scenario, self.ic_params
        saved = self._saved

        def patched(sim, scen):
            if scen == scenario and ic:
                getattr(sim, f"init_{scen}")(**ic)
                sim.enforce_incompressibility()
                sim.record_energy()
                return
            return saved(sim, scen)

        pcd._init_dns_scenario = patched
        return self

    def __exit__(self, *exc):
        self._mod._init_dns_scenario = self._saved
        return False


def dns_signature(dns_trace, hot_start):
    """Empreinte de la trajectoire, pour PROUVER qu'elle a change."""
    hs = hot_start
    return float(np.sum([np.sum(np.abs(hs[k])) for k in
                         ("vx", "vy", "Bx", "By")]))


def build_traces(T, key, cfg, scenario, unseen):
    """Trace DNS canonique ou inedite, avec verification du changement."""
    if not unseen:
        return T._precompute_dns_for([(key, cfg)], label=f"canonical/{key}")
    spec = UNSEEN_CONDITIONS.get(scenario, {})
    cfg2 = unseen_config(cfg, scenario)
    with init_override(scenario, spec.get("ic")):
        tr = T._precompute_dns_for([(key, cfg2)], label=f"unseen/{key}")
    return tr


#: Les champs qu'un point de sauvegarde doit partager avec la relance pour
#: que ses tirages soient comparables. `fold` et `mode` sont lus a la racine
#: de l'artefact, les deux autres dans son `cli_args`.
_RESUME_KEYS = ("fold", "mode", "repeats", "matched_reference")


def checkpoint_is_reusable(prev, args):
    """Le point de sauvegarde `prev` a-t-il ete ecrit sous la MEME config ?

    Reprendre sous une AUTRE configuration melangerait des tirages
    incomparables dans une seule moyenne publiee — la forme exacte de defaut
    que ce module traque. On refuse plutot que de deviner.

    Extraite pour etre testable directement : le seul garde textuel,
    `test_resume_reuses_only_matching_configurations`, cherche quatre
    chaines dans le TEXTE du fichier plutot que le comportement --
    remplacer un seul `and` par un `or` casse la decision (`fold` et
    `mode` egaux suffiraient alors a accepter un point ecrit sous un
    autre `--repeats`) sans faire bouger aucune des quatre chaines, et
    `pytest tests/study/test_t24_leak_free.py` rendrait toujours **26
    passed**.
    """
    cli = prev.get("cli_args", {}) or {}
    return (prev.get("fold") == args.fold
            and prev.get("mode") == args.mode
            and cli.get("repeats") == args.repeats
            and cli.get("matched_reference") == args.matched_reference)


def apply_leak_free_threshold(hp_q, rec):
    """Remplace le seuil QAOA fuyant par celui du bras classique du fold.

    Extrait de `main()` pour etre mesurable sans rejouer les heures de
    DNS d'un fold : le garde textuel est fragile -- les quatre chaines
    que `test_no_leak_mode_is_gone_and_leak_free_is_wired` cherchait dans
    ce fichier restaient toutes presentes meme en reecrivant le seuil sur
    `LEAKED_THRESHOLD` une ligne plus bas, donnant 35 tests verts sous un
    artefact nomme `leak-free` malgre la fuite. Le corps est INCHANGE,
    seul son emplacement bouge.

    D13 : le seuil QAOA par defaut (0.1496) a ete ajuste sur les QUATRE
    classes, classe tenue comprise. Celui du bras classique de ce fold vient
    de `train_classical_threshold_excluding`, donc des seules classes
    d'entrainement : le reprendre supprime la fuite.
    """
    leak_free_thr = float(rec["classical_params"]["threshold_amr"])
    assert abs(hp_q["threshold_amr"] - LEAKED_THRESHOLD) < 1e-9, (
        "the QAOA arm was not at the leaked threshold; check the fold")
    hp_q["threshold_amr"] = leak_free_thr
    print(f"  LEAK-FREE: QAOA threshold {LEAKED_THRESHOLD:.6f} "
          f"-> {leak_free_thr:.6f} (tuned on training classes only)",
          flush=True)
    return hp_q


def main():
    p = argparse.ArgumentParser(
        description="V4 T22: leak-free tuning and unseen initial conditions")
    from config import RESULTS_DIR

    p.add_argument("--fold", required=True)
    p.add_argument("--mode", choices=["unseen-ic", "leak-free"],
                   default="unseen-ic",
                   help="unseen-ic: conditions initiales inedites. "
                        "leak-free: seuil QAOA repris du bras classique du "
                        "fold, regle hors classe tenue (supprime D13)")
    p.add_argument("--no-resume", action="store_true",
                   help="ignorer un point de sauvegarde existant et tout "
                        "recalculer depuis le premier tirage")
    p.add_argument("--prefix", default="t15_level3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1,
                   help="tirages Q-HAS par condition (D11 : CV 17-49%%, "
                        "un seul tirage ne separe rien du bruit)")
    p.add_argument("--matched-reference", action="store_true",
                   help="comparer au seuil classique BUDGET-APPARIE partout, "
                        "pas seulement quand le bras regle diverge (D4)")
    args = p.parse_args()

    path = os.path.join(RESULTS_DIR, f"{args.prefix}_fold_{args.fold}.json")
    if not os.path.exists(path):
        raise SystemExit(f"fold {args.fold} not computed yet")
    rec = json.load(open(path))
    scenario = rec["scenario"]

    print("=" * 84)
    print(f"  V4 T22 - fold {args.fold} ({scenario})  mode={args.mode}")
    print(f"  unseen condition: {UNSEEN_CONDITIONS[scenario]['note']}")
    print("=" * 84, flush=True)

    T = _load_v1_training_module()
    cfg = dict(fold_scenarios(T))[args.fold]

    hp_q = dict(rec["hyperparams"])
    if args.mode == "leak-free":
        apply_leak_free_threshold(hp_q, rec)
    # Le bras classique doit partir d'un point de fonctionnement QUI TERMINE :
    # sur `rotor` le seuil regle diverge, et comparer Q-HAS a une trajectoire
    # tronquee ne mesure rien (le piege a deja fausse T15, T20 et un premier
    # passage de T22).
    hp_c, cls_src, cls_done = safe_classical_hyperparams(
        rec, RESULTS_DIR, args.fold, always_matched=args.matched_reference)
    print(f"  classical reference: {cls_src}")
    leaked = abs(hp_q.get("threshold_amr", 0) - LEAKED_THRESHOLD) < 1e-9
    print(f"  QAOA arm threshold = {hp_q.get('threshold_amr')}"
          f"{'   <- LEAKED (fitted on all 4 classes)' if leaked else ''}")
    print(f"  classical threshold = {hp_c.get('threshold_amr')}"
          f"   (tuned on training classes only)", flush=True)

    prov = provenance.start()   # D15 : le hash AVANT le calcul
    t0 = time.time()
    out = {"fold": args.fold, "scenario": scenario, "mode": args.mode,
           "classical_reference_source": cls_src,
           "classical_tuned_arm_completed": cls_done,
           "unseen_condition": UNSEEN_CONDITIONS[scenario],
           "qaoa_threshold_leaked": bool(leaked),
           # Les DEUX seuils, explicitement. En mode leak-free ils different
           # (le QAOA prend le seuil classique regle du fold, le controle
           # prend le seuil budget-apparie) : sans ces champs, un lecteur
           # supposerait une comparaison a point de fonctionnement egal, ce
           # qu'elle n'est pas.
           "qaoa_threshold_amr": float(hp_q.get("threshold_amr", float("nan"))),
           "classical_threshold_amr": float(
               hp_c.get("threshold_amr", float("nan"))),
           "arms": {}}
    out["thresholds_match"] = bool(
        abs(out["qaoa_threshold_amr"] - out["classical_threshold_amr"]) < 1e-12)

    # --- reference: canonical initial condition -----------------------
    dns_can = build_traces(T, args.fold, cfg, scenario, unseen=False)
    sig_can = dns_signature(*dns_can[args.fold])

    # --- unseen initial condition -------------------------------------
    dns_uns = build_traces(T, args.fold, cfg, scenario, unseen=True)
    sig_uns = dns_signature(*dns_uns[args.fold])
    rel = abs(sig_uns - sig_can) / max(abs(sig_can), 1e-30)
    print(f"\n  DNS signature canonical={sig_can:.6e}  unseen={sig_uns:.6e}")
    print(f"  relative shift: {rel:.4%}")

    # Trois controles, pas un seul. « A change » ne suffit pas :
    #  - la trace doit etre FINIE (une DNS partie en vrille passerait le
    #    test de changement avec une signature enorme) ;
    #  - le changement doit etre REEL (l'override a bien pris) ;
    #  - et surtout SIGNIFICATIF : sur `ot` la seule variation possible est
    #    le Reynolds, et 400->600 ne deplace le hot start que de 0.3 %,
    #    contre 7-17 % pour les trois classes ou l'on peut varier la
    #    condition initiale elle-meme. Un ecart de cet ordre ne teste rien.
    if not np.isfinite(sig_uns) or not np.isfinite(sig_can):
        raise SystemExit("non-finite DNS signature: the trajectory diverged")
    if not 0.05 < (sig_uns / sig_can if sig_can else 0) < 20.0:
        raise SystemExit(
            f"DNS signature ratio {sig_uns / sig_can:.3g} is out of any "
            f"physical band: the unseen condition likely diverged")
    if rel <= 1e-6:
        raise SystemExit(
            "the unseen condition produced an IDENTICAL trajectory; the "
            "override did not take effect and the comparison would be void.")
    WEAK = 0.01
    weak = rel < WEAK
    if weak:
        print(f"  WARNING: the unseen condition shifts the trajectory by "
              f"only {rel:.4%} (< {WEAK:.0%}). This fold's transfer test is "
              f"NEARLY VACUOUS and must be reported as such.")
    changed = True
    out["dns_signature_canonical"] = sig_can
    out["dns_signature_unseen"] = sig_uns
    out["dns_relative_shift"] = float(rel)
    out["unseen_condition_is_weak"] = bool(weak)

    cfg_uns = unseen_config(cfg, scenario)
    KEYS = ("combined", "phys_score", "patch_ratio")

    def repeat(hp, only, cfg_, dns_, n, tag, on_run=None, done=None):
        """n executions, chacune verifiee CONTRE LA DIVERGENCE.

        Sans ce controle une trajectoire avortee se melange aux autres :
        sur `tearing` un tirage Q-HAS a rendu phys = 0.601 quand ses quatre
        voisins donnaient 0.0017-0.0027, soit un facteur 300, et moyenne
        comme ecart-type en devenaient absurdes (deg 14.4 +- 32.0). Le
        statut ne peut PAS etre retrouve apres coup : le bras Q-HAS n'est
        pas deterministe (D11), donc un rejeu ne reproduit pas le tirage
        fautif. Il faut le capturer AU MOMENT de l'execution.
        """
        runs = list(done or [])
        if runs:
            print(f"    {tag}: resuming, {len(runs)}/{n} draw(s) reused "
                  f"from the checkpoint", flush=True)
        for i in range(len(runs), n):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r = run_arm(T, args.fold, cfg_, dns_, hp, only, verbose=True)
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
            ri = {k: float(r.get(k, np.nan)) for k in KEYS}
            ri["completed"] = ab is None
            ri["abort"] = ab
            runs.append(ri)
            tail = ("" if ab is None
                    else "   **ABORTED step %d**" % ab["abort_step"])
            print(f"    {tag} run {i + 1}/{n}: "
                  f"phys={ri['phys_score']:.5f} "
                  f"patch={ri['patch_ratio']:.4f}{tail}", flush=True)
            # reprise apres CHAQUE execution, pas seulement en fin de
            # condition : une execution coute ~7 min, une condition ~35, et
            # le conteneur est recycle sans preavis. La granularite fine ne
            # coute qu'une ecriture JSON.
            if on_run is not None:
                on_run(runs, i + 1, n)
        return runs

    op = os.path.join(RESULTS_DIR,
                      f"t22_unseen_{args.mode}_{args.fold}.json")

    # ---- reprise depuis un point de sauvegarde -----------------------
    # Un fold coute ~4 h sur `kh` et `ot` ; le conteneur vit ~1 h 30. Sans
    # reprise, le point de sauvegarde ne fait que CONSERVER les donnees, il
    # ne permet pas de terminer : chaque relance repartait du tirage 1, et
    # ces deux folds ne pouvaient donc structurellement jamais aboutir.
    #
    # Les tirages repris viennent d'un AUTRE processus. C'est sans effet sur
    # la statistique — le bras est non deterministe (D11), les tirages sont
    # i.i.d. et le bras classique est exactement reproductible — mais cela
    # ne doit pas etre invisible : `resumed_from_checkpoint` et
    # `n_runs_resumed` l'enregistrent.
    _resume = {}
    if os.path.exists(op) and not args.no_resume:
        try:
            prev = json.load(open(op))
        except ValueError:
            prev = None
        if prev is not None and prev.get("status") == "partial":
            same = checkpoint_is_reusable(prev, args)
            if same:
                _resume = prev.get("arms", {})
                print(f"  RESUMING from checkpoint "
                      f"({prev.get('partial_stage')})", flush=True)
            else:
                # une reprise sous une AUTRE configuration melangerait des
                # tirages incomparables : on refuse plutot que de deviner
                print("  checkpoint found but its configuration differs; "
                      "starting over", flush=True)

    def resume_runs(arm, cond, n):
        """Tirages deja faits pour ce bras/condition, tronques a n."""
        got = _resume.get(arm, {}).get(f"{cond}_runs", []) or []
        return got[:n]

    out["resumed_from_checkpoint"] = bool(_resume)
    out["n_runs_resumed"] = sum(
        len(v.get(f"{c}_runs", []) or [])
        for v in _resume.values() for c in ("canonical", "unseen"))

    def checkpoint(stage):
        """Ecrit l'etat PARTIEL, explicitement marque comme tel.

        Ces executions durent ~2 h 30 sur les folds couteux, le conteneur
        est recycle toutes les ~1 h 30, et l'artefact n'etait ecrit qu'a la
        toute fin : `ot` et `kh` ont ainsi ete perdus trois fois, chaque
        fois integralement. Un point de reprise apres chaque condition ne
        coute rien et borne la perte a une seule execution.

        Le marquage est la partie qui compte. Un artefact partiel qui
        ressemblerait a un artefact complet serait precisement le motif que
        cette campagne traque, introduit par la mesure censee la sauver :
        `status` vaut donc `"partial"` et `partial_stage` dit ou l'on en
        etait. `t24` et `t16` ne lisent que `status == "completed"` ou
        `"total_abort"`.
        """
        snap = dict(out)
        snap["status"] = "partial"
        snap["partial_stage"] = stage
        snap["partial_warning"] = (
            "INCOMPLETE RUN — written as a resume point after a container "
            "reclaim. Arm statistics present here are computed over the "
            "draws finished so far and MUST NOT be quoted.")
        snap.update(provenance.finish(prov))
        snap["cli_args"] = vars(args)
        snap["wall_s"] = time.time() - t0
        json.dump(snap, open(op, "w"), indent=1)
        print(f"    [checkpoint] partial state saved after {stage}",
              flush=True)

    for arm, hp, only in (("qhas", hp_q, False), ("classical", hp_c, True)):
        # le bras classique ne comporte aucun echantillonnage (T20 : etendue
        # exactement nulle sur 8 rejeux), 2 executions suffisent en controle
        n = args.repeats if arm == "qhas" else min(2, args.repeats)
        out["arms"][arm] = {"status": "in_progress"}

        def _tick(cond):
            def cb(runs_so_far, k, total):
                out["arms"][arm][f"{cond}_runs"] = list(runs_so_far)
                checkpoint(f"{arm}/{cond} {k}/{total}")
            return cb

        can = repeat(hp, only, cfg, dns_can, n, f"[{arm}] canonical",
                     on_run=_tick("canonical"),
                     done=resume_runs(arm, "canonical", n))
        out["arms"][arm]["canonical_runs"] = can
        uns = repeat(hp, only, cfg_uns, dns_uns, n, f"[{arm}] unseen",
                     on_run=_tick("unseen"),
                     done=resume_runs(arm, "unseen", n))
        out["arms"][arm]["unseen_runs"] = uns
        # une trajectoire avortee n'est pas un point de mesure : on la
        # compte et on l'exclut, jamais on ne la moyenne avec les autres
        can_ok = [r for r in can if r["completed"]]
        uns_ok = [r for r in uns if r["completed"]]
        n_ab = (len(can) - len(can_ok)) + (len(uns) - len(uns_ok))
        if n_ab:
            print(f"    [{arm}] {n_ab} run(s) ABORTED, excluded from stats",
                  flush=True)
        if not can_ok or not uns_ok:
            # Un bras qui avorte sur la TOTALITE d'une condition est un
            # resultat, pas une panne : il dit que le point de fonctionnement
            # ne tient pas la trajectoire. Sortir ici sans rien ecrire rendait
            # ce resultat indiscernable d'une execution jamais lancee. On
            # enregistre donc le constat, avec un ratio NON DEFINI -- jamais
            # un nombre reconstitue a partir des tirages avortes.
            #
            # Et surtout on POURSUIT avec l'autre bras : la question qui
            # compte n'est pas « Q-HAS avorte-t-il ? » mais « avorte-t-il la
            # ou le bras classique, AU MEME SEUIL, tient la trajectoire ? ».
            # S'arreter ici laissait justement cette comparaison non mesuree.
            out["arms"][arm] = {
                "n_runs": n,
                "n_completed_canonical": len(can_ok),
                "n_completed_unseen": len(uns_ok),
                "n_aborted": int(n_ab),
                "canonical_runs": can, "unseen_runs": uns,
                "status": "total_abort",
                "degradation_ratio": float("nan"),
            }
            print(f"    [{arm}] EVERY run aborted on one condition — no "
                  f"operating point; continuing to the other arm", flush=True)
            continue
        mc = float(np.mean([r["phys_score"] for r in can_ok]))
        mu = float(np.mean([r["phys_score"] for r in uns_ok]))
        sc = (float(np.std([r["phys_score"] for r in can_ok], ddof=1))
              if len(can_ok) > 1 else 0.0)
        su = (float(np.std([r["phys_score"] for r in uns_ok], ddof=1))
              if len(uns_ok) > 1 else 0.0)
        out["arms"][arm] = {
            "n_runs": n,
            "n_completed_canonical": len(can_ok),
            "n_completed_unseen": len(uns_ok),
            "n_aborted": int(n_ab),
            "canonical_runs": can, "unseen_runs": uns,
            "canonical": {k: float(np.mean([r[k] for r in can_ok]))
                          for k in KEYS},
            "unseen": {k: float(np.mean([r[k] for r in uns_ok]))
                       for k in KEYS},
            "canonical_phys_sd": sc, "unseen_phys_sd": su,
            "status": "completed",
            "degradation_ratio": float(mu / mc) if mc else float("nan"),
        }
        print(f"  [{arm}] canonical phys={mc:.5f}+-{sc:.5f}  "
              f"unseen phys={mu:.5f}+-{su:.5f}  "
              f"degradation x{mu / mc if mc else float('nan'):.2f}",
              flush=True)

    dead = [a for a in ("qhas", "classical")
            if out["arms"][a].get("status") == "total_abort"]
    print("\n  " + "-" * 78)
    if dead:
        # Un ratio de degradation n'existe pas quand un bras n'a aucune
        # execution valide. Le comparatif est donc SANS OBJET et doit le
        # rester : `qhas_degrades_more` reste absent plutot que faux.
        out["status"] = "total_abort"
        out["total_abort_arms"] = dead
        alive = [a for a in ("qhas", "classical") if a not in dead]
        print(f"  no degradation ratio: the {', '.join(dead)} arm(s) aborted "
              f"on every draw of one condition")
        if alive:
            # NE PAS ecrire « au meme point de fonctionnement » : c'est faux.
            # En mode leak-free le bras QAOA tourne au seuil classique REGLE
            # du fold, tandis que le bras classique de controle tourne au
            # seuil BUDGET-APPARIE (--matched-reference). Les deux seuils
            # different, souvent beaucoup (rotor : 0.5864 contre 0.0969).
            # Affirmer l'egalite du point de fonctionnement aurait ete
            # exactement le motif traque ici : une phrase qui ne decrit pas
            # le calcul qu'elle accompagne.
            print(f"  the {', '.join(alive)} arm(s) completed "
                  f"{out['arms'][alive[0]]['n_completed_canonical']}"
                  f"+{out['arms'][alive[0]]['n_completed_unseen']} draws "
                  f"at threshold {hp_c.get('threshold_amr'):.4f}, "
                  f"against the QAOA arm's {hp_q.get('threshold_amr'):.4f} "
                  f"— DIFFERENT operating points, so this is not evidence "
                  f"of an arm-specific instability at equal budget")
    else:
        dq = out["arms"]["qhas"]["degradation_ratio"]
        dc = out["arms"]["classical"]["degradation_ratio"]
        out["status"] = "completed"
        out["qhas_degrades_more"] = bool(dq > dc)
        print(f"  degradation on the unseen condition: Q-HAS x{dq:.2f}  "
              f"classical x{dc:.2f}")
        print(f"  => {'Q-HAS' if dq > dc else 'the classical rule'} degrades "
              f"more when the initial condition is new")

    out.update(provenance.finish(prov))
    out["cli_args"] = vars(args)
    out["wall_s"] = time.time() - t0
    # `status` a deja ete fixe a "completed" ou "total_abort" plus haut ;
    # cette ecriture ecrase le dernier point de reprise partiel.
    out.pop("partial_stage", None)
    out.pop("partial_warning", None)
    json.dump(out, open(op, "w"), indent=1)
    print(f"\n  saved: {os.path.basename(op)} ({time.time() - t0:.0f}s)")
    print("\nV4 Task 22 complete.")


if __name__ == "__main__":
    main()
