#!/usr/bin/env python3
"""
V4 Task 22 - Fuite d'apprentissage et conditions initiales INEDITES.

DEUX PROBLEMES DISTINCTS, dont un seul etait traite jusqu'ici.

(1) FUITE DE PARAMETRE (defaut D13, decouvert ici).
    Le protocole Level-3 pretend que la classe tenue n'intervient dans
    AUCUN reglage. C'est faux pour le bras QAOA. Dans
    `TrainHyperParam_v2.make_composite_objective` le seuil de decision est
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
  mode `no-leak`    : re-regle le seuil du bras QAOA sur les classes
                      d'entrainement SEULEMENT, exactement comme le bras
                      classique, puis evalue. Supprime D13. Couteux.

Les conditions inedites sont obtenues en substituant temporairement
`_init_dns_scenario` de V1 pour transmettre des parametres physiques aux
`init_*`, puis en la restaurant. V1 n'est pas modifie, et la substitution
est verifiee (la trajectoire DOIT differer de la canonique).

`orszag_tang` n'expose AUCUN parametre : sa condition initiale ne peut pas
varier sans toucher V1. Pour cette classe seule, la condition inedite est
un nombre de Reynolds different, ce qui est declare comme tel.

Sortie : results/t22_unseen_{mode}_{fold}.json
Usage :
  python study/v4/t22_unseen_conditions.py --fold kh --mode unseen-ic
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
from t19_arm_divergence_audit import (parse_abort,
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


def main():
    p = argparse.ArgumentParser(
        description="V4 T22: leak-free tuning and unseen initial conditions")
    from config import RESULTS_DIR

    p.add_argument("--fold", required=True)
    p.add_argument("--mode", choices=["unseen-ic", "no-leak"],
                   default="unseen-ic")
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
    cfg = dict(fold_scenarios(T, warn=False))[args.fold]

    hp_q = dict(rec["hyperparams"])
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

    t0 = time.time()
    out = {"fold": args.fold, "scenario": scenario, "mode": args.mode,
           "classical_reference_source": cls_src,
           "classical_tuned_arm_completed": cls_done,
           "unseen_condition": UNSEEN_CONDITIONS[scenario],
           "qaoa_threshold_leaked": bool(leaked), "arms": {}}

    # --- reference: canonical initial condition -----------------------
    dns_can = build_traces(T, args.fold, cfg, scenario, unseen=False)
    sig_can = dns_signature(*dns_can[args.fold])

    # --- unseen initial condition -------------------------------------
    dns_uns = build_traces(T, args.fold, cfg, scenario, unseen=True)
    sig_uns = dns_signature(*dns_uns[args.fold])
    changed = abs(sig_uns - sig_can) > 1e-6 * max(abs(sig_can), 1.0)
    print(f"\n  DNS signature canonical={sig_can:.6e}  unseen={sig_uns:.6e}")
    print(f"  trajectory actually changed: {changed}")
    if not changed:
        raise SystemExit(
            "the unseen condition produced an IDENTICAL trajectory; the "
            "override did not take effect and the comparison would be void.")
    out["dns_signature_canonical"] = sig_can
    out["dns_signature_unseen"] = sig_uns

    cfg_uns = unseen_config(cfg, scenario)
    KEYS = ("combined", "phys_score", "patch_ratio")

    def repeat(hp, only, cfg_, dns_, n, tag):
        """n executions, chacune verifiee CONTRE LA DIVERGENCE.

        Sans ce controle une trajectoire avortee se melange aux autres :
        sur `tearing` un tirage Q-HAS a rendu phys = 0.601 quand ses quatre
        voisins donnaient 0.0017-0.0027, soit un facteur 300, et moyenne
        comme ecart-type en devenaient absurdes (deg 14.4 +- 32.0). Le
        statut ne peut PAS etre retrouve apres coup : le bras Q-HAS n'est
        pas deterministe (D11), donc un rejeu ne reproduit pas le tirage
        fautif. Il faut le capturer AU MOMENT de l'execution.
        """
        runs = []
        for i in range(n):
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
        return runs

    for arm, hp, only in (("qhas", hp_q, False), ("classical", hp_c, True)):
        # le bras classique ne comporte aucun echantillonnage (T20 : etendue
        # exactement nulle sur 8 rejeux), 2 executions suffisent en controle
        n = args.repeats if arm == "qhas" else min(2, args.repeats)
        can = repeat(hp, only, cfg, dns_can, n, f"[{arm}] canonical")
        uns = repeat(hp, only, cfg_uns, dns_uns, n, f"[{arm}] unseen")
        # une trajectoire avortee n'est pas un point de mesure : on la
        # compte et on l'exclut, jamais on ne la moyenne avec les autres
        can_ok = [r for r in can if r["completed"]]
        uns_ok = [r for r in uns if r["completed"]]
        n_ab = (len(can) - len(can_ok)) + (len(uns) - len(uns_ok))
        if n_ab:
            print(f"    [{arm}] {n_ab} run(s) ABORTED, excluded from stats",
                  flush=True)
        if not can_ok or not uns_ok:
            raise SystemExit(f"[{arm}] every run aborted on one condition")
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
            "degradation_ratio": float(mu / mc) if mc else float("nan"),
        }
        print(f"  [{arm}] canonical phys={mc:.5f}+-{sc:.5f}  "
              f"unseen phys={mu:.5f}+-{su:.5f}  "
              f"degradation x{mu / mc if mc else float('nan'):.2f}",
              flush=True)

    dq = out["arms"]["qhas"]["degradation_ratio"]
    dc = out["arms"]["classical"]["degradation_ratio"]
    out["qhas_degrades_more"] = bool(dq > dc)
    print("\n  " + "-" * 78)
    print(f"  degradation on the unseen condition: Q-HAS x{dq:.2f}  "
          f"classical x{dc:.2f}")
    print(f"  => {'Q-HAS' if dq > dc else 'the classical rule'} degrades more "
          f"when the initial condition is new")

    out["git_hash"] = git_commit_hash()
    out["cli_args"] = vars(args)
    out["wall_s"] = time.time() - t0
    op = os.path.join(RESULTS_DIR,
                      f"t22_unseen_{args.mode}_{args.fold}.json")
    json.dump(out, open(op, "w"), indent=1)
    print(f"\n  saved: {os.path.basename(op)} ({time.time() - t0:.0f}s)")
    print("\nV4 Task 22 complete.")


if __name__ == "__main__":
    main()
