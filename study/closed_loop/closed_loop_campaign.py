#!/usr/bin/env python3
"""
V4 Task 15 - Niveau 3 : transfert EN BOUCLE FERMEE (protocole v3 section 4 ;
audit, Priorite 0 - experience decisive).

QUESTION. Les niveaux 1 et 2 evaluent le selecteur hors boucle. Le niveau 3
demande si la boucle AMR complete (TTL, warm start, retroaction par l'etat
resolu) compense ou amplifie l'echec observe en open loop, lorsqu'une classe
d'instabilite est ENTIEREMENT exclue du reglage.

PROTOCOLE (fold LOSO au niveau pipeline).
  Pour chaque classe tenue s :
    1. les hyperparametres sont regles par Optuna sur la perte composite des
       AUTRES classes uniquement (`make_composite_objective`, V1, importe) ;
       la classe s n'intervient dans AUCUN choix : ni parametres du
       Hamiltonien, ni seuil AMR ;
    2. le seuil du bras CLASSIQUE est regle sur les memes classes
       d'entrainement (`make_classical_composite_objective`), de sorte que
       les deux bras subissent la meme exclusion ;
    3. les deux bras tournent sur s avec la MEME trace DNS, le meme etat
       initial, le meme budget hybride et la meme profondeur.

ENDPOINTS. `pipeline(..., return_details=True)` renvoie
  phys_score   erreur L2 relative moyenne contre la DNS  (fidelite)
  patch_ratio  fraction de grille raffinee               (cout)
  combined     (phys + lambda*patch) / (1+lambda)        (perte composite)
On rapporte les trois : une amelioration de fidelite payee par du cout n'est
pas une amelioration. La comparaison appariee par fold utilise les
statistiques confirmatoires de `stats_confirmatory` (Holm + TOST).

DEVIATIONS PAR RAPPORT AU PROTOCOLE (a journaliser dans RESULTS) :
  - le module d'entrainement V1 expose 6 classes, pas 8 : 6 folds ;
  - le budget Optuna est un parametre (`--n-trials`) ; le protocole gele
    170 essais. Toute valeur inferieure doit etre declaree ;
  - une seule graine physique par fold (le pipeline initialise chaque
    scenario de maniere deterministe) ; la replication par graine reste
    a faire.

Sortie : results/t15_level3_fold_{scenario}.json (incremental, resumable)
         results/t15_level3_summary.npz
Usage :
  python study/v4/t15_level3_closed_loop.py --list
  python study/v4/t15_level3_closed_loop.py --folds OT --n-trials 12
  nohup python study/v4/t15_level3_closed_loop.py --n-trials 40 \
        > logs/v4/level3.log 2>&1 &
"""
import argparse, json, os, sys, time
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

# Parametres de Hamiltonien geles hors optimisation (valeurs V1 de reference).
FROZEN_DEFAULTS = dict(gamma_hydro=2.0, gamma_mag=0.5, kappa=10.0)


def _load_v1_training_module():
    """Importe le module d'entrainement V1 (scenarios, objectifs, DNS)."""
    import TrainHyperParam_v2 as T
    return T


def fold_scenarios(T, only=None, warn=True):
    """Liste DEDOUBLONNEE des classes disponibles pour les folds LOSO.

    ATTENTION (defaut du module d'entrainement V1). `T.SCENARIOS_ALL` vaut
    `SCENARIOS_ISOLATED + SCENARIOS_COMPLEX`, or SCENARIOS_ISOLATED contient
    deja `ot` et `rotor` et SCENARIOS_COMPLEX les reintroduit a l'identique :
    la liste compte 6 entrees pour 4 classes distinctes, et la perte
    composite `mean(Loss_i)` pondere donc OT et rotor deux fois plus que KH
    et tearing. Pour un fold LOSO, laisser le doublon reviendrait a garder la
    classe « tenue » dans l'entrainement, c'est-a-dire a fabriquer une fuite.
    On deduplique par cle en conservant le premier exemplaire.
    """
    seen, scen = set(), []
    dupes = []
    for k, c in T.SCENARIOS_ALL:
        if k in seen:
            dupes.append(k)
            continue
        seen.add(k)
        scen.append((k, c))
    if dupes and warn:
        print(f"  [WARNING] TrainHyperParam_v2.SCENARIOS_ALL lists {dupes} "
              f"twice; de-duplicated for LOSO (see docstring).", flush=True)
    if only:
        keep = {o.lower() for o in only}
        scen = [(k, c) for k, c in scen
                if k.lower() in keep or c["scenario"].lower() in keep]
    return scen


def _persistent_study(storage_path, study_name, seed):
    """Etude Optuna adossee a un fichier SQLite, reprise si elle existe.

    Sans cela, un essai coute ~15 min et le tuning complet ~90 min : plus
    long que la duree de vie du conteneur entre deux recyclages, donc le
    reglage ne convergeait jamais. Avec un stockage persistant, chaque essai
    TERMINE est conserve et une reprise repart du compte courant.
    """
    import optuna
    return optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed))


def _n_completed(study):
    import optuna
    return sum(t.state == optuna.trial.TrialState.COMPLETE
               for t in study.trials)


def train_params_excluding(T, dns_traces, train_list, n_trials, seed=0,
                           lambda_cost=None, verbose=False,
                           storage_path=None, study_name="qaoa"):
    """Regle les hyperparametres QAOA sur `train_list` SEULEMENT.

    Reutilise `make_composite_objective` de V1 : la fonction de perte, le
    pipeline et les bornes de recherche sont exactement ceux de
    l'entrainement V1, seule la LISTE DES SCENARIOS change. Le reglage est
    repris essai par essai lorsqu'un `storage_path` est fourni.
    """
    import optuna
    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING)
    lam = T.LAMBDA_COST_SOFT if lambda_cost is None else lambda_cost
    obj = T.make_composite_objective(
        dns_traces, train_list, split_michelson=True, lambda_cost=lam)
    if storage_path:
        study = _persistent_study(storage_path, study_name, seed)
        done = _n_completed(study)
        if done:
            print(f"  [resume] {study_name}: {done}/{n_trials} trials "
                  f"already stored", flush=True)
        todo = max(0, n_trials - done)
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        todo = n_trials
    if todo:
        study.optimize(obj, n_trials=todo, catch=(Exception,))
    best = dict(FROZEN_DEFAULTS)
    best.update(study.best_params)
    best.setdefault("threshold_amr", 0.14959824837662078)
    return best, float(study.best_value), _n_completed(study)


def train_classical_threshold_excluding(T, dns_traces, train_list, n_trials,
                                        seed=0, lambda_cost=None,
                                        storage_path=None,
                                        study_name="classical"):
    """Regle le seuil AMR du bras classique sur les memes classes.

    Sans cela, le bras classique beneficierait d'un seuil choisi en voyant
    la classe tenue : la comparaison ne serait plus appariee. Meme reprise
    par essai que le bras QAOA.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lam = T.LAMBDA_COST_SOFT if lambda_cost is None else lambda_cost
    obj = T.make_classical_composite_objective(
        dns_traces, train_list, lambda_cost=lam)
    if storage_path:
        study = _persistent_study(storage_path, study_name, seed)
        todo = max(0, n_trials - _n_completed(study))
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        todo = n_trials
    if todo:
        study.optimize(obj, n_trials=todo, catch=(Exception,))
    return dict(study.best_params), float(study.best_value)


def run_arm(T, key, config, dns_traces, hyperparams, classical_only,
            lambda_cost=None, verbose=False):
    """Execute UN bras du pipeline complet sur la classe tenue.

    Les deux bras recoivent la meme trace DNS, le meme hot start, le meme
    budget hybride et la meme profondeur : seule la regle de decision change.
    """
    from pipeline import pipeline
    lam = T.LAMBDA_COST_SOFT if lambda_cost is None else lambda_cost
    dns_trace, hot_start = dns_traces[key]
    DT = config["DT"]
    t0 = time.time()
    res = pipeline(
        N=config["N"], VQA_N=2, T_MAX=config["T_MAX"], DT=DT,
        HYBRID=int(config["HYBRID_DT"] / DT),
        verbose=verbose, argus=T.create_argus(config),
        hyperparams=hyperparams, lambda_cost=lam, trial=None,
        dns_trace=dns_trace, hot_start_state=hot_start,
        min_patch_size=config.get("min_patch_size", 6),
        max_depth_override=config.get("max_depth_override", None),
        scenario=config["scenario"], return_details=True,
        classical_only=classical_only,
    )
    if not isinstance(res, dict):
        res = {"combined": float(res)}
    res = dict(res)
    res["wall_s"] = time.time() - t0
    res.pop("field_errors", None)
    return res


def _tune_ckpt_path(results_dir, prefix, held_key):
    return os.path.join(results_dir, f"{prefix}_tuning_{held_key}.json")


def load_or_tune(T, results_dir, prefix, held_key, dns_train, train_list,
                 n_trials, n_cls, seed, lambda_cost, verbose):
    """Regle les hyperparametres, ou relit un checkpoint existant.

    Le tuning est de loin l'etape la plus couteuse (mesure : ~87 min pour
    4 essais a N=256). Sans checkpoint, toute interruption la fait
    recommencer a zero. Le checkpoint est ecrit DES que le tuning est
    termine, avant l'execution des bras.
    """
    path = _tune_ckpt_path(results_dir, prefix, held_key)
    if os.path.exists(path):
        ck = json.load(open(path))
        print(f"  [resume] tuning checkpoint found -> "
              f"{os.path.basename(path)}", flush=True)
        if ck.get("classical_params") is not None:
            return (ck["hyperparams"], ck["best_train_loss"], ck["n_trials"],
                    ck["classical_params"], ck["classical_train_loss"],
                    ck.get("t_tune", 0.0), ck.get("t_tune_classical", 0.0))
        # checkpoint partiel : le tuning QAOA est acquis, le classique reste
        print("  [resume] classical arm not tuned yet; completing it",
              flush=True)
        t0 = time.time()
        cls_params, cls_loss = train_classical_threshold_excluding(
            T, dns_train, train_list, n_cls, seed=seed,
            lambda_cost=lambda_cost,
            storage_path=_tune_ckpt_path(results_dir, prefix, held_key)
            .replace("_tuning_", "_optuna_").replace(".json", ".db"),
            study_name=f"classical_{held_key}")
        t_tune_c = time.time() - t0
        ck.update(classical_params=cls_params, classical_train_loss=cls_loss,
                  t_tune_classical=t_tune_c)
        json.dump(ck, open(path, "w"), indent=1, default=float)
        print(f"  classical tuning: best loss {cls_loss:.4f}, params "
              f"{cls_params}, {t_tune_c:.0f}s", flush=True)
        return (ck["hyperparams"], ck["best_train_loss"], ck["n_trials"],
                cls_params, cls_loss, ck.get("t_tune", 0.0), t_tune_c)

    db = os.path.join(results_dir, f"{prefix}_optuna_{held_key}.db")
    t0 = time.time()
    hp, best_loss, n_done = train_params_excluding(
        T, dns_train, train_list, n_trials, seed=seed,
        lambda_cost=lambda_cost, verbose=verbose,
        storage_path=db, study_name=f"qaoa_{held_key}")
    t_tune = time.time() - t0
    print(f"  QAOA tuning: {n_done} trials, best composite loss "
          f"{best_loss:.4f}, {t_tune:.0f}s", flush=True)

    t0 = time.time()
    cls_params, cls_loss = train_classical_threshold_excluding(
        T, dns_train, train_list, n_cls, seed=seed, lambda_cost=lambda_cost,
        storage_path=db, study_name=f"classical_{held_key}")
    t_tune_c = time.time() - t0
    print(f"  classical tuning: best loss {cls_loss:.4f}, params "
          f"{cls_params}, {t_tune_c:.0f}s", flush=True)

    json.dump(dict(hyperparams=hp, best_train_loss=best_loss,
                   n_trials=n_done, classical_params=cls_params,
                   classical_train_loss=cls_loss, t_tune=t_tune,
                   t_tune_classical=t_tune_c),
              open(path, "w"), indent=1, default=float)
    print(f"  tuning checkpoint saved -> {os.path.basename(path)}",
          flush=True)
    return (hp, best_loss, n_done, cls_params, cls_loss, t_tune, t_tune_c)


def run_fold(T, held_key, held_cfg, all_scen, n_trials, n_trials_classical,
             seed=0, lambda_cost=None, verbose=False,
             results_dir=None, prefix="t15_level3"):
    """Un fold LOSO complet : reglage hors classe, puis les deux bras."""
    train_list = [(k, c) for k, c in all_scen if k != held_key]
    print(f"\n{'='*84}\n  FOLD held-out = {held_key} "
          f"({held_cfg['scenario']})  |  tuned on "
          f"{[k for k, _ in train_list]}\n{'='*84}", flush=True)

    t0 = time.time()
    dns_train = T._precompute_dns_for(train_list, label=f"train/{held_key}")
    dns_held = T._precompute_dns_for([(held_key, held_cfg)],
                                     label=f"held/{held_key}")
    t_dns = time.time() - t0
    print(f"  DNS traces ready in {t_dns:.0f}s", flush=True)

    (hp, best_loss, n_done, cls_params, cls_loss,
     t_tune, t_tune_c) = load_or_tune(
        T, results_dir, prefix, held_key, dns_train, train_list,
        n_trials, n_trials_classical, seed, lambda_cost, verbose)
    print(f"  params: "
          f"{ {k: round(v, 4) for k, v in hp.items() if isinstance(v, float)} }",
          flush=True)

    hp_classical = dict(hp)
    hp_classical.update(cls_params)

    q = run_arm(T, held_key, held_cfg, dns_held, hp, False,
                lambda_cost=lambda_cost, verbose=verbose)
    print(f"  [Q-HAS]     combined={q.get('combined'):.4f} "
          f"phys={q.get('phys_score', float('nan')):.4f} "
          f"patch={q.get('patch_ratio', float('nan')):.4f} "
          f"({q['wall_s']:.0f}s)", flush=True)
    c = run_arm(T, held_key, held_cfg, dns_held, hp_classical, True,
                lambda_cost=lambda_cost, verbose=verbose)
    print(f"  [classical] combined={c.get('combined'):.4f} "
          f"phys={c.get('phys_score', float('nan')):.4f} "
          f"patch={c.get('patch_ratio', float('nan')):.4f} "
          f"({c['wall_s']:.0f}s)", flush=True)

    return dict(
        fold=held_key, scenario=held_cfg["scenario"],
        train_on=[k for k, _ in train_list],
        n_trials=n_done, best_train_loss=best_loss,
        classical_params=cls_params, classical_train_loss=cls_loss,
        hyperparams={k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in hp.items()},
        qhas=q, classical=c,
        t_dns=t_dns, t_tune=t_tune, t_tune_classical=t_tune_c,
    )


def summarise(records):
    """Comparaison appariee par fold + Holm + equivalence TOST."""
    keys = ("combined", "phys_score", "patch_ratio")
    out = {}
    for k in keys:
        q = np.array([r["qhas"].get(k, np.nan) for r in records], float)
        c = np.array([r["classical"].get(k, np.nan) for r in records], float)
        m = np.isfinite(q) & np.isfinite(c)
        d = q[m] - c[m]
        out[k] = dict(
            q=q, c=c, delta=d,
            mean_delta=float(np.mean(d)) if len(d) else np.nan,
            n_qhas_better=int(np.sum(d < 0)), n=int(len(d)))
    return out


def main():
    p = argparse.ArgumentParser(
        description="V4 Task 15: Level-3 closed-loop LOSO")
    from config import RESULTS_DIR

    p.add_argument("--folds", nargs="+", default=None,
                   help="cles de scenario a traiter (defaut : toutes)")
    p.add_argument("--n-trials", type=int, default=40,
                   help="essais Optuna par fold (protocole : 170)")
    p.add_argument("--n-trials-classical", type=int, default=None,
                   help="defaut : moitie de --n-trials")
    p.add_argument("--lambda-cost", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--list", action="store_true",
                   help="liste les folds disponibles et sort")
    p.add_argument("--out-prefix", default="t15_level3")
    p.add_argument("--smoke", action="store_true",
                   help="valide le CHEMIN DE CODE de bout en bout a cout "
                        "reduit (N et T_MAX rabaisses). Resultats non "
                        "scientifiques : sert uniquement a de-risquer un "
                        "run long.")
    p.add_argument("--smoke-N", type=int, default=64)
    p.add_argument("--smoke-tmax", type=float, default=0.4)
    args = p.parse_args()

    T = _load_v1_training_module()
    all_scen = fold_scenarios(T)
    if args.list:
        print("Available LOSO folds (key -> scenario, N, T_MAX, Re):")
        for k, c in all_scen:
            print(f"  {k:<14} -> {c['scenario']:<20} N={c['N']} "
                  f"T_MAX={c['T_MAX']} Re={c['Re']}")
        return

    if args.smoke:
        print(f"  [SMOKE] scaling every scenario to N={args.smoke_N}, "
              f"T_MAX={args.smoke_tmax}; results are NOT scientific.")
        for _, c in all_scen:
            c["N"] = args.smoke_N
            c["T_MAX"] = args.smoke_tmax
            c["T_START"] = min(c.get("T_START", 0.0), args.smoke_tmax / 2)
            c["K_opt"] = min(c.get("K_opt", 30), 8)
            c["max_depth_override"] = 2
    todo = fold_scenarios(T, args.folds)
    n_cls = args.n_trials_classical
    if n_cls is None:
        n_cls = max(4, args.n_trials // 2)

    print("=" * 88)
    print("  V4 Task 15: LEVEL 3 - closed-loop LOSO transfer")
    print(f"  folds={[k for k, _ in todo]}  optuna trials/fold={args.n_trials}"
          f" (classical {n_cls})")
    print("  Held-out class excluded from ALL tuning, both arms.")
    if args.n_trials < 170:
        print(f"  DEVIATION: protocol freezes 170 trials; running "
              f"{args.n_trials}. Logged in the output.")
    print("=" * 88, flush=True)

    records = []
    for key, cfg in todo:
        path = os.path.join(RESULTS_DIR, f"{args.out_prefix}_fold_{key}.json")
        if os.path.exists(path):
            print(f"\n  [resume] fold {key} already done -> {path}",
                  flush=True)
            records.append(json.load(open(path)))
            continue
        try:
            rec = run_fold(T, key, cfg, all_scen, args.n_trials, n_cls,
                           seed=args.seed, lambda_cost=args.lambda_cost,
                           verbose=args.verbose, results_dir=RESULTS_DIR,
                           prefix=args.out_prefix)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  FOLD {key} FAILED: {exc}", flush=True)
            continue
        rec["git_hash"] = git_commit_hash()
        rec["cli_args"] = vars(args)
        json.dump(rec, open(path, "w"), indent=1, default=float)
        print(f"  saved fold -> {os.path.basename(path)}", flush=True)
        records.append(rec)

    if not records:
        print("\nno completed fold."); return

    s = summarise(records)
    print("\n" + "=" * 88)
    print("  LEVEL-3 RESULTS (paired per fold; negative delta favours Q-HAS)")
    print(f"  {'fold':<14} {'phys Q':>9} {'phys C':>9} {'d phys':>9} "
          f"{'patch Q':>9} {'patch C':>9} {'d comb':>9}")
    for i, r in enumerate(records):
        print(f"  {r['fold']:<14} "
              f"{r['qhas'].get('phys_score', np.nan):>9.4f} "
              f"{r['classical'].get('phys_score', np.nan):>9.4f} "
              f"{s['phys_score']['q'][i] - s['phys_score']['c'][i]:>+9.4f} "
              f"{r['qhas'].get('patch_ratio', np.nan):>9.4f} "
              f"{r['classical'].get('patch_ratio', np.nan):>9.4f} "
              f"{s['combined']['q'][i] - s['combined']['c'][i]:>+9.4f}")
    print("  " + "-" * 84)
    for k in ("combined", "phys_score", "patch_ratio"):
        v = s[k]
        print(f"  {k:<14} mean delta = {v['mean_delta']:+.4f}   "
              f"Q-HAS better on {v['n_qhas_better']}/{v['n']} folds")

    if s["combined"]["n"] >= 2:
        d = s["combined"]["delta"]
        from scipy import stats as _st
        t, pv = _st.ttest_1samp(d, 0.0)
        holm = holm_correction([pv])
        margin = 0.05 * float(np.mean(np.abs(s["combined"]["c"])) + 1e-12)
        eq = tost_equivalence(s["combined"]["q"], s["combined"]["c"],
                              margin=margin)
        print(f"\n  paired t-test on combined delta: p={pv:.4f} "
              f"(Holm-adjusted {holm['p_adjusted'][0]:.4f})")
        print(f"  TOST equivalence at margin {margin:.4f}: "
              f"{'EQUIVALENT' if eq['equivalent'] else 'not established'} "
              f"(p={eq['p_tost']:.4f})")

    out = os.path.join(RESULTS_DIR, f"{args.out_prefix}_summary.npz")
    np.savez_compressed(
        out,
        fold=np.array([r["fold"] for r in records]),
        combined_q=s["combined"]["q"], combined_c=s["combined"]["c"],
        phys_q=s["phys_score"]["q"], phys_c=s["phys_score"]["c"],
        patch_q=s["patch_ratio"]["q"], patch_c=s["patch_ratio"]["c"],
        n_trials=args.n_trials, git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nV4 Task 15 complete.")


if __name__ == "__main__":
    main()
