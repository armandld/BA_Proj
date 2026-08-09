#!/usr/bin/env python3
"""
V4 Task 16 - Agregation : table maitresse de la reponse a l'audit.

Meme role que `study/v3/t10_aggregate.py` pour V3 : une seule commande qui
rassemble tous les chiffres titres de la campagne V4, chacun accompagne de
sa valeur de REFERENCE publiee dans `study/v4/RESULTS_V4.md` et d'un statut
OK / DIFF / MISSING. La table est donc auto-verifiante : une execution sur
checkout propre doit rendre OK partout.

Couverture :
  T11   attribution quantique (panel de solveurs)
  T11b  deplacement variationnel du QAOA
  T12   equivariance et erreur d'orbite (+ plancher de bruit)
  T13   ablations causales des familles de termes (mappers v1 et v2)
  T14   validation numerique (convergence, splitting, conservation)
  T15   niveau 3 closed loop, par fold
  T15b  comparaison a budget apparie, par fold
  T15c  synthese inter-folds (regles de decision pre-enregistrees)
  T17   fenetre d'incertitude (mecanisme de l'inertie ZZ)
  T18   contrefactuel fenetre neutralisee
  T20   variance d'execution du bras Q-HAS (defaut D11)
  T22   transfert vers conditions initiales inedites + plancher

Les lignes de niveau 3 sont declarees MISSING tant que le fold n'a pas
tourne : l'agregat montre donc explicitement l'etat d'avancement de la
campagne plutot que de le masquer.

Sorties : results/v4_master_table.md / .csv / v4_master.npz
Usage :
  python study/v4/t16_aggregate_v4.py
  python study/v4/t16_aggregate_v4.py --strict   # code retour != 0 si DIFF
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
from t10_aggregate import make_row, status_of, load_npz   # v3, reutilise

TOL = 0.002


def _mean_where(d, key, mask):
    v = np.asarray(d[key], dtype=float)[mask]
    return float(np.mean(v)) if v.size else None


# -------------------------------------------------------------------
# Extracteurs par tache
# -------------------------------------------------------------------

def rows_t11(d):
    metrics = ["diagonal Hamiltonian", "exhaustive hit-optimum",
               "greedy hit-optimum", "QAOA p1 mask match",
               "QAOA p2 mask match", "cold SA hit-optimum"]
    if d is None:
        return [make_row("t11", m, None, None) for m in metrics]
    solver = np.array([str(x) for x in d["solver"]])
    out = [make_row("t11", "diagonal Hamiltonian",
                    float(bool(d["diagonal_all"])), 1.0)]
    for name, key, ref in (("exhaustive", "exhaustive", 1.0),
                           ("greedy", "greedy", 1.0),
                           ("cold SA", "sa", 0.583)):
        out.append(make_row("t11", f"{name} hit-optimum",
                            _mean_where(d, "hit", solver == key), ref))
    for reps, ref in ((1, 1.0), (2, 1.0)):
        out.append(make_row("t11", f"QAOA p{reps} mask match",
                            _mean_where(d, "match",
                                        solver == f"qaoa_p{reps}"), ref))
    return out


def rows_t11b(d):
    metrics = ["ground state uniform (fraction)", "mean variational progress",
               "progress at reps=1", "progress at reps=4"]
    if d is None:
        return [make_row("t11b", m, None, None) for m in metrics]
    reps = np.asarray(d["reps"], dtype=int)
    return [
        make_row("t11b", "ground state uniform (fraction)",
                 float(d["frac_uniform"]), 1.0),
        make_row("t11b", "mean variational progress",
                 float(d["mean_progress"]), 0.0854),
        make_row("t11b", "progress at reps=1",
                 _mean_where(d, "progress", reps == 1), 0.1588),
        make_row("t11b", "progress at reps=4",
                 _mean_where(d, "progress", reps == 4), -0.0132),
    ]


def rows_t12(d, tag, refs):
    metrics = ["classical-route orbit error",
               "ground-state-route orbit error", "solver commutation rot180",
               "ground-state reproducibility floor"]
    if d is None:
        return [make_row(f"t12/{tag}", m, None, None) for m in metrics]
    route = np.array([str(x) for x in d["route"]])
    comm_op = np.array([str(x) for x in d["comm_op"]])
    out = [
        make_row(f"t12/{tag}", "classical-route orbit error",
                 _mean_where(d, "eps_orbit", route == "classical"),
                 refs["classical"]),
        make_row(f"t12/{tag}", "ground-state-route orbit error",
                 _mean_where(d, "eps_orbit", route == "ground_state"),
                 refs["gs"]),
    ]
    eps180 = _mean_where(d, "comm_eps", comm_op == "rot180")
    out.append(make_row(f"t12/{tag}", "solver commutation rot180",
                        0.0 if (eps180 is not None and eps180 < 1e-12)
                        else eps180, 0.0))
    floor = float(np.mean(d["floor"])) if len(d.get("floor", [])) else None
    out.append(make_row(f"t12/{tag}", "ground-state reproducibility floor",
                        floor, refs["floor"]))
    return out


def rows_t13(d, mapper, refs):
    names = ["full", "no_Z", "no_ZZ", "no_ZZZZ", "Z_only"]
    if d is None:
        return [make_row(f"t13/{mapper}", f"{n} decisions changed", None,
                         None) for n in names]
    abl = np.array([str(x) for x in d["ablation"]])
    return [make_row(f"t13/{mapper}", f"{n} decisions changed",
                     _mean_where(d, "changed", abl == n), refs[n])
            for n in names]


def rows_t13_degeneracy(d, mapper, refs):
    """`n_optima` par ablation — la degenerescence de l'objectif.

    Le chiffre « 64.8/256 » figurait dans le tableau des revendications
    sans etre verifie par personne, et il etait attribue a T11 alors qu'il
    vient de T13, sur le mappeur v1 seulement (le v2 rend 1.0).
    """
    if d is None or "n_optima" not in d:
        return [make_row(f"t13/{mapper}", f"{n} n_optima", None, None)
                for n in refs]
    abl = np.array([str(x) for x in d["ablation"]])
    return [make_row(f"t13/{mapper}", f"{n} n_optima",
                     _mean_where(d, "n_optima", abl == n), refs[n], tol=0.05)
            for n in refs]


def rows_t17_spearman(d):
    """La correlation rang C_edges / poids de fenetre, par scenario.

    Publiee comme « -0.37 a -0.50 », un intervalle qui EXCLUDE `ot` sans le
    dire (-0.01 la-bas). L'exclusion est defendable — la fenetre n'y laisse
    aucune masse ZZ, donc il n'y a rien a correler — mais elle doit etre
    ecrite, pas subie.
    """
    ref = {"init_kelvin_helmholtz": -0.3725, "init_orszag_tang": -0.0084,
           "init_mhd_rotor": -0.4595, "init_harris_tearing": -0.5021}
    if d is None or "spearman_c_w" not in d:
        return [make_row("t17", f"spearman C/w ({k})", None, None)
                for k in ref]
    sc = np.array([str(x) for x in d["scenario"]])
    pa = np.array([str(x) for x in d["params"]])
    out = []
    for k, v in ref.items():
        m = (sc == k) & (pa == "level3_trained")
        out.append(make_row("t17", f"spearman C/w ({k})",
                            _mean_where(d, "spearman_c_w", m), v, tol=0.005))
    return out


def rows_t14(d):
    metrics = ["self-convergence order", "temporal order with projection",
               "temporal order without projection", "max |div B| / rms|B|",
               "all conservation checks pass"]
    if d is None:
        return [make_row("t14", m, None, None) for m in metrics]
    err = np.asarray(d["conv_err"], dtype=float)
    order = (float(np.log2(err[0] / err[1]))
             if err.size >= 2 and err[1] > 0 else None)
    out = [make_row("t14", "self-convergence order", order, 1.00, tol=0.05)]
    for key, label, ref in (("split_with", "temporal order with projection",
                             1.12),
                            ("split_without",
                             "temporal order without projection", 4.00)):
        arr = np.asarray(d.get(key, np.zeros((0, 3))), dtype=float)
        o = arr[:, 2][np.isfinite(arr[:, 2])] if arr.size else np.array([])
        out.append(make_row("t14", label,
                            float(np.mean(o)) if o.size else None,
                            ref, tol=0.05))
    out.append(make_row("t14", "max |div B| / rms|B|",
                        float(np.max(d["cons_divB"])), 0.0, tol=1e-3))
    out.append(make_row("t14", "all conservation checks pass",
                        float(bool(d["all_checks_pass"])), 1.0))
    return out


def rows_level3(results_dir, folds, prefix="t15_level3"):
    """Une ligne par fold et par endpoint ; MISSING si le fold n'a pas tourne.

    Les references ne sont fixees que pour les folds deja publies dans
    RESULTS_V4.md ; les autres sont informatifs (ref=None -> OK des qu'ils
    existent), ce qui evite de figer un chiffre avant sa publication.
    """
    published = {
        "ot": dict(qhas_phys=0.1940, classical_phys=0.4845,
                   qhas_patch=0.6797, matched_phys=0.0827,
                   matched_patch=0.6412, delta_matched=0.1113),
        "kh": dict(qhas_phys=0.0070, classical_phys=0.0020,
                   qhas_patch=0.8376, matched_phys=0.0017,
                   matched_patch=0.7943, delta_matched=0.0053),
        # `rotor` et `tearing` n'avaient AUCUNE reference : leurs lignes
        # passaient donc quoi qu'il arrive. Elles sont figees ici. Le
        # 1.1731 de `rotor` est la valeur du bras classique REGLE, qui a
        # AVORTE (T19) : on l'epingle pour qu'elle ne bouge pas en silence,
        # pas parce qu'elle serait un point de mesure.
        "rotor": dict(qhas_phys=0.1678, classical_phys=1.1731,
                      qhas_patch=0.3761, matched_phys=0.0536,
                      matched_patch=0.3562, delta_matched=0.1141),
        "tearing": dict(qhas_phys=0.0185, classical_phys=0.0044,
                        qhas_patch=0.7692, matched_phys=0.0044,
                        matched_patch=0.6250, delta_matched=0.0141),
    }
    out = []
    for f in folds:
        p = os.path.join(results_dir, f"{prefix}_fold_{f}.json")
        ref = published.get(f, {})
        if not os.path.exists(p):
            for m in ("Q-HAS phys", "classical phys", "Q-HAS patch"):
                out.append(make_row(f"t15/{f}", m, None, None))
        else:
            r = json.load(open(p))
            out += [
                make_row(f"t15/{f}", "Q-HAS phys",
                         r["qhas"].get("phys_score"), ref.get("qhas_phys")),
                make_row(f"t15/{f}", "classical phys",
                         r["classical"].get("phys_score"),
                         ref.get("classical_phys")),
                make_row(f"t15/{f}", "Q-HAS patch",
                         r["qhas"].get("patch_ratio"), ref.get("qhas_patch")),
            ]
        pb = os.path.join(results_dir, f"t15b_budget_matched_{f}.json")
        if not os.path.exists(pb):
            for m in ("budget-matched classical phys",
                      "budget-matched patch", "delta phys at equal budget"):
                out.append(make_row(f"t15b/{f}", m, None, None))
        else:
            b = json.load(open(pb))
            out += [
                make_row(f"t15b/{f}", "budget-matched classical phys",
                         b["matched_classical"]["phys_score"],
                         ref.get("matched_phys")),
                make_row(f"t15b/{f}", "budget-matched patch",
                         b["matched_classical"]["patch_ratio"],
                         ref.get("matched_patch")),
                make_row(f"t15b/{f}", "delta phys at equal budget",
                         b["delta_phys_matched"], ref.get("delta_matched")),
            ]
    return out


def rows_t17(d):
    """Fenetre d'incertitude : fraction de masse ZZ conservee, au jeu de
    parametres REELLEMENT deploye. Les references sont celles publiees
    dans RESULTS_V4.md."""
    # Le jeu de parametres est NOMME dans chaque ligne : il existe deux
    # sigma « entraines » distincts (0.023 en boucle ouverte, 0.1888 pour
    # le fold Level-3) et les valeurs different de 20 ordres de grandeur.
    ref = {
        "level3_trained": {"kelvin_helmholtz": 1.142e-01,
                           "harris_tearing": 1.990e-03,
                           "mhd_rotor": 3.951e-04,
                           "orszag_tang": 9.679e-05},
        "deployed_openloop": {"kelvin_helmholtz": 1.319e-02,
                              "harris_tearing": 3.855e-154,
                              "mhd_rotor": 7.652e-28,
                              "orszag_tang": 4.187e-125},
    }
    if d is None:
        return [make_row("t17", f"ZZ mass kept ({s}, {p})", None, None)
                for p, r in ref.items() for s in r]
    scen = np.array([str(x) for x in d["scenario"]])
    par = np.array([str(x) for x in d["params"]])
    out = []
    for pset, refs in ref.items():
        for s, r in refs.items():
            m = (scen == f"init_{s}") & (par == pset)
            v = (float(np.mean(np.asarray(d["zz_mass_kept"])[m]))
                 if m.any() else None)
            # tolerance relative : ces valeurs couvrent plus de 150 ordres
            # de grandeur, une tolerance absolue les declarerait toutes OK
            out.append(make_row("t17", f"ZZ mass kept ({s}, {pset})", v, r,
                                tol=abs(0.05 * r)))
    return out


def rows_t18(d):
    """Contrefactuel : l'ablation ZZ doit rester nulle DANS LES DEUX BRAS,
    et le controle `full` doit etre nul dans chacun."""
    metrics = ["no_ZZ changed (windowed)", "no_ZZ changed (no window)",
               "no_ZZZZ changed (no window)", "control full (windowed)",
               "control full (no window)", "window's own effect"]
    if d is None:
        return [make_row("t18", m, None, None) for m in metrics]
    arm = np.array([str(x) for x in d["arm"]])
    abl = np.array([str(x) for x in d["ablation"]])
    ch = np.asarray(d["changed"], dtype=float)

    def mean_of(a, b):
        m = (arm == a) & (abl == b)
        return float(np.mean(ch[m])) if m.any() else None

    out = [
        make_row("t18", "no_ZZ changed (windowed)",
                 mean_of("windowed", "no_ZZ"), 0.0),
        make_row("t18", "no_ZZ changed (no window)",
                 mean_of("no_window", "no_ZZ"), 0.0),
        make_row("t18", "no_ZZZZ changed (no window)",
                 mean_of("no_window", "no_ZZZZ"), 0.0),
        make_row("t18", "control full (windowed)",
                 mean_of("windowed", "full"), 0.0),
        make_row("t18", "control full (no window)",
                 mean_of("no_window", "full"), 0.0),
    ]
    if "cross_changed" in d:
        out.append(make_row("t18", "window's own effect",
                            float(np.mean(d["cross_changed"])), 0.25))
    return out


def rows_t20(results_dir, folds):
    """Variance d'execution du bras Q-HAS, par fold (defaut D11).

    Les valeurs couvrent deux ordres de grandeur d'un fold a l'autre : la
    tolerance est donc RELATIVE, sinon toutes les lignes passeraient.
    Le determinisme du bras classique est le CONTROLE : s'il tombe, la
    dispersion mesuree n'est plus attribuable au seul chemin QAOA.
    """
    # Valeurs de la passe VERIFIEE (avortements captures a l'execution et
    # exclus). Les references precedentes venaient de la passe non protegee.
    ref = {
        "ot": dict(mean=0.10727, sd=0.01823, ratio=1.30),
        "kh": dict(mean=0.00320, sd=0.00203, ratio=1.90),
        "rotor": dict(mean=0.14725, sd=0.04062, ratio=2.74),
        "tearing": dict(mean=0.00801, sd=0.00193, ratio=1.81),
    }
    out, n_det, n_seen = [], 0, 0
    for f in folds:
        p = os.path.join(results_dir, f"t20_qhas_run_variance_{f}.json")
        r = ref.get(f, {})
        if not os.path.exists(p):
            for m in ("Q-HAS phys mean", "Q-HAS phys sd",
                      "ratio vs matched (mean-based)"):
                out.append(make_row(f"t20/{f}", m, None, None))
            continue
        d = json.load(open(p))
        # EXCLURE les tirages avortes : une trajectoire tronquee n'est pas
        # un point de mesure. Sans ce filtre la table republiait une moyenne
        # contaminee — sur `rotor`, 0.3328 au lieu de 0.1473, parce que les
        # deux tirages divergents y etaient moyennes avec les trois valides.
        runs = [x for x in d["qhas_runs"] if x.get("completed", True)]
        n_ab = len(d["qhas_runs"]) - len(runs)
        if len(runs) < 2:
            for m in ("Q-HAS phys mean", "Q-HAS phys sd",
                      "ratio vs matched (mean-based)"):
                out.append(make_row(f"t20/{f}", m, None, None))
            continue
        q = np.array([x["phys_score"] for x in runs], dtype=float)
        mean, sd = float(np.mean(q)), float(np.std(q, ddof=1))
        out += [
            make_row(f"t20/{f}", "Q-HAS phys mean", mean, r.get("mean"),
                     tol=max(1e-6, 0.03 * r.get("mean", 1.0))),
            make_row(f"t20/{f}", "Q-HAS phys sd", sd, r.get("sd"),
                     tol=max(1e-6, 0.05 * r.get("sd", 1.0))),
            make_row(f"t20/{f}", "ratio vs matched (mean-based)",
                     (mean / d["stored_classical_phys"]
                      if d.get("stored_classical_phys") else None),
                     r.get("ratio"),
                     tol=max(1e-3, 0.03 * r.get("ratio", 1.0))),
        ]
        n_seen += 1
        if d.get("classical_deterministic"):
            n_det += 1
    if n_seen:
        # controle global : le bras classique doit etre deterministe partout
        out.append(make_row("t20", "folds with deterministic classical arm",
                            float(n_det), float(n_seen)))
    return out


def rows_t22(results_dir, folds):
    """Transfert vers conditions initiales INEDITES (T22/T22c) et plancher
    atteignable (T22d).

    Comme partout ailleurs, les tirages AVORTES sont exclus : ils ne sont
    pas des points de mesure. Les references sont celles publiees dans
    RESULTS_V4.md apres la passe repetee a 5 tirages.
    """
    ref = {
        "ot": dict(dominated=4, z=0.02, floor_c=9.53, floor_q=14.28),
        "kh": dict(dominated=5, z=0.66, floor_c=1.39, floor_q=2.28),
        "rotor": dict(dominated=4, z=1.78, floor_c=0.98, floor_q=1.44),
        "tearing": dict(dominated=5, z=3.45, floor_c=1.11, floor_q=1.55),
    }
    out = []
    for f in folds:
        r = ref.get(f, {})
        p22 = os.path.join(results_dir, f"t22_unseen_unseen-ic_{f}.json")
        if not os.path.exists(p22):
            for m in ("dominated on unseen", "separation z",
                      "classical / floor (unseen)", "Q-HAS / floor (unseen)"):
                out.append(make_row(f"t22/{f}", m, None, None))
            continue
        d = json.load(open(p22))
        c = d["arms"]["classical"]["unseen"]
        runs = [x for x in d["arms"]["qhas"].get("unseen_runs", [])
                if x.get("completed", True)]
        dom = sum(1 for x in runs
                  if x["phys_score"] > c["phys_score"]
                  and x["patch_ratio"] > c["patch_ratio"])
        out.append(make_row(f"t22/{f}", "dominated on unseen", float(dom),
                            r.get("dominated"), tol=0.5))
        # z de separabilite : recalcule comme dans t22c
        q = d["arms"]["qhas"]
        qc, qu = q["canonical"]["phys_score"], q["unseen"]["phys_score"]
        cc, cu = d["arms"]["classical"]["canonical"]["phys_score"], c["phys_score"]
        sqc, squ = q.get("canonical_phys_sd", 0.0), q.get("unseen_phys_sd", 0.0)
        if qc and cc and qu:
            sd = abs(qu / qc) * np.sqrt((squ / qu) ** 2 + (sqc / qc) ** 2)
            z = abs(qu / qc - cu / cc) / sd if sd else float("nan")
        else:
            z = None
        out.append(make_row(f"t22/{f}", "separation z", z, r.get("z"),
                            tol=0.05))
        pfl = os.path.join(results_dir, f"t22d_unseen_floor_{f}.json")
        if os.path.exists(pfl):
            a = json.load(open(pfl))["arms"]
            out += [
                make_row(f"t22/{f}", "classical / floor (unseen)",
                         a["classical"]["unseen_over_floor"],
                         r.get("floor_c"), tol=0.05),
                make_row(f"t22/{f}", "Q-HAS / floor (unseen)",
                         a["qhas"]["unseen_over_floor"], r.get("floor_q"),
                         tol=0.05),
            ]
        else:
            for m in ("classical / floor (unseen)", "Q-HAS / floor (unseen)"):
                out.append(make_row(f"t22/{f}", m, None, None))
    return out


def rows_t23(results_dir, folds):
    """Le decompte de tete, RECALCULE depuis les artefacts.

    Il etait auparavant compose a la main dans RESULTS_V4.md et personne ne
    le verifiait — d'ou 19/20 la ou les artefacts disent 18/18 (colonnes
    transposees sur `kh`, tirages avortes comptes au denominateur sur
    `rotor`). Le nombre le plus cite de l'etude etait le seul a n'avoir
    aucun controle de transcription. Il en a un maintenant.
    """
    from t23_headline_counts import fold_counts, totals
    ref_fold = {
        "ot": (5, 5, 5), "kh": (5, 4, 4),
        "rotor": (3, 2, 2), "tearing": (5, 5, 5),
    }
    ref_total = dict(n=18, less=18, cost=16, dom=16, ab=2, cab=0, crun=8)
    out, got = [], []
    for f in folds:
        r = fold_counts(results_dir, f)
        if r is None:
            for m in ("less faithful", "costlier", "dominated"):
                out.append(make_row(f"t23/{f}", m, None, None))
            continue
        got.append(r)
        rf = ref_fold.get(f, (None, None, None))
        for m, v, rv in (("less faithful", r["less_faithful"], rf[0]),
                         ("costlier", r["costlier"], rf[1]),
                         ("dominated", r["dominated"], rf[2])):
            out.append(make_row(f"t23/{f}", m, float(v),
                                None if rv is None else float(rv), tol=0.5))
    if got:
        t = totals(got)
        for m, k, rk in (("total runs (completed)", "n_completed", "n"),
                         ("total less faithful", "less_faithful", "less"),
                         ("total costlier", "costlier", "cost"),
                         ("total dominated", "dominated", "dom"),
                         ("Q-HAS aborts at matched point", "n_aborted", "ab"),
                         ("classical replays at matched point",
                          "n_classical_runs", "crun"),
                         ("classical aborts at matched point",
                          "n_classical_aborted", "cab")):
            out.append(make_row("t23", m, float(t[k]),
                                float(ref_total[rk]), tol=0.5))
    return out


def rows_t24(results_dir, folds):
    """Le resultat sans fuite (D13 retire), recalcule depuis les artefacts.

    Ne pose de reference que pour les folds deja publies ; les autres
    apparaissent des qu'ils existent, sans figer un chiffre d'avance.
    """
    from t24_leak_free_summary import analyse
    published = {
        "tearing": {"canonical": dict(phys=3.7351, ratio=2.0771),
                    "unseen": dict(phys=2.5600, ratio=1.6954)},
        "rotor": {"canonical": dict(n_ok=0.0), "unseen": dict(phys=0.8535)},
        "kh": {"canonical": dict(phys=0.0274, ratio=1.8649),
               "unseen": dict(phys=0.1327, ratio=4.4733)},
        "ot": {"canonical": dict(phys=0.5991, ratio=1.6352),
               "unseen": dict(phys=0.5041, ratio=1.3661)},
    }
    out = []
    for f in folds:
        r = analyse(results_dir, f)
        if r is None:
            for m in ("leak-free canonical phys", "leak-free unseen phys"):
                out.append(make_row(f"t24/{f}", m, None, None))
            continue
        ref = published.get(f, {})
        for cond in ("canonical", "unseen"):
            rec = r["conditions"][cond]
            rf = ref.get(cond, {})
            if not rec["n_completed"]:
                # aucun tirage valide : on publie le CONSTAT, pas une moyenne
                out.append(make_row(f"t24/{f}", f"leak-free {cond} completed",
                                    0.0, rf.get("n_ok"), tol=0.5))
                continue
            out.append(make_row(f"t24/{f}", f"leak-free {cond} phys",
                                rec["qhas_phys"], rf.get("phys"), tol=0.01))
            if rec.get("ratio_vs_frontier") is not None:
                out.append(make_row(
                    f"t24/{f}", f"leak-free {cond} ratio vs frontier",
                    rec["ratio_vs_frontier"], rf.get("ratio"), tol=0.01))
    return out


def rows_t25(results_dir, folds):
    """Robustesse physique : decompte et refus, recalcules.

    Ce sont des nombres qui QUALIFIENT la conclusion plutot qu'ils ne la
    soutiennent ; raison de plus pour qu'ils soient verifies comme les
    autres et non recopies.
    """
    ref = {"attempted": 7.0, "vacuous": 2.0, "refused": 3.0,
           "decidable": 2.0, "direction_held": 1.0}
    att = vac = refu = dec = held = 0
    seen = False
    for f in folds:
        p = os.path.join(results_dir, f"t25_physics_robustness_{f}.json")
        if not os.path.exists(p):
            continue
        seen = True
        for c in json.load(open(p)).get("conditions", []):
            att += 1
            if c.get("skipped_as_vacuous"):
                vac += 1
            elif c.get("ratio_vs_frontier") is None:
                refu += 1
            else:
                dec += 1
                held += bool(c.get("qhas_worse"))
    if not seen:
        return [make_row("t25", m, None, None) for m in ref]
    got = dict(attempted=att, vacuous=vac, refused=refu,
               decidable=dec, direction_held=held)
    return [make_row("t25", m, float(got[m]), ref[m], tol=0.5) for m in ref]


def rows_t26(results_dir, N=256, mapper="v1"):
    """Le scan en taille : l'inertie des couplages tient-elle ?

    C'est le resultat qui repond a l'objection centrale (« a 8 qubits,
    evidemment »), donc il doit etre verifie comme les autres.
    """
    ref = {2: dict(no_ZZ=0.0, no_ZZZZ=0.0, Z_only=0.0, uniform=1.0,
                   f1_full=0.3333, f1_Zonly=0.3333, f1_classical=0.3889,
                   f1_gain_from_couplings=0.0),
           4: dict(no_ZZ=0.0, no_ZZZZ=0.0312, Z_only=0.0312, uniform=0.75,
                   f1_full=0.5199, f1_Zonly=0.5524, f1_classical=0.5524,
                   f1_gain_from_couplings=-0.0325),
           8: dict(no_ZZ=0.0469, no_ZZZZ=0.0690, Z_only=0.0794,
                   uniform=0.1667, f1_full=0.5916, f1_Zonly=0.6481,
                   f1_classical=0.6481, f1_gain_from_couplings=-0.0565)}
    p = os.path.join(results_dir, f"t26_size_scan_N{N}_{mapper}.json")
    if not os.path.exists(p):
        return [make_row(f"t26/dim{d}", m, None, None)
                for d in ref for m in ("no_ZZ", "no_ZZZZ", "Z_only")]
    got = {s["dim"]: s for s in json.load(open(p)).get("summary", [])}
    out = []
    for d, r in ref.items():
        s_ = got.get(d)
        for m in ("no_ZZ", "no_ZZZZ", "Z_only", "uniform",
                  "f1_full", "f1_Zonly", "f1_classical",
                  "f1_gain_from_couplings"):
            out.append(make_row(f"t26/dim{d}", m,
                                None if s_ is None else float(s_[m]),
                                r[m], tol=0.002))
        # le controle `full` doit valoir 0 partout, sinon rien n'est lisible
        if s_ is not None:
            out.append(make_row(f"t26/dim{d}", "full (control)",
                                float(s_["full"]), 0.0))
    # le contrôle glouton force : a dim=2 il doit rendre 0 comme l'exhaustif
    cp = os.path.join(results_dir,
                      f"t26_size_scan_N{N}_forcegreedy_{mapper}.json")
    if os.path.exists(cp):
        g = {s["dim"]: s for s in json.load(open(cp)).get("summary", [])}
        if 2 in g:
            out.append(make_row("t26/control", "greedy at dim2 (Z_only)",
                                float(g[2]["Z_only"]), 0.0))
    return out


def rows_t15c(results_dir, folds):
    """Lignes AGREGEES du niveau 3 : les comptages sur lesquels reposent
    les conclusions, recalcules ici a partir des JSON de fold plutot que
    recopies. `n_folds` figure explicitement : un comptage de victoires
    n'est lisible que rapporte au nombre de folds acheves.

    Ces lignes ne dupliquent pas celles de `rows_level3` : celles-ci
    portent sur les fold pris un a un, celles-la sur la regle de decision.
    """
    from t15c_fold_synthesis import (load_fold, primary_analysis,
                                     secondary_analysis)

    recs = [r for r in (load_fold(results_dir, f) for f in folds)
            if r is not None]
    if not recs:
        return [make_row("t15c", m, None, None) for m in
                ("folds completed", "folds where Q-HAS better (combined)",
                 "folds where Q-HAS Pareto-dominated at equal budget")]

    pri = primary_analysis(recs)
    sec = secondary_analysis(recs)
    # Ces lignes n'avaient AUCUNE reference : elles passaient donc quoi
    # qu'il arrive, y compris « 4/4 folds domines », qui est la forme sous
    # laquelle la revendication E circule. Une ligne sans reference n'est
    # pas un controle, c'est un affichage.
    out = [
        make_row("t15c", "folds completed", float(pri["n_folds"]),
                 4.0, tol=0.5),
        make_row("t15c", "folds where Q-HAS better (combined)",
                 float(pri["n_qhas_better"]), 2.0, tol=0.5),
        make_row("t15c", "budget-matched folds",
                 float(sec["n_folds"]), 4.0, tol=0.5),
        make_row("t15c", "folds where Q-HAS Pareto-dominated "
                 "at equal budget", float(sec["n_qhas_dominated"]),
                 4.0, tol=0.5),
    ]
    if sec["n_folds"]:
        out.append(make_row("t15c", "mean delta phys at equal budget "
                            "(>0 = Q-HAS worse)",
                            sec["mean_delta_phys_matched"], 0.0612,
                            tol=0.001))
    return out


# -------------------------------------------------------------------
# Collecte et sorties
# -------------------------------------------------------------------

def collect(results_dir, N=256, dim=2, folds=("ot", "kh", "rotor",
                                              "tearing")):
    rows = []
    rows += rows_t11(load_npz(os.path.join(
        results_dir, f"t11_solver_attribution_N{N}_dim{dim}.npz")))
    rows += rows_t11b(load_npz(os.path.join(
        results_dir, f"t11b_qaoa_displacement_N{N}_dim{dim}.npz")))
    rows += rows_t12(load_npz(os.path.join(
        results_dir, f"t12_equivariance_N{N}_dim8.npz")), "dim8",
        dict(classical=0.0146, gs=0.4219, floor=0.3613))
    rows += rows_t12(load_npz(os.path.join(
        results_dir, f"t12_equivariance_N{N}_dim2.npz")), "dim2",
        dict(classical=0.0, gs=0.0, floor=0.0))
    rows += rows_t13(load_npz(os.path.join(
        results_dir, f"t13_term_ablation_N{N}_dim{dim}.npz")), "v1",
        {"full": 0.0, "no_Z": 0.75, "no_ZZ": 0.0, "no_ZZZZ": 0.0,
         "Z_only": 0.0})
    rows += rows_t13_degeneracy(load_npz(os.path.join(
        results_dir, f"t13_term_ablation_N{N}_dim{dim}.npz")), "v1",
        {"full": 64.8, "no_Z": 88.0, "no_ZZ": 64.8, "no_ZZZZ": 64.8,
         "Z_only": 64.8})
    rows += rows_t13_degeneracy(load_npz(os.path.join(
        results_dir, f"t13_term_ablation_N{N}_dim{dim}_v2.npz")), "v2",
        {"full": 1.0, "no_Z": 8.0, "no_ZZ": 1.0, "no_ZZZZ": 1.0,
         "Z_only": 1.0})
    rows += rows_t17_spearman(load_npz(os.path.join(
        results_dir, "t17_uncertainty_window.npz")))
    rows += rows_t14(load_npz(os.path.join(
        results_dir, "t14_numerical_validation.npz")))
    rows += rows_level3(results_dir, folds)
    rows += rows_t15c(results_dir, folds)
    rows += rows_t17(load_npz(os.path.join(
        results_dir, "t17_uncertainty_window.npz")))
    rows += rows_t18(load_npz(os.path.join(
        results_dir, f"t18_window_counterfactual_N{N}_dim{dim}.npz")))
    rows += rows_t20(results_dir, folds)
    rows += rows_t22(results_dir, folds)
    rows += rows_t23(results_dir, folds)
    rows += rows_t24(results_dir, folds)
    rows += rows_t25(results_dir, folds)
    rows += rows_t26(results_dir, N)
    return rows


def to_markdown(rows, git_hash):
    lines = ["# V4 master table",
             "",
             f"Generated by `study/v4/t16_aggregate_v4.py` at commit "
             f"`{git_hash[:12]}`. Every row carries the reference value "
             "published in `study/v4/RESULTS_V4.md`; MISSING marks a study "
             "or Level-3 fold that has not been run yet.",
             "",
             "| task | metric | value | reference | status |",
             "|---|---|---|---|---|"]
    for r in rows:
        v = "—" if r["value"] is None else f"{r['value']:.4f}"
        ref = "—" if r["ref"] is None else f"{r['ref']:.4f}"
        lines.append(f"| {r['task']} | {r['metric']} | {v} | {ref} | "
                     f"{r['status']} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="V4 Task 16: master table")
    from config import RESULTS_DIR

    p.add_argument("--N", type=int, default=256)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--folds", nargs="+",
                   default=["ot", "kh", "rotor", "tearing"])
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on any DIFF (MISSING is tolerated: "
                        "the Level-3 campaign is incremental)")
    args = p.parse_args()

    gh = git_commit_hash()
    rows = collect(RESULTS_DIR, args.N, args.dim, tuple(args.folds))
    md = to_markdown(rows, gh)
    print(md)

    n_ok = sum(r["status"] == "OK" for r in rows)
    n_diff = sum(r["status"] == "DIFF" for r in rows)
    n_miss = sum(r["status"] == "MISSING" for r in rows)
    print(f"  rows: {len(rows)}  OK={n_ok}  DIFF={n_diff}  MISSING={n_miss}")

    md_path = os.path.join(RESULTS_DIR, "v4_master_table.md")
    csv_path = os.path.join(RESULTS_DIR, "v4_master_table.csv")
    open(md_path, "w").write(md)
    with open(csv_path, "w") as fh:
        fh.write(f"# git_hash={gh}\ntask,metric,value,reference,status\n")
        for r in rows:
            v = "" if r["value"] is None else f"{r['value']:.6f}"
            ref = "" if r["ref"] is None else f"{r['ref']:.6f}"
            fh.write(f"{r['task']},{r['metric'].replace(',', ';')},"
                     f"{v},{ref},{r['status']}\n")
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "v4_master.npz"),
        task=np.array([r["task"] for r in rows]),
        metric=np.array([r["metric"] for r in rows]),
        value=np.array([np.nan if r["value"] is None else r["value"]
                        for r in rows]),
        reference=np.array([np.nan if r["ref"] is None else r["ref"]
                            for r in rows]),
        status=np.array([r["status"] for r in rows]),
        git_hash=gh, cli_args=json.dumps(vars(args)))
    print(f"  saved: v4_master_table.md / .csv / v4_master.npz")

    if args.strict and n_diff:
        sys.exit(1)
    print("\nV4 Task 16 complete.")


if __name__ == "__main__":
    main()
