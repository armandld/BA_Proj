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
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "v3"))
sys.path.insert(0, _HERE)

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
                   qhas_patch=0.8376),
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
    out = [
        make_row("t15c", "folds completed", float(pri["n_folds"]), None),
        make_row("t15c", "folds where Q-HAS better (combined)",
                 float(pri["n_qhas_better"]), None),
        make_row("t15c", "budget-matched folds",
                 float(sec["n_folds"]), None),
        make_row("t15c", "folds where Q-HAS Pareto-dominated "
                 "at equal budget", float(sec["n_qhas_dominated"]), None),
    ]
    if sec["n_folds"]:
        out.append(make_row("t15c", "mean delta phys at equal budget "
                            "(>0 = Q-HAS worse)",
                            sec["mean_delta_phys_matched"], None))
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
    rows += rows_t14(load_npz(os.path.join(
        results_dir, "t14_numerical_validation.npz")))
    rows += rows_level3(results_dir, folds)
    rows += rows_t15c(results_dir, folds)
    rows += rows_t17(load_npz(os.path.join(
        results_dir, "t17_uncertainty_window.npz")))
    rows += rows_t18(load_npz(os.path.join(
        results_dir, f"t18_window_counterfactual_N{N}_dim{dim}.npz")))
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
