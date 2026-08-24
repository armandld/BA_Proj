#!/usr/bin/env python3
"""T29 — les ecarts LOSO avec un intervalle de confiance, et pas de verdict sans.

Pourquoi
--------
`phase11b_loso.py` imprime des conclusions du type
« stencil > site by +0.033 under LOSO ==> neighbourhood couplings help for
transfer » a partir d'une moyenne sur QUATRE folds dont l'ecart-type vaut
0.29, et dont deux valent exactement 0.400 (tout positif) ou 0.000 (tout
negatif) — des constantes, pas des modeles.

Ce script recalcule les folds et re-echantillonne les trajectoires physiques
completes (scenario, Re, graine). Un verdict n'est emis QUE si l'intervalle
ne contient pas zero.
Sinon la ligne dit « indecidable », ce qui est l'information reelle.

Il signale aussi les folds ou le modele s'est effondre sur une constante :
un F1 obtenu en predisant toujours la meme classe n'est pas une performance.

Reutilise `_gather_scenario`, `make_model`, `fit_eval`, `best_threshold_f1`
de la chaine existante — rien n'est reimplemente.

Usage
-----
  python study/h2b_prediction/t29_loso_delta_ci.py --dim 4 16
  python study/h2b_prediction/t29_loso_delta_ci.py --dim 16 --label-suffix _globalthr
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score

# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------

from config import (
    DNS_N, PHYSICS_SEEDS, RESULTS_DIR, RE_VALUES, SCENARIOS,
)
from data_catalog import labelled_trajectory_paths
from h2b_ceiling_random_split import make_model, fit_eval, best_threshold_f1
from h2b_loso_transfer import _gather_scenario
from h2b_feature_selection import git_commit_hash
from stats import bootstrap_by_trajectory

B_DEFAULT = 1000


def _f1(y, pred):
    return float(f1_score(y, pred, zero_division=0))


def delta_ci(y, pred_a, pred_b, traj, B, seed):
    """IC bootstrap par trajectoire sur F1(b) - F1(a).

    Le bloc de reechantillonnage est une trajectoire physique complete.
    """
    idx = np.arange(len(y))

    def stat(sel):
        sel = sel.astype(int)
        return _f1(y[sel], pred_b[sel]) - _f1(y[sel], pred_a[sel])

    return bootstrap_by_trajectory(idx.astype(float), traj, B=B,
                                   statistic=stat, seed=seed)


def constant_predictor(pred):
    """True si le modele a predit une seule classe sur tout le fold."""
    return pred.min() == pred.max()


def run_dim(dim, N, re_values, scenarios, physics_seeds, max_snaps, seed,
            B, suffix):
    by_scene = labelled_trajectory_paths(
        RESULTS_DIR, scenarios, re_values, N, dim, physics_seeds,
        label_suffix=suffix)
    if len(by_scene) < 3:
        raise SystemExit(
            f"dim={dim}{suffix}: {len(by_scene)} scenarios disponibles, il en "
            "faut au moins 3 pour un LOSO")

    data = {}
    for sc, rows in by_scene.items():
        pieces = []
        for trajectory_id, row in enumerate(rows):
            Xs, Xn, Y, S = _gather_scenario([row], dim, max_snaps)
            pieces.append((Xs, Xn, Y, S,
                           np.full(len(Y), trajectory_id, dtype=int)))
        data[sc] = dict(
            X_site=np.concatenate([piece[0] for piece in pieces]),
            X_sten=np.concatenate([piece[1] for piece in pieces]),
            Y=np.concatenate([piece[2] for piece in pieces]),
            S=np.concatenate([piece[3] for piece in pieces]),
            traj=np.concatenate([piece[4] for piece in pieces]),
        )

    print(f"\n{'='*78}")
    print(f"  dim={dim}  N={N}  labels='{suffix or 'par scenario'}'  "
          f"bootstrap B={B}")
    print(f"{'='*78}")
    print(f"  {'held-out':<18} {'n_val':>8} {'prev':>6} "
          f"{'F1_cls':>7} {'F1_site':>8} {'F1_sten':>8} "
          f"{'d_sten-site [IC95]':>26}  etat")
    print("  " + "-" * 90)

    out = []
    for held in by_scene:
        tr = [sc for sc in by_scene if sc != held]
        Xtr_site = np.concatenate([data[sc]["X_site"] for sc in tr])
        Xtr_sten = np.concatenate([data[sc]["X_sten"] for sc in tr])
        Ytr = np.concatenate([data[sc]["Y"] for sc in tr])
        Str = np.concatenate([data[sc]["S"] for sc in tr])

        Yva, Sva = data[held]["Y"], data[held]["S"]
        traj = data[held]["traj"]

        thr_cls, _ = best_threshold_f1(Str, Ytr)
        pred_cls = (Sva > thr_cls).astype(int)

        r_site = fit_eval(make_model("gbt", seed), Xtr_site, Ytr,
                          data[held]["X_site"], Yva)
        r_sten = fit_eval(make_model("gbt", seed), Xtr_sten, Ytr,
                          data[held]["X_sten"], Yva)
        pred_site = (r_site["p"] > r_site["thr"]).astype(int)
        pred_sten = (r_sten["p"] > r_sten["thr"]).astype(int)

        ci = delta_ci(Yva, pred_site, pred_sten, traj, B, seed)
        # D-79 : `constant` decide qui VOTE, et la quantite votee est
        # F1(sten) - F1(site). Le predicteur classique n'y entre pas : son
        # effondrement ne dit rien des deux modeles compares. Il etait
        # pourtant dans la meme liste, et ecartait des folds dont les deux
        # bras compares etaient sains — mesure sur le rejeu qui a produit
        # l'artefact publie : `kelvin_helmholtz` ecarte avec un IC95
        # entierement negatif (-0,027 [-0,050, -0,001]), donc decisif.
        # L'effondrement classique reste imprime : c'est une information,
        # pas un motif d'exclusion.
        compared = [n for n, p in (("site", pred_site), ("sten", pred_sten))
                    if constant_predictor(p)]
        flags = list(compared)
        if constant_predictor(pred_cls):
            flags.append("cls")
        state = ("constant: " + ",".join(flags)) if flags else "ok"

        print(f"  {held:<18} {len(Yva):>8d} {Yva.mean():>6.3f} "
              f"{_f1(Yva, pred_cls):>7.3f} {_f1(Yva, pred_site):>8.3f} "
              f"{_f1(Yva, pred_sten):>8.3f} "
              f"{ci['estimate']:>+9.3f} [{ci['ci_low']:+.3f},{ci['ci_high']:+.3f}]"
              f"  {state}")

        out.append(dict(
            held=held, dim=dim, suffix=suffix, n_val=int(len(Yva)),
            prevalence=float(Yva.mean()),
            f1_class=_f1(Yva, pred_cls), f1_site=_f1(Yva, pred_site),
            f1_sten=_f1(Yva, pred_sten),
            delta=float(ci["estimate"]), ci_low=float(ci["ci_low"]),
            ci_high=float(ci["ci_high"]),
            constant=",".join(flags),
            # ce qui decide le vote : seuls les deux bras compares
            constant_compared=",".join(compared),
            n_traj=int(ci["n_traj"]),
        ))
    return out


def verdict(rows):
    """Emet une conclusion UNIQUEMENT si les IC la soutiennent.

    Regle : un fold soutient « les voisins aident » si son IC95 est
    entierement > 0, « ils nuisent » s'il est entierement < 0. Les folds ou
    l'un des DEUX BRAS COMPARES (site, stencil) s'est effondre sur une
    constante ne votent pas — leur F1 ne mesure pas un modele.

    D-79 : le predicteur classique etait compte dans cette regle alors qu'il
    n'entre pas dans F1(sten) - F1(site). Un fold dont les deux modeles
    compares etaient sains pouvait donc etre ecarte parce qu'un TROISIEME
    predicteur, etranger a la comparaison, etait constant.
    """
    voting = [r for r in rows if not r.get("constant_compared", r["constant"])]
    helps = [r for r in voting if r["ci_low"] > 0]
    hurts = [r for r in voting if r["ci_high"] < 0]
    print(f"\n  folds retenus : {len(voting)}/{len(rows)} "
          f"({len(rows) - len(voting)} ecartes pour predicteur constant)")
    if not voting:
        print("  VERDICT : INDECIDABLE — aucun fold ne produit deux modeles "
              "non degeneres.")
        return "indecidable"
    print(f"  IC95 strictement positifs : {len(helps)}/{len(voting)}   "
          f"strictement negatifs : {len(hurts)}/{len(voting)}")
    if len(helps) == len(voting):
        print("  VERDICT : les voisins aident sur TOUS les folds retenus.")
        return "aident"
    if len(hurts) == len(voting):
        print("  VERDICT : les voisins nuisent sur TOUS les folds retenus.")
        return "nuisent"
    print("  VERDICT : INDECIDABLE — les folds ne s'accordent pas. Aucune "
          "conclusion sur le transfert ne peut etre tiree de ces donnees.")
    return "indecidable"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", nargs="+", type=int, default=[4, 16])
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--phys-seed", nargs="+", type=int,
                   default=list(PHYSICS_SEEDS))
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=B_DEFAULT)
    p.add_argument("--label-suffix", default="",
                   help="variante de label, ex. _globalthr (T28)")
    args = p.parse_args()

    print("=" * 78)
    print("  T29 — ecarts LOSO avec intervalle de confiance")
    print("=" * 78)
    print(f"  args: {' '.join(sys.argv[1:]) or '(defauts)'}")

    t0 = time.time()
    all_rows, verdicts = [], {}
    for dim in args.dim:
        rows = run_dim(dim, args.N, args.re, args.scenario, args.phys_seed,
                       args.max_snaps, args.seed, args.bootstrap,
                       args.label_suffix)
        verdicts[dim] = verdict(rows)
        all_rows.extend(rows)

    assert all_rows, "aucun fold evalue"

    out = os.path.join(
        RESULTS_DIR,
        f"t29_loso_delta_ci_N{args.N}{args.label_suffix or '_perscenario'}.npz")
    np.savez_compressed(
        out,
        held=np.array([r["held"] for r in all_rows]),
        dim=np.array([r["dim"] for r in all_rows]),
        n_val=np.array([r["n_val"] for r in all_rows]),
        prevalence=np.array([r["prevalence"] for r in all_rows]),
        f1_class=np.array([r["f1_class"] for r in all_rows]),
        f1_site=np.array([r["f1_site"] for r in all_rows]),
        f1_sten=np.array([r["f1_sten"] for r in all_rows]),
        delta=np.array([r["delta"] for r in all_rows]),
        ci_low=np.array([r["ci_low"] for r in all_rows]),
        ci_high=np.array([r["ci_high"] for r in all_rows]),
        constant=np.array([r["constant"] for r in all_rows]),
        constant_compared=np.array([r["constant_compared"] for r in all_rows]),
        verdict_by_dim=np.array([f"{d}:{v}" for d, v in verdicts.items()]),
        label_suffix=args.label_suffix,
        physics_seeds=np.array(args.phys_seed),
        git_hash=git_commit_hash(),
        cli_args=" ".join(sys.argv[1:]),
    )
    print(f"\n  saved: {os.path.basename(out)}   ({time.time() - t0:.1f}s)")
    print("\nT29 complete.")


if __name__ == "__main__":
    main()
