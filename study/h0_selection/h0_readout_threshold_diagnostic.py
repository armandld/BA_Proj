#!/usr/bin/env python3
"""Lecture du QAOA et seuil du biais : ce que mesure le F1 du QAOA de H0.

QUESTION. Le panel H0 (`h0_optimiser_equivalence.py`) trouve que mieux
resoudre le Hamiltonien degrade la decision : rho(E_gap, F1) > 0, l'optimum
exact a le F1 le plus bas, le QAOA un des plus hauts. Pourquoi ?

DEUX FAITS DE CODE.
  * Le biais est centre sur le seuil AMR deploye : h_i ~ (s_i - seuil), avec
    0,15 pour V2 et 0,1496 pour V1 (`study/pipeline/config.py`).
  * L'etude lit le QAOA par vote majoritaire, P(1) > 0,5 par qubit
    (`qaoa_inputs.majority_readout`) ; le solveur deploye compare la moyenne
    des deux marginales d'une cellule au seuil AMR
    (`qaoa_inputs.deployed_readout`, `Simulation.refinement._run_level`).
    Chaque qubit part de P(1) = s : la lecture deployee du circuit NON
    optimise est la decision classique, la lecture majoritaire est s > 0,5.

CE QUE LE SCRIPT RECALCULE, sur les instantanes du panel H0 (quatre
scenarios canoniques, Re = 400, N = 96, dim = 3, trois instantanes chacun,
selection `h0_snapshot_indices`) :
  1. les masques « decision classique » (s > seuil deploye), « lecture
     majoritaire du circuit non optimise » (s > 0,5) et « lecture deployee
     du circuit non optimise », et le nombre de cellules ou ils different ;
  2. le F1 moyen de ces masques et du masque « tout raffiner » ;
  3. dans les artefacts H0 publies (V1 du 28 aout, V2 du 16 aout), les
     instances ou le F1 du QAOA egale celui de la lecture majoritaire non
     optimisee, et celles ou le F1 de l'optimum exact egale celui de « tout
     raffiner » (les masques eux-memes ne sont pas stockes : c'est une
     egalite de F1, rapportee comme telle) ;
  4. les seuils F1-optimaux du bras classique de la replication
     confirmatoire, lus dans son artefact, et leur recalcul independant par
     `bias_threshold.loso_f1_thresholds` (sauf `--skip-calibration`).

Aucun QAOA n'est execute : tout se lit sur les scores, les labels et les
artefacts deja produits. Un fichier manquant leve une erreur.

Sortie : results/h0_readout_threshold_diagnostic.json (+ provenance, argv).

Usage :
  python study/h0_selection/h0_readout_threshold_diagnostic.py
"""
import argparse
import json
import os
import sys

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

import provenance                                           # hash au DEMARRAGE
from bias_threshold import (CANONICAL_SCENARIOS, block_scores_and_labels,
                            loso_f1_thresholds)
from h0_optimiser_equivalence import f1_from_masks, h0_snapshot_indices
from qaoa_inputs import deployed_readout, majority_readout

#: Artefacts H0 publies a dim = 3 sur les douze instantanes du panel.
H0_PANELS = {
    "v1_28aug": ("h0_optimiser_equivalence_N96_dim3_harris_tearing-"
                 "kelvin_helmholtz-mhd_rotor-orszag_tang_v1.npz"),
    "v2_16aug": "h0_optimiser_equivalence_N96_dim3_hamiltonien_corrige.npz",
}
#: Artefact de la replication confirmatoire (seuils classiques par pli).
CONFIRMATORY = "h2b_v2_hamiltonian_vs_gbt_loso_N96_dim3_n10_multire.npz"
QAOA_SOLVERS = ("qaoa_p1", "qaoa_p2", "qaoa_p3", "qaoa_shots_p3")
OUT_NAME = "h0_readout_threshold_diagnostic.json"


def h0_instances(results_dir, N=96, dim=3, re=400, n_snaps=3,
                 scenarios=CANONICAL_SCENARIOS):
    """{(scenario, instantane): (scores, labels)} des instantanes du panel."""
    out = {}
    for sc in scenarios:
        scores, labels, idx = block_scores_and_labels(
            sc, re, N, dim, results_dir,
            lambda n: h0_snapshot_indices(n, n_snaps))
        for k, si in enumerate(idx):
            out[(sc, int(si))] = (scores[k], labels[k])
    return out


def unoptimised_masks(scores, dim, threshold):
    """Masques de cellule (dim*dim,) du circuit NON optimise, P(1) = s sur
    les deux aretes, et de la decision classique du panel."""
    s = np.asarray(scores, dtype=float).ravel()
    marginals = np.concatenate([s, s])
    mh, mv = majority_readout(marginals, dim)
    dh, dv = deployed_readout(marginals, dim, threshold)
    return {
        "classical": s > threshold,
        "majority_unoptimised": (mh | mv).ravel(),
        "deployed_unoptimised": (dh | dv).ravel(),
        "refine_all": np.ones_like(s, dtype=bool),
    }


def artifact_f1(path, solver):
    """{(scenario, instantane): F1} d'un solveur dans un artefact H0."""
    d = np.load(path, allow_pickle=True)
    m = d["solver"] == solver
    if not m.any():
        raise KeyError(f"solveur {solver} absent de {os.path.basename(path)}")
    return {(str(sc), int(sn)): float(f)
            for sc, sn, f in zip(d["scenario"][m], d["snap"][m], d["f1"][m])}


def _restrict(f1_by_key, scenarios):
    return {k: v for k, v in f1_by_key.items() if k[0] in scenarios}


def diagnose(results_dir, threshold, dim=3, calibration=True,
             scenarios=CANONICAL_SCENARIOS):
    """Tous les nombres du constat, recalcules. Voir le docstring du module.

    `scenarios` restreint le panel ; les artefacts publies sont lus sur les
    memes scenarios."""
    scenarios = tuple(scenarios)
    inst = h0_instances(results_dir, dim=dim, scenarios=scenarios)
    keys = sorted(inst)
    rows = []
    n_cls = n_maj = n_diff = 0
    diff_by_scenario = {}
    for key in keys:
        s, gt = inst[key]
        masks = unoptimised_masks(s, dim, threshold)
        n_cls += int(masks["classical"].sum())
        n_maj += int(masks["majority_unoptimised"].sum())
        d = int((masks["classical"] != masks["majority_unoptimised"]).sum())
        n_diff += d
        diff_by_scenario[key[0]] = diff_by_scenario.get(key[0], 0) + d
        rows.append({
            "scenario": key[0], "snap": key[1], "n_positive": int(gt.sum()),
            "deployed_equals_classical": bool(np.array_equal(
                masks["deployed_unoptimised"], masks["classical"])),
            **{f"f1_{name}": f1_from_masks(mask, gt)
               for name, mask in masks.items()},
        })
    n_cells = len(keys) * dim * dim

    def mean_f1(name):
        return float(np.mean([r[f"f1_{name}"] for r in rows]))

    by_key = {(r["scenario"], r["snap"]): r for r in rows}
    informative = sorted(
        k for k, r in by_key.items()
        if abs(r["f1_majority_unoptimised"] - r["f1_classical"]) > 1e-12
        and abs(r["f1_majority_unoptimised"] - r["f1_refine_all"]) > 1e-12)

    panels = {}
    for tag, name in H0_PANELS.items():
        path = os.path.join(results_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"artefact H0 absent : {name}")
        cls = _restrict(artifact_f1(path, "classical_init"), scenarios)
        exh = _restrict(artifact_f1(path, "exhaustive"), scenarios)
        if set(cls) != set(keys):
            raise ValueError(f"{name} ne porte pas les instantanes du panel")
        qaoa = {}
        for solver in QAOA_SOLVERS:
            f = _restrict(artifact_f1(path, solver), scenarios)
            qaoa[solver] = {
                "equal_majority_unoptimised": sum(
                    abs(f[k] - by_key[k]["f1_majority_unoptimised"]) < 1e-9
                    for k in keys),
                "equal_on_informative": sum(
                    abs(f[k] - by_key[k]["f1_majority_unoptimised"]) < 1e-9
                    for k in informative),
                "mean_f1": float(np.mean(list(f.values()))),
            }
        panels[tag] = {
            "artifact": name,
            "classical_equal_recomputed": sum(
                abs(cls[k] - by_key[k]["f1_classical"]) < 1e-9 for k in keys),
            "exhaustive_equal_refine_all": sum(
                abs(exh[k] - by_key[k]["f1_refine_all"]) < 1e-9
                for k in keys),
            "exhaustive_mean_f1": float(np.mean(list(exh.values()))),
            "qaoa": qaoa,
        }

    conf_path = os.path.join(results_dir, CONFIRMATORY)
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"artefact confirmatoire absent : {CONFIRMATORY}")
    conf = np.load(conf_path, allow_pickle=True)
    conf_thr = {str(h): float(t)
                for h, t in zip(conf["held"], conf["thr_classical"])}

    out = {
        "n_instances": len(keys),
        "n_cells": n_cells,
        "threshold": float(threshold),
        "n_refine_classical": n_cls,
        "n_refine_majority_unoptimised": n_maj,
        "n_cells_classical_ne_majority": n_diff,
        "cells_classical_ne_majority_by_scenario": diff_by_scenario,
        "mean_f1": {name: mean_f1(name) for name in
                    ("classical", "majority_unoptimised",
                     "deployed_unoptimised", "refine_all")},
        "informative_instances": [list(k) for k in informative],
        "panels": panels,
        "confirmatory_thresholds": conf_thr,
        "rows": rows,
    }
    if calibration:
        cal = loso_f1_thresholds(results_dir, N=96, dim=dim)
        out["recomputed_thresholds"] = {
            sc: {"threshold": t, "train_f1": f} for sc, (t, f) in cal.items()}
        out["recomputed_equal_confirmatory"] = all(
            abs(cal[sc][0] - conf_thr[sc]) < 1e-12 for sc in conf_thr)
    return out


def main():
    from config import RESULTS_DIR, V2_THRESHOLD
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scenario", nargs="+",
                   default=list(CANONICAL_SCENARIOS),
                   help="scenarios du panel (defaut : les quatre canoniques)")
    p.add_argument("--skip-calibration", action="store_true",
                   help="ne pas recalculer les seuils LOSO (160 instantanes)")
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, OUT_NAME))
    args = p.parse_args()

    started = provenance.start()
    res = diagnose(RESULTS_DIR, V2_THRESHOLD,
                   calibration=not args.skip_calibration,
                   scenarios=args.scenario)
    res.update(provenance.finish(started))
    res["argv"] = sys.argv
    res["cli_args"] = vars(args)

    print(f"  {res['n_instances']} instantanes, {res['n_cells']} cellules, "
          f"seuil deploye {res['threshold']}")
    print(f"  raffinees : classique {res['n_refine_classical']}, lecture "
          f"majoritaire non optimisee {res['n_refine_majority_unoptimised']}"
          f" ; differentes : {res['n_cells_classical_ne_majority']} "
          f"{res['cells_classical_ne_majority_by_scenario']}")
    print("  F1 moyen : " + ", ".join(
        f"{k} {v:.3f}" for k, v in res["mean_f1"].items()))
    for tag, pan in res["panels"].items():
        print(f"  {tag} : exact = tout raffiner sur "
              f"{pan['exhaustive_equal_refine_all']}/{res['n_instances']} ; "
              + " ; ".join(
                  f"{s} = lecture majoritaire non optimisee sur "
                  f"{q['equal_majority_unoptimised']}/{res['n_instances']} "
                  f"({q['equal_on_informative']}/"
                  f"{len(res['informative_instances'])} informatives)"
                  for s, q in pan["qaoa"].items()))
    print(f"  seuils classiques confirmatoires : {res['confirmatory_thresholds']}")
    if "recomputed_thresholds" in res:
        print(f"  recalcul LOSO identique : {res['recomputed_equal_confirmatory']}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"  ecrit : {args.out}")


if __name__ == "__main__":
    main()
