#!/usr/bin/env python3
"""Recentrage du biais : synthese du plan d'experience 2 x 2 x 2.

QUESTION (pre-enregistree dans `rho_gap_f1.py`). Mieux resoudre le
Hamiltonien ameliore-t-il la decision, rho(E_gap, F1) < 0, une fois corriges
les deux faits identifies par `h0_readout_threshold_diagnostic.py` : un biais
centre sur un seuil (0,15) tres inferieur au seuil F1-optimal (0,51-0,60), et
une lecture du QAOA (vote majoritaire) qui n'est pas celle du solveur deploye ?

PLAN. Configuration du panel H0 V1 du 28 aout (N=96, dim=3, Re=400, quatre
scenarios canoniques, trois instantanes chacun, p = 1, 2, 3, K_opt = 60,
graine 0), pour chaque mappeur (V1, V2), chaque seuil du biais (deploye,
`loso-f1`) et chaque lecture (majoritaire, deployee) : huit artefacts
`h0_optimiser_equivalence_..._recentrage.npz`, produits par le meme commit.

SORTIE. Pour chaque artefact : rho(E_gap, F1) (`rho_gap_f1`), et par solveur
le taux d'optimum atteint, l'ecart d'energie moyen et le F1 moyen ; plus le
F1 de l'optimum exact moins celui de la decision classique du MEME seuil,
compte par trajectoire. Pour chaque couple (mappeur, seuil), l'optimum exact
est RECALCULE sans QAOA par le meme chemin que `solver_panel`, pour dire ce
qu'il est : uniforme (tout raffiner, rien raffiner) ou non, egal ou non a la
decision classique ; son F1 doit reproduire celui de l'artefact, instance
par instance, sinon la synthese leve. psi n'entre pas dans l'optimum exact
(il ne touche que les angles initiaux). Ecrit
results/h0_recentring_summary.json (+ provenance). Un artefact manquant leve
une erreur.

Usage :
  python study/h0_selection/h0_recentring_summary.py
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
from rho_gap_f1 import rho_gap_f1

STEM = ("h0_optimiser_equivalence_N96_dim3_harris_tearing-kelvin_helmholtz-"
        "mhd_rotor-orszag_tang")
TAG = "recentrage"
MAPPERS = ("v1", "v2")
BIAS = ("deployed", "loso-f1")
READOUT = ("majority", "deployed")
OUT_NAME = "h0_recentring_summary.json"


def artifact_name(mapper, bias, readout, tag=TAG):
    """Nom produit par `h0_optimiser_equivalence._output_path`."""
    return (STEM
            + ("" if bias == "deployed" else "_thrlosof1")
            + ("" if readout == "majority" else "_readdeployed")
            + ("" if mapper == "v2" else f"_{mapper}")
            + f"_{tag}.npz")


def summarise(path):
    """Taux d'optimum, ecart d'energie et F1 moyens par solveur, et rho."""
    d = np.load(path, allow_pickle=True)
    sol = d["solver"]
    per_solver = {}
    for s in sorted(set(sol.tolist())):
        m = sol == s
        per_solver[s] = {
            "hit": float(np.nanmean(d["hit"][m].astype(float))),
            "E_gap": float(np.nanmean(d["E_gap"][m].astype(float))),
            "f1": float(np.nanmean(d["f1"][m].astype(float))),
            "match_exact": float(np.nanmean(d["match"][m].astype(float))),
        }
    r = rho_gap_f1(path)
    exh = per_solver["exhaustive"]["f1"]
    cls = per_solver["classical_init"]["f1"]
    # L'unite d'inference est la trajectoire (CLAUDE.md) : la comparaison
    # exact / classique se compte par scenario, sur la moyenne de ses
    # instantanes, jamais instantane par instantane.
    by_traj = {}
    for sc in sorted(set(d["scenario"].tolist())):
        ms = d["scenario"] == sc
        by_traj[sc] = {s: float(np.nanmean(d["f1"][ms & (sol == s)].astype(float)))
                       for s in ("exhaustive", "classical_init", "qaoa_p1",
                                 "qaoa_p2", "qaoa_p3")}
    diffs = [v["exhaustive"] - v["classical_init"] for v in by_traj.values()]
    return {
        "artifact": os.path.basename(path),
        "git_hash": str(d["git_hash"]) if "git_hash" in d else None,
        "dirty_at_start": (bool(d["dirty_at_start"])
                           if "dirty_at_start" in d else None),
        "bias_threshold_mode": str(d["bias_threshold_mode"]),
        "readout": str(d["readout"]),
        "thresholds": {str(k): float(v) for k, v in
                       zip(d["bias_threshold_scenarios"],
                           d["bias_threshold_values"])},
        "n_instances": int(len(set(zip(d["scenario"].tolist(),
                                       d["snap"].tolist())))),
        "rho": r.get("rho"), "p": r.get("p"),
        "rho_error": r.get("erreur"),
        "n_solvers": r.get("n_solveurs"),
        "f1_exact_minus_classical": exh - cls,
        "trajectories_exact_vs_classical": {
            "better": int(sum(x > 1e-12 for x in diffs)),
            "equal": int(sum(abs(x) <= 1e-12 for x in diffs)),
            "worse": int(sum(x < -1e-12 for x in diffs)),
        },
        "by_trajectory": by_traj,
        "per_solver": per_solver,
    }


def ground_state_structure(results_dir, mapper, thresholds, re=400, N=96,
                           dim=3, n_snaps=3):
    """Optimum exact et decision classique de chaque instance du panel,
    recalcules par le chemin de `solver_panel` (sans QAOA, sans psi)."""
    from bias_threshold import _input_paths
    from h0_optimiser_equivalence import f1_from_masks, h0_snapshot_indices
    from ising_terms_and_annealing import (build_ising_terms,
                                           exhaustive_ground_state,
                                           spins_to_decisions)
    from qaoa_inputs import prepare_qaoa_inputs

    rows = []
    for sc in sorted(thresholds):
        thr = float(thresholds[sc])
        dp, pp = _input_paths(sc, re, N, dim, results_dir)
        dns, pat = np.load(dp), np.load(pp)
        for si in h0_snapshot_indices(len(dns["vx"]), n_snaps):
            fields = [dns[k][si].astype(np.float64)
                      for k in ("vx", "vy", "Bx", "By")]
            _data, hp, score = prepare_qaoa_inputs(
                *fields, N, dim, re, use_v2=(mapper == "v2"),
                threshold_amr=thr)
            h, e, pl = build_ising_terms(hp, dim)
            spins, _E, n_opt = exhaustive_ground_state(h, e, pl,
                                                       2 * dim * dim)
            dh, dv = spins_to_decisions(spins, dim)
            exact = (dh | dv).ravel()
            classical = np.asarray(score, dtype=float).ravel() > thr
            gt = np.asarray(pat["l2_errors"][si]
                            >= float(pat["l2_threshold"])).ravel()
            rows.append({
                "scenario": sc, "snap": int(si), "n_optima": int(n_opt),
                "refine_all": bool(exact.all()),
                "refine_none": bool(not exact.any()),
                "equals_classical": bool(np.array_equal(exact, classical)),
                "n_cells_exact_ne_classical": int((exact != classical).sum()),
                "n_refined_exact": int(exact.sum()),
                "n_refined_classical": int(classical.sum()),
                "f1_exact": f1_from_masks(exact, gt),
            })
    return rows


def _check_against_artifact(rows, path):
    """Le F1 recalcule de l'optimum doit etre celui de l'artefact."""
    d = np.load(path, allow_pickle=True)
    m = d["solver"] == "exhaustive"
    stored = {(str(sc), int(sn)): float(f) for sc, sn, f in
              zip(d["scenario"][m], d["snap"][m], d["f1"][m])}
    bad = [(r["scenario"], r["snap"]) for r in rows
           if abs(stored[(r["scenario"], r["snap"])] - r["f1_exact"]) > 1e-9]
    if bad or len(stored) != len(rows):
        raise ValueError(
            f"{os.path.basename(path)} : l'optimum recalcule ne reproduit "
            f"pas l'artefact sur {bad} (ou les instances different) ; la "
            "description de l'optimum ne porterait pas sur le meme objet.")


def _structure_counts(rows):
    return {
        "n_instances": len(rows),
        "n_unique_optimum": sum(r["n_optima"] == 1 for r in rows),
        "n_refine_all": sum(r["refine_all"] for r in rows),
        "n_refine_none": sum(r["refine_none"] for r in rows),
        "n_equals_classical": sum(r["equals_classical"] for r in rows),
        "mean_cells_refined_exact": float(np.mean(
            [r["n_refined_exact"] for r in rows])),
        "mean_cells_refined_classical": float(np.mean(
            [r["n_refined_classical"] for r in rows])),
    }


def collect(results_dir, structure=True):
    out = {}
    for mapper in MAPPERS:
        for bias in BIAS:
            for readout in READOUT:
                name = artifact_name(mapper, bias, readout)
                path = os.path.join(results_dir, name)
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"artefact du plan absent : {name}. Une synthese sur "
                        "un plan incomplet ne repondrait plus a la question "
                        "posee.")
                out[f"{mapper}|{bias}|{readout}"] = summarise(path)
    commits = {v["git_hash"] for v in out.values()}
    if len(commits) != 1:
        raise ValueError(f"le plan melange plusieurs commits : {commits}")
    if any(v["dirty_at_start"] for v in out.values()):
        raise ValueError("un artefact du plan vient d'un arbre modifie")
    if structure:
        for mapper in MAPPERS:
            for bias in BIAS:
                thr = out[f"{mapper}|{bias}|majority"]["thresholds"]
                rows = ground_state_structure(results_dir, mapper, thr)
                for readout in READOUT:
                    _check_against_artifact(rows, os.path.join(
                        results_dir, artifact_name(mapper, bias, readout)))
                    cell = out[f"{mapper}|{bias}|{readout}"]
                    cell["ground_state"] = _structure_counts(rows)
                    cell["ground_state_rows"] = rows
    return out


def main():
    from config import RESULTS_DIR
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, OUT_NAME))
    args = p.parse_args()
    started = provenance.start()
    cells = collect(RESULTS_DIR)

    print(f"  {'mappeur':<7} {'seuil':<9} {'lecture':<9} {'rho':>7} "
          f"{'p':>6} {'F1 exact':>9} {'F1 class.':>9} {'F1 QAOA p1-3':>15} "
          f"{'hit QAOA':>9}  optimum : tout/rien/=classique")
    for key, c in cells.items():
        mapper, bias, readout = key.split("|")
        ps = c["per_solver"]
        q = [ps[s]["f1"] for s in ("qaoa_p1", "qaoa_p2", "qaoa_p3")]
        h = [ps[s]["hit"] for s in ("qaoa_p1", "qaoa_p2", "qaoa_p3")]
        rho = "indef." if c["rho"] is None else f"{c['rho']:+.3f}"
        pv = "" if c["p"] is None else f"{c['p']:.3f}"
        print(f"  {mapper:<7} {bias:<9} {readout:<9} {rho:>7} {pv:>6} "
              f"{ps['exhaustive']['f1']:>9.3f} "
              f"{ps['classical_init']['f1']:>9.3f} "
              f"{min(q):>7.3f}-{max(q):.3f} "
              f"{min(h):>4.2f}-{max(h):.2f}  "
              + ("" if "ground_state" not in c else
                 f"{c['ground_state']['n_refine_all']}/"
                 f"{c['ground_state']['n_refine_none']}/"
                 f"{c['ground_state']['n_equals_classical']} "
                 f"sur {c['ground_state']['n_instances']}"))
    res = {"cells": cells, **provenance.finish(started),
           "argv": sys.argv, "cli_args": vars(args)}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"  ecrit : {args.out}")


if __name__ == "__main__":
    main()
