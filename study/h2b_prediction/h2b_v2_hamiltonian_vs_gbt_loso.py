#!/usr/bin/env python3
"""V2 (hamiltonien SANS PARAMETRE) contre GBT, LOSO, sur DNS reelle --
sans aucun entrainement de V2 : ses poids sont figes, D-22 (la campagne
Optuna, jamais lancee -- trop couteuse) ne le concerne pas.

dim=3 (18 qubits) : la seule taille certifiee non degeneree pour QAOA
dans ce depot (docs/EVALUATION.md -- dim=2 est degenere, D-45/D-47).
N=96/Re=400 : les memes DNS et labels deja utilises par
h0_optimiser_equivalence.py et h2b_ceiling_random_split.py, aucun nouveau
calcul DNS.

Psi cable (with_psi=True, instantane precedent de la meme trajectoire
DNS) -- comme h3_toy_instability_check.py, pas comme h3_toy_model_check.py
(champ statique, psi=0 exact) : une DNS reelle a un instant precedent.

Le GBT et le seuil classique SONT ajustes -- sur les scenarios
d'ENTRAINEMENT du pli LOSO, jamais sur le scenario tenu (meme protocole
que h2b_loso_transfer.py). QAOA n'a rien a ajuster : evalue une seule
fois par instantane, la meme decision sert a chaque pli.

Usage:
  python study/h2b_prediction/h2b_v2_hamiltonian_vs_gbt_loso.py --n-snaps 5
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qaoa_inputs import prepare_qaoa_inputs, run_qaoa_on_snapshot
from h2b_ceiling_random_split import (
    extract_features_2d, N_FEATS, best_threshold_f1, make_model, fit_eval)
from config import RESULTS_DIR
import provenance

SCENARIOS_4 = ("harris_tearing", "kelvin_helmholtz", "mhd_rotor", "orszag_tang")


def gather_scenario(sc, re, N, dim, n_snaps, k_opt, shots, seed):
    """Une ligne par instantane : score classique, verite terrain, features
    GBT, decision QAOA -- toutes calculees UNE fois, reutilisees dans
    chaque pli LOSO ou ce scenario sert de train ou de tenu."""
    dns = np.load(os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{N}.npz"))
    patches = np.load(
        os.path.join(RESULTS_DIR, f"patches_{sc}_Re{re}_N{N}_dim{dim}.npz"))
    vx_all = dns["vx"].astype(np.float64)
    vy_all = dns["vy"].astype(np.float64)
    Bx_all = dns["Bx"].astype(np.float64)
    By_all = dns["By"].astype(np.float64)
    l2_all = patches["l2_errors"]
    l2_thr = float(patches["l2_threshold"])

    n_total = len(vx_all)
    step = max(1, (n_total - 1) // n_snaps)
    idx = list(range(1, n_total, step))[:n_snaps]  # >=1 : instant precedent requis

    rows = []
    for si in idx:
        vx, vy, Bx, By = vx_all[si], vy_all[si], Bx_all[si], By_all[si]
        feats_2d, score = extract_features_2d(vx, vy, Bx, By, N, dim, re)
        gt = (l2_all[si] >= l2_thr).ravel().astype(int)

        prev_fields = dict(vx=vx_all[si - 1], vy=vy_all[si - 1],
                           Bx=Bx_all[si - 1], By=By_all[si - 1])
        data_in, hamilt_params, _ = prepare_qaoa_inputs(
            vx, vy, Bx, By, N=N, n_patches=dim, Re=re, use_v2=True,
            with_psi=True, prev_fields=prev_fields)
        _marg, qh, qv, _opt_p, wall = run_qaoa_on_snapshot(
            data_in, hamilt_params, dim=dim, reps=2,
            K_opt=k_opt, shots=shots, seed=seed)
        qaoa_refine = (qh | qv).ravel().astype(int)

        rows.append(dict(
            scenario=sc, snap=int(si), score=score.ravel(), gt=gt,
            feats_site=feats_2d.reshape(-1, N_FEATS),
            qaoa_refine=qaoa_refine, wall=wall))
    return rows


def _cat(rows, key):
    return np.concatenate([r[key] for r in rows])


def run(re, N, dim, n_snaps, k_opt, shots, seed):
    started = provenance.start()
    t0 = time.time()
    by_sc = {}
    for sc in SCENARIOS_4:
        by_sc[sc] = gather_scenario(sc, re, N, dim, n_snaps, k_opt, shots, seed)
        wall = sum(r["wall"] for r in by_sc[sc])
        print(f"  {sc:<18} {len(by_sc[sc])} instantanes  "
              f"QAOA wall={wall:.0f}s  ({time.time()-t0:.0f}s cumule)",
              flush=True)
    print()

    header = (f"  {'tenu a l ecart':<18} {'F1 classique':>13} "
              f"{'F1 GBT':>8} {'F1 QAOA (V2)':>13}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows_out = []
    for held in SCENARIOS_4:
        train = [by_sc[s] for s in SCENARIOS_4 if s != held]
        train_rows = [r for scen in train for r in scen]
        held_rows = by_sc[held]

        Str, Ytr = _cat(train_rows, "score"), _cat(train_rows, "gt")
        Sva, Yva = _cat(held_rows, "score"), _cat(held_rows, "gt")
        thr_cls, _ = best_threshold_f1(Str, Ytr)
        f1_classical = f1_score(Yva, (Sva > thr_cls).astype(int),
                                zero_division=0)

        Xtr = np.concatenate([r["feats_site"] for r in train_rows], axis=0)
        Xva = np.concatenate([r["feats_site"] for r in held_rows], axis=0)
        res_gbt = fit_eval(make_model("gbt", seed, early_stopping=True),
                           Xtr, Ytr, Xva, Yva)

        Qva = _cat(held_rows, "qaoa_refine")
        f1_qaoa = f1_score(Yva, Qva, zero_division=0)

        rows_out.append(dict(held=held, f1_classical=f1_classical,
                             f1_gbt=res_gbt["f1"], f1_qaoa=f1_qaoa,
                             thr_classical=thr_cls))
        print(f"  {held:<18} {f1_classical:>13.3f} {res_gbt['f1']:>8.3f} "
              f"{f1_qaoa:>13.3f}")

    print()
    mean_cls = float(np.mean([r["f1_classical"] for r in rows_out]))
    mean_gbt = float(np.mean([r["f1_gbt"] for r in rows_out]))
    mean_qaoa = float(np.mean([r["f1_qaoa"] for r in rows_out]))
    n_qaoa_beats_cls = sum(r["f1_qaoa"] > r["f1_classical"] for r in rows_out)
    n_qaoa_beats_gbt = sum(r["f1_qaoa"] > r["f1_gbt"] for r in rows_out)
    print(f"  moyenne LOSO : classique={mean_cls:.3f}  gbt={mean_gbt:.3f}  "
          f"qaoa(V2)={mean_qaoa:.3f}")
    print(f"  QAOA bat classique sur {n_qaoa_beats_cls}/4 plis, "
          f"GBT sur {n_qaoa_beats_gbt}/4 plis")

    payload = dict(
        re=re, N=N, dim=dim, n_snaps=n_snaps, k_opt=k_opt, shots=shots,
        seed=seed, scenarios=np.array(SCENARIOS_4),
        held=np.array([r["held"] for r in rows_out]),
        f1_classical=np.array([r["f1_classical"] for r in rows_out]),
        f1_gbt=np.array([r["f1_gbt"] for r in rows_out]),
        f1_qaoa=np.array([r["f1_qaoa"] for r in rows_out]),
        thr_classical=np.array([r["thr_classical"] for r in rows_out]),
        **provenance.finish(started),
    )
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--re", type=int, default=400)
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--dim", type=int, default=3,
                   help="18 qubits a dim=3 : la seule taille QAOA "
                        "certifiee non degeneree ici (voir docstring)")
    p.add_argument("--n-snaps", type=int, default=5)
    p.add_argument("--k-opt", type=int, default=60)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    payload = run(args.re, args.N, args.dim, args.n_snaps, args.k_opt,
                 args.shots, args.seed)
    out_path = os.path.join(
        RESULTS_DIR,
        f"h2b_v2_hamiltonian_vs_gbt_loso_N{args.N}_dim{args.dim}"
        f"_n{args.n_snaps}.npz")
    np.savez(out_path, **payload)
    print(f"\nEcrit : {out_path}")


if __name__ == "__main__":
    main()
