#!/usr/bin/env python3
"""Le modele jouet (`study/common/toy_model.py`) doit se comporter comme les
scenarios DNS reels sur les proprietes deja etablies (H0a, H0b), pas juste
"marcher sans planter". Ce script mesure, sur N instantanes jouets
independants, exactement les memes quantites que T26/D-53 sur des scenarios
reels : l'optimum exact du hamiltonien (enumeration, pas d'approximation),
la decision QAOA reellement executee, et un seuil classique ajuste par F1 --
toutes les trois contre la meme verite terrain generique (`patch_l2_errors`,
deja utilisee sur DNS reelles, aucune version "jouet" separee).

Pourquoi verifier contre des DNS reelles plutot que de faire confiance au
generateur seul : une premiere version de ce script trouvait l'optimum exact
toujours a "tout raffiner" (20/20 tirages, F1=0.500 sans variance) et deux
tentatives de correction du generateur (echelle, bruit) n'ont RIEN change.
Rejoue sur cinq instantanes DNS reels (harris_tearing, Re=400, N=96, meme
dim), le meme optimum uniforme est apparu 5 fois sur 5 -- ce n'etait donc ni
un defaut du generateur ni un bug de ce script. `T26` (docs/RESULTS.md)
mesure deja "uniformite du fondamental = 0.50" a dim=3 sur des scenarios
reels aux hyperparametres de reference : une proprete connue du hamiltonien,
coherente avec H0b (mieux resoudre H degrade la decision), pas une anomalie.
Ce script la retrouve sur donnees jouets -- signe que le modele jouet est
fidele sur ce point precis, pas un defaut a corriger.

Usage:
  python study/h3_representation/h3_toy_model_check.py --n-seeds 20
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from toy_model import generate_toy_snapshot
from qaoa_inputs import prepare_qaoa_inputs, run_qaoa_on_snapshot
from hard_patch_labels import patch_l2_errors
from exact_diagonalisation import build_patch_hamiltonian
from ising_terms_and_annealing import (
    build_ising_terms, exhaustive_ground_state, spins_to_decisions, _metrics)
from h2b_ceiling_random_split import best_threshold_f1
from config import L2_PERCENTILE_HARD, V2_THRESHOLD
import provenance

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")


def _stat(name, arr):
    arr = np.asarray(arr)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}


def run(n_seeds, N, n_patches, re, k_opt, shots, threshold_amr, base_seed):
    started = provenance.start()
    t0 = time.time()
    rows = []
    for i in range(n_seeds):
        seed = base_seed + i
        vx, vy, Bx, By = generate_toy_snapshot(N=N, seed=seed)

        l2 = patch_l2_errors(vx, vy, Bx, By, n_patches)
        gt = (l2 >= np.percentile(l2, L2_PERCENTILE_HARD)).ravel()

        hp, score_vqa, _ = build_patch_hamiltonian(
            vx, vy, Bx, By, N, n_patches, re,
            threshold_amr=threshold_amr, use_v2=True)
        h_bias, edges, plaqs = build_ising_terms(hp, n_patches)
        n_q = 2 * n_patches * n_patches
        exact_spins, exact_E, degeneracy = exhaustive_ground_state(
            h_bias, edges, plaqs, n_q)
        dec_h, dec_v = spins_to_decisions(exact_spins, n_patches)
        exact_refine = (dec_h | dec_v).ravel()

        data_in, hamilt_params, score_vqa2 = prepare_qaoa_inputs(
            vx, vy, Bx, By, N=N, n_patches=n_patches, Re=re)
        _marg, qh, qv, _opt_p, wall = run_qaoa_on_snapshot(
            data_in, hamilt_params, dim=n_patches, reps=2,
            K_opt=k_opt, shots=shots, seed=seed)
        qaoa_refine = (qh | qv).ravel()

        rows.append(dict(seed=seed, score=score_vqa.ravel(), gt=gt,
                         exact_refine=exact_refine, qaoa_refine=qaoa_refine,
                         exact_E=exact_E, degeneracy=degeneracy))

    all_scores = np.concatenate([r["score"] for r in rows])
    all_gt = np.concatenate([r["gt"] for r in rows])
    classical_thr, f1_pool = best_threshold_f1(all_scores, all_gt)

    f1_exact, f1_qaoa, f1_classical = [], [], []
    agree_qaoa_exact, agree_qaoa_classical, agree_classical_exact = [], [], []
    exact_frac, qaoa_frac = [], []
    for r in rows:
        classical_refine = r["score"] > classical_thr
        f1_exact.append(_metrics(r["exact_refine"], r["gt"])["f1"])
        f1_qaoa.append(_metrics(r["qaoa_refine"], r["gt"])["f1"])
        f1_classical.append(_metrics(classical_refine, r["gt"])["f1"])
        agree_qaoa_exact.append(float(np.mean(
            r["qaoa_refine"] == r["exact_refine"])))
        agree_qaoa_classical.append(float(np.mean(
            r["qaoa_refine"] == classical_refine)))
        agree_classical_exact.append(float(np.mean(
            classical_refine == r["exact_refine"])))
        exact_frac.append(float(r["exact_refine"].mean()))
        qaoa_frac.append(float(r["qaoa_refine"].mean()))

    summary = {
        "f1_exact_vs_gt": _stat("f1_exact_vs_gt", f1_exact),
        "f1_qaoa_vs_gt": _stat("f1_qaoa_vs_gt", f1_qaoa),
        "f1_classical_vs_gt": _stat("f1_classical_vs_gt", f1_classical),
        "agree_qaoa_exact": _stat("agree_qaoa_exact", agree_qaoa_exact),
        "agree_qaoa_classical": _stat("agree_qaoa_classical",
                                      agree_qaoa_classical),
        "agree_classical_exact": _stat("agree_classical_exact",
                                       agree_classical_exact),
        "exact_frac": _stat("exact_frac", exact_frac),
        "qaoa_frac": _stat("qaoa_frac", qaoa_frac),
    }

    print(f"n_seeds={n_seeds}  N={N}  n_patches={n_patches}  "
          f"classical_thr={classical_thr:.4f} (F1 pool={f1_pool:.3f})  "
          f"wall={time.time()-t0:.0f}s")
    for name, s in summary.items():
        print(f"  {name:24s} {s['mean']:.3f} +/- {s['std']:.3f}  "
              f"(min {s['min']:.3f}, max {s['max']:.3f})")

    payload = dict(
        n_seeds=n_seeds, N=N, n_patches=n_patches, re=re, k_opt=k_opt,
        shots=shots, threshold_amr=threshold_amr, base_seed=base_seed,
        classical_thr=classical_thr, f1_pool=f1_pool,
        seeds=np.array([r["seed"] for r in rows]),
        exact_E=np.array([r["exact_E"] for r in rows]),
        degeneracy=np.array([r["degeneracy"] for r in rows]),
        f1_exact=np.array(f1_exact), f1_qaoa=np.array(f1_qaoa),
        f1_classical=np.array(f1_classical),
        agree_qaoa_exact=np.array(agree_qaoa_exact),
        agree_qaoa_classical=np.array(agree_qaoa_classical),
        agree_classical_exact=np.array(agree_classical_exact),
        exact_frac=np.array(exact_frac), qaoa_frac=np.array(qaoa_frac),
        **provenance.finish(started),
    )
    return payload, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--N", type=int, default=48)
    p.add_argument("--dim", dest="n_patches", type=int, default=3,
                   help="patches par cote (nom `--dim` : convention "
                        "partagee avec le reste de study/h2b_prediction)")
    p.add_argument("--re", type=float, default=800.0)
    p.add_argument("--k-opt", type=int, default=40)
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--threshold-amr", type=float, default=V2_THRESHOLD)
    p.add_argument("--seed", type=int, default=0,
                   help="graine du premier instantane jouet (base_seed)")
    args = p.parse_args()

    payload, _ = run(args.n_seeds, args.N, args.n_patches, args.re,
                     args.k_opt, args.shots, args.threshold_amr, args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR,
        f"h3_toy_model_check_N{args.N}_dim{args.n_patches}"
        f"_n{args.n_seeds}.npz")
    np.savez(out_path, **payload)
    print(f"\nEcrit : {out_path}")


if __name__ == "__main__":
    main()
