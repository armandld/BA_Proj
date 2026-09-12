#!/usr/bin/env python3
"""Le seuil classique (`threshold_amr`) ajuste sur donnees jouets doit
tenir sur des instances jamais vues au moment de l'ajuster -- sinon le F1
mesure la memoire du train, pas un plafond reel (`CLAUDE.md` : « aucun
reglage ne voit le scenario tenu ou les labels d'evaluation »).

Genere un pool d'instantanes jouets (score classique seul, pas de QAOA --
rapide), le coupe en train/val par plusieurs partages disjoints, ajuste le
seuil sur le train, evalue sur le val. Un seuil instable d'un partage a
l'autre signalerait un ajustement qui a memorise le train.

Usage:
  python study/h2b_prediction/h2b_toy_ceiling.py --n-instances 300 --n-splits 10
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
from exact_diagonalisation import build_patch_hamiltonian
from hard_patch_labels import patch_l2_errors
from ising_terms_and_annealing import _metrics
from h2b_ceiling_random_split import best_threshold_f1
from config import L2_PERCENTILE_HARD, V2_THRESHOLD
import provenance

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")


def gather_pool(n_instances, N, n_patches, re, base_seed):
    scores, gts = [], []
    for i in range(n_instances):
        seed = base_seed + i
        vx, vy, Bx, By = generate_toy_snapshot(N=N, seed=seed)
        l2 = patch_l2_errors(vx, vy, Bx, By, n_patches)
        gt = l2 >= np.percentile(l2, L2_PERCENTILE_HARD)
        _hp, score_vqa, _ = build_patch_hamiltonian(
            vx, vy, Bx, By, N, n_patches, re,
            threshold_amr=V2_THRESHOLD, use_v2=True)
        scores.append(score_vqa.ravel())
        gts.append(gt.ravel())
    return np.array(scores), np.array(gts)


def evaluate_split(scores, gt, train_idx, val_idx):
    tr_scores, tr_gt = scores[train_idx].ravel(), gt[train_idx].ravel()
    va_scores, va_gt = scores[val_idx].ravel(), gt[val_idx].ravel()
    thr, f1_train = best_threshold_f1(tr_scores, tr_gt)
    f1_val = _metrics(va_scores > thr, va_gt)["f1"]
    return float(thr), float(f1_train), float(f1_val)


def run(n_instances, n_splits, train_frac, N, n_patches, re, base_seed,
       split_seed0):
    started = provenance.start()
    t0 = time.time()
    scores, gt = gather_pool(n_instances, N, n_patches, re, base_seed)
    n_train = int(round(n_instances * train_frac))

    thresholds, f1_trains, f1_vals = [], [], []
    for s in range(n_splits):
        rng = np.random.default_rng(split_seed0 + s)
        perm = rng.permutation(n_instances)
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        thr, f1_tr, f1_va = evaluate_split(scores, gt, train_idx, val_idx)
        thresholds.append(thr); f1_trains.append(f1_tr); f1_vals.append(f1_va)

    thresholds = np.array(thresholds)
    f1_trains = np.array(f1_trains)
    f1_vals = np.array(f1_vals)

    print(f"n_instances={n_instances}  n_splits={n_splits}  "
          f"train_frac={train_frac}  wall={time.time()-t0:.0f}s")
    print(f"  seuil          {thresholds.mean():.4f} +/- {thresholds.std():.4f}  "
          f"(min {thresholds.min():.4f}, max {thresholds.max():.4f})")
    print(f"  F1 train       {f1_trains.mean():.3f} +/- {f1_trains.std():.3f}")
    print(f"  F1 val         {f1_vals.mean():.3f} +/- {f1_vals.std():.3f}  "
          f"(min {f1_vals.min():.3f}, max {f1_vals.max():.3f})")
    print(f"  ecart train-val {(f1_trains - f1_vals).mean():+.3f} "
          f"+/- {(f1_trains - f1_vals).std():.3f}")

    payload = dict(
        n_instances=n_instances, n_splits=n_splits, train_frac=train_frac,
        N=N, n_patches=n_patches, re=re, base_seed=base_seed,
        thresholds=thresholds, f1_trains=f1_trains, f1_vals=f1_vals,
        **provenance.finish(started),
    )
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-instances", type=int, default=300)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--N", type=int, default=48)
    p.add_argument("--dim", dest="n_patches", type=int, default=3,
                   help="patches par cote (nom `--dim` : convention "
                        "partagee avec le reste de study/h2b_prediction)")
    p.add_argument("--re", type=float, default=800.0)
    p.add_argument("--seed", type=int, default=0,
                   help="graine du premier instantane jouet (base_seed)")
    p.add_argument("--split-seed", type=int, default=0)
    args = p.parse_args()

    payload = run(args.n_instances, args.n_splits, args.train_frac,
                 args.N, args.n_patches, args.re, args.seed, args.split_seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR,
        f"h2b_toy_ceiling_N{args.N}_dim{args.n_patches}"
        f"_n{args.n_instances}.npz")
    np.savez(out_path, **payload)
    print(f"\nEcrit : {out_path}")


if __name__ == "__main__":
    main()
