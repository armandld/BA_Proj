#!/usr/bin/env python3
"""GBT/RF/LR sur donnees jouets, ressources comparables au pipeline QAOA :
compare le seuil classique (`score_classical`) a des modeles appris sur les
memes features locales (`h2b_ceiling_random_split.py`, reutilisees telles
quelles), en evitant deux pieges deja trouves sur DNS reelles (D-198,
`docs/DEFAUTS.md`) :

  1. `early_stopping="auto"` (defaut sklearn) ne se declenche qu'au-dela de
     10000 echantillons, jamais atteint ici -- `early_stopping=True` est
     explicite des le depart.
  2. Seuil ET modeles sont ajustes sur un train et evalues sur un val
     jamais vu, par partage d'INSTANCE (pas de patch), sur plusieurs
     partages disjoints -- meme discipline que `h2b_toy_ceiling.py`.

Ce script NE reproduit PAS la cause dominante de D-198 (le signe
score->label s'inverse d'un scenario reel a l'autre, mesure par LOSO) :
les instances jouets sont des tirages i.i.d. de la meme distribution, pas
quatre regimes physiques entre lesquels un modele devrait transferer. Un
partage aleatoire (pas un LOSO artificiel) est donc le protocole honnete
ici.

Usage:
  python study/h2b_prediction/h2b_toy_gbt_ceiling.py --n-instances 300 --n-splits 5
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

from toy_model import generate_toy_snapshot
from hard_patch_labels import patch_l2_errors
from h2b_ceiling_random_split import (
    extract_features_2d, stencil_features, make_model, fit_eval,
    best_threshold_f1)
from config import L2_PERCENTILE_HARD
import provenance

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")


def gather_pool(n_instances, N, n_patches, re, base_seed):
    """Une entree par instance jouet -- split plus tard par INSTANCE, pas
    par patch, meme discipline que `h2b_toy_ceiling.py`."""
    X_site, X_sten, Y, S = [], [], [], []
    for i in range(n_instances):
        seed = base_seed + i
        vx, vy, Bx, By = generate_toy_snapshot(N=N, seed=seed)
        feats_2d, score = extract_features_2d(vx, vy, Bx, By, N, n_patches, re)
        l2 = patch_l2_errors(vx, vy, Bx, By, n_patches)
        gt = (l2 >= np.percentile(l2, L2_PERCENTILE_HARD)).ravel().astype(int)
        X_site.append(feats_2d.reshape(-1, feats_2d.shape[-1]))
        X_sten.append(stencil_features(feats_2d))
        Y.append(gt)
        S.append(score.ravel())
    return X_site, X_sten, Y, S


def _stack(pool, idxs):
    return np.concatenate([pool[i] for i in idxs], axis=0)


def evaluate_split(X_site, X_sten, Y, S, train_idx, val_idx, seed):
    Xtr_site, Xva_site = _stack(X_site, train_idx), _stack(X_site, val_idx)
    Xtr_sten, Xva_sten = _stack(X_sten, train_idx), _stack(X_sten, val_idx)
    Ytr, Yva = _stack(Y, train_idx), _stack(Y, val_idx)
    Str, Sva = _stack(S, train_idx), _stack(S, val_idx)

    thr_star, f1_class_tr = best_threshold_f1(Str, Ytr)
    f1_class_va = f1_score(Yva, (Sva > thr_star).astype(int), zero_division=0)

    # `early_stopping=True` explicite partout (D-198) ; sans effet sur
    # "lr"/"rf", ou le parametre est simplement ignore.
    site_f1 = {}
    for name in ("lr", "rf", "gbt"):
        r = fit_eval(make_model(name, seed, early_stopping=True),
                     Xtr_site, Ytr, Xva_site, Yva)
        site_f1[name] = r["f1"]
    best_site_model = max(site_f1, key=site_f1.get)
    f1_site_best = site_f1[best_site_model]

    r_sten = fit_eval(make_model("gbt", seed, early_stopping=True),
                       Xtr_sten, Ytr, Xva_sten, Yva)

    return dict(
        thr=thr_star, f1_class_train=f1_class_tr, f1_class_val=f1_class_va,
        f1_site_lr=site_f1["lr"], f1_site_rf=site_f1["rf"],
        f1_site_gbt=site_f1["gbt"], f1_site_best=f1_site_best,
        best_site_model=best_site_model, f1_stencil_gbt=r_sten["f1"],
    )


def run(n_instances, n_splits, train_frac, N, n_patches, re, base_seed,
        split_seed0):
    started = provenance.start()
    t0 = time.time()
    X_site, X_sten, Y, S = gather_pool(n_instances, N, n_patches, re,
                                        base_seed)
    n_train = int(round(n_instances * train_frac))

    keys = ("thr", "f1_class_train", "f1_class_val", "f1_site_lr",
            "f1_site_rf", "f1_site_gbt", "f1_site_best", "f1_stencil_gbt")
    per_split = {k: [] for k in keys}
    best_site_models = []
    for s in range(n_splits):
        rng = np.random.default_rng(split_seed0 + s)
        perm = rng.permutation(n_instances)
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        r = evaluate_split(X_site, X_sten, Y, S, train_idx, val_idx,
                            seed=split_seed0 + s)
        for k in keys:
            per_split[k].append(r[k])
        best_site_models.append(r["best_site_model"])

    arrs = {k: np.array(v) for k, v in per_split.items()}

    print(f"n_instances={n_instances}  n_splits={n_splits}  "
          f"train_frac={train_frac}  wall={time.time()-t0:.0f}s")
    print(f"  classical F1 val   {arrs['f1_class_val'].mean():.3f} +/- "
          f"{arrs['f1_class_val'].std():.3f}  (train "
          f"{arrs['f1_class_train'].mean():.3f} +/- "
          f"{arrs['f1_class_train'].std():.3f})")
    print(f"  Q1 site LR         {arrs['f1_site_lr'].mean():.3f} +/- "
          f"{arrs['f1_site_lr'].std():.3f}")
    print(f"  Q1 site RF         {arrs['f1_site_rf'].mean():.3f} +/- "
          f"{arrs['f1_site_rf'].std():.3f}")
    print(f"  Q1 site GBT        {arrs['f1_site_gbt'].mean():.3f} +/- "
          f"{arrs['f1_site_gbt'].std():.3f}")
    print(f"  Q1 site best       {arrs['f1_site_best'].mean():.3f} +/- "
          f"{arrs['f1_site_best'].std():.3f}  "
          f"(delta vs classical {(arrs['f1_site_best'] - arrs['f1_class_val']).mean():+.3f})")
    print(f"  Q2 stencil GBT     {arrs['f1_stencil_gbt'].mean():.3f} +/- "
          f"{arrs['f1_stencil_gbt'].std():.3f}  "
          f"(delta vs Q1 best {(arrs['f1_stencil_gbt'] - arrs['f1_site_best']).mean():+.3f})")

    payload = dict(
        n_instances=n_instances, n_splits=n_splits, train_frac=train_frac,
        N=N, n_patches=n_patches, re=re, base_seed=base_seed,
        best_site_models=np.array(best_site_models),
        **{k: arrs[k] for k in keys},
        **provenance.finish(started),
    )
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-instances", type=int, default=300)
    p.add_argument("--n-splits", type=int, default=5)
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
                  args.N, args.n_patches, args.re, args.seed,
                  args.split_seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR,
        f"h2b_toy_gbt_ceiling_N{args.N}_dim{args.n_patches}"
        f"_n{args.n_instances}.npz")
    np.savez(out_path, **payload)
    print(f"\nEcrit : {out_path}")


if __name__ == "__main__":
    main()
