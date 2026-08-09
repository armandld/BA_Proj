#!/usr/bin/env python3
"""T28 — relabelliser les patches avec un seuil L2 GLOBAL.

Pourquoi
--------
`phase2_hard_patches.py` calcule le seuil « dur » comme le percentile 75 des
erreurs L2 **de chaque scenario pris separement**. Chaque scenario a donc
exactement 25 % de patches durs, avec des seuils qui different d'un facteur
2.8 (0.2779 pour harris_tearing, 0.7698 pour mhd_rotor a dim=4).

Le label est alors un RANG INTRA-SCENARIO : un patch d'erreur 0.5 est dur
dans tearing et facile dans rotor. En LOSO, le modele doit predire ce rang
sur un scenario dont il n'a jamais vu le seuil — l'information manque de son
entree, et il bascule vers une constante (F1 = 0.400 exactement = tout
positif, ou 0.000 = tout negatif).

Ce script produit la variante ABSOLUE : un seul seuil, le percentile 75 des
erreurs L2 des quatre scenarios reunis. La prevalence cesse alors d'etre
imposee a 25 % partout ; elle devient une propriete mesuree de chaque
scenario. C'est cette variante qui rend la question LOSO bien posee.

Aucune simulation n'est rejouee : les `l2_errors` sont deja dans les
artefacts de la phase 2.

Sortie
------
`patches_{scenario}_Re{Re}_N{N}_dim{D}_globalthr.npz`, format identique a
l'entree pour que la chaine aval fonctionne sans changement.

Usage
-----
  python study/pipeline/labels_global_threshold.py --dim 4 16 32 64
"""
import argparse
import glob
import os
import sys

import numpy as np

# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------

from config import L2_PERCENTILE_HARD, RESULTS_DIR

SUFFIX = "_globalthr"


def collect(dim, N, Re):
    """Artefacts de phase 2 pour cette (dim, N, Re), variante par scenario."""
    pattern = os.path.join(RESULTS_DIR, f"patches_*_Re{Re}_N{N}_dim{dim}.npz")
    paths = sorted(p for p in glob.glob(pattern) if SUFFIX not in p)
    return paths


def relabel(dim, N, Re, percentile=L2_PERCENTILE_HARD, outdir=RESULTS_DIR):
    paths = collect(dim, N, Re)
    if not paths:
        raise SystemExit(
            f"aucun artefact patches_*_Re{Re}_N{N}_dim{dim}.npz — lancer "
            "d'abord phase2_hard_patches.py"
        )

    pooled = []
    payloads = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        l2 = np.asarray(d["l2_errors"], dtype=float)
        pooled.append(l2.ravel())
        payloads.append((p, d, l2))
    pooled = np.concatenate(pooled)

    thr_global = float(np.percentile(pooled, percentile))

    # Un seuil qui ne separe rien est le defaut que ce script doit rendre
    # visible, pas propager.
    if not np.isfinite(thr_global) or thr_global <= 0.0:
        raise SystemExit(
            f"seuil global degenere ({thr_global}) : les erreurs L2 sont "
            "constantes ou nulles a cette echelle"
        )

    print(f"\n  dim={dim} N={N} Re={Re} — {len(paths)} scenarios, "
          f"{pooled.size} patches")
    print(f"  seuil GLOBAL (p{percentile}) = {thr_global:.6f}")
    print(f"  {'scenario':<20s} {'seuil par scenario':>18s} "
          f"{'prevalence avant':>17s} {'prevalence apres':>17s}")
    print(f"  {'-'*20} {'-'*18} {'-'*17} {'-'*17}")

    written, prevalences = [], []
    for path, d, l2 in payloads:
        scenario = str(d["scenario"])
        thr_local = float(d["l2_threshold"])
        is_hard = l2 >= thr_global
        prev_before = float(np.asarray(d["is_hard"]).mean())
        prev_after = float(is_hard.mean())
        prevalences.append(prev_after)

        print(f"  {scenario:<20s} {thr_local:>18.6f} "
              f"{prev_before:>17.4f} {prev_after:>17.4f}")

        out = os.path.join(
            outdir, os.path.basename(path).replace(".npz", f"{SUFFIX}.npz"))
        np.savez_compressed(
            out,
            l2_errors=l2,
            classical_scores=d["classical_scores"],
            is_hard=is_hard,
            l2_threshold=thr_global,
            l2_threshold_per_scenario=thr_local,
            label_variant="global_percentile",
            percentile=percentile,
            t=d["t"], scenario=d["scenario"], Re=d["Re"], N=d["N"],
            n_patches=d["n_patches"],
        )
        written.append(out)

    spread = max(prevalences) - min(prevalences)
    print(f"\n  ecart de prevalence entre scenarios : {spread:.4f}")
    if spread < 1e-9:
        raise SystemExit(
            "les prevalences restent identiques apres relabellisation : le "
            "seuil global n'a rien change, ce qui est impossible si les "
            "distributions differaient. Verifier l'entree."
        )
    return written, thr_global, prevalences


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", nargs="+", type=int, default=[4, 16, 32, 64])
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--re", type=int, default=400)
    p.add_argument("--percentile", type=float, default=L2_PERCENTILE_HARD)
    p.add_argument("--seed", type=int, default=0)  # deterministe, garde l'API
    args = p.parse_args()

    print("=" * 70)
    print("  T28 — labels a seuil L2 global")
    print("=" * 70)
    print(f"  args: {' '.join(sys.argv[1:]) or '(defauts)'}")

    total = 0
    for dim in args.dim:
        written, thr, prev = relabel(dim, args.N, args.re, args.percentile)
        total += len(written)

    print(f"\n  {total} artefacts ecrits (suffixe {SUFFIX})")
    assert total > 0, "aucun artefact ecrit"
    print("\nT28 complete.")


if __name__ == "__main__":
    main()
