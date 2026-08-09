#!/usr/bin/env python3
"""T30 — labelliser les patches par une TOLERANCE D'ERREUR, pas un percentile.

Pourquoi
--------
`phase2_hard_patches.py` seuille au percentile 75 de chaque scenario : le
label est un rang intra-scenario, chaque scenario a exactement 25 % de
patches durs, et le LOSO demande de predire ce rang sans avoir vu le seuil
du scenario tenu a l'ecart.

T28 a remplace ca par un percentile calcule sur les quatre scenarios
reunis. C'etait mieux — le label redevient une fonction du patch seul — mais
le seuil voyait le scenario de test : retirer orszag_tang le fait chuter de
57 % a dim=16. C'etait donc une fuite.

T30 supprime le probleme a la racine : le seuil est une CONSTANTE choisie
d'avance, pas une statistique des donnees. Aucun scenario, present ou futur,
ne peut la deplacer, donc il n'y a rien a recalculer par fold et rien a
fuiter.

Lecture physique
----------------
`l2` est deja une erreur RELATIVE : ecart-type intra-patch des quatre champs,
divise par le RMS global du meme instantane. Donc

    is_hard = (l2 >= tau)

se lit : « ecraser ce patch a sa moyenne fait perdre plus de tau x 100 % de
l'amplitude RMS du champ a cet instant ». La normalisation reste interne a
l'instantane — elle ne depend d'aucun autre scenario.

La prevalence cesse d'etre un parametre : elle devient une mesure, par
scenario ET par echelle.

Un seul tau serait arbitraire a son tour. Le script en balaie plusieurs et
refuse ceux qui degenerent (prevalence 0 ou 1 quelque part) : un label qui
ne separe rien n'est pas un label.

Sortie
------
`patches_{scenario}_Re{Re}_N{N}_dim{D}_tau{TAU}.npz`, format identique a
l'entree.

Usage
-----
  python study/pipeline/t30_tolerance_labels.py --dim 16 64 \
      --tau 0.02 0.05 0.10 0.20
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

from config import RESULTS_DIR

#: en-deca de cette prevalence (ou au-dela de 1 - ce nombre) dans UN
#: scenario, le label est degenere : le modele n'a plus qu'a repondre
#: toujours la meme chose.
MIN_PREVALENCE = 0.02


def tau_tag(tau):
    """Etiquette de fichier sans point decimal, pour rester lisible."""
    return f"tau{tau:g}".replace(".", "p")


def sources(dim, N, Re):
    pattern = os.path.join(RESULTS_DIR, f"patches_*_Re{Re}_N{N}_dim{dim}.npz")
    return sorted(p for p in glob.glob(pattern)
                  if "_globalthr" not in p and "_tau" not in p)


def label_at(tau, dim, N, Re, outdir=RESULTS_DIR):
    paths = sources(dim, N, Re)
    if not paths:
        raise SystemExit(
            f"aucun artefact patches_*_Re{Re}_N{N}_dim{dim}.npz ; lancer "
            "d'abord phase2_hard_patches.py")

    rows, written = [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        l2 = np.asarray(d["l2_errors"], dtype=float)
        is_hard = l2 >= tau
        prev = float(is_hard.mean())
        rows.append((str(d["scenario"]), prev))

        out = os.path.join(
            outdir,
            os.path.basename(p).replace(".npz", f"_{tau_tag(tau)}.npz"))
        np.savez_compressed(
            out,
            l2_errors=l2,
            classical_scores=d["classical_scores"],
            is_hard=is_hard,
            l2_threshold=float(tau),
            l2_threshold_per_scenario=float(d["l2_threshold"]),
            label_variant="absolute_tolerance",
            tau=float(tau),
            t=d["t"], scenario=d["scenario"], Re=d["Re"], N=d["N"],
            n_patches=d["n_patches"],
        )
        written.append(out)

    prevs = [p for _, p in rows]
    usable = (min(prevs) >= MIN_PREVALENCE
              and max(prevs) <= 1.0 - MIN_PREVALENCE)
    return rows, written, usable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", nargs="+", type=int, default=[16, 64])
    p.add_argument("--tau", nargs="+", type=float,
                   default=[0.02, 0.05, 0.10, 0.20])
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--re", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)  # deterministe, garde l'API
    args = p.parse_args()

    print("=" * 78)
    print("  T30 — labels par tolerance d'erreur absolue")
    print("=" * 78)
    print(f"  args: {' '.join(sys.argv[1:]) or '(defauts)'}")
    print(f"  is_hard = (l2 >= tau) ; l2 = ecart intra-patch / RMS(snapshot)")

    total, usable_taus = 0, {}
    for dim in args.dim:
        print(f"\n  --- dim={dim} (patches de {args.N // dim}x{args.N // dim} "
              f"cellules) ---")
        header = f"  {'tau':>7} " + "".join(
            f"{s[:12]:>14}" for s, _ in label_at(args.tau[0], dim, args.N,
                                                 args.re)[0]) + "   utilisable"
        print(header)
        print("  " + "-" * (len(header) - 2))
        ok_here = []
        for tau in args.tau:
            rows, written, usable = label_at(tau, dim, args.N, args.re)
            total += len(written)
            if usable:
                ok_here.append(tau)
            print(f"  {tau:>7.3f} "
                  + "".join(f"{prev:>14.4f}" for _, prev in rows)
                  + f"   {'oui' if usable else 'NON (degenere)'}")
        usable_taus[dim] = ok_here

    print(f"\n  {total} artefacts ecrits")
    print("  tolerances utilisables par echelle :")
    for dim, taus in usable_taus.items():
        print(f"    dim={dim:<3} {taus if taus else 'AUCUNE'}")

    # Un balayage qui ne laisse aucune tolerance exploitable doit crier.
    assert any(usable_taus.values()), (
        "aucune tolerance ne separe les quatre scenarios sans degenerer ; "
        "elargir --tau avant d'interpreter quoi que ce soit")
    print("\nT30 complete.")


if __name__ == "__main__":
    main()
