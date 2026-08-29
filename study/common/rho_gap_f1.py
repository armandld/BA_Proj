#!/usr/bin/env python3
"""rho(E_gap, F1) — le critere de decision de la reoptimisation.

MESURE DE REFERENCE, avant campagne, sur
`h0_optimiser_equivalence_N96_dim3_hamiltonien_corrige.npz` (rejouee par ce
module meme, coherente avec RESULTS.md) :

    rho = +0.870   p = 0.0023   9 solveurs

Un rho POSITIF signifie que **mieux resoudre H degrade la decision** : le
solveur qui atteint l'optimum certifie a le F1 le plus bas, celui qui s'en
ecarte le plus a le meilleur.

C'est le critere pre-enregistre de la campagne :

  * rho passe NEGATIF  -> il existe des hyperparametres pour lesquels
                          l'optimum de H est la bonne decision AMR. Le
                          modele tient, le reglage suffisait.
  * rho reste POSITIF  -> c'est la FORME de l'hamiltonien qu'il faut
                          revoir, pas ses coefficients. Aucune campagne
                          Optuna ne trouvera ca.

Post-traitement (option A) : ce module ne tourne PAS dans la boucle
d'entrainement. Il se lance apres, sur les artefacts de
`h0_optimiser_equivalence`. La campagne n'en porte donc aucun risque.

Usage :
    python study/common/rho_gap_f1.py results/h0_*.npz
    python study/common/rho_gap_f1.py --json rho.json results/h0_*.npz
"""

import argparse
import json
import sys

import numpy as np


def rho_gap_f1(chemin):
    """rho de Spearman entre l'ecart a l'optimum et le F1, par solveur."""
    from scipy.stats import spearmanr

    d = np.load(chemin, allow_pickle=True)
    for cle in ("solver", "E_gap", "f1"):
        if cle not in d:
            return {"fichier": chemin, "erreur": f"cle '{cle}' absente"}

    sol = d["solver"]
    noms = sorted(set(sol.tolist()))
    gap = [float(np.nanmean(d["E_gap"][sol == s].astype(float))) for s in noms]
    f1 = [float(np.nanmean(d["f1"][sol == s].astype(float))) for s in noms]

    fini = [i for i in range(len(noms))
            if np.isfinite(gap[i]) and np.isfinite(f1[i])]
    if len(fini) < 3:
        return {"fichier": chemin,
                "erreur": f"{len(fini)} solveurs exploitables, 3 minimum"}
    g = [gap[i] for i in fini]
    f = [f1[i] for i in fini]
    if np.ptp(g) == 0 or np.ptp(f) == 0:
        return {"fichier": chemin,
                "erreur": "E_gap ou F1 constant — rho indefini"}

    r = spearmanr(g, f)
    i_best, i_worst = int(np.argmax(f)), int(np.argmin(f))
    return {
        "fichier": chemin,
        "rho": float(r.statistic),
        "p": float(r.pvalue),
        "n_solveurs": len(fini),
        "meilleur_F1": {"solveur": noms[fini[i_best]],
                        "F1": f[i_best], "E_gap": g[i_best]},
        "pire_F1": {"solveur": noms[fini[i_worst]],
                    "F1": f[i_worst], "E_gap": g[i_worst]},
        "verdict": ("l'optimum de H N'EST PAS la bonne decision"
                    if r.statistic > 0 else
                    "l'optimum de H EST la bonne decision"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("artefacts", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print("=" * 72)
    print("  rho(E_gap, F1) — mieux resoudre H aide-t-il a mieux decider ?")
    print("  Reference avant campagne : rho = +0.870 (p = 0.0023, 9 solveurs)")
    print("=" * 72)

    rapport = []
    for chemin in args.artefacts:
        r = rho_gap_f1(chemin)
        rapport.append(r)
        nom = chemin.split("/")[-1]
        if "erreur" in r:
            print(f"\n  {nom}\n      IGNORE : {r['erreur']}")
            continue
        print(f"\n  {nom}")
        print(f"      rho = {r['rho']:+.3f}   p = {r['p']:.4f}   "
              f"({r['n_solveurs']} solveurs)")
        print(f"      meilleur F1 : {r['meilleur_F1']['solveur']} "
              f"(F1={r['meilleur_F1']['F1']:.3f}, "
              f"E_gap={r['meilleur_F1']['E_gap']:.4f})")
        print(f"      pire     F1 : {r['pire_F1']['solveur']} "
              f"(F1={r['pire_F1']['F1']:.3f}, "
              f"E_gap={r['pire_F1']['E_gap']:.4f})")
        print(f"      -> {r['verdict']}")

    exploitables = [r for r in rapport if "rho" in r]
    if exploitables:
        signes = {np.sign(r["rho"]) for r in exploitables}
        print("\n" + "=" * 72)
        if signes == {1.0}:
            print("  rho POSITIF partout : la forme de l'hamiltonien est en cause.")
        elif signes == {-1.0}:
            print("  rho NEGATIF partout : le reglage suffisait, le modele tient.")
        else:
            print("  SIGNES MELANGES : rho depend de la configuration, a instruire.")
        print("=" * 72)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rapport, fh, indent=2)
        print(f"\nrapport : {args.json}")
    return 0 if exploitables else 1


if __name__ == "__main__":
    sys.exit(main())
