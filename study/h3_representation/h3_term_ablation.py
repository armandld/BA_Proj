#!/usr/bin/env python3
"""
V4 Task 13 - Ablations causales des termes du Hamiltonien (audit, Priorite 1).

QUESTION. Le papier attribue a posteriori certains basculements de decision
aux termes ZZ (tearing) et ZZZZ (OT). Ces attributions sont correlationnelles
et regroupees apres coup. L'audit demande l'ablation causale : retirer un
terme a la fois, rejouer la decision, et mesurer ce qui change.

PROTOCOLE. Pour chaque snapshot, on part du Hamiltonien complet et on annule
selectivement des familles de coefficients (`zero_hamiltonian_terms`), puis on
recalcule :
  - l'etat fondamental exact (enumeration, tache 11) et son masque ;
  - la decision QAOA reellement deployee (marginales seuillees), optionnelle.
On rapporte, par ablation : la fraction de patches dont la decision change vis
a vis du Hamiltonien complet, l'uniformite du nouvel etat fondamental, et le
F1 contre la verite terrain L2-hard.

L'ablation « aucun terme » sert de controle : elle doit donner exactement
zero changement, ce qui valide la chaine de mesure elle-meme.

Sortie : results/t13_term_ablation_N{N}_dim{D}.npz
Usage :
  python study/v4/t13_term_ablation.py --N 64 --dim 2 --n-snaps 2
"""
import argparse, json, os, sys
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

from h2b_feature_selection import git_commit_hash
from ising_terms_and_annealing import build_ising_terms, spins_to_decisions
from h0_optimiser_equivalence import exhaustive_ground_state, f1_from_masks
from h0_qaoa_displacement import mask_uniformity

# Familles de termes ablatables et cle correspondante dans hamilt_params.
TERM_KEYS = {"Z": "H_edges", "ZZ": "C_edges", "ZZZZ": "K_plaquettes"}

ABLATIONS = [
    ("full", ()),            # controle : doit donner 0 changement
    ("no_Z", ("Z",)),
    ("no_ZZ", ("ZZ",)),
    ("no_ZZZZ", ("ZZZZ",)),
    ("Z_only", ("ZZ", "ZZZZ")),
    ("couplings_only", ("Z",)),   # alias lisible de no_Z
]


def zero_hamiltonian_terms(hamilt_params, drop):
    """Copie de hamilt_params avec les familles de `drop` mises a zero.

    Ne modifie jamais le dictionnaire d'entree : le Hamiltonien complet
    reste disponible pour la comparaison.
    """
    out = dict(hamilt_params)
    for name in drop:
        key = TERM_KEYS[name]
        val = hamilt_params.get(key)
        if val is None:
            continue
        if isinstance(val, (tuple, list)):
            out[key] = tuple(np.zeros_like(np.asarray(a, dtype=float))
                             for a in val)
        else:
            out[key] = np.zeros_like(np.asarray(val, dtype=float))
    if "K_xpoint" in out and "ZZZZ" in drop and out["K_xpoint"] is not None:
        out["K_xpoint"] = np.zeros_like(np.asarray(out["K_xpoint"],
                                                   dtype=float))
    return out


def ground_state_mask(hamilt_params, dim):
    """Masque de raffinement de l'etat fondamental exact (dim <= 3)."""
    h, e, pq = build_ising_terms(hamilt_params, dim)
    n_q = 2 * dim * dim
    gs, E, n_opt = exhaustive_ground_state(h, e, pq, n_q)
    dh, dv = spins_to_decisions(np.asarray(gs), dim)
    return (dh | dv), float(E), int(n_opt), mask_uniformity(gs)


def coefficients_removed(hamilt_params, hamilt_params_ablated, dim):
    """max|Delta| de ce que `build_ising_terms` produit REELLEMENT.

    D-54 : le controle `full` ne peut pas distinguer « ablation qui retire un
    terme sans effet causal » de « ablation qui ne retire rien ». Mesure
    faite avec l'operateur assorti — pas sur les cles de `hamilt_params`,
    mais sur les trois tableaux que `ground_state_mask` consomme : mettre a
    zero une cle que `build_ising_terms` ne lit pas rend 0,0 ici, et c'est
    exactement le cas de `K_xpoint` (D-51).
    """
    h0, e0, p0 = build_ising_terms(hamilt_params, dim)
    h1, e1, p1 = build_ising_terms(hamilt_params_ablated, dim)
    deltas = [float(np.max(np.abs(np.asarray(h0) - np.asarray(h1))))
              if len(h0) else 0.0]
    # `build_ising_terms` n'emet un terme que si |coefficient| > 1e-12 : une
    # ablation RACCOURCIT donc les listes d'index au lieu d'y mettre des
    # zeros. Comparer les tableaux position par position n'a pas de sens ;
    # on les compare indexes par leur tuple de qubits, sur l'union des deux.
    for (idx0, c0), (idx1, c1) in ((e0, e1), (p0, p1)):
        d0 = {tuple(int(q) for q in row): float(c)
              for row, c in zip(np.asarray(idx0), np.asarray(c0, dtype=float))}
        d1 = {tuple(int(q) for q in row): float(c)
              for row, c in zip(np.asarray(idx1), np.asarray(c1, dtype=float))}
        keys = set(d0) | set(d1)
        deltas.append(max((abs(d0.get(k, 0.0) - d1.get(k, 0.0))
                           for k in keys), default=0.0))
    return float(max(deltas))


def main():
    p = argparse.ArgumentParser(
        description="V4 Task 13: causal term ablations")
    from config import RESULTS_DIR, SCENARIOS, DNS_N

    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--re", nargs="+", type=int, default=[400])
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-snaps", type=int, default=2)
    p.add_argument("--mapper", choices=["v1", "v2"], default="v2")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from qaoa_inputs import prepare_qaoa_inputs

    print("=" * 88)
    print("  V4 Task 13: causal ablation of Hamiltonian term families")
    print(f"  N={args.N}  dim={args.dim}  mapper={args.mapper}  "
          f"snaps/cfg={args.n_snaps}")
    print("  'full' is a control: it must produce exactly zero change.")
    print("=" * 88)
    print()

    rows = []
    for sc in args.scenario:
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(
                RESULTS_DIR,
                f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if not (os.path.exists(dp) and os.path.exists(pp)):
                print(f"  SKIP {sc} Re={re}: missing input"); continue
            dns = np.load(dp); pat = np.load(pp)
            vx = dns["vx"].astype(np.float64); vy = dns["vy"].astype(np.float64)
            Bx = dns["Bx"].astype(np.float64); By = dns["By"].astype(np.float64)
            l2 = pat["l2_errors"]; thr = float(pat["l2_threshold"])
            sel = sorted(set(int(round(i)) for i in np.linspace(
                0, len(vx) - 1, args.n_snaps + 1)[1:]))
            for si in sel:
                _, hp, _ = prepare_qaoa_inputs(
                    vx[si], vy[si], Bx[si], By[si], args.N, args.dim, re,
                    use_v2=(args.mapper == "v2"))
                gt = np.asarray(l2[si] >= thr)
                base_mask, base_E, _, base_uni = ground_state_mask(hp, args.dim)
                for name, drop in ABLATIONS:
                    hp_ab = zero_hamiltonian_terms(hp, drop)
                    mask, E, n_opt, uni = ground_state_mask(hp_ab, args.dim)
                    rows.append(dict(
                        scenario=sc, re=re, snap=si, ablation=name,
                        changed=float(np.mean(mask != base_mask)),
                        uniform=bool(uni), n_optima=n_opt,
                        f1=f1_from_masks(mask, gt),
                        refined=float(np.mean(mask)),
                        # D-54 : sans cette colonne, « changed = 0 » sur une
                        # ablation qui n'a rien retire est indiscernable d'un
                        # terme reellement inerte — les deux impriment 0,0000.
                        removed_max=coefficients_removed(hp, hp_ab, args.dim),
                        dE=float(E - base_E)))
                print(f"  {sc:<18} Re={re} snap={si:<3} "
                      f"base_uniform={base_uni}")

    if not rows:
        # D-55 : le script imprimait « no input. » et sortait avec le code 0,
        # sans ecrire d'artefact — donc en laissant en place celui d'une
        # campagne precedente, indiscernable d'une campagne reussie. Meme
        # defaut, meme formulation que la correction deja faite dans
        # `h0_optimiser_equivalence.main`.
        raise RuntimeError(
            f"balayage vide : aucun des scenarios {args.scenario} n'a "
            f"d'artefacts d'entree a N={args.N} dim={args.dim} "
            f"(dns_*_N{args.N}.npz et patches_*_N{args.N}_dim{args.dim}.npz "
            f"dans {RESULTS_DIR}). La tache sortait ici avec le code 0, sans "
            "artefact : celui de la campagne precedente restait en place et "
            "une campagne qui n'avait rien mesure etait indiscernable d'une "
            "campagne reussie.")

    names = [n for n, _ in ABLATIONS]
    print("\n  " + "=" * 92)
    print(f"  {'ablation':<18} {'changed':>9} {'removed_max':>12} "
          f"{'uniform':>9} {'refined':>9} {'F1':>8} {'n_optima':>10}")
    print("  " + "-" * 92)
    for n in names:
        rs = [r for r in rows if r["ablation"] == n]
        if not rs:
            continue
        print(f"  {n:<18} {np.mean([r['changed'] for r in rs]):>9.4f} "
              f"{np.max([r['removed_max'] for r in rs]):>12.4e} "
              f"{np.mean([r['uniform'] for r in rs]):>9.3f} "
              f"{np.mean([r['refined'] for r in rs]):>9.3f} "
              f"{np.mean([r['f1'] for r in rs]):>8.3f} "
              f"{np.mean([r['n_optima'] for r in rs]):>10.1f}")
    print("  " + "-" * 92)

    print("\n" + control_and_reading(rows))

    # Le nom porte le mappeur : sans lui, relancer la tache avec l'autre
    # mappeur ECRASAIT silencieusement le resultat precedent, alors que la
    # comparaison v1/v2 est justement l'un des points de la tache. Le nom
    # historique (sans suffixe) reste ecrit pour v1 afin de ne pas casser
    # les references deja publiees.
    out = os.path.join(
        RESULTS_DIR,
        f"t13_term_ablation_N{args.N}_dim{args.dim}_{args.mapper}.npz")
    np.savez_compressed(
        out,
        scenario=np.array([r["scenario"] for r in rows]),
        re=np.array([r["re"] for r in rows]),
        snap=np.array([r["snap"] for r in rows]),
        ablation=np.array([r["ablation"] for r in rows]),
        changed=np.array([r["changed"] for r in rows]),
        uniform=np.array([r["uniform"] for r in rows]),
        n_optima=np.array([r["n_optima"] for r in rows]),
        f1=np.array([r["f1"] for r in rows]),
        refined=np.array([r["refined"] for r in rows]),
        # D-54 : ce que `build_ising_terms` a reellement produit en moins.
        removed_max=np.array([r["removed_max"] for r in rows]),
        dE=np.array([r["dE"] for r in rows]),
        seed=args.seed, git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    if args.mapper == "v1":
        # compatibilite : le nom historique designe le mappeur deploye
        legacy = os.path.join(
            RESULTS_DIR, f"t13_term_ablation_N{args.N}_dim{args.dim}.npz")
        import shutil
        shutil.copyfile(out, legacy)
        print(f"  also written as: {os.path.basename(legacy)} (legacy name)")
    print("\nV4 Task 13 complete.")


# ══════════════════════════════════════════════════════════════════════
#  CONTROLE — D-54
# ══════════════════════════════════════════════════════════════════════
#
#  Le controle `full` compare `ground_state_mask(zero_hamiltonian_terms(hp,
#  ()))` a `ground_state_mask(hp)` : la MEME fonction sur la MEME entree.
#  Il vaut 0 par construction et ne peut echouer que sur un indeterminisme
#  d'`exhaustive_ground_state`, qui n'en a pas.
#
#  Mesure : en sabotant `TERM_KEYS` pour que plus rien ne soit jamais mis a
#  zero (orszag_tang Re=400 N=64 dim=2, 2 instantanes), le controle rend
#  0,000000 des DEUX cotes, et `no_ZZ` / `no_ZZZZ` / `Z_only` rendent
#  0,0000 des deux cotes — les trois lignes memes qui portent la lecture
#  « causalement inertes ». Le controle ne les distingue pas.
#
#  Ce qui les distingue est `removed_max` : ce que `build_ising_terms`
#  produit reellement en moins. Une ablation a `changed = 0` ET
#  `removed_max = 0` n'a rien retire — c'est le cas de `K_xpoint` (D-51) —
#  et ne dit rien de l'inertie du terme.
CONTROL_ABLATION = "full"


def control_and_reading(rows):
    """Le bloc de conclusion, extrait pour etre testable sans rejouer le
    balayage (meme decoupage que D-46 / D-50 / D-52). Leve si le controle
    ne vaut pas exactement 0 : il etait jusqu'ici imprime avec la mention
    « (must be 0.0) » et rien ne l'exigeait."""
    ctrl_rows = [r for r in rows if r["ablation"] == CONTROL_ABLATION]
    if not ctrl_rows:
        raise RuntimeError(
            f"aucune ligne de controle '{CONTROL_ABLATION}' : la chaine de "
            "mesure n'est pas verifiee du tout")
    ctrl = float(np.mean([r["changed"] for r in ctrl_rows]))
    if ctrl != 0.0:
        raise RuntimeError(
            f"le controle '{CONTROL_ABLATION}' vaut {ctrl:.6f} au lieu de 0 : "
            "rejouer le hamiltonien complet ne redonne pas la meme decision, "
            "donc aucune ablation n'est interpretable")

    out = [f"  control ('{CONTROL_ABLATION}' ablation) changed fraction = "
           f"{ctrl:.6f} (required 0.0, checked)"]

    # Une ablation qui n'a rien retire n'est pas une ablation.
    empty = sorted({r["ablation"] for r in rows
                    if r["ablation"] != CONTROL_ABLATION
                    and max(x["removed_max"] for x in rows
                            if x["ablation"] == r["ablation"]) == 0.0})
    if empty:
        out.append("  EMPTY ABLATIONS (removed_max = 0 -- build_ising_terms "
                   "produced exactly the same operator): "
                   + ", ".join(empty))
        out.append("  Their 'changed = 0' says NOTHING about the term's "
                   "causal role: nothing was removed. See D-51 / D-54.")
    out.append("  READING: a term whose removal changes no decision -- AND "
               "whose removal is\n  visible in removed_max -- is inert for "
               "the deployed decision at this grid size,\n  whatever its "
               "magnitude in the cost function.")
    return "\n".join(out)


if __name__ == "__main__":
    main()
