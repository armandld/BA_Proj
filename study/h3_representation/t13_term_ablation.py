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

from t1_feature_selection import git_commit_hash
from phase7_sa_baseline import build_ising_terms, spins_to_decisions
from t11_solver_attribution import exhaustive_ground_state, f1_from_masks
from t11b_qaoa_displacement import mask_uniformity

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

    from phase5_qaoa_eval import prepare_qaoa_inputs

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
                        dE=float(E - base_E)))
                print(f"  {sc:<18} Re={re} snap={si:<3} "
                      f"base_uniform={base_uni}")

    if not rows:
        print("no input."); return

    names = [n for n, _ in ABLATIONS]
    print("\n  " + "=" * 80)
    print(f"  {'ablation':<18} {'changed':>9} {'uniform':>9} "
          f"{'refined':>9} {'F1':>8} {'n_optima':>10}")
    print("  " + "-" * 80)
    for n in names:
        rs = [r for r in rows if r["ablation"] == n]
        if not rs:
            continue
        print(f"  {n:<18} {np.mean([r['changed'] for r in rs]):>9.4f} "
              f"{np.mean([r['uniform'] for r in rs]):>9.3f} "
              f"{np.mean([r['refined'] for r in rs]):>9.3f} "
              f"{np.mean([r['f1'] for r in rs]):>8.3f} "
              f"{np.mean([r['n_optima'] for r in rs]):>10.1f}")
    print("  " + "-" * 80)

    ctrl = np.mean([r["changed"] for r in rows if r["ablation"] == "full"])
    print(f"\n  control ('full' ablation) changed fraction = {ctrl:.6f} "
          f"(must be 0.0)")
    print("  READING: a term whose removal changes no decision is inert for "
          "the deployed\n  decision at this grid size, whatever its "
          "magnitude in the cost function.")

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


if __name__ == "__main__":
    main()
