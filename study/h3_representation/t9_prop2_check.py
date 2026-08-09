#!/usr/bin/env python3
"""
V3 Task 9 - Verificateur de la condition de la Proposition 2
(protocole v3, section 2, livrable theorique).

Proposition 2 (reformulee conditionnellement) : si pour chaque site i

    sum_{j voisin de i} 2|C_ij|  +  sum_{p contenant i} 4|K_p|  <  |h_i|

alors l'etat fondamental exact est s_i* = -sign(h_i) (aucun
retournement collectif n'est energetiquement favorable) : la decision
est strictement par site. Ce script rapporte la fraction de sites
satisfaisant la condition STRICTE, par scenario / Re / dim, pour les
Hamiltoniens V1 et V2 (via `build_patch_hamiltonian`, jamais
re-implemente).

Topologie (miroir exact de VQA/cost_hamiltonian.create_period_hamiltonian) :
  - qubits 0..dim^2-1 : liens horizontaux H(i,j) ; dim^2..2dim^2-1 :
    liens verticaux V(i,j) ;
  - Z : H_edges[0][i,j] -> H(i,j) ; H_edges[1][i,j] -> V(i,j) ;
  - ZZ : C_edges[0][i,j] couple H(i,j)-H(i,j+1) ;
         C_edges[1][i,j] couple V(i,j)-V(i+1,j) (periodique) ;
  - ZZZZ : K_plaquettes[i,j] sur [H(i,j), V(i,j+1), H(i+1,j), V(i,j)] ;
    K_xpoint (si present et non nul) sur la meme plaquette.

Parametres des mappers (provenance Task 5) : V2 = sans parametre
(c_bias=1.0, thr=0.15, comme extract_features_2d) ; V1 = TRAINED_* de
study/config.py (essai Optuna #4).

Sortie : results/t9_prop2_N{N}.npz ; une table par dim.
Usage :
  python study/v3/t9_prop2_check.py --N 256 --dim 2 4
"""
import argparse, json, os, sys, time
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
from t8_dns_extension import seeded_dns_path

MAPPERS = ("v1", "v2")


# -------------------------------------------------------------------
# Comptabilite par site (fonction pure, testable sans qiskit)
# -------------------------------------------------------------------

def per_site_condition(hamilt_params, dim, include_xpoint=True):
    """Pour chaque qubit-lien i : lhs_i = somme des 2|C| incidents +
    somme des 4|K| des plaquettes contenant i ; condition stricte
    lhs_i < |h_i| (formule pre-enregistree, section 2).

    Retourne (lhs (2 dim^2,), h SIGNE (2 dim^2,), satisfied (bool,))."""
    n = 2 * dim * dim
    offset_v = dim * dim

    def idx_h(y, x):
        return (y % dim) * dim + (x % dim)

    def idx_v(y, x):
        return offset_v + (y % dim) * dim + (x % dim)

    H = hamilt_params["H_edges"]
    C = hamilt_params["C_edges"]
    K = hamilt_params["K_plaquettes"]
    KX = hamilt_params.get("K_xpoint") if include_xpoint else None

    h = np.zeros(n)
    lhs = np.zeros(n)
    for i in range(dim):
        for j in range(dim):
            h[idx_h(i, j)] += H[0][i, j]
            h[idx_v(i, j)] += H[1][i, j]

            c = abs(C[0][i, j])
            lhs[idx_h(i, j)] += 2 * c
            lhs[idx_h(i, j + 1)] += 2 * c
            c = abs(C[1][i, j])
            lhs[idx_v(i, j)] += 2 * c
            lhs[idx_v(i + 1, j)] += 2 * c

            kk = abs(K[i, j])
            if KX is not None:
                kk += abs(np.asarray(KX)[i, j])
            if kk > 0:
                for q in (idx_h(i, j), idx_v(i, j + 1),
                          idx_h(i + 1, j), idx_v(i, j)):
                    lhs[q] += 4 * kk

    return lhs, h, lhs < np.abs(h)


def mean_field_state(hamilt_params, dim):
    """Etat champ-moyen s_i* = -sign(h_i) (sign(0) -> +1)."""
    _, h, _ = per_site_condition(hamilt_params, dim)
    s = -np.sign(h)
    s[s == 0] = 1
    return s.astype(int)


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

def build_params(vx, vy, Bx, By, N, dim, re, mapper):
    """Hamiltonien V1 (TRAINED_*, essai #4) ou V2 (sans parametre)."""
    from phase4_exact_diag import build_patch_hamiltonian
    from config import (TRAINED_THRESHOLD, TRAINED_SIGMA,
                        TRAINED_BETA_CURL, TRAINED_BETA_XPOINT,
                        TRAINED_W_Z_FRAC, TRAINED_GAMMA_HYDRO,
                        TRAINED_GAMMA_MAG, TRAINED_KAPPA)
    if mapper == "v2":
        hp, _, _ = build_patch_hamiltonian(
            vx, vy, Bx, By, N, dim, re,
            threshold_amr=0.15, use_v2=True, c_bias=1.0)
    else:
        hp, _, _ = build_patch_hamiltonian(
            vx, vy, Bx, By, N, dim, re,
            threshold_amr=TRAINED_THRESHOLD, use_v2=False,
            sigma=TRAINED_SIGMA, beta_curl=TRAINED_BETA_CURL,
            beta_xpoint=TRAINED_BETA_XPOINT,
            w_z_frac=TRAINED_W_Z_FRAC,
            gamma_hydro=TRAINED_GAMMA_HYDRO,
            gamma_mag=TRAINED_GAMMA_MAG, kappa=TRAINED_KAPPA)
    return hp


def main():
    p = argparse.ArgumentParser(
        description="V3 Task 9: Proposition-2 strict mean-field "
                    "condition checker")
    from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N
    from t8_dns_extension import EXTRA_SCENARIOS

    all_scenarios = SCENARIOS + EXTRA_SCENARIOS
    p.add_argument("--scenario", nargs="+", default=all_scenarios)
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--dim", nargs="+", type=int, default=[2, 4])
    p.add_argument("--phys-seed", nargs="+", type=int, default=[0])
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0,
                   help="enregistre (pipeline deterministe)")
    args = p.parse_args()

    print("=" * 88)
    print("  V3 Task 9: Proposition-2 condition "
          "sum 2|C| + sum 4|K| < |h| per site")
    print(f"  N={args.N}  dims={args.dim}  phys-seeds={args.phys_seed}  "
          f"max-snaps/cfg={args.max_snaps}")
    print("  mappers: v1 = TRAINED_* (Optuna trial #4), "
          "v2 = parameter-free (c_bias=1.0, thr=0.15)")
    print("=" * 88)
    print()

    rows = []  # dict(scenario, re, seed, dim, mapper, frac, n_sites)
    t0 = time.time()
    for sc in args.scenario:
        for re in args.re:
            for seed in args.phys_seed:
                dp = seeded_dns_path(RESULTS_DIR, sc, re, args.N, seed)
                if not os.path.exists(dp):
                    print(f"  SKIP {sc} Re={re} seed={seed}: no DNS")
                    continue
                dns = np.load(dp)
                vx_a = dns["vx"].astype(np.float64)
                vy_a = dns["vy"].astype(np.float64)
                Bx_a = dns["Bx"].astype(np.float64)
                By_a = dns["By"].astype(np.float64)
                n_snaps = len(vx_a)
                step = max(1, n_snaps // args.max_snaps)
                idx = list(range(0, n_snaps, step))[:args.max_snaps]

                for dim in args.dim:
                    fracs = {m: [] for m in MAPPERS}
                    for si in idx:
                        for m in MAPPERS:
                            hp = build_params(
                                vx_a[si], vy_a[si], Bx_a[si], By_a[si],
                                args.N, dim, re, m)
                            _, _, sat = per_site_condition(hp, dim)
                            fracs[m].append(sat.mean())
                    for m in MAPPERS:
                        rows.append(dict(
                            scenario=sc, re=re, seed=seed, dim=dim,
                            mapper=m,
                            frac=float(np.mean(fracs[m])),
                            n_snaps=len(idx),
                            n_sites=2 * dim * dim))
    print(f"  built {len(rows)} (config, dim, mapper) entries "
          f"in {time.time() - t0:.1f}s\n")
    if not rows:
        print("no input."); return

    # ---- tables par dim ----
    for dim in args.dim:
        sub = [r for r in rows if r["dim"] == dim]
        if not sub:
            continue
        print(f"  [dim={dim}]  fraction of sites with strict "
              "mean-field condition (mean over snapshots)")
        print(f"  {'scenario':<20} {'Re':>5} {'seed':>5} "
              f"{'v1-frac':>8} {'v2-frac':>8}")
        print("  " + "-" * 50)
        keys = sorted({(r["scenario"], r["re"], r["seed"])
                       for r in sub},
                      key=lambda k: (all_scenarios.index(k[0]),
                                     k[1], k[2]))
        for sc, re, seed in keys:
            cell = {m: next(r["frac"] for r in sub
                            if (r["scenario"], r["re"], r["seed"],
                                r["mapper"]) == (sc, re, seed, m))
                    for m in MAPPERS}
            print(f"  {sc:<20} {re:>5} {seed:>5} "
                  f"{cell['v1']:>8.3f} {cell['v2']:>8.3f}")
        for m in MAPPERS:
            mean = float(np.mean([r["frac"] for r in sub
                                  if r["mapper"] == m]))
            print(f"  {'MEAN ' + m:<32} {mean:>8.3f}")
        print()

    # ---- sauvegarde ----
    out = os.path.join(RESULTS_DIR, f"t9_prop2_N{args.N}.npz")
    np.savez_compressed(
        out,
        scenario=np.array([r["scenario"] for r in rows]),
        re=np.array([r["re"] for r in rows]),
        phys_seed=np.array([r["seed"] for r in rows]),
        dim=np.array([r["dim"] for r in rows]),
        mapper=np.array([r["mapper"] for r in rows]),
        frac=np.array([r["frac"] for r in rows]),
        n_snaps=np.array([r["n_snaps"] for r in rows]),
        seed=args.seed,
        git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"  saved: {os.path.basename(out)}")
    print("\nV3 Task 9 complete.")


if __name__ == "__main__":
    main()
