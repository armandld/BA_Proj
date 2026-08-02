#!/usr/bin/env python3
"""
V4 Task 11b - Deplacement variationnel : le QAOA optimise-t-il vraiment son
propre Hamiltonien ? (audit, Priorite 0 - attribution quantique)

MOTIVATION. La tache 11 etablit que l'etat fondamental exact du Hamiltonien
de cout est atteint par tous les solveurs classiques. Il reste a savoir OU se
situe la decision reellement prise par le pipeline. Le pipeline V1 ne lit pas
un etat fondamental : il seuille les marginales P(q_i = 1) d'un circuit a
profondeur finie (`refinement.py`, prob_map vs effective_thr). Or ces
marginales partent de l'encodage d'amplitude theta = 2 asin(sqrt(score)),
qui contient deja toute la decision classique.

MESURE. Dans l'espace des marginales m in [0,1]^n on compare trois points :
  m_theta : encodage classique seul (avant toute couche QAOA)
  m_qaoa  : sortie du circuit optimise a profondeur `reps`
  m_gs    : etat fondamental exact (marginales 0/1, cf. tache 11)

et on definit le DEPLACEMENT RELATIF VERS L'OPTIMUM

    progress = <m_qaoa - m_theta, m_gs - m_theta> / ||m_gs - m_theta||^2

qui vaut 0.0 si le circuit laisse la decision classique inchangee et 1.0 s'il
atteint l'optimum de son propre cout. On rapporte aussi la norme du
deplacement ||m_qaoa - m_theta|| et la distance restante ||m_gs - m_qaoa||.

INTERPRETATION PRE-SPECIFIEE.
  progress ~ 0  -> le circuit est une perturbation de l'encodage classique ;
                   tout effet observe est attribuable a l'encodage et a la
                   non-convergence, pas a la minimisation du cout.
  progress ~ 1  -> le circuit resout effectivement son objectif ; la
                   comparaison avec les solveurs classiques (tache 11)
                   devient le test d'attribution pertinent.
Une progression qui n'augmente pas avec la profondeur est rapportee comme
telle : elle signifie que l'objectif declare n'est pas l'objectif optimise.

Sortie : results/t11b_qaoa_displacement_N{N}_dim{D}.npz
Usage :
  python study/v4/t11b_qaoa_displacement.py --N 64 --dim 2 --reps 1 2 3 4
"""
import argparse, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "v3"))
sys.path.insert(0, _HERE)

from t1_feature_selection import git_commit_hash
from phase7_sa_baseline import build_ising_terms
from t11_solver_attribution import exhaustive_ground_state


# -------------------------------------------------------------------
# Helpers purs
# -------------------------------------------------------------------

def theta_marginals(data_in):
    """Marginales P(q=1) induites par le seul encodage d'amplitude.

    Convention V1 (`init_qbits_state`) : theta = 2 asin(sqrt(score)) donne
    P(|1>) = sin^2(theta/2) = score. Les blocs horizontal et vertical sont
    concatenes dans l'ordre des qubits (H puis V).
    """
    th = np.asarray(data_in["theta_h"], dtype=float).ravel()
    tv = np.asarray(data_in["theta_v"], dtype=float).ravel()
    return np.concatenate([np.sin(th / 2.0) ** 2, np.sin(tv / 2.0) ** 2])


def ground_state_marginals(spins):
    """Marginales de l'etat fondamental : 1.0 si spin -1 (raffiner), 0 sinon."""
    return (np.asarray(spins).ravel() == -1).astype(float)


def variational_progress(m_theta, m_qaoa, m_gs, eps=1e-12):
    """Fraction du chemin parcouru de l'encodage classique vers l'optimum.

    Projection scalaire du deplacement realise sur le deplacement requis.
    Retourne dict(progress, disp_norm, remaining, required).
    """
    m_theta = np.asarray(m_theta, float).ravel()
    m_qaoa = np.asarray(m_qaoa, float).ravel()
    m_gs = np.asarray(m_gs, float).ravel()
    required = m_gs - m_theta
    realised = m_qaoa - m_theta
    den = float(np.dot(required, required))
    progress = float(np.dot(realised, required) / den) if den > eps else np.nan
    return dict(
        progress=progress,
        disp_norm=float(np.linalg.norm(realised)),
        required=float(np.sqrt(den)),
        remaining=float(np.linalg.norm(m_gs - m_qaoa)),
    )


def mask_uniformity(spins):
    """True si la configuration est uniforme (raffiner tout / rien).

    Un etat fondamental uniforme ne porte aucune information spatiale :
    c'est l'analogue, au niveau du Hamiltonien, des planchers de
    degenerescence utilises au niveau des metriques (protocole v3, 1.3-B3).
    """
    s = np.asarray(spins).ravel()
    return bool(np.all(s == s[0]))


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="V4 Task 11b: variational displacement of the QAOA")
    from config import RESULTS_DIR, SCENARIOS, DNS_N, V2_THRESHOLD, TRAINED_THRESHOLD

    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--re", nargs="+", type=int, default=[400])
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-snaps", type=int, default=2)
    p.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--k-opt", type=int, default=100)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--mapper", choices=["v1", "v2"], default="v2")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from phase5_qaoa_eval import (
        prepare_qaoa_inputs, run_qaoa_on_snapshot, classical_warm_start_params)

    thr = V2_THRESHOLD if args.mapper == "v2" else TRAINED_THRESHOLD
    print("=" * 88)
    print("  V4 Task 11b: does the QAOA optimise its own Hamiltonian?")
    print(f"  N={args.N}  dim={args.dim} ({2*args.dim*args.dim} qubits)  "
          f"mapper={args.mapper}  reps={args.reps}  K_opt={args.k_opt}")
    print("  progress = 0 -> decision unchanged from the classical encoding;")
    print("  progress = 1 -> circuit reaches the optimum of its own cost.")
    print("=" * 88)
    print()

    rows = []
    for sc in args.scenario:
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            if not os.path.exists(dp):
                print(f"  SKIP {sc} Re={re}: missing DNS"); continue
            dns = np.load(dp)
            vx = dns["vx"].astype(np.float64); vy = dns["vy"].astype(np.float64)
            Bx = dns["Bx"].astype(np.float64); By = dns["By"].astype(np.float64)
            sel = sorted(set(int(round(i)) for i in np.linspace(
                0, len(vx) - 1, args.n_snaps + 1)[1:]))
            for si in sel:
                data_in, hp, score = prepare_qaoa_inputs(
                    vx[si], vy[si], Bx[si], By[si], args.N, args.dim, re,
                    use_v2=(args.mapper == "v2"))
                h, e, pq = build_ising_terms(hp, args.dim)
                n_q = 2 * args.dim * args.dim
                gs, E_gs, n_opt = exhaustive_ground_state(h, e, pq, n_q)
                m_theta = theta_marginals(data_in)
                m_gs = ground_state_marginals(gs)
                uni = mask_uniformity(gs)
                for reps in args.reps:
                    ws = classical_warm_start_params(score, thr, reps)
                    t0 = time.time()
                    marg, dh, dv, _, _ = run_qaoa_on_snapshot(
                        data_in, hp, args.dim, reps=reps, K_opt=args.k_opt,
                        shots=args.shots, backend_name="state_vector",
                        warm_start_params=ws)
                    d = variational_progress(m_theta, np.asarray(marg), m_gs)
                    dec = np.concatenate([dh.ravel(), dv.ravel()])
                    rows.append(dict(
                        scenario=sc, re=re, snap=si, reps=reps,
                        progress=d["progress"], disp=d["disp_norm"],
                        required=d["required"], remaining=d["remaining"],
                        gs_uniform=uni, n_optima=n_opt,
                        agree_gs=float(np.mean(dec == (gs == -1))),
                        mean_marg=float(np.mean(marg)),
                        wall=time.time() - t0))
                print(f"  {sc:<18} Re={re} snap={si:<3} gs_uniform={uni} "
                      f"n_optima={n_opt}")

    if not rows:
        print("no input."); return

    print("\n  " + "=" * 84)
    print(f"  {'reps':>5} {'progress':>10} {'||disp||':>10} {'||required||':>13} "
          f"{'||remaining||':>14} {'mean marg':>10} {'agree_gs':>9}")
    print("  " + "-" * 84)
    for reps in args.reps:
        rs = [r for r in rows if r["reps"] == reps]
        print(f"  {reps:>5} {np.nanmean([r['progress'] for r in rs]):>10.4f} "
              f"{np.mean([r['disp'] for r in rs]):>10.4f} "
              f"{np.mean([r['required'] for r in rs]):>13.4f} "
              f"{np.mean([r['remaining'] for r in rs]):>14.4f} "
              f"{np.mean([r['mean_marg'] for r in rs]):>10.4f} "
              f"{np.mean([r['agree_gs'] for r in rs]):>9.3f}")
    print("  " + "-" * 84)

    frac_uni = float(np.mean([r["gs_uniform"] for r in rows]))
    prog_all = np.nanmean([r["progress"] for r in rows])
    slope = (np.nanmean([r["progress"] for r in rows
                         if r["reps"] == max(args.reps)])
             - np.nanmean([r["progress"] for r in rows
                           if r["reps"] == min(args.reps)]))
    print(f"\n  exact ground state is a UNIFORM mask on "
          f"{frac_uni*100:.1f}% of snapshots "
          f"(no spatial information at the optimum)")
    print(f"  mean variational progress toward that optimum: {prog_all:.4f}")
    print(f"  change in progress from reps={min(args.reps)} to "
          f"{max(args.reps)}: {slope:+.4f}")
    print("\n  READING: " + (
        "the circuit stays at the classical encoding; the deployed decision "
        "is not a minimiser of its declared cost."
        if abs(prog_all) < 0.1 else
        "the circuit moves substantially toward its own optimum."))

    out = os.path.join(
        RESULTS_DIR, f"t11b_qaoa_displacement_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        scenario=np.array([r["scenario"] for r in rows]),
        re=np.array([r["re"] for r in rows]),
        snap=np.array([r["snap"] for r in rows]),
        reps=np.array([r["reps"] for r in rows]),
        progress=np.array([r["progress"] for r in rows]),
        disp=np.array([r["disp"] for r in rows]),
        required=np.array([r["required"] for r in rows]),
        remaining=np.array([r["remaining"] for r in rows]),
        gs_uniform=np.array([r["gs_uniform"] for r in rows]),
        n_optima=np.array([r["n_optima"] for r in rows]),
        agree_gs=np.array([r["agree_gs"] for r in rows]),
        mean_marg=np.array([r["mean_marg"] for r in rows]),
        frac_uniform=frac_uni, mean_progress=float(prog_all),
        seed=args.seed, git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nV4 Task 11b complete.")


if __name__ == "__main__":
    main()
