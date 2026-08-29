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

# Le nom porte le mappeur des qu'il n'est pas le defaut v2 :
# sans cela, relancer avec --mapper v1 ecraserait le resultat v2
# et la comparaison entre mappeurs ne tiendrait pas dans les
# artefacts (defaut D9, deja rencontre sur t13 et t19).
Sortie : results/h0_qaoa_displacement_N{N}_dim{D}.npz
Usage :
  python study/h0_selection/h0_qaoa_displacement.py --N 64 --dim 2 --reps 1 2 3 4
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

from h2b_feature_selection import git_commit_hash
from ising_terms_and_annealing import build_ising_terms, exhaustive_ground_state


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


# ══════════════════════════════════════════════════════════════════════
#  CRITERE D'ACCEPTATION
# ══════════════════════════════════════════════════════════════════════
#
# Ce fichier produit trois nombres epingles dans la table maitresse
# (progression moyenne, progression a reps=1, a reps=4). Il n'avait aucune
# assertion : la lecture etait imprimee en prose et le script sortait 0 quoi
# qu'il mesure.
MAX_FRAC_UNDEFINED = 0.50   # au-dela, la moyenne ne decrit plus l'ensemble
MIN_PAIRED = 1              # une pente sans paire appariee n'est pas une pente


READING_FLAT = ("progress toward the optimum does not grow with circuit "
                "depth; the deployed decision may not be attributable to "
                "the QAOA minimisation itself.")
READING_MOVES = ("progress toward the optimum grows with circuit depth, "
                  "consistent with the circuit minimising its declared cost.")
READING_THRESHOLD = 0.1


def reading_message(slope):
    """La phrase de conclusion de T11b, extraite pour etre testable.

    Le verdict lisait `prog_all`, la moyenne d'UN tirage QAOA par
    instantane : trois executions de la commande publiee (`--N 256 --dim
    2 --n-snaps 2`, reps 1-4) rendaient 0.1034 / 0.0850 / 0.0859 contre
    le seuil de 0.1 — une execution sur trois imprimait la conclusion
    inverse.

    Il lit desormais `slope` (`slope_paired` dans l'artefact) : la pente
    APPARIEE progress(reps=max) - progress(reps=min), sur les memes
    instantanes aux deux profondeurs (voir `main`, juste au-dessus de
    l'appel). Question differente, plus proche de la motivation du
    fichier : pas seulement decide -- e.g. le progres BOUGE-T-IL avec
    la profondeur, ce qui distingue directement une perturbation de
    l'encodage classique (pente ~ 0) d'une minimisation reelle du cout
    (pente qui croit avec la profondeur).

    Le seuil 0.1 est INCHANGE et reste sans provenance ecrite propre a
    `slope` : sa nature a change (d'une moyenne de tirage a une pente
    appariee), sa reproductibilite n'a pas ete remesuree sur plusieurs
    executions independantes -- seulement verifiee sur UNE execution que
    la logique s'applique sans planter. Une inversion de conclusion d'une
    execution a l'autre reste possible ; personne ne l'a mesuree pour
    cette grandeur.
    """
    return (READING_FLAT if abs(slope) < READING_THRESHOLD
            else READING_MOVES)


def check_expected_behaviour(rows, frac_undef, prog_all, paired, slope):
    assert rows, "aucun instantane : rien a juger"

    assert frac_undef <= MAX_FRAC_UNDEFINED, (
        f"la progression est indefinie sur {frac_undef*100:.1f} % des "
        f"instantanes (limite {MAX_FRAC_UNDEFINED*100:.0f} %). La moyenne "
        "porterait sur une minorite selectionnee par la degenerescence "
        "elle-meme : elle ne doit pas etre publiee en l'etat.")

    assert np.isfinite(prog_all), (
        "progression moyenne indefinie : aucun instantane exploitable")

    assert len(paired) >= MIN_PAIRED and np.isfinite(slope), (
        f"pente calculee sur {len(paired)} instantanes apparies : deux "
        "moyennes independantes compareraient deux populations differentes")

    print(f"\n  [ACCEPTANCE] progression definie sur "
          f"{(1 - frac_undef)*100:.1f} % des instantanes, pente appariee sur "
          f"{len(paired)} -> nombres publiables.")


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

    from qaoa_inputs import (
        prepare_qaoa_inputs, run_qaoa_on_snapshot, constant_initial_params)

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
                    ws = constant_initial_params(reps)
                    t0 = time.time()
                    marg, dh, dv, _, _ = run_qaoa_on_snapshot(
                        data_in, hp, args.dim, reps=reps, K_opt=args.k_opt,
                        shots=args.shots, backend_name="state_vector",
                        warm_start_params=ws, seed=args.seed)
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
        raise SystemExit(
            "aucun instantane traite : un balayage vide ne doit pas sortir 0, "
            "sinon il est indiscernable d'un balayage qui a tout verifie")

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

    # `progress` est INDEFINIE quand le deplacement requis est nul, c'est-a-dire
    # quand l'initialisation classique EST deja l'etat fondamental. Ces
    # instantanes sont exactement les cas degeneres que cette tache cherche a
    # caracteriser ; les faire disparaitre dans un `nanmean` revient a moyenner
    # sur un sous-ensemble selectionne par la propriete etudiee.
    n_undef = int(np.sum([not np.isfinite(r["progress"]) for r in rows]))
    frac_undef = n_undef / len(rows)
    prog_all = np.nanmean([r["progress"] for r in rows])

    # La pente doit comparer LES MEMES instantanes aux deux profondeurs.
    # Deux `nanmean` independants comparent deux populations differentes,
    # puisque le motif d'indefinition depend de reps.
    p_min, p_max = min(args.reps), max(args.reps)
    key = lambda r: (r["scenario"], r["re"], r["snap"])
    at = {q: {key(r): r["progress"] for r in rows if r["reps"] == q}
          for q in (p_min, p_max)}
    paired = [k for k in at[p_min]
              if k in at[p_max]
              and np.isfinite(at[p_min][k]) and np.isfinite(at[p_max][k])]
    slope = (float(np.mean([at[p_max][k] - at[p_min][k] for k in paired]))
             if paired else float("nan"))
    print(f"\n  exact ground state is a UNIFORM mask on "
          f"{frac_uni*100:.1f}% of snapshots "
          f"(no spatial information at the optimum)")
    print(f"  mean variational progress toward that optimum: {prog_all:.4f}"
          f"   [indefinie sur {n_undef}/{len(rows)} instantanes "
          f"= {frac_undef*100:.1f} %, ecartes de cette moyenne]")
    print(f"  change in progress from reps={p_min} to {p_max}: {slope:+.4f}"
          f"   [apparie sur {len(paired)} instantanes definis aux deux "
          f"profondeurs]")
    print("\n  READING: " + reading_message(slope))

    out = os.path.join(
        RESULTS_DIR, f"h0_qaoa_displacement_N{args.N}_dim{args.dim}"
        + ("" if args.mapper == "v2" else f"_{args.mapper}")
        + ".npz")
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
        n_undefined_progress=n_undef, frac_undefined_progress=frac_undef,
        slope_paired=slope, n_paired=len(paired),
        seed=args.seed, git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    check_expected_behaviour(rows, frac_undef, prog_all, paired, slope)
    print("\nV4 Task 11b complete.")


if __name__ == "__main__":
    main()
