#!/usr/bin/env python3
"""
V4 Task 12 - Test d'equivariance et erreur d'orbite (audit, Priorite 1).

MOTIVATION. Le papier signale que les masques Q-HAS sur des configurations
nominalement symetriques (KH, tearing) ne sont pas exactement equivariants,
et traite cette asymetrie comme un mode d'echec plutot que comme de la
physique detectee. L'audit demande le test correspondant : appliquer une
transformation de symetrie au champ, rejouer la carte de decision complete,
et rapporter une erreur d'orbite AVANT toute comparaison d'exactitude.

PROTOCOLE EN DEUX TEMPS (l'ordre est essentiel).

  (1) VALIDER LA TRANSFORMATION. Une transformation T n'est utilisable que
      si elle est une symetrie du solveur discret lui-meme. On mesure donc
      d'abord le defaut de commutation
          eps_solver = || T(step(U)) - step(T(U)) || / || step(U) ||
      avec `MHDSolver.step_full` (V1, importe). Si eps_solver est de l'ordre
      de l'erreur machine, T est une symetrie exacte du solveur et tout
      defaut ulterieur est imputable a la CARTE DE DECISION, pas a la
      physique ni a une convention de signe mal choisie.

  (2) MESURER L'ERREUR D'ORBITE de la decision
          eps_orbit = fraction de patches ou  D(T(U)) != T(D(U)).
      Rapportee pour chaque route de decision : score classique seuille,
      etat fondamental exact du Hamiltonien, et marginales QAOA seuillees
      (la route reellement deployee).

CONVENTIONS VECTORIELLES. La vitesse est un vecteur polaire ; le champ
magnetique est axial (pseudo-vecteur) : sous une reflexion, ses composantes
acquierent un signe supplementaire. Les deux conventions sont implementees
et c'est le test (1) qui tranche empiriquement laquelle commute avec le
solveur : aucune n'est postulee.

# Le nom porte le mappeur des qu'il n'est pas le defaut v2 :
# sans cela, relancer avec --mapper v1 ecraserait le resultat v2
# et la comparaison entre mappeurs ne tiendrait pas dans les
# artefacts (defaut D9, deja rencontre sur t13 et t19).
Sortie : results/t12_equivariance_N{N}_dim{D}.npz
Usage :
  python study/h3_representation/h3_equivariance.py --N 64 --dim 2 --n-snaps 2
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

# Groupe engendre par les reflexions d'axes et la rotation d'un quart de tour.
# Pour chaque operation : (transformation du tableau, matrice 2x2 agissant sur
# les composantes (f0, f1), signe supplementaire pour un champ axial).
SYMMETRY_OPS = {
    "flip0":  dict(arr=lambda f: np.flip(f, axis=0),
                   mat=np.array([[-1.0, 0.0], [0.0, 1.0]]), axial_sign=-1.0),
    "flip1":  dict(arr=lambda f: np.flip(f, axis=1),
                   mat=np.array([[1.0, 0.0], [0.0, -1.0]]), axial_sign=-1.0),
    "rot180": dict(arr=lambda f: np.flip(np.flip(f, axis=0), axis=1),
                   mat=np.array([[-1.0, 0.0], [0.0, -1.0]]), axial_sign=1.0),
    "rot90":  dict(arr=lambda f: np.rot90(f, k=1),
                   mat=np.array([[0.0, -1.0], [1.0, 0.0]]), axial_sign=1.0),
}


# -------------------------------------------------------------------
# Helpers purs
# -------------------------------------------------------------------

def apply_symmetry(vx, vy, Bx, By, op, axial_B=True):
    """Applique une operation de symetrie aux champs (v polaire, B axial).

    Retourne (vx', vy', Bx', By'). `axial_B=False` traite B comme un
    vecteur polaire : les deux variantes sont testees par le check de
    commutation avec le solveur.
    """
    o = SYMMETRY_OPS[op]
    m = o["mat"]
    s = o["axial_sign"] if axial_B else 1.0
    ax, ay = o["arr"](vx), o["arr"](vy)
    bx, by = o["arr"](Bx), o["arr"](By)
    vx2 = m[0, 0] * ax + m[0, 1] * ay
    vy2 = m[1, 0] * ax + m[1, 1] * ay
    Bx2 = s * (m[0, 0] * bx + m[0, 1] * by)
    By2 = s * (m[1, 0] * bx + m[1, 1] * by)
    return vx2, vy2, Bx2, By2


def apply_symmetry_mask(mask, op):
    """Applique la meme transformation spatiale a un masque de patches."""
    return SYMMETRY_OPS[op]["arr"](np.asarray(mask))


def orbit_error(mask_transformed_input, mask_transformed_output):
    """Fraction de patches ou D(T(U)) differe de T(D(U))."""
    a = np.asarray(mask_transformed_input, dtype=bool)
    b = np.asarray(mask_transformed_output, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("mask shapes differ")
    return float(np.mean(a != b))


def solver_noise_floor(vx, vy, Bx, By, N, dim, re, seeds=(0, 1, 2),
                       **kwargs):
    """Plancher de bruit : desaccord de la route « etat fondamental » entre
    graines de recuit, sur le MEME champ non transforme.

    CONTROLE INDISPENSABLE. Si l'etat fondamental est obtenu par recuit sur
    un paysage quasi degenere, deux graines donnent deja des masques
    differents. Une erreur d'orbite inferieure ou egale a ce plancher ne
    mesure PAS un defaut d'equivariance : elle mesure l'irreproductibilite
    de l'optimiseur. L'enumeration exhaustive est deterministe et donne un
    plancher exactement nul.

    Retourne (floor_moyen, fractions_raffinees).
    """
    masks = []
    for s in seeds:
        d = decision_maps(vx, vy, Bx, By, N, dim, re, seed=s, **kwargs)
        masks.append(np.asarray(d["ground_state"], dtype=bool))
    diffs = [float(np.mean(masks[0] != m)) for m in masks[1:]]
    fracs = [float(m.mean()) for m in masks]
    return (float(np.mean(diffs)) if diffs else 0.0), fracs


def solver_commutation_defect(vx, vy, Bx, By, N, re, op, axial_B=True,
                              dt=1e-3, cfl=0.4):
    """Defaut de commutation entre la symetrie et un pas du solveur V1.

    eps = || T(step(U)) - step(T(U)) || / || step(U) ||, norme L2 sur les
    quatre champs. Utilise `MHDSolver.step_full` (V1, jamais reimplemente)
    avec le MEME dt pour les deux branches.
    """
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver

    def _step(fields, dt_fixed):
        sim = MHDSolver(PeriodicGrid(N), dt=dt_fixed, Re=re, Rm=re)
        sim.vx, sim.vy, sim.Bx, sim.By = [np.array(f, copy=True)
                                          for f in fields]
        sim.step_full(record_stats=False)
        return (sim.vx, sim.vy, sim.Bx, sim.By)

    # dt commun, fige a partir de l'etat de reference (CFL identique)
    sim0 = MHDSolver(PeriodicGrid(N), dt=dt, Re=re, Rm=re)
    sim0.vx, sim0.vy, sim0.Bx, sim0.By = vx, vy, Bx, By
    dt_fixed = float(sim0.adapt_dt(cfl_target=cfl))

    stepped = _step((vx, vy, Bx, By), dt_fixed)
    t_then_step = _step(apply_symmetry(vx, vy, Bx, By, op, axial_B), dt_fixed)
    step_then_t = apply_symmetry(*stepped, op=op, axial_B=axial_B)

    num = np.sqrt(sum(np.sum((a - b) ** 2)
                      for a, b in zip(t_then_step, step_then_t)))
    den = np.sqrt(sum(np.sum(a ** 2) for a in stepped)) + 1e-30
    return float(num / den)


# -------------------------------------------------------------------
# Cartes de decision (routes reellement utilisees par le pipeline)
# -------------------------------------------------------------------

def decision_maps(vx, vy, Bx, By, N, dim, re, use_v2=True,
                  reps=2, k_opt=60, shots=4096, run_qaoa=True,
                  gs_solver="auto", sa_sweeps=400, sa_restarts=4, seed=0):
    """Retourne les masques (dim, dim) des routes de decision.

    `gs_solver` : 'exhaustive' (exact, <= 22 qubits), 'sa' (recuit, pour les
    grilles ou l'enumeration est impossible) ou 'auto' (exact si possible).
    Le choix effectif est journalise par l'appelant : un etat fondamental
    obtenu par recuit est une solution certifiee seulement a la precision du
    recuit, ce que la tache 11 a par ailleurs valide a 8 qubits.
    """
    from qaoa_inputs import (
        prepare_qaoa_inputs, run_qaoa_on_snapshot, constant_initial_params)
    from ising_terms_and_annealing import sa_multi_restart
    from config import V2_THRESHOLD, TRAINED_THRESHOLD

    thr = V2_THRESHOLD if use_v2 else TRAINED_THRESHOLD
    data_in, hp, score = prepare_qaoa_inputs(
        vx, vy, Bx, By, N, dim, re, use_v2=use_v2)
    out = {"classical": np.asarray(score > thr)}

    h, e, pq = build_ising_terms(hp, dim)
    n_q = 2 * dim * dim
    use_exact = (gs_solver == "exhaustive"
                 or (gs_solver == "auto" and n_q <= 22))
    if use_exact:
        gs, _, _ = exhaustive_ground_state(h, e, pq, n_q)
    else:
        gs, _, _ = sa_multi_restart(
            h, e, pq, n_q, sweeps=sa_sweeps, n_restarts=sa_restarts,
            rng=np.random.default_rng(seed))
    refine = (np.asarray(gs) == -1)
    n_cells = dim * dim
    out["ground_state"] = (refine[:n_cells].reshape(dim, dim)
                           | refine[n_cells:].reshape(dim, dim))

    if run_qaoa:
        ws = constant_initial_params(reps)
        _, dh, dv, _, _ = run_qaoa_on_snapshot(
            data_in, hp, dim, reps=reps, K_opt=k_opt, shots=shots,
            backend_name="state_vector", warm_start_params=ws, seed=seed)
        out["qaoa"] = np.asarray(dh | dv)
    return out


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="V4 Task 12: equivariance / orbit error")
    from config import RESULTS_DIR, SCENARIOS, DNS_N

    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--re", nargs="+", type=int, default=[400])
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-snaps", type=int, default=2)
    p.add_argument("--ops", nargs="+", default=list(SYMMETRY_OPS))
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--k-opt", type=int, default=60)
    p.add_argument("--mapper", choices=["v1", "v2"], default="v2")
    p.add_argument("--no-qaoa", action="store_true")
    p.add_argument("--gs-solver", choices=["auto", "exhaustive", "sa"],
                   default="auto",
                   help="'sa' permet les grilles > 22 qubits (dim >= 4)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  V4 Task 12: equivariance test (solver commutation, then orbit error)")
    print(f"  N={args.N}  dim={args.dim}  ops={args.ops}  mapper={args.mapper}")
    print("=" * 88)
    print()

    rows, comm, floors = [], [], []
    for sc in args.scenario:
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            if not os.path.exists(dp):
                print(f"  SKIP {sc} Re={re}"); continue
            dns = np.load(dp)
            vx = dns["vx"].astype(np.float64); vy = dns["vy"].astype(np.float64)
            Bx = dns["Bx"].astype(np.float64); By = dns["By"].astype(np.float64)
            sel = sorted(set(int(round(i)) for i in np.linspace(
                0, len(vx) - 1, args.n_snaps + 1)[1:]))
            for si in sel:
                f = (vx[si], vy[si], Bx[si], By[si])
                base = decision_maps(*f, args.N, args.dim, re,
                                     use_v2=(args.mapper == "v2"),
                                     reps=args.reps, k_opt=args.k_opt,
                                     run_qaoa=not args.no_qaoa,
                                     gs_solver=args.gs_solver, seed=args.seed)
                fl, fr = solver_noise_floor(
                    *f, args.N, args.dim, re, seeds=(0, 1, 2),
                    use_v2=(args.mapper == "v2"), reps=args.reps,
                    k_opt=args.k_opt, run_qaoa=False,
                    gs_solver=args.gs_solver)
                floors.append(dict(scenario=sc, snap=si, floor=fl,
                                   frac_spread=float(np.ptp(fr))))
                for op in args.ops:
                    for axial in (True, False):
                        eps = solver_commutation_defect(
                            *f, args.N, re, op, axial_B=axial)
                        comm.append(dict(scenario=sc, op=op, axial=axial,
                                         eps=eps))
                    # convention retenue : celle qui commute le mieux
                    e_ax = [c["eps"] for c in comm
                            if c["scenario"] == sc and c["op"] == op][-2:]
                    axial_best = bool(e_ax[0] <= e_ax[1])
                    tf = apply_symmetry(*f, op=op, axial_B=axial_best)
                    trans = decision_maps(*tf, args.N, args.dim, re,
                                          use_v2=(args.mapper == "v2"),
                                          reps=args.reps, k_opt=args.k_opt,
                                          run_qaoa=not args.no_qaoa,
                                          gs_solver=args.gs_solver,
                                          seed=args.seed)
                    for route in base:
                        rows.append(dict(
                            scenario=sc, re=re, snap=si, op=op, route=route,
                            eps_orbit=orbit_error(
                                trans[route],
                                apply_symmetry_mask(base[route], op)),
                            eps_solver=min(e_ax), axial=axial_best))
                print(f"  {sc:<18} Re={re} snap={si} done")

    if not rows:
        # Same failure mode as elsewhere in study/ (D-56): silently
        # exiting 0 without writing an artifact would leave the previous
        # campaign's file in place, indistinguishable from a fresh,
        # successful run.
        raise RuntimeError(
            "balayage vide : aucune orbites d'equivariance n'a d'artefact d'entree pour les "
            "arguments donnes. Le script sortait ici avec le code 0 et sans "
            "artefact, donc sans se distinguer d'une campagne reussie.")

    print("\n  [1] solver commutation defect  eps = ||T(step(U)) - step(T(U))||"
          " / ||step(U)||")
    print(f"  {'op':<8} {'axial B':>9} {'mean eps':>12} {'max eps':>12}")
    for op in args.ops:
        for axial in (True, False):
            es = [c["eps"] for c in comm if c["op"] == op and c["axial"] == axial]
            if es:
                print(f"  {op:<8} {str(axial):>9} {np.mean(es):>12.3e} "
                      f"{np.max(es):>12.3e}")

    routes = list(dict.fromkeys(r["route"] for r in rows))
    print("\n  [2] orbit error of the decision map "
          "(fraction of patches where D(T(U)) != T(D(U)))")
    head = f"  {'op':<8}" + "".join(f"{r:>16}" for r in routes)
    print(head); print("  " + "-" * (len(head) - 2))
    for op in args.ops:
        cells = ""
        for route in routes:
            es = [r["eps_orbit"] for r in rows
                  if r["op"] == op and r["route"] == route]
            cells += f"{np.mean(es):>16.4f}" if es else f"{'-':>16}"
        print(f"  {op:<8}{cells}")
    print("  " + "-" * (len(head) - 2))
    cells = ""
    for route in routes:
        es = [r["eps_orbit"] for r in rows if r["route"] == route]
        cells += f"{np.mean(es):>16.4f}"
    print(f"  {'MEAN':<8}{cells}")

    if floors:
        mf = float(np.mean([x["floor"] for x in floors]))
        ms = float(np.mean([x["frac_spread"] for x in floors]))
        print(f"\n  [3] CONTROL - ground-state solver noise floor "
              f"(same field, different anneal seeds): {mf:.4f}")
        print(f"      spread of refined fraction across seeds: {ms:.4f}")
        gs_orbit = float(np.mean([r["eps_orbit"] for r in rows
                                  if r["route"] == "ground_state"])) \
            if any(r["route"] == "ground_state" for r in rows) else float("nan")
        print(f"      ground-state orbit error {gs_orbit:.4f} vs floor {mf:.4f}"
              f" -> " + ("NOT interpretable: the optimiser is less "
                         "reproducible than the effect being measured"
                         if gs_orbit <= 2.0 * mf else
                         "clearly above the floor; residual is "
                         "non-equivariance of the decision map"))
    print("\n  READING: a solver defect at machine precision with a non-zero "
          "orbit error\n  localises the asymmetry in the decision map, not in "
          "the physics.")

    out = os.path.join(RESULTS_DIR,
                       f"t12_equivariance_N{args.N}_dim{args.dim}"
        + ("" if args.mapper == "v2" else f"_{args.mapper}")
        + ".npz")
    np.savez_compressed(
        out,
        scenario=np.array([r["scenario"] for r in rows]),
        op=np.array([r["op"] for r in rows]),
        route=np.array([r["route"] for r in rows]),
        snap=np.array([r["snap"] for r in rows]),
        eps_orbit=np.array([r["eps_orbit"] for r in rows]),
        eps_solver=np.array([r["eps_solver"] for r in rows]),
        axial=np.array([r["axial"] for r in rows]),
        comm_op=np.array([c["op"] for c in comm]),
        comm_axial=np.array([c["axial"] for c in comm]),
        comm_eps=np.array([c["eps"] for c in comm]),
        floor=np.array([x["floor"] for x in floors]),
        floor_frac_spread=np.array([x["frac_spread"] for x in floors]),
        seed=args.seed, git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nV4 Task 12 complete.")


if __name__ == "__main__":
    main()
