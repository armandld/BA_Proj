#!/usr/bin/env python3
"""
V4 Task 14 - Validation numerique du solveur (audit, Priorite 1).

L'audit demande trois elements que le manuscrit ne fournit pas encore :
convergence en maillage avec controle du pas de temps, lois de conservation
et divergence magnetique suivies pendant toute la trajectoire, et variation
de Re/Rm hors de la grille d'entrainement. Les trois sont mesures ici sur le
solveur V1 (`MHDSolver`, importe, jamais reimplemente).

(A) AUTO-CONVERGENCE. Le meme scenario est integre jusqu'a un temps commun a
    plusieurs resolutions N. Toutes les solutions sont ramenees a la grille
    la plus grossiere par moyenne de blocs (l'operateur de coarsening de la
    phase 2), puis comparees deux a deux en norme L2 relative. L'ordre
    observe est estime par
        ordre = log2( ||u_N - u_2N|| / ||u_2N - u_4N|| ).
    Le pas de temps est pilote par la meme cible CFL a toutes les
    resolutions, de sorte que le raffinement est spatio-temporel.

(B) CONSERVATION ET CONTRAINTE SOLENOIDALE. Sur chaque trajectoire on suit
    l'energie totale (decroissance attendue pour un systeme dissipatif non
    force), sa monotonie, et max|div B| / rms|B| via l'operateur FD4 du
    solveur — celui qui construit `rhs_B` et donc celui qui garantit la
    contrainte depuis D-25 (voir `div_B_matched`, D-72). L'operateur
    spectral de la phase 1b mesurerait l'ecart entre deux stencils, pas la
    contrainte.

(C) HORS GRILLE D'ENTRAINEMENT. Les memes diagnostics sont evalues a des
    Re/Rm situes en dehors de {400, 800, 1200, 1600}.

Sortie : results/t14_numerical_validation.npz
Usage :
  python study/h1_solver/h1_solver_convergence.py --scenario orszag_tang \
      --grids 32 64 128 --t-end 0.5 --re-out 200 3200
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
from h2b_dynamic_ground_truth import downsample_fields          # v3, reutilise
from dns_validation import total_energy   # V2, reutilise

FIELDS = ("vx", "vy", "Bx", "By")


def div_B_matched(Bx, By, dx):
    """Divergence de B avec l'operateur qui la GARANTIT — pas un autre.

    D-72. Ce diagnostic utilisait `dns_validation.div_B`, qui est SPECTRALE.
    Sa docstring dit pourquoi c'etait juste quand elle a ete ecrite : « same
    convention as the solver's FFT projection ». Ce n'est plus la convention
    du solveur depuis D-25 : `MHDSolver.PROJECT_B = False`, B n'est plus
    projete spectralement.

    Ce que le solveur garantit aujourd'hui est une divergence nulle AUX
    DIFFERENCES FINIES : l'induction est en forme rotationnelle
    `rhs_B = (dEz/dy, -dEz/dx)`, dont la divergence FD4 vaut
    `d2Ez/dxdy - d2Ez/dydx`, exactement nulle puisque les decalages de
    `np.roll` commutent. B est solenoidal par construction, dans l'operateur
    meme qui construit le second membre.

    Mesurer ce champ au spectral ne mesure donc pas la contrainte : cela
    mesure l'ecart entre les deux operateurs. Mesure sur la configuration
    publiee de T14 (orszag_tang, grilles 32/64/128, t_end=0.5, Re=400 puis
    200/3200), rejouee a HEAD :

      max|div B|/rms|B|, spectral (avant)   3.9029e-02   -> ALL CHECKS False
      max|div B|/rms|B|, FD4 assorti (apres) 2.0266e-14  -> ALL CHECKS True

    contre un seuil d'acceptation de 1e-3, et une valeur publiee « entre
    5.6e-15 et 8.0e-14 — machine precision ». Le faux signal croissait quand
    la grille grossissait (N=128 2.3103e-04, N=64 4.5675e-03, N=32
    3.9029e-02) : la validation passait ou echouait selon la RESOLUTION,
    pour une contrainte respectee a 1e-14 partout.

    Le stencil n'est pas reimplemente : `_fd_grad` est celui de V1, celui-la
    meme qui assemble `rhs_Bx`/`rhs_By` dans `_compute_rhs_fd`.
    """
    from Simulation.solver import MHDSolver
    g_Bx_x, _ = MHDSolver._fd_grad(Bx, dx)
    _, g_By_y = MHDSolver._fd_grad(By, dx)
    return g_Bx_x + g_By_y


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def evolve_to(scenario, N, re, t_end, cfl=0.4, dt0=1e-3, record=None):
    """Integre le scenario jusqu'a t_end avec le solveur V1.

    `record` : liste optionnelle remplie de diagnostics par pas
    (t, energie, max|div B| / rms|B|). Retourne (vx, vy, Bx, By, t, n_steps).
    """
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver
    from dns_extension import _extended_init        # v3, 8 scenarios

    sim = MHDSolver(PeriodicGrid(N), dt=dt0, Re=re, Rm=re)
    _extended_init(sim, scenario, seed=0, amplitude=0.0)
    dx = 2 * np.pi / N
    t, k = 0.0, 0
    while t < t_end - 1e-12:
        sim.dt = min(sim.adapt_dt(cfl_target=cfl), t_end - t)
        sim.step_full(record_stats=False)
        t += sim.dt
        k += 1
        if record is not None:
            rms_B = np.sqrt((sim.Bx ** 2 + sim.By ** 2).mean()) + 1e-30
            record.append(dict(
                t=t,
                E=float(total_energy(sim.vx, sim.vy, sim.Bx, sim.By)),
                divB=float(np.abs(div_B_matched(sim.Bx, sim.By, dx)).max()
                           / rms_B)))
    return sim.vx, sim.vy, sim.Bx, sim.By, t, k


def to_common_grid(fields, N_target):
    """Ramene un jeu de champs a la grille N_target par moyenne de blocs.

    Reutilise `downsample_fields` (tache 6) en ajoutant l'axe snapshot
    attendu par cette fonction.
    """
    N = fields[0].shape[0]
    if N == N_target:
        return tuple(np.asarray(f, dtype=float) for f in fields)
    if N % N_target:
        raise ValueError(f"{N} not divisible by {N_target}")
    factor = N // N_target
    return tuple(downsample_fields(np.asarray(f, dtype=float)[None, ...],
                                   factor)[0] for f in fields)


def relative_l2(a, b):
    """Norme L2 relative entre deux jeux de quatre champs."""
    num = np.sqrt(sum(np.sum((x - y) ** 2) for x, y in zip(a, b)))
    den = np.sqrt(sum(np.sum(y ** 2) for y in b)) + 1e-30
    return float(num / den)


def observed_order(err_coarse, err_fine):
    """Ordre de convergence observe entre deux ecarts successifs."""
    if err_fine <= 0 or err_coarse <= 0:
        return float("nan")
    return float(np.log2(err_coarse / err_fine))


def splitting_order_diagnostic(scenario="orszag_tang", N=64, re=400,
                               T=0.2, steps=(16, 32, 64, 128), ref_steps=512):
    """Convergence TEMPORELLE a pas fixe, avec et sans la projection.

    `MHDSolver.step_full` enchaine un pas RK4 complet PUIS la projection a
    divergence nulle. Ce decouplage est un splitting de Lie, formellement
    d'ordre 1, qui borne la precision temporelle du schema complet quel que
    soit l'ordre du noyau RK4. Le diagnostic separe les deux effets en
    rejouant la meme integration avec et sans l'etape de projection : si
    l'ordre passe de ~4 (sans) a ~1 (avec), le splitting est bien la source.

    Les deux branches utilisent `_rk4_step` et `enforce_incompressibility`
    de V1, sans reimplementation.

    Retourne dict(with_projection=[(n, err, order)...], without=[...]).
    """
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver
    from dns_extension import _extended_init

    def _run(n_steps, project):
        sim = MHDSolver(PeriodicGrid(N), dt=T / n_steps, Re=re, Rm=re)
        _extended_init(sim, scenario, seed=0, amplitude=0.0)
        sim.dt = T / n_steps
        for _ in range(n_steps):
            sim.vx, sim.vy, sim.Bx, sim.By = sim._rk4_step(
                sim.vx, sim.vy, sim.Bx, sim.By, sim.dx, sim.dt)
            if project:
                sim.enforce_incompressibility()
        return (sim.vx, sim.vy, sim.Bx, sim.By)

    out = {}
    for project, key in ((True, "with_projection"), (False, "without")):
        ref = _run(ref_steps, project)
        rows, prev = [], None
        for n in steps:
            err = relative_l2(_run(n, project), ref)
            rows.append((n, err,
                         observed_order(prev, err) if prev else float("nan")))
            prev = err
        out[key] = rows
    return out


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="V4 Task 14: numerical validation of the V1 solver")
    from config import RESULTS_DIR

    p.add_argument("--scenario", nargs="+", default=["orszag_tang"])
    p.add_argument("--grids", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--t-end", type=float, default=0.5)
    p.add_argument("--re", type=int, default=400)
    p.add_argument("--re-out", nargs="+", type=int, default=[200, 3200])
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--skip-splitting", action="store_true")
    p.add_argument("--split-N", type=int, default=None,
                   help="resolution du diagnostic de splitting "
                        "(defaut : la grille la plus fine)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  V4 Task 14: grid convergence, conservation, out-of-grid Re")
    print(f"  scenarios={args.scenario}  grids={args.grids}  "
          f"t_end={args.t_end}  CFL={args.cfl}")
    print("=" * 88)

    conv_rows, cons_rows = [], []
    grids = sorted(args.grids)

    # ---------- (A) auto-convergence ----------
    print("\n  [A] self-convergence (all solutions compared on the coarsest "
          "grid)")
    for sc in args.scenario:
        sols = {}
        for N in grids:
            t0 = time.time()
            rec = []
            *f, t_fin, k = evolve_to(sc, N, args.re, args.t_end,
                                     cfl=args.cfl, record=rec)
            sols[N] = tuple(f)
            E = np.array([r["E"] for r in rec])
            dv = np.array([r["divB"] for r in rec])
            mono = bool(np.all(np.diff(E) <= 1e-3))
            cons_rows.append(dict(
                scenario=sc, N=N, re=args.re, n_steps=k,
                E0=float(E[0]) if len(E) else np.nan,
                E_end=float(E[-1]) if len(E) else np.nan,
                E_drop=float(1 - E[-1] / E[0]) if len(E) else np.nan,
                monotone=mono, divB_max=float(dv.max()) if len(dv) else np.nan,
                wall=time.time() - t0))
            print(f"    {sc:<16} N={N:<5} steps={k:<6} "
                  f"[{time.time()-t0:.0f}s]")
        errs = []
        for a, b in zip(grids[:-1], grids[1:]):
            e = relative_l2(to_common_grid(sols[a], grids[0]),
                            to_common_grid(sols[b], grids[0]))
            errs.append(e)
            conv_rows.append(dict(scenario=sc, coarse=a, fine=b, err=e))
        print(f"    {'pair':<16} {'||u_N - u_2N||_rel':>20}")
        for (a, b), e in zip(zip(grids[:-1], grids[1:]), errs):
            print(f"    {f'{a} vs {b}':<16} {e:>20.4e}")
        for i in range(len(errs) - 1):
            print(f"    observed order between the two gaps: "
                  f"{observed_order(errs[i], errs[i+1]):.2f}")

    # ---------- (B)/(C) conservation, dont hors grille ----------
    print("\n  [B] conservation and solenoidal constraint along the trajectory")
    print(f"  {'scenario':<16} {'N':>5} {'Re':>6} {'E(0)':>8} {'E drop':>9} "
          f"{'E monotone':>11} {'max|divB|/rmsB':>16}")
    for r in cons_rows:
        print(f"  {r['scenario']:<16} {r['N']:>5} {r['re']:>6} "
              f"{r['E0']:>8.3f} {r['E_drop']*100:>8.1f}% "
              f"{str(r['monotone']):>11} {r['divB_max']:>16.2e}")

    print("\n  [C] Reynolds numbers outside the training grid "
          "{400, 800, 1200, 1600}")
    N_out = grids[min(1, len(grids) - 1)]
    for sc in args.scenario:
        for re in args.re_out:
            rec = []
            t0 = time.time()
            *_, k = evolve_to(sc, N_out, re, args.t_end, cfl=args.cfl,
                              record=rec)
            E = np.array([r["E"] for r in rec])
            dv = np.array([r["divB"] for r in rec])
            mono = bool(np.all(np.diff(E) <= 1e-3))
            cons_rows.append(dict(
                scenario=sc, N=N_out, re=re, n_steps=k,
                E0=float(E[0]), E_end=float(E[-1]),
                E_drop=float(1 - E[-1] / E[0]), monotone=mono,
                divB_max=float(dv.max()), wall=time.time() - t0))
            print(f"  {sc:<16} N={N_out:<5} Re={re:<6} "
                  f"E drop={100*(1-E[-1]/E[0]):>5.1f}%  monotone={mono}  "
                  f"max|divB|/rmsB={dv.max():.2e}  [{time.time()-t0:.0f}s]")

    split = None
    if not args.skip_splitting:
        print("\n  [D] temporal order with and without the divergence-free "
              "projection\n      (step_full applies RK4 then projection: a "
              "first-order Lie splitting)")
        split_N = args.split_N or grids[-1]
        print(f"      resolution N={split_N}")
        split = splitting_order_diagnostic(scenario=args.scenario[0],
                                           N=split_N, re=args.re)
        for key in ("with_projection", "without"):
            print(f"    {key}:")
            for n, err, o in split[key]:
                print(f"      steps={n:>4}  err={err:.4e}  order="
                      + (f"{o:5.2f}" if o == o else "  n/a"))
        ords = [o for _, _, o in split["with_projection"] if o == o]
        ords0 = [o for _, _, o in split["without"] if o == o]
        print(f"    observed order: with projection "
              f"{np.mean(ords):.2f} | without {np.mean(ords0):.2f}")

    ok = all(r["divB_max"] <= 1e-3 for r in cons_rows) and \
        all(r["monotone"] for r in cons_rows)
    print(f"\n  ALL CHECKS (divB <= 1e-3 and monotone energy): {ok}")

    out = os.path.join(RESULTS_DIR, "t14_numerical_validation.npz")
    np.savez_compressed(
        out,
        conv_scenario=np.array([r["scenario"] for r in conv_rows]),
        conv_coarse=np.array([r["coarse"] for r in conv_rows]),
        conv_fine=np.array([r["fine"] for r in conv_rows]),
        conv_err=np.array([r["err"] for r in conv_rows]),
        cons_scenario=np.array([r["scenario"] for r in cons_rows]),
        cons_N=np.array([r["N"] for r in cons_rows]),
        cons_re=np.array([r["re"] for r in cons_rows]),
        cons_E0=np.array([r["E0"] for r in cons_rows]),
        cons_Edrop=np.array([r["E_drop"] for r in cons_rows]),
        cons_monotone=np.array([r["monotone"] for r in cons_rows]),
        cons_divB=np.array([r["divB_max"] for r in cons_rows]),
        all_checks_pass=bool(ok),
        split_with=np.array([[n, e, o] for n, e, o in split["with_projection"]]
                            ) if split else np.zeros((0, 3)),
        split_without=np.array([[n, e, o] for n, e, o in split["without"]]
                               ) if split else np.zeros((0, 3)),
        t_end=args.t_end, cfl=args.cfl,
        seed=args.seed, git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"  saved: {os.path.basename(out)}")
    print("\nV4 Task 14 complete.")


if __name__ == "__main__":
    main()
