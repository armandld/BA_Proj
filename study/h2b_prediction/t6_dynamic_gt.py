#!/usr/bin/env python3
"""
V3 Task 6 - Verite terrain dynamique d_i (protocole v3, section 1.2).

Pour un snapshot t et un patch i : on construit une variante du champ ou
le patch i SEUL est moyenne par bloc a la resolution patch puis prolonge
(logique `coarsen_field` de phase 2 restreinte a la fenetre du patch :
la fenetre devient sa moyenne), tous les autres patches restant fins.
On evolue la reference ET la variante avec `MHDSolver.step_full` pendant
delta_t = 0.1 (un pas hybride), avec la MEME sequence de dt CFL :
la sequence est enregistree pendant l'evolution de reference
(`adapt_dt(cfl_target)`, dernier pas tronque pour atterrir sur delta_t)
puis rejouee a l'identique pour chaque variante.

d_i = difference L2 relative des champs COMPLETS a t+delta_t,
normalisee comme en phase 2 : sqrt(mean(somme des diff^2 des 4 champs))
/ RMS global des champs de reference a t+delta_t.

PILOTE D'ABORD (obligatoire, section 8.4) : par defaut la commande nue
execute le pilote N=128 (sous-echantillonne depuis le DNS N=256 via
--source-N), une config (orszag_tang, Re=400), 2 snapshots, les 16
patches ; le wall-clock est mesure et le cout complet projete AVANT
tout lancement N=256 (<= 10 snapshots/config).

Sortie : results/d_patches_{sc}_Re{re}_N{N}_dim{D}.npz
  - memes cles que phase 2 (l2_errors, is_hard, l2_threshold, t,
    scenario, Re, N, n_patches) pour etre une source de labels drop-in
    pour les builders de phase 11. ATTENTION : l2_errors est de
    longueur DNS complete avec NaN aux snapshots non calcules ;
    `computed_mask` / `snap_indices` donnent l'alignement (Task 7).
  - extras : d_computed, e_static (e_i de phase 2 recalcule sur les
    memes snapshots), spearman_d_e, delta_t, cfl, n_steps,
    wallclock_per_snap, hash git, arguments CLI.

Usage :
  python study/v3/t6_dynamic_gt.py                      # pilote N=128
  python study/v3/t6_dynamic_gt.py --N 256 --n-snaps 10 \
      --scenario orszag_tang harris_tearing ... --re 400 800 1200 1600
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

FIELDS = ("vx", "vy", "Bx", "By")


# -------------------------------------------------------------------
# Helpers purs
# -------------------------------------------------------------------

def coarsen_patch_window(field, i0, i1, j0, j1):
    """Copie du champ ou la fenetre [i0:i1, j0:j1] est remplacee par sa
    moyenne (coarsen_field de phase 2, facteur = taille de fenetre,
    restreint a un seul bloc : moyenne puis prolongation constante)."""
    out = np.array(field, copy=True)
    out[i0:i1, j0:j1] = field[i0:i1, j0:j1].mean()
    return out


def coarsen_one_patch(fields, pi, pj, patch_size):
    """Variante (vx, vy, Bx, By) avec le patch (pi, pj) seul coarseni."""
    i0, i1 = pi * patch_size, (pi + 1) * patch_size
    j0, j1 = pj * patch_size, (pj + 1) * patch_size
    return tuple(coarsen_patch_window(f, i0, i1, j0, j1) for f in fields)


def rel_l2_diff(ref_fields, var_fields):
    """Normalisation phase 2 : sqrt(mean(somme diff^2 des 4 champs))
    / RMS global des champs de reference."""
    diff_sq = sum((a - b) ** 2 for a, b in zip(ref_fields, var_fields))
    rms = np.sqrt(np.mean(sum(f ** 2 for f in ref_fields)))
    if rms < 1e-15:
        rms = 1.0
    return float(np.sqrt(np.mean(diff_sq)) / rms)


def downsample_fields(arr, factor):
    """Moyenne par blocs (n, N, N) -> (n, N/factor, N/factor)."""
    n, N, _ = arr.shape
    m = N // factor
    return arr.reshape(n, m, factor, m, factor).mean(axis=(2, 4))


# -------------------------------------------------------------------
# Evolution (solveur V1, importe — jamais re-implemente)
# -------------------------------------------------------------------

def _make_sim(fields, N, re, dt0=1e-4):
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver
    sim = MHDSolver(PeriodicGrid(N), dt=dt0, Re=re, Rm=re)
    sim.vx, sim.vy, sim.Bx, sim.By = [np.array(f, copy=True)
                                      for f in fields]
    return sim


def reference_evolution(fields, N, re, delta_t, cfl):
    """Evolution de reference : sequence de dt CFL enregistree
    (dernier pas tronque pour atterrir exactement sur delta_t)."""
    sim = _make_sim(fields, N, re)
    dts, t = [], 0.0
    while t < delta_t - 1e-12:
        sim.adapt_dt(cfl_target=cfl)
        dt = min(sim.dt, delta_t - t)
        sim.dt = dt
        sim.step_full(record_stats=False)
        dts.append(dt)
        t += dt
    return (sim.vx, sim.vy, sim.Bx, sim.By), dts


def replay_evolution(fields, N, re, dts):
    """Rejoue exactement la sequence de dt de la reference."""
    sim = _make_sim(fields, N, re)
    for dt in dts:
        sim.dt = dt
        sim.step_full(record_stats=False)
    return (sim.vx, sim.vy, sim.Bx, sim.By)


def dynamic_gt_snapshot(fields, N, dim, re, delta_t, cfl):
    """d_i pour les dim^2 patches d'un snapshot.
    Retourne (d (dim, dim), n_steps)."""
    patch_size = N // dim
    ref_T, dts = reference_evolution(fields, N, re, delta_t, cfl)
    d = np.zeros((dim, dim))
    for pi in range(dim):
        for pj in range(dim):
            var = coarsen_one_patch(fields, pi, pj, patch_size)
            var_T = replay_evolution(var, N, re, dts)
            d[pi, pj] = rel_l2_diff(ref_T, var_T)
    return d, len(dts)


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

def select_snapshots(n_dns, n_snaps):
    """Sous-echantillonnage uniforme en excluant la condition initiale."""
    idx = np.linspace(0, n_dns - 1, n_snaps + 1)[1:]
    return sorted(set(int(round(i)) for i in idx))


def main():
    p = argparse.ArgumentParser(
        description="V3 Task 6: dynamic ground truth d_i (pilot first)")
    from config import RESULTS_DIR, L2_PERCENTILE_HARD

    p.add_argument("--scenario", nargs="+", default=["orszag_tang"])
    p.add_argument("--re", nargs="+", type=int, default=[400])
    p.add_argument("--N", type=int, default=128,
                   help="resolution d'evolution (pilote: 128)")
    p.add_argument("--source-N", type=int, default=256,
                   help="si dns_N absent, sous-echantillonne depuis ce N")
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--n-snaps", type=int, default=2,
                   help="snapshots par config (pilote: 2; complet: <=10)")
    p.add_argument("--delta-t", type=float, default=0.1,
                   help="un pas hybride V1 (section 1.2)")
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0,
                   help="enregistre (pipeline deterministe, pas de RNG)")
    args = p.parse_args()

    from phase2_hard_patches import patch_l2_errors

    print("=" * 88)
    print("  V3 Task 6: dynamic ground truth d_i "
          f"(delta_t={args.delta_t}, cfl={args.cfl})")
    print(f"  N={args.N}  dim={args.dim}  n-snaps/cfg={args.n_snaps}  "
          f"scenarios={args.scenario}  Re={args.re}")
    print("=" * 88)
    print()

    per_snap_times = []
    all_d, all_e = [], []

    for sc in args.scenario:
        for re in args.re:
            dp = os.path.join(RESULTS_DIR,
                              f"dns_{sc}_Re{re}_N{args.N}.npz")
            factor = 1
            if not os.path.exists(dp) and args.source_N != args.N:
                src = os.path.join(
                    RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.source_N}.npz")
                if os.path.exists(src):
                    dp = src
                    factor = args.source_N // args.N
                    print(f"  [{sc} Re={re}] N={args.N} absent -> "
                          f"sous-echantillonnage x{factor} depuis "
                          f"N={args.source_N}")
            if not os.path.exists(dp):
                print(f"  SKIP {sc} Re={re}: no DNS input")
                continue

            dns = np.load(dp)
            t_all = dns["t"]
            raw = {f: dns[f].astype(np.float64) for f in FIELDS}
            if factor > 1:
                raw = {f: downsample_fields(raw[f], factor)
                       for f in FIELDS}
            n_dns = len(t_all)
            sel = select_snapshots(n_dns, args.n_snaps)
            print(f"  [{sc} Re={re}] {n_dns} DNS snaps -> computing "
                  f"d_i on snapshots {sel}")

            d_full = np.full((n_dns, args.dim, args.dim), np.nan)
            e_sel, d_sel, n_steps_sel = [], [], []
            for si in sel:
                fields = tuple(raw[f][si] for f in FIELDS)
                t0 = time.time()
                d, n_steps = dynamic_gt_snapshot(
                    fields, args.N, args.dim, re,
                    args.delta_t, args.cfl)
                dt_wall = time.time() - t0
                per_snap_times.append(dt_wall)
                e = patch_l2_errors(*fields, args.dim)
                d_full[si] = d
                d_sel.append(d); e_sel.append(e)
                n_steps_sel.append(n_steps)
                print(f"    snap {si:>4} (t={t_all[si]:.2f}): "
                      f"{n_steps} solver steps x {1 + args.dim ** 2} "
                      f"evolutions  d in [{d.min():.2e}, {d.max():.2e}]"
                      f"  [{dt_wall:.1f}s]")

            d_sel = np.array(d_sel); e_sel = np.array(e_sel)
            all_d.append(d_sel.ravel()); all_e.append(e_sel.ravel())

            thr = float(np.percentile(d_sel.ravel(),
                                      L2_PERCENTILE_HARD))
            mask = np.zeros(n_dns, dtype=bool); mask[sel] = True
            out = os.path.join(
                RESULTS_DIR,
                f"d_patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            np.savez_compressed(
                out,
                # cles phase 2 (drop-in ; NaN hors snapshots calcules)
                l2_errors=d_full,
                is_hard=np.where(np.isnan(d_full), False,
                                 d_full >= thr),
                l2_threshold=thr,
                t=t_all, scenario=sc, Re=re, N=args.N,
                n_patches=args.dim,
                # alignement + extras v3
                computed_mask=mask,
                snap_indices=np.array(sel),
                d_computed=d_sel, e_static=e_sel,
                n_steps=np.array(n_steps_sel),
                delta_t=args.delta_t, cfl=args.cfl,
                wallclock_per_snap=np.array(per_snap_times[-len(sel):]),
                seed=args.seed,
                git_hash=git_commit_hash(),
                cli_args=json.dumps(vars(args)),
            )
            print(f"    saved: {os.path.basename(out)}")

    if not all_d:
        print("no output."); return

    # ---- sanity check d'acceptation : Spearman(d_i, e_i) > 0 ----
    from metrics import spearman
    rho = spearman(np.concatenate(all_d), np.concatenate(all_e))
    n_pairs = sum(len(x) for x in all_d)
    print(f"\n  sanity check: Spearman(d_i, e_i) = {rho:.3f}  "
          f"(n={n_pairs} patches)  -> {'PASS' if rho > 0 else 'FAIL'} "
          "(accept: > 0)")

    # ---- wall-clock + projection du cout complet ----
    per_snap = float(np.mean(per_snap_times))
    scale = (256 / args.N) ** 3   # cellules x(256/N)^2, pas CFL x(256/N)
    n_cfg_full, n_snap_full = 16, 10
    proj = per_snap * scale * n_cfg_full * n_snap_full
    print(f"\n  wall-clock: {per_snap:.1f}s per snapshot at N={args.N} "
          f"({1 + args.dim ** 2} evolutions of delta_t={args.delta_t})")
    print(f"  projection full campaign (N=256, {n_cfg_full} configs x "
          f"{n_snap_full} snaps, scaling x(256/N)^3 = x{scale:.0f}): "
          f"{proj / 3600:.1f} h  ({proj / 60:.0f} min)")
    print("\nV3 Task 6 complete.")


if __name__ == "__main__":
    main()
