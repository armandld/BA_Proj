#!/usr/bin/env python3
"""Dynamic ground truth for the error caused by coarsening one patch.

Each patch is replaced by its mean and evolved with the exact CFL step
sequence used by the full-resolution reference. The reported distance is
measured over the whole domain after one patch-crossing time by default:

``t_x = patch_width / (v_rms + b_rms)``.

An explicit ``--delta-t`` remains available for horizon ablations. Artifact
keys distinguish the dynamic error, its instantaneous component, and the
static label; no key changes meaning between label types.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np

# --- chemins du depot (bloc unique, generé) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------
from config import RESULTS_DIR, L2_PERCENTILE_HARD          # noqa: E402
from hard_patch_labels import patch_l2_errors, patch_classical_scores  # noqa: E402
from Simulation.grid import PeriodicGrid                     # noqa: E402
from Simulation.solver import MHDSolver                      # noqa: E402

#: Default horizon: one patch-crossing time, computed per snapshot.
DELTA_T = None
CROSSING_MULTIPLE = 1.0
REDUNDANCY_RHO_LIMIT = 0.95
MIN_AMPLIFICATION_LOG_IQR = np.log(1.10)

#: Garde-fou : au-dela, la sequence de pas est trop longue pour un horizon
#: aussi court — signe que `adapt_dt` a rendu un pas absurde.
MAX_SUBSTEPS = 5000


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# -------------------------------------------------------------------
#  1. la variante : un patch remplace par sa moyenne
# -------------------------------------------------------------------
def coarsen_one_patch(field, pi, pj, patch_size):
    """Remplace le patch (pi, pj) par sa moyenne ; le reste est intact.

    C'est `coarsen_field(field, patch_size)` restreint a une fenetre : a la
    resolution des patches, un patch EST une cellule, donc le grossir revient
    a le remplacer par sa moyenne. Meme operation que celle dont `e_i` mesure
    l'ecart — les deux labels sont ainsi directement comparables.
    """
    out = field.copy()
    i0, i1 = pi * patch_size, (pi + 1) * patch_size
    j0, j1 = pj * patch_size, (pj + 1) * patch_size
    out[i0:i1, j0:j1] = field[i0:i1, j0:j1].mean()
    return out


def downsample_fields(array, factor):
    """Block-average an array shaped ``(snapshots, N, N)``."""
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"forme attendue (snapshots, N, N), recue {array.shape}")
    n_snapshots, nx, ny = array.shape
    if nx != ny or factor <= 0 or nx % factor:
        raise ValueError(
            f"grille {array.shape} incompatible avec le facteur {factor}")
    coarse = nx // factor
    return array.reshape(
        n_snapshots, coarse, factor, coarse, factor).mean(axis=(2, 4))


# -------------------------------------------------------------------
#  2. l'evolution, a sequence de pas imposee
# -------------------------------------------------------------------
def _solveur(N, Re, champs, dt):
    sim = MHDSolver(PeriodicGrid(N), dt=dt, Re=Re, Rm=Re)
    sim.vx, sim.vy, sim.Bx, sim.By = (np.array(c, dtype=float) for c in champs)
    return sim


def patch_crossing_time(vx, vy, Bx, By, n_patches):
    """Characteristic time for velocity/Alfven transport across one patch."""
    if not isinstance(n_patches, (int, np.integer)) or n_patches <= 0:
        raise ValueError(f"n_patches doit etre un entier positif, recu {n_patches!r}")
    v_rms = float(np.sqrt(np.mean(vx ** 2 + vy ** 2)))
    b_rms = float(np.sqrt(np.mean(Bx ** 2 + By ** 2)))
    speed = v_rms + b_rms
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError(
            "v_rms + b_rms doit etre fini et strictement positif pour "
            "definir le temps de traversee d'un patch")
    return (2.0 * np.pi / n_patches) / speed


def sequence_de_pas(N, Re, champs, delta_t, cfl_target=0.4):
    """Sequence de pas ADAPTATIVE, calculee une fois sur la reference.

    Rendue separement pour pouvoir etre rejouee a l'identique sur chaque
    variante : c'est la condition « same CFL dt sequence » du protocole. Le
    dernier pas est rogne pour tomber EXACTEMENT sur `delta_t`, sans quoi
    reference et variantes n'arriveraient pas au meme instant.
    """
    if not np.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError(f"delta_t doit etre fini et positif, recu {delta_t!r}")
    sim = _solveur(N, Re, champs, dt=1e-3)
    pas, t = [], 0.0
    while t < delta_t - 1e-15:
        dt = min(sim.adapt_dt(cfl_target=cfl_target), delta_t - t)
        if dt <= 0.0:
            raise RuntimeError(
                f"pas de temps nul ou negatif ({dt:.3e}) : `adapt_dt` a rendu "
                "une valeur inutilisable, le champ a probablement diverge")
        sim.dt = dt
        sim.step_full(record_stats=False)
        pas.append(dt)
        t += dt
        if len(pas) > MAX_SUBSTEPS:
            raise RuntimeError(
                f"plus de {MAX_SUBSTEPS} sous-pas pour delta_t={delta_t} : "
                "`adapt_dt` rend un pas absurde, refuser plutot que de tourner")
    return np.asarray(pas, dtype=float), (sim.vx, sim.vy, sim.Bx, sim.By)


def evolue(N, Re, champs, pas):
    """Rejoue la sequence `pas` telle quelle. Aucune adaptation."""
    sim = _solveur(N, Re, champs, dt=float(pas[0]))
    for dt in pas:
        sim.dt = float(dt)
        sim.step_full(record_stats=False)
    return sim.vx, sim.vy, sim.Bx, sim.By


# -------------------------------------------------------------------
#  3. le label
# -------------------------------------------------------------------
def _l2_relatif(a, b, rms):
    """||a - b||_2 sur le CHAMP ENTIER, normalise comme en phase 2."""
    diff = sum((x - y) ** 2 for x, y in zip(a, b))
    return float(np.sqrt(np.mean(diff)) / rms)


def dynamic_patch_errors(vx, vy, Bx, By, n_patches, Re, delta_t=DELTA_T,
                         crossing_multiple=CROSSING_MULTIPLE, verbose=False):
    """Rend `(d, d0, meta)` — le label dynamique, sa part instantanee, le cout.

    `d[pi, pj]`  : distance au champ de reference APRES `delta_t`.
    `d0[pi, pj]` : la meme distance AVANT toute evolution, c'est-a-dire la
                   perturbation de grossissement elle-meme. Vaut exactement
                   `e_i / n_patches` — identite epinglee par un test.
    """
    N = vx.shape[0]
    if N % n_patches:
        raise ValueError(f"N={N} n'est pas divisible par n_patches={n_patches}")
    patch_size = N // n_patches
    if patch_size < 2:
        raise ValueError(
            f"patch de {patch_size}x{patch_size} cellule(s) : a p=1 le "
            "grossissement est l'identite et tout label est nul. "
            f"Exiger dim <= N/8 (ici dim <= {N // 8}).")

    champs = tuple(np.asarray(c, dtype=float) for c in (vx, vy, Bx, By))
    crossing_time = patch_crossing_time(*champs, n_patches)
    if not np.isfinite(crossing_multiple) or crossing_multiple <= 0.0:
        raise ValueError("crossing_multiple doit etre fini et positif")
    if delta_t is None:
        delta_t = crossing_multiple * crossing_time
        horizon_mode = "patch_crossing"
    else:
        if not np.isfinite(delta_t) or delta_t <= 0.0:
            raise ValueError("delta_t doit etre fini et positif")
        horizon_mode = "fixed"
    rms = float(np.sqrt(np.mean(sum(c ** 2 for c in champs))))
    if rms < 1e-15:
        rms = 1.0

    t0 = time.time()
    pas, ref = sequence_de_pas(N, Re, champs, delta_t)
    t_ref = time.time() - t0
    if verbose:
        print(f"    reference : {len(pas)} sous-pas, dt in "
              f"[{pas.min():.2e}, {pas.max():.2e}], {t_ref:.1f} s")

    d = np.zeros((n_patches, n_patches))
    d0 = np.zeros((n_patches, n_patches))
    for pi in range(n_patches):
        for pj in range(n_patches):
            var = tuple(coarsen_one_patch(c, pi, pj, patch_size) for c in champs)
            d0[pi, pj] = _l2_relatif(var, champs, rms)
            d[pi, pj] = _l2_relatif(evolue(N, Re, var, pas), ref, rms)
        if verbose:
            print(f"    ligne {pi + 1}/{n_patches} faite "
                  f"({time.time() - t0:.1f} s)")

    meta = {
        "n_substeps": int(len(pas)),
        "dt_min": float(pas.min()), "dt_max": float(pas.max()),
        "delta_t": float(delta_t),
        "patch_crossing_time": float(crossing_time),
        "crossing_multiple": float(delta_t / crossing_time),
        "horizon_mode": horizon_mode,
        "wall_seconds": float(time.time() - t0),
        "wall_seconds_reference": float(t_ref),
        "rms_global": rms,
    }
    return d, d0, meta


# -------------------------------------------------------------------
#  4. un instantane, de bout en bout
# -------------------------------------------------------------------
def analyse_snapshot(dns_path, snap_index, n_patches, delta_t=DELTA_T,
                     percentile=L2_PERCENTILE_HARD,
                     crossing_multiple=CROSSING_MULTIPLE, verbose=False):
    z = np.load(dns_path, allow_pickle=True)
    N = int(z["meta_N"])
    Re = int(z["meta_Re"])
    vx, vy, Bx, By = (z[k][snap_index].astype(float)
                      for k in ("vx", "vy", "Bx", "By"))

    d, d0, meta = dynamic_patch_errors(vx, vy, Bx, By, n_patches, Re,
                                       delta_t,
                                       crossing_multiple=crossing_multiple,
                                       verbose=verbose)
    e = patch_l2_errors(vx, vy, Bx, By, n_patches)
    scores = patch_classical_scores(vx, vy, Bx, By, n_patches, 2 * np.pi / N)

    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.where(d0 > 0, d / d0, np.nan)

    meta.update({
        "scenario": str(z["meta_scenario"]), "Re": Re, "N": N,
        "n_patches": n_patches, "snap_index": int(snap_index),
        "t": float(z["t"][snap_index]),
    })
    # PAS de seuil ici : il se calcule sur TOUS les instantanes a la fois,
    # voir `seuil_global`. Un seuil par instantane forcerait exactement
    # (100-p) % de patches durs dans CHAQUE instantane, ce qui n'est pas ce
    # que fait la phase 2 et efface l'information « cet instantane est plus
    # actif que celui-la ».
    return {"d_errors": d, "d0_errors": d0, "amplification": amp,
            "l2_errors": e, "classical_scores": scores, "meta": meta}


def seuil_global(d_empile, percentile=L2_PERCENTILE_HARD):
    """UN seuil pour toute la serie, comme la phase 2.

    `hard_patch_labels.py` aplatit `all_l2` sur les instantanes AVANT de
    prendre son percentile, et rend un scalaire. Reproduire ce choix n'est
    pas cosmetique : un seuil par instantane fixerait la prevalence a
    (100-p) % dans chacun, si bien qu'un instantane calme et un instantane
    turbulent auraient la MEME proportion de patches durs. La comparabilite
    avec le label statique tomberait avec.
    """
    seuil = float(np.percentile(np.asarray(d_empile).ravel(), percentile))
    return seuil, (np.asarray(d_empile) >= seuil).astype(int)


def spearman(a, b):
    """rho de Spearman, sans dependance : rang puis Pearson."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra, rb = _rangs(a), _rangs(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def label_diagnostics(dynamic_error, static_error, amplification):
    """Diagnostics used to reject a dynamic label that repeats the static one."""
    finite_amp = np.asarray(amplification, float)
    finite_amp = finite_amp[np.isfinite(finite_amp) & (finite_amp > 0.0)]
    if finite_amp.size < 3:
        log_iqr = float("nan")
    else:
        q25, q75 = np.quantile(np.log(finite_amp), [0.25, 0.75])
        log_iqr = float(q75 - q25)
    rho = spearman(dynamic_error, static_error)
    informative = bool(
        np.isfinite(rho)
        and np.isfinite(log_iqr)
        and (rho < REDUNDANCY_RHO_LIMIT
             or log_iqr >= MIN_AMPLIFICATION_LOG_IQR)
    )
    return {
        "rho_d_vs_e": rho,
        "amplification_log_iqr": log_iqr,
        "informative": informative,
    }


def _rangs(x):
    ordre = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    r[ordre] = np.arange(x.size, dtype=float)
    # moyenne des rangs sur les ex aequo
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


# -------------------------------------------------------------------
#  5. pilote
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Phase 2 dynamique : verite terrain d_i (protocole §1.2)")
    ap.add_argument("--scenario", default="harris_tearing")
    ap.add_argument("--re", type=int, default=400)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--snaps", type=int, default=2,
                    help="nombre d'instantanes, repartis sur la trajectoire")
    ap.add_argument(
        "--delta-t", type=float, default=None,
        help="horizon fixe pour une ablation ; par defaut, un temps de traversee")
    ap.add_argument(
        "--crossing-multiple", type=float, default=CROSSING_MULTIPLE,
        help="multiple du temps de traversee utilise sans --delta-t")
    ap.add_argument("--percentile", type=float, default=L2_PERCENTILE_HARD)
    ap.add_argument("--seed", type=int, default=0,
                    help="aucun tirage aleatoire ici ; consigne pour la trace")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--allow-redundant", action="store_true",
        help="ecrire une ablation meme si le label reste equivalent au statique")
    ap.add_argument("--dry-run", action="store_true",
                    help="projette le cout sans calculer")
    args = ap.parse_args()

    dns = os.path.join(RESULTS_DIR,
                       f"dns_{args.scenario}_Re{args.re}_N{args.N}.npz")
    if not os.path.exists(dns):
        dispo = sorted(os.path.basename(p) for p in
                       glob.glob(os.path.join(RESULTS_DIR, "dns_*.npz")))
        raise SystemExit(
            f"artefact DNS absent : {os.path.basename(dns)}\n"
            f"  {len(dispo)} artefacts presents, par exemple : {dispo[:3]}")

    z = np.load(dns, allow_pickle=True)
    n_snap = len(z["t"])
    if args.snaps < 1:
        raise SystemExit("--snaps doit valoir au moins 1")
    idx = list(range(0, n_snap, max(1, n_snap // args.snaps)))[:args.snaps]
    if not idx:
        raise SystemExit(
            f"balayage vide : {n_snap} instantanes dans l'artefact et "
            f"--snaps {args.snaps} n'en selectionne aucun")

    p = args.N // args.dim
    if p < 8:
        print(f"  [ATTENTION] patch de {p}x{p} cellules : la contrainte "
              f"confortable est dim <= N/8, soit dim <= {args.N // 8}")

    if args.delta_t is not None and (
            not np.isfinite(args.delta_t) or args.delta_t <= 0.0):
        raise SystemExit("--delta-t doit etre fini et positif")
    if not np.isfinite(args.crossing_multiple) or args.crossing_multiple <= 0.0:
        raise SystemExit("--crossing-multiple doit etre fini et positif")

    selected_horizons = []
    for si in idx:
        fields = tuple(z[k][si].astype(float) for k in ("vx", "vy", "Bx", "By"))
        tx = patch_crossing_time(*fields, args.dim)
        selected_horizons.append(
            args.delta_t if args.delta_t is not None else args.crossing_multiple * tx)

    print(f"Verite terrain DYNAMIQUE — {args.scenario} Re={args.re} "
          f"N={args.N} dim={args.dim}")
    print(f"  {len(idx)} instantane(s) {idx}, {args.dim ** 2} patches chacun, "
          f"delta_t in [{min(selected_horizons):.4g}, "
          f"{max(selected_horizons):.4g}]")
    print(f"  soit {len(idx) * (args.dim ** 2 + 1)} evolutions de "
          f"l'horizon indique a N={args.N}")
    if args.dry_run:
        return

    res, t0 = [], time.time()
    for k, si in enumerate(idx):
        print(f"  instantane {k + 1}/{len(idx)} (index {si}, "
              f"t={float(z['t'][si]):.3f})")
        r = analyse_snapshot(dns, si, args.dim, args.delta_t,
                             args.percentile,
                             crossing_multiple=args.crossing_multiple,
                             verbose=True)
        diagnostic = label_diagnostics(
            r["d_errors"], r["l2_errors"], r["amplification"])
        fini = np.isfinite(r["amplification"])
        print(f"    rho(d, e) = {diagnostic['rho_d_vs_e']:+.4f} | "
              f"amplification mediane = "
              f"{np.median(r['amplification'][fini]):.2f}x | "
              f"log-IQR={diagnostic['amplification_log_iqr']:.3f} | "
              f"{'INFORMATIF' if diagnostic['informative'] else 'REDONDANT'} | "
              f"{r['meta']['wall_seconds']:.1f} s")
        r["diagnostic"] = diagnostic
        res.append(r)

    if (not any(r["diagnostic"]["informative"] for r in res)
            and not args.allow_redundant):
        raise RuntimeError(
            "label dynamique redondant : aucun instantane ne satisfait "
            f"rho(d,e) < {REDUNDANCY_RHO_LIMIT:g} ou log-IQR(amplification) "
            f">= {MIN_AMPLIFICATION_LOG_IQR:.3f}. Augmenter "
            "--crossing-multiple ou utiliser --delta-t pour une ablation ; "
            "aucun artefact n'a ete ecrit.")

    horizon_tag = (f"dt{args.delta_t:g}" if args.delta_t is not None
                   else f"tx{args.crossing_multiple:g}")
    out = args.out or os.path.join(
        RESULTS_DIR,
        f"d_patches_{args.scenario}_Re{args.re}_N{args.N}"
        f"_dim{args.dim}_{horizon_tag}.npz")
    empile = lambda k: np.stack([r[k] for r in res])          # noqa: E731
    d_tous = empile("d_errors")
    seuil, dur = seuil_global(d_tous, args.percentile)
    frac = dur.reshape(len(res), -1).mean(axis=1)
    print(f"  seuil global (p{args.percentile:g}) = {seuil:.6f} ; "
          f"fraction dure par instantane = "
          f"[{frac.min():.3f}, {frac.max():.3f}]")
    np.savez_compressed(
        out,
        label_kind="dynamic",
        d_errors=d_tous,
        d0_errors=empile("d0_errors"),
        amplification=empile("amplification"),
        l2_errors=empile("l2_errors"),
        classical_scores=empile("classical_scores"),
        is_hard_dynamic=dur,
        d_threshold=seuil,
        hard_fraction_par_instantane=frac,
        t=np.array([r["meta"]["t"] for r in res]),
        snap_index=np.array(idx),
        n_substeps=np.array([r["meta"]["n_substeps"] for r in res]),
        wall_seconds=np.array([r["meta"]["wall_seconds"] for r in res]),
        rho_d_vs_e=np.array(
            [r["diagnostic"]["rho_d_vs_e"] for r in res]),
        amplification_log_iqr=np.array(
            [r["diagnostic"]["amplification_log_iqr"] for r in res]),
        label_informative=np.array(
            [r["diagnostic"]["informative"] for r in res]),
        scenario=args.scenario, Re=args.re, N=args.N, n_patches=args.dim,
        delta_t=np.array([r["meta"]["delta_t"] for r in res]),
        patch_crossing_time=np.array(
            [r["meta"]["patch_crossing_time"] for r in res]),
        crossing_multiple=np.array(
            [r["meta"]["crossing_multiple"] for r in res]),
        horizon_mode=np.array([r["meta"]["horizon_mode"] for r in res]),
        allow_redundant=bool(args.allow_redundant),
        percentile=args.percentile,
        git_hash=git_hash(), argv=" ".join(sys.argv),
    )
    tot = time.time() - t0
    print(f"\n  ecrit : {os.path.basename(out)} "
          f"({os.path.getsize(out) / 1024:.0f} KB)")
    print(f"  cout total {tot:.1f} s, soit {tot / len(idx):.1f} s/instantane")
    print(f"  projection N=256 dim=8 : "
          f"~{tot / len(idx) * (65 / (args.dim ** 2 + 1)) * (256 / args.N) ** 3:.0f} "
          f"s/instantane (echelle N^3 : N^2 cellules x N pas)")


if __name__ == "__main__":
    main()
