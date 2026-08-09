#!/usr/bin/env python3
"""
V4 Task 17 - Mecanisme de l'inertie causale des termes ZZ.

ORIGINE. T13 a etabli un FAIT au niveau des decisions : annuler la famille
ZZ ne change 0.0000 decision, sur le mappeur v2 comme sur le mappeur v1
deploye. Cette tache en donne le MECANISME, et elle le fait a partir d'une
piste trouvee dans la suite de tests de V1 elle-meme : deux tests
pre-existants de `tests/test_v9_metrics.py` et
`tests/test_module_validation.py` echouent sur un checkout propre en
affirmant precisement le contraire (« Orszag-Tang should produce
significant C_edges »), avec max|C_edges| ~ 1e-48.

MECANISME. Dans `HamiltParams.compute_coefficients`, la famille ZZ est
multipliee par une fenetre gaussienne centree sur le seuil de decision :

    w(score) = exp(-((score - threshold_amr) / sigma)^2)
    C_horiz *= w ;  C_vert *= w

L'intention documentee est de concentrer le couplage la ou la decision
classique est incertaine. La consequence est qu'une classe dont le score
physique reste LOIN du seuil voit toute sa famille ZZ s'eteindre, quelle
que soit la physique presente dans le champ. Orszag-Tang est exactement ce
cas : son score reste confine dans le haut de l'intervalle.

Le point n'est pas anecdotique : la docstring du filtre threshold-contrast
justifie son adoption par le fait que la normalisation de Michelson
« tue le signal quand le domaine est uniformement actif ». La fenetre
d'incertitude reintroduit ce meme mode de defaillance un cran plus loin,
au niveau du score et non plus du champ.

CE QUE LA TACHE MESURE. Pour chaque scenario et chaque jeu de parametres
(celui des tests V1 qui echouent, et celui REELLEMENT entraine et deploye
au niveau 3), on rapporte : l'etendue du score, la statistique de la
fenetre w, et l'amplitude resultante de C_edges et K_plaquettes. Aucune
conclusion n'est tiree d'un seul jeu de parametres : la distinction entre
« exactement zero » et « fortement supprime » depend de sigma, et les deux
regimes sont rapportes separement.

V1 est lu, jamais modifie.

Sortie : results/t17_uncertainty_window.npz
Usage :
  python study/v4/t17_uncertainty_window.py --N 64 --steps 30
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

# les quatre classes des folds Level-3, sous leurs noms V1 exacts
SCENARIOS = ("init_kelvin_helmholtz", "init_orszag_tang",
             "init_mhd_rotor", "init_harris_tearing")

# ATTENTION : il existe DEUX sigma « entraines » distincts, et les
# confondre fausse la lecture.
#   - `TRAINED_SIGMA` (= 0.023) est la constante du pipeline ouvert,
#     utilisee par phase5 et donc par les taches 11, 13 et 18 ;
#   - sigma = 0.1888 est la valeur trouvee par Optuna pour le fold
#     Level-3 `ot`, donc propre a la boucle fermee.
# Les deux sont rapportees separement. La premiere est lue du module pour
# qu'elle ne puisse pas diverger de ce qui tourne reellement.
def _deployed_params():
    import qaoa_inputs as p5
    return dict(sigma=float(p5.TRAINED_SIGMA),
                threshold_amr=float(p5.TRAINED_THRESHOLD))


PARAM_SETS = {
    "v1_test_default": dict(sigma=0.05, threshold_amr=0.0),
    "deployed_openloop": _deployed_params(),
    "level3_trained": dict(sigma=0.1888, threshold_amr=0.1496),
}


def uncertainty_window(score, threshold_amr, sigma, axis=1):
    """Fenetre gaussienne d'une famille d'aretes, exactement comme
    `HamiltParams.compute_coefficients` la calcule : moyenne du score sur
    les deux voisins relies par l'arete. `axis=1` donne les aretes
    horizontales (i,j)-(i,j+1), `axis=0` les verticales (i,j)-(i+1,j).
    Les deux familles n'ont pas la meme fenetre et ne doivent pas etre
    appariees l'une a l'autre."""
    sigma = max(float(sigma), 1e-6)
    score_avg = 0.5 * (score + np.roll(score, -1, axis=axis))
    return np.exp(-((score_avg - float(threshold_amr)) / sigma) ** 2)


def evolve(scenario, N, steps, Re=400, Rm=400, dt=1e-3, cfl=0.4):
    """Evolue V1 sans le modifier. Retourne (sim, fluxes)."""
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver

    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=dt, Re=Re, Rm=Rm)
    if not hasattr(sim, scenario):
        return None, None
    getattr(sim, scenario)()
    for _ in range(steps):
        sim.adapt_dt(cfl_target=cfl)
        sim.step_full(record_stats=False)
    return sim, sim.get_fluxes()


def probe(scenario, N, steps, param_sets=PARAM_SETS):
    """Une mesure par (scenario, jeu de parametres)."""
    from Simulation.HamiltParams import PhysicalMapper

    sim, fields = evolve(scenario, N, steps)
    if sim is None:
        return []
    def _coeffs(hm, score, thr):
        """Retourne (|C_horiz|, |C_vert|, max|K|) sans rien modifier."""
        hp = hm.compute_coefficients(sim, score, fields, thr)
        c_h, c_v = hp["C_edges"]
        return (np.abs(np.asarray(c_h)), np.abs(np.asarray(c_v)),
                float(np.max(np.abs(np.asarray(hp["K_plaquettes"])))))

    rows = []
    for name, ps in param_sets.items():
        hm = PhysicalMapper(cs=1.0, eta_mhd=0.01)
        hm.sigma = ps["sigma"]
        score = hm.physical_score(fields)
        thr = ps["threshold_amr"]
        w_h = uncertainty_window(score, thr, hm.sigma, axis=1)
        w_v = uncertainty_window(score, thr, hm.sigma, axis=0)
        w = w_h                      # statistiques rapportees sur les aretes h
        ch, cv, k_max = _coeffs(hm, score, thr)
        c_max = float(max(ch.max(), cv.max()))

        # Amplitude ZZ AVANT la fenetre : on neutralise la gaussienne en
        # prenant sigma -> +inf (w = exp(-0) = 1 partout), sans toucher a
        # V1. Comparer les deux separe les deux causes possibles d'un ZZ
        # nul : la fenetre d'incertitude, ou les gates amont
        # (topologie de deformation, seuils de Reynolds de maille).
        hm_nw = PhysicalMapper(cs=1.0, eta_mhd=0.01)
        hm_nw.sigma = 1e9
        ch_nw, cv_nw, _ = _coeffs(hm_nw, score, thr)
        c_max_nw = float(max(ch_nw.max(), cv_nw.max()))
        w_nw = uncertainty_window(score, thr, hm_nw.sigma)
        assert float(w_nw.min()) > 1.0 - 1e-9, "window not neutralised"

        # RECOUVREMENT des supports. La fenetre est centree la ou le score
        # est INCERTAIN ; le couplage physique est grand la ou les
        # gradients sont forts, ce qui produit justement des scores
        # CONFIANTS. Si les deux supports sont disjoints, la fenetre
        # eteint le couplage meme quand ni l'un ni l'autre n'est nul —
        # d'ou une suppression bien plus forte que ne le suggere max(w).
        # chaque famille d'aretes est appariee A SA PROPRE fenetre
        c_flat = np.concatenate([ch_nw.ravel(), cv_nw.ravel()])
        w_flat = np.concatenate([w_h.ravel(), w_v.ravel()])
        mass_kept = (float(np.sum(c_flat * w_flat) / np.sum(c_flat))
                     if np.sum(c_flat) > 0 else float("nan"))
        if c_flat.std() > 0 and w_flat.std() > 0:
            corr = float(np.corrcoef(c_flat, w_flat)[0, 1])
            # rang : robuste a la forme tres non lineaire de la gaussienne
            from scipy.stats import spearmanr
            rho = float(spearmanr(c_flat, w_flat).statistic)
        else:
            corr, rho = float("nan"), float("nan")

        rows.append(dict(
            scenario=scenario, params=name,
            sigma=float(hm.sigma), threshold_amr=float(thr),
            score_min=float(score.min()), score_max=float(score.max()),
            score_mean=float(score.mean()),
            z_min=float(np.min(np.abs((score - thr) / max(hm.sigma, 1e-6)))),
            w_max=float(w.max()), w_mean=float(w.mean()),
            frac_w_gt_1em3=float(np.mean(w > 1e-3)),
            c_edges_max=c_max, c_edges_max_nowindow=c_max_nw,
            k_plaq_max=k_max,
            zz_mass_kept=mass_kept, corr_c_w=corr, spearman_c_w=rho,
        ))
    return rows


def main():
    p = argparse.ArgumentParser(
        description="V4 T17: ZZ uncertainty-window diagnostic")
    from config import RESULTS_DIR
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    args = p.parse_args()
    np.random.seed(args.seed)

    t0 = time.time()
    print("=" * 78)
    print("  V4 T17 - ZZ uncertainty window: mechanism of causal inertness")
    print("=" * 78)
    print(f"  N={args.N}  steps={args.steps}")
    print()

    rows = []
    for s in args.scenarios:
        r = probe(s, args.N, args.steps)
        if not r:
            print(f"  {s}: not available in V1 solver, skipped")
            continue
        rows += r
        for d in r:
            print(f"  {d['scenario']:<22s} [{d['params']:<15s}] "
                  f"score=[{d['score_min']:.4f},{d['score_max']:.4f}] "
                  f"w_max={d['w_max']:.3e} w_mean={d['w_mean']:.3e} "
                  f"frac(w>1e-3)={d['frac_w_gt_1em3']:.4f}")
            print(f"  {'':<22s} {'':<17s} "
                  f"max|C_edges|={d['c_edges_max']:.3e}  "
                  f"(no window: {d['c_edges_max_nowindow']:.3e})  "
                  f"max|K_plaq|={d['k_plaq_max']:.3e}")
            print(f"  {'':<22s} {'':<17s} "
                  f"ZZ mass kept={d['zz_mass_kept']:.3e}  "
                  f"corr(|C|,w)={d['corr_c_w']:+.3f}  "
                  f"spearman={d['spearman_c_w']:+.3f}")
        print()

    if not rows:
        raise SystemExit("no scenario produced a measurement")

    # lecture : la fenetre est-elle sous le seuil d'underflow utile ?
    print("  " + "-" * 74)
    for name in PARAM_SETS:
        sub = [d for d in rows if d["params"] == name]
        dead = [d["scenario"] for d in sub if d["w_max"] < 1e-10]
        supp = [d["scenario"] for d in sub
                if 1e-10 <= d["w_max"] and d["frac_w_gt_1em3"] < 0.05]
        print(f"  [{name}] ZZ numerically dead on: "
              f"{', '.join(dead) if dead else 'none'}")
        print(f"  [{name}] ZZ strongly suppressed (<5% of edges active) on: "
              f"{', '.join(supp) if supp else 'none'}")
        # attribution : la fenetre, ou les gates amont ?
        upstream = [d["scenario"] for d in sub
                    if d["c_edges_max_nowindow"] < 1e-10]
        print(f"  [{name}] ZZ already zero BEFORE the window (upstream "
              f"gates): {', '.join(upstream) if upstream else 'none'}")

    out = os.path.join(RESULTS_DIR, "t17_uncertainty_window.npz")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    keys = [k for k in rows[0] if k not in ("scenario", "params")]
    np.savez(out,
             scenario=np.array([d["scenario"] for d in rows]),
             params=np.array([d["params"] for d in rows]),
             **{k: np.array([d[k] for d in rows], dtype=float)
                for k in keys},
             git_hash=git_commit_hash(),
             cli_args=json.dumps(vars(args)),
             wall_s=time.time() - t0)
    print()
    print(f"  saved: {os.path.basename(out)}  ({time.time() - t0:.0f}s)")
    print("\nV4 Task 17 complete.")


if __name__ == "__main__":
    main()
