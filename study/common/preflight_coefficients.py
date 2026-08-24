#!/usr/bin/env python3
"""Contrôle avant vol de l'architecture des coefficients.

Les contrôles portent sur la spécificité des familles, leur équilibre,
leur activité à la résolution d'entraînement, leur pertinence spatiale et
la coïncidence exacte entre les chemins Study et QAOA. La pertinence exige
que l'espace de recherche contienne une configuration qui dépasse le score
classique sur le problème de contrôle.

Usage :
    python study/common/preflight_coefficients.py
    python study/common/preflight_coefficients.py --json rapport.json
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.grid import PeriodicGrid                       # noqa: E402
from Simulation.solver import MHDSolver                        # noqa: E402
from Simulation.PhysToAngle import AngleMapper                 # noqa: E402
from Simulation.HamiltParams import PhysicalMapper             # noqa: E402
from train_hyperparams import FIXED_PARAMS, SEARCH_SPACE        # noqa: E402

HP = dict(gamma_hydro=2.1272, gamma_mag=2.3611, kappa=14.3321, sigma=0.05,
          beta_curl=0.8199, beta_xpoint=0.4256, w_z_frac=0.1013,
          relative_percentile=90.0)
RE = RM = 800
RELEVANCE_PROBES = 256
RELEVANCE_MARGIN = 0.01


def relevance_is_sufficient(rho_nominal, rho_classical, rho_best,
                            margin=RELEVANCE_MARGIN):
    """Return whether the nominal signal is live and the search adds value."""
    return bool(
        np.isfinite(rho_nominal)
        and rho_nominal > 0.6
        and np.isfinite(rho_classical)
        and np.isfinite(rho_best)
        and rho_best - rho_classical > margin + 1e-12
    )


def _coeffs(sim, grid, seuil=None, mapper_params=None):
    if seuil is None:
        seuil = FIXED_PARAMS["threshold_amr"]
    params = HP if mapper_params is None else mapper_params
    m = PhysicalMapper(cs=1.0, nu=grid.L / RE, eta_mhd=grid.L / RM,
                       dx=grid.dx, **params)
    st = sim.get_fluxes()
    score = AngleMapper.classical_score(st)
    return m.compute_coefficients(sim, score, st, threshold_amr=seuil,
                                  advanced_anomalies_enabled=True), score


def _champ(n, f):
    g = PeriodicGrid(n)
    sim = MHDSolver(g, dt=1e-3, Re=RE, Rm=RM)
    x = np.arange(n) * g.dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    sim.vx, sim.vy, sim.Bx, sim.By = f(X, Y, g)
    return sim, g


def _amax(v):
    return float(np.abs(np.asarray(v)).max())


def controle_specificite():
    z = lambda X: np.zeros_like(X)
    o = lambda X: np.ones_like(X)
    vortex, gv = _champ(64, lambda X, Y, g: (-np.sin(Y), np.sin(X), o(X), z(X)))
    calme, gc = _champ(64, lambda X, Y, g: (z(X), z(X), o(X), z(X)))
    cv, _ = _coeffs(vortex, gv)
    cc, _ = _coeffs(calme, gc)

    mesures = {"vortex_K_plaq": _amax(cv["K_plaquettes"]),
               "vortex_K_xpoint": _amax(cv["K_xpoint"]),
               "calme_K_plaq": _amax(cc["K_plaquettes"]),
               "calme_C_edges": max(_amax(a) for a in cc["C_edges"])}
    ok = (mesures["vortex_K_plaq"] > 0.1
          and mesures["vortex_K_xpoint"] < 1e-12
          and mesures["calme_K_plaq"] < 1e-12
          and mesures["calme_C_edges"] < 1e-12)
    return ok, mesures


def controle_equilibre():
    z = lambda X: np.zeros_like(X)
    o = lambda X: np.ones_like(X)
    vortex, gv = _champ(64, lambda X, Y, g: (-np.sin(Y), np.sin(X), o(X), z(X)))
    nappe, gn = _champ(64, lambda X, Y, g: (z(X), z(X),
                                            1 + 0.8 * np.tanh(3 * np.sin(Y)), z(X)))
    kv = _amax(_coeffs(vortex, gv)[0]["K_plaquettes"])
    kn = _amax(_coeffs(nappe, gn)[0]["K_plaquettes"])
    r = kn / kv if kv > 0 else float("inf")
    return 0.1 < r < 10.0, {"fluide": kv, "magnetique": kn, "rapport": r}


def controle_vivant():
    g = PeriodicGrid(256)
    sim = MHDSolver(g, dt=1e-3, Re=RE, Rm=RM)
    sim.init_harris_tearing()
    for _ in range(200):
        sim.step_full()
    c, _ = _coeffs(sim, g)
    m = {"K_plaquettes": _amax(c["K_plaquettes"]), "K_xpoint": _amax(c["K_xpoint"])}
    return m["K_plaquettes"] > 0.05 and m["K_xpoint"] > 0.1, m


def controle_pertinence():
    from scipy.stats import spearmanr
    nb, NF, NC, pas = 8, 128, 32, 200
    gf = PeriodicGrid(NF); sf = MHDSolver(gf, dt=1e-3, Re=RE, Rm=RM)
    sf.init_harris_tearing()
    gc = PeriodicGrid(NC); sc = MHDSolver(gc, dt=1e-3, Re=RE, Rm=RM)
    sc.init_harris_tearing()
    for _ in range(pas):
        sf.step_full()
    for _ in range(pas):
        sc.step_full()

    bm = lambda a: a.reshape(nb, a.shape[0] // nb, nb, a.shape[0] // nb).mean(axis=(1, 3))
    ff, fc = sf.get_fluxes(), sc.get_fluxes()
    err = np.zeros((nb, nb))
    for v in ("vx", "vy", "Bx", "By"):
        d = bm(ff[v])
        c_ = bm(np.repeat(np.repeat(fc[v], NF // NC, 0), NF // NC, 1))
        err += np.abs(d - c_) / (np.abs(d).mean() + 1e-12)

    if np.ptp(err) == 0:
        return False, {"raison": "erreur de reference constante"}

    def rho(field):
        block = bm(np.abs(np.asarray(field)))
        if np.ptp(block) == 0:
            return float("nan")
        return float(spearmanr(block.ravel(), err.ravel()).statistic)

    score = AngleMapper.classical_score(ff)
    rho_classical = rho(score)
    nominal, _ = _coeffs(sf, gf, mapper_params=HP)
    rho_nominal = rho(nominal["K_plaquettes"])

    # Sondage déterministe de l'espace réellement transmis à Optuna. Le
    # meilleur point reste un diagnostic et n'est pas injecté comme graine.
    mapper_names = tuple(HP)
    rng = np.random.default_rng(0)
    best_rho = rho_nominal
    best_params = dict(HP)
    for _ in range(RELEVANCE_PROBES):
        params = {}
        for name in mapper_names:
            low, high, log = SEARCH_SPACE[name]
            if log:
                params[name] = float(
                    np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                params[name] = float(rng.uniform(low, high))
        candidate, _ = _coeffs(sf, gf, mapper_params=params)
        candidate_rho = rho(candidate["K_plaquettes"])
        if np.isfinite(candidate_rho) and candidate_rho > best_rho:
            best_rho = candidate_rho
            best_params = params

    gap = best_rho - rho_classical
    ok = relevance_is_sufficient(rho_nominal, rho_classical, best_rho)
    return ok, {
        "rho_nominal": rho_nominal,
        "rho_classical": rho_classical,
        "rho_best_probe": best_rho,
        "gap_best_vs_classical": gap,
        "required_margin": RELEVANCE_MARGIN,
        "n_probes": RELEVANCE_PROBES,
        "best_probe_params": best_params,
    }


def controle_coincidence():
    from study.common.ising_terms_and_annealing import build_ising_terms, total_energy
    from VQA.cost_hamiltonian import create_period_hamiltonian

    rng = np.random.default_rng(0)
    dim = 2
    nq = 2 * dim * dim
    hp = {"H_edges": (rng.normal(size=(dim, dim)) * 0.1,
                      rng.normal(size=(dim, dim)) * 0.1),
          "C_edges": (-np.abs(rng.normal(size=(dim, dim))),
                      -np.abs(rng.normal(size=(dim, dim)))),
          "K_plaquettes": -np.abs(rng.normal(size=(dim, dim))),
          "K_xpoint": -np.abs(rng.normal(size=(dim, dim)))}
    diag = np.real(np.diag(create_period_hamiltonian(hp, dim).to_matrix()))
    h, e, p = build_ising_terms(hp, dim)
    en = np.array([total_energy(np.array([1 - 2 * x for x in b[::-1]]), h, e, p)
                   for b in itertools.product([0, 1], repeat=nq)])
    ecart = float(np.abs(en - diag).max())
    return ecart < 1e-9, {"ecart_max": ecart}


CONTROLES = [
    ("specificite", controle_specificite,
     "chaque famille repond a SON instabilite, le controle uniforme rend zero"),
    ("equilibre", controle_equilibre,
     "canal magnetique / fluide dans [0.1, 10] — reference 2.29"),
    ("vivant", controle_vivant,
     "termes a quatre corps non nuls a N=256, la resolution d'entrainement"),
    ("pertinence", controle_pertinence,
     "un point de l'espace depasse le score classique sur l'erreur reelle"),
    ("coincidence", controle_coincidence,
     "study/ et le circuit rendent la meme energie — reference 5.3e-15"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=None, help="ecrire le rapport ici")
    args = ap.parse_args()

    print("=" * 72)
    print("  CONTROLE AVANT VOL DES COEFFICIENTS")
    print("  A passer avant de lancer une campagne (~224 h CPU).")
    print("=" * 72)

    rapport, tout_ok = {}, True
    for nom, fn, desc in CONTROLES:
        try:
            ok, mesures = fn()
        except Exception as exc:                     # noqa: BLE001
            ok, mesures = False, {"exception": f"{type(exc).__name__}: {exc}"}
        tout_ok &= ok
        rapport[nom] = {"ok": bool(ok), "mesures": mesures, "description": desc}
        print(f"\n  [{'OK ' if ok else 'ECHEC'}] {nom} — {desc}")
        for k, v in mesures.items():
            print(f"          {k:18s} {v}")

    print("\n" + "=" * 72)
    if tout_ok:
        print("  VERDICT : les coefficients font leur travail. Campagne possible.")
    else:
        print("  VERDICT : AU MOINS UN CONTROLE ECHOUE. Ne pas lancer la campagne.")
        print("  Un coefficient qui ne detecte pas ne se corrige pas par un reglage.")
    print("=" * 72)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rapport, fh, indent=2)
        print(f"\nrapport : {args.json}")

    sys.exit(0 if tout_ok else 1)


if __name__ == "__main__":
    main()
