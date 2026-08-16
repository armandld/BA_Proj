#!/usr/bin/env python3
"""Controle AVANT VOL des coefficients — a passer avant de louer des coeurs.

Une campagne de reoptimisation coute ~224 h CPU. Ce module verifie, en
quelques minutes, que les coefficients font ce qu'ils doivent faire AVANT
qu'on les regle. Il rend un code de sortie non nul si un seul controle
echoue.

Les cinq controles, chacun avec sa mesure de reference :

  1. SPECIFICITE   chaque famille repond a SON instabilite, le controle
                   uniforme rend zero sur les quatre.
  2. EQUILIBRE     canal magnetique / canal fluide dans [0.1, 10].
                   Mesure de reference : 2.29 (fluide 0.501, magnetique
                   1.148). RESULTS.md annoncait « 0.44 » : c'etait le
                   rapport INVERSE, fluide/magnetique. Meme mesure, libelle
                   faux — corrige ici et la-bas. Avant l'harmonisation des
                   unites des portes g, le rapport valait 1/27 500.
  3. VIVANT        les termes a quatre corps sont non nuls A LA RESOLUTION
                   D'ENTRAINEMENT (N=256). Ils etaient identiquement nuls
                   avant le critere relatif.
  4. PERTINENCE    le coefficient correle avec l'erreur REELLE
                   DNS-vs-grossier. Reference : rho = 0.798 sur
                   harris_tearing, APRES l'harmonisation des portes g.
                   Il valait 0.897 avant — la correction a legerement
                   baisse la correlation tout en reveillant le canal
                   magnetique. Remesure, non ajustee.
  5. COINCIDENCE   le chemin `study/` et le chemin deploye rendent la meme
                   energie. Reference : 5.3e-15.

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

HP = dict(gamma_hydro=2.1272, gamma_mag=2.3611, kappa=14.3321, sigma=0.05,
          beta_curl=0.8199, beta_xpoint=0.4256, w_z_frac=0.1013)
RE = RM = 800


def _coeffs(sim, grid, seuil=0.3):
    m = PhysicalMapper(cs=1.0, nu=grid.L / RE, eta_mhd=grid.L / RM,
                       dx=grid.dx, **HP)
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

    c, _ = _coeffs(sf, gf)
    kb = bm(np.abs(np.asarray(c["K_plaquettes"])))
    if np.ptp(kb) == 0 or np.ptp(err) == 0:
        return False, {"rho": None, "raison": "coefficient ou erreur constant"}
    rho = float(spearmanr(kb.ravel(), err.ravel()).statistic)
    return rho > 0.6, {"rho": rho}


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
    diag = np.real(np.diag(create_period_hamiltonian(
        hp, dim, advanced_anomalies_enabled=True).to_matrix()))
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
     "le coefficient correle avec l'erreur reelle — reference rho = 0.798"),
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
