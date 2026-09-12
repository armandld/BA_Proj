#!/usr/bin/env python3
"""D-195, troisieme cause : l'optimum exact du hamiltonien de
`test_qaoa_noise_and_early.py::test_noise_robustness` (Orszag-Tang, sans
bruit) coincide-t-il avec la decision classique ?

Reconstruit EXACTEMENT le hamiltonien que `qaoa_block_scores` construit a
cette configuration (meme mapper, meme `get_adaptive_flux`, meme graine
physique implicite du solveur), puis calcule son etat fondamental par
enumeration exhaustive (`exhaustive_ground_state`, memes fonctions que
T13/D-53 -- rien de reimplemente) au lieu de laisser QAOA l'approcher.

Sortie : results/h3_noise_robustness_exact_check.npz
Usage :
  python study/h3_representation/h3_noise_robustness_exact_check.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "tests", "quantum")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import get_adaptive_flux
from ising_terms_and_annealing import (
    build_ising_terms, exhaustive_ground_state, spins_to_decisions,
)
from h2b_feature_selection import git_commit_hash
# Reutilise le protocole du test plutot que de le reconstruire a la main :
# meme trajectoire DNS, meme selection top-k, meme fraction capturee.
from test_qaoa_noise_and_early import (
    ground_truth_errors, select_top_k, captured_fraction,
)

# Configuration EXACTE de test_noise_robustness pour Orszag-Tang, noise=0.
N = 64
N_BLOCKS = 3
THRESHOLD = 0.3
W_Z_FRAC = 0.15
BUDGET = 2
N_STEPS = 500


def build_configuration():
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    sim.init_orszag_tang()
    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    phi_prev = None
    for i in range(N_STEPS):
        if i == N_STEPS - 1:
            phi_prev = mapper.compute_stress_flux(sim.get_fluxes())
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)

    gt = ground_truth_errors(sim, N, N_BLOCKS)
    gt_sel = select_top_k(gt, BUDGET)
    gt_frac = captured_fraction(gt_sel, gt)

    nu = grid.L / 800
    eta = grid.L / 800
    hamilt_mapper = PhysicalMapper(
        cs=1.0, nu=nu, eta_mhd=eta, beta_curl=0.5, beta_xpoint=0.5,
        dx=grid.dx, gamma_hydro=0.5, gamma_mag=0.5, kappa=5.0,
        w_z_frac=W_Z_FRAC,
    )
    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)
    full_score = AngleMapper.classical_score(physics_state)
    hamilt_params = hamilt_mapper.compute_coefficients(
        sim, full_score, physics_state, THRESHOLD,
        advanced_anomalies_enabled=True)

    phi_h, phi_v = Phi["phi_horizontal"], Phi["phi_vertical"]
    prev_h, prev_v = phi_prev["phi_horizontal"], phi_prev["phi_vertical"]
    _, _, _, _, mini_hp, mini_score = get_adaptive_flux(
        phi_h, phi_v, prev_h, prev_v, full_score, hamilt_params,
        target_dim=N_BLOCKS, type_filter=True)
    mini_score = np.clip(mini_score, 0.0, 1.0)
    return mini_hp, mini_score, gt, gt_sel, gt_frac


def main():
    print("=" * 78)
    print("  D-195 : l'optimum exact coincide-t-il avec la decision classique ?")
    print(f"  Orszag-Tang, N={N}, n_blocks={N_BLOCKS}, sans bruit, "
          f"budget={BUDGET}")
    print("=" * 78)

    mini_hp, mini_score, gt, gt_sel, gt_frac = build_configuration()
    cl_sel = select_top_k(mini_score, BUDGET)
    cl_frac = captured_fraction(cl_sel, gt)

    h, e, pq = build_ising_terms(mini_hp, N_BLOCKS)
    n_q = 2 * N_BLOCKS * N_BLOCKS
    gs, energy, n_optima = exhaustive_ground_state(h, e, pq, n_q)
    dh, dv = spins_to_decisions(np.asarray(gs), N_BLOCKS)
    exact_mask = (dh | dv)
    exact_frac_refined = float(exact_mask.mean())
    uniform = bool(exact_mask.all() or not exact_mask.any())

    print(f"\n  selection classique (top-{BUDGET}) : {sorted(cl_sel)}")
    print(f"  fraction capturee classique        : {cl_frac:.4f}")
    print(f"\n  etat fondamental exact -- fraction raffinee : "
          f"{exact_frac_refined:.4f}")
    print(f"  degenerescence (n_optima)                   : {n_optima}")
    print(f"  uniforme (raffine-tout ou raffine-rien)     : {uniform}")

    coincide = uniform and cl_frac > 0
    print("\n  LECTURE : ", end="")
    if uniform:
        print("l'optimum exact est UNIFORME (raffine partout ou nulle part) "
              "-- il ne PEUT PAS coincider avec une selection classique non "
              "triviale de %d cellules sur %d. La 3e cause de D-195 (« "
              "l'optimum coincide avec la decision classique ») est "
              "REFUTEE ici : il n'y a pas de decision non triviale a "
              "coincider avec." % (BUDGET, N_BLOCKS * N_BLOCKS))
    else:
        print("l'optimum exact n'est pas uniforme -- comparer directement "
              "a la selection classique.")

    out = os.path.join(_REPO_ROOT, "results",
                       "h3_noise_robustness_exact_check.npz")
    np.savez_compressed(
        out,
        classical_selection=np.array(sorted(cl_sel)),
        classical_captured_fraction=cl_frac,
        ground_truth_captured_fraction=gt_frac,
        exact_ground_state_mask=exact_mask,
        exact_fraction_refined=exact_frac_refined,
        exact_ground_state_uniform=uniform,
        n_optima=n_optima,
        energy=energy,
        n_blocks=N_BLOCKS, N=N, threshold=THRESHOLD, w_z_frac=W_Z_FRAC,
        budget=BUDGET, n_steps=N_STEPS,
        git_hash=git_commit_hash(),
        cli_args=json.dumps({"N": N, "n_blocks": N_BLOCKS,
                             "threshold": THRESHOLD, "w_z_frac": W_Z_FRAC,
                             "budget": BUDGET, "n_steps": N_STEPS}),
    )
    print(f"\n  saved: {os.path.basename(out)}")


if __name__ == "__main__":
    main()
