#!/usr/bin/env python3
"""
Diagnostic test: compare X-point reconnection detection approaches.

The X-point detector uses ZZZZ plaquette terms (K_xpoint) based on
det(J_B) = dBx/dx * dBy/dy - dBx/dy * dBy/dx.
The signal is max(0, -det(J_B)) — positive only at X-points
(hyperbolic magnetic nulls).

K_xpoint uses the SAME plaquette topology as K_plaquettes:
  {H(i,j), V(i,j+1), H(i+1,j), V(i,j)}

K_xpoint is computed in HamiltParams.py when advanced_anomalies_enabled=True
and is sliced in utils.py slice_hamiltonian_params as a single 2D array.

This test creates an island coalescence profile on a 2D grid and verifies
the spatial selectivity of the X-point detection approach.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.HamiltParams import PhysicalMapper


def build_island_coalescence_fields(N=16, B0=0.3):
    """Create an island coalescence scenario and evolve to develop X-points."""
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=0.001, Re=1000, Rm=1000)
    sim.init_island_coalescence(B0=B0)
    # Evolve a few steps so the X-point structure develops
    for _ in range(20):
        sim.step_full(record_stats=False)
    fields = sim.get_fluxes()
    return sim, grid, fields


def xpoint_coefficients(mapper, sim, fields, dx):
    """Compute K_xpoint from the PhysicalMapper pipeline."""
    score = mapper.physical_score(fields)
    hp = mapper.compute_coefficients(
        sim, score, fields, 0.0,
        advanced_anomalies_enabled=True,
    )
    K_xpoint = hp.get('K_xpoint', np.zeros((fields['vx'].shape)))
    return K_xpoint, hp


def test_xpoint_selectivity():
    """Verify spatial selectivity of K_xpoint: concentrated at reconnection sites."""
    N = 32
    sim, grid, fields = build_island_coalescence_fields(N=N, B0=0.3)
    dx = grid.dx
    mapper = PhysicalMapper(cs=1.0, nu=1e-3, eta_mhd=1e-3, beta_xpoint=0.5, dx=dx)

    K_xpoint, hp = xpoint_coefficients(mapper, sim, fields, dx)

    total_cells = N * N

    # Cells with any X-point activity (|K_xpoint| > small threshold)
    active_xpoint = np.sum(np.abs(K_xpoint) > 0.01) / total_cells

    print("=" * 70)
    print("X-POINT RECONNECTION DETECTION SELECTIVITY")
    print("=" * 70)

    print(f"\n{'Metric':<45} {'Value':>10}")
    print("-" * 55)
    print(f"{'K_xpoint field: min / mean / max':<45} {K_xpoint.min():>4.4f} / {K_xpoint.mean():>4.4f} / {K_xpoint.max():>4.4f}")
    print(f"{'|K_xpoint| > 0.01 fraction':<45} {active_xpoint:>10.1%}")
    print(f"{'|K_xpoint| mean':<45} {np.mean(np.abs(K_xpoint)):>10.4f}")
    print(f"{'|K_xpoint| max':<45} {np.max(np.abs(K_xpoint)):>10.4f}")

    # Compute the spatial concentration ratio: max / mean
    # Higher = more focused on the X-point region
    conc = np.max(np.abs(K_xpoint)) / max(np.mean(np.abs(K_xpoint)), 1e-10)
    print(f"{'Concentration (max/mean)':<45} {conc:>10.1f}")

    print("\n--- Interpretation ---")
    if active_xpoint < 0.5:
        print(f"  X-point detection activates {active_xpoint:.1%} of cells (spatially selective)")
    else:
        print(f"  X-point detection activates {active_xpoint:.1%} of cells (broad activation)")

    if conc > 2.0:
        print(f"  Concentration ratio {conc:.1f} indicates focused detection")
    else:
        print(f"  Concentration ratio {conc:.1f} indicates diffuse detection")

    print()

    # Sanity check: Kelvin-Helmholtz (no X-points expected)
    print("=" * 70)
    print("SANITY CHECK: Kelvin-Helmholtz (no X-points expected)")
    print("=" * 70)
    grid_kh = PeriodicGrid(resolution_N=N)
    sim_kh = MHDSolver(grid_kh, dt=0.001, Re=1000, Rm=1000)
    sim_kh.init_kelvin_helmholtz()
    for _ in range(20):
        sim_kh.step_full(record_stats=False)
    fields_kh = sim_kh.get_fluxes()

    K_xpoint_kh, _ = xpoint_coefficients(mapper, sim_kh, fields_kh, grid_kh.dx)

    active_kh = np.sum(np.abs(K_xpoint_kh) > 0.01) / total_cells

    print(f"\n{'Metric':<45} {'Value':>10}")
    print("-" * 55)
    print(f"{'|K_xpoint| > 0.01 fraction':<45} {active_kh:>10.1%}")
    print(f"{'|K_xpoint| mean':<45} {np.mean(np.abs(K_xpoint_kh)):>10.4f}")

    print()

    # ── Assertions ────────────────────────────────────────────────
    #
    # Ce test n'en portait AUCUNE. Son verdict — les deux branches
    # `good_selectivity` / `good_concentration` — vivait dans le bloc
    # `if __name__ == "__main__"`, que pytest n'execute JAMAIS. Le test
    # imprimait un rapport, rendait un tuple, et passait quoi qu'il arrive.
    # Seul cas du depot (balayage AST sur ~1970 tests, helpers resolus).
    #
    # Seuils MESURES, pas choisis :
    #   island_coalescence : 27.5 % de cellules actives, concentration 8.4
    #   kelvin_helmholtz   :  0.0 % de cellules actives, moyenne 0.0000
    #
    # C'est Kelvin-Helmholtz qui fait de ce test une mesure : sans ce
    # controle, « 27.5 % de cellules actives » ne separerait rien.

    assert active_kh == pytest.approx(0.0, abs=1e-12), (
        f"Kelvin-Helmholtz ne porte pas de point X, or {active_kh:.1%} des "
        f"cellules sont actives — le detecteur repond a autre chose")
    assert active_xpoint > 0.05, (
        f"island_coalescence porte des points X, or seules "
        f"{active_xpoint:.1%} des cellules sont actives (mesure : 27.5 %)")
    assert active_xpoint < 0.5, (
        f"{active_xpoint:.1%} des cellules actives : detection non selective "
        f"(mesure : 27.5 %)")
    assert conc > 2.0, (
        f"concentration max/moyenne = {conc:.1f}, detection diffuse "
        f"(mesure : 8.4)")

    print(f"  Selectivite verifiee : {active_xpoint:.1%} sur "
          f"island_coalescence contre {active_kh:.1%} sur Kelvin-Helmholtz.")


if __name__ == "__main__":
    test_xpoint_selectivity()
