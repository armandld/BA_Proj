"""Contracts for the consolidated eight-scenario DNS campaign."""

import os
import sys

import numpy as np
import pytest


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _path in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", name) for name in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import (
    PHYSICS_NOISE_AMPLITUDE, PHYSICS_SEEDS, SCENARIOS, SCENARIO_CONFIG,
)
from dns_sweep import (
    dns_path, init_scenario, patches_path, presence_matrix,
)
from dns_validation import (
    check_kh, energy_non_increasing, fluctuating_KE, mean_sq_current,
)
from data_catalog import labelled_trajectory_paths


def _fresh_sim(N=16):
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver
    return MHDSolver(PeriodicGrid(N), dt=1e-4, Re=400, Rm=400)


def test_protocol_uses_eight_scenarios_and_five_physics_seeds():
    assert len(SCENARIOS) == len(set(SCENARIOS)) == 8
    assert PHYSICS_SEEDS == [0, 1, 2, 3, 4]
    assert set(SCENARIO_CONFIG) == set(SCENARIOS)
    assert set(PHYSICS_NOISE_AMPLITUDE) == set(SCENARIOS)
    assert PHYSICS_NOISE_AMPLITUDE["kelvin_helmholtz"] == 0.005


def test_seeded_paths_preserve_seed_zero_compatibility():
    assert dns_path("/r", "orszag_tang", 400, 256, 0) == \
        "/r/dns_orszag_tang_Re400_N256.npz"
    assert dns_path("/r", "lamb_oseen", 800, 256, 3) == \
        "/r/dns_lamb_oseen_Re800_N256_seed3.npz"
    assert patches_path("/r", "lamb_oseen", 800, 256, 4, 3) == \
        "/r/patches_lamb_oseen_Re800_N256_dim4_seed3.npz"


def test_presence_matrix_counts_requested_reynolds_values(tmp_path):
    dns_path(str(tmp_path), "orszag_tang", 400, 256, 0)
    for re, seed in ((400, 0), (800, 0), (400, 1)):
        path = dns_path(str(tmp_path), "orszag_tang", re, 256, seed)
        open(path, "a", encoding="utf-8").close()
    matrix = presence_matrix(
        str(tmp_path), ["orszag_tang", "lamb_oseen"],
        [400, 800], 256, [0, 1])
    assert matrix[("orszag_tang", 0)] == 2
    assert matrix[("orszag_tang", 1)] == 1
    assert matrix[("lamb_oseen", 0)] == 0


def test_catalog_refuses_a_partial_physics_seed_panel(tmp_path):
    dns = dns_path(str(tmp_path), "orszag_tang", 400, 256, 0)
    open(dns, "a", encoding="utf-8").close()
    with pytest.raises(FileNotFoundError, match="incomplete trajectory panel"):
        labelled_trajectory_paths(
            str(tmp_path), ["orszag_tang"], [400], 256, 4, [0, 1])


def test_catalog_keeps_physics_seeds_as_distinct_trajectories(tmp_path):
    for seed in (0, 1):
        for path in (
                dns_path(str(tmp_path), "orszag_tang", 400, 256, seed),
                patches_path(
                    str(tmp_path), "orszag_tang", 400, 256, 4, seed)):
            open(path, "a", encoding="utf-8").close()
    rows = labelled_trajectory_paths(
        str(tmp_path), ["orszag_tang"], [400], 256, 4, [0, 1])
    assert [(re, seed) for re, seed, _, _ in rows["orszag_tang"]] == [
        (400, 0), (400, 1)]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_scenarios_initialise_with_finite_fields(scenario):
    sim = _fresh_sim()
    init_scenario(sim, scenario)
    for field in (sim.vx, sim.vy, sim.Bx, sim.By):
        assert field.shape == (16, 16)
        assert np.all(np.isfinite(field))


def test_seed_zero_is_identity_and_positive_seeds_are_reproducible():
    plain = _fresh_sim()
    plain.init_lamb_oseen_vortex()
    seed0 = _fresh_sim()
    init_scenario(seed0, "lamb_oseen", phys_seed=0)
    np.testing.assert_array_equal(plain.vx, seed0.vx)
    np.testing.assert_array_equal(plain.By, seed0.By)

    first = _fresh_sim()
    again = _fresh_sim()
    other = _fresh_sim()
    init_scenario(first, "lamb_oseen", phys_seed=1)
    init_scenario(again, "lamb_oseen", phys_seed=1)
    init_scenario(other, "lamb_oseen", phys_seed=2)
    np.testing.assert_array_equal(first.vx, again.vx)
    assert not np.array_equal(first.vx, other.vx)


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="unknown scenario"):
        init_scenario(_fresh_sim(), "not_a_scenario")


def _kh_fields(N=32, amplitude=0.0):
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.tanh((Y - np.pi) / 0.5), amplitude * np.sin(X)


def test_kh_observable_removes_the_base_flow():
    vx, vy = _kh_fields(amplitude=0.0)
    assert fluctuating_KE(vx, vy) < 1e-30
    vx, vy = _kh_fields(amplitude=0.1)
    assert fluctuating_KE(vx, vy) == pytest.approx(0.0025)


def test_current_uses_repository_axis_convention():
    N = 64
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    assert mean_sq_current(-np.sin(X), -np.sin(Y), 2 * np.pi / N) < 1e-20
    assert mean_sq_current(-np.sin(Y), np.zeros_like(X), 2 * np.pi / N) > 0.4


def test_kh_check_detects_growth():
    times = np.linspace(0.0, 1.2, 13)
    energies = np.exp(2.0 * times)
    result = {"t": times, "Ep": energies}
    assert check_kh(result)["ok"]
    result["Ep"] = np.ones_like(times)
    assert not check_kh(result)["ok"]


def test_energy_monotonicity_has_explicit_tolerance():
    assert energy_non_increasing([1.0, 0.9, 0.85])[0]
    assert energy_non_increasing([1.0, 1.0005], tol=1e-3)[0]
    ok, maximum = energy_non_increasing([1.0, 0.9, 0.95])
    assert not ok and maximum == pytest.approx(0.05)
