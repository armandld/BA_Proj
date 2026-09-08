"""Le generateur de modele jouet (`study/common/toy_model.py`) doit produire
des champs solenoidaux par construction, reproductibles par graine, et
effectivement divers -- les trois proprietes dont depend tout le reste
(entrainement diversifie sans surapprendre a des formes fixes).
"""
import numpy as np

from toy_model import generate_toy_snapshot
from Simulation.solver import MHDSolver

N = 48


def _div_fd4(fx, fy, dx):
    gx, _ = MHDSolver._fd_grad(fx, dx)
    _, gy = MHDSolver._fd_grad(fy, dx)
    return gx + gy


def test_shapes_and_finiteness():
    vx, vy, Bx, By = generate_toy_snapshot(N=N, seed=0)
    for field in (vx, vy, Bx, By):
        assert field.shape == (N, N)
        assert np.all(np.isfinite(field))


def test_velocity_and_field_are_solenoidal_by_construction():
    """Rotationnel d'une fonction de flux : divergence nulle a la troncature
    pres, pas approximativement -- meme borne que `enforce_incompressibility`
    (~1e-14 pour un champ non projete)."""
    vx, vy, Bx, By = generate_toy_snapshot(N=N, seed=0)
    dx = 2 * np.pi / N
    assert np.max(np.abs(_div_fd4(vx, vy, dx))) < 1e-10
    assert np.max(np.abs(_div_fd4(Bx, By, dx))) < 1e-10


def test_same_seed_is_reproducible():
    a = generate_toy_snapshot(N=N, seed=7)
    b = generate_toy_snapshot(N=N, seed=7)
    for fa, fb in zip(a, b):
        np.testing.assert_array_equal(fa, fb)


def test_different_seeds_differ():
    vx_a, _, Bx_a, _ = generate_toy_snapshot(N=N, seed=0)
    vx_b, _, Bx_b, _ = generate_toy_snapshot(N=N, seed=1)
    assert not np.array_equal(vx_a, vx_b)
    assert not np.array_equal(Bx_a, Bx_b)


def test_many_seeds_never_collapse_to_a_constant_field():
    """Un champ constant (bug de tirage, structures qui s'annulent
    systematiquement) serait invisible au mappeur -- toute la valeur du
    modele jouet tient a ce que chaque instance porte un vrai signal."""
    for seed in range(10):
        vx, vy, Bx, By = generate_toy_snapshot(N=N, seed=seed)
        for field in (vx, vy, Bx, By):
            assert np.ptp(field) > 1e-6, f"seed={seed} : champ quasi constant"
