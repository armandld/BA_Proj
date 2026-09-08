#!/usr/bin/env python3
"""Modele jouet statique : champs vx/vy/Bx/By synthetiques, pas de DNS.

Chaque instantane superpose un nombre aleatoire de structures localisees
(vortex, point X, nappe de courant) a des positions/echelles/amplitudes
tirees au hasard, sur un fond de bruit doux. vx/vy et Bx/By sont chacun le
rotationnel d'une fonction de flux (meme convention que
`MHDSolver._curl_z_fd4`) : solenoidaux PAR CONSTRUCTION, jamais poses
composante par composante -- c'est le bug qu'a corrige D-1/D-6/D-7 sur les
scenarios DNS, pas la peine de le refaire ici.

Statique : une seule paire de fonctions de flux, pas d'evolution temporelle.
`with_psi=False` en aval (flux temporel nul) est donc le cas d'usage exact,
deja un chemin propre du pipeline (D-122).

Usage:
    from toy_model import generate_toy_snapshot
    vx, vy, Bx, By = generate_toy_snapshot(N=64, seed=0)
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver

STRUCTURE_KINDS = ("vortex", "saddle", "sheet")


def _wrapped_offset(coord, centre, length):
    """Distance signee la plus courte sur un domaine periodique [0, L)."""
    return (coord - centre + length / 2.0) % length - length / 2.0


def _structure_bump(X, Y, L, centre, amplitude, radius, kind, orientation):
    """Une structure localisee dans une fonction de flux, en unites de psi.

    `vortex` : bosse circulaire -> rotationnel = tourbillon isole.
    `saddle` : bosse hyperbolique -> rotationnel = point X (meme signature
        que `det(grad B) < 0`, verifiee dans test_analytic_fields.py).
    `sheet`  : profil tanh le long d'un axe tire au hasard, fenetre
        gaussienne dans l'axe perpendiculaire pour rester localisee (pas
        besoin de la double-nappe qu'exige la periodicite d'un scenario
        DNS complet -- une seule nappe suffit ici).
    """
    dx = _wrapped_offset(X, centre[0], L)
    dy = _wrapped_offset(Y, centre[1], L)
    c, s = np.cos(orientation), np.sin(orientation)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    if kind == "vortex":
        r2 = u ** 2 + v ** 2
        return amplitude * np.exp(-r2 / (2.0 * radius ** 2))
    if kind == "saddle":
        r2 = u ** 2 + v ** 2
        return amplitude * (u ** 2 - v ** 2) / radius ** 2 * np.exp(-r2 / (2.0 * radius ** 2))
    if kind == "sheet":
        return amplitude * np.tanh(u / (0.35 * radius)) * np.exp(-v ** 2 / (2.0 * radius ** 2))
    raise ValueError(f"structure kind inconnu : {kind!r}")


def _random_streamfunction(rng, X, Y, L, n_structures, amplitude_range, radius_range):
    psi = np.zeros_like(X)
    for _ in range(n_structures):
        kind = rng.choice(STRUCTURE_KINDS)
        centre = rng.uniform(0.0, L, size=2)
        amplitude = rng.uniform(*amplitude_range) * rng.choice([-1.0, 1.0])
        radius = rng.uniform(*radius_range)
        orientation = rng.uniform(0.0, np.pi)
        psi += _structure_bump(X, Y, L, centre, amplitude, radius, kind, orientation)
    return psi


def generate_toy_snapshot(N, seed, n_structures_range=(2, 6),
                          velocity_amplitude_range=(0.3, 1.5),
                          field_amplitude_range=(0.3, 1.5),
                          radius_range=None, background_noise=0.02,
                          length_L=2 * np.pi):
    """Un instantane statique : vx, vy, Bx, By, chacun (N, N).

    Nombre, type, position, echelle et amplitude des structures sont tires
    de `seed` -- meme seed, memes champs ; seed different, instance
    differente. `radius_range` par defaut : d'un huitieme a un tiers du
    domaine, assez grand pour survivre a la coarsification que
    `hard_patch_labels.py` applique pour mesurer la verite terrain.
    """
    if radius_range is None:
        radius_range = (length_L / 8.0, length_L / 3.0)
    rng = np.random.default_rng(seed)
    grid = PeriodicGrid(N, length_L=length_L)
    n_v = int(rng.integers(*n_structures_range))
    n_b = int(rng.integers(*n_structures_range))

    psi_v = _random_streamfunction(rng, grid.X, grid.Y, length_L, n_v,
                                    velocity_amplitude_range, radius_range)
    psi_b = _random_streamfunction(rng, grid.X, grid.Y, length_L, n_b,
                                    field_amplitude_range, radius_range)
    if background_noise:
        psi_v = psi_v + background_noise * rng.standard_normal(grid.X.shape)
        psi_b = psi_b + background_noise * rng.standard_normal(grid.X.shape)

    vx, vy = MHDSolver._curl_z_fd4(psi_v, grid.dx)
    Bx, By = MHDSolver._curl_z_fd4(psi_b, grid.dx)
    return vx, vy, Bx, By
