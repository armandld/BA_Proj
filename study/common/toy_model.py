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


def _rescale_to_target_rms(fx, fy, target_rms):
    """Ramene (fx, fy) a une intensite typique fixe, sans toucher au motif
    spatial. Le rotationnel d'une bosse de rayon r amplifie en 1/r, donc
    amplitude et rayon tires independamment produisent une intensite
    globale imprevisible -- sans cette normalisation le nombre et le
    rayon des structures deviendraient des parametres d'echelle plutot
    que de forme."""
    rms = np.sqrt(np.mean(fx ** 2 + fy ** 2))
    if rms < 1e-12:
        return fx, fy
    scale = target_rms / rms
    return fx * scale, fy * scale


def generate_toy_snapshot(N, seed, n_structures_range=(1, 3),
                          radius_range=None, target_velocity_rms=0.5,
                          target_field_rms=0.5, background_noise=0.0,
                          length_L=2 * np.pi):
    """Un instantane statique : vx, vy, Bx, By, chacun (N, N). Meme seed,
    memes champs. Intensite globale fixee separement de la forme
    (`target_velocity_rms`/`target_field_rms`, voir `_rescale_to_target_rms`).

    `radius_range` par defaut : plus petit qu'un patch typique, pour que
    la plupart restent calmes et seuls quelques-uns portent une
    structure -- ce contraste rend la decision de raffinement non
    triviale (voir RESULTS.md pour la mesure qui l'a etabli).

    `background_noise` desactive par defaut : un bruit blanc derive par
    le rotationnel s'amplifie en 1/dx (comparable a l'intensite des
    structures elles-memes) -- a ne remettre que filtre en frequence,
    jamais brut.
    """
    if radius_range is None:
        radius_range = (length_L / 24.0, length_L / 10.0)
    rng = np.random.default_rng(seed)
    grid = PeriodicGrid(N, length_L=length_L)
    n_v = int(rng.integers(*n_structures_range))
    n_b = int(rng.integers(*n_structures_range))

    psi_v = _random_streamfunction(rng, grid.X, grid.Y, length_L, n_v,
                                    (0.5, 1.0), radius_range)
    psi_b = _random_streamfunction(rng, grid.X, grid.Y, length_L, n_b,
                                    (0.5, 1.0), radius_range)
    if background_noise:
        # Bruit ajoute a la fonction de flux, PAS aux champs : le
        # rotationnel d'un bruit reste a divergence nulle, l'ajouter apres
        # coup ne le serait pas (verifie par
        # test_velocity_and_field_are_solenoidal_by_construction).
        psi_v = psi_v + background_noise * rng.standard_normal(grid.X.shape)
        psi_b = psi_b + background_noise * rng.standard_normal(grid.X.shape)

    vx, vy = MHDSolver._curl_z_fd4(psi_v, grid.dx)
    Bx, By = MHDSolver._curl_z_fd4(psi_b, grid.dx)
    vx, vy = _rescale_to_target_rms(vx, vy, target_velocity_rms)
    Bx, By = _rescale_to_target_rms(Bx, By, target_field_rms)
    return vx, vy, Bx, By
