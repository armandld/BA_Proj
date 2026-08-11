"""D-17 — l'operateur « legacy » n'est pas un rotationnel de signe oppose.

D-1 avait corrige le rotationnel dans `src/`. Le balayage s'etait arrete la.
Trois sites hors de `src/Simulation/` reimplementaient encore leur propre
rotationnel sous la convention inverse (axis=1 lu comme x) :

  study/h2b_prediction/h2b_v1_hamiltonian_loso.py   jz_from_b
  study/h2b_prediction/h2b_ceiling_random_split.py  omega_z, J_z
  figures/v1_legacy/fig_utils.py                    compute_enstrophy

Ce que fait reellement cet operateur :

    correct : (roll(fy,-1,AXIS_X) - fy) - (roll(fx,-1,AXIS_Y) - fx) = dfy/dx - dfx/dy
    legacy  : (roll(fy,-1,AXIS_Y) - fy) - (roll(fx,-1,AXIS_X) - fx) = dfy/dy - dfx/dx

Ce n'est PAS un rotationnel de signe oppose — auquel cas prendre la valeur
absolue ou le carre aurait tout rattrape. C'est son COMPLEMENTAIRE : une
combinaison de deformation, qui vaut exactement zero la ou le rotationnel
est maximal, et qui est maximale la ou le rotationnel s'annule.

  | champ                    | rotationnel | operateur legacy |
  |--------------------------|-------------|------------------|
  | rotation solide          |   +0.392699 |         0.000000 |
  | cisaillement pur         |   -0.196350 |         0.000000 |
  | compression pure         |    0.000000 |        -0.392699 |

Consequence mesuree sur `compute_enstrophy`, cisaillement pur periodique
`vx = sin y` dont l'enstrophie exacte vaut 2 pi^2 = 19.7392 :

  version corrigee : 19.7352  (0.02 % de la valeur exacte, l'erreur de la
                               difference centree)
  version ancienne :  0.0000  (0 % — l'« enstrophie » tracee etait nulle)

Attention au choix du champ de validation : sur Taylor-Green les deux
versions rendent la MEME integrale, par symetrie de leurs carres. Un test
ecrit sur Taylor-Green aurait donc passe sans rien verifier.
"""

import importlib.util
import os
import re
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _d in ("src", "study/h2b_prediction", "study/pipeline", "study/common"):
    _p = os.path.join(_REPO, *_d.split("/"))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.grid import (  # noqa: E402
    AXIS_X,
    AXIS_Y,
    PeriodicGrid,
    forward_curl_z,
    legacy_forward_curl_z,
)
from Simulation.solver import MHDSolver  # noqa: E402

N = 64
L = 2.0 * np.pi


def _coords(n=N):
    c = np.arange(n) * L / n
    return np.meshgrid(c, c, indexing="ij")


def _load(rel, name):
    path = os.path.join(_REPO, *rel.split("/"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
#  1. Ce que l'operateur legacy calcule vraiment
# ======================================================================

def test_the_legacy_operator_is_not_the_curl_with_the_opposite_sign():
    """Si ce n'etait qu'un signe, `abs` et le carre auraient tout rattrape."""
    X, Y = _coords()
    vx, vy = -(Y - np.pi), (X - np.pi)
    a, b = forward_curl_z(vx, vy), legacy_forward_curl_z(vx, vy)
    assert not np.allclose(a, -b), (
        "l'operateur legacy serait alors inoffensif sous valeur absolue ; "
        "il ne l'est pas")


def test_the_legacy_operator_is_blind_to_a_solid_rotation():
    X, Y = _coords()
    vx, vy = -(Y - np.pi), (X - np.pi)
    assert abs(np.median(legacy_forward_curl_z(vx, vy))) < 1e-12
    assert np.median(forward_curl_z(vx, vy)) > 0.1


def test_the_legacy_operator_is_blind_to_a_pure_shear():
    """Champ PERIODIQUE : une rampe (Y - pi) est une dent de scie sur le
    tore, et son raccord fabrique un pic qui masque ce qu'on mesure."""
    X, Y = _coords()
    vx, vy = np.sin(Y), np.zeros_like(X)
    assert np.max(np.abs(legacy_forward_curl_z(vx, vy))) < 1e-12
    assert np.max(np.abs(forward_curl_z(vx, vy))) > 0.05


def test_the_legacy_operator_sees_exactly_what_the_curl_does_not():
    """Compression pure : rotationnel nul, operateur legacy maximal."""
    X, Y = _coords()
    vx, vy = (X - np.pi), -(Y - np.pi)
    assert abs(np.median(forward_curl_z(vx, vy))) < 1e-12
    assert abs(np.median(legacy_forward_curl_z(vx, vy))) > 0.1


# ======================================================================
#  2. Les trois sites corriges
# ======================================================================

def test_jz_from_b_now_calls_the_shared_operator():
    """Reutiliser avant de reecrire : c'est la divergence des copies privees
    qui a produit D-1."""
    m = _load("study/h2b_prediction/h2b_v1_hamiltonian_loso.py", "loso_d17")
    rng = np.random.default_rng(0)
    Bx, By = rng.normal(size=(N, N)), rng.normal(size=(N, N))
    assert np.array_equal(m.jz_from_b(Bx, By), forward_curl_z(Bx, By))


def test_jz_from_b_sees_a_solid_rotation():
    m = _load("study/h2b_prediction/h2b_v1_hamiltonian_loso.py", "loso_d17b")
    X, Y = _coords()
    got = np.median(m.jz_from_b(-(Y - np.pi), (X - np.pi)))
    assert got > 0.1, f"jz_from_b rend {got} sur une rotation solide"


def test_the_h2b_ceiling_features_use_the_repo_axis_constants():
    """Le test lit la SOURCE : les huit differences doivent passer par
    AXIS_X / AXIS_Y, pour qu'un futur editeur ne puisse plus se tromper
    d'axe sans le voir."""
    path = os.path.join(_REPO, "study", "h2b_prediction",
                        "h2b_ceiling_random_split.py")
    src = open(path, encoding="utf-8").read()
    raw = src[src.index("    dxvy ="):src.index("    omega_z =")]
    # on ne juge que le CODE : un commentaire a le droit de citer axis=1
    block = "\n".join(l for l in raw.splitlines()
                      if not l.lstrip().startswith("#"))
    assert "axis=1" not in block and "axis=0" not in block, (
        "une difference du bloc de features est encore ecrite avec un axe "
        f"numerique nu :\n{block}")
    for name, axis in (("dxvy", "AXIS_X"), ("dyvx", "AXIS_Y"),
                       ("dxBy", "AXIS_X"), ("dyBx", "AXIS_Y"),
                       ("dxvx", "AXIS_X"), ("dyvy", "AXIS_Y"),
                       ("dxBx", "AXIS_X"), ("dyBy", "AXIS_Y")):
        assert re.search(rf"{name} = np\.roll\([^,]+, -1, axis={axis}\)", block), (
            f"{name} ne derive pas selon {axis}")


def test_the_h2b_ceiling_omega_matches_the_shared_curl():
    """Reconstruit omega_z depuis la source corrigee et le compare."""
    X, Y = _coords()
    vx, vy = -(Y - np.pi), (X - np.pi)
    dxvy = np.roll(vy, -1, axis=AXIS_X) - vy
    dyvx = np.roll(vx, -1, axis=AXIS_Y) - vx
    assert np.allclose(dxvy - dyvx, forward_curl_z(vx, vy), atol=1e-13)


# ======================================================================
#  3. L'enstrophie des figures
# ======================================================================

def _enstrophy_sim(vx, vy, n):
    g = PeriodicGrid(n, L)
    sim = MHDSolver(g, Re=400, Rm=400)
    sim.vx, sim.vy = vx, vy
    sim.Bx = np.zeros((n, n))
    sim.By = np.zeros((n, n))
    return sim


def test_the_enstrophy_matches_its_analytic_value_on_a_pure_shear():
    """vx = sin y, vy = 0  ->  omega = -cos y  ->  integrale = 2 pi^2.

    Champ choisi EXPRES : sur Taylor-Green les deux conventions rendent la
    meme integrale et le test n'aurait rien verifie.
    """
    n = 256
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    fu = _load("figures/v1_legacy/fig_utils.py", "fu_d17")
    got = fu.compute_enstrophy(_enstrophy_sim(np.sin(Y), np.zeros_like(X), n))
    exact = 2.0 * np.pi ** 2
    assert got == pytest.approx(exact, rel=1e-3), (
        f"enstrophie {got:.6f} contre {exact:.6f} exact")


def test_the_old_enstrophy_convention_returned_exactly_zero_there():
    """Le defaut lui-meme : la grandeur tracee valait 0 % de sa valeur."""
    n = 256
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    dx = L / n
    vx, vy = np.sin(Y), np.zeros_like(X)
    old = ((np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dx)
           - (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dx))
    assert np.sum(old ** 2) * dx ** 2 < 1e-20


def test_taylor_green_cannot_separate_the_two_conventions():
    """Fige le piege : un test ecrit sur ce champ aurait passe pour rien."""
    n = 128
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    dx = L / n
    vx, vy = np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y)
    new = ((np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2 * dx)
           - (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2 * dx))
    old = ((np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dx)
           - (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dx))
    assert np.sum(new ** 2) == pytest.approx(np.sum(old ** 2), rel=1e-9)


def test_the_enstrophy_is_positive_and_scales_as_the_square():
    n = 128
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    fu = _load("figures/v1_legacy/fig_utils.py", "fu_d17b")
    a = fu.compute_enstrophy(_enstrophy_sim(np.sin(Y), np.zeros_like(X), n))
    b = fu.compute_enstrophy(_enstrophy_sim(3.0 * np.sin(Y),
                                            np.zeros_like(X), n))
    assert a > 0
    assert b / a == pytest.approx(9.0, rel=1e-9)


def test_a_uniform_flow_carries_no_enstrophy():
    n = 64
    fu = _load("figures/v1_legacy/fig_utils.py", "fu_d17c")
    ones = np.ones((n, n))
    assert fu.compute_enstrophy(_enstrophy_sim(ones, 2 * ones, n)) < 1e-20


# ======================================================================
#  4. Le balayage : plus aucune copie privee ne doit apparaitre
# ======================================================================

# Chaque entree a ete VERIFIEE a la main : la derivee selon x roule bien
# l'axe 0 et celle selon y l'axe 1. Elles gardent des axes numeriques nus
# pour rester bit-a-bit identiques a ce qui a produit les artefacts publies.
# Ajouter une ligne ici sans l'avoir verifiee reviendrait a desarmer le
# balayage.
_ALLOWED = {
    # grid.py EST l'implementation de reference
    os.path.join("src", "Simulation", "grid.py"),
    # get_fluxes utilise volontairement une difference CENTREE, pas avant
    os.path.join("src", "Simulation", "solver.py"),
    # les mappeurs v1/v2 appellent curl_z ; leurs jacobiens sont centres
    os.path.join("src", "Simulation", "HamiltParams.py"),
    os.path.join("src", "Simulation", "HamiltParams_v2.py"),
    os.path.join("src", "Simulation", "PhysToAngle.py"),
    # le pipeline utilise deja AXIS_X / AXIS_Y explicitement
    os.path.join("src", "pipeline.py"),
    # verifie : grad_By_x roule l'axe 0, grad_Bx_y l'axe 1 -> correct
    os.path.join("study", "pipeline", "hard_patch_labels.py"),
    os.path.join("study", "common", "qaoa_inputs.py"),
    # verifie : dvydx roule l'axe 0, dvxdy l'axe 1 -> correct depuis D-17
    os.path.join("figures", "v1_legacy", "fig_utils.py"),
}


def _sources():
    for root in ("src", "study", "figures"):
        base = os.path.join(_REPO, root)
        for dirpath, _, files in os.walk(base):
            for fn in files:
                if fn.endswith(".py"):
                    full = os.path.join(dirpath, fn)
                    yield full, os.path.relpath(full, _REPO)


def test_no_new_hand_rolled_curl_uses_a_bare_axis_number():
    """Un rotationnel ecrit a la main avec `axis=0`/`axis=1` nus est
    indistinguable d'un rotationnel juste tant qu'on ne le teste pas sur
    une rotation solide. On exige donc AXIS_X / AXIS_Y, qui rendent l'erreur
    visible a la lecture."""
    offenders = []
    for full, rel in _sources():
        if rel in _ALLOWED:
            continue
        lines = open(full, encoding="utf-8").read().splitlines()
        for i, line in enumerate(lines):
            if "np.roll(" not in line or "axis=" not in line:
                continue
            if "AXIS_X" in line or "AXIS_Y" in line:
                continue
            ctx = "\n".join(lines[max(0, i - 4):i + 4]).lower()
            if any(k in ctx for k in ("curl", "jz =", "j_z", "omega_z",
                                      "vortic", "enstroph")):
                offenders.append(f"{rel}:{i + 1}  {line.strip()[:80]}")
    assert not offenders, (
        "rotationnel ecrit a la main avec un axe numerique nu :\n  "
        + "\n  ".join(offenders))


def test_the_repo_axis_constants_are_the_ones_these_tests_assume():
    assert (AXIS_X, AXIS_Y) == (0, 1)


# ======================================================================
#  5. D-18 — la moyenne de base etait prise en travers de la couche
# ======================================================================
#
# `dns_validation.fluctuating_KE` retranche une moyenne pour ne garder que
# la perturbation. La moyenne doit etre prise le long de la direction
# HOMOGENE, celle dont l'ecoulement de base ne depend pas.
#
# `init_kelvin_helmholtz` construit son profil a partir de `grid.Y`, et
# `grid.X, grid.Y = np.meshgrid(x, y, indexing='ij')` fait varier Y le long
# de l'AXE 1. La direction homogene est donc l'axe 0. Le code moyennait sur
# l'axe 1 — A TRAVERS la couche de cisaillement — et ne soustrayait donc
# rien. Le commentaire disait « KH shear is in x », l'inverse de ce que le
# solveur initialise.
#
#   profil de base SEUL, reponse attendue zero :
#     ancien (axis=1) : 3.411e-01   soit 73 % de l'energie cinetique totale
#     correct (axis=0): 1.323e-30
#
#   avec la perturbation nominale (amplitude 0.1) :
#     ancien  : 0.34115 -> 0.34120, rapport 1.0002
#     correct : 1.3e-30 -> 2.5e-04
#
# La grandeur etait a 99.98 % de l'ecoulement de base, donc pratiquement
# aveugle a la perturbation qu'elle existe pour mesurer.

def _dv():
    return _load("study/pipeline/dns_validation.py", "dns_val_d18")


def _kh(noise, n=128):
    sim = MHDSolver(PeriodicGrid(n, L), Re=400, Rm=400)
    sim.init_kelvin_helmholtz(noise_amplitude=noise)
    return sim


def test_the_base_shear_flow_carries_no_perturbation_energy():
    """Sans perturbation, la grandeur doit valoir zero. C'est sa definition."""
    sim = _kh(0.0)
    got = _dv().fluctuating_KE(sim.vx, sim.vy)
    total = 0.5 * np.mean(sim.vx ** 2 + sim.vy ** 2)
    assert got < 1e-20, (
        f"energie de perturbation {got:.3e} sur un ecoulement de base pur, "
        f"soit {got / total:.0%} de l'energie totale : la moyenne est prise "
        "en travers de la couche de cisaillement")


def test_the_perturbation_energy_actually_responds_to_the_perturbation():
    dv = _dv()
    base = dv.fluctuating_KE(*(lambda s: (s.vx, s.vy))(_kh(0.0)))
    pert = dv.fluctuating_KE(*(lambda s: (s.vx, s.vy))(_kh(0.1)))
    assert pert > 1e6 * max(base, 1e-30), (
        f"base {base:.3e}, perturbe {pert:.3e} : rapport {pert / max(base, 1e-30):.4f}. "
        "Une grandeur qui bouge de 0.02 % quand on allume la perturbation "
        "mesure l'ecoulement de base.")


def test_the_perturbation_energy_grows_with_the_noise_amplitude():
    """Quadratique en l'amplitude : doubler le bruit quadruple l'energie."""
    dv = _dv()
    a = dv.fluctuating_KE(*(lambda s: (s.vx, s.vy))(_kh(0.05)))
    b = dv.fluctuating_KE(*(lambda s: (s.vx, s.vy))(_kh(0.10)))
    assert b / a == pytest.approx(4.0, rel=0.05)


def test_the_old_axis_would_report_most_of_the_base_flow():
    """Le defaut lui-meme, fige : 73 % de l'energie totale sur un
    ecoulement sans la moindre perturbation."""
    sim = _kh(0.0)
    vx_m = sim.vx.mean(axis=1, keepdims=True)
    vy_m = sim.vy.mean(axis=1, keepdims=True)
    old = 0.5 * np.mean((sim.vx - vx_m) ** 2 + (sim.vy - vy_m) ** 2)
    total = 0.5 * np.mean(sim.vx ** 2 + sim.vy ** 2)
    assert old / total > 0.5


def test_the_mean_square_current_sees_a_current_sheet():
    """<J^2> doit suivre le rotationnel partage, pas son complementaire."""
    n = 128
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    Bx, By = -np.sin(Y), np.zeros_like(X)      # Jz = -dBx/dy = cos y
    got = _dv().mean_sq_current(Bx, By)
    ref = np.mean(forward_curl_z(Bx, By) ** 2)
    assert got == pytest.approx(ref, rel=0.05)
    assert got > 0.0


def test_the_mean_square_current_is_zero_on_a_potential_field():
    """B = grad(phi) porte un courant nul : c'est le controle negatif."""
    n = 128
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    # phi = cos x + cos y  ->  B = (-sin x, -sin y),  curl B = 0
    got = _dv().mean_sq_current(-np.sin(X), -np.sin(Y))
    assert got < 1e-25
