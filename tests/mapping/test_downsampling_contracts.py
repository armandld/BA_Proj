"""Audit de contrat de la reduction de patch : le score et les champs
parlent-ils de la MEME region ?

Un patch descend vers le VQA par deux chemins independants :

  champs vx, vy, Bx, By, Jz  ->  `refinement._downsample_fields`  (mean-pool)
  score classique            ->  `RescaleArrays._process_score`   (max-pool)

Rien n'oblige les deux a decouper le patch de la meme facon, et rien en aval
ne le verifie. Si les decoupages divergent, la cellule (i,j) du score ne
designe plus la cellule (i,j) des champs : le biais Z d'un qubit se calcule
alors sur une region, ses couplages sur une autre. Le resultat reste un
Hamiltonien valide.

DEFAUT TROUVE ICI : `_downsample_fields` decoupait `patch[:out_dim*bh,
:out_dim*bw]` et jetait le reste de la division, tandis que le max-pool du
score couvre 100 % du patch.

La perte tombe toujours du meme cote — les dernieres lignes et colonnes —
donc c'est un biais, pas du bruit. Ces dernieres lignes sont exactement le
HALO droit et bas, c'est-a-dire l'information de voisinage que l'etude
cherche a evaluer.

Le patch vaut `extent + 2*pad` et la cible `dim + 2*pad` : la division tombe
rarement juste. Pour N=256 a la taille DEPLOYEE dim=2, la couverture vaut
100 % a la profondeur 0 (pad=0) puis 98.5 %, 97.0 % et 94.1 % aux
profondeurs 1, 2 et 3. Le chemin deploye etait donc bien touche des la
premiere descente.
"""

import os
import sys

import numpy as np
import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.RescaleArrays import _maxabs_pool_2d, _process_score  # noqa: E402
from Simulation.refinement import _downsample_fields  # noqa: E402
from Simulation.utils import get_periodic_patch  # noqa: E402

KEYS = ("vx", "vy", "Bx", "By", "Jz")


def _fields(n, seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.normal(size=(n, n)) for k in KEYS}


# ======================================================================
#  1. Couverture : aucune cellule du patch ne doit disparaitre
# ======================================================================

# Extraits par `get_periodic_patch`, les patchs valent extent + 2*pad. Les
# cas ci-dessous sont ceux ou la division ne tombe pas juste — dont la
# taille DEPLOYEE (dim=2, pad=1) a chaque profondeur.
_LOSSY = [
    (128, 2, 1),     # patch 130, out 4  -> profondeur 1 du chemin deploye
    (64, 2, 1),      # patch  66, out 4  -> profondeur 2
    (32, 2, 1),      # patch  34, out 4  -> profondeur 3
    (128, 4, 1),     # patch 130, out 6
    (64, 8, 1),      # patch  66, out 10
    (256, 3, 0),     # patch 256, out 3
]


@pytest.mark.parametrize("extent,target,pad", _LOSSY)
def test_no_cell_of_the_patch_is_dropped_by_the_mean_pool(extent, target, pad):
    """Test par tracage : un pic unique doit se retrouver quelque part.

    On place le pic sur la DERNIERE cellule du patch halo compris — celle
    que la troncature jetait — et on verifie qu'il a laisse une trace.
    """
    out_dim = target + 2 * pad
    n = extent + 2 * pad
    f = {k: np.zeros((n, n)) for k in KEYS}
    # avec pad, get_periodic_patch(0, extent, .., pad) prend les indices
    # -pad .. extent+pad-1 modulo n : la derniere ligne du patch est donc
    # l'indice (extent + pad - 1) % n de l'entree.
    last = (extent + pad - 1) % n
    f["vx"][last, last] = 1000.0
    out = _downsample_fields(f, 0, extent, 0, extent, target, pad=pad)["vx"]
    assert out.shape == (out_dim, out_dim)
    assert np.max(np.abs(out)) > 0.0, (
        "un pic place sur la derniere cellule du patch a disparu de la "
        "reduction : cette region n'entre jamais dans les champs grossiers")
    assert np.unravel_index(np.argmax(np.abs(out)), out.shape) == \
        (out_dim - 1, out_dim - 1), "le pic n'a pas atterri dans le dernier bloc"


@pytest.mark.parametrize("extent,target,pad", _LOSSY[:3])
def test_the_mean_pool_preserves_the_patch_mean(extent, target, pad):
    """La moyenne d'une moyenne de blocs qui pavent tout = la moyenne totale.

    Ne vaut exactement que si les blocs sont de meme taille ; on tolere
    l'ecart du a l'arrondi des bornes, mais pas celui d'une troncature.
    """
    n = extent + 2 * pad
    f = _fields(n, seed=1)
    out = _downsample_fields(f, 0, extent, 0, extent, target, pad=pad)["vx"]
    assert out.mean() == pytest.approx(f["vx"].mean(), abs=0.2 * f["vx"].std())


@pytest.mark.parametrize("extent,target,pad", [(256, 2, 0), (256, 4, 0),
                                               (256, 8, 0), (128, 3, 1)])
def test_an_exact_division_keeps_the_historical_blocks(extent, target, pad):
    """La ou la division tombait juste, la sortie ne doit pas avoir bouge."""
    out_dim = target + 2 * pad
    n = extent + 2 * pad
    assert n % out_dim == 0
    f = _fields(n, seed=2)
    got = _downsample_fields(f, 0, extent, 0, extent, target, pad=pad)["vx"]
    patch = np.roll(np.roll(f["vx"], pad, axis=0), pad, axis=1) if pad else f["vx"]
    bh = n // out_dim
    ref = patch.reshape(out_dim, bh, out_dim, bh).mean(axis=(1, 3))
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("extent,target,pad,couverture", [
    (128, 2, 1, 128 / 130),      # profondeur 1, taille deployee
    (64, 2, 1, 64 / 66),         # profondeur 2
    (32, 2, 1, 32 / 34),         # profondeur 3
])
def test_the_legacy_truncation_really_did_lose_the_last_rows(
        extent, target, pad, couverture):
    """Le defaut lui-meme, reconstitue, avec la couverture qu'il laissait."""
    n = extent + 2 * pad
    out_dim = target + 2 * pad
    a = np.zeros((n, n))
    a[-1, -1] = 1000.0
    bh = n // out_dim
    legacy = a[:out_dim * bh, :out_dim * bh]
    assert legacy.max() == 0.0, (
        "l'ancien decoupage est cense perdre le pic ; s'il ne le perd plus, "
        "ce n'est plus l'ancien decoupage")
    assert (out_dim * bh) / n == pytest.approx(couverture, rel=1e-9)


# ======================================================================
#  2. Cross-path : score et champs doivent decouper pareil
# ======================================================================

@pytest.mark.parametrize("extent,target,pad", _LOSSY[:4])
def test_the_score_and_the_fields_cover_the_same_region(extent, target, pad):
    """Le contrat qui manquait : les deux reductions pavent le meme patch.

    On marque une cellule et on verifie qu'elle tombe dans le MEME bloc
    (i,j) des deux cotes. Un decoupage decale ferait diverger les indices.
    """
    out_dim = target + 2 * pad
    n = extent + 2 * pad
    lo, hi = (-pad) % n, (extent + pad - 1) % n
    for corner in ((lo, lo), (hi, hi), (hi, lo), (lo, hi)):
        a = np.zeros((n, n))
        a[corner] = 1000.0
        f = {k: (a.copy() if k == "vx" else np.zeros((n, n))) for k in KEYS}
        fld = _downsample_fields(f, 0, extent, 0, extent, target, pad=pad)["vx"]
        sco = _process_score(
            get_periodic_patch(a, 0, extent, 0, extent, pad), True, out_dim)
        assert np.unravel_index(np.argmax(np.abs(fld)), fld.shape) == \
            np.unravel_index(np.argmax(np.abs(sco)), sco.shape), (
            f"la cellule {corner} tombe dans deux blocs differents selon "
            "qu'on reduit les champs ou le score")


def test_both_reductions_return_the_same_shape():
    for target, pad in ((2, 1), (3, 1), (4, 1), (4, 0)):
        out_dim = target + 2 * pad
        f = _fields(64, seed=3)
        fld = _downsample_fields(f, 0, 64, 0, 64, target, pad=pad)["vx"]
        sco = _process_score(f["vx"], True, out_dim)
        assert fld.shape == sco.shape == (out_dim, out_dim)


# ======================================================================
#  3. Ce que chaque reduction PRETEND faire
# ======================================================================

def test_the_mean_pool_averages_and_does_not_pick_a_maximum():
    """Champs physiques : la moyenne d'aire, pas le pic."""
    n = 64
    f = {k: np.zeros((n, n)) for k in KEYS}
    f["vx"][:] = 1.0
    f["vx"][0, 0] = 101.0                    # un pic dans le premier bloc
    out = _downsample_fields(f, 0, n, 0, n, 2, pad=1)["vx"]
    bh = n // 4
    assert out[0, 0] == pytest.approx(1.0 + 100.0 / (bh * bh))


def test_the_max_pool_keeps_the_peak_and_does_not_average_it_away():
    """Score d'anomalie : le pic, pas la moyenne — c'est sa raison d'etre."""
    n = 64
    a = np.zeros((n, n))
    a[0, 0] = 7.0
    out = _maxabs_pool_2d(a, 4, 4)
    assert out[0, 0] == pytest.approx(7.0)


def test_the_max_pool_keeps_the_sign_of_the_extremum():
    """max-ABS : c'est la valeur signee qui doit ressortir, pas son module."""
    a = np.zeros((8, 8))
    a[0, 0] = -9.0
    a[0, 1] = 3.0
    assert _maxabs_pool_2d(a, 2, 2)[0, 0] == pytest.approx(-9.0)


def test_a_constant_field_survives_both_reductions_unchanged():
    n = 64
    f = {k: np.full((n, n), 2.5) for k in KEYS}
    fld = _downsample_fields(f, 0, n, 0, n, 4, pad=1)["vx"]
    assert np.allclose(fld, 2.5)
    assert np.allclose(_process_score(f["vx"], True, 6), 2.5)


def test_every_declared_field_key_comes_back():
    """Une cle manquante ferait planter le mappeur bien plus loin."""
    out = _downsample_fields(_fields(64, 4), 0, 64, 0, 64, 3, pad=1)
    assert set(out) == set(KEYS)
    for k in KEYS:
        assert out[k].shape == (5, 5) and np.all(np.isfinite(out[k]))


def test_the_reduction_is_linear_in_the_field():
    """Une moyenne d'aire est lineaire : 2*champ -> 2*sortie."""
    f = _fields(64, seed=5)
    a = _downsample_fields(f, 0, 64, 0, 64, 4, pad=1)["vx"]
    f2 = {k: 2.0 * v for k, v in f.items()}
    b = _downsample_fields(f2, 0, 64, 0, 64, 4, pad=1)["vx"]
    assert np.allclose(b, 2.0 * a, atol=1e-12)


def test_a_target_larger_than_the_patch_falls_back_to_interpolation():
    """bh < 1 : on ne peut plus regrouper, il faut interpoler — pas planter."""
    f = {k: np.random.default_rng(6).normal(size=(4, 4)) for k in KEYS}
    out = _downsample_fields(f, 0, 4, 0, 4, 8, pad=1)["vx"]
    assert out.shape == (10, 10) and np.all(np.isfinite(out))
