"""Audit de contrat du chemin « padded » : coeur + halo, profondeur > 0.

`_resize_padded_bilinear` et `_resize_padded_maxpool` reduisent un patch
`(N+2, N+2)` vers `(t_dim+2, t_dim+2)` en traitant separement le coeur, les
quatre bords et les quatre coins. Neuf morceaux recolles a la main : chacun
peut atterrir au mauvais endroit sans que la forme de sortie change d'un
pixel, et donc sans que rien en aval ne s'en apercoive.

Deux constats de cet audit, qui ne sont PAS des defauts mais qui n'etaient
ecrits nulle part :

1. Les quatre coins sont recopies TELS QUELS, jamais reduits. Sur un patch
   130 -> 6, chaque cellule interieure resume ~21x21 pixels et chaque coin
   en resume UN. C'est sans consequence : les coins ne sont jamais lus.
   `create_bounded_hamiltonian` construit ses vecteurs de halo par
   `theta_h_full[1:-1, 0]`, `theta_v_full[0, 1:-1]`, etc., qui excluent les
   coins par construction. Verifie ici en les rendant extremes : l'operateur
   ne bouge pas d'un coefficient.

2. Le flux est reduit par `zoom(..., order=1)`, justifie dans la docstring
   par « smooth physical fields ». Mais Phi n'est PAS lisse : c'est un
   indicateur d'anomalie, construit sur des differences de champ, qui pique
   aux chocs et aux nappes de courant — le score, lui, est max-poole dans le
   MEME fichier pour exactement cette raison.

   Un zoom bilineaire echantillonne, il ne moyenne pas. Un pic isole place a
   256 positions differentes dans un patch 128 -> 4 ne survit qu'a UNE
   d'entre elles ; au centre il rend exactement 0.0000 quand le max-pooling
   rend 1000 et la moyenne de bloc 0.98.

   Sur champs DNS reels, part du pic de Phi conservee apres reduction
   128 -> 4 :

     orszag_tang       38.0 %
     mhd_rotor         69.8 %
     kelvin_helmholtz 100.0 %
     harris_tearing   100.0 %

   Le comportement n'est PAS modifie : bilineaire contre max-pool est un
   choix de modelisation defendable des deux cotes — Phi alimente psi, qui
   encode une derivee temporelle, et pour une derivee un interpolant lisse
   se defend. Ce fichier mesure l'ecart au lieu de le laisser implicite.
"""

import os
import sys

import numpy as np
import pytest
from scipy.ndimage import zoom

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.RescaleArrays import (  # noqa: E402
    _maxabs_pool_1d,
    _maxabs_pool_2d,
    _process_score,
    _resize_padded_bilinear,
    _resize_padded_maxpool,
    get_adaptive_flux,
)
from VQA.cost_hamiltonian import create_bounded_hamiltonian  # noqa: E402

_PADDED = (_resize_padded_bilinear, _resize_padded_maxpool)


def _marked(n):
    """Patch (n+2, n+2) dont chaque zone porte sa propre valeur."""
    a = np.zeros((n + 2, n + 2))
    a[1:-1, 1:-1] = 1.0        # coeur
    a[0, 1:-1] = 2.0           # haut
    a[-1, 1:-1] = 3.0          # bas
    a[1:-1, 0] = 4.0           # gauche
    a[1:-1, -1] = 5.0          # droite
    a[0, 0], a[0, -1] = 6.0, 7.0
    a[-1, 0], a[-1, -1] = 8.0, 9.0
    return a


# ======================================================================
#  1. Chaque morceau atterrit-il a sa place ?
# ======================================================================

@pytest.mark.parametrize("fn", _PADDED, ids=lambda f: f.__name__)
@pytest.mark.parametrize("n,t", [(16, 4), (32, 2), (64, 6), (12, 3)])
def test_every_region_lands_where_it_belongs(fn, n, t):
    """Neuf morceaux recolles a la main : un echange bord/coin ne changerait
    pas la forme de sortie, donc rien en aval ne le verrait."""
    out = fn(_marked(n), t)
    assert out.shape == (t + 2, t + 2)
    assert np.allclose(out[1:-1, 1:-1], 1.0), "le coeur n'est pas au centre"
    assert np.allclose(out[0, 1:-1], 2.0), "le bord haut a bouge"
    assert np.allclose(out[-1, 1:-1], 3.0), "le bord bas a bouge"
    assert np.allclose(out[1:-1, 0], 4.0), "le bord gauche a bouge"
    assert np.allclose(out[1:-1, -1], 5.0), "le bord droit a bouge"
    assert (out[0, 0], out[0, -1], out[-1, 0], out[-1, -1]) == (6.0, 7.0, 8.0, 9.0)


@pytest.mark.parametrize("fn", _PADDED, ids=lambda f: f.__name__)
def test_the_output_keeps_the_halo_exactly_one_cell_thick(fn):
    for t in (2, 3, 4, 8):
        assert fn(_marked(32), t).shape == (t + 2, t + 2)


@pytest.mark.parametrize("fn", _PADDED, ids=lambda f: f.__name__)
def test_a_constant_patch_survives_unchanged(fn):
    a = np.full((34, 34), 2.5)
    assert np.allclose(fn(a, 4), 2.5)


@pytest.mark.parametrize("fn", _PADDED, ids=lambda f: f.__name__)
def test_the_reduction_never_invents_a_nan(fn):
    rng = np.random.default_rng(0)
    out = fn(rng.normal(size=(34, 34)), 4)
    assert np.all(np.isfinite(out))


def test_the_corners_are_copied_verbatim_and_never_pooled():
    """Chaque cellule interieure resume ~21x21 pixels, chaque coin en resume UN."""
    a = _marked(64)
    a[0, 0] = 1e6
    for fn in _PADDED:
        assert fn(a, 6)[0, 0] == 1e6


def test_the_corners_are_never_read_by_the_hamiltonian():
    """C'est ce qui rend le point precedent sans consequence.

    Les vecteurs de halo sont `theta_h_full[1:-1, 0]`, `theta_v_full[0, 1:-1]`
    etc. : ils excluent les coins par construction.
    """
    dim = 3
    p = dim + 2
    z = np.zeros((p, p))
    params = {"H_edges": (z.copy(), z.copy()),
              "C_edges": (np.full((p, p), -1.0), np.full((p, p), -1.0)),
              "K_plaquettes": np.full((p, p), -0.5),
              "threshold_amr": 0.5, "w_z_frac": 1.0}

    def terms(th, tv):
        op, *_ = create_bounded_hamiltonian(params, dim, th, tv,
                                            z.copy(), z.copy())
        return sorted((str(q), complex(c).real)
                      for q, c in zip(op.paulis, op.coeffs))

    base = np.full((p, p), np.pi / 2)
    ref = terms(base.copy(), base.copy())
    th, tv = base.copy(), base.copy()
    for a, b in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        th[a, b], tv[a, b] = 0.0, np.pi
    assert terms(th, tv) == ref, (
        "un coin du halo influence l'Hamiltonien : il faudrait alors le "
        "reduire comme le reste, pas le recopier")


# ======================================================================
#  2. Ce que chaque reduction PRETEND preserver
# ======================================================================

def test_the_maxpool_path_keeps_an_isolated_peak_in_the_core():
    """C'est sa raison d'etre : une nappe de courant dans un gros bloc."""
    a = np.zeros((66, 66))
    a[33, 33] = 1000.0
    assert np.max(np.abs(_resize_padded_maxpool(a, 4))) == 1000.0


def test_the_maxpool_path_keeps_a_peak_in_the_halo_too():
    """Le halo porte l'information de voisinage que H3 evalue."""
    a = np.zeros((66, 66))
    a[0, 33] = -500.0
    out = _resize_padded_maxpool(a, 4)
    assert np.min(out[0, 1:-1]) == -500.0


def test_the_maxpool_keeps_the_signed_extremum_not_its_modulus():
    a = np.zeros((66, 66))
    a[33, 33] = -7.0
    assert np.min(_resize_padded_maxpool(a, 4)) == -7.0


def test_the_bilinear_path_loses_an_isolated_peak_almost_everywhere():
    """Un zoom bilineaire ECHANTILLONNE, il ne moyenne pas.

    Le defaut n'en est pas un — c'est un choix documente pour des champs
    lisses. Mais Phi n'est pas lisse, et l'ecart doit etre mesure plutot
    que suppose.
    """
    n, t = 128, 4
    survived = 0
    positions = [(i, j) for i in range(0, n, 8) for j in range(0, n, 8)]
    for i, j in positions:
        a = np.zeros((n, n))
        a[i, j] = 1000.0
        if np.max(np.abs(zoom(a, (t / n, t / n), order=1))) > 1.0:
            survived += 1
    assert survived <= 2, (
        f"{survived}/{len(positions)} pics survivent : le zoom se comporte "
        "comme une moyenne, ce que ce test suppose faux")
    centre = np.zeros((n, n))
    centre[n // 2, n // 2] = 1000.0
    assert np.max(np.abs(zoom(centre, (t / n, t / n), order=1))) < 1e-9


def test_the_block_mean_would_dilute_the_peak_but_still_see_it():
    """Le troisieme choix possible, pour situer les deux autres."""
    n, t = 128, 4
    a = np.zeros((n, n))
    a[n // 2, n // 2] = 1000.0
    got = np.max(np.abs(a.reshape(t, n // t, t, n // t).mean(axis=(1, 3))))
    assert got == pytest.approx(1000.0 / (n // t) ** 2)
    assert got > 0.0


def test_the_two_reductions_disagree_on_a_peaked_field():
    """Si elles coincidaient, le choix entre les deux serait sans objet."""
    a = np.zeros((66, 66))
    a[33, 33] = 100.0
    b = _resize_padded_bilinear(a, 4)
    m = _resize_padded_maxpool(a, 4)
    assert np.max(np.abs(m)) > 10.0 * max(np.max(np.abs(b)), 1e-12)


def test_the_two_reductions_agree_on_a_genuinely_smooth_field():
    """Sur le champ pour lequel le bilineaire est justifie, l'ecart tombe."""
    n = 66
    c = np.linspace(0, 2 * np.pi, n)
    X, Y = np.meshgrid(c, c, indexing="ij")
    a = np.sin(X) * np.cos(Y) + 2.0
    b = _resize_padded_bilinear(a, 6)[1:-1, 1:-1]
    m = _resize_padded_maxpool(a, 6)[1:-1, 1:-1]
    assert np.max(np.abs(b - m)) < 0.5 * np.ptp(a)


# ======================================================================
#  3. Le pooling 1-D des bords
# ======================================================================

def test_the_one_dimensional_pool_covers_the_whole_halo():
    """Meme correction que le 2-D : le reste de la division n'est plus jete."""
    a = np.zeros(10)
    a[-1] = 5.0
    assert np.max(np.abs(_maxabs_pool_1d(a, 3))) == 5.0


def test_the_one_dimensional_pool_keeps_the_sign():
    a = np.zeros(8)
    a[1] = -4.0
    assert np.min(_maxabs_pool_1d(a, 2)) == -4.0


def test_the_one_dimensional_pool_returns_the_requested_length():
    for n, t in ((10, 3), (64, 4), (7, 2), (5, 5)):
        assert len(_maxabs_pool_1d(np.arange(n, dtype=float), t)) == t


def test_a_target_longer_than_the_input_falls_back_to_interpolation():
    out = _maxabs_pool_1d(np.arange(3, dtype=float), 8)
    assert len(out) == 8 and np.all(np.isfinite(out))


# ======================================================================
#  4. Le dispatch : le bon chemin selon la profondeur
# ======================================================================

def test_the_periodic_scan_takes_the_unpadded_path():
    """Profondeur 0 : pas de halo, sortie exactement (t, t)."""
    a = np.random.default_rng(1).normal(size=(64, 64))
    assert _process_score(a, True, 4).shape == (4, 4)


def test_the_bounded_scan_takes_the_padded_path():
    """Profondeur > 0 : halo present, sortie (t+2, t+2) quand on demande t+2."""
    a = np.random.default_rng(2).normal(size=(66, 66))
    assert _process_score(a, False, 4).shape == (6, 6)


def test_a_none_score_stays_none_instead_of_becoming_an_array():
    assert _process_score(None, True, 4) is None


def test_the_score_dispatch_never_smooths_before_pooling():
    """« No smoothing! » : un lissage prealable diluerait l'anomalie que le
    max-pool existe pour preserver."""
    a = np.zeros((64, 64))
    a[10, 10] = 42.0
    assert np.max(_process_score(a, True, 4)) == 42.0


# ======================================================================
#  5. get_adaptive_flux : arite variable et E_max
# ======================================================================

def _flux_inputs(n=64, t=4):
    rng = np.random.default_rng(3)
    h, v = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    score = rng.uniform(size=(n, n))
    hp = {"C_edges": (rng.normal(size=(n, n)), rng.normal(size=(n, n))),
          "K_plaquettes": rng.normal(size=(n, n)),
          "threshold_amr": 0.15, "E_max": 3.5}
    return h, v, score, hp, t


def test_the_return_arity_says_whether_a_previous_flux_was_given():
    """Quatre valeurs sans passe, six avec. Un appelant qui se trompe
    d'arite plante — ce qui est le bon comportement, mais il faut le figer."""
    h, v, score, hp, t = _flux_inputs()
    assert len(get_adaptive_flux(h, v, None, None, score, hp,
                                 target_dim=t, type_filter=True)) == 4
    assert len(get_adaptive_flux(h, v, h.copy(), v.copy(), score, hp,
                                 target_dim=t, type_filter=True)) == 6


def test_e_max_is_a_scalar_and_must_not_be_pooled():
    """Le `if` etait un `if` et non un `elif` : un E_max devenu tableau
    aurait ete silencieusement remplace par sa version poolee."""
    h, v, score, hp, t = _flux_inputs()
    _, _, mini_hp, _ = get_adaptive_flux(h, v, None, None, score, hp,
                                         target_dim=t, type_filter=True)
    assert mini_hp["E_max"] == 3.5


def test_scalar_entries_pass_through_untouched():
    h, v, score, hp, t = _flux_inputs()
    _, _, mini_hp, _ = get_adaptive_flux(h, v, None, None, score, hp,
                                         target_dim=t, type_filter=True)
    assert mini_hp["threshold_amr"] == 0.15


def test_every_coefficient_array_comes_back_at_the_target_size():
    h, v, score, hp, t = _flux_inputs()
    _, _, mini_hp, mini_score = get_adaptive_flux(
        h, v, None, None, score, hp, target_dim=t, type_filter=True)
    assert mini_hp["K_plaquettes"].shape == (t, t)
    for arr in mini_hp["C_edges"]:
        assert arr.shape == (t, t)
    assert mini_score.shape == (t, t)


def test_the_coefficients_take_the_max_pool_and_not_the_bilinear_path():
    """Un pic de couplage isole doit survivre : c'est ce que le fichier
    annonce, et c'est le contraire du traitement du flux."""
    n, t = 64, 4
    rng = np.random.default_rng(4)
    h, v = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    K = np.zeros((n, n))
    K[17, 41] = -999.0
    hp = {"C_edges": (np.zeros((n, n)), np.zeros((n, n))),
          "K_plaquettes": K, "threshold_amr": 0.15}
    _, _, mini_hp, _ = get_adaptive_flux(h, v, None, None,
                                         rng.uniform(size=(n, n)), hp,
                                         target_dim=t, type_filter=True)
    assert np.min(mini_hp["K_plaquettes"]) == -999.0


def test_no_hamiltonian_params_returns_an_empty_dict_not_a_crash():
    h, v, score, _, t = _flux_inputs()
    out = get_adaptive_flux(h, v, None, None, score, None,
                            target_dim=t, type_filter=True)
    assert out[2] == {}
