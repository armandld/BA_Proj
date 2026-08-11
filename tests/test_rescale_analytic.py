"""Le rééchantillonnage qui alimente le VQA, sur des tableaux à réponse connue.

`RescaleArrays` est la porte par laquelle TOUT passe avant d'atteindre le
circuit : score classique, flux de contrainte, coefficients de l'hamiltonien.
Six de ses huit fonctions n'avaient aucun test.

Sa raison d'être est écrite dans son propre docstring : « a single strong
signal (shock, current sheet) in a large block must survive the
downsampling ». Ces tests vérifient cette promesse cellule par cellule, et
épinglent le cas où elle est rompue.

D-1  TRONCATURE SILENCIEUSE. `_maxabs_pool_2d` et `_maxabs_pool_1d` coupent
     le reste de la division : pour une entrée 10×10 vers 3×3, le bloc vaut
     3×3 et le tableau est tronqué à 9×9. **Les lignes et colonnes 9 sont
     jetées.** Un pic isolé qui s'y trouve disparaît sans trace — exactement
     l'anomalie que le pooling existe pour préserver. Mesuré : un pic de
     100.0 en (9,9) ressort à 0.0.

     Le chemin déployé n'y est pas exposé : N et target_dim y sont des
     puissances de deux (256 → 2, 4, 8), donc la division tombe juste. Le
     défaut est armé, pas déclenché — et rien dans le code ne dit qu'il
     dépend de cette coïncidence.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.RescaleArrays import (  # noqa: E402
    _maxabs_pool_1d, _maxabs_pool_2d, _process_score,
    _resize_padded_bilinear, _resize_padded_maxpool, get_adaptive_flux,
)


# ═══════════════════════════════════════════════════════════════════════
#  1. Max-abs pooling : la promesse du module
# ═══════════════════════════════════════════════════════════════════════

def test_a_lone_spike_survives_pooling():
    """La raison d'etre du module : un pic isole doit traverser."""
    a = np.zeros((8, 8))
    a[2, 5] = 42.0
    out = _maxabs_pool_2d(a, 2, 2)
    assert out.shape == (2, 2)
    assert out.max() == 42.0, (
        "le pic a ete moyenne au lieu d'etre conserve")
    #  et il ressort dans le BON bloc : ligne 2 -> bloc 0, colonne 5 -> bloc 1
    assert out[0, 1] == 42.0
    assert np.count_nonzero(out) == 1


def test_pooling_keeps_the_sign_of_the_extremum():
    """max|x| doit renvoyer x, pas |x| : un pic negatif reste negatif."""
    a = np.zeros((4, 4))
    a[0, 0] = -5.0
    a[0, 1] = 3.0
    out = _maxabs_pool_2d(a, 2, 2)
    assert out[0, 0] == -5.0, (
        f"le signe est perdu : {out[0, 0]} au lieu de -5.0")


def test_pooling_beats_averaging_on_a_spike():
    """Controle quantitatif contre l'alternative que le docstring rejette."""
    a = np.zeros((32, 32))
    a[7, 7] = 1.0
    pooled = _maxabs_pool_2d(a, 4, 4).max()
    averaged = a.reshape(4, 8, 4, 8).mean(axis=(1, 3)).max()
    assert pooled == 1.0
    assert averaged == pytest.approx(1.0 / 64)
    assert pooled > 60 * averaged


def test_pooling_is_exact_on_a_uniform_field():
    a = np.full((12, 12), 3.5)
    np.testing.assert_array_equal(_maxabs_pool_2d(a, 3, 3), np.full((3, 3), 3.5))


def test_pooling_reduces_to_identity_at_equal_size():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 5))
    np.testing.assert_array_equal(_maxabs_pool_2d(a, 5, 5), a)


@pytest.mark.parametrize("n,t", [(8, 2), (8, 4), (16, 2), (16, 4), (12, 3),
                                 (9, 3), (256, 2), (256, 8)])
def test_every_input_cell_can_reach_the_output_when_divisible(n, t):
    """Tant que n % t == 0, AUCUNE cellule ne doit etre ignorable.

    On place le pic successivement dans chaque cellule et on verifie qu'il
    ressort. C'est le test qui fait tomber D-1 sur les tailles divisibles.
    """
    assert n % t == 0
    lost = []
    for i in range(n):
        for j in range(n):
            a = np.zeros((n, n))
            a[i, j] = 1.0
            if _maxabs_pool_2d(a, t, t).max() != 1.0:
                lost.append((i, j))
    assert not lost, f"{len(lost)} cellules perdues, ex. {lost[:5]}"


# ── D-1 : la troncature ───────────────────────────────────────────────

@pytest.mark.parametrize("n,t,n_lost", [(10, 3, 19), (10, 4, 36), (7, 2, 13)])
def test_non_divisible_shapes_drop_the_trailing_cells(n, t, n_lost):
    """Le defaut, epingle avec son compte exact.

    `arr[:target*bs]` jette le reste. Pour 10 -> 3, bs = 3, le tableau est
    tronque a 9x9 : la ligne 9 et la colonne 9 disparaissent, soit 19
    cellules sur 100.
    """
    lost = [(i, j) for i in range(n) for j in range(n)
            if _maxabs_pool_2d(np.eye(1, 1) * 0 + _spike(n, i, j), t, t).max() != 1.0]
    assert len(lost) == n_lost, (
        f"{len(lost)} cellules perdues pour {n}->{t}, attendu {n_lost} ; "
        "si ce nombre change, la troncature a ete modifiee")
    #  toutes les pertes sont dans la queue, pas dispersees
    keep = t * (n // t)
    assert all(i >= keep or j >= keep for i, j in lost)


def _spike(n, i, j):
    a = np.zeros((n, n))
    a[i, j] = 1.0
    return a


def test_the_deployed_sizes_are_all_divisible():
    """Le chemin deploye echappe a D-1 — mais par coincidence, pas par
    construction.

    Si une taille non divisible entrait un jour dans la pipeline, des
    anomalies disparaitraient en silence. Ce test documente la dependance.
    """
    _cfg = os.path.join(_REPO_ROOT, "study", "pipeline")
    if _cfg not in sys.path:
        sys.path.insert(0, _cfg)
    from config import DNS_N, VQA_DIMS
    for t in VQA_DIMS:
        assert DNS_N % t == 0, (
            f"DNS_N={DNS_N} n'est pas divisible par VQA_DIM={t} : le pooling "
            "tronquerait et perdrait les dernieres lignes/colonnes")


def test_one_d_pooling_matches_the_two_d_behaviour():
    b = np.zeros(12)
    b[7] = -9.0
    out = _maxabs_pool_1d(b, 3)
    assert out.shape == (3,)
    assert out[1] == -9.0
    assert np.count_nonzero(out) == 1


def test_one_d_pooling_also_truncates():
    b = np.zeros(10)
    b[9] = 100.0
    assert _maxabs_pool_1d(b, 3).max() == 0.0, (
        "si la troncature 1D a ete corrigee, mettre a jour ce fichier")


def test_pooling_upward_falls_back_to_interpolation():
    """Quand la cible est PLUS GRANDE, il n'y a plus de bloc a reduire.

    Le module bascule alors sur `zoom` d'ordre 1, qui interpole : le
    maximum n'est plus garanti. On le verifie pour que personne ne croie
    que la promesse « max-abs » vaut aussi vers le haut.
    """
    a = np.zeros((2, 2))
    a[0, 0] = 1.0
    out = _maxabs_pool_2d(a, 4, 4)
    assert out.shape == (4, 4)
    assert out.max() <= 1.0 + 1e-12


# ═══════════════════════════════════════════════════════════════════════
#  2. Structures avec halo (profondeur > 0)
# ═══════════════════════════════════════════════════════════════════════

def _padded(n, fill_core=1.0, fill_halo=2.0):
    a = np.full((n + 2, n + 2), fill_halo)
    a[1:-1, 1:-1] = fill_core
    return a


@pytest.mark.parametrize("fn", [_resize_padded_bilinear, _resize_padded_maxpool])
def test_padded_resize_keeps_the_halo_structure(fn):
    """Sortie (t+2, t+2), coeur au centre, halo au bord."""
    a = _padded(8)
    out = fn(a, 4)
    assert out.shape == (6, 6)
    np.testing.assert_allclose(out[1:-1, 1:-1], 1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(out[0, 1:-1], 2.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(out[1:-1, 0], 2.0, rtol=0, atol=1e-9)


@pytest.mark.parametrize("fn", [_resize_padded_bilinear, _resize_padded_maxpool])
def test_padded_resize_copies_the_corners_verbatim(fn):
    """Les quatre coins ne sont ni interpoles ni pooles : ils sont copies.

    C'est un choix, pas un accident — on le verrouille pour qu'un
    changement soit visible.
    """
    a = _padded(8)
    a[0, 0], a[0, -1], a[-1, 0], a[-1, -1] = 11.0, 12.0, 13.0, 14.0
    out = fn(a, 4)
    assert (out[0, 0], out[0, -1], out[-1, 0], out[-1, -1]) == (11.0, 12.0, 13.0, 14.0)


def test_padded_maxpool_preserves_a_spike_the_bilinear_one_loses():
    """La difference de traitement entre flux et coefficients, mesuree.

    Les coefficients passent par max-abs, les flux par interpolation : sur
    un pic isole, les deux doivent donner des resultats tres differents.
    """
    a = _padded(8, fill_core=0.0, fill_halo=0.0)
    a[3, 3] = 50.0
    assert _resize_padded_maxpool(a, 4).max() == 50.0
    assert _resize_padded_bilinear(a, 4).max() < 30.0


# ═══════════════════════════════════════════════════════════════════════
#  3. Le point d'entree
# ═══════════════════════════════════════════════════════════════════════

def test_score_and_coefficients_take_the_maxabs_path():
    """Le score et les coefficients ne doivent PAS etre lisses.

    `_process_score` documente « No smoothing! » : on le verifie plutot que
    de le croire.
    """
    a = np.zeros((16, 16))
    a[5, 5] = 1.0
    np.testing.assert_array_equal(_process_score(a, True, 4),
                                  _maxabs_pool_2d(a, 4, 4))
    assert _process_score(a, True, 4).max() == 1.0


def test_flux_takes_the_smoothing_path_and_loses_the_spike():
    """Les flux, eux, sont lisses puis interpoles : le pic doit s'y diluer.

    Ce contraste est le coeur du module. S'il disparaissait, flux et
    coefficients recevraient le meme traitement sans que rien ne le dise.
    """
    a = np.zeros((16, 16))
    a[5, 5] = 1.0
    zero = np.zeros((16, 16))
    mini_h, _mini_v, _hp, _sc = get_adaptive_flux(
        a, zero, None, None, zero, None, target_dim=4, type_filter=True)
    assert mini_h.max() < 0.3, (
        f"le flux conserve le pic ({mini_h.max():.3f}) : il n'est plus "
        "lisse, et flux et coefficients suivent le meme chemin")


def test_e_max_is_never_pooled():
    """`E_max` est une echelle scalaire, pas un champ.

    Le code portait deux `if` successifs au lieu d'un `if`/`elif` : un
    E_max devenu tableau aurait ete silencieusement remplace par sa version
    poolee. Le test couvre les deux formes.
    """
    zero = np.zeros((16, 16))
    for e_max in (7.5, np.arange(16 * 16, dtype=float).reshape(16, 16)):
        _h, _v, hp, _s = get_adaptive_flux(
            zero, zero, None, None, zero, {"E_max": e_max},
            target_dim=4, type_filter=True)
        got = hp["E_max"]
        if np.isscalar(e_max):
            assert got == e_max
        else:
            assert np.shape(got) == np.shape(e_max), (
                "E_max a ete reduit : c'est une echelle, pas un champ")
            np.testing.assert_array_equal(got, e_max)


def test_coefficient_arrays_are_pooled_and_scalars_passed_through():
    zero = np.zeros((16, 16))
    field = np.zeros((16, 16))
    field[9, 9] = -3.0
    hp_in = {"C": field, "K": (field, field), "w": 0.25, "name": "x"}
    _h, _v, hp, _s = get_adaptive_flux(
        zero, zero, None, None, zero, hp_in, target_dim=4, type_filter=True)
    assert hp["C"].shape == (4, 4)
    assert hp["C"].min() == -3.0, "le signe ou l'extremum a ete perdu"
    assert isinstance(hp["K"], tuple) and len(hp["K"]) == 2
    assert hp["K"][0].shape == (4, 4)
    assert hp["w"] == 0.25 and hp["name"] == "x"


def test_previous_flux_changes_the_return_arity():
    """Avec prev_h/prev_v la fonction rend 6 valeurs, sinon 4.

    Une arite qui change selon les arguments est un piege classique ; on
    la fige.
    """
    zero = np.zeros((8, 8))
    assert len(get_adaptive_flux(zero, zero, None, None, zero, None,
                                 target_dim=2, type_filter=True)) == 4
    assert len(get_adaptive_flux(zero, zero, zero, zero, zero, None,
                                 target_dim=2, type_filter=True)) == 6


def test_none_inputs_propagate_as_none():
    zero = np.zeros((8, 8))
    _h, _v, hp, sc = get_adaptive_flux(zero, zero, None, None, None, None,
                                       target_dim=2, type_filter=True)
    assert sc is None
    assert hp == {}
