"""Les portes de l'hamiltonien et les aides de grille, sur entrées connues.

Six fonctions de `HamiltParams`, six de `grid`, deux de `utils` et
l'indexation des qubits n'avaient aucun test. Ce sont pourtant elles qui
décident de la FORME de chaque coefficient : une porte mal orientée ne
plante pas, elle produit des coefficients plausibles et faux.

Chaque test compare à une valeur calculable à la main — point fixe d'une
sigmoïde, continuité d'une porte par morceaux, identité entre deux portes
complémentaires — jamais à un seuil choisi.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.HamiltParams import PhysicalMapper          # noqa: E402
from Simulation.grid import AXIS_X, AXIS_Y, PeriodicGrid    # noqa: E402
from Simulation.utils import get_periodic_patch             # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
#  1. _f_gate — porte normal/critique
# ═══════════════════════════════════════════════════════════════════════

def test_f_gate_is_continuous_at_the_critical_value():
    """Les deux branches doivent valoir 1.0 exactement en x = x_crit."""
    xc, g = 3.0, 2.0
    eps = 1e-9
    left = float(PhysicalMapper._f_gate(np.array([xc - eps]), xc, g)[0])
    right = float(PhysicalMapper._f_gate(np.array([xc + eps]), xc, g)[0])
    assert left == pytest.approx(1.0, abs=1e-8)
    assert right == pytest.approx(1.0, abs=1e-8)
    assert abs(left - right) < 1e-7, "porte discontinue au raccord"


def test_f_gate_is_linear_below_the_critical_value():
    """Regime normal : f = x / x_crit.

    La porte divise par `x_crit + 1e-10` pour eviter 0/0. Ce garde-fou
    decale le resultat d'un facteur relatif 1e-10 / x_crit — mesure
    2.5e-11 pour x_crit = 4. La tolerance est donc DERIVEE de l'epsilon du
    code, pas choisie : exiger mieux testerait l'epsilon, pas la linearite.
    """
    xc = 4.0
    x = np.array([0.0, 1.0, 2.0, 3.999])
    np.testing.assert_allclose(PhysicalMapper._f_gate(x, xc, 2.0), x / xc,
                               rtol=10 * 1e-10 / xc, atol=0)


def test_f_gate_is_logarithmic_above_the_critical_value():
    """Regime critique : f = 1 + gamma ln(x / x_crit)."""
    xc, g = 2.0, 1.5
    x = np.array([4.0, 8.0, 16.0])
    np.testing.assert_allclose(PhysicalMapper._f_gate(x, xc, g),
                               1.0 + g * np.log(x / xc), rtol=1e-9)


def test_f_gate_is_clamped_and_monotone():
    """La borne existe (sinon un coefficient extreme passerait) et la porte
    ne redescend jamais."""
    xc, g, fmax = 1.0, 5.0, 10.0
    x = np.logspace(-3, 6, 200)
    f = PhysicalMapper._f_gate(x, xc, g, f_max=fmax)
    assert f.max() == pytest.approx(fmax)
    assert np.all(np.diff(f) >= -1e-12), "porte non monotone"
    #  la borne doit MORDRE : sans elle, f atteindrait ~1+5*ln(1e6) ~ 70
    assert 1.0 + g * np.log(x.max() / xc) > 3 * fmax


def test_f_gate_survives_a_zero_critical_value():
    """x_crit = 0 ne doit pas produire de NaN (l'epsilon est la pour ca)."""
    f = PhysicalMapper._f_gate(np.array([0.0, 1.0]), 0.0, 2.0)
    assert np.all(np.isfinite(f))


# ═══════════════════════════════════════════════════════════════════════
#  2. Portes topologiques g_rot / g_strain / g_mag
# ═══════════════════════════════════════════════════════════════════════

def test_g_rot_and_g_strain_are_exactly_complementary():
    """g_rot(Q) + g_strain(Q) = 1 pour tout Q.

    Les deux portent la meme sigmoide de signes opposes. Si l'identite se
    rompait, une region serait comptee deux fois ou pas du tout.
    """
    Q = np.linspace(-50, 50, 401)
    a = PhysicalMapper._g_rot(Q, 2.0, 5.0)
    b = PhysicalMapper._g_strain(Q, 2.0, 5.0)
    np.testing.assert_allclose(a + b, 1.0, rtol=0, atol=1e-12)


def test_the_topological_gates_are_undecided_at_zero():
    """Q = 0 est la frontiere : les deux portes valent 1/2."""
    for fn in (PhysicalMapper._g_rot, PhysicalMapper._g_strain):
        assert float(fn(np.array([0.0]), 2.0, 5.0)[0]) == pytest.approx(0.5)


def test_g_rot_selects_rotation_and_g_strain_selects_strain():
    """Le sens des deux portes, contre la convention documentee.

    `_compute_q_criterion` rend Q > 0 pour la rotation. Une inversion de
    signe laisserait les deux portes bornees dans [0, 1] et parfaitement
    complementaires : seul ce test la detecte.
    """
    strong_rot = np.array([100.0])
    strong_strain = np.array([-100.0])
    assert float(PhysicalMapper._g_rot(strong_rot, 1.0, 5.0)[0]) > 0.99
    assert float(PhysicalMapper._g_rot(strong_strain, 1.0, 5.0)[0]) < 0.01
    assert float(PhysicalMapper._g_strain(strong_strain, 1.0, 5.0)[0]) > 0.99
    assert float(PhysicalMapper._g_strain(strong_rot, 1.0, 5.0)[0]) < 0.01


def test_the_topological_gates_are_monotone():
    Q = np.linspace(-30, 30, 200)
    assert np.all(np.diff(PhysicalMapper._g_rot(Q, 2.0, 3.0)) >= -1e-15)
    assert np.all(np.diff(PhysicalMapper._g_strain(Q, 2.0, 3.0)) <= 1e-15)


def test_g_mag_turns_on_at_the_critical_current():
    """|Jz| = J_crit est le point de bascule : g = 1/2 exactement."""
    assert float(PhysicalMapper._g_mag(np.array([3.0]), 3.0, 5.0)[0]) == \
        pytest.approx(0.5)
    assert float(PhysicalMapper._g_mag(np.array([30.0]), 3.0, 5.0)[0]) > 0.99
    assert float(PhysicalMapper._g_mag(np.array([0.0]), 3.0, 5.0)[0]) < 0.01


def test_g_mag_ignores_the_sign_of_the_current():
    """La porte lit |Jz| : deux courants opposes doivent l'activer pareil."""
    J = np.array([-7.0, 7.0])
    g = PhysicalMapper._g_mag(J, 3.0, 5.0)
    assert g[0] == pytest.approx(g[1])


@pytest.mark.parametrize("fn,args", [
    (PhysicalMapper._g_rot, (1e-30, 1e6)),
    (PhysicalMapper._g_strain, (1e-30, 1e6)),
    (PhysicalMapper._g_mag, (1e-30, 1e6)),
])
def test_the_gates_never_overflow(fn, args):
    """Le clip a +-500 doit empecher tout NaN/inf sur entrees extremes."""
    x = np.array([-1e30, -1e6, 0.0, 1e6, 1e30])
    out = fn(x, *args)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))


# ═══════════════════════════════════════════════════════════════════════
#  3. Filtres de contraste
# ═══════════════════════════════════════════════════════════════════════

def test_michelson_is_zero_on_a_uniform_field():
    """« If the entire domain is uniform, Mic -> 0 » — verifie."""
    v = np.full(5, 2.0)
    assert np.max(PhysicalMapper._michelson_relu(v, 2.0, 1.0)) == \
        pytest.approx(0.0, abs=1e-9)


def test_michelson_is_bounded_in_zero_one():
    rng = np.random.default_rng(0)
    v = np.abs(rng.standard_normal(500)) * 10
    m = PhysicalMapper._michelson_relu(v, 1.0, 2.0)
    assert np.all(m >= 0.0) and np.all(m < 1.0)


def test_michelson_matches_its_formula():
    val, avg, beta = 4.0, 1.0, 2.0
    got = float(PhysicalMapper._michelson_relu(np.array([val]), avg, beta)[0])
    assert got == pytest.approx((beta * val - avg) / (beta * val + avg), rel=1e-6)


def test_threshold_contrast_is_silent_below_the_threshold():
    """C'est ce qui distingue ce filtre de Michelson : un domaine
    uniformement actif ne doit pas etre annule, mais un domaine sous le
    seuil doit l'etre."""
    v = np.array([0.0, 0.5, 1.0])
    assert np.all(PhysicalMapper._threshold_contrast(v, 1.0, 3.0) == 0.0)


def test_threshold_contrast_matches_its_formula_and_clamp():
    beta, vc, tcmax = 3.0, 2.0, 10.0
    v = np.array([4.0, 6.0, 1e6])
    got = PhysicalMapper._threshold_contrast(v, vc, beta, tc_max=tcmax)
    assert got[0] == pytest.approx(beta * (4.0 / vc - 1.0))
    assert got[1] == pytest.approx(beta * (6.0 / vc - 1.0))
    assert got[2] == pytest.approx(tcmax), "la borne ne mord pas"


# ═══════════════════════════════════════════════════════════════════════
#  4. det(grad B) — le discriminant topologique, dans le vrai code
# ═══════════════════════════════════════════════════════════════════════

INNER = (slice(1, -1), slice(1, -1))


def test_det_jacobian_separates_x_points_from_o_points():
    """B = (y, x) -> det = -1 (point X) ; B = (-y, x) -> det = +1 (point O).

    `tests/test_analytic_fields.py` verifie la meme chose via `grid.grad` ;
    ici c'est la fonction REELLEMENT appelee par le mappeur.
    """
    g = PeriodicGrid(64)
    X, Y, dx = g.X, g.Y, g.dx
    x_pt = PhysicalMapper._compute_det_jacobian_B(Y, X, dx)[INNER]
    o_pt = PhysicalMapper._compute_det_jacobian_B(-Y, X, dx)[INNER]
    np.testing.assert_allclose(x_pt, -1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(o_pt, 1.0, rtol=0, atol=1e-9)


def test_det_jacobian_is_zero_on_a_uniform_field():
    g = PeriodicGrid(32)
    one = np.ones_like(g.X)
    d = PhysicalMapper._compute_det_jacobian_B(0.7 * one, -0.2 * one, g.dx)
    assert np.max(np.abs(d)) < 1e-12


def test_det_jacobian_uses_the_declared_axes():
    """B = (0, x) : dBy/dx = 1, tout le reste nul, donc det = 0.
    Mais B = (y, 0) doit aussi donner det = 0, et B = (x, y) donner +1."""
    g = PeriodicGrid(64)
    X, Y, dx = g.X, g.Y, g.dx
    zero = np.zeros_like(X)
    np.testing.assert_allclose(
        PhysicalMapper._compute_det_jacobian_B(zero, X, dx)[INNER], 0.0,
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        PhysicalMapper._compute_det_jacobian_B(X, Y, dx)[INNER], 1.0,
        rtol=0, atol=1e-9)


# ═══════════════════════════════════════════════════════════════════════
#  5. Sauts de grille
# ═══════════════════════════════════════════════════════════════════════

def test_vector_jump_is_zero_on_a_uniform_field():
    g = PeriodicGrid(16)
    one = np.ones((16, 16))
    for axis in (AXIS_X, AXIS_Y):
        assert np.max(np.abs(g._get_vector_jump(one, 2 * one, axis))) < 1e-15


def test_vector_jump_measures_the_euclidean_difference():
    """Saut connu : (3, 4) d'ecart -> norme 5."""
    g = PeriodicGrid(4)
    fx = np.zeros((4, 4)); fy = np.zeros((4, 4))
    fx[1, :] = 3.0
    fy[1, :] = 4.0
    j = g._get_vector_jump(fx, fy, AXIS_X)
    assert j[0, 0] == pytest.approx(5.0)
    assert j[1, 0] == pytest.approx(5.0)


def test_vector_jump_is_direction_specific():
    """Un champ qui ne varie que selon X ne doit produire aucun saut selon Y."""
    g = PeriodicGrid(16)
    f = np.zeros((16, 16))
    f[::2, :] = 1.0                      # varie le long de l'axe 0 (= X)
    zero = np.zeros((16, 16))
    assert np.max(g._get_vector_jump(f, zero, AXIS_X)) > 0.5
    assert np.max(g._get_vector_jump(f, zero, AXIS_Y)) < 1e-15


def test_vector_jump_is_never_negative():
    rng = np.random.default_rng(0)
    g = PeriodicGrid(16)
    j = g._get_vector_jump(rng.standard_normal((16, 16)),
                           rng.standard_normal((16, 16)), AXIS_X)
    assert np.all(j >= 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  6. Extraction périodique de patch
# ═══════════════════════════════════════════════════════════════════════

def test_periodic_patch_wraps_around_the_torus():
    """Un patch qui deborde doit revenir de l'autre bord, pas etre tronque."""
    a = np.arange(16).reshape(4, 4)
    p = get_periodic_patch(a, 0, 2, 0, 2, pad=1)
    assert p.shape == (4, 4)
    #  coin haut-gauche du patch = cellule (-1, -1) = (3, 3) = 15
    assert p[0, 0] == a[3, 3]
    assert p[1, 1] == a[0, 0]


def test_periodic_patch_without_padding_is_a_plain_slice():
    a = np.arange(36).reshape(6, 6)
    np.testing.assert_array_equal(get_periodic_patch(a, 1, 4, 2, 5),
                                  a[1:4, 2:5])


def test_periodic_patch_covering_the_whole_grid_is_the_grid():
    a = np.arange(25).reshape(5, 5)
    np.testing.assert_array_equal(get_periodic_patch(a, 0, 5, 0, 5), a)


def test_periodic_patch_returns_a_copy_not_a_view():
    """np.ix_ copie ; si cela devenait une vue, ecrire dans le patch
    modifierait la grille en silence."""
    a = np.zeros((4, 4))
    p = get_periodic_patch(a, 0, 2, 0, 2, pad=1)
    p[0, 0] = 99.0
    assert a.max() == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  7. Indexation des qubits
# ═══════════════════════════════════════════════════════════════════════

def test_qubit_indexing_is_a_bijection_on_the_periodic_map():
    """Les 2*dim^2 indices doivent couvrir exactement [0, 2*dim^2).

    Une collision ferait porter deux aretes physiques par le meme qubit ;
    un trou laisserait un qubit sans signification. Ni l'un ni l'autre ne
    fait planter le circuit.
    """
    for dim in (2, 3, 4, 8):
        offset_v = dim * dim
        idx_h = lambda y, x: (y % dim) * dim + (x % dim)
        idx_v = lambda y, x: offset_v + (y % dim) * dim + (x % dim)
        got = sorted([idx_h(i, j) for i in range(dim) for j in range(dim)]
                     + [idx_v(i, j) for i in range(dim) for j in range(dim)])
        assert got == list(range(2 * dim * dim))


def test_qubit_indexing_wraps_periodically():
    """idx(dim, x) doit designer la meme arete que idx(0, x)."""
    dim = 4
    idx_h = lambda y, x: (y % dim) * dim + (x % dim)
    assert idx_h(dim, 2) == idx_h(0, 2)
    assert idx_h(-1, 0) == idx_h(dim - 1, 0)


def test_expected_z_matches_its_definition():
    """<Z> = cos(theta) pour Ry(theta) : les deux extremes et le milieu."""
    from VQA.cost_hamiltonian import get_expected_Z
    assert get_expected_Z(0.0) == pytest.approx(1.0)
    assert get_expected_Z(np.pi) == pytest.approx(-1.0)
    assert get_expected_Z(np.pi / 2) == pytest.approx(0.0, abs=1e-15)
