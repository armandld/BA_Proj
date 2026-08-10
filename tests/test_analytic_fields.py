"""Grandeurs physiques nommees, verifiees sur des champs a reponse connue.

Un test qui compare une grandeur a un seuil ne teste pas la grandeur, il teste
le seuil. Les tests de ce fichier n'utilisent aucun seuil : chaque quantite
nommee du depot (rotationnel, divergence, densite de courant, determinant du
jacobien magnetique, critere Q) est evaluee sur un champ de manuel dont la
valeur exacte se calcule a la main.

Les champs lineaires (vx = -y, vy = x, ...) sont derivables exactement par
differences finies : sur l'interieur du domaine, hors du raccord periodique,
la reponse discrete est la reponse analytique a l'arrondi machine pres. C'est
ce qui permet d'exiger atol=1e-12 plutot qu'une inegalite.

Trois ecarts entre le nom et le calcul sont epingles ici, et chacun a son
test dedie :

  D-a  Les mappeurs (HamiltParams, HamiltParams_v2, PhysToAngle) forment leur
       rotationnel et leur divergence sous la convention indexing='xy', alors
       que `grid.py` declare indexing='ij' (AXIS_X=0, AXIS_Y=1). Sous la
       convention du depot, leur « vorticite » vaut df_y/dy - df_x/dx et leur
       « divergence » df_x/dy + df_y/dx : deux composantes du tenseur des
       deformations. La premiere est aveugle a la rotation solide, la seconde
       a la compression isotrope.

  D-b  `_compute_q_criterion` pondere la deformation de moitie par rapport a
       la rotation : un cisaillement pur, exactement neutre au sens
       d'Okubo-Weiss, y sort a Q = +0.25 et se lit donc « domine par la
       rotation ».

  D-c  Le meme critere compte la partie isotrope du tenseur dans la
       deformation : une expansion pure, sans rotation ni deformation
       deviatorique, y sort a Q = -1.

Les operateurs corriges existent dans `grid.py` sous les noms
`forward_curl_z` / `forward_divergence` et sont accessibles dans les mappeurs
via `fixed_curl=True`. Le chemin par defaut reste inchange : le test
`test_default_path_is_the_legacy_operator_bit_for_bit` le verrouille.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.grid import (  # noqa: E402
    AXIS_X, AXIS_Y, PeriodicGrid, curl_z, divergence,
    forward_curl_z, forward_divergence,
    legacy_forward_curl_z, legacy_forward_divergence,
)

N = 64
#  Le raccord periodique fausse une cellule par bord en differences avant et
#  deux en differences centrees. On mesure sur l'interieur strict.
INNER_FWD = (slice(0, -1), slice(0, -1))
INNER_CTR = (slice(1, -1), slice(1, -1))


@pytest.fixture(scope="module")
def grid():
    return PeriodicGrid(N)


def _uniform(arr, expected, tol=1e-12):
    """Le champ discret doit etre constant ET egal a la valeur analytique."""
    assert np.ptp(arr) < tol, (
        f"la reponse devrait etre uniforme sur un champ lineaire ; "
        f"amplitude mesuree {np.ptp(arr):.3e}")
    np.testing.assert_allclose(np.mean(arr), expected, rtol=0, atol=tol)


# ═══════════════════════════════════════════════════════════════════════
#  1. Rotationnel et divergence — champs lineaires, reponse exacte
# ═══════════════════════════════════════════════════════════════════════

#  nom, vx(X, Y), vy(X, Y), omega_z attendu, div attendu
LINEAR_FIELDS = [
    ("rotation solide",  lambda X, Y: (-Y, X),                    2.0,  0.0),
    ("cisaillement pur", lambda X, Y: (Y, np.zeros_like(X)),     -1.0,  0.0),
    ("expansion pure",   lambda X, Y: (X, Y),                     0.0,  2.0),
    ("deformation pure", lambda X, Y: (X, -Y),                    0.0,  0.0),
]


@pytest.mark.parametrize("name,build,omega,div", LINEAR_FIELDS)
def test_forward_curl_matches_the_analytic_vorticity(grid, name, build, omega, div):
    """omega_z = dv_y/dx - dv_x/dy, exact sur un champ lineaire."""
    vx, vy = build(grid.X, grid.Y)
    _uniform(forward_curl_z(vx, vy)[INNER_FWD] / grid.dx, omega)


@pytest.mark.parametrize("name,build,omega,div", LINEAR_FIELDS)
def test_forward_divergence_matches_the_analytic_divergence(grid, name, build, omega, div):
    """div v = dv_x/dx + dv_y/dy, exact sur un champ lineaire."""
    vx, vy = build(grid.X, grid.Y)
    _uniform(forward_divergence(vx, vy)[INNER_FWD] / grid.dx, div)


def test_the_two_operators_use_the_declared_axes(grid):
    """Garde-fou : si AXIS_X/AXIS_Y changeaient, les formules doivent suivre.

    Sans ce test, inverser les constantes en tete de `grid.py` laisserait
    passer les tests ci-dessus tant que les operateurs sont ecrits avec des
    litteraux 0 et 1.
    """
    vx, vy = -grid.Y, grid.X
    expected = ((np.roll(vy, -1, axis=AXIS_X) - vy)
                - (np.roll(vx, -1, axis=AXIS_Y) - vx))
    np.testing.assert_array_equal(forward_curl_z(vx, vy), expected)


# ── D-a : ce que les mappeurs calculent reellement ────────────────────

@pytest.mark.parametrize("name,build,omega,div", LINEAR_FIELDS)
def test_legacy_operators_compute_the_strain_tensor_not_the_curl(
        grid, name, build, omega, div):
    """La forme historique vaut dv_y/dy - dv_x/dx et dv_x/dy + dv_y/dx.

    C'est-a-dire : la difference des deformations normales et la deformation
    de cisaillement. On l'exige explicitement, pour que toute correction du
    chemin par defaut fasse tomber ce test au lieu de passer inapercue.
    """
    vx, vy = build(grid.X, grid.Y)
    dvx_dx = (np.roll(vx, -1, axis=AXIS_X) - vx)
    dvx_dy = (np.roll(vx, -1, axis=AXIS_Y) - vx)
    dvy_dx = (np.roll(vy, -1, axis=AXIS_X) - vy)
    dvy_dy = (np.roll(vy, -1, axis=AXIS_Y) - vy)

    np.testing.assert_allclose(
        legacy_forward_curl_z(vx, vy)[INNER_FWD],
        (dvy_dy - dvx_dx)[INNER_FWD], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        legacy_forward_divergence(vx, vy)[INNER_FWD],
        (dvx_dy + dvy_dx)[INNER_FWD], rtol=0, atol=1e-12)


def test_legacy_curl_is_blind_to_solid_rotation(grid):
    """Le cas qui compte : un tourbillon en rotation solide sort a zero."""
    vx, vy = -grid.Y, grid.X
    legacy = legacy_forward_curl_z(vx, vy)[INNER_FWD] / grid.dx
    fixed = forward_curl_z(vx, vy)[INNER_FWD] / grid.dx
    _uniform(legacy, 0.0)
    _uniform(fixed, 2.0)


def test_legacy_curl_fires_on_pure_deformation(grid):
    """Reciproquement, un champ de vorticite nulle y sort a -2."""
    vx, vy = grid.X, -grid.Y
    _uniform(legacy_forward_curl_z(vx, vy)[INNER_FWD] / grid.dx, -2.0)
    _uniform(forward_curl_z(vx, vy)[INNER_FWD] / grid.dx, 0.0)


def test_legacy_divergence_is_blind_to_expansion_and_fires_on_shear(grid):
    """Symetriquement pour la divergence."""
    vx, vy = grid.X, grid.Y                       # expansion pure, div = 2
    _uniform(legacy_forward_divergence(vx, vy)[INNER_FWD] / grid.dx, 0.0)
    _uniform(forward_divergence(vx, vy)[INNER_FWD] / grid.dx, 2.0)

    vx, vy = grid.Y, np.zeros_like(grid.X)        # cisaillement pur, div = 0
    _uniform(legacy_forward_divergence(vx, vy)[INNER_FWD] / grid.dx, 1.0)
    _uniform(forward_divergence(vx, vy)[INNER_FWD] / grid.dx, 0.0)


# ── Ordre de convergence sur un champ periodique lisse ────────────────

def _taylor_green(n):
    g = PeriodicGrid(n)
    vx = np.sin(g.X) * np.cos(g.Y)
    vy = -np.cos(g.X) * np.sin(g.Y)
    return g, vx, vy


def test_forward_curl_converges_at_first_order():
    """Taylor-Green : omega_z = 2 sin(x) sin(y), div = 0 exactement.

    Les differences avant sont d'ordre 1 : l'erreur doit etre divisee par
    deux quand la maille l'est. Un operateur faux ne convergerait pas.
    """
    errs = []
    for n in (64, 128, 256):
        g, vx, vy = _taylor_green(n)
        got = forward_curl_z(vx, vy) / g.dx
        exact = 2.0 * np.sin(g.X) * np.sin(g.Y)
        errs.append(np.max(np.abs(got - exact)))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(1.8 < r < 2.2 for r in ratios), (
        f"ordre 1 attendu (rapports ~2), mesure {ratios} pour erreurs {errs}")


def test_taylor_green_is_divergence_free_to_truncation():
    """Le meme champ a une divergence analytiquement nulle."""
    errs = []
    for n in (64, 128, 256):
        g, vx, vy = _taylor_green(n)
        errs.append(np.max(np.abs(forward_divergence(vx, vy) / g.dx)))
    assert errs[0] > errs[-1] > 0.0
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(1.8 < r < 2.2 for r in ratios), (
        f"la divergence residuelle doit etre purement de troncature : {ratios}")


# ═══════════════════════════════════════════════════════════════════════
#  2. Densite de courant Jz du solveur
# ═══════════════════════════════════════════════════════════════════════

def _solver_with_B(n, Bx, By):
    from Simulation.solver import MHDSolver
    g = PeriodicGrid(n)
    sim = MHDSolver(g, dt=1e-4, Re=400, Rm=400)
    sim.Bx, sim.By = Bx, By
    return g, sim


@pytest.mark.parametrize("name,build,jz", [
    ("B = (-y, x)  -> Jz = 2", lambda X, Y: (-Y, X), 2.0),
    ("B = (y, x)   -> Jz = 0", lambda X, Y: (Y, X), 0.0),
    ("B = (0, x)   -> Jz = 1", lambda X, Y: (np.zeros_like(X), X), 1.0),
    ("B = (y, 0)   -> Jz = -1", lambda X, Y: (Y, np.zeros_like(X)), -1.0),
])
def test_solver_current_density_matches_the_analytic_value(name, build, jz):
    """Jz = dB_y/dx - dB_x/dy, differences centrees, exact sur un champ lineaire."""
    g = PeriodicGrid(N)
    Bx, By = build(g.X, g.Y)
    _g, sim = _solver_with_B(N, Bx, By)
    got = sim.get_fluxes()["Jz"][INNER_CTR]
    _uniform(got, jz, tol=1e-10)


def test_current_density_and_fixed_curl_agree_on_the_same_field():
    """Deux chemins independants pour la meme grandeur doivent coincider.

    `solver.get_fluxes` utilise des differences centrees, `forward_curl_z`
    des differences avant : sur un champ lineaire les deux sont exacts, donc
    egaux. C'est ce que le chemin historique ne verifie pas.
    """
    g = PeriodicGrid(N)
    Bx, By = -g.Y, g.X
    _g, sim = _solver_with_B(N, Bx, By)
    centred = sim.get_fluxes()["Jz"][INNER_CTR]
    forward = (forward_curl_z(Bx, By) / g.dx)[INNER_CTR]
    np.testing.assert_allclose(centred, forward, rtol=0, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════
#  3. Determinant du jacobien magnetique : point X contre point O
# ═══════════════════════════════════════════════════════════════════════

def _det_grad_B(grid_obj, Bx, By):
    dBx_dx, dBx_dy = grid_obj.grad(Bx)
    dBy_dx, dBy_dy = grid_obj.grad(By)
    return (dBx_dx * dBy_dy - dBx_dy * dBy_dx)[INNER_CTR]


def test_x_point_has_a_negative_jacobian_determinant(grid):
    """B = (y, x) : grad B = [[0, 1], [1, 0]], det = -1 (point selle)."""
    _uniform(_det_grad_B(grid, grid.Y, grid.X), -1.0, tol=1e-10)


def test_o_point_has_a_positive_jacobian_determinant(grid):
    """B = (-y, x) : grad B = [[0, -1], [1, 0]], det = +1 (centre)."""
    _uniform(_det_grad_B(grid, -grid.Y, grid.X), 1.0, tol=1e-10)


def test_the_determinant_separates_the_two_topologies(grid):
    """Le signe doit distinguer les deux, sinon le detecteur ne detecte rien."""
    x_pt = np.mean(_det_grad_B(grid, grid.Y, grid.X))
    o_pt = np.mean(_det_grad_B(grid, -grid.Y, grid.X))
    assert x_pt < 0.0 < o_pt


# ═══════════════════════════════════════════════════════════════════════
#  4. Critere Q d'Okubo-Weiss
# ═══════════════════════════════════════════════════════════════════════

def _okubo_weiss(grid_obj, vx, vy):
    """Forme standard : omega^2 - (S_n^2 + S_s^2), nulle au cisaillement pur."""
    dvx_dx, dvx_dy = grid_obj.grad(vx)
    dvy_dx, dvy_dy = grid_obj.grad(vy)
    omega = dvy_dx - dvx_dy
    S_n = dvx_dx - dvy_dy
    S_s = dvy_dx + dvx_dy
    return (omega ** 2 - S_n ** 2 - S_s ** 2)[INNER_CTR]


def test_q_criterion_is_positive_on_solid_rotation(grid):
    """Rotation solide : omega = 2, deformation nulle -> Q = 0.5 * 4 = 2."""
    _uniform(grid._compute_q_criterion(-grid.Y, grid.X)[INNER_CTR], 2.0, tol=1e-9)


def test_q_criterion_is_negative_on_pure_deformation(grid):
    """Deformation pure : omega = 0, S_11 = 1, S_22 = -1 -> Q = -1."""
    _uniform(grid._compute_q_criterion(grid.X, -grid.Y)[INNER_CTR], -1.0, tol=1e-9)


def test_q_criterion_weighs_strain_half_against_rotation(grid):
    """D-b : le cisaillement pur, neutre au sens d'Okubo-Weiss, sort positif.

    Pour vx = y, vy = 0 : omega = -1, S_n = 0, S_s = 1, donc la forme
    standard vaut exactement 0 — le cisaillement est la frontiere entre
    rotation et deformation. Le critere du depot y renvoie +0.25 et le
    classe donc du cote « domine par la rotation ».
    """
    vx, vy = grid.Y, np.zeros_like(grid.X)
    _uniform(_okubo_weiss(grid, vx, vy), 0.0, tol=1e-9)
    _uniform(grid._compute_q_criterion(vx, vy)[INNER_CTR], 0.25, tol=1e-9)


def test_q_criterion_counts_isotropic_expansion_as_strain(grid):
    """D-c : une expansion pure, sans rotation ni deformation deviatorique.

    Pour vx = x, vy = y : omega = 0, S_n = 0, S_s = 0, la forme standard
    vaut 0. Le critere du depot renvoie -1 parce que son terme de
    deformation retient S_11^2 + S_22^2, partie isotrope comprise.
    """
    vx, vy = grid.X, grid.Y
    _uniform(_okubo_weiss(grid, vx, vy), 0.0, tol=1e-9)
    _uniform(grid._compute_q_criterion(vx, vy)[INNER_CTR], -1.0, tol=1e-9)


# ═══════════════════════════════════════════════════════════════════════
#  5. Cablage de fixed_curl dans les trois mappeurs
# ═══════════════════════════════════════════════════════════════════════

def _turbulent_state(n=48, steps=20):
    from Simulation.solver import MHDSolver
    g = PeriodicGrid(n)
    sim = MHDSolver(g, dt=1e-3, Re=400, Rm=400)
    sim.init_orszag_tang()
    for _ in range(steps):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    return g, sim


@pytest.fixture(scope="module")
def turbulent():
    g, sim = _turbulent_state()
    return g, sim, sim.get_fluxes()


def test_default_path_is_the_legacy_operator_bit_for_bit(turbulent):
    """Le chemin sans drapeau doit rester celui de la campagne Optuna.

    On reconstruit le score a la main avec l'operateur historique et on
    exige l'egalite exacte : une correction faite « au passage » dans les
    mappeurs ferait tomber ce test.
    """
    from Simulation.PhysToAngle import AngleMapper, _lohner_estimator

    _g, _sim, ps = turbulent
    vx, vy, Bx, By = ps["vx"], ps["vy"], ps["Bx"], ps["By"]

    def _norm(arr):
        mx = np.max(arr)
        return arr / mx if mx > 1e-12 else arr

    s_vort = _norm(np.abs(legacy_forward_curl_z(vx, vy)))
    s_div = _norm(np.abs(legacy_forward_divergence(vx, vy)))
    s_jz = _norm(np.abs(ps["Jz"]))
    s_loh = _norm(_lohner_estimator(np.sqrt(Bx ** 2 + By ** 2)))
    expected = np.sqrt((s_vort ** 2 + s_div ** 2 + s_jz ** 2 + s_loh ** 2) / 4.0)

    np.testing.assert_array_equal(AngleMapper.classical_score(ps), expected)


def test_fixed_curl_actually_changes_the_classical_score(turbulent):
    """Un drapeau accepte puis ignore serait indiscernable de son absence."""
    from Simulation.PhysToAngle import AngleMapper

    _g, _sim, ps = turbulent
    base = AngleMapper.classical_score(ps)
    fixed = AngleMapper.classical_score(ps, fixed_curl=True)
    assert not np.array_equal(base, fixed)
    assert np.max(np.abs(base - fixed)) > 1e-3, (
        f"ecart max {np.max(np.abs(base - fixed)):.3e} : le drapeau ne "
        "change pratiquement rien, donc il ne branche rien")


def test_fixed_curl_actually_changes_the_physical_score(turbulent):
    from Simulation.HamiltParams import PhysicalMapper

    g, _sim, ps = turbulent
    kw = dict(cs=1.0, nu=1 / 400.0, eta_mhd=1 / 400.0, dx=g.dx)
    base = PhysicalMapper(**kw).physical_score(ps)
    fixed = PhysicalMapper(fixed_curl=True, **kw).physical_score(ps)
    assert not np.array_equal(base, fixed)


def test_fixed_curl_reaches_the_v1_plaquette_coefficients(turbulent):
    """K_plaquettes est le canal ou passe le rotationnel : il doit bouger."""
    from Simulation.HamiltParams import PhysicalMapper
    from Simulation.PhysToAngle import AngleMapper

    g, sim, ps = turbulent
    score = AngleMapper.classical_score(ps)
    kw = dict(cs=1.0, nu=1 / 400.0, eta_mhd=1 / 400.0, dx=g.dx)
    base = PhysicalMapper(**kw).compute_coefficients(sim, score, ps, 0.35)
    fixed = PhysicalMapper(fixed_curl=True, **kw).compute_coefficients(
        sim, score, ps, 0.35)
    kb = np.asarray(base["K_plaquettes"], dtype=float)
    kf = np.asarray(fixed["K_plaquettes"], dtype=float)
    assert kb.shape == kf.shape and kb.size > 0
    assert not np.array_equal(kb, kf)


def test_fixed_curl_reaches_the_v2_plaquette_coefficients(turbulent):
    from Simulation.HamiltParams_v2 import PhysicalMapperV2
    from Simulation.PhysToAngle import AngleMapper

    g, sim, ps = turbulent
    score = AngleMapper.classical_score(ps)
    base = PhysicalMapperV2(dx=g.dx).compute_coefficients(sim, score, ps, 0.35)
    fixed = PhysicalMapperV2(dx=g.dx, fixed_curl=True).compute_coefficients(
        sim, score, ps, 0.35)
    kb = np.asarray(base["K_plaquettes"], dtype=float)
    kf = np.asarray(fixed["K_plaquettes"], dtype=float)
    assert not np.array_equal(kb, kf)


def test_the_mapper_selectors_dispatch_on_the_flag(turbulent):
    """`curl_z`/`divergence` doivent renvoyer l'une ou l'autre forme, pas un
    melange."""
    _g, _sim, ps = turbulent
    vx, vy = ps["vx"], ps["vy"]
    np.testing.assert_array_equal(curl_z(vx, vy, False),
                                  legacy_forward_curl_z(vx, vy))
    np.testing.assert_array_equal(curl_z(vx, vy, True),
                                  forward_curl_z(vx, vy))
    np.testing.assert_array_equal(divergence(vx, vy, False),
                                  legacy_forward_divergence(vx, vy))
    np.testing.assert_array_equal(divergence(vx, vy, True),
                                  forward_divergence(vx, vy))
