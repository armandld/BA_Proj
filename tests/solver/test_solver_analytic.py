"""Le solveur MHD, verifie sur des champs a reponse connue.

`tests/test_solver_convergence.py` mesure la convergence du solveur COMPLET
(pas de temps, projection, non-linearites), ce qui melange toutes les
sources d'erreur : un ordre observe de 1.2 n'y designe aucun coupable.

Ce fichier separe les couches et donne a chacune une reponse exacte :

  1. les operateurs de derivation, contre la derivee analytique ;
  2. le second membre complet, contre une evaluation SPECTRALE du meme
     systeme — sur un champ periodique lisse, les derivees spectrales sont
     exactes a l'arrondi machine, donc l'ecart mesure est exactement
     l'erreur de troncature de FD4 ;
  3. les invariants que le schema pretend preserver : div B = 0, energie
     sans dissipation, helicite croisee ;
  4. la projection, qui doit annuler la divergence, pas la reduire.

Un test qui compare une grandeur a un seuil ne teste pas la grandeur, il
teste le seuil. On mesure donc des ORDRES DE CONVERGENCE (rapports d'erreur
entre deux resolutions), qui ne dependent d'aucune constante choisie.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.grid import AXIS_X, AXIS_Y, PeriodicGrid  # noqa: E402
from Simulation.solver import MHDSolver                    # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
#  Outils : champs lisses periodiques et derivees spectrales exactes
# ═══════════════════════════════════════════════════════════════════════

def _mesh(N):
    g = PeriodicGrid(N)
    return g, g.X, g.Y, g.dx


def _spectral_grad(f, N):
    """Derivees exactes (a l'arrondi pres) d'un champ periodique lisse."""
    k = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")   # AXIS_X=0, AXIS_Y=1
    fh = np.fft.fft2(f)
    return (np.real(np.fft.ifft2(1j * KX * fh)),
            np.real(np.fft.ifft2(1j * KY * fh)))


def _spectral_laplacian(f, N):
    k = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    fh = np.fft.fft2(f)
    return np.real(np.fft.ifft2(-(KX ** 2 + KY ** 2) * fh))


def _orders(errs):
    """Ordres observes entre resolutions successives (facteur 2)."""
    return [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]


#  Champ test lisse, non separable, sans symetrie accidentelle.
def _smooth(X, Y):
    return (np.sin(X) * np.cos(2 * Y)
            + 0.3 * np.cos(3 * X + Y)
            + 0.2 * np.sin(X - 2 * Y))


# ═══════════════════════════════════════════════════════════════════════
#  1. Les operateurs de derivation
# ═══════════════════════════════════════════════════════════════════════

RES = (32, 64, 128)


def test_fd_grad_is_fourth_order():
    """(-f2 + 8f1 - 8f-1 + f-2)/(12h) doit converger en h^4."""
    ex, ey = [], []
    for N in RES:
        _g, X, Y, dx = _mesh(N)
        f = _smooth(X, Y)
        gx, gy = MHDSolver._fd_grad(f, dx)
        sx, sy = _spectral_grad(f, N)
        ex.append(float(np.max(np.abs(gx - sx))))
        ey.append(float(np.max(np.abs(gy - sy))))
    for name, e in (("d/dx", ex), ("d/dy", ey)):
        o = _orders(e)
        assert all(3.6 < v < 4.4 for v in o), (
            f"{name}: ordre attendu 4, mesure {o} (erreurs {e})")


def test_fd_grad_uses_the_declared_axes():
    """d/dx doit deriver selon AXIS_X, pas selon l'autre axe.

    Un operateur qui inverserait les deux resterait d'ordre 4 et passerait
    le test precedent : c'est le defaut qui a ete trouve dans les mappeurs.
    """
    N = 64
    _g, X, Y, dx = _mesh(N)
    f = np.sin(X)                      # ne depend QUE de x
    gx, gy = MHDSolver._fd_grad(f, dx)
    assert np.max(np.abs(gy)) < 1e-12, (
        "d/dy d'un champ independant de y n'est pas nul : les axes sont "
        f"inverses (max {np.max(np.abs(gy)):.3e})")
    #  Tolerance DERIVEE, pas choisie : l'erreur de troncature de la formule
    #  a 5 points vaut h^4/30 * |f^(5)|, soit h^4/30 pour sin. A N=64,
    #  h = 2pi/64 et h^4/30 ~ 3.1e-6. On prend trois fois cette valeur.
    h = dx
    tol = 3.0 * h ** 4 / 30.0
    np.testing.assert_allclose(gx, np.cos(X), rtol=0, atol=tol)


def test_fd_laplacian_is_fourth_order():
    errs = []
    for N in RES:
        _g, X, Y, dx = _mesh(N)
        f = _smooth(X, Y)
        errs.append(float(np.max(np.abs(
            MHDSolver._fd_laplacian(f, dx) - _spectral_laplacian(f, N)))))
    o = _orders(errs)
    assert all(3.6 < v < 4.4 for v in o), (
        f"laplacien : ordre attendu 4, mesure {o} (erreurs {errs})")


def test_fd_laplacian_matches_an_eigenfunction_exactly():
    """Controle absolu : lap(sin(x)sin(y)) = -2 sin(x)sin(y)."""
    N = 256
    _g, X, Y, dx = _mesh(N)
    f = np.sin(X) * np.sin(Y)
    lap = MHDSolver._fd_laplacian(f, dx)
    np.testing.assert_allclose(lap, -2.0 * f, rtol=2e-4, atol=2e-4)


# ═══════════════════════════════════════════════════════════════════════
#  2. Le second membre complet
# ═══════════════════════════════════════════════════════════════════════

def _fields(X, Y):
    """Champ lisse, div-free en v et en B (Taylor-Green + perturbation)."""
    vx = np.sin(X) * np.cos(Y)
    vy = -np.cos(X) * np.sin(Y)
    Bx = 0.4 * np.sin(2 * Y)
    By = 0.3 * np.sin(X)
    return vx, vy, Bx, By


def _spectral_rhs(vx, vy, Bx, By, N, nu, eta):
    """Le MEME systeme, derivees spectrales : reference exacte du RHS."""
    d = lambda f: _spectral_grad(f, N)
    g_vx_x, g_vx_y = d(vx)
    g_vy_x, g_vy_y = d(vy)
    g_Bx_x, g_Bx_y = d(Bx)
    g_By_x, g_By_y = d(By)

    g_vxx_x, _ = d(vx * vx)
    g_vxy_x, g_vxy_y = d(vx * vy)
    _, g_vyy_y = d(vy * vy)

    adv_x = 0.5 * (vx * g_vx_x + vy * g_vx_y + g_vxx_x + g_vxy_y)
    adv_y = 0.5 * (vx * g_vy_x + vy * g_vy_y + g_vxy_x + g_vyy_y)

    Jz = g_By_x - g_Bx_y
    Ez = vx * By - vy * Bx
    g_Ez_x, g_Ez_y = d(Ez)

    return (-adv_x - Jz * By + nu * _spectral_laplacian(vx, N),
            -adv_y + Jz * Bx + nu * _spectral_laplacian(vy, N),
            g_Ez_y + eta * _spectral_laplacian(Bx, N),
            -g_Ez_x + eta * _spectral_laplacian(By, N))


def test_the_rhs_is_fourth_order_in_space():
    """Le second membre complet doit converger en h^4 vers le RHS spectral.

    C'est LA mesure qui isole la discretisation spatiale : aucun pas de
    temps, aucune projection, aucune accumulation.
    """
    nu = eta = 1.0 / 400.0
    errs = []
    for N in RES:
        g, X, Y, dx = _mesh(N)
        sim = MHDSolver(g, dt=1e-4, Re=400, Rm=400)
        f = _fields(X, Y)
        got = sim._compute_rhs_fd(*f, dx, nu=nu, eta=eta)
        ref = _spectral_rhs(*f, N, nu, eta)
        errs.append(max(float(np.max(np.abs(a - b)))
                        for a, b in zip(got, ref)))
    o = _orders(errs)
    assert all(3.5 < v < 4.5 for v in o), (
        f"RHS : ordre spatial attendu 4, mesure {o} (erreurs {errs})")


def test_the_lorentz_force_vanishes_on_a_potential_field():
    """B = grad(phi) porte un courant nul, donc aucune force de Lorentz.

    Controle physique : avec Bx = cos(x), By = cos(y), Jz = 0 exactement,
    et les deux composantes de J x B doivent s'annuler.
    """
    N = 128
    g, X, Y, dx = _mesh(N)
    sim = MHDSolver(g, dt=1e-4, Re=1e12, Rm=1e12)
    Bx, By = np.cos(X), np.cos(Y)
    zero = np.zeros_like(X)
    rhs = sim._compute_rhs_fd(zero, zero, Bx, By, dx, nu=0.0, eta=0.0)
    assert np.max(np.abs(rhs[0])) < 1e-12
    assert np.max(np.abs(rhs[1])) < 1e-12


def test_a_uniform_field_is_a_steady_state():
    """Champs uniformes : toutes les derivees nulles, donc RHS nul."""
    N = 64
    g, X, Y, dx = _mesh(N)
    sim = MHDSolver(g, dt=1e-4, Re=400, Rm=400)
    one = np.ones_like(X)
    rhs = sim._compute_rhs_fd(0.7 * one, -0.2 * one, 0.3 * one, 0.5 * one,
                              dx, nu=1e-3, eta=1e-3)
    for r in rhs:
        assert np.max(np.abs(r)) < 1e-12


# ═══════════════════════════════════════════════════════════════════════
#  3. Les invariants
# ═══════════════════════════════════════════════════════════════════════

def _div(fx, fy, dx):
    gx, _ = MHDSolver._fd_grad(fx, dx)
    _, gy = MHDSolver._fd_grad(fy, dx)
    return gx + gy


def _fd_second_cross(f, dx):
    """d2f/dxdy discrete : l'echelle des deux termes qui doivent se compenser
    dans div(dB/dt)."""
    _, gy = MHDSolver._fd_grad(f, dx)
    gxy, _ = MHDSolver._fd_grad(gy, dx)
    return gxy


def test_the_induction_term_preserves_div_B_to_machine_precision():
    """d(div B)/dt = 0 EXACTEMENT au niveau discret, pas seulement en h^4.

    La forme rotationnelle rhs_B = (dEz/dy, -dEz/dx) combinee au meme
    operateur `_fd_grad` pour la divergence donne div(rhs_B) = d2Ez/dxdy -
    d2Ez/dydx. Les differences finies centrees COMMUTENT, donc ce residu
    est nul a l'arrondi machine, quelle que soit la resolution.

    Mesure : 1.1e-15, 4.7e-15, 1.9e-14 a N = 32, 64, 128 — il ne DECROIT
    pas en h^4, il est deja au plancher et croit legerement avec le nombre
    de points, comme toute somme d'arrondis. C'est un resultat plus fort
    que la convergence : le schema conserve div B par construction.
    """
    for N in RES:
        g, X, Y, dx = _mesh(N)
        sim = MHDSolver(g, dt=1e-4, Re=1e12, Rm=1e12)
        f = _fields(X, Y)
        rhs = sim._compute_rhs_fd(*f, dx, nu=0.0, eta=0.0)
        resid = float(np.max(np.abs(_div(rhs[2], rhs[3], dx))))
        #  Echelle de reference : l'amplitude des termes qui se compensent.
        scale = float(np.max(np.abs(_fd_second_cross(rhs[2], dx))))
        assert resid < 1e-10 * max(scale, 1.0), (
            f"N={N} : residu div(dB/dt) = {resid:.3e}, echelle {scale:.3e}. "
            "La forme rotationnelle devrait annuler ce terme exactement")


def test_div_B_stays_at_roundoff_over_a_long_integration():
    """L'invariant doit tenir dans le temps, pas seulement a l'instant 0.

    Un terme d'induction correct au premier pas mais qui laisserait fuir
    div B au fil des pas passerait le test precedent.
    """
    N = 64
    g, X, Y, dx = _mesh(N)
    sim = MHDSolver(g, dt=2e-3, Re=1e12, Rm=1e12)
    sim.vx, sim.vy, sim.Bx, sim.By = _fields(X, Y)
    b_rms = float(np.sqrt(np.mean(sim.Bx ** 2 + sim.By ** 2)))
    d0 = float(np.max(np.abs(_div(sim.Bx, sim.By, dx)))) / b_rms
    for _ in range(200):
        sim.vx, sim.vy, sim.Bx, sim.By = sim._rk4_step(
            sim.vx, sim.vy, sim.Bx, sim.By, dx, sim.dt, nu=0.0, eta=0.0)
    d1 = float(np.max(np.abs(_div(sim.Bx, sim.By, dx)))) / b_rms
    assert d0 < 1e-12, f"div B initiale non nulle : {d0:.3e}"
    assert d1 < 1e-9, (
        f"div B relative apres 200 pas : {d1:.3e} (depart {d0:.3e}) — "
        "l'invariant fuit")


def test_the_projection_annihilates_divergence_not_merely_reduces_it():
    """La projection spectrale doit rendre la divergence SPECTRALE nulle.

    On mesure la divergence avec le meme operateur (spectral) que celui que
    la projection utilise : sinon on mesure l'ecart entre deux operateurs,
    pas l'effet de la projection.
    """
    N = 64
    g, X, Y, dx = _mesh(N)
    vx = np.sin(X) * np.cos(Y) + 0.5 * np.sin(X)      # divergence non nulle
    vy = np.cos(X) * np.sin(Y)

    def spectral_div(fx, fy):
        gx, _ = _spectral_grad(fx, N)
        _, gy = _spectral_grad(fy, N)
        return gx + gy

    before = float(np.max(np.abs(spectral_div(vx, vy))))
    px, py = g.project_divergence_free(vx, vy)
    after = float(np.max(np.abs(spectral_div(px, py))))
    assert before > 0.1, f"le champ test doit avoir une divergence ({before:.3e})"
    assert after < 1e-12, (
        f"divergence apres projection {after:.3e} : la projection reduit au "
        "lieu d'annuler")


def test_the_projection_leaves_a_solenoidal_field_untouched():
    """Idempotence : projeter un champ deja solenoidal ne doit rien changer."""
    N = 64
    g, X, Y, dx = _mesh(N)
    vx = np.sin(X) * np.cos(Y)
    vy = -np.cos(X) * np.sin(Y)
    px, py = g.project_divergence_free(vx, vy)
    np.testing.assert_allclose(px, vx, rtol=0, atol=1e-12)
    np.testing.assert_allclose(py, vy, rtol=0, atol=1e-12)


def test_the_projection_preserves_the_mean():
    """Le mode k=0 ne doit pas bouger : c'est ce que K2[0,0]=1 protege."""
    N = 64
    g, X, Y, dx = _mesh(N)
    vx = np.sin(X) * np.cos(Y) + 1.7
    vy = np.cos(X) * np.sin(Y) - 0.4
    px, py = g.project_divergence_free(vx, vy)
    assert abs(px.mean() - vx.mean()) < 1e-12
    assert abs(py.mean() - vy.mean()) < 1e-12


def _energy(vx, vy, Bx, By, dx):
    return 0.5 * np.sum(vx ** 2 + vy ** 2 + Bx ** 2 + By ** 2) * dx ** 2


def _cross_helicity(vx, vy, Bx, By, dx):
    return np.sum(vx * Bx + vy * By) * dx ** 2


def test_energy_is_conserved_without_dissipation():
    """Sans nu ni eta, l'energie totale doit etre conservee.

    On exige que la derive DIMINUE quand le pas de temps diminue — un
    schema qui perdrait de l'energie par construction garderait une derive
    constante. Le seuil porte donc sur un RAPPORT, pas sur une valeur.
    """
    N = 64
    drifts = []
    for dt in (2e-3, 1e-3, 5e-4):
        g, X, Y, dx = _mesh(N)
        sim = MHDSolver(g, dt=dt, Re=1e12, Rm=1e12)
        sim.nu = sim.eta = 0.0
        sim.vx, sim.vy, sim.Bx, sim.By = _fields(X, Y)
        e0 = _energy(sim.vx, sim.vy, sim.Bx, sim.By, dx)
        n = int(round(0.2 / dt))
        for _ in range(n):
            sim.vx, sim.vy, sim.Bx, sim.By = sim._rk4_step(
                sim.vx, sim.vy, sim.Bx, sim.By, dx, dt, nu=0.0, eta=0.0)
        e1 = _energy(sim.vx, sim.vy, sim.Bx, sim.By, dx)
        drifts.append(abs(e1 - e0) / e0)
    assert drifts[0] > drifts[-1], (
        f"la derive d'energie ne diminue pas avec le pas de temps : {drifts}")
    assert drifts[-1] < 1e-6, (
        f"derive residuelle {drifts[-1]:.3e} au pas le plus fin")


def test_cross_helicity_drift_shrinks_with_the_timestep():
    """L'helicite croisee est un invariant ideal de la MHD 2-D."""
    N = 64
    drifts = []
    for dt in (2e-3, 1e-3, 5e-4):
        g, X, Y, dx = _mesh(N)
        sim = MHDSolver(g, dt=dt, Re=1e12, Rm=1e12)
        sim.vx, sim.vy, sim.Bx, sim.By = _fields(X, Y)
        h0 = _cross_helicity(sim.vx, sim.vy, sim.Bx, sim.By, dx)
        n = int(round(0.2 / dt))
        for _ in range(n):
            sim.vx, sim.vy, sim.Bx, sim.By = sim._rk4_step(
                sim.vx, sim.vy, sim.Bx, sim.By, dx, dt, nu=0.0, eta=0.0)
        h1 = _cross_helicity(sim.vx, sim.vy, sim.Bx, sim.By, dx)
        drifts.append(abs(h1 - h0) / (abs(h0) + 1e-30))
    assert drifts[0] > drifts[-1], (
        f"la derive d'helicite croisee ne diminue pas : {drifts}")


# ═══════════════════════════════════════════════════════════════════════
#  4. L'integrateur temporel
# ═══════════════════════════════════════════════════════════════════════

def test_rk4_is_fourth_order_in_time():
    """A grille FIXE, l'erreur temporelle doit decroitre en dt^4.

    Melanger raffinement spatial et temporel — ce que fait la campagne de
    convergence via `adapt_dt` — rend impossible d'attribuer un ordre. Ici
    la grille ne bouge pas.
    """
    N = 48
    g, X, Y, dx = _mesh(N)
    f0 = _fields(X, Y)
    T = 0.1

    def march(dt):
        s = MHDSolver(g, dt=dt, Re=1e12, Rm=1e12)
        s.vx, s.vy, s.Bx, s.By = tuple(np.array(a) for a in f0)
        for _ in range(int(round(T / dt))):
            s.vx, s.vy, s.Bx, s.By = s._rk4_step(
                s.vx, s.vy, s.Bx, s.By, dx, dt, nu=0.0, eta=0.0)
        return (s.vx, s.vy, s.Bx, s.By)

    ref = march(T / 3200)
    errs = [max(float(np.max(np.abs(a - b)))
                for a, b in zip(march(T / n), ref))
            for n in (50, 100, 200)]
    o = _orders(errs)
    assert all(3.5 < v < 4.6 for v in o), (
        f"RK4 : ordre temporel attendu 4, mesure {o} (erreurs {errs})")


def test_rk2_is_second_order_in_time():
    """Controle croise : le meme protocole doit donner 2 pour RK2.

    Sans lui, un protocole de mesure casse donnerait « 4 » a n'importe quoi.
    """
    N = 48
    g, X, Y, dx = _mesh(N)
    f0 = _fields(X, Y)
    T = 0.1

    def march(dt):
        s = MHDSolver(g, dt=dt, Re=1e12, Rm=1e12)
        s.vx, s.vy, s.Bx, s.By = tuple(np.array(a) for a in f0)
        for _ in range(int(round(T / dt))):
            s.vx, s.vy, s.Bx, s.By = s._rk2_step(
                s.vx, s.vy, s.Bx, s.By, dx, dt, nu=0.0, eta=0.0)
        return (s.vx, s.vy, s.Bx, s.By)

    ref = march(T / 6400)
    errs = [max(float(np.max(np.abs(a - b)))
                for a, b in zip(march(T / n), ref))
            for n in (50, 100, 200)]
    o = _orders(errs)
    assert all(1.6 < v < 2.4 for v in o), (
        f"RK2 : ordre temporel attendu 2, mesure {o} (erreurs {errs})")


# ═══════════════════════════════════════════════════════════════════════
#  5. adapt_dt et le garde-fou CFL
# ═══════════════════════════════════════════════════════════════════════

def test_adapt_dt_respects_the_requested_cfl():
    """dt doit satisfaire (|v|+|B|) dt / dx <= cfl demande."""
    N = 64
    g, X, Y, dx = _mesh(N)
    sim = MHDSolver(g, dt=1.0, Re=400, Rm=400)
    sim.vx, sim.vy, sim.Bx, sim.By = _fields(X, Y)
    for cfl in (0.1, 0.4, 0.8):
        dt = sim.adapt_dt(cfl_target=cfl)
        v_max = max(np.max(np.abs(sim.vx)), np.max(np.abs(sim.vy)))
        b_max = max(np.max(np.abs(sim.Bx)), np.max(np.abs(sim.By)))
        assert (v_max + b_max) * dt / dx <= cfl + 1e-12


def test_adapt_dt_also_respects_the_diffusive_limit():
    """A viscosite forte, c'est la diffusion qui doit borner dt."""
    N = 64
    g, X, Y, dx = _mesh(N)
    sim = MHDSolver(g, dt=1.0, Re=1.0, Rm=1.0)
    sim.vx, sim.vy, sim.Bx, sim.By = _fields(X, Y)
    dt = sim.adapt_dt(cfl_target=0.4)
    nu_max = max(sim.nu, sim.eta)
    assert dt <= 0.5 * 0.4 * dx ** 2 / nu_max + 1e-15


def test_check_cfl_detects_a_violation():
    """Le detecteur doit detecter — controle positif ET negatif."""
    N = 64
    g, X, Y, dx = _mesh(N)
    sim = MHDSolver(g, dt=1e-6, Re=400, Rm=400)
    sim.vx, sim.vy, sim.Bx, sim.By = _fields(X, Y)
    assert sim.check_cfl() < 1.0
    sim.dt = 10.0
    assert sim.check_cfl() > 1.0


def test_is_diverged_catches_nan_inf_and_blowup():
    N = 32
    g, X, Y, dx = _mesh(N)
    sim = MHDSolver(g, dt=1e-4, Re=400, Rm=400)
    sim.vx, sim.vy, sim.Bx, sim.By = _fields(X, Y)
    assert not sim.is_diverged()
    for bad in (np.nan, np.inf, 1e200):
        s = MHDSolver(g, dt=1e-4, Re=400, Rm=400)
        s.vx, s.vy, s.Bx, s.By = tuple(np.array(a) for a in _fields(X, Y))
        s.vx[0, 0] = bad
        assert s.is_diverged(), f"{bad} non detecte"


# ═══════════════════════════════════════════════════════════════════════
#  6. D-7 — le mode de Nyquist dans la projection
# ═══════════════════════════════════════════════════════════════════════

def _spectral_div_nyq(fx, fy, dx, zero_nyquist=True):
    n = fx.shape[0]
    k = np.fft.fftfreq(n, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    KX, KY = KX.copy(), KY.copy()
    if zero_nyquist:
        KX[n // 2, :] = 0.0
        KY[:, n // 2] = 0.0
    return np.real(np.fft.ifft2(1j * KX * np.fft.fft2(fx)
                                + 1j * KY * np.fft.fft2(fy)))


def test_the_projection_is_exact_on_a_noisy_field():
    """D-7 : le bruit excite le mode de Nyquist, que la projection ignorait.

    Pour un champ REEL de taille paire, +N/2 et -N/2 sont indiscernables et
    le coefficient de Fourier y est reel ; le multiplier par i*k le rend
    imaginaire pur, et le `np.real(ifft2(...))` final le jette. La
    divergence portee par ce mode traversait donc la projection intacte.

    Mesure avant correction : 6.5 % de l'energie de divergence y vivait, et
    projeter trois fois de suite donnait 5.05 -> 0.378 -> 0.270 -> 0.213 au
    lieu de zero.
    """
    g, X, Y, dx = _mesh(64)
    rng = np.random.default_rng(42)
    bx = 1.0 + 0.05 * rng.standard_normal((64, 64))
    by = 0.05 * rng.standard_normal((64, 64))
    before = float(np.max(np.abs(_spectral_div_nyq(bx, by, dx))))
    assert before > 1.0, "le champ test doit avoir une divergence franche"
    px, py = g.project_divergence_free(bx, by)
    after = float(np.max(np.abs(_spectral_div_nyq(px, py, dx))))
    assert after < 1e-12, f"divergence residuelle {after:.3e} sur champ bruite"


def test_the_projection_is_idempotent_on_a_noisy_field():
    """P(P(x)) = P(x) : c'est la definition d'une projection.

    Elle ne tenait pas avant D-7 — chaque application reduisait la
    divergence d'un facteur ~1.3 sans jamais l'annuler.
    """
    g, X, Y, dx = _mesh(64)
    rng = np.random.default_rng(7)
    bx = 1.0 + 0.05 * rng.standard_normal((64, 64))
    by = 0.05 * rng.standard_normal((64, 64))
    once = g.project_divergence_free(bx, by)
    twice = g.project_divergence_free(*once)
    np.testing.assert_allclose(twice[0], once[0], rtol=0, atol=1e-13)
    np.testing.assert_allclose(twice[1], once[1], rtol=0, atol=1e-13)


def test_the_projection_still_preserves_the_mean_on_a_noisy_field():
    """La correction du Nyquist ne doit pas toucher au mode k=0."""
    g, X, Y, dx = _mesh(64)
    rng = np.random.default_rng(3)
    bx = 1.3 + 0.05 * rng.standard_normal((64, 64))
    by = -0.4 + 0.05 * rng.standard_normal((64, 64))
    px, py = g.project_divergence_free(bx, by)
    assert abs(px.mean() - bx.mean()) < 1e-12
    assert abs(py.mean() - by.mean()) < 1e-12


def test_the_projection_leaves_a_pure_nyquist_field_alone():
    """Un champ porte par le seul mode de Nyquist n'a pas de divergence
    representable : la projection ne doit pas le detruire."""
    n = 64
    g, X, Y, dx = _mesh(n)
    checker = np.indices((n, n)).sum(axis=0) % 2 * 2.0 - 1.0   # (-1)^(i+j)
    px, py = g.project_divergence_free(checker, np.zeros_like(checker))
    assert float(np.max(np.abs(px - checker))) < 1e-10, (
        "le mode de Nyquist a ete altere par la projection")


def test_the_projection_stays_exact_on_smooth_fields():
    """Non-regression : la correction ne casse pas le cas lisse."""
    g, X, Y, dx = _mesh(64)
    vx = np.sin(X) * np.cos(Y) + 0.5 * np.sin(X)
    vy = np.cos(X) * np.sin(Y)
    px, py = g.project_divergence_free(vx, vy)
    assert float(np.max(np.abs(_spectral_div_nyq(px, py, dx)))) < 1e-12
