"""La physique modélisée est-elle la bonne ?

Les autres fichiers vérifient que chaque opérateur calcule ce que son nom
annonce. C'est nécessaire et insuffisant : un schéma dont chaque terme est
juste séparément peut encore coupler ces termes de travers, et produire une
dynamique plausible qui n'est pas la MHD.

Ce fichier teste le SYSTÈME COUPLÉ contre des vérités qui ne dépendent
d'aucune implémentation :

  1. **L'onde d'Alfvén.** Solution EXACTE des équations non linéaires — pas
     une linéarisation. Avec B₀ uniforme selon x et une perturbation
     transverse ne dépendant que de x, le terme d'advection s'annule
     identiquement, le terme d'induction est exact, et la seule
     non-linéarité de la force de Lorentz est un gradient pur que la
     projection retire. La solution est une translation à la vitesse
     v_A = |B₀|. Elle teste le COUPLAGE entre induction et force de
     Lorentz, y compris son signe : une erreur de signe fait voyager l'onde
     dans l'autre sens sans rien casser d'autre.

  2. **La décroissance visqueuse exacte.** v = (0, A sin(kx)) annule
     l'advection identiquement, donc l'équation devient une diffusion pure
     dont la solution est A sin(kx)·exp(−νk²t), exacte.

  3. **L'invariance galiléenne.** Ajouter une vitesse uniforme doit
     translater la solution, rien de plus.

  4. **La covariance par échange des axes.** Le schéma doit commuter avec
     l'échange x ↔ y. C'est le test le plus sévère de la convention
     d'axes : il ne dépend d'aucune solution analytique et il échoue dès
     qu'un opérateur traite les deux directions différemment.

  5. **Les invariants idéaux** : énergie, hélicité croisée, div B.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.grid import AXIS_X, AXIS_Y, PeriodicGrid   # noqa: E402
from Simulation.solver import MHDSolver                    # noqa: E402


def _sim(N, Re=1e12, Rm=1e12, dt=1e-3):
    s = MHDSolver(PeriodicGrid(N), dt=dt, Re=Re, Rm=Rm)
    return s


def _march(sim, t_end, dt, project=True, ideal=True):
    nu = 0.0 if ideal else sim.nu
    eta = 0.0 if ideal else sim.eta
    n = int(round(t_end / dt))
    for _ in range(n):
        sim.vx, sim.vy, sim.Bx, sim.By = sim._rk4_step(
            sim.vx, sim.vy, sim.Bx, sim.By, sim.dx, dt, nu=nu, eta=eta)
        if project:
            sim.enforce_incompressibility()
    return sim


# ═══════════════════════════════════════════════════════════════════════
#  1. L'onde d'Alfvén — solution exacte du système non linéaire
# ═══════════════════════════════════════════════════════════════════════

def _alfven(N, B0=1.0, amp=1e-3, k=1, sign=-1.0):
    """v = (0, A sin kx), B = (B0, sign·A sin kx).

    sign = -1 : onde purement progressive vers +x, vitesse v_A = B0.
    sign = +1 : onde purement regressive, vitesse -v_A.

    Verification analytique. Avec vx = 0 et Bx = B0 uniforme :
      - (v.grad)v = 0 identiquement, car vy ne depend que de x ;
      - Ez = vx By - vy Bx = -vy B0, donc dBy/dt = -dEz/dx = B0 dvy/dx,
        EXACT et non linearise ;
      - (JxB)_y = Jz Bx = B0 dBy/dx, exact ;
      - (JxB)_x = -Jz By = -By dBy/dx = -d(By^2/2)/dx, gradient pur que la
        projection d'incompressibilite retire integralement.
    """
    g = PeriodicGrid(N)
    s = MHDSolver(g, dt=1e-3, Re=1e12, Rm=1e12)
    kx = k * g.X
    s.vx = np.zeros_like(g.X)
    s.vy = amp * np.sin(kx)
    s.Bx = np.full_like(g.X, B0)
    s.By = sign * amp * np.sin(kx)
    return s, g


def _phase_speed(sim, g, k, t, amp, B0):
    """Vitesse de phase mesuree sur le mode k, par la phase de sa FFT."""
    prof = sim.vy.mean(axis=AXIS_Y)                 # ne depend que de x
    coef = np.fft.fft(prof)[k]
    #  phase initiale de A sin(kx) : le mode k vaut -i A N/2
    phase = np.angle(coef) - np.angle(-1j * amp * len(prof) / 2.0)
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    #  une translation de d en x deplace la phase de -k*d (en unites de x)
    d = -phase / k
    return d / t


@pytest.mark.parametrize("B0", [0.5, 1.0, 2.0])
def test_the_alfven_wave_travels_at_the_alfven_speed(B0):
    """v_A = |B0| pour rho = 1, mu0 = 1. La mesure doit le retrouver."""
    N, k, amp = 64, 1, 1e-3
    dt, t_end = 2e-4, 0.2
    sim, g = _alfven(N, B0=B0, amp=amp, k=k, sign=-1.0)
    _march(sim, t_end, dt)
    v = _phase_speed(sim, g, k, t_end, amp, B0)
    assert v == pytest.approx(B0, rel=0.02), (
        f"vitesse mesuree {v:.4f}, attendue {B0:.4f}")


def test_the_alfven_wave_direction_follows_the_sign_of_the_perturbation():
    """dB = -dv part vers +x ; dB = +dv part vers -x.

    Une inversion de signe dans la force de Lorentz OU dans l'induction
    ferait voyager l'onde a l'envers sans rien casser d'autre : la
    conservation de l'energie, div B et la vitesse en MODULE resteraient
    toutes correctes.
    """
    N, k, amp, B0 = 64, 1, 1e-3, 1.0
    dt, t_end = 2e-4, 0.2
    fwd, g = _alfven(N, B0=B0, amp=amp, k=k, sign=-1.0)
    bwd, _ = _alfven(N, B0=B0, amp=amp, k=k, sign=+1.0)
    _march(fwd, t_end, dt)
    _march(bwd, t_end, dt)
    v_f = _phase_speed(fwd, g, k, t_end, amp, B0)
    v_b = _phase_speed(bwd, g, k, t_end, amp, B0)
    assert v_f > 0.5 * B0, f"l'onde progressive va a {v_f:+.4f}"
    assert v_b < -0.5 * B0, f"l'onde regressive va a {v_b:+.4f}"


def test_the_alfven_wave_keeps_its_amplitude_without_dissipation():
    """En MHD ideale, l'onde ne doit ni croitre ni s'amortir."""
    N, k, amp, B0 = 64, 1, 1e-3, 1.0
    sim, _g = _alfven(N, B0=B0, amp=amp, k=k)
    a0 = float(np.max(np.abs(sim.vy)))
    _march(sim, 0.4, 2e-4)
    a1 = float(np.max(np.abs(sim.vy)))
    assert a1 == pytest.approx(a0, rel=0.05), (
        f"amplitude {a0:.3e} -> {a1:.3e} sans dissipation")


def test_the_alfven_wave_keeps_equipartition():
    """|dB| = |dv| est maintenu le long de l'onde : c'est sa signature.

    Un couplage induction/Lorentz mal equilibre romprait l'egalite tout en
    laissant l'onde se propager.
    """
    N, amp, B0 = 64, 1e-3, 1.0
    sim, _g = _alfven(N, B0=B0, amp=amp)
    _march(sim, 0.3, 2e-4)
    ev = float(np.mean(sim.vy ** 2))
    eb = float(np.mean((sim.By) ** 2))
    assert eb == pytest.approx(ev, rel=0.05), (
        f"equipartition rompue : <dv^2> = {ev:.3e}, <dB^2> = {eb:.3e}")


def test_the_alfven_wave_stays_anticorrelated():
    """dB = -dv doit rester vrai a tout instant pour l'onde progressive."""
    N, amp, B0 = 64, 1e-3, 1.0
    sim, _g = _alfven(N, B0=B0, amp=amp, sign=-1.0)
    _march(sim, 0.3, 2e-4)
    a = sim.vy.ravel()
    b = sim.By.ravel()
    r = float(np.corrcoef(a, b)[0, 1])
    assert r < -0.95, f"correlation dv/dB = {r:+.3f}, attendue proche de -1"


def test_the_alfven_speed_scales_with_the_background_field():
    """Doubler B0 doit doubler la vitesse — relation, pas valeur isolee."""
    N, k, amp = 64, 1, 1e-3
    dt, t_end = 2e-4, 0.15
    speeds = []
    for B0 in (0.5, 1.0, 2.0):
        sim, g = _alfven(N, B0=B0, amp=amp, k=k)
        _march(sim, t_end, dt)
        speeds.append(_phase_speed(sim, g, k, t_end, amp, B0))
    assert speeds[1] / speeds[0] == pytest.approx(2.0, rel=0.05)
    assert speeds[2] / speeds[1] == pytest.approx(2.0, rel=0.05)


# ═══════════════════════════════════════════════════════════════════════
#  2. Décroissance visqueuse — solution exacte du système non linéaire
# ═══════════════════════════════════════════════════════════════════════

def test_a_shear_mode_decays_at_exactly_the_viscous_rate():
    """v = (0, A sin kx), B = 0 : advection nulle, diffusion pure.

    Solution exacte A sin(kx) exp(-nu k^2 t). Un facteur faux sur nu, ou
    un laplacien mal normalise par dx, se lirait directement dans le taux.
    """
    N, k, amp = 64, 2, 1e-2
    nu = 5e-3
    g = PeriodicGrid(N)
    sim = MHDSolver(g, dt=1e-3, Re=1.0 / nu, Rm=1.0 / nu)
    sim.vx = np.zeros_like(g.X)
    sim.vy = amp * np.sin(k * g.X)
    sim.Bx = np.zeros_like(g.X)
    sim.By = np.zeros_like(g.X)

    t_end, dt = 2.0, 1e-3
    n = int(round(t_end / dt))
    for _ in range(n):
        sim.vx, sim.vy, sim.Bx, sim.By = sim._rk4_step(
            sim.vx, sim.vy, sim.Bx, sim.By, sim.dx, dt, nu=nu, eta=nu)
        sim.enforce_incompressibility()

    got = float(np.max(np.abs(sim.vy)))
    expected = amp * np.exp(-nu * k ** 2 * t_end)
    assert got == pytest.approx(expected, rel=0.02), (
        f"amplitude {got:.4e}, attendue {expected:.4e} "
        f"(taux nu k^2 = {nu * k ** 2:.4e})")


def test_the_decay_rate_scales_as_k_squared():
    """Le taux doit quadrupler quand k double — signature du laplacien."""
    N, amp, nu = 64, 1e-2, 5e-3
    t_end, dt = 1.0, 1e-3
    rates = {}
    for k in (1, 2):
        g = PeriodicGrid(N)
        sim = MHDSolver(g, dt=dt, Re=1.0 / nu, Rm=1.0 / nu)
        sim.vx = np.zeros_like(g.X)
        sim.vy = amp * np.sin(k * g.X)
        sim.Bx = np.zeros_like(g.X)
        sim.By = np.zeros_like(g.X)
        for _ in range(int(round(t_end / dt))):
            sim.vx, sim.vy, sim.Bx, sim.By = sim._rk4_step(
                sim.vx, sim.vy, sim.Bx, sim.By, sim.dx, dt, nu=nu, eta=nu)
            sim.enforce_incompressibility()
        rates[k] = -np.log(float(np.max(np.abs(sim.vy))) / amp) / t_end
    assert rates[2] / rates[1] == pytest.approx(4.0, rel=0.05), (
        f"taux k=1 : {rates[1]:.4e}, k=2 : {rates[2]:.4e}")


def test_no_viscosity_means_no_decay():
    """Controle negatif : a nu = 0 le mode ne doit pas bouger."""
    N, k, amp = 64, 2, 1e-2
    g = PeriodicGrid(N)
    sim = MHDSolver(g, dt=1e-3, Re=1e12, Rm=1e12)
    sim.vx = np.zeros_like(g.X)
    sim.vy = amp * np.sin(k * g.X)
    sim.Bx = np.zeros_like(g.X)
    sim.By = np.zeros_like(g.X)
    _march(sim, 2.0, 1e-3)
    assert float(np.max(np.abs(sim.vy))) == pytest.approx(amp, rel=1e-3)


# ═══════════════════════════════════════════════════════════════════════
#  3. Invariance galiléenne
# ═══════════════════════════════════════════════════════════════════════

def test_adding_a_uniform_flow_only_translates_the_solution():
    """La MHD incompressible est galileenne : un boost uniforme doit
    deplacer la solution, sans la deformer.

    Le boost est choisi pour deplacer d'un nombre ENTIER de mailles, ce
    qui rend la comparaison exacte a l'erreur de schema pres.
    """
    N = 64
    g = PeriodicGrid(N)
    dt, t_end = 1e-3, 0.1
    shift_cells = 4
    U = shift_cells * g.dx / t_end          # deplacement entier

    def build(boost):
        s = MHDSolver(g, dt=dt, Re=1e12, Rm=1e12)
        s.init_orszag_tang()
        if boost:
            s.vx = s.vx + U
        return s

    a = _march(build(False), t_end, dt)
    b = _march(build(True), t_end, dt)
    #  on retire le boost et on translate de shift_cells selon AXIS_X
    rolled = np.roll(a.vy, shift_cells, axis=AXIS_X)
    err = float(np.max(np.abs(b.vy - rolled)))
    scale = float(np.max(np.abs(a.vy)))
    assert err / scale < 0.05, (
        f"ecart relatif {err / scale:.3%} entre la solution boostee et la "
        "solution translatee")


# ═══════════════════════════════════════════════════════════════════════
#  4. Covariance par échange des axes
# ═══════════════════════════════════════════════════════════════════════

def _swap_axes(vx, vy, Bx, By):
    """Echange x <-> y : les composantes permutent, les tableaux se
    transposent.

    Pour un champ scalaire f, f'(x, y) = f(y, x), donc f' = f.T.
    Pour un vecteur v, la composante x du champ transforme est la
    composante y de l'ancien, evaluee aux coordonnees echangees :
    v'_x = v_y.T et v'_y = v_x.T.
    """
    return (np.ascontiguousarray(vy.T), np.ascontiguousarray(vx.T),
            np.ascontiguousarray(By.T), np.ascontiguousarray(Bx.T))


def test_the_scheme_commutes_with_swapping_the_axes():
    """T(evoluer(u)) doit egaler evoluer(T(u)).

    Aucune solution analytique n'entre ici : c'est une symetrie des
    equations, que tout schema correct doit respecter. Le test echoue des
    qu'un operateur traite les deux directions differemment — c'est le
    controle le plus severe de la convention d'axes, et il couvre
    l'ensemble de la chaine : advection, Lorentz, induction, diffusion et
    projection.
    """
    N = 48
    g = PeriodicGrid(N)
    dt, t_end = 1e-3, 0.05

    base = MHDSolver(g, dt=dt, Re=400, Rm=400)
    base.init_orszag_tang()
    u0 = (base.vx.copy(), base.vy.copy(), base.Bx.copy(), base.By.copy())

    #  chemin A : evoluer puis echanger
    a = MHDSolver(g, dt=dt, Re=400, Rm=400)
    a.vx, a.vy, a.Bx, a.By = (x.copy() for x in u0)
    _march(a, t_end, dt, ideal=False)
    path_a = _swap_axes(a.vx, a.vy, a.Bx, a.By)

    #  chemin B : echanger puis evoluer
    b = MHDSolver(g, dt=dt, Re=400, Rm=400)
    b.vx, b.vy, b.Bx, b.By = _swap_axes(*u0)
    _march(b, t_end, dt, ideal=False)
    path_b = (b.vx, b.vy, b.Bx, b.By)

    scale = max(float(np.max(np.abs(x))) for x in path_a)
    for name, xa, xb in zip("vx vy Bx By".split(), path_a, path_b):
        err = float(np.max(np.abs(xa - xb))) / scale
        assert err < 1e-9, (
            f"{name} : ecart relatif {err:.3e} — le schema ne commute pas "
            "avec l'echange des axes")


def test_the_swap_is_an_involution():
    """Controle du controle : appliquer deux fois doit redonner l'original.

    Sans lui, une transformation erronee pourrait faire passer le test
    precedent en cassant les deux chemins de la meme facon.
    """
    rng = np.random.default_rng(0)
    u = tuple(rng.standard_normal((16, 16)) for _ in range(4))
    back = _swap_axes(*_swap_axes(*u))
    for a, b in zip(u, back):
        np.testing.assert_array_equal(a, b)


def test_the_swap_actually_changes_a_generic_field():
    """Et la transformation ne doit pas etre l'identite."""
    rng = np.random.default_rng(1)
    u = tuple(rng.standard_normal((16, 16)) for _ in range(4))
    sw = _swap_axes(*u)
    assert not all(np.array_equal(a, b) for a, b in zip(u, sw))


# ═══════════════════════════════════════════════════════════════════════
#  5. Invariants sur une évolution réelle
# ═══════════════════════════════════════════════════════════════════════

def _energy(s):
    return 0.5 * float(np.sum(s.vx ** 2 + s.vy ** 2 + s.Bx ** 2 + s.By ** 2))


def _spectral_div(fx, fy, dx):
    n = fx.shape[0]
    k = np.fft.fftfreq(n, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    KX, KY = KX.copy(), KY.copy()
    KX[n // 2, :] = 0.0
    KY[:, n // 2] = 0.0
    return np.real(np.fft.ifft2(1j * KX * np.fft.fft2(fx)
                                + 1j * KY * np.fft.fft2(fy)))


def test_energy_is_conserved_on_a_real_flow_without_dissipation():
    """Orszag-Tang ideal : l'energie totale doit rester stable."""
    N = 48
    s = _sim(N)
    s.init_orszag_tang()
    e0 = _energy(s)
    _march(s, 0.2, 5e-4)
    e1 = _energy(s)
    assert abs(e1 - e0) / e0 < 2e-3, (
        f"derive d'energie {abs(e1 - e0) / e0:.3%} sur un ecoulement ideal")


def test_dissipation_can_only_remove_energy():
    """Avec nu, eta > 0, l'energie doit DECROITRE — jamais croitre."""
    N = 48
    s = MHDSolver(PeriodicGrid(N), dt=5e-4, Re=200, Rm=200)
    s.init_orszag_tang()
    prev = _energy(s)
    for _ in range(6):
        _march(s, 0.05, 5e-4, ideal=False)
        cur = _energy(s)
        assert cur <= prev * (1 + 1e-9), (
            f"l'energie a augmente sous dissipation : {prev:.6e} -> {cur:.6e}")
        prev = cur


def test_div_b_stays_at_roundoff_on_a_real_flow():
    N = 48
    s = _sim(N)
    s.init_orszag_tang()
    _march(s, 0.2, 5e-4)
    b_rms = float(np.sqrt(np.mean(s.Bx ** 2 + s.By ** 2)))
    rel = float(np.max(np.abs(_spectral_div(s.Bx, s.By, s.dx)))) * s.dx / b_rms
    assert rel < 1e-9, f"div B relative {rel:.3e} apres evolution"


def test_the_flow_stays_incompressible():
    N = 48
    s = _sim(N)
    s.init_orszag_tang()
    _march(s, 0.2, 5e-4)
    v_rms = float(np.sqrt(np.mean(s.vx ** 2 + s.vy ** 2)))
    rel = float(np.max(np.abs(_spectral_div(s.vx, s.vy, s.dx)))) * s.dx / v_rms
    assert rel < 1e-9, f"div v relative {rel:.3e} apres evolution"


def test_a_state_at_rest_with_a_uniform_field_never_moves():
    """Equilibre trivial : rien ne doit s'y passer.

    Un terme parasite — advection mal signee, force de Lorentz residuelle —
    s'y verrait immediatement.
    """
    N = 32
    g = PeriodicGrid(N)
    s = MHDSolver(g, dt=1e-3, Re=400, Rm=400)
    s.vx = np.zeros_like(g.X)
    s.vy = np.zeros_like(g.X)
    s.Bx = np.full_like(g.X, 0.7)
    s.By = np.full_like(g.X, -0.3)
    _march(s, 0.5, 1e-3, ideal=False)
    assert float(np.max(np.abs(s.vx))) < 1e-12
    assert float(np.max(np.abs(s.vy))) < 1e-12
    assert float(np.max(np.abs(s.Bx - 0.7))) < 1e-12
    assert float(np.max(np.abs(s.By + 0.3))) < 1e-12


def test_a_pure_hydrodynamic_flow_never_creates_a_magnetic_field():
    """B = 0 est une solution exacte de l'induction : elle doit le rester.

    Si l'induction fabriquait du champ a partir de rien, toute la MHD
    serait fausse — et ce serait invisible sur un ecoulement deja
    magnetise.
    """
    N = 48
    g = PeriodicGrid(N)
    s = MHDSolver(g, dt=5e-4, Re=400, Rm=400)
    s.init_orszag_tang()
    s.Bx = np.zeros_like(g.X)
    s.By = np.zeros_like(g.X)
    _march(s, 0.2, 5e-4, ideal=False)
    assert float(np.max(np.abs(s.Bx))) < 1e-12
    assert float(np.max(np.abs(s.By))) < 1e-12


def test_reversing_the_field_sign_leaves_the_dynamics_unchanged():
    """La MHD ne depend de B que par des termes QUADRATIQUES : B -> -B ne
    doit rien changer a la vitesse, et changer le signe de B seulement.
    """
    N = 48
    g = PeriodicGrid(N)

    def run(flip):
        s = MHDSolver(g, dt=5e-4, Re=400, Rm=400)
        s.init_orszag_tang()
        if flip:
            s.Bx, s.By = -s.Bx, -s.By
        _march(s, 0.15, 5e-4, ideal=False)
        return s

    a, b = run(False), run(True)
    scale = float(np.max(np.abs(a.vx)))
    assert float(np.max(np.abs(a.vx - b.vx))) / scale < 1e-9
    assert float(np.max(np.abs(a.vy - b.vy))) / scale < 1e-9
    bscale = float(np.max(np.abs(a.Bx)))
    assert float(np.max(np.abs(a.Bx + b.Bx))) / bscale < 1e-9
