"""Les dix conditions initiales du solveur, contrôlées une par une.

Aucune n'avait de test. Ce sont pourtant elles qui définissent la physique
que toute l'étude mesure : un scénario mal initialisé ne plante pas, il
produit une simulation plausible d'autre chose.

Chaque scénario est vérifié sur ce qui doit être vrai de LUI et pas des
autres — un tourbillon doit tourner, une nappe de courant doit porter du
courant, un mode de déchirement doit avoir un champ qui change de signe.
Sans ces contrôles spécifiques, un test générique « les champs sont finis »
passerait sur un scénario qui aurait été remplacé par un autre.
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

N = 64

SCENARIOS = [
    "kelvin_helmholtz", "orszag_tang", "magnetic_twist", "noisy_uniform",
    "harris_tearing", "ghost_twisting", "lamb_oseen_vortex",
    "double_tearing", "mhd_rotor", "island_coalescence",
]
FIELDS = ("vx", "vy", "Bx", "By")


def _sim(scenario, n=N):
    s = MHDSolver(PeriodicGrid(n), dt=1e-4, Re=400, Rm=400)
    getattr(s, "init_" + scenario)()
    return s


def _curl(fx, fy, dx):
    return ((np.roll(fy, -1, axis=AXIS_X) - fy)
            - (np.roll(fx, -1, axis=AXIS_Y) - fx)) / dx


def _div(fx, fy, dx):
    return ((np.roll(fx, -1, axis=AXIS_X) - fx)
            + (np.roll(fy, -1, axis=AXIS_Y) - fy)) / dx


# ═══════════════════════════════════════════════════════════════════════
#  1. Contrat commun à tous les scénarios
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_fills_all_four_fields_at_the_grid_size(scenario):
    s = _sim(scenario)
    for k in FIELDS:
        arr = getattr(s, k)
        assert arr.shape == (N, N), f"{k} a la forme {arr.shape}"
        assert arr.dtype.kind == "f", f"{k} n'est pas flottant"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_is_finite(scenario):
    s = _sim(scenario)
    for k in FIELDS:
        assert np.all(np.isfinite(getattr(s, k))), f"{k} contient NaN ou inf"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_no_scenario_is_trivially_empty(scenario):
    """Un scenario ou tout serait nul passerait tous les tests generiques."""
    s = _sim(scenario)
    total = sum(float(np.max(np.abs(getattr(s, k)))) for k in FIELDS)
    assert total > 1e-6, f"{scenario} ne pose aucun champ (somme {total:.3e})"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_no_scenario_is_uniform_in_every_field(scenario):
    """Un champ partout constant ne definit aucune structure a raffiner."""
    s = _sim(scenario)
    spreads = [float(np.ptp(getattr(s, k))) for k in FIELDS]
    assert max(spreads) > 1e-6, (
        f"{scenario} : tous les champs sont uniformes {spreads}")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_is_bounded(scenario):
    """Des amplitudes demesurees feraient exploser le CFL des le premier pas."""
    s = _sim(scenario)
    for k in FIELDS:
        mx = float(np.max(np.abs(getattr(s, k))))
        assert mx < 1e3, f"{scenario}.{k} atteint {mx:.3e}"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_is_deterministic(scenario):
    """Deux initialisations successives doivent coincider bit-a-bit.

    Les scenarios bruites doivent fixer leur graine : sans cela, deux
    executions de la meme campagne ne seraient pas comparables et la
    variance mesuree melangerait le bruit d'initialisation.
    """
    a, b = _sim(scenario), _sim(scenario)
    for k in FIELDS:
        np.testing.assert_array_equal(
            getattr(a, k), getattr(b, k),
            err_msg=f"{scenario}.{k} n'est pas reproductible")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_produces_a_finite_timestep(scenario):
    s = _sim(scenario)
    dt = s.adapt_dt(cfl_target=0.4)
    assert np.isfinite(dt) and dt > 0.0
    assert s.check_cfl() <= 0.4 + 1e-12


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_survives_a_few_steps(scenario):
    """Une condition initiale qui diverge en cinq pas n'est pas exploitable."""
    s = _sim(scenario, n=32)
    for _ in range(5):
        s.adapt_dt(cfl_target=0.4)
        s.step_full(record_stats=False)
    assert not s.is_diverged(), f"{scenario} diverge en 5 pas"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_yields_a_usable_flux_dict(scenario):
    s = _sim(scenario, n=32)
    f = s.get_fluxes()
    assert set(f) >= {"vx", "vy", "Bx", "By", "Jz"}
    for k, v in f.items():
        assert np.all(np.isfinite(v)), f"{scenario}.{k} non fini"


def _spectral_div(fx, fy, dx):
    """Divergence SPECTRALE — l'operateur que la projection annule.

    Mesurer avec une difference avant testerait l'ecart entre deux
    operateurs, pas l'effet de la projection : sur un champ bruite, les
    deux different fortement aux hautes frequences.

    La derivee au mode de Nyquist est annulee, comme dans la projection :
    pour un champ reel de taille paire, +N/2 et -N/2 sont indiscernables et
    i*k y produit un imaginaire pur que `np.real` jetterait.
    """
    n = fx.shape[0]
    k = np.fft.fftfreq(n, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    KX, KY = KX.copy(), KY.copy()
    KX[n // 2, :] = 0.0
    KY[:, n // 2] = 0.0
    return np.real(np.fft.ifft2(1j * KX * np.fft.fft2(fx)
                                + 1j * KY * np.fft.fft2(fy)))


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_magnetic_field_starts_solenoidal(scenario):
    """div B = 0 est une contrainte physique, pas une preference."""
    s = _sim(scenario)
    dx = s.grid.dx
    b_rms = float(np.sqrt(np.mean(s.Bx ** 2 + s.By ** 2))) + 1e-30
    rel = float(np.max(np.abs(_spectral_div(s.Bx, s.By, dx)))) * dx / b_rms
    assert rel < 1e-10, (
        f"{scenario} : div B spectrale relative {rel:.3e} des "
        "l'initialisation")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_velocity_field_starts_solenoidal(scenario):
    """Meme contrainte sur v : le solveur suppose l'incompressibilite."""
    s = _sim(scenario)
    dx = s.grid.dx
    v_rms = float(np.sqrt(np.mean(s.vx ** 2 + s.vy ** 2)))
    if v_rms < 1e-12:
        pytest.skip(f"{scenario} part au repos")
    rel = float(np.max(np.abs(_spectral_div(s.vx, s.vy, dx)))) * dx / v_rms
    assert rel < 1e-10, f"{scenario} : div v spectrale relative {rel:.3e}"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenarios_differ_from_one_another(scenario):
    """Deux scenarios ne doivent pas produire le meme etat.

    Si un `init_` appelait par erreur un autre, rien ne le signalerait.
    """
    ref = _sim(scenario, n=32)
    for other in SCENARIOS:
        if other == scenario:
            continue
        o = _sim(other, n=32)
        same = all(np.array_equal(getattr(ref, k), getattr(o, k))
                   for k in FIELDS)
        assert not same, f"{scenario} et {other} donnent le meme etat"


# ═══════════════════════════════════════════════════════════════════════
#  2. Ce qui est propre à chaque scénario
# ═══════════════════════════════════════════════════════════════════════

def test_orszag_tang_is_the_canonical_vortex():
    """Orszag-Tang : v et B sinusoidaux, moyennes nulles, tourbillons.

    C'est le scenario de reference de la MHD 2-D ; ses moyennes doivent
    etre nulles par symetrie.
    """
    s = _sim("orszag_tang")
    for k in FIELDS:
        assert abs(float(np.mean(getattr(s, k)))) < 1e-10, (
            f"moyenne de {k} non nulle")
    #  vorticite non triviale, des deux signes
    w = _curl(s.vx, s.vy, s.grid.dx)
    assert w.max() > 0.1 and w.min() < -0.1, "pas de tourbillons des deux sens"


def test_kelvin_helmholtz_has_two_counter_streaming_layers():
    """KH : un cisaillement, donc vx change de signe selon y."""
    s = _sim("kelvin_helmholtz")
    profile = s.vx.mean(axis=AXIS_X)          # moyenne le long de x
    assert profile.max() > 0.05 and profile.min() < -0.05, (
        "aucun contre-courant : le cisaillement n'est pas pose")
    #  et le cisaillement est perpendiculaire au courant moyen
    assert float(np.max(np.abs(s.vx))) > float(np.max(np.abs(s.vy)))


def test_kelvin_helmholtz_noise_is_reproducible_and_small():
    a, b = _sim("kelvin_helmholtz"), _sim("kelvin_helmholtz")
    np.testing.assert_array_equal(a.vy, b.vy)
    assert float(np.max(np.abs(a.vy))) < float(np.max(np.abs(a.vx))), (
        "la perturbation domine l'ecoulement de base")


def test_mhd_rotor_spins_a_dense_core():
    """Le rotor : rotation concentree au centre, calme au bord."""
    s = _sim("mhd_rotor")
    n = s.grid.N
    w = _curl(s.vx, s.vy, s.grid.dx)
    c = int(n * 0.1)
    core = float(np.mean(np.abs(w[n // 2 - c:n // 2 + c, n // 2 - c:n // 2 + c])))
    edge = float(np.mean(np.abs(w[:c, :c])))
    assert core > 5 * edge, (
        f"le coeur ne tourne pas plus que le bord ({core:.3e} vs {edge:.3e})")


def test_mhd_rotor_rotation_has_a_definite_sense():
    """Une rotation solide a une vorticite de signe constant en son coeur."""
    s = _sim("mhd_rotor")
    n = s.grid.N
    c = int(n * 0.08)
    core = _curl(s.vx, s.vy, s.grid.dx)[n // 2 - c:n // 2 + c,
                                        n // 2 - c:n // 2 + c]
    assert abs(float(np.mean(np.sign(core)))) > 0.9, (
        "le coeur du rotor ne tourne pas dans un sens defini")


def test_harris_tearing_reverses_its_field_and_carries_a_current():
    """Nappe de Harris : B change de signe, donc Jz est concentre."""
    s = _sim("harris_tearing")
    profile = s.Bx.mean(axis=AXIS_X)
    assert profile.max() > 0.2 and profile.min() < -0.2, (
        "le champ ne s'inverse pas : ce n'est pas une nappe de Harris")
    jz = s.get_fluxes()["Jz"]
    assert float(np.max(np.abs(jz))) > 1e-2, "aucune densite de courant"


def test_double_tearing_has_two_current_sheets():
    """Deux inversions, donc deux maxima de |Jz| separes."""
    s = _sim("double_tearing")
    jz = np.abs(s.get_fluxes()["Jz"]).mean(axis=AXIS_X)
    thr = 0.5 * jz.max()
    above = jz > thr
    #  compte les groupes contigus (avec enroulement periodique)
    groups = int(np.sum(np.diff(above.astype(int)) == 1)) + int(above[0])
    assert groups >= 2, f"{groups} nappe(s) de courant, attendu au moins 2"


def test_lamb_oseen_vortex_is_centred_and_axisymmetric():
    """Tourbillon de Lamb-Oseen : |v| ne depend que du rayon."""
    s = _sim("lamb_oseen_vortex")
    g = s.grid
    speed = np.sqrt(s.vx ** 2 + s.vy ** 2)
    r = np.sqrt((g.X - np.pi) ** 2 + (g.Y - np.pi) ** 2)
    #  a rayon comparable, la vitesse doit varier peu
    band = (r > 0.9) & (r < 1.1)
    assert band.sum() > 20
    v = speed[band]
    assert float(np.std(v) / (np.mean(v) + 1e-30)) < 0.35, (
        "le tourbillon n'est pas axisymetrique")
    #  et la vitesse s'annule au centre
    assert speed[g.N // 2, g.N // 2] < 0.3 * float(speed.max())


def test_island_coalescence_carries_magnetic_islands():
    """Des ilots magnetiques : det(grad B) doit prendre les deux signes.

    det < 0 designe un point X (reconnexion), det > 0 un point O (centre
    d'ilot). Les deux doivent etre presents.
    """
    from Simulation.HamiltParams import PhysicalMapper

    s = _sim("island_coalescence")
    det = PhysicalMapper._compute_det_jacobian_B(s.Bx, s.By, s.grid.dx)
    inner = det[2:-2, 2:-2]
    assert inner.min() < 0.0 < inner.max(), (
        "aucune paire point X / point O dans la configuration d'ilots")


def test_noisy_uniform_is_a_perturbed_uniform_field():
    """Champ quasi uniforme : ecart-type petit devant la moyenne."""
    s = _sim("noisy_uniform")
    b_mean = float(np.mean(np.sqrt(s.Bx ** 2 + s.By ** 2)))
    b_std = float(np.std(np.sqrt(s.Bx ** 2 + s.By ** 2)))
    assert b_mean > 0.5, "le champ de fond a disparu"
    assert b_std < 0.5 * b_mean, f"le bruit domine ({b_std:.3f} vs {b_mean:.3f})"


def test_noisy_uniform_honours_its_seed():
    """Le seul scenario a graine explicite : deux graines, deux etats."""
    a = MHDSolver(PeriodicGrid(32), dt=1e-4, Re=400, Rm=400)
    a.init_noisy_uniform(seed=1)
    b = MHDSolver(PeriodicGrid(32), dt=1e-4, Re=400, Rm=400)
    b.init_noisy_uniform(seed=2)
    assert not np.array_equal(a.Bx, b.Bx), "la graine ne change rien"
    c = MHDSolver(PeriodicGrid(32), dt=1e-4, Re=400, Rm=400)
    c.init_noisy_uniform(seed=1)
    np.testing.assert_array_equal(a.Bx, c.Bx)


@pytest.mark.parametrize("twist", [np.pi / 3, np.pi / 2, np.pi])
def test_magnetic_twist_sweeps_exactly_the_requested_angle(twist):
    """D-6 corrige : la direction de B tourne de `twist_angle`.

    Le scenario posait auparavant B = (B0 cos alpha, B0 sin alpha), dont
    la divergence vaut alpha'(y) cos(alpha) — mesuree 2.62 — et dont toute
    la composante By etait un pur gradient. `enforce_incompressibility`
    l'annulait donc integralement : |By| tombait de 0.707 a 1.6e-6 et
    l'amplitude d'angle a 6.4e-7. Le scenario ne posait AUCUNE torsion, et
    servait pourtant de classe dans `study/pipeline/dns_extension.py`.

    La construction actuelle — Bx variable, By constant — est solenoidale
    par construction, donc la projection ne la touche pas.
    """
    s = MHDSolver(PeriodicGrid(64), dt=1e-4, Re=400, Rm=400)
    s.init_magnetic_twist(twist_angle=twist)
    ang = np.arctan2(s.By, s.Bx).mean(axis=AXIS_X)
    assert float(np.ptp(ang)) == pytest.approx(twist, rel=1e-3), (
        f"balayage {np.ptp(ang):.4f} au lieu de {twist:.4f}")


def test_magnetic_twist_is_exactly_solenoidal_before_projection():
    """La construction ne doit RIEN devoir a la projection.

    On verifie que la divergence discrete est nulle a l'arrondi : si elle
    ne l'etait pas, la projection modifierait le champ pose et le scenario
    ne serait plus celui qu'on croit.
    """
    s = MHDSolver(PeriodicGrid(64), dt=1e-4, Re=400, Rm=400)
    s.init_magnetic_twist()
    d = ((np.roll(s.Bx, -1, axis=AXIS_X) - s.Bx)
         + (np.roll(s.By, -1, axis=AXIS_Y) - s.By))
    assert float(np.max(np.abs(d))) < 1e-15


def test_magnetic_twist_keeps_a_nonzero_guide_component():
    """|By| doit survivre : c'est lui qui porte la torsion."""
    s = _sim("magnetic_twist")
    assert float(np.max(np.abs(s.By))) > 0.5, (
        f"|By| = {np.max(np.abs(s.By)):.3e} : la composante guide a disparu")


def test_ghost_twisting_is_localised():
    """La torsion « fantome » doit etre concentree, pas globale."""
    s = _sim("ghost_twisting")
    energy = s.vx ** 2 + s.vy ** 2 + s.Bx ** 2 + s.By ** 2
    assert float(energy.max()) > 3 * float(np.median(energy)), (
        "la structure n'est pas localisee")


# ═══════════════════════════════════════════════════════════════════════
#  3. Le scan de raffinement voit ces structures
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_classical_score_is_defined_and_bounded_on_every_scenario(scenario):
    """Le score doit rester dans [0, 1] quelle que soit la physique posee."""
    from Simulation.PhysToAngle import AngleMapper

    s = _sim(scenario, n=32)
    sc = AngleMapper.classical_score(s.get_fluxes())
    assert sc.shape == (32, 32)
    assert np.all(np.isfinite(sc))
    assert sc.min() >= 0.0 and sc.max() <= 1.0


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_classical_score_discriminates_within_every_scenario(scenario):
    """Un score constant ne designerait aucune region a raffiner.

    C'est le controle qui manquait : un scenario dont le score serait plat
    rendrait l'AMR sans objet, et rien ne le disait.
    """
    from Simulation.PhysToAngle import AngleMapper

    s = _sim(scenario, n=32)
    sc = AngleMapper.classical_score(s.get_fluxes())
    assert float(np.ptp(sc)) > 1e-3, (
        f"{scenario} : score plat (amplitude {np.ptp(sc):.3e})")
