"""D-72 — T14 doit mesurer la contrainte solenoidale avec l'operateur qui la
GARANTIT, pas avec celui qui la garantissait autrefois.

`study/h1_solver/h1_solver_convergence.py` (T14) suit `max|div B| / rms|B|`
le long de chaque trajectoire et en fait son critere d'acceptation
(`ALL CHECKS`, stocke en `all_checks_pass`, lu par le master table). Il
mesurait cette divergence avec `dns_validation.div_B`, qui est SPECTRALE —
sa docstring l'assume : « same convention as the solver's FFT projection ».

Cette convention n'est plus celle du solveur depuis D-25 :
`MHDSolver.PROJECT_B = False`, B n'est plus projete spectralement. Il est
solenoidal AUX DIFFERENCES FINIES par construction — l'induction est ecrite
en forme rotationnelle `rhs_B = (dEz/dy, -dEz/dx)`, dont la divergence FD4
vaut `d2Ez/dxdy - d2Ez/dydx`, exactement nulle puisque les decalages de
`np.roll` commutent.

Mesurer un champ FD-solenoidal avec un operateur spectral ne mesure pas la
contrainte : cela mesure l'ecart entre les deux operateurs.

Le champ qui SEPARE : `mhd_rotor`, N=32, t_end=0.05 (5 pas). Spectral
6.0470e-01, FD4 assorti 2.0905e-15 — onze ordres de grandeur, sur un champ
dont la contrainte est respectee. Sur la configuration publiee de T14
(orszag_tang, 32/64/128, t=0.5), le maximum spectral vaut 3.9029e-02 contre
2.0266e-14 assorti, et `all_checks_pass` bascule True -> False contre le
seuil de 1e-3 — alors que `RESULTS.md` publie « entre 5.6e-15 et 8.0e-14 —
machine precision ».
"""

import importlib.util
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_T14_SCRIPT = os.path.join(
    _REPO_ROOT, "study", "h1_solver", "h1_solver_convergence.py")

# Le champ qui separe, et le moins cher qui separe : 5 pas.
_SCENARIO, _N, _T_END = "mhd_rotor", 32, 0.05

# Seuil d'acceptation de T14 lui-meme (`ok = all(divB_max <= 1e-3)`).
_T14_TOLERANCE = 1e-3
# Ce que la contrainte vaut reellement, mesure : 2.09e-15 sur cette
# configuration, 2.0266e-14 au pire sur la configuration publiee.
_MACHINE_PRECISION = 1e-12


def _spectral_divergence(Bx, By, dx):
    N = Bx.shape[0]
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    return np.real(np.fft.ifft2(
        1j * KX * np.fft.fft2(Bx) + 1j * KY * np.fft.fft2(By)))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def t14():
    return _load(_T14_SCRIPT, "t14_divb_mod")


@pytest.fixture(scope="module")
def trajectory(t14):
    """Un etat produit par le solveur TEL QU'IL EST, plus la trace de T14."""
    record = []
    vx, vy, Bx, By, _t, _k = t14.evolve_to(
        _SCENARIO, _N, 400, _T_END, cfl=0.4, record=record)
    assert record, "balayage vide : aucun pas n'a ete integre"
    return dict(vx=vx, vy=vy, Bx=Bx, By=By,
                dx=2 * np.pi / _N, record=record)


def test_t14_reports_the_constraint_at_machine_precision(t14, trajectory):
    """Ce que T14 CONSIGNE doit etre la contrainte, pas l'ecart de stencils.

    Echoue sur la version d'avant D-72 : la trace spectrale atteint
    6.0470e-01 sur cette configuration, soit 600x le seuil d'acceptation de
    T14 lui-meme, pour un champ dont la contrainte est respectee.
    """
    worst = max(r["divB"] for r in trajectory["record"])
    assert worst <= _MACHINE_PRECISION, (
        f"T14 consigne max|div B|/rms|B| = {worst:.4e} sur {_SCENARIO} "
        f"N={_N} — la contrainte que le solveur garantit vaut 2.09e-15 ici. "
        "Un operateur non assorti mesure l'ecart entre deux stencils, pas la "
        "contrainte (D-72)")


def test_the_reported_value_stays_below_t14_own_acceptance_threshold(
        t14, trajectory):
    """Le verdict `ALL CHECKS` doit refleter la contrainte, pas la resolution.

    Avant D-72 ce meme verdict basculait a False sur cette configuration, et
    dependait de N sur la configuration publiee (N=32 3.9029e-02, N=64
    4.5675e-03, N=128 2.3103e-04) : la validation passait ou echouait selon
    la RESOLUTION.
    """
    worst = max(r["divB"] for r in trajectory["record"])
    assert worst <= _T14_TOLERANCE, (
        f"ALL CHECKS basculerait a False : {worst:.4e} > {_T14_TOLERANCE:.0e}")


def test_the_matched_operator_annihilates_the_divergence_of_the_induction_rhs(
        t14):
    """La garantie, verifiee sur l'objet meme qui la porte : le second membre.

    `rhs_B` sort de `_compute_rhs_fd`. L'operateur que T14 emploie doit
    annuler SA divergence — c'est la definition d'etre assorti. On
    l'interroge par le comportement, jamais par le texte du source.

    L'operateur spectral, lui, ne l'annule pas : c'est ce qui separe les
    deux hypotheses, et sans cette branche le test passerait meme si T14
    etait revenu au spectral.
    """
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver

    N = 32
    dx = 2 * np.pi / N
    sim = MHDSolver(PeriodicGrid(N), dt=1e-3, Re=400, Rm=400)
    sim.init_mhd_rotor()
    rhs = sim._compute_rhs_fd(sim.vx, sim.vy, sim.Bx, sim.By, dx,
                              nu=0.0, eta=0.0)
    rhs_Bx, rhs_By = rhs[2], rhs[3]

    scale = float(np.max(np.abs(rhs_Bx))) + float(np.max(np.abs(rhs_By)))
    assert scale > 0, "second membre magnetique identiquement nul"

    matched = float(np.max(np.abs(
        t14.div_B_matched(rhs_Bx, rhs_By, dx)))) / scale
    spectral = float(np.max(np.abs(
        _spectral_divergence(rhs_Bx, rhs_By, dx)))) / scale

    assert matched < 1e-12, (
        f"l'operateur de T14 n'annule pas div(rhs_B) : {matched:.3e}. Il "
        "n'est donc pas celui qui construit le second membre, et ne mesure "
        "pas la contrainte que le solveur garantit")
    assert spectral > 1e3 * matched, (
        f"le spectral rend {spectral:.3e} contre {matched:.3e} pour "
        "l'assorti : si les deux coincidaient, ce test ne separerait rien")


def test_the_two_operators_still_disagree_by_orders_of_magnitude(trajectory):
    """Epingle l'ANCIEN comportement : sans lui, D-72 se defait en silence.

    Si quelqu'un remet l'operateur spectral, les tests ci-dessus tombent.
    Celui-ci dit pourquoi ils tombent, en chiffrant l'ecart mesure.

    Il mord aussi dans l'autre sens : le jour ou `PROJECT_B` repasse a True,
    B redevient spectralement solenoidal, l'ecart se referme et ce test
    echoue — ce qui est le comportement voulu, le choix d'operateur devant
    alors etre retranche.
    """
    from Simulation.solver import MHDSolver

    assert MHDSolver.PROJECT_B is False, (
        "PROJECT_B est repasse a True : B redevient spectralement "
        "solenoidal et le choix d'operateur de D-72 doit etre retranche")

    rms_B = float(np.sqrt((trajectory["Bx"] ** 2
                           + trajectory["By"] ** 2).mean()))
    spectral = float(np.max(np.abs(_spectral_divergence(
        trajectory["Bx"], trajectory["By"], trajectory["dx"])))) / rms_B

    assert spectral > _T14_TOLERANCE, (
        f"l'operateur spectral rend {spectral:.4e} sur {_SCENARIO} N={_N} ; "
        "il valait 6.0470e-01 a la mesure de D-72. S'il est retombe sous le "
        f"seuil de {_T14_TOLERANCE:.0e}, l'ecart entre les deux operateurs "
        "a change et D-72 doit etre remesure")
