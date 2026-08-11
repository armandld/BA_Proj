"""Audit de contrat des portes physiques du mappeur v1.

Cinq fonctions statiques portent tout le raisonnement physique du v1 :
`_f_gate` (nombre de Reynolds), `_threshold_contrast` (contraste au seuil),
`_g_strain` / `_g_rot` (interrupteurs topologiques d'Okubo-Weiss) et
`_g_mag` (activite magnetique). Leurs docstrings enoncent des contrats
precis — continuite, bornes, sens d'activation. Ce fichier les verifie un a
un plutot que de les croire.

VERDICT : les cinq honorent ce qu'elles annoncent. Aucun defaut de calcul.
Deux constats structurels meritaient d'etre ecrits.

CONSTAT 1 — `_g_strain` et `_g_rot` ne sont pas deux interrupteurs, mais un
seul. Elles somment a 1 EXACTEMENT, pour tout Q :

    1/(1+e^x) + 1/(1+e^-x) = 1

Elles ne peuvent donc jamais etre actives ensemble, ni inactives ensemble.
Le ZZ (porte par g_strain) et le ZZZZ (porte par g_rot) sont une PARTITION
d'un unique scalaire d'Okubo-Weiss, pas deux detecteurs independants. Cela
change la lecture d'une ablation : retirer le ZZ ne retire pas une source
d'information distincte du ZZZZ, il deplace le poids d'un cote a l'autre du
meme signal.

  | Q      | g_strain | g_rot  | somme |
  |--------|----------|--------|-------|
  | -10    | 1.000000 | 0.000000 | 1.0 |
  |   0    | 0.500000 | 0.500000 | 1.0 |
  | +10    | 0.000000 | 1.000000 | 1.0 |

CONSTAT 2 — l'exemple de la docstring de `_f_gate` est inatteignable avec
la valeur par defaut. Elle illustre la croissance logarithmique par
« Re=3000, x_crit=10, gamma=2 -> f ~ 12 (not infinity) ». La formule rend
bien 12.4076, mais `f_max=10.0` par defaut la ramene a 10.0000 : la valeur
citee ne peut jamais sortir de la fonction telle qu'elle est appelee.
"""

import os
import sys

import numpy as np
import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.HamiltParams import PhysicalMapper as PM  # noqa: E402


def _a(*v):
    return np.array(v, dtype=float)


# ======================================================================
#  1. _f_gate — porte de Reynolds
# ======================================================================

@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0, 5.0])
def test_the_reynolds_gate_is_continuous_at_the_critical_value(gamma):
    """La docstring l'annonce : les deux regimes valent 1.0 au raccord.

    Une discontinuite ferait sauter le coefficient d'un patch a l'autre pour
    une difference de champ infinitesimale.
    """
    xc = 10.0
    left = PM._f_gate(_a(xc - 1e-9), xc, gamma)[0]
    right = PM._f_gate(_a(xc + 1e-9), xc, gamma)[0]
    assert left == pytest.approx(1.0, abs=1e-8)
    assert right == pytest.approx(1.0, abs=1e-8)
    assert abs(right - left) < 1e-8


def test_the_reynolds_gate_is_linear_below_the_critical_value():
    """Regime normal : f = x / x_crit, proportionnel."""
    xc = 10.0
    for x in (0.0, 2.5, 5.0, 9.9):
        assert PM._f_gate(_a(x), xc, 2.0)[0] == pytest.approx(x / xc, rel=1e-9)


def test_the_reynolds_gate_grows_logarithmically_above_it():
    xc, g = 10.0, 2.0
    for r in (2.0, 10.0, 100.0):
        got = PM._f_gate(_a(r * xc), xc, g, f_max=1e9)[0]
        assert got == pytest.approx(1.0 + g * np.log(r), rel=1e-9)


def test_the_reynolds_gate_is_monotone_over_five_decades():
    """Une porte non monotone rendrait un champ plus violent moins detecte."""
    f = PM._f_gate(np.linspace(0.0, 1e4, 5000), 10.0, 2.0)
    assert np.all(np.diff(f) >= -1e-12)


def test_the_reynolds_gate_is_clamped_and_never_diverges():
    assert PM._f_gate(_a(1e12), 10.0, 2.0)[0] == 10.0
    assert np.all(np.isfinite(PM._f_gate(_a(0.0, 1e300), 10.0, 2.0)))


def test_the_docstring_example_is_unreachable_with_the_default_clamp():
    """« Re=3000, x_crit=10, gamma=2 -> f ~ 12 » : la formule le donne, la
    valeur par defaut de f_max ne le laisse pas sortir."""
    assert PM._f_gate(_a(3000.0), 10.0, 2.0, f_max=1e9)[0] == pytest.approx(
        12.4076, abs=1e-3)
    assert PM._f_gate(_a(3000.0), 10.0, 2.0)[0] == 10.0


def test_a_zero_critical_value_does_not_divide_by_zero():
    assert np.all(np.isfinite(PM._f_gate(_a(1.0), 0.0, 2.0)))


def test_a_zero_growth_rate_saturates_the_gate_at_one():
    """gamma = 0 : plus aucune croissance au-dela du seuil."""
    assert PM._f_gate(_a(1e6), 10.0, 0.0)[0] == pytest.approx(1.0)


# ======================================================================
#  2. _threshold_contrast — contraste relatif au seuil
# ======================================================================

def test_the_contrast_is_exactly_zero_below_and_at_the_threshold():
    """Au seuil, pas « presque zero » : zero. Sinon un domaine calme
    porterait un couplage partout."""
    for v in (0.0, 5.0, 9.999, 10.0):
        assert PM._threshold_contrast(_a(v), 10.0, 1.0)[0] == 0.0


def test_the_contrast_is_linear_in_the_excess_above_the_threshold():
    for v, expected in ((20.0, 1.0), (30.0, 2.0), (110.0, 10.0)):
        assert PM._threshold_contrast(_a(v), 10.0, 1.0, tc_max=1e9)[0] == \
            pytest.approx(expected, rel=1e-9)


def test_beta_scales_the_contrast_and_nothing_else():
    a = PM._threshold_contrast(_a(30.0), 10.0, 1.0, tc_max=1e9)[0]
    b = PM._threshold_contrast(_a(30.0), 10.0, 3.0, tc_max=1e9)[0]
    assert b / a == pytest.approx(3.0, rel=1e-9)
    assert PM._threshold_contrast(_a(5.0), 10.0, 3.0)[0] == 0.0


def test_a_negative_value_gives_zero_and_not_a_negative_coupling():
    """Un contraste negatif retournerait le signe du couplage."""
    assert PM._threshold_contrast(_a(-50.0), 10.0, 1.0)[0] == 0.0


def test_the_contrast_is_clamped():
    assert PM._threshold_contrast(_a(1e12), 10.0, 1.0)[0] == 10.0


def test_the_contrast_compares_to_a_fixed_threshold_not_to_the_mean():
    """C'est la difference revendiquee avec Michelson : un domaine
    UNIFORMEMENT actif doit garder un signal, pas le voir s'annuler."""
    uniform = np.full(64, 50.0)
    got = PM._threshold_contrast(uniform, 10.0, 1.0, tc_max=1e9)
    assert np.all(got == pytest.approx(4.0)), (
        "un domaine uniformement au-dessus du seuil rend un signal nul : "
        "c'est le comportement de Michelson que cette fonction remplace")


def test_the_contrast_is_monotone():
    v = np.linspace(0.0, 500.0, 2000)
    assert np.all(np.diff(PM._threshold_contrast(v, 10.0, 1.0, tc_max=1e9)) >= -1e-12)


# ======================================================================
#  3. g_strain / g_rot — un seul interrupteur, pas deux
# ======================================================================

@pytest.mark.parametrize("Q", [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
def test_the_two_topological_gates_sum_to_exactly_one(Q):
    """Constat structurel : ce sont deux faces d'un meme scalaire."""
    gs = PM._g_strain(_a(Q), 1.0, 5.0)[0]
    gr = PM._g_rot(_a(Q), 1.0, 5.0)[0]
    assert gs + gr == pytest.approx(1.0, abs=1e-12)


def test_the_two_gates_can_never_be_active_together():
    """Consequence directe : ZZ et ZZZZ partagent un budget, ils ne sont pas
    deux sources d'information independantes."""
    for Q in np.linspace(-50, 50, 101):
        gs = PM._g_strain(_a(Q), 1.0, 5.0)[0]
        gr = PM._g_rot(_a(Q), 1.0, 5.0)[0]
        assert not (gs > 0.9 and gr > 0.9)
        assert not (gs < 0.1 and gr < 0.1)


def test_strain_activates_where_strain_dominates():
    """Convention : Q = 0.5(omega^2 - strain^2), donc Q < 0 = deformation."""
    assert PM._g_strain(_a(-100.0), 1.0, 5.0)[0] > 0.99
    assert PM._g_strain(_a(+100.0), 1.0, 5.0)[0] < 0.01


def test_rotation_activates_where_rotation_dominates():
    assert PM._g_rot(_a(+100.0), 1.0, 5.0)[0] > 0.99
    assert PM._g_rot(_a(-100.0), 1.0, 5.0)[0] < 0.01


def test_both_gates_are_one_half_at_the_neutral_point():
    assert PM._g_strain(_a(0.0), 1.0, 5.0)[0] == pytest.approx(0.5)
    assert PM._g_rot(_a(0.0), 1.0, 5.0)[0] == pytest.approx(0.5)


def test_kappa_only_sharpens_the_switch_and_never_moves_it():
    """Le point de bascule doit rester a Q = 0 quelle que soit la raideur."""
    for k in (0.5, 5.0, 50.0):
        assert PM._g_strain(_a(0.0), 1.0, k)[0] == pytest.approx(0.5)
    soft = PM._g_strain(_a(-1.0), 1.0, 0.5)[0]
    hard = PM._g_strain(_a(-1.0), 1.0, 50.0)[0]
    assert hard > soft


def test_the_gates_never_overflow_on_extreme_input():
    """Le clip a +/-500 existe pour cela ; sans lui, exp() deborde."""
    for Q in (-1e300, 1e300, np.inf, -np.inf):
        gs = PM._g_strain(_a(Q), 1.0, 5.0)[0]
        gr = PM._g_rot(_a(Q), 1.0, 5.0)[0]
        assert np.isfinite(gs) and np.isfinite(gr)
        assert 0.0 <= gs <= 1.0 and 0.0 <= gr <= 1.0


def test_the_gates_stay_in_the_unit_interval_everywhere():
    Q = np.linspace(-1e6, 1e6, 10001)
    for g in (PM._g_strain(Q, 1.0, 5.0), PM._g_rot(Q, 1.0, 5.0)):
        assert np.all(g >= 0.0) and np.all(g <= 1.0)


# ======================================================================
#  4. _g_mag — activite magnetique
# ======================================================================

def test_the_magnetic_gate_switches_at_the_critical_current():
    assert PM._g_mag(_a(1.0), 1.0, 5.0)[0] == pytest.approx(0.5)


def test_the_magnetic_gate_saturates_on_a_strong_current_sheet():
    assert PM._g_mag(_a(100.0), 1.0, 5.0)[0] > 0.999


def test_the_magnetic_gate_is_leaky_and_not_exactly_zero_when_quiet():
    """« Leaky sigmoid » : le nom annonce un plancher non nul. Il vaut
    0.0067 a courant nul — assez petit pour ne rien declencher, assez
    grand pour qu'un gradient ne meure pas."""
    got = PM._g_mag(_a(0.0), 1.0, 5.0)[0]
    assert 0.0 < got < 0.01


def test_the_magnetic_gate_ignores_the_sign_of_the_current():
    """|Jz| : une nappe de courant est dangereuse dans les deux sens."""
    assert PM._g_mag(_a(-3.0), 1.0, 5.0)[0] == PM._g_mag(_a(3.0), 1.0, 5.0)[0]


def test_the_magnetic_gate_is_monotone_in_the_current_magnitude():
    j = np.linspace(0.0, 50.0, 2000)
    assert np.all(np.diff(PM._g_mag(j, 1.0, 5.0)) >= -1e-12)


def test_the_magnetic_gate_stays_in_the_unit_interval():
    j = np.linspace(-1e6, 1e6, 10001)
    g = PM._g_mag(j, 1.0, 5.0)
    assert np.all(g >= 0.0) and np.all(g <= 1.0)


def test_a_zero_critical_current_does_not_divide_by_zero():
    assert np.all(np.isfinite(PM._g_mag(_a(1.0), 0.0, 5.0)))


# ======================================================================
#  5. Ce que les portes font ENSEMBLE dans le coefficient
# ======================================================================

def test_the_hydro_branch_depends_on_the_cell_reynolds_number_alone():
    """`f_Re` et `mic_v` sont deux reparametrages MONOTONES du meme scalaire :
    Re_h = v_jump*dx/nu, et v_jump/v_jump_crit = Re_h/RE_CRIT.

    Le coefficient presente donc deux facteurs physiques distincts la ou il
    n'y a qu'une variable. Ce n'est pas faux, mais il faut le savoir avant
    d'interpreter une ablation de l'un des deux.
    """
    RE_CRIT, nu, dx = 10.0, 1e-3, 0.02
    v_jump_crit = RE_CRIT * nu / dx
    for v_jump in (0.1, 1.0, 5.0, 50.0):
        Re_h = v_jump * dx / nu
        assert v_jump / v_jump_crit == pytest.approx(Re_h / RE_CRIT, rel=1e-12)


def test_a_uniformly_quiet_field_produces_no_coupling_at_all():
    """Sous les deux seuils, le produit doit etre exactement nul —
    pas « petit »."""
    f = PM._f_gate(_a(1.0), 10.0, 2.0)[0]
    tc = PM._threshold_contrast(_a(1.0), 10.0, 1.0)[0]
    assert f > 0.0 and tc == 0.0
    assert f * tc == 0.0


def test_the_product_of_gates_stays_bounded():
    """f_max=10, tc_max=10, g dans [0,1] : le coefficient ne peut pas
    exploser, quel que soit le champ."""
    worst = (PM._f_gate(_a(1e12), 10.0, 5.0)[0]
             * PM._threshold_contrast(_a(1e12), 10.0, 5.0)[0]
             * PM._g_strain(_a(-1e12), 1.0, 5.0)[0])
    assert worst <= 100.0 + 1e-9
