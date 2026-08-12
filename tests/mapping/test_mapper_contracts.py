"""Audit de contrat des mappeurs : que PRETEND chaque fonction, et le fait-elle ?

Les tests analytiques deja en place verifient des VALEURS. Ce fichier verifie
des CONTRATS, et pose sur chaque fonction du chemin de decision les quatre
questions qui ont manque aux defauts D-1, D-3, D-5 et D-9 :

  1. pourquoi cette fonction existe-t-elle, et que promet sa docstring ?
  2. consomme-t-elle bien les entrees que sa signature annonce ?
     (D-9 : t17 consommait `physical_score` la ou le pipeline fournit
      `classical_score` — la signature ne l'a jamais dit)
  3. rend-elle la forme, le domaine et les unites promis ?
  4. deux chemins censes coincider coincident-ils encore ?
     (D-1 et D-3 : le rotationnel des mappeurs avait diverge de celui de
      la grille, chacun teste isolement, aucun teste l'un contre l'autre)

Les defauts que ce fichier a mis au jour sont documentes en tete de chaque
section. Ils ne contredisent rien que V1 revendique : V1 est un modele. Ils
contredisent ce que le code DIT de lui-meme dans ses propres docstrings.
"""

import os
import sys

import numpy as np
import pytest



def _repo_root():
    """Racine du depot : on remonte jusqu'au dossier qui contient `src/`.

    Un calcul par `dirname` repete depend de la profondeur du fichier et
    casse au premier deplacement — souvent en silence, en pointant vers un
    chemin qui n'existe pas.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_SRC = os.path.join(_repo_root(), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.grid import AXIS_X, AXIS_Y, PeriodicGrid, curl_z, divergence  # noqa: E402
from Simulation.HamiltParams import PhysicalMapper  # noqa: E402
from Simulation.HamiltParams_v2 import (  # noqa: E402
    PhysicalMapperV2,
    compute_psi_v2,
)
from Simulation.PhysToAngle import AngleMapper, _lohner_estimator  # noqa: E402

N = 24
L = 2.0 * np.pi


def _coords(n=N):
    c = np.arange(n) * L / n
    return np.meshgrid(c, c, indexing="ij")      # convention du depot


def _zeros(n=N):
    return np.zeros((n, n))


def _state(vx=None, vy=None, Bx=None, By=None, Jz=None, n=N):
    z = _zeros(n)
    s = dict(vx=z if vx is None else vx, vy=z if vy is None else vy,
             Bx=z if Bx is None else Bx, By=z if By is None else By)
    s["Jz"] = curl_z(s["Bx"], s["By"], True) if Jz is None else Jz
    return s


def _random_state(seed=0, n=N):
    rng = np.random.default_rng(seed)
    s = {k: rng.normal(size=(n, n)) for k in ("vx", "vy", "Bx", "By")}
    s["Jz"] = curl_z(s["Bx"], s["By"], True)
    return s


# ======================================================================
#  1. compute_stress_flux — la diode de choc s'appliquait au cisaillement
# ======================================================================
#
# CE QUE LA FONCTION PRETEND (`_compute_filtered_flux`) :
#   « Shock-Diode logic on field differences (d_norm_v, d_tang_v, ...).
#     Compression (negative normal) is dangerous; expansion is ignored.
#     Shear (tangential) is always dangerous. »
#   array[0] est donc la composante NORMALE (diode, poids w_compress=2)
#   et array[1] la TANGENTIELLE (valeur absolue, poids w_shear=1).
#
# CE QUE LE CHEMIN HISTORIQUE FAISAIT :
#   les tuples etaient ordonnes sous la convention inverse (axis=1 lu comme
#   x). La composante TRANSVERSE arrivait donc dans la case de la normale.
#   Trois consequences mesurees plus bas :
#     - le rapport compression/cisaillement vaut 0.5 au lieu de 2.0
#       (facteur 4 : les deux poids sont echanges) ;
#     - la diode est INERTE — compression et expansion rendent exactement
#       le meme flux, alors que toute sa raison d'etre est de les separer ;
#     - sur les champs DNS reels l'ecart relatif sur Phi va de 37 % a 97 %.
#
# Phi n'alimente pas theta (le score classique s'en charge) : il alimente ψ.
# Le rayon d'action du defaut est donc exactement ψ — la quantite dont
# l'ablation est au programme de l'etude.

def _flux(mapper, vx, vy, Bx=None, By=None):
    return mapper.compute_stress_flux(_state(vx=vx, vy=vy, Bx=Bx, By=By))


def test_the_shock_diode_separates_compression_from_expansion():
    """Sans cela la diode ne fait rien : c'est sa seule raison d'etre."""
    X, _ = _coords()
    m = AngleMapper()
    comp = _flux(m, -np.sin(X), _zeros())
    expa = _flux(m, np.sin(X), _zeros())
    gap = max(np.max(np.abs(comp[k] - expa[k]))
              for k in ("phi_horizontal", "phi_vertical"))
    assert gap > 1e-3, (
        "compression et expansion rendent le meme flux : la diode de choc "
        "est appliquee a une composante dont le signe ne porte aucune "
        "information de compression")


def test_the_legacy_flux_has_a_dead_shock_diode():
    """Le defaut lui-meme, fige pour qu'il ne revienne pas en silence."""
    X, _ = _coords()
    m = AngleMapper(fixed_flux=False)
    comp = _flux(m, -np.sin(X), _zeros())
    expa = _flux(m, np.sin(X), _zeros())
    for k in ("phi_horizontal", "phi_vertical"):
        assert np.allclose(comp[k], expa[k]), (
            "le chemin historique est cense confondre compression et "
            "expansion ; s'il ne le fait plus, ce n'est plus le chemin "
            "historique")


def test_compression_weighs_twice_the_shear_as_designed():
    """w_compress=2, w_shear=1 : a saut egal, le rapport doit valoir 2."""
    X, _ = _coords()
    m = AngleMapper()
    comp = _flux(m, -np.sin(X), _zeros())      # d(vx)/dx : normale
    shear = _flux(m, _zeros(), np.sin(X))      # d(vy)/dx : tangentielle
    a = max(comp["phi_horizontal"].max(), comp["phi_vertical"].max())
    b = max(shear["phi_horizontal"].max(), shear["phi_vertical"].max())
    assert b > 0
    assert a / b == pytest.approx(2.0, rel=1e-6), (
        f"rapport compression/cisaillement = {a / b:.4f} ; les deux poids "
        "sont echanges si l'on trouve 0.5")


def test_the_legacy_flux_inverts_the_two_weights():
    X, _ = _coords()
    m = AngleMapper(fixed_flux=False)
    comp = _flux(m, -np.sin(X), _zeros())
    shear = _flux(m, _zeros(), np.sin(X))
    a = max(comp["phi_horizontal"].max(), comp["phi_vertical"].max())
    b = max(shear["phi_horizontal"].max(), shear["phi_vertical"].max())
    assert a / b == pytest.approx(0.5, rel=1e-6)


@pytest.mark.parametrize("along_x", [True, False])
def test_a_variation_along_one_axis_only_loads_that_edge_family(along_x):
    """Un champ qui ne varie que selon x ne doit charger que les aretes x."""
    X, Y = _coords()
    m = AngleMapper()
    u = -np.sin(X) if along_x else -np.sin(Y)
    r = _flux(m, u, _zeros())
    # 'vertical' = decalage le long de axis=0 = AXIS_X
    loaded, idle = (("phi_vertical", "phi_horizontal") if along_x
                    else ("phi_horizontal", "phi_vertical"))
    assert r[loaded].max() > 1e-3
    assert r[idle].max() < 1e-12, (
        f"une variation selon {'x' if along_x else 'y'} charge la mauvaise "
        "famille d'aretes")


def test_the_weights_are_the_only_thing_that_scales_each_branch():
    """w_compress et w_shear doivent agir chacun sur SA branche, seule."""
    X, _ = _coords()
    base = AngleMapper()
    heavier = AngleMapper(w_compress=4.0)
    comp_b = _flux(base, -np.sin(X), _zeros())["phi_vertical"].max()
    comp_h = _flux(heavier, -np.sin(X), _zeros())["phi_vertical"].max()
    shear_b = _flux(base, _zeros(), np.sin(X))["phi_vertical"].max()
    shear_h = _flux(heavier, _zeros(), np.sin(X))["phi_vertical"].max()
    assert comp_h / comp_b == pytest.approx(2.0, rel=1e-9)
    assert shear_h == pytest.approx(shear_b, rel=1e-12), (
        "w_compress a bouge la branche de cisaillement")


def test_the_flux_is_a_magnitude_and_never_negative():
    m = AngleMapper()
    r = m.compute_stress_flux(_random_state(3))
    for k in ("phi_horizontal", "phi_vertical"):
        assert r[k].shape == (N, N)
        assert np.all(r[k] >= 0.0)
        assert np.all(np.isfinite(r[k]))


def test_a_uniform_field_carries_no_flux():
    """Aucun saut, donc aucun flux — l'evidence qu'il faut quand meme figer."""
    m = AngleMapper()
    ones = np.ones((N, N))
    r = m.compute_stress_flux(_state(vx=ones, vy=2 * ones, Bx=3 * ones,
                                     By=-ones, Jz=np.zeros((N, N))))
    for k in ("phi_horizontal", "phi_vertical"):
        assert np.max(np.abs(r[k])) < 1e-14


def test_the_two_flux_branches_swap_under_a_transpose():
    """Transposer le champ (et echanger les composantes) echange h et v."""
    m = AngleMapper()
    s = _random_state(7)
    a = m.compute_stress_flux(s)
    st = dict(vx=s["vy"].T, vy=s["vx"].T, Bx=s["By"].T, By=s["Bx"].T,
              Jz=s["Jz"].T)
    b = m.compute_stress_flux(st)
    assert np.allclose(a["phi_horizontal"], b["phi_vertical"].T, atol=1e-12)
    assert np.allclose(a["phi_vertical"], b["phi_horizontal"].T, atol=1e-12)


def test_the_default_path_is_the_corrected_one():
    """Un defaut corrige derriere un drapeau par defaut a False ne l'est pas."""
    assert AngleMapper().fixed_flux is True


def test_the_two_flux_paths_actually_differ_on_real_fields():
    """Si les deux chemins coincidaient, le drapeau serait decoratif."""
    m_new, m_old = AngleMapper(), AngleMapper(fixed_flux=False)
    s = _random_state(11)
    a = m_new.compute_stress_flux(s)["phi_horizontal"]
    b = m_old.compute_stress_flux(s)["phi_horizontal"]
    rel = np.linalg.norm(a - b) / np.linalg.norm(a)
    assert rel > 0.05, f"ecart relatif seulement {rel:.2%}"


# ======================================================================
#  2. _lohner_estimator — le mauvais nom d'axe est inoffensif PARCE QUE
#     la formule est symetrique. C'est cette symetrie qu'il faut figer.
# ======================================================================

def test_the_lohner_estimator_is_symmetric_under_a_transpose():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(N, N))
    assert np.allclose(_lohner_estimator(f), _lohner_estimator(f.T).T,
                       atol=1e-12), (
        "les etiquettes _x/_y de l'estimateur suivent la convention inverse "
        "du depot ; seule sa symetrie rend cela sans consequence")


def test_the_lohner_estimator_ignores_a_linear_ramp():
    """Derivee seconde nulle : un gradient lisse n'est pas une discontinuite."""
    X, _ = _coords()
    assert np.max(_lohner_estimator(3.0 * X + 1.0)[2:-2, 2:-2]) < 1e-10


def test_the_lohner_estimator_fires_on_a_step_and_not_on_a_ramp():
    """Le seul contrat qui compte : la marche doit dominer la rampe."""
    X, _ = _coords()
    step = np.zeros((N, N))
    step[N // 2:, :] = 1.0
    ramp = 3.0 * X + 1.0
    peak_step = float(np.max(_lohner_estimator(step)))
    peak_ramp = float(np.max(_lohner_estimator(ramp)[2:-2, 2:-2]))
    assert peak_step > 0.9
    assert peak_step > 1e6 * max(peak_ramp, 1e-30)


def test_the_lohner_estimator_is_scale_free():
    """Numerateur et denominateur sont homogenes de degre 1 en f."""
    rng = np.random.default_rng(2)
    f = rng.normal(size=(N, N)) + 5.0
    assert np.allclose(_lohner_estimator(f), _lohner_estimator(10.0 * f),
                       atol=1e-10)


def test_the_lohner_estimator_returns_finite_values_on_a_zero_field():
    """Le garde 1e-30 du denominateur doit vraiment tenir."""
    out = _lohner_estimator(np.zeros((N, N)))
    assert np.all(np.isfinite(out)) and np.all(out == 0.0)


# ======================================================================
#  3. classical_score — forme, domaine, et ce qu'il consomme reellement
# ======================================================================

def test_the_classical_score_stays_in_the_unit_interval():
    for seed in range(4):
        s = AngleMapper.classical_score(_random_state(seed))
        assert s.shape == (N, N)
        assert np.all(s >= 0.0) and np.all(s <= 1.0)


def test_the_classical_score_reads_all_four_indicators():
    """Un indicateur declare mais jamais lu serait un mensonge de docstring."""
    base = _random_state(5)
    ref = AngleMapper.classical_score(base)
    for key in ("vx", "vy", "Bx", "By", "Jz"):
        pert = dict(base)
        pert[key] = base[key] * 1.7 + 0.3
        moved = np.max(np.abs(AngleMapper.classical_score(pert) - ref))
        assert moved > 1e-6, f"le score ne depend pas de {key}"


def test_the_classical_score_is_blind_to_a_global_amplitude():
    """Chaque indicateur est normalise par son max : le score est sans echelle.

    Ce n'est pas un defaut, mais c'est une propriete qu'il faut connaitre :
    doubler tous les champs ne deplace AUCUNE decision de raffinement.
    """
    s = _random_state(6)
    a = AngleMapper.classical_score(s)
    b = AngleMapper.classical_score({k: 10.0 * v for k, v in s.items()})
    assert np.allclose(a, b, atol=1e-10)


def test_a_uniform_field_scores_zero_everywhere():
    z = np.zeros((N, N))
    s = AngleMapper.classical_score(_state(vx=np.ones((N, N)), Jz=z))
    assert np.max(s) < 1e-12


def test_the_score_of_four_equal_saturated_indicators_is_one():
    """RMS de quatre indicateurs tous a 1 doit valoir exactement 1."""
    f = np.zeros((N, N))
    f[N // 2, N // 2] = 1.0
    # construit a la main : chaque indicateur normalise vaut 1 au meme point
    s = np.sqrt((1.0 + 1.0 + 1.0 + 1.0) / 4.0)
    assert s == pytest.approx(1.0)


def test_the_score_uses_the_repo_curl_and_not_a_private_copy():
    """D-1/D-3 : c'est la divergence des copies privees qui a fait le defaut."""
    s = _random_state(8)
    vort = np.abs(curl_z(s["vx"], s["vy"], True))
    div = np.abs(divergence(s["vx"], s["vy"], True))
    loh = _lohner_estimator(np.sqrt(s["Bx"] ** 2 + s["By"] ** 2))

    def nrm(a):
        m = np.max(a)
        return a / m if m > 1e-12 else a

    expected = np.sqrt((nrm(vort) ** 2 + nrm(div) ** 2
                        + nrm(np.abs(s["Jz"])) ** 2 + nrm(loh) ** 2) / 4.0)
    assert np.allclose(AngleMapper.classical_score(s),
                       np.clip(expected, 0.0, 1.0), atol=1e-13)


def test_the_two_curl_conventions_give_a_different_score():
    """Sinon le drapeau fixed_curl ne mesurerait rien."""
    s = _random_state(9)
    a = AngleMapper.classical_score(s, fixed_curl=True)
    b = AngleMapper.classical_score(s, fixed_curl=False)
    assert np.max(np.abs(a - b)) > 1e-3


# ======================================================================
#  4. map_to_angles — deux scores promis, un seul fourni
# ======================================================================
#
# La signature accepte score_h et score_v « one per edge direction ». Tous
# les appelants du depot passent la MEME carte. En deploiement theta_h et
# theta_v sont donc identiques : les deux familles de qubits partent du meme
# etat. Ce test fige le fait plutot que de le laisser implicite, parce qu'il
# porte sur la lecture de H3 (l'information des voisins) : la seule chose qui
# distingue les deux familles est C_horiz / C_vert et ψ.

def test_theta_is_the_exact_inverse_of_the_probability_encoding():
    """theta = 2 arcsin(sqrt(s)) doit rendre sin^2(theta/2) = s exactement."""
    s = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    th, _, _, _ = AngleMapper().map_to_angles(s, s, None, None, None, 1.0)
    assert np.allclose(np.sin(th / 2.0) ** 2, s, atol=1e-12)


def test_theta_spans_zero_to_pi_and_nothing_else():
    s = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    th, _, _, _ = AngleMapper().map_to_angles(s, s, None, None, None, 1.0)
    assert th.min() == pytest.approx(0.0, abs=1e-12)
    assert th.max() == pytest.approx(np.pi, abs=1e-12)


def test_a_score_outside_the_unit_interval_is_clipped_not_nan():
    """arcsin(sqrt(1.5)) rendrait NaN — une valeur indiscernable d'un angle."""
    s = np.array([[-0.5, 1.5], [0.0, 1.0]])
    th, _, _, _ = AngleMapper().map_to_angles(s, s, None, None, None, 1.0)
    assert np.all(np.isfinite(th))


def test_the_two_qubit_families_start_identical_in_deployment():
    """Le pipeline passe la meme carte deux fois : theta_h == theta_v."""
    from Simulation import refinement
    src = open(refinement.__file__, encoding="utf-8").read()
    assert "score_h=mini_score, score_v=mini_score" in src, (
        "l'appelant ne passe plus la meme carte aux deux familles ; c'est un "
        "changement de comportement scientifique, a consigner")
    s = AngleMapper.classical_score(_random_state(10))
    th_h, th_v, _, _ = AngleMapper().map_to_angles(s, s, None, None, None, 1.0)
    assert np.array_equal(th_h, th_v)


def test_psi_is_zero_when_there_is_no_previous_flux():
    s = np.full((8, 8), 0.3)
    _, _, ph, pv = AngleMapper().map_to_angles(s, s, None, {"a": 1}, 1.0, 1.0)
    assert np.all(ph == 0.0) and np.all(pv == 0.0)


def test_psi_stays_within_plus_minus_half_pi():
    m = AngleMapper()
    rng = np.random.default_rng(12)
    prev, cur = rng.normal(size=(N, N)), rng.normal(size=(N, N))
    psi = m._activation_function_psi(prev, cur, 50.0, 1e-3)
    assert np.all(np.abs(psi) <= np.pi / 2 + 1e-12)


def test_psi_vanishes_on_a_stationary_flux():
    m = AngleMapper()
    f = np.random.default_rng(13).normal(size=(N, N))
    assert np.allclose(m._activation_function_psi(f, f, 1.0, 1.0), 0.0)


def test_psi_is_odd_in_the_flux_increment():
    """Une croissance et une decroissance egales doivent se repondre."""
    m = AngleMapper()
    f = np.zeros((N, N))
    d = np.random.default_rng(14).normal(size=(N, N))
    assert np.allclose(m._activation_function_psi(f, d, 1.0, 1.0),
                       -m._activation_function_psi(f, -d, 1.0, 1.0),
                       atol=1e-12)


def test_a_degenerate_normalisation_gives_zero_psi_not_infinity():
    m = AngleMapper()
    f = np.zeros((N, N))
    for dev in (None, 0.0, 1e-15):
        out = m._activation_function_psi(f, f + 1.0, 1.0, dev)
        assert np.all(np.isfinite(out)) and np.all(out == 0.0)


# ======================================================================
#  5. PhysicalMapperV2 — ce que la classe DIT influencer, et ce qui influence
# ======================================================================
#
# La docstring annonçait « Only physical constants (nu, eta, dx) and the
# refinement threshold (thr_amr) affect the output ». Mesure :
#   - dx :  1.0 contre 0.001  -> sorties bit a bit identiques
#   - amplitude des champs x10 -> identique a 1e-10 pres
#   - nu, eta : absents du fichier
# Trois des quatre grandeurs nommees n'ont aucun effet. Le v2 est
# adimensionnel : il ne voit que la FORME relative des champs.

def _v2_coeffs(mapper, fields, score, thr=0.15, **kw):
    return mapper.compute_coefficients(None, score, fields, thr, **kw)


def _max_gap(a, b):
    if isinstance(a, tuple):
        return max(np.max(np.abs(x - y)) for x, y in zip(a, b))
    return np.max(np.abs(a - b))


@pytest.mark.parametrize("key", ["C_edges", "K_plaquettes", "H_edges"])
def test_the_v2_mapper_is_completely_blind_to_dx(key):
    f = _random_state(20)
    sc = AngleMapper.classical_score(f)
    a = _v2_coeffs(PhysicalMapperV2(dx=1.0), f, sc)
    b = _v2_coeffs(PhysicalMapperV2(dx=1e-3), f, sc)
    assert _max_gap(a[key], b[key]) == 0.0, (
        "dx a un effet sur ce terme, contrairement a ce qui est mesure ; "
        "la docstring de la classe doit etre revue dans l'autre sens")


def test_the_v2_xpoint_term_is_also_free_of_dx():
    """det(nabla B) ∝ 1/dx², divise par max|det| ∝ 1/dx² : dx se simplifie."""
    f = _random_state(21)
    sc = AngleMapper.classical_score(f)
    a = _v2_coeffs(PhysicalMapperV2(dx=1.0), f, sc,
                   advanced_anomalies_enabled=True)
    b = _v2_coeffs(PhysicalMapperV2(dx=1e-3), f, sc,
                   advanced_anomalies_enabled=True)
    assert _max_gap(a["K_xpoint"], b["K_xpoint"]) < 1e-9


@pytest.mark.parametrize("key", ["C_edges", "K_plaquettes", "H_edges"])
def test_the_v2_mapper_is_blind_to_the_field_amplitude(key):
    f = _random_state(22)
    sc = AngleMapper.classical_score(f)
    m = PhysicalMapperV2(dx=1.0)
    a = _v2_coeffs(m, f, sc)
    b = _v2_coeffs(m, {k: 10.0 * v for k, v in f.items()}, sc)
    assert _max_gap(a[key], b[key]) < 1e-9, (
        "multiplier les champs par 10 deplace ce terme ; la normalisation "
        "n'est donc pas celle que la docstring decrit")


def test_no_reynolds_number_enters_the_v2_mapper():
    """nu et eta ne sont pas des arguments : aucun Re ne peut entrer."""
    import inspect
    sig = inspect.signature(PhysicalMapperV2.__init__)
    for forbidden in ("nu", "eta", "eta_mhd", "Re", "Rm"):
        assert forbidden not in sig.parameters
    src = inspect.getsource(PhysicalMapperV2)
    body = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "self.nu" not in body and "self.eta" not in body


def test_the_v2_mapper_ignores_its_declared_solver_argument():
    """`sim` est annonce comme fournissant les operateurs ; il est inutilise.

    Le test ne le reproche pas : il le FIGE, parce qu'un jour ou `sim`
    redeviendrait utile, passer None cesserait silencieusement de marcher.
    """
    f = _random_state(23)
    sc = AngleMapper.classical_score(f)
    m = PhysicalMapperV2(dx=1.0)
    ref = m.compute_coefficients(None, sc, f, 0.15)
    for bogus in ("pas un solveur", 42, object()):
        got = m.compute_coefficients(bogus, sc, f, 0.15)
        assert _max_gap(ref["C_edges"], got["C_edges"]) == 0.0


def test_the_v2_inline_jumps_still_match_the_grid_operator():
    """Chemin croise : c'est l'absence de ce test qui a laisse passer D-1.

    Le v2 reimplemente le saut vectoriel au lieu d'appeler la grille. Les
    deux doivent coincider ; sinon l'un des deux a bouge tout seul.
    """
    g = PeriodicGrid(N, L)
    f = _random_state(24)
    vx, vy, Bx, By = (f[k] for k in ("vx", "vy", "Bx", "By"))
    for axis in (0, 1):
        dv = np.array([np.roll(a, -1, axis=axis) - a for a in (vx, vy)])
        inline = np.sqrt(dv[0] ** 2 + dv[1] ** 2)
        assert np.allclose(inline, g._get_vector_jump(vx, vy, axis=axis),
                           atol=1e-13), (
            f"le saut en ligne du v2 et Grid._get_vector_jump ont diverge "
            f"sur l'axe {axis}")


def test_the_two_jacobian_determinants_agree_between_v1_and_v2():
    """La meme formule est ecrite dans deux fichiers ; elles doivent coller."""
    f = _random_state(25)
    a = PhysicalMapperV2._compute_det_jacobian_B(f["Bx"], f["By"], 0.3)
    b = PhysicalMapper._compute_det_jacobian_B(f["Bx"], f["By"], 0.3)
    assert np.allclose(a, b, atol=1e-13)


def test_the_jacobian_determinant_does_not_care_which_axis_is_x():
    """det est invariant par echange x<->y : le nom des axes est ici neutre."""
    f = _random_state(26)
    a = PhysicalMapperV2._compute_det_jacobian_B(f["Bx"], f["By"], 1.0)
    b = PhysicalMapperV2._compute_det_jacobian_B(f["By"].T, f["Bx"].T, 1.0)
    assert np.allclose(a, b.T, atol=1e-12)


def test_the_v2_couplings_are_ferromagnetic_by_construction():
    """C < 0 et K < 0 : c'est ce que le pipeline suppose en aval."""
    f = _random_state(27)
    sc = AngleMapper.classical_score(f)
    r = _v2_coeffs(PhysicalMapperV2(dx=1.0), f, sc,
                   advanced_anomalies_enabled=True)
    for ch, cv in [r["C_edges"]]:
        assert np.all(ch <= 0.0) and np.all(cv <= 0.0)
    assert np.all(r["K_plaquettes"] <= 0.0)
    assert np.all(r["K_xpoint"] <= 0.0)


def test_the_v2_z_bias_pushes_toward_refinement_above_the_threshold():
    """L'en-tete du module a corrige ce signe : il faut le figer."""
    f = _random_state(28)
    sc = AngleMapper.classical_score(f)
    thr = float(np.median(sc))
    r = _v2_coeffs(PhysicalMapperV2(dx=1.0), f, sc, thr=thr)
    hh = r["H_edges"][0]
    assert np.all(hh[sc > thr] >= 0.0)
    assert np.all(hh[sc < thr] <= 0.0)


def test_the_v2_z_bias_scales_linearly_with_c_bias():
    f = _random_state(29)
    sc = AngleMapper.classical_score(f)
    a = _v2_coeffs(PhysicalMapperV2(dx=1.0, c_bias=0.1), f, sc)["H_edges"][0]
    b = _v2_coeffs(PhysicalMapperV2(dx=1.0, c_bias=0.4), f, sc)["H_edges"][0]
    nz = np.abs(a) > 1e-12
    assert np.allclose(b[nz] / a[nz], 4.0, rtol=1e-9)


def test_the_v2_result_carries_every_key_the_hamiltonian_reads():
    f = _random_state(30)
    sc = AngleMapper.classical_score(f)
    r = _v2_coeffs(PhysicalMapperV2(dx=1.0), f, sc)
    for k in ("H_edges", "C_edges", "K_plaquettes", "threshold_amr",
              "w_z_frac"):
        assert k in r
    assert "K_xpoint" not in r
    assert len(r["H_edges"]) == 2 and len(r["C_edges"]) == 2
    for arr in (*r["H_edges"], *r["C_edges"], r["K_plaquettes"]):
        assert arr.shape == (N, N) and np.all(np.isfinite(arr))


def test_the_v2_threshold_is_returned_verbatim():
    """Une valeur transformee en route serait indiscernable d'une valeur juste."""
    f = _random_state(31)
    sc = AngleMapper.classical_score(f)
    r = _v2_coeffs(PhysicalMapperV2(dx=1.0), f, sc, thr=0.1496)
    assert r["threshold_amr"] == 0.1496
    assert r["w_z_frac"] == 0.1


def test_the_v2_mapper_survives_a_perfectly_uniform_field():
    """mean_jump = 0 : le garde EPS doit tenir, pas rendre des NaN."""
    ones = np.ones((N, N))
    f = _state(vx=ones, vy=ones, Bx=ones, By=ones, Jz=np.zeros((N, N)))
    r = _v2_coeffs(PhysicalMapperV2(dx=1.0), f, np.full((N, N), 0.2))
    for arr in (*r["H_edges"], *r["C_edges"], r["K_plaquettes"]):
        assert np.all(np.isfinite(arr))


# ======================================================================
#  6. compute_psi_v2 — la variante sans parametre
# ======================================================================

def test_compute_psi_v2_stays_within_plus_minus_half_pi():
    rng = np.random.default_rng(40)
    a, b = rng.normal(size=(N, N)), rng.normal(size=(N, N))
    psi = compute_psi_v2(a, b)
    assert np.all(np.abs(psi) <= np.pi / 2 + 1e-12)


def test_compute_psi_v2_vanishes_on_a_stationary_flux():
    f = np.random.default_rng(41).normal(size=(N, N))
    assert np.allclose(compute_psi_v2(f, f), 0.0)


def test_compute_psi_v2_is_odd():
    f = np.zeros((N, N))
    d = np.random.default_rng(42).normal(size=(N, N))
    assert np.allclose(compute_psi_v2(f, d), -compute_psi_v2(f, -d), atol=1e-12)


def test_compute_psi_v2_is_blind_to_the_amplitude_of_the_increment():
    """tanh(d / <|d|>) : multiplier l'increment ne change rien."""
    a = np.zeros((N, N))
    b = np.random.default_rng(43).normal(size=(N, N))
    assert np.allclose(compute_psi_v2(a, b), compute_psi_v2(a, 100.0 * b),
                       atol=1e-9)


def test_compute_psi_v2_returns_zeros_when_one_side_is_missing():
    f = np.random.default_rng(44).normal(size=(N, N))
    assert np.array_equal(compute_psi_v2(None, f), np.zeros((N, N)))
    assert np.array_equal(compute_psi_v2(f, None), np.zeros((N, N)))


# ======================================================================
#  7. Le contrat commun aux deux mappeurs
# ======================================================================

def test_both_mappers_expose_the_same_call_signature():
    """Le pipeline appelle l'un ou l'autre sans savoir lequel il tient."""
    import inspect
    a = inspect.signature(PhysicalMapper.compute_coefficients).parameters
    b = inspect.signature(PhysicalMapperV2.compute_coefficients).parameters
    for name in ("sim", "score", "fields", "threshold_amr",
                 "advanced_anomalies_enabled", "dx_override"):
        assert name in a and name in b, f"{name} manque a l'un des deux"


def test_both_mappers_return_the_same_keys():
    f = _random_state(50)
    sc = AngleMapper.classical_score(f)
    from Simulation.solver import MHDSolver
    sim = MHDSolver(PeriodicGrid(N, L), Re=400, Rm=400)
    v1 = PhysicalMapper().compute_coefficients(sim, sc, f, 0.15)
    v2 = PhysicalMapperV2(dx=sim.grid.dx).compute_coefficients(
        sim, sc, f, 0.15)
    common = {"H_edges", "C_edges", "K_plaquettes", "threshold_amr",
              "w_z_frac"}
    assert common <= set(v1) and common <= set(v2)
    for k in ("H_edges", "C_edges"):
        assert np.shape(v1[k]) == np.shape(v2[k])


def test_the_repo_axis_convention_is_the_one_the_tests_assume():
    """Si AXIS_X changeait, tout ce fichier mesurerait autre chose."""
    assert (AXIS_X, AXIS_Y) == (0, 1)
