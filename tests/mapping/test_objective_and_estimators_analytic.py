"""La fonction objectif et les estimateurs restants, sur entrées connues.

`pipeline.score` est la fonction que TOUTE l'optimisation minimise : chacun
des 345 essais Optuna a été jugé par elle. Elle n'avait aucun test.

Deux écarts trouvés en la relisant, tous deux vérifiés ici :

  D-3  Sa carte de poids construisait une « vorticité » avec la convention
       d'axes inversée — le même défaut que dans les mappeurs, ici au cœur
       de l'objectif. Sur une rotation solide elle valait 0 au lieu de 2 :
       les tourbillons ne recevaient aucun sur-poids, alors que c'est la
       raison d'être de la pondération. CORRIGÉ.

  D-4  Le commentaire annonçait `w = 1 + 0.5*(...)` là où le code applique
       `1 + 0.25*(...)`. CORRIGÉ (commentaire aligné sur le calcul).

  D-5  Le chemin de divergence notait avec une formule NON pondérée, sous
       la même clé `combined`. CORRIGÉ : les deux chemins partagent
       `instability_weight_map` et `weighted_relative_error`.
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


_REPO_ROOT = _repo_root()
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.PhysToAngle import AngleMapper, _lohner_estimator   # noqa: E402
from Simulation.grid import AXIS_X, AXIS_Y, PeriodicGrid            # noqa: E402
from pipeline import score                                          # noqa: E402

FIELDS = ("vx", "vy", "Bx", "By", "Jz")
N = 16
NSQ = N * N


def _flux(seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return {k: scale * rng.standard_normal((N, N)) for k in FIELDS}


# ═══════════════════════════════════════════════════════════════════════
#  1. score() — la fonction que 345 essais Optuna ont minimisée
# ═══════════════════════════════════════════════════════════════════════

def test_a_perfect_reconstruction_has_zero_physical_error():
    """Le controle le plus elementaire, et il n'existait pas."""
    f = _flux()
    out = score(f, f, 0.5, NSQ * 10, 10, NSQ)
    assert out["phys_score"] == pytest.approx(0.0, abs=1e-12)
    for v in out["field_errors"].values():
        assert v == pytest.approx(0.0, abs=1e-12)


def test_the_result_carries_every_documented_key():
    out = score(_flux(0), _flux(1), 0.5, NSQ, 1, NSQ)
    assert set(out) == {"combined", "phys_score", "patch_ratio", "field_errors"}
    assert set(out["field_errors"]) == set(FIELDS)


def test_predicting_zero_gives_a_relative_error_of_one():
    """Erreur relative : un bras qui rend zero doit obtenir exactement 1.

    C'est le point d'ancrage de l'echelle. Sans lui, on ne sait pas ce que
    « phys_score = 0.4 » veut dire.
    """
    ref = _flux(2)
    zero = {k: np.zeros((N, N)) for k in FIELDS}
    out = score(zero, ref, 0.0, NSQ, 1, NSQ)
    for k, v in out["field_errors"].items():
        assert v == pytest.approx(1.0, rel=1e-9), f"{k} : {v}"
    assert out["phys_score"] == pytest.approx(1.0, rel=1e-9)


def test_the_physical_error_is_scale_invariant():
    """Multiplier les DEUX bras par c ne doit rien changer.

    L'erreur est normalisee par le RMS de reference ; si elle ne l'etait
    pas, un scenario a champs forts dominerait la perte composite.
    """
    q, t = _flux(3), _flux(4)
    base = score(q, t, 0.5, NSQ, 1, NSQ)["phys_score"]
    for c in (1e-3, 1e3):
        scaled_q = {k: c * v for k, v in q.items()}
        scaled_t = {k: c * v for k, v in t.items()}
        got = score(scaled_q, scaled_t, 0.5, NSQ, 1, NSQ)["phys_score"]
        #  L'invariance n'est pas EXACTE : le denominateur porte un
        #  `epsilon_security = 1e-10` additif, dont le poids relatif croit
        #  quand les champs faiblissent. Tolerance derivee de cet epsilon.
        assert got == pytest.approx(base, rel=1e-6), f"c={c}"


def test_the_security_epsilon_breaks_scale_invariance_on_weak_fields():
    """Le prix de `ref_rms + 1e-10`, mesure au lieu d'etre suppose.

    L'ecart relatif vaut ~ eps / ref_rms : negligeable a amplitude 1,
    mais il atteint le pourcent des que les champs descendent vers 1e-8.
    Un scenario a champs tres faibles verrait donc son erreur SOUS-estimee.
    """
    q, t = _flux(3), _flux(4)
    base = score(q, t, 0.0, NSQ, 1, NSQ)["phys_score"]
    dev = {}
    for c in (1.0, 1e-4, 1e-8):
        sq = {k: c * v for k, v in q.items()}
        st = {k: c * v for k, v in t.items()}
        got = score(sq, st, 0.0, NSQ, 1, NSQ)["phys_score"]
        dev[c] = abs(got - base) / base
    assert dev[1.0] < 1e-12
    assert dev[1e-8] > dev[1e-4] > dev[1.0], (
        f"la derive ne croit pas quand les champs faiblissent : {dev}")
    assert dev[1e-8] > 1e-4, (
        f"derive a amplitude 1e-8 : {dev[1e-8]:.3e} — si l'epsilon a ete "
        "rendu relatif, mettre a jour ce test")


def test_the_physical_error_is_the_mean_of_the_five_fields():
    out = score(_flux(5), _flux(6), 0.5, NSQ, 1, NSQ)
    assert out["phys_score"] == pytest.approx(
        np.mean(list(out["field_errors"].values())), rel=1e-12)


def test_a_larger_deviation_scores_worse():
    """Monotonie : plus on s'ecarte, plus le score monte."""
    ref = _flux(7)
    prev = -1.0
    for eps in (0.0, 0.1, 0.5, 1.0):
        q = {k: v + eps * np.ones((N, N)) for k, v in ref.items()}
        s = score(q, ref, 0.0, NSQ, 1, NSQ)["phys_score"]
        assert s > prev, f"non monotone a eps={eps}"
        prev = s


# ── la combinaison coût / fidélité ───────────────────────────────────

def test_lambda_zero_reduces_the_score_to_the_physical_error():
    out = score(_flux(8), _flux(9), 0.0, NSQ // 2, 1, NSQ)
    assert out["combined"] == pytest.approx(out["phys_score"], rel=1e-12)


def test_a_huge_lambda_reduces_the_score_to_the_patch_ratio():
    out = score(_flux(8), _flux(9), 1e9, NSQ // 4, 1, NSQ)
    assert out["combined"] == pytest.approx(out["patch_ratio"], rel=1e-6)


def test_the_combined_score_is_a_weighted_mean_of_its_two_terms():
    """Il doit rester ENTRE ses deux composantes, quel que soit lambda."""
    q, t = _flux(10), _flux(11)
    for lam in (0.0, 0.25, 1.0, 4.0):
        out = score(q, t, lam, NSQ // 3, 1, NSQ)
        lo = min(out["phys_score"], out["patch_ratio"])
        hi = max(out["phys_score"], out["patch_ratio"])
        assert lo - 1e-12 <= out["combined"] <= hi + 1e-12, f"lambda={lam}"


def test_the_combined_score_matches_its_formula():
    q, t = _flux(12), _flux(13)
    lam = 0.7
    out = score(q, t, lam, NSQ // 2, 1, NSQ)
    assert out["combined"] == pytest.approx(
        (out["phys_score"] + lam * out["patch_ratio"]) / (1 + lam), rel=1e-12)


@pytest.mark.parametrize("used,steps,expected", [
    (NSQ, 1, 1.0), (NSQ // 2, 1, 0.5), (NSQ * 4, 4, 1.0), (0, 3, 0.0),
])
def test_the_patch_ratio_is_the_average_cost_per_step(used, steps, expected):
    out = score(_flux(14), _flux(15), 0.5, used, steps, NSQ)
    assert out["patch_ratio"] == pytest.approx(expected)


def test_zero_steps_now_raises_instead_of_scoring(_lambda_seuil=0.2361):
    """Un run qui n'a fait AUCUN pas LEVE — il ne se note plus (D-67).

    Ce test epinglait l'ancien comportement : `patch_ratio = 1.0`, donc un
    score defini pour un run qui n'avait rien calcule. Sa docstring notait
    deja le risque — « exploitable par Optuna au lieu d'echouer » — et le
    figeait comme « choix visible ». D-66 a montre que ce choix rendait un
    defaut invisible : la CLI ne tournait pas du tout et annoncait
    `combined = 0.333333` sans que rien ne signale l'anomalie.

    Le risque etait reel et se chiffre. Avec les valeurs mesurees du run
    de reference (phys = 0.140052, patch = 0.4067), un run vide bat un run
    reel des que

        lambda < phys / (1 - patch) = 0.2361

    A `LAMBDA_COST_SOFT = 0.4` le run vide perd. Mais `recompute_lambda_scores`
    rescore les essais a lambda = 0.0, 0.1, 0.2 — et sous ces trois valeurs
    un essai degenere devient le MEILLEUR essai de la campagne.

    Remesure, pas ajustee : l'ancienne valeur (1.0) et la nouvelle (une
    exception) sont toutes deux consignees ici.
    """
    with pytest.raises(ValueError, match="total_steps"):
        score(_flux(16), _flux(17), 0.5, 0, 0, NSQ)

    # Le garde-fou du chiffre ci-dessus : sous le seuil, le vide gagnait.
    phys, patch = 0.140052, 0.4067
    for lam in (0.0, 0.1, 0.2):
        vide = lam / (1 + lam)
        reel = (phys + lam * patch) / (1 + lam)
        assert vide < reel, (
            f"a lambda={lam}, un run vide ({vide:.4f}) devrait battre un run "
            f"reel ({reel:.4f}) — si ce n'est plus le cas, remesurer le seuil")
    for lam in (0.4, 0.5):
        assert lam / (1 + lam) > (phys + lam * patch) / (1 + lam), (
            f"a lambda={lam}, le run vide devrait perdre")


def test_zero_steps_guard_does_not_break_the_nominal_path():
    """Garde-fou : refuser zero pas ne doit pas refuser un pas."""
    out = score(_flux(16), _flux(17), 0.5, NSQ // 2, 1, NSQ)
    assert out["patch_ratio"] == pytest.approx(0.5)
    assert np.isfinite(out["combined"])


def test_a_zero_reference_field_does_not_produce_nan():
    """Le garde epsilon doit tenir sur un champ de reference nul."""
    t = {k: np.zeros((N, N)) for k in FIELDS}
    q = _flux(18)
    out = score(q, t, 0.5, NSQ, 1, NSQ)
    assert np.isfinite(out["combined"])
    assert all(np.isfinite(v) for v in out["field_errors"].values())


# ── la carte de poids ────────────────────────────────────────────────

def _weight_map(t):
    """Reproduit la ponderation de `score` pour l'inspecter."""
    Jz_abs = np.abs(t["Jz"])
    Jz_mean = np.mean(Jz_abs) + 1e-10
    omega = np.abs((np.roll(t["vy"], -1, axis=AXIS_X) - t["vy"])
                   - (np.roll(t["vx"], -1, axis=AXIS_Y) - t["vx"]))
    omega_mean = np.mean(omega) + 1e-10
    return 1.0 + 0.5 * (Jz_abs / Jz_mean + omega / omega_mean) * 0.5


def test_the_weight_map_is_uniform_on_a_featureless_reference():
    """Sans structure, aucune region ne doit etre privilegiee."""
    t = {k: np.ones((N, N)) for k in FIELDS}
    w = _weight_map(t)
    assert np.ptp(w) < 1e-12
    #  Jz uniforme -> Jz/mean = 1, omega = 0 -> w = 1 + 0.25*1 = 1.25
    assert w[0, 0] == pytest.approx(1.25, rel=1e-9)


def test_the_weight_map_favours_strong_current_regions():
    """La promesse du docstring : les nappes de courant pesent plus."""
    t = {k: np.zeros((N, N)) for k in FIELDS}
    t["Jz"][8, 8] = 100.0
    w = _weight_map(t)
    assert w[8, 8] > 10 * w[0, 0], (
        "une nappe de courant isolee ne recoit pas de sur-poids")


def test_the_weight_map_actually_changes_the_score():
    """Une ponderation qui ne pondere rien serait indiscernable de son
    absence."""
    t = {k: np.zeros((N, N)) for k in FIELDS}
    t["Jz"][8, 8] = 50.0
    t["vx"] = np.ones((N, N))
    err_here = dict(t); err_here = {k: v.copy() for k, v in t.items()}
    err_here["vx"] = t["vx"].copy(); err_here["vx"][8, 8] += 1.0
    err_far = {k: v.copy() for k, v in t.items()}
    err_far["vx"] = t["vx"].copy(); err_far["vx"][0, 0] += 1.0
    s_here = score(err_here, t, 0.0, NSQ, 1, NSQ)["field_errors"]["vx"]
    s_far = score(err_far, t, 0.0, NSQ, 1, NSQ)["field_errors"]["vx"]
    assert s_here > s_far, (
        "une erreur dans la region ponderee ne coute pas plus cher "
        f"({s_here:.3e} contre {s_far:.3e})")


# ── D-3 et D-4 ───────────────────────────────────────────────────────

def test_the_objective_weight_map_now_sees_solid_rotation():
    """D-3 corrige : la ponderation sur-pondere enfin les tourbillons.

    Elle formait auparavant dv_y/dy - dv_x/dx, exactement nulle sur une
    rotation solide : les tourbillons ne recevaient aucun sur-poids, alors
    que c'est la raison d'etre de la carte.
    """
    g = PeriodicGrid(64)
    vx, vy = -g.Y, g.X                       # rotation solide, omega = 2
    inner = (slice(0, -1), slice(0, -1))
    om = np.abs((np.roll(vy, -1, axis=AXIS_X) - vy)
                - (np.roll(vx, -1, axis=AXIS_Y) - vx))[inner] / g.dx
    assert np.mean(om) == pytest.approx(2.0, rel=1e-9)

    #  et la carte reelle, sur un tourbillon localise, doit sur-ponderer
    n = 32
    gg = PeriodicGrid(n)
    r2 = (gg.X - np.pi) ** 2 + (gg.Y - np.pi) ** 2
    swirl = np.exp(-r2 / 0.3)
    t = {k: np.zeros((n, n)) for k in FIELDS}
    t["vx"] = -(gg.Y - np.pi) * swirl
    t["vy"] = (gg.X - np.pi) * swirl
    from pipeline import instability_weight_map
    w = instability_weight_map(t)
    centre = w[n // 2, n // 2]
    corner = w[0, 0]
    assert centre > 1.5 * corner, (
        f"le tourbillon ne recoit pas de sur-poids : centre {centre:.3f}, "
        f"coin {corner:.3f}")


def test_the_weight_formula_matches_its_documentation():
    """D-4 corrige : le commentaire annonce desormais 0.25, comme le code.

    Il annoncait `1 + 0.5*(...)` alors que le code applique
    `1 + 0.5*(...)*0.5`, soit un coefficient deux fois plus petit.
    """
    src = open(os.path.join(_SRC, "pipeline.py"), encoding="utf-8").read()
    assert "w = 1 + 0.25 × (|Jz|/⟨|Jz|⟩ + |ωz|/⟨|ωz|⟩)" in src, (
        "la documentation n'annonce plus le facteur effectif 0.25")
    assert "1 + 0.5*(|Jz|" not in src, (
        "l'ancienne annonce a 0.5 est revenue")
    t = {k: np.ones((N, N)) for k in FIELDS}
    from pipeline import instability_weight_map
    assert instability_weight_map(t)[0, 0] == pytest.approx(1.25)


def test_the_weight_map_helper_matches_the_score_it_feeds():
    """La carte extraite doit etre celle que `score` applique reellement."""
    from pipeline import instability_weight_map
    t = _flux(21)
    np.testing.assert_allclose(instability_weight_map(t), _weight_map(t),
                               rtol=0, atol=1e-15)


# ═══════════════════════════════════════════════════════════════════════
#  2. Estimateur de Löhner
# ═══════════════════════════════════════════════════════════════════════

def test_lohner_is_zero_on_a_uniform_field():
    assert np.max(np.abs(_lohner_estimator(np.full((16, 16), 4.0)))) < 1e-12


def test_lohner_ignores_a_linear_ramp():
    """« insensitive to smooth gradients » : derivee seconde nulle.

    On mesure a l'interieur, le raccord periodique d'une rampe etant une
    discontinuite reelle que l'estimateur DOIT voir.
    """
    g = PeriodicGrid(64)
    e = _lohner_estimator(g.X)[2:-2, 2:-2]
    assert np.max(np.abs(e)) < 1e-9


def test_lohner_separates_a_discontinuity_by_distribution_not_by_maximum():
    """« peaks at discontinuities » — vrai en distribution, pas en maximum.

    Mesure a N=64 :

        champ            max      mediane   fraction > 0.5
        sin(x)           0.9631   0.0467    3.1 %
        marche           1.0000   0.0000    6.2 %

    Le maximum ne separe presque pas les deux (0.963 contre 1.000) parce
    que l'estimateur culmine la ou la derivee PREMIERE s'annule — sur
    sin(x), le maximum tombe exactement en x = pi/2. C'est le comportement
    connu du denominateur de Löhner, que le terme de garde `eps*|f|` avec
    eps = 0.01 ne neutralise qu'en partie ; la formule implementee est
    bien celle publiee.

    Consequence pratique : un extremum lisse de |B| est signale comme une
    discontinuite. Ce qui distingue reellement les deux regimes est la
    MEDIANE, nulle sur la marche et non nulle sur le champ lisse.
    """
    g = PeriodicGrid(64)
    smooth = np.sin(g.X)
    step = np.zeros((64, 64))
    step[:, 32:] = 1.0

    e_s = _lohner_estimator(smooth)
    e_d = _lohner_estimator(step)

    #  Le maximum ne discrimine pas.
    assert e_d.max() < 2.0 * e_s.max(), (
        "le maximum separe desormais les deux regimes : l'estimateur a "
        "change, mettre a jour ce test")

    #  La mediane, si.
    assert np.median(e_d) < 1e-12
    assert np.median(e_s) > 1e-3

    #  Et le maximum du champ lisse tombe bien a l'extremum de sin.
    i, j = np.unravel_index(np.argmax(e_s), e_s.shape)
    assert abs(abs(smooth[i, j]) - 1.0) < 1e-6, (
        "le maximum ne tombe plus a l'extremum du champ")


def test_lohner_is_quiet_between_the_extrema_of_a_smooth_field():
    """Loin des extremums, un champ lisse doit rester bas."""
    g = PeriodicGrid(64)
    e = _lohner_estimator(np.sin(g.X))
    assert np.median(e) < 0.1, (
        f"mediane {np.median(e):.3f} : le champ lisse est signale partout")


def test_lohner_is_bounded_and_finite():
    rng = np.random.default_rng(0)
    for f in (rng.standard_normal((32, 32)), np.zeros((32, 32)),
              np.full((32, 32), 1e12)):
        e = _lohner_estimator(f)
        assert np.all(np.isfinite(e))
        assert np.all(e >= 0.0)


def test_lohner_is_symmetric_under_axis_exchange():
    """L'estimateur est symetrique en x et y — donc l'inversion d'axes
    qu'il porte dans ses NOMS (`fp_x` depuis axis=1) est sans consequence
    numerique. On le verifie pour pouvoir l'ecrire.
    """
    rng = np.random.default_rng(1)
    f = rng.standard_normal((32, 32))
    np.testing.assert_allclose(_lohner_estimator(f.T), _lohner_estimator(f).T,
                               rtol=1e-12, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════
#  3. Flux de contrainte
# ═══════════════════════════════════════════════════════════════════════

def _state(N_=16, uniform=False, seed=0):
    if uniform:
        return {k: np.full((N_, N_), 1.0 + i) for i, k in enumerate(FIELDS)}
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal((N_, N_)) for k in FIELDS}


def test_stress_flux_returns_both_orientations_at_full_size():
    phi = AngleMapper().compute_stress_flux(_state())
    assert set(phi) >= {"phi_horizontal", "phi_vertical"}
    for k in ("phi_horizontal", "phi_vertical"):
        assert np.asarray(phi[k]).shape == (N, N)


def test_stress_flux_vanishes_on_a_uniform_state():
    """Aucun saut entre voisins -> aucun flux de contrainte."""
    phi = AngleMapper().compute_stress_flux(_state(uniform=True))
    for k in ("phi_horizontal", "phi_vertical"):
        assert np.max(np.abs(phi[k])) < 1e-12, f"{k} non nul sur champ plat"


def test_stress_flux_is_never_negative():
    phi = AngleMapper().compute_stress_flux(_state(seed=3))
    for k in ("phi_horizontal", "phi_vertical"):
        assert np.all(np.asarray(phi[k]) >= 0.0)


def test_stress_flux_separates_the_two_orientations():
    """Un champ qui ne varie que le long d'un axe ne doit charger qu'un
    seul flux. Sans cela, l'information directionnelle serait perdue."""
    s = {k: np.zeros((16, 16)) for k in FIELDS}
    s["vx"][::2, :] = 1.0                 # varie le long de l'axe 0
    phi = AngleMapper().compute_stress_flux(s)
    h = float(np.max(np.abs(phi["phi_horizontal"])))
    v = float(np.max(np.abs(phi["phi_vertical"])))
    assert abs(h - v) > 1e-9, (
        f"les deux orientations repondent pareil (h={h:.3e}, v={v:.3e}) : "
        "l'information directionnelle est perdue")


# ═══════════════════════════════════════════════════════════════════════
#  4. Opérateurs de grille restants
# ═══════════════════════════════════════════════════════════════════════

def test_grid_laplacian_matches_an_eigenfunction():
    """lap(sin x sin y) = -2 sin x sin y, au second ordre."""
    g = PeriodicGrid(256)
    f = np.sin(g.X) * np.sin(g.Y)
    np.testing.assert_allclose(g.laplacian(f), -2.0 * f, rtol=2e-4, atol=2e-4)


def test_grid_laplacian_is_second_order():
    """Ordre 2, a distinguer du `_fd_laplacian` d'ordre 4 du solveur."""
    errs = []
    for n in (32, 64, 128):
        g = PeriodicGrid(n)
        f = np.sin(g.X) * np.cos(2 * g.Y)
        exact = -5.0 * f
        errs.append(float(np.max(np.abs(g.laplacian(f) - exact))))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert all(1.7 < o < 2.3 for o in orders), f"ordre {orders}"


def test_grid_laplacian_is_zero_on_a_constant():
    g = PeriodicGrid(16)
    assert np.max(np.abs(g.laplacian(np.full((16, 16), 7.0)))) < 1e-9


def test_smoothing_preserves_the_mean_and_reduces_variance():
    """Un filtre passe-bas ne doit pas deplacer la moyenne."""
    rng = np.random.default_rng(0)
    g = PeriodicGrid(32)
    f = rng.standard_normal((32, 32))
    s = g.smooth_field(f)
    assert s.mean() == pytest.approx(f.mean(), rel=1e-9)
    assert s.var() < f.var(), "le lissage n'a pas reduit la variance"


def test_smoothing_leaves_a_constant_untouched():
    g = PeriodicGrid(16)
    c = np.full((16, 16), 2.5)
    np.testing.assert_allclose(g.smooth_field(c), c, rtol=0, atol=1e-12)


def test_patch_extraction_wraps_and_matches_a_plain_slice():
    g = PeriodicGrid(8)
    a = np.arange(64, dtype=float).reshape(8, 8)
    np.testing.assert_array_equal(g.extract_patch_data(a, 2, 3, 4),
                                  a[2:6, 3:7])
    #  a cheval sur le bord
    wrapped = g.extract_patch_data(a, 6, 6, 4)
    assert wrapped.shape == (4, 4)
    assert wrapped[0, 0] == a[6, 6]
    assert wrapped[2, 2] == a[0, 0]


def test_patch_extraction_of_the_whole_grid_is_the_grid():
    g = PeriodicGrid(8)
    a = np.arange(64, dtype=float).reshape(8, 8)
    np.testing.assert_array_equal(g.extract_patch_data(a, 0, 0, 8), a)


def test_second_order_jump_is_zero_on_a_linear_field():
    """Il mesure une COURBURE : un champ lineaire ne doit rien produire.

    Mesure a l'interieur, le raccord periodique d'une rampe etant une
    vraie discontinuite.
    """
    g = PeriodicGrid(32)
    zero = np.zeros((32, 32))
    for axis in (AXIS_X, AXIS_Y):
        j = g._get_second_order_jump(g.X, zero, axis)[3:-3, 3:-3]
        assert np.max(np.abs(j)) < 1e-9, f"axis={axis} : {np.max(np.abs(j)):.3e}"


def test_second_order_jump_reacts_to_curvature():
    g = PeriodicGrid(32)
    zero = np.zeros((32, 32))
    curved = np.sin(g.X)
    flat = g._get_second_order_jump(g.X, zero, AXIS_X)[3:-3, 3:-3].max()
    bent = g._get_second_order_jump(curved, zero, AXIS_X)[3:-3, 3:-3].max()
    assert bent > flat + 1e-9


# ═══════════════════════════════════════════════════════════════════════
#  5. Le chemin de divergence de pipeline()
# ═══════════════════════════════════════════════════════════════════════

_PIPELINE_SRC = open(os.path.join(_SRC, "pipeline.py"), encoding="utf-8").read()


def test_both_scoring_paths_now_use_the_same_formula():
    """D-5 corrige : une seule definition de l'erreur, partagee.

    Le chemin de divergence calculait une L2 NON ponderee la ou `score`
    pondere ; les deux partaient vers Optuna sous la meme cle `combined`,
    avec un ecart de 1.8 % sur un champ a nappe de courant. Ils partagent
    desormais `instability_weight_map` et `weighted_relative_error`.
    """
    from pipeline import instability_weight_map, weighted_relative_error

    assert "field_errors[var] = weighted_relative_error(" in _PIPELINE_SRC, (
        "le chemin de divergence n'utilise plus le calcul partage")
    assert "rel_err = np.sqrt(np.mean((arr_q - arr_r)**2))" not in _PIPELINE_SRC

    rng = np.random.default_rng(0)
    n = 32
    t = {k: rng.standard_normal((n, n)) * 0.1 for k in FIELDS}
    t["Jz"][14:18, :] = 5.0
    q = {k: v + rng.standard_normal((n, n)) * 0.05 for k, v in t.items()}

    w = instability_weight_map(t).flatten()
    w_sum = w.sum()
    via_helper = float(np.mean([
        weighted_relative_error(q[v], t[v], w, w_sum) for v in FIELDS]))
    via_score = score(q, t, 0.0, n * n, 1, n * n)["phys_score"]
    assert via_score == pytest.approx(via_helper, rel=1e-15), (
        f"les deux chemins divergent encore : {via_score} vs {via_helper}")


def test_the_divergence_penalty_has_a_single_definition():
    """Elle etait redefinie quatre fois, dont trois masquaient la constante
    de module : changer la valeur en tete n'aurait eu d'effet qu'une fois
    sur quatre."""
    import re
    values = re.findall(r"DIVERGENCE_PENALTY\s*=\s*([0-9.]+)", _PIPELINE_SRC)
    assert len(values) == 1, (
        f"{len(values)} definitions : {values}. Le comportement redeviendrait "
        "dependant de l'endroit du code ou la penalite est lue")
    assert values[0] == "10.0"


def test_a_scoring_bug_is_now_distinguishable_from_a_divergence():
    """`except Exception` avalait tout sans le nommer.

    Le filet est conserve — un essai Optuna ne doit pas faire tomber la
    campagne — mais la cause est journalisee et rendue sous la cle
    `scoring_error`, donc distinguable d'une divergence physique.

    Remesure du 12 aout (D-36). La version precedente cherchait la chaine
    `_out['scoring_error'] = scoring_error` dans le source. Elle a disparu
    quand les QUATRE sorties detaillees de `pipeline` ont ete routees par
    un `_details` unique — auparavant une seule portait la provenance,
    celle du chemin de divergence, donc la trace n'existait que sur les
    runs qu'on jette. Le test interroge desormais l'AST : toute sortie
    sous `if return_details` passe par le helper, quel que soit le nom
    des variables locales.
    """
    import ast

    assert "except Exception as exc:" in _PIPELINE_SRC
    assert "[SCORING-ERROR]" in _PIPELINE_SRC
    assert "traceback.print_exc" in _PIPELINE_SRC

    tree = ast.parse(_PIPELINE_SRC)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "pipeline")

    #  le helper existe et pose la cle
    helper = next(n for n in ast.walk(fn)
                  if isinstance(n, ast.FunctionDef) and n.name == "_details")
    poses = [ast.unparse(n) for n in ast.walk(helper)
             if isinstance(n, ast.Assign) and "scoring_error" in ast.unparse(n)]
    assert poses, "`_details` ne pose pas `scoring_error`"

    #  et AUCUNE sortie detaillee ne s'en echappe
    echappees = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.If)
                and "return_details" in ast.unparse(node.test)):
            continue
        for ret in (n for n in ast.walk(node) if isinstance(n, ast.Return)):
            texte = ast.unparse(ret)
            if not ("_details(" in texte or "_divergence_details(" in texte):
                echappees.append(texte[:80])
    assert echappees == [], (
        f"sorties detaillees sans provenance : {echappees}")

    assert "scoring_error = None" in _PIPELINE_SRC, (
        "la cle doit valoir None quand tout s'est bien passe")
