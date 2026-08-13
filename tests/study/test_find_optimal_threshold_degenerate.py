"""D-43 : `hamiltonian_coefficients.find_optimal_threshold` balayait ses
seuils avec `flat_e >= thr`. Sur une energie CONSTANTE — le cas mesure par
D-40/D-41, ou aucun coefficient v1 ne franchit son seuil critique — les 100
percentiles valent tous la meme chose et chaque candidat predit TOUS les
patchs durs. Le F1 rendu etait celui du classifieur tout-positif,
2p/(p+1), pas une mesure de separation : 0.400 a prevalence 0.250, 0.376 a
0.231. A cote du 0.519 authentique d'orszag_tang, cela se lit comme un
signal reel un peu plus faible.

Le sibling de cette fonction dans `pipeline_verification.py` rend 0.000 et
signale `degenerate_E` sur exactement les memes donnees (D-40).
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hamiltonian_coefficients as hc


def _all_positive_f1(prevalence):
    """F1 du classifieur qui predit TOUT positif : precision=p, rappel=1."""
    return 2.0 * prevalence / (prevalence + 1.0)


def test_constant_energy_yields_no_threshold():
    """Une energie constante ne separe rien : NaN, pas un F1."""
    is_hard = np.zeros((5, 4, 4), dtype=bool)
    is_hard.reshape(-1)[:20] = True          # prevalence exactement 0.250
    E = np.zeros((5, 4, 4))

    thr, f1, thrs, f1s = hc.find_optimal_threshold(E, is_hard)

    assert np.isnan(thr)
    assert np.isnan(f1)
    assert np.all(np.isnan(thrs))
    assert np.all(np.isnan(f1s))


def test_old_behaviour_was_the_all_positive_f1():
    """
    Epingle ce que l'ancienne version rendait, pour que la correction ne
    puisse pas etre defaite en silence : avec `>=` et une entree
    constante, le balayage predisait tout positif, donc F1 = 2p/(p+1).

    Ce test REFAIT le calcul de l'ancienne version. S'il cessait de valoir
    0.400 / 0.376 c'est que la prevalence a change, pas la correction.
    """
    # 20/80 = prevalence 0.250 -> 0.400, la valeur mesuree sur
    # harris_tearing ; 18/80 = 0.225 -> 0.3673, du meme ordre que le 0.376
    # mesure sur kelvin_helmholtz (prevalence 0.231).
    for n_hard, expected in ((20, 0.400), (18, 0.3673)):
        is_hard = np.zeros((5, 4, 4), dtype=bool)
        is_hard.reshape(-1)[:n_hard] = True
        flat_h = is_hard.flatten()
        flat_e = np.zeros(flat_h.size)

        # l'ancien balayage, tel qu'il etait ecrit
        thresholds = np.percentile(flat_e, np.linspace(5, 95, 100))
        pred = flat_e >= thresholds[0]
        assert pred.all(), "le balayage constant predisait bien tout positif"

        tp = np.sum(pred & flat_h)
        fp = np.sum(pred & ~flat_h)
        prec = tp / (tp + fp)
        old_f1 = 2 * prec * 1.0 / (prec + 1.0)

        assert old_f1 == pytest.approx(expected, abs=1e-3)
        assert old_f1 == pytest.approx(_all_positive_f1(flat_h.mean()), abs=1e-12)

        # la nouvelle version refuse de rendre ce nombre
        assert np.isnan(hc.find_optimal_threshold(flat_e.reshape(is_hard.shape),
                                                  is_hard)[1])


def test_varying_energy_still_returns_a_real_threshold():
    """
    Le champ qui SEPARE : une energie correlee a la durete doit rendre un
    seuil et un F1 nettement au-dessus du tout-positif, sinon le test ne
    distingue pas la correction d'un NaN systematique.
    """
    rng = np.random.default_rng(0)
    l2 = rng.random((5, 4, 4))
    is_hard = l2 >= np.percentile(l2, 75)
    E = l2 + 0.01 * rng.random(l2.shape)

    thr, f1, thrs, f1s = hc.find_optimal_threshold(E, is_hard)

    assert np.isfinite(thr)
    assert f1 > _all_positive_f1(is_hard.mean()) + 0.2
    assert np.all(np.isfinite(f1s))


def test_real_artifacts_degenerate_exactly_where_D40_said():
    """
    Sur les artefacts reels : harris_tearing et kelvin_helmholtz ont une
    energie v1 constante (E.ptp = 0), mhd_rotor et orszag_tang non
    (1.819 et 0.742). La fonction doit rendre NaN sur les deux premiers et
    un F1 fini sur les deux autres — 0.950 et 0.519 mesures.
    """
    results_dir = os.path.join(_REPO_ROOT, "results")
    expected = {
        "harris_tearing": None,
        "kelvin_helmholtz": None,
        "mhd_rotor": 0.950,
        "orszag_tang": 0.519,
    }

    seen = 0
    for scenario, want in expected.items():
        coef = os.path.join(
            results_dir, f"coefficients_{scenario}_Re400_N256_dim4.npz")
        patch = os.path.join(
            results_dir, f"patches_{scenario}_Re400_N256_dim4.npz")
        if not (os.path.exists(coef) and os.path.exists(patch)):
            continue
        seen += 1

        c = np.load(coef)
        E = c[next(k for k in c.files if k.endswith("_E"))]
        p = np.load(patch)
        is_hard_full = p["is_hard"]
        n_full = is_hard_full.shape[0]
        idx = list(range(0, n_full, max(1, n_full // 10)))
        if len(idx) < 3:
            idx = list(range(n_full))
        is_hard = is_hard_full[idx[:E.shape[0]]]

        _, f1, _, _ = hc.find_optimal_threshold(E, is_hard)
        if want is None:
            assert np.isnan(f1), f"{scenario}: E constant, F1 devrait etre NaN"
        else:
            assert f1 == pytest.approx(want, abs=5e-3), scenario

    # un balayage vide doit crier
    assert seen == 4, f"{seen}/4 artefacts trouves — le test n'a rien mesure"
