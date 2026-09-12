"""D-70 — `_hard_patches` (h1_curl_convention_gap.py) doit calculer la MEME
grandeur que `study/pipeline/hard_patch_labels.py::patch_l2_errors`, comme
sa docstring l'annonce : l'erreur L2 de reconstruction par grossissement en
bloc puis relevement, sommee sur les 4 champs, normalisee par le RMS global.

Avant D-70 la fonction calculait l'ecart-type intra-patch de la NORME du
champ (sqrt(vx^2+vy^2+Bx^2+By^2)) — une formule differente qui s'accorde
avec la definition canonique sur un champ lisse mais l'INVERSE des qu'un
patch oscille a magnitude constante : l'ecart-type y est nul (rien ne
"varie" en norme) alors que l'information fine y est totalement detruite
par le grossissement en bloc, donc l'erreur de reconstruction y est
maximale. Les deux formules ne classaient pas les patches dans le meme
ordre — la "durete" utilisee comme verite terrain dans tout T31 n'etait
donc pas celle que la docstring revendiquait.
"""

import importlib.util
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_T31_SCRIPT = os.path.join(_REPO_ROOT, "study", "h1_solver", "h1_curl_convention_gap.py")
_HP_SCRIPT = os.path.join(_REPO_ROOT, "study", "pipeline", "hard_patch_labels.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def t31():
    return _load(_T31_SCRIPT, "t31_hard_patches_mod")


@pytest.fixture(scope="module")
def hard_patch_labels():
    return _load(_HP_SCRIPT, "hard_patch_labels_mod")


def _old_buggy_formula(vx, vy, Bx, By, N, dim, quantile):
    """L'implementation D-70 remplacee : ecart-type intra-patch de la norme.

    Recopiee ici (et non importee) exprès : c'est le comportement que le
    depot rendait avant la correction, garde pour que ce test puisse
    demontrer qu'il divergeait de la definition canonique.
    """
    ps = N // dim
    mag = np.sqrt(vx ** 2 + vy ** 2 + Bx ** 2 + By ** 2)
    local = mag.reshape(dim, ps, dim, ps).std(axis=(1, 3))
    rms = float(np.sqrt(np.mean(mag ** 2)))
    l2 = local / (rms + 1e-30)
    thr = float(np.quantile(l2, quantile))
    return l2, thr, (l2 > thr)


def _field_with_checkerboard_patch(N=32, dim=4, seed=0):
    """Champ qui SEPARE les deux formules.

    Fond : bruit lisse de faible amplitude (pour un RMS global non nul).
    Patch (0,0) : damier +1/-1 sur vx — magnitude CONSTANTE partout dans
    le patch (donc ecart-type de la norme = 0), mais moyenne de bloc = 0
    (les +1 et -1 s'annulent) : l'information est totalement perdue par
    le grossissement en bloc, donc l'erreur de reconstruction y est
    maximale, pas nulle.
    """
    rng = np.random.default_rng(seed)
    vx = 0.05 * rng.standard_normal((N, N))
    vy = 0.05 * rng.standard_normal((N, N))
    Bx = 0.05 * rng.standard_normal((N, N))
    By = 0.05 * rng.standard_normal((N, N))
    ps = N // dim
    checker = np.indices((ps, ps)).sum(axis=0) % 2
    vx[0:ps, 0:ps] = np.where(checker == 0, 1.0, -1.0)
    # les trois autres champs restent EXACTEMENT nuls dans ce patch : la
    # norme y est alors rigoureusement constante (= 1), pas seulement en
    # moyenne, ce qui rend l'ecart-type nul a la precision machine.
    vy[0:ps, 0:ps] = 0.0
    Bx[0:ps, 0:ps] = 0.0
    By[0:ps, 0:ps] = 0.0
    return vx, vy, Bx, By, ps


def test_new_formula_matches_canonical_patch_l2_errors(t31, hard_patch_labels):
    N, dim = 32, 4
    rng = np.random.default_rng(1)
    vx = rng.standard_normal((N, N))
    vy = rng.standard_normal((N, N))
    Bx = rng.standard_normal((N, N))
    By = rng.standard_normal((N, N))

    l2_canonical = hard_patch_labels.patch_l2_errors(vx, vy, Bx, By, dim)
    l2_new, _, _ = t31._hard_patches(vx, vy, Bx, By, N, dim, 0.75)

    np.testing.assert_allclose(l2_canonical, l2_new, atol=1e-12, rtol=1e-12)


def test_old_formula_disagreed_with_canonical_on_the_checkerboard_patch():
    """Epingle l'ancien comportement : sur ce champ il inversait le classement."""
    N, dim = 32, 4
    vx, vy, Bx, By, ps = _field_with_checkerboard_patch(N, dim)

    l2_old, _, _ = _old_buggy_formula(vx, vy, Bx, By, N, dim, 0.75)

    # La norme est exactement constante dans le patch damier -> ecart-type nul.
    assert l2_old[0, 0] == pytest.approx(0.0, abs=1e-12)
    # Et c'est le MINIMUM de tout le champ : la vieille formule le juge
    # le patch le plus FACILE, alors qu'il concentre toute l'information
    # perdue par le grossissement en bloc.
    assert l2_old[0, 0] == l2_old.min()


def test_new_formula_ranks_the_checkerboard_patch_as_the_hardest(t31, hard_patch_labels):
    """Ce que D-70 restaure : le damier doit ressortir comme le patch le
    plus difficile, conformement a `patch_l2_errors` (verite canonique)."""
    N, dim = 32, 4
    vx, vy, Bx, By, ps = _field_with_checkerboard_patch(N, dim)

    l2_canonical = hard_patch_labels.patch_l2_errors(vx, vy, Bx, By, dim)
    l2_new, _, is_hard_new = t31._hard_patches(vx, vy, Bx, By, N, dim, 0.75)

    assert np.unravel_index(np.argmax(l2_canonical), l2_canonical.shape) == (0, 0)
    assert np.unravel_index(np.argmax(l2_new), l2_new.shape) == (0, 0)
    assert is_hard_new[0, 0]
    np.testing.assert_allclose(l2_canonical, l2_new, atol=1e-12, rtol=1e-12)
