"""D-180 : le verdict de la phase 6 dependait du sigma, et le sigma
dependait de l'ordre des membres d'un `.npz`.

`pipeline_verification.analyze`, branche `--v1`, prenait la premiere cle se
terminant par `_E` (`for k in coefs.files: ... break`). La phase 3 en ecrit
SIX dans le meme artefact (`sigma_values = [0.023, 0.05, 0.10, 0.15, 0.20,
0.30]`). Le verdict F1 flippe entre eux, a echantillon apparie — les memes
2 lignes sur 4 survivent au tri de D-40 a TOUS les sigmas :

    sigma   AUC(E)   F1(E)   F1(cl)   verdict F1
    0.023   0.8737   0.7288  0.6543   PASS     <- les nombres publies
    0.050   0.8653   0.6988  0.6543   PASS
    0.100   0.8351   0.6114  0.6543   WARN
    0.150   0.8253   0.6372  0.6543   TIE
    0.200   0.8194   0.6256  0.6543   WARN
    0.300   0.7932   0.6269  0.6543   WARN

Les nombres publies par D-40 et re-verifies par D-77 — 0.874 / 0.729 contre
0.654, verdict PASS — sont ceux de `0.023 = TRAINED_SIGMA`, atteint parce
que `sigma_values` est ecrit dans cet ordre. Aucune ligne de la sortie ne
le disait.

Ces tests echouent sur la version d'avant : `analyze` y ignore son argument
`sigma` (qu'elle n'a pas) et rend les nombres de 0.023 quoi qu'on demande.

Le test 3 est celui qui SEPARE : il exige que deux sigmas differents
rendent des nombres differents. Un test qui n'aurait verifie que 0.023
serait passe sur l'ancienne version comme sur la nouvelle, et n'aurait rien
prouve.
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

import pipeline_verification as pv
from config import TRAINED_SIGMA

RESULTS = os.path.join(_REPO_ROOT, "results")
RE, N, DIM = 400, 256, 4
SCENARIOS = ["orszag_tang", "harris_tearing", "kelvin_helmholtz", "mhd_rotor"]

# Mesures du 20 aout, artefacts du depot, deux executions bit-a-bit
# identiques. Ecrites en clair pour qu'une derive se voie.
PUBLISHED = {"auc_E": 0.874, "f1_E": 0.729, "f1_c": 0.654}
SWEEP = {0.023: (0.8737, 0.7288), 0.050: (0.8653, 0.6988),
         0.100: (0.8351, 0.6114), 0.150: (0.8253, 0.6372),
         0.200: (0.8194, 0.6256), 0.300: (0.7932, 0.6269)}


def _means(sigma):
    """Les trois moyennes que la phase 6 imprime, au sigma demande."""
    rows = []
    for sc in SCENARIOS:
        r = pv.analyze(sc, RE, DIM, N, use_v2=False, sigma=sigma)
        assert r is not None, (
            f"coefficients_{sc}_Re{RE}_N{N}_dim{DIM}.npz absent : ce test "
            f"mesure sur les artefacts du depot, il ne peut pas se taire")
        rows.append(r)
    assert len(rows) == len(SCENARIOS)
    clean, _ = pv.split_degenerate(rows)
    assert clean, "toutes les lignes degenerees : rien n'a ete mesure"
    return (float(np.nanmean([r["auc_E"] for r in clean])),
            float(np.nanmean([r["f1_E"] for r in clean])),
            float(np.nanmean([r.get("f1_c", np.nan) for r in clean])),
            sorted(r["scenario"] for r in clean))


# ------------------------------------------------------------------
# 1. Les nombres publies sont ceux de TRAINED_SIGMA, et le restent
# ------------------------------------------------------------------

def test_the_published_numbers_are_the_trained_sigma_ones():
    """D-40 / D-77 : 0.874 / 0.729 contre 0.654, verdict PASS."""
    assert TRAINED_SIGMA == 0.023
    auc_E, f1_E, f1_c, _ = _means(TRAINED_SIGMA)

    assert round(auc_E, 3) == PUBLISHED["auc_E"], (auc_E, PUBLISHED)
    assert round(f1_E, 3) == PUBLISHED["f1_E"], (f1_E, PUBLISHED)
    assert round(f1_c, 3) == PUBLISHED["f1_c"], (f1_c, PUBLISHED)
    assert f1_E > f1_c + 0.02, "le verdict publie est PASS"


# ------------------------------------------------------------------
# 2. Un sigma absent LEVE — pas de repli sur le premier venu
# ------------------------------------------------------------------

def test_a_missing_sigma_raises_instead_of_falling_back():
    """Le repli silencieux etait le defaut : il rend un nombre juste-en-forme
    attribue au mauvais sigma. L'absence doit crier, et nommer ce qu'il y a."""
    with pytest.raises(KeyError) as exc:
        pv.analyze(SCENARIOS[0], RE, DIM, N, use_v2=False, sigma=0.077)

    msg = str(exc.value)
    assert "s0.077_E" in msg, msg
    assert "s0.023" in msg, f"les sigmas disponibles doivent etre nommes : {msg}"


# ------------------------------------------------------------------
# 3. Le champ qui SEPARE : deux sigmas, deux verdicts
# ------------------------------------------------------------------

@pytest.mark.parametrize("sigma", sorted(SWEEP))
def test_each_sigma_returns_its_own_numbers(sigma):
    """Sur l'ancienne version, les six rendaient les nombres de 0.023."""
    auc_E, f1_E, _, clean = _means(sigma)
    exp_auc, exp_f1 = SWEEP[sigma]

    assert auc_E == pytest.approx(exp_auc, abs=5e-4), (sigma, auc_E, exp_auc)
    assert f1_E == pytest.approx(exp_f1, abs=5e-4), (sigma, f1_E, exp_f1)

    # La comparaison entre sigmas est appariee : c'est ce qui permet de dire
    # que le verdict bouge a cause du sigma, et non de l'echantillon.
    assert clean == ["mhd_rotor", "orszag_tang"], (sigma, clean)


def test_the_pass_verdict_does_not_survive_every_sigma():
    """Le fait scientifique que D-180 remet a USER : le PASS tient a 2/6.

    Il est epingle ici pour qu'un futur « nettoyage » du choix de sigma ne
    puisse pas le faire disparaitre en silence.
    """
    verdicts = {}
    for sigma in sorted(SWEEP):
        _, f1_E, f1_c, _ = _means(sigma)
        verdicts[sigma] = ("PASS" if f1_E > f1_c + 0.02
                           else "TIE" if f1_E > f1_c - 0.02 else "WARN")

    assert verdicts == {0.023: "PASS", 0.050: "PASS", 0.100: "WARN",
                        0.150: "TIE", 0.200: "WARN", 0.300: "WARN"}, verdicts
    assert sum(v == "PASS" for v in verdicts.values()) == 2, verdicts
