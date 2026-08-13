"""D-50 — la phrase de conclusion de T11b n'est pas reproductible.

`h0_qaoa_displacement` imprime l'une de deux phrases opposées selon que
`|progress moyen| < 0,1`. Ce seuil n'a aucune provenance écrite et il
s'applique à une grandeur dont la dispersion inter-exécutions, **mesurée**,
est plus large que la distance de la valeur publiée au seuil.

Ces tests **épinglent la déviation** : le seuil et les deux textes sont
inchangés, seule l'extraction en fonction est nouvelle (même geste que D-46).
Le test qui compte est `test_the_three_measured_repeats_disagree` — il fait
échouer la suite le jour où quelqu'un touche au seuil sans remesurer, parce
qu'il porte sur les nombres réellement mesurés, pas sur une construction.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Trois exécutions de la commande PUBLIÉE (`--N 256 --dim 2 --n-snaps 2`,
# reps 1-4, mapper v2, k_opt=100, shots=4096), moyennes sur les 4 profondeurs.
# Ce ne sont pas des nombres construits : ce sont les mesures de D-50.
MEASURED_REPEATS = (0.1034, 0.0850, 0.0859)
PUBLISHED_MEAN = 0.0854


def test_published_mean_prints_the_flat_reading():
    """La valeur publiée tombe du côté « reste à l'encodage classique »."""
    from h0_qaoa_displacement import reading_message, READING_FLAT

    assert reading_message(PUBLISHED_MEAN) == READING_FLAT


def test_the_three_measured_repeats_disagree():
    """Trois exécutions identiques, deux conclusions opposées.

    C'est le défaut : la phrase publiée est décidée par le tirage. Si un jour
    ce test passe (une seule phrase pour les trois), c'est que le seuil, la
    grandeur ou le protocole ont bougé — auquel cas il faut remesurer et
    consigner l'ancienne et la nouvelle valeur, pas mettre le test à jour.
    """
    from h0_qaoa_displacement import (reading_message, READING_FLAT,
                                      READING_MOVES)

    verdicts = {reading_message(v) for v in MEASURED_REPEATS}
    assert verdicts == {READING_FLAT, READING_MOVES}, (
        f"D-50 : les trois répétitions mesurées {MEASURED_REPEATS} rendent "
        f"maintenant {len(verdicts)} verdict(s) au lieu de 2. Remesurer avant "
        f"de toucher à ce test.")


def test_dispersion_exceeds_the_margin_to_the_threshold():
    """La marge au seuil est plus petite que la dispersion mesurée.

    marge = |0,0854 − 0,1| = 0,0146 ; dispersion = étendue des 3 répétitions.
    Tant que dispersion > marge, aucun seuil à cette place ne tranche.
    """
    from h0_qaoa_displacement import READING_THRESHOLD

    margin = abs(PUBLISHED_MEAN - READING_THRESHOLD)
    dispersion = max(MEASURED_REPEATS) - min(MEASURED_REPEATS)
    assert dispersion > margin, (
        f"dispersion {dispersion:.4f} vs marge {margin:.4f} — si la "
        f"dispersion est passée sous la marge, la grandeur est devenue "
        f"reproductible à cette précision : le dire et remesurer")
    assert dispersion == pytest.approx(0.0184, abs=5e-4)
    assert margin == pytest.approx(0.0146, abs=5e-4)


@pytest.mark.parametrize("value,expect_flat", [
    (0.0, True), (0.0999, True), (-0.0999, True),
    (0.1, False), (0.15, False), (-0.15, False), (0.1859, False),
])
def test_threshold_is_where_the_docstring_says(value, expect_flat):
    """Le seuil est bien à 0,1, strict, sur la valeur absolue.

    Sépare : 0.1859 est la moyenne du bras `cold` de D-48 — l'initialisation
    par défaut du dépôt bascule le verdict de façon systématique, là où le
    tirage le bascule une fois sur trois.
    """
    from h0_qaoa_displacement import (reading_message, READING_FLAT,
                                      READING_MOVES)

    expected = READING_FLAT if expect_flat else READING_MOVES
    assert reading_message(value) == expected


def test_acceptance_gate_does_not_look_at_the_verdict():
    """`check_expected_behaviour` ne garde pas la distance au seuil.

    Il vérifie deux choses réelles — fraction d'indéfinis et nombre de paires
    — et laisse passer les deux conclusions opposées. On l'interroge par son
    comportement : il accepte une progression des deux côtés du seuil.
    """
    from h0_qaoa_displacement import check_expected_behaviour

    rows = [{"progress": 0.0}] * 8
    paired = list(range(8))
    for prog in (0.0850, 0.1034):        # de part et d'autre de 0,1
        check_expected_behaviour(rows, 0.0, prog, paired, -0.116)
