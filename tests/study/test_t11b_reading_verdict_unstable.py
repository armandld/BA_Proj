"""D-50 — CORRIGE (decision USER, 25 aout). Le verdict T11b lit `slope`.

`h0_qaoa_displacement` imprimait l'une de deux phrases opposees selon que
`|progress moyen| < 0,1` — `progress moyen` etant la moyenne d'UN tirage
QAOA par instantane. Mesure historique (motivant la correction) : trois
executions de la commande publiee (`--N 256 --dim 2 --n-snaps 2`, reps 1-4)
rendaient 0,1034 / 0,0850 / 0,0859, deux conclusions opposees sur trois
executions identiques.

Correction retenue parmi les trois options de `docs/DEFAUTS.md` D-50 :
`reading_message` lit desormais `slope` (`slope_paired` dans l'artefact) —
la pente APPARIEE progress(reps=max) - progress(reps=min) sur les memes
instantanes, deja calculee (elle alimentait deja `check_expected_behaviour`)
et repondant a une question plus proche de la motivation du fichier :
le progres croit-il avec la profondeur du circuit.

**Ce qui n'a PAS ete refait** : une etude de reproductibilite empirique de
`slope` sur plusieurs executions independantes, comme celle qui avait
revele l'instabilite de `progress moyen`. Le changement est mecanique
(reutilise un calcul deja existant, aucun nouveau calcul) ; la stabilite
du NOUVEAU verdict a cette precision reste a mesurer.
"""
import ast
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_H0 = os.path.join(_REPO_ROOT, "study", "h0_selection")
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Mesure historique qui motivait la correction (D-50) : `progress moyen`,
# PAS `slope` — conservee comme fait, plus comme test de `reading_message`
# puisque cette fonction ne prend plus cette grandeur en argument.
HISTORICAL_PROGRESS_REPEATS = (0.1034, 0.0850, 0.0859)


def test_reading_message_prend_desormais_slope_pas_progress():
    """Le champ qui SEPARE la correction de l'ancien defaut : la fonction ne
    doit plus prendre `prog_all`/`progress` comme nom de parametre, faute de
    quoi rien n'empecherait de la rebrancher sur la meme grandeur instable.
    """
    path = os.path.join(_H0, "h0_qaoa_displacement.py")
    with open(path, encoding="utf-8") as f:
        arbre = ast.parse(f.read())
    fn = next((n for n in ast.walk(arbre)
               if isinstance(n, ast.FunctionDef) and n.name == "reading_message"),
              None)
    assert fn is not None, "reading_message a disparu de h0_qaoa_displacement"
    params = [a.arg for a in fn.args.args]
    assert params == ["slope"], (
        f"reading_message prend {params} : devrait etre ['slope'] apres la "
        "correction D-50")


def test_le_flux_principal_appelle_reading_message_avec_slope():
    """Verifie l'appel reel dans `main`, pas seulement la signature de la
    fonction (qui pourrait etre correcte sans etre branchee)."""
    path = os.path.join(_H0, "h0_qaoa_displacement.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "reading_message(slope)" in src, (
        f"{path} n'appelle plus reading_message(slope) : verifier que "
        "l'appel n'a pas ete rebranche sur prog_all")


@pytest.mark.parametrize("value,expect_flat", [
    (0.0, True), (0.0999, True), (-0.0999, True),
    (0.1, False), (0.15, False), (-0.15, False), (0.1859, False),
])
def test_threshold_is_where_the_docstring_says(value, expect_flat):
    """Le seuil est INCHANGE a 0,1, strict, sur la valeur absolue — seule la
    grandeur qu'il seuille a change (D-50)."""
    from h0_qaoa_displacement import (reading_message, READING_FLAT,
                                      READING_MOVES)

    expected = READING_FLAT if expect_flat else READING_MOVES
    assert reading_message(value) == expected


def test_le_texte_des_deux_verdicts_decrit_une_pente_pas_une_position():
    """Les deux phrases parlaient de position absolue (« stays at »,
    « moves toward ») ; elles doivent desormais parler de croissance avec
    la profondeur, ce que `slope` mesure reellement.
    """
    from h0_qaoa_displacement import READING_FLAT, READING_MOVES

    for text in (READING_FLAT, READING_MOVES):
        assert "depth" in text, (
            f"{text!r} ne mentionne pas la profondeur du circuit : le "
            "texte decrit peut-etre encore la position absolue de "
            "prog_all, pas la pente de slope")


def test_le_fait_historique_qui_motivait_la_correction_reste_ecrit():
    """`progress moyen` restait instable sur trois executions identiques —
    c'est le fait qui a motive le passage a `slope`. Il n'est plus testable
    via `reading_message` (qui ne prend plus cette grandeur), donc ce test
    ne verifie plus que les trois nombres sont bien ceux consignes.
    """
    assert HISTORICAL_PROGRESS_REPEATS == (0.1034, 0.0850, 0.0859)
    dispersion = max(HISTORICAL_PROGRESS_REPEATS) - min(HISTORICAL_PROGRESS_REPEATS)
    assert dispersion == pytest.approx(0.0184, abs=5e-4)


def test_acceptance_gate_does_not_look_at_the_verdict():
    """`check_expected_behaviour` ne garde pas la distance au seuil.

    Il verifie deux choses reelles — fraction d'indefinis et nombre de
    paires — et laisse passer les deux conclusions opposees. On
    l'interroge par son comportement : il accepte une progression des
    deux cotes du seuil.
    """
    from h0_qaoa_displacement import check_expected_behaviour

    rows = [{"progress": 0.0}] * 8
    paired = list(range(8))
    for slope in (0.0850, 0.1034):        # de part et d'autre de 0,1
        check_expected_behaviour(rows, 0.0, 0.0, paired, slope)
