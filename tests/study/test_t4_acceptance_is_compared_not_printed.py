"""D-85 : le critère d'acceptation de la tâche 4 était imprimé, jamais comparé.

Le protocole v3 (§8.3, tâche 4) pose l'acceptation comme *« one table ;
random-split numbers match Task 0 »*. `h2b_blocked_split.py` imprimait

    acceptance refs (Task 0): B2 random F1 = 0.475, B4 gbt-9 (max) random F1 = 0.980

et s'arrêtait là — aucune comparaison, aucun drapeau, rien dans l'artefact.
Et l'en-tête de `tests/study/test_t4_blocked_split.py` renvoyait explicitement
la vérification chiffrée à *« l'exécution sur les vraies données »*, où elle
n'avait pas lieu. Un critère en prose ne peut pas échouer : même forme que
D-52 (`h0_optimiser_equivalence`).

**Et il échoue.** Mesuré à HEAD, `--dim 4 --N 256 --seed 0` (Re=400 ; nombres
identiques à `--max-snaps 30` et `--max-snaps 80`, le GBT étant déterministe
à graine fixée) :

| ligne | mesuré | référence tâche 0 | écart |
|---|---|---|---|
| `B2 classical (block_max)` | 0,472 | 0,475 | +0,003 — dans la bande |
| `B4 gbt-9 (max)` | **0,908** | **0,980** | **−0,072 — hors bande** |

Les deux références sont des nombres d'**archive d'avant l'audit**, de la même
provenance que celles de `aggregate_v3.py` (D-49), que `docs/archive/README.md`
déclare obsolètes. Elles ne sont donc **pas** réajustées : un seuil périmé se
remesure, il ne se retouche pas. L'écart est signalé, écrit dans l'artefact,
et laissé à trancher.

Ces tests-ci n'utilisent aucune donnée MHD : ils portent sur `check_acceptance`,
extrait pour être testable sans rejouer la campagne — même geste que
`interpretation_message` (D-46), `reading_message` (D-50) et
`decision_rule_lines` (D-52).
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from h2b_blocked_split import (  # noqa: E402
    ACCEPTANCE_REFS, TOL_ACCEPT, check_acceptance,
)

# les valeurs réellement mesurées à HEAD (voir l'en-tête)
MESURE_HEAD = {"B2 classical (block_max)": 0.472,
               "B4 gbt-9 (max)": 0.908}


def test_the_measured_run_fails_the_criterion_on_one_of_two_rows():
    """Épingle l'état mesuré : B2 passe, B4 non. C'est ce que le script
    imprimait sans jamais le regarder."""
    rows = check_acceptance(MESURE_HEAD)
    par_nom = {r[0]: r for r in rows}

    _, ref_b2, got_b2, ok_b2 = par_nom["B2 classical (block_max)"]
    assert (ref_b2, got_b2, ok_b2) == (0.475, 0.472, True)

    _, ref_b4, got_b4, ok_b4 = par_nom["B4 gbt-9 (max)"]
    assert (ref_b4, got_b4) == (0.980, 0.908)
    assert ok_b4 is False, (
        "l'écart de 0,072 doit être hors bande : sinon la tolérance a été "
        "élargie jusqu'à faire passer la mesure, ce que D-85 refuse")
    assert abs(got_b4 - ref_b4) == pytest.approx(0.072, abs=1e-3)


def test_the_criterion_can_pass():
    """Un critère qui échoue toujours ne vaut pas mieux qu'un qui ne peut pas."""
    exact = {name: ref for name, ref in ACCEPTANCE_REFS}
    assert all(ok for *_, ok in check_acceptance(exact))
    dedans = {name: ref + 0.9 * TOL_ACCEPT for name, ref in ACCEPTANCE_REFS}
    assert all(ok for *_, ok in check_acceptance(dedans))


def test_the_criterion_bites_just_past_the_tolerance():
    juste_au_dela = {name: ref + TOL_ACCEPT * 1.001
                     for name, ref in ACCEPTANCE_REFS}
    assert not any(ok for *_, ok in check_acceptance(juste_au_dela))


def test_a_reference_naming_no_row_is_a_loud_failure():
    """Le piège du balayage vide : une référence qui ne désigne aucune ligne
    ne compare rien, et passerait pour un succès."""
    with pytest.raises(KeyError, match="absente de la table"):
        check_acceptance({"B2 classical (block_max)": 0.475})

    # et le nom de chaque référence désigne bien une ligne que le script produit
    from h2b_blocked_split import _evaluate_split  # noqa: F401  (existence)
    assert {name for name, _ in ACCEPTANCE_REFS} == {
        "B2 classical (block_max)", "B4 gbt-9 (max)"}
