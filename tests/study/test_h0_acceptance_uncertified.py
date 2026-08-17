"""D-52 : le critere d'acceptation de H0 annoncait « H0 refutee » sur une
campagne ou aucun optimum n'avait ete certifie.

Sans reference exacte — `--no-exact`, ou `n_q > MAX_ENUM_QUBITS`, ce qui est
le cas de TOUT `dim >= 4` (32 qubits contre un plafond de 22) —
`solver_panel` ecrit NaN dans `hit_optimum` et `exact_match`.
`check_expected_behaviour` les comparait par `<` a MIN_HIT / MIN_MASK_MATCH,
et `nan < 1.0` vaut False : les deux dictionnaires de violation restaient
vides quoi qu'il arrive. Le critere ne pouvait pas echouer — il ne mesurait
rien — et imprimait la ligne [ACCEPTANCE] trois lignes sous une DECISION RULE
qui affirmait l'inverse (« QAOA deviates from the certified optimum »).

Mesure : `--scenario orszag_tang --re 400 --N 64 --dim 2 --n-snaps 1
--no-exact`, code de sortie 0, huit solveurs a hit=nan / mask_match=nan.
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

from h0_optimiser_equivalence import (            # noqa: E402
    MAX_ENUM_QUBITS, MIN_HIT, MIN_MASK_MATCH,
    check_expected_behaviour, decision_rule_lines, is_certified,
)

NAN = float("nan")

#: Le panel tel qu'il existe quand `certified` est False : « exhaustive »
#: n'est pas enregistre, tout le reste l'est.
UNCERTIFIED_SOLVERS = ["sa", "sa_warm", "greedy", "classical_init",
                       "qaoa_p1", "qaoa_p2", "qaoa_p3", "qaoa_shots_p3"]
CERTIFIED_SOLVERS = ["exhaustive"] + UNCERTIFIED_SOLVERS


def _summary(solvers, hit, match, f1=0.667):
    return {s: dict(hit=hit, match=match, agree=hit, gap=0.0, f1=f1, wall=1.0)
            for s in solvers}


# ---------------------------------------------------------------- l'ancien
def test_old_predicate_could_not_reject_a_nan_panel():
    """Epingle le MECANISME de l'ancien comportement, pour que la correction
    ne puisse pas etre defaite en silence.

    C'est litteralement l'expression que portaient `missed` et `diverging` :
    sur NaN elle repond « aucune violation », donc les deux assertions
    passaient sur une campagne ou rien n'avait ete mesure.
    """
    summary = _summary(UNCERTIFIED_SOLVERS, NAN, NAN)
    optimisers = [s for s in UNCERTIFIED_SOLVERS if s != "classical_init"]

    missed = {s: summary[s]["hit"] for s in optimisers
              if not s.startswith("sa") and summary[s]["hit"] < MIN_HIT}
    diverging = {s: summary[s]["match"] for s in optimisers
                 if s.startswith("qaoa") and summary[s]["match"] < MIN_MASK_MATCH}

    assert missed == {}, "l'ancien predicat rejetait deja ce panel"
    assert diverging == {}, "l'ancien predicat rejetait deja ce panel"
    # ... et pourtant aucune de ces valeurs n'est mesuree :
    assert all(np.isnan(summary[s]["hit"]) for s in optimisers)


# ------------------------------------------------------- le cas non certifie
def test_uncertified_panel_is_not_certified():
    assert not is_certified(_summary(UNCERTIFIED_SOLVERS, NAN, NAN),
                            UNCERTIFIED_SOLVERS)


def test_uncertified_panel_prints_indecidable_not_acceptance(capsys):
    """Le coeur de D-52 : sur hit/match indefinis, le critere doit dire qu'il
    est sans objet, jamais que H0 est refutee."""
    check_expected_behaviour(_summary(UNCERTIFIED_SOLVERS, NAN, NAN),
                             UNCERTIFIED_SOLVERS, [True, True])
    out = capsys.readouterr().out
    assert "[INDECIDABLE]" in out
    assert "ACCEPTANCE" not in out
    assert "refutee" not in out.replace("ni refutee ni confirmee", "")


def test_uncertified_panel_does_not_print_a_nan_hit_rate(capsys):
    """La ligne [NOTE] annoncait « optimum atteint sur nan des instantanes »
    pour le recuit : un taux qui n'existe pas, imprime comme une mesure."""
    check_expected_behaviour(_summary(UNCERTIFIED_SOLVERS, NAN, NAN),
                             UNCERTIFIED_SOLVERS, [True])
    assert "nan" not in capsys.readouterr().out.lower()


def test_uncertified_decision_rule_says_undecidable_not_deviating():
    """Les deux verdicts d'une meme execution se contredisaient : la
    DECISION RULE disait « QAOA deviates from the certified optimum » — en
    parlant d'un optimum qui n'existait pas — et l'[ACCEPTANCE] disait que
    tous les solveurs l'atteignaient."""
    lines = "\n".join(decision_rule_lines(
        _summary(UNCERTIFIED_SOLVERS, NAN, NAN), UNCERTIFIED_SOLVERS))
    assert "UNDECIDABLE" in lines
    assert "deviates" not in lines
    assert "attributable to the Hamiltonian" not in lines


def test_real_no_qaoa_no_exact_panel_still_raises():
    """Le panel de l'artefact `_N64_dim4_orszag_tang_noexact.npz` du depot
    (--no-qaoa + --no-exact) : 3 optimiseurs, aucun bras QAOA. Il echouait
    avant la correction, il doit continuer — le retour anticipe ne doit pas
    le rendre acceptable."""
    solvers = ["sa", "sa_warm", "greedy", "classical_init"]
    with pytest.raises(AssertionError, match="solveurs compares"):
        check_expected_behaviour(_summary(solvers, NAN, NAN), solvers, [True])


def test_missing_qaoa_arm_still_raises_when_uncertified():
    """Le controle « aucun bras QAOA » doit rester atteignable meme quand le
    panel est assez large pour passer le controle de taille : sans bras
    QAOA, H0 n'est pas testee, certifie ou non."""
    solvers = ["sa", "sa_warm", "greedy", "exact_alt", "classical_init"]
    with pytest.raises(AssertionError, match="aucun bras QAOA"):
        check_expected_behaviour(_summary(solvers, NAN, NAN), solvers, [True])


# ----------------------------------------------------------- le cas certifie
def test_certified_panel_still_prints_acceptance(capsys):
    """Le chemin certifie est inchange : verifie contre un run reel
    (`--N 64 --dim 2 --n-snaps 1`, orszag_tang, Re=400) dont les lignes de
    verdict sont bit-a-bit identiques avant et apres la correction."""
    check_expected_behaviour(_summary(CERTIFIED_SOLVERS, 1.0, 1.0),
                             CERTIFIED_SOLVERS, [True, True])
    out = capsys.readouterr().out
    assert "[ACCEPTANCE]" in out
    assert "H0 refutee" in out
    assert "[INDECIDABLE]" not in out


def test_certified_deterministic_miss_still_raises():
    summary = _summary(CERTIFIED_SOLVERS, 1.0, 1.0)
    summary["greedy"]["hit"] = 0.75
    with pytest.raises(AssertionError, match="n'atteignent plus l'optimum"):
        check_expected_behaviour(summary, CERTIFIED_SOLVERS, [True])


def test_certified_qaoa_divergence_still_raises():
    summary = _summary(CERTIFIED_SOLVERS, 1.0, 1.0)
    summary["qaoa_p2"]["match"] = 0.5
    with pytest.raises(AssertionError, match="ne renvoie plus le masque"):
        check_expected_behaviour(summary, CERTIFIED_SOLVERS, [True])


def test_certified_decision_rule_text_unchanged():
    lines = "\n".join(decision_rule_lines(
        _summary(CERTIFIED_SOLVERS, 1.0, 1.0), CERTIFIED_SOLVERS))
    assert ("quantum optimisation is NOT the source of any gain; value is "
            "attributable to the Hamiltonian.") in lines
    assert "UNDECIDABLE" not in lines


# ------------------------------------------- la configuration qui y mene
def test_dim4_exceeds_the_enumeration_cap_without_any_flag():
    """La branche non certifiee n'a pas besoin de `--no-exact` : `dim = 4`
    suffit. `VQA_DIMS` declare pourtant 2, 4 et 8."""
    assert 2 * 2 * 2 <= MAX_ENUM_QUBITS          # dim=2 : 8 qubits, certifie
    assert 2 * 3 * 3 <= MAX_ENUM_QUBITS          # dim=3 : 18 qubits, certifie
    assert 2 * 4 * 4 > MAX_ENUM_QUBITS           # dim=4 : 32 qubits, NON
    assert 2 * 8 * 8 > MAX_ENUM_QUBITS           # dim=8 : 128 qubits, NON
