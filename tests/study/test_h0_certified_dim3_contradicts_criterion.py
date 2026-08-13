"""D-53 : la seule taille a la fois CERTIFIEE et NON DEGENEREE jamais
executee contredit le critere d'acceptation de H0, et son resultat n'est
ecrit dans aucun document.

`RESULTS.md` T11 publie `dim = 2` (8 qubits) et conclut « Pre-registered
rule fires: quantum optimisation is not the source of any gain » ; `dim = 2`
est aussi la taille ou l'etat fondamental exact vaut « raffiner partout »
(D-45 / D-47), donc ou l'accord entre solveurs est acquis d'avance.

Trois artefacts `dim = 3` (18 qubits, optimum enumere exactement) existent
dans `results/` et disent l'inverse. Ces tests EPINGLENT leurs nombres. Ils
sont deliberement des tests de DEVIATION, pas de regression : ils ne
pouvaient pas echouer sur un commit anterieur, ils echouent le jour ou l'un
des trois artefacts est remplace ou ou le critere est requalifie -- ce qui
est precisement le moment ou D-53 doit etre relu.

Aucun nombre publie n'en depend : `aggregate_master_table.collect` lit
`h0_optimiser_equivalence_N{N}_dim{dim}` a `N=256, dim=2`.
"""
import io
import contextlib
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
    MIN_HIT, MIN_MASK_MATCH, check_expected_behaviour,
)

RESULTS = os.path.join(_REPO_ROOT, "results")
DIM3 = os.path.join(RESULTS, "h0_optimiser_equivalence_N96_dim3.npz")
DIM3_KOPT = os.path.join(RESULTS, "h0_optimiser_equivalence_N96_dim3_scalekopt.npz")
DIM2 = os.path.join(RESULTS, "h0_optimiser_equivalence_N256_dim2.npz")


def _summary(path):
    """(summary, solvers, diag_flags) reconstruits depuis un artefact, dans
    la forme exacte que `main()` passe au critere."""
    if not os.path.exists(path):
        pytest.skip(f"artefact absent : {os.path.basename(path)}")
    d = np.load(path, allow_pickle=True)
    sol = d["solver"]
    solvers = list(dict.fromkeys(sol.tolist()))
    summary = {
        s: dict(hit=float(np.mean(d["hit"][sol == s])),
                match=float(np.mean(d["match"][sol == s])),
                agree=float(np.mean(d["agree"][sol == s])),
                gap=float(np.mean(d["E_gap"][sol == s])),
                f1=float(np.mean(d["f1"][sol == s])),
                wall=0.0)
        for s in solvers}
    return summary, solvers, [bool(d["diagonal_all"])]


def test_dim2_artifact_satisfies_the_criterion():
    """La reference du critere, verifiee et non supposee : a la taille
    deployee il passe bien."""
    summary, solvers, diag = _summary(DIM2)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        check_expected_behaviour(summary, solvers, diag)
    assert "[ACCEPTANCE]" in buf.getvalue()


def test_dim3_artifact_makes_the_criterion_raise():
    """Le coeur de D-53 : sur la seule taille certifiee non degeneree, le
    critere du module leve — et son propre message dit « H0 redevient
    plausible »."""
    summary, solvers, diag = _summary(DIM3)
    with pytest.raises(AssertionError) as exc:
        check_expected_behaviour(summary, solvers, diag)
    assert "redevient plausible" in str(exc.value)
    assert "qaoa_p1" in str(exc.value)


def test_dim3_qaoa_lands_further_from_the_optimum_than_the_classical_rule():
    """Mesure epinglee : le QAOA n'est pas seulement sous le critere, il est
    sous la regle classique dont il part."""
    summary, _, _ = _summary(DIM3)
    classical = summary["classical_init"]["match"]
    assert classical == pytest.approx(0.500, abs=5e-3)
    for depth in (1, 2, 3, 4, 5, 6):
        m = summary[f"qaoa_p{depth}"]["match"]
        assert m < classical, (
            f"qaoa_p{depth} match={m:.3f} n'est plus sous la regle classique "
            f"({classical:.3f}) : D-53 est a relire")
    assert summary["qaoa_p1"]["match"] == pytest.approx(0.15625, abs=5e-3)
    assert summary["qaoa_p6"]["match"] == pytest.approx(0.21875, abs=5e-3)


def test_dim3_qaoa_hit_rates_are_pinned():
    """Les taux eux-memes, pour qu'une derive se voie. 0,062–0,156 contre le
    1,000 exige."""
    summary, _, _ = _summary(DIM3)
    expected = {"qaoa_p1": 0.15625, "qaoa_p2": 0.15625, "qaoa_p3": 0.125,
                "qaoa_p4": 0.09375, "qaoa_p5": 0.0625, "qaoa_p6": 0.0625}
    for name, value in expected.items():
        assert summary[name]["hit"] == pytest.approx(value, abs=5e-3)
        assert summary[name]["hit"] < MIN_HIT
        assert summary[name]["match"] < MIN_MASK_MATCH


def test_scaled_budget_does_not_rescue_the_qaoa_arm():
    """L'objection « c'est un budget COBYLA trop court » est celle que
    `--scale-kopt` existe pour lever : avec le budget proportionnel a p, le
    bras QAOA tombe a 0,000 sur les quatre profondeurs, greedy restant a
    1,000."""
    summary, solvers, diag = _summary(DIM3_KOPT)
    for s in [s for s in solvers if s.startswith("qaoa")]:
        assert summary[s]["hit"] == pytest.approx(0.0, abs=1e-9)
    assert summary["greedy"]["hit"] == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(AssertionError, match="redevient plausible"):
        check_expected_behaviour(summary, solvers, diag)


def test_the_decision_not_to_correct_stays_written():
    """`VIGIL.md` : une deviation connue mais non consignee se fait
    recorriger par erreur ; toute decision de ne pas corriger s'ecrit, et un
    test verifie que la mention y reste.

    Volontairement PAS l'inverse (« dim3 est absent de RESULTS.md ») : ce
    test-la echouerait le jour ou quelqu'un fait la bonne chose et publie le
    balayage, c'est-a-dire sur un changement voulu — le piege que `VIGIL.md`
    documente trois fois.
    """
    with open(os.path.join(_REPO_ROOT, "docs", "DEFAUTS.md"), encoding="utf-8") as fh:
        defauts = fh.read()
    assert "D-53" in defauts, (
        "la decision de ne pas corriger D-53 a disparu de DEFAUTS.md : sans "
        "elle, le critere MIN_HIT=1.0 se relit comme valide a toute taille")
