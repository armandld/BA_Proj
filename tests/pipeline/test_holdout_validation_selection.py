"""Refonte train/val de la campagne (D-199, USER 26 aout).

Les 3 phases de `train_hyperparams.py` choisissent `best_params` par
score EN ECHANTILLON : Optuna evalue sur les 8 scenarios eux-memes, a
Re=800 et une seule graine physique implicite -- rien n'est jamais tenu
a l'ecart. `select_by_holdout_validation` ajoute une selection finale
qui reclasse le top-k des essais de la phase 3 par leur perte sur un
regime physique (Re, graine) jamais vu par aucune phase.

Le risque a garder invisible : que la fonction se contente de rendre
`study.best_params` sous un autre nom. Le seul test qui le detecte
construit un cas ou le gagnant EN ECHANTILLON et le gagnant EN
VALIDATION sont des essais DIFFERENTS, et verifie que c'est bien celui
de la validation qui sort -- sans quoi toute la refonte serait un
habillage sans effet. `_composite_loop` (le calcul physique reel) est
remplace par un espion controle : ce banc teste la LOGIQUE DE SELECTION,
pas la physique, deja couverte ailleurs.
"""
import optuna
import pytest

import train_hyperparams as training


def _study_with_trials(specs):
    """`specs`: liste de (value_en_echantillon, marker). Renvoie une
    etude Optuna avec un essai COMPLETE par entree, portant
    `hyperparams_resolved={"marker": marker}`."""
    study = optuna.create_study(direction="minimize")
    for i, (value, marker) in enumerate(specs):
        trial = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=value,
            params={"beta": 1.0},
            distributions={"beta": optuna.distributions.FloatDistribution(0.0, 10.0)},
            user_attrs={"hyperparams_resolved": {"marker": marker}},
        )
        study.add_trial(trial)
    return study


@pytest.fixture
def no_real_dns(monkeypatch):
    """`_precompute_dns_for` ne doit jamais tourner pour de vrai dans ces
    tests : ils verifient la selection, pas le solveur."""
    monkeypatch.setattr(training, "_precompute_dns_for",
                        lambda scenario_list, label="": {
                            k: (None, None) for k, _ in scenario_list})


def _mock_composite_loop(holdout_scores):
    """Fabrique un remplacant de `_composite_loop` qui lit sa perte dans
    `holdout_scores[hyperparams["marker"]]` au lieu de simuler quoi que
    ce soit."""
    def fake(trial, scenario_list, dns_traces, hyperparams, lambda_cost,
             classical_only=False):
        return holdout_scores[hyperparams["marker"]]
    return fake


def test_prefers_the_holdout_winner_over_the_in_sample_winner(
        monkeypatch, no_real_dns):
    """Le coeur de la refonte : A gagne en echantillon, B gagne en
    validation -- c'est B qui doit sortir."""
    study = _study_with_trials([
        (0.1, "A"),   # meilleur EN ECHANTILLON
        (0.2, "B"),
        (0.3, "C"),   # pire en echantillon
    ])
    monkeypatch.setattr(training, "_composite_loop",
                        _mock_composite_loop({"A": 0.9, "B": 0.05, "C": 0.5}))

    result = training.select_by_holdout_validation(
        study, [("fake_scenario", {"scenario": "fake"})],
        top_k=3, classical_only=False, label="test")

    assert result["winner"]["params"]["marker"] == "B", (
        "la selection a rendu le meilleur en echantillon, pas le "
        "meilleur en validation : la refonte n'a aucun effet")
    assert result["train_winner_differs"] is True
    trials_by_marker = {t.user_attrs["hyperparams_resolved"]["marker"]: t.number
                        for t in study.trials}
    assert result["train_winner_trial"] == trials_by_marker["A"]


def test_agrees_with_in_sample_when_it_is_also_the_holdout_winner(
        monkeypatch, no_real_dns):
    """Non-regression : quand le meilleur en echantillon est AUSSI le
    meilleur en validation, la selection doit converger dessus, pas
    tomber sur un autre candidat par accident."""
    study = _study_with_trials([(0.1, "A"), (0.2, "B"), (0.3, "C")])
    monkeypatch.setattr(training, "_composite_loop",
                        _mock_composite_loop({"A": 0.05, "B": 0.5, "C": 0.9}))

    result = training.select_by_holdout_validation(
        study, [("fake_scenario", {"scenario": "fake"})],
        top_k=3, classical_only=False, label="test")

    assert result["winner"]["params"]["marker"] == "A"
    assert result["train_winner_differs"] is False


def test_top_k_bounds_which_candidates_are_reconsidered(monkeypatch, no_real_dns):
    """Le vrai gagnant en validation ("D") est hors du top_k=2 en
    echantillon : il ne doit PAS etre repeche. Sans cette borne, la
    fonction reevaluerait toute l'etude a chaque appel."""
    study = _study_with_trials([
        (0.1, "A"), (0.2, "B"), (0.3, "C"), (0.4, "D"),
    ])
    monkeypatch.setattr(
        training, "_composite_loop",
        _mock_composite_loop({"A": 0.9, "B": 0.05, "C": 0.5, "D": 0.001}))

    result = training.select_by_holdout_validation(
        study, [("fake_scenario", {"scenario": "fake"})],
        top_k=2, classical_only=False, label="test")

    assert result["winner"]["params"]["marker"] == "B", (
        "D (hors top_k) ne doit pas pouvoir gagner")
    assert {c["params"]["marker"] for c in result["candidates"]} == {"A", "B"}


def test_no_completed_trials_returns_a_named_empty_result(monkeypatch, no_real_dns):
    study = optuna.create_study(direction="minimize")
    result = training.select_by_holdout_validation(
        study, [("fake_scenario", {"scenario": "fake"})],
        top_k=5, classical_only=False, label="test")
    assert result["winner"] is None
    assert result["candidates"] == []


def test_holdout_scenario_config_overrides_only_re_rm_seed():
    base = {"scenario": "kh", "Re": 800, "Rm": 800, "N": 256,
           "T_MAX": 2.5, "K_opt": 30}
    cfg = training._holdout_scenario_config(base)
    assert cfg["Re"] == training.HOLDOUT_RE
    assert cfg["Rm"] == training.HOLDOUT_RE
    assert cfg["phys_seed"] == training.HOLDOUT_PHYS_SEED
    assert cfg["Re"] != base["Re"], (
        "le holdout doit differer du regime d'entrainement, sinon ce "
        "n'est pas une validation tenue a l'ecart")
    # tout le reste du scenario (scenario/N/T_MAX/K_opt...) est preserve
    for key in ("scenario", "N", "T_MAX", "K_opt"):
        assert cfg[key] == base[key]
