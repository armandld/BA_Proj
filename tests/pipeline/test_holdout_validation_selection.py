"""Refonte train/val de la campagne (D-199, USER 26 aout) et son damier
(D-200, USER 26 aout).

Les 3 phases de `train_hyperparams.py` choisissent `best_params` par
score EN ECHANTILLON : Optuna evalue sur les 8 scenarios eux-memes, a
Re=800 et une seule graine physique implicite -- rien n'est jamais tenu
a l'ecart. `select_by_holdout_validation` ajoute une selection finale
qui reclasse le top-k des essais de la phase 3 par leur perte MOYENNE
sur un DAMIER de regimes physiques (plusieurs Re, plusieurs graines)
jamais vus par aucune phase.

Le risque a garder invisible : que la fonction se contente de rendre
`study.best_params` sous un autre nom. Le seul test qui le detecte
construit un cas ou le gagnant EN ECHANTILLON et le gagnant EN
VALIDATION sont des essais DIFFERENTS, et verifie que c'est bien celui
de la validation qui sort -- sans quoi toute la refonte serait un
habillage sans effet. `_composite_loop` (le calcul physique reel) est
remplace par un espion controle : ce banc teste la LOGIQUE DE SELECTION,
pas la physique, deja couverte ailleurs.

Deuxieme risque, propre au damier : qu'un candidat gagne en etant bon
sur UN SEUL point et mediocre partout ailleurs -- exactement le
surapprentissage que le damier existe pour empecher, deplace d'un cran
si le classement ne regardait qu'un point. Un test dedie construit ce
cas et verifie que c'est le candidat REGULIER qui gagne.
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
    ce soit. Le meme score sort a chaque point du damier : suffisant
    pour les tests qui ne portent que sur le classement des candidats."""
    def fake(trial, scenario_list, dns_traces, hyperparams, lambda_cost,
             classical_only=False):
        return holdout_scores[hyperparams["marker"]]
    return fake


def _mock_composite_loop_per_regime(scores_by_marker_and_point):
    """Variante de `_mock_composite_loop` dont la perte depend aussi du
    POINT du damier — lu dans `scenario_list`, que
    `select_by_holdout_validation` construit via `_with_physical_regime`
    (donc `scenario_list[0][1]["Re"]`/`["phys_seed"]` portent le point
    courant). Necessaire pour tester que le classement resiste a un
    candidat bon sur un seul point."""
    def fake(trial, scenario_list, dns_traces, hyperparams, lambda_cost,
             classical_only=False):
        point = (scenario_list[0][1]["Re"], scenario_list[0][1]["phys_seed"])
        return scores_by_marker_and_point[hyperparams["marker"]][point]
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


def test_with_physical_regime_overrides_only_re_rm_seed():
    base = {"scenario": "kh", "Re": 800, "Rm": 800, "N": 256,
           "T_MAX": 2.5, "K_opt": 30}
    cfg = training._with_physical_regime(base, re=400, phys_seed=2)
    assert cfg["Re"] == 400
    assert cfg["Rm"] == 400
    assert cfg["phys_seed"] == 2
    assert cfg["Re"] != base["Re"], (
        "le holdout doit differer du regime d'entrainement, sinon ce "
        "n'est pas une validation tenue a l'ecart")
    # tout le reste du scenario (scenario/N/T_MAX/K_opt...) est preserve
    for key in ("scenario", "N", "T_MAX", "K_opt"):
        assert cfg[key] == base[key]


def test_holdout_grid_varies_both_re_and_physical_seed():
    """Demande USER, 26 aout : « il faudrait que cette campagne ait
    plusieurs parametrisations physiques et plusieurs graines » -- un
    damier qui ne ferait varier qu'un seul axe ne repondrait qu'a moitie
    a la demande."""
    res = {re for re, _ in training.HOLDOUT_GRID}
    seeds = {seed for _, seed in training.HOLDOUT_GRID}
    assert len(res) >= 2, "un seul Re dans le damier : pas de diversite physique"
    assert len(seeds) >= 2, "une seule graine dans le damier : pas de diversite de graine"
    assert len(training.HOLDOUT_GRID) == len(set(training.HOLDOUT_GRID)), (
        "des points dupliques gonfleraient le cout sans ajouter d'information")
    training_regime = (800, 0)  # Re=Rm=800, graine physique implicite 0
    assert training_regime not in training.HOLDOUT_GRID, (
        "un point du damier egal au regime d'entrainement ne validerait rien")


def test_the_winner_generalises_across_the_whole_grid_not_one_lucky_point(
        monkeypatch, no_real_dns):
    """D-200 (USER, 26 aout) : un damier a plusieurs points peut quand
    meme etre surappris s'il n'est regarde que par son MEILLEUR point.
    "A" brille sur un seul point du damier (perte quasi nulle) et
    s'effondre sur tous les autres ; "B" est regulier partout, moins bon
    que "A" sur son point favori mais bien meilleur en moyenne. Si "A"
    gagne, le classement ne resume pas vraiment le damier."""
    study = _study_with_trials([(0.1, "A"), (0.2, "B")])
    grid = training.HOLDOUT_GRID
    assert len(grid) >= 2
    lucky_point, *rest = grid
    scores = {
        "A": {lucky_point: 0.0, **{point: 0.9 for point in rest}},
        "B": {point: 0.3 for point in grid},
    }
    monkeypatch.setattr(training, "_composite_loop",
                        _mock_composite_loop_per_regime(scores))

    result = training.select_by_holdout_validation(
        study, [("fake_scenario", {"scenario": "fake"})],
        top_k=2, classical_only=False, label="test")

    assert result["winner"]["params"]["marker"] == "B", (
        "'A' n'est bon que sur un point du damier (0.0) et s'effondre "
        "ailleurs (0.9) ; sa moyenne est pire que celle de 'B', regulier "
        "a 0.3 -- si 'A' gagne, la selection ne regarde qu'un point")
    winner_scored = next(c for c in result["candidates"]
                         if c["params"]["marker"] == "B")
    assert winner_scored["holdout_value"] == pytest.approx(0.3)
    loser_scored = next(c for c in result["candidates"]
                        if c["params"]["marker"] == "A")
    assert loser_scored["holdout_worst"] == pytest.approx(0.9)
    assert set(winner_scored["holdout_per_point"]) == {
        f"Re={re}_seed={seed}" for re, seed in grid}
