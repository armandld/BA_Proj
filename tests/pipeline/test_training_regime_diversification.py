"""Diversification de l'entrainement (USER, 26 aout, apres D-199/D-200).

`select_by_holdout_validation` (D-199, damier depuis D-200) ne protege
que la SELECTION finale de la phase 3 : les 600+600+400 essais Optuna de
l'entrainement lui-meme restaient tous a Re=Rm=800, graine physique 0
implicite. Demande USER, apres avoir vu le damier de validation :
« ok mais moi je veux quand meme une campagne plus diversifiee. »

`make_composite_objective`/`make_classical_composite_objective` tirent
desormais, par essai, un regime physique dans `TRAINING_REGIME_GRID` --
independamment du sampler TPE d'Optuna (`_training_regime_for_trial`,
fonction pure du numero d'essai). Ce banc teste :
- que le tirage est deterministe et varie reellement ;
- que `TRAINING_REGIME_GRID` et `HOLDOUT_GRID` ne partagent AUCUN point
  (sinon la validation tenue a l'ecart jugerait un regime deja vu) ;
- que l'ancien comportement (pas de damier) est bit-pour-bit inchange ;
- que le nouveau chemin route les BONS regime/traces vers `_composite_loop`,
  via un espion controle plutot que la vraie physique (deja couverte
  ailleurs) ;
- que le contrat de campagne (`_qhas_contract`) qu'ouvre `prepare_phase1`
  correspond EXACTEMENT a celui que `_run_phase1` ouvrira ensuite pour de
  vrai -- sans quoi `--prepare-only` amorcerait une etude que la vraie
  execution ne pourrait plus reprendre (`campaign contract mismatch`).
"""
import optuna
import pytest

import train_hyperparams as training


# ══════════════════════════════════════════════════════════════════
#  Le tirage de regime lui-meme
# ══════════════════════════════════════════════════════════════════

def test_training_regime_for_trial_is_deterministic():
    for trial_number in (0, 1, 7, 42, 999):
        first = training._training_regime_for_trial(trial_number)
        second = training._training_regime_for_trial(trial_number)
        assert first == second, (
            f"essai #{trial_number} : deux appels rendent des regimes "
            f"differents -- une reprise de campagne ne retrouverait plus "
            f"le meme regime pour le meme essai")


def test_training_regime_for_trial_actually_varies_across_trials():
    drawn = {training._training_regime_for_trial(n) for n in range(200)}
    assert len(drawn) > 1, (
        "200 essais tirent tous le meme regime : la diversification est "
        "degeneree, sans effet reel")
    assert drawn <= set(training.TRAINING_REGIME_GRID)


def test_training_regime_for_trial_only_draws_from_the_given_grid():
    tiny_grid = ((111, 1), (222, 2))
    drawn = {training._training_regime_for_trial(n, grid=tiny_grid)
            for n in range(50)}
    assert drawn <= set(tiny_grid)


# ══════════════════════════════════════════════════════════════════
#  Le damier d'entrainement lui-meme
# ══════════════════════════════════════════════════════════════════

def test_training_regime_grid_varies_both_re_and_physical_seed():
    res = {re for re, _ in training.TRAINING_REGIME_GRID}
    seeds = {seed for _, seed in training.TRAINING_REGIME_GRID}
    assert len(res) >= 2, "un seul Re : pas de diversite physique"
    assert len(seeds) >= 2, "une seule graine : pas de diversite de graine"


def test_training_and_holdout_grids_never_share_a_point():
    """Le point critique : si l'entrainement voit un regime que le
    damier de validation croit tenu a l'ecart, `select_by_holdout_validation`
    ne validerait plus rien -- elle jugerait un regime deja appris."""
    overlap = set(training.TRAINING_REGIME_GRID) & set(training.HOLDOUT_GRID)
    assert overlap == set(), (
        f"points partages entre entrainement et validation : {overlap} -- "
        f"la validation tenue a l'ecart serait circulaire sur ces points")


def test_training_regime_grid_never_hits_no_duplicate_points():
    grid = training.TRAINING_REGIME_GRID
    assert len(grid) == len(set(grid)), (
        "des points dupliques biaiseraient la frequence de tirage sans le dire")


# ══════════════════════════════════════════════════════════════════
#  `make_composite_objective` : le vieux chemin reste inchange
# ══════════════════════════════════════════════════════════════════

def _spy_composite_loop(monkeypatch, calls):
    def fake(trial, scenario_list, dns_traces, hyperparams, lambda_cost,
             classical_only=False):
        calls.append({"scenario_list": scenario_list, "dns_traces": dns_traces,
                      "classical_only": classical_only})
        return 0.42
    monkeypatch.setattr(training, "_composite_loop", fake)


def test_objective_without_a_regime_grid_never_tags_the_trial(monkeypatch):
    """Non-regression : sans `dns_traces_by_regime`, le comportement doit
    etre EXACTEMENT celui d'avant cette diversification."""
    calls = []
    _spy_composite_loop(monkeypatch, calls)
    traces = {"kh": (None, None)}
    scenarios = (("kh", {"scenario": "kh", "Re": 800, "Rm": 800}),)
    objective = training.make_composite_objective(traces, scenarios)

    study = optuna.create_study()
    study.optimize(objective, n_trials=1)

    assert "training_regime" not in study.trials[0].user_attrs
    assert len(calls) == 1
    assert calls[0]["scenario_list"] == scenarios, (
        "le scenario_list transmis a _composite_loop a change alors "
        "qu'aucun damier n'est fourni")
    assert calls[0]["dns_traces"] is traces
    assert objective._qhas_contract["training_regime_grid"] is None


# ══════════════════════════════════════════════════════════════════
#  `make_composite_objective` : le nouveau chemin diversifie
# ══════════════════════════════════════════════════════════════════

def test_objective_with_a_regime_grid_tags_the_trial_and_routes_the_right_regime(
        monkeypatch):
    calls = []
    _spy_composite_loop(monkeypatch, calls)
    scenarios = (("kh", {"scenario": "kh", "Re": 800, "Rm": 800, "N": 32}),)
    grid = ((111, 1), (222, 2), (333, 3))
    dns_by_regime = {point: {"kh": (f"trace_{point}", None)} for point in grid}
    objective = training.make_composite_objective(
        None, scenarios, dns_traces_by_regime=dns_by_regime,
        training_regime_grid=grid)

    study = optuna.create_study()
    study.optimize(objective, n_trials=5)

    for trial in study.trials:
        expected_re, expected_seed = training._training_regime_for_trial(
            trial.number, grid)
        assert trial.user_attrs["training_regime"] == \
            f"Re={expected_re}_seed={expected_seed}"

    # Chaque appel a _composite_loop doit porter le Re/Rm/graine du
    # regime tire pour CET essai, pas un melange ou le regime par defaut.
    for trial, call in zip(study.trials, calls):
        expected_re, expected_seed = training._training_regime_for_trial(
            trial.number, grid)
        (key, cfg), = call["scenario_list"]
        assert key == "kh"
        assert cfg["Re"] == expected_re
        assert cfg["Rm"] == expected_re
        assert cfg["phys_seed"] == expected_seed
        assert call["dns_traces"] == {"kh": (f"trace_{(expected_re, expected_seed)}", None)}


def test_classical_objective_diversifies_the_same_way(monkeypatch):
    calls = []
    _spy_composite_loop(monkeypatch, calls)
    scenarios = (("kh", {"scenario": "kh", "Re": 800, "Rm": 800, "N": 32}),)
    grid = ((111, 1), (222, 2))
    dns_by_regime = {point: {"kh": (None, None)} for point in grid}
    objective = training.make_classical_composite_objective(
        None, scenarios, dns_traces_by_regime=dns_by_regime,
        training_regime_grid=grid)

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

    assert len(calls) == 3
    assert all(c["classical_only"] for c in calls)
    for trial in study.trials:
        assert "training_regime" in trial.user_attrs
    assert objective._qhas_contract["training_regime_grid"] == \
        [list(point) for point in grid]


def test_contract_is_none_for_classical_when_not_diversifying():
    scenarios = (("kh", {"scenario": "kh", "Re": 800, "Rm": 800}),)
    objective = training.make_classical_composite_objective(
        {"kh": (None, None)}, scenarios)
    assert objective._qhas_contract["training_regime_grid"] is None


# ══════════════════════════════════════════════════════════════════
#  Bonne forme : erreurs claires avant le premier essai, pas au milieu
# ══════════════════════════════════════════════════════════════════

def test_assert_regime_traces_wellformed_catches_a_missing_regime():
    scenarios = (("kh", {"scenario": "kh"}),)
    with pytest.raises(KeyError, match="absent"):
        training._assert_regime_traces_wellformed(
            scenarios, {(800, 0): {"kh": (None, None)}},
            ((800, 0), (900, 1)))


def test_assert_regime_traces_wellformed_catches_a_missing_scenario():
    scenarios = (("kh", {"scenario": "kh"}), ("rotor", {"scenario": "rotor"}))
    with pytest.raises(KeyError, match="manquantes"):
        training._assert_regime_traces_wellformed(
            scenarios, {(800, 0): {"kh": (None, None)}}, ((800, 0),))


def test_make_composite_objective_refuses_an_incomplete_regime_grid():
    scenarios = (("kh", {"scenario": "kh"}), ("rotor", {"scenario": "rotor"}))
    with pytest.raises(KeyError):
        training.make_composite_objective(
            None, scenarios,
            dns_traces_by_regime={(800, 0): {"kh": (None, None)}},
            training_regime_grid=((800, 0),))


# ══════════════════════════════════════════════════════════════════
#  `_precompute_dns_by_regime` : reutilise `_precompute_dns_for`, ne
#  reimplemente rien
# ══════════════════════════════════════════════════════════════════

def test_precompute_dns_by_regime_calls_precompute_dns_for_once_per_point(
        monkeypatch):
    seen = []

    def fake_precompute(scenario_list, label="scenarios"):
        seen.append((label, [(k, cfg["Re"], cfg["phys_seed"])
                             for k, cfg in scenario_list]))
        return {k: (None, None) for k, _ in scenario_list}

    monkeypatch.setattr(training, "_precompute_dns_for", fake_precompute)
    scenarios = (("kh", {"scenario": "kh", "Re": 800, "Rm": 800}),)
    grid = ((400, 1), (1200, 2))

    result = training._precompute_dns_by_regime(scenarios, grid=grid, label="test")

    assert len(seen) == len(grid), (
        "_precompute_dns_by_regime doit appeler _precompute_dns_for UNE "
        "FOIS PAR POINT du damier, jamais plus (le cout par essai reste "
        "inchange, seul ce precalcul, fait une fois par phase, grossit)")
    got_points = {(re, seed) for _, points in seen for _, re, seed in points}
    assert got_points == set(grid)
    assert set(result) == set(grid)
    for point in grid:
        assert set(result[point]) == {"kh"}


# ══════════════════════════════════════════════════════════════════
#  `prepare_phase1` / `_run_phase1` : le meme contrat, ou une reprise
#  casse au premier essai reel (`campaign contract mismatch`)
# ══════════════════════════════════════════════════════════════════

def test_prepare_phase1_contract_matches_a_real_diversified_phase1_contract():
    """C'est le defaut trouve en cablant cette diversification :
    `prepare_phase1` construisait un objectif SANS damier, `_run_phase1`
    en construit un AVEC -- deux contrats differents pour la MEME etude,
    donc `_open_phase_study` aurait leve `campaign contract mismatch` au
    premier `--phase 1` reel apres un `--prepare-only`."""
    placeholders = {key: (None, None) for key, _ in training.SCENARIOS_ISOLATED}
    placeholders_by_regime = {point: dict(placeholders)
                              for point in training.TRAINING_REGIME_GRID}

    prepare_objective = training.make_composite_objective(
        None, training.SCENARIOS_ISOLATED,
        dns_traces_by_regime=placeholders_by_regime)

    real_dns_by_regime = {point: dict(placeholders)
                          for point in training.TRAINING_REGIME_GRID}
    real_objective = training.make_composite_objective(
        None, training.SCENARIOS_ISOLATED,
        dns_traces_by_regime=real_dns_by_regime)

    assert prepare_objective._qhas_contract == real_objective._qhas_contract, (
        "le contrat de `--prepare-only` ne correspond pas a celui de la "
        "vraie phase 1 : une reprise casserait avec `campaign contract "
        "mismatch`")
