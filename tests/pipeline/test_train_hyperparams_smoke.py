"""Repetition generale de la campagne, en miniature et sur le vrai code.

Les contrats sont testes ailleurs, avec un `pipeline` simule. Ici rien
n'est simule : le vrai solveur, le vrai circuit, la vraie base Optuna,
le vrai JSON de sortie — seulement a N=32, deux pas de temps et une
profondeur de raffinement. Une campagne d'une semaine sur des coeurs
loues ne doit pas etre le premier endroit ou l'on decouvre qu'un
scenario ne s'initialise pas.

Ce que ces tests NE montrent pas : que l'objectif discrimine. A cette
resolution il n'y a qu'une decision de raffinement, donc les six
sous-pertes sont egales. Ils montrent que le chemin complet s'execute
et que les artefacts en sortent complets.
"""
import json
import os
import sys
import warnings

import numpy as np
import optuna
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

import train_hyperparams as TH
from Simulation.pre_compute_dns import precompute_dns


def _tiny(config, key):
    """Le meme scenario, assez petit pour tourner en une seconde."""
    return {**config, "N": 32, "T_MAX": 0.06, "T_START": 0.02, "DT": 5e-3,
            "HYBRID_DT": 0.02, "K_opt": 3, "shots": 32,
            "max_depth_override": 1, "study_name": f"dns_{key}"}


@pytest.fixture(scope="module")
def tiny_campaign():
    """Les 8 scenarios, DNS pre-calculee, en miniature."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scenarios = tuple((k, _tiny(c, k)) for k, c in TH.SCENARIOS_ALL)
        traces = {k: precompute_dns(c) for k, c in scenarios}
    return scenarios, traces


def test_every_scenario_runs_through_the_real_pipeline(tiny_campaign):
    """Les huit scenarios s'initialisent, se simulent et rendent une perte
    finie. C'est la garantie minimale avant de louer des coeurs."""
    scenarios, traces = tiny_campaign
    objective = TH.make_composite_objective(traces, scenarios)
    study = optuna.create_study()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=1)

    trial = study.best_trial
    for key, _ in scenarios:
        loss = trial.user_attrs[f"loss_{key}"]
        assert np.isfinite(loss), key
        assert loss < 10.0, f"{key} a pris la penalite d'exception"
        assert f"patch_{key}" in trial.user_attrs, key
    assert np.isfinite(study.best_value)


def test_the_resolved_hyperparameters_reach_the_trial(tiny_campaign):
    """Le dictionnaire complet doit etre attache a l'essai — c'est la
    seule trace qui permette de redeployer sans le reconstruire."""
    scenarios, traces = tiny_campaign
    study = optuna.create_study()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(TH.make_composite_objective(traces, scenarios[:1]),
                       n_trials=1)
    resolved = study.best_trial.user_attrs["hyperparams_resolved"]
    assert set(resolved) == set(TH.SEARCH_SPACE) | set(TH.FIXED_PARAMS)
    assert all(np.isfinite(v) for v in resolved.values())


def test_the_classical_arm_runs_the_same_scenarios(tiny_campaign):
    scenarios, traces = tiny_campaign
    study = optuna.create_study()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(TH.make_classical_composite_objective(traces, scenarios),
                       n_trials=1)
    assert study.best_params.keys() == {"threshold_amr"}
    lo, hi = TH.CLASSICAL_THRESHOLD_RANGE
    assert lo <= study.best_params["threshold_amr"] <= hi
    for key, _ in scenarios:
        assert np.isfinite(study.best_trial.user_attrs[f"loss_{key}"]), key


def test_a_full_phase_writes_a_database_and_a_deployable_json(
        tiny_campaign, tmp_path, monkeypatch):
    """De bout en bout : run_phase -> base sqlite -> JSON deployable.

    C'est exactement l'enchainement que les coeurs loues executeront,
    a la taille de grille pres.
    """
    scenarios, traces = tiny_campaign
    monkeypatch.setattr(TH, "WORKER_TRIALS", None)
    monkeypatch.setattr(TH, "data_dir", str(tmp_path))
    monkeypatch.setattr(TH, "JOURNAL_DIR", str(tmp_path))
    monkeypatch.setattr(TH, "_DIRS_READY", True)

    config = {"n_trials": 3, "study_name": "smoke_phase"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = TH.run_phase("smoke", config,
                             TH.make_composite_objective(traces, scenarios),
                             seed_params=TH.phase1_seeds(), seed=7)

    assert TH.trials_done(study) == 3
    assert (tmp_path / "smoke_phase.log").exists()

    # D-199 : select_by_holdout_validation lit SCENARIOS_ALL (l'echelle de
    # production), pas les scenarios minuscules de `tiny_campaign` -- sans
    # ce drapeau ce banc ferait tourner `pipeline()` a l'echelle reelle.
    path = TH._save_results(study, study, study, filename="deploy.json",
                            run_holdout_validation=False)
    saved = json.load(open(path))
    deployed = saved["deploy"]["quantum"]
    assert set(deployed) == set(TH.SEARCH_SPACE) | set(TH.FIXED_PARAMS)
    assert saved["quantum"]["phase3"]["best_params_source"] == "trial_user_attr"
    # chaque scenario a bien une sous-perte dans le fichier
    assert set(saved["quantum"]["phase3"]["per_scenario"]) == \
        {k for k, _ in TH.SCENARIOS_ALL}


def test_the_first_seed_is_the_one_the_frozen_campaign_used(
        tiny_campaign, tmp_path, monkeypatch):
    """Une graine enfilee doit REELLEMENT etre le premier essai. Optuna
    ignore en silence une cle qui ne correspond a aucune distribution :
    une graine ignoree ressemble a une graine appliquee."""
    scenarios, traces = tiny_campaign
    monkeypatch.setattr(TH, "WORKER_TRIALS", None)
    monkeypatch.setattr(TH, "data_dir", str(tmp_path))
    monkeypatch.setattr(TH, "JOURNAL_DIR", str(tmp_path))
    monkeypatch.setattr(TH, "_DIRS_READY", True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = TH.run_phase("seeded", {"n_trials": 1, "study_name": "seeded"},
                             TH.make_composite_objective(traces, scenarios[:1]),
                             seed_params=TH.phase1_seeds(), seed=7)

    expected = TH.phase1_seeds()[0]
    assert study.trials[0].params == pytest.approx(expected)


def test_the_deployed_hyperparameters_are_accepted_by_the_pipeline(tiny_campaign):
    """Question 3, de bout en bout : ce que la campagne ecrira est-il ce
    que le deploiement sait lire ? `sigma` s'etait perdu exactement la."""
    from pipeline import pipeline
    scenarios, traces = tiny_campaign
    key, config = scenarios[0]
    hyperparams = {**{n: (lo + hi) / 2 for n, (lo, hi, _) in TH.SEARCH_SPACE.items()},
                   **TH.FIXED_PARAMS}
    dns_trace, hot_start = traces[key]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pipeline(
            N=config["N"], VQA_N=TH.VQA_N_TRAINING, T_MAX=config["T_MAX"],
            DT=config["DT"], HYBRID=int(config["HYBRID_DT"] / config["DT"]),
            verbose=False, argus=TH.create_argus(config),
            hyperparams=hyperparams, lambda_cost=TH.LAMBDA_COST_SOFT,
            trial=None, dns_trace=dns_trace, hot_start_state=hot_start,
            max_depth_override=config["max_depth_override"],
            scenario=config["scenario"], return_details=True)

    assert np.isfinite(result["combined"])
    # aucun repli silencieux : sigma vient bien des hyperparametres fournis
    assert result["sigma_source"] == "loaded"
    assert result["sigma"] == pytest.approx(hyperparams["sigma"])
    assert not [w for w in caught if "sigma absent" in str(w.message)]


def test_the_pipeline_shouts_when_sigma_is_missing(tiny_campaign):
    """Le controle du test precedent : sans `sigma`, le meme appel doit
    avertir ET marquer l'artefact. Un test qui ne peut pas echouer est un
    defaut — celui-ci verifie que le premier pouvait."""
    from pipeline import pipeline
    scenarios, traces = tiny_campaign
    key, config = scenarios[0]
    hyperparams = {n: (lo + hi) / 2 for n, (lo, hi, _) in TH.SEARCH_SPACE.items()
                   if n != "sigma"}
    dns_trace, hot_start = traces[key]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pipeline(
            N=config["N"], VQA_N=TH.VQA_N_TRAINING, T_MAX=config["T_MAX"],
            DT=config["DT"], HYBRID=int(config["HYBRID_DT"] / config["DT"]),
            verbose=False, argus=TH.create_argus(config),
            hyperparams=hyperparams, lambda_cost=TH.LAMBDA_COST_SOFT,
            trial=None, dns_trace=dns_trace, hot_start_state=hot_start,
            max_depth_override=config["max_depth_override"],
            scenario=config["scenario"], return_details=True)

    assert result["sigma_source"] == "default"
    assert [w for w in caught if "sigma absent" in str(w.message)]


def test_the_defaulted_sigma_value_is_exactly_0_05(tiny_campaign):
    """D-121 — la valeur du repli n'etait verifiee nulle part par un test
    COMPORTEMENTAL : `test_the_pipeline_falls_back_to_a_hard_coded_sigma`
    (`test_hyperparams_provenance_break.py`) ne lit que le SOURCE
    (`assert "_defaults.get('sigma', 0.05)" in src`), et le test au-dessus
    ne verifie que `sigma_source == "default"`, jamais la valeur.

    Mutation verifiee : `pipeline.py:394`, `0.05` change en `0.07` (le
    reste du fichier intact). Suite rejouee sur les deux fichiers de repli
    de sigma : SEUL le test qui lit le source rougit (`1 failed, 21 passed,
    1 xfailed`) — ce test-ci, absent avant D-121, n'existait pas pour le
    voir passer au travers. Une reecriture EQUIVALENTE de la ligne 394 (la
    meme valeur 0.05 rendue par une variable nommee plutot que le litteral)
    aurait, elle, fait rougir a tort le test source sans qu'aucun defaut
    n'existe — troisieme defaut de coherence de cette forme dans ce depot.
    Ce test verifie desormais la valeur REELLEMENT utilisee, pas le texte
    qui pretend la produire.
    """
    from pipeline import pipeline
    scenarios, traces = tiny_campaign
    key, config = scenarios[0]
    hyperparams = {n: (lo + hi) / 2 for n, (lo, hi, _) in TH.SEARCH_SPACE.items()
                   if n != "sigma"}
    dns_trace, hot_start = traces[key]

    result = pipeline(
        N=config["N"], VQA_N=TH.VQA_N_TRAINING, T_MAX=config["T_MAX"],
        DT=config["DT"], HYBRID=int(config["HYBRID_DT"] / config["DT"]),
        verbose=False, argus=TH.create_argus(config),
        hyperparams=hyperparams, lambda_cost=TH.LAMBDA_COST_SOFT,
        trial=None, dns_trace=dns_trace, hot_start_state=hot_start,
        max_depth_override=config["max_depth_override"],
        scenario=config["scenario"], return_details=True)

    assert result["sigma_source"] == "default"
    assert result["sigma"] == pytest.approx(0.05)
