"""Cablage de l'entrainement sur le pool jouet (USER : pool synthetique bon
marche plutot que les 8 scenarios reels fixes, ceux-ci reserves a la
validation tenue a l'ecart).

Trois risques a garder invisibles, chacun avec son test dedie :
  1. Que le pool jouet et les 8 scenarios reels se melangent quelque part
     -- `run_toy_holdout` doit passer `SCENARIOS_ALL` (reel), jamais
     `TOY_SCENARIOS_ALL`, a `select_by_holdout_validation`.
  2. Qu'une etude Optuna jouet ecrive dans le meme journal qu'une etude
     reelle (meme `study_name`) -- `PHASES["toy_*"]` doit rester disjoint
     de `PHASES` reel.
  3. Que le routage scenario/regime pour le pool jouet diverge du chemin
     reel deja verifie (`test_training_regime_diversification.py`) --
     verifie ici par le meme espion controle sur `_composite_loop`,
     jamais la physique reelle.
"""
import warnings

import numpy as np
import optuna
import pytest

import train_hyperparams as training
from Simulation.pre_compute_dns import precompute_dns


# ══════════════════════════════════════════════════════════════════
#  Un seul test sur le VRAI pipeline (le reste espionne _composite_loop)
# ══════════════════════════════════════════════════════════════════
#
# Meme technique que `test_train_hyperparams_smoke.py` (`_tiny`) : le
# reste de ce fichier verifie le CABLAGE avec un `_composite_loop`
# simule, deliberement -- mais aucun de ces tests ne prouve que le
# solveur reel accepte une config `toy_random`. Celui-ci le fait, une
# fois, a une taille qui tourne en ~1s.

def _tiny(config, key):
    return {**config, "N": 32, "T_MAX": 0.06, "T_START": 0.02, "DT": 5e-3,
            "HYBRID_DT": 0.02, "K_opt": 3, "shots": 32,
            "max_depth_override": 1, "study_name": f"dns_{key}"}


def test_a_toy_scenario_runs_through_the_real_pipeline():
    """Preuve directe que `pipeline()` (solveur + AMR + mapping Ising +
    QAOA) accepte `toy_random` de bout en bout, pas seulement
    `precompute_dns` (deja teste dans tests/solver/)."""
    scenarios = tuple((k, _tiny(c, k)) for k, c in training.TOY_SCENARIOS_A)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traces = {k: precompute_dns(c) for k, c in scenarios}
        objective = training.make_composite_objective(traces, scenarios)
        study = optuna.create_study()
        study.optimize(objective, n_trials=1)

    trial = study.best_trial
    for key, _ in scenarios:
        loss = trial.user_attrs[f"loss_{key}"]
        assert np.isfinite(loss), key
        assert loss < 10.0, f"{key} a pris la penalite d'exception"
    assert np.isfinite(study.best_value)


@pytest.fixture
def no_real_dns(monkeypatch):
    """`_precompute_dns_for` ne doit jamais tourner pour de vrai ici : ces
    tests verifient le cablage, pas le solveur (deja teste directement
    dans tests/solver/)."""
    monkeypatch.setattr(training, "_precompute_dns_for",
                        lambda scenario_list, label="": {
                            k: (None, None) for k, _ in scenario_list})


def _spy_composite_loop(monkeypatch, calls):
    def fake(trial, scenario_list, dns_traces, hyperparams, lambda_cost,
             classical_only=False):
        calls.append({"scenario_list": scenario_list,
                      "classical_only": classical_only})
        return 0.42
    monkeypatch.setattr(training, "_composite_loop", fake)


# ══════════════════════════════════════════════════════════════════
#  Les scenarios jouets eux-memes
# ══════════════════════════════════════════════════════════════════

def test_toy_scenarios_cover_eight_distinct_seeds():
    all_seeds = [cfg["toy_seed"] for _, cfg in training.TOY_SCENARIOS_ALL]
    assert len(all_seeds) == 8
    assert len(set(all_seeds)) == 8, "des graines jouettes dupliquees"


def test_toy_scenarios_a_and_b_never_share_a_seed():
    """Le lot B doit etre du jamais-vu pour la phase amorcee par le lot A
    -- meme principe que `TRAINING_REGIME_GRID`/`HOLDOUT_GRID`."""
    seeds_a = {cfg["toy_seed"] for _, cfg in training.TOY_SCENARIOS_A}
    seeds_b = {cfg["toy_seed"] for _, cfg in training.TOY_SCENARIOS_B}
    assert seeds_a & seeds_b == set()
    assert training.TOY_SCENARIOS_A + training.TOY_SCENARIOS_B == \
        training.TOY_SCENARIOS_ALL


def test_toy_scenarios_declare_every_key_create_argus_requires():
    for key, cfg in training.TOY_SCENARIOS_ALL:
        missing = [k for k in training.REQUIRED_SCENARIO_KEYS if k not in cfg]
        assert not missing, f"{key} : cles manquantes {missing}"
        training.create_argus(cfg)  # ne doit pas lever


def test_toy_scenarios_pass_the_wellformed_guard():
    training._assert_scenarios_wellformed(training.TOY_SCENARIOS_ALL)
    training._assert_scenarios_wellformed(training.TOY_SCENARIOS_A)
    training._assert_scenarios_wellformed(training.TOY_SCENARIOS_B)


def test_toy_scenarios_never_collide_with_a_real_scenario_key():
    real_keys = {k for k, _ in training.SCENARIOS_ALL}
    toy_keys = {k for k, _ in training.TOY_SCENARIOS_ALL}
    assert real_keys & toy_keys == set()


# ══════════════════════════════════════════════════════════════════
#  Etudes Optuna : journaux jamais partages avec le chemin reel
# ══════════════════════════════════════════════════════════════════

def test_toy_phases_have_their_own_study_names_disjoint_from_the_real_ones():
    real_names = {v["study_name"] for k, v in training.PHASES.items()
                 if not k.startswith("toy_")}
    toy_names = {v["study_name"] for k, v in training.PHASES.items()
                if k.startswith("toy_")}
    assert len(toy_names) == 6, "les 6 phases jouettes (3 quantiques + 3 classiques)"
    assert real_names & toy_names == set(), (
        "une etude jouette partagerait le journal Optuna d'une etude reelle")


def test_toy_phase_contract_differs_from_the_real_one():
    """Si les deux contrats coincidaient, `_open_phase_study` ne pourrait
    plus distinguer une reprise reelle d'une reprise jouette."""
    real_obj = training.make_composite_objective(
        {k: (None, None) for k, _ in training.SCENARIOS_ISOLATED},
        training.SCENARIOS_ISOLATED)
    toy_obj = training.make_composite_objective(
        {k: (None, None) for k, _ in training.TOY_SCENARIOS_A},
        training.TOY_SCENARIOS_A)
    _, real_hash = training._campaign_contract(
        "phase1_composite", training.PHASES["phase1_composite"], real_obj)
    _, toy_hash = training._campaign_contract(
        "toy_phase1_composite", training.PHASES["toy_phase1_composite"], toy_obj)
    assert real_hash != toy_hash


# ══════════════════════════════════════════════════════════════════
#  `_run_toy_phase1/2/3` : routage, espion sur `_composite_loop`
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def isolated_journal(monkeypatch, tmp_path):
    """`run_phase` ouvre un journal Optuna reel sur disque
    (`JOURNAL_DIR`/`data_dir`) : sans cette isolation, deux appels de test
    avec des `n_trials` differents rouvriraient le MEME journal partage
    avec le reste du depot et se heurteraient a `campaign contract
    mismatch` -- ce que `test_train_hyperparams_smoke.py` isole deja de
    la meme facon. `JOURNAL_DIR = tmp_path` (pas un sous-dossier) et
    `_DIRS_READY = True` : `tmp_path` existe deja (cree par pytest), donc
    `ensure_dirs()` n'a besoin de rien creer -- un sous-dossier non cree
    resterait absent si `_DIRS_READY` etait deja passe a True par un test
    precedent dans le meme processus."""
    monkeypatch.setattr(training, "data_dir", str(tmp_path))
    monkeypatch.setattr(training, "JOURNAL_DIR", str(tmp_path))
    monkeypatch.setattr(training, "_DIRS_READY", True)


def test_run_toy_phase1_trains_only_on_the_first_toy_batch(
        monkeypatch, no_real_dns, isolated_journal):
    calls = []
    _spy_composite_loop(monkeypatch, calls)
    training._run_toy_phase1(seed=0, n_trials=3)
    assert len(calls) == 3
    seen_keys = {k for c in calls for k, _ in c["scenario_list"]}
    assert seen_keys <= {k for k, _ in training.TOY_SCENARIOS_A}
    assert not any(c["classical_only"] for c in calls)


def test_run_toy_classical_phase1_flags_classical_only(
        monkeypatch, no_real_dns, isolated_journal):
    calls = []
    _spy_composite_loop(monkeypatch, calls)
    training._run_toy_classical_phase1(seed=0)
    assert calls, "aucun essai n'a tourne"
    assert all(c["classical_only"] for c in calls)


def test_toy_phase1_and_classical_phase1_can_share_a_precomputed_regime(
        monkeypatch, no_real_dns, isolated_journal):
    """Meme discipline que le chemin reel : un seul precalcul partage
    entre les deux bras quand l'appelant le fournit (cout, et 'les bras
    compares partagent DNS' -- CLAUDE.md). `_run_toy_classical_phase1`
    n'a pas de parametre `n_trials` (comme son pendant reel
    `_run_classical_phase1`) : il tourne sur son budget par defaut
    complet -- rapide ici car `_composite_loop` est simule."""
    shared = training._precompute_dns_by_regime(
        training.TOY_SCENARIOS_A, label="test")
    calls = []
    _spy_composite_loop(monkeypatch, calls)
    training._run_toy_phase1(seed=0, n_trials=1,
                             dns_traces_by_regime=shared)
    training._run_toy_classical_phase1(seed=0, dns_traces_by_regime=shared)
    assert any(not c["classical_only"] for c in calls), "bras quantique absent"
    assert any(c["classical_only"] for c in calls), "bras classique absent"
    toy_a_keys = {k for k, _ in training.TOY_SCENARIOS_A}
    assert all({k for k, _ in c["scenario_list"]} == toy_a_keys for c in calls)


# ══════════════════════════════════════════════════════════════════
#  `run_toy_holdout` : LE point du pool jouet -- valide sur le REEL
# ══════════════════════════════════════════════════════════════════

def test_run_toy_holdout_validates_against_the_real_scenarios_not_the_toy_pool(
        monkeypatch, no_real_dns):
    """Si ceci validait contre `TOY_SCENARIOS_ALL`, l'entrainement et la
    validation partageraient la meme source de donnees -- exactement ce
    que la refonte USER visait a eviter."""
    seen_scenario_lists = []

    def fake(trial, scenario_list, dns_traces, hyperparams, lambda_cost,
             classical_only=False):
        seen_scenario_lists.append([k for k, _ in scenario_list])
        return 0.1
    monkeypatch.setattr(training, "_composite_loop", fake)

    study = optuna.create_study(direction="minimize")
    trial = optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE, value=0.1,
        params={"beta": 1.0},
        distributions={"beta": optuna.distributions.FloatDistribution(0.0, 10.0)},
        user_attrs={"hyperparams_resolved": {"beta": 1.0}},
    )
    study.add_trial(trial)

    training.run_toy_holdout(study, study)

    real_keys = {k for k, _ in training.SCENARIOS_ALL}
    toy_keys = {k for k, _ in training.TOY_SCENARIOS_ALL}
    seen_keys = {k for lst in seen_scenario_lists for k in lst}
    assert seen_keys, "select_by_holdout_validation n'a rien evalue"
    assert seen_keys <= real_keys, (
        f"cles vues hors des 8 scenarios reels : {seen_keys - real_keys}")
    assert not (seen_keys & toy_keys), (
        "la validation a vu des scenarios JOUETS -- elle doit rester "
        "tenue a l'ecart du pool d'entrainement")


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("phase", [
    "toy_1", "toy_2", "toy_3", "toy_classical_1", "toy_classical_2",
    "toy_classical_3", "toy_classical", "toy_all"])
def test_every_toy_phase_choice_parses(phase):
    args = training.parse_args(["--phase", phase])
    assert args.phase == phase


def test_n_trials_is_accepted_for_toy_1_but_not_toy_2():
    training.parse_args(["--phase", "toy_1", "--n-trials", "5"])
    with pytest.raises(SystemExit):
        training.parse_args(["--phase", "toy_2", "--n-trials", "5"])


def test_prepare_only_is_refused_for_toy_phases():
    """Pas d'equivalent jouet pour --prepare-only/--finalize-only dans ce
    cablage (documente, pas un oubli) : refuser plutot que faire semblant."""
    with pytest.raises(SystemExit):
        training.parse_args(["--phase", "toy_1", "--prepare-only"])
