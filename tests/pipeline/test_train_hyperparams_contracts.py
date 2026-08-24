"""Contrats de `src/train_hyperparams.py` — le seul script d'entrainement.

Ce fichier existe parce que ce script va tourner une semaine sur des
coeurs loues. Une erreur qui ne se voit qu'a la fin coute la campagne
entiere, et les defauts trouves a l'audit etaient tous de ce type : rien
ne levait, tout paraissait normal, et le resultat etait faux.

Les quatre questions, appliquees ici :
  1. pourquoi cette fonction existe-t-elle,
  2. que promet-elle,
  3. consomme-t-elle ce que sa signature annonce,
  4. deux chemins censes coincider coincident-ils encore ?

La quatrieme a produit la majorite de ce qui suit : « ce que le module
declare explorer » contre « ce qu'Optuna recoit reellement », « le jeu de
scenarios annonce » contre « le jeu parcouru », « le budget d'essais
demande » contre « le budget consomme par N workers ».
"""
import json
import os
import subprocess
import sys

import numpy as np
import optuna
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

optuna.logging.set_verbosity(optuna.logging.WARNING)

import train_hyperparams as TH


#: Le perimetre decide pour la reoptimisation. Ecrit ici en toutes
#: lettres : si quelqu'un ajoute ou retire un parametre a `SEARCH_SPACE`
#: sans decision, ce test le dit.
#:
#: Il l'a dit. `relative_percentile` est le NEUVIEME, ajoute
#: deliberement : le critere relatif `min(absolu, percentile)` est ce qui
#: rend les coefficients vivants a N=256, et son percentile etait la
#: derniere constante en dur du chemin de decision. Neuf tests de ce
#: fichier sont tombes sur cette ligne — c'est le comportement voulu.
#: Cablage et mutation verifies par
#: `tests/pipeline/test_relative_percentile_is_trainable.py`.
PERIMETRE_9 = (
    "beta", "w_z_frac", "sigma", "beta_curl", "beta_xpoint",
    "gamma_hydro", "gamma_mag", "kappa", "relative_percentile",
)


# ══════════════════════════════════════════════════════════════════
#  1. L'espace de recherche declare est celui qu'Optuna recoit
# ══════════════════════════════════════════════════════════════════

def test_search_space_is_the_nine_parameter_perimeter():
    assert set(TH.search_space()) == set(PERIMETRE_9)
    assert len(TH.SEARCH_SPACE) == 9


def test_threshold_amr_is_fixed_not_searched():
    """Le seuil est gele pour que la comparaison porte sur le quantique.

    C'est une decision, pas un oubli : elle doit rester lisible depuis
    l'exterieur du module.
    """
    assert "threshold_amr" not in TH.SEARCH_SPACE
    assert TH.FIXED_PARAMS["threshold_amr"] == TH.CLASSICAL_BEST_THRESHOLD


def test_the_frozen_threshold_is_inside_the_classical_range():
    """Sinon le bras quantique serait gele sur une valeur que le bras
    classique n'avait pas le droit de proposer — les deux bras ne
    seraient plus comparables."""
    lo, hi = TH.CLASSICAL_THRESHOLD_RANGE
    assert lo <= TH.CLASSICAL_BEST_THRESHOLD <= hi


def _params_actually_offered_to_optuna():
    """Ce qu'Optuna a REELLEMENT vu, distributions comprises."""
    study = optuna.create_study()
    captured = {}

    def objective(trial):
        TH.suggest_hyperparams(trial)
        captured.update(trial.distributions)
        return 0.0

    study.optimize(objective, n_trials=1)
    return captured


def test_optuna_receives_exactly_what_the_module_declares():
    """Question 4 : la declaration et l'echantillonnage coincident-ils ?

    Ils ne coincidaient pas. Quatre valeurs ecrites `if "x" not in
    frozen: HyperParams["x"] = <constante>` passaient pour des
    parametres ; la campagne gelee croyait en explorer neuf et en
    explorait cinq. C'est l'origine de D-22.
    """
    dists = _params_actually_offered_to_optuna()
    assert set(dists) == set(PERIMETRE_9)
    for name, (lo, hi, log) in TH.SEARCH_SPACE.items():
        assert dists[name].low == lo, name
        assert dists[name].high == hi, name
        assert dists[name].log is log, name


def test_suggest_hyperparams_returns_a_complete_dict():
    """Complet = explore + fixe. C'est ce dictionnaire qui decrit le run,
    pas `trial.params`."""
    study = optuna.create_study()
    out = {}

    def objective(trial):
        out.update(TH.suggest_hyperparams(trial))
        return 0.0

    study.optimize(objective, n_trials=1)
    assert set(out) == set(PERIMETRE_9) | {"threshold_amr"}
    assert out["threshold_amr"] == TH.CLASSICAL_BEST_THRESHOLD


def test_frozen_params_leave_the_search_space_and_win():
    study = optuna.create_study()
    out, dists = {}, {}

    def objective(trial):
        out.update(TH.suggest_hyperparams(trial, frozen={"kappa": 3.0,
                                                         "threshold_amr": 0.4}))
        dists.update(trial.distributions)
        return 0.0

    study.optimize(objective, n_trials=1)
    assert "kappa" not in dists and "threshold_amr" not in dists
    assert out["kappa"] == 3.0
    assert out["threshold_amr"] == 0.4


def test_phase1_seeds_only_name_parameters_that_exist():
    for seed in TH.phase1_seeds():
        assert set(seed) <= set(TH.SEARCH_SPACE)
    assert TH.phase1_seeds(), "grille de graines vide"


def test_a_seed_for_an_unknown_parameter_shouts(monkeypatch):
    """Optuna ignore EN SILENCE une cle enfilee qui ne correspond a
    aucune distribution : une graine entierement ignoree ressemble a une
    graine appliquee."""
    monkeypatch.setitem(TH.PHASE1_SEED_GRID, "beta_michelson", [0.5])
    with pytest.raises(KeyError):
        TH.phase1_seeds()


def test_no_michelson_split_survives_in_the_source():
    """Epinglage D-31. `beta_michelson` etait propose a Optuna alors que
    `pipeline.py` ne le lit nulle part : la phase 1 optimisait un
    parametre sans effet. Le mot ne doit plus apparaitre qu'en
    commentaire."""
    import ast
    tree = ast.parse(
        open(os.path.join(_REPO_ROOT, "src", "train_hyperparams.py")).read())
    banned = {"split_michelson", "beta_michelson"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in banned:
            found.add(node.id)
        elif isinstance(node, ast.arg) and node.arg in banned:
            found.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg in banned:
            found.add(node.arg)
        elif isinstance(node, ast.Constant) and node.value in banned:
            found.add(node.value)
    assert found == set(), f"encore vivants dans le code : {sorted(found)}"


# ══════════════════════════════════════════════════════════════════
#  2. Les jeux de scenarios
# ══════════════════════════════════════════════════════════════════

def test_no_scenario_is_counted_twice():
    """`SCENARIOS_ALL = ISOLATED + COMPLEX` valait six entrees pour
    quatre scenarios distincts : `ot` et `rotor` etaient simules deux
    fois par essai, ponderes double, pour le double du cout."""
    for name in ("SCENARIOS_ISOLATED", "SCENARIOS_COMPLEX", "SCENARIOS_ALL"):
        keys = [k for k, _ in getattr(TH, name)]
        assert len(keys) == len(set(keys)), f"{name} contient un doublon : {keys}"


def test_the_protocol_exposes_eight_unique_loso_scenarios():
    """Le protocole repose sur des scenarios qui isolent UN type
    d'anomalie. Le JSON deploye le confirme : son bloc `per_scenario` de
    phase 1 liste ces quatre-la."""
    assert [k for k, _ in TH.SCENARIOS_ISOLATED] == \
        ["kh", "vortex", "tearing", "coalescence",
         "double_tearing", "magnetic_twist"]
    assert [k for k, _ in TH.SCENARIOS_COMPLEX] == ["ot", "rotor"]
    assert len(TH.SCENARIOS_ALL) == 8


def test_every_declared_scenario_is_used():
    """Un SCENARIO_* defini et jamais parcouru est du code mort qui
    ressemble a de la configuration."""
    declared = {name: getattr(TH, name) for name in dir(TH)
                if name.startswith("SCENARIO_")}
    used = {id(cfg) for _, cfg in TH.SCENARIOS_ALL}
    unused = [n for n, cfg in declared.items() if id(cfg) not in used]
    assert unused == [], f"scenarios definis mais jamais utilises : {unused}"


def test_assert_scenarios_wellformed_catches_a_duplicate():
    with pytest.raises(ValueError, match="double"):
        TH._assert_scenarios_wellformed(
            [("ot", TH.SCENARIO_OT), ("ot", TH.SCENARIO_OT)])


def test_assert_scenarios_wellformed_catches_an_empty_sweep():
    with pytest.raises(ValueError):
        TH._assert_scenarios_wellformed([])


def test_assert_scenarios_wellformed_catches_a_missing_dns_trace():
    with pytest.raises(KeyError):
        TH._assert_scenarios_wellformed([("ot", TH.SCENARIO_OT)], dns_traces={})


# ══════════════════════════════════════════════════════════════════
#  3. create_argus — le repli qui a coute le terme de point X
# ══════════════════════════════════════════════════════════════════

def test_every_scenario_enables_advanced_anomalies():
    """Orszag-Tang ne portait pas la cle. `create_argus` repliait sur
    False : OT etait le seul scenario a tourner sans anomalies avancees,
    donc sans terme ZZZZ de point X. La phase 2 entrainait
    `beta_xpoint` sur un jeu ou l'un des deux scenarios ne pouvait pas
    l'exprimer."""
    for key, cfg in TH.SCENARIOS_ALL:
        assert cfg.get("AdvAnomaliesEnable") is True, key
        assert TH.create_argus(cfg).AdvAnomaliesEnable is True, key


def test_create_argus_raises_on_an_incomplete_config():
    incomplete = {k: v for k, v in TH.SCENARIO_OT.items()
                  if k != "AdvAnomaliesEnable"}
    with pytest.raises(KeyError, match="AdvAnomaliesEnable"):
        TH.create_argus(incomplete)


def test_create_argus_carries_the_scenario_physics():
    argus = TH.create_argus(TH.SCENARIO_KH)
    assert argus.Re == TH.SCENARIO_KH["Re"]
    assert argus.Rm == TH.SCENARIO_KH["Rm"]
    assert argus.shots == TH.SCENARIO_KH["shots"]
    assert argus.K_opt == TH.SCENARIO_KH["K_opt"]


# ══════════════════════════════════════════════════════════════════
#  4. La boucle composite et l'elagage
# ══════════════════════════════════════════════════════════════════

class _FakeTrial:
    """Un essai qui enregistre ce qu'on lui rapporte, sans base."""

    def __init__(self, prune_at=None):
        self.number = 0
        self.reports = []
        self.user_attrs = {}
        self._prune_at = prune_at

    def report(self, value, step):
        self.reports.append((step, value))

    def should_prune(self):
        return (self._prune_at is not None
                and self.reports and self.reports[-1][0] >= self._prune_at)

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def _fake_pipeline_returning(values):
    """Remplace `pipeline` : renvoie une perte par scenario, dans l'ordre."""
    seq = iter(values)

    def fake(**kwargs):
        return {"combined": next(seq), "phys_score": 0.0, "patch_ratio": 0.5,
                "field_errors": {}}
    return fake


def test_composite_loop_averages_the_sub_losses(monkeypatch):
    losses = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    monkeypatch.setattr(TH, "pipeline", _fake_pipeline_returning(losses))
    trial = _FakeTrial()
    traces = {k: (None, None) for k, _ in TH.SCENARIOS_ISOLATED}
    out = TH._composite_loop(trial, TH.SCENARIOS_ISOLATED, traces, {}, 0.4)
    assert out == pytest.approx(0.7)
    assert trial.user_attrs["loss_kh"] == pytest.approx(0.2)
    assert trial.user_attrs["loss_coalescence"] == pytest.approx(0.8)


def test_the_running_mean_is_reported_at_every_step(monkeypatch):
    """C'est ce qui rend le MedianPruner utilisable : une valeur
    comparable entre essais, a chaque step."""
    losses = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    monkeypatch.setattr(TH, "pipeline", _fake_pipeline_returning(losses))
    trial = _FakeTrial()
    traces = {k: (None, None) for k, _ in TH.SCENARIOS_ISOLATED}
    TH._composite_loop(trial, TH.SCENARIOS_ISOLATED, traces, {}, 0.4)
    assert [s for s, _ in trial.reports] == list(range(6))
    assert [round(v, 6) for _, v in trial.reports] == \
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def test_pruning_stops_the_loop_before_the_remaining_scenarios(monkeypatch):
    """Un essai elague au 3e scenario ne doit pas simuler le 4e — c'est
    tout l'interet de l'elagage sur des coeurs loues."""
    calls = []

    def fake(**kwargs):
        calls.append(kwargs["scenario"])
        return {"combined": 1.0, "phys_score": 0.0, "patch_ratio": 0.5,
                "field_errors": {}}

    monkeypatch.setattr(TH, "pipeline", fake)
    trial = _FakeTrial(prune_at=2)
    traces = {k: (None, None) for k, _ in TH.SCENARIOS_ISOLATED}
    with pytest.raises(optuna.TrialPruned):
        TH._composite_loop(trial, TH.SCENARIOS_ISOLATED, traces, {}, 0.4)
    assert len(calls) == 3
    # les sous-pertes deja calculees sont conservees, pas perdues
    assert "loss_kh" in trial.user_attrs


def test_reporting_only_at_step_zero_would_never_prune():
    """Epinglage de l'ancien comportement. `n_warmup_steps=2` fait que
    `should_prune()` renvoie toujours False au step 0 : le pruner etait
    decoratif. Mesure : 1e9 rapporte au step 0 apres 40 essais a 1.0 ne
    declenche rien."""
    study = optuna.create_study(pruner=TH.make_pruner())
    # 40 essais de reference, qui rapportent A CHAQUE step — c'est la
    # condition pour que le MedianPruner ait une mediane a comparer.
    for i in range(40):
        t = study.ask()
        for step in range(4):
            t.report(1.0 + i, step=step)
        study.tell(t, 1.0 + i)

    t = study.ask()
    t.report(1e9, step=0)
    assert not t.should_prune(), "le step 0 seul n'elague jamais"
    t.report(1e9, step=1)
    t.report(1e9, step=2)
    assert t.should_prune(), "au 3e scenario, il doit mordre"


def test_a_failing_scenario_costs_a_finite_penalty(monkeypatch):
    """Un `inf` n'ordonne rien : le TPE ne peut pas modeliser l'espace
    autour d'un essai infini."""
    def boom(**kwargs):
        raise RuntimeError("solveur diverge")

    monkeypatch.setattr(TH, "pipeline", boom)
    trial = _FakeTrial()
    loss = TH._run_one_scenario(trial, "ot", TH.SCENARIO_OT,
                                {"ot": (None, None)}, {}, 0.4)
    assert loss == 10.0
    assert np.isfinite(loss)
    assert trial.user_attrs["patch_ot"] == 1.0


def test_a_nan_loss_becomes_the_same_finite_penalty(monkeypatch):
    monkeypatch.setattr(TH, "pipeline", _fake_pipeline_returning([np.nan]))
    trial = _FakeTrial()
    loss = TH._run_one_scenario(trial, "ot", TH.SCENARIO_OT,
                                {"ot": (None, None)}, {}, 0.4)
    assert loss == 10.0


def test_an_incomplete_pipeline_result_cannot_be_selected(monkeypatch):
    def incomplete(**kwargs):
        return {
            "combined": 0.001,
            "phys_score": 0.001,
            "patch_ratio": 0.01,
            "completed": False,
            "abort": {"kind": "numerical_divergence"},
        }

    monkeypatch.setattr(TH, "pipeline", incomplete)
    trial = _FakeTrial()
    loss = TH._run_one_scenario(
        trial, "ot", TH.SCENARIO_OT, {"ot": (None, None)}, {}, 0.4)
    assert loss == TH.DIVERGENCE_PENALTY
    assert trial.user_attrs["completed_ot"] is False
    assert trial.user_attrs["patch_ot"] == 1.0


def test_the_objective_refuses_a_malformed_scenario_set_before_any_trial():
    """Le refus doit tomber a la CONSTRUCTION, pas au milieu du premier
    essai — c'est-a-dire apres le pre-calcul DNS."""
    with pytest.raises(ValueError):
        TH.make_composite_objective({"ot": (None, None)},
                                    [("ot", TH.SCENARIO_OT),
                                     ("ot", TH.SCENARIO_OT)])


# ══════════════════════════════════════════════════════════════════
#  5. Le budget d'essais partage entre workers
# ══════════════════════════════════════════════════════════════════

def _count_objective(counter):
    def objective(trial):
        counter.append(trial.number)
        return float(trial.suggest_float("x", 0.0, 1.0))
    return objective


def test_n_workers_share_one_budget_they_do_not_multiply_it(tmp_path, monkeypatch):
    """Le defaut le plus cher de l'audit : `remaining = cible - faits`
    etait calcule UNE fois. Huit workers demarrant ensemble lisaient tous
    « 0 fait » et faisaient la campagne entiere chacun."""
    storage = f"sqlite:///{tmp_path / 'budget.db'}"
    monkeypatch.setattr(TH, "_get_storage", lambda _: storage)
    monkeypatch.setattr(TH, "WORKER_TRIALS", None)
    config = {"n_trials": 12, "study_name": "budget"}

    counter = []
    for _ in range(4):                      # 4 « workers », l'un apres l'autre
        TH.run_phase("budget", config, _count_objective(counter))

    study = optuna.load_study(study_name="budget", storage=storage)
    assert TH.trials_done(study) == 12, "budget global depasse ou incomplet"
    assert len(counter) == 12


def test_worker_trials_caps_one_worker(tmp_path, monkeypatch):
    storage = f"sqlite:///{tmp_path / 'cap.db'}"
    monkeypatch.setattr(TH, "_get_storage", lambda _: storage)
    monkeypatch.setattr(TH, "WORKER_TRIALS", 3)
    config = {"n_trials": 100, "study_name": "cap"}

    counter = []
    TH.run_phase("cap", config, _count_objective(counter))
    assert len(counter) == 3


def test_a_finished_phase_runs_nothing(tmp_path, monkeypatch):
    storage = f"sqlite:///{tmp_path / 'done.db'}"
    monkeypatch.setattr(TH, "_get_storage", lambda _: storage)
    monkeypatch.setattr(TH, "WORKER_TRIALS", None)
    config = {"n_trials": 3, "study_name": "done"}

    first, second = [], []
    TH.run_phase("done", config, _count_objective(first))
    TH.run_phase("done", config, _count_objective(second))
    assert len(first) == 3
    assert second == []


def test_trials_done_ignores_queued_seeds(tmp_path, monkeypatch):
    """Une graine en file d'attente n'est pas un essai fait : la compter
    ferait s'arreter la campagne avant d'avoir calcule quoi que ce soit."""
    storage = f"sqlite:///{tmp_path / 'waiting.db'}"
    study = optuna.create_study(study_name="w", storage=storage)
    for v in (0.1, 0.2, 0.3):
        study.enqueue_trial({"x": v})
    assert TH.trials_done(study) == 0


def test_interrupted_trials_are_failed_and_retried_in_the_budget(
        tmp_path, monkeypatch):
    storage = f"sqlite:///{tmp_path / 'interrupted.db'}"
    study = optuna.create_study(study_name="interrupted", storage=storage)
    interrupted = study.ask()
    assert interrupted.number == 0
    assert TH.trials_done(study) == 1

    assert TH.fail_interrupted_trials(study) == 1
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert TH.trials_done(study) == 0


def test_prepare_phase1_binds_contract_and_queues_seed(tmp_path, monkeypatch):
    storage = f"sqlite:///{tmp_path / 'prepare.db'}"
    monkeypatch.setattr(TH, "_get_storage", lambda _: storage)
    study, recovered = TH.prepare_phase1(7)
    assert recovered == 0
    assert study.user_attrs["campaign_contract_sha256"]
    assert TH.trials_done(study) == 0
    waiting = [trial for trial in study.trials
               if trial.state == optuna.trial.TrialState.WAITING]
    assert len(waiting) == len(TH.phase1_seeds())


def test_the_sampler_seed_makes_a_phase_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(TH, "WORKER_TRIALS", None)

    def run(tag, seed):
        storage = f"sqlite:///{tmp_path / (tag + '.db')}"
        monkeypatch.setattr(TH, "_get_storage", lambda _: storage)
        drawn = []

        def objective(trial):
            x = trial.suggest_float("x", 0.0, 1.0)
            drawn.append(x)
            return x

        TH.run_phase(tag, {"n_trials": 8, "study_name": tag}, objective, seed=seed)
        return drawn

    assert run("a", 1234) == run("b", 1234)
    assert run("c", 4321) != run("a2", 1234)


def test_campaign_contract_refuses_a_changed_budget_on_resume(
        tmp_path, monkeypatch):
    storage = f"sqlite:///{tmp_path / 'contract.db'}"
    monkeypatch.setattr(TH, "_get_storage", lambda _: storage)
    monkeypatch.setattr(TH, "WORKER_TRIALS", None)
    TH.run_phase(
        "contract", {"n_trials": 1, "study_name": "contract"},
        _count_objective([]), seed=4)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        TH.run_phase(
            "contract", {"n_trials": 2, "study_name": "contract"},
            _count_objective([]), seed=4)


def test_trials_record_worker_seed_and_contract(tmp_path, monkeypatch):
    storage = f"sqlite:///{tmp_path / 'attrs.db'}"
    monkeypatch.setattr(TH, "_get_storage", lambda _: storage)
    monkeypatch.setattr(TH, "WORKER_TRIALS", None)
    study = TH.run_phase(
        "attrs", {"n_trials": 1, "study_name": "attrs"},
        _count_objective([]), seed=17)
    trial = study.trials[0]
    assert trial.user_attrs["worker_seed"] == 17
    assert trial.user_attrs["campaign_contract_sha256"] == \
        study.user_attrs["campaign_contract_sha256"]


# ══════════════════════════════════════════════════════════════════
#  6. Le JSON de sortie — la provenance de D-22
# ══════════════════════════════════════════════════════════════════

def _study_with_one_trial(name="s"):
    study = optuna.create_study(study_name=name)

    def objective(trial):
        hp = TH.suggest_hyperparams(trial)
        trial.set_user_attr("hyperparams_resolved", hp)
        for key, _ in TH.SCENARIOS_ALL:
            trial.set_user_attr(f"loss_{key}", 0.25)
        return 0.25

    study.optimize(objective, n_trials=2)
    return study


def test_deployable_params_are_complete_where_best_params_are_not():
    """`study.best_params` ne porte que l'echantillonne. Un JSON bati
    dessus perd les parametres fixes, et le deploiement les comble par
    des replis que personne n'a choisis. C'est le mecanisme de D-22."""
    study = _study_with_one_trial()
    assert "threshold_amr" not in study.best_params        # le manque
    params, source = TH.deployable_params(study)
    assert set(params) == set(PERIMETRE_9) | {"threshold_amr"}
    assert source == "trial_user_attr"


def test_deployable_params_says_so_when_it_has_to_rebuild():
    study = optuna.create_study()
    study.optimize(lambda t: float(TH.suggest_hyperparams(t)["beta"]), n_trials=1)
    params, source = TH.deployable_params(study)   # pas d'attribut resolu
    assert source == "rebuilt_from_best_params"
    assert set(params) == set(PERIMETRE_9) | {"threshold_amr"}


def test_saved_json_carries_everything_needed_to_redeploy(tmp_path, monkeypatch):
    monkeypatch.setattr(TH, "data_dir", str(tmp_path))
    monkeypatch.setattr(TH, "_DIRS_READY", True)
    p1, p2, p3 = (_study_with_one_trial(n) for n in ("p1", "p2", "p3"))

    path = TH._save_results(p1, p2, p3, filename="out.json")
    saved = json.load(open(path))

    assert set(saved["deploy"]["quantum"]) == set(PERIMETRE_9) | {"threshold_amr"}
    assert saved["provenance"]["git_commit"] != ""
    assert "argv" in saved["provenance"]
    assert set(saved["search_space"]) == set(PERIMETRE_9)
    assert saved["fixed_params"]["threshold_amr"] == TH.CLASSICAL_BEST_THRESHOLD
    assert saved["scenarios"]["all"] == [k for k, _ in TH.SCENARIOS_ALL]
    assert saved["lambda_cost"] == TH.LAMBDA_COST_SOFT


def test_saved_json_survives_a_phase_where_nothing_completed(tmp_path, monkeypatch):
    """Une phase sans essai termine ne doit pas faire lever `best_value`
    au moment d'ecrire — sinon la campagne perd aussi ce qui a marche."""
    monkeypatch.setattr(TH, "data_dir", str(tmp_path))
    monkeypatch.setattr(TH, "_DIRS_READY", True)
    empty = optuna.create_study(study_name="empty")
    ok = _study_with_one_trial("ok")

    path = TH._save_results(ok, empty, ok, filename="partial.json")
    saved = json.load(open(path))
    assert saved["quantum"]["phase2"]["best_score"] is None


def test_report_best_does_not_raise_on_an_empty_study(capsys):
    TH._report_best(optuna.create_study(), "Phase X", TH.SCENARIOS_ALL)
    assert "aucun essai termine" in capsys.readouterr().out


# ══════════════════════════════════════════════════════════════════
#  7. Le lancement lui-meme — ce que les coeurs loues executeront
# ══════════════════════════════════════════════════════════════════

def test_importing_the_module_writes_nothing_and_prints_nothing(tmp_path):
    """Un import qui cree des repertoires et affiche s'execute une fois
    par worker, et rend le module intestable."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import train_hyperparams"
         % os.path.join(_REPO_ROOT, "src")],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"import bavard : {out.stdout!r}"
    assert not (tmp_path / "Train_results").exists()


def test_print_space_answers_before_any_core_is_rented(capsys):
    """`--print-space` doit repondre sans rien calculer : c'est la
    verification qu'on fait AVANT de lancer une semaine de calcul."""
    assert TH.main(["--print-space"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["search_space"]) == set(PERIMETRE_9)
    assert payload["fixed_params"]["threshold_amr"] == TH.CLASSICAL_BEST_THRESHOLD
    assert payload["scenarios"]["all"] == [k for k, _ in TH.SCENARIOS_ALL]
    assert set(payload["n_trials"]) == set(TH.PHASES)


def test_print_space_from_the_command_line():
    out = subprocess.run(
        [sys.executable, os.path.join(_REPO_ROOT, "src", "train_hyperparams.py"),
         "--print-space"],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=300,
        env={**os.environ, "PYTHONPATH": os.path.join(_REPO_ROOT, "src")})
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout[out.stdout.index("{"):])
    assert set(payload["search_space"]) == set(PERIMETRE_9)


def test_every_phase_name_the_cli_accepts_is_routable():
    """Une phase acceptee par l'analyseur d'arguments mais absente du
    routage sortirait en silence sans rien faire."""
    for phase in TH.PHASE_CHOICES:
        args = TH.parse_args(["--phase", phase])
        assert args.phase == phase
    routed = {"1": "phase1_composite", "2": "phase2_complex",
              "3": "phase3_validation", "classical_1": "classical_phase1",
              "classical_2": "classical_phase2", "classical_3": "classical_phase3"}
    for key in routed.values():
        assert key in TH.PHASES


def test_seed_is_only_an_explicit_cli_argument():
    assert TH.parse_args([]).seed is None
    assert TH.parse_args(["--seed", "77"]).seed == 77


def test_the_phases_table_only_carries_keys_that_are_read():
    """Une constante deguisee en reglage est le motif d'erreur le plus
    couteux de ce depot. `classical_only: True` ne rendait pas la phase
    classique ; `split_michelson` n'etait jamais relu."""
    for name, config in TH.PHASES.items():
        assert set(config) == {"n_trials", "study_name"}, name


def test_run_phase_reads_only_those_two_keys(tmp_path, monkeypatch):
    """Question 3 : consomme-t-elle ce que sa signature annonce ?"""
    storage = f"sqlite:///{tmp_path / 'min.db'}"
    monkeypatch.setattr(TH, "_get_storage", lambda _: storage)
    monkeypatch.setattr(TH, "WORKER_TRIALS", None)
    counter = []
    TH.run_phase("minimal", {"n_trials": 2, "study_name": "minimal"},
                 _count_objective(counter))
    assert len(counter) == 2


# ══════════════════════════════════════════════════════════════════════
#  8. Le routage des phases — D-30 vivait exactement ici
# ══════════════════════════════════════════════════════════════════════
#
# `_run_phase1(study_p1, dns_traces)` : la fonction prend UN argument. Le
# chemin sequentiel levait donc `TypeError` APRES les 600 essais de la
# phase 1. Rien dans les tests ne parcourait ce chemin, parce qu'il coute
# des heures. Les tests ci-dessous le parcourent en substituant les deux
# seules fonctions couteuses — le pre-calcul DNS et l'execution des
# essais — et rien d'autre : le routage, les signatures, les graines et
# l'enchainement des etudes restent le vrai code.

@pytest.fixture
def cheap_phases(monkeypatch):
    """Remplace le DNS et l'execution d'essais. Enregistre les appels."""
    calls = {"dns": [], "phases": []}

    def fake_dns(scenario_list, label="scenarios"):
        TH._assert_scenarios_wellformed(scenario_list)
        calls["dns"].append([k for k, _ in scenario_list])
        return {k: (None, None) for k, _ in scenario_list}

    def fake_run_phase(phase_name, phase_config, objective_fn,
                       seed_params=None, seed=None):
        calls["phases"].append({
            "name": phase_name,
            "study_name": phase_config["study_name"],
            "n_trials": phase_config["n_trials"],
            "n_seeds": len(seed_params or []),
            "seed": seed,
        })
        study = optuna.create_study(study_name=phase_config["study_name"])

        def wrapped(trial):
            hp = TH.suggest_hyperparams(trial)
            trial.set_user_attr("hyperparams_resolved", hp)
            for key, _ in TH.SCENARIOS_ALL:
                trial.set_user_attr(f"loss_{key}", 0.3)
            return 0.3

        study.optimize(wrapped, n_trials=1)
        return study

    monkeypatch.setattr(TH, "_precompute_dns_for", fake_dns)
    monkeypatch.setattr(TH, "run_phase", fake_run_phase)
    monkeypatch.setattr(TH, "_load_study",
                        lambda key: optuna.create_study(study_name=key))
    return calls


@pytest.mark.parametrize("runner,scenarios", [
    ("_run_phase1", "isolated"),
    ("_run_classical_phase1", "isolated"),
])
def test_the_first_phases_take_dns_traces_and_a_seed(runner, scenarios, cheap_phases):
    traces = {k: (None, None) for k, _ in TH.SCENARIOS_ISOLATED}
    study = getattr(TH, runner)(traces, 11)
    assert cheap_phases["phases"][-1]["seed"] == 11
    assert cheap_phases["phases"][-1]["n_seeds"] > 0, "phase amorcee sans graine"
    assert study.best_value == pytest.approx(0.3)


def test_phase1_trial_target_can_be_overridden(cheap_phases):
    traces = {k: (None, None) for k, _ in TH.SCENARIOS_ISOLATED}
    TH._run_phase1(traces, seed=11, n_trials=17)
    assert cheap_phases["phases"][-1]["n_trials"] == 17


@pytest.mark.parametrize("runner,expected_scenarios", [
    ("_run_phase2", ["ot", "rotor"]),
    ("_run_phase3", [k for k, _ in TH.SCENARIOS_ALL]),
    ("_run_classical_phase2", ["ot", "rotor"]),
    ("_run_classical_phase3", [k for k, _ in TH.SCENARIOS_ALL]),
])
def test_the_later_phases_take_the_previous_study(runner, expected_scenarios,
                                                  cheap_phases):
    """Elles prennent UNE etude et une graine. Une signature qui derive de
    son appelant est ce qui a fait tomber le chemin sequentiel."""
    upstream = optuna.create_study()
    upstream.optimize(lambda t: float(TH.suggest_hyperparams(t)["beta"]), n_trials=2)
    getattr(TH, runner)(upstream, 11)
    assert cheap_phases["dns"][-1] == expected_scenarios


@pytest.mark.parametrize("phase,expected_studies", [
    ("1", ["q_has_v2_phase1"]),
    ("2", ["q_has_v2_phase2"]),
    ("3", ["q_has_v2_phase3"]),
    ("classical_1", ["classical_v2_phase1"]),
    ("classical_2", ["classical_v2_phase2"]),
    ("classical_3", ["classical_v2_phase3"]),
    ("classical", ["classical_v2_phase1", "classical_v2_phase2",
                   "classical_v2_phase3"]),
    ("all", ["q_has_v2_phase1", "q_has_v2_phase2", "q_has_v2_phase3",
             "classical_v2_phase1", "classical_v2_phase2", "classical_v2_phase3"]),
])
def test_every_cli_phase_runs_the_studies_it_names(phase, expected_studies,
                                                   cheap_phases, tmp_path,
                                                   monkeypatch):
    """Chaque valeur de `--phase` est parcourue de bout en bout. C'est le
    seul test qui aurait attrape D-30 avant les coeurs loues."""
    monkeypatch.setattr(TH, "data_dir", str(tmp_path))
    monkeypatch.setattr(TH, "_DIRS_READY", True)
    assert TH.main(["--phase", phase, "--seed", "3"]) == 0
    assert [p["study_name"] for p in cheap_phases["phases"]] == expected_studies
    assert all(p["seed"] == 3 for p in cheap_phases["phases"])


def test_the_full_run_writes_its_json(cheap_phases, tmp_path, monkeypatch):
    monkeypatch.setattr(TH, "data_dir", str(tmp_path))
    monkeypatch.setattr(TH, "_DIRS_READY", True)
    TH.main(["--phase", "all", "--seed", "3"])
    saved = json.load(open(tmp_path / "best_hyperparams.json"))
    assert set(saved["deploy"]["quantum"]) == set(PERIMETRE_9) | {"threshold_amr"}
    assert set(saved["deploy"]["classical"]) == set(PERIMETRE_9) | {"threshold_amr"}


def test_phase1_cli_writes_an_explicit_candidate(
        cheap_phases, tmp_path, monkeypatch):
    monkeypatch.setattr(TH, "data_dir", str(tmp_path))
    monkeypatch.setattr(TH, "_DIRS_READY", True)
    path = tmp_path / "candidate.json"

    TH.main(["--phase", "1", "--seed", "3", "--n-trials", "17",
             "--result-path", str(path)])

    saved = json.load(open(path))
    assert saved["artifact"] == "phase_candidate"
    assert saved["status"] == "partial"
    assert saved["target_trials"] == 17
    assert saved["study_name"] == "q_has_v2_phase1"
    assert saved["result"]["best_params"] is not None
    assert not list(tmp_path.glob(".candidate.json.*.tmp"))


def test_phase_candidate_is_complete_only_when_no_trial_is_running(
        tmp_path, monkeypatch):
    monkeypatch.setattr(TH, "data_dir", str(tmp_path))
    monkeypatch.setattr(TH, "_DIRS_READY", True)
    study = _study_with_one_trial("candidate")
    path = tmp_path / "candidate.json"

    TH.save_phase_candidate(
        study, "phase1_composite", TH.SCENARIOS_ISOLATED,
        target_trials=2, output_path=str(path))

    saved = json.load(open(path))
    assert saved["status"] == "complete"
    assert saved["consumed_trials"] == 2
    assert saved["concurrency_excess_trials"] == 0


def test_phase1_budget_options_are_not_accepted_for_other_phases():
    with pytest.raises(SystemExit):
        TH.parse_args(["--phase", "2", "--n-trials", "10"])
    with pytest.raises(SystemExit):
        TH.parse_args(["--phase", "all", "--result-path", "x.json"])
    with pytest.raises(SystemExit):
        TH.parse_args(["--phase", "1", "--n-trials", "0"])


def test_a_missing_seed_is_announced_not_silently_random(cheap_phases, capsys,
                                                        tmp_path, monkeypatch):
    """Sans graine, le TPE tire au hasard : la campagne n'est pas
    reproductible. Ce n'est pas interdit, mais ca doit se voir."""
    monkeypatch.setattr(TH, "data_dir", str(tmp_path))
    monkeypatch.setattr(TH, "_DIRS_READY", True)
    TH.main(["--phase", "1"])
    assert "pas de --seed" in capsys.readouterr().out


def test_rescore_seeding_reads_the_columns_it_names(tmp_path, monkeypatch):
    """`extract_top_params_from_rescore` promet de lire les colonnes
    `param_*` du CSV et de trier par `new_score`."""
    monkeypatch.setattr(TH, "data_dir", str(tmp_path))
    folder = tmp_path / "rescore_demo_lambda0.4000"
    folder.mkdir()
    (folder / "trials_lambda0.4000.csv").write_text(
        "trial,new_score,param_beta,param_kappa\n"
        "7,0.90,1.0,5.0\n"
        "3,0.10,2.0,6.0\n"
        "5,0.50,3.0,7.0\n")
    out = TH.extract_top_params_from_rescore("demo", [0.4], top_k=2)
    assert out == [{"beta": 2.0, "kappa": 6.0}, {"beta": 3.0, "kappa": 7.0}]


def test_rescore_seeding_returns_nothing_when_there_is_no_csv(tmp_path, monkeypatch):
    """Un balayage vide doit rendre une liste vide, pas planter — et les
    appelants doivent alors le dire, pas croire avoir amorce."""
    monkeypatch.setattr(TH, "data_dir", str(tmp_path))
    assert TH.extract_top_params_from_rescore("absent", [0.4], top_k=2) == []
