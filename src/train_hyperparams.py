"""Optimisation progressive des hyperparamètres Q-HAS.

La perte composite couvre six scénarios isolés, puis deux scénarios complexes,
puis les huit scénarios. Le contrôle classique suit les mêmes partitions et
n'optimise que ``threshold_amr``. Les neuf paramètres quantiques sont déclarés
dans :data:`SEARCH_SPACE`; le seuil classique gelé est dans
:data:`FIXED_PARAMS`.

Le Hamiltonien combine un biais Z adaptatif, un couplage ZZ et des plaquettes
ZZZZ de circulation et de point X. Tous les termes utilisent la convention de
signe ferromagnétique (coefficients non positifs).
"""

import argparse
import optuna
import os
import json
import csv
import hashlib
import random
import subprocess
import sys
import itertools
import tempfile
import numpy as np
from pipeline import DIVERGENCE_PENALTY, pipeline
from types import SimpleNamespace
from Simulation.pre_compute_dns import precompute_dns

optuna.logging.set_verbosity(optuna.logging.INFO)
sys.stdout.reconfigure(line_buffering=True)

# ============================================================
#  CONFIGURATION
# ============================================================
LAMBDA_COST_SOFT = 0.4

N_TRAINING         = 256
MAX_DEPTH_TRAINING = 4

#: Taille du registre VQA a l'entrainement : 2 -> 8 qubits.
VQA_N_TRAINING = 2

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# One rented machine, one local journal shared by all worker processes.
CAMPAIGN_DIR = os.path.abspath(os.environ.get(
    "QHAS_CAMPAIGN_DIR",
    os.path.join(project_root, "results", "hyperparams", "reoptimisation"),
))
JOURNAL_DIR = os.path.abspath(os.environ.get(
    "QHAS_JOURNAL_DIR", os.path.join(CAMPAIGN_DIR, "journal")))
WORKER_TRIALS  = os.environ.get("WORKER_TRIALS", None)
if WORKER_TRIALS is not None:
    WORKER_TRIALS = int(WORKER_TRIALS)

def announce_environment():
    """Ce que le worker a compris de son environnement. Appelee par `main`."""
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    print(f"[RENTED MACHINE] Shared Optuna journal: {JOURNAL_DIR}")
    if WORKER_TRIALS:
        print(f"[RENTED MACHINE] Max trials per worker: {WORKER_TRIALS}")

#: Emplacement des journaux Optuna et du JSON final. Aucun répertoire n'est
#: créé à l'import.
data_dir = CAMPAIGN_DIR

_DIRS_READY = False


def ensure_dirs():
    """Create local campaign and journal directories once."""
    global _DIRS_READY
    if _DIRS_READY:
        return data_dir
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    _DIRS_READY = True
    return data_dir


def _git(*argv):
    try:
        return subprocess.check_output(
            ["git", *argv], cwd=project_root,
            stderr=subprocess.DEVNULL, timeout=30).decode().strip()
    except Exception:
        return ""


def provenance():
    """Return the commit, worktree state, CLI and campaign environment."""
    return {
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": list(sys.argv),
        "env": {k: os.environ.get(k) for k in
                ("QHAS_CAMPAIGN_DIR", "QHAS_JOURNAL_DIR",
                 "WORKER_TRIALS")},
    }


# ============================================================
#  SCENARIO CONFIGURATIONS
# ============================================================
# Each isolated scenario has its own physics parameters (Re, Rm)
# tuned so the targeted anomaly develops clearly within the time window.

SCENARIO_KH = {
    "scenario": "kelvin_helmholtz",
    "N": N_TRAINING,
    "max_depth_override": MAX_DEPTH_TRAINING,
    "T_MAX": 2.5,
    "T_START": 0.9,      # KH instability develops around t~1.0-1.5
    "DT": 1e-3,
    "HYBRID_DT": 0.10,
    "K_opt": 30,
    "Re": 800,
    "Rm": 800,
    "shots": 256,
    "AdvAnomaliesEnable": True,
}

SCENARIO_VORTEX = {
    "scenario": "lamb_oseen_vortex",
    "N": N_TRAINING,
    "max_depth_override": MAX_DEPTH_TRAINING,
    "T_MAX": 1.2,
    "T_START": 0.0,       # Vortex is present from t=0, start early
    "DT": 1e-3,
    "HYBRID_DT": 0.10,
    "K_opt": 30,
    "Re": 800,
    "Rm": 800,
    "shots": 256,
    "AdvAnomaliesEnable": True,
}

SCENARIO_TEARING = {
    "scenario": "harris_tearing",
    "N": N_TRAINING,
    "max_depth_override": MAX_DEPTH_TRAINING,
    "T_MAX": 1.2,
    "T_START": 0.3,       # Tearing mode develops around t~0.5-1.0
    "DT": 1e-3,
    "HYBRID_DT": 0.10,
    "K_opt": 30,
    "Re": 800,
    "Rm": 800,
    "shots": 256,
    "AdvAnomaliesEnable": True,
}

SCENARIO_DOUBLE_TEARING = {
    "scenario": "double_tearing",
    "N": N_TRAINING,
    "max_depth_override": MAX_DEPTH_TRAINING,
    "T_MAX": 1.2,
    "T_START": 0.3,
    "DT": 1e-3,
    "HYBRID_DT": 0.10,
    "K_opt": 30,
    "Re": 800,
    "Rm": 800,
    "shots": 256,
    "AdvAnomaliesEnable": True,
}

SCENARIO_MAGNETIC_TWIST = {
    "scenario": "magnetic_twist",
    "N": N_TRAINING,
    "max_depth_override": MAX_DEPTH_TRAINING,
    "T_MAX": 1.2,
    "T_START": 0.3,
    "DT": 1e-3,
    "HYBRID_DT": 0.10,
    "K_opt": 30,
    "Re": 800,
    "Rm": 800,
    "shots": 256,
    "AdvAnomaliesEnable": True,
}

# ── Island Coalescence: merging magnetic islands ──
# Two chains of magnetic islands separated by X-points.
# The perturbation drives island merging via reconnection.
# The reconnection develops on the Alfvén timescale and creates
# dynamic X-points between merging islands.
#
# HYBRID_DT rationale: reconnection timescale ~ L / v_A.
# At B0=1, the Alfvén speed ~ B0/sqrt(rho) ~ 1.
# Reconnection events are localized and fast — calling Q-HAS
# every 0.10 captures the X-point formation and migration.
SCENARIO_COALESCENCE = {
    "scenario": "island_coalescence",
    "N": N_TRAINING,
    "max_depth_override": MAX_DEPTH_TRAINING,
    "T_MAX": 1.0,
    "T_START": 0.0,       # Reconnection develops around t~0.2-0.5
    "DT": 1e-3,
    "HYBRID_DT": 0.10,    # Reconnection is fast — frequent VQA calls
    "K_opt": 30,
    "Re": 800,
    "Rm": 800,
    "shots": 256,
    "AdvAnomaliesEnable": True,  # X-point detection requires advanced anomalies
}

SCENARIO_OT = {
    "scenario": "orszag_tang",
    "N": N_TRAINING,
    "max_depth_override": MAX_DEPTH_TRAINING,
    "T_MAX": 3.2,
    "T_START": 2.0,
    "DT": 1e-3,
    "HYBRID_DT": 0.10,
    "K_opt": 30,
    "Re": 800,
    "Rm": 800,
    "shots": 256,
    # Required for the X-point term.
    "AdvAnomaliesEnable": True,
}

# ── MHD Rotor: rotating magnetised cylinder ──
# The rotor winds up the B-field from t~0.2, producing strong vortical
# structures and magnetic compression.  T_START=0.3 captures early
# dynamics; T_MAX=1.5 reaches the saturated state.
SCENARIO_ROTOR = {
    "scenario": "mhd_rotor",
    "N": N_TRAINING,
    "max_depth_override": MAX_DEPTH_TRAINING,
    "T_MAX": 1.5,
    "T_START": 0.2,
    "DT": 1e-3,
    "HYBRID_DT": 0.10,
    "K_opt": 30,
    "Re": 800,
    "Rm": 800,
    "shots": 256,
    "AdvAnomaliesEnable": True,
}

# ============================================================
#  PHASE DEFINITIONS
# ============================================================
#
# Chaque phase ne déclare que son étude et son budget. La configuration
# physique reste dans les dictionnaires ``SCENARIO_*``.

PHASES = {
    "phase1_composite":  {"n_trials": 600, "study_name": "q_has_v2_phase1"},
    "phase2_complex":    {"n_trials": 600, "study_name": "q_has_v2_phase2"},
    "phase3_validation": {"n_trials": 400, "study_name": "q_has_v2_phase3"},

    "classical_phase1":  {"n_trials": 300, "study_name": "classical_v2_phase1"},
    "classical_phase2":  {"n_trials": 300, "study_name": "classical_v2_phase2"},
    "classical_phase3":  {"n_trials": 300, "study_name": "classical_v2_phase3"},
}


# ============================================================
#  HELPERS
# ============================================================

#: Les cles qu'un SCENARIO_* doit porter pour que `create_argus` reponde.
REQUIRED_SCENARIO_KEYS = (
    "scenario", "N", "max_depth_override", "T_MAX", "T_START", "DT",
    "HYBRID_DT", "K_opt", "Re", "Rm", "shots", "AdvAnomaliesEnable",
)


def create_argus(scenario_config):
    """Build the argus namespace for a given scenario config.

    LEVE si une cle manque, `AdvAnomaliesEnable` comprise. Elle etait lue
    avec `.get(..., False)` : Orszag-Tang, seul scenario a ne pas la
    porter, tournait donc sans anomalies avancees — donc sans terme de
    point X — sans que rien ne le signale. Un repli silencieux sur une
    valeur valide est exactement ce qu'on ne veut pas ici.
    """
    missing = [k for k in REQUIRED_SCENARIO_KEYS if k not in scenario_config]
    if missing:
        raise KeyError(
            f"config de scenario incomplete : cles manquantes {missing}. "
            f"Recue : {sorted(scenario_config)}")
    return SimpleNamespace(
        reps=2,
        mode="simulator",
        backend="state_vector",
        shots=scenario_config["shots"],
        method="COBYLA",
        opt_level=1,
        AdvAnomaliesEnable=scenario_config["AdvAnomaliesEnable"],
        K_opt=scenario_config["K_opt"],
        eps=1e-2,
        seed=0,
        eta=0.001,
        Bz_guide=0.1,
        c_s=1.0,
        Re=scenario_config["Re"],
        Rm=scenario_config["Rm"],
    )


def _get_storage(phase_config):
    """Return the journal shared by the local worker processes."""
    ensure_dirs()
    from optuna.storages.journal import (JournalFileBackend,
                                         JournalFileOpenLock)
    journal_path = os.path.join(
        JOURNAL_DIR, f"{phase_config['study_name']}.log")
    return optuna.storages.JournalStorage(
        JournalFileBackend(
            journal_path, lock_obj=JournalFileOpenLock(journal_path)))


_BUDGET_EXCESS_ATTR = "budget_guard_excess"


def _trial_consumes_budget(trial):
    return (trial.state not in (optuna.trial.TrialState.WAITING,
                                optuna.trial.TrialState.FAIL)
            and not trial.user_attrs.get(_BUDGET_EXCESS_ATTR, False))


def trials_done(study):
    """Essais qui consomment le budget : en cours, termines ou elagues.

    Un essai WAITING est une graine en file d'attente, pas un essai fait.
    Un essai RUNNING est en cours chez un autre worker : il compte, sinon
    N workers demarrant ensemble liraient tous « 0 fait » et lanceraient
    chacun la campagne entiere.

    Un essai FAIL ne consomme pas le budget : la campagne s'arrete sur une
    exception normale, et un essai interrompu est marque FAIL lors de la
    reprise afin que sa place soit recalculee.

    Les essais créés lors d'une course entre workers mais arrêtés avant
    l'objectif coûteux par le garde de budget ne sont pas comptés.
    """
    return sum(_trial_consumes_budget(t) for t in study.trials)


def make_pruner():
    """MedianPruner. `n_warmup_steps=2` : elague au plus tot apres le 3e
    scenario rapporte, jamais avant — un scenario seul ne dit pas assez."""
    return optuna.pruners.MedianPruner(
        n_startup_trials=15,
        n_warmup_steps=2,
        interval_steps=1,
        n_min_trials=5,
    )


def _campaign_contract(phase_name, phase_config, objective_fn):
    contract = {
        "schema": 1,
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "phase": phase_name,
        "study_name": phase_config["study_name"],
        "target_trials": int(phase_config["n_trials"]),
        "objective": getattr(objective_fn, "_qhas_contract", None),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return encoded, digest


def _open_phase_study(phase_name, phase_config, objective_fn, seed=None):
    """Open a study and bind it to the exact campaign contract."""
    sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else None
    study = optuna.create_study(
        study_name=phase_config["study_name"],
        storage=_get_storage(phase_config),
        load_if_exists=True,
        direction="minimize",
        pruner=make_pruner(),
        sampler=sampler,
    )
    contract_json, contract_hash = _campaign_contract(
        phase_name, phase_config, objective_fn)
    previous_hash = study.user_attrs.get("campaign_contract_sha256")
    if previous_hash is None:
        if study.trials:
            raise RuntimeError(
                f"study {study.study_name!r} already contains trials but has "
                "no campaign contract; refusing to mix an unverified run")
        study.set_user_attr("campaign_contract", contract_json)
        study.set_user_attr("campaign_contract_sha256", contract_hash)
    elif previous_hash != contract_hash:
        raise RuntimeError(
            f"campaign contract mismatch for study {study.study_name!r}: "
            f"stored={previous_hash}, requested={contract_hash}. Use a new "
            "campaign directory or resume the original commit and protocol")
    return study, contract_hash


def fail_interrupted_trials(study):
    """Mark trials left RUNNING by a stopped process as retryable failures.

    This must only be called before the worker pool starts.
    """
    running = [trial for trial in study.get_trials(deepcopy=False)
               if trial.state == optuna.trial.TrialState.RUNNING]
    for trial in running:
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    return len(running)


def prepare_phase1(target_trials):
    """Create/validate phase 1, recover interruptions, and queue its seed.

    Le contrat de campagne (`_qhas_contract`) doit avoir EXACTEMENT la
    meme forme que celui que `_run_phase1` ouvrira ensuite pour de vrai —
    `training_regime_grid` compris — sinon `--prepare-only` ecrirait un
    hash que la vraie execution ne pourrait plus retrouver
    (`_open_phase_study` leverait `campaign contract mismatch` au premier
    essai reel). Les valeurs des traces n'ont pas besoin d'etre reelles
    ici : seule la FORME du contrat compte, jamais executee par
    `_open_phase_study`.
    """
    config = dict(PHASES["phase1_composite"])
    config["n_trials"] = int(target_trials)
    placeholders = {key: (None, None) for key, _ in SCENARIOS_ISOLATED}
    placeholders_by_regime = {point: dict(placeholders)
                              for point in TRAINING_REGIME_GRID}
    objective = make_composite_objective(
        None, SCENARIOS_ISOLATED, dns_traces_by_regime=placeholders_by_regime)
    study, _ = _open_phase_study(
        "phase1_composite", config, objective, seed=0)
    recovered = fail_interrupted_trials(study)
    if trials_done(study) == 0 and not any(
            trial.state == optuna.trial.TrialState.WAITING
            for trial in study.get_trials(deepcopy=False)):
        for params in phase1_seeds():
            study.enqueue_trial(params, skip_if_exists=True)
    return study, recovered


def run_phase(phase_name, phase_config, objective_fn, seed_params=None,
              seed=None):
    """Run one phase against a global multi-process trial budget.

    The budget is reread before every trial. `WORKER_TRIALS` is only a
    per-process ceiling; it never changes the global target.
    """
    study, contract_hash = _open_phase_study(
        phase_name, phase_config, objective_fn, seed=seed)

    if seed_params is not None and trials_done(study) == 0:
        for params in seed_params:
            study.enqueue_trial(params, skip_if_exists=True)

    target_trials = phase_config["n_trials"]

    def budgeted_objective(trial):
        # L'allocation du numéro d'essai par le backend est atomique. En cas
        # de course à la dernière place, seul le plus petit numéro restant
        # exécute l'objectif; les autres sont élagués avant tout calcul MHD.
        eligible = sorted(
            (t for t in study.get_trials(deepcopy=False)
             if _trial_consumes_budget(t)),
            key=lambda t: t.number,
        )
        slot = next(i for i, t in enumerate(eligible, start=1)
                    if t.number == trial.number)
        trial.set_user_attr("campaign_budget_slot", slot)
        trial.set_user_attr("worker_seed", seed)
        trial.set_user_attr("campaign_contract_sha256", contract_hash)
        if slot > target_trials:
            trial.set_user_attr(_BUDGET_EXCESS_ATTR, True)
            raise optuna.TrialPruned(
                f"essai excedentaire cree par concurrence (slot {slot}, "
                f"cible {target_trials})")
        return objective_fn(trial)

    done_at_start = trials_done(study)
    print(f"Phase '{phase_name}' : {done_at_start}/{target_trials} essais deja "
          f"dans la base"
          + (f", plafond de {WORKER_TRIALS} pour ce worker." if WORKER_TRIALS
             else "."))

    by_this_worker = 0
    while trials_done(study) < target_trials:
        if WORKER_TRIALS is not None and by_this_worker >= WORKER_TRIALS:
            print(f"Plafond du worker atteint ({WORKER_TRIALS} essais).")
            break
        study.optimize(budgeted_objective, n_trials=1)
        by_this_worker += 1

    print(f"Phase '{phase_name}' : {trials_done(study)}/{target_trials} essais "
          f"au total, {by_this_worker} faits par ce worker.")
    return study


def extract_top_params(study, top_k=10):
    """Extract the top_k best parameter sets from a completed study."""
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value < float("inf")
    ]
    completed.sort(key=lambda t: t.value)
    return [t.params for t in completed[:top_k]]


def extract_top_params_from_rescore(phase_prefix, lambdas, top_k=2):
    """Read the best distinct parameter sets from rescore CSV files."""
    seen_trials = set()
    results = []

    for lam in lambdas:
        lam_str = f"{lam:.4f}"
        csv_path = os.path.join(
            data_dir, f"rescore_{phase_prefix}_lambda{lam_str}",
            f"trials_lambda{lam_str}.csv"
        )
        if not os.path.exists(csv_path):
            print(f"[WARN] Rescore CSV not found: {csv_path}")
            continue

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = sorted(reader, key=lambda r: float(r["new_score"]))

        count = 0
        for row in rows:
            if count >= top_k:
                break
            trial_id = int(row["trial"])
            if trial_id in seen_trials:
                continue
            seen_trials.add(trial_id)

            params = {}
            for col in row:
                if col.startswith("param_"):
                    key = col[len("param_"):]
                    params[key] = float(row[col])
            results.append(params)
            count += 1
            print(f"  Seed from rescore λ={lam_str}: trial #{trial_id} "
                  f"(score={float(row['new_score']):.6f})")

    print(f"[RESCORE SEEDING] Extracted {len(results)} unique trials "
          f"from {len(lambdas)} lambda values")
    return results


# ============================================================
#  COMPOSITE MULTI-SCENARIO OBJECTIVE (QAOA)
# ============================================================

# ══════════════════════════════════════════════════════════════════════
#  L'espace de recherche, declare — pas devine en relisant la base
# ══════════════════════════════════════════════════════════════════════
#
# Les bornes etaient ecrites en dur a l'interieur de l'objectif, sous la
# forme `if "x" not in frozen: HyperParams["x"] = <constante>`, qui fait
# passer une constante pour un parametre conditionnel. Quatre valeurs
# etaient dans ce cas ; la campagne gelee croyait explorer neuf
# parametres et en explorait cinq. C'est l'origine de D-22 : trois des
# valeurs deployees n'ont jamais ete echantillonnees par personne.
#
# Ici les bornes sont des donnees. `search_space()` les rend lisibles
# AVANT de louer des coeurs pour une semaine, et un test verifie que ce
# qu'Optuna a reellement propose coincide avec cette declaration.

#: Meilleur essai de l'etude classique gelee (#42, perte 0.2148).
CLASSICAL_BEST_THRESHOLD = 0.14959824837662078

#: nom -> (bas, haut, log). Les 8 parametres du perimetre de
#: reoptimisation.
SEARCH_SPACE = {
    # Encodage : raideur de la sigmoide score -> angle.
    "beta":        (0.5,  10.0,  False),
    # Poids adaptatif du biais Z, en fraction de la mediane des couplages.
    # La borne haute (1000) n'est PAS une fraction ; elle vient de la
    # campagne gelee, dont la graine valait 500. Conservee telle quelle
    # pour ne pas changer la science en meme temps que le code — mais
    # signalee : voir docs/DEFAUTS.md.
    "w_z_frac":    (0.10, 1000.0, True),
    # Largeur de la fenetre d'incertitude autour de threshold_amr.
    "sigma":       (0.02, 0.30,  False),
    # Sensibilites par type de terme (0.0 desactive le terme).
    "beta_curl":   (0.0,  5.0,   False),
    "beta_xpoint": (0.0,  5.0,   False),
    # Croissance logarithmique des portes f au-dela du regime critique.
    "gamma_hydro": (0.1,  5.0,   False),
    "gamma_mag":   (0.1,  5.0,   False),
    # Raideur des sigmoides fuyantes g. Echelle log : kappa -> 0 rend
    # toutes les portes egales a 0.5, kappa grand en fait des marches.
    # `g_strain + g_rot == 1` exactement : kappa ne pilote qu'UN degre de
    # liberte, pas deux.
    "kappa":       (0.5,  50.0,  True),
    # Percentile du critere RELATIF (`PhysicalMapper._effective_crit`).
    # Le seuil effectif vaut `min(absolu, percentile(signal))` : ce
    # parametre ne pilote QUE le regime ou aucune cellule n'atteint le
    # critere physique de Reynolds-maille — c'est-a-dire la resolution
    # d'entrainement N=256, ou le seuil absolu croit en 1/dx^2 et efface
    # tout. Sans dimension : c'est un rang dans la distribution du champ
    # courant, pas une amplitude.
    #
    # Bornes. 50 = la mediane : la moitie des cellules passent le seuil,
    # ce qui sature l'AMR — deja une borne large. 99 = une cellule sur
    # cent, en dessous du grain d'un patch (min_patch_size=6, soit 36
    # cellules) : au-dela, le critere relatif ne designerait plus assez
    # de cellules pour former un patch et redeviendrait le seuil absolu
    # par un autre chemin. Echelle lineaire.
    "relative_percentile": (50.0, 99.0, False),
}

#: Ce que l'objectif FIXE, avec sa valeur — donc ce qu'aucune campagne ne
#: peut decouvrir, quelle que soit sa duree.
FIXED_PARAMS = {
    # Gele au meilleur essai classique pour que la comparaison porte sur
    # ce que le quantique ajoute, et non sur un seuil different.
    "threshold_amr": CLASSICAL_BEST_THRESHOLD,
}


def search_space(names_only=True):
    """Les parametres que `make_composite_objective` proposera reellement.

    Une campagne qui croit optimiser `kappa` doit pouvoir le verifier
    avant de lancer, plutot que le decouvrir en relisant la base a
    posteriori.
    """
    if names_only:
        return tuple(SEARCH_SPACE)
    return dict(SEARCH_SPACE)


def suggest_hyperparams(trial, frozen=None, tune_threshold=False):
    """Propose les parametres de `SEARCH_SPACE` a Optuna et renvoie le
    dictionnaire COMPLET.

    Complet veut dire : parametres explores + parametres fixes + parametres
    geles par l'appelant. C'est ce dictionnaire-la, et non
    `trial.params`, qui decrit le run — `trial.params` ne contient que ce
    qui a ete echantillonne, ce qui est exactement pourquoi le JSON
    deploye a perdu `sigma` et invente `gamma_hydro`.
    """
    frozen = frozen or {}
    hp = {}
    for name, (lo, hi, log) in SEARCH_SPACE.items():
        if name in frozen:
            continue
        hp[name] = trial.suggest_float(name, lo, hi, log=log)
    for name, value in FIXED_PARAMS.items():
        if name not in frozen and not (tune_threshold
                                       and name == "threshold_amr"):
            hp[name] = value
    if tune_threshold and "threshold_amr" not in frozen:
        lo, hi = CLASSICAL_THRESHOLD_RANGE
        hp["threshold_amr"] = trial.suggest_float("threshold_amr", lo, hi)
    hp.update(frozen)
    return hp


def _run_one_scenario(trial, scenario_key, scenario_config, dns_traces,
                      hyperparams, lambda_cost, classical_only=False):
    """Un scenario, une sous-perte. Renvoie la perte finie a ajouter.

    Une exception coute `DIVERGENCE_PENALTY` — penalite FINIE, pour
    qu'Optuna puisse continuer a modeliser l'espace au lieu de recevoir un
    `inf` qui n'ordonne rien. La valeur est importee de `pipeline` plutot
    que recopiee : elle y etait deja definie quatre fois, dont trois dans
    des portees qui masquaient la premiere.
    """
    dns_trace, hot_start_state = dns_traces[scenario_key]
    DT = scenario_config["DT"]
    argus = create_argus(scenario_config)

    try:
        result = pipeline(
            N=scenario_config["N"], VQA_N=VQA_N_TRAINING,
            T_MAX=scenario_config["T_MAX"], DT=DT,
            HYBRID=int(scenario_config["HYBRID_DT"] / DT),
            verbose=False, argus=argus, hyperparams=hyperparams,
            lambda_cost=lambda_cost, trial=None,  # elagage gere ici, pas dedans
            dns_trace=dns_trace, hot_start_state=hot_start_state,
            min_patch_size=scenario_config.get("min_patch_size", 6),
            max_depth_override=scenario_config.get("max_depth_override", None),
            scenario=scenario_config["scenario"],
            return_details=True,
            classical_only=classical_only,
        )
    except Exception as e:
        print(f"[Trial {trial.number}] FAILED on {scenario_key}: {e}")
        import traceback
        traceback.print_exc()
        trial.set_user_attr(f"phys_{scenario_key}", DIVERGENCE_PENALTY)
        trial.set_user_attr(f"patch_{scenario_key}", 1.0)
        return DIVERGENCE_PENALTY

    if isinstance(result, dict) and result.get("completed") is False:
        trial.set_user_attr(f"completed_{scenario_key}", False)
        trial.set_user_attr(f"abort_{scenario_key}", result.get("abort"))
        trial.set_user_attr(f"phys_{scenario_key}", DIVERGENCE_PENALTY)
        trial.set_user_attr(f"patch_{scenario_key}", 1.0)
        return DIVERGENCE_PENALTY

    combined = result['combined'] if isinstance(result, dict) else result
    if np.isnan(combined) or np.isinf(combined):
        combined = DIVERGENCE_PENALTY

    if isinstance(result, dict):
        trial.set_user_attr(f"phys_{scenario_key}", float(result.get('phys_score', 0)))
        trial.set_user_attr(f"patch_{scenario_key}", float(result.get('patch_ratio', 0)))
        for field, err in result.get('field_errors', {}).items():
            trial.set_user_attr(f"error_{field}_{scenario_key}", float(err))
        # D-22 / D-35 : d'ou vient sigma, essai par essai. Un artefact ne
        # doit jamais laisser croire qu'une valeur vient de l'entrainement
        # alors qu'elle vient d'un repli.
        if result.get('sigma_source') is not None:
            trial.set_user_attr(f"sigma_source_{scenario_key}",
                                result['sigma_source'])
    return float(combined)


def _composite_loop(trial, scenario_list, dns_traces, hyperparams,
                    lambda_cost, classical_only=False):
    """Average scenario losses and report each fixed-order partial mean."""
    sub_losses = {}
    total = 0.0
    for i, (scenario_key, scenario_config) in enumerate(scenario_list):
        loss = _run_one_scenario(trial, scenario_key, scenario_config,
                                 dns_traces, hyperparams, lambda_cost,
                                 classical_only=classical_only)
        sub_losses[scenario_key] = loss
        total += loss
        trial.report(total / (i + 1), step=i)
        if trial.should_prune():
            for key, value in sub_losses.items():
                trial.set_user_attr(f"loss_{key}", float(value))
            raise optuna.TrialPruned()

    for key, value in sub_losses.items():
        trial.set_user_attr(f"loss_{key}", float(value))
    return total / len(scenario_list)


def make_composite_objective(dns_traces, scenario_list,
                             frozen_params=None,
                             lambda_cost=LAMBDA_COST_SOFT,
                             tune_threshold=False,
                             dns_traces_by_regime=None,
                             training_regime_grid=None):
    """
    Composite loss across a set of scenarios (QAOA method).

    Loss = mean(Loss_i) over all scenarios in scenario_list.

    Les memes hyperparametres traversent TOUS les scenarios : l'optimiseur
    doit donc trouver un reglage qui vaut pour tous les types d'anomalie,
    pas un par scenario.

    Parameters
    ----------
    dns_traces : dict scenario_key -> (dns_trace, hot_start_state). Utilise
        tel quel si `dns_traces_by_regime` est None ; peut alors etre None
        (seul le controle de doublons de `_assert_scenarios_wellformed`
        s'applique encore).
    scenario_list : liste de (key, config) — quels scenarios composent la
        perte. L'ORDRE compte : il definit les steps de l'elagage.
    frozen_params : hyperparametres imposes par l'appelant, retires de
        l'espace de recherche.
    dns_traces_by_regime : dict (Re, graine) -> dns_traces (meme forme que
        `dns_traces`, un jeu complet par point). Si fourni, DIVERSIFICATION
        D'ENTRAINEMENT (USER, 26 aout) : chaque essai tire un regime
        physique dans `training_regime_grid`, INDEPENDAMMENT du sampler
        Optuna (`_training_regime_for_trial`), et s'entraine sur ce regime
        plutot que sur `dns_traces` seul. Le regime tire est ecrit dans
        `trial.user_attrs["training_regime"]`.
    training_regime_grid : damier a tirer si `dns_traces_by_regime` est
        fourni. None -> `TRAINING_REGIME_GRID`.
    """
    frozen = frozen_params or {}
    grid = training_regime_grid if training_regime_grid is not None \
        else TRAINING_REGIME_GRID
    _assert_scenarios_wellformed(scenario_list, dns_traces)
    if dns_traces_by_regime is not None:
        _assert_regime_traces_wellformed(scenario_list, dns_traces_by_regime, grid)

    def objective(trial):
        hyperparams = suggest_hyperparams(
            trial, frozen, tune_threshold=tune_threshold)
        # Le dictionnaire complet est attache a l'essai : c'est la seule
        # trace qui permette de redeployer un essai sans le reconstruire
        # a la main. `trial.params` ne porte que l'echantillonne.
        trial.set_user_attr("hyperparams_resolved", hyperparams)
        if dns_traces_by_regime is not None:
            re, phys_seed = _training_regime_for_trial(trial.number, grid)
            trial.set_user_attr("training_regime", f"Re={re}_seed={phys_seed}")
            regime_scenarios = [(k, _with_physical_regime(cfg, re, phys_seed))
                                for k, cfg in scenario_list]
            return _composite_loop(trial, regime_scenarios,
                                   dns_traces_by_regime[(re, phys_seed)],
                                   hyperparams, lambda_cost)
        return _composite_loop(trial, scenario_list, dns_traces, hyperparams,
                               lambda_cost)

    objective._qhas_contract = {
        "kind": "qaoa_composite",
        "lambda_cost": float(lambda_cost),
        "scenarios": [
            {"key": key, "config": dict(config)}
            for key, config in scenario_list
        ],
        "frozen_params": dict(frozen),
        "search_space": {
            key: {"low": low, "high": high, "log": log_scale}
            for key, (low, high, log_scale) in SEARCH_SPACE.items()
        },
        "fixed_params": ({k: v for k, v in FIXED_PARAMS.items()
                          if not (tune_threshold and k == "threshold_amr")}),
        "tune_threshold": bool(tune_threshold),
        "training_regime_grid": ([list(point) for point in grid]
                                 if dns_traces_by_regime is not None else None),
    }
    if tune_threshold:
        lo, hi = CLASSICAL_THRESHOLD_RANGE
        objective._qhas_contract["search_space"]["threshold_amr"] = {
            "low": lo, "high": hi, "log": False,
        }
    return objective


# ============================================================
#  CLASSICAL COMPOSITE OBJECTIVE
# ============================================================

def make_classical_composite_objective(dns_traces, scenario_list,
                                       lambda_cost=LAMBDA_COST_SOFT,
                                       dns_traces_by_regime=None,
                                       training_regime_grid=None):
    """
    Composite loss across scenarios for the CLASSICAL AMR method.

    N'entraine que `threshold_amr` (1 parametre) : pas de circuit, donc
    ~100x plus rapide par essai. C'est le bras de comparaison — meme
    perte, memes scenarios, meme agregation que le bras QAOA, seul le
    critere de raffinement change. `dns_traces_by_regime`/
    `training_regime_grid` : meme diversification d'entrainement que
    `make_composite_objective` (le bras classique doit tirer les MEMES
    regimes, pour que la comparaison reste equitable — voir
    `_training_regime_for_trial`, fonction du seul numero d'essai).

    Parameters
    ----------
    dns_traces : dict scenario_key -> (dns_trace, hot_start_state). Peut
        etre None si `dns_traces_by_regime` est fourni.
    scenario_list : liste de (key, config), meme contrat que ci-dessus.
    """
    grid = training_regime_grid if training_regime_grid is not None \
        else TRAINING_REGIME_GRID
    _assert_scenarios_wellformed(scenario_list, dns_traces)
    if dns_traces_by_regime is not None:
        _assert_regime_traces_wellformed(scenario_list, dns_traces_by_regime, grid)

    def objective(trial):
        lo, hi = CLASSICAL_THRESHOLD_RANGE
        hyperparams = {"threshold_amr": trial.suggest_float("threshold_amr", lo, hi)}
        trial.set_user_attr("hyperparams_resolved", hyperparams)
        if dns_traces_by_regime is not None:
            re, phys_seed = _training_regime_for_trial(trial.number, grid)
            trial.set_user_attr("training_regime", f"Re={re}_seed={phys_seed}")
            regime_scenarios = [(k, _with_physical_regime(cfg, re, phys_seed))
                                for k, cfg in scenario_list]
            return _composite_loop(trial, regime_scenarios,
                                   dns_traces_by_regime[(re, phys_seed)],
                                   hyperparams, lambda_cost, classical_only=True)
        return _composite_loop(trial, scenario_list, dns_traces, hyperparams,
                               lambda_cost, classical_only=True)

    objective._qhas_contract = {
        "kind": "classical_composite",
        "lambda_cost": float(lambda_cost),
        "scenarios": [
            {"key": key, "config": dict(config)}
            for key, config in scenario_list
        ],
        "threshold_range": list(CLASSICAL_THRESHOLD_RANGE),
        "training_regime_grid": ([list(point) for point in grid]
                                 if dns_traces_by_regime is not None else None),
    }
    return objective


#: Bornes du seuil classique. `CLASSICAL_BEST_THRESHOLD` doit tomber
#: dedans, sinon le bras quantique est gele sur une valeur que le bras
#: classique n'avait pas le droit de proposer — un test le verifie.
CLASSICAL_THRESHOLD_RANGE = (0.05, 0.8)


# ============================================================
#  ORCHESTRATION
# ============================================================
#
# Six scénarios isolent les structures ciblées; OT et le rotor combinent
# plusieurs mécanismes. Chaque scénario apparaît exactement une fois.
SCENARIOS_ISOLATED = (
    ("kh",          SCENARIO_KH),
    ("vortex",      SCENARIO_VORTEX),
    ("tearing",     SCENARIO_TEARING),
    ("coalescence", SCENARIO_COALESCENCE),
    ("double_tearing", SCENARIO_DOUBLE_TEARING),
    ("magnetic_twist", SCENARIO_MAGNETIC_TWIST),
)

SCENARIOS_COMPLEX = (
    ("ot",    SCENARIO_OT),
    ("rotor", SCENARIO_ROTOR),
)

SCENARIOS_ALL = SCENARIOS_ISOLATED + SCENARIOS_COMPLEX


def _assert_scenarios_wellformed(scenario_list, dns_traces=None):
    """Un jeu de scenarios sans doublon, et dont chaque trace existe.

    Un doublon ne leve rien a l'execution : il double une ponderation et
    un cout, en silence. Un scenario absent de `dns_traces` leverait un
    KeyError au milieu du premier essai — c'est-a-dire apres le
    pre-calcul DNS, soit des dizaines de minutes trop tard.
    """
    if not scenario_list:
        raise ValueError("jeu de scenarios vide")
    keys = [k for k, _ in scenario_list]
    if len(keys) != len(set(keys)):
        dups = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(
            f"scenarios en double {dups} : chacun serait simule et compte "
            f"deux fois. Recu : {keys}")
    if dns_traces is not None:
        missing = [k for k in keys if k not in dns_traces]
        if missing:
            raise KeyError(f"traces DNS absentes pour {missing}")



def _precompute_dns_for(scenario_list, label="scenarios"):
    """Pre-compute DNS traces for a list of (key, config) scenarios."""
    _assert_scenarios_wellformed(scenario_list)
    print(f"\n--- Pre-computing DNS traces for {label} ---")
    dns_traces = {}
    for key, config in scenario_list:
        config_with_name = {**config, "study_name": f"dns_{key}"}
        dns_trace, hot_start = precompute_dns(config_with_name)
        dns_traces[key] = (dns_trace, hot_start)
    print(f"--- DNS pre-computation complete ({label}) ---\n")
    return dns_traces


# ============================================================
#  DIVERSIFICATION DE L'ENTRAINEMENT (USER, 26 aout)
# ============================================================
#
# Les 3 phases entrainaient toutes a Re=Rm=800, graine physique 0
# implicite : un seul regime physique, jamais varie d'un essai a
# l'autre. `select_by_holdout_validation` (D-199) protege la SELECTION
# finale avec un damier tenu a l'ecart (`HOLDOUT_GRID`), mais rien ne
# diversifiait encore la boucle d'ENTRAINEMENT elle-meme — les
# 600+600+400 essais Optuna. Demande USER, 26 aout, apres avoir vu le
# damier de validation : « ok mais moi je veux quand meme une campagne
# plus diversifiee. »
#
# Option ecartee : evaluer CHAQUE essai sur TOUS les regimes d'un damier
# (multiplie le cout par essai par la taille du damier, sur une campagne
# deja de l'ordre de 2000 h CPU a un seul regime — c'est l'option
# explicitement pesee et rejetee dans `RESULTS.md`, section
# "Refonte train/val de la campagne", pour `select_by_holdout_validation`).
#
# Option retenue : chaque essai tire UN SEUL regime dans
# `TRAINING_REGIME_GRID`, choisi par une fonction PURE de son numero
# d'essai (`_training_regime_for_trial`) — jamais par le sampler TPE
# d'Optuna. Cout par essai INCHANGE (toujours 6, 2 ou 8 scenarios
# simules, jamais plus) ; seul le PRECALCUL DNS, fait une fois par phase
# et non par essai, est multiplie par `len(TRAINING_REGIME_GRID)`. Sur
# des centaines d'essais par phase, cette multiplication reste
# negligeable face au cout de la recherche Optuna elle-meme.
#
# Effet de bord assume, pas cache : `run_phase` elague par
# `MedianPruner`, qui compare la perte intermediaire des essais AU MEME
# STEP. Deux essais sur des regimes differents ne sont plus strictement
# comparables a mi-parcours — un essai sur un regime plus difficile peut
# sembler pire qu'un bon reglage ne l'est vraiment. Accepte comme le cout
# normal d'un entrainement diversifie (l'equivalent, pour ce depot, du
# bruit qu'introduit un batch different a chaque epoque d'un reseau de
# neurones) — pas mesure ici, a surveiller sur la vraie campagne.

#: Re=600/800/1000 (Re=800 deux fois, pour garder un pied dans le regime
#: historique) x quatre graines physiques disjointes de celles du damier
#: de validation (`HOLDOUT_GRID` : graines 1/2/3) — verifie par
#: `test_training_and_holdout_grids_never_share_a_point`. Une COINCIDENCE
#: entre les deux damiers rendrait la validation tenue a l'ecart
#: circulaire : elle jugerait un regime que l'entrainement a deja vu.
TRAINING_REGIME_GRID = (
    (800, 0),
    (600, 10),
    (1000, 20),
    (800, 30),
)


def _training_regime_for_trial(trial_number, grid=TRAINING_REGIME_GRID):
    """Le regime physique (Re, graine) d'un essai : fonction PURE de son
    numero, reproductible entre reprises d'une meme etude Optuna (le
    numero d'essai est stable, le journal Optuna le persiste). Jamais
    tire par le sampler TPE : s'il voyait ce choix, il pourrait apprendre
    a preferer les regimes les plus faciles plutot que les
    hyperparametres qui generalisent — l'inverse de ce qu'on veut."""
    return grid[random.Random(trial_number).randrange(len(grid))]


def _assert_regime_traces_wellformed(scenario_list, dns_traces_by_regime, grid):
    """Chaque point du damier d'entrainement doit porter une trace pour
    CHAQUE scenario. Verifie a la CONSTRUCTION de l'objectif, pas
    decouvert au milieu d'un essai qui tire ce point-la — potentiellement
    des heures apres le debut de la campagne."""
    keys = {k for k, _ in scenario_list}
    for point in grid:
        if point not in dns_traces_by_regime:
            raise KeyError(
                f"regime {point} absent de dns_traces_by_regime "
                f"(damier {grid})")
        missing = keys - set(dns_traces_by_regime[point])
        if missing:
            raise KeyError(
                f"regime {point} : traces DNS manquantes pour {sorted(missing)}")


def _precompute_dns_by_regime(scenario_list, grid=TRAINING_REGIME_GRID,
                              label="scenarios"):
    """Precalcule les traces DNS de `scenario_list` a CHAQUE point de
    `grid`. Reutilise `_precompute_dns_for` (donc `precompute_dns`) tel
    quel, un point a la fois — aucune physique reimplementee, et les
    tests qui simulent deja `_precompute_dns_for` couvrent ce chemin
    sans modification."""
    return {
        (re, phys_seed): _precompute_dns_for(
            [(k, _with_physical_regime(cfg, re, phys_seed))
             for k, cfg in scenario_list],
            label=f"{label} (regime Re={re}, graine={phys_seed})")
        for re, phys_seed in grid
    }


def _report_best(study, phase_label, scenario_list):
    """Meilleur essai et sous-pertes. Ne LEVE pas si rien n'a abouti."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"\n{phase_label} — aucun essai termine, pas de meilleur essai.")
        return
    print(f"\n{phase_label} — Best composite score: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    for key, _ in scenario_list:
        loss_key = f"loss_{key}"
        if loss_key in study.best_trial.user_attrs:
            print(f"  {key:>16}: {study.best_trial.user_attrs[loss_key]:.6f}")


#: Grille de depart de la phase 1. Une seule combinaison : ce n'est pas
#: une recherche par grille, c'est la graine que la campagne gelee a
#: utilisee, conservee pour que la reprise parte du meme point.
PHASE1_SEED_GRID = {
    "beta":        [0.7],
    "sigma":       [0.10],
    "beta_curl":   [0.1],
    "beta_xpoint": [0.1],
    "w_z_frac":    [500.0],
    "gamma_hydro": [2.0],
    "gamma_mag":   [0.5],
    "kappa":       [10.0],
    # La campagne gelee ne connaissait pas ce parametre : la graine vaut
    # `PhysicalMapper.RELATIVE_PERCENTILE`, de sorte que le premier essai
    # reproduise EXACTEMENT le comportement actuel et que tout ecart
    # mesure ensuite vienne de l'exploration, pas du point de depart.
    "relative_percentile": [90.0],
}


def phase1_seeds():
    """Produit du grille cartesienne de `PHASE1_SEED_GRID`.

    Chaque graine ne porte QUE des noms de `SEARCH_SPACE` : Optuna ignore
    en silence une cle enfilee qui ne correspond a aucune distribution,
    et une graine entierement ignoree ressemble a une graine appliquee.
    """
    names = list(PHASE1_SEED_GRID)
    unknown = [n for n in names if n not in SEARCH_SPACE]
    if unknown:
        raise KeyError(f"graines pour des parametres hors espace : {unknown}")
    return [dict(zip(names, values))
            for values in itertools.product(*(PHASE1_SEED_GRID[n] for n in names))]


def _run_phase1(dns_traces, seed=None, n_trials=None, dns_traces_by_regime=None):
    """Phase 1 : perte composite sur les 6 scenarios isoles.

    Le nombre de parametres n'est pas ecrit ici : il vaut
    `len(SEARCH_SPACE)`, et la ligne « Training: » ci-dessous
    l'imprime. Une valeur figee dans une docstring se serait
    desynchronisee au premier ajout — elle l'avait deja fait.
    """
    print("=" * 60)
    print("PHASE 1: Composite Training (6 scenarios isoles)")
    print(f"  Training: {', '.join(search_space())}")
    print(f"  Fixed:    {FIXED_PARAMS}")
    print(f"  Scenarios: {', '.join(k for k, _ in SCENARIOS_ISOLATED)}")
    print("=" * 60)

    objective = make_composite_objective(
        dns_traces, SCENARIOS_ISOLATED, dns_traces_by_regime=dns_traces_by_regime)
    phase_config = dict(PHASES["phase1_composite"])
    if n_trials is not None:
        phase_config["n_trials"] = n_trials
    study = run_phase("phase1_composite", phase_config,
                      objective, seed_params=phase1_seeds(), seed=seed)
    _report_best(study, "Phase 1", SCENARIOS_ISOLATED)
    return study


RESCORE_LAMBDAS_SOFT = [0.4]


def _seeds_for(phase_prefix, fallback_study, top_k, label):
    """Graines : CSV de rescore si presents, sinon meilleurs essais Optuna.

    Renvoie une liste eventuellement vide — une phase peut demarrer sans
    graine, c'est seulement moins efficace. Ce qui n'est PAS acceptable,
    c'est de croire avoir amorce alors que rien ne l'a ete : d'ou le
    message explicite dans les deux branches.
    """
    seeds = extract_top_params_from_rescore(phase_prefix, RESCORE_LAMBDAS_SOFT,
                                            top_k=top_k)
    if seeds:
        print(f"\n[{label}] Amorcage par les CSV de rescore : {len(seeds)} graines")
        return seeds
    print(f"[{label}] Aucun CSV de rescore, repli sur l'etude Optuna")
    seeds = extract_top_params(fallback_study, top_k=top_k)
    if not seeds:
        print(f"[{label}] Aucun essai n'a abouti en amont : demarrage sans graine")
    return seeds


def _run_phase2(study_p1, seed=None):
    """Phase 2 : les 2 scenarios complexes, amorcee par la phase 1."""
    print("\n" + "=" * 60)
    print("PHASE 2: Composite Training sur Orszag-Tang + rotor")
    print(f"  Training: {', '.join(search_space())}")
    print("=" * 60)

    dns_traces_by_regime = _precompute_dns_by_regime(SCENARIOS_COMPLEX,
                                                     label="OT + Rotor")
    seed_params = _seeds_for("q_has_v2_phase1", study_p1, 20, "PHASE 2")

    objective = make_composite_objective(
        None, SCENARIOS_COMPLEX, dns_traces_by_regime=dns_traces_by_regime)
    study = run_phase("phase2_complex", PHASES["phase2_complex"],
                      objective, seed_params=seed_params, seed=seed)
    _report_best(study, "Phase 2", SCENARIOS_COMPLEX)
    return study


def _run_phase3(study_p2, seed=None):
    """Phase 3 : les 8 scenarios distincts, amorcee par la phase 2."""
    print("\n" + "=" * 60)
    print("PHASE 3: Composite Training sur les 8 scenarios")
    print(f"  Scenarios: {', '.join(k for k, _ in SCENARIOS_ALL)}")
    print("=" * 60)

    dns_traces_by_regime = _precompute_dns_by_regime(SCENARIOS_ALL,
                                                     label="8 scenarios")
    seed_params = _seeds_for("q_has_v2_phase2", study_p2, 15, "PHASE 3")

    objective = make_composite_objective(
        None, SCENARIOS_ALL, dns_traces_by_regime=dns_traces_by_regime)
    study = run_phase("phase3_validation", PHASES["phase3_validation"],
                      objective, seed_params=seed_params, seed=seed)
    _report_best(study, "Phase 3", SCENARIOS_ALL)
    return study


def _classical_grid_seeds(n=20):
    lo, hi = CLASSICAL_THRESHOLD_RANGE
    return [{"threshold_amr": t} for t in np.linspace(lo, hi, n).tolist()]


def _run_classical_phase1(dns_traces, seed=None, dns_traces_by_regime=None):
    """Classique 1 : `threshold_amr` sur les 6 scenarios isoles."""
    print("\n" + "=" * 60)
    print("CLASSICAL PHASE 1 (*): threshold_amr, 6 scenarios isoles")
    print("=" * 60)

    objective = make_classical_composite_objective(
        dns_traces, SCENARIOS_ISOLATED, dns_traces_by_regime=dns_traces_by_regime)
    study = run_phase("classical_phase1", PHASES["classical_phase1"],
                      objective, seed_params=_classical_grid_seeds(20), seed=seed)
    _report_best(study, "Classical Phase 1", SCENARIOS_ISOLATED)
    return study


def _run_classical_phase2(study_c1, seed=None):
    """Classique 2 : `threshold_amr` sur OT + rotor."""
    print("\n" + "=" * 60)
    print("CLASSICAL PHASE 2 (**): threshold_amr, OT + rotor")
    print("=" * 60)

    dns_traces_by_regime = _precompute_dns_by_regime(
        SCENARIOS_COMPLEX, label="OT + Rotor (classique)")
    seeds = _seeds_for("classical_v2_phase1", study_c1, 15, "CLASSICAL PHASE 2")
    if not seeds:
        seeds = _classical_grid_seeds(15)

    objective = make_classical_composite_objective(
        None, SCENARIOS_COMPLEX, dns_traces_by_regime=dns_traces_by_regime)
    study = run_phase("classical_phase2", PHASES["classical_phase2"],
                      objective, seed_params=seeds, seed=seed)
    _report_best(study, "Classical Phase 2", SCENARIOS_COMPLEX)
    return study


def _run_classical_phase3(study_c2, seed=None):
    """Classique 3 : `threshold_amr` sur les 8 scenarios."""
    print("\n" + "=" * 60)
    print("CLASSICAL PHASE 3 (***): threshold_amr, 8 scenarios")
    print("=" * 60)

    dns_traces_by_regime = _precompute_dns_by_regime(
        SCENARIOS_ALL, label="8 scenarios (classique)")
    seeds = _seeds_for("classical_v2_phase2", study_c2, 15, "CLASSICAL PHASE 3")
    if not seeds:
        seeds = _classical_grid_seeds(15)

    objective = make_classical_composite_objective(
        None, SCENARIOS_ALL, dns_traces_by_regime=dns_traces_by_regime)
    study = run_phase("classical_phase3", PHASES["classical_phase3"],
                      objective, seed_params=seeds, seed=seed)
    _report_best(study, "Classical Phase 3", SCENARIOS_ALL)
    return study


# ============================================================
#  SELECTION PAR VALIDATION TENUE A L'ECART (refonte train/val, USER 26 aout)
# ============================================================
#
# Les 3 phases ci-dessus choisissent TOUJOURS `best_params` par score EN
# ECHANTILLON : la phase 3 evalue Optuna sur les 8 scenarios eux-memes, a
# Re=800 et une seule graine physique implicite (`phys_seed=0` partout) —
# rien n'est jamais tenu a l'ecart de la selection. Le deploiement final
# peut donc surapprendre a CETTE configuration physique precise, le
# risque qu'on nommerait sans hesiter pour un modele de machine learning
# classique (c'est la question posee par D-198 : le plafond GBT en aval
# souffre du meme genre de piege). `precompute_dns` accepte deja
# `phys_seed`/`Re`/`Rm` par scenario (voir `Simulation/pre_compute_dns.py`)
# — l'infrastructure existe, seuls les `SCENARIO_*` ci-dessus ne
# variaient jamais ces trois cles.
#
# Cette section ajoute UNE selection finale, tenue a l'ecart des 3
# phases : parmi les `HOLDOUT_TOP_K` meilleurs essais EN ECHANTILLON de
# la phase 3, le gagnant est celui qui a la MEILLEURE perte MOYENNE sur
# un DAMIER de regimes physiques jamais vus par aucune phase — plusieurs
# Re et plusieurs graines, memes 8 scenarios. Cout : re-evaluer
# `HOLDOUT_TOP_K` jeux de parametres deja tires (aucune nouvelle
# recherche Optuna), sur des traces DNS precalculees UNE SEULE FOIS par
# point du damier. De l'ordre de `HOLDOUT_TOP_K * len(HOLDOUT_GRID)`
# essais equivalents, une fraction de la campagne (~600+600+400 essais
# Optuna) — pas une nouvelle campagne.
#
# Mis a jour le 26 aout — un SEUL point tenu a l'ecart (Re=1200,
# graine=1) restait lui-meme surapprenable : un candidat pouvait gagner
# la validation en etant bon PRECISEMENT sur ce point, sans que rien ne
# le distingue d'un candidat robuste sur l'ensemble du domaine. Demande
# USER, 26 aout : « il faudrait que cette campagne ait plusieurs
# parametrisations physiques et plusieurs graines ». `HOLDOUT_GRID`
# remplace le point unique par un damier ; le classement se fait sur la
# MOYENNE du damier (voir `select_by_holdout_validation`), le pire point
# restant journalise a cote pour diagnostic.
#
# Portee assumee, pas un oubli : seule la selection FINALE (phase 3, les
# deux bras) est protegee. Les cascades d'amorcage phase1->phase2 et
# phase2->phase3 restent en-echantillon — etendre plus loin est possible
# mais multiplie le cout de validation par le nombre de phases amorcees.

#: Deux Re (sous et sur le point d'entrainement Re=Rm=800) x trois
#: graines physiques : la demande USER porte sur les DEUX axes, pas un
#: seul. Aucun point ne doit coincider avec le regime d'entrainement
#: (Re=800, graine implicite 0), sinon ce n'est pas une validation tenue
#: a l'ecart — verifie par
#: `test_holdout_grid_varies_both_re_and_physical_seed`.
HOLDOUT_GRID = (
    (400, 1), (400, 2), (400, 3),
    (1200, 1), (1200, 2), (1200, 3),
)
HOLDOUT_TOP_K = 15


def _with_physical_regime(base_config, re, phys_seed):
    """Variante d'un config `SCENARIO_*` a un point d'un damier physique
    (entrainement ou validation) : meme scenario et memes reglages
    temporels, Re et graine physique remplaces par le point donne.
    Partagee par `select_by_holdout_validation` (damier tenu a l'ecart)
    et par la diversification d'entrainement (`TRAINING_REGIME_GRID`) —
    le nom ne doit plus dire seulement « holdout »."""
    cfg = dict(base_config)
    cfg["Re"] = re
    cfg["Rm"] = re
    cfg["phys_seed"] = phys_seed
    return cfg


def _top_completed_candidates(study, top_k):
    """Les `top_k` essais complets les mieux classes EN ECHANTILLON, avec
    leur jeu COMPLET d'hyperparametres — jamais `trial.params` seul, voir
    `deployable_params` : il manquerait les parametres figes."""
    completed = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value < float("inf")]
    completed.sort(key=lambda t: t.value)
    out = []
    for t in completed[:top_k]:
        resolved = t.user_attrs.get("hyperparams_resolved")
        if resolved is None:
            resolved = {**FIXED_PARAMS, **t.params}
        out.append((t.number, float(t.value), dict(resolved)))
    return out


class _HoldoutTrial:
    """`_run_one_scenario`/`_composite_loop` ecrivent sur `trial`
    (`set_user_attr`, `report`, `should_prune`) : une evaluation de
    validation n'appartient a aucune etude Optuna, donc cet objet jetable
    absorbe les ecritures sans rien persister. `should_prune` renvoie
    toujours faux — un candidat de validation doit etre note sur les 8
    scenarios, jamais coupe en cours de route par le pruner median d'une
    autre etude."""

    def __init__(self, number):
        self.number = number

    def set_user_attr(self, *args, **kwargs):
        pass

    def report(self, *args, **kwargs):
        pass

    def should_prune(self):
        return False


def select_by_holdout_validation(study, scenario_list, top_k=HOLDOUT_TOP_K,
                                 classical_only=False, label="",
                                 holdout_grid=HOLDOUT_GRID):
    """Reclasse les `top_k` meilleurs essais EN ECHANTILLON de `study`
    par leur perte MOYENNE sur un DAMIER de regimes physiques TENUS A
    L'ECART des 3 phases (plusieurs Re, plusieurs graines), et rend le
    gagnant.

    Un candidat qui brille sur un seul point du damier mais s'effondre
    sur les autres ne doit pas gagner : c'est le meme surapprentissage
    que celui que le damier existe pour detecter, deplace d'un cran.
    Classer par MOYENNE plutot que par un point unique le rend visible
    (voir `test_the_winner_generalises_across_the_whole_grid_not_one_lucky_point`).
    Le pire point de chaque candidat est aussi journalise
    (`holdout_worst`), pour distinguer en aval un gagnant regulier d'un
    gagnant qui doit sa moyenne a un point tres favorable.

    Ne relance AUCUNE recherche Optuna : re-evalue des jeux de
    parametres deja tires contre des donnees DNS neuves, precalculees
    une seule fois par point du damier pour tout le top_k.
    """
    candidates = _top_completed_candidates(study, top_k)
    if not candidates:
        return {"winner": None, "candidates": [],
                "train_winner_differs": None}

    holdout_scenarios_by_point = {}
    dns_by_point = {}
    for re, phys_seed in holdout_grid:
        holdout_scenarios = [(k, _with_physical_regime(cfg, re, phys_seed))
                             for k, cfg in scenario_list]
        holdout_scenarios_by_point[(re, phys_seed)] = holdout_scenarios
        dns_by_point[(re, phys_seed)] = _precompute_dns_for(
            holdout_scenarios,
            label=f"{label} validation tenue a l'ecart (Re={re}, graine={phys_seed})")

    scored = []
    for number, train_value, params in candidates:
        per_point = {}
        for point in holdout_grid:
            per_point[point] = _composite_loop(
                _HoldoutTrial(number), holdout_scenarios_by_point[point],
                dns_by_point[point], params, LAMBDA_COST_SOFT,
                classical_only=classical_only)
        losses = list(per_point.values())
        scored.append({
            "trial": number, "train_value": train_value,
            "holdout_value": float(np.mean(losses)),
            "holdout_worst": float(np.max(losses)),
            "holdout_per_point": {f"Re={re}_seed={s}": v
                                  for (re, s), v in per_point.items()},
            "params": params,
        })

    scored.sort(key=lambda r: r["holdout_value"])
    winner = scored[0]
    train_winner_trial = candidates[0][0]
    print(f"\n[{label}] selection par validation tenue a l'ecart "
          f"(top {len(candidates)} essais reclasses sur "
          f"{len(holdout_grid)} points) :")
    print(f"  meilleur EN ECHANTILLON : essai #{train_winner_trial}")
    print(f"  meilleur EN VALIDATION  : essai #{winner['trial']} "
          f"(perte moyenne tenue a l'ecart {winner['holdout_value']:.6f}, "
          f"pire point {winner['holdout_worst']:.6f})")
    if winner["trial"] != train_winner_trial:
        print("  -> DIFFERENT : le meilleur en echantillon n'est PAS le "
              "meilleur en validation ; la selection a ecarte un "
              "candidat en surapprentissage")
    return {
        "winner": winner,
        "candidates": scored,
        "train_winner_trial": train_winner_trial,
        "train_winner_differs": winner["trial"] != train_winner_trial,
        "holdout_grid": list(holdout_grid),
        "top_k": top_k,
    }


# ============================================================
#  SORTIE — un JSON qui suffit a redeployer
# ============================================================

def deployable_params(study):
    """Le jeu COMPLET d'hyperparametres du meilleur essai.

    `study.best_params` ne contient que ce qu'Optuna a echantillonne. Un
    JSON construit a partir de lui perd les parametres fixes et se
    retrouve complete au deploiement par des replis — c'est le mecanisme
    exact de D-22 : `sigma` disparu, `gamma_hydro` / `gamma_mag` /
    `kappa` presents dans le fichier deploye sans qu'aucune base ne les
    ait jamais echantillonnes.

    L'essai porte le dictionnaire resolu en attribut ; on le relit. S'il
    manque (essai d'une ancienne campagne), on le reconstruit et on le
    signale, plutot que de renvoyer un dictionnaire incomplet qui
    ressemble a un complet.
    """
    resolved = study.best_trial.user_attrs.get("hyperparams_resolved")
    if resolved is not None:
        return dict(resolved), "trial_user_attr"
    rebuilt = {**FIXED_PARAMS, **study.best_params}
    missing = [n for n in SEARCH_SPACE if n not in rebuilt]
    if missing:
        print(f"[WARN] essai sans `hyperparams_resolved` et sans {missing} : "
              f"le JSON ne suffira pas a redeployer")
    return rebuilt, "rebuilt_from_best_params"


def _phase_block(study, scenario_list):
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return {"n_trials": len(study.trials), "best_score": None,
                "best_params": None, "per_scenario": {}}
    params, source = deployable_params(study)
    return {
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "best_score": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": params,
        "best_params_source": source,
        "sampled_only": study.best_params,
        "per_scenario": {
            key: study.best_trial.user_attrs.get(f"loss_{key}")
            for key, _ in scenario_list
        },
    }


def _atomic_write_json(path, payload):
    """Write a JSON artifact atomically, including on shared storage."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=4)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_phase_candidate(study, phase_key, scenario_list, target_trials,
                         output_path):
    """Export the current best trial without presenting it as deployable.

    A candidate is marked complete only after the global budget is consumed
    and no trial remains WAITING or RUNNING. Multiple workers may call this
    function; the file replacement is atomic.
    """
    trials = study.get_trials(deepcopy=False)
    counts = {
        state.name.lower(): sum(t.state == state for t in trials)
        for state in optuna.trial.TrialState
    }
    consumed = sum(_trial_consumes_budget(t) for t in trials)
    excess = sum(t.user_attrs.get(_BUDGET_EXCESS_ATTR, False) for t in trials)
    active = [t for t in trials if _trial_consumes_budget(t)]
    complete = (consumed >= target_trials
                and not any(t.state == optuna.trial.TrialState.RUNNING
                            for t in active)
                and counts.get("waiting", 0) == 0
                and counts.get("complete", 0) > 0)
    payload = {
        "artifact": "phase_candidate",
        "status": "complete" if complete else "partial",
        "phase": phase_key,
        "study_name": study.study_name,
        "target_trials": target_trials,
        "consumed_trials": consumed,
        "concurrency_excess_trials": excess,
        "trial_states": counts,
        "campaign_contract_sha256": study.user_attrs.get(
            "campaign_contract_sha256"),
        "campaign_contract": study.user_attrs.get("campaign_contract"),
        "provenance": provenance(),
        "search_space": {k: {"low": lo, "high": hi, "log": log}
                         for k, (lo, hi, log) in SEARCH_SPACE.items()},
        "fixed_params": dict(FIXED_PARAMS),
        "result": _phase_block(study, scenario_list),
    }
    _atomic_write_json(output_path, payload)
    print(f"Candidat de phase ecrit dans {output_path} ({payload['status']}).")
    return output_path


def _save_results(study_p1, study_p2, study_p3,
                  study_c1=None, study_c2=None, study_c3=None,
                  filename="best_hyperparams.json",
                  run_holdout_validation=True):
    """Ecrit le JSON final : parametres complets + provenance.

    Le fichier doit se suffire a lui-meme. Toute valeur qu'il ne porte pas
    sera comblee au deploiement par un repli que personne n'a choisi.

    `run_holdout_validation=False` desactive la re-evaluation a Re/graine
    tenus a l'ecart (voir `select_by_holdout_validation`) — reserve aux
    tests et aux runs partiels qui n'ont pas de phase 3 complete ; le
    deploiement final d'une vraie campagne doit toujours la laisser
    active.
    """
    output_path = os.path.join(ensure_dirs(), filename)
    results = {
        "provenance": provenance(),
        "search_space": {k: {"low": lo, "high": hi, "log": log}
                         for k, (lo, hi, log) in SEARCH_SPACE.items()},
        "fixed_params": dict(FIXED_PARAMS),
        "scenarios": {
            "isolated": [k for k, _ in SCENARIOS_ISOLATED],
            "complex":  [k for k, _ in SCENARIOS_COMPLEX],
            "all":      [k for k, _ in SCENARIOS_ALL],
        },
        "lambda_cost": LAMBDA_COST_SOFT,
        "quantum": {
            "phase1": _phase_block(study_p1, SCENARIOS_ISOLATED),
            "phase2": _phase_block(study_p2, SCENARIOS_COMPLEX),
            "phase3": _phase_block(study_p3, SCENARIOS_ALL),
        },
    }
    if study_c1 is not None:
        results["classical"] = {
            "phase1": _phase_block(study_c1, SCENARIOS_ISOLATED),
            "phase2": _phase_block(study_c2, SCENARIOS_COMPLEX) if study_c2 else None,
            "phase3": _phase_block(study_c3, SCENARIOS_ALL) if study_c3 else None,
        }

    # Ce qu'on deploie : la phase 3 reclassee par validation tenue a
    # l'ecart (Re/graine jamais vus en entrainement), pas le meilleur en
    # echantillon brut — voir `select_by_holdout_validation` ci-dessus.
    # `best_params` de la phase 3 reste ecrit tel quel plus haut, pour
    # que le score en echantillon reste lisible a cote de celui qui a
    # vraiment tranche le deploiement.
    quantum_deploy = results["quantum"]["phase3"]["best_params"]
    classical_deploy = (results["classical"]["phase3"]["best_params"]
                        if study_c3 is not None else None)
    if run_holdout_validation and results["quantum"]["phase3"]["best_params"] is not None:
        holdout_q = select_by_holdout_validation(
            study_p3, SCENARIOS_ALL, classical_only=False, label="quantique")
        results["quantum"]["holdout_validation"] = holdout_q
        if holdout_q["winner"] is not None:
            quantum_deploy = holdout_q["winner"]["params"]
    if (run_holdout_validation and study_c3 is not None
            and results["classical"]["phase3"]["best_params"] is not None):
        holdout_c = select_by_holdout_validation(
            study_c3, SCENARIOS_ALL, classical_only=True, label="classique")
        results["classical"]["holdout_validation"] = holdout_c
        if holdout_c["winner"] is not None:
            classical_deploy = holdout_c["winner"]["params"]

    results["deploy"] = {"quantum": quantum_deploy, "classical": classical_deploy}

    _atomic_write_json(output_path, results)
    print(f"\nResultats finaux ecrits dans {output_path}")
    print(f"  quantique  : {results['deploy']['quantum']}")
    print(f"  classique  : {results['deploy']['classical']}")

    return output_path


def _deploy(staged_path):
    """Copy a just-completed campaign result to where `study/` reads it.

    D-22 : `_save_results` ecrit dans `CAMPAIGN_DIR`
    (`results/hyperparams/reoptimisation/`), un registre permanent, jamais
    ecrase. `hyperparams_loader.resolve_hyperparams_path()` lit un chemin
    DIFFERENT (`results/hyperparams/best_hyperparams.json`) par defaut, et
    rien ne copiait l'un vers l'autre avant cette fonction : une campagne
    pouvait tourner jusqu'au bout et son resultat n'atteignait jamais
    `pipeline.py`/`study/` sans qu'un humain se souvienne d'une etape
    manuelle. Provenance de l'ancien fichier deploye desormais sans objet
    (il va etre retrace ci-dessous) : ce n'est plus la question qui
    compte, celle qui compte est que CE resultat-ci soit bien celui que
    tout le reste consomme.

    Appelee uniquement depuis `main()`, apres un `--phase all` complet
    (les deux bras, les trois phases) : `staged_path` est donc toujours un
    resultat termine, jamais un candidat partiel.
    """
    from hyperparams_loader import resolve_hyperparams_path
    deploy_path = resolve_hyperparams_path()
    if os.path.abspath(deploy_path) == os.path.abspath(staged_path):
        return deploy_path
    with open(staged_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    _atomic_write_json(deploy_path, payload)
    print(f"Deploye vers {deploy_path} "
          f"(chemin que `hyperparams_loader` lit par defaut).")
    return deploy_path


# ============================================================
#  MAIN
# ============================================================

PHASE_CHOICES = ("1", "2", "3", "classical_1", "classical_2", "classical_3",
                 "classical", "all")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Entrainement des hyperparametres Q-HAS (9 parametres).")
    p.add_argument("--phase", choices=PHASE_CHOICES,
                   default="all", help="phase a executer (defaut : tout)")
    p.add_argument("--seed", type=int,
                   default=None,
                   help="graine du sampler TPE. Sans elle, Optuna tire au "
                        "hasard et la campagne n'est pas reproductible.")
    p.add_argument("--n-trials", type=int,
                   help="cible globale d'essais pour --phase 1; remplace la "
                        "valeur du protocole pour cette execution")
    p.add_argument("--result-path",
                   help="chemin du JSON candidat de --phase 1")
    p.add_argument("--print-space", action="store_true",
                   help="affiche l'espace de recherche et sort, sans rien "
                        "calculer. A lancer AVANT de louer des coeurs.")
    p.add_argument("--prepare-only", action="store_true",
                   help="prepare/reprend la phase 1 avant de lancer les workers")
    p.add_argument("--finalize-only", action="store_true",
                   help="valide et exporte le candidat de phase 1 sans calcul")
    p.add_argument(
        "--no-deploy", action="store_true",
        help="n'ecrit pas vers le chemin que hyperparams_loader lit par "
             "defaut (D-22) ; le resultat final reste dans "
             "CAMPAIGN_DIR seulement, a deployer a la main")
    args = p.parse_args(argv)
    if args.n_trials is not None and args.n_trials < 1:
        p.error("--n-trials doit etre >= 1")
    if ((args.n_trials is not None or args.result_path is not None)
            and args.phase != "1"):
        p.error("--n-trials et --result-path sont reserves a --phase 1")
    if (args.prepare_only or args.finalize_only) and args.phase != "1":
        p.error("--prepare-only/--finalize-only sont reserves a --phase 1")
    if args.prepare_only and args.finalize_only:
        p.error("--prepare-only et --finalize-only sont exclusifs")
    return args


def _load_study(phase_key):
    """Charge une etude existante pour amorcer la suivante."""
    return optuna.load_study(study_name=PHASES[phase_key]["study_name"],
                             storage=_get_storage(PHASES[phase_key]))


def main(argv=None):
    args = parse_args(argv)

    if args.print_space:
        print(json.dumps({
            "search_space": {k: {"low": lo, "high": hi, "log": log}
                             for k, (lo, hi, log) in SEARCH_SPACE.items()},
            "fixed_params": dict(FIXED_PARAMS),
            "scenarios": {"isolated": [k for k, _ in SCENARIOS_ISOLATED],
                          "complex": [k for k, _ in SCENARIOS_COMPLEX],
                          "all": [k for k, _ in SCENARIOS_ALL]},
            "n_trials": {k: v["n_trials"] for k, v in PHASES.items()},
            "provenance": provenance(),
        }, indent=2))
        return 0

    announce_environment()
    if args.prepare_only or args.finalize_only:
        target = (args.n_trials if args.n_trials is not None
                  else PHASES["phase1_composite"]["n_trials"])
        study, recovered = prepare_phase1(target)
        if args.finalize_only:
            result_path = (args.result_path or os.path.join(
                ensure_dirs(), "candidate_phase1.json"))
            save_phase_candidate(
                study, "phase1_composite", SCENARIOS_ISOLATED,
                target, result_path)
            return 0
        print(f"Campagne prete : {trials_done(study)}/{target} essais; "
              f"{recovered} essai(s) interrompu(s) marques a recalculer.")
        return 0
    if args.seed is None:
        print("[WARN] pas de --seed : le sampler TPE est aleatoire, cette "
              "campagne ne sera pas reproductible telle quelle.")

    if args.phase == "1":
        target = (args.n_trials if args.n_trials is not None
                  else PHASES["phase1_composite"]["n_trials"])
        study = _run_phase1(
            None, args.seed, n_trials=target,
            dns_traces_by_regime=_precompute_dns_by_regime(
                SCENARIOS_ISOLATED, label="6 isoles"))
        result_path = (args.result_path or
                       os.path.join(ensure_dirs(), "candidate_phase1.json"))
        save_phase_candidate(study, "phase1_composite", SCENARIOS_ISOLATED,
                             target, result_path)

    elif args.phase == "2":
        _run_phase2(_load_study("phase1_composite"), args.seed)

    elif args.phase == "3":
        _run_phase3(_load_study("phase2_complex"), args.seed)

    elif args.phase == "classical_1":
        _run_classical_phase1(
            None, args.seed,
            dns_traces_by_regime=_precompute_dns_by_regime(
                SCENARIOS_ISOLATED, label="6 isoles"))

    elif args.phase == "classical_2":
        _run_classical_phase2(_load_study("classical_phase1"), args.seed)

    elif args.phase == "classical_3":
        _run_classical_phase3(_load_study("classical_phase2"), args.seed)

    elif args.phase == "classical":
        dns_by_regime = _precompute_dns_by_regime(SCENARIOS_ISOLATED,
                                                   label="6 isoles")
        c1 = _run_classical_phase1(None, args.seed,
                                   dns_traces_by_regime=dns_by_regime)
        c2 = _run_classical_phase2(c1, args.seed)
        _run_classical_phase3(c2, args.seed)

    else:  # "all"
        # Meme damier (`dns_by_regime`, DIVERSIFICATION DE L'ENTRAINEMENT
        # ci-dessus) partage entre les deux bras, comme l'etait deja `dns`
        # avant : seule la regle de decision doit differer (`CLAUDE.md`).
        dns_by_regime = _precompute_dns_by_regime(SCENARIOS_ISOLATED,
                                                   label="6 isoles")
        p1 = _run_phase1(None, args.seed, dns_traces_by_regime=dns_by_regime)
        p2 = _run_phase2(p1, args.seed)
        p3 = _run_phase3(p2, args.seed)
        c1 = _run_classical_phase1(None, args.seed,
                                   dns_traces_by_regime=dns_by_regime)
        c2 = _run_classical_phase2(c1, args.seed)
        c3 = _run_classical_phase3(c2, args.seed)
        staged_path = _save_results(p1, p2, p3, c1, c2, c3)
        if args.no_deploy:
            print(f"--no-deploy : reste dans {staged_path}, "
                  f"pas copie vers le chemin lu par defaut.")
        else:
            _deploy(staged_path)
        print("\n" + "=" * 60)
        print("ALL TRAINING COMPLETE")
        print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
