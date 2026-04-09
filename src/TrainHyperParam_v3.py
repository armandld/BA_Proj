"""
TrainHyperParam_v3 — Multi-Scenario Progressive Training
=========================================================

Rationale:
----------
Orszag-Tang generates ALL anomaly classes simultaneously (shear, vortex,
current sheets, reconnection).  When anomalies coexist spatially, the optimizer
cannot decouple their contributions — there is a degeneracy in parameter space.

v10 Hamiltonian architecture (Adaptive Z + Decoupled f × g × ThrContrast):
--------------------------------------------------------------------------
Z-terms REINTRODUCED with adaptive weight: alpha_z = w_z_frac × median(|C|,|K|).
θ initialization encodes P(|1⟩) = classical_score.
Z bias breaks ground-state degeneracy; ZZ/ZZZZ encode spatial correlations.

Hamiltonian terms (sign convention: all ≤ 0, ferromagnetic/even-parity):
- Z (bias):      alpha_z × (score − threshold_amr), adaptive weight from median coupling
- ZZ (gradient): −2 × g_strain(Q_OW) × √((f_hydro(Re)×ThrContrast(Δv))² + ...)
- ZZZZ (circ):   −1 × √((g_rot(Q)×f_hydro(Re)×ThrContrast(ωz))² + ...)
- ZZZZ (xpoint): −f_Rm × ThrContrast(max(0, −det(J_B)))  [reconnection X-points]

Threshold-relative contrast (replaces Michelson):
  ThrContrast(val, val_crit, β) = β × max(0, val/val_crit − 1)
Survives in spatially uniform regimes (Michelson killed the signal).

Sensitivity is SPLIT by term type:
- β_grad:  gradient ZZ (shear layers, velocity/magnetic jumps)
- β_curl:  plaquette ZZZZ (vorticity, current density — curl-like quantities)
- β_xpoint: X-point ZZZZ (reconnection — very sparse, localized at X-points)

Training:
----------------
    Phase 1  : Harris Tearing

    Classical Phase 1 (*) : Classical AMR on 4 isolated scenarios.
               Trains threshold_amr only (no quantum circuit — ~100x faster).
"""

import optuna
import os
import json
import csv
import sys
import itertools
import shutil
import numpy as np
from pipeline import pipeline
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

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# ── DISTRIBUTED TRAINING ─────────────────────────────────────
OPTUNA_STORAGE = os.environ.get("OPTUNA_STORAGE", None)
OPTUNA_JOURNAL = os.environ.get("OPTUNA_JOURNAL", None)
WORKER_PHASE   = os.environ.get("WORKER_PHASE", None)
WORKER_TRIALS  = os.environ.get("WORKER_TRIALS", None)
if WORKER_TRIALS is not None:
    WORKER_TRIALS = int(WORKER_TRIALS)

DISTRIBUTED = OPTUNA_STORAGE is not None
JOURNAL_DIR = OPTUNA_JOURNAL

if DISTRIBUTED:
    print(f"[DISTRIBUTED MODE] Storage: {OPTUNA_STORAGE.split('@')[-1]}")
elif JOURNAL_DIR is not None:
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    print(f"[DISTRIBUTED MODE] Journal storage: {JOURNAL_DIR}")
if WORKER_PHASE:
    print(f"[DISTRIBUTED MODE] Worker phase: {WORKER_PHASE}")
if WORKER_TRIALS:
    print(f"[DISTRIBUTED MODE] Max trials per phase: {WORKER_TRIALS}")

# --- DETECTION AUTOMATIQUE DE L'ENVIRONNEMENT ---
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Environnement Google Colab detecte. Activation du mode 'Copie Drive'.")
    drive_dir = os.path.join(project_root, "Train_results")
    os.makedirs(drive_dir, exist_ok=True)
    local_dir = "/content/Train_results_local"
    os.makedirs(local_dir, exist_ok=True)
    for file in os.listdir(drive_dir):
        if file.endswith(".db"):
            shutil.copy2(os.path.join(drive_dir, file), os.path.join(local_dir, file))
    data_dir = local_dir
else:
    print("Environnement Local detecte. Ecriture directe sur le disque.")
    data_dir = os.path.join(project_root, "Train_results")
    os.makedirs(data_dir, exist_ok=True)
    drive_dir = None


# ============================================================
#  SCENARIO CONFIGURATIONS
# ============================================================
# Each isolated scenario has its own physics parameters (Re, Rm)
# tuned so the targeted anomaly develops clearly within the time window.

SCENARIO = {
    "scenario": "ghost_twisting",
    "N": N_TRAINING,
    "max_depth_override": MAX_DEPTH_TRAINING,
    "T_MAX": 0.8,
    "T_START": 0.0,
    "DT": 1e-3,
    "HYBRID_DT": 0.10,
    "K_opt": 30,
    "Re": 800,
    "Rm": 800,
    "shots": 256,
    "AdvAnomaliesEnable": True,
}


# ============================================================
#  PHASE DEFINITIONS (v2 — 3-phase progressive split-β)
# ============================================================

PHASES = {
    "phase1": {
        # Phase 1: Composite loss on all 4 isolated scenarios.
        # 6 trainable params — all Michelson terms share a single β_michelson.
        # Actual scenario configs are SCENARIO_KH, SCENARIO_VORTEX, etc.
        "n_trials": 600,
        "train_hamiltonian": True,
        "split_michelson": False,      # Single β_michelson for all terms
        "AdvAnomaliesEnable": True,
        "study_name": "q_has_v2_phase1",
        # Placeholder values (not used directly by pipeline)
        "N": N_TRAINING,
        "max_depth_override": MAX_DEPTH_TRAINING,
        "T_MAX": 3.0,
        "T_START": 1.0,
        "DT": 1e-3,
        "HYBRID_DT": 0.10,
        "K_opt": 30,
        "Re": 800,
        "Rm": 800,
        "shots": 256,
    },

    # ── CLASSICAL TRAINING PHASES ──────────────────────────────
    "classical_phase1": {
        # Phase *: Classical AMR on 4 isolated scenarios.
        # Trains threshold_amr only (no quantum circuit → ~100x faster).
        # Same composite loss as QAOA Phase 1.
        "n_trials": 300,
        "train_hamiltonian": False,
        "split_michelson": False,
        "classical_only": True,
        "AdvAnomaliesEnable": True,
        "study_name": "classical_v2_phase1",
        "N": N_TRAINING,
        "max_depth_override": MAX_DEPTH_TRAINING,
        "T_MAX": 3.0,
        "T_START": 1.0,
        "DT": 1e-3,
        "HYBRID_DT": 0.10,
        "K_opt": 30,
        "Re": 800,
        "Rm": 800,
        "shots": 256,
    },
}


# ============================================================
#  HELPERS
# ============================================================

def create_argus(scenario_config):
    """Build the argus namespace for a given scenario config."""
    return SimpleNamespace(
        reps=2,
        mode="simulator",
        backend="state_vector",
        shots=scenario_config["shots"],
        method="COBYLA",
        opt_level=1,
        AdvAnomaliesEnable=scenario_config.get("AdvAnomaliesEnable", False),
        K_opt=scenario_config["K_opt"],
        eps=1e-2,
        eta=0.001,
        Bz_guide=0.1,
        c_s=1.0,
        Re=scenario_config["Re"],
        Rm=scenario_config["Rm"],
    )


def _get_storage(phase_config):
    """Return the Optuna storage backend for this phase."""
    if DISTRIBUTED:
        # Neon's pooler drops idle SSL connections after a few minutes.
        # Each trial can take 10-20 min of computation, so the DB connection
        # goes stale.  pool_pre_ping=True makes SQLAlchemy test the connection
        # before each use and reconnect if needed.  pool_recycle=300 proactively
        # refreshes connections older than 5 min.
        return optuna.storages.RDBStorage(
            url=OPTUNA_STORAGE,
            engine_kwargs={
                "pool_pre_ping": True,
                "pool_recycle": 300,
            },
        )
    if JOURNAL_DIR is not None:
        journal_path = os.path.join(JOURNAL_DIR, f"{phase_config['study_name']}.log")
        lock = optuna.storages.JournalFileOpenLock(journal_path)
        return optuna.storages.JournalStorage(
            optuna.storages.JournalFileBackend(journal_path, lock_obj=lock)
        )
    db_path = os.path.join(data_dir, f"{phase_config['study_name']}.db")
    return f"sqlite:///{db_path}"


def run_phase(phase_name, phase_config, objective_fn, seed_params=None):
    """Run one training phase with Optuna MedianPruner."""
    storage = _get_storage(phase_config)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=15,
        n_warmup_steps=2,
        interval_steps=1,
        n_min_trials=5,
    )

    study = optuna.create_study(
        study_name=phase_config["study_name"],
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
    )

    if seed_params is not None and len(study.trials) == 0:
        for params in seed_params:
            study.enqueue_trial(params)

    db_path = os.path.join(data_dir, f"{phase_config['study_name']}.db") if not DISTRIBUTED else None

    def callback_save(study, trial):
        if IN_COLAB and not DISTRIBUTED and trial.number % 10 == 0:
            shutil.copy2(db_path, os.path.join(drive_dir, f"{phase_config['study_name']}.db"))

    trials_done = len([
        t for t in study.trials
        if t.state != optuna.trial.TrialState.WAITING
    ])
    target_trials = phase_config["n_trials"]
    remaining_trials = target_trials - trials_done

    if WORKER_TRIALS is not None:
        remaining_trials = min(remaining_trials, WORKER_TRIALS)

    if remaining_trials > 0:
        if trials_done > 0:
            print(f"Reprise de la phase '{phase_name}' : {trials_done} trials existants. "
                  f"Ce worker va en faire {remaining_trials}.")
        else:
            print(f"Lancement de la phase '{phase_name}' : {target_trials} trials a faire. "
                  f"Ce worker va en faire {remaining_trials}.")
        study.optimize(objective_fn, n_trials=remaining_trials, callbacks=[callback_save])
    else:
        print(f"Phase '{phase_name}' deja terminee ({trials_done}/{target_trials} trials).")

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
    """
    Extract top-k parameter sets from rescore CSV files across multiple lambdas.

    Reads Train_results/rescore_{phase_prefix}_lambda{X}/trials_lambda{X}.csv
    for each lambda value and returns the top-k trials per lambda.

    Parameters
    ----------
    phase_prefix : str
        e.g. "q_has_v2_phase1" or "q_has_v2_phase1b" or "classic_v2_phase1"
    lambdas : list of float
        Lambda values to read from, e.g. [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    top_k : int
        Number of top trials to extract per lambda (default 2)

    Returns
    -------
    list of dict : parameter dictionaries (deduplicated by trial number)
    """
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


# Beta split strategies for Phase 1b seeding
# Each strategy maps beta_michelson → (sigma, beta_curl, beta_xpoint)
SPLIT_BETA_STRATEGIES = [
    # 1. Tight uncertainty + uniform curl/xpoint
    {"name": "tight_uniform",     "sigma": 0.05, "curl": 1.0, "xpoint": 1.0},
    # 2. Medium uncertainty + curl-dominant
    {"name": "medium_curl",       "sigma": 0.10, "curl": 2.0, "xpoint": 0.5},
    # 3. Tight uncertainty + xpoint-dominant
    {"name": "tight_xpoint",      "sigma": 0.05, "curl": 0.5, "xpoint": 2.0},
    # 4. Wide uncertainty (closer to always-on baseline)
    {"name": "wide_uniform",      "sigma": 0.20, "curl": 1.0, "xpoint": 1.0},
]


# ============================================================
#  COMPOSITE MULTI-SCENARIO OBJECTIVE (QAOA — Phase 1 & 2)
# ============================================================

def make_composite_objective(dns_traces, scenario_list, split_michelson=False, frozen_params=None, lambda_cost=LAMBDA_COST_SOFT):
    """
    Composite loss across a set of scenarios (QAOA method).

    Loss = mean(Loss_i) over all scenarios in scenario_list.

    Each sub-loss uses the SAME hyperparameters but a DIFFERENT scenario,
    so the optimizer must find params that work across all anomaly types.

    Parameters
    ----------
    dns_traces : dict mapping scenario keys to (dns_trace, hot_start_state) tuples.
    scenario_list : list of (key, config) tuples — which scenarios to include.
    split_michelson : if False (Phase 1), trains a single beta_michelson.
                      if True  (Phase 2), trains sigma, beta_curl, beta_xpoint.
    frozen_params : dict of hyperparams that are FIXED (not optimized).
    """
    frozen = frozen_params or {}
    VQA_N = 2

    def objective(trial):
        HyperParams = {}

        # ── BASELINE CLASSIQUE ──
        HyperParams["threshold_amr"] = trial.suggest_float("threshold_amr", 0.15, 0.40)
        
        # ── LE GAIN QUANTIQUE GLOBAL ──
        HyperParams["beta"] = trial.suggest_float("beta", 0.0, 15.0)

        # ── COUPLAGE SYMÉTRIQUE (Gammas) ──
        # On colle Hydro et Mag ensemble
        
        HyperParams["gamma_mag"]   = HyperParams["gamma_mag"] = trial.suggest_float("gamma_mag", 0.5, 4.0)
        HyperParams["gamma_hydro"] = HyperParams["gamma_mag"]

        # ── ANCRAGE ET KERNEL ──
        HyperParams["kappa"]    = trial.suggest_float("kappa", 1.0, 15.0)
        HyperParams["w_z_frac"] = trial.suggest_float("w_z_frac", 0.1, 2.0)
        HyperParams["sigma"]    = trial.suggest_float("sigma", 0.01, 0.25)

        # ── SENSIBILITÉ SYMÉTRIQUE (Bétas sub-moteurs) ──
        # On colle Curl et X-point ensemble
        
        HyperParams["beta_curl"]   = trial.suggest_float("beta_curl", 0.5, 5.0)
        HyperParams["beta_xpoint"] = HyperParams["beta_curl"]

        # Inject frozen params
        for k, v in frozen.items():
            HyperParams[k] = v

        total_loss = 0.0
        sub_losses = {}

        for scenario_key, scenario_config in scenario_list:
            dns_trace, hot_start_state = dns_traces[scenario_key]
            N = scenario_config["N"]
            T_MAX = scenario_config["T_MAX"]
            DT = scenario_config["DT"]
            HYBRID = int(scenario_config["HYBRID_DT"] / DT)
            min_patch_size = scenario_config.get("min_patch_size", 6)
            max_depth_override = scenario_config.get("max_depth_override", None)
            argus = create_argus(scenario_config)

            try:
                result = pipeline(
                    N=N, VQA_N=VQA_N, T_MAX=T_MAX, DT=DT, HYBRID=HYBRID,
                    verbose=False, argus=argus, hyperparams=HyperParams,
                    lambda_cost=lambda_cost, trial=None,  # No pruning on sub-losses
                    dns_trace=dns_trace, hot_start_state=hot_start_state,
                    min_patch_size=min_patch_size,
                    max_depth_override=max_depth_override,
                    scenario=scenario_config["scenario"],
                    return_details=True,
                )
            except Exception as e:
                print(f"[Trial {trial.number}] FAILED on {scenario_key}: {e}")
                import traceback; traceback.print_exc()
                penalty = 10.0
                sub_losses[scenario_key] = penalty
                total_loss += penalty
                trial.set_user_attr(f"phys_{scenario_key}", penalty)
                trial.set_user_attr(f"patch_{scenario_key}", 1.0)
                continue

            combined = result['combined'] if isinstance(result, dict) else result
            if np.isnan(combined) or np.isinf(combined):
                combined = 10.0

            sub_losses[scenario_key] = combined
            total_loss += combined

            if isinstance(result, dict):
                trial.set_user_attr(f"phys_{scenario_key}", float(result.get('phys_score', 0)))
                trial.set_user_attr(f"patch_{scenario_key}", float(result.get('patch_ratio', 0)))
                for field, err in result.get('field_errors', {}).items():
                    trial.set_user_attr(f"error_{field}_{scenario_key}", float(err))

        for key, loss in sub_losses.items():
            trial.set_user_attr(f"loss_{key}", float(loss))

        composite = total_loss / len(scenario_list)

        if trial is not None:
            trial.report(composite, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return composite

    return objective


# ============================================================
#  CLASSICAL COMPOSITE OBJECTIVE (Phase * & **)
# ============================================================

def make_classical_composite_objective(dns_traces, scenario_list, lambda_cost=LAMBDA_COST_SOFT):
    """
    Composite loss across scenarios for the CLASSICAL AMR method.

    Only trains threshold_amr (1 parameter). No quantum circuit is
    involved, making each trial ~100x faster than QAOA.

    Parameters
    ----------
    dns_traces : dict mapping scenario keys to (dns_trace, hot_start_state) tuples.
    scenario_list : list of (key, config) tuples — which scenarios to include.
    """
    VQA_N = 2

    def objective(trial):
        threshold_amr = trial.suggest_float("threshold_amr", 0.05, 0.8)

        HyperParams = {"threshold_amr": threshold_amr}

        total_loss = 0.0
        sub_losses = {}

        for scenario_key, scenario_config in scenario_list:
            dns_trace, hot_start_state = dns_traces[scenario_key]
            N = scenario_config["N"]
            T_MAX = scenario_config["T_MAX"]
            DT = scenario_config["DT"]
            HYBRID = int(scenario_config["HYBRID_DT"] / DT)
            min_patch_size = scenario_config.get("min_patch_size", 6)
            max_depth_override = scenario_config.get("max_depth_override", None)
            argus = create_argus(scenario_config)

            try:
                result = pipeline(
                    N=N, VQA_N=VQA_N, T_MAX=T_MAX, DT=DT, HYBRID=HYBRID,
                    verbose=False, argus=argus, hyperparams=HyperParams,
                    lambda_cost=lambda_cost, trial=None,
                    dns_trace=dns_trace, hot_start_state=hot_start_state,
                    min_patch_size=min_patch_size,
                    max_depth_override=max_depth_override,
                    scenario=scenario_config["scenario"],
                    return_details=True,
                    classical_only=True,
                )
            except Exception as e:
                print(f"[Classical Trial {trial.number}] FAILED on {scenario_key}: {e}")
                import traceback; traceback.print_exc()
                penalty = 10.0
                sub_losses[scenario_key] = penalty
                total_loss += penalty
                trial.set_user_attr(f"phys_{scenario_key}", penalty)
                trial.set_user_attr(f"patch_{scenario_key}", 1.0)
                continue

            combined = result['combined'] if isinstance(result, dict) else result
            if np.isnan(combined) or np.isinf(combined):
                combined = 10.0

            sub_losses[scenario_key] = combined
            total_loss += combined

            if isinstance(result, dict):
                trial.set_user_attr(f"phys_{scenario_key}", float(result.get('phys_score', 0)))
                trial.set_user_attr(f"patch_{scenario_key}", float(result.get('patch_ratio', 0)))
                for field, err in result.get('field_errors', {}).items():
                    trial.set_user_attr(f"error_{field}_{scenario_key}", float(err))

        for key, loss in sub_losses.items():
            trial.set_user_attr(f"loss_{key}", float(loss))

        composite = total_loss / len(scenario_list)

        if trial is not None:
            trial.report(composite, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return composite

    return objective


# ============================================================
#  ORCHESTRATION
# ============================================================
SCENARIO_GT = [("gt", SCENARIO)]


def _precompute_dns_for(scenario_list, label="scenarios"):
    """Pre-compute DNS traces for a list of (key, config) scenarios."""
    print(f"\n--- Pre-computing DNS traces for {label} ---")
    dns_traces = {}
    for key, config in scenario_list:
        config_with_name = {**config, "study_name": f"dns_{key}"}
        dns_trace, hot_start = precompute_dns(config_with_name)
        dns_traces[key] = (dns_trace, hot_start)
    print(f"--- DNS pre-computation complete ({label}) ---\n")
    return dns_traces


def _precompute_composite_dns():
    """Pre-compute DNS traces for all 4 isolated scenarios (backward compat)."""
    return _precompute_dns_for(SCENARIO_GT, label="4 isolated scenarios")


def _run_phase1(dns_traces):
    """Phase 1: Composite training with shared β_michelson (7 params)."""
    print("=" * 60)
    print("PHASE 1: Composite Training (shared Michelson)")
    print("  Training: beta, threshold_amr, beta_michelson, gamma_hydro, gamma_mag, kappa, w_z_frac")
    print("  Scenarios: KH + Vortex + Tearing + Coalescence")
    print("=" * 60)

    initial_seeds = [
    # -------------------------------------------------------------------------
    # SEED 1 : LE TÉMOIN "QUASI-CLASSIQUE" (Baseline)
    # But : Servir de point de contrôle. Presque aucune influence quantique.
    # Si le score est identique au classique pur, c'est normal.
    # -------------------------------------------------------------------------
    {
        "threshold_amr": 0.322, 
        "beta": 0.001,            # Influence quantique négligeable
        "beta_curl": 0.01,         # Presque éteint
        "gamma_mag": 1.0, 
        "kappa": 5.0, 
        "w_z_frac": 3.0,          # Ton ancre de biais
        "sigma": 0.10
    },
    
    # -------------------------------------------------------------------------
    # SEED 2 : LE "CHASSEUR DE TORSION" (Maximum Quantum)
    # But : Forcer le Hamiltonien à détecter la rotation de phase (Twist).
    # On booste beta_curl pour vaincre le biais massif de 3.0.
    # -------------------------------------------------------------------------
    {
        "threshold_amr": 0.322, 
        "beta": 8.5,              # Gain global fort pour surmonter le biais Z
        "beta_curl": 1.5,          # Sensibilité maximale à la rotation (Plaquette ZZZZ)
        "gamma_mag": 2.5,          # On sur-pondère la physique magnétique
        "kappa": 12.0,             # "Zoom" élevé sur les variations de phase
        "w_z_frac": 3.0,           # On garde l'ancre identique
        "sigma": 0.04              # Focus très fin pour éviter de lisser le signal
    }
]

    objective = make_composite_objective(dns_traces, SCENARIO_GT, split_michelson=True)
    study = run_phase("phase1", PHASES["phase1"],
                      objective, seed_params=initial_seeds)

    print(f"\nPhase 1 — Best composite score: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    best_trial = study.best_trial
    for key in ["gt"]:
        loss_key = f"loss_{key}"
        if loss_key in best_trial.user_attrs:
            print(f"  {key:>16}: {best_trial.user_attrs[loss_key]:.6f}")

    return study


def _run_classical_phase1(dns_traces):
    """Classical Phase * : threshold_amr on 4 isolated scenarios."""
    print("\n" + "=" * 60)
    print("CLASSICAL PHASE 1 (*): Training threshold_amr")
    print("  Training: threshold_amr (1 param)")
    print("  Scenarios: KH + Vortex + Tearing + Coalescence")
    print("  Method: classical AMR (no quantum circuit)")
    print("=" * 60)

    # Grid-search seeds for the single parameter
    initial_seeds = [{"threshold_amr": t} for t in
                     np.linspace(0.05, 0.8, 20).tolist()]

    objective = make_classical_composite_objective(dns_traces, SCENARIO_GT)
    study = run_phase("classical_phase1", PHASES["classical_phase1"],
                      objective, seed_params=initial_seeds)

    print(f"\nClassical Phase 1 — Best composite score: {study.best_value:.6f}")
    print(f"Best threshold_amr: {study.best_params}")

    best_trial = study.best_trial
    for key in ["gt"]:
        loss_key = f"loss_{key}"
        if loss_key in best_trial.user_attrs:
            print(f"  {key:>16}: {best_trial.user_attrs[loss_key]:.6f}")

    return study

# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    if WORKER_PHASE == "1":
        dns_traces = _precompute_composite_dns()
        _run_phase1(dns_traces)


    elif WORKER_PHASE == "classical_1":
        # Classical Phase *: 4 isolated scenarios
        dns_traces = _precompute_composite_dns()
        _run_classical_phase1(dns_traces)

    else:
        print(f"Worker launched without a valid WORKER_PHASE. Exiting.")