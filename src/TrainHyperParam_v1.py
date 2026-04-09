import optuna
import os
import json
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
LAMBDA_COST = 0.5

# ── FAST TRAINING MODE ─────────────────────────────────────
#  FAST_TRAINING=True  → ~2-5 min/trial  (default for iteration)
#  FAST_TRAINING=False → ~70 min/trial   (full resolution, final run)
#
#  Strategy: cap VQA recursion at MAX_DEPTH_TRAINING levels.
#  This reduces worst-case VQA circuit calls
#  from O(VQA_N^max_depth) ≈ 1365 to O(VQA_N^3) ≈ 85 per hybrid step,
#  giving a ~50-65x speedup. Hyperparameters alpha, beta, threshold_amr
#  and Hamiltonian weights are resolution-independent and
N_TRAINING         = 256
MAX_DEPTH_TRAINING = 3      # Cap recursion depth (None = auto from N)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# ── DISTRIBUTED TRAINING ─────────────────────────────────────
#  Three storage backends (in priority order):
#
#  1. OPTUNA_STORAGE  → PostgreSQL URL (Colab + Neon, or any remote DB)
#  2. OPTUNA_JOURNAL  → JournalFileStorage on shared filesystem (HPC / NFS)
#  3. (default)       → local SQLite in Train_results/
#
#  Environment variables:
#    OPTUNA_STORAGE   : PostgreSQL URL  e.g. "postgresql://user:pass@host/db"
#    OPTUNA_JOURNAL   : Path to shared dir for journal files (HPC clusters)
#    WORKER_PHASE     : Run only this phase: "1", "2", "3" (default: all)
#    WORKER_TRIALS    : Max trials this worker will run per phase (default: all remaining)
#
#  Example — Colab/Neon (remote PostgreSQL):
#    export OPTUNA_STORAGE="postgresql://user:pass@ep-xxx.neon.tech/optuna_db?sslmode=require"
#    export WORKER_PHASE="1" WORKER_TRIALS="50"
#    python TrainHyperParam.py
#
#  Example — HPC cluster (SLURM + shared filesystem):
#    export OPTUNA_JOURNAL="$SCRATCH/qhas_journal"
#    export WORKER_PHASE="1" WORKER_TRIALS="50"
#    python TrainHyperParam.py
# ──────────────────────────────────────────────────────────────
OPTUNA_STORAGE = os.environ.get("OPTUNA_STORAGE", None)
OPTUNA_JOURNAL = os.environ.get("OPTUNA_JOURNAL", None)      # path to shared dir (HPC/NFS)
WORKER_PHASE   = os.environ.get("WORKER_PHASE", None)        # "1", "2", "3", or None (all)
WORKER_TRIALS  = os.environ.get("WORKER_TRIALS", None)       # int or None (all remaining)
if WORKER_TRIALS is not None:
    WORKER_TRIALS = int(WORKER_TRIALS)

DISTRIBUTED = OPTUNA_STORAGE is not None
JOURNAL_DIR = OPTUNA_JOURNAL  # shared filesystem path for JournalFileStorage

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

    print("Verification des sauvegardes sur Google Drive...")
    for file in os.listdir(drive_dir):
        if file.endswith(".db"):
            shutil.copy2(os.path.join(drive_dir, file), os.path.join(local_dir, file))
            print(f"Base de donnees restauree en local : {file}")

    data_dir = local_dir
else:
    print("Environnement Local detecte. Ecriture directe sur le disque.")
    data_dir = os.path.join(project_root, "Train_results")
    os.makedirs(data_dir, exist_ok=True)
    drive_dir = None

# ============================================================
#  HYPERPARAMETER STRUCTURE (v7 — Decoupled f × g × Mic)
#
#  Three tiers with clean separation of concerns:
#
#  Tier 1 — Encoding + AMR calibration (Phase 1):
#    beta           : Phase gain (temporal derivative sens.)   [0.5, 10.0]
#    threshold_amr  : Recursion decision threshold             [0.1, 1.0]
#
#  Tier 2 — Hamiltonian f×g×Mic gates (Phase 2):
#    gamma_hydro    : f() log-growth rate for Re               [0.1, 2.5]
#    gamma_mag      : f() log-growth rate for Rm               [0.1, 2.5]
#    beta_michelson : Mic() contrast sensitivity               [0.1, 1.0]
#    kappa          : g() leaky sigmoid steepness              [1.0, 15.0]
#    w_z_frac       : Adaptive Z bias weight (fraction of max|C|,|K|)  [0.05, 0.5]
#
#  Key design choices:
#  - Absolute physical non-dimensionalization (no domain-max normalization)
#  - Leaky sigmoid gates (g_strain, g_rot, g_mag) replace exp(α×Σsᵢ)
#  - X-point reconnection detector via det(J_B) ZZZZ plaquette term
#  - 1D compression for compress term
#  - K_opt = 30: COBYLA needs n+1=5 evals for the initial simplex
#    (4 QAOA params). 30 iterations gives ~25 genuine optimization steps.
# ============================================================

PHASES = {
    "phase1_encoding": {
        "N": N_TRAINING,           
        "max_depth_override": MAX_DEPTH_TRAINING,
        "T_MAX": 3.0,
        "T_START": 2.0,
        "DT": 1e-3,
        "HYBRID_DT": 0.10,
        "K_opt": 30,               
        "Re": 400,
        "Rm": 400,
        "shots": 128,
        "n_trials": 200,
        "train_hamiltonian": False,
        "AdvAnomaliesEnable": False,
        "study_name": "q_has_phase1",
    },
    "phase2_hamiltonian": {
        "N": N_TRAINING,            
        "max_depth_override": MAX_DEPTH_TRAINING,  
        "T_MAX": 3.0,
        "T_START": 2.0,
        "DT": 1e-3,
        "HYBRID_DT": 0.10,
        "K_opt": 30,
        "Re": 400,            
        "Rm": 400,
        "shots": 256,
        "n_trials": 400,
        "train_hamiltonian": True,
        "AdvAnomaliesEnable": False,
        "study_name": "q_has_phase2",
    },
    "phase3_anomalies": {
        "N": N_TRAINING,            
        "max_depth_override": MAX_DEPTH_TRAINING,  
        "T_MAX": 3.0,
        "T_START": 1.5,
        "DT": 1e-3,
        "HYBRID_DT": 0.10,
        "K_opt": 30,                
        "Re": 800,
        "Rm": 800,
        "shots": 256,
        "n_trials": 600,
        "train_hamiltonian": True,
        "AdvAnomaliesEnable": True,
        "study_name": "q_has_phase3",
    },
}

def create_argus(phase_config):
    """Build the argus namespace for a given training phase."""
    return SimpleNamespace(
        reps=2,
        mode="simulator",
        backend="aer",
        shots=phase_config["shots"],
        method="COBYLA",
        opt_level=1,
        AdvAnomaliesEnable=phase_config["AdvAnomaliesEnable"],
        K_opt=phase_config["K_opt"],
        eps=1e-2,
        eta=0.001,
        Bz_guide=0.1,
        c_s=1.0,
        Re=phase_config["Re"],
        Rm=phase_config["Rm"],
    )


def make_objective(phase_config, frozen_params=None, dns_trace=None, hot_start_state=None):
    """
    Factory: creates an Optuna objective closure for a specific phase.

    frozen_params: dict of hyperparams that are FIXED (not optimized).
        These override any suggested values — the suggest_float calls
        are skipped entirely for frozen keys.
    dns_trace: precomputed DNS trajectory (from precompute_dns).
    hot_start_state: initial state for trials (from precompute_dns).
    """
    N = phase_config["N"]
    T_MAX = phase_config["T_MAX"]
    DT = phase_config["DT"]
    HYBRID = int(phase_config["HYBRID_DT"] / DT)
    VQA_N = 2
    min_patch_size = phase_config.get("min_patch_size", 6)
    max_depth_override = phase_config.get("max_depth_override", None)
    argus = create_argus(phase_config)
    frozen = frozen_params or {}
    train_hamiltonian = phase_config.get("train_hamiltonian", False)

    def objective(trial):
        HyperParams = {}

        # ── Tier 1: Encoding + AMR ──
        if "beta" not in frozen:
            HyperParams["beta"] = trial.suggest_float("beta", 0.5, 10.0, log=True)
        if "threshold_amr" not in frozen:
            HyperParams["threshold_amr"] = trial.suggest_float("threshold_amr", 0.1, 1.0)

        # ── Tier 2: Hamiltonian f×g×Mic gates ──
        if train_hamiltonian:
            if "gamma_hydro" not in frozen:
                HyperParams["gamma_hydro"] = trial.suggest_float("gamma_hydro", 0.1, 2.5)
            if "gamma_mag" not in frozen:
                HyperParams["gamma_mag"] = trial.suggest_float("gamma_mag", 0.1, 2.5)
            if "beta_michelson" not in frozen:
                HyperParams["beta_michelson"] = trial.suggest_float("beta_michelson", 0.1, 1.0)
            if "kappa" not in frozen:
                HyperParams["kappa"] = trial.suggest_float("kappa", 1.0, 15.0)
            if "w_z_frac" not in frozen:
                HyperParams["w_z_frac"] = trial.suggest_float("w_z_frac", 0.05, 0.5)

        # Inject frozen params (these ALWAYS win over suggested values)
        for k, v in frozen.items():
            HyperParams[k] = v

        try:
            result = pipeline(
                N=N,
                VQA_N=VQA_N,
                T_MAX=T_MAX,
                DT=DT,
                HYBRID=HYBRID,
                verbose=False,
                argus=argus,
                hyperparams=HyperParams,
                lambda_cost=LAMBDA_COST,
                trial=trial,
                dns_trace=dns_trace,
                hot_start_state=hot_start_state,
                min_patch_size=min_patch_size,
                max_depth_override=max_depth_override,
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"[Trial {trial.number}] FAILED: {e}")
            return 10.0  # Finite penalty instead of inf

        if np.isnan(result) or np.isinf(result):
            return 10.0  # Finite penalty instead of inf

        return result

    return objective


def _get_storage(phase_config):
    """Return the Optuna storage backend for this phase."""
    if DISTRIBUTED:
        # Instead of returning the raw string, we configure the SQLAlchemy engine
        return optuna.storages.RDBStorage(
            url=OPTUNA_STORAGE,
            engine_kwargs={
                "pool_pre_ping": True,  # Forces SQLAlchemy to check if connection is alive and reconnect if dead
                "pool_recycle": 300,    # Recycle connections every 5 minutes to beat Neon's idle timeout
                "pool_size": 1,         # Strict limit: 1 primary connection per Colab worker
                "max_overflow": 1       # Strict limit: Max 1 backup connection
            }
        )
    if JOURNAL_DIR is not None:
        journal_path = os.path.join(JOURNAL_DIR, f"{phase_config['study_name']}.log")
        lock = optuna.storages.JournalFileOpenLock(journal_path)
        return optuna.storages.JournalStorage(
            optuna.storages.JournalFileBackend(journal_path, lock_obj=lock)
        )
    db_path = os.path.join(data_dir, f"{phase_config['study_name']}.db")
    return f"sqlite:///{db_path}"


def run_phase(phase_name, phase_config, seed_params=None, frozen_params=None, dns_trace=None, hot_start_state=None):
    """Run one training phase with Optuna MedianPruner.

    Supports distributed execution: multiple workers can call this
    function concurrently on different machines. Optuna coordinates
    trial distribution via the shared storage backend.
    """
    storage = _get_storage(phase_config)

    # Pruning: more conservative startup to avoid premature pruning
    # based on a biased baseline from seed trials.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=15,  # Let 15 trials complete before pruning (was 5)
        n_warmup_steps=2,
        interval_steps=1,
        n_min_trials=5,       # Reliable median needs >= 5 completed (was 3)
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

    objective = make_objective(phase_config, frozen_params=frozen_params,
                               dns_trace=dns_trace, hot_start_state=hot_start_state)

    # Callback: save to Drive periodically (Colab only, local SQLite only)
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

    # In distributed mode, cap trials per worker
    if WORKER_TRIALS is not None:
        remaining_trials = min(remaining_trials, WORKER_TRIALS)

    if remaining_trials > 0:
        if trials_done > 0:
            print(f"Reprise de la phase '{phase_name}' : {trials_done} trials existants. "
                  f"Ce worker va en faire {remaining_trials}.")
        else:
            print(f"Lancement de la phase '{phase_name}' : {target_trials} trials à faire. "
                  f"Ce worker va en faire {remaining_trials}.")
        study.optimize(objective, n_trials=remaining_trials, callbacks=[callback_save])
    else:
        print(f"Phase '{phase_name}' déjà terminée ({trials_done}/{target_trials} trials).")

    return study


def extract_top_params(study, top_k=10):
    """Extract the top_k best parameter sets from a completed study."""
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value < float("inf")
    ]
    completed.sort(key=lambda t: t.value)
    return [t.params for t in completed[:top_k]]


def _run_phase1(dns_trace, hot_start_state):
    """Phase 1: Encoding + AMR Calibration — beta, threshold_amr."""
    print("=" * 60)
    print("PHASE 1: Encoding + AMR Calibration")
    print("  Training: beta, threshold_amr")
    print("=" * 60)

    grid_beta = [1.0, 4.0, 8.0]
    grid_threshold_amr = [0.2, 0.5]

    initial_seeds = []
    for b, t in itertools.product(grid_beta, grid_threshold_amr):
        initial_seeds.append({"beta": b, "threshold_amr": t})

    study = run_phase(
        "phase1_encoding", PHASES["phase1_encoding"],
        seed_params=initial_seeds,
        dns_trace=dns_trace, hot_start_state=hot_start_state,
    )

    print(f"\nPhase 1 — Best score: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    return study


def _run_phase2(study_p1, dns_trace, hot_start_state):
    """Phase 2: Hamiltonian Weights — w_data, w_shear, w_vortex, etc."""
    print("\n" + "=" * 60)
    print("PHASE 2: Hamiltonian Training")
    print("  Training: gamma_hydro, gamma_mag, beta_michelson, kappa")
    print("  Retraining: beta, threshold_amr")
    print("=" * 60)

    top_params_p1 = extract_top_params(study_p1, top_k=5)

    if len(top_params_p1) == 0:
        print("[FATAL] No trial succeeded in Phase 1. Check pipeline for bugs.")
        sys.exit(1)

    grid_gamma_hydro = [0.3, 1.0, 2.0]
    grid_gamma_mag = [0.3, 1.0, 2.0]
    grid_kappa = [3.0, 8.0]

    hamilt_grid_seeds = []
    for gh, gm, k in itertools.product(grid_gamma_hydro, grid_gamma_mag, grid_kappa):
        hamilt_grid_seeds.append({"gamma_hydro": gh, "gamma_mag": gm, "kappa": k})

    hamilt_seeds = [{**p, **h} for p in top_params_p1 for h in hamilt_grid_seeds]

    study = run_phase(
        "phase2_hamiltonian",
        PHASES["phase2_hamiltonian"].copy(),
        seed_params=hamilt_seeds,
        dns_trace=dns_trace, hot_start_state=hot_start_state,
    )

    print(f"\nPhase 2 — Best score: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    return study


def _run_phase3(study_p2):
    """Phase 3: Advanced Anomalies (higher Re/Rm) — beta_xpoint + retrain all."""
    dns_trace_p3, hot_start_p3 = precompute_dns(PHASES["phase3_anomalies"])

    print("\n" + "=" * 60)
    print("PHASE 3: Advanced Anomalies (higher Re/Rm)")
    print("  Retraining: all Tier 1+2 params")
    print("=" * 60)

    top_params_p2 = extract_top_params(study_p2, top_k=15)
    anomaly_seeds = top_params_p2

    study = run_phase(
        "phase3_anomalies",
        PHASES["phase3_anomalies"].copy(),
        seed_params=anomaly_seeds,
        dns_trace=dns_trace_p3, hot_start_state=hot_start_p3,
    )

    print(f"\nPhase 3 — Best score: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    return study


def _save_results(study_p1, study_p2, study_p3):
    """Save final results to JSON."""
    final_params = {**study_p3.best_params}
    output_path = os.path.join(data_dir, "best_hyperparams.json")
    results = {
        "best_score": study_p3.best_value,
        "best_params": final_params,
        "n_trials_phase1": len(study_p1.trials),
        "n_trials_phase2": len(study_p2.trials),
        "n_trials_phase3": len(study_p3.trials),
        "phase1_best_score": study_p1.best_value,
        "phase1_best_params": study_p1.best_params,
        "phase2_best_score": study_p2.best_value,
        "phase2_best_params": study_p2.best_params,
        "phase3_best_score": study_p3.best_value,
        "phase3_best_params": study_p3.best_params,
    }
    print(f"\nBEST COMBINED SCORE: {study_p3.best_value:.6f}")
    print(f"BEST PARAMS (all): {final_params}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        print(f"\nFinal results saved to {output_path}")

    if IN_COLAB:
        print("\nCopie des resultats finaux vers Google Drive en cours...")
        try:
            shutil.copytree(local_dir, drive_dir, dirs_exist_ok=True)
            print(f"Succes ! Donnees sauvegardees sur Drive : {drive_dir}")
        except Exception as e:
            print(f"Erreur lors de la copie vers Drive : {e}")


if __name__ == "__main__":

    # ── Distributed single-phase worker mode ──
    # WORKER_PHASE="1" → only run phase 1 (can launch N workers in parallel)
    # WORKER_PHASE="2" → only run phase 2 (reads phase 1 results from shared DB)
    # WORKER_PHASE="3" → only run phase 3 (reads phase 2 results from shared DB)
    # WORKER_PHASE=None → sequential: phase 1 → 2 → 3 (original behavior)

    if WORKER_PHASE == "1":
        dns_trace, hot_start_state = precompute_dns(PHASES["phase1_encoding"])
        _run_phase1(dns_trace, hot_start_state)

    elif WORKER_PHASE == "2":
        dns_trace, hot_start_state = precompute_dns(PHASES["phase1_encoding"])
        # Load phase 1 study to extract seeds
        storage = _get_storage(PHASES["phase1_encoding"])
        study_p1 = optuna.load_study(
            study_name=PHASES["phase1_encoding"]["study_name"],
            storage=storage,
        )
        _run_phase2(study_p1, dns_trace, hot_start_state)

    elif WORKER_PHASE == "3":
        # Load phase 2 study to extract seeds
        storage = _get_storage(PHASES["phase2_hamiltonian"])
        study_p2 = optuna.load_study(
            study_name=PHASES["phase2_hamiltonian"]["study_name"],
            storage=storage,
        )
        _run_phase3(study_p2)

    else:
        # ── Full sequential run (original behavior) ──
        dns_trace, hot_start_state = precompute_dns(PHASES["phase1_encoding"])

        study_p1 = _run_phase1(dns_trace, hot_start_state)

        top_p1 = extract_top_params(study_p1, top_k=5)
        if len(top_p1) == 0:
            print("[FATAL] No trial succeeded in Phase 1. Check pipeline for bugs.")
            sys.exit(1)

        study_p2 = _run_phase2(study_p1, dns_trace, hot_start_state)
        study_p3 = _run_phase3(study_p2)
        _save_results(study_p1, study_p2, study_p3)
