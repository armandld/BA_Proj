"""
TrainHyperParam_v2 — Multi-Scenario Progressive Training
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

Training phases:
----------------
    Phase 1  : Composite loss on 4 isolated scenarios, 7 params
               (beta, threshold_amr, beta_thr, gamma_hydro, gamma_mag, kappa, w_z_frac)
               All terms share a single β_thr sensitivity.
    
    Phase 1b : Same 4 isolated scenarios, 9 params (split β).
               Decouples β_grad, β_curl, β_xpoint on known terrain.
               Seeded from Phase 1 best trials.
    
    Phase 2  : Composite loss on OT + Rotor (complex scenarios), 9 params
               Validates split-β on multi-anomaly scenarios.
               Seeded from Phase 1b best trials.
    
    Phase 3  : Composite loss on ALL 6 scenarios (retrain all 9 params)
               Validates that split sensitivities generalize across
               all anomaly types. Seeded from Phase 2 best trials.

    Classical Phase 1 (*) : Classical AMR on 4 isolated scenarios.
               Trains threshold_amr only (no quantum circuit — ~100x faster).

    Classical Phase 2 (**): Classical AMR on OT + Rotor.
               Validates classical threshold on complex multi-anomaly scenarios.

    Classical Phase 3 (***): Classical AMR on ALL 6 scenarios.
               Validates classical threshold across all anomaly types.
               Seeded from Classical Phase 2 best trials.
               Mirrors Q-HAS Phase 3 for fair comparison.
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
#  PHASE DEFINITIONS (v2 — 3-phase progressive split-β)
# ============================================================

PHASES = {
    "phase1_composite": {
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

    "phase2_complex": {
        # Phase 2: Composite loss on OT + Rotor with split Michelson
        # sensitivity (β_grad, β_curl, β_xpoint — 9 trainable params).
        # These complex scenarios mix ALL anomaly types simultaneously,
        # validating that split sensitivities generalize beyond isolated cases.
        # Seeded with top results from Phase 1b (already split β).
        "n_trials": 600,
        "train_hamiltonian": True,
        "split_michelson": True,       # Split β per term type
        "AdvAnomaliesEnable": True,
        "study_name": "q_has_v2_phase2",
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

    "phase3_validation": {
        # Phase 3: Composite loss on ALL 6 scenarios (4 isolated + 2 complex).
        # Validates that split sensitivities generalize across all anomaly types.
        # 9 trainable params (split β).
        # Seeded with top results from Phase 2.
        "n_trials": 400,
        "train_hamiltonian": True,
        "split_michelson": True,
        "AdvAnomaliesEnable": True,
        "study_name": "q_has_v2_phase3",
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

    "classical_phase2": {
        # Phase **: Classical AMR on OT + Rotor.
        # Validates classical threshold on complex multi-anomaly scenarios.
        # Seeded with top results from classical Phase 1.
        "n_trials": 300,
        "train_hamiltonian": False,
        "split_michelson": False,
        "classical_only": True,
        "AdvAnomaliesEnable": True,
        "study_name": "classical_v2_phase2",
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

    "classical_phase3": {
        # Phase ***: Classical AMR on ALL 6 scenarios.
        # Validates classical threshold across all anomaly types.
        # Seeded with top results from classical Phase 2.
        "n_trials": 300,
        "train_hamiltonian": False,
        "split_michelson": False,
        "classical_only": True,
        "AdvAnomaliesEnable": True,
        "study_name": "classical_v2_phase3",
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


def expand_split_beta_seeds(base_params_list):
    """
    Expand Phase 1 parameter sets into Phase 1b seeds with 4 split combos.

    For each base trial, beta_michelson is replaced by (sigma, beta_curl,
    beta_xpoint) using SPLIT_BETA_STRATEGIES.

    Parameters
    ----------
    base_params_list : list of dict
        Parameter dicts containing 'beta_michelson' (from Phase 1 rescore).

    Returns
    -------
    list of dict : expanded seeds (len = len(base_params_list) * 4)
    """
    seeds = []
    for params in base_params_list:
        bm = params.pop("beta_michelson", params.get("beta_michelson", 0.5))
        # Remove beta_michelson from the dict if still present
        base = {k: v for k, v in params.items() if k != "beta_michelson"}

        for strat in SPLIT_BETA_STRATEGIES:
            seed = {
                **base,
                "sigma":      strat["sigma"],
                "beta_curl":  np.clip(bm * strat["curl"],  0.1, 2.0),
                "beta_xpoint": np.clip(bm * strat["xpoint"], 0.1, 2.0),
            }
            seeds.append(seed)

    print(f"[SPLIT-β EXPANSION] {len(base_params_list)} base trials "
          f"× {len(SPLIT_BETA_STRATEGIES)} strategies = {len(seeds)} seeds")
    return seeds


# ============================================================
#  COMPOSITE MULTI-SCENARIO OBJECTIVE (QAOA — Phase 1 & 2)
# ============================================================

# ══════════════════════════════════════════════════════════════════════
#  Constantes de l'objectif — ce qui N'EST PAS explore
# ══════════════════════════════════════════════════════════════════════
#
# Ces quatre valeurs etaient ecrites en dur a l'interieur de
# `make_composite_objective`, sous une forme (`if "x" not in frozen:`) qui
# les faisait passer pour des parametres conditionnels. Elles sont
# nommees ici pour que la difference entre « fixe » et « explore » soit
# lisible depuis l'exterieur.

#: Meilleur essai de l'etude classique gelee (#42, perte 0.2148).
CLASSICAL_BEST_THRESHOLD = 0.14959824837662078
FIXED_GAMMA_HYDRO = 2.0
FIXED_GAMMA_MAG = 0.5
FIXED_KAPPA = 10.0

#: Les parametres reellement proposes a Optuna, par phase.
SEARCH_SPACE = {
    "phase1": ("beta", "w_z_frac", "beta_michelson"),
    "phase2+": ("beta", "w_z_frac", "sigma", "beta_curl", "beta_xpoint"),
}

#: Les parametres que l'objectif FIXE, avec leur valeur.
FIXED_PARAMS = {
    "threshold_amr": CLASSICAL_BEST_THRESHOLD,
    "gamma_hydro": FIXED_GAMMA_HYDRO,
    "gamma_mag": FIXED_GAMMA_MAG,
    "kappa": FIXED_KAPPA,
}


def search_space(split_michelson):
    """Les noms que `make_composite_objective` proposera reellement.

    Une campagne qui croit optimiser `kappa` ne le fait pas : cette
    fonction permet de le verifier avant de lancer, plutot que de le
    decouvrir en relisant la base a posteriori.
    """
    return SEARCH_SPACE["phase2+"] if split_michelson else SEARCH_SPACE["phase1"]


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

        # ── Tier 1: Encoding + AMR ──
        if "beta" not in frozen:
            HyperParams["beta"] = trial.suggest_float("beta", 0.5, 10.0)

        # threshold_amr, gamma_hydro, gamma_mag et kappa sont des CONSTANTES,
        # pas des parametres explores. La forme `if "x" not in frozen:` les
        # faisait passer pour conditionnels : l'espace de recherche reel
        # compte cinq parametres (beta, w_z_frac, sigma, beta_curl,
        # beta_xpoint), pas neuf.
        #
        # `search_space()` plus bas rend cette liste consultable par un
        # appelant, pour qu'aucune campagne ne puisse plus croire optimiser
        # ce qu'elle fixe.
        if "threshold_amr" not in frozen:
            # Meilleur essai de l'etude classique (#42, perte 0.2148) : gele
            # pour que la comparaison porte sur ce que le quantique ajoute et
            # non sur un seuil different. Verifie contre la base dans
            # tests/test_hyperparams_provenance_break.py.
            HyperParams["threshold_amr"] = CLASSICAL_BEST_THRESHOLD

        # ── Tier 2: Hamiltonian gates (constantes, non explorees) ──
        if "gamma_hydro" not in frozen:
            HyperParams["gamma_hydro"] = FIXED_GAMMA_HYDRO
        if "gamma_mag" not in frozen:
            HyperParams["gamma_mag"] = FIXED_GAMMA_MAG
        if "kappa" not in frozen:
            HyperParams["kappa"] = FIXED_KAPPA
        if "w_z_frac" not in frozen:
            HyperParams["w_z_frac"] = trial.suggest_float("w_z_frac", 0.10, 1000, log=True)

        # ── Threshold-contrast sensitivity ──
        if split_michelson:
            if "sigma" not in frozen:
                HyperParams["sigma"] = trial.suggest_float("sigma", 0.02, 0.30)
            if "beta_curl" not in frozen:
                HyperParams["beta_curl"] = trial.suggest_float("beta_curl", 0.0, 5.0)
            if "beta_xpoint" not in frozen:
                HyperParams["beta_xpoint"] = trial.suggest_float("beta_xpoint", 0.0, 5.0)
        else:
            if "beta_michelson" not in frozen:
                HyperParams["beta_michelson"] = trial.suggest_float("beta_michelson", 0.1, 2.0)

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
#  PHASE 3: VALIDATION + FINE-TUNE ON ORSZAG-TANG
# ============================================================

def make_phase3_objective(frozen_params, dns_trace, hot_start_state, lambda_cost= LAMBDA_COST_SOFT):
    """Phase 3: Fine-tune all 9 params (split β + w_z_frac) on Orszag-Tang."""
    phase_config = PHASES["phase3_validation"]
    N = phase_config["N"]
    T_MAX = phase_config["T_MAX"]
    DT = phase_config["DT"]
    HYBRID = int(phase_config["HYBRID_DT"] / DT)
    VQA_N = 2
    min_patch_size = phase_config.get("min_patch_size", 6)
    max_depth_override = phase_config.get("max_depth_override", None)
    argus = create_argus(phase_config)
    frozen = frozen_params or {}

    def objective(trial):
        HyperParams = {}

        # ── Tier 1: Encoding + AMR ──
        if "beta" not in frozen:
            HyperParams["beta"] = trial.suggest_float("beta", 0.5, 10.0)
        if "threshold_amr" not in frozen:
            HyperParams["threshold_amr"] = 0.14959824837662078

        # ── Tier 2: Hamiltonian gates ──
        if "gamma_hydro" not in frozen:
            HyperParams["gamma_hydro"] = 2.0
        if "gamma_mag" not in frozen:
            HyperParams["gamma_mag"] = 0.5
        if "kappa" not in frozen:
            HyperParams["kappa"] = 10.0
        if "w_z_frac" not in frozen:
            HyperParams["w_z_frac"] = trial.suggest_float("w_z_frac", 0.10, 100, log=True)

        # ── Threshold-contrast sensitivity ──
        if "sigma" not in frozen:
            HyperParams["sigma"] = trial.suggest_float("sigma", 0.02, 0.30)
        if "beta_curl" not in frozen:
            HyperParams["beta_curl"] = trial.suggest_float("beta_curl", 0.1, 5.0)
        if "beta_xpoint" not in frozen:
            HyperParams["beta_xpoint"] = trial.suggest_float("beta_xpoint", 0.1, 5.0)

        for k, v in frozen.items():
            HyperParams[k] = v

        try:
            result = pipeline(
                N=N, VQA_N=VQA_N, T_MAX=T_MAX, DT=DT, HYBRID=HYBRID,
                verbose=False, argus=argus, hyperparams=HyperParams,
                lambda_cost=lambda_cost, trial=trial,
                dns_trace=dns_trace, hot_start_state=hot_start_state,
                min_patch_size=min_patch_size,
                max_depth_override=max_depth_override,
                scenario="orszag_tang",
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


# ============================================================
#  ORCHESTRATION
# ============================================================

SCENARIOS_ISOLATED = [
    ("kh",      SCENARIO_KH),
    ("ot",  SCENARIO_OT),
    ("tearing", SCENARIO_TEARING),
    ("rotor", SCENARIO_ROTOR),
]

SCENARIOS_COMPLEX = [
    ("ot",    SCENARIO_OT),
    ("rotor", SCENARIO_ROTOR),
]

SCENARIOS_ALL = SCENARIOS_ISOLATED + SCENARIOS_COMPLEX


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
    return _precompute_dns_for(SCENARIOS_ISOLATED, label="4 isolated scenarios")


def _run_phase1(dns_traces):
    """Phase 1: Composite training with shared β_michelson (7 params)."""
    print("=" * 60)
    print("PHASE 1: Composite Training (shared Michelson)")
    print("  Training: beta, threshold_amr, beta_michelson, gamma_hydro, gamma_mag, kappa, w_z_frac")
    print("  Scenarios: KH + OT + Tearing + Rotor")
    print("=" * 60)

    grid_w_z_frac = [500.0]
    grid_beta_curl = [0.1]
    grid_beta_xpoint = [0.1]     
    grid_sigma = [0.10]
    grid_beta = [0.7]

    initial_seeds = []
    for b, sg, bc, bx, wz in itertools.product(
        grid_beta, grid_sigma, grid_beta_curl, grid_beta_xpoint,
        grid_w_z_frac,
    ):
        initial_seeds.append({
            "beta": b,
            "sigma": sg, "beta_curl": bc, "beta_xpoint": bx,
            "w_z_frac": wz,
        })

    objective = make_composite_objective(dns_traces, SCENARIOS_ISOLATED, split_michelson=True)
    study = run_phase("phase1_composite", PHASES["phase1_composite"],
                      objective, seed_params=initial_seeds)

    print(f"\nPhase 1 — Best composite score: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    best_trial = study.best_trial
    for key in ["kh", "ot", "tearing", "rotor"]:
        loss_key = f"loss_{key}"
        if loss_key in best_trial.user_attrs:
            print(f"  {key:>16}: {best_trial.user_attrs[loss_key]:.6f}")

    return study


RESCORE_LAMBDAS_SOFT = [0.4]

def _run_phase2(study_p1b):
    """Phase 2: Split-β training on OT + Rotor (complex scenarios, 9 params).

    Seeding strategy:
      - If rescore CSVs exist (rescore_q_has_v2_phase1b_lambda*/),
        use top 2 trials per lambda as seeds (already split-β).
      - Otherwise, fall back to top 15 from Phase 1b Optuna study.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Composite Training (split Michelson) on OT + Rotor")
    print("  Training: beta, threshold_amr, sigma, beta_curl, beta_xpoint,")
    print("            gamma_hydro, gamma_mag, kappa, w_z_frac")
    print("  Scenarios: Orszag-Tang + MHD Rotor")
    print("=" * 60)

    # Pre-compute DNS for the complex scenarios
    dns_traces_complex = _precompute_dns_for(SCENARIOS_COMPLEX, label="OT + Rotor")

    # Try rescore-based seeding first (top 2 per lambda, already split-β)
    rescore_seeds = extract_top_params_from_rescore(
        "q_has_v2_phase1b", RESCORE_LAMBDAS_SOFT, top_k=20
    )

    if len(rescore_seeds) > 0:
        print(f"\n[PHASE 2] Using rescore-based seeding: {len(rescore_seeds)} seeds")
        seed_params = rescore_seeds
    else:
        # Fallback: original behavior from Optuna study
        print("[PHASE 2] No rescore CSVs found, falling back to Optuna study seeding")
        seed_params = extract_top_params(study_p1b, top_k=20)
        if len(seed_params) == 0:
            print("[FATAL] No trial succeeded in Phase 1b.")
            sys.exit(1)

    objective = make_composite_objective(dns_traces_complex, SCENARIOS_COMPLEX, split_michelson=True)
    study = run_phase("phase2_complex", PHASES["phase2_complex"],
                      objective, seed_params=seed_params)

    print(f"\nPhase 2 — Best composite score: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    best_trial = study.best_trial
    for key in ["ot", "rotor"]:
        loss_key = f"loss_{key}"
        if loss_key in best_trial.user_attrs:
            print(f"  {key:>16}: {best_trial.user_attrs[loss_key]:.6f}")

    return study


def _run_phase3(study_p2):
    """Phase 3: Composite training on ALL 6 scenarios (split β, 9 params).
    Seeded from Phase 2 best trials."""
    print("\n" + "=" * 60)
    print("PHASE 3: Composite Training on ALL 6 scenarios")
    print("  Training: all 9 params (split Michelson + w_z_frac)")
    print("  Scenarios: KH + Vortex + Tearing + Coalescence + OT + Rotor")
    print("=" * 60)

    config_p3 = PHASES["phase3_validation"]

    # Pre-compute DNS for all 6 scenarios
    dns_traces_all = _precompute_dns_for(SCENARIOS_ALL, label="All 6 scenarios")

    # Seed from Phase 2 best trials (rescore CSV first, Optuna fallback)
    rescore_seeds = extract_top_params_from_rescore(
        "q_has_v2_phase2", RESCORE_LAMBDAS_SOFT, top_k=15
    )
    if len(rescore_seeds) > 0:
        print(f"\n[PHASE 3] Using rescore-based seeding: {len(rescore_seeds)} seeds")
        top_params = rescore_seeds
    else:
        print("[PHASE 3] No rescore CSVs found, falling back to Optuna study seeding")
        top_params = extract_top_params(study_p2, top_k=15)

    objective = make_composite_objective(dns_traces_all, SCENARIOS_ALL, split_michelson=True)
    study = run_phase("phase3_validation", config_p3,
                      objective, seed_params=top_params)

    print(f"\nPhase 3 — Best composite score: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    best_trial = study.best_trial
    for key in ["kh", "vortex", "tearing", "coalescence", "ot", "rotor"]:
        loss_key = f"loss_{key}"
        if loss_key in best_trial.user_attrs:
            print(f"  {key:>16}: {best_trial.user_attrs[loss_key]:.6f}")

    return study


def _run_classical_phase1(dns_traces):
    """Classical Phase * : threshold_amr on 4 isolated scenarios."""
    print("\n" + "=" * 60)
    print("CLASSICAL PHASE 1 (*): Training threshold_amr")
    print("  Training: threshold_amr (1 param)")
    print("  Scenarios: KH + OT + Tearing + Rotor")
    print("  Method: classical AMR (no quantum circuit)")
    print("=" * 60)

    # Grid-search seeds for the single parameter
    initial_seeds = [{"threshold_amr": t} for t in
                     np.linspace(0.05, 0.8, 20).tolist()]

    objective = make_classical_composite_objective(dns_traces, SCENARIOS_ISOLATED)
    study = run_phase("classical_phase1", PHASES["classical_phase1"],
                      objective, seed_params=initial_seeds)

    print(f"\nClassical Phase 1 — Best composite score: {study.best_value:.6f}")
    print(f"Best threshold_amr: {study.best_params}")

    best_trial = study.best_trial
    for key in ["kh", "ot", "tearing", "rotor"]:
        loss_key = f"loss_{key}"
        if loss_key in best_trial.user_attrs:
            print(f"  {key:>16}: {best_trial.user_attrs[loss_key]:.6f}")

    return study

def _run_classical_phase2(study_c1):
    """Classical Phase ** : threshold_amr on OT + Rotor.

    Seeding strategy:
      - If rescore CSVs exist (rescore_classical_v2_phase1_lambda*/),
        use top 2 trials per lambda as seeds.
      - Otherwise, fall back to top 15 from classical Phase 1 Optuna study.
    """
    print("\n" + "=" * 60)
    print("CLASSICAL PHASE 2 (**): Training threshold_amr on OT + Rotor")
    print("  Training: threshold_amr (1 param)")
    print("  Scenarios: Orszag-Tang + MHD Rotor")
    print("  Method: classical AMR (no quantum circuit)")
    print("=" * 60)

    dns_traces_complex = _precompute_dns_for(SCENARIOS_COMPLEX, label="OT + Rotor (classical)")

    # Try rescore-based seeding first
    rescore_seeds = extract_top_params_from_rescore(
        "classical_v2_phase1", RESCORE_LAMBDAS_SOFT, top_k=2
    )

    if len(rescore_seeds) > 0:
        print(f"\n[CLASSICAL PHASE 2] Using rescore-based seeding: {len(rescore_seeds)} seeds")
        seed_params = rescore_seeds
    else:
        # Fallback: original behavior
        print("[CLASSICAL PHASE 2] No rescore CSVs found, falling back to Optuna study seeding")
        seed_params = extract_top_params(study_c1, top_k=15)
        if len(seed_params) == 0:
            seed_params = [{"threshold_amr": t} for t in
                           np.linspace(0.05, 0.8, 15).tolist()]

    objective = make_classical_composite_objective(dns_traces_complex, SCENARIOS_COMPLEX)
    study = run_phase("classical_phase2", PHASES["classical_phase2"],
                      objective, seed_params=seed_params)

    print(f"\nClassical Phase 2 — Best composite score: {study.best_value:.6f}")
    print(f"Best threshold_amr: {study.best_params}")

    best_trial = study.best_trial
    for key in ["ot", "rotor"]:
        loss_key = f"loss_{key}"
        if loss_key in best_trial.user_attrs:
            print(f"  {key:>16}: {best_trial.user_attrs[loss_key]:.6f}")

    return study

def _run_classical_phase3(study_c2):
    """Classical Phase *** : threshold_amr on ALL 6 scenarios.

    Seeding strategy:
      - If rescore CSVs exist (rescore_classical_v2_phase2_lambda*/),
        use top trials per lambda as seeds.
      - Otherwise, fall back to top 15 from classical Phase 2 Optuna study.
    """
    print("\n" + "=" * 60)
    print("CLASSICAL PHASE 3 (***): Training threshold_amr on ALL 6 scenarios")
    print("  Training: threshold_amr (1 param)")
    print("  Scenarios: KH + Vortex + Tearing + Coalescence + OT + Rotor")
    print("  Method: classical AMR (no quantum circuit)")
    print("=" * 60)

    dns_traces_all = _precompute_dns_for(SCENARIOS_ALL, label="All 6 scenarios (classical)")

    # Try rescore-based seeding first
    rescore_seeds = extract_top_params_from_rescore(
        "classical_v2_phase2", RESCORE_LAMBDAS_SOFT, top_k=15
    )

    if len(rescore_seeds) > 0:
        print(f"\n[CLASSICAL PHASE 3] Using rescore-based seeding: {len(rescore_seeds)} seeds")
        seed_params = rescore_seeds
    else:
        # Fallback: original behavior
        print("[CLASSICAL PHASE 3] No rescore CSVs found, falling back to Optuna study seeding")
        seed_params = extract_top_params(study_c2, top_k=15)
        if len(seed_params) == 0:
            seed_params = [{"threshold_amr": t} for t in
                           np.linspace(0.05, 0.8, 15).tolist()]

    objective = make_classical_composite_objective(dns_traces_all, SCENARIOS_ALL)
    study = run_phase("classical_phase3", PHASES["classical_phase3"],
                      objective, seed_params=seed_params)

    print(f"\nClassical Phase 3 — Best composite score: {study.best_value:.6f}")
    print(f"Best threshold_amr: {study.best_params}")

    best_trial = study.best_trial
    for key in ["kh", "vortex", "tearing", "coalescence", "ot", "rotor"]:
        loss_key = f"loss_{key}"
        if loss_key in best_trial.user_attrs:
            print(f"  {key:>16}: {best_trial.user_attrs[loss_key]:.6f}")

    return study


def _save_results(study_p1, study_p1b, study_p2, study_p3):
    """Save final results to JSON."""
    final_params = {**study_p3.best_params}
    output_path = os.path.join(data_dir, "best_hyperparams_v2.json")
    results = {
        "version": "v2_split_michelson_adaptive_z",
        "best_score": study_p3.best_value,
        "best_params": final_params,
        "n_trials_phase1": len(study_p1.trials),
        "n_trials_phase1b": len(study_p1b.trials),
        "n_trials_phase2": len(study_p2.trials),
        "n_trials_phase3": len(study_p3.trials),
        "phase1_best_score": study_p1.best_value,
        "phase1_best_params": study_p1.best_params,
        "phase1_per_scenario": {
            key: study_p1.best_trial.user_attrs.get(f"loss_{key}", None)
            for key in ["kh", "vortex", "tearing", "coalescence"]
        },
        "phase1b_best_score": study_p1b.best_value,
        "phase1b_best_params": study_p1b.best_params,
        "phase1b_per_scenario": {
            key: study_p1b.best_trial.user_attrs.get(f"loss_{key}", None)
            for key in ["kh", "vortex", "tearing", "coalescence"]
        },
        "phase2_best_score": study_p2.best_value,
        "phase2_best_params": study_p2.best_params,
        "phase2_per_scenario": {
            key: study_p2.best_trial.user_attrs.get(f"loss_{key}", None)
            for key in ["ot", "rotor"]
        },
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


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    if WORKER_PHASE == "1":
        dns_traces = _precompute_composite_dns()
        _run_phase1(dns_traces)

    elif WORKER_PHASE == "2":
        # Load Phase 1b study
        storage = _get_storage(PHASES["phase1_composite"])
        study_p1b = optuna.load_study(
            study_name=PHASES["phase1_composite"]["study_name"],
            storage=storage,
        )
        _run_phase2(study_p1b)
    

    elif WORKER_PHASE == "3":
        # Load Phase 2 study
        storage = _get_storage(PHASES["phase2_complex"])
        study_p2 = optuna.load_study(
            study_name=PHASES["phase2_complex"]["study_name"],
            storage=storage,
        )
        _run_phase3(study_p2)

    elif WORKER_PHASE == "classical_1":
        # Classical Phase *: 4 isolated scenarios
        dns_traces = _precompute_composite_dns()
        _run_classical_phase1(dns_traces)

    elif WORKER_PHASE == "classical_2":
        # Classical Phase **: OT + Rotor (loads Phase * results for seeding)
        storage = _get_storage(PHASES["classical_phase1"])
        study_c1 = optuna.load_study(
            study_name=PHASES["classical_phase1"]["study_name"],
            storage=storage,
        )
        _run_classical_phase2(study_c1)

    elif WORKER_PHASE == "classical_3":
        # Classical Phase ***: ALL 6 scenarios (loads Phase ** results for seeding)
        storage = _get_storage(PHASES["classical_phase2"])
        study_c2 = optuna.load_study(
            study_name=PHASES["classical_phase2"]["study_name"],
            storage=storage,
        )
        _run_classical_phase3(study_c2)

    elif WORKER_PHASE == "classical":
        # All 3 classical phases sequentially
        dns_traces = _precompute_composite_dns()
        study_c1 = _run_classical_phase1(dns_traces)
        study_c2 = _run_classical_phase2(study_c1)
        study_c3 = _run_classical_phase3(study_c2)
        print("\n" + "=" * 60)
        print("CLASSICAL TRAINING COMPLETE")
        print(f"  Phase 1 best threshold_amr: {study_c1.best_params.get('threshold_amr', '?'):.4f}")
        print(f"  Phase 2 best threshold_amr: {study_c2.best_params.get('threshold_amr', '?'):.4f}")
        print(f"  Phase 3 best threshold_amr: {study_c3.best_params.get('threshold_amr', '?'):.4f}")
        print("=" * 60)

    else:
        # Full sequential run (QAOA phases + classical phases)
        dns_traces = _precompute_composite_dns()

        # ── QAOA Phases ──
        study_p1 = _run_phase1(dns_traces)
        top_p1 = extract_top_params(study_p1, top_k=5)
        if len(top_p1) == 0:
            print("[FATAL] No trial succeeded in Phase 1.")
            sys.exit(1)

        study_p1 = _run_phase1(study_p1, dns_traces)
        study_p2 = _run_phase2(study_p1)
        study_p3 = _run_phase3(study_p2)
        _save_results(study_p1, study_p1, study_p2, study_p3)

        # ── Classical Phases ──
        study_c1 = _run_classical_phase1(dns_traces)
        study_c2 = _run_classical_phase2(study_c1)
        study_c3 = _run_classical_phase3(study_c2)

        print("\n" + "=" * 60)
        print("ALL TRAINING COMPLETE")
        print(f"  QAOA best score (Phase 3): {study_p3.best_value:.6f}")
        print(f"  Classical Phase 1 best threshold: {study_c1.best_params.get('threshold_amr', '?'):.4f}")
        print(f"  Classical Phase 2 best threshold: {study_c2.best_params.get('threshold_amr', '?'):.4f}")
        print(f"  Classical Phase 3 best threshold: {study_c3.best_params.get('threshold_amr', '?'):.4f}")
        print("=" * 60)