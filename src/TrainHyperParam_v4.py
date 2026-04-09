"""
TrainHyperParam_v4 — Minimal Proof-of-Concept Training
=======================================================

Goal: simplest possible comparison of Quantum AMR vs Classical AMR.
Per-scenario training → per-scenario evaluation → fair comparison.

Scenarios:
  1. harris_tearing  — X-point selective refinement (quantum advantage: det(J_B))
  2. kelvin_helmholtz — shear layer detection (baseline comparison)

Design:
  - Each scenario trained independently (no composite loss)
  - Classical: 1 param (threshold_amr)
  - Quantum: 6 params (threshold_amr, w_z_frac, sigma, beta_curl, beta_xpoint, kappa)
  - Fast: N=128, short T_MAX, reduced COBYLA iterations

Usage:
  WORKER_PHASE=harris_q    python src/TrainHyperParam_v4.py   # quantum on harris
  WORKER_PHASE=harris_c    python src/TrainHyperParam_v4.py   # classical on harris
  WORKER_PHASE=kh_q        python src/TrainHyperParam_v4.py   # quantum on KH
  WORKER_PHASE=kh_c        python src/TrainHyperParam_v4.py   # classical on KH
  WORKER_PHASE=all         python src/TrainHyperParam_v4.py   # all 4 sequentially
"""

import optuna
import os
import sys
import numpy as np
from pipeline import pipeline
from types import SimpleNamespace
from Simulation.pre_compute_dns import precompute_dns

optuna.logging.set_verbosity(optuna.logging.INFO)
sys.stdout.reconfigure(line_buffering=True)

# ============================================================
#  CONFIGURATION — kept minimal
# ============================================================
LAMBDA_COST = 0.4       # phys_weight * (1-lambda) + patch_ratio * lambda
N_TRAINING = 128        # Grid resolution (fast)
MAX_DEPTH = 4           # Hierarchical depth (decisions happen at depth 2+)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, "Train_results")
os.makedirs(data_dir, exist_ok=True)

# Distributed support
OPTUNA_STORAGE = os.environ.get("OPTUNA_STORAGE", None)
OPTUNA_JOURNAL = os.environ.get("OPTUNA_JOURNAL", None)
WORKER_PHASE   = os.environ.get("WORKER_PHASE", None)
WORKER_TRIALS  = int(os.environ.get("WORKER_TRIALS", "0")) or None

DISTRIBUTED = OPTUNA_STORAGE is not None
if OPTUNA_JOURNAL:
    os.makedirs(OPTUNA_JOURNAL, exist_ok=True)

# ============================================================
#  SCENARIOS
# ============================================================
SCENARIOS = {
    "harris": {
        "scenario": "harris_tearing",
        "N": N_TRAINING,
        "max_depth_override": MAX_DEPTH,
        "T_MAX": 1.0,
        "T_START": 0.2,
        "DT": 1e-3,
        "HYBRID_DT": 0.10,
        "K_opt": 30,
        "Re": 800, "Rm": 800,
        "shots": 256,
        "AdvAnomaliesEnable": True,
    },
    "kh": {
        "scenario": "kelvin_helmholtz",
        "N": N_TRAINING,
        "max_depth_override": MAX_DEPTH,
        "T_MAX": 2.0,
        "T_START": 0.8,
        "DT": 1e-3,
        "HYBRID_DT": 0.10,
        "K_opt": 30,
        "Re": 800, "Rm": 800,
        "shots": 256,
        "AdvAnomaliesEnable": True,
    },
}


# ============================================================
#  HELPERS
# ============================================================
def _make_argus(cfg):
    return SimpleNamespace(
        reps=2, mode="simulator", backend="state_vector",
        shots=cfg["shots"], method="COBYLA", opt_level=1,
        AdvAnomaliesEnable=cfg.get("AdvAnomaliesEnable", False),
        K_opt=cfg["K_opt"], eps=1e-2,
        eta=0.001, Bz_guide=0.1, c_s=1.0,
        Re=cfg["Re"], Rm=cfg["Rm"],
    )


def _get_storage(study_name):
    if DISTRIBUTED:
        return optuna.storages.RDBStorage(
            url=OPTUNA_STORAGE,
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 300},
        )
    if OPTUNA_JOURNAL:
        path = os.path.join(OPTUNA_JOURNAL, f"{study_name}.log")
        lock = optuna.storages.JournalFileOpenLock(path)
        return optuna.storages.JournalStorage(
            optuna.storages.JournalFileBackend(path, lock_obj=lock))
    return f"sqlite:///{os.path.join(data_dir, study_name + '.db')}"


def _precompute(scenario_key):
    cfg = SCENARIOS[scenario_key]
    cfg_dns = {**cfg, "study_name": f"dns_v4_{scenario_key}"}
    dns_trace, hot_start = precompute_dns(cfg_dns)
    return dns_trace, hot_start


# ============================================================
#  QUANTUM OBJECTIVE — 6 free params
# ============================================================
def make_quantum_objective(scenario_key, dns_trace, hot_start):
    cfg = SCENARIOS[scenario_key]
    argus = _make_argus(cfg)
    N = cfg["N"]
    T_MAX = cfg["T_MAX"]
    DT = cfg["DT"]
    HYBRID = int(cfg["HYBRID_DT"] / DT)
    min_patch = cfg.get("min_patch_size", 6)
    max_depth = cfg.get("max_depth_override", None)
    VQA_N = 2

    def objective(trial):
        hp = {
            "threshold_amr": trial.suggest_float("threshold_amr", 0.10, 0.70),
            "beta":          trial.suggest_float("beta", 0.5, 10.0, log=True),
            "w_z_frac":      trial.suggest_float("w_z_frac", 0.1, 100.0, log=True),
            "sigma":         trial.suggest_float("sigma", 0.02, 0.40),
            "gamma_hydro":   trial.suggest_float("gamma_hydro", 0.5, 4.0),
            "gamma_mag":     trial.suggest_float("gamma_mag", 0.1, 4.0),
            "kappa":         trial.suggest_float("kappa", 1.0, 15.0),
            "beta_curl":     trial.suggest_float("beta_curl", 0.1, 5.0, log=True),
            "beta_xpoint":   trial.suggest_float("beta_xpoint", 0.1, 5.0, log=True),
        }

        try:
            result = pipeline(
                N=N, VQA_N=VQA_N, T_MAX=T_MAX, DT=DT, HYBRID=HYBRID,
                verbose=False, argus=argus, hyperparams=hp,
                lambda_cost=LAMBDA_COST, trial=None,
                dns_trace=dns_trace, hot_start_state=hot_start,
                min_patch_size=min_patch, max_depth_override=max_depth,
                scenario=cfg["scenario"], return_details=True,
            )
        except Exception as e:
            print(f"[Q Trial {trial.number}] FAILED: {e}")
            import traceback; traceback.print_exc()
            return 10.0

        combined = result['combined'] if isinstance(result, dict) else result
        if np.isnan(combined) or np.isinf(combined):
            return 10.0

        if isinstance(result, dict):
            trial.set_user_attr("phys_score", float(result.get('phys_score', 0)))
            trial.set_user_attr("patch_ratio", float(result.get('patch_ratio', 0)))
            for field, err in result.get('field_errors', {}).items():
                trial.set_user_attr(f"error_{field}", float(err))

        return combined

    return objective


# ============================================================
#  CLASSICAL OBJECTIVE — 1 free param
# ============================================================
def make_classical_objective(scenario_key, dns_trace, hot_start):
    cfg = SCENARIOS[scenario_key]
    argus = _make_argus(cfg)
    N = cfg["N"]
    T_MAX = cfg["T_MAX"]
    DT = cfg["DT"]
    HYBRID = int(cfg["HYBRID_DT"] / DT)
    min_patch = cfg.get("min_patch_size", 6)
    max_depth = cfg.get("max_depth_override", None)
    VQA_N = 2

    def objective(trial):
        hp = {
            "threshold_amr": trial.suggest_float("threshold_amr", 0.05, 0.80),
        }

        try:
            result = pipeline(
                N=N, VQA_N=VQA_N, T_MAX=T_MAX, DT=DT, HYBRID=HYBRID,
                verbose=False, argus=argus, hyperparams=hp,
                lambda_cost=LAMBDA_COST, trial=None,
                dns_trace=dns_trace, hot_start_state=hot_start,
                min_patch_size=min_patch, max_depth_override=max_depth,
                scenario=cfg["scenario"], return_details=True,
                classical_only=True,
            )
        except Exception as e:
            print(f"[C Trial {trial.number}] FAILED: {e}")
            import traceback; traceback.print_exc()
            return 10.0

        combined = result['combined'] if isinstance(result, dict) else result
        if np.isnan(combined) or np.isinf(combined):
            return 10.0

        if isinstance(result, dict):
            trial.set_user_attr("phys_score", float(result.get('phys_score', 0)))
            trial.set_user_attr("patch_ratio", float(result.get('patch_ratio', 0)))
            for field, err in result.get('field_errors', {}).items():
                trial.set_user_attr(f"error_{field}", float(err))

        return combined

    return objective


# ============================================================
#  TRAINING RUNNER
# ============================================================
def run_study(study_name, objective_fn, n_trials, seed_params=None):
    storage = _get_storage(study_name)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=0, n_min_trials=5)

    study = optuna.create_study(
        study_name=study_name, storage=storage,
        load_if_exists=True, direction="minimize", pruner=pruner,
    )

    if seed_params and len(study.trials) == 0:
        for p in seed_params:
            study.enqueue_trial(p)

    done = len([t for t in study.trials
                if t.state != optuna.trial.TrialState.WAITING])
    remaining = n_trials - done
    if WORKER_TRIALS:
        remaining = min(remaining, WORKER_TRIALS)

    if remaining > 0:
        print(f"  Running {remaining} trials ({done} already done)...")
        study.optimize(objective_fn, n_trials=remaining)
    else:
        print(f"  Already done ({done}/{n_trials} trials).")

    return study


def print_results(study, method_name, scenario_name):
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"  No completed trials for {method_name} on {scenario_name}!")
        return

    print(f"\n{'='*60}")
    print(f"  {method_name} on {scenario_name}")
    print(f"{'='*60}")
    print(f"  Best score:    {study.best_value:.6f}")
    print(f"  Best params:   {study.best_params}")
    bt = study.best_trial
    print(f"  phys_score:    {bt.user_attrs.get('phys_score', '?')}")
    print(f"  patch_ratio:   {bt.user_attrs.get('patch_ratio', '?')}")
    for k, v in bt.user_attrs.items():
        if k.startswith('error_'):
            print(f"  {k}: {v:.6f}")
    print(f"  Total trials:  {len(completed)} completed")


# ============================================================
#  PHASE RUNNERS
# ============================================================
N_TRIALS_Q = 200    # Quantum trials per scenario
N_TRIALS_C = 100    # Classical trials per scenario

def _quantum_seeds():
    """Initial seeds covering the parameter space."""
    seeds = []
    for thr in [0.20, 0.30, 0.40]:
        for wz in [1.0, 5.0, 20.0, 50.0]:
            seeds.append({
                "threshold_amr": thr, "beta": 1.0,
                "w_z_frac": wz, "sigma": 0.10,
                "gamma_hydro": 2.0, "gamma_mag": 0.5,
                "kappa": 10.0, "beta_curl": 1.0, "beta_xpoint": 1.5,
            })
    return seeds


def _classical_seeds():
    return [{"threshold_amr": t}
            for t in np.linspace(0.05, 0.80, 20).tolist()]


def run_scenario(scenario_key, methods=("quantum", "classical")):
    """Run quantum and/or classical training for one scenario."""
    print(f"\n{'#'*60}")
    print(f"# Scenario: {scenario_key}")
    print(f"{'#'*60}")

    dns_trace, hot_start = _precompute(scenario_key)
    studies = {}

    if "quantum" in methods:
        print(f"\n--- Quantum training on {scenario_key} ---")
        obj = make_quantum_objective(scenario_key, dns_trace, hot_start)
        study = run_study(
            f"v4_{scenario_key}_quantum", obj, N_TRIALS_Q,
            seed_params=_quantum_seeds())
        print_results(study, "QUANTUM", scenario_key)
        studies["quantum"] = study

    if "classical" in methods:
        print(f"\n--- Classical training on {scenario_key} ---")
        obj = make_classical_objective(scenario_key, dns_trace, hot_start)
        study = run_study(
            f"v4_{scenario_key}_classical", obj, N_TRIALS_C,
            seed_params=_classical_seeds())
        print_results(study, "CLASSICAL", scenario_key)
        studies["classical"] = study

    # Print head-to-head comparison
    if "quantum" in studies and "classical" in studies:
        q_score = studies["quantum"].best_value
        c_score = studies["classical"].best_value
        delta = q_score - c_score
        print(f"\n{'='*60}")
        print(f"  HEAD-TO-HEAD: {scenario_key}")
        print(f"{'='*60}")
        print(f"  Quantum:   {q_score:.6f}")
        print(f"  Classical: {c_score:.6f}")
        print(f"  Delta:     {delta:+.6f} ({'QUANTUM WINS' if delta < 0 else 'CLASSICAL WINS'})")

        qt = studies["quantum"].best_trial
        ct = studies["classical"].best_trial
        q_phys = qt.user_attrs.get('phys_score', '?')
        c_phys = ct.user_attrs.get('phys_score', '?')
        q_patch = qt.user_attrs.get('patch_ratio', '?')
        c_patch = ct.user_attrs.get('patch_ratio', '?')
        print(f"  Quantum:   phys={q_phys}, patches={q_patch}")
        print(f"  Classical: phys={c_phys}, patches={c_patch}")

    return studies


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    if WORKER_PHASE == "harris_q":
        run_scenario("harris", methods=("quantum",))

    elif WORKER_PHASE == "harris_c":
        run_scenario("harris", methods=("classical",))

    elif WORKER_PHASE == "harris":
        run_scenario("harris")

    elif WORKER_PHASE == "kh_q":
        run_scenario("kh", methods=("quantum",))

    elif WORKER_PHASE == "kh_c":
        run_scenario("kh", methods=("classical",))

    elif WORKER_PHASE == "kh":
        run_scenario("kh")

    elif WORKER_PHASE == "all":
        all_studies = {}
        for sc in ["harris", "kh"]:
            all_studies[sc] = run_scenario(sc)

        print(f"\n{'#'*60}")
        print(f"# FINAL SUMMARY")
        print(f"{'#'*60}")
        for sc, studies in all_studies.items():
            if "quantum" in studies and "classical" in studies:
                q = studies["quantum"].best_value
                c = studies["classical"].best_value
                print(f"  {sc:20s}: Q={q:.6f}  C={c:.6f}  Δ={q-c:+.6f}")

    else:
        print("Usage: WORKER_PHASE=<phase> python TrainHyperParam_v4.py")
        print("Phases: harris_q, harris_c, harris, kh_q, kh_c, kh, all")
        sys.exit(1)
