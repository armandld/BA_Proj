import optuna
import os
import json
from pipeline import pipeline
from types import SimpleNamespace


# Target: 20% grid usage (LAMBDA_COST = 0.2). If we use more, we pay.
LAMBDA_COST = 0.2

N_TRIALS = 100  # Nombre total d'essais d'Optuna

# On récupère le dossier où se trouve le script actuel
current_dir = os.path.dirname(os.path.abspath(__file__))

db_name = "optuna_study.db"

# On remonte d'un niveau pour atteindre la racine (/BA_Proj)
project_root = os.path.dirname(current_dir)

# On pointe vers le dossier data à la racine
data_dir = os.path.join(project_root, "data")

# On s'assure que le dossier existe (au cas où)
os.makedirs(data_dir, exist_ok=True)

# Chemins finaux
db_path = os.path.join(data_dir, db_name)
output_path = os.path.join(data_dir, "best_hyperparams.json")
def objective(trial):

    N=256
    VQA_N=2
    T_MAX=4e-3
    DT=1e-4
    HYBRID_DT = 4e-4
    HYBRID = int(HYBRID_DT / DT)
    verbose=False

    argus_mock = SimpleNamespace(
        depth=1,                # Profondeur quantique par défaut
        mode="simulator",       # Simulation rapide
        backend="aer",          # Backend Qiskit
        shots=1000,
        method="COBYLA",
        opt_level=1,
        AdvAnomaliesEnable=False # On desactive les anomalies avancées pour que l'IA apprenne à les régler
    )

    # 1. Optuna choisit des valeurs
    #State ones
    alpha_try = trial.suggest_float("alpha", 0, 10.0)
    beta_try = trial.suggest_float("beta", 0.5, 10.0)
    thresh_try = trial.suggest_float("threshold", 0.1, 1.0)

    #Hamiltonian ones:
    bias_try=trial.suggest_float("bias", 1.0, 10.0)
    gamma1_try=trial.suggest_float("gamma1", 0.5, 3.0)
    gamma2_try=trial.suggest_float("gamma2", 0.5, 3.0)
    Rm_crit_try=trial.suggest_float("Rm_crit", 100.0, 1000.0)
    """  
    delta_shock_try=trial.suggest_float("delta_shock", 1.0, 10.0)
    d_kink_try=trial.suggest_float("d_kink", 1.0, 5.0)
    epsilon_try=trial.suggest_float("epsilon", 1e-6, 1e-5)
    """

    HyperParams = {
        'alpha': alpha_try,
        'beta': beta_try,
        'threshold': thresh_try,
        'bias': bias_try,
        'gamma1': gamma1_try,
        'gamma2': gamma2_try,
        'Rm_crit': Rm_crit_try
    }
    """
        'delta_shock': delta_shock_try,
        'd_kink': d_kink_try,
        'epsilon': epsilon_try
    """

    print(f"Testing with :")
    print(f"Bias={bias_try:.2f}")
    print(f"Alpha={alpha_try:.2f}")
    print(f"Threshold={thresh_try:.2f}")
    print(f"Bias={bias_try:.2f}")
    print(f"gamma1={gamma1_try:.2f}")
    print(f"gamma2={gamma2_try:.2f}")
    print(f"Rm_crit={Rm_crit_try:.2f}")
    """
    print(f"delta_shock={delta_shock_try:.2f}")
    print(f"d_kink={d_kink_try:.2f}")
    print(f"epsilon={epsilon_try:.2f}")
    """
    print("-----")

    # 2. On lance la simu avec ces valeurs
    try:
        score = pipeline(
            # Params fixes pour aller vite pendant l'entrainement
            N=N,       # Plus petit que 256 pour l'entrainement !
            VQA_N=VQA_N,   # Pas besoin de faire T_MAX=1.0
            T_MAX=T_MAX,
            DT=DT,
            HYBRID=HYBRID,
            verbose=verbose,
            argus=argus_mock,
            hyperparams=HyperParams,
            lambda_cost=LAMBDA_COST
        )
    except Exception as e:
        print(f"Crash avec ces params: {e}")
        return float('inf') # Pire score possible si ça plante

    return score

if __name__ == "__main__":
    # Création de l'étude
    storage_url = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="q_has_v1",
        storage=storage_url,
        load_if_exists=True,
        direction="minimize"
    )
    # --- A. INITIALISATION (WARM START) ---
    # C'est ici qu'on force les valeurs initiales précises
    initial_params = {
        "alpha": 1.0,
        "beta": 1.0,
        "threshold": 0.5,
        "bias": 4.0,
        "gamma1": 1.0,
        "gamma2": 2.0,
        "Rm_crit": 1000.0,

    }
    """
        "delta_shock": 5.0,
        "d_kink": 2.0,
        "epsilon": 1e-6
    """
    print("Injecting initial known parameters...")
    study.enqueue_trial(initial_params)

    # --- B. OPTIMISATION ---
    print("Starting Optimization...")
    # n_trials=50 fera 1 essai avec vos valeurs initiales + 49 essais exploratoires
    study.optimize(objective, n_trials=N_TRIALS)

    # --- C. RÉSULTATS & SAUVEGARDE JSON ---
    print("\n------------------------------------------------")
    print("BEST PARAMS FOUND:", study.best_params)
    print("BEST SCORE:", study.best_value)
    print("------------------------------------------------")

    # Sauvegarde dans un fichier JSON    
    results = {
        "best_score": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials)
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ Résultats sauvegardés dans {output_path}")