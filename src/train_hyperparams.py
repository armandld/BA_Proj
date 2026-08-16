"""
train_hyperparams — Multi-Scenario Progressive Training
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
- sigma:    largeur de la fenetre gaussienne du couplage ZZ autour de
            `threshold_amr` — remplace l'ancien β_grad
- β_curl:  plaquette ZZZZ (vorticity, current density — curl-like quantities)
- β_xpoint: X-point ZZZZ (reconnection — very sparse, localized at X-points)

Espace de recherche — 9 parametres, un seul, identique aux trois phases
----------------------------------------------------------------------
    beta, w_z_frac, sigma, beta_curl, beta_xpoint,
    gamma_hydro, gamma_mag, kappa, relative_percentile

`relative_percentile` est le neuvieme, ajoute apres la correction du
dimensionnement des coefficients : le critere de maille est ABSOLU et
croit en 1/dx^2, si bien qu'a la resolution d'entrainement N=256 aucune
cellule ne l'atteignait et les termes a quatre corps etaient nuls sur les
QUATRE scenarios. Le critere relatif `min(absolu, percentile)` les
ranime ; son percentile etait la derniere constante en dur du chemin de
decision, et rien ne justifiait de la fixer a la main.

`threshold_amr` n'en fait PAS partie : il est gele a la valeur du meilleur
essai de l'etude classique, pour que la comparaison porte sur ce que le
quantique ajoute et non sur un seuil different. `SEARCH_SPACE` et
`FIXED_PARAMS` le declarent, `search_space()` le rend interrogeable AVANT
de lancer une campagne d'une semaine.

Training phases:
----------------
    Phase 1  : perte composite sur les 4 scenarios isoles
               (KH, Lamb-Oseen, Harris, coalescence d'ilots).
    Phase 2  : perte composite sur les 2 scenarios complexes (OT, rotor),
               amorcee par les meilleurs essais de la phase 1.
    Phase 3  : perte composite sur les 6, amorcee par la phase 2.

    Classical Phase 1 / 2 / 3 : memes jeux de scenarios, AMR classique,
               `threshold_amr` seul entraine (pas de circuit — ~100x plus
               rapide). Miroir exact des phases QAOA, pour que la
               comparaison porte sur la meme perte et le meme budget.

Ce que ce module N'a PAS
------------------------
Il n'y a pas de « phase 1b », pas de `beta_michelson`, pas de
`split_michelson`. `beta_michelson` etait propose a Optuna par la phase 1
alors que `pipeline.py` ne le lit nulle part : la phase optimisait un
parametre sans effet. Voir docs/RESULTS.md, D-31.
"""

import argparse
import optuna
import os
import json
import csv
import subprocess
import sys
import itertools
import shutil
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

# ── DISTRIBUTED TRAINING ─────────────────────────────────────
OPTUNA_STORAGE = os.environ.get("OPTUNA_STORAGE", None)
OPTUNA_JOURNAL = os.environ.get("OPTUNA_JOURNAL", None)
WORKER_PHASE   = os.environ.get("WORKER_PHASE", None)
WORKER_TRIALS  = os.environ.get("WORKER_TRIALS", None)
if WORKER_TRIALS is not None:
    WORKER_TRIALS = int(WORKER_TRIALS)

DISTRIBUTED = OPTUNA_STORAGE is not None
JOURNAL_DIR = OPTUNA_JOURNAL


def announce_environment():
    """Ce que le worker a compris de son environnement. Appelee par `main`."""
    if DISTRIBUTED:
        print(f"[DISTRIBUTED MODE] Storage: {OPTUNA_STORAGE.split('@')[-1]}")
    elif JOURNAL_DIR is not None:
        os.makedirs(JOURNAL_DIR, exist_ok=True)
        print(f"[DISTRIBUTED MODE] Journal storage: {JOURNAL_DIR}")
    else:
        print(f"[LOCAL MODE] SQLite dans {data_dir}")
    if WORKER_PHASE:
        print(f"[DISTRIBUTED MODE] Worker phase: {WORKER_PHASE}")
    if WORKER_TRIALS:
        print(f"[DISTRIBUTED MODE] Max trials per worker: {WORKER_TRIALS}")

# --- DETECTION AUTOMATIQUE DE L'ENVIRONNEMENT ---
IN_COLAB = 'google.colab' in sys.modules

#: Ou vivent les bases Optuna et le JSON final. Les repertoires ne sont PAS
#: crees a l'import : importer ce module ne doit rien ecrire sur le disque
#: ni rien afficher. Un import qui a des effets de bord ne peut pas etre
#: teste, et sur les coeurs loues il s'execute une fois par worker.
drive_dir = os.path.join(project_root, "Train_results") if IN_COLAB else None
local_dir = "/content/Train_results_local" if IN_COLAB else None
data_dir = local_dir if IN_COLAB else os.path.join(project_root, "Train_results")

_DIRS_READY = False


def ensure_dirs():
    """Cree `data_dir` (et rapatrie les .db du Drive sous Colab). Idempotent.

    Appelee par `_get_storage` et `_save_results` — c'est-a-dire au premier
    ecrit reel, jamais a l'import.
    """
    global _DIRS_READY
    if _DIRS_READY:
        return data_dir
    if IN_COLAB:
        os.makedirs(drive_dir, exist_ok=True)
        os.makedirs(local_dir, exist_ok=True)
        for file in os.listdir(drive_dir):
            if file.endswith(".db"):
                shutil.copy2(os.path.join(drive_dir, file),
                             os.path.join(local_dir, file))
    else:
        os.makedirs(data_dir, exist_ok=True)
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
    """Hash du commit, proprete de l'arbre, et arguments effectifs.

    CLAUDE.md exige le hash du commit et les arguments CLI dans chaque
    sortie. `dirty` vrai signifie qu'AUCUN hash ne decrit exactement ce qui
    a tourne — il faut le dire plutot que de laisser deviner.
    """
    return {
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": list(sys.argv),
        "env": {k: os.environ.get(k) for k in
                ("OPTUNA_STORAGE", "OPTUNA_JOURNAL", "WORKER_PHASE",
                 "WORKER_TRIALS", "OPTUNA_SEED")},
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
    # Etait ABSENT — `create_argus` repliait alors sur False, et Orszag-Tang
    # etait le seul scenario a tourner sans anomalies avancees. Le terme
    # ZZZZ de point X n'existe pas sans elles : la phase 2 entrainait donc
    # `beta_xpoint` sur un jeu ou l'un des deux scenarios ne pouvait pas
    # l'exprimer. `create_argus` LEVE desormais si la cle manque.
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
# Une entree de phase ne porte que ce qui est REELLEMENT lu : le nom de
# l'etude Optuna et le nombre d'essais vise. Les entrees portaient aussi
# `train_hamiltonian`, `classical_only`, `split_michelson`, `N`, `T_MAX`,
# `DT`, `HYBRID_DT`, `K_opt`, `Re`, `Rm`, `shots`, `AdvAnomaliesEnable` —
# aucune n'etait jamais relue. `classical_only: True` en particulier ne
# rendait pas la phase classique : c'est l'objectif construit par
# `_run_classical_phase*` qui le fait. Une constante deguisee en reglage
# est le motif d'erreur le plus couteux de ce depot ; ces cles sont
# supprimees plutot que documentees.
#
# La physique de chaque scenario vit dans SCENARIO_* et nulle part
# ailleurs : c'est ce dictionnaire-la que `create_argus` et l'objectif
# lisent.

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
        eta=0.001,
        Bz_guide=0.1,
        c_s=1.0,
        Re=scenario_config["Re"],
        Rm=scenario_config["Rm"],
    )


def _get_storage(phase_config):
    """Return the Optuna storage backend for this phase."""
    if DISTRIBUTED:
        # Un pooler Postgres distant ferme les connexions SSL inactives
        # au bout de quelques minutes.
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
    db_path = os.path.join(ensure_dirs(), f"{phase_config['study_name']}.db")
    return f"sqlite:///{db_path}"


def trials_done(study):
    """Essais qui consomment le budget : termines, elagues, ou echoues.

    Un essai WAITING est une graine en file d'attente, pas un essai fait.
    Un essai RUNNING est en cours chez un autre worker : il compte, sinon
    N workers demarrant ensemble liraient tous « 0 fait » et lanceraient
    chacun la campagne entiere.
    """
    return len([t for t in study.trials
                if t.state != optuna.trial.TrialState.WAITING])


def make_pruner():
    """MedianPruner. `n_warmup_steps=2` : elague au plus tot apres le 3e
    scenario rapporte, jamais avant — un scenario seul ne dit pas assez."""
    return optuna.pruners.MedianPruner(
        n_startup_trials=15,
        n_warmup_steps=2,
        interval_steps=1,
        n_min_trials=5,
    )


def run_phase(phase_name, phase_config, objective_fn, seed_params=None,
              seed=None):
    """Run one training phase with Optuna MedianPruner.

    Budget partage entre workers
    ----------------------------
    L'ancienne version calculait `remaining = n_trials - deja_faits` UNE
    fois, puis demandait ce nombre a `study.optimize`. Huit workers loues
    demarrant ensemble lisaient tous « 0 fait » et faisaient 600 essais
    chacun : 4 800 au lieu de 600, huit fois le cout annonce.

    Ici la boucle relit le compte a chaque essai et s'arrete des que la
    cible est atteinte, quel que soit le nombre de workers. Le cout est
    une lecture de la base par essai, contre 10 a 20 minutes de calcul.

    `WORKER_TRIALS` reste un plafond PAR worker (utile pour un temps de
    location borne) ; il ne remplace pas la cible globale.
    """
    storage = _get_storage(phase_config)

    sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else None
    study = optuna.create_study(
        study_name=phase_config["study_name"],
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=make_pruner(),
        sampler=sampler,
    )

    if seed_params is not None and trials_done(study) == 0:
        for params in seed_params:
            study.enqueue_trial(params, skip_if_exists=True)

    db_path = (os.path.join(data_dir, f"{phase_config['study_name']}.db")
               if not DISTRIBUTED else None)

    def callback_save(study, trial):
        if IN_COLAB and not DISTRIBUTED and trial.number % 10 == 0:
            shutil.copy2(db_path,
                         os.path.join(drive_dir, f"{phase_config['study_name']}.db"))

    target_trials = phase_config["n_trials"]
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
        study.optimize(objective_fn, n_trials=1, callbacks=[callback_save])
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


# `SPLIT_BETA_STRATEGIES` et `expand_split_beta_seeds` vivaient ici. Code
# mort : aucun appelant. La fonction mutait de surcroit son argument
# (`params.pop`), et son repli `params.pop(k, params.get(k, 0.5))`
# n'atteignait jamais le `get` puisque le `pop` avait deja retire la cle.
# Supprimes plutot que reparees : il n'y a plus de phase 1b a amorcer.


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


def suggest_hyperparams(trial, frozen=None):
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
        if name not in frozen:
            hp[name] = value
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
    """Moyenne des sous-pertes, avec elagage progressif.

    La moyenne COURANTE est rapportee apres chaque scenario, a `step` egal
    a l'indice du scenario. C'est ce qui rend le MedianPruner actif :
    l'ancienne version ne rapportait qu'une fois, au step 0, et
    `n_warmup_steps=2` fait que `should_prune()` y renvoie toujours False.
    Le pruner etait donc decoratif — verifie, 1e9 au step 0 apres 40
    essais ne declenche rien.

    Pour que les steps soient comparables entre essais, l'ordre des
    scenarios doit etre le meme d'un essai a l'autre : c'est l'ordre de
    `scenario_list`, un tuple fixe.
    """
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
                             lambda_cost=LAMBDA_COST_SOFT):
    """
    Composite loss across a set of scenarios (QAOA method).

    Loss = mean(Loss_i) over all scenarios in scenario_list.

    Les memes hyperparametres traversent TOUS les scenarios : l'optimiseur
    doit donc trouver un reglage qui vaut pour tous les types d'anomalie,
    pas un par scenario.

    Parameters
    ----------
    dns_traces : dict scenario_key -> (dns_trace, hot_start_state).
    scenario_list : liste de (key, config) — quels scenarios composent la
        perte. L'ORDRE compte : il definit les steps de l'elagage.
    frozen_params : hyperparametres imposes par l'appelant, retires de
        l'espace de recherche.
    """
    frozen = frozen_params or {}
    _assert_scenarios_wellformed(scenario_list, dns_traces)

    def objective(trial):
        hyperparams = suggest_hyperparams(trial, frozen)
        # Le dictionnaire complet est attache a l'essai : c'est la seule
        # trace qui permette de redeployer un essai sans le reconstruire
        # a la main. `trial.params` ne porte que l'echantillonne.
        trial.set_user_attr("hyperparams_resolved", hyperparams)
        return _composite_loop(trial, scenario_list, dns_traces, hyperparams,
                               lambda_cost)

    return objective


# ============================================================
#  CLASSICAL COMPOSITE OBJECTIVE
# ============================================================

def make_classical_composite_objective(dns_traces, scenario_list,
                                       lambda_cost=LAMBDA_COST_SOFT):
    """
    Composite loss across scenarios for the CLASSICAL AMR method.

    N'entraine que `threshold_amr` (1 parametre) : pas de circuit, donc
    ~100x plus rapide par essai. C'est le bras de comparaison — meme
    perte, memes scenarios, meme agregation que le bras QAOA, seul le
    critere de raffinement change.

    Parameters
    ----------
    dns_traces : dict scenario_key -> (dns_trace, hot_start_state).
    scenario_list : liste de (key, config), meme contrat que ci-dessus.
    """
    _assert_scenarios_wellformed(scenario_list, dns_traces)

    def objective(trial):
        lo, hi = CLASSICAL_THRESHOLD_RANGE
        hyperparams = {"threshold_amr": trial.suggest_float("threshold_amr", lo, hi)}
        trial.set_user_attr("hyperparams_resolved", hyperparams)
        return _composite_loop(trial, scenario_list, dns_traces, hyperparams,
                               lambda_cost, classical_only=True)

    return objective


#: Bornes du seuil classique. `CLASSICAL_BEST_THRESHOLD` doit tomber
#: dedans, sinon le bras quantique est gele sur une valeur que le bras
#: classique n'avait pas le droit de proposer — un test le verifie.
CLASSICAL_THRESHOLD_RANGE = (0.05, 0.8)


# ============================================================
#  ORCHESTRATION
# ============================================================
#
# `make_phase3_objective` vivait ici : 64 lignes, aucun appelant. Elle
# reimplementait l'objectif sur Orszag-Tang SEUL, alors que la phase 3
# est definie comme la validation sur les 6 scenarios — et elle portait sa
# propre copie non nommee des quatre constantes (0.14959..., 2.0, 0.5,
# 10.0). Deux definitions du meme objectif, dont une morte : supprimee.


# Les quatre scenarios ISOLES : un type d'anomalie chacun. C'est tout
# l'argument du protocole — Orszag-Tang les melange, donc l'optimiseur ne
# peut pas y decoupler leurs contributions.
#
# La liste contenait `ot` et `rotor` a la place de `vortex` et
# `coalescence`. Trois consequences, toutes silencieuses :
#   - les deux scenarios complexes etaient dans le jeu « isole », qui
#     n'isolait donc plus rien ;
#   - `SCENARIO_VORTEX` et `SCENARIO_COALESCENCE` etaient definis et
#     jamais utilises ;
#   - `SCENARIOS_ALL = ISOLATED + COMPLEX` valait six entrees pour quatre
#     scenarios distincts : `ot` et `rotor` etaient simules DEUX fois par
#     essai, comptes deux fois dans la somme, et divises par six. La
#     phase 3 les ponderait donc double, pour le double du cout.
#
# Le JSON deploye tranche : son bloc `per_scenario` de la phase 1 liste
# kelvin_helmholtz, lamb_oseen_vortex, harris_tearing, island_coalescence.
# C'est la liste ci-dessous qui a produit la campagne gelee ; la version
# precedente etait une regression.
SCENARIOS_ISOLATED = (
    ("kh",          SCENARIO_KH),
    ("vortex",      SCENARIO_VORTEX),
    ("tearing",     SCENARIO_TEARING),
    ("coalescence", SCENARIO_COALESCENCE),
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


def _run_phase1(dns_traces, seed=None):
    """Phase 1 : perte composite sur les 4 scenarios isoles.

    Le nombre de parametres n'est pas ecrit ici : il vaut
    `len(SEARCH_SPACE)`, et la ligne « Training: » ci-dessous
    l'imprime. Une valeur figee dans une docstring se serait
    desynchronisee au premier ajout — elle l'avait deja fait.
    """
    print("=" * 60)
    print("PHASE 1: Composite Training (4 scenarios isoles)")
    print(f"  Training: {', '.join(search_space())}")
    print(f"  Fixed:    {FIXED_PARAMS}")
    print(f"  Scenarios: {', '.join(k for k, _ in SCENARIOS_ISOLATED)}")
    print("=" * 60)

    objective = make_composite_objective(dns_traces, SCENARIOS_ISOLATED)
    study = run_phase("phase1_composite", PHASES["phase1_composite"],
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

    dns_traces_complex = _precompute_dns_for(SCENARIOS_COMPLEX, label="OT + Rotor")
    seed_params = _seeds_for("q_has_v2_phase1", study_p1, 20, "PHASE 2")

    objective = make_composite_objective(dns_traces_complex, SCENARIOS_COMPLEX)
    study = run_phase("phase2_complex", PHASES["phase2_complex"],
                      objective, seed_params=seed_params, seed=seed)
    _report_best(study, "Phase 2", SCENARIOS_COMPLEX)
    return study


def _run_phase3(study_p2, seed=None):
    """Phase 3 : les 6 scenarios distincts, amorcee par la phase 2."""
    print("\n" + "=" * 60)
    print("PHASE 3: Composite Training sur les 6 scenarios")
    print(f"  Scenarios: {', '.join(k for k, _ in SCENARIOS_ALL)}")
    print("=" * 60)

    dns_traces_all = _precompute_dns_for(SCENARIOS_ALL, label="6 scenarios")
    seed_params = _seeds_for("q_has_v2_phase2", study_p2, 15, "PHASE 3")

    objective = make_composite_objective(dns_traces_all, SCENARIOS_ALL)
    study = run_phase("phase3_validation", PHASES["phase3_validation"],
                      objective, seed_params=seed_params, seed=seed)
    _report_best(study, "Phase 3", SCENARIOS_ALL)
    return study


def _classical_grid_seeds(n=20):
    lo, hi = CLASSICAL_THRESHOLD_RANGE
    return [{"threshold_amr": t} for t in np.linspace(lo, hi, n).tolist()]


def _run_classical_phase1(dns_traces, seed=None):
    """Classique 1 : `threshold_amr` sur les 4 scenarios isoles."""
    print("\n" + "=" * 60)
    print("CLASSICAL PHASE 1 (*): threshold_amr, 4 scenarios isoles")
    print("=" * 60)

    objective = make_classical_composite_objective(dns_traces, SCENARIOS_ISOLATED)
    study = run_phase("classical_phase1", PHASES["classical_phase1"],
                      objective, seed_params=_classical_grid_seeds(20), seed=seed)
    _report_best(study, "Classical Phase 1", SCENARIOS_ISOLATED)
    return study


def _run_classical_phase2(study_c1, seed=None):
    """Classique 2 : `threshold_amr` sur OT + rotor."""
    print("\n" + "=" * 60)
    print("CLASSICAL PHASE 2 (**): threshold_amr, OT + rotor")
    print("=" * 60)

    dns_traces_complex = _precompute_dns_for(SCENARIOS_COMPLEX,
                                             label="OT + Rotor (classique)")
    seeds = _seeds_for("classical_v2_phase1", study_c1, 15, "CLASSICAL PHASE 2")
    if not seeds:
        seeds = _classical_grid_seeds(15)

    objective = make_classical_composite_objective(dns_traces_complex,
                                                   SCENARIOS_COMPLEX)
    study = run_phase("classical_phase2", PHASES["classical_phase2"],
                      objective, seed_params=seeds, seed=seed)
    _report_best(study, "Classical Phase 2", SCENARIOS_COMPLEX)
    return study


def _run_classical_phase3(study_c2, seed=None):
    """Classique 3 : `threshold_amr` sur les 6 scenarios."""
    print("\n" + "=" * 60)
    print("CLASSICAL PHASE 3 (***): threshold_amr, 6 scenarios")
    print("=" * 60)

    dns_traces_all = _precompute_dns_for(SCENARIOS_ALL,
                                         label="6 scenarios (classique)")
    seeds = _seeds_for("classical_v2_phase2", study_c2, 15, "CLASSICAL PHASE 3")
    if not seeds:
        seeds = _classical_grid_seeds(15)

    objective = make_classical_composite_objective(dns_traces_all, SCENARIOS_ALL)
    study = run_phase("classical_phase3", PHASES["classical_phase3"],
                      objective, seed_params=seeds, seed=seed)
    _report_best(study, "Classical Phase 3", SCENARIOS_ALL)
    return study


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


def _save_results(study_p1, study_p2, study_p3,
                  study_c1=None, study_c2=None, study_c3=None,
                  filename="best_hyperparams.json"):
    """Ecrit le JSON final : parametres complets + provenance.

    Le fichier doit se suffire a lui-meme. Toute valeur qu'il ne porte pas
    sera comblee au deploiement par un repli que personne n'a choisi.
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

    # Ce qu'on deploie : la phase 3, les deux bras.
    results["deploy"] = {
        "quantum": results["quantum"]["phase3"]["best_params"],
        "classical": (results["classical"]["phase3"]["best_params"]
                      if study_c3 is not None else None),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResultats finaux ecrits dans {output_path}")
    print(f"  quantique  : {results['deploy']['quantum']}")
    print(f"  classique  : {results['deploy']['classical']}")

    if IN_COLAB:
        try:
            shutil.copytree(local_dir, drive_dir, dirs_exist_ok=True)
            print(f"Copie vers Drive : {drive_dir}")
        except Exception as e:
            print(f"Erreur lors de la copie vers Drive : {e}")
    return output_path


# ============================================================
#  MAIN
# ============================================================

PHASE_CHOICES = ("1", "2", "3", "classical_1", "classical_2", "classical_3",
                 "classical", "all")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Entrainement des hyperparametres Q-HAS (8 parametres).")
    p.add_argument("--phase", choices=PHASE_CHOICES,
                   default=os.environ.get("WORKER_PHASE") or "all",
                   help="phase a executer (defaut : WORKER_PHASE, sinon tout)")
    p.add_argument("--seed", type=int,
                   default=(int(os.environ["OPTUNA_SEED"])
                            if os.environ.get("OPTUNA_SEED") else None),
                   help="graine du sampler TPE. Sans elle, Optuna tire au "
                        "hasard et la campagne n'est pas reproductible.")
    p.add_argument("--print-space", action="store_true",
                   help="affiche l'espace de recherche et sort, sans rien "
                        "calculer. A lancer AVANT de louer des coeurs.")
    return p.parse_args(argv)


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
    if args.seed is None:
        print("[WARN] pas de --seed : le sampler TPE est aleatoire, cette "
              "campagne ne sera pas reproductible telle quelle.")

    if args.phase == "1":
        _run_phase1(_precompute_dns_for(SCENARIOS_ISOLATED, "4 isoles"), args.seed)

    elif args.phase == "2":
        _run_phase2(_load_study("phase1_composite"), args.seed)

    elif args.phase == "3":
        _run_phase3(_load_study("phase2_complex"), args.seed)

    elif args.phase == "classical_1":
        _run_classical_phase1(_precompute_dns_for(SCENARIOS_ISOLATED, "4 isoles"),
                              args.seed)

    elif args.phase == "classical_2":
        _run_classical_phase2(_load_study("classical_phase1"), args.seed)

    elif args.phase == "classical_3":
        _run_classical_phase3(_load_study("classical_phase2"), args.seed)

    elif args.phase == "classical":
        dns = _precompute_dns_for(SCENARIOS_ISOLATED, "4 isoles")
        c1 = _run_classical_phase1(dns, args.seed)
        c2 = _run_classical_phase2(c1, args.seed)
        _run_classical_phase3(c2, args.seed)

    else:  # "all"
        dns = _precompute_dns_for(SCENARIOS_ISOLATED, "4 isoles")
        p1 = _run_phase1(dns, args.seed)
        p2 = _run_phase2(p1, args.seed)
        p3 = _run_phase3(p2, args.seed)
        c1 = _run_classical_phase1(dns, args.seed)
        c2 = _run_classical_phase2(c1, args.seed)
        c3 = _run_classical_phase3(c2, args.seed)
        _save_results(p1, p2, p3, c1, c2, c3)
        print("\n" + "=" * 60)
        print("ALL TRAINING COMPLETE")
        print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
