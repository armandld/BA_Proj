"""Shared configuration for the physics-first Hamiltonian study."""
import hashlib
import os
import sys

# -- paths --
# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------

RESULTS_DIR = os.path.abspath(os.environ.get(
    "QHAS_RESULTS_DIR", os.path.join(_REPO_ROOT, "results")))
os.makedirs(RESULTS_DIR, exist_ok=True)

# -- DNS parameters --
SCENARIOS = [
    "orszag_tang", "harris_tearing", "kelvin_helmholtz", "mhd_rotor",
    "lamb_oseen", "island_coalescence", "double_tearing", "magnetic_twist",
]
# Keys used by the closed-loop LOSO campaign. Keep this order aligned with
# ``src/train_hyperparams.py::SCENARIOS_ALL``.
FOLD_KEYS = (
    "kh", "vortex", "tearing", "coalescence", "double_tearing",
    "magnetic_twist", "ot", "rotor",
)
RE_VALUES = [400, 800, 1200, 1600]
RM_VALUES = RE_VALUES                     # Rm = Re (unit magnetic Prandtl)
PHYSICS_SEEDS = [0, 1, 2, 3, 4]
DNS_N = 256                               # primary resolution
DNS_N_HIGH = 512                          # optional high-res validation
DT_INIT = 1e-3                            # initial dt (CFL-adapted)
PHYSICS_NOISE_AMPLITUDE = {
    scenario: (0.005 if scenario == "kelvin_helmholtz" else 0.1)
    for scenario in SCENARIOS
}

# scenario-specific run lengths (long enough for instabilities to develop)
SCENARIO_CONFIG = {
    "orszag_tang": {
        "t_max": 3.0,
        "snapshot_dt": 0.10,
    },
    "harris_tearing": {
        "t_max": 2.0,
        "snapshot_dt": 0.10,
    },
    "kelvin_helmholtz": {
        "t_max": 3.0,
        "snapshot_dt": 0.10,
    },
    "mhd_rotor": {
        "t_max": 1.5,
        "snapshot_dt": 0.05,
    },
    "lamb_oseen": {
        "t_max": 3.0,
        "snapshot_dt": 0.10,
    },
    "island_coalescence": {
        "t_max": 2.0,
        "snapshot_dt": 0.10,
    },
    "double_tearing": {
        "t_max": 2.0,
        "snapshot_dt": 0.10,
    },
    "magnetic_twist": {
        "t_max": 2.0,
        "snapshot_dt": 0.10,
    },
}

# -- Patch analysis --
VQA_DIMS = [2, 4, 8]                     # coarse grid sizes to test
L2_PERCENTILE_HARD = 75                   # top 25% L2 error = "hard"

# -- V1 parameter source -------------------------------------------------
# The built-in values reproduce the reference configuration. Setting
# QHAS_HYPERPARAMS_PATH makes every study module import the completed campaign
# candidate instead, without editing source code after the campaign.
_REFERENCE_TRAINED = {
    "threshold_amr": 0.1496,
    "sigma": 0.023,
    "beta_curl": 4.27,
    "beta_xpoint": 2.39,
    "w_z_frac": 10.40,
    "beta": 9.94,
    "gamma_hydro": 2.0,
    "gamma_mag": 0.5,
    "kappa": 10.0,
    "relative_percentile": 90.0,
}
CAMPAIGN_HYPERPARAMS_PATH = os.environ.get("QHAS_HYPERPARAMS_PATH")
if CAMPAIGN_HYPERPARAMS_PATH:
    from hyperparams_loader import load_hyperparams, resolve_hyperparams_path
    _TRAINED = load_hyperparams(path=CAMPAIGN_HYPERPARAMS_PATH)
    CAMPAIGN_HYPERPARAMS_PATH = resolve_hyperparams_path(
        CAMPAIGN_HYPERPARAMS_PATH)
else:
    _TRAINED = dict(_REFERENCE_TRAINED)

TRAINED_THRESHOLD = float(_TRAINED["threshold_amr"])
TRAINED_SIGMA = float(_TRAINED["sigma"])
TRAINED_BETA_CURL = float(_TRAINED["beta_curl"])
TRAINED_BETA_XPOINT = float(_TRAINED["beta_xpoint"])
TRAINED_W_Z_FRAC = float(_TRAINED["w_z_frac"])
TRAINED_BETA = float(_TRAINED["beta"])
TRAINED_GAMMA_HYDRO = float(_TRAINED["gamma_hydro"])
TRAINED_GAMMA_MAG = float(_TRAINED["gamma_mag"])
TRAINED_KAPPA = float(_TRAINED["kappa"])
TRAINED_RELATIVE_PERCENTILE = float(_TRAINED["relative_percentile"])


def trained_mapper_params():
    """Paramètres déployés du mappeur V1, sous les noms de son constructeur."""
    return {
        "sigma": TRAINED_SIGMA,
        "beta_curl": TRAINED_BETA_CURL,
        "beta_xpoint": TRAINED_BETA_XPOINT,
        "w_z_frac": TRAINED_W_Z_FRAC,
        "gamma_hydro": TRAINED_GAMMA_HYDRO,
        "gamma_mag": TRAINED_GAMMA_MAG,
        "kappa": TRAINED_KAPPA,
        "relative_percentile": TRAINED_RELATIVE_PERCENTILE,
    }


def trained_configuration():
    """Return the resolved V1 parameters and their immutable source label."""
    source = "built_in_reference"
    digest = None
    if CAMPAIGN_HYPERPARAMS_PATH:
        source = CAMPAIGN_HYPERPARAMS_PATH
        with open(CAMPAIGN_HYPERPARAMS_PATH, "rb") as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
    return {
        "source": source,
        "sha256": digest,
        "params": dict(_TRAINED),
    }

# -- A-priori Hamiltonian V2 constants used by phases 3--8 --
V2_THRESHOLD = 0.15
V2_W_ZZ = 2.0                            # ZZ coupling weight (fixed)
V2_W_ZZZZ = 1.0                          # ZZZZ coupling weight (fixed)
