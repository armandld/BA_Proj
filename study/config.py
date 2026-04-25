"""
Shared configuration for the physics-first Hamiltonian study.

Phases:
  1. DNS sweep at multiple Re on OT + Tearing
  2. Hard-patch identification by L2 error
  3. Hamiltonian coefficient analysis + threshold stability
  4. Exact diagonalization on hard patches
  5. QAOA evaluation on promising patches
"""
import os, sys

# -- paths --
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -- DNS parameters --
SCENARIOS = ["orszag_tang", "harris_tearing", "kelvin_helmholtz", "mhd_rotor"]
RE_VALUES = [400, 800, 1200, 1600]
RM_VALUES = RE_VALUES                     # Rm = Re (unit magnetic Prandtl)
DNS_N = 256                               # primary resolution
DNS_N_HIGH = 512                          # optional high-res validation
DT_INIT = 1e-3                            # initial dt (CFL-adapted)

# scenario-specific run lengths (long enough for instabilities to develop)
SCENARIO_CONFIG = {
    "orszag_tang": {
        "warmup_steps": 120,
        "t_max": 3.0,
        "snapshot_dt": 0.10,
    },
    "harris_tearing": {
        "warmup_steps": 80,
        "t_max": 2.0,
        "snapshot_dt": 0.10,
    },
    "kelvin_helmholtz": {
        "warmup_steps": 100,
        "t_max": 3.0,
        "snapshot_dt": 0.10,
    },
    "mhd_rotor": {
        "warmup_steps": 60,
        "t_max": 1.5,
        "snapshot_dt": 0.05,
    },
}

# -- Patch analysis --
VQA_DIMS = [2, 4, 8]                     # coarse grid sizes to test
L2_PERCENTILE_HARD = 75                   # top 25% L2 error = "hard"

# -- Hamiltonian v1 defaults (from previous training) --
TRAINED_THRESHOLD = 0.1496
TRAINED_SIGMA = 0.023
TRAINED_BETA_CURL = 4.27
TRAINED_BETA_XPOINT = 2.39
TRAINED_W_Z_FRAC = 10.40
TRAINED_BETA = 9.94
TRAINED_GAMMA_HYDRO = 2.0
TRAINED_GAMMA_MAG = 0.5
TRAINED_KAPPA = 10.0

# -- Hamiltonian v2 defaults (parameter-free, physics-first) --
# Only thr_amr is a free parameter. Fixed weights:
V2_THRESHOLD = 0.15                       # physical choice, not trained
V2_W_ZZ = 2.0                            # ZZ coupling weight (fixed)
V2_W_ZZZZ = 1.0                          # ZZZZ coupling weight (fixed)
V2_C_BIAS = 0.1                          # Z bias: fraction of median(|C|,|K|)
