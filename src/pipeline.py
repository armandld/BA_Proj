import warnings
import argparse
import sys
from math import log


import numpy as np
from types import SimpleNamespace

from Simulation.grid import AXIS_X, AXIS_Y, PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.refinement import run_adaptive_vqa, run_adaptive_classical
from Simulation.pre_compute_dns import precompute_dns
from VQA.runtime import VQARuntime
from visual import plot_amr_state
from hyperparams_loader import load_hyperparams


FREQUENCY = 1  # Fréquence d'affichage (en nombre de pas de temps)
DIVERGENCE_PENALTY = 10.0  # Finite penalty for diverged trials (replaces inf)
# Cette constante etait redefinie trois fois de plus, dans des portees
# locales qui masquaient celle-ci : changer la valeur ici n'aurait eu
# d'effet que dans un cas sur quatre. Definition unique desormais.

# ══════════════════════════════════════════════════════════════════
#  Configuration par scenario — la table qui fait foi
# ══════════════════════════════════════════════════════════════════
#
# Sortie du corps de `main()` pour etre TESTABLE : c'est de la donnee,
# pas de la logique, et c'est elle qui decide ce qu'une campagne mesure.
# Un test verifie que chaque entree porte T_MAX > T_START — l'invariant
# dont la violation faisait D-66.

N_TRAINING         = 256
MAX_DEPTH_TRAINING = 4

PHASE={
    "orszag_tang": {
        "scenario": "orszag_tang",
        "N": N_TRAINING,
        "max_depth_override": MAX_DEPTH_TRAINING,
        "T_MAX": 2.8,
        "T_START": 2.3,
        "DT": 1e-3,
        "HYBRID_DT": 0.10,
        "K_opt": 30,
        "Re": 800,
        "Rm": 800,
        "shots": 256,
        # Required for the X-point term.
        "AdvAnomaliesEnable": True,
    },

    "kelvin_helmholtz": {
        "scenario": "kelvin_helmholtz",
        "N": N_TRAINING,
        "max_depth_override": MAX_DEPTH_TRAINING,
        "T_MAX": 1.7,
        "T_START": 1.3,      # KH instability develops around t~1.0-1.5
        "DT": 1e-3,
        "HYBRID_DT": 0.10,
        "K_opt": 30,
        "Re": 800,
        "Rm": 800,
        "shots": 256,
        "AdvAnomaliesEnable": True,
    },

    "lamb_oseen_vortex": {
        "scenario": "lamb_oseen_vortex",
        "N": N_TRAINING,
        "max_depth_override": MAX_DEPTH_TRAINING,
        "T_MAX": 1.0,
        "T_START": 0.6,       # Vortex is present from t=0, start early
        "DT": 1e-3,
        "HYBRID_DT": 0.10,
        "K_opt": 30,
        "Re": 800,
        "Rm": 800,
        "shots": 256,
        "AdvAnomaliesEnable": True,
    },

    "harris_tearing" : {
        "scenario": "harris_tearing",
        "N": N_TRAINING,
        "max_depth_override": MAX_DEPTH_TRAINING,
        "T_MAX": 1.1,
        "T_START": 0.7,       # Tearing mode develops around t~0.5-1.0
        "DT": 1e-3,
        "HYBRID_DT": 0.10,
        "K_opt": 30,
        "Re": 800,
        "Rm": 800,
        "shots": 256,
        "AdvAnomaliesEnable": True,
    },

    "double_tearing": {
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
    },

    "magnetic_twist": {
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
    },

    "island_coalescence" : {
        "scenario": "island_coalescence",
        "N": N_TRAINING,
        "max_depth_override": MAX_DEPTH_TRAINING,
        "T_MAX": 0.8,
        "T_START": 0.4,       # Shock develops around t~0.2-0.5
        "DT": 1e-3,
        "HYBRID_DT": 0.10,    # Reconnection is fast — frequent VQA calls
        "K_opt": 30,
        "Re": 800,
        "Rm": 800,
        "shots": 256,
        "AdvAnomaliesEnable": True,  # X-point detection requires advanced anomalies
    },

    "mhd_rotor" : {
        "scenario": "mhd_rotor",
        "N": N_TRAINING,
        "max_depth_override": MAX_DEPTH_TRAINING,
        "T_MAX": 0.7,
        "T_START": 0.3,       # Rotor winds up B-field around t~0.2-0.5
        "DT": 1e-3,
        "HYBRID_DT": 0.10,
        "K_opt": 30,
        "Re": 800,
        "Rm": 800,
        "shots": 256,
        "AdvAnomaliesEnable": True,
    },

}


def main():
    sys.stdout.reconfigure(line_buffering=True) # Pour un affichage immédiat des print() à enlever pour une meilleure perf

    parser = argparse.ArgumentParser(description="Mapping VQA")
    parser.add_argument("--out-dir", default="../data", help="Output directory for mapping")
    parser.add_argument("--in-file", default="../input/mapping_input.json", help="Input directory for mapping")
    parser.add_argument("--verbose", action="store_true")
    # D-66 : ces sept options valent `None` par defaut et sont resolues
    # depuis `PHASE[scenario]`. Elles valaient auparavant des constantes de
    # CLI qui ECRASAIENT la configuration du scenario -- voir `_resolve`.
    parser.add_argument("--AdvAnomaliesEnable", action="store_true", default=None)
    parser.add_argument("--grid-size", type=int, default=2, help="Coarse grid dimension N (NxN)")
    parser.add_argument("--dns-resolution", type=int, default=None, help="High-Res Grid for Ground Truth (defaut : celle du scenario)")
    parser.add_argument("--t-max", type=float, default=None, help="Simulation end time (defaut : celle du scenario)")
    parser.add_argument("--dt", type=float, default=None, help="Time step size (defaut : celui du scenario)")
    parser.add_argument("--hybrid-dt", type=float, default=None, help="Hybrid simulation time step size (defaut : celui du scenario)")
    parser.add_argument("--reps", type=int, default=-1, required=False, help="Number of repetitions for the QAOA ansatz.")
    # `hardware` retire des choix : aucun backend IBM reel n'est cable, et
    # un run demande en materiel s'executait sur simulateur sans le signaler
    # (D-48). L'annoncer dans l'aide de la CLI en faisait une option
    # credible.
    parser.add_argument("--mode", default="simulator", choices=["simulator"])
    parser.add_argument(
        "--backend", default="state_vector",
        choices=["aer", "matrix_product_state", "state_vector"],
    )
    parser.add_argument("--shots", type=int, default=None, help="(defaut : celui du scenario)")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed shared by transpilation, Estimator and Sampler",
    )
    parser.add_argument("--method", default="COBYLA", choices=["COBYLA", "L-BFGS-B", "Powell"])
    parser.add_argument("--opt-level", type=int, default=1, choices=[0,1,2,3], help="Optimization level for transpilation.")
    parser.add_argument("--K-opt", type=int, default=None, help="Maximum number of iterations for the optimizer (defaut : celui du scenario)")
    parser.add_argument(
        "--hyperparams-file",
        help="Completed campaign candidate or deploy export. Defaults to "
             "QHAS_HYPERPARAMS_PATH, then the reference artifact.")
    parser.add_argument("--eps", type=float, default=1e-2, help="Convergence tolerance for the optimizer.")
    # Les choix sont DERIVES de `PHASE`, pas recopies a cote.
    #
    # La liste ecrite a la main en annoncait dix quand `PHASE` en porte
    # sept : `magnetic_twist`, `noisy_uniform` et `double_tearing` etaient
    # acceptes par la CLI puis levaient `KeyError` sur `PHASE[scenario]`.
    # Meme famille que D-48 : une option affichee dans l'aide est une
    # promesse.
    parser.add_argument("--scenario", default="orszag_tang",
                        choices=sorted(PHASE),
                        help="Initial condition scenario")

    args = parser.parse_args()

    verbose = args.verbose
    VQA_N = args.grid_size                    # Résolution Grossière

    cfg = PHASE[args.scenario]

    # ── D-66 : la configuration du scenario fait foi ──────────────────
    #
    # `main` precalculait le DNS avec `PHASE[scenario]` puis passait a
    # `pipeline()` les DEFAUTS DE LA CLI. Sept des neuf cles etaient
    # ignorees, et le DNS tournait sous une physique quand la boucle
    # hybride tournait sous une autre :
    #
    #     T_MAX  2.8 (PHASE) contre 1.0 (CLI)
    #     DT     1e-3        contre 1e-4
    #     Re/Rm  800         contre 1000
    #     shots  256         contre 1024
    #     K_opt  30          contre 80
    #     AdvAnomaliesEnable True contre False
    #
    # Le hot start place `t_current` a T_START = 2.3 ; avec T_MAX = 1.0 la
    # condition `while t_current < T_MAX` est fausse d'entree. La boucle ne
    # s'executait JAMAIS. L'etat final restait l'etat DNS, d'ou une erreur
    # exactement nulle sur les cinq champs et un `combined = 0.333333`
    # parfaitement plausible -- pour un run qui n'avait rien calcule.
    #
    # Mesure apres correction, orszag_tang, N=256, profondeur 4 :
    #   Q-HAS      combined 0.228928  phys 0.140052  patch 0.4067
    #   Classique  combined 0.212591  phys 0.117626  patch 0.4025
    #
    # `_resolve` : la valeur du scenario, sauf si la CLI l'a passee
    # EXPLICITEMENT (defaut `None`).
    def _resolve(cli_value, cle, defaut=None):
        if cli_value is not None:
            return cli_value
        return cfg.get(cle, defaut)

    N     = _resolve(args.dns_resolution, "N")
    T_MAX = _resolve(args.t_max,          "T_MAX")
    DT    = _resolve(args.dt,             "DT")
    hybrid_dt = _resolve(args.hybrid_dt,  "HYBRID_DT")
    HYBRID = int(hybrid_dt / DT)          # Fréquence de mise à jour hybride

    argus = SimpleNamespace(
        reps=args.reps if args.reps > 0 else (VQA_N-1) * 2, # 2 for 2D, 3 for 3D
        mode=args.mode,
        backend=args.backend,
        shots=_resolve(args.shots, "shots"),
        method=args.method,
        opt_level=args.opt_level,
        AdvAnomaliesEnable=_resolve(args.AdvAnomaliesEnable, "AdvAnomaliesEnable", False),
        K_opt=_resolve(args.K_opt, "K_opt"),
        eps=args.eps,
        seed=args.seed,
        eta=0.001,       # Faible résistivité pour laisser l'instabilité grandir
        Bz_guide=0.1,    # Faible champ guide pour la stabilité
        c_s=1.0,         # Référence de vitesse acoustique
        Re=_resolve(None, "Re"),
        Rm=_resolve(None, "Rm"),
    )

    # Le hot start demarre a T_START : un T_MAX anterieur rend la boucle
    # vide, ce qui produisait un score plausible sans aucun calcul.
    t_start = cfg.get("T_START", 0.0)
    if T_MAX <= t_start:
        raise ValueError(
            f"T_MAX={T_MAX} <= T_START={t_start} pour le scenario "
            f"'{args.scenario}' : le hot start place t_current a T_START, "
            f"donc `while t_current < T_MAX` serait faux des l'entree et la "
            f"boucle ne tournerait pas. Le run rendrait une erreur nulle et "
            f"un score plausible sans rien calculer. Voir D-66.")

    dns_trace, hot_start_state = precompute_dns(cfg)

    if verbose:
        print(f"Configuration resolue pour '{args.scenario}' : "
              f"N={N}, T_START={t_start}, T_MAX={T_MAX}, DT={DT}, "
              f"HYBRID_DT={hybrid_dt}, Re={argus.Re}, Rm={argus.Rm}, "
              f"shots={argus.shots}, K_opt={argus.K_opt}, "
              f"AdvAnomalies={argus.AdvAnomaliesEnable}")

    print(f"Starting pipeline... saved in{args.out_dir}")
    pipeline(N, VQA_N, T_MAX, DT, HYBRID, verbose, argus,
             lambda_cost=0.5, trial=None, classic_AMR_comp=True,
             dns_trace=dns_trace, hot_start_state=hot_start_state,
             max_depth_override=cfg.get("max_depth_override", 4),
             scenario=args.scenario, save_dir=args.out_dir,
             hyperparams_path=args.hyperparams_file)



def _init_scenario(sim, scenario):
    """Dispatch l'initialisation du solveur selon le scénario choisi."""
    init_map = {
        'orszag_tang':       sim.init_orszag_tang,
        'kelvin_helmholtz':  sim.init_kelvin_helmholtz,
        'magnetic_twist':    sim.init_magnetic_twist,
        'noisy_uniform':     sim.init_noisy_uniform,
        'harris_tearing':    sim.init_harris_tearing,
        'double_tearing':    sim.init_double_tearing,
        'lamb_oseen_vortex': sim.init_lamb_oseen_vortex,
        'island_coalescence': sim.init_island_coalescence,
        'mhd_rotor':         sim.init_mhd_rotor,
    }
    init_map[scenario]()


def validate_precomputed_run(dns_trace, hot_start_state, N, T_MAX):
    """Validate that a DNS trace covers exactly the requested continuation."""
    if not isinstance(dns_trace, dict) or not dns_trace:
        raise ValueError("dns_trace must be a non-empty dictionary")
    keys = sorted(dns_trace)
    if keys != list(range(keys[0], keys[-1] + 1)):
        raise ValueError("dns_trace step indices must be contiguous")

    if hot_start_state is None:
        start_step, start_time = keys[0], 0.0
    else:
        required = {"vx", "vy", "Bx", "By", "t_current", "step"}
        missing = required - set(hot_start_state)
        if missing:
            raise KeyError(f"hot_start_state is missing {sorted(missing)}")
        start_step = int(hot_start_state["step"])
        start_time = float(hot_start_state["t_current"])
        for field in ("vx", "vy", "Bx", "By"):
            values = np.asarray(hot_start_state[field])
            if values.shape != (N, N) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"hot-start field {field} must be finite with shape {(N, N)}")

    if start_step not in dns_trace:
        raise ValueError(f"hot-start step {start_step} is absent from dns_trace")
    if not np.isfinite(start_time) or start_time >= T_MAX:
        raise ValueError(
            f"hot-start time {start_time} must be finite and smaller than "
            f"T_MAX={T_MAX}")

    duration = 0.0
    for step in range(start_step, keys[-1] + 1):
        dt = float(dns_trace[step].get("dt", np.nan))
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dns_trace[{step}]['dt'] must be finite and > 0")
        duration += dt
    tolerance = max(1e-10, 1e-9 * max(1.0, abs(T_MAX)))
    if not np.isclose(start_time + duration, T_MAX, rtol=0.0,
                      atol=tolerance):
        raise ValueError(
            f"dns_trace covers t={start_time + duration:.16g}, expected "
            f"T_MAX={T_MAX:.16g}")
    if "fluxes" not in dns_trace[keys[-1]]:
        raise ValueError("the final DNS trace entry has no reference fluxes")


def pipeline(N, VQA_N, T_MAX, DT, HYBRID, verbose, argus,
             hyperparams=None, lambda_cost=0.5, trial=None,
             classic_AMR_comp=False, dns_trace=None, hot_start_state=None,
             min_patch_size=6, max_depth_override=None,
             scenario='orszag_tang', save_dir=None, return_details=False,
             classical_only=False, hyperparams_path=None):

    #Paramètres physiques
    eta = argus.eta       # Faible résistivité pour laisser l'instabilité grandir
    Bz_guide = argus.Bz_guide    # Faible champ guide pour la stabilité
    c_s = argus.c_s         # Référence de vitesse acoustique

    total_pixel_used = 0
    steps_hybrid_count = 0

    # 2. Initialisation
    if verbose:
        print(f"Initialisation {scenario} (Grille {N}x{N}, Re={argus.Re}, Rm={argus.Rm}) — T_MAX={T_MAX}, hybrid every {HYBRID*DT:.3f}s")

    grid = PeriodicGrid(resolution_N=N)
    sim_quantum = MHDSolver(grid, dt=DT, Re=argus.Re, Rm=argus.Rm)

    if hot_start_state is not None:
        sim_quantum.vx = hot_start_state['vx'].copy()
        sim_quantum.vy = hot_start_state['vy'].copy()
        sim_quantum.Bx = hot_start_state['Bx'].copy()
        sim_quantum.By = hot_start_state['By'].copy()
        t_current = hot_start_state['t_current']
        step = hot_start_state['step']
    else:
        _init_scenario(sim_quantum, scenario)
        t_current = 0.0
        step = 0

    dns_presence = dns_trace is not None
    if dns_presence:
        validate_precomputed_run(dns_trace, hot_start_state, N, T_MAX)
    # sim_temoin is the DNS witness — only needed when there is no precomputed
    # dns_trace (live comparison mode).  When dns_trace is provided, the
    # reference comes from the precomputed trace, so we skip allocating and
    # stepping the witness solver entirely.  This halves the cost per trial
    # for training phases that use precomputed DNS (classical + quantum).
    sim_temoin = None
    sim_classical = None
    if classic_AMR_comp:
        sim_classical = MHDSolver(grid, dt=DT, Re=argus.Re, Rm=argus.Rm)

    if not dns_presence:
        sim_temoin = MHDSolver(grid, dt=DT, Re=argus.Re, Rm=argus.Rm)
        _init_scenario(sim_temoin, scenario)
        if classic_AMR_comp:
            _init_scenario(sim_classical, scenario)
    else:
        # No need for sim_temoin — dns_trace is the reference
        if classic_AMR_comp:
            sim_classical.vx = sim_quantum.vx.copy()
            sim_classical.vy = sim_quantum.vy.copy()
            sim_classical.Bx = sim_quantum.Bx.copy()
            sim_classical.By = sim_quantum.By.copy()

    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)

    # Load defaults from best_hyperparams.json, then override with caller's hp
    _defaults = load_hyperparams(path=hyperparams_path)
    hp = {**_defaults, **(hyperparams or {})}

    # ── Encoding hyperparameters (Tier 1) ──
    beta            = hp.get('beta', _defaults['beta'])
    threshold_amr   = hp.get('threshold_amr', _defaults['threshold_amr'])

    # ── Hamiltonian hyperparameters (Tier 2) ──
    # sigma: uncertainty width for ZZ coupling (replaces beta_grad)
    #
    # `best_hyperparams.json` ne contient PAS sigma, alors que la campagne
    # Optuna gelee l'echantillonne — son meilleur essai trouve 0.0230. Le
    # repli sur 0.05 n'est donc pas un defaut raisonnable : c'est une valeur
    # que rien n'a choisie, appliquee au parametre au coeur de D-9 (la
    # largeur de la fenetre gaussienne). Voir D-22 dans docs/RESULTS.md.
    #
    # On ne leve pas — cela arreterait toute campagne en cours — mais le
    # repli est signale une fois et consigne dans les details du run, pour
    # qu'aucun artefact ne puisse laisser croire que sigma vient de
    # l'entrainement.
    _sigma_defaulted = 'sigma' not in hp
    if _sigma_defaulted:
        warnings.warn(
            "sigma absent des hyperparametres charges : repli sur 0.05, une "
            "valeur qu'aucun essai n'a choisie. La campagne gelee "
            "l'echantillonne pourtant (meilleur essai : 0.0230). Voir D-22.",
            RuntimeWarning, stacklevel=2)
    sigma      = hp.get('sigma',       _defaults.get('sigma', 0.05))
    beta_curl  = hp.get('beta_curl',   _defaults['beta_curl'])
    beta_xpoint = hp.get('beta_xpoint',  _defaults['beta_xpoint'])

    # ── v7 trainable parameters (Tier 2) ──
    gamma_hydro = hp.get('gamma_hydro', _defaults['gamma_hydro'])
    gamma_mag   = hp.get('gamma_mag', _defaults['gamma_mag'])
    kappa       = hp.get('kappa', _defaults['kappa'])
    w_z_frac    = hp.get('w_z_frac', _defaults['w_z_frac'])

    # Percentile du critere relatif : quand aucune cellule n'atteint le
    # seuil absolu de Reynolds-maille, le seuil effectif devient ce
    # percentile du signal. `None` => PhysicalMapper.RELATIVE_PERCENTILE.
    relative_percentile = hp.get('relative_percentile',
                                 _defaults.get('relative_percentile', None))
    

    # ── Physical constants derived from simulation ──
    nu      = grid.L / argus.Re   # kinematic viscosity from Reynolds number
    eta_mhd = grid.L / argus.Rm   # magnetic resistivity from Magnetic Reynolds

    if verbose:
        print(f"Hyperparameters used for this run:")
        print(f"  Encoding:    beta={beta}")
        print(f"  AMR/H_bias:  threshold_amr={threshold_amr}, w_z_frac={w_z_frac}")
        print(f"  Hamiltonian: sigma={sigma}, beta_curl={beta_curl}, beta_xpoint={beta_xpoint}")
        print(f"               gamma_hydro={gamma_hydro}, gamma_mag={gamma_mag}, kappa={kappa}")
        print(f"  Physics:     Re={argus.Re}, Rm={argus.Rm}, nu={nu:.6f}, eta_mhd={eta_mhd:.6f}, cs={c_s}")

    def _details(payload, scoring_error=None, completed=True, abort=None):
        """Sortie detaillee — TOUS les chemins passent par ici.

        `pipeline` a quatre sorties `return_details`. Une seule portait la
        provenance de sigma : celle du chemin de divergence. Autrement
        dit, la trace exigee par D-22 n'existait que sur les runs qu'on
        jette, et jamais sur ceux qu'on publie. Deux chemins censes
        rendre le meme dictionnaire ne le rendaient pas.

        (Champs `completed`/`abort`/`physics_seed`/`physics_noise_amplitude`
        ajoutes depuis : meme principe de schema unique, etendu.)
        """
        out = dict(payload)
        out['scoring_error'] = scoring_error
        out['sigma'] = float(sigma)
        out['sigma_source'] = 'default' if _sigma_defaulted else 'loaded'
        out['completed'] = bool(completed)
        out['abort'] = abort
        out['physics_seed'] = int(
            hot_start_state.get('phys_seed', 0)
            if hot_start_state is not None else 0)
        out['physics_noise_amplitude'] = float(
            hot_start_state.get('physics_noise_amplitude', 0.0)
            if hot_start_state is not None else 0.0)
        return out

    def _divergence_details(scoring_error=None):
        return _details({
            'combined': DIVERGENCE_PENALTY,
            'phys_score': DIVERGENCE_PENALTY,
            'patch_ratio': 1.0,
            'field_errors': {v: DIVERGENCE_PENALTY
                             for v in ['vx', 'vy', 'Bx', 'By', 'Jz']},
        }, scoring_error=scoring_error, completed=False,
           abort={"kind": "invalid_final_score"})

    # Skip Hamiltonian + VQA setup in classical-only mode (no quantum circuit needed)
    HamiltMapper = None
    vqa_runtime = None
    if not classical_only:
        HamiltMapper = PhysicalMapper(
                cs=c_s,
                nu=nu,
                eta_mhd=eta_mhd,
                dx=grid.dx,
                gamma_hydro=gamma_hydro,
                gamma_mag=gamma_mag,
                kappa=kappa,
                sigma=sigma,
                beta_curl=beta_curl,
                beta_xpoint=beta_xpoint,
                w_z_frac=w_z_frac,
                relative_percentile=relative_percentile,
            )

        # Create VQA runtime ONCE — reused across all hybrid steps and VQA calls.
        # All backends (including state_vector) use Aer's compiled C++ engine.
        vqa_runtime = VQARuntime(
            backend_name=argus.backend,
            mode=argus.mode,
            shots=argus.shots,
            opt_level=argus.opt_level,
            seed=getattr(argus, "seed", 0),
        )

    active_patches = []
    classical_patches = []
    ttl_map = {}   # TTL memory: bounds → remaining steps (persists across hybrid steps)
    ttl_map_classical = {}  # Separate TTL memory for classical AMR baseline
    warm_start_cache = {}  # QAOA optimal params: bounds → params (persists across hybrid steps)
    min_size = min_patch_size
    solve_max_depth = max(1, int(log(N / min_size) / log(VQA_N)))
    scan_max_depth = solve_max_depth
    if max_depth_override is not None:
        scan_max_depth = min(solve_max_depth, max_depth_override)
    # scan_max_depth : profondeur de récursion VQA (moins d'appels si override)
    # solve_max_depth: max_depth naturel utilisé par step_layered — garantit que
    #   les patches leaf_depth sont résolus à local_factor=1 (full DNS) et que
    #   les patches coarse_leaf restent très grossiers (local_factor=VQA_N^solve_max_depth)

    total_pixel_classical = 0

    # Préparation de la première itération
    # Step 0 : snapshot de l'état physique à T_START - HYBRID_DT
    # La première hybridation VQA se fera à t = T_START

    Phi_ema = None          # Exponential moving average of stress flux
    EMA_ALPHA = 0.3         # EMA smoothing factor: higher = more weight on recent
    physics_state = sim_quantum.get_fluxes()
    if hot_start_state is not None:
        Phi_ema = mapper.compute_stress_flux(physics_state)

    # 3. Boucle Temporelle
    HYBRID_DT = HYBRID * DT   # Physical time between VQA updates
    # NOTE: `step` was already set above — either from hot_start_state['step']
    # or 0 for cold start.  Do NOT reset it here: the dns_trace is indexed
    # by absolute step number, so the pipeline must read the correct dt.
    step_simulated = 0
    next_lock_time = t_current
    next_hybrid_time = t_current

    while t_current < T_MAX:
        did_hybrid = False
        if t_current >= next_hybrid_time:
            did_hybrid = True
            sim_quantum.tau_buffer = {}
            active_patches_wo_vqa = []
            if classical_only:
                # Classical-only mode: use deterministic detector, no quantum circuit
                active_patches, Phi = run_adaptive_classical(
                    sim_quantum, mapper,
                    threshold_amr=threshold_amr,
                    target_dim=VQA_N,
                    max_depth=scan_max_depth,
                    solve_max_depth=solve_max_depth,
                    min_size=min_size,
                    verbose=verbose,
                    ttl_map=ttl_map,
                )
            else:
                active_patches, active_patches_wo_vqa, Phi = run_adaptive_vqa(
                    sim_quantum, mapper, HamiltMapper, argus, Phi_ema,
                    verbose=verbose,
                    beta=beta,
                    threshold_amr=threshold_amr,
                    target_dim=VQA_N,
                    max_depth=scan_max_depth,
                    solve_max_depth=solve_max_depth,
                    min_size=min_size,
                    vqa_runtime=vqa_runtime,
                    ttl_map=ttl_map,
                    warm_start_cache=warm_start_cache,
                )
            # Update EMA: Φ_ema = α·Φ_current + (1-α)·Φ_ema_old
            # ΔΦ = Φ_current - Φ_ema is smoother than single-step differences,
            # reducing false positives from transient fluctuations.
            if Phi_ema is None:
                Phi_ema = Phi
            else:
              Phi_ema = {
                  key: EMA_ALPHA * Phi[key] + (1.0 - EMA_ALPHA) * Phi_ema[key]
                  for key in Phi
              }

            # Classical AMR: same BFS structure, deterministic detector
            if classic_AMR_comp:
                sim_classical.tau_buffer = {}
                classical_patches, _ = run_adaptive_classical(
                    sim_classical, mapper,
                    threshold_amr=threshold_amr,
                    target_dim=VQA_N,
                    max_depth=scan_max_depth,
                    solve_max_depth=solve_max_depth,
                    min_size=min_size,
                    verbose=verbose,
                    ttl_map=ttl_map_classical,
                )

            steps_hybrid_count += 1
            next_hybrid_time += HYBRID_DT
            

        if t_current >= next_lock_time:
            if sim_temoin is not None:
                plot_amr_state(sim_temoin, [], t_current, VQA_N, verbose=verbose, save_dir=save_dir, suffix="dns")
            plot_amr_state(sim_quantum, active_patches, t_current, VQA_N, verbose=verbose, save_dir=save_dir, suffix="quantum_amr")
            plot_amr_state(sim_quantum, active_patches_wo_vqa, t_current, VQA_N, verbose=verbose, save_dir=save_dir, suffix="quantum_amr_wo_vqa")
            if classic_AMR_comp:
                plot_amr_state(sim_classical, classical_patches, t_current, VQA_N, verbose=verbose, save_dir=save_dir, suffix="classic_amr")
            next_lock_time += HYBRID_DT

        # Adaptive dt: use the SAME dt for ALL solvers to keep them synchronized
        if dns_presence:
            if step in dns_trace:
                dt = dns_trace[step]['dt']
                sim_quantum.dt = dt
                if classic_AMR_comp:
                    sim_classical.dt = dt
            else:
                raise RuntimeError(
                    f"dns_trace ended before pipeline step {step}; refusing "
                    "to score a truncated trajectory")
        else:
            dt = DT
            dt_q = sim_quantum.adapt_dt(cfl_target=0.4)
            dt_t = sim_temoin.adapt_dt(cfl_target=0.4)
            if classic_AMR_comp:
                dt_c = sim_classical.adapt_dt(cfl_target=0.4)
                dt = min(dt_q, dt_t, dt_c, T_MAX - t_current)
                sim_classical.dt = dt
            else :
                dt = min(dt_q, dt_t, T_MAX - t_current)
            sim_quantum.dt = dt
            sim_temoin.dt = dt

        if sim_temoin is not None:
            sim_temoin.step_full()

        pixels = sim_quantum.step_layered(
            active_patches, max_depth=solve_max_depth, target_dim=VQA_N,
        )
        pixels_classical = 0
        if classic_AMR_comp:
            pixels_classical = sim_classical.step_layered(
                classical_patches, max_depth=solve_max_depth, target_dim=VQA_N,
            )


        total_pixel_used += pixels
        if classic_AMR_comp:
            total_pixel_classical += pixels_classical
        t_current += dt
        step += 1
        step_simulated += 1

        # --- Divergence guard ---
        quantum_diverged = sim_quantum.is_diverged()
        temoin_diverged = sim_temoin.is_diverged() if sim_temoin is not None else False
        classical_diverged = (classic_AMR_comp and sim_classical.is_diverged())
        cfl_exceeded = sim_quantum.check_cfl() > 1.0
        if (quantum_diverged or temoin_diverged or classical_diverged
                or cfl_exceeded):
            if verbose:
                print(f"[ABORT] Divergence detected at step {step-1} (t={t_current:.4f})")
            # Try to compute a partial score from the fields that haven't
            # diverged yet.  For each field we check individually: if it
            # contains NaN/inf we assign DIVERGENCE_PENALTY for that field
            # only; otherwise we keep the real (partial) error.  This lets
            # Optuna learn from the *valid* part of the simulation.
            scoring_error = None
            try:
                q_fluxes = sim_quantum.get_fluxes()
                # Pick the best available reference
                if dns_presence:
                    last_ok = step - 1
                    while last_ok >= 0 and 'fluxes' not in dns_trace.get(last_ok, {}):
                        last_ok -= 1
                    if last_ok >= 0:
                        ref_fluxes = dns_trace[last_ok]['fluxes']
                    elif sim_temoin is not None:
                        ref_fluxes = sim_temoin.get_fluxes()
                    else:
                        raise RuntimeError("No reference available")
                else:
                    ref_fluxes = sim_temoin.get_fluxes()
                # Use the same weighted metric as the completed-run score.
                variables = ['vx', 'vy', 'Bx', 'By', 'Jz']
                field_errors = {}
                n_diverged = 0
                _ref_finite = all(np.all(np.isfinite(ref_fluxes[v]))
                                  for v in variables)
                if _ref_finite:
                    _w = instability_weight_map(ref_fluxes).flatten()
                    _w_sum = np.sum(_w)
                for var in variables:
                    arr_q = q_fluxes[var]
                    arr_r = ref_fluxes[var]
                    if (not _ref_finite
                            or np.any(~np.isfinite(arr_q))
                            or np.any(~np.isfinite(arr_r))):
                        field_errors[var] = DIVERGENCE_PENALTY
                        n_diverged += 1
                    else:
                        field_errors[var] = weighted_relative_error(
                            arr_q, arr_r, _w, _w_sum)
                phys_score = np.mean(list(field_errors.values()))
                patch_ratio = total_pixel_used / (step_simulated * N**2) if step_simulated > 0 else 1.0
                combined = (phys_score + lambda_cost * patch_ratio) / (1 + lambda_cost)
                # If everything diverged, fall back to flat penalty
                if n_diverged == len(variables):
                    combined = DIVERGENCE_PENALTY
                    phys_score = DIVERGENCE_PENALTY
                    patch_ratio = 1.0
                    field_errors = {v: DIVERGENCE_PENALTY for v in variables}
                if verbose:
                    print(f"[DIVERGE] Partial score: combined={combined:.4f}, "
                          f"diverged_fields={n_diverged}/{len(variables)}")
            except Exception as exc:
                # Ce filet attrapait TOUTE exception sans la nommer : une
                # erreur de programmation dans le calcul du score etait
                # rapportee comme une divergence physique, et l'essai
                # penalise au lieu d'echouer. On garde le filet — un essai
                # Optuna ne doit pas faire tomber la campagne — mais la
                # cause est desormais journalisee et rendue avec le
                # resultat, donc distinguable d'une vraie divergence.
                import traceback
                print(f"[SCORING-ERROR] {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                combined = DIVERGENCE_PENALTY
                phys_score = DIVERGENCE_PENALTY
                patch_ratio = 1.0
                field_errors = {v: DIVERGENCE_PENALTY for v in ['vx','vy','Bx','By','Jz']}
                scoring_error = f"{type(exc).__name__}: {exc}"
            if return_details:
                abort = {
                    'kind': 'numerical_divergence',
                    'step': int(step - 1),
                    'time': float(t_current),
                    'quantum_diverged': bool(quantum_diverged),
                    'reference_diverged': bool(temoin_diverged),
                    'classical_diverged': bool(classical_diverged),
                    'cfl_exceeded': bool(cfl_exceeded),
                }
                return _details({'combined': combined, 'phys_score': phys_score,
                                 'patch_ratio': patch_ratio,
                                 'field_errors': field_errors},
                                scoring_error=scoring_error,
                                completed=False, abort=abort)
            return combined

        # --- Intermediate scoring for Optuna pruning ---
        if trial is not None and did_hybrid and steps_hybrid_count > 1:
            score_result = {}
            dns_entry = dns_trace.get(step - 1, {}) if dns_presence else {}
            if 'fluxes' in dns_entry:
                ground_truth_fluxes = dns_entry['fluxes']
                score_result = score(sim_quantum.get_fluxes(), ground_truth_fluxes, lambda_cost, total_pixel_used, step_simulated, N**2)
            elif sim_temoin is not None:
                score_result = score(sim_quantum.get_fluxes(), sim_temoin.get_fluxes(), lambda_cost,
                                    total_pixel_used, step_simulated, N**2)
            else:
                score_result = None
            intermediate = score_result['combined'] if score_result else float('nan')
            if np.isnan(intermediate) or np.isinf(intermediate):
                intermediate = DIVERGENCE_PENALTY
            trial.report(intermediate, step=step)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        if verbose:
            physics_state = sim_quantum.get_fluxes()
            max_current = np.max(np.abs(physics_state['Jz']))
            print(f"Step {step} (t={t_current:.4f}) - Pixel ratio : {pixels/(N**2):.4f} -  Max Jz: {max_current:.4f}")

    # ── Final scoring: Q-HAS vs reference ──
    score_result = {}
    score_classical = {}
    if dns_presence:
        last_step = step - 1
        while last_step >= 0 and 'fluxes' not in dns_trace.get(last_step, {}):
            last_step -= 1
        if last_step < 0:
            if return_details:
                return _divergence_details(
                    scoring_error='no dns flux snapshot before the last step')
            return DIVERGENCE_PENALTY
        ground_truth_fluxes = dns_trace[last_step]['fluxes']
        score_result = score(sim_quantum.get_fluxes(), ground_truth_fluxes, lambda_cost, total_pixel_used, step_simulated, N**2)
        if classic_AMR_comp:
            score_classical = score(sim_classical.get_fluxes(), ground_truth_fluxes, lambda_cost, total_pixel_classical, step_simulated, N**2)
    else:
        ref_fluxes = sim_temoin.get_fluxes()
        score_result = score(sim_quantum.get_fluxes(), ref_fluxes, lambda_cost,
                            total_pixel_used, step_simulated, N**2)
        if classic_AMR_comp:
            score_classical = score(sim_classical.get_fluxes(), ref_fluxes, lambda_cost,
                                total_pixel_classical, step_simulated, N**2)

    final_score = score_result['combined']
    if np.isnan(final_score) or np.isinf(final_score):
        if return_details:
            return _divergence_details(
                scoring_error='final score is NaN or inf')
        return DIVERGENCE_PENALTY

    # Store decomposed metrics for hyperparameter analysis
    if trial is not None:
        trial.set_user_attr('phys_score', float(score_result['phys_score']))
        trial.set_user_attr('patch_ratio', float(score_result['patch_ratio']))
        for field, err in score_result['field_errors'].items():
            trial.set_user_attr(f'error_{field}', float(err))
        # Store classical baseline for comparison
        if classic_AMR_comp:
            trial.set_user_attr('classical_combined', float(score_classical['combined']))
            trial.set_user_attr('classical_phys_score', float(score_classical['phys_score']))
            trial.set_user_attr('classical_patch_ratio', float(score_classical['patch_ratio']))

    if verbose:
        if classic_AMR_comp:
            print("\n" + "=" * 60)
            print("FINAL COMPARISON: Q-HAS vs Classical AMR vs DNS")
            print("=" * 60)
            print(f"{'Metric':<20} {'Q-HAS':>12} {'Classical':>12}")
            print(f"{'-'*20} {'-'*12} {'-'*12}")
            print(f"{'combined':.<20} {score_result['combined']:>12.6f} {score_classical['combined']:>12.6f}")
            print(f"{'phys_score':.<20} {score_result['phys_score']:>12.6f} {score_classical['phys_score']:>12.6f}")
            print(f"{'patch_ratio':.<20} {score_result['patch_ratio']:>12.4f} {score_classical['patch_ratio']:>12.4f}")
            for field in score_result['field_errors']:
                q_err = score_result['field_errors'][field]
                c_err = score_classical['field_errors'][field]
                better = "<" if q_err < c_err else ">"
                print(f"{'error_' + field:.<20} {q_err:>12.6f} {c_err:>12.6f}  Q-HAS {better} Classical")
            print("=" * 60)
        else:
            print("\n" + "=" * 40)
            print("FINAL SCORE: Q-HAS vs Reference")
            print("=" * 40)
            print(f"{'Metric':<20} {'Q-HAS':>12}")
            print(f"{'-'*20} {'-'*12}")
            print(f"{'combined':.<20} {score_result['combined']:>12.6f}")
            print(f"{'phys_score':.<20} {score_result['phys_score']:>12.6f}")
            print(f"{'patch_ratio':.<20} {score_result['patch_ratio']:>12.4f}")
            for field in score_result['field_errors']:
                q_err = score_result['field_errors'][field]
                print(f"{'error_' + field:.<20} {q_err:>12.6f}")
            print("=" * 40)

    if return_details:
        return _details(score_result)
    return final_score


def instability_weight_map(ref_fluxes):
    """Carte de poids asymétrique, construite sur la référence.

    w = 1 + 0.25 × (|Jz|/⟨|Jz|⟩ + |ωz|/⟨|ωz|⟩)

    Les régions à fort courant OU forte vorticité pèsent davantage : rater
    une instabilité coûte plus cher que raffiner une région calme.

    Le chemin de divergence et le score final utilisent cette même carte.
    """
    Jz_abs = np.abs(ref_fluxes['Jz'])
    Jz_mean = np.mean(Jz_abs) + 1e-10

    vx_ref, vy_ref = ref_fluxes['vx'], ref_fluxes['vy']
    omega_z = np.abs(
        (np.roll(vy_ref, -1, axis=AXIS_X) - vy_ref)
        - (np.roll(vx_ref, -1, axis=AXIS_Y) - vx_ref)
    )
    omega_mean = np.mean(omega_z) + 1e-10

    instability_weight = 0.5
    return 1.0 + instability_weight * (
        Jz_abs / Jz_mean + omega_z / omega_mean
    ) * 0.5


def weighted_relative_error(arr_q, arr_r, weight_flat, weight_sum):
    """Erreur L2 relative pondérée d'un champ contre sa référence.

    Vaut 0 sur une reconstruction exacte et 1 quand le bras rend zéro.
    """
    diff_sq = (arr_q.flatten() - arr_r.flatten()) ** 2
    weighted_rmse = np.sqrt(np.sum(weight_flat * diff_sq) / weight_sum)
    ref_rms = np.sqrt(np.sum(weight_flat * arr_r.flatten() ** 2) / weight_sum)
    return float(weighted_rmse / (ref_rms + 1e-10))


def score(sim_quantum_fluxes, sim_temoin_fluxes, lambda_cost, total_pixel_used, total_steps, N_square, verbose=False):
    """
    Computes a multi-variable physical fidelity score with asymmetric weighting.

    Regions with high current density (|Jz|) AND high vorticity are weighted
    MORE heavily because missing an instability (false negative) is far more
    damaging than wasting compute on a quiet region (false positive).

    Returns a dict with:
        'combined'     : weighted score  (phys + lambda*patch_ratio) / (1+lambda)
        'phys_score'   : average relative L2 error across 5 MHD fields
        'patch_ratio'  : fraction of grid actively refined (0=minimal, 1=full)
        'field_errors' : {vx, vy, Bx, By, Jz} individual relative L2 errors
    """

    variables = ['vx', 'vy', 'Bx', 'By', 'Jz']

    # ── Asymmetric weight map ──
    # Extraite dans `instability_weight_map` pour que le chemin de
    # divergence de `pipeline()` emploie EXACTEMENT la meme ponderation.
    weight_map = instability_weight_map(sim_temoin_fluxes)
    weight_flat = weight_map.flatten()
    weight_sum = np.sum(weight_flat)

    total_error = 0.0
    detailed_errors = {}

    for var in variables:
        rel_err = weighted_relative_error(
            sim_quantum_fluxes[var], sim_temoin_fluxes[var],
            weight_flat, weight_sum)
        detailed_errors[var] = rel_err
        total_error += rel_err

    phys_score = total_error / len(variables)

    # D-67 : `total_steps == 0` signifie qu'AUCUN pas n'a ete integre.
    # Le repli `avg_pixel_used = N_square` transformait cela en
    # `patch_ratio = 1.0`, donc en `combined = lambda/(1+lambda)` -- un
    # nombre parfaitement plausible (0.333333 a lambda=0.5) pour un run qui
    # n'avait rien calcule. C'est ainsi que D-66 est reste invisible.
    # Un run vide doit crier, pas se noter.
    if total_steps <= 0:
        raise ValueError(
            f"score() appele avec total_steps={total_steps} : aucun pas de "
            f"temps n'a ete integre, il n'y a rien a noter. Verifier que "
            f"T_MAX est posterieur a T_START (voir D-66 et D-67).")

    avg_pixel_used = total_pixel_used / total_steps
    patch_ratio = avg_pixel_used / N_square

    if verbose:
        print(f"Patch ratio : {patch_ratio:.4f}")
    final_combined_score = (phys_score + lambda_cost * patch_ratio) / (1 + lambda_cost)

    return {
        'combined': final_combined_score,
        'phys_score': phys_score,
        'patch_ratio': patch_ratio,
        'field_errors': detailed_errors,
    }

if __name__ == "__main__":
    main()
