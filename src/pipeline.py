import argparse
import sys
from math import log


import numpy as np
from types import SimpleNamespace

from Simulation.grid import PeriodicGrid
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

def main():
    sys.stdout.reconfigure(line_buffering=True) # Pour un affichage immédiat des print() à enlever pour une meilleure perf

    parser = argparse.ArgumentParser(description="Mapping VQA")
    parser.add_argument("--out-dir", default="../data", help="Output directory for mapping")
    parser.add_argument("--in-file", default="../input/mapping_input.json", help="Input directory for mapping")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--AdvAnomaliesEnable", action="store_true")
    parser.add_argument("--grid-size", type=int, default=2, help="Coarse grid dimension N (NxN)")
    parser.add_argument("--dns-resolution", type=int, default=256, help="High-Res Grid for Ground Truth")
    parser.add_argument("--t-max", type=float, default=1.0, help="Simulation end time")
    parser.add_argument("--dt", type=float, default=1e-4, help="Time step size")
    parser.add_argument("--hybrid-dt", type=float, default=0.1, help="Hybrid simulation time step size")
    parser.add_argument("--reps", type=int, default=-1, required=False, help="Number of repetitions for the QAOA ansatz.")
    parser.add_argument("--mode", default="simulator", choices=["simulator", "hardware"])
    parser.add_argument("--backend", default="state_vector", choices=["aer", "estimator","state_vector"])
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--method", default="L-BFGS-B", choices=["COBYLA", "L-BFGS-B", "Powell"])
    parser.add_argument("--opt-level", type=int, default=1, choices=[0,1,2,3], help="Optimization level for transpilation.")
    parser.add_argument("--K-opt", type=int, default=80, help="Maximum number of iterations for the optimizer.")
    parser.add_argument("--eps", type=float, default=1e-2, help="Convergence tolerance for the optimizer.")
    parser.add_argument("--scenario", default="lamb_oseen_vortex",
                        choices=["orszag_tang", "kelvin_helmholtz",
                                 "magnetic_twist", "noisy_uniform",
                                 "harris_tearing", "double_tearing",
                                 "lamb_oseen_vortex", "island_coalescence",
                                 "mhd_rotor", "ghost_twisting"],
                        help="Initial condition scenario")

    args = parser.parse_args()

    verbose = args.verbose

    N = args.dns_resolution                   # Résolution moyenne (DNS)
    VQA_N = args.grid_size                    # Résolution Grossière
    T_MAX = args.t_max                        # Temps final
    DT = args.dt                              # Pas de temps
    HYBRID = int(args.hybrid_dt / DT)         # Fréquence de mise à jour hybride

    Re = 1000
    Rm = 1000

    argus = SimpleNamespace(
        reps=args.reps if args.reps > 0 else (VQA_N-1) * 2, # 2 for 2D, 3 for 3D
        mode=args.mode,
        backend=args.backend,
        shots=args.shots,
        method=args.method,
        opt_level=args.opt_level,
        AdvAnomaliesEnable=args.AdvAnomaliesEnable,
        K_opt=args.K_opt,
        eps=args.eps,
        eta=0.001,       # Faible résistivité pour laisser l'instabilité grandir
        Bz_guide=0.1,    # Faible champ guide pour la stabilité
        c_s=1.0,         # Référence de vitesse acoustique
        Re= Re,          # Reynolds number
        Rm= Rm           # Magnetic Reynolds number
    )

    N_TRAINING         = N
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

        "ghost_twisting" : {
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
        },
    }
    dns_trace, hot_start_state = precompute_dns(PHASE[args.scenario])

    print(f"Starting pipeline... saved in{args.out_dir}")
    pipeline(N, VQA_N, T_MAX, DT, HYBRID, verbose, argus, lambda_cost=0.5, trial= None, classic_AMR_comp=True, dns_trace=dns_trace, hot_start_state=hot_start_state, max_depth_override= 4, scenario=args.scenario, save_dir=args.out_dir)



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
        'ghost_twisting':      sim.init_ghost_twisting,
    }
    init_map[scenario]()


def pipeline(N, VQA_N, T_MAX, DT, HYBRID, verbose, argus, hyperparams=None, lambda_cost=0.5, trial=None, classic_AMR_comp = False, dns_trace=None, hot_start_state=None, min_patch_size=6, max_depth_override=None, scenario='orszag_tang', save_dir=None, return_details=False, classical_only=False):

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
    _defaults = load_hyperparams()
    hp = {**_defaults, **(hyperparams or {})}

    """
    # ── Encoding hyperparameters (Tier 1) ──
    beta            = hp.get('beta', 0.8683654005538312)
    threshold_amr   = hp.get('threshold_amr', 0.5891029179142372)

    # ── Hamiltonian hyperparameters (Tier 2) ──
    # Split Michelson sensitivity: fall back to shared beta_michelson
    beta_grad  = hp.get('beta_grad',  0.44356154664122427)
    beta_curl  = hp.get('beta_curl',  0.44356154664122427)
    beta_xpoint = hp.get('beta_xpoint', 0.44356154664122427)

    # ── v7 trainable parameters (Tier 2) ──
    gamma_hydro = hp.get('gamma_hydro', 1.6529193289578792)
    gamma_mag   = hp.get('gamma_mag', 3.3558897780227754)
    kappa       = hp.get('kappa', 5.41718485540701)
    w_z_frac    = hp.get('w_z_frac',0.49259437288557695)
    """

    
    # ── Encoding hyperparameters (Tier 1) ──
    beta            = hp.get('beta', _defaults['beta'])
    threshold_amr   = hp.get('threshold_amr', _defaults['threshold_amr'])

    # ── Hamiltonian hyperparameters (Tier 2) ──
    # sigma: uncertainty width for ZZ coupling (replaces beta_grad)
    sigma      = hp.get('sigma',       _defaults.get('sigma', 0.05))
    beta_curl  = hp.get('beta_curl',   _defaults['beta_curl'])
    beta_xpoint = hp.get('beta_xpoint',  _defaults['beta_xpoint'])

    # ── v7 trainable parameters (Tier 2) ──
    gamma_hydro = hp.get('gamma_hydro', _defaults['gamma_hydro'])
    gamma_mag   = hp.get('gamma_mag', _defaults['gamma_mag'])
    kappa       = hp.get('kappa', _defaults['kappa'])
    w_z_frac    = hp.get('w_z_frac', _defaults['w_z_frac'])
    

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
            )

        # Create VQA runtime ONCE — reused across all hybrid steps and VQA calls.
        # All backends (including state_vector) use Aer's compiled C++ engine.
        vqa_runtime = VQARuntime(
            backend_name=argus.backend,
            mode=argus.mode,
            shots=argus.shots,
            opt_level=argus.opt_level,
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
    if hot_start_state is not None :
        first_step_with_flux = min([s for s, v in dns_trace.items() if 'fluxes' in v])
        Phi_ema = mapper.compute_stress_flux(dns_trace[first_step_with_flux]['fluxes'])
    physics_state = sim_quantum.get_fluxes()

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
                break
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
        temoin_diverged = sim_temoin.is_diverged() if sim_temoin is not None else False
        if sim_quantum.is_diverged() or temoin_diverged or (classic_AMR_comp and sim_classical.is_diverged()) or sim_quantum.check_cfl() > 1.0:
            if verbose:
                print(f"[ABORT] Divergence detected at step {step-1} (t={t_current:.4f})")
            # Try to compute a partial score from the fields that haven't
            # diverged yet.  For each field we check individually: if it
            # contains NaN/inf we assign DIVERGENCE_PENALTY for that field
            # only; otherwise we keep the real (partial) error.  This lets
            # Optuna learn from the *valid* part of the simulation.
            DIVERGENCE_PENALTY = 10.0
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
                # Score each field individually — keep valid ones
                variables = ['vx', 'vy', 'Bx', 'By', 'Jz']
                field_errors = {}
                n_diverged = 0
                for var in variables:
                    arr_q = q_fluxes[var]
                    arr_r = ref_fluxes[var]
                    if np.any(~np.isfinite(arr_q)) or np.any(~np.isfinite(arr_r)):
                        field_errors[var] = DIVERGENCE_PENALTY
                        n_diverged += 1
                    else:
                        ref_rms = np.sqrt(np.mean(arr_r**2))
                        rel_err = np.sqrt(np.mean((arr_q - arr_r)**2)) / (ref_rms + 1e-10)
                        field_errors[var] = float(rel_err)
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
            except Exception:
                combined = DIVERGENCE_PENALTY
                phys_score = DIVERGENCE_PENALTY
                patch_ratio = 1.0
                field_errors = {v: DIVERGENCE_PENALTY for v in ['vx','vy','Bx','By','Jz']}
            if return_details:
                return {'combined': combined, 'phys_score': phys_score,
                        'patch_ratio': patch_ratio, 'field_errors': field_errors}
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
        last_step = step if step in dns_trace else step - 1
        while last_step >= 0 and 'fluxes' not in dns_trace.get(last_step, {}):
            last_step -= 1
        if last_step < 0:
            DIVERGENCE_PENALTY = 10.0
            if return_details:
                return {'combined': DIVERGENCE_PENALTY, 'phys_score': DIVERGENCE_PENALTY,
                        'patch_ratio': 1.0, 'field_errors': {v: DIVERGENCE_PENALTY for v in ['vx','vy','Bx','By','Jz']}}
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
        DIVERGENCE_PENALTY = 10.0
        if return_details:
            return {'combined': DIVERGENCE_PENALTY, 'phys_score': DIVERGENCE_PENALTY,
                    'patch_ratio': 1.0, 'field_errors': {v: DIVERGENCE_PENALTY for v in ['vx','vy','Bx','By','Jz']}}
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
        return score_result
    return final_score


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
    # Combine |Jz| (current sheets) AND |ωz| (vorticity) for weighting.
    # This catches instabilities where either B or v is active.
    Jz_ref = sim_temoin_fluxes['Jz']
    vx_ref = sim_temoin_fluxes['vx']
    vy_ref = sim_temoin_fluxes['vy']

    Jz_abs = np.abs(Jz_ref)
    Jz_mean = np.mean(Jz_abs) + 1e-10

    # Discrete vorticity |ωz| ≈ |∂vy/∂x − ∂vx/∂y|
    omega_z = np.abs(
        (np.roll(vy_ref, -1, axis=1) - vy_ref)
        - (np.roll(vx_ref, -1, axis=0) - vx_ref)
    )
    omega_mean = np.mean(omega_z) + 1e-10

    # w = 1 + 0.5*(|Jz|/mean + |ωz|/mean) — weights instability regions
    instability_weight = 0.5
    weight_map = 1.0 + instability_weight * (
        Jz_abs / Jz_mean + omega_z / omega_mean
    ) * 0.5
    weight_flat = weight_map.flatten()
    weight_sum = np.sum(weight_flat)

    total_error = 0.0
    detailed_errors = {}

    for var in variables:
        arr_q = sim_quantum_fluxes[var].flatten()
        arr_t = sim_temoin_fluxes[var].flatten()

        # Weighted L2: sum(w * (q-t)^2) / sum(w), then sqrt for norm-like behavior
        diff_sq = (arr_q - arr_t) ** 2
        weighted_mse = np.sum(weight_flat * diff_sq) / weight_sum
        weighted_rmse = np.sqrt(weighted_mse)

        ref_rms = np.sqrt(np.sum(weight_flat * arr_t ** 2) / weight_sum)
        epsilon_security = 1e-10
        rel_err = weighted_rmse / (ref_rms + epsilon_security)

        detailed_errors[var] = rel_err
        total_error += rel_err

    phys_score = total_error / len(variables)

    if total_steps > 0:
        avg_pixel_used = total_pixel_used / total_steps
    else:
        avg_pixel_used = N_square

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
