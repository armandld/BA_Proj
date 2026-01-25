import argparse
import sys
from math import log


import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.refinement import run_adaptive_vqa

from visual import plot_amr_state

from help_visual import plot_grid_topology, plot_flux_on_edges

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
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"])
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--method", default="COBYLA", choices=["COBYLA", "L-BFGS-B", "Powell"])
    parser.add_argument("--opt-level", type=int, default=1, choices=[0,1,2,3], help="Optimization level for transpilation.")
    parser.add_argument("--K-opt", type=int, default=100, help="Maximum number of iterations for the optimizer.")
    parser.add_argument("--eps", type=float, default=1e-2, help="Convergence tolerance for the optimizer.")

    args = parser.parse_args()

    verbose = args.verbose

    N = args.dns_resolution                   # Résolution moyenne (DNS)
    VQA_N = args.grid_size                    # Résolution Grossière
    T_MAX = args.t_max                        # Temps final
    DT = args.dt                              # Pas de temps
    HYBRID = int(args.hybrid_dt / DT)         # Fréquence de mise à jour hybride

    
    argus = SimpleNamespace(
        reps=args.reps if args.reps > 0 else (VQA_N-1) * 2, # 2 for 2D, 3 for 3D 
        mode=args.mode, 
        backend=args.backend, 
        shots=args.shots, 
        method=args.method, 
        opt_level=args.opt_level, 
        AdvAnomaliesEnable=args.AdvAnomaliesEnable,
        K_opt=100,   # Max iterations for optimizer
        eps=1e-2     # Convergence tolerance
    )

    pipeline(N, VQA_N, T_MAX, DT, HYBRID, verbose, argus, lambda_cost=0.5)

def pipeline(N, VQA_N, T_MAX, DT, HYBRID, verbose, argus, hyperparams=None, lambda_cost=0.5):

    #Paramètres physiques
    eta = 0.001       # Faible résistivité pour laisser l'instabilité grandir
    Bz_guide = 0.1    # Faible champ guide pour la stabilité
    c_s = 1.0         # Référence de vitesse acoustique

    STEPS = int(T_MAX / DT)

    total_patches_used = 0
    steps_hybrid_count = 0
    
    # 2. Initialisation
    if verbose:
        print(f"Initialisation Orszag-Tang (Grille {N}x{N}) avec pas de temps {DT} et hybrid every {HYBRID} steps pour un temp total de {T_MAX} ({STEPS} steps)...")
    
    grid = PeriodicGrid(resolution_N=N)
    sim_quantum = MHDSolver(grid, dt=DT, Re=1000, Rm=1000) # A CHANGER: PARAMETRER R_max et R_e max
    sim_quantum.init_kelvin_helmholtz() #init_kelvin_helmholtz() init_orszag_tang()
    sim_temoin = MHDSolver(grid, dt=DT, Re=1000, Rm=1000)  # Pour la visualisation finale
    sim_temoin.init_kelvin_helmholtz() #init_kelvin_helmholtz() init_orszag_tang()
    mapper = AngleMapper(v0=1.0, B0=1.0, w_shock=2.0, w_shear=1.0)

    alpha =hyperparams.get('alpha',1.0) if hyperparams else 1.0
    beta =hyperparams.get('beta',1.0) if hyperparams else 1.0
    threshold=hyperparams.get('threshold',0.5) if hyperparams else 0.5

    #Hamiltonian ones:
    bias=hyperparams.get('bias',4.0) if hyperparams else 4.0
    gamma1=hyperparams.get('gamma1',1.0) if hyperparams else 1.0
    gamma2=hyperparams.get('gamma2',2.0) if hyperparams else 2.0
    Rm_crit=hyperparams.get('Rm_crit',1000.0) if hyperparams else 1000.0
    delta_shock=hyperparams.get('delta_shock',5.0) if argus.AdvAnomaliesEnable and hyperparams else 5.0
    d_kink=hyperparams.get('d_kink',2.0) if argus.AdvAnomaliesEnable and hyperparams else 2.0
    epsilon=hyperparams.get('epsilon',1e-6) if argus.AdvAnomaliesEnable and hyperparams else 1e-6
      
    
    HamiltMapper = PhysicalMapper(
            c_s, eta, Bz_guide,
            bias=bias, gamma1=gamma1, gamma2=gamma2,
            Rm_crit=Rm_crit, delta_shock=delta_shock,
            d_kink=d_kink, epsilon=epsilon
        )

    active_patches = []
    max_depth = int(log(N)/log(VQA_N))+1
    
    # Préparation de la première itération
    Phi_prev = None
    physics_state = sim_quantum.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)
    """
    if verbose:
        plot_grid_topology(grid)
        plot_flux_on_edges(grid, Phi)
        print(f"Lancement pour {STEPS} pas de temps...")
    """
    # 3. Boucle Temporelle
    for t in range(STEPS):
        if t % HYBRID == 0:
            active_patches, Phi = run_adaptive_vqa(
                sim_quantum, mapper, HamiltMapper, argus, Phi_prev, #Phi_prev = Phi in this function
                verbose=verbose,
                alpha = alpha,
                beta = beta,
                threshold=threshold,
                target_dim = VQA_N,
                max_depth= max_depth,
                max_patches=N,
                min_size = 6,
                DT=DT
            )
            Phi_prev = Phi

            total_patches_used += len(active_patches)
            steps_hybrid_count += 1

        sim_quantum.step_layered(active_patches, max_depth)
        sim_temoin.step_full()
        if t % HYBRID == 0 and verbose:
                plot_amr_state(sim_quantum, active_patches, t, DT, t, VQA_N)
                plot_amr_state(sim_temoin, [], t, DT, t, VQA_N)

        max_current = np.max(np.abs(physics_state['Jz']))
        if verbose:
            print(f"Step {t}/{STEPS} (t={t*DT:.4f}) - Max Jz: {max_current:.4f}")

    print("SCORE : ", score(sim_quantum, sim_temoin, lambda_cost, total_patches_used, steps_hybrid_count, N**2))
    return score(sim_quantum, sim_temoin, lambda_cost, total_patches_used, steps_hybrid_count, N**2)

        
"""
    # 4. Visualisation Finale
    print("Simulation terminée. Génération du plot comparatif...")
    final_state_quantum = sim_quantum.get_fluxes()
    final_state_temoin = sim_temoin.get_fluxes()

    # --- Préparation de la comparaison ---
    # On récupère les données Jz
    Jz_q = final_state_quantum['Jz'].T
    Jz_t = final_state_temoin['Jz'].T

    # CRITIQUE : Calculer une échelle de couleurs commune pour une comparaison honnête.
    # On cherche le max absolu sur les DEUX simulations.
    global_max = max(np.max(np.abs(Jz_q)), np.max(np.abs(Jz_t)))
    vmin, vmax = -global_max, global_max

    # Création de la figure avec 1 ligne et 2 colonnes
    # figsize=(largeur, hauteur) -> on élargit pour faire tenir les deux
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # --- Plot 1 : Simulation Quantique Hybride ---
    im1 = ax1.imshow(Jz_q, origin='lower', cmap='RdBu', 
                     extent=[0, 2*np.pi, 0, 2*np.pi], vmin=vmin, vmax=vmax)
    ax1.set_title(f"Q-HAS Hybride (AMR) \n (t={T_MAX})")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # --- Plot 2 : Simulation Témoin (DNS) ---
    im2 = ax2.imshow(Jz_t, origin='lower', cmap='RdBu', 
                     extent=[0, 2*np.pi, 0, 2*np.pi], vmin=vmin, vmax=vmax)
    ax2.set_title(f"Témoin Classique (DNS Full) \n (t={T_MAX})")
    ax2.set_xlabel('x')
    # Pas besoin de ylabel sur le deuxième plot car sharey=True

    # --- Finitions ---
    # Une seule barre de couleur pour les deux, placée à droite
    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    cbar.set_label('Densité de Courant Jz (Échelle commune)')

    plt.suptitle(f"Comparaison Finale Kelvin-Helmholtz", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuste pour le titre principal
    plt.show()
"""

def score(sim_quantum, sim_temoin, lambda_cost, total_patches_used, steps_hybrid_count, N_square):
    """
    Computes a multi-variable physical fidelity score.
    Returns the average relative L2 error across all physical fields.
    """
    # 1. Get States
    st_q = sim_quantum.get_fluxes()
    st_t = sim_temoin.get_fluxes()

    # 2. Define Variables to compare
    # We include Jz because it captures fine-scale gradients (shocks)
    # We include Bx, By because they capture magnetic topology
    # We include vx, vy because they capture kinetic energy
    variables = ['vx', 'vy', 'Bx', 'By', 'Jz']
    
    total_error = 0.0
    detailed_errors = {}
    
    # 3. Compute Error for each field
    for var in variables:
        # Flattening ensures we compute one single scalar norm for the whole grid
        # independent of shapes (e.g. .T issues)
        arr_q = st_q[var].flatten()
        arr_t = st_t[var].flatten()

        # L2 Norm of the Difference
        diff_norm = np.linalg.norm(arr_q - arr_t)
        
        # L2 Norm of the Ground Truth (Reference)
        ref_norm = np.linalg.norm(arr_t)
        
        # Safety epsilon to avoid division by zero (e.g. if vy is 0 everywhere)
        epsilon_security = 1e-10
        
        rel_err = diff_norm / (ref_norm + epsilon_security)
        
        detailed_errors[var] = rel_err
        total_error += rel_err

    # 4. Average Score
    # This treats magnetic accuracy and kinetic accuracy with equal importance
    phys_score = total_error / len(variables)

    if steps_hybrid_count > 0:
        avg_patches = total_patches_used / steps_hybrid_count
    else:
        avg_patches = N_square # Cas par défaut (pire cas)
        
    patch_ratio = avg_patches / N_square
    
    # C. Score Total
    # On cherche à MINIMISER le score.
    # Score = Erreur + lambda * Coût
    # lambda règle l'importance de l'économie. 
    # Une valeur de 0.5 signifie qu'on tolère un peu d'erreur si on économise beaucoup de calcul.
    
    final_combined_score = phys_score + (lambda_cost * patch_ratio)

    return final_combined_score

if __name__ == "__main__":
    main()