import argparse
import sys
from math import log


import numpy as np
import matplotlib.pyplot as plt

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
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
    parser.add_argument("--depth", type=int, required=False, help="Depth of the ULA ansatz.")
    parser.add_argument("--mode", default="simulator", choices=["simulator", "hardware"])
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"])
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--method", default="COBYLA", choices=["COBYLA", "L-BFGS-B", "Powell"])
    parser.add_argument("--opt-level", type=int, default=1, choices=[0,1,2,3], help="Optimization level for transpilation.")


    args = parser.parse_args()
    # 1. Configuration
    N = args.dns_resolution          # Résolution moyenne (DNS)
    VQA_N = args.grid_size      # Résolution Grossière
    T_MAX = args.t_max         # Temps final
    DT = args.dt           # Pas de temps
    HYBRID = int(args.hybrid_dt / DT)         # Fréquence de mise à jour hybride
    STEPS = int(T_MAX / DT)
    
    # 2. Initialisation
    print(f"Initialisation Orszag-Tang (Grille {N}x{N}) avec pas de temps {DT} et hybrid every {HYBRID} steps pour un temp total de {T_MAX} ({STEPS} steps)...")
    grid = PeriodicGrid(resolution_N=N)
    sim_quantum = MHDSolver(grid, dt=DT, Re=500, Rm=500) # A CHANGER: PARAMETRER R_max et R_e max
    sim_quantum.init_kelvin_helmholtz()
    sim_temoin = MHDSolver(grid, dt=DT, Re=500, Rm=500)  # Pour la visualisation finale
    sim_temoin.init_kelvin_helmholtz()
    mapper = AngleMapper(v0=1.0, B0=1.0)
    active_patches = []
    max_depth = int(log(N)/log(VQA_N))+1
    
    # Préparation de la première itération
    Phi_prev = None
    physics_state = sim_quantum.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)


    if args.verbose:
        plot_grid_topology(grid)
        plot_flux_on_edges(grid, Phi)

    # 3. Boucle Temporelle
    print(f"Lancement pour {STEPS} pas de temps...")
    for t in range(STEPS):
        if t % HYBRID == 0:
            active_patches = run_adaptive_vqa(
                sim_quantum, mapper, args, Phi_prev,
                threshold=0.65,
                target_dim = VQA_N,
                max_depth= max_depth,
                max_patches=N,
                min_size = 6,
                DT=DT,
                c_s = 1.0,
                eta = 0.01,
                Bz_guide = 1.0
            )
            if args.verbose:
                plot_amr_state(sim_quantum, active_patches, t, DT, t, VQA_N)
                plot_amr_state(sim_temoin, [], t, DT, t, VQA_N)

        sim_quantum.step_layered(active_patches, max_depth)
        Phi_prev = Phi
        sim_temoin.step_full()

        max_current = np.max(np.abs(physics_state['Jz']))
        print(f"Step {t}/{STEPS} (t={t*DT:.2f}) - Max Jz: {max_current:.4f}")
        

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

if __name__ == "__main__":
    main()