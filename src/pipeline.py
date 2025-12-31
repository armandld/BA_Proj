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
from call_vqa_shell import call_vqa_shell
from patches import patches_to_mask

from help_visual import plot_grid_topology, plot_flux_on_edges

def main():
    sys.stdout.reconfigure(line_buffering=True) # Pour un affichage immédiat des print() à enlever pour une meilleure perf

    parser = argparse.ArgumentParser(description="Mapping VQA")
    parser.add_argument("--out-dir", default="../data", help="Output directory for mapping")
    parser.add_argument("--in-file", default="../input/mapping_input.json", help="Input directory for mapping")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--grid-size", type=int, default=2, help="Coarse grid dimension N (NxN)")
    parser.add_argument("--dns-resolution", type=int, default=256, help="High-Res Grid for Ground Truth")
    parser.add_argument("--t-max", type=float, default=1.0, help="Simulation end time")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("--hybrid-dt", type=float, default=0.1, help="Hybrid simulation time step size")
    parser.add_argument("--depth", type=int, required=False, help="Depth of the ULA ansatz.")
    parser.add_argument("--mode", default="simulator", choices=["simulator", "hardware"])
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"])
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--method", default="COBYLA", choices=["COBYLA", "L-BFGS-B", "Powell"])
    parser.add_argument("--opt_level", type=int, default=1, choices=[0,1,2,3], help="Optimization level for transpilation.")


    args = parser.parse_args()
    # 1. Configuration
    N = args.dns_resolution          # Résolution moyenne (DNS)
    VQA_N = args.grid_size      # Résolution Grossière
    T_MAX = args.t_max         # Temps final
    DT = args.dt           # Pas de temps
    HYBRID = int(args.hybrid_dt / DT)         # Fréquence de mise à jour hybride
    STEPS = int(T_MAX / DT)
    
    # 2. Initialisation
    print(f"Initialisation Orszag-Tang (Grille {N}x{N})...")
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=DT, Re=500, Rm=500) # A CHANGER: PARAMETRER R_max et R_e max
    sim.init_orszag_tang()
    mapper = AngleMapper(v0=1.0, B0=1.0)
    active_patches = []
    
    # 3. Boucle Temporelle
    print(f"Lancement pour {STEPS} pas de temps...")
    Phi_prev = None
    mask_calcul = None
    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)
    #plot_grid_topology(grid)
    #plot_flux_on_edges(grid, Phi)
    for t in range(STEPS):

        sim.step_masked(mask_calcul)
        # MaJ Quantum périodique
        if t % HYBRID == 0:
            active_patches = run_adaptive_vqa(
                sim, mapper, args, Phi_prev,
                threshold=0.7,
                target_dim = VQA_N,
                max_depth= int(log(N)/log(VQA_N))+1,
                max_patches=N,
                min_size = 6,
                DT=DT,
                c_s = 1.0,
                eta = 0.01,
                Bz_guide = 1.0
            )
            mask_calcul = patches_to_mask((N,N), active_patches)
            plot_amr_state(sim, active_patches, t, DT, t)
            Phi_prev = Phi

        max_current = np.max(np.abs(physics_state['Jz']))
        print(f"Step {t}/{STEPS} (t={t*DT:.2f}) - Max Jz: {max_current:.4f}")
        

    # 4. Visualisation Finale
    print("Simulation terminée. Génération du plot...")
    final_state = sim.get_fluxes()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(final_state['Jz'].T, origin='lower', cmap='RdBu', extent=[0, 2*np.pi, 0, 2*np.pi])
    plt.colorbar(label='Densité de Courant Jz')
    plt.title(f"Orszag-Tang Vortex (t={T_MAX}) - Détection de Reconnexion")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    main()