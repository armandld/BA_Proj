import json
import subprocess
import os
import argparse
import sys


import numpy as np
import matplotlib.pyplot as plt

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.refinement import refinement

from visual import plot_amr_state,plot_recursive_state, simple_hierarchical_plot
from Simulation.refinement import recursive_vqa_check
from call_vqa_shell import call_vqa_shell
    

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
    N = args.grid_size              # Résolution Grossière
    D_MAX = args.dns_resolution/N   # Profondeur Max de Raffinement
    T_MAX = args.t_max         # Temps final
    DT = args.dt           # Pas de temps
    HYBRID = int(args.hybrid_dt / DT)         # Fréquence de mise à jour hybride
    STEPS = int(T_MAX / DT)

    Phi_prev = None
    
    # 2. Initialisation
    print(f"Initialisation Orszag-Tang (Grille {N}x{N})...")
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=DT, Re=500, Rm=500) # A CHANGER: PARAMETRER R_max et R_e max
    sim.init_orszag_tang()
    mapper = AngleMapper(v0=1.0, B0=1.0)
    
    plt.ion()

    for t in range(STEPS):
        
        # --- Step 1: Classical Coarse Update ---
        # U_pred <- ClassicalSolver(U_t)
        sim.time_step()
        U_pred = [sim.vx, sim.vy, sim.Bx, sim.By] # Snapshot
        
        # --- Step 2, 3, 4: Recursive VQA Refinement ---
        # On lance la récursion depuis la racine (Grille 0)
        
        # Info de base pour la racine
        root_info = {
            'abs_i': 0, 'abs_j': 0, 
            'scale_factor': 1, 'global_scale': 1.0
        }
        
        print(f"--- Step {t}: Quantum Risk Assessment (Recursion) ---")
        
        # Cette fonction va faire Step 2 -> Step 3 -> Recurse Step 4
        # Elle retourne la liste finale des solveurs (R_final)
        fine_solvers = recursive_vqa_check(
            U_pred, root_info, current_depth=0, max_depth=D_MAX, 
            mapper=mapper, args=args, Phi_prev=Phi_prev
        )
        
        # --- Step 5: Correction & Advance ---
        # Apply Fine Mesh AMR on final regions R_final
        if len(fine_solvers) > 0:
            print(f" -> Execution: {len(fine_solvers)} solvers actifs (dont coarse).")
            
            # On fait avancer chaque solveur fin
            # Note: Le solveur de profondeur 0 est déjà avancé en Step 1, 
            # ici on avance les raffinements pour préparer le feedback (Step 6 hypothétique)
            for f_sim in fine_solvers:
                if f_sim.meta['depth'] > 0:
                    # Les patchs profonds doivent faire plus de pas pour rattraper le temps
                    steps_needed = 2**f_sim.meta['depth'] 
                    for _ in range(steps_needed):
                        f_sim.time_step()
        
        # Coarsen Mesh back ? (Feedback)
        # Dans cette implémentation, on met simplement à jour Phi pour le prochain tour
        physics_state = sim.get_fluxes()
        Phi_prev = mapper.compute_stress_flux(physics_state)

        # Visualisation rapide
        simple_hierarchical_plot(sim, fine_solvers, t, args.dt)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()