import numpy as np
from Simulation.solver import MHDSolver

def refinement(patches_to_create, coarse_sim, grid, DT):
    """
    Gère la création et la mise à jour des patchs raffinés.
    Args:
        patches_to_create: Liste des patchs à créer (dictionnaires avec métadonnées)
        coarse_sim: Instance du solveur grossier (MHDSolver)
        grid: Grille grossière (PeriodicGrid)
        DT: Pas de temps global
    Returns:
        active_patches: Liste des instances de solveurs fins actifs
    """
    active_patches = []
    
    for p in patches_to_create:
        factor = p['factor'] # 2 ou 4
        
        # 1. Création Grille (x2 ou x4)
        data_list = [coarse_sim.vx, coarse_sim.vy, coarse_sim.Bx, coarse_sim.By]
        fine_grid, fine_data = grid.create_refined_grid(
            data_list, 
            p['i_start'], p['j_start'], p['width'], 
            factor=factor # Dynamique !
        )
        
        # 2. Création Solveur
        # Le DT doit suivre le DX pour la condition CFL (Courant-Friedrichs-Lewy)
        # Si dx est divisé par factor, dt doit l'être aussi (environ)
        fine_sim = MHDSolver(fine_grid, dt=DT/factor, Re=500, Rm=500)
        
        # 3. Injection Données
        fine_sim.vx = fine_data[0]
        fine_sim.vy = fine_data[1]
        fine_sim.Bx = fine_data[2]
        fine_sim.By = fine_data[3]
        
        # Métadonnées pour l'affichage et la boucle
        fine_sim.meta = p 
        active_patches.append(fine_sim)
        
    return active_patches

import json
import subprocess
import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper

# ==============================================================================
# 1. FONCTIONS UTILITAIRES (SHELL & VQA)
# ==============================================================================

def call_vqa_shell(angles_tuple, args, script_path="run_VQA_pipeline.sh"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_path)
    data_dir = os.path.abspath(os.path.join(current_dir, "../data"))
    os.makedirs(data_dir, exist_ok=True)

    input_file = os.path.join(data_dir, "vqa_input.json")
    output_file = os.path.join(data_dir, "vqa_output.json")
    
    data = {
        "theta_h": angles_tuple[0].tolist(), "theta_v": angles_tuple[1].tolist(),
        "psi_h": angles_tuple[2].tolist(),   "psi_v": angles_tuple[3].tolist()
    }
    
    with open(input_file, "w") as f:
        json.dump(data, f)
    
    cmd = [
        "bash", script_path, "--in-file", input_file, "--out-dir", data_dir,
        "--out-file", output_file, "--backend", args.backend,
        "--method", args.method, "--mode", args.mode,
        "--opt_level", str(args.opt_level), "--shots", str(args.shots),
        "--depth", str(args.depth), "--numqbits", str(args.grid_size * args.grid_size * 2),
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # Silence total
    except subprocess.CalledProcessError:
        return None

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            return np.array(json.load(f))
    return None

# ==============================================================================
# 2. COEUR DE L'ALGORITHME : RAFFINEMENT RÉCURSIF (STEP 2, 3, 4)
# ==============================================================================

def recursive_vqa_check(sim_data, grid_info, current_depth, max_depth, mapper, args, Phi_prev=None):
    """
    Implémente la logique récursive : 
    VQA -> Identification Risque -> Si Risque et Depth < Max -> Recurse
    """
    active_solvers = []
    
    # --- Step 2: Quantum Risk Assessment ---
    # On simule une extraction de flux sur la grille locale courante
    # sim_data est une liste [vx, vy, Bx, By]
    # On reconstruit un état physique temporaire pour le mapper
    # Note: Simplification ici, on passe directement les tableaux au mapper adapté
    # Pour ce code, on suppose que sim_data est déjà formaté ou on l'utilise tel quel.
    
    # 1. Calculer Phi (Flux) pour cette échelle
    # (Ici on fait une approximation : on mappe les données locales vers des angles)
    # Dans une vraie implém, il faudrait un objet 'PhysicsState' local. 
    # Hack pour la démo: On utilise la moyenne ou un slice du sim global si depth=0
    
    # Appel VQA (On suppose que le mapping est fait en amont ou ici)
    # Pour la récursion, on a besoin de mapper les données locales (sim_data) vers des angles.
    # Ici, pour simplifier l'exemple, on va dire que si c'est profond, on utilise les données interpolées.
    
    # Calcul simplifié des flux locaux (Jz)
    vx, vy, Bx, By = sim_data
    Phi = mapper.compute_stress_flux(sim_data)
    angles = mapper.map_to_angles(Phi, Phi_prev, alpha=np.pi, beta=1.0, dt=args.dt)
    
    probs = call_vqa_shell(angles, args)
    
    if probs is None: return [] # Echec VQA

    # --- Step 3: Identify High-Risk Zones ---
    # On utilise la grille courante pour décoder où ça chauffe
    # i_start, j_start sont relatifs à la grille PARENTE, ici on gère des coordonnées absolues
    patches_candidates = PeriodicGrid(resolution_N=len(vx)).decode_refinement_patches(
        probs, low_thresh=0.6, high_thresh=0.90, padding=0
    )
    
    # --- Step 4: Recursive Refinement (Zoom) ---
    is_leaf = True
    
    if len(patches_candidates) > 0 and current_depth < max_depth:
        is_leaf = False
        # On a détecté un risque ET on peut encore zoomer
        print(f"   [Depth {current_depth}] ⚠️ Risque détecté dans {len(patches_candidates)} zones. Zooming...")
        
        for p in patches_candidates:
            # 1. Generate Sub-grid (Interpolation)
            # On découpe les données locales pour créer la donnée de l'enfant
            # create_refined_grid attend des indices globaux, ici on bricole pour la recursion locale
            # Pour simplifier: On extrait la slice et on interpole x2 (factor 2 par défaut pour quadtree)
            
            # Extraction basique (slice)
            s_i, s_j, w = p['i_start'], p['j_start'], p['width']
            sub_data = [d[s_i:s_i+w, s_j:s_j+w] for d in sim_data]
            
            # Interpolation (Raffinement x2 pour l'étape suivante)
            # On utilise la méthode de la classe Grid mais "manuellement" car on est en local
            refined_data = []
            for d in sub_data:
                # Interpolation x2 simple (Kron product)
                refined_d = d.repeat(2, axis=0).repeat(2, axis=1)
                refined_data.append(refined_d)
            
            # Calcul des nouvelles coordonnées absolues pour le traçage
            abs_i = grid_info['abs_i'] + s_i * grid_info['scale_factor']
            abs_j = grid_info['abs_j'] + s_j * grid_info['scale_factor']
            new_scale = grid_info['scale_factor'] # On garde l'échelle relative au grossier
            
            new_info = {
                'abs_i': abs_i, 'abs_j': abs_j, 
                'width': w, 'factor': 2, # Relatif au parent
                'global_scale': grid_info['global_scale'] / 2 # L'échelle diminue
            }

            # --- RECURSE ---
            # On appelle la fonction sur l'enfant
            child_solvers = recursive_vqa_check(
                refined_data, new_info, current_depth + 1, max_depth, mapper, args, Phi
            )
            active_solvers.extend(child_solvers)
            
    # Si c'est une feuille (soit pas de risque, soit profondeur max atteinte)
    # OU si on a décidé de s'arrêter là.
    if is_leaf:
        # On crée le solveur pour cette région
        # C'est ici qu'on définit la "Fine Mesh AMR" finale
        # On calcule le facteur de zoom total par rapport à la grille 0
        total_zoom = 2**current_depth
        
        # On crée une grille périodique locale
        local_N = len(vx)
        local_grid = PeriodicGrid(resolution_N=local_N)
        
        # Le DT doit être réduit : DT_base / total_zoom
        fine_sim = MHDSolver(local_grid, dt=args.dt/total_zoom, Re=500, Rm=500)
        fine_sim.vx, fine_sim.vy, fine_sim.Bx, fine_sim.By = sim_data
        
        # Metadata pour le plot
        fine_sim.meta = {
            'depth': current_depth,
            'abs_i': grid_info['abs_i'],
            'abs_j': grid_info['abs_j'],
            'width_in_coarse_units': len(vx) / (2**current_depth), # Approx
            'zoom': total_zoom
        }
        active_solvers.append(fine_sim)

    return active_solvers