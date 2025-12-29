import numpy as np
from scipy.ndimage import zoom, label, find_objects, binary_dilation

from Simulation.solver import MHDSolver

from call_vqa_shell import call_vqa_shell

from help_visual import visualize_vqa_step

from Simulation.MixingVHarrays import get_adaptive_flux
from Simulation.reorder_array import extract_periodic_patch

def recursive_vqa_scan(
    # Données physiques complètes (ne changent pas, on passe des références)
    full_phi_h, full_phi_v, 
    full_prev_h, full_prev_v, 
    
    # Paramètres de récursion
    bounds,        # Tuple (y_start, y_end, x_start, x_end) de la zone actuelle
    depth,         # Profondeur actuelle de récursion
    
    # Objets et Configs
    mapper, args, DT,
    
    # Accumulateur de résultats
    active_patches, # Liste où on stocke les zones finales identifiées
    
    # Hyper-paramètres
    target_dim=4,    # Dimension de la grille d'entrée du VQA (ex: 3x3)
    max_depth=3,      # Combien de fois on peut zoomer (ex: 256 -> 85 -> 28 -> 9)
    min_size=4,       # Taille minimale d'un patch en pixels physiques
    threshold=0.6,    # Seuil pour dire "C'est turbulent, faut creuser"
    max_patches=256,   # Budget computationnel
    cross_h = False,
    cross_v = False,
):
    """
    Fonction récursive qui explore le domaine physique guidée par le VQA.
    """
    # Si on a déjà explosé le budget, on arrête tout
    if len(active_patches) >= max_patches:
        return

    y_s, y_e, x_s, x_e = bounds

    if cross_h :
        height = len(full_phi_v) - y_e - y_s
    else :
        height = y_e - y_s
    
    if cross_v :
        width = len(full_phi_h) - x_e - x_s
    else :
        width = x_e - x_s

    # Sécurité : Si la zone est trop petite ou vide, on arrête
    if height < min_size or width < min_size:
        # On considère cette toute petite zone comme un "patch final"
        active_patches.append({
            'bounds': bounds,'cross_h': cross_h,'cross_v': cross_v, 'depth': depth, 'type': 'leaf_limit'
        })
        return

    # --- 1. Extraction et Préparation Locale --- 
    # On découpe les données physiques correspondant à la zone actuelle
    local_h = extract_periodic_patch(full_phi_h, y_s, y_e, x_s, x_e, cross_h, cross_v)
    local_v = extract_periodic_patch(full_phi_v, y_s, y_e, x_s, x_e, cross_h, cross_v)
    

    mini_h, mini_v = get_adaptive_flux(local_h, local_v, target_dim=target_dim, mixing_ratio=0.5)
    # Reconstruction du dictionnaire pour le mapper
    mini_Phi_dict = {'phi_horizontal': mini_h, 'phi_vertical': mini_v}

    # Gestion du phi_prev (si disponible)
    if full_prev_h is not None:
        local_prev_h = extract_periodic_patch(full_prev_h, y_s, y_e, x_s, x_e, cross_h, cross_v)
        local_prev_v = extract_periodic_patch(full_prev_v, y_s, y_e, x_s, x_e, cross_h, cross_v)

        mini_prev_h, mini_prev_v = get_adaptive_flux(
            local_prev_h,
            local_prev_v,
            target_dim=target_dim,
            mixing_ratio=0.5
        )
        mini_Phi_prev_dict = {'phi_horizontal': mini_prev_h, 'phi_vertical': mini_prev_v}
    else:
        # Cold start: on passe l'état actuel comme "prev" (dérivée nulle) ou des zéros
        mini_Phi_prev_dict = mini_Phi_dict 
    # --- 2. L'Oracle VQA (Appel Quantique) ---
    # Le VQA analyse la zone et retourne 18 angles -> Probabilités sur la grille 3x3
    angles = mapper.map_to_angles(mini_Phi_dict, mini_Phi_prev_dict, alpha=np.pi, beta=1.0, dt=DT)
    # Appel Shell (Simulé ou Réel) - Force la grille 3x3
    probs = call_vqa_shell(angles, args, script_path="run_VQA_pipeline.sh", period_bound = depth==0)

    # Visualiser les steps
    visualize_vqa_step(local_h, local_v, bounds, depth)

    if probs is None: return # Erreur technique


    # Décodage : 3x3 ProbMap
    num_edges = target_dim * target_dim
    probs_h = probs[:num_edges].reshape(target_dim, target_dim)
    probs_v = probs[num_edges:].reshape(target_dim, target_dim)
    prob_map = np.maximum(probs_h, probs_v) # Carte de chaleur 3x3 de la zone actuelle

    # --- 3. Décision Récursive (AMR Logic) ---
    
    # Cas de base : On a atteint la profondeur max
    if depth >= max_depth:
        # On ajoute ce patch entier à la liste finale
        active_patches.append({
            'bounds': bounds,
            'cross_h': cross_h,
            'cross_v': cross_v,
            'depth': depth, 
            'score': np.max(prob_map),
            'type': 'leaf_depth'
        })
        return

    # Cas Récursif : On analyse les 9 sous-blocs suggérés par le VQA
    # Taille des sous-blocs
    step_y = height // 3
    step_x = width // 3
        
    for i in range(3):
        for j in range(3):
            # Probabilité locale vue par le VQA pour ce sous-secteur
            local_prob_h = probs_h[i, j]
            local_prob_v = probs_v[i, j]

            #Possibly improve the expression of the threshold depending on depth
            new_threshold = threshold #+ (1 - threshold)/3

            new_cross_h = False
            new_cross_v = False

            lim_h = False
            lim_v = False

            y_s_h = y_s - height//2
            if y_s_h < 0:
                y_s_h += len(full_phi_v)
            y_e_h = y_e - height//2

            # Coordonnées physiques du sous-secteur horizontal
            sub_y_s_h = y_s_h + i * step_y
            sub_y_e_h = y_s_h + (i + 1) * step_y if i < 2 else y_e_h # Le dernier prend le reste
            sub_y_s_h = sub_y_s_h % len(full_phi_v)
            sub_y_e_h = sub_y_e_h % len(full_phi_v)
            if(sub_y_e_h<sub_y_s_h): cross_v = True

            sub_x_s_h = x_s + j * step_x
            sub_x_e_h = x_s + (j + 1) * step_x if j < 2 else x_e
            
            sub_bounds_h = (sub_y_s_h, sub_y_e_h, sub_x_s_h, sub_x_e_h)
        

            x_s_v = y_s - height//2
            if x_s_v < 0:
                x_s_v += len(full_phi_h)
            x_e_v = y_e - height//2

            # Coordonnées physiques du sous-secteur vertical
            sub_x_s_v = x_s_v + j * step_x
            sub_x_e_v = x_s_v + (j + 1) * step_x if j < 2 else x_e_v
            sub_x_s_v = sub_x_s_v % len(full_phi_h)
            sub_x_e_v = sub_x_e_v % len(full_phi_h)
            if(sub_x_e_v<sub_x_s_v): cross_h = True
            sub_y_s_v = y_s + i * step_y
            sub_y_e_v = y_s + (i + 1) * step_y if i < 2 else y_e # Le dernier prend le reste

            sub_bounds_v = (sub_y_s_v, sub_y_e_v, sub_x_s_v, sub_x_e_v)

            if local_prob_h > threshold:
                # TURBULENCE DÉTECTÉE : On plonge plus profond 
                
                recursive_vqa_scan(
                    full_phi_h, full_phi_v, full_prev_h, full_prev_v,
                    sub_bounds_h, depth + 1,
                    mapper, args, DT, active_patches,
                    target_dim = target_dim, max_depth = max_depth, min_size = min_size,
                    threshold = new_threshold, max_patches = max_patches
                )
            else:
                active_patches.append({
                    'bounds': sub_bounds_h,
                    'cross_h': cross_h,
                    'cross_v': cross_v,
                    'depth': depth,
                    'score': local_prob_h,
                    'type': 'coarse_leaf' # C'est un gros patch calme
                })

            if local_prob_v > threshold:
                # TURBULENCE DÉTECTÉE : On plonge plus profond 
                
                recursive_vqa_scan(
                    full_phi_h, full_phi_v, full_prev_h, full_prev_v,
                    sub_bounds_v, depth + 1,
                    mapper, args, DT, active_patches,
                    target_dim = target_dim, max_depth = max_depth, min_size = min_size,
                    threshold = new_threshold, max_patches = max_patches,
                    cross_v = new_cross_v, cross_h = new_cross_h
                )
            else:
                active_patches.append({
                    'bounds': sub_bounds_v,
                    'cross_h': cross_h,
                    'cross_v': cross_v,
                    'depth': depth,
                    'score': local_prob_v,
                    'type': 'coarse_leaf' # C'est un gros patch calme
                })


def run_adaptive_vqa(sim, mapper, args, Phi_prev, threshold=0.65, target_dim=3, max_depth=4, max_patches=256, min_size=6, DT=0.1):
    """
    Point d'entrée principal à appeler dans ton main().
    """
    # 1. Préparation des données complètes (Read-Only pour la récursion)
    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)
    
    full_h = Phi['phi_horizontal']
    full_v = Phi['phi_vertical']
    
    if Phi_prev:
        full_prev_h = Phi_prev['phi_horizontal']
        full_prev_v = Phi_prev['phi_vertical']
    else:
        full_prev_h = None
        full_prev_v = None
        
    H, W = full_h.shape
    initial_bounds = (0, H, 0, W)
    
    # Liste qui recevra les résultats
    final_patches = []
    
    print(f"--- START RECURSIVE VQA SCAN (Budget: {max_patches} patches) ---")
    
    # 2. Lancement de la récursion
    recursive_vqa_scan(
        full_h, full_v, full_prev_h, full_prev_v,
        initial_bounds, depth=0,
        mapper=mapper, args=args, DT=DT,
        active_patches=final_patches,
        target_dim=target_dim,          # Grille d'entrée du VQA
        max_depth=max_depth,         # Profondeur max (256 -> ~3px)
        min_size=min_size,          # Taille min patch
        threshold=threshold,      # Sensibilité
        max_patches=max_patches,
        cross_h = False,
        cross_v = False
    )
    if len(final_patches) == 0:
        print(">>> VQA found nothing active. Defaulting to FULL COMPUTATION.")
        H, W = full_h.shape
        # On ajoute un gros patch qui couvre tout
        final_patches.append({
            'bounds': (0, H, 0, W), 'cross_h': False, 'cross_v': False, 'depth': 0, 'type': 'fallback'})
    
    print(f"--- SCAN COMPLETE: {len(final_patches)} Active Zones Identified ---")
    print(final_patches)
    return final_patches
