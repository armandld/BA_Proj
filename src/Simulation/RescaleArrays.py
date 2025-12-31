import numpy as np
from scipy.ndimage import uniform_filter, zoom

def get_adaptive_flux(local_h, local_v, hamilt_params, target_dim=3, mixing_ratio=0.3, type_filter=True):
    """
    Adapte les flux et les paramètres à la dimension cible du VQA.
    
    - Si type_filter=True (Scan global) : Lissage global + Zoom global.
    - Si type_filter=False (Zoom local) : Traitement séparé Cœur/Halo pour préserver les CL.
    """
    
    # --- 1. Préparation des données (Casting) ---
    proc_h = local_h.astype(float)
    proc_v = local_v.astype(float)
    
    # --- 2. Mixing (Cross-Talk Physique) ---
    # On applique le mélange AVANT tout redimensionnement pour capturer la physique locale
    w_self = 1.0 - mixing_ratio
    w_cross = mixing_ratio
    
    mixed_h = (w_self * proc_h) + (w_cross * proc_v)
    mixed_v = (w_self * proc_v) + (w_cross * proc_h)

    # --- 3. Définition de la logique de transformation ---

    def _resize_padded_structure(arr, t_dim):
        """
        Transforme une matrice (N+2, M+2) en (t_dim+2, t_dim+2).
        - Le Cœur est zoomé en (t_dim, t_dim).
        - Les Halos (Lignes/Colonnes) sont zoomés en longueur t_dim.
        - Les Coins sont copiés tels quels.
        """
        h_in, w_in = arr.shape
        
        # A. Extraction des composants
        # On suppose que l'entrée a un padding de 1 (donc h_in = h_core + 2)
        core = arr[1:-1, 1:-1]
        
        # Halos (Bordures sans les coins)
        top_halo    = arr[0, 1:-1]
        bottom_halo = arr[-1, 1:-1]
        left_halo   = arr[1:-1, 0]
        right_halo  = arr[1:-1, -1]
        
        # Coins (Scalaires)
        corners = {
            'tl': arr[0, 0],   'tr': arr[0, -1],
            'bl': arr[-1, 0],  'br': arr[-1, -1]
        }

        # B. Calcul des facteurs de zoom
        h_core, w_core = core.shape
        # Facteurs pour le coeur
        z_y = t_dim / h_core
        z_x = t_dim / w_core
        
        # C. Zoom des composants
        # 1. Cœur -> (t_dim, t_dim)
        new_core = zoom(core, (z_y, z_x), order=1)
        
        # 2. Halos -> Vecteurs de taille t_dim
        # Note : zoom sur 1D demande un tuple de facteur (z,)
        new_top    = zoom(top_halo,    (t_dim / len(top_halo),),    order=1)
        new_bottom = zoom(bottom_halo, (t_dim / len(bottom_halo),), order=1)
        new_left   = zoom(left_halo,   (t_dim / len(left_halo),),   order=1)
        new_right  = zoom(right_halo,  (t_dim / len(right_halo),),  order=1)

        # D. Reconstruction de la matrice (t_dim+2, t_dim+2)
        final_arr = np.zeros((t_dim + 2, t_dim + 2))
        
        # Placement Cœur
        final_arr[1:-1, 1:-1] = new_core
        
        # Placement Halos
        final_arr[0, 1:-1]  = new_top
        final_arr[-1, 1:-1] = new_bottom
        final_arr[1:-1, 0]  = new_left
        final_arr[1:-1, -1] = new_right
        
        # Placement Coins (Inchangés)
        final_arr[0, 0]   = corners['tl']
        final_arr[0, -1]  = corners['tr']
        final_arr[-1, 0]  = corners['bl']
        final_arr[-1, -1] = corners['br']
        
        return final_arr

    def _process_dispatch(arr, is_periodic_scan):
        """Dispatche vers la méthode globale ou locale."""
        if arr is None: return None
        
        if is_periodic_scan:
            # --- CAS 1 : Type Filter = True (Périodique / Global) ---
            # Comportement original : Lissage global + Zoom global
            h, w = arr.shape
            min_side = min(h, w)
            
            # Lissage conditionnel
            processed = arr
            if min_side > target_dim:
                processed = uniform_filter(arr, size=3, mode='wrap')
            
            # Zoom global standard
            z_y = target_dim / h
            z_x = target_dim / w
            return zoom(processed, (z_y, z_x), order=1)
            
        else:
            # --- CAS 2 : Type Filter = False (Sous-domaine avec Halo) ---
            # Pas de lissage global destructeur.
            # Redimensionnement structurel (Cœur vs Halo).
            return _resize_padded_structure(arr, target_dim)

    # --- 4. Exécution sur les données ---
    
    # Traitement des Flux
    mini_h = _process_dispatch(mixed_h, type_filter)
    mini_v = _process_dispatch(mixed_v, type_filter)
    
    # Traitement des Paramètres Hamiltoniens (Récursif pour les dictionnaires/tuples)
    mini_hamilt_params = {}
    if hamilt_params is not None:
        for key, value in hamilt_params.items():
            if isinstance(value, (tuple, list)):
                # Gère C_edges et D_edges (Tuple de matrices)
                mini_hamilt_params[key] = tuple(
                    _process_dispatch(v, type_filter) for v in value
                )
            elif isinstance(value, np.ndarray):
                # Matrices simples (Delta, M, K)
                mini_hamilt_params[key] = _process_dispatch(value, type_filter)
            else:
                mini_hamilt_params[key] = value

    return mini_h, mini_v, mini_hamilt_params