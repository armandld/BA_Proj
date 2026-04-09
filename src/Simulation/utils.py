import numpy as np
from scipy.ndimage import uniform_filter, zoom

def get_periodic_patch(arr, y_s, y_e, x_s, x_e, pad=0):
    """
    Extrait un patch avec padding en respectant la périodicité globale (Tore).
    Si on dépasse les bords, on va chercher les pixels de l'autre côté.
    """
    H, W = arr.shape
    
    # 1. Générer les plages d'indices théoriques (peuvent être négatifs ou > taille)
    # Exemple : si y_s=0 et pad=1, on veut l'indice -1
    y_range = np.arange(y_s - pad, y_e + pad)
    x_range = np.arange(x_s - pad, x_e + pad)
    
    # 2. Appliquer le Modulo pour 'wrapper' les indices
    # L'indice -1 devient H-1, l'indice H devient 0
    y_indices = y_range % H
    x_indices = x_range % W
    
    # 3. Extraction via np.ix_ (Meshgrid d'indices)
    # Cela crée une copie du sous-tableau avec les bonnes valeurs enveloppées
    return arr[np.ix_(y_indices, x_indices)]


def slice_hamiltonian_params(params, y_s, y_e, x_s, x_e, advanced_anomalies_enabled = False, pad= 0):
    """
    Découpe une sous-section des paramètres physiques pour un patch local.
    Gère intelligemment les tuples (C_edges, D_edges) et les matrices.
    """
    local_params = {}
    def extract(arr):
        return get_periodic_patch(arr, y_s, y_e, x_s, x_e, pad)

    # 1. Termes définis sur les Noeuds (Nodes)
    # Ils ont la même taille que la grille de pixels
    if advanced_anomalies_enabled and 'K_xpoint' in params:
        local_params['K_xpoint'] = extract(params['K_xpoint'])

    # 2. Termes définis sur les Arêtes (Edges) - Stockés sous forme de tuple (Horizontal, Vertical)
    # Note : Les matrices d'arêtes sont physiquement plus petites de 1 pixel dans une dimension,
    # mais le slicing numpy [start:end] gère ça sans erreur (il s'arrête juste à la fin).
    h_horiz, h_vert = params['H_edges']
    local_params['H_edges'] = (
        extract(h_horiz), 
        extract(h_vert)
    )

    # C_shear
    c_horiz, c_vert = params['C_edges']
    local_params['C_edges'] = (
        extract(c_horiz), 
        extract(c_vert)
    )

    # D_kink
    """
    if advanced_anomalies_enabled:
        d_horiz, d_vert = params['D_edges']
        local_params['D_edges'] = (
            extract(d_horiz), 
            extract(d_vert)
        )
    """
    # 3. Termes définis sur les Plaquettes
    local_params['K_plaquettes'] = extract(params['K_plaquettes'])

    return local_params


def compute_local_factor(patch_height, patch_width, depth, max_depth,
                         target_dim=2):
    """
    Effective downsampling factor for a patch at given depth.
    Shared between solver (physics) and pipeline (cost metric)
    to guarantee consistency.

    No artificial cap — factor is the full theoretical value
    target_dim^(max_depth - depth), constrained only by:
      1. Divisibility (patch dims must be divisible by factor)
      2. Padding efficiency (FD4+RK4 padding = 8*factor per side;
         if padding >= patch dim, factor is auto-reduced)

    Returns an integer >= 1. Higher depth → smaller factor → finer physics.
    """
    base_pad = 8   # FD4 stencil (±2) × RK4 (4 stages)

    theoretical_factor = int(target_dim ** max(0, max_depth - depth))
    local_factor = theoretical_factor

    # Constraint 1 — divisibility
    while local_factor > 1:
        if patch_height % local_factor == 0 and patch_width % local_factor == 0:
            break
        local_factor //= target_dim

    # Constraint 2 — padding efficiency
    # pad_fine = base_pad * local_factor  (fine-grid pixels per side)
    # Ensure interior > padding: pad_fine < min(patch_height, patch_width)
    min_dim = min(patch_height, patch_width)
    while local_factor > 1 and base_pad * local_factor >= min_dim:
        local_factor //= target_dim

    return max(1, local_factor)