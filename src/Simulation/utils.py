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
    if advanced_anomalies_enabled:
        local_params['Delta_nodes'] = extract(params['Delta_nodes'])

    # 2. Termes définis sur les Arêtes (Edges) - Stockés sous forme de tuple (Horizontal, Vertical)
    # Note : Les matrices d'arêtes sont physiquement plus petites de 1 pixel dans une dimension,
    # mais le slicing numpy [start:end] gère ça sans erreur (il s'arrête juste à la fin).
    
    # C_shear
    c_horiz, c_vert = params['C_edges']
    local_params['C_edges'] = (
        extract(c_horiz), 
        extract(c_vert)
    )

    # D_kink
    if advanced_anomalies_enabled:
        d_horiz, d_vert = params['D_edges']
        local_params['D_edges'] = (
            extract(d_horiz), 
            extract(d_vert)
        )

    # 3. Termes définis sur les Plaquettes
    local_params['K_plaquettes'] = extract(params['K_plaquettes'])

    return local_params