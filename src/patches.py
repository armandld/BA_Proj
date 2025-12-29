import numpy as np
from scipy.ndimage import binary_dilation

def patches_to_mask(sim_shape, patches, padding=4):
    """
    Convertit SEULEMENT les patchs 'fins' (actifs) en masque de calcul.
    Les patchs 'coarse' (calmes) sont ignorés ici, donc ils seront False dans le masque.
    """
    H, W = sim_shape
    mask = np.zeros((H, W), dtype=bool)
    
    active_types = ['fine_leaf', 'leaf_limit', 'leaf_depth']
    
    for p in patches:
        # --- FILTRE INTELLIGENT ---
        # Si le patch est marqué "coarse" (calme), on ne l'active pas dans le masque.
        # Il sera traité par la diffusion globale (step_masked: else)
        if p.get('type') not in active_types:
            continue

        # Récupération bounds
        ys, ye, xs, xe = p['bounds']
        mask[ys:ye,xs:xe]=True

    # Sécurité Padding
    if padding > 0:
        mask = binary_dilation(mask, iterations=padding)
        
    return mask