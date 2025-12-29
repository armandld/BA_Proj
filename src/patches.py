import numpy as np
from scipy.ndimage import binary_dilation
from Simulation.reorder_array import extract_periodic_patch

def apply_patch_mask(mask, bounds,cross_v,cross_h):
    """
    Sets mask to True for the region defined by bounds (y_s, y_e, x_s, x_e).
    Handles periodic wrapping correctly.
    """
    ys, ye, xs, xe = bounds
    H, W = mask.shape
    
    # --- 1. Determine Vertical Slices ---
    # If ye < ys, it wraps: we need two slices (ys -> End) and (Start -> ye)
    if cross_v:
        y_slices = [slice(ye, H), slice(0, ys)]
    else:
        y_slices = [slice(ys, ye)]
        
    # --- 2. Determine Horizontal Slices ---
    if cross_h:
        x_slices = [slice(xe, W), slice(0, xs)]
    else:
        x_slices = [slice(xs, xe)]

    # --- 3. Apply Mask ---
    # We iterate over all combinations of slices (usually 1x1, sometimes 2x1, 1x2 or 2x2)
    for y_sl in y_slices:
        for x_sl in x_slices:
            mask[y_sl, x_sl] = True

    return mask

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
        mask = apply_patch_mask(mask, bounds = p['bounds'],cross_v = p['cross_v'], cross_h = p['cross_h'])
        
    # Sécurité Padding
    if padding > 0:
        mask = binary_dilation(mask, iterations=padding)
        
    return mask