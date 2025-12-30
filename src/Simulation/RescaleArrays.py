from scipy.ndimage import uniform_filter, zoom
import numpy as np
from Simulation.utils import zoom_hamilt, uniform_filter_hamilt

def get_adaptive_flux(local_h, local_v, hamilt_params, target_dim=3, mixing_ratio=0.3, type_filter = True):
    """
    Version robuste aux échelles variables.
    Adapte le prétraitement selon la taille de l'entrée pour éviter le sur-lissage.
    """
    h, w = local_h.shape
    min_side = min(h, w)
    
    # Copie pour travailler
    proc_h = local_h.astype(float)
    proc_v = local_v.astype(float)
    
    # --- 1. Lissage Spatial Adaptatif ---
    # On ne lisse que si l'image est nettement plus grande que la cible.
    # Règle empirique : il faut au moins 2 pixels pour en lisser 1 proprement.
    if min_side > target_dim:
        if type_filter:
            mode = 'wrap'
        else:
            mode = 'reflect'
        # Grand patch : on peut lisser pour réduire le bruit
        proc_h = uniform_filter(proc_h, size=3, mode= mode)
        proc_v = uniform_filter(proc_v, size=3, mode= mode)
        if hamilt_params is not None:
            hamilt_params = uniform_filter_hamilt(hamilt_params, size=3, mode= mode)
    else:
        # Tout petit patch (ex: 4x4 vers 3x3) : INTERDIT DE LISSER
        # On veut garder chaque pixel d'information brute.
        pass

    # --- 2. Cross-Talk (Mélange H <-> V) ---
    # Ça, c'est de la physique, c'est valide quelle que soit l'échelle.
    # On veut toujours savoir si un flux V est corrélé à un flux H local.
    w_self = 1.0 - mixing_ratio
    w_cross = mixing_ratio
    
    mixed_h = (w_self * proc_h) + (w_cross * proc_v)
    mixed_v = (w_self * proc_v) + (w_cross * proc_h)
    
    # --- 3. Réduction (Zoom) ---
    # Si la taille est déjà 3x3, zoom renvoie l'original (identity), donc c'est safe.
    zoom_y = target_dim / h
    zoom_x = target_dim / w
    
    # order=1 (bilinéaire) est bien. 
    # Pour les très petits patchs, order=0 (nearest) peut parfois être mieux pour garder les pics,
    # mais order=1 est un bon compromis général.
    mini_h = zoom(mixed_h, (zoom_y, zoom_x), order=1)
    mini_v = zoom(mixed_v, (zoom_y, zoom_x), order=1)
    if hamilt_params is not None:
        mini_hamilt_params = zoom_hamilt(hamilt_params, zoom_y= zoom_y, zoom_x= zoom_x, order=1)
        return mini_h, mini_v, mini_hamilt_params
    
    return mini_h, mini_v