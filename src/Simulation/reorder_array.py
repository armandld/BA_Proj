import numpy as np

def extract_periodic_patch(full_arr, y_s, y_e, x_s, x_e, cross_h, cross_v):
    """
    Extrait un patch en gérant les conditions aux limites périodiques.
    """
    # 1. Gestion Verticale (Lignes)
    if cross_v:
        # Cas "Cross V" : On colle la fin (bas) au-dessus du début (haut)
        part_bottom = full_arr[y_e:, :]  
        part_top    = full_arr[:y_s, :]  
        strip = np.vstack((part_bottom, part_top))
    else:
        strip = full_arr[y_s:y_e, :]

    # 2. Gestion Horizontale (Colonnes) sur la bande extraite
    if cross_h:
        # Cas "Cross H" : On colle la fin (droite) à gauche du début (gauche)
        part_right = strip[:, x_e:]  
        part_left  = strip[:, :x_s]  
        patch = np.hstack((part_right, part_left))
    else:
        patch = strip[:, x_s:x_e]
        
    return patch