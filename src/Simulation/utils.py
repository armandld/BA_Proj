from scipy.ndimage import uniform_filter, zoom

def slice_hamiltonian_params(params, y_s, y_e, x_s, x_e):
    """
    Découpe une sous-section des paramètres physiques pour un patch local.
    Gère intelligemment les tuples (C_edges, D_edges) et les matrices.
    """
    local_params = {}

    # 1. Termes définis sur les Noeuds (Nodes)
    # Ils ont la même taille que la grille de pixels
    local_params['Delta_nodes'] = params['Delta_nodes'][y_s:y_e, x_s:x_e]
    local_params['M_nodes']     = params['M_nodes'][y_s:y_e, x_s:x_e]

    # 2. Termes définis sur les Arêtes (Edges) - Stockés sous forme de tuple (Horizontal, Vertical)
    # Note : Les matrices d'arêtes sont physiquement plus petites de 1 pixel dans une dimension,
    # mais le slicing numpy [start:end] gère ça sans erreur (il s'arrête juste à la fin).
    
    # C_shear
    c_horiz, c_vert = params['C_edges']
    local_params['C_edges'] = (
        c_horiz[y_s:y_e, x_s:x_e], 
        c_vert[y_s:y_e, x_s:x_e]
    )

    # D_kink
    d_horiz, d_vert = params['D_edges']
    local_params['D_edges'] = (
        d_horiz[y_s:y_e, x_s:x_e], 
        d_vert[y_s:y_e, x_s:x_e]
    )

    # 3. Termes définis sur les Plaquettes
    local_params['K_plaquettes'] = params['K_plaquettes'][y_s:y_e, x_s:x_e]

    return local_params

def uniform_filter_hamilt(params, size, mode):
    local_params = {}

    # 1. Termes définis sur les Noeuds (Nodes)
    # Ils ont la même taille que la grille de pixels
    local_params['Delta_nodes'] = uniform_filter(params['Delta_nodes'], size= size, mode= mode)
    local_params['M_nodes']     = uniform_filter(params['M_nodes'], size= size, mode= mode)

    # 2. Termes définis sur les Arêtes (Edges) - Stockés sous forme de tuple (Horizontal, Vertical)
    # Note : Les matrices d'arêtes sont physiquement plus petites de 1 pixel dans une dimension,
    # mais le slicing numpy [start:end] gère ça sans erreur (il s'arrête juste à la fin).
    
    # C_shear
    c_horiz, c_vert = params['C_edges']
    local_params['C_edges'] = (
        uniform_filter(c_horiz, size= size, mode= mode), 
        uniform_filter(c_vert, size= size, mode= mode)
    )

    # D_kink
    d_horiz, d_vert = params['D_edges']
    local_params['D_edges'] = (
        uniform_filter(d_horiz, size= size, mode= mode),
        uniform_filter(d_vert, size= size, mode= mode)
    )

    # 3. Termes définis sur les Plaquettes
    local_params['K_plaquettes'] = uniform_filter(params['K_plaquettes'], size= size, mode= mode)

    return local_params

def zoom_hamilt(params, zoom_y, zoom_x, order=1):
    local_params = {}

    # 1. Termes définis sur les Noeuds (Nodes)
    # Ils ont la même taille que la grille de pixels
    local_params['Delta_nodes'] = zoom(params['Delta_nodes'], (zoom_y, zoom_x), order= order)
    local_params['M_nodes']     = zoom(params['M_nodes'], (zoom_y, zoom_x), order= order)

    # 2. Termes définis sur les Arêtes (Edges) - Stockés sous forme de tuple (Horizontal, Vertical)
    # Note : Les matrices d'arêtes sont physiquement plus petites de 1 pixel dans une dimension,
    # mais le slicing numpy [start:end] gère ça sans erreur (il s'arrête juste à la fin).
    
    # C_shear
    c_horiz, c_vert = params['C_edges']
    local_params['C_edges'] = (
        zoom(c_horiz, (zoom_y, zoom_x), order= order),
        zoom(c_vert, (zoom_y, zoom_x), order= order)
    )

    # D_kink
    d_horiz, d_vert = params['D_edges']
    local_params['D_edges'] = (
        zoom(d_horiz, (zoom_y, zoom_x), order= order),
        zoom(d_vert, (zoom_y, zoom_x), order= order)
    )

    # 3. Termes définis sur les Plaquettes
    local_params['K_plaquettes'] = zoom(params['K_plaquettes'], (zoom_y, zoom_x), order= order)

    return local_params