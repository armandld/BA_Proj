import numpy as np
from scipy.ndimage import zoom


# ═══════════════════════════════════════════════════════════════════════
#  MAX-ABS POOLING — preserves localized anomalies during downsampling
# ═══════════════════════════════════════════════════════════════════════

def _maxabs_pool_2d(arr, target_h, target_w):
    """Pool by keeping the value with maximum absolute value in each block.

    For anomaly detection: a single strong signal (shock, current sheet)
    in a large block survives the downsampling.  Bilinear interpolation
    would average it out with ~1000 background cells.
    """
    h, w = arr.shape
    bh = h // target_h
    bw = w // target_w
    if bh < 1 or bw < 1:
        return zoom(arr, (target_h / h, target_w / w), order=1)

    # Les blocs sont delimites par des bornes reparties sur TOUTE l'etendue.
    #
    # La version precedente decoupait `arr[:target_h*bh, :target_w*bw]` :
    # le reste de la division etait purement jete. Pour 10x10 -> 3x3, les
    # lignes et colonnes 9 disparaissaient, soit 19 cellules sur 100, et un
    # pic isole qui s'y trouvait s'evanouissait sans trace — exactement
    # l'anomalie que ce pooling existe pour preserver.
    #
    # Quand h % target_h == 0 (le cas du chemin deploye, 256 vers 2/4/8),
    # les bornes retombent sur les memes blocs qu'avant : la sortie est
    # alors bit-a-bit identique.
    ii = np.linspace(0, h, target_h + 1).astype(int)
    jj = np.linspace(0, w, target_w + 1).astype(int)
    out = np.empty((target_h, target_w), dtype=float)
    for a in range(target_h):
        for b in range(target_w):
            block = arr[ii[a]:ii[a + 1], jj[b]:jj[b + 1]]
            flat = block.reshape(-1)
            out[a, b] = flat[np.argmax(np.abs(flat))]
    return out


def _maxabs_pool_1d(arr, target_len):
    """1D max-abs pooling for halo (boundary) arrays."""
    n = len(arr)
    bs = n // target_len
    if bs < 1:
        return zoom(arr, (target_len / n,), order=1)
    # Meme correction qu'en 2D : les bornes couvrent toute l'etendue au lieu
    # de tronquer le reste de la division, qui faisait disparaitre les
    # dernieres cellules. Sortie inchangee quand n % target_len == 0.
    kk = np.linspace(0, n, target_len + 1).astype(int)
    out = np.empty(target_len, dtype=float)
    for a in range(target_len):
        seg = np.asarray(arr[kk[a]:kk[a + 1]]).reshape(-1)
        out[a] = seg[np.argmax(np.abs(seg))]
    return out


# ═══════════════════════════════════════════════════════════════════════
#  PADDED STRUCTURE HANDLERS (depth > 0 : core + halo)
# ═══════════════════════════════════════════════════════════════════════

def _resize_padded_bilinear(arr, t_dim):
    """Bilinear resize for flux arrays (smooth fields).
    Input shape: (N+2, M+2) with 1-pixel halo.
    Output shape: (t_dim+2, t_dim+2).
    """
    core = arr[1:-1, 1:-1]
    top_halo    = arr[0, 1:-1]
    bottom_halo = arr[-1, 1:-1]
    left_halo   = arr[1:-1, 0]
    right_halo  = arr[1:-1, -1]
    corners = {
        'tl': arr[0, 0],   'tr': arr[0, -1],
        'bl': arr[-1, 0],  'br': arr[-1, -1]
    }

    h_core, w_core = core.shape
    new_core   = zoom(core, (t_dim / h_core, t_dim / w_core), order=1)
    new_top    = zoom(top_halo,    (t_dim / len(top_halo),),    order=1)
    new_bottom = zoom(bottom_halo, (t_dim / len(bottom_halo),), order=1)
    new_left   = zoom(left_halo,   (t_dim / len(left_halo),),   order=1)
    new_right  = zoom(right_halo,  (t_dim / len(right_halo),),  order=1)

    final = np.zeros((t_dim + 2, t_dim + 2))
    final[1:-1, 1:-1] = new_core
    final[0, 1:-1]    = new_top
    final[-1, 1:-1]   = new_bottom
    final[1:-1, 0]    = new_left
    final[1:-1, -1]   = new_right
    final[0, 0]   = corners['tl'];  final[0, -1]  = corners['tr']
    final[-1, 0]  = corners['bl'];  final[-1, -1] = corners['br']
    return final


def _resize_padded_maxpool(arr, t_dim):
    """Max-abs pool for Hamiltonian coefficients (anomaly preservation).
    Input shape: (N+2, M+2) with 1-pixel halo.
    Output shape: (t_dim+2, t_dim+2).
    """
    core = arr[1:-1, 1:-1]
    top_halo    = arr[0, 1:-1]
    bottom_halo = arr[-1, 1:-1]
    left_halo   = arr[1:-1, 0]
    right_halo  = arr[1:-1, -1]
    corners = {
        'tl': arr[0, 0],   'tr': arr[0, -1],
        'bl': arr[-1, 0],  'br': arr[-1, -1]
    }

    new_core   = _maxabs_pool_2d(core, t_dim, t_dim)
    new_top    = _maxabs_pool_1d(top_halo,    t_dim)
    new_bottom = _maxabs_pool_1d(bottom_halo, t_dim)
    new_left   = _maxabs_pool_1d(left_halo,   t_dim)
    new_right  = _maxabs_pool_1d(right_halo,  t_dim)

    final = np.zeros((t_dim + 2, t_dim + 2))
    final[1:-1, 1:-1] = new_core
    final[0, 1:-1]    = new_top
    final[-1, 1:-1]   = new_bottom
    final[1:-1, 0]    = new_left
    final[1:-1, -1]   = new_right
    final[0, 0]   = corners['tl'];  final[0, -1]  = corners['tr']
    final[-1, 0]  = corners['bl'];  final[-1, -1] = corners['br']
    return final


# ═══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
def _process_score(arr, is_periodic_scan, target_dim):
    if arr is None:
        return None
    arr = arr.astype(float)
    if is_periodic_scan:
        # No smoothing!  Max-pool directly to preserve localized features.
        return _maxabs_pool_2d(arr, target_dim, target_dim)
    else:
        return _resize_padded_maxpool(arr, target_dim)


def get_adaptive_flux(local_h, local_v, local_prev_h, local_prev_v, score, hamilt_params, target_dim=3, type_filter=True):
    """
    Adapte les flux et les paramètres à la dimension cible du VQA.

    Les TROIS chemins qui descendent vers le VQA — score, coefficients
    d'Hamiltonien, flux de contrainte — appliquent la meme reduction :
    max-abs pooling. Un signal fort et isole (choc, nappe de courant) dans
    un gros bloc doit survivre a la reduction ; c'est la raison d'etre de
    ces trois quantites.

    Le flux passait auparavant par un lissage puis une interpolation
    bilineaire, au motif qu'il serait un champ lisse. Il ne l'est pas :
    Phi est bati sur des DIFFERENCES de champ et pique la ou le score
    pique. Voir `_process_flux` pour la mesure.

    type_filter=True  (depth 0) : global periodic scan.
    type_filter=False (depth>0) : local sub-domain with halo.

    NOTE: No cross-component mixing.  Horizontal and vertical flux encode
    physically independent quantities (shear across different interfaces).
    Mixing them destroys directional information — e.g. a vortex has
    |Phi_h| ~ |Phi_v| with opposite signs; mixing cancels that signal.
    """

    proc_h = local_h.astype(float)
    proc_v = local_v.astype(float)

    # ── Flux dispatch (max-abs pool — anomaly preservation) ───────────
    #
    # Le flux passait auparavant par un lissage 3x3 puis `zoom(order=1)`,
    # justifie par « smooth physical fields ». Mais Phi n'est PAS un champ
    # lisse : c'est un indicateur d'anomalie, construit sur des DIFFERENCES
    # de champ, qui pique aux chocs et aux nappes de courant — exactement
    # comme le score et les coefficients, que ce fichier max-poole deja.
    #
    # Un zoom bilineaire ECHANTILLONNE, il ne moyenne pas. Mesure : un pic
    # isole place a 256 positions differentes dans un patch 128 -> 4 ne
    # survivait qu'a UNE d'entre elles ; place au centre il rendait
    # exactement 0.0000 la ou le max-pooling rend 1000 et la moyenne de
    # bloc 0.98. Sur champs DNS reels, part du pic de Phi conservee :
    # orszag_tang 38 %, mhd_rotor 70 %, kelvin_helmholtz et harris_tearing
    # 100 %.
    #
    # Le lissage prealable aggravait le tout : il diluait le pic AVANT de
    # l'echantillonner, alors que `_process_score` porte explicitement
    # « No smoothing! » pour cette raison.
    #
    # Les trois chemins qui descendent vers le VQA — score, coefficients,
    # flux — appliquent desormais la meme reduction.
    def _process_flux(arr, is_periodic_scan):
        if arr is None:
            return None
        if is_periodic_scan:
            return _maxabs_pool_2d(arr, target_dim, target_dim)
        else:
            return _resize_padded_maxpool(arr, target_dim)

    # ── Hamiltonian dispatch (max-abs pool — anomaly preservation) ────
    def _process_hamilt(arr, is_periodic_scan):
        if arr is None:
            return None
        if is_periodic_scan:
            # No smoothing!  Max-pool directly to preserve localized features.
            return _maxabs_pool_2d(arr, target_dim, target_dim)
        else:
            return _resize_padded_maxpool(arr, target_dim)

    # ── Process flux arrays ───────────────────────────────────────────
    mini_h = _process_flux(proc_h, type_filter)
    mini_v = _process_flux(proc_v, type_filter)
    mini_score = _process_score(score, type_filter, target_dim)

    # ── Process Hamiltonian coefficients ──────────────────────────────
    mini_hamilt_params = {}
    if hamilt_params is not None:
        for key, value in hamilt_params.items():
            if key == 'E_max':
                # E_max est un scalaire d'echelle, pas un champ : il ne doit
                # PAS etre reduit. Le `if` suivant etait un `if` et non un
                # `elif`, si bien qu'un E_max devenu tableau aurait ete
                # silencieusement remplace par sa version poolee.
                mini_hamilt_params[key] = value
            elif isinstance(value, (tuple, list)):
                mini_hamilt_params[key] = tuple(
                    _process_hamilt(v, type_filter) for v in value
                )
            elif isinstance(value, np.ndarray):
                mini_hamilt_params[key] = _process_hamilt(value, type_filter)
            else:
                mini_hamilt_params[key] = value
    
    if local_prev_h is not None and local_prev_v is not None:
        proc_h_prev = local_prev_h.astype(float)
        proc_v_prev = local_prev_v.astype(float)
        mini_prev_h = _process_flux(proc_h_prev, type_filter)
        mini_prev_v = _process_flux(proc_v_prev, type_filter)
        return mini_h, mini_v, mini_prev_h, mini_prev_v, mini_hamilt_params, mini_score

    return mini_h, mini_v, mini_hamilt_params, mini_score
