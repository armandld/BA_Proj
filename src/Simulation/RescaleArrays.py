import numpy as np
from scipy.ndimage import uniform_filter, zoom


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
    arr_c = arr[:target_h * bh, :target_w * bw]
    # (target_h, bh, target_w, bw) → (target_h, target_w, bh*bw)
    blocks = arr_c.reshape(target_h, bh, target_w, bw)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(target_h, target_w, -1)
    idx = np.argmax(np.abs(blocks), axis=-1)
    return np.take_along_axis(blocks, idx[..., np.newaxis], axis=-1).squeeze(-1)


def _maxabs_pool_1d(arr, target_len):
    """1D max-abs pooling for halo (boundary) arrays."""
    n = len(arr)
    bs = n // target_len
    if bs < 1:
        return zoom(arr, (target_len / n,), order=1)
    arr_c = arr[:target_len * bs]
    blocks = arr_c.reshape(target_len, bs)
    idx = np.argmax(np.abs(blocks), axis=-1)
    return np.take_along_axis(blocks, idx[..., np.newaxis], axis=-1).squeeze(-1)


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

    - Flux arrays      → bilinear interpolation (smooth physical fields).
    - Hamiltonian coefs → max-abs pooling (anomaly detection: a single
      strong signal in a large block must survive downsampling).

    type_filter=True  (depth 0) : global periodic scan.
    type_filter=False (depth>0) : local sub-domain with halo.

    NOTE: No cross-component mixing.  Horizontal and vertical flux encode
    physically independent quantities (shear across different interfaces).
    Mixing them destroys directional information — e.g. a vortex has
    |Phi_h| ~ |Phi_v| with opposite signs; mixing cancels that signal.
    """

    proc_h = local_h.astype(float)
    proc_v = local_v.astype(float)

    # ── Flux dispatch (bilinear — smooth fields) ──────────────────────
    def _process_flux(arr, is_periodic_scan):
        if arr is None:
            return None
        if is_periodic_scan:
            h, w = arr.shape
            processed = arr
            if min(h, w) > target_dim:
                processed = uniform_filter(arr, size=3, mode='wrap')
            return zoom(processed, (target_dim / h, target_dim / w), order=1)
        else:
            return _resize_padded_bilinear(arr, target_dim)

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
                mini_hamilt_params[key] = value
            if isinstance(value, (tuple, list)):
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
