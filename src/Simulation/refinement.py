import numpy as np
from scipy.ndimage import zoom

from call_vqa_shell import call_vqa_shell
from VQA.cost_hamiltonian import NullHamiltonianError

from help_visual import visualize_vqa_step

from Simulation.RescaleArrays import get_adaptive_flux, _process_score


from Simulation.utils import slice_hamiltonian_params, get_periodic_patch


#: Patches dont l'Hamiltonien était vide (tous coefficients < COEFF_MIN).
#: Renseigné à l'exécution ; consultable via `null_hamiltonian_patches()`.
#: Ces patches conservent leur décision classique — le VQA n'est pas appelé.
_NULL_HAMILTONIAN_PATCHES = []


def null_hamiltonian_patches():
    """Patches rencontrés sans Hamiltonien, depuis le dernier reset."""
    return list(_NULL_HAMILTONIAN_PATCHES)


def reset_null_hamiltonian_patches():
    """Vide le compteur (à appeler en début de run si on veut le mesurer)."""
    _NULL_HAMILTONIAN_PATCHES.clear()


# ═══════════════════════════════════════════════════════════════════════
#  TTL (Time-To-Live) — temporal memory for patch persistence
#
#  Problem: an anomaly detected at time t may have its local signal
#  weaken at t+1 (due to zoom, noise, or transient fluctuation).
#  Without memory, the patch is immediately dropped → false negative.
#
#  Solution: each refined patch carries a TTL counter (default τ=1).
#  When a patch is flagged for refinement, ttl = τ.
#  At each hybrid step, patches with ttl > 0 survive even if their
#  VQA probability drops below threshold (ttl is decremented).
#  This stabilizes the refinement tree without forcing spatial inheritance.
#
#  NOTE: τ=3 caused oscillations of period 3 in Jaccard stability —
#  patches survived artificially for 3 steps then dropped abruptly.
#  τ=1 gives the QAOA one grace period while allowing intrinsic
#  stability to be evaluated honestly.
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_TTL = 1   # Survives 1 hybrid step after last detection


def _boundary_activation(prob_map, target_dim):
    """
    Detect if an anomaly is touching the patch boundary.

    Checks whether edge qubits (top/bottom/left/right rows/columns)
    have significantly higher activation than interior qubits.
    Returns a dict with directional flags: {'top', 'bottom', 'left', 'right'}
    indicating which boundaries have elevated probability.

    This is a cheap O(N²) test that signals "something is happening
    near the boundary" — the anomaly may extend beyond this patch.
    """
    if target_dim < 2:
        return {}

    interior = prob_map[1:-1, 1:-1] if target_dim > 2 else prob_map
    mean_interior = np.mean(interior) if interior.size > 0 else 0.0
    threshold = max(mean_interior + 0.1, 0.3)

    flags = {}
    if np.mean(prob_map[0, :]) > threshold:
        flags['top'] = True
    if np.mean(prob_map[-1, :]) > threshold:
        flags['bottom'] = True
    if np.mean(prob_map[:, 0]) > threshold:
        flags['left'] = True
    if np.mean(prob_map[:, -1]) > threshold:
        flags['right'] = True
    return flags


def _downsample_fields(fields, y_s, y_e, x_s, x_e, target_dim, pad=0):
    """Crop and area-average physics fields to target_dim × target_dim.

    Uses mean-pooling (area averaging) which preserves the physical
    mean in each coarse cell — appropriate for velocity/magnetic fields.
    """
    result = {}
    for key in ('vx', 'vy', 'Bx', 'By', 'Jz'):
        patch = get_periodic_patch(fields[key], y_s, y_e, x_s, x_e, pad)
        h, w = patch.shape
        # Mean-pool to target_dim (+ 2*pad if bounded)
        out_dim = target_dim + 2 * pad
        bh = h // out_dim
        bw = w // out_dim
        if bh < 1 or bw < 1:
            result[key] = zoom(patch, (out_dim / h, out_dim / w), order=1)
        else:
            cropped = patch[:out_dim * bh, :out_dim * bw]
            result[key] = cropped.reshape(out_dim, bh, out_dim, bw).mean(axis=(1, 3))
    return result


def _prepare_vqa_input(
    full_phi_h, full_phi_v, full_prev_h, full_prev_v,
    full_score,
    full_physics_state, bounds, depth, mapper, args,
    AveragePhiDev, beta, target_dim,
    HamiltMapper, sim, threshold_amr=0.5,
):
    """
    Extract, downsample, and encode one patch for VQA.

    Returns (angles, mini_hamilt_params, mini_score) or None.
    angles = (theta_h, theta_v, psi_h, psi_v) where θ encodes the classical score.
    mini_score = downsampled (target_dim, target_dim) score for before-QAOA comparison.

    For depth==0 (periodic): no pad, just the score
    For depth>0 (bounded): score has no halo, extract without pad

    Hamiltonian coefficients are computed at VQA resolution (target_dim ×
    target_dim) with effective dx = patch_physical_size / target_dim.
    This ensures threshold-contrast is evaluated at the scale where the
    physics IS under-resolved (high Re_cell), producing non-zero
    coefficients for anomalous regions.
    """
    y_s, y_e, x_s, x_e = bounds
    pad = 1 if depth > 0 else 0

    mini_Phi_dict = None
    mini_Phi_prev_dict = None

    local_h = get_periodic_patch(full_phi_h, y_s, y_e, x_s, x_e, pad)
    local_v = get_periodic_patch(full_phi_v, y_s, y_e, x_s, x_e, pad)
    local_score = get_periodic_patch(full_score, y_s, y_e, x_s, x_e, pad)

    # ── Compute Hamiltonian at VQA resolution ──────────────────────
    # Downsample physics fields to VQA grid and compute coefficients
    # with the effective cell size dx_eff = patch_phys_size / target_dim.
    # At VQA resolution, Re_cell >> 1 (under-resolved), so the
    # threshold-contrast produces non-zero coefficients.
    mini_fields = _downsample_fields(
        full_physics_state, y_s, y_e, x_s, x_e,
        target_dim, pad=pad,
    )
    patch_phys_size = (y_e - y_s) / full_phi_h.shape[0] * sim.grid.L
    dx_eff = patch_phys_size / target_dim
    mini_score_for_hamilt = _process_score(
        local_score, depth == 0, target_dim + 2 * pad if pad > 0 else target_dim,
    )
    mini_hamilt_params = HamiltMapper.compute_coefficients(
        sim, mini_score_for_hamilt, mini_fields, threshold_amr,
        advanced_anomalies_enabled=args.AdvAnomaliesEnable,
        dx_override=dx_eff,
    )

    if full_prev_h is not None:
        local_prev_h = get_periodic_patch(full_prev_h, y_s, y_e, x_s, x_e, pad)
        local_prev_v = get_periodic_patch(full_prev_v, y_s, y_e, x_s, x_e, pad)
        mini_h, mini_v, mini_prev_h, mini_prev_v, _, mini_score = get_adaptive_flux(
            local_h, local_v, local_prev_h, local_prev_v, local_score, None,
            target_dim=target_dim,
            type_filter=depth == 0,
        )
        mini_Phi_prev_dict = {'phi_horizontal': mini_prev_h, 'phi_vertical': mini_prev_v}
    else:
        mini_h, mini_v, _, mini_score = get_adaptive_flux(
            local_h, local_v, None, None, local_score, None,
            target_dim=target_dim,
            type_filter=depth == 0,
        )
        mini_Phi_prev_dict = None

    mini_Phi_dict = {'phi_horizontal': mini_h, 'phi_vertical': mini_v}


    mini_score = np.clip(mini_score, 0.0, 1.0)

    # The same score is used for both h and v qubit initialization
    angles = mapper.map_to_angles(
        score_h=mini_score, score_v=mini_score,
        phi_dict_prev=mini_Phi_prev_dict, phi_dict=mini_Phi_dict,
        AveragePhiDev=AveragePhiDev, beta=beta,
    )
    return angles, mini_hamilt_params, mini_score


def _run_level(
    pending_patches,
    depth,
    full_phi_h, full_phi_v, full_prev_h, full_prev_v,
    full_score,
    full_physics_state,
    mapper, args,
    AveragePhiDev,
    beta,
    target_dim, max_depth, min_size, threshold_amr,
    active_patches,
    verbose=False,
    vqa_runtime=None,
    solve_max_depth=None,
    ttl_map=None,
    warm_start_cache=None,
    HamiltMapper=None,
    sim=None,
):
    """Process all patches at one depth level and return the next level's patches."""
    _solve_depth = solve_max_depth if solve_max_depth is not None else max_depth
    _offset = _solve_depth - max_depth

    next_level = []
    period_bound = (depth == 0)

    for bounds in pending_patches:
        y_s, y_e, x_s, x_e = bounds
        height = y_e - y_s
        width = x_e - x_s

        if height < min_size or width < min_size:
            active_patches.append({
                'bounds': bounds, 'depth': _solve_depth, 'type': 'leaf_limit'
            })
            continue

        prep = _prepare_vqa_input(
            full_phi_h, full_phi_v, full_prev_h, full_prev_v,
            full_score,
            full_physics_state, bounds, depth, mapper, args,
            AveragePhiDev, beta, target_dim,
            HamiltMapper=HamiltMapper, sim=sim,
            threshold_amr=threshold_amr,
        )
        if prep is None:
            continue
        angles, mini_hamilt_params, mini_score = prep

        # Before-QAOA baseline: classical score (same probabilities as θ encoding)
        prob_map_avant_qaoa = mini_score
        if depth > 0:
            prob_map_avant_qaoa = mini_score[1:-1, 1:-1]

        # At max depth, no further refinement decision is needed — skip VQA entirely
        if depth >= max_depth:
            if verbose:
                print(f"\n  ┌─ Depth {depth} | Patch {bounds} | eff_threshold=1.000 (max depth, VQA skipped)")
                print(f"  └─")
            active_patches.append({
                'bounds': bounds, 'depth': _solve_depth,
                'score': np.max(prob_map_avant_qaoa), 'type': 'leaf_depth',
            })
            continue

        # Warm-start: use previous optimal params for this patch (or depth level)
        ws_params = None
        if warm_start_cache is not None:
            ws_params = warm_start_cache.get(bounds)
            if ws_params is None:
                ws_params = warm_start_cache.get('_global')

        try:
            result = call_vqa_shell(
                angles, mini_hamilt_params, verbose, args,
                period_bound=period_bound,
                vqa_runtime=vqa_runtime,
                warm_start_params=ws_params,
            )
        except NullHamiltonianError as exc:
            # Le patch ne définit aucun problème d'optimisation : tous ses
            # coefficients sont sous COEFF_MIN. On garde explicitement la
            # décision classique (θ-init) et on compte l'événement, au lieu
            # de faire tourner le VQA contre un opérateur fabriqué.
            _NULL_HAMILTONIAN_PATCHES.append({'bounds': bounds, 'depth': depth})
            if verbose:
                print(f"\n  ┌─ Depth {depth} | Patch {bounds} | {exc}")
                print(f"  │  décision classique conservée (VQA non appelé)")
                print(f"  └─")
            prob_map = np.asarray(prob_map_avant_qaoa, dtype=float)
            optimal_params = ws_params
            result = None
        else:
            if result is None:
                continue
            probs, optimal_params = result

            num_edges = target_dim * target_dim
            probs_h = probs[:num_edges].reshape(target_dim, target_dim)
            probs_v = probs[num_edges:].reshape(target_dim, target_dim)
            prob_map = 0.5 * (probs_h + probs_v)

        # Store optimal params for warm-starting next hybrid step
        if warm_start_cache is not None and optimal_params is not None:
            warm_start_cache[bounds] = optimal_params
            warm_start_cache['_global'] = optimal_params  # fallback for new patches

        # Boundary activation detection (for directional probing info)
        boundary_flags = _boundary_activation(prob_map, target_dim)

        if verbose:
            effective_thr = threshold_amr + (1.0 - threshold_amr) * depth / max_depth
            print(f"\n  ┌─ Depth {depth} | Patch {bounds} | eff_threshold={effective_thr:.3f}")
            print(f"  │  θ-only  (before QAOA): {np.array2string(prob_map_avant_qaoa, precision=3, suppress_small=True)}")
            print(f"  │  QAOA    (after  QAOA): {np.array2string(prob_map, precision=3, suppress_small=True)}")
            refine_theta = prob_map_avant_qaoa >= effective_thr
            refine_qaoa  = prob_map >= effective_thr
            print(f"  │  θ-only  refine: {refine_theta.astype(int)}")
            print(f"  │  QAOA    refine: {refine_qaoa.astype(int)}")
            agree = np.sum(refine_theta == refine_qaoa)
            total = refine_theta.size
            print(f"  │  Agreement: {agree}/{total} cells")
            if boundary_flags:
                print(f"  │  Boundary activation: {list(boundary_flags.keys())}")
            print(f"  └─")

        step_y = height // target_dim
        step_x = width // target_dim
        effective_threshold = threshold_amr # threshold_amr + (1.0 - threshold_amr) * depth / max_depth

        for i in range(target_dim):
            for j in range(target_dim):
                local_prob = prob_map[i, j]
                sub_y_s = y_s + i * step_y
                sub_y_e = y_s + (i + 1) * step_y if i < target_dim - 1 else y_e
                sub_x_s = x_s + j * step_x
                sub_x_e = x_s + (j + 1) * step_x if j < target_dim - 1 else x_e
                sub_bounds = (sub_y_s, sub_y_e, sub_x_s, sub_x_e)

                # TTL check: if this sub-patch has remaining TTL, force refinement
                ttl_key = sub_bounds
                has_ttl = (ttl_map is not None
                           and ttl_key in ttl_map
                           and ttl_map[ttl_key] > 0)

                if local_prob >= effective_threshold:
                    next_level.append(sub_bounds)
                    # Reset TTL on fresh detection
                    if ttl_map is not None:
                        ttl_map[ttl_key] = DEFAULT_TTL
                elif has_ttl:
                    # Signal dropped but TTL still active → keep refining
                    next_level.append(sub_bounds)
                    ttl_map[ttl_key] -= 1
                    if verbose:
                        print(f"  │  TTL keep: {sub_bounds} (ttl={ttl_map[ttl_key]})")
                else:
                    active_patches.append({
                        'bounds': sub_bounds, 'depth': depth + _offset,
                        'score': local_prob, 'type': 'coarse_leaf',
                    })

                # Boundary-aware probing: if anomaly touches the boundary
                # toward this sub-cell, force refinement even if prob is marginal
                if not has_ttl and local_prob < effective_threshold:
                    should_probe = False
                    if i == 0 and 'top' in boundary_flags:
                        should_probe = True
                    if i == target_dim - 1 and 'bottom' in boundary_flags:
                        should_probe = True
                    if j == 0 and 'left' in boundary_flags:
                        should_probe = True
                    if j == target_dim - 1 and 'right' in boundary_flags:
                        should_probe = True
                    if should_probe and local_prob >= effective_threshold * 0.5:
                        # Marginal signal + boundary activation → probe deeper
                        next_level.append(sub_bounds)
                        if verbose:
                            print(f"  │  Boundary probe: {sub_bounds} (prob={local_prob:.3f})")

    return next_level


# ═══════════════════════════════════════════════════════════════════════
#  CLASSICAL AMR — multi-indicator detector (baseline for Q-HAS comparison)
#
#  Uses AngleMapper.classical_score() which combines:
#    1. Vorticity  |ωz|           — shear layers, vortex cores, KH rolls
#    2. Velocity divergence |∇·v| — compression / shocks
#    3. Current density |Jz|      — current sheets, reconnection
#    4. Löhner error estimator    — scale-free second-derivative sensor
# ═══════════════════════════════════════════════════════════════════════



def _run_level_classical(
    pending_patches, depth,
    full_score,
    target_dim, max_depth, min_size, threshold_amr,
    active_patches,
    verbose=False,
    solve_max_depth=None,
    ttl_map=None,
):
    """Process all patches at one depth using the multi-indicator classical detector."""
    _solve_depth = solve_max_depth if solve_max_depth is not None else max_depth
    _offset = _solve_depth - max_depth

    next_level = []

    for bounds in pending_patches:
        y_s, y_e, x_s, x_e = bounds
        height, width = y_e - y_s, x_e - x_s

        if height < min_size or width < min_size:
            active_patches.append({
                'bounds': bounds, 'depth': _solve_depth, 'type': 'leaf_limit',
            })
            continue

        y_s, y_e, x_s, x_e = bounds

        pad = 1 if depth > 0 else 0
        local_score = get_periodic_patch(full_score, y_s, y_e, x_s, x_e, pad=pad)
        
        is_periodic = (depth == 0)
        score_map_padded = _process_score(local_score, is_periodic, target_dim)
        
        # Extraire le cœur 2x2 de la matrice 4x4
        if depth > 0:
            score_map = score_map_padded[1:-1, 1:-1]
        else:
            score_map = score_map_padded

        if depth >= max_depth:
            active_patches.append({
                'bounds': bounds, 'depth': _solve_depth,
                'score': float(np.max(score_map)), 'type': 'leaf_depth',
            })
            continue

        # Boundary activation detection (same as VQA path for fair comparison)
        boundary_flags = _boundary_activation(score_map, target_dim)

        step_y = height // target_dim
        step_x = width // target_dim
        effective_threshold = threshold_amr # threshold_amr + (1.0 - threshold_amr) * depth / max_depth

        for i in range(target_dim):
            for j in range(target_dim):
                local_score = score_map[i, j]
                sub_y_s = y_s + i * step_y
                sub_y_e = y_s + (i + 1) * step_y if i < target_dim - 1 else y_e
                sub_x_s = x_s + j * step_x
                sub_x_e = x_s + (j + 1) * step_x if j < target_dim - 1 else x_e
                sub_bounds = (sub_y_s, sub_y_e, sub_x_s, sub_x_e)

                # TTL check: if this sub-patch has remaining TTL, force refinement
                ttl_key = sub_bounds
                has_ttl = (ttl_map is not None
                           and ttl_key in ttl_map
                           and ttl_map[ttl_key] > 0)

                if local_score >= effective_threshold:
                    next_level.append(sub_bounds)
                    # Reset TTL on fresh detection
                    if ttl_map is not None:
                        ttl_map[ttl_key] = DEFAULT_TTL
                elif has_ttl:
                    # Signal dropped but TTL still active → keep refining
                    next_level.append(sub_bounds)
                    ttl_map[ttl_key] -= 1
                    if verbose:
                        print(f"  │  TTL keep (classical): {sub_bounds} (ttl={ttl_map[ttl_key]})")
                else:
                    active_patches.append({
                        'bounds': sub_bounds, 'depth': depth + _offset,
                        'score': local_score, 'type': 'coarse_leaf',
                    })

                # Boundary-aware probing (same logic as VQA path)
                if not has_ttl and local_score < effective_threshold:
                    should_probe = False
                    if i == 0 and 'top' in boundary_flags:
                        should_probe = True
                    if i == target_dim - 1 and 'bottom' in boundary_flags:
                        should_probe = True
                    if j == 0 and 'left' in boundary_flags:
                        should_probe = True
                    if j == target_dim - 1 and 'right' in boundary_flags:
                        should_probe = True
                    if should_probe and local_score >= effective_threshold * 0.5:
                        next_level.append(sub_bounds)
                        if verbose:
                            print(f"  │  Boundary probe (classical): {sub_bounds} (score={local_score:.3f})")

    return next_level


def run_adaptive_classical(
    sim, mapper, threshold_amr, target_dim, max_depth, min_size,
    verbose=False,
    solve_max_depth=None,
    ttl_map=None,
):
    """
    Classical AMR baseline — same BFS structure as run_adaptive_vqa,
    but uses standard multi-indicator detection instead of a quantum circuit.

    ttl_map : dict mapping bounds → remaining TTL steps. Passed in from the
    pipeline to persist across hybrid steps. If None, TTL is disabled.
    """
    from Simulation.PhysToAngle import AngleMapper

    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)

    full_score = AngleMapper.classical_score(physics_state)

    H, W = full_score.shape
    initial_bounds = (0, H, 0, W)

    _solve_depth = solve_max_depth if solve_max_depth is not None else max_depth

    final_patches = []
    if verbose:
        print(f"--- START CLASSICAL AMR SCAN ---")

    pending = [initial_bounds]
    depth = 0

    while pending and depth <= max_depth:
        if verbose:
            print(f"  Depth {depth}: {len(pending)} patch(es)")

        pending = _run_level_classical(
            pending, depth,
            full_score=full_score,
            target_dim=target_dim, max_depth=max_depth,
            min_size=min_size, threshold_amr=threshold_amr,
            active_patches=final_patches,
            verbose=verbose,
            solve_max_depth=_solve_depth,
            ttl_map=ttl_map,
        )
        depth += 1

    for bounds in pending:
        final_patches.append({
            'bounds': bounds, 'depth': _solve_depth, 'type': 'leaf_depth',
        })

    if len(final_patches) == 0:
        if verbose:
            print(">>> Classical AMR found nothing. Defaulting to FULL.")
        final_patches.append({
            'bounds': (0, H, 0, W), 'depth': 0, 'type': 'fallback',
        })

    if verbose:
        print(f"--- CLASSICAL SCAN COMPLETE: {len(final_patches)} patches ---")

    return final_patches, Phi


def run_adaptive_vqa(
    sim, mapper, HamiltMapper, args,
    Phi_prev,
    beta,
    threshold_amr,
    target_dim,
    max_depth,
    min_size,
    verbose=False,
    vqa_runtime=None,
    solve_max_depth=None,
    ttl_map=None,
    warm_start_cache=None,
    ):
    """
    Point d'entree principal — level-by-level VQA scan.

    Instead of a recursive DFS, we process ALL patches at the same depth
    before moving to the next level. This enables future batched execution
    and makes the control flow easier to reason about.

    threshold_amr   : passed to the level scan for the recursion decision.

    solve_max_depth : si fourni, les patches leaf_depth sont stockés avec
    depth=solve_max_depth plutôt que depth=scan_depth. Cela permet à
    step_layered (appelé avec max_depth=solve_max_depth) de calculer
    local_factor=1 (full DNS) pour les zones instables, même si le scan
    VQA s'est arrêté plus tôt (max_depth < solve_max_depth).

    ttl_map : dict mapping bounds → remaining TTL steps. Passed in from the
    pipeline to persist across hybrid steps. If None, TTL is disabled.
    """
    _solve_depth = solve_max_depth if solve_max_depth is not None else max_depth

    # 1. Prepare full-domain data (read-only for the scan)
    physics_state = sim.get_fluxes()
    Phi = mapper.compute_stress_flux(physics_state)

    full_h = Phi['phi_horizontal']
    full_v = Phi['phi_vertical']

    full_prev_h = None
    full_prev_v = None
    AveragePhiDev = None

    if Phi_prev is not None:
        full_prev_h = Phi_prev['phi_horizontal']
        full_prev_v = Phi_prev['phi_vertical']
        AveragePhiDev = 0.5 * (np.mean(np.abs(full_h - full_prev_h))
                                + np.mean(np.abs(full_v - full_prev_v)))

    # Classical multi-indicator score for θ initialization.
    # Uses domain-max normalization (same as the classical AMR baseline)
    # so that VQA and classical AMR start from the SAME score map.
    # This ensures a fair comparison: any difference in refinement
    # decisions is due to the QAOA circuit, not different scoring.
    from Simulation.PhysToAngle import AngleMapper as _AM
    full_score = _AM.classical_score(physics_state)

    # H_edges = 0 (v9): Z-terms removed from Hamiltonian.
    # The classical score is encoded in θ init, not in H.
    # The QAOA cost contains only ZZ/ZZZZ spatial correlations.
    # Coefficients are now computed PER-PATCH at VQA resolution inside
    # _prepare_vqa_input, so we no longer compute them globally here.

    H, W = full_h.shape
    initial_bounds = (0, H, 0, W)

    final_patches = []
    if verbose:
        print(f"--- START LEVEL-BY-LEVEL VQA SCAN ---")
        print("Average Phi Dev:", AveragePhiDev)

    # 2. Level-by-level scan (BFS instead of DFS)
    pending = [initial_bounds]
    depth = 0

    while pending and depth <= max_depth:
        if verbose:
            print(f"  Depth {depth}: {len(pending)} patch(es) to evaluate")

        pending = _run_level(
            pending, depth,
            full_phi_h=full_h, full_phi_v=full_v,
            full_prev_h=full_prev_h, full_prev_v=full_prev_v,
            full_score=full_score,
            full_physics_state=physics_state,
            mapper=mapper, args=args,
            AveragePhiDev=AveragePhiDev,
            beta=beta,
            target_dim=target_dim, max_depth=max_depth,
            min_size=min_size, threshold_amr=threshold_amr,
            active_patches=final_patches,
            verbose=verbose,
            vqa_runtime=vqa_runtime,
            solve_max_depth=_solve_depth,
            ttl_map=ttl_map,
            warm_start_cache=warm_start_cache,
            HamiltMapper=HamiltMapper,
            sim=sim,
        )
        depth += 1

    # Any patches still pending after max_depth are leaves
    for bounds in pending:
        final_patches.append({
            'bounds': bounds, 'depth': _solve_depth, 'type': 'leaf_depth',
        })

    if len(final_patches) == 0:
        if verbose:
            print(">>> VQA found nothing active. Defaulting to FULL COMPUTATION.")
        final_patches.append({
            'bounds': (0, H, 0, W), 'depth': 0, 'type': 'fallback'})
    if verbose:
        print(f"--- SCAN COMPLETE: {len(final_patches)} Active Zones Identified ---")
        print(final_patches)

    # 3. Classical baseline on the SAME physics state (independent BFS)
    #    This replaces the broken piggyback tracking that was inside _run_level.
    final_patches_wo_vqa = []
    pending_cl = [initial_bounds]
    depth_cl = 0
    while pending_cl and depth_cl <= max_depth:
        pending_cl = _run_level_classical(
            pending_cl, depth_cl,
            full_score=full_score,
            target_dim=target_dim, max_depth=max_depth,
            min_size=min_size, threshold_amr=threshold_amr,
            active_patches=final_patches_wo_vqa,
            verbose=False,
            solve_max_depth=_solve_depth,
            ttl_map=None,
        )
        depth_cl += 1
    for bounds in pending_cl:
        final_patches_wo_vqa.append({
            'bounds': bounds, 'depth': _solve_depth, 'type': 'leaf_depth',
        })
    if len(final_patches_wo_vqa) == 0:
        final_patches_wo_vqa.append({
            'bounds': (0, H, 0, W), 'depth': 0, 'type': 'fallback'})

    return final_patches, final_patches_wo_vqa, Phi