import numpy as np

from VQA.mapping import mapping
from VQA.execute import execute
from VQA.postprocess import postprocess


def call_vqa_shell(angles_tuple, hamilt_params, verbose, args, period_bound=True, vqa_runtime=None, warm_start_params=None):
    """Run one VQA evaluation.

    Parameters
    ----------
    angles_tuple : (theta_h, theta_v, psi_h, psi_v)
        θ encodes classical multi-indicator score: P(|1⟩) = sin²(θ/2) = score.
        ψ encodes stress flux temporal evolution.
    hamilt_params : dict
        Downsampled Hamiltonian coefficients for this patch.
    vqa_runtime : VQARuntime, optional
        Shared runtime with cached estimator/sampler/ansatz.
    warm_start_params : np.ndarray, optional
        Previous optimal QAOA parameters for warm-starting.
    """

    data = {
        "theta_h": angles_tuple[0].tolist(),
        "theta_v": angles_tuple[1].tolist(),
        "psi_h": angles_tuple[2].tolist(),
        "psi_v": angles_tuple[3].tolist()
    }

    reps = args.reps

    # ── Coefficient normalization ──────────────────────────────────────
    # Scale ALL Hamiltonian coefficients by 1/max(|coeff|) so the QAOA
    # cost landscape has O(1) coefficients.  This does NOT change the
    # ground state (uniform positive scaling preserves the minimum),
    # but makes COBYLA's step sizes and tolerance appropriate.
    #
    # Without this, ZZ coefficients reach -1193 (Orszag-Tang) or -526
    # (KH at depth 1), creating a rugged landscape where COBYLA
    # converges to the wrong degenerate minimum (all-|0⟩ instead of
    # all-|1⟩), inverting the QAOA probabilities.
    max_coeff = 0.0
    for key, value in hamilt_params.items():
        if isinstance(value, (tuple, list)):
            for v in value:
                if isinstance(v, np.ndarray):
                    max_coeff = max(max_coeff, np.max(np.abs(v)))
        elif isinstance(value, np.ndarray):
            max_coeff = max(max_coeff, np.max(np.abs(value)))

    if max_coeff > 1e-10:
        norm = max_coeff
        normalized_params = {}
        for key, value in hamilt_params.items():
            if isinstance(value, (tuple, list)):
                normalized_params[key] = tuple(
                    v / norm if isinstance(v, np.ndarray) else v
                    for v in value
                )
            elif isinstance(value, np.ndarray):
                normalized_params[key] = value / norm
            else:
                normalized_params[key] = value
        hamilt_params = normalized_params
        # After normalization all patches have O(1) coefficients, so
        # warm-start params from a previous (also normalized) call are
        # compatible.  We keep warm_start_params as-is.

    # Recompute E_max on the (now normalized) coefficients
    E_max = 0
    for key, value in hamilt_params.items():
        if isinstance(value, (tuple, list)):
            for v in value:
                if isinstance(v, np.ndarray):
                    E_max += np.sum(np.abs(v))
        elif isinstance(value, np.ndarray):
            E_max += np.sum(np.abs(value))

    if E_max < 1e-10:
        E_max = 1.0

    qc, cost_hamiltonian = mapping(data, hamilt_params, args.AdvAnomaliesEnable, period_bound, reps)

    # Transpile (skipped for state_vector when runtime is provided)
    if vqa_runtime is not None:
        qc = vqa_runtime.transpile(qc, verbose)
    else:
        from VQA.optimize import optimize
        qc = optimize(qc, args.backend, args.opt_level, verbose)

    probs_list, optimal_params = execute(
        qc, cost_hamiltonian, args.mode, args.backend,
        args.shots, reps, args.K_opt, args.eps, E_max, verbose,
        vqa_runtime=vqa_runtime,
        method=args.method,
        warm_start_params=warm_start_params,
    )

    probs_list = postprocess(probs_list, qc.num_qubits, verbose)

    return np.array(probs_list), optimal_params