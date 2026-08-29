# scripts/execute.py
import numpy as np
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit.quantum_info import Statevector


SUPPORTED_BACKENDS = ("state_vector", "matrix_product_state", "aer")


def _validated_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    return seed


def execute(qc, cost_hamiltonian, mode, backend_name, shots, reps, K_opt,
            eps, E_max, verbose, vqa_runtime=None, method="COBYLA",
            warm_start_params=None, seed=None):

    if mode != "simulator":
        raise ValueError(
            f"Unsupported mode: {mode!r}. Only 'simulator' is implemented.")
    if backend_name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend: {backend_name!r}. Expected one of "
            f"{SUPPORTED_BACKENDS}.")

    # `seed=None` ne veut pas dire "0" -- il veut dire "aucune graine
    # demandee". Avec un runtime, on herite silencieusement de la graine
    # QU'IL a deja resolue (fixe si on lui en a donne une, tiree une fois
    # s'il n'en avait pas) : c'est lui, pas cet appel, qui possede l'etat
    # partage entre les appels QAOA d'une meme execution. Sans runtime --
    # le chemin des appels directs, y compris les tests qui mesurent la
    # dispersion propre du bras QAOA -- une graine neuve est tiree a
    # CHAQUE appel, pour que la dispersion reste mesurable. Un `seed`
    # explicite reste verifie contre celle du runtime : c'est la seule
    # incoherence que cette fonction ne peut pas trancher seule.
    if vqa_runtime is not None:
        if seed is not None:
            seed = _validated_seed(seed)
            if vqa_runtime.seed != seed:
                raise ValueError(
                    f"runtime seed {vqa_runtime.seed} does not match "
                    f"requested seed {seed}")
        seed = vqa_runtime.seed
    elif seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32))
    else:
        seed = _validated_seed(seed)

    if verbose:
        for pauli, coeff in cost_hamiltonian.to_list():
            print(f"  {pauli}: {coeff}")
        print("\n","\n")
        print(f"Information about the circuit: \nNumber of qubits: {qc.num_qubits}\n Depth : {qc.depth()}")

    # 1. Resolve backend + primitives (reuse from runtime when available)
    if vqa_runtime is not None:
        if vqa_runtime.mode != mode:
            raise ValueError(
                f"runtime mode {vqa_runtime.mode!r} does not match "
                f"requested mode {mode!r}")
        if vqa_runtime.backend_name != backend_name:
            raise ValueError(
                f"runtime backend {vqa_runtime.backend_name!r} does not "
                f"match requested backend {backend_name!r}")
        # `seed` est deja celle du runtime (resolue plus haut) : rien a
        # revalider ici, la seule incoherence possible l'a ete avant.
        estimator = vqa_runtime.estimator
        sampler = vqa_runtime.sampler
        backend = vqa_runtime._backend
    else:
        if backend_name == "aer":
            backend = AerSimulator(seed_simulator=seed)
        elif backend_name == "state_vector":
            backend = AerSimulator(
                method="statevector", seed_simulator=seed)
        elif backend_name == "matrix_product_state":
            backend = AerSimulator(
                method="matrix_product_state", seed_simulator=seed)

        estimator = Estimator(mode=backend)
        estimator.options.default_shots = shots
        estimator.options.simulator.seed_simulator = seed
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = shots
        sampler.options.simulator.seed_simulator = seed

    # 2. ISA Hamiltonian
    if qc.layout is not None:
        isa_hamiltonian = cost_hamiltonian.apply_layout(qc.layout)
    else:
        isa_hamiltonian = cost_hamiltonian

        # A null Hamiltonian carries no QAOA correction; keep theta-init.
    _all_coeffs_zero = np.allclose(np.abs(isa_hamiltonian.coeffs), 0.0)

    if len(isa_hamiltonian) == 0 or _all_coeffs_zero:
        if verbose:
            print(f"\n--- Null Hamiltonian detected (all coefficients zero). "
                  f"Skipping optimization — returning θ-init marginals. ---")

        # Zero mixer and cost angles preserve the encoded marginal probabilities.
        optimal_params = np.zeros(2 * reps)

        optimized_circuit = qc.assign_parameters(optimal_params)

    else:
        # 3. Cost function
        objective_func_vals = []

        def cost_func_estimator(params, ansatz, hamiltonian, est):
            pub = (ansatz, hamiltonian, params)
            job = est.run([pub])
            result = job.result()[0]
            cost = result.data.evs

            objective_func_vals.append(cost)
            if len(objective_func_vals) % 10 == 0:
                if verbose:
                    print(f"Iter {len(objective_func_vals)}: Cost = {cost}")

            return cost

        # 4. Initial parameters (QAOA linear ramp, or warm-start from previous step)
        if warm_start_params is not None and len(warm_start_params) == 2 * reps:
            initial_params = warm_start_params
            if verbose:
                print(f"  Warm-starting from previous optimal params")
        else:
            gamma_total = np.pi / E_max
            k = np.arange(1, reps + 1)
            gamma_init = (2 * k) / (reps * (reps + 1)) * gamma_total
            initial_params = np.concatenate([
                np.full(reps, 0.0),   # Beta/Omega (mixer) — must start at zero
                gamma_init,            # Gamma (cost/phase separator) — linear ramp
            ])

        if verbose:
            print(f"\n--- Starting Optimization Loop (method={method}) ---")

        # 5. Build optimizer kwargs per method
        # ── Mixer angle bound ──────────────────────────────────────────
        # The QAOA cost layer exp(-iγH) only adds phases to the initial
        # state; it cannot change measurement probabilities on its own.
        # Only the mixer exp(-iβ ΣXi) changes P(|1⟩).  With the default
        # COBYLA rhobeg=1.0, the first trial point is β=1.0, which
        # rotates qubits ~60° from |1⟩ → P(|1⟩)≈0.25.  COBYLA then
        # gets trapped in this basin, suppressing ALL refinement probs.
        #
        # Fix: bound β so the mixer stays perturbative (small correction
        # to θ-init), and set rhobeg ≪ 1 so COBYLA explores locally.
        beta_max = np.pi / (4 * reps)   # ≈0.39 for p=2

        def _build_minimize_kwargs(est):
            """Return dict of kwargs for scipy.optimize.minimize."""
            common = dict(
                fun=cost_func_estimator,
                x0=initial_params,
                args=(qc, isa_hamiltonian, est),
                method=method,
                options={'maxiter': K_opt},
            )

            # Powell/L-BFGS-B accept bounds; COBYLA needs inequalities.
            bounds_beta  = [(-beta_max, beta_max)] * reps
            bounds_gamma = [(0.0, 2.0 * np.pi)] * reps

            if method == "L-BFGS-B":
                common['bounds'] = bounds_beta + bounds_gamma
                fd_eps = max(eps, 1.0 / np.sqrt(shots))
                common['options']['ftol'] = eps
                common['options']['gtol'] = eps * 10
                common['options']['eps'] = fd_eps
            elif method == "Powell":
                common['tol'] = eps
                common['bounds'] = bounds_beta + bounds_gamma
            elif method == "COBYLA":
                common['tol'] = eps
                # Small initial simplex keeps COBYLA near θ-init
                common['options']['rhobeg'] = 0.05
                # Bound mixer angles via inequality constraints
                constraints = []
                for i in range(reps):
                    constraints.append({'type': 'ineq',
                                        'fun': lambda x, _i=i: beta_max - x[_i]})
                    constraints.append({'type': 'ineq',
                                        'fun': lambda x, _i=i: x[_i] + beta_max})
                common['constraints'] = constraints
            else:
                # Refuse methods for which the mixer bound is not implemented.
                raise ValueError(
                    f"methode d'optimisation '{method}' non supportee : la "
                    f"borne sur le mixer (|beta| <= {beta_max:.4f}) ne peut "
                    "pas lui etre imposee. Utiliser COBYLA, Powell ou "
                    "L-BFGS-B.")

            return common

        # 6. Optimization
        result = minimize(**_build_minimize_kwargs(estimator))

        if verbose:
            print(f"Optimization success: {result.success}")
            print(f"Optimal Params: {result.x}")

        optimized_circuit = qc.assign_parameters(result.x)

    if backend_name == "state_vector":
        # Exact probabilities from statevector — no shot noise
        if verbose:
            print("\n--- Final Statevector (exact) ---")
        sv = Statevector.from_instruction(optimized_circuit)
        final_distribution = sv.probabilities_dict()
    elif backend_name == "matrix_product_state":
        # MPS: use sampler for the final distribution (statevector extraction
        # would defeat the purpose). Use a large shot count so marginals
        # approximate the exact distribution well.
        if verbose:
            print("\n--- Final MPS Sampling ---")
        optimized_circuit.measure_all()
        mps_shots = max(shots, 8192)
        # Re-configure the sampler's shot count for the final readout.
        # Volontairement non protégé : si l'affectation échoue, la lecture
        # se ferait au mauvais nombre de tirs et toutes les marginales en
        # aval seraient silencieusement plus bruitées.
        #
        # `sampler` peut appartenir a `vqa_runtime`, donc etre PARTAGE par
        # tous les appels de la campagne. L'ecrasement etait definitif :
        # apres un seul patch en MPS, chaque appel ulterieur tirait 8192
        # coups quel que soit `shots`. On restaure la valeur d'origine.
        previous_shots = sampler.options.default_shots
        try:
            sampler.options.default_shots = mps_shots
            pub = (optimized_circuit,)
            job = sampler.run([pub])
            pub_result = job.result()[0]
            counts_bin = pub_result.data.meas.get_counts()
        finally:
            sampler.options.default_shots = previous_shots
        total_shots = sum(counts_bin.values())
        final_distribution = {key: val / total_shots for key, val in counts_bin.items()}
    else:
        if verbose:
            print("\n--- Final Sampling ---")
        optimized_circuit.measure_all()
        pub = (optimized_circuit,)
        job = sampler.run([pub])
        pub_result = job.result()[0]
        counts_bin = pub_result.data.meas.get_counts()
        total_shots = sum(counts_bin.values())
        final_distribution = {key: val / total_shots for key, val in counts_bin.items()}

    # Return optimal params: from COBYLA result or from the null-Hamiltonian shortcut
    if _all_coeffs_zero or len(isa_hamiltonian) == 0:
        return final_distribution, optimal_params
    else:
        return final_distribution, result.x
