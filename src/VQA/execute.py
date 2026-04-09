# scripts/execute.py
import numpy as np
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit.quantum_info import Statevector

def execute(qc, cost_hamiltonian, mode, backend_name, shots, reps, K_opt, eps, E_max, verbose, vqa_runtime=None, method="COBYLA", warm_start_params=None):

    if verbose:
        for pauli, coeff in cost_hamiltonian.to_list():
            print(f"  {pauli}: {coeff}")
        print("\n","\n")
        print(f"Information about the circuit: \nNumber of qubits: {qc.num_qubits}\n Depth : {qc.depth()}")

    # 1. Resolve backend + primitives (reuse from runtime when available)
    if vqa_runtime is not None:
        estimator = vqa_runtime.estimator
        sampler = vqa_runtime.sampler
        backend = vqa_runtime._backend
    else:
        # Fallback: create fresh primitives (legacy path)
        if backend_name == "aer":
            backend = AerSimulator()
        elif backend_name == "estimator":
            backend = FakeFez()
        elif backend_name == "state_vector":
            backend = AerSimulator(method='statevector')
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")

        estimator = Estimator(mode=backend)
        estimator.options.default_shots = shots
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = shots

    # 2. ISA Hamiltonian
    if qc.layout is not None:
        isa_hamiltonian = cost_hamiltonian.apply_layout(qc.layout)
    else:
        isa_hamiltonian = cost_hamiltonian

    # Safety: detect Hamiltonians whose coefficients are ALL zero.
    # Qiskit's EstimatorV2 internally simplifies the SparsePauliOp, dropping
    # zero-coefficient terms. If all terms are zero, the observable becomes
    # empty and crashes with "Empty observable was detected."
    # In this case, the patch is genuinely calm — skip COBYLA and return
    # the θ-init marginals unchanged (no QAOA correction needed).
    _all_coeffs_zero = np.allclose(np.abs(isa_hamiltonian.coeffs), 0.0)

    if len(isa_hamiltonian) == 0 or _all_coeffs_zero:
        if verbose:
            print(f"\n--- Null Hamiltonian detected (all coefficients zero). "
                  f"Skipping optimization — returning θ-init marginals. ---")

        # Use warm-start params if available, otherwise zeros (p=0 = pure θ-init)
        if warm_start_params is not None and len(warm_start_params) == 2 * reps:
            optimal_params = warm_start_params
        else:
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

            if method == "L-BFGS-B":
                bounds_beta  = [(-beta_max, beta_max)] * reps
                bounds_gamma = [(0.0, 2.0 * np.pi)] * reps
                common['bounds'] = bounds_beta + bounds_gamma
                fd_eps = max(eps, 1.0 / np.sqrt(shots))
                common['options']['ftol'] = eps
                common['options']['gtol'] = eps * 10
                common['options']['eps'] = fd_eps
            elif method in ("COBYLA", "Powell"):
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
                common['tol'] = eps

            return common

        # 6. Optimization
        if mode == "simulator":
            result = minimize(**_build_minimize_kwargs(estimator))
        else:
            with Session(backend=backend) as session:
                hw_estimator = Estimator(mode=session)
                hw_estimator.options.default_shots = shots
                hw_estimator.options.dynamical_decoupling.enable = True
                hw_estimator.options.twirling.enable_gates = True

                result = minimize(**_build_minimize_kwargs(hw_estimator))

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