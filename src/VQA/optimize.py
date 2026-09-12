"""Circuit transpilation for calls that do not use :class:`VQARuntime`."""

from qiskit_aer import AerSimulator
from qiskit.transpiler import generate_preset_pass_manager


SUPPORTED_BACKENDS = ("state_vector", "matrix_product_state", "aer")


def optimize(qc, backend, opt_level, verbose, seed=0):
    """Transpile ``qc`` for a seeded local simulator."""
    if backend == "state_vector":
        simulator = AerSimulator(
            method="statevector", seed_simulator=seed)
        effective_level = 0
    elif backend == "matrix_product_state":
        simulator = AerSimulator(
            method="matrix_product_state", seed_simulator=seed)
        effective_level = 0
    elif backend == "aer":
        simulator = AerSimulator(seed_simulator=seed)
        effective_level = opt_level
    else:
        raise ValueError(
            f"Unsupported backend: {backend!r}. Expected one of "
            f"{SUPPORTED_BACKENDS}.")

    pass_manager = generate_preset_pass_manager(
        optimization_level=effective_level,
        backend=simulator,
        seed_transpiler=seed,
    )
    circuit = pass_manager.run(qc)
    if verbose:
        print("Optimization Level:", effective_level)
    return circuit
