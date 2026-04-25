# VQA/runtime.py
#
# Singleton runtime that holds reusable Qiskit primitives and circuit caches.
# Created ONCE per pipeline run and threaded through every VQA call to avoid
# repeated instantiation of primitives and repeated transpilation / QAOAAnsatz
# construction.

import numpy as np
from qiskit.circuit.library import QAOAAnsatz


class VQARuntime:
    """Reusable VQA execution context.

    Holds:
      - A single EstimatorV2  (backed by AerSimulator for all backends)
      - A single SamplerV2    (backed by AerSimulator for all backends)
      - A cache of QAOAAnsatz circuits keyed by (num_qubits, period_bound)
        so that the expensive ansatz construction + decomposition happens
        only once per topology.

    All backends (including ``state_vector``) use Aer's compiled C++ engine
    via ``EstimatorV2`` / ``SamplerV2``.  The ``state_vector`` backend forces
    ``AerSimulator(method='statevector')`` which is functionally equivalent
    to ``aer`` for small circuits (Aer auto-selects statevector for ≤~14 qubits).
    """

    def __init__(self, backend_name, mode, shots, opt_level):
        self.backend_name = backend_name
        self.mode = mode
        self.shots = shots
        self.opt_level = opt_level

        # Lazy-initialized primitives (created on first use)
        self._estimator = None
        self._sampler = None
        self._backend = None

        # Circuit cache: (num_qubits, period_bound) -> transpiled QAOAAnsatz
        self._ansatz_cache = {}

        self._init_backend()

    # ------------------------------------------------------------------
    #  Backend / primitive initialization
    # ------------------------------------------------------------------
    def _init_backend(self):
        if self.backend_name == "state_vector":
            from qiskit_aer import AerSimulator
            self._backend = AerSimulator(method='statevector')
            self._init_aer_primitives()
        elif self.backend_name == "matrix_product_state":
            from qiskit_aer import AerSimulator
            # MPS simulator: scales to larger qubit counts when entanglement
            # is limited (local 2D Hamiltonians). bond_dim auto-grows, can be
            # capped via matrix_product_state_max_bond_dimension if needed.
            self._backend = AerSimulator(method='matrix_product_state')
            self._init_aer_primitives()
        elif self.backend_name == "aer":
            from qiskit_aer import AerSimulator
            self._backend = AerSimulator()
            self._init_aer_primitives()
        elif self.backend_name == "estimator":
            from qiskit_ibm_runtime.fake_provider import FakeFez
            self._backend = FakeFez()
            self._init_aer_primitives()

    def _init_aer_primitives(self):
        from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
        self._estimator = Estimator(mode=self._backend)
        self._estimator.options.default_shots = self.shots
        self._sampler = Sampler(mode=self._backend)
        self._sampler.options.default_shots = self.shots

    @property
    def estimator(self):
        return self._estimator

    @property
    def sampler(self):
        return self._sampler

    # ------------------------------------------------------------------
    #  Ansatz cache
    # ------------------------------------------------------------------
    def get_ansatz(self, cost_hamiltonian, reps, num_qubits, period_bound):
        """Return a cached QAOAAnsatz for this topology, or build + cache it."""
        key = (num_qubits, period_bound, reps)
        if key not in self._ansatz_cache:
            ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
            self._ansatz_cache[key] = ansatz
        return self._ansatz_cache[key]

    def invalidate_ansatz_cache(self):
        """Clear the cache (e.g. when hyperparams change the Hamiltonian structure)."""
        self._ansatz_cache.clear()

    # ------------------------------------------------------------------
    #  Transpilation
    # ------------------------------------------------------------------
    def transpile(self, qc, verbose=False):
        """Transpile a circuit against its backend.

        For state_vector: transpiles to AerSimulator(method='statevector')
        native gates at opt_level=0 (decompose PauliEvolutionGate, no routing
        needed for an ideal statevector backend).

        For aer/estimator: full transpilation to ISA-compliant native gates.
        """
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        # MPS and state_vector are ideal simulators — no routing/coupling needed
        level = 0 if self.backend_name in ("state_vector",
                                           "matrix_product_state") else self.opt_level
        pm = generate_preset_pass_manager(
            optimization_level=level, backend=self._backend
        )
        if verbose:
            print(f"Transpiling circuit (backend={self.backend_name}, opt_level={level})")
        return pm.run(qc)
