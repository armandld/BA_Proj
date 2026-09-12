# VQA/runtime.py
#
# Reusable Qiskit primitives and circuit caches for one pipeline run.

import numpy as np
from qiskit.circuit.library import QAOAAnsatz


def _hamiltonian_fingerprint(op):
    """Empreinte hachable d'un SparsePauliOp : etiquettes ET coefficients.

    Les coefficients sont arrondis a 12 decimales pour qu'une difference de
    dernier bit ne fasse pas exploser le cache, sans jamais confondre deux
    Hamiltoniens physiquement distincts.
    """
    return tuple(sorted(
        (str(p), round(complex(c).real, 12), round(complex(c).imag, 12))
        for p, c in zip(op.paulis, op.coeffs)
    ))


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

    SUPPORTED_BACKENDS = ("state_vector", "matrix_product_state", "aer")
    SUPPORTED_MODES = ("simulator",)

    def __init__(self, backend_name, mode, shots, opt_level, seed=None):
        self.backend_name = backend_name
        self.mode = mode
        self.shots = shots
        self.opt_level = opt_level
        if seed is None:
            # Aucune graine demandee -> le bras QAOA doit rester
            # stochastique par defaut (le protocole confirmatoire passe
            # toujours --qaoa-seed explicitement). Un `seed=0` par defaut
            # rendrait tout appelant deterministe sans le demander ; on
            # tire donc une graine reelle, une fois, a la construction du
            # runtime — les appels qui reutilisent CE runtime (une seule
            # execution de `pipeline()`) restent coherents entre eux,
            # tandis que deux executions independantes en tirent deux.
            seed = int(np.random.default_rng().integers(0, 2**32))
        elif isinstance(seed, (bool, np.bool_)) or not isinstance(
                seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        elif not 0 <= int(seed) <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 2**32 - 1")
        self.seed = int(seed)

        # Lazy-initialized primitives (created on first use)
        self._estimator = None
        self._sampler = None
        self._backend = None

        # Circuit cache: (num_qubits, period_bound) -> transpiled QAOAAnsatz
        self._ansatz_cache = {}

        self._validate_mode()
        self._init_backend()

    def _validate_mode(self):
        """Refuse un mode que le depot ne sait pas honorer.

        `mode` est STOCKE mais lu NULLE PART ailleurs : `_init_backend` ne
        dispatche que sur `backend_name`, et rend le meme `AerSimulator`
        pour `mode='simulator'` et pour `mode='hardware'`. Aucun chemin de
        ce depot ne resout un backend IBM reel.

        Sans cette garde, un mode `hardware` ne leverait pas : `execute`
        ouvrirait `Session(backend=AerSimulator)` — que qiskit-ibm-runtime
        ACCEPTE — puis y construirait un estimateur avec decouplage
        dynamique et twirling, des options qui n'y veulent rien dire sur un
        simulateur. Un run demande en `hardware` s'executerait donc sur un
        simulateur et rendrait des nombres parfaitement plausibles sans
        jamais le signaler.
        """
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode: {self.mode!r}. Expected one of "
                f"{self.SUPPORTED_MODES}.")

    # ------------------------------------------------------------------
    #  Backend / primitive initialization
    # ------------------------------------------------------------------
    def _init_backend(self):
        from qiskit_aer import AerSimulator

        if self.backend_name == "state_vector":
            self._backend = AerSimulator(
                method="statevector", seed_simulator=self.seed)
        elif self.backend_name == "matrix_product_state":
            self._backend = AerSimulator(
                method="matrix_product_state", seed_simulator=self.seed)
        elif self.backend_name == "aer":
            self._backend = AerSimulator(seed_simulator=self.seed)
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name!r}. Expected one "
                f"of {self.SUPPORTED_BACKENDS}.")
        self._init_aer_primitives()

    def _init_aer_primitives(self):
        from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
        self._estimator = Estimator(mode=self._backend)
        self._estimator.options.default_shots = self.shots
        self._estimator.options.simulator.seed_simulator = self.seed
        self._sampler = Sampler(mode=self._backend)
        self._sampler.options.default_shots = self.shots
        self._sampler.options.simulator.seed_simulator = self.seed

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
        """Return an ansatz cached by topology and Hamiltonian content."""
        key = (num_qubits, period_bound, reps,
               _hamiltonian_fingerprint(cost_hamiltonian))
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

        For aer: full transpilation to ISA-compliant native gates.
        """
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        # MPS and state_vector are ideal simulators — no routing/coupling needed
        level = 0 if self.backend_name in ("state_vector",
                                           "matrix_product_state") else self.opt_level
        pm = generate_preset_pass_manager(
            optimization_level=level,
            backend=self._backend,
            seed_transpiler=self.seed,
        )
        if verbose:
            print(f"Transpiling circuit (backend={self.backend_name}, opt_level={level})")
        return pm.run(qc)
