# VQA/runtime.py
#
# Singleton runtime that holds reusable Qiskit primitives and circuit caches.
# Created ONCE per pipeline run and threaded through every VQA call to avoid
# repeated instantiation of primitives and repeated transpilation / QAOAAnsatz
# construction.

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

        self._validate_mode()
        self._init_backend()

    #: Les seuls modes que ce depot sait executer.
    SUPPORTED_MODES = ("simulator",)

    def _validate_mode(self):
        """Refuse un mode que le depot ne sait pas honorer.

        `mode` etait STOCKE et lu NULLE PART : `_init_backend` ne
        dispatche que sur `backend_name`, et rend le meme `AerSimulator`
        pour `mode='simulator'` et pour `mode='hardware'`. Aucun chemin de
        ce depot ne resout un backend IBM reel.

        Le mode materiel ne levait donc pas : `execute` ouvrait
        `Session(backend=AerSimulator)` — que qiskit-ibm-runtime ACCEPTE —
        puis y construisait un estimateur avec decouplage dynamique et
        twirling. Un run demande en `hardware` s'executait sur un
        simulateur, avec des options qui n'y veulent rien dire, et rendait
        des nombres parfaitement plausibles sans jamais le signaler.

        Mesure : `VQARuntime(backend_name=b, mode='hardware')._backend`
        rend `AerSimulator` pour state_vector / matrix_product_state / aer
        et `FakeFez` pour estimator — identique a `mode='simulator'` dans
        les quatre cas. Voir D-48.
        """
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"mode={self.mode!r} non supporte : aucun backend materiel "
                f"n'est cable dans ce depot, et `_init_backend` rend un "
                f"simulateur quel que soit le mode. Un run demande en "
                f"'{self.mode}' tournerait sur simulateur sans le dire. "
                f"Attendu l'un de {list(self.SUPPORTED_MODES)}.")

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
        else:
            # Sans ce refus, un nom inconnu laissait _backend, _estimator et
            # _sampler a None et le constructeur rendait la main sans erreur.
            # La panne ne surgissait que bien plus loin, dans `execute`, sous
            # la forme d'un AttributeError sur NoneType — a des dizaines de
            # lignes de sa cause. `execute` et `optimize` levent tous deux
            # ValueError pour la meme valeur ; les trois sites doivent dire
            # la meme chose.
            raise ValueError(
                f"Unsupported backend: {self.backend_name!r}. Attendu l'un de "
                "'state_vector', 'matrix_product_state', 'aer', 'estimator'."
            )

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
        """Return a cached QAOAAnsatz for this topology, or build + cache it.

        La cle inclut une empreinte des COEFFICIENTS, pas seulement la
        topologie. L'ansatz QAOA encode `exp(-i gamma H)` : il depend de
        l'Hamiltonien terme par terme, pas seulement du nombre de qubits.

        La cle precedente `(num_qubits, period_bound, reps)` faisait
        collisionner deux patchs de meme taille aux coefficients differents :
        le second recevait l'ansatz construit pour le PREMIER, et se voyait
        donc optimise contre la physique d'un autre patch. Aucun appelant
        n'utilise cette methode aujourd'hui — c'etait un piege arme, pret a
        se declencher au premier branchement.
        """
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
