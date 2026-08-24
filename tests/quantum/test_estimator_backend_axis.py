"""Backend contract for the local QAOA pipeline."""

import ast
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from VQA.execute import execute
from VQA.optimize import optimize
from VQA.postprocess import postprocess
from VQA.runtime import VQARuntime


ROOT = Path(__file__).resolve().parents[2]
SUPPORTED = {"state_vector", "matrix_product_state", "aer"}


def _pipeline_backend_choices():
    tree = ast.parse((ROOT / "src" / "pipeline.py").read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--backend"):
            continue
        choices = next(
            keyword.value for keyword in node.keywords
            if keyword.arg == "choices"
        )
        return set(ast.literal_eval(choices))
    raise AssertionError("--backend is missing from src/pipeline.py")


def test_pipeline_only_advertises_backends_that_can_return_logical_marginals():
    assert _pipeline_backend_choices() == SUPPORTED


def test_runtime_and_transpiler_share_the_backend_contract():
    assert set(VQARuntime.SUPPORTED_BACKENDS) == SUPPORTED
    for backend in SUPPORTED:
        runtime = VQARuntime(backend, "simulator", shots=64, opt_level=1)
        assert type(runtime._backend).__name__ == "AerSimulator"
        optimize(QuantumCircuit(2), backend, opt_level=1, verbose=False)


@pytest.mark.parametrize("entrypoint", ["runtime", "optimize", "execute"])
def test_obsolete_estimator_backend_is_rejected(entrypoint):
    if entrypoint == "runtime":
        call = lambda: VQARuntime("estimator", "simulator", 64, 1)
    elif entrypoint == "optimize":
        call = lambda: optimize(QuantumCircuit(1), "estimator", 1, False)
    else:
        call = lambda: execute(
            QuantumCircuit(1), None, "simulator", "estimator",
            64, 1, 1, 1e-2, 1.0, False,
        )
    with pytest.raises(ValueError, match="Unsupported backend"):
        call()


def test_seeded_aer_sampling_agrees_with_exact_statevector():
    shots = 8192
    circuit = QuantumCircuit(3)
    circuit.ry(0.4, 0)
    circuit.ry(1.2, 1)
    circuit.ry(2.1, 2)
    exact = np.array(postprocess(
        Statevector.from_instruction(circuit).probabilities_dict(), 3, False))

    runtime = VQARuntime("aer", "simulator", shots, 1, seed=17)
    sampled_circuit = runtime.transpile(circuit)
    sampled_circuit.measure_all()
    counts = runtime.sampler.run(
        [(sampled_circuit,)]).result()[0].data.meas.get_counts()
    sampled = np.array(postprocess(
        {key: value / shots for key, value in counts.items()}, 3, False))

    np.testing.assert_allclose(sampled, exact, atol=2 / np.sqrt(shots))
