"""Reproducibility contract for transpilation and finite-shot QAOA."""

import ast
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit

from VQA.runtime import VQARuntime


ROOT = Path(__file__).resolve().parents[2]


def _measured_counts(seed, shots=512):
    runtime = VQARuntime("aer", "simulator", shots, 1, seed=seed)
    circuit = QuantumCircuit(3)
    circuit.ry(0.7, 0)
    circuit.ry(1.4, 1)
    circuit.ry(2.2, 2)
    circuit = runtime.transpile(circuit)
    circuit.measure_all()
    return runtime.sampler.run(
        [(circuit,)]).result()[0].data.meas.get_counts()


def test_the_same_seed_reproduces_finite_shot_counts_exactly():
    assert _measured_counts(23) == _measured_counts(23)


def test_a_different_seed_changes_the_finite_shot_realisation():
    assert _measured_counts(23) != _measured_counts(24)


def test_runtime_seeds_backend_estimator_sampler_and_transpiler(monkeypatch):
    runtime = VQARuntime("aer", "simulator", 128, 2, seed=41)
    assert runtime.seed == 41
    assert runtime._backend.options.seed_simulator == 41
    assert runtime.estimator.options.simulator.seed_simulator == 41
    assert runtime.sampler.options.simulator.seed_simulator == 41

    seen = {}

    class PassManager:
        def run(self, circuit):
            return circuit

    def fake_pass_manager(**kwargs):
        seen.update(kwargs)
        return PassManager()

    import qiskit.transpiler.preset_passmanagers as preset
    monkeypatch.setattr(preset, "generate_preset_pass_manager",
                        fake_pass_manager)
    runtime.transpile(QuantumCircuit(1))
    assert seen["seed_transpiler"] == 41


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_out_of_range_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        VQARuntime("state_vector", "simulator", 64, 0, seed=seed)


@pytest.mark.parametrize("seed", [1.5, "7", True])
def test_non_integer_seeds_are_rejected(seed):
    with pytest.raises(TypeError, match="seed"):
        VQARuntime("state_vector", "simulator", 64, 0, seed=seed)


def test_pipeline_exposes_a_reproducibility_seed():
    tree = ast.parse((ROOT / "src" / "pipeline.py").read_text())
    seed_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--seed"
    ]
    assert len(seed_calls) == 1


def test_study_result_names_and_payload_separate_quantum_seeds(tmp_path):
    from study.common.qaoa_inputs import save_results

    row = {
        "snap_idx": 0,
        "marginals": np.array([0.4, 0.6]),
        "qaoa_h": np.array([[False]]),
        "qaoa_v": np.array([[True]]),
        "wall_time": 0.1,
        "seed": 9,
        "comparison": {
            "qaoa": {"f1": 0.5},
            "exact": {"f1": 0.6},
            "classical": {"f1": 0.4},
            "qaoa_exact_agreement": 0.5,
            "qaoa_exact_agreement_raw": 0.5,
            "exact_ground_degeneracy": 1,
        },
    }
    meta = {
        "scenario": "case", "Re": 400, "N": 32, "n_patches": 1,
        "reps": 1, "K_opt": 2, "backend": "state_vector", "seed": 9,
        "suffix": "_v2",
    }
    path = save_results([row], meta, outdir=tmp_path)
    assert path.endswith("_state_vector_seed9_v2.npz")
    with np.load(path) as artifact:
        assert int(artifact["seed"]) == 9
        np.testing.assert_array_equal(artifact["seeds"], [9])
