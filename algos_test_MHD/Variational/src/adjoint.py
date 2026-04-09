# scripts/mapping.py

"""
Stage 1: Problem Mapping for MaxCut (QAOA pipeline)
Compatible with Qiskit 2.x (uses SparsePauliOp instead of qiskit.opflow)
"""

import argparse
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from networkx.readwrite import json_graph
from scipy.stats import unitary_group

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import UnitaryGate, AndGate, PauliEvolutionGate
from qiskit.circuit import QuantumCircuit, ControlledGate, ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import qpy


from itertools import product
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

def pauli_basis(n, max_weight=None):
    """Return n-qubit Pauli strings. If max_weight given, only include terms with weight <= max_weight (exclude 'I'*n)."""
    single = ["I", "X", "Y", "Z"]
    all_paulis = ["".join(p) for p in product(single, repeat=n)]
    # remove identity-only
    all_paulis = [p for p in all_paulis if p != "I"*n]
    if max_weight is None:
        return all_paulis
    def weight(s):
        return sum(1 for c in s if c != "I")
    return [p for p in all_paulis if weight(p) <= max_weight]

def parametrized_unitary_gate(num_qubits, prefix, max_weight=None):
    """
    Build a single Gate U(prefix) that implements U = prod_k exp(-i * P_k * theta_k)
    where each P_k is an n-qubit Pauli string, and theta_k is a ParameterVector element.
    Returns: (gate, params)
    - gate is a qiskit.circuit.Gate acting on num_qubits qubits and parametrized by params.
    - params is a ParameterVector of length len(paulis).
    """
    paulis = pauli_basis(num_qubits, max_weight=max_weight)
    params = ParameterVector(prefix, len(paulis))

    # create a small circuit on num_qubits that appends each PauliEvolutionGate(P_k, time=params[k])
    qc = QuantumCircuit(num_qubits, name=f"U_{prefix}")

    for p_str, param in zip(paulis, params):
        # H = P_k (coefficient 1.0). PauliEvolutionGate accepts SparsePauliOp and param as time.
        H = SparsePauliOp([p_str], [1.0])
        gate_term = PauliEvolutionGate(H, time=param)
        # append the parametric evolution acting on all num_qubits (SparsePauliOp encodes which qubits are I)
        qc.append(gate_term, list(range(num_qubits)))

    # convert to a Gate so it can be reused and controlled
    u_gate = qc.to_gate(label=f"U_{prefix}")
    return u_gate, params, paulis


# -----------------------------
# Main entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Map MaxCut problem to Hamiltonian operator")
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"], help="Quantum backend")
    parser.add_argument("--out-dir", default="results", help="Output directory for mapping")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--numqbits", type=int, required=True, help="Number of qubits used.")
    

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Number of qubits you want
    num_qbits = args.numqbits

    # Seed for reproducibility
    U_theta, theta_params, theta_paulis = parametrized_unitary_gate(num_qbits, "θ", max_weight=2)
    U_phi,   phi_params,   phi_paulis   = parametrized_unitary_gate(num_qbits, "φ", max_weight=2)
    U_eps,   eps_params,   eps_paulis   = parametrized_unitary_gate(num_qbits, "ε", max_weight=2)

    def H_TEST_DIM_1(num_qubits, U_theta, U_phi, U_eps):
        qc = QuantumCircuit(2*num_qubits + 1, 1)

        # qubit 0 = ancilla
        # [1 .. num_qubits] = first register
        # [num_qubits+1 .. 2*num_qubits] = second register

        qc.h(0)

        # Controlled U(θ)
        qc.append(U_theta.control(1), [0] + list(range(1, num_qubits+1)))

        # Controlled U(φ)
        qc.append(U_phi.control(1), [0] + list(range(num_qubits+1, 2*num_qubits+1)))

        # Ladder of CCX
        for k in range(num_qubits):
            qc.ccx(0, k+1, num_qubits+1+k)

        # Controlled U(θ)†
        qc.append(U_eps.inverse().control(1), [0] + list(range(1, num_qubits+1)))

        qc.h(0)
        qc.measure(0, 0)

        return qc
    
    def H_TEST_DIM_2(num_qubits, U_theta, U_phi_1, U_phi_2, U_eps):
        qc = QuantumCircuit(3*num_qubits + 1, 1)

        # qubit 0 = ancilla
        # [1 .. num_qubits] = first register
        # [num_qubits+1 .. 2*num_qubits] = second register

        qc.h(0)

        # Controlled U(θ)
        qc.append(U_theta.control(1), [0] + list(range(1, num_qubits+1)))

        # Controlled U(φ)
        qc.append(U_phi_1.control(1), [0] + list(range(num_qubits+1, 2*num_qubits+1)))

        qc.append(U_phi_2.control(1), [0] + list(range(2*num_qubits+1, 3*num_qubits+1)))

        # Ladder of CCX

        for k in range(num_qubits):
            qc.ccx(0, num_qubits+1+k, 2*num_qubits+1+k)

        for k in range(num_qubits):
            qc.ccx(0, k+1, num_qubits+1+k)

        # Controlled U(θ)†
        qc.append(U_eps.inverse().control(1), [0] + list(range(1, num_qubits+1)))

        qc.h(0)
        qc.measure(0, 0)

        return qc
    
    if args.verbose:
        plt.show()

    # Création de plusieurs circuits
    circuit_list = []

    # Circuit 1
    qc1 = H_TEST_DIM_1(num_qbits, U_theta, U_phi, U_eps)
    circuit_list.append({
        "name": "C1",
        "circuit": qc1,
        "measurement": [{"qubit": 0, "pauli": "Z", "coeff": 1.0}]
    })

    # Circuit 2 (partage certains paramètres)
    qc2 = H_TEST_DIM_2(num_qbits, U_theta, U_phi, U_phi, U_eps)
    circuit_list.append({
        "name": "C2",
        "circuit": qc2,
        "measurement": [{"qubit": 0, "pauli": "Z", "coeff": 0.5}]
    })

    # Export JSON
    mapping_data = {
        "backend": args.backend,
        "num_qubits": num_qbits,
        "parameters": [p.name for p in theta_params] +
                    [p.name for p in phi_params] +
                    [p.name for p in eps_params],
        "circuits": []
    }

    for c in circuit_list:
        circuit_file = os.path.join(args.out_dir, f"{c['name']}.qpy")
        with open(circuit_file, "wb") as f:
            qpy.dump(c["circuit"], f)

        mapping_data["circuits"].append({
            "name": c["name"],
            "file": f"{c['name']}.qpy",
            "measurement": c["measurement"]
        })

    # Sauvegarde JSON
    json_file = os.path.join(args.out_dir, "mapping.json")
    with open(json_file, "w") as f:
        json.dump(mapping_data, f, indent=2)


if __name__ == "__main__":
    main()
