# scripts/H_TEST.py


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

"""
def pauli_basis(n, max_weight=None):
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
"""

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

    # Controlled U(ɛ)†
    U_eps_inv = U_eps.inverse()
    U_eps_inv.name = f"{U_eps.name}†" if U_eps.name else U_eps.name
    qc.append(U_eps_inv.control(1), [0] + list(range(1, num_qubits+1)))
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
    U_eps_inv = U_eps.inverse()
    U_eps_inv.name = f"{U_eps.name}†" if U_eps.name else U_eps.name
    qc.append(U_eps_inv.control(1), [0] + list(range(1, num_qubits+1)))
    qc.h(0)
    qc.measure(0, 0)

    return qc


# -----------------------------
# Main entry point
# -----------------------------
def main():
    pass

if __name__ == "__main__":
    main()
