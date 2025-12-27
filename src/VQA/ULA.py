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
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import qpy


from itertools import product
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

def add_so4_block(qc, q_ctrl, q_target, params):
    """
    Implements the SO(4) block defined in Fig 12.
    It takes 6 parameters.
    Structure: Ry-Ry -> CNOT -> Ry-Ry -> CNOT -> Ry-Ry
    """
    # Layer 1: Ry on both
    qc.ry(params[0], q_ctrl)
    qc.ry(params[1], q_target)
    
    # Entanglement 1
    qc.cx(q_ctrl, q_target)
    
    # Layer 2: Ry on both
    qc.ry(params[2], q_ctrl)
    qc.ry(params[3], q_target)
    
    # Entanglement 2
    qc.cx(q_ctrl, q_target)
    
    # Layer 3: Ry on both
    qc.ry(params[4], q_ctrl)
    qc.ry(params[5], q_target)

def ULA_gate(num_qubits, depth):
    """
    Constructs the Universal Linear Ansatz (ULA) with parametrized depth.
    
    Args:
        num_qubits (int): Total number of qubits (n).
        depth (int): Number of full brick-wall layers (d).
        
    Returns:
        QuantumCircuit: The parameterized circuit.
    """
    qr = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(qr, name=f"ULA_n{num_qubits}_d{depth}")
    
    # Calculate total parameters: 6 per block * (n-1) blocks per depth * d depths
    # Explanation: In one depth unit, we cover all adjacent pairs (0,1), (1,2), (2,3)...
    # There are n-1 such pairs in total.
    total_params = 6 * (num_qubits - 1) * depth
    theta = ParameterVector('ε', total_params)
    
    param_idx = 0
    
    for d in range(depth):
        # --- Sub-layer A (Odd pairs: 0-1, 2-3, ...) ---
        # Corresponds to the first column of blocks in the diagram
        for i in range(0, num_qubits - 1, 2):
            params = theta[param_idx : param_idx + 6]
            add_so4_block(qc, qr[i], qr[i+1], params)
            param_idx += 6
            
       
        
        # --- Sub-layer B (Even pairs: 1-2, 3-4, ...) ---
        # Corresponds to the second column of blocks in the diagram
        for i in range(1, num_qubits - 1, 2):
            params = theta[param_idx : param_idx + 6]
            add_so4_block(qc, qr[i], qr[i+1], params)
            param_idx += 6
            
    return qc.to_gate(), theta

# -----------------------------
# Main entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Map MaxCut problem to Hamiltonian operator")
    parser.add_argument("--numqbits", type=int, required=True, help="Number of qubits used.")
    parser.add_argument("--depth", type=int, required=True, help="Depth of the ULA ansatz.")

    args = parser.parse_args()

    # Number of qubits you want
    num_qbits = args.numqbits

    depth = args.depth
    ula_circuit, theta_params = ULA_gate(num_qbits, depth)
    q = QuantumCircuit(num_qbits)
    q.append(ula_circuit, range(num_qbits))
    q.draw("mpl", fold=False, idle_wires=False)
    plt.show()

if __name__ == "__main__":
    main()
