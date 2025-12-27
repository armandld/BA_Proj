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
from numpy.linalg import inv
from itertools import product

from networkx.readwrite import json_graph
from scipy.stats import unitary_group

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import HGate, UnitaryGate, AndGate, PauliEvolutionGate, UCRYGate, UCRZGate
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import qpy


from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

from prep_state_uniform_rot import prep_state_for_vqa as uniform_rot_prep
from prep_state_Schmidt import prep_state_for_vqa as Schmidt_prep

from Adder import get_adder_gate


def ZGR_QFT_gate(m_qubits, num_qubits, U_matrix=None, V_matrix=None, rot=False):
    
    # Gate |c>
    if(rot == True):
        gate_c, gate_c_conj, theta = uniform_rot_prep(m_qubits)
        params = theta
    else:
        if(U_matrix is None or V_matrix is None):
            raise ValueError("U_matrix and V_matrix must be provided for Schmidt preparation.")
        gate_c, gate_c_conj, params = Schmidt_prep(m_qubits, U_matrix, V_matrix)
    
    
    # Construction of the full ZGR QFT circuit

    full_qc = QuantumCircuit(num_qubits)
    
    # Construct the |c> gates
    full_qc.h(0)
    full_qc.x(0)
    full_qc.append(gate_c.control(1), [0] + list(range(num_qubits - m_qubits, num_qubits)))
    full_qc.x(0)
    full_qc.append(gate_c_conj.control(1), [0] + list(range(num_qubits - m_qubits, num_qubits)))
    

    gate, ver = get_adder_gate(m_qubits, num_qubits)
    

    # Construct the adder/decrementer gate
    qubit_mapping = [0]+[1] + list(range(num_qubits - m_qubits, num_qubits)) # Assuming the first qubit is the control

    if(ver):
        qubit_mapping += list(range(2, 1 + m_qubits))
    
    full_qc.append(gate.control(1), qubit_mapping)
    
    # Construct the CNOT and CCX gates
    full_qc.cx(0,1)

    for k in range(m_qubits):
        full_qc.cx(0, num_qubits - m_qubits + k)
    
    for k in range(m_qubits-1):
        full_qc.ccx(0, 1, 2 + k)

    for k in range(m_qubits-1):
        full_qc.x(num_qubits - m_qubits + k)

    full_qc.append(HGate().control(m_qubits), list(range(num_qubits - m_qubits, num_qubits))+[0])

    for k in range(m_qubits-1):
        full_qc.x(num_qubits - m_qubits + k)
    
    full_qc.append(QFT(num_qubits, inverse= True, do_swaps=True).to_gate(), range(num_qubits))

    return full_qc.to_gate(), params

# -----------------------------
# Main entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Map MaxCut problem to Hamiltonian operator")
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"], help="Quantum backend")
    parser.add_argument("--out-dir", default="data", help="Output directory for mapping")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--numqbits", type=int, required=True, help="Number of qubits used for the whole circuit.")
    parser.add_argument("--Mqbits", type=int, required=True, help="Number of qubits used for Fourier coeffs.")
    parser.add_argument("--rot", action="store_true", help="Use uniform rotation state preparation")
    

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Building ZGR QFT Quantum Circuit...")

    m_qubits = args.Mqbits
    num_qubits = args.numqbits
    rot = args.rot

    U_Matrix = np.identity(2**(m_qubits//2))
    V_Matrix = np.identity(2**(m_qubits - m_qubits//2))
    full_qc = ZGR_QFT_gate(m_qubits, num_qubits, U_Matrix, V_Matrix, rot)

    qc = QuantumCircuit(num_qubits)
    qc.append(full_qc[0], range(num_qubits))

    # 4. Save Outputs
    qc.draw("mpl")

    plt.show()

    """# B. Serialize Circuit (QPY)
    qpy_file = os.path.join(args.out_dir, "zgr_circuit.qpy")
    with open(qpy_file, "wb") as f:
        qpy.dump(full_qc, f)
    print(f"Circuit serialized to {qpy_file}")
    
    # C. Visualize (Optional, for debug)
    print(full_qc.draw(fold=-1))"""

if __name__ == "__main__":
    main()
