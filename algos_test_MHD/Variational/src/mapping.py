# scripts/mapping.py

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

from ZGR_QFT import ZGR_QFT_gate
from ULA import ULA_gate

from H_TEST import H_TEST_DIM_1, H_TEST_DIM_2


# -----------------------------
# Main entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Hadamard Test VQA")
    parser.add_argument("--out-dir", default="../data", help="Output directory for mapping")
    parser.add_argument("--conf-dir", default="../config", help="Output directory for configuration parameters")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--numqbits", type=int, required=True, help="Number of qubits used.")
    parser.add_argument("--depth", type=int, required=False, help="Depth of the ULA ansatz.")
    parser.add_argument("--mqbits", type=int, required=False, help="Number of qubits used for ZGR_QFT.")
    

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Number of qubits you want
    num_qbits = args.numqbits
    m_qubits = args.mqbits if args.mqbits else num_qbits // 2
    depth = args.depth if args.depth else 2

    U_Matrix = np.identity(2**(m_qubits//2))
    V_Matrix = np.identity(2**(m_qubits - m_qubits//2))

    # Création de plusieurs circuits
    circuit_list = []

    U_theta, theta_params = ULA_gate(num_qbits, depth)
    U_phi, phi_params = ZGR_QFT_gate(m_qubits, num_qubits=num_qbits, U_matrix=U_Matrix, V_matrix=V_Matrix, rot=False)
    U_eps = U_theta.inverse()

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

    if( args.verbose ):
        qc1.draw("mpl", fold=False, idle_wires=False)
        qc2.draw("mpl", fold=False, idle_wires=False)
        plt.show()

    # Export JSON
    mapping_data = {
        "num_qubits": num_qbits,
        "m_qubits": m_qubits,
        "depth": depth,
        "parameter_vectors": {
                        "theta": {
                            "name": theta_params[0].name[:-3],    # retire [0], [1], ...
                            "length": len(theta_params),
                            "params": [p.name for p in theta_params]
                        },
                        "phi": {
                            "name": phi_params[0].name[:-3],
                            "length": len(phi_params),
                            "params": [p.name for p in phi_params]
                        }
                    },
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

    print("QC1 PARAMETERS : ", qc1.parameters)
    print("QC2 PARAMETERS : ", qc2.parameters)

    # qc1 est ton QuantumCircuit
    used_params = qc1.parameters  # set de paramètres réellement utilisés dans le circuit

    # paramètres de θ non utilisés
    unused_theta = [p for p in theta_params if p not in used_params]

    # paramètres de φ non utilisés
    unused_phi = [p for p in phi_params if p not in used_params]

    print("Theta non utilisés pour qc1:", unused_theta)
    print("Phi non utilisés pour qc1:", unused_phi)

    # qc1 est ton QuantumCircuit
    used_params = qc2.parameters  # set de paramètres réellement utilisés dans le circuit

    # paramètres de θ non utilisés
    unused_theta = [p for p in theta_params if p not in used_params]

    # paramètres de φ non utilisés
    unused_phi = [p for p in phi_params if p not in used_params]

    print("Theta non utilisés pour qc2:", unused_theta)
    print("Phi non utilisés pour qc2:", unused_phi)

if __name__ == "__main__":
    main()
