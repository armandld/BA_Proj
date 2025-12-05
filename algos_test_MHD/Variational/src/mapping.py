# scripts/mapping.py

import argparse
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import log2
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

from init_optimization_param import extract_initial_params_from_state, calcul_alpha_y, calcul_alpha_z, get_global_phase

# -----------------------------
# Main entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Mapping VQA")
    parser.add_argument("--out-dir", default="../data", help="Output directory for mapping")
    parser.add_argument("--in-dir", default="../input/mapping_input.json", help="Input directory for mapping")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--numqbits", type=int, required=True, help="Number of qubits used.")
    parser.add_argument("--depth", type=int, required=False, help="Depth of the ULA ansatz.")
    

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Number of qubits you want
    num_qbits = args.numqbits
    depth = args.depth if args.depth else 2

    # Export JSON
    mapping_data = {
        "num_qubits": num_qbits,
        "depth": depth,
        "parameter_vectors": {},
        "circuits": [],
        "transpiled_circuits": []
    }

    gates_store = {
        "theta": {"gate": None, "params": None},
        "phi":   {"gate": None, "params": None},
        "eps":   {"gate": None, "params": None}
    }

    with open(args.in_dir, "r") as f:
        data = json.load(f)
    
    for entry in data:
        role = entry.get("role", "unknown") # 'theta', 'phi', ou 'eps'
        prep_method = entry.get("prep", "ULA")
        raw_data = entry.get("vector", [])
        # On reconstruit les complexes : Re + j*Im
        state_vector = np.array([x[0] + 1j * x[1] for x in raw_data])

        # Calcul de m_qubits basé sur le vecteur si présent, sinon défaut
        if ((state_vector is not None and len(state_vector) > 0)):
            m_qubits = int(log2(len(state_vector)))
        elif( prep_method != "ULA" ):
            raise ValueError(f"State vector is required for role '{role}' with prep method '{prep_method}'")

        # Variables temporaires pour stocker le résultat de cette itération
        current_gate = None
        current_params = None
        init_values = []

        print(f"Processing role: {role} with method: {prep_method}")

        # --- CAS 1 : Schmidt ---
        if prep_method == "ZGR_Schmidt":
            # 1. Extraction params initiaux & Matrices
            # init_vals_A est une liste de floats
            init_vals_A, U_Mat, V_Mat = extract_initial_params_from_state(state_vector)

            # 2. Construction Gate
            current_gate, current_params = ZGR_QFT_gate(
                m_qubits, 
                num_qubits=args.numqbits, 
                U_matrix=U_Mat, 
                V_matrix=V_Mat, 
                rot=False
            )
            # 3. Stockage valeurs initiales
            init_values = list(init_vals_A)

        # --- CAS 2 : ZGR Rotations ---
        elif prep_method == "ZGR_rot":
            # 1. Extraction params initiaux
            alpha_y = calcul_alpha_y(state_vector) # Retourne liste floats
            alpha_z = calcul_alpha_z(state_vector) # Retourne liste floats
            g_phase = get_global_phase(state_vector) # Retourne float
            
            # 2. Construction Gate
            current_gate, current_params = ZGR_QFT_gate(
                m_qubits, 
                num_qubits=args.numqbits, 
                rot=True
            )
            
            # 3. Aplatir les listes de listes pour correspondre au ParameterVector
            # Assurez-vous que l'ordre ici matche exactement l'ordre de création dans ZGR_QFT_gate
            # (Généralement theta_y, puis theta_z, puis alpha)
            flat_y = [item for sublist in alpha_y for item in sublist]
            flat_z = [item for sublist in alpha_z for item in sublist]
            init_values = flat_y + flat_z + [g_phase]

        # --- CAS 3 : ULA (Ansatz) ---
        elif prep_method == "ULA":
            # 1. Construction Gate (Abstraite)
            current_gate, current_params = ULA_gate(args.numqbits, args.depth)
            
            # Sinon valeurs aléatoires ou zéros
            init_values = list(np.random.uniform(0, 2*np.pi, len(current_params)))

        # --- MISE À JOUR DU STOCKAGE ---
        if role in gates_store:
            gates_store[role]["gate"] = current_gate
            gates_store[role]["params"] = current_params
            
            # Mise à jour du JSON mapping data
            mapping_data["parameter_vectors"][role] = {
                "name": current_params[0].name[:-3] if len(current_params) > 0 else "none",
                "length": len(current_params),
                "params": [p.name for p in current_params],
                "init_values": init_values # Liste de floats prête pour JSON
            }
        else:
            print(f"Warning: Role '{role}' unknown, skipping storage.")


    
    # Création de plusieurs circuits
    circuit_list = []

    U_theta = gates_store["theta"]["gate"]
    U_phi   = gates_store["phi"]["gate"]
    U_eps   = gates_store["eps"]["gate"]

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
