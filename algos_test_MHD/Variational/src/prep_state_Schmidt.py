

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
from qiskit.circuit.library import UnitaryGate, AndGate, PauliEvolutionGate, UCRYGate, UCRZGate
from qiskit.circuit import QuantumCircuit,QuantumRegister, ControlledGate, ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import qpy


from itertools import product
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from ucrYZ_decomp import ucrz_decomp, ucry_decomp

# ---------------------------------------------------------
# Preparation of |c>'s angles Parameters
# ---------------------------------------------------------

# =========================================================
# State Preparation (Version Paramétrable)
# =========================================================

def build_gate_A_parameterized(num_qubits_A, theta_vector):
    """
    Construit la Gate A en utilisant un ParameterVector Qiskit.
    Ceci permet l'optimisation variationnelle des coefficients de Schmidt.
    """
    qc = QuantumCircuit(num_qubits_A, name="Gate_A(θ)")
    
    param_idx = 0
    # Construction de l'arbre (MSB -> LSB)
    for layer in range(num_qubits_A):
        target = num_qubits_A - 1 - layer
        controls = list(range(num_qubits_A - layer, num_qubits_A))
        
        num_rots = 2**layer
        # On tranche le ParameterVector
        layer_params = theta_vector[param_idx : param_idx + num_rots]
        param_idx += num_rots
        
        if not controls:
            qc.ry(layer_params[0], target)
        else:
            # UCRY fonctionne avec des Paramètres Qiskit SI on les convertit en liste explicitement.
            # L'erreur "ParameterExpression is not iterable" arrive si on passe le slice direct sans list().
            ucry_decomp(qc, list(layer_params), controls, target)
    return qc

def append_Matrix_SVD(matrix, qc, n, qubits,label=None):
    if matrix is not None:
            # UnitaryGate gère la décomposition optimale lors de la transpilation
            if(matrix.shape != (2**n, 2**n)):
                raise ValueError(f"Matrix must be of shape {(2**n, 2**n)}")
            qc.append(UnitaryGate(matrix, label=label), qubits)

def append_Matrix_A(gate_A, qc, n_A, n_B, qubits_A, qubits_B):
    qc.append(gate_A, qubits_A)
    
    # --- 2. Cascade CNOT (Fixe : Intrication) ---
    min_dim = min(n_A, n_B)
    for i in range(min_dim):
        # CNOT copie l'état de base |k>_A vers |k>_B
        qc.cx(qubits_A[i], qubits_B[i])

def prep_state_for_vqa(n_total, U_matrix, V_matrix):
    """
    Construit le circuit paramétré complet incluant U et V.
    
    Args:
        n_total (int): Nombre total de qubits.
        U_matrix (np.ndarray): Matrice unitaire pour le système A (optionnel).
        V_matrix (np.ndarray): Matrice unitaire pour le système B (optionnel).
                             
    Returns:
        qc (QuantumCircuit): Le circuit quantique prêt pour l'optimisation.
        params_A (ParameterVector): Le vecteur des paramètres ajustables pour Gate A.
    """
    n_A = n_total // 2
    n_B = n_total - n_A
    
    qr = QuantumRegister(n_total, 'q')
    qc_c = QuantumCircuit(qr, name="Var_Schmidt")
    qc_c_conj = QuantumCircuit(qr, name="Var_Schmidt")
    
    qubits_A = qr[0:n_A]      # LSBs (Système A)
    qubits_B = qr[n_A:n_total] # MSBs (Système B)
    
    # --- 1. Gate A (Paramétrée : Coefficients de Schmidt) ---
    # Nombre de params = 2^n_A - 1
    num_params_A = 2**n_A - 1
    params_A = ParameterVector('θ_Schmidt', num_params_A)
    
    gate_A = build_gate_A_parameterized(n_A, params_A).to_gate()
    append_Matrix_A(gate_A, qc_c, n_A, n_B, qubits_A, qubits_B)
    
    # --- 3. Unitaires U et V (Fixes ou Identité) ---
    # On insère les matrices calculées par la SVD.
    # Elles sont statiques (non paramétrées par theta) car elles définissent 
    # la base propre de l'état cible. L'optimisation se fait sur les poids (Gate A).
    print("NA:", n_A, " NB:", n_B)
    append_Matrix_SVD(U_matrix, qc_c, n_A, qubits_A, label="U")

    append_Matrix_SVD(V_matrix, qc_c, n_B, qubits_B, label="V")

    append_Matrix_A(gate_A, qc_c_conj, n_A, n_B, qubits_A, qubits_B)

    append_Matrix_SVD(inv(U_matrix), qc_c_conj, n_A, qubits_A, label="U†")
        
    append_Matrix_SVD(inv(V_matrix), qc_c_conj, n_B, qubits_B, label="V†")
    
    return qc_c.to_gate(), qc_c_conj.to_gate(), params_A


# -----------------------------
# Main entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Map MaxCut problem to Hamiltonian operator")
    parser.add_argument("--Mqbits", type=int, required=True, help="Number of qubits used for Fourier coeffs.")
    parser.add_argument("--numqbits", type=int, required=True, help="Number of qubits used for the whole circuit.")


    args = parser.parse_args()

    print("Building Quantum Circuit...")

    # Gate |c>
    gate_c, params_A = prep_state_for_vqa(args.Mqbits,None, None)
    

    # Construct the full register circuit
    # Assuming we want |c> on first m qubits and |c*> on second m qubits (common in these protocols)
    # Or just applying |c> as requested. Let's make a generic one.
    
    full_qc = QuantumCircuit(args.numqbits)
    
    # Apply |c> on first register
    full_qc.append(gate_c, range(args.Mqbits))
    
    # 4. Save Outputs
    full_qc.draw("mpl")

    plt.show()
    

    # B. Serialize Circuit (QPY)
    qpy_file = os.path.join(args.out_dir, "zgr_circuit.qpy")
    with open(qpy_file, "wb") as f:
        qpy.dump(full_qc, f)
    print(f"Circuit serialized to {qpy_file}")
    
    # C. Visualize (Optional, for debug)
    print(full_qc.draw(fold=-1))

if __name__ == "__main__":
    main()
