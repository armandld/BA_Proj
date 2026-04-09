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

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit import qpy

from ucrYZ_decomp import ucrz_decomp, ucry_decomp

# ---------------------------------------------------------
# Preparation of |c>'s matrix Parametrized
# ---------------------------------------------------------
# =========================================================
# State Preparation (Version Paramétrable)
# =========================================================

def prep_state(num_qubits, alpha_y_layers, alpha_z_layers, alpha, conjugate=False):
    """
    Constructs the ZGR state preparation circuit layer by layer.
    Uses manual decomposition to support VQA Parameters.
    """
    qc = QuantumCircuit(num_qubits, name="U_c" + ("_Conj" if conjugate else ""))
    
    sign = -1.0 if conjugate else 1.0
    
    # 1. Cascade Y
    for i in range(num_qubits):
        angles = alpha_y_layers[i]
        
        # Contrôles: i+1 à n-1. Cible: i
        controls = list(range(i + 1, num_qubits))
        target = i
        
        # REMPLACEMENT ICI : On utilise ucry_decomp au lieu de UCRYGate
        ucry_decomp(qc, angles, controls, target)

    # 2. Cascade Z
    for i in range(num_qubits):
        raw_angles = alpha_z_layers[i]
        
        prep_factor = -1.0
        final_sign = prep_factor * sign
        
        # Multiplication symbolique
        adjusted_angles = [a * final_sign for a in raw_angles]

        controls = list(range(i + 1, num_qubits))
        target = i
        
        # REMPLACEMENT ICI : On utilise ucrz_decomp au lieu de UCRZGate
        ucrz_decomp(qc, adjusted_angles, controls, target)

    # 3. Phase Globale
    qc.global_phase = alpha * sign
    return qc.to_gate()

def prep_state_for_vqa(m_qubits):
    """
    Prépare le circuit paramétré (squelette) et les vecteurs de paramètres.
    """
    n_params = 2**m_qubits - 1
    theta = ParameterVector('θ', 2*n_params+1)

    # Restructuration en couches pour le constructeur
    y_layers_params = []
    z_layers_params = []
    
    current_idx = 0
    # Ordre i=0 (feuilles) à i=m-1 (racine)
    for i in range(m_qubits):
        n_controls = m_qubits - 1 - i
        n_angles = 2**n_controls
        
        layer_y = theta[0:n_params][current_idx : current_idx + n_angles]
        layer_z = theta[n_params:2*n_params][current_idx : current_idx + n_angles]
        
        y_layers_params.append(layer_y)
        z_layers_params.append(layer_z)
        current_idx += n_angles

    # Construction du circuit
    gate_c = prep_state(m_qubits, y_layers_params, z_layers_params, theta[-1], conjugate=False)
    gate_c_conj = prep_state(m_qubits, y_layers_params, z_layers_params, theta[-1], conjugate=True)
    
    return gate_c, gate_c_conj, theta

# -----------------------------
# Main entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Map MaxCut problem to Hamiltonian operator")
    parser.add_argument("--Mqbits", type=int, required=True, help="Number of qubits used for Fourier coeffs.")
    

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    
    # Gate |c>
    gate_c, gate_c_conj, theta_y_vec, theta_z_vec, alpha_vec = prep_state_for_vqa(args.Mqbits)
    

    # Construct the full register circuit
    # Assuming we want |c> on first m qubits and |c*> on second m qubits (common in these protocols)
    # Or just applying |c> as requested. Let's make a generic one.
    
    full_qc = QuantumCircuit(args.Mqbits)
    
    # Apply |c> on first register
    full_qc.append(gate_c, range(args.Mqbits))
    full_qc.append(gate_c_conj, range(args.Mqbits))
    
    # 4. Save Outputs
    full_qc.draw("mpl")

    plt.show()
    
"""
    # B. Serialize Circuit (QPY)
    qpy_file = os.path.join(args.out_dir, "zgr_circuit.qpy")
    with open(qpy_file, "wb") as f:
        qpy.dump(full_qc, f)
    print(f"Circuit serialized to {qpy_file}")
    
    # C. Visualize (Optional, for debug)
    print(full_qc.draw(fold=-1))
    
"""

if __name__ == "__main__":
    main()
