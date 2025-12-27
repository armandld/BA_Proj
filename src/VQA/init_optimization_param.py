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

from scipy.linalg import svd

from networkx.readwrite import json_graph
from scipy.stats import unitary_group


#---------------------------------------------------------
# Functions to extract ZGR QFT parameters from target state |c>
#---------------------------------------------------------

# ---------------------------------------------------------
# 1. Uniform Rotations 
# ---------------------------------------------------------

def calcul_alpha_y(state_vector):
    """
    Computes the Y-rotation angles (amplitudes).
    CORRECTED: Logic adapted to handle recursive list reduction.
    """
    probs = np.abs(state_vector)**2
    n = int(np.log2(len(state_vector)))
    alpha_y = []

    for k in range(1, n + 1):
        new_probs = []
        layer_angles = []
        
        # CORRECTION : On parcourt toujours par paires (0,1), (2,3)...
        # car la liste 'probs' réduit de taille à chaque étape.
        for j in range(0, len(probs), 2):
            sum_even = probs[j]     # Partie gauche (paire)
            sum_odd = probs[j+1]    # Partie droite (impaire)
            sum_total = sum_even + sum_odd
            
            if sum_total < 1e-15:
                angle = 0.0
            else:
                # Eq (8): 2 * arcsin(sqrt(P_odd / P_total))
                angle = 2 * np.arcsin(np.sqrt(sum_odd / sum_total))
            
            layer_angles.append(angle)
            new_probs.append(sum_total)
            
        alpha_y.append(layer_angles)
        probs = new_probs 

    return alpha_y

def calcul_alpha_z(state_vector):
    """
    Computes the Z-rotation angles (phases) based on Walsh-Hadamard transform.
    Corrected to handle the recursive list reduction properly.
    """
    phases = np.angle(state_vector)
    n = int(np.log2(len(state_vector)))
    alpha_z = []

    for k in range(1, n + 1):
        new_phases = []
        layer_angles = []
        
        # Since 'phases' shrinks by half each time, we simply compare 
        # adjacent elements (0 vs 1, 2 vs 3...) in the current list.
        # We step by 2 every time.
        for j in range(0, len(phases), 2):
            idx_left = j
            idx_right = j + 1
            
            diff = phases[idx_right] - phases[idx_left]
            
            # Wrap to [-pi, pi] to ensure we take the shortest path on the circle
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            
            layer_angles.append(diff)
            
            avg_phase = (phases[idx_left] + phases[idx_right]) / 2.0
            new_phases.append(avg_phase)

        alpha_z.append(layer_angles)
        phases = new_phases 

    return alpha_z

def get_global_phase(c_vector):
    phases = np.angle(c_vector)
    return np.mean(phases)

# ---------------------------------------------------------
# 2. Schmidt Decomposition Functions
# ---------------------------------------------------------

def get_bloch_angles(amplitude_0, amplitude_1):
    """ Calcule theta pour RY tel que RY|0> ~ a|0> + b|1> """
    norm = np.sqrt(np.abs(amplitude_0)**2 + np.abs(amplitude_1)**2)
    if norm < 1e-9: return 0.0
    return 2 * np.arccos(np.abs(amplitude_0) / norm)

def get_schmidt_angles_from_coeffs(coeffs):
    """
    Transforme les coefficients de Schmidt (valeurs singulières) 
    en une liste plate d'angles pour l'arbre de rotations RY.
    """
    # 1. Padding pour avoir une puissance de 2
    target_len = 2**int(np.ceil(np.log2(len(coeffs))))
    padded_coeffs = np.zeros(target_len)
    padded_coeffs[:len(coeffs)] = coeffs
    
    # 2. Récursion pour trouver les angles
    angles = []
    current_parts = [padded_coeffs]
    n_layers = int(np.log2(target_len))
    
    for layer in range(n_layers):
        next_parts = []
        for part in current_parts:
            mid = len(part) // 2
            left, right = part[:mid], part[mid:]
            norm_L, norm_R = np.linalg.norm(left), np.linalg.norm(right)
            angles.append(get_bloch_angles(norm_L, norm_R))
            next_parts.extend([left, right])
        current_parts = next_parts
    return angles

def extract_initial_params_from_state(state_vector):
    """
    Analyse un état cible |c> et retourne les valeurs initiales pour l'optimisation.
    
    Returns:
        initial_params_A (list[float]): Les angles pour la Gate A (Schmidt Coeffs).
        U_matrix (np.ndarray): La matrice unitaire exacte pour le système A.
        V_matrix (np.ndarray): La matrice unitaire exacte pour le système B.
    """
    state = np.array(state_vector)
    n_total = int(np.log2(len(state)))
    n_A = n_total // 2
    n_B = n_total - n_A
    
    # 1. Reshape et SVD
    dim_A = 2**n_A
    dim_B = 2**n_B
    # Transpose pour aligner avec la convention Qiskit (Little Endian / Tensor order)
    # Psi ~ sum lambda_i |i>_B (x) |i>_A  (selon reshape)
    psi_matrix = state.reshape((dim_B, dim_A)).T 
    
    U, S, Vh = svd(psi_matrix, full_matrices=True)
    
    # 2. Extraction des angles pour la Gate A
    initial_params_A = get_schmidt_angles_from_coeffs(S)
    
    # 3. Matrices Unitaires
    # U agit sur A. Vh agit sur B.
    # Dans la décomposition SVD : M = U . Sigma . Vh
    # Le circuit applique : (U (x) V_circuit) . State_Sigma
    # Donc V_circuit doit correspondre à Vh.T (transposée car Vh agit sur les bras droits)
    V_matrix = Vh.T
    
    return initial_params_A, U, V_matrix


# -----------------------------
# Main entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Map MaxCut problem to Hamiltonian operator")
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"], help="Quantum backend")
    parser.add_argument("--out-dir", default="results", help="Output directory for mapping")
    parser.add_argument("--json-file", required=True, help="Path to JSON file containing input vector 'c'")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load Vector
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    # Handle complex vector loading (assuming list of [re, im] or simple list if real)
    raw_vec = data['vector']
    if isinstance(raw_vec[0], list):
        c_vector = np.array([complex(x[0], x[1]) for x in raw_vec])
    else:
        c_vector = np.array([complex(x, 0) for x in raw_vec])

    # Normalize
    norm = np.linalg.norm(c_vector)
    if abs(norm - 1.0) > 1e-6:
        print(f"Warning: Vector not normalized (norm={norm}). Normalizing now.")
        c_vector = c_vector / norm

    # Check size
    if len(c_vector) != 2**args.Mqbits:
        raise ValueError(f"Vector length {len(c_vector)} does not match 2^m ({2**args.Mqbits})")

    # 2. Compute Parameters
    print("Computing ZGR parameters...")
    alpha_y_layers = calcul_alpha_y(c_vector)
    alpha_z_layers = calcul_alpha_z(c_vector)
    global_alpha = get_global_phase(c_vector)

    # Flatten parameters for JSON output (useful for VQA optimizers)
    flat_theta_y = [item for sublist in alpha_y_layers for item in sublist]
    flat_theta_z = [item for sublist in alpha_z_layers for item in sublist]
    
    # A. JSON Parameters
    output_params = {
        "m_qubits": args.Mqbits,
        "parameters": {
            "theta_y": flat_theta_y,
            "theta_z": flat_theta_z,
            "alpha": global_alpha
        },
        "structure_info": "theta lists are flattened layers (leaves to root). alpha is global phase."
    }
    
    param_file = os.path.join(args.out_dir, "zgr_parameters.json")
    with open(param_file, "w") as f:
        json.dump(output_params, f, indent=4)
    print(f"Parameters saved to {param_file}")


if __name__ == "__main__":
    main()
