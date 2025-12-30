# scripts/mapping.py

import argparse
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

from networkx.readwrite import json_graph
from scipy.stats import unitary_group

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import UnitaryGate, AndGate, PauliEvolutionGate
from qiskit.circuit import QuantumCircuit, ControlledGate, ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import qpy
from qiskit.circuit.library import QAOAAnsatz



from itertools import product
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, RGate
from qiskit.quantum_info import SparsePauliOp

from VQA.init_qbits_state import init_qbits_state

def create_bounded_hamiltonian(hamilt_params, dim) -> SparsePauliOp:
    """
    Construit l'Hamiltonien MHD sur une grille torique (Périodique).
    Utilise SparsePauliOp pour la performance et corrige la topologie des plaquettes/vertex.
    """
    sparse_list = []
    
    # Helpers pour récupérer l'index linéaire du qubit correspondant à un lien
    # Qubits 0 à N^2-1 : Liens Horizontaux (H)
    # Qubits N^2 à 2N^2-1 : Liens Verticaux (V)
    offset_v = dim * dim
    
    def idx_H(y, x): return (y % dim) * dim + (x % dim)
    def idx_V(y, x): return offset_v + (y % dim) * dim + (x % dim)

    for i in range(dim):
        for j in range(dim):
            
            # --- 1. SHEAR (Viscosité) : Interactions ZZ ---
            # Horizontal Shear : Entre lien H(i,j) et H(i, j+1) (voisins sur la même ligne)
            c_h = hamilt_params['C_edges'][0][i, j]
            if abs(c_h) > 1e-6:
                sparse_list.append(("ZZ", [idx_H(i, j), idx_H(i, j+1)], c_h))

            # Vertical Shear : Entre lien V(i,j) et V(i+1, j) (voisins sur la même colonne)
            c_v = hamilt_params['C_edges'][1][i, j]
            if abs(c_v) > 1e-6:
                sparse_list.append(("ZZ", [idx_V(i, j), idx_V(i+1, j)], c_v))
    
            # --- 2. VORTICITY (Plaquette) : Terme ZZZZ ---
            # Une plaquette fermée implique : Haut -> Droite -> Bas -> Gauche
            k_val = hamilt_params['K_plaquettes'][i, j]
            if abs(k_val) > 1e-6:
                qubits_plaquette = [
                    idx_H(i, j),      # Haut (Lien H sur ligne i)
                    idx_V(i, j+1),    # Droite (Lien V sur colonne j+1)
                    idx_H(i+1, j),    # Bas (Lien H sur ligne i+1)
                    idx_V(i, j)       # Gauche (Lien V sur colonne j)
                ]
                sparse_list.append(("ZZZZ", qubits_plaquette, k_val))

            # --- 3. SHOCK (Divergence/Vertex) : Terme ZZZZ (Séparé !) ---
            # Un noeud implique les 4 liens qui forment une croix (+) autour de lui.
            # Entrant/Sortant pour tester la divergence div(B)=0
            delta_val = hamilt_params['Delta_nodes'][i, j]
            if abs(delta_val) > 1e-6:
                qubits_vertex = [
                    idx_H(i, j),      # Sortant Droite
                    idx_H(i, j-1),    # Entrant Gauche (j-1)
                    idx_V(i, j),      # Sortant Bas
                    idx_V(i-1, j)     # Entrant Haut (i-1)
                ]
                sparse_list.append(("ZZZZ", qubits_vertex, delta_val))

            # --- 4. KINK (Chiralité) : Termes XY - YX ---
            # Horizontal Kink (le long de la ligne)
            d_h = hamilt_params['D_edges'][0][i, j]
            if abs(d_h) > 1e-6:
                # Interaction entre H(i,j) et son voisin H(i,j+1)
                q1, q2 = idx_H(i, j), idx_H(i, j+1)
                sparse_list.append(("XY", [q1, q2], d_h))
                sparse_list.append(("YX", [q1, q2], -d_h))

            # Vertical Kink (le long de la colonne)
            d_v = hamilt_params['D_edges'][1][i, j]
            if abs(d_v) > 1e-6:
                # Interaction entre V(i,j) et son voisin V(i+1,j)
                q1, q2 = idx_V(i, j), idx_V(i+1, j)
                sparse_list.append(("XY", [q1, q2], d_v))
                sparse_list.append(("YX", [q1, q2], -d_v))

            # --- 5. CONTROL (Champ Magnétique) ---
            m_val = hamilt_params['M_nodes'][i, j]
            if abs(m_val) > 1e-6:
                # On applique le champ sur les liens sortants du noeud
                sparse_list.append(("X", [idx_H(i, j)], m_val))
                sparse_list.append(("X", [idx_V(i, j)], m_val))
    

def create_NO_bounded_hamiltonian(hamilt_params, dim) -> SparsePauliOp:
    """
    Construit l'Hamiltonien MHD sur une grille torique (Périodique).
    Utilise SparsePauliOp pour la performance et corrige la topologie des plaquettes/vertex.
    """
    sparse_list = []
    
    # Helpers pour récupérer l'index linéaire du qubit correspondant à un lien
    # Qubits 0 à N^2-1 : Liens Horizontaux (H)
    # Qubits N^2 à 2N^2-1 : Liens Verticaux (V)
    offset_v = dim * dim
    
    def idx_H(y, x): return (y % dim) * dim + (x % dim)
    def idx_V(y, x): return offset_v + (y % dim) * dim + (x % dim)

    for i in range(dim):
        for j in range(dim):
            
            # --- 1. SHEAR (Viscosité) : Interactions ZZ ---
            # Horizontal Shear : Entre lien H(i,j) et H(i, j+1) (voisins sur la même ligne)
            c_h = hamilt_params['C_edges'][0][i, j]
            if abs(c_h) > 1e-6:
                sparse_list.append(("ZZ", [idx_H(i, j), idx_H(i, j+1)], c_h))

            # Vertical Shear : Entre lien V(i,j) et V(i+1, j) (voisins sur la même colonne)
            c_v = hamilt_params['C_edges'][1][i, j]
            if abs(c_v) > 1e-6:
                sparse_list.append(("ZZ", [idx_V(i, j), idx_V(i+1, j)], c_v))
    
            # --- 2. VORTICITY (Plaquette) : Terme ZZZZ ---
            # Une plaquette fermée implique : Haut -> Droite -> Bas -> Gauche
            k_val = hamilt_params['K_plaquettes'][i, j]
            if abs(k_val) > 1e-6:
                qubits_plaquette = [
                    idx_H(i, j),      # Haut (Lien H sur ligne i)
                    idx_V(i, j+1),    # Droite (Lien V sur colonne j+1)
                    idx_H(i+1, j),    # Bas (Lien H sur ligne i+1)
                    idx_V(i, j)       # Gauche (Lien V sur colonne j)
                ]
                sparse_list.append(("ZZZZ", qubits_plaquette, k_val))

            # --- 3. SHOCK (Divergence/Vertex) : Terme ZZZZ (Séparé !) ---
            # Un noeud implique les 4 liens qui forment une croix (+) autour de lui.
            # Entrant/Sortant pour tester la divergence div(B)=0
            delta_val = hamilt_params['Delta_nodes'][i, j]
            if abs(delta_val) > 1e-6:
                qubits_vertex = [
                    idx_H(i, j),      # Sortant Droite
                    idx_H(i, j-1),    # Entrant Gauche (j-1)
                    idx_V(i, j),      # Sortant Bas
                    idx_V(i-1, j)     # Entrant Haut (i-1)
                ]
                sparse_list.append(("ZZZZ", qubits_vertex, delta_val))

            # --- 4. KINK (Chiralité) : Termes XY - YX ---
            # Horizontal Kink (le long de la ligne)
            d_h = hamilt_params['D_edges'][0][i, j]
            if abs(d_h) > 1e-6:
                # Interaction entre H(i,j) et son voisin H(i,j+1)
                q1, q2 = idx_H(i, j), idx_H(i, j+1)
                sparse_list.append(("XY", [q1, q2], d_h))
                sparse_list.append(("YX", [q1, q2], -d_h))

            # Vertical Kink (le long de la colonne)
            d_v = hamilt_params['D_edges'][1][i, j]
            if abs(d_v) > 1e-6:
                # Interaction entre V(i,j) et son voisin V(i+1,j)
                q1, q2 = idx_V(i, j), idx_V(i+1, j)
                sparse_list.append(("XY", [q1, q2], d_v))
                sparse_list.append(("YX", [q1, q2], -d_v))

            # --- 5. CONTROL (Champ Magnétique) ---
            m_val = hamilt_params['M_nodes'][i, j]
            if abs(m_val) > 1e-6:
                # On applique le champ sur les liens sortants du noeud
                sparse_list.append(("X", [idx_H(i, j)], m_val))
                sparse_list.append(("X", [idx_V(i, j)], m_val))

# -----------------------------
# Main entry point
# -----------------------------
def mapping(data_in, hamilt_params, bounded_cond):
    dim = len(data_in["theta_h"])
    theta_h = np.array(data_in.get("theta_h", []))
    theta_v = np.array(data_in.get("theta_v", []))
    psi_h   = np.array(data_in.get("psi_h",   []))
    psi_v   = np.array(data_in.get("psi_v",   []))

    qc = init_qbits_state( theta_h, theta_v, psi_h, psi_v)

    if bounded_cond:
        cost_hamiltonian = create_bounded_hamiltonian(hamilt_params, dim)
    else:
        cost_hamiltonian = create_NO_bounded_hamiltonian(hamilt_params, dim)

    """
    # -----------------------------
    # Prepare data for JSON
    # -----------------------------
    hamiltonian_terms = []
    for label, coeff in cost_hamiltonian.to_list():
        # Convert complex → float safely
        real_coeff = float(np.real_if_close(coeff))
        hamiltonian_terms.append([label, real_coeff])

    mapping_data = {
        "num_qubits": num_qubits,
        "edges": edges,
        "hamiltonian": hamiltonian_terms,
    }"""

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
    circuit.measure_all()
    
    circuit.draw("mpl")



if __name__ == "__main__":
    mapping(None, None, None)
