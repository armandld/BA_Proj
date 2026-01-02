# scripts/mapping.py

import argparse
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

from qiskit.quantum_info import SparsePauliOp

def get_expected_Z(theta):
    """Calcule <Z> = cos(theta) pour un état Ry(theta)."""
    return np.cos(theta)

def create_bounded_hamiltonian(
        hamilt_params, dim, 
        theta_h_full, theta_v_full, 
        psi_h_full, psi_v_full,
        advanced_anomalies_enabled = False
    ):
    """
    Construit l'Hamiltonien MHD avec conditions aux limites ouvertes (Halo).
    Optimisé pour éviter la concaténation de chaînes répétitive.
    """
    sparse_list = []
    
    # --- A. Extraction Cœur vs Halo ---
    # Cœur = indices [1:-1, 1:-1]
    
    # 1. Angles du Cœur (pour le retour)
    core_slice = (slice(1, -1), slice(1, -1))
    core_theta_h = theta_h_full[core_slice]
    core_theta_v = theta_v_full[core_slice]
    core_psi_h   = psi_h_full[core_slice]
    core_psi_v   = psi_v_full[core_slice]

    # 2. Valeurs moyennes <Z> du Halo
    # Halo Haut (Ligne 0)
    z_halo_top    = get_expected_Z(theta_v_full[0, 1:-1])
    # Halo Bas (Ligne -1)
    z_halo_bottom = get_expected_Z(theta_v_full[-1, 1:-1])
    # Halo Gauche (Col 0)
    z_halo_left   = get_expected_Z(theta_h_full[1:-1, 0])
    # Halo Droite (Col -1)
    z_halo_right  = get_expected_Z(theta_h_full[1:-1, -1])

    # --- B. Helpers d'indices ---
    offset_v = dim * dim
    
    # Tables de lookup pour éviter les concaténations de string lentes
    # On sait qu'on aura au max 4 qubits (ZZZZ)
    PAULI_Z = ["", "Z", "ZZ", "ZZZ", "ZZZZ"] 

    def idx_H(y, x): 
        return y * dim + x if (0 <= y < dim and 0 <= x < dim) else -1

    def idx_V(y, x): 
        return offset_v + y * dim + x if (0 <= y < dim and 0 <= x < dim) else -1

    # --- C. Construction de l'Hamiltonien ---

    for i in range(dim):
        for j in range(dim):
            
            # -----------------------------
            # 1. SHEAR (Viscosité)
            # -----------------------------
            
            # --- Horizontal (H_i,j <-> H_i,j+1) ---
            c_h = hamilt_params['C_edges'][0][i, j]
            if abs(c_h) > 1e-6:
                q_curr = idx_H(i, j)
                q_next = idx_H(i, j+1)
                
                if q_next != -1:
                    sparse_list.append(("ZZ", [q_curr, q_next], c_h))
                else:
                    # Bord Droit
                    sparse_list.append(("Z", [q_curr], c_h * z_halo_right[i]))

            # --- Vertical (V_i,j <-> V_i+1,j) ---
            c_v = hamilt_params['C_edges'][1][i, j]
            if abs(c_v) > 1e-6:
                q_curr = idx_V(i, j)
                q_next = idx_V(i+1, j)
                
                if q_next != -1:
                    sparse_list.append(("ZZ", [q_curr, q_next], c_v))
                else:
                    # Bord Bas
                    sparse_list.append(("Z", [q_curr], c_v * z_halo_bottom[j]))

            # --- Bords Gauche et Haut (Champs manquants) ---
            if j == 0:
                # Bord Gauche
                val_halo = z_halo_left[i]
                c_left = hamilt_params['C_edges'][0][i, 0] 
                sparse_list.append(("Z", [idx_H(i, 0)], c_left * val_halo))
                
            if i == 0:
                # Bord Haut
                val_halo = z_halo_top[j]
                c_top = hamilt_params['C_edges'][1][0, j]
                sparse_list.append(("Z", [idx_V(0, j)], c_top * val_halo))

            # -----------------------------
            # 2. VORTICITY (Plaquette)
            # -----------------------------
            if hamilt_params['K_plaquettes'] is not None:
                k_val = hamilt_params['K_plaquettes'][i, j]
                if abs(k_val) > 1e-6:
                    # Liste des potentiels candidats
                    candidates = [
                        (idx_H(i, j),   1.0),              # Top (Toujours in)
                        (idx_V(i, j+1), z_halo_right[i]),  # Right (Peut être out)
                        (idx_H(i+1, j), z_halo_bottom[j]), # Bottom (Peut être out)
                        (idx_V(i, j),   1.0)               # Left (Toujours in)
                    ]
                    
                    active_qubits = []
                    effective_k = k_val
                    
                    # Filtrage optimisé
                    for q_idx, halo_val in candidates:
                        if q_idx != -1:
                            active_qubits.append(q_idx)
                        else:
                            # Si le qubit est hors limite (Halo), il devient un coeff
                            effective_k *= halo_val
                    
                    if active_qubits:
                        # Utilisation de la lookup table pour éviter "Z"*len
                        label = PAULI_Z[len(active_qubits)]
                        sparse_list.append((label, active_qubits, effective_k))

            if advanced_anomalies_enabled:
                # -----------------------------
                # 3. SHOCK (Vertex)
                # -----------------------------
                if hamilt_params['K_plaquettes'] is not None:
                    delta_val = hamilt_params['Delta_nodes'][i, j]
                    if abs(delta_val) > 1e-6:
                        candidates = [
                            (idx_H(i, j),   1.0),            # Out Right
                            (idx_H(i, j-1), z_halo_left[i]), # In Left (Peut être out)
                            (idx_V(i, j),   1.0),            # Out Bottom
                            (idx_V(i-1, j), z_halo_top[j])   # In Top (Peut être out)
                        ]
                        
                        active_qubits = []
                        effective_delta = delta_val
                        
                        for q_idx, halo_val in candidates:
                            if q_idx != -1:
                                active_qubits.append(q_idx)
                            else:
                                effective_delta *= halo_val
                        
                        if active_qubits:
                            label = PAULI_Z[len(active_qubits)]
                            sparse_list.append((label, active_qubits, effective_delta))

                # -----------------------------
                # 4. KINK (Chiralité)
                # -----------------------------
                # Ici on garde seulement les interactions internes complètes
                if hamilt_params['D_edges'] is not None:
                    d_h = hamilt_params['D_edges'][0][i, j]
                    q1, q2 = idx_H(i, j), idx_H(i, j+1)
                    if abs(d_h) > 1e-6 and q2 != -1:
                        sparse_list.append(("XY", [q1, q2], d_h))
                        sparse_list.append(("YX", [q1, q2], -d_h))
                    d_v = hamilt_params['D_edges'][1][i, j]
                    q1, q2 = idx_V(i, j), idx_V(i+1, j)
                    if abs(d_v) > 1e-6 and q2 != -1:
                        sparse_list.append(("XY", [q1, q2], d_v))
                        sparse_list.append(("YX", [q1, q2], -d_v))

    num_qubits = 2 * dim * dim
    
    # Retourne l'Opérateur ET les 4 tableaux d'angles du cœur
    return (
        SparsePauliOp.from_sparse_list(sparse_list, num_qubits=num_qubits), 
        core_theta_h, core_theta_v, core_psi_h, core_psi_v
    )



def create_period_hamiltonian(hamilt_params, dim, advanced_anomalies_enabled = False) -> SparsePauliOp:
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

            if advanced_anomalies_enabled:
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
                
    return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=2*dim*dim)