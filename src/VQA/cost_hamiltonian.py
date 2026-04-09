# scripts/cost_hamiltonian.py

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

    Halo contraction — centered around the decision boundary:
    When a boundary ZZ term C·Z_i·Z_j has qubit j in the halo,
    the mean-field contraction replaces Z_j by its expectation ⟨Z_j⟩.
    We CENTER this around the threshold so the halo only contributes
    a bias when it clearly differs from the decision boundary:

        C · (cos(θ_halo) − cos(θ_threshold)) · Z_i

    where cos(θ_threshold) = 1 − 2·threshold_amr.

    This gives:
      - halo score > threshold → positive Z → push toward |1⟩ (refine)
      - halo score < threshold → negative Z → push toward |0⟩ (don't)
      - halo score = threshold → zero contribution (neutral)
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
    # Raw halo values for plaquette contractions (ZZZZ → ZZZ/ZZ scaling)
    z_halo_top_raw    = get_expected_Z(theta_v_full[0, 1:-1])
    z_halo_bottom_raw = get_expected_Z(theta_v_full[-1, 1:-1])
    z_halo_left_raw   = get_expected_Z(theta_h_full[1:-1, 0])
    z_halo_right_raw  = get_expected_Z(theta_h_full[1:-1, -1])

    # Centered + scaled halo values for shear ZZ → Z contractions.
    # Center around the decision boundary: cos(θ_thr) = 1 − 2·threshold_amr
    # Scale by w_z_frac so halo Z has the same weight as the designed
    # H_edges Z-bias relative to the multi-body coupling strength.
    threshold_amr = hamilt_params.get('threshold_amr', 0.0)
    w_z_frac = hamilt_params.get('w_z_frac', 1.0)
    z_threshold = 1.0 - 2.0 * threshold_amr

    z_halo_top    = w_z_frac * (z_halo_top_raw    - z_threshold)
    z_halo_bottom = w_z_frac * (z_halo_bottom_raw  - z_threshold)
    z_halo_left   = w_z_frac * (z_halo_left_raw    - z_threshold)
    z_halo_right  = w_z_frac * (z_halo_right_raw   - z_threshold)

    # --- B. Helpers d'indices ---
    offset_v = dim * dim
    num_qubits = 2 * dim * dim

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
            # 1. DATA VALIDITY
            # -----------------------------

            # --- Core offset: hamilt_params arrays are padded (dim+2, dim+2) ---
            # The core data is at [1:-1, 1:-1], so offset indices by +1
            ci, cj = i + 1, j + 1

            # --- Horizontal (H_i,j) ---
            h_h = hamilt_params['H_edges'][0][ci, cj]
            if abs(h_h) > 1e-6:
                sparse_list.append(("Z", [idx_H(i, j)], h_h))

            # --- Vertical (H_i,j) ---
            h_v = hamilt_params['H_edges'][1][ci, cj]
            if abs(h_v) > 1e-6:
                sparse_list.append(("Z", [idx_V(i, j)], h_v))

            # -----------------------------
            # 1. SHEAR (Viscosité)
            # -----------------------------

            # --- Horizontal (H_i,j <-> H_i,j+1) ---
            # Sign convention is in HamiltParams (ferromagnetic: C < 0)
            c_h = hamilt_params['C_edges'][0][ci, cj]
            if abs(c_h) > 1e-6:
                q_curr = idx_H(i, j)
                q_next = idx_H(i, j+1)

                if q_next != -1:
                    sparse_list.append(("ZZ", [q_curr, q_next], c_h))
                else:
                    # Bord Droit: centered halo contraction
                    sparse_list.append(("Z", [q_curr], c_h * z_halo_right[i]))

            # --- Vertical (V_i,j <-> V_i+1,j) ---
            c_v = hamilt_params['C_edges'][1][ci, cj]
            if abs(c_v) > 1e-6:
                q_curr = idx_V(i, j)
                q_next = idx_V(i+1, j)

                if q_next != -1:
                    sparse_list.append(("ZZ", [q_curr, q_next], c_v))
                else:
                    # Bord Bas: centered halo contraction
                    sparse_list.append(("Z", [q_curr], c_v * z_halo_bottom[j]))

            # --- Bords Gauche et Haut (Champs manquants) ---
            if j == 0:
                # Bord Gauche: centered halo contraction
                val_halo = z_halo_left[i]
                c_left = hamilt_params['C_edges'][0][ci, 1]
                sparse_list.append(("Z", [idx_H(i, 0)], c_left * val_halo))

            if i == 0:
                # Bord Haut: centered halo contraction
                val_halo = z_halo_top[j]
                c_top = hamilt_params['C_edges'][1][1, cj]
                sparse_list.append(("Z", [idx_V(0, j)], c_top * val_halo))

            # -----------------------------
            # 2. VORTICITY (Plaquette)
            # -----------------------------
            # Sign convention is in HamiltParams (even-parity: K < 0)
            # Plaquette contractions use RAW halo values (not centered/scaled)
            # because they produce multi-body terms (ZZZ, ZZ), not 1-body Z bias.
            if hamilt_params['K_plaquettes'] is not None:
                k_val = hamilt_params['K_plaquettes'][ci, cj]
                if abs(k_val) > 1e-6:
                    # Liste des potentiels candidats
                    candidates = [
                        (idx_H(i, j),   1.0),                  # Top (Toujours in)
                        (idx_V(i, j+1), z_halo_right_raw[i]),  # Right (Peut être out)
                        (idx_H(i+1, j), z_halo_bottom_raw[j]), # Bottom (Peut être out)
                        (idx_V(i, j),   1.0)                   # Left (Toujours in)
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
                # 3. X-POINT RECONNECTION (Plaquette ZZZZ)
                # -----------------------------
                # Same plaquette topology as K_plaquettes but with
                # X-point reconnection coefficient (det(J_B) < 0).
                # Sign convention is in HamiltParams (even-parity: K < 0)
                if hamilt_params.get('K_xpoint') is not None:
                    kx_val = hamilt_params['K_xpoint'][ci, cj]
                    if abs(kx_val) > 1e-6:
                        candidates_xp = [
                            (idx_H(i, j),   1.0),
                            (idx_V(i, j+1), z_halo_right_raw[i]),
                            (idx_H(i+1, j), z_halo_bottom_raw[j]),
                            (idx_V(i, j),   1.0)
                        ]
                        active_qubits_xp = []
                        effective_kx = kx_val
                        for q_idx, halo_val in candidates_xp:
                            if q_idx != -1:
                                active_qubits_xp.append(q_idx)
                            else:
                                effective_kx *= halo_val
                        if active_qubits_xp:
                            label = PAULI_Z[len(active_qubits_xp)]
                            sparse_list.append((label, active_qubits_xp, effective_kx))

    # Safety: avoid empty Hamiltonian (Qiskit crashes on "Empty observable")
    # Use 1e-3 (not 1e-6) to survive any internal simplify() calls
    if not sparse_list:
        sparse_list.append(("Z", [0], 1e-3))

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
            # --- 0. DATA VALIDITY : Interactions Z ---
            # Horizontal VALID : Entre lien H(i,j) et H(i, j+1) (voisins sur la même ligne)
            h_h = hamilt_params['H_edges'][0][i, j]
            if abs(h_h) > 1e-6:
                sparse_list.append(("Z", [idx_H(i, j)], h_h))

            # Vertical VALID : Entre lien V(i,j) et V(i+1, j) (voisins sur la même colonne)
            h_v = hamilt_params['H_edges'][1][i, j]
            if abs(h_v) > 1e-6:
                sparse_list.append(("Z", [idx_V(i, j)], h_v))
            
            # --- 1. SHEAR (Viscosité) : Interactions ZZ ---
            # Sign convention is in HamiltParams (ferromagnetic: C < 0)
            c_h = hamilt_params['C_edges'][0][i, j]
            if abs(c_h) > 1e-6:
                sparse_list.append(("ZZ", [idx_H(i, j), idx_H(i, j+1)], c_h))

            c_v = hamilt_params['C_edges'][1][i, j]
            if abs(c_v) > 1e-6:
                sparse_list.append(("ZZ", [idx_V(i, j), idx_V(i+1, j)], c_v))

            # --- 2. VORTICITY (Plaquette) : Terme ZZZZ ---
            # Sign convention is in HamiltParams (even-parity: K < 0)
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
                # --- 3. X-POINT RECONNECTION (Plaquette ZZZZ) ---
                # Same plaquette topology as K_plaquettes but with
                # X-point reconnection coefficient (det(J_B) < 0).
                if hamilt_params.get('K_xpoint') is not None:
                    kx_val = hamilt_params['K_xpoint'][i, j]
                    if abs(kx_val) > 1e-6:
                        qubits_xp = [
                            idx_H(i, j),
                            idx_V(i, j+1),
                            idx_H(i+1, j),
                            idx_V(i, j)
                        ]
                        sparse_list.append(("ZZZZ", qubits_xp, kx_val))
    # Safety: if all coefficients were below threshold, sparse_list is empty.
    # Qiskit's EstimatorV2 crashes on a zero Hamiltonian ("Empty observable").
    # Add a tiny identity-like term so the Hamiltonian is valid but has no
    # physical effect (COBYLA will converge immediately on a near-flat landscape).
    # Use 1e-3 (not 1e-6) to survive any internal simplify() calls.
    if not sparse_list:
        sparse_list.append(("Z", [0], 1e-3))

    return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=2*dim*dim)