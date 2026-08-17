# scripts/cost_hamiltonian.py

import argparse
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

from qiskit.quantum_info import SparsePauliOp

#: Coefficients strictement inférieurs à ce seuil ne sont pas encodés.
COEFF_MIN = 1e-6


class NullHamiltonianError(ValueError):
    """Aucun coefficient n'atteint COEFF_MIN : il n'y a pas d'Hamiltonien.

    Le patch ne pose aucun problème d'optimisation. C'est une information,
    pas une panne : l'appelant doit décider quoi en faire (typiquement,
    conserver la décision classique issue de l'initialisation θ). Ce qu'il
    ne faut pas faire, c'est renvoyer un opérateur de remplissage — il
    serait indiscernable d'un Hamiltonien réel en aval.
    """

    def __init__(self, num_qubits, threshold=COEFF_MIN):
        self.num_qubits = num_qubits
        self.threshold = threshold
        super().__init__(
            f"aucun coefficient >= {threshold:g} sur {num_qubits} qubits : "
            "le patch ne définit aucun Hamiltonien de coût"
        )


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

    # --- A0. Le coeur demande doit etre celui des tableaux fournis ---
    #
    # Toutes les lectures ci-dessous sont indexees par `dim` sur des tableaux
    # supposes (dim+2, dim+2). Un tableau TROP GRAND ne declenche aucune
    # erreur : la boucle lit simplement un sous-bloc du coin superieur gauche
    # et rend un Hamiltonien parfaitement valide, calcule sur la mauvaise
    # portion du patch. Rien en aval ne peut le detecter.
    #
    # Un tableau trop petit, lui, finit par lever un IndexError — mais
    # seulement au premier bord atteint, donc apres avoir deja encode des
    # termes. On refuse les deux cas d'emblee.
    expected = (dim + 2, dim + 2)
    _shapes = {
        'C_edges[0]': np.shape(hamilt_params['C_edges'][0]),
        'C_edges[1]': np.shape(hamilt_params['C_edges'][1]),
        'H_edges[0]': np.shape(hamilt_params['H_edges'][0]),
        'H_edges[1]': np.shape(hamilt_params['H_edges'][1]),
        'theta_h_full': np.shape(theta_h_full),
        'theta_v_full': np.shape(theta_v_full),
    }
    if hamilt_params.get('K_plaquettes') is not None:
        _shapes['K_plaquettes'] = np.shape(hamilt_params['K_plaquettes'])
    _bad = {k: v for k, v in _shapes.items() if tuple(v) != expected}
    if _bad:
        raise ValueError(
            f"create_bounded_hamiltonian(dim={dim}) attend des tableaux "
            f"{expected} (coeur dim x dim + halo d'epaisseur 1) ; recu "
            + ", ".join(f"{k}={tuple(v)}" for k, v in sorted(_bad.items()))
        )

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

    # Valeurs <Z> brutes des liens du halo que les PLAQUETTES contractent.
    #
    # Une plaquette a quatre membres : Haut = H(i,j), Droite = V(i,j+1),
    # Bas = H(i+1,j), Gauche = V(i,j). Sur la colonne de droite (j = dim-1)
    # le membre manquant est un lien V ; sur la ligne du bas (i = dim-1)
    # c'est un lien H. Le <Z> qui remplace un qubit manquant doit venir du
    # theta de CE lien : theta_v_full pour un lien V, theta_h_full pour un
    # lien H (init_qbits_state place theta_h sur les qubits idx_H et theta_v
    # sur les qubits idx_V).
    #
    # Le code lisait ici z_halo_right_raw (issu de theta_h_full) pour le
    # membre Droite, qui est un lien V, et z_halo_bottom_raw (issu de
    # theta_v_full) pour le membre Bas, qui est un lien H : les deux familles
    # etaient echangees. Les POSITIONS etaient bonnes, seul le tableau lu
    # etait le mauvais -- defaut present depuis le premier commit (cf93ba3).
    #
    # Mesure (D-113, docs/RESULTS.md) : en deploiement theta_h et theta_v
    # sont le MEME tableau (`refinement._prepare_vqa_input` passe `mini_score`
    # deux fois, `PhysToAngle.map_to_angles` le documente), donc l'echange
    # etait sans effet : 36 configurations aleatoires, operateur identique
    # bit a bit avant/apres. Sur theta_h != theta_v il change le signe du
    # terme (k = -0.5 rendu +0.5) et peut l'annuler entierement.
    z_plaq_right_raw  = get_expected_Z(theta_v_full[1:-1, -1])
    z_plaq_bottom_raw = get_expected_Z(theta_h_full[-1, 1:-1])

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
                    _halo_term = c_h * z_halo_right[i]
                    if abs(_halo_term) > COEFF_MIN:
                        sparse_list.append(("Z", [q_curr], _halo_term))

            # --- Vertical (V_i,j <-> V_i+1,j) ---
            c_v = hamilt_params['C_edges'][1][ci, cj]
            if abs(c_v) > 1e-6:
                q_curr = idx_V(i, j)
                q_next = idx_V(i+1, j)

                if q_next != -1:
                    sparse_list.append(("ZZ", [q_curr, q_next], c_v))
                else:
                    # Bord Bas: centered halo contraction
                    _halo_term = c_v * z_halo_bottom[j]
                    if abs(_halo_term) > COEFF_MIN:
                        sparse_list.append(("Z", [q_curr], _halo_term))

            # --- Bords Gauche et Haut (Champs manquants) ---
            #
            # `C_edges[0][a, b]` couple la cellule (a, b) a la cellule
            # (a, b+1) — c'est la convention des deux mappeurs, qui forment
            # leurs sauts par `champ - np.roll(champ, -1, axis=1)`.
            # L'arete qui relie le halo de gauche a la premiere colonne du
            # coeur est donc `C_edges[0][ci, 0]`, et non `[ci, 1]`.
            #
            # `[ci, 1]` est l'arete INTERIEURE (j=0)-(j=1), deja consommee
            # quelques lignes plus haut comme `c_h`. Le bord gauche
            # reutilisait donc un couplage interieur a la place du sien,
            # alors que le bon coefficient existe : les parametres sont
            # calcules sur un patch (dim+2, dim+2) qui contient le halo.
            #
            # Les bords DROIT et BAS, eux, lisent `[ci, cj]` a cj = dim,
            # c'est-a-dire l'arete (dim)-(dim+1) : le bon coefficient. Le
            # defaut rendait donc l'Hamiltonien asymetrique entre gauche et
            # droite sur un patch pourtant symetrique.
            if j == 0:
                # Bord Gauche: centered halo contraction
                val_halo = z_halo_left[i]
                c_left = hamilt_params['C_edges'][0][ci, 0]
                _halo_term = c_left * val_halo
                if abs(_halo_term) > COEFF_MIN:
                    sparse_list.append(("Z", [idx_H(i, 0)], _halo_term))

            if i == 0:
                # Bord Haut: centered halo contraction
                val_halo = z_halo_top[j]
                c_top = hamilt_params['C_edges'][1][0, cj]
                _halo_term = c_top * val_halo
                if abs(_halo_term) > COEFF_MIN:
                    sparse_list.append(("Z", [idx_V(0, j)], _halo_term))

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
                        (idx_V(i, j+1), z_plaq_right_raw[i]),  # Right : lien V -> theta_v
                        (idx_H(i+1, j), z_plaq_bottom_raw[j]), # Bottom : lien H -> theta_h
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
                        # Meme topologie que K_plaquettes, donc meme correction
                        # de famille (D-113).
                        candidates_xp = [
                            (idx_H(i, j),   1.0),
                            (idx_V(i, j+1), z_plaq_right_raw[i]),
                            (idx_H(i+1, j), z_plaq_bottom_raw[j]),
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

    # Aucun coefficient n'a survécu au seuil : on le dit au lieu d'injecter
    # un terme de remplissage que l'aval prendrait pour un vrai Hamiltonien.
    #
    # Les quatre termes de contraction du halo n'étaient PAS élagués : avec
    # des coefficients tous nuls, ils remplissaient `sparse_list` de termes
    # de valeur exactement 0.0, la liste n'était donc pas vide, et
    # l'opérateur nul repartait vers l'aval comme s'il était réel — le
    # défaut même que cette exception devait empêcher.
    if not sparse_list:
        raise NullHamiltonianError(num_qubits)

    # Retourne l'Opérateur ET les 4 tableaux d'angles du cœur
    return (
        SparsePauliOp.from_sparse_list(sparse_list, num_qubits=num_qubits),
        core_theta_h, core_theta_v, core_psi_h, core_psi_v
    )



def create_period_hamiltonian(hamilt_params, dim, advanced_anomalies_enabled = False) -> SparsePauliOp:
    """
    Construit l'Hamiltonien MHD sur une grille torique (Périodique).
    Utilise SparsePauliOp pour la performance et corrige la topologie des plaquettes/vertex.

    D-59 — CORRIGÉ. À dim = 2 l'anneau périodique dégénère : le lien ZZ
    (i,0)->(i,1) et (i,1)->(i,0 mod 2) relient la MÊME paire de qubits, et
    les deux itérations ajoutaient chacune une entrée au lieu d'être
    fusionnées. Les coefficients étant symétriques par construction
    (`C_edges[0][i,0] == C_edges[0][i,1]` au bit près), le couplage shear
    était appliqué DEUX FOIS : poids effectif ×2. `K_plaquettes` n'a pas ce
    défaut — les 4 quadruplets à dim = 2 sont distincts deux à deux.

    Les liens sont désormais dédupliqués par paire de qubits. À dim ≥ 3
    aucune paire ne se répète, donc l'opérateur est INCHANGÉ bit à bit ; la
    correction ne mord qu'à dim = 2.

    Corrigé AVANT la campagne et non après, alors que l'impact mesuré est
    nul aujourd'hui (0 décision changée sur 12) : c'est le biais Z qui
    domine de 2 à 6,6× (D-47) et masque le doublement. La réoptimisation
    rééquilibre précisément ces poids — si `w_z_frac` se resserre ou `σ`
    s'élargit, le ZZ redevient actif et le facteur 2 devient réel, à
    dim = 2 qui est la seule taille de toutes les campagnes publiées.
    Corriger après coup obligerait à tout rejouer.
    """
    sparse_list = []
    
    # Helpers pour récupérer l'index linéaire du qubit correspondant à un lien
    # Qubits 0 à N^2-1 : Liens Horizontaux (H)
    # Qubits N^2 à 2N^2-1 : Liens Verticaux (V)
    offset_v = dim * dim
    
    def idx_H(y, x): return (y % dim) * dim + (x % dim)
    def idx_V(y, x): return offset_v + (y % dim) * dim + (x % dim)

    # D-59 : paires de qubits ZZ deja emises. La deduplication porte sur la
    # PAIRE NON ORDONNEE — c'est elle qui identifie le lien physique.
    _liens_zz_emis = set()

    def _lien_zz_neuf(a, b):
        """Vrai si ce lien n'a pas deja ete emis. Enregistre au passage."""
        cle = (a, b) if a <= b else (b, a)
        if cle in _liens_zz_emis:
            return False
        _liens_zz_emis.add(cle)
        return True

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
            if abs(c_h) > 1e-6 and _lien_zz_neuf(idx_H(i, j), idx_H(i, j+1)):
                sparse_list.append(("ZZ", [idx_H(i, j), idx_H(i, j+1)], c_h))

            c_v = hamilt_params['C_edges'][1][i, j]
            if abs(c_v) > 1e-6 and _lien_zz_neuf(idx_V(i, j), idx_V(i+1, j)):
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
    # Aucun coefficient n'a survécu au seuil : on le dit au lieu d'injecter
    # un terme de remplissage que l'aval prendrait pour un vrai Hamiltonien.
    if not sparse_list:
        raise NullHamiltonianError(2 * dim * dim)

    return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=2*dim*dim)