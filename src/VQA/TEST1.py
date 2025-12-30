# src/VQA/TEST1.py

import argparse
import json
import os
import random
import numpy as np

# ---------------------------------------------------------
# Fonctions Utilitaires
# ---------------------------------------------------------

def generate_random_counts(num_qubits, total_shots):
    """
    Génère un dictionnaire de counts { '0010': 120, ... } aléatoire.
    """
    counts = {}
    remaining_shots = total_shots
    
    # On choisit arbitrairement quelques états actifs pour simuler une structure
    num_active_states = min(2**num_qubits, 15) 
    
    active_bitstrings = []
    for _ in range(num_active_states):
        # Génération bitstring aléatoire
        bits = [str(random.randint(0, 1)) for _ in range(num_qubits)]
        active_bitstrings.append("".join(bits))
    
    # Distribution des shots
    for _ in range(num_active_states - 1):
        if remaining_shots <= 0: break
        shot_chunk = random.randint(1, remaining_shots // 2)
        key = active_bitstrings.pop()
        counts[key] = counts.get(key, 0) + shot_chunk
        remaining_shots -= shot_chunk
        
    # Le reste
    if active_bitstrings:
        key = active_bitstrings[0]
        counts[key] = counts.get(key, 0) + remaining_shots
        
    return counts

def counts_to_marginals(counts, num_qubits):
    """
    Convertit {string: count} en liste [P(q0), P(q1)...]
    Nécessaire pour le décodage AMR.
    """
    hits = np.zeros(num_qubits)
    total = sum(counts.values())
    
    if total == 0: return hits.tolist()

    for bitstring, count in counts.items():
        # On parcourt la chaîne. 
        # Convention: bitstring[0] correspond au Qubit 0 ici (ordre gauche->droite)
        for i, bit in enumerate(bitstring):
            if i < num_qubits and bit == '1':
                hits[i] += count
                
    return (hits / total).tolist()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def TEST1(data_in, shots):

    print(f"--- [VQA WORKER] Démarrage ---")

    # Vérification simple du contenu
    # Le format attendu par pipeline.py est {'theta_h': [...], 'theta_v': [...]}
    theta_h = np.array(data_in.get("theta_h", [])).flatten()
    theta_v = np.array(data_in.get("theta_v", [])).flatten()
    psi_h   = np.array(data_in.get("psi_h",   [])).flatten()
    psi_v   = np.array(data_in.get("psi_v",   [])).flatten()

    print(f"✅ Input Lu avec succès.")
    print(f"   -> Theta_h: {len(theta_h)} angles")
    print(f"   -> Theta_v: {len(theta_v)} angles")
    print(f"   -> Psi_h: {len(psi_h)} angles")
    print(f"   -> Psi_v: {len(psi_v)} angles")
    
    num_qubits = 2*len(theta_h)
    # (Ici on n'utilise pas les angles pour le calcul random, mais on prouve qu'on les a lus)

    # 2. Simulation (Génération aléatoire de Counts)
    # -----------------------------------------------------
    print(f"⚙️  Génération de {shots} shots pour {num_qubits} qubits...")
    
    counts = generate_random_counts(num_qubits, shots)
    
    # Affichage console pour vérification immédiate
    print(f"📊 Résultat Counts (Extrait): {list(counts.items())[:3]} ...")

    # 3. Traitement des Sorties
    # -----------------------------------------------------
    
    # A. Conversion en Probabilités (CRITIQUE pour pipeline.py)
    # pipeline.py attend une LISTE de floats, pas un dictionnaire.
    probs = counts_to_marginals(counts, num_qubits)
    
    # DEBUG: Force une instabilité pour tester l'AMR visuellement
    
    for i in range(len(probs)):
        probs[i] = 0

    if len(probs) >= 2:
        probs[9] = 0.95 # Qubit 0 instable
        #probs[0] = 0.85 # Qubit 1 instable"""

    print("PROBABILITIES : ", probs)

    return probs


if __name__ == "__main__":
    TEST1()