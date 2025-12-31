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

from VQA.init_qbits_state import init_qbits_state
from VQA.cost_hamiltonian import create_bounded_hamiltonian, create_period_hamiltonian


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

def mapping(data_in, hamilt_params, period_bound=True, reps=2):

    # On adapte la lecture de dimension de mon tableau de données pour la séparation Halo/coeur
    raw_theta = np.array(data_in.get("theta_h", []))
    
    # Détermine si l'entrée est plate (N^2) ou carrée (NxN)
    if raw_theta.ndim == 1:
        side_len = int(np.sqrt(len(raw_theta)))
    else:
        side_len = raw_theta.shape[0]

    # Gestion Halo : Si period_bound=True (Tore), halo=0. Sinon halo=2 (1px de chaque coté)
    halo_dim = 0 if period_bound else 2
    dim = side_len - halo_dim

    theta_h = np.array(data_in.get("theta_h", []))
    theta_v = np.array(data_in.get("theta_v", []))
    psi_h   = np.array(data_in.get("psi_h",   []))
    psi_v   = np.array(data_in.get("psi_v",   []))

    cost_hamiltonian = None
    if period_bound:
        cost_hamiltonian = create_period_hamiltonian(hamilt_params, dim)
    else:
        cost_hamiltonian, theta_h, theta_v, psi_h, psi_v = create_bounded_hamiltonian(
            hamilt_params, dim, theta_h, theta_v, psi_h, psi_v
        )

    qc = init_qbits_state(theta_h, theta_v, psi_h, psi_v)

    ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps, initial_state=qc)

    qc.compose(ansatz, inplace=True)

    qc.barrier()

    qc.measure_all()

    return qc, cost_hamiltonian
    
    qc.draw("mpl")

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

    num_qubits = 18
    # (Ici on n'utilise pas les angles pour le calcul random, mais on prouve qu'on les a lus)

    # 2. Simulation (Génération aléatoire de Counts)
    # -----------------------------------------------------
    print(f"⚙️  Génération de {1000} shots pour {num_qubits} qubits...")
    
    counts = generate_random_counts(num_qubits, 1000)
    
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
        probs[-1] = 0.95 # Qubit 0 instable
        #probs[0] = 0.85 # Qubit 1 instable"""

    print("PROBABILITIES : ", probs)

    return probs
