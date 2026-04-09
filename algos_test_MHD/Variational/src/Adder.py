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

from networkx.readwrite import json_graph
from scipy.stats import unitary_group

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import UnitaryGate, AndGate, PauliEvolutionGate, UCRYGate, UCRZGate
from qiskit.circuit import QuantumCircuit, ControlledGate, ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import qpy


from itertools import product
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import QFT

def build_boolean_incrementer(m_qubits):
    """
    Construit un Incrémenteur (+1) booléen utilisant m-1 ancillas.
    Complexité linéaire O(m).
    """
    # Registres
    # 1. Gestion des registres
    qr = QuantumRegister(m_qubits, 'q')
    
    # Optimisation : On combine q0 et q1 dès le départ, donc on économise 1 ancilla.
    # Pour m=3, on a besoin de 1 ancilla. Pour m=4, 2 ancillas.
    num_ancillas = max(0, m_qubits - 2)
    
    if num_ancillas > 0:
        ar = AncillaRegister(num_ancillas, 'ancilla')
        qc = QuantumCircuit(qr, ar, name="Boolean_Decr")
    else:
        qc = QuantumCircuit(qr, name="Boolean_Decr")

    # Cas triviaux
    if m_qubits == 1:
        qc.x(qr[0])
        return qc
    if m_qubits == 2:
        # q1 bascule si q0 est 0
        qc.x(qr[0])     # Détection du 0
        qc.cx(qr[0], qr[1])
        qc.x(qr[0]) 
        qc.x(qr[0])
        return qc

    qc.x(qr[0:m_qubits-1])

    # --- Étape 2 : Calcul des Emprunts (Borrows) ---
    # On monte la chaîne des ancillas.
    
    # Premier ancilla : combine q0 et q1 (qui sont inversés par les X précédents)
    # ar[0] sera 1 si (q0 original == 0) ET (q1 original == 0)
    qc.ccx(qr[0], qr[1], ar[0])
    
    # Ancillas suivants : combine l'ancilla précédent et le qubit suivant
    for i in range(1, num_ancillas):
        # ar[i] = ar[i-1] AND q[i+1]
        qc.ccx(ar[i-1], qr[i+1], ar[i])

    # --- Étape 3 : Application et Nettoyage (Ripple Down) ---
    # On applique le flip au MSB, puis on nettoie et on descend.
    # C'est crucial de faire "Flip -> Uncompute" bit par bit pour que
    # les contrôles (qr) soient toujours dans le bon état pour le uncompute.
    
    # A. Flip du MSB (q_m-1) si l'emprunt final est actif
    qc.cx(ar[num_ancillas-1], qr[m_qubits-1])
    
    # B. Boucle descendante pour les bits intermédiaires
    for i in range(num_ancillas-1, 0, -1):
        # 1. Uncompute l'ancilla actuel (inverse de l'étape 2)
        # On utilise qr[i+1] qui n'a PAS encore été modifié par la décrémentation
        qc.ccx(ar[i-1], qr[i+1], ar[i])
        
        # 2. Flip du qubit correspondant à l'emprunt précédent
        qc.cx(ar[i-1], qr[i+1])
        
    # C. Traitement des premiers qubits (q0, q1, ar[0])
    
    # Uncompute ar[0]
    qc.ccx(qr[0], qr[1], ar[0])
    
    # Flip q1 (contrôlé par q0 qui est à 1 grâce à la porte X initiale si q0 était 0)
    qc.cx(qr[0], qr[1])
    
    # --- Étape 4 : Finalisation ---
    # On remet les X pour restaurer l'état original des contrôles
    qc.x(qr[0:m_qubits-1])
    
    # Le LSB (q0) change TOUJOURS dans une incrémentation ou décrémentation
    qc.x(qr[0])

    return qc

def build_qft_decrementer(m_qubits):
    """
    Construit un Décrémenteur (-1) basé sur la QFT (Draper).
    Utilise 0 ancilla.
    """
    qc = QuantumCircuit(m_qubits, name="QFT_Decr")
    
    # 1. QFT
    qc.append(QFT(m_qubits, do_swaps=True).to_gate(), range(m_qubits))
    
    # 2. Phase Shift (Ajouter -1)
    # Angle = 2*pi * (-1) / 2^(k+1)
    for i in range(m_qubits):
        angle = -2 * np.pi / (2**(i + 1))
        qc.p(angle, i)
        
    # 3. Inverse QFT
    qc.append(QFT(m_qubits, inverse=True, do_swaps=True).to_gate(), range(m_qubits))
    
    return qc

def get_adder_gate(m_qubits, total_available_qubits):
    """
    Fonction principale qui choisit et retourne la bonne Gate.
    Implémente l'opérateur |k-1><k|.
    """
    
    # Condition du papier : Si n > 2m, on utilise la méthode booléenne (avec ancillas)
    # Sinon, on utilise la méthode QFT (compacte)
    if total_available_qubits > 2 * m_qubits:
        print(f"Condition n ({total_available_qubits}) > 2m ({2*m_qubits}).")
        print("-> Utilisation de la méthode Booléenne (Rapide, avec Ancillas)")
        
        # 1. Créer l'incrémenteur
        incr_circuit = build_boolean_incrementer(m_qubits+1)
        
        # 2. Prendre l'inverse pour avoir le décrémenteur (|k-1>)
        decr_circuit = incr_circuit.inverse()
        decr_circuit.name = "Bool_Decr(-1)"
        
        # Note : Ce circuit attend m qubits de données + (m-2) ancillas
        return decr_circuit.to_gate(), True
        
    else:
        print(f"Condition n ({total_available_qubits}) <= 2m ({2*m_qubits}).")
        print("-> Utilisation de la méthode QFT (Compacte, sans Ancilla)")
        
        decr_circuit = build_qft_decrementer(m_qubits+1)
        return decr_circuit.to_gate(), False # Indique qu'on utilise une méthode particulière (# qubits différent)



# -----------------------------
# Main entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Map MaxCut problem to Hamiltonian operator")
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"], help="Quantum backend")
    parser.add_argument("--out-dir", default="data", help="Output directory for mapping")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--numqbits", type=int, required=True, help="Number of qubits used for the whole circuit.")
    parser.add_argument("--Mqbits", type=int, required=True, help="Number of qubits used for Fourier coeffs.")
    

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Building ZGR QFT Quantum Circuit...")
    
    gate,ver = get_adder_gate(args.Mqbits, args.numqbits)
    
    full_qc = QuantumCircuit(args.numqbits)
    
    # Apply |c> on first register
    qubit_mapping = [1] + list(range(args.numqbits - args.Mqbits, args.numqbits)) #+ list(range(2, 1 + args.Mqbits))
    full_qc.append(gate, qubit_mapping)
    
    # 4. Save Outputs
    full_qc.draw("mpl")

    plt.show()

if __name__ == "__main__":
    main()
