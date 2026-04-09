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


# ---------------------------------------------------------
# Preparation of |c>'s matrix Parametrized
# ---------------------------------------------------------
# =========================================================
# 1. HELPERS: Décomposition compatible avec les Paramètres
# =========================================================

def ucry_decomp(qc, angles, controls, target):
    """
    Implémentation récursive d'une UCRY qui accepte des ParameterVector.
    Décompose la porte en une suite de CNOT et RY.
    """
    if len(controls) == 0:
        # Cas de base : Rotation simple sur la cible
        qc.ry(angles[0], target)
        return

    # Étape récursive (Démultiplexage)
    n_angles = len(angles)
    half = n_angles // 2
    
    theta_0 = angles[:half] # Partie Control=0
    theta_1 = angles[half:] # Partie Control=1
    
    # Calcul symbolique (supporté par Qiskit Parameter)
    theta_sum  = [(t0 + t1) / 2.0 for t0, t1 in zip(theta_0, theta_1)]
    theta_diff = [(t0 - t1) / 2.0 for t0, t1 in zip(theta_0, theta_1)]
    
    control_qubit = controls[-1]
    remaining_controls = controls[:-1]
    
    # 1. Appliquer la somme (partie commune)
    ucry_decomp(qc, theta_sum, remaining_controls, target)
    
    # 2. CNOT
    qc.cx(control_qubit, target)
    
    # 3. Appliquer la différence (correction)
    ucry_decomp(qc, theta_diff, remaining_controls, target)
    
    # 4. CNOT
    qc.cx(control_qubit, target)


def ucrz_decomp(qc, angles, controls, target):
    """
    Implémentation récursive d'une UCRZ qui accepte des ParameterVector.
    """
    if len(controls) == 0:
        qc.rz(angles[0], target)
        return

    n_angles = len(angles)
    half = n_angles // 2
    
    theta_0 = angles[:half]
    theta_1 = angles[half:]
    
    theta_sum  = [(t0 + t1) / 2.0 for t0, t1 in zip(theta_0, theta_1)]
    theta_diff = [(t0 - t1) / 2.0 for t0, t1 in zip(theta_0, theta_1)]
    
    control_qubit = controls[-1]
    remaining_controls = controls[:-1]
    
    ucrz_decomp(qc, theta_sum, remaining_controls, target)
    qc.cx(control_qubit, target)
    ucrz_decomp(qc, theta_diff, remaining_controls, target)
    qc.cx(control_qubit, target)

# -----------------------------
# Main entry point
# -----------------------------
def main():
    pass

if __name__ == "__main__":
    main()
