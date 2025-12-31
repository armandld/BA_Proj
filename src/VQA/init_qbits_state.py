# scripts/init_qbits_state.py

import matplotlib.pyplot as plt
import numpy as np

from qiskit.circuit import QuantumCircuit

def init_qbits_state(theta_h, theta_v, psi_h, psi_v):

    theta_h = theta_h.flatten()
    theta_v = theta_v.flatten()
    psi_h   = psi_h.flatten()
    psi_v   = psi_v.flatten()

    num_qubits = 2*len(theta_h)  # Supposant une grille rectangulaire
    qc = QuantumCircuit(num_qubits)

    for i in range(len(theta_h)):
        qc.r(theta_h[i], - psi_h[i] -np.pi/2, i)
        qc.r(theta_v[i], - psi_v[i] -np.pi/2, i+len(theta_h))

    return qc
