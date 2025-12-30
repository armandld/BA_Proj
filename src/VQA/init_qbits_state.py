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

def init_qbits_state(theta_h, theta_v, psi_h, psi_v):
    """Initialisation des qubits selon les angles fournis."""
    num_qubits = 2*len(theta_h)  # Supposant une grille rectangulaire
    qc = QuantumCircuit(num_qubits)

    for i in range(len(theta_h)):
        qc.r(theta_h[i], - psi_h[i] -np.pi/2, i)
        qc.r(theta_v[i], - psi_v[i] -np.pi/2, i+len(theta_h))

    return qc
