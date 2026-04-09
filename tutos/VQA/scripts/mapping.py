# scripts/mapping.py

"""
Stage 1: Problem Mapping for MaxCut (QAOA pipeline)
Compatible with Qiskit 2.x (uses SparsePauliOp instead of qiskit.opflow)
"""

import argparse
import json
import os
import random
from IPython.display import display
from scipy.optimize import minimize
import scipy as sp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from math import pi
from networkx.readwrite import json_graph

from qiskit import qpy
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import TwoLocal, zz_feature_map, NLocal, CCXGate, CRZGate, RXGate, QAOAAnsatz, efficient_su2
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.result import QuasiDistribution
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Session,
    EstimatorOptions,
    EstimatorV2 as Estimator,
)
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# Necessary imports and definitions to track time in microseconds
import time
 
import itertools as it
from typing import Union, List
import warnings

from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
 
# from qiskit.providers.fake_provider import Fake20QV1
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator, Batch
 
import itertools as it
 
warnings.filterwarnings("ignore")
 
def solve_regularized_gen_eig(
    h: np.ndarray,
    s: np.ndarray,
    threshold: float,
    k: int = 1,
    return_dimn: bool = False,
) -> Union[float, List[float]]:
    """
    Method for solving the generalized eigenvalue problem with regularization
 
    Args:
        h (numpy.ndarray):
            The effective representation of the matrix in our Krylov subspace
        s (numpy.ndarray):
            The matrix of overlaps between vectors of our Krylov subspace
        threshold (float):
            Cut-off value for the eigenvalue of s
        k (int):
            Number of eigenvalues to return
        return_dimn (bool):
            Whether to return the size of the regularized subspace
 
    Returns:
        lowest k-eigenvalue(s) that are the solution of the regularized generalized eigenvalue problem
 
 
    """
    s_vals, s_vecs = sp.linalg.eigh(s)
    s_vecs = s_vecs.T
    good_vecs = np.array([vec for val, vec in zip(s_vals, s_vecs) if val > threshold])
    h_reg = good_vecs.conj() @ h @ good_vecs.T
    s_reg = good_vecs.conj() @ s @ good_vecs.T
    if k == 1:
        if return_dimn:
            return sp.linalg.eigh(h_reg, s_reg)[0][0], len(good_vecs)
        else:
            return sp.linalg.eigh(h_reg, s_reg)[0][0]
    else:
        if return_dimn:
            return sp.linalg.eigh(h_reg, s_reg)[0][:k], len(good_vecs)
        else:
            return sp.linalg.eigh(h_reg, s_reg)[0][:k]
        

def single_particle_gs(H_op, n_qubits):
    """
    Find the ground state of the single particle(excitation) sector
    """
    H_x = []
    for p, coeff in H_op.to_list():
        H_x.append(set([i for i, v in enumerate(Pauli(p).x) if v]))
 
    H_z = []
    for p, coeff in H_op.to_list():
        H_z.append(set([i for i, v in enumerate(Pauli(p).z) if v]))
 
    H_c = H_op.coeffs
 
    print("n_sys_qubits", n_qubits)
 
    n_exc = 1
    sub_dimn = int(sp.special.comb(n_qubits + 1, n_exc))
    print("n_exc", n_exc, ", subspace dimension", sub_dimn)
 
    few_particle_H = np.zeros((sub_dimn, sub_dimn), dtype=complex)
 
    sparse_vecs = [
        set(vec) for vec in it.combinations(range(n_qubits + 1), r=n_exc)
    ]  # list all of the possible sets of n_exc indices of 1s in n_exc-particle states
 
    m = 0
    for i, i_set in enumerate(sparse_vecs):
        for j, j_set in enumerate(sparse_vecs):
            m += 1
 
            if len(i_set.symmetric_difference(j_set)) <= 2:
                for p_x, p_z, coeff in zip(H_x, H_z, H_c):
                    if i_set.symmetric_difference(j_set) == p_x:
                        sgn = ((-1j) ** len(p_x.intersection(p_z))) * (
                            (-1) ** len(i_set.intersection(p_z))
                        )
                    else:
                        sgn = 0
 
                    few_particle_H[i, j] += sgn * coeff
 
    gs_en = min(np.linalg.eigvalsh(few_particle_H))
    print("single particle ground state energy: ", gs_en)
    return gs_en

# -----------------------------
# Point d'Entrée Principal (Modifié pour SSVQE)
# -----------------------------
def main():

    parser = argparse.ArgumentParser(description="Map MaxCut problem to Hamiltonian operator")
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"], help="Quantum backend")
    parser.add_argument("--out-dir", default="results", help="Output directory for mapping")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)    

    # Define problem Hamiltonian.
    n_qubits = 10
    # coupling strength for XX, YY, and ZZ interactions
    JX = 1
    JY = 3
    JZ = 2
    
    # Define the Hamiltonian:
    H_int = [["I"] * n_qubits for _ in range(3 * (n_qubits - 1))]
    for i in range(n_qubits - 1):
        H_int[i][i] = "Z"
        H_int[i][i + 1] = "Z"
    for i in range(n_qubits - 1):
        H_int[n_qubits - 1 + i][i] = "X"
        H_int[n_qubits - 1 + i][i + 1] = "X"
    for i in range(n_qubits - 1):
        H_int[2 * (n_qubits - 1) + i][i] = "Y"
        H_int[2 * (n_qubits - 1) + i][i + 1] = "Y"
    H_int = ["".join(term) for term in H_int]
    H_tot = [
        (term, JZ)
        if term.count("Z") == 2
        else (term, JY)
        if term.count("Y") == 2
        else (term, JX)
        for term in H_int
    ]
    
    # Get operator
    H_op = SparsePauliOp.from_list(H_tot)
    print(H_tot)

    # Get Hamiltonian restricted to single-particle states
    single_particle_H = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(i + 1):
            for p, coeff in H_op.to_list():
                p_x = Pauli(p).x
                p_z = Pauli(p).z
                if all(p_x[k] == ((i == k) + (j == k)) % 2 for k in range(n_qubits)):
                    sgn = ((-1j) ** sum(p_z[k] and p_x[k] for k in range(n_qubits))) * (
                        (-1) ** p_z[i]
                    )
                else:
                    sgn = 0
                single_particle_H[i, j] += sgn * coeff
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            single_particle_H[i, j] = np.conj(single_particle_H[j, i])
    
    # Set dt according to spectral norm
    dt = np.pi / np.linalg.norm(single_particle_H, ord=2)
    print(dt)

    # Set parameters for quantum Krylov algorithm
    krylov_dim = 5  # size of krylov subspace
    num_trotter_steps = 6
    dt_circ = dt / num_trotter_steps

    qc_state_prep = QuantumCircuit(n_qubits)
    qc_state_prep.x(int(n_qubits / 2) + 1)
    qc_state_prep.draw("mpl", scale=0.5)
    plt.show()


    t = Parameter("t")
    
    ## Create the time-evo op circuit
    evol_gate = PauliEvolutionGate(
        H_op, time=t, synthesis=LieTrotter(reps=num_trotter_steps)
    )
    
    qr = QuantumRegister(n_qubits)
    qc_evol = QuantumCircuit(qr)
    qc_evol.append(evol_gate, qargs=qr)

    ## Create the time-evo op circuit
    evol_gate = PauliEvolutionGate(
        H_op, time=dt, synthesis=LieTrotter(reps=num_trotter_steps)
    )
    
    ## Create the time-evo op dagger circuit
    evol_gate_d = PauliEvolutionGate(
        H_op, time=dt, synthesis=LieTrotter(reps=num_trotter_steps)
    )
    evol_gate_d = evol_gate_d.inverse()
    
    # Put pieces together
    qc_reg = QuantumRegister(n_qubits)
    qc_temp = QuantumCircuit(qc_reg)
    qc_temp.compose(qc_state_prep, inplace=True)
    for _ in range(num_trotter_steps):
        qc_temp.append(evol_gate, qargs=qc_reg)
    for _ in range(num_trotter_steps):
        qc_temp.append(evol_gate_d, qargs=qc_reg)
    qc_temp.compose(qc_state_prep.inverse(), inplace=True)
    
    # Create controlled version of the circuit
    controlled_U = qc_temp.to_gate().control(1)
    
    # Create hadamard test circuit for real part
    qr = QuantumRegister(n_qubits + 1)
    qc_real = QuantumCircuit(qr)
    qc_real.h(0)
    qc_real.append(controlled_U, list(range(n_qubits + 1)))
    qc_real.h(0)
    
    print("Circuit for calculating the real part of the overlap in S via Hadamard test")
    qc_real.draw("mpl", fold=-1, scale=0.5)

    print(
        "Number of layers of 2Q operations",
        qc_real.decompose(reps=2).depth(lambda x: x[0].num_qubits == 2),
    )

if __name__ == "__main__":
    main()
