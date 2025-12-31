# scripts/execute.py
import argparse
import json
import os
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.optimize import minimize
from collections import defaultdict
from typing import Sequence


from qiskit import qpy
from qiskit_aer import Aer
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp

# =========================================================
# 1. GESTION DES PARAMÈTRES GLOBAUX
# =========================================================

def build_parameter_map(mapping_data):
    """
    Construit le vecteur x0 global et une map {nom_param: index_dans_x0}.
    """
    x0 = []
    name_to_idx = {}
    current_idx = 0
    
    # L'ordre d'itération ici définit la structure de x0
    # On itère sur les rôles: theta, phi, eps
    for role, data in mapping_data["parameter_vectors"].items():
        names = data["params"]
        values = data["init_values"]
        
        if values is None or len(values) == 0:
            # Si pas de valeurs, on met des 0 ou aléatoire
            values = np.random.uniform(0, 2*np.pi, len(names)).tolist()
            
        if len(names) != len(values):
            raise ValueError(f"Mismatch in length for {role}: {len(names)} names vs {len(values)} values.")
            
        for name, val in zip(names, values):
            x0.append(val)
            name_to_idx[name] = current_idx
            current_idx += 1
            
    return np.array(x0), name_to_idx

def map_circuit_params_to_global(circuit, name_to_idx):
    """
    Pour un circuit donné, crée une liste d'indices qui dit :
    "Le paramètre #i de ce circuit correspond à l'index J du vecteur global x".
    """
    circuit_indices = []
    for param in circuit.parameters:
        p_name = param.name
        if p_name not in name_to_idx:
            raise KeyError(f"Parameter '{p_name}' in circuit '{circuit.name}' not found in global JSON mapping.")
        circuit_indices.append(name_to_idx[p_name])
    return circuit_indices

# =========================================================
# 2. CONSTRUCTION DES OBSERVABLES
# =========================================================

def build_observable(num_qubits, meas_config):
    """
    Construit l'opérateur SparsePauliOp pour la mesure.
    Ex: qubit=0, pauli=Z, num_qubits=4 -> "IIIZ"
    """
    # Liste de coeffs et ops pour la somme
    ops = []
    coeffs = []
    
    for m in meas_config:
        qubit_idx = m["qubit"]
        pauli_char = m["pauli"]
        coeff = m["coeff"]
        
        # Qiskit String Order: q3 q2 q1 q0
        pauli_list = ["I"] * num_qubits
        # On remplace à la position (N - 1 - k)
        pauli_list[num_qubits - 1 - qubit_idx] = pauli_char
        pauli_str = "".join(pauli_list)
        
        ops.append(pauli_str)
        coeffs.append(coeff)
        
    return SparsePauliOp(ops, coeffs)

# =========================================================
# 3. MAIN EXECUTION
# =========================================================

def execute(qc, cost_hamiltonian, mode, backend, shots, reps):
    num_qubits = qc.num_qubits

    initial_gamma = np.pi
    initial_beta = np.pi / 2
    init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]

    # Select backend
    if mode == "simulator":
        backend = Aer.get_backend('qasm_simulator')
    else:
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        backend = provider.get_backend('ibmq_qasm_simulator')  # choose hardware device

    def cost_func_estimator(params, ansatz, hamiltonian, estimator):
        # transform the observable defined on virtual qubits to
        # an observable defined on all physical qubits
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    
        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])
    
        results = job.result()[0]
        cost = results.data.evs
    
        objective_func_vals.append(cost)
    
        return cost

    objective_func_vals = []  # Global variable
    with Session(backend=backend) as session:
        # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=`
        estimator = Estimator(mode=session)
        estimator.options.default_shots = shots
        if mode != "simulator":
            # Only set options for real hardware
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            estimator.options.twirling.enable_gates = True
            estimator.options.twirling.num_randomizations = "auto"
    
        result = minimize(
            cost_func_estimator,
            init_params,
            args=(qc, cost_hamiltonian, estimator),
            method="COBYLA",
            tol=1e-2,
        )
        print(result)

    if args.verbose:
        plt.figure(figsize=(12, 6))
        plt.plot(objective_func_vals)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()
    
    optimized_circuit = qc.assign_parameters(result.x)
    optimized_circuit.draw("mpl", fold=False, idle_wires=False)

    # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `backend=`
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    
    # Set simple error suppression/mitigation options
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"
    
    pub = (optimized_circuit,)
    job = sampler.run([pub], shots=int(shots))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    final_distribution_int = {key: val / shots for key, val in counts_int.items()}
    final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}
    print(final_distribution_int)
    return final_distribution_int