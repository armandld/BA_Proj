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

def main():
    parser = argparse.ArgumentParser(description="Execute Variational Optimization")
    parser.add_argument("--out-dir", default="../data", help="Data directory")
    parser.add_argument("--mode", default="simulator", choices=["simulator", "hardware"])
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"])
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--method", default="COBYLA", choices=["COBYLA", "L-BFGS-B", "Powell"])
    parser.add_argument("--mu", type=float, default=1.0, help="Weight for the second circuit (H2)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mode == "simulator":
        backend = Aer.get_backend('qasm_simulator')
    else:
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        backend = provider.get_backend('ibmq_qasm_simulator')  # choose hardware device

    # Ignorer les warnings de param vector partiels lors du chargement QPY
    warnings.filterwarnings("ignore", message="The ParameterVector: .* is not fully identical")

    # 1. Chargement du Mapping
    mapping_path = os.path.join(args.out_dir, "mapping.json")
    with open(mapping_path, "r") as f:
        mapping_data = json.load(f)

    # 2. Chargement des Circuits Transpilés
    circuits = []
    observables = []
    # Liste de listes d'indices pour lier x_global -> circuit_local
    circuit_mappings = [] 
    
    # Vecteur global initial et map de noms
    x0, name_to_idx = build_parameter_map(mapping_data)
    num_qubits = mapping_data["num_qubits"]

    print(f"Loading {len(mapping_data['transpiled_circuits'])} circuits...")
    
    for entry in mapping_data["transpiled_circuits"]:
        # Load QPY
        fpath = os.path.join(args.out_dir, entry["file"])
        with open(fpath, "rb") as f:
            qc = qpy.load(f)[0]
            circuits.append(qc)
        
        logical_obs = build_observable(num_qubits, entry["measurement"])

        # Build Observable (ex: 0.5 * Z0)
        if qc.layout is not None:
            obs = logical_obs.apply_layout(qc.layout)
            observables.append(obs)
        else:
            observables.append(logical_obs)
        
        # Map Parameters
        indices = map_circuit_params_to_global(qc, name_to_idx)
        circuit_mappings.append(indices)

    # 3. Setup Backend & Estimator
    with Session(backend=backend) as session:
        estimator = Estimator(mode=session) # Utilisation Estimator V2 local
        
        # 4. Fonction de Coût Globale
        # H_total = H_1 + mu * H_2
        history = []

        def cost_function(params_values):
            # params_values est le vecteur x global fourni par l'optimiseur
            
            pubs = []
            
            # Préparation des PUBs (Primitive Unified Blocs) pour l'Estimator V2
            # Chaque PUB est un tuple (circuit, observable, valeurs_parametres)
            for i, qc in enumerate(circuits):
                # Extraction des valeurs spécifiques pour ce circuit
                # On utilise le mapping précalculé pour être très rapide
                indices = circuit_mappings[i]
                local_values = params_values[indices]
                
                # Note: Estimator V2 attend un tableau de valeurs 2D (shots, n_params) 
                # ou 1D si 1 shot. Ici on passe [local_values] pour 1 set de paramètres.
                pubs.append((qc, observables[i], local_values))
                
            # Exécution en batch (tous les circuits d'un coup)
            job = estimator.run(pubs, precision=1.0/np.sqrt(args.shots))
            results = job.result()
            
            total_cost = 0.0
            
            # Somme pondérée des résultats
            # C1 (index 0) + mu * C2 (index 1)
            # Note: Les coefficients Z (1.0 ou 0.5) sont DÉJÀ dans l'observable SparsePauliOp
            
            val1 = results[0].data.evs # Expectation value circuit 1
            total_cost += val1
            
            if len(results) > 1:
                val2 = results[1].data.evs # Expectation value circuit 2
                total_cost += args.mu * val2
                
            history.append(total_cost)
            if len(history) % 10 == 0:
                print(f"Iter {len(history)}: Cost = {total_cost:.6f}")
                
            return total_cost

        # 5. Lancement Optimisation
        print(f"Starting optimization with {args.method} on {len(x0)} parameters...")
        print(f"Initial Cost evaluation...")
        
        res = minimize(
            cost_function,
            x0,
            method=args.method,
            options={'maxiter': 200, 'disp': True}
        )

        print("\nOptimization Result:")
        print(f"Success: {res.success}")
        print(f"Final Cost: {res.fun:.6f}")
    
    # 6. Sauvegarde et Affichage
    if args.verbose:
        plt.figure(figsize=(10, 5))
        plt.plot(history)
        plt.xlabel("Iterations")
        plt.ylabel("Cost Function")
        plt.title("VQA Convergence")
        plt.grid(True)
        plt.show()

    # Sauvegarde des résultats
    result_data = {
        "final_cost": res.fun,
        "iterations": res.nfev,
        "optimal_parameters": res.x.tolist(),
        "history": [float(h) for h in history]
    }
    
    res_path = os.path.join(args.out_dir, "optimization_results.json")
    with open(res_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Results saved to {res_path}")

if __name__ == "__main__":
    main()