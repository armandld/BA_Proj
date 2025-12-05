# scripts/execute.py
import argparse
import json
import os
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from collections import defaultdict
from typing import Sequence


from qiskit import qpy
from qiskit_aer import Aer
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp


def main():
    parser = argparse.ArgumentParser(description="Execute VQA circuit")
    parser.add_argument("--mode", default="simulator", choices=["simulator", "hardware"])
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"])
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--out-dir", default="data", help="Output directory for mapping")
    parser.add_argument("--method", default="COBYLA", required=True, choices = ["COBYLA", "Nelder-Mead", "Powell", "L-BFGS-B"], help="Optimization method for minimize, default: COBYLA.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    

    # Load mapping JSON
    out_file = os.path.join(args.out_dir, "mapping.json")
    with open(out_file, "r") as f:
        mapping_data = json.load(f)
        
    circuits = []
    for k in range(len(mapping_data["transpiled_circuits"])):
        print(f"Loading transpiled circuit number {k+1}...")
        circuit_file = os.path.join(args.out_dir, f"{mapping_data['transpiled_circuits'][k]['file']}")
        with open(circuit_file, "rb") as f:
            circuits.append(qpy.load(f)[0])

    num_qubits = mapping_data["num_qubits"]
    hamiltonian_terms = mapping_data["hamiltonian"]

    # Rebuild SparsePauliOp
    cost_hamiltonian = SparsePauliOp.from_list(hamiltonian_terms)

    seed = 2
    
    random.seed(seed)
    np.random.seed(seed)

    initial_gamma = np.pi
    initial_beta = np.pi / 2
    init_params = [initial_beta, initial_beta, initial_gamma, initial_gamma]

    # Select backend
    if args.mode == "simulator":
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
    optimized_circuits = []
    with Session(backend=backend) as session:
        # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=`
        estimator = Estimator(mode=session)
        estimator.options.default_shots = args.shots
        if args.mode!= "simulator":
            # Only set options for real hardware
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            estimator.options.twirling.enable_gates = True
            estimator.options.twirling.num_randomizations = "auto"
        results = []
        for c in circuits:
            result = minimize(
                cost_func_estimator,
                init_params,
                args=(c, cost_hamiltonian, estimator),
                method=args.method,
                tol=1e-2,
            )
            print(result)
            results.append(result)

        if args.verbose:
            plt.figure(figsize=(12, 6))
            plt.plot(objective_func_vals)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.show()
        
        optimized_circuit = c.assign_parameters(result.x)
        optimized_circuits.append(optimized_circuit)
        if args.verbose:
            optimized_circuit.draw("mpl", fold=False, idle_wires=False)

    # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `backend=`
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = args.shots
    
    # Set simple error suppression/mitigation options
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"
    final_distribution_int = []
    final_distribution_bin = []
    for optimized_circuit in optimized_circuits:
        pub = (optimized_circuit,)
        job = sampler.run([pub], shots=int(args.shots))
        counts_int = job.result()[0].data.meas.get_int_counts()
        counts_bin = job.result()[0].data.meas.get_counts()
        shots = sum(counts_int.values())
        final_distribution_int = {key: val / shots for key, val in counts_int.items()}
        final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}
        final_distribution_int.append(final_distribution_int)
        final_distribution_bin.append(final_distribution_bin)
        print(final_distribution_int)

    # Save results
    out_file = f"{args.out_dir}/execution_result.json"
    execution_data = {
        "optimal_parameters": result.x.tolist(),
        "objective_value": result.fun,
        "final_distribution_int": final_distribution_int,
        "final_distribution_bin": final_distribution_bin
    }
    with open(out_file, "w") as f:
        json.dump(execution_data, f, indent=2)

    if args.verbose:
        print("Execution result saved to", out_file)
        print("Objective value:", execution_data["objective_value"])

if __name__ == "__main__":
    main()
