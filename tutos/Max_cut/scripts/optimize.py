import argparse
import json
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt

# 1. Importation correcte de Maxcut (minuscule 'c')
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit.transpiler import generate_preset_pass_manager
from qiskit import qpy

def load_mapping(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Optimize QAOA parameters for Max-Cut")
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"])
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mapping = load_mapping(f"{args.out_dir}/mapping.json")
    edges = mapping["edges"]

    circuit_file = os.path.join(args.out_dir, "qaoa_circuit.qpy")
    with open(circuit_file, "rb") as f:
        circuit = qpy.load(f)[0]  # load returns a list of circuits


    # 4. ***** LOGIQUE BACKEND UTILISANT SAMPLER V2 *****
    if args.backend == "aer":
        # Primitive V2 du simulateur Aer
        backend = FakeFez()
    elif args.backend == "estimator":
        # Primitive V2 de référence de Qiskit
        # (Nous l'appelons 'estimator' mais c'est un Sampler)
        primitive = FakeFez()
    else:
        raise ValueError("Unsupported backend")
    
    # Create pass manager for transpilation
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    
    candidate_circuit = pm.run(circuit)
    candidate_circuit.draw("mpl", fold=False, idle_wires=False)

    candidate_circuit_file = os.path.join(args.out_dir, "qaoa_circuit_optimal.qpy")
    with open(candidate_circuit_file, "wb") as f:
        qpy.dump(candidate_circuit, f)  # saves the circuit to a binary file

    if args.verbose:
        plt.show()

if __name__ == "__main__":
    main()