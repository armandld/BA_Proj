import argparse
import warnings
import json
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit.transpiler import generate_preset_pass_manager
from qiskit import qpy

from qiskit.circuit import ParameterVector

def load_mapping(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Optimize VQA circuits")
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"], help="Quantum backend")
    parser.add_argument("--out-dir", default="../data", help="Output directory for mapping")
    parser.add_argument("--opt_level", type=int, required=True, choices = [0, 1, 2, 3], help="Optimization level for transpiler, default: 3.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="The ParameterVector: .* is not fully identical")

    if args.backend == "aer":
        # Primitive V2 du simulateur Aer
        backend = FakeFez()
    elif args.backend == "estimator":
        # Primitive V2 de référence de Qiskit
        # (Nous l'appelons 'estimator' mais c'est un Sampler)
        backend = FakeFez()
    else:
        raise ValueError("Unsupported backend")
    
    # Create pass manager for transpilation
    pm = generate_preset_pass_manager(optimization_level=args.opt_level, backend=backend)

    print("Optimization Level: ", args.opt_level)

    mapping = load_mapping(f"{args.out_dir}/mapping.json")
    circuits = []
    for k in range(len(mapping["circuits"])):
        print(f"Optimizing circuit number {k+1}...")
        circuit_file = os.path.join(args.out_dir, f"{mapping['circuits'][k]['file']}")
        with open(circuit_file, "rb") as f:
            circuits.append(qpy.load(f)[0])  # load returns a list of circuits
        
        candidate_circuit = pm.run(circuits[k])
        
        if args.verbose:
            candidate_circuit.draw("mpl", fold=False, idle_wires=False)

        candidate_circuit_file = os.path.join(args.out_dir, f"transpiled_circuit_{k+1}.qpy")

        with open(candidate_circuit_file, "wb") as f:
            qpy.dump(candidate_circuit, f)  # saves the circuit to a binary file

        mapping["transpiled_circuits"].append({
            "name": f"transpiled_{mapping['circuits'][k]['name']}",
            "file": f"transpiled_circuit_{k+1}.qpy",
            "measurement": mapping["circuits"][k]["measurement"]
        })

    if args.verbose:
        plt.show()

if __name__ == "__main__":
    main()