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
import random
from networkx.readwrite import json_graph

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import qpy


# -----------------------------
# Utility: Build MaxCut Hamiltonian
# -----------------------------
def create_maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """Construct the MaxCut Hamiltonian H = 1/2 Σ_ij w_ij (I - Z_i Z_j)."""
    num_qubits = graph.number_of_nodes()
    pauli_terms = []

    for i, j, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)
        z_term = ["I"] * num_qubits
        z_term[i] = "Z"
        z_term[j] = "Z"
        pauli_terms.append(("".join(z_term), -0.5 * weight))
        pauli_terms.append(("I" * num_qubits, 0.5 * weight))  # Constant offset

    return SparsePauliOp.from_list(pauli_terms)


# -----------------------------
# Main entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Map MaxCut problem to Hamiltonian operator")
    parser.add_argument("--backend", default="aer", choices=["aer", "estimator"], help="Quantum backend")
    parser.add_argument("--out-dir", default="results", help="Output directory for mapping")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes for the random graph.")
    parser.add_argument("--edges", type=int, required=True, help="Number of random edges to create for the graph.")
    

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------
    # Example: Graph construction
    # -----------------------------
    # Number of nodes
    n = args.nodes

    # Number of random edges you want
    num_edges = args.edges

    # Seed for reproducibility
    seed = 2
    
    random.seed(seed)
    np.random.seed(seed)

    # Create empty graph
    G = nx.Graph()
    G.add_nodes_from(np.arange(0, n, 1))

    # Generate all possible edges without self-loops
    possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]

    # Randomly select edges
    random_edges = random.sample(possible_edges, num_edges)

    # Add edges with weight 1
    weighted_edges = [(u, v, 1.0) for u, v in random_edges]
    G.add_weighted_edges_from(weighted_edges)

    

    if args.verbose:
        print(f"🔹 Using {args.backend.upper()} backend")
        print("🔹 Graph edges:", list(G.edges))
        # Draw the graph
        pos = nx.spring_layout(G, seed=seed)  # layout with same seed
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=14)
        plt.show()  # non-blocking because interactive mode is on

    # -----------------------------
    # Map to Hamiltonian
    # -----------------------------
    num_qubits = G.number_of_nodes()

    def build_max_cut_paulis(graph: nx.Graph) -> list[tuple[str, float]]:
        """
        Convert a NetworkX graph to a list of Pauli terms for MaxCut Hamiltonian.
        Returns a list of (pauli_string, coefficient).
        """
        num_qubits = graph.number_of_nodes()
        pauli_terms = []

        for i, j, data in graph.edges(data=True):
            weight = data.get("weight", 1.0)
            z_term = ["I"] * num_qubits
            z_term[i] = "Z"
            z_term[j] = "Z"
            pauli_terms.append(("".join(z_term), 0.5 * weight))
            pauli_terms.append(("I" * num_qubits, -0.5 * weight))  # constant offset

        return pauli_terms
    
    
    pauli_list = build_max_cut_paulis(G)
    cost_hamiltonian = SparsePauliOp.from_list(pauli_list)

    # -----------------------------
    # Prepare data for JSON
    # -----------------------------
    hamiltonian_terms = []
    for label, coeff in cost_hamiltonian.to_list():
        # Convert complex → float safely
        real_coeff = float(np.real_if_close(coeff))
        hamiltonian_terms.append([label, real_coeff])

    edges = [(int(u), int(v)) for u, v in G.edges]

    mapping_data = {
        "backend": args.backend,
        "num_qubits": num_qubits,
        "edges": edges,
        "hamiltonian": hamiltonian_terms,
    }

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
    circuit.measure_all()
    
    circuit.draw("mpl")

    if args.verbose:
        print("cost_hamiltonian : ",cost_hamiltonian)
        plt.show()
    
    # -----------------------------
    # Save mapping to JSON
    # -----------------------------
    out_file = os.path.join(args.out_dir, "mapping.json")
    with open(out_file, "w") as f:
        json.dump(mapping_data, f, indent=2)
    circuit_file = os.path.join(args.out_dir, "qaoa_circuit.qpy")

    with open(circuit_file, "wb") as f:
        qpy.dump(circuit, f)  # saves the circuit to a binary file
    
    # Convert graph to a JSON-serializable format
    graph_data = json_graph.node_link_data(G)

    # Convert all node ids to int and edge weights to float
    for node in graph_data["nodes"]:
        node["id"] = int(node["id"])
    for edge in graph_data["links"]:
        edge["source"] = int(edge["source"])
        edge["target"] = int(edge["target"])
        if "weight" in edge:
            edge["weight"] = float(edge["weight"])

    graph_file = os.path.join(args.out_dir, "graph.json")
    with open(graph_file, "w") as f:
        json.dump(graph_data, f, indent=2)

    if args.verbose:
        print(f"✅ Mapping complete. Saved to {out_file}")
        print("Hamiltonian terms:")
        for label, coeff in hamiltonian_terms:
            print(f"  {label} : {coeff}")


if __name__ == "__main__":
    main()
