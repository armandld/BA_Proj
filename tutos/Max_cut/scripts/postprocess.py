# scripts/postprocess.py
import argparse
import json
import matplotlib
import os
from typing import Sequence
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph

def main():
    parser = argparse.ArgumentParser(description="Post-process QAOA results")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load execution results
    out_file = f"{args.out_dir}/execution_result.json"
    with open(out_file, "r") as f:
        data = json.load(f)

    graph_file = os.path.join(args.out_dir, "graph.json")
    with open(graph_file, "r") as f:
        graph_data = json.load(f)

    G = json_graph.node_link_graph(graph_data, edges="links")

    # auxiliary functions to sample most likely bitstring
    def to_bitstring(integer, num_bits):
        # integer must be an int
        result = np.binary_repr(integer, width=num_bits)
        return [int(digit) for digit in result]
    
    
    keys = list(data["final_distribution_int"].keys())
    values = list(data["final_distribution_int"].values())
    most_likely = keys[np.argmax(np.abs(values))]
    most_likely_int = int(most_likely)  # decimal integer
    most_likely_bitstring = to_bitstring(most_likely_int, len(G))
    most_likely_bitstring.reverse()
    
    print("Result bitstring:", most_likely_bitstring)

    matplotlib.rcParams.update({"font.size": 10})
    final_bits = data["final_distribution_bin"]
    values = np.abs(list(final_bits.values()))
    top_4_values = sorted(values, reverse=True)[:4]
    positions = []
    for value in top_4_values:
        positions.append(np.where(values == value)[0])
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks(rotation=45)
    plt.title("Result Distribution")
    plt.xlabel("Bitstrings (reversed)")
    plt.ylabel("Probability")
    ax.bar(list(final_bits.keys()), list(final_bits.values()), color="tab:grey")
    for p in positions:
        ax.get_children()[int(p[0])].set_color("tab:purple")
    plt.show()

    def plot_result(G: nx.Graph, x: Sequence[int]):
        """
        Plot the graph G with node colors based on the bitstring x.
        0 -> grey, 1 -> purple
        """
        colors = ["tab:grey" if bit == 0 else "tab:purple" for bit in x]
        pos = nx.spring_layout(G, seed=42)  # fixed layout for reproducibility
        plt.figure(figsize=(6, 6))
        nx.draw(
            G,
            pos,
            node_color=colors,
            with_labels=True,
            node_size=700,
            font_size=12,
            edge_color="black",
            alpha=0.8
        )
        plt.title("MaxCut Result")
        plt.show()


    def evaluate_sample(x: Sequence[int], graph: nx.Graph) -> float:
        """
        Evaluate the cut value for a bitstring x on the given graph.
        """
        assert len(x) == len(graph.nodes()), "Length of x must match number of nodes"
        return sum(
            x[u] * (1 - x[v]) + x[v] * (1 - x[u])
            for u, v in graph.edges()
        )


    # Example usage:
    cut_value = evaluate_sample(most_likely_bitstring, G)
    print("The value of the cut is:", cut_value)

    plot_result(G, most_likely_bitstring)

if __name__ == "__main__":
    main()
