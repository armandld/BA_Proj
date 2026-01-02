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

def postprocess(distribution, num_qubits, verbose):
    """
    Convertit {string: count} en liste [P(q0), P(q1)...]
    Nécessaire pour le décodage AMR.
    """
    hits = np.zeros(num_qubits)

    for bitstring, count in distribution.items():
        # On parcourt la chaîne. 
        # Convention: bitstring[0] correspond au Qubit 0 ici (ordre gauche->droite)
        for i, bit in enumerate(bitstring):
            if i < num_qubits and bit == '1':
                hits[i] += count
                
    marginals = (hits).tolist()
    if verbose:
        print(f"Marginals: {marginals}")
    
    return marginals

