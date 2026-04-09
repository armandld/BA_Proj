import argparse
import warnings
import json
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit.transpiler import generate_preset_pass_manager


def optimize(qc, backend, opt_level, verbose):

    if backend == "aer":
        backend = AerSimulator()
    elif backend == "estimator":
        backend = FakeFez()
    elif backend == "state_vector":
        backend = AerSimulator(method='statevector')
        opt_level = 0
    else:
        raise ValueError("Unsupported backend")

    # Create pass manager for transpilation
    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)

    circuit = pm.run(qc)
    if verbose:
        print("Optimization Level: ", opt_level)
    
    return circuit