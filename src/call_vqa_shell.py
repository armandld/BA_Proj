import json
import subprocess
import os
import argparse
import sys


import numpy as np
import matplotlib.pyplot as plt

from VQA.TEST1 import TEST1

from VQA.mapping import mapping

from VQA.optimize import optimize

from VQA.execute import execute

from VQA.postprocess import postprocess

def call_vqa_shell(angles_tuple, hamilt_params, args, period_bound=True):
    
    # 1. Sérialisation des données (Angles -> JSON)
    # Les tableaux numpy ne sont pas sérialisables directement, on utilise .tolist()
    data = {
        "theta_h": angles_tuple[0].tolist(),
        "theta_v": angles_tuple[1].tolist(),
        "psi_h": angles_tuple[2].tolist(),
        "psi_v": angles_tuple[3].tolist()
    }
    reps = 2
    qc, cost_hamiltonian = mapping(data, hamilt_params, args.AdvAnomaliesEnable, period_bound, reps)
    qc = optimize(qc, args.backend, args.opt_level, args.verbose)
    probs_list = execute(qc, cost_hamiltonian, args.mode, args.backend, args.shots, reps, args.verbose)
    probs_list = postprocess(probs_list, qc.num_qubits, args.verbose)
    #probs_list = TEST1(data, args.backend, args.shots, period_bound, args.grid_size * args.grid_size * 2)
    return np.array(probs_list)