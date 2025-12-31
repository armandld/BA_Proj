import json
import subprocess
import os
import argparse
import sys


import numpy as np
import matplotlib.pyplot as plt

from VQA.TEST1 import TEST1

def call_vqa_shell(angles_tuple, hamilt_params, args, period_bound=True):
    
    # 1. Sérialisation des données (Angles -> JSON)
    # Les tableaux numpy ne sont pas sérialisables directement, on utilise .tolist()
    data = {
        "theta_h": angles_tuple[0].tolist(),
        "theta_v": angles_tuple[1].tolist(),
        "psi_h": angles_tuple[2].tolist(),
        "psi_v": angles_tuple[3].tolist()
    }
    print("Paramètres physiques de l'hamiltonien :", hamilt_params)
    probs_list = TEST1(data, hamilt_params, args.shots, period_bound)
    return np.array(probs_list)