import json
import subprocess
import os
import argparse
import sys


import numpy as np
import matplotlib.pyplot as plt

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.refinement import refinement

from visual import plot_amr_state

def call_vqa_shell(angles_tuple, args, script_path="run_VQA_pipeline.sh"):
    """
    Appelle le script shell externe pour le calcul VQA.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_path)
    
    data_dir = os.path.abspath(os.path.join(current_dir, "../data"))

    # SECURITÉ : On crée le dossier data s'il n'existe pas encore
    os.makedirs(data_dir, exist_ok=True)

    input_file = os.path.join(data_dir, "vqa_input.json")
    output_file = os.path.join(data_dir, "vqa_output.json")
    
    # 1. Sérialisation des données (Angles -> JSON)
    # Les tableaux numpy ne sont pas sérialisables directement, on utilise .tolist()
    data = {
        "theta_h": angles_tuple[0].tolist(),
        "theta_v": angles_tuple[1].tolist(),
        "psi_h": angles_tuple[2].tolist(),
        "psi_v": angles_tuple[3].tolist()
    }
    
    with open(input_file, "w") as f:
        json.dump(data, f)
    
    
    # 2. Construction de la commande Shell
    # On transmet les paramètres quantiques reçus par le main
    cmd = [
        "bash", script_path,
        "--in-file", input_file,
        "--out-dir", data_dir,
        "--out-file", output_file,
        "--backend", args.backend,
        "--method", args.method,
        "--mode", args.mode,
        "--opt_level", str(args.opt_level),
        "--shots", str(args.shots),
        "--depth", str(args.depth),
        "--numqbits", str(args.grid_size * args.grid_size * 2),
    ]
    
    if args.verbose:
        cmd.append("--verbose")

    # 3. Exécution
    try:
        if args.verbose:
            print(f"[Python] Calling VQA Shell...")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur critique dans le VQA Shell: {e}")
        return None

    # 4. Récupération
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            probs_list = json.load(f)
        return np.array(probs_list)
    else:
        print("❌ Erreur: Pas de fichier de sortie VQA.")
        return None