# Research Project, Armand Le Douarec
# Q-HAS: Quantum-Hierarchical Adaptive Steering

I propose a hybrid quantum-classical architecture designed to optimize the simulation of magnetohydrodynamics (MHD) instabilities. Instead of solving the full dynamical equations on a fine grid, we utilize
a Variational Quantum Algorithm (VQA) to identify topological defects (flux reconnections, turbulence
onset) on a coarse-grained graph. This allows for targeted Adaptive Mesh Refinement (AMR).

This repo contains all the code of my project, using  **Python**, **C++** and **Qiskit**. This README explains how to configure the env of development to run the code and how to run it. 
---

## 1️⃣ Structure du dépôt

```bash
BA_Proj/
├── LICENSE
├── README.md
├── TO DO LIST.rtf
├── TrainHP_GoogleColab.sh
├── TrainHyperParams.sh
├── Train_results
│   ├── best_hyperparams.json
│   └── optuna_study.db
├── advanced_project_idea
│   ├── Idée pour améliorer projet.rtf
│   ├── Remarques.rtf
│   └── papers
├── algos_test_MHD
│   ├── Variational
│   └── helloworld.py
├── best_hyperparams.json
├── data
├── environment.yaml
├── logs
│   ├── pipeline[2026-01-13_13-23-39].log
│   ├── pipeline[2026-01-13_13-25-19].log
│   ├── pipeline[2026-01-13_13-27-17].log
│   ├── pipeline[2026-01-22_03-07-53].log
│   ├── pipeline[2026-01-22_03-08-59].log
│   ├── pipeline[2026-01-22_03-18-06].log
│   ├── pipeline[2026-01-22_03-19-56].log
│   ├── pipeline[2026-01-25_17-24-57].log
│   ├── pipeline[2026-01-25_17-26-40].log
│   ├── pipeline[2026-01-25_17-27-59].log
│   └── pipeline[2026-01-25_17-28-30].log
├── notebooks
│   ├── A faire_developper.rtf
│   ├── ALA.pdf
│   ├── Feynman_formalism.pdf
│   ├── MAIN.pdf
│   ├── MAIN2.pdf
│   ├── Q_adv_for_DE.pdf
│   ├── VQA_research.pdf
│   ├── ZGR_QFT.pdf
│   ├── alternatin_prep.pdf
│   ├── nonlin2_ex.pdf
│   └── uniform_prep_controlled_rot.pdf
├── run_pipeline.sh
├── setup_env.sh
├── src
│   ├── Simulation
│   ├── TrainHyperParam.py
│   ├── VQA
│   ├── call_vqa_shell.py
│   ├── help_visual.py
│   ├── logs
│   ├── patches.py
│   ├── pipeline.py
│   ├── run_VQA_pipeline.sh
│   ├── utils.py
│   └── visual.py
├── tutos
│   ├── Max_cut
│   ├── VQA
│   └── helloworld.py
└── update.sh

```

The ```bash src/``` folder will contain the codes executed for this project.
The ```bash log/``` folder will contain the console logs of each test, with date & time of the log.
The ```bash results/``` folder will contain the outputs (&inputs if reused) of the files from the ```bash src/``` folder.

In fact, a similar structure is already presented in the ```bash tutos/Max_cut/``` folder.

---

## 2️⃣ Prérequis

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou [Anaconda](https://www.anaconda.com/)
- Git
- macOS, Linux ou Windows (avec WSL pour Windows)
- CMake et un compilateur C++ (gcc / clang)
- Python 3.11 recommandé

---

# Installation

## Installation using .sh files (recommended)

```bash
source setup_env.sh
```
## Update environment

```bash
source update_env.sh
```
# Train the hyperparameters guiding the Q-HAS algorithm using optuna:

In order to guide the optimization of the parameters, all hyperparameters for the optimization (of the hyperparameters of the Q-HAS) are at the top of the ```bash setup_env.sh ```.

```bash
bash TrainHyperParams.sh
```

Depending on which type of flux anomalies you are trying to identify through the Q-HAS, training only the adequate parameters of the Hamiltonian greatly shorten the timelength of the optimization.

Of course, if the type of anomalies is not know training all of it remains the best option.

# Launch the pipeline (all hyperparameters set):
By decomposing the major parts of the computation, here is a result of a viable pipeline to use:

```bash
bash run_pipeline.sh --backend aer --grid-size 2 --opt-level 1 --depth 1 --dns-resolution 256 --shots 1000 --t-max 9e-4 --dt 1e-4 --hybrid-dt 4e-4 --verbose
```

The parameters describe the following in the Q-HAS:

Options : 

```bash  --backend <aer|estimator> ```      Quantum backend (default: aer)

```bash  --mode <simulator|hardware> ```    Simulator or IBM Quantum (default: simulator)

```bash  --shots <int> ```                  Number of shots (default: 1024)

```bash  --numqbits <int> ```               Number of qubits (default: 4)

```bash  --depth <int> ```                  Depth of the ULA ansatz (default: 2)

```bash  --opt_level <0|1|2|3> ```          Optimization level for transpiler (default: 3)

```bash  --out-dir <dir>  ```               Output directory (default: data)

```bash  --verbose ```                      Enable verbose logging

```bash  --skip-cleanup ```                 Skip deleting previous data

```bash  --method <COBYLA|Nelder-Mead|Powell|L-BFGS-B> ``` Optimization method for minimize (default: COBYLA)

Custom Domain Parameters:

```bash  --grid-size <int> ```              Coarse grid dimension N (NxN) (default: 16)

```bash  --dns-resolution <int> ```         High-Res Grid for Ground Truth (default: 256)

```bash  --t-max <float> ```                Simulation end time (default: 1.0)

```bash  --dt <float> ```                   Time step size (default: 0.01)

```bash  --hybrid-dt <float> ```              Hybrid simulation time step size (default: 0.1)

```bash  --AdvAnomaliesEnable ```            Enable advanced anomaly handling in mapping

Stage control (choose one):

```bash  --only-mapping ```                 Run mapping stage only

```bash  --only-optimize ```                Run optimization stage only

```bash  --only-execute ```                 Run execution stage only

```bash  --only-postprocess ```             Run post-processing stage only


## Details if installation using .sh files do not work:

### Try the following

```bash
conda env create -f environment.yml
```

### To activate the venv 

Take a look at the file ```bash environment.yml```, a venv name should be visible.

Usually:

```bash
conda activate qiskit-project # TO ADAPT IF CHANGED
```

## Details if update using .sh files do not work:

Before updating the environment, it is recommended to do a backup of packages in order to allow yourself to reset to an older viable version if necessary :


### Update conda gestion

```bash
conda update -n base -c defaults conda -y
```

### Update the list :

Usually :

```bash
conda env update --name qiskit-project --file environment.yaml --prune # TO ADAPT IF CHANGED
```

# Launch a tutorial
By decomposing the major parts of the computation, here is a result of a viable pipeline to use:

once in the project in ```bash BA_proj/```, type

```bash
bash tutos/Max_cut/run_pipeline.sh --backend aer --nodes 10 --edges 12 --mode simulator --verbose
```
