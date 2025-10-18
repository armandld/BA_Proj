# BA_Proj
Quantum Computation on Plasma Physics

Ce dépôt contient le code pour notre projet, utilisant **Python**, **C++** et **Qiskit**. Ce README explique comment configurer l’environnement de développement pour tous les membres de l’équipe et comment utiliser le code.

---

## 1️⃣ Structure du dépôt

```bash
BA_Proj/
├── LICENSE
├── README.md
├── environment.yaml
├── notebooks
├── setup_env.sh
├── src
├── tutos
│   ├── Max_cut
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

# Launch a tutorial
By decomposing the major parts of the computation, here is a result of a viable pipeline to use:

once in the project in ```bash BA_proj/```, type

```bash
bash tutos/Max_cut/run_pipeline.sh --backend aer --nodes 10 --edges 12 --mode simulator --verbose
```
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

Avant de mettre à jour l’environnement, il est recommandé de faire un backup des packages pour pouvoir revenir à une version stable si nécessaire :

### Update conda gestion

```bash
conda update -n base -c defaults conda -y
```

### Update the list :

Usually :

```bash
conda env update --name qiskit-project --file environment.yaml --prune # TO ADAPT IF CHANGED
```
