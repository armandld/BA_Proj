# BA_Proj
Quantum Computation on Plasma Physics

Ce dépôt contient le code pour notre projet, utilisant **Python**, **C++** et **Qiskit**. Ce README explique comment configurer l’environnement de développement pour tous les membres de l’équipe et comment utiliser le code.

---

## 1️⃣ Structure du dépôt

```bash
BA_Proj/
├── LICENSE
├── README.md
├── conda-packages.txt
├── cpp
│   └── hello.cpp
├── notebooks
├── pip-packages.txt
└── python
```

---

## 2️⃣ Prérequis

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou [Anaconda](https://www.anaconda.com/)
- Git
- macOS, Linux ou Windows (avec WSL pour Windows)
- CMake et un compilateur C++ (gcc / clang)
- Python 3.12 recommandé

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

## Details if installation using .sh files do not work:

### 1️⃣ Créer un nouvel environnement vide avec la version de Python souhaitée

```bash
# Remplacez mon_env par le nom souhaité pour votre environnement
ENV_NAME="mon_env"
PYTHON_VERSION="3.12"

# Créer l'environnement
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Important pour que le shell reconnaisse conda activate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME
```

### 2️⃣ Installer tous les packages Conda listés

```bash
conda install -y --file conda-packages.txt
```

### 3️⃣ Installer tous les packages pip listés (en ignorant les fichiers locaux)

```bash
# Filtrer les packages locaux pour éviter les erreurs
grep -v "file://" pip-packages.txt | xargs -n 1 pip install

# Vérifier et installer Qiskit si nécessaire
pip install --upgrade qiskit
```
### To activate the environment

```bash
conda activate $ENV_NAME
```

## Details if update using .sh files do not work:

Avant de mettre à jour l’environnement, il est recommandé de faire un backup des packages pour pouvoir revenir à une version stable si nécessaire :

### 1️⃣ Sauvegarder la liste des packages Conda

```bash
conda list --export | grep -v "@" > conda-packages-backup.txt
```

### 2️⃣ Sauvegarder la liste des packages pip (en ignorant les fichiers locaux)

```bash
pip freeze | grep -v "file://" > pip-packages-backup.txt
```

### 3️⃣ Mettre à jour Conda

```bash
conda update -n base -c defaults conda -y
```

### 4️⃣ Mettre à jour tous les packages Conda

```bash
conda update --all -y
```

### 5️⃣ Mettre à jour les packages pip obsolètes individuellement

```bash
# Lire ligne par ligne pour éviter les erreurs avec les retours chariot
pip list --outdated --format=columns | tail -n +3 | awk '{print $1}' | tr -d '\r' | while read pkg; do
    echo "Mise à jour de $pkg..."
    pip install --upgrade "$pkg"
done

# Vérification et mise à jour de Qiskit
pip install --upgrade qiskit
```