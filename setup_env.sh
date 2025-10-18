#!/bin/bash

# Nom de l'environnement
ENV_NAME="mon_env"
PYTHON_VERSION="3.12"

echo "🔹 Création de l'environnement Conda..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

echo "🔹 Activation de l'environnement..."
# Important pour que le script continue à utiliser l'environnement
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "🔹 Installation des packages Conda depuis conda-packages.txt..."
conda install -y --file conda-packages.txt

echo "🔹 Installation des packages pip depuis pip-packages.txt (en ignorant les fichiers locaux)..."
# Filtrer les lignes file:// pour éviter les erreurs
grep -v "file://" pip-packages.txt | xargs -n 1 pip install

echo "🔹 Vérification et installation de Qiskit si nécessaire..."
pip install --upgrade qiskit

echo "✅ L'environnement '$ENV_NAME' est prêt !"
echo "Pour l'activer à l'avenir : conda activate $ENV_NAME"
