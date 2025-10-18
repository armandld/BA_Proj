#!/bin/bash

# Nom de l'environnement
ENV_NAME="mon_env"

echo "🔹 Activation de l'environnement..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "🔹 Sauvegarde des packages existants..."
conda list --export | grep -v "@" > conda-packages-backup.txt
pip freeze | grep -v "file://" > pip-packages-backup.txt

echo "🔹 Mise à jour de Conda..."
conda update -n base -c defaults conda -y

echo "🔹 Mise à jour de tous les packages Conda..."
conda update --all -y

echo "🔹 Mise à jour des packages pip obsolètes..."
# Lire ligne par ligne pour éviter les retours chariot collés
pip list --outdated --format=columns | tail -n +3 | awk '{print $1}' | tr -d '\r' | while read pkg; do
    echo "Mise à jour de $pkg..."
    pip install --upgrade "$pkg"
done

echo "🔹 Vérification et mise à jour de Qiskit..."
pip install --upgrade qiskit

echo "✅ Mise à jour terminée !"
