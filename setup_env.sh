#!/bin/bash

# Nom du fichier d'environnement
ENV_FILE="environment.yaml"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Erreur : Fichier '$ENV_FILE' introuvable."
    echo "Veuillez créer un fichier environment.yaml d'abord."
    exit 1
fi

echo "🔹 Création de l'environnement Conda depuis $ENV_FILE..."
# Cette seule commande crée l'environnement ET installe
# tous les packages (conda et pip) listés dans le fichier.
conda env create -f "$ENV_FILE"

# Récupérer le nom de l'environnement depuis le fichier .yaml pour l'afficher
ENV_NAME=$(grep 'name:' "$ENV_FILE" | cut -d ' ' -f 2)

echo "✅ L'environnement '$ENV_NAME' est prêt !"
echo "Pour l'activer : conda activate $ENV_NAME"