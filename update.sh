#!/bin/bash

# Le nom du fichier qui définit l'environnement
ENV_FILE="environment.yaml"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Erreur : Fichier '$ENV_FILE' introuvable."
    echo "Veuillez vous assurer que $ENV_FILE est dans ce répertoire."
    exit 1
fi

# 1. Lire le nom de l'environnement directement depuis le fichier .yml
ENV_NAME=$(grep 'name:' $ENV_FILE | cut -d ' ' -f 2)

if [ -z "$ENV_NAME" ]; then
    echo "❌ Erreur : Impossible de lire 'name:' depuis $ENV_FILE."
    exit 1
fi

echo "🔹 Mise à jour du gestionnaire Conda (base)..."
conda update -n base -c defaults conda -y

echo "🔹 Synchronisation de l'environnement '$ENV_NAME' avec $ENV_FILE..."

# C'est la commande clé.
# 'conda env update' lit le fichier et met à jour l'environnement pour qu'il corresponde.
# '--prune' supprime tous les packages de l'environnement qui ne sont PAS listés
# dans le fichier .yml, gardant votre environnement parfaitement propre.
conda env update --name $ENV_NAME --file $ENV_FILE --prune

echo "✅ L'environnement '$ENV_NAME' est synchronisé avec $ENV_FILE !"
echo "Pour ajouter ou supprimer des packages, modifiez $ENV_FILE et relancez ce script."