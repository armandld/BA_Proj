#!/bin/bash
# V4 - Level 3 : execute un fold complet (t15 puis t15b) dans un processus
# dedie, BLAS mono-thread pour permettre l'execution parallele d'un fold par
# coeur sans contention.
#
# Reprise : le stockage Optuna persistant (study/results/t15_level3_optuna_
# {fold}.db) rend la reprise essai-par-essai automatique ; les checkpoints
# JSON evitent de re-tuner un fold deja regle. Relancer la meme commande
# apres une interruption reprend au dernier essai enregistre.
#
# Usage :
#   bash scripts/run_fold.sh kh
#   for f in kh rotor tearing; do
#       nohup bash scripts/run_fold.sh "$f" > logs/v4/fold_$f.log 2>&1 &
#   done
set -u
f="${1:?usage: run_fold.sh <fold>}"
n_trials="${2:-4}"
n_trials_classical="${3:-2}"

# D-71 : ce script vivait sous study/v4/ (deux niveaux sous la racine) au
# moment ou "../.." a ete ecrit ; le deplacement vers scripts/ (un seul
# niveau) l'a laisse pointer au-dessus de la racine du depot, ou "study/"
# n'existe pas — mesure : cd y reussit (le dossier parent existe), mais
# `python study/...` y echoue avec FileNotFoundError.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# un thread BLAS par processus : la parallelisation se fait au niveau fold
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1

echo "=== [$f] start $(date -u +%H:%M:%S) ==="
python study/closed_loop/closed_loop_campaign.py --folds "$f" \
    --n-trials "$n_trials" --n-trials-classical "$n_trials_classical"
echo "=== [$f] t15 done $(date -u +%H:%M:%S) ==="
python study/closed_loop/closed_loop_budget_matched.py --fold "$f" --max-iter 4
echo "=== [$f] COMPLETE $(date -u +%H:%M:%S) ==="
