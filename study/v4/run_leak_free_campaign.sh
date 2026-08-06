#!/usr/bin/env bash
# Pilote la campagne leak-free jusqu'a son terme, fold par fold.
#
# POURQUOI CE SCRIPT
# Un fold coute ~4 h sur `kh` et `ot`, le conteneur est recycle toutes les
# ~1 h 30, et chaque relance devait etre faite a la main. `t22` sait
# desormais reprendre depuis son point de sauvegarde : il ne manquait que
# la boucle qui relance. Sans elle, chaque interruption attendait une
# intervention, et c'est ainsi que ces deux folds ont ete perdus trois fois.
#
# Le script relance tant que l'artefact n'est pas TERMINE. « Termine » veut
# dire `status` valant `completed` ou `total_abort` — un avortement total
# est un resultat, pas un echec, et relancer dessus tournerait en boucle.
# Un etat `partial` signifie interrompu : on relance, et `t22` reprend.
#
# Garde-fou : un nombre maximal de tentatives par fold, pour qu'une panne
# reproductible ne relance pas indefiniment en silence.
#
# Usage :
#   bash study/v4/run_leak_free_campaign.sh kh ot
#   MAX_ATTEMPTS=20 bash study/v4/run_leak_free_campaign.sh ot
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RESULTS="$ROOT/study/results"
LOGDIR="${LEAK_FREE_LOGDIR:-$ROOT/logs/v4}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"
REPEATS="${REPEATS:-5}"

mkdir -p "$LOGDIR"

fold_status() {
    python - "$1" <<'PY'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print("absent"); sys.exit()
try:
    print(json.load(open(p)).get("status", "unknown"))
except ValueError:
    print("unreadable")
PY
}

for fold in "$@"; do
    art="$RESULTS/t22_unseen_leak-free_${fold}.json"
    log="$LOGDIR/t22lf_${fold}.log"
    attempt=0
    while :; do
        st="$(fold_status "$art")"
        if [ "$st" = "completed" ] || [ "$st" = "total_abort" ]; then
            echo "[campaign] $fold: DONE (status=$st)"
            break
        fi
        attempt=$((attempt + 1))
        if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
            echo "[campaign] $fold: GIVING UP after $MAX_ATTEMPTS attempts" \
                 "(status=$st) — a reproducible failure, not a reclaim"
            break
        fi
        echo "[campaign] $fold: attempt $attempt (status=$st)"
        python "$HERE/t22_unseen_conditions.py" --fold "$fold" \
            --mode leak-free --repeats "$REPEATS" --matched-reference \
            >> "$log" 2>&1
        rc=$?
        echo "[campaign] $fold: exited rc=$rc"
        # rc != 0 sur un avortement total : la boucle le detecte au tour
        # suivant via `status`, pas via le code retour.
        sleep 5
    done
done
echo "[campaign] finished"
