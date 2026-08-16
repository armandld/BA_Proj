#!/bin/bash
# ============================================================
# V3 Task 10 - Orchestration (protocole v3, section 5.4)
#
# `bash scripts/run_study_v3.sh --all` regenere chaque chiffre
# titre depuis le commit tague, puis produit la table maitresse
# auto-verifiante (study/common/aggregate_v3.py, statut
# OK/DIFF/MISSING par ligne contre les references publiees).
#
# ETAT : ce lanceur datait de deux reorganisations en arriere. Les
# quinze chemins qu'il nommait pointaient TOUS dans le vide
# (study/v3/, study/results/, tests/v3/), et sa seule etape qui
# "passait" etait un pytest sur un dossier vide -- un balayage vide
# qui rendait 0. Chemins repointes, ROOT_DIR corrige.
# tests/lint/test_scripts_point_somewhere.py (D-116) interdit la recidive.
#
# Pre-requis : results/ contient les donnees phases 1-2
# (dns_*.npz + patches_*.npz). La regeneration des DONNEES n'est
# pas incluse (heures de DNS) : utiliser study/pipeline/dns_extension.py.
#
# Usage :
#   bash scripts/run_study_v3.sh --all          # tout (~30-40 min)
#   bash scripts/run_study_v3.sh --skip-t6      # sans le pilote t6
#   bash scripts/run_study_v3.sh --only t4 t9   # sous-ensemble
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# `..` et non `../..` : le script vivait dans study/v3/, il vit dans
# scripts/. Laisse a `../..`, ROOT_DIR designait le PARENT du depot et
# chaque chemin construit dessous etait faux.
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$ROOT_DIR/results"
LOG_DIR="$ROOT_DIR/results/logs_v3"
STAMP="$(date +'%Y-%m-%d_%H-%M-%S')"
mkdir -p "$LOG_DIR"

N=256
DIM=4
SKIP_T6=false
ONLY=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) shift ;;
        --skip-t6) SKIP_T6=true; shift ;;
        --N) N="$2"; shift 2 ;;
        --dim) DIM="$2"; shift 2 ;;
        --only) shift; while [[ $# -gt 0 && "$1" != --* ]]; do
                    ONLY+=("$1"); shift; done ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

want() {
    # execute l'etape si --only absent ou si elle y figure
    if [[ ${#ONLY[@]} -eq 0 ]]; then return 0; fi
    for o in "${ONLY[@]}"; do [[ "$o" == "$1" ]] && return 0; done
    return 1
}

run_step() {
    local name="$1"; shift
    local log="$LOG_DIR/${STAMP}_${name}.log"
    echo ""
    echo "=== [$name] $* ==="
    echo "    log: $log"
    "$@" 2>&1 | tee "$log"
}

# ---- garde : donnees phases 1-2 presentes ----
if [[ ! -f "$RESULTS_DIR/dns_orszag_tang_Re400_N${N}.npz" ]]; then
    echo "ERROR: results/ has no phase-1 DNS data at N=$N."
    echo "Generate it first (hours): python study/pipeline/dns_extension.py"
    exit 1
fi

echo "V3 regeneration run — N=$N dim=$DIM  (commit $(git -C "$ROOT_DIR" rev-parse --short HEAD))"

# ---- gate : suite de tests ----
want tests && run_step tests python -m pytest "$ROOT_DIR/tests/study" -q

# ---- task 0 : regression V2 (scripts V2 intouches) ----
want task0 && run_step task0a python "$ROOT_DIR/study/h2b_prediction/h2b_ceiling_random_split.py" --N "$N" --dim "$DIM"
want task0 && run_step task0b python "$ROOT_DIR/study/h2b_prediction/h2b_loso_transfer.py" --N "$N" --dim "$DIM"

# ---- taches v3 (t2/t3 sont des bibliotheques : validees par pytest) ----
want t1  && run_step t1  python "$ROOT_DIR/study/h2b_prediction/h2b_feature_selection.py" --N "$N" --dim "$DIM"
want t1b && run_step t1b python "$ROOT_DIR/study/h2b_prediction/h2b_neighbour_cone_curve.py" --N "$N" --dim "$DIM"
want t4  && run_step t4  python "$ROOT_DIR/study/h2b_prediction/h2b_blocked_split.py" --N "$N" --dim "$DIM"
want t5  && run_step t5  python "$ROOT_DIR/study/h2b_prediction/h2b_psi_feature_loso.py" --N "$N" --dim "$DIM"
if ! $SKIP_T6 && want t6; then
    run_step t6 python "$ROOT_DIR/study/h2b_prediction/h2b_dynamic_ground_truth.py"   # pilote
fi
want t7 && run_step t7 python "$ROOT_DIR/study/h2b_prediction/h2b_prediction_horizon.py" --N "$N" --dim "$DIM"
want t9 && run_step t9 python "$ROOT_DIR/study/h3_representation/h3_locality_proposition.py" --N "$N" --dim 2 4

# ---- table maitresse auto-verifiante ----
want t10 && run_step t10 python "$ROOT_DIR/study/common/aggregate_v3.py" --N "$N" --dim "$DIM" --strict

echo ""
echo "V3 regeneration complete. Master table: results/v3_master_table.md"
