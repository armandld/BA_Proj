#!/bin/bash
# ============================================================
# V3 Task 10 - Orchestration (protocole v3, section 5.4)
#
# NE FONCTIONNE PLUS (D-49, docs/DEFAUTS.md). Ce script invoque neuf
# generateurs (t1_feature_selection.py, t1b_cone_curve.py,
# t4_blocked_split.py, t5_v1_psi_loso.py, t6_dynamic_gt.py,
# t7_horizon.py, t9_prop2_check.py, phase11_upper_bound.py,
# phase11b_loso.py) qui n'existent plus dans ce depot, sous un chemin
# `study/v3/` qui n'existe plus non plus (voir CLAUDE.md pour
# l'arborescence actuelle : les generateurs restants vivent dans
# `study/<hypothese>/`, la table dans `study/common/aggregate_v3.py`).
# Le garde-fou ci-dessous echoue avant meme d'atteindre ce probleme,
# `study/results/` n'existant pas non plus (les donnees sont a la
# racine, `results/`).
#
# Conserve pour l'historique de la commande documentee au protocole ;
# ne pas debugger les chemins en pensant le remettre en etat sans lire
# D-49 d'abord — la question qu'il pose (reconstruire les 9 generateurs,
# ou archiver ce script et aggregate_v3.py) n'est pas tranchee.
#
# `bash study/v3/run_study_v3.sh --all` regenere chaque chiffre
# titre depuis le commit tague, puis produit la table maitresse
# (statut OK/DIFF/MISSING par ligne contre le baseline V3 ARCHIVE,
# pas contre docs/RESULTS.md — voir D-49).
#
# Usage (documentaire seulement) :
#   bash study/v3/run_study_v3.sh --all          # tout (~30-40 min)
#   bash study/v3/run_study_v3.sh --skip-t6      # sans le pilote t6
#   bash study/v3/run_study_v3.sh --only t4 t9   # sous-ensemble
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$ROOT_DIR/study/results"
LOG_DIR="$ROOT_DIR/logs/v3"
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
    echo "ERROR: study/results/ has no phase-1 DNS data at N=$N."
    echo "Generate it first (hours): python study/v3/dns_extension.py"
    exit 1
fi

echo "V3 regeneration run — N=$N dim=$DIM  (commit $(git -C "$ROOT_DIR" rev-parse --short HEAD))"

# ---- gate : suite de tests ----
want tests && run_step tests python -m pytest "$ROOT_DIR/tests/v3" -q

# ---- task 0 : regression V2 (scripts V2 intouches) ----
want task0 && run_step task0a python "$ROOT_DIR/study/phase11_upper_bound.py" --N "$N" --dim "$DIM"
want task0 && run_step task0b python "$ROOT_DIR/study/phase11b_loso.py" --N "$N" --dim "$DIM"

# ---- taches v3 (t2/t3 sont des bibliotheques : validees par pytest) ----
want t1  && run_step t1  python "$ROOT_DIR/study/v3/t1_feature_selection.py" --N "$N" --dim "$DIM"
want t1b && run_step t1b python "$ROOT_DIR/study/v3/t1b_cone_curve.py" --N "$N" --dim "$DIM"
want t4  && run_step t4  python "$ROOT_DIR/study/v3/t4_blocked_split.py" --N "$N" --dim "$DIM"
want t5  && run_step t5  python "$ROOT_DIR/study/v3/t5_v1_psi_loso.py" --N "$N" --dim "$DIM"
if ! $SKIP_T6 && want t6; then
    run_step t6 python "$ROOT_DIR/study/v3/t6_dynamic_gt.py"   # pilote
fi
want t7 && run_step t7 python "$ROOT_DIR/study/v3/t7_horizon.py" --N "$N" --dim "$DIM"
want t9 && run_step t9 python "$ROOT_DIR/study/v3/t9_prop2_check.py" --N "$N" --dim 2 4

# ---- table maitresse auto-verifiante ----
want t10 && run_step t10 python "$ROOT_DIR/study/v3/aggregate_v3.py" --N "$N" --dim "$DIM" --strict

echo ""
echo "V3 regeneration complete. Master table: study/results/v3_master_table.md"
