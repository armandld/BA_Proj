#!/bin/bash
set -e
set -o pipefail
export LC_NUMERIC=C
# ============================================================
# Generate Comparison Figures — Q-HAS vs Classical AMR
# ============================================================
#
# Usage:
#   ./generate_figures.sh <phase> [--lambda <value>]
#
# Examples:
#   ./generate_figures.sh 1        # Phase 1 (isolated scenarios)
#   ./generate_figures.sh 2        # Phase 2 (complex scenarios)
#   ./generate_figures.sh 3        # Phase 3 (all scenarios)
#   ./generate_figures.sh 3 --lambda 0.40
#
# For each phase, this script:
#   1. Extracts best hyperparams from rescore CSVs (quantum + classical)
#   2. Generates ALL comparison figures
#   3. Saves output to figures/phase<N>/
#
# ============================================================
#
# ETAT MESURE (D-111, docs/RESULTS.md) — CE LANCEUR NE TOURNE PAS EN L'ETAT.
#
# `ROOT_DIR` est corrige ci-dessous (il designait `<depot>/scripts` depuis le
# deplacement du script dans `scripts/`). Mais TROIS cibles n'existent plus
# sous AUCUNE racine, et leur correspondance dans l'arborescence actuelle est
# une DECISION, pas une correction de chemin — elle n'est donc pas faite ici :
#
#   figures_code/            -> le code des figures vit dans `figures/v1_legacy/`
#   Train_results/           -> les campagnes vivent dans
#                               `results/hyperparams/optuna_studies/`
#   best_hyperparams.json    -> le fichier deploye est
#                               `results/hyperparams/best_hyperparams.json`,
#                               une entree GELEE (voir son PROVENANCE.md) :
#                               le regenerer ici l'ecraserait
#
# Le dernier point est le motif de ne pas trancher tout seul : `--output` de
# ce script pointe vers un fichier que le depot declare gele et non
# reproductible. Voir D-22 et D-111 avant de rebrancher quoi que ce soit.
#
# `tests/test_launcher_paths_resolve.py` verifie que cette note reste ici.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# D-111 : ce lanceur vivait a la racine, ou `$SCRIPT_DIR` ETAIT la racine.
# Depuis son deplacement dans `scripts/`, la meme ligne designait
# `<depot>/scripts` : `$ROOT_DIR/scripts/extract_best_hyperparams.py`
# resolvait en `scripts/scripts/...`. Un niveau au-dessus.
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FIGURES_CODE_DIR="$ROOT_DIR/figures_code"
TRAIN_RESULTS_DIR="$ROOT_DIR/Train_results"

# Defaults
LAMBDA_COST=0.40
PHASE=0

# ── Argument parsing ─────────────────────────────────────────
display_help() {
    echo "Usage: $0 <phase> [options]"
    echo ""
    echo "Arguments:"
    echo "  <phase>              Phase number: 1, 2, or 3"
    echo ""
    echo "Options:"
    echo "  --lambda <value>     Lambda cost value (default: 0.40)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Phase definitions:"
    echo "  1  Phase 1 scenarios (KH, Tearing, Orszag-Tang, MHD Rotor)"
    echo "  2  Complex scenarios (Orszag-Tang, MHD Rotor)"
    echo "  3  All 6 scenarios"
}

if [[ $# -lt 1 ]]; then
    display_help
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --lambda)
            LAMBDA_COST="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        -h|--help)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
done

if [[ $PHASE != 1 && $PHASE != 2 && $PHASE != 3 ]]; then
    echo "Error: phase must be 1, 2, or 3 (got '$PHASE')"
    display_help
    exit 1
fi

# ── Logging ──────────────────────────────────────────────────
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# ── Phase-specific quantum study names ───────────────────────
# Maps phase number to the quantum and classical study prefixes
# used in rescore directory names.
case $PHASE in
    1)
        # Phase 1: KH, Tearing, OT, Rotor
        Q_PHASE_PREFIX="q_has_v2_phase1"
        C_PHASE_PREFIX="classical_v2_phase1"
        PHASE_LABEL="Phase 1 — KH, Tearing, Orszag-Tang, MHD Rotor"
        ;;
    2)
        # Phase 2: complex scenarios
        Q_PHASE_PREFIX="q_has_v2_phase2"
        C_PHASE_PREFIX="classical_v2_phase2"
        PHASE_LABEL="Phase 2 — Complex Scenarios (Orszag-Tang, MHD Rotor)"
        ;;
    3)
        # Phase 3: all scenarios
        Q_PHASE_PREFIX="q_has_v2_phase3"
        C_PHASE_PREFIX="classical_v2_phase3"
        PHASE_LABEL="Phase 3 — All 6 Scenarios"
        ;;
esac

LAMBDA_FMT=$(printf "%.4f" "$LAMBDA_COST")
OUTPUT_DIR="$ROOT_DIR/figures/phase${PHASE}"

log "============================================================"
log "Generating Figures: $PHASE_LABEL"
log "  Lambda cost: $LAMBDA_COST"
log "  Output dir:  $OUTPUT_DIR"
log "============================================================"

# ── Step 1: Extract best hyperparams ─────────────────────────
log "Step 1: Extracting best hyperparameters..."

# Check that rescore directories exist
Q_RESCORE_DIR="$TRAIN_RESULTS_DIR/rescore_${Q_PHASE_PREFIX}_lambda${LAMBDA_FMT}"
C_RESCORE_DIR="$TRAIN_RESULTS_DIR/rescore_${C_PHASE_PREFIX}_lambda${LAMBDA_FMT}"

if [[ ! -d "$Q_RESCORE_DIR" ]]; then
    # Try 2-digit lambda format
    LAMBDA_FMT2=$(printf "%.2f" "$LAMBDA_COST")
    Q_RESCORE_DIR="$TRAIN_RESULTS_DIR/rescore_${Q_PHASE_PREFIX}_lambda${LAMBDA_FMT2}"
fi

if [[ -d "$Q_RESCORE_DIR" ]]; then
    log "  Found quantum rescore: $Q_RESCORE_DIR"
else
    log "  WARNING: No quantum rescore directory found for ${Q_PHASE_PREFIX} at λ=${LAMBDA_COST}"
    log "  Looked for: $TRAIN_RESULTS_DIR/rescore_${Q_PHASE_PREFIX}_lambda${LAMBDA_FMT}"
fi

if [[ ! -d "$C_RESCORE_DIR" ]]; then
    LAMBDA_FMT2=$(printf "%.2f" "$LAMBDA_COST")
    C_RESCORE_DIR="$TRAIN_RESULTS_DIR/rescore_${C_PHASE_PREFIX}_lambda${LAMBDA_FMT2}"
fi

if [[ -d "$C_RESCORE_DIR" ]]; then
    log "  Found classical rescore: $C_RESCORE_DIR"
else
    log "  WARNING: No classical rescore directory found for ${C_PHASE_PREFIX} at λ=${LAMBDA_COST}"
    log "  (Classical training may not have been run yet)"
fi

# Map phase number to the quantum and classical phase filter strings.
# The quantum training has sub-phases (1, 1_agr, 1b, 1b_agr) but for
# comparison we want:
#   Phase 1: quantum phase1b (split β, 9 params) vs classical phase1
#   Phase 2: quantum phase2 vs classical phase2
#   Phase 3: quantum phase3 vs classical phase3
case $PHASE in
    1) Q_PHASE_FILTER="phase1" ; C_PHASE_FILTER="phase1" ;;
    2) Q_PHASE_FILTER="phase2"  ; C_PHASE_FILTER="phase2" ;;
    3) Q_PHASE_FILTER="phase3"  ; C_PHASE_FILTER="phase3" ;;
esac

# Run the extraction script with separate quantum/classical phase filters.
python "$ROOT_DIR/scripts/extract_best_hyperparams.py" \
    --train-dir "$TRAIN_RESULTS_DIR" \
    --output "$ROOT_DIR/best_hyperparams.json" \
    --lambda-cost "$LAMBDA_COST" \
    --top-k 3 \
    --quantum-phase-filter "$Q_PHASE_FILTER" \
    --classical-phase-filter "$C_PHASE_FILTER"

log "  best_hyperparams.json updated."

# Export phase filters for figure scripts that read rescore dirs directly
# (e.g. fig0_pareto_lambda.py). The filter value is the phase suffix
# extracted by the regex, e.g. "1b" for rescore_q_has_v2_phase1b_lambda*
case $PHASE in
    1) export FIGURE_Q_PHASE_FILTER="1" ; export FIGURE_C_PHASE_FILTER="1" ;;
    2) export FIGURE_Q_PHASE_FILTER="2"  ; export FIGURE_C_PHASE_FILTER="2" ;;
    3) export FIGURE_Q_PHASE_FILTER="3"  ; export FIGURE_C_PHASE_FILTER="3" ;;
esac

# Export FIGURE_PHASE so figure scripts only run scenarios matching this phase:
#   Phase 1: KH, Tearing, OT, Rotor
#   Phase 2: (reserved)
#   Phase 3: all scenarios
export FIGURE_PHASE="$PHASE"

log "  Phase filters: quantum=${FIGURE_Q_PHASE_FILTER}, classical=${FIGURE_C_PHASE_FILTER}"
log "  Scenario filter: FIGURE_PHASE=${FIGURE_PHASE}"

# ── Step 2: Prepare output directory ─────────────────────────
log "Step 2: Preparing output directory..."
mkdir -p "$OUTPUT_DIR"
log "  Output: $OUTPUT_DIR"

# ── Step 3: Generate all figures ─────────────────────────────
log "Step 3: Generating figures..."

FAILED=0
SUCCEEDED=0

# List of all figure scripts to run
FIGURE_SCRIPTS=(
    "fig0_pareto_lambda.py"
    "fig1_noise_robustness.py"
    "fig2_early_detection.py"
    "fig3_spatial_coherence.py"
    "fig4_comprehensive_comparison.py"
    "fig5_qaoa_detailed_analysis.py"
    "fig6_statistical_validation.py"
    "fig7_physical_fidelity.py"
    "fig8_hierarchical_comparison.py"
    "fig9_synthetic_unit_tests.py"
    "fig10_grid_scaling.py"
    "fig11_hamiltonian_design.py"
    "fig12_depth_analysis.py"
    "fig13_sigma_ablation.py"
    "fig14_boundary_correction.py"
    "fig15_decision_flip_analysis.py"
    "fig16_decision_landscape.py"
    "fig17_topological_attribution.py"
)

for script in "${FIGURE_SCRIPTS[@]}"; do
    script_path="$FIGURES_CODE_DIR/$script"
    if [[ ! -f "$script_path" ]]; then
        log "  SKIP: $script (not found)"
        continue
    fi

    fig_name="${script%.py}"
    log "  Running $script ..."

    if python "$script_path" 2>&1 | tail -5; then
        SUCCEEDED=$((SUCCEEDED + 1))
        log "  OK: $script"
    else
        FAILED=$((FAILED + 1))
        log "  FAILED: $script (continuing...)"
    fi
done

# ── Summary ──────────────────────────────────────────────────
log "============================================================"
log "Figure generation complete for $PHASE_LABEL"
log "  Succeeded: $SUCCEEDED"
log "  Failed:    $FAILED"
log "  Output:    $OUTPUT_DIR"
log ""
log "  Figures saved:"
find "$OUTPUT_DIR" -type f \( -name "*.png" -o -name "*.pdf" \) -printf "    %f\n" 2>/dev/null | sort
log "============================================================"

if [[ $FAILED -gt 0 ]]; then
    log "WARNING: $FAILED figure(s) failed. Check output above for details."
    exit 1
fi

exit 0
