#!/bin/bash
# =============================================================================
# Q-HAS v2 Hamiltonian Quantum-Advantage Study
#
# Runs the full pipeline (phases 1-8) using the parameter-free v2 Hamiltonian
# to test whether the Hamiltonian's low-energy structure matches the L2-hard
# ground truth better than the classical AMR indicator, and to compare QAOA
# against simulated annealing on the SAME objective.
#
# Phase map:
#   1 DNS sweep          4 Exact diag (dim=2)        7 SA baseline on the same H
#   2 Hard patches       5 QAOA eval    (dim=2)      8 Circuit depth / pruning
#   3 Coefficients (dim=4)    6 Hard-patch detection report
#
# Uses the `qiskit-project` conda environment defined in environment.yaml.
#
# Usage:
#   ./study/run_study_v2.sh              # Tier 1: N=128, fast (~15-25 min)
#   ./study/run_study_v2.sh --full       # Tier 2: N=256, publication (~60-90 min)
#   ./study/run_study_v2.sh --v1         # also run v1 (trained) for comparison
#   ./study/run_study_v2.sh --mps        # QAOA on MPS backend (enables dim=3)
#   ./study/run_study_v2.sh --warm       # classical warm-start for QAOA/SA
#   ./study/run_study_v2.sh --prune 0.1  # prune QAOA coeffs |c| < 0.1*max
#   ./study/run_study_v2.sh 3 6          # only phases 3 and 6
#   ./study/run_study_v2.sh 7 8          # only SA baseline + depth report
#   ./study/run_study_v2.sh 10            # joint train (c_bias, thr_amr)
#   ./study/run_study_v2.sh 11            # upper-bound diagnostic (GBT/RF/LR + stencil)
#   ./study/run_study_v2.sh 11b           # leave-one-scenario-out validation
#   ./study/run_study_v2.sh 11c           # learned mean-field Hamiltonian h_i = w.phi - b
#   ./study/run_study_v2.sh 12            # quantum classifier baselines (VQC + QKE)
#   ./study/run_study_v2.sh 13            # aggregate ALL results into SUMMARY file
#   ./study/run_study_v2.sh 11 11b 11c 13 # end-to-end falsification study
#
# Default (no argument): runs phases 1..8, 10, 11, 11b, 11c, 12, 13.
# Expect ~20 min on N=128 (quick), ~90-120 min on N=256 (full).
# =============================================================================
set -euo pipefail

# -- locate repo root --
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="qiskit-project"
ENV_FILE="$REPO_ROOT/environment.yaml"

# -- activate conda env --
activate_env() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not found on PATH." >&2
        echo "  Install Miniconda/Anaconda, or source /etc/profile.d/conda.sh" >&2
        exit 1
    fi

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo ">>> Conda env '$ENV_NAME' not found. Creating from $ENV_FILE ..."
        conda env create -f "$ENV_FILE"
    fi

    echo ">>> Activating conda env: $ENV_NAME"
    conda activate "$ENV_NAME"
}

# -- parse args --
TIER="quick"                                           # quick = N=128, full = N=256
PHASES=""
RUN_V1=0
USE_MPS=0
USE_WARM=0
PRUNE_EPS=""
SCENARIOS="orszag_tang harris_tearing kelvin_helmholtz mhd_rotor"
RE_LIST="400 800 1200 1600"
DIM_COEFF="4"                                          # dim for phases 3 and 6
DIM_QUBIT="2"                                          # dim for phases 4 and 5 (8 qubits)
SA_SWEEPS=2000
SA_RESTARTS=10

# argument parser: flag + value pairs (--prune 0.1)
i=1
args=( "$@" )
while [ $i -le $# ]; do
    arg="${args[$((i-1))]}"
    case "$arg" in
        --full)     TIER="full" ;;
        --quick)    TIER="quick" ;;
        --v1)       RUN_V1=1 ;;
        --mps)      USE_MPS=1 ;;
        --warm)     USE_WARM=1 ;;
        --prune)
            i=$((i+1))
            PRUNE_EPS="${args[$((i-1))]}"
            ;;
        [1-9]|2b|10|11|11b|11b2|11c|11d|11e|11f|11g|11h|12|13) PHASES="$PHASES $arg" ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--quick|--full] [--v1] [--mps] [--warm]" \
                 "[--prune EPS] [phase_numbers]" >&2
            exit 1
            ;;
    esac
    i=$((i+1))
done

if [ -z "$PHASES" ]; then
    # Default run: full pipeline + falsification study + quantum classifier
    # Phase 13 (aggregation) is always last so it sees every upstream .npz.
    PHASES="1 2 2b 3 4 5 6 7 8 10 11 11b 11c 12 13"
fi

# -- resolution & dim defaults --
if [ "$TIER" = "full" ]; then
    N_GRID=256
else
    N_GRID=128
fi

# MPS enables dim=3 for QAOA (18 qubits); otherwise stay at dim=2
if [ "$USE_MPS" -eq 1 ]; then
    DIM_QUBIT_LIST="2 3"
    BACKEND="matrix_product_state"
else
    DIM_QUBIT_LIST="$DIM_QUBIT"
    BACKEND="state_vector"
fi

echo "============================================================"
echo "  Q-HAS v2 Hamiltonian Quantum-Advantage Study"
echo "------------------------------------------------------------"
echo "  Tier:      $TIER  (N=$N_GRID)"
echo "  Phases:    $PHASES"
echo "  Scenarios: $SCENARIOS"
echo "  Re values: $RE_LIST"
echo "  v1 compare: $([ "$RUN_V1" -eq 1 ] && echo yes || echo no)"
echo "  QAOA backend: $BACKEND  (dim list: $DIM_QUBIT_LIST)"
echo "  Warm-start:   $([ "$USE_WARM" -eq 1 ] && echo yes || echo no)"
echo "  Prune eps:    ${PRUNE_EPS:-none}"
echo "============================================================"
echo ""

activate_env

run_phase() {
    local phase=$1
    local script=$2
    shift 2
    echo ""
    echo "============================================================"
    echo "  PHASE $phase  -  $(basename "$script" .py)"
    echo "============================================================"
    python "$script" "$@"
}

# -- driver --
for phase in $PHASES; do
    case "$phase" in
        1)  # DNS sweep
            run_phase 1 study/phase1_dns_sweep.py \
                --N "$N_GRID" \
                --scenario $SCENARIOS \
                --re $RE_LIST
            ;;
        2)  # Hard-patch identification (dim=2 and dim=4)
            run_phase 2 study/phase2_hard_patches.py \
                --N "$N_GRID" --dim $DIM_QUBIT $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST
            ;;
        2b) # Percentile sensitivity: does the LOSO delta change sign
            # when the hard-patch percentile threshold moves?
            run_phase 2b study/phase2b_percentile_sensitivity.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST
            ;;
        3)  # Hamiltonian coefficients (v2, dim=4)
            run_phase 3 study/phase3_coefficients.py \
                --N "$N_GRID" --dim $DIM_COEFF --v2 \
                --scenario $SCENARIOS \
                --re $RE_LIST
            if [ "$RUN_V1" -eq 1 ]; then
                run_phase 3 study/phase3_coefficients.py \
                    --N "$N_GRID" --dim $DIM_COEFF \
                    --scenario $SCENARIOS \
                    --re $RE_LIST
            fi
            ;;
        4)  # Exact diagonalization (v2, dim=2 = 8 qubits)
            run_phase 4 study/phase4_exact_diag.py \
                --N "$N_GRID" --dim $DIM_QUBIT --v2 \
                --scenario $SCENARIOS \
                --re $RE_LIST
            if [ "$RUN_V1" -eq 1 ]; then
                run_phase 4 study/phase4_exact_diag.py \
                    --N "$N_GRID" --dim $DIM_QUBIT \
                    --scenario $SCENARIOS \
                    --re $RE_LIST
            fi
            ;;
        5)  # QAOA evaluation (v2)
            extra_args=( --backend "$BACKEND" )
            [ "$USE_WARM" -eq 1 ] && extra_args+=( --warm-start )
            [ -n "$PRUNE_EPS" ] && extra_args+=( --prune-eps "$PRUNE_EPS" )

            run_phase 5 study/phase5_qaoa_eval.py \
                --N "$N_GRID" --dim $DIM_QUBIT_LIST --v2 \
                --reps 2 --K_opt 80 \
                "${extra_args[@]+${extra_args[@]}}" \
                --scenario $SCENARIOS \
                --re $RE_LIST
            if [ "$RUN_V1" -eq 1 ]; then
                run_phase 5 study/phase5_qaoa_eval.py \
                    --N "$N_GRID" --dim $DIM_QUBIT_LIST \
                    --reps 2 --K_opt 80 \
                    "${extra_args[@]+${extra_args[@]}}" \
                    --scenario $SCENARIOS \
                    --re $RE_LIST
            fi
            ;;
        6)  # Hard-patch detection: Hamiltonian vs classical
            run_phase 6 study/phase6_verify.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST
            if [ "$RUN_V1" -eq 1 ]; then
                run_phase 6 study/phase6_verify.py \
                    --N "$N_GRID" --dim $DIM_COEFF --v1 \
                    --scenario $SCENARIOS \
                    --re $RE_LIST
            fi
            ;;
        7)  # SA baseline on the same Hamiltonian (the fair classical baseline)
            sa_args=()
            [ "$USE_WARM" -eq 1 ] && sa_args+=( --classical-warm )
            run_phase 7 study/phase7_sa_baseline.py \
                --N "$N_GRID" --dim $DIM_QUBIT_LIST $DIM_COEFF \
                --sweeps $SA_SWEEPS --n-restarts $SA_RESTARTS \
                "${sa_args[@]+${sa_args[@]}}" \
                --scenario $SCENARIOS \
                --re $RE_LIST
            if [ "$RUN_V1" -eq 1 ]; then
                run_phase 7 study/phase7_sa_baseline.py \
                    --N "$N_GRID" --dim $DIM_QUBIT_LIST $DIM_COEFF \
                    --sweeps $SA_SWEEPS --n-restarts $SA_RESTARTS \
                    --v1 \
                    "${sa_args[@]+${sa_args[@]}}" \
                    --scenario $SCENARIOS \
                    --re $RE_LIST
            fi
            ;;
        8)  # Circuit depth & pruning report
            eps_list="0.0 0.05 0.1 0.2"
            [ -n "$PRUNE_EPS" ] && eps_list="0.0 $PRUNE_EPS"
            run_phase 8 study/phase8_depth_report.py \
                --N "$N_GRID" --dim $DIM_QUBIT_LIST $DIM_COEFF \
                --reps 2 --prune-eps $eps_list \
                --scenario $SCENARIOS \
                --re $RE_LIST
            ;;
        10) # MF analytical init (10a) + closed-loop training (10) of
            # v2 (c_bias, thr_amr). 10a is fast and writes
            # analytical_N{N}_dim{D}.npz that 10 picks up automatically.
            run_phase 10a study/phase10a_analytical.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST
            run_phase 10 study/phase10_train_hamiltonian.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST \
                --n-iters 40 --optimiser cma \
                --analytical-init auto
            ;;
        11) # Upper-bound diagnostic: is the hard-patch problem
            # Hamiltonian-learnable at all?
            #   Q1 (mean-field ceiling):  LR, RF, GBT on per-site features
            #   Q2 (neighbourhood ceiling): GBT on stencil (self + 4 nbrs)
            # Verdict printed + saved to upper_bound_N{N}_dim{D}.npz
            run_phase 11 study/phase11_upper_bound.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST \
                --max-snaps 30
            ;;
        11b) # Leave-one-scenario-out cross-validation of the ceiling.
            # Tests whether the F1 ~= 0.99 reported by phase 11 is a
            # genuine per-site property or partially inter-scenario
            # memorisation.
            run_phase 11b study/phase11b_loso.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST \
                --max-snaps 30
            ;;
        11c) # Learned mean-field Hamiltonian h_i = w . phi_i - b
            # (logistic regression on the 9 per-site features).
            # This is the effective Hamiltonian that reaches the
            # phase 11 ceiling. --loso also evaluates cross-scenario.
            run_phase 11c study/phase11c_learned_h.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST \
                --max-snaps 30 --loso
            ;;
        11b2) # LOSO with snapshot-level paired bootstrap CIs.
            # Upgrades phase 11B's fold-std to snapshot-level CIs +
            # one-sided p-value per fold for H0: F1_site >= F1_class.
            run_phase 11b2 study/phase11b2_bootstrap.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST \
                --max-snaps 80 --n-boot 500
            ;;
        11d) # Per-scenario specialisation ceiling + misrouting cost.
            # Measures how much of the LOSO collapse is a scenario-
            # transfer problem by evaluating each scenario's H on
            # every other scenario.
            run_phase 11d study/phase11d_specialisation.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST \
                --max-snaps 30 --model gbt
            ;;
        11e) # V1 tuned H + psi under the V2 LOSO protocol.
            # Uses V1's best Optuna params (trial 85) via the
            # input-side score as a tight proxy for the QAOA pipeline.
            # Includes per-fold snapshot-level paired bootstrap CI +
            # p-value on the delta F1(v1+psi) - F1(v2_class).
            run_phase 11e study/phase11e_v1h_loso.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST \
                --max-snaps 80 --n-boot 500
            ;;
        11f) # Multi-seed wrapper (10 seeds, matches V1 Fig. 6).
            # Re-runs phase 11 (random split) + 11B (LOSO) at seeds
            # 0..9; reports mean +/- std per fold across seeds.
            run_phase 11f study/phase11f_multiseed.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST \
                --max-snaps 30 --n-seeds 10
            ;;
        11g) # Scenario-identity ablation: append one-hot scenario id
            # to the 9-feature vector under LOSO. Mechanistic test of
            # whether the collapse is feature-locality (recoverable
            # by knowing scenario) or something more fundamental.
            run_phase 11g study/phase11g_scenario_ablation.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST \
                --max-snaps 30
            ;;
        11h) # Random-split bootstrap CI on the 0.989 ceiling.
            # Snapshot-level paired bootstrap + 95% CIs for
            # F1_class / F1_site / F1_stencil and paired deltas.
            run_phase 11h study/phase11h_random_split_bootstrap.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST \
                --max-snaps 80 --n-boot 500
            ;;
        12) # Quantum classifier baselines (VQC + QKE).
            # Probes the *other* quantum paradigm -- bypass the
            # Hamiltonian, use a quantum circuit as a classifier
            # directly. Slow (COBYLA over a 4-qubit ansatz).
            run_phase 12 study/phase12_vqc.py \
                --N "$N_GRID" --dim $DIM_COEFF \
                --scenario $SCENARIOS \
                --re $RE_LIST \
                --n-train 1500 --n-val 500 \
                --d-q 4 --reps-fm 2 --reps-ansatz 2 --maxiter 80
            ;;
        13) # Cross-phase aggregation: builds SUMMARY_N{N}_dim{D}.txt
            # and .csv from every available upstream .npz.
            run_phase 13 study/phase13_aggregate.py \
                --N "$N_GRID" --dim $DIM_COEFF
            ;;
    esac
done

echo ""
echo "============================================================"
echo "  ALL PHASES COMPLETE"
echo "============================================================"
echo ""
echo "Results in: study/results/"
ls -lh study/results/*.npz 2>/dev/null | tail -20 || echo "  (no results yet)"
