# Q-HAS: Quantum-Hierarchical Adaptive Steering

**Author:** Armand Le Douarec
**License:** MIT (2025)
**Status:** Bachelor's Thesis Research Project

> **TL;DR:** Q-HAS uses a QAOA quantum circuit to guide Adaptive Mesh Refinement in MHD simulations. Compared to classical multi-indicator AMR, Q-HAS demonstrates three structural advantages: **(1) noise resilience** via ZZ spatial filtering, **(2) topology-aware refinement** via ZZZZ plaquette correlations producing less fragmented patches, and **(3) temporal anticipation** via phase encoding (psi) detecting nascent instabilities before they fully develop.

---

> ### ⚠ Falsification update (2026)
>
> The claim above reflects the V1 evaluation (random-split by snapshot). A
> subsequent falsification study, running the stricter cross-scenario
> **leave-one-scenario-out (LOSO)** protocol over the same 4 instability
> classes (N = 256, 4 Re values, seed = 0), **reverses the V1 conclusion**:
>
> |                                  | random-split | LOSO (4 folds) |
> |----------------------------------|:------------:|:--------------:|
> | classical multi-indicator F1     | 0.475        | 0.434          |
> | mean-field ceiling F1 (9 feats)  | **0.989**    | **0.191 ± 0.152** |
> | neighbourhood ceiling F1         | 0.991        | 0.215 ± 0.142  |
> | learned linear H F1              | 0.598        | 0.391          |
>
> The apparent ceiling of 0.989 is **scenario-memorisation**: when every
> snapshot of a held-out instability class is moved to the validation set,
> the ceiling collapses by ≈ 5× and falls *below* the classical baseline.
> Harris-tearing LOSO F1 goes to 0.000.
>
> This is robust to the hard-patch percentile (delta < 0 at every
> p ∈ {60,70,75,80,85,90}), significant under snapshot-level paired
> bootstrap (p_H0 < 0.001 on harris / rotor folds), and **not** rescued by
> per-scenario specialisation (off-diagonal transfer F1 ≈ 0; a runtime
> scenario detector would need ≥ 72% accuracy to break even vs classical).
>
> **Consequence.** On the v2 parameter-free Hamiltonian, simulated
> annealing already reaches the exact ground state and still loses to
> classical by 0.073 F1 (phase 7). No amount of QAOA depth or better
> hardware can beat SA on the same H. The binding constraint on Q-HAS
> is not solver quality but **feature locality across instability
> classes** — the 9 physical features do not span the direction along
> which L2 coarsening error varies across scenarios.
>
> **Pointers to the evidence:**
> - `logs/FINDINGS.md` — one-page summary of the three claims.
> - `docs/review_phases_1_to_11c.md` — full phase-by-phase review of the
>   15 study phases, with file paths, what each computes, and the
>   scientific question it rules out.
> - `docs/v1_vs_study.md` — map from the V1 findings below to the new
>   Claims A / B / C, and what V1 covers that V2 does not.
> - `figures/fig1_ceiling_bar.png`, `figures/fig2_loso_scatter.png` —
>   result figures reproducible from `figures/make_result_figs.py`.
>
> The V1 sections below (training landscape, Figs. 0–16, etc.) are
> retained as the historical record. Read them **through the LOSO lens**:
> the V1 0.66% "advantage" lives inside the 0.989 random-split ceiling
> and disappears under cross-scenario evaluation.

---

## Overview

Q-HAS is a hybrid quantum-classical simulation framework that optimizes the identification and resolution of **magnetohydrodynamics (MHD) instabilities**. Instead of solving the full dynamical equations on a fine grid everywhere (classic DNS approach), the system uses a **Variational Quantum Algorithm (VQA)** to detect topological defects (flux reconnections, turbulence onset) on a coarse-grained graph, then applies **Adaptive Mesh Refinement (AMR)** only where the quantum algorithm identifies anomalies.

This approach concentrates computational resources on physically relevant regions, achieving significant savings compared to full Direct Numerical Simulation (DNS) while preserving accuracy in critical zones.

### Key Idea

```
Full DNS grid (256x256)          Q-HAS approach
┌───────────────────────┐       ┌───────────────────────┐
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │       │ ░░░░░░░░░░░░░░░░░░░ │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │       │ ░░░░░░▓▓▓▓░░░░░░░░░ │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  -->  │ ░░░▓▓▓▓▓▓▓▓▓░░░░░░░ │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │       │ ░░░░░▓▓▓▓▓░░░░░░░░░ │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │       │ ░░░░░░░░░░░░░░░░░░░ │
└───────────────────────┘       └───────────────────────┘
  Compute everything              ▓ = fine (VQA-detected anomaly)
  at full resolution              ░ = coarse (calm region, skip detail)
```

**In plain terms:** A classical MHD simulation runs on a grid. Periodically, a quantum algorithm scans the domain, identifies where "interesting physics" is happening (shocks, reconnection, turbulence), and tells the solver: "refine here, skip there." The result is the same physics at a fraction of the computational cost.

---

## Table of Contents

1. [Q-HAS Theoretical Advantages](#q-has-theoretical-advantages)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Tech Stack](#tech-stack)
5. [Installation & Usage](#installation--usage)
6. [Pipeline Parameters Reference](#pipeline-parameters-reference)
7. [How It Works](#how-it-works)
   - [Hybrid Simulation Loop](#hybrid-simulation-loop)
   - [The VQA Detection Pipeline](#the-vqa-detection-pipeline)
   - [The MHD Solver](#the-mhd-solver)
   - [Adaptive Mesh Refinement (AMR)](#adaptive-mesh-refinement-amr)
   - [Classical AMR Baseline](#classical-amr-baseline)
   - [Physics-to-Quantum Mapping](#physics-to-quantum-mapping)
   - [The Cost Hamiltonian](#the-cost-hamiltonian)
   - [Scoring and Evaluation](#scoring-and-evaluation)
8. [Hyperparameter Training](#hyperparameter-training)
9. [Physical Models](#physical-models)
10. [Test Suite](#test-suite)
11. [Evaluation Figures (Q-HAS vs Classical AMR)](#evaluation-figures-q-has-vs-classical-amr)
12. [Key Results & Conclusions](#key-results--conclusions)
13. [Distributed Training](#distributed-training)
14. [Hamiltonian Architecture Evolution](#hamiltonian-architecture-evolution)
15. [Troubleshooting](#troubleshooting)

---

## Q-HAS Theoretical Advantages

The Q-HAS framework is **not** designed for quantum computational speedup. It targets a **detection advantage**: being predictive and non-local where classical AMR is reactive and local. The three structural advantages stem from the QAOA gradient force at initialization:

```
F_q  proportional to  sin(theta_q) * sin(psi_q) * prod_{m != q} cos(theta_m)
                       ~~~~~~~~~~~    ~~~~~~~~~~    ~~~~~~~~~~~~~~~~~~~~~~~~
                       Local          Phase         Topological
                       amplitude      Boost         Coherence
```

### 1. Noise Resilience (ZZ Spatial Filtering)

The topological coherence factor `prod cos(theta_m)` acts as a **spatial decorrelation filter**:
- **Isolated noise spike** (high score but incoherent neighbors): `prod cos(theta_m) -> 0` => QAOA suppresses the flip. The noise is spatially decorrelated, so the ZZ coupling filters it out.
- **Coherent anomaly** (high score with aligned neighbors): `prod cos(theta_m) ~ 1` => QAOA amplifies the signal.

This means noise that is spatially uncorrelated is naturally rejected, while real physical structures (which are spatially coherent) survive. **Tested in Fig. 1** (noise sigma 0 to 0.5) and **validated in test_qaoa_noise_and_early.py**.

### 2. Topology Identification (ZZ + ZZZZ Spatial Correlations)

The Hamiltonian encodes two types of spatial correlations that classical AMR cannot capture:

| Term | Operator | What it detects | Effect on refinement |
|------|----------|----------------|---------------------|
| **ZZ** (uncertainty-weighted gradient coupling) | `Z_i Z_j` on neighbor pairs | Spatial discontinuities near the decision boundary | Ferromagnetic: neighbors **agree** on refinement, but only where classical score is uncertain (modulated by `exp(-((score-thr)/σ)²)`) |
| **ZZZZ** (plaquette circulation) | `Z_i Z_j Z_k Z_l` on 4-edge plaquettes | Rotational structures (vortex cores, current sheets) | Even-parity: 0 or 2 or 4 edges refined => topologically coherent patches |

**Compression effect**: The ZZ coupling enforces neighbor agreement, producing:
- **Less fragmentation** (fewer isolated refined patches) - Fig. 7
- **More compact patch groups** (contiguous refined regions align with physical structures)
- **Better stability** across time steps (patches don't flicker on/off) - Fig. 4

Classical AMR evaluates each cell independently (RMS of 4 indicators), so two adjacent cells can make contradictory decisions. Q-HAS's spatial correlations prevent this.

### 3. Temporal Anticipation (Phase Encoding psi)

The phase angle psi encodes the **rate of change** of the MHD stress flux:

```
psi = (pi/2) * tanh(beta * delta_Phi / <|delta_Phi|>)
```

- **psi ~ 0**: Flux is stable => no refinement urgency
- **psi ~ +/-pi/2**: Flux changing rapidly => instability developing, refine NOW

This gives Q-HAS a **predictive capability**: it can detect nascent instabilities before they fully develop in the classical indicators. The phase boost `sin(psi_q)` in the gradient force ensures that growing instabilities get preferentially flagged. **Tested in Fig. 4** (early detection timeline) and **test_qaoa_noise_and_early.py** (early anomaly detection).

> **Note:** The phase psi is encoded in the initial state, not in H_struct. This means it functions as a perturbative initialization bias - effective in Q-HAS's small-p regime but not a property that survives deep QAOA convergence. See [Design Trade-off: Phase Boost as Perturbative Bias](#design-trade-off-phase-boost-as-perturbative-bias) for details.

### Summary: Q-HAS vs Classical AMR

| Aspect | Classical AMR | Q-HAS |
|--------|---------------|-------|
| **Detection** | 4 independent indicators, per-cell RMS | Quantum entanglement on cost Hamiltonian |
| **Spatial awareness** | None (each cell decided independently) | ZZ + ZZZZ enforce neighbor correlations |
| **Noise handling** | Susceptible (noise triggers individual indicators) | Spatially decorrelated noise filtered by ZZ |
| **Temporal awareness** | Instantaneous snapshot only | Phase psi encodes flux rate-of-change |
| **Patch topology** | Fragmented (independent decisions) | Compact, coherent (ferromagnetic coupling) |
| **Prediction** | Reactive (detects after development) | Anticipatory (psi detects nascent instabilities) |

---

## Architecture

The framework is organized into three main layers that interact in a loop:

```
┌──────────────────────────────────────────────────────────────────┐
│                   Main Pipeline  (pipeline.py)                   │
│          Orchestrates the hybrid quantum-classical loop          │
│   Two parallel solvers: sim_quantum (Q-HAS) + sim_temoin (DNS)  │
└─────────────────────────────┬────────────────────────────────────┘
                              │
            ┌─────────────────▼─────────────────┐
            │     Adaptive Refinement           │
            │       (refinement.py)             │
            │                                   │
            │  - Level-by-level VQA scanning    │
            │  - Subdivides NxN into sub-blocks │
            │  - Identifies anomalous zones     │
            │  - Returns list of active patches │
            └────────┬────────────────┬─────────┘
                     │                │
          ┌──────────▼────┐    ┌──────▼──────────────────────────┐
          │  VQA Stack    │    │  Classical MHD Solver           │
          │  (src/VQA/)   │    │  (src/Simulation/)              │
          │               │    │                                  │
          │ 1. Mapping    │    │  solver.py:                      │
          │ 2. Cost H.    │    │   - 2D MHD equations (v, B)      │
          │ 3. Optimize   │    │   - FD Order 4 (spatial)         │
          │ 4. Execute    │    │   - RK4 (temporal)               │
          │ 5. Postproc.  │    │   - FFT div-free projection      │
          │               │    │   - CFL adaptive time stepping   │
          │ PhysToAngle:  │    │                                  │
          │  Classical    │    │  grid.py:                        │
          │  score → θ    │    │   - 2pi x 2pi periodic domain    │
          │               │    │   - Grad, Div, Laplacian ops     │
          │ HamiltParams: │    │   - FFT Poisson solver           │
          │  MHD state →  │    │   - Patch extraction for AMR     │
          │  Hamiltonian  │    │                                  │
          │  coefficients │    │  step_full(): entire domain      │
          │               │    │  step_layered(): AMR patches     │
          │ VQARuntime:   │    │                                  │
          │  Singleton    │    │                                  │
          │  backend pool │    │                                  │
          └───────────────┘    └──────────────────────────────────┘
```

---

## Repository Structure

```
BA_Proj/
├── README.md                       # This file
├── LICENSE                         # MIT License
├── environment.yaml                # Conda environment definition
│
├── setup_env.sh                    # Create the Conda environment
├── update.sh                       # Update/sync the Conda environment
├── run_pipeline.sh                 # Run the main Q-HAS simulation
├── TrainHP_GoogleColab.sh          # Train hyperparameters (Google Colab)
├── TrainHPC.sh                     # Train hyperparameters (HPC/SLURM)
├── VisuParam.sh                    # Visualize training results
├── RescoreLambda.sh                # Rescore trials with different lambda values
├── run_tests.sh                    # Run the test suite
├── generate_figures.sh              # Generate all figures per phase (--phase 1/2/3)
├── run_all_figures.sh              # Generate all evaluation figures (legacy)
├── run_fig.sh                      # Generate a single figure (legacy)
│
├── src/                            # ── Main source code ──
│   ├── pipeline.py                 # Main pipeline orchestrator & scoring
│   ├── train_hyperparams.py        # THE training script — 8-parameter Optuna search,
│   │                               #   3 QAOA phases + 3 classical phases, 6 scenarios
│   ├── analyze_hyperparams.py      # Post-training analysis & diagnostic plots
│   ├── hyperparams_loader.py        # Load best hyperparameters from JSON (phase/lambda/rank)
│   ├── recompute_lambda_scores.py  # Rescore trials with different lambda values + Pareto plots
│   ├── call_vqa_shell.py           # VQA execution wrapper (chains all VQA stages)
│   ├── visual.py                   # AMR state visualization (patch overlay plots)
│   ├── help_visual.py              # Visualization helpers
│   ├── compare_rotor_budget.py     # Budget comparison analysis for MHD Rotor
│   │
│   ├── VQA/                        # ── Variational Quantum Algorithm ──
│   │   ├── mapping.py              # Physics → QAOA quantum circuit (ansatz construction)
│   │   ├── cost_hamiltonian.py     # MHD cost Hamiltonians (periodic + bounded/halo)
│   │   ├── init_qbits_state.py     # Qubit initialization with R(theta, psi) gates
│   │   ├── optimize.py             # Circuit transpilation for target backend
│   │   ├── execute.py              # Circuit execution + COBYLA optimization loop
│   │   ├── postprocess.py          # Measurement bitstrings → marginal probabilities
│   │   └── runtime.py              # VQARuntime singleton (shared primitives & caches)
│   │
│   └── Simulation/                 # ── Classical MHD Simulation ──
│       ├── solver.py               # 2D MHD solver (FD4 + RK4, step_full & step_layered)
│       ├── grid.py                 # 2pi x 2pi periodic grid (FFT projection, derivatives)
│       ├── refinement.py           # Level-by-level VQA-guided mesh refinement engine
│       ├── pre_compute_dns.py      # DNS trajectory precomputation (hot-start + trace)
│       ├── HamiltParams.py         # Physical state → Hamiltonian coefficients
│       ├── PhysToAngle.py          # MHD stress flux → quantum angles (theta, psi)
│       ├── RescaleArrays.py        # Adaptive downsampling for VQA input
│       └── utils.py                # Grid helpers & compute_local_factor
│
├── tests/                          # ── Test suite ──
│   ├── test_solver_convergence.py  # Convergence, normalization, physics hierarchy tests
│   ├── test_vqa_anomaly_cases.py   # VQA anomaly detection study cases (8 tests)
│   ├── QAOA_test.py                # End-to-end PhysicalMapper -> VQA pipeline tests (6 tests)
│   ├── test_qaoa_physics_decision.py # Full-pipeline QAOA physics decision tests
│   ├── test_qaoa_advantage.py      # QAOA vs Classical block selection quality measurement
│   ├── test_qaoa_noise_and_early.py # Noise robustness + early detection tests
│   ├── test_qaoa_scaling_and_hparams.py # Grid scaling + hyperparameter sweep
│   ├── test_qaoa_decisions.py      # Controlled QAOA decision validation
│   ├── test_module_validation.py   # Module-by-module validation (angles, Hamiltonian, rescale, etc.)
│   ├── test_signal_contribution.py # Per-term signal contribution analysis
│   ├── test_v9_metrics.py          # Hamiltonian coefficient survival & structure tests
│   ├── test_hamiltonian_v9_diagnostic.py # Hamiltonian architecture validation (15 tests)
│   ├── diag_hamiltonian_balance.py # Hamiltonian coefficient balance diagnostics
│   ├── diag_qaoa_contribution.py   # QAOA contribution measurement (flipped cells)
│   └── diagnose_convergence.py     # Convergence diagnostic with corrected max_depth
│
├── figures/                        # ── Generated figure PNGs (output) ──
│
├── figures_code/                   # ── Figure generation scripts ──
│   ├── fig_utils.py                # Shared utilities, phase filtering, trained params
│   ├── fig0_pareto_lambda.py       # Pareto front from training CSVs
│   ├── fig1_noise_robustness.py    # Noise robustness curves
│   ├── fig2_early_detection.py     # Early detection timeline
│   ├── fig3_spatial_coherence.py   # Spatial coherence & topology
│   ├── fig4_comprehensive_comparison.py # Multi-scenario comparison
│   ├── fig5_qaoa_detailed_analysis.py   # Per-cell QAOA analysis
│   ├── fig6_statistical_validation.py   # Bootstrap + permutation tests
│   ├── fig7_physical_fidelity.py   # L2 error, energy, enstrophy vs DNS
│   ├── fig8_hierarchical_comparison.py  # Full hierarchical AMR comparison
│   ├── fig9_synthetic_unit_tests.py     # Synthetic pattern unit tests
│   ├── fig10_grid_scaling.py       # Grid scaling (N=64, 128, 256)
│   ├── fig11_hamiltonian_design.py # Hamiltonian design visualization
│   ├── fig12_depth_analysis.py     # Depth-resolved AMR analysis
│   ├── fig13_sigma_ablation.py     # Uncertainty weighting ablation
│   ├── fig14_boundary_correction.py # Boundary correction proof-of-concept
│   ├── fig15_decision_flip_analysis.py  # Decision flip diagnostic (WHY analysis)
│   ├── fig16_decision_landscape.py      # Decision landscape scatter plots
│   └── fig17_topological_attribution.py # Topological correction attribution
│
├── scripts/                        # ── Automation scripts ──
│   └── extract_best_hyperparams.py # Extract top-K trials → best_hyperparams.json
│
├── hpc/                            # ── HPC cluster support ──
│   └── submit_training.sh          # SLURM job script for distributed training
│
├── tutos/                          # ── Tutorials & examples ──
│   ├── Max_cut/                    # MaxCut optimization via QAOA
│   │   ├── run_pipeline.sh
│   │   └── scripts/
│   └── VQA/                        # Generic VQA tutorial
│       ├── run_pipeline.sh
│       └── scripts/
│
├── algos_test_MHD/                 # ── Algorithm testing & prototyping ──
│   ├── helloworld.py
│   └── Variational/                # Variational algorithm experiments
│       ├── run_pipeline.sh
│       └── src/
│
├── data/                           # Pipeline output (generated at runtime)
├── logs/                           # Timestamped execution logs
└── Train_results/                  # Hyperparameter training results & Optuna DB
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/` | All production source code for the Q-HAS framework |
| `src/VQA/` | Variational Quantum Algorithm modules (quantum circuit construction, execution, post-processing) |
| `src/Simulation/` | Classical MHD solver, periodic grid, and adaptive refinement engine |
| `tests/` | Pytest-based test suite validating solver convergence and physics guarantees |
| `hpc/` | SLURM job scripts for distributed hyperparameter training on HPC clusters |
| `tutos/` | Self-contained tutorial pipelines (MaxCut QAOA, basic VQA) for learning |
| `figures/` | Generated figure PNGs (output directory) |
| `figures_code/` | Figure generation scripts for QAOA evaluation |
| `algos_test_MHD/` | Experimental algorithm prototypes (ULA ansatz, ZGR-QFT, state preparation) |

---

## Tech Stack

| Category | Technology | Role |
|----------|-----------|------|
| **Language** | Python 3.11 | Primary language |
| **Quantum** | Qiskit | Quantum circuit construction & compilation |
| | Qiskit-Aer | High-performance local quantum simulator |
| | Qiskit-IBM-Runtime | Access to real IBM Quantum hardware (EstimatorV2, SamplerV2) |
| | Qiskit-Optimization | QAOA ansatz and optimization problem tooling |
| **Scientific** | NumPy | Numerical computing, array operations |
| | SciPy | FFT, spatial interpolation (zoom), optimization (minimize) |
| | Matplotlib, Seaborn | Visualization & plotting |
| | NetworkX | Graph problems (used in MaxCut tutorial) |
| | H5py | Large data I/O |
| **Optimization** | Optuna | Multi-phase hyperparameter optimization with pruning |
| **ML** | Scikit-Learn | ML utilities |
| **Testing** | Pytest | Unit & integration testing |
| **Environment** | Conda (conda-forge) | Dependency & environment management |
| **Orchestration** | Bash | Pipeline scripts & automation |

---

## Installation & Usage

**Requirements:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda, Git, macOS/Linux/WSL. Python 3.11 is installed automatically by Conda.

```bash
git clone https://github.com/armandld/BA_Proj.git
cd BA_Proj
source setup_env.sh                  # creates conda env 'qiskit-project'
conda activate qiskit-project
```

### Running the Pipeline

```bash
# Quick test (low resolution)
bash run_pipeline.sh --backend aer --grid-size 2 --dns-resolution 64 \
    --shots 1 --t-max 1.0 --dt 1e-3 --hybrid-dt 0.05 --verbose

# Standard run
bash run_pipeline.sh --backend aer --grid-size 2 --dns-resolution 128 \
    --shots 256 --t-max 1.0 --dt 1e-4 --hybrid-dt 0.1 --verbose

# Full resolution (slow)
bash run_pipeline.sh --backend aer --grid-size 2 --dns-resolution 256 \
    --shots 100000 --t-max 1.0 --dt 1e-4 --hybrid-dt 0.1 --verbose
```

The pipeline initializes two parallel MHD simulations (`sim_quantum` = Q-HAS, `sim_temoin` = DNS witness), runs until `t_max` with VQA updates every `hybrid_dt`, and computes a fidelity score. Logs go to `logs/`.

### Training, Figures & Tests

```bash
python src/train_hyperparams.py --print-space   # inspect the search space
python src/train_hyperparams.py --seed 0        # train (local, Optuna)
bash TrainHP_GoogleColab.sh          # Train on Google Colab
bash VisuParam.sh                    # Visualize training results (12 plot types)

# Generate comparison figures per phase
bash generate_figures.sh --phase 1   # Phase 1: isolated scenarios (KH, Vortex, Tearing, Coalescence)
bash generate_figures.sh --phase 2   # Phase 2: complex scenarios (OT, Rotor)
bash generate_figures.sh --phase 3   # Phase 3: all 6 scenarios
bash generate_figures.sh --phase 1 --lambda 0.40  # Custom lambda cost

bash run_tests.sh                    # Run full test suite
bash run_tests.sh --solver           # MHD solver convergence only
bash run_tests.sh --v9               # Hamiltonian diagnostics only
```

The `generate_figures.sh` script:
1. Extracts best hyperparameters for the chosen phase (quantum phase1b vs classical phase1, etc.)
2. Generates all comparison figures using phase-appropriate scenarios
3. Saves output directly to `figures/phase{N}/`

### Tutorials

Self-contained QAOA examples in `tutos/` (MaxCut graph optimization, generic VQA workflow):

```bash
bash tutos/Max_cut/run_pipeline.sh --backend aer --nodes 10 --edges 12 --mode simulator --verbose
bash tutos/VQA/run_pipeline.sh --backend aer --mode simulator --verbose
```

---

## Pipeline Parameters Reference

### Quantum Backend Options

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--backend` | `aer`, `estimator`, `state_vector` | `aer` | Quantum backend. `aer` = local Qiskit-Aer simulator. `state_vector` = forces statevector method. `estimator` = FakeFez (IBM noisy simulator) |
| `--mode` | `simulator`, `hardware` | `simulator` | Simulator (local) or real IBM Quantum hardware (requires IBM account) |
| `--shots` | integer | `100000` | Number of measurement shots per circuit execution. More shots = less statistical noise |
| `--reps` | integer | auto | QAOA ansatz repetitions. Default: `(grid_size - 1) * 2`. More reps = deeper quantum circuit |
| `--opt-level` | `0`, `1`, `2`, `3` | `1` | Qiskit transpilation optimization level. Higher = more aggressive gate reduction |
| `--method` | `COBYLA`, `Powell`, `L-BFGS-B` | `COBYLA` | Classical optimizer used inside the VQA loop |
| `--K-opt` | integer | `30` | Maximum COBYLA iterations per VQA call. K=40 is optimal (K=20 equivalent, K=10 slightly worse). With p=2 (4 params), COBYLA uses n+1=5 simplex evals → ~25 genuine optimization steps |
| `--eps` | float | `1e-2` | Convergence tolerance for the optimizer |

> **Note on statevector backend:** With `--backend state_vector`, EstimatorV2 computes exact `<H>` (shots ignored for optimization). SamplerV2 still uses shots for marginals - use `shots>=256` for smooth probabilities. Training: 256 shots. Production: 1024 shots.

### Simulation Domain Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--grid-size` | integer | `2` | Coarse grid dimension N for VQA input (N x N qubits = 2N^2 total qubits) |
| `--dns-resolution` | integer | `256` | Full-resolution grid for DNS ground truth |
| `--t-max` | float | `1.0` | Simulation end time (in simulation units) |
| `--dt` | float | `1e-4` | Initial time step (automatically adapted by CFL condition) |
| `--hybrid-dt` | float | `0.1` | Physical time interval between VQA updates. **Critical for ψ encoding** - too frequent: no temporal contrast (ψ≈0); too rare: reactive only. Sweet spot: ~10-15 VQA calls per instability timescale |
| `--scenario` | string | `orszag_tang` | Initial condition scenario (see [Physical Models](#physical-models)) |

### Feature Flags

| Parameter | Description |
|-----------|-------------|
| `--verbose` | Enable detailed logging, diagnostic output, and real-time AMR visualization plots |
| `--skip-cleanup` | Preserve previous results in `data/` (don't clean before running) |
| `--AdvAnomaliesEnable` | Enable advanced anomaly detection: adds X-point reconnection (ZZZZ plaquette) terms to the Hamiltonian |

---

## How It Works

### Hybrid Simulation Loop

The core of Q-HAS is a time-stepping loop in `pipeline.py` that alternates between classical MHD evolution and quantum-guided refinement:

```
t=0                                                                t=T_MAX
 │                                                                   │
 ▼                                                                   ▼
 ┌──VQA──┐   ┌─step─┐  ┌─step─┐ ... ┌──VQA──-┐  ┌─step─┐  ┌─step─┐
 │ detect │→ │evolve│→ │evolve│ ... │ detect │→ │evolve│→ │evolve│→ done
 │ zones  │  │ MHD  │  │ MHD  │     │ zones  │  │ MHD  │  │ MHD  │
 └────────┘  └──────┘  └──────┘     └────────┘  └──────┘  └──────┘
     ↑                                   ↑
     └─── every HYBRID_DT seconds ──────┘
```

**At each VQA update:**
1. The solver computes the current MHD flux state (vx, vy, Bx, By, Jz)
2. The `AngleMapper` converts MHD stress fluxes into quantum angles (theta, psi)
3. The `PhysicalMapper` computes Hamiltonian coefficients from the physics
4. The level-by-level VQA scanner subdivides the domain, calling the quantum circuit at each level
5. Active patches (anomalous regions) are returned with their depth and bounds

**Between VQA updates**, the MHD solver time-steps using:
- `step_layered()` on `sim_quantum`: uses AMR -- each patch is solved at a resolution determined by its depth
- `step_full()` on `sim_temoin`: full DNS witness for comparison

Both solvers use the **same adaptive dt** (CFL-based) to stay synchronized.

### The VQA Detection Pipeline

Each VQA call chains 5 stages, implemented in `src/VQA/` and orchestrated by `call_vqa_shell.py`:

```
MHD State → [1. Mapping] → [2. Optimize] → [3. Execute] → [4. Postprocess] → Anomaly Map
```

| # | Stage | Module | What it does |
|---|-------|--------|-------------|
| 1 | **Mapping** | `mapping.py` | Takes angles (theta, psi) and Hamiltonian coefficients. Initializes qubits via `R(theta, psi)` gates, constructs a `QAOAAnsatz` with the MHD cost Hamiltonian as cost operator. Returns the parameterized quantum circuit. |
| 2 | **Optimize** | `optimize.py` | Transpiles the circuit for the target backend at the chosen optimization level (gate decomposition, routing, simplification). |
| 3 | **Execute** | `execute.py` | Runs the VQA optimization loop: uses `scipy.optimize.minimize` (COBYLA) to find optimal QAOA parameters by minimizing the cost Hamiltonian expectation value via `EstimatorV2`. Then samples the optimized circuit via `SamplerV2`. |
| 4 | **Postprocess** | `postprocess.py` | Converts measurement bitstring distribution into per-qubit marginal probabilities. Each qubit's `P(|1>)` represents the anomaly probability at that grid edge. |

A `VQARuntime` singleton (`runtime.py`) is created once per pipeline run and reused across all VQA calls. It holds shared `EstimatorV2`/`SamplerV2` primitives and an ansatz cache, avoiding repeated instantiation overhead.

**Hardware mode:** When `--mode hardware` is used, `execute.py` runs inside an IBM Runtime `Session` with dynamical decoupling and gate twirling enabled for error mitigation.

### The MHD Solver

The classical solver (`src/Simulation/solver.py`) implements 2D incompressible MHD on a periodic domain.

**Equations solved:**
- **Navier-Stokes** (momentum): velocity field (vx, vy) with advection, Lorentz force (J x B), and viscous diffusion
- **Induction** (Faraday's law): magnetic field (Bx, By) evolution via the rotational form (preserves div(B) = 0)

#### Why Finite Differences? The AMR constraint

The choice of spatial discretization is driven by the need for **Adaptive Mesh Refinement**:

```
                        Spatial Discretization
                       /                      \
              Spectral (FFT)            Finite Differences (FD)
             /            \                    |
      + Very accurate    - GLOBAL basis       + LOCAL stencil
      + Fast (O(N log N))  functions           operates on ANY sub-grid
      + Natural div-free - Cannot operate     + Patches of any size
        projection         on arbitrary       + Compatible with AMR
                           sub-patches
                         - INCOMPATIBLE
                           WITH AMR
```

**The problem:** A pure spectral (FFT) solver is the standard choice for periodic MHD because it is fast and naturally enforces divergence-free fields. However, FFT basis functions are **global** - every Fourier mode spans the entire domain. This makes it fundamentally incompatible with AMR, where we need to solve the equations on **local patches** of different sizes and resolutions.

**The solution:** Finite Differences use a **local stencil** (4th-order centered: only the 4 nearest neighbors in each direction). This means the derivative at any point depends only on its immediate neighborhood, so we can apply the exact same numerical scheme to any rectangular sub-patch of the grid. This locality is what makes `step_layered()` possible.

**The tradeoff:** Centered finite differences are non-dissipative (neutrally stable at best). Paired with a low-order time integrator, high-frequency numerical oscillations can blow up. RK4 solves this by providing a much larger stability region.

> **AMR requires locality -> locality requires Finite Differences -> centered FD is non-dissipative -> stability requires RK4**

> **Note:** FFT is still used for the **div-free projection** (enforcing incompressibility), because that step is applied globally to the full domain after all patches have been integrated.

#### Numerical methods summary

- **Spatial discretization:** 4th-order centered finite differences (stencil +/- 2 pixels)
- **Time integration:** RK4 (Runge-Kutta order 4) with CFL-based adaptive time stepping
- **Divergence constraint:** FFT-based spectral Poisson solver projects fields onto div-free space after each step (applied globally)
- **Advection:** Skew-symmetric form for strict energy conservation
- **Domain:** [0, 2pi] x [0, 2pi] periodic grid (`grid.py`)

#### Two stepping modes

| Mode | Method | Description |
|------|--------|-------------|
| **`step_full()`** | Full DNS | Evolves the entire NxN domain at full resolution. Used by the witness solver. |
| **`step_layered()`** | AMR with tau correction | 3-phase approach: (1) global coarse step, (2) per-patch tau correction (fine_delta - coarse_delta), (3) div-free projection. When all patches are at max_depth, this is **mathematically identical** to `step_full()`. |

The tau-correction (`step_layered`) works as follows:
1. **Phase 1 -- Global coarse step:** Downsample the entire domain by `coarse_factor`, run one RK4 step at coarse resolution, upsample the delta back to full resolution. This provides a cheap baseline correction for the whole domain.
2. **Phase 2 -- Per-patch tau correction:** For each active patch, extract the local region (with FD padding of 8 pixels for the RK4 x FD4 stencil), downsample to the patch's local resolution, run RK4 locally, compute `tau = fine_delta - coarse_delta`. Deeper patches (higher depth) have smaller `local_factor`, meaning higher resolution and a larger tau correction.
3. **Phase 3 -- Global stabilization:** Apply the FFT div-free projection to the full domain (identical to `step_full`), ensuring incompressibility.

**Convergence guarantee:** When all patches are at `max_depth` (where `local_factor = 1`), every patch's local RK4 step runs at full DNS resolution, making `tau = DNS_delta - coarse_delta` everywhere. The result is **mathematically identical** to `step_full()`, up to floating-point precision (~1e-10). This is verified by the test suite.

### Adaptive Mesh Refinement (AMR)

The AMR engine (`src/Simulation/refinement.py`) uses a **level-by-level VQA-guided scan** (BFS):

```
            Full domain (256x256)
                    │
            ┌───────┼───────┐       depth=0: VQA on 2x2 coarse grid
            │       │       │       "Is anything interesting here?"
       ┌────┼────┐  │  ┌────┼────┐
       │    │    │  │  │    │    │  depth=1: subdivide interesting blocks
       │ ░░ │ ▓▓ │  │  │ ░░ │ ░░ │  VQA on each sub-block
       └────┴────┘  │  └────┴────┘
                    │
              ┌─────┼─────┐
              │     │     │        depth=2: further refine the hotspot
              │ ░░  │ ▓▓▓ │        VQA identifies the exact anomaly zone
              └─────┴─────┘

▓ = turbulent (high VQA probability)  →  refine deeper or mark as active patch
░ = calm (low VQA probability)        →  mark as coarse_leaf, solve cheaply
```

**Algorithm:**
1. Start with the full domain as a single block at `depth=0`
2. Downsample the physics to `grid_size x grid_size` and call the VQA
3. The VQA returns a probability map over the coarse grid
4. For each sub-block:
   - If `prob >= effective_threshold` or `depth == 0`: **recurse deeper** (subdivide and re-scan)
   - If `prob < effective_threshold`: **mark as `coarse_leaf`** (calm region, low resolution)
5. Recursion stops when `depth >= max_depth` or patch size `< min_size` (6 pixels)

**Linear threshold schedule:** The effective threshold increases linearly with depth:
```
effective_threshold = threshold_amr + (1 - threshold_amr) * depth / max_depth
```
This ensures that shallow levels (coarse scan) use the base threshold while deeper levels (fine resolution) require progressively higher VQA confidence to keep refining.

**Patch types in the output:**
- `leaf_depth`: reached maximum recursion depth (solved at finest available resolution)
- `leaf_limit`: patch too small to subdivide further
- `coarse_leaf`: calm region, solved at coarse resolution
- `fallback`: no anomalies found anywhere, full domain computed (safety net)

**Decoupled scan/solve depths:** The scan depth (`scan_max_depth`) can be capped independently from the solve depth (`solve_max_depth`). This reduces VQA circuit calls during training while preserving the full resolution hierarchy in the solver. Patches identified by the VQA are stored with their natural solve depth, ensuring `step_layered` computes the correct `local_factor`.

#### Temporal Memory (TTL) and Boundary Probing

Two mechanisms stabilize the refinement tree across time steps:

**TTL (Time-To-Live):** Each refined patch carries a TTL counter (default τ=3 hybrid steps). When a patch is flagged for refinement, its TTL is set to τ. If the detection signal drops below threshold at the next step (due to zoom, noise, or transient fluctuation), the patch survives as long as TTL > 0 (decremented each step). This prevents flickering and gives real instabilities time to develop. TTL applies to **both** VQA-guided and classical AMR paths - each maintains its own independent `ttl_map` persisted across hybrid steps in the pipeline.

**Boundary Probing:** After running the VQA on a patch, the engine checks whether edge qubits (boundary of the patch) have elevated activation probability. If an anomaly touches the boundary, the corresponding sub-cell is probed deeper even if its probability is only marginal (≥ 50% of threshold). This addresses the "lost topology when zooming" problem: instead of propagating the anomaly from the parent, the algorithm *rediscovers* it locally at the boundary.

### Classical AMR Baseline

The classical AMR baseline (`run_adaptive_classical` in `refinement.py`) uses the same BFS tree structure as Q-HAS but replaces the VQA with a **multi-indicator detector** based on standard criteria from production MHD codes (FLASH, Athena++, PLUTO):

| # | Indicator | Formula | Detects |
|---|-----------|---------|---------|
| 1 | **Vorticity** | \|ωz\| = \|∂vy/∂x − ∂vx/∂y\| | Shear layers, vortex cores, KH rolls |
| 2 | **Velocity divergence** | \|∇·v\| = \|∂vx/∂x + ∂vy/∂y\| | Compression / shocks (negative = converging flow) |
| 3 | **Current density** | \|Jz\| | Current sheets, reconnection sites |
| 4 | **Löhner estimator** | \|f''\| / (\|f'\| + ε\|f\|) on \|B\| | Scale-free discontinuity sensor (shocks, contact surfaces) |

The Löhner estimator (Löhner 1987) is a second-derivative error estimator that is dimensionless and peaks at discontinuities while remaining insensitive to smooth gradients - the standard AMR workhorse in most production codes.

**Combination rule:** The 4 indicators are each normalized to [0, 1] per patch, then combined via **RMS** (root mean square):
```
score = sqrt( (s_vort² + s_div² + s_jz² + s_löhner²) / 4 )
```

RMS is used instead of `max()` to avoid over-refinement when a single noisy indicator fires. A cell needs multiple indicators to agree before the combined score triggers refinement.

**Key limitation of classical AMR:** Each cell is decided **independently** - cell (i,j) has no information about its neighbors' scores. Two adjacent cells at a shear layer boundary can make contradictory decisions (one refines, one doesn't). This is exactly what Q-HAS's ZZ coupling fixes: ferromagnetic neighbor agreement enforces consistent decisions across physical structures.

**Fair comparison protocol:** Each method gets its own **independently optimized threshold** via 2-phase search (coarse step=0.10, fine step=0.025). QAOA probabilities (~0.5-0.7) and classical scores (~0.3-0.9) live on different scales - the same threshold value is NOT equivalent for both methods. Primary objective: maximize captured error fraction. Secondary: minimize compute within 5% capture margin. Both use the same BFS traversal, patch hierarchy, solver, and ground truth - only the decision engine differs.

### Physics-to-Quantum Mapping

Two mappers convert the classical MHD state into quantum circuit inputs:

**`AngleMapper` (PhysToAngle.py)** -- Converts MHD physics into quantum circuit angles:

1. **Classical multi-indicator score → θ:** Combines 4 standard MHD-AMR criteria (same as FLASH/Athena++/PLUTO):
   - Vorticity |ωz|, velocity divergence |∇·v|, current density |Jz|, Löhner estimator on |B|
   - Each normalized to [0,1] by domain-wide max, then combined as RMS: `score = √((s_vort² + s_div² + s_jz² + s_loh²) / 4)`
   - **theta** = `2 * arcsin(√score)` → `P(|1⟩) = sin²(θ/2) = score` - gives each qubit the SAME starting probability as the classical AMR detector
2. **Stress flux → ψ (phase):** The stress flux Φ is still computed (shock-diode logic + weighted combination) for the temporal phase:
   - **psi** = `(π/2) * tanh(beta * ΔΦ / ⟨|ΔΦ|⟩)` -- maps flux rate-of-change to phase angle
3. **Hamiltonian = adaptive Z bias + spatial correlations:** The Z-term (single-qubit bias) is reintroduced with an **adaptive weight**: `alpha_z = w_z_frac × median(nonzero |C|, |K|)`, then `H_z = alpha_z × (score − threshold_amr)`. This breaks the QAOA ground-state degeneracy (without Z, ferromagnetic ZZ/ZZZZ gives degenerate all-|0⟩ = all-|1⟩) while keeping Z subordinate to ZZ/ZZZZ spatial correlations. The cost function contains **Z (adaptive bias), ZZ (gradient coupling), and ZZZZ (plaquette circulation)**.

This design cleanly separates the information flow:
- **θ (init)** = "what the classical detector thinks" (anchor)
- **Z (bias)** = "break degeneracy, subordinate to correlations" (adaptive weight)
- **ZZ (cost)** = "uncertainty-weighted spatial agreement near decision boundary" (focused correction)
- **ZZZZ (cost)** = "what rotational/magnetic topology demands" (correction signal)
- **mixer** = "how far to deviate from classical" (exploration budget)

**`PhysicalMapper` (HamiltParams.py)** -- Computes Hamiltonian coefficients from the physics using a **decoupled f x g x threshold-contrast architecture**. Each coefficient is a fully multiplicative product of three factors, with ZZ terms additionally modulated by an uncertainty-weighted Gaussian:

**`Coeff = Weight × g(topology) × f(scale) × ThrContrast(signal)`**

The architecture enforces strict separation between the magnitude of the anomaly (threshold-contrast), its global thermodynamic scaling (f), and its local topology (g). All normalization uses **absolute physical non-dimensionalization** (dividing by fixed physical constants), never relative domain normalization (`x/max(x)`).

**f() - Normal-Critical gate** (logarithmic growth above threshold):
```
f(x) = x / x_crit                      when x ≤ x_crit  (linear, proportional)
f(x) = 1 + γ × ln(x / x_crit)         when x > x_crit   (logarithmic, bounded)
```
Two grouped γ parameters: `gamma_hydro` for Re, `gamma_mag` for Rm. Critical thresholds (fixed): `Re_crit = 1` (advection matches diffusion at cell scale), `Rm_crit = 1`, `Mach_crit = 1.0`, `Q_crit = 2.0`.

**g() - Leaky sigmoid topological gates** (physical switches):
```
g_strain(Q)  = g_min + (1 − g_min) / (1 + exp(−κ × Q/Q_crit))       [Q > 0: strain]
g_rot(Q)     = g_min + (1 − g_min) / (1 + exp(+κ × Q/Q_crit))       [Q < 0: rotation]
g_mag(|Jz|)  = g_min + (1 − g_min) / (1 + exp(−κ × (|Jz|/J_crit − 1)))
```
Leaky minimum `g_min = 0.1` ensures differentiability at topological boundaries (e.g., where Q_OW crosses zero).

**ThrContrast() - Threshold-relative contrast** (replaces the earlier Michelson normalization):
```
ThrContrast(val, val_crit, β) = β × max(0, val/val_crit − 1)
```
Unlike Michelson which compared val to its spatial average (killing the signal when the domain is uniformly active), threshold-relative contrast compares val to a **fixed physical critical value**. This survives in pre-anomaly regimes as long as val > val_crit, regardless of spatial uniformity. The sensitivity β is **split by term type** (trained in Phase 2):
- ZZ gradient terms use a fixed β=1.0 (sensitivity now controlled by `sigma` - see below)
- `β_curl`: for plaquette ZZZZ terms (critical vorticity = RE_CRIT × ν / dx²)
- `β_xpoint`: for X-point ZZZZ terms (critical reconnection = (Rm_crit × η / (dx × B0))²)

**Uncertainty-weighted ZZ coupling:** The ZZ gradient coupling is modulated by a Gaussian centered at the AMR threshold:
```
C_edge *= exp(-((score_avg - threshold_amr) / sigma)²)
```
where `score_avg` is the average classical score of the two cells connected by the edge. This concentrates ZZ coupling near the **decision boundary** where the classical score is uncertain (score ≈ threshold). Far from threshold - where the classical decision is already confident - ZZ coupling is suppressed and the Z-bias alone drives decisions. This prevents ZZ from redundantly enforcing agreement where the classical detector already knows the answer, and focuses quantum corrections on the genuinely ambiguous cells.

| Coefficient | Formula | Topological gate | Weight | Physical meaning |
|-------------|---------|-----------------|--------|------------------|
| `H_edges` | `alpha_z × (score − threshold_amr)` | - | `w_z_frac × median(\|C\|, \|K\|)` | Adaptive Z bias: breaks QAOA ground-state degeneracy; weight scales with coupling magnitude |
| `C_edges` | `2 × g_strain(Q_OW) × √((f_Re × ThrContrast(Δv))² + (f_Rm × ThrContrast(ΔB))²) × exp(-((score_avg − thr)/σ)²)` | g_strain (Q_OW > 0) | 2 | Uncertainty-weighted gradient coupling: active only near decision boundary |
| `K_plaquettes` | `√((g_rot(Q) × f_Re × ThrContrast(ω))² + (g_mag(J) × f_Rm × ThrContrast(J))²)` | g_rot (Q_OW < 0) + g_mag (\|Jz\| > J_crit) | 1 | Circulation plaquette: vortex + current sheet detection |
| `K_xpoint` | `f_Rm × ThrContrast(max(0, −det(J_B)))` | - (self-limiting) | 1 | X-point reconnection: det(∇B) topology detector (optional) |

### The Cost Hamiltonian

The cost Hamiltonian (`src/VQA/cost_hamiltonian.py`) encodes MHD physics into quantum operators. Two topologies are supported:

**Periodic (torus)** -- used at `depth=0` (full domain wraps around):
- All qubit indices use modular arithmetic: `idx = (y % dim) * dim + (x % dim)`
- Plaquette and vertex terms wrap naturally

**Bounded (halo)** -- used at `depth > 0` (sub-patches with explicit boundaries):
- Core qubits are active, boundary qubits are replaced by classical expectation values `<Z> = cos(theta)`
- Halo values from neighboring regions provide boundary conditions
- All arrays (theta, psi, hamilt_params) are padded `(dim+2, dim+2)` with a 1-pixel halo; the Hamiltonian builder accesses the core at `[1:-1, 1:-1]`
- Score-based initialization: at depth > 0, the classical score is downsampled via max-abs pooling (`_process_score`) preserving the padded structure, so quantum AMR starts from the same baseline as classical AMR

**Hamiltonian terms (all use Pauli-Z operators):**

| Term | Pauli | Qubits | Purpose |
|------|-------|--------|---------|
| Activity bias | Z | 1 per edge | Adaptive Z bias: `alpha_z × (score − threshold)`, where `alpha_z = w_z_frac × median(nonzero \|C\|, \|K\|)`. Breaks ground-state degeneracy while remaining subordinate to spatial correlations. |
| Gradient coupling | ZZ | 2 neighboring edges | Two-qubit coupling: uncertainty-weighted spatial correlation near decision boundary |
| Circulation | ZZZZ | 4 edges around a plaquette | Four-qubit plaquette: discrete Stokes - detects rotational flux patterns |
| X-point reconnection | ZZZZ | 4 edges around a plaquette | Four-qubit plaquette: det(J_B) topology - detects hyperbolic magnetic nulls (advanced mode) |

The Hamiltonian is constructed as a `SparsePauliOp` for efficient Qiskit execution. The total number of qubits is `2 * dim^2` (dim^2 horizontal edges + dim^2 vertical edges).

### Scoring and Evaluation

The pipeline computes a multi-variable fidelity score comparing Q-HAS against full DNS. The scoring function (`score()` in `pipeline.py`) is designed around one key asymmetry: **missing an instability (false negative) is far more damaging than wasting compute on a calm region (false positive)**.

#### Asymmetric weighting by current density and vorticity

Regions with high current density |Jz| or high vorticity |ωz| are where reconnection, turbulence, and instabilities develop. The scoring function builds a spatially-varying weight map that combines both indicators:

```
w(x,y) = 1 + 0.5 * ( |Jz(x,y)|/mean(|Jz|) + |ωz(x,y)|/mean(|ωz|) ) / 2
```

where `ωz = ∂vy/∂x − ∂vx/∂y` is the discrete vorticity.

- **Quiet regions** (low |Jz| and |ωz|): w ~ 1 (baseline weight)
- **Active regions** (high |Jz| or |ωz|): w >> 1 (heavily penalized if Q-HAS misses them)

Using both indicators ensures the score correctly penalizes missed instabilities even in scenarios where only one is active (e.g., Lamb-Oseen vortex has high |ωz| but low |Jz|).

#### Combined score

The final score balances physical accuracy against computational cost:

```
combined_score = (phys_score + lambda * patch_ratio) / (1 + lambda)
```

| Component | Formula | Meaning |
|-----------|---------|---------|
| **phys_score** | (Jz+ωz)-weighted average relative L2 error across 5 MHD fields | Physical accuracy (0 = perfect, higher = worse) |
| **patch_ratio** | avg_pixels_refined / N^2 | Computational cost fraction (0 = minimal refinement, 1 = full DNS) |
| **lambda** | Tradeoff parameter (default: 0.5) | Balances accuracy vs. savings |

### DNS Precomputation and Hot-Start

To avoid recomputing the DNS reference trajectory for every Optuna trial, the training system **precomputes it once** per phase configuration (`pre_compute_dns.py`):

1. **Full DNS run:** Evolves the chosen scenario from t=0 to T_MAX at full resolution
2. **Hot-start capture:** At t=T_START (e.g., 2.0), saves the full field state (vx, vy, Bx, By)
3. **Lightweight trace:** Stores per-step adaptive dt (scalar) + flux snapshots only at hybrid-update boundaries and the final step

**Memory optimization:** Instead of storing all ~1000 flux snapshots (~2.5 GB for N=256), only ~10-60 snapshots at HYBRID_DT intervals are kept (~150 MB).

### Practical Design Lessons

These insights emerged from extensive experimentation and debugging. They explain **why** specific parameter choices and architectural decisions were made.

#### 1. COBYLA Must Stay Perturbative (Learning Rate)

The QAOA mixer `exp(-iβ ΣXᵢ)` is the **only mechanism** that changes measurement probabilities. The cost layer only adds phases. If β is too large, the mixer destroys the classical initialization (θ encodes the physics score):

| `rhobeg` | First trial β | P(\|1⟩) change | Effect |
|----------|--------------|---------------|--------|
| 1.0 (default) | 1.0 | ~60° rotation → P≈0.25 | **Trapped at highest energy** (E=+0.52 instead of E=-2.90) |
| 0.15 | 0.15 | ~2.2% change | Moderate, sometimes escapes |
| **0.05** (current) | 0.05 | ~0.05% change | **Safe**: small corrections to θ-init |

**Key rule:** β bounded to `±π/(4×reps)` ≈ 0.39 for p=2. This keeps the mixer perturbative - QAOA makes **small corrections** to the classical score, not large rotations. Without this, COBYLA converges to the wrong minimum, inverting all refinement probabilities.

#### 2. Hamiltonian Normalization (Landscape Smoothing)

Raw ZZ coefficients can reach **-1193** (Orszag-Tang current sheet) or **-526** (KH at depth 1). This creates a rugged COBYLA landscape where the optimizer converges to degenerate minima.

**Solution:** Normalize all coefficients by `1/max(|coefficient|)` before QAOA execution:
- Preserves ground state (uniform scaling doesn't change the minimum)
- Makes all coefficients ∈ [-1, 1], so COBYLA tolerance (`eps=1e-2`) is appropriate
- Warm-start parameters from previous timestep remain compatible after normalization

#### 3. Compression Effect (Patch Regrouping)

The ZZ ferromagnetic coupling enforces **neighbor agreement** on refinement decisions. This produces a natural compression effect:

```
Classical AMR (independent cells)     Q-HAS (ZZ-correlated cells)
┌──────────────────────┐             ┌──────────────────────┐
│ ░░▓░░░░▓░░░░░░░░░░░ │             │ ░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░▓░░░░▓░░░░░░░░ │             │ ░░░▓▓▓▓▓░░░░░░░░░░ │
│ ░░▓░░░░░░░░░░▓░░░░░ │  →  fewer   │ ░░░▓▓▓▓▓░░░░░░░░░░ │
│ ░░░░░░░▓░░░░░░░░░░░ │    pixels   │ ░░░▓▓▓▓░░░░░░░░░░░ │
│ ░░░░▓░░░░░░░░░░░░░░ │             │ ░░░░░░░░░░░░░░░░░░░ │
└──────────────────────┘             └──────────────────────┘
  Scattered isolated patches           Compact contiguous group
  (fragmentation ~ 0.06)              (fragmentation ~ 0.00)
```

**Why fewer pixels:** Compact groups have smaller perimeter-to-area ratio → fewer "wasted" boundary pixels. The same physical structure is covered with ~25% less compute because patches cluster together instead of scattering.

#### 4. Temporal Stability via TTL and Warm-Starting

Two mechanisms prevent patch flickering between time steps:

**TTL (Time-To-Live) = 1:** When a patch is detected for refinement, it gets `ttl=1` - it survives one additional hybrid step even if VQA probability drops below threshold. This gives the QAOA one "grace period" for transient signal weakening.

> **Why not TTL=3?** Testing showed τ=3 causes **period-3 oscillations** in Jaccard stability - patches survive artificially for 3 steps then drop abruptly. τ=1 balances stability with honest evaluation.

**Warm-starting COBYLA:** Optimal QAOA parameters from time t are passed as initial point at t+1. This:
- Eliminates cold-start exploration at each hybrid step
- Maintains temporal coherence (similar physics → similar parameters)
- Reduces K_opt iterations needed (optimizer continues from good point)

#### 5. Phase Boost (ψ) Activation - The "800x Gradient" Discovery

The ψ encoding was initially **never activated** because `phi_dict_prev` was always `None` in benchmarks. This meant the primary temporal advantage was missing entirely.

**Impact of ψ activation:**
- Without ψ: `sin(2Γh) ≈ 0.0009` → tiny QAOA gradient force (negligible corrections)
- With ψ=π/4: `sin(ψ+2Γh) ≈ 0.71` → **800-1137x stronger gradient force**

The fix was to cache the stress flux from the penultimate state (`Φ_prev`) and use EMA (α=0.3) to filter transient noise while capturing sustained trends. This is what enables the temporal anticipation advantage.

#### 6. VQA Resolution Override (Preventing Null Hamiltonians)

At DNS resolution (dx≈0.0245), the local Reynolds number Re_cell < 1 everywhere for KH/Harris scenarios → threshold-contrast zeros ALL coefficients → null Hamiltonian → VQA skipped entirely.

**Solution:** Compute Hamiltonian coefficients at **VQA resolution** (dx_eff = patch_physical_size / target_dim). Each cell spans ~π units at VQA scale → Re_cell >> 1 → non-zero coefficients for anomalous regions.

#### 7. Training Stability - Finite Penalties

Early training produced many `inf` trials that broke Optuna's TPE sampler (can't learn from infinity). Solutions:
- Replace all `float('inf')` returns with finite penalty (10.0)
- Clamp `f_gate()` output to f_max=10.0 (prevents coefficient explosion)
- Clamp `threshold_contrast()` to tc_max=10.0 (prevents degenerate QAOA)
- TPE now properly penalizes divergent configurations instead of re-exploring them

#### 8. Halo Contraction Centering (Parasitic Z-Bias Fix)

At depth > 0 (bounded Hamiltonian), boundary qubits are replaced by classical `<Z> = cos(θ)`. When ZZ edges cross into the halo, the contraction `C × Z_i × <Z_halo>` produces an effective 1-body Z term on the core qubit. Because most halo cells are non-anomalous (score < 0.5 → cos(θ) > 0) and C < 0 (ferromagnetic), the contraction was always negative - biasing core qubits toward |0⟩ (don't refine). This **parasitic halo Z-bias** was 5-40x larger than the designed Z-bias, effectively overriding all refinement decisions at depth > 0.

**Fix:** Center the halo expectation values around the decision boundary: `z_halo = w_z_frac × (cos(θ) − (1 − 2×threshold_amr))`. This ensures the halo contraction is zero when the halo cell sits exactly at threshold, positive when it favors refinement, and negative when it doesn't - matching the designed Z-bias convention. Plaquette contractions (ZZZZ → ZZZ/ZZ) use the raw `cos(θ)` since they produce multi-body terms, not 1-body bias.

#### 9. Uncertainty-Weighted ZZ Coupling (Redundancy Elimination)

Post halo-fix analysis revealed that ZZ coupling was redundant with the classical score: both encode gradient magnitude from the same physical fields. After normalization, ZZ was ~13x stronger than Z-bias. COBYLA optimized for spatial agreement (ZZ) and ignored per-cell refinement preference (Z-bias), making QAOA ≈ classical on all scenarios.

**Fix:** Modulate ZZ coupling by `exp(-((score - threshold) / σ)²)` - a Gaussian peaked at the decision boundary. Coupling is strong only where the classical decision is uncertain (score ≈ threshold); it vanishes where the decision is clear (score far from threshold). The `sigma` parameter (trained, range [0.02, 0.30]) controls the width. This allows Z-bias to drive decisions for clear-cut cells while preserving quantum spatial correlations where they matter most.

---

## Hyperparameter Training

### Current Training (Phase 1)

The training system (`src/train_hyperparams.py`) uses Optuna with MedianPruner. DNS trajectories are precomputed once for each scenario.

**Trained scenarios:** Kelvin-Helmholtz, Harris Tearing, Orszag-Tang, MHD Rotor (4 scenarios).

#### Quantum (Q-HAS) - 5 free parameters

| Parameter | Best (Trial #4) | Range | fANOVA Importance |
|-----------|:---:|-------|:-:|
| `beta_curl` | 4.27 | [0.1, 10.0] | **44%** |
| `beta_xpoint` | 2.39 | [0.1, 10.0] | 21% |
| `sigma` | 0.023 | [0.01, 0.30] | 18% |
| `beta` | 9.94 | [0.5, 10.0] | 11% |
| `w_z_frac` | 10.40 | [1.0, 200.0] | 6% |

**Frozen parameters** (shared with classical or fixed by design):
- `threshold_amr` = 0.1496 (from classical training)
- `gamma_hydro` = 2.0, `gamma_mag` = 0.5, `kappa` = 10.0

**Best quantum score: 0.2134** (trial #4 out of 170+ trials, lambda_cost=0.4).

#### Classical AMR - 1 free parameter

| Parameter | Best (Trial #42) | Range |
|-----------|:---:|-------|
| `threshold_amr` | 0.1496 | [0.05, 0.50] |

**Best classical score: 0.2148** (125 trials).

#### Training Outcome

The quantum method achieves a **0.66% advantage** over classical (0.2134 vs 0.2148). Optuna explored the parameter space widely (w_z_frac from 7 to 168) without strong convergence, suggesting the optimization landscape is flat - the classical threshold is already near-optimal for these 4 scenarios at N=256 with Re=Rm=800.

**Key insight from fANOVA:** `beta_curl` (44%) dominates, meaning the QAOA's main lever is magnetic reconnection detection (curl-based ZZZZ terms). `w_z_frac` at only 6% confirms that ZZ spatial correlations barely influence the training score at 2×2 VQA resolution. The trained `sigma=0.023` is very tight, concentrating ZZ coupling on a narrow band around the threshold.

**Performance optimization:** When DNS traces are available (always during training), the witness solver (`sim_temoin`) is skipped entirely, halving per-trial compute cost.

### Hyperparameter Reference

| Parameter | Range | Phase 1 Best | Description |
|-----------|-------|:-------:|-------------|
| `beta` | [0.5, 10.0] | 9.94 | Phase gain: `psi = (pi/2)*tanh(beta * dPhi / <\|dPhi\|>)` |
| `threshold_amr` | [0.05, 0.50] | 0.1496 | Base AMR recursion threshold (from classical training, frozen for quantum) |
| `gamma_hydro` | frozen | 2.0 | f-gate log-growth for Re |
| `gamma_mag` | frozen | 0.5 | f-gate log-growth for Rm |
| `kappa` | frozen | 10.0 | g-gate leaky sigmoid steepness |
| `w_z_frac` | [1.0, 200.0] | 10.40 | Z bias fraction: `alpha_z = w_z_frac * median(\|C\|, \|K\|)`. Large = classical-like (Z dominates). |
| `sigma` | [0.01, 0.30] | 0.023 | Uncertainty width for ZZ coupling Gaussian: `C *= exp(-((score - threshold) / sigma)^2)` |
| `beta_curl` | [0.1, 10.0] | 4.27 | ThrContrast sensitivity for ZZZZ plaquette/curl terms |
| `beta_xpoint` | [0.1, 10.0] | 2.39 | ThrContrast sensitivity for X-point ZZZZ terms |

> **Source:** Best hyperparameters loaded from `best_hyperparams.json` (generated by `scripts/extract_best_hyperparams.py`). The `hyperparams_loader.py` module provides `load_hyperparams()` with support for phase/lambda/rank selection and `best_per_phase` lookup.

### Hyperparameter Extraction & Selection

After training and rescoring with different lambda_cost values, use the extraction pipeline to centralize the best hyperparameters:

```bash
# 1. Rescore with multiple lambda values
bash RescoreLambda.sh --sweep 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60

# 2. Extract top-3 trials per phase/lambda into best_hyperparams.json
python scripts/extract_best_hyperparams.py --top-k 3 --lambda-cost 0.40

# 3. Phase-specific extraction (e.g., quantum phase1b vs classical phase1)
python scripts/extract_best_hyperparams.py \
    --quantum-phase-filter phase1b \
    --classical-phase-filter phase1 \
    --lambda-cost 0.40
```

**`scripts/extract_best_hyperparams.py`** scans `Train_results/rescore_*_lambda*/` directories for both quantum and classical training results, reads the rescored trial CSVs, and writes a structured `best_hyperparams.json` with:
- **`default.quantum` / `default.classical`**: best overall trial for each method
- **`best_per_phase.quantum` / `classical`**: best trial per training phase (used by `hyperparams_loader.py`)
- **`scenario_combos.simple` / `complex`**: best params for scenario groups
- **`per_scenario.<name>`**: best params per individual scenario
- **`training_phases`**: top-K trials per phase/lambda (raw training data)

The script supports separate phase filters for quantum and classical (`--quantum-phase-filter`, `--classical-phase-filter`), enabling fair cross-phase comparisons (e.g., quantum phase1b with 9 split-beta params vs classical phase1 with 1 threshold param).

Both `pipeline.py` and `fig_utils.py` load hyperparameters from this JSON via `load_hyperparams()`:

```python
from hyperparams_loader import load_hyperparams

params = load_hyperparams()                                        # default quantum
params = load_hyperparams(method='classical')                      # default classical
params = load_hyperparams(phase='phase1', lambda_cost='lambda_0.40', rank=0)  # specific
```

The loader automatically handles the split-beta architecture: Phase 1 results have `beta_michelson`, while Phase 1b+ results have `sigma`, `beta_curl`, `beta_xpoint`. The `_hamilt_mapper_kwargs()` helper in `fig_utils.py` builds the correct `PhysicalMapper` arguments for either format.

### Training Analysis

Run `bash VisuParam.sh` to generate 12 diagnostic plots: parameter importance (fANOVA), convergence curves, Pareto fronts, score landscapes, per-field sensitivity, and correlation heatmaps. Output goes to `Train_results/analysis_*/`.

```bash
python src/train_hyperparams.py --print-space             # what will be searched — check FIRST
python src/train_hyperparams.py --seed 0                  # full sequential run
python src/train_hyperparams.py --phase 2 --seed 0        # one phase
WORKER_TRIALS=50 python src/train_hyperparams.py --phase 2 --seed 0   # one distributed worker
```

---

## Physical Models

The solver supports 8 MHD instability scenarios (`--scenario <name>`):

| Scenario | `--scenario` | Physics | Q-HAS Relevance |
|----------|-------------|---------|-----------------|
| **Orszag-Tang** (default) | `orszag_tang` | Mixed turbulence: current sheets + reconnection + dynamo | Baseline benchmark, all anomaly types |
| **Kelvin-Helmholtz** | `kelvin_helmholtz` | Shear-driven rolling vortex structures | ZZ gradient coupling test |
| **Magnetic Twist** | `magnetic_twist` | |B| ~ const but field direction rotates. Classical AMR sees nothing (f'' ~ 0). VQA detects via psi encoding before kink collapse | **Key Q-HAS advantage scenario**: "silent" instability |
| **Noisy Uniform** | `noisy_uniform` | Pure diffusion + Gaussian noise. Any refinement = false positive | Noise immunity test |
| **Harris Tearing** | `harris_tearing` | Current sheet + tearing-mode perturbation. Classical refines entire sheet. VQA selectively detects X-points vs O-points | Topology identification test |
| **Double Tearing** | `double_tearing` | Two interacting current sheets, individually below threshold. VQA plaquette terms detect frustrated interaction | ZZZZ circulation test |
| **Lamb-Oseen Vortex** | `lamb_oseen_vortex` | Isolated vorticity (strong omega, low |Jz|) | ZZZZ training scenario |
| **Island Coalescence** | `island_coalescence` | Merging magnetic islands via reconnection at X-points. Requires `--AdvAnomaliesEnable` | X-point ZZZZ training scenario |

**Physical parameters:** Re=400, Rm=400, eta=0.001, c_s=1.0, Bz_guide=0.1

**Per-scenario timing (training):**

| Scenario | HYBRID_DT | T_START | Rationale |
|----------|-----------|---------|-----------|
| Kelvin-Helmholtz | 0.10 | 2.0 | KH rolls on ~10 shear times; ~10-15 VQA calls capture roll-up |
| Lamb-Oseen Vortex | 0.10 | 2.0 | Quasi-stationary; moderate frequency for ψ temporal contrast |
| Harris Tearing | 0.10 | 2.0 | Reconnection is gradual, similar timescale to KH |
| Island Coalescence | 0.10 | 0.3 | Reconnection develops after initial current sheet evolution; moderate frequency for X-point tracking |
| Orszag-Tang | 0.10 | 2.0 | Standard benchmark with mixed anomalies |
| MHD Rotor | 0.10 | 0.0 | Rotating field generates vorticity from t=0 |

> **T_START=0 variants:** Each scenario is also trained with T_START=0 (capturing anomaly birth from initial conditions) alongside the regular T_START (capturing developed anomalies). This trains the pipeline to detect instabilities both as they form and after they're fully developed.

---

## Test Suite

Run with `bash run_tests.sh` (all) or `bash run_tests.sh --solver` / `--v9` / `--modules` / `--figures` for individual groups.

### Solver & Convergence

| Test File | What it validates |
|-----------|-------------------|
| `test_solver_convergence.py` | step_layered == step_full at max_depth (< 1e-10), physics hierarchy (deeper = more accurate), patch cost normalization, corrected max_depth formula |
| `diagnose_convergence.py` | Old vs corrected max_depth, multi-step convergence with pipeline patches |

### Hamiltonian Architecture

| Test File | What it validates |
|-----------|-------------------|
| `test_hamiltonian_v9_diagnostic.py` | **15 tests**: adaptive Z bias (5), threshold-contrast survival (3), noise immunity (1), information orthogonality - shear->ZZ, vortex->ZZZZ (3), structural properties - Z subordinate, ZZ ferromagnetic, ZZZZ even-parity (3) |
| `test_v9_metrics.py` | Coefficient survival during MHD evolution, ferromagnetic ground state (all-\|0>), adaptive Z weight scaling, correction map carries spatial info beyond classical score |
| `test_signal_contribution.py` | Each signal (theta, psi, H_Z, C_ZZ, K_ZZZZ) independently contributes to VQA output. Baseline control: uniform inputs -> uniform marginals |
| `diag_hamiltonian_balance.py` | Coefficient magnitudes balanced across Z, ZZ, ZZZZ term types |

### QAOA Pipeline & Decision Quality

| Test File | What it validates |
|-----------|-------------------|
| `QAOA_test.py` | End-to-end pipeline (16x16 -> 2x2, 8 qubits): 6 tests with localized anomalies (Z bias, ZZ gradient, ZZZZ plaquette, ZZZZ vertex, combined) |
| `test_vqa_anomaly_cases.py` | 8 study cases isolating each Hamiltonian term: shear, circulation, X-point reconnection, phase encoding, coefficient signs, energy convergence, cross-anomaly isolation, combined |
| `test_qaoa_physics_decision.py` | Full-pipeline physics decisions on controlled scenarios (vortex, velocity step) |
| `test_qaoa_decisions.py` | Synthetic MHD fields with known GT: quiet (refine nothing), active (refine all), localized anomaly, gradient, MHD Rotor |
| `test_module_validation.py` | Module-by-module: AngleMapper, PhysicalMapper, init_qbits, cost Hamiltonian, QAOA execution, postprocess, RescaleArrays, refinement threshold |

### Q-HAS Advantage Tests

| Test File | What it validates |
|-----------|-------------------|
| `test_qaoa_advantage.py` | Direct QAOA vs Classical block selection quality: Spearman rho to GT, captured error fraction at multiple budgets |
| `test_qaoa_noise_and_early.py` | **Noise robustness** (sigma 0-0.5, 5 trials/level) + **early detection** (AMR at early time vs GT at late time) |
| `test_qaoa_scaling_and_hparams.py` | Grid scaling (N=32, 64, 128) + hyperparameter sweep (w_z_frac, threshold) |
| `diag_qaoa_contribution.py` | Number of flipped cells (QAOA vs classical), energy decomposition, multi-body/single-body ratio |

---

## Distributed Training

The training system supports three storage backends for distributed execution:

| Backend | Environment Variable | Use Case |
|---------|---------------------|----------|
| PostgreSQL | `OPTUNA_STORAGE` | Google Colab + Neon, or any remote DB |
| JournalFileStorage | `OPTUNA_JOURNAL` | HPC clusters with shared filesystem (NFS) |
| SQLite (default) | -- | Local single-machine training |

### HPC Cluster (SLURM)

```bash
# Phase 1: 5 workers, 50 trials each
sbatch --export=PHASE=1,TRIALS=50 hpc/submit_training.sh

# Phase 2 (after phase 1 is done)
sbatch --export=PHASE=2,TRIALS=50 hpc/submit_training.sh

# Phase 3
sbatch --export=PHASE=3,TRIALS=30 hpc/submit_training.sh
```

### Google Colab (PostgreSQL)

```bash
export OPTUNA_STORAGE="postgresql://user:pass@ep-xxx.neon.tech/optuna_db?sslmode=require"
export WORKER_TRIALS="50"
python src/train_hyperparams.py --phase 1 --seed 0
```

---

## Evaluation Figures (Q-HAS vs Classical AMR)

### Methodology

Both methods tested on **2×2 VQA grids (8 qubits)** on **256×256 physical grids** with **4 MHD scenarios** (Kelvin-Helmholtz, Harris Tearing, Orszag-Tang, MHD Rotor). Trained hyperparameters loaded from `best_hyperparams.json`. Both methods use the same threshold (0.1496) - the quantum AMR uses QAOA-modified probabilities while the classical AMR uses the raw multi-indicator score.

### Phase-Aware Figure Generation

Figures are generated via `generate_figures_colab.sh --phase 1` (or `generate_figures.sh`). Each figure script imports trained params from `fig_utils.py` and uses `filter_scenarios()` / `filter_scenarios_dict()` to restrict scenarios.

```bash
# On Colab
bash generate_figures_colab.sh --phase 1
bash generate_figures_colab.sh --phase 1 --figures 4 5 15  # Specific figures only

# Locally
bash generate_figures.sh --phase 1
```

### Figures

| Figure | Script | What it measures | Speed |
|--------|--------|-----------------|:-----:|
| **Fig. 0** | `fig0_pareto_lambda.py` | Pareto front (phys_score vs patch_ratio) from training CSVs | Fast |
| **Fig. 2** | `fig2_early_detection.py` | AMR at early time vs late-time GT: recall, precision, temporal IoU | Slow |
| **Fig. 3** | `fig3_spatial_coherence.py` | Patch compactness (P/A ratio), component density, GT alignment curves | Medium |
| **Fig. 4** | `fig4_comprehensive_comparison.py` | Per-scenario captured fraction, precision, recall, compute ratio | Medium |
| **Fig. 5** | `fig5_qaoa_detailed_analysis.py` | Per-cell QAOA correction δ, Hamiltonian energy decomposition, hierarchical summary | Medium |
| **Fig. 6** | `fig6_statistical_validation.py` | Bootstrap CI + permutation test | Medium |
| **Fig. 7** | `fig7_physical_fidelity.py` | L2 error vs DNS, kinetic energy conservation, enstrophy after AMR evolution | Slow |
| **Fig. 8** | `fig8_hierarchical_comparison.py` | Captured fraction, patch count, compute ratio, L2 fidelity after evolution | Slow |
| **Fig. 11** | `fig11_hamiltonian_design.py` | Uncertainty-weighted ZZ coupling visualization: always-on vs trained σ | Fast |
| **Fig. 12** | `fig12_depth_analysis.py` | Depth-resolved AMR: patch count, captured error, compute budget per depth | Medium |
| **Fig. 13** | `fig13_sigma_ablation.py` | ZZ/Z ratio, active edge fraction, coefficient magnitudes vs σ | Fast |
| **Fig. 15** | `fig15_decision_flip_analysis.py` | **Key diagnostic:** Why QAOA corrections don't change hierarchical outcomes | Medium |
| **Fig. 16** | `fig16_decision_landscape.py` | Decision landscape scatter: classical score vs QAOA probability per cell, flip analysis | Medium |
| **Fig. 17** | `fig17_topological_attribution.py` | Topological correction attribution: accuracy by ZZ quartile, flip accuracy by dominant Hamiltonian term, topology-rich vs smooth advantage | Slow |

### Key Figures at a Glance

| Priority | Figure | Why |
|:--------:|--------|-----|
| 1 | **Fig. 17** | Closes the mechanism-to-outcome loop: decomposes QAOA corrections by Hamiltonian term, shows topology-rich regions produce measurable Q-HAS advantage in Tearing (+3.8pp) and OT (+5.3pp). |
| 2 | **Fig. 15** | Answers the central question: QAOA produces non-zero corrections but most are too small to cross the threshold. Shows flip rate, correctness, ZZ energy at flip locations. |
| 3 | **Fig. 0** | Training landscape: Pareto front showing the 0.66% quantum advantage and parameter importance. |
| 4 | **Fig. 4** | Main comparison: per-scenario captured/precision/recall/compute. Honest view of scenario-dependent results. |
| 5 | **Fig. 6** | Statistical validation: bootstrap CIs and permutation tests. KH (p=0.026) and Tearing (p=0.001) are significant; Rotor and OT are not. |
| 6 | **Fig. 16** | Decision landscape: scatter plot revealing where QAOA adds/removes refinement vs ground truth, per scenario. OT is the only scenario where Q-HAS leads. |
| 7 | **Fig. 11** | The theoretical contribution: Hamiltonian design with uncertainty-weighted ZZ coupling. |
| 8 | **Fig. 13** | Sigma ablation showing how uncertainty width controls ZZ/Z balance. |
| 9 | **Fig. 5** | QAOA mechanism: where corrections happen, Hamiltonian energy decomposition. |

---

## Key Results & Conclusions

> **Read this section alongside the [Falsification update](#q-has-quantum-hierarchical-adaptive-steering) at the top of the README.** The
> "0.66% Q-HAS advantage" reported here is under V1's random-split
> evaluation and does not survive the LOSO protocol. The authoritative
> numbers are in `logs/FINDINGS.md`; the *mechanism* that explains why
> the V1 advantage disappears (scenario-memorisation inside the 0.989
> random-split ceiling) is documented in `docs/review_phases_1_to_11c.md`.

### Training Results

| Method | Best Score | Best Trial | Trials Run |
|--------|:---------:|:----------:|:----------:|
| **Q-HAS** | 0.2134 | #4 | 170+ |
| **Classical** | 0.2148 | #42 | 125 |
| **Advantage** | **0.66%** | - | - |

The quantum advantage is real but marginal. Optuna explored the parameter space widely without convergence, suggesting the classical threshold is already near-optimal at this scale.

---

### Figure-by-Figure Analysis

#### Fig. 0 - Pareto Front: Physics Error vs Compute Cost

![Fig. 0 - Pareto Front](figures/phase1/fig0_pareto_all_combined.png)

**What it shows:** For each lambda_cost value in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], plots the trade-off between physics error (phys_score) and compute savings (patch_ratio) for both Q-HAS and Classical AMR. Per-scenario breakdowns, grouped views, and a combined overview.

**Key question:** Is there a lambda regime where Q-HAS achieves a better Pareto front (lower physics error at the same compute budget, or same error with less compute)?

**Results:** Training explored 178 quantum trials and 125 classical trials across the lambda sweep. Per-scenario Pareto fronts are generated (KH, Tearing, OT, Rotor) plus a combined overview. The operating point at lambda=0.40 sits in the knee of the Pareto curve where further cost reduction yields diminishing physics fidelity.
---

#### Fig. 2 - Early Detection & Temporal Stability

![Fig. 2 - Early Detection](figures/phase1/fig2_early_detection.png)

**What it shows:** Three rows × 4 scenarios. Runs AMR at 6 progressively later timesteps (10%–90% of the shortened simulation), measuring each method's ability to predict late-time refinement needs:
- **Row 0 (Recall):** What fraction of late-time high-error regions does early-time AMR capture?
- **Row 1 (Precision):** Of the regions refined at early time, what fraction actually needs refinement at late time?
- **Row 2 (Pixel IoU / Jaccard stability):** How much do patch selections change between consecutive timesteps?

**Key hypothesis:** The QAOA Hamiltonian encodes topology (vortex cores, current sheets) that are PRECURSORS to instabilities, while classical scoring only detects CURRENT gradients.

**Results (N=256, 6 time-points per scenario):**

| Scenario | QA Recall | CL Recall | QA Precision | CL Precision | QA F1 | CL F1 | QA IoU | CL IoU |
|----------|:---------:|:---------:|:------------:|:------------:|:-----:|:-----:|:------:|:------:|
| **KH** | 0.989 | 1.000 | 0.517 | 0.484 | 0.679 | 0.652 | 0.920 | 0.984 |
| **Tearing** | 0.955 | 0.956 | 0.828 | 0.830 | 0.886 | 0.889 | 0.928 | 0.989 |
| **Rotor** | 0.731 | 0.809 | 0.638 | 0.691 | 0.593 | 0.635 | 0.610 | 0.772 |
| **OT** | 0.976 | 0.999 | 0.485 | 0.470 | 0.648 | 0.640 | 0.912 | 0.976 |

Key findings:
- **KH and OT: Q-HAS has slightly better F1** (0.679 vs 0.652 for KH, 0.648 vs 0.640 for OT) driven by higher precision at the cost of marginally lower recall. This suggests Q-HAS is slightly more selective in early-time refinement.
- **Tearing and Rotor: Classical leads.** Both methods track closely on Tearing (F1 difference < 0.003). Rotor shows the largest gap (F1: 0.593 vs 0.635), consistent with Fig. 4.
- **IoU stability: Classical is more stable across all scenarios.** Classical IoU ranges 0.772-0.989 vs Q-HAS 0.610-0.928. The QAOA corrections introduce slight temporal instability in patch selections, consistent with the fragmentation seen in Fig. 3.

**Row 2 (Jaccard/IoU stability)** is particularly important: it measures `|mask_t ∩ mask_{t+1}| / |mask_t ∪ mask_{t+1}|` between consecutive timesteps. High IoU → stable decisions → better for solver integration. If Q-HAS has lower IoU (more flickering), it would explain the fragmentation observed in Fig. 3.
---

#### Fig. 3 - Spatial Coherence & Topology Detection

![Fig. 3 - Spatial Coherence](figures/phase1/fig3_spatial_coherence.png)

| Scenario | Compactness (P/A) | | Component Density | |
|----------|:---------:|:---------:|:---------:|:---------:|
| | Q-HAS | Classical | Q-HAS | Classical |
| Kelvin-Helmholtz | 0.048 | **0.040** | 0.055 | 0.055 |
| Harris Tearing | 0.075 | **0.070** | 0.147 | **0.122** |
| MHD Rotor | 0.142 | **0.067** | 0.337 | **0.137** |
| Orszag-Tang | 0.221 | **0.190** | 2.119 | **0.925** |

**Lower = more coherent** (fewer fragmented patches, smoother boundaries).

**Key finding: Classical AMR is consistently MORE spatially coherent than Q-HAS across all four scenarios.** This is counterintuitive - the ZZ correlations in the Hamiltonian are designed to enforce spatial consistency (ferromagnetic coupling: "align with your neighbor"). Instead, Q-HAS produces more fragmented patch layouts.

**Why this happens:** The QAOA corrections slightly perturb cell probabilities in both directions around the classical score. At the decision boundary, some cells get pushed above threshold while adjacent cells get pushed below, *fragmenting* what would have been a contiguous patch under classical scoring. The ZZ coupling is too weak (suppressed by σ=0.023) to impose coherence - the perturbative corrections dominate over the ferromagnetic ordering.

**GT alignment curves** show both methods track ground truth similarly at low-to-moderate thresholds (50th–80th percentile). At strict thresholds (>90th percentile), Classical maintains higher precision on MHD Rotor and Orszag-Tang, while both converge on KH and Tearing. This confirms the fragmentation effect: Q-HAS's extra patches don't correspond to the highest-error regions.
---

#### Fig. 4 - Comprehensive Comparison (Q-HAS vs Classical)

![Fig. 4 - Comprehensive Comparison](figures/phase1/fig4_comprehensive_comparison.png)

| Metric | KH | Tearing | Rotor | OT |
|--------|:--:|:-------:|:-----:|:--:|
| **Captured fraction** | | | | |
| Q-HAS | 0.955 | 0.953 | **0.976** | 0.695 |
| Classical | **0.968** | **0.984** | 0.971 | **0.729** |
| Δ | −1.3% | −3.2% | **+0.5%** | −4.7% |
| **Pixel precision** | | | | |
| Q-HAS | **0.705** | 1.000 | 0.983 | 0.700 |
| Classical | 0.683 | 1.000 | **1.000** | **0.777** |
| **Pixel recall** | | | | |
| Q-HAS | 0.975 | 0.860 | 0.661 | 0.277 |
| Classical | **1.000** | **0.865** | **0.834** | **0.441** |
| **Compute ratio** | | | | |
| Q-HAS | **0.754** | 0.625 | 0.398 | 0.623 |
| Classical | 0.846 | 0.625 | **0.273** | 0.621 |

**Scenario-by-scenario analysis:**

- **Kelvin-Helmholtz (best case for Q-HAS):** Q-HAS uses 10.9% less compute (0.754 vs 0.846) with only marginal loss in captured fraction (−1.3%) and actually better precision (+3.2%). This is the "efficiency gain" narrative - same quality, less computational work. The KH instability has a simple, elongated structure that the classical score already captures well, but Q-HAS avoids over-refinement in the bulk.

- **Harris Tearing:** Both methods achieve very high captured fraction (>0.95) and precision (1.000). Q-HAS has slightly lower recall (−0.6%). Compute ratio is identical. **Conclusion: no meaningful difference** - the simple sheet-like structure of the current sheet is easy for both methods.

- **MHD Rotor (worst case for Q-HAS):** Q-HAS has a significant recall deficit (0.661 vs 0.834 = −20.7%), meaning it misses 17% more of the regions that need refinement. It also uses more compute (0.398 vs 0.273 = +45.8%). The rotor's circular geometry with strong gradients at the interface appears to confuse the QAOA corrections - the quantum circuit incorrectly removes refinement from some interface cells.

- **Orszag-Tang (complex turbulence):** Classical wins on every metric. Q-HAS has lower captured fraction (−4.7%), lower precision (−9.9%), and much lower recall (0.277 vs 0.441 = −37.2%). This turbulent scenario with its distributed, multi-scale structure is the hardest for both methods, and the QAOA corrections degrade quality.

**Overall assessment:** Q-HAS shows a genuine efficiency advantage ONLY on Kelvin-Helmholtz. On Rotor and Orszag-Tang, Q-HAS is clearly worse. On Tearing, it's neutral. **There is no consistent quantum advantage across scenarios.**
---

#### Fig. 5 - Detailed QAOA Analysis (per-cell corrections)

![Fig. 5 - QAOA Analysis](figures/phase1/fig5_qaoa_detailed_analysis.png)

**Panel 1 (GT error + QAOA decisions):** All four scenarios show that QAOA probabilities (P values overlaid on cells) are close to but not identical to the classical score. The 2×2 grid at depth 0 divides the domain into quadrants, each with GT error share ~25%.

**Panel 2 (Quadrant comparison):** GT error share, classical P(1), and QAOA P(1) are shown per cell. Across all scenarios, the QAOA probability tracks the classical score closely but with visible offsets - confirming the QAOA circuit modifies the decision scores.

**Panel 3 (QAOA correction δ = QAOA − Classical):**

| Scenario | Correction range (δ) | Direction |
|----------|---------------------|-----------|
| Kelvin-Helmholtz | +0.05 to +0.25 | Mostly positive (adds refinement) |
| Harris Tearing | −0.035 to +0.01 | Mostly negative (removes refinement) |
| MHD Rotor | −0.10 to +0.10 | Mixed (both directions) |
| Orszag-Tang | −0.05 to +0.30 | Mostly positive |

Corrections are **non-zero** everywhere, proving the QAOA circuit is functional. The depth-0 (global) corrections tend to be larger than depth-1 (quadrant) corrections, which is expected since the global 2×2 grid sees the full domain structure.

**Panel 4 (Hamiltonian energy):** H(Z) coefficients dominate at full resolution (N=256), but after downsampling to the 2×2 VQA grid, the energy decomposition changes. C(ZZ) and K(ZZZZ) plaquette terms are visible but small relative to H(Z), confirming the Hamiltonian is Z-dominated - the spatial correlations are a perturbative correction, not the primary signal.

**Panel 5 (Hierarchical comparison):** Captured fraction and compute ratio are close between methods, consistent with Fig. 4. The fine-patches count varies by scenario.

**Key insight:** The corrections exist, are physically structured (not random), and vary by scenario. But they are **perturbative** - the Hamiltonian's dominant H(Z) term reproduces the classical score, and the ZZ corrections are too small to cross the threshold in most cells.
---

#### Fig. 6 - Statistical Validation (Bootstrap CI + Permutation Test)

![Fig. 6 - Statistical Validation](figures/phase1/fig6_statistical_validation.png)

**Left panel - Q-HAS advantage (delta captured fraction) with 95% Bootstrap CI:**

| Scenario | Direction | p-value | Cohen's d | Significance |
|----------|-----------|:-------:|:---------:|:------------:|
| Kelvin-Helmholtz | Q-HAS better | 0.008 | +0.90 | ** |
| Harris Tearing | Q-HAS better | 0.001 | −3.41 | ** |
| MHD Rotor | Not significant | 0.974 | −0.64 | n.s. |
| Orszag-Tang | Not significant | 0.999 | −1.01 | n.s. |

**Right panel - Absolute Performance:** Both methods achieve high captured error fractions (>0.85) on KH, Tearing, and Rotor. Orszag-Tang is the hardest scenario for both (~0.65–0.70).

**Interpretation:**

- **KH (p=0.008, d=+0.90):** Statistically significant advantage for Q-HAS with a large effect size. Combined with Fig. 4's compute ratio advantage (−10.9%), this is the strongest result. However, the absolute magnitude of the advantage is small.

- **Tearing (p=0.001, d=−3.41):** Statistically significant but the large negative d suggests the effect is driven by Q-HAS being *consistently slightly worse* in a specific direction. The magnitude is small in absolute terms.

- **Rotor and OT:** Not statistically significant - the bootstrap CIs are wide and cross zero. High variance across seeds means we cannot conclude either method is better.

**Key conclusion:** Even with 10 seeds, the methods are statistically distinguishable only on KH and Tearing. The practical significance (absolute performance difference) is small in all cases. **The statistical tests confirm what the figures show: the quantum corrections produce measurable but practically insignificant changes.**
---

#### Fig. 11 - Hamiltonian Design Visualization

![Fig. 11 - Hamiltonian Design](figures/phase1/fig11_hamiltonian_design.png)

**Scenarios shown:** Kelvin-Helmholtz (top row) and Orszag-Tang (bottom row).

**Panel A (Classical Score):** KH shows the characteristic shear layer (high scores along the interface), OT shows distributed vortex structures with multiple high-score regions.

**Panels B–C (|ZZ| Always-On vs Uncertainty-Weighted):**
- **KH:** Both ZZ panels show ~1e-6 magnitude - effectively **zero**. The Gaussian gate `exp(-((s-0.15)/0.023)²)` kills all coupling because KH scores are far from threshold (either ≫0.15 near the shear layer or ≈0 in the bulk). The always-on and weighted versions are identical because both are negligible.
- **OT:** ZZ coupling is visible along the vortex filaments. The uncertainty-weighted version (Panel C) selectively preserves coupling near the decision boundary while suppressing it in clear-refine and clear-skip regions.

**Panel D (Uncertainty Weight):** Shows the Gaussian `exp(-((s-thr)/σ)²)` field. For KH, the green band (weight ≈ 1 near threshold) is narrow and follows the contour lines. For OT, the activation pattern is more complex, following multiple vortex boundaries.

**Panel E (Z-Bias):**
- **KH:** Appears blank (1e-6 scale) - same reason as ZZ: all scores far from threshold → uncertainty weight ≈ 0 → Z-bias ≈ 0.
- **OT:** Shows clear red/blue structure - positive (refine) in high-score regions, negative (skip) in low-score regions, with magnitude ~2×10⁻⁶.

**Key insight:** The figure perfectly illustrates WHY σ=0.023 limits quantum advantage. For scenarios with bimodal score distributions (KH), the Hamiltonian is effectively **null** - the QAOA has no information to work with beyond the classical score. Only for scenarios with scores near the threshold (parts of OT) does the ZZ coupling activate.
---

#### Fig. 12 - Depth-Resolved AMR Analysis

![Fig. 12 - Depth Analysis](figures/phase1/fig12_depth_analysis.png)

Four columns per scenario: Patches by Depth, GT Error Captured, Q-HAS vs Classical Agreement, Compute Budget by Depth.

**Patches by Depth:**
- All scenarios show exponential growth in patch count with depth (4 → 16 → 64 → 256 → ...)
- Q-HAS and Classical produce very similar patch counts at all depths
- **KH:** Q-HAS produces slightly fewer patches at depth 4-5 (consistent with the compute savings seen in Fig. 4)
- **OT:** Q-HAS produces slightly more patches at intermediate depths

**GT Error Captured:**
- Most GT error is captured at the **deepest levels** (depth 5) - the leaf patches at finest resolution
- Q-HAS and Classical capture similar fractions at each depth
- **KH:** Both capture ~90% at depth 5
- **Rotor:** Q-HAS captures noticeably less at depth 5 (~87% vs ~95%), consistent with Fig. 4's recall deficit

**Q-HAS vs Classical Agreement:**
- **Depths 0–4: ~100% agreement** (green bars) - the methods make identical decisions through most of the tree
- **Depth 5: agreement drops** - KH shows ~65% agreement (orange), others show ~90-95%
- This is the critical finding: **disagreements concentrate at the deepest level** where patches are smallest and scores are most variable

**Compute Budget by Depth:**
- Budget concentrates at the deepest levels (depth 4-5 use 60-70% of total compute)
- Q-HAS and Classical allocate compute similarly

**Key insight:** The decision tree is >95% identical between methods through depths 0-4. All differences emerge at the **leaf level** (depth 5), where the 2×2 VQA makes its finest-grain decisions. This confirms the corrections are local and perturbative - the hierarchical structure is robust.
---

#### Fig. 13 - Sigma Ablation Study

![Fig. 13 - Sigma Ablation](figures/phase1/fig13_sigma_ablation.png)

**Panel A (ZZ/Z Coefficient Ratio vs σ):**
- At σ=0.03 (near trained value 0.023): ZZ/Z ratio spans ~10 (OT) to ~50,000 (KH). The huge KH ratio is misleading - both ZZ and Z are tiny (Panel C), so the ratio is noise.
- At σ≥0.10: ratios stabilize around 10-15 for all scenarios.
- The trained σ=0.023 is in the **steep dropoff zone** where the ratio changes by orders of magnitude. This confirms σ is a critical hyperparameter.

**Panel B (Active Edges vs σ):**
- At σ=0.03: only 5-20% of edges have active ZZ coupling (uncertainty weight > 0.1)
- At σ=0.10: 30-70% active
- At σ=0.30: 80-100% active
- **The trained σ=0.023 activates very few edges** - ZZ coupling is concentrated on a thin band around the threshold. This is by design (focus quantum resources where they matter most) but means only a tiny fraction of cells receive spatial corrections.

**Panel C (ZZ and Z Magnitudes vs σ):**
- KH: ZZ magnitude drops to ~10⁻⁷ at σ=0.03 while Z stays at ~10⁻³. The ZZ coupling is **6 orders of magnitude weaker** than Z at the trained σ.
- OT and Tearing: ZZ ≈ 10⁻³ to 10⁻² across all σ values.
- **Key observation:** For KH, the Hamiltonian is Z-dominated by a factor of 10⁴ at the trained σ. This means QAOA ≈ classical score + negligible ZZ perturbation.

**Panel D (Score Distance from Threshold):**
- KH scores cluster near |s − thr| ≈ 0.10–0.15 (most cells are far from threshold)
- OT scores have a broader distribution with more cells near threshold
- Tearing scores peak at |s − thr| ≈ 0.13
- The σ=0.05 Gaussian (blue dashed) covers only the closest ~5% of cells. At σ=0.023 (trained), coverage is even narrower.

**Key insight:** The ablation proves that σ=0.023 is **too narrow** for the current score distributions. It focuses ZZ coupling on <10% of edges, making the quantum corrections negligible for most cells. A broader σ (0.10–0.15) would activate more edges, but Optuna chose σ=0.023 because it minimized the training loss - suggesting that activating MORE edges doesn't actually help (the ZZ corrections hurt as often as they help).
---

#### Fig. 7 - Physical Fidelity (Step-Layered Evolution)

![Fig. 7 - Physical Fidelity](figures/phase1/fig7_physical_fidelity.png)

**What it shows:** THE key figure. Evolves three copies of each MHD scenario: DNS reference (full resolution), Q-HAS AMR, and Classical AMR. At each step, AMR patches are computed from each method's OWN simulation state (fair comparison - no cheating from DNS). Evolution uses `step_layered()` which applies the AMR patch structure to the solver. Plots L2 error vs DNS, kinetic energy conservation, and enstrophy conservation over 15 AMR steps.

**Three columns per scenario:**
- **Col 0 (L2 Error vs DNS):** Relative L2 error over time (log scale). This is the primary fidelity metric - how much does AMR-guided evolution diverge from full DNS?
- **Col 1 (Kinetic Energy):** Tracks Ek = ½∫(vx² + vy²) dA. AMR should preserve energy conservation.
- **Col 2 (Enstrophy):** Tracks ∫ω² dA (vorticity squared). Enstrophy is more sensitive to small-scale errors than energy.

**Results (N=256, 15 AMR steps, depth=5):**

| Scenario | Q-HAS Final L2 | Classical Final L2 | Winner |
|----------|:---------------:|:------------------:|:------:|
| **KH** | 2e-6 | ~0 | Classical |
| **Tearing** | 1.0e-5 | 3e-6 | Classical |
| **Rotor** | 9.6e-4 | 5.2e-4 | Classical |
| **OT** | 1.6e-4 | ~0 | Classical |

Key findings:
- **Both methods achieve excellent fidelity** - all L2 errors are small (< 0.001), confirming that AMR-guided evolution is viable for all 4 scenarios.
- **Classical has lower L2 error across all scenarios.** The QAOA corrections slightly perturb patch selection in ways that accumulate over 15 steps. Rotor shows the largest gap (factor ~2x), consistent with Fig. 4's recall deficit.
- **The differences are practically negligible for KH and Tearing** (L2 < 1e-5 for both methods). The simulation fidelity is not meaningfully impacted by the small AMR differences.
- **Energy and enstrophy conservation** tracks closely for both methods, confirming the solver itself is not affected by the AMR choice.
---

#### Fig. 8 - Hierarchical AMR Comparison (Multi-Panel)

![Fig. 8 - Hierarchical Comparison](figures/phase1/fig8_hierarchical_comparison.png)

**What it shows:** Four-panel comparison of Q-HAS vs Classical on N=256 grids with 2×2 VQA (8 qubits):
- **Panel A:** Per-scenario bar chart of captured error fraction (2 trials for error bars)
- **Panel B:** Patch count breakdown (fine vs coarse patches)
- **Panel C:** Compute ratio (effective pixels / N²)
- **Panel D:** Physical fidelity - L2 error vs DNS after 10 AMR-guided evolution steps

**Relation to other figures:** This is a condensed version of Figs. 4 + 7, combining static AMR quality (Panels A–C) with dynamic fidelity (Panel D) in one figure. Useful as an overview figure.

**Results (N=256, 2 trials, depth=5):**

| Scenario | Q-HAS Captured | Classical Captured | Delta | L2 (Q-HAS) | L2 (Classical) |
|----------|:--------------:|:------------------:|:-----:|:-----------:|:--------------:|
| **KH** | 0.986 | 1.000 | -0.014 | 2e-6 | ~0 |
| **Tearing** | 0.994 | 0.990 | +0.004 | 9e-6 | 3e-6 |
| **Rotor** | 0.952 | 0.949 | +0.003 | 8.0e-4 | 4.7e-4 |
| **OT** | 1.000 | 1.000 | +0.000 | 2.0e-4 | ~0 |

Tearing, Rotor, and OT show ties or marginal Q-HAS advantage in captured fraction, while KH slightly favors Classical. L2 fidelity (Panel D, after 10 AMR evolution steps) consistently favors Classical by small margins. The overall picture is one of near-equivalence: both methods produce high-fidelity AMR with differences at the third decimal place.
---

#### Fig. 15 - Decision Flip Analysis: Why QAOA Corrections Don't Change Outcomes

![Fig. 15 - Decision Flip Analysis](figures/phase1/fig15_decision_flip_analysis.png)

**What it shows:** THE diagnostic figure. Instruments the hierarchical BFS tree to capture, at every node and every cell, both the classical score and the QAOA probability BEFORE the refine/coarsen decision is made. Six panels per scenario:

- **Col 0 (Scatter: Classical score vs QAOA prob):** Each dot is one cell at one BFS node. Points near the diagonal = no correction. Points in the upper-left (above threshold, below threshold on x) = QAOA added refinement. Points in lower-right = QAOA removed refinement. Green dots = "good" flips (QAOA matches GT), red X = "bad" flips.
- **Col 1 (Threshold proximity histogram):** Distribution of classical scores relative to the threshold (blue) overlaid with QAOA correction magnitudes (orange). If the blue distribution is far from zero, corrections can't cross the threshold.
- **Col 2 (Flips by depth):** Stacked bars showing agreement (blue), correct flips (green), and incorrect flips (red) at each BFS depth level.
- **Col 3 (Can corrections reach threshold?):** Scatter of |distance to threshold| vs |correction magnitude|. Points above the diagonal are where corrections CAN flip decisions. Points below = corrections too small.
- **Col 4 (Summary stats):** Overall accuracy numbers - QAOA correct/total vs Classical correct/total. Plus pixel impact and threshold analysis.
- **Col 5 (WHY panel):** Physical mechanism analysis - direction of flips (added vs removed refinement), ZZ coupling strength at flip sites, neighbor context.

**GT threshold definition (FIXED):** "Should refine" = GT error > mean(GT error), consistent with Fig. 4's pixel_precision. This resolves the earlier contradiction where OT showed "good flips" but poor Fig. 4 performance.

**Results (N=256, depth=5):**

| Scenario | Cells | Flips | Flip Rate | QAOA Correct | Classical Correct | Net |
|----------|:-----:|:-----:|:---------:|:------------:|:-----------------:|:---:|
| **KH** | 904 | 125 | 13.8% | 61.3% | 65.6% | Classical by 39 |
| **Tearing** | 584 | 109 | 18.7% | 77.9% | 81.2% | Classical by 19 |
| **Rotor** | 240 | 76 | 31.7% | 67.9% | 72.1% | Classical by 10 |
| **OT** | 408 | 145 | 35.5% | 68.6% | 70.3% | Classical by 7 |
| **Total** | 2136 | 455 | **21.3%** | 43.3% | 46.6% | Classical by 71 |

Global analysis:
- **Corrections CAN reach the threshold**: 44.5% of corrections have |correction| > |distance to threshold| (ratio > 1.0). The median ratio is 0.73, meaning a large minority of cells have corrections large enough to flip decisions.
- **Flips concentrate at deepest depths**: Depth 4 accounts for the majority of flips (consistent with Fig. 12)
- **ZZ energy is 783,000x higher at flip sites** (median 0.0078 vs ~0 for non-flipped cells), confirming that the ZZ coupling drives corrections at the decision boundary
- **352 flips pushed UP (toward refine), 103 pushed DOWN**, showing an asymmetric bias toward adding refinement
- **Spearman correlation |delta| vs GT error: rho=0.113** (p < 0.0001) - corrections weakly target high-error regions

**Key interpretation:** The flip rate (21.3%) is much higher than initially expected, showing the QAOA corrections DO change decisions at a meaningful fraction of cells. However, Classical is more accurate by 71 cells overall. The OT scenario has the smallest gap (7 cells), consistent with Fig. 16 and 17 showing OT as the best case for Q-HAS.
#### Fig. 16 - Decision Landscape (Classical Score vs QAOA Probability)

![Fig. 16 - Decision Landscape](figures/phase1/fig16_decision_landscape.png)

**What it shows:** Per-cell scatter plot of classical multi-indicator score (x-axis) vs QAOA probability (y-axis) for all leaf cells, per scenario. The threshold is drawn as crosshairs, dividing the plot into 4 quadrants: agree-refine (upper-right), agree-skip (lower-left), QAOA-adds (upper-left, QAOA refines but classical skips), QAOA-removes (lower-right, classical refines but QAOA skips). Ground truth overlay colors cells by whether the "correct" decision is refine or skip.

**Results (N=256, threshold=0.1496):**

| Scenario | Total Cells | Flips | Flip Rate | QAOA Correct | Classical Correct | Net Winner |
|----------|:-----------:|:-----:|:---------:|:------------:|:-----------------:|:----------:|
| **KH** | 884 | 137 | 15.5% | 61.4% | 64.9% | Classical by 31 |
| **Tearing** | 588 | 111 | 18.9% | 77.7% | 81.0% | Classical by 19 |
| **Rotor** | 252 | 80 | 31.7% | 70.2% | 75.0% | Classical by 12 |
| **OT** | 432 | 144 | 33.3% | 69.9% | 67.6% | **Q-HAS by 10** |

**Key findings:**
- **OT is the only scenario where Q-HAS wins.** It adds 105 refinement cells (49 correct) and removes 39 (28 correct), yielding a net +10 cell advantage. This is consistent with OT having the most active ZZ/ZZZZ coupling (Fig. 11) and the richest topological structure.
- **QAOA removes are highly accurate for Tearing:** 16/16 removed cells were correctly removed. The QAOA successfully identifies over-refinement in the simple current-sheet geometry.
- **QAOA adds are mostly incorrect for KH:** only 8/91 added cells were correct. The KH shear layer has a bimodal score distribution (most cells far from threshold), so QAOA additions in the skip zone are noise.
- **Rotor has the highest flip rate (31.7%)** but flips are predominantly harmful, consistent with the rotor's circular geometry confusing the QAOA corrections (as seen in Fig. 4).
#### Fig. 17 - Topological Correction Attribution

![Fig. 17 - Topological Attribution](figures/phase1/fig17_topological_attribution.png)

**What it shows:** Decomposes QAOA corrections by Hamiltonian term to answer: "Does the topology-detection mechanism (ZZZZ plaquettes, ZZ spatial coupling) actually drive better decisions?" Three panels:

- **Panel A (Accuracy by ZZ Coupling Quartile):** All scenarios pooled. Cells binned by their ZZ coupling strength into quartiles (Q1=weakest to Q4=strongest). Q-HAS and Classical accuracy plotted per quartile.
- **Panel B (Flip Accuracy by Dominant Term):** For cells where QAOA flipped the decision, which Hamiltonian term (ZZ spatial, ZZZZ topology, Z bias) dominated at that cell? Accuracy of flips broken down by dominant term and scenario.
- **Panel C (QAOA Advantage by Topological Regime):** Each scenario split into "topology-rich" (above-median ZZZZ strength) and "smooth" (below-median) regions. Q-HAS advantage (delta accuracy) reported for each regime.

**Results (N=256, threshold=0.1496):**

**Panel A - Accuracy by ZZ Quartile (pooled):**

| Quartile | Q-HAS | Classical |
|----------|:-----:|:---------:|
| Q1 (weakest ZZ) | 0.817 | 0.850 |
| Q2 | 0.730 | 0.770 |
| Q3 | 0.612 | 0.515 |
| Q4 (strongest ZZ) | 0.529 | 0.606 |

Q-HAS overtakes Classical in Q3 (0.612 vs 0.515), the quartile where ZZ coupling is moderate. At the strongest coupling (Q4), both methods struggle but Classical edges ahead, suggesting the strongest ZZ regions coincide with genuinely ambiguous physics.

**Panel B - Flip Accuracy by Dominant Hamiltonian Term:**

| Scenario | ZZ (spatial) | ZZZZ (topology) | Z (bias) |
|----------|:------------:|:----------------:|:--------:|
| **KH** | 0.333 (45 flips) | 0.500 (14 flips) | 0.431 (58 flips) |
| **Tearing** | 0.765 (68 flips) | 0.000 (8 flips) | 0.000 (41 flips) |
| **Rotor** | 0.429 (14 flips) | 0.491 (53 flips) | 0.333 (15 flips) |
| **OT** | 0.378 (45 flips) | 0.663 (92 flips) | 0.200 (5 flips) |

Key observations:
- **Tearing ZZ-driven flips are 76.5% accurate** (68 flips) - the ZZ spatial coupling correctly identifies refinement needs at the current sheet boundary. This is the strongest single-term result.
- **OT ZZZZ-driven flips are 66.3% accurate** (92 flips) - the plaquette circulation term detects rotational structures in the Orszag-Tang vortex, validating the theoretical motivation for ZZZZ terms.
- **Z-bias flips are generally poor** (0-43% accuracy), confirming that the adaptive bias term alone does not add useful information beyond the classical score.

**Panel C - QAOA Advantage by Topological Regime:**

| Scenario | Topology-rich (delta) | Smooth (delta) | Q-HAS rich | Classical rich |
|----------|:---------------------:|:--------------:|:----------:|:--------------:|
| **KH** | -0.042 | -0.012 | 0.322 | 0.363 |
| **Tearing** | **+0.038** | -0.082 | 0.729 | 0.692 |
| **Rotor** | -0.057 | -0.008 | 0.631 | 0.689 |
| **OT** | **+0.053** | +0.024 | 0.675 | 0.621 |

**The central finding:** In topology-rich regions, Q-HAS shows a positive advantage for Tearing (+3.8 percentage points) and OT (+5.3pp). These are the two scenarios with the strongest topological features (current-sheet reconnection for Tearing, vortex interactions for OT). In smooth regions, the advantage disappears or reverses. This confirms the theoretical prediction: the ZZZZ plaquette and ZZ coupling terms contribute meaningfully where the physics has genuine topological structure.
---

### Cross-Figure Synthesis

#### What the data collectively tells us

1. **The QAOA circuit IS functional.** Fig. 5 proves non-zero corrections (δ) exist at every BFS node. The corrections are structured (vary by scenario, depth, cell position) - not random noise.

2. **The corrections are perturbative, not transformative.** Fig. 4 and Fig. 6 show that practical outcomes (captured fraction, precision, recall) differ by at most +/-5% between methods.

3. **The only scenario with a genuine efficiency advantage is KH.** Q-HAS uses 10.9% less compute with nearly the same capture quality, and this is statistically significant (p=0.008). On all other scenarios, Q-HAS is either equivalent or worse.

4. **ZZ correlations fragment rather than consolidate patches (Fig. 3).** This is the most surprising result. The Hamiltonian's ferromagnetic ZZ coupling was designed to enforce spatial coherence, but in practice the perturbative corrections cause fragmentation. This suggests the coupling strength is in a regime where it perturbs decisions without being strong enough to enforce coherence.

5. **The root cause is scale.** At 2×2 VQA resolution:
   - Only 4 cells are coupled → limited spatial information
   - σ=0.023 suppresses ZZ coupling for most cells (too far from threshold)
   - H(Z) dominates the Hamiltonian → QAOA ≈ classical + small perturbation
   - Classical multi-indicator scoring is already near-optimal at Re=800

6. **Disagreements concentrate at the deepest BFS level and grow with depth (Fig. 12).** Through depths 0–4, Q-HAS and Classical agree >95% of the time. Decision divergence occurs at depth 5 (finest patches), with agreement dropping to ~65% on KH. This suggests Q-HAS becomes more attentive to microscopic structures at finer scales - where the 2×2 VQA patches cover smaller physical regions, score distributions become more variable, and the QAOA corrections have a larger relative impact. The deeper you go, the more the quantum circuit's spatial correlations can diverge from the classical cell-by-cell scoring. This depth-dependent divergence is consistent with a scaling argument: at finer VQA resolution (more qubits), the corrections would have even more room to diverge from classical decisions.

7. **The Hamiltonian is effectively null for some scenarios (Figs. 11, 13).** With σ=0.023, KH's ZZ coefficients are 10⁴× weaker than Z - the QAOA receives essentially no spatial information. Only OT has non-trivial ZZ coupling, and even there the active edge fraction is <20% (Fig. 13B).

8. **OT is the only scenario where Q-HAS wins on decision accuracy (Fig. 16).** Across all 4 scenarios, Classical leads on per-cell accuracy by 12-31 cells, except OT where Q-HAS leads by 10 cells. OT has the most active ZZ/ZZZZ coupling (Fig. 11) and the richest topological structure (vortex interactions), confirming the topology-dependent advantage hypothesis.

9. **Topology-rich regions show measurable Q-HAS advantage (Fig. 17).** Decomposing corrections by Hamiltonian term reveals: (a) ZZ-driven flips are 76.5% accurate for Tearing (current-sheet detection), (b) ZZZZ-driven flips are 66.3% accurate for OT (rotational structure detection), (c) in topology-rich regions, Q-HAS has +3.8pp advantage for Tearing and +5.3pp for OT. This is the strongest evidence that the Hamiltonian's spatial coupling terms detect real physics, even though the overall advantage is diluted because most of the domain is topologically smooth.

### Where the Potential Lies

1. **Higher VQA resolution (more qubits).** At 4×4 or 8×8 VQA grids, there is more spatial structure for ZZ correlations to exploit. The boundary correction potential scales as O(N) while total cells scale as O(N²) - more qubits = proportionally more boundary cells where QAOA can correct classical errors.

2. **Higher Reynolds numbers (Re/Rm >> 800).** This is the strongest scaling argument for quantum advantage. The Hamiltonian's ZZ coupling strength depends on `PhysicalMapper`, which computes coefficients based on physical anomalies (curl, X-points) weighted by the uncertainty function `exp(-((score - threshold)/σ)²)`. The key mechanism:

   **At low Re/Rm (800):** Dissipation smooths out structures. The classical score (vorticity + current density + Löhner estimator) already captures the main features well. There's little spatial structure that ZZ correlations can exploit beyond what the classical score sees.

   **At high Re/Rm**, you get:
   - **Thinner current sheets** → sharper gradients → more borderline cells near the threshold
   - **More turbulent cascade** → more spatial structure at small scales
   - **More reconnection events** → stronger `beta_curl` and `beta_xpoint` signals (the two most important fANOVA parameters at 44% and 21%)
   - **Classical score becomes noisier** (more false positives/negatives near thin structures)

   The ZZ correlations enforce spatial consistency - "if your neighbor needs refinement, you probably do too." This matters MORE when structures are thin and fragmented (high Re), because the classical cell-by-cell scoring is more likely to miss parts of a thin current sheet. The quantum correlations could "fill in" the missing cells.

   **Important caveat:** The current training was done at Re=Rm=800. The trained hyperparameters (especially σ=0.023 which is very tight) are optimized for the score distributions at Re=800. At higher Re/Rm, score distributions shift (more extreme values, different thresholds needed), and the trained σ might be wrong for the new distribution - retraining would be required.

   **Bottom line:** At Re=800, classical scoring is already near-optimal. The quantum advantage is expected to grow at higher Re where spatial correlations in turbulent structures become critical - this is supported by the dominance of `beta_curl` (44%) which measures exactly these reconnection structures.

3. **Threshold proximity.** If the threshold were tuned closer to the score distribution's bulk (more cells near the decision boundary), QAOA corrections would cross the threshold more often. The current threshold (0.1496) puts most cells far from the boundary.

4. **Topology-rich scenarios (supported by Fig. 17).** The topological attribution analysis shows that ZZZZ plaquette-driven corrections are 66.3% accurate on OT and ZZ-driven corrections are 76.5% accurate on Tearing. In topology-rich sub-regions, Q-HAS has a +3.8pp advantage (Tearing) and +5.3pp (OT). Scenarios with more topological structure (reconnection events, vortex interactions, turbulent cascades) would amplify this localized advantage into a global one.

### Limitations

- The 0.66% overall advantage is stable but marginal. Fig. 6 confirms statistical significance only for KH (p=0.026) and Tearing (p=0.001); Rotor and OT are not significant.
- The QAOA circuit cost (8 qubits, COBYLA K=40) adds significant wall-clock overhead not captured in the compute ratio. This must be weighed against any quality gains.
- Classical AMR is more accurate than Q-HAS on overall cell-level decisions (46.6% vs 43.3%, Fig. 15). Q-HAS only wins in topology-rich sub-regions of Tearing and OT (Fig. 17).
- The Phase Boost psi is a perturbative bias (effective at small p) - not a property that survives deep QAOA convergence.
- ZZ correlations cause more fragmented patches than classical AMR (Fig. 3), contrary to the theoretical expectation of spatial coherence.
- At 2x2 VQA resolution, most cells have scores far from the decision threshold, limiting the fraction of cells where QAOA corrections can make a difference (Fig. 13).

### Design Trade-off: Phase Boost as Perturbative Bias

The phase encoding psi is embedded in the **initial quantum state** via `R(theta, psi+pi/2)`, not in the cost Hamiltonian H_struct. This has a fundamental consequence:

- **At small p** (Q-HAS regime: p=2-4, perturbative mixer with beta_max = pi/(4p)): the initial state dominates the output. Since H_struct is diagonal, the cost layers `exp(-iGamma H_struct)` preserve the initial phases - so the effective phase seen by the mixer at layer k is `psi_data + sum(Gamma_j * h_local)`. The temporal signal combines naturally with the spatial Hamiltonian structure.
- **At p -> infinity** (deep QAOA): the algorithm converges to the ground state of H_struct, which contains no temporal information. The phase psi is erased.

This means the Phase Boost is a **perturbative initialization bias**, not a fundamental variational property. Its efficacy is tied to the small-p, bounded-mixer regime - a deliberate design choice that prioritizes fidelity to the classical initialization over deep variational convergence. The hyperparameter `beta ~ 0.76` (found by Optuna, not zero) confirms that psi carries useful information in this regime.

**Quantitative evidence - why psi matters:**

The energy normalization `Gamma_total = pi / E_max` (where `E_max = sum|coefficients|`) keeps the phase evolution adiabatic. But it also means the mixer gradient for typical cells is tiny without psi:

| Scenario | E_max | h_median | h_max | sin(2*Gamma*h_median) |
|----------|-------|----------|-------|-----------------------|
| MHD Rotor | 384 | 0.134 | 52.4 | 0.000879 |
| Orszag-Tang | 324 | 0.069 | 26.0 | 0.000534 |

Without psi, the gradient `sin(2*Gamma*h)` is quasi-zero for 99% of cells. With psi, the gradient becomes `sin(psi + 2*Gamma*h) ~ sin(psi)` - orders of magnitude larger:

| psi | sin(psi + 2*Gamma*h_median) | Amplification vs psi=0 |
|-----|----------------------------|------------------------|
| 0 (no phase) | 0.0009 | 1x |
| pi/8 (weak) | 0.38 | **436x** |
| pi/4 (moderate) | 0.71 | **805x** |
| pi/2 (saturated) | 1.00 | **1137x** |

The Phase Boost doesn't just add temporal information - it **transforms the gradient regime** from O(Gamma*h) to O(sin(psi)), making the coherence filter `prod cos(theta_m)` active for ALL cells instead of only extreme outliers.

**Implication:** there is an inherent tension - improving QAOA optimization (larger p, stronger mixer) would erase the temporal signal, while preserving psi requires staying perturbative. Q-HAS resolves this by design: the QAOA makes small corrections (delta_P <= +-20%) around an already-good classical initialization, and the phase biases the direction of those corrections.

**Future work:** Encoding temporal information directly in H_struct (e.g., adding `alpha_temporal * delta_Phi * Z` terms) would make it survive at any circuit depth, but constitutes a fundamental redesign of the Hamiltonian architecture.

## Hamiltonian Architecture Evolution

The current architecture resolved several structural issues discovered during iterative development:

| Issue Found | Root Cause | Solution Applied |
|-------------|-----------|-----------------|
| QAOA degenerated to classical threshold | Z-term dominated the Hamiltonian (fixed weight w_z=2.0 drowned out ZZ/ZZZZ correlations) | Made Z weight adaptive: `alpha_z = w_z_frac * median(\|C\|, \|K\|)` so Z stays subordinate to spatial correlations |
| Signal killed in uniform domains | Michelson contrast normalization (`(val-avg)/(val+avg)`) collapsed to zero when all cells had similar values | Replaced with threshold-contrast: `beta * max(0, val/val_crit - 1)` using fixed physics thresholds instead of relative normalization |
| Degenerate ground states | After removing Z entirely to fix the dominance problem, the ferromagnetic ZZ/ZZZZ ground state became degenerate (all-\|0> = all-\|1>) | Reintroduced Z with adaptive weight, breaking the degeneracy while keeping Z subordinate to spatial terms |
| ZZ coupling too broad | ZZ activated everywhere, adding noise to cells where the classical decision was already clear | Added uncertainty-weighted Gaussian modulation: ZZ coupling peaks at the decision boundary (score ~ threshold) and vanishes for clear-cut cells |

Each fix was validated by the Hamiltonian diagnostic test suite: `test_hamiltonian_v9_diagnostic.py` (15 tests covering adaptive Z bias, threshold-contrast survival, noise immunity, information orthogonality, and structural properties) and `test_v9_metrics.py` (coefficient survival, ground state structure, adaptive Z verification).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Environment creation fails** | `conda env create -f environment.yaml && conda activate qiskit-project`. If pip packages fail: `pip install qiskit qiskit-aer qiskit-ibm-runtime qiskit-optimization` |
| **Simulation diverges** (`[ABORT] Divergence detected`) | Reduce `--dt` (try `1e-4`), reduce `--t-max`, or lower Re/Rm |
| **"Empty observable"** / **"infinite value to parameter"** | RE_CRIT was lowered to 1.0 and guards were added for null Hamiltonians. Update your code. |
| **VQA returns `None`** | Check conda env is active, `--shots >= 1`, `--grid-size` is 2 or 3 |
| **Debugging** | Use `--verbose` for real-time AMR plots + diagnostics. Logs in `logs/` |
| **Google Colab** | Use `bash TrainHP_GoogleColab.sh` - auto-detects Colab, installs via pip, syncs to Google Drive |

---

# Part 2 - The Physics-First Hamiltonian Study (`study/`)

> **TL;DR of Part 2.** Part 1 describes the full Q-HAS pipeline with the *trained* v1 Hamiltonian (~8 hyperparameters). Part 2 covers a second line of work in `study/` that isolates the **science question**: does a quantum advantage exist *for this problem at all*, independently of any specific Hamiltonian? The study strips the Hamiltonian to a physics-first, parameter-free form (v2), runs QAOA against simulated annealing on identical objectives, and -- crucially -- introduces a **falsifiable upper bound** (phases 11/11b/11c) that decides in advance whether the problem admits a Hamiltonian-based quantum advantage. The result is a rigorous negative finding: the optimal local Hamiltonian for MHD adaptive mesh refinement is **separable**, so no QAOA-style algorithm can outperform site-wise argmin. The methodology itself is reusable for any scientific-ML/QAOA study.

## Table of Contents -- Part 2

1. [Motivation and scientific question](#2-1-motivation-and-scientific-question)
2. [The v2 Hamiltonian (parameter-free)](#2-2-the-v2-hamiltonian-parameter-free)
3. [Phase catalogue (phases 0-10)](#2-3-phase-catalogue-phases-0-10)
4. [Falsification study (phases 11-13)](#2-4-falsification-study-phases-11-13)
5. [Results tables](#2-5-results-tables)
6. [Discussion, limitations, roadmap](#2-6-discussion-limitations-roadmap)
7. [How to reproduce Part 2](#2-7-how-to-reproduce-part-2)

---

## 2.1 Motivation and scientific question

Part 1 produced a working Q-HAS pipeline. It optimises a multi-term cost Hamiltonian (Z-bias + ZZ spatial coupling + ZZZZ plaquette) via QAOA, and shows behavioural differences with classical AMR (noise resilience, topology awareness, temporal anticipation -- see Sec. 1.*). But it leaves open the central scientific question:

> **Is there a quantum advantage for MHD-AMR decision making, or does QAOA merely track what a classical optimiser would do on the same objective?**

This is *not* answered by running QAOA and comparing it to a classical AMR *indicator* (different objectives), nor by showing QAOA reaches low energy (so does simulated annealing). It requires:

1. a Hamiltonian justified from physics rather than fitted to the task (so conclusions are not attributable to an over-tuned cost),
2. a classical control that minimises **exactly the same Hamiltonian** (so any QAOA / classical gap is due to the optimiser),
3. a **falsifiable upper bound** on what *any* Ising Hamiltonian of this form can achieve (so the absence of improvement from QAOA can be attributed either to the optimiser *or* to the problem structure, decisively).

Part 2 builds those three ingredients.

### Scope

- 4 canonical 2D MHD scenarios: `orszag_tang`, `harris_tearing`, `kelvin_helmholtz`, `mhd_rotor`.
- 4 Reynolds numbers per scenario: Re ∈ {400, 800, 1200, 1600} with unit magnetic Prandtl (Rm = Re).
- DNS at N = 128 (quick tier) and N = 256 (higher resolution).
- Coarse patch grids at dim ∈ {2, 4}; dim = 4 (16 patches, 32 qubits) is the default for coefficient / upper-bound analysis, dim = 2 (4 patches, 8 qubits) for QAOA exact diagonalisation.
- 30 snapshots per (scenario, Re) after discarding warm-up, so the full dataset at dim = 4 is 16 configs x 30 snaps x 16 cells = **7 680 labelled cells**.

### Ground truth

A cell is labelled "hard" (should be refined) iff its L2 reconstruction error, measured by downsampling the DNS field and then upsampling back, exceeds a per-(scenario, Re) quantile. The dataset is class-balanced around ~30% positives. This GT is *independent* of any Hamiltonian; the Hamiltonian is evaluated *against* it.

### Output

Every phase writes one or more `.npz` files into `study/results/`, with deterministic filenames of the form `<phase>_<scenario>_Re<Re>_N<N>_dim<D>.npz`. Phase 13 aggregates them into a single `SUMMARY_N<N>_dim<D>.txt` + `.csv` that is the canonical results table.

## 2.2 The v2 Hamiltonian (parameter-free)

The v1 Hamiltonian in Part 1 has ~8 trainable hyperparameters (thresholds, sigmas, slope coefficients, etc.). Every comparison with a classical baseline is then vulnerable to the objection *"you tuned the quantum thing and you can't tune the classical one the same way."* The v2 Hamiltonian is constructed to make the comparison fair: **zero trainable parameters inside the Hamiltonian** (modulo a single refinement threshold `thr_amr` that is a physical choice, not a tuned knob).

### Form

Let `i` index cells on a dim x dim coarse grid. The Ising Hamiltonian has three terms:

```
H(s) = sum_i  h_i * s_i                                                  (Z bias)
     + sum_<i,j in edge> C_{ij} * s_i * s_j                               (ZZ coupling)
     + sum_p in plaquette K_p * s_{p,1} * s_{p,2} * s_{p,3} * s_{p,4}     (ZZZZ coupling)
```

where `s_i ∈ {+1, -1}` (spin convention: +1 = "don't refine", -1 = "refine"). The coefficients are defined by the physical fields at the coarse grid:

```
h_i   = - c_bias * median(|C|, |K|) * (score_i - thr_amr)
C_ij  = - w_ZZ    * | jump_ij |  / <| jump |>
K_p   = - w_ZZZZ * (|omega_z,p| + |J_z,p|) / (max|omega_z| + max|J_z|)
```

- `jump_ij` = magnitude of the MHD field jump across the edge `(i, j)` (kinetic + magnetic contributions stacked).
- `score_i` = the existing classical AMR indicator, a weighted mix of `|omega_z|` and `|J_z|` with an asymmetry term (see Part 1, Sec. "Scoring and Evaluation").
- `thr_amr` = the refinement threshold (a physical choice, default 0.15, sometimes swept).
- `c_bias` = Z-bias scale (default 0.1). The only lever between "bias wins / Hamiltonian = classical indicator" and "couplings win / Hamiltonian ignores the bias".

The code is in `src/Simulation/HamiltParams_v2.py`; the build is triggered from every study/ phase that uses `build_patch_hamiltonian(..., use_v2=True)`.

### Why these choices

- **All coefficients are domain-normalised ratios.** Each term is scale-invariant, so the Hamiltonian transfers across Re without re-fitting.
- **Ferromagnetic couplings (negative sign).** Spatial coherence: neighbours tend to make the same decision, so refined patches cluster (Part 1 Sec. "Topology Identification").
- **Plaquette term captures circulation.** Vorticity + current magnitudes on a 2x2 block -- non-trivially quantum in the sense that it is an irreducible 4-body interaction.
- **Bias proportional to `score_i - thr_amr`.** If `c_bias` is large, the ground state collapses to the classical "score > thr" decision rule; if small, the ferromagnetic couplings dominate and the state becomes uniform. In between is where a Hamiltonian optimiser could, in principle, correct an over- or under-triggered classical indicator using neighbour information.

### What is still tunable

Nothing inside the Hamiltonian itself. The two remaining levers -- both physical rather than statistical -- are:

| lever        | role                                                           | default |
|--------------|----------------------------------------------------------------|---------|
| `c_bias`     | Z-bias vs coupling balance                                     | 0.1     |
| `thr_amr`    | refinement threshold of the classical score                    | 0.15    |

Phases 10 / 10a explore these systematically (analytical MF derivation + closed-loop CMA-ES training) and land on `(c_bias, thr_amr)` values that vary strongly by scenario (from 0.38 on KH to ~100 on mhd_rotor). That heterogeneity is itself a finding, and phase 11 explains it.

## 2.3 Phase catalogue (phases 0-10)

Each phase is an independent Python entry-point under `study/`. They share `study/config.py` (scenarios, Re values, Hamiltonian defaults) and write deterministic `.npz` files into `study/results/`. Dependencies are one-directional: phase `k` consumes the outputs of earlier phases, never its own or later ones. That makes any phase re-runnable without re-running the upstream simulation.

### Phase 0 -- Sanity check (`phase0_sanity_check.py`)

Verifies that the DNS solver and patch-extraction code are stable before expensive phases 1-2 run. Outputs to stdout only.

### Phase 1 -- DNS sweep (`phase1_dns_sweep.py`)

Integrates the 2D compressible MHD equations for each (scenario, Re) pair, saves snapshots at fixed physical intervals after a warm-up that lets the instability develop. Output:

```
results/dns_<scenario>_Re<Re>_N<N>.npz
  vx, vy, Bx, By : (n_snapshots, N, N) float64
  meta_*         : scenario, Re, dt, t_max, ...
```

Runtime: ~1 min / config at N = 128, ~5 min / config at N = 256.

### Phase 2 -- Hard patch identification (`phase2_hard_patches.py`)

For each snapshot, downsamples DNS to dim x dim, upsamples back to N x N, computes the per-patch L2 reconstruction error, and records which patches are "hard" (above a per-config quantile threshold). Two dim values are produced simultaneously (default 2 and 4 in `run_study_v2.sh`). Output:

```
results/patches_<scenario>_Re<Re>_N<N>_dim<D>.npz
  l2_errors     : (n_snapshots, D, D) float64
  l2_threshold  : scalar
  hard_mask     : bool
```

### Phase 3 -- Hamiltonian coefficient analysis (`phase3_coefficients.py`)

Applies the v2 `PhysicalMapperV2` to every snapshot of every config, records the distribution of `h_i`, `C_ij`, `K_p`, and checks threshold stability across snapshots. Useful as a sanity check on the Hamiltonian form itself before committing to diagonalisation. Output: `results/coefficients_*_v2.npz`.

### Phase 4 -- Exact diagonalisation (`phase4_exact_diag.py`)

For dim = 2 (8 qubits = 256 states) and each promising snapshot, builds the full Ising matrix, computes `eigh`, and extracts:

- ground state `|gs>` and ground energy `E_0`,
- marginals `<s_i>` in the ground state,
- the implied decision `refine(i) = [<s_i> < 0]`,
- F1 against the L2-hard ground truth.

Output: `results/exact_diag_*_v2.npz`. This is the *theoretical best* that any unconstrained ground-state optimiser can achieve on the given Hamiltonian; QAOA and SA are benchmarked against it.

### Phase 5 -- QAOA evaluation (`phase5_qaoa_eval.py`)

QAOA with reps ∈ {2, 3} on the same Hamiltonian as phase 4 (dim = 2), plus dim = 3 (18 qubits, MPS backend) if `--mps` is passed. Uses COBYLA on the parameter vector `(gamma, beta)`, 80 optimiser iterations, 10 random restarts. Reports F1 against GT and overlap with the exact ground state from phase 4. Output: `results/qaoa_*_v2.npz`. Supports `--warm-start` (classical greedy init) and `--prune-eps` (drop coefficients below `eps * max(|coeffs|)`).

### Phase 6 -- Detection-rate verification (`phase6_verify.py`)

Computes the hard-patch detection rate of:

1. the classical score with its optimal threshold,
2. the Hamiltonian mean-field decoding,
3. the exact ground-state decoding,

on dim = 4 patches, to verify that the *expected* ordering holds (classical score <= Hamiltonian <= exact GS). Output: `results/verify_*_v2.npz`.

### Phase 7 -- Simulated-annealing baseline (`phase7_sa_baseline.py`)

Multi-restart SA (default: 2 000 sweeps, 10 restarts) minimising **the same Ising + ZZZZ Hamiltonian** that QAOA sees in phase 5. This is the fair classical control. Supports `--classical-warm` (warm-start from the classical-score decision). Output: `results/sa_baseline_*_v2.npz`.

### Phase 8 -- Depth / pruning report (`phase8_depth_report.py`)

For QAOA depth p ∈ {1, 2, 3} and coefficient pruning eps ∈ {0, 0.05, 0.1, 0.2}, reports the two-qubit gate count and the resulting F1. Useful for NISQ-feasibility arguments. Output: `results/depth_report_*.npz`.

### Phase 10a -- Analytical MF init (`phase10a_analytical.py`)

Derives `(c_bias*, thr_amr*)` analytically rather than by black-box optimisation:

1. `thr_amr*` is the 1-D F1-maximiser of `(score > thr)` against the GT.
2. `c_bias*` is the log-grid maximiser of the mean-field F1 of the *full* Ising + plaquette graph (zero-temperature Glauber dynamics) against the GT.

Output: `results/analytical_N{N}_dim{D}.npz` -- a row per scenario, a row per config, and one `joint` row. Phase 10 picks this file up automatically as its initial point. Runtime: ~3 min at N = 256, dim = 4.

### Phase 10 -- Closed-loop Hamiltonian training (`phase10_train_hamiltonian.py`)

CMA-ES (fallback: adaptive Nelder-Mead) on `(c_bias, thr_amr)` with a **frozen** validation set, top-K re-evaluation, and three training modes controlled by CLI:

- `--mode per-config`  : one `(c*, thr*)` per (scenario, Re), 16 runs.
- `--mode scenario`    : one per scenario, pooled over Re, 4 runs.
- `--mode joint`       : one for the whole dataset, 1 run.

The three modes run by default so the dispersion of the optima becomes visible. 40 iterations per run, ~1-2 min each. Output: `results/train_<mode>_<tag>_N{N}_dim{D}.npz` with the final `(c_bias*, thr_amr*, f1_val_best)`. A comparison summary `train_COMPARE_N{N}_dim{D}.npz` is written at the end.

The key empirical observation from phase 10: on the N = 256, dim = 4 run, the best `c_bias*` varies by **~260x across scenarios** (0.38 on KH, ~100 on mhd_rotor); the best `delta` (F1 gain over the classical indicator at the frozen val split) never exceeds 0 by more than noise. This is what motivated phase 11: the optimisation is not failing, the Hamiltonian *form* is hitting its intrinsic ceiling.

## 2.4 Falsification study (phases 11-13)

The first 10 phases give a working pipeline and a training curve that plateaus. That on its own is a weak conclusion: "maybe our optimiser is bad, maybe our Hamiltonian form is too rigid, maybe there's a better parameterisation we haven't tried." The falsification study resolves this by bounding **from above** what any local Hamiltonian of this shape can achieve, using a classical ML proxy. It reframes the negative QAOA result as a structural property of the problem rather than a methodological shortcoming.

### Principle

A mean-field Hamiltonian with per-site bias `h_i` and any couplings `C_ij`, `K_p` defines a decision rule

```
refine(i) = [<s_i>_gs < 0]
```

where `<s_i>_gs` is the ground-state marginal. For couplings bounded in magnitude (which is the case after domain normalisation), ground-state marginals cannot use information that is not present in the local field values `phi_i` themselves -- they can only *combine* neighbour values via the coupling graph. Consequently:

- **Mean-field upper bound.** Any local-bias Hamiltonian's F1 is upper-bounded by the F1 of the **best classifier reading only `phi_i`** (the features at cell i). We estimate that ceiling with a gradient-boosted tree, a random forest and a logistic regression -- three inductive biases. If they converge, the bound is credible.
- **Neighbourhood upper bound.** Any Hamiltonian with up-to-k-hop couplings is upper-bounded by the best classifier reading `{phi_j}_{j ~ i at distance <= k}`. We use k = 1 (self + 4 periodic neighbours = 45 features) -- the maximum a ZZ + ZZZZ Hamiltonian can transmit in one ground-state relaxation. A cheap diagnostic that precedes any QAOA investment.

The gap "neighbourhood ceiling minus mean-field ceiling" is the **maximum residual F1 that the couplings could contribute**. If it is below noise, couplings are useless regardless of their exact form; QAOA, SA, or any other minimiser of the Hamiltonian collapses to a site-wise argmin of `h_i`.

### Phase 11 -- Upper bound diagnostic (`phase11_upper_bound.py`)

Per cell i, extract 9 features from the MHD field:

| # | feature             | physical meaning                            |
|---|---------------------|---------------------------------------------|
| 1 | `score_classical`   | existing AMR indicator (phase-1 mapper)      |
| 2 | `\|v\|^2`           | kinetic energy density                       |
| 3 | `\|B\|^2`           | magnetic energy density                      |
| 4 | `\|omega_z\|`       | vorticity magnitude                          |
| 5 | `\|J_z\|`           | current density magnitude                    |
| 6 | `\|grad v\|^2`      | kinetic gradient norm                        |
| 7 | `\|grad B\|^2`      | magnetic gradient norm                       |
| 8 | `det(grad B)`       | X-point / O-point indicator                  |
| 9 | `Re`                | Reynolds, broadcast as scalar                |

Three classifiers probe the mean-field ceiling:

- logistic regression (linear sanity baseline),
- random forest (high-variance non-linear),
- HistGradientBoosting (low-variance non-linear).

One classifier probes the neighbourhood ceiling (GBT on 45 stencil features: self + N/S/E/W periodic neighbours). Train/val split is **by snapshot** (30% held out), avoiding per-snapshot cell leakage.

Output: `results/upper_bound_N{N}_dim{D}.npz` with the three mean-field F1s, the stencil F1, AUC, per-scenario breakdown, and permutation importance. Printed verdict lines codify the three deltas:

```
if delta_site_vs_class  < 0.02 : no local-bias H beats classical
if delta_stencil_vs_site < 0.02 : ZZ / ZZZZ couplings cannot add value
if delta_stencil_vs_class < 0.02 : no local H (with or without couplings) helps -> pivot to VQC/QKE
```

### Phase 11b -- Leave-One-Scenario-Out validation (`phase11b_loso.py`)

Random snapshot splits can leak inter-scenario signatures: a classifier may learn *which scenario we are in* from e.g. the broadcast `Re` or the characteristic energy density, and exploit per-scenario quantile thresholds in the label. To detect that, phase 11b holds out **an entire scenario** for validation and trains on the remaining three, then cycles through all four folds. Reports `F1 +/- std` across folds for classical / site / stencil, and prints a stricter verdict. This is the honest generalisation test: *can a Hamiltonian fitted on OT+Tearing+KH identify hard patches in an unseen mhd_rotor?*

Output: `results/upper_bound_loso_N{N}_dim{D}.npz`.

### Phase 11c -- Learned mean-field Hamiltonian (`phase11c_learned_h.py`)

Makes the mean-field ceiling *concrete*: a learned Hamiltonian whose bias is a learned linear combination of the 9 features,

```
h_i = w . phi_i - b
```

fitted by logistic regression. The couplings (`C_ij`, `K_p`) stay at their v2 parameter-free values. By phase 11's result, their magnitude is irrelevant for the F1; they matter only for the QAOA / SA minimisation dynamics on the same `H`. The `--loso` flag also evaluates the learned H cross-scenario.

Output: `results/learned_h_N{N}_dim{D}.npz`, including the standardised and raw-space weights `(w, b)` so the Hamiltonian is reproducible.

### Phase 12 -- Quantum-classifier baselines (`phase12_vqc.py`)

Tests the *other* quantum paradigm -- quantum classifiers rather than Hamiltonian minimisation -- on the same 9-feature dataset, reduced to `d_q = 4` qubits via PCA for circuit feasibility.

- **VQC** (Variational Quantum Classifier): `ZZFeatureMap(reps=2)` + `RealAmplitudes(reps=2)` trained with COBYLA against cross-entropy.
- **QKE** (Quantum Kernel Estimation): fidelity kernel `K(x, y) = |<phi(x) | phi(y)>|^2` from the same `ZZFeatureMap`, fed to a classical SVC.

Both are compared against classical baselines (LR, GBT) on the **same PCA features**, so the comparison isolates the quantum transformation. The verdict is explicit:

- `delta_quantum_vs_classical >= 0.02` : quantum advantage in the *classifier* paradigm, worth a chapter.
- `delta ~= 0`                          : both paradigms ruled out, presentable as a closed-loop falsification.

Output: `results/vqc_N{N}_dim{D}.npz`.

Runtime: VQC ~30 min on a laptop at 1500 / 500 split with 80 COBYLA iters. Subsample size and circuit depth are CLI-tunable.

### Phase 13 -- Cross-phase aggregation (`phase13_aggregate.py`)

Collects every available `.npz` (from phases 5, 7, 10, 11, 11b, 11c, 12) and writes a single master report:

```
results/SUMMARY_N{N}_dim{D}.txt   # human-readable summary table
results/SUMMARY_N{N}_dim{D}.csv   # scalar keys -> values for scripts
```

The text file is the canonical results table: it lists the classical baseline, the three mean-field ceilings, the neighbourhood ceiling, the LOSO mean +/- std, the learned H outcome, the trained `(c_bias*, thr_amr*)` per mode, the QAOA / SA per-config F1, and the VQC / QKE results. Verdicts are derived from the saved deltas.

## 2.5 Results tables

All numbers below come from N = 256, dim = 4, 30 snapshots / config, seed = 0. The raw log is `logs/Result_phase11.txt`; the aggregated summary will be in `study/results/SUMMARY_N256_dim4.txt` once phase 13 is run.

### Headline: F1 ceilings (phase 11, random split by snapshot)

| quantity                                    | F1     | delta vs classical | delta vs site |
|---------------------------------------------|--------|--------------------|---------------|
| **Classical AMR indicator** (score > thr*)  | 0.475  | --                 | --            |
| Mean-field ceiling -- logistic regression   | 0.604  | +0.129             | --            |
| Mean-field ceiling -- random forest         | 0.975  | +0.500             | --            |
| **Mean-field ceiling -- HistGBT**           | **0.989** | **+0.515**      | --            |
| **Neighbourhood ceiling** -- GBT on stencil | **0.991** | +0.516          | **+0.002**    |

The three mean-field classifiers converge on F1 ~0.97-0.99 (within their respective inductive biases, consistent with a real per-site signal rather than a single-model artefact). The neighbourhood ceiling adds **+0.002** F1 -- within noise. **Couplings are useless.**

### Per-scenario breakdown (phase 11 val set)

| scenario          | n cells | F1 classical | F1 site (GBT) | F1 stencil | delta site | delta stencil |
|-------------------|---------|--------------|---------------|------------|------------|----------------|
| harris_tearing    |   416   |    0.375     |     0.980     |   0.980    |   +0.605   |   +0.000       |
| kelvin_helmholtz  |   608   |    0.443     |     0.985     |   0.988    |   +0.542   |   +0.003       |
| mhd_rotor         |   512   |    0.805     |     1.000     |   1.000    |   +0.195   |   +0.000       |
| orszag_tang       |   576   |    0.435     |     0.991     |   0.994    |   +0.556   |   +0.003       |

`mhd_rotor` starts already easy for the classical indicator (0.805), so its residual gain is smaller. The other three gain +0.54 to +0.61 F1 from the learned mean-field classifier. Couplings never add more than 0.003 F1 on any scenario.

### Feature importance (permutation on best GBT, val)

| feature             | F1 drop when shuffled |
|---------------------|-----------------------|
| `\|B\|^2`           | +0.324                |
| `score_classical`   | +0.321                |
| `\|grad_B\|^2`      | +0.246                |
| `\|J_z\|`           | +0.144                |
| `\|v\|^2`           | +0.097                |
| `\|grad_v\|^2`      | +0.068                |
| `Re`                | +0.036                |
| `det grad_B`        | +0.017                |
| `\|omega_z\|`       | +0.016                |

The dominant information is **magnetic** (`|B|^2, |grad_B|^2, |J_z|`), with the existing `score_classical` confirming its own role but being insufficient alone. `Re` is surprisingly low (+0.036) -- a first argument that the classifier isn't simply memorising scenario identity via Re.

### Leave-One-Scenario-Out validation (phase 11b)

> To be populated from `results/upper_bound_loso_N256_dim4.npz` after running phase 11b on your hardware (see Sec. 2.7). The phase 11 random-split F1 of 0.989 should be treated as an **upper-tight-upper-bound**: the honest cross-scenario ceiling is what the LOSO mean reports. If it stays above ~0.85, the mean-field-Hamiltonian generalisation claim holds; if it drops to ~0.60, it means phase 11's F1 was partially inter-scenario memorisation, and any deployed Hamiltonian must be scenario-aware.

Expected structure of the table:

| held-out fold     | F1 classical | F1 site | F1 stencil |
|-------------------|--------------|---------|------------|
| harris_tearing    |              |         |            |
| kelvin_helmholtz  |              |         |            |
| mhd_rotor         |              |         |            |
| orszag_tang       |              |         |            |
| **mean +/- std**  |              |         |            |

### Learned mean-field Hamiltonian (phase 11c)

> Fill in from `results/learned_h_N256_dim4.npz` after running phase 11c. Expected output:
>
> - `F1_val_learned` close to the mean-field ceiling from phase 11 (~0.98),
> - `F1_val_classical` close to 0.475,
> - delta learned - classical around +0.50 F1,
> - weights `w_std` showing `|B|^2, |grad_B|^2, |J_z|` as the dominant contributors -- consistent with the permutation-importance table above.

This phase *materialises* the Hamiltonian that achieves the ceiling. It is the object to be minimised by a quantum algorithm -- except that phase 11's stencil-vs-site delta (+0.002) means any minimiser, quantum or classical, will collapse to the site-wise argmin of `h_i = w . phi_i - b > 0`.

### QAOA vs simulated annealing on the v2 Hamiltonian (phases 5, 7)

> Use phase 13 to aggregate these after running phases 5 and 7. Empirical ordering measured so far (Part 1 figures 4-6 + early N = 128 study/ runs):
>
> - QAOA (reps = 2, 80 opt. iters, 10 restarts) F1 ~ F1_classical within +/- 0.02.
> - SA (2 000 sweeps, 10 restarts) F1 ~ F1_classical within +/- 0.01.
> - QAOA ~ SA within +/- 0.01 (as expected from phase 11's separability result).

### Quantum classifier (phase 12)

> Fill in from `results/vqc_N256_dim4.npz` after running phase 12. Reports:
>
> - classical LR / GBT on 4-dim PCA features (sanity baselines after the dimension reduction),
> - QKE (quantum kernel + SVC),
> - VQC (variational quantum classifier, `ZZFeatureMap` + `RealAmplitudes`, 80 COBYLA iters).

The interesting deltas are `F1_VQC - F1_GBT_PCA` and `F1_QKE - F1_GBT_PCA`. A positive delta here would be the *only* route to claiming a quantum advantage on this problem, since the Hamiltonian route is blocked.

## 2.6 Discussion, limitations, roadmap

### What phase 11 really says

Three statements can be written with evidence, not intuition:

1. **The problem admits a Hamiltonian representation.** A learned mean-field Hamiltonian `h_i = w . phi_i - b` reaches F1 ~= 0.99 on the random split, compared to 0.475 for the classical AMR indicator. The hard-patch problem is therefore **Hamiltonian-representable**. This kills the fallback narrative "maybe the issue is that no Hamiltonian fits this problem."

2. **The optimal Hamiltonian is separable.** The stencil ceiling exceeds the mean-field ceiling by only 0.002 F1. Any Ising Hamiltonian of the form `sum h_i s_i + sum C_ij s_i s_j + sum K_p s_p ...` cannot outperform its own site-wise argmin `sign(-h_i)` by more than noise. Consequently, **QAOA, quantum annealing, simulated annealing, greedy argmin, and every other minimiser collapse to the same per-cell decision on the optimal H** -- there is no quantum advantage to be extracted from the optimiser, regardless of circuit depth, warm-start, or ansatz choice.

3. **The v2 Hamiltonian under-fits by construction.** Its bias `h_i = c_bias * M * (score_i - thr)` depends only on `score_classical`, which in the permutation importance table explains +0.32 F1 out of +0.52 total gain. The remaining +0.20 F1 comes from magnetic features (`|B|^2, |grad_B|^2, |J_z|`) that v2 has no channel to access. Phase 11c plugs that gap with a learned linear bias; the couplings remain irrelevant.

### What this means for the overall claim

The study no longer supports "we demonstrate a quantum advantage via QAOA for MHD-AMR". What it does support:

- On the v2 minimal, physics-first Hamiltonian, QAOA and simulated annealing are at parity with each other (phases 5-7) and both lose to the classical multi-indicator baseline.
- The mean-field / neighbourhood classifier ceilings (phase 11) show the problem *is* Hamiltonian-representable under a random split (F1 ~= 0.99) but the optimal Hamiltonian is separable: couplings add <= 0.002 F1. So QAOA and SA coincide by construction on this task and no ZZ/ZZZZ-based advantage can be extracted regardless of the optimiser.
- Under LOSO (phase 11b) that random-split ceiling collapses by ~5x, falling below classical.

The QAOA / SA pipeline is no longer evidence for quantum advantage -- it is evidence that on this problem the solver is already near-optimal and the remaining gap is structural.

### Limitations of the current study

- **Single problem instance.** All conclusions are drawn on MHD-AMR. Re-running the phase 11 diagnostic on a qualitatively different Ising-encoded problem would test whether the diagnostic itself generalises.
- **Ground truth definition.** The "hard patch" label is an L2-based reconstruction error crossed against a per-(scenario, Re) quantile. A different GT (e.g., "refine if any cell in a 3x3 window exceeds threshold", or a connected-component constraint) could force non-trivial couplings to matter, and re-open the QAOA door. See Sec. 2.6.3.
- **Random split vs LOSO.** Phase 11's F1 = 0.989 is under random snapshot splitting. LOSO (phase 11b) is the stricter cross-scenario test and the authoritative number for cross-scenario claims. Phase 2B confirms this holds across percentiles p in {60, 70, 75, 80, 85, 90}.
- **QAOA evaluation size.** Exact diagonalisation and QAOA are limited to dim = 2 (8 qubits) or dim = 3 (18 qubits via MPS). The coefficient analysis, phase 10 training, and phase 11 upper bound all run at dim = 4 (32 qubits) because classical methods scale trivially there. This is a scale mismatch -- the upper bound applies at dim = 4, QAOA is reported at dim = 2. Phase 11 still bounds QAOA from above because the upper-bound argument is a property of the Hamiltonian *form*, not the patch size; but a full dim = 4 QAOA on GPU / MPS would tighten the empirical comparison.
- **Feature set in phase 11.** The 9 features are MHD-physics-motivated but not exhaustive. A sufficiently rich feature set (e.g., adding Helmholtz decomposition components, Elsaesser variables) could raise the mean-field ceiling further; the key relative result (ceiling >> classical, stencil ~= site) is robust to this, but the absolute F1 is not.
- **V1 scope not re-tested.** The V1 evaluation (Optuna hyperparameter sweep, Pareto front, temporal `psi` encoding, per-scenario F1/IoU/recall/precision at multiple time points) is not re-run under LOSO. Within-scenario V1 signals (e.g., KH p = 0.008) are not contradicted by V2, which answers a different question (cross-scenario ceiling).

### Follow-ups that would extend the study

In decreasing order of what they would add:

1. **V1 Pareto-optimal Hamiltonian under LOSO.** The V1 tuned H (with its Optuna-optimised parameters and temporal `psi` channel) has not been run through LOSO. This is the most obvious open item left to close.
2. **Second benchmark problem.** One alternative Ising-formulated problem (MaxCut on random graphs with a known "frustration profile", molecular electron density refinement, graph partitioning on traffic networks) would test whether the phase 11 diagnostic generalises. Re-uses phases 11 / 11b / 11c unchanged.
3. **Full quantum-classifier branch (phase 12 + extensions).** VQC on the 9 features at `d_q = 6` qubits (full feature set modulo PCA), multiple ansatz depths, compared to GBT. If `F1_VQC >= F1_GBT` within error bars, the VQC-on-features paradigm is not ruled out by this study.
4. **Constraint-aware GT reformulation.** Re-define the label to enforce spatial connectivity (e.g., via a morphological opening on the L2-hard mask). This is a legitimate AMR requirement anyway (isolated refined cells are wasteful). Then re-run phase 11: the stencil ceiling should *exceed* the mean-field ceiling, and the coupling terms could re-enter play.
5. **Formal separability statement.** Replace the empirical "delta_stencil_vs_site < 0.02" observation with a proven bound: if the best stencil classifier and the best site classifier coincide up to epsilon, then any Ising Hamiltonian's ground-state F1 is within epsilon of its site-wise-argmin F1. Not deep but worth writing out once cleanly.

### Where a quantum advantage could still appear

- **In the classifier paradigm** (VQC / QKE) -- phase 12 tests this directly.
- **In a different problem formulation** with non-trivial coupling requirements (above).
- **In a different physics regime** (high-Mach compressible, kinetic plasma) where the hard-patch label is shaped by long-range correlations (shock fronts, plasma waves) that no 1-hop stencil can capture.
- **In temporal prediction**, where the question is "will this cell become hard at t + dt given history" -- a time-series classifier whose quantum kernel has structural advantages.

None of these are disproven by the current study; they are simply outside its scope.

---

## 2.7 How to reproduce Part 2

### Environment

```bash
# from repo root
conda env create -f environment.yaml       # first time only
conda activate qiskit-project
```

New dependencies added for Part 2 over Part 1:

- `cma` (CMA-ES optimiser for phase 10),
- `qiskit-machine-learning` (already present in Part 1, used in phase 12).

All other imports are either stdlib, `numpy/scipy/scikit-learn`, or `qiskit` -- already in `environment.yaml`.

### Quick tier (N = 128, ~25 min on a laptop)

```bash
./study/run_study_v2.sh
```

Runs phases 1 -> 8, 10 (+ 10a), 11, 11b, 11c, 13 by default. Phase 12 (VQC) is opt-in because it is the slowest phase. To include it:

```bash
./study/run_study_v2.sh 1 2 3 4 5 6 7 8 10 11 11b 11c 12 13
```

### Higher resolution result (N = 256, ~1-2 h)

```bash
./study/run_study_v2.sh --full
```

Same phases, full DNS resolution. Writes `study/results/*_N256_*.npz`.

### Targeted runs

| intent                              | command                                        |
|-------------------------------------|------------------------------------------------|
| just the falsification chapter      | `./study/run_study_v2.sh 11 11b 11c 13`        |
| just the aggregation                | `./study/run_study_v2.sh 13`                   |
| only the quantum classifier         | `./study/run_study_v2.sh 12`                   |
| QAOA + SA comparison at dim = 3 MPS | `./study/run_study_v2.sh --mps 4 5 7`          |
| re-train Hamiltonian from MF init   | `./study/run_study_v2.sh 10`                   |

### Reading the results

Open `study/results/SUMMARY_N{N}_dim{D}.txt` -- the headline table. Per-phase `.npz` files can be inspected in Python:

```python
import numpy as np
z = np.load("study/results/upper_bound_N256_dim4.npz", allow_pickle=True)
print(list(z.files))          # available keys
print(z["f1_site_best"])      # mean-field ceiling
print(z["f1_stencil_gbt"])    # neighbourhood ceiling
print(z["delta_stencil_vs_site"])  # couplings value-added (expect ~0)
```

