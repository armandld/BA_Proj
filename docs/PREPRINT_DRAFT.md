# Searching for topological instabilities in 2D MHD with an Ising Hamiltonian and QAOA: complete results of the Q-HAS project

**Status:** draft. Author names, affiliations, and funding statement are not
filled in. Every number below was re-checked on 5 October 2026 against the
repository at commit `294319d0e179b0a2d3e2afa06a42da3b974090bc` (branch
`claude/project-evaluation-mit-preprint-t6f3hp`), with the pinned environment
of `requirements.txt`: either recomputed from its artifact, or pinned by an
automated test that was run on that commit (605 passed, one expected
failure; Section 12). The measurements added by this review (the readout
diagnostic and the recentring plan of Section 5.2) were produced from clean
working trees at commits `d2a43c2` and `420404a` and are pinned by their own
tests. Each result names its script, its test where one exists, the date of
the artifact it comes from, and its evidence level (Section 2.8).

---

## Abstract

Q-HAS tries to find, inside a chaotic two-dimensional magnetohydrodynamic
(MHD) simulation, the topological structures of the velocity and magnetic
fields that need a finer mesh — shear layers, vortices, current sheets and
magnetic X-points — by encoding them as a local Ising Hamiltonian on the
cells of a coarse grid and letting QAOA choose which cells to refine. This
report gives every result the project produced, positive and negative, with
its evidence level.

What works. The solver and its reference simulations are validated (div B at
machine precision, reconnection observed on 6 of 6 Harris current-sheet
runs), with one documented limit: the scheme converges at first order in
time. After a series of measured corrections to the physics-to-Ising mapping
— the "vorticity" and "divergence" indicators had been computing strain
components, the X-point detector was exactly zero at the null it was meant
to flag, the magnetic gate compared quantities in two unit systems, and the
four-body terms vanished at the training resolution — the Hamiltonian's
sensors respond to the structures they are designed for. The X-point term is
the only type-selective sensor; on a Harris current sheet the plaquette
coefficient ranks blocks by their true coarse-grid error about as well as
the classical indicator (Spearman ρ = 0.80 against 0.81).

What does not. Turning those sensors into a refinement decision through the
Ising ground state does not help. At the only lattice size where the exact
optimum is both enumerable and non-degenerate (18 qubits), QAOA reaches the
optimum of its own Hamiltonian on at most a quarter of instances, and never
on the trained mapper (H0a); solving the
Hamiltonian better makes the decision worse (H0b): the exact optimum
never decides better than the classical rule at the same threshold, on
both mapper implementations, and ρ(energy gap, F1) is positive on the
trained mapper (+0.891, replayed bit for bit on the current code). This
review tested the obvious repair. The bias is centred on a threshold
inherited from the deployed pipeline (0.15), far below the one that
maximises F1 for this label (0.51-0.60); recentring it does not turn ρ
negative (+0.83 to +0.98) and does not make the optimum a better decision,
because the couplings drive the ground state toward a uniform mask (on the
parameter-free mapper, uniform on 24 of 24 instance-threshold pairs; the
threshold only picks refine-everything or refine-nothing). The QAOA
decision, as the study code reads it, gives the same F1 as a fixed
threshold of 0.5 on the classical score on 11 of 12 instances of the
trained mapper. The coupling terms never improve the exact decision and
lower F1 by 0.033 to 0.065 once they are not inert (H3). On real
simulations, four instability
scenarios × four Reynolds numbers, 40 held-out snapshots per fold, a
classical threshold beats or ties both QAOA and a trained gradient-boosted
tree on every fold, and beats QAOA with a 95% bootstrap interval excluding
zero on three of four. The temporal phase meant to anticipate instabilities
does not improve the decision.

What cannot be claimed. A closed-loop comparison inside the adaptive solver
was run and found Q-HAS less accurate than a budget-matched classical rule on
every completed run, but its artifacts predate a defect that affected only
the Q-HAS arm; it is reported as historical, not as evidence. No full
hyperparameter campaign was run (cost), transfer to unseen conditions (H4)
remains a conjecture, and two synthetic-model comparisons were found, during
this review, to compare QAOA and the exact optimum on two different
Hamiltonians. More than 190 contract defects were found and closed by a
systematic audit; the methodological lessons are reported as results in
their own right.

## Presentation

Q-HAS (Quantum-Hierarchical Adaptive Steering) originates from an
undergraduate project at Imperial College London (SPC-Appelbe-1, supervised
by Dr. Brian Appelbe), which constructed the mapping from coarse-grained
MHD observables to qubit states, defined the structured Ising Hamiltonian
used throughout this report, and implemented the full hybrid software
pipeline. That project's own evaluation, at 2×2 VQA resolution (8 qubits,
N = 256), reported a 0.66% composite-loss advantage for Q-HAS over
classical AMR (0.2134 vs 0.2148, 170+ Optuna trials), positive and
statistically significant on two of four canonical scenarios
(Kelvin-Helmholtz, Δ = +0.004, p = 0.026; Harris Tearing,
Δ = +0.003, p = 0.001), negative and statistically significant on the
other two (MHD Rotor, Δ = -0.010; Orszag-Tang, Δ = -0.096). On
aggregate per-cell decision accuracy, classical AMR was already ahead
(46.6% of 2136 cells vs 43.3%). The strongest positive signal was localised
to topologically rich sub-regions: +3.8 and +5.3 percentage points on
Tearing and Orszag-Tang respectively. That report's own abstract closed
with the question this one answers: "Whether the localised advantage
scales to larger VQA grids or higher Reynolds numbers remains an open
question."

The present report carries that test out, with one methodological
correction made necessary by the scaling itself: at 2×2 resolution the
exact ground state of the Hamiltonian is, on every tested instance, the
trivial "refine everything" state, independent of the Hamiltonian's own
coefficients (Section 4.5) — a degeneracy not identified in the earlier
project, and one that makes the 2×2 point uninformative about whether the
framework's optimisation or representation choices are sound, prior to any
question of scale. `dim = 3` (18 qubits) is the smallest resolution above
this degenerate point at which the exact ground state remains exhaustively
enumerable, and is used throughout as the certified
reference size. Scaling further, to `dim = 4` and `dim = 8`, does show the
coupling terms' relative influence growing with resolution, exactly as the
earlier report's further-work section anticipated — but in the opposite
direction: the gap between the full Hamiltonian and a bias-only version
widens from -0.033 F1 at `dim = 4` to -0.057 at `dim = 8` (Table 4),
rather than closing. The remainder of this report gives the full
measurement, on the certified size and, in Section 5.4, replicated at a
larger sample size directly on real DNS.

## 1. Introduction

### 1.1 The problem: finding the structures that need resolution

A two-dimensional MHD simulation at Reynolds numbers of a few hundred to a
few thousand becomes chaotic quickly: thin current sheets form and tear,
shear layers roll up into vortices, magnetic islands merge at X-points.
These are topological features of the velocity field v and the magnetic
field B — places where vorticity ω = ∂x v_y − ∂y v_x or current density
J = ∂x B_y − ∂y B_x concentrates, or where magnetic field lines change
connectivity. The accuracy of the whole simulation depends on resolving
them, and at any time they occupy a small, moving fraction of the domain.
Adaptive mesh refinement (AMR) refines only where it is needed; the hard
part is the criterion that decides where. Classical criteria threshold a
local indicator (a gradient, an error estimator, vorticity, current) and
react once a gradient has already formed.

### 1.2 The Q-HAS idea

Q-HAS (Quantum-Hierarchical Adaptive Steering) encodes the local state of
the plasma on the cells of a coarse grid as an Ising Hamiltonian whose terms
are designed as sensors for specific instabilities (Appendix A): a
single-site bias built from the classical score, a two-site coupling across
strong jumps (shear layers, shocks), a four-site plaquette around local
circulation (vortices, current sheets), and a four-site term at magnetic
X-points (reconnection). Each qubit is initialised from the classical score
(its amplitude) and from the local growth rate of the stress flux (its
phase), and QAOA is asked for a low-energy configuration, which is read as
the refinement decision. Two expectations motivated the design: that the
couplings would add neighbour information a local threshold does not have,
and that the phase would let the circuit anticipate an instability before
its amplitude becomes large.

### 1.3 What this is not

This is not a test of quantum computational advantage in the complexity
sense. At the lattice sizes used here (up to `dim = 8`, 128 qubits, and
principally `dim = 3`, 18 qubits) the Hamiltonians are exactly
diagonalizable or well approximated on a classical machine in seconds to
minutes. We make no claim, implicit or explicit, that this problem is
classically hard. The question is narrower: does a shallow NISQ circuit
(`reps` 1 to 6, a few dozen classical optimizer evaluations), used as a
refinement decision rule, do better than an inexpensive classical heuristic
on this specific task? That is a legitimate empirical question independent
of classical tractability, and it is the only one this work answers. All
quantum results are noiseless statevector simulations; no hardware run
exists (Section 2.4).

### 1.4 A prerequisite: verifying that the implementation computes what it claims

A comparison between two decision rules is only informative if both are
implemented as documented. Before and while the results below were
produced, the code was audited systematically — five questions applied to
every function on the decision path (what does it promise, what does it
consume, does it fail loudly when its assumptions break, do two paths that
should agree still agree, can the test that guards it actually fail) —
rather than by line-by-line review. This found and closed more than 190
contract defects, from a curl operator written in the wrong axis convention
to acceptance criteria that were printed but never compared. Several of
them changed which decision a criterion made, and some of them changed the
interpretation of results that had already been written down; Section 7
reports the audit and its lessons as results.

### 1.5 What this report contains

The results, in the order they are presented:

1. The reference solver and its simulations are validated, with one
   documented limit: first-order convergence in time (Section 4.1).
2. After a series of measured corrections, the Hamiltonian's sensors
   detect the structures they were designed for; only the X-point term is
   type-selective (Section 4.2).
3. The plaquette coefficient points at the blocks that need refinement
   about as well as the classical indicator (Section 4.3).
4. The cost layer of the circuit cannot move a measurement probability;
   everything passes through a mixer bounded at 0.393 rad, and the mixer
   alone accounts for about half of what a perfect optimizer could move
   (Section 4.4).
5. At `dim = 2`, the size of the deployed pipeline and of the originating
   report, the exact ground state is "refine everything" whatever the
   Hamiltonian; every comparison at that size is vacuous (Section 4.5).
6. H0a: QAOA reaches the optimum of its own Hamiltonian on 0 to 17% of
   instances at `dim = 3` (at most 25% once the bias is recentred; never on
   the trained mapper) (Sections 5.1-5.2).
7. H0b: solving the Hamiltonian better makes the refinement decision worse.
   The exact optimum never decides better than the classical rule at the
   same threshold, recentring the bias on the F1-optimal threshold does not
   change this, and the couplings drive the ground state toward a uniform
   mask; the QAOA decision measured by the study code is essentially a
   fixed threshold at 0.5 on the classical score (Section 5.2).
8. H3: the coupling terms never help the exact decision and cost F1 once
   they are active; neighbour features help a learned model only in a
   setting whose average cannot be cited (Section 5.3).
9. H2b and confirmatory replication: across 4 scenarios × 4 Reynolds
   numbers the classical threshold beats or ties QAOA and a trained GBT on
   every fold (Section 5.4).
10. Synthetic generators: classical and learned ceilings are stable; the
    synthetic H0a/H0b comparisons are affected by a mapper mismatch found
    during this review (Section 5.5).
11. The temporal phase does not improve the decision (Section 5.6).
12. H5: on three of four canonical scenarios the static label is almost a
    deterministic function of the classical score; a dynamic label at the
    physical crossing time differs from it on two of four (Section 5.7).
13. The closed-loop study inside the adaptive solver, and why its numbers
    are historical (Sections 6.1-6.2).
14. Transfer to unseen conditions (H4) and numerical defects (H1)
    (Sections 6.3-6.4).
15. Methodological findings of the audit (Section 7).

## 2. Methods

### 2.1 MHD solver, scenarios and reference simulations

The solver integrates incompressible visco-resistive MHD on a periodic
domain [0, 2π]² with fourth-order centred finite differences in space and
fourth-order Runge-Kutta (RK4) in time, the time step adapted to a CFL
target of 0.4. The velocity is made divergence-free by a spectral projection
applied after each RK4 step; the magnetic field is not projected, because
its curl-form induction equation keeps the finite-difference divergence of
B at round-off by construction (Section 4.1). The scheme converges at first
order in time (Section 4.1); the limit is common to every arm compared here.

Four canonical scenarios are used in most analyses: a Harris current sheet
(`harris_tearing`), a Kelvin-Helmholtz shear layer (`kelvin_helmholtz`), an
MHD rotor (`mhd_rotor`) and the Orszag-Tang vortex (`orszag_tang`). Four
more belong to the training set and to the frozen protocol of the next
campaign: a Lamb-Oseen vortex, island coalescence, double tearing and a
magnetic twist. Initial perturbations are written as curls of flux
functions with the solver's own stencil, so that div B is at round-off from
the first step. Reynolds numbers Re = Rm ∈ {400, 800, 1200, 1600};
resolutions N ∈ {64, 96, 256}. Direct numerical simulations (DNS) on the
fine grid are the ground truth. They pass hard gates before use: finite
fields, no divergence of the solver, relative finite-difference divergence
of B ≤ 10⁻³, energy non-increasing to 10⁻³.

The adaptive solver refines patches by a factor 2 per level, up to
`max_depth = 4` in the deployed configuration, with a tau-correction
between levels. A run's cost is `patch_ratio`, the refined pixels divided
by (steps × N²). Its fidelity is `phys_score`, an instability-weighted
relative L2 error against the DNS, with weight
w = 1 + 0.25 (|J_z|/⟨|J_z|⟩ + |ω|/⟨|ω|⟩) computed on the reference fields
and identical for every arm.

### 2.2 The refinement label

The domain is partitioned into `dim × dim` patches. The static label of a
patch is the L2 deviation of the fine field from that patch's own mean; the
top 25% of patches (percentile over all snapshots of a trajectory) are
"hard". With a patch side of p = N/dim cells, the label is estimated on p²
points and is identically zero at p = 1, so the analyses respect
`dim ≤ N/8` where possible; four artifacts at N = 64, `dim = 64` have an
all-zero label and are used nowhere.

A dynamic label d_i is also measured (Section 5.7): the whole-field L2
difference after replacing patch i alone by its mean and evolving both
fields over a horizon δt with the same, frozen sequence of time steps
(otherwise 100% of patches would adapt their own step sequence on three of
four scenarios, and the label would count a time-step difference as
physics). The default horizon is the physical crossing time of a patch,
t_x = 2π / (dim · (v + b)_rms).

A refinement criterion is scored by F1 against the label.

### 2.3 The physical-to-Ising mapping: two mappers

Each patch carries two qubits, one per edge family. Each qubit starts in
cos(θ/2)|0⟩ + exp(iψ) sin(θ/2)|1⟩ with θ = 2 arcsin(√s), where s is the
classical score of the cell, so that P(|1⟩) = s before any QAOA layer; ψ is
the temporal phase built from the change of the local stress flux Φ between
two snapshots (Appendix A.8, where it is written φ). The cost Hamiltonian
has a Z bias h, ZZ couplings C between neighbouring edges, and ZZZZ
plaquette terms K and K_xpoint. Two implementations exist.

**V1** (`src/Simulation/HamiltParams.py`, `PhysicalMapper`) is the mapper
of the deployed AMR pipeline and the one a hyperparameter campaign would
tune. Each coefficient is a product weight × g(topology) × f(scale) ×
T_rcf(signal) (Appendix A); ZZ carries a Gaussian uncertainty window around
the AMR threshold; mesh thresholds are relative, min(absolute,
percentile). It has nine tunable hyperparameters — `beta`, `w_z_frac`,
`sigma`, `beta_curl`, `beta_xpoint`, `gamma_hydro`, `gamma_mag`, `kappa`,
`relative_percentile` — plus `threshold_amr`, fixed at 0.1496, the best
classical trial (verified with `python src/train_hyperparams.py
--print-space`). Because the deployed hyperparameter file is incomplete and
has no reproducible provenance (Section 8), every V1 measurement made by
`study/` uses the reference values of `study/pipeline/config.py`
(threshold 0.1496, σ = 0.023); the closed loop (Section 6) tuned its own
values per fold.

**V2** (`src/Simulation/HamiltParams_v2.py`, `PhysicalMapperV2`) has no
free parameter apart from the AMR threshold it receives. In its current
form (normalisation `max`, default since 21 August; joint plaquette
normalisation since 22 August, Appendix B), with ω̂ = |ω|/max|ω|,
Ĵ = |J|/max|J| and X̂ = max(0, −det ∇B)/max(...), each set to zero below a
round-off floor:

```
C_ij       = -2 * |jump_ij| / max|jump|,   jump = sqrt(dvx^2 + dvy^2 + dBx^2 + dBy^2)
K_p        = -1 * (w^ + J^) / max(w^ + J^ + X^)
K_xpoint,p = -1 * X^       / max(w^ + J^ + X^)
h_i        = 0.1 * max(|C|, |K|) * (s_i - threshold)
```

V2 is dimensionless: changing dx from 1.0 to 0.001 leaves C, K and h
bit-for-bit identical, and ν and η do not enter it, so its coefficients
cannot distinguish a viscous from an inertial flow (D-12).

The two mappers are not interchangeable, and the published artifacts were
produced over several weeks while the mappers were being corrected. Which
result uses which version:

| result | mapper | version | artifact date | section |
|---|---|---|---|---|
| H0a/H0b panel, 32 instances | V2 | legacy normalisation, before the curl-convention fix | 9 Aug | 5.1-5.2 |
| H0a/H0b panel, corrected, 12 instances | V2 | legacy normalisation, after the curl fix and study/circuit alignment | 16 Aug | 5.1-5.2 |
| H0a/H0b panel, 12 instances | V1 | current (replayed bit for bit on 5 Oct) | 28 Aug | 5.1-5.2 |
| recentring plan, 8 panels of 12 instances | V1 and V2 | current | 5 Oct | 5.2 |
| coupling ablation at `dim = 3` | V1 and V2 | current | 29 Aug | 5.3 |
| size scan `dim = 2, 4, 8` | V1 | early (before the curl, threshold and gate fixes) | 7 Aug | 5.3 |
| confirmatory LOSO, 4 Re | V2 | current | 11 Sep | 5.4 |
| synthetic QAOA vs exact | QAOA on V1, exact on V2 | current (mismatch, Section 5.5) | 8-10 Sep | 5.5 |
| ψ feature, LOSO | V1 | current | 27 Aug | 5.6 |
| closed loop | V1 | early | 2-7 Aug | 6 |

### 2.4 QAOA and the exact references

QAOA runs on the Qiskit Aer simulator, noiseless, statevector backend, with
`reps` from 1 to 6, 4096 shots (256 in the closed loop) and COBYLA with a
budget `K_opt = 60` evaluations in the study panels. The mixer angle is
bounded at π/(4·reps). No path to quantum hardware exists in the code: a
`mode="hardware"` option used to run silently on the simulator and now
refuses (D-48). Since D-191 the QAOA seed is drawn at random unless one is
passed; every study panel passes `--seed 0`, and the frozen confirmatory
protocol fixes the QAOA seed while physical seeds vary.

**How a decision is read from QAOA.** The circuit returns a distribution
over bit strings; no code path uses the single most probable string. The
study code (`study/common/qaoa_inputs.py`, `run_qaoa_on_snapshot`) reads
each edge qubit by majority — refine when its marginal probability P(1)
exceeds 0.5 — and refines a cell when either of its two edge qubits does;
the exact optimum and the other solvers of Section 5 are read from their
spins in the same way. The deployed adaptive solver
(`src/Simulation/refinement.py`) instead averages the two marginals of a
cell and refines when the average reaches the AMR threshold. The two rules
coincide only when that threshold is 0.5. Because every qubit starts with
P(1) = s, the deployed rule applied to the unoptimised circuit returns the
classical decision s ≥ threshold, while the study rule returns s > 0.5, a
different decision (Section 5.2).

References against which QAOA is compared on the same Hamiltonian:

- exhaustive enumeration of all 2^n states, available up to 22 qubits
  (`dim = 2`: 8 qubits; `dim = 3`: 18 qubits). At `dim = 3` the ground state
  was unique on every one of 160 snapshots of the confirmatory run;
- a greedy descent warm-started from the classical decision, used where
  enumeration is impossible (`dim ≥ 4`), validated against enumeration at
  `dim = 2` (the two agree on 75% of cells and are both insensitive to the
  ablations);
- simulated annealing from a random (cold) or classical (warm) start, and
  the classical decision itself, scored as a solver.

The study code and the circuit build the same operator: the largest energy
difference over the 256 states of a `dim = 2` problem is 3.6 × 10⁻¹⁵
(preflight, run during this review).

### 2.5 Baselines

The classical rule is a single threshold on the classical multi-indicator
score s (vorticity, velocity divergence, current density, and a Löhner
error estimator on |B|), fit by maximising F1 on the training scenarios
only and applied unchanged to the held-out one. The learned baseline is a
histogram gradient-boosted tree (GBT) on nine local features (s, |v|²,
|B|², |ω|, |J|, |∇v|², |∇B|², det ∇B, Re), optionally with neighbour
(stencil or k-hop cone) features, trained with early stopping; logistic
regression and random forests are added in the synthetic studies.

### 2.6 The closed-loop protocol (Level 3)

Each fold holds one instability class out of all tuning (four folds: `kh`,
`ot`, `rotor`, `tearing`). The Q-HAS arm's hyperparameters are tuned by
Optuna on the other classes (4 trials per fold, against 170 in the
protocol); the classical arm's threshold is tuned on the same classes. Both
arms then run on the held-out class with the same DNS trace, hot start,
hybrid budget and refinement depth; only the decision routine differs.
Endpoints: `phys_score`, `patch_ratio`, and the pre-registered composite
(phys + λ·patch)/(1 + λ) with λ = 0.4. Because the arms end at different
costs, the classical error is also measured along a bisected
threshold-to-cost frontier and compared at the cost Q-HAS actually spent
("budget-matched"). The Q-HAS arm is non-deterministic, so it is repeated
five times per fold, and each run's completion status is recorded at
execution time. Section 6.1 explains why the numbers of this study are
historical.

### 2.7 Statistics

Confidence intervals on paired differences between two decision rules are
snapshot-level percentile bootstraps: 1000 resamples of the held-out
snapshots (cells within a snapshot are not independent), 95% intervals,
reported with the one-sided bootstrap probability. A verdict that one rule
beats another on a fold requires the interval to exclude zero. Rank
correlations are Spearman. In the closed loop, conclusions are stated as
dominance counts over repeated draws and as verdicts over a sweep of λ,
not as single ratios. Where an analysis resamples whole trajectories or
scenarios rather than snapshots, it says so.

### 2.8 Evidence levels and provenance

Every artifact records the commit it was produced from, its command-line
arguments and, for recent artifacts, whether the working tree was clean.
Following the project's evaluation rules (`docs/EVALUATION.md`), each
result below carries one of four levels:

- **A** — reproducible on the current code and pinned by a test that can
  fail;
- **B** — correctly obtained, but on code that has changed since; to be
  re-measured;
- **C** — not conclusive: the variance of the measurement is comparable to
  the effect;
- **D** — obsolete: obtained on code now known to have been wrong for that
  quantity; reported as history only.

The aggregated master table (`study/common/aggregate_master_table.py`)
recomputes 268 published values from their artifacts: 139 match, 6 differ
(each explained in `docs/RESULTS.md`), 123 are missing, all of them rows of
the confirmatory campaign that was not run (recomputed during this review;
identical to the committed table). Some artifacts were produced from a
working tree with uncommitted changes; their own provenance record says so,
and they are flagged where cited.

## 3. Hypotheses and verdicts

| label | question | verdict | section |
|---|---|---|---|
| H0a | Does QAOA reach the exact ground state of its own Hamiltonian? | **No** — 0 to 17% of instances at `dim = 3` (at most 25% with the bias recentred), read by majority | 5.1-5.2 |
| H0b | If a solver reaches it, is the refinement decision better? | **No** — the exact optimum never decides better than the classical rule at the same threshold; ρ(E_gap, F1) > 0 on the trained mapper and wherever the bias is recentred | 5.2 |
| H1 | Are solver and numerical defects, on their own, sufficient to explain a failure? | **Partial** — they matter; nothing isolates them as sufficient | 6.4 |
| H2b | Does a more flexible model beat the classical threshold? | **No** — no model tested beats it under leave-one-scenario-out | 5.4-5.5 |
| H3 | Do the ZZ/ZZZZ couplings improve the decision over the bias alone? | **No** — never for the exact decision; they lower F1 once active | 5.3 |
| H4 | Does the approach transfer to physical conditions outside its training? | **Conjecture** — no dedicated experiment isolates it | 6.3 |
| H5 | Is a failure explained by how the label is specified? | **Mixed** — the static label is almost a function of the classical score on 3 of 4 scenarios; a dynamic label differs on 2 of 4 | 5.7 |

H0a and H0b together test whether the optimization step is the problem; H3
tests whether the representation (the couplings) is the problem,
independently of how well it is solved; H2b tests whether restricting the
decision to a physically motivated Ising form, rather than a flexible
learned model, is the bottleneck. H1, H4 and H5 are auxiliary axes.

## 4. Results I — the physics and the instrument

### 4.1 The ground truth: solver and scenario validation

**Spatial operators.** The finite-difference gradient and Laplacian are
fourth order to the digit on a smooth periodic field (observed order 4.00
at every refinement). [A]

**Order of the scheme.** The scheme as a whole converges at first order in
time, not fourth. Self-convergence on grids 32 → 64 → 128 (t = 0.5) and 64 →
128 → 256 (t = 0.25) gives an observed order of 1.00 (relative differences
3.34 × 10⁻² and 1.67 × 10⁻² on the finer pair). At a fixed grid, refining
only the time step, the error falls at order 1.04 to 1.22 (mean 1.12) with
the divergence-free projection, and at order 4.00 without it. The cause is
the splitting: a full RK4 step is taken, then the state is projected — a
first-order (Lie) splitting of a differential-algebraic system. Projecting
the right-hand side at every RK4 stage instead restores order 4.00 (N = 96,
Orszag-Tang, T = 0.5: error 2.1 × 10⁻¹¹ instead of 1.1 × 10⁻³ at 256 steps,
52 000 times smaller, with the same divergence control), but it cannot be
applied to the adaptive path, whose levels advance a downsampled global
field and non-periodic local patches; it is therefore disabled (`PROJECT_RHS
= False`). A Strang splitting does not apply: the projection is idempotent,
and P∘RK4∘P gives errors identical to P∘RK4 to the last digit. An
incremental-pressure (Van Kan) correction leaves the order at 1.10 (C — not
conclusive; a mismatch between the spectral projection and the
finite-difference right-hand side is suspected, not verified). The order
drop is common to both arms of every comparison in this report. [A] (script:
`study/h1_solver/h1_solver_convergence.py`; tests:
`tests/solver/test_solver_convergence.py`; artifact 11 Aug.)

**Constraint and conservation.** On every reference trajectory
max|div B|/rms|B| lies between 5.6 × 10⁻¹⁵ and 8.0 × 10⁻¹⁴ and the energy
decreases monotonically (drop 0.3-1.8%); Re = 200 and Re = 3200, outside
the grid of the study, pass the same checks. The magnetic field is not
projected: applying the spectral projection to a field that the
finite-difference induction already keeps solenoidal raised its
finite-difference divergence from 1.0 × 10⁻¹⁴ to 4.6 × 10⁻⁷ after 50 steps,
for an error identical to the fourth decimal (D-25). The same pass showed
that a discrete quantity must be measured with the operator that produced
it: measured with a spectral divergence, the defect was invisible
(9.5 × 10⁻²). [A]

**Initial conditions.** Four scenarios set non-solenoidal initial
perturbations that the projection then partly removed: the Harris current
sheet kept only 27.5% of the perturbation that seeds its tearing mode
(island coalescence 27.5%, double tearing 77.3%, a noisy-uniform test case
55.7%). Rewritten as curls of flux functions differentiated with the
solver's own fourth-order stencil, they are solenoidal to ~10⁻¹⁶ with their
nominal amplitude intact (D-27). A first rewrite with analytic derivatives
reached only 2.1 × 10⁻⁵, the same mismatched-operator trap. Two initialisers
did not set the field they claimed: the magnetic twist set no twist
(6.4 × 10⁻⁷ instead of π/2, D-6) and a "ghost twisting" scenario set an
impossible field (angle 0.027 instead of 1.906 rad, D-26; the scenario has
since been removed). [A] (tests: `tests/solver/test_scenarios_analytic.py`.)

**Reconnection is present.** On all six Harris current-sheet trajectories
(Re 400 to 1600, N 64 to 256), the mean-square current, once the
equilibrium current of the sheet is removed, grows by 8.1× (Re 400, N 64)
to 17.5× (Re 1600, N 96) and is still rising at the end of the simulated
window (D-39). Before the equilibrium current was removed, the diagnostic
saw a growth of 1.00-1.10× and rejected all six. [A] (test:
`tests/study/test_check_tearing_end_pinned_peak.py`.)

**Symmetry.** A rotation by 180° commutes with a time step to 2.8 × 10⁻¹⁶;
reflections and a 90° rotation to 7.8 × 10⁻⁶ (N = 64). The classical
decision map is nearly equivariant under these operations (orbit error
0.044 at N = 64 and 0.015 at N = 256 at `dim = 8`, attributed to one-sided
differences in the indicator). The equivariance of a ground-state decision
cannot be judged at `dim = 8`: two annealing seeds already disagree on 27%
(N = 64) to 36% (N = 256) of patches on the same, untransformed field, a
floor as large as the orbit error being measured. [B] (T12, artifacts 2 Jul
and 11 Aug.)

**Adaptive numerics.** Three defects of the adaptive path were fixed, two
common to both arms and one specific to Q-HAS: the global prolongation
sampled cell centres on a node-based grid with a non-periodic boundary mode
(maximum error 2.49 × 10⁻¹ on sin x cos y, 7.74 × 10⁻⁶ after the fix; the
error survived only on the unrefined background, about 1.7% of the field,
D-2); the patch list could cover the same region twice (up to 25% of the
domain at the deployed threshold, up to three patches on one cell, D-16);
the bias of a patch was read from the wrong quarter of the patch at every
level below the root (41% of the largest coefficient, D-37 — this one
affected only the Q-HAS arm, Section 6.1). After the fixes the patch list
tiles the 256² domain exactly, without gap or overlap, the adaptive step
equals the global step bit for bit at depth 0, and the CFL guard has a
margin of 2.5×. [A] (tests: `tests/amr/test_amr_resampling_analytic.py`,
`tests/amr/test_amr_tiling_contracts.py`,
`tests/amr/test_patch_encoding_shapes.py`.)

### 4.2 What the sensors detect

The mapping is the instrument that is supposed to see topological structure.
It was checked against fields whose answer is known analytically and
against real DNS. Before the corrections below, it did not see what it
claimed to see. Each correction was measured before and after, and each is
pinned by a test that fails on the old code. Together they are the main
technical advance of the project, independent of the verdict on QAOA. [A]

**(a) The vorticity and divergence indicators measured strain (D-1, T31).**
Under the grid's declared convention (axis 0 = x), the three mappers formed
their "vorticity" as ∂y v_y − ∂x v_x and their "divergence" as
∂y v_x + ∂x v_y — two components of the strain-rate tensor, not a curl and
a divergence. On linear fields:

| field | true ω | mapper ω (before) | true div v | mapper div v (before) |
|---|---|---|---|---|
| solid rotation v = (−y, x) | +2 | **0** | 0 | 0 |
| pure shear v = (y, 0) | −1 | **0** | 0 | **+1** |
| pure expansion v = (x, y) | 0 | 0 | +2 | **0** |
| pure strain v = (x, −y) | 0 | **−2** | 0 | 0 |

The Okubo-Weiss criterion, on the correct axes, weighted the strain by one
half and kept its isotropic part: a pure shear read +0.25
("rotation-dominated") instead of 0, a pure expansion −1 instead of 0. The
same convention survived in three places outside the solver: two of the nine
GBT features (|ω| and |J|), the enstrophy of a figure (0 instead of 2π² on
vx = sin y) and a current computed in a study script. All are corrected.
Correcting the convention at unchanged hyperparameters does not improve the
ranking of hard patches: at matched budget, on 4 scenarios × 6 snapshots at
N = 128, the change in Spearman correlation and in F1 is −0.027 and −0.031
at `dim = 8`, −0.053 and −0.060 at `dim = 16`, all four negative and none
significant (95% intervals by scenario all include zero). The publishable
statement is "no measurable gain, and nothing that excludes a degradation";
the hyperparameters had been tuned on the wrong operator. (script:
`study/h1_solver/h1_curl_convention_gap.py`; tests:
`tests/solver/test_analytic_fields.py`,
`tests/study/test_curl_convention_gap.py`,
`tests/study/test_fixed_curl_variant.py`,
`tests/study/test_no_private_curl_survives.py`.)

**(b) The circulation sensor was blind to vortices.** Because of (a), the
plaquette term, whose only purpose is to detect circulation, returned
exactly zero on a solid rotation. On a Lamb-Oseen vortex its coefficient
rises from 0.055 to 1.255 (×22.7, deterministic) once the curl is correct.
Whether this changes QAOA's decision on a vortex is not established: two
estimates of the decision contrast, +0.0186 ± 0.0067 (16 draws) and
+0.0053 ± 0.0029 (8 draws), differ by a factor 3.5. [coefficient: A;
decision contrast: C]

**(c) The X-point detector was zero at the X-point.** The X-point term was
multiplied by a magnetic gate f(|B|·dx/η), and an X-point is by definition a
zero of B: on a two-cell current sheet at N = 256 the detector read 0.0000
at the null (six sheet thicknesses tested) and 0.854 on the surrounding
ring. With the gate removed, its maximum falls exactly on the null. Its
normalisation also compared a squared field gradient to a squared threshold
with one power of dx² missing, so the signal-to-threshold ratio scaled as
dx⁴ and fell 244× from N = 64 to N = 256; rewritten with √(−det ∇B), which
has the units of a current, it scales as dx² like the current channel
(0.414, 0.103, 0.026 at N = 64, 128, 256) and fires on sheets four cells
thick at N = 128. (tests:
`tests/mapping/test_xpoint_at_training_resolution.py`,
`tests/mapping/test_coefficient_families_contract.py`.)

**(d) The four-body terms vanished at the training resolution.** The mesh
thresholds of V1 were absolute and scale as 1/dx²: at N = 256 no cell
crossed them, and both four-body terms were exactly zero on all four
canonical scenarios, although the raw signal was strongly contrasted
(maximum over median: 1104 for √(−det ∇B) on the Harris sheet, 223 on island
coalescence, 752 on the rotor). With the relative criterion min(absolute
threshold, 90th percentile of the signal), the four-body terms exist on all
four scenarios, and on the two reconnection scenarios the X-point channel
dominates the current channel (×370 on the Harris sheet and ×36 on island
coalescence before correction (e); ×4 on the Harris sheet after it), as the
physics predicts. A uniform field still gives zero: the criterion does not
invent structure. The percentile became the ninth tunable parameter (at the
bounds 50, 90 and 99 it leaves 2040, 404 and 32 of 4096 plaquettes
non-zero).

**(e) The magnetic gate compared two unit systems.** The rotation gate
compared the Okubo-Weiss Q, in physical units, to its threshold; the
magnetic gate compared a current computed without dx, in grid units, to a
physical threshold, so it was harder to open by a factor 1/dx (10.2, 20.4
and 40.7 at N = 64, 128, 256). Corrected, the plaquette coefficient on a
current sheet goes from 1.8 × 10⁻⁵ to 1.148, and the ratio between the
fluid and magnetic channels from 27 500 to 0.44 (magnetic/fluid 2.29).

**(f) Selectivity: only the X-point term recognises a type of structure.**
On analytic fields at `dim = 32` (V2): the X-point term answers only where
det ∇B < 0, is exactly zero where det ∇B > 0 (elliptic points, island
cores) and survives a ×10 rescaling of the field. The plaquette term, which
sums |ω| and |J|, returns the same value (1.000) on a pure vortex and on a
pure current sheet: it says "something rotational or magnetic happens
here", not which. The ZZ coupling puts hydrodynamic and magnetic jumps under
one square root. In V1 the gates of the two coupling families satisfy
g_strain + g_rot = 1 identically, so ZZ and ZZZZ partition a single
Okubo-Weiss scalar rather than acting as two independent detectors, and the
Reynolds gate and the hydrodynamic contrast are monotone re-parametrisations
of the same scalar (equal to 10⁻¹²). These are properties of the form of
the coefficients, measured on fields with a known answer; they say nothing
by themselves about the quality of a refinement decision. (tests:
`tests/mapping/test_selectivite_des_coefficients.py`,
`tests/mapping/test_gate_contracts.py`.)

**(g) Normalisation of V2.** Under the original common denominator
(|ω| + |J|)/(max|ω| + max|J|), the weaker of the two structures vanished
from the plaquette: over 12 snapshots at N = 256, the median weight of
vorticity was 0.003 on the Harris sheet (where max|J|/max|ω| = 179) and
0.993 on Kelvin-Helmholtz (current absent; ratio 84). Each magnitude is now
normalised by its own maximum before the sum. Under the legacy
normalisation the weight of the ZZ family relative to the ZZZZ family
drifted by ×2.59 with the spikiness of the snapshot (3.12 to 8.10); under
`max` it equals the design ratio 2.000; with the X-point term enabled the
ratio drifted again (1.03 to 2.00 on the four scenarios) until vorticity,
current and X-point were normalised together (D-190: 2.000 on all four; the
X-point now carries 30-45% of the plaquette family on real fields). (tests:
`tests/mapping/test_normalisation_max_invariante.py`,
`tests/mapping/test_plaquette_signal_negligeable.py`.)

**(h) Calm fields build no Hamiltonian.** On a uniform field every
coefficient falls below the pruning threshold. The circuit used to receive
a fabricated single-qubit term (10⁻³ on qubit 0) that biased it to refine
one edge; it now raises an error and the patch keeps the classical
decision. Today (preflight, run during this review): a vortex lattice gives
a plaquette coefficient of 0.501 and an X-point coefficient of 0; a uniform
field gives 0 on every family; at N = 256 on the Harris sheet the
plaquette and X-point terms are alive (0.243 and 0.981). (script:
`study/common/preflight_coefficients.py`.)

### 4.3 Do the sensors point where refinement is needed?

The one check that bears directly on the purpose of the model: the same
scenario is run at N = 32 (coarse) and N = 128 (reference) for the same
number of steps; the relative error of the coarse solution is computed on
8 × 8 blocks; and the block-mean coefficient is correlated (Spearman) with
the block error.

- **Current measurement** (preflight, this review; Harris sheet; V1 at the
  reference hyperparameters): plaquette coefficient ρ = 0.798; classical
  score ρ = 0.814; the best of 256 random points of the nine-parameter
  hyperparameter space ρ = 0.843, 0.029 above the classical score. [A]
- **Earlier measurement on four scenarios**, before the magnetic gate was
  corrected (4.2e) [B]: plaquette ρ = 0.897 (Harris sheet), 0.877 (island
  coalescence), 0.755 (rotor), 0.249 (Orszag-Tang), against 0.814, 0.912,
  0.528 and 0.422 for the classical score. On the rotor the plaquette
  coefficient beat the classical score (0.755 against 0.528). Only the
  Harris value has been re-measured since the correction (0.897 → 0.798);
  the rotor comparison has not.

Reading: the sensors carry spatial information about where the coarse grid
fails, at a level comparable to the classical indicator. The rest of the
report asks whether turning that information into an Ising ground state,
solved by QAOA, produces a better refinement decision. (test:
`tests/mapping/test_coefficient_families_contract.py`, which requires
ρ > 0.6 on the Harris sheet.)

### 4.4 What the circuit can and cannot move

- **The cost layer is diagonal.** exp(−iγH) only adds phases. Sweeping γ
  from 0 to 2π with the mixer off moves no measurement probability (largest
  change 4.4 × 10⁻¹⁶). Only the mixer exp(−iβ ΣX) moves P(|1⟩), and its
  angle is bounded at π/(4·reps) = 0.393 rad for `reps = 2`. [A]
- **What a perfect optimizer could move.** Sweeping the whole admissible
  (β, γ) grid on five Orszag-Tang patches (V2, N = 256): the median largest
  displacement of a marginal is 0.254 with the mixer alone and 0.490 with
  the Hamiltonian, so the Hamiltonian adds a median 0.236. The physics is
  not inert, but it acts through a bounded channel, and about half of what
  the circuit can move is a mixer rotation independent of any physics. The
  correct control for the Hamiltonian's contribution is therefore "mixer
  alone", not the classical score. It is not part of the solver panel of
  Section 5, because COBYLA has nothing to optimise on a null cost and the
  code refuses to build a null Hamiltonian. [A]
- **The initial state encodes the classical score.** P(|1⟩) = sin²(θ/2) = s
  to 2.4 × 10⁻¹⁵; ψ moves phases and never probabilities; with all
  parameters at zero the circuit returns the θ initialisation exactly. Read
  with the deployed rule, this initial state is the classical decision;
  read with the study's majority rule, it is the decision s > 0.5 (Section
  2.4). The full chain
  score (i, j) → qubit → Pauli term → Qiskit bit string → marginal is
  pinned, because a single reversed convention would mirror the decision
  map spatially while passing every test of values. [A] (test:
  `tests/quantum/test_vqa_chain_contracts.py`.)
- **In V1 the scale of the bias depends on the threshold** through the
  median of the windowed couplings: on a shear layer whose score takes two
  values, it moves by ×173, non-monotonically, with the threshold alone. [B]
- **The QAOA arm is sampled.** Ten identical calls on a rotor case (Re 800,
  N = 64, 3 × 3 blocks) give a per-call range of block scores from 0.18 to
  0.36 of the [0, 1] scale, but a median rank autocorrelation of 0.933
  between calls (minimum 0.350). Conclusions based on a ranking are robust;
  conclusions that would rest on one value are not. [C for values, A for
  ranks] (test: `tests/quantum/test_qaoa_arm_is_sampled.py`.)

### 4.5 Why `dim = 2` decides nothing

At `dim = 2` (8 qubits) — the size the deployed AMR pipeline solves, and
the size at which the originating report measured everything — the exact
ground state of V1 is "refine everything" on 40 of 40 snapshots, identical
to the classical baseline on 40 of 40, with an equal F1 on 40 of 40 and
never a higher one (D-47). At the deployed settings, (s − threshold)/σ ≥ 8.4
on every patch, so the ZZ window is at most 1.15 × 10⁻³¹ and no ZZ term
survives the pruning at 10⁻⁶; the Z bias is positive everywhere and 2.0 to
6.6 times the largest plaquette coefficient. V2 under the current
normalisation is uniform on 36 of 40 (legacy normalisation: 39 of 40). [A]
(test: `tests/study/test_phase5_ne_filtre_plus_sur_promising.py`; slow test
pinning both rates.)

Consequences, all measured at that size [B, artifacts 2 Jul to 11 Aug]:

- every solver of the panel reaches the optimum, including the classical
  decision alone and every QAOA depth (N = 64 and N = 256); the only one
  that sometimes misses it, cold-start simulated annealing (42% of
  snapshots at N = 256), is the only one that does not start from the
  classical decision;
- removing every ZZ and every ZZZZ term changes no decision, for both
  mappers; V1's cost function has on average 64.8 of its 256 configurations
  tied at the optimum;
- the explanation first written for this inertness — an uncertainty window
  annihilating the ZZ coupling — was retracted: measured on the score the
  deployed path actually uses, the window keeps 3.3-12.1% of the ZZ mass at
  the open-loop setting (σ = 0.023) and 33.8-59.4% at the closed-loop
  setting (σ = 0.189); with the window neutralised (couplings of 25 to 155)
  the couplings still change no decision, and the window changes 25% of
  decisions only through the normalisation of the bias, never as a
  coupling (T17, T18);
- ρ(E_gap, F1) = −1.000 across the solver panel (N = 256): a perfect but
  meaningless correlation, because the classical rule already is the
  optimum, every solver that reaches it returns the same mask, and all F1
  values lie between 0.367 and 0.389; there is nothing to rank.

The correct explanation of these null results is the degeneracy of
`dim = 2`, not the Ising formalism and not an implementation defect. It
also accounts for the originating report's small composite-loss difference:
with a uniform optimum and inert couplings, the deployed QAOA decision is a
small perturbation of the classical amplitude encoding (the audit counted
109 decisions flipped relative to the classical rule in that report's
runs, 45 correct and 64 incorrect [D, historical]). This is why `dim = 3`,
the smallest size above this degenerate point at which the exact optimum
can still be enumerated, is the reference size for the rest of the report.

## 5. Results II — the decision

All results in this section are at `dim = 3` (18 qubits) unless stated,
the smallest size above the degenerate point (Section 4.5) at which the
exact optimum can still be enumerated, on the four canonical scenarios at
Re = 400 and N = 96 unless stated.

### 5.1 H0a: QAOA rarely reaches the optimum of its own Hamiltonian

**Table 1. Fraction of instances on which each solver reaches the exact
ground state of the same Hamiltonian, three independent panels.**

| solver | V2, 9 Aug (32 instances) | V2 corrected, 16 Aug (12) | V1, 28 Aug (12) |
|---|---|---|---|
| exhaustive enumeration | 1.000 | 1.000 | 1.000 |
| greedy, warm start | 0.844 | 0.833 | 0.750 |
| simulated annealing, cold | 0.594 | 0.500 | 1.000 |
| simulated annealing, warm | 0.750 | 0.833 | 1.000 |
| classical decision alone | 0.500 | 0.500 | 0.500 |
| QAOA p = 1 | 0.156 | 0.083 | 0.000 |
| QAOA p = 2 | 0.156 | 0.083 | 0.000 |
| QAOA p = 3 | 0.125 | 0.083 | 0.000 |
| QAOA p = 4 / 5 / 6 | 0.094 / 0.062 / 0.062 | — | — |
| QAOA, more shots (p = 6 / 3 / 3) | 0.062 | 0.083 | 0.000 |

QAOA reaches the optimum of its own Hamiltonian less often than the
classical decision alone (0.500), at every depth, on both mappers. These
rates use the study's majority readout, the one that asks whether the
circuit's state concentrates on the ground state; read with the deployed
rule, which starts from the classical decision, QAOA reaches the optimum as
often as that decision does on V1 (Table 3).
A correct claim of optimality would require 1.000. The drop from 0.156 to
0.083 between the first two panels is within QAOA's run-to-run dispersion;
the ranking of solvers is the same. The corrected panel differs from the
first by four alignments between the study code and the circuit: the
X-point term included, the anomaly flag enabled, the magnetic gate in
physical units, and the pruning threshold raised from 10⁻¹² to 10⁻⁶ (which
had kept 25% more terms than the circuit on the Harris sheet). [V1 panel:
A, replayed bit for bit on the current code during this review and pinned
by `tests/study/test_h0_recentring_summary.py`. V2 panels: B, measured
under the legacy V2 normalisation; on the current V2, QAOA reaches the
optimum on 0 to 17% of the same twelve instances (Table 3).] (script:
`study/h0_selection/h0_optimiser_equivalence.py`; test:
`tests/study/test_h0_certified_dim3_contradicts_criterion.py`.)

**It is not a budget problem in the usual sense.** On the Harris sheet
(6 instances, artifact 9 Aug), with the optimizer budget scaled with the
depth, QAOA reaches the optimum on 0 of 6 at depths 1, 3 and 6, while the
greedy descent reaches it on 6 of 6. [A] In another configuration a larger
budget does repair a symptom: on a rotor case (V1, 3 × 3 blocks), at the
deployed budget `K_opt = 80` QAOA gives the cell with the largest true
error the lowest score of the nine; at `K_opt = 300` still the lowest; at
`K_opt = 800` the highest, and the captured fraction of the true error goes
from 0.177 to 0.603, with the Hamiltonian unchanged (D-195). Both are
instances of the same fact: at the depths and budgets used, the variational
optimizer does not converge to the ground state of the Hamiltonian it is
given. [Harris panel: A. Rotor measurement: reproducible by the command
given under D-195 in `docs/RESULTS.md`, not pinned by a test.]

At larger scale (Section 5.4) the agreement between QAOA's decision and the
exact optimum, per held-out scenario, is 0.667, 0.994, 0.106 and 0.622.

### 5.2 H0b: reaching the optimum makes the decision worse

**Table 2. F1 against the DNS label, same three panels, and the rank
correlation between a solver's energy gap to the optimum and its F1.**

| solver | V2, 9 Aug | V2 corrected, 16 Aug | V1, 28 Aug |
|---|---|---|---|
| exhaustive (E_gap = 0) | 0.391 | 0.386 | 0.437 |
| simulated annealing, cold | 0.392 | 0.393 | 0.437 |
| simulated annealing, warm | 0.381 | 0.378 | 0.437 |
| greedy, warm start | 0.432 | 0.424 | 0.420 |
| classical decision alone | 0.471 | 0.460 | 0.468 |
| QAOA p = 1 | 0.491 | 0.481 | 0.519 |
| QAOA p = 2 | 0.513 | 0.470 | 0.519 |
| QAOA p = 3 | 0.515 | 0.470 | 0.503 |
| QAOA p = 4 / 5 / 6 | 0.509 / 0.507 / 0.526 | — | — |
| QAOA, more shots | 0.539 | 0.423 | 0.503 |
| **ρ(E_gap, F1)** | **+0.706** (p = 0.010, 12 solvers) | **+0.870** (p = 0.002, 9) | **+0.891** (p = 0.001, 9) |

A positive correlation means that, across each panel, solvers further from
the exact optimum score better. The exact solution is not exempt from the
trend; it sits at the losing end. In the V1 panel the best solver, QAOA at
depth 1, never once reaches the optimum (0 of 12) and scores 0.519, against
0.437 for every solver that does. This is the most direct finding of the
work, because it does not depend on QAOA: the panel contains the exact
solution, so a perfect optimizer would not improve on it. [+0.891 (V1) is
A: this review replayed the V1 panel on the current code and obtained it
bit for bit (Table 3). +0.870 and +0.706 (V2) are B: they were measured
under the legacy V2 normalisation, and the current V2 does not reproduce
them (ρ = −0.148 at the deployed threshold, Table 3). All three are
recomputed from their artifacts by the same script.] (script:
`study/common/rho_gap_f1.py`; test:
`tests/study/test_rho_gap_f1_reference.py`.)

**What QAOA's F1 measures, and why the optimum refines everything at this
threshold** (found during this review). Three facts, each checked on the
instances of the panels.

1. The Hamiltonian's bias is centred on the AMR threshold inherited from
   the deployed pipeline, 0.15 for V2 and 0.1496 for V1. On these block
   scores that threshold marks 88 of 108 cells for refinement, while the
   threshold that maximises F1 against this label, fitted on the training
   scenarios of the confirmatory run, is 0.51 to 0.60 (Section 5.4). The
   ground state follows its bias toward refining nearly everything: its F1
   is that of the refine-everything mask on 12 of 12 instances of the
   16 August V2 panel and on 9 of 12 of the V1 panel.
2. The study reads QAOA by majority (Section 2.4), and the unoptimised
   circuit read that way is the mask s > 0.5. It differs from the classical
   decision s > 0.15 on 34 of 108 cells (31%; on every scenario except the
   Harris sheet, whose block scores are either below 0.1 or above 0.67) and
   has a mean F1 of 0.507, against 0.468 for the classical decision (the
   V1-panel value of Table 2, reproduced exactly) and 0.386 for refining
   everything.
3. On the V1 panel, QAOA's F1 equals that of the mask s > 0.5 on 11 of 12
   instances at every depth, including 6 of the 7 instances on which the
   three masks give three different F1 values: the optimised circuit
   returns, to within one instance, the readout of the unoptimised one. On
   the V2 panel of 16 August, produced on older code, the count is 8 or 9
   of 12.

So the QAOA F1 of Table 2 is, on the trained mapper, essentially the F1 of
a threshold at 0.5 on the classical score. It beats the exact optimum
because 0.5 is closer to the F1-optimal threshold for this label than the
0.15 on which the bias is centred, which makes the optimum refine (nearly)
everything. That raises a direct question — is H0b only the consequence of
a threshold set for another task? — which no campaign could have answered,
because `threshold_amr` is outside its search space (Section 8). [A:
recomputed from the panels' DNS and from the per-instance F1
stored in the H0 artifacts by
`study/h0_selection/h0_readout_threshold_diagnostic.py` (artifact 5 Oct,
clean tree), and pinned by
`tests/study/test_h0_readout_and_bias_threshold.py`. The same test proves
that the study's implementation of the deployed rule returns, for given
marginals, exactly the cells the deployed solver refines. The artifacts
store F1 values, not masks, so the equalities above are equalities of F1.]

**Recentring the bias does not change the verdict** (experiment of this
review). The V1 panel of Table 2 was run again for both mappers under a
2 × 2 plan fixed before execution: the bias centred on the deployed
threshold or on the F1-optimal threshold of the classical score, fitted on
the three other scenarios as the classical arm of Section 5.4 does
("LOSO-F1": 0.52, 0.60, 0.59 and 0.51 when the Harris sheet,
Kelvin-Helmholtz, the rotor and Orszag-Tang are held out; the classical
decision of the panel uses the same threshold); and QAOA read by majority
or with the deployed rule. All eight runs started together from one
commit and a clean tree.

**Table 3. The recentring plan: same twelve instances as Tables 1-2,
current code. "Optimum" columns count instances whose exact ground state
refines every cell, refines none, or equals the classical decision.**

| mapper | bias threshold | QAOA readout | ρ(E_gap, F1) | p | F1 exact | F1 classical | F1 QAOA, p = 1-3 | QAOA reaches the optimum | optimum: all / none / = classical (of 12) |
|---|---|---|---|---|---|---|---|---|---|
| V1 | deployed | majority | +0.891 | 0.001 | 0.437 | 0.468 | 0.503-0.519 | 0 | 9 / 0 / 6 |
| V1 | deployed | deployed | +0.632 | 0.068 | 0.437 | 0.468 | 0.468-0.488 | 0.50 | 9 / 0 / 6 |
| V1 | LOSO-F1 | majority | +0.863 | 0.003 | 0.405 | 0.479 | 0.515-0.519 | 0 | 1 / 2 / 4 |
| V1 | LOSO-F1 | deployed | +0.863 | 0.003 | 0.405 | 0.479 | 0.479-0.507 | 0 | 1 / 2 / 4 |
| V2 | deployed | majority | −0.148 | 0.70 | 0.386 | 0.468 | 0.334-0.411 | 0-0.17 | 12 / 0 / 6 |
| V2 | deployed | deployed | −0.130 | 0.74 | 0.386 | 0.468 | 0.356-0.362 | 0.58-0.67 | 12 / 0 / 6 |
| V2 | LOSO-F1 | majority | +0.979 | < 0.001 | 0.108 | 0.479 | 0.341-0.357 | 0.17-0.25 | 4 / 8 / 0 |
| V2 | LOSO-F1 | deployed | +0.828 | 0.006 | 0.108 | 0.479 | 0.265-0.294 | 0.25-0.42 | 4 / 8 / 0 |

The first row replays the V1 column of Table 2 bit for bit: 108 rows,
identical F1 values and energies, five weeks later, in a different
environment, with one OpenMP thread per process. The plan answers the
question in four points.

- **Recentring never turns ρ negative.** Recentred, ρ is positive and
  significant on both mappers and both readouts (+0.83 to +0.98,
  p ≤ 0.006). The pre-registered condition for "tuning suffices" is met in
  none of the eight cells.
- **The exact optimum never decides better than the classical rule at the
  same threshold**, on average: 0.437 against 0.468 and 0.405 against
  0.479 for V1, 0.386 against 0.468 and 0.108 against 0.479 for V2. Counted
  by trajectory, the inference unit, the optimum wins 1 comparison of 16
  (V1 recentred, Orszag-Tang, 0.450 against 0.250), loses 11 and ties 4.
- **The mechanism is the coupling structure, not the threshold.** The V2
  ground state is a uniform mask on all 24 instance-threshold pairs:
  refine everything at 0.15; refine nothing on 8 instances and everything
  on 4 once recentred; never the classical decision. Its ferromagnetic
  couplings dominate a bias capped at 0.1 times the largest coupling, so
  the threshold only decides, through the sum of the biases, which uniform
  mask wins. On the recentred Harris sheet, 6 of 9 cells score above the
  threshold and the optimum refines none of them, because the 3 calm cells
  lie far below it. The V1 optimum, whose bias weighs more, is less often
  uniform once recentred (3 of 12) but not better: it equals the classical
  decision on 4 of 12 instances and loses on 3 of 4 trajectories.
- **The readout changes QAOA's F1, not the conclusion.** Read by majority,
  QAOA almost never reaches the optimum (0 on V1, at most 0.25 on V2); that
  is H0a. Read with the deployed rule at 0.15, it starts from the classical
  decision and reaches the optimum exactly as often as that decision on V1
  (0.50; at p = 1 its F1 equals the classical rule's). H0a is therefore the
  majority-readout statement, the only one that asks whether the circuit's
  state concentrates on the ground state.

On the current V2 code at the deployed threshold, ρ is no longer positive
(−0.148, p = 0.70): the +0.870 of Table 2 was measured under the legacy
normalisation. QAOA decides worse there than it did under that
normalisation (0.334-0.411 against 0.470-0.481), for a reason not isolated
here; the exact optimum still decides worse than the classical rule. The
robust form of H0b is therefore not "ρ > 0 everywhere" but "the exact
optimum never decides better than the classical rule at the same
threshold", true in all eight cells, with ρ positive wherever the bias is
recentred. [A: artifacts of 5 Oct from one commit and a clean tree; the
summary recomputes every exact ground state without QAOA and checks that it
reproduces the stored F1 instance by instance. Twelve instances on four
trajectories at Re = 400: the direction is clear, its generality is not
established, and the replication of Section 5.4 was not rerun with the
recentred threshold.] (scripts:
`study/h0_selection/h0_optimiser_equivalence.py` with `--bias-threshold`,
`--readout` and `--tag recentrage`, `study/common/bias_threshold.py`,
`study/h0_selection/h0_recentring_summary.py`; test:
`tests/study/test_h0_recentring_summary.py`.)

What "the optimum" is at this size. At `dim = 3` the ground state is
unique (non-degenerate) on every instance measured, but it is often a
uniform mask: in the coupling-ablation artifact of 29 August it is uniform
on 8 of 8 instances for V2 and on 4 of 8 for V1, and Table 3 finds it
uniform on 24 of 24 V2 instance-threshold pairs. For V2, H0b therefore says
that the Hamiltonian's ground state ignores the cell-by-cell evidence. The
pre-registered reading of this criterion (`rho_gap_f1.py`): a
hyperparameter campaign that turned ρ negative at `dim = 3` would show that
tuning suffices; ρ positive at the reference point and on the tuned mapper
(V1) puts the form of the Hamiltonian, not its tuning, in question — and
it stays positive when the one fixed parameter, the threshold, is
recentred (Table 3). At `dim = 2`, ρ = −1.000 is meaningless (Section
4.5).

At larger scale (Section 5.4), F1(QAOA) − F1(exact) has a 95% interval
excluding zero in favour of QAOA on three of four held-out scenarios, with
Orszag-Tang the one exception in all measurements.

### 5.3 H3: the coupling terms

We separate two questions: do the couplings change the exact decision at
all, and, when they do, is the changed decision better against the DNS
label?

**Table 4. Exact-optimum decisions and F1 against the DNS label, full
Hamiltonian versus bias only (couplings zeroed), by lattice size.**

| `dim` | qubits | search | mapper, artifact | decisions changed: no ZZ / no ZZZZ / bias only | F1 full | F1 bias only | F1 classical rule | coupling effect on F1 |
|---|---|---|---|---|---|---|---|---|
| 2 | 8 | exhaustive | V1, 7 Aug | 0 / 0 / 0 | 0.333 | 0.333 | 0.389 | +0.000 (degenerate) |
| 3 | 18 | exhaustive | V1, 29 Aug | 6.9% / 15.3% / 15.3% | 0.405 | 0.451 | not comparable at this pooling | −0.046 |
| 3 | 18 | exhaustive | V2, 29 Aug | 5.6% / 5.6% / 16.7% | 0.386 | 0.451 | not comparable at this pooling | −0.065 |
| 4 | 32 | greedy | V1, 7 Aug | 0 / 3.1% / 3.1% | 0.520 | 0.552 | 0.552 | −0.033 |
| 8 | 128 | greedy | V1, 7 Aug | 4.7% / 6.9% / 7.9% | 0.592 | 0.648 | 0.648 | −0.057 |

The coupling terms are never associated with a higher F1 than the
bias-only Hamiltonian, at any size where the search is informative. At
`dim = 2` they change nothing because nothing can change there (Section
4.5). From `dim = 3` on they change up to 17% of decisions, and each time
F1 goes down, by 0.033 to 0.065. At `dim = 4` and `dim = 8` the
bias-only F1 equals the classical rule's F1 exactly (0.5524 and 0.6481 to
four digits): the best case of this Ising formulation is to reproduce the
classical threshold, not to beat it. F1 rises with `dim` for every arm
alike (0.33 to 0.59 and 0.33 to 0.65): that is the decomposition getting
finer, not an effect of the couplings or of QAOA.

Controls and reservations. The control "full Hamiltonian, nothing removed"
changes exactly 0 decisions at every size, and the greedy proxy, forced at
`dim = 2` where enumeration is available, also finds 0 changes; but the
greedy and the exhaustive solutions differ on 25% of cells at `dim = 2`, so
for `dim ≥ 4` the scan measures "do the couplings change the decision of
the deployed-style solver", not "do they change the exact optimum". The
`dim = 3` rows are on the current code [A]; the `dim = 2, 4, 8` rows were
produced on 7 August on an earlier V1 — before the curl-convention,
relative-threshold and gate corrections of Section 4.2 — from a working
tree with uncommitted changes, and have not been re-run since [B]. (scripts:
`study/h3_representation/h3_term_ablation.py`,
`study/h3_representation/h3_size_scan.py`; tests:
`tests/study/test_t13_dim3_couplings_not_inert.py`,
`tests/study/test_t26_proxy_validation_surfaced.py`.)

For the exact optimum the same conclusion holds at larger scale with
confidence intervals (Section 5.4): the full Hamiltonian never beats the
bias-only version (two exact ties, two confidently negative folds).

The mean-field solution of the Hamiltonian points the same way. Sweeping
the weight of the bias relative to the couplings
(`study/h2b_prediction/h2b_analytical_solution.py`, Harris sheet, Re 400,
N = 96, `dim = 4`), the F1-optimal weight is at the bias-only limit: F1
saturates at 0.7405, below the classical baseline (0.830). An earlier sweep
of 52 configurations had returned flat curves whose "optimum" was the left
edge of the grid; the flatness was an artefact of the legacy normalisation,
and the optimum of the corrected sweep first sat on the right edge, until
the grid was widened to 10⁵ and an interior-optimum check separated an
unresolved edge from a genuine bias-only plateau (D-86, D-186). Not all 52
configurations were replayed under the corrected mechanism, and the 0.7405
is not pinned by a test. [B]

**Neighbour information for a learned model (cone curve).** A separate
question is whether neighbour information helps any model. A GBT is given
the features of its k-hop neighbourhood, k = 0 to 3, under
leave-one-scenario-out, on four Reynolds numbers, 20 snapshots, N = 96. At
`dim = 8` the k = 3 neighbourhood already covers 76.6% of the grid, so it
is no longer a neighbourhood; `dim = 16` is the first size at which all
four k are local (19.1% at k = 3). [A, artifact 21 Aug]

| | dim 8, 4 folds | dim 8, without Harris | dim 16, 4 folds | dim 16, without Harris |
|---|---|---|---|---|
| classical | 0.443 | 0.369 | 0.577 | 0.444 |
| k = 0 | 0.245 | 0.327 | 0.322 | 0.429 |
| k = 1 | 0.250 | 0.333 | 0.445 | 0.593 |
| k = 2 | 0.168 | 0.223 | 0.369 | 0.491 |
| k = 3 | 0.261 | 0.349 | 0.469 | 0.625 |

The curve is not flat: the steps at `dim = 16` are +0.123, −0.076 and
+0.100, against a pre-registered retirement threshold of 0.01, and the gain
of the first hop grows from `dim = 8` to `dim = 16` (+0.006 to +0.164
without the Harris fold). But no average can be cited in either direction:
on the Harris fold the GBT predicts no positive at all, so the conclusion
changes sign depending on whether that fold is counted (with it the cone
stays below the classical rule, 0.469 against 0.577; without it, it
exceeds it, 0.625 against 0.444). That zero is not physical: the GBT ranks
the Harris patches well (AUC 0.908, F1 0.659 at matched budget), but the
probability threshold fitted on the three other scenarios is never reached
(maximum probability 0.124 against a threshold of 0.400). The curve is also
non-monotone, measured at two sizes with one seed, and its `dim = 16` point
has a patch side of 6 cells (below the `dim ≤ N/8` rule). It shows that
neighbour features can help a learned model; it does not show that the
Ising couplings help, and Table 4 shows that they do not. (script:
`study/h2b_prediction/h2b_neighbour_cone_curve.py`; tests:
`tests/study/test_t1b_cone_curve.py`,
`tests/study/test_seuil_non_transfere_vs_absence_de_signal.py`.)

### 5.4 H2b and the confirmatory replication on real DNS

Sections 5.1-5.3 use reference settings and one Reynolds number. To test
whether the findings hold at larger scale, without any hyperparameter
training, the parameter-free mapper V2 was run with real QAOA on real DNS
across the four canonical scenarios and the four Reynolds numbers (400,
800, 1200, 1600), 10 snapshots per (scenario, Re) pair, giving 40 held-out
snapshots per leave-one-scenario-out fold (QAOA `reps = 2`, `K_opt = 60`,
4096 shots, ψ from the previous snapshot of the same trajectory). For each
held-out snapshot: the classical threshold and a GBT, both fitted on the
three training scenarios across all Re; QAOA on the full Hamiltonian; the
exact ground state of the same Hamiltonian; and both again with the
couplings zeroed. The classical threshold and the GBT never see the
held-out scenario; QAOA and the exact optimum have nothing to fit. The
operating points therefore differ: the classical threshold fitted on the
training scenarios is 0.51 to 0.60 depending on the fold, QAOA is read by
majority (Section 2.4), and the Hamiltonian's bias is centred on 0.15. [A,
artifact 11 Sep, clean tree; two commits landed during the 78.5-minute run,
neither touching the code it executed.]

**Table 5. F1 by held-out scenario, n = 40 snapshots per fold.**

| held-out scenario | classical | GBT | QAOA (full) | exact (full) | QAOA (bias only) | exact (bias only) |
|---|---|---|---|---|---|---|
| harris_tearing | 0.667 | 0.667 | 0.667 | 0.500 | 0.667 | 0.667 |
| kelvin_helmholtz | 0.571 | 0.571 | 0.423 | 0.421 | 0.422 | 0.421 |
| mhd_rotor | 0.690 | 0.456 | 0.476 | 0.351 | 0.485 | 0.569 |
| orszag_tang | 0.379 | 0.286 | 0.220 | 0.349 | 0.374 | 0.349 |
| mean | **0.577** | **0.495** | **0.446** | **0.405** | 0.487 | 0.501 |

**QAOA does not beat the classical threshold on any fold, and loses with
confidence on three of four.** The bootstrap interval on F1(QAOA) −
F1(classical) is strictly negative on `kelvin_helmholtz` [−0.154, −0.143],
`mhd_rotor` [−0.337, −0.119] and `orszag_tang` [−0.246, −0.079] (bootstrap
p = 0.000 each). On `harris_tearing` the interval is exactly [0.000, 0.000]:
the classical, GBT and QAOA rules produce the same F1 on every one of the
40 held-out snapshots, which is the most parsimonious explanation for a
zero-width interval, though cell-by-cell agreement was not verified
directly. **The GBT does no better:** it ties the classical rule on
`harris_tearing` and `kelvin_helmholtz` and loses on `mhd_rotor` and
`orszag_tang`; it never strictly beats it. Against the GBT, QAOA wins on
one fold (`mhd_rotor`, 0.476 against 0.456), ties on one and loses on two.

**H0a and H0b replicate.** Agreement between QAOA and the exact optimum is
0.667, 0.994, 0.106 and 0.622 by fold. F1(QAOA) − F1(exact) has a 95%
interval excluding zero in favour of QAOA on `harris_tearing`
[+0.167, +0.167], `kelvin_helmholtz` [+0.000, +0.005] and `mhd_rotor`
[+0.071, +0.187]; `orszag_tang` is again the exception, in the opposite
direction [−0.200, −0.064].

**H3 replicates for the exact optimum.** Full minus bias-only, exact
optimum: `harris_tearing` [−0.167, −0.167], `kelvin_helmholtz` exact tie,
`mhd_rotor` [−0.263, −0.169], `orszag_tang` exact tie. The full Hamiltonian
never beats the bias alone. For QAOA, which does not solve the Hamiltonian,
the picture is the same at this scale (one exact tie, one near-tie
[+0.000, +0.003], one inconclusive fold [−0.108, +0.075], one confidently
negative [−0.233, −0.085]).

**What this replaces.** A first run of the same script (Re = 400 only, 5
snapshots per scenario, no intervals, 10 Sep) had QAOA ahead of the
classical threshold on average (0.465 against 0.419, three folds of four)
and the couplings slightly helpful for QAOA. Both readings disappear at 8
times the sample and four Reynolds numbers; the first run is kept in
`docs/RESULTS.md` as an illustration of what an underpowered comparison
shows, and is not cited as evidence. (script:
`study/h2b_prediction/h2b_v2_hamiltonian_vs_gbt_loso.py --re 400 800 1200
1600 --n-snaps 10`; test:
`tests/study/test_h2b_v2_hamiltonian_vs_gbt_loso_multire.py`.)

**Other learned models.** No model tested in `study/h2b_prediction/` beats
the classical threshold under leave-one-scenario-out. One comparison must
not be cited as evidence that "the physics beats machine learning": with
the V1 mapper and ψ at `dim = 4`, N = 256, Re = 400, the classical and
V1-derived scores reach F1 0.52-0.55 while the GBT ceiling reaches only
0.29-0.32, but the GBT's weakness there has an identified cause unrelated
to the physics. The classical score is one of its nine features, and the
relation between that score and the label changes sign from one scenario
to the next (mean score of positive against negative patches: 0.677/0.649
on the Harris sheet, 0.732/0.740 on Kelvin-Helmholtz, 0.647/0.057 on the
rotor, 0.381/0.485 on Orszag-Tang). A GBT that learns the relation on three
scenarios transfers it badly to the fourth; on the rotor fold a raw
threshold on that single feature reaches 0.636 where the GBT on the same
feature reaches 0.163. Normalising each scenario by its own statistics
repairs the rotor fold and breaks the Harris one (mean 0.278 → 0.165). This
remains unresolved (D-198). [A for the measurements; the comparison itself
is not admissible as a verdict.]

### 5.5 Synthetic generators

Two synthetic generators were built to test the pipeline away from the
eight fixed scenarios.

**Static generator** (`study/common/toy_model.py`): random superpositions
of vortices, X-points and current sheets, with v and B written as curls of
flux functions, solenoidal to 10⁻¹⁰. On 300 instances split 70/30 by
instance ten times, the classical threshold is stable (0.4980 ± 0.0040) and
generalises without memorisation (F1 0.695 ± 0.004 train, 0.698 ± 0.012
validation). Learned ceilings on the same splits: logistic regression
0.726 ± 0.009 (best on 5 of 5 splits), random forest 0.686 ± 0.020, GBT
0.677 ± 0.016, GBT with 45 neighbour features 0.723 ± 0.019, against
0.702 ± 0.012 for the classical threshold. A local learned model gains
+0.024 over the threshold; neighbour features add nothing (−0.003). A
random split is the honest protocol here because the instances are
independent draws from one distribution; it says nothing about transfer.
[A; these artifacts were produced from a working tree with uncommitted
changes.] (scripts: `study/h2b_prediction/h2b_toy_ceiling.py`,
`h2b_toy_gbt_ceiling.py`; tests: `tests/study/test_h2b_toy_ceiling.py`,
`tests/study/test_h2b_toy_gbt_ceiling.py`.)

**Dynamic generator** (`init_toy_instability`): one of seven instability
recipes of the solver, with randomised physical parameters and a random
periodic shift, evolved by the real solver. It replaced a filtered-noise
generator that produced a degenerate "refine everything" decision at the
training resolution whatever its bandwidth: filtered noise diffuses, while
a real instability concentrates structure over time (a real Harris sheet
under the same short settings gave a non-degenerate decision, 0.735 and
0.781 refined for the two arms). With the new generator 2 of 5 draws give a
non-degenerate decision.

**A comparison that does not test what it was written to test.** Both
synthetic harnesses (`h3_toy_model_check.py`, static, 20 instances,
`dim = 3`; `h3_toy_instability_check.py`, dynamic, 8 instances, N = 256,
`dim = 2`) were written to replicate H0a and H0b. During this review we
found that they compute the exact optimum with V2
(`build_patch_hamiltonian(..., use_v2=True)`) and run QAOA through
`prepare_qaoa_inputs` with its default, V1. The "agreement between QAOA and
its own exact optimum" (0.622 ± 0.147 static, 0.719 ± 0.232 dynamic) and
"QAOA beats the exact optimum" (F1 0.673 against 0.500 static, 0.629 against
0.425 dynamic) therefore compare two different Hamiltonians and are not
replications of H0a or H0b. What these numbers do show is three decision
rules scored against the same label: the exact optimum of V2 refines
everything on 20 of 20 static instances (F1 0.500), QAOA on V1 reaches
0.673 ± 0.101 and the classical threshold 0.699 ± 0.154. The defect is
open (D-202 in `docs/DEFAUTS.md`) and pinned by a strict expected-failure
test that will fail the day it is fixed
(`tests/study/test_toy_harness_mapper_mismatch.py`); the real-DNS panels
of Sections 5.1-5.4 are not
affected (the confirmatory script uses V2 for both QAOA and the exact
optimum, and the panels of Tables 1-2 use one mapper per run).

### 5.6 The temporal phase ψ does not improve the decision

The phase ψ was meant to let the circuit anticipate an instability: two
adjacent cells whose stress flux grows together would interfere
constructively before their amplitude becomes large.

- **Under leave-one-scenario-out** (V1, `dim = 4`, N = 256, Re = 400, four
  canonical scenarios, artifact 27 Aug): adding ψ lowers the mean F1, from
  0.548 for the V1 classical score without ψ to 0.518 for the best ψ
  variant (the V2 classical score: 0.552). [A] (script:
  `study/h2b_prediction/h2b_psi_feature_loso.py`.)
- **Sign of the mechanism.** In V1's own unit tests, re-armed and measured
  over 30 draws, the "phase boost" lowers the refinement probability of the
  cell it marks instead of raising it: contrast −0.0572 (t = −8.4, negative
  on 93% of draws) in one construction, −0.0723 (t = −14.6) in another. The
  old assertions took the absolute value of the contrast, which is why the
  sign was never seen. [B: measured before the flux corrections below, not
  re-measured.]
- **The flux that ψ is built from was wrong until mid-August.** A
  compression "diode" was applied to the tangential (shear) difference
  instead of the normal one (ratio 0.500 instead of 2.0; 37-97% relative
  error on Φ on real snapshots, D-11), and the flux was downsampled by a
  smoothing and bilinear path that kept 38% of its peak on Orszag-Tang and
  70% on the rotor (D-21). Every ψ ablation before these corrections used a
  wrong flux. The one ψ ablation stored in the H0 artifacts was found to be
  empty: the "zero-ψ" run had ψ = 0 before the ablation (D-122).

### 5.7 H5: is the label the problem?

**On three scenarios of four the static label is almost given by the
classical score.** The area
under the ROC curve of the classical score alone against the static label
(`dim = 16`, N = 96) is 1.000 on the Harris sheet, 0.997 on
Kelvin-Helmholtz, 0.948 on the rotor and 0.592 on Orszag-Tang. On the
first three, the label is almost a deterministic function of the
classical score: the task leaves little room above the baseline, and a gap
measured against a near-perfect baseline does not measure what one thinks.
[A]

**A dynamic label at the protocol's horizon repeats the static one.** With
the dynamic label of Section 2.2 at the protocol's δt = 0.1 (N = 96,
`dim = 8`, 5 snapshots per scenario), its Spearman correlation with the
static label is 1.000 (Harris), 0.997 (Kelvin-Helmholtz), 0.992 (rotor) and
0.982 (Orszag-Tang): at that horizon a perturbation travels only 0.11 to
0.25 of a patch width, and the amplification d_i/d0_i hardly varies between
patches. At δt = 2.0, Orszag-Tang separates (ρ = 0.596; the only scenario
where perturbations are amplified, median ×1.38), the other three stay at
0.93-1.00. [A, artifacts 22 Aug]

**At the physical crossing time the verdict is mixed.** Regenerated at
t_x (0.39-0.43 on Orszag-Tang to 0.87-0.88 on the Harris sheet, 4 to 9
times the protocol's horizon), the dynamic label stays redundant on the
Harris sheet and Kelvin-Helmholtz (ρ ≥ 0.965 on every snapshot) and
diverges on the rotor (0.817 at the first snapshot, one of five below the
module's redundancy limit of 0.95) and on Orszag-Tang (0.658 and 0.919 at
the first two; four snapshots of five below 0.95). Correcting the horizon
is necessary and exposes a signal on half of the canonical panel, not on
all of it; any future task using d_i must set its horizon on t_x. The
script's own "informative" flag is true on all 20 snapshots, because it
also fires when the amplification d_i/d0_i varies between patches
(log-IQR 0.115 to 0.389, threshold log 1.10 = 0.095): even where the
ranking repeats the static label, the rate at which the error grows
differs between patches, which a static label cannot carry. Whether that
rate helps a refinement decision was not tested. [A, artifacts 26 Aug]
(script: `study/pipeline/dynamic_patch_labels.py`; test:
`tests/study/test_dynamic_patch_labels.py`.)

H5 is a secondary, scenario-dependent effect. All comparisons in Sections
5.1-5.4 use the same static label for every arm.

## 6. Results III — the closed loop, transfer, and numerical defects

### 6.1 Status of the closed-loop artifacts

The closed-loop study (protocol in Section 2.6) is the experiment that asks
the original question directly: inside the adaptive solver, at equal cost,
does the Q-HAS decision give a more accurate simulation than a classical
threshold? It was run between 2 and 7 August. Every one of its artifacts
predates corrections made afterwards to the code it executed: the bias of a
refined patch read from the wrong quarter of the patch at every level below
the root (D-37, fixed 12 August; it affected only the Q-HAS arm, at three of
the four levels of the deployed depth), the curl convention of the mapper
and the score (D-1, 11 August), the global prolongation (D-2), the flux
diode and the flux downsampling feeding ψ (D-11, D-21), the misaligned field
and score reductions (D-14) and the self-overlapping patch list (D-16).
Under the project's evaluation rules, every Q-HAS number obtained with more
than one refinement level before the D-37 fix is obsolete (level D); every
historical run used `max_depth = 4`. The numbers below are therefore
reported as the history of a study whose methodology remains valid and whose
conclusion has not been re-measured on the current code; they are not
evidence about the current Q-HAS. Re-running it is part of the campaign that
was not run (Section 8).

### 6.2 What the closed-loop study measured [D]

- **The first reading was an artefact of the operating point.** On fold
  `ot`, the pre-registered composite favoured Q-HAS (0.333 against 0.439),
  but the two arms ran at different costs: Q-HAS used a threshold fixed at
  0.1496 while the classical arm tuned its own to 0.462. Against the
  classical error measured at the cost Q-HAS actually spent, the classical
  rule achieved a 2.3 times lower error (0.083 against 0.194) at slightly
  less cost (0.641 against 0.680). On fold `kh` the classical arm won every
  endpoint at its own tuned point (error 0.0020 against 0.0070, cost 0.625
  against 0.838).
- **The Q-HAS arm is not deterministic.** Replayed with identical inputs,
  the classical arm reproduced its stored value bit for bit on all four
  folds; Q-HAS on none (a 44% swing in error on `ot`). Repeated five times,
  Q-HAS's error has a coefficient of variation of 17% to 64% per fold. At
  five draws, which fold's gap exceeds two standard deviations is itself
  unstable between two passes (`ot` then `rotor`), so per-fold magnitudes
  cannot be quoted. The verified mean Q-HAS error exceeds the
  budget-matched classical error on 4 of 4 folds.
- **Dominance counts over repeated draws.** Over 18 completed runs on four
  held-out classes, Q-HAS was less accurate than the budget-matched
  classical rule on 18 of 18, more expensive on 16 of 18, and strictly
  dominated on both coordinates on 16 of 18; no run reversed the order on
  both at once. This count was first written by hand as 19/20, 18/20,
  17/20; recomputed from the artifacts it is 18/18, 16/18, 16/18.
- **The verdict does not depend on λ.** With the failed fold excluded as
  the pre-registration requires (on `rotor`, the tuned classical threshold
  diverged), the classical arm holds the majority at every λ of a 12-point
  grid from 0 to 100 (2 of 3 folds up to λ = 0.8, 3 of 3 from λ = 1.0); two
  of three usable folds are decided by Pareto dominance alone.
- **Robustness.** At the compared operating point, 2 of 20 Q-HAS draws
  aborted (solver divergence) against 0 of 8 classical replays; the
  classical rule also diverges at other thresholds (the tuned threshold on
  `rotor`), so divergence is a property of the threshold, but at the
  operating point actually compared one arm completed and the other did
  not. An aborted run can return a plausible value (0.407 among valid draws
  of 0.054-0.219) and can even look better than a valid one (truncated runs
  accumulate less error), so completion must be recorded at execution
  time; it cannot be inferred from the value.
- **A leak, removed.** The Q-HAS arm's threshold 0.1496 had been fitted on
  all four classes, including the held-out one (D13), an advantage to Q-HAS.
  With the leak removed (the fold's own classical threshold substituted,
  without re-tuning Q-HAS), Q-HAS lay above the classical frontier at its
  own realised cost on all three folds where a ratio could be computed
  (about 1.6×, 1.9× and 2.1× on the canonical conditions), had no operating
  point at all on `rotor` (5 of 5 canonical draws aborted), and aborted on
  12 of 40 draws against 0 of 16 for the classical arm. This is a bound,
  not the definitive experiment, which would re-tune Q-HAS with the
  threshold in its search space.
- **Cost of the decision itself.** The cost axis counts refined pixels
  only. The Q-HAS arm took 2.7 to 3.3 times the classical arm's wall time
  on a simulated 8-qubit circuit; counting it would only strengthen the
  direction above.

### 6.3 Transfer to unseen conditions (H4)

Evidence from the closed loop, all level D for the reason given in 6.1:

- On genuinely new initial conditions (thinner current sheet and mode 2
  for Harris; narrower shear layer, weaker seed and faster drift for
  Kelvin-Helmholtz; slower, smaller rotor), Q-HAS was strictly dominated on
  18 of 20 runs. The difference in degradation between the two arms was
  separable on one fold of four only (`tearing`, where Q-HAS degraded
  less), and that one apparent transfer advantage reversed once the leaked
  threshold was removed (degradation ×0.685 for Q-HAS against ×0.389 for
  the classical rule; on `kh` ×4.84 against ×1.36). Orszag-Tang exposes no
  initial-condition parameter; its only "unseen" condition, a different
  Reynolds number, moved the trajectory by 0.3%, and it was excluded from
  any transfer claim before its result was known.
- Against a near-full-refinement reference, Q-HAS was further from the
  reference than the classical arm in 8 of 8 comparisons (four folds, two
  conditions each, each at the same operating point).
- There is no physical seed to vary: three of the four canonical
  initialisers are deterministic, and the rotor's hard-coded random seed
  moves the trajectory by 0.0022%. The pre-registered "three physics seeds"
  was never available. Of seven alternative initial conditions tried, two
  were vacuous, three gave no sound verdict (non-monotone or unconverged
  frontier, or a budget outside the swept range), and the two decidable
  ones split: Q-HAS 1.24× worse on one, 0.86× (better) on the other. The
  guards that refused a verdict removed evidence favouring the study's
  direction and kept the one result against it.

What is current [A]: V2's coefficients do not depend on ν, η or dx (Section
2.3), so a transfer across Reynolds numbers is trivially satisfied by them;
any Reynolds dependence of a V2 decision comes from the classical score.
The level-3 campaign that would test H4 has 4 of its 8 folds, at smoke
scale (4 Optuna trials instead of 170) and without the campaign-contract
hash the current code requires (D-197). **H4 remains a conjecture.**

### 6.4 Numerical defects (H1)

Several numerical defects found by the audit mattered: the first-order
time convergence (Section 4.1), the prolongation error on the unrefined
background (~1.7% of the field), the misaligned field and score reductions
(coverage 94.1-98.5% at the deployed size), the self-overlapping patch list
and the misread bias at depth > 0. Most were common to both arms; D-37, D-1
on the plaquette, and the flux defects behind ψ were specific to Q-HAS.
Nothing in this work isolates whether such defects, on their own, would be
sufficient to explain a failure; the optimization (H0a, H0b) and
representation (H3) results of Section 5 are measured on corrected code
and do not depend on them. **H1 is partial.**

## 7. Results IV — methodological findings

The audit that preceded and accompanied the results produced findings that
hold independently of Q-HAS.

**Contracts, not values.** Each function on the decision path was asked
five questions: what it promises, what it consumes, whether it fails
loudly, whether two paths that should agree still agree, and whether the
test guarding it can fail. The registers of `docs/RESULTS.md` and
`docs/COUVERTURE.md` record more than 190 contract defects found and
closed this way, numbered up to D-201, each measured before and after and
pinned by a test that fails on the old code. Three of the four first
contract defects were found by the single question "do two paths that
should coincide still coincide?" (a diode against its own documentation,
a left boundary against a right boundary, the field reduction against the
score reduction).

**A computation that fails but returns a plausible value.** In the closed
loop alone, seventeen instances of one failure mode were found: a
divergence guard returning a partial score with the same keys as a
complete one (4 times), a fixed output filename overwriting a previous
result (6), an aggregation averaging aborted draws with valid ones (1), a
documented command-line mode never implemented (1), and five more found by
recomputing published numbers from their artifacts (a total abort discarded
before saving, the hand-written headline count, a reference recorded at the
wrong threshold, a provenance stamp taken at save time instead of start
time, and a printed sentence describing a computation that did not happen).
Every number that no script produced turned out to be wrong; every number
recomputed mechanically from its artifact was right. The defence that
worked was making each published number a function of its artifact and
checking it automatically (the master table, Section 2.8).

**The operator that measures must match the operator that produced.** At
least five times, a quantity measured with a different discrete operator
from the one that produced it gave the wrong verdict: a spectral divergence
hid an eight-order-of-magnitude defect in a finite-difference field; an
analytic derivative made a perturbation look non-solenoidal at 2 × 10⁻⁵; a
reconstruction of a coefficient that omitted the code's own filter made
correct code look wrong; a bit-order convention made two identical
operators look different by 18.8. The mapper mismatch in the synthetic
harnesses (Section 5.5), found during this review, is one more instance.

**Tests that cannot fail.** Before the audit, 44 of 175 tests of the
original code base failed, and 8 of the 17 stages of the default test run
contained no assertion at all while printing "Classical" as the winner on
6 of 6 rows. The original unit tests contained, in red, a falsification of
the model's central claim: the coupling that was supposed to add spatial
information was multiplied by the uncertainty window down to 1.8 × 10⁻⁴²
(the ratio equals exp(−100) to double precision). A later audit of the test
suite itself (D-118 to D-175) found guards that only searched the source
text for a string (defeated by an import alias or a comment), sweep floors
far below the real counts (40 against 153), and checks that the inventory
of covered names was searched in a corpus that contained the inventory
itself.

**Evidence of a single draw.** Seven assertions on the QAOA arm had been
calibrated on one draw. Measured over repeats, a vortex "detection"
contrast was centred on zero with a sign that flips from run to run, and
the displacement of a marginal ranged from 0.07 to 0.47 over 12 identical
calls. A quantity produced by a stochastic arm needs its own variance
measured before any threshold is set on it.

**Provenance.** A commit hash stamped at save time pointed at code that an
hour-long run had not executed; hashes are now captured at start, together
with whether the working tree was clean and whether HEAD moved during the
run. A campaign-contract hash refuses to resume a study under a different
definition. The deployed hyperparameters were found to have no reproducible
provenance (Section 8).

## 8. Why no full hyperparameter campaign was run

The protocol foresees a full Optuna re-optimisation of V1 (nine
hyperparameters, eight training scenarios, 600 + 600 + 400 trials for the
quantum arm and 3 × 300 for the classical one, final selection on six
held-out physical regimes) and an eight-fold matched-budget closed-loop
confirmatory comparison (170 Optuna trials per fold). Both are implemented
and smoke-tested; neither was run. Measured on the hardware available, not
assumed: the re-optimisation alone requires several weeks of continuous
compute (one quantum-arm trial costs a median 56 minutes of CPU; the frozen
historical campaign cost about 224 CPU-hours), and the confirmatory
campaign a comparable order.

We judged this cost unjustified for three reasons. The reference starting
point that a campaign would tune from already sits in the pathological
regime it would be run to test (H0b on V1, Section 5.2). The one
parameter that sits outside the search space, the threshold on which the
bias is centred (`threshold_amr`), was moved by hand to its F1-optimal
value during this review, and ρ stayed positive (Table 3): its absence from
the search space does not weaken this reason. The three central
hypotheses (H0a, H0b, H3) could be re-measured without any training,
directly on real DNS, at a larger sample size than a campaign's own folds
would have given (Section 5.4). And the deployed hyperparameters have no
reproducible provenance to start from: the deployed file carries values
(`gamma_hydro` 2.127, `gamma_mag` 2.361, `kappa` 14.332) that appear in no
Optuna database, omits `sigma` (whose best sampled value was 0.0230) so
that the pipeline falls back on a hard-coded 0.05, and announces a trial
whose recorded loss and parameters do not match the database; the frozen
campaign had only ever sampled five parameters (D-22). The substitution is
reasonable but not equivalent: it does not rule out that some point of the
nine-dimensional space reverses the sign of ρ(E_gap, F1), only that the
reference point does not. The preflight of Section 4.3 found a point of
that space whose plaquette coefficient correlates slightly better with the
true error than the classical score (0.843 against 0.814); whether its
Ising ground state is a better decision is exactly what the campaign would
have to show.

## 9. Discussion

The project set out to find topological instabilities in a chaotic MHD
flow and to let a quantum optimizer decide where to refine. The two halves
of that sentence have different outcomes.

The detection half worked, once the instrument was repaired. The sensors
now respond to the structures they were designed for: the X-point term is
selective for hyperbolic nulls and silent on islands, the plaquette term
separates circulation and current from calm flow, and the coefficients rank
blocks by their true coarse-grid error about as well as the classical
indicator. Getting there required finding that two indicators measured
strain instead of vorticity and divergence, that the X-point detector was
gated to zero at the X-point, that a gate compared quantities in two unit
systems, and that the four-body terms vanished at the training resolution.
These corrections are the main technical contribution of the work, and
they are what makes the negative result below meaningful: the decision was
tested on an instrument that sees.

The decision half did not work, and it fails along two independent angles
that agree. Optimization: QAOA does not reach the ground state of its
Hamiltonian (H0a), and reaching it would not help, because the ground state
is a worse decision than the classical rule at the same threshold (H0b).
The obvious suspect, a bias centred on a threshold of about 0.15 far below
the 0.51-0.60 that maximises F1 for this label, was tested and cleared
(Section 5.2): recentred, the ground state is no better and ρ stays
positive. What the measurements show instead is structural: the couplings
pull the ground state toward a uniform mask (always, on the
parameter-free mapper), and the threshold only decides which one. QAOA's F1
is better than the exact optimum's not because it optimizes better — it
rarely reaches the optimum — but because the study reads it by majority,
and on the trained mapper the optimised circuit read that way gives the
same F1 as a threshold at 0.5 on the classical score on 11 of 12
instances.
Representation: the coupling terms that were meant to add neighbour
information never improve the exact decision and lower it once they are
active (H3); the best case of the formulation is to reproduce the classical
threshold. A third check (H2b) shows that relaxing the Ising form to a
flexible learned model does not close the gap either: at the largest scale
measured, the classical threshold beats or ties both QAOA and a GBT on every
fold.

Two observations narrow what a future attempt would have to change.
Neighbour information is not useless in itself: a learned model gains from
it at the one size where the question is well posed, in a setting whose
average cannot be cited (Section 5.3). And the coefficients can be pushed
slightly above the classical indicator in ranking quality somewhere in
their hyperparameter space (Section 4.3). Neither says anything about the
ground state of the Hamiltonian. The cheapest change, centring the bias on
a threshold calibrated for the decision being scored and reading QAOA with
the deployed rule, has been tried and does not help (Table 3). What is
left to change is the balance between the local term and the couplings —
and Section 5.3 shows that the limit of that change, the bias alone,
reproduces the classical rule rather than beating it. The diagnostic any
new proposal should pass
first is the one that settled H0b: does solving the proposed Hamiltonian
better make the refinement decision better (ρ(E_gap, F1) < 0) on a size
where the optimum is not trivial?

An earlier version of this analysis rejected the optimization hypothesis
outright, on measurements made entirely at `dim = 2`. At that size the
exact ground state is "refine everything" whatever the Hamiltonian, every
solver reaches it, and no comparison is informative. Separating H0a from
H0b and measuring both at `dim = 3` is a materially better-supported basis
for the same qualitative conclusion.

## 10. Limitations

**One classical reference rule.** The comparison is against one indicator
and one threshold, refit per fold. We do not claim it is the best possible
classical AMR criterion, only that it is the inexpensive baseline the
quantum approach would need to beat, and that it was not beaten.

**No hyperparameter campaign** (Section 8). V1 is measured at reference
hyperparameters, at one point of a nine-dimensional space.

**Small panels at the certified size.** The H0 panels have 12 to 32
instances at `dim = 3`; the confirmatory run has 40 held-out snapshots per
fold on four scenarios. The `dim = 2, 4, 8` rows of Table 4 are on an
earlier version of V1 (level B).

**At `dim = 3` the exact optimum is often the trivial mask** (Section 5.2),
so H0b at that size is partly a statement about a near-trivial optimum.
Larger sizes cannot be enumerated; the greedy proxy used there measures the
decision of a deployed-style solver, not the exact optimum.

**The study reads QAOA differently from the deployed solver** (Section
2.4). The QAOA F1 of Sections 5.1-5.4 is that of a majority vote on each
qubit, which on the trained mapper gives the F1 of a threshold at 0.5 on
the classical score on 11 of 12 instances; the deployed rule reads the same
circuit against the AMR threshold and starts from the classical decision.
QAOA's F1 and its rate of reaching the optimum depend on this choice; H0a
is stated for the majority readout, the only one that asks whether the
state concentrates on the ground state. The H0b conclusion does not depend
on it: Table 3 measures both readouts and finds ρ positive on the trained
mapper under each.

**QAOA variance.** The QAOA arm is non-deterministic between calls (per-call
range of block scores 0.18-0.36). Conclusions based on rank are supported;
we avoid conclusions that would depend on the value of a single run.

**Shared numerical limits.** First-order time convergence and the
remaining numerical limits of Section 4.1 affect both arms identically and
are not expected to bias the comparison, but they bound the absolute
accuracy either arm can reach.

**Simulation only.** All circuits are noiseless statevector simulations of
at most 18 qubits run as circuits (128 qubits only through the classical
greedy proxy); nothing is claimed about hardware.

**The label is almost given by the classical score on three scenarios of
four** (Section 5.7), which limits the room any criterion has above the
classical baseline.

**The closed loop is historical** (Section 6.1), and transfer (H4) has no
current dedicated experiment.

**Artifacts from a modified working tree.** The size scan of Table 4
(`dim = 2, 4, 8`), the static synthetic-model artifacts, and some
closed-loop artifacts record that they were produced from a working tree
with uncommitted changes. Their commands reproduce them from a clean
commit; these particular files were not.

## 11. Conclusion

Restated plainly. The Q-HAS mapping can be made to see the topological
structures of a 2D MHD flow — current sheets, vortices, X-points — and its
coefficients point at the regions that need refinement about as well as a
classical indicator. But a local Ising Hamiltonian solved by QAOA, used as
the adaptive-mesh-refinement decision, does not outperform a classical
threshold on the same physical score at any scale at which we tested it.
The failure closes along two independent angles that agree — optimization
(H0a, H0b) and representation (H3) — and relaxing the model to a flexible
learned form does not recover the gap (H2b). The one fixed parameter that
could have explained H0b, a bias centred on a threshold far below the one
that maximises F1 for the scored label, was recentred and does not change
the verdict: the couplings drive the ground state toward a uniform mask
that ignores the local evidence. A replication on
real DNS across four Reynolds numbers with bootstrap intervals, without any
hyperparameter training, reproduces the verdict and adds one: at that
scale the classical threshold beats or ties both the quantum and the
learned alternative on every fold. We report this as a negative result
obtained with the measurement rigour the project would have applied to a
positive one, together with the instrument corrections and the
methodological findings that made the measurement possible.

## 12. Code and data availability

All code, data-generation scripts, artifacts and the exact commands that
reproduce every number in this report are in the repository; the numbers
were re-checked at commit `294319d0e179b0a2d3e2afa06a42da3b974090bc` with
the pinned environment of `requirements.txt` (Python 3.11). Each result
names its script and, where one exists, the test that pins its value.

What was run for this revision: the 41 test files that pin the cited
numbers (605 passed, 1 expected failure — the acceptance criterion of the
unrun campaign —, slow tests deselected), the coefficient preflight (5 of 5
checks pass), the master-table aggregation (268 rows: 139 OK, 6 DIFF, 123
MISSING, identical to the committed table), and `rho_gap_f1.py` on every
`dim = 3` H0 artifact. The full suites of the project's four-check
verification (`CLAUDE.md`) were not re-run for this revision; no claim is
made here that the whole code base passes them. The project's own record of
its last full run (`docs/DEFAUTS.md`, commit `c52c1de`, 29 August) lists two
tests of the fast suite as red by construction since D-195
(`test_hyperparameter_sweep`, `test_noise_robustness`): they pin values that
are themselves evidence for H0a, not regressions.

Selected reproduction commands:

```bash
# Sections 4.2-4.3: coefficient preflight (specificity, balance, liveness, relevance, coincidence)
python study/common/preflight_coefficients.py

# Section 4.1: solver order and conservation
python study/h1_solver/h1_solver_convergence.py

# Tables 1-2 (H0a, H0b; dim = 3). The script no longer selects the legacy V2
# normalisation of the 9 and 16 August panels: the first command measures the
# current V2 (12 instances; --qaoa-reps 1 2 3 4 5 6 --n-snaps 8 for the size
# of the 9 August panel), the second regenerates the V1 panel, and
# rho_gap_f1.py re-scores the committed artifacts of all panels as they are.
python study/h0_selection/h0_optimiser_equivalence.py \
    --scenario harris_tearing kelvin_helmholtz mhd_rotor orszag_tang \
    --re 400 --N 96 --dim 3 --qaoa-reps 1 2 3 --n-snaps 3 --k-opt 60 --seed 0
python study/h0_selection/h0_optimiser_equivalence.py \
    --scenario harris_tearing kelvin_helmholtz mhd_rotor orszag_tang \
    --re 400 --N 96 --dim 3 --mapper v1 --qaoa-reps 1 2 3 --n-snaps 3 --k-opt 60 --seed 0
python study/common/rho_gap_f1.py results/h0_optimiser_equivalence_N96_dim3*.npz

# Section 5.2: readout diagnostic, then the recentring plan of Table 3 (one
# run per mapper x bias threshold x readout, all from the same clean commit)
python study/h0_selection/h0_readout_threshold_diagnostic.py
python study/h0_selection/h0_optimiser_equivalence.py \
    --scenario harris_tearing kelvin_helmholtz mhd_rotor orszag_tang \
    --re 400 --N 96 --dim 3 --qaoa-reps 1 2 3 --n-snaps 3 --k-opt 60 --seed 0 \
    --mapper v1 --bias-threshold loso-f1 --readout deployed --tag recentrage
python study/h0_selection/h0_recentring_summary.py

# Table 4 (H3): dim = 3 rows (once per mapper), then the dim = 2, 4, 8 rows
python study/h3_representation/h3_term_ablation.py \
    --scenario harris_tearing kelvin_helmholtz mhd_rotor orszag_tang \
    --re 400 --N 96 --dim 3 --n-snaps 2 --mapper v1     # and --mapper v2
python study/h3_representation/h3_size_scan.py \
    --scenario orszag_tang harris_tearing kelvin_helmholtz mhd_rotor \
    --re 400 --N 256 --dims 2 4 8 --n-snaps 3 --mapper v1

# Table 5 (confirmatory replication, real DNS, 4 Re values)
python study/h2b_prediction/h2b_v2_hamiltonian_vs_gbt_loso.py \
    --re 400 800 1200 1600 --n-snaps 10

# Section 5.7: dynamic label at the crossing time (once per canonical scenario)
python study/pipeline/dynamic_patch_labels.py --scenario orszag_tang \
    --re 400 --N 96 --dim 8 --snaps 5 --seed 0 --allow-redundant

# regression tests pinning the main numbers
pytest tests/study/test_h0_certified_dim3_contradicts_criterion.py \
       tests/study/test_rho_gap_f1_reference.py \
       tests/study/test_h0_readout_and_bias_threshold.py \
       tests/study/test_h0_recentring_summary.py \
       tests/study/test_toy_harness_mapper_mismatch.py \
       tests/study/test_t13_dim3_couplings_not_inert.py \
       tests/study/test_t26_proxy_validation_surfaced.py \
       tests/study/test_h2b_v2_hamiltonian_vs_gbt_loso_multire.py \
       tests/study/test_t1b_cone_curve.py \
       tests/study/test_dynamic_patch_labels.py \
       tests/mapping/test_coefficient_families_contract.py \
       tests/mapping/test_selectivite_des_coefficients.py -v
```

## Acknowledgments

[Not filled in.]

## References

Carried over from the Q-HAS project's originating report (Presentation);
not independently re-verified here, and not yet checked for completeness
against this manuscript's own citations in the text above.

1. J. P. Freidberg, *Ideal MHD* (Cambridge University Press, 2014).
2. J. P. H. Goedbloed and S. Poedts, *Principles of Magnetohydrodynamics*
   (Cambridge University Press, 2004).
3. J. Wesson, *Tokamaks*, 4th ed. (Oxford University Press, 2011).
4. S. Chandrasekhar, *Hydrodynamic and Hydromagnetic Stability* (Clarendon
   Press, Oxford, 1961).
5. H. P. Furth, J. Killeen, and M. N. Rosenbluth, "Finite-resistivity
   instabilities of a sheet pinch," Phys. Fluids 6, 459-484 (1963).
6. G. Bateman, *MHD Instabilities* (MIT Press, 1978).
7. S. B. Pope, *Turbulent Flows* (Cambridge University Press, 2000).
8. M. J. Berger and J. Oliger, "Adaptive mesh refinement for hyperbolic
   partial differential equations," J. Comput. Phys. 53, 484-512 (1984).
9. M. J. Berger and P. Colella, "Local adaptive mesh refinement for shock
   hydrodynamics," J. Comput. Phys. 82, 64-84 (1989).
10. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum
    Information*, 10th anniversary ed. (Cambridge University Press, 2010).
11. A. W. Harrow, A. Hassidim, and S. Lloyd, "Quantum algorithm for linear
    systems of equations," Phys. Rev. Lett. 103, 150502 (2009).
12. R. P. Feynman, "Simulating physics with computers," Int. J. Theor.
    Phys. 21, 467-488 (1982).
13. E. Farhi, J. Goldstone, and S. Gutmann, "A quantum approximate
    optimization algorithm," arXiv:1411.4028 (2014).
14. J. Preskill, "Quantum computing in the NISQ era and beyond," Quantum 2,
    79 (2018).
15. A. Lucas, "Ising formulations of many NP problems," Front. Phys. 2, 5
    (2014).
16. J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven,
    "Barren plateaus in quantum neural network training landscapes," Nat.
    Commun. 9, 4812 (2018).
17. I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning* (MIT Press,
    2016).
18. M. J. D. Powell, "A direct search optimization method that models the
    objective and constraint functions by linear interpolation," in
    *Advances in Optimization and Numerical Analysis*, ed. S. Gomez and
    J.-P. Hennart (Springer, 1994), pp. 51-67.
19. T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A
    next-generation hyperparameter optimization framework," in Proc. 25th
    ACM SIGKDD (2019), pp. 2623-2631.
20. Qiskit contributors, "Qiskit: An open-source framework for quantum
    computing," https://github.com/Qiskit/qiskit (2024).
21. C. R. Harris et al., "Array programming with NumPy," Nature 585,
    357-362 (2020).
22. P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific
    computing in Python," Nat. Methods 17, 261-272 (2020).
23. A. J. Chorin, "Numerical solution of the Navier-Stokes equations,"
    Math. Comput. 22, 745-762 (1968).
24. S. A. Orszag and C. M. Tang, "Small-scale structure of
    two-dimensional magnetohydrodynamic turbulence," J. Fluid Mech. 90,
    129-143 (1979).
25. B. Fryxell et al., "FLASH: An adaptive mesh hydrodynamics code for
    modeling astrophysical thermonuclear flashes," Astrophys. J. Suppl.
    Ser. 131, 273-334 (2000).
26. J. M. Stone, T. A. Gardiner, P. Teuben, J. F. Hawley, and J. B. Simon,
    "Athena: a new code for astrophysical MHD," Astrophys. J. Suppl. Ser.
    178, 137-177 (2008).
27. A. P. Solon et al., "Pressure is not a state function for generic
    active fluids," Phys. Rev. E 92, 062111 (2015).
28. R. Löhner, "An adaptive finite element scheme for transient problems
    in CFD," Comput. Methods Appl. Mech. Eng. 61, 323-338 (1987).
29. A. Okubo, "Horizontal dispersion of floatable particles in the
    vicinity of velocity singularities such as convergences," Deep-Sea
    Res. 17, 445-454 (1970).
30. J. Weiss, "The dynamics of enstrophy transfer in two-dimensional
    hydrodynamics," Physica D 48, 273-294 (1991).
31. A. Mignone, P. Rossi, G. Bodo, A. Ferrari, and S. Massaglia, "PLUTO: A
    numerical code for computational astrophysics," Astrophys. J. Suppl.
    Ser. 170, 228-242 (2007).

## Appendix A: Hamiltonian architecture

This appendix carries over the Hamiltonian design from the originating
report (Presentation), at the level of detail needed to reproduce or extend
it. It describes V1, the mapper of the deployed pipeline; Section 2.3
summarises both mappers and Appendix B describes V2.

At high Reynolds number, MHD dynamics arise from a small number of
physically distinct anomaly types: shear flows, vortices, magnetic
reconnection, and helical kink modes. Each has a characteristic spatial
signature, and the Hamiltonian assigns one term per signature. Classical AMR
detects an instability only once it has already produced a large gradient;
the design intent behind Q-HAS was to encode the structure of the plasma
state at time t to anticipate which regions will require refinement, rather
than to react after the fact.

### A.1 Design principle: a decoupled weight x topology x scale x signal product

Every coefficient in the Hamiltonian is a product of three independent
factors, not a single tuned number:

```
Coefficient = Weight x g(topology) x f(scale) x T_rcf(signal)
```

This separates three things the design treats as independent: the magnitude
of the anomaly (T_rcf, a threshold-relative contrast filter), global
thermodynamic scaling (f, a normal-critical gate), and local topology (g, a
leaky sigmoid gate). The ZZ and ZZZZ interaction weights are fixed at 2 and
1 respectively and are not tunable. The single-qubit Z bias instead uses an
adaptive weight, alpha_z = w_z_frac x median(|C|, |K|), where w_z_frac is a
free parameter and the median is taken over the ZZ coefficients C and ZZZZ
coefficients K elsewhere in the same Hamiltonian instance. This keeps the Z
term subordinate to the spatial-correlation terms while breaking a
degeneracy that would otherwise appear: without it, the ferromagnetic
ZZ/ZZZZ coupling makes the all-0 and all-1 states exactly degenerate.

All normalization uses fixed physical constants (e.g. Re_crit = 1,
Rm_crit = 1), never a relative in-domain normalization such as dividing by
the maximum value present in one snapshot, so a coefficient means the same
thing at every grid resolution and simulation time.

**Trainable parameters, then and now.** The originating report lists five
trainable hyperparameters — the encoding steepness (beta), the
uncertainty-band width (sigma), the two per-term contrast sensitivities
(beta_curl, beta_xpoint) and the Z-bias fraction (w_z_frac) — and freezes
gamma_hydro = 2.0, gamma_mag = 0.5 and kappa = 10.0, with threshold_amr =
0.1496 taken from the classical training. This matches what the frozen
historical campaign did: its quantum Optuna database sampled exactly those
five parameters (the classical one sampled only the threshold), and
`PHASE1_SEED_GRID` in `src/train_hyperparams.py` (beta 0.7, sigma 0.10,
beta_curl 0.1, beta_xpoint 0.1, w_z_frac 500, with gamma_hydro 2.0,
gamma_mag 0.5, kappa 10.0) is the starting point of that campaign, not its
result. The current search space is wider: nine free parameters, the eight
above plus the percentile of the relative threshold introduced with the
relative criterion (Section 4.2d), threshold_amr still fixed (verified with
`python src/train_hyperparams.py --print-space`). The hyperparameter file
the pipeline deploys corresponds to neither: it carries gamma_hydro 2.127,
gamma_mag 2.361 and kappa 14.332, which no database ever sampled, and omits
sigma and the relative percentile (Section 8, D-22). The V1 measurements of
`study/` do not depend on that file: the loader of
`study/pipeline/config.py` rejects an incomplete file and falls back, with
a warning, on the full set of reference values it carries. No campaign was
run under either search space (Section 8).

**Where the current code differs from this description.** This appendix
reproduces the design of the originating report. The current V1 differs in
four measured ways, each explained in Section 4.2: the curl and divergence
operators follow the grid's axis convention (4.2a); the mesh thresholds are
no longer purely absolute but min(absolute threshold, percentile of the
signal), because the absolute thresholds scale as 1/dx² and silenced the
four-body terms at the training resolution (4.2d) — so the statement above
that all normalisation uses fixed physical constants no longer holds for
the thresholds; the magnetic gate receives the current in physical units
(4.2e); and the X-point term has lost its f_mag gate and uses √(−det ∇B)
(compare A.6 with 4.2c). Finally, no code path reads the most probable bit
string as A.8 describes: the deployed solver compares the average of each
cell's two marginal probabilities with the AMR threshold, and the study
code takes a majority vote on each qubit (Section 2.4).

### A.2 The structural and mixer Hamiltonians

QAOA is used for two reasons: the flux-graph formulation of instability
detection maps onto the combinatorial graph-optimization problem class QAOA
targets, and the Hamiltonian's Ising-model structure needs no
diagonalization to define its phase operator, which is what makes QAOA
computationally natural for it. QAOA alternates a structural (cost)
Hamiltonian, which imprints a phase on each basis state proportional to its
energy, with a mixer Hamiltonian, which drives transitions between basis
states; over p layers the circuit is intended to converge toward the
structural Hamiltonian's ground state.

The structural Hamiltonian has four terms:

```
H_struct = sum_i h_i Z_i                        Activity Bias         (adaptive weight alpha_z)
         + sum_<i,j> C_ij Z_i Z_j                Gradient Coupling     (weight 2)
         + sum_p K_p (product_{l in p} Z_l)      Circulation Plaquette (weight 1)
         + sum_p K_xpoint,p (product_{l in p} Z_l)  X-point Reconnection  (weight 1, optional)
```

and the mixer is H_mixer = product_i X_i.

The X-point term is a 4-body plaquette operator, enabled by a separate flag
and detailed in A.6. A fifth term, for kink modes, is defined in the
theoretical formulation but disabled in the 2D implementation used
throughout this report: kink modes require helical deformation along a
third (toroidal) axis, so the term has no 2D analogue, and it involves a
Dzyaloshinskii-Moriya XY - YX interaction that is not diagonal in the Z
basis QAOA uses, which would also make it more expensive to implement than
the four active terms.

### A.3 Activity bias — the validity sensor

This single-body term biases each qubit by whether its classical
instability score exceeds the AMR threshold:

```
h_i = alpha_z . (s_i - threshold_amr),   alpha_z = w_z_frac x median(|C|, |K|)
```

s_i is the same classical multi-indicator score used by the classical
baseline (Section 2.5). When s_i exceeds the threshold, h_i is positive and
biases the qubit toward |1> (refine); when it is below, h_i is negative and
biases toward |0> (do not refine). Because the qubit's initial amplitude
(A.8) is set from the same score s, the bias and the initial state agree by
construction, and the QAOA interaction terms can only move probability near
the threshold, where the classical score is genuinely uncertain.

### A.4 Gradient coupling — the boundary sensor

This 2-body term detects spatial discontinuities between neighboring edges
— a strong gradient marks a boundary (shear layer, shock front, vortex
edge) that needs refinement:

```
C_ij = 2 . g_strain(Q_OW) . || ( f_hydro(Re).T_rcf(delta_v), f_mag(Rm).T_rcf(delta_B) ) ||
         . exp( -((s_bar_ij - threshold_amr) / sigma)^2 )
```

s_bar_ij is the average classical score of the two cells the edge connects,
the leading 2 is the fixed (non-trainable) ZZ weight, and:

- g_strain is a leaky sigmoid gate on the Okubo-Weiss Q-criterion,
  g_strain(Q) = 1 / (1 + exp(-kappa . Q / Q_crit)); positive Q marks
  strain-dominated regions, and kappa (trainable) sets the transition's
  steepness.
- f is a normal-critical gate: f(x, x_crit, gamma) = x / x_crit below the
  critical value, and 1 + gamma . ln(x / x_crit) above it, applied
  separately to the hydrodynamic (Re) and magnetic (Rm) channels. The
  logarithmic branch bounds growth (Re = 3000, gamma = 2 gives f ≈ 17, not
  infinity), so one extreme edge cannot dominate the Hamiltonian.
- T_rcf is a threshold-relative contrast filter:
  T_rcf(val, val_crit, beta) = beta . max(0, val/val_crit - 1). Unlike a
  Michelson-style contrast, which vanishes once the whole domain is active,
  T_rcf compares against a fixed physical threshold, so the signal survives
  even when the whole domain is already disturbed.

The trailing Gaussian concentrates the ZZ coupling near the decision
boundary (s_bar ≈ threshold_amr), where the classical score is least
trustworthy; sigma (trainable, in [0.02, 0.30]) sets its width. Away from
the boundary the Gaussian suppresses C_ij exponentially and the bias term
alone drives the decision, which also keeps circuit depth down. For the ZZ
term specifically, T_rcf's own sensitivity is fixed at beta = 1.0 — the
effective sensitivity comes from sigma instead.

### A.5 Circulation plaquette — the vortex and current sensor

Vorticity and current density are both detected as 4-body plaquette
interactions, implementing a discrete Stokes theorem: the circulation
around a closed loop measures the curl it encloses. Horizontal edges carry
v_x and B_x; vertical edges carry v_y and B_y, giving two independent
discrete curls:

```
omega_z(i,j)   = v_x(i,j) - v_x(i,j+1) + v_y(i+1,j) - v_y(i,j)   (discrete vorticity)
J_z,curl(i,j)  = B_x(i,j) - B_x(i,j+1) + B_y(i+1,j) - B_y(i,j)   (discrete current density)
```

each approximating a continuum curl times a cell area (partial_x v_y -
partial_y v_x for omega_z, Ampere's law partial_x B_y - partial_y B_x for
J_z,curl). The plaquette coefficient follows the same f x g architecture as
the gradient coupling, but with independent gates for the fluid and
magnetic channels:

```
K_p = || ( g_rot(Q_OW).f_hydro(Re).T_rcf(omega_z, omega_z_crit, beta_curl),
           g_mag(|J_z|).f_mag(Rm).T_rcf(J_z_curl, J_z_crit, beta_curl) ) ||
```

with omega_z_crit = Re_crit . nu / dx^2 and J_z_crit = Rm_crit . eta / dx^2.
g_rot(Q) = 1 / (1 + exp(+kappa . Q/Q_crit)) activates in rotation-dominated
regions (Q_OW < 0), the complement of g_strain; g_mag(|J_z|) =
1 / (1 + exp(-kappa . (|J_z|/J_crit - 1))) activates once the current
density crosses its own critical threshold. Because the two branches are
gated independently, a vortex without a current sheet only activates the
g_rot branch, and a current sheet without vorticity only activates g_mag; a
uniform or irrotational region gives K_p = 0.

### A.6 X-point reconnection — optional, advanced anomalies

Magnetic reconnection at X-points (hyperbolic null points of the magnetic
field) is detected by a second 4-body plaquette term, gated on the
determinant of the magnetic field's Jacobian:

```
K_xpoint = f_mag(Rm) . T_rcf( max(0, -det(grad B)), val_crit, beta_xpoint )
det(grad B) = (partial_x B_x)(partial_y B_y) - (partial_x B_y)(partial_y B_x)
```

At an X-point the field lines form a hyperbolic pattern and det(grad B) < 0;
the max(0, -det(grad B)) factor keeps only topologically hyperbolic
regions. val_crit = (Rm_crit . eta / (dx . B_0))^2. This term is
self-limiting — it activates only where the local magnetic topology is
genuinely hyperbolic, without needing a separate topological gate — and is
enabled by a dedicated flag rather than always active.

### A.7 Summary of anomaly mapping

| Anomaly | Physical origin | Term | Operator | Weight |
|---|---|---|---|---|
| Activity bias | alpha_z . (s_i - threshold_amr) | Z bias | Z_i | adaptive (alpha_z) |
| Shear / gradient | g_strain x f x T_rcf x Gaussian(sigma) | ZZ coupling | Z_i Z_j | 2 (fixed) |
| Vortex / current | omega_z, J_z,curl (Stokes) | ZZZZ plaquette | loop product of Z_k | 1 (fixed) |
| X-point reconnection | f_mag x T_rcf(-det(grad B)) | ZZZZ plaquette | loop product of Z_k | 1 (optional) |
| Kink (3D only) | J . B | DM interaction | X_i Y_j - Y_i X_j | disabled |

### A.8 State encoding and the variational ansatz

Each qubit is mapped to a point on the Bloch sphere, (theta, phi), and the
two angles are given deliberately different physical roles.

theta is derived from the classical score s, not from the raw stress flux:

```
theta_ij = 2 . arcsin(sqrt(s_ij))   so that   P(|1>) = sin^2(theta_ij/2) = s_ij
```

This gives every qubit exactly the classical detector's own flagging
probability before any QAOA layer runs, so the circuit refines the
classical baseline rather than starting from scratch.

phi carries the temporal-derivative phase that Section 2.3 above refers to
as psi — the same quantity; this appendix keeps the originating report's
own symbol, phi, to match its formulas:

```
phi_ij(t) = (pi/2) . tanh( beta . (Phi_ij(t) - Phi_ij(t - dt_hybrid)) / mean(|delta_Phi|) )
```

Phi_ij is the local stress flux behind the classical score, dt_hybrid is the
interval between VQA updates, beta is the same trainable encoding-steepness
parameter as A.1, and mean(|delta_Phi|) is the mean absolute flux change
over the whole domain, used as a dimensionless normalization. This maps phi
into [-pi/2, +pi/2]: phi ≈ +pi/2 for a locally growing instability, phi ≈ 0
for background evolution, phi ≈ -pi/2 for damping.

Each edge qubit is initialized as |q_ij> = cos(theta_ij/2)|0> +
exp(i.phi_ij) sin(theta_ij/2)|1>, and the full circuit starts from the
product state |psi_0> = (tensor product over all edges <i,j>) |q_ij> — a
warm start, not a uniform superposition. This constrains the search to the
neighborhood of the classical solution, which reduces the QAOA iterations
needed and mitigates the barren-plateau problem common to randomly
initialized variational circuits.

For depth p, the trial state is:

```
|psi(Omega, Gamma)> = product_{k=1}^{p} ( U_mixer(Omega_k) . U_struct(Gamma_k) ) |psi_0>
```

with U_struct(Gamma_k) = exp(-i . Gamma_k . H_struct) embedding the MHD
constraints into each basis state's phase, and U_mixer(Omega_k) =
exp(-i . Omega_k . H_mixer) = product_i exp(-i . Omega_k . X_i) generating
the spin flips that let interference amplify low-energy states. The
variational parameters Gamma, Omega are optimized by a classical COBYLA
loop to minimize E = <psi|H_struct|psi>; the resulting state's most
probable bitstring is read out as the refinement decision.

This is the architecture tested in Section 5: the Hamiltonian-exactness
results (H0a, Section 5.1) ask whether QAOA's evolution actually reaches
this H_struct's ground state; the decision-quality results (H0b,
Section 5.2) ask whether that ground state, when reached, is the better
refinement decision; and the ablations (H3, Section 5.3) turn the
ZZ and ZZZZ terms above off one at a time to ask whether the coupling
structure earns its added circuit depth.

## Appendix B: The parameter-free mapper V2

V2 (`src/Simulation/HamiltParams_v2.py`) was written as a control for V1:
the same three families of terms, with no tunable weight, no gate and no
uncertainty window. It is the mapper of the confirmatory replication
(Section 5.4) and of most of `study/`.

**Current form** (normalisation `max`, the default since 21 August, and
joint normalisation of the plaquette since 22 August). With
ω̂ = |ω|/max|ω|, Ĵ = |J|/max|J| and X̂ = max(0, −det ∇B)/max(...), each set
to zero when its peak does not exceed a round-off floor (128 × machine
epsilon × the largest input field):

```
C_ij       = -W_ZZ   * |jump_ij| / max|jump|          W_ZZ   = 2
K_p        = -W_ZZZZ * (w^ + J^) / max(w^ + J^ + X^)  W_ZZZZ = 1
K_xpoint,p = -W_ZZZZ * X^        / max(w^ + J^ + X^)
h_i        = +C_BIAS * max(|C|, |K|) * (s_i - threshold)   C_BIAS = 0.1
jump_ij    = sqrt(dvx^2 + dvy^2 + dBx^2 + dBy^2) across the edge
```

Properties, each pinned by a test: the largest ZZ coefficient equals W_ZZ
and the largest effective ZZZZ coefficient (plaquette plus X-point, which
sit on the same four qubits) equals W_ZZZZ on any non-uniform field, so the
relative weight of the families is the design ratio 2.000 on every
scenario; the bias-to-coupling ratio is exactly C_BIAS, independent of
`dim` (identical to 10⁻¹² at `dim` = 4, 8, 16, 32); multiplying v and B by
10 changes nothing to 4.8 × 10⁻¹⁶; changing dx from 1.0 to 0.001 changes
nothing bit for bit; a uniform field gives no coefficient at all.

**History.** Until 21 August (`norm = "legacy"`, kept only to reproduce
frozen artifacts): C was normalised by the mean jump, so max|C| varied with
the intermittency of the snapshot (3.12 to 8.10 against a fixed
plaquette), K by max|ω| + max|J| taken at two different points, so the
weaker structure vanished (Section 4.2g) and the ZZZZ family was
devalued by up to 26% on the rotor, and h by the median of the couplings.
The H0a/H0b panels of 9 and 16 August (Tables 1-2) used the legacy form;
the coupling ablation of 29 August and the confirmatory run of 11 September
use the current one. At `dim = 2` the exact ground state of V2 is uniform on
39 of 40 snapshots under the legacy form and on 36 of 40 under the current
one.

**What V2 cannot do.** Its coefficients depend only on the relative shape
of the fields, never on their scale or on the Reynolds number, so they
cannot distinguish a viscous from an inertial flow, and a strong and a
weak structure of the same shape receive the same coefficient. The
plaquette is not type-selective (Section 4.2f).

## Appendix C: Index of all results

Every entry of `docs/RESULTS.md`, with the evidence level of Section 2.8
and where it is discussed above. "Engineering" marks a defect whose fix
did not change a published scientific number; such entries are counted in
Section 7 and not detailed.

The findings of this review are the last three rows: the readout and
bias-threshold diagnostic and the recentring plan (entries of
`docs/RESULTS.md`, each with its test), and the synthetic-harness mapper
mismatch, an open defect (D-202 in `docs/DEFAUTS.md`) pinned by a strict
expected-failure test.

| `docs/RESULTS.md` entry | what it established | level | here |
|---|---|---|---|
| Registry of corrected defects (D-1 to D-201) | each defect measured before and after, pinned by a test | A | 4.1, 4.2, 7 |
| T11 — quantum-contribution attribution | all solvers reach the optimum at `dim = 2`; vacuous (uniform optimum) | B | 4.5 |
| T11b — QAOA displacement toward its optimum | progress ≈ 0 at `dim = 2`; reading requalified (constant warm start, D-48) and verdict unstable between runs (D-50) | B/C | 4.5 |
| T13 — causal ablation at `dim = 2` | couplings change no decision; explained by degeneracy | B | 4.5 |
| T12 — equivariance | solver symmetries; classical map nearly equivariant; ground-state map not judgeable | B | 4.1 |
| T14 — solver validation | first-order time convergence; div B at machine precision | A | 4.1 |
| N = 256 confirmation | T11-T14 conclusions hold at production resolution | B | 4.5 |
| T15, T15b, fold `kh` | closed-loop folds; budget-matched reversal | D | 6.2 |
| T17 — uncertainty window | window attenuates, does not annihilate ZZ; earlier explanation retracted | B | 4.5 |
| T18 — counterfactual without window | couplings inert at `dim = 2` with or without window | B | 4.5 |
| T19/T20/T21, D-92 | Q-HAS arm non-deterministic; endpoint well posed; single-draw ratios retracted | D | 6.2 |
| D13, T22, T22b, T22d, T22 leak-free | leak removed; transfer; distance to near-full refinement | D | 6.2, 6.3 |
| Trap sweep; fresh-eyes review | silent failures; `phys_score` is instability-weighted; decision cost excluded | — | 2.1, 6.2, 7 |
| T25 — physics robustness | no physics seed exists; direction not established on new initial conditions | D | 6.3 |
| T26 — size scan | couplings inert only at `dim = 2`; they lower F1 when active | B (dim 2, 4, 8) | 5.3 |
| Closing the closed-loop study | one-sentence result, conditions, what would overturn it | D | 6.2 |
| V1 test suite re-armed; eight stages that could not fail | original tests contained the falsification; zero-assertion stages | A (method) | 5.6, 7 |
| V1 no longer fabricates a Hamiltonian | calm fields build none; placeholder removed | A | 4.2h |
| T31 — axis convention of the mappers | indicators measured strain; correction gives no measurable gain | A | 4.2a |
| The QAOA arm is sampled | per-call range 0.18-0.36; ranks stable (0.933) | A/C | 4.4 |
| D-2 — AMR prolongation | 2.49 × 10⁻¹ → 7.74 × 10⁻⁶ | A | 4.1 |
| Contract audit D-11 to D-14 | flux diode on shear; V2 dimensionless; halo edges; field/score mismatch | A | 4.2, 5.6, 6.4 |
| What the circuit can move | diagonal cost layer; mixer bound; mixer alone 0.254, Hamiltonian adds 0.236 | A | 4.4 |
| D-16 — overlapping patch list | up to 25% of the domain counted twice | A | 4.1 |
| QAOA suite verdict | plaquette dead on a vortex before D-1 (×22.7) | A/C | 4.2b |
| D-17/D-18 | convention defect outside `src/`; KH perturbation-energy diagnostic 99.98% base flow | A | 4.2a |
| D-19/D-20 | unknown backend; ansatz cache sharing two Hamiltonians | engineering | 7 |
| Audit of the physical gates | g_strain + g_rot = 1; two reparametrised factors | A | 4.2f |
| D-21 — flux downsampling | peak kept 38-70% → 100% | A | 5.6 |
| D-22 — provenance of the hyperparameters | deployed values have no reproducible origin | A | 8 |
| Strang splitting; D-24; Van Kan; D-25; D-27 | order 1 explained; RHS projection order 4 (disabled); B not projected; solenoidal initial perturbations | A (Van Kan: C) | 4.1 |
| D-29 to D-36 — training script | duplicated scenarios, unread parameter, decorative pruning, budget × workers | engineering | 8 |
| D-37 — bias and couplings on two grids | 41% error at depth > 0, Q-HAS arm only | A | 4.1, 6.1 |
| D-38, D-48, D-49, D-50, D-136, D-116, D-68, D-10 | guards, hardware mode on simulator, exit codes, launchers, figure axes | engineering | 2.4, 7 |
| D-66/D-67 — the full launch computed nothing | CLI loop never ran; after the fix, one unretuned run: classical better on 4 of 5 fields at equal cost (single run, not a comparison) | engineering | — |
| D-91 — rotor bench ground truth | relative error per block; after fix all three selections coincide (bench validates the chain, does not separate the arms) | A | — |
| K_xpoint gate removed; four coefficient families; dimensional correction | X-point detector zero at the null; dx⁴ → dx² | A | 4.2c |
| The criterion becomes relative; D-117 | four-body terms alive at N = 256; ninth parameter | A | 4.2d |
| Do the coefficients point where refinement is needed? | Spearman with true block error | A (Harris, current) / B (four scenarios) | 4.3 |
| Balance between families; fluid/magnetic imbalance fixed | magnetic gate in grid units; 27 500 → 0.44 | A | 4.2e |
| `study/` sees the X-point term; thresholds aligned | study and circuit identical to 5.3 × 10⁻¹⁵ | A | 2.4, 5.1 |
| `dim = 3` relaunch on the corrected Hamiltonian; "the main defect is the Hamiltonian"; ρ criterion; D-172 | H0a and H0b on V2 | B (legacy V2 normalisation; current V2 differs, Table 3) | 5.1, 5.2 |
| Coefficient preflight | five checks | A | 4.2h, 4.3 |
| D-53 — the certified size contradicts the refutation of H0 | H0a/H0b separate | A (separation); V2 values B | 5.1, 5.2 |
| D-51, D-59 | X-point term reaches `study/`; duplicated ZZ link at `dim = 2` (no published number moves) | A | 5.1 |
| D-47 — degeneracy at `dim = 2` | exact optimum "refine everything" on 40 of 40 | A | 4.5 |
| Campaign database naming, ghost trials | 298 abandoned trials counted | engineering | — |
| Normalisation independent of `dim` (`norm = "max"`) | bias/coupling ratio exactly C_BIAS | A | App. B |
| Cone curve | neighbour features help a GBT; average not citable | A (with stated limits) | 5.3 |
| Corpus in two curl conventions | frozen artifacts at N = 256/64 vs current at N = 96 | caution | 2.2 |
| `dim` relative to N | label zero at p = 1; rule `dim ≤ N/8` | A | 2.2 |
| The Harris zero is a transferred threshold | AUC 0.908; matched-budget F1 0.659 | A | 5.3 |
| Selectivity of the coefficients; split plaquette (withdrawn) | only K_xpoint type-selective | instrument property | 4.2f |
| Half of the ZZZZ term dead on 2 of 4 scenarios; `norm = max` default; D-86; D-190 | normalisation corrections; bias-only plateau | A (D-86 value: B) | 4.2g, 5.3 |
| Dynamic ground truth; D-188 | redundant at δt = 0.1; mixed at t_x | A | 5.7 |
| D-192, D-158, D-194, D-98, D-100, D-50, D-191, D-187, D-189 | provenance restored, aggregator, launchers, figures, seeds, round-off floor | engineering | 2.4, 2.8 |
| D-39 — tearing diagnostic | reconnection on 6 of 6 trajectories | A | 4.1 |
| D-22 — campaign result reaches `study/` | deployment path fixed | engineering | 8 |
| D-195 | two symptoms explained as H0a | A | 5.1 |
| D-196 | master-table pin 4 → 6 DIFF | A | 2.8 |
| H1/H3/H4 implementation audit | wiring fixed; H4 not answerable (4 of 8 folds) | A | 6.3 |
| T5 — ψ feature under LOSO | ψ lowers F1; GBT ceiling comparison not admissible (D-198) | A | 5.4, 5.6 |
| Train/validation redesign; training diversification | implemented and tested, never run | — | 8 |
| D-200 | H0b on V1 | A | 5.1, 5.2 |
| Static and dynamic synthetic models | ceilings stable; QAOA-vs-exact comparisons use two mappers (found in this review) | A (ceilings) / invalid as replication | 5.5 |
| V2 against GBT, Re = 400, n = 5 | superseded at larger scale | superseded | 5.4 |
| V2 against GBT, multi-Re with bootstrap | confirmatory replication | A | 5.4 |
| D-92 bis | synthesis script reused retracted single-draw ratios; fixed | A | 7 |
| Lecture du QAOA et seuil du biais (review, 5 Oct) | study reads QAOA by majority, deployed solver against the AMR threshold; bias centred at 0.15, F1-optimal 0.51-0.60; QAOA F1 = threshold at 0.5 on 11 of 12 V1 instances | A | 2.4, 5.2 |
| Recentrage du biais (review, 5 Oct) | V1 panel replayed bit for bit; recentring never turns ρ negative; exact optimum never better than the classical rule; V2 optimum uniform on 24 of 24 | A | 5.2 |
| D-202 (open) — synthetic harnesses on two mappers | QAOA on V1, exact optimum on V2; synthetic H0a/H0b numbers are not replications | open, pinned | 5.5 |
