# V4 results log — experimental answer to the scientific audit

One entry per study. V4 answers the audit's missing proof blocks that are
executable without the (still pending) closed-loop Level-3 campaign:
**quantum attribution** (P0), **confirmatory statistics** (P0),
**equivariance** (P1), **causal term ablations** (P1) and **numerical
validation** (P1).

Continuity rule: no V1/V2/V3 symbol is redefined. Everything reusable is
imported — `MHDSolver`, `build_patch_hamiltonian`, `build_ising_terms`,
`sa_multi_restart`, `spins_to_decisions`, `prepare_qaoa_inputs`,
`run_qaoa_on_snapshot`, `div_B`, `total_energy`, `downsample_fields`,
`bootstrap_by_trajectory`, `git_commit_hash`.

Test gate: **147 pytests pass** (118 v3 + 29 v4).

**All studies were re-run at production resolution N=256** (4 scenarios,
Re=400, DNS regenerated here with `phase1_dns_sweep.run_dns`). Both the
N=64 exploratory pass and the N=256 confirmation are reported; every
qualitative conclusion is identical at the two resolutions. Section
"N=256 confirmation" at the end gives the side-by-side table.

---

## T11 — Quantum-contribution attribution (audit P0)

`study/v4/t11_solver_attribution.py --N 64 --dim 2 --n-snaps 2`

At the **deployed size** (`VQA_N = 2` → 8 qubits, periodic root scan, i.e.
exactly the configuration `refinement.py` solves at depth 0).

| solver | hit optimum | E gap | spin agreement | mask match | wall (s) |
|---|---|---|---|---|---|
| exhaustive (certified) | 1.000 | 0 | 1.000 | 1.000 | 0.000 |
| simulated annealing | 1.000 | 0 | 1.000 | 1.000 | 0.121 |
| SA warm-started | 1.000 | 0 | 1.000 | 1.000 | 0.123 |
| greedy local search | 1.000 | 0 | 1.000 | 1.000 | 0.000 |
| classical decision alone | 1.000 | 0 | 1.000 | 1.000 | 0.000 |
| QAOA p=1 (statevector) | 1.000 | 0 | 1.000 | 1.000 | 0.414 |
| QAOA p=2 (statevector) | 1.000 | 0 | 1.000 | 1.000 | 0.612 |
| QAOA p=2, 4096 shots | 1.000 | 0 | 1.000 | 1.000 | 0.617 |

- The cost Hamiltonian is **diagonal** (Z/ZZ/ZZZZ only), verified at runtime
  by `is_diagonal_cost_hamiltonian` on every snapshot. Its ground state is a
  computational basis state, so "exact diagonalisation" reduces to a
  classical enumeration of 2^8 = 256 configurations.
- Every solver reaches the certified optimum and returns the same mask.
  **Pre-registered rule fires: quantum optimisation is not the source of any
  gain.** A closed-loop improvement would attribute value to the
  Hamiltonian, not to its quantum optimiser.

**Caveat that makes the agreement partly vacuous** (see T11b): the optimum
itself is uniform, so the solvers agree on a trivial problem.

---

## T11b — Does the QAOA optimise its own Hamiltonian? (audit P0)

`study/v4/t11b_qaoa_displacement.py --N 64 --dim 2 --reps 1 2 3 4`

Position of three points in marginal space: `m_theta` (amplitude encoding of
the classical score alone), `m_qaoa` (optimised circuit), `m_gs` (exact
ground state). `progress` = projection of the realised displacement on the
required one; 0 = decision unchanged, 1 = optimum reached.

| reps | progress | ‖displacement‖ | ‖required‖ | ‖remaining‖ | mean marginal |
|---|---|---|---|---|---|
| 1 | +0.0590 | 0.1276 | 0.8381 | 0.8010 | 0.7217 |
| 2 | +0.0563 | 0.1178 | 0.8381 | 0.8030 | 0.7205 |
| 3 | −0.0298 | 0.1178 | 0.8381 | 0.8536 | 0.7044 |
| 4 | −0.0584 | 0.1883 | 0.8381 | 0.8830 | 0.6980 |

- **The exact ground state is a UNIFORM mask on 100% of snapshots**
  (8/8: 4 scenarios × 2 snapshots) — refine-all, carrying no spatial
  information. Cause (consistent with V3 Task 9): the ferromagnetic
  couplings dominate the Z bias, |C| ≈ 2.0 and |K| = 1.0 against
  |h| ≈ 0.071, a ratio ≈ 28.
- **Mean variational progress = 0.0068** (0.68%). The circuit's displacement
  is essentially orthogonal to the direction of its own optimum.
- Progress **does not increase with depth**; it becomes negative by reps=4
  (−0.117 from reps 1 to 4). Deeper circuits move slightly *away* from the
  optimum of the declared cost.

**Reading.** The deployed decision is not a minimiser of the declared cost
function. It is a ≤4%-in-norm perturbation of the amplitude encoding
θ = 2·arcsin(√score), i.e. of the classical score itself.

---

## T13 — Causal ablation of term families (audit P1)

`study/v4/t13_term_ablation.py --N 64 --dim 2 --n-snaps 2`

Exact ground state recomputed after zeroing each family (control `full`
must change nothing).

| ablation | decisions changed | uniform | refined fraction | F1 | n_optima |
|---|---|---|---|---|---|
| full (control) | **0.000000** | 1.000 | 1.000 | 0.317 | 1.0 |
| no_Z | 1.0000 | 1.000 | 0.000 | 0.000 | 8.0 |
| no_ZZ | **0.0000** | 1.000 | 1.000 | 0.317 | 1.0 |
| no_ZZZZ | **0.0000** | 1.000 | 1.000 | 0.317 | 1.0 |
| Z only (both couplings removed) | **0.0000** | 1.000 | 1.000 | 0.317 | 1.0 |
| couplings only (Z removed) | 1.0000 | 1.000 | 0.000 | 0.000 | 8.0 |

- Removing **all** ZZ and **all** ZZZZ couplings changes **no decision**.
  The single-site Z bias alone reproduces the full-Hamiltonian decision
  exactly.
- Removing the Z bias destroys the decision entirely and leaves an
  8-fold degenerate ferromagnet.
- The control is exactly 0, which validates the measurement chain.

**Reading.** At the deployed grid size the coupling terms — the entire
motivation for an Ising/quantum formulation — are **causally inert**. This
is a causal statement, unlike the post-hoc ZZ/ZZZZ attributions of the
manuscript.

---

## T12 — Equivariance and orbit error (audit P1)

`study/v4/t12_equivariance.py` (dim=2 exact; dim=8 with annealed ground
state and a mandatory reproducibility control).

Step 1 — the transformation must be a symmetry of the discrete solver:
`eps = ‖T(step(U)) − step(T(U))‖ / ‖step(U)‖`.

| op | eps (N=64) |
|---|---|
| rot180 | 2.8e-16 (machine precision — exact symmetry) |
| flip0 / flip1 / rot90 | 7.8e-6 |

Step 2 — orbit error of the decision map, dim=8 (structured masks):

| op | classical route | ground-state route |
|---|---|---|
| flip0 | 0.0195 | 0.3984 |
| flip1 | 0.0508 | 0.3555 |
| rot180 | 0.0547 | 0.3359 |
| rot90 | 0.0508 | 0.3047 |
| **mean** | **0.0439** | **0.3486** |

Step 3 — **mandatory control** (`solver_noise_floor`): disagreement of the
ground-state route between annealing seeds **on the same, untransformed
field** = **0.2676**, with the refined fraction swinging by 0.15 across
seeds.

- The classical score map is **nearly equivariant** (4.4% orbit error,
  deterministic, floor = 0). The residual is attributable to the one-sided
  finite differences used in the indicator.
- The ground-state route's 0.349 orbit error is **not interpretable as
  non-equivariance**: the annealed optimiser is itself irreproducible at a
  comparable magnitude (floor 0.268). The verdict printed by the script
  requires a 2× margin over the floor, which is not met.
- At dim=2 with exact enumeration, orbit error is exactly 0 for all routes —
  but only because the mask is uniform, so the test is vacuous there.

**Reading.** What this establishes is not an equivariance defect but a
**degeneracy defect**: at dim=8 the objective is flat enough that two
annealing seeds disagree on 14–37% of patches. A decision defined as
"the ground state" is not well posed at that size.

---

## T14 — Numerical validation of the V1 solver (audit P1)

`study/v4/t14_numerical_validation.py`

**(A) Self-convergence**, all solutions restricted to the coarsest grid:

| grids | ‖u_N − u_2N‖_rel | observed order |
|---|---|---|
| 32 → 64 → 128 (t=0.5) | 7.41e-02, 3.71e-02 | **1.00** |
| 64 → 128 → 256 (t=0.25) | 3.34e-02, 1.67e-02 | **1.00** |

**(B) Conservation and solenoidal constraint** (every trajectory):
energy monotonically decreasing, drop 0.3–1.8%; `max|div B| / rms|B|`
between 5.6e-15 and 8.0e-14 — machine precision.

**(C) Reynolds numbers outside the training grid** {400, 800, 1200, 1600}:
Re = 200 and Re = 3200 both pass (monotone energy, div B ≈ 1.5e-14).

**(D) Localisation of the first-order behaviour** — temporal convergence at
fixed dt, with and without the projection step:

| n_steps | with projection (as in `step_full`) | without projection |
|---|---|---|
| 16 | 3.35e-03 | 3.53e-07 |
| 32 | 1.63e-03 (order 1.04) | 2.22e-08 (order 3.99) |
| 64 | 7.61e-04 (order 1.10) | 1.39e-09 (order 4.00) |
| 128 | 3.26e-04 (order 1.22) | 8.66e-11 (order 4.00) |
| **mean order** | **1.12** | **4.00** |

Direct order test of the spatial operators on a smooth periodic field:
`_fd_grad` and `_fd_laplacian` are **exactly 4th order** (4.00 at every
refinement).

**Reading — see the defect note below.** The spatial stencils and the RK4
kernel are both 4th order, but `step_full` applies a full RK4 step *then*
the divergence-free projection. That Lie splitting is first order and caps
the whole scheme at first order in time; since CFL ties dt to dx, the
space–time self-convergence is first order.

---

## Defect notes for the manuscript

**D-V4-1 (numerical, material for the methods section).** The paper
describes the solver as "fourth-order finite differences in space, RK4 in
time". Both components are verified 4th order in isolation, but the
*scheme* converges at **order ≈ 1** because the incompressibility
projection is applied as a first-order operator splitting after the
complete RK4 step (`solver.py::step_full`). Isolated, reproducible
diagnostic in T14(D). This does not invalidate the comparisons — both arms
share the solver, the runs are paired, div B is at machine precision and
all phase-1b invariants pass — but the accuracy statement must be corrected,
and any convergence claim must quote order 1.

**D-V4-2 (modelling, material for the results section).** At the deployed
size the exact ground state of the cost Hamiltonian is uniform (T11b),
the coupling terms are causally inert (T13), and the circuit realises 0.68%
of the displacement toward its own optimum (T11b). The Q-HAS decision is
therefore a small perturbation of the classical score encoding rather than
an optimisation outcome. This mechanistically explains the 0.66% composite
gain, the 109 flipped decisions with 45 correct and 64 incorrect, and the
mask asymmetry, without invoking any quantum effect.

**D-V4-3 (methodological).** A "ground state" obtained by annealing at
dim ≥ 4 is not reproducible across seeds (14–37% of patches, T12 control).
Any statement about ground-state decisions above 8 qubits requires that
floor to be reported alongside.


---

## N=256 confirmation (production resolution)

Command set: `logs/v4/v4_N256.log`. 4 scenarios, Re=400, 12 snapshots for
T11/T13, 8 for T11b, dim=2 (deployed size) and dim=8 (structured masks).

### Every conclusion holds; the numbers sharpen

| quantity | N=64 | **N=256** | verdict |
|---|---|---|---|
| exact ground state uniform | 100% | **100%** | unchanged |
| cost Hamiltonian diagonal | True | **True** (12/12 snapshots) | unchanged |
| solvers reaching certified optimum | all | **all except cold SA** | see below |
| QAOA mask = exact ground state | 1.000 | **1.000** (p=1,2,3 + shots) | unchanged |
| variational progress toward own optimum | 0.0068 | **0.0854** | still ≈ 0 |
| progress change, reps 1 → 4 | −0.117 | **−0.172** | still *decreasing* |
| ablation: remove all ZZ | 0.0000 changed | **0.0000 changed** | unchanged |
| ablation: remove all ZZZZ | 0.0000 changed | **0.0000 changed** | unchanged |
| ablation: remove Z bias | 1.0000 changed | **1.0000 changed** | unchanged |
| classical-route orbit error (dim=8) | 0.0439 | **0.0146** | improves with resolution |
| self-convergence order | 1.00 | **1.00** | unchanged |
| temporal order, projection ON / OFF | 1.12 / 4.00 | **1.12 / 4.00** | unchanged |

### T11 at N=256 — one new observation

| solver | hit optimum | E gap | mask match | F1 | wall (s) |
|---|---|---|---|---|---|
| exhaustive (certified) | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.000 |
| simulated annealing (cold) | **0.583** | 1.41e-02 | 0.583 | 0.367 | 0.139 |
| SA warm-started | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.133 |
| greedy local search | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.000 |
| classical decision alone | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.000 |
| QAOA p=1 / p=2 / p=3 | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.75 / 1.12 / 1.42 |
| QAOA p=3, 4096 shots | 1.000 | −1.4e-17 | 1.000 | 0.389 | 1.603 |

New at N=256: **cold-start simulated annealing misses the certified optimum
on 42% of snapshots** (E gap 1.4e-2) while the warm-started variant, greedy
descent and every QAOA depth reach it exactly. The optimum is trivially
reachable *from the classical decision* but not from a random start — the
landscape is flat with a narrow basin. This strengthens rather than weakens
the attribution conclusion: the only solver that struggles is the one that
does not start from the classical answer.

### T11b at N=256

| reps | progress | ‖disp‖ | ‖required‖ | ‖remaining‖ | mean marginal |
|---|---|---|---|---|---|
| 1 | +0.1588 | 0.1685 | 0.7487 | 0.6504 | 0.7739 |
| 2 | +0.1192 | 0.1569 | 0.7487 | 0.6759 | 0.7653 |
| 3 | +0.0766 | 0.1392 | 0.7487 | 0.7055 | 0.7555 |
| 4 | −0.0132 | 0.1706 | 0.7487 | 0.7662 | 0.7376 |

Mean progress 0.0854, monotonically decreasing with depth and negative by
reps = 4. The ground state is uniform on 100% of snapshots.

### T12 at N=256

dim=8 orbit error: classical route **0.0146** (flip0 0.0078, flip1 0.0156,
rot180 0.0195, rot90 0.0156) — three times smaller than at N=64, consistent
with the one-sided-finite-difference explanation (the defect scales with
grid spacing). Ground-state route 0.4219 against a reproducibility floor of
**0.3613** → the script correctly refuses the interpretation. At dim=2 with
exact enumeration everything is 0 (uniform mask, vacuous).

### T14 at N=256 — the solver order question, settled

Self-convergence on grids 64 → 128 → 256 at t = 0.25: errors 3.344e-02 and
1.673e-02, **observed order 1.00**. Splitting diagnostic run *at N=256*
(`--split-N 256`): with projection order **1.12** (err 3.35e-03 → 3.27e-04),
without projection order **4.00** (err 3.76e-07 → 9.21e-11). Conservation:
energy monotone at every resolution, `max|div B|/rms|B|` ≤ 8.0e-14, and
Re = 200 / 3200 (outside the training grid) both pass.

**The first-order behaviour is not a low-resolution artefact.** It is
identical at N=64 and N=256, and the diagnostic isolates the cause at
production resolution: the Lie splitting between the RK4 step and the
divergence-free projection in `solver.py::step_full`.
