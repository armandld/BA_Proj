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

---

## T15 — Level 3, closed-loop LOSO (audit P0, decisive experiment)

`study/v4/t15_level3_closed_loop.py`

### Status: driver built, code path validated, campaign not yet run

The driver performs a true pipeline-level LOSO fold: for each held-out
instability class it (1) tunes the QAOA hyperparameters with Optuna on the
composite loss of the **other** classes only, reusing V1's own
`make_composite_objective`; (2) tunes the **classical** arm's AMR threshold
on the same training classes via `make_classical_composite_objective`, so
both arms suffer the identical exclusion; (3) runs both arms on the held-out
class with the same DNS trace, hot start, hybrid budget and depth. Endpoints
come from `pipeline(..., return_details=True)`: `phys_score` (relative L2 vs
DNS), `patch_ratio` (compute) and `combined`. Per-fold results are written
incrementally to JSON, so an interrupted campaign resumes.

**End-to-end validation** (`--smoke`, N=64, T_MAX=0.4): the complete path
runs to completion and writes both outputs. Smoke numbers are degenerate by
construction (both arms refine everything, delta = 0) and are not
scientific; the mode exists only to de-risk a day-long run.

### Defect found in the V1 training module (blocking for LOSO)

`TrainHyperParam_v2.SCENARIOS_ALL = SCENARIOS_ISOLATED + SCENARIOS_COMPLEX`
where `SCENARIOS_ISOLATED` already contains `ot` and `rotor` and
`SCENARIOS_COMPLEX` re-adds **the same config objects**. The list therefore
has 6 entries for **4 distinct classes**, and since the composite loss is
`mean(Loss_i)` over the list, OT and rotor are weighted 2/6 each against
1/6 for KH and tearing — an undocumented 2:1 tilt in every Phase-3 training
run. For a LOSO fold the consequence is worse: excluding `ot` would leave
its duplicate in the training list, i.e. **manufacture leakage**.
`fold_scenarios` de-duplicates by key and prints a warning. Related: the
module defines `SCENARIO_VORTEX` and `SCENARIO_COALESCENCE` (lamb_oseen,
island_coalescence) but never uses them, while its own docstring claims
Phase 1 trains on "KH, VORTEX, TEARING, COALESCENCE".

### Measured cost model (N=256, this container)

| stage | measured |
|---|---|
| DNS traces per fold (3 train + 1 held) | 225 s |
| one full `pipeline()` run at N=256 | **≥ 5 min** |
| one Optuna trial = 3 training scenarios | ≈ 15 min |

Per fold ≈ 4 min (DNS) + 15·`n_trials` min (QAOA tuning) + ≈ 6·`n_cls` min
(classical tuning) + 7 min (both final arms).

| `--n-trials` | per fold | 4 folds |
|---|---|---|
| 8 | ≈ 2.6 h | ≈ 10 h |
| 10 | ≈ 3.2 h | ≈ 13 h |
| 12 | ≈ 3.8 h | ≈ 15 h |
| 170 (protocol) | ≈ 43 h | ≈ 7 days |

**Deviation to log when the campaign runs:** the protocol freezes the V1
Optuna budget at 170 trials; a one-day campaign affords 8–12. The script
prints the deviation itself when `--n-trials < 170`. Other standing
deviations: 4 folds (the V1 module exposes 4 distinct classes, not the 8 of
protocol §1.1) and a single physics seed per fold.

### Recommended command for a one-day run

```
nohup python study/v4/t15_level3_closed_loop.py \
      --n-trials 10 --n-trials-classical 5 \
      > logs/v4/level3.log 2>&1 &
```

Resumable: each completed fold is skipped on restart. Monitor with
`grep -E "FOLD|tuning|Q-HAS|classical\]" logs/v4/level3.log`.

### T13 with the **deployed V1 mapper** (N=256, dim=2)

The ablation above used the parameter-free V2 mapper. Re-run with the V1
mapper (`--mapper v1`, the `TRAINED_*` coefficients the pipeline actually
deploys):

| ablation | decisions changed | uniform GS | refined | F1 | n_optima |
|---|---|---|---|---|---|
| full (control) | **0.000000** | 1.000 | 0.750 | 0.333 | 64.8 |
| no_Z | 0.7500 | 1.000 | 0.000 | 0.000 | 88.0 |
| no_ZZ | **0.0000** | 1.000 | 0.750 | 0.333 | 64.8 |
| no_ZZZZ | **0.0000** | 1.000 | 0.750 | 0.333 | 64.8 |
| Z only (both couplings removed) | **0.0000** | 1.000 | 0.750 | 0.333 | 64.8 |

Same conclusion as for V2: the ZZ and ZZZZ families are **causally inert**
for the deployed Hamiltonian. Two V1-specific observations: the ground state
is uniform on 100% of snapshots but is *refine-all* on only 75% of them, and
the V1 cost function is **massively degenerate** — 64.8 of the 256
configurations are optimal on average (88 once the bias is removed).
Inspection of the coefficients explains both: at dim=2 the V1 mapper yields
median |C| = 0 with |h| ≈ 200–240, and on harris_tearing every coefficient
is zero, i.e. an identically null Hamiltonian on which the QAOA has nothing
to optimise.

---

## T15 / T15b — Level 3 closed loop: first fold, and the budget-matched reversal

### T15, fold `ot` (Orszag–Tang excluded from all tuning)

`study/v4/t15_level3_closed_loop.py --folds ot --n-trials 4 --n-trials-classical 2`

| endpoint | Q-HAS | tuned classical | Δ (Q−C) |
|---|---|---|---|
| combined (primary) | 0.3328 | 0.4386 | **−0.1058** |
| phys_score (L2 vs DNS) | 0.1940 | 0.4845 | −0.2905 |
| patch_ratio (compute) | 0.6797 | 0.3238 | **+0.3558** |

Taken at face value this favours Q-HAS on the pre-registered primary
endpoint. **It must not be read that way**, because the two arms are not at
the same point of the error–cost frontier, and the asymmetry is inherited
from the V1 training module:

- `make_composite_objective` (QAOA arm) **hard-codes**
  `HyperParams["threshold_amr"] = 0.14959824837662078` — never suggested to
  Optuna, with the source comment "le meilleur classique";
- `make_classical_composite_objective` (classical arm) optimises
  `trial.suggest_float("threshold_amr", 0.05, 0.8)` freely and selected
  **0.4616** for this fold.

A 3× threshold difference explains the 2.1× compute gap and hence the
fidelity gap. This is exactly the "budget-matched comparison" the audit
demanded, and it is a **third defect** in the comparison design: it applies
to V1's own closed-loop numbers, not only to this fold.

### T15b, budget-matched classical arm (same fold)

`study/v4/t15b_budget_matched.py --fold ot --max-iter 4` — bisection on the
classical threshold to reproduce the Q-HAS compute budget, everything else
(DNS trace, hot start, hybrid budget, depth) held fixed.

Classical error–cost frontier on the held-out class:

| threshold | patch_ratio | phys_score |
|---|---|---|
| 0.0500 | 0.9480 | 0.0111 |
| 0.1438 | 0.7369 | 0.0649 |
| **0.1906** | **0.6412** | **0.0827** |
| 0.2375 | 0.5866 | 0.1027 |
| 0.4250 | 0.3554 | 0.2899 |
| 0.8000 | 0.0156 | 0.5894 |
| *Q-HAS* | *0.6797* | *0.1940* |

**Budget-matched result: Δ phys = +0.1113 in favour of the classical arm.**
At *slightly less* compute (0.6412 vs 0.6797) the classical rule achieves
**2.3× lower** L2 error against DNS (0.0827 vs 0.1940). Q-HAS lies well
above the classical frontier — it is **strictly Pareto-dominated** on this
fold.

Two readings sharpen this further:
- At a *matched threshold* the conclusion is the same: classical at
  thr = 0.1438 gives phys = 0.0649 at patch = 0.7369, while Q-HAS at
  thr = 0.1496 gives phys = 0.1940 at patch = 0.6797 — 3× worse fidelity
  at comparable settings. The gap is therefore not a threshold artefact:
  the QAOA perturbation of the θ encoding actively degrades the decision
  relative to plain thresholding of the same score.
- This is coherent with T11b and T13: the circuit does not optimise its own
  cost (progress ≈ 0, decreasing with depth) and the coupling terms are
  causally inert, so the perturbation it applies carries no useful
  information.

**Pre-registered decision rules (`docs/level3_preregistration.md`).**
P1 (equivalence) is **not** supported on this fold: the arms differ, and
under budget matching the difference is large and favours the classical
rule. P3 (any fidelity gain is paid in compute) is **confirmed and then
some** — the gain does not survive paying for the compute. The
`combined`-endpoint verdict of T15 is superseded by the budget-matched
comparison, which is the interpretable one.

**Scope.** One fold (`ot`), one physics seed, 4 Optuna trials. The campaign
was interrupted twice by container reclamation while running folds `kh`,
`rotor`, `tearing`; those folds remain to be run. No claim of general
closed-loop falsification is made from n = 1. What *is* established is that
the apparent closed-loop advantage of the primary endpoint does not survive
the audit's budget-matched control on the fold measured.

---

## T17 — ZZ uncertainty window: the mechanism behind causal inertness

```
python study/v4/t17_uncertainty_window.py --N 64 --steps 30
```
git hash: see `results/t17_uncertainty_window.npz`  ·  runtime ≈ 1 s
(the four DNS spin-ups dominate; N=64, 30 steps each)

**Why this task exists.** T13 established a *fact*: zeroing the ZZ family
changes 0.0000 decisions. T17 establishes the *mechanism*. The lead came
from V1's own test suite — see defect **D6** below — which contains two
failing tests asserting the opposite.

**Mechanism.** `HamiltParams.compute_coefficients` multiplies the entire ZZ
family by a Gaussian centred on the AMR decision threshold,
`w = exp(-((score - threshold_amr)/sigma)^2)`. The intent is to concentrate
coupling where the classical decision is uncertain. The effect is that the
coupling is removed from exactly the cells where it is largest: strong
gradients produce large `|C|` *and* confident (far-from-threshold) scores.

**Measurements** (four classes × two parameter sets). `no window` is
obtained by setting σ → 1e9 so that `w ≡ 1`; V1 is never modified. Mass
kept = Σ|C|·w / Σ|C|, each edge family paired with its own window.

**Three parameter sets, not two.** There are two distinct "trained" σ, and
conflating them changes the numbers by 100+ orders of magnitude:
`TRAINED_SIGMA` = **0.023** is the open-loop pipeline constant used by
phase5 and therefore by T11/T13/T18; σ = **0.1888** is what Optuna found for
the Level-3 fold `ot`, i.e. closed loop only. The deployed set is read from
the module rather than hard-coded, so it cannot drift from what runs.

Deployed **open-loop** parameters — the configuration behind T11/T13/T18
(σ = 0.023, threshold = 0.1496). This is the harshest case:

| class | max\|C\| no window | mass kept | Spearman(\|C\|,w) |
|---|---|---|---|
| kelvin_helmholtz | 53.92 | 1.319e-02 | −0.372 |
| mhd_rotor | 136.0 | 7.652e-28 | −0.400 |
| orszag_tang | 63.59 | 4.187e-125 | −0.012 (degenerate) |
| harris_tearing | 42.32 | 3.855e-154 | −0.502 |

ZZ is **numerically dead on three of four classes** at the deployed
open-loop setting, and retains 1.3 % on the fourth.

Level-3 **closed-loop** parameters (σ = 0.1888, threshold = 0.1496) — the
most permissive setting, and the one governing the T15 folds:

| class | max\|C\| no window | max\|C\| with window | mass kept | Spearman(\|C\|,w) |
|---|---|---|---|---|
| kelvin_helmholtz | 53.92 | 36.71 | 1.142e-01 | −0.372 |
| harris_tearing | 42.32 | 0.0935 | 1.990e-03 | −0.502 |
| mhd_rotor | 136.0 | 1.331 | 3.951e-04 | −0.460 |
| orszag_tang | 63.59 | 0.6955 | 9.679e-05 | −0.008 (degenerate) |

Parameters of the failing V1 tests (σ = 0.05, threshold = 0):

| class | w_max | max\|C\| with window | mass kept |
|---|---|---|---|
| kelvin_helmholtz | 9.964e-01 | 19.60 | 7.449e-03 |
| harris_tearing | 2.626e-01 | 2.626e-58 | 2.537e-60 |
| mhd_rotor | 1.010e-19 | 9.943e-18 | 9.547e-23 |
| orszag_tang | 4.228e-50 | 1.773e-48 | 1.314e-53 |

**Reading.** Before the window the coupling is healthy on *every* class
(40–136). After it, three of four classes retain under 0.2 % of the
coupling mass, and the best case retains 11.4 %. The rank correlation
between coupling magnitude and window weight is negative wherever it is
defined, i.e. the suppression is not uniform noise — it is targeted at the
strongest couplings. At the tests' parameters the window underflows
outright (4e-50 on OT, 1e-19 on rotor).

Note `harris_tearing` under the test parameters: `w_max` = 0.26 looks
healthy, yet mass kept = 2.5e-60. `max(|C|·w) ≠ max|C|·max(w)` — the window
is large only where the coupling is not. This is the anti-correlation in its
starkest form and is why the window's effect cannot be judged from `w_max`.

**Consequence.** The Ising formulation's rationale is the multi-body
coupling. The deployed pipeline discards ~99 % of it before the QAOA ever
sees it, which is a sufficient explanation for T13's null ablations and for
T11b's near-zero variational progress.

**Defect D6.** `bash run_tests.sh` does **not** pass on a clean checkout.
Re-running the V1 suite in a detached worktree at `cf93ba3` (the last commit
touching `src/` or `tests/`, well before any V3/V4 work) reproduces an
identical set of 8 failures:

- 6 × `TypeError: PhysicalMapper.__init__() got an unexpected keyword
  argument 'beta'` — the tests call a signature that no longer exists.
- 2 × substantive assertions:
  `test_coefficients_survive_orszag_tang` ("Orszag-Tang should produce
  significant C_edges", actual 1.77e-48) and
  `test_hamiltonian_carries_spatial_info_beyond_score` ("C_edges should be
  nonzero at velocity boundary", actual 1.79e-42).

The two substantive failures are the V1 author's own guard against exactly
the failure mode T17 characterises. They have been failing, not passing.

**Defect D7.** The uncertainty window annihilates the family it is meant to
focus (numbers above). Documented irony: V1 replaced Michelson
normalisation because it *"kills the signal when the domain is uniformly
active"*; the uncertainty window reintroduces that failure mode at the score
level.

Tests: `tests/v4/test_t17_uncertainty_window.py` (9).

---

## T18 — counterfactual: are the ZZ terms inert *without* the window?

```
python study/v4/t18_window_counterfactual.py --N 256 --dim 2 --n-snaps 2
```
runtime ≈ 2 s (reuses the stored DNS/patch inputs) · deployed v1 mapper

**Why this task exists.** T17 shows the uncertainty window discards most of
the ZZ coupling. That immediately raises the question a referee will ask,
and the answer decides how far the paper's conclusion reaches:

> is the causal inertness of ZZ a property of the **Ising formulation**, or
> an artefact of **this implementation**?

If the window were solely responsible, the defect would be a repairable
engineering bug and the critique would not touch the approach.

**Protocol.** Two Hamiltonians per snapshot, same physics, same deployed v1
mapper: `windowed` (the pipeline as it runs) and `no_window` (σ → 1e9, so
w ≡ 1). Neutralisation is done by substituting the module constant used to
*construct* the mapper and restoring it in a `finally`; V1 is never
modified, and the substitution is asserted, not assumed (|C| without the
window must dominate |C| with it). The T13 ablations are then replayed on
each arm — `zero_hamiltonian_terms` and `ground_state_mask` are imported,
never redefined.

**Coupling amplitude at the deployed configuration** (N=256, dim=2). Note
these are *more* extreme than the N=64 figures in T17: at VQA resolution the
patch-averaged fields are smoother, so the score sits even further from the
threshold.

| class | snap | max\|C\| windowed | max\|C\| no window |
|---|---|---|---|---|
| orszag_tang | 14 | 1.33e-189 | 137.5 |
| orszag_tang | 29 | 5.65e-145 | 154.5 |
| harris_tearing | 10, 19 | **0.000e+00** | 24.89 |
| kelvin_helmholtz | 14 | **0.000e+00** | 124.2 |
| kelvin_helmholtz | 29 | **0.000e+00** | 77.32 |
| mhd_rotor | 14 | 1.25e-189 | 117.2 |
| mhd_rotor | 29 | 2.70e-200 | 143.9 |

At the deployed size the ZZ family is **identically zero in double
precision** on Kelvin–Helmholtz and Harris tearing, and at 1e-145 or below
on the others.

**Ablations, both arms:**

| arm | ablation | changed | uniform GS | refined | F1 | n_optima |
|---|---|---|---|---|---|---|
| windowed | full (control) | **0.0000** | 1.000 | 0.750 | 0.250 | 64.8 |
| windowed | no_Z | 0.7500 | 1.000 | 0.000 | 0.000 | 88.0 |
| windowed | no_ZZ | **0.0000** | 1.000 | 0.750 | 0.250 | 64.8 |
| windowed | no_ZZZZ | **0.0000** | 1.000 | 0.750 | 0.250 | 64.8 |
| no_window | full (control) | **0.0000** | 1.000 | 1.000 | 0.250 | 1.0 |
| no_window | no_Z | 1.0000 | 1.000 | 0.000 | 0.000 | 22.0 |
| no_window | **no_ZZ** | **0.0000** | 1.000 | 1.000 | 0.250 | 1.0 |
| no_window | **no_ZZZZ** | **0.0000** | 1.000 | 1.000 | 0.250 | 1.0 |

**Result.** With the coupling restored from numerically zero to O(25–155),
ablating ZZ *still* changes **0.0000** decisions; likewise ZZZZ. The
inertness is therefore **not** an artefact of the uncertainty window. It is
a property of the formulation at the deployed size: the Z bias alone fixes
the ground state, and the multi-body terms cannot move it.

This is the stronger result for the paper — it forecloses the "your
implementation was simply buggy" rebuttal. The window is a real defect
(D7), but repairing it would not make the coupling terms matter.

**A separate, subtler finding.** The window does change decisions —
**25.0 %** of them (full Hamiltonian, windowed vs neutralised) — but *not*
by acting as coupling. |C| feeds `C_scale`, the median of non-zero |C| and
|K| that sets the Z-bias amplitude `alpha_z = w_z_frac × C_scale`.
Suppressing C therefore rescales the **Z bias**, and the decision moves
through that normalisation side-channel. The coupling influences the outcome
only as an input to a scale factor — never as a coupling. Between the arms
the ground state also goes from 64.8-fold degenerate to unique.

Note the control (`full` = 0.0000) holds in both arms, so the measurement
chain is validated separately for each.

Tests: `tests/v4/test_t18_window_counterfactual.py` (7), including a
positive control — the instrument is shown to detect a change when one
exists, without which "changed = 0" everywhere would prove nothing.

### T18 addendum — an *independent* counterfactual: the V2 mapper

The σ → ∞ neutralisation in T18 is a manipulation of the v1 mapper, so a
referee may reasonably ask whether the conclusion is an artefact of the
manipulation. It is not, and the repository already contained the control:

**`PhysicalMapperV2` has no uncertainty window at all.** Its own docstring
lists what was removed relative to v1: *"Removed: sigma (Gaussian
uncertainty width) … Removed: f-gate, g-gate, threshold-contrast, Gaussian
weighting"*. It is parameter-free, using plain domain-normalised ratios.

Its ZZ coupling is consequently healthy — measured at the deployed
configuration (N=256, dim=2), max|C_edges|:

| class | snap | v2 (no window) | v1 (windowed) |
|---|---|---|---|
| orszag_tang | 14 / 29 | 2.455 / 2.613 | 1.33e-189 / 5.65e-145 |
| kelvin_helmholtz | 14 / 29 | 2.774 / 2.522 | **0.000e+00** / **0.000e+00** |
| mhd_rotor | 14 / 29 | 2.017 / 2.101 | 1.25e-189 / 2.70e-200 |
| harris_tearing | 14 | 3.989 | **0.000e+00** |

And the ablations on that mapper (N=256, dim=2, `--n-snaps 3`, 72 rows):

| ablation | changed | uniform GS | refined | F1 | n_optima |
|---|---|---|---|---|---|
| full (control) | **0.0000** | 1.000 | 1.000 | 0.389 | 1.0 |
| no_Z | 1.0000 | 1.000 | 0.000 | 0.000 | 8.0 |
| **no_ZZ** | **0.0000** | 1.000 | 1.000 | 0.389 | 1.0 |
| **no_ZZZZ** | **0.0000** | 1.000 | 1.000 | 0.389 | 1.0 |

So the conclusion now rests on **two independent routes**:

1. **v1 with the window neutralised** (T18): coupling restored to O(25–155)
   → ZZ ablation 0.0000.
2. **v2, independently designed without a window** (T13, mapper v2):
   coupling natively O(2–4) → ZZ ablation 0.0000.

The second route involves no manipulation of any kind. The causal inertness
of the multi-body terms is a property of the formulation at the deployed
size, not of the v1 implementation and not of the σ → ∞ device.

**Defect D9 (in V4's own code, now fixed).** `t13_term_ablation.py` wrote
`t13_term_ablation_N{N}_dim{D}.npz` *regardless of `--mapper`*, so running
the v2 comparison silently overwrote the v1 result — precisely the
comparison the task exists to make. The filename now carries the mapper;
the historical name is still written for v1 so published references keep
resolving. Found by re-deriving the v2 numbers instead of citing them.

**Reproducibility check.** Re-running the published v1 configuration
(`--n-snaps 3`) reproduces the stored artifact **bit-exactly** across all
72 rows (`scenario`, `snap`, `ablation`, `changed`, `uniform`, `n_optima`,
`f1`, `refined`, `dE`).
