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

### D6 follow-up — how far does the signature drift reach?

D6 reports 8 failures in the V1 suite on a clean checkout, 6 of them
`TypeError: PhysicalMapper.__init__() got an unexpected keyword argument
'beta'`. The question that matters for the paper is whether that drift
touches the code which produced the results. It does not.

Every call site, checked exhaustively:

| call site | kind | uses removed `beta=` |
|---|---|---|
| `src/pipeline.py:325` | **production pipeline** | no — current signature |
| `study/phase0_sanity_check.py:95` | study | no |
| `study/phase3_coefficients.py:68` | study | no |
| `study/phase4_exact_diag.py:68` | study | no |
| `study/phase5_qaoa_eval.py:136` | study (feeds T11/T13/T18) | no |
| `src/compare_rotor_budget.py:110` | orphaned analysis script | **yes → dead** |
| 6 × `tests/…` | stale tests | **yes → the D6 failures** |

**Verdict.** The simulations behind every V3 and V4 number were produced by
code that constructs the mapper correctly. The drift is confined to stale
tests and to one script that nothing imports.

**Defect D10.** `src/compare_rotor_budget.py` raises `TypeError` at line 108
and cannot execute. It and `HamiltParams.py` were both last modified in
`cf93ba3` and are unchanged since; the repository has full (non-shallow)
history of 57 commits. **As committed, this script has never been runnable
in this repository.** It is referenced only by a file listing in
`README.md`. If any rotor budget-comparison figure or number in the
manuscript is attributed to it, that attribution needs checking — the script
in its committed form could not have produced it.

---

## T15 — Level-3 fold `kh` (Kelvin–Helmholtz held out)

```
bash study/v4/run_fold.sh kh
```
tuning: QAOA 4 trials (best train loss 0.2590), classical 2 trials (0.3841)

| arm | combined | phys (rel. L2 vs DNS) | patch ratio | wall (s) |
|---|---|---|---|---|
| Q-HAS | 0.2443 | 0.0070 | 0.8376 | 579 |
| **classical** | **0.1800** | **0.0020** | **0.6250** | 213 |

**The classical arm wins on every endpoint simultaneously**: better fidelity
(3.5× lower L2), cheaper (25 % fewer refined pixels), and better composite.
Unlike fold `ot`, this needs **no budget-matched control** — Q-HAS is
**strictly Pareto-dominated at the tuned operating point itself**. The
budget-matched run is still executed, but only to map the frontier; it
cannot change the direction of the conclusion.

Note the training losses reproduce fold `ot`'s pattern — QAOA better than
classical on the *training* composite (0.2590 vs 0.3841 here, 0.1984 vs
0.2979 on `ot`) while losing on the *held-out* class. That is defect **D4**
in action: the QAOA arm's `threshold_amr` is pinned at 0.1496 while the
classical arm tunes its own freely, so a training-loss advantage reflects a
different operating point rather than a better decision rule.

### Cross-fold state after 2 of 4 folds

| fold | Q-HAS combined | classical combined | Δ (Q-HAS − cl) | better |
|---|---|---|---|---|
| ot | 0.3328 | 0.4386 | −0.1058 | Q-HAS |
| kh | 0.2443 | 0.1800 | **+0.0643** | **classical** |

Pre-registered readings, stated at their true scope:

- **Counting rule** (`docs/level3_preregistration.md` §4): 1–1 at n = 2.
  Neither arm meets the ≥ 3/4 threshold. **Nothing is established yet.**
- **TOST**: margin 0.0155 (5 % of mean classical `combined`, per the frozen
  formula), diff −0.0208, p_TOST = 0.520 → **equivalence not established**.
- **Difference test**: paired t p = 0.848, Holm-adjusted 1.000 → no
  significant difference. Exact sign test p = 1.000, and note the minimum
  attainable at n = 2 is 0.500 — the design cannot produce significance here
  regardless of the data.
- **Budget-matched (secondary, post-hoc):** Q-HAS dominated on 1/1 folds so
  far; `kh` is dominated already without the control.

The honest summary at this point: on the two folds measured, Q-HAS is
Pareto-dominated on both — on `ot` only after correcting the operating-point
asymmetry, on `kh` outright. The *primary* pre-registered endpoint remains
undecided by its own counting rule until 3 or 4 folds are in.

---

## T19/T20 — the Q-HAS arm is not deterministic (defect D11)

The T19 audit replays each Level-3 arm with **identical** inputs (same DNS
trace, same hot start, same hyperparameters) and checks it reproduces the
stored value. Fold `ot`:

| arm | stored `combined` | replayed `combined` | stored phys | replayed phys |
|---|---|---|---|---|
| classical | 0.4386 | **0.4386** (exact) | 0.4845 | 0.4845 |
| **Q-HAS** | 0.3328 | **0.3108** | 0.1940 | **0.1345** |

The classical arm reproducing bit-exactly proves the trace, hot start and
configuration are identical — so the variance is specific to the QAOA path.
A 44 % swing in `phys_score` between two runs of the same configuration.

**Cause.** No RNG seed is fixed anywhere in V1's VQA chain: `AerSimulator`
is built without `seed_simulator`, and both `Estimator` and `Sampler` run at
`default_shots = 256` (`create_argus`: `shots=256`, `backend="state_vector"`,
`method="COBYLA"`). The Q-HAS arm is therefore doubly stochastic:

1. the objective COBYLA minimises is a 256-shot estimate, so the optimiser
   follows a different trajectory each run;
2. the final marginal read-out is itself a 256-shot draw.

The classical arm samples nothing, hence its exact reproducibility — which
is what makes it a valid control rather than a coincidence.

**Consequence.** Every published Level-3 Q-HAS number is **one draw** from a
distribution whose spread has never been measured. `--seed` cannot fix this:
the randomness is inside V1's unseeded Aer backend, and seeding it would
require modifying V1.

**Scope of the damage — what still holds.** On fold `ot` the two observed
Q-HAS draws are phys ∈ {0.1345, 0.1940}; the budget-matched classical arm
achieves **0.0827**. Both draws are worse, so the *direction* (Q-HAS
Pareto-dominated) survives, while the *magnitude* (quoted as 2.3×) is
uncertain over roughly 1.6×–2.3×. The same caution applies to `kh`
(Q-HAS 0.0070 vs matched classical 0.0017).

**T20** quantifies the spread directly: K repeats of the Q-HAS arm on one
fold with identical inputs, plus classical repeats as a determinism control,
and reports the between-arm gap divided by the Q-HAS run-to-run standard
deviation. A gap smaller than ~2 standard deviations means a single run per
arm cannot support a directional claim on that fold.

```
python study/v4/t20_qhas_run_variance.py --fold kh --repeats 5
```

**This is the strongest methodological caveat in the V4 set, and it applies
to V1's own published closed-loop numbers too** — those were also single
runs of the same unseeded pipeline.

---

## T19 complete + T21 — the endpoint judgement becomes a measurement

### T19 arm audit, all four folds

| fold | Q-HAS arm | classical arm | verdict |
|---|---|---|---|
| `ot` | completed | completed | **usable** |
| `kh` | completed | completed | **usable** |
| `rotor` | completed | **ABORTED, step 208 (t=0.2739)** | **failed** |
| `tearing` | completed | completed | **usable** |

The classical arm reproduced its stored value **bit-exactly on all four
folds**; the Q-HAS arm reproduced on **none** — the D11 signature.

### T19 bisection-trace audit

| fold | aborted points |
|---|---|
| `rotor` | **2/6** — thr 0.4250 (step 371), thr 0.8000 (step 198) |
| `tearing` | **0/6** |

**A heuristic would have been wrong here.** `tearing`'s point at
phys = 4.1258 looks like a divergence and is not: it *completed*. It is a
genuine operating point at thr = 0.8, patch = 0.0727 — refine almost
nothing and the solution is badly wrong but stable. A rule such as
"phys > 1 ⇒ diverged" would have deleted a valid frontier point. The
criterion used is V1's own execution trace, never the value.

`rotor`'s two aborts also explain its fold failure: the tuned classical
threshold, 0.4616, sits inside the unstable band between 0.4250 and 0.8000.
The tuner selected an operating point that diverges on the held-out class —
a second instance of D4 doing damage.

### T21 — is the primary endpoint well posed?

Replaces the *argument* "the primary endpoint is contaminated by D4" with
three measurements, none requiring new simulation. `rotor` excluded per
pre-registration §5 (failed audit).

**1. Pareto dominance — no λ involved.**

| fold | dominates | λ-free verdict |
|---|---|---|
| `kh` | **classical** | yes |
| `tearing` | **classical** | yes |
| `ot` | incomparable | no |

**2/3 folds are decided without any λ, both for the classical arm, none
for Q-HAS.**

**2. λ crossover**, for the fold dominance cannot decide. The two arms'
`combined` cross at λ\* = (phys_c − phys_q)/(patch_q − patch_c):

- `ot`: **λ\* = 0.8164**. Q-HAS wins below, classical above. The
  pre-registered λ = 0.4 sits **below** the crossover.

**3. Count stability across λ:**

| λ | Q-HAS wins | classical wins |
|---|---|---|
| 0.0 – 0.8 | 1 | **2** |
| **≥ 1.0** | **0** | **3** |

**Correction to an earlier reading.** The "2–2 split establishes nothing"
reported before the audit **included `rotor`**, whose classical arm had
diverged and was therefore scored as a Q-HAS win. With `rotor` excluded as
pre-registration §5 requires, the primary endpoint favours the classical arm
**2–1 at the pre-registered λ**, and **3–0 for λ ≥ 1**.

At λ ≥ 1 the classical arm meets the pre-registered refutation threshold
(§4: *"If the classical arm wins on ≥ 3/4 folds … the falsification is
complete and closed-loop"*), on 3/3 valid folds.

**What this measures and what it does not.** It measures that the verdict is
partly a property of the chosen λ rather than of the arms — ill-posedness,
quantified, not asserted. It does **not** remove D4. Removing it requires
re-tuning the QAOA arm with `threshold_amr` in the search space so both arms
optimise the same free parameters: hours of compute, and the definitive
experiment.

### Figure updated

`figures_v4/pareto_panel.*` now (a) excludes `rotor`'s two aborted points
from the plotted frontier, and (b) uses a **logarithmic error axis** — the
classes span 1–3 decades, and since the compared quantity *is* a ratio, a
log axis makes a given ratio span the same vertical distance in every panel.
The full data, including excluded points, remains in the `.csv`.

---

## T20 — Q-HAS run-to-run variance on fold `kh` (D11 quantified)

```
python study/v4/t20_qhas_run_variance.py --fold kh --repeats 5
```
5 Q-HAS runs + 2 classical controls, identical inputs, 3216 s.

| metric | Q-HAS mean | std | range | CV | classical range |
|---|---|---|---|---|---|
| combined | 0.2500 | 0.0104 | 0.0232 | 0.042 | **0.00e+00** |
| **phys_score** | **0.00324** | **0.00158** | 0.0039 | **0.489** | **0.00e+00** |
| patch_ratio | 0.8670 | 0.0376 | 0.0785 | 0.043 | **0.00e+00** |

Q-HAS `phys` draws: **0.0015, 0.0020, 0.0031, 0.0042, 0.0053**.

**The control passes.** The classical arm's range is exactly **0.00e+00** on
all three metrics across both repeats — a fifth independent confirmation of
its determinism. Without that, the Q-HAS spread could have been an artefact
of the measurement chain; with it, the spread is attributable to the
unseeded QAOA path (D11) and nothing else.

**A 48.9 % coefficient of variation on the fidelity metric.**

### The published `kh` numbers were one draw, and it was the extreme one

The fold's stored Q-HAS value, 0.00700, sits at the **100th percentile** of
all six known draws — it is the largest. Everything computed from it is
correspondingly inflated:

| quantity | from the stored draw | **from the mean of 5 draws** |
|---|---|---|
| gap / std | 3.15 → "direction survives" | **0.77 → a single run cannot support a directional claim** |
| ratio vs budget-matched classical | 4.16× (published as 4.41×) | **1.93×** |

**The `kh` ratio is roughly halved.** T20 originally reported only the
stored-draw figure, which is the optimistic choice; it now computes both and
quotes the mean-based one.

### What survives, and it is the dominance count, not the ratio

Against the budget-matched classical arm (phys 0.00168 at patch 0.7943):

- Q-HAS costs **more on 5/5 draws** (patch 0.830–0.908 vs 0.794);
- Q-HAS is less faithful on **4/5 draws**;
- on the remaining draw the arms are **incomparable** (Q-HAS more faithful,
  but more expensive) — **never reversed**.

So the direction holds as a **dominance count over draws**, not as a point
ratio. The honest statement for `kh` is *"classical is cheaper on every
draw and more faithful on four of five"*, not *"Q-HAS is 4.4× worse"*.

### Consequence for the other folds

`ot`, `rotor` and `tearing` each have **one** Q-HAS draw (plus a replay for
`ot`). Their published ratios rest on the same single-draw basis and should
be read as **point estimates of a quantity with ≈50 % CV**, not as measured
magnitudes. Repeating T20 per fold is the fix; it costs ~1 h per fold.

---

## T20 complete — Claim E restated as a dominance count over repeated draws

> **SUPERSEDED — do not quote the per-fold numbers in this section.** This
> pass did not capture each draw's abort status, so `rotor`'s mean silently
> included 2 diverged trajectories. See *T20 verified* below for the numbers
> that stand (1.30×, 1.90×, 2.74×, 1.81×); the section is kept because the
> comparison between the two passes is what shows how much an unguarded
> draw distorts a mean.

5 Q-HAS repeats per fold, identical inputs, plus 2 classical repeats per
fold as a determinism control. **The classical control's range is exactly
0.00e+00 on every metric of every fold** — 8 independent replays. The spread
below is therefore attributable to the unseeded QAOA path (D11) alone.

### Per-fold distribution, against the **budget-matched** classical arm

| fold | Q-HAS mean | sd | CV | matched ref | gap/sd | ratio published → **mean-based** |
|---|---|---|---|---|---|---|
| `ot` | 0.1291 | 0.0222 | 17.2 % | 0.0827 | **2.09** | 2.35× → **1.56×** |
| `kh` | 0.0032 | 0.0016 | 48.9 % | 0.00168 | **0.98** | 4.16× → **1.93×** |
| `rotor` | 0.1537 | 0.0642 | 41.8 % | 0.0536 | **1.56** | 3.13× → **2.86×** |
| `tearing` | 0.0091 | 0.0034 | 37.4 % | 0.00443 | **1.37** | 4.19× → **2.05×** |

**On three folds of four the gap/sd is below 2**: a single run per arm
cannot support a claim about *magnitude*. Every published ratio was inflated
by a factor 1.1–2.2, because each rested on one draw.

### Why the reference must be the budget-matched arm, always

T20 first compared against the *tuned* classical arm, which is wrong twice
over and produced two spectacular non-results:

- **`rotor`**: the tuned classical arm had **aborted**, so its stored value
  is a partial score. gap/sd came out **15.88** — against a crashed run.
- **`ot`**: the tuned classical arm *completes* but runs at a different
  budget (patch 0.324 against Q-HAS's 0.680, defect D4). gap/sd came out
  **16.01**, measuring the operating point, not the decision rule.

Both are now excluded by construction: the reference is the budget-matched
point, whose completion the T19 trace audit verified.

### The robust statement

```bash
python study/v4/t23_headline_counts.py     # recomputes the table below
```

| fold | n | aborted | less faithful | costlier | strictly dominated |
|---|---|---|---|---|---|
| `ot` | 5 | 0 | 5/5 | 5/5 | **5/5** |
| `kh` | 5 | 0 | 5/5 | 4/5 | **4/5** |
| `rotor` | 3 | 2 | 3/3 | 2/3 | **2/3** |
| `tearing` | 5 | 0 | 5/5 | 5/5 | **5/5** |
| **total** | **18** | 2 | **18/18** | **16/18** | **16/18** |

> Across four held-out classes and **18 completed** closed-loop runs, Q-HAS
> is less faithful than the budget-matched classical rule on **every one of
> the 18**, more expensive on **16 of 18**, and strictly Pareto-dominated on
> **16 of 18**. No run reverses the ordering on both coordinates at once.

**Correction — this table previously read 19/20, 18/20, 17/20.** It was the
only headline in the study composed by hand rather than computed, and it did
not reproduce from the artifacts. Two errors, both of a kind already in the
register:

1. on `kh`, *less faithful* and *costlier* were **transposed** (4/5 and 5/5
   instead of 5/5 and 4/5);
2. on `rotor`, the **2 aborted draws were counted in the denominator**,
   giving a total out of 20 when only 18 runs completed — the exact defect
   ("an aggregation mixing aborted draws with valid ones") that had been
   fixed in the code and reappeared in the prose.

The corrected count is **stronger on fidelity** (unanimous, 18/18, where the
old figure conceded one run) and **weaker on cost** (16/18). The direction of
the conclusion is unchanged. T23 now computes it and `t16` checks it, so the
number can no longer drift from its artifacts.

This is the form Claim E should take in the manuscript. It is weaker-sounding
than "2.6–4.4× worse" and far harder to attack: it depends on no single draw,
no choice of λ, and no scalarisation.

### Correction to an earlier claim of mine

I wrote that the published value was the maximum draw "on all four folds".
That was true for `kh`, `ot` and `tearing` but **not** `rotor`, whose stored
value sits at the 67th percentile. Three of four, generalised too early from
three observations.

---

## D13 — a train/test leak in the Level-3 protocol, and the unseen-condition test

### The leak

`docs/level3_preregistration.md` states the held-out class is excluded from
**all** tuning of both arms. That is **false for the QAOA arm**.

`TrainHyperParam_v2.make_composite_objective` hard-codes the decision
threshold:

```python
if "threshold_amr" not in frozen:
    HyperParams["threshold_amr"] = 0.14959824837662078   # le meilleur classique
```

and that number comes from `_run_classical_phase1`, whose own banner reads
**"Scenarios: KH + OT + Tearing + Rotor"** — all four classes. So on every
fold, the QAOA arm decides using a threshold fitted on data that includes
the held-out class. My driver reproduced it verbatim:
`best.setdefault("threshold_amr", 0.14959824837662078)`.

The classical arm has no such problem: `train_classical_threshold_excluding`
re-tunes its threshold per fold on the training classes only.

**The leak is asymmetric and favours Q-HAS.** It is therefore *conservative*
with respect to the conclusion — Q-HAS is beaten on all 18 completed runs
despite holding an advantage it should not have. But the protocol's claim of a clean
LOSO is wrong as written and must be corrected in the manuscript.

This is also the precise form of defect D4: not merely "different operating
points" but a genuine information leak.

### The second, independent problem: the initial condition was never new

Even with the parameter leak removed, V1's `_init_dns_scenario` calls every
`init_*` **without arguments**, so every evaluation uses the canonical
initial condition. A model that generalises must face a condition it has
never met, not the canonical trajectory of a class it merely did not tune on.

**T22** supplies that test. It substitutes `_init_dns_scenario` temporarily
(V1 unmodified, restored in a context manager, and the substitution is
*verified*: the run aborts if the trajectory does not actually change) to
pass physical parameters to the initialisers:

| class | unseen condition |
|---|---|
| Kelvin–Helmholtz | narrower shear layer, weaker seed, faster drift |
| Harris tearing | thinner current sheet, **mode 2** instead of mode 1 |
| MHD rotor | slower, smaller rotor, wider taper |
| Orszag–Tang | **no IC parameters exist** — the only available unseen condition is a different Reynolds number, declared as such |

Verified distinct at N=64 before launching: KH 3773.6 → 4118.3,
tearing 3546.8 → 2951.0, rotor 4739.8 → 4409.3, and V1's function object
restored identically afterwards.

The reported quantity is the **degradation ratio** of each arm,
phys(unseen) / phys(canonical), so the comparison is between how the two
decision rules *transfer*, not between their absolute errors.

---

## Trap sweep — where else can an invalid run masquerade as a valid one?

The recurring failure mode in this campaign is a computation that fails but
returns a value **indistinguishable from a valid one**. It has now surfaced
five times (T15 fold scoring, T20 gap/sd, T22 classical reference, T22
Q-HAS draws, and the T13/T19 filename overwrites). A systematic sweep of
every `run_arm` call site in `study/v4/`:

| call site | guarded? | recoverable after the fact? |
|---|---|---|
| `t15:313` Q-HAS fold arm | no | **no** — non-deterministic (D11) |
| `t15:319` classical fold arm | no | yes — deterministic, T19 audits it |
| `t15b:66` bisection points | no | yes — classical only, T19 `--trace-only` audits it |
| `t19:88` audit replay | **yes** | — |
| `t20:120` Q-HAS variance draws | **was no** | **no** |
| `t20:129` classical control | was no | yes |
| `t22:250` both arms | **yes** (fixed) | — |

**The one that mattered: T20's Q-HAS draws.** Those 18 completed runs (of
20 launched) underpin the
restated Claim E, and their completion was never verified. Because the arm
is non-deterministic, it **cannot** be verified now — replaying does not
reproduce the draw.

Evidence bounding the risk, short of a re-run: a divergence produces a
partial score wildly out of family with its siblings — the T22 case was
**300×**. The T20 spreads are max/min = 1.5 (`ot`), 2.9 (`tearing`),
2.7 (`rotor`), 3.6 (`kh`), and no draw exceeds phys = 1. All are consistent
with D11's measured CV of 17–49 %, none shows the divergence signature. So
contamination is **unlikely but unproven**.

`t20` now captures the abort marker per run and excludes aborted runs from
the statistics. A verified re-run is queued behind T22b; until it lands, the
Claim E numbers carry this caveat.

### Two smaller findings from the same sweep

**Optuna tuning was clean.** All completed trial values across the three
persisted studies lie in 0.23–0.51 — none at the divergence penalty (10.0),
none above 1. So no fold was tuned against diverged evaluations. The
`catch=(Exception,)` in `study.optimize` is a latent trap (a systematically
failing objective would be silently skipped) but did not fire: zero `FAIL`
states in any study.

**Fold `ot` has weaker tuning provenance than the other three.** It was
tuned before per-trial Optuna persistence existed, so
`t15_level3_optuna_ot.db` does not exist and its per-trial values are
unrecoverable. Its checkpoint carries an explicit provenance note: *"recovered
from logs/v4/level3.log after the container was reclaimed mid-run; QAOA
params printed at 4-decimal precision"*. The other three folds have full
trial-level records.

### Trap sweep, second pass: is the "unseen" condition actually unseen?

The T22 guard checked only that the trajectory *changed*. Two failure modes
slipped through it:

**(a) A diverged DNS would pass.** A trajectory that blows up produces a
huge signature, which reads as "changed". Checked by hand across all four
folds: signature ratios are 0.83–1.08, modest shifts with no blow-up, so
this did not fire. A finiteness test and a physical band (0.05–20) are now
enforced automatically.

**(b) A negligible change also passes.** This one *did* fire:

| fold | trajectory shift at hot start |
|---|---|
| `harris_tearing` | −16.7 % |
| `mhd_rotor` | −15.9 % |
| `kelvin_helmholtz` | +7.5 % |
| **`orszag_tang`** | **−0.3 %** |

`orszag_tang` exposes no initial-condition parameters, so its only available
"unseen condition" is a different Reynolds number — and Re 400 → 600 moves
the hot-start trajectory by **0.3 %**, some 20–50× less than the three
classes where the initial condition itself can be varied.

**Fold `ot`'s transfer test is therefore nearly vacuous** and must be
reported as such rather than counted alongside the other three. T22 now
warns below a 1 % shift and records `unseen_condition_is_weak`; T22c prints
the affected folds and refuses to let them carry a transfer claim.

This is a limitation of V1's API, not of the test: `init_orszag_tang()`
takes no arguments, and `src/` is read-only.

---

## Fresh-eyes review — assumptions re-examined from scratch

Six load-bearing assumptions, re-derived from the source rather than
from memory. Three held, three did not.

### HELD — the ablation is clean

**Both arms differ only in the decision routine.** `classical_only` swaps
`run_adaptive_vqa` → `run_adaptive_classical` on the *same* simulator
object, with the same mapper, `threshold_amr`, `target_dim`, `max_depth`,
`min_size` and TTL map (`pipeline.py:391`).

**Both arms threshold the same score.** `refinement.py:474` (classical) and
`:579` (VQA) both call `AngleMapper.classical_score(physics_state)`. The
QAOA route perturbs exactly the quantity the classical route thresholds, so
the comparison isolates the decision rule and nothing else.

**Both arms are scored at the same physical instant.** With a DNS trace
supplied — the Level-3 case — `dt = dns_trace[step]['dt']`
(`pipeline.py:458`), so both arms march on the DNS time grid and are
compared against the same `dns_trace[last_step]['fluxes']`. The
"adaptive dt desynchronises the arms" trap does **not** fire.

### DID NOT HOLD — three corrections

**(1) `phys_score` is not a plain relative L2.** It is an
*instability-weighted* relative L2: `score()` builds
`w = 1 + 0.25·(|Jz|/⟨|Jz|⟩ + |ω|/⟨|ω|⟩)` from the reference fields and
weights every field's error by it. Every table and figure axis in this
repository has called it "relative L2 vs DNS", which is wrong. Both arms are
scored identically so no bias follows, but the label must be corrected to
**"instability-weighted relative L2 vs DNS"** throughout the manuscript.

**(2) The cost axis excludes the cost of the decision.**
`patch_ratio = total_pixel_used / (steps · N²)` counts refined pixels only.
The QAOA circuit does not appear in it, yet the Q-HAS arm takes **2.7–3.3×**
the classical arm's wall time (ot 1069 s vs 371 s, kh 579 vs 213, tearing
240 vs 73) — on a *simulated* 8-qubit circuit, so hardware would be worse.
"Equal budget" therefore means "equal AMR budget, with Q-HAS's decision
compute free". This makes the conclusion **more conservative**, not less,
but the axis is mis-specified and must be declared.

**(3) T21's ill-posedness claim was overstated — my error.** T21 tested
whether the *count* changes with λ and concluded the endpoint was
ill-posed. Count and verdict are different things. Re-checked over
λ ∈ [0, 100] with `rotor` excluded:

| λ | Q-HAS | classical |
|---|---|---|
| 0.0 – 0.8 | 1 | **2** |
| 1.0 – 100 | 0 | **3** |

**The classical arm holds the majority at every λ tested.** The verdict
never flips; only the margin moves (2–1 → 3–0). The endpoint is *not*
ill-posed in its direction, and saying it was overstated the case. T21 now
separates "margin changes" from "verdict flips" and reports both; the
λ grid was extended to 100 because stability on [0, 5] proves nothing about
[0, ∞).

This correction **strengthens and simplifies** the result: the pre-registered
endpoint, once the failed fold is excluded as its own §5 requires, favours
the classical arm robustly rather than ambiguously.

---

## T22b complete — the transfer signal does not survive replication

56 runs, **zero aborted**, 5 Q-HAS draws per condition per fold, classical
reference budget-matched everywhere.

| fold | deg Q-HAS | deg classical | \|z\| | separable |
|---|---|---|---|---|
| `ot` † | 0.955 ± 0.373 | 0.946 | **0.02** | no |
| `kh` | 1.027 ± 0.509 | 1.364 | **0.66** | no |
| `rotor` | 0.312 ± 0.120 | 0.526 | **1.78** | no |
| `tearing` | 0.166 ± 0.065 | 0.389 | 3.45 | **yes** |

**1 fold of 4.** The single-run pass had suggested Q-HAS transfers
relatively better on *all four* folds (ratios narrowing 0.22→0.17,
2.52→1.81, 3.67→1.88, 2.94→1.01). Repeated with 5 draws, that pattern
evaporates: on `ot` the two arms degrade identically (|z| = 0.02).

† `ot` is unusable for this question regardless: its "unseen" condition
shifts the trajectory by only 0.3 % (no IC parameters exist on
`init_orszag_tang`).

**What holds — the reference-free count:**

| fold | ratio Q/C canonical → unseen | dominated on unseen |
|---|---|---|
| `ot` | 1.48× → 1.50× | 4/5 |
| `kh` | 2.18× → 1.64× | 5/5 |
| `rotor` | 2.48× → 1.47× | 4/5 |
| `tearing` | 3.27× → 1.39× | 5/5 |
| **total** | ratio narrows but never crosses 1 | **18/20** |

> Q-HAS is strictly Pareto-dominated on **18 of 20** runs against initial
> conditions it has never seen — less faithful *and* more expensive.

**Answer to the leakage question.** The concern was well founded but the
mechanism is sharper than "the model saw the end of a trajectory it trained
on":

1. a leak does exist (**D13**) — the QAOA arm's threshold was fitted on all
   four classes including the held-out one — and it **favours Q-HAS**;
2. the initial condition was never new, which T22 fixes;
3. and facing genuinely unseen conditions, Q-HAS remains **strictly
   dominated on 18 of 20 runs** — less faithful *and* more expensive.

On the third point, be precise about what is and is not claimed. Q-HAS's
*relative* degradation is smaller than the classical arm's on the one fold
where the difference is separable (`tearing`, 0.166 against 0.389, |z| =
3.45). That is a real observation and it is **not** evidence that Q-HAS
transfers better: it degrades less from a starting point that was already
worse, and it is still dominated on both coordinates on 5/5 of that fold's
unseen runs. T22d tests the obvious alternative explanation — that both arms
are approaching a common attainable floor — and that confound is not
resolved. So the honest statement is *"Q-HAS is not shown to transfer
better, and remains dominated in absolute terms"*, not *"Q-HAS transfers
worse"*.

So the conclusion does not rest on the leak: Q-HAS loses **despite** an
undue advantage, and loses again on conditions it has never met.

**Still open:** the common-floor confound on `tearing`, the one separable
fold. T22d measures it.

---

## T22d — distance to near-full refinement, all four folds

One classical run per condition at threshold 0.05 (refine almost
everything), the lowest point already swept by t15b's bisections.

| fold | reference can / uns | classical can / uns | Q-HAS can / uns |
|---|---|---|---|
| `tearing` | 0.00397 / 0.00155 | **1.12× / 1.11×** | 3.65× / 1.55× |
| `kh` | 0.00126 / 0.00166 | **1.33× / 1.39×** | 2.90× / 2.28× |
| `rotor` | 0.03395 / 0.02874 | **1.58× / 0.98×** | 3.91× / 1.44× |
| `ot` | 0.01111 / 0.00821 | **7.45× / 9.53×** | 11.04× / 14.28× |

### Three corrections to what I first claimed from this table

**(1) The reference is not a lower bound.** `rotor`'s classical arm scores
**0.98×** on the unseen condition — it *beats* near-full refinement. So
refining almost everything is not always optimal, and this quantity is an
estimate of the achievable optimum, not a certified floor. Any arm below
1.00× is now flagged by the script as proof of exactly that.

**(2) "The classical rule occupies the ceiling" holds on 3 folds, not 4.**
On `ot` **both** arms sit 7–14× above near-full refinement. There is
substantial headroom on that class which neither arm exploits, so the claim
that "there is nothing left for any method to gain" is false there.

**(3) Distance-to-reference is confounded by the operating point.** The
reference refines ~0.95 of the domain; `ot`'s classical arm runs at ~0.37,
`tearing`'s at 0.625. A cheaper operating point is mechanically further from
the full-refinement error, so these distances are **not comparable across
folds**.

### What survives without reservation

Within every fold and on both conditions, **Q-HAS is further from the
reference than the classical arm** — 11.04 vs 7.45, 2.90 vs 1.33, 3.91 vs
1.58, 3.65 vs 1.12. Eight comparisons, eight in the same direction, each one
between two arms at the same operating point on the same trajectory.

That is the only reading these measurements license, and it is enough: at
matched budget the quantum decision rule extracts strictly less of the
available accuracy than plain thresholding, on every class and both under
canonical and unseen initial conditions.

---

## Verified T20 — an aborted run does not always look anomalous

Re-running T20 with the abort marker captured at execution time (the
original pass had no such guard, and being non-deterministic could not be
audited afterwards) produced the finding that most changes how the earlier
numbers must be read.

**Fold `rotor`, Q-HAS draws:**

| draw | phys | status |
|---|---|---|
| 1 | 0.2191 | ok |
| 2 | 0.0978 | ok |
| **3** | **0.6877** | **ABORTED** |
| 4 | 0.0536 | ok |
| **5** | **0.4069** | **ABORTED** |

**Two of five draws diverged — 40 %, not the 1-in-5 I estimated.**

**And draw 5 returned 0.4069, a value that does not stand out.** The valid
draws span 0.054–0.219; 0.407 is high but not absurd. So an aborted run can
land inside the plausible range.

### This retracts my earlier bounding argument

I had written, to bound the risk on the unguarded pass: *"a divergence lands
300× out of family (the T22 case), while T20's spreads are 1.5–3.6× with no
draw above phys = 1 — consistent with D11's CV, no divergence signature.
Contamination unlikely but unproven."*

That reasoning is **wrong**. Contamination need not leave a visible
signature. `rotor`'s original five draws (max 0.2581) could perfectly well
have contained aborted runs, and no inspection of the values would reveal
it. The correct statement is not "unlikely but unproven" — it is
**unknowable without the guard**, which is precisely why the guard had to be
added and the pass repeated.

### A flaw in T20's own control

On `rotor`, **both classical control runs also aborted** (1.1731 twice).
T20 runs its determinism control at the *tuned* threshold, which diverges on
this fold. The control still shows determinism — the divergence reproduces
exactly — but it no longer validates the measurement chain, which is its
purpose. It should run at the budget-matched threshold, as the *reference
value* already does.

### D14 — the fix landed after two of the four folds had started

`always_matched=True` was added to T20's control, and the campaign was
*not* re-run: `ot` and `kh` had already been launched. Their control
therefore replays the **tuned** threshold while their artifact records
`classical_reference_source = "budget-matched classical"`. Both statements
are individually true — the field describes the *reference value*, read
correctly from T15b — but a reader naturally attaches it to the neighbouring
`classical_stats` block, and that block is something else entirely:

| fold | matched thr | replayed thr | matched phys | replayed phys |
|---|---|---|---|---|
| `ot` | 0.1906 | 0.4616 (tuned) | 0.0827 | **0.4845** |
| `kh` | 0.1906 | 0.4616 (tuned) | 0.00168 | 0.00202 |
| `rotor` | 0.0969 | 0.0969 ✓ | 0.05365 | 0.05365 |
| `tearing` | 0.4250 | 0.4250 ✓ | 0.00443 | 0.00443 |

`rotor` and `tearing` agree because the pre-fix code already fell back to
the matched threshold when the tuned arm had aborted.

**On `ot` this is enough to invert the fold.** Against the matched 0.0827,
Q-HAS's 0.1291 is 1.56× worse; against the replayed 0.4845 it is 3.75×
*better*. The published numbers use the matched value and are unaffected,
but anyone recomputing from `classical_stats` — as I did while building T23 —
gets the opposite sign on that fold. The two references are now split into
distinct fields and T23 documents which one is correct.

### D15 — the provenance stamp is taken at the wrong moment

`git_commit_hash()` runs when the artifact is *saved*. A run lasting an hour
is therefore stamped with whatever was committed while it was still
executing. That is exactly how the `ot` and `kh` artifacts carry a hash
postdating the `always_matched=True` commit while having executed the
pre-fix code — the stamp actively pointed away from the truth.

CLAUDE.md requires the commit hash in every output. It is necessary but
**not sufficient for long runs**: the hash must be captured at start, and a
run that spans a commit to its own source should say so.

### Consequence

Every variance figure published from the unguarded pass — the CVs, the
mean-based ratios (1.56×, 1.93×, 2.86×, 2.05×), the gap/sd values — rests on
draws of unknown status. They are superseded by this pass, and on `rotor`
the mean is now computed from **3 valid draws**, not 5.

---

## T20 verified — final numbers, and why the per-fold magnitudes cannot be quoted

All four folds re-run with the abort marker captured at execution time, the
classical control at a non-diverging threshold, and aborted draws excluded
from the statistics.

| fold | valid draws | mean phys | sd | CV | gap/sd | ratio vs matched classical |
|---|---|---|---|---|---|---|
| `ot` | 5/5 | 0.10727 | 0.01823 | 17.0 % | 1.35 | 1.30× |
| `kh` | 5/5 | 0.00320 | 0.00203 | **63.6 %** | 0.75 | 1.90× |
| `rotor` | **3/5** | 0.14725 | 0.04062 | 27.6 % | **2.30** | 2.74× |
| `tearing` | 5/5 | 0.00801 | 0.00193 | 24.1 % | 1.86 | 1.81× |

**Only 1 fold of 4 reaches gap/sd ≥ 2.**

### The magnitudes have now shrunk twice

| fold | first published (1 draw) | unguarded 5-draw mean | **verified 5-draw mean** |
|---|---|---|---|
| `ot` | 2.57× | 1.56× | **1.30×** |
| `kh` | 4.41× | 1.93× | **1.90×** |
| `rotor` | 3.62× | 2.86× | **2.74×** |
| `tearing` | 4.38× | 2.05× | **1.81×** |

### The decisive observation: which fold "passes" is not stable

| fold | gap/sd unguarded | gap/sd verified |
|---|---|---|
| `ot` | 2.09 → **separable** | 1.35 → not |
| `rotor` | 1.56 → not | 2.30 → **separable** |
| `kh` | 0.98 | 0.75 |
| `tearing` | 1.37 | 1.86 |

Both passes report "1 of 4 folds separable" — **but not the same fold**. `ot`
fell below the threshold and `rotor` rose above it. At n = 5 draws, the
separability verdict is itself unstable, which is the clearest possible
evidence that **per-fold magnitude claims are not supportable at this sample
size**. Reporting "Q-HAS is 2.7× worse on rotor" would be reporting a number
whose confidence interval is wide enough to swallow the effect.

**What survives is the direction and the dominance count**, which do not
depend on any single fold's ratio: the verified mean exceeds the
budget-matched classical value on **4 folds of 4** (1.30×, 1.90×, 2.74×,
1.81×), and Q-HAS was strictly Pareto-dominated on 18 of 20 unseen-condition
runs (T22c).

### A robustness asymmetry not captured by any metric

`rotor`'s Q-HAS arm **aborted on 2 of 5 draws (40 %)** while its classical
control at the same budget completed both times, deterministically (0.0536
twice). Across the campaign, 6 Q-HAS aborts have been observed on `rotor`
against 0 for the classical arm at a matched threshold.

None of `phys_score`, `patch_ratio`, the dominance count or the λ analysis
measures this: they all presuppose a run that finishes. The quantum decision
rule produces refinement configurations that destabilise the solver at a
rate the classical rule does not, and that is a distinct failure mode
deserving its own line in the manuscript.

---

# CLOSING THE CLOSED-LOOP STUDY (Level 3)

Everything below is measured, carries the control that validated it, and is
covered by `t16_aggregate_v4.py` (100 rows, 0 DIFF, 0 MISSING).

## The one-sentence result

> Across four held-out instability classes, a Q-HAS closed loop is less
> faithful than a plain threshold rule at matched compute on **18 of 18**
> completed repeated runs, more expensive on **16 of 18**, strictly
> Pareto-dominated on **16 of 18**, and it additionally destabilises the
> solver on a fraction of
> runs where the classical rule never does.

## What the closed loop establishes, by strength of evidence

**1. Direction — robust, no free parameter.** The verified Q-HAS mean
exceeds the budget-matched classical value on **4 folds of 4** (1.30×, 1.90×,
2.74×, 1.81×). The pre-registered `combined` endpoint gives the classical arm
the majority at **every λ from 0 to 100** — the verdict never flips, only the
margin (2–1 below λ = 0.82, 3–0 above). Two of three usable folds are decided
by Pareto dominance alone, needing no λ at all.

**2. Robustness — a failure mode outside every metric.** `rotor`'s Q-HAS arm
aborted on **2 of 5** verified draws (40 %) while its classical control at
the same budget completed every time, deterministically. Six Q-HAS aborts
against zero classical across the campaign. `phys_score`, `patch_ratio`, the
dominance count and the λ analysis all presuppose a run that finishes.

**3. Transfer — no effect.** On genuinely unseen initial conditions,
**1 fold of 4** shows a separable difference in degradation; on `ot`, |z| =
0.02. The single-draw pass had suggested Q-HAS transfers *better* on all
four; repeated with 5 draws that pattern evaporates.

**4. Magnitudes — not supportable.** Both variance passes report "1 fold of
4 separable" **but not the same fold** (`ot` 2.09 → 1.35, `rotor` 1.56 →
2.30). At n = 5 the separability verdict is itself unstable. Quote the
direction and the counts; **do not quote per-fold ratios**.

## Conditions under which the result was obtained — all adverse to the classical arm

The conclusion is **conservative**: three known asymmetries favour Q-HAS and
it loses anyway.

| asymmetry | direction | status |
|---|---|---|
| **D13** — QAOA threshold fitted on all 4 classes incl. the held-out one | favours Q-HAS | measured, **not removed** |
| **cost axis** excludes the QAOA circuit; Q-HAS uses 2.7–3.3× the wall time | favours Q-HAS | declared |
| aborted Q-HAS draws excluded from its own statistics | favours Q-HAS | necessary, declared |

## What would overturn it

- removing D13 (`t22 --mode leak-free`) and finding Q-HAS wins;
- ≥ 3 physics seeds per fold showing the direction is seed-specific;
- the full 170-trial Optuna budget lifting Q-HAS above the matched classical;
- counting decision cost, which would only make the result stronger.

## What this study cannot say

- **Nothing about magnitude** per fold (n = 5, unstable separability).
- **Nothing about transfer on `ot`** — its unseen condition shifts the
  trajectory by 0.3 %, `init_orszag_tang()` taking no parameters.
- **Nothing about hardware**: the circuit is simulated, 8 qubits, noiseless.
- **Nothing about larger `VQA_N`**: everything here is the deployed depth-0
  size where the ground state is uniform (Claim A).

## The methodological finding, stated for the manuscript

Twelve distinct instances of one failure mode were found and fixed: **a
computation that fails, or does not do what it says, but returns a value
indistinguishable from a valid one**:

| form | count | where |
|---|---|---|
| V1's divergence guard returns a partial score with identical keys | 4× | T15, T20, T22 (×2) |
| a fixed output filename silently overwrites the prior result | 6× | T13 mappers, T19 folds, T20 pass, then T11, T11b, T12 (`--mapper` absent from the name) |
| an aggregation averaging aborted draws with valid ones | 1× | T16 |
| a CLI mode accepted and documented but never implemented | 1× | `--mode no-leak`: only the filename changed |

**Four of the twelve were in the verification code written to catch the
others**, and three more were found only by `tests/v4/test_silent_failure_sweep.py`,
which sweeps the mechanically checkable forms. Searching as you go is
demonstrably not enough.

A thirteenth defect is the *mirror* of the motif rather than an instance of
it: a genuine outcome made **indistinguishable from a run never launched**.
When every Q-HAS draw aborted at the leak-free operating point on `rotor`,
the guard raised `SystemExit` before writing any artifact — and it exited on
the first arm, leaving unmeasured the one question that mattered (does the
classical rule survive at that same threshold?). Recorded now as
`status = "total_abort"` with an undefined ratio.

One aborted draw returned `phys = 0.4069` against valid draws of
0.054–0.219: **contamination need not be visible in the values**. Any
closed-loop AMR study of this kind should record run completion status at
execution time, because with a non-deterministic arm it cannot be recovered
afterwards.
