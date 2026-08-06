# Q-HAS — consolidated results for the arXiv paper

**Purpose.** Single hand-off document gathering every result available for
writing the final manuscript: what is established, with which numbers, at
which scope, and what is still pending. It supersedes nothing — every number
traces to `study/v4/RESULTS_V4.md`, `study/v3/RESULTS.md`,
`docs/v3_master_table_ca7f815.md` or the named `.npz`/`.json` outputs.

**Reproducibility.** V3: `bash study/v3/run_study_v3.sh --all` → 51/51 OK at
commit `ca7f815`. V4: `python study/v4/t16_aggregate_v4.py` → self-checking
master table. Test gate: **188 pytests** (118 v3 + 70 v4).

---

## 1. Status board

| block | scope | status |
|---|---|---|
| V1 closed loop, in-distribution | 4 classes, N=256 | published (unchanged) |
| V3 L1 open loop (static sufficiency) | 4 classes × 4 Re, blocked + LOSO | complete |
| V3 L2 open loop (prediction, h ∈ {1,2,4,8}) | same | complete |
| **V4 T11 quantum attribution** | 4 classes, N=256, 12 snapshots | **complete** |
| **V4 T11b variational displacement** | 4 classes, N=256 | **complete** |
| **V4 T13 causal ablation** | 4 classes, N=256, mappers v1 **and** v2 (both artifacts stored) | **complete** |
| **V4 T12 equivariance** | 4 classes, N=256, dim 2 and 8 | **complete** (one honest negative) |
| **V4 T14 numerical validation** | grids 64/128/256, Re in and out of grid | **complete** |
| **V4 T15/T15b Level 3 closed loop** | fold `ot` done; `kh`/`rotor`/`tearing` running | **partial (1/4 folds)** |
| **V4 T17 uncertainty-window mechanism** | 4 classes × 3 parameter sets, N=64 | **complete** |
| **V4 T18 window counterfactual** | 4 classes, N=256, both arms controlled | **complete** |
| Hardware / noise / shots | — | not attempted (audit: premature without positive L3) |
| ≥ 5 physics seeds | — | not attempted |

---

## 2. The V4 result set, as paper claims

### Claim A — the cost Hamiltonian's optimum carries no spatial information

At the deployed size (`VQA_N = 2` → 8 qubits, periodic root scan, exactly
what `refinement.py` solves at depth 0):

- the cost Hamiltonian is **diagonal** (Z/ZZ/ZZZZ only), verified at runtime
  on 12/12 snapshots → its ground state is a computational basis state and
  "exact diagonalisation" is a classical enumeration of 2⁸ = 256 states;
- the exact ground state is a **uniform mask on 100 % of snapshots**;
- cause: ferromagnetic couplings dominate the bias — V2 mapper |C| ≈ 2.0,
  |K| = 1.0 against |h| ≈ 0.071 (ratio ≈ 28); V1 mapper median |C| = 0 with
  |h| ≈ 200–240, and on harris_tearing **every coefficient is zero**
  (identically null Hamiltonian);
- the V1 objective is **massively degenerate**: 64.8 of 256 configurations
  are optimal on average (88 with the bias removed).

*Source:* T11, T11b. *Figure candidate:* none needed; a table suffices.

### Claim B — every optimiser, quantum or classical, reaches that optimum

| solver | hit optimum | E gap | mask = exact GS | wall (s) |
|---|---|---|---|---|
| exhaustive (certified) | 1.000 | −1.4e-17 | 1.000 | 0.000 |
| cold simulated annealing | **0.583** | 1.41e-02 | 0.583 | 0.139 |
| SA warm-started | 1.000 | −1.4e-17 | 1.000 | 0.133 |
| greedy local search | 1.000 | −1.4e-17 | 1.000 | 0.000 |
| classical decision alone | 1.000 | −1.4e-17 | 1.000 | 0.000 |
| QAOA p = 1 / 2 / 3 | 1.000 | −1.4e-17 | 1.000 | 0.75 / 1.12 / 1.42 |
| QAOA p = 3, 4096 shots | 1.000 | −1.4e-17 | 1.000 | 1.603 |

The only solver that struggles is cold-start SA — the one that does *not*
begin from the classical answer. Pre-registered attribution rule fires:
quantum optimisation is not the source of any gain.

*Source:* T11 at N=256.

### Claim C — the deployed circuit does not optimise its declared cost

Displacement in marginal space from the amplitude encoding
θ = 2·arcsin(√score) toward the exact ground state:

| reps | progress | ‖disp‖ | ‖required‖ | mean marginal |
|---|---|---|---|---|
| 1 | +0.1588 | 0.1685 | 0.7487 | 0.7739 |
| 2 | +0.1192 | 0.1569 | 0.7487 | 0.7653 |
| 3 | +0.0766 | 0.1392 | 0.7487 | 0.7555 |
| 4 | −0.0132 | 0.1706 | 0.7487 | 0.7376 |

Mean progress **0.0854**, **monotonically decreasing with depth**, negative
by p = 4. The deployed decision is a small, largely orthogonal perturbation
of the classical score encoding — not a minimiser of the declared objective.

*Source:* T11b at N=256.

### Claim D — the coupling terms are causally inert

Exact ground state recomputed after zeroing each family (control must
change nothing), **deployed V1 mapper**, N=256:

| ablation | decisions changed | uniform GS | refined | F1 | n_optima |
|---|---|---|---|---|---|
| full (control) | **0.000000** | 1.000 | 0.750 | 0.333 | 64.8 |
| no_Z | 0.7500 | 1.000 | 0.000 | 0.000 | 88.0 |
| **no_ZZ** | **0.0000** | 1.000 | 0.750 | 0.333 | 64.8 |
| **no_ZZZZ** | **0.0000** | 1.000 | 0.750 | 0.333 | 64.8 |
| **Z only** | **0.0000** | 1.000 | 0.750 | 0.333 | 64.8 |

Identical conclusion for the V2 mapper. Removing *all* ZZ and *all* ZZZZ
couplings — the entire motivation for an Ising/quantum formulation — changes
**no decision**; removing the Z bias destroys the decision entirely.

This is a **causal** statement and it **contradicts the post-hoc topological
attribution** of the current manuscript (Fig. 3: "ZZ-dominated flips correct
in 52/68 on tearing", "ZZZZ-dominated in 61/92 on OT"). Those correlations
survive as observations; their causal reading does not.

*Source:* T13, both mappers, N=256.

#### D-bis — *why* they are inert: the uncertainty window discards the coupling

T13 establishes the decision-level fact. T17 establishes the mechanism, and
it was found in **V1's own test suite**: two pre-existing tests
(`test_v9_metrics.py::test_coefficients_survive_orszag_tang`,
`test_module_validation.py`) fail on a clean checkout while asserting the
opposite — *"Orszag-Tang should produce significant C_edges"* — with
max|C_edges| ≈ 1.8e-48.

In `HamiltParams.compute_coefficients`, the whole ZZ family is multiplied by
a Gaussian centred on the AMR decision threshold:

```
w(score) = exp(-((score - threshold_amr) / sigma)^2)
C_horiz *= w ;  C_vert *= w
```

The documented intent is to concentrate coupling where the classical
decision is *uncertain*. But the physics-derived coupling is largest where
field gradients are strongest — exactly the cells whose score is
*confident*. The two supports are therefore anti-correlated **by
construction**, and the window discards most of the coupling the physical
model computes.

With the window neutralised (σ → ∞, V1 untouched), max|C_edges| is
O(40–140) on **all four** classes — the coupling is healthy before the
window.

**Two distinct "trained" σ exist and must not be conflated** — they differ
by 100+ orders of magnitude in effect. `TRAINED_SIGMA` = **0.023** is the
open-loop pipeline constant used by T11/T13/T18; σ = **0.1888** is Optuna's
value for the Level-3 fold `ot`, closed loop only.

Fraction of ZZ coupling mass retained, Σ|C|·w / Σ|C|, at the **deployed
open-loop** setting (σ = 0.023) — the configuration behind the open-loop
claims:

| class | max\|C\| (no window) | ZZ mass kept | Spearman(\|C\|, w) |
|---|---|---|---|
| Kelvin–Helmholtz | 53.9 | **1.3 %** | −0.372 |
| MHD rotor | 136.0 | **7.7e-28** | −0.400 |
| Orszag–Tang | 63.6 | **4.2e-125** | −0.012 † |
| Harris tearing | 42.3 | **3.9e-154** | −0.502 |

At the deployed setting the ZZ family is **numerically dead on three of the
four classes**, and retains 1.3 % on the fourth.

At the more permissive Level-3 closed-loop setting (σ = 0.1888) the same
quantity is 11.4 % (KH), 0.199 % (tearing), 0.040 % (rotor), 0.0097 % (OT).

† degenerate: the window is numerically constant there, so the rank
correlation carries no information; the mass ratio is the meaningful figure.

So the window discards between **88.6 %** (best class, most permissive
setting) and **essentially all** of the physics-derived ZZ coupling, and it
removes it preferentially from the cells where it is largest. The ZZ family
that actually reaches the QAOA is a small, systematically mis-sited remnant
— which is precisely why ablating it changes no decisions.

At the parameters used by the failing V1 tests (σ = 0.05, threshold = 0),
the window likewise **underflows to zero**: w_max = 4.2e-50 on OT and
1.0e-19 on rotor. The failing tests are therefore not testing an exotic
corner — they sit between the two trained settings.

The irony is documented in V1 itself: the threshold-contrast filter replaced
Michelson normalisation because Michelson *"kills the signal when the domain
is uniformly active"*. The uncertainty window reintroduces that exact
failure mode one level up — at the score rather than the field.

*Source:* T17, N=64, 30 steps, four classes × two parameter sets.

#### D-ter — the counterfactual: repairing the window would **not** rescue ZZ

D-bis invites an obvious rebuttal: *"your ZZ terms are inert because your
pipeline destroys them — fix the window and the Ising formulation works."*
T18 tests exactly that, and the rebuttal fails.

Two Hamiltonians per snapshot, same physics and same deployed v1 mapper:
`windowed` (as it runs) and `no_window` (σ → 1e9, w ≡ 1; V1 untouched, the
substitution asserted rather than assumed). At the deployed configuration
(N=256, dim=2) the windowed coupling is **identically 0.000e+00** on
Kelvin–Helmholtz and Harris tearing and ≤ 1e-145 elsewhere; neutralising the
window restores it to **O(25–155)**.

| arm | no_ZZ | no_ZZZZ | full (control) |
|---|---|---|---|
| windowed | 0.0000 | 0.0000 | 0.0000 |
| **no_window** | **0.0000** | **0.0000** | 0.0000 |

With the coupling restored from numerically zero to O(100), ablating ZZ
still changes **no decision**. The inertness is therefore **a property of
the formulation at the deployed size, not an artefact of the window**: the Z
bias alone fixes the ground state and the multi-body terms cannot move it.

This is the stronger claim, and it is the one to make in the manuscript: the
critique survives a full repair of the implementation defect.

One subtlety worth a sentence in the paper: the window *does* flip **25.0 %**
of decisions, but not by acting as coupling. |C| feeds `C_scale`, the median
of non-zero |C|,|K| that sets the Z-bias amplitude
`alpha_z = w_z_frac × C_scale`; suppressing C rescales the **Z bias**. The
coupling reaches the decision only as an input to a scale factor.

**An independent confirmation, with no manipulation at all.** σ → ∞ is a
device applied to v1, so the result could be questioned as an artefact of
the device. It cannot: **`PhysicalMapperV2` has no uncertainty window** — its
docstring lists *"Removed: sigma (Gaussian uncertainty width) … Removed:
f-gate, g-gate, threshold-contrast, Gaussian weighting"* — and its ZZ
coupling is natively healthy, max|C| = **2.0–4.0** at N=256 where v1's is
0.000e+00. On that mapper the ablations give `no_ZZ` = **0.0000**,
`no_ZZZZ` = **0.0000**, `no_Z` = 1.0000, control = 0.0000.

Two independent routes, one manipulated and one not, reach the same place.
The manuscript should cite the v2 route as primary, since it requires no
intervention.

*Source:* T18 (v1, σ → ∞) and T13 mapper v2, both N=256, dim=2, both arms
controlled (`full` = 0.0000 in each). The published v1 run reproduces
**bit-exactly** on re-execution (72/72 rows).

### Claim E — closed loop: the apparent advantage reverses at equal budget

Fold `ot`, Orszag–Tang excluded from **all** tuning of both arms:

| endpoint | Q-HAS | tuned classical | Δ |
|---|---|---|---|
| combined (pre-registered primary) | 0.3328 | 0.4386 | −0.1058 |
| phys (instability-weighted relative L2 vs DNS) | 0.1940 | 0.4845 | −0.2905 |
| patch_ratio (compute) | 0.6797 | 0.3238 | +0.3558 |

Taken at face value Q-HAS wins. It must not be read that way: the arms sit
at different points of the error–cost frontier, because
`make_composite_objective` **hard-codes** the QAOA arm's `threshold_amr` at
0.1496 (never offered to Optuna) while the classical objective optimises it
freely and picked 0.4616.

Budget-matched classical frontier on the held-out class:

| threshold | patch_ratio | phys |
|---|---|---|
| 0.0500 | 0.9480 | 0.0111 |
| 0.1438 | 0.7369 | 0.0649 |
| **0.1906** | **0.6412** | **0.0827** |
| 0.2375 | 0.5866 | 0.1027 |
| 0.4250 | 0.3554 | 0.2899 |
| 0.8000 | 0.0156 | 0.5894 |
| *Q-HAS* | *0.6797* | *0.1940* |

**At equal (indeed slightly lower) compute the classical rule achieves
lower error, and Q-HAS is strictly Pareto-dominated on 5/5 repeated runs.**

The `0.1940` above is **one draw** of a non-deterministic arm (D11). Over 5
repeats the mean is **0.10727 ± 0.01823**, so the honest ratios are
**1.30×** against the measured budget-matched point and **1.79×** against
the interpolated frontier at the realised mean budget. The single-draw
figure of 2.57× is retracted; `figures_v4/pareto_panel.csv` now carries
both denominators, and `study/v4/t23_headline_counts.py` computes the
dominance counts.

**Robustness note for the referee:** this compares Q-HAS to a *frontier*,
not to one classical setting. Being dominated by a frontier neutralises the
"your Q-HAS was under-tuned" objection — a better-tuned Q-HAS would still
have to beat the frontier, not a point.

*Source:* T15, T15b. *Figure:* `figures_v4/pareto_frontier_ot.pdf` (+ `.png`,
`.csv`). **Scope: n = 1 fold.**

### Claim F — the solver converges at first order, not fourth

- Self-convergence on 64 → 128 → 256, t = 0.25: errors 3.344e-02, 1.673e-02,
  **observed order 1.00** (identical at 32→64→128).
- Cause isolated **at N=256**: with the divergence-free projection applied
  after the complete RK4 step, temporal order is **1.12**; with the
  projection removed it is **4.00** (errors 3.8e-07 → 9.2e-11). The FD4
  stencils are independently verified at order **4.00**.
- Conservation is clean: energy monotone at every resolution,
  `max|div B|/rms|B| ≤ 8.0e-14`; Re = 200 and 3200 (outside the training
  grid) both pass.

`step_full` applies a first-order Lie splitting. The components are 4th
order; the *scheme* is 1st order. Both arms share the solver and all
comparisons are paired, so this does not invalidate them — but the methods
description must be corrected and absolute L2 values scoped (the DNS
reference itself carries ≈ 1.7 % discretisation error between N=128 and 256).

*Source:* T14 at N=256.

### Claim G — equivariance: one measurable defect, one non-result

- Classical route orbit error at dim = 8: **0.0146** (deterministic, floor 0),
  three times smaller than at N=64 — consistent with the one-sided
  finite-difference explanation.
- Ground-state route: 0.4219 against a **reproducibility floor of 0.3613**
  (same field, different anneal seeds). The script refuses the
  interpretation: the optimiser is less reproducible than the effect.
  What this establishes is a **degeneracy defect**, not an equivariance
  defect.
- rot180 commutes with the solver at machine precision (2.8e-16), so the
  transformation itself is exact.

*Source:* T12. This is the honest negative that replaces the Fig. 4
asymmetry speculation.

---

## 3. Defect register (all to be disclosed in the manuscript)

| # | defect | where | consequence |
|---|---|---|---|
| **D1** | KH physics-seed amplitude 0.1 overwhelms the intended mode | V3 Task 8 | amplitude reduced to 0.005; logged in `protocol_deviations.md` |
| **D2** | phase-1b KH growth observable subtracts the mean along the wrong axis, leaving the base-flow variance (≈0.341 vs ≈2.5e-4) in `Ep` | `phase1b_dns_validation.fluctuating_KE` | the published check could **never** detect KH growth, including at seed 0; repaired in the v3 copy, phase 1b untouched |
| **D3** | `SCENARIOS_ALL` = ISOLATED + COMPLEX lists `ot` and `rotor` twice → 6 entries for 4 distinct classes | `TrainHyperParam_v2` | OT and rotor weighted 2:1 in every Phase-3 composite loss; for LOSO it would **manufacture leakage** (de-duplicated in the V4 driver). Also `SCENARIO_VORTEX` / `SCENARIO_COALESCENCE` defined but never used, contradicting the module's own docstring |
| **D4** | QAOA arm's `threshold_amr` hard-coded at 0.1496 while the classical arm's is tuned freely | `make_composite_objective` vs `make_classical_composite_objective` | the two arms are compared at different operating points — affects the V1 closed-loop numbers, not only Level 3 (see Claim E) |
| **D5** | RK4 then projection = first-order Lie splitting | `solver.py::step_full` | scheme converges at order 1 despite 4th-order components (Claim F) |
| **D6** | the V1 regression suite does **not** pass on a clean checkout: 8 tests fail at `cf93ba3`, the last commit touching `src/`/`tests/` — before any V3/V4 work | `tests/test_module_validation.py`, `tests/test_v9_metrics.py` | 6 are signature drift (`PhysicalMapper(beta=…)` no longer exists); **2 are substantive** and assert that ZZ coupling survives on Orszag–Tang. `CLAUDE.md`'s premise that `run_tests.sh` "must pass unchanged" was already false. Verified by re-running the suite in a detached worktree at `cf93ba3`: identical 8 failures |
| **D7** | the Gaussian uncertainty window annihilates the ZZ family it is meant to focus | `HamiltParams.compute_coefficients` | 88.6 %–99.99 % of the physics-derived ZZ coupling mass is discarded, preferentially where the coupling is largest (Spearman −0.37 to −0.50); at the deployed N=256 the family is **identically 0.000e+00** on KH and tearing. This is the **mechanism** behind Claim D (D-bis). Repairing it does **not** restore causal relevance (D-ter), so it weakens the implementation, not the conclusion |
| **D8** | the ZZ coupling reaches the decision only through a normalisation side-channel | `HamiltParams.compute_coefficients` | `|C|` feeds `C_scale` = median(non-zero \|C\|,\|K\|), which sets the Z-bias amplitude `alpha_z`. Suppressing the coupling therefore rescales the **Z bias** and flips 25.0 % of decisions — while the coupling never acts as a coupling. Any claim that "ZZ terms influence the outcome" is true only in this degenerate sense |
| **D9** *(V4's own code, fixed)* | `t13_term_ablation.py` wrote the same filename regardless of `--mapper`, so the v2 run silently overwrote the v1 result | `study/v4/t13_term_ablation.py` | the v1/v2 comparison is the point of the task, and the artifact could not hold both. Filename now carries the mapper; the historical name is still written for v1. Found by re-deriving the v2 numbers rather than citing them from the earlier session |
| **D10** | `src/compare_rotor_budget.py` cannot run: it constructs `PhysicalMapper(..., beta=0.5, ...)`, a keyword removed from the signature | `src/compare_rotor_budget.py:110` | raises `TypeError` at construction. Both this file and `HamiltParams.py` were last touched in `cf93ba3` and are unchanged since, and the repository has full history (57 commits) — so **as committed, this script has never been runnable here**. If any rotor budget-comparison number in the manuscript is attributed to it, that attribution must be checked |
| **D11** | the Q-HAS arm is **not deterministic**: no RNG seed anywhere in V1's VQA chain (`AerSimulator` without `seed_simulator`; `Estimator`/`Sampler` at 256 shots) | `src/VQA/execute.py`, `runtime.py` | replaying fold `ot` with identical inputs gives phys **0.1345 vs 0.1940** stored (44 % swing), while the classical arm reproduces bit-exactly. Every Level-3 Q-HAS number — and **V1's own published closed-loop numbers** — is a single unreplicated draw. `--seed` cannot fix it: the randomness is inside V1 |
| **D12** | `pipeline` aborts on divergence and returns a **partial score with identical keys** to a completed run | `src/pipeline.py:499` | a crashed arm is indistinguishable from a finished one in the stored output, so a diverged *classical* arm reads as a Q-HAS victory. Fold `rotor` hit exactly this (classical aborted at step 208, t=0.2739, reported phys 1.1731). Detection added in T19; pre-registration §5 required the exclusion but nothing implemented it |
| **D13** | **train/test leak in the Level-3 protocol**: the QAOA arm's decision threshold is hard-coded at `0.14959824837662078`, a value fitted by `_run_classical_phase1` on **"KH + OT + Tearing + Rotor"** — all four classes, including the one being held out | `TrainHyperParam_v2.make_composite_objective:632`, reproduced at `study/v4/t15_level3_closed_loop.py:154` | the held-out class is *not* excluded from all tuning, contrary to what the pre-registration states. The classical arm re-tunes its threshold per fold on training classes only, so the leak is **asymmetric and favours Q-HAS**. Q-HAS loses anyway (18/18 completed runs), so the conclusion is conservative — but the protocol claim must be corrected |

**Scope of the signature drift — the reassuring half.** The drift does *not*
touch the production path. `src/pipeline.py` and every study entry point
(`phase0`, `phase3`, `phase4`, `phase5`) construct `PhysicalMapper` with the
current signature (`sigma`, `beta_curl`, `beta_xpoint`, `w_z_frac`). Only
the 6 stale tests (D6) and the orphaned `compare_rotor_budget.py` (D10) use
the removed `beta=`. **The simulations behind every V3/V4 number were
produced by code that runs as intended**; what is broken is one unused
analysis script and a set of tests that were never updated after a refactor.

D6 and D7 were found by running the V1 suite rather than assuming it green;
D7's substance came from taking its two failing physics assertions
seriously instead of dismissing them as stale tests.

---

## 4. Proposed claim structure for the manuscript

1. **Construction** (V1, unchanged) — a topology-aware variational circuit
   placed inside an AMR loop without asking a quantum computer to integrate
   MHD. Technically functional.
2. **In-distribution gains are marginal** (V1, unchanged) — 0.66 % composite,
   mixed physical endpoints, classical closer to DNS in all four scenarios.
3. **Open-loop transfer fails** (V3, unchanged) — leakage +0.399 F1, the
   learned selectors collapse under LOSO, the temporal channel is retired,
   the information cone is retired.
4. **NEW — the quantum component is inert** (Claims A–D). Three independent,
   cheap, exactly reproducible measurements. This is the *mechanism* the
   audit said was missing, and it **predicts** every V1 observation
   (0.66 %, 109 flips with 45 right / 64 wrong, mask asymmetry, classical
   winning on L2) without invoking any quantum effect.
5. **NEW — and it is not a bug we could fix** (D-bis, D-ter). The pipeline
   does discard almost all of its own coupling, via a Gaussian window whose
   support is anti-correlated with the coupling's (D-bis) — but restoring
   the coupling to full strength changes **no decision**, and the
   window-free V2 mapper, whose coupling is natively healthy, is **equally
   inert** (D-ter). The inertness is a property of the formulation at the
   deployed size, not of the implementation.
6. **NEW — closed loop, budget-matched** (Claim E) — the apparent
   advantage reverses; Q-HAS is Pareto-dominated. **State this with its
   two controls attached** (D11, D12): the Q-HAS arm is unseeded and
   varies run to run, and a diverged arm is invisible in the stored
   output. The direction survives both; the magnitudes carry the spread.
7. **NEW — methods corrections** (Claims F–G, defect register D1–D10).

**Rebuttals this forecloses.** Worth stating explicitly, because each is the
natural first objection and each already has its control:

| objection | answer |
|---|---|
| "your couplings are inert because your window destroys them" | D-ter: with the window neutralised (coupling O(25–155)) ablation still changes 0.0000 decisions |
| "the σ → ∞ trick is an artefact of your manipulation" | the V2 mapper has **no window by construction**, coupling natively O(2–4), and is equally inert — no manipulation involved |
| "you only tested the broken v1 mapper" | both mappers, both with stored artifacts (D9 fixed the collision that previously prevented this) |
| "the null could be a measurement-chain failure" | the `full` control returns exactly 0.0000 in every arm, and T18's tests include a positive control showing the instrument detects real changes |
| "the solver's low order explains the result" | Claim F is reported independently; the ablations are decisions on identical fields, unaffected by time-integration order |

**Recommended framing.** Not "the bottleneck is the transfer of learned
coefficient selection" (the current text) but: *at the deployed size there
is nothing to transfer* — the couplings are inert, the circuit does not
optimise its objective, and the decision is the classical score plus an
uninformative perturbation. The paper becomes a mechanistic falsification
study: here is how to place a variational circuit in an AMR loop, and here
is the rigorous account of why this construction does not do what it
appears to do.

**Sections to rewrite:** the topological attribution (causal reading falls,
Claim D); **the stated rationale for the Ising formulation** — the
multi-body coupling is the reason to reach for an Ising model at all, and
at the deployed size it does no work (D-bis, D-ter), so the motivation
paragraph cannot stand as written; the solver description (order 1,
Claim F); the asymmetry discussion (degeneracy, not equivariance, Claim G);
the conclusion (mechanism instead of "bottleneck = transfer").

**One attribution to verify on the `.tex` side:** if any rotor
budget-comparison number is credited to `src/compare_rotor_budget.py`, that
credit is wrong — the script raises `TypeError` at construction and, as
committed, has never been runnable in this repository (D10).

---

## 5. What is still pending

| item | cost | why it matters |
|---|---|---|
| Level-3 folds `kh`, `rotor`, `tearing` (+ their budget-matched arms) | ≈ 2 h each | takes Claim E from n = 1 to n = 4 |
| ≥ 5 physics seeds per fold | hours | protocol §1.1 target; currently 1 |
| Full Task-6 dynamic ground-truth campaign | ≈ 4.6 h | would let Level 2 use `d_i` targets |
| Hardware / noise / shots | — | audit: premature without a positive L3 |

Commands (resumable per Optuna **trial**, not just per fold — the campaign
survived five container reclamations this way):

```
# one fold per core, BLAS pinned to one thread each
for f in kh rotor tearing; do
  nohup bash study/v4/run_fold.sh "$f" > logs/v4/fold_$f.log 2>&1 &
done

python study/v4/level3_status.py             # read-only progress probe
python study/v4/t16_aggregate_v4.py          # master table, self-checking
python study/v4/t15c_fold_synthesis.py       # cross-fold, pre-registered rules
python study/v4/make_pareto_panel.py         # all folds, one figure
python study/v4/make_pareto_figure.py --fold kh   # or one figure per fold
```

Re-running a killed fold resumes from the last persisted trial with no
recomputation; `level3_status.py` opens the Optuna store read-only, so it is
safe to call while the folds run.

**Verdict on sufficiency.** Claims A–D (with D-bis/D-ter) and F–G are
finished science: they are deterministic, exactly reproducible, cover all
four scenarios at production resolution, and are covered by the test gate.
D-ter in particular closes the last escape route — the result is not
contingent on the implementation defect it uncovered. They alone support the
mechanistic falsification narrative. Claim E is the only one at n = 1; it is
strengthened by comparing against a frontier rather than a point, and by the
mechanism in A–D which predicts it. The paper can be written now, with
Claim E stated at its true scope and the remaining folds added as they
complete.
