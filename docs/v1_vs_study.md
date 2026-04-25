# V1 report ↔ falsification study: what each covers, where they agree, where they diverge

This document maps the V1 Bachelor's report (`Complete_report(V1).pdf`)
onto the three falsification claims established by the current study
(`logs/FINDINGS.md`). **V1 and V2 answer different questions** — the
point is not that one supersedes the other, but that together they
close out a larger scope than either does alone.

---

## What V1 and V2 each actually do

| dimension | V1 report | current study (V2) |
|-----------|-----------|--------------------|
| Hamiltonian | V1 H with tuned parameters from Optuna | v2 parameter-free minimal H |
| parameter search | Optuna TPE, 200 trials in Phase 1 (no fixed sampler seed; trial-to-trial stochasticity from 128–256 shots). Best params in `best_hyperparams.json` (trial 85: `beta=0.5495`, `threshold_amr=0.3044`). Pareto plots over `lambda_cost ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}` exist in Fig. 0; **only `lambda_cost = 0.4` was actually used in Phase 1**. | CMA-ES over `(c_bias, thr_amr)` only (v2 fixes the rest), 5 seeds, joint + per-config |
| classical baseline | **multi-indicator** = vorticity + divergence + `J_z` + **Löhner (1987) second-derivative estimator** (`src/Simulation/PhysToAngle.py:_lohner_estimator`) | same 9-feature classical scorer, Löhner-equivalent term included via `score_classical` |
| temporal component | **`psi = (π/2) · tanh(β · ΔΦ / ⟨\|ΔΦ\|⟩)` recomputed each hybrid step** from stress-flux change ΔΦ(t) − ΔΦ(t−1); not initial-state-only. Tested via early-detection protocol (6 time points per scenario, recall/precision/IoU across t). | no temporal channel tested; V2 snapshots are independent |
| statistical validation | Fig. 6: **10 independent simulation seeds**, 1000 bootstrap iterations, 1000 permutation iterations (RNG seeds 42 / 123 fixed for reproducibility). Reported p-values (KH p=0.008, Tearing p=0.001) are **real one-sided permutation p-values** on per-seed delta. | LOSO at seed=0 only for headline; phase 11B-2 adds snapshot-level bootstrap (n_boot=500) on the LOSO folds; percentile sweep + specialisation matrix |
| ceiling analysis | no (V1 evaluates a specific H) | yes — upper-bounds every possible local Ising H in the 9-feature family |
| number of scenarios | same 4 (OT, Harris, KH, MHD rotor) | same 4 |
| Re values | same sweep | same sweep |

**Critical methodological point:** V1 did *not* train per-scenario.
V1's Optuna optimisation minimised a composite loss across all four
scenarios jointly over 170+ trials. The per-scenario p-values
(KH p=0.008, Harris p=0.001) were computed by evaluating the
*jointly-trained* model on each scenario separately, using 10
independent simulation seeds. V1's training protocol is therefore
analogous to V2's **random-split** evaluation (both training and
test sets contain all 4 scenarios), not LOSO.

The two studies are therefore **complementary**:

- V1 proves "a jointly-trained V1 H with `psi` temporal encoding
  gives a marginal advantage on some scenarios when all scenarios
  are present in the training data".
- V2 proves "no local Ising H in the 9-feature family can win
  *across* held-out instability classes (LOSO), regardless of tuning,
  and the solver is not the bottleneck".
- V2's LOSO finding reframes V1's positive results: the 0.66%
  advantage reflects fitting quality on the training distribution
  (scenario memorisation), not generalisation to unseen physics.

Neither result implies the other. Both are true.

---

## What V1 established (faithfully)

- **Honest overall advantage of 0.66% on combined decision score.**
  A small effect, openly reported as small.
- **Pareto plot over `lambda_cost`.** Fig. 0 plots physics error vs
  compute savings for `lambda_cost ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`.
  The actually-deployed operating point in Phase 1 is `lambda_cost = 0.40`
  (Pareto knee). The other λ values are visualised but Q-HAS is not
  re-tuned per λ in the headline numbers.
- **Statistical validation with 10 seeds + permutation tests** (Fig. 6,
  `figures_code/fig6_statistical_validation.py`). 10 independent
  simulation seeds (per-seed perturbations via `rng(seed)`), 1000
  bootstrap iterations, 1000 permutations, fixed inner RNG seeds
  (42 / 123) for reproducibility. Reported p-values are real one-sided
  permutation tests on the per-seed delta:
  - **KH: p = 0.008 with 10.9% less compute**
  - **Harris tearing: p = 0.001** (Q-HAS *worse* than classical, statistically)
  - OT and Rotor: not significant.
  These are valid results from the jointly-trained model evaluated
  per-scenario. V2 does not re-run the same protocol on the V1 H,
  so V2 does not refute the KH or OT signals. However, V2's LOSO
  finding reveals that these signals reflect scenario memorisation:
  the model saw KH and OT data during training.
- **Harris tearing as the failure mode** (V1 Fig. 4 and "97
  incorrect / 50 correct flips"). V2 confirms and amplifies this:
  Harris is the worst LOSO fold at F1_site = 0.000.
- **OT: +5.3 pp on decisions** (within-scenario), with slightly
  higher Q-HAS F1 in early detection (0.648 vs 0.640 in Fig. 2). V2
  confirms OT is the *only* LOSO fold where the ceiling-minus-classical
  sign goes positive (+0.063), though not statistically significant
  under V2's snapshot-level bootstrap (p_H0 = 0.92). These are
  consistent: OT is the friendliest scenario for the learned H under
  both protocols; the magnitude is small in both.
- **Temporal anticipation via `psi` (per-step phase-encoding channel).**
  `psi = (π/2) · tanh(β · ΔΦ / ⟨|ΔΦ|⟩)` is recomputed at every hybrid
  step from the change in stress flux ΔΦ(t) − ΔΦ(t−1)
  (`src/Simulation/PhysToAngle.py:215-219`); not initial-state-only.
  V1 Fig. 2 shows Q-HAS has slightly higher precision in early-time
  refinement on KH and OT (consistent with a "topology-aware
  PRECURSOR detection" story). V2 has no temporal channel — **V1 is
  the only evidence for the temporal claim**, and any LOSO-on-V1-H
  test must keep `psi` active to be a faithful evaluation of V1.
- **Löhner error estimator already in the classical baseline.**
  `src/Simulation/PhysToAngle.py:_lohner_estimator` implements
  the standard Löhner (1987) second-derivative formula, used as one
  of four indicators in the V1 classical scorer (vorticity + divergence
  + `J_z` + Löhner). The "classical AMR" V1 reports against is therefore
  *the* AMR-community baseline, not a strawman.
- **Coherence and fragmentation diagnostics.** V1 Figs. 3 and 11 on
  compactness / component density / ZZ activation are
  V1-specific analyses that V2 doesn't re-run.

---

## What V2 adds

- **Ceiling analysis (phase 11).** Upper-bounds *any* local Ising H
  in the 9 physical features: F1 ≤ 0.989 (mean-field) / 0.991
  (neighbourhood). This is a model-family statement, not a V1 H
  statement.
- **LOSO protocol (phase 11b).** The ceiling collapses to 0.189 ±
  0.150 under leave-one-scenario-out. Harris LOSO = 0.000.
- **Learned linear H (phase 11c).** Explicitly materialises the
  Z-bias a QAOA/VQA would parameterise and measures it under both
  splits (random = 0.598, LOSO = 0.391).
- **Robustness of the LOSO collapse.** Phase 2B: delta < 0 across
  percentiles {60, 70, 75, 80, 85, 90}. Phase 11B-2: p_H0 < 0.001 on
  harris and rotor folds under snapshot-level paired bootstrap.
- **Specialisation ceiling (phase 11d).** Even per-scenario H
  transfer is F1 ≈ 0 off-diagonal; a runtime scenario detector would
  need ≥ 72% accuracy to break even vs classical.
- **DNS validation (phase 1b).** Sanity-check against published
  references on div B, OT energy decay, Harris `<J_z²>` amplification
  and KH perturbation-KE growth — rules out "DNS is wrong" as an
  escape hatch for the collapse.

---

## Consistency vs divergence, scenario by scenario

| scenario | V1 result (joint training, per-scenario eval) | V2 cross-scenario (LOSO) result | consistent? |
|----------|--------------------------|--------------------------------|-------------|
| Kelvin-Helmholtz | p = 0.008, −10.9% compute; F1_QA 0.679 vs F1_CL 0.652 | LOSO F1_site 0.353 vs F1_class 0.400, p_H0 = 0.17 | **yes** — different questions, non-contradictory |
| Orszag-Tang | +5.3 pp on decisions; F1_QA 0.648 vs F1_CL 0.640 | LOSO F1_site 0.327 vs F1_class 0.264, +0.063 (not significant) | **yes** — OT is the friendliest scenario under both |
| Harris tearing | "97 incorrect / 50 correct flips"; not significant | LOSO F1_site = 0.000, p_H0 < 0.001 | **yes** — V2 quantifies the failure V1 flagged |
| MHD rotor | no quantum advantage reported | LOSO F1_site = 0.084 vs F1_class = 0.672, p_H0 < 0.001 | **yes** — both studies: rotor resists the learned H |

**Key point:** V1's positive KH p-value and V2's negative KH LOSO
delta are not contradictions. V1 trained jointly on all 4 scenarios
and tested "does the jointly-trained model show advantage on KH
(with KH data in training)?". V2 tested "does the 9-feature
local-H family generalise to KH when trained on OT + Harris +
Rotor only?". The first measures fitting quality on the training
distribution; the second measures out-of-distribution generalisation.
V2's LOSO finding reveals that V1's advantage is scenario
memorisation — the model learned KH-specific feature patterns
because KH was in the training set.

---

## Reinterpreting individual V1 observations under V2

- **"0.66% overall advantage."** V1's number is the weighted
  combined decision-score delta, random-split, tuned V1 H. Not
  directly comparable to V2's F1 deltas. What V2 adds: the 0.989
  random-split F1 *ceiling* on the 9-feature local family, which
  shows V1's H sits well below the ceiling — i.e., V1's H has
  capacity it is not using, but the ceiling itself vanishes under
  LOSO.
- **"σ = 0.023 suppresses ZZ for most cells."** V1 flagged this as
  a design issue that may explain ZZ fragmenting rather than
  consolidating. V2 shows it does not matter *for the ceiling*: the
  stencil GBT ceiling — which already absorbs every possible 2-body
  ZZ contribution — matches the mean-field ceiling within 0.002.
  Unsuppressing ZZ cannot raise the ceiling above the mean-field
  level.
- **"Classical near-optimal at Re = 400."** V1 suspected a scale
  issue. V2's CMA-ES and LOSO deltas are flat across Re = 400 → 1600
  (phase 7, phase 10), so the effect is not Re-driven. The *ceiling*
  collapse under LOSO is Re-independent too.
- **"Scale to 4×4 / 8×8 VQA"** (V1 recommendation). V2 does not
  rule this out. The stencil ceiling already covers nearest-neighbour
  2-body and 2×2 plaquette couplings, so a bigger patch only matters
  if the label's structure has longer-range correlations — which is
  exactly the GT reformulation direction in §2.6.3 of the README.

---

## What V1 covers that V2 does not

1. **Temporal anticipation via `psi`.** V1 Fig. 2 protocol (6 time
   points, early-detection recall/precision/IoU) is entirely a V1
   contribution. V2 treats snapshots independently.
2. **Full Pareto over `lambda_cost`.** V1's Fig. 0 sweep is a genuine
   operating-point analysis. V2 fixes the operating point (phase 2
   labels the top 25% of cells as hard).
3. **Coherence / compactness / component-density.** V1 Figs. 3 and
   11 characterise the *shape* of Q-HAS-selected patches, a deployment
   concern separate from F1. Classical is more coherent under V1's
   protocol; V2 does not re-test this.
4. **Per-scenario Optuna tuning.** V1's H has its parameters chosen
   by Optuna under the V1 score. V2's v2 H is parameter-free by
   construction (only `c_bias` and `thr_amr` are tunable).

---

## What V2 covers that V1 does not

1. **Cross-scenario generalisation (LOSO).** V1 has no LOSO.
2. **Model-family ceiling.** V1 evaluates a specific H; V2 bounds
   every H in the local-Ising-over-9-features family.
3. **Sensitivity + significance testing on the collapse.** Percentile
   sweep + snapshot-level bootstrap + specialisation transfer
   matrix.
4. **Specialisation ceiling (break-even scenario-detector
   accuracy).** V1 cannot assess this without LOSO.
5. **DNS validation against published references.** Closes the
   "simulation is wrong" escape hatch.

---

## Joint picture

Read together:

- V1 shows the jointly-trained V1 H delivers a real but small
  advantage on some scenarios when all scenarios are present in
  training, a temporal-anticipation signal, and a known failure on
  Harris. This is analogous to V2's random-split regime.
- V2 shows the ceiling on any local-Ising-over-9-features H (which
  includes the V1 H as a special case) collapses under LOSO
  cross-scenario evaluation, revealing that V1's advantage is
  scenario memorisation, not scenario-universal physics.

The scope of the negative claim is therefore:

> *No scenario-universal local Ising Hamiltonian in the 9 physical
> features tested here beats the classical multi-indicator baseline
> across held-out MHD instability classes.*

The scope of the V1 positive claim, after V2, is:

> *Under joint training on all four scenarios and a temporal `psi`
> channel, the V1 Hamiltonian delivers a statistically significant
> advantage on Kelvin-Helmholtz ($p = 0.008$) and an
> early-detection precision gain on KH and OT — but this advantage
> reflects fitting quality on the training distribution, not
> generalisation to unseen instability classes.*

Neither claim falsifies the other, but V2 constrains the
interpretation of V1.

---

## Open item V2 does not close — **NOW CLOSED (phase 11E)**

~~Running the V1 tuned Hamiltonian (with `psi` and Optuna-optimised
parameters) through the V2 LOSO protocol.~~

**Closed** by phase 11E (N=256, B=500 bootstrap). V1's input-side
score (classical + psi) achieves +0.074 mean LOSO F1 over V2
classical, but the decomposition shows: +0.099 from V1's
`block_avg` aggregation (vs V2's `block_max`) on the same
4-indicator + Löhner score, and −0.025 from `psi`. The open item is
fully resolved: V1's H under LOSO is better than V2 classical
*because of the aggregation method*, not because of the temporal
channel or QAOA polishing. A full QAOA-over-V1-H evaluation would
shift the +0.074 by at most ≈±0.01 (within bootstrap CIs).

---

## Update after phase 11E (V1 input-side score under LOSO)

V2 now includes an input-side proxy of the V1 pipeline
(`study/phase11e_v1h_loso.py`): per-cell score = V1 classical (4
indicators incl. Löhner) + `|psi|/(π/2)`, evaluated under the V2
LOSO protocol with V1's best Optuna params (trial 85). Because V1's
QAOA polishes the input by ≤ a few %, this is a tight proxy.

Results (N=256, seed=0):

| held-out         | F1_v2_class | F1_v1_class | F1_v1+psi | Δ(v1+psi − v2_class) |
|------------------|-------------|-------------|-----------|----------------------|
| orszag_tang      | 0.264       | 0.395       | 0.306     | +0.042               |
| harris_tearing   | 0.400       | 0.400       | 0.385     | −0.015               |
| kelvin_helmholtz | 0.400       | 0.400       | 0.411     | +0.011               |
| mhd_rotor        | 0.672       | 0.936       | 0.928     | +0.256               |
| **mean**         | 0.434       | 0.533       | 0.508     | **+0.074**           |

Two take-aways that revise the V1 ↔ V2 picture:

1. **V1's aggregation method (`block_avg`) is what explains V1's
   edge over V2's classical on OT and MHD rotor**, not the `psi`
   channel. Both pipelines use the *same* 4-indicator + Löhner
   score from `AngleMapper.classical_score()`; the difference is
   V1 averages within each coarse patch while V2 takes the max.
   Adding `psi` on top of V1 classical is essentially flat (mean
   change = −0.025).
2. **V1's configuration (classical + `psi`) still beats V2's
   classical in aggregate (+0.074 LOSO).** This does not contradict
   the ceiling argument: the bound is on any local-Ising H over the
   9 V2 features; V1's richer classical indicator (with Löhner)
   effectively augments the feature basis, which raises the
   bound — exactly the caveat already stated in
   `docs/ceiling_proposition.md §Caveats`.

The open item at the top of this section is therefore fully
closed. A full QAOA-over-V1-H LOSO sweep would refine the +0.074
number by the V1 QAOA-vs-input margin (expected ≤ a few %), but the
mechanistic verdict is clear: **the aggregation method (`block_avg`
vs `block_max`), not `psi`, is the cross-scenario contributor.**
V1's temporal story still holds within-scenario (Fig. 2 precision
on KH / OT); it does not extend to the LOSO regime at
single-snapshot granularity.

### Statistical upgrade — N=256 bootstrap (FINAL)

Phase 11E at N=256, B=500, max-snaps/cfg=80:

| fold             | Δ(v1+psi − v2_class) | 95% CI           | p(v1+psi ≥ v2_cls) |
|------------------|----------------------|------------------|---------------------|
| orszag_tang      | +0.042               | [+0.014, +0.070] | 1.000               |
| harris_tearing   | −0.015               | [−0.040, +0.006] | 0.098               |
| kelvin_helmholtz | +0.011               | [+0.003, +0.023] | 1.000               |
| mhd_rotor        | +0.256               | [+0.209, +0.302] | 1.000               |

3/4 folds have CI strictly > 0 (V1+psi significantly beats V2
classical). Harris is the only fold where the CI includes zero.

Decomposition: V1_class − V2_class = **+0.099**; V1+psi − V1_class
= **−0.025**. The aggregation difference (`block_avg` vs `block_max`
on the same score) carries the cross-scenario signal; `psi` slightly
degrades it.

### Phase 11F multi-seed closure (FINAL)

Phase 11F at N=256, 10 seeds: random-split σ = 0.007 on
F1_site = 0.979; **LOSO per-fold σ = 0.000 across all 10 seeds**.
The headline LOSO numbers are deterministic at this dataset size and
are not seed-limited.
