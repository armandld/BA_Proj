# Does a local Ising–QAOA criterion improve adaptive mesh refinement for 2D MHD? A negative result

**Status:** draft. Author names, affiliations, and funding statement are not
filled in. Numbers are verified against the repository at commit
`6de979a9ffa2f6626463e399b30fd178e0a33ecf` (branch
`claude/project-evaluation-mit-preprint-t6f3hp`); each result below names the
script and test that reproduce it.

---

## Abstract

We test whether a local refinement decision derived from an Ising Hamiltonian
solved by QAOA improves block-structured adaptive mesh refinement (AMR) for a
2D magnetohydrodynamics (MHD) solver, compared with a cheap classical
threshold on the same input features. The question is narrow and empirical,
not a claim about computational complexity: the Hamiltonians used here (up to
18 qubits) are exactly diagonalizable classically in seconds. We ask only
whether a shallow-depth QAOA circuit, used as a decision rule, outperforms a
one-feature classical rule on this specific task.

It does not. Three independent lines of evidence close the question. First,
QAOA rarely reaches the exact ground state of its own Hamiltonian at the only
lattice size verified non-degenerate (18 qubits): a hit rate of 6.25%-15.6%
across circuit depths, against a required 100%. Second, and more decisively,
even solvers that do reach the exact optimum produce a *worse* refinement
decision than the classical rule: the correlation between optimization
quality and task performance is positive (Spearman ρ = +0.87 to +0.89 across
nine solvers, two independent mapper implementations), meaning the
Hamiltonian's true optimum systematically disagrees with the useful decision
more than a partially-optimized state does. Third, the ZZ/ZZZZ coupling terms
that were meant to encode neighbor information never improve the exact
decision at any lattice size tested (2 to 128 qubits) and degrade it once the
lattice is large enough for the couplings to matter at all.

We replicate the first two findings on real DNS trajectories, four canonical
instability scenarios, four Reynolds numbers, leave-one-scenario-out
cross-validation, and 95% bootstrap confidence intervals, without any
hyperparameter training. At this scale the classical threshold beats or ties
both QAOA and a trained gradient-boosted-tree baseline on every held-out
scenario, confidently on three of four (bootstrap CI on the paired F1 delta
excludes zero, p = 0.000). A full Optuna hyperparameter campaign was not run;
we report why, and show that the untrained starting point is not the source
of the failure.

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
coefficients (Section 2.4) — a degeneracy not identified in the earlier
project, and one that makes the 2×2 point uninformative about whether the
framework's optimisation or representation choices are sound, prior to any
question of scale. `dim = 3` (18 qubits) is the smallest resolution above
this degenerate point at which the exact ground state remains exhaustively
enumerable, and is used throughout as the certified
reference size. Scaling further, to `dim = 4` and `dim = 8`, does show the
coupling terms' relative influence growing with resolution, exactly as the
earlier report's further-work section anticipated — but in the opposite
direction: the gap between the full Hamiltonian and a bias-only version
widens from -0.033 F1 at `dim = 4` to -0.057 at `dim = 8` (Table 1),
rather than closing. The remainder of this report gives the full
measurement, on the certified size and, in Section 4.5, replicated at a
larger sample size directly on real DNS.

## 1. Introduction

### 1.1 Motivation

MHD instabilities — current sheets, vortices, shocks — are local: a cell's
need for refinement depends mainly on its immediate neighborhood. Ising
Hamiltonians defined on a lattice, with local bias terms and nearest-neighbor
couplings, are a natural encoding for a decision with that structure. QAOA is
a standard near-term algorithm for approximately solving such Hamiltonians.
The question we test is whether this combination — a QAOA-solved local Ising
Hamiltonian used as the AMR refinement criterion — outperforms a classical
threshold on the same underlying physical score.

### 1.2 What this is not

This is not a test of quantum computational advantage in the complexity
sense. At the lattice sizes used here (up to `dim = 8`, 128 qubits, and
principally `dim = 3`, 18 qubits) the Hamiltonians are exactly diagonalizable
on a classical machine in seconds to minutes. We make no claim, implicit or
explicit, that this problem is classically hard. The question is narrower:
does a shallow NISQ circuit (`reps = 2`, a few dozen classical optimizer
evaluations), used as a refinement decision rule, do better than an
inexpensive classical heuristic on this specific task? That is a legitimate
empirical question independent of classical tractability, and it is the only
one this work answers.

### 1.3 A prerequisite: verifying the implementation computes what it claims

A comparison between two decision rules is only informative if both are
implemented as documented. Before any of the results below were produced, the
codebase was audited systematically — five standard questions applied to
every module (what does this promise, what does it consume, does it fail
loudly when its assumptions break, do two paths that should agree still
agree, is a test that guards this claim actually able to fail) rather than by
line-by-line review. This found and closed more than 190 contract defects,
ranging from a sign convention silently flipped by a refactor to acceptance
criteria that were printed but never compared against a reference. We treat
this as a necessary precondition for the results reported here, not as a
separate contribution to highlight; a small number of defects remain open and
are documented in the repository, none of which change the verdicts below.

## 2. Methods

### 2.1 MHD solver and validation data

The reference solver uses fourth-order finite differences in space and
fourth-order Runge-Kutta in time on a periodic 2D grid. Four canonical
scenarios are used: a Harris current sheet (`harris_tearing`), a
Kelvin-Helmholtz shear layer (`kelvin_helmholtz`), an MHD rotor
(`mhd_rotor`), and the Orszag-Tang vortex (`orszag_tang`), each run at
Reynolds numbers 400, 800, 1200, and 1600. Direct numerical simulation (DNS)
trajectories from this solver are the ground truth against which every
refinement decision is scored.

### 2.2 The refinement decision as binary classification

The domain is partitioned into a `dim × dim` grid of patches. Each patch is
labeled "hard" or "not hard" by whether a DNS-measured local error exceeds a
percentile threshold computed per scenario. A refinement criterion is a rule
that predicts this label from locally available features, scored by F1
against the DNS-derived label.

### 2.3 Physical-to-Ising mapping

Each patch (edge, for the coupling terms) is mapped to a term in an Ising
Hamiltonian: a per-site bias (Z) built from a classical multi-indicator
instability score (vorticity, velocity divergence, current density, a
discontinuity estimator), a temporal-derivative phase (ψ) built from the
change in local stress flux between the current and previous snapshot, a
gradient coupling (ZZ) between adjacent sites, and a four-site plaquette
coupling (ZZZZ) meant to encode local circulation. Two mapper
implementations are used: V1, with eight free coefficients tuned by an
Optuna hyperparameter search, and V2, with fixed weights and zero free
parameters. Where a result is reported for only one mapper, it is stated.

### 2.4 QAOA

QAOA is run at `reps = 2` with 4096 shots per circuit evaluation and a
COBYLA optimizer budget of 60 evaluations, unless stated otherwise. All
Hamiltonian-exactness results (Sections 4.1-4.3) use `dim = 3` (18 qubits),
the only lattice size at which the exact ground state was verified
non-degenerate by exhaustive enumeration. At `dim = 2` (8 qubits) the exact
ground state is the trivial "refine everything" predictor on every tested
instance, so no solver comparison at that size can be informative about
optimization quality; this was previously mistaken for a result and is
corrected here (Section 4).

### 2.5 Classical baseline

The classical rule is a single threshold on the same physical score used as
the Ising bias term, fit by maximizing F1 on the training scenarios and
applied without adjustment to the held-out scenario (leave-one-scenario-out,
LOSO): four canonical scenarios, each held out once, the threshold refit each
time on the other three.

### 2.6 Gradient-boosted-tree baseline

A histogram gradient-boosted classifier is trained on nine features (the
same local indicators as the classical score, plus neighbor-cone features),
under the same LOSO discipline: trained on three scenarios, evaluated on the
one held out, refit per fold.

### 2.7 Statistics

Confidence intervals on paired differences between two decision rules (e.g.
QAOA F1 minus classical F1) are computed by a snapshot-level percentile
bootstrap: 1000 resamples of the held-out snapshots (not of individual grid
cells, which are not independent within a snapshot), 95% percentile
interval. A verdict for one rule beating another on a given fold requires
the interval to exclude zero; we report the interval and the one-sided
bootstrap probability alongside every such comparison, not just the point
estimate.

## 3. Hypotheses tested

| label | question |
|---|---|
| H0a | Does QAOA reach the exact ground state of its own Hamiltonian? |
| H0b | If a solver does reach it, does that produce a better refinement decision? |
| H2b | Does a more flexible model (gradient-boosted trees) beat the classical threshold? |
| H3 | Do the ZZ/ZZZZ coupling terms improve the decision over a bias-only model? |
| H1 | Are solver and numerical defects, on their own, sufficient to explain a failure? |
| H4 | Does the approach transfer to physical conditions outside the training set? |
| H5 | Is a failure explained by how the target label is specified rather than by the model? |

H0a and H0b together test whether the *optimization* step is the problem;
H3 tests whether the *representation* (the coupling terms) is the problem,
independently of whether it is solved well. H2b tests whether restricting the
model to a physically motivated Ising form, rather than a more flexible
learned model, is the bottleneck. H1, H4, and H5 are auxiliary axes: H1 asks
whether purely numerical defects (not the QAOA/representation choices) could
explain the failure on their own; H4 asks about generalization to unseen
physical regimes; H5 asks whether the way the target label is defined
biases the comparison.

## 4. Results

### 4.1 The coupling terms never help the exact decision, at any tested size

We separate two questions. First, do the coupling terms change any
decision at all, once the exact optimum is found by exhaustive (`dim = 2,
3`) or exhaustive-controlled greedy (`dim = 4, 8`) search? At `dim = 3`,
removing the ZZ term alone changes 6.9% of decisions, removing ZZZZ alone
changes 15.3%, and removing both changes 15.3% (ZZZZ dominates); at
`dim = 2` the couplings change 0% of decisions, because the exact ground
state is uniform regardless of the Hamiltonian at that size (Section 2.4) —
nothing can appear causal there. Second, and separately: when the couplings
do change a decision, is the changed decision better or worse against DNS
ground truth? Table 1 answers this by F1.

**Table 1. Exact-optimum F1 against DNS ground truth, full Hamiltonian vs.
bias-only (couplings zeroed), by lattice size.**

| `dim` | qubits | search | F1, full Hamiltonian | F1, bias only | F1, classical rule | coupling effect on F1 |
|---|---|---|---|---|---|---|
| 2 | 8 | exhaustive | 0.333 | 0.333 | 0.389 | +0.000 (degenerate; both below classical) |
| 3 | 18 | exhaustive | 0.405 | 0.451 | not comparable at this pooling | −0.046 |
| 4 | 32 | greedy, exhaustive-controlled | 0.520 | 0.552 | 0.552 | −0.033 |
| 8 | 128 | greedy | 0.592 | 0.648 | 0.648 | −0.057 |

The coupling terms are never associated with a higher F1 than the
bias-only version, at any size where the search is informative, and cost
0.033-0.057 in F1 once they are not numerically inert. At `dim = 4` and
`dim = 8` the bias-only Hamiltonian's F1 equals the classical rule's F1
exactly (0.5524 and 0.6481 respectively, to four digits): the best case
of this Ising formulation is to reproduce the classical threshold term for
term, not to beat it. F1 rises with `dim` for both the full and the
bias-only Hamiltonian alike (0.33 to 0.59 and 0.33 to 0.65) — this is the
refinement grid getting finer, identically for both arms, and should not
be read as an effect of the coupling terms or of QAOA. This directly
answers H3: the neighbor information the coupling terms were designed to
add does not help the decision, and actively hurts it once it is not
numerically inert.
(script: `study/h3_representation/h3_size_scan.py`,
`study/h3_representation/h3_term_ablation.py`; test:
`tests/study/test_t26_proxy_validation_surfaced.py`,
`tests/study/test_t13_dim3_couplings_not_inert.py`.)

### 4.2 QAOA rarely reaches its own Hamiltonian's exact optimum (H0a)

At `dim = 3`, comparing each solver's output against the exact ground state
on the same 32 instances: exhaustive search reaches it by construction
(100%); a greedy heuristic reaches it 84.4% of the time; simulated annealing
reaches it 59.4% (75.0% with a warm start); QAOA, across circuit depths
`p1`-`p6` and an increased-shots variant, reaches it 6.25%-15.6% of the time,
against the 100% a correct claim of optimality would require. Using the
production mapper (V1, the one an eventual hyperparameter campaign would
tune) instead of the parameter-free V2 gives the same qualitative result: 0
of 12 tested instances. (script:
`study/h0_selection/h0_optimiser_equivalence.py`; test:
`tests/study/test_h0_certified_dim3_contradicts_criterion.py`.)

### 4.3 Reaching the exact optimum makes the decision worse, not better (H0b)

Across a panel of nine solvers and solver variants — exact enumeration,
greedy search, simulated annealing (with and without a warm start), a
classical-score-derived initial state, and several QAOA circuit depths —
the Spearman correlation between a solver's distance from the exact
ground-state energy (`E_gap`) and its task F1 is **positive**: ρ = +0.87
(V2 mapper) and ρ = +0.891, p = 0.0013 (V1 mapper, the production mapper,
one measured point in its 9-dimensional hyperparameter space rather than a
full sweep). A positive correlation between energy gap and F1 means that,
as a trend across this panel, solvers *further* from the exact optimum
score *better*. The panel includes exact enumeration itself, at `E_gap = 0`
by construction: for the V1 mapper it scores F1 = 0.437, below several
solvers that do not reach the optimum (the best in that panel, `qaoa_p1`,
scores F1 = 0.519 while never once landing on the exact optimum across 12
tested instances). The exact optimum is not exempt from the trend — it
sits on it, at the losing end. This is the most direct finding in this
work, because it does not depend on QAOA specifically: it holds for a panel
that already contains the exact solution, so a hypothetical perfect
optimizer would not improve on it, and it holds independently for both
mapper implementations, so it is not an artifact of the parameter-free
mapper's particular weight choices. (Same script and test references as
4.2.)

### 4.4 H2b: a more flexible model does not close the gap either

Restricting the earlier question to model form rather than optimizer: does a
gradient-boosted-tree classifier, unconstrained by the physically motivated
Ising structure, beat the classical threshold under the same LOSO protocol?
It does not, at the scale reported in Section 4.5: across four held-out
scenarios the GBT ties the classical threshold on two, loses on two, and
never wins outright. Restricting the model family further, to an Ising form,
is therefore not obviously the bottleneck either — the classical threshold
is hard to beat with the features available, by any model tested.

### 4.5 Confirmatory replication on real DNS, four Reynolds numbers, with bootstrap confidence intervals

Sections 4.2-4.4 use reference hyperparameters and a single Reynolds number
per instance. To test whether these findings hold at larger scale, without
any hyperparameter training, we ran the parameter-free mapper (V2) with real
QAOA against real DNS across all four canonical scenarios and all four
available Reynolds numbers (400, 800, 1200, 1600), 10 snapshots per
(scenario, Reynolds number) pair, giving 40 held-out snapshots per
leave-one-scenario-out fold. For each held-out snapshot we scored: the
classical threshold, a trained GBT, QAOA on the full Hamiltonian, the exact
ground state of the same Hamiltonian, and both QAOA and the exact optimum
with the coupling terms zeroed (bias-only). Every confidence interval below
is a 1000-resample snapshot-level percentile bootstrap on the paired F1
delta between two arms (Section 2.7).

**Table 2. F1 by held-out scenario, four decision rules, n = 40
snapshots/fold.**

| held-out scenario | classical | GBT | QAOA (full) | exact (full) | QAOA (bias only) | exact (bias only) |
|---|---|---|---|---|---|---|
| harris_tearing | 0.667 | 0.667 | 0.667 | 0.500 | 0.667 | 0.667 |
| kelvin_helmholtz | 0.571 | 0.571 | 0.423 | 0.421 | 0.422 | 0.421 |
| mhd_rotor | 0.690 | 0.456 | 0.476 | 0.351 | 0.485 | 0.569 |
| orszag_tang | 0.379 | 0.286 | 0.220 | 0.349 | 0.374 | 0.349 |
| mean | **0.577** | **0.495** | **0.446** | **0.405** | 0.487 | 0.501 |

**QAOA does not beat the classical threshold on any fold, at 95% confidence
on three of four.** The bootstrap interval on (QAOA F1 minus classical F1)
is strictly negative on `kelvin_helmholtz` [-0.154, -0.143], `mhd_rotor`
[-0.337, -0.119], and `orszag_tang` [-0.246, -0.079] (bootstrap p = 0.000 on
each). On `harris_tearing` the interval is exactly [0.000, 0.000]: the
classical, GBT, and QAOA rules produce the identical F1 on every one of the
40 held-out snapshots in that fold, which is the most parsimonious
explanation for a zero-width bootstrap interval, though we have not verified
cell-by-cell agreement directly. **The gradient-boosted tree does no
better**: it ties the classical rule exactly on `harris_tearing` and
`kelvin_helmholtz`, and loses on `mhd_rotor` and `orszag_tang`; it never
strictly beats the classical threshold on any fold at this scale.

H0a and H0b replicate without change of direction: agreement between QAOA
and the exact optimum on the full Hamiltonian is again highly
scenario-dependent (10.6% to 99.4%), and QAOA F1 exceeds the exact optimum's
F1, with a 95% interval excluding zero, on three of four folds
(`harris_tearing`, `kelvin_helmholtz`, `mhd_rotor`); `orszag_tang` is again
the one exception, with the interval excluding zero in the *opposite*
direction. For the exact optimum, the full Hamiltonian never beats the
bias-only version at this scale either (0 of 4 folds, two exact ties, two
confidently negative intervals) — the same conclusion as Table 1, now with
confidence intervals. For QAOA specifically, the smaller-sample result
reported first (Reynolds 400 only, 5 snapshots/fold, no confidence
intervals) had suggested the coupling terms gave a small benefit; at this
larger scale that does not hold up (one near-exact tie, one statistically
inconclusive fold, one confidently negative fold) — QAOA's coupling
sensitivity converges toward the exact optimum's reading once the sample is
large enough to tell the difference from noise. (script:
`study/h2b_prediction/h2b_v2_hamiltonian_vs_gbt_loso.py --re 400 800 1200
1600`; test: `tests/study/test_h2b_v2_hamiltonian_vs_gbt_loso_multire.py`.)

### 4.6 H5: task specification matters for half the canonical panel

The refinement label used above is static: an instantaneous measure of
local non-smoothness. A dynamic alternative — the actual error incurred by
coarsening one patch, measured at the physical time a perturbation takes to
cross that patch — agrees with the static label almost perfectly on two of
the four canonical scenarios (`harris_tearing`, `kelvin_helmholtz`, Spearman
ρ ≈ 1.0) and diverges meaningfully on the other two (`mhd_rotor`,
`orszag_tang`, ρ as low as 0.66 at some snapshots). The static label is
therefore not obviously wrong, but it is not obviously sufficient either;
this is a secondary, scenario-dependent effect, not a primary explanation
for the results in Section 4.5, all of which use the static label
consistently across arms.

### 4.7 H1 and H4 remain open

Numerical and solver defects found during the audit (Section 1.3) matter —
several changed which decision a criterion made — but nothing in this work
isolates whether such defects, on their own, would be sufficient to explain
the negative result absent the optimization and representation problems in
Sections 4.2-4.5; we report H1 as partial, not closed. H4, whether a
trained system would transfer to physical conditions outside its training
set, has no dedicated experiment in this work: the closed-loop confirmatory
protocol that would answer it requires training across held-out physical
regimes and was not run, for the same cost reasons given in Section 5. We
report H4 as an open conjecture, not as evidence either way.

## 5. Why no full hyperparameter campaign was run

Two experiments were designed into the protocol but not executed: a full
Optuna re-optimization of the eight free V1 mapper coefficients, and an
eight-fold matched-budget closed-loop confirmatory comparison. Both are
implemented and were smoke-tested; neither was run to completion. Directly
measured, not assumed, on the hardware available for this work: the
re-optimization campaign alone requires on the order of several weeks of
continuous compute, and the closed-loop confirmatory campaign a comparable
order of magnitude. We judged this cost unjustified given that Section 4.3
already shows the *reference* starting point — the one such a campaign would
tune from — sits in the same pathological regime the campaign would be run
to test (better optimization correlates with worse decisions at the
reference point already), and that the same three hypotheses (H0a, H0b, H3)
were independently re-measurable without any training, directly on real DNS,
at a larger sample size than the campaign's own per-fold budget would have
given (Section 4.5). We consider that substitution reasonable but not
equivalent: it does not rule out that some point in the 8-dimensional
hyperparameter space reverses the sign of ρ(E_gap, F1), only that the
untrained starting point does not.

A closed-loop, matched-budget pilot using four of the eight planned folds
(the other four require the same unrun campaign) is directionally
consistent with Section 4.5: on the four folds available, Q-HAS is
1.3x-2.7x worse than a matched-budget classical rule at equal compute cost.
We report this as directionally consistent, not as evidence: four of eight
folds is short of the pre-registered decision rule's threshold, and we do
not treat it as a confirmatory result.

## 6. Discussion

Three independent lines of evidence close the question posed in the
abstract, along two separate angles. H0b shows that the *optimization*
angle cannot be the fix: even a solver that reaches the Hamiltonian's exact
ground state produces a worse decision than a partially optimized one, so
improving QAOA's convergence would not help. H3 shows that the
*representation* angle cannot be the fix either: the coupling terms that
were meant to add neighbor information never improve the exact decision,
and hurt it once they are not numerically inert, independent of whether any
solver reaches that optimum. These are separate arguments — one about how
well the Hamiltonian is solved, one about what the Hamiltonian encodes —
and they agree. Section 4.4 adds a third angle: even discarding the
Ising-specific form entirely in favor of a more flexible learned model
(GBT) does not close the gap, at the same real-DNS scale.

A caveat on the GBT baseline used for H2b independently of the results
above: on one component comparison not used to decide the primary verdict, a
GBT model trained on classical-score-derived features showed a
scenario-dependent sign flip between the score and the label that inflates
the apparent gap between the physical baseline and the learned model on one
specific scenario; this is documented, does not change the H2b verdict
(which does not rest on that single comparison), and is not resolved.

An earlier version of this analysis concluded a more blanket rejection of
the optimization hypothesis (H0), measured entirely at `dim = 2`. At that
size the exact ground state is the trivial "refine everything" predictor on
every tested instance (Section 2.4), so every solver trivially reaches it
and no optimization comparison there is informative; that conclusion rested
on a degenerate case. The results reported here separate H0a from H0b and
measure both at the only size independently verified non-degenerate, which
is a materially different and better-supported basis for the same
qualitative conclusion.

## 7. Limitations

**Single classical reference rule.** The comparison throughout is against
one feature, one threshold, refit per fold. We do not claim this is the best
possible classical AMR criterion, only that it is the cheap baseline the
Ising/QAOA approach would need to beat to be worth its cost, and that it was
not beaten.

**No full hyperparameter campaign.** Discussed in Section 5. The reference
hyperparameters used throughout are not the output of the tuning process a
deployed version of this approach would use; we give direct evidence
(Section 4.3) that this starting point is not favorably or unfavorably
biased relative to the pathology being measured, but a full sweep of the
9-dimensional hyperparameter space was not performed for the production
mapper (V1); one point was.

**QAOA run-to-run variance.** QAOA is not deterministic between calls
(dispersion of 0.18-0.36 in the displaced-probability measure used to
characterize it, measured directly). Conclusions based on rank (does one
arm beat another) are supported; conclusions that would depend on an exact
point estimate of a single QAOA run are not, and we have avoided stating
any.

**Shared numerical order-of-accuracy limitations.** Both arms share the
same underlying solver and are subject to the same finite-difference order
drop under certain refinement configurations; this affects both arms
identically and is not expected to bias the comparison between them, but it
bounds the absolute accuracy either arm can achieve.

**The closed-loop confirmatory pilot is underpowered.** Section 5,
final paragraph.

## 8. Conclusion

Restated plainly: a local Ising Hamiltonian solved by QAOA and used as an
adaptive-mesh-refinement decision rule does not outperform a classical
threshold on the same physical score, for 2D MHD instability detection, at
every scale we tested this at. The failure closes along two independent
angles — optimization (H0a, H0b) and representation (H3) — that agree with
each other, and a third check (H2b) shows that relaxing the model to a more
flexible learned form does not recover the gap either. A large-scale,
multi-Reynolds-number, bootstrap-confidence-interval replication on real DNS
trajectories, run without any hyperparameter training, reproduces the same
qualitative verdict and adds a further one not visible at smaller scale: at
this scale, the classical threshold beats or ties both the quantum and the
classical-learned alternative on every fold tested. We report this as a
negative result obtained with the same measurement rigor the project would
have applied to a positive one, per its stated evaluation standard, not as
an absence of evidence.

## 9. Code and data availability

All code, data-generation scripts, and the exact commands that reproduce
every number in this manuscript are in the repository at commit
`6de979a9ffa2f6626463e399b30fd178e0a33ecf`. Each numbered result above names
the script that produces it and the automated test that pins its value
against regression. Selected reproduction commands:

```bash
# Table 1 (exact-optimum coupling ablation, dim = 2..8)
python study/h3_representation/h3_size_scan.py

# Section 4.2-4.3 (H0a, H0b; nine solvers, dim = 3)
python study/h0_selection/h0_optimiser_equivalence.py

# Table 2 and Section 4.5 (confirmatory replication, real DNS, 4 Re values)
python study/h2b_prediction/h2b_v2_hamiltonian_vs_gbt_loso.py \
    --re 400 800 1200 1600 --n-snaps 10

# regression tests locking every number above to the current codebase
pytest tests/study/test_h0_certified_dim3_contradicts_criterion.py \
       tests/study/test_t13_dim3_couplings_not_inert.py \
       tests/study/test_t26_proxy_validation_surfaced.py \
       tests/study/test_h2b_v2_hamiltonian_vs_gbt_loso_multire.py -v
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
it. Sections 2.3-2.4 above describe the mapping only at the level needed to
read the results.

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

**Trainable parameters — a discrepancy between the originating report and
this codebase.** The originating report's own architecture chapter lists
five trainable hyperparameters: the encoding steepness (beta), the
uncertainty-band width (sigma), the two per-term contrast sensitivities
(beta_curl, beta_xpoint), and the Z-bias fraction (w_z_frac). It freezes
three more at values fixed by that report's own training: gamma_hydro = 2.0,
gamma_mag = 0.5, kappa = 10.0 (and threshold_amr = 0.1496, frozen from
classical training rather than from this Hamiltonian). The present
codebase's search space (`src/train_hyperparams.py`, `SEARCH_SPACE`;
`docs/DEFAUTS.md`, D-22) instead treats all eight of beta, w_z_frac, sigma,
beta_curl, beta_xpoint, gamma_hydro, gamma_mag, and kappa as free — wider
than what the originating report trained. The seed the codebase uses to
reproduce the originating report's own result (`train_hyperparams.py`,
`PHASE1_SEED_GRID`) still carries that report's exact frozen values —
gamma_hydro=2.0, gamma_mag=0.5, kappa=10.0 — alongside its trained values,
beta=0.7, sigma=0.10, beta_curl=0.1, beta_xpoint=0.1, w_z_frac=500.0. So the
two accounts agree on what was actually trained and measured; they disagree
only on how wide a future re-optimization should search, and no number in
this report depends on that wider search, since no hyperparameter campaign
was run under either scheme (Section 5).

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

This is the architecture tested in Section 4: the Hamiltonian-exactness
results (H0a, Section 4.2) ask whether QAOA's evolution actually reaches
this H_struct's ground state; the decision-quality results (H0b,
Section 4.3) ask whether that ground state, when reached, is the better
refinement decision; and the ablations (H3, throughout Section 4) turn the
ZZ and ZZZZ terms above off one at a time to ask whether the coupling
structure earns its added circuit depth.
