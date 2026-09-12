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

[Not filled in — this section intentionally left for the authors to
populate with prior work on QAOA, Ising encodings for classification tasks,
and classical/ML-based adaptive mesh refinement.]
