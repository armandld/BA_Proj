# Q-HAS Falsification Study — Findings Summary

All numbers are at N=256, 4 scenarios, Re ∈ {400, 800, 1200, 1600},
dim=4, seed=0 unless otherwise noted. Reproducible from the `.npz`
outputs under `study/results/` and the logs under `logs/`.

**Methodological note (v3 update).** The `fit_eval` threshold is now
fitted on *training* predictions, not validation — eliminating the
mild data leakage present in prior runs. Phase 11D now uses GBT
(not logistic regression) for a tighter specialisation ceiling. All
headline numbers below reflect the N=256 rerun with corrected code.

---

## The Question

Can a classical AMR indicator be beaten by a **local Ising Hamiltonian**
(the class of cost functions that QAOA, SA, D-Wave, and any VQA-driven
refinement scheme ultimately optimise) for identifying hard-to-coarsen
MHD patches across a range of instability classes?

Equivalent phrasing: does the Q-HAS framework have an in-principle
ceiling above the classical multi-indicator baseline?

---

## The Answer: **No**, under strict cross-scenario evaluation.

### Claim A — QAOA ≤ exact diagonalisation ≤ classical on the v2 Hamiltonian

Phases 4, 5, 6, 7.

| method                          | F1    | source          |
|---------------------------------|-------|-----------------|
| Classical multi-indicator       | 0.475 | phase 11        |
| Simulated annealing on v2 H     | 0.336 | phase 7 overall |
| QAOA (reps=2, state-vector)     | ≈ SA  | phase 5         |
| Exact diagonalisation (dim=2)   | ≈ SA  | phase 4         |

**SA reaches the exact ground state of the v2 Hamiltonian and still
loses to the classical indicator.** Any better QAOA (more depth,
less noise) cannot exceed SA on the same Hamiltonian. The bottleneck
is the **Hamiltonian**, not the **solver**.

### Claim B — v2 Hamiltonian is not scenario-universal

Phase 10 (CMA-ES on (c_bias, thr_amr)).

- Joint training: delta F1 vs classical = −0.003 (CMA-ES, N=256).
- Per-scenario optima: `c_bias*` ∈ {0.38, 2.50, 5.25, 99.43} — 264×
  spread, and `mhd_rotor` hits the upper bound of the log-c_bias grid.
- Per-scenario training: delta ≤ 0 in all 4 scenarios (OT: 0.000,
  Harris: −0.021, KH: −0.005, Rotor: −0.071).

### Claim C — No local Ising H beats classical across unseen scenarios

Phase 11 (random split) vs phase 11B (leave-one-scenario-out),
with snapshot-level bootstrap CIs (phase 11B-2 and 11H).

| classifier                                   | split         | F1 (95% CI)                    |
|----------------------------------------------|---------------|--------------------------------|
| Classical multi-indicator                    | random by snap| 0.471 [0.423, 0.514]           |
| Mean-field GBT, 9 per-site features          | random by snap| **0.986 [0.972, 0.995]**       |
| Neighbourhood GBT, 9×5 stencil features      | random by snap| 0.990 [0.978, 0.998]           |
| Classical multi-indicator                    | LOSO (n=4)    | 0.434 ± 0.148                  |
| Mean-field GBT, 9 per-site features          | LOSO (n=4)    | **0.189 ± 0.150**              |
| Neighbourhood GBT, 9×5 stencil features      | LOSO (n=4)    | 0.215 ± 0.142                  |
| Learned linear mean-field H (phase 11c)      | LOSO (n=4)    | 0.391 (mean delta = −0.043)    |

**The random-split 0.986 ceiling is scenario-memorisation.** Under
LOSO it collapses by ≈5×. Per-fold bootstrap (phase 11B-2):

| held-out         | Δ site−class | 95% CI            | p(site ≥ class) |
|------------------|--------------|-------------------|-----------------|
| orszag_tang      | +0.063       | [+0.043, +0.084]  | 1.000           |
| harris_tearing   | −0.400       | [−0.463, −0.338]  | 0.000           |
| kelvin_helmholtz | −0.056       | [−0.067, −0.046]  | 0.000           |
| mhd_rotor        | −0.588       | [−0.633, −0.544]  | 0.000           |

**3/4 folds reject site ≥ class at p < 0.05.** Orszag-Tang is the
only fold with a positive delta (consistent with V1's jointly-trained
model showing advantage on OT). Stencil-vs-site gap CI upper bound = +0.009 — the formal
ceiling proposition is tight to within noise.

**Note on degenerate folds.** The classical multi-indicator reaches
F1 = 0.400 on the Harris and KH held-out folds. With a positive
rate of 0.250 (the top-quartile label), predicting *all* cells as
positive yields F1 = 2 × 0.25 / (1 + 0.25) = 0.400 (precision =
0.25, recall = 1.0). These two folds are therefore at the
degenerate all-positive prediction floor, not a meaningful operating
point. The classical threshold simply fails to discriminate on
those held-out scenarios.

### Claim D — The bottleneck is **feature insufficiency on unseen classes**, not cross-scenario transfer per se

Phase 11G (scenario-identity ablation).

| augmentation                            | LOSO mean F1 |
|-----------------------------------------|--------------|
| 9 features only                         | 0.189        |
| 9 features + one-hot scenario ID        | 0.188        |
| 9 features + one-hot WRONG scenario     | 0.196        |

Δ(9+id − 9) = −0.001 ≈ 0. **Telling the model which instability
class it is does not help.**

**LOSO nuance.** Under LOSO, the held-out scenario's one-hot column
is all-zero during training (it never appears) and all-one during
validation. GBT cannot learn from a feature it never saw vary, so
the Δ ≈ 0 result is expected mechanically. The informative
comparison is the "wrong scenario" row: the model *does* see those
one-hot columns vary in training, and shuffling scenario identity
still yields Δ ≈ 0. This confirms that even the
process-of-elimination signal (3 known columns → the unknown 4th)
does not help.

The stronger finding remains: **the L2-hard label is not a
per-snapshot function of the 9 features on a held-out class, even
when the class is named.**

### Claim E — V1's temporal `psi` channel does not rescue the cross-scenario bound; V1's aggregation method (block-avg vs block-max) explains V1's edge

Phase 11E (V1 tuned H + `psi` under V2 LOSO, as input-side proxy to
the full QAOA pipeline). N=256 with snapshot-level paired bootstrap
(B=500, max-snaps/cfg=80):

| held-out         | F1_v2_class | F1_v1_class | F1_v1+psi | Δ(v1+psi − v2_class) | 95% CI Δ          | p(v1+psi ≥ v2_cls) |
|------------------|-------------|-------------|-----------|----------------------|-------------------|---------------------|
| orszag_tang      | 0.264       | 0.395       | 0.306     | +0.042               | [+0.014, +0.070]  | 1.000               |
| harris_tearing   | 0.400       | 0.400       | 0.385     | −0.015               | [−0.040, +0.006]  | 0.098               |
| kelvin_helmholtz | 0.400       | 0.400       | 0.411     | +0.011               | [+0.003, +0.023]  | 1.000               |
| mhd_rotor        | 0.672       | 0.936       | 0.928     | +0.256               | [+0.209, +0.302]  | 1.000               |
| **mean**         | 0.434       | 0.533       | 0.508     | **+0.074**           |                   |                     |

**3/4 folds have CI strictly > 0** (OT, KH, Rotor); Harris is the
only fold where the CI includes zero (p = 0.098, not significant).

Decomposition of the +0.074 mean:

| component             | mean Δ  | interpretation                                 |
|-----------------------|---------|------------------------------------------------|
| V1_class − V2_class   | **+0.099** | aggregation method: block-avg (V1) vs block-max (V2) on the same 4-indicator + Löhner score |
| V1+psi − V1_class     | **−0.025** | psi temporal channel hurts on average         |

Both V1 and V2 compute the *same* 4-indicator classical score
(vorticity, divergence, |J_z|, Löhner) via
`AngleMapper.classical_score()`. The +0.099 gap is driven by
**downsampling**: V1 averages within each coarse patch (`block_avg`),
while V2 takes the maximum (`block_max`). Averaging preserves
sub-patch variation that the threshold can exploit; the max operator
saturates high-gradient patches and compresses the dynamic range.

**The V1 pipeline's advantage is entirely aggregation method, not
the temporal channel.** The `psi` channel slightly *degrades* the
LOSO signal (−0.025). This is consistent with V1's psi being tuned
on within-scenario data (Optuna trial 85): it helps where it was
trained (KH within-scenario, +0.011), but does not transfer.

### Per-scenario specialisation

Phase 11D (GBT). Within-scenario F1 = 0.90–0.98 (diagonal of
transfer matrix); off-diagonal F1 ranges from 0.08 to 0.50.
Avg(off-diag − classical) = −0.256. The binding constraint is MHD
rotor: its within-scenario classical is already F1 = 0.925, yet
off-diagonal transfers into rotor average only 0.232. A scenario
detector must achieve **≥ 97% accuracy on rotor** just to break
even with the classical baseline on that fold. Across all four
folds the worst-case break-even is 97% (rotor), while the average
break-even is 39%.

**"Train per scenario, switch at runtime" is viable only with a
near-perfect scenario detector for the hardest fold.**

---

## Theoretical closure for Q-HAS

The binding constraint on Q-HAS is **not** the quantum solver's
quality, **not** cross-scenario feature locality (phase 11G rules
this out), and **not** the temporal channel (phase 11E rules this
out). It is **feature insufficiency on held-out MHD instability
classes**: the 9 per-site physical features (vorticity, current,
|B|², |grad B|², …, Re) cannot reconstruct the L2-hard label on a
class not in the training set, even when the class identity is
given as an input.

The formal ceiling proposition (`docs/ceiling_proposition.md`):
since F1*_stencil − F1*_site ≤ +0.009 empirically (phase 11H),
**no Hamiltonian in the local-Ising-over-9-features family can
exceed the mean-field ceiling by more than ≈0.01**, regardless of
coupling range, solver, depth, or noise level.

Consequence: **MHD adaptive refinement is not Hamiltonian-representable
in the scenario-universal sense at the 9-feature level.** Any Q-HAS
deployment either (i) retrains per scenario (at which point
"quantum advantage" becomes "a fitted model helps on its training
distribution") or (ii) fails by an average of ≈0.24 F1 relative to
the classical multi-indicator baseline on unseen classes.

---

### Phase 11F — Multi-seed validation (CLOSED)

Phase 11F (N=256, 10 seeds, 4 scenarios, 440 snapshots):

| split        | metric               | mean ± σ (10 seeds) |
|--------------|----------------------|---------------------|
| random       | F1 mean-field GBT    | 0.968 ± 0.013       |
| random       | F1 stencil GBT       | 0.974 ± 0.012       |
| random       | F1 classical thr     | 0.460 ± 0.017       |
| **LOSO**     | **per-fold σ across seeds** | **0.000**     |

**Every LOSO fold gives identical F1 across all 10 seeds** (to 3
significant figures). The GBT fit is fully deterministic at this
dataset size. The headline LOSO numbers (0.189 ± 0.150) are not
seed-limited; the ±0.150 is inter-scenario variability only.

Random-split across-seed σ = 0.013 confirms the 0.986 [0.972,
0.995] headline from phase 11H is reproducible within ≈ 2 pp.

---

## Scope and caveats

1. **Feature basis is fixed to 9 physical features.** The bound is
   over the 9-feature local-Ising family. Adding richer features
   (Helmholtz decomposition, Elsasser variables, spectral
   descriptors) would raise F1*_MF and therefore raise the bound.
   Phase 11E already demonstrates a related effect: V1's
   `block_avg` aggregation (vs V2's `block_max`) on the same
   4-indicator score shifts the mean LOSO by +0.099, showing that
   how features are preprocessed matters as much as which features
   are included.

2. **Phase 11E is an input-side proxy**, not a full QAOA-over-V1-H
   evaluation. V1's QAOA polishes the input by at most ≈0.66% in
   combined decision score (V1 report), so the +0.074 LOSO number
   would shift by at most ≈±0.01 under full QAOA — within the
   bootstrap CIs above.

3. **Temporal channel at multi-step granularity.** V1's `psi` is
   per-step (per hybrid iteration). Phase 11E evaluates it on
   single-snapshot labels; a multi-step protocol like V1's Fig. 2
   early-detection window is not re-run under LOSO here.

4. **Scope: 2D MHD, Re ≤ 1600, 4 instability classes.**
   Extending to 3D, higher Re, or additional scenarios (e.g.,
   magnetic reconnection with guide field, or turbulent dynamo)
   would widen the LOSO evaluation but is beyond the current
   experimental scope.

---

## Reproducibility

- Pipeline: `bash study/run_study_v2.sh --full <phase>` for each of
  {1, 2, 2b, 3, 4, 5, 6, 7, 8, 10, 11, 11b, 11b2, 11c, 11d, 11e,
  11f, 11g, 11h, 13} at N=256.
- New diagnostics (this study): `phase1b_dns_validation.py`,
  `phase2b_percentile_sensitivity.py`, `phase11b2_bootstrap.py`,
  `phase11d_specialisation.py`, `phase11e_v1h_loso.py`,
  `phase11f_multiseed.py`, `phase11g_scenario_ablation.py`,
  `phase11h_random_split_bootstrap.py`.
- DNS validation (phase 1b): max |div B| / rms |B| ≤ 5 × 10⁻⁵
  across all snapshots; OT total energy decays monotonically
  within the Dahlburg & Picone (1989) window at Re ≥ 1000.
- V1 best-params reference: `best_hyperparams.json` trial 85
  (β=0.5495, threshold_amr=0.3044, etc.).
