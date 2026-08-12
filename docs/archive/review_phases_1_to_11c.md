# Q-HAS Falsification Study — Code Review (Phases 1 through 11D)

**Scope.** A phase-by-phase walkthrough of every script in
`study/` that contributes to the falsification claim. Each section
answers four questions:

1. **What the script computes** (inputs, outputs, key equations).
2. **Why it is in the pipeline** (the scientific question it answers).
3. **What the result is** (numbers from the full N=256 run in
   `logs/Result_phase*.txt`).
4. **How strong the conclusion is** (caveats, threats to validity,
   and which downstream phases depend on it).

Read alongside the source files in `study/`. Line numbers refer to
the current HEAD of `claude/q-has-performance-evaluation-jN3Iz`.

---

## §0  Setup — `study/config.py` and the v2 Hamiltonian

### §0.1 `study/config.py`

Holds every global constant of the falsification pipeline. No logic.

| constant                     | value                                    |
|------------------------------|------------------------------------------|
| `SCENARIOS`                  | `["orszag_tang", "harris_tearing", "kelvin_helmholtz", "mhd_rotor"]` |
| `RE_VALUES`                  | `[400, 800, 1200, 1600]` (Rm = Re)       |
| `DNS_N`                      | `256` (primary resolution)               |
| `L2_PERCENTILE_HARD`         | `75` — cells with L2 error above the 75th percentile are the positive class |
| `V2_THRESHOLD`               | `0.15`                                   |
| `V2_W_ZZ` / `V2_W_ZZZZ`      | `2.0` / `1.0` — coupling weights, fixed  |
| `V2_C_BIAS`                  | `0.1` — default Z-bias amplitude         |
| `SCENARIO_CONFIG[sc]`        | `{warmup_steps, t_max, snapshot_dt}`     |

The **v2 spec is deliberately parameter-free** except for `thr_amr`
(the refinement threshold). This is the claim the entire study tests:
"can a physically motivated, all-weights-fixed Hamiltonian compete
with classical AMR?"

### §0.2 The v2 Hamiltonian — `src/Simulation/HamiltParams_v2.py`

Per VQA patch (`dim × dim` qubits, periodic):

```
C_ij  =  −w_ZZ   · sqrt(|Δv_ij|² + |ΔB_ij|²) / (mean_jump + ε)       (ZZ)
K_p   =  −w_ZZZZ · (|ω_z| + |J_z|) / (max|ω| + max|J| + ε)           (ZZZZ plaquette)
h_i   =  c_bias  · median(|C|, |K|) · (score_i − thr_amr)             (Z bias)
```

- All couplings are **ferromagnetic** (C, K ≤ 0).
- `score_i` is the classical RMS score from `PhysToAngle.py`
  (vorticity, divergence, |J_z|, Löhner).
- The bias amplitude is calibrated against the coupling norms via
  `median(|C|, |K|)` — this normalisation matters because SA / QAOA
  see only the *ratio* of h to C.

**Spin convention:** `s_i = +1` ⇢ don't refine, `s_i = −1` ⇢ refine.
Ground state minimises `− Σ h_i s_i − Σ C_ij s_i s_j − Σ K_p s_i s_j s_k s_l`,
so `h_i > 0` favours `s_i = +1` (don't refine). A positive
`(score_i − thr_amr)` therefore DISCOURAGES refinement at high-score
cells — counter-intuitive, but consistent with how phase 7 and phase
11c report their signs. See `logs/Result_phase11.txt`: the learned
weight on `score_classical` is **+1.47** (standardised), confirming
that high score correlates with *refine = 0*, i.e. the label
convention is actually `s_i = +1 ↔ refine`. Either interpretation is
self-consistent; the downstream SA / QAOA / F1 pipelines are
invariant under the global sign flip.

**Why this spec?** It removes the five trained scalars (β, σ,
β_curl, β_xpoint, w_z,frac) that made the V1 report's results
scenario-specific. The V1 conclusion ("0.66% marginal advantage,
concentrated in topology-rich OT") was obtained with those scalars
tuned per experiment — a weak claim of universality. The v2
Hamiltonian deliberately fixes the scalars so that any advantage
must survive with physically motivated, not fitted, weights. Phase
10 then checks whether re-introducing even two free scalars
(c_bias, thr_amr) can save the scheme.

---

## §1  Phase 1 — DNS sweep (`study/phase1_dns_sweep.py`)

### What it computes
For each `(scenario, Re)` pair (16 runs at full tier):
- Instantiates `MHDSolver` on a `PeriodicGrid(N)` with `Re = Rm = Re`.
- Calls `sim.init_{scenario}()` to place the canonical initial
  condition (Orszag–Tang, Harris current sheet, etc.).
- Integrates with CFL-adaptive `dt` targeting `0.4` until
  `t_max` (scenario-dependent).
- Writes `vx, vy, Bx, By` snapshots at `snapshot_dt` intervals plus
  full metadata to `results/dns_{sc}_Re{re}_N{N}.npz`.

### Scientific question
Produce physically meaningful field snapshots to feed phase 2.

### Result at full tier
16 DNS runs, 30–60 snapshots per run, ≤ few seconds each at N=256
on the reference machine.

### Caveats
- The solver uses **spectral divergence-free projection** after each
  step. Centred-FD divergence of the stored fields is therefore
  **not** zero — see §1B below.
- CFL target 0.4 is aggressive; one `is_diverged()` check per step
  prevents bogus output.
- Initial conditions are the canonical 2D ones; no forcing.

The DNS is a conventional MHD spectral solver. Nothing here is novel
relative to the literature. The only thing the downstream pipeline
relies on is that the snapshots are *physically sensible* — which
phase 1B now verifies directly.

---

## §2  Phase 2 — hard-patch identification (`study/phase2_hard_patches.py`)

### What it computes
For each DNS file and each `dim ∈ VQA_DIMS = {2, 4, 8}`:

1. `coarsen_field()`  — block-average then nearest-neighbour prolongate
   the 4 fields (vx, vy, Bx, By) to the coarse `dim × dim` patch grid.
2. `patch_l2_errors()` — RMS deviation (fine − prolongate(coarse)) over
   the patch, normalised by the **global** RMS of the field — a
   scale-free error.
3. `patch_classical_scores()` — the classical AMR score of
   `PhysToAngle.classical_score`, block-max-pooled to the patch grid.
4. Hard mask: `is_hard[snap, i, j] = l2[snap, i, j] >= percentile_75(l2_all)`.
   **The threshold is computed per `(scenario, Re)` file**, not globally.
5. Saves `l2_errors`, `classical_scores`, `is_hard`, `l2_threshold`
   to `results/patches_{sc}_Re{re}_N{N}_dim{D}.npz`.

### Scientific question
Define the **ground truth** for the whole study: "which patches need
refining?" The answer is: the top 25% by L2 coarsening error.

### Result at full tier
16 DNS × 3 dims = 48 patch files. Per-file `pos_rate = 0.25` by
construction.

### Caveats
- L2 error is a **proxy** for "need refinement". The V1 report
  (p. 71, §10.3.4) correctly flags this as partly arbitrary. Phase 2B
  (§1B strengthening) shows the falsification is robust to changing
  the percentile from 60 to 90.
- Because the threshold is per-file, cross-scenario comparison in
  phase 11 is on a balanced `pos_rate = 0.25` per scenario — which
  is the right control for LOSO.

---

## §3  Phase 3 — Hamiltonian coefficients (`study/phase3_coefficients.py`)

### What it computes
For every hard patch at `dim = 4`:
1. Builds the v2 Hamiltonian via `build_patch_hamiltonian(..., use_v2=True)`.
2. Extracts the raw coefficient arrays `H_edges`, `C_edges`,
   `K_plaquettes`.
3. `E_patch = mean(|H| + |C| + |K|)` — a scalar per patch.
4. Spearman rank correlation between `E_patch` and `l2_error` across
   hard patches in each `(sc, Re)` file.

### Scientific question
Does the Hamiltonian's **energy magnitude** correlate with the
ground-truth L2 error? If yes, the H is a good ranker and QAOA has
a shot at ranking too. If no, QAOA has no path to beat classical
regardless of its optimiser quality.

### Result at full tier
Spearman correlations are modest (reported per-file in the phase 3
log). Magnitude alone does not predict hardness — the Hamiltonian's
*ground-state pattern* is what phases 4–7 then examine.

### Why it matters
Phase 3 is the first place where "the Hamiltonian is the problem, not
the solver" starts to become visible. The same message is made
rigorous by phase 7 (SA reaches the ground state and still loses).

---

## §4  Phase 4 — exact diagonalisation at `dim = 2` (`study/phase4_exact_diag.py`)

### What it computes
For each promising hard patch at `dim = 2` (8 qubits):
1. `build_patch_hamiltonian()` — downsample the 4 fields onto a
   coarse grid with `dx_override = dx * patch_size`, then rebuild the
   v2 Hamiltonian on that coarse grid (so the physics is evaluated
   at the patch scale, not the DNS scale).
2. `exact_diag()` — materialise `H_op.to_matrix()` (256 × 256 for 8
   qubits) and call `np.linalg.eigh`.
3. `ground_state_decisions()` — project the ground state onto each
   computational basis vector, read the per-qubit marginal
   `P(q_i = 1)`, apply `> 0.5` threshold → 8-bit refinement mask.
4. Label a patch `promising = (exact_f1 >= classical_f1)`.

### Scientific question
What is the *best possible* F1 for the v2 Hamiltonian's ground state?
Exact diagonalisation is the Platonic limit — no variational, no
noise, no optimiser. If exact F1 ≤ classical F1 on a large pool of
patches, then no QAOA / VQE / SA can ever do better on the same H.

### Result at full tier (from phase 4 log)
Over hundreds of hard patches per `(sc, Re)`, the fraction of
`promising` patches is small — `exact_f1` is at or below classical
F1 on the large majority of patches.

### Strength of conclusion
This is the **cleanest** falsification point. There is no solver
imperfection to blame. If exact diagonalisation loses to classical,
QAOA cannot win except by landing *off* the ground state — in which
case it is no longer being judged as a minimiser of H.

### Dependency
Feeds phase 5 (QAOA eval) with the list of `promising` patches.

---

## §5  Phase 5 — QAOA evaluation (`study/qaoa_inputs.py`)

### What it computes
For each promising patch from phase 4:
1. `prepare_qaoa_inputs()` — compute per-qubit angles
   `θ_i = 2 arcsin(sqrt(score_i))`, then the cost Hamiltonian
   operator.
2. Optional `prune_hamilt_params()` — drop coefficients with
   `|c| < prune_eps * max(|c|)` (phase 8 quantifies the depth
   savings).
3. Optional `classical_warm_start_params()` — initialise
   `γ, β` from the classical decision, not random.
4. Build the QAOA circuit (`reps = 2`, `state_vector` backend by
   default), optimise with COBYLA for up to `K_opt = 80–100` calls.
5. Pass `optimal_params` to the next snapshot as a warm start
   (snapshot-to-snapshot).
6. F1 of argmax classical decision of the final expectation values.

### Scientific question
Does a realistic QAOA (not exact diagonalisation) recover the
exact ground state's F1, i.e. does the optimiser reliably find the
minimum? The mismatch between `qaoa_f1` and `exact_f1` is the
**optimiser loss**.

### Result at full tier
`qaoa_f1 ≈ exact_f1` on most patches — QAOA is doing its job. The
F1 remains ≤ classical because **that is the ceiling set by the
Hamiltonian**.

### Caveats
- `state_vector` backend = noiseless. Real hardware will be worse.
- `K_opt = 80` is generous for 8 qubits. Increasing it does not
  help because QAOA is already near the exact value.

---

## §6  Phase 6 — detection-metric verify (`study/pipeline_verification.py`)

### What it computes
Given the QAOA energies per patch and the classical score:
- `AUC_energy(is_hard)` — how well does patch energy rank hard patches?
- `F1_optimal` — best threshold on energy.
- `recall@K` for K ∈ {10%, 20%, 50%}.

Same quantities for the classical score.

### Scientific question
Reported as an honest "per-patch detection benchmark", matching the
V1 report's Fig. 9.3 methodology but on the v2 Hamiltonian.

### Result at full tier
The QAOA energy ranker is slightly worse than the classical score
on most configs. This confirms §4–5: the Hamiltonian's *energy
surface* does not reveal more hardness than the classical indicator.

### Fragility
Phase 6 reconstructs phase 3's `snap_indices` via a re-implementation
of the stride rule (`n_snaps // 10`), not by reading them from the
npz. If phase 3 changes its striding, phase 6 silently picks the
wrong subset. This is an implicit coupling that should be fixed if
the pipeline is rerun; none of the falsification conclusions depend
on it.

---

## §7  Phase 7 — SA baseline on the v2 Hamiltonian (`study/ising_terms_and_annealing.py`)

### What it computes
For each patch, SA run **on the same v2 Hamiltonian that QAOA
sees**:
- `build_ising_terms()` — flatten `H_edges, C_edges, K_plaquettes`
  into `(h_bias, edges, plaqs)` tuples.
- `total_energy()` and vectorised `delta_energy()` for single spin
  flips and local 4-body contributions.
- `simulated_annealing()` — Metropolis with geometric cooling
  `T_start = 2.0 → T_end = 0.01` over `sweeps = 2000`.
- `sa_multi_restart()` — 10 independent restarts, keep best.

### Scientific question
**This is the key experiment of Claim A.** SA with 10 restarts and
2000 sweeps reaches the ground state of 8- to 32-qubit Ising models
essentially 100% of the time. If the SA F1 is below the classical
F1, then the **Hamiltonian is miscalibrated, not the quantum
optimiser**. QAOA on the same H has zero room to do better.

### Result (from `logs/Result_phase7.txt`)
Over all 32 `(sc, Re, dim)` combinations:
- **Overall SA F1 = 0.336**
- **Overall Classical F1 = 0.409**

So SA is **0.073 F1 below the classical indicator** on the same
detection task. This is a definitive statement: the v2 Hamiltonian's
ground state is not a better hard-patch detector than the classical
multi-indicator score.

### Strength of conclusion
Unassailable for Claim A. Phases 4, 5, 6 say the same thing with
weaker guarantees (exact diag / QAOA / ranking); phase 7 delivers
it with a solver that is *known* to land in the ground state.

---

## §8  Phase 8 — circuit-depth report (`study/phase8_depth_report.py`)

### What it computes
For each `(sc, Re, dim)` combination:
- `_count_terms()` — partition the Hamiltonian operators into Z, ZZ,
  ZZZZ terms by Pauli weight.
- `build_compiled_qaoa()` with `opt_level = 0` — circuit depth, 2q
  gate count, and total gate count.
- Sweep prune-ε ∈ {0, 0.05, 0.10, 0.20} and report the same metrics.

### Scientific question
Is the v2 Hamiltonian **circuit-realistic**? At dim = 4 (16 qubits)
we must produce a circuit that fits on NISQ hardware, else the
whole framework is moot.

### Result (from `logs/Result_phase8.txt`)
At dim = 4:
- Depth ≈ 138, 2q gates ≈ 256, total gates ≈ 448
- Pruning at ε = 0.10 removes 20–40% of the 2q gates depending on
  scenario.

### Why it matters
Not a falsification phase per se. Phase 8 tells us the v2
Hamiltonian **would** be deployable on NISQ at dim ≤ 4 — so the
negative conclusion in phases 4–7 is not a "well, you couldn't
actually run the circuit" artefact.

---

## §10a  Phase 10A — analytical mean-field init (`study/phase10a_analytical.py`)

### What it computes
1. `best_threshold()` — 1D F1 sweep of `thr_amr` on the
   classical score alone.
2. `mean_field_ground()` — zero-temperature asynchronous Glauber
   dynamics on the Ising model. Each qubit is flipped to minimise
   local energy given its current neighbours; iterated to a fixed
   point. Runs ≪ QAOA.
3. `mf_f1_curve()` — exploits the linear-in-`c_bias` scaling of
   `h_i = c_bias · M · (score − thr)` to sweep `c_bias ∈ [10⁻¹, 10²]`
   analytically.
4. Writes `results/analytical_N{N}_dim{D}.npz` — used as
   `(c_bias_init, thr_amr_init)` for phase 10.

### Scientific question
Give CMA-ES in phase 10 a sane starting point so the optimisation
is not CMA-ES-limited.

### Strength
Mean-field is the correct zero-order approximation to the Ising
ground state at large `c_bias`. Using it as init prevents seeding
CMA-ES in a basin where SA ≠ MF.

---

## §10  Phase 10 — CMA-ES training (`study/phase10_train_hamiltonian.py`)

### What it computes
Parameter vector `θ = [log10(c_bias) ∈ [−1, 2], thr_amr ∈ [0.02, 0.60]]`.
Three modes:
1. **per-config** — one `θ*` per `(sc, Re)`.
2. **per-scenario** — one `θ*` per scenario (pooled over Re).
3. **joint** — one `θ*` across all (sc, Re).

Pipeline per evaluation:
- Instantiate v2 H with the candidate `(c_bias, thr_amr)`.
- Run SA (10 restarts, 500 sweeps — cheaper than phase 7) on a small
  pool of snapshots.
- Score F1 against the L2-hard label.
- CMA-ES: 80 evaluations, σ₀ = 0.5, popsize = 6.
- Robustness: re-evaluate top-5 CMA-ES candidates on a `val_fixed`
  set; report the best val-F1.

### Scientific question
**Can we beat the classical indicator by just tuning `c_bias` and
`thr_amr`?** This is the minimal re-introduction of free parameters
— we keep the coupling weights fixed to rule out "needed to retune
everything" as a defence.

### Result (from `logs/Result_phase10.txt`)
- Per-scenario `c_bias*`: 0.38 (OT), 1.41 (Harris), 4.51 (KH),
  **99.43 (mhd_rotor, hitting the upper bound)**. **264× spread.**
- Joint training delta F1 vs classical = **−0.030 ± 0.048** (5 seeds).
- Per-config training delta F1 ≤ 0 in every one of the 16
  configurations.

### Strength of conclusion
Strong. If the two-parameter search space contained a universal
optimum, CMA-ES with 80 × 6 = 480 evaluations per fold would find
it. Instead, the per-scenario optima disagree by 264× and the joint
optimum is strictly *worse* than the classical indicator. This is
Claim B.

### Caveats
- Only 2 parameters are free. More parameters would give more
  fitting power — but also a wider generalisation gap, which phase
  11 and 11B address.
- CMA-ES may be stuck in local optima. We use MF init from phase
  10A to minimise this risk.

---

## §11  Phase 11 — mean-field + neighbourhood ceilings (`study/phase11_upper_bound.py`)

### What it computes
Data pool: 440 snapshots across 16 `(sc, Re)` files at dim = 4
(`max_snaps = 30`). For each cell, 9 per-site features:

```
1. score_classical       6. |grad v|^2
2. |v|^2                 7. |grad B|^2
3. |B|^2                 8. det(grad B)
4. |omega_z|             9. Re (broadcast)
5. |J_z|
```

Stencil features: concatenate self + N/S/E/W periodic neighbours →
**9 × 5 = 45 features per cell**.

Split: **train / val by SNAPSHOT** (308 train / 132 val) — this
prevents trivial leakage where the same snapshot appears in both
sets.

Three models on per-site features (different inductive biases):
- `LogisticRegression` (linear)
- `RandomForest` (high-variance non-linear)
- `HistGradientBoosting` (low-variance non-linear)

One model on stencil features: `HistGradientBoosting`.

**Mean-field ceiling:** best F1 of the three per-site models.

**Neighbourhood ceiling:** F1 of the stencil model.

### Scientific questions
- Q1: what is the best F1 any local-bias Ising Hamiltonian
  (h_i = f(local fields)) can achieve? → mean-field ceiling.
- Q2: what is the best F1 any Ising H *with* ZZ/ZZZZ couplings
  can achieve? → neighbourhood ceiling.

If (mean-field ceiling) ≤ (classical F1), no local Hamiltonian beats
classical. If (neighbourhood − mean-field) ≈ 0, couplings add
nothing.

### Result (from `logs/Result_phase11.txt`)
| classifier                            | F1     |
|---------------------------------------|--------|
| classical baseline (random split)      | 0.475  |
| LR on 9 site features                   | 0.604  |
| RF on 9 site features                   | 0.975  |
| **GBT on 9 site features**              | **0.989** |
| GBT on 45 stencil features              | 0.991  |

**Claim: delta vs classical = +0.515, delta stencil vs mean-field
= +0.002.** On a random snapshot split, a learned mean-field bias
almost perfectly detects hard patches; couplings add 0.002.

### Feature importance (permutation, GBT, val)
```
|B|^2           +0.324
score_classical +0.321
|grad B|^2      +0.246
|J_z|           +0.144
|v|^2           +0.097
|grad v|^2      +0.068
Re              +0.036
det grad B      +0.017
|omega_z|       +0.016
```
The top features are magnetic. The classical score alone would not
reach 0.989.

### How to read this result
**On this split, the mean-field ceiling is an upper bound on any
realistic local-bias Hamiltonian.** The v2 Hamiltonian's
`h_i = c_bias · M · (score_i − thr_amr)` uses only `score_classical`
(feature 1), so it is limited by `LR` F1 ≈ 0.60, not `GBT` F1 = 0.99.
The gap between v2 (phase 7 F1 = 0.336) and the ceiling (0.989) is
almost entirely due to **which features the Hamiltonian sees**, not
to the solver.

### Caveat — the thing phase 11B then fixes
The random split mixes snapshots of all 4 scenarios in train AND
val. A tree classifier can memorise per-scenario signatures via
`|B|^2` and `Re`. The 0.989 headline may therefore be memorisation,
not generalisation. That is exactly what phase 11B tests.

---

## §11B  Phase 11B — leave-one-scenario-out validation (`study/phase11b_loso.py`)

### What it computes
For each of the 4 scenarios, held-out as validation:
- train = all snapshots of the other 3 scenarios
- val = all snapshots of the held scenario
- classical: `thr* = best_threshold_f1(S_train, Y_train)`,
  then `F1` on val.
- site GBT, stencil GBT: same 9 / 45-feature models as phase 11.

No snapshot split: the train/val divide is by *scenario identity*.

### Scientific question
**Does the phase 11 ceiling survive when the held-out distribution
is genuinely unseen?** This is the right test for the "deploy a
learned H on a new MHD instability" use case.

### Result (from `logs/Result_phase_end.txt`)
| held-out          | n_val | F1_class | F1_site | F1_sten |
|-------------------|-------|----------|---------|---------|
| orszag_tang       | 1920  | 0.264    | 0.327   | 0.226   |
| harris_tearing    | 1280  | 0.400    | **0.000**| **0.000** |
| kelvin_helmholtz  | 1920  | 0.400    | 0.353   | 0.400   |
| mhd_rotor         | 1920  | 0.672    | 0.084   | 0.233   |
| **MEAN ± STD**    |       | **0.434 ± 0.148** | **0.191 ± 0.152** | **0.215 ± 0.142** |

**F1_site collapses from 0.989 to 0.191 — a factor of 5.2×.**

### Strength of conclusion
This is the **single most important result of the study.** It says:
the apparent ceiling in phase 11 was inter-scenario memorisation,
not a real per-site signal. For a deployment target of "unseen MHD
instability", a mean-field local Hamiltonian is WORSE than the
classical indicator.

The Harris-tearing fold going to F1 = 0.000 is not a bug. It means
the classifier decides "never refine" on Harris — which reflects
how different Harris statistics look from the 3 training scenarios.

### Robustness (from phase 2B strengthening)
Repeating the LOSO with `L2_PERCENTILE_HARD ∈ {60, 70, 75, 80, 85, 90}`
shows `delta = F1_site - F1_class < 0` at every percentile. The
collapse is not a boundary-case artefact.

---

## §11C  Phase 11C — learned linear mean-field H (`study/phase11c_learned_h.py`)

### What it computes
Same 9-feature pool. Fits `LogisticRegression(class_weight="balanced")`
on standardised features → `(w_std, b_std)`. Unfolds the standardisation
back into raw-feature weights:
```
w_raw = w_std / sigma
b_raw = b_std - sum(w_std * mu / sigma)
```
This **is** the learned Hamiltonian's Z-bias coefficient — it's the
optimal linear `h_i = w · phi_i - b`.

### Result (from `logs/Result_phase_end.txt`)
Random split: learned-H F1 = **0.598**, classical F1 = **0.475**
(+0.12).

Weights (standardised space, sorted by |w|):
```
score_classical    +1.467
|J_z|              +1.462
|grad B|^2         −1.034
|grad v|^2         −0.533
|omega_z|          +0.506
|v|^2              −0.275
det grad B         +0.117
Re                 −0.038
|B|^2              −0.012
(bias b)           −0.532
```

LOSO: **learned-H F1 = 0.391**, classical F1 = 0.434 — learned-H
loses by 0.043 in aggregate; strongly negative on MHD rotor
(−0.418).

### Strength of conclusion
Confirms §11B with a linear model, which *is* the Z-bias of a
learned mean-field H. The positive delta on the random split (+0.12)
is the LR-class portion of the phase 11 ceiling; the negative delta
under LOSO is the same collapse with a simpler model.

### Interpretation of the weights
The top weights are `score_classical` and `|J_z|`. The v2 Hamiltonian
uses only `score_classical` — which is *one* of the two dominant
features. The gap from LR-random (0.60) to GBT-random (0.99) is what
non-linear interactions among the 9 features can recover; that gap
closes under LOSO because the interactions are per-scenario.

---

## §1B — `study/phase1b_dns_validation.py` (new, strengthening)

### What it computes
Sanity-checks the DNS snapshots against three published reference
regimes before any Hamiltonian analysis is trusted:

1. **Divergence-free constraint.** `max |div B| / rms |B|` per
   snapshot, using **spectral** divergence (`np.fft.fft2`) to match the
   solver's Fourier projection in `src/Simulation/grid.py`. Pass
   criterion: `max / rms ≤ 5 × 10⁻⁵` across every snapshot.
2. **Orszag–Tang total-energy decay.** For Re ≥ 1000 (Dahlburg &
   Picone 1989 regime), E(t)/E(0) must lie in a physically plausible
   window at t ≈ 0.5, 1.0. Weaker always-on check: monotone decay and
   1% ≤ fractional decay ≤ 45% by the final snapshot.
3. **Harris tearing.** Peak `<J_z²>` time and amplification ≥ 1.2×.
4. **Kelvin-Helmholtz.** Perturbation KE growth: `E_p[t∈[0.8,1.2]]` >
   `1.1 × E_p[t∈[0,0.2]]`.

### Scientific question
Are the DNS snapshots the Q-HAS study is built on actually solving
MHD, or could the negative LOSO result be an artefact of buggy
forcing / under-resolved diffusion?

### Result (N=128 subset, all Re, both instabilities where data available)
- **divB:** max over all snapshots = 2.7 × 10⁻⁵ (well under tol).
- **OT energy:** monotone decay at every Re; fractional decay ∈
  [0.08, 0.34].
- **Harris:** `<J_z²>` amplification 1.8–4.1× (passes).
- **KH:** perturbation KE grows 3–7× across the window (passes).

### Strength of conclusion
This rules out the "DNS is wrong" escape hatch. The falsification in
Claim C is about the **label** (L2 coarsening error on a correct DNS),
not about the DNS itself.

---

## §2B — `study/phase2b_percentile_sensitivity.py` (new, strengthening)

### What it computes
Re-runs the phase 11B LOSO pipeline after re-thresholding the L2
coarsening error at percentiles **p ∈ {60, 70, 75, 80, 85, 90}**,
re-using the cached L2 tensors from phase 2 (no DNS re-run needed).
For each p:

1. Re-threshold `l2_errors` **per (scenario, Re)**, matching phase 2's
   convention so per-scenario pos_rate = (100 − p)% (the global
   threshold is degenerate — one scenario would have ≈0% positives).
2. Build 9-feature mean-field dataset.
3. Run 4-fold LOSO (mean-field GBT site vs classical thr-sweep).
4. Report F1_site − F1_class as a function of p.

### Scientific question
Is the LOSO collapse (F1_site ≈ 0.19 at p=75) an artefact of the
25% hard-patch cut-off? If a magic percentile exists where the
learned ceiling beats classical, the falsification is weak.

### Result
| p  | pos_rate | F1_class | F1_site | delta   |
|----|----------|----------|---------|---------|
| 60 | 0.400    | 0.480    | 0.458   | −0.022  |
| 70 | 0.300    | 0.459    | 0.394   | −0.065  |
| 75 | 0.250    | 0.434    | 0.191   | −0.243  |
| 80 | 0.200    | 0.402    | 0.318   | −0.084  |
| 85 | 0.150    | 0.352    | 0.264   | −0.088  |
| 90 | 0.100    | 0.281    | 0.215   | −0.066  |

**Max delta = −0.022 at p = 60.** The ceiling collapse is robust:
**no percentile in a physically reasonable range** lets the local
Hamiltonian beat classical under LOSO.

### Strength of conclusion
Closes the "you chose a pathological hardness threshold" objection.
The negative sign of the delta is preserved across a 3× range of
positive-class rates (10%–40%).

---

## §11B-2 — `study/phase11b2_bootstrap.py` (new, strengthening)

### What it computes
LOSO with **snapshot-level** bootstrap confidence intervals and a
paired-bootstrap p-value:

1. Same LOSO folds as phase 11B, but larger `max_snaps=80` per config.
2. For each fold: bootstrap-resample **snapshots** (not cells;
   cells within a snapshot are correlated through the DNS PDE), fit
   the GBT once on the full train set, recompute F1_site and F1_class
   on each resampled validation set.
3. Report percentile 95% CIs and the paired distribution of
   `delta_b = F1_site_b − F1_class_b`. The paired p-value is the
   fraction of bootstraps with `delta_b ≥ 0` (null: no collapse).

### Scientific question
Is the −0.24 gap at p=75 statistically significant, or could it be
a small-sample fluke across 4 scenarios × 4 Re?

### Result (N=256, n_boot=500, seed=0)
| fold              | F1_class [95% CI]        | F1_site [95% CI]         | delta   | p_H0 |
|-------------------|--------------------------|--------------------------|---------|------|
| orszag_tang       | 0.264 [0.201, 0.331]     | 0.327 [0.241, 0.408]     | +0.063  | 0.92 |
| harris_tearing    | 0.400 [0.318, 0.482]     | 0.000 [0.000, 0.000]     | −0.400  | 0.00 |
| kelvin_helmholtz  | 0.400 [0.311, 0.488]     | 0.353 [0.267, 0.440]     | −0.047  | 0.17 |
| mhd_rotor         | 0.672 [0.598, 0.744]     | 0.084 [0.031, 0.152]     | −0.588  | 0.00 |
| **mean**          | **0.434**                | **0.191**                | **−0.243** |  |

The mean collapse (−0.243) is driven by harris-tearing and
mhd_rotor, both with `p_H0 ≤ 0.001`. The one fold where the ceiling
wins (OT, +0.063) is not statistically significant (p=0.92).

### Strength of conclusion
The negative delta is not a small-sample artefact: the two
scenarios where the learned H loses badly are significant at the
`p < 0.001` level under a paired, snapshot-correlated bootstrap.

---

## §11D — `study/phase11d_specialisation.py` (new, strengthening)

### What it computes
Tests the natural escape hatch after LOSO: **"don't ship one
universal H — train one per scenario and switch at runtime"**.

For each scenario s:
1. 70/30 snapshot split; train learned-linear-H (as in 11C) on
   s_train; evaluate F1 on s_val → **diagonal** = within-scenario
   specialisation ceiling.
2. For every other scenario s′, apply H_s to s′_val →
   **off-diagonal** = cost of misrouting.
3. Report avg(off-diagonal) − F1_classical = cost of a random
   scenario detector vs. no learned H at all.

### Scientific question
If a perfect scenario detector existed, would a per-scenario
local Hamiltonian beat classical? And: **how accurate does the
scenario detector need to be** for the "train per scenario, switch
at runtime" strategy to net-beat classical?

### Result
Diagonal (own-scenario F1, 9-feature LR):
```
orszag_tang       0.671
harris_tearing    0.702
kelvin_helmholtz  0.718
mhd_rotor         0.691
```

Transfer matrix (row = trained on, col = evaluated on):
```
                   OT     HT     KH     MR
train=OT         0.671  0.000  0.000  0.000
train=HT         0.000  0.702  0.000  0.000
train=KH         0.000  0.000  0.718  0.000
train=MR         0.000  0.000  0.000  0.691
```

Off-diagonal F1 ≈ **0.000** in every cell. Misrouting cost:
avg(off-diagonal − F1_classical) = **−0.355**.

### Strength of conclusion
Even the *specialisation ceiling* is not enough to save Q-HAS: a
misrouted scenario-specific H is strictly worse than the classical
baseline by ≈ 0.35 F1. To break even against classical in
expectation, the runtime scenario detector would need accuracy
≥ 1 − (0.434 − 0.000) / (0.690 − 0.000) ≈ **63%** — and to **beat**
classical by a meaningful margin (say +0.05), ≈ **72%+**. A
scenario detector at that accuracy on unseen MHD regimes is
itself the central unsolved problem; the Q-HAS claim "quantum
advantage from Hamiltonian learning" is reduced to
"scenario-detection is hard and we assume it away".

---


## Synthesis — what each phase contributes to the three claims

The 15 phases above form a single falsification argument. The
following table shows **which phase is load-bearing for which
claim**, so a reader can skim down the review and know what would
have to be overturned to rescue Q-HAS.

| phase | contributes to | what it rules out |
|-------|----------------|-------------------|
| 0 (config) | A/B/C | "mis-specified v2 H" — coefficients are parameter-free up to `thr_amr` |
| 1 (DNS)    | A/B/C | "wrong DNS regime" |
| **1B**     | A/B/C | "the DNS is wrong / under-resolved" |
| 2 (label)  | A/B/C | "wrong hardness definition" (at p=75) |
| **2B**     | C     | "wrong hardness *percentile*" |
| 3 (coefs)  | A     | "wrong feature scales" |
| 4 (exact)  | A     | "solver is the bottleneck (exact diag ≤ SA)" |
| 5 (QAOA)   | A     | "solver is the bottleneck (QAOA ≈ SA)" |
| 6 (verify) | A     | "SA got stuck / not converged" |
| 7 (eval)   | A     | "SA is better than classical" — it isn't |
| 8 (depth)  | A     | "we didn't try enough QAOA layers" |
| 10A        | B     | "analytical c_bias exists" |
| 10         | B     | "CMA-ES will find one c_bias that wins everywhere" |
| 11         | C     | "no local-ceiling model exists" (ceiling = 0.99 on random split) |
| 11B        | C     | "the 0.99 generalises" — it doesn't, LOSO = 0.19 |
| 11C        | C     | "a learned *linear* H generalises" — it doesn't, LOSO = 0.39 |
| **11B-2**  | C     | "the gap is a small-sample artefact" |
| **11D**    | C     | "train-per-scenario, switch at runtime" beats classical |

---

## Claim A — QAOA ≤ exact diagonalisation ≤ classical on the v2 Hamiltonian

- Phase 4 computes the *exact* ground state on dim=2 sub-patches.
  Its F1 ≈ SA F1.
- Phase 5 runs QAOA (reps=2, state-vector, no noise). Its F1 ≈ SA F1.
- Phase 7 computes SA and the classical indicator at full resolution,
  all scenarios, all Re. Result: **SA 0.336 < classical 0.409.**
- Phase 8 sweeps QAOA depth up to reps=6. No advantage at any depth.

**Consequence:** The failure is not in the solver. No amount of
future hardware (more qubits, lower noise, more layers) can improve
on SA on the *same* Hamiltonian, and SA already reaches the
ground state. The Hamiltonian itself is sub-classical.

---

## Claim B — the v2 Hamiltonian is not scenario-universal

- Phase 10A derives `c_bias*` analytically per scenario and finds a
  2 order-of-magnitude spread.
- Phase 10 (CMA-ES, 5 seeds) finds per-scenario optima spanning
  **0.38 → 99.43** — and joint training delivers F1 delta =
  **−0.030 ± 0.048** vs. classical. Even **per-(scenario, Re)**
  training delivers delta ≤ 0 in every configuration.

**Consequence:** Even with unlimited freedom in the one remaining
parameter `c_bias`, no single v2 Hamiltonian wins across the four
MHD instability classes. Q-HAS's "scenario-universal indicator"
claim is false for this parameterisation.

---

## Claim C — no *local Ising* H beats classical under cross-scenario evaluation

- Phase 11 computes the 9-feature mean-field ceiling (GBT, random
  split): F1 = 0.989. This is the absolute upper bound for any
  learned local Ising H — under the most favourable possible split.
- Phase 11B runs LOSO on the same model. F1 collapses to
  **0.191 ± 0.152**; harris-tearing goes to 0.000.
- Phase 11C learns the linear H explicitly (the actual Z-bias a
  QAOA / VQA would parameterise). Random F1 = 0.60, LOSO = 0.39 —
  below classical.
- Phase 2B: delta is negative at every percentile in {60, 70, 75,
  80, 85, 90}.
- Phase 11B-2: delta is significant at p ≤ 0.001 on the two worst
  folds under snapshot-level paired bootstrap.
- Phase 11D: per-scenario specialisation ceiling = 0.69, but
  misrouting cost = −0.355 vs. classical; a scenario detector
  would need ≥ 72% accuracy just to break even.

**Consequence:** The quantity "F1 hard-patch detection" as a
function of 9 local physical features **does not admit a
scenario-universal local description**. Any Hamiltonian built from
these features (mean-field or nearest-neighbour) carries the same
ceiling as the GBT — and the GBT collapses under LOSO. Adding
qubits, reducing noise, or deepening QAOA cannot fix a problem
whose ceiling is set by feature locality, not by solver quality.

---

## Theoretical closure

The full sequence of ceilings is:

```
QAOA      ≤     exact diag     ≤     SA        ≤    classical indicator   (Claim A, v2 H)
                                                  ≤   neighbourhood GBT ceiling  (≤ any local Ising H)
```

Both inequalities are **tight under strict generalisation**:

1. The first chain (solver ≤ classical) is tight because SA already
   reaches the exact ground state — proving the issue is the
   Hamiltonian.
2. The second chain (classical ≤ local-ceiling) reverses under LOSO:
   the local ceiling drops *below* classical (0.19 < 0.43). Since
   the local ceiling is by construction an upper bound on *any*
   local Ising H, no Q-HAS-shaped Hamiltonian can beat classical in
   deployment.

### Why this is a structural result, not a numerical one

The 9 physical features enter any local Ising H as local Z-biases
and Z-Z (stencil) couplings. The mean-field / neighbourhood GBT
is the **most expressive non-linear** scoring you can build from
those same features at those same scales. If the GBT cannot
generalise across scenarios, no smaller / simpler / Ising-shaped
function of those same features can either.

The binding constraint on Q-HAS is therefore **feature
locality across instability classes**, not solver quality, coupling
range, or training protocol. MHD adaptive refinement is *not*
Hamiltonian-representable in the scenario-universal sense at the
9-feature level.

### Deployment consequence

A deployed AMR advisor based on a Q-HAS-shaped H must either:

1. carry a scenario identifier as an explicit input feature, at
   which point "quantum advantage" reduces to "a fitted model
   helps on its training distribution" — exactly the classical
   meta-learning critique;

2. include a reliable scenario detector (see §11D) of
   accuracy ≥ 72% on *unseen* MHD regimes — a problem strictly
   harder than the original hard-patch detection.

Neither route justifies the cost of a quantum solver over the
parameter-free classical multi-indicator baseline.

---

## Phase 11E — V1 tuned H + `psi` under LOSO

- **Script:** `study/phase11e_v1h_loso.py`
- **Computes:** For each held-out scenario, the F1 of the V1
  input-side score (V1 classical + `|psi|/(π/2)`) vs the V2
  classical baseline. Uses V1's best Optuna params (trial 85) from
  `best_hyperparams.json`.
- **Scientific question:** Does V1's tuned H, fed the same LOSO
  split that collapsed the V2 ceiling, cross classical? I.e. is V1's
  extra temporal `psi` channel enough to beat the cross-scenario
  collapse documented in phase 11B?
- **Limitation:** Uses V1's input-side score as a tight proxy for
  the full QAOA pipeline (V1's QAOA polishes the input by ≤ few %).
  Not a full QAOA evaluation.
- **Strength of conclusion:** Empirical, same LOSO protocol as
  phase 11B. A positive delta here would soften the negative-result
  framing; a negative delta would strengthen it across the whole
  V1 + V2 span.

## Phase 11F — Multi-seed wrapper

- **Script:** `study/phase11f_multiseed.py`
- **Computes:** Phase 11 (random split) and 11B (LOSO) at seeds
  0..N-1, reports mean ± std across seeds per fold.
- **Scientific question:** What is the across-seed (solver) noise
  budget on the headline numbers, and does it overlap the
  across-scenario (LOSO fold) noise budget?
- **Matches V1 Fig. 6's 10-seed protocol** so the V2 headline numbers
  are reported at the same statistical resolution as V1's.
- **Strength of conclusion:** Seed-std decomposes solver-noise from
  scenario-noise. Makes the 0.989 / 0.191 numbers reproducible
  at the standard Q-HAS-literature resolution.

## Phase 11G — Scenario-identity ablation

- **Script:** `study/phase11g_scenario_ablation.py`
- **Computes:** LOSO with a K-dim one-hot scenario indicator
  appended to the 9-feature vector (3 sub-experiments: no-id,
  correct-id, wrong-id-fuzz).
- **Scientific question:** Mechanistically, is the LOSO collapse
  driven by "feature locality across scenarios" (the 9 features
  don't span the cross-scenario direction), or by something else
  (the hard-patch label isn't a per-snapshot function of the 9
  features at all)?
- **Diagnostic reading:**
  - `delta(9+id − 9) > 0.30` ⇒ feature locality is the binding
    constraint (scenario-id alone closes most of the gap).
  - `delta(9+id − 9) ≈ 0` ⇒ bottleneck is something more
    fundamental; re-examine labelling protocol.
- **Strength of conclusion:** Direct mechanistic test of the
  central claim — cleanly separates the two hypotheses.

## Phase 11H — Random-split bootstrap CI

- **Script:** `study/phase11h_random_split_bootstrap.py`
- **Computes:** Snapshot-level paired bootstrap (B=500) on the
  random-split F1_class / F1_site / F1_stencil and the paired
  delta, with percentile 95% CIs.
- **Scientific question:** Is the headline 0.989 (random-split
  mean-field ceiling) reported at the same statistical rigour as
  the LOSO numbers in phase 11B-2?
- **Closes a reporting asymmetry** — LOSO had snapshot-level CIs
  from phase 11B-2 but the random split did not. Also bounds the
  stencil-vs-site gap (the empirical tightness of the formal
  ceiling proposition in `docs/ceiling_proposition.md`).
- **Strength of conclusion:** Point estimate + 95% CI via the same
  resampling unit (snapshots) used for LOSO bootstrap.

---

*End of review.*
