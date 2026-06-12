# V3 study — results log

One entry per task (§8.3 of `docs/protocol_v3_evaluation.md`): command,
git hash, numbers. Headline numbers are regenerated only by
`study/run_study_v3.sh` once it exists (Task 10).

---

## Task 0 — Regression baseline (gate)

**Commands** (run on the local workstation holding `study/results/`, conda
env `qiskit-project`; cached phase 1–2 data, no DNS regeneration needed):

```
python study/phase11_upper_bound.py --N 256 --dim 4
python study/phase11b_loso.py --N 256 --dim 4
```

**Git state:** V2 code unchanged since the published run (phase 11/11b
scripts untouched); reference outputs `logs/Result_phase11.txt`,
`logs/Result_phase11b.txt`.

**Phase 11A (random split, dim=4, N=256, 440 snaps / 7040 cells):**

| quantity | value | published | match |
|---|---|---|---|
| classical baseline F1_val (thr*=0.120) | 0.475 | 0.475 | yes |
| mean-field ceiling (gbt, 9 feats) | 0.980 | 0.980 | yes |
| neighbourhood ceiling (gbt-sten, 45 feats) | 0.980 | 0.980 | yes |

**Phase 11B (LOSO, dim=4, N=256), per-fold F1:**

| held-out | n_val | F1_class | F1_site | F1_sten |
|---|---|---|---|---|
| orszag_tang | 1920 | 0.264 | 0.327 | 0.226 |
| harris_tearing | 1280 | 0.400 | 0.000 | 0.000 |
| kelvin_helmholtz | 1920 | 0.400 | 0.344 | 0.400 |
| mhd_rotor | 1920 | 0.672 | 0.084 | 0.233 |
| **mean ± std** | | **0.434 ± 0.148** | **0.189 ± 0.150** | **0.215 ± 0.142** |

**Acceptance:** classical 0.434 and site GBT 0.189 reproduce the published
values exactly; all four per-fold rows are identical to
`logs/Result_phase11b.txt` (zero mismatch to record). The stencil mean
0.215 also matches the Task-1b reference value. Gate passed: later diffs
are attributable to V3 changes, not to data or environment drift.

---

## Task 1 — B5 score-only GBT + greedy forward selection under LOSO

**Command** (local workstation, conda env `qiskit-project`; runtime
180.5 s for the 45 subsets × 4 folds):

```
python -m pytest tests/v3/ -v
python study/v3/t1_feature_selection.py --N 256 --dim 4
```

**Git state:** code commit `2a37d00` (`v3-task-1`); the `.npz` output
(`results/t1_feature_selection_N256_dim4.npz`) records the exact git hash
and full CLI args of the run. Defaults: max-snaps=30, seed=0; folds and
data identical to phase 11b.

**Per-fold F1 under the phase-11b LOSO folds** (p75 label, prevalence 0.25):

| feature set | orszag_tang | harris_tearing | kelvin_helmholtz | mhd_rotor | mean |
|---|---|---|---|---|---|
| classical (thr on train) | 0.264 | 0.400 | 0.400 | 0.672 | 0.434 |
| B5: score_classical only (GBT) | 0.227 | 0.340 | 0.290 | 0.165 | 0.256 |
| fwd-1: +Re | 0.400 | 0.400 | 0.400 | 0.400 | 0.400 |
| fwd-2: +\|J_z\| | 0.473 | 0.333 | 0.115 | 0.374 | 0.324 |
| fwd-3: +det_grad_B | 0.435 | 0.238 | 0.039 | 0.306 | 0.255 |
| fwd-4: +score_classical | 0.279 | 0.276 | 0.358 | 0.526 | 0.360 |
| fwd-5: +\|grad_B\|^2 | 0.340 | 0.450 | 0.394 | 0.414 | 0.400 |
| fwd-6: +\|B\|^2 | 0.353 | 0.667 | 0.400 | 0.068 | 0.372 |
| fwd-7: +\|grad_v\|^2 | 0.353 | 0.657 | 0.400 | 0.072 | 0.371 |
| fwd-8: +\|omega_z\| | 0.348 | 0.608 | 0.400 | 0.084 | 0.360 |
| fwd-9: +\|v\|^2 | 0.327 | 0.000 | 0.344 | 0.084 | 0.189 |
| full-9 (B4, phase 11b repro) | 0.327 | 0.000 | 0.344 | 0.084 | 0.189 |

**Acceptance: PASS.** The full-9 row reproduces the published B4 per-fold
exactly (0.327 / 0.000 / 0.344 / 0.084, mean 0.189); the B5 row is present
(mean 0.256); pytest 5/5.

**Degeneracy flags (§1.3 B3; p = 0.25 ⇒ refine-all floor
F1 = 2p/(1+p) = 0.400, refine-none floor = 0):**
- fwd-1 (+Re) is exactly 0.400 on all four folds — the refine-all floor
  (Re is constant per configuration; the thresholded model predicts
  everything positive). The nominal "best forward subset" (k=1, Re,
  mean 0.400) is therefore a **degenerate optimum** and is excluded from
  win/loss counts. The same flag applies to fwd-5 (mean 0.400 with
  kelvin_helmholtz at 0.394–0.400 floor-adjacent values).
- The classical baseline itself sits on the refine-all floor on the
  harris_tearing and kelvin_helmholtz folds (0.400 each) — degenerate
  operating points, consistent with the published phase-11b rows.
- fwd-9 / full-9 on harris_tearing = 0.000 — the refine-none floor.

**§2 decision rule — Branch 2 selected.**
- Branch 1 ("B5 ≈ B1/B2 under LOSO while B4 collapses → the failure is ERM
  feature selection under shift"): **not selected** — B5 does not match
  classical: Δ(B5 − classical) = −0.178.
- Branch 2 ("even B5 collapses under LOSO → threshold/scale transfer
  failure; investigate per-scenario score normalisation before
  concluding"): **selected.** A GBT trained on the classical score alone
  transfers worse (0.256) than a raw threshold on the same scalar (0.434):
  the score's *ranking* information survives the shift, but the learned
  probability scale / decision threshold does not.
- Secondary observation (not a branch): B5 > full-9 by +0.067, so widening
  the feature set degrades transfer *further* — an ERM-selection effect is
  present on top of the scale-transfer failure; the forward path peaks at
  non-degenerate fwd-2 (0.324) and decays monotonically toward 0.189 after
  fwd-5, never reaching the classical 0.434.
- **Open follow-up mandated by Branch 2:** per-scenario score
  normalisation is not specified as a task in §8.3; it must be specced
  (and logged in `docs/protocol_deviations.md` if added) before any
  conclusion is drawn from this table.
- **Follow-up spec (user-approved, no deviation entry needed):** folded
  into Task 4 — `t4_blocked_split.py` will evaluate B5 with both (a) a
  globally-fitted threshold and (b) a per-fold threshold fitted on train
  splits; if (b) closes the gap to B1/B2, record as "scale transfer
  failure confirmed; ranking transfers, calibration does not."

---

## Task 1b — Information-cone curve (k-hop GBT)

**Command** (local workstation, conda env `qiskit-project`):

```
python -m pytest tests/v3/ -v          # 16 passed
python study/v3/t1b_cone_curve.py --N 256 --dim 4
```

**Git state:** code commit `6629157` (`v3-task-1b`); `.npz` output
(`results/t1b_cone_curve_N256_dim4.npz`) records git hash + CLI args.
Defaults: max-snaps=30, seed=0, train-frac=0.6. Feature counts per k:
9 / 45 / 225 / 441 (k=1 = exact phase-11 stencil construction).

**Blocked split** (Task-4 rule: first 60% of each (scenario, Re)
trajectory → train; 4224 train / 2816 val cells. Val prevalence = 0.319
— hardness concentrates in late snapshots, so the late-40% val set has
more positives than the global 0.25 — refine-all floor F1 = 0.484):

| k | n_feats | n_tr/F | F1 | Δ/hop | capped (√F/F) |
|---|---|---|---|---|---|
| classical | – | – | 0.517 | – | – |
| 0 | 9 | 469.3 | 0.581 | – | – |
| 1 | 45 | 93.9 | 0.733 | +0.153 | – |
| 2 | 225 | 18.8 | 0.816 | +0.083 | 0.858 [FLAG n_tr/F<20] |
| 3 | 441 | 9.6 | 0.816 | +0.000 | 0.866 [FLAG n_tr/F<20] |

**LOSO** (phase-11b folds; per-fold prevalence 0.250, refine-all floor
F1 = 0.400):

| k | n_feats | n_tr/F | orszag_tang | harris_tearing | kelvin_helmholtz | mhd_rotor | mean | Δ/hop |
|---|---|---|---|---|---|---|---|---|
| classical | – | – | 0.264 | 0.400 | 0.400 | 0.672 | 0.434 | – |
| 0 | 9 | 568.9 | 0.327 | 0.000 | 0.344 | 0.084 | 0.189 | – |
| 1 | 45 | 113.8 | 0.226 | 0.000 | 0.400 | 0.233 | **0.215** | +0.026 |
| 2 | 225 | 22.8 | 0.437 | 0.000 | 0.123 | 0.000 | 0.140 | −0.075 |
| 3 | 441 | 11.6 | 0.437 | 0.000 | 0.123 | 0.000 | 0.140 | +0.000 |

(k=3 capped fit under LOSO: mean 0.000 — full collapse to the
refine-none floor on every fold. [FLAG n_tr/F<20])

**Acceptance: PASS.** k=1 LOSO mean = 0.215, per-fold
0.226 / 0.000 / 0.400 / 0.233 — identical to the published phase-11b
stencil row; k=0 likewise reproduces B4 (0.189).

**Degeneracy flags (§1.3 B3):** harris_tearing = 0.000 (refine-none
floor) at every k; kelvin_helmholtz k=1 = 0.400 (refine-all floor);
mhd_rotor = 0.000 at k ≥ 2; classical sits at the 0.400 floor on
harris/kelvin folds (as in Task 1). The blocked classical baseline
(0.517) is only +0.033 above its floor (0.484) — flagged as
floor-adjacent.

**§2 decision rule — the cone is RETIRED as a scaling axis.**
- LOSO per-hop deltas: +0.026, −0.075, +0.000. The curve is **not
  rising** (no slope to quote): it peaks at k=1 and collapses *below
  the k=0 level* for k ≥ 2 (0.140 < 0.189). Strictly, the curve is not
  "flat" by the pre-registered |Δ| ≤ 0.01 criterion either — it
  *declines*, which retires the cone a fortiori: enlarging the
  neighbourhood actively hurts transfer (the extra features feed the
  ERM-selection failure documented in Task 1).
- Within-distribution (blocked split), the picture inverts: F1 rises
  through k=1 (+0.153) and k=2 (+0.083), saturating at k=3 (+0.000;
  capped fits reach 0.858–0.866). The neighbourhood information exists,
  but it is scenario-specific — consistent with §7's demotion of phase
  11h ("valid within-distribution only; never under LOSO").
- Manuscript consequence: the "larger VQA grid → larger useful cone"
  hypothesis has no classical support for *transfer*; any cone claim
  must be scoped to within-distribution fits.

---

## Task 2 — Metrics module

**Command:**

```
python -m pytest tests/v3/test_metrics.py -v
```

**Code:** `study/v3/metrics.py` — `captured_error_at_budget(scores, e,
budgets=(0.10, 0.25, 0.50))` (rank by score, top-⌈bn⌉, Σe_top/Σe; also
returns the full CE curve AUC, trapezoid on [0,1] with CE(0)=0),
`ce_curve`, `spearman`, `degeneracy_floors(p)` (refine-all
F1 = 2p/(1+p), refine-none = 0) and `degeneracy_flag(pred, prevalence,
tol=0.005, gt=None)` implementing §1.3-B3 (with gt: realised F1 within
tol of a floor; without gt: (quasi-)constant prediction, whose F1 is at
a floor by construction). API note: the optional `gt` argument extends
the §8.3 signature so the flag can test metric distance, not only
constant predictions — no metric definition was changed.

**Acceptance: PASS.** Hand-computed 6-patch example verified
(CE(0.10)=0.5, CE(0.25)=0.6, CE(0.50)=0.6, AUC=0.6166̄); refine-all
floor verified analytically against sklearn's F1 for p ∈ {0.25, 0.319}
(2p/(1+p) = 0.400, 0.4837…) and refine-none = 0; Spearman ±1 on
monotone/antitone inputs; uniform-error edge case CE(b) = ⌈bn⌉/n;
zero-total-error returns NaN. Pytest: 15 passed (31 total under tests/v3/).

---

## Task 3 — Trajectory block bootstrap

**Command:**

```
python -m pytest tests/v3/test_stats.py -v
```

**Code:** `study/v3/stats.py` — `bootstrap_by_trajectory(values,
traj_ids, B=1000, statistic=np.mean, seed=0, ci=95.0)` (resample
trajectory IDs with replacement, recompute the statistic on the pooled
values of each resample, percentile CI) and `paired_delta_bootstrap`
(§1.5 paired variant: per-trajectory deltas Δ_t = stat(a_t) − stat(b_t),
bootstrap over trajectory IDs; reports mean Δ, percentile 95% CI, and
the fraction of trajectories with Δ > 0). Deterministic given `seed`;
numpy-only.

**Acceptance: PASS.** 10 tests on synthetic data with known sampling
distribution:
- maximal within-trajectory autocorrelation (constant value per
  trajectory, T=50, m=10): bootstrap std 0.1401 vs analytic standard
  error std(effects)/√T = 0.1407;
- on the same input, the trajectory-level 95% CI is **3.06× wider**
  than the (forbidden, §1.5) snapshot-level CI (0.540 vs 0.177;
  theoretical ratio √m = 3.16) — the required demonstration that
  snapshot resampling underestimates uncertainty on autocorrelated
  data;
- paired variant: constant shift → mean Δ = 1, frac_positive = 1, CI
  excluding 0; antisymmetric shift → mean Δ = 0, frac_positive = 0.5,
  CI containing 0; unequal trajectory sizes handled (per-trajectory
  statistic first, unweighted mean of deltas);
- determinism given seed; input validation.
Pytest: 10 passed (41 total under tests/v3/).

---

## Task 4 — Temporally blocked split (B1–B7, dual aggregation, leakage)

**Command** (local workstation, conda env `qiskit-project`; dataset build
14.4 s, 440 snapshots):

```
python -m pytest tests/v3/ -v          # 48 passed
python study/v3/t4_blocked_split.py --N 256 --dim 4
```

**Git state:** code commit `36e149d` (`v3-task-4`); `.npz` output
(`results/t4_blocked_split_N256_dim4.npz`) records git hash + CLI args.
Defaults: seed=0, blocked 60/40 per (scenario, Re), random 70/30
(phase-11A permutation). B1/B2 aggregate the SAME fine-grid indicator
(`full_score`) with mean/max; runtime consistency check passed. CE/ρ
computed per snapshot (ranking the 16 patches of one timestep, V1
per-step budget semantics) then averaged; ρ against continuous e_i.

**Blocked split** (264/176 snaps; val prevalence 0.319, refine-all floor
0.484):

| method | F1 | flag | CE@0.10 | CE@0.25 | CE@0.50 | CE-AUC | ρ |
|---|---|---|---|---|---|---|---|
| B1 classical (block_avg) | 0.579 | | 0.209 | 0.386 | 0.646 | 0.595 | 0.767 |
| B2 classical (block_max) | 0.517 | | 0.195 | 0.375 | 0.631 | 0.582 | 0.365 |
| B3 refine-all | 0.484 | DEGEN | – | – | – | – | – |
| B3 refine-none | 0.000 | DEGEN | – | – | – | – | – |
| B4 gbt-9 (max) | 0.581 | | 0.182 | 0.345 | 0.636 | 0.580 | 0.640 |
| B4 gbt-9 (avg) | 0.738 | | 0.187 | 0.382 | 0.640 | 0.587 | 0.694 |
| B5 gbt-score (max, thr global) | 0.415 | | 0.183 | 0.351 | 0.622 | 0.572 | 0.276 |
| B5 gbt-score (max, thr per-cfg) | 0.380 | | 0.183 | 0.351 | 0.622 | 0.572 | 0.276 |
| B5 gbt-score (avg, thr global) | 0.518 | | 0.179 | 0.359 | 0.626 | 0.576 | 0.330 |
| B5 gbt-score (avg, thr per-cfg) | 0.548 | | 0.179 | 0.359 | 0.626 | 0.576 | 0.330 |
| B6 linear-H (max) | 0.489 | DEGEN | 0.196 | 0.377 | 0.639 | 0.589 | 0.524 |
| B6 linear-H (avg) | 0.463 | | 0.195 | 0.361 | 0.601 | 0.573 | 0.569 |
| B7 random ranking | 0.480 | DEGEN | 0.121 | 0.246 | 0.507 | 0.499 | 0.019 |

**Random split, phase-11A reproduction** (308/132 snaps; val prevalence
0.265, floor 0.419):

| method | F1 | flag | CE@0.10 | CE@0.25 | CE@0.50 | CE-AUC | ρ |
|---|---|---|---|---|---|---|---|
| B1 classical (block_avg) | 0.492 | | 0.215 | 0.405 | 0.632 | 0.592 | 0.654 |
| B2 classical (block_max) | 0.475 | | 0.206 | 0.396 | 0.623 | 0.585 | 0.395 |
| B3 refine-all | 0.419 | DEGEN | – | – | – | – | – |
| B3 refine-none | 0.000 | DEGEN | – | – | – | – | – |
| B4 gbt-9 (max) | 0.980 | | 0.214 | 0.415 | 0.645 | 0.599 | 0.792 |
| B4 gbt-9 (avg) | 0.975 | | 0.214 | 0.415 | 0.641 | 0.597 | 0.750 |
| B5 gbt-score (max, thr global) | 0.538 | | 0.193 | 0.372 | 0.622 | 0.578 | 0.390 |
| B5 gbt-score (max, thr per-cfg) | 0.554 | | 0.193 | 0.372 | 0.622 | 0.578 | 0.390 |
| B5 gbt-score (avg, thr global) | 0.584 | | 0.198 | 0.396 | 0.620 | 0.581 | 0.354 |
| B5 gbt-score (avg, thr per-cfg) | 0.593 | | 0.198 | 0.396 | 0.620 | 0.581 | 0.354 |
| B6 linear-H (max) | 0.598 | | 0.205 | 0.395 | 0.630 | 0.589 | 0.595 |
| B6 linear-H (avg) | 0.568 | | 0.199 | 0.364 | 0.605 | 0.573 | 0.619 |
| B7 random ranking | 0.415 | DEGEN | 0.130 | 0.255 | 0.505 | 0.502 | −0.005 |

**Acceptance: PASS.** Random-split B2 = 0.475 and B4 gbt-9 (max) = 0.980
match Task 0 exactly. Pytest 48/48.

**Leakage quantification (§7, reported once; F1 random − blocked):**
B4 (max) **+0.399** (0.980 → 0.581), B4 (avg) +0.237, B5 +0.045…+0.174,
B6 ≈ +0.11; classical baselines have *negative* gaps (−0.04…−0.09).
Caveat for reading F1 gaps: the two val sets have different prevalence
(0.319 blocked vs 0.265 random), so floors shift (+0.065); the
threshold-free columns are the clean comparison. There, B4 (max)
CE-AUC drops only 0.599 → 0.580: **the phase-11 0.98 "ceiling" was a
binary-view artifact of near-duplicate leakage — the model's actual
ranking of patches by coarsening error was never much better than the
classical baseline, even on the leaky split** (design principle §0.2
vindicated: continuous quantities first).

**Aggregation finding (dual-aggregation rule §5.3 paying off):**
B1 (block_avg) dominates B2 (block_max) on every metric of both splits;
most strikingly ρ = 0.767 vs 0.365 on the blocked split. The V1
aggregation convention is substantially better aligned with the
continuous error than the V2 convention that phases 11/11b used. B4
(avg) > B4 (max) by +0.157 F1 blocked. Under the blocked split, the
best F1 is B4 (avg) 0.738, but the best ranking metrics belong to the
plain B1 classical baseline (CE@0.10 = 0.209, ρ = 0.767) — the learned
models mainly relocate the decision boundary; they do not rank better.

**B5 dual-threshold check (user-approved Branch-2 follow-up):
conditional NOT met — outcome mixed.**
- (avg): per-cfg threshold helps, 0.518 → 0.548 (+0.030), closing about
  half the gap to B1 (residual −0.031).
- (max): per-cfg threshold *hurts*, 0.415 → 0.380 (−0.035), widening the
  gap to B2.
- Therefore "scale transfer failure confirmed; ranking transfers,
  calibration does not" is **not recorded** from this test: per-config
  calibration does not consistently close the gap within distribution.
  Note the scope limit: this blocked-split test calibrates per
  (scenario, Re) trajectory seen in training; it does not test the
  cross-scenario calibration that failed in Task 1's LOSO. The
  Branch-2 question stays open pending Task 5/L2 evidence.

**Degeneracy flags:** B7 flagged on both splits (0.480 vs floor 0.484;
0.415 vs 0.419 — within tol 0.005, flag mechanism validated on real
data); B6 linear-H (max) flagged on the blocked split (0.489 vs floor
0.484); B3 rows flagged by construction and excluded from win/loss
counts.

---

## Task 5 — Phase 11E rerun, fixed (signed ψ, trial-4 params, dual aggregation)

**Command** (local workstation, conda env `qiskit-project`; score-variant
build 23.2 s):

```
python -m pytest tests/v3/ -v          # 59 passed
python study/v3/t5_v1_psi_loso.py --N 256 --dim 4
```

**Git state:** code commit `f08afdf` (`v3-task-5`); `.npz` output
(`results/t5_v1_psi_loso_N256_dim4.npz`) records git hash + CLI args.
Param discrepancy logged: trial #4 (`config.TRAINED_*`) β=9.94,
thr=0.1496 vs trial #85 (phase-11e hardcoded) β=0.5495, thr=0.3044.
Signed ψ: per-edge V1 machinery (shared ⟨|ΔΦ|⟩ normalisation as in
phase 11e), sign-preserving larger-magnitude combination,
`score_aug = clip(score + 0.5·sin(ψ), 0, 1)`; ψ_v2 = `compute_psi_v2`
per edge component, same combination; legacy |ψ| kept only as the 11e
comparison arm.

**LOSO F1, block_avg** (per-fold prevalence 0.250, refine-all floor 0.400):

| variant | orszag_tang | harris_tearing | kelvin_helmholtz | mhd_rotor | mean |
|---|---|---|---|---|---|
| v2-classical | 0.396 | 0.400 | 0.400 | 0.927 | 0.531 |
| v1-classical (no ψ) | 0.395 | 0.400 | 0.400 | 0.936 | 0.533 |
| v1+ψ signed β=9.94 (trial4) | 0.384 | 0.385 | 0.438 | 0.025 | 0.308 |
| v1+ψ signed β=0.5495 (trial85) | 0.368 | 0.400 | 0.418 | 0.000 | 0.296 |
| v1+ψ legacy-abs β=9.94 (trial4) | 0.412 | 0.385 | 0.411 | 0.790 | 0.500 |
| v1+ψ legacy-abs β=0.5495 (11e repro) | 0.306 | 0.385 | 0.411 | 0.928 | 0.507 |
| v1+ψ_v2 signed (param-free) | 0.396 | 0.400 | 0.417 | 0.000 | 0.303 |

**LOSO F1, block_max:**

| variant | orszag_tang | harris_tearing | kelvin_helmholtz | mhd_rotor | mean |
|---|---|---|---|---|---|
| v2-classical | 0.264 | 0.400 | 0.400 | 0.672 | 0.434 |
| v1-classical (no ψ) | 0.275 | 0.400 | 0.400 | 0.796 | 0.468 |
| v1+ψ signed β=9.94 (trial4) | 0.227 | 0.385 | 0.411 | 0.786 | 0.453 |
| v1+ψ signed β=0.5495 (trial85) | 0.239 | 0.385 | 0.412 | 0.770 | 0.452 |
| v1+ψ legacy-abs β=9.94 (trial4) | 0.276 | 0.385 | 0.411 | 0.777 | 0.462 |
| v1+ψ legacy-abs β=0.5495 (11e repro) | 0.282 | 0.385 | 0.411 | 0.748 | 0.457 |
| v1+ψ_v2 signed (param-free) | 0.320 | 0.385 | 0.411 | 0.756 | 0.468 |

**2×2×2 summary (mean LOSO F1; params × ψ-handling × aggregation):**

| ψ-handling | params | block_avg | block_max |
|---|---|---|---|
| signed | trial4 | 0.308 | 0.453 |
| signed | trial85 | 0.296 | 0.452 |
| legacy-abs | trial4 | 0.500 | 0.462 |
| legacy-abs | trial85 | 0.507 | 0.457 |
| signed ψ_v2 | param-free | 0.303 | 0.468 |

**Trajectory bootstrap (Task 3; n_traj=16, B=1000; Δ = F1(variant) −
F1(v1-classical), paired per trajectory):**

| variant | agg | mean Δ | 95% CI | frac>0 |
|---|---|---|---|---|
| v1+ψ signed trial4 | avg | −0.225 | [−0.414, −0.049] | 0.38 |
| v1+ψ signed trial4 | max | −0.015 | [−0.027, −0.004] | 0.31 |
| v1+ψ signed trial85 | avg | −0.236 | [−0.423, −0.058] | 0.25 |
| v1+ψ signed trial85 | max | −0.016 | [−0.027, −0.006] | 0.31 |
| v1+ψ legacy-abs trial4 | avg | −0.033 | [−0.071, +0.001] | 0.50 |
| v1+ψ legacy-abs trial4 | max | −0.005 | [−0.013, +0.002] | 0.44 |
| v1+ψ legacy-abs trial85 (11e) | avg | −0.026 | [−0.051, −0.005] | 0.44 |
| v1+ψ legacy-abs trial85 (11e) | max | −0.011 | [−0.024, +0.001] | 0.44 |
| v1+ψ_v2 signed (param-free) | avg | −0.229 | [−0.412, −0.052] | 0.38 |
| v1+ψ_v2 signed (param-free) | max | +0.000 | [−0.015, +0.018] | 0.56 |
| (ref) v1-class − v2-class | avg | +0.002 | [−0.000, +0.005] | 0.31 |
| (ref) v1-class − v2-class | max | +0.033 | [+0.009, +0.062] | 0.50 |

**Acceptance: PASS** (2×2×2 table delivered; pytest 59/59; legacy
trial-85 block_avg arm retained as the phase-11e reproduction).

**Statement: "ψ does not transfer" SURVIVES the fair test — and is
strengthened.** Under the pre-registered signed handling, ψ is
significantly *harmful* under LOSO, not merely useless:
- block_avg: Δ ≈ −0.23 with 95% CIs excluding 0 for all three signed
  variants (trial-4, trial-85, parameter-free ψ_v2). The damage is
  concentrated on mhd_rotor (0.936 → 0.025/0.000, the refine-none
  floor — DEGEN-flagged cells): the ±0.5·sin(ψ) term, with ψ saturated
  by the rotor's coherent flux changes, dominates the [0,1] score and
  destroys the ranking that the classical indicator had.
- block_max: small but still significantly negative for the V1-β signed
  variants (CIs exclude 0); the parameter-free ψ_v2 is exactly null
  (+0.000 [−0.015, +0.018]).
- The β/params axis is immaterial: trial-4 vs trial-85 differ by
  ≤ 0.012 mean F1 in every cell — the tanh saturates similarly at both
  β values, so the V1/V2 calibration discrepancy explains nothing.
- The legacy |ψ| arm reproduces the published phase-11e conclusion
  (small negative-to-null deltas; Lohner carries V1's modest edge:
  v1-class − v2-class = +0.033 [+0.009, +0.062] under block_max).

Scope per §3: this is the h=0 (instantaneous-label) prerequisite. The
anticipation claim proper (e_i(t+h) targets, easy-now-hard-later
subset, CE(b) metrics) is decided at Level 2 (Task 7); given the h=0
outcome, §3's first decision branch ("ψ adds nothing at any h → the
anticipation claim of V1 is retired, with a fair test on record") is
the live hypothesis Task 7 must confirm or refute.

**Degeneracy flags:** all harris/kelvin baseline cells sit at the
refine-all floor 0.400 (both aggregations, as in Tasks 1/1b); signed-ψ
mhd_rotor block_avg cells at/adjacent to the refine-none floor (0.000
flagged; 0.025 floor-adjacent); ψ-variant harris cells 0.385 and
kelvin cells 0.411–0.438 are floor-adjacent but outside the 0.005 tol.

---

## Task 6 — Dynamic ground truth d_i (pilot)

**Command** (local workstation, conda env `qiskit-project`):

```
python -m pytest tests/v3/ -v          # 68 passed
python study/v3/t6_dynamic_gt.py       # bare command = the mandated pilot
```

**Git state:** code commit `ce7763c` (`v3-task-6`); output
`results/d_patches_orszag_tang_Re400_N128_dim4.npz` (phase-2 key
layout, full-DNS-length `l2_errors` with NaN at uncomputed snapshots,
`computed_mask`/`snap_indices` for alignment; git hash + CLI args
inside). Pilot data provenance: no N=128 DNS exists, so the pilot
fields are the N=256 DNS block-averaged 2× (`--source-N 256`, logged
in the run output).

**Pilot (N=128, orszag_tang Re=400, 2 snapshots, all 16 patches,
δt=0.1, CFL 0.4):**

| snap | t | solver steps | evolutions | d_i range | wall-clock |
|---|---|---|---|---|---|
| 14 | 1.40 | 18 | 17 | [8.75e-02, 2.11e-01] | 17.4 s |
| 29 | 2.90 | 21 | 17 | [1.26e-01, 1.90e-01] | 23.2 s |

**Acceptance: PASS.**
- Pilot completes; pytest 68/68 (incl. a real miniature-solver
  integration test: bit-exact d=0 on constant patches via dt-sequence
  replay, argmax(d) = argmax(e)).
- Sanity check: **Spearman(d_i, e_i) = 0.989** (n = 32 patches) > 0.

**Pilot-scope observation (one config, n=32 — not a claim):** the
static coarsening error e_i is a near-perfect *ranking* proxy for the
dynamic evolution error d_i at δt = 0.1 on this config. If this holds
across scenarios/Re in the full campaign, the d_i-target variant of
Level 2 (Task 7) is expected to closely mirror the e_i-target variant.

**Cost projection (§8.4 gate):** 20.3 s/snapshot at N=128 (17
evolutions of δt=0.1). Scaling ×(256/N)³ = ×8 (cells ×4, CFL steps ×2)
→ full campaign (N=256, 16 configs × 10 snapshots) ≈ **7.2 h** on the
local workstation. Decision on launching the full N=256 run (as-is /
reduced snapshot count / HPC) rests with the user; not launched in
this session.

---

## Task 7 — Predictive dataset, Level 2 (horizons, ψ, causal cone)

**Command** (local workstation, conda env `qiskit-project`; sequence
build 18.3 s):

```
python -m pytest tests/v3/ -v          # 78 passed
python study/v3/t7_horizon.py --N 256 --dim 4
```

**Git state:** code commits `37aaec5` + `c062ce4` (`v3-task-7`); output
`results/t7_horizon_N256_dim4.npz` (git hash + CLI args inside).
Fix recorded: harris_tearing trajectories have 20 snapshots (others
30), so they have **no blocked val pairs at h=8**; the paired ψ-delta
bootstrap runs on the intersection of trajectories present in both
arms (n_tr column; 12 instead of 16 on blocked h=8 rows).
Targets: e_i(t+h) only — no `d_patches_*` files at N=256/dim=4 yet
(Task-6 full campaign pending); per §8.4 the e-variant needs no new
simulation.

**Headline numbers (CE@0.25 against continuous e(t+h); LOSO = mean
over folds):**

| method | split | h=1 | h=2 | h=4 | h=8 |
|---|---|---|---|---|---|
| B1 classical score (avg) | blocked | 0.384 | 0.381 | 0.378 | 0.384 |
| base9 GBT | blocked | 0.365 | 0.374 | 0.370 | 0.370 |
| base9+ψ4 | blocked | 0.374 | 0.374 | 0.372 | 0.361 |
| B1 classical score (avg) | LOSO | 0.412 | 0.412 | 0.410 | 0.404 |
| base9 GBT | LOSO | 0.215 | 0.243 | 0.227 | 0.207 |
| base9+ψ4 | LOSO | 0.241 | 0.235 | 0.216 | 0.227 |
| base9+ψv2 | LOSO | 0.271 | 0.244 | 0.216 | 0.198 |

Every learned predictor collapses under LOSO to CE@0.25 ≈ 0.20–0.30 vs
the raw classical score's 0.40–0.41, at **every** horizon — the L1
transfer failure (Tasks 1/1b/4) propagates unchanged to L2. Most LOSO
F1 cells are DEG-flagged (transferred thresholds sit at degeneracy
floors), so the high LOSO ENHL-recall values of the baselines
(0.89–0.95) are an artifact of near-refine-all operating points; the
ranking columns are the meaningful ones.

**Lead-time tables (capture@0.25 of future-hard patches vs h):**

| method | split | h=1 | h=2 | h=4 | h=8 |
|---|---|---|---|---|---|
| B1 classical (avg) | blocked | 0.693 | 0.687 | 0.672 | 0.615 |
| B1 classical (avg) | LOSO | 0.694 | 0.697 | 0.686 | 0.665 |
| B2 classical (max) | LOSO | 0.499 | 0.498 | 0.517 | 0.532 |
| base9 GBT | LOSO | 0.283 | 0.350 | 0.317 | 0.098 |
| base9+ψ4 | LOSO | 0.333 | 0.285 | 0.285 | 0.199 |
| base9+ψv2 | LOSO | 0.323 | 0.314 | 0.303 | 0.267 |

**ψ deltas (CE@0.25 per trajectory, paired bootstrap B=1000):**

| pair | split | h | n_tr | mean Δ | 95% CI | frac>0 |
|---|---|---|---|---|---|---|
| +ψ4 − base9 | blocked | 1 | 16 | +0.008 | [+0.002, +0.016] | 0.62 |
| +ψv2 − base9 | blocked | 1 | 16 | +0.006 | [+0.001, +0.011] | 0.75 |
| full − base9+D9 | blocked | 1 | 16 | +0.001 | [+0.000, +0.001] | 0.44 |
| +ψ4 − base9 | blocked | 2 | 16 | +0.000 | [−0.001, +0.001] | 0.44 |
| +ψv2 − base9 | blocked | 2 | 16 | +0.001 | [+0.000, +0.002] | 0.50 |
| +ψ4 − base9 | blocked | 4 | 16 | +0.002 | [+0.001, +0.003] | 0.50 |
| +ψv2 − base9 | blocked | 4 | 16 | +0.003 | [+0.001, +0.005] | 0.62 |
| +ψ4 − base9 | blocked | 8 | 12 | −0.008 | [−0.026, +0.005] | 0.75 |
| +ψv2 − base9 | blocked | 8 | 12 | +0.007 | [+0.003, +0.011] | 0.83 |
| full − base9+D9 | blocked | 8 | 12 | −0.018 | [−0.038, −0.000] | 0.67 |
| +ψ4 − base9 | LOSO | 1 | 16 | +0.026 | [+0.004, +0.050] | 0.38 |
| +ψv2 − base9 | LOSO | 1 | 16 | +0.056 | [+0.012, +0.101] | 0.50 |
| full − base9+D9 | LOSO | 1 | 16 | −0.013 | [−0.024, −0.004] | 0.44 |
| +ψ4 − base9 | LOSO | 2 | 16 | −0.008 | [−0.015, −0.002] | 0.06 |
| +ψv2 − base9 | LOSO | 2 | 16 | +0.001 | [−0.006, +0.009] | 0.19 |
| +ψ4 − base9 | LOSO | 4 | 16 | −0.011 | [−0.020, −0.003] | 0.12 |
| +ψv2 − base9 | LOSO | 4 | 16 | −0.011 | [−0.023, −0.000] | 0.50 |
| +ψ4 − base9 | LOSO | 8 | 16 | +0.020 | [+0.006, +0.034] | 0.62 |
| +ψv2 − base9 | LOSO | 8 | 16 | −0.008 | [−0.024, +0.005] | 0.44 |

**Causal-cone k×h matrices:**

blocked F1: k=0: 0.645/0.628/0.499/0.415; k=1: 0.734/0.706/0.648/0.610;
k=2: 0.791/0.722/0.643/0.442 (h = 1/2/4/8).
blocked CE@0.25: flat (0.365–0.386 everywhere).
LOSO CE@0.25: k=0: 0.215/0.243/0.227/0.207; k=1: 0.285/0.302/0.269/
0.239; k=2: 0.201/0.237/0.304/0.226. LOSO F1: all collapsed (0.10–0.30).

**Acceptance: PASS** (lead-time tables + k×h matrix delivered; pytest
78/78).

**§3 decision rule — Branch 1 selected: "ψ adds nothing at any h, any
split → the anticipation claim of V1 is retired, with a fair test on
record."**
- LOSO ψ deltas are sign-inconsistent across adjacent horizons
  (positive h=1, negative h=2/h=4, mixed h=8) and the positive cells
  are minority-driven (frac>0 ≤ 0.62; ψv2 h=1: mean +0.056 with only
  half the trajectories positive). No horizon has both ψ variants
  agreeing positively.
- Blocked deltas are formally positive at h=1 (CIs exclude 0) but of
  magnitude ≤ +0.008 CE@0.25 — recorded verbatim; ≈ 2% relative, an
  order of magnitude below the B1−GBT gap.
- Branch 3 (ENHL subset under LOSO) not triggered: ψ-augmented GBTs do
  beat the ψ-less GBT on subset-CE at some horizons (e.g. h=8: 0.318
  vs 0.120), but all remain below the raw classical baseline's subset
  capture at every h (B1: 0.32–0.46) — value added inside a collapsed
  regime, not value. No L3 escalation from L2.

**Secondary findings:**
- **Persistence headline:** the classical score at time t captures
  0.665–0.697 of future-hard patches at budget 0.25 under LOSO,
  essentially flat from h=1 to h=8. Hardness is persistent at these
  horizons: anticipation is unnecessary, which is *why* a fair ψ test
  finds nothing to anticipate. The learned predictors' capture decays
  to 0.10–0.28 by h=8.
- **Causal-cone (physically grounded form) retired:** the k-gain
  *shrinks* with h instead of growing (blocked F1 gain k1→k2: +0.057
  at h=1 → −0.005 at h=4), opposite to the advection prediction.
  Consistently with the physics: k* ≈ v·h·Δt_snap/Δx_patch ≈ 0.5 hop
  at h=8 (v ~ O(1), Δt_snap = 0.1, Δx_patch = 2π/4) — the tested
  horizons never advect hardness beyond one patch, so no growing cone
  is expected or observed. Confirms Task 1b's retirement.
- Level-2 verdict joins Level 1: under §6, the running narrative is
  "levels 1 and 2 negative", with the persistence result and the
  inductive-bias inversion (raw physics score transfers, learned
  models collapse) as the mechanism.

---

## Task 8 — Data extension (8 scenarios × physics seeds)

**Commands** (local workstation, conda env `qiskit-project`; ~hours of
DNS for the overnight campaign):

```
python -m pytest tests/v3/ -v                      # 100 passed
python study/v3/t8_dns_extension.py --dry-run      # plan: 48 runs
python study/v3/t8_dns_extension.py                # overnight campaign
python study/v3/t8_dns_extension.py --scenario kelvin_helmholtz \
    --phys-seed 1 --noise-amplitude 0.005 --no-skip-existing   # D1 final
python study/v3/t8_dns_extension.py --validate-only            # 64/64 OK
```

**Git state:** code commits `a67da73` (wrapper), `4347129` (D2 corrected
KH observable + `--validate-only`), `deaf33d` (D1 final amplitude).
Every `.npz` records git hash, CLI args, `phys_seed` and noise
amplitude. Phase-2 labels regenerated for every trajectory
(seed-aware naming `_seed{k}`; seed 0 keeps V2 names).

**Non-pre-registered parameters (logged, user-approved):** new-scenario
run lengths — lamb_oseen t_max=3.0, island_coalescence / double_tearing
/ magnetic_twist t_max=2.0, all snapshot_dt=0.10; physics-seed noise =
band-limited (|k| ≤ 8) Gaussian on (vx, vy) + div-free projection,
amplitude 0.1 (V1 KH level) for 7 scenarios, **0.005 for KH**
(deviation **D1**, `docs/protocol_deviations.md`).

**Presence matrix:** 8 scenarios × 2 seeds × 4 Re — complete
(64 trajectories, 4 per cell).

**Validation (corrected KH observable, deviation D2): 64/64 OK.**
- div B ≤ 1.5e-4 and monotone energy decay on every trajectory.
- OT decay 9.4–15.2% (within the 1–45% window) at both seeds.
- KH growth, corrected observable: seed-0 **1.41–1.43×**, seed-1
  (amplitude 0.005) **1.36–1.37×** — vs the 1.37× predicted by the
  calibrated dilution model. **D2:** the published phase-1b
  `fluctuating_KE` subtracts the Y-mean and leaves the base shear
  profile (variance ≈ 0.341 vs perturbation ≈ 2.5e-4) inside Ep, so
  the original check read ≈ 1.00× for every trajectory including
  seed 0 — it could never detect KH growth. The v3 copy subtracts the
  X-average (same windows, same > 1.1× criterion); phase 1b is
  untouched.
- Tearing-like ⟨J²⟩ checks: seed-0 amplifications 2.65–23.8× with
  late peaks; seed-1 trajectories show **early reconnection onsets**
  (harris 6.2–6.6e4× at t ≈ 0.40; island_coalescence ≈ 2.5–2.7e3× at
  t ≈ 0.40; double_tearing ≈ 4.1–4.5e3× at t ≈ 0.30) — the physics
  seeds produce genuinely distinct dynamical histories, exactly what
  trajectory-level statistics (§1.5) need.

**Acceptance: PASS** — 8 scenarios × ≥ 2 seeds present, validation log
clean. The §1.1 dataset now supports ≥ 8 LOSO folds; remaining §1.1
gap: ≥ 5 physics seeds per (scenario, Re) (this run delivers 2; the
wrapper takes `--phys-seed 0 1 2 3 4` unchanged).

---

## Task 9 — Proposition-2 strict mean-field condition

**Command** (local workstation, conda env `qiskit-project`; 34.6 s for
128 (config, dim, mapper) entries):

```
python -m pytest tests/v3/ -v          # 109 passed
python study/v3/t9_prop2_check.py --N 256 --dim 2 4
```

**Git state:** code commit `5077a0a` (`v3-task-9`); output
`results/t9_prop2_N256.npz` (git hash + CLI args inside). Condition as
pre-registered (§2): per site i, Σ_{j∋i} 2|C_ij| + Σ_{p∋i} 4|K_p| <
|h_i| over the 2·dim² edge-qubits, topology mirroring
`create_period_hamiltonian` exactly (bookkeeping verified in pytest
against an exhaustive 256-state Ising minimizer: condition everywhere
⇒ exact ground state = −sign(h); strong-coupling counterexample
included). Mappers: v1 = `TRAINED_*` (trial #4); v2 = parameter-free
(c_bias=1.0, thr=0.15). 8 scenarios × 4 Re × seed 0, 30 snaps/config.

**Fraction of sites satisfying the strict condition (mean over
snapshots):**

dim=2 (8 qubits):

| scenario | v1 (Re 400/800/1200/1600) | v2 (Re 400/800/1200/1600) |
|---|---|---|
| orszag_tang | 0.000 / 0.008 / 0.000 / 0.000 | 0 / 0 / 0 / 0 |
| harris_tearing | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| kelvin_helmholtz | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| mhd_rotor | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| lamb_oseen | 0.179 / 0.058 / 0.008 / 0.000 | 0 / 0 / 0 / 0 |
| island_coalescence | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| double_tearing | 0 / 0 / 0 / 0 | 0.400 / 0.175 / 0.225 / 0.200 |
| magnetic_twist | 0 / 0 / 0 / 0 | 0.025 / 0.025 / 0.025 / 0.025 |
| **MEAN** | **0.008** | **0.034** |

dim=4 (32 qubits):

| scenario | v1 (Re 400/800/1200/1600) | v2 (Re 400/800/1200/1600) |
|---|---|---|
| orszag_tang | 0.068 / 0.074 / 0.073 / 0.072 | 0 / 0 / 0 / 0 |
| harris_tearing | 0.700 / 0.466 / 0.456 / 0.472 | 0 / 0 / 0 / 0 |
| kelvin_helmholtz | 0.257 / 0.173 / 0.135 / 0.110 | 0 / 0 / 0 / 0 |
| mhd_rotor | 0.022 / 0.015 / 0.010 / 0.010 | 0 / 0 / 0 / 0 |
| lamb_oseen | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| island_coalescence | 0.453 / 0.517 / 0.512 / 0.506 | 0 / 0 / 0 / 0 |
| double_tearing | 0.000 / 0.000 / 0.031 / 0.000 | 0 / 0 / 0 / 0 |
| magnetic_twist | 0.600 / 0.525 / 0.425 / 0.375 | 0 / 0 / 0 / 0 |
| **MEAN** | **0.221** | **0.000** |

**Acceptance: PASS** (table delivered; pytest 109/109).

**Reading for the conditional Proposition 2 (manuscript):**
- **The premise is essentially never satisfied for the V2 Hamiltonian**
  (exactly 0.000 at dim=4 on all 32 configs; 0.034 at dim=2, carried
  only by double_tearing/magnetic_twist). The strict sufficient
  condition therefore CANNOT certify the per-site decision for V2: the
  couplings are not provably decorative. Since phase 11A showed
  empirically that couplings add ≤ +0.002 F1 (stencil ≈ mean-field)
  and Task 1b retired the cone, the correct manuscript framing is:
  the per-site sufficiency of the V2 H is an *empirical* finding, not
  a theorem-backed one — Prop 2 is stated conditionally and its
  premise reported false in practice.
- **V1 at dim=4 is partially certified** (mean 22% of sites; strongly
  scenario-dependent: harris 0.47–0.70, magnetic_twist 0.38–0.60,
  island_coalescence 0.45–0.52, but ≈ 0 for lamb_oseen/rotor/
  double_tearing). At dim=2 the premise fails almost everywhere for
  both mappers — each qubit touches a larger share of the couplings on
  the small graph, so the incident-coupling sum is relatively larger.
- The fractions are a statement about the *sufficient* condition only:
  a 0 fraction does not imply the couplings matter, it implies the
  cheap proof route is unavailable (consistent with §2's replacement
  of "ceiling" language by "selection" language).

---

## Task 10 — Orchestration + v3 master table

**Command** (local workstation, conda env `qiskit-project`; full
regeneration ≈ 12 min, step logs under `logs/v3/`):

```
bash study/v3/run_study_v3.sh --all
```

**Git state:** code commit `ca7f815` (`v3-task-10`). Path note: §5.4
cites `study/run_study_v3.sh`; Task 10 (§8.3) and the §8.2 guardrail
("all new code in study/v3/") place it at `study/v3/run_study_v3.sh` —
resolved in favour of §8.3, documented in the script header.

**Pipeline executed:** pytest gate (118 passed) → phase 11A/11B
(Task-0 regression, V2 scripts untouched) → t1 → t1b → t4 → t5 →
t6 pilot → t7 → t9 → `t10_aggregate.py --strict`. The aggregator
emits `results/v3_master_table.{md,csv}` + `v3_master_N256.npz`
(git hash + CLI args inside); every row carries the reference value
from this log and a status OK / DIFF / MISSING (tolerance 0.002).

**Acceptance: PASS — clean regeneration, 51 rows: OK=51, DIFF=0,
MISSING=0** at commit `ca7f8153f416`. Every headline number of Tasks
0–9 reproduced exactly: the Task-0 anchors (0.475 / 0.980 / 0.434 /
0.189 / 0.215), the Task-1 B5 collapse (0.256), the Task-1b cone
curves (both splits, all k), the Task-4 leakage gap (+0.399) and
aggregation contrast (ρ 0.767 vs 0.365), the Task-5 signed-ψ bootstrap
CIs, the Task-6 pilot Spearman (0.989, recomputed from stored d/e
arrays), the Task-7 ψ-deltas, B1 capture rows and cone×horizon cells,
and the Task-9 Prop-2 fractions.

**§5 status:** rule 4 (single source of truth) is now operational —
the manuscript cites only `t10_aggregate.py` output, regenerated by
one command from the tagged commit. The §8.3 brief (Tasks 0–10) is
complete. Out of scope of this brief and still open: the Task-6 full
d_i campaign (≈ 4.6 h at current hardware, per the regenerated pilot
projection), the §1.1 ≥ 5-seed extension (`t8_dns_extension.py
--phys-seed 0 1 2 3 4`), and Level 3 (§4), which per §8.4 receives a
separate brief now that Tasks 0–7 have fixed the decision engines
worth the closed-loop cost.
