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
