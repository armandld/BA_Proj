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
