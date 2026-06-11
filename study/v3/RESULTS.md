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
