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
