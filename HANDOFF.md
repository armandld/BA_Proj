# Q-HAS V4 — hand-off for merge

**Branch:** `claude/kind-babbage-927g10` · **Base:** `origin/main`
**Scope:** 199 commits, 186 files, ~23 000 added lines
(76 files are tracked artifacts under `study/results/`).
**The closed-loop study is closed** — see *CLOSING THE CLOSED-LOOP STUDY*
at the end of `study/v4/RESULTS_V4.md` for the one-sentence result, the
evidence ranked by strength, and what would overturn it.

**Read `docs/CODE_REVIEW_GUIDE.md` first** — it orders the review by risk
per minute. This file is the single entry point: what was done, what it
concluded, what is unfinished, and what to check before merging.

---

## 1. Merge checklist

```bash
BASE=$(git merge-base HEAD origin/main)
git diff --name-only $BASE HEAD -- src/                            # MUST be empty (V1 read-only)
git diff --name-only $BASE HEAD -- study/ \
  | grep -v '^study/v[34]/' | grep -v '^study/results/'   # MUST be empty (V2 code read-only)
python -m pytest tests/v3 tests/v4 -q                              # 298 passed, 20 skipped
python study/v4/t16_aggregate_v4.py                                # 129 rows, 0 DIFF (4 MISSING = ot/kh leak-free, in flight)
```

All four held: `src/` **0** files changed, V2 phase **code** **0** files
changed (the 76 files under `study/results/` are tracked artifacts, not V2
code — `study/results/` was un-ignored so the numbers are verifiable from a
fresh clone),
**298 pytests passed** (20 skipped), master table **125 OK, 0 DIFF, 4
MISSING**. All four Level-3 folds are present; the 4 MISSING rows are the
`ot`/`kh` leak-free runs, which were still in flight when this was written.
`MISSING` means "not produced yet", never "produced and lost".

Nothing outside `study/v3/`, `study/v4/`, `tests/v3/`, `tests/v4/`, `docs/`,
`figures_v4/`, `logs/v4/`, `CLAUDE.md` was modified.

**`bash run_tests.sh` fails 8 tests — do not treat this as a regression.**
It is defect D6, pre-existing, reproduced at `cf93ba3` (the last commit
touching `src/`/`tests/`, before any of this work).

---

## 2. The artifacts are in the repo

`study/results/` **is tracked** — 76 artifacts, 6.3 MB of `.json`/`.npz`.
Every number in these documents is reproducible from a fresh clone. Earlier
drafts of this file said the opposite, because the directory was gitignored
for most of the campaign and the container has been reclaimed eight times;
un-ignoring it is what removed that dependency.

Only `study/results/dns_*_N256.npz` stays out — 4 files, 94 MB of DNS
reference snapshots. They are **inputs**, not results, and the solver is
deterministic, so `t15_level3_closed_loop.py` regenerates them identically.

`study/v4/t16_aggregate_v4.py` is the transcription cross-check: it
recomputes each published number from its artifact and diffs against the
Markdown. It prints `0 DIFF` over 129 rows.

Cheap to regenerate (seconds–minutes): T11, T11b, T12, T13, T14, T17, T18.
Expensive (hours): the Level-3 campaign, T20, T22.

---

## 3. What this branch establishes

The audit asked for a mechanism explaining V1's marginal gains. The answer,
at the deployed size (`VQA_N = 2` → 8 qubits):

| claim | finding | source |
|---|---|---|
| **A** | cost Hamiltonian is diagonal; exact ground state is a **uniform mask on 100 %** of snapshots; V1 objective is 64.8/256-fold degenerate | T11 |
| **B** | every solver — exhaustive, greedy, SA, QAOA p1–p3, classical alone — reaches that optimum. Only *cold* SA struggles: the one not warm-started from the classical answer | T11 |
| **C** | the circuit realises **0–8.5 %** of the displacement toward its own optimum, **decreasing with depth**, negative by p = 4 | T11b |
| **D** | ablating **all** ZZ and **all** ZZZZ couplings changes **0.0000 decisions**; removing the Z bias destroys the decision entirely | T13, both mappers |
| **D-bis** | *why*: a Gaussian window centred on the AMR threshold discards 88.6–99.99 % of the ZZ coupling, preferentially where it is largest (Spearman −0.37…−0.50) | T17 |
| **D-ter** | **repairing that window does not rescue the couplings.** With the window neutralised (coupling restored to O(25–155)) ablation still changes 0.0000 decisions; and the V2 mapper, which has no window by construction, is equally inert | T18, T13-v2 |
| **E** | closed loop: **Q-HAS is Pareto-dominated on 4/4 folds at equal budget** | T15/T15b |
| **F** | solver converges at **order 1**, not 4 (RK4 → projection is Lie splitting) | T14 |
| **G** | the mask asymmetry is degeneracy, not broken equivariance | T12 |

**D-ter is the load-bearing one.** It forecloses the natural rebuttal
("your implementation is buggy, fix it and the Ising formulation works"):
the inertness survives a full repair of the defect it uncovered, and is
confirmed independently by a mapper that never had the defect.

### Claim E in detail — 18 completed runs, 4 held-out classes

Q-HAS repeated **5×** per fold (identical inputs), of which 18 draws
completed — `rotor` lost 2 to divergence; classical arm verified
deterministic across 8 replays (range exactly 0.00e+00). Comparison is
against the **budget-matched** classical point, whose completion the T19
trace audit verified.

| fold | n | Q-HAS mean ± sd | matched classical | ratio (mean) | gap/sd |
|---|---|---|---|---|---|
| `ot` | 5 | 0.10727 ± 0.01823 | **0.08270** | 1.30× | 1.35 |
| `kh` | 5 | 0.00320 ± 0.00203 | **0.00168** | 1.90× | 0.75 |
| `rotor` † | 3 | 0.14725 ± 0.04062 | **0.05365** | 2.74× | 2.30 |
| `tearing` | 5 | 0.00801 ± 0.00193 | **0.00443** | 1.81× | 1.86 |

This table previously read 1.56×, 1.93×, 2.86×, 2.05×. Those came from the
**pre-guard** pass, whose draws were not checked for divergence; the guarded
re-run superseded them and `rotor` dropped to 3 valid draws. `t16` checks
every cell above against `t20_qhas_run_variance_<fold>.json`.

**The headline is a count, not a ratio:**

> Over 18 completed closed-loop runs, Q-HAS is less faithful than the
> budget-matched classical rule on **18/18**, more expensive on **16/18**,
> and strictly Pareto-dominated on **16/18**. No run reverses the ordering
> on both coordinates at once.

Quote it this way. The per-fold ratios (1.30–2.74×) are means of a quantity
with 17–49 % CV, and **gap/sd is below 2 on three folds of four** — a single
run per arm cannot support a magnitude claim. The ratios first published
(2.57–4.41×) each rested on one draw and were inflated 1.1–2.2×.

† `rotor`'s *tuned* classical arm diverged, so its primary comparison is
void; its budget-matched point was separately verified complete and
reproduces exactly, so this row stands.

---

## 4. Defect register (D1–D15)

Fifteen claims that code is wrong — D1–D13 about existing code, D14–D15
about my own verification code. Full detail and per-defect
verification commands in `docs/v4_final_results_for_paper.md` §3 and
`docs/CODE_REVIEW_GUIDE.md` §3.

| # | defect | where |
|---|---|---|
| D1 | KH physics-seed amplitude overwhelms the intended mode | V3 Task 8 |
| D2 | phase-1b KH observable averages on the wrong axis — the check could **never** detect KH growth | `phase1b_dns_validation` |
| D3 | `SCENARIOS_ALL` lists `ot`/`rotor` twice → 2:1 loss weighting; for LOSO would manufacture leakage | `TrainHyperParam_v2` |
| D4 | QAOA threshold hard-coded at 0.1496 while the classical arm tunes freely → arms compared at different operating points | `make_composite_objective` |
| D5 | RK4-then-projection = first-order Lie splitting; scheme is order 1 | `solver.py::step_full` |
| D6 | V1 suite fails 8 tests on a clean checkout (6 signature drift, **2 substantive**) | `tests/test_v9_metrics.py` etc. |
| D7 | the uncertainty window annihilates the ZZ family it is meant to focus | `HamiltParams.compute_coefficients` |
| D8 | ZZ reaches the decision only via `C_scale`, a normalisation side-channel — never as coupling | same |
| D9 | *(mine, fixed)* `t13` wrote one filename for both mappers, silently overwriting the comparison | `study/v4/t13_term_ablation.py` |
| D10 | `compare_rotor_budget.py` raises `TypeError`; **as committed it has never been runnable** | `src/compare_rotor_budget.py:110` |
| **D11** | **the Q-HAS arm is not deterministic** — no RNG seed anywhere in V1's VQA chain | `src/VQA/execute.py`, `runtime.py` |
| **D12** | aborted runs return a **partial score with keys identical** to a completed run | `src/pipeline.py:499` |
| **D13** | **train/test leak**: the QAOA arm's threshold `0.1496` was fitted on *all four* classes, including the held-out one, while the classical arm re-tunes per fold | `TrainHyperParam_v2:632`, `t15:154` |
| **D14** | *(mine, found late)* T20's classical **determinism control** ran at the *tuned* threshold on `ot` and `kh` while the artifact field reads `"budget-matched classical"`. The field describes the reference *value* (correctly read from T15b), not the replayed control. Recomputing from `classical_stats` gives phys **0.4845** on `ot` against the matched **0.0827** — enough to flip the fold's direction | `t20:122`, `t20:298` |
| **D15** | *(mine, found late)* `git_commit_hash()` is called at **save** time, so a 1-hour run is stamped with code committed *after* it started. This is how the `ot`/`kh` T20 artifacts carry a hash that postdates the `always_matched=True` fix while having executed the pre-fix code. The provenance stamp CLAUDE.md requires is therefore not sufficient on its own for long runs | every `study/v4/t*.py` |

**D13 is the one to disclose first.** The pre-registration claims the
held-out class is excluded from **all** tuning of both arms; that is false
for the QAOA arm, whose decision threshold comes from
`_run_classical_phase1` fitted on "KH + OT + Tearing + Rotor". The leak is
**asymmetric and favours Q-HAS**, which still loses 18/18 — so the
conclusion is conservative, but the protocol statement must be corrected.
**Removing the leak has since been measured** on 2 folds of 4: Q-HAS gets
*worse*, not better (2.1× the classical frontier at its own budget on
`tearing`, total divergence on `rotor`), and the one transfer result that
had favoured it reverses.

**D11 and D12 most affect the conclusions.** D11: replaying fold `ot` with
identical inputs gave phys **0.1345 vs 0.1940** stored (44 % swing) while
the classical arm reproduced bit-exactly — so every Level-3 Q-HAS number,
**and V1's own published closed-loop numbers**, is one unreplicated draw.
D12: fold `rotor`'s classical arm aborted at step 208 and would have scored
as a large Q-HAS win.

Two of D6's failures are substantive — they assert ZZ coupling survives on
Orszag–Tang and have been *failing*. Taking them seriously is what produced
D7/D8 and the D-bis/D-ter mechanism.

---

## 5. Robustness of Claim E against D11 and D12

**D11 quantified — and the magnitudes do not survive.** T20 was re-run with
the abort marker captured at execution time (the first pass had no guard and,
the arm being non-deterministic, could never be audited afterwards).

| fold | valid | CV | gap/sd | ratio vs matched classical |
|---|---|---|---|---|
| `ot` | 5/5 | 17.0 % | 1.35 | **1.30×** |
| `kh` | 5/5 | **63.6 %** | 0.75 | **1.90×** |
| `rotor` | **3/5** | 27.6 % | **2.30** | **2.74×** |
| `tearing` | 5/5 | 24.1 % | 1.86 | **1.81×** |

The published ratios have shrunk twice: 2.57/4.41/3.62/4.38× (single draw) →
1.56/1.93/2.86/2.05× (unguarded means) → **1.30/1.90/2.74/1.81×** (verified).

**Only 1 fold of 4 reaches gap/sd ≥ 2 — and it is not the same fold as in
the unguarded pass** (`ot` fell from 2.09 to 1.35, `rotor` rose from 1.56 to
2.30). At n = 5 the separability verdict is itself unstable. **Do not quote
per-fold magnitudes.** Quote the direction (verified mean above the matched
classical on 4/4) and the dominance count (18/20 on unseen conditions).

**A failure mode no metric captures.** `rotor`'s Q-HAS arm aborted on 2 of 5
draws (40 %) while its classical control at the same budget completed both
times deterministically. `phys_score`, `patch_ratio`, the dominance count and
the λ analysis all presuppose a run that finishes. The quantum rule
destabilises the solver at a rate the classical rule does not — a distinct
result deserving its own line.

**D12 handled.** Divergence audit (T19), replaying every arm and checking it
reproduces its stored value:

- `ot`, `kh`, `tearing` — both arms completed → **usable**
- `rotor` — classical arm aborted at step 208, **deterministically** (its
  control reproduces 1.1731 exactly) → **not usable for the primary
  endpoint**; its budget-matched point was separately verified complete and
  reproduces exactly, so its Pareto row stands

Bisection traces audited too: `rotor` 2/6 points came from aborted runs
(excluded from the plotted frontier), `tearing` 0/6 — its phys = 4.13 point
**completed** and is a genuine operating point, so a "phys > 1 ⇒ diverged"
heuristic would have deleted valid data.

---

## 6. Unfinished — do not let this be mistaken for done

| item | state | consequence |
|---|---|---|
| **T22 unseen initial conditions** | **done** (T22b: 5 draws x 2 conditions x 4 folds, 56 runs, 0 aborted) | the single-run signal (Q-HAS relatively better on unseen conditions, all 4 folds) sits inside D11's 17–49 % CV. The repeated pass (5 draws × 2 conditions, budget-matched reference) is what can settle it |
| **D13 removal** | **partially done**: `--mode leak-free` measured on `rotor` and `tearing` (`ot`, `kh` running). Leak-free, Q-HAS is 2.1× worse than the classical frontier at its own budget on `tearing` and aborts on 5/5 canonical draws on `rotor` — the leak was propping it up | the *definitive* version still requires re-tuning the QAOA arm with `threshold_amr` in its Optuna search space on the training classes. `t24_leak_free_summary.py` reports what is measured |
| physics seeds | still **1 per class** | T20's 18 completed runs vary QAOA sampling only; T22 varies the initial condition but at n=1 per condition so far |
| **T19 audit of `tearing`** | **done** | the only fold whose arms are unverified |
| **T19 `--trace-only`** | **done**, figure re-rendered | `rotor`'s 2 aborted points are excluded from the plotted frontier, and the Q-HAS marker is now the mean of the repeated draws with its spread, not a single run |
| ≥ 3 physics seeds | not attempted | protocol wanted ≥3; everything is n = 1 per class |
| 170 Optuna trials | not attempted (4 used) | Q-HAS deliberately under-tuned; declared, but "Q-HAS loses" is partly "Q-HAS was barely tuned" |
| Hardware / noise | not attempted | audit judged it premature without a positive L3 |

---

## 7. Weakest points, stated plainly

- **n = 1 physics seed** everywhere at Level 3.
- **4 Optuna trials instead of 170** — the budget-matched control mitigates
  the under-tuning objection but does not remove it.
- **`rotor`'s primary comparison is void**, not merely noisy. A stricter
  reviewer would drop the fold entirely rather than fall back to its
  budget-matched comparison.
- **The interpretive step in §3** — rejecting the primary endpoint as
  D4-contaminated — carries the whole reading and is a judgement call.
- **The 4-panel figure** plotted diverged points as attainable operating
  points, and its Q-HAS marker was a single draw annotated with the
  retracted ratios. Both are fixed and the figure re-rendered; if you pull
  an older copy of the PDF, check it against `pareto_panel.csv`.

---

## 8. Map of what was added

```
docs/
  CODE_REVIEW_GUIDE.md         <- start here for review
  v4_final_results_for_paper.md<- the full argument, claims + defects
  level3_preregistration.md    <- decision rules, frozen before L3 ran
  protocol_deviations.md
study/v4/
  t11..t14                     open-loop attribution, ablation, numerics
  t15  t15b  t15c              Level 3, budget-matched, cross-fold synthesis
  t16                          self-checking master table
  t17  t18                     window mechanism + counterfactual
  t19                          divergence audit (arms and bisection trace)
  t20                          Q-HAS run-to-run variance
  make_pareto_figure.py        single-fold figure
  make_pareto_panel.py         4-fold panel
  level3_status.py  run_fold.sh
  RESULTS_V4.md                every number with its command
tests/v4/                      70 tests
figures_v4/                    pdf + png + csv per figure
```

Every script accepts `--seed` and writes its git hash and full CLI args into
its output. The one determinism guarantee that **cannot** be honoured is the
Q-HAS arm's (D11) — the randomness is inside V1's unseeded Aer backend.
