# How to review this branch efficiently

199 commits, 186 files, ~23 000 added lines (76 of those files are the
tracked artifacts under `study/results/`). Reading it front to back is the
wrong strategy. This guide orders the review by **risk per minute**: the
checks that would change your conclusions come first, and each one is a
command you can paste.

Budget: **~15 min** for the gate checks, **~90 min** for a serious review,
**a day** for line-by-line.

---

## 0. Before anything: the artifacts are in the repo

`study/results/` **is tracked** — 76 artifacts, 6.3 MB of `.json`/`.npz`.
Every number quoted in the docs can therefore be recomputed from a fresh
clone. This was not always so: the directory was gitignored for most of the
campaign, and the guide said the numbers lived only on the producing
machine. That is no longer true and the change matters for how you review.

The only exclusion is `study/results/dns_*_N256.npz` — 4 files, 94 MB, the
DNS reference snapshots. They are **inputs**, not results: the solver is
deterministic, so `study/v4/t15_level3_closed_loop.py` regenerates them
byte-identically (a few minutes each).

Consequences for your review:

- the numbers in `RESULTS_V4.md` are **diffable**, not merely claims: run
  `python study/v4/t16_aggregate_v4.py`, which recomputes every headline
  number from the artifacts and compares it to the published value. It must
  print `0 DIFF` (119 rows at the time of writing);
- a `MISSING` row means the artifact was never produced, not that it was
  lost — the tracked set is the complete set;
- everything except Level 3 re-runs in seconds to minutes (T11–T14, T17,
  T18); Level 3 and T20/T22 are hours.

---

## 1. Gate checks (~15 min) — if one fails, stop reviewing and ask

### 1a. The guardrails actually held

`CLAUDE.md` forbids touching V1 (`src/`) and the V2 phases. This is one
command and it is the single most important check:

```bash
BASE=$(git merge-base HEAD origin/main)
git diff --name-only $BASE HEAD -- src/                      # MUST be empty
git diff --name-only $BASE HEAD -- study/ \
  | grep -v '^study/v[34]/' | grep -v '^study/results/'      # MUST be empty
```

Both are empty at HEAD. The second command **must** exclude
`study/results/`: those are tracked artifacts, not V2 phase code, and
without the exclusion it returns 76 files and looks like a guardrail
breach. Everything new lives in `study/v3/`, `study/v4/`, `tests/v3/`,
`tests/v4/`, `docs/`, `figures_v4/`, `logs/v4/`.

### 1b. The test gate

```bash
python -m pytest tests/v3 tests/v4 -q      # expect 298 passed, 20 skipped
```

Note `bash run_tests.sh` (the V1 suite) **fails 8 tests on a clean
checkout** — that is defect **D6**, pre-existing, reproduced at `cf93ba3`
before any of this work. Do not read it as a regression; see §3.

### 1c. The self-checking master table

```bash
python study/v4/t16_aggregate_v4.py        # expect: 0 DIFF
```

`DIFF` means a published number no longer reproduces from its artifact.
`MISSING` only means a task has not been run on this machine.

---

## 2. Read these five files, in this order (~45 min)

You will understand 80 % of the work from these:

| # | file | why |
|---|---|---|
| 1 | `docs/v4_final_results_for_paper.md` | the whole argument, claims A–G, defect register D1–D13. **Start here.** |
| 2 | `docs/level3_preregistration.md` | the decision rules, frozen *before* Level 3 ran. Check §4 against what §5 of this guide says I actually did |
| 3 | `study/v4/t15c_fold_synthesis.py` | applies those rules; ~360 lines; where a bent rule would hide |
| 4 | `study/v4/t19_arm_divergence_audit.py` | the catch that changed the conclusions (see §3) |
| 5 | `study/v4/RESULTS_V4.md` | every number with its command and reading |

---

## 3. Review the *accusations* hardest (~20 min)

Thirteen defects (D1–D13) are claims that existing code is wrong. Those are
the most damaging things here if any is mistaken. Each is independently
checkable in a couple of minutes — verify these before trusting anything
downstream of them:

| defect | one-line check |
|---|---|
| **D5** solver is order 1, not 4 | `python study/v4/t14_numerical_validation.py` — RK4 without projection gives 4.00, with it 1.12 |
| **D6** V1 suite fails on clean checkout | `git worktree add /tmp/v1 cf93ba3 && cd /tmp/v1 && python -m pytest tests/test_v9_metrics.py -q` → same 8 failures, *before* my work |
| **D7** window kills the ZZ family | `python study/v4/t17_uncertainty_window.py` — `no window` column O(40–140) vs windowed ~0 |
| **D10** `compare_rotor_budget.py` is dead | `python -c "import sys;sys.path.insert(0,'src');from Simulation.HamiltParams import PhysicalMapper;PhysicalMapper(beta=0.5)"` → TypeError |
| **D11** Q-HAS arm is unseeded | `grep -rn seed src/VQA/` → **no output at all** |
| **D12** aborted runs look complete | read `src/pipeline.py:494-556` — the abort path returns the same keys as the normal path |

**D11 and D12 are the two that most affect the conclusions.** If you
disagree with either, most of the Level-3 reading changes.

---

## 4. Where *my* code is most likely to be wrong

Reviewer time is best spent here. These are the places where a bug would
silently flip a result, ranked:

1. **`t15c_fold_synthesis.py::primary_analysis`** — sign convention.
   `combined` is a **cost**, so `delta = qhas - classical < 0` means Q-HAS
   *better*. Get this backwards and every conclusion inverts.
   Covered by `tests/v4/test_t15c_synthesis.py::test_sign_convention_lower_combined_is_better`.

2. **`t19_arm_divergence_audit.py::parse_abort`** — a **false negative here
   is catastrophic** (a crashed arm silently counted as a win, which is
   exactly the bug it was written to catch). A false positive merely
   discards a good fold. Check the regex against real V1 output, and check
   `test_cfl_warning_alone_is_not_an_abort` — V1 emits CFL warnings without
   aborting, and conflating them would throw away valid folds.

3. **`t17_uncertainty_window.py::PARAM_SETS`** — I got this wrong once.
   There are **two** trained σ (0.023 open-loop, 0.1888 for the Level-3
   fold) and I initially labelled the closed-loop one "deployed", which
   understated the suppression by ~120 orders of magnitude on OT. It is now
   read from `phase5` rather than hard-coded. Verify the label matches the
   configuration each number came from.

4. **`t18_window_counterfactual.py::prepare_both_arms`** — the σ → ∞ trick.
   Check the assertion that the no-window coupling really dominates the
   windowed one; if the substitution silently failed, the counterfactual
   would be vacuous.

5. **`make_pareto_panel.py::draw_panel`** — the annotated ratio must equal
   `q_phys / interp_frontier(front, q_patch)` and nothing else. A figure
   that computes its own number differently from the table is a classic
   way to ship an inconsistency.

---

## 5. Deviations from the pre-registration — check I declared them all

Read `docs/level3_preregistration.md` §2 and §5, then confirm:

- **declared before running:** 4 folds not 8, 4 Optuna trials not 170,
  1 physics seed not ≥3;
- **added after seeing fold `ot`** and labelled post-hoc everywhere:
  the budget-matched comparison (`t15b`). It is the analysis the
  conclusion now rests on, so its post-hoc status matters — it is reported
  as secondary and exploratory, never as confirming the frozen plan;
- **§5 exclusion rule** (failed folds excluded, count stated) was
  pre-registered but **not implemented** until T19. Fold `rotor`'s tuned
  classical arm aborted at step 208 and would have been scored as a large
  Q-HAS win.

If you find a deviation I did *not* declare, that is a real finding.

---

## 6. What the conclusions rest on, and how to stress them

The headline is a **count over 18 completed runs**, not a ratio:

> Q-HAS is less faithful than the budget-matched classical rule on
> **18/18**, more expensive on **16/18**, strictly Pareto-dominated
> on **16/18**.

Verify it with `python study/v4/t23_headline_counts.py`. Earlier drafts
said 19/20, 18/20, 17/20 — that table was written by hand, transposed two
columns on `kh` and counted `rotor`'s 2 aborted draws in the denominator.

An earlier draft of this guide quoted per-fold ratios of 2.6×, 4.4×, 3.6×,
4.4×. **Those are retracted**: each rested on a single draw of a
non-deterministic arm and was inflated 1.1–2.2× relative to the 5-draw
means. Do not quote per-fold magnitudes at all — gap/sd is below 2 on three
folds of four. Three ways to attack the count:

1. **Sampling noise (D11).** Q-HAS is unseeded. This is why the claim is a
   count and not a magnitude. Check `t20_qhas_run_variance.py` for the
   gap/σ ratio per fold — under ~2, a single run per arm proves nothing,
   which is exactly what happened to the retracted ratios above.
2. **Divergence (D12).** Verify the *matched* classical runs completed, not
   just the tuned ones. `rotor`'s matched point was separately confirmed
   clean and reproduced exactly.
3. **The primary endpoint disagrees.** It splits 2–2 and establishes
   nothing under its own ≥3/4 rule. I argue that endpoint is contaminated
   by D4 (arms tuned at different operating points). **You should decide
   whether you accept that argument** — it is the main interpretive step in
   the whole analysis, and it is a judgement call, not a measurement.

---

## 7. Fast structural checks

```bash
# nothing redefines existing functions (CLAUDE.md requirement)
grep -rn "^def \(build_dataset\|extract_features_2d\|make_model\|fit_eval\)" study/v4/

# every script takes --seed and records provenance
grep -Ln "git_commit_hash" study/v4/t*.py

# determinism claim: re-run a cheap task twice, compare
python study/v4/t13_term_ablation.py --N 256 --dim 2 --n-snaps 3 --mapper v1
# then diff against the previous artifact — it reproduced bit-exactly for me
```

---

## 8. Things I would push back on if I were you

Honest list of the weakest points, so you do not have to find them:

- **n = 1 physics seed per fold.** The protocol wanted ≥3. Everything at
  Level 3 is one initial condition per class.
- **4 Optuna trials, not 170.** The Q-HAS arm is deliberately
  under-optimised; this is declared, but it means "Q-HAS loses" is
  partly "Q-HAS was barely tuned". The budget-matched control mitigates
  this but does not remove it.
- **n = 5 draws per fold in T20**, which pins the direction but not the
  magnitude. Only 1 fold of 4 is separable, and not the same fold across
  passes.
- **The `rotor` primary comparison is void**, not merely noisy. I report
  its budget-matched comparison instead and say so, but a stricter reading
  would drop the fold entirely.
- **The trace-audit exclusion** (diverged points on the plotted frontier)
  was implemented after the 4-panel figure was first rendered. It has since
  been re-rendered, and the Q-HAS marker changed from a single draw to the
  mean of the repeated draws with error bars — the annotated ratios moved
  from 2.6/4.4/3.6/4.4× to 1.79/2.10/2.49/1.98×. If you have an older PDF,
  it carries retracted numbers.
