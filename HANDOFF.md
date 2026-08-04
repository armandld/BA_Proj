# Q-HAS V4 — hand-off for merge

**Branch:** `claude/kind-babbage-927g10` · **Base:** `origin/main`
**Scope:** 74 commits, 77 files, ~15 700 added lines.
**Read `docs/CODE_REVIEW_GUIDE.md` first** — it orders the review by risk
per minute. This file is the single entry point: what was done, what it
concluded, what is unfinished, and what to check before merging.

---

## 1. Merge checklist

```bash
BASE=$(git merge-base HEAD origin/main)
git diff --name-only $BASE HEAD -- src/                            # MUST be empty (V1 read-only)
git diff --name-only $BASE HEAD -- study/ | grep -v '^study/v[34]/'  # MUST be empty (V2 read-only)
python -m pytest tests/v3 tests/v4 -q                              # 196 passed
python study/v4/t16_aggregate_v4.py                                # 71 rows, 0 DIFF, 0 MISSING
```

All four held: `src/` 0 files changed, V2 phases 0 files changed,
**196 pytests passed**, master table **71/71 OK, 0 DIFF, 0 MISSING**
(all four Level-3 folds are now present, so nothing is outstanding in it). Nothing outside `study/v3/`, `study/v4/`,
`tests/v3/`, `tests/v4/`, `docs/`, `figures_v4/`, `logs/v4/`, `CLAUDE.md`
was modified.

**`bash run_tests.sh` fails 8 tests — do not treat this as a regression.**
It is defect D6, pre-existing, reproduced at `cf93ba3` (the last commit
touching `src/`/`tests/`, before any of this work).

---

## 2. Before the container dies: rescue the artifacts

`study/results/` is **gitignored** — 65 artifacts, ~100 MB. Every number in
these documents lives **only on this container**, which has already been
reclaimed seven times. The repo has the code that regenerates them and the
values transcribed into Markdown, but a fresh clone can verify nothing.

`study/v4/t16_aggregate_v4.py` is the only transcription cross-check: it
recomputes each published number from its artifact and diffs against the
Markdown. It printed `0 DIFF`.

Cheap to regenerate (seconds–minutes): T11, T11b, T12, T13, T14, T17, T18.
Expensive (hours): the Level-3 campaign.

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

### Claim E in detail

| fold | Q-HAS (patch, phys) | budget-matched classical | ratio | dominated |
|---|---|---|---|---|
| `ot` | 0.6797, 0.1940 | 0.6412, **0.0827** | 2.57× | yes |
| `kh` | 0.8376, 0.0070 | 0.7943, **0.0017** | 4.41× | yes |
| `rotor` | 0.3761, 0.1678 | 0.3562, **0.0536** | 3.62× | yes |
| `tearing` | 0.7692, 0.0185 | 0.6250, **0.0044** | 4.38× | yes |

In every case the classical arm is **both more faithful and cheaper**.

The pre-registered *primary* endpoint splits **2–2** and establishes
nothing under its own ≥3/4 rule. I argue it is contaminated by D4 (the two
arms are tuned at different operating points). **That argument is the main
interpretive step in the analysis and is a judgement, not a measurement —
decide it yourself.**

---

## 4. Defect register (D1–D12)

Twelve claims that existing code is wrong. Full detail and per-defect
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

| fold | Q-HAS phys, observed draws | matched classical | worst-case ratio |
|---|---|---|---|
| `ot` | 0.1345 – 0.1940 | 0.0827 | 1.6× |
| `kh` | 0.0048 – 0.0070 | 0.0017 | 2.8× |
| `rotor` | 0.1678 (1 draw) | 0.0536 | 3.1× |

Even the **worst** Q-HAS draw stays above the classical arm on every fold,
so the *direction* survives. The *magnitudes* are ranges, not points, and
are written that way throughout.

Divergence audit (T19), replaying every arm and checking it reproduces the
stored value:

- `ot` — both arms completed → **usable**
- `kh` — both arms completed → **usable**
- `rotor` — classical aborted at step 208 → **not usable for the primary
  endpoint**; its budget-matched point was separately verified clean and
  reproduced exactly, so its Pareto comparison stands
- `tearing` — audit running at time of writing

---

## 6. Unfinished — do not let this be mistaken for done

| item | state | consequence |
|---|---|---|
| **T20 Q-HAS run variance** | implemented, **not yet run** | variance characterised from 2 draws on 2 folds, not a distribution. `python study/v4/t20_qhas_run_variance.py --fold kh --repeats 5` |
| **T19 audit of `tearing`** | running | the only fold whose arms are unverified |
| **T19 `--trace-only`** | running | diverged points still plotted on the "attainable frontier" in `figures_v4/pareto_panel.*`. **Re-render before using the figure in the paper** |
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
- **The 4-panel figure currently plots diverged points** as if they were
  attainable operating points (fix running).

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
