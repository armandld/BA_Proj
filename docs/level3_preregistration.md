# Level 3 (closed loop) — pre-registration

**Frozen before any Level-3 computation.** Written after Tasks 11, 11b, 13
and 14 (which are open-loop / structural) and before the first closed-loop
fold is run. Any deviation from this file must be logged in
`docs/protocol_deviations.md`.

Protocol reference: `docs/protocol_v3_evaluation.md` §4. Driver:
`study/v4/t15_level3_closed_loop.py`.

---

## 1. Question

Does the full AMR loop — with time-to-live memory, warm start, and feedback
through the evolved state — compensate or amplify the open-loop failure,
when an instability class is excluded from **all** tuning?

## 2. Design (unchanged from §4, with declared reductions)

For each held-out class: QAOA hyperparameters tuned by Optuna on the
composite loss of the **other** classes only; the classical arm's AMR
threshold tuned on the **same** training classes; both arms then run on the
held-out class with identical DNS trace, hot start, hybrid budget and depth.

**Declared deviations (all forced by cost or by the V1 codebase):**

| item | protocol | this run | reason |
|---|---|---|---|
| folds | 8 classes | **4** | `TrainHyperParam_v2` exposes 4 distinct classes (see the duplication defect in `study/v4/RESULTS.md`) |
| Optuna budget | 170 trials | **4** (classical 2) | measured cost ≈ 55 min per trial across the 4 folds; 170 would take ≈ 7 days |
| physics seeds | ≥ 3 | **1** | the pipeline initialises each scenario deterministically |

This run is therefore a **pilot of the decisive experiment**, not the
definitive campaign. It is powered to detect a large effect, not a subtle
one. No claim of equivalence will be made from a non-significant difference
alone; only the TOST procedure below can support parity.

## 3. Prediction (the point of pre-registering)

Tasks 11b and 13 establish, at the deployed size (8 qubits, `VQA_N = 2`):
the exact ground state of the cost Hamiltonian is a **uniform** mask on 100%
of snapshots; the ZZ and ZZZZ families are **causally inert** for both the
V2 and the deployed V1 mapper (0.0000 decisions changed); and the circuit
realises only 0–8.5% of the displacement toward its own optimum, decreasing
with depth.

**We therefore predict, before running:**

> **P1.** The Q-HAS and classical arms will produce closed-loop endpoints
> that are equal within the equivalence margin on a majority of folds:
> |Δ combined| small, with no consistent sign across folds.
>
> **P2.** Where the arms do differ, the difference will come from the
> amplitude encoding and the TTL/feedback dynamics, not from the coupling
> terms — i.e. differences will not correlate with coupling magnitude.
>
> **P3.** Any Q-HAS fidelity gain will be paid for in `patch_ratio`
> (compute), consistent with the V1 closed-loop result already reported
> (tearing gain at +28.8% refined pixels).

## 4. Decision rules (pre-specified)

- **Primary endpoint:** `combined` = (phys + λ·patch)/(1+λ), paired per fold.
  Secondary: `phys_score` (**instability-weighted** relative L2 vs DNS —
  `pipeline.score()` weights each field's error by
  `w = 1 + 0.25(|Jz|/<|Jz|> + |omega|/<|omega|>)` built from the reference
  fields, so it is not a plain L2) and `patch_ratio` (compute),
  reported jointly — a fidelity gain bought with compute is not a gain.
- **Confirmation of P1:** TOST equivalence at a margin fixed here as **5% of
  the mean classical `combined`**, computed before seeing the deltas. TOST
  p ≤ 0.05 ⇒ parity established. A non-significant difference test alone
  does **not** establish parity.
- **Refutation of P1:** Q-HAS better on ≥ 3/4 folds with a paired
  Holm-adjusted p ≤ 0.05 ⇒ the closed loop rescues a selector that fails
  open loop; this becomes the central result and the open-loop conclusions
  are scoped accordingly.
- **If the classical arm wins on ≥ 3/4 folds** with CI excluding 0 ⇒ the
  falsification is complete and closed-loop.
- **Under-tuning objection:** with 4 trials the Q-HAS arm is deliberately
  under-optimised. Any "Q-HAS ≈ classical" result will be reported with
  that limitation stated in the same sentence, and cannot be upgraded to
  "Q-HAS is falsified in closed loop" without the full-budget campaign.

## 5. Exclusions

Folds that fail to complete (divergence penalty, solver failure) are
reported as failures and excluded from the paired statistics; the count is
stated. No fold is dropped after seeing its result. No post-hoc selection of
the "best" endpoint, horizon, or λ.
