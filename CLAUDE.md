# BA_Proj — working context

Full specification: `docs/protocol_v3_evaluation.md` (pre-registered V3
evaluation protocol). Read §8.1–§8.2 there before any change; this file
reproduces them for convenience.

## Context

This repository (BA_Proj) contains Q-HAS: a hybrid quantum-classical AMR
framework for 2D MHD. `src/` is the V1 production pipeline (MHD solver
FD4/RK4 in `src/Simulation/`, QAOA stack in `src/VQA/`) — frozen, working,
covered by `run_tests.sh`. `study/` is the V2 falsification study: a batch
pipeline (phases 1→13) that generates DNS snapshots, defines a per-patch
"hard to coarsen" label, builds Ising Hamiltonians from physical features,
and bounds the approach with ML surrogates (GBT) under random-split and
leave-one-scenario-out (LOSO) evaluation. The V3 work plan
(docs/protocol_v3_evaluation.md) repairs V2's evaluation layer: continuous
metrics, blocked/LOSO splits, trajectory-level statistics, a predictive
level, and a closed-loop level. Key entry points:

- `study/config.py` (constants)
- `study/phase2_hard_patches.py` (labels)
- `study/phase4_exact_diag.py::build_patch_hamiltonian` (fields→Hamiltonian)
- `study/phase11_upper_bound.py` (features, dataset, models)
- `study/phase11b_loso.py` (LOSO)
- `src/Simulation/HamiltParams_v2.py` (V2 coefficients + `compute_psi_v2`)
- `logs/FINDINGS.md` (published V2 numbers, the regression reference)

## Global guardrails (non-negotiable)

- **V1 is read-only.** Never modify `src/` (solver, grid, refinement,
  PhysToAngle, HamiltParams, VQA) unless a task explicitly lists the file.
  `bash run_tests.sh` must pass unchanged after every task.
- **V2 phases are read-only.** All new code lives in `study/v3/` (a new
  package). When a task "repairs" a phase, copy it into `study/v3/` and
  modify the copy; the original must remain runnable for regression.
- Every task ships: code + a pytest under `tests/v3/` + an entry in
  `study/v3/RESULTS.md` (command, git hash, numbers).
- Determinism: every new script accepts `--seed`, and writes the git commit
  hash and full CLI args into its `.npz` outputs.
- Reuse before rewriting: import `build_dataset`, `extract_features_2d`,
  `make_model`, `fit_eval`, `best_threshold_f1` from
  `study/phase11_upper_bound.py`; import the solver, never re-implement
  numerics.
- French comments are fine; code identifiers and RESULTS.md in English.
- One task per session, in order (§8.3 task list). If a task requires
  anything not specified in the protocol, stop and ask — never improvise a
  metric, label, split or subset.
