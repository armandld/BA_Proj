# Q-HAS: quantum-assisted adaptive mesh refinement

This repository evaluates whether a local QAOA decision rule can improve
adaptive mesh refinement for two-dimensional magnetohydrodynamics. It contains
the simulation and training code (`src/`), the falsification and evaluation
experiments (`study/`), and their contract tests (`tests/`).

The repository is under scientific review. Existing result artifacts predate
some corrections and must not be treated as final evidence. Campaign readiness
requires a clean full test run, a successful coefficient preflight, a complete
training artifact, and the confirmatory closed-loop study outputs.

## Layout

- `src/`: MHD solver, AMR logic, physical-to-Ising mapping, QAOA execution,
  hyperparameter training and campaign analysis.
- `study/`: dataset construction, exact baselines, hypothesis tests,
  closed-loop comparisons and aggregators.
- `tests/`: executable scientific and software contracts.
- `scripts/`: the single-machine campaign launcher and study launchers.
- `docs/`: protocol, open defects, evaluation rules and campaign instructions.
- `results/`: versioned reference inputs and prior artifacts; new local outputs
  are written under ignored campaign/result directories.

## Installation and verification

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q -m "not slow"
.venv/bin/python src/train_hyperparams.py --print-space
```

Run the slow numerical convergence tests separately before the final campaign:

```bash
.venv/bin/python -m pytest tests -q -m slow
```

## Training campaign

Training runs on one rented multi-core machine. Rehearse the concurrent journal
locally, then launch phase 1 from a clean committed tree:

```bash
bash scripts/repetition_campagne.sh
bash scripts/run_rented_campaign.sh 16 600 0
```

The launcher is resumable and enforces one global Optuna budget across all
workers. See [docs/MODE_EMPLOI_CAMPAGNE.md](docs/MODE_EMPLOI_CAMPAGNE.md) for
the storage contract, outputs and recovery procedure.

## Scientific workflow

The governing protocol is [docs/protocol_v3_evaluation.md](docs/protocol_v3_evaluation.md).
Before interpreting any result, also read:

- [docs/PLAN_PREPRINT.md](docs/PLAN_PREPRINT.md): claims and preprint structure;
- [docs/DEFAUTS.md](docs/DEFAUTS.md): unresolved blockers;
- [docs/EVALUATION.md](docs/EVALUATION.md): admissible evidence;
- [docs/RESULTS.md](docs/RESULTS.md): reproducible result ledger;
- [docs/CODE_REVIEW.md](docs/CODE_REVIEW.md): review status.

Archived documents and legacy artifacts are historical context, not current
protocol outputs.

After training, the canonical data and confirmatory commands are:

```bash
bash scripts/run_dns_campaign.sh 8 256
QHAS_HYPERPARAMS_PATH=results/hyperparams/reoptimisation/candidate_phase1.json \
  bash scripts/run_study_v3.sh --all
bash scripts/run_confirmatory_campaign.sh 4 170 85
```
