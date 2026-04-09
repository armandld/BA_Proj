#!/usr/bin/env python3
"""
Q-HAS Training Monitor — Check progress across all distributed workers.

Usage:
    python monitor.py                                    # uses OPTUNA_STORAGE env var
    python monitor.py "postgresql://user:pass@host/db"   # explicit URL
"""
import sys
import os
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

STUDY_NAMES = ["q_has_phase1", "q_has_phase2", "q_has_phase3"]
TARGET_TRIALS = {"q_has_phase1": 200, "q_has_phase2": 400, "q_has_phase3": 600}


def monitor(storage_url):
    print(f"\nStorage: {storage_url.split('@')[-1] if '@' in storage_url else storage_url}")
    print("=" * 70)

    for name in STUDY_NAMES:
        try:
            study = optuna.load_study(study_name=name, storage=storage_url)
        except KeyError:
            print(f"  {name:20s}  — not started")
            continue

        states = {}
        for t in study.trials:
            s = t.state.name
            states[s] = states.get(s, 0) + 1

        done = states.get("COMPLETE", 0)
        running = states.get("RUNNING", 0)
        failed = states.get("FAIL", 0)
        pruned = states.get("PRUNED", 0)
        waiting = states.get("WAITING", 0)
        target = TARGET_TRIALS.get(name, "?")
        pct = done / target * 100 if isinstance(target, int) and target > 0 else 0

        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(f"\n  {name}")
        print(f"    [{bar}] {pct:5.1f}%")
        print(f"    Done: {done}/{target}  |  Running: {running}  |  "
              f"Pruned: {pruned}  |  Failed: {failed}  |  Waiting: {waiting}")

        if done > 0:
            print(f"    Best score: {study.best_value:.6f}")
            print(f"    Best params: {study.best_params}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = os.environ.get("OPTUNA_STORAGE")

    if not url:
        # Fallback: try local SQLite
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_dir = os.path.join(project_root, "Train_results")
        if os.path.isdir(db_dir):
            for name in STUDY_NAMES:
                db_path = os.path.join(db_dir, f"{name}.db")
                if os.path.exists(db_path):
                    url = f"sqlite:///{db_path}"
                    break

    if not url:
        print("Usage: python monitor.py <storage_url>")
        print("   or: export OPTUNA_STORAGE=postgresql://...")
        sys.exit(1)

    monitor(url)
