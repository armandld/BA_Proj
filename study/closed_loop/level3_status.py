#!/usr/bin/env python3
"""
V4 - Etat d'avancement d'une campagne Level-3 (lecture seule).

Interroge le stockage Optuna persistant de chaque fold en mode READ-ONLY
(URI `?mode=ro`) : aucune ecriture, aucun verrou pris sur la base pendant
que le driver tourne. Affiche une ligne par fold : essais termines par
etude, presence des sorties t15 / t15b.

Usage :
  python study/v4/level3_status.py
  python study/v4/level3_status.py --folds kh rotor tearing --oneline
"""
import argparse, os, sqlite3, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------

ALL_FOLDS = ("ot", "kh", "rotor", "tearing")


def optuna_progress(db_path):
    """Retourne {study_name: (n_complete, n_running, best_value)} ou {}.

    Ouvre la base en lecture seule : sans droit d'ecriture, une lecture
    concurrente d'un writer SQLite ne peut ni bloquer ni corrompre.
    """
    if not os.path.exists(db_path):
        return {}
    uri = "file:" + os.path.abspath(db_path) + "?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=2.0)
    except sqlite3.Error:
        return {}
    out = {}
    try:
        cur = con.execute(
            "SELECT s.study_name, t.state, COUNT(*) "
            "FROM trials t JOIN studies s ON s.study_id = t.study_id "
            "GROUP BY s.study_name, t.state")
        for name, state, n in cur.fetchall():
            d = out.setdefault(name, {"COMPLETE": 0, "RUNNING": 0,
                                      "FAIL": 0, "best": None})
            d[state] = d.get(state, 0) + n
        cur = con.execute(
            "SELECT s.study_name, MIN(v.value) "
            "FROM trial_values v JOIN trials t ON t.trial_id = v.trial_id "
            "JOIN studies s ON s.study_id = t.study_id "
            "WHERE t.state = 'COMPLETE' GROUP BY s.study_name")
        for name, best in cur.fetchall():
            if name in out:
                out[name]["best"] = best
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return out


def fold_status(results_dir, fold, prefix="t15_level3"):
    db = os.path.join(results_dir, f"{prefix}_optuna_{fold}.db")
    return {
        "fold": fold,
        "optuna": optuna_progress(db),
        "tuning_json": os.path.exists(
            os.path.join(results_dir, f"{prefix}_tuning_{fold}.json")),
        "t15": os.path.exists(
            os.path.join(results_dir, f"{prefix}_fold_{fold}.json")),
        "t15b": os.path.exists(
            os.path.join(results_dir, f"t15b_budget_matched_{fold}.json")),
    }


def format_status(st, oneline=False):
    parts = []
    for name, d in sorted(st["optuna"].items()):
        b = "" if d["best"] is None else f"@{d['best']:.4f}"
        run = f"+{d['RUNNING']}r" if d.get("RUNNING") else ""
        parts.append(f"{name}={d['COMPLETE']}{run}{b}")
    trials = " ".join(parts) if parts else "no-trials-yet"
    flags = "".join([
        "T" if st["tuning_json"] else "-",
        "1" if st["t15"] else "-",
        "B" if st["t15b"] else "-",
    ])
    if oneline:
        return f"[{st['fold']}] {flags} {trials}"
    return f"  {st['fold']:>8}  [{flags}]  {trials}"


def main():
    p = argparse.ArgumentParser(description="Level-3 campaign status")
    from config import RESULTS_DIR
    p.add_argument("--folds", nargs="+", default=list(ALL_FOLDS))
    p.add_argument("--prefix", default="t15_level3")
    p.add_argument("--oneline", action="store_true")
    args = p.parse_args()

    if not args.oneline:
        print("  flags: T=tuning checkpoint  1=t15 fold json  B=t15b budget")
    for f in args.folds:
        print(format_status(fold_status(RESULTS_DIR, f, args.prefix),
                            args.oneline))


if __name__ == "__main__":
    main()
