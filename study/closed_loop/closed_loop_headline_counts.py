#!/usr/bin/env python3
"""Aggregate the confirmatory trajectory-level closed-loop results."""

import argparse
import json
import os
import sys

import numpy as np


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _path in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", name) for name in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import provenance
from closed_loop_campaign import (
    _atomic_json, _load_v1_training_module, fold_scenarios,
)
from stats import bootstrap_by_trajectory
from stats_confirmatory import holm_correction, hierarchical_bootstrap


def _sign_flip_pvalue(deltas):
    """Exact two-sided randomisation p-value for paired trajectory deltas."""
    deltas = np.asarray(deltas, dtype=float)
    if not 1 <= deltas.size <= 20:
        raise ValueError("exact sign-flip test requires 1 to 20 deltas")
    observed = abs(float(np.mean(deltas)))
    masks = np.arange(2 ** deltas.size, dtype=np.uint64)[:, None]
    bits = (masks >> np.arange(deltas.size, dtype=np.uint64)) & 1
    signs = 2.0 * bits.astype(float) - 1.0
    permuted = np.abs(np.mean(signs * deltas[None, :], axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def fold_counts(results_dir, fold, prefix="t20_qhas_run_variance",
                n_boot=1000, seed=0):
    """Return trajectory-level inference for one schema-2 artifact."""
    path = os.path.join(results_dir, f"{prefix}_{fold}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as stream:
        artifact = json.load(stream)
    if artifact.get("schema") != 2:
        raise RuntimeError(
            f"{path} uses an obsolete schema without per-run budget matches")
    if artifact.get("replication_unit") != "trajectory":
        raise RuntimeError(
            f"{path} does not use physics trajectories as replicates")

    runs = artifact.get("qhas_runs", [])
    physics_seeds = [run.get("physics_seed") for run in runs]
    if any(value is None for value in physics_seeds):
        raise RuntimeError(f"{path} omits a physics seed")
    if len(physics_seeds) != len(set(physics_seeds)):
        raise RuntimeError(f"{path} repeats a physics seed")
    qaoa_seeds = {run.get("qaoa_seed") for run in runs}
    if None in qaoa_seeds or len(qaoa_seeds) != 1:
        raise RuntimeError(f"{path} does not hold the QAOA seed fixed")
    completed = [run for run in runs if run.get("completed")]
    paired = [run for run in completed
              if run.get("budget_match", {}).get("converged")]
    deltas = np.asarray(
        [run["budget_match"]["delta_phys"] for run in paired], dtype=float)
    if deltas.size and not np.all(np.isfinite(deltas)):
        raise RuntimeError(f"{path} contains a non-finite paired delta")

    inference = None
    if deltas.size:
        trajectory_ids = np.asarray(
            [f"{fold}:{run['physics_seed']}" for run in paired])
        boot = bootstrap_by_trajectory(
            deltas, trajectory_ids, B=n_boot, seed=seed)
        inference = {
            "mean_delta_phys": boot["estimate"],
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "n_trajectories": boot["n_traj"],
            "sign_flip_p": _sign_flip_pvalue(deltas),
            "classical_confirmed": bool(boot["ci_low"] > 0.0),
            "qhas_confirmed": bool(boot["ci_high"] < 0.0),
        }

    return {
        "fold": fold,
        "scenario": artifact.get("scenario"),
        "Re": artifact.get("Re"),
        "status": artifact.get("status"),
        "n_runs": len(runs),
        "n_completed": len(completed),
        "n_aborted": len(runs) - len(completed),
        "n_paired": len(paired),
        "n_unmatched": len(completed) - len(paired),
        "qhas_lower_error": int(np.sum(deltas < 0)),
        "classical_lower_error": int(np.sum(deltas > 0)),
        "ties": int(np.sum(deltas == 0)),
        "mean_delta_phys": float(np.mean(deltas)) if deltas.size else None,
        "deltas_phys": deltas.tolist(),
        "physics_seeds": physics_seeds,
        "qaoa_seed": next(iter(qaoa_seeds)),
        "inference": inference,
        "parent_campaign_contract_sha256": artifact.get(
            "parent_campaign_contract_sha256"),
    }


def totals(rows):
    keys = (
        "n_runs", "n_completed", "n_aborted", "n_paired", "n_unmatched",
        "qhas_lower_error", "classical_lower_error", "ties",
    )
    out = {key: sum(row[key] for row in rows) for key in keys}
    weighted = [
        (row["mean_delta_phys"], row["n_paired"]) for row in rows
        if row["mean_delta_phys"] is not None and row["n_paired"]
    ]
    out["mean_delta_phys"] = (
        sum(value * count for value, count in weighted)
        / sum(count for _, count in weighted)
        if weighted else None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--prefix", default="t20_qhas_run_variance")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--allow-protocol-deviation", action="store_true")
    args = parser.parse_args()

    from config import RESULTS_DIR

    prov = provenance.start()
    if args.folds is None:
        training = _load_v1_training_module()
        args.folds = [key for key, _ in fold_scenarios(training)]

    rows = []
    missing = []
    for fold in args.folds:
        row = fold_counts(
            RESULTS_DIR, fold, args.prefix, args.bootstrap, args.seed)
        if row is None:
            missing.append(fold)
        else:
            rows.append(row)
    if missing and not args.allow_missing:
        raise RuntimeError(f"missing sensitivity artifacts: {', '.join(missing)}")
    if not rows:
        raise RuntimeError("no sensitivity artifact to aggregate")
    incomplete = [row["fold"] for row in rows
                  if row["status"] != "complete"
                  or row["n_paired"] != row["n_runs"]]
    if incomplete:
        raise RuntimeError(
            f"incomplete matched comparisons: {', '.join(incomplete)}")
    deviations = []
    if len(rows) != 8:
        deviations.append(f"{len(rows)} folds instead of 8")
    wrong_replicates = [row["fold"] for row in rows
                        if row["n_runs"] != 3 or row["n_paired"] != 3]
    if wrong_replicates:
        deviations.append(
            "not exactly 3 paired trajectories: " + ", ".join(wrong_replicates))
    if deviations and not args.allow_protocol_deviation:
        raise RuntimeError("protocol deviation: " + "; ".join(deviations))

    inferential_rows = [row for row in rows if row["inference"] is not None]
    adjusted = holm_correction(
        [row["inference"]["sign_flip_p"] for row in inferential_rows])
    for index, row in enumerate(inferential_rows):
        row["inference"]["holm_p"] = float(adjusted["p_adjusted"][index])
        row["inference"]["holm_reject"] = bool(adjusted["reject"][index])

    all_deltas = []
    class_ids = []
    regime_ids = []
    for row in rows:
        all_deltas.extend(row["deltas_phys"])
        class_ids.extend([row["fold"]] * len(row["deltas_phys"]))
        regime_ids.extend([row.get("Re") or 400] * len(row["deltas_phys"]))
    overall = None
    if all_deltas:
        result = hierarchical_bootstrap(
            all_deltas, class_ids, regime_ids,
            B=args.bootstrap, seed=args.seed)
        overall = {key: value for key, value in result.items()
                   if key != "boot"}
        overall["frac_classical_lower_error"] = float(
            np.mean(np.asarray(all_deltas) > 0.0))
        overall["n_classical_confirmed_folds"] = sum(
            row["inference"]["classical_confirmed"]
            for row in inferential_rows)
        overall["n_qhas_confirmed_folds"] = sum(
            row["inference"]["qhas_confirmed"]
            for row in inferential_rows)
        overall["classical_falsification_rule_met"] = bool(
            len(rows) == 8
            and overall["n_classical_confirmed_folds"] >= 6)

    total = totals(rows)
    print("| fold | paired | Q-HAS lower | classical lower | mean delta | 95% CI | Holm p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        inf = row["inference"]
        print(f"| {row['fold']} | {row['n_paired']} | "
              f"{row['qhas_lower_error']} | {row['classical_lower_error']} | "
              f"{row['mean_delta_phys']:+.6f} | "
              f"[{inf['ci_low']:+.6f}, {inf['ci_high']:+.6f}] | "
              f"{inf['holm_p']:.4f} |")

    payload = {
        "artifact": "closed_loop_headline_counts",
        "schema": 2,
        "folds": rows,
        "total": total,
        "overall_hierarchical": overall,
        "protocol_deviations": deviations,
        "missing": missing,
        "cli_args": vars(args),
    }
    payload.update(provenance.finish(prov))
    output = os.path.join(RESULTS_DIR, "t23_headline_counts.json")
    _atomic_json(output, payload)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
