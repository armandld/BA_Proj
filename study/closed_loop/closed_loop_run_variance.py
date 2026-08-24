#!/usr/bin/env python3
"""Run the confirmatory closed-loop replicates across physics seeds.

Each replicate has a distinct perturbed initial condition. Q-HAS and its
budget-matched classical comparator share that trajectory exactly. The QAOA
seed remains fixed, so the statistical unit is the trajectory required by the
protocol, not a repeated quantum draw on one trajectory.

Output: ``results/t20_qhas_run_variance_<fold>.json``.
"""
import argparse, json, os, sys, time

import numpy as np

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

import provenance
from closed_loop_budget_matched import bisect_threshold_for_budget
from closed_loop_campaign import (_atomic_json, _load_v1_training_module,
                                  fold_scenarios, run_arm)

METRICS = ("combined", "phys_score", "patch_ratio")


def summarise(runs):
    """Statistiques descriptives par metrique."""
    out = {}
    for m in METRICS:
        v = np.array([r[m] for r in runs], dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            out[m] = None
            continue
        out[m] = {
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "range": float(np.max(v) - np.min(v)),
            "cv": (float(np.std(v, ddof=1) / abs(np.mean(v)))
                   if v.size > 1 and np.mean(v) != 0 else None),
            "values": v.tolist(),
        }
    return out


def main():
    p = argparse.ArgumentParser(
        description="Closed-loop physics-seed replicates at matched budget")
    from config import RESULTS_DIR

    p.add_argument("--fold", default="kh")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--prefix", default="t15_level3")
    p.add_argument("--seed", type=int, default=0,
                   help="first physics seed; repeats use consecutive seeds")
    p.add_argument("--qaoa-seed", type=int, default=0,
                   help="fixed QAOA seed shared by physics replicates")
    p.add_argument("--match-max-iter", type=int, default=8)
    p.add_argument("--match-tol", type=float, default=0.02)
    p.add_argument("--allow-protocol-deviation", action="store_true")
    args = p.parse_args()
    if args.repeats < 2:
        p.error("--repeats must be >= 2")
    prov = provenance.start()
    if args.repeats != 3 and not args.allow_protocol_deviation:
        p.error("the confirmatory protocol requires exactly 3 physics seeds")
    if prov["dirty_at_start"]:
        raise RuntimeError(
            "refusing confirmatory physics-seed runs from a dirty tree")

    path = os.path.join(RESULTS_DIR, f"{args.prefix}_fold_{args.fold}.json")
    if not os.path.exists(path):
        raise SystemExit(f"fold {args.fold} not computed yet ({path})")
    with open(path, encoding="utf-8") as stream:
        rec = json.load(stream)
    if not rec.get("campaign_contract_sha256"):
        raise RuntimeError(
            "fold artifact has no campaign contract; refusing an "
            "unverifiable sensitivity study")

    print("=" * 84)
    print(f"  Closed-loop physics-seed replicates, fold {args.fold}")
    print(f"  physics seeds: {args.seed}..{args.seed + args.repeats - 1}")
    print(f"  fixed QAOA seed: {args.qaoa_seed}")
    print("=" * 84, flush=True)

    T = _load_v1_training_module()
    all_scen = fold_scenarios(T)
    cfg = dict(all_scen)[args.fold]

    hp_q = dict(rec["hyperparams"])
    t0 = time.time()

    def guarded(hp, only, run_cfg, dns_held, physics_seed):
        """Execute one arm and preserve its explicit completion status."""
        r = run_arm(T, args.fold, run_cfg, dns_held, hp, only,
                    verbose=False, seed=args.qaoa_seed)
        d = {m: float(r.get(m, np.nan)) for m in METRICS}
        d["seed"] = int(physics_seed)
        d["physics_seed"] = int(physics_seed)
        d["qaoa_seed"] = int(args.qaoa_seed)
        d["wall_s"] = float(r.get("wall_s", np.nan))
        d["completed"] = bool(r.get("completed", True))
        d["abort"] = r.get("abort")
        return d

    q_runs = []
    run_contexts = []
    for i in range(args.repeats):
        physics_seed = args.seed + i
        run_cfg = {**cfg, "phys_seed": physics_seed}
        dns_held = T._precompute_dns_for(
            [(args.fold, run_cfg)],
            label=f"physics-seed/{args.fold}/{physics_seed}")
        q_runs.append(guarded(
            hp_q, False, run_cfg, dns_held, physics_seed))
        run_contexts.append((run_cfg, dns_held))
        print(f"  Q-HAS run {i + 1}/{args.repeats}: "
              f"combined={q_runs[-1]['combined']:.4f} "
              f"phys={q_runs[-1]['phys_score']:.4f} "
              f"patch={q_runs[-1]['patch_ratio']:.4f}"
              f"{'' if q_runs[-1]['completed'] else '   **ABORTED**'}",
              flush=True)

    for i, (run, (run_cfg, dns_held)) in enumerate(
            zip(q_runs, run_contexts)):
        if not run["completed"]:
            run["budget_match"] = None
            continue
        try:
            best, trace = bisect_threshold_for_budget(
                T, args.fold, run_cfg, dns_held, hp_q,
                run["patch_ratio"],
                max_iter=args.match_max_iter, tol=args.match_tol,
                seed=args.qaoa_seed)
            mismatch = float(best["patch_ratio"] - run["patch_ratio"])
            converged = bool(abs(mismatch) <= args.match_tol)
            run["budget_match"] = {
                "converged": converged,
                "mismatch": mismatch,
                "classical": best,
                "trace": trace,
                "delta_phys": float(run["phys_score"] - best["phys_score"]),
            }
        except RuntimeError as exc:
            run["budget_match"] = {
                "converged": False, "error": str(exc), "trace": []}
        print(f"  matched budget {i + 1}/{args.repeats}: "
              f"{run['budget_match'].get('converged', False)}", flush=True)

    c_runs = [run["budget_match"]["classical"] for run in q_runs
              if run.get("budget_match", {}).get("converged")]

    n_ab = sum(1 for r in q_runs if not r["completed"])
    if n_ab:
        print(f"\n  {n_ab} run(s) ABORTED — excluded from the statistics",
              flush=True)
    q_ok = [r for r in q_runs if r["completed"]]
    c_ok = c_runs
    q_stats = summarise(q_ok) if q_ok else None
    c_stats = summarise(c_ok) if c_ok else None

    if q_stats:
        print("\n  " + "-" * 80)
        print(f"  {'metric':<14}{'Q-HAS mean':>12}{'std':>10}{'range':>10}")
        for metric in METRICS:
            stats = q_stats[metric]
            print(f"  {metric:<14}{stats['mean']:>12.4f}"
                  f"{stats['std']:>10.4f}{stats['range']:>10.4f}")

    paired = [run for run in q_ok
              if run.get("budget_match", {}).get("converged")]
    deltas = np.array(
        [run["budget_match"]["delta_phys"] for run in paired], dtype=float)
    comparison = {
        "n_paired": len(paired),
        "mean_delta_phys": (float(np.mean(deltas)) if deltas.size else None),
        "std_delta_phys": (float(np.std(deltas, ddof=1))
                           if deltas.size > 1 else 0.0 if deltas.size else None),
        "qhas_lower_error": int(np.sum(deltas < 0)),
        "classical_lower_error": int(np.sum(deltas > 0)),
        "ties": int(np.sum(deltas == 0)),
    }
    print(f"\n  matched comparisons: {len(paired)}/{args.repeats}")
    if deltas.size:
        print(f"  mean phys delta (Q-HAS - classical): "
              f"{comparison['mean_delta_phys']:+.5f}")

    out = {
        "artifact": "closed_loop_physics_seed_replicates",
        "schema": 2,
        "replication_unit": "trajectory",
        "status": ("complete" if len(paired) == args.repeats
                   and len(q_ok) == args.repeats else "incomplete"),
        "fold": args.fold,
        "scenario": rec["scenario"],
        "Re": cfg.get("Re"),
        "repeats": args.repeats,
        "n_aborted": int(n_ab),
        "n_completed_qhas": len(q_ok),
        "qhas_runs": q_runs,
        "classical_runs": c_runs,
        "qhas_stats": q_stats,
        "classical_stats": c_stats,
        "physics_seeds": [args.seed + i for i in range(args.repeats)],
        "qaoa_seed": args.qaoa_seed,
        "matched_comparison": comparison,
        "parent_campaign_contract_sha256": rec["campaign_contract_sha256"],
        "shots": cfg.get("shots"),
        **provenance.finish(prov),
        "cli_args": vars(args),
        "wall_s": time.time() - t0,
    }
    output_prefix = ("t20_qhas_run_variance" if args.prefix == "t15_level3"
                     else f"{args.prefix}_physics_seed_replicates")
    op = os.path.join(RESULTS_DIR, f"{output_prefix}_{args.fold}.json")
    _atomic_json(op, out)
    print(f"\n  saved: {os.path.basename(op)} ({time.time() - t0:.0f}s)")
    if out["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
