#!/usr/bin/env python3
"""Aggregate the current V3 task artifacts into a master table.

The ``reference`` column is an archived, pre-audit baseline retained for
context only. ``DIFF`` therefore does not mean regression. Current campaign
validation refuses missing inputs; comparison to the archive is opt-in.

Sorties :
  results/v3_master_table.csv
  results/v3_master_table.md
  results/v3_master_N{N}.npz   (hash git + arguments CLI, garde-fous v3)

Usage :
  python study/common/aggregate_v3.py --N 256 --dim 4
  python study/common/aggregate_v3.py --strict
  python study/common/aggregate_v3.py --check-archive
"""
import argparse, json, os, sys
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

from metrics import spearman
from h2b_feature_selection import git_commit_hash

TOL = 0.002   # les references de RESULTS.md sont a 3 decimales


# -------------------------------------------------------------------
# Helpers purs (testables)
# -------------------------------------------------------------------

def status_of(value, ref, tol=TOL):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "MISSING"
    if ref is None:
        return "OK"          # ligne informative sans reference
    return "OK" if abs(float(value) - float(ref)) <= tol else "DIFF"


def make_row(task, metric, value, ref=None, tol=TOL):
    return dict(task=task, metric=metric,
                value=(None if value is None else float(value)),
                ref=(None if ref is None else float(ref)),
                status=status_of(value, ref, tol))


def load_npz(path):
    """dict ou None si absent (les lignes deviennent MISSING)."""
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def missing_rows(task, metrics):
    return [make_row(task, m, None, None) for m in metrics]


# -------------------------------------------------------------------
# Extracteurs par tache (prennent des dicts -> testables)
# -------------------------------------------------------------------

def rows_t1(d):
    metrics = ["LOSO classical mean F1", "LOSO B5 score-only mean F1",
               "LOSO full-9 (B4) mean F1"]
    if d is None:
        return missing_rows("t1", metrics)
    return [
        make_row("t1", metrics[0], np.mean(d["f1_classical"]), 0.434),
        make_row("t1", metrics[1], np.mean(d["f1_b5"]), 0.256),
        make_row("t1", metrics[2], np.mean(d["f1_full9"]), 0.189),
    ]


def rows_t1b(d):
    refs_loso = {0: 0.189, 1: 0.215, 2: 0.140, 3: 0.140}
    refs_blk = {0: 0.581, 1: 0.733, 2: 0.816, 3: 0.816}
    metrics = ([f"LOSO mean F1 k={k}" for k in refs_loso]
               + [f"blocked F1 k={k}" for k in refs_blk])
    if d is None:
        return missing_rows("t1b", metrics)
    out = []
    for ki, k in enumerate(d["k_values"]):
        out.append(make_row("t1b", f"LOSO mean F1 k={k}",
                            d["f1_loso_mean"][ki], refs_loso[int(k)]))
    for ki, k in enumerate(d["k_values"]):
        out.append(make_row("t1b", f"blocked F1 k={k}",
                            d["f1_blocked"][ki], refs_blk[int(k)]))
    return out


def rows_t4(d):
    spec = [  # (nom de methode, champ, ref)
        ("B2 classical (block_max)", "random_f1", 0.475),
        ("B4 gbt-9 (max)", "random_f1", 0.980),
        ("B4 gbt-9 (max)", "blocked_f1", 0.581),
        ("B4 gbt-9 (avg)", "blocked_f1", 0.738),
        ("B1 classical (block_avg)", "blocked_rho", 0.767),
        ("B2 classical (block_max)", "blocked_rho", 0.365),
        ("B1 classical (block_avg)", "blocked_auc", 0.595),
    ]
    if d is None:
        return missing_rows("t4", [f"{n} [{f}]" for n, f, _ in spec]
                            + ["leakage gap B4(max) F1"])
    names = [str(x) for x in d["names"]]
    out = []
    for n, f, ref in spec:
        i = names.index(n)
        out.append(make_row("t4", f"{n} [{f}]", d[f][i], ref))
    i = names.index("B4 gbt-9 (max)")
    out.append(make_row("t4", "leakage gap B4(max) F1",
                        d["random_f1"][i] - d["blocked_f1"][i], 0.399))
    return out


def rows_t5(d):
    spec_mean = [  # (variant, agg, ref)
        ("v1-classical (no psi)", "avg", 0.533),
        ("v1-classical (no psi)", "max", 0.468),
        ("v1+psi signed b=9.94 (trial4)", "avg", 0.308),
        ("v1+psi signed b=9.94 (trial4)", "max", 0.453),
        ("v1+psi legacy-abs b=0.5495 (11e repro)", "avg", 0.507),
        ("v1+psi_v2 signed (param-free)", "avg", 0.303),
    ]
    spec_boot = [  # (variant|agg, ref_mean, ref_lo, ref_hi)
        ("v1+psi signed b=9.94 (trial4)|avg", -0.225, -0.414, -0.049),
        ("v1+psi_v2 signed (param-free)|avg", -0.229, -0.412, -0.052),
    ]
    if d is None:
        return missing_rows(
            "t5", [f"LOSO mean F1 {v} [{a}]" for v, a, _ in spec_mean]
            + [f"boot dpsi {k}" for k, *_ in spec_boot])
    variants = [str(x) for x in d["variants"]]
    aggs = [str(x) for x in d["aggs"]]
    out = []
    for v, a, ref in spec_mean:
        f1 = d["f1_fold"][variants.index(v), aggs.index(a)]
        out.append(make_row("t5", f"LOSO mean F1 {v} [{a}]",
                            np.mean(f1), ref))
    boot_keys = [str(x) for x in d["boot_variant"]]
    for key, ref_m, ref_lo, ref_hi in spec_boot:
        i = boot_keys.index(key)
        out.append(make_row("t5", f"boot dpsi {key} mean",
                            d["boot_mean"][i], ref_m))
        out.append(make_row("t5", f"boot dpsi {key} CI_lo",
                            d["boot_ci_low"][i], ref_lo))
        out.append(make_row("t5", f"boot dpsi {key} CI_hi",
                            d["boot_ci_high"][i], ref_hi))
    return out


def rows_t6(d):
    if d is None:
        return missing_rows("t6", ["dynamic-label Spearman(d_i, e_i)"])
    rho = spearman(d["d_errors"].ravel(), d["l2_errors"].ravel())
    return [make_row("t6", "dynamic-label Spearman(d_i, e_i)", rho)]


def rows_t7(d):
    if d is None:
        return missing_rows(
            "t7", ["B1 LOSO capture@0.25 h=1", "B1 LOSO capture@0.25 h=8",
                   "base9 LOSO CE@0.25 h=1"]
            + [f"boot psi4-base9 LOSO h={h}" for h in (1, 2, 4, 8)])
    methods = [str(x) for x in d["methods"]]
    horizons = list(d["horizons"])
    budgets = list(d["budgets"])
    i_loso = 1                      # splits = [blocked, loso]
    i_b1 = methods.index("B1 classical score (avg)")
    i_b9 = methods.index("base9")
    out = [
        make_row("t7", "B1 LOSO capture@0.25 h=1",
                 d["capture25"][i_loso, i_b1, horizons.index(1)], 0.694),
        make_row("t7", "B1 LOSO capture@0.25 h=8",
                 d["capture25"][i_loso, i_b1, horizons.index(8)], 0.665),
        make_row("t7", "base9 LOSO CE@0.25 h=1",
                 d["ce"][i_loso, i_b9, horizons.index(1),
                         budgets.index(0.25)], 0.215),
    ]
    boot = json.loads(str(d["boot"]))
    refs = {1: 0.026, 2: -0.008, 4: -0.011, 8: 0.020}
    for h, ref in refs.items():
        e = next(b for b in boot
                 if b["split"] == "loso" and b["h"] == h
                 and b["a"] == "base9+psi4" and b["b"] == "base9")
        out.append(make_row("t7", f"boot psi4-base9 LOSO h={h}",
                            e["mean"], ref))
    # cone LOSO h=1 (k = 0, 1, 2)
    for ki, ref in enumerate((0.215, 0.285, 0.201)):
        out.append(make_row("t7", f"cone LOSO CE@0.25 k={ki} h=1",
                            d["cone_ce25"][i_loso, ki,
                                           horizons.index(1)], ref))
    return out


def rows_t9(d):
    refs = {("v1", 2): 0.008, ("v2", 2): 0.034,
            ("v1", 4): 0.221, ("v2", 4): 0.000}
    if d is None:
        return missing_rows(
            "t9", [f"mean frac {m} dim={k}" for m, k in refs])
    mappers = np.array([str(x) for x in d["mapper"]])
    out = []
    for (m, dim), ref in refs.items():
        mask = (mappers == m) & (d["dim"] == dim)
        val = float(np.mean(d["frac"][mask])) if mask.any() else None
        out.append(make_row("t9", f"mean frac {m} dim={dim}", val, ref))
    return out


def rows_t29(d, dim):
    """Trajectory-bootstrap LOSO deltas for the requested patch grid."""
    metrics = ("mean stencil-site delta", "CI-positive folds",
               "CI-negative folds", "non-degenerate folds")
    if d is None:
        return missing_rows("t29", metrics)
    mask = np.asarray(d["dim"], dtype=int) == dim
    if not mask.any():
        return missing_rows("t29", metrics)
    delta = np.asarray(d["delta"], dtype=float)[mask]
    low = np.asarray(d["ci_low"], dtype=float)[mask]
    high = np.asarray(d["ci_high"], dtype=float)[mask]
    constant = np.asarray(d["constant_compared"]).astype(str)[mask]
    valid = constant == ""
    return [
        make_row("t29", metrics[0], float(np.mean(delta[valid]))
                 if valid.any() else None),
        make_row("t29", metrics[1], float(np.sum(valid & (low > 0)))),
        make_row("t29", metrics[2], float(np.sum(valid & (high < 0)))),
        make_row("t29", metrics[3], float(np.sum(valid))),
    ]


# -------------------------------------------------------------------
# Collecte + sorties
# -------------------------------------------------------------------

def collect(results_dir, N, dim):
    rows = []
    rows += rows_t1(load_npz(os.path.join(
        results_dir, f"t1_feature_selection_N{N}_dim{dim}.npz")))
    rows += rows_t1b(load_npz(os.path.join(
        results_dir, f"t1b_cone_curve_N{N}_dim{dim}.npz")))
    rows += rows_t4(load_npz(os.path.join(
        results_dir, f"t4_blocked_split_N{N}_dim{dim}.npz")))
    rows += rows_t5(load_npz(os.path.join(
        results_dir, f"t5_v1_psi_loso_N{N}_dim{dim}.npz")))
    rows += rows_t6(load_npz(os.path.join(
        results_dir,
        f"d_patches_orszag_tang_Re400_N{N}_dim{dim}_tx1.npz")))
    rows += rows_t7(load_npz(os.path.join(
        results_dir, f"t7_horizon_N{N}_dim{dim}.npz")))
    rows += rows_t9(load_npz(os.path.join(results_dir,
                                          f"t9_prop2_N{N}.npz")))
    rows += rows_t29(load_npz(os.path.join(
        results_dir, f"t29_loso_delta_ci_N{N}_perscenario.npz")), dim)
    return rows


def to_markdown(rows, git_hash):
    lines = ["# V3 master table",
             "",
             f"Generated by `study/common/aggregate_v3.py` at commit "
             f"`{git_hash[:12]}`.",
             "",
             "**`reference` is the archived, pre-audit V3 baseline, not a "
             "current acceptance threshold.** OK/DIFF only describes that "
             "historical comparison.",
             "",
             "| task | metric | value | reference (archived V3 baseline) | status |",
             "|---|---|---|---|---|"]
    for r in rows:
        v = "—" if r["value"] is None else f"{r['value']:.3f}"
        ref = "—" if r["ref"] is None else f"{r['ref']:.3f}"
        lines.append(f"| {r['task']} | {r['metric']} | {v} | {ref} | "
                     f"{r['status']} |")
    return "\n".join(lines) + "\n"


def to_csv(rows, git_hash, cli):
    lines = [f"# git_hash={git_hash}", f"# cli={json.dumps(cli)}",
             "task,metric,value,reference,status"]
    for r in rows:
        v = "" if r["value"] is None else f"{r['value']:.6f}"
        ref = "" if r["ref"] is None else f"{r['ref']:.6f}"
        metric = r["metric"].replace(",", ";")
        lines.append(f"{r['task']},{metric},{v},{ref},{r['status']}")
    return "\n".join(lines) + "\n"


def output_paths(results_dir, N, dim, default_N=256, default_dim=4):
    suffix = "" if (N, dim) == (default_N, default_dim) else f"_N{N}_dim{dim}"
    return {
        "markdown": os.path.join(results_dir,
                                 f"v3_master_table{suffix}.md"),
        "csv": os.path.join(results_dir, f"v3_master_table{suffix}.csv"),
        "npz": os.path.join(results_dir, f"v3_master{suffix or f'_N{N}'}.npz"),
    }


def main():
    p = argparse.ArgumentParser(
        description="V3 Task 10: master table aggregation (§5.4)")
    from config import RESULTS_DIR, DNS_N

    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on any MISSING row")
    p.add_argument("--check-archive", action="store_true",
                   help="also exit non-zero on differences from the archive")
    p.add_argument("--allow-missing", action="store_true",
                   help="write a partial, configuration-scoped table")
    p.add_argument("--seed", type=int, default=0,
                   help="enregistre (agregation deterministe)")
    args = p.parse_args()

    gh = git_commit_hash()
    rows = collect(RESULTS_DIR, args.N, args.dim)

    md = to_markdown(rows, gh)
    print(md)
    n_ok = sum(r["status"] == "OK" for r in rows)
    n_diff = sum(r["status"] == "DIFF" for r in rows)
    n_miss = sum(r["status"] == "MISSING" for r in rows)
    print(f"  rows: {len(rows)}  OK={n_ok}  DIFF={n_diff}  "
          f"MISSING={n_miss}")

    if n_miss and not args.allow_missing:
        raise SystemExit(
            f"refusing to write a master table with {n_miss} MISSING rows; "
            "complete the campaign or pass --allow-missing")

    paths = output_paths(RESULTS_DIR, args.N, args.dim, DNS_N, 4)
    md_path = paths["markdown"]
    csv_path = paths["csv"]
    open(md_path, "w").write(md)
    open(csv_path, "w").write(to_csv(rows, gh, vars(args)))
    np.savez_compressed(
        paths["npz"],
        task=np.array([r["task"] for r in rows]),
        metric=np.array([r["metric"] for r in rows]),
        value=np.array([np.nan if r["value"] is None else r["value"]
                        for r in rows]),
        reference=np.array([np.nan if r["ref"] is None else r["ref"]
                            for r in rows]),
        status=np.array([r["status"] for r in rows]),
        seed=args.seed, git_hash=gh, cli_args=json.dumps(vars(args)),
    )
    print(f"  saved: {os.path.basename(md_path)}, "
          f"{os.path.basename(csv_path)}, {os.path.basename(paths['npz'])}")

    if args.strict and n_miss:
        sys.exit(1)
    if args.check_archive and n_diff:
        sys.exit(1)
    print("\nV3 Task 10 complete.")


if __name__ == "__main__":
    main()
