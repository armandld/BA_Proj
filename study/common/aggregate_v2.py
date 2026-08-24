#!/usr/bin/env python3
"""Phase 13: descriptive cross-phase aggregation.

The report preserves each QAOA/SA protocol separately. A headline mean is
computed only when every artifact has complete metadata and the same solver
protocol; incompatible backends, budgets or ablations are never pooled.

Output:
  results/SUMMARY_N{N}_dim{D}.csv
  results/SUMMARY_N{N}_dim{D}.txt

Usage:
  python study/common/aggregate_v2.py --dim 4
"""
import argparse, csv, glob, os, sys
import numpy as np

# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------
from config import RESULTS_DIR, DNS_N


def _safe_load(path):
    if not os.path.exists(path):
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"  (failed to load {os.path.basename(path)}: {e!r})")
        return None


def _get(z, key, default=None):
    if z is None:
        return default
    if key not in z.files:
        return default
    v = z[key]
    # scalar 0-d numpy -> python scalar
    if v.shape == ():
        return v.item()
    return v


def fmt(v, prec=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.{prec}f}"
    return str(v)


def _protocol_record(path, metric, protocol_fields):
    """Summarise one stochastic-solver artifact and its protocol."""
    z = _safe_load(path)
    if z is None:
        return None
    values = _get(z, metric)
    if values is None:
        return None
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return None

    protocol = []
    missing = []
    for field in protocol_fields:
        value = _get(z, field)
        if value is None:
            missing.append(field)
        else:
            protocol.append((field, value))
    return {
        "file": os.path.basename(path),
        "mean_f1": float(np.mean(values)),
        "n": int(len(values)),
        "protocol": tuple(protocol) if not missing else None,
        "missing_protocol_fields": tuple(missing),
        "seed": _get(z, "seed"),
    }


def _compatible_weighted_mean(records):
    """Pool artifact means only for one fully specified solver protocol."""
    if not records or any(r["protocol"] is None for r in records):
        return None
    protocols = {r["protocol"] for r in records}
    if len(protocols) != 1:
        return None
    total = sum(r["n"] for r in records)
    return sum(r["mean_f1"] * r["n"] for r in records) / total


def _protocol_label(record):
    if record["protocol"] is None:
        missing = ",".join(record["missing_protocol_fields"])
        return f"incomplete metadata (missing {missing})"
    return ", ".join(f"{key}={value}" for key, value in record["protocol"])


def collect(N, dim):
    """Collect all available results for (N, dim)."""
    R = {}

    # ---- phase 11 upper bound ----
    z = _safe_load(os.path.join(
        RESULTS_DIR, f"upper_bound_N{N}_dim{dim}.npz"))
    R["p11_f1_class"]      = _get(z, "f1_class_val")
    R["p11_f1_site_lr"]    = _get(z, "f1_site_lr")
    R["p11_f1_site_rf"]    = _get(z, "f1_site_rf")
    R["p11_f1_site_gbt"]   = _get(z, "f1_site_gbt")
    R["p11_f1_site_best"]  = _get(z, "f1_site_best")
    R["p11_f1_stencil"]    = _get(z, "f1_stencil_gbt")
    R["p11_delta_site"]    = _get(z, "delta_site_vs_class")
    R["p11_delta_sten"]    = _get(z, "delta_stencil_vs_site")

    # ---- phase 11b LOSO ----
    z = _safe_load(os.path.join(
        RESULTS_DIR, f"upper_bound_loso_N{N}_dim{dim}.npz"))
    R["p11b_f1_class_mean"] = _get(z, "f1_class_mean")
    R["p11b_f1_class_std"]  = _get(z, "f1_class_std")
    R["p11b_f1_site_mean"]  = _get(z, "f1_site_mean")
    R["p11b_f1_site_std"]   = _get(z, "f1_site_std")
    R["p11b_f1_sten_mean"]  = _get(z, "f1_sten_mean")
    R["p11b_f1_sten_std"]   = _get(z, "f1_sten_std")
    R["p11b_per_fold"]      = None
    if z is not None and "held" in z.files:
        R["p11b_per_fold"] = list(zip(
            z["held"], z["f1_class"], z["f1_site"], z["f1_sten"]))

    # ---- phase 11c learned H ----
    z = _safe_load(os.path.join(
        RESULTS_DIR, f"learned_h_N{N}_dim{dim}.npz"))
    R["p11c_f1_val"]       = _get(z, "f1_val")
    R["p11c_f1_class_val"] = _get(z, "f1_class_val")
    R["p11c_w_std"]        = _get(z, "w_std")
    R["p11c_features"]     = _get(z, "feature_names")
    R["p11c_loso_mean"]    = None
    if z is not None and "loso_f1_learned" in z.files:
        R["p11c_loso_mean"] = float(np.mean(z["loso_f1_learned"]))
        R["p11c_loso_class_mean"] = float(np.mean(z["loso_f1_class"]))

    # ---- phase 12 VQC / QKE ----
    z = _safe_load(os.path.join(RESULTS_DIR, f"vqc_N{N}_dim{dim}.npz"))
    R["p12_f1_classical_score"] = _get(z, "f1_classical_score")
    R["p12_f1_lr_pca"]  = _get(z, "f1_lr_pca")
    R["p12_f1_gbt_pca"] = _get(z, "f1_gbt_pca")
    R["p12_f1_qke"]     = _get(z, "f1_qke")
    R["p12_f1_vqc"]     = _get(z, "f1_vqc")

    # ---- phase 10 trained H ----
    p10_f1 = {}
    for kind in ("joint",
                 "scenario-orszag_tang",
                 "scenario-harris_tearing",
                 "scenario-kelvin_helmholtz",
                 "scenario-mhd_rotor"):
        pth = os.path.join(
            RESULTS_DIR, f"train_{kind}_N{N}_dim{dim}.npz")
        z10 = _safe_load(pth)
        if z10 is None:
            continue
        f1 = _get(z10, "best_f1_test")
        c_bias = _get(z10, "best_c_bias")
        threshold = _get(z10, "best_thr")
        p10_f1[kind] = (f1, (c_bias, threshold))
    R["p10_trained"] = p10_f1

    # ---- phase 5 QAOA joint / per config ----
    qaoa_files = sorted(glob.glob(os.path.join(
        RESULTS_DIR, f"qaoa_*_N{N}_dim{dim}*v2.npz")))
    qaoa_f1 = []
    for fp in qaoa_files:
        record = _protocol_record(
            fp, "qaoa_f1",
            ("reps", "K_opt", "backend", "constant_initialisation",
             "prune_eps", "zero_psi"))
        if record is not None:
            qaoa_f1.append(record)
    R["p5_qaoa"] = qaoa_f1
    R["p5_qaoa_mean"] = _compatible_weighted_mean(qaoa_f1)

    # ---- phase 7 SA joint / per config ----
    sa_files = sorted(glob.glob(os.path.join(
        RESULTS_DIR, f"sa_baseline_*_N{N}_dim{dim}*_v2.npz")))
    sa_f1 = []
    for fp in sa_files:
        record = _protocol_record(
            fp, "sa_f1", ("sweeps", "n_restarts", "classical_warm"))
        if record is not None:
            sa_f1.append(record)
    R["p7_sa"] = sa_f1
    R["p7_sa_mean"] = _compatible_weighted_mean(sa_f1)

    return R


def format_report(R, N, dim):
    lines = []
    lines.append("=" * 88)
    lines.append(f"  Q-HAS study summary    N={N}  dim={dim}")
    lines.append("=" * 88)

    # -- headline table --
    lines.append("")
    lines.append("DESCRIPTIVE F1 SUMMARY")
    lines.append("-" * 88)
    lines.append(f"  {'quantity':<48} {'F1':>8}  {'source':<20}")
    lines.append("  " + "-" * 82)
    rows = [
        ("Classical AMR indicator (score > thr*)",
         R.get("p11_f1_class"), "phase 11"),
        ("Mean-field ceiling -- logistic regression",
         R.get("p11_f1_site_lr"), "phase 11"),
        ("Mean-field ceiling -- random forest",
         R.get("p11_f1_site_rf"), "phase 11"),
        ("Mean-field ceiling -- gradient boosting",
         R.get("p11_f1_site_gbt"), "phase 11"),
        ("Neighbourhood ceiling -- GBT on stencil",
         R.get("p11_f1_stencil"), "phase 11"),
        ("Learned mean-field Hamiltonian h_i = w.phi - b",
         R.get("p11c_f1_val"), "phase 11c"),
        ("QAOA  (v2 H, joint over scenarios)",
         R.get("p5_qaoa_mean"), "phase 5"),
        ("SA    (v2 H, joint over scenarios)",
         R.get("p7_sa_mean"), "phase 7"),
        ("VQC   (PCA-reduced features)",
         R.get("p12_f1_vqc"), "phase 12"),
        ("QKE   (quantum kernel + SVC)",
         R.get("p12_f1_qke"), "phase 12"),
    ]
    for name, v, src in rows:
        lines.append(f"  {name:<48} {fmt(v):>8}  {src:<20}")

    # -- LOSO cross-validation --
    lines.append("")
    lines.append("LEAVE-ONE-SCENARIO-OUT (cross-scenario generalisation)")
    lines.append("-" * 88)
    cmean = R.get("p11b_f1_class_mean")
    cstd  = R.get("p11b_f1_class_std")
    smean = R.get("p11b_f1_site_mean")
    sstd  = R.get("p11b_f1_site_std")
    nmean = R.get("p11b_f1_sten_mean")
    nstd  = R.get("p11b_f1_sten_std")
    if cmean is not None:
        lines.append(f"  {'Classical F1':<32} "
                     f"{fmt(cmean)} +/- {fmt(cstd)}")
        lines.append(f"  {'Mean-field LOSO F1':<32} "
                     f"{fmt(smean)} +/- {fmt(sstd)}")
        lines.append(f"  {'Stencil    LOSO F1':<32} "
                     f"{fmt(nmean)} +/- {fmt(nstd)}")
        if R.get("p11b_per_fold"):
            lines.append("  per fold:")
            lines.append(f"    {'held-out':<20} {'class':>8} "
                         f"{'site':>8} {'sten':>8}")
            for held, fc, fs, fn in R["p11b_per_fold"]:
                lines.append(f"    {str(held):<20} {fmt(fc):>8} "
                             f"{fmt(fs):>8} {fmt(fn):>8}")
    else:
        lines.append("  (not run)")

    # -- learned weights --
    lines.append("")
    lines.append("LEARNED MEAN-FIELD WEIGHTS (standardised features)")
    lines.append("-" * 88)
    w = R.get("p11c_w_std")
    feats = R.get("p11c_features")
    if w is not None and feats is not None:
        order = np.argsort(-np.abs(w))
        lines.append(f"  {'feature':<20} {'w_std':>10}")
        for k in order:
            lines.append(f"  {str(feats[k]):<20} {w[k]:>+10.3f}")
    else:
        lines.append("  (phase 11c not run)")

    # -- phase 10 trained outcomes --
    lines.append("")
    lines.append("PHASE 10 RESCUE FIT: UNTOUCHED TEST OUTCOMES")
    lines.append("-" * 88)
    p10 = R.get("p10_trained", {})
    if p10:
        lines.append(f"  {'mode':<30} {'F1_test':>8}  {'theta*':<32}")
        for k, (f, th) in p10.items():
            th_s = ("(c={:.2f}, thr={:.2f})".format(th[0], th[1])
                    if (th is not None and len(np.atleast_1d(th)) >= 2
                        and th[0] is not None and th[1] is not None)
                    else "--")
            lines.append(f"  {k:<30} {fmt(f):>8}  {th_s:<32}")
    else:
        lines.append("  (no phase 10 results found)")

    # -- QAOA / SA per config --
    lines.append("")
    lines.append("QAOA / SA PER CONFIG (v2 Hamiltonian)")
    lines.append("-" * 88)
    qaoa = R.get("p5_qaoa", [])
    sa   = R.get("p7_sa", [])
    if qaoa:
        lines.append(f"  {'QAOA file':<60} {'F1':>8} {'n':>5}")
        for record in qaoa:
            lines.append(
                f"  {record['file']:<60} {fmt(record['mean_f1']):>8} "
                f"{record['n']:>5}")
            lines.append(f"    protocol: {_protocol_label(record)}")
    if sa:
        lines.append(f"  {'SA   file':<60} {'F1':>8} {'n':>5}")
        for record in sa:
            lines.append(
                f"  {record['file']:<60} {fmt(record['mean_f1']):>8} "
                f"{record['n']:>5}")
            lines.append(f"    protocol: {_protocol_label(record)}")
    if qaoa and R.get("p5_qaoa_mean") is None:
        lines.append("  QAOA headline omitted: protocols are incomplete or incompatible.")
    if sa and R.get("p7_sa_mean") is None:
        lines.append("  SA headline omitted: protocols are incomplete or incompatible.")

    # -- verdicts --
    lines.append("")
    lines.append("DESCRIPTIVE DELTAS")
    lines.append("-" * 88)
    d_site = R.get("p11_delta_site")
    d_sten = R.get("p11_delta_sten")
    if d_site is not None:
        if d_sten is not None:
            lines.append(
                f"  mean-field ceiling beats classical by {d_site:+.3f}")
            lines.append(
                f"  stencil ceiling  beats mean-field by {d_sten:+.3f}")
    if (R.get("p11c_f1_val") is not None
            and R.get("p11c_f1_class_val") is not None):
        d = R["p11c_f1_val"] - R["p11c_f1_class_val"]
        lines.append(
            f"  learned mean-field H    F1 gain over classical = "
            f"{d:+.3f}   (phase 11c)")
    lines.append(
        "  These deltas are descriptive; claims require the registered "
        "paired uncertainty and closed-loop analyses.")

    lines.append("=" * 88)
    return "\n".join(lines)


def write_csv(path, R):
    rows = []
    for k, v in R.items():
        if isinstance(v, (list, dict)) or v is None:
            continue
        if isinstance(v, np.ndarray):
            if v.shape == ():
                v = v.item()
            else:
                continue
        rows.append((k, v))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in rows:
            w.writerow([k, v])


def main():
    p = argparse.ArgumentParser(
        description="Phase 13: aggregate results across phases")
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    args = p.parse_args()

    R = collect(args.N, args.dim)
    has_data = any(
        value is not None and not (
            isinstance(value, (list, dict)) and len(value) == 0)
        for value in R.values()
    )
    if not has_data:
        raise SystemExit(
            f"no study artifacts found for N={args.N}, dim={args.dim}")
    report = format_report(R, args.N, args.dim)
    print(report)

    out_txt = os.path.join(
        RESULTS_DIR, f"SUMMARY_N{args.N}_dim{args.dim}.txt")
    out_csv = os.path.join(
        RESULTS_DIR, f"SUMMARY_N{args.N}_dim{args.dim}.csv")
    with open(out_txt, "w") as f:
        f.write(report + "\n")
    write_csv(out_csv, R)
    print(f"\n  saved: {os.path.basename(out_txt)}")
    print(f"  saved: {os.path.basename(out_csv)}")


if __name__ == "__main__":
    main()
