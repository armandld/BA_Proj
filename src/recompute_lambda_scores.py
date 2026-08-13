#!/usr/bin/env python3
"""
Recompute Optuna trial scores with different lambda_cost values.
================================================================

Loads existing Optuna studies from SQLite, recalculates the combined score
using a user-specified lambda_cost, and produces:

  1. A ranked CSV of all trials with recalculated scores
  2. A text summary (top trials, best params)
  3. Pareto front plot (phys_score vs patch_ratio) with new iso-score lines
  4. Convergence plot (re-ranked by new score)
  5. Score decomposition vs hyperparameters
  6. Lambda sweep comparison (if --lambda-sweep is used)

Output directory is tagged with the lambda value, e.g.:
    Train_results/rescore_phase1_lambda0.30/

Usage:
    # Single lambda:
    python recompute_lambda_scores.py --db-path ../Train_results/q_has_v2_phase1.db \\
        --study-name q_has_v2_phase1 --lambda-cost 0.3

    # Sweep multiple lambdas at once:
    python recompute_lambda_scores.py --db-path ../Train_results/q_has_v2_phase1.db \\
        --study-name q_has_v2_phase1 --lambda-sweep 0.0 0.1 0.2 0.3 0.5 1.0

    # Multi-scenario (Phase 1/2) studies:
    python recompute_lambda_scores.py --db-path ../Train_results/q_has_v2_phase1.db \\
        --study-name q_has_v2_phase1 --lambda-cost 0.5
"""

import argparse
import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────

ALL_SCENARIO_KEYS = [
    "kh", "vortex", "tearing", "coalescence",
    "kh_predict", "vortex_predict", "tearing_predict", "coalescence_predict",
    "ot", "rotor",
    "ot_predict", "rotor_predict","gt", "gt_predict",
]
SCENARIO_LABELS = {
    "kh": "Kelvin-Helmholtz", "vortex": "Lamb-Oseen Vortex",
    "tearing": "Harris Tearing", "coalescence": "Island Coalescence",
    "kh_predict": "Kelvin-Helmholtz", "vortex_predict": "Lamb-Oseen Vortex",
    "tearing_predict": "Harris Tearing", "coalescence_predict": "Island Coalescence",
    "ot": "Orszag-Tang", "rotor": "MHD Rotor",
    "ot_predict": "Orszag-Tang", "rotor_predict": "MHD Rotor",
    "gt": "Ghost Twisting", "gt_predict": "Ghost Twisting",
}
SCENARIO_COLORS = {
    "kh": "tab:blue", "vortex": "tab:green",
    "tearing": "tab:orange", "coalescence": "tab:red",
    "kh_predict": "tab:blue", "vortex_predict": "tab:green",
    "tearing_predict": "tab:orange", "coalescence_predict": "tab:red",
    "ot": "tab:purple", "rotor": "tab:brown",
    "ot_predict": "tab:purple", "rotor_predict": "tab:brown",
    "gt": "tab:pink", "gt_predict": "tab:pink",
}


# ─────────────────────────────────────────────────────────
#  Data extraction
# ─────────────────────────────────────────────────────────

def load_completed_trials(db_path, study_name):
    """Load study and return list of completed trials with finite scores."""
    storage_url = f"sqlite:///{os.path.abspath(db_path)}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value < float("inf")
    ]
    print(f"Loaded study '{study_name}': "
          f"{len(study.trials)} total, {len(completed)} completed (finite)")
    return study, completed


def _detect_scenario_keys(completed):
    """Auto-detect which scenario keys are actually present in user_attrs."""
    found = set()
    for t in completed[:10]:
        for key in ALL_SCENARIO_KEYS:
            if f"phys_{key}" in t.user_attrs:
                found.add(key)
    # Return in the canonical order defined by ALL_SCENARIO_KEYS
    return [k for k in ALL_SCENARIO_KEYS if k in found]


def is_multi_scenario(completed):
    """True if the study has per-scenario data (Phase 1/2 composite)."""
    return len(_detect_scenario_keys(completed)) > 0


def recompute_score(trial, lambda_cost):
    """
    Recompute the combined score for a trial with a new lambda_cost.

    For single-scenario trials:
        new_combined = (phys_score + lambda * patch_ratio) / (1 + lambda)

    For multi-scenario trials:
        per-scenario: sub_loss_i = (phys_i + lambda * patch_i) / (1 + lambda)
        composite = mean(sub_losses)
    """
    attrs = trial.user_attrs
    scenario_keys = _detect_scenario_keys([trial])

    if scenario_keys:
        # Multi-scenario: recompute each sub-loss then average
        sub_losses = []
        for key in scenario_keys:
            phys = attrs.get(f"phys_{key}", None)
            patch = attrs.get(f"patch_{key}", None)
            if phys is not None and patch is not None:
                sub = (phys + lambda_cost * patch) / (1 + lambda_cost)
                sub_losses.append(sub)
        if sub_losses:
            return np.mean(sub_losses)
        # Fallback: use global attrs
        return _recompute_global(attrs, lambda_cost, trial.value)
    else:
        return _recompute_global(attrs, lambda_cost, trial.value)


def _recompute_global(attrs, lambda_cost, original_value):
    """Recompute from global phys_score / patch_ratio."""
    phys = attrs.get("phys_score", None)
    patch = attrs.get("patch_ratio", None)
    if phys is not None and patch is not None:
        return (phys + lambda_cost * patch) / (1 + lambda_cost)
    # Cannot recompute — return original
    return original_value


def build_trial_table(completed, lambda_cost):
    """Build a list of dicts with trial info and recomputed scores."""
    scenario_keys = _detect_scenario_keys(completed)
    rows = []
    for t in completed:
        row = {
            "trial": t.number,
            "original_score": t.value,
            "new_score": recompute_score(t, lambda_cost),
        }
        # Global decomposition — fall back to averaging per-scenario values
        global_phys = t.user_attrs.get("phys_score", None)
        global_patch = t.user_attrs.get("patch_ratio", None)
        if global_phys is None and scenario_keys:
            per_phys = [t.user_attrs.get(f"phys_{k}", np.nan) for k in scenario_keys]
            per_phys = [v for v in per_phys if not np.isnan(v)]
            global_phys = float(np.mean(per_phys)) if per_phys else np.nan
        if global_patch is None and scenario_keys:
            per_patch = [t.user_attrs.get(f"patch_{k}", np.nan) for k in scenario_keys]
            per_patch = [v for v in per_patch if not np.isnan(v)]
            global_patch = float(np.mean(per_patch)) if per_patch else np.nan
        row["phys_score"] = global_phys if global_phys is not None else np.nan
        row["patch_ratio"] = global_patch if global_patch is not None else np.nan

        # Per-field errors
        for field in ["vx", "vy", "Bx", "By", "Jz"]:
            row[f"error_{field}"] = t.user_attrs.get(f"error_{field}", np.nan)

        # Per-scenario
        for key in scenario_keys:
            phys = t.user_attrs.get(f"phys_{key}", np.nan)
            patch = t.user_attrs.get(f"patch_{key}", np.nan)
            if not np.isnan(phys) and not np.isnan(patch):
                row[f"new_loss_{key}"] = (phys + lambda_cost * patch) / (1 + lambda_cost)
            else:
                row[f"new_loss_{key}"] = np.nan
            row[f"phys_{key}"] = phys
            row[f"patch_{key}"] = patch

        # Hyperparameters
        for k, v in t.params.items():
            row[f"param_{k}"] = v
        rows.append(row)

    rows.sort(key=lambda r: r["new_score"])
    return rows


# ─────────────────────────────────────────────────────────
#  Output: CSV + summary
# ─────────────────────────────────────────────────────────

def save_csv(rows, output_dir, lambda_cost):
    """Write trial table to CSV."""
    import csv
    path = os.path.join(output_dir, f"trials_lambda{lambda_cost:.4f}.csv")
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {os.path.basename(path)} ({len(rows)} trials)")


def save_summary(rows, completed, lambda_cost, output_dir, original_lambda=None):
    """Text summary of re-ranked results."""
    lines = []
    lines.append("=" * 65)
    lines.append(f"Lambda Rescore Summary  (lambda_cost = {lambda_cost:.4f})")
    if original_lambda is not None:
        lines.append(f"Original training lambda_cost = {original_lambda}")
    lines.append("=" * 65)
    lines.append(f"Total completed trials: {len(rows)}")

    if rows:
        best = rows[0]
        lines.append(f"\nBest trial (new ranking): #{best['trial']}")
        lines.append(f"  new_score    = {best['new_score']:.6f}")
        lines.append(f"  original_score = {best['original_score']:.6f}")
        lines.append(f"  phys_score   = {best['phys_score']:.6f}")
        lines.append(f"  patch_ratio  = {best['patch_ratio']:.6f}")

        # Field errors
        for field in ["vx", "vy", "Bx", "By", "Jz"]:
            val = best.get(f"error_{field}", np.nan)
            if not np.isnan(val):
                lines.append(f"  error_{field:>2s}    = {val:.6f}")

        # Per-scenario
        scenario_keys = _detect_scenario_keys(completed)
        if scenario_keys:
            lines.append(f"\n  Per-scenario (new lambda):")
            for key in scenario_keys:
                new_loss = best.get(f"new_loss_{key}", np.nan)
                phys = best.get(f"phys_{key}", np.nan)
                patch = best.get(f"patch_{key}", np.nan)
                if not np.isnan(new_loss):
                    label = SCENARIO_LABELS.get(key, key)
                    lines.append(f"    {label:>20s}: loss={new_loss:.6f}  "
                                 f"phys={phys:.6f}  patch={patch:.4f}")

        # Params
        param_keys = [k for k in best.keys() if k.startswith("param_")]
        lines.append(f"\n  Best params:")
        for k in param_keys:
            lines.append(f"    {k.replace('param_', ''):>18s} = {best[k]}")

        lines.append(f"\nTop 10 trials (re-ranked):")
        lines.append(f"  {'#':>4s}  {'new_score':>10s}  {'orig_score':>10s}  "
                      f"{'phys':>8s}  {'patch':>8s}")
        lines.append(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")
        for r in rows[:10]:
            lines.append(f"  {r['trial']:>4d}  {r['new_score']:>10.6f}  "
                          f"{r['original_score']:>10.6f}  "
                          f"{r['phys_score']:>8.6f}  {r['patch_ratio']:>8.4f}")

        # Rank changes
        lines.append(f"\nBiggest rank changes vs original:")
        orig_rank = {r["trial"]: i for i, r in
                     enumerate(sorted(rows, key=lambda x: x["original_score"]))}
        new_rank = {r["trial"]: i for i, r in enumerate(rows)}
        changes = [(t, orig_rank[t] - new_rank[t]) for t in orig_rank]
        changes.sort(key=lambda x: abs(x[1]), reverse=True)
        for trial_num, delta in changes[:5]:
            direction = "UP" if delta > 0 else "DOWN"
            lines.append(f"  Trial #{trial_num:>3d}: moved {abs(delta):>3d} places {direction}")

    summary = "\n".join(lines)
    print(summary)
    path = os.path.join(output_dir, f"summary_lambda{lambda_cost:.4f}.txt")
    with open(path, "w") as f:
        f.write(summary)


# ─────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────

def _save(fig, output_dir, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


def _pareto_front(points):
    is_pareto = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if is_pareto[i]:
            dominated = (np.all(points[i] <= points, axis=1)
                         & np.any(points[i] < points, axis=1))
            is_pareto[dominated] = False
            is_pareto[i] = True
    return is_pareto


def _add_trend(ax, x_vals, y_vals, color="red", n_bins=15):
    """Médiane par classe. Copie de celle d'`analyze_hyperparams`.

    D-61, second site : la dernière classe est FERMEE. Le dernier bord vaut
    `x.max()`, donc un `<` strict excluait de toute classe l'essai portant
    la plus grande valeur du paramètre. Les deux copies doivent rendre la
    même chose — `tests/pipeline/test_trend_last_bin_closed.py` le vérifie
    en les comparant sur la même entrée.
    """
    x, y = np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)
    if len(x) < 5:
        return
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    centers, medians = [], []
    for k in range(n_bins):
        if k == n_bins - 1:
            mask = (x >= bins[k]) & (x <= bins[k + 1])
        else:
            mask = (x >= bins[k]) & (x < bins[k + 1])
        if mask.sum() >= 2:
            centers.append((bins[k] + bins[k + 1]) / 2)
            medians.append(np.median(y[mask]))
    if len(centers) >= 3:
        ax.plot(centers, medians, color=color, linewidth=2, alpha=0.8)


def _get_global_phys_patch(trial, scenario_keys):
    """Get global phys_score/patch_ratio, averaging per-scenario if needed."""
    phys = trial.user_attrs.get("phys_score", None)
    patch = trial.user_attrs.get("patch_ratio", None)
    if phys is None and scenario_keys:
        vals = [trial.user_attrs.get(f"phys_{k}", np.nan) for k in scenario_keys]
        vals = [v for v in vals if not np.isnan(v)]
        phys = float(np.mean(vals)) if vals else np.nan
    if patch is None and scenario_keys:
        vals = [trial.user_attrs.get(f"patch_{k}", np.nan) for k in scenario_keys]
        vals = [v for v in vals if not np.isnan(v)]
        patch = float(np.mean(vals)) if vals else np.nan
    return phys if phys is not None else np.nan, patch if patch is not None else np.nan


def plot_pareto_with_isocost(completed, lambda_cost, output_dir):
    """
    Pareto front with iso-score lines for the chosen lambda.
    Iso-score line: phys = score*(1+lambda) - lambda*patch
    """
    scenario_keys = _detect_scenario_keys(completed)
    phys = np.array([_get_global_phys_patch(t, scenario_keys)[0] for t in completed])
    patch = np.array([_get_global_phys_patch(t, scenario_keys)[1] for t in completed])
    mask = ~(np.isnan(phys) | np.isnan(patch))
    phys, patch = phys[mask], patch[mask]
    completed_f = [t for t, m in zip(completed, mask) if m]

    if len(phys) == 0:
        print("  [SKIP] No phys_score/patch_ratio data for Pareto plot")
        return

    new_scores = np.array([recompute_score(t, lambda_cost) for t in completed_f])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Iso-score lines
    patch_range = np.linspace(0, 1, 100)
    score_levels = np.quantile(new_scores, [0.05, 0.1,0.25, 0.5])
    for s in score_levels:
        iso_phys = s * (1 + lambda_cost) - lambda_cost * patch_range
        ax.plot(patch_range, iso_phys, "--", color="gray", alpha=0.4, linewidth=0.8)
        # Label at right edge
        y_at_1 = s * (1 + lambda_cost) - lambda_cost
        if 0 <= y_at_1 <= phys.max() * 1.2:
            ax.text(1.01, y_at_1, f"S={s:.4f}", fontsize=7, color="gray", va="center")

    sc = ax.scatter(patch, phys, c=new_scores, cmap="viridis_r",
                    s=40, alpha=0.7, edgecolors="k", linewidth=0.3)
    fig.colorbar(sc, ax=ax, label=f"Combined Score (lambda={lambda_cost:.4f})")

    best_idx = np.argmin(new_scores)
    ax.scatter(patch[best_idx], phys[best_idx], s=250, c="red", marker="*",
               zorder=5, label=f"Best (score={new_scores[best_idx]:.4f})")

    pts = np.column_stack([patch, phys])
    pmask = _pareto_front(pts)
    pareto = pts[pmask]
    pareto = pareto[pareto[:, 0].argsort()]
    if len(pareto) >= 2:
        ax.plot(pareto[:, 0], pareto[:, 1], "r--", linewidth=2, alpha=0.7,
                label="Pareto front")

    ax.set_xlabel("Patch Ratio (computational cost)", fontsize=12)
    ax.set_ylabel("Physics Score (L2 error)", fontsize=12)
    # D-62 : la fenêtre était codée en dur à (-0,05 ; 0,40). Sur l'étude
    # classique, 9 essais sur 125 et **3 des 46 points du front de Pareto**
    # tombaient hors cadre : la figure montrait un front qui s'arrête sans
    # rien dire de ce qui continue. On garde la fenêtre quand tout y entre
    # — la figure de l'étude quantique est inchangée, 0/178 hors cadre — et
    # on l'élargit aux données sinon. Aucun seuil inventé : les bornes
    # viennent des points tracés.
    ax.set_ylim(min(-0.05, float(phys.min()) - 0.05),
                max(0.4, float(phys.max()) * 1.05))
    ax.set_title(f"Pareto Front with Iso-Score Lines (lambda={lambda_cost:.4f})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, f"pareto_lambda{lambda_cost:.4f}.png")


def plot_convergence_reranked(completed, lambda_cost, output_dir):
    """Trial scores (recomputed) with running best."""
    new_scores = [(t.number, recompute_score(t, lambda_cost)) for t in completed]
    new_scores.sort(key=lambda x: x[0])  # sort by trial number

    numbers = [x[0] for x in new_scores]
    scores = [x[1] for x in new_scores]

    running_best = []
    best = float("inf")
    for s in scores:
        best = min(best, s)
        running_best.append(best)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(numbers, scores, s=15, alpha=0.5, c="steelblue", label="Rescored trials")
    ax.plot(numbers, running_best, "r-", linewidth=2, label="Running best")
    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel(f"Combined Score (lambda={lambda_cost:.4f})", fontsize=12)
    ax.set_title(f"Convergence (Rescored with lambda={lambda_cost:.4f})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, f"convergence_lambda{lambda_cost:.4f}.png")


def plot_decomposition_rescored(completed, lambda_cost, output_dir):
    """phys_score and patch_ratio vs each hyperparameter, colored by new score."""
    scenario_keys = _detect_scenario_keys(completed)
    phys = np.array([_get_global_phys_patch(t, scenario_keys)[0] for t in completed])
    patch = np.array([_get_global_phys_patch(t, scenario_keys)[1] for t in completed])
    mask = ~(np.isnan(phys) | np.isnan(patch))
    if mask.sum() == 0:
        return

    phys, patch = phys[mask], patch[mask]
    completed_f = [t for t, m in zip(completed, mask) if m]
    param_names = list(completed_f[0].params.keys())
    n = len(param_names)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, param in enumerate(param_names):
        vals = np.array([t.params[param] for t in completed_f])
        ax1 = axes[i]
        ax2 = ax1.twinx()

        ax1.scatter(vals, phys, s=15, alpha=0.4, c="tab:blue", label="phys_score")
        _add_trend(ax1, vals, phys, color="tab:blue")

        ax2.scatter(vals, patch, s=15, alpha=0.4, c="tab:orange", label="patch_ratio")
        _add_trend(ax2, vals, patch, color="tab:orange")

        ax1.set_xlabel(param, fontsize=12)
        ax1.set_ylabel("Physics Score", color="tab:blue", fontsize=10)
        ax2.set_ylabel("Patch Ratio", color="tab:orange", fontsize=10)
        ax1.set_title(param, fontsize=12, fontweight="bold")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.2)

    fig.suptitle(f"Score Decomposition (lambda={lambda_cost:.4f})", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir, f"decomposition_lambda{lambda_cost:.4f}.png")


def plot_scenario_reranked(completed, lambda_cost, output_dir):
    """Per-scenario rescored breakdown for top 10 trials."""
    scenario_keys = _detect_scenario_keys(completed)
    if not scenario_keys:
        return

    new_scores = [(t, recompute_score(t, lambda_cost)) for t in completed]
    new_scores.sort(key=lambda x: x[1])
    top = new_scores[:10]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top))
    width = 0.8

    bottom = np.zeros(len(top))
    for key in scenario_keys:
        losses = []
        for t, _ in top:
            phys = t.user_attrs.get(f"phys_{key}", 0)
            patch = t.user_attrs.get(f"patch_{key}", 0)
            losses.append((phys + lambda_cost * patch) / (1 + lambda_cost))
        losses = np.array(losses)
        ax.bar(x, losses, width, bottom=bottom,
               label=SCENARIO_LABELS.get(key, key),
               color=SCENARIO_COLORS.get(key, None), alpha=0.85)
        bottom += losses

    ax.set_xticks(x)
    ax.set_xticklabels([f"#{t.number}" for t, _ in top], fontsize=10)
    ax.set_xlabel("Trial (re-ranked)", fontsize=12)
    ax.set_ylabel(f"Loss (lambda={lambda_cost:.4f})", fontsize=12)
    ax.set_title(f"Per-Scenario Breakdown, Top 10 (lambda={lambda_cost:.4f})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    _save(fig, output_dir, f"scenario_breakdown_lambda{lambda_cost:.4f}.png")


# ─────────────────────────────────────────────────────────
#  Lambda sweep comparison
# ─────────────────────────────────────────────────────────

def plot_lambda_sweep(completed, lambdas, output_dir):
    """
    Compare how different lambda values change the best trial and top-5 ranking.
    Produces:
      - Best score vs lambda
      - Best trial ID vs lambda
      - Top-5 stability heatmap
    """
    print(f"\n=== Lambda Sweep: {lambdas} ===")

    best_scores = []
    best_trials = []
    top5_per_lambda = []

    for lam in lambdas:
        scores = [(t.number, recompute_score(t, lam)) for t in completed]
        scores.sort(key=lambda x: x[1])
        best_trials.append(scores[0][0])
        best_scores.append(scores[0][1])
        top5_per_lambda.append([s[0] for s in scores[:5]])

    # --- Plot 1: best score vs lambda ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(lambdas, best_scores, "o-", color="tab:blue", linewidth=2, markersize=8)
    ax1.set_xlabel("lambda_cost", fontsize=12)
    ax1.set_ylabel("Best Combined Score", fontsize=12)
    ax1.set_title("Best Score vs Lambda", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Annotate with trial number
    for i, (lam, score, trial) in enumerate(zip(lambdas, best_scores, best_trials)):
        ax1.annotate(f"#{trial}", (lam, score), textcoords="offset points",
                     xytext=(5, 8), fontsize=8, color="gray")

    # --- Plot 2: top-5 trial stability ---
    # Find all unique trials in any top-5
    all_top5 = sorted(set(t for top5 in top5_per_lambda for t in top5))
    trial_to_idx = {t: i for i, t in enumerate(all_top5)}

    heatmap = np.full((len(all_top5), len(lambdas)), np.nan)
    for j, (lam, top5) in enumerate(zip(lambdas, top5_per_lambda)):
        for rank, trial_num in enumerate(top5):
            heatmap[trial_to_idx[trial_num], j] = rank + 1

    im = ax2.imshow(heatmap, cmap="YlOrRd_r", aspect="auto", vmin=1, vmax=5)
    ax2.set_xticks(range(len(lambdas)))
    ax2.set_xticklabels([f"{l:.4f}" for l in lambdas], fontsize=10)
    ax2.set_yticks(range(len(all_top5)))
    ax2.set_yticklabels([f"#{t}" for t in all_top5], fontsize=9)
    ax2.set_xlabel("lambda_cost", fontsize=12)
    ax2.set_ylabel("Trial", fontsize=12)
    ax2.set_title("Top-5 Ranking Stability", fontsize=13)
    fig.colorbar(im, ax=ax2, label="Rank (1=best)", shrink=0.8)

    # Annotate cells with rank
    for i in range(len(all_top5)):
        for j in range(len(lambdas)):
            val = heatmap[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f"{int(val)}", ha="center", va="center",
                         fontsize=10, fontweight="bold",
                         color="white" if val <= 2 else "black")

    fig.suptitle("Lambda Sensitivity Analysis", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir, "lambda_sweep_comparison.png")

    # --- Plot 3: Pareto front colored by lambda preference ---
    scenario_keys = _detect_scenario_keys(completed)
    phys = np.array([_get_global_phys_patch(t, scenario_keys)[0] for t in completed])
    patch = np.array([_get_global_phys_patch(t, scenario_keys)[1] for t in completed])
    mask = ~(np.isnan(phys) | np.isnan(patch))

    if mask.sum() > 0:
        phys_f, patch_f = phys[mask], patch[mask]
        completed_f = [t for t, m in zip(completed, mask) if m]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(patch_f, phys_f, s=20, alpha=0.3, c="gray", edgecolors="none")

        colors = plt.cm.coolwarm(np.linspace(0, 1, len(lambdas)))
        for lam, color in zip(lambdas, colors):
            scores = np.array([recompute_score(t, lam) for t in completed_f])
            best_idx = np.argmin(scores)
            ax.scatter(patch_f[best_idx], phys_f[best_idx], s=200, c=[color],
                       marker="*", zorder=5, edgecolors="k", linewidth=0.5)

            # Iso-score line for the best score
            best_s = scores[best_idx]
            p_line = np.linspace(0, 1, 100)
            iso = best_s * (1 + lam) - lam * p_line
            valid = iso >= 0
            ax.plot(p_line[valid], iso[valid], "--", color=color, alpha=0.5, linewidth=1)

        legend_elements = [Line2D([0], [0], marker="*", color="w",
                                  markerfacecolor=c, markersize=12,
                                  label=f"lambda={l:.4f}")
                           for l, c in zip(lambdas, colors)]
        ax.legend(handles=legend_elements, fontsize=9)
        ax.set_xlabel("Patch Ratio (computational cost)", fontsize=12)
        ax.set_ylabel("Physics Score (L2 error)", fontsize=12)
        ax.set_title("Best Trial per Lambda on Pareto Plane", fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, output_dir, "lambda_sweep_pareto.png")

    # Save sweep results as JSON
    sweep_data = {
        "lambdas": lambdas,
        "best_scores": [float(s) for s in best_scores],
        "best_trials": best_trials,
        "top5_per_lambda": top5_per_lambda,
    }
    path = os.path.join(output_dir, "lambda_sweep_results.json")
    with open(path, "w") as f:
        json.dump(sweep_data, f, indent=2)
    print(f"  -> lambda_sweep_results.json")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def run_single_lambda(completed, lambda_cost, output_dir):
    """Full analysis for one lambda value."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  Rescoring with lambda_cost = {lambda_cost:.4f}")
    print(f"  Output: {output_dir}")
    print(f"{'='*50}")

    rows = build_trial_table(completed, lambda_cost)
    save_csv(rows, output_dir, lambda_cost)
    save_summary(rows, completed, lambda_cost, output_dir)

    print(f"\n--- Plots (lambda={lambda_cost:.4f}) ---")
    plot_pareto_with_isocost(completed, lambda_cost, output_dir)
    plot_convergence_reranked(completed, lambda_cost, output_dir)
    plot_decomposition_rescored(completed, lambda_cost, output_dir)
    plot_scenario_reranked(completed, lambda_cost, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Recompute Optuna scores with different lambda_cost values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db-path", required=True,
                        help="Path to Optuna SQLite database (.db file)")
    parser.add_argument("--study-name", required=True,
                        help="Optuna study name inside the database")
    parser.add_argument("--lambda-cost", type=float, default=None,
                        help="Single lambda_cost value to rescore with")
    parser.add_argument("--lambda-sweep", type=float, nargs="+", default=None,
                        help="Multiple lambda values for sweep analysis "
                             "(e.g. --lambda-sweep 0.0 0.1 0.3 0.5 1.0)")
    parser.add_argument("--output-dir", default=None,
                        help="Base output directory (default: next to .db)")
    args = parser.parse_args()

    if args.lambda_cost is None and args.lambda_sweep is None:
        parser.error("Provide either --lambda-cost or --lambda-sweep (or both)")
    # D-63 : le `try` couvrait TOUT le corps de `main` et n'a jamais rendu
    # autre chose que 0. Un échec d'écriture ou de tracé, survenu bien après
    # le chargement, s'annonçait « Erreur lors du chargement » — mesuré :
    # l'étude se charge (« 178 completed »), puis `os.makedirs` échoue, et le
    # script sort **0** en laissant en place les artefacts du run précédent.
    # `CLAUDE.md` : un balayage vide doit crier. Seul le chargement est
    # rattrapé ici ; le reste remonte avec sa trace et son code non nul.
    try:
        study, completed = load_completed_trials(args.db_path, args.study_name)
    except Exception as e:
        print(f"Erreur lors du chargement : {e}", file=sys.stderr)
        sys.exit(1)

    if not completed:
        print("[ERROR] No completed trials with finite score.")
        sys.exit(1)

    base_dir = args.output_dir or os.path.dirname(os.path.abspath(args.db_path))

    # Collect all lambda values to process
    lambdas_to_run = []
    if args.lambda_cost is not None:
        lambdas_to_run.append(args.lambda_cost)
    if args.lambda_sweep is not None:
        lambdas_to_run.extend(args.lambda_sweep)
    # Deduplicate while preserving order
    seen = set()
    unique_lambdas = []
    for l in lambdas_to_run:
        if l not in seen:
            seen.add(l)
            unique_lambdas.append(l)

    # Run per-lambda analysis
    for lam in unique_lambdas:
        out = os.path.join(base_dir, f"rescore_{args.study_name}_lambda{lam:.4f}")
        run_single_lambda(completed, lam, out)

    # Lambda sweep comparison (if multiple values)
    sweep_lambdas = args.lambda_sweep or unique_lambdas
    if len(sweep_lambdas) >= 2:
        sweep_dir = os.path.join(base_dir, f"rescore_{args.study_name}_sweep")
        os.makedirs(sweep_dir, exist_ok=True)
        plot_lambda_sweep(completed, sorted(sweep_lambdas), sweep_dir)

    print(f"\nDone. All outputs in: {base_dir}/rescore_*")


if __name__ == "__main__":
    main()
