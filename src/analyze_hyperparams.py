#!/usr/bin/env python3
"""
Hyperparameter Analysis for Q-HAS Training
===========================================

Loads an Optuna study from the campaign journal (or a legacy SQLite file) and
generates diagnostic plots:

  Section 1 — Optuna built-ins:
      parameter importance (fANOVA), contour, slice, parallel coords, history

  Section 2 — Convergence:
      trial-by-trial scores, running best, pruned trial markers

  Section 3 — 2D landscape:
      interpolated score heatmaps for every parameter pair

  Section 4 — Decomposed score (requires user_attrs from pipeline):
      Pareto front, per-field sensitivity, score decomposition,
      field-importance correlation heatmap

Usage:
    python analyze_hyperparams.py --journal-path ../results/hyperparams/reoptimisation/journal/q_has_v2_phase1.log --study-name q_has_v2_phase1
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import optuna

# Silence Optuna's own logging during analysis
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Optuna's matplotlib visualization module (no plotly dependency)
from optuna.visualization.matplotlib import (
    plot_param_importances,
    plot_contour,
    plot_slice,
    plot_parallel_coordinate,
    plot_optimization_history,
)


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────

def _loaded(study, study_name):
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value < float("inf")
    ]
    print(f"Loaded study '{study_name}': "
          f"{len(study.trials)} total, {len(completed)} completed (finite)")
    return study, completed


def load_study(db_path, study_name):
    """Load a legacy SQLite study."""
    return _loaded(optuna.load_study(
        study_name=study_name, storage=f"sqlite:///{db_path}"), study_name)


def load_journal(journal_path, study_name):
    """Load the journal produced by the rented-machine campaign."""
    from optuna.storages.journal import (JournalFileBackend,
                                         JournalFileOpenLock)
    storage = optuna.storages.JournalStorage(JournalFileBackend(
        journal_path, lock_obj=JournalFileOpenLock(journal_path)))
    return _loaded(
        optuna.load_study(study_name=study_name, storage=storage), study_name)


def get_param_names(completed):
    if not completed:
        return []
    return list(completed[0].params.keys())


def has_decomposed_data(completed):
    if not completed:
        return False
    # Check multiple trials in case early ones lack attrs
    return any("phys_score" in t.user_attrs for t in completed[:10])


ALL_SCENARIO_KEYS = [
    # v2 keys (without _predict suffix)
    "kh", "vortex", "tearing", "coalescence",
    # v2 keys (with _predict suffix)
    "kh_predict", "vortex_predict", "tearing_predict", "coalescence_predict",

    "ot", "rotor",
    "ot_predict", "rotor_predict",
    "gt", "gt_predict",
]
SCENARIO_LABELS = {
    "kh": "Kelvin-Helmholtz",
    "vortex": "Lamb-Oseen Vortex",
    "tearing": "Harris Tearing",
    "coalescence": "Island Coalescence",
    "kh_predict": "Kelvin-Helmholtz",
    "vortex_predict": "Lamb-Oseen Vortex",
    "tearing_predict": "Harris Tearing",
    "coalescence_predict": "Island Coalescence",
    "ot": "Orszag-Tang",
    "rotor": "MHD Rotor",
    "ot_predict": "Orszag-Tang",
    "rotor_predict": "MHD Rotor",
    "gt": "Ghost Twisting",
}
SCENARIO_COLORS = {
    "kh": "tab:blue",
    "vortex": "tab:green",
    "tearing": "tab:orange",
    "coalescence": "tab:red",
    "kh_predict": "tab:blue",
    "vortex_predict": "tab:green",
    "tearing_predict": "tab:orange",
    "coalescence_predict": "tab:red",
    "ot": "tab:purple",
    "rotor": "tab:brown",
    "ot_predict": "tab:purple",
    "rotor_predict": "tab:brown",
    "gt": "tab:pink",
}


def _detect_scenario_keys(completed):
    """Rend les cles de scenario REELLEMENT presentes dans `user_attrs`.

    Verifie chaque cle individuellement plutot que de rendre toute sa
    famille (les sept `_predict`, ou les sept non-`_predict`) des qu'une
    seule y est vue : une etude ne couvre pas toujours les sept scenarios
    d'une famille, et un appelant qui ferait confiance a la liste
    obtiendrait un `KeyError`, ou pire une moyenne polluee de `NaN` sur des
    cles inventees.

    Le comportement doit rester identique a la copie dans
    `recompute_lambda_scores._detect_scenario_keys`.
    """
    trouvees = set()
    for t in completed[:10]:
        for key in ALL_SCENARIO_KEYS:
            if f"loss_{key}" in t.user_attrs:
                trouvees.add(key)
    # Ordre canonique d'ALL_SCENARIO_KEYS, comme dans recompute_lambda_scores.
    return [k for k in ALL_SCENARIO_KEYS if k in trouvees]


def has_scenario_data(completed):
    """Check if trials have per-scenario loss data (Phase 2 composite)."""
    return len(_detect_scenario_keys(completed)) > 0


def _find_available_keys(completed, candidates, prefix=""):
    """Find which keys exist in user_attrs across first few trials."""
    available = []
    for key in candidates:
        full_key = f"{prefix}{key}"
        if any(full_key in t.user_attrs for t in completed[:10]):
            available.append(key)
    return available


def _add_trend(ax, x_vals, y_vals, color="red", n_bins=15):
    """Binned median trend line.

    La derniere classe est FERMEE : les bornes viennent de
    `linspace(x.min(), x.max())`, donc le dernier bord EST `x.max()`. Avec
    un `<` strict, l'essai qui porte la plus grande valeur du parametre
    n'entrerait dans aucune classe — la tendance omettrait silencieusement
    le point extreme, celui qui decide justement du sens de la pente au
    bord du domaine echantillonne.
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


def _pareto_front(points):
    """Boolean mask of Pareto-optimal rows (minimize both dims)."""
    is_pareto = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if is_pareto[i]:
            dominated = np.all(points[i] <= points, axis=1) & np.any(points[i] < points, axis=1)
            is_pareto[dominated] = False
            is_pareto[i] = True
    return is_pareto


def _save(fig, output_dir, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


# ─────────────────────────────────────────────────────────
#  Section 1 — Optuna built-in plots
# ─────────────────────────────────────────────────────────

def plot_optuna_builtins(study, output_dir, param_names, full):
    print("\n=== Section 1: Optuna Built-in Plots ===")

    # 1. Parameter importance (fANOVA)
    try:
        ax = plot_param_importances(study)
        fig = ax.figure
        fig.set_size_inches(10, 6)
        fig.suptitle("Hyperparameter Importance (fANOVA)", fontsize=14)
        fig.tight_layout()
        _save(fig, output_dir, "01_param_importance.png")
    except Exception as e:
        print(f"  [SKIP] param importance: {e}")

    # 2. Optimization history
    try:
        ax = plot_optimization_history(study)
        fig = ax.figure
        fig.set_size_inches(12, 6)
        fig.suptitle("Optimization History", fontsize=14)
        fig.tight_layout()
        _save(fig, output_dir, "02_optimization_history.png")
    except Exception as e:
        print(f"  [SKIP] optimization history: {e}")

    # 3. Slice (1-D marginal)
    try:
        ax = plot_slice(study)
        fig = ax.figure if not isinstance(ax, np.ndarray) else ax.flat[0].figure
        fig.set_size_inches(5 * len(param_names), 5)
        fig.suptitle("1-D Marginal Effects", fontsize=14)
        fig.tight_layout()
        _save(fig, output_dir, "03_slice_plots.png")
    except Exception as e:
        print(f"  [SKIP] slice plots: {e}")

    # 4. Parallel coordinates
    try:
        ax = plot_parallel_coordinate(study)
        fig = ax.figure
        fig.set_size_inches(12, 6)
        fig.suptitle("Parallel Coordinates", fontsize=14)
        fig.tight_layout()
        _save(fig, output_dir, "04_parallel_coordinates.png")
    except Exception as e:
        print(f"  [SKIP] parallel coordinates: {e}")

    # 5. Contour plots (all pairs)
    if full and len(param_names) >= 2:
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        scores = [t.value for t in completed_trials]
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                p1, p2 = param_names[i], param_names[j]
                try:
                    ax = plot_contour(study, params=[p1, p2])
                    fig = ax.figure
                    fig.set_size_inches(8, 6)
                    fig.suptitle(f"Contour: {p1} vs {p2}", fontsize=14)
                    fig.tight_layout()
                    _save(fig, output_dir, f"05a_contour_{p1}_vs_{p2}.png")
                except Exception as e:
                    print(f"  [SKIP] contour {p1} vs {p2}: {e}")
    
                try:
                    x = np.array([t.params[p1] for t in completed_trials])
                    y = np.array([t.params[p2] for t in completed_trials])
                    z = np.array(scores)
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Create a denser grid for a smoother surface
                    xi = np.linspace(x.min(), x.max(), 100)
                    yi = np.linspace(y.min(), y.max(), 100)
                    Xi, Yi = np.meshgrid(xi, yi)
                    
                    # Interpolate Z values
                    Zi = griddata((x, y), z, (Xi, Yi), method="cubic")
                    
                    # Plot a solid, opaque surface with wireframe lines for depth
                    surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis_r', 
                                           edgecolor='black', linewidth=0.2, alpha=0.95, antialiased=True)
                    
                    ax.set_xlabel(p1, labelpad=10)
                    ax.set_ylabel(p2, labelpad=10)
                    ax.set_zlabel('Score', labelpad=10)
                    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label='Score')
                    
                    # View angle
                    ax.view_init(elev=35, azim=45)
                    
                    fig.suptitle(f"3D Surface Contour: {p1} vs {p2}", fontsize=14)
                    fig.tight_layout()
                    _save(fig, output_dir, f"05b_3D_contour_{p1}_vs_{p2}.png")
                except Exception as e:
                    print(f"  [SKIP] 3D contour {p1} vs {p2}: {e}")


# ─────────────────────────────────────────────────────────
#  Section 2 — Convergence analysis
# ─────────────────────────────────────────────────────────

def plot_convergence(study, completed, output_dir):
    print("\n=== Section 2: Convergence ===")

    trial_numbers = [t.number for t in completed]
    scores = [t.value for t in completed]

    running_best = []
    best_so_far = float("inf")
    for s in scores:
        best_so_far = min(best_so_far, s)
        running_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(trial_numbers, scores, s=15, alpha=0.5, c="steelblue", label="Trial score")
    ax.plot(trial_numbers, running_best, "r-", linewidth=2, label="Running best")

    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if pruned:
        prune_nums = [t.number for t in pruned]
        ymax = ax.get_ylim()[1]
        ax.scatter(prune_nums, [ymax * 0.98] * len(pruned),
                   s=8, alpha=0.3, c="gray", marker="|",
                   label=f"Pruned ({len(pruned)})")

    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("Combined Score", fontsize=12)
    ax.set_title(f"Convergence ({len(completed)} completed, {len(pruned)} pruned)",
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "06_convergence.png")


# ─────────────────────────────────────────────────────────
#  Section 3 — 2-D score landscape
# ─────────────────────────────────────────────────────────

def plot_2d_landscapes(completed, param_names, output_dir):
    print("\n=== Section 3: 2-D Score Landscapes ===")
    if len(param_names) < 2:
        print("  [SKIP] need >= 2 parameters")
        return

    scores = np.array([t.value for t in completed])

    for i in range(len(param_names)):
        for j in range(i + 1, len(param_names)):
            p1, p2 = param_names[i], param_names[j]
            x = np.array([t.params[p1] for t in completed])
            y = np.array([t.params[p2] for t in completed])

            fig, ax = plt.subplots(figsize=(9, 7))

            # Interpolated background
            xi = np.linspace(x.min(), x.max(), 60)
            yi = np.linspace(y.min(), y.max(), 60)
            Xi, Yi = np.meshgrid(xi, yi)
            try:
                Zi = griddata((x, y), scores, (Xi, Yi), method="cubic")
                cf = ax.contourf(Xi, Yi, Zi, levels=25, cmap="viridis_r", alpha=0.85)
                fig.colorbar(cf, ax=ax, label="Combined Score")
            except Exception as exc:
                # Le contour interpolé peut échouer (points colinéaires,
                # trop peu d'essais). On garde le nuage de points, mais on
                # le DIT : une figure amputée de sa surface ressemble
                # sinon à une figure normale.
                print(f"[FIGURE] contour interpole indisponible : "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)

            # Actual points
            ax.scatter(x, y, c=scores, cmap="viridis_r", s=25,
                       edgecolors="white", linewidth=0.5, zorder=5)

            # Best
            best_idx = np.argmin(scores)
            ax.scatter(x[best_idx], y[best_idx], s=200, c="red",
                       marker="*", zorder=6, label="Best")

            ax.set_xlabel(p1, fontsize=12)
            ax.set_ylabel(p2, fontsize=12)
            ax.set_title(f"Score Landscape: {p1} vs {p2}", fontsize=13)
            ax.legend()
            fig.tight_layout()
            _save(fig, output_dir, f"07a_landscape_{p1}_vs_{p2}.png")


            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Higher resolution grid for smoother terrain
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            Xi, Yi = np.meshgrid(xi, yi)
            
            try:
                Zi = griddata((x, y), scores, (Xi, Yi), method="cubic")
                # Solid surface (alpha=0.95) with a light wireframe (edgecolor)
                surf = ax.plot_surface(Xi, Yi, Zi, cmap="viridis_r", alpha=0.95, 
                                       edgecolor='gray', linewidth=0.3, antialiased=True)
                fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, aspect=10, label="Combined Score")
            except Exception as e:
                print(f"  [Warning] Surface interpolation failed for {p1} vs {p2}: {e}")

            # Plot actual points, but make them smaller so they don't block the surface
            ax.scatter(x, y, scores, c='black', s=10, alpha=0.8, zorder=5)

            # Highlight the best point clearly, floating slightly above to prevent clipping
            best_idx = np.argmin(scores)
            z_offset = (scores.max() - scores.min()) * 0.02 # Lift it 2% above the surface
            ax.scatter(x[best_idx], y[best_idx], scores[best_idx] + z_offset, 
                       s=150, c="red", marker="*", zorder=6, label="Best")

            ax.set_xlabel(p1, labelpad=10, fontsize=12)
            ax.set_ylabel(p2, labelpad=10, fontsize=12)
            ax.set_zlabel("Score", labelpad=10, fontsize=12)
            ax.set_title(f"3D Score Landscape: {p1} vs {p2}", fontsize=13)
            ax.legend()
            
            # Slightly higher elevation to see down into the "valleys"
            ax.view_init(elev=35, azim=50) 
            
            fig.tight_layout()
            _save(fig, output_dir, f"07b_3D_landscape_{p1}_vs_{p2}.png")
# ─────────────────────────────────────────────────────────
#  Section 4 — Decomposed score analysis (needs user_attrs)
# ─────────────────────────────────────────────────────────

def plot_pareto_front(completed, output_dir):
    """phys_score vs patch_ratio, colored by combined score."""
    phys  = np.array([t.user_attrs["phys_score"]  for t in completed])
    patch = np.array([t.user_attrs["patch_ratio"] for t in completed])
    comb  = np.array([t.value for t in completed])

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(patch, phys, c=comb, cmap="viridis_r",
                    s=40, alpha=0.7, edgecolors="k", linewidth=0.3)
    fig.colorbar(sc, ax=ax, label="Combined Score")

    # Best trial
    best_idx = np.argmin(comb)
    ax.scatter(patch[best_idx], phys[best_idx], s=250, c="red", marker="*",
               zorder=5, label=f"Best (score={comb[best_idx]:.4f})")

    # Pareto front line
    pts = np.column_stack([patch, phys])
    mask = _pareto_front(pts)
    pareto = pts[mask]
    pareto = pareto[pareto[:, 0].argsort()]
    if len(pareto) >= 2:
        ax.plot(pareto[:, 0], pareto[:, 1], "r--", linewidth=2, alpha=0.7,
                label="Pareto front")

    ax.set_xlabel("Patch Ratio (computational cost)", fontsize=12)
    ax.set_ylabel("Physics Score (L2 error)", fontsize=12)
    ax.set_title("Pareto Front: Accuracy vs Computational Cost", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "08_pareto_front.png")


def plot_score_decomposition(completed, param_names, output_dir):
    """Dual-axis: phys_score (blue) and patch_ratio (orange) vs each param."""
    n = len(param_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    phys  = np.array([t.user_attrs["phys_score"]  for t in completed])
    patch = np.array([t.user_attrs["patch_ratio"] for t in completed])

    for i, param in enumerate(param_names):
        vals = np.array([t.params[param] for t in completed])
        ax1 = axes[i]
        ax2 = ax1.twinx()

        ax1.scatter(vals, phys,  s=15, alpha=0.4, c="tab:blue",   label="phys_score")
        _add_trend(ax1, vals, phys, color="tab:blue")

        ax2.scatter(vals, patch, s=15, alpha=0.4, c="tab:orange", label="patch_ratio")
        _add_trend(ax2, vals, patch, color="tab:orange")

        ax1.set_xlabel(param, fontsize=12)
        ax1.set_ylabel("Physics Score", color="tab:blue", fontsize=10)
        ax2.set_ylabel("Patch Ratio",   color="tab:orange", fontsize=10)
        ax1.set_title(param, fontsize=12, fontweight="bold")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.2)

    fig.suptitle("Score Decomposition vs Hyperparameters", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir, "09_score_decomposition.png")


def plot_per_field_sensitivity(completed, param_names, output_dir):
    """Grid of scatter plots: per-field error vs each hyperparameter."""
    fields = ["vx", "vy", "Bx", "By", "Jz"]

    # Check which fields are available
    available = _find_available_keys(completed, fields, prefix="error_")
    if not available:
        print("  [SKIP] no per-field error data")
        return

    nrows, ncols = len(available), len(param_names)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for row, field in enumerate(available):
        errors = np.array([t.user_attrs[f"error_{field}"] for t in completed])
        for col, param in enumerate(param_names):
            ax = axes[row, col]
            vals = np.array([t.params[param] for t in completed])

            ax.scatter(vals, errors, s=12, alpha=0.4, c="steelblue")
            _add_trend(ax, vals, errors, color="red")

            if row == 0:
                ax.set_title(param, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"L2 err ({field})", fontsize=10)
            if row == nrows - 1:
                ax.set_xlabel(param, fontsize=10)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Per-Field Error Sensitivity", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, output_dir, "10_per_field_sensitivity.png")


def plot_field_correlation_heatmap(completed, param_names, output_dir):
    """
    Pearson correlation matrix: hyperparameters x field errors.
    Shows which physics each hyperparameter controls.
    """
    fields = ["vx", "vy", "Bx", "By", "Jz"]
    available = _find_available_keys(completed, fields, prefix="error_")
    if not available:
        return

    corr = np.zeros((len(available), len(param_names)))
    for j, param in enumerate(param_names):
        vals = np.array([t.params[param] for t in completed])
        for i, field in enumerate(available):
            errors = np.array([t.user_attrs[f"error_{field}"] for t in completed])
            if np.std(vals) > 0 and np.std(errors) > 0:
                corr[i, j] = np.corrcoef(vals, errors)[0, 1]

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(param_names)), 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, fontsize=11, rotation=30, ha="right")
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels(available, fontsize=11)

    for i in range(len(available)):
        for j in range(len(param_names)):
            color = "white" if abs(corr[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Hyperparameter \u2013 Field Error Correlation\n"
                 "(+1 = higher param \u2192 higher error)", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir, "11_field_correlation_heatmap.png")


#: Les noms sous lesquels le seuil de raffinement a réellement été
#: échantillonné. `train_hyperparams.make_classical_composite_objective`
#: appelle `trial.suggest_float("threshold_amr", ...)` ; `"threshold"` tout
#: court n'apparaît dans aucune base du dépôt ni dans aucune ligne de
#: `src/`, mais reste le nom qu'une fonction en aval exigeait.
THRESHOLD_PARAM_NAMES = ("threshold_amr", "threshold")


def _threshold_param_name(trial):
    """Le nom sous lequel CET essai porte son seuil, ou None."""
    for name in THRESHOLD_PARAM_NAMES:
        if name in trial.params:
            return name
    return None


def _decomposed_series(completed):
    """`(phys, patch)` par essai, quel que soit le schéma d'attributs.

    Deux écrivains, deux schémas, et cette fonction ne lisait que le
    premier :

      * `pipeline.py` (objectif mono-scénario) écrit `phys_score` et
        `patch_ratio` ;
      * `train_hyperparams._run_one_scenario` — le chemin de la campagne
        déployée — écrit `phys_<scenario>` et `patch_<scenario>`, un par
        scénario, et **jamais** les deux clés globales.

    L'agrégation composite est la MOYENNE des scénarios parce que c'est
    celle que la perte elle-même applique (`_composite_loop` rend
    `total / len(scenario_list)`) : on mesure avec l'opérateur qui a
    construit la grandeur, pas avec un autre.

    Rend `(None, None, None)` si aucun des deux schémas n'est présent —
    c'est l'appelant qui le dit, il ne se tait pas. Le troisième élément
    nomme la provenance, pour qu'une figure ne puisse pas laisser croire
    qu'un seul scénario a été mesuré là où quatre ont été moyennés.
    """
    if not completed:
        return None, None, None

    if all("phys_score" in t.user_attrs and "patch_ratio" in t.user_attrs
           for t in completed):
        phys  = np.array([t.user_attrs["phys_score"]  for t in completed])
        patch = np.array([t.user_attrs["patch_ratio"] for t in completed])
        return phys, patch, "single run"

    keys = _find_available_keys(completed, ALL_SCENARIO_KEYS, prefix="phys_")
    keys = [k for k in keys
            if all(f"phys_{k}" in t.user_attrs and f"patch_{k}" in t.user_attrs
                   for t in completed)]
    if not keys:
        return None, None, None

    phys  = np.array([[t.user_attrs[f"phys_{k}"]  for k in keys]
                      for t in completed]).mean(axis=1)
    patch = np.array([[t.user_attrs[f"patch_{k}"] for k in keys]
                      for t in completed]).mean(axis=1)
    return phys, patch, "mean over " + ", ".join(keys)


def plot_threshold_operating_curve(completed, output_dir):
    """
    If a refinement threshold is among the optimized params, plot the
    threshold operating curve: phys_score and patch_ratio vs threshold.
    """
    name = _threshold_param_name(completed[0]) if completed else None
    if name is None:
        return

    completed = [t for t in completed if name in t.params]
    phys, patch, source = _decomposed_series(completed)
    if phys is None:
        # Un balayage vide doit crier : sans ce message, une étude dont le
        # seuil EST le paramètre optimisé rendait une analyse sans sa
        # figure de décision, indiscernable d'une analyse complète.
        print("[FIGURE] courbe de seuil indisponible : ni "
              "(phys_score, patch_ratio) ni (phys_<scenario>, "
              "patch_<scenario>) dans les user_attrs", file=sys.stderr)
        return

    thresholds = np.array([t.params[name] for t in completed])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.scatter(thresholds, phys,  s=20, alpha=0.4, c="tab:blue")
    ax2.scatter(thresholds, patch, s=20, alpha=0.4, c="tab:orange")
    _add_trend(ax1, thresholds, phys,  color="tab:blue",   n_bins=20)
    _add_trend(ax2, thresholds, patch, color="tab:orange", n_bins=20)

    ax1.set_xlabel(name, fontsize=13)
    ax1.set_ylabel(f"Physics Score (L2 error, {source})",
                   color="tab:blue", fontsize=12)
    ax2.set_ylabel(f"Patch Ratio (cost, {source})",
                   color="tab:orange", fontsize=12)
    ax1.set_title("Threshold Operating Curve\n"
                  "Higher threshold \u2192 fewer patches \u2192 cheaper but less accurate",
                  fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "12_threshold_operating_curve.png")


# ─────────────────────────────────────────────────────────
#  Summary report
# ─────────────────────────────────────────────────────────

def generate_summary(study, completed, param_names, output_dir):
    lines = []
    lines.append("=" * 60)
    lines.append("Q-HAS Hyperparameter Analysis Summary")
    lines.append("=" * 60)
    lines.append(f"Study          : {study.study_name}")
    lines.append(f"Total trials   : {len(study.trials)}")
    lines.append(f"Completed      : {len(completed)}")
    pruned = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.PRUNED])
    lines.append(f"Pruned         : {pruned}")
    lines.append(f"Parameters     : {param_names}")

    if completed:
        best = min(completed, key=lambda t: t.value)
        lines.append(f"\nBest trial     : #{best.number}")
        lines.append(f"Best score     : {best.value:.6f}")
        lines.append(f"Best params    : {best.params}")

        if has_decomposed_data(completed):
            lines.append(f"  phys_score   : {best.user_attrs['phys_score']:.6f}")
            lines.append(f"  patch_ratio  : {best.user_attrs['patch_ratio']:.6f}")
            for field in ["vx", "vy", "Bx", "By", "Jz"]:
                key = f"error_{field}"
                if key in best.user_attrs:
                    lines.append(f"  error_{field:>2s}    : {best.user_attrs[key]:.6f}")

        detected_keys = _detect_scenario_keys(completed)
        if detected_keys:
            lines.append(f"\n  Per-scenario breakdown (best trial):")
            for key in detected_keys:
                loss_key = f"loss_{key}"
                if loss_key in best.user_attrs:
                    label = SCENARIO_LABELS.get(key, key)
                    loss_val = best.user_attrs[loss_key]
                    phys_key = f"phys_{key}"
                    patch_key = f"patch_{key}"
                    if phys_key in best.user_attrs:
                        lines.append(f"    {label:>20s}:  loss={loss_val:.6f}  "
                                     f"phys={best.user_attrs[phys_key]:.6f}  "
                                     f"patch={best.user_attrs[patch_key]:.4f}")
                        # Per-field errors for this scenario
                        field_parts = []
                        for field in ["vx", "vy", "Bx", "By", "Jz"]:
                            fkey = f"error_{field}_{key}"
                            if fkey in best.user_attrs:
                                field_parts.append(f"{field}={best.user_attrs[fkey]:.4f}")
                        if field_parts:
                            lines.append(f"{'':>26s}[{', '.join(field_parts)}]")
                    else:
                        lines.append(f"    {label:>20s}:  loss={loss_val:.6f}")

        lines.append(f"\nTop 5 trials:")
        for t in sorted(completed, key=lambda t: t.value)[:5]:
            lines.append(f"  #{t.number:>3d}  score={t.value:.6f}  {t.params}")
            if detected_keys:
                scenario_parts = []
                for key in detected_keys:
                    loss_key = f"loss_{key}"
                    if loss_key in t.user_attrs:
                        short = key.replace("_predict", "")
                        scenario_parts.append(f"{short}={t.user_attrs[loss_key]:.4f}")
                if scenario_parts:
                    lines.append(f"        [{', '.join(scenario_parts)}]")

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary)


# ─────────────────────────────────────────────────────────
#  Section 5 — Per-scenario analysis (Phase 2 composite)
# ─────────────────────────────────────────────────────────

def plot_scenario_breakdown_bar(completed, output_dir):
    """Stacked bar chart: per-scenario loss for top 10 trials."""
    top = sorted(completed, key=lambda t: t.value)[:10]
    available = _find_available_keys(top, ALL_SCENARIO_KEYS, prefix="loss_")
    if not available:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top))
    width = 0.8

    bottom = np.zeros(len(top))
    for key in available:
        losses = np.array([t.user_attrs.get(f"loss_{key}", 0) for t in top])
        ax.bar(x, losses, width, bottom=bottom, label=SCENARIO_LABELS.get(key, key),
               color=SCENARIO_COLORS.get(key, None), alpha=0.85)
        bottom += losses

    ax.set_xticks(x)
    ax.set_xticklabels([f"#{t.number}" for t in top], fontsize=10)
    ax.set_xlabel("Trial", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Per-Scenario Loss Breakdown (Top 10 Trials)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    _save(fig, output_dir, "13_scenario_breakdown_bar.png")


def plot_scenario_sensitivity(completed, param_names, output_dir):
    """Per-scenario loss vs each hyperparameter (scatter + trend)."""
    available = _find_available_keys(completed, ALL_SCENARIO_KEYS, prefix="loss_")
    if not available:
        return

    # Filter to trials that have scenario data
    completed = [t for t in completed if f"loss_{available[0]}" in t.user_attrs]

    nrows, ncols = len(available), len(param_names)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for row, key in enumerate(available):
        losses = np.array([t.user_attrs[f"loss_{key}"] for t in completed])
        for col, param in enumerate(param_names):
            ax = axes[row, col]
            vals = np.array([t.params[param] for t in completed])

            ax.scatter(vals, losses, s=12, alpha=0.4,
                       c=SCENARIO_COLORS.get(key, "steelblue"))
            _add_trend(ax, vals, losses, color="red")

            if row == 0:
                ax.set_title(param, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Loss ({SCENARIO_LABELS.get(key, key)[:8]})", fontsize=9)
            if row == nrows - 1:
                ax.set_xlabel(param, fontsize=10)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Per-Scenario Loss Sensitivity", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, output_dir, "14_scenario_sensitivity.png")


def plot_scenario_correlation_heatmap(completed, param_names, output_dir):
    """Pearson correlation: hyperparameters x scenario losses."""
    available = _find_available_keys(completed, ALL_SCENARIO_KEYS, prefix="loss_")
    if not available:
        return

    completed = [t for t in completed if f"loss_{available[0]}" in t.user_attrs]

    corr = np.zeros((len(available), len(param_names)))
    for j, param in enumerate(param_names):
        vals = np.array([t.params[param] for t in completed])
        for i, key in enumerate(available):
            losses = np.array([t.user_attrs[f"loss_{key}"] for t in completed])
            if np.std(vals) > 0 and np.std(losses) > 0:
                corr[i, j] = np.corrcoef(vals, losses)[0, 1]

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(param_names)), 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, fontsize=11, rotation=30, ha="right")
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels([SCENARIO_LABELS.get(k, k) for k in available], fontsize=11)

    for i in range(len(available)):
        for j in range(len(param_names)):
            color = "white" if abs(corr[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Hyperparameter \u2013 Scenario Loss Correlation\n"
                 "(+1 = higher param \u2192 higher loss)", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir, "15_scenario_correlation_heatmap.png")


def plot_scenario_pairwise(completed, output_dir):
    """Scatter plots of scenario losses against each other (detect trade-offs)."""
    available = _find_available_keys(completed, ALL_SCENARIO_KEYS, prefix="loss_")
    if len(available) < 2:
        return
    completed = [t for t in completed if f"loss_{available[0]}" in t.user_attrs]
    if len(available) < 2:
        return

    from itertools import combinations
    pairs = list(combinations(available, 2))
    ncols = min(3, len(pairs))
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    comb = np.array([t.value for t in completed])

    for idx, (k1, k2) in enumerate(pairs):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        x = np.array([t.user_attrs[f"loss_{k1}"] for t in completed])
        y = np.array([t.user_attrs[f"loss_{k2}"] for t in completed])

        sc = ax.scatter(x, y, c=comb, cmap="viridis_r", s=25, alpha=0.7,
                        edgecolors="k", linewidth=0.3)

        # Best trial
        best_idx = np.argmin(comb)
        ax.scatter(x[best_idx], y[best_idx], s=200, c="red", marker="*", zorder=5)

        label1 = SCENARIO_LABELS.get(k1, k1).split()[0]  # Short name
        label2 = SCENARIO_LABELS.get(k2, k2).split()[0]
        ax.set_xlabel(f"Loss {label1}", fontsize=11)
        ax.set_ylabel(f"Loss {label2}", fontsize=11)
        ax.grid(True, alpha=0.2)

        # Correlation annotation
        if np.std(x) > 0 and np.std(y) > 0:
            r = np.corrcoef(x, y)[0, 1]
            ax.set_title(f"{label1} vs {label2}  (r={r:.2f})", fontsize=12)
        else:
            ax.set_title(f"{label1} vs {label2}", fontsize=12)

    # Hide unused axes
    for idx in range(len(pairs), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Scenario Loss Trade-offs (color = composite score)", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir, "16_scenario_pairwise.png")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Q-HAS Hyperparameter Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    storage = parser.add_mutually_exclusive_group(required=True)
    storage.add_argument("--journal-path",
                         help="Path to the Optuna campaign journal")
    storage.add_argument("--db-path",
                         help="Path to a legacy Optuna SQLite database")
    parser.add_argument("--study-name", required=True,
                        help="Optuna study name inside the database")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for plots (default: next to .db)")
    parser.add_argument("--full", action="store_true", help="all graphs built")
    parser.add_argument("--show", action="store_true",
                        help="Also display plots interactively (plt.show)")
    args = parser.parse_args()

    if args.output_dir is None:
        storage_path = args.journal_path or args.db_path
        args.output_dir = os.path.join(
            os.path.dirname(storage_path),
            f"analysis_{args.study_name}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Switch to interactive backend if --show
    if args.show:
        matplotlib.use("TkAgg")

    # Ce `try` ne couvre que le CHARGEMENT de l'étude ; l'échec sort en
    # code 1. Une exception levée par l'une des treize fonctions de trace
    # plus bas (clé d'attribut absente, scénario manquant) ne doit pas
    # être confondue avec une étude introuvable — élargir ce bloc
    # masquerait sa vraie cause derrière un message de chargement.
    storage_path = args.journal_path or args.db_path
    try:
        loader = load_journal if args.journal_path else load_study
        study, completed = loader(storage_path, args.study_name)
    except Exception as e:
        print(f"[ERREUR] chargement de '{args.study_name}' depuis "
              f"{storage_path} : {e}", file=sys.stderr)
        sys.exit(1)

    param_names = get_param_names(completed)

    if not completed:
        print(f"[ERREUR] '{args.study_name}' ne contient aucun essai COMPLETE "
              f"a valeur finie — rien a analyser.", file=sys.stderr)
        sys.exit(1)

    # Summary
    generate_summary(study, completed, param_names, args.output_dir)

    # Section 1: Optuna built-in
    plot_optuna_builtins(study, args.output_dir, param_names, args.full)

    # Section 2: Convergence
    plot_convergence(study, completed, args.output_dir)

    # Section 3: 2D Landscapes
    if args.full:
        plot_2d_landscapes(completed, param_names, args.output_dir)

    # Section 4: Decomposed analysis (only with user_attrs)
    if has_decomposed_data(completed):
        print("\n=== Section 4: Decomposed Score Analysis ===")
        plot_pareto_front(completed, args.output_dir)
        plot_score_decomposition(completed, param_names, args.output_dir)
        plot_per_field_sensitivity(completed, param_names, args.output_dir)
        plot_field_correlation_heatmap(completed, param_names, args.output_dir)
    else:
        print("\n[INFO] No decomposed score data (phys_score, patch_ratio, per-field errors).")
        print("       Sections 1–3 are available from existing trials.")

    # Volontairement hors de la garde `has_decomposed_data` : celle-ci ne
    # teste que `phys_score`, que seul l'objectif mono-scénario de
    # `pipeline.py` écrit. La garder empêcherait la courbe de seuil de
    # sortir sur toute étude composite, y compris l'étude classique dont
    # le seuil EST le seul paramètre optimisé — la fonction porte ses
    # propres gardes.
    plot_threshold_operating_curve(completed, args.output_dir)

    # Section 5: Per-scenario analysis (Phase 2 composite)
    if has_scenario_data(completed):
        print("\n=== Section 5: Per-Scenario Analysis (Composite) ===")
        plot_scenario_breakdown_bar(completed, args.output_dir)
        plot_scenario_sensitivity(completed, param_names, args.output_dir)
        plot_scenario_correlation_heatmap(completed, param_names, args.output_dir)
        plot_scenario_pairwise(completed, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")

    if args.show:
        plt.show()



if __name__ == "__main__":
    main()
