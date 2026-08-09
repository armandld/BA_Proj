#!/usr/bin/env python3
"""
V3 Task 1 - B5 (GBT sur score_classical seul) + selection avant gloutonne
sous LOSO (protocole v3, section 2, regle de decision 1).

Reutilise EXACTEMENT les folds LOSO de phase 11b : memes fichiers d'entree,
meme assemblage par scenario (`_gather_scenario`), meme ordre de
concatenation des scenarios d'entrainement, meme `make_model("gbt", seed)`
et meme `fit_eval` (seuil choisi sur train). La ligne "full-9" doit donc
reproduire le 0.189 publie (logs/Result_phase11b.txt).

Lignes produites :
  - classical  : baseline phase 11b (seuil optimal sur train, appli. val)
  - B5         : GBT restreint a la colonne `score_classical`
  - fwd-k      : selection avant gloutonne, k = 1..9 (a chaque etape on
                 ajoute la feature qui maximise le F1 LOSO moyen)
  - full-9     : GBT sur les 9 features (reproduction du B4 publie)

Les sous-ensembles de colonnes sont toujours tries en ordre canonique
(ordre de FEATURE_NAMES), si bien que l'etape fwd-9 et la ligne full-9
sont le meme fit.

Sortie : results/t1_feature_selection_N{N}_dim{D}.npz
         (inclut hash git + arguments CLI complets, cf. garde-fous v3)

Usage :
  python study/v3/t1_feature_selection.py --N 256 --dim 4
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
from sklearn.metrics import f1_score

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

# NB : les imports lourds (config, phase11_upper_bound, phase11b_loso)
# sont faits paresseusement dans main() / les helpers, pour que les
# fonctions pures restent testables sans la pile qiskit.


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def select_columns(X, feat_idx):
    """Sous-ensemble de colonnes en ordre canonique (indices tries)."""
    return X[:, sorted(feat_idx)]


def _default_fit_tools():
    from phase11_upper_bound import make_model, fit_eval
    return (lambda seed: make_model("gbt", seed)), fit_eval


def loso_f1_subset(data, scenarios, feat_idx, seed,
                   model_factory=None, fit_fn=None):
    """F1 par fold LOSO pour un sous-ensemble de features.

    data : {scenario: {"X_site": (n, F), "Y": (n,)}}
    Retourne {scenario_tenu: f1}. L'ordre de concatenation des scenarios
    d'entrainement suit `scenarios` (= l'ordre de phase 11b).
    """
    if model_factory is None or fit_fn is None:
        d_factory, d_fit = _default_fit_tools()
        model_factory = model_factory or d_factory
        fit_fn = fit_fn or d_fit
    per_fold = {}
    for held in scenarios:
        Xtr = np.concatenate([select_columns(data[sc]["X_site"], feat_idx)
                              for sc in scenarios if sc != held])
        Ytr = np.concatenate([data[sc]["Y"]
                              for sc in scenarios if sc != held])
        Xva = select_columns(data[held]["X_site"], feat_idx)
        Yva = data[held]["Y"]
        r = fit_fn(model_factory(seed), Xtr, Ytr, Xva, Yva)
        per_fold[held] = float(r["f1"])
    return per_fold


def classical_loso_f1(data, scenarios, thr_fn=None):
    """Baseline classique de phase 11b : seuil F1-optimal sur train,
    applique au scenario tenu. data[sc]["S"] = score classique."""
    if thr_fn is None:
        from phase11_upper_bound import best_threshold_f1
        thr_fn = best_threshold_f1
    per_fold = {}
    for held in scenarios:
        Str = np.concatenate([data[sc]["S"]
                              for sc in scenarios if sc != held])
        Ytr = np.concatenate([data[sc]["Y"]
                              for sc in scenarios if sc != held])
        thr, _ = thr_fn(Str, Ytr)
        pred = (data[held]["S"] > thr).astype(int)
        per_fold[held] = float(f1_score(data[held]["Y"], pred,
                                        zero_division=0))
    return per_fold


def forward_selection(data, scenarios, n_feats, seed,
                      model_factory=None, fit_fn=None, verbose=False):
    """Selection avant gloutonne sous LOSO.

    Depart : ensemble vide. A chaque etape, on evalue chaque feature
    restante (F1 LOSO moyen du sous-ensemble courant + candidate) et on
    ajoute l'argmax. S'arrete quand les n_feats sont incluses.

    Retourne une liste d'etapes :
      {"added": idx, "selected": [idx...], "per_fold": {...}, "mean": m}
    """
    remaining = list(range(n_feats))
    selected = []
    path = []
    while remaining:
        best = None
        for cand in remaining:
            trial = selected + [cand]
            per_fold = loso_f1_subset(data, scenarios, trial, seed,
                                      model_factory, fit_fn)
            mean = float(np.mean(list(per_fold.values())))
            if best is None or mean > best[1]:
                best = (cand, mean, per_fold)
            if verbose:
                print(f"      cand +[{cand}] -> mean F1 = {mean:.3f}")
        cand, mean, per_fold = best
        selected.append(cand)
        remaining.remove(cand)
        path.append(dict(added=cand, selected=list(selected),
                         per_fold=per_fold, mean=mean))
        if verbose:
            print(f"    step {len(selected)}: + feature {cand}  "
                  f"mean F1 = {mean:.3f}")
    return path


def _print_row(label, per_fold, scenarios):
    vals = [per_fold[sc] for sc in scenarios]
    cells = " ".join(f"{v:>8.3f}" for v in vals)
    print(f"  {label:<34} {cells}  {np.mean(vals):>7.3f}")


def main():
    p = argparse.ArgumentParser(
        description="V3 Task 1: B5 score-only GBT + forward selection (LOSO)")
    # imports lourds ici seulement
    from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N
    from phase11_upper_bound import FEATURE_NAMES, N_FEATS
    from phase11b_loso import _gather_scenario

    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print("=" * 88)
    print("  V3 Task 1: B5 (score-only GBT) + greedy forward selection under LOSO")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}  seed={args.seed}")
    print("  Folds:", ", ".join(args.scenario))
    print("=" * 88)
    print()

    # ---- memes entrees que phase 11b ----
    by_scene = {}
    for sc in args.scenario:
        rows = []
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(RESULTS_DIR,
                              f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if os.path.exists(dp) and os.path.exists(pp):
                rows.append((re, dp, pp))
        if rows:
            by_scene[sc] = rows
    if len(by_scene) < 2:
        print("need >=2 scenarios with data to run LOSO."); return

    print("  building per-scenario feature matrices...")
    t0 = time.time()
    data = {}
    for sc, rows in by_scene.items():
        Xs, _, Y, S = _gather_scenario(rows, args.dim, args.max_snaps)
        data[sc] = dict(X_site=Xs, Y=Y, S=S)
        print(f"    {sc:<18} cells={len(Y):>6}  pos_rate={Y.mean():.3f}")
    print(f"  done in {time.time() - t0:.1f}s\n")

    scenarios = list(by_scene.keys())
    score_idx = FEATURE_NAMES.index("score_classical")

    # ---- lignes de reference ----
    cls_pf = classical_loso_f1(data, scenarios)
    b5_pf = loso_f1_subset(data, scenarios, [score_idx], args.seed)
    full_pf = loso_f1_subset(data, scenarios, list(range(N_FEATS)), args.seed)

    # ---- selection avant ----
    print("  greedy forward selection (45 subsets x "
          f"{len(scenarios)} folds)...")
    t0 = time.time()
    path = forward_selection(data, scenarios, N_FEATS, args.seed,
                             verbose=args.verbose)
    print(f"  done in {time.time() - t0:.1f}s\n")

    # ---- table ----
    head_cells = " ".join(f"{sc[:8]:>8}" for sc in scenarios)
    print(f"  {'feature set':<34} {head_cells}  {'mean':>7}")
    print("  " + "-" * (34 + 9 * len(scenarios) + 9))
    _print_row("classical (thr on train)", cls_pf, scenarios)
    _print_row("B5: score_classical only (GBT)", b5_pf, scenarios)
    for k, step in enumerate(path, start=1):
        name = FEATURE_NAMES[step["added"]]
        _print_row(f"fwd-{k}: +{name}", step["per_fold"], scenarios)
    _print_row("full-9 (B4, phase 11b repro)", full_pf, scenarios)
    print("  " + "-" * (34 + 9 * len(scenarios) + 9))

    cls_m = float(np.mean([cls_pf[sc] for sc in scenarios]))
    b5_m = float(np.mean([b5_pf[sc] for sc in scenarios]))
    full_m = float(np.mean([full_pf[sc] for sc in scenarios]))
    best_step = max(path, key=lambda s: s["mean"])

    print(f"\n  classical mean = {cls_m:.3f}   B5 mean = {b5_m:.3f}   "
          f"full-9 mean = {full_m:.3f}")
    print(f"  delta(B5 - classical)     = {b5_m - cls_m:+.3f}")
    print(f"  delta(full-9 - classical) = {full_m - cls_m:+.3f}")
    print(f"  best forward subset (k={len(best_step['selected'])}): "
          + ", ".join(FEATURE_NAMES[i]
                      for i in sorted(best_step["selected"]))
          + f"   mean F1 = {best_step['mean']:.3f}")
    print("\n  Section-2 decision rule inputs: compare B5 vs classical "
          "(B1/B2) and vs full-9 (B4); the branch is stated in "
          "study/v3/RESULTS.md.")

    # ---- sauvegarde (hash git + CLI, garde-fous v3) ----
    out = os.path.join(RESULTS_DIR,
                       f"t1_feature_selection_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        scenarios=np.array(scenarios),
        feature_names=np.array(FEATURE_NAMES),
        f1_classical=np.array([cls_pf[sc] for sc in scenarios]),
        f1_b5=np.array([b5_pf[sc] for sc in scenarios]),
        f1_full9=np.array([full_pf[sc] for sc in scenarios]),
        fwd_added=np.array([s["added"] for s in path]),
        fwd_mean=np.array([s["mean"] for s in path]),
        fwd_per_fold=np.array([[s["per_fold"][sc] for sc in scenarios]
                               for s in path]),
        seed=args.seed,
        git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nV3 Task 1 complete.")


if __name__ == "__main__":
    main()
