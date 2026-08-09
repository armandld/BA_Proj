#!/usr/bin/env python3
"""
Phase 11D - Per-scenario learned Hamiltonian ("local-specialisation ceiling").

Phase 11 showed that a SINGLE learned mean-field H collapses from
F1 ~= 0.99 (random split) to F1 ~= 0.19 (LOSO). A natural follow-up
proposal -- if you accept the LOSO result -- is:

  "Fine: don't ship one universal H. Train one H per scenario and
   switch at runtime based on a scenario detector."

This is the LOCAL-SPECIALISATION CEILING. It is the best F1 you could
achieve if (i) you already knew which scenario you were simulating,
and (ii) you trained a dedicated learned mean-field H for that scenario.

This phase measures it:

  For each scenario s:
    (a) split s-snapshots into 70/30 train/val.
    (b) fit logistic regression on the 9-feature mean-field (phase 11c).
    (c) evaluate F1_s on held-out s-snapshots  -> specialisation ceiling.
    (d) evaluate F1_s_on_{s'} for every OTHER scenario s' != s
        -> transfer penalty (full confusion matrix).

Reporting:
  diagonal    (F1_s_on_s)       -> how good can you get per scenario?
  off-diag    (F1_s_on_s')      -> how bad is a misrouted classifier?
  ratio       (F1_s_on_s') /
              (F1_s'_on_s')      -> relative cost of misrouting

The 2nd matrix answers the practical question: "if my scenario
detector has an error rate p, what fraction of F1 do I lose per error?"

Expected narrative:
  diagonal F1s ~ 0.7-0.95 (each scenario IS learnable when you know
                           its identity)
  off-diagonal often  < classical baseline  (transfer is worse than
                                             not learning at all)

This is the cleanest way to show that the Q-HAS bottleneck is not
model capacity but cross-scenario feature drift.

Input:  results/dns_{sc}_Re{re}_N{N}.npz
        results/patches_{sc}_Re{re}_N{N}_dim{D}.npz
Output: results/specialisation_N{N}_dim{D}.npz

Usage:
  python study/phase11d_specialisation.py --dim 4
"""
import argparse, os, sys, time
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
from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N

from phase11_upper_bound import (
    FEATURE_NAMES, N_FEATS,
    extract_features_2d, best_threshold_f1, make_model, fit_eval,
)
from phase11c_learned_h import fit_learned_h, predict_h

from sklearn.metrics import f1_score


def gather_by_scenario(configs, dim, max_snaps):
    per_sc = {}
    for sc, re, dns_path, patches_path in configs:
        dns = np.load(dns_path)
        patches = np.load(patches_path)
        vx = dns["vx"].astype(np.float64); vy = dns["vy"].astype(np.float64)
        Bx = dns["Bx"].astype(np.float64); By = dns["By"].astype(np.float64)
        N = vx.shape[1]
        l2_all = patches["l2_errors"]; thr = float(patches["l2_threshold"])

        n_snaps = len(vx)
        step = max(1, n_snaps // max_snaps)
        idx = list(range(0, n_snaps, step))[:max_snaps]

        for si in idx:
            feats_2d, score = extract_features_2d(
                vx[si], vy[si], Bx[si], By[si], N, dim, re)
            per_sc.setdefault(sc, dict(X=[], Y=[], S=[]))
            per_sc[sc]["X"].append(feats_2d.reshape(-1, N_FEATS))
            per_sc[sc]["Y"].append((l2_all[si] >= thr).ravel().astype(int))
            per_sc[sc]["S"].append(score.ravel())
    return per_sc


def split_scenario(d, train_frac, seed):
    """Random snapshot split inside one scenario."""
    rng = np.random.default_rng(seed)
    n = len(d["X"])
    perm = rng.permutation(n)
    n_tr = max(1, int(train_frac * n))
    tr = perm[:n_tr]; va = perm[n_tr:]
    Xtr = np.concatenate([d["X"][i] for i in tr])
    Ytr = np.concatenate([d["Y"][i] for i in tr])
    Str = np.concatenate([d["S"][i] for i in tr])
    Xva = np.concatenate([d["X"][i] for i in va]) if len(va) else np.zeros(
        (0, N_FEATS))
    Yva = np.concatenate([d["Y"][i] for i in va]) if len(va) else np.zeros(
        0, dtype=int)
    Sva = np.concatenate([d["S"][i] for i in va]) if len(va) else np.zeros(0)
    return Xtr, Ytr, Str, Xva, Yva, Sva


def main():
    p = argparse.ArgumentParser(
        description="Phase 11D: per-scenario specialisation ceiling")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=40)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=("lr", "gbt"), default="lr",
                   help="lr = linear H (learned mean-field); "
                        "gbt = non-linear ceiling")
    args = p.parse_args()

    print("=" * 88)
    print("  Phase 11D: per-scenario local-specialisation ceiling")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}  "
          f"model={args.model}")
    print("=" * 88)
    print()

    # -- discover configs --
    configs = []
    for sc in args.scenario:
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(
                RESULTS_DIR, f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if os.path.exists(dp) and os.path.exists(pp):
                configs.append((sc, re, dp, pp))
    if not configs:
        print("no input."); return

    t0 = time.time()
    per_sc = gather_by_scenario(configs, args.dim, args.max_snaps)
    print(f"  built dataset in {time.time()-t0:.1f}s")
    for sc in per_sc:
        n_snaps = len(per_sc[sc]["X"])
        pr = np.concatenate(per_sc[sc]["Y"]).mean()
        print(f"    {sc:<18} snaps={n_snaps:>4}  pos_rate={pr:.3f}")
    scenarios = list(per_sc.keys())
    print()

    # -- 1. per-scenario split: train H_s on s_train, eval on s_val ---
    per_scene_models = {}
    per_scene_val = {}
    classical_val = {}
    print("  [1] within-scenario specialisation ceiling:")
    print(f"  {'scenario':<18} {'n_tr':>6} {'n_va':>6} "
          f"{'F1_class':>9} {'F1_spec':>9} {'delta':>9}")
    for sc in scenarios:
        Xtr, Ytr, Str, Xva, Yva, Sva = split_scenario(
            per_sc[sc], args.train_frac, args.seed)
        if len(Xva) == 0 or len(np.unique(Ytr)) < 2 or len(np.unique(Yva)) < 2:
            print(f"  {sc:<18} degenerate split (not enough variety)")
            continue
        # classical
        thr_cls, _ = best_threshold_f1(Str, Ytr)
        f1_cls = f1_score(Yva, (Sva > thr_cls).astype(int), zero_division=0)

        # learned H
        if args.model == "lr":
            m = fit_learned_h(Xtr, Ytr, seed=args.seed)
            h_tr = predict_h(m, Xtr); h_va = predict_h(m, Xva)
            thr_h, _ = best_threshold_f1(
                h_tr, Ytr, grid=np.linspace(h_tr.min(), h_tr.max(), 201))
            f1_spec = f1_score(Yva, (h_va > thr_h).astype(int),
                                zero_division=0)
            per_scene_models[sc] = dict(
                kind="lr", model=m, thr=thr_h, thr_cls=thr_cls)
        else:
            model = make_model("gbt", args.seed).fit(Xtr, Ytr)
            p_tr = model.predict_proba(Xtr)[:, 1]
            p_va = model.predict_proba(Xva)[:, 1]
            thr_h, _ = best_threshold_f1(p_tr, Ytr,
                                          grid=np.linspace(0.05, 0.95, 91))
            f1_spec = f1_score(Yva, (p_va > thr_h).astype(int),
                                zero_division=0)
            per_scene_models[sc] = dict(
                kind="gbt", model=model, thr=thr_h, thr_cls=thr_cls)

        per_scene_val[sc] = dict(Xva=Xva, Yva=Yva, Sva=Sva, f1_spec=f1_spec)
        classical_val[sc] = f1_cls

        print(f"  {sc:<18} {len(Xtr):>6} {len(Xva):>6} "
              f"{f1_cls:>9.3f} {f1_spec:>9.3f} "
              f"{f1_spec - f1_cls:>+9.3f}")

    # -- 2. cross-application: H_s applied to val-split of s' --
    print()
    print("  [2] transfer matrix (row = model trained on, col = eval on):")
    sc_list = list(per_scene_val.keys())
    T = np.full((len(sc_list), len(sc_list)), np.nan)
    for i, train_sc in enumerate(sc_list):
        if train_sc not in per_scene_models:
            continue
        m = per_scene_models[train_sc]
        for j, eval_sc in enumerate(sc_list):
            v = per_scene_val[eval_sc]
            Xv, Yv = v["Xva"], v["Yva"]
            if m["kind"] == "lr":
                h_v = predict_h(m["model"], Xv)
                pred = (h_v > m["thr"]).astype(int)
            else:
                p = m["model"].predict_proba(Xv)[:, 1]
                pred = (p > m["thr"]).astype(int)
            T[i, j] = f1_score(Yv, pred, zero_division=0)

    # pretty print
    head = "  " + " " * 20 + " ".join(f"{s[:14]:>14}" for s in sc_list)
    print(head)
    for i, train_sc in enumerate(sc_list):
        row = "  train={:<14}".format(train_sc[:14]) + " "
        for j in range(len(sc_list)):
            cell = T[i, j]
            marker = "*" if i == j else " "
            row += f"{marker}{cell:>13.3f}"
        print(row)

    # -- 3. "misrouted classifier" cost --
    print()
    print("  [3] misrouting cost (avg off-diagonal F1 -- "
          "what a random scenario detector yields):")
    print(f"  {'model':<18} {'diag F1':>9} {'avg off-diag':>14} "
          f"{'classical':>10} {'delta_off_cls':>14}")
    off_stats = []
    for i, sc in enumerate(sc_list):
        diag = T[i, i]
        others = np.delete(T[i, :], i)
        off = float(np.nanmean(others))
        fc = classical_val.get(sc, float("nan"))
        print(f"  {sc:<18} {diag:>9.3f} {off:>14.3f} {fc:>10.3f} "
              f"{off - fc:>+14.3f}")
        off_stats.append(dict(
            sc=sc, diag=float(diag), off=off, classical=float(fc),
            delta_off_vs_class=float(off - fc),
        ))

    avg_delta_off = float(np.mean([r["delta_off_vs_class"] for r in off_stats]))
    print()
    print("  INTERPRETATION:")
    if avg_delta_off < 0:
        print(f"  * avg(off-diagonal - classical) = "
              f"{avg_delta_off:+.3f} < 0  "
              f"==>  a misrouted learned H is WORSE than classical "
              f"baseline. The 'train-per-scenario and switch' "
              f"strategy requires a near-perfect scenario detector "
              f"to beat classical on average.")
    else:
        print(f"  * avg(off-diagonal - classical) = "
              f"{avg_delta_off:+.3f} >= 0  ==>  even a misrouted "
              f"learned H beats classical on average. Specialisation "
              f"is safe.")

    # -- save --
    out = os.path.join(RESULTS_DIR,
                       f"specialisation_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        scenarios=np.array(sc_list),
        f1_specialisation=np.array([per_scene_val[s]["f1_spec"]
                                     for s in sc_list]),
        f1_classical=np.array([classical_val[s] for s in sc_list]),
        transfer=T,
        off_diag_mean=np.array([r["off"] for r in off_stats]),
        delta_off_vs_classical=np.array(
            [r["delta_off_vs_class"] for r in off_stats]),
        model=args.model,
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 11D complete.")


if __name__ == "__main__":
    main()
