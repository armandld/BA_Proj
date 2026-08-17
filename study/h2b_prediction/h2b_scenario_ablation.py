#!/usr/bin/env python3
"""
Phase 11G - Scenario-identity feature ablation.

The phase 11B LOSO collapse (mean-field GBT 0.989 -> 0.191) is
attributed to "feature locality across scenarios": the 9 local
features do not span the cross-scenario direction. This phase tests
that claim mechanistically:

  Augment the 9-feature mean-field vector with a K-dim ONE-HOT
  scenario indicator (K = number of scenarios). Re-run two splits:

  (A) random-split-by-snapshot:
      Should match phase 11 (0.989) - the one-hot adds a constant
      per scenario inside each split, no information beyond the
      9 features. Used as a sanity baseline.

  (B) leave-one-scenario-out:
      With the held-out scenario's one-hot column being a value the
      model has never seen with non-zero training data, we have two
      sub-experiments:

      (B1) "blind LOSO with id" -- training rows have correct one-hot;
           validation rows have correct one-hot for the held-out
           scenario (a column that is all-1 in val and all-0 in train).
           If F1_site recovers to ~0.99, then **feature locality is the
           binding constraint** (the model just needs to know which
           scenario it is, then the per-site features are sufficient).
           If F1_site stays low, the bottleneck is something else
           (per-scenario non-linearity in the 9 features themselves).

      (B2) "scenario-fuzz LOSO" -- validation rows get a random other
           scenario's one-hot. Reproduces the phase 11D off-diagonal
           number; included as a sanity check.

This is the cleanest mechanistic test of the central paper claim.

Output: results/scenario_ablation_N{N}_dim{D}.npz

Usage:
  python study/phase11g_scenario_ablation.py --dim 4 --max-snaps 30
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

from h2b_ceiling_random_split import (
    FEATURE_NAMES, N_FEATS, extract_features_2d,
    make_model, fit_eval, best_threshold_f1,
)
from sklearn.metrics import f1_score


def gather_per_snapshot(scenarios, res, N, dim, max_snaps):
    Xs, Ys, Ss, tags = [], [], [], []
    for sc in scenarios:
        for re in res:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{N}.npz")
            pp = os.path.join(RESULTS_DIR, f"patches_{sc}_Re{re}_N{N}_dim{dim}.npz")
            if not (os.path.exists(dp) and os.path.exists(pp)):
                continue
            dns = np.load(dp); patches = np.load(pp)
            vx = dns["vx"].astype(np.float64); vy = dns["vy"].astype(np.float64)
            Bx = dns["Bx"].astype(np.float64); By = dns["By"].astype(np.float64)
            Nf = vx.shape[1]
            l2 = patches["l2_errors"]; thr = float(patches["l2_threshold"])
            n = len(vx); step = max(1, n // max_snaps)
            idx = list(range(0, n, step))[:max_snaps]
            for si in idx:
                f2d, sc_v = extract_features_2d(
                    vx[si], vy[si], Bx[si], By[si], Nf, dim, re)
                Xs.append(f2d.reshape(-1, N_FEATS))
                Ys.append((l2[si] >= thr).ravel().astype(int))
                Ss.append(sc_v.ravel())
                tags.append(sc)
    return Xs, Ys, Ss, tags


def attach_onehot(X_list, tags, scenario_index, K, override_tag=None):
    """Append a K-dim one-hot to each per-snapshot feature matrix.

    override_tag: if given, use that scenario for the one-hot (used
                  in the 'fuzz' experiment).
    """
    out = []
    for x, t in zip(X_list, tags):
        oh = np.zeros((x.shape[0], K), dtype=x.dtype)
        use = override_tag if override_tag is not None else t
        if use in scenario_index:
            oh[:, scenario_index[use]] = 1.0
        out.append(np.concatenate([x, oh], axis=1))
    return out


def main():
    p = argparse.ArgumentParser(
        description="Phase 11G: scenario-identity ablation")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.30)
    args = p.parse_args()

    print("=" * 88)
    print("  Phase 11G: scenario-identity ablation (one-hot augmentation)")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}")
    print("=" * 88)
    print()

    Xs, Ys, Ss, tags = gather_per_snapshot(
        args.scenario, args.re, args.N, args.dim, args.max_snaps)
    if not Xs:
        # D-75 : cette garde faisait `print(...); return` — code 0, aucun
        # artefact ecrit, donc indiscernable d'une campagne reussie (meme
        # famille que D-56 et D-74). Le detecteur AST de D-56 ne voyait que
        # la forme `if not <accumulateur nomme>:` ; celle-ci lui echappait.
        raise RuntimeError(
            "balayage vide : aucune configuration (scenario, Re) n'a d'artefact "
            "d'entree pour les arguments donnes. Le script sortait ici avec le "
            "code 0 et sans artefact (D-75).")
    scs = sorted(set(tags))
    sc_idx = {sc: i for i, sc in enumerate(scs)}
    K = len(scs)
    print(f"  scenarios: {scs}  (K={K})")
    print(f"  snapshots: {len(Xs)}\n")

    # one-hot augmented X (with correct tag per snapshot)
    Xs_aug = attach_onehot(Xs, tags, sc_idx, K)

    # ---- (A) random split, baseline (no one-hot) vs augmented ----
    rng = np.random.default_rng(args.seed)
    n_snap = len(Xs); perm = rng.permutation(n_snap)
    n_va = max(1, int(args.val_frac * n_snap))
    va = perm[:n_va]; tr = perm[n_va:]
    Xtr_b = np.concatenate([Xs[i]    for i in tr]); Ytr = np.concatenate([Ys[i] for i in tr])
    Xv_b  = np.concatenate([Xs[i]    for i in va]); Yv  = np.concatenate([Ys[i] for i in va])
    Str   = np.concatenate([Ss[i]    for i in tr]); Sv  = np.concatenate([Ss[i] for i in va])
    Xtr_a = np.concatenate([Xs_aug[i] for i in tr])
    Xv_a  = np.concatenate([Xs_aug[i] for i in va])

    thr_c, _ = best_threshold_f1(Str, Ytr)
    f1_class_rs = f1_score(Yv, (Sv > thr_c).astype(int), zero_division=0)
    r_b = fit_eval(make_model("gbt", args.seed), Xtr_b, Ytr, Xv_b, Yv)
    r_a = fit_eval(make_model("gbt", args.seed), Xtr_a, Ytr, Xv_a, Yv)
    print("  (A) RANDOM SPLIT")
    print(f"      F1 classical                 = {f1_class_rs:.3f}")
    print(f"      F1 mean-field GBT (9 feats)  = {r_b['f1']:.3f}")
    print(f"      F1 mean-field GBT (9+id)     = {r_a['f1']:.3f}   "
          f"(should match -- no novel info per scenario)")
    print()

    # ---- (B) LOSO ----
    by_sc = {sc: [i for i, t in enumerate(tags) if t == sc] for sc in scs}
    print("  (B) LOSO")
    print(f"  {'held':<18} {'F1_class':>9} {'F1_9feat':>9} "
          f"{'F1_9+id':>9} {'F1_9+fuzz':>10} {'delta(9+id - 9)':>16}")
    rows = []
    for held in scs:
        tr_idx = [i for sc, ix in by_sc.items() if sc != held for i in ix]
        va_idx = by_sc[held]

        Xtr_9  = np.concatenate([Xs[i] for i in tr_idx])
        Ytr    = np.concatenate([Ys[i] for i in tr_idx])
        Str    = np.concatenate([Ss[i] for i in tr_idx])
        Xv_9   = np.concatenate([Xs[i] for i in va_idx])
        Yv     = np.concatenate([Ys[i] for i in va_idx])
        Sv     = np.concatenate([Ss[i] for i in va_idx])
        Xtr_a  = np.concatenate([Xs_aug[i] for i in tr_idx])
        Xv_a   = np.concatenate([Xs_aug[i] for i in va_idx])

        # fuzz: validation rows get a random WRONG one-hot
        rng_f = np.random.default_rng(args.seed + 17 * sc_idx[held])
        wrongs = [s for s in scs if s != held]
        Xs_va_fuzz = []
        for i in va_idx:
            wrong_tag = wrongs[int(rng_f.integers(len(wrongs)))]
            oh = np.zeros((Xs[i].shape[0], K))
            oh[:, sc_idx[wrong_tag]] = 1.0
            Xs_va_fuzz.append(np.concatenate([Xs[i], oh], axis=1))
        Xv_fuzz = np.concatenate(Xs_va_fuzz)

        if len(np.unique(Ytr)) < 2 or len(np.unique(Yv)) < 2:
            rows.append(dict(held=held, f1_class=float("nan"),
                             f1_9=float("nan"), f1_9id=float("nan"),
                             f1_9fz=float("nan"),
                             f1_9fz_thr_on_val=float("nan"))); continue

        thr_c, _ = best_threshold_f1(Str, Ytr)
        f1_class = f1_score(Yv, (Sv > thr_c).astype(int), zero_division=0)
        r9   = fit_eval(make_model("gbt", args.seed), Xtr_9, Ytr, Xv_9, Yv)
        rid  = fit_eval(make_model("gbt", args.seed), Xtr_a, Ytr, Xv_a, Yv)
        # fuzz uses augmented model (trained with correct id); val rows get wrong id
        m_a = make_model("gbt", args.seed).fit(Xtr_a, Ytr)
        p_fuzz = m_a.predict_proba(Xv_fuzz)[:, 1]
        # D-82 : le seuil etait choisi sur `(p_fuzz, Yv)` — les labels de
        # VALIDATION — alors que les trois autres colonnes de cette meme
        # table (classique, 9 features, 9+id) passent par `fit_eval`, donc
        # par un seuil pris sur le train. La colonne « fuzz » etait la seule
        # a beneficier de ses propres labels de test, et c'est precisement
        # elle qui mesure la CHUTE quand l'identite de scenario est fausse :
        # un F1 gonfle sous-estime la chute. `rid` est le meme modele
        # (meme graine, memes donnees), son seuil vient de son train.
        thr_fz = rid["thr"]
        f1_fz = f1_score(Yv, (p_fuzz > thr_fz).astype(int), zero_division=0)
        # l'ancien nombre, garde pour que le biais reste mesurable
        _, f1_fz_thr_on_val = best_threshold_f1(
            p_fuzz, Yv, grid=np.linspace(0.05, 0.95, 91))

        rows.append(dict(held=held, f1_class=f1_class,
                         f1_9=r9["f1"], f1_9id=rid["f1"], f1_9fz=f1_fz,
                         f1_9fz_thr_on_val=f1_fz_thr_on_val))
        print(f"  {held:<18} {f1_class:>9.3f} {r9['f1']:>9.3f} "
              f"{rid['f1']:>9.3f} {f1_fz:>10.3f} "
              f"{(rid['f1']-r9['f1']):>+16.3f}")

    print()
    f9   = np.array([r["f1_9"]    for r in rows])
    fid  = np.array([r["f1_9id"]  for r in rows])
    ffz  = np.array([r["f1_9fz"]  for r in rows])
    ffz_v = np.array([r["f1_9fz_thr_on_val"] for r in rows])
    fcl  = np.array([r["f1_class"] for r in rows])
    print(f"  LOSO mean: classical {np.nanmean(fcl):.3f}, "
          f"9-feat {np.nanmean(f9):.3f}, 9+id {np.nanmean(fid):.3f}, "
          f"9+fuzz {np.nanmean(ffz):.3f}"
          f"   [fuzz au seuil de validation, biaise, D-82 : "
          f"{np.nanmean(ffz_v):.3f}]")

    print()
    print("  INTERPRETATION:")
    delta_id = np.nanmean(fid) - np.nanmean(f9)
    if delta_id > 0.30:
        print(f"  * delta(9+id - 9) = {delta_id:+.3f}  >>  0.")
        print("    Adding scenario identity LARGELY recovers the random-split")
        print("    ceiling. ==> The bottleneck IS feature locality across")
        print("    scenarios; the 9 per-site features are sufficient WITHIN a")
        print("    scenario but do not span the cross-scenario direction.")
        print("    This empirically confirms the central claim.")
    elif delta_id > 0.05:
        print(f"  * delta(9+id - 9) = {delta_id:+.3f}  >  0 modestly.")
        print("    Scenario identity helps but does not fully recover the")
        print("    random-split ceiling. ==> Feature locality is part of the")
        print("    bottleneck; per-scenario non-linearity in the 9 features")
        print("    themselves accounts for the rest.")
    else:
        print(f"  * delta(9+id - 9) = {delta_id:+.3f}  ~  0.")
        print("    Scenario identity does NOT help. The bottleneck is NOT")
        print("    feature locality but something more fundamental (e.g. the")
        print("    L2-hard label is not a per-snapshot function of the 9")
        print("    features at all). Re-investigate the labelling protocol.")

    out = os.path.join(RESULTS_DIR,
                       f"scenario_ablation_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        scenarios=np.array(scs),
        rs_class=f1_class_rs, rs_site9=r_b["f1"], rs_site9id=r_a["f1"],
        loso_held=np.array([r["held"] for r in rows]),
        loso_class=fcl, loso_site9=f9, loso_site9id=fid, loso_site9fuzz=ffz,
        # D-82 : l'ancien nombre, seuil pris sur les labels de validation.
        # Garde pour que le biais reste mesurable ; jamais compare aux
        # trois autres colonnes.
        loso_site9fuzz_thr_on_val=ffz_v,
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 11G complete.")


if __name__ == "__main__":
    main()
