#!/usr/bin/env python3
"""
V3 Task 1b - Courbe "cone d'information" : GBT a k sauts, k in {0,1,2,3}
(protocole v3, section 2 ; teste classiquement l'hypothese d'echelle
"plus de qubits = voisinage utile plus grand").

Generalisation de `stencil_features` (phase 11) a des voisinages k-hop
periodiques (decalages np.roll) :
  k=0 ->   9 features (cellule seule)
  k=1 ->  45 features (self + N/S/E/W : EXACTEMENT le stencil de phase 11,
           memes decalages, meme ordre de colonnes -> la ligne k=1 sous
           LOSO doit reproduire le 0.215 publie)
  k=2 -> 225 features (carre de Moore 5x5)
  k=3 -> 441 features (carre de Moore 7x7)
(comptes NOMINAUX : la boule de Chebyshev, sans egard a la periodicite de
la grille. D-88 : sur une grille periodique de cote `dim`, deux decalages
nominaux distincts peuvent s'enrouler sur le meme residu et produire deux
colonnes identiques des que 2k+1 >= dim. A dim=4 c'est deja le cas a k=2 :
le carre 5x5 couvre les 4 residus de chaque axe, donc la grille entiere ;
k=3 n'ajoute alors AUCUNE colonne distincte de plus que k=2 -- 225 et 441
se reduisent tous deux a 144 colonnes reelles. Voir `khop_distinct_footprint`
et la colonne `n_distinct` des tables ci-dessous.)

Deux evaluations par k :
  (a) split temporellement bloque (regle Task 4 : 60 % premiers snapshots
      de chaque trajectoire (scenario, Re) -> train, 40 % restants -> val)
  (b) folds LOSO de phase 11b (memes donnees, meme ordre de concatenation)

Garde-fou ratio echantillons/features : on rapporte n_train/n_features ;
si < 20, un fit supplementaire avec `max_features` plafonne a sqrt(F)/F
(regle racine classique) est rapporte et signale [FLAG].

Sortie : results/t1b_cone_curve_N{N}_dim{D}.npz
         (hash git + arguments CLI complets, cf. garde-fous v3)

Usage :
  python study/v3/t1b_cone_curve.py --N 256 --dim 4
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

from h2b_feature_selection import git_commit_hash, loso_f1_subset

K_VALUES = [0, 1, 2, 3]


# -------------------------------------------------------------------
# Voisinages k-hop (fonctions pures, testables sans qiskit)
# -------------------------------------------------------------------

def khop_offsets(k):
    """Decalages source (dy, dx) du voisinage k-hop.

    k=1 reproduit EXACTEMENT l'ordre de `stencil_features` de phase 11 :
    [self, N, S, E, W] avec N = roll(-1, axis=0), etc. Pour k>=2 :
    carre de Moore complet, self d'abord puis anneaux de Chebyshev
    croissants, ordre lexicographique dans chaque anneau.
    """
    if k == 0:
        return [(0, 0)]
    if k == 1:
        return [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    offs = [(0, 0)]
    for r in range(1, k + 1):
        offs += sorted((dy, dx)
                       for dy in range(-r, r + 1)
                       for dx in range(-r, r + 1)
                       if max(abs(dy), abs(dx)) == r)
    return offs


def khop_features(feats_2d, k):
    """(dim, dim, F) -> (dim*dim, n_offsets*F), periodique.

    Pour k=1, bit-identique a `stencil_features(feats_2d)` de phase 11
    (memes np.roll, meme concatenation, meme reshape).
    """
    parts = [np.roll(np.roll(feats_2d, -dy, axis=0), -dx, axis=1)
             for dy, dx in khop_offsets(k)]
    cat = np.concatenate(parts, axis=-1)
    return cat.reshape(-1, cat.shape[-1])


def khop_distinct_footprint(k, dim):
    """Nombre de decalages REELLEMENT distincts du voisinage k-hop sur une
    grille periodique de cote `dim` (D-88).

    `khop_offsets(k)` enumere les decalages nominaux d'une boule de
    Chebyshev sans savoir que `khop_features` les applique par `np.roll`
    PERIODIQUE (mod `dim`). Des que la largeur du carre `2k+1` atteint ou
    depasse `dim`, deux decalages nominaux distincts retombent sur le meme
    residu `(dy % dim, dx % dim)` -> deux colonnes bit-a-bit identiques
    dans `khop_features`. `len(khop_offsets(k))` n'est alors plus le
    nombre de voisins reellement vus : c'est un compte de doublons.

    Mesure a dim=4 (D-88, meme construction que `khop_offsets`/
    `khop_features` - pas de reimplementation parallele) : k=2 (largeur 5)
    couvre deja les 4 residus de chaque axe, donc les 16 cellules de la
    grille entiere -> k=3 (largeur 7) n'ajoute AUCUN residu de plus. Les
    225 et 441 features nominales de k=2 et k=3 se reduisent toutes deux a
    144 = 16 * 9 colonnes distinctes ; verifie contre
    `np.unique(khop_features(...), axis=1).shape[1]`, qui rend le meme
    nombre (comparaison a l'operateur assorti, pas a une reconstruction).
    """
    return len({(dy % dim, dx % dim) for dy, dx in khop_offsets(k)})


def blocked_split_indices(n_snaps, train_frac=0.6):
    """Regle Task 4 : les premiers 60 % (ordre temporel) -> train,
    le reste -> val. Retourne (idx_train, idx_val), contigus."""
    n_tr = max(1, int(train_frac * n_snaps))
    if n_tr >= n_snaps:
        n_tr = n_snaps - 1
    return list(range(n_tr)), list(range(n_tr, n_snaps))


def capped_model_factory(n_features, seed, base_factory=None):
    """GBT de phase 11 avec max_features plafonne a sqrt(F)/F.
    Retourne None si la version de sklearn ne supporte pas max_features."""
    if base_factory is None:
        from h2b_ceiling_random_split import make_model
        base_factory = lambda s: make_model("gbt", s)
    m = base_factory(seed)
    frac = float(np.sqrt(n_features) / n_features)
    try:
        m.set_params(max_features=frac)
    except ValueError:
        return None
    return m


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

def _gather_feats_per_snapshot(by_scene, dim, max_snaps):
    """Comme phase11b._gather_scenario mais conserve les tenseurs
    (dim, dim, 9) par snapshot + l'ordre temporel par (scenario, Re).

    Retourne {sc: list of dict(re=, pos=, feats=, y=, s=)} dans l'ordre
    exact de phase 11b (boucle Re puis temps)."""
    from h2b_ceiling_random_split import extract_features_2d
    out = {}
    for sc, rows in by_scene.items():
        snaps = []
        for re, physics_seed, dns_path, patches_path in rows:
            dns = np.load(dns_path)
            patches = np.load(patches_path)
            vx_all = dns["vx"].astype(np.float64)
            vy_all = dns["vy"].astype(np.float64)
            Bx_all = dns["Bx"].astype(np.float64)
            By_all = dns["By"].astype(np.float64)
            N = vx_all.shape[1]
            l2_all = patches["l2_errors"]
            l2_thr = float(patches["l2_threshold"])

            n_snaps = len(vx_all)
            step = max(1, n_snaps // max_snaps)
            idx = list(range(0, n_snaps, step))[:max_snaps]

            for pos, si in enumerate(idx):
                feats_2d, score = extract_features_2d(
                    vx_all[si], vy_all[si], Bx_all[si], By_all[si],
                    N, dim, re,
                )
                snaps.append(dict(
                    re=re, physics_seed=physics_seed,
                    pos=pos, n_pos=len(idx), feats=feats_2d,
                    y=(l2_all[si] >= l2_thr).ravel().astype(int),
                    s=score.ravel(),
                ))
        out[sc] = snaps
    return out


def main():
    p = argparse.ArgumentParser(
        description="V3 Task 1b: information-cone curve (k-hop GBT)")
    from config import (
        DNS_N, PHYSICS_SEEDS, RESULTS_DIR, RE_VALUES, SCENARIOS,
    )
    from data_catalog import labelled_trajectory_paths
    from h2b_ceiling_random_split import make_model, fit_eval, best_threshold_f1

    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--phys-seed", nargs="+", type=int,
                   default=list(PHYSICS_SEEDS))
    p.add_argument("--train-frac", type=float, default=0.6,
                   help="fraction train du split bloque (regle Task 4)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  V3 Task 1b: information-cone curve, k-hop GBT, k in "
          f"{K_VALUES}")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}  "
          f"seed={args.seed}  train-frac={args.train_frac}")
    print("  Folds:", ", ".join(args.scenario))
    print("=" * 88)
    print()

    # ---- memes entrees que phase 11b ----
    by_scene = labelled_trajectory_paths(
        RESULTS_DIR, args.scenario, args.re, args.N, args.dim,
        args.phys_seed)
    if len(by_scene) < 2:
        raise RuntimeError("the information-cone LOSO requires two scenarios")

    print("  building per-snapshot feature tensors...")
    t0 = time.time()
    snaps = _gather_feats_per_snapshot(by_scene, args.dim, args.max_snaps)
    for sc in snaps:
        y = np.concatenate([d["y"] for d in snaps[sc]])
        print(f"    {sc:<18} snaps={len(snaps[sc]):>4}  cells={len(y):>6}  "
              f"pos_rate={y.mean():.3f}")
    print(f"  done in {time.time() - t0:.1f}s\n")

    scenarios = list(snaps.keys())
    gbt_factory = lambda s: make_model("gbt", s)

    # matrices k-hop par scenario (concatenation = ordre phase 11b)
    Xk = {k: {sc: np.concatenate([khop_features(d["feats"], k)
                                  for d in snaps[sc]])
              for sc in scenarios} for k in K_VALUES}
    Y = {sc: np.concatenate([d["y"] for d in snaps[sc]])
         for sc in scenarios}
    S = {sc: np.concatenate([d["s"] for d in snaps[sc]])
         for sc in scenarios}

    results = {"loso": {}, "blocked": {}}
    ratios = {"loso": {}, "blocked": {}}
    capped = {"loso": {}, "blocked": {}}

    # =========================== (b) LOSO ===========================
    print("  [LOSO] folds of phase 11b")
    n_cells_total = sum(len(Y[sc]) for sc in scenarios)
    for k in K_VALUES:
        nf = Xk[k][scenarios[0]].shape[1]
        data_k = {sc: dict(X_site=Xk[k][sc], Y=Y[sc]) for sc in scenarios}
        pf = loso_f1_subset(data_k, scenarios, list(range(nf)), args.seed)
        results["loso"][k] = pf
        n_tr_min = min(n_cells_total - len(Y[sc]) for sc in scenarios)
        ratios["loso"][k] = n_tr_min / nf
        if ratios["loso"][k] < 20:
            pf_cap = {}
            ok = True
            for held in scenarios:
                m = capped_model_factory(nf, args.seed, gbt_factory)
                if m is None:
                    ok = False; break
                Xtr = np.concatenate([Xk[k][sc] for sc in scenarios
                                      if sc != held])
                Ytr = np.concatenate([Y[sc] for sc in scenarios
                                      if sc != held])
                r = fit_eval(m, Xtr, Ytr, Xk[k][held], Y[held])
                pf_cap[held] = float(r["f1"])
            capped["loso"][k] = pf_cap if ok else None
        print(f"    k={k} done ({nf} feats)")

    # ====================== (a) split bloque =========================
    # train = premiers 60 % de chaque trajectoire (scenario, Re)
    print("\n  [blocked] first 60% of each (scenario, Re) trajectory -> train")

    def _blocked_concat(k_or_none, part):
        """part in {'tr','va'}; k_or_none=None -> (Y, S) scalaires."""
        chunks = []
        for sc in scenarios:
            by_cfg = {}
            for j, d in enumerate(snaps[sc]):
                by_cfg.setdefault(
                    (d["re"], d["physics_seed"]), []).append(j)
            for config in sorted(by_cfg):
                idx = by_cfg[config]
                tr, va = blocked_split_indices(len(idx), args.train_frac)
                use = tr if part == "tr" else va
                for u in use:
                    d = snaps[sc][idx[u]]
                    if k_or_none is None:
                        chunks.append((d["y"], d["s"]))
                    else:
                        chunks.append(khop_features(d["feats"], k_or_none))
        if k_or_none is None:
            return (np.concatenate([c[0] for c in chunks]),
                    np.concatenate([c[1] for c in chunks]))
        return np.concatenate(chunks)

    Ytr_b, Str_b = _blocked_concat(None, "tr")
    Yva_b, Sva_b = _blocked_concat(None, "va")
    print(f"    blocked split: {len(Ytr_b)} train cells, "
          f"{len(Yva_b)} val cells")

    for k in K_VALUES:
        Xtr = _blocked_concat(k, "tr")
        Xva = _blocked_concat(k, "va")
        nf = Xtr.shape[1]
        r = fit_eval(gbt_factory(args.seed), Xtr, Ytr_b, Xva, Yva_b)
        results["blocked"][k] = float(r["f1"])
        ratios["blocked"][k] = len(Ytr_b) / nf
        if ratios["blocked"][k] < 20:
            m = capped_model_factory(nf, args.seed, gbt_factory)
            if m is not None:
                rc = fit_eval(m, Xtr, Ytr_b, Xva, Yva_b)
                capped["blocked"][k] = float(rc["f1"])
            else:
                capped["blocked"][k] = None
        print(f"    k={k} done ({nf} feats)")

    # baseline classique (reference B2, agregation block_max de phase 11b)
    cls_loso = {}
    for held in scenarios:
        Str = np.concatenate([S[sc] for sc in scenarios if sc != held])
        Ytr = np.concatenate([Y[sc] for sc in scenarios if sc != held])
        thr, _ = best_threshold_f1(Str, Ytr)
        cls_loso[held] = float(f1_score(
            Y[held], (S[held] > thr).astype(int), zero_division=0))
    thr_b, _ = best_threshold_f1(Str_b, Ytr_b)
    cls_blocked = float(f1_score(Yva_b, (Sva_b > thr_b).astype(int),
                                 zero_division=0))

    # ============================ tables =============================
    prev_b = float(Yva_b.mean())
    print("\n  " + "=" * 84)
    print(f"  [blocked split]  (val prevalence={prev_b:.3f}; refine-all "
          f"floor F1={2 * prev_b / (1 + prev_b):.3f})")
    print(f"  {'k':>3} {'n_feats':>8} {'n_distinct':>10} {'n_tr/F':>8} "
          f"{'F1':>8} {'delta/hop':>10} {'capped':>8}")
    print(f"  {'cls':>3} {'-':>8} {'-':>10} {'-':>8} {cls_blocked:>8.3f} "
          f"{'-':>10} {'-':>8}")
    prev_f1 = None
    prev_footprint = None
    for k in K_VALUES:
        nf = len(khop_offsets(k)) * 9
        footprint = khop_distinct_footprint(k, args.dim)
        nf_distinct = footprint * 9
        f1 = results["blocked"][k]
        d = "-" if prev_f1 is None else f"{f1 - prev_f1:+.3f}"
        cap = capped["blocked"].get(k)
        cap_s = ("-" if k not in capped["blocked"]
                 else ("n/a" if cap is None else f"{cap:.3f}"))
        flag = "  [FLAG n_tr/F<20]" if ratios["blocked"][k] < 20 else ""
        # D-88 : `nf` (nominal) surcompte des lors que 2k+1 >= dim -- deux
        # decalages distincts s'enroulent sur le meme residu et rendent la
        # meme colonne. Ce drapeau dit quand k n'ajoute aucune information
        # de plus que k-1 sur CETTE grille : le "delta/hop" ne compare alors
        # plus deux voisinages de tailles differentes.
        if prev_footprint is not None and footprint == prev_footprint:
            flag += (f"  [FLAG k={k} n'agrandit pas le voisinage : meme "
                     f"empreinte que k={k - 1} sur grille dim={args.dim}]")
        print(f"  {k:>3} {nf:>8} {nf_distinct:>10} "
              f"{ratios['blocked'][k]:>8.1f} {f1:>8.3f} "
              f"{d:>10} {cap_s:>8}{flag}")
        prev_f1 = f1
        prev_footprint = footprint

    print("\n  [LOSO]  (per-fold prevalence=0.250; refine-all floor "
          "F1=0.400)")
    head = " ".join(f"{sc[:8]:>8}" for sc in scenarios)
    print(f"  {'k':>3} {'n_feats':>8} {'n_distinct':>10} {'n_tr/F':>8} "
          f"{head}  {'mean':>7} {'delta/hop':>10} {'capped':>8}")
    cls_cells = " ".join(f"{cls_loso[sc]:>8.3f}" for sc in scenarios)
    cls_m = float(np.mean(list(cls_loso.values())))
    print(f"  {'cls':>3} {'-':>8} {'-':>10} {'-':>8} {cls_cells}  "
          f"{cls_m:>7.3f} {'-':>10} {'-':>8}")
    prev_f1 = None
    prev_footprint = None
    means = {}
    for k in K_VALUES:
        nf = len(khop_offsets(k)) * 9
        footprint = khop_distinct_footprint(k, args.dim)
        nf_distinct = footprint * 9
        pf = results["loso"][k]
        m = float(np.mean(list(pf.values())))
        means[k] = m
        cells = " ".join(f"{pf[sc]:>8.3f}" for sc in scenarios)
        d = "-" if prev_f1 is None else f"{m - prev_f1:+.3f}"
        cap = capped["loso"].get(k)
        cap_s = ("-" if k not in capped["loso"]
                 else ("n/a" if cap is None
                       else f"{np.mean(list(cap.values())):.3f}"))
        flag = "  [FLAG n_tr/F<20]" if ratios["loso"][k] < 20 else ""
        if prev_footprint is not None and footprint == prev_footprint:
            flag += (f"  [FLAG k={k} n'agrandit pas le voisinage : meme "
                     f"empreinte que k={k - 1} sur grille dim={args.dim}]")
        print(f"  {k:>3} {nf:>8} {nf_distinct:>10} "
              f"{ratios['loso'][k]:>8.1f} {cells}  "
              f"{m:>7.3f} {d:>10} {cap_s:>8}{flag}")
        prev_f1 = m
        prev_footprint = footprint

    deltas = [means[K_VALUES[i + 1]] - means[K_VALUES[i]]
              for i in range(len(K_VALUES) - 1)]
    print(f"\n  k=1 LOSO mean = {means[1]:.3f}  "
          "(acceptance: published stencil 0.215 +/- 0.001)")
    print("  per-hop deltas (LOSO mean): "
          + ", ".join(f"{d:+.3f}" for d in deltas))
    print("  Section-2 decision rule (flat: every |delta| <= 0.01 -> cone "
          "retired; rising -> slope quoted): stated in study/v3/RESULTS.md.")

    footprints = [khop_distinct_footprint(k, args.dim) for k in K_VALUES]
    saturated_ks = [K_VALUES[i] for i in range(1, len(K_VALUES))
                    if footprints[i] == footprints[i - 1]]
    if saturated_ks:
        # D-88 : mesure, dim=4, khop_distinct_footprint -- k=2 et k=3 rendent
        # tous deux 16 residus distincts (144 colonnes) sur cette grille : k=3
        # n'est PAS un voisinage plus grand que k=2 ici, seulement plus de
        # colonnes dupliquees (441 nominales contre 144 reelles, meme compte
        # qu'a k=2). Un delta/hop calcule sur deux k satures compare deux jeux
        # de colonnes structurellement identiques, pas deux tailles de cone.
        first_sat = K_VALUES[footprints.index(footprints[-1])]
        print(f"  [D-88] k in {saturated_ks} n'agrandit pas le voisinage sur "
              f"cette grille (dim={args.dim}) : empreinte saturee a "
              f"{footprints[-1]} residus distincts des k={first_sat}. Ne pas "
              "lire le delta/hop de ces k comme un point de la courbe de "
              "cone -- ils comparent des colonnes structurellement "
              "identiques, pas un voisinage plus grand.")

    # ---- sauvegarde ----
    out = os.path.join(RESULTS_DIR,
                       f"t1b_cone_curve_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        k_values=np.array(K_VALUES),
        scenarios=np.array(scenarios),
        f1_loso=np.array([[results["loso"][k][sc] for sc in scenarios]
                          for k in K_VALUES]),
        f1_loso_mean=np.array([means[k] for k in K_VALUES]),
        f1_blocked=np.array([results["blocked"][k] for k in K_VALUES]),
        f1_class_loso=np.array([cls_loso[sc] for sc in scenarios]),
        f1_class_blocked=cls_blocked,
        ratio_loso=np.array([ratios["loso"][k] for k in K_VALUES]),
        ratio_blocked=np.array([ratios["blocked"][k] for k in K_VALUES]),
        f1_loso_capped=json.dumps({str(k): v for k, v
                                   in capped["loso"].items()}),
        f1_blocked_capped=json.dumps({str(k): v for k, v
                                      in capped["blocked"].items()}),
        # D-88 : n_feats nominal (boule de Chebyshev) contre n_feats_distinct
        # (colonnes reellement distinctes sur cette grille periodique) --
        # les deux divergent des que 2k+1 >= dim, et l'ecart dit combien de
        # colonnes de la matrice d'entrainement sont des doublons exacts.
        n_feats_nominal=np.array([len(khop_offsets(k)) * 9
                                  for k in K_VALUES]),
        n_feats_distinct=np.array([f * 9 for f in footprints]),
        k_saturated=np.array([k in saturated_ks for k in K_VALUES]),
        n_train_blocked=len(Ytr_b), n_val_blocked=len(Yva_b),
        seed=args.seed,
        git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nV3 Task 1b complete.")


if __name__ == "__main__":
    main()
