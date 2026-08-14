#!/usr/bin/env python3
"""
V3 Task 4 - Split temporellement bloque (protocole v3, section 2a) :
copie de la boucle d'evaluation de phase 11, avec :

  - split bloque : les premiers 60 % des snapshots de chaque trajectoire
    (scenario, Re) en train, les 40 % restants en val (regle Task 4,
    `blocked_split_indices` de t1b) ;
  - split aleatoire de phase 11A recalcule UNE fois (memes seed=0 et
    train-frac=0.7, meme permutation) : l'ecart bloque vs aleatoire est
    la quantification de la fuite (leakage) due aux snapshots
    quasi-dupliques (section 7) ;
  - baselines B1-B7 (section 1.3) avec double agregation du score
    classique : B1 = block_avg (convention V1), B2 = block_max
    (convention V2). Les deux agregent le MEME champ fin
    (`full_score` de build_patch_hamiltonian), seul l'operateur change
    (miroir du `_block_avg` de phase11e_v1h_loso.py) ;
  - metriques Task 2 : CE(b) pour b dans {0.10, 0.25, 0.50} + AUC de la
    courbe CE et rho de Spearman contre le e_i CONTINU, calcules PAR
    SNAPSHOT (classement des dim^2 patches d'un pas de temps, fidele au
    budget par pas du pipeline V1) puis moyennes ; F1 au label p75
    (seuil ajuste sur train) avec drapeaux de degenerescence 1.3-B3 ;
  - B5 a double seuil (spec utilisateur, suite de la branche 2 du
    Task 1) : (a) seuil global ajuste sur le train regroupe ;
    (b) seuil par trajectoire (scenario, Re) ajuste sur la partie train
    de cette trajectoire. CE/rho sont identiques entre (a) et (b)
    (sans seuil) ; seul F1 change.

Sortie : results/t4_blocked_split_N{N}_dim{D}.npz
         (hash git + arguments CLI complets)

Usage :
  python study/v3/t4_blocked_split.py --N 256 --dim 4
"""
import argparse, json, os, sys, time
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

from metrics import captured_error_at_budget, degeneracy_flag, spearman
from h2b_feature_selection import git_commit_hash
from h2b_neighbour_cone_curve import blocked_split_indices

BUDGETS = (0.10, 0.25, 0.50)

# D-85 : critere d'acceptation de la tache 4 (protocole v3, section 8.3) —
# « les nombres du split aleatoire correspondent a la tache 0 ». Il etait
# imprime en prose et jamais compare, donc il ne pouvait pas echouer ; le
# test du fichier renvoyait sa verification a « l'execution sur les vraies
# donnees », ou elle n'avait pas lieu. Meme forme que D-52.
#
# Les deux references sont des nombres d'ARCHIVE d'avant l'audit (meme
# provenance que celles de `aggregate_v3.py`, cf. D-49). Elles ne sont pas
# reajustees : un seuil perime se remesure, il ne se retouche pas.
ACCEPTANCE_REFS = (
    ("B2 classical (block_max)", 0.475),
    ("B4 gbt-9 (max)", 0.980),
)
TOL_ACCEPT = 0.01


def check_acceptance(f1_by_name, refs=ACCEPTANCE_REFS, tol=TOL_ACCEPT):
    """Compare les F1 du split aleatoire aux references de la tache 0.

    Rend [(nom, reference, mesure, ok), ...]. Leve si un nom attendu
    n'est pas dans la table : une reference qui ne designe aucune ligne
    ne compare rien, et passerait pour un succes (piege du balayage vide).
    """
    rows = []
    for name, ref in refs:
        if name not in f1_by_name:
            raise KeyError(
                f"reference d'acceptation « {name} » absente de la table du "
                f"split aleatoire ({sorted(f1_by_name)}) : elle ne comparerait "
                "rien (D-85)")
        got = float(f1_by_name[name])
        rows.append((name, float(ref), got, abs(got - ref) <= tol))
    return rows


# -------------------------------------------------------------------
# Helpers purs (testables sans qiskit)
# -------------------------------------------------------------------

def replace_score_column(X, s_new):
    """Copie de X avec la colonne 0 (score_classical) remplacee."""
    X2 = np.array(X, copy=True)
    X2[:, 0] = np.asarray(s_new).ravel()
    return X2


def split_indices_random(n_snaps, seed=0, train_frac=0.7):
    """Reproduction exacte du split aleatoire de phase 11A."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_snaps)
    n_tr = max(1, int(train_frac * n_snaps))
    return list(perm[:n_tr]), list(perm[n_tr:])


def split_indices_blocked(cfg_of_snap, train_frac=0.6):
    """Split bloque : pour chaque config (scenario, Re), les premiers
    60 % des snapshots (ordre temporel) en train, le reste en val.

    cfg_of_snap : liste (ordonnee dans le temps par config) des cles de
    config, une par snapshot. Retourne (idx_train, idx_val)."""
    by_cfg = {}
    for j, cfg in enumerate(cfg_of_snap):
        by_cfg.setdefault(cfg, []).append(j)
    tr_idx, va_idx = [], []
    for cfg in by_cfg:
        idx = by_cfg[cfg]
        tr, va = blocked_split_indices(len(idx), train_frac)
        tr_idx += [idx[u] for u in tr]
        va_idx += [idx[u] for u in va]
    return sorted(tr_idx), sorted(va_idx)


def per_config_thresholds(p_tr, y_tr, cfg_tr, thr_fn, grid=None):
    """Seuil F1-optimal ajuste PAR config sur sa partie train."""
    thr_map = {}
    p_tr = np.asarray(p_tr).ravel()
    y_tr = np.asarray(y_tr).ravel()
    cfg_tr = np.asarray(cfg_tr).ravel()
    for cfg in np.unique(cfg_tr):
        m = cfg_tr == cfg
        thr_map[cfg] = thr_fn(p_tr[m], y_tr[m], grid=grid)[0]
    return thr_map


def apply_per_config_threshold(p_va, cfg_va, thr_map, fallback_thr):
    """Predictions binaires avec le seuil propre a chaque config."""
    p_va = np.asarray(p_va).ravel()
    cfg_va = np.asarray(cfg_va).ravel()
    pred = np.empty(len(p_va), dtype=int)
    for cfg in np.unique(cfg_va):
        m = cfg_va == cfg
        thr = thr_map.get(cfg, fallback_thr)
        pred[m] = (p_va[m] > thr).astype(int)
    return pred


def ranking_metrics_per_snapshot(scores_snaps, e_snaps, budgets=BUDGETS):
    """CE(b), AUC de la courbe CE et Spearman, calcules par snapshot
    (classement des dim^2 patches d'un pas de temps) puis moyennes
    (nanmean : un snapshot a e total nul ou score constant donne NaN)."""
    ce_all = {b: [] for b in budgets}
    auc_all, rho_all = [], []
    for s, e in zip(scores_snaps, e_snaps):
        ce, auc = captured_error_at_budget(s, e, budgets)
        for b in budgets:
            ce_all[b].append(ce[b])
        auc_all.append(auc)
        if np.all(s == s[0]) or np.all(e == e[0]):
            rho_all.append(np.nan)
        else:
            rho_all.append(spearman(s, e))
    return ({b: float(np.nanmean(ce_all[b])) for b in budgets},
            float(np.nanmean(auc_all)), float(np.nanmean(rho_all)))


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

def _gather(by_scene, dim, max_snaps):
    """Liste plate de snapshots dans l'ordre EXACT de build_dataset
    (phase 11A) : scenario-major, puis Re, puis temps. Chaque entree :
    dict(cfg, feats(dim,dim,9), e, y, s_max, s_avg)."""
    from h2b_ceiling_random_split import extract_features_2d, N_FEATS, _block_avg
    from exact_diagonalisation import build_patch_hamiltonian

    snaps = []
    for sc, rows in by_scene.items():
        for re, dns_path, patches_path in rows:
            dns = np.load(dns_path)
            patches = np.load(patches_path)
            vx_all = dns["vx"].astype(np.float64)
            vy_all = dns["vy"].astype(np.float64)
            Bx_all = dns["Bx"].astype(np.float64)
            By_all = dns["By"].astype(np.float64)
            N = vx_all.shape[1]
            l2_all = patches["l2_errors"]
            l2_thr = float(patches["l2_threshold"])
            ps = N // dim

            n_snaps = len(vx_all)
            step = max(1, n_snaps // max_snaps)
            idx = list(range(0, n_snaps, step))[:max_snaps]

            for si in idx:
                vx, vy, Bx, By = (vx_all[si], vy_all[si],
                                  Bx_all[si], By_all[si])
                feats_2d, score_max = extract_features_2d(
                    vx, vy, Bx, By, N, dim, re)
                # meme champ fin, agregation moyenne (B1) :
                _, _, full_score = build_patch_hamiltonian(
                    vx, vy, Bx, By, N, dim, re,
                    threshold_amr=0.15, use_v2=True, c_bias=1.0)
                s_avg = _block_avg(full_score, ps, dim)
                # coherence : le canal score de feats = block_max(fin)
                chk = full_score.reshape(dim, ps, dim, ps).max(axis=(1, 3))
                if not np.array_equal(chk, score_max):
                    raise RuntimeError("score aggregation mismatch")

                snaps.append(dict(
                    cfg=f"{sc}|Re{re}",
                    X=feats_2d.reshape(-1, N_FEATS),
                    e=l2_all[si].ravel().astype(np.float64),
                    y=(l2_all[si] >= l2_thr).ravel().astype(int),
                    s_max=score_max.ravel(),
                    s_avg=s_avg.ravel(),
                ))
    return snaps


def _evaluate_split(snaps, tr_idx, va_idx, seed, label):
    """Table B1-B7 (double agregation) sur un split donne."""
    from h2b_ceiling_random_split import make_model, fit_eval, best_threshold_f1
    from h2b_learned_meanfield_h import fit_learned_h, predict_h

    n_cells = len(snaps[0]["y"])

    def cat(key, idxs):
        return np.concatenate([snaps[i][key] for i in idxs])

    Xtr, Xva = cat("X", tr_idx), cat("X", va_idx)
    Ytr, Yva = cat("y", tr_idx), cat("y", va_idx)
    Sx_tr, Sx_va = cat("s_max", tr_idx), cat("s_max", va_idx)
    Sa_tr, Sa_va = cat("s_avg", tr_idx), cat("s_avg", va_idx)
    cfg_tr = np.concatenate([[snaps[i]["cfg"]] * n_cells for i in tr_idx])
    cfg_va = np.concatenate([[snaps[i]["cfg"]] * n_cells for i in va_idx])
    e_snaps = [snaps[i]["e"] for i in va_idx]
    prev = float(Yva.mean())

    Xtr_avg = replace_score_column(Xtr, Sa_tr)
    Xva_avg = replace_score_column(Xva, Sa_va)

    rows = []

    def add_row(name, scores_va, thr, pred=None):
        pred = ((scores_va > thr).astype(int) if pred is None else pred)
        f1 = f1_score(Yva, pred, zero_division=0)
        flag = degeneracy_flag(pred, prev, gt=Yva)
        s_snaps = np.split(scores_va, len(va_idx))
        ce, auc, rho = ranking_metrics_per_snapshot(s_snaps, e_snaps)
        rows.append(dict(name=name, f1=float(f1), flag=bool(flag),
                         ce=ce, auc=auc, rho=rho))

    # B1/B2 : score classique, deux agregations (seuil sur train)
    thr_a, _ = best_threshold_f1(Sa_tr, Ytr)
    add_row("B1 classical (block_avg)", Sa_va, thr_a)
    thr_x, _ = best_threshold_f1(Sx_tr, Ytr)
    add_row("B2 classical (block_max)", Sx_va, thr_x)

    # B3 : planchers (F1 seulement ; pas de classement associe)
    f1_all = f1_score(Yva, np.ones_like(Yva), zero_division=0)
    rows.append(dict(name="B3 refine-all", f1=float(f1_all), flag=True,
                     ce=None, auc=None, rho=None))
    rows.append(dict(name="B3 refine-none", f1=0.0, flag=True,
                     ce=None, auc=None, rho=None))

    # B4 : GBT 9 features, deux variantes de la colonne score
    r4x = fit_eval(make_model("gbt", seed), Xtr, Ytr, Xva, Yva)
    add_row("B4 gbt-9 (max)", r4x["p"], r4x["thr"])
    r4a = fit_eval(make_model("gbt", seed), Xtr_avg, Ytr, Xva_avg, Yva)
    add_row("B4 gbt-9 (avg)", r4a["p"], r4a["thr"])

    # B5 : GBT score seul, double agregation x double seuil
    grid = np.linspace(0.05, 0.95, 91)  # grille de fit_eval
    for tag, str_, sva_ in (("max", Sx_tr, Sx_va), ("avg", Sa_tr, Sa_va)):
        r5 = fit_eval(make_model("gbt", seed),
                      str_.reshape(-1, 1), Ytr, sva_.reshape(-1, 1), Yva)
        add_row(f"B5 gbt-score ({tag}, thr global)", r5["p"], r5["thr"])
        m = make_model("gbt", seed).fit(str_.reshape(-1, 1), Ytr)
        p_tr = m.predict_proba(str_.reshape(-1, 1))[:, 1]
        thr_map = per_config_thresholds(p_tr, Ytr, cfg_tr,
                                        best_threshold_f1, grid=grid)
        pred = apply_per_config_threshold(r5["p"], cfg_va, thr_map,
                                          r5["thr"])
        add_row(f"B5 gbt-score ({tag}, thr per-cfg)", r5["p"],
                r5["thr"], pred=pred)

    # B6 : H lineaire appris (phase 11c), deux variantes
    for tag, xt, xv in (("max", Xtr, Xva), ("avg", Xtr_avg, Xva_avg)):
        m6 = fit_learned_h(xt, Ytr, seed=seed)
        h_tr, h_va = predict_h(m6, xt), predict_h(m6, xv)
        thr6, _ = best_threshold_f1(
            h_tr, Ytr, grid=np.linspace(h_tr.min(), h_tr.max(), 201))
        add_row(f"B6 linear-H ({tag})", h_va, thr6)

    # B7 : classement aleatoire
    rng = np.random.default_rng(seed)
    r_tr = rng.uniform(size=len(Ytr))
    r_va = rng.uniform(size=len(Yva))
    thr7, _ = best_threshold_f1(r_tr, Ytr)
    add_row("B7 random ranking", r_va, thr7)

    # ---- impression ----
    floor_all = 2 * prev / (1 + prev)
    print(f"\n  [{label}]  {len(tr_idx)} train snaps / {len(va_idx)} val "
          f"snaps  (val prevalence={prev:.3f}, refine-all floor "
          f"F1={floor_all:.3f})")
    head = (f"  {'method':<34} {'F1':>7} {'flag':>6} "
            + " ".join(f"{'CE@' + format(b, '.2f'): >8}" for b in BUDGETS)
            + f" {'CE-AUC':>8} {'rho':>7}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        if r["ce"] is None:
            cells = " ".join(f"{'-':>8}" for _ in BUDGETS)
            print(f"  {r['name']:<34} {r['f1']:>7.3f} "
                  f"{'DEGEN' if r['flag'] else '':>6} {cells} "
                  f"{'-':>8} {'-':>7}")
        else:
            cells = " ".join(f"{r['ce'][b]:>8.3f}" for b in BUDGETS)
            print(f"  {r['name']:<34} {r['f1']:>7.3f} "
                  f"{'DEGEN' if r['flag'] else '':>6} {cells} "
                  f"{r['auc']:>8.3f} {r['rho']:>7.3f}")
    return rows, prev


def main():
    p = argparse.ArgumentParser(
        description="V3 Task 4: temporally blocked split, B1-B7 table")
    from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N

    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--train-frac-blocked", type=float, default=0.6)
    p.add_argument("--train-frac-random", type=float, default=0.7,
                   help="fraction du split aleatoire de phase 11A")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  V3 Task 4: temporally blocked split vs phase-11A random split")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}  "
          f"seed={args.seed}")
    print("=" * 88)
    print()

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
    if not by_scene:
        # D-56 : ce garde imprimait « no input. » et rendait la main avec le
        # code 0, sans ecrire d'artefact — donc en laissant en place celui de
        # la campagne precedente. Une campagne qui n'avait rien mesure etait
        # indiscernable d'une campagne reussie. Onze autres modules de
        # `study/` levaient deja ici ; ceux-ci ne le faisaient pas.
        raise RuntimeError(
            "balayage vide : aucune scenarios n'a d'artefact d'entree pour les "
            "arguments donnes. Le script sortait ici avec le code 0 et sans "
            "artefact, donc sans se distinguer d'une campagne reussie.")

    print("  building dataset (features + dual-aggregation scores)...")
    t0 = time.time()
    snaps = _gather(by_scene, args.dim, args.max_snaps)
    print(f"  built {len(snaps)} snapshots in {time.time() - t0:.1f}s")

    cfg_of_snap = [s["cfg"] for s in snaps]
    tr_b, va_b = split_indices_blocked(cfg_of_snap,
                                       args.train_frac_blocked)
    tr_r, va_r = split_indices_random(len(snaps), args.seed,
                                      args.train_frac_random)

    rows_b, prev_b = _evaluate_split(snaps, tr_b, va_b, args.seed,
                                     "blocked split (60/40 temporal)")
    rows_r, prev_r = _evaluate_split(snaps, tr_r, va_r, args.seed,
                                     "random split (phase 11A repro)")

    # ---- ecart bloque vs aleatoire = quantification de la fuite ----
    print("\n  [leakage quantification]  random - blocked, per method:")
    print(f"  {'method':<34} {'F1_rand':>8} {'F1_blk':>8} {'gap':>8}")
    by_name_r = {r["name"]: r for r in rows_r}
    for rb in rows_b:
        rr = by_name_r[rb["name"]]
        print(f"  {rb['name']:<34} {rr['f1']:>8.3f} {rb['f1']:>8.3f} "
              f"{rr['f1'] - rb['f1']:>+8.3f}")

    # D-85 : compare, au lieu d'imprimer. Mesure au moment de la correction
    # (`--dim 4 --N 256 --seed 0`, Re=400, identique a --max-snaps 30 et 80,
    # GBT deterministe) : B2 classique 0,472 contre 0,475 (ecart 0,003, dans
    # la bande) ; B4 gbt-9(max) 0,908 contre 0,980 (ecart 0,072, HORS bande).
    print(f"\n  [ACCEPTANCE] tache 0, split aleatoire "
          f"(tolerance {TOL_ACCEPT:.3f})")
    acc_rows = check_acceptance({r["name"]: r["f1"] for r in rows_r})
    for name, ref, got, ok in acc_rows:
        print(f"    {name:<28} mesure {got:.3f}  reference {ref:.3f}  "
              f"ecart {got - ref:+.3f}   {'OK' if ok else 'MISMATCH'}")
    if not all(ok for *_, ok in acc_rows):
        print("    -> MISMATCH : les references de la tache 0 sont des "
              "nombres d'archive d'avant l'audit (D-49) ; l'ecart se "
              "consigne, la reference ne se retouche pas (D-85).")

    # ---- sauvegarde ----
    from config import RESULTS_DIR as RD
    def pack(rows):
        return dict(
            names=np.array([r["name"] for r in rows]),
            f1=np.array([r["f1"] for r in rows]),
            flag=np.array([r["flag"] for r in rows]),
            ce=np.array([[r["ce"][b] if r["ce"] else np.nan
                          for b in BUDGETS] for r in rows]),
            auc=np.array([r["auc"] if r["auc"] is not None else np.nan
                          for r in rows]),
            rho=np.array([r["rho"] if r["rho"] is not None else np.nan
                          for r in rows]),
        )
    pb, pr = pack(rows_b), pack(rows_r)
    out = os.path.join(RD, f"t4_blocked_split_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        budgets=np.array(BUDGETS),
        names=pb["names"],
        blocked_f1=pb["f1"], blocked_flag=pb["flag"], blocked_ce=pb["ce"],
        blocked_auc=pb["auc"], blocked_rho=pb["rho"],
        random_f1=pr["f1"], random_flag=pr["flag"], random_ce=pr["ce"],
        random_auc=pr["auc"], random_rho=pr["rho"],
        prevalence_blocked=prev_b, prevalence_random=prev_r,
        seed=args.seed,
        # D-85 : le critere d'acceptation, ecrit plutot qu'imprime
        acceptance_names=np.array([r[0] for r in acc_rows]),
        acceptance_ref=np.array([r[1] for r in acc_rows]),
        acceptance_measured=np.array([r[2] for r in acc_rows]),
        acceptance_ok=np.array([r[3] for r in acc_rows]),
        acceptance_tol=TOL_ACCEPT,
        git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nV3 Task 4 complete.")


if __name__ == "__main__":
    main()
