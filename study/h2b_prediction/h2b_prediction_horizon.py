#!/usr/bin/env python3
"""
V3 Task 7 - Dataset predictif, niveau 2 (protocole v3, section 3).

Tache : features a l'instant t -> predire la durete a t+h, pour des
horizons h in {1, 2, 4, 8} pas de snapshot.

Features a t (grille dim x dim) :
  - les 9 features instantanees de phase 11 (extract_features_2d) ;
  - psi SIGNE, deux variantes de Task 5 (jamais |psi|) :
      psi4  = machinerie V1, beta essai #4 (config.TRAINED_BETA) ;
      psiv2 = compute_psi_v2 sans parametre ;
    combinaison preservant le signe, agregees en block_avg (convention
    V1, coherente avec le block_avg des champs dans extract_features) ;
  - differences finies a un pas des 9 features (D9 = F9(t) - F9(t-1)).
  Toutes les paires utilisent t >= 1 (psi et D9 exigent t-1), pour une
  comparabilite stricte entre jeux de features.

Cibles : e_i(t+h) continu + y_i(t+h) = 1{e > P75} ; et d_i(t+h)
(verite terrain dynamique de Task 6) quand des fichiers d_patches_*
existent (paires restreintes aux snapshots calcules, computed_mask).

Modele : GBT de phase 11 entraine sur y(t+h) ; sa probabilite sert de
score de classement pour CE(b) / Spearman contre la cible CONTINUE.

Splits :
  - bloque (regle Task 4) au niveau PAIRES : train = paires entierement
    dans les premiers 60 % (t et t+h < t0) ; val = paires avec t >= t0 ;
    les paires a cheval sur la frontiere sont abandonnees (anti-fuite) ;
  - LOSO (folds phase 11b).

Metriques (Task 2, par snapshot puis moyennees, nan-safe) : CE(b),
AUC de courbe CE, rho ; F1 seuille sur train (drapeau 1.3-B3) ;
sous-ensemble facile-maintenant-difficile-a-t+h (y(t)=0 et y(t+h)=1) :
rappel + CE@0.25 restreinte au sous-ensemble ; capture@b des patchs
futurs-difficiles (table de delai d'anticipation, section 3).
Deltas (avec-psi - sans-psi) : bootstrap par blocs trajectoire (Task 3)
sur la CE@0.25 moyenne par trajectoire (scenario, Re).

Croisement cone causal : GBT sur les features k-hop de Task 1b,
k in {0, 1, 2}, a chaque horizon h -> matrice k x h (CE@0.25 et F1).

Sortie : results/t7_horizon_N{N}_dim{D}.npz
Usage :
  python study/v3/t7_horizon.py --N 256 --dim 4
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

from metrics import captured_error_at_budget, degeneracy_flag
from stats import paired_delta_bootstrap
from h2b_feature_selection import git_commit_hash
from h2b_neighbour_cone_curve import blocked_split_indices, khop_features
from h2b_blocked_split import ranking_metrics_per_snapshot
from h2b_psi_feature_loso import block_agg, psi_signed_v1, signed_combine

HORIZONS = (1, 2, 4, 8)
BUDGETS = (0.10, 0.25, 0.50)
K_CONE = (0, 1, 2)

FEATURE_SETS = [
    "base9",
    "base9+D9",
    "base9+psi4",
    "base9+psiv2",
    "full (base9+D9+psi4+psiv2)",
]
RAW_BASELINES = ["B1 classical score (avg)", "B2 classical score (max)"]


# -------------------------------------------------------------------
# Helpers purs (testables sans qiskit)
# -------------------------------------------------------------------

def horizon_pairs(n_snaps, h, min_t=1):
    """Paires (t, t+h) valides ; t >= min_t (psi/D9 exigent t-1)."""
    return [(t, t + h) for t in range(min_t, n_snaps - h)]


def blocked_pair_split(n_snaps, h, train_frac=0.6, min_t=1):
    """Split bloque au niveau paires : train si t ET t+h < t0 ;
    val si t >= t0 ; paires a cheval abandonnees (anti-fuite)."""
    tr_idx, _ = blocked_split_indices(n_snaps, train_frac)
    t0 = tr_idx[-1] + 1
    pairs = horizon_pairs(n_snaps, h, min_t)
    train = [(t, th) for t, th in pairs if th < t0]
    val = [(t, th) for t, th in pairs if t >= t0]
    return train, val


def enhl_mask(y_t, y_th):
    """Facile maintenant, difficile a t+h : y(t)=0 et y(t+h)=1."""
    return (np.asarray(y_t) == 0) & (np.asarray(y_th) == 1)


def _top_mask(scores, budget):
    n = len(scores)
    k = min(n, int(np.ceil(budget * n)))
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    m = np.zeros(n, dtype=bool)
    m[order[:k]] = True
    return m


def subset_captured_error(scores, e_target, subset, budget):
    """Part de l'erreur future du sous-ensemble capturee au budget b :
    sum(e[subset & top-k]) / sum(e[subset]). NaN si sous-ensemble vide
    ou d'erreur nulle."""
    subset = np.asarray(subset, bool)
    e_target = np.asarray(e_target, float)
    denom = e_target[subset].sum()
    if not subset.any() or denom <= 0:
        return np.nan
    top = _top_mask(scores, budget)
    return float(e_target[subset & top].sum() / denom)


def capture_at_budget(scores, y_future, budget):
    """Fraction des patchs futurs-difficiles dans le top-b (rappel au
    budget) ; NaN si aucun patch futur-difficile."""
    hard = np.asarray(y_future) == 1
    if not hard.any():
        return np.nan
    top = _top_mask(scores, budget)
    return float((hard & top).sum() / hard.sum())


def finite_diff_features(F9_seq):
    """D9(t) = F9(t) - F9(t-1) ; indefini a t=0 (rempli de NaN)."""
    out = np.full_like(F9_seq, np.nan)
    out[1:] = F9_seq[1:] - F9_seq[:-1]
    return out


# -------------------------------------------------------------------
# Gather
# -------------------------------------------------------------------

def _gather(by_scene, dim, max_snaps, beta4):
    """{(sc, re): dict de sequences temporelles} :
    F9 (T,16,9), D9, PSI4 (T,16), PSIV2, E (T,16), Y, S_avg, S_max,
    FEATS2D (T,dim,dim,9) pour les stencils k-hop."""
    from Simulation.PhysToAngle import AngleMapper
    from Simulation.HamiltParams_v2 import compute_psi_v2
    from h2b_ceiling_random_split import extract_features_2d, N_FEATS
    from exact_diagonalisation import build_patch_hamiltonian
    from h2b_v1_hamiltonian_loso import v1_state

    am = AngleMapper()
    out = {}
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
            n_snaps = len(vx_all)
            step = max(1, n_snaps // max_snaps)
            idx = list(range(0, n_snaps, step))[:max_snaps]
            ps = N // dim

            F9, FE2D, E, Y, Sa, Sx = [], [], [], [], [], []
            P4, PV2 = [], []
            phi_prev = None
            for si in idx:
                vx, vy, Bx, By = (vx_all[si], vy_all[si],
                                  Bx_all[si], By_all[si])
                feats_2d, s_max = extract_features_2d(
                    vx, vy, Bx, By, N, dim, re)
                _, _, full_score = build_patch_hamiltonian(
                    vx, vy, Bx, By, N, dim, re,
                    threshold_amr=0.15, use_v2=True, c_bias=1.0)
                phi = am.compute_stress_flux(v1_state(vx, vy, Bx, By))
                if phi_prev is None:
                    p4 = np.zeros((dim, dim))
                    pv2 = np.zeros((dim, dim))
                else:
                    dph = phi["phi_horizontal"] - phi_prev["phi_horizontal"]
                    dpv = phi["phi_vertical"] - phi_prev["phi_vertical"]
                    p4 = block_agg(psi_signed_v1(dph, dpv, beta4),
                                   dim, "avg")
                    pv2 = block_agg(signed_combine(
                        compute_psi_v2(phi_prev["phi_horizontal"],
                                       phi["phi_horizontal"]),
                        compute_psi_v2(phi_prev["phi_vertical"],
                                       phi["phi_vertical"])), dim, "avg")
                phi_prev = phi

                F9.append(feats_2d.reshape(-1, N_FEATS))
                FE2D.append(feats_2d)
                E.append(l2_all[si].ravel().astype(np.float64))
                Y.append((l2_all[si] >= l2_thr).ravel().astype(int))
                Sa.append(full_score.reshape(dim, ps, dim, ps)
                          .mean(axis=(1, 3)).ravel())
                Sx.append(s_max.ravel())
                P4.append(p4.ravel())
                PV2.append(pv2.ravel())

            seq = dict(F9=np.array(F9), FEATS2D=np.array(FE2D),
                       E=np.array(E), Y=np.array(Y),
                       S_avg=np.array(Sa), S_max=np.array(Sx),
                       PSI4=np.array(P4), PSIV2=np.array(PV2))
            seq["D9"] = finite_diff_features(seq["F9"])
            out[(sc, re)] = seq
    return out


def method_features(seq, t, name):
    """Matrice de features (16, F) d'un snapshot pour un jeu donne."""
    if name == "base9":
        return seq["F9"][t]
    if name == "base9+D9":
        return np.concatenate([seq["F9"][t], seq["D9"][t]], axis=1)
    if name == "base9+psi4":
        return np.concatenate(
            [seq["F9"][t], seq["PSI4"][t][:, None]], axis=1)
    if name == "base9+psiv2":
        return np.concatenate(
            [seq["F9"][t], seq["PSIV2"][t][:, None]], axis=1)
    if name == "full (base9+D9+psi4+psiv2)":
        return np.concatenate(
            [seq["F9"][t], seq["D9"][t],
             seq["PSI4"][t][:, None], seq["PSIV2"][t][:, None]], axis=1)
    if name.startswith("khop"):
        k = int(name[4:])
        return khop_features(seq["FEATS2D"][t], k)
    raise ValueError(name)


# -------------------------------------------------------------------
# Evaluation d'une (methode, horizon, split)
# -------------------------------------------------------------------

def _assemble(data, items, name, target="E"):
    """items = [((sc, re), (t, th)), ...] -> X, y(t+h), listes par
    snapshot (cible continue, y_t, y_th, score brut, tag config)."""
    X, Y = [], []
    per_snap = []
    for cfg, (t, th) in items:
        seq = data[cfg]
        X.append(method_features(seq, t, name)
                 if name in FEATURE_SETS or name.startswith("khop")
                 else None)
        Y.append(seq["Y"][th])
        raw = (seq["S_avg"][t] if name == RAW_BASELINES[0]
               else seq["S_max"][t] if name == RAW_BASELINES[1]
               else None)
        per_snap.append(dict(cfg=cfg, e=seq[target][th],
                             y_t=seq["Y"][t], y_th=seq["Y"][th],
                             raw=raw))
    Xc = np.concatenate(X) if X and X[0] is not None else None
    return Xc, np.concatenate(Y), per_snap


def eval_method(data, train_items, val_items, name, seed,
                make_model, fit_eval, best_threshold_f1, target="E"):
    """Retourne dict de metriques + CE@0.25 par snapshot (tags cfg)."""
    Xtr, Ytr, snaps_tr = _assemble(data, train_items, name, target)
    Xva, Yva, snaps = _assemble(data, val_items, name, target)

    if name in RAW_BASELINES:
        s_tr = np.concatenate([d["raw"] for d in snaps_tr])
        scores = [d["raw"] for d in snaps]
        thr, _ = best_threshold_f1(s_tr, Ytr)
        p_va = np.concatenate(scores)
    else:
        r = fit_eval(make_model("gbt", seed), Xtr, Ytr, Xva, Yva)
        p_va = r["p"]
        thr = r["thr"]
        scores = list(np.split(p_va, len(snaps)))

    e_snaps = [d["e"] for d in snaps]
    ce, auc, rho = ranking_metrics_per_snapshot(scores, e_snaps,
                                                BUDGETS)
    pred = (p_va > thr).astype(int)
    f1 = float(f1_score(Yva, pred, zero_division=0))
    flag = bool(degeneracy_flag(pred, float(Yva.mean()), gt=Yva))

    # sous-ensemble facile-maintenant-difficile-plus-tard + captures
    enhl_rec, enhl_ce, capt = [], [], {b: [] for b in BUDGETS}
    ce25_by_cfg = {}
    pred_snaps = np.split(pred, len(snaps))
    for s, d, pr in zip(scores, snaps, pred_snaps):
        m = enhl_mask(d["y_t"], d["y_th"])
        if m.any():
            enhl_rec.append(float(pr[m].mean()))
            enhl_ce.append(subset_captured_error(s, d["e"], m, 0.25))
        for b in BUDGETS:
            capt[b].append(capture_at_budget(s, d["y_th"], b))
        ce_s, _ = captured_error_at_budget(s, d["e"], (0.25,))
        ce25_by_cfg.setdefault(d["cfg"], []).append(ce_s[0.25])

    return dict(
        name=name, ce=ce, auc=auc, rho=rho, f1=f1, flag=flag,
        enhl_recall=float(np.nanmean(enhl_rec)) if enhl_rec else np.nan,
        enhl_ce25=float(np.nanmean(enhl_ce)) if enhl_ce else np.nan,
        capture={b: float(np.nanmean(capt[b])) for b in BUDGETS},
        ce25_per_traj={c: float(np.nanmean(v))
                       for c, v in ce25_by_cfg.items()},
        n_val_pairs=len(snaps),
    )


def _print_table(rows, label):
    print(f"\n  [{label}]")
    head = (f"  {'method':<30} {'F1':>6} {'flg':>4} "
            + " ".join(f"{'CE@' + format(b, '.2f'):>7}" for b in BUDGETS)
            + f" {'rho':>6} {'ENHLrec':>8} {'ENHLce':>7} "
            + " ".join(f"{'cap@' + format(b, '.2f'):>8}" for b in BUDGETS))
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        ces = " ".join(f"{r['ce'][b]:>7.3f}" for b in BUDGETS)
        caps = " ".join(f"{r['capture'][b]:>8.3f}" for b in BUDGETS)
        print(f"  {r['name']:<30} {r['f1']:>6.3f} "
              f"{'DEG' if r['flag'] else '':>4} {ces} {r['rho']:>6.3f} "
              f"{r['enhl_recall']:>8.3f} {r['enhl_ce25']:>7.3f} {caps}")


def common_traj_values(ra, rb, cfgs):
    """Trajectoires presentes et finies dans les DEUX bras (un config
    court peut n'avoir aucune paire val a grand h, p.ex. harris a
    h=8 bloque) ; le bootstrap apparie s'applique a l'intersection."""
    common = [c for c in cfgs
              if c in ra and c in rb
              and np.isfinite(ra[c]) and np.isfinite(rb[c])]
    return (np.array([ra[c] for c in common]),
            np.array([rb[c] for c in common]), common)


def main():
    p = argparse.ArgumentParser(
        description="V3 Task 7: predictive dataset, Level 2")
    from config import (RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N,
                        TRAINED_BETA)
    from h2b_ceiling_random_split import make_model, fit_eval, best_threshold_f1

    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  V3 Task 7: predictive dataset (Level 2) — horizons "
          f"h in {HORIZONS} snapshot steps")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}"
          f"  seed={args.seed}  psi4 beta={TRAINED_BETA}")
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
    if len(by_scene) < 2:
        # D-75 : cette garde faisait `print(...); return` — code 0, aucun
        # artefact ecrit, donc indiscernable d'une campagne reussie (meme
        # famille que D-56 et D-74). Le detecteur AST de D-56 ne voyait que
        # la forme `if not <accumulateur nomme>:` ; celle-ci lui echappait.
        raise RuntimeError(
            "balayage vide : l'horizon predictif exige au moins 2 scenarios avec "
            f"artefacts d'entree, {len(by_scene)} trouve(s) ({sorted(by_scene)}). "
            "Le script sortait ici avec le code 0 et sans artefact (D-75).")

    print("  building per-config temporal sequences...")
    t0 = time.time()
    data = _gather(by_scene, args.dim, args.max_snaps, TRAINED_BETA)
    print(f"  done in {time.time() - t0:.1f}s")
    cfgs = list(data.keys())
    scenarios = list(dict.fromkeys(sc for sc, _ in cfgs))

    def loso_items(h, held):
        tr, va = [], []
        for cfg in cfgs:
            pairs = horizon_pairs(len(data[cfg]["Y"]), h)
            tgt = va if cfg[0] == held else tr
            tgt += [(cfg, pr) for pr in pairs]
        return tr, va

    def blocked_items(h):
        tr, va = [], []
        for cfg in cfgs:
            ptr, pva = blocked_pair_split(len(data[cfg]["Y"]), h,
                                          args.train_frac)
            tr += [(cfg, pr) for pr in ptr]
            va += [(cfg, pr) for pr in pva]
        return tr, va

    methods = RAW_BASELINES + FEATURE_SETS
    results = {}          # (split, h, name) -> row
    eval_kw = dict(make_model=make_model, fit_eval=fit_eval,
                   best_threshold_f1=best_threshold_f1)

    for h in HORIZONS:
        # ---- bloque ----
        tr, va = blocked_items(h)
        rows = [eval_method(data, tr, va, m, args.seed, **eval_kw)
                for m in methods]
        for r in rows:
            results[("blocked", h, r["name"])] = r
        _print_table(rows, f"blocked split, h={h} "
                     f"({len(va)} val pairs)")

        # ---- LOSO : moyenne des folds (metriques par snapshot) ----
        rows_l = []
        for m in methods:
            fold_rows = []
            for held in scenarios:
                tr, va = loso_items(h, held)
                fold_rows.append(eval_method(data, tr, va, m,
                                             args.seed, **eval_kw))
            agg = dict(
                name=m,
                ce={b: float(np.nanmean([fr["ce"][b]
                                         for fr in fold_rows]))
                    for b in BUDGETS},
                auc=float(np.nanmean([fr["auc"] for fr in fold_rows])),
                rho=float(np.nanmean([fr["rho"] for fr in fold_rows])),
                f1=float(np.mean([fr["f1"] for fr in fold_rows])),
                flag=any(fr["flag"] for fr in fold_rows),
                enhl_recall=float(np.nanmean(
                    [fr["enhl_recall"] for fr in fold_rows])),
                enhl_ce25=float(np.nanmean(
                    [fr["enhl_ce25"] for fr in fold_rows])),
                capture={b: float(np.nanmean([fr["capture"][b]
                                              for fr in fold_rows]))
                         for b in BUDGETS},
                ce25_per_traj={k: v for fr in fold_rows
                               for k, v in fr["ce25_per_traj"].items()},
                n_val_pairs=sum(fr["n_val_pairs"] for fr in fold_rows),
            )
            rows_l.append(agg)
            results[("loso", h, m)] = agg
        _print_table(rows_l, f"LOSO (mean over folds), h={h}")

    # ---- table de delai d'anticipation (capture@0.25 vs h) ----
    for split in ("blocked", "loso"):
        print(f"\n  [lead-time table, {split}: capture@0.25 of "
              "future-hard patches vs horizon]")
        print(f"  {'method':<30} "
              + " ".join(f"{'h=' + str(h):>7}" for h in HORIZONS))
        for m in methods:
            cells = " ".join(
                f"{results[(split, h, m)]['capture'][0.25]:>7.3f}"
                for h in HORIZONS)
            print(f"  {m:<30} {cells}")

    # ---- deltas avec-psi - sans-psi, bootstrap trajectoire ----
    pairs_delta = [("base9+psi4", "base9"),
                   ("base9+psiv2", "base9"),
                   ("full (base9+D9+psi4+psiv2)", "base9+D9")]
    print(f"\n  [psi deltas: CE@0.25 per trajectory, paired bootstrap "
          f"B={args.n_boot}]")
    print(f"  {'pair':<46} {'split':<8} {'h':>2} {'n_tr':>4} "
          f"{'mean_d':>8} {'CI95':>18} {'frac>0':>7}")
    boot_rows = []
    for split in ("blocked", "loso"):
        for h in HORIZONS:
            for a, b in pairs_delta:
                ra = results[(split, h, a)]["ce25_per_traj"]
                rb = results[(split, h, b)]["ce25_per_traj"]
                va, vb, common = common_traj_values(ra, rb, cfgs)
                if len(common) < 2:
                    print(f"  {a + ' - ' + b:<46} {split:<8} {h:>2} "
                          f"{len(common):>4} (skipped: <2 trajectories)")
                    continue
                r = paired_delta_bootstrap(va, vb,
                                           np.arange(len(common)),
                                           B=args.n_boot,
                                           seed=args.seed)
                boot_rows.append((split, h, a, b, len(common), r))
                print(f"  {a + ' - ' + b:<46} {split:<8} {h:>2} "
                      f"{len(common):>4} {r['mean_delta']:>+8.3f} "
                      f"[{r['ci_low']:>+7.3f},{r['ci_high']:>+7.3f}] "
                      f"{r['frac_positive']:>7.2f}")

    # ---- croisement cone causal : matrice k x h ----
    print("\n  [causal-cone cross: GBT on k-hop features]")
    cone = {}
    for split in ("blocked", "loso"):
        for metric in ("ce25", "f1"):
            print(f"\n  {split}, {'CE@0.25' if metric == 'ce25' else 'F1'}:")
            print(f"  {'k':>3} " + " ".join(f"{'h=' + str(h):>7}"
                                            for h in HORIZONS))
            for k in K_CONE:
                cells = []
                for h in HORIZONS:
                    key = (split, h, f"khop{k}")
                    if key not in cone:
                        if split == "blocked":
                            tr, va = blocked_items(h)
                            cone[key] = eval_method(
                                data, tr, va, f"khop{k}", args.seed,
                                **eval_kw)
                        else:
                            frs = []
                            for held in scenarios:
                                tr, va = loso_items(h, held)
                                frs.append(eval_method(
                                    data, tr, va, f"khop{k}",
                                    args.seed, **eval_kw))
                            cone[key] = dict(
                                ce={0.25: float(np.nanmean(
                                    [f["ce"][0.25] for f in frs]))},
                                f1=float(np.mean(
                                    [f["f1"] for f in frs])))
                    r = cone[key]
                    cells.append(r["ce"][0.25] if metric == "ce25"
                                 else r["f1"])
                print(f"  {k:>3} " + " ".join(f"{c:>7.3f}"
                                              for c in cells))

    # ---- cible d_i (Task 6) si disponible ----
    d_found = []
    for (sc, re) in cfgs:
        dpath = os.path.join(
            RESULTS_DIR, f"d_patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
        if os.path.exists(dpath):
            d_found.append((sc, re, dpath))
    if d_found:
        print(f"\n  [d_i target: {len(d_found)} d_patches files found "
              "— evaluating e-trained rankings against d(t+h) is left "
              "to the full Task-6 campaign aggregation]")
    else:
        print("\n  [d_i target: no d_patches files at this N/dim — "
              "e-target only (per section 8.4, no new simulation "
              "needed for the e-variant)]")

    # ---- sauvegarde ----
    out = os.path.join(RESULTS_DIR,
                       f"t7_horizon_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        horizons=np.array(HORIZONS), budgets=np.array(BUDGETS),
        methods=np.array(methods),
        splits=np.array(["blocked", "loso"]),
        ce=np.array([[[results[(s, h, m)]["ce"][b] for b in BUDGETS]
                      for h in HORIZONS]
                     for s in ("blocked", "loso") for m in methods]
                    ).reshape(2, len(methods), len(HORIZONS),
                              len(BUDGETS)),
        f1=np.array([[results[(s, h, m)]["f1"] for h in HORIZONS]
                     for s in ("blocked", "loso") for m in methods]
                    ).reshape(2, len(methods), len(HORIZONS)),
        rho=np.array([[results[(s, h, m)]["rho"] for h in HORIZONS]
                      for s in ("blocked", "loso") for m in methods]
                     ).reshape(2, len(methods), len(HORIZONS)),
        enhl_recall=np.array(
            [[results[(s, h, m)]["enhl_recall"] for h in HORIZONS]
             for s in ("blocked", "loso") for m in methods]
        ).reshape(2, len(methods), len(HORIZONS)),
        enhl_ce25=np.array(
            [[results[(s, h, m)]["enhl_ce25"] for h in HORIZONS]
             for s in ("blocked", "loso") for m in methods]
        ).reshape(2, len(methods), len(HORIZONS)),
        capture25=np.array(
            [[results[(s, h, m)]["capture"][0.25] for h in HORIZONS]
             for s in ("blocked", "loso") for m in methods]
        ).reshape(2, len(methods), len(HORIZONS)),
        cone_ce25=np.array(
            [[[cone[(s, h, f"khop{k}")]["ce"][0.25]
               for h in HORIZONS] for k in K_CONE]
             for s in ("blocked", "loso")]),
        cone_f1=np.array(
            [[[cone[(s, h, f"khop{k}")]["f1"]
               for h in HORIZONS] for k in K_CONE]
             for s in ("blocked", "loso")]),
        boot=json.dumps([dict(split=s, h=h, a=a, b=b, n_traj=n,
                              mean=r["mean_delta"], lo=r["ci_low"],
                              hi=r["ci_high"],
                              frac=r["frac_positive"])
                         for s, h, a, b, n, r in boot_rows]),
        seed=args.seed,
        git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nV3 Task 7 complete.")


if __name__ == "__main__":
    main()
