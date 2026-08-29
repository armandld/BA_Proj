#!/usr/bin/env python
"""T31 — de combien la convention d'axes des mappeurs deplace-t-elle la decision ?

Les mappeurs forment leur rotationnel et leur divergence sous la convention
indexing='xy' alors que `grid.py` declare indexing='ij' (AXIS_X=0, AXIS_Y=1).
Sous la convention du depot, leur « vorticite » vaut dv_y/dy - dv_x/dx et leur
« divergence » dv_x/dy + dv_y/dx : deux composantes du tenseur des
deformations. La premiere est exactement nulle sur une rotation solide, la
seconde exactement nulle sur une compression isotrope
(`tests/test_analytic_fields.py`).

Ce script ne corrige rien. Il mesure ce que la correction changerait, pour
que la decision de re-optimiser ou non les hyperparametres — une semaine de
calcul, cf. `results/hyperparams/PROVENANCE.md` — repose sur un nombre.

Quatre grandeurs sont comparees entre `fixed_curl=False` (chemin historique,
celui sur lequel Optuna a regle ses valeurs) et `fixed_curl=True` :

  - le score classique lui-meme (ecart RMS et max, correlation de Pearson) ;
  - les coefficients de l'hamiltonien, canal par canal ;
  - le CLASSEMENT des patches par le score, confronte a la durete continue
    du label : correlation de Spearman, et F1 a budget appraie (les k
    patches les mieux classes, k = nombre de patches durs) ;
  - la decision au seuil entraine, gardee en diagnostic seulement.

La troisieme famille porte la conclusion. Les deux premieres montrent
seulement que la convention change quelque chose ; la quatrieme est
inexploitable telle quelle, car au seuil entraine (0.1496) le score de patch
sature et les deux bras degenerent en « tout raffiner » — le F1 y vaut alors
celui du predicteur constant, dans les deux conventions. Le budget appraie
retire le seuil de la comparaison : les deux bras raffinent exactement le
meme NOMBRE de patches et ne different plus que par LESQUELS.

Si le classement ne bouge pas, la convention est sans effet sur la tache et
la campagne Optuna n'a pas besoin d'etre refaite.

Exemple
-------
    python study/h1_solver/h1_curl_convention_gap.py --N 128 --dim 8 \
        --n-snaps 6 --seed 0
"""

import argparse
import json
import os
import sys

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

import provenance                               # noqa: E402

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
SCENARIOS = ("orszag_tang", "kelvin_helmholtz", "mhd_rotor", "harris_tearing")


def _f1(pred, truth):
    pred = np.asarray(pred, bool).ravel()
    truth = np.asarray(truth, bool).ravel()
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    return 0.0 if (2 * tp + fp + fn) == 0 else 2.0 * tp / (2 * tp + fp + fn)


def _spearman(a, b):
    """Correlation de rang, sans seuil. NaN si l'un des deux est constant."""
    from scipy.stats import spearmanr
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if np.ptp(a) < 1e-15 or np.ptp(b) < 1e-15:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _top_k(score, k):
    """Budget apparie : les k patches les mieux classes, k = nombre de durs.

    Neutralise le seuil. Les deux conventions raffinent alors exactement le
    meme NOMBRE de patches et ne different que par LESQUELS — c'est la seule
    difference imputable a la convention d'axes.
    """
    score = np.asarray(score, dtype=float).ravel()
    out = np.zeros(score.size, dtype=bool)
    if k > 0:
        out[np.argsort(-score, kind="stable")[:k]] = True
    return out


def bootstrap_delta_ci(rows, key_legacy, key_fixed, n_boot=10000, seed=0):
    """IC95 de l'ecart appraie, rechantillonne par SCENARIO.

    Les instantanes d'une meme trajectoire ne sont pas independants : ils
    partagent la condition initiale et se suivent de quelques pas de temps.
    Rechantillonner les instantanes un par un retrecirait l'intervalle d'un
    facteur ~sqrt(n_snaps) et rendrait significatif a peu pres n'importe quel
    ecart. Le bloc est donc le scenario, ce qui laisse peu de blocs et donne
    un intervalle large — c'est la reponse honnete avec quatre trajectoires.
    """
    rng = np.random.default_rng(seed)
    by_scen = {}
    for r in rows:
        d = float(r[key_fixed]) - float(r[key_legacy])
        by_scen.setdefault(r["scenario"], []).append(d)
    blocks = [np.array(v, dtype=float) for v in by_scen.values()]
    if len(blocks) < 2:
        return float("nan"), float("nan"), float("nan")

    observed = float(np.mean(np.concatenate(blocks)))
    idx = np.arange(len(blocks))
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(idx, size=len(blocks), replace=True)
        means[b] = np.mean(np.concatenate([blocks[i] for i in pick]))
    return observed, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(lo, hi):
    """Trois issues, et deux seulement autorisent une conclusion."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "indecidable"
    if lo > 0.0:
        return "la convention corrigee ameliore"
    if hi < 0.0:
        return "la convention corrigee degrade"
    return "indecidable"


def _hard_patches(vx, vy, Bx, By, N, dim, quantile):
    """Label « patch dur » : erreur L2 de reconstruction par grossissement en bloc.

    Meme definition que `study/pipeline/hard_patch_labels.py::patch_l2_errors`
    (moyenne par bloc puis relevement plus proche voisin, ecart quadratique
    moyen sur les 4 champs, normalise par le RMS global), recopiee ici sur les
    seuls champs dont ce script dispose. Le quantile est calcule par
    instantane : ce script compare deux variantes SUR LE MEME label, donc le
    choix du seuil n'entre pas dans l'ecart mesure.

    L'ancienne formule calculait l'ecart-type intra-patch de la NORME du
    champ (sqrt(vx^2+vy^2+Bx^2+By^2)) — differente de celle que la
    docstring annoncait, et qui s'accorde avec `patch_l2_errors` sur un
    champ lisse mais diverge totalement des qu'un champ oscille a
    magnitude constante (ecart-type nul, alors que l'information est
    perdue par le grossissement en bloc : erreur de reconstruction maximale).
    Mesure sur un patch en damier +1/-1 (magnitude constante, information
    fine detruite par la moyenne de bloc) : ancienne formule 0.0000 (patch
    juge LE PLUS FACILE), `patch_l2_errors` 3.327 (patch le plus difficile
    du champ, x60 le second) — classement invertie. Voir
    tests/study/test_hard_patches_matches_canonical.py.
    """
    ps = N // dim

    def _coarsen_prolong(field):
        coarse = field.reshape(dim, ps, dim, ps).mean(axis=(1, 3))
        return np.repeat(np.repeat(coarse, ps, axis=0), ps, axis=1)

    rms = float(np.sqrt(np.mean(vx ** 2 + vy ** 2 + Bx ** 2 + By ** 2)))
    rms = rms if rms > 1e-15 else 1.0
    diff_sq = (
        (vx - _coarsen_prolong(vx)) ** 2
        + (vy - _coarsen_prolong(vy)) ** 2
        + (Bx - _coarsen_prolong(Bx)) ** 2
        + (By - _coarsen_prolong(By)) ** 2
    )
    l2 = np.sqrt(diff_sq.reshape(dim, ps, dim, ps).mean(axis=(1, 3))) / rms
    thr = float(np.quantile(l2, quantile))
    return l2, thr, (l2 > thr)


def run(args):
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver
    from qaoa_inputs import prepare_qaoa_inputs
    from config import TRAINED_THRESHOLD, V2_THRESHOLD

    # Ce balayage est entierement deterministe : le solveur part d'une
    # condition initiale fixee et aucun tirage n'intervient. --seed est
    # accepte et consigne pour la provenance, il ne pilote rien ici.
    use_v2 = (args.mapper == "v2")
    thr_amr = V2_THRESHOLD if use_v2 else TRAINED_THRESHOLD

    rows = []
    for sc in args.scenario:
        sim = MHDSolver(PeriodicGrid(args.N), dt=1e-3, Re=args.re, Rm=args.re)
        getattr(sim, "init_" + sc)()
        # on laisse la turbulence s'etablir avant d'echantillonner
        for _ in range(args.spinup):
            sim.adapt_dt(cfl_target=0.4)
            sim.step_full(record_stats=False)

        for si in range(args.n_snaps):
            for _ in range(args.stride):
                sim.adapt_dt(cfl_target=0.4)
                sim.step_full(record_stats=False)
            f = sim.get_fluxes()
            vx, vy, Bx, By = f["vx"], f["vy"], f["Bx"], f["By"]

            _l2, _thr, is_hard = _hard_patches(
                vx, vy, Bx, By, args.N, args.dim, args.quantile)

            out = {}
            for tag, fc in (("legacy", False), ("fixed", True)):
                out[tag] = prepare_qaoa_inputs(
                    vx, vy, Bx, By, args.N, args.dim, args.re,
                    use_v2=use_v2, fixed_curl=fc)

            sc_l = np.asarray(out["legacy"][2], dtype=float)
            sc_f = np.asarray(out["fixed"][2], dtype=float)
            dec_l = sc_l > thr_amr
            dec_f = sc_f > thr_amr

            # correlation : indefinie si l'un des deux est constant
            if np.ptp(sc_l) < 1e-15 or np.ptp(sc_f) < 1e-15:
                pearson = float("nan")
            else:
                pearson = float(np.corrcoef(sc_l.ravel(), sc_f.ravel())[0, 1])

            row = dict(
                scenario=sc, snap=si,
                score_rms_gap=float(np.sqrt(np.mean((sc_l - sc_f) ** 2))),
                score_max_gap=float(np.max(np.abs(sc_l - sc_f))),
                score_pearson=pearson,
                # Metriques a seuil, gardees en diagnostic seulement : au
                # seuil entraine (0.1496) le score de patch sature et les
                # deux bras degenerent en « tout raffiner ».
                decision_agreement=float(np.mean(dec_l == dec_f)),
                frac_refine_legacy=float(np.mean(dec_l)),
                frac_refine_fixed=float(np.mean(dec_f)),
                prevalence_hard=float(np.mean(is_hard)),
                f1_thr_legacy=_f1(dec_l, is_hard),
                f1_thr_fixed=_f1(dec_f, is_hard),
                # Un F1 obtenu en decidant toujours pareil n'est pas un
                # score (leçon de T29).
                degenerate=bool(dec_l.min() == dec_l.max()
                                or dec_f.min() == dec_f.max()),
                # Metriques SANS seuil : elles comparent le CLASSEMENT des
                # patches par les deux conventions, ce qui est la seule
                # question posee ici. Le seuil entraine appartient a la
                # campagne Optuna, pas a la convention d'axes.
                spearman_legacy=_spearman(sc_l, _l2),
                spearman_fixed=_spearman(sc_f, _l2),
                f1_matched_legacy=_f1(_top_k(sc_l, int(is_hard.sum())), is_hard),
                f1_matched_fixed=_f1(_top_k(sc_f, int(is_hard.sum())), is_hard),
            )

            # coefficients, canal par canal
            for chan in ("h_bias", "C_couplings", "K_plaquettes"):
                a = out["legacy"][1].get(chan)
                b = out["fixed"][1].get(chan)
                if a is None or b is None:
                    continue
                a = np.asarray(a, dtype=float)
                b = np.asarray(b, dtype=float)
                den = max(float(np.max(np.abs(a))), 1e-30)
                row[f"{chan}_rel_gap"] = float(
                    np.max(np.abs(a - b)) / den)
                row[f"{chan}_pearson"] = (
                    float("nan") if np.ptp(a) < 1e-30 or np.ptp(b) < 1e-30
                    else float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
            rows.append(row)

    if not rows:
        raise RuntimeError(
            "balayage vide : aucun instantane n'a ete evalue, le script "
            "n'aurait alors rien mesure")
    return rows


def check_expected_behaviour(rows):
    """Ce que le script doit trouver s'il branche vraiment les deux variantes.

    Sans ces assertions, un fixed_curl qui ne se propagerait pas produirait
    des tableaux d'ecarts nuls, parfaitement lisibles et parfaitement faux.
    """
    gaps = np.array([r["score_max_gap"] for r in rows])
    agree = np.array([r["decision_agreement"] for r in rows])

    assert np.all(np.isfinite(gaps)), "des ecarts non finis dans le balayage"
    assert np.max(gaps) > 1e-6, (
        f"les deux conventions donnent le meme score partout (ecart max "
        f"{np.max(gaps):.3e}) : le drapeau ne branche rien")
    assert np.all(agree <= 1.0 + 1e-12) and np.all(agree >= 0.0)

    k = [r.get("K_plaquettes_rel_gap") for r in rows
         if r.get("K_plaquettes_rel_gap") is not None]
    assert k and max(k) > 1e-6, (
        "K_plaquettes est le canal ou entre le rotationnel : il ne peut pas "
        "etre identique dans les deux conventions")

    # Les metriques sans seuil doivent etre definies partout : elles sont
    # celles sur lesquelles la conclusion s'appuie.
    for key in ("spearman_legacy", "spearman_fixed",
                "f1_matched_legacy", "f1_matched_fixed"):
        vals = np.array([r[key] for r in rows], dtype=float)
        assert np.all(np.isfinite(vals)), (
            f"{key} contient des valeurs non definies : le score de patch "
            "est constant sur au moins un instantane, la comparaison de "
            "classement n'y veut alors rien dire")

    usable = [r for r in rows if not r["degenerate"]]
    return dict(
        n_rows=len(rows),
        n_usable=len(usable),
        n_degenerate=len(rows) - len(usable),
        score_max_gap=float(np.max(gaps)),
        decision_agreement_min=float(np.min(agree)),
        decision_agreement_mean=float(np.mean(agree)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", nargs="+", default=list(SCENARIOS))
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--dim", type=int, default=8,
                   help="patches par cote ; dim^2 patches au total")
    p.add_argument("--re", type=int, default=400)
    p.add_argument("--n-snaps", type=int, default=6)
    p.add_argument("--spinup", type=int, default=40)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--quantile", type=float, default=0.75,
                   help="quantile du label « patch dur » (le meme pour les "
                        "deux variantes, donc sans effet sur l'ecart)")
    p.add_argument("--mapper", choices=["v1", "v2"], default="v2")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    started = provenance.start()
    rows = run(args)
    summary = check_expected_behaviour(rows)

    keys = ["score_max_gap", "score_pearson", "spearman_legacy",
            "spearman_fixed", "f1_matched_legacy", "f1_matched_fixed",
            "K_plaquettes_rel_gap"]
    print(f"\n{'scenario':17s} {'snap':>4s} " +
          " ".join(f"{k.replace('_', ' '):>17s}" for k in keys))
    for r in rows:
        print(f"{r['scenario']:17s} {r['snap']:4d} " +
              " ".join(f"{r.get(k, float('nan')):17.5f}" for k in keys))

    _sd = (lambda a: a.std(ddof=1) if a.size > 1 else float("nan"))

    print("\n--- classement des patches, sans seuil (metrique principale) ---")
    verdicts = {}
    for name, kl, kf in (("Spearman vs durete", "spearman_legacy",
                          "spearman_fixed"),
                         ("F1 a budget appraie", "f1_matched_legacy",
                          "f1_matched_fixed")):
        a = np.array([r[kl] for r in rows], dtype=float)
        b = np.array([r[kf] for r in rows], dtype=float)
        d = b - a
        obs, lo, hi = bootstrap_delta_ci(rows, kl, kf, seed=args.seed)
        verdicts[kf] = dict(observed=obs, ci_low=lo, ci_high=hi,
                            verdict=verdict(lo, hi))
        print(f"{name:22s} historique {a.mean():+.4f} +- {_sd(a):.4f}   "
              f"corrige {b.mean():+.4f} +- {_sd(b):.4f}   "
              f"delta {d.mean():+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  "
              f"n={d.size}")
        print(f"{'':22s} -> {verdicts[kf]['verdict']}")

    print("\n--- au seuil entraine (diagnostic seulement) ---")
    print(f"{summary['n_degenerate']}/{summary['n_rows']} instantanes "
          "degeneres : un des deux bras y decide toujours pareil.")
    if summary["n_usable"]:
        usable = [r for r in rows if not r["degenerate"]]
        a = np.array([r["f1_thr_legacy"] for r in usable])
        b = np.array([r["f1_thr_fixed"] for r in usable])
        print(f"F1 au seuil, sur les {len(usable)} non degeneres : "
              f"historique {a.mean():.4f}, corrige {b.mean():.4f}, "
              f"delta {(b - a).mean():+.4f}")
    print(f"accord des decisions  {summary['decision_agreement_mean']:.4f} "
          f"(min {summary['decision_agreement_min']:.4f})")

    out = os.path.join(
        RESULTS_DIR,
        f"h1_curl_convention_gap_N{args.N}_dim{args.dim}_{args.mapper}.npz")
    payload = {k: np.array([r.get(k, np.nan) for r in rows])
               for k in rows[0] if k != "scenario"}
    payload["scenario"] = np.array([r["scenario"] for r in rows])
    np.savez_compressed(
        out,
        summary=json.dumps(summary),
        verdicts=json.dumps(verdicts),
        spearman_delta_mean=float(np.mean([r["spearman_fixed"] - r["spearman_legacy"]
                                           for r in rows])),
        f1_matched_delta_mean=float(np.mean([r["f1_matched_fixed"] - r["f1_matched_legacy"]
                                             for r in rows])),
        provenance=json.dumps(provenance.finish(started)),
        cli_args=json.dumps(vars(args)),
        argv=json.dumps(sys.argv),
        **payload)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
