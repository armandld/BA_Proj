#!/usr/bin/env python3
"""
V4 - Figure : frontiere erreur-cout classique et position de Q-HAS.

Forme choisie : nuage + ligne monotone, en mode EMPHASE. La frontiere
classique est le contexte (serie 1), le point Q-HAS est le sujet (serie 2).
Deux series -> legende obligatoire, et les deux points compares portent en
plus une etiquette directe : l'identite n'est jamais portee par la couleur
seule. Un seul axe par grandeur, aucune echelle secondaire.

Palette : slots categoriels 1 et 2 de la palette de reference
(#2a78d6 bleu, #eb6834 orange), valides par
`scripts/validate_palette.js --mode light` (tous les controles PASS :
bande de luminosite, plancher de chroma, separation CVD dE 24.7,
vision normale dE 33.6, contraste >= 3:1).

D-92 : `main()` prend desormais le point Q-HAS moyenne sur les tirages
repetes de T20 (repli sur le tirage unique de t15b, annonce comme tel) et
retire de la frontiere les points issus d'une trace avortee (audit T19) --
exactement les deux corrections que `pareto_panel.py` applique deja pour la
planche V4, mais que ce script, execute seul, ne faisait jamais : lance sans
elles, il reproduisait les rapports RETRACTES que `docs/RESULTS.md`
documente ("Figure updated"). Le CSV porte aussi le denominateur MESURE par
t15b (`matched_classical`), a cote de celui interpole sur la trace : les
deux repondent a des questions differentes, voir `pareto_panel.py` pour le
detail de pourquoi ils different.

Sortie : results/figures/pareto_frontier_{fold}.pdf et .png (+ .csv des points)
Usage :
  python figures/pareto_frontier.py --fold ot
"""
import argparse, json, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------

# Palette de reference (slots categoriels 1 et 2), mode clair.
C_FRONTIER = "#2a78d6"   # slot 1 — contexte : frontiere classique
C_QHAS = "#eb6834"       # slot 2 — sujet : point Q-HAS
INK_PRIMARY = "#1a1a19"
INK_SECONDARY = "#54534a"
INK_MUTED = "#8f8e85"
SURFACE = "#fcfcfb"


def load_points(results_dir, fold):
    """Lit le JSON budget-apparie et retourne (frontiere, qhas, tuned)."""
    path = os.path.join(results_dir, f"t15b_budget_matched_{fold}.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run t15b for fold {fold} first.")
    d = json.load(open(path))
    front = sorted(({"patch": r["patch_ratio"], "phys": r["phys_score"],
                     "thr": r["threshold"]} for r in d["trace"]),
                   key=lambda r: r["patch"])
    q = {"patch": d["qhas"]["patch_ratio"], "phys": d["qhas"]["phys_score"]}
    t = {"patch": d["tuned_classical"]["patch_ratio"],
         "phys": d["tuned_classical"]["phys_score"],
         "thr": d["matched_classical"]["threshold"]}
    return front, q, t, d


def interp_frontier(front, patch):
    """Erreur de la frontiere classique au budget `patch` (interpolation
    lineaire entre les deux points encadrants)."""
    xs = np.array([r["patch"] for r in front])
    ys = np.array([r["phys"] for r in front])
    return float(np.interp(patch, xs, ys))


def verified_qhas_point(results_dir, fold):
    """Point Q-HAS a partir des tirages REPETES de T20, ou None.

    D-92 : deplacee ici depuis `pareto_panel.py`, qui l'appelait deja pour
    corriger exactement le defaut que cette fonction meme decrit --
    `main()`, plus bas dans CE fichier, ne l'appelait pas. Une seule
    definition desormais ; `pareto_panel` l'importe d'ici.

    `t15b["qhas"]` est UN tirage unique d'un bras non deterministe (D11).
    On prend donc la moyenne des tirages ACHEVES (les avortes ne sont pas
    des points de mesure) et on rend aussi leur dispersion.
    """
    p = os.path.join(results_dir, f"t20_qhas_run_variance_{fold}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    ok = [r for r in d.get("qhas_runs", []) if r.get("completed")]
    if len(ok) < 2:
        return None
    ph = np.array([r["phys_score"] for r in ok], dtype=float)
    pa = np.array([r["patch_ratio"] for r in ok], dtype=float)
    return {"patch": float(pa.mean()), "phys": float(ph.mean()),
            "patch_sd": float(pa.std(ddof=1)),
            "phys_sd": float(ph.std(ddof=1)),
            "n": len(ok), "n_aborted": len(d.get("qhas_runs", [])) - len(ok)}


def load_trace_audit(results_dir):
    """Points de bissection dont la trajectoire a AVORTE, par fold.

    D-92 : deplacee ici depuis `pareto_panel.py`, meme raison que
    `verified_qhas_point` ci-dessus.
    """
    p = os.path.join(results_dir, "t19_budget_trace_audit.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {t["fold"]: [pt["threshold"] for pt in t["points"]
                        if not pt["completed"]]
            for t in d.get("traces", [])}


def drop_aborted(front, aborted_thresholds, tol=1e-9):
    """Retire de la frontiere les points marques avortes par l'audit.

    D-92 : deplacee ici depuis `pareto_panel.py`, meme raison que
    `verified_qhas_point` ci-dessus.
    """
    if not aborted_thresholds:
        return front, 0
    keep = [r for r in front
            if not any(abs(r.get("thr", float("nan")) - t) < tol
                       for t in aborted_thresholds)]
    return keep, len(front) - len(keep)


def build_figure(front, q, tuned, fold, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["patch"] for r in front]
    ys = [r["phys"] for r in front]
    q_ref = interp_frontier(front, q["patch"])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # --- grille et axes recessifs -----------------------------------
    ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cd")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)

    # --- serie 1 : frontiere classique (contexte) -------------------
    ax.plot(xs, ys, "-", color=C_FRONTIER, linewidth=2.0, zorder=3,
            label="Classical AMR: attainable error–cost frontier")
    ax.plot(xs, ys, "o", color=C_FRONTIER, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4)

    # --- ecart vertical au budget de Q-HAS ---------------------------
    ax.plot([q["patch"], q["patch"]], [q_ref, q["phys"]],
            linestyle=(0, (3, 3)), color=INK_MUTED, linewidth=1.4, zorder=2)
    ax.annotate(f"{q['phys'] / q_ref:.1f}× worse\nat equal budget",
                xy=(q["patch"], 0.5 * (q_ref + q["phys"])),
                xytext=(10, 0), textcoords="offset points",
                color=INK_SECONDARY, fontsize=9, va="center")

    # --- serie 2 : Q-HAS (sujet) -------------------------------------
    # D-92 : barres d'erreur quand `q` vient de `verified_qhas_point` (bras
    # non deterministe, D11) — sans elles la figure affirme une precision
    # qu'un tirage unique n'a pas, exactement ce que D-92 corrige.
    if q.get("n"):
        ax.errorbar([q["patch"]], [q["phys"]],
                    xerr=[q.get("patch_sd", 0.0)],
                    yerr=[q.get("phys_sd", 0.0)],
                    fmt="none", ecolor=C_QHAS, elinewidth=1.4,
                    capsize=4, capthick=1.4, zorder=4.5, alpha=0.85)
    ax.plot([q["patch"]], [q["phys"]], "D", color=C_QHAS, markersize=10,
            markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5,
            label=("Q-HAS (closed loop, held-out class): mean of "
                   f"{q['n']} runs ± sd" if q.get("n")
                   else "Q-HAS (closed loop, held-out class): single run"))

    # --- etiquettes directes sur les deux points compares ------------
    ax.annotate("Q-HAS", xy=(q["patch"], q["phys"]),
                xytext=(0, 14), textcoords="offset points",
                ha="center", color=INK_PRIMARY, fontsize=10, weight="bold")
    # etiquette directe du point apparie, avec ligne de rappel : sans elle
    # le texte parait designer le point voisin de la frontiere
    i_match = int(np.argmin([abs(x - q["patch"]) for x in xs]))
    ax.annotate(f"budget-matched\nclassical (thr {front[i_match]['thr']:.3f})",
                xy=(xs[i_match], ys[i_match]),
                xytext=(0, -38), textcoords="offset points",
                ha="center", color=INK_PRIMARY, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED,
                                linewidth=1.0,
                                shrinkA=2, shrinkB=6))

    ax.set_xlabel("Compute:  refined-pixel ratio  (lower is cheaper)",
                  color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Error:  instability-weighted relative $L_2$ vs DNS  "
                  "(lower is better)",
                  color=INK_SECONDARY, fontsize=10)
    ax.set_title(f"Q-HAS lies above the classical frontier  "
                 f"(Level-3 fold: {fold})",
                 color=INK_PRIMARY, fontsize=12, weight="bold",
                 loc="left", pad=12)
    ax.set_xlim(-0.03, 1.02)
    # marge basse elargie : l'etiquette du point apparie se loge sous la
    # frontiere sans percuter les graduations
    y_top = max(max(ys), q["phys"]) * 1.12
    ax.set_ylim(-0.105, y_top)
    # graduations a partir de 0 : une erreur relative negative n'existe pas
    ax.set_yticks(np.arange(0.0, y_top, 0.1))

    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    fig.text(0.005, 0.005,
             "Both arms: identical DNS trace, hot start, hybrid budget and "
             "depth; held-out class excluded from all tuning.",
             fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"pareto_frontier_{fold}")
    fig.savefig(base + ".pdf", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    return base, q_ref


def main():
    p = argparse.ArgumentParser(description="V4: Pareto frontier figure")
    from config import RESULTS_DIR
    p.add_argument("--fold", default="ot")
    p.add_argument("--out-dir",
                   default=os.path.join(_REPO_ROOT, "results", "figures"))
    args = p.parse_args()

    front, q, tuned, d = load_points(RESULTS_DIR, args.fold)

    # D-92 : deux corrections que `pareto_panel.py` applique deja (elle
    # importe `interp_frontier`/`load_points` d'ici, mais portait sa propre
    # copie de ces deux etapes) et que ce script, execute seul, ne faisait
    # jamais. Sans elles, `main()` reproduit les rapports RETRACTES —
    # 2,57x / 4,41x / 3,62x / 4,38x sur ot/kh/rotor/tearing — que
    # `docs/RESULTS.md` documente deja comme retires (voir "Figure
    # updated" : un tirage unique d'un bras non deterministe, gonfle de
    # 1,1 a 2,2x contre la moyenne sur 5 tirages, et une frontiere qui
    # pouvait inclure un point issu d'une trace avortee).
    audit = load_trace_audit(RESULTS_DIR)
    if audit is None:
        print("  WARNING: no t19 trace audit; frontier may include points "
              "from aborted runs")
    else:
        front, n_drop = drop_aborted(front, audit.get(args.fold, []))
        if n_drop:
            print(f"  dropped {n_drop} frontier point(s) from aborted runs "
                  f"(t19 audit)")

    qv = verified_qhas_point(RESULTS_DIR, args.fold)
    if qv is not None:
        print(f"  Q-HAS point = mean of {qv['n']} completed runs"
              + (f" ({qv['n_aborted']} aborted, excluded)"
                 if qv["n_aborted"] else ""))
        q = qv
    else:
        print("  Q-HAS point = SINGLE t15b draw (no repeated runs "
              "available) — the ratio below is one draw of a "
              "non-deterministic arm")

    base, q_ref = build_figure(front, q, tuned, args.fold, args.out_dir)

    # table de donnees accompagnant la figure (accessibilite : la figure
    # n'est jamais le seul acces aux nombres). D-92 : ajoute la ligne
    # `matched_classical` — le denominateur MESURE par t15b, different de
    # celui qu'annote la figure (interpole sur la trace). Les deux sont
    # justes, ils ne repondent pas a la meme question ; ecrire les deux
    # evite qu'un lecteur ait a deviner lequel il regarde (meme principe
    # que la colonne `ratio_vs_matched` de `pareto_panel.csv`).
    mc = d["matched_classical"]
    csv = base + ".csv"
    with open(csv, "w") as fh:
        fh.write("series,threshold,patch_ratio,phys_score\n")
        for r in front:
            fh.write(f"classical,{r['thr']:.6f},{r['patch']:.6f},"
                     f"{r['phys']:.6f}\n")
        fh.write(f"qhas,,{q['patch']:.6f},{q['phys']:.6f}\n")
        fh.write(f"matched_classical,{mc['threshold']:.6f},"
                 f"{mc['patch_ratio']:.6f},{mc['phys_score']:.6f}\n")

    ratio_vs_matched = q["phys"] / mc["phys_score"]
    print(f"  frontier points: {len(front)}")
    print(f"  Q-HAS           : patch={q['patch']:.4f} phys={q['phys']:.4f}")
    print(f"  frontier at that budget (interpolated): {q_ref:.4f}")
    print(f"  ratio Q-HAS / frontier (interpolated)  = "
          f"{q['phys'] / q_ref:.2f}x worse")
    print(f"  ratio Q-HAS / matched_classical (t15b) = "
          f"{ratio_vs_matched:.2f}x worse")
    print(f"  saved: {os.path.basename(base)}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
