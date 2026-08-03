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

Sortie : figures_v4/pareto_frontier_{fold}.pdf et .png (+ .csv des points)
Usage :
  python study/v4/make_pareto_figure.py --fold ot
"""
import argparse, json, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

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
    ax.plot([q["patch"]], [q["phys"]], "D", color=C_QHAS, markersize=10,
            markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5,
            label="Q-HAS (closed loop, held-out class)")

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
    ax.set_ylabel("Error:  relative $L_2$ vs DNS  (lower is better)",
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
                   default=os.path.join(_HERE, "..", "..", "figures_v4"))
    args = p.parse_args()

    front, q, tuned, d = load_points(RESULTS_DIR, args.fold)
    base, q_ref = build_figure(front, q, tuned, args.fold, args.out_dir)

    # table de donnees accompagnant la figure (accessibilite : la figure
    # n'est jamais le seul acces aux nombres)
    csv = base + ".csv"
    with open(csv, "w") as fh:
        fh.write("series,threshold,patch_ratio,phys_score\n")
        for r in front:
            fh.write(f"classical,{r['thr']:.6f},{r['patch']:.6f},"
                     f"{r['phys']:.6f}\n")
        fh.write(f"qhas,,{q['patch']:.6f},{q['phys']:.6f}\n")

    print(f"  frontier points: {len(front)}")
    print(f"  Q-HAS           : patch={q['patch']:.4f} phys={q['phys']:.4f}")
    print(f"  frontier at that budget (interpolated): {q_ref:.4f}")
    print(f"  ratio Q-HAS / frontier = {q['phys'] / q_ref:.2f}x worse")
    print(f"  saved: {os.path.basename(base)}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
