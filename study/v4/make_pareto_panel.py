#!/usr/bin/env python3
"""
V4 - Figure : panneau multi-folds de la frontiere erreur-cout (Level 3).

Forme choisie : PETITS MULTIPLES, un panneau par classe tenue. Chaque
panneau reprend exactement la grammaire de `make_pareto_figure` (frontiere
classique en contexte, point Q-HAS en sujet, ecart vertical au budget
egal) ; seule la mise en page change. Le lecteur compare la POSITION de
Q-HAS par rapport a sa propre frontiere, panneau par panneau.

Choix delibere : l'axe des ERREURS est INDEPENDANT par panneau. Les
classes d'instabilite n'ont ni la meme dynamique ni la meme amplitude
d'erreur ; un axe partage ecraserait trois panneaux pour en servir un et
suggererait une comparabilite inter-classes qui n'existe pas. L'axe des
COUTS, lui, est partage : `patch_ratio` est la meme grandeur bornee [0,1]
partout, et c'est le long de cet axe que se lit l'appariement de budget.
Le nombre annote dans chaque panneau (rapport a la frontiere) est
sans dimension et reste, lui, comparable d'un panneau a l'autre.

Palette, encres et fond : importes de `make_pareto_figure` (slots
categoriels 1 et 2 valides par `scripts/validate_palette.js --mode light`).
Aucune couleur, aucune fonction n'est redefinie ici.

Sortie : figures_v4/pareto_panel.pdf / .png / .csv
Usage :
  python study/v4/make_pareto_panel.py
  python study/v4/make_pareto_panel.py --folds ot kh rotor tearing
"""
import argparse, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from make_pareto_figure import (
    C_FRONTIER, C_QHAS, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE,
    interp_frontier, load_points,
)

# Intitules lisibles : le code de fold n'est pas un nom de physique.
FOLD_TITLES = {
    "ot": "Orszag–Tang",
    "kh": "Kelvin–Helmholtz",
    "rotor": "MHD rotor",
    "tearing": "Harris tearing",
}


def available_folds(results_dir, folds):
    """Ne garde que les folds dont la comparaison budget-appariee existe."""
    keep = []
    for f in folds:
        p = os.path.join(results_dir, f"t15b_budget_matched_{f}.json")
        if os.path.exists(p):
            keep.append(f)
    return keep


def _style_axes(ax):
    """Grille et cadre recessifs, identiques a la figure mono-fold."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cd")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)


def draw_panel(ax, front, q, fold):
    """Un panneau. Retourne (q_ref, ratio) pour la table d'accompagnement."""
    xs = [r["patch"] for r in front]
    ys = [r["phys"] for r in front]
    q_ref = interp_frontier(front, q["patch"])
    ratio = q["phys"] / q_ref if q_ref > 0 else float("nan")

    _style_axes(ax)
    ax.plot(xs, ys, "-", color=C_FRONTIER, linewidth=1.8, zorder=3,
            label="Classical AMR: attainable error–cost frontier")
    ax.plot(xs, ys, "o", color=C_FRONTIER, markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
    ax.plot([q["patch"], q["patch"]], [q_ref, q["phys"]],
            linestyle=(0, (3, 3)), color=INK_MUTED, linewidth=1.2, zorder=2)
    ax.plot([q["patch"]], [q["phys"]], "D", color=C_QHAS, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=5,
            label="Q-HAS (closed loop, held-out class)")

    # Les deux etiquettes sont EMPILEES AU-DESSUS du marqueur, en unites de
    # points. Ancrer le rapport au milieu de l'ecart semblait naturel, mais
    # cet ecart peut etre minuscule en unites d'axe : sur Kelvin-Helmholtz
    # la frontiere est plate vers 0.002 et le texte tombait dessus. Q-HAS
    # etant toujours AU-DESSUS de la frontiere (c'est la lecture meme de la
    # figure), le demi-plan superieur est le seul degage a coup sur.
    # L'etiquette directe reste obligatoire : l'identite d'un point n'est
    # jamais portee par la seule couleur.
    if np.isfinite(ratio):
        ax.annotate(f"{ratio:.1f}× worse\nat equal budget",
                    xy=(q["patch"], q["phys"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom",
                    color=INK_SECONDARY, fontsize=8)
        y_lab = 34          # au-dessus des deux lignes du rapport
    else:
        y_lab = 11
    ax.annotate("Q-HAS", xy=(q["patch"], q["phys"]),
                xytext=(0, y_lab), textcoords="offset points", ha="center",
                color=INK_PRIMARY, fontsize=8.5, weight="bold")

    ax.set_title(FOLD_TITLES.get(fold, fold), color=INK_PRIMARY,
                 fontsize=10, weight="bold", loc="left", pad=6)
    ax.set_xlim(-0.03, 1.03)
    # marge haute : les deux etiquettes empilees occupent ~40 pt
    # au-dessus du marqueur, il faut donc de la place meme quand
    # Q-HAS est le point le plus haut du panneau
    y_top = max(max(ys), q["phys"] * 1.55) * 1.12
    ax.set_ylim(0.0, y_top)   # une erreur relative negative n'existe pas
    return q_ref, ratio


def build_panel(records, out_dir, ncols=2):
    """Mise en page a marges RESERVEES explicitement, en pouces.

    On n'utilise ni `tight_layout` ni `bbox_inches="tight"` : un texte de
    figure non enveloppe (la note de bas de figure, le titre) etirerait la
    boite englobante et deformerait la planche — d'autant plus qu'avec un
    seul panneau le titre est plus large que le panneau lui-meme. Les
    bandeaux haut et bas sont donc dimensionnes en pouces, et les textes
    sont enveloppes a la largeur reellement disponible.
    """
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(records)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    # bandeaux, en pouces. La hauteur du bandeau haut depend du nombre de
    # lignes du titre APRES enveloppement : la calculer avant reviendrait a
    # deborder des que le titre passe sur deux lignes.
    bot_in, left_in, right_in = 0.92, 0.72, 0.18
    W = max(3.5 * ncols + left_in + right_in, 6.9)

    # Le titre porte la PORTEE reelle : tant que les quatre folds ne sont
    # pas tous mesures, « every held-out class » affirmerait plus que ce
    # que la planche montre.
    n_total = len(FOLD_TITLES)
    scope = ("on every held-out class" if n == n_total
             else f"on each held-out class measured ({n} of {n_total})")
    title = f"Q-HAS lies above the classical error\u2013cost frontier {scope}"
    note = ("Per-panel error axes are independent: instability classes "
            "differ in dynamic range, and a shared axis would imply a "
            "cross-class comparability that does not hold. The annotated "
            "ratio is dimensionless and is comparable across panels.")
    # ~9.6 caracteres par pouce a 12.5 pt gras ; marge droite reservee
    title_lines = textwrap.wrap(title, width=max(24, int((W - 0.25) * 9.6)))
    note_lines = textwrap.wrap(note, width=max(60, int(W * 21)))

    title_in = 0.26 * len(title_lines)
    legend_in = 0.46          # legende + respiration avant le premier titre
    top_in = 0.30 + title_in + legend_in
    bot_in = 0.60 + 0.14 * len(note_lines)
    H = 2.85 * nrows + top_in + bot_in

    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(nrows, ncols,
                          left=left_in / W, right=1.0 - right_in / W,
                          top=1.0 - top_in / H, bottom=bot_in / H,
                          wspace=0.28, hspace=0.42)

    rows, axes = [], []
    for i, rec in enumerate(records):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        axes.append(ax)
        q_ref, ratio = draw_panel(ax, rec["front"], rec["q"], rec["fold"])
        rows.append({"fold": rec["fold"], "q": rec["q"],
                     "q_ref": q_ref, "ratio": ratio})

    # un seul intitule par grandeur : le repeter sous chaque panneau
    # n'ajoute aucune information
    fig.text(0.5 * (left_in / W + 1.0 - right_in / W),
             (bot_in - 0.34) / H,
             "Compute:  refined-pixel ratio  (lower is cheaper)",
             color=INK_SECONDARY, fontsize=9.5, ha="center", va="center")
    fig.text(0.13 / W, 0.5 * (bot_in / H + 1.0 - top_in / H),
             "Error:  relative $L_2$ vs DNS  (lower is better)",
             color=INK_SECONDARY, fontsize=9.5, ha="center", va="center",
             rotation=90)

    fig.text(0.012, 1.0 - (0.22 + 0.5 * title_in) / H,
             "\n".join(title_lines), color=INK_PRIMARY, fontsize=12.5,
             weight="bold", ha="left", va="center", linespacing=1.25)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, ncol=2, frameon=False, fontsize=8.5,
                     loc="center",
                     bbox_to_anchor=(0.5, 1.0 - (0.30 + title_in
                                                 + 0.18) / H))
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    fig.text(0.012, (0.10 + 0.07 * len(note_lines)) / H,
             "\n".join(note_lines), fontsize=7, color=INK_MUTED,
             ha="left", va="center", linespacing=1.3)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "pareto_panel")
    fig.savefig(base + ".pdf", facecolor=SURFACE)
    fig.savefig(base + ".png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return base, rows


def main():
    p = argparse.ArgumentParser(
        description="V4: multi-fold Pareto frontier panel")
    from config import RESULTS_DIR
    p.add_argument("--folds", nargs="+",
                   default=["ot", "kh", "rotor", "tearing"])
    p.add_argument("--ncols", type=int, default=2)
    p.add_argument("--out-dir",
                   default=os.path.join(_HERE, "..", "..", "figures_v4"))
    args = p.parse_args()

    folds = available_folds(RESULTS_DIR, args.folds)
    missing = [f for f in args.folds if f not in folds]
    if missing:
        print(f"  folds without a budget-matched run (skipped): "
              f"{', '.join(missing)}")
    if not folds:
        raise SystemExit("no budget-matched fold available; run t15b first.")

    records = []
    for f in folds:
        front, q, tuned, d = load_points(RESULTS_DIR, f)
        records.append({"fold": f, "front": front, "q": q, "tuned": tuned})

    base, rows = build_panel(records, args.out_dir, ncols=args.ncols)

    # table d'accompagnement : la figure n'est jamais le seul acces aux
    # nombres qu'elle montre
    with open(base + ".csv", "w") as fh:
        fh.write("fold,qhas_patch_ratio,qhas_phys,frontier_phys_at_budget,"
                 "ratio\n")
        for r in rows:
            fh.write(f"{r['fold']},{r['q']['patch']:.6f},"
                     f"{r['q']['phys']:.6f},{r['q_ref']:.6f},"
                     f"{r['ratio']:.4f}\n")

    print(f"  panels: {len(rows)}  ({', '.join(r['fold'] for r in rows)})")
    for r in rows:
        print(f"    {r['fold']:>8}: Q-HAS phys={r['q']['phys']:.4f} at "
              f"patch={r['q']['patch']:.4f}; frontier={r['q_ref']:.4f}; "
              f"ratio={r['ratio']:.2f}x")
    print(f"  saved: {os.path.basename(base)}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
