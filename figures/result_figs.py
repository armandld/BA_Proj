#!/usr/bin/env python3
"""
Generate the two result figures for the Q-HAS falsification study.

Reads numbers from the full-run log files (N=256, 4 scenarios, seed=0)
that are checked into `attic/logs/Result_phase*.txt`. The numbers are reported
to 3 decimals -- more than adequate for a bar chart.

Outputs:
  results/figures/fig1_ceiling_bar.png    -- classical / SA / QAOA-equiv /
                                     mean-field / neighbourhood / LOSO
  results/figures/fig2_loso_scatter.png   -- per-scenario F1_site vs F1_class
                                     under LOSO, with x=y identity line
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.abspath(os.path.join(HERE, "..")), "results", "figures")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------
# Fig 1. Ceiling bar chart
# ---------------------------------------------------------------
# Numbers from logs/Result_phase7.txt (SA, classical) and
# logs/Result_phase11.txt (random split) and logs/Result_phase_end.txt (LOSO).

bar_labels = [
    "Classical\nAMR",
    "Simulated\nannealing\n(on v2 H)",
    "Mean-field\nceiling\n(random split)",
    "Neighbourhood\nceiling\n(random split)",
    "Mean-field\nceiling\n(LOSO)",
    "Neighbourhood\nceiling\n(LOSO)",
]
bar_values = [0.409, 0.336, 0.989, 0.991, 0.191, 0.215]
bar_errors = [0.0,   0.0,   0.0,   0.0,   0.152, 0.142]
bar_colors = [
    "#888888", "#aaaaaa",             # classical / SA
    "#3B82F6", "#1D4ED8",              # random-split ceilings
    "#EF4444", "#B91C1C",              # LOSO ceilings
]

fig, ax = plt.subplots(figsize=(9.5, 4.8))
x = np.arange(len(bar_labels))
bars = ax.bar(x, bar_values, yerr=bar_errors, color=bar_colors,
              edgecolor="black", linewidth=0.6,
              error_kw=dict(ecolor="black", lw=1.0, capsize=4))

for rect, v in zip(bars, bar_values):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.025,
            f"{v:.2f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold")

ax.axhline(0.409, color="#444444", linestyle="--", linewidth=0.8,
           alpha=0.5)
ax.text(5.35, 0.415, "classical baseline", fontsize=8,
        color="#444444", ha="right")

ax.set_xticks(x)
ax.set_xticklabels(bar_labels, fontsize=9)
ax.set_ylabel(r"$F_1$ (hard-patch detection)", fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_title("Q-HAS falsification: ceilings collapse under cross-scenario evaluation",
             fontsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()
out1 = os.path.join(OUT, "fig1_ceiling_bar.png")
plt.savefig(out1, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {out1}")


# ---------------------------------------------------------------
# Fig 2. Per-scenario LOSO scatter
# ---------------------------------------------------------------
# From logs/Result_phase_end.txt
scenarios = ["orszag_tang", "harris_tearing", "kelvin_helmholtz", "mhd_rotor"]
f1_class_loso = np.array([0.264, 0.400, 0.400, 0.672])
f1_site_loso  = np.array([0.327, 0.000, 0.353, 0.084])
f1_sten_loso  = np.array([0.226, 0.000, 0.400, 0.233])
f1_learn_loso = np.array([0.300, 0.653, 0.357, 0.253])  # phase 11c LOSO

fig, ax = plt.subplots(figsize=(6.6, 6.0))

# diagonal (x = y)
lim = 0.85
ax.plot([0, lim], [0, lim], ls="--", lw=1, color="#888888",
        label="F$_1$(learned H) = F$_1$(classical)")
ax.fill_between([0, lim], [0, lim], [0, 0], color="#EF4444", alpha=0.06,
                label="classical wins")
ax.fill_between([0, lim], [0, lim], [lim, lim], color="#10B981", alpha=0.06,
                label="learned H wins")

# mean-field GBT site ceiling under LOSO
ax.scatter(f1_class_loso, f1_site_loso, s=110, marker="o",
           c="#EF4444", edgecolor="black", linewidth=0.6,
           label="mean-field GBT (site, 9 feats)", zorder=3)
for sc, xc, yc in zip(scenarios, f1_class_loso, f1_site_loso):
    ax.annotate(sc, (xc, yc), xytext=(7, 4), textcoords="offset points",
                fontsize=9)

# stencil GBT
ax.scatter(f1_class_loso, f1_sten_loso, s=80, marker="s",
           c="#3B82F6", edgecolor="black", linewidth=0.6,
           label="neighbourhood GBT (stencil, 45 feats)", zorder=3, alpha=0.9)

# learned linear H (phase 11c)
ax.scatter(f1_class_loso, f1_learn_loso, s=70, marker="^",
           c="#10B981", edgecolor="black", linewidth=0.6,
           label="learned linear H (phase 11c)", zorder=3, alpha=0.9)

ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel(r"$F_1$ classical indicator (held-out scenario)", fontsize=11)
ax.set_ylabel(r"$F_1$ learned H / ceiling (held-out scenario)", fontsize=11)
ax.set_title("LOSO collapse: every held-out instability class\n"
             "puts the ceiling below or at the classical baseline",
             fontsize=11)
ax.legend(loc="upper left", fontsize=9, frameon=True)
ax.grid(linestyle=":", alpha=0.4)
plt.tight_layout()
out2 = os.path.join(OUT, "fig2_loso_scatter.png")
plt.savefig(out2, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {out2}")

print("Done.")
