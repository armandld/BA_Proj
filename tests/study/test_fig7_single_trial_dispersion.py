"""D-105 — `fig7_physical_fidelity.py` imprimait la dispersion d'un essai
unique comme si elle avait ete mesuree, a une echelle ou la grandeur ne
s'ecrit pas.

Le fichier tourne a `N_TRIALS = 1` et annoncait « Multiple trials with error
bands for statistical confidence ». Deux consequences, toutes deux dans la
classe « valeur plausible mais fausse » :

1. `np.std(x, axis=0)` sur UN echantillon vaut **0,0**, sans avertissement
   (avec `ddof=1` la meme quantite vaut `nan` et previent). La ligne de
   resume imprimait `+/-0.000000` : une dispersion nulle *mesuree* est
   indiscernable d'une dispersion *jamais mesuree*. La bande
   `fill_between(x, mu-0, mu+0)` etait tracee de largeur nulle pour la
   meme raison.
2. Le format `%.6f` sur une grandeur de l'ordre de 1e-06 imprimait
   `QA=0.000001  CL=0.000001` — les deux bras indiscernables. La colonne
   correspondante de la figure est tracee en `set_yscale('log')` : le
   resume imprime contredisait l'echelle de son propre axe.

Mesure (`init_harris_tearing`, N=256, warmup=80, 3 pas d'AMR, `trial=0`,
seuils du depot) : `l2_qa = l2_cl = 8,182e-07`, soit

| | avant | apres |
|---|---|---|
| ligne imprimee | `QA=0.000001+/-0.000000` | `QA=8.1819e-07 (1 essai, dispersion non mesuree)` |
| chiffres significatifs sur la valeur | 1 | 5 |
| largeur de la bande, 1 essai | 0,0 (tracee) | pas de bande |

Aucun nombre publie ne bouge : aucune figure `results/figures/fig7_*` n'est
committee dans ce depot, et le calcul physique est inchange.

Les tests portent sur le comportement du fichier committe (fonctions
extraites par AST puis executees), pas sur son texte source.
"""
import ast
import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_FIG7 = os.path.join(_REPO_ROOT, "figures", "v1_legacy",
                     "fig7_physical_fidelity.py")

# La vraie courbe mesuree a trial=0 (init_harris_tearing, N=256, warmup=80).
L2_MESUREE = np.array([[2.770709869846107e-07,
                        5.497559945992208e-07,
                        8.181868983482631e-07]])


def _load(*names):
    """Extrait des fonctions du fichier committe sans executer le script.

    fig7 produit sa figure a l'import : on ne peut pas l'importer.
    """
    with open(_FIG7, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=_FIG7)
    wanted = []
    for name in names:
        fn = next((n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == name), None)
        if fn is None:                                    # pragma: no cover
            pytest.fail(
                f"fig7_physical_fidelity.py n'expose plus `{name}` : "
                "l'etat d'avant D-105, ou la correction a ete defaite.")
        wanted.append(fn)
    module_ast = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    g = {"np": np}
    exec(compile(module_ast, _FIG7, "exec"), g)           # noqa: S102
    return [g[n] for n in names]


def _ancienne_ligne(all_curves):
    """La ligne d'avant D-105, reproduite mot pour mot."""
    a = np.asarray(all_curves)
    return f"{a[:, -1].mean():.6f}+/-{a[:, -1].std():.6f}"


# ══════════════════════════════════════════════════════════════════
#  1. L'ancien comportement, epingle
# ══════════════════════════════════════════════════════════════════

def test_lancienne_ligne_imprimait_zero_pour_les_deux_champs():
    """Sur la vraie mesure : la valeur ET la dispersion tombent a zero."""
    ligne = _ancienne_ligne(L2_MESUREE)
    assert ligne == "0.000001+/-0.000000", (
        f"reproduction de l'ancienne ligne : {ligne!r}")
    # la vraie valeur, pour que l'ecart se voie
    assert float(L2_MESUREE[0, -1]) == pytest.approx(8.181868983482631e-07)


def test_lecart_type_dun_echantillon_unique_est_zero_sans_prevenir():
    """La raison structurelle : `ddof=0` rend 0,0, `ddof=1` rend nan.

    C'est l'estimateur qui SAIT qu'il ne peut pas mesurer qui previent ;
    celui qui etait utilise ne prevenait pas.
    """
    assert float(np.std(L2_MESUREE[:, -1])) == 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        with pytest.warns(RuntimeWarning):
            nan = np.std(L2_MESUREE[:, -1], ddof=1)
    assert np.isnan(nan)


# ══════════════════════════════════════════════════════════════════
#  2. La garantie apres correction
# ══════════════════════════════════════════════════════════════════

def test_un_essai_unique_est_dit_comme_tel_et_a_la_bonne_echelle():
    (_final_l2,) = _load("_final_l2")
    ligne = _final_l2(L2_MESUREE)
    assert "8.1819e-07" in ligne, (
        f"la valeur n'est plus lisible a son echelle : {ligne!r}")
    assert "1 essai" in ligne and "non mesur" in ligne, (
        f"l'absence de dispersion n'est pas dite : {ligne!r}")
    assert "+/-" not in ligne, (
        f"une dispersion est encore imprimee sur un essai unique : {ligne!r}")


def test_deux_essais_rendent_une_dispersion_reelle():
    """« Ne rien imprimer » ne doit pas devenir la reponse a tout."""
    (_final_l2,) = _load("_final_l2")
    deux = np.array([[1.0e-06, 2.0e-06, 3.0e-06],
                     [1.0e-06, 2.0e-06, 5.0e-06]])
    ligne = _final_l2(deux)
    assert "+/-" in ligne and "n=2" in ligne, ligne
    # ecart-type non biaise de {3e-06, 5e-06} = 1,4142e-06
    assert "1.4142e-06" in ligne, ligne
    assert "4.0000e-06" in ligne, ligne


def test_la_bande_nest_pas_tracee_sur_un_essai_unique():
    (_plot_with_band,) = _load("_plot_with_band")
    fig, ax = plt.subplots()
    try:
        _plot_with_band(ax, np.arange(L2_MESUREE.shape[1]), L2_MESUREE,
                        "C0", "Q-HAS")
        assert len(ax.collections) == 0, (
            "une bande de largeur nulle est tracee sur un essai unique")
        assert len(ax.lines) == 1
    finally:
        plt.close(fig)


def test_la_bande_est_tracee_des_deux_essais():
    """La correction ne doit pas supprimer la bande quand elle a un sens."""
    (_plot_with_band,) = _load("_plot_with_band")
    fig, ax = plt.subplots()
    try:
        deux = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 4.0]])
        _plot_with_band(ax, np.arange(3), deux, "C0", "Q-HAS")
        assert len(ax.collections) == 1, "la bande a disparu a n=2"
        assert "n=2" in ax.lines[0].get_label()
    finally:
        plt.close(fig)
