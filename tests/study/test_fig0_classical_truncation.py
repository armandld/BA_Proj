"""D-95 — `fig0_pareto_lambda.py` : le bras classique tronque a la fenetre du bras quantique.

`v_min = min(q_scores)` servait a deux choses : l'echelle de couleur commune
(son role) et un filtre sur les DONNEES classiques. Consequence structurelle :
tout essai classique meilleur que TOUT le quantique tombait hors fenetre et
etait jete — puis le front de Pareto classique et l'etoile « Best Classical »
etaient calcules sur ce reste tronque. Le biais va toujours dans le meme sens,
contre le bras classique, du cote ou la comparaison se joue.

Le champ qui SEPARE : il faut un essai classique STRICTEMENT meilleur que le
meilleur quantique. Sur `harris_tearing`, aucun essai classique ne tombe sous
la fenetre — un test ecrit sur ce seul scenario serait passe sans rien
verifier. Le cas synthetique ci-dessous construit l'ecart explicitement.

Ces tests echouent sur la version d'avant la correction.
"""
import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_V1_LEGACY = os.path.join(_REPO_ROOT, "figures", "v1_legacy")
if _V1_LEGACY not in sys.path:
    sys.path.insert(0, _V1_LEGACY)

import fig0_pareto_lambda as M  # noqa: E402


def _rows(pairs, phys_col="phys_x", patch_col="patch_x"):
    return {("t", 0.40): [{phys_col: str(p), patch_col: str(r)} for p, r in pairs]}


def _axe_classique(fig):
    for ax in fig.axes:
        if ax.get_title().endswith("Classical"):
            return ax
    raise AssertionError("pas de panneau Classical dans la figure")


def _etiquette_best(ax):
    for t in ax.get_legend().get_texts():
        if t.get_text().startswith("Best Classical"):
            return t.get_text()
    raise AssertionError("pas d'etiquette « Best Classical »")


def _n_points_traces(ax):
    """Nombre de points du nuage — l'etoile est sa propre collection a 1 point."""
    return max(len(c.get_offsets()) for c in ax.collections)


@pytest.fixture
def cas_qui_separe():
    """Un classique STRICTEMENT meilleur que tout le quantique, un pire.

    lam_cost = 0.40, score = phys + 0.40 * patch.
      quantique : S = 0.50 et 0.70
      classique : S = 0.20 (le meilleur de tous), 0.60 (dans la fenetre),
                  0.90 (au-dessus)
    """
    qd = _rows([(0.50, 0.0), (0.70, 0.0)])
    cd = _rows([(0.20, 0.0), (0.60, 0.0), (0.90, 0.0)])
    return qd, cd


def test_le_meilleur_essai_classique_n_est_plus_jete(cas_qui_separe):
    """Avant D-95 : l'etoile annoncait S=0.6000 — le meilleur classique, jete."""
    qd, cd = cas_qui_separe
    fig = M.plot_pareto_scenario(qd, cd, "phys_x", "patch_x", "cas")
    try:
        ax = _axe_classique(fig)
        assert _etiquette_best(ax) == "Best Classical (S=0.2000)"
        assert _n_points_traces(ax) == 3, "un essai classique a ete jete"
    finally:
        plt.close(fig)


def test_aucun_point_classique_n_est_jete_sur_la_planche_agregee(cas_qui_separe):
    """Second site : `plot_grouped_pareto`, meme defaut."""
    qd, cd = cas_qui_separe
    scen = {"x": {"phys": "phys_x", "patch": "patch_x"}}
    fig = M.plot_grouped_pareto(qd, cd, scen, "cas", lam_cost=M.TARGET_LAMBDA)
    try:
        ax = _axe_classique(fig)
        assert _etiquette_best(ax) == "Best Classical (S=0.2000)"
        assert _n_points_traces(ax) == 3
    finally:
        plt.close(fig)


def test_lechelle_de_couleur_couvre_les_deux_bras(cas_qui_separe):
    """La correction ne doit pas re-creer le probleme sous une autre forme :
    restituer les points sans elargir l'echelle les ferait tous saturer.
    """
    qd, cd = cas_qui_separe
    fig = M.plot_pareto_scenario(qd, cd, "phys_x", "patch_x", "cas")
    try:
        ax = _axe_classique(fig)
        nuage = max(ax.collections, key=lambda c: len(c.get_offsets()))
        vmin, vmax = nuage.get_clim()
        assert vmin <= 0.20 + 1e-12, "vmin=%r exclut le meilleur classique" % vmin
        assert vmax >= 0.90 - 1e-12, "vmax=%r exclut le pire classique" % vmax
    finally:
        plt.close(fig)


# ── Regression sur les donnees gelees du depot ────────────────────────────
# Nombres mesures le 15 aout sur results/hyperparams/optuna_studies/, lambda 0.40.
# Ecrits ici pour qu'une derive se voie. « annonce_avant » est la valeur que
# la version d'avant la correction affichait.
_REEL = {
    "kelvin_helmholtz": dict(vrai_min=0.129020, annonce_avant=0.306590, n_jetes=56),
    "harris_tearing":   dict(vrai_min=0.254429, annonce_avant=0.254429, n_jetes=0),
    "orszag_tang":      dict(vrai_min=0.326180, annonce_avant=0.348250, n_jetes=47),
    "mhd_rotor":        dict(vrai_min=0.183508, annonce_avant=0.192481, n_jetes=6),
}


@pytest.fixture(scope="module")
def donnees_reelles():
    if not os.path.isdir(M.TRAIN_DIR):
        pytest.fail("TRAIN_DIR absent : %s (voir D-94)" % M.TRAIN_DIR)
    qd = M.load_all_trials(M.TRAIN_DIR, M.QUANTUM_PATTERN)
    cd = M.load_all_trials(M.TRAIN_DIR, M.CLASSICAL_PATTERN)
    assert qd and cd, "balayage vide : aucun essai lu"
    return qd, cd


@pytest.mark.parametrize("sc_name", sorted(_REEL))
def test_donnees_reelles_letoile_annonce_le_vrai_minimum(donnees_reelles, sc_name):
    qd, cd = donnees_reelles
    info = M.SCENARIOS_ALL[sc_name]
    ref = _REEL[sc_name]

    _, _, c_scores = M._collect_points(cd, info["phys"], info["patch"], M.TARGET_LAMBDA)
    vrai_min = float(np.min(c_scores))
    assert vrai_min == pytest.approx(ref["vrai_min"], abs=5e-6), (
        "le minimum classique de %s a derive : %.6f mesure ici, %.6f consigne"
        % (sc_name, vrai_min, ref["vrai_min"]))

    fig = M.plot_pareto_scenario(qd, cd, info["phys"], info["patch"], sc_name)
    try:
        ax = _axe_classique(fig)
        annonce = float(_etiquette_best(ax).split("S=")[1].rstrip(")"))
        assert annonce == pytest.approx(vrai_min, abs=5e-5)
        assert _n_points_traces(ax) == len(c_scores), (
            "%d essais classiques collectes, %d traces"
            % (len(c_scores), _n_points_traces(ax)))
    finally:
        plt.close(fig)


@pytest.mark.parametrize("sc_name", sorted(_REEL))
def test_donnees_reelles_le_nombre_dessais_jetes_par_lancienne_regle(donnees_reelles, sc_name):
    """Epingle l'ancien comportement : combien d'essais classiques la fenetre
    quantique jetait, et de quel cote. Sans ce test, la correction pourrait
    etre defaite sans que rien ne le dise.
    """
    qd, cd = donnees_reelles
    info = M.SCENARIOS_ALL[sc_name]
    _, _, q_scores = M._collect_points(qd, info["phys"], info["patch"], M.TARGET_LAMBDA)
    _, _, c_scores = M._collect_points(cd, info["phys"], info["patch"], M.TARGET_LAMBDA)

    n_jetes = int(np.sum(c_scores < np.min(q_scores)))
    assert n_jetes == _REEL[sc_name]["n_jetes"], (
        "%s : %d essais classiques sous la fenetre quantique, %d consignes"
        % (sc_name, n_jetes, _REEL[sc_name]["n_jetes"]))
