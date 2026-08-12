"""T28/T29 — le label ne doit plus etre un rang intra-scenario, et aucune
conclusion de transfert ne doit sortir sans intervalle de confiance.

Deux defauts sont verrouilles ici :

1. `phase2_hard_patches.py` seuille au percentile 75 DE CHAQUE SCENARIO, donc
   chaque scenario a exactement 25 % de patches durs et les seuils different
   d'un facteur 2.8. Le label devient un rang intra-scenario, que le LOSO
   demande de predire sans jamais montrer le seuil du scenario tenu a
   l'ecart. T28 produit la variante a seuil global.

2. `phase11b_loso.py` imprime « neighbourhood couplings help for transfer »
   a partir d'une moyenne sur quatre folds d'ecart-type 0.29, dont deux sont
   des predicteurs constants. T29 exige un IC95 par fold et refuse de
   conclure quand il contient zero.
"""

import os
import sys

import numpy as np
import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS = os.path.join(_REPO_ROOT, "results")


def _study_file(name):
    for _d in ("pipeline", "h0_selection", "h1_solver", "h2b_prediction",
               "h3_representation", "h4_transfer", "closed_loop", "common"):
        _c = os.path.join(_REPO_ROOT, "study", _d, name)
        if os.path.exists(_c):
            return _c
    raise FileNotFoundError(name)


# ═══════════════════════════════════════════════════════════════════════
#  T28 — le label a seuil global
# ═══════════════════════════════════════════════════════════════════════

SCENARIOS = ("orszag_tang", "kelvin_helmholtz", "mhd_rotor", "harris_tearing")


def _paths(dim, suffix=""):
    return [os.path.join(RESULTS,
                         f"patches_{sc}_Re400_N256_dim{dim}{suffix}.npz")
            for sc in SCENARIOS]


@pytest.mark.parametrize("dim", [4, 16, 32, 64])
def test_per_scenario_labels_force_the_same_prevalence(dim):
    """Le defaut lui-meme, epingle : 25.00 % partout, par construction."""
    paths = _paths(dim)
    if not all(os.path.exists(p) for p in paths):
        pytest.skip(f"artefacts dim={dim} absents")
    prevalences, thresholds = [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        prevalences.append(float(np.asarray(d["is_hard"]).mean()))
        thresholds.append(float(d["l2_threshold"]))
    assert max(prevalences) - min(prevalences) < 1e-3, (
        "les labels par scenario sont censes imposer la meme prevalence ; "
        f"mesure {prevalences}"
    )
    assert max(thresholds) / min(thresholds) > 2.0, (
        "si les seuils devenaient comparables, le probleme decrit ici "
        f"n'existerait plus : {thresholds}"
    )


@pytest.mark.parametrize("dim", [4, 16, 32, 64])
def test_global_labels_let_the_prevalence_vary(dim):
    """La variante a seuil global doit rendre la prevalence mesuree."""
    paths = _paths(dim, "_globalthr")
    if not all(os.path.exists(p) for p in paths):
        pytest.skip(f"artefacts dim={dim} _globalthr absents")
    prevalences, thresholds = [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        prevalences.append(float(np.asarray(d["is_hard"]).mean()))
        thresholds.append(float(d["l2_threshold"]))
        assert str(d["label_variant"]) == "global_percentile"
    assert len(set(np.round(thresholds, 12))) == 1, (
        f"un seuil global doit etre unique : {thresholds}"
    )
    assert max(prevalences) - min(prevalences) > 0.10, (
        "avec un seuil commun la prevalence doit refleter la physique de "
        f"chaque scenario ; mesure {prevalences}"
    )


def test_the_variant_never_overwrites_the_original():
    """Le suffixe doit apparaitre dans le nom de sortie (defaut D9)."""
    src = open(_study_file("labels_global_threshold.py"),
               encoding="utf-8").read()
    assert 'SUFFIX = "_globalthr"' in src
    assert 'replace(".npz", f"{SUFFIX}.npz")' in src, (
        "l'artefact relabellise doit porter le suffixe, sinon il ecrase la "
        "variante par scenario et les deux deviennent indiscernables"
    )


def test_the_relabeller_refuses_a_degenerate_threshold():
    """Un seuil nul labelliserait 100 % des patches comme durs.

    C'est ce qui arrive a N=64 dim=64 (patches de 1x1 cellule) dans
    phase2 : seuil 0.000000, 100 % durs, et rien ne crie.
    """
    src = open(_study_file("labels_global_threshold.py"),
               encoding="utf-8").read()
    assert "seuil global degenere" in src
    assert "thr_global <= 0.0" in src


# ═══════════════════════════════════════════════════════════════════════
#  T29 — pas de verdict sans intervalle
# ═══════════════════════════════════════════════════════════════════════

def test_verdict_requires_confidence_intervals():
    """La fonction verdict n'a que trois sorties, et deux exigent un IC."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "t29mod", _study_file("h2b_loso_delta_ci.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    def row(lo, hi, constant=""):
        return dict(ci_low=lo, ci_high=hi, constant=constant)

    # IC tous strictement positifs -> conclusion permise
    assert m.verdict([row(0.01, 0.05), row(0.02, 0.09)]) == "aident"
    # IC tous strictement negatifs -> conclusion permise
    assert m.verdict([row(-0.09, -0.02), row(-0.05, -0.01)]) == "nuisent"
    # un seul IC qui contient zero -> indecidable
    assert m.verdict([row(0.01, 0.05), row(-0.02, 0.09)]) == "indecidable"
    # folds qui se contredisent -> indecidable
    assert m.verdict([row(0.01, 0.05), row(-0.09, -0.02)]) == "indecidable"


def test_constant_predictors_do_not_vote():
    """Un F1 obtenu en predisant toujours la meme classe n'est pas un score.

    A dim=32 et 64, `harris_tearing` donne F1 = 0.000 exactement (tout
    negatif) et `orszag_tang` 0.400 exactement (tout positif, prevalence
    0.25). Ces folds ne doivent pas peser dans une conclusion.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "t29mod2", _study_file("h2b_loso_delta_ci.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    assert m.constant_predictor(np.zeros(10, dtype=int))
    assert m.constant_predictor(np.ones(10, dtype=int))
    assert not m.constant_predictor(np.array([0, 1, 0, 1]))

    rows = [dict(ci_low=0.01, ci_high=0.05, constant="sten"),
            dict(ci_low=0.02, ci_high=0.09, constant="site")]
    assert m.verdict(rows) == "indecidable", (
        "tous les folds sont degeneres : aucune conclusion possible, meme "
        "avec des IC positifs"
    )


def test_bootstrap_blocks_are_snapshots_not_patches():
    """Les patches d'un meme instantane ne sont pas independants.

    Rechantillonner par patch retrecirait l'intervalle d'un facteur ~dim,
    ce qui rendrait n'importe quel ecart significatif.
    """
    src = open(_study_file("h2b_loso_delta_ci.py"), encoding="utf-8").read()
    assert "np.arange(len(Y)) // (dim * dim)" in src, (
        "le bloc de bootstrap doit etre l'instantane"
    )
    assert "bootstrap_by_trajectory" in src
