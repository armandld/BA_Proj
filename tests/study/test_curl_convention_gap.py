"""T31 — les statistiques qui portent le verdict sur la convention d'axes.

Le script `h1_curl_convention_gap.py` conclut a partir de deux comparaisons
sans seuil (Spearman contre la durete continue, F1 a budget appraie) et d'un
intervalle de confiance rechantillonne par scenario. Chacune de ces trois
pieces peut echouer en silence :

  - un budget appraie mal calcule comparerait deux nombres de patches
    differents, et l'ecart mesurerait le budget au lieu de la convention ;
  - un bootstrap rechantillonnant les instantanes plutot que les scenarios
    retrecirait l'intervalle jusqu'a rendre significatif n'importe quoi ;
  - un verdict qui ne regarde pas l'intervalle conclurait sur le signe d'une
    moyenne, ce que ce depot s'interdit.
"""

import importlib.util
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

_SCRIPT = os.path.join(_REPO_ROOT, "study", "h1_solver",
                       "h1_curl_convention_gap.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("t31mod", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── budget appraie ────────────────────────────────────────────────────

def test_matched_budget_selects_exactly_k_patches(mod):
    score = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
    for k in range(len(score) + 1):
        sel = mod._top_k(score, k)
        assert sel.sum() == k


def test_matched_budget_takes_the_highest_scores(mod):
    score = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
    np.testing.assert_array_equal(
        mod._top_k(score, 2), np.array([False, True, False, False, True]))


def test_matched_budget_is_perfect_when_the_score_is_the_label(mod):
    """Controle positif : sans lui, un _top_k casse passerait inapercu."""
    rng = np.random.default_rng(0)
    hardness = rng.random(64)
    truth = hardness > np.quantile(hardness, 0.75)
    assert mod._f1(mod._top_k(hardness, int(truth.sum())), truth) == 1.0


def test_matched_budget_is_near_zero_when_the_score_is_inverted(mod):
    """Controle negatif : un classement inverse ne doit rien retrouver."""
    rng = np.random.default_rng(0)
    hardness = rng.random(64)
    truth = hardness > np.quantile(hardness, 0.75)
    assert mod._f1(mod._top_k(-hardness, int(truth.sum())), truth) == 0.0


# ── Spearman ──────────────────────────────────────────────────────────

def test_spearman_is_one_on_a_monotone_transform(mod):
    x = np.linspace(0.1, 5.0, 40)
    assert mod._spearman(x, np.exp(x)) == pytest.approx(1.0)
    assert mod._spearman(x, -np.exp(x)) == pytest.approx(-1.0)


def test_spearman_refuses_a_constant_input(mod):
    """Une correlation avec une constante n'existe pas : NaN, pas 0.

    Renvoyer 0.0 la rendrait indiscernable d'une absence de correlation
    reellement mesuree.
    """
    x = np.linspace(0, 1, 20)
    assert np.isnan(mod._spearman(x, np.ones(20)))
    assert np.isnan(mod._spearman(np.ones(20), x))


# ── bootstrap par scenario ────────────────────────────────────────────

def _rows(deltas_by_scenario):
    rows = []
    for sc, ds in deltas_by_scenario.items():
        for i, d in enumerate(ds):
            rows.append(dict(scenario=sc, snap=i, a=0.0, b=float(d)))
    return rows


def test_bootstrap_recovers_a_clear_positive_effect(mod):
    rows = _rows({f"s{j}": [0.30, 0.32, 0.31, 0.29] for j in range(6)})
    obs, lo, hi = mod.bootstrap_delta_ci(rows, "a", "b", n_boot=2000, seed=0)
    assert obs == pytest.approx(0.305, abs=1e-3)
    assert lo > 0.0
    assert mod.verdict(lo, hi) == "la convention corrigee ameliore"


def test_bootstrap_recovers_a_clear_negative_effect(mod):
    rows = _rows({f"s{j}": [-0.30, -0.32, -0.31, -0.29] for j in range(6)})
    _obs, lo, hi = mod.bootstrap_delta_ci(rows, "a", "b", n_boot=2000, seed=0)
    assert hi < 0.0
    assert mod.verdict(lo, hi) == "la convention corrigee degrade"


def test_bootstrap_stays_undecided_when_scenarios_disagree(mod):
    """Deux scenarios qui se contredisent ne font pas une conclusion."""
    rows = _rows({"a1": [0.4] * 4, "a2": [0.35] * 4,
                  "b1": [-0.4] * 4, "b2": [-0.35] * 4})
    _obs, lo, hi = mod.bootstrap_delta_ci(rows, "a", "b", n_boot=4000, seed=0)
    assert lo < 0.0 < hi
    assert mod.verdict(lo, hi) == "indecidable"


def test_the_block_is_the_scenario_not_the_snapshot(mod):
    """Le defaut qui rendrait tout significatif, epingle par la mesure.

    Meme effet moyen, meme nombre de lignes : si le bloc etait l'instantane,
    doubler le nombre d'instantanes par scenario retrecirait l'intervalle.
    En rechantillonnant les scenarios, l'intervalle ne doit PAS se resserrer
    quand on ne fait qu'allonger les trajectoires.
    """
    few = _rows({f"s{j}": [0.2, -0.1] for j in range(4)})
    many = _rows({f"s{j}": [0.2, -0.1] * 8 for j in range(4)})
    _o1, lo1, hi1 = mod.bootstrap_delta_ci(few, "a", "b", n_boot=4000, seed=1)
    _o2, lo2, hi2 = mod.bootstrap_delta_ci(many, "a", "b", n_boot=4000, seed=1)
    np.testing.assert_allclose([lo1, hi1], [lo2, hi2], rtol=0, atol=1e-12)


def test_bootstrap_refuses_a_single_scenario(mod):
    """Avec un seul bloc il n'y a rien a rechantillonner."""
    rows = _rows({"solo": [0.2, 0.3, 0.25]})
    obs, lo, hi = mod.bootstrap_delta_ci(rows, "a", "b", n_boot=100, seed=0)
    assert np.isnan(lo) and np.isnan(hi)
    assert mod.verdict(lo, hi) == "indecidable"


def test_verdict_never_concludes_on_an_interval_touching_zero(mod):
    assert mod.verdict(0.0, 0.5) == "indecidable"
    assert mod.verdict(-0.5, 0.0) == "indecidable"
    assert mod.verdict(float("nan"), 0.5) == "indecidable"
    assert mod.verdict(1e-9, 0.5) == "la convention corrigee ameliore"


# ── l'artefact publie ─────────────────────────────────────────────────

@pytest.mark.parametrize("dim", [8, 16])
def test_the_published_artefact_carries_its_verdict(dim):
    """Un resultat sans son intervalle n'est pas publiable dans ce depot."""
    import json
    p = os.path.join(_REPO_ROOT, "results",
                     f"h1_curl_convention_gap_N128_dim{dim}_v2.npz")
    if not os.path.exists(p):
        pytest.skip(f"artefact dim={dim} absent")
    d = np.load(p, allow_pickle=True)
    v = json.loads(str(d["verdicts"]))
    assert set(v) == {"spearman_fixed", "f1_matched_fixed"}
    for key, block in v.items():
        assert {"observed", "ci_low", "ci_high", "verdict"} <= set(block)
        assert block["ci_low"] <= block["observed"] <= block["ci_high"], (
            f"{key}: l'observation doit tomber dans son propre intervalle")
    assert "provenance" in d and "cli_args" in d
