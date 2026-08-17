"""D-60 — la courbe de seuil du bras classique ne pouvait sortir d'aucune etude.

`analyze_hyperparams.plot_threshold_operating_curve` produit la seule figure
qui montre l'arbitrage precision/cout du bras classique : erreur physique et
taux de patchs contre le seuil de raffinement. C'est la figure de decision de
la campagne classique, dont le seuil EST le seul parametre optimise.

Elle n'a jamais pu s'executer, pour trois raisons independantes :

1. elle exigeait un parametre nomme `threshold`. Aucune base du depot, aucune
   ligne de `src/` ne porte ce nom : `make_classical_composite_objective`
   echantillonne `threshold_amr` ;

2. elle lisait `phys_score` / `patch_ratio`, que seul l'objectif mono-scenario
   de `pipeline.py` ecrit. La campagne deployee passe par
   `train_hyperparams._run_one_scenario`, qui ecrit `phys_<scenario>` et
   `patch_<scenario>` — jamais les cles globales ;

3. son unique appelant etait garde par `has_decomposed_data`, qui teste
   `phys_score` : faux pour toute etude composite.

Sur quelle entree ces tests echouent-ils ? Sur la version d'avant D-60 :
`test_curve_is_produced_from_the_frozen_classical_study` n'y ecrit aucun
fichier, et `test_main_reaches_the_curve` non plus.
"""

import os
import subprocess
import sys

import numpy as np
import pytest


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_ROOT = _repo_root()
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_CLASSICAL_DB = os.path.join(_ROOT, "results", "hyperparams",
                             "optuna_studies", "classical_v2_phase1.db")
_CLASSICAL_STUDY = "classical_v2_phase1"

#: Mesure du 13 aout 2026 sur la base gelee, 125 essais complets finis.
#: Ecrites ici pour qu'une derive se voie — pas des seuils calibres apres coup.
_N_COMPLETED = 125
_R_PATCH = -0.9690        # threshold_amr contre le taux de patchs moyen
_R_PHYS = +0.8369         # threshold_amr contre l'erreur physique moyenne
_SCENARIO_KEYS = ["kh", "tearing", "ot", "rotor"]


class _Trial:
    """Le contrat que les fonctions auditees consomment : params, user_attrs."""

    def __init__(self, params, user_attrs):
        self.params = params
        self.user_attrs = user_attrs


@pytest.fixture(scope="module")
def analyzer():
    return pytest.importorskip("analyze_hyperparams")


@pytest.fixture(scope="module")
def classical_completed(analyzer):
    if not os.path.exists(_CLASSICAL_DB):
        pytest.skip("base gelee absente : " + _CLASSICAL_DB)
    _study, completed = analyzer.load_study(_CLASSICAL_DB, _CLASSICAL_STUDY)
    assert len(completed) == _N_COMPLETED, (
        "la base gelee a change : %d essais complets finis au lieu de %d"
        % (len(completed), _N_COMPLETED))
    return completed


def test_the_optimised_parameter_is_not_named_threshold(classical_completed):
    """La cause de D-60, epinglee : le nom exige n'existe nulle part."""
    params = classical_completed[0].params
    assert "threshold" not in params
    assert "threshold_amr" in params


def test_decomposed_series_reads_the_composite_schema(analyzer,
                                                      classical_completed):
    """L'agregation est la MOYENNE : celle que la perte elle-meme applique.

    `_composite_loop` rend `total / len(scenario_list)`. Mesurer avec un
    autre operateur (somme, max) ne mesurerait pas la grandeur que la
    campagne a minimisee.
    """
    phys, patch, source = analyzer._decomposed_series(classical_completed)
    assert phys is not None, "schema composite non reconnu"
    assert len(phys) == len(patch) == _N_COMPLETED
    assert source.startswith("mean over ")
    for key in _SCENARIO_KEYS:
        assert key in source

    expected_phys = np.array(
        [[t.user_attrs["phys_%s" % k] for k in _SCENARIO_KEYS]
         for t in classical_completed]).mean(axis=1)
    np.testing.assert_allclose(phys, expected_phys, rtol=0, atol=0)

    thr = np.array([t.params["threshold_amr"] for t in classical_completed])
    assert np.corrcoef(thr, patch)[0, 1] == pytest.approx(_R_PATCH, abs=1e-3)
    assert np.corrcoef(thr, phys)[0, 1] == pytest.approx(_R_PHYS, abs=1e-3)


def test_curve_is_produced_from_the_frozen_classical_study(analyzer,
                                                           classical_completed,
                                                           tmp_path):
    """Le fait meme que D-60 rapporte : avant, aucun fichier n'etait ecrit."""
    analyzer.plot_threshold_operating_curve(classical_completed, str(tmp_path))
    assert os.path.exists(str(tmp_path / "12_threshold_operating_curve.png"))


def test_single_run_schema_still_works(analyzer, tmp_path):
    """Epingle l'ANCIEN contrat : `threshold` + `phys_score` / `patch_ratio`.

    C'est ce que `pipeline.py` ecrit quand il sert d'objectif. La correction
    de D-60 elargit la lecture, elle ne la deplace pas — si ce test tombe,
    c'est que l'ancien schema a ete perdu en chemin.
    """
    completed = [
        _Trial({"threshold": 0.1 * i},
               {"phys_score": 0.01 * i, "patch_ratio": 1.0 - 0.1 * i})
        for i in range(1, 6)
    ]
    phys, patch, source = analyzer._decomposed_series(completed)
    assert source == "single run"
    np.testing.assert_allclose(phys, [0.01, 0.02, 0.03, 0.04, 0.05])
    np.testing.assert_allclose(patch, [0.9, 0.8, 0.7, 0.6, 0.5])

    analyzer.plot_threshold_operating_curve(completed, str(tmp_path))
    assert os.path.exists(str(tmp_path / "12_threshold_operating_curve.png"))


def test_missing_decomposition_screams(analyzer, tmp_path, capsys):
    """Un balayage vide doit crier : sans les attributs, on le DIT.

    Une analyse amputee de sa figure de decision ressemble sinon a une
    analyse complete.
    """
    completed = [_Trial({"threshold_amr": 0.1 * i}, {"loss_kh": 0.2})
                 for i in range(1, 6)]
    analyzer.plot_threshold_operating_curve(completed, str(tmp_path))
    assert not os.path.exists(str(tmp_path / "12_threshold_operating_curve.png"))
    assert "courbe de seuil indisponible" in capsys.readouterr().err


def test_no_threshold_no_curve_and_no_noise(analyzer, tmp_path, capsys):
    """Une etude sans seuil n'est pas un defaut : rien a tracer, rien a dire."""
    completed = [_Trial({"beta": 1.0 * i}, {"phys_score": 0.1,
                                            "patch_ratio": 0.5})
                 for i in range(1, 6)]
    analyzer.plot_threshold_operating_curve(completed, str(tmp_path))
    assert os.listdir(str(tmp_path)) == []
    assert capsys.readouterr().err == ""


def test_main_reaches_the_curve(tmp_path):
    """Bout en bout : la garde `has_decomposed_data` ne doit plus l'enfermer.

    L'etude classique n'ecrit aucun `phys_score`, donc la section 4 est
    sautee — et c'est precisement l'etude dont le seuil est le parametre
    optimise. Avant D-60, la figure manquait sans qu'aucun message ne le
    signale.
    """
    if not os.path.exists(_CLASSICAL_DB):
        pytest.skip("base gelee absente : " + _CLASSICAL_DB)
    out = subprocess.run(
        [sys.executable, os.path.join(_SRC, "analyze_hyperparams.py"),
         "--db-path", _CLASSICAL_DB,
         "--study-name", _CLASSICAL_STUDY,
         "--output-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=900,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "No decomposed score data" in out.stdout, (
        "la section 4 devait rester sautee : si elle ne l'est plus, ce test "
        "ne prouve plus que la courbe sort HORS de sa garde")
    assert os.path.exists(str(tmp_path / "12_threshold_operating_curve.png"))
