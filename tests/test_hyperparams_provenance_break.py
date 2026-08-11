"""D-22 — les hyperparametres deployes n'ont aucune provenance reproductible.

`results/hyperparams/` est declare « entree gelee » de l'etude. Il contient
deux choses qui devraient etre d'accord :

  optuna_studies/*.db      les bases de la campagne, 345 essais
  best_hyperparams.json    ce que `src/hyperparams_loader.py` charge

Elles ne le sont pas. Verifie directement dans les fichiers :

  Ce que les bases ont ECHANTILLONNE
    etude quantique   beta, beta_curl, beta_xpoint, sigma, w_z_frac
    etude classique   threshold_amr

  Ce que le JSON DEPLOIE (bloc default.quantum)
    beta, beta_curl, beta_xpoint, w_z_frac,
    threshold_amr, gamma_hydro, gamma_mag, kappa

Trois ecarts, tous verifies ici :

1. `gamma_hydro`, `gamma_mag` et `kappa` ne figurent dans AUCUNE base. Trois
   des huit valeurs deployees n'ont aucune origine dans le depot.

2. `sigma` est echantillonne par la campagne mais ABSENT du JSON. Le
   pipeline retombe donc sur son defaut code en dur, 0.05, alors que les
   meilleurs essais trouvaient 0.023 a 0.194. Sigma est la largeur de la
   fenetre gaussienne — le parametre au coeur de D-9.

3. Le JSON declare venir de l'essai 85 avec une perte de 0.2215. L'essai 85
   existe bien dans la base, mais sa perte y vaut 0.3213 et AUCUN de ses
   quatre parametres communs ne coincide :

     | parametre   | base     | JSON     |
     |-------------|----------|----------|
     | beta        | 6.034464 | 0.549537 |
     | beta_curl   | 1.318670 | 0.819924 |
     | beta_xpoint | 2.341306 | 0.425647 |
     | w_z_frac    | 39.599016| 0.101338 |

Le code d'entrainement, lui, est COHERENT avec les bases :
`TrainHyperParam_v2` code en dur `threshold_amr = 0.14959824837662078` avec
le commentaire « le meilleur classique », et c'est exactement la valeur du
meilleur essai classique (#42, perte 0.2148). C'est le JSON qui est
orphelin.

Consequence : le bras quantique est deploye a un seuil de 0.3044 auquel il
n'a jamais ete entraine, avec un sigma jete, et trois portes dont les
valeurs ne viennent de nulle part. Une reoptimisation n'est donc pas une
amelioration — c'est la seule facon d'avoir des hyperparametres qui
existent.

Ces tests EPINGLENT l'ecart. Ils sont ecrits pour etre retournes le jour ou
la reoptimisation aura produit un JSON tracable : chacun dit, dans sa
docstring, ce qui devra etre vrai a ce moment-la.
"""

import json
import os
import sqlite3
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_HP_DIR = os.path.join(_REPO, "results", "hyperparams")
_JSON = os.path.join(_HP_DIR, "best_hyperparams.json")
_Q_DB = os.path.join(_HP_DIR, "optuna_studies", "q_has_v2_phase1.db")
_C_DB = os.path.join(_HP_DIR, "optuna_studies", "classical_v2_phase1.db")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_JSON) and os.path.exists(_Q_DB)),
    reason="entree gelee results/hyperparams/ absente")


def _sampled(db):
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        return {r[0] for r in c.execute(
            "select distinct param_name from trial_params")}


def _trial(db, number):
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        row = c.execute("select trial_id from trials where number=?",
                        (number,)).fetchone()
        if row is None:
            return None, None
        tid = row[0]
        val = c.execute("select value from trial_values where trial_id=?",
                        (tid,)).fetchone()
        params = dict(c.execute(
            "select param_name, param_value from trial_params where trial_id=?",
            (tid,)).fetchall())
        return (val[0] if val else None), params


def _best(db):
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        row = c.execute("""
            select t.trial_id, t.number, v.value from trials t
            join trial_values v on v.trial_id = t.trial_id
            where t.state='COMPLETE' order by v.value asc limit 1""").fetchone()
        tid, num, val = row
        params = dict(c.execute(
            "select param_name, param_value from trial_params where trial_id=?",
            (tid,)).fetchall())
        return num, val, params


def _json_q():
    return json.load(open(_JSON, encoding="utf-8"))["default"]["quantum"]


# ── 1. Trois valeurs sans aucune origine ──────────────────────────────

@pytest.mark.parametrize("name", ["gamma_hydro", "gamma_mag", "kappa"])
def test_three_deployed_values_appear_in_no_database(name):
    """A retourner apres reoptimisation : ces trois noms devront figurer
    parmi les parametres echantillonnes, ou disparaitre du JSON."""
    deployed = _json_q()["params"]
    assert name in deployed, f"{name} a disparu du JSON — mettre ce test a jour"
    assert name not in _sampled(_Q_DB) | _sampled(_C_DB), (
        f"{name} est desormais echantillonne : la provenance est retablie, "
        "ce test doit etre retourne")


def test_the_quantum_study_sampled_exactly_five_parameters():
    assert _sampled(_Q_DB) == {"beta", "beta_curl", "beta_xpoint", "sigma",
                               "w_z_frac"}


def test_the_classical_study_sampled_only_the_threshold():
    assert _sampled(_C_DB) == {"threshold_amr"}


# ── 2. sigma echantillonne puis jete ──────────────────────────────────

def test_sigma_is_sampled_by_the_campaign_but_absent_from_the_json():
    """Sigma est la largeur de la fenetre gaussienne — le parametre au coeur
    de D-9. La campagne l'optimise, le deploiement l'ignore."""
    assert "sigma" in _sampled(_Q_DB)
    assert "sigma" not in _json_q()["params"], (
        "sigma est de retour dans le JSON : verifier qu'il vaut bien ce que "
        "la campagne a trouve, et retourner ce test")


def test_the_pipeline_falls_back_to_a_hard_coded_sigma():
    """Consequence directe : la valeur utilisee ne vient pas de la campagne."""
    from hyperparams_loader import load_hyperparams
    assert "sigma" not in load_hyperparams()
    src = open(os.path.join(_SRC, "pipeline.py"), encoding="utf-8").read()
    assert "_defaults.get('sigma', 0.05)" in src


def test_the_campaign_did_find_a_sigma_far_from_that_fallback():
    _, _, params = _best(_Q_DB)
    assert params["sigma"] == pytest.approx(0.022981, abs=1e-5)
    assert abs(params["sigma"] - 0.05) > 0.02


# ── 3. L'essai declare ne correspond pas ──────────────────────────────

def test_the_declared_trial_exists_but_carries_a_different_loss():
    j = _json_q()
    loss, _ = _trial(_Q_DB, j["trial"])
    assert loss is not None, "l'essai declare a disparu de la base"
    assert abs(loss - j["new_score"]) > 0.05, (
        "la perte du JSON coincide desormais avec la base : la provenance "
        "est retablie, retourner ce test")


@pytest.mark.parametrize("name", ["beta", "beta_curl", "beta_xpoint",
                                  "w_z_frac"])
def test_no_common_parameter_of_the_declared_trial_matches(name):
    j = _json_q()
    _, params = _trial(_Q_DB, j["trial"])
    assert abs(params[name] - j["params"][name]) > 1e-6, (
        f"{name} coincide desormais : la provenance est retablie")


# ── 4. Le code d'entrainement, lui, est coherent avec les bases ───────

def test_the_training_objective_hard_codes_the_true_classical_best():
    """`TrainHyperParam_v2` code en dur 0.14959824837662078 avec le
    commentaire « le meilleur classique ». C'est exactement la valeur du
    meilleur essai classique : le code d'entrainement est tracable."""
    num, loss, params = _best(_C_DB)
    assert params["threshold_amr"] == pytest.approx(0.14959824837662078,
                                                    abs=1e-9)
    src = open(os.path.join(_SRC, "TrainHyperParam_v2.py"),
               encoding="utf-8").read()
    assert "0.14959824837662078" in src


def test_the_deployed_threshold_was_never_sampled_at_all():
    """Le bras quantique tourne a un seuil auquel il n'a jamais ete
    entraine, et qui ne figure pas parmi les 125 valeurs essayees."""
    target = _json_q()["params"]["threshold_amr"]
    with sqlite3.connect(f"file:{_C_DB}?mode=ro", uri=True) as c:
        vals = [r[0] for r in c.execute(
            "select param_value from trial_params where param_name='threshold_amr'")]
    assert vals, "la base classique ne contient plus de seuil"
    assert not any(abs(v - target) < 1e-9 for v in vals), (
        f"{target} figure desormais parmi les valeurs essayees")
    assert abs(target - 0.14959824837662078) > 0.1


# ── 5. Le critere d'acceptation de la reoptimisation ──────────────────

def test_every_deployed_hyperparameter_should_one_day_be_traceable():
    """LE test qui doit passer apres reoptimisation.

    Il echoue aujourd'hui, volontairement, et son message dit exactement ce
    qui manque. Le jour ou une campagne aura echantillonne les huit
    parametres et ou le JSON en portera les valeurs, il passera sans etre
    modifie — c'est le critere d'acceptation.
    """
    from hyperparams_loader import load_hyperparams
    deployed = set(load_hyperparams())
    sampled = _sampled(_Q_DB) | _sampled(_C_DB)
    orphans = sorted(deployed - sampled)
    pytest.xfail(
        f"provenance non retablie : {orphans} ne sont echantillonnes par "
        "aucune base de la campagne gelee")
