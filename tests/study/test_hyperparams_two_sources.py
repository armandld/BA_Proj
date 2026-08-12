"""Le depot contient DEUX jeux d'hyperparametres. Lequel gouverne quoi ?

Trouve en revoyant `src/hyperparams_loader.py`. Les deux sources divergent
d'un facteur 100 sur `w_z_frac`, et rien dans le depot ne le disait :

    grandeur         study/pipeline/config.py     best_hyperparams.json
    -----------      ------------------------     ---------------------
    threshold_amr    0.1496                       0.304446
    beta             9.94                         0.549537
    beta_curl        4.27                         0.819924
    beta_xpoint      2.39                         0.425647
    w_z_frac         10.40                        0.101338
    gamma_mag        0.5                          2.361084
    kappa            10.0                         14.332145

Qui lit quoi :

  - tout `study/` lit les constantes de `config.py` ;
  - `src/pipeline.py` — le pipeline DEPLOYE — lit le JSON via
    `load_hyperparams()`, puis fusionne le dict de l'appelant par-dessus.

La resolution est rassurante mais devait etre verifiee, pas supposee :

  1. les constantes de `config.py` correspondent a l'essai 5 du bras
     quantique, RANG 1 sur 178 — l'etude utilise bien le meilleur essai,
     et les valeurs gelees (`threshold_amr` = meilleur classique,
     gamma_hydro=2.0, gamma_mag=0.5, kappa=10.0) sont exactement celles que
     `TrainHyperParam_v2` fixe en dur pendant phase1 ;
  2. la campagne de boucle fermee passe a `pipeline()` un dict COMPLET
     (FROZEN_DEFAULTS + les 5 parametres regles + le seuil), qui couvre
     toutes les cles lues : le JSON n'y fuit donc pas.

Ce qui reste dangereux, et que ces tests verrouillent : tout appelant qui
passerait un dict PARTIEL heriterait silencieusement des valeurs du JSON,
qui ne sont pas celles de l'etude.
"""

import collections
import glob
import json
import os
import sqlite3
import sys

import pytest



def _repo_root():
    """Racine du depot : on remonte jusqu'au dossier qui contient `src/`.

    Un calcul par `dirname` repete depend de la profondeur du fichier et
    casse au premier deplacement — souvent en silence, en pointant vers un
    chemin qui n'existe pas.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_REPO_ROOT = _repo_root()
for _p in [os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "study", "pipeline")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HP = os.path.join(_REPO_ROOT, "results", "hyperparams")
_JSON = os.path.join(_HP, "best_hyperparams.json")
_QDB = os.path.join(_HP, "optuna_studies", "q_has_v2_phase1.db")
_CDB = os.path.join(_HP, "optuna_studies", "classical_v2_phase1.db")


def _json():
    return json.load(open(_JSON, encoding="utf-8"))


def _trials(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        par = collections.defaultdict(dict)
        for t, n, v in con.execute(
                "select trial_id, param_name, param_value from trial_params"):
            par[t][n] = v
        val = {t: v for t, v in con.execute(
            "select trial_id, value from trial_values")}
        return par, val
    finally:
        con.close()


# ── 1. Les deux sources divergent ────────────────────────────────────

def test_the_two_sources_really_disagree():
    """Le fait lui-meme. S'ils convergeaient un jour, ce test le dirait."""
    import config

    d = _json()["default"]["quantum"]["params"]
    pairs = [("threshold_amr", config.TRAINED_THRESHOLD),
             ("beta", config.TRAINED_BETA),
             ("beta_curl", config.TRAINED_BETA_CURL),
             ("beta_xpoint", config.TRAINED_BETA_XPOINT),
             ("w_z_frac", config.TRAINED_W_Z_FRAC),
             ("gamma_mag", config.TRAINED_GAMMA_MAG),
             ("kappa", config.TRAINED_KAPPA)]
    differing = [k for k, cfg in pairs
                 if k in d and abs(d[k] - cfg) > 1e-3 * max(abs(cfg), 1e-9)]
    assert len(differing) >= 6, (
        f"seulement {differing} different ; si les deux sources ont ete "
        "reconciliees, mettre a jour ce fichier et RESULTS.md")
    assert abs(d["w_z_frac"] - config.TRAINED_W_Z_FRAC) > 10.0, (
        "l'ecart de 100x sur w_z_frac a disparu")


# ── 2. L'etude utilise le MEILLEUR essai ─────────────────────────────

def test_the_study_constants_are_the_best_quantum_trial():
    """config.py doit correspondre a l'essai de rang 1, pas a un essai
    quelconque.

    C'est la question qui compte : l'etude evalue-t-elle V1 a son meilleur
    reglage, ou a un reglage arbitraire ? Un reglage arbitraire rendrait
    toute conclusion de sous-performance partiellement imputable au choix.
    """
    import config

    par, val = _trials(_QDB)
    target = dict(beta=config.TRAINED_BETA,
                  beta_curl=config.TRAINED_BETA_CURL,
                  beta_xpoint=config.TRAINED_BETA_XPOINT,
                  sigma=config.TRAINED_SIGMA,
                  w_z_frac=config.TRAINED_W_Z_FRAC)
    match = [t for t, p in par.items()
             if all(k in p and abs(p[k] - v) <= max(0.01, 0.005 * abs(v))
                    for k, v in target.items())]
    assert len(match) == 1, (
        f"{len(match)} essais correspondent aux constantes de config.py ; "
        "elles devraient en designer exactement un")

    ranked = sorted(((v, t) for t, v in val.items() if v is not None))
    rank = next(i for i, (_v, t) in enumerate(ranked) if t == match[0]) + 1
    assert rank == 1, (
        f"config.py correspond a l'essai de rang {rank}/{len(ranked)}, pas "
        "au meilleur : l'etude n'evalue pas V1 a son meilleur reglage")


def test_the_frozen_constants_match_the_training_script():
    """threshold, gammas et kappa etaient GELES pendant phase1.

    Ils ne figurent donc pas dans la base : leur seule source est le code
    d'entrainement. config.py doit reproduire ces valeurs, sinon l'etude
    tourne sur un reglage que la campagne n'a jamais evalue.
    """
    import config

    src = open(os.path.join(_REPO_ROOT, "src", "TrainHyperParam_v2.py"),
               encoding="utf-8").read()
    assert "0.14959824837662078" in src
    assert abs(config.TRAINED_THRESHOLD - 0.14959824837662078) < 1e-4
    for name, value in (("gamma_hydro", 2.0), ("gamma_mag", 0.5),
                        ("kappa", 10.0)):
        assert f'HyperParams["{name}"] = {value}' in src, (
            f"{name}={value} n'est plus gele dans TrainHyperParam_v2")
    assert config.TRAINED_GAMMA_HYDRO == 2.0
    assert config.TRAINED_GAMMA_MAG == 0.5
    assert config.TRAINED_KAPPA == 10.0


def test_the_classical_threshold_is_the_best_classical_trial():
    par, val = _trials(_CDB)
    ranked = sorted(((v, t) for t, v in val.items() if v is not None))
    best_t = ranked[0][1]
    assert abs(par[best_t]["threshold_amr"] - 0.14959824837662078) < 1e-9, (
        "le seuil gele n'est plus le meilleur essai classique")


# ── 3. Le JSON ne doit pas fuir dans la boucle fermee ────────────────

_PIPELINE = os.path.join(_REPO_ROOT, "src", "pipeline.py")
_CAMPAIGN = os.path.join(_REPO_ROOT, "study", "closed_loop",
                         "closed_loop_campaign.py")


def _live_pipeline_keys():
    """Cles hp lues par pipeline.py, hors blocs neutralises en chaine."""
    import re
    src = open(_PIPELINE, encoding="utf-8").read()
    src = re.sub(r'""".*?"""', "", src, flags=re.S)      # retire les blocs morts
    return set(re.findall(r"hp\.get\(\s*['\"]([a-z_]+)['\"]", src))


def test_the_closed_loop_covers_every_key_the_pipeline_reads():
    """Un dict PARTIEL ferait heriter les valeurs perimees du JSON.

    C'est le piege : `hp = {**_defaults, **hyperparams}`. Toute cle absente
    du dict de l'appelant vient du JSON, dont les valeurs ne sont PAS celles
    de l'etude.
    """
    camp = open(_CAMPAIGN, encoding="utf-8").read()
    assert "FROZEN_DEFAULTS = dict(gamma_hydro=2.0, gamma_mag=0.5, kappa=10.0)" in camp
    assert 'best.setdefault("threshold_amr", 0.14959824837662078)' in camp

    provided = {"gamma_hydro", "gamma_mag", "kappa", "threshold_amr"}
    #  les 5 parametres explores par l'objectif V1
    provided |= {"beta", "beta_curl", "beta_xpoint", "sigma", "w_z_frac"}
    missing = _live_pipeline_keys() - provided
    assert not missing, (
        f"la campagne ne fournit pas {sorted(missing)} ; ces cles seraient "
        "prises dans best_hyperparams.json, qui n'est pas la source de "
        "l'etude")


def test_the_pipeline_still_merges_the_json_underneath():
    """Le piege existe toujours : on le documente au lieu de l'oublier."""
    src = open(_PIPELINE, encoding="utf-8").read()
    assert "_defaults = load_hyperparams()" in src
    assert "hp = {**_defaults, **(hyperparams or {})}" in src


# ── 4. Le chargeur ne substitue plus en silence ──────────────────────

def test_the_loader_refuses_to_substitute_the_other_arm(tmp_path):
    """Demander 'quantum' et recevoir 'classical' viderait de son sens la
    comparaison des deux bras."""
    from hyperparams_loader import load_hyperparams

    doc = {"default": {"classical": {"params": {"threshold_amr": 0.3}}}}
    p = tmp_path / "hp.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(KeyError, match="quantum"):
        load_hyperparams(path=str(p), method="quantum")
    #  le bras present, lui, se charge normalement
    assert load_hyperparams(path=str(p), method="classical") == {
        "threshold_amr": 0.3}


def test_the_loader_refuses_an_ambiguous_cost_weight(tmp_path):
    """Plusieurs lambdas : choisir le premier alphabetiquement est arbitraire."""
    from hyperparams_loader import load_hyperparams

    doc = {"default": {"quantum": {"params": {"beta": 1.0}}},
           "training_phases": {"quantum": {"phase1": {
               "lambda_0.4000": [{"params": {"beta": 1.0}}],
               "lambda_0.8000": [{"params": {"beta": 2.0}}]}}}}
    p = tmp_path / "hp.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(KeyError, match="cost weights|lambda_cost"):
        load_hyperparams(path=str(p), method="quantum", phase="phase1")
    #  desambiguise : accepte
    got = load_hyperparams(path=str(p), method="quantum", phase="phase1",
                           lambda_cost="lambda_0.8000")
    assert got == {"beta": 2.0}


def test_a_single_cost_weight_stays_implicit(tmp_path):
    """Un seul lambda : le choix est force, donc licite."""
    from hyperparams_loader import load_hyperparams

    doc = {"default": {"quantum": {"params": {"beta": 1.0}}},
           "training_phases": {"quantum": {"phase1": {
               "lambda_0.4000": [{"params": {"beta": 7.0}}]}}}}
    p = tmp_path / "hp.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert load_hyperparams(path=str(p), method="quantum",
                            phase="phase1") == {"beta": 7.0}


def test_the_real_file_still_loads_for_both_arms():
    """Non-regression : le fichier gele doit continuer a se charger."""
    from hyperparams_loader import load_hyperparams

    for arm in ("quantum", "classical"):
        p = load_hyperparams(method=arm)
        assert isinstance(p, dict) and p


# ── 5. Le bloc « par scenario » n'est pas ce que son nom dit ─────────

def test_the_per_scenario_quantum_block_is_one_set_repeated():
    """Les 4 scenarios renseignes portent des parametres IDENTIQUES.

    « Hyperparametres par scenario » designe donc, pour le bras quantique,
    quatre copies du meme jeu — celui du bloc `default`. Le manuscrit ne
    peut pas presenter cela comme un reglage par scenario.
    """
    ps = _json()["per_scenario"]
    got = {sc: v["quantum"]["params"] for sc, v in ps.items()
           if v.get("quantum")}
    assert got, "aucun scenario ne porte de parametres quantiques"
    uniq = {json.dumps(p, sort_keys=True) for p in got.values()}
    assert len(uniq) == 1, (
        f"{len(uniq)} jeux distincts pour {len(got)} scenarios : le bloc "
        "n'est plus une repetition, mettre a jour PROVENANCE.md")
    default = json.dumps(_json()["default"]["quantum"]["params"],
                         sort_keys=True)
    assert uniq == {default}, "les jeux par scenario different du default"


def test_two_study_scenarios_have_no_per_scenario_entry():
    """orszag_tang et mhd_rotor sont absents des deux bras.

    Ce sont deux des quatre classes de l'etude. Le chargeur leve pour
    elles — ce qui est le bon comportement, et qu'on verifie ici pour que
    personne ne « repare » ce bloc en y recopiant le default.
    """
    from hyperparams_loader import load_hyperparams

    ps = _json()["per_scenario"]
    for sc in ("orszag_tang", "mhd_rotor"):
        assert sc in ps
        for arm in ("quantum", "classical"):
            assert not ps[sc].get(arm), (
                f"{sc}/{arm} a ete renseigne ; verifier d'ou viennent ces "
                "valeurs avant de les utiliser")
            with pytest.raises(KeyError):
                load_hyperparams(scenario=sc, method=arm)
