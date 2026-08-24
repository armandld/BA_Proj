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
     `train_hyperparams` fixe en dur pendant phase1 ;
  2. la campagne de boucle fermee passe a `pipeline()` un dict COMPLET
     (FROZEN_DEFAULTS + les 5 parametres regles + le seuil), qui couvre
     toutes les cles lues : le JSON n'y fuit donc pas.

Ce qui reste dangereux, et que ces tests verrouillent : tout appelant qui
passerait un dict PARTIEL heriterait silencieusement des valeurs du JSON,
qui ne sont pas celles de l'etude.
"""

import ast
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


def test_closed_loop_searches_the_current_hamiltonian_parameters():
    import config
    import train_hyperparams as T

    assert T.FIXED_PARAMS["threshold_amr"] == T.CLASSICAL_BEST_THRESHOLD
    assert abs(config.TRAINED_THRESHOLD - T.CLASSICAL_BEST_THRESHOLD) < 1e-4

    for name in ("gamma_hydro", "gamma_mag", "kappa"):
        assert name in T.SEARCH_SPACE, (
            f"{name} n'est plus explore : si c'est voulu, il doit revenir "
            f"dans FIXED_PARAMS et ce test doit etre remesure")
        assert name not in T.FIXED_PARAMS

    # The built-in study configuration remains a reproducibility reference;
    # the closed-loop campaign must not freeze these values.
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


#  Nombre de cles `hp` lues par le code VIVANT de `pipeline.py`, MESURE le
#  17 aout 2026 : beta, beta_curl, beta_xpoint, gamma_hydro, gamma_mag,
#  kappa, relative_percentile, sigma, threshold_amr, w_z_frac.
#  Ecrit ici pour qu'une derive se voie : un balayage qui rendrait moins que
#  ca ne prouverait plus rien, et c'est le piege que D-145 a mesure.
_CLES_HP_MESUREES = 10


def _live_pipeline_keys():
    """Cles `hp` lues par le code VIVANT de `pipeline.py` — par l'AST.

    D-145. La version precedente cherchait `hp.get('…')` par un REGEX, apres
    avoir retire les blocs `\"\"\"…\"\"\"` du texte. Elle ne voyait donc pas
    `hp['cle']`, l'autre facon d'ecrire la meme lecture. Mesure : un
    `hp['nouvelle_ponderation']` ajoute a `pipeline.py` — une cle qu'aucune
    source de la campagne ne fournit, donc heritee du JSON, la fuite exacte
    que le test ci-dessous interdit — laissait le fichier a **13 passed**.

    L'AST voit les deux formes, et il n'a pas besoin qu'on lui retire le bloc
    mort : un litteral de chaine reste une CONSTANTE, son contenu n'est pas
    du code. C'est deja la technique de `_cles_hp_get_vivantes`
    (`tests/pipeline/test_relative_percentile_is_trainable.py`), qui balaie
    le meme fichier dans l'autre sens.

    Une cle DYNAMIQUE (`hp.get(nom)`, `hp[nom]`) ne peut pas etre enumeree
    statiquement : on ne la passe pas sous silence, on leve — sinon le
    balayage redeviendrait aveugle sans que personne ne le voie.
    """
    arbre = ast.parse(open(_PIPELINE, encoding="utf-8").read())
    cles, dynamiques = set(), []
    for n in ast.walk(arbre):
        #  hp['cle']
        if (isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name) and n.value.id == "hp"):
            if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                cles.add(n.slice.value)
            else:
                dynamiques.append(f"hp[…] ligne {n.lineno}")
        #  hp.get('cle', …)
        if (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "hp"):
            if n.args and isinstance(n.args[0], ast.Constant) \
                    and isinstance(n.args[0].value, str):
                cles.add(n.args[0].value)
            else:
                dynamiques.append(f"hp.get(…) ligne {n.lineno}")

    assert not dynamiques, (
        f"{dynamiques} : `pipeline.py` lit `hp` avec une cle calculee. "
        f"Aucun balayage statique ne peut la couvrir, donc la garantie "
        f"« la campagne fournit toutes les cles lues » n'est plus verifiable "
        f"telle quelle — la porter autrement plutot que la laisser passer.")
    assert len(cles) >= _CLES_HP_MESUREES, (
        f"{len(cles)} cles trouvees pour {_CLES_HP_MESUREES} mesurees le "
        f"17 aout : le balayage a perdu des lectures, il ne prouve plus "
        f"l'absence de fuite. Un balayage vide doit crier (D-145).")
    return cles


def test_the_closed_loop_covers_every_key_the_pipeline_reads():
    """Un dict PARTIEL ferait heriter les valeurs perimees du JSON.

    C'est le piege : `hp = {**_defaults, **hyperparams}`. Toute cle absente
    du dict de l'appelant vient du JSON, dont les valeurs ne sont PAS celles
    de l'etude.
    """
    import train_hyperparams as T

    provided = set(T.search_space()) | {"threshold_amr"}
    missing = _live_pipeline_keys() - provided
    assert not missing, (
        f"la campagne ne fournit pas {sorted(missing)} ; ces cles seraient "
        "prises dans best_hyperparams.json, qui n'est pas la source de "
        "l'etude")


def test_closed_loop_requests_a_fold_specific_threshold_search():
    from closed_loop_campaign import train_params_excluding

    class _FakeT:
        LAMBDA_COST_SOFT = 0.0
        CLASSICAL_THRESHOLD_RANGE = (0.8, 0.9)

        def __init__(self):
            self.seen_tune_threshold = None

        def make_composite_objective(self, dns_traces, train_list,
                                     lambda_cost=None,
                                     tune_threshold=False):
            self.seen_tune_threshold = tune_threshold

            def obj(trial):
                x = trial.suggest_float("beta", 0.0, 1.0)
                threshold = trial.suggest_float("threshold_amr", 0.8, 0.9)
                trial.set_user_attr(
                    "hyperparams_resolved",
                    {"beta": x, "threshold_amr": threshold})
                return x

            return obj

    T_fake = _FakeT()
    best, _val, _n = train_params_excluding(
        T_fake, dns_traces=None, train_list=None, n_trials=1, seed=0)
    assert T_fake.seen_tune_threshold is True
    assert 0.8 <= best["threshold_amr"] <= 0.9


def test_the_pipeline_still_merges_the_json_underneath():
    """Explicit trial parameters override the selected deployment artifact."""
    src = open(_PIPELINE, encoding="utf-8").read()
    assert "_defaults = load_hyperparams(path=hyperparams_path)" in src
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


def _campaign_candidate(status="complete"):
    params = {
        "beta": 1.0, "w_z_frac": 2.0, "sigma": 0.1,
        "beta_curl": 0.2, "beta_xpoint": 0.3,
        "gamma_hydro": 1.1, "gamma_mag": 1.2, "kappa": 3.0,
        "relative_percentile": 90.0, "threshold_amr": 0.15,
    }
    return {
        "artifact": "phase_candidate",
        "status": status,
        "campaign_contract_sha256": "a" * 64,
        "campaign_contract": "{}",
        "result": {"best_params": params},
    }


def test_completed_campaign_candidate_is_a_loadable_parameter_source(tmp_path):
    from hyperparams_loader import load_hyperparams

    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(_campaign_candidate()), encoding="utf-8")
    params = load_hyperparams(path=str(path))
    assert params["sigma"] == 0.1
    assert len(params) == 10


def test_partial_or_uncontracted_candidate_is_refused(tmp_path):
    from hyperparams_loader import load_hyperparams

    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(_campaign_candidate(status="partial")), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not complete"):
        load_hyperparams(path=str(path))

    uncontracted = _campaign_candidate()
    uncontracted.pop("campaign_contract_sha256")
    path.write_text(json.dumps(uncontracted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="campaign contract"):
        load_hyperparams(path=str(path))


def test_environment_selects_the_campaign_artifact(tmp_path, monkeypatch):
    from hyperparams_loader import load_hyperparams, resolve_hyperparams_path

    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(_campaign_candidate()), encoding="utf-8")
    monkeypatch.setenv("QHAS_HYPERPARAMS_PATH", str(path))
    assert resolve_hyperparams_path() == str(path)
    assert load_hyperparams()["w_z_frac"] == 2.0


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
